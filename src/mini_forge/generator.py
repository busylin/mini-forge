from __future__ import annotations

import contextlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from .metrics import diversity_metrics, ranking_metrics
from .utils import (
    atomic_torch_save,
    cosine_schedule,
    ensure_dir,
    read_json,
    read_jsonl,
    resolve_device,
    seed_everything,
    write_json,
    write_jsonl,
)


SPECIAL_TOKENS = [
    "<PAD>", "<EOS>",
    "<USER>", "</USER>",
    "<ITEM_LIST>", "</ITEM_LIST>",
]


def _user_tokens(example: dict[str, Any]) -> list[str]:
    required = ("user_id", "gender", "age", "occupation")
    missing = [field for field in required if field not in example]
    if missing:
        raise ValueError(
            f"Example is missing user profile fields: {', '.join(missing)}. "
            "Run the data preparation command again."
        )
    return [
        f"<USER_ID_{example['user_id']}>",
        f"<USER_GENDER_{example['gender']}>",
        f"<USER_AGE_{example['age']}>",
        f"<USER_OCCUPATION_{example['occupation']}>",
    ]


def build_vocabulary(
    mapping: dict[str, Any],
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    tokens = set(SPECIAL_TOKENS)
    for value in mapping.values():
        tokens.update(value["tokens"])
    for example in examples or []:
        tokens.update(_user_tokens(example))
    ordered = SPECIAL_TOKENS + sorted(tokens - set(SPECIAL_TOKENS))
    return {token: index for index, token in enumerate(ordered)}


class SIDDataset(Dataset):
    def __init__(self, path: str | Path, mapping: dict[str, Any], vocabulary: dict[str, int]):
        self.examples = read_jsonl(path)
        self.mapping = mapping
        self.vocabulary = vocabulary
        self.eos_token = vocabulary["<EOS>"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        source_tokens = ["<USER>", *_user_tokens(example), "</USER>", "<ITEM_LIST>"]
        for movie_id in example["history"]:
            sid = self.mapping[str(movie_id)]["tokens"]
            source_tokens.extend(sid)
        source_tokens.append("</ITEM_LIST>")
        source = [self.vocabulary[token] for token in source_tokens]
        target = [self.vocabulary[token] for token in self.mapping[str(example["target"])]["tokens"]] + [self.eos_token]
        return {
            "input_ids": source,
            "labels": target,
            "history": [int(value) for value in example["history"]],
            "target": int(example["target"]),
            "user_id": int(example["user_id"]),
        }


class SIDCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_source = max(len(row["input_ids"]) for row in rows)
        max_target = max(len(row["labels"]) for row in rows)
        input_ids = torch.full((len(rows), max_source), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_source), dtype=torch.long)
        labels = torch.full((len(rows), max_target), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            source_length, target_length = len(row["input_ids"]), len(row["labels"])
            input_ids[index, :source_length] = torch.tensor(row["input_ids"])
            attention_mask[index, :source_length] = 1
            labels[index, :target_length] = torch.tensor(row["labels"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "histories": [row["history"] for row in rows],
            "targets": [row["target"] for row in rows],
            "user_ids": [row["user_id"] for row in rows],
        }


def create_t5_model(vocabulary_size: int, generator_config: dict[str, Any]):
    from transformers import T5Config, T5ForConditionalGeneration

    model_config = T5Config(
        vocab_size=vocabulary_size,
        d_model=int(generator_config["d_model"]),
        d_kv=int(generator_config["d_kv"]),
        d_ff=int(generator_config["d_ff"]),
        num_layers=int(generator_config["encoder_layers"]),
        num_decoder_layers=int(generator_config["decoder_layers"]),
        num_heads=int(generator_config["heads"]),
        dropout_rate=float(generator_config["dropout"]),
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        feed_forward_proj="relu",
    )
    return T5ForConditionalGeneration(model_config)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def _teacher_forced_metrics(model, loader: DataLoader, device: torch.device, label_smoothing: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    exact = 0
    count = 0
    level_correct: defaultdict[int, int] = defaultdict(int)
    level_total: defaultdict[int, int] = defaultdict(int)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        decoder_input_ids = model._shift_right(labels)
        with _autocast(device):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing)
        total_loss += float(loss)
        predictions = logits.argmax(-1)
        valid = labels.ne(-100)
        exact += int(((predictions == labels) | ~valid).all(dim=1).sum())
        count += len(labels)
        for position in range(labels.shape[1] - 1):
            position_valid = valid[:, position]
            level_correct[position] += int(((predictions[:, position] == labels[:, position]) & position_valid).sum())
            level_total[position] += int(position_valid.sum())
    result = {"loss": total_loss / max(1, len(loader)), "exact_sid_accuracy": exact / max(1, count)}
    for position in level_total:
        result[f"token_accuracy_{position + 1}"] = level_correct[position] / max(1, level_total[position])
    return result


class TokenTrie:
    def __init__(self, sequences: list[list[int]], eos_token_id: int = 1):
        self.root: dict[int, Any] = {}
        self.eos_token_id = eos_token_id
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(int(token), {})
            node[eos_token_id] = {}

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            if token not in node:
                return [self.eos_token_id]
            node = node[token]
        return list(node.keys()) or [self.eos_token_id]


def _load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_t5_model(checkpoint["vocabulary_size"], checkpoint["generator_config"])
    model.load_state_dict(checkpoint["state_dict"])
    print(device)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def recommend_batch(
    model,
    batch: dict[str, Any],
    mapping: dict[str, Any],
    vocabulary: dict[str, int],
    device: torch.device,
    num_beams: int,
    remove_seen: bool,
) -> tuple[list[list[int]], dict[str, float]]:
    reverse_mapping = {tuple(vocabulary[token] for token in value["tokens"]): int(movie_id) for movie_id, value in mapping.items()}
    tries: list[TokenTrie] = []
    for history in batch["histories"]:
        seen = set(history) if remove_seen else set()
        sequences = [list(tokens) for tokens, movie_id in reverse_mapping.items() if movie_id not in seen]
        tries.append(TokenTrie(sequences))

    def prefix_allowed_tokens(batch_id: int, generated: torch.Tensor) -> list[int]:
        prefix = generated.tolist()
        if prefix and prefix[0] == 0:
            prefix = prefix[1:]
        return tries[batch_id].allowed(prefix)

    max_sid_length = max(len(value["tokens"]) for value in mapping.values())
    generated = model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_sid_length + 1,
        do_sample=False,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        early_stopping=True,
    )
    sequences = generated.sequences.cpu().tolist()
    results: list[list[int]] = []
    valid = 0
    duplicates = 0
    for batch_index in range(len(batch["targets"])):
        candidates: list[int] = []
        for sequence in sequences[batch_index * num_beams:(batch_index + 1) * num_beams]:
            tokens = tuple(token for token in sequence if token not in {0, 1})
            movie_id = reverse_mapping.get(tokens)
            if movie_id is not None:
                valid += 1
                if movie_id in candidates:
                    duplicates += 1
                else:
                    candidates.append(movie_id)
        results.append(candidates)
    total = max(1, len(sequences))
    return results, {"valid_sid_rate": valid / total, "duplicate_candidate_rate": duplicates / total}


def evaluate_generator(
    config: dict[str, Any],
    split: str = "test",
    sid_map_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    device = resolve_device(config.get("device", "auto"))
    artifact_root = Path(config["paths"]["artifact_dir"])
    sid_map_path = Path(sid_map_path or artifact_root / "sid" / "movie_to_sid.json")
    run_name = sid_map_path.parent.name
    output_dir = ensure_dir(artifact_root / "generator" / run_name)
    mapping = read_json(sid_map_path)
    checkpoint_path = Path(checkpoint_path or output_dir / "best.pt")
    model, checkpoint = _load_model(checkpoint_path, device)
    vocabulary = checkpoint["vocabulary"]
    dataset = SIDDataset(Path(config["paths"]["processed_dir"]) / f"{split}.jsonl", mapping, vocabulary)
    loader = DataLoader(dataset, batch_size=int(config["inference"]["batch_size"]), shuffle=False, collate_fn=SIDCollator())
    all_recommendations: list[list[int]] = []
    all_targets: list[int] = []
    all_user_ids: list[int] = []
    valid_sum = duplicate_sum = 0.0
    teacher_forced = _teacher_forced_metrics(model, loader, device, 0.0)
    for batch in tqdm(loader, desc=f"Evaluating {split}"):
        recommendations, generation = recommend_batch(
            model, batch, mapping, vocabulary, device, int(config["inference"]["num_beams"]),
            bool(config["inference"].get("remove_seen_items", True)),
        )
        all_recommendations.extend(recommendations)
        all_targets.extend(batch["targets"])
        all_user_ids.extend(batch["user_ids"])
        valid_sum += generation["valid_sid_rate"] * len(batch["targets"])
        duplicate_sum += generation["duplicate_candidate_rate"] * len(batch["targets"])
    result = ranking_metrics(all_targets, all_recommendations, config["inference"]["top_k"], len(mapping))
    result.update(teacher_forced)
    max_cutoff = max(int(value) for value in config["inference"]["top_k"])
    processed = Path(config["paths"]["processed_dir"])
    result.update(diversity_metrics(
        all_recommendations, processed / "movies.csv", processed / "interactions.csv", max_cutoff,
    ))
    result["valid_sid_rate"] = valid_sum / max(1, len(dataset))
    result["duplicate_candidate_rate"] = duplicate_sum / max(1, len(dataset))
    result["average_unique_candidates"] = sum(len(value) for value in all_recommendations) / max(1, len(all_recommendations))
    write_json(output_dir / f"{split}_metrics.json", result)
    write_jsonl(output_dir / f"{split}_recommendations.jsonl", (
        {"user_id": user_id, "target": target, "recommendations": ranked}
        for user_id, target, ranked in zip(all_user_ids, all_targets, all_recommendations)
    ))
    return result


def train_generator(config: dict[str, Any], sid_map_path: str | Path | None = None) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    artifact_root = Path(config["paths"]["artifact_dir"])
    sid_map_path = Path(sid_map_path or artifact_root / "sid" / "movie_to_sid.json")
    run_name = sid_map_path.parent.name
    output_dir = ensure_dir(artifact_root / "generator" / run_name)
    mapping = read_json(sid_map_path)
    processed = Path(config["paths"]["processed_dir"])
    train_examples = read_jsonl(processed / "train.jsonl")
    validation_examples = read_jsonl(processed / "validation.jsonl")
    vocabulary = build_vocabulary(mapping, train_examples + validation_examples)
    write_json(output_dir / "vocabulary.json", vocabulary)
    train_dataset = SIDDataset(processed / "train.jsonl", mapping, vocabulary)
    validation_dataset = SIDDataset(processed / "validation.jsonl", mapping, vocabulary)
    collator = SIDCollator()
    gen_cfg = config["generator"]
    train_loader = DataLoader(
        train_dataset, batch_size=int(gen_cfg["batch_size"]), shuffle=True,
        num_workers=int(gen_cfg.get("num_workers", 0)), collate_fn=collator, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_dataset, batch_size=int(gen_cfg["batch_size"]), shuffle=False, collate_fn=collator)
    model = create_t5_model(len(vocabulary), gen_cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(gen_cfg["learning_rate"]), weight_decay=float(gen_cfg["weight_decay"]))
    accumulation = int(gen_cfg["gradient_accumulation"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * int(gen_cfg["epochs"])
    warmup_updates = round(total_updates * float(gen_cfg["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_schedule(step, total_updates, warmup_updates))
    best_score = -1.0
    patience = 0
    history: list[dict[str, Any]] = []

    for epoch in trange(int(gen_cfg["epochs"]), desc="Training Mini-T5"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            with _autocast(device):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    decoder_input_ids=model._shift_right(labels),
                ).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
                    label_smoothing=float(gen_cfg["label_smoothing"]),
                )
            (loss / accumulation).backward()
            running_loss += float(loss.detach())
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gen_cfg["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        validation = _teacher_forced_metrics(model, validation_loader, device, float(gen_cfg["label_smoothing"]))
        selection_name = "exact_sid_accuracy"
        if bool(gen_cfg.get("validation_ranking", True)):
            validation_recommendations: list[list[int]] = []
            validation_targets: list[int] = []
            for validation_batch in validation_loader:
                ranked, _ = recommend_batch(
                    model, validation_batch, mapping, vocabulary, device,
                    int(config["inference"]["num_beams"]),
                    bool(config["inference"].get("remove_seen_items", True)),
                )
                validation_recommendations.extend(ranked)
                validation_targets.extend(validation_batch["targets"])
            ranking = ranking_metrics(
                validation_targets, validation_recommendations,
                config["inference"]["top_k"], len(mapping),
            )
            validation.update(ranking)
            selection_name = f"NDCG@{max(int(value) for value in config['inference']['top_k'])}"
        score = validation[selection_name]
        epoch_result = {"epoch": epoch + 1, "train_loss": running_loss / max(1, len(train_loader)), **validation}
        history.append(epoch_result)
        if score > best_score:
            best_score = score
            patience = 0
            atomic_torch_save({
                "state_dict": model.state_dict(),
                "generator_config": dict(gen_cfg),
                "vocabulary": vocabulary,
                "vocabulary_size": len(vocabulary),
                "sid_map_path": str(sid_map_path),
            }, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= int(gen_cfg["early_stop_patience"]):
                break
    write_json(output_dir / "history.json", history)
    summary = {
        "parameter_count": parameter_count,
        "vocabulary_size": len(vocabulary),
        "epochs_trained": len(history),
        "selection_metric": selection_name,
        "best_validation_score": best_score,
        "device": str(device),
    }
    write_json(output_dir / "train_summary.json", summary)
    return summary
