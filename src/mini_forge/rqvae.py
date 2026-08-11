from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .utils import atomic_torch_save, cosine_schedule, ensure_dir, resolve_device, seed_everything, write_json


class ResidualQuantizer(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, dimension: int, commitment_beta: float = 0.25):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dimension = dimension
        self.commitment_beta = commitment_beta
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dimension) for _ in range(num_codebooks)])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        quantized = torch.zeros_like(latent)
        losses: list[torch.Tensor] = []
        all_codes: list[torch.Tensor] = []
        for codebook in self.codebooks:
            distances = (
                residual.square().sum(-1, keepdim=True)
                + codebook.weight.square().sum(-1).unsqueeze(0)
                - 2.0 * residual @ codebook.weight.t()
            )
            codes = distances.argmin(-1)
            selected = codebook(codes)
            losses.append(F.mse_loss(selected, residual.detach()))
            losses.append(self.commitment_beta * F.mse_loss(residual, selected.detach()))
            quantized = quantized + selected
            residual = residual - selected.detach()
            all_codes.append(codes)
        straight_through = latent + (quantized - latent).detach()
        return straight_through, torch.stack(losses).sum(), torch.stack(all_codes, dim=-1)


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "commitment_beta": commitment_beta,
        }
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualQuantizer(num_codebooks, codebook_size, latent_dim, commitment_beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(values)
        quantized, vq_loss, codes = self.quantizer(latent)
        reconstruction = self.decoder(quantized)
        return {"reconstruction": reconstruction, "vq_loss": vq_loss, "codes": codes, "quantized": quantized}

    @torch.inference_mode()
    def initialize_codebooks(self, values: torch.Tensor, iterations: int = 30) -> None:
        residual = self.encoder(values)
        for codebook in self.quantizer.codebooks:
            centers, assignments = _torch_kmeans(residual, codebook.num_embeddings, iterations)
            codebook.weight.copy_(centers)
            residual = residual - centers[assignments]


@torch.inference_mode()
def _torch_kmeans(values: torch.Tensor, clusters: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(values) >= clusters:
        centers = values[torch.randperm(len(values), device=values.device)[:clusters]].clone()
    else:
        repeats = math.ceil(clusters / len(values))
        centers = values.repeat(repeats, 1)[:clusters].clone()
    assignments = torch.zeros(len(values), dtype=torch.long, device=values.device)
    for _ in range(iterations):
        distances = torch.cdist(values, centers)
        assignments = distances.argmin(1)
        for index in range(clusters):
            members = values[assignments == index]
            if len(members):
                centers[index] = members.mean(0)
            else:
                centers[index] = values[torch.randint(len(values), (1,), device=values.device)]
    return centers, assignments


def _info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.t() / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_rqvae(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    artifact_root = Path(config["paths"]["artifact_dir"])
    content_dir = artifact_root / "content"
    output_dir = ensure_dir(artifact_root / "rqvae")
    embeddings = torch.from_numpy(np.load(content_dir / "movie_embeddings.npy")).float()
    movie_ids = np.load(content_dir / "movie_ids.npy").astype(np.int64)
    rq_cfg = config["rqvae"]
    if embeddings.shape[1] != int(rq_cfg["input_dim"]):
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match rqvae.input_dim={rq_cfg['input_dim']}")
    model = RQVAE(**{
        key: rq_cfg[key] for key in ["input_dim", "hidden_dim", "latent_dim", "num_codebooks", "codebook_size", "commitment_beta"]
    }).to(device)
    model.initialize_codebooks(embeddings.to(device), int(rq_cfg.get("kmeans_iterations", 30)))

    generator = torch.Generator().manual_seed(int(config["seed"]))
    validation_count = max(1, round(len(embeddings) * float(rq_cfg.get("validation_ratio", 0.1))))
    permutation = torch.randperm(len(embeddings), generator=generator)
    validation_indices, train_indices = permutation[:validation_count], permutation[validation_count:]
    train_loader = DataLoader(TensorDataset(embeddings[train_indices]), batch_size=int(rq_cfg["batch_size"]), shuffle=True)
    validation_loader = DataLoader(TensorDataset(embeddings[validation_indices]), batch_size=int(rq_cfg["batch_size"]), shuffle=False)

    raw_pairs = np.load(Path(config["paths"]["processed_dir"]) / "i2i_pairs.npy")
    id_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    valid_pairs = [(id_to_index[int(a)], id_to_index[int(b)], float(w)) for a, b, w in raw_pairs if int(a) in id_to_index and int(b) in id_to_index]
    if valid_pairs:
        pair_array = np.asarray(valid_pairs, dtype=np.float64)
        pair_probabilities = pair_array[:, 2] / pair_array[:, 2].sum()
    else:
        pair_array = np.empty((0, 3), dtype=np.float64)
        pair_probabilities = np.empty(0, dtype=np.float64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rq_cfg["learning_rate"]), weight_decay=float(rq_cfg["weight_decay"]))
    total_steps = int(rq_cfg["epochs"]) * max(1, len(train_loader))
    warmup_steps = int(rq_cfg["warmup_epochs"]) * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps))
    best_validation = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in trange(int(rq_cfg["epochs"]), desc="Training RQ-VAE"):
        model.train()
        totals = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "i2i": 0.0}
        for (batch,) in train_loader:
            batch = batch.to(device)
            output = model(batch)
            reconstruction_loss = F.mse_loss(output["reconstruction"], batch)
            contrastive_loss = torch.zeros((), device=device)
            if len(pair_array) and float(rq_cfg["i2i_weight"]) > 0:
                sample_count = min(int(rq_cfg["pair_batch_size"]), len(pair_array))
                sampled = np.random.choice(len(pair_array), size=sample_count, replace=True, p=pair_probabilities)
                left_indices = torch.as_tensor(pair_array[sampled, 0], dtype=torch.long)
                right_indices = torch.as_tensor(pair_array[sampled, 1], dtype=torch.long)
                left = model(embeddings[left_indices].to(device))["quantized"]
                right = model(embeddings[right_indices].to(device))["quantized"]
                contrastive_loss = _info_nce(left, right, float(rq_cfg["temperature"]))
            loss = reconstruction_loss + output["vq_loss"] + float(rq_cfg["i2i_weight"]) * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(rq_cfg["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            for key, value in [("loss", loss), ("recon", reconstruction_loss), ("vq", output["vq_loss"]), ("i2i", contrastive_loss)]:
                totals[key] += float(value.detach())

        model.eval()
        validation_recon = 0.0
        validation_cosine = 0.0
        with torch.inference_mode():
            for (batch,) in validation_loader:
                batch = batch.to(device)
                output = model(batch)
                validation_recon += float(F.mse_loss(output["reconstruction"], batch))
                validation_cosine += float(F.cosine_similarity(output["reconstruction"], batch).mean())
        validation_recon /= max(1, len(validation_loader))
        validation_cosine /= max(1, len(validation_loader))
        epoch_result = {key: value / max(1, len(train_loader)) for key, value in totals.items()}
        epoch_result.update({"epoch": epoch + 1, "validation_recon": validation_recon, "validation_cosine": validation_cosine})
        history.append(epoch_result)
        if validation_recon < best_validation:
            best_validation = validation_recon
            patience = 0
            atomic_torch_save({"model_config": model.model_config, "state_dict": model.state_dict()}, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= int(rq_cfg["early_stop_patience"]):
                break
    write_json(output_dir / "history.json", history)
    summary = {"best_validation_recon": best_validation, "epochs_trained": len(history), "device": str(device)}
    write_json(output_dir / "train_summary.json", summary)
    return summary


def load_rqvae(path: str | Path, device: torch.device) -> RQVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RQVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()

