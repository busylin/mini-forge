from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import ensure_dir, resolve_device, write_json


def _hashing_embeddings(texts: list[str], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().replace("|", " ").split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            matrix[row, value % dimension] += 1.0 if value & 1 else -1.0
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


@torch.inference_mode()
def _transformer_embeddings(texts: list[str], model_name: str, batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding movie content"):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        batches.append(F.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(batches)


def encode_movies(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(config["paths"]["processed_dir"])
    artifact_dir = ensure_dir(Path(config["paths"]["artifact_dir"]) / "content")
    movies = pd.read_csv(processed_dir / "movies.csv")
    texts = [f"{title} [SEP] {str(genres).replace('|', ' | ')}" for title, genres in zip(movies.title, movies.genres)]
    content_cfg = config["content"]
    if content_cfg["encoder"] == "hashing":
        embeddings = _hashing_embeddings(texts, int(content_cfg.get("output_dim", 384)))
    else:
        embeddings = _transformer_embeddings(
            texts, content_cfg["encoder"], int(content_cfg["batch_size"]),
            int(content_cfg["max_length"]), resolve_device(config.get("device", "auto")),
        )
    if content_cfg.get("normalize", True):
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    np.save(artifact_dir / "movie_embeddings.npy", embeddings.astype(np.float32))
    np.save(artifact_dir / "movie_ids.npy", movies.movie_id.to_numpy(np.int64))
    metadata = {"encoder": content_cfg["encoder"], "shape": list(embeddings.shape), "normalized": True}
    write_json(artifact_dir / "metadata.json", metadata)
    return metadata
