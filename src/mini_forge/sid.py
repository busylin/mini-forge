from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .rqvae import load_rqvae
from .utils import ensure_dir, resolve_device, seed_everything, write_json


def _codebook_stats(codes: np.ndarray, codebook_size: int) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=codebook_size).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        used = int((counts > 0).sum())
        result.append({
            "level": level + 1,
            "utilization": used / codebook_size,
            "dead_code_rate": 1.0 - used / codebook_size,
            "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
        })
    return result


def _genre_jaccard(movies: pd.DataFrame, codes: np.ndarray, seed: int) -> dict[str, float]:
    genres = [set(str(value).split("|")) for value in movies.genres]
    rng = np.random.default_rng(seed)

    def average_for_prefix(levels: int, cap: int = 20000) -> float:
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, code in enumerate(codes):
            buckets[tuple(int(x) for x in code[:levels])].append(index)
        pairs: list[tuple[int, int]] = []
        for indices in buckets.values():
            pairs.extend(itertools.combinations(indices, 2))
            if len(pairs) >= cap:
                break
        if not pairs:
            return 0.0
        if len(pairs) > cap:
            chosen = rng.choice(len(pairs), cap, replace=False)
            pairs = [pairs[index] for index in chosen]
        return float(np.mean([len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b])) for a, b in pairs]))

    random_pairs = rng.integers(0, len(genres), size=(min(20000, len(genres) * 5), 2))
    random_score = float(np.mean([len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b])) for a, b in random_pairs if a != b]))
    return {"same_l1": average_for_prefix(1), "same_l1_l2": average_for_prefix(min(2, codes.shape[1])), "random": random_score}


def _neighbor_recall(original: np.ndarray, quantized: np.ndarray, k: int = 10) -> float:
    original = original / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
    quantized = quantized / np.maximum(np.linalg.norm(quantized, axis=1, keepdims=True), 1e-12)
    original_similarity = original @ original.T
    quantized_similarity = quantized @ quantized.T
    np.fill_diagonal(original_similarity, -np.inf)
    np.fill_diagonal(quantized_similarity, -np.inf)
    original_top = np.argpartition(original_similarity, -k, axis=1)[:, -k:]
    quantized_top = np.argpartition(quantized_similarity, -k, axis=1)[:, -k:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(original_top, quantized_top)]))


def export_rqvae_sid(config: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(config.get("device", "auto"))
    artifact_root = Path(config["paths"]["artifact_dir"])
    output_dir = ensure_dir(artifact_root / "sid")
    embeddings = np.load(artifact_root / "content" / "movie_embeddings.npy").astype(np.float32)
    movie_ids = np.load(artifact_root / "content" / "movie_ids.npy").astype(np.int64)
    movies = pd.read_csv(Path(config["paths"]["processed_dir"]) / "movies.csv").set_index("movie_id").loc[movie_ids].reset_index()
    model = load_rqvae(artifact_root / "rqvae" / "best.pt", device)
    with torch.inference_mode():
        output = model(torch.from_numpy(embeddings).to(device))
    codes = output["codes"].cpu().numpy().astype(np.int64)
    reconstructed = output["reconstruction"].cpu().numpy()
    quantized = output["quantized"].cpu().numpy()

    collision_indices = np.zeros(len(movie_ids), dtype=np.int64)
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, code in enumerate(codes):
        buckets[tuple(int(value) for value in code)].append(index)
    for indices in buckets.values():
        for collision, index in enumerate(sorted(indices, key=lambda value: int(movie_ids[value]))):
            collision_indices[index] = collision
    mapping: dict[str, Any] = {}
    for movie_id, code, collision in zip(movie_ids, codes, collision_indices):
        tokens = [f"<L{level + 1}_{int(value)}>" for level, value in enumerate(code)] + [f"<C_{int(collision)}>" ]
        mapping[str(int(movie_id))] = {"codes": [int(value) for value in code], "collision": int(collision), "tokens": tokens}
    write_json(output_dir / "movie_to_sid.json", mapping)
    np.save(output_dir / "codes.npy", codes)
    np.save(output_dir / "quantized_embeddings.npy", quantized.astype(np.float32))
    collision_movies = sum(len(indices) for indices in buckets.values() if len(indices) > 1)
    raw_pairs = np.load(Path(config["paths"]["processed_dir"]) / "i2i_pairs.npy")
    code_by_movie = {int(movie_id): code for movie_id, code in zip(movie_ids, codes)}
    collaborative_rows = [
        (code_by_movie[int(a)], code_by_movie[int(b)], float(weight))
        for a, b, weight in raw_pairs if int(a) in code_by_movie and int(b) in code_by_movie
    ]
    if collaborative_rows:
        pair_weights = np.asarray([row[2] for row in collaborative_rows])
        shared_l1 = np.asarray([row[0][0] == row[1][0] for row in collaborative_rows], dtype=np.float64)
        prefix_lengths = np.asarray([
            next((level for level, (left, right) in enumerate(zip(row[0], row[1])) if left != right), len(row[0]))
            for row in collaborative_rows
        ], dtype=np.float64)
        if np.std(pair_weights) > 0 and np.std(prefix_lengths) > 0:
            correlation = spearmanr(pair_weights, prefix_lengths).statistic
        else:
            correlation = 0.0
        collaborative_quality = {
            "shared_l1_rate": float(shared_l1.mean()),
            "shared_l1_l2_rate": float(np.mean([np.array_equal(row[0][:2], row[1][:2]) for row in collaborative_rows])),
            "cooccurrence_prefix_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        }
    else:
        collaborative_quality = {"shared_l1_rate": 0.0, "shared_l1_l2_rate": 0.0, "cooccurrence_prefix_spearman": 0.0}
    metrics = {
        "reconstruction_mse": float(np.mean((embeddings - reconstructed) ** 2)),
        "reconstruction_cosine": float(np.mean(np.sum(
            embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            * reconstructed / np.maximum(np.linalg.norm(reconstructed, axis=1, keepdims=True), 1e-12), axis=1,
        ))),
        "neighbor_recall_at_10": _neighbor_recall(embeddings, quantized, min(10, len(embeddings) - 1)),
        "codebooks": _codebook_stats(codes, int(config["rqvae"]["codebook_size"])),
        "unique_semantic_ids": len(buckets),
        "collision_rate": collision_movies / len(movie_ids),
        "average_bucket_size": len(movie_ids) / len(buckets),
        "max_bucket_size": max(len(indices) for indices in buckets.values()),
        "unique_mapping_rate_after_collision_token": len(set(tuple(value["tokens"]) for value in mapping.values())) / len(mapping),
        "genre_jaccard": _genre_jaccard(movies, codes, int(config["seed"])),
        "collaborative_quality": collaborative_quality,
    }
    write_json(output_dir / "quality_metrics.json", metrics)
    return metrics


def create_alternative_sid(config: dict[str, Any], mode: str) -> Path:
    if mode not in {"random", "direct"}:
        raise ValueError("mode must be random or direct")
    seed_everything(int(config["seed"]))
    artifact_root = Path(config["paths"]["artifact_dir"])
    movie_ids = np.load(artifact_root / "content" / "movie_ids.npy").astype(np.int64)
    output_dir = ensure_dir(artifact_root / f"sid_{mode}")
    mapping: dict[str, Any] = {}
    if mode == "direct":
        for movie_id in movie_ids:
            mapping[str(int(movie_id))] = {"codes": [int(movie_id)], "collision": 0, "tokens": [f"<MOVIE_{int(movie_id)}>"]}
    else:
        levels = int(config["rqvae"]["num_codebooks"])
        size = int(config["rqvae"]["codebook_size"])
        capacity = size ** levels
        if len(movie_ids) > capacity:
            raise ValueError("Random SID capacity is smaller than the item catalog")
        rng = np.random.default_rng(int(config["seed"]))
        flat_codes = rng.choice(capacity, size=len(movie_ids), replace=False)
        for movie_id, flat in zip(movie_ids, flat_codes):
            code: list[int] = []
            value = int(flat)
            for _ in range(levels):
                code.append(value % size)
                value //= size
            mapping[str(int(movie_id))] = {
                "codes": code, "collision": 0,
                "tokens": [f"<L{level + 1}_{number}>" for level, number in enumerate(code)] + ["<C_0>"],
            }
    path = output_dir / "movie_to_sid.json"
    write_json(path, mapping)
    return path
