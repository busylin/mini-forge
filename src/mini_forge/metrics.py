from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ranking_metrics(
    targets: list[int],
    recommendations: list[list[int]],
    cutoffs: Iterable[int],
    catalog_size: int,
) -> dict[str, float]:
    cutoffs = sorted(set(int(value) for value in cutoffs))
    result: dict[str, float] = {}
    ranks: list[int | None] = []
    for target, ranked in zip(targets, recommendations):
        try:
            ranks.append(ranked.index(target) + 1)
        except ValueError:
            ranks.append(None)
    for cutoff in cutoffs:
        hits = [rank is not None and rank <= cutoff for rank in ranks]
        result[f"HR@{cutoff}"] = sum(hits) / max(1, len(hits))
        result[f"NDCG@{cutoff}"] = sum(
            1.0 / math.log2(rank + 1) if rank is not None and rank <= cutoff else 0.0 for rank in ranks
        ) / max(1, len(ranks))
    max_cutoff = max(cutoffs)
    result[f"MRR@{max_cutoff}"] = sum(
        1.0 / rank if rank is not None and rank <= max_cutoff else 0.0 for rank in ranks
    ) / max(1, len(ranks))
    recommended = {item for ranked in recommendations for item in ranked[:max_cutoff]}
    result[f"CatalogCoverage@{max_cutoff}"] = len(recommended) / max(1, catalog_size)
    return result


def diversity_metrics(
    recommendations: list[list[int]],
    movies_path: str | Path,
    interactions_path: str | Path,
    cutoff: int,
) -> dict[str, float]:
    movies = pd.read_csv(movies_path)
    genre_by_movie = {
        int(row.movie_id): set(str(row.genres).split("|")) for row in movies.itertuples(index=False)
    }
    catalog_genres = set().union(*genre_by_movie.values()) if genre_by_movie else set()
    train = pd.read_csv(interactions_path)
    train = train[train.split == "train"]
    popularity = Counter(int(value) for value in train.movie_id)
    total_interactions = max(1, len(train))
    genre_counts: Counter[str] = Counter()
    pair_diversities: list[float] = []
    popularity_values: list[float] = []
    novelty_values: list[float] = []
    for ranked in recommendations:
        selected = ranked[:cutoff]
        selected_genres = [genre_by_movie.get(int(movie_id), set()) for movie_id in selected]
        for movie_id, genres in zip(selected, selected_genres):
            genre_counts.update(genres)
            count = popularity.get(int(movie_id), 0)
            popularity_values.append(float(count))
            novelty_values.append(-math.log2((count + 1) / (total_interactions + len(movies))))
        for left in range(len(selected_genres)):
            for right in range(left + 1, len(selected_genres)):
                union = selected_genres[left] | selected_genres[right]
                pair_diversities.append(1.0 - len(selected_genres[left] & selected_genres[right]) / max(1, len(union)))
    probabilities = np.asarray(list(genre_counts.values()), dtype=np.float64)
    probabilities = probabilities / probabilities.sum() if probabilities.sum() else probabilities
    return {
        f"GenreCoverage@{cutoff}": len(genre_counts) / max(1, len(catalog_genres)),
        f"GenreEntropy@{cutoff}": float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0,
        f"IntraListGenreDiversity@{cutoff}": float(np.mean(pair_diversities)) if pair_diversities else 0.0,
        f"AveragePopularity@{cutoff}": float(np.mean(popularity_values)) if popularity_values else 0.0,
        f"Novelty@{cutoff}": float(np.mean(novelty_values)) if novelty_values else 0.0,
    }
