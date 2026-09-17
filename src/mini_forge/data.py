from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .utils import ensure_dir, write_json, write_jsonl

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _looks_like_movielens_1m(raw_dir: Path) -> bool:
    ratings = raw_dir / "ratings.dat"
    movies = raw_dir / "movies.dat"
    users = raw_dir / "users.dat"
    return (
        ratings.exists()
        and movies.exists()
        and users.exists()
        and ratings.stat().st_size > 10_000_000
        and movies.stat().st_size > 100_000
    )


def download_movielens(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    if _looks_like_movielens_1m(raw_dir):
        return raw_dir
    archive = ensure_dir(raw_dir.parent) / "ml-1m.zip"
    with requests.get(MOVIELENS_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(raw_dir.parent)
    return raw_dir


def _k_core(frame: pd.DataFrame, min_users: int, min_items: int) -> pd.DataFrame:
    current = frame
    while True:
        before = len(current)
        user_counts = current.groupby("user_id").size()
        current = current[current.user_id.isin(user_counts[user_counts >= min_users].index)]
        item_counts = current.groupby("movie_id").size()
        current = current[current.movie_id.isin(item_counts[item_counts >= min_items].index)]
        if len(current) == before:
            return current.copy()


def _make_i2i_pairs(sequences: dict[int, list[int]], window: int) -> np.ndarray:
    weights: Counter[tuple[int, int]] = Counter()
    for sequence in sequences.values():
        for left in range(len(sequence)):
            for right in range(left + 1, min(len(sequence), left + window + 1)):
                a, b = sorted((sequence[left], sequence[right]))
                weights[(a, b)] += 1.0 / (right - left)
    if not weights:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([(a, b, weight) for (a, b), weight in weights.items()], dtype=np.float64)


def prepare_movielens(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(config["paths"]["raw_dir"])
    output_dir = ensure_dir(config["paths"]["processed_dir"])
    if not (raw_dir / "ratings.dat").exists():
        raise FileNotFoundError(f"Missing {raw_dir / 'ratings.dat'}; run the download command first")
    ratings = pd.read_csv(
        raw_dir / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        raw_dir / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    users_path = raw_dir / "users.dat"
    if users_path.exists():
        users = pd.read_csv(
            users_path, sep="::", engine="python", encoding="latin-1",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
        )
    else:
        # Small synthetic fixtures may omit users.dat. Keep their pipeline usable
        # while making the missing attributes explicit in the generated tokens.
        users = pd.DataFrame({
            "user_id": sorted(ratings.user_id.unique()),
            "gender": "UNK",
            "age": "UNK",
            "occupation": "UNK",
            "zip_code": "UNK",
        })
    data_cfg = config["data"]
    ratings = ratings[ratings.rating >= data_cfg["positive_rating"]]
    ratings = _k_core(
        ratings,
        int(data_cfg["min_user_interactions"]),
        int(data_cfg["min_item_interactions"]),
    ).sort_values(["user_id", "timestamp", "movie_id"])

    interactions: list[dict[str, Any]] = []
    train_sequences: dict[int, list[int]] = {}
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    train_examples: list[dict[str, Any]] = []
    min_history = int(data_cfg["min_history"])
    max_history = int(data_cfg["max_history"])
    stride = int(data_cfg["stride"])
    user_profiles = users.set_index("user_id").to_dict("index")

    def user_fields(user_id: int) -> dict[str, Any]:
        profile = user_profiles.get(int(user_id))
        if profile is None:
            raise ValueError(f"Missing profile for user_id={user_id} in users.dat")
        return {
            "user_id": int(user_id),
            "gender": str(profile["gender"]),
            "age": str(profile["age"]),
            "occupation": str(profile["occupation"]),
        }

    for user_id, group in ratings.groupby("user_id", sort=True):
        records = group.sort_values(["timestamp", "movie_id"]).to_dict("records")
        movie_ids = [int(row["movie_id"]) for row in records]
        train_items, validation_item, test_item = movie_ids[:-2], movie_ids[-2], movie_ids[-1]
        train_sequences[int(user_id)] = train_items
        for position, row in enumerate(records):
            split = "train" if position < len(records) - 2 else ("validation" if position == len(records) - 2 else "test")
            interactions.append({**{key: int(row[key]) for key in ["user_id", "movie_id", "rating", "timestamp"]}, "split": split})
        for target_position in range(min_history, len(train_items), stride):
            train_examples.append({
                **user_fields(int(user_id)),
                "history": train_items[max(0, target_position - max_history):target_position],
                "target": train_items[target_position],
            })
        validation_rows.append({
            **user_fields(int(user_id)),
            "history": train_items[-max_history:],
            "target": validation_item,
        })
        test_rows.append({
            **user_fields(int(user_id)),
            "history": (train_items + [validation_item])[-max_history:],
            "target": test_item,
        })

    kept_movies = movies[movies.movie_id.isin(ratings.movie_id.unique())].copy().sort_values("movie_id")
    kept_users = users[users.user_id.isin(ratings.user_id.unique())].copy().sort_values("user_id")
    kept_movies.to_csv(output_dir / "movies.csv", index=False)
    kept_users.to_csv(output_dir / "users.csv", index=False)
    pd.DataFrame(interactions).to_csv(output_dir / "interactions.csv", index=False)
    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    pairs = _make_i2i_pairs(train_sequences, int(data_cfg["i2i_window"]))
    np.save(output_dir / "i2i_pairs.npy", pairs)
    checksum = hashlib.sha256((raw_dir / "ratings.dat").read_bytes()).hexdigest()
    metadata = {
        "num_users": int(ratings.user_id.nunique()),
        "num_movies": int(ratings.movie_id.nunique()),
        "num_positive_interactions": int(len(ratings)),
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(validation_rows),
        "num_test_examples": len(test_rows),
        "num_i2i_pairs": int(len(pairs)),
        "ratings_sha256": checksum,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
