from __future__ import annotations

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "smoke_raw" / "ml-1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    movies = [
        (1, "Alpha (2000)", "Action|Adventure"),
        (2, "Beta (2001)", "Comedy"),
        (3, "Gamma (2002)", "Drama"),
        (4, "Delta (2003)", "Action|Thriller"),
        (5, "Epsilon (2004)", "Comedy|Romance"),
        (6, "Zeta (2005)", "Drama|Romance"),
        (7, "Eta (2006)", "Animation|Children's"),
        (8, "Theta (2007)", "Sci-Fi|Adventure"),
    ]
    (raw_dir / "movies.dat").write_text(
        "\n".join(f"{movie_id}::{title}::{genres}" for movie_id, title, genres in movies) + "\n",
        encoding="latin-1",
    )
    rows: list[str] = []
    for user_id in range(1, 7):
        sequence = [((offset + user_id - 1) % len(movies)) + 1 for offset in range(7)]
        for position, movie_id in enumerate(sequence):
            rows.append(f"{user_id}::{movie_id}::{4 + ((position + user_id) % 2)}::{1_000_000 + user_id * 100 + position}")
    (raw_dir / "ratings.dat").write_text("\n".join(rows) + "\n", encoding="latin-1")
    print(raw_dir)


if __name__ == "__main__":
    main()
