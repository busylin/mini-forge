from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .content import encode_movies
from .data import download_movielens, prepare_movielens
from .generator import evaluate_generator, train_generator
from .rqvae import train_rqvae
from .sid import export_rqvae_sid


def _print_result(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-forge",
        description="Semantic-ID generation and constrained generative recommendation",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override a dotted configuration key",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    subparsers.add_parser("prepare")
    subparsers.add_parser("embed")
    subparsers.add_parser("train-rqvae")
    subparsers.add_parser("export-sid")
    subparsers.add_parser("train-generator")
    generator_eval = subparsers.add_parser("eval-generator")
    generator_eval.add_argument("--checkpoint")
    generator_eval.add_argument("--split", choices=["validation", "test"], default="test")
    pipeline = subparsers.add_parser("run-all")
    pipeline.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.set)
    if args.command == "download":
        _print_result({"raw_dir": str(download_movielens(config["paths"]["raw_dir"]))})
    elif args.command == "prepare":
        _print_result(prepare_movielens(config))
    elif args.command == "embed":
        _print_result(encode_movies(config))
    elif args.command == "train-rqvae":
        _print_result(train_rqvae(config))
    elif args.command == "export-sid":
        _print_result(export_rqvae_sid(config))
    elif args.command == "train-generator":
        _print_result(train_generator(config))
    elif args.command == "eval-generator":
        _print_result(evaluate_generator(config, args.split, checkpoint_path=args.checkpoint))
    elif args.command == "run-all":
        if not args.skip_download:
            download_movielens(config["paths"]["raw_dir"])
        results = {
            "prepare": prepare_movielens(config),
            "content": encode_movies(config),
            "rqvae_train": train_rqvae(config),
            "sid_quality": export_rqvae_sid(config),
            "generator_train": train_generator(config),
            "generator_test": evaluate_generator(config),
        }
        _print_result(results)


if __name__ == "__main__":
    main()

