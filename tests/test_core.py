from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mini_forge.data import prepare_movielens
from mini_forge.generator import TokenTrie, build_vocabulary
from mini_forge.rqvae import RQVAE


class CoreTests(unittest.TestCase):
    def test_rqvae_shapes_and_gradients(self) -> None:
        model = RQVAE(16, 12, 4, 3, 8)
        values = torch.randn(10, 16)
        output = model(values)
        self.assertEqual(output["reconstruction"].shape, values.shape)
        self.assertEqual(output["codes"].shape, (10, 3))
        loss = torch.nn.functional.mse_loss(output["reconstruction"], values) + output["vq_loss"]
        loss.backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_vocabulary_and_trie(self) -> None:
        mapping = {
            "1": {"tokens": ["<L1_0>", "<L2_1>", "<C_0>"]},
            "2": {"tokens": ["<L1_0>", "<L2_2>", "<C_0>"]},
        }
        vocabulary = build_vocabulary(mapping)
        sequences = [
            [vocabulary[token] for token in value["tokens"]]
            for value in mapping.values()
        ]
        trie = TokenTrie(sequences)
        self.assertEqual(trie.allowed([]), [vocabulary["<L1_0>"]])
        self.assertEqual(
            set(trie.allowed([vocabulary["<L1_0>"]])),
            {vocabulary["<L2_1>"], vocabulary["<L2_2>"]},
        )

    def test_prepare_leave_one_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "movies.dat").write_text(
                "1::A::Drama\n2::B::Comedy\n3::C::Action\n4::D::Drama\n5::E::Comedy\n",
                encoding="latin-1",
            )
            ratings = []
            for user in [1, 2]:
                for timestamp, movie in enumerate([1, 2, 3, 4, 5], start=1):
                    ratings.append(f"{user}::{movie}::5::{timestamp}")
            (raw / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
            config = {
                "paths": {"raw_dir": str(raw), "processed_dir": str(root / "processed")},
                "data": {
                    "positive_rating": 4,
                    "min_user_interactions": 3,
                    "min_item_interactions": 1,
                    "min_history": 2,
                    "max_history": 3,
                    "stride": 1,
                    "i2i_window": 2,
                },
            }
            metadata = prepare_movielens(config)
            self.assertEqual(metadata["num_users"], 2)
            self.assertEqual(metadata["num_validation_examples"], 2)
            self.assertEqual(metadata["num_test_examples"], 2)
            self.assertEqual(metadata["num_train_examples"], 2)


if __name__ == "__main__":
    unittest.main()

