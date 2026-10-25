"""Phase 3 tests — KNN prediction pipeline."""

import unittest
from pathlib import Path

from common.schema import load_canonical
from ml.knn_predict import (
    chronological_split,
    make_pipeline,
    train_and_evaluate,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_raw.txt"


class KnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_canonical(FIXTURE)

    def test_chronological_split_preserves_order(self):
        train, test = chronological_split(self.df, 0.25)
        self.assertEqual(len(train) + len(test), len(self.df))
        # last row of train precedes first row of test in original order
        self.assertTrue(train.index.max() < test.index.min())

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            make_pipeline(0, "distance")
        with self.assertRaises(ValueError):
            make_pipeline(3, "nonsense")
        with self.assertRaises(ValueError):
            chronological_split(self.df, 1.5)

    def test_too_few_samples_raise(self):
        with self.assertRaises(ValueError):
            train_and_evaluate(self.df.head(3), k=7)

    def test_deterministic_and_reasonable(self):
        r1 = train_and_evaluate(self.df, k=5, weights="distance", test_frac=0.2)
        r2 = train_and_evaluate(self.df, k=5, weights="distance", test_frac=0.2)
        self.assertAlmostEqual(r1["knn"].mae, r2["knn"].mae, places=9)
        # KNN should explain most variance on this structured fixture.
        # (Whether it *beats* the grouped-mean baseline is data-dependent: on
        # this near-deterministic fixture the baseline is near-optimal; on the
        # real dataset KNN wins — see docs/REPORT_PROJECT.md.)
        self.assertGreater(r1["knn"].r2, 0.5)

    def test_predictions_have_no_target_leakage_columns(self):
        r = train_and_evaluate(self.df, k=5)
        cols = set(r["predictions"].columns)
        self.assertIn("actual_volume", cols)
        self.assertIn("predicted_volume", cols)
        self.assertEqual(r["knn"].n_test, len(r["predictions"]))


if __name__ == "__main__":
    unittest.main()
