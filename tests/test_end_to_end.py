"""Phase 6 — lightweight end-to-end integration over the fixture.

Chains: validate -> MapReduce (local streaming sim) -> KNN -> anomaly,
asserting each stage produces consistent, non-empty output. Uses only the
committed fixture and temp dirs; no Docker/HDFS required.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.anomaly_detection import AnomalyDetector
from common.schema import load_canonical
from ml.knn_predict import train_and_evaluate
from tests.test_mapreduce import run_streaming

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_raw.txt"


class EndToEndTests(unittest.TestCase):
    def test_full_chain(self):
        canonical = load_canonical(FIXTURE)
        self.assertGreater(len(canonical), 0)

        # Phase 2: MapReduce aggregates
        mr = run_streaming(canonical.to_csv(index=False))
        self.assertGreater(len(mr), 0)
        self.assertEqual(
            set(mr.columns),
            {"road_id", "time_bucket", "count", "avg_volume", "min_volume", "max_volume", "sum_volume"},
        )

        # Phase 3: KNN prediction (R² sanity; baseline comparison is data-dependent)
        result = train_and_evaluate(canonical, k=5, weights="distance", test_frac=0.2)
        self.assertGreater(result["knn"].n_test, 0)
        self.assertGreater(result["knn"].r2, 0.5)

        # Phase 4: anomaly detection
        with tempfile.TemporaryDirectory() as d:
            det = AnomalyDetector(min_samples=3, state_path=Path(d) / "s.json",
                                  alerts_path=Path(d) / "a.csv")
            det.fit_baseline(canonical)
            road = canonical["road_id"].iloc[0]
            row = canonical[(canonical.road_id == road) & (canonical.hour == 0)].iloc[0].copy()
            row["volume"] = 95.0
            alerts = det.process_batch(pd.DataFrame([row]), "e2e")
            self.assertEqual(len(alerts), 1)


if __name__ == "__main__":
    unittest.main()
