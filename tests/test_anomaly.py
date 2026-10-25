"""Phase 4 tests — near-real-time anomaly detection."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.anomaly_detection import AnomalyDetector, compute_baselines
from common.schema import load_canonical

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_raw.txt"


class AnomalyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_canonical(FIXTURE)

    def _detector(self, tmp: Path, clock=None) -> AnomalyDetector:
        kwargs = dict(
            threshold=3.5,
            min_samples=3,
            state_path=tmp / "state.json",
            alerts_path=tmp / "alerts.csv",
        )
        if clock is not None:
            kwargs["clock"] = clock
        det = AnomalyDetector(**kwargs)
        det.fit_baseline(self.history)
        return det

    def test_baselines_built(self):
        b = compute_baselines(self.history, min_samples=3)
        self.assertGreater(len(b), 0)

    def test_injected_anomaly_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(Path(d))
            # A clearly anomalous low-hour spike for an existing road.
            road = self.history["road_id"].iloc[0]
            row = self.history[(self.history.road_id == road) & (self.history.hour == 0)].iloc[0].copy()
            row["volume"] = 95.0
            batch = pd.DataFrame([row])
            alerts = det.process_batch(batch, "b_anom")
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["direction"], "high")
            self.assertGreaterEqual(abs(alerts[0]["deviation_score"]), 3.5)

    def test_normal_batch_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(Path(d))
            road = self.history["road_id"].iloc[0]
            normal = self.history[(self.history.road_id == road) & (self.history.hour == 0)].head(3)
            alerts = det.process_batch(normal, "b_norm")
            self.assertEqual(len(alerts), 0)

    def test_duplicate_batch_skipped_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(Path(d))
            road = self.history["road_id"].iloc[0]
            row = self.history[(self.history.road_id == road) & (self.history.hour == 0)].iloc[0].copy()
            row["volume"] = 95.0
            batch = pd.DataFrame([row])
            first = det.process_batch(batch, "b1")
            again = det.process_batch(batch, "b1")  # same id
            self.assertEqual(len(first), 1)
            self.assertEqual(len(again), 0)

    def test_warmup_skip_for_unknown_bucket(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(Path(d))
            row = self.history.iloc[0].copy()
            row["hour"] = 5  # a bucket absent from the fixture -> no baseline
            alerts = det.process_batch(pd.DataFrame([row]), "b_warm")
            self.assertEqual(len(alerts), 0)
            self.assertGreaterEqual(det.stats["warmup_skips"], 1)

    def test_malformed_rows_not_alerted(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(Path(d))
            bad = pd.DataFrame(
                {"lat": [12.90], "lon": [77.60], "day_of_week": [0], "hour": [99], "volume": [95]}
            )
            alerts = det.process_batch(bad, "b_bad")
            self.assertEqual(len(alerts), 0)
            self.assertGreaterEqual(det.stats["malformed"], 1)

    def test_state_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            det = self._detector(tmp)
            road = self.history["road_id"].iloc[0]
            row = self.history[(self.history.road_id == road) & (self.history.hour == 0)].iloc[0].copy()
            row["volume"] = 95.0
            det.process_batch(pd.DataFrame([row]), "b1")
            det.save_state()

            det2 = AnomalyDetector(
                threshold=3.5, min_samples=3,
                state_path=tmp / "state.json", alerts_path=tmp / "alerts.csv",
            )
            det2.load_state()
            self.assertIn("b1", det2.processed_batches)
            self.assertGreater(len(det2.baselines), 0)
            # reprocessing the same batch after recovery yields no new alerts
            self.assertEqual(len(det2.process_batch(pd.DataFrame([row]), "b1")), 0)

    def test_latency_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            ticks = iter([100.0, 100.05, 100.06, 100.07])
            det = self._detector(Path(d), clock=lambda: next(ticks))
            road = self.history["road_id"].iloc[0]
            row = self.history[(self.history.road_id == road) & (self.history.hour == 0)].iloc[0].copy()
            row["volume"] = 95.0
            alerts = det.process_batch(pd.DataFrame([row]), "b1", arrival_time=100.0)
            self.assertEqual(len(alerts), 1)
            self.assertGreaterEqual(alerts[0]["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
