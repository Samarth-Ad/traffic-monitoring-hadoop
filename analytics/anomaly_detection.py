"""Phase 4 (beyond syllabus) — near-real-time traffic anomaly detection.

Detects unusual traffic behaviour as new batches arrive, using an
interpretable, robust statistical method:

    baseline per (road_id, hour) = median + MAD (median absolute deviation)
    robust z-score  z = 0.6745 * (observed - median) / MAD
    flag when |z| >= THRESHOLD   (default 3.5)

Why median/MAD rather than mean/std: traffic volume is skewed and the whole
point is to find outliers, so a *robust* location/scale estimate that outliers
don't inflate is the right tool. MAD-based z is the standard robust detector.

What it flags (described as *potential unusual traffic events*, never asserted
to be accidents):
    * unusually high volume for a road at that hour  (z >= +THRESHOLD)
    * unusually low  volume for a road at that hour  (z <= -THRESHOLD)

Guarantees / requirements implemented:
    * Baselines are built from historical/training data only — future batches
      never leak into past decisions.
    * Each batch is processed exactly once (processed-batch ids persisted).
    * One alert per (road_id, day_of_week, hour) within a batch (dedup) —
      no repeated alerts for the same event.
    * Event bucket (day/hour) and wall-clock detection time recorded
      separately; per-alert ingestion->detection latency measured.
    * Warm-up: buckets with fewer than ``min_samples`` history points are
      skipped (counted), never guessed.
    * Malformed / missing-field rows are skipped, never alerted on.
    * State persists to JSON and recovery is explicit (re-running resumes
      without reprocessing or duplicate alerts).

Outputs (under ``data/analytics/anomalies/``):
    state.json     baselines + processed batches + emitted alert keys
    alerts.csv     appended alert feed consumed by the dashboard
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402
from common.schema import load_canonical, validate  # noqa: E402

LOG = get_logger("anomaly_detection")

DEFAULT_THRESHOLD = 3.5
DEFAULT_MIN_SAMPLES = 5
_MAD_TO_STD = 0.6745  # 1 / 1.4826

ALERT_FIELDS = [
    "road_id",
    "day_of_week",
    "hour",
    "observed_volume",
    "expected_volume",
    "deviation_score",
    "direction",
    "severity",
    "batch_id",
    "event_bucket",
    "arrival_time",
    "detection_time",
    "latency_ms",
]


def classify_severity(abs_z: float, threshold: float) -> str:
    if abs_z >= threshold * 2.3:
        return "severe"
    if abs_z >= threshold * 1.45:
        return "high"
    return "moderate"


@dataclass
class Baseline:
    median: float
    mad: float
    std: float
    n: int

    def scale(self) -> float | None:
        """Robust scale: MAD if non-degenerate, else std, else None."""
        if self.mad > 1e-9:
            return self.mad / _MAD_TO_STD  # express on a std-like scale
        if self.std > 1e-9:
            return self.std
        return None


def compute_baselines(df: pd.DataFrame, min_samples: int = DEFAULT_MIN_SAMPLES) -> dict[str, Baseline]:
    """Median/MAD/std per (road_id, hour) from history. Key = 'road#hour'."""
    baselines: dict[str, Baseline] = {}
    for (road_id, hour), grp in df.groupby(["road_id", "hour"]):
        vals = grp["volume"].to_numpy(dtype=float)
        if len(vals) < min_samples:
            continue
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        std = float(np.std(vals))
        baselines[f"{road_id}#{int(hour)}"] = Baseline(median=median, mad=mad, std=std, n=len(vals))
    return baselines


def robust_z(observed: float, baseline: Baseline) -> float | None:
    scale = baseline.scale()
    if scale is None:
        return None
    return (observed - baseline.median) / scale


class AnomalyDetector:
    """Stateful, incremental detector with JSON persistence."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        state_path: Path | None = None,
        alerts_path: Path | None = None,
        clock=time.time,
    ):
        self.threshold = threshold
        self.min_samples = min_samples
        self.state_path = Path(state_path) if state_path else (config.ANOMALY_DIR / "state.json")
        self.alerts_path = Path(alerts_path) if alerts_path else (config.ANOMALY_DIR / "alerts.csv")
        self.clock = clock
        self.baselines: dict[str, Baseline] = {}
        self.processed_batches: set[str] = set()
        self.emitted_keys: set[str] = set()
        self.stats = {"batches": 0, "rows": 0, "alerts": 0, "warmup_skips": 0, "malformed": 0}

    # ---- persistence -------------------------------------------------------
    def load_state(self) -> None:
        if not self.state_path.exists():
            LOG.info("No prior state at %s (cold start).", self.state_path)
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.baselines = {k: Baseline(**v) for k, v in data.get("baselines", {}).items()}
        self.processed_batches = set(data.get("processed_batches", []))
        self.emitted_keys = set(data.get("emitted_keys", []))
        LOG.info(
            "Recovered state: %d baselines, %d processed batches, %d emitted alerts",
            len(self.baselines), len(self.processed_batches), len(self.emitted_keys),
        )

    def save_state(self) -> None:
        config.ensure_dir(self.state_path.parent)
        payload = {
            "threshold": self.threshold,
            "min_samples": self.min_samples,
            "baselines": {k: asdict(v) for k, v in self.baselines.items()},
            "processed_batches": sorted(self.processed_batches),
            "emitted_keys": sorted(self.emitted_keys),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- baseline ----------------------------------------------------------
    def fit_baseline(self, history: pd.DataFrame) -> None:
        self.baselines = compute_baselines(history, self.min_samples)
        LOG.info("Built %d (road,hour) baselines from %d history rows", len(self.baselines), len(history))

    # ---- batch processing --------------------------------------------------
    def process_batch(self, batch: pd.DataFrame, batch_id: str, arrival_time: float | None = None) -> list[dict]:
        """Score one batch and return newly emitted alerts (deduplicated)."""
        if batch_id in self.processed_batches:
            LOG.info("Batch %s already processed; skipping.", batch_id)
            return []
        arrival_time = arrival_time if arrival_time is not None else self.clock()

        # Validate so malformed rows are skipped (not alerted on).
        report = validate(batch, drop_duplicates=False)
        self.stats["malformed"] += report.n_rejected
        clean = report.clean

        alerts: list[dict] = []
        seen_in_batch: set[str] = set()
        for row in clean.itertuples(index=False):
            self.stats["rows"] += 1
            key_bucket = f"{row.road_id}#{int(row.hour)}"
            baseline = self.baselines.get(key_bucket)
            if baseline is None:
                self.stats["warmup_skips"] += 1
                continue
            z = robust_z(float(row.volume), baseline)
            if z is None or abs(z) < self.threshold:
                continue

            dedup_key = f"{row.road_id}|{int(row.day_of_week)}|{int(row.hour)}|{batch_id}"
            if dedup_key in seen_in_batch or dedup_key in self.emitted_keys:
                continue
            seen_in_batch.add(dedup_key)
            self.emitted_keys.add(dedup_key)

            detection_time = self.clock()
            alert = {
                "road_id": row.road_id,
                "day_of_week": int(row.day_of_week),
                "hour": int(row.hour),
                "observed_volume": round(float(row.volume), 3),
                "expected_volume": round(baseline.median, 3),
                "deviation_score": round(float(z), 3),
                "direction": "high" if z > 0 else "low",
                "severity": classify_severity(abs(z), self.threshold),
                "batch_id": batch_id,
                "event_bucket": f"dow{int(row.day_of_week)}_h{int(row.hour):02d}",
                "arrival_time": round(arrival_time, 6),
                "detection_time": round(detection_time, 6),
                "latency_ms": round((detection_time - arrival_time) * 1000.0, 3),
            }
            alerts.append(alert)

        self.processed_batches.add(batch_id)
        self.stats["batches"] += 1
        self.stats["alerts"] += len(alerts)
        if alerts:
            self._append_alerts(alerts)
        LOG.info("Batch %s: %d rows, %d alerts", batch_id, len(clean), len(alerts))
        return alerts

    def _append_alerts(self, alerts: Iterable[dict]) -> None:
        config.ensure_dir(self.alerts_path.parent)
        new_file = not self.alerts_path.exists()
        with self.alerts_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ALERT_FIELDS)
            if new_file:
                writer.writeheader()
            for alert in alerts:
                writer.writerow(alert)


def _iter_batch_files(stream_dir: Path) -> list[Path]:
    return sorted(p for p in stream_dir.rglob("*.csv") if p.is_file())


def run_directory(
    stream_dir: Path,
    history_csv: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """Single incremental pass over all (unseen) batch files in a directory."""
    detector = AnomalyDetector(threshold=threshold, min_samples=min_samples)
    detector.load_state()
    if not detector.baselines:
        detector.fit_baseline(load_canonical(history_csv))

    for path in _iter_batch_files(stream_dir):
        batch_id = path.stem
        batch = pd.read_csv(path)
        arrival = path.stat().st_mtime
        detector.process_batch(batch, batch_id, arrival_time=arrival)

    detector.save_state()
    LOG.info("Anomaly pass complete: %s", detector.stats)
    return detector.stats


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Incremental traffic anomaly detection")
    parser.add_argument("--stream-dir", default=str(config.STREAM_DIR))
    parser.add_argument("--history", default=str(config.CANONICAL_CSV))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = parser.parse_args()

    stream_dir = Path(args.stream_dir)
    if not stream_dir.exists():
        LOG.error("Stream dir not found: %s. Run the local streamer first.", stream_dir)
        return 1
    history = Path(args.history)
    if not history.exists():
        LOG.error("History CSV not found: %s. Run python/prepare_dataset.py first.", history)
        return 1

    run_directory(stream_dir, history, threshold=args.threshold, min_samples=args.min_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
