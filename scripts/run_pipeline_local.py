"""Reproducible, Docker-free, bash-free run of the whole analytics pipeline.

Pure Python so it runs anywhere Python does (Windows PowerShell included):

    python scripts/run_pipeline_local.py

Steps: prepare -> MapReduce (local streaming sim) -> KNN -> local stream
       -> anomaly detection (incl. one injected anomaly batch).
Produces every output the dashboard reads, plus timing measurements.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

LOG = get_logger("pipeline")


def _step(title: str):
    LOG.info("=" * 60)
    LOG.info(title)
    LOG.info("=" * 60)
    return time.perf_counter()


def _done(t0: float):
    LOG.info("  (%.2fs)", time.perf_counter() - t0)


def mapreduce_local(input_csv: Path, output_csv: Path) -> int:
    """Run mapper.py | sort | reducer.py entirely in Python (no shell)."""
    mapper = ROOT / "mapreduce" / "mapper.py"
    reducer = ROOT / "mapreduce" / "reducer.py"
    text = input_csv.read_text(encoding="utf-8")
    mapped = subprocess.run(
        [sys.executable, str(mapper)], input=text, capture_output=True, text=True, check=True
    )
    sorted_lines = "\n".join(sorted(mapped.stdout.splitlines())) + "\n"
    reduced = subprocess.run(
        [sys.executable, str(reducer)], input=sorted_lines, capture_output=True, text=True, check=True
    )
    config.ensure_dir(output_csv.parent)
    output_csv.write_text(reduced.stdout, encoding="utf-8")
    return max(0, len(reduced.stdout.splitlines()) - 1)


def main() -> int:
    import pandas as pd

    # Phase 1
    t = _step("Phase 1: prepare canonical dataset")
    from python.prepare_dataset import main as prepare_main
    prepare_main()
    _done(t)

    # Phase 2
    t = _step("Phase 2: MapReduce (local Hadoop-Streaming simulation)")
    out = config.MAPREDUCE_DIR / "volume_by_road_hour.csv"
    groups = mapreduce_local(config.CANONICAL_CSV, out)
    LOG.info("  %d groups -> %s", groups, out)
    _done(t)

    # Phase 3
    t = _step("Phase 3: KNN prediction")
    import ml.knn_predict as knn
    sys.argv = ["knn", "--k", "7", "--weights", "distance", "--test-frac", "0.2"]
    knn.main()
    _done(t)

    # Phase 4
    t = _step("Phase 4: local stream + anomaly detection")
    for d in (config.STREAM_DIR, config.ANOMALY_DIR):
        if d.exists():
            shutil.rmtree(d)
    import os
    os.environ.update(
        SINK="local", BATCH_SIZE="300", BATCH_INTERVAL_SECONDS="0", MAX_BATCHES="5",
        DATASET_CSV=str(config.CANONICAL_CSV),
    )
    from python.stream_producer import run as stream_run
    stream_run()

    # inject one controlled anomaly batch
    df = pd.read_csv(config.CANONICAL_CSV).head(3).copy()
    df["hour"], df["day_of_week"], df["volume"] = 2, 1, 99
    day_dir = sorted(config.STREAM_DIR.glob("*"))[-1]
    inj = day_dir / "batch-INJECTED-anomaly.csv"
    df.to_csv(inj, index=False)
    LOG.info("  injected anomaly batch -> %s", inj)

    from analytics.anomaly_detection import run_directory
    stats = run_directory(config.STREAM_DIR, config.CANONICAL_CSV)
    _done(t)

    _step("Outputs")
    for d in (config.MAPREDUCE_DIR, config.MODELS_DIR, config.ANOMALY_DIR):
        if d.exists():
            for f in sorted(d.iterdir()):
                LOG.info("  %s", f)
    LOG.info("Anomaly stats: %s", stats)
    LOG.info("Done. View results with the dashboard (see README section 6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
