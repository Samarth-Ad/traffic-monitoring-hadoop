#!/bin/bash
# Reproducible, Docker-free run of the whole analytics pipeline on the host.
# Produces every output the dashboard reads, plus timing measurements.
#
#   bash scripts/run_pipeline_local.sh
#
# Steps: prepare -> MapReduce (local streaming sim) -> KNN -> local stream
#        -> anomaly detection (incl. one injected anomaly batch).
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
echo "### Repo: $(pwd)"

echo; echo "===== Phase 1: prepare canonical dataset ====="
time "$PY" python/prepare_dataset.py

echo; echo "===== Phase 2: MapReduce (local Hadoop-Streaming simulation) ====="
time PYTHON="$PY" bash mapreduce/run_local.sh

echo; echo "===== Phase 3: KNN prediction ====="
time "$PY" ml/knn_predict.py --k 7 --weights distance --test-frac 0.2

echo; echo "===== Phase 4: local stream + anomaly detection ====="
rm -rf data/stream data/analytics/anomalies
SINK=local BATCH_SIZE=300 BATCH_INTERVAL_SECONDS=0 MAX_BATCHES=5 "$PY" python/stream_producer.py
# Inject one controlled anomaly batch so the detector has a positive to find.
"$PY" - <<'PYEOF'
from pathlib import Path
import pandas as pd
from common import config
df = pd.read_csv(config.CANONICAL_CSV)
inj = df.head(3).copy()
inj["hour"] = 2; inj["day_of_week"] = 1; inj["volume"] = 99
day_dir = sorted((config.STREAM_DIR).glob("*"))[-1]
inj.to_csv(day_dir / "batch-INJECTED-anomaly.csv", index=False)
print("Injected anomaly batch ->", day_dir / "batch-INJECTED-anomaly.csv")
PYEOF
time "$PY" analytics/anomaly_detection.py --stream-dir data/stream --history data/raw/bangalore_traffic.csv

echo; echo "===== Outputs ====="
ls -1 data/analytics/mapreduce/ data/models/ data/analytics/anomalies/ 2>/dev/null || true
echo "Done."
