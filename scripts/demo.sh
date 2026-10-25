#!/bin/bash
# Viva / demonstration script. Prints each step; run interactively.
#   bash scripts/demo.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

echo "A. Prepare the canonical dataset"
python python/prepare_dataset.py

echo; echo "B. Hadoop MapReduce aggregate (local simulation)"
bash mapreduce/run_local.sh
echo "   -> data/analytics/mapreduce/volume_by_road_hour.csv"

echo; echo "C. KNN prediction + evaluation"
python ml/knn_predict.py --k 7 --weights distance --test-frac 0.2
cat data/models/knn_metrics.json

echo; echo "D. Stream batches locally and introduce a controlled anomaly"
rm -rf data/stream data/analytics/anomalies
SINK=local BATCH_SIZE=300 BATCH_INTERVAL_SECONDS=0 MAX_BATCHES=5 python python/stream_producer.py
python - <<'PYEOF'
import pandas as pd; from common import config
df = pd.read_csv(config.CANONICAL_CSV).head(3).copy()
df["hour"], df["day_of_week"], df["volume"] = 2, 1, 99
d = sorted(config.STREAM_DIR.glob("*"))[-1]
df.to_csv(d/"batch-INJECTED-anomaly.csv", index=False)
print("Injected anomaly batch written.")
PYEOF

echo; echo "E. Run anomaly detection and show the alert for the injected event"
python analytics/anomaly_detection.py --stream-dir data/stream
echo "--- injected-anomaly alert ---"
grep INJECTED data/analytics/anomalies/alerts.csv || true

echo; echo "F. Detection latency (ms) summary"
python - <<'PYEOF'
import pandas as pd
a = pd.read_csv("data/analytics/anomalies/alerts.csv")
print("alerts:", len(a), "median latency ms:", round(a.latency_ms.median(),1))
PYEOF

echo; echo "G. Open the dashboard:  docker compose up -d dashboard  ->  http://localhost:8501"
