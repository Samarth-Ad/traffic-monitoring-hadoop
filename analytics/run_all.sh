#!/bin/bash
# Run every Phase-3 Spark job in sequence. Executed inside the poc-spark container.
set -euo pipefail

cd /app/analytics

export PYSPARK_PYTHON=python3

for job in traffic_volume.py avg_speed.py congestion.py peak_hour.py; do
    echo "=============================================="
    echo "Running $job"
    echo "=============================================="
    spark-submit --master "local[*]" "$job"
done

echo "All Phase-3 analytics jobs complete. Outputs under hdfs:///traffic/analytics/ and /data/analytics/."
