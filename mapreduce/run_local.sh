#!/bin/bash
# Local Hadoop-Streaming SIMULATION (no cluster needed).
#
# Runs the exact same mapper -> sort -> reducer pipeline that Hadoop Streaming
# executes, so the output is byte-for-byte what the cluster job produces. Use
# this for development and for the automated correctness test.
#
# Usage:
#   bash mapreduce/run_local.sh [INPUT_CSV] [OUTPUT_CSV]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:-$HERE/../data/raw/bangalore_traffic.csv}"
OUTPUT="${2:-$HERE/../data/analytics/mapreduce/volume_by_road_hour.csv}"
PY="${PYTHON:-python}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Input : $INPUT"
echo "Output: $OUTPUT"
echo "Bucket: ${MR_TIME_BUCKET:-hour}"

"$PY" "$HERE/mapper.py" < "$INPUT" | LC_ALL=C sort | "$PY" "$HERE/reducer.py" > "$OUTPUT"

echo "Done. $(( $(wc -l < "$OUTPUT") - 1 )) groups written."
