#!/bin/bash
# Genuine Hadoop MapReduce via Hadoop Streaming, run INSIDE the poc-hadoop
# container against HDFS.
#
#   docker cp mapreduce poc-hadoop:/opt/mapreduce        # once
#   docker exec -it poc-hadoop bash /opt/mapreduce/run_streaming.sh
#
# It reads the canonical CSV from an HDFS input path and writes aggregated
# output to an HDFS output path. The output directory is checked first so an
# existing result is never silently overwritten (Hadoop itself also refuses).
set -euo pipefail

INPUT="${HDFS_INPUT:-/traffic/input}"
OUTPUT="${HDFS_MAPREDUCE_OUTPUT:-/traffic/mapreduce/output}"
BUCKET="${MR_TIME_BUCKET:-hour}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Locate the streaming jar shipped with Hadoop.
STREAM_JAR="$(find "${HADOOP_HOME:-/opt/hadoop}" -name 'hadoop-streaming-*.jar' 2>/dev/null | head -n1)"
if [[ -z "$STREAM_JAR" ]]; then
  echo "ERROR: hadoop-streaming jar not found under ${HADOOP_HOME:-/opt/hadoop}" >&2
  exit 2
fi

# Refuse to clobber an existing output directory.
if hdfs dfs -test -d "$OUTPUT" 2>/dev/null; then
  echo "ERROR: HDFS output '$OUTPUT' already exists. Remove it or set HDFS_MAPREDUCE_OUTPUT." >&2
  echo "       hdfs dfs -rm -r $OUTPUT" >&2
  exit 3
fi

echo "Streaming jar : $STREAM_JAR"
echo "Input  (HDFS) : $INPUT"
echo "Output (HDFS) : $OUTPUT"
echo "Time bucket   : $BUCKET"

# Design notes:
#  * No -combiner: a Streaming combiner must emit the SAME key\tvalue format
#    its reducer consumes, but reducer.py emits final CSV. Reusing it as a
#    combiner would feed CSV back into the reducer and corrupt results, so it
#    is deliberately omitted. (A correct combiner would need its own partial
#    "key\tcount:sum:min:max" format — unnecessary at this data size.)
#  * No custom -partitioner: the default hash partitioner routes every record
#    of a given key ("road_id#bucket") to a single reducer, which is exactly
#    the grouping we need. A custom partitioner would add no value here.
hadoop jar "$STREAM_JAR" \
  -D mapreduce.job.name="traffic-volume-by-road-$BUCKET" \
  -files "$HERE/mapper.py,$HERE/reducer.py" \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -cmdenv "MR_TIME_BUCKET=$BUCKET" \
  -input "$INPUT" \
  -output "$OUTPUT"

echo "===== job complete; output listing ====="
hdfs dfs -ls "$OUTPUT"
echo "===== sample output ====="
hdfs dfs -cat "$OUTPUT/part-00000" | head -20
