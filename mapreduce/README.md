# Phase 2 — Hadoop MapReduce (Hadoop Streaming, Python)

A genuine MapReduce job that aggregates traffic **volume by road × time bucket**
using Hadoop's shuffle-and-sort to group records by key.

## Why Hadoop Streaming (not native Java)
The platform is already Python + Docker. Hadoop Streaming lets the mapper and
reducer be plain Python programs shipped to the cluster with `-files`, so there
is no second (Java/Maven) build system and the exact same scripts are testable
on the host. The abandoned empty Java scaffold is intentionally **not** revived.

## Files
| File | Role |
|------|------|
| `mapper.py`  | stdin → `key \t volume`, key = `road_id#<hour>`; counts malformed rows via Hadoop counters |
| `reducer.py` | grouped stream → CSV `road_id,time_bucket,count,avg,min,max,sum` |
| `run_local.sh`     | local `map \| sort \| reduce` simulation (no cluster) |
| `run_streaming.sh` | real job inside `poc-hadoop` against HDFS |

Time bucket: `hour` (default) or `day_of_week` via `MR_TIME_BUCKET`.

## Run locally (no Docker)
```bash
bash mapreduce/run_local.sh            # -> data/analytics/mapreduce/volume_by_road_hour.csv
```

## Run on the cluster
```bash
# load input into HDFS once
docker exec -it poc-hadoop hdfs dfs -mkdir -p /traffic/input
docker cp data/raw/bangalore_traffic.csv poc-hadoop:/tmp/bt.csv
docker exec -it poc-hadoop hdfs dfs -put -f /tmp/bt.csv /traffic/input/

docker cp mapreduce poc-hadoop:/opt/mapreduce
docker exec -it poc-hadoop bash /opt/mapreduce/run_streaming.sh
```

## Design decisions
* **No combiner** — a Streaming combiner must emit the mapper's `key\tvalue`
  format, but the reducer emits final CSV; reusing it would corrupt results, so
  it is deliberately omitted.
* **No custom partitioner** — the default hash partitioner already routes all
  records of one key to a single reducer, which is exactly the grouping needed.
* **Overwrite-safe** — `run_streaming.sh` refuses to clobber an existing HDFS
  output directory.

Correctness is verified against an independent pandas aggregation in
`tests/test_mapreduce.py`.
