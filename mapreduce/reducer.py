#!/usr/bin/env python3
"""Hadoop Streaming reducer (Phase 2).

Consumes the mapper's ``key \\t volume`` stream. Because Hadoop (and the local
``sort`` simulation) guarantees input is sorted by key, all values for one key
arrive contiguously. For each key = ``<road_id>#<bucket>`` the reducer computes
the aggregation and emits one CSV line::

    road_id,time_bucket,count,avg_volume,min_volume,max_volume,sum_volume

A header line is emitted first so the output is directly loadable by pandas /
the dashboard. The aggregation here is associative + commutative on
(count, sum, min, max), so the same logic is safely reusable as a **combiner**
for local pre-aggregation (see run_streaming.sh ``-combiner``).

Self-contained for shipping to the cluster with ``-file reducer.py``.
"""

import sys


def counter(group, name, amount=1):
    sys.stderr.write(f"reporter:counter:{group},{name},{amount}\n")


def emit(key, count, total, vmin, vmax):
    road_id, bucket = key.split("#", 1)
    avg = total / count if count else 0.0
    sys.stdout.write(
        f"{road_id},{bucket},{count},{avg:.4f},{vmin:.4f},{vmax:.4f},{total:.4f}\n"
    )
    counter("TrafficReducer", "groups_emitted")


def main():
    sys.stdout.write("road_id,time_bucket,count,avg_volume,min_volume,max_volume,sum_volume\n")

    cur_key = None
    count = 0
    total = 0.0
    vmin = None
    vmax = None

    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            continue
        try:
            key, value = line.split("\t", 1)
            volume = float(value)
        except ValueError:
            counter("TrafficReducer", "malformed_intermediate")
            continue

        if cur_key is not None and key != cur_key:
            emit(cur_key, count, total, vmin, vmax)
            count, total, vmin, vmax = 0, 0.0, None, None

        cur_key = key
        count += 1
        total += volume
        vmin = volume if vmin is None else min(vmin, volume)
        vmax = volume if vmax is None else max(vmax, volume)

    if cur_key is not None:
        emit(cur_key, count, total, vmin, vmax)


if __name__ == "__main__":
    main()
