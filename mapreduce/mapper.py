#!/usr/bin/env python3
"""Hadoop Streaming mapper (Phase 2).

Reads canonical traffic CSV rows from stdin and emits, for each valid record,
a tab-separated ``key \\t volume`` pair where::

    key = "<road_id>#<time_bucket>"

The time bucket is ``hour`` by default, overridable with the env var
``MR_TIME_BUCKET`` (``hour`` or ``day_of_week``). Hadoop's shuffle-and-sort
then groups all volumes sharing a key before the reducer sees them.

Malformed / out-of-range rows are skipped and counted via Hadoop Streaming
counters written to stderr (``reporter:counter:...``) so nothing is silently
dropped — the counts appear in the job report.

This script is intentionally self-contained (no project imports) so it can be
shipped to the cluster with ``-file mapper.py`` and run on any node.
"""

import os
import sys

# Canonical CSV column order (see common/schema.py).
COLS = ["road_id", "area", "road", "lat", "lon", "day_of_week", "hour", "volume"]
IDX = {name: i for i, name in enumerate(COLS)}

TIME_BUCKET = os.environ.get("MR_TIME_BUCKET", "hour")
if TIME_BUCKET not in ("hour", "day_of_week"):
    TIME_BUCKET = "hour"


def counter(group, name, amount=1):
    sys.stderr.write(f"reporter:counter:{group},{name},{amount}\n")


def main():
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split(",")
        # Skip the header row if present.
        if fields[0] == "road_id":
            continue
        if len(fields) != len(COLS):
            counter("TrafficMapper", "malformed_field_count")
            continue
        try:
            road_id = fields[IDX["road_id"]]
            hour = int(fields[IDX["hour"]])
            dow = int(fields[IDX["day_of_week"]])
            volume = float(fields[IDX["volume"]])
        except (ValueError, IndexError):
            counter("TrafficMapper", "unparseable_value")
            continue
        if not (0 <= hour <= 23) or not (0 <= dow <= 6) or volume < 0:
            counter("TrafficMapper", "out_of_range")
            continue

        bucket = hour if TIME_BUCKET == "hour" else dow
        counter("TrafficMapper", "valid_records")
        sys.stdout.write(f"{road_id}#{bucket}\t{volume}\n")


if __name__ == "__main__":
    main()
