"""Phase 2 tests — Hadoop Streaming mapper/reducer correctness.

Runs the real mapper.py and reducer.py as subprocesses (exactly as Hadoop
Streaming invokes them), with an in-Python sort between them to mimic the
shuffle, and compares against an independent pandas reference.
"""

import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path

import pandas as pd

from common.schema import load_canonical

ROOT = Path(__file__).resolve().parent.parent
MAPPER = ROOT / "mapreduce" / "mapper.py"
REDUCER = ROOT / "mapreduce" / "reducer.py"
FIXTURE = ROOT / "tests" / "fixtures" / "mini_raw.txt"


def run_streaming(canonical_csv_text: str) -> pd.DataFrame:
    mapped = subprocess.run(
        [sys.executable, str(MAPPER)],
        input=canonical_csv_text, capture_output=True, text=True, check=True,
    )
    sorted_lines = "\n".join(sorted(mapped.stdout.splitlines())) + "\n"
    reduced = subprocess.run(
        [sys.executable, str(REDUCER)],
        input=sorted_lines, capture_output=True, text=True, check=True,
    )
    return pd.read_csv(StringIO(reduced.stdout))


class MapReduceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = load_canonical(FIXTURE)
        cls.csv_text = cls.canonical.to_csv(index=False)
        cls.out = run_streaming(cls.csv_text)

    def test_matches_pandas_reference(self):
        ref = (
            self.canonical.groupby(["road_id", "hour"])["volume"]
            .agg(count="count", avg_volume="mean", min_volume="min",
                 max_volume="max", sum_volume="sum")
            .reset_index()
        )
        mr = self.out.rename(columns={"time_bucket": "hour"})
        merged = ref.merge(mr, on=["road_id", "hour"], suffixes=("_ref", "_mr"))
        self.assertEqual(len(ref), len(mr))
        self.assertEqual(len(merged), len(ref))
        for col in ["count", "min_volume", "max_volume", "sum_volume"]:
            self.assertTrue((merged[f"{col}_ref"] - merged[f"{col}_mr"]).abs().max() < 1e-6)
        # avg is written to 4 dp by the reducer
        self.assertTrue((merged["avg_volume_ref"] - merged["avg_volume_mr"]).abs().max() < 5e-4)

    def test_malformed_rows_skipped_not_crash(self):
        bad = self.csv_text + "garbage,row,with,wrong,count\nloc_00,Bangalore,road,12.9,77.6,x,y,z\n"
        out = run_streaming(bad)
        # Output still parses and group count is unchanged (bad rows dropped).
        self.assertEqual(len(out), len(self.out))

    def test_empty_input(self):
        header = ",".join(self.canonical.columns) + "\n"
        out = run_streaming(header)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
