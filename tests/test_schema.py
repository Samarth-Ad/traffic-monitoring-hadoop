"""Phase 1 tests — canonical schema validation."""

import unittest
from pathlib import Path

import pandas as pd

from common.schema import CANONICAL_COLUMNS, load_any, load_canonical, validate

FIXTURE = Path(__file__).parent / "fixtures" / "mini_raw.txt"


class SchemaTests(unittest.TestCase):
    def test_load_positional_fixture(self):
        df = load_any(FIXTURE)
        self.assertEqual(list(df.columns), ["lat", "lon", "day_of_week", "hour", "volume"])
        self.assertEqual(len(df), 168)

    def test_valid_rows_are_canonicalised(self):
        clean = load_canonical(FIXTURE)
        self.assertEqual(list(clean.columns), list(CANONICAL_COLUMNS))
        self.assertEqual(clean["road_id"].nunique(), 3)
        self.assertTrue(clean["hour"].between(0, 23).all())
        self.assertTrue(clean["day_of_week"].between(0, 6).all())

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"lat": [12.9], "lon": [77.6], "hour": [8], "volume": [50]})
        with self.assertRaises(ValueError):
            validate(df)  # day_of_week missing

    def test_malformed_and_out_of_range_rejected_with_reasons(self):
        df = pd.DataFrame(
            {
                "lat": [12.9, 12.9, 12.9, 99.0],       # last: lat outside -90..90
                "lon": [77.6, 77.6, 77.6, 77.6],
                "day_of_week": [0, 9, 0, 0],            # second: dow out of range
                "hour": [8, 8, 30, 8],                  # third: hour out of range
                "volume": [50, 50, 50, 50],
            }
        )
        report = validate(df)
        self.assertEqual(report.n_clean, 1)
        self.assertEqual(report.n_rejected, 3)
        self.assertIn("day_of_week_out_of_range", report.reasons)
        self.assertIn("hour_out_of_range", report.reasons)
        self.assertIn("coords_out_of_range", report.reasons)

    def test_derive_time_from_timestamp(self):
        # Named-column real-world style: datetime + volume, no hour/day columns.
        df = pd.DataFrame(
            {
                "timestamp": ["2020-01-06 08:00:00", "2020-01-07 22:00:00"],  # Mon 8h, Tue 22h
                "volume": [1200, 300],
                "road": ["I-94", "I-94"],
            }
        )
        report = validate(df)
        self.assertEqual(report.n_clean, 2)
        self.assertEqual(sorted(report.clean["hour"].tolist()), [8, 22])
        self.assertEqual(sorted(report.clean["day_of_week"].tolist()), [0, 1])

    def test_optional_latlon_uses_road_for_id(self):
        # No coordinates at all -> road_id derived from the road/segment column.
        df = pd.DataFrame(
            {
                "road": ["A1", "A1", "B2"],
                "day_of_week": [0, 1, 2],
                "hour": [8, 9, 10],
                "volume": [100, 110, 50],
            }
        )
        report = validate(df)
        self.assertEqual(report.n_clean, 3)
        self.assertEqual(report.clean["road_id"].nunique(), 2)
        self.assertTrue(report.clean["lat"].isna().all())

    def test_non_numeric_rejected(self):
        # Non-numeric value in a REQUIRED field (volume) is rejected.
        df = pd.DataFrame(
            {"lat": [12.9], "lon": [77.6], "day_of_week": [0], "hour": [8], "volume": ["x"]}
        )
        report = validate(df)
        self.assertEqual(report.n_clean, 0)
        self.assertEqual(report.reasons.get("non_numeric_or_missing"), 1)

    def test_empty_input(self):
        df = pd.DataFrame(columns=["lat", "lon", "day_of_week", "hour", "volume"])
        report = validate(df)
        self.assertEqual(report.n_clean, 0)
        self.assertEqual(report.n_rejected, 0)

    def test_duplicates_counted(self):
        df = pd.DataFrame(
            {
                "lat": [12.9, 12.9],
                "lon": [77.6, 77.6],
                "day_of_week": [0, 0],
                "hour": [8, 8],
                "volume": [50, 50],
            }
        )
        report = validate(df, drop_duplicates=True)
        self.assertEqual(report.n_clean, 1)
        self.assertEqual(report.reasons.get("duplicate"), 1)

    def test_road_ids_stable(self):
        clean1 = load_canonical(FIXTURE)
        clean2 = load_canonical(FIXTURE)
        self.assertEqual(
            clean1.drop_duplicates("road_id")[["lat", "road_id"]].values.tolist(),
            clean2.drop_duplicates("road_id")[["lat", "road_id"]].values.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
