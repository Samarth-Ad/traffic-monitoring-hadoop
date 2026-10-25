"""Canonical traffic schema, loaders and validation (Phase 1).

The platform has two possible data sources:

1. The real dataset shipped in this repo:
   ``data/source/bangalore_traffic_raw.txt`` — a **headerless** CSV with the
   positional schema ``lat, lon, day_of_week, hour, volume`` (14 coordinate
   locations in Bangalore, no road names, no absolute timestamp).

2. The optional Kaggle "Banglore City Traffic" CSV with *named* columns
   (Area Name, Road, Traffic Volume, Average Speed, ...), which requires a
   Kaggle account to download and is therefore not committed.

Both are normalised to one **canonical schema** so every downstream component
(MapReduce, KNN, anomaly detection, dashboard) shares a single data contract:

    road_id       str    stable id derived from (lat, lon), e.g. "loc_00"
    area          str    coarse area label (single city here: "Bangalore")
    road          str    human-ish road label (coordinate-derived)
    lat           float  latitude  (Bangalore bbox ~12.7..13.2)
    lon           float  longitude (Bangalore bbox ~77.4..77.9)
    day_of_week   int    0..6  (0 = first observed bucket)
    hour          int    0..23
    volume        float  traffic volume / intensity (peaks at rush hours)

Design notes / honest limitations:

* The 5th raw column is interpreted as **traffic volume / intensity** because
  its mean rises at the morning and evening commute peaks (see docs). It is NOT
  free-flow speed (which would dip at peaks).
* The raw data has **no absolute timestamp** — only cyclic day-of-week and
  hour. "Time" for streaming / chronological evaluation therefore means
  *replay order* (the order rows appear in the file), which is explicit and
  reproducible. This is documented wherever it matters.
* Road **names** are unavailable; ``road_id`` / ``road`` are coordinate-derived
  and stable, never fabricated names.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = (
    "road_id",
    "area",
    "road",
    "lat",
    "lon",
    "day_of_week",
    "hour",
    "volume",
)

# Positional schema of the raw headerless source file.
POSITIONAL_RAW_COLUMNS: tuple[str, ...] = ("lat", "lon", "day_of_week", "hour", "volume")

# Case-insensitive alias map: canonical name -> candidate source names.
# Covers the headerless source, the Kaggle named CSV, and common real-world
# traffic datasets (UCI Metro, Dublin SCATS, UK DfT, generic sensor feeds).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "road_id": ("road_id", "RoadID", "segment_id", "sensor_id", "detector_id", "location_id", "id"),
    "area": ("area", "Area Name", "Area", "Neighborhood", "city", "region", "zone"),
    "road": ("road", "Road/Intersection Name", "Road", "Road Name", "segment", "junction",
             "location", "street", "name", "highway"),
    "lat": ("lat", "latitude", "Latitude", "y"),
    "lon": ("lon", "lng", "longitude", "Longitude", "x"),
    "day_of_week": ("day_of_week", "dow", "DayOfWeek", "day"),
    "hour": ("hour", "Hour", "hour_of_day"),
    "volume": ("volume", "Traffic Volume", "TrafficVolume", "traffic_volume", "Vehicle Count",
               "vehicle_count", "count", "Count", "flow", "Flow", "total_flow", "vehicles", "intensity"),
    # A timestamp lets us derive hour + day_of_week when they are not given directly.
    "timestamp": ("timestamp", "Timestamp", "datetime", "date_time", "DateTime", "date", "Date",
                  "time", "Date Time"),
}

# Bangalore bounding box — used only to *label* the area of in-region data, no
# longer to reject rows (the tool accepts any city's data for testing).
BANGALORE_BBOX = {"lat_min": 12.6, "lat_max": 13.3, "lon_min": 77.2, "lon_max": 78.0}
# Global coordinate validity + a generous volume bound.
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0
VOLUME_MIN, VOLUME_MAX = 0.0, 1_000_000.0


@dataclass
class ValidationReport:
    """Outcome of validating a raw dataframe. Rejected rows keep a reason."""

    clean: pd.DataFrame
    rejected: pd.DataFrame
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n_input(self) -> int:
        return len(self.clean) + len(self.rejected)

    @property
    def n_clean(self) -> int:
        return len(self.clean)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def summary(self) -> str:
        parts = [f"input={self.n_input}", f"clean={self.n_clean}", f"rejected={self.n_rejected}"]
        for reason, count in sorted(self.reasons.items()):
            parts.append(f"{reason}={count}")
        return "  ".join(parts)


def road_id_from_coords(lat: float, lon: float) -> str:
    """Deterministic, stable road id from rounded coordinates.

    Rounds to 6 dp (the precision present in the source) so identical
    locations always map to the same id regardless of row order.
    """
    return f"{round(float(lat), 6):.6f},{round(float(lon), 6):.6f}"


def _area_label(df: pd.DataFrame) -> pd.Series:
    """Label area as 'Bangalore' for in-bbox coords, else 'unknown'."""
    if "lat" in df.columns and "lon" in df.columns and df["lat"].notna().any():
        in_box = df["lat"].between(BANGALORE_BBOX["lat_min"], BANGALORE_BBOX["lat_max"]) & df["lon"].between(
            BANGALORE_BBOX["lon_min"], BANGALORE_BBOX["lon_max"]
        )
        return in_box.map({True: "Bangalore", False: "unknown"})
    return pd.Series(["unknown"] * len(df), index=df.index)


def assign_road_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable ``road_id``/``road``/``area`` labels (``loc_00`` ...).

    The grouping key is taken from the best available location signal — an
    explicit id, a road/segment name, an area, or coordinates — falling back to
    a single site when a dataset has no location at all. Ordering is by first
    appearance so ids are reproducible across runs on the same input.
    """
    df = df.copy()
    if "road_id" in df.columns and df["road_id"].notna().all():
        key = df["road_id"].astype(str)
    elif "road" in df.columns and df["road"].notna().any():
        key = df["road"].astype(str)
    elif "area" in df.columns and df["area"].notna().any():
        key = df["area"].astype(str)
    elif "lat" in df.columns and "lon" in df.columns and df["lat"].notna().any():
        key = df.apply(
            lambda r: road_id_from_coords(r["lat"], r["lon"]) if pd.notna(r["lat"]) else "na", axis=1
        )
    else:
        key = pd.Series(["site_00"] * len(df), index=df.index)

    ordered_unique = list(dict.fromkeys(key))  # preserves first-seen order
    mapping = {k: f"loc_{i:02d}" for i, k in enumerate(ordered_unique)}
    df["road_id"] = key.map(mapping)

    if "road" not in df.columns or df["road"].isna().all():
        df["road"] = df["road_id"].map(lambda rid: f"road_{rid}")
    if "area" not in df.columns or df["area"].isna().all():
        df["area"] = _area_label(df)
    for coord in ("lat", "lon"):
        if coord not in df.columns:
            df[coord] = pd.NA
    return df


def load_raw_positional(path: str | Path) -> pd.DataFrame:
    """Load the headerless source file into a dataframe with raw column names."""
    df = pd.read_csv(path, header=None, names=list(POSITIONAL_RAW_COLUMNS))
    return df


def _resolve_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a *named-column* dataframe to canonical names via aliases."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            actual = lower_to_actual.get(cand.lower())
            if actual is not None:
                rename[actual] = canonical
                break
    return df.rename(columns=rename)


def load_any_text(text: str) -> pd.DataFrame:
    """Load a CSV given as a string (e.g. an uploaded file's contents).

    Detection: if the first line parses as 5 all-numeric fields we treat it as
    the headerless positional source; otherwise we assume a header row and
    resolve column aliases.
    """
    lines = text.splitlines()
    first = lines[0].strip() if lines else ""
    fields = first.split(",")
    looks_positional = len(fields) == len(POSITIONAL_RAW_COLUMNS) and all(
        _is_number(f) for f in fields
    )
    if looks_positional:
        return pd.read_csv(io.StringIO(text), header=None, names=list(POSITIONAL_RAW_COLUMNS))
    return _resolve_named_columns(pd.read_csv(io.StringIO(text)))


def load_any(path: str | Path) -> pd.DataFrame:
    """Load either the positional raw file or a named-column CSV from a path."""
    return load_any_text(Path(path).read_text(encoding="utf-8"))


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate(df: pd.DataFrame, *, drop_duplicates: bool = True) -> ValidationReport:
    """Validate and normalise a dataframe to the canonical schema.

    Never silently discards rows: every rejected row is returned with a
    ``reject_reason`` column and the reasons are counted in the report.
    """
    reasons: dict[str, int] = {}
    work = df.copy()

    # Derive hour / day_of_week from a timestamp column when they are absent.
    if ("hour" not in work.columns or "day_of_week" not in work.columns) and "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], errors="coerce")
        if "hour" not in work.columns:
            work["hour"] = ts.dt.hour
        if "day_of_week" not in work.columns:
            work["day_of_week"] = ts.dt.dayofweek  # Monday=0 .. Sunday=6

    # hour, day_of_week and volume are required (lat/lon are optional).
    required = ["hour", "day_of_week", "volume"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns {missing} and no 'timestamp' column "
            "was present to derive hour/day_of_week."
        )

    # Coordinates count as "present" only if the columns exist AND carry at
    # least one real value — so canonical data with empty lat/lon (datasets
    # that never had coordinates) is not wrongly rejected as coords_missing.
    has_coords = (
        "lat" in work.columns
        and "lon" in work.columns
        and (work["lat"].notna().any() or work["lon"].notna().any())
    )
    numeric = required + (["lat", "lon"] if has_coords else [])
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rejected_frames: list[pd.DataFrame] = []

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal work
        if mask.any():
            bad = work[mask].copy()
            bad["reject_reason"] = reason
            rejected_frames.append(bad)
            reasons[reason] = reasons.get(reason, 0) + int(mask.sum())
            work = work[~mask]

    reject(work[required].isna().any(axis=1), "non_numeric_or_missing")
    reject(~work["day_of_week"].between(0, 6), "day_of_week_out_of_range")
    reject(~work["hour"].between(0, 23), "hour_out_of_range")
    reject(~work["volume"].between(VOLUME_MIN, VOLUME_MAX, inclusive="right"), "volume_out_of_range")
    if has_coords:
        reject(work["lat"].isna() | work["lon"].isna(), "coords_missing")
        reject(
            ~(work["lat"].between(LAT_MIN, LAT_MAX) & work["lon"].between(LON_MIN, LON_MAX)),
            "coords_out_of_range",
        )

    # Normalise dtypes on surviving rows.
    if not work.empty:
        work["day_of_week"] = work["day_of_week"].astype(int)
        work["hour"] = work["hour"].astype(int)
        work["volume"] = work["volume"].astype(float)
        work = assign_road_ids(work)

    if drop_duplicates and not work.empty:
        # Timestamped data: two different timestamps are distinct observations
        # even if (day, hour, volume) coincide, so keep the timestamp in the key.
        if "timestamp" in work.columns:
            subset = [c for c in ("road_id", "timestamp", "volume") if c in work.columns]
        else:
            subset = ["road_id", "day_of_week", "hour", "volume"]
        before = len(work)
        work = work.drop_duplicates(subset=subset)
        dup = before - len(work)
        if dup:
            reasons["duplicate"] = dup

    clean = (
        work[list(CANONICAL_COLUMNS)].reset_index(drop=True)
        if not work.empty
        else pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    )
    rejected = (
        pd.concat(rejected_frames, ignore_index=True)
        if rejected_frames
        else pd.DataFrame(columns=list(df.columns) + ["reject_reason"])
    )
    return ValidationReport(clean=clean, rejected=rejected, reasons=reasons)


def load_canonical(path: str | Path, *, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load + validate in one call, returning only the clean canonical frame."""
    report = validate(load_any(path), drop_duplicates=drop_duplicates)
    return report.clean


def iter_canonical_records(paths: Iterable[str | Path]) -> Iterable[dict]:
    """Yield clean canonical records (dicts) from one or more files."""
    for path in paths:
        clean = load_canonical(path)
        for record in clean.to_dict(orient="records"):
            yield record
