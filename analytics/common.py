"""Shared Spark session + schema helpers for all Phase-3 analytics jobs."""

from __future__ import annotations

import os
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

HDFS_INPUT_DEFAULT = "hdfs://hadoop:9000/traffic/input/*.csv"
HDFS_STREAM_DEFAULT = "hdfs://hadoop:9000/traffic/stream/*/*.csv"
HDFS_ANALYTICS_ROOT = "hdfs://hadoop:9000/traffic/analytics"
LOCAL_ANALYTICS_ROOT = "/data/analytics"


# Canonical column names used across all jobs. Values are candidate source names
# we try to find in the raw CSV (case-insensitive).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("Date", "date", "Timestamp", "timestamp", "DateTime"),
    "area": ("Area Name", "Area", "area", "Neighborhood"),
    "road": ("Road/Intersection Name", "Road", "road", "Road Name"),
    "volume": ("Traffic Volume", "TrafficVolume", "volume", "Vehicle Count"),
    "speed": ("Average Speed", "AverageSpeed", "Avg Speed", "speed"),
    "congestion": ("Congestion Level", "CongestionLevel", "congestion"),
    "tti": ("Travel Time Index", "TravelTimeIndex"),
    "capacity_util": ("Road Capacity Utilization", "RoadCapacityUtilization"),
}


def make_spark(app_name: str) -> SparkSession:
    """Return a SparkSession suitable for the Phase-3 batch jobs."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def read_traffic_csv(spark: SparkSession, path: str | None = None) -> DataFrame:
    """Load the raw Bangalore traffic CSV with inferred schema."""
    path = path or os.environ.get("TRAFFIC_INPUT", HDFS_INPUT_DEFAULT)
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(path)
    )


def resolve_columns(df: DataFrame) -> dict[str, str]:
    """Map canonical names -> actual column names present in ``df``."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            actual = lower_to_actual.get(cand.lower())
            if actual is not None:
                resolved[canonical] = actual
                break
    return resolved


def normalize(df: DataFrame) -> DataFrame:
    """Rename resolved columns to canonical lowercase names and parse the date."""
    cols = resolve_columns(df)
    renamed = df
    for canonical, actual in cols.items():
        if canonical != actual:
            renamed = renamed.withColumnRenamed(actual, canonical)

    if "date" in cols:
        renamed = renamed.withColumn("date_ts", F.to_timestamp("date"))
        renamed = renamed.withColumn("day_of_week", F.date_format("date_ts", "EEEE"))
        renamed = renamed.withColumn("hour", F.hour("date_ts"))
    return renamed


def write_results(df: DataFrame, name: str, partitions: Iterable[str] = ()) -> None:
    """Persist ``df`` to both HDFS and the shared local volume (for the dashboard)."""
    writer = df.coalesce(1).write.mode("overwrite")
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.parquet(f"{HDFS_ANALYTICS_ROOT}/{name}")
    writer.parquet(f"{LOCAL_ANALYTICS_ROOT}/{name}")
