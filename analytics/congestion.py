"""Phase 3.3 — Threshold-based congestion flagging + ranking."""

from __future__ import annotations

import os

from pyspark.sql import functions as F

from common import make_spark, normalize, read_traffic_csv, write_results

CONGESTION_THRESHOLD = float(os.environ.get("CONGESTION_THRESHOLD", "70.0"))


def main() -> None:
    spark = make_spark("Congestion")
    df = normalize(read_traffic_csv(spark))

    flagged = df.withColumn(
        "is_congested",
        F.coalesce(F.col("congestion"), F.lit(0.0)) >= F.lit(CONGESTION_THRESHOLD),
    )

    by_road = (
        flagged.groupBy("area", "road")
        .agg(
            F.avg("congestion").alias("avg_congestion"),
            F.max("congestion").alias("peak_congestion"),
            F.avg(F.col("is_congested").cast("int")).alias("congested_fraction"),
            F.count("*").alias("samples"),
        )
        .orderBy(F.desc("avg_congestion"))
    )
    write_results(by_road, "congestion_by_road")

    hourly = (
        flagged.groupBy("area", "road", "hour")
        .agg(F.avg("congestion").alias("avg_congestion"))
    )
    write_results(hourly, "congestion_by_hour")

    spark.stop()


if __name__ == "__main__":
    main()
