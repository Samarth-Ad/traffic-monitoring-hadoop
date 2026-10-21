"""Phase 3.4 — Temporal clustering: identify recurring peak hours per road.

If the raw dataset has no hour granularity, we fall back to the strongest
day-of-week signal — the method is the same, only the time axis differs.
"""

from __future__ import annotations

from pyspark.sql import Window
from pyspark.sql import functions as F

from common import make_spark, normalize, read_traffic_csv, write_results


def main() -> None:
    spark = make_spark("PeakHour")
    df = normalize(read_traffic_csv(spark))

    has_hour = df.filter(F.col("hour").isNotNull()).limit(1).count() > 0
    time_col = "hour" if has_hour else "day_of_week"

    by_time = (
        df.groupBy("area", "road", time_col)
        .agg(F.avg("volume").alias("avg_volume"))
    )

    window = Window.partitionBy("area", "road").orderBy(F.desc("avg_volume"))
    ranked = by_time.withColumn("rank", F.row_number().over(window))
    peaks = ranked.filter(F.col("rank") <= 3).withColumnRenamed(time_col, "peak_window")

    write_results(peaks, "peak_windows")

    spark.stop()


if __name__ == "__main__":
    main()
