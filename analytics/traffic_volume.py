"""Phase 3.1 — Traffic volume aggregated by road segment and time window."""

from __future__ import annotations

from pyspark.sql import functions as F

from common import make_spark, normalize, read_traffic_csv, write_results


def main() -> None:
    spark = make_spark("TrafficVolume")
    df = normalize(read_traffic_csv(spark))

    by_road = (
        df.groupBy("area", "road")
        .agg(
            F.sum("volume").alias("total_volume"),
            F.avg("volume").alias("avg_volume"),
            F.max("volume").alias("peak_volume"),
            F.count("*").alias("samples"),
        )
        .orderBy(F.desc("total_volume"))
    )
    write_results(by_road, "traffic_volume_by_road")

    by_road_day = (
        df.groupBy("area", "road", "day_of_week")
        .agg(F.sum("volume").alias("total_volume"))
    )
    write_results(by_road_day, "traffic_volume_by_road_day")

    spark.stop()


if __name__ == "__main__":
    main()
