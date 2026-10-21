"""Phase 3.2 — Average vehicle speed per road segment."""

from __future__ import annotations

from pyspark.sql import functions as F

from common import make_spark, normalize, read_traffic_csv, write_results


def main() -> None:
    spark = make_spark("AvgSpeed")
    df = normalize(read_traffic_csv(spark))

    by_road = (
        df.groupBy("area", "road")
        .agg(
            F.avg("speed").alias("avg_speed_kmh"),
            F.min("speed").alias("min_speed_kmh"),
            F.max("speed").alias("max_speed_kmh"),
            F.stddev("speed").alias("stddev_speed_kmh"),
            F.count("*").alias("samples"),
        )
        .orderBy("avg_speed_kmh")
    )
    write_results(by_road, "avg_speed_by_road")

    spark.stop()


if __name__ == "__main__":
    main()
