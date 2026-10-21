# analytics/ — Phase 3

PySpark 3.5 batch jobs that read the Bangalore traffic CSV from HDFS
(`hdfs://hadoop:9000/traffic/input/`) and write Parquet to both HDFS
(`/traffic/analytics/`) and the shared volume (`/data/analytics/`) for the
Streamlit dashboard.

Jobs:

| Job                     | Output table(s)                              |
|-------------------------|----------------------------------------------|
| `traffic_volume.py`     | `traffic_volume_by_road`, `..._by_road_day`  |
| `avg_speed.py`          | `avg_speed_by_road`                          |
| `congestion.py`         | `congestion_by_road`, `congestion_by_hour`   |
| `peak_hour.py`          | `peak_windows`                               |

Run them all:

```bash
docker exec -it poc-spark bash /app/analytics/run_all.sh
```

Input override (env): `TRAFFIC_INPUT=hdfs://hadoop:9000/traffic/stream/*/*.csv`
to analyse the Phase-2 streamed batches instead of the one-shot upload.
