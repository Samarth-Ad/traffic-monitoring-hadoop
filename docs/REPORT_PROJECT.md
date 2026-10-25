# Project Report — Bangalore Traffic Intelligence

**A Distributed Big Data Platform for Traffic Analytics, KNN-Based Traffic
Prediction, and Near-Real-Time Anomaly Detection**

BDC (Big Data Computing), Semester 7 — mini-project report.

---

## 1. Problem statement

Urban traffic data is a canonical Big Data problem: it is **high volume**,
**continuously arriving** (velocity), **varied** (spatial + temporal + network
structure), of **uncertain quality** (veracity), and only useful if turned into
decisions (value). This project builds an end-to-end platform that stores,
processes, models and visualises a real Bangalore traffic dataset using the
standard Big Data stack (HDFS, YARN, MapReduce, Spark) plus a machine-learning
predictor and an anomaly-detection layer.

## 2. Objectives & scope

1. Store the dataset on **HDFS** and stream it in batches (ingestion).
2. Run a genuine **Hadoop MapReduce** aggregation (group-by road × time bucket).
3. Run **Spark** batch analytics (volume / speed / congestion / peak hour).
4. Predict traffic volume with **K-Nearest-Neighbours** regression.
5. Detect **unusual traffic events in near-real-time** (beyond syllabus).
6. Build the city **road network graph** and run PageRank / centrality /
   shortest paths.
7. Present everything in a **Streamlit dashboard**, with a reproducible test
   suite and documentation.

**Out of scope / honest limits:** no live sensor feed (historical replay only);
no calendar timestamp in the raw data (only cyclic day/hour); traffic "volume"
is an index, not a calibrated vehicle count.

## 3. Existing system vs. improvements

| Area | Before this work | After |
|------|------------------|-------|
| Dataset | Headerless `.txt`, undocumented schema; empty Java MapReduce scaffold | Documented canonical schema + validation; dead Java scaffold removed |
| MapReduce | Empty `Mapper/Reducer/Driver` classes | Working Hadoop Streaming job + local test + reference check |
| Prediction | None | KNN regressor with leakage-free evaluation and baseline |
| Anomaly detection | None | Incremental robust-statistics detector with state + dedup |
| Dashboard | Volume/Speed/Congestion/Peak/Graph views | + MapReduce, KNN and Anomaly views |
| Tests | None | 25-test `unittest` suite + end-to-end chain |
| Config | Hard-coded paths | Central `common/config.py`, env-overridable |

## 4. Architecture

```
 data/source/*.txt ──prepare_dataset──▶ data/raw/bangalore_traffic.csv (canonical)
        │                                      │
        │               ┌──────────────┬───────┴────────┬───────────────┐
        ▼               ▼              ▼                ▼               ▼
   stream_producer   MapReduce     Spark jobs        KNN (ml/)     anomaly_detection
   → HDFS / local    (Streaming)   (optional)        sklearn       robust median/MAD
        │            road×hour agg                    data/models   data/analytics/anomalies
        │
   OpenStreetMap ─build_graph/algorithms─▶ data/graph/*
        │
        └──────────────────────▶ Streamlit dashboard (port 8501, reads data/*)
```

Four Docker services (`poc-hadoop`, `poc-spark`, `poc-streamer`,
`poc-dashboard`) share the `./data` bind mount. HDFS is the primary store for
the streaming + Spark + MapReduce stages; the dashboard is deliberately
file-based (reads CSV/Parquet) so results can be viewed without a live cluster.

## 5. Component responsibilities

### 5.1 Data contract — `common/schema.py`
Single source of truth for the canonical schema, loaders (headerless positional
**or** Kaggle named-column via aliases), and validation. Validation coerces
numerics, range-checks every field, removes exact duplicates and returns a
`ValidationReport` of clean vs. rejected rows **with reasons** — nothing is
silently discarded. Result on the real data: 1517 → **1478** clean (40
duplicates), **0** rejected, **14** roads.

### 5.2 Ingestion — `python/stream_producer.py`
Replays the CSV in configurable batches (default 500 rows / 2 s). Two sinks:
`SINK=hdfs` (WebHDFS → `/traffic/stream/…`, the original behaviour) or
`SINK=local` (→ `data/stream/…`, for a Docker-free demo). Replay order is the
system's notion of arrival time.

### 5.3 Hadoop MapReduce — `mapreduce/`
Hadoop Streaming job (Python mapper + reducer). The **mapper** emits
`road_id#hour \t volume`; Hadoop's **shuffle-and-sort** groups by key; the
**reducer** computes count, average, min, max, sum per group. Chosen over native
Java because the stack is already Python and the same scripts run both on the
cluster (`-files mapper.py,reducer.py`) and locally. No combiner (would need a
separate partial-aggregate format) and no custom partitioner (default hashing
already groups each key at one reducer). Overwrite-protected on HDFS.
**Result:** 1478 records → **319** (road, hour) groups, verified byte-for-byte
against an independent pandas aggregation (average matches to 4 dp).

### 5.4 Spark analytics — `analytics/` (optional path)
Existing PySpark jobs (`traffic_volume`, `avg_speed`, `congestion`, `peak_hour`)
over the Kaggle named-column CSV, writing Parquet to HDFS and `./data`.
Preserved unchanged; they target the optional Kaggle dataset.

### 5.5 KNN prediction — `ml/knn_predict.py`
Predicts `volume` for a `(road_id, day_of_week, hour)` bucket.
Feature engineering: `road_id` one-hot (an id is never a numeric distance),
`hour`/`day_of_week` as cyclical sin/cos, standardised. **Chronological split by
replay order** (first 80 % train / last 20 % test) — no future row informs a
past prediction, target excluded from features. Compared against a historical
per-`(road, hour)` mean baseline. **Result:** KNN MAE **6.66**, RMSE **9.35**,
R² **0.849** vs baseline 14.44 / 18.34 / 0.418 → **53.9 %** MAE reduction.

### 5.6 Anomaly detection — `analytics/anomaly_detection.py` (beyond syllabus)
See `docs/REPORT_BEYOND_SYLLABUS.md`. Robust median/MAD baselines per
`(road, hour)`; incremental per-batch scoring with persisted state,
deduplication, warm-up handling and latency measurement.

### 5.7 Road graph — `graph/`
OSMnx downloads Bangalore's drivable network; NetworkX computes PageRank,
approximate betweenness, sampled closeness, and sample shortest paths.

### 5.8 Dashboard — `dashboard/app.py`
Streamlit views for every output; each view shows a friendly message if its file
is absent, so the app never crashes on a partial pipeline.

## 6. Data flows

* **MapReduce flow:** canonical CSV → HDFS `/traffic/input` → mapper(key=road#hour)
  → shuffle/sort → reducer(agg) → HDFS `/traffic/mapreduce/output` + local CSV.
* **KNN flow:** canonical CSV → feature matrix (one-hot road + cyclical time) →
  replay-order split → fit KNN → predict test → MAE/RMSE/R² + predictions CSV.
* **Anomaly flow:** history → median/MAD baselines → each arriving batch scored
  (robust z) → dedup → alerts CSV + persisted state.

## 7. Measured results

| Metric | Value | How measured |
|--------|-------|--------------|
| Input / clean / rejected rows | 1517 / 1478 / 0 | `prepare_dataset.py` + `validation_report.json` |
| Duplicates removed | 40 | validation report |
| MapReduce groups (road×hour) | 319 | `run_local.sh` + reference test |
| MapReduce vs reference | exact (avg to 4 dp) | `tests/test_mapreduce.py` |
| KNN MAE / RMSE / R² | 6.66 / 9.35 / 0.849 | `knn_metrics.json` |
| Baseline MAE / RMSE / R² | 14.44 / 18.34 / 0.418 | `knn_metrics.json` |
| KNN improvement | 53.9 % MAE | computed |
| Anomaly alerts (6-batch demo) | 50 (24/19/7 mod/high/severe) | `alerts.csv` |
| Injected anomaly | detected, z≈54, severe | demo run |
| Median detection latency | ≈1.6 s (batch-replay) | `alerts.csv` latency_ms |
| Tests | 25 passed | `unittest` |

> **Measurement honesty.** KNN/MapReduce numbers are real runs on the full
> dataset. The MapReduce timing (~0.24 s) is the *local simulation*; a real
> cluster run adds JVM/YARN scheduling overhead. The anomaly latency is measured
> over batches written to disk and then processed in a single pass, so it
> reflects processing time, not a live end-to-end sensor latency.

## 8. Mapping to the syllabus

| Syllabus unit | Covered by |
|---------------|-----------|
| U1 Hadoop/HDFS/YARN | Dockerized HDFS + YARN cluster; WebHDFS ingestion |
| U2 MapReduce: map, shuffle/sort, reduce, combiner, partitioner; Group-By & Aggregation | `mapreduce/` job (group-by road×hour aggregation; combiner/partitioner discussed and justified) |
| U3 Link Analysis: PageRank | `graph/algorithms.py` PageRank on the road graph |
| U4 Recommendation metrics: nearest-neighbour, MAE/RMSE | KNN regressor + MAE/RMSE/R² evaluation |
| U5 Social-network analytics: Degree/Closeness/Betweenness centrality | `graph/algorithms.py` betweenness + closeness |
| U6 Spark: DataFrames, transformations/actions | `analytics/*.py` PySpark jobs |

## 9. Known limitations & future work

* **Live data:** integrate a real feed (e.g., Kafka topic) instead of replay.
* **Timestamps:** with a calendar timestamp, do a true temporal split and
  seasonality features.
* **Distributed ML:** KNN here is single-node (sklearn); Spark MLlib would make
  it distributed.
* **Anomaly baselines:** could decay over time (online update) rather than a
  fixed historical baseline.
* **Road names:** join coordinates to OSM names for friendlier labels.

## 10. Viva Q&A (selected)

* **Why Hadoop/HDFS?** Distributed, fault-tolerant storage (block replication)
  that lets MapReduce move computation to the data across nodes.
* **Mapper/Reducer/shuffle/sort?** Mapper emits key-value pairs; Hadoop groups
  and sorts by key (shuffle & sort); reducer aggregates each key's values. Here
  key = `road#hour`, values = volumes.
* **Combiner vs partitioner?** A combiner does local pre-aggregation (must share
  the reducer's key/value types — omitted here for that reason); a partitioner
  routes keys to reducers (default hashing suffices here).
* **MapReduce vs Spark?** MapReduce writes to disk between stages; Spark keeps
  RDDs/DataFrames in memory and is faster for iterative work.
* **Why KNN, and choosing k?** Simple, non-parametric, captures local
  road/time similarity; k trades bias vs variance (k=7, distance-weighted).
* **Temporal leakage?** Future rows must not inform past predictions — hence the
  replay-order split and no target in the feature vector.
* **MAE vs RMSE?** MAE is the mean absolute error (robust); RMSE penalises large
  errors more; both compared against a historical-mean baseline.
* **Robust anomaly detection?** Median/MAD are outlier-resistant location/scale
  estimates; robust z ≥ threshold flags unusual volumes.
* **Near-real-time vs live stream?** We replay historical batches (reproducible)
  rather than consuming a live producer — the processing path is the same.
