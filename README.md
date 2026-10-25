# Bangalore Traffic Intelligence

**A Distributed Big Data Platform for Traffic Analytics, KNN-Based Traffic
Prediction, and Near-Real-Time Anomaly Detection.**

BDC (Big Data Computing) Semester 7 mini-project. A Dockerized Hadoop + Spark
pipeline that ingests Bangalore traffic data, runs a genuine **Hadoop MapReduce**
aggregation, a **scikit-learn KNN** volume predictor, an interpretable
**near-real-time anomaly detector** (beyond syllabus), builds an OpenStreetMap
**road graph** with classic graph algorithms, and serves everything in a
**Streamlit** dashboard.

> Hadoop 3.4.2 · Spark 3.5 · Hadoop Streaming (Python) · scikit-learn · OSMnx /
> NetworkX · Streamlit · Docker Compose.

---

## 1. Problem statement & scope

Urban traffic data is high-volume, continuously arriving, and spatially
structured. This project demonstrates the Big Data toolchain end to end on a
real Bangalore traffic dataset:

1. **Store & distribute** the data on HDFS.
2. **Batch-aggregate** it with MapReduce and Spark (volume / speed / congestion
   / peak hours).
3. **Predict** near-term traffic volume per road/time with KNN.
4. **Detect** unusual traffic events as new batches arrive (anomaly detection).
5. **Analyse the road network** as a graph (PageRank, centrality, shortest path).
6. **Visualise** all of it in one dashboard.

**Honest framing:** the producer *replays historical data* (a reproducible
"near-real-time" replay), it does **not** receive live sensor feeds. "Time"
means replay/arrival order because the raw dataset has no absolute timestamp.

## 2. Dataset & canonical schema

The committed source is `data/source/bangalore_traffic_raw.txt` — a **headerless**
CSV with the positional schema below (1517 rows → **1478** after de-duplication,
**14** coordinate locations across Bangalore):

```
lat , lon , day_of_week(0-6) , hour(0-23) , volume
```

`python/prepare_dataset.py` validates it and writes the **canonical schema**
used by every component:

| column | type | notes |
|--------|------|-------|
| `road_id` | str | stable id from coordinates (`loc_00`…); no fabricated names |
| `area` | str | `Bangalore` |
| `road` | str | coordinate-derived label |
| `lat`, `lon` | float | Bangalore bbox checked |
| `day_of_week` | int | 0–6 |
| `hour` | int | 0–23 |
| `volume` | float | traffic volume / intensity (peaks at commute hours) |

**Data-quality rules** (see `common/schema.py`): required columns, numeric
coercion, range checks (dow 0–6, hour 0–23, volume ≥ 0, coords in bbox), and
exact-duplicate removal. Rejected rows are **never silently dropped** — they are
written to `data/raw/rejected_rows.csv` with a reason and counted in
`data/raw/validation_report.json`.

> The optional Kaggle *named-column* CSV (`python/download_dataset.py`) is also
> supported via column aliases, but requires a Kaggle account, so it is not
> committed. The Spark analytics jobs target that CSV; the MapReduce/KNN/anomaly
> path targets the committed coordinate dataset.

## 3. Architecture

```
 data/source/*.txt ──prepare_dataset──▶ data/raw/bangalore_traffic.csv  (canonical)
        │                                        │
        │                          ┌─────────────┼───────────────────────────┐
        ▼                          ▼             ▼                           ▼
   stream_producer           MapReduce        KNN (ml/)              anomaly_detection
   SINK=hdfs → HDFS          (Hadoop          sklearn →              (robust median/MAD)
   SINK=local → data/stream   Streaming) →    data/models/*          → data/analytics/anomalies/*
        │                     data/analytics/mapreduce/*
        │
   OpenStreetMap ─build_graph/algorithms─▶ data/graph/*   (host, OSMnx)
        │
        └────────────────────────▶  Streamlit dashboard  (reads data/*, port 8501)
```

The host `./data` folder is bind-mounted into every container, so outputs
written by any stage are instantly visible to the dashboard (file-based; no live
HDFS needed to view results).

### Services (`docker-compose.yml`)
| Service | Container | Role |
|---------|-----------|------|
| hadoop | `poc-hadoop` | HDFS (NameNode/DataNode/SecondaryNN) + YARN |
| spark | `poc-spark` | PySpark 3.5 batch analytics (workbench) |
| streamer | `poc-streamer` | replays CSV → HDFS (or local dir) |
| dashboard | `poc-dashboard` | Streamlit app (http://localhost:8501) |

## 4. Quick start

### Option A — host only (no Docker), reproducible in seconds
Works in any shell incl. Windows PowerShell (no bash needed):
```bash
pip install -r requirements-dev.txt
python scripts/run_pipeline_local.py
```
(There is also a bash version, `bash scripts/run_pipeline_local.sh`, if you
prefer; the `.py` runner is identical and portable.) This runs Phases 1–4
(prepare → MapReduce → KNN → stream + anomaly) and writes all dashboard inputs
under `data/`. Run the tests with `python -m unittest discover -s tests -p "test_*.py"`.

### Option B — full Docker stack
```bash
# 0. (optional) real Kaggle CSV:  python python/download_dataset.py
python python/prepare_dataset.py          # build canonical CSV first
docker compose build
docker compose up -d                      # NameNode UI: http://localhost:9871
```
Host ports: NameNode `9871` · DataNode `9865` · ResourceManager `8089` ·
NodeManager `8043` · HDFS RPC `9002` · Dashboard `8501`.

## 5. Running each phase

| Phase | What | Where | Command |
|-------|------|-------|---------|
| 1 | Prepare canonical dataset | host/container | `python python/prepare_dataset.py` |
| 2 | **MapReduce** volume by road×hour | cluster | `docker exec -it poc-hadoop bash /opt/mapreduce/run_streaming.sh` (see `mapreduce/README.md`) |
| 2′ | MapReduce local simulation | host | `bash mapreduce/run_local.sh` |
| 3 | **KNN** prediction | host | `python ml/knn_predict.py --k 7 --weights distance` |
| 4 | **Anomaly** detection | host | stream with `SINK=local python/stream_producer.py` then `python analytics/anomaly_detection.py --stream-dir data/stream` |
| S | Spark analytics (optional, Kaggle CSV) | `poc-spark` | `docker exec -it poc-spark bash /app/analytics/run_all.sh` |
| G | Road graph + algorithms | host | `pip install -r graph/requirements.txt && python graph/build_graph.py && python graph/algorithms.py` |
| 6 | Dashboard | `poc-dashboard` | already running → http://localhost:8501 |

## 6. Dashboard

There are **two** dashboards:

### 6a. Interactive live dashboard — `dashboard/app.py` (recommended for demos)
A self-contained app: choose the **built-in dataset or upload your own CSV**,
then watch the whole pipeline run live — validation, the real MapReduce
aggregate, KNN prediction, and a **streaming anomaly feed** where alerts appear
batch-by-batch. No Spark/OSMnx/Docker needed; it computes everything on the fly.

Run on the host (needs only `streamlit`, `pandas`, `scikit-learn` — already in
`requirements-dev.txt`, plus streamlit):
```bash
pip install streamlit
streamlit run dashboard/app.py          # http://localhost:8501
```
Upload format: headerless `lat,lon,day_of_week,hour,volume`, or any CSV whose
columns map to the canonical schema. Uses Streamlit-native charts (no
plotly/folium required).

### Test datasets (real, public)
The loader accepts real-world CSVs, not just the built-in format. It auto-detects
columns: a **timestamp** (`date_time`, `Date`, `timestamp`, …) becomes `hour` +
`day_of_week`; **lat/lon are optional**; and a **road / segment / sensor** column
becomes the road id. Fetch a ready one:
```bash
python scripts/fetch_sample_datasets.py   # -> data/demo_uploads/metro_interstate_traffic_volume.csv
```
Then in the dashboard pick **Upload my own CSV** and choose that file.

| Dataset | Source | Notes |
|---------|--------|-------|
| UCI Metro Interstate Traffic Volume (48k hourly rows) | `archive.ics.uci.edu/dataset/492` | `date_time` + `traffic_volume`; loads directly (1 site). Verified: KNN MAE 297, R² 0.93. |
| Kaggle "Banglore City Traffic Dataset" (project's native) | `kaggle datasets download -d preethamgouda/banglore-city-traffic-dataset --unzip` | Area/Road names + Traffic Volume + Date; loads via aliases. Needs a Kaggle account. |
| Dublin City SCATS volumes | `data.gov.ie` | 5-min intervals; coords in a separate file — needs a join first. |
| UK DfT road traffic counts | `roadtraffic.dft.gov.uk` | Per-site raw counts + coordinates; join required. |

### 6b. Pipeline viewer — `dashboard/app_pipeline.py` (file-based, full stack)
The original multi-phase viewer (Volume/Speed/Congestion/Peak/Road-network/
Shortest-paths + MapReduce/KNN/Anomaly), reading Parquet/CSV produced by the
Spark and OSMnx phases. Best run via Docker after `docker compose up -d`:
```bash
docker compose up -d --build dashboard   # requires Docker Desktop running
```
Views whose inputs (Spark Parquet, OSMnx graph) have not been generated show a
friendly "not available" message rather than crashing.

## 7. Testing

```bash
python -m unittest discover -s tests -p "test_*.py"   # 25 tests, stdlib only
```
Covers schema validation, MapReduce correctness (vs an independent pandas
reference), KNN (determinism, leakage-free split, invalid params), anomaly
detection (injected anomaly, no false positive, dedup/idempotency, warm-up,
malformed rows, state recovery, latency) and an end-to-end chain.

## 8. Measured results

| Metric | Value |
|--------|-------|
| Input rows / clean / rejected | 1517 / 1478 / 0 (40 duplicates removed) |
| Roads, MapReduce groups (road×hour) | 14 / 319 |
| KNN MAE / RMSE / R² | **6.66 / 9.35 / 0.849** |
| Baseline MAE / RMSE / R² | 14.44 / 18.34 / 0.418 |
| KNN MAE improvement vs baseline | **53.9 %** |
| Anomaly alerts (demo, 6 batches) | 50 (24 moderate / 19 high / 7 severe) |
| Injected anomaly detected | yes (z ≈ 54, severe) |
| Unit/integration tests | 25 passed |

## 9. Repository layout
```
common/        schema + validation + config + logging (shared data contract)
python/        prepare_dataset.py, stream_producer.py (hdfs|local), download_dataset.py
mapreduce/     Hadoop Streaming mapper/reducer + local & cluster runners
ml/            KNN prediction pipeline
analytics/     Spark jobs (volume/speed/congestion/peak) + anomaly_detection.py
graph/         OSMnx graph builder + NetworkX algorithms (host)
dashboard/     Streamlit app
tests/         unittest suite + committed fixture
scripts/       run_pipeline_local.sh (host), demo.sh (viva)
docker/        Dockerfiles + Hadoop config
docs/          REPORT_PROJECT.md, REPORT_BEYOND_SYLLABUS.md
data/          source (committed) · sample (committed) · raw/stream/analytics/models/graph (generated)
```

## 10. Known limitations
* **No live data** — historical replay only (reproducible demo mode).
* **No absolute timestamp** — only cyclic day/hour, so chronological evaluation
  uses replay order; a true calendar split is not possible on this dataset.
* **Volume units** are an index, not a calibrated vehicle count.
* **Anomaly latency** (~1.6 s median in the demo) is measured over pre-written
  batches processed in one pass, not a continuous watcher — see the report.
* The Spark speed/congestion jobs need the Kaggle named-column CSV; on the
  coordinate dataset the MapReduce path is the primary aggregation.
