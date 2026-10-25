# Beyond-Syllabus Report

**What we implemented that goes beyond the BDC syllabus — and why it qualifies.**

The course syllabus (Units 1–6) covers Hadoop/HDFS/YARN, MapReduce, NoSQL &
Kafka, stream sampling/filtering (reservoir sampling, Bloom filter, DGIM,
Flajolet-Martin), link analysis (PageRank/HITS), recommendation systems, social
network analytics (centrality, community detection), and Spark/Hive/HBase.

The project's headline extension is a feature the syllabus does **not** cover:

---

## 1. Primary feature — Near-real-time traffic anomaly detection

**File:** `analytics/anomaly_detection.py`

### 1.1 Why this is beyond the syllabus
The Streaming unit (Unit 3) teaches *sampling and counting* over streams
(reservoir sampling, Bloom filter, DGIM, Flajolet-Martin, concept drift) — i.e.
summarising a stream under memory constraints. It does **not** cover **online
outlier / anomaly detection** using robust statistics with **stateful,
deduplicated, incremental batch processing**. That is exactly what this module
adds: a streaming *decision* layer on top of the ingestion pipeline.

### 1.2 Method (interpretable, robust)
For each `(road_id, hour)` bucket we learn a baseline from history:

```
median_rh  = median(volume in bucket)
MAD_rh     = median(|volume - median_rh|)          # median absolute deviation
scale_rh   = MAD_rh / 0.6745   (fallback to std if MAD = 0)
robust_z   = (observed - median_rh) / scale_rh
flag if |robust_z| >= THRESHOLD   (default 3.5)
```

Median/MAD are used instead of mean/std because traffic volume is skewed and the
whole point is to find outliers — a robust location/scale estimate is not itself
distorted by the outliers we are hunting. Direction (`high`/`low`) and severity
(`moderate`/`high`/`severe`) come from the sign and magnitude of `z`.

### 1.3 Engineering guarantees (all implemented & tested)
| Requirement | Implementation |
|-------------|----------------|
| No future→past leakage | baselines built only from historical/training data |
| Each batch processed once | `processed_batches` set persisted in `state.json` |
| No repeated alert for one event | dedup key `road\|dow\|hour\|batch`, persisted |
| Event vs detection time separated | `event_bucket` (dow/hour) + wall-clock `detection_time` |
| Latency measured | `latency_ms = detection_time − arrival_time` per alert |
| Warm-up handled | buckets with `< min_samples` history are skipped & counted |
| Malformed rows never alert | batch is validated first; rejects counted |
| State recovery | re-running reloads baselines + seen batches/alerts (resumes) |
| Outputs isolated from raw data | alerts/state under `data/analytics/anomalies/` |

### 1.4 Incremental / "near-real-time" behaviour
`run_directory()` scans a stream directory, processes **only unseen** batch
files (tracked by id), and appends alerts — it does not rescan history every
time. Paired with `stream_producer.py SINK=local`, new batch files are detected
and scored as they arrive, with no manual re-run of analytics.

### 1.5 Results (reproducible demo)
Run: `bash scripts/run_pipeline_local.sh`

* 148 `(road, hour)` baselines built from 1478 rows.
* 6 batches processed (5 replayed + 1 **injected** anomaly).
* A controlled injected spike — `loc_00`, day 1, hour 2, volume 99 vs expected
  12 — was flagged **severe** with `z ≈ 54.4`.
* 50 alerts total (24 moderate / 19 high / 7 severe); a normal 200-row batch
  produced **0** false alerts in the unit test.
* Median detection latency ≈ **1.6 s** (measured over pre-written batches
  processed in one pass — this is processing latency, not live sensor latency).
* Re-running is **idempotent**: already-processed batches are skipped and no
  duplicate alerts are emitted (verified in `tests/test_anomaly.py`).

### 1.6 Tests (`tests/test_anomaly.py`)
Injected-anomaly detection, normal-batch (no false positive), duplicate-batch
skip + idempotency, warm-up skip, malformed-row safety, state recovery, and
latency recording — 8 cases, all passing.

---

## 2. Secondary extensions

### 2.1 KNN as a spatiotemporal *regressor* (syllabus-adjacent extension)
Unit 4 frames nearest-neighbours under **collaborative-filtering
recommendation** and teaches MAE/RMSE as evaluation metrics. We reuse the
nearest-neighbour idea but as a **supervised traffic-volume regressor** with:
* cyclical `sin/cos` encoding of hour and day (so 23:00 neighbours 00:00),
* one-hot road encoding (an id is never a numeric distance),
* a **leakage-free chronological (replay-order) split**, and
* a **baseline comparison** (per-road/hour historical mean).

The engineering around temporal leakage and feature scaling for a distance
metric goes beyond the syllabus's recommendation-metric treatment.
**Result:** 53.9 % lower MAE than the baseline (6.66 vs 14.44).

### 2.2 Hadoop Streaming in Python
The syllabus presents MapReduce conceptually (and typically in Java). Here it is
implemented with **Hadoop Streaming** so the mapper/reducer are language-agnostic
Python programs, with an explicit, justified decision to omit the combiner
(key/value-type mismatch) and the custom partitioner (default hashing suffices).

### 2.3 Real-world road network via OSMnx
Unit 5 covers graph representation and centrality abstractly. We construct a
**real** Bangalore drivable network from OpenStreetMap (OSMnx) and run PageRank,
approximate betweenness, sampled closeness and Dijkstra shortest paths on it —
applying the syllabus graph theory to live map data.

### 2.4 Dockerized multi-service orchestration + reproducible testing
A 4-service Docker Compose stack (HDFS/YARN, Spark, streamer, dashboard) with a
shared data volume, WebHDFS ingestion, environment-driven configuration, and a
25-test `unittest` suite with an end-to-end chain — production-style engineering
not required by the syllabus.

---

## 3. Summary table

| Extension | Beyond which unit | Evidence |
|-----------|-------------------|----------|
| Robust anomaly detection (median/MAD, stateful, incremental) | U3 (sampling ≠ anomaly detection) | `analytics/anomaly_detection.py`, `tests/test_anomaly.py`, `alerts.csv` |
| KNN leakage-free spatiotemporal regression | U4 (beyond recommendation metrics) | `ml/knn_predict.py`, `knn_metrics.json` |
| Hadoop Streaming (Python) MapReduce | U2 (impl. + combiner/partitioner analysis) | `mapreduce/`, `tests/test_mapreduce.py` |
| OSMnx real road-network graph | U5 (abstract → real data) | `graph/` outputs |
| Docker orchestration + reproducible tests | engineering | `docker-compose.yml`, `tests/` |

Every number in this report is produced by `bash scripts/run_pipeline_local.sh`
and the test suite; no metric is fabricated.
