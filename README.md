# Bangalore Traffic Analytics — Dockerized Hadoop + Spark + Graph + Dashboard

Big Data Computing mini-project. A full pipeline that **streams** the Kaggle
Bangalore traffic dataset into HDFS, runs **Spark batch analytics** over it,
builds a **road-network graph** of the city from OpenStreetMap, runs classic
**graph algorithms** on it, and serves everything in a **Streamlit dashboard**.

Hadoop 3.4.2 · Spark 3.5 · Python 3 · Streamlit · OSMnx.

---

## How the pieces fit together

```
                         ┌──────────────────────────────────────────┐
  Kaggle CSV ──────────▶ │  streamer   replays CSV row-by-row         │
  (data/raw/)            │  (poc-streamer)   ──WebHDFS──▶ HDFS        │
                         └──────────────────────────────────────────┘
                                             │  /traffic/stream/...
                                             ▼
                         ┌──────────────────────────────────────────┐
                         │  hadoop     HDFS + YARN (poc-hadoop)       │
                         └──────────────────────────────────────────┘
                                             │  hdfs://hadoop:9000
                                             ▼
                         ┌──────────────────────────────────────────┐
                         │  spark      PySpark batch jobs (poc-spark) │
                         │  writes → hdfs:///traffic/analytics/       │
                         │         → /data/analytics/*.parquet        │
                         └──────────────────────────────────────────┘
                                             │
   OpenStreetMap ───▶ build_graph.py ───▶ data/graph/*.graphml,*.parquet
   (host, osmnx)      algorithms.py                  │
                                                     ▼
                         ┌──────────────────────────────────────────┐
                         │  dashboard  Streamlit (poc-dashboard)      │
                         │  reads /data/analytics + /data/graph       │
                         │  http://localhost:8501                     │
                         └──────────────────────────────────────────┘
```

**Key idea:** the host `./data` folder is mounted into every container, so
output written by Spark (or the host graph scripts) is instantly visible to the
dashboard. HDFS is still the real store for the streaming + Spark stages; the
dashboard is deliberately **file-based** (reads Parquet, no live HDFS needed).

### Services

| Service    | Image          | Container       | Role                                          |
|------------|----------------|-----------------|-----------------------------------------------|
| hadoop     | poc-hadoop     | `poc-hadoop`    | HDFS NameNode / DataNode / SecondaryNN + YARN |
| spark      | poc-spark      | `poc-spark`     | PySpark 3.5 driver (batch analytics)          |
| streamer   | poc-streamer   | `poc-streamer`  | Python producer, replays CSV → HDFS           |
| dashboard  | poc-dashboard  | `poc-dashboard` | Streamlit app (port 8501)                     |

`spark` and `streamer` idle on `tail -f /dev/null` — they're workbenches you
`docker exec` into to run jobs, not auto-running services.

---

## Quick start

> **Heads up for a fresh clone:** everything under `data/` (raw, stream,
> analytics, graph) is gitignored, so after cloning those folders are empty.
> The dashboard will look blank until you run Phases 2–5 below to generate the
> data. Nothing is broken — you just haven't produced the outputs yet.

### 0. Prerequisites

- **Docker Desktop** (running).
- **Python 3.10+ on the host** — only needed for the graph phases (4 & 5),
  which use OSMnx and need internet access to the OpenStreetMap Overpass API.
- A **Kaggle account + API token** for the dataset.

### 1. Get the dataset

```bash
pip install kaggle
# Kaggle → Account → Create New API Token, save as ~/.kaggle/kaggle.json
python python/download_dataset.py
```

This downloads `preethamgouda/banglore-city-traffic-dataset` into
`data/raw/Banglore_traffic_Dataset.csv`.

### 2. Build & start the cluster

```bash
docker compose build
docker compose up -d
```

First build is slow (Hadoop + Spark images). Check it's up:

```bash
docker compose ps
```

Then open the **NameNode UI** at http://localhost:9871 to confirm HDFS is alive.

**Host ports:** NameNode UI `9871` · DataNode UI `9865` · ResourceManager `8089`
· NodeManager `8043` · HDFS RPC `9002` · Dashboard `8501`.

---

## Running the pipeline (phase by phase)

| Phase | What it does                              | Where it runs | Command |
|-------|-------------------------------------------|---------------|---------|
| 1     | HDFS + YARN cluster                       | containers    | `docker compose up -d` (step 2 above) |
| 2     | Stream CSV → HDFS                         | `poc-streamer`| `docker exec -it poc-streamer python stream_producer.py` |
| 3     | Spark batch analytics                     | `poc-spark`   | `docker exec -it poc-spark bash /app/analytics/run_all.sh` |
| 4     | Build OSM road-network graph              | **host**      | `pip install -r graph/requirements.txt` then `python graph/build_graph.py` |
| 5     | Shortest path / PageRank / centrality     | **host**      | `python graph/algorithms.py` |
| 6     | Streamlit dashboard                       | `poc-dashboard`| already running → http://localhost:8501 |

### Why Phases 4 & 5 run on the host, not in a container

`build_graph.py` and `algorithms.py` use **OSMnx**, which downloads the
Bangalore road network over HTTPS from the Overpass API. They're easiest to run
in a plain host Python env with internet access. They write their outputs into
`data/graph/` (which is mounted into the dashboard container), so the dashboard
picks them up automatically — no container rebuild needed.

What each phase produces:

- **Phase 2** → `hdfs:///traffic/stream/YYYY-MM-DD/batch-*.csv` (batches of 500 rows, every 2s; tunable via env in `docker-compose.yml`).
- **Phase 3** → `hdfs:///traffic/analytics/` **and** `data/analytics/*.parquet` (traffic volume, avg speed, congestion, peak hour).
- **Phase 4** → `data/graph/bangalore.graphml`, `nodes.parquet`, `edges.parquet`.
- **Phase 5** → `data/graph/pagerank.parquet`, `betweenness.parquet`, `closeness.parquet`, `sample_shortest_path.parquet`.
- **Phase 6** → reads all of the above and renders charts + a folium map.

---

## Repository layout

```
docker/            Dockerfiles + Hadoop config (core-site, hdfs-site, yarn-site, ...)
docker-compose.yml Wires the 4 services together and mounts ./data everywhere
python/            Host/streamer scripts: download_dataset.py, stream_producer.py
analytics/         Phase-3 Spark jobs + run_all.sh (runs inside poc-spark)
graph/             Phase-4/5 OSMnx + networkx scripts (run on the host)
dashboard/         Phase-6 Streamlit app (app.py)
data/              Shared data volume — gitignored, populated by the pipeline
```

Each folder has its own `README.md` with details specific to that stage.

---

## Working on this with Claude Code

If you're picking this up and using Claude to help:

- The **data flow diagram above** is the mental model — most confusion comes
  from forgetting that `./data` is shared across containers and that the graph
  phases run on the host.
- Container names (`poc-spark`, etc.) are fixed in `docker-compose.yml`; use
  them for `docker exec`.
- Logs: `docker compose logs -f <service>` (e.g. `dashboard`, `hadoop`).
- To wipe and restart clean: `docker compose down -v` (the `-v` also drops the
  `hadoop-data` HDFS volume), then rebuild. Your `data/` outputs on the host are
  separate and survive this.
- If the dashboard is empty, it's almost always because Phases 2–5 haven't been
  run yet in this environment — not a code bug.

---

## Common issues

| Symptom | Likely cause / fix |
|---------|--------------------|
| Dashboard loads but shows nothing | Phases 3–5 haven't run yet → no Parquet in `data/`. Run them. |
| `kaggle: command not found` | `pip install kaggle` and put the token at `~/.kaggle/kaggle.json`. |
| Streamer can't reach HDFS | Confirm `poc-hadoop` is healthy (NameNode UI on `9871`); services talk over the `bdc` docker network as `hadoop:9000` / `hadoop:9870`. |
| OSMnx errors in Phase 4 | Needs host internet to the Overpass API; run it outside Docker in a host venv. |
| Port already in use | Another process owns `8501`/`9871`/etc. on the host — stop it or remap in `docker-compose.yml`. |
