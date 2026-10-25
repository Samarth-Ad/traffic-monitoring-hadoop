"""Centralised, environment-overridable configuration.

Every path used by the platform is derived here so that the same code runs:

* on the host (this repo checked out locally), and
* inside the Docker containers (where ``/data`` is the shared bind mount).

Override any location with an environment variable of the same name, e.g.::

    DATA_DIR=/tmp/mydata python ml/knn_predict.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this ``common`` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


# ---- Local (host / shared-volume) paths -------------------------------------

# In containers DATA_DIR is /data (the bind mount). On the host it is ./data.
DATA_DIR = _path_env("DATA_DIR", PROJECT_ROOT / "data")

RAW_DIR = _path_env("RAW_DIR", DATA_DIR / "raw")
SOURCE_DIR = _path_env("SOURCE_DIR", DATA_DIR / "source")      # committed raw source
SAMPLE_DIR = _path_env("SAMPLE_DIR", DATA_DIR / "sample")      # committed fixture
STREAM_DIR = _path_env("STREAM_DIR", DATA_DIR / "stream")      # local mirror of HDFS stream
ANALYTICS_DIR = _path_env("ANALYTICS_DIR", DATA_DIR / "analytics")
GRAPH_DIR = _path_env("GRAPH_DIR", DATA_DIR / "graph")
MODELS_DIR = _path_env("MODELS_DIR", DATA_DIR / "models")
ANOMALY_DIR = _path_env("ANOMALY_DIR", ANALYTICS_DIR / "anomalies")
MAPREDUCE_DIR = _path_env("MAPREDUCE_DIR", ANALYTICS_DIR / "mapreduce")

# Canonical, prepared dataset (produced by python/prepare_dataset.py).
CANONICAL_CSV = _path_env("CANONICAL_CSV", RAW_DIR / "bangalore_traffic.csv")
SAMPLE_CSV = _path_env("SAMPLE_CSV", SAMPLE_DIR / "sample_traffic.csv")

# ---- HDFS paths -------------------------------------------------------------

HADOOP_NAMENODE = os.environ.get("HADOOP_NAMENODE", "hdfs://hadoop:9000")
HDFS_INPUT = os.environ.get("HDFS_INPUT", "/traffic/input")
HDFS_STREAM_ROOT = os.environ.get("HDFS_STREAM_ROOT", "/traffic/stream")
HDFS_MAPREDUCE_OUTPUT = os.environ.get("HDFS_MAPREDUCE_OUTPUT", "/traffic/mapreduce/output")
WEBHDFS_URL = os.environ.get("WEBHDFS_URL", "http://hadoop:9870")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
