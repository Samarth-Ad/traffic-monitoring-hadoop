"""
Phase 2 — Traffic streaming simulator with buffered HDFS ingestion.

Replays the Kaggle Bangalore traffic CSV row-by-row at a configurable rate,
accumulating rows into an in-memory buffer, and flushing each batch to HDFS
under /traffic/stream/YYYY-MM-DD/batch-<epoch>.csv via WebHDFS.

Env vars (see docker-compose.yml):
    WEBHDFS_URL             http://hadoop:9870
    HDFS_USER               root
    DATASET_CSV             /data/raw/Banglore_traffic_Dataset.csv
    BATCH_SIZE              500
    BATCH_INTERVAL_SECONDS  2
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
from hdfs import InsecureClient

LOG = logging.getLogger("stream_producer")

HDFS_STREAM_ROOT = "/traffic/stream"


def iter_rows(csv_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of ``chunksize`` rows from a (possibly large) CSV."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        yield chunk


def flush_batch(client: InsecureClient, batch: pd.DataFrame, batch_id: int) -> str:
    """Write a batch to HDFS as CSV and return the HDFS path."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    hdfs_dir = f"{HDFS_STREAM_ROOT}/{today}"
    hdfs_path = f"{hdfs_dir}/batch-{int(time.time())}-{batch_id:06d}.csv"
    client.makedirs(hdfs_dir)
    buf = io.BytesIO()
    batch.to_csv(buf, index=False)
    buf.seek(0)
    with client.write(hdfs_path, overwrite=True) as writer:
        writer.write(buf.read())
    return hdfs_path


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    webhdfs_url = os.environ.get("WEBHDFS_URL", "http://hadoop:9870")
    hdfs_user = os.environ.get("HDFS_USER", "root")
    dataset_csv = Path(os.environ.get("DATASET_CSV", "/data/raw/Banglore_traffic_Dataset.csv"))
    batch_size = int(os.environ.get("BATCH_SIZE", "500"))
    batch_interval = float(os.environ.get("BATCH_INTERVAL_SECONDS", "2"))

    if not dataset_csv.exists():
        LOG.error("Dataset not found at %s. Run python/download_dataset.py first.", dataset_csv)
        return 1

    LOG.info("WebHDFS=%s user=%s", webhdfs_url, hdfs_user)
    LOG.info("CSV=%s batch_size=%d interval=%.1fs", dataset_csv, batch_size, batch_interval)

    client = InsecureClient(webhdfs_url, user=hdfs_user)
    client.makedirs(HDFS_STREAM_ROOT)

    total_rows = 0
    for batch_id, batch in enumerate(iter_rows(dataset_csv, batch_size)):
        hdfs_path = flush_batch(client, batch, batch_id)
        total_rows += len(batch)
        LOG.info("Flushed batch #%d (%d rows) -> hdfs://%s  total=%d",
                 batch_id, len(batch), hdfs_path, total_rows)
        time.sleep(batch_interval)

    LOG.info("Streaming complete. %d rows written to HDFS.", total_rows)
    return 0


if __name__ == "__main__":
    sys.exit(run())
