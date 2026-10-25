"""
Phase 2 — Traffic streaming simulator (two interchangeable sinks).

Replays the traffic CSV batch-by-batch at a configurable rate. Each batch is
flushed to a sink:

    SINK=hdfs   (default) -> WebHDFS, /traffic/stream/YYYY-MM-DD/batch-*.csv
                             (the original containerised behaviour, preserved)
    SINK=local            -> local directory DATA_DIR/stream/YYYY-MM-DD/batch-*.csv
                             (cluster-free; lets the anomaly detector and the
                             dashboard run a reproducible near-real-time demo
                             on the host with no Hadoop)

The replay order of rows is treated as arrival/"time" order (the raw dataset
has no absolute timestamp).

Env vars (see docker-compose.yml):
    SINK                    hdfs | local            (default hdfs)
    WEBHDFS_URL             http://hadoop:9870
    HDFS_USER               root
    DATASET_CSV             dataset to replay       (default canonical CSV)
    LOCAL_STREAM_DIR        local sink dir          (default DATA_DIR/stream)
    BATCH_SIZE              500
    BATCH_INTERVAL_SECONDS  2
    MAX_BATCHES             optional cap for demos/tests
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

LOG = get_logger("stream_producer")

HDFS_STREAM_ROOT = os.environ.get("HDFS_STREAM_ROOT", "/traffic/stream")


def iter_rows(csv_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of ``chunksize`` rows from a (possibly large) CSV."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        yield chunk


def _batch_name(batch_id: int) -> str:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return today, f"batch-{int(time.time())}-{batch_id:06d}.csv"


class HdfsSink:
    def __init__(self, url: str, user: str):
        from hdfs import InsecureClient  # imported lazily so local mode needs no hdfs pkg

        self.client = InsecureClient(url, user=user)
        self.client.makedirs(HDFS_STREAM_ROOT)

    def write(self, batch: pd.DataFrame, batch_id: int) -> str:
        day, name = _batch_name(batch_id)
        hdfs_dir = f"{HDFS_STREAM_ROOT}/{day}"
        hdfs_path = f"{hdfs_dir}/{name}"
        self.client.makedirs(hdfs_dir)
        buf = io.BytesIO()
        batch.to_csv(buf, index=False)
        buf.seek(0)
        with self.client.write(hdfs_path, overwrite=True) as writer:
            writer.write(buf.read())
        return hdfs_path


class LocalSink:
    def __init__(self, root: Path):
        self.root = config.ensure_dir(Path(root))

    def write(self, batch: pd.DataFrame, batch_id: int) -> str:
        day, name = _batch_name(batch_id)
        out_dir = config.ensure_dir(self.root / day)
        out_path = out_dir / name
        batch.to_csv(out_path, index=False)
        return str(out_path)


def run() -> int:
    sink_kind = os.environ.get("SINK", "hdfs").lower()
    dataset_csv = Path(os.environ.get("DATASET_CSV", str(config.CANONICAL_CSV)))
    batch_size = int(os.environ.get("BATCH_SIZE", "500"))
    batch_interval = float(os.environ.get("BATCH_INTERVAL_SECONDS", "2"))
    max_batches = int(os.environ.get("MAX_BATCHES", "0")) or None

    if not dataset_csv.exists():
        LOG.error("Dataset not found at %s. Run python/prepare_dataset.py first.", dataset_csv)
        return 1

    if sink_kind == "local":
        sink: HdfsSink | LocalSink = LocalSink(os.environ.get("LOCAL_STREAM_DIR", str(config.STREAM_DIR)))
        LOG.info("Sink=local dir=%s", sink.root)
    else:
        webhdfs_url = os.environ.get("WEBHDFS_URL", config.WEBHDFS_URL)
        hdfs_user = os.environ.get("HDFS_USER", "root")
        sink = HdfsSink(webhdfs_url, hdfs_user)
        LOG.info("Sink=hdfs url=%s user=%s", webhdfs_url, hdfs_user)

    LOG.info("CSV=%s batch_size=%d interval=%.1fs max_batches=%s",
             dataset_csv, batch_size, batch_interval, max_batches)

    total_rows = 0
    for batch_id, batch in enumerate(iter_rows(dataset_csv, batch_size)):
        if max_batches is not None and batch_id >= max_batches:
            break
        path = sink.write(batch, batch_id)
        total_rows += len(batch)
        LOG.info("Flushed batch #%d (%d rows) -> %s  total=%d", batch_id, len(batch), path, total_rows)
        time.sleep(batch_interval)

    LOG.info("Streaming complete. %d rows written via %s sink.", total_rows, sink_kind)
    return 0


if __name__ == "__main__":
    sys.exit(run())
