"""Download real, public traffic datasets for testing the dashboard.

Writes ready-to-upload CSVs into ``data/demo_uploads/``. The dashboard's loader
auto-detects their columns (a timestamp becomes hour + day_of_week; lat/lon are
optional; a road/segment/sensor column becomes the road id).

    python scripts/fetch_sample_datasets.py

Datasets:
  * UCI "Metro Interstate Traffic Volume" — 48k hourly rows, westbound I-94
    (Minneapolis). Columns: date_time + traffic_volume (+ weather). Public, UCI.

Other good datasets that also load (require their own download/account):
  * Kaggle "Banglore City Traffic Dataset" (preethamgouda) — the project's
    native dataset; Area/Road names + Traffic Volume + Date. Use the Kaggle CLI:
        kaggle datasets download -d preethamgouda/banglore-city-traffic-dataset --unzip
  * Dublin City SCATS volumes (data.gov.ie) and UK DfT road traffic counts
    (roadtraffic.dft.gov.uk) — need a timestamp/coords join first.
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

LOG = get_logger("fetch_datasets")

UCI_METRO_ZIP = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
UA = "Mozilla/5.0 (compatible; traffic-intelligence/1.0)"


def _download(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_metro(dest: Path) -> None:
    LOG.info("Downloading UCI Metro Interstate Traffic Volume…")
    blob = _download(UCI_METRO_ZIP)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    gz_name = next(n for n in zf.namelist() if n.endswith(".gz"))
    csv_bytes = gzip.decompress(zf.read(gz_name))
    dest.write_bytes(csv_bytes)
    LOG.info("Wrote %s (%d rows, %d bytes)", dest, csv_bytes.count(b"\n") - 1, len(csv_bytes))


def main() -> int:
    out = config.ensure_dir(config.DATA_DIR / "demo_uploads")
    try:
        fetch_metro(out / "metro_interstate_traffic_volume.csv")
    except Exception as exc:  # noqa: BLE001 - network failures are expected offline
        LOG.error("Metro download failed (%s). Check your internet connection.", exc)
        return 1
    LOG.info("Done. Upload the CSV(s) in %s via the dashboard (source = 'Upload my own CSV').", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
