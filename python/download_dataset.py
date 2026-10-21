"""
Download the Bangalore City Traffic dataset from Kaggle into ../data/raw/.

Prereq:
    pip install kaggle
    Place your Kaggle API token (kaggle.json) at ~/.kaggle/kaggle.json
    (Kaggle -> Account -> Create New API Token)

Usage:
    python python/download_dataset.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "preethamgouda/banglore-city-traffic-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle", file=sys.stderr)
        print("Then place your API token at ~/.kaggle/kaggle.json", file=sys.stderr)
        return 1

    print(f"Downloading {DATASET_SLUG} -> {DATA_RAW}")
    result = subprocess.run(
        [kaggle, "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_RAW), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print("Kaggle download failed. If this is an auth error, verify ~/.kaggle/kaggle.json.", file=sys.stderr)
        return result.returncode

    files = sorted(p.name for p in DATA_RAW.iterdir() if p.is_file())
    print(f"Downloaded files: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
