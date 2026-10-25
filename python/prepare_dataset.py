"""Phase 1 — Convert the raw source dataset into the canonical CSV.

Reads the committed headerless source
(``data/source/bangalore_traffic_raw.txt``, or a Kaggle named-column CSV in
``data/raw/`` if present), validates it against the canonical schema, and
writes:

    data/raw/bangalore_traffic.csv        full canonical dataset (gitignored)
    data/sample/sample_traffic.csv        small committed fixture (first rows)
    data/raw/validation_report.json       counts of clean / rejected rows

Run (host or container):
    python python/prepare_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402
from common.schema import load_any, validate  # noqa: E402

LOG = get_logger("prepare_dataset")

SAMPLE_ROWS = 60


def _find_source() -> Path:
    candidates = [
        config.SOURCE_DIR / "bangalore_traffic_raw.txt",
        config.RAW_DIR / "Banglore_traffic_Dataset.csv",
        config.RAW_DIR / "bangalore_traffic_raw.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No source dataset found. Expected one of:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPlace the raw file or run python/download_dataset.py."
    )


def main() -> int:
    source = _find_source()
    LOG.info("Source dataset: %s", source)

    report = validate(load_any(source))
    LOG.info("Validation: %s", report.summary())

    config.ensure_dir(config.RAW_DIR)
    config.ensure_dir(config.SAMPLE_DIR)

    report.clean.to_csv(config.CANONICAL_CSV, index=False)
    LOG.info("Wrote canonical dataset -> %s (%d rows)", config.CANONICAL_CSV, report.n_clean)

    sample = report.clean.head(SAMPLE_ROWS)
    sample.to_csv(config.SAMPLE_CSV, index=False)
    LOG.info("Wrote committed sample -> %s (%d rows)", config.SAMPLE_CSV, len(sample))

    if report.n_rejected:
        rejected_path = config.RAW_DIR / "rejected_rows.csv"
        report.rejected.to_csv(rejected_path, index=False)
        LOG.info("Wrote %d rejected rows -> %s", report.n_rejected, rejected_path)

    report_path = config.RAW_DIR / "validation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "n_input": report.n_input,
                "n_clean": report.n_clean,
                "n_rejected": report.n_rejected,
                "reasons": report.reasons,
                "roads": int(report.clean["road_id"].nunique()) if report.n_clean else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("Wrote validation report -> %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
