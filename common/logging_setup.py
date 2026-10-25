"""Single place to configure structured logging with a predictable format."""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Level via ``LOG_LEVEL`` (default INFO)."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
