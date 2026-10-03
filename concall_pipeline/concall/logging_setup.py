"""Console + file logging with timestamps."""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from .config import LOG_DIR

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("concall")
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOG_DIR / f"run_{stamp}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Third-party noise.
    for noisy in ("urllib3", "pdfminer", "PIL", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logger.info("logging to %s", fh.baseFilename)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"concall.{name}")
