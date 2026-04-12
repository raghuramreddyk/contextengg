"""core/logger.py — Structured JSON logger for the entire application."""
from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with Rich console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
