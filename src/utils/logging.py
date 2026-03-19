"""Logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console handler (and optional file handler)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "run.log")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
