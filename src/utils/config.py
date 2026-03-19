"""YAML config loading utility."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file and return as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
