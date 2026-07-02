"""YAML config loader."""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
