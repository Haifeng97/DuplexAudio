from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .io import read_json


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> Dict[str, Any]:
    config = _expand(read_json(path))
    if not config.get("run_name"):
        raise ValueError("config requires run_name")
    config.setdefault("run_dir", f"runs/{config['run_name']}")
    config["config_path"] = str(path.resolve())
    return config
