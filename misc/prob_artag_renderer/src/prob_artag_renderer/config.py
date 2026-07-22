"""YAML configuration loading without importing rendering dependencies."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_PROJECT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default_scene.yaml"
_PACKAGED_CONFIG = Path(__file__).resolve().with_name("default_scene.yaml")
DEFAULT_CONFIG = _PROJECT_CONFIG if _PROJECT_CONFIG.exists() else _PACKAGED_CONFIG


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if path:
        with Path(path).open("r", encoding="utf-8") as stream:
            user = yaml.safe_load(stream) or {}
        config = _merge(config, user)
    if overrides:
        config = _merge(config, overrides)
    return config
