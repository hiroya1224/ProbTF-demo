"""Minimal JSON contract for one rosbag and one local-time interval."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SingleBagInput:
    bag_path: Path
    start_seconds: float
    end_seconds: float


def _finite_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be a finite number".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    return result


def load_single_bag_input(path: Path) -> SingleBagInput:
    """Read only ``bag_path``, ``start_seconds`` and ``end_seconds``.

    Any other JSON member is deliberately ignored.  This keeps old scientific
    options from silently changing the estimator configuration.
    """

    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("bag JSON cannot be read: {}".format(source)) from error
    if not isinstance(raw, dict):
        raise ValueError("bag JSON root must be an object")
    required = ("bag_path", "start_seconds", "end_seconds")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError("bag JSON is missing: {}".format(", ".join(missing)))
    if not isinstance(raw["bag_path"], str) or not raw["bag_path"].strip():
        raise ValueError("bag_path must be a non-empty string")
    bag_path = Path(raw["bag_path"]).expanduser()
    if not bag_path.is_absolute():
        bag_path = source.parent / bag_path
    bag_path = bag_path.resolve()
    start = _finite_seconds(raw["start_seconds"], "start_seconds")
    end = _finite_seconds(raw["end_seconds"], "end_seconds")
    if end <= start:
        raise ValueError("end_seconds must be greater than start_seconds")
    return SingleBagInput(bag_path, start, end)
