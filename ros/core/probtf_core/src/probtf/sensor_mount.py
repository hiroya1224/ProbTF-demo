"""Robot-specific placement metadata for observation sources."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from probtf.geometry import quat_normalize


def _identifier(value, name):
    result = str(value).strip()
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result


def _frame_id(value, name):
    result = _identifier(value, name).lstrip("/")
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result


def _position(values):
    array = np.asarray(values, dtype=float)
    if array.size != 3:
        raise ValueError("position_xyz must contain 3 values.")
    array = array.reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError("position_xyz must contain only finite values.")
    return array


@dataclass(frozen=True)
class SensorMount:
    """Placement of an observation frame relative to a robot frame."""

    source_id: str
    frame_id: str
    parent_frame_id: str
    position_xyz: Sequence[float] = (0.0, 0.0, 0.0)
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "parent_frame_id",
            _frame_id(self.parent_frame_id, "parent_frame_id"),
        )
        object.__setattr__(self, "position_xyz", _position(self.position_xyz))
        object.__setattr__(
            self,
            "orientation_wxyz",
            quat_normalize(self.orientation_wxyz),
        )
