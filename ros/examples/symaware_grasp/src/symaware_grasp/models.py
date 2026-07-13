from dataclasses import dataclass

import numpy as np

from probtf.geometry import quat_normalize


def _vector(values, size, name):
    try:
        return np.asarray(values, dtype=float).reshape(size)
    except ValueError as exc:
        raise ValueError(f"{name} must contain {size} values.") from exc


@dataclass
class GraspCandidate:
    grasp_id: str
    object_to_grasp_position: np.ndarray
    object_to_grasp_orientation_wxyz: np.ndarray
    approach_axis: np.ndarray
    finger_axis: np.ndarray
    weight: float = 1.0

    def __post_init__(self):
        self.grasp_id = str(self.grasp_id)
        self.object_to_grasp_position = _vector(
            self.object_to_grasp_position,
            3,
            "object_to_grasp_position",
        )
        self.object_to_grasp_orientation_wxyz = quat_normalize(
            self.object_to_grasp_orientation_wxyz
        )
        self.approach_axis = _vector(self.approach_axis, 3, "approach_axis")
        self.finger_axis = _vector(self.finger_axis, 3, "finger_axis")
        self.weight = float(self.weight)
