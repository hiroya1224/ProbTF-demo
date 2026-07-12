from dataclasses import dataclass

import numpy as np

from probik.geometry import quat_normalize


def _vector(values, size, name):
    try:
        return np.asarray(values, dtype=float).reshape(size)
    except ValueError as exc:
        raise ValueError(f"{name} must contain {size} values.") from exc


def _symmetric_matrix(values, size, name):
    try:
        matrix = np.asarray(values, dtype=float).reshape(size, size)
    except ValueError as exc:
        raise ValueError(f"{name} must contain {size * size} values.") from exc
    return 0.5 * (matrix + matrix.T)


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


@dataclass
class ProbabilisticTransform:
    parent_frame_id: str
    child_frame_id: str
    position_mean: np.ndarray
    position_covariance: np.ndarray
    orientation_bingham: np.ndarray
    orientation_mode_wxyz: np.ndarray
    approximation_type: str = "gaussian_position_bingham_orientation"

    def __post_init__(self):
        self.parent_frame_id = str(self.parent_frame_id)
        self.child_frame_id = str(self.child_frame_id)
        self.position_mean = _vector(self.position_mean, 3, "position_mean")
        self.position_covariance = _symmetric_matrix(
            self.position_covariance,
            3,
            "position_covariance",
        )
        self.orientation_bingham = _symmetric_matrix(
            self.orientation_bingham,
            4,
            "orientation_bingham",
        )
        self.orientation_mode_wxyz = quat_normalize(self.orientation_mode_wxyz)
        self.approximation_type = str(self.approximation_type)

