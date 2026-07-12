"""ROS-independent domain models shared by ProbTF producers and consumers."""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from probtf.geometry import quat_normalize


def _vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.size != size:
        raise ValueError("{} must contain {} values.".format(name, size))
    array = array.reshape(size)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values.".format(name))
    return array


def _symmetric_matrix(values, size, name, positive_semidefinite=False):
    array = np.asarray(values, dtype=float)
    if array.size != size * size:
        raise ValueError("{} must contain {} values.".format(name, size * size))
    array = array.reshape(size, size)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values.".format(name))
    if not np.allclose(array, array.T, atol=1e-10):
        raise ValueError("{} must be symmetric.".format(name))
    array = 0.5 * (array + array.T)
    if positive_semidefinite and np.min(np.linalg.eigvalsh(array)) < -1e-10:
        raise ValueError("{} must be positive semidefinite.".format(name))
    return array


def _frame_id(value, name):
    frame_id = str(value).strip()
    if not frame_id:
        raise ValueError("{} must not be empty.".format(name))
    if frame_id.startswith("/"):
        frame_id = frame_id[1:]
    return frame_id


@dataclass
class GaussianPosition:
    """Gaussian translation expressed in the transform's parent frame."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.mean = _vector(self.mean, 3, "mean")
        self.covariance = _symmetric_matrix(
            self.covariance,
            3,
            "covariance",
            positive_semidefinite=True,
        )


@dataclass
class BinghamRotation:
    """Quaternion Bingham parameter in the ``[w, x, y, z]`` basis."""

    parameter: np.ndarray
    mode_wxyz: Optional[np.ndarray] = None
    second_moment: Optional[np.ndarray] = None
    fourth_moment: Optional[np.ndarray] = None
    moment_matched: bool = False

    def __post_init__(self):
        self.parameter = _symmetric_matrix(self.parameter, 4, "parameter")
        max_eigenvalue = float(np.max(np.linalg.eigvalsh(self.parameter)))
        self.parameter = self.parameter - max_eigenvalue * np.eye(4)

        if self.mode_wxyz is None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.parameter)
            self.mode_wxyz = eigenvectors[:, int(np.argmax(eigenvalues))]
        self.mode_wxyz = quat_normalize(self.mode_wxyz)

        if self.second_moment is not None:
            self.second_moment = _symmetric_matrix(
                self.second_moment,
                4,
                "second_moment",
                positive_semidefinite=True,
            )
            if not np.isclose(np.trace(self.second_moment), 1.0, atol=1e-6):
                raise ValueError("second_moment must have unit trace.")

        if self.fourth_moment is not None:
            fourth = np.asarray(self.fourth_moment, dtype=float)
            if fourth.shape != (4, 4, 4, 4) or not np.all(np.isfinite(fourth)):
                raise ValueError("fourth_moment must be a finite 4x4x4x4 tensor.")
            self.fourth_moment = fourth
        self.moment_matched = bool(self.moment_matched)


@dataclass
class ProbabilisticTransform:
    """Serializable summary of one directed random rigid transform.

    The rotation maps child-frame vectors into the parent frame. Translation is
    the child origin expressed in the parent frame. A reverse lookup must be a
    view of this same latent transform rather than an independent edge.
    """

    parent_frame_id: str
    child_frame_id: str
    position: GaussianPosition
    orientation: BinghamRotation
    stamp: Optional[float] = None
    edge_id: str = ""
    source_id: str = ""
    evidence_source_ids: Tuple[str, ...] = field(default_factory=tuple)
    approximation_type: str = "gaussian_position_bingham_orientation"
    closure_approximation: bool = False

    def __post_init__(self):
        self.parent_frame_id = _frame_id(self.parent_frame_id, "parent_frame_id")
        self.child_frame_id = _frame_id(self.child_frame_id, "child_frame_id")
        if self.parent_frame_id == self.child_frame_id:
            raise ValueError("parent_frame_id and child_frame_id must differ.")
        if not isinstance(self.position, GaussianPosition):
            raise TypeError("position must be a GaussianPosition.")
        if not isinstance(self.orientation, BinghamRotation):
            raise TypeError("orientation must be a BinghamRotation.")

        if self.stamp is not None:
            self.stamp = float(self.stamp)
            if not np.isfinite(self.stamp) or self.stamp < 0.0:
                raise ValueError("stamp must be a finite non-negative time in seconds.")

        self.edge_id = str(self.edge_id).strip() or "{}__to__{}".format(
            self.parent_frame_id,
            self.child_frame_id,
        )
        self.source_id = str(self.source_id).strip()
        self.evidence_source_ids = tuple(str(value).strip() for value in self.evidence_source_ids)
        if any(not value for value in self.evidence_source_ids):
            raise ValueError("evidence_source_ids must not contain empty identifiers.")
        if len(set(self.evidence_source_ids)) != len(self.evidence_source_ids):
            raise ValueError("evidence_source_ids must be unique.")
        self.approximation_type = str(self.approximation_type).strip()
        if not self.approximation_type:
            raise ValueError("approximation_type must not be empty.")
        self.closure_approximation = bool(self.closure_approximation)

    @property
    def position_mean(self):
        return self.position.mean

    @property
    def position_covariance(self):
        return self.position.covariance

    @property
    def orientation_bingham(self):
        return self.orientation.parameter

    @property
    def orientation_mode_wxyz(self):
        return self.orientation.mode_wxyz

    @classmethod
    def from_arrays(
        cls,
        parent_frame_id,
        child_frame_id,
        position_mean,
        position_covariance,
        orientation_bingham,
        orientation_mode_wxyz=None,
        **kwargs
    ):
        return cls(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            position=GaussianPosition(position_mean, position_covariance),
            orientation=BinghamRotation(orientation_bingham, orientation_mode_wxyz),
            **kwargs
        )


@dataclass
class ImuKinematics:
    """Locally fitted IMU kinematics used by relative-pose producers."""

    frame_id: str
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    specific_force: np.ndarray
    angular_velocity_covariance: np.ndarray
    angular_acceleration_covariance: np.ndarray
    specific_force_covariance: np.ndarray
    stamp: Optional[float] = None

    def __post_init__(self):
        self.frame_id = _frame_id(self.frame_id, "frame_id")
        self.angular_velocity = _vector(self.angular_velocity, 3, "angular_velocity")
        self.angular_acceleration = _vector(
            self.angular_acceleration,
            3,
            "angular_acceleration",
        )
        self.specific_force = _vector(self.specific_force, 3, "specific_force")
        self.angular_velocity_covariance = _symmetric_matrix(
            self.angular_velocity_covariance,
            3,
            "angular_velocity_covariance",
            positive_semidefinite=True,
        )
        self.angular_acceleration_covariance = _symmetric_matrix(
            self.angular_acceleration_covariance,
            3,
            "angular_acceleration_covariance",
            positive_semidefinite=True,
        )
        self.specific_force_covariance = _symmetric_matrix(
            self.specific_force_covariance,
            3,
            "specific_force_covariance",
            positive_semidefinite=True,
        )
        if self.stamp is not None:
            self.stamp = float(self.stamp)
            if not np.isfinite(self.stamp) or self.stamp < 0.0:
                raise ValueError("stamp must be a finite non-negative time in seconds.")


@dataclass(frozen=True)
class SensorMount:
    """Robot-specific placement metadata for an observation source."""

    source_id: str
    frame_id: str
    parent_frame_id: str
    position_xyz: Sequence[float] = (0.0, 0.0, 0.0)
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "source_id", str(self.source_id).strip())
        if not self.source_id:
            raise ValueError("source_id must not be empty.")
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "parent_frame_id",
            _frame_id(self.parent_frame_id, "parent_frame_id"),
        )
        object.__setattr__(self, "position_xyz", _vector(self.position_xyz, 3, "position_xyz"))
        object.__setattr__(
            self,
            "orientation_wxyz",
            quat_normalize(self.orientation_wxyz),
        )
