"""Prob-TF point-moment queries used by the deflecomp ROS consumer."""

from dataclasses import dataclass

import numpy as np

from probtf.distributions import DistributionStatus
from probtf.temporal import TemporalPolicy


def _vector3(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite vector with shape (3,).".format(name))
    return np.array(array, copy=True)


def _covariance3(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite matrix with shape (3, 3).")
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    if float(np.linalg.eigvalsh(matrix)[0]) < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite.")
    return matrix


@dataclass(frozen=True, eq=False)
class PointMomentObservation:
    target_frame: str
    source_frame: str
    source_point: np.ndarray
    resolved_stamp: float
    edge_ids: tuple
    path_diagnostics: tuple
    mean: np.ndarray
    covariance: np.ndarray
    approximation: object

    def __post_init__(self):
        target = str(self.target_frame).strip().strip("/")
        source = str(self.source_frame).strip().strip("/")
        if not target or not source:
            raise ValueError("target_frame and source_frame must be non-empty.")
        stamp = float(self.resolved_stamp)
        if not np.isfinite(stamp) or stamp < 0.0:
            raise ValueError("resolved_stamp must be finite and non-negative.")
        source_point = _vector3(self.source_point, "source_point")
        mean = _vector3(self.mean, "mean")
        covariance = _covariance3(self.covariance)
        source_point.setflags(write=False)
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "source_frame", source)
        object.__setattr__(self, "source_point", source_point)
        object.__setattr__(self, "resolved_stamp", stamp)
        object.__setattr__(self, "edge_ids", tuple(self.edge_ids))
        object.__setattr__(self, "path_diagnostics", tuple(self.path_diagnostics))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


def lookup_point_moment(
    listener,
    target_frame,
    source_frame,
    source_point=(0.0, 0.0, 0.0),
    policy=TemporalPolicy.LATEST_COMMON,
    tolerance=0.0,
    max_age=None,
):
    """Resolve one path and evaluate its transformed point moments."""

    if not isinstance(policy, TemporalPolicy):
        raise TypeError("policy must be TemporalPolicy.")
    point = _vector3(source_point, "source_point")
    path = listener.lookup_path(
        target_frame,
        source_frame,
        stamp=None,
        policy=policy,
        tolerance=tolerance,
        max_age=max_age,
    )
    evaluation_policy = (
        TemporalPolicy.LATEST
        if policy is TemporalPolicy.LATEST_COMMON
        else policy
    )
    result = listener.lookup_point_moments(
        target_frame,
        source_frame,
        point,
        stamp=path.resolved_stamp,
        policy=evaluation_policy,
        tolerance=tolerance,
        max_age=max_age,
    )
    if result.status is not DistributionStatus.OK:
        raise RuntimeError(
            "Point-moment evaluation returned status '{}'.".format(
                result.status.value
            )
        )
    return PointMomentObservation(
        target_frame=path.target_frame,
        source_frame=path.source_frame,
        source_point=point,
        resolved_stamp=path.resolved_stamp,
        edge_ids=tuple(view.edge_id for view in path.edge_views),
        path_diagnostics=path.diagnostics,
        mean=result.value.mean,
        covariance=result.value.covariance,
        approximation=result.approximation,
    )


def covariance_axis_segments(mean, covariance, sigma_scale=2.0):
    """Return three principal covariance line segments around ``mean``."""

    center = _vector3(mean, "mean")
    matrix = _covariance3(covariance)
    scale = float(sigma_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("sigma_scale must be finite and non-negative.")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    radii = scale * np.sqrt(np.clip(eigenvalues, 0.0, np.inf))
    return tuple(
        (
            center - radii[index] * eigenvectors[:, index],
            center + radii[index] * eigenvectors[:, index],
        )
        for index in range(3)
    )
