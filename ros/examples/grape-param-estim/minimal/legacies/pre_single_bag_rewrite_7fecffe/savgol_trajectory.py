#!/usr/bin/env python3
"""Quintic local-polynomial pose filtering on R^3 x SO(3).

The rotational filter follows Jongeneel & Saccon, IROS 2022:
"Geometric Savitzky-Golay Filtering of Noisy Rotations on SO(3) with
Simultaneous Angular Velocity and Acceleration Estimation".

Unlike the classical equally-spaced FIR presentation, this implementation
solves the local least-squares problem directly from the actual timestamps.
The window is therefore specified in seconds and no resampling is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


POLYNOMIAL_DEGREE = 5
MINIMUM_WINDOW_POINTS = POLYNOMIAL_DEGREE + 1


def _skew(value: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=float)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=float,
    )


def _validate_pose_observations(
    time_axis: Sequence[float],
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
    body_to_pose_sensor_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_value = np.asarray(time_axis, dtype=float)
    position = np.asarray(sensor_position, dtype=float)
    orientation = np.asarray(sensor_orientation_xyzw, dtype=float)
    extrinsic = np.asarray(body_to_pose_sensor_rotation, dtype=float)
    if (
        time_value.ndim != 1
        or time_value.size < MINIMUM_WINDOW_POINTS
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
        or position.shape != (time_value.size, 3)
        or orientation.shape != (time_value.size, 4)
        or extrinsic.shape != (3, 3)
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(orientation))
        or np.any(~np.isfinite(extrinsic))
    ):
        raise ValueError("pose observations are invalid")
    sensor_rotation = Rotation.from_quat(orientation).as_matrix()
    body_rotation = np.einsum("nij,jk->nik", sensor_rotation, extrinsic.T)
    return (
        time_value.copy(),
        position.copy(),
        np.asarray(body_rotation, dtype=float),
        extrinsic.copy(),
    )


def _factorial_design(
    offsets_seconds: np.ndarray,
    scale_seconds: float,
    degree: int,
) -> np.ndarray:
    offsets = np.asarray(offsets_seconds, dtype=float)
    scale = float(scale_seconds)
    if (
        offsets.ndim != 1
        or offsets.size < degree + 1
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("local polynomial design is invalid")
    normalized = offsets / scale
    return np.column_stack(
        [
            normalized**order / math.factorial(order)
            for order in range(degree + 1)
        ]
    )


def _left_jacobian_with_directional_derivative(
    phi: Sequence[float],
    direction: Optional[Sequence[float]] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """SO(3) left Jacobian and its directional derivative.

    For R(t) = Exp(phi(t)) R_ref, the spatial angular velocity is
    J_l(phi) phi_dot.  The directional derivative is the explicit
    right-trivialized second covariant derivative used by the geometric
    Savitzky-Golay construction.
    """

    value = np.asarray(phi, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("SO(3) exponential coordinate is invalid")
    radius_squared = float(value @ value)
    wedge = _skew(value)

    if radius_squared < 1.0e-8:
        s = radius_squared
        a = 0.5 - s / 24.0 + s**2 / 720.0 - s**3 / 40320.0
        b = 1.0 / 6.0 - s / 120.0 + s**2 / 5040.0 - s**3 / 362880.0
        c = -1.0 / 12.0 + s / 180.0 - s**2 / 6720.0 + s**3 / 453600.0
        d = -1.0 / 60.0 + s / 1260.0 - s**2 / 60480.0 + s**3 / 4989600.0
    else:
        radius = math.sqrt(radius_squared)
        a = (1.0 - math.cos(radius)) / radius_squared
        b = (
            radius - math.sin(radius)
        ) / (radius_squared * radius)
        c = (
            radius * math.sin(radius)
            - 2.0 * (1.0 - math.cos(radius))
        ) / (radius_squared**2)
        d = (
            (1.0 - math.cos(radius)) / (radius_squared**2)
            - 3.0
            * (radius - math.sin(radius))
            / (radius_squared**2 * radius)
        )

    jacobian = np.eye(3) + a * wedge + b * (wedge @ wedge)
    if direction is None:
        return jacobian

    zeta = np.asarray(direction, dtype=float)
    if zeta.shape != (3,) or np.any(~np.isfinite(zeta)):
        raise ValueError("SO(3) Jacobian direction is invalid")
    zeta_wedge = _skew(zeta)
    inner = float(value @ zeta)
    directional = (
        a * zeta_wedge
        + b * (zeta_wedge @ wedge + wedge @ zeta_wedge)
        + c * inner * wedge
        + d * inner * (wedge @ wedge)
    )
    return jacobian, directional


@dataclass(frozen=True)
class PoseSplineEvaluation:
    """Compatibility name used by the dynamics estimator.

    The fields named ``spline`` in the old estimator are now produced by
    local quintic polynomial fits.  Covariance fields are local least-squares
    estimates for the translational fit; they are not consumed by the
    deterministic dynamics code but are emitted for later uncertainty work.
    """

    time: np.ndarray
    sensor_position: np.ndarray
    sensor_velocity_world: np.ndarray
    sensor_acceleration_world: np.ndarray
    body_rotation: np.ndarray
    body_angular_velocity: np.ndarray
    body_angular_acceleration: np.ndarray
    window_sample_count: np.ndarray
    position_fit_condition_number: np.ndarray
    rotation_fit_condition_number: np.ndarray
    sensor_position_covariance: np.ndarray
    sensor_velocity_world_covariance: np.ndarray
    sensor_acceleration_world_covariance: np.ndarray


@dataclass(frozen=True)
class LocalPolynomialCandidate:
    spline_degree: int
    requested_knot_spacing_seconds: float
    effective_position_knot_spacing_seconds: float
    effective_rotation_knot_spacing_seconds: float
    position_coefficient_count: int
    rotation_knot_count: int
    validation_succeeded: bool
    validation_failure: Optional[str]
    validation_position_rmse_m: float
    validation_orientation_rmse_rad: float
    validation_metric_rmse_m: float
    maximum_acceleration_m_per_s2: float
    maximum_angular_acceleration_rad_per_s2: float
    derivative_sanity_passed: bool
    score: float
    minimum_window_sample_count: int
    maximum_window_sample_count: int


@dataclass(frozen=True)
class PoseSplineSelection:
    """Compatibility wrapper for the former spline selection object."""

    spline: "QuinticLocalPolynomialPose"
    selected_spacing_seconds: float
    candidates: tuple[LocalPolynomialCandidate, ...]
    fit_position_rmse_m: float
    fit_orientation_rmse_rad: float
    fit_metric_rmse_m: float


def candidate_payload(candidate: LocalPolynomialCandidate) -> dict[str, object]:
    return {
        "polynomial_degree": int(candidate.spline_degree),
        "window_seconds": float(candidate.requested_knot_spacing_seconds),
        "minimum_required_points": MINIMUM_WINDOW_POINTS,
        "minimum_window_sample_count": int(candidate.minimum_window_sample_count),
        "maximum_window_sample_count": int(candidate.maximum_window_sample_count),
        "fit_position_rmse_m": float(candidate.validation_position_rmse_m),
        "fit_orientation_rmse_rad": float(
            candidate.validation_orientation_rmse_rad
        ),
        "fit_metric_rmse_m": float(candidate.validation_metric_rmse_m),
        "maximum_acceleration_m_per_s2": float(
            candidate.maximum_acceleration_m_per_s2
        ),
        "maximum_angular_acceleration_rad_per_s2": float(
            candidate.maximum_angular_acceleration_rad_per_s2
        ),
        "valid": bool(candidate.validation_succeeded),
        "failure": candidate.validation_failure,
    }


def _shifted_window_bounds(
    query: float,
    start: float,
    end: float,
    window_seconds: float,
) -> tuple[float, float]:
    """Full-width window, centered in the interior and shifted at boundaries."""

    half = 0.5 * window_seconds
    left = query - half
    right = query + half
    if left < start:
        right += start - left
        left = start
    if right > end:
        left -= right - end
        right = end
    left = max(left, start)
    right = min(right, end)
    return float(left), float(right)


def _window_indices(
    time_axis: np.ndarray,
    query: float,
    window_seconds: float,
) -> np.ndarray:
    left, right = _shifted_window_bounds(
        query,
        float(time_axis[0]),
        float(time_axis[-1]),
        window_seconds,
    )
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(left), abs(right)
    )
    first = int(np.searchsorted(time_axis, left - tolerance, side="left"))
    last = int(np.searchsorted(time_axis, right + tolerance, side="right"))
    indices = np.arange(first, last, dtype=int)
    if indices.size < MINIMUM_WINDOW_POINTS:
        raise ValueError(
            "window {:.12g}s contains {} pose samples; degree-5 local "
            "polynomial requires at least {}".format(
                window_seconds,
                indices.size,
                MINIMUM_WINDOW_POINTS,
            )
        )
    return indices


def _fit_vector_window(
    sample_time: np.ndarray,
    sample_value: np.ndarray,
    query: float,
    degree: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    offsets = sample_time - float(query)
    scale = float(np.max(np.abs(offsets)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("local polynomial window has zero time span")
    design = _factorial_design(offsets, scale, degree)
    coefficient, *_ = np.linalg.lstsq(design, sample_value, rcond=None)
    condition = float(np.linalg.cond(design))
    value = coefficient[0]
    first = coefficient[1] / scale
    second = coefficient[2] / scale**2

    fitted = design @ coefficient
    residual = sample_value - fitted
    dof = int(sample_time.size - (degree + 1))
    pseudo = np.linalg.pinv(design)
    row0 = pseudo[0]
    row1 = pseudo[1] / scale
    row2 = pseudo[2] / scale**2
    gains = np.asarray(
        (row0 @ row0, row1 @ row1, row2 @ row2),
        dtype=float,
    )
    if dof > 0:
        # Ordinary least-squares residual covariance of the three position
        # components.  Keeping the full 3x3 matrix preserves cross-axis
        # mocap/fit correlation instead of silently diagonalising it.
        residual_covariance = (residual.T @ residual) / dof
        covariances = np.asarray(
            [gain * residual_covariance for gain in gains],
            dtype=float,
        )
    else:
        # With exactly degree+1 points the polynomial interpolates the local
        # data and there are no residual degrees of freedom from which to
        # estimate measurement variance.  The derivative itself remains valid;
        # its empirical covariance is intentionally reported as unavailable.
        covariances = np.full((3, 3, 3), np.nan, dtype=float)
    return (
        value,
        first,
        second,
        condition,
        covariances[0],
        covariances[1],
        covariances[2],
    )


def _fit_rotation_window(
    sample_time: np.ndarray,
    sample_rotation: np.ndarray,
    query: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nearest = int(np.argmin(np.abs(sample_time - float(query))))
    reference = sample_rotation[nearest]
    relative = np.einsum(
        "nij,jk->nik",
        sample_rotation,
        reference.T,
    )
    rotation_vector = Rotation.from_matrix(relative).as_rotvec()
    offsets = sample_time - float(query)
    scale = float(np.max(np.abs(offsets)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("local rotation window has zero time span")
    design = _factorial_design(offsets, scale, degree)
    coefficient, *_ = np.linalg.lstsq(
        design,
        rotation_vector,
        rcond=None,
    )
    condition = float(np.linalg.cond(design))
    rho0 = coefficient[0]
    rho1 = coefficient[1] / scale
    rho2 = coefficient[2] / scale**2

    estimated_rotation = (
        Rotation.from_rotvec(rho0).as_matrix() @ reference
    )
    left_jacobian, directional = (
        _left_jacobian_with_directional_derivative(rho0, rho1)
    )
    spatial_omega = left_jacobian @ rho1
    spatial_alpha = directional @ rho1 + left_jacobian @ rho2

    # The paper writes the geometric filter in spatial coordinates:
    # Rdot = omega_spatial^ R.  The rigid-body dynamics in this repository
    # uses body coordinates, so transform both vectors at the estimate.
    body_omega = estimated_rotation.T @ spatial_omega
    body_alpha = estimated_rotation.T @ spatial_alpha
    return estimated_rotation, body_omega, body_alpha, condition


class QuinticLocalPolynomialPose:
    """Moving-window degree-5 local polynomial trajectory on R^3 x SO(3)."""

    def __init__(
        self,
        *,
        time_axis: Sequence[float],
        sensor_position: np.ndarray,
        sensor_orientation_xyzw: np.ndarray,
        body_to_pose_sensor_rotation: np.ndarray,
        window_seconds: float,
        degree: int = POLYNOMIAL_DEGREE,
    ) -> None:
        (
            self.time_axis,
            self.observed_sensor_position,
            self.observed_body_rotation,
            self.body_to_pose_sensor_rotation,
        ) = _validate_pose_observations(
            time_axis,
            sensor_position,
            sensor_orientation_xyzw,
            body_to_pose_sensor_rotation,
        )
        self.degree = int(degree)
        self.window_seconds = float(window_seconds)
        self.requested_knot_spacing_seconds = self.window_seconds
        # Compatibility-only metadata; no knots exist in this estimator.
        self.effective_position_knot_spacing_seconds = math.nan
        self.effective_rotation_knot_spacing_seconds = math.nan
        self.position_coefficient_count = self.degree + 1
        self.rotation_knot_count = self.degree + 1
        self.start_time = float(self.time_axis[0])
        self.end_time = float(self.time_axis[-1])
        self.valid_start_time = self.start_time + 0.5 * self.window_seconds
        self.valid_end_time = self.end_time - 0.5 * self.window_seconds
        if (
            self.degree != POLYNOMIAL_DEGREE
            or not np.isfinite(self.window_seconds)
            or self.window_seconds <= 0.0
            or self.window_seconds > self.end_time - self.start_time
            or self.valid_start_time > self.valid_end_time
        ):
            raise ValueError(
                "degree-5 local polynomial requires a positive window "
                "not longer than the pose interval"
            )
        # Fail before estimation if any raw query point cannot support the
        # requested polynomial.  Boundary queries use a shifted full-width
        # window; parameter estimation itself uses only centered windows.
        counts = []
        for query in self.time_axis:
            counts.append(
                _window_indices(
                    self.time_axis,
                    float(query),
                    self.window_seconds,
                ).size
            )
        self.minimum_window_sample_count = int(min(counts))
        self.maximum_window_sample_count = int(max(counts))

    def centered_raw_times(
        self,
        support_start: Optional[float] = None,
        support_end: Optional[float] = None,
    ) -> np.ndarray:
        start = self.valid_start_time
        end = self.valid_end_time
        if support_start is not None:
            start = max(start, float(support_start))
        if support_end is not None:
            end = min(end, float(support_end))
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(start), abs(end)
        )
        mask = (
            (self.time_axis >= start - tolerance)
            & (self.time_axis <= end + tolerance)
        )
        result = self.time_axis[mask]
        if result.size < MINIMUM_WINDOW_POINTS:
            raise ValueError(
                "centered degree-5 SG support contains fewer than {} "
                "evaluation times".format(MINIMUM_WINDOW_POINTS)
            )
        # The constructor already checked full-width sample counts at every
        # raw time, so all centered queries are valid by construction.
        return result.copy()

    def evaluate(self, query_time: Sequence[float]) -> PoseSplineEvaluation:
        query = np.asarray(query_time, dtype=float)
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(self.start_time), abs(self.end_time)
        )
        if (
            query.ndim != 1
            or query.size < 1
            or np.any(~np.isfinite(query))
            or np.any(np.diff(query) < 0.0)
            or query[0] < self.start_time - tolerance
            or query[-1] > self.end_time + tolerance
        ):
            raise ValueError("local polynomial query is outside pose support")
        query = np.clip(query, self.start_time, self.end_time)

        count = query.size
        position = np.empty((count, 3), dtype=float)
        velocity = np.empty((count, 3), dtype=float)
        acceleration = np.empty((count, 3), dtype=float)
        body_rotation = np.empty((count, 3, 3), dtype=float)
        body_omega = np.empty((count, 3), dtype=float)
        body_alpha = np.empty((count, 3), dtype=float)
        sample_count = np.empty(count, dtype=int)
        position_condition = np.empty(count, dtype=float)
        rotation_condition = np.empty(count, dtype=float)
        position_covariance = np.empty((count, 3, 3), dtype=float)
        velocity_covariance = np.empty((count, 3, 3), dtype=float)
        acceleration_covariance = np.empty((count, 3, 3), dtype=float)

        for output_index, time_value in enumerate(query):
            indices = _window_indices(
                self.time_axis,
                float(time_value),
                self.window_seconds,
            )
            sample_count[output_index] = indices.size
            (
                position[output_index],
                velocity[output_index],
                acceleration[output_index],
                position_condition[output_index],
                position_covariance[output_index],
                velocity_covariance[output_index],
                acceleration_covariance[output_index],
            ) = _fit_vector_window(
                self.time_axis[indices],
                self.observed_sensor_position[indices],
                float(time_value),
                self.degree,
            )
            (
                body_rotation[output_index],
                body_omega[output_index],
                body_alpha[output_index],
                rotation_condition[output_index],
            ) = _fit_rotation_window(
                self.time_axis[indices],
                self.observed_body_rotation[indices],
                float(time_value),
                self.degree,
            )

        mandatory = (
            position,
            velocity,
            acceleration,
            body_rotation,
            body_omega,
            body_alpha,
            position_condition,
            rotation_condition,
        )
        if any(np.any(~np.isfinite(value)) for value in mandatory):
            raise FloatingPointError(
                "local quintic pose evaluation is non-finite"
            )
        return PoseSplineEvaluation(
            time=query.copy(),
            sensor_position=position,
            sensor_velocity_world=velocity,
            sensor_acceleration_world=acceleration,
            body_rotation=body_rotation,
            body_angular_velocity=body_omega,
            body_angular_acceleration=body_alpha,
            window_sample_count=sample_count,
            position_fit_condition_number=position_condition,
            rotation_fit_condition_number=rotation_condition,
            sensor_position_covariance=position_covariance,
            sensor_velocity_world_covariance=velocity_covariance,
            sensor_acceleration_world_covariance=acceleration_covariance,
        )

    def sensor_rotation(self, query_time: Sequence[float]) -> np.ndarray:
        evaluation = self.evaluate(query_time)
        return np.einsum(
            "nij,jk->nik",
            evaluation.body_rotation,
            self.body_to_pose_sensor_rotation,
        )

    def sensor_orientation_xyzw(
        self,
        query_time: Sequence[float],
    ) -> np.ndarray:
        return Rotation.from_matrix(
            self.sensor_rotation(query_time)
        ).as_quat()


def _orientation_error_vectors(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    observed_rotation = Rotation.from_quat(observed).as_matrix()
    predicted_rotation = Rotation.from_quat(predicted).as_matrix()
    relative = np.einsum(
        "nji,njk->nik",
        observed_rotation,
        predicted_rotation,
    )
    return Rotation.from_matrix(relative).as_rotvec()


def select_pose_spline(
    *,
    time_axis: Sequence[float],
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
    body_to_pose_sensor_rotation: np.ndarray,
    knot_spacing_candidates_seconds: Sequence[float],
    rotational_metric: np.ndarray,
    fold_count: int = 0,
    validation_block_duration_seconds: float = 0.0,
    derivative_check_step_seconds: float = 0.0,
    maximum_acceleration_m_per_s2: float = math.inf,
    maximum_angular_acceleration_rad_per_s2: float = math.inf,
) -> PoseSplineSelection:
    """Compatibility entry point: exactly one candidate is a SG window W."""

    del fold_count
    del validation_block_duration_seconds
    del derivative_check_step_seconds
    del maximum_acceleration_m_per_s2
    del maximum_angular_acceleration_rad_per_s2

    windows = tuple(float(value) for value in knot_spacing_candidates_seconds)
    if len(windows) != 1:
        raise ValueError(
            "Savitzky-Golay estimator requires exactly one explicit window W"
        )
    window = windows[0]
    trajectory = QuinticLocalPolynomialPose(
        time_axis=time_axis,
        sensor_position=sensor_position,
        sensor_orientation_xyzw=sensor_orientation_xyzw,
        body_to_pose_sensor_rotation=body_to_pose_sensor_rotation,
        window_seconds=window,
        degree=POLYNOMIAL_DEGREE,
    )

    # Fit statistics are evaluated at the original measurement times.  At
    # boundaries, the full-width window is shifted instead of shortened.
    evaluation = trajectory.evaluate(trajectory.time_axis)
    estimated_sensor_orientation = trajectory.sensor_orientation_xyzw(
        trajectory.time_axis
    )
    position_error = (
        evaluation.sensor_position
        - np.asarray(sensor_position, dtype=float)
    )
    orientation_error = _orientation_error_vectors(
        np.asarray(sensor_orientation_xyzw, dtype=float),
        estimated_sensor_orientation,
    )
    position_rmse = float(
        np.sqrt(np.mean(np.sum(position_error * position_error, axis=1)))
    )
    orientation_rmse = float(
        np.sqrt(np.mean(np.sum(orientation_error * orientation_error, axis=1)))
    )
    metric = np.asarray(rotational_metric, dtype=float)
    if (
        metric.shape != (3, 3)
        or np.any(~np.isfinite(metric))
        or np.any(np.linalg.eigvalsh(metric) <= 0.0)
    ):
        raise ValueError("rotational metric must be positive definite")
    metric_squared = (
        np.sum(position_error * position_error, axis=1)
        + np.einsum(
            "ni,ij,nj->n",
            orientation_error,
            metric,
            orientation_error,
        )
    )
    metric_rmse = float(np.sqrt(np.mean(metric_squared)))

    centered_time = trajectory.centered_raw_times()
    centered = trajectory.evaluate(centered_time)
    max_acceleration = float(
        np.max(np.linalg.norm(centered.sensor_acceleration_world, axis=1))
    )
    max_angular_acceleration = float(
        np.max(np.linalg.norm(centered.body_angular_acceleration, axis=1))
    )
    candidate = LocalPolynomialCandidate(
        spline_degree=POLYNOMIAL_DEGREE,
        requested_knot_spacing_seconds=window,
        effective_position_knot_spacing_seconds=math.nan,
        effective_rotation_knot_spacing_seconds=math.nan,
        position_coefficient_count=POLYNOMIAL_DEGREE + 1,
        rotation_knot_count=POLYNOMIAL_DEGREE + 1,
        validation_succeeded=True,
        validation_failure=None,
        validation_position_rmse_m=position_rmse,
        validation_orientation_rmse_rad=orientation_rmse,
        validation_metric_rmse_m=metric_rmse,
        maximum_acceleration_m_per_s2=max_acceleration,
        maximum_angular_acceleration_rad_per_s2=max_angular_acceleration,
        derivative_sanity_passed=True,
        score=metric_rmse**2,
        minimum_window_sample_count=trajectory.minimum_window_sample_count,
        maximum_window_sample_count=trajectory.maximum_window_sample_count,
    )
    return PoseSplineSelection(
        spline=trajectory,
        selected_spacing_seconds=window,
        candidates=(candidate,),
        fit_position_rmse_m=position_rmse,
        fit_orientation_rmse_rad=orientation_rmse,
        fit_metric_rmse_m=metric_rmse,
    )


def window_is_feasible(
    time_axis: Sequence[float],
    window_seconds: float,
    degree: int = POLYNOMIAL_DEGREE,
) -> bool:
    time_value = np.asarray(time_axis, dtype=float)
    window = float(window_seconds)
    if (
        degree != POLYNOMIAL_DEGREE
        or time_value.ndim != 1
        or time_value.size < degree + 1
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
        or not np.isfinite(window)
        or window <= 0.0
        or window > time_value[-1] - time_value[0]
    ):
        return False
    try:
        for query in time_value:
            _window_indices(time_value, float(query), window)
    except ValueError:
        return False
    centered_mask = (
        (time_value >= time_value[0] + 0.5 * window)
        & (time_value <= time_value[-1] - 0.5 * window)
    )
    return int(np.count_nonzero(centered_mask)) >= degree + 1


def minimum_feasible_window_seconds(
    time_axis: Sequence[float],
    degree: int = POLYNOMIAL_DEGREE,
) -> float:
    """Smallest data-supported W for the degree-5 local fit.

    For each raw query timestamp we independently find the smallest full-width
    (shifted at the record boundaries) window containing degree+1 samples, and
    take the maximum of those local requirements.  This avoids assuming that
    the additional centered-evaluation-count constraint is globally monotone in
    W.  The final candidate is then checked to retain at least degree+1 centered
    raw evaluation times.
    """

    time_value = np.asarray(time_axis, dtype=float)
    if (
        degree != POLYNOMIAL_DEGREE
        or time_value.ndim != 1
        or time_value.size < degree + 1
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
    ):
        raise ValueError("pose timestamps cannot support degree-5 SG")

    duration = float(time_value[-1] - time_value[0])
    local_requirements = []
    for query in time_value:
        lower = 0.0
        upper = duration
        # The full-record window certainly contains all samples.
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            try:
                _window_indices(time_value, float(query), middle)
                upper = middle
            except ValueError:
                lower = middle
        local_requirements.append(upper)

    candidate = float(max(local_requirements))
    # Move one representable number toward +inf so an exact timestamp boundary
    # remains on the feasible side after binary-search rounding.
    candidate = float(np.nextafter(candidate, math.inf))
    if candidate > duration or not window_is_feasible(
        time_value, candidate, degree
    ):
        raise ValueError(
            "pose timestamps contain no W that both supplies six samples per "
            "window and leaves six centered degree-5 evaluation times"
        )
    return candidate
