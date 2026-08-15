#!/usr/bin/env python3
"""Actual-timestamp local-polynomial filtering on :math:`R^3 x SO(3)`.

The module deliberately exposes every local fit object needed by covariance
propagation: actual sample indices, factorial design, pseudoinverse derivative
rows, coefficients, fitted values, and translation/rotation-vector residuals.
Only centered windows are used by parameter estimation; shifted windows remain
available for diagnostics at record boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


POLYNOMIAL_DEGREE = 5
MINIMUM_WINDOW_POINTS = POLYNOMIAL_DEGREE + 1


def skew(value: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=float)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def factorial_design(
    offsets_seconds: Sequence[float], scale_seconds: float, degree: int
) -> np.ndarray:
    """Return a factorial local-polynomial design on normalized time."""

    offsets = np.asarray(offsets_seconds, dtype=float)
    scale = float(scale_seconds)
    degree_value = int(degree)
    if (
        offsets.ndim != 1
        or degree_value < 2
        or offsets.size < degree_value + 1
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("local polynomial design is invalid")
    normalized = offsets / scale
    return np.column_stack(
        [
            normalized**order / math.factorial(order)
            for order in range(degree_value + 1)
        ]
    )


def left_jacobian_with_directional_derivative(
    phi: Sequence[float],
    direction: Optional[Sequence[float]] = None,
    second_direction: Optional[Sequence[float]] = None,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Return ``J_l`` and optional first/second directional derivatives.

    When both directions are supplied, the third returned value is
    ``D^2 J_l[direction, second_direction]``.  All scalar coefficients use the
    same small-angle branch so covariance propagation never needs numerical
    differentiation.
    """

    value = np.asarray(phi, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("SO(3) exponential coordinate is invalid")
    radius_squared = float(value @ value)
    wedge = skew(value)
    if radius_squared < 1.0e-8:
        s = radius_squared
        a = 0.5 - s / 24.0 + s**2 / 720.0 - s**3 / 40320.0
        b = 1.0 / 6.0 - s / 120.0 + s**2 / 5040.0 - s**3 / 362880.0
        c = -1.0 / 12.0 + s / 180.0 - s**2 / 6720.0 + s**3 / 453600.0
        d = -1.0 / 60.0 + s / 1260.0 - s**2 / 60480.0 + s**3 / 4989600.0
        e = 1.0 / 90.0 - s / 1680.0 + s**2 / 75600.0 - s**3 / 5987520.0
        f = 1.0 / 630.0 - s / 15120.0 + s**2 / 831600.0 - s**3 / 77837760.0
    else:
        radius = math.sqrt(radius_squared)
        a = (1.0 - math.cos(radius)) / radius_squared
        b = (radius - math.sin(radius)) / (radius_squared * radius)
        c = (
            radius * math.sin(radius) - 2.0 * (1.0 - math.cos(radius))
        ) / radius_squared**2
        d = (
            (1.0 - math.cos(radius)) / radius_squared**2
            - 3.0
            * (radius - math.sin(radius))
            / (radius_squared**2 * radius)
        )
        e = (
            radius_squared * math.cos(radius)
            - 5.0 * radius * math.sin(radius)
            + 8.0 * (1.0 - math.cos(radius))
        ) / radius**6
        f = (
            radius_squared * math.sin(radius)
            + 7.0 * radius * math.cos(radius)
            + 8.0 * radius
            - 15.0 * math.sin(radius)
        ) / radius**7
    jacobian = np.eye(3) + a * wedge + b * (wedge @ wedge)
    if direction is None:
        if second_direction is not None:
            raise ValueError("second SO(3) direction requires the first direction")
        return jacobian
    eta = np.asarray(direction, dtype=float)
    if eta.shape != (3,) or np.any(~np.isfinite(eta)):
        raise ValueError("SO(3) Jacobian direction is invalid")
    eta_wedge = skew(eta)
    inner = float(value @ eta)
    directional = (
        a * eta_wedge
        + b * (eta_wedge @ wedge + wedge @ eta_wedge)
        + c * inner * wedge
        + d * inner * (wedge @ wedge)
    )
    if second_direction is None:
        return jacobian, directional
    zeta = np.asarray(second_direction, dtype=float)
    if zeta.shape != (3,) or np.any(~np.isfinite(zeta)):
        raise ValueError("second SO(3) Jacobian direction is invalid")
    zeta_wedge = skew(zeta)
    phi_eta = inner
    phi_zeta = float(value @ zeta)
    eta_zeta = float(eta @ zeta)
    second = (
        c * phi_zeta * eta_wedge
        + d
        * phi_zeta
        * (eta_wedge @ wedge + wedge @ eta_wedge)
        + b * (eta_wedge @ zeta_wedge + zeta_wedge @ eta_wedge)
        + e * phi_eta * phi_zeta * wedge
        + c * eta_zeta * wedge
        + c * phi_eta * zeta_wedge
        + f * phi_eta * phi_zeta * (wedge @ wedge)
        + d * eta_zeta * (wedge @ wedge)
        + d
        * phi_eta
        * (zeta_wedge @ wedge + wedge @ zeta_wedge)
    )
    return jacobian, directional, second


@dataclass(frozen=True)
class LocalPolynomialFit:
    """One vector-valued local least-squares polynomial fit."""

    sample_indices: np.ndarray
    sample_time: np.ndarray
    query_time: float
    scale_seconds: float
    design: np.ndarray
    pseudoinverse: np.ndarray
    derivative_rows: np.ndarray
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    condition_number: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices, dtype=int)
        times = np.asarray(self.sample_time, dtype=float)
        design = np.asarray(self.design, dtype=float)
        pseudo = np.asarray(self.pseudoinverse, dtype=float)
        rows = np.asarray(self.derivative_rows, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        fitted = np.asarray(self.fitted, dtype=float)
        residual = np.asarray(self.residual, dtype=float)
        degree_plus_one = design.shape[1]
        expected = (times.size, 3)
        if (
            indices.shape != times.shape
            or design.shape[0] != times.size
            or pseudo.shape != (degree_plus_one, times.size)
            or rows.shape != (3, times.size)
            or coefficients.shape != (degree_plus_one, 3)
            or fitted.shape != expected
            or residual.shape != expected
            or np.any(~np.isfinite(times))
            or any(
                np.any(~np.isfinite(item))
                for item in (design, pseudo, rows, coefficients, fitted, residual)
            )
            or not np.isfinite(self.query_time)
            or not np.isfinite(self.scale_seconds)
            or self.scale_seconds <= 0.0
            or not np.isfinite(self.condition_number)
        ):
            raise ValueError("local polynomial fit is invalid")
        for name, value in (
            ("sample_indices", indices),
            ("sample_time", times),
            ("design", design),
            ("pseudoinverse", pseudo),
            ("derivative_rows", rows),
            ("coefficients", coefficients),
            ("fitted", fitted),
            ("residual", residual),
        ):
            object.__setattr__(self, name, _readonly(value))

    @property
    def value(self) -> np.ndarray:
        return self.coefficients[0].copy()

    @property
    def first(self) -> np.ndarray:
        return self.coefficients[1].copy()

    @property
    def second(self) -> np.ndarray:
        return self.coefficients[2].copy()


@dataclass(frozen=True)
class LocalSgWindow:
    """Translation and spatial rotation-vector fits sharing one window."""

    translation: LocalPolynomialFit
    rotation_vector: LocalPolynomialFit
    rotation_reference: np.ndarray

    def __post_init__(self) -> None:
        reference = np.asarray(self.rotation_reference, dtype=float)
        if (
            reference.shape != (3, 3)
            or np.any(~np.isfinite(reference))
            or not np.allclose(reference.T @ reference, np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(reference), 1.0, atol=1e-10)
            or not np.array_equal(
                self.translation.sample_indices,
                self.rotation_vector.sample_indices,
            )
        ):
            raise ValueError("local SG window is invalid")
        object.__setattr__(self, "rotation_reference", _readonly(reference))


@dataclass(frozen=True)
class PoseSgEvaluation:
    time: np.ndarray
    sensor_position: np.ndarray
    sensor_velocity_world: np.ndarray
    sensor_acceleration_world: np.ndarray
    body_rotation: np.ndarray
    body_angular_velocity: np.ndarray
    body_angular_acceleration: np.ndarray
    rotation_vector_coefficients: np.ndarray
    window_sample_count: np.ndarray
    position_fit_condition_number: np.ndarray
    rotation_fit_condition_number: np.ndarray
    local_windows: tuple[LocalSgWindow, ...]

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        count = time.size
        shapes = {
            "sensor_position": (count, 3),
            "sensor_velocity_world": (count, 3),
            "sensor_acceleration_world": (count, 3),
            "body_rotation": (count, 3, 3),
            "body_angular_velocity": (count, 3),
            "body_angular_acceleration": (count, 3),
            "rotation_vector_coefficients": (count, 3, 3),
            "window_sample_count": (count,),
            "position_fit_condition_number": (count,),
            "rotation_fit_condition_number": (count,),
        }
        if time.ndim != 1 or count < 1 or len(self.local_windows) != count:
            raise ValueError("SG evaluation time/local-window size is invalid")
        object.__setattr__(self, "time", _readonly(time))
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError("{} is invalid".format(name))
            object.__setattr__(self, name, _readonly(value))


# Compatibility name retained for non-legacy callers that used the former SG
# module.  The new object contains strictly more local-fit information.
PoseSplineEvaluation = PoseSgEvaluation


def _validate_pose_observations(
    time_axis: Sequence[float],
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
    pose_sensor_to_body_rotation: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_value = np.asarray(time_axis, dtype=float)
    position = np.asarray(sensor_position, dtype=float)
    orientation = np.asarray(sensor_orientation_xyzw, dtype=float)
    extrinsic = np.asarray(pose_sensor_to_body_rotation, dtype=float)
    if (
        time_value.ndim != 1
        or time_value.size < degree + 1
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
        or position.shape != (time_value.size, 3)
        or orientation.shape != (time_value.size, 4)
        or extrinsic.shape != (3, 3)
        or any(np.any(~np.isfinite(x)) for x in (position, orientation, extrinsic))
    ):
        raise ValueError("pose observations are invalid")
    # scipy 1.5's Cython memoryview rejects FlightData's read-only arrays.
    sensor_rotation = Rotation.from_quat(orientation.copy()).as_matrix()
    # R_WS R_BS^T = R_WB because the supplied extrinsic maps S coordinates
    # into B coordinates (R_BS).
    body_rotation = np.einsum("nij,jk->nik", sensor_rotation, extrinsic.T)
    return (
        time_value.copy(),
        position.copy(),
        np.asarray(body_rotation, dtype=float),
        extrinsic.copy(),
    )


def _shifted_window_bounds(
    query: float, start: float, end: float, window_seconds: float
) -> tuple[float, float]:
    half = 0.5 * window_seconds
    left = query - half
    right = query + half
    if left < start:
        right += start - left
        left = start
    if right > end:
        left -= right - end
        right = end
    return float(max(left, start)), float(min(right, end))


def window_indices(
    time_axis: Sequence[float],
    query: float,
    window_seconds: float,
    degree: int = POLYNOMIAL_DEGREE,
    centered: bool = False,
) -> np.ndarray:
    times = np.asarray(time_axis, dtype=float)
    half = 0.5 * float(window_seconds)
    if centered:
        left, right = float(query) - half, float(query) + half
        if left < times[0] or right > times[-1]:
            raise ValueError("centered SG window is outside pose support")
    else:
        left, right = _shifted_window_bounds(
            float(query), float(times[0]), float(times[-1]), float(window_seconds)
        )
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(left), abs(right))
    first = int(np.searchsorted(times, left - tolerance, side="left"))
    last = int(np.searchsorted(times, right + tolerance, side="right"))
    indices = np.arange(first, last, dtype=int)
    if indices.size < int(degree) + 1:
        raise ValueError(
            "window {:.12g}s contains {} samples; degree {} requires {}"
            .format(window_seconds, indices.size, degree, int(degree) + 1)
        )
    return indices


def fit_vector_window(
    sample_indices: Sequence[int],
    sample_time: Sequence[float],
    sample_value: np.ndarray,
    query: float,
    degree: int,
) -> LocalPolynomialFit:
    indices = np.asarray(sample_indices, dtype=int)
    times = np.asarray(sample_time, dtype=float)
    values = np.asarray(sample_value, dtype=float)
    offsets = times - float(query)
    scale = float(np.max(np.abs(offsets)))
    if values.shape != (times.size, 3) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("local polynomial window is invalid")
    design = factorial_design(offsets, scale, degree)
    normalized_coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    pseudo = np.linalg.pinv(design)
    scaling = np.ones(int(degree) + 1, dtype=float)
    for order in range(1, int(degree) + 1):
        scaling[order] = scale**order
    coefficients = normalized_coefficients / scaling[:, None]
    derivative_rows = np.vstack(
        (pseudo[0], pseudo[1] / scale, pseudo[2] / scale**2)
    )
    fitted = design @ normalized_coefficients
    return LocalPolynomialFit(
        sample_indices=indices,
        sample_time=times,
        query_time=float(query),
        scale_seconds=scale,
        design=design,
        pseudoinverse=pseudo,
        derivative_rows=derivative_rows,
        coefficients=coefficients,
        fitted=fitted,
        residual=values - fitted,
        condition_number=float(np.linalg.cond(design)),
    )


class GeometricSavitzkyGolayPose:
    """Moving-window local polynomial trajectory on :math:`R^3 x SO(3)`."""

    def __init__(
        self,
        *,
        time_axis: Sequence[float],
        sensor_position: np.ndarray,
        sensor_orientation_xyzw: np.ndarray,
        pose_sensor_to_body_rotation: Optional[np.ndarray] = None,
        body_to_pose_sensor_rotation: Optional[np.ndarray] = None,
        window_seconds: float,
        degree: int = POLYNOMIAL_DEGREE,
    ) -> None:
        degree_value = int(degree)
        if degree_value < 2:
            raise ValueError("SG degree must be at least two")
        if pose_sensor_to_body_rotation is None:
            if body_to_pose_sensor_rotation is None:
                raise ValueError("pose sensor rotation extrinsic is required")
            # Compatibility keyword from the previous local SG implementation.
            pose_sensor_to_body_rotation = body_to_pose_sensor_rotation
        elif body_to_pose_sensor_rotation is not None:
            raise ValueError("specify only one pose rotation extrinsic")
        (
            self.time_axis,
            self.observed_sensor_position,
            self.observed_body_rotation,
            self.pose_sensor_to_body_rotation,
        ) = _validate_pose_observations(
            time_axis,
            sensor_position,
            sensor_orientation_xyzw,
            np.asarray(pose_sensor_to_body_rotation, dtype=float),
            degree_value,
        )
        self.body_to_pose_sensor_rotation = self.pose_sensor_to_body_rotation
        self.degree = degree_value
        self.window_seconds = float(window_seconds)
        self.start_time = float(self.time_axis[0])
        self.end_time = float(self.time_axis[-1])
        self.valid_start_time = self.start_time + 0.5 * self.window_seconds
        self.valid_end_time = self.end_time - 0.5 * self.window_seconds
        if (
            not np.isfinite(self.window_seconds)
            or self.window_seconds <= 0.0
            or self.window_seconds > self.end_time - self.start_time
            or self.valid_start_time > self.valid_end_time
        ):
            raise ValueError("SG window is invalid for the pose interval")
        counts = [
            window_indices(
                self.time_axis, time, self.window_seconds, self.degree, False
            ).size
            for time in self.time_axis
        ]
        self.minimum_window_sample_count = int(min(counts))
        self.maximum_window_sample_count = int(max(counts))

    def centered_raw_times(
        self,
        support_start: Optional[float] = None,
        support_end: Optional[float] = None,
        require_covariance_dof: bool = True,
    ) -> np.ndarray:
        start = self.valid_start_time
        end = self.valid_end_time
        if support_start is not None:
            start = max(start, float(support_start))
        if support_end is not None:
            end = min(end, float(support_end))
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(start), abs(end))
        result = self.time_axis[
            (self.time_axis >= start - tolerance)
            & (self.time_axis <= end + tolerance)
        ]
        minimum = self.degree + (2 if require_covariance_dof else 1)
        if result.size < minimum:
            raise ValueError("centered SG support has too few evaluation times")
        if require_covariance_dof:
            for query in result:
                count = window_indices(
                    self.time_axis,
                    float(query),
                    self.window_seconds,
                    self.degree,
                    True,
                ).size
                if count <= self.degree + 1:
                    raise ValueError(
                        "SG covariance requires n > degree + 1 in every window"
                    )
        return result.copy()

    def _local_window(self, query: float, centered: bool) -> LocalSgWindow:
        indices = window_indices(
            self.time_axis,
            query,
            self.window_seconds,
            self.degree,
            centered,
        )
        translation = fit_vector_window(
            indices,
            self.time_axis[indices],
            self.observed_sensor_position[indices],
            query,
            self.degree,
        )
        nearest_local = int(
            np.argmin(np.abs(self.time_axis[indices] - float(query)))
        )
        reference = self.observed_body_rotation[indices[nearest_local]]
        relative = np.einsum(
            "nij,jk->nik", self.observed_body_rotation[indices], reference.T
        )
        rotation_vectors = Rotation.from_matrix(relative).as_rotvec()
        rotation_fit = fit_vector_window(
            indices,
            self.time_axis[indices],
            rotation_vectors,
            query,
            self.degree,
        )
        return LocalSgWindow(translation, rotation_fit, reference)

    def evaluate(
        self, query_time: Sequence[float], *, centered: bool = False,
        geometric_correction: bool = True,
    ) -> PoseSgEvaluation:
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
            raise ValueError("SG query is outside pose support")
        query = np.clip(query, self.start_time, self.end_time)
        count = query.size
        position = np.empty((count, 3))
        velocity = np.empty((count, 3))
        acceleration = np.empty((count, 3))
        rotations = np.empty((count, 3, 3))
        omega = np.empty((count, 3))
        alpha = np.empty((count, 3))
        rho = np.empty((count, 3, 3))
        sample_count = np.empty(count, dtype=int)
        p_condition = np.empty(count)
        r_condition = np.empty(count)
        local_windows: list[LocalSgWindow] = []
        for index, time_value in enumerate(query):
            local = self._local_window(float(time_value), centered)
            local_windows.append(local)
            position[index] = local.translation.value
            velocity[index] = local.translation.first
            acceleration[index] = local.translation.second
            rho0 = local.rotation_vector.value
            rho1 = local.rotation_vector.first
            rho2 = local.rotation_vector.second
            rho[index] = np.vstack((rho0, rho1, rho2))
            rotation = Rotation.from_rotvec(rho0).as_matrix() @ local.rotation_reference
            if geometric_correction:
                jacobian, directional = left_jacobian_with_directional_derivative(
                    rho0, rho1
                )
                spatial_omega = jacobian @ rho1
                spatial_alpha = directional @ rho1 + jacobian @ rho2
            else:
                spatial_omega = rho1
                spatial_alpha = rho2
            rotations[index] = rotation
            omega[index] = rotation.T @ spatial_omega
            alpha[index] = rotation.T @ spatial_alpha
            sample_count[index] = local.translation.sample_time.size
            p_condition[index] = local.translation.condition_number
            r_condition[index] = local.rotation_vector.condition_number
        return PoseSgEvaluation(
            time=query,
            sensor_position=position,
            sensor_velocity_world=velocity,
            sensor_acceleration_world=acceleration,
            body_rotation=rotations,
            body_angular_velocity=omega,
            body_angular_acceleration=alpha,
            rotation_vector_coefficients=rho,
            window_sample_count=sample_count,
            position_fit_condition_number=p_condition,
            rotation_fit_condition_number=r_condition,
            local_windows=tuple(local_windows),
        )

    def sensor_rotation(self, query_time: Sequence[float]) -> np.ndarray:
        evaluation = self.evaluate(query_time)
        return np.einsum(
            "nij,jk->nik",
            evaluation.body_rotation,
            self.pose_sensor_to_body_rotation,
        )

    def sensor_orientation_xyzw(self, query_time: Sequence[float]) -> np.ndarray:
        return Rotation.from_matrix(self.sensor_rotation(query_time)).as_quat()


QuinticLocalPolynomialPose = GeometricSavitzkyGolayPose


def window_is_feasible(
    time_axis: Sequence[float],
    window_seconds: float,
    degree: int = POLYNOMIAL_DEGREE,
    require_covariance_dof: bool = False,
) -> bool:
    times = np.asarray(time_axis, dtype=float)
    degree_value = int(degree)
    if (
        degree_value < 2
        or times.ndim != 1
        or times.size < degree_value + 1
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not np.isfinite(window_seconds)
        or window_seconds <= 0.0
        or window_seconds > times[-1] - times[0]
    ):
        return False
    required = degree_value + (2 if require_covariance_dof else 1)
    try:
        for query in times:
            if window_indices(
                times, float(query), window_seconds, degree_value, False
            ).size < required:
                return False
    except ValueError:
        return False
    centered_mask = (
        (times >= times[0] + 0.5 * window_seconds)
        & (times <= times[-1] - 0.5 * window_seconds)
    )
    return int(np.count_nonzero(centered_mask)) >= required


def minimum_feasible_window_seconds(
    time_axis: Sequence[float],
    degree: int = POLYNOMIAL_DEGREE,
    require_covariance_dof: bool = False,
) -> float:
    times = np.asarray(time_axis, dtype=float)
    degree_value = int(degree)
    required = degree_value + (2 if require_covariance_dof else 1)
    if (
        degree_value < 2
        or times.ndim != 1
        or times.size < required
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("pose timestamps cannot support requested SG degree")
    duration = float(times[-1] - times[0])
    local_requirements = []
    for query in times:
        lower, upper = 0.0, duration
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            try:
                count = window_indices(
                    times, float(query), middle, degree_value, False
                ).size
                if count >= required:
                    upper = middle
                else:
                    lower = middle
            except ValueError:
                lower = middle
        local_requirements.append(upper)
    candidate = float(np.nextafter(max(local_requirements), math.inf))
    if candidate > duration or not window_is_feasible(
        times, candidate, degree_value, require_covariance_dof
    ):
        raise ValueError("pose timestamps contain no feasible SG window")
    return candidate
