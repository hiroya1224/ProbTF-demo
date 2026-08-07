#!/usr/bin/env python3
"""Pose-only continuous-time trajectory splines for minimal estimators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from grape_param_estim.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_log,
)


@dataclass(frozen=True)
class PoseSplineEvaluation:
    time: np.ndarray
    sensor_position: np.ndarray
    sensor_velocity_world: np.ndarray
    sensor_acceleration_world: np.ndarray
    body_rotation: np.ndarray
    body_angular_velocity: np.ndarray
    body_angular_acceleration: np.ndarray


@dataclass(frozen=True)
class CrossValidationCandidate:
    spline_degree: int
    requested_knot_spacing_seconds: float
    effective_position_knot_spacing_seconds: float
    effective_rotation_knot_spacing_seconds: float
    position_coefficient_count: int
    rotation_knot_count: int
    validation_position_rmse_m: float
    validation_orientation_rmse_rad: float
    validation_metric_rmse_m: float
    maximum_acceleration_m_per_s2: float
    maximum_angular_acceleration_rad_per_s2: float
    derivative_sanity_passed: bool
    score: float


@dataclass(frozen=True)
class PoseSplineSelection:
    spline: "PoseSpline"
    selected_spacing_seconds: float
    candidates: tuple[CrossValidationCandidate, ...]
    fit_position_rmse_m: float
    fit_orientation_rmse_rad: float
    fit_metric_rmse_m: float


class QuinticQuaternionSpline:
    """C4 normalized-quaternion B-spline with analytic body kinematics."""

    def __init__(self, component_spline: BSpline) -> None:
        if component_spline.k != 5:
            raise ValueError("rotation component spline must be quintic")
        self.component_spline = component_spline
        self.degree = int(component_spline.k)

    def evaluate(
        self, query_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.asarray(self.component_spline(query_time, 0), dtype=float)
        raw_first = np.asarray(
            self.component_spline(query_time, 1), dtype=float
        )
        raw_second = np.asarray(
            self.component_spline(query_time, 2), dtype=float
        )
        norm = np.linalg.norm(raw, axis=1)
        if (
            raw.shape != (query_time.size, 4)
            or np.any(~np.isfinite(raw))
            or np.any(~np.isfinite(raw_first))
            or np.any(~np.isfinite(raw_second))
            or np.any(norm <= 64.0 * np.finfo(float).eps)
        ):
            raise FloatingPointError("quintic quaternion spline is singular")
        norm_squared = norm * norm
        logarithmic_norm_rate = np.einsum(
            "ni,ni->n", raw, raw_first
        ) / norm_squared
        logarithmic_norm_acceleration = (
            (
                np.einsum("ni,ni->n", raw_first, raw_first)
                + np.einsum("ni,ni->n", raw, raw_second)
            )
            / norm_squared
            - 2.0 * logarithmic_norm_rate**2
        )
        quaternion = raw / norm[:, None]
        quaternion_first = (
            raw_first - logarithmic_norm_rate[:, None] * raw
        ) / norm[:, None]
        quaternion_second = (
            raw_second
            - 2.0 * logarithmic_norm_rate[:, None] * raw_first
            + (
                logarithmic_norm_rate**2
                - logarithmic_norm_acceleration
            )[:, None]
            * raw
        ) / norm[:, None]
        vector = quaternion[:, :3]
        scalar = quaternion[:, 3]

        def inverse_product_vector(derivative: np.ndarray) -> np.ndarray:
            return (
                scalar[:, None] * derivative[:, :3]
                - derivative[:, 3, None] * vector
                - np.cross(vector, derivative[:, :3])
            )

        body_angular_velocity = 2.0 * inverse_product_vector(
            quaternion_first
        )
        body_angular_acceleration = 2.0 * inverse_product_vector(
            quaternion_second
        )
        rotation = Rotation.from_quat(quaternion).as_matrix()
        return (
            np.asarray(rotation, dtype=float),
            body_angular_velocity,
            body_angular_acceleration,
        )


class PoseSpline:
    """Quintic position and normalized-quaternion pose spline."""

    def __init__(
        self,
        *,
        start_time: float,
        end_time: float,
        position_spline: BSpline,
        rotation_spline: QuinticQuaternionSpline,
        body_to_pose_sensor_rotation: np.ndarray,
        requested_knot_spacing_seconds: float,
        effective_position_knot_spacing_seconds: float,
        effective_rotation_knot_spacing_seconds: float,
        position_coefficient_count: int,
        rotation_knot_count: int,
    ) -> None:
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.position_spline = position_spline
        self.rotation_spline = rotation_spline
        self.body_to_pose_sensor_rotation = np.asarray(
            body_to_pose_sensor_rotation, dtype=float
        ).copy()
        self.requested_knot_spacing_seconds = float(
            requested_knot_spacing_seconds
        )
        self.effective_position_knot_spacing_seconds = float(
            effective_position_knot_spacing_seconds
        )
        self.effective_rotation_knot_spacing_seconds = float(
            effective_rotation_knot_spacing_seconds
        )
        self.position_coefficient_count = int(position_coefficient_count)
        self.rotation_knot_count = int(rotation_knot_count)
        self.degree = int(position_spline.k)
        if (
            not self.start_time < self.end_time
            or self.degree != 5
            or rotation_spline.degree != 5
            or self.body_to_pose_sensor_rotation.shape != (3, 3)
            or np.any(~np.isfinite(self.body_to_pose_sensor_rotation))
        ):
            raise ValueError("pose spline metadata is invalid")

    def evaluate(self, query_time: Sequence[float]) -> PoseSplineEvaluation:
        time_value = np.asarray(query_time, dtype=float)
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(self.start_time), abs(self.end_time)
        )
        if (
            time_value.ndim != 1
            or time_value.size < 1
            or np.any(~np.isfinite(time_value))
            or np.any(np.diff(time_value) < 0.0)
            or time_value[0] < self.start_time - tolerance
            or time_value[-1] > self.end_time + tolerance
        ):
            raise ValueError("pose spline query is outside its support")
        clipped_time = np.clip(
            time_value, self.start_time, self.end_time
        )
        sensor_position = np.asarray(
            self.position_spline(clipped_time, 0), dtype=float
        )
        sensor_velocity = np.asarray(
            self.position_spline(clipped_time, 1), dtype=float
        )
        sensor_acceleration = np.asarray(
            self.position_spline(clipped_time, 2), dtype=float
        )
        (
            body_rotation,
            body_angular_velocity,
            body_angular_acceleration,
        ) = self.rotation_spline.evaluate(clipped_time)
        arrays = (
            sensor_position,
            sensor_velocity,
            sensor_acceleration,
            body_rotation,
            body_angular_velocity,
            body_angular_acceleration,
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError("pose spline evaluation is non-finite")
        return PoseSplineEvaluation(
            time=time_value.copy(),
            sensor_position=sensor_position,
            sensor_velocity_world=sensor_velocity,
            sensor_acceleration_world=sensor_acceleration,
            body_rotation=body_rotation,
            body_angular_velocity=body_angular_velocity,
            body_angular_acceleration=body_angular_acceleration,
        )

    def sensor_rotation(self, query_time: Sequence[float]) -> np.ndarray:
        evaluation = self.evaluate(query_time)
        return np.einsum(
            "nij,jk->nik",
            evaluation.body_rotation,
            self.body_to_pose_sensor_rotation,
        )

    def sensor_orientation_xyzw(
        self, query_time: Sequence[float]
    ) -> np.ndarray:
        return np.asarray(
            [matrix_to_quaternion(value) for value in self.sensor_rotation(query_time)],
            dtype=float,
        )


def _validate_observations(
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
        or time_value.size < 5
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
    sensor_rotation = np.asarray(
        [quaternion_to_matrix(value) for value in orientation], dtype=float
    )
    body_rotation = np.einsum(
        "nij,jk->nik", sensor_rotation, extrinsic.T
    )
    return time_value, position, body_rotation, extrinsic


def _position_coefficient_count(
    time_value: np.ndarray,
    requested_spacing: float,
    degree: int,
    available_count: int,
) -> int:
    duration = float(time_value[-1] - time_value[0])
    interval_count = max(1, int(math.ceil(duration / requested_spacing)))
    return min(available_count, max(degree + 1, interval_count + degree))


def _open_uniform_knots(
    start: float,
    end: float,
    coefficient_count: int,
    degree: int,
) -> np.ndarray:
    interior_count = coefficient_count - degree - 1
    interior = (
        np.linspace(start, end, interior_count + 2)[1:-1]
        if interior_count > 0
        else np.empty(0, dtype=float)
    )
    return np.concatenate(
        (
            np.full(degree + 1, start, dtype=float),
            interior,
            np.full(degree + 1, end, dtype=float),
        )
    )


def _fit_position_spline(
    time_value: np.ndarray,
    position: np.ndarray,
    requested_spacing: float,
    degree: int,
) -> tuple[BSpline, float, int]:
    coefficient_count = _position_coefficient_count(
        time_value,
        requested_spacing,
        degree,
        time_value.size,
    )
    knots = _open_uniform_knots(
        float(time_value[0]),
        float(time_value[-1]),
        coefficient_count,
        degree,
    )
    basis = np.asarray(
        BSpline(knots, np.eye(coefficient_count), degree)(time_value),
        dtype=float,
    )
    coefficient = np.linalg.lstsq(basis, position, rcond=None)[0]
    spline = BSpline(knots, coefficient, degree, extrapolate=False)
    effective_spacing = float(time_value[-1] - time_value[0]) / max(
        1, coefficient_count - degree
    )
    return spline, effective_spacing, coefficient_count


def _fit_rotation_spline(
    time_value: np.ndarray,
    body_rotation: np.ndarray,
    requested_spacing: float,
) -> tuple[QuinticQuaternionSpline, float, int]:
    degree = 5
    quaternion = Rotation.from_matrix(body_rotation).as_quat()
    quaternion = np.asarray(quaternion, dtype=float)
    for index in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    component_spline, effective_spacing, coefficient_count = (
        _fit_position_spline(
            time_value,
            quaternion,
            requested_spacing,
            degree,
        )
    )
    return (
        QuinticQuaternionSpline(component_spline),
        effective_spacing,
        coefficient_count,
    )


def fit_pose_spline_fixed(
    *,
    time_axis: Sequence[float],
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
    body_to_pose_sensor_rotation: np.ndarray,
    knot_spacing_seconds: float,
    degree: int = 5,
) -> PoseSpline:
    time_value, position, body_rotation, extrinsic = _validate_observations(
        time_axis,
        sensor_position,
        sensor_orientation_xyzw,
        body_to_pose_sensor_rotation,
    )
    spacing = float(knot_spacing_seconds)
    if not np.isfinite(spacing) or spacing <= 0.0 or degree != 5:
        raise ValueError("only positive-spacing quintic pose splines are supported")
    position_spline, position_spacing, coefficient_count = (
        _fit_position_spline(time_value, position, spacing, degree)
    )
    rotation_spline, rotation_spacing, rotation_knot_count = (
        _fit_rotation_spline(time_value, body_rotation, spacing)
    )
    return PoseSpline(
        start_time=float(time_value[0]),
        end_time=float(time_value[-1]),
        position_spline=position_spline,
        rotation_spline=rotation_spline,
        body_to_pose_sensor_rotation=extrinsic,
        requested_knot_spacing_seconds=spacing,
        effective_position_knot_spacing_seconds=position_spacing,
        effective_rotation_knot_spacing_seconds=rotation_spacing,
        position_coefficient_count=coefficient_count,
        rotation_knot_count=rotation_knot_count,
    )


def _pose_errors(
    spline: PoseSpline,
    time_value: np.ndarray,
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
    rotational_metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluation = spline.evaluate(time_value)
    position_error = evaluation.sensor_position - observed_position
    predicted_sensor_rotation = spline.sensor_rotation(time_value)
    observed_sensor_rotation = np.asarray(
        [quaternion_to_matrix(value) for value in observed_orientation_xyzw],
        dtype=float,
    )
    orientation_error = np.asarray(
        [
            so3_log(observed_sensor_rotation[index].T @ predicted_sensor_rotation[index])
            for index in range(time_value.size)
        ],
        dtype=float,
    )
    metric_squared = (
        np.sum(position_error * position_error, axis=1)
        + np.einsum(
            "ni,ij,nj->n",
            orientation_error,
            rotational_metric,
            orientation_error,
        )
    )
    return position_error, orientation_error, metric_squared


def _blocked_validation_masks(sample_count: int, fold_count: int) -> tuple[np.ndarray, ...]:
    if sample_count < 7 or fold_count < 2:
        raise ValueError("blocked cross-validation requires more samples and folds")
    interior = np.arange(1, sample_count - 1, dtype=int)
    blocks = [value for value in np.array_split(interior, fold_count) if value.size]
    masks = []
    for block in blocks:
        mask = np.zeros(sample_count, dtype=bool)
        mask[block] = True
        masks.append(mask)
    return tuple(masks)


def select_pose_spline(
    *,
    time_axis: Sequence[float],
    sensor_position: np.ndarray,
    sensor_orientation_xyzw: np.ndarray,
    body_to_pose_sensor_rotation: np.ndarray,
    knot_spacing_candidates_seconds: Sequence[float],
    rotational_metric: np.ndarray,
    fold_count: int = 5,
    derivative_check_step_seconds: float = 0.01,
    maximum_acceleration_m_per_s2: float = 250.0,
    maximum_angular_acceleration_rad_per_s2: float = 250.0,
) -> PoseSplineSelection:
    time_value, position, _body_rotation, extrinsic = _validate_observations(
        time_axis,
        sensor_position,
        sensor_orientation_xyzw,
        body_to_pose_sensor_rotation,
    )
    orientation = np.asarray(sensor_orientation_xyzw, dtype=float)
    candidates = np.asarray(knot_spacing_candidates_seconds, dtype=float)
    metric = np.asarray(rotational_metric, dtype=float)
    if (
        candidates.ndim != 1
        or candidates.size < 1
        or np.any(~np.isfinite(candidates))
        or np.any(candidates <= 0.0)
        or metric.shape != (3, 3)
        or np.any(~np.isfinite(metric))
        or derivative_check_step_seconds <= 0.0
        or maximum_acceleration_m_per_s2 <= 0.0
        or maximum_angular_acceleration_rad_per_s2 <= 0.0
    ):
        raise ValueError("pose spline selection settings are invalid")
    masks = _blocked_validation_masks(time_value.size, fold_count)
    dense_count = max(
        3,
        int(
            math.floor(
                (time_value[-1] - time_value[0])
                / derivative_check_step_seconds
            )
        )
        + 1,
    )
    dense_time = np.linspace(time_value[0], time_value[-1], dense_count)
    records: list[CrossValidationCandidate] = []
    for spacing in candidates:
        validation_position = []
        validation_orientation = []
        validation_metric = []
        failed = False
        for holdout in masks:
            keep = ~holdout
            try:
                fold_spline = fit_pose_spline_fixed(
                    time_axis=time_value[keep],
                    sensor_position=position[keep],
                    sensor_orientation_xyzw=orientation[keep],
                    body_to_pose_sensor_rotation=extrinsic,
                    knot_spacing_seconds=float(spacing),
                )
                position_error, orientation_error, metric_squared = _pose_errors(
                    fold_spline,
                    time_value[holdout],
                    position[holdout],
                    orientation[holdout],
                    metric,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                failed = True
                break
            validation_position.append(position_error)
            validation_orientation.append(orientation_error)
            validation_metric.append(metric_squared)
        final_spline = fit_pose_spline_fixed(
            time_axis=time_value,
            sensor_position=position,
            sensor_orientation_xyzw=orientation,
            body_to_pose_sensor_rotation=extrinsic,
            knot_spacing_seconds=float(spacing),
        )
        dense = final_spline.evaluate(dense_time)
        maximum_acceleration = float(
            np.max(np.linalg.norm(dense.sensor_acceleration_world, axis=1))
        )
        maximum_angular_acceleration = float(
            np.max(np.linalg.norm(dense.body_angular_acceleration, axis=1))
        )
        sanity = bool(
            not failed
            and np.isfinite(maximum_acceleration)
            and np.isfinite(maximum_angular_acceleration)
            and maximum_acceleration <= maximum_acceleration_m_per_s2
            and maximum_angular_acceleration
            <= maximum_angular_acceleration_rad_per_s2
        )
        if failed:
            position_rmse = float("inf")
            orientation_rmse = float("inf")
            metric_rmse = float("inf")
        else:
            position_errors = np.vstack(validation_position)
            orientation_errors = np.vstack(validation_orientation)
            metric_values = np.concatenate(validation_metric)
            position_rmse = float(
                np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
            )
            orientation_rmse = float(
                np.sqrt(np.mean(np.sum(orientation_errors**2, axis=1)))
            )
            metric_rmse = float(np.sqrt(np.mean(metric_values)))
        score = metric_rmse if sanity else float("inf")
        records.append(
            CrossValidationCandidate(
                spline_degree=final_spline.degree,
                requested_knot_spacing_seconds=float(spacing),
                effective_position_knot_spacing_seconds=(
                    final_spline.effective_position_knot_spacing_seconds
                ),
                effective_rotation_knot_spacing_seconds=(
                    final_spline.effective_rotation_knot_spacing_seconds
                ),
                position_coefficient_count=(
                    final_spline.position_coefficient_count
                ),
                rotation_knot_count=final_spline.rotation_knot_count,
                validation_position_rmse_m=position_rmse,
                validation_orientation_rmse_rad=orientation_rmse,
                validation_metric_rmse_m=metric_rmse,
                maximum_acceleration_m_per_s2=maximum_acceleration,
                maximum_angular_acceleration_rad_per_s2=(
                    maximum_angular_acceleration
                ),
                derivative_sanity_passed=sanity,
                score=score,
            )
        )
    valid_indices = [
        index for index, record in enumerate(records) if np.isfinite(record.score)
    ]
    if not valid_indices:
        raise ValueError("no knot-spacing candidate passed derivative sanity")
    selected_index = min(
        valid_indices,
        key=lambda index: (
            records[index].score,
            -records[index].requested_knot_spacing_seconds,
        ),
    )
    selected_record = records[selected_index]
    selected_spline = fit_pose_spline_fixed(
        time_axis=time_value,
        sensor_position=position,
        sensor_orientation_xyzw=orientation,
        body_to_pose_sensor_rotation=extrinsic,
        knot_spacing_seconds=(
            selected_record.requested_knot_spacing_seconds
        ),
    )
    position_error, orientation_error, metric_squared = _pose_errors(
        selected_spline,
        time_value,
        position,
        orientation,
        metric,
    )
    return PoseSplineSelection(
        spline=selected_spline,
        selected_spacing_seconds=(
            selected_record.requested_knot_spacing_seconds
        ),
        candidates=tuple(records),
        fit_position_rmse_m=float(
            np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        ),
        fit_orientation_rmse_rad=float(
            np.sqrt(np.mean(np.sum(orientation_error**2, axis=1)))
        ),
        fit_metric_rmse_m=float(np.sqrt(np.mean(metric_squared))),
    )


def candidate_payload(candidate: CrossValidationCandidate) -> dict[str, Any]:
    return {
        field: getattr(candidate, field)
        for field in candidate.__dataclass_fields__
    }
