#!/usr/bin/env python3
"""Multi-bag gradient matching from pose-only continuous-time splines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-minimal-matplotlib")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import deterministic_continuation_estimator as continuation  # noqa: E402
import deterministic_estimator as baseline  # noqa: E402
import deterministic_multi_bag_multiple_shooting_estimator as multi  # noqa: E402
import deterministic_multiple_shooting_estimator as strict  # noqa: E402
import deterministic_smooth_lag_multiple_shooting_estimator as smooth  # noqa: E402
from smooth_command import QuinticSmoothZoh  # noqa: E402
from spline_trajectory import (  # noqa: E402
    PoseSplineEvaluation,
    PoseSplineSelection,
    candidate_payload,
    select_pose_spline,
)
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
    skew,
    so3_log,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    GRAVITY,
    ActuatorParameters,
    ActuatorState,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/minimal-deterministic-spline-dynamics/v1"
OUTPUT_SUBDIRECTORY = "deterministic_spline_dynamics"
GLOBAL_DIMENSION = strict.PHYSICAL_DIMENSION + 1
DELAY_INDEX = strict.PHYSICAL_DIMENSION
DEFAULT_ESTIMATOR_RESULT = (
    Path(__file__).resolve().parent
    / "output"
    / multi.OUTPUT_SUBDIRECTORY
    / "result.json"
)
COMPONENT_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SplineSettings:
    knot_spacing_candidates_seconds: tuple[float, ...]
    collocation_step_seconds: float


@dataclass(frozen=True)
class SplineEstimatorConfig:
    multi_bag: multi.MultiBagConfig
    spline: SplineSettings


@dataclass(frozen=True)
class InitialEstimate:
    physical_coordinate: np.ndarray
    delay_seconds: float
    source_kind: str
    source_path: Optional[Path]
    source_mass_kg: float
    selected_mass_kg: float


@dataclass(frozen=True)
class BagSplineData:
    specification: multi.BagSpecification
    normalized_weight: float
    flight: Any
    direct_problem: baseline.DirectShootingProblem
    spline_selection: PoseSplineSelection
    collocation: PoseSplineEvaluation
    rotor_history: QuinticSmoothZoh
    gimbal_history: QuinticSmoothZoh
    initial_gimbal: np.ndarray

    @property
    def collocation_time(self) -> np.ndarray:
        return self.collocation.time


@dataclass(frozen=True)
class BagDynamicsEvaluation:
    acceleration_residual: np.ndarray
    acceleration_jacobian: np.ndarray
    required_body_wrench: np.ndarray
    modeled_body_wrench: np.ndarray
    residual_body_wrench: np.ndarray
    residual_wrench_jacobian: np.ndarray
    cog_position: np.ndarray
    cog_velocity_world: np.ndarray
    cog_acceleration_world: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray

    @property
    def dynamics_loss(self) -> float:
        residual = np.asarray(self.acceleration_residual, dtype=float)
        return 0.5 * float(np.mean(np.sum(residual * residual, axis=1)))


@dataclass(frozen=True)
class JointDynamicsEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    bag_evaluations: tuple[BagDynamicsEvaluation, ...]
    decoded: Any
    physical_coordinate: np.ndarray
    delay_seconds: float
    data_loss: float
    prior_cost: float


@dataclass(frozen=True)
class DynamicsSolution:
    physical_coordinate: np.ndarray
    delay_seconds: float
    evaluation: JointDynamicsEvaluation
    optimizer: Mapping[str, Any]


class BodyWrenchHistory:
    """Linearly interpolated body-frame external force and torque history."""

    def __init__(
        self,
        time_axis: Sequence[float],
        body_wrench: np.ndarray,
    ) -> None:
        time_value = np.asarray(time_axis, dtype=float)
        wrench_value = np.asarray(body_wrench, dtype=float)
        if (
            time_value.ndim != 1
            or time_value.size < 2
            or np.any(~np.isfinite(time_value))
            or np.any(np.diff(time_value) <= 0.0)
            or wrench_value.shape != (time_value.size, 6)
            or np.any(~np.isfinite(wrench_value))
        ):
            raise ValueError("external body wrench history is invalid")
        self.time = time_value.copy()
        self.body_wrench = wrench_value.copy()

    def value_at(self, time_value: float) -> np.ndarray:
        query = float(time_value)
        if not np.isfinite(query):
            raise ValueError("external body wrench query must be finite")
        return np.asarray(
            [
                np.interp(query, self.time, self.body_wrench[:, component])
                for component in range(6)
            ],
            dtype=float,
        )

    def __call__(
        self,
        time_value: float,
        _state: RigidBodyState,
    ) -> np.ndarray:
        return self.value_at(time_value)


def load_spline_config(path: Path) -> SplineEstimatorConfig:
    config_path = path.expanduser().resolve()
    base = multi.load_multi_bag_config(config_path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        spline_raw = raw.get("spline", {})
        candidates = tuple(
            float(value)
            for value in spline_raw.get(
                "knot_spacing_candidates_seconds", (0.05, 0.1, 0.2)
            )
        )
        collocation_step = float(
            spline_raw.get("collocation_step_seconds", 0.01)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("spline settings in multi-bag config are invalid") from error
    values = np.asarray(candidates + (collocation_step,), dtype=float)
    if (
        not candidates
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("spline spacings must be finite and positive")
    return SplineEstimatorConfig(
        multi_bag=base,
        spline=SplineSettings(candidates, collocation_step),
    )


def load_initial_estimate(
    path: Optional[Path],
    fallback_delay: float,
    corrected_mass: Optional[float],
) -> InitialEstimate:
    source_path = None if path is None else path.expanduser().resolve()
    coordinate = np.zeros(strict.PHYSICAL_DIMENSION, dtype=float)
    delay = float(fallback_delay)
    source_kind = "nominal_fallback"
    if source_path is not None and source_path.is_file():
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
            selected = raw.get("selection", raw)
            coordinate = np.asarray(
                selected["physical_coordinate"], dtype=float
            )
            delay = float(selected["delay_seconds"])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "existing estimator result is not compatible: {}".format(
                    source_path
                )
            ) from error
        source_kind = "multiple_shooting_result"
    if (
        coordinate.shape != (strict.PHYSICAL_DIMENSION,)
        or np.any(~np.isfinite(coordinate))
        or not np.isfinite(delay)
        or delay < 0.0
    ):
        raise ValueError("initial estimator coordinate or delay is invalid")
    parameterization = strict.FullyPhysicalInertiaParameterization(
        VehicleParameters.nominal()
    )
    decoded = parameterization.decode(
        continuation._expand_coordinate(coordinate, delay)
    )
    source_mass = float(decoded.parameters.mass)
    selected_mass = source_mass if corrected_mass is None else float(corrected_mass)
    if not np.isfinite(selected_mass) or selected_mass <= 0.0:
        raise ValueError("corrected mass must be finite and positive")
    coordinate = coordinate.copy()
    coordinate[0] = math.log(selected_mass / VehicleParameters.nominal().mass)
    return InitialEstimate(
        physical_coordinate=coordinate,
        delay_seconds=delay,
        source_kind=source_kind,
        source_path=(source_path if source_kind != "nominal_fallback" else None),
        source_mass_kg=source_mass,
        selected_mass_kg=selected_mass,
    )


def _collocation_grid(
    start: float,
    end: float,
    step: float,
) -> np.ndarray:
    count = int(math.floor((end - start) / step + 1.0e-12)) + 1
    if count < 3:
        raise ValueError("collocation support is too short")
    return start + step * np.arange(count, dtype=float)


def cog_kinematics_from_pose_spline(
    evaluation: PoseSplineEvaluation,
    pose_sensor_position_in_body: Sequence[float],
    cog_offset_in_body: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sensor-pose spline derivatives to CoG world kinematics."""

    sensor_offset = np.asarray(pose_sensor_position_in_body, dtype=float)
    cog_offset = np.asarray(cog_offset_in_body, dtype=float)
    if (
        sensor_offset.shape != (3,)
        or cog_offset.shape != (3,)
        or np.any(~np.isfinite(sensor_offset))
        or np.any(~np.isfinite(cog_offset))
    ):
        raise ValueError("sensor and CoG offsets must be finite 3-vectors")
    lever = sensor_offset - cog_offset
    rotation = evaluation.body_rotation
    omega = evaluation.body_angular_velocity
    alpha = evaluation.body_angular_acceleration
    rotational_velocity = np.cross(omega, lever)
    rotational_acceleration = np.cross(alpha, lever) + np.cross(
        omega, np.cross(omega, lever)
    )
    position = evaluation.sensor_position - np.einsum(
        "nij,j->ni", rotation, lever
    )
    velocity = evaluation.sensor_velocity_world - np.einsum(
        "nij,nj->ni", rotation, rotational_velocity
    )
    acceleration = evaluation.sensor_acceleration_world - np.einsum(
        "nij,nj->ni", rotation, rotational_acceleration
    )
    return position, velocity, acceleration


def _build_bag_data(
    specification: multi.BagSpecification,
    normalized_weight: float,
    flight: Any,
    initial_delay: float,
    settings: SplineSettings,
    arguments: argparse.Namespace,
) -> BagSplineData:
    direct = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=initial_delay,
        prior_weight=0.0,
    )
    nominal = VehicleParameters.nominal()
    selection = select_pose_spline(
        time_axis=direct.output_time,
        sensor_position=direct.observations.sensor_position,
        sensor_orientation_xyzw=(
            direct.observations.sensor_orientation_xyzw
        ),
        body_to_pose_sensor_rotation=(
            direct.pose_body_to_sensor_rotation
        ),
        knot_spacing_candidates_seconds=(
            settings.knot_spacing_candidates_seconds
        ),
        rotational_metric=nominal.inertia / nominal.mass,
        fold_count=arguments.spline_cv_folds,
        derivative_check_step_seconds=settings.collocation_step_seconds,
        maximum_acceleration_m_per_s2=(
            arguments.maximum_spline_acceleration
        ),
        maximum_angular_acceleration_rad_per_s2=(
            arguments.maximum_spline_angular_acceleration
        ),
    )
    collocation_time = _collocation_grid(
        float(direct.output_time[0]),
        float(direct.output_time[-1]),
        settings.collocation_step_seconds,
    )
    collocation = selection.spline.evaluate(collocation_time)
    initial_gimbal = baseline._linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((collocation_time[0],), dtype=float),
    )[0]
    return BagSplineData(
        specification=specification,
        normalized_weight=float(normalized_weight),
        flight=flight,
        direct_problem=direct,
        spline_selection=selection,
        collocation=collocation,
        rotor_history=QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        ),
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        ),
        initial_gimbal=initial_gimbal,
    )


class SplineDynamicsProblem:
    """Shared physical parameters against fixed bag-local pose splines."""

    def __init__(
        self,
        bags: Sequence[BagSplineData],
        prior_weight: float,
    ) -> None:
        self.bags = tuple(bags)
        if not self.bags:
            raise ValueError("spline dynamics requires at least one bag")
        weights = np.asarray(
            [bag.normalized_weight for bag in self.bags], dtype=float
        )
        if not np.isclose(np.sum(weights), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("normalized bag weights must sum to one")
        self.prior_weight = float(prior_weight)
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0.0:
            raise ValueError("prior weight must be finite and nonnegative")
        self.prior_scales = strict.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS.copy()
        self.parameterization = strict.FullyPhysicalInertiaParameterization(
            VehicleParameters.nominal()
        )
        nominal = VehicleParameters.nominal()
        self.angular_factor = np.linalg.cholesky(
            nominal.inertia / nominal.mass
        ).T

    def _decode(
        self,
        physical_coordinate: np.ndarray,
        delay: float,
        dimension: int,
    ) -> tuple[Any, Any]:
        decoded, jacobian = strict._physical_parameter_jacobian(
            self.parameterization,
            physical_coordinate,
            delay,
        )
        return decoded, strict._extend_parameter_jacobian(jacobian, dimension)

    @staticmethod
    def _command(
        bag: BagSplineData,
        time_value: float,
        delay: float,
        smooth_mode: bool,
        width_fraction: float,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        if smooth_mode:
            rotor = bag.rotor_history.evaluate(
                time_value, delay, width_fraction
            )
            gimbal = bag.gimbal_history.evaluate(
                time_value, delay, width_fraction
            )
            return (
                smooth._command(rotor.value, gimbal.value),
                rotor.delay_derivative,
                gimbal.delay_derivative,
            )
        rotor_value = bag.rotor_history.exact_zoh(time_value, delay)
        gimbal_value = bag.gimbal_history.exact_zoh(time_value, delay)
        return (
            smooth._command(rotor_value, gimbal_value),
            np.zeros(4, dtype=float),
            np.zeros(4, dtype=float),
        )

    def _actuator_series(
        self,
        bag: BagSplineData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_axis = bag.collocation_time
        delay = float(decoded.delay)
        initial_command, rotor_derivative, _gimbal_derivative = self._command(
            bag,
            float(time_axis[0]),
            delay,
            smooth_mode,
            width_fraction,
        )
        state = ActuatorState(
            thrust=initial_command.thrust,
            gimbal_angle=bag.initial_gimbal,
        )
        sensitivity = np.zeros((8, dimension), dtype=float)
        if smooth_mode:
            sensitivity[:4, DELAY_INDEX] = rotor_derivative
        thrust = np.empty((time_axis.size, 4), dtype=float)
        gimbal = np.empty((time_axis.size, 4), dtype=float)
        state_jacobian = np.empty((time_axis.size, 8, dimension), dtype=float)
        for index, sample_time in enumerate(time_axis):
            thrust[index] = state.thrust
            gimbal[index] = state.gimbal_angle
            state_jacobian[index] = sensitivity
            if index == time_axis.size - 1:
                break
            dt = float(time_axis[index + 1] - sample_time)
            midpoint_time = float(sample_time + 0.5 * dt)
            command, thrust_derivative, gimbal_derivative = self._command(
                bag,
                midpoint_time,
                delay,
                smooth_mode,
                width_fraction,
            )
            command_sensitivity = np.zeros((8, dimension), dtype=float)
            if smooth_mode:
                command_sensitivity[:4, DELAY_INDEX] = thrust_derivative
                command_sensitivity[4:, DELAY_INDEX] = gimbal_derivative
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state,
                sensitivity,
                command,
                decoded,
                parameter_jacobian,
                0.5 * dt,
                command_sensitivity,
            )
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state,
                sensitivity,
                command,
                decoded,
                parameter_jacobian,
                0.5 * dt,
                command_sensitivity,
            )
        return thrust, gimbal, state_jacobian

    def _evaluate_bag(
        self,
        bag: BagSplineData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> BagDynamicsEvaluation:
        parameters = decoded.parameters
        collocation = bag.collocation
        count = collocation.time.size
        thrust, gimbal, actuator_jacobian = self._actuator_series(
            bag,
            decoded,
            parameter_jacobian,
            dimension,
            smooth_mode,
            width_fraction,
        )
        acceleration_residual = np.empty((count, 6), dtype=float)
        acceleration_jacobian = np.empty((count, 6, dimension), dtype=float)
        required_wrench = np.empty((count, 6), dtype=float)
        modeled_wrench = np.empty((count, 6), dtype=float)
        residual_wrench = np.empty((count, 6), dtype=float)
        residual_wrench_jacobian = np.empty(
            (count, 6, dimension), dtype=float
        )
        cog_position = np.empty((count, 3), dtype=float)
        cog_velocity = np.empty((count, 3), dtype=float)
        cog_acceleration = np.empty((count, 3), dtype=float)
        zero_rotation_sensitivity = np.zeros((3, dimension), dtype=float)
        zero_omega_sensitivity = np.zeros((3, dimension), dtype=float)
        gravity_world = np.asarray((0.0, 0.0, -GRAVITY), dtype=float)
        cog_position[:], cog_velocity[:], cog_acceleration[:] = (
            cog_kinematics_from_pose_spline(
                collocation,
                bag.direct_problem.pose_sensor_position,
                parameters.cog_offset,
            )
        )
        for index in range(count):
            rotation = collocation.body_rotation[index]
            omega = collocation.body_angular_velocity[index]
            alpha = collocation.body_angular_acceleration[index]
            lever = (
                bag.direct_problem.pose_sensor_position
                - parameters.cog_offset
            )
            angular_kinematics = (
                skew(alpha) + skew(omega) @ skew(omega)
            )
            velocity = cog_velocity[index]
            acceleration = cog_acceleration[index]
            velocity_sensitivity = (
                rotation
                @ skew(omega)
                @ parameter_jacobian.cog_offset
            )
            acceleration_body = rotation.T @ (
                acceleration - gravity_world
            )
            acceleration_body_sensitivity = (
                angular_kinematics @ parameter_jacobian.cog_offset
            )
            actuators = ActuatorState(
                thrust=thrust[index],
                gimbal_angle=gimbal[index],
            )
            wrench, wrench_jacobian = strict._body_wrench_with_sensitivity(
                bag.direct_problem,
                decoded,
                parameter_jacobian,
                rotation,
                zero_rotation_sensitivity,
                velocity,
                velocity_sensitivity,
                omega,
                zero_omega_sensitivity,
                actuators,
                actuator_jacobian[index],
            )
            force = wrench[:3]
            force_jacobian = wrench_jacobian[:3]
            predicted_acceleration = force / parameters.mass
            predicted_acceleration_jacobian = (
                force_jacobian / parameters.mass
                - np.outer(force, parameter_jacobian.mass)
                / parameters.mass**2
            )
            inertia_omega = parameters.inertia @ omega
            angular_rhs = wrench[3:] - np.cross(omega, inertia_omega)
            predicted_alpha = np.linalg.solve(
                parameters.inertia, angular_rhs
            )
            inertia_omega_jacobian = np.einsum(
                "ijk,j->ik", parameter_jacobian.inertia, omega
            )
            angular_rhs_jacobian = (
                wrench_jacobian[3:]
                - skew(omega) @ inertia_omega_jacobian
            )
            inertia_alpha_jacobian = np.einsum(
                "ijk,j->ik", parameter_jacobian.inertia, predicted_alpha
            )
            predicted_alpha_jacobian = np.linalg.solve(
                parameters.inertia,
                angular_rhs_jacobian - inertia_alpha_jacobian,
            )
            linear_error = acceleration_body - predicted_acceleration
            angular_error = alpha - predicted_alpha
            acceleration_residual[index, :3] = linear_error
            acceleration_residual[index, 3:] = (
                self.angular_factor @ angular_error
            )
            acceleration_jacobian[index, :3] = (
                acceleration_body_sensitivity
                - predicted_acceleration_jacobian
            )
            acceleration_jacobian[index, 3:] = (
                -self.angular_factor @ predicted_alpha_jacobian
            )
            required_force = parameters.mass * acceleration_body
            required_force_jacobian = (
                np.outer(acceleration_body, parameter_jacobian.mass)
                + parameters.mass * acceleration_body_sensitivity
            )
            required_torque = (
                parameters.inertia @ alpha
                + np.cross(omega, inertia_omega)
            )
            required_torque_jacobian = (
                np.einsum("ijk,j->ik", parameter_jacobian.inertia, alpha)
                + skew(omega) @ inertia_omega_jacobian
            )
            required_wrench[index, :3] = required_force
            required_wrench[index, 3:] = required_torque
            modeled_wrench[index] = wrench
            residual_wrench[index] = required_wrench[index] - wrench
            residual_wrench_jacobian[index, :3] = (
                required_force_jacobian - force_jacobian
            )
            residual_wrench_jacobian[index, 3:] = (
                required_torque_jacobian - wrench_jacobian[3:]
            )
        arrays = (
            acceleration_residual,
            acceleration_jacobian,
            required_wrench,
            modeled_wrench,
            residual_wrench,
            residual_wrench_jacobian,
            cog_position,
            cog_velocity,
            cog_acceleration,
            thrust,
            gimbal,
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError("spline dynamics evaluation is non-finite")
        return BagDynamicsEvaluation(
            acceleration_residual=acceleration_residual,
            acceleration_jacobian=acceleration_jacobian,
            required_body_wrench=required_wrench,
            modeled_body_wrench=modeled_wrench,
            residual_body_wrench=residual_wrench,
            residual_wrench_jacobian=residual_wrench_jacobian,
            cog_position=cog_position,
            cog_velocity_world=cog_velocity,
            cog_acceleration_world=cog_acceleration,
            actuator_thrust=thrust,
            actuator_gimbal=gimbal,
        )

    def evaluate_smooth(
        self,
        coordinate: Sequence[float],
        width_fraction: float,
    ) -> JointDynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (GLOBAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("smooth spline-dynamics coordinate must be 14-D")
        return self._evaluate_joint(
            physical_coordinate=value[: strict.PHYSICAL_DIMENSION],
            delay=float(value[DELAY_INDEX]),
            dimension=GLOBAL_DIMENSION,
            smooth_mode=True,
            width_fraction=float(width_fraction),
        )

    def evaluate_strict(
        self,
        physical_coordinate: Sequence[float],
        delay: float,
    ) -> JointDynamicsEvaluation:
        physical = np.asarray(physical_coordinate, dtype=float)
        if (
            physical.shape != (strict.PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(physical))
            or not np.isfinite(delay)
            or delay < 0.0
        ):
            raise ValueError("strict spline-dynamics coordinate is invalid")
        return self._evaluate_joint(
            physical_coordinate=physical,
            delay=float(delay),
            dimension=strict.PHYSICAL_DIMENSION,
            smooth_mode=False,
            width_fraction=1.0,
        )

    def _evaluate_joint(
        self,
        *,
        physical_coordinate: np.ndarray,
        delay: float,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> JointDynamicsEvaluation:
        decoded, parameter_jacobian = self._decode(
            physical_coordinate, delay, dimension
        )
        residual_blocks = []
        jacobian_blocks = []
        bag_evaluations = []
        data_loss = 0.0
        for bag in self.bags:
            evaluation = self._evaluate_bag(
                bag,
                decoded,
                parameter_jacobian,
                dimension,
                smooth_mode,
                width_fraction,
            )
            bag_evaluations.append(evaluation)
            root_scale = math.sqrt(
                bag.normalized_weight / bag.collocation_time.size
            )
            residual_blocks.append(
                root_scale * evaluation.acceleration_residual.ravel()
            )
            jacobian_blocks.append(
                root_scale
                * evaluation.acceleration_jacobian.reshape(-1, dimension)
            )
            data_loss += bag.normalized_weight * evaluation.dynamics_loss
        prior_residual = (
            math.sqrt(self.prior_weight)
            * physical_coordinate
            / self.prior_scales
        )
        prior_jacobian = np.zeros(
            (strict.PHYSICAL_DIMENSION, dimension), dtype=float
        )
        prior_jacobian[:, : strict.PHYSICAL_DIMENSION] = np.diag(
            math.sqrt(self.prior_weight) / self.prior_scales
        )
        residual_blocks.append(prior_residual)
        jacobian_blocks.append(prior_jacobian)
        residual = np.concatenate(residual_blocks)
        jacobian = np.vstack(jacobian_blocks)
        prior_cost = 0.5 * float(prior_residual @ prior_residual)
        if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(jacobian)):
            raise FloatingPointError("joint spline dynamics is non-finite")
        return JointDynamicsEvaluation(
            residual=residual,
            jacobian=jacobian,
            bag_evaluations=tuple(bag_evaluations),
            decoded=decoded,
            physical_coordinate=physical_coordinate.copy(),
            delay_seconds=delay,
            data_loss=float(data_loss),
            prior_cost=prior_cost,
        )


class _CachedObjective:
    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.coordinate: Optional[np.ndarray] = None
        self.evaluation: Optional[JointDynamicsEvaluation] = None

    def _get(self, coordinate: np.ndarray) -> JointDynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if self.coordinate is None or not np.array_equal(value, self.coordinate):
            self.coordinate = value.copy()
            self.evaluation = self.evaluator(value)
        if self.evaluation is None:
            raise RuntimeError("cached objective has no evaluation")
        return self.evaluation

    def residual(self, coordinate: np.ndarray) -> np.ndarray:
        return self._get(coordinate).residual

    def jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        return self._get(coordinate).jacobian


def _optimizer_payload(result: Any) -> dict[str, Any]:
    return {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
    }


def _physical_bounds(
    initial_coordinate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isinf(scale):
        return (
            np.full(strict.PHYSICAL_DIMENSION, -np.inf),
            np.full(strict.PHYSICAL_DIMENSION, np.inf),
        )
    span = scale * strict.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS
    return initial_coordinate - span, initial_coordinate + span


def _solve_smooth(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    width_fraction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, JointDynamicsEvaluation, Mapping[str, Any]]:
    objective = _CachedObjective(
        lambda value: problem.evaluate_smooth(value, width_fraction)
    )
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.smooth_max_nfev,
        verbose=0,
    )
    return result.x, problem.evaluate_smooth(result.x, width_fraction), _optimizer_payload(result)


def _solve_strict(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    delay: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> DynamicsSolution:
    objective = _CachedObjective(
        lambda value: problem.evaluate_strict(value, delay)
    )
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev,
        verbose=0,
    )
    evaluation = problem.evaluate_strict(result.x, delay)
    return DynamicsSolution(
        physical_coordinate=result.x.copy(),
        delay_seconds=float(delay),
        evaluation=evaluation,
        optimizer=_optimizer_payload(result),
    )


def forward_rollout(
    bag: BagSplineData,
    physical_coordinate: Sequence[float],
    delay: float,
    external_body_wrench: Optional[BodyWrenchHistory] = None,
) -> baseline.Simulation:
    """Strict-ZOH rollout, optionally driven by an external body wrench."""

    if external_body_wrench is not None and not isinstance(
        external_body_wrench, BodyWrenchHistory
    ):
        raise TypeError("external_body_wrench must be BodyWrenchHistory")

    parameterization = strict.FullyPhysicalInertiaParameterization(
        VehicleParameters.nominal()
    )
    decoded = parameterization.decode(
        continuation._expand_coordinate(physical_coordinate, delay)
    )
    parameters = decoded.parameters
    direct = bag.direct_problem
    initial_spline = bag.spline_selection.spline.evaluate(
        np.asarray((direct.internal_time[0],), dtype=float)
    )
    rotation = initial_spline.body_rotation[0]
    omega = initial_spline.body_angular_velocity[0]
    lever_pose = direct.pose_sensor_position - parameters.cog_offset
    rigid = RigidBodyState(
        position=(
            initial_spline.sensor_position[0] - rotation @ lever_pose
        ),
        orientation_xyzw=matrix_to_quaternion(rotation),
        linear_velocity=(
            initial_spline.sensor_velocity_world[0]
            - rotation @ np.cross(omega, lever_pose)
        ),
        angular_velocity=omega,
    )
    actuator_parameters = ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=0.0,
        delay=0.0,
    )
    actuators = ActuatorState(
        thrust=bag.rotor_history.exact_zoh(
            float(direct.internal_time[0]), delay
        ),
        gimbal_angle=bag.initial_gimbal,
    )
    plant = FullSixDofPlant(
        parameters,
        direct.geometry,
        model_discrepancy_wrench=external_body_wrench,
    )
    output_count = direct.output_time.size
    arrays = {
        "sensor_position": np.empty((output_count, 3)),
        "sensor_orientation_xyzw": np.empty((output_count, 4)),
        "sensor_velocity_world": np.empty((output_count, 3)),
        "angular_velocity_sensor": np.empty((output_count, 3)),
        "specific_force_sensor": np.empty((output_count, 3)),
        "cog_position": np.empty((output_count, 3)),
        "cog_velocity_world": np.empty((output_count, 3)),
        "actuator_thrust": np.empty((output_count, 4)),
        "actuator_gimbal": np.empty((output_count, 4)),
    }

    def store(output_index: int, simulation_time: float) -> None:
        body_rotation = quaternion_to_matrix(rigid.orientation_xyzw)
        pose_lever = direct.pose_sensor_position - parameters.cog_offset
        velocity_lever = (
            direct.velocity_sensor_position - parameters.cog_offset
        )
        imu_lever = direct.imu_sensor_position - parameters.cog_offset
        wrench = plant.total_body_wrench(simulation_time, rigid, actuators)
        angular_acceleration = np.linalg.solve(
            parameters.inertia,
            wrench[3:]
            - np.cross(
                rigid.angular_velocity,
                parameters.inertia @ rigid.angular_velocity,
            ),
        )
        specific_force_body = (
            wrench[:3] / parameters.mass
            + np.cross(angular_acceleration, imu_lever)
            + np.cross(
                rigid.angular_velocity,
                np.cross(rigid.angular_velocity, imu_lever),
            )
        )
        arrays["sensor_position"][output_index] = (
            rigid.position + body_rotation @ pose_lever
        )
        arrays["sensor_orientation_xyzw"][output_index] = (
            matrix_to_quaternion(
                body_rotation @ direct.pose_body_to_sensor_rotation
            )
        )
        arrays["sensor_velocity_world"][output_index] = (
            rigid.linear_velocity
            + body_rotation
            @ np.cross(rigid.angular_velocity, velocity_lever)
        )
        arrays["angular_velocity_sensor"][output_index] = (
            direct.body_to_imu_rotation @ rigid.angular_velocity
            + direct.gyro_bias
        )
        arrays["specific_force_sensor"][output_index] = (
            direct.body_to_imu_rotation @ specific_force_body
            + direct.accelerometer_bias
        )
        arrays["cog_position"][output_index] = rigid.position
        arrays["cog_velocity_world"][output_index] = rigid.linear_velocity
        arrays["actuator_thrust"][output_index] = actuators.thrust
        arrays["actuator_gimbal"][output_index] = actuators.gimbal_angle

    store(0, float(direct.internal_time[0]))
    output_index = 1
    for step_index in range(direct.internal_time.size - 1):
        start = float(direct.internal_time[step_index])
        dt = direct.integration_step
        midpoint = start + 0.5 * dt
        command = smooth._command(
            bag.rotor_history.exact_zoh(midpoint, delay),
            bag.gimbal_history.exact_zoh(midpoint, delay),
        )
        midpoint_actuators = advance_actuators(
            actuators, command, actuator_parameters, 0.5 * dt
        )
        rigid = plant.step(start, rigid, midpoint_actuators, dt)
        actuators = advance_actuators(
            midpoint_actuators, command, actuator_parameters, 0.5 * dt
        )
        if (step_index + 1) % direct.output_stride == 0:
            store(output_index, start + dt)
            output_index += 1
    if output_index != output_count:
        raise RuntimeError("forward rollout grids disagree")
    return baseline.Simulation(time=direct.output_time, **arrays)


def _orientation_errors(
    observed_xyzw: np.ndarray,
    predicted_xyzw: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            so3_log(
                quaternion_to_matrix(observed).T
                @ quaternion_to_matrix(predicted)
            )
            for observed, predicted in zip(observed_xyzw, predicted_xyzw)
        ],
        dtype=float,
    )


def _pose_metrics(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
) -> dict[str, Any]:
    position_error = simulation.sensor_position - observations.sensor_position
    orientation_error = _orientation_errors(
        observations.sensor_orientation_xyzw,
        simulation.sensor_orientation_xyzw,
    )
    position_norm = np.linalg.norm(position_error, axis=1)
    orientation_norm = np.linalg.norm(orientation_error, axis=1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_norm**2))),
        "position_component_rmse_m": np.sqrt(
            np.mean(position_error**2, axis=0)
        ),
        "orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(orientation_norm**2))
        ),
        "orientation_angle_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(orientation_norm**2)))
        ),
        "terminal_position_error_m": float(position_norm[-1]),
        "terminal_orientation_error_deg": float(
            np.degrees(orientation_norm[-1])
        ),
    }


def _axis_validation_statistics(
    time_axis: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    dt = float(np.median(np.diff(time_axis)))
    statistics = []
    for component in range(3):
        reference = np.asarray(observed[:, component], dtype=float)
        estimate = np.asarray(predicted[:, component], dtype=float)
        error = estimate - reference
        centered_reference = reference - np.mean(reference)
        centered_estimate = estimate - np.mean(estimate)
        denominator = float(
            np.linalg.norm(centered_reference)
            * np.linalg.norm(centered_estimate)
        )
        correlation = (
            float(np.dot(centered_reference, centered_estimate) / denominator)
            if denominator > 1.0e-15
            else float("nan")
        )
        cross = np.correlate(
            centered_estimate, centered_reference, mode="full"
        )
        lags = np.arange(-reference.size + 1, reference.size, dtype=int)
        maximum_index = int(np.argmax(cross))
        statistics.append(
            {
                "axis": COMPONENT_NAMES[component],
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_bias": float(np.mean(error)),
                "pearson_correlation": correlation,
                "maximum_cross_correlation_time_shift_seconds": float(
                    lags[maximum_index] * dt
                ),
            }
        )
    return tuple(statistics)


def _sensor_metrics(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
) -> dict[str, Any]:
    return {
        "world_velocity": _axis_validation_statistics(
            observations.time,
            observations.sensor_velocity_world,
            simulation.sensor_velocity_world,
        ),
        "gyro": _axis_validation_statistics(
            observations.time,
            observations.angular_velocity_sensor,
            simulation.angular_velocity_sensor,
        ),
        "specific_force": _axis_validation_statistics(
            observations.time,
            observations.specific_force_sensor,
            simulation.specific_force_sensor,
        ),
    }


def _wrench_statistics(
    time_axis: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    relative_time = np.asarray(time_axis, dtype=float) - float(time_axis[0])
    duration = float(relative_time[-1])
    centered_time = relative_time - np.mean(relative_time)
    time_energy = float(centered_time @ centered_time)
    slopes = (
        centered_time @ (residual - np.mean(residual, axis=0)) / time_energy
        if time_energy > 0.0
        else np.zeros(6, dtype=float)
    )
    integral = np.trapz(residual, time_axis, axis=0)
    return {
        "component_names": ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z"),
        "mean": np.mean(residual, axis=0),
        "rmse": np.sqrt(np.mean(residual**2, axis=0)),
        "linear_trend_per_second": slopes,
        "cumulative_impulse": integral,
        "duration_seconds": duration,
    }


def _common_3d_limits(*series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.vstack(series)
    lower = np.min(joined, axis=0)
    upper = np.max(joined, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 1.0e-3) * 1.05
    return center - radius, center + radius


def _write_trajectory_pdf(
    path: Path,
    bag: BagSplineData,
    estimated: baseline.Simulation,
    estimated_with_external_wrench: baseline.Simulation,
    nominal: baseline.Simulation,
) -> None:
    observed = bag.direct_problem.observations
    relative_time = observed.time - observed.time[0]
    estimated_position_error = estimated.sensor_position - observed.sensor_position
    estimated_orientation_error = _orientation_errors(
        observed.sensor_orientation_xyzw,
        estimated.sensor_orientation_xyzw,
    )
    external_position_error = (
        estimated_with_external_wrench.sensor_position
        - observed.sensor_position
    )
    external_orientation_error = _orientation_errors(
        observed.sensor_orientation_xyzw,
        estimated_with_external_wrench.sensor_orientation_xyzw,
    )
    observed_rpy = baseline._rpy_series(
        observed.sensor_orientation_xyzw
    )
    estimated_rpy = baseline._rpy_series(
        estimated.sensor_orientation_xyzw
    )
    external_rpy = baseline._rpy_series(
        estimated_with_external_wrench.sensor_orientation_xyzw
    )
    primary_lower, primary_upper = _common_3d_limits(
        observed.sensor_position,
        estimated.sensor_position,
        estimated_with_external_wrench.sensor_position,
    )
    forced_lower, forced_upper = _common_3d_limits(
        observed.sensor_position,
        estimated_with_external_wrench.sensor_position,
    )
    comparison_lower, comparison_upper = _common_3d_limits(
        observed.sensor_position,
        estimated.sensor_position,
        estimated_with_external_wrench.sensor_position,
        nominal.sensor_position,
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        axis = figure.add_subplot(121, projection="3d")
        axis.plot(
            observed.sensor_position[:, 0],
            observed.sensor_position[:, 1],
            observed.sensor_position[:, 2],
            color="#1e5abe",
            linewidth=2.5,
            label="observed",
        )
        axis.plot(
            estimated.sensor_position[:, 0],
            estimated.sensor_position[:, 1],
            estimated.sensor_position[:, 2],
            color="#d2691e",
            linewidth=2.0,
            linestyle="--",
            label="estimated full forward rollout",
        )
        axis.plot(
            estimated_with_external_wrench.sensor_position[:, 0],
            estimated_with_external_wrench.sensor_position[:, 1],
            estimated_with_external_wrench.sensor_position[:, 2],
            color="#1e965f",
            linewidth=2.0,
            linestyle=":",
            label="estimated + inferred external wrench",
        )
        axis.set_xlim(primary_lower[0], primary_upper[0])
        axis.set_ylim(primary_lower[1], primary_upper[1])
        axis.set_zlim(primary_lower[2], primary_upper[2])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.set_title(
            "Observed vs estimated free/forced (full scale)"
        )
        axis.legend(loc="best")
        forced_axis = figure.add_subplot(122, projection="3d")
        forced_axis.plot(
            observed.sensor_position[:, 0],
            observed.sensor_position[:, 1],
            observed.sensor_position[:, 2],
            color="#1e5abe",
            linewidth=2.5,
            label="observed",
        )
        forced_axis.plot(
            estimated_with_external_wrench.sensor_position[:, 0],
            estimated_with_external_wrench.sensor_position[:, 1],
            estimated_with_external_wrench.sensor_position[:, 2],
            color="#1e965f",
            linewidth=2.0,
            linestyle=":",
            label="estimated + external wrench",
        )
        forced_axis.set_xlim(forced_lower[0], forced_upper[0])
        forced_axis.set_ylim(forced_lower[1], forced_upper[1])
        forced_axis.set_zlim(forced_lower[2], forced_upper[2])
        forced_axis.set_xlabel("x [m]")
        forced_axis.set_ylabel("y [m]")
        forced_axis.set_zlabel("z [m]")
        forced_axis.set_title("Observed vs externally-forced estimated")
        forced_axis.legend(loc="best")
        figure.suptitle(
            "PRIMARY trajectory comparison with inferred external body wrench"
        )
        pdf.savefig(figure)
        plt.close(figure)

        primary_pages = (
            (
                "PRIMARY: observed vs estimated sensor position",
                observed.sensor_position,
                estimated.sensor_position,
                estimated_with_external_wrench.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "PRIMARY: observed vs estimated sensor orientation",
                observed_rpy,
                estimated_rpy,
                external_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
            (
                "Observed vs estimated world-frame velocity",
                observed.sensor_velocity_world,
                estimated.sensor_velocity_world,
                estimated_with_external_wrench.sensor_velocity_world,
                ("v_x [m/s]", "v_y [m/s]", "v_z [m/s]"),
            ),
            (
                "Observed vs estimated gyroscope",
                observed.angular_velocity_sensor,
                estimated.angular_velocity_sensor,
                estimated_with_external_wrench.angular_velocity_sensor,
                ("omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"),
            ),
            (
                "Observed vs estimated specific force",
                observed.specific_force_sensor,
                estimated.specific_force_sensor,
                estimated_with_external_wrench.specific_force_sensor,
                ("f_x [m/s2]", "f_y [m/s2]", "f_z [m/s2]"),
            ),
        )
        for title, reference, prediction, forced_prediction, labels in primary_pages:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, component_axis in enumerate(axes):
                component_axis.plot(
                    relative_time,
                    reference[:, component],
                    color="#1e5abe",
                    linewidth=2.2,
                    label="observed",
                )
                component_axis.plot(
                    relative_time,
                    prediction[:, component],
                    color="#d2691e",
                    linewidth=1.8,
                    linestyle="--",
                    label="estimated full forward rollout",
                )
                component_axis.plot(
                    relative_time,
                    forced_prediction[:, component],
                    color="#1e965f",
                    linewidth=1.8,
                    linestyle=":",
                    label="estimated + inferred external wrench",
                )
                component_axis.set_ylabel(labels[component])
                component_axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        for title, estimated_error, forced_error, labels in (
            (
                "Estimated free/externally-forced sensor position error",
                estimated_position_error,
                external_position_error,
                ("x error [m]", "y error [m]", "z error [m]"),
            ),
            (
                "Estimated free/externally-forced orientation log error",
                estimated_orientation_error,
                external_orientation_error,
                ("roll-like [rad]", "pitch-like [rad]", "yaw-like [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for component, component_axis in enumerate(axes):
                component_axis.plot(
                    relative_time,
                    estimated_error[:, component],
                    color="#d2691e",
                    label="estimated minus observed",
                )
                component_axis.plot(
                    relative_time,
                    forced_error[:, component],
                    color="#1e965f",
                    linestyle=":",
                    label="estimated + external wrench minus observed",
                )
                component_axis.axhline(
                    0.0, color="black", linewidth=0.7, alpha=0.5
                )
                component_axis.set_ylabel(labels[component])
                component_axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        for subplot, rollout, title, color in (
            (
                121,
                estimated,
                "Estimated parameters",
                "#d2691e",
            ),
            (
                122,
                nominal,
                "Nominal parameters (auxiliary)",
                "#8b4bb7",
            ),
        ):
            axis = figure.add_subplot(subplot, projection="3d")
            axis.plot(
                observed.sensor_position[:, 0],
                observed.sensor_position[:, 1],
                observed.sensor_position[:, 2],
                color="#1e5abe",
                linewidth=2.0,
                label="observed",
            )
            axis.plot(
                rollout.sensor_position[:, 0],
                rollout.sensor_position[:, 1],
                rollout.sensor_position[:, 2],
                color=color,
                linestyle="--",
                label=title,
            )
            if rollout is estimated:
                axis.plot(
                    estimated_with_external_wrench.sensor_position[:, 0],
                    estimated_with_external_wrench.sensor_position[:, 1],
                    estimated_with_external_wrench.sensor_position[:, 2],
                    color="#1e965f",
                    linestyle=":",
                    label="estimated + external wrench",
                )
            axis.set_xlim(comparison_lower[0], comparison_upper[0])
            axis.set_ylim(comparison_lower[1], comparison_upper[1])
            axis.set_zlim(comparison_lower[2], comparison_upper[2])
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_zlabel("z [m]")
            axis.set_title(title)
            axis.legend(loc="best")
        figure.suptitle(
            "Auxiliary estimated/nominal comparison; primary comparison is observed/estimated"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _write_sensor_validation_pdf(
    path: Path,
    bag: BagSplineData,
    estimated: baseline.Simulation,
    metrics: Mapping[str, Any],
    estimated_with_external_wrench: baseline.Simulation,
    external_metrics: Mapping[str, Any],
) -> None:
    observed = bag.direct_problem.observations
    relative_time = observed.time - observed.time[0]
    pages = (
        (
            "World-frame velocity",
            "world_velocity",
            observed.sensor_velocity_world,
            estimated.sensor_velocity_world,
            estimated_with_external_wrench.sensor_velocity_world,
            ("v_x [m/s]", "v_y [m/s]", "v_z [m/s]"),
        ),
        (
            "Gyroscope",
            "gyro",
            observed.angular_velocity_sensor,
            estimated.angular_velocity_sensor,
            estimated_with_external_wrench.angular_velocity_sensor,
            ("omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"),
        ),
        (
            "Specific force",
            "specific_force",
            observed.specific_force_sensor,
            estimated.specific_force_sensor,
            estimated_with_external_wrench.specific_force_sensor,
            ("f_x [m/s^2]", "f_y [m/s^2]", "f_z [m/s^2]"),
        ),
    )
    with PdfPages(path) as pdf:
        for title, key, reference, prediction, forced_prediction, labels in pages:
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for component, axis in enumerate(axes):
                statistic = metrics[key][component]
                forced_statistic = external_metrics[key][component]
                axis.plot(
                    relative_time,
                    reference[:, component],
                    color="#1e5abe",
                    linewidth=2.0,
                    label="measured (validation only)",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    color="#d2691e",
                    linestyle="--",
                    label="estimated full rollout",
                )
                axis.plot(
                    relative_time,
                    forced_prediction[:, component],
                    color="#1e965f",
                    linestyle=":",
                    label="estimated + inferred external wrench",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
                axis.text(
                    0.01,
                    0.03,
                    "free: RMSE={:.4g}, bias={:.4g}, r={:.4g}, "
                    "lag={:+.4g}s\nforced: RMSE={:.4g}, bias={:.4g}, "
                    "r={:.4g}, lag={:+.4g}s".format(
                        statistic["rmse"],
                        statistic["mean_bias"],
                        statistic["pearson_correlation"],
                        statistic["maximum_cross_correlation_time_shift_seconds"],
                        forced_statistic["rmse"],
                        forced_statistic["mean_bias"],
                        forced_statistic["pearson_correlation"],
                        forced_statistic[
                            "maximum_cross_correlation_time_shift_seconds"
                        ],
                    ),
                    transform=axis.transAxes,
                    fontsize=8,
                )
            axes[0].set_title(title + " (not used by estimator loss)")
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)


def _write_residual_wrench_pdf(
    path: Path,
    bag: BagSplineData,
    evaluation: BagDynamicsEvaluation,
    statistics: Mapping[str, Any],
) -> None:
    relative_time = bag.collocation_time - bag.collocation_time[0]
    cumulative = np.zeros_like(evaluation.residual_body_wrench)
    increments = 0.5 * (
        evaluation.residual_body_wrench[1:]
        + evaluation.residual_body_wrench[:-1]
    ) * np.diff(bag.collocation_time)[:, None]
    cumulative[1:] = np.cumsum(increments, axis=0)
    names = statistics["component_names"]
    units = ("N", "N", "N", "N m", "N m", "N m")
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for row in range(3):
            for column, component in ((0, row), (1, row + 3)):
                axis = axes[row, column]
                axis.plot(
                    relative_time,
                    evaluation.residual_body_wrench[:, component],
                    color="#1e965f",
                    linewidth=1.5,
                )
                axis.axhline(
                    0.0, color="black", linewidth=0.7, alpha=0.5
                )
                axis.set_ylabel(
                    "{} [{}]".format(names[component], units[component])
                )
                axis.grid(True, alpha=0.25)
        axes[0, 0].set_title(
            "Inferred external body wrench: force (left) and torque (right)"
        )
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        for offset, kind in ((0, "body force"), (3, "body torque")):
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(
                    relative_time,
                    evaluation.required_body_wrench[:, component],
                    color="#1e5abe",
                    label="required by pose spline",
                )
                axis.plot(
                    relative_time,
                    evaluation.modeled_body_wrench[:, component],
                    color="#d2691e",
                    linestyle="--",
                    label="physical model",
                )
                axis.set_ylabel("{} [{}]".format(names[component], units[component]))
                axis.grid(True, alpha=0.25)
            axes[0].set_title("Required and modeled " + kind)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(
                    relative_time,
                    evaluation.residual_body_wrench[:, component],
                    color="#8b4bb7",
                )
                axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
                axis.set_ylabel("{} residual [{}]".format(names[component], units[component]))
                axis.grid(True, alpha=0.25)
                axis.text(
                    0.01,
                    0.03,
                    "mean={:.4g}, RMSE={:.4g}, trend={:+.4g}/s, impulse={:+.4g}".format(
                        statistics["mean"][component],
                        statistics["rmse"][component],
                        statistics["linear_trend_per_second"][component],
                        statistics["cumulative_impulse"][component],
                    ),
                    transform=axis.transAxes,
                    fontsize=8,
                )
            axes[0].set_title("Residual " + kind)
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(relative_time, cumulative[:, component], color="#1e965f")
                axis.set_ylabel("int {} dt".format(names[component]))
                axis.grid(True, alpha=0.25)
            axes[0].set_title("Cumulative residual " + kind + " impulse")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)


def _write_spline_fit_pdf(path: Path, bag: BagSplineData) -> None:
    direct = bag.direct_problem
    observed = direct.observations
    spline = bag.spline_selection.spline.evaluate(observed.time)
    spline_sensor_rotation = bag.spline_selection.spline.sensor_rotation(
        observed.time
    )
    spline_quaternion = np.asarray(
        [matrix_to_quaternion(value) for value in spline_sensor_rotation],
        dtype=float,
    )
    observed_rpy = baseline._rpy_series(observed.sensor_orientation_xyzw)
    spline_rpy = baseline._rpy_series(spline_quaternion)
    position_error = spline.sensor_position - observed.sensor_position
    orientation_error = _orientation_errors(
        observed.sensor_orientation_xyzw, spline_quaternion
    )
    relative_time = observed.time - observed.time[0]
    collocation = bag.collocation
    collocation_relative = collocation.time - collocation.time[0]
    with PdfPages(path) as pdf:
        for title, reference, prediction, labels in (
            (
                "Pose-only spline position fit",
                observed.sensor_position,
                spline.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Pose-only spline orientation fit",
                observed_rpy,
                spline_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    label="observed",
                    color="#1e5abe",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    label="spline",
                    color="#1e965f",
                    linestyle="--",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; selected knot spacing={:.4g}s".format(
                    title, bag.spline_selection.selected_spacing_seconds
                )
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        for title, value, labels in (
            (
                "Spline pose residual",
                np.column_stack((position_error, orientation_error)),
                (
                    "dx [m]",
                    "dy [m]",
                    "dz [m]",
                    "dRx [rad]",
                    "dRy [rad]",
                    "dRz [rad]",
                ),
            ),
            (
                "Spline translational derivatives",
                np.column_stack(
                    (
                        collocation.sensor_velocity_world,
                        collocation.sensor_acceleration_world,
                    )
                ),
                (
                    "vx [m/s]",
                    "vy [m/s]",
                    "vz [m/s]",
                    "ax [m/s2]",
                    "ay [m/s2]",
                    "az [m/s2]",
                ),
            ),
            (
                "Spline rotational derivatives (body frame)",
                np.column_stack(
                    (
                        collocation.body_angular_velocity,
                        collocation.body_angular_acceleration,
                    )
                ),
                (
                    "wx [rad/s]",
                    "wy [rad/s]",
                    "wz [rad/s]",
                    "alphax [rad/s2]",
                    "alphay [rad/s2]",
                    "alphaz [rad/s2]",
                ),
            ),
        ):
            time_value = (
                relative_time
                if value.shape[0] == relative_time.size
                else collocation_relative
            )
            figure, axes = plt.subplots(
                3,
                2,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes.ravel()):
                axis.plot(time_value, value[:, component], color="#8b4bb7")
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0, 0].set_title(title)
            axes[-1, 0].set_xlabel("time [s]")
            axes[-1, 1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
        spacings = [
            candidate.requested_knot_spacing_seconds
            for candidate in bag.spline_selection.candidates
        ]
        scores = [candidate.score for candidate in bag.spline_selection.candidates]
        axis.plot(spacings, scores, marker="o")
        axis.axvline(
            bag.spline_selection.selected_spacing_seconds,
            color="#1e965f",
            linestyle="--",
            label="selected",
        )
        axis.set_xlabel("knot spacing [s]")
        axis.set_ylabel("blocked-CV pose score [m2]")
        axis.set_title("Pose-only blocked cross-validation")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        pdf.savefig(figure)
        plt.close(figure)


def _write_delay_profile_pdf(
    path: Path,
    smooth_delay: float,
    candidate_delays: np.ndarray,
    screening_costs: np.ndarray,
    solutions: Sequence[DynamicsSolution],
    selected: DynamicsSolution,
) -> None:
    figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
    axis.plot(
        1000.0 * candidate_delays,
        screening_costs,
        marker="o",
        label="strict ZOH screening",
    )
    axis.scatter(
        [1000.0 * solution.delay_seconds for solution in solutions],
        [
            0.5
            * float(
                solution.evaluation.residual
                @ solution.evaluation.residual
            )
            for solution in solutions
        ],
        marker="s",
        s=70,
        label="strict ZOH, physical parameters reoptimized",
    )
    axis.axvline(
        1000.0 * smooth_delay,
        color="#9467bd",
        linestyle="--",
        label="smoothstep estimate",
    )
    axis.scatter(
        [1000.0 * selected.delay_seconds],
        [0.5 * float(selected.evaluation.residual @ selected.evaluation.residual)],
        marker="*",
        s=220,
        color="#1e965f",
        label="selected strict ZOH",
        zorder=5,
    )
    axis.set_xlabel("recorded-command lag [ms]")
    axis.set_ylabel("joint gradient-matching objective")
    axis.set_title("Smooth lag estimate and strict-ZOH local polish")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    with PdfPages(path) as pdf:
        pdf.savefig(figure)
    plt.close(figure)


def _spline_fit_metrics(bag: BagSplineData) -> dict[str, Any]:
    observations = bag.direct_problem.observations
    evaluation = bag.spline_selection.spline.evaluate(observations.time)
    sensor_rotation = bag.spline_selection.spline.sensor_rotation(
        observations.time
    )
    quaternion = np.asarray(
        [matrix_to_quaternion(value) for value in sensor_rotation],
        dtype=float,
    )
    position_error = evaluation.sensor_position - observations.sensor_position
    rotation_error = _orientation_errors(
        observations.sensor_orientation_xyzw, quaternion
    )
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_vector_rmse_m": float(
            np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        ),
        "orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(np.sum(rotation_error**2, axis=1)))
        ),
        "orientation_angle_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.sum(rotation_error**2, axis=1))))
        ),
        "maximum_acceleration_m_per_s2": float(
            np.max(np.linalg.norm(bag.collocation.sensor_acceleration_world, axis=1))
        ),
        "maximum_angular_acceleration_rad_per_s2": float(
            np.max(np.linalg.norm(bag.collocation.body_angular_acceleration, axis=1))
        ),
    }


def _parameter_lines(
    initial: InitialEstimate,
    selected: DynamicsSolution,
    bags: Sequence[BagSplineData],
    bag_payloads: Sequence[Mapping[str, Any]],
    smooth_stages: Sequence[Mapping[str, Any]],
    strict_candidates: Sequence[Mapping[str, Any]],
    collocation_step: float,
) -> list[str]:
    parameters = selected.evaluation.decoded.parameters
    principal = np.linalg.eigvalsh(parameters.inertia)
    lines = [
        "Deterministic pose-spline dynamics estimator",
        "",
        "Estimator structure",
        "  pose-only continuous-time spline -> analytic derivatives",
        "  fixed-spline gradient matching -> shared physical parameters and lag",
        "  shooting nodes: none",
        "  continuity constraints: none",
        "  sensor/IMU channels in loss: none",
        "",
        "Initial estimate",
        "  source kind: {}".format(initial.source_kind),
        "  source path: {}".format(initial.source_path),
        "  source mass [kg]: {:.12g}".format(initial.source_mass_kg),
        "  selected initial mass [kg]: {:.12g}".format(initial.selected_mass_kg),
        "",
        "Selected strict-ZOH parameters",
        "  mass [kg]: {:.12g}".format(parameters.mass),
        "  inertia [kg m^2]:",
    ]
    lines.extend(
        "    [{: .12g}, {: .12g}, {: .12g}]".format(*row)
        for row in parameters.inertia
    )
    lines.extend(
        [
            "  principal moments [kg m^2]: "
            "[{:.12g}, {:.12g}, {:.12g}]".format(*principal),
            "  inertia triangle margin [kg m^2]: {:.12g}".format(
                selected.evaluation.decoded.inertia_triangle_margin
            ),
            "  CoG offset [m]: [{:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.cog_offset
            ),
            "  rotor force effectiveness: "
            "[{:.12g}, {:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.force_effectiveness
            ),
            "  lag [s]: {:.12g}".format(selected.delay_seconds),
            "  joint dynamics loss: {:.12g}".format(selected.evaluation.data_loss),
            "  soft-prior cost: {:.12g}".format(selected.evaluation.prior_cost),
            "",
            "Smooth physical coordinates",
        ]
    )
    lines.extend(
        "  {:50s} {: .12g}".format(name, value)
        for name, value in zip(
            strict.PHYSICAL_PARAMETER_NAMES,
            selected.physical_coordinate,
        )
    )
    lines.extend(
        [
            "",
            "Spline and collocation",
            "  collocation step [s]: {:.12g}".format(collocation_step),
        ]
    )
    for bag, payload in zip(bags, bag_payloads):
        lines.extend(
            [
                "",
                "Bag {}".format(bag.specification.bag_id),
                "  selected spline knot spacing [s]: {:.12g}".format(
                    bag.spline_selection.selected_spacing_seconds
                ),
                "  dynamics loss: {:.12g}".format(payload["dynamics_loss"]),
                "  estimated forward position RMSE [m]: {:.12g}".format(
                    payload["estimated_forward_metrics"]["position_rmse_m"]
                ),
                "  estimated + external wrench position RMSE [m]: "
                "{:.12g}".format(
                    payload[
                        "estimated_with_external_wrench_forward_metrics"
                    ]["position_rmse_m"]
                ),
                "  external wrench improves estimated free rollout: {}".format(
                    payload[
                        "external_wrench_improves_estimated_free_rollout"
                    ]
                ),
                "  nominal forward position RMSE [m]: {:.12g}".format(
                    payload["nominal_forward_metrics"]["position_rmse_m"]
                ),
                "  estimated improves nominal: {}".format(
                    payload["estimated_improves_over_nominal"]
                ),
                "  residual wrench mean: {}".format(
                    np.array2string(
                        np.asarray(
                            payload["residual_wrench_statistics"]["mean"]
                        ),
                        precision=6,
                    )
                ),
                "  residual wrench RMSE: {}".format(
                    np.array2string(
                        np.asarray(
                            payload["residual_wrench_statistics"]["rmse"]
                        ),
                        precision=6,
                    )
                ),
                "  residual wrench trend/s: {}".format(
                    np.array2string(
                        np.asarray(
                            payload["residual_wrench_statistics"][
                                "linear_trend_per_second"
                            ]
                        ),
                        precision=6,
                    )
                ),
                "  residual wrench cumulative impulse: {}".format(
                    np.array2string(
                        np.asarray(
                            payload["residual_wrench_statistics"][
                                "cumulative_impulse"
                            ]
                        ),
                        precision=6,
                    )
                ),
            ]
        )
    lines.extend(["", "Smoothstep continuation"])
    for stage in smooth_stages:
        lines.append(
            "  width={:.6g}: delay={:.9g}s, cost={:.12g}, nfev={}".format(
                stage["width_fraction"],
                stage["delay_seconds"],
                stage["objective_cost"],
                stage["optimizer"]["nfev"],
            )
        )
    lines.extend(["", "Strict-ZOH polish"])
    for candidate in strict_candidates:
        lines.append(
            "  delay={:.9g}s: screening={:.12g}, refined={:.12g}, selected={}".format(
                candidate["delay_seconds"],
                candidate["screening_cost"],
                candidate.get("refined_cost", float("nan")),
                candidate.get("selected", False),
            )
        )
    return lines


def _write_parameters_pdf(path: Path, lines: Sequence[str]) -> None:
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=96,
                subsequent_indent="    ",
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    page_size = 55
    with PdfPages(path) as pdf:
        for start in range(0, len(wrapped_lines), page_size):
            figure = plt.figure(figsize=(8.3, 11.7), constrained_layout=True)
            figure.text(
                0.04,
                0.98,
                "\n".join(wrapped_lines[start : start + page_size]),
                va="top",
                ha="left",
                family="monospace",
                fontsize=7.5,
            )
            pdf.savefig(figure)
            plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate shared rigid-body parameters and command lag by "
            "pose-only spline gradient matching across multiple bags."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--estimator-result",
        type=Path,
        default=DEFAULT_ESTIMATOR_RESULT,
        help=(
            "initial multiple-shooting result.json; missing path falls back "
            "to nominal physical parameters"
        ),
    )
    parser.add_argument(
        "--corrected-mass",
        type=float,
        default=None,
        help="override only the initial mass while retaining other seed coordinates",
    )
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--smooth-max-nfev", type=int, default=60)
    parser.add_argument("--strict-max-nfev", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument("--delay-bounds", type=float, nargs=2, default=(0.0, 0.20))
    parser.add_argument("--initial-delay", type=float, default=None)
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(0.50, 0.20, 0.05),
    )
    parser.add_argument("--zoh-polish-radius", type=float, default=0.004)
    parser.add_argument("--zoh-polish-step", type=float, default=0.001)
    parser.add_argument("--zoh-polish-top-k", type=int, default=3)
    parser.add_argument("--physical-bound-scale", type=float, default=np.inf)
    parser.add_argument("--spline-cv-folds", type=int, default=5)
    parser.add_argument("--maximum-spline-acceleration", type=float, default=250.0)
    parser.add_argument(
        "--maximum-spline-angular-acceleration", type=float, default=1000.0
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def _validate_arguments(
    arguments: argparse.Namespace,
    config: SplineEstimatorConfig,
    initial_delay: float,
) -> None:
    positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.smooth_max_nfev,
        arguments.strict_max_nfev,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.zoh_polish_top_k,
        arguments.spline_cv_folds,
        arguments.maximum_spline_acceleration,
        arguments.maximum_spline_angular_acceleration,
        config.spline.collocation_step_seconds,
    )
    bounds = np.asarray(arguments.delay_bounds, dtype=float)
    widths = np.asarray(arguments.smoothstep_width_fractions, dtype=float)
    ratio = arguments.sample_step / arguments.integration_step
    if (
        any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or np.isnan(arguments.physical_bound_scale)
        or arguments.physical_bound_scale <= 0.0
        or not np.isclose(ratio, round(ratio), atol=1.0e-12, rtol=0.0)
        or not np.isfinite(arguments.prior_weight)
        or arguments.prior_weight < 0.0
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or not bounds[0] <= initial_delay <= bounds[1]
        or widths.ndim != 1
        or widths.size == 0
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit("spline-dynamics estimator settings are invalid")


def _solution_cost(solution: DynamicsSolution) -> float:
    residual = solution.evaluation.residual
    return 0.5 * float(residual @ residual)


def _rollout_pose_score(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
) -> float:
    position_error = simulation.sensor_position - observations.sensor_position
    orientation_error = _orientation_errors(
        observations.sensor_orientation_xyzw,
        simulation.sensor_orientation_xyzw,
    )
    nominal = VehicleParameters.nominal()
    rotational_metric = nominal.inertia / nominal.mass
    squared = np.sum(position_error**2, axis=1) + np.einsum(
        "ni,ij,nj->n", orientation_error, rotational_metric, orientation_error
    )
    return 0.5 * float(np.mean(squared))


def _strict_candidate_payloads(
    candidate_delays: np.ndarray,
    screening_costs: np.ndarray,
    solutions: Sequence[DynamicsSolution],
    selected: DynamicsSolution,
) -> list[dict[str, Any]]:
    refined = {
        round(solution.delay_seconds, 12): solution for solution in solutions
    }
    payloads = []
    for delay, screening in zip(candidate_delays, screening_costs):
        solution = refined.get(round(float(delay), 12))
        payload: dict[str, Any] = {
            "delay_seconds": float(delay),
            "screening_cost": float(screening),
            "refined": solution is not None,
            "selected": bool(
                np.isclose(delay, selected.delay_seconds, atol=5.0e-13, rtol=0.0)
            ),
        }
        if solution is not None:
            payload.update(
                {
                    "refined_cost": _solution_cost(solution),
                    "physical_coordinate": solution.physical_coordinate,
                    "optimizer": solution.optimizer,
                }
            )
        payloads.append(payload)
    return payloads


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run(arguments: argparse.Namespace) -> int:
    try:
        config = load_spline_config(arguments.config)
        initial = load_initial_estimate(
            arguments.estimator_result,
            config.multi_bag.initial_delay_seconds,
            arguments.corrected_mass,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    initial_delay = (
        initial.delay_seconds
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    _validate_arguments(arguments, config, initial_delay)
    started = time.perf_counter()
    if initial.source_kind == "nominal_fallback":
        print(
            "initial estimator result not found; using nominal physical parameters",
            flush=True,
        )
    else:
        print(
            "initial physical parameters: {}".format(initial.source_path),
            flush=True,
        )
    if arguments.corrected_mass is not None:
        print(
            "initial mass override: {:.9g} kg (source {:.9g} kg)".format(
                initial.selected_mass_kg, initial.source_mass_kg
            ),
            flush=True,
        )

    raw_weights = np.asarray(
        [specification.weight for specification in config.multi_bag.bags],
        dtype=float,
    )
    normalized_weights = raw_weights / np.sum(raw_weights)
    bags = []
    for index, (specification, weight) in enumerate(
        zip(config.multi_bag.bags, normalized_weights)
    ):
        print(
            "loading and fitting pose spline {}/{} {}: {} [{:.3f}, {:.3f}] s".format(
                index + 1,
                len(config.multi_bag.bags),
                specification.bag_id,
                specification.path,
                specification.start,
                specification.end,
            ),
            flush=True,
        )
        flight = load_flight_data(
            str(specification.path),
            start_local=specification.start,
            end_local=specification.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        bag = _build_bag_data(
            specification,
            float(weight),
            flight,
            initial_delay,
            config.spline,
            arguments,
        )
        print(
            "  selected knot spacing {:.6g}s from {}".format(
                bag.spline_selection.selected_spacing_seconds,
                tuple(config.spline.knot_spacing_candidates_seconds),
            ),
            flush=True,
        )
        bags.append(bag)
    problem = SplineDynamicsProblem(bags, arguments.prior_weight)

    physical_lower, physical_upper = _physical_bounds(
        initial.physical_coordinate, arguments.physical_bound_scale
    )
    smooth_lower = np.concatenate(
        (physical_lower, np.asarray((arguments.delay_bounds[0],), dtype=float))
    )
    smooth_upper = np.concatenate(
        (physical_upper, np.asarray((arguments.delay_bounds[1],), dtype=float))
    )
    coordinate = np.concatenate(
        (initial.physical_coordinate, np.asarray((initial_delay,), dtype=float))
    )
    coordinate = np.minimum(np.maximum(coordinate, smooth_lower), smooth_upper)
    smooth_stage_payloads = []
    final_smooth_evaluation: Optional[JointDynamicsEvaluation] = None
    for stage_index, width in enumerate(arguments.smoothstep_width_fractions):
        print(
            "smoothstep gradient matching {}/{}: width_fraction={:.6g}".format(
                stage_index + 1,
                len(arguments.smoothstep_width_fractions),
                width,
            ),
            flush=True,
        )
        coordinate, evaluation, optimizer = _solve_smooth(
            problem,
            coordinate,
            float(width),
            smooth_lower,
            smooth_upper,
            arguments,
        )
        final_smooth_evaluation = evaluation
        stage_cost = 0.5 * float(evaluation.residual @ evaluation.residual)
        print(
            "  cost={:.9g}, delay={:.6f}s, nfev={}".format(
                stage_cost, coordinate[DELAY_INDEX], optimizer["nfev"]
            ),
            flush=True,
        )
        smooth_stage_payloads.append(
            {
                "width_fraction": float(width),
                "objective_cost": stage_cost,
                "data_loss": evaluation.data_loss,
                "prior_cost": evaluation.prior_cost,
                "physical_coordinate": coordinate[: strict.PHYSICAL_DIMENSION],
                "delay_seconds": float(coordinate[DELAY_INDEX]),
                "optimizer": optimizer,
            }
        )
    if final_smooth_evaluation is None:
        raise RuntimeError("smooth continuation did not run")

    smooth_physical = coordinate[: strict.PHYSICAL_DIMENSION].copy()
    smooth_delay = float(coordinate[DELAY_INDEX])
    candidate_delays = smooth.zoh_polish_delays(
        smooth_delay,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.delay_bounds,
    )
    screening_costs = np.asarray(
        [
            0.5
            * float(
                (evaluation := problem.evaluate_strict(smooth_physical, float(delay))).residual
                @ evaluation.residual
            )
            for delay in candidate_delays
        ],
        dtype=float,
    )
    top_count = min(arguments.zoh_polish_top_k, candidate_delays.size)
    top_indices = np.argsort(screening_costs, kind="stable")[:top_count]
    strict_solutions = []
    for rank, candidate_index in enumerate(top_indices):
        delay = float(candidate_delays[candidate_index])
        print(
            "strict-ZOH polish {}/{}: delay={:.6f}s, screening={:.9g}".format(
                rank + 1, top_count, delay, screening_costs[candidate_index]
            ),
            flush=True,
        )
        solution = _solve_strict(
            problem,
            smooth_physical,
            delay,
            physical_lower,
            physical_upper,
            arguments,
        )
        print(
            "  refined cost={:.9g}, nfev={}".format(
                _solution_cost(solution), solution.optimizer["nfev"]
            ),
            flush=True,
        )
        strict_solutions.append(solution)
    selected = min(strict_solutions, key=_solution_cost)
    strict_payloads = _strict_candidate_payloads(
        candidate_delays, screening_costs, strict_solutions, selected
    )
    print(
        "selected strict lag {:.6f}s; producing correction-free rollouts and reports".format(
            selected.delay_seconds
        ),
        flush=True,
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    )
    bags_directory = output_directory / "bags"
    bags_directory.mkdir(parents=True, exist_ok=True)
    bag_payloads = []
    bag_sources = []
    bag_outputs: dict[str, Any] = {}
    selected_parameters = selected.evaluation.decoded.parameters
    for index, bag in enumerate(bags):
        bag_id = bag.specification.bag_id
        bag_directory = bags_directory / bag_id
        bag_directory.mkdir(parents=True, exist_ok=True)
        evaluation = selected.evaluation.bag_evaluations[index]
        external_wrench_history = BodyWrenchHistory(
            bag.collocation_time,
            evaluation.residual_body_wrench,
        )
        estimated_rollout = forward_rollout(
            bag, selected.physical_coordinate, selected.delay_seconds
        )
        external_wrench_rollout = forward_rollout(
            bag,
            selected.physical_coordinate,
            selected.delay_seconds,
            external_body_wrench=external_wrench_history,
        )
        nominal_rollout = forward_rollout(
            bag,
            np.zeros(strict.PHYSICAL_DIMENSION, dtype=float),
            initial_delay,
        )
        observations = bag.direct_problem.observations
        estimated_metrics = _pose_metrics(observations, estimated_rollout)
        external_wrench_metrics = _pose_metrics(
            observations, external_wrench_rollout
        )
        nominal_metrics = _pose_metrics(observations, nominal_rollout)
        estimated_score = _rollout_pose_score(observations, estimated_rollout)
        external_wrench_score = _rollout_pose_score(
            observations, external_wrench_rollout
        )
        nominal_score = _rollout_pose_score(observations, nominal_rollout)
        sensor_metrics = _sensor_metrics(observations, estimated_rollout)
        external_wrench_sensor_metrics = _sensor_metrics(
            observations, external_wrench_rollout
        )
        wrench_statistics = _wrench_statistics(
            bag.collocation_time, evaluation.residual_body_wrench
        )
        spline_metrics = _spline_fit_metrics(bag)
        bag_payload = {
            "id": bag_id,
            "normalized_weight": bag.normalized_weight,
            "spline": {
                "selected_knot_spacing_seconds": (
                    bag.spline_selection.selected_spacing_seconds
                ),
                "blocked_cross_validation": [
                    candidate_payload(candidate)
                    for candidate in bag.spline_selection.candidates
                ],
                "fit_metrics": spline_metrics,
            },
            "collocation_count": int(bag.collocation_time.size),
            "dynamics_loss": evaluation.dynamics_loss,
            "residual_wrench_statistics": wrench_statistics,
            "inferred_external_wrench": {
                "definition": "required body wrench minus modeled body wrench",
                "frame": "body",
                "components": ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z"),
                "force_unit": "N",
                "torque_unit": "N m",
                "rollout_interpolation": "linear in time with endpoint hold",
            },
            "estimated_forward_metrics": estimated_metrics,
            "estimated_with_external_wrench_forward_metrics": (
                external_wrench_metrics
            ),
            "nominal_forward_metrics": nominal_metrics,
            "estimated_forward_pose_score_m2": estimated_score,
            "estimated_with_external_wrench_forward_pose_score_m2": (
                external_wrench_score
            ),
            "external_wrench_improves_estimated_free_rollout": bool(
                external_wrench_score < estimated_score
            ),
            "nominal_forward_pose_score_m2": nominal_score,
            "estimated_improves_over_nominal": bool(
                estimated_score < nominal_score
            ),
            "sensor_validation": sensor_metrics,
            "sensor_validation_with_external_wrench": (
                external_wrench_sensor_metrics
            ),
            "nominal_rollout_lag_seconds": initial_delay,
        }
        _write_spline_fit_pdf(bag_directory / "spline_fit.pdf", bag)
        _write_trajectory_pdf(
            bag_directory / "trajectory_3d.pdf",
            bag,
            estimated_rollout,
            external_wrench_rollout,
            nominal_rollout,
        )
        shutil.copyfile(
            bag_directory / "trajectory_3d.pdf",
            bag_directory / "trajectory.pdf",
        )
        _write_sensor_validation_pdf(
            bag_directory / "sensor_validation.pdf",
            bag,
            estimated_rollout,
            sensor_metrics,
            external_wrench_rollout,
            external_wrench_sensor_metrics,
        )
        _write_residual_wrench_pdf(
            bag_directory / "residual_wrench.pdf",
            bag,
            evaluation,
            wrench_statistics,
        )
        shutil.copyfile(
            bag_directory / "residual_wrench.pdf",
            bag_directory / "external_wrench.pdf",
        )
        np.savez_compressed(
            bag_directory / "spline_dynamics.npz",
            collocation_time=bag.collocation_time,
            output_time=observations.time,
            observed_sensor_position=observations.sensor_position,
            observed_sensor_orientation_xyzw=(
                observations.sensor_orientation_xyzw
            ),
            observed_sensor_velocity_world=(
                observations.sensor_velocity_world
            ),
            observed_angular_velocity_sensor=(
                observations.angular_velocity_sensor
            ),
            observed_specific_force_sensor=(
                observations.specific_force_sensor
            ),
            spline_sensor_position=bag.collocation.sensor_position,
            spline_sensor_velocity_world=(
                bag.collocation.sensor_velocity_world
            ),
            spline_sensor_acceleration_world=(
                bag.collocation.sensor_acceleration_world
            ),
            spline_body_rotation=bag.collocation.body_rotation,
            spline_body_angular_velocity=(
                bag.collocation.body_angular_velocity
            ),
            spline_body_angular_acceleration=(
                bag.collocation.body_angular_acceleration
            ),
            required_body_wrench=evaluation.required_body_wrench,
            modeled_body_wrench=evaluation.modeled_body_wrench,
            residual_body_wrench=evaluation.residual_body_wrench,
            inferred_external_body_wrench_time=bag.collocation_time,
            inferred_external_body_wrench=(
                evaluation.residual_body_wrench
            ),
            estimated_forward_sensor_position=(
                estimated_rollout.sensor_position
            ),
            estimated_forward_sensor_orientation_xyzw=(
                estimated_rollout.sensor_orientation_xyzw
            ),
            estimated_forward_sensor_velocity_world=(
                estimated_rollout.sensor_velocity_world
            ),
            estimated_forward_angular_velocity_sensor=(
                estimated_rollout.angular_velocity_sensor
            ),
            estimated_forward_specific_force_sensor=(
                estimated_rollout.specific_force_sensor
            ),
            external_wrench_forward_sensor_position=(
                external_wrench_rollout.sensor_position
            ),
            external_wrench_forward_sensor_orientation_xyzw=(
                external_wrench_rollout.sensor_orientation_xyzw
            ),
            external_wrench_forward_sensor_velocity_world=(
                external_wrench_rollout.sensor_velocity_world
            ),
            external_wrench_forward_angular_velocity_sensor=(
                external_wrench_rollout.angular_velocity_sensor
            ),
            external_wrench_forward_specific_force_sensor=(
                external_wrench_rollout.specific_force_sensor
            ),
            nominal_forward_sensor_position=nominal_rollout.sensor_position,
            nominal_forward_sensor_orientation_xyzw=(
                nominal_rollout.sensor_orientation_xyzw
            ),
        )
        bag_result = {
            "schema": SCHEMA + "/bag-result",
            "source": {
                "id": bag_id,
                "path": str(bag.specification.path),
                "sha256": baseline._sha256(bag.specification.path),
                "requested_interval_seconds": (
                    bag.specification.start,
                    bag.specification.end,
                ),
                "raw_weight": bag.specification.weight,
                "normalized_weight": bag.normalized_weight,
            },
            "shared_parameters": baseline._physical_parameters(
                selected_parameters
            ),
            "shared_physical_coordinate": selected.physical_coordinate,
            "shared_delay_seconds": selected.delay_seconds,
            "diagnostics": bag_payload,
            "outputs": {
                "spline_fit_pdf": "spline_fit.pdf",
                "trajectory_pdf": "trajectory.pdf",
                "trajectory_3d_pdf": "trajectory_3d.pdf",
                "sensor_validation_pdf": "sensor_validation.pdf",
                "residual_wrench_pdf": "residual_wrench.pdf",
                "external_wrench_pdf": "external_wrench.pdf",
                "spline_dynamics_npz": "spline_dynamics.npz",
            },
        }
        baseline._write_json(
            bag_directory / "result.json", _json_sanitize(bag_result)
        )
        bag_payloads.append(bag_payload)
        bag_sources.append(bag_result["source"])
        relative = "bags/{}/".format(bag_id)
        bag_outputs[bag_id] = {
            "result_json": relative + "result.json",
            "spline_fit_pdf": relative + "spline_fit.pdf",
            "trajectory_pdf": relative + "trajectory.pdf",
            "trajectory_3d_pdf": relative + "trajectory_3d.pdf",
            "sensor_validation_pdf": relative + "sensor_validation.pdf",
            "residual_wrench_pdf": relative + "residual_wrench.pdf",
            "external_wrench_pdf": relative + "external_wrench.pdf",
            "spline_dynamics_npz": relative + "spline_dynamics.npz",
        }

    lines = _parameter_lines(
        initial,
        selected,
        bags,
        bag_payloads,
        smooth_stage_payloads,
        strict_payloads,
        config.spline.collocation_step_seconds,
    )
    strict._write_text(output_directory / "parameters.txt", lines)
    _write_parameters_pdf(output_directory / "parameters.pdf", lines)
    _write_delay_profile_pdf(
        output_directory / "delay_profile.pdf",
        smooth_delay,
        candidate_delays,
        screening_costs,
        strict_solutions,
        selected,
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(arguments.config.expanduser().resolve()),
        "method": {
            "name": "deterministic_spline_dynamics",
            "description": (
                "pose-only continuous-time spline gradient matching with "
                "shared physical parameters and command lag"
            ),
            "uses_multiple_shooting_nodes": False,
            "uses_continuity_constraints": False,
            "uses_augmented_lagrangian": False,
            "sensor_channels_in_parameter_loss": False,
            "pose_role": "constructs fixed bag-local splines only",
            "command_mode_during_search": "quintic smoothstep ZOH",
            "command_mode_final": "strict ZOH",
        },
        "initial_estimate": {
            "source_kind": initial.source_kind,
            "source_path": initial.source_path,
            "source_mass_kg": initial.source_mass_kg,
            "selected_mass_kg": initial.selected_mass_kg,
            "physical_coordinate": initial.physical_coordinate,
            "delay_seconds": initial_delay,
        },
        "settings": {
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
            "collocation_step_seconds": (
                config.spline.collocation_step_seconds
            ),
            "knot_spacing_candidates_seconds": (
                config.spline.knot_spacing_candidates_seconds
            ),
            "prior_weight": arguments.prior_weight,
            "physical_coordinate_bounds": (
                "unbounded"
                if np.isinf(arguments.physical_bound_scale)
                else {
                    "center": "initial physical coordinate",
                    "soft_prior_standard_deviation_scale": (
                        arguments.physical_bound_scale
                    ),
                }
            ),
            "spline_cv_folds": arguments.spline_cv_folds,
            "maximum_spline_acceleration_m_per_s2": (
                arguments.maximum_spline_acceleration
            ),
            "maximum_spline_angular_acceleration_rad_per_s2": (
                arguments.maximum_spline_angular_acceleration
            ),
            "delay_bounds_seconds": arguments.delay_bounds,
            "smoothstep_width_fractions": (
                arguments.smoothstep_width_fractions
            ),
        },
        "bags": bag_sources,
        "smoothstep_stages": smooth_stage_payloads,
        "strict_zoh_polish": strict_payloads,
        "selection": {
            "physical_parameter_names": strict.PHYSICAL_PARAMETER_NAMES,
            "physical_coordinate": selected.physical_coordinate,
            "delay_seconds": selected.delay_seconds,
            "parameters": baseline._physical_parameters(selected_parameters),
            "inertia_principal_moments_kg_m2": (
                selected.evaluation.decoded.inertia_principal_moments
            ),
            "inertia_triangle_margin_kg_m2": (
                selected.evaluation.decoded.inertia_triangle_margin
            ),
            "joint_dynamics_loss": selected.evaluation.data_loss,
            "soft_prior_cost": selected.evaluation.prior_cost,
            "joint_objective_cost": _solution_cost(selected),
            "optimizer": selected.optimizer,
        },
        "bag_diagnostics": bag_payloads,
        "elapsed_seconds": elapsed,
        "outputs": {
            "parameters_txt": "parameters.txt",
            "parameters_pdf": "parameters.pdf",
            "delay_profile_pdf": "delay_profile.pdf",
            "bags": bag_outputs,
        },
    }
    baseline._write_json(
        output_directory / "result.json", _json_sanitize(result)
    )
    print("wrote {}".format(output_directory / "result.json"), flush=True)
    print("elapsed {:.3f}s".format(elapsed), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
