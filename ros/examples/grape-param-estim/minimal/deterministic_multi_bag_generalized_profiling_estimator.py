#!/usr/bin/env python3
"""Multi-bag generalized profiling with an SE(3) correction spline.

This estimator is deliberately downstream of the deterministic multi-bag
multiple-shooting estimator.  It loads that estimator's shared physical
coordinate when available, constructs one uncorrected strict-ZOH rollout per
bag, and fits a low-dimensional right-invariant SE(3) correction spline to
each rollout.  The fitted analysis trajectory trades pose agreement against
the rigid-body equation residual and a spline-curvature penalty.

The shared physical parameters can then be updated against the analysis
trajectories before the bag-local splines are profiled again.  Mass is always
held fixed during this outer update: the source estimate is used by default,
and ``--corrected-mass`` can replace it when an independently corrected mass
is known.  A missing source result is not an error; nominal parameters and the
config's initial command delay are used instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-minimal-matplotlib")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.interpolate import BSpline, CubicSpline  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation, RotationSpline  # noqa: E402

import deterministic_continuation_estimator as continuation  # noqa: E402
import deterministic_estimator as baseline  # noqa: E402
import deterministic_multi_bag_multiple_shooting_estimator as multi  # noqa: E402
import deterministic_multiple_shooting_estimator as strict  # noqa: E402
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
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


SCHEMA = "grape-param-estim/minimal-multi-bag-generalized-profiling/v1"
OUTPUT_SUBDIRECTORY = "generalized_profiling_multi"
DEFAULT_ESTIMATOR_RESULT = (
    Path(__file__).resolve().parent
    / "output"
    / multi.OUTPUT_SUBDIRECTORY
    / "result.json"
)
WRENCH_COMPONENT_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
WRENCH_COMPONENT_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")


@dataclass(frozen=True)
class ParameterSeed:
    physical_coordinate: np.ndarray
    delay_seconds: float
    source_kind: str
    source_path: Optional[Path]
    corrected_mass_kg: float
    source_mass_kg: float


@dataclass(frozen=True)
class SplineBasis:
    knots: np.ndarray
    degree: int
    value: np.ndarray
    first: np.ndarray
    second: np.ndarray

    @property
    def coefficient_count(self) -> int:
        return int(self.value.shape[1])


@dataclass(frozen=True)
class AnalysisEvaluation:
    body_position: np.ndarray
    body_rotation: np.ndarray
    body_velocity_world: np.ndarray
    body_acceleration_world: np.ndarray
    angular_velocity_body: np.ndarray
    angular_acceleration_body: np.ndarray
    sensor_position: np.ndarray
    sensor_rotation: np.ndarray
    pose_residual: np.ndarray
    wrench_residual: np.ndarray
    smooth_second_derivative: np.ndarray


@dataclass(frozen=True)
class BagProfileProblem:
    specification: multi.BagSpecification
    normalized_weight: float
    direct_problem: baseline.DirectShootingProblem
    strict_problem: strict.MultipleShootingProblem
    basis: SplineBasis
    base_body_position: np.ndarray
    base_body_rotation: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray
    pose_factor: np.ndarray
    lambda_dynamics: float
    lambda_smooth: float
    pose_scale: float
    force_scale: float
    torque_scale: float
    smooth_translation_scale: float
    smooth_rotation_scale: float

    @property
    def time(self) -> np.ndarray:
        return self.direct_problem.output_time

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return (self.basis.coefficient_count, 6)

    @property
    def coefficient_dimension(self) -> int:
        return self.basis.coefficient_count * 6

    def reshape_coefficients(self, value: Sequence[float]) -> np.ndarray:
        coefficients = np.asarray(value, dtype=float)
        if coefficients.shape == self.coefficient_shape:
            result = coefficients
        elif coefficients.shape == (self.coefficient_dimension,):
            result = coefficients.reshape(self.coefficient_shape)
        else:
            raise ValueError("spline coefficient array has the wrong shape")
        if np.any(~np.isfinite(result)):
            raise ValueError("spline coefficients must be finite")
        return result

    def initial_coefficients(
        self,
        parameters: VehicleParameters,
        use_observations: bool,
        translation_bound: float,
        rotation_bound: float,
    ) -> np.ndarray:
        if not use_observations:
            return np.zeros(self.coefficient_shape, dtype=float)
        direct = self.direct_problem
        pose_lever = direct.pose_sensor_position - parameters.cog_offset
        observed_body_rotation = direct.observed_body_rotation
        observed_body_position = (
            direct.observations.sensor_position
            - np.einsum("nij,j->ni", observed_body_rotation, pose_lever)
        )
        target = np.empty((self.time.size, 6), dtype=float)
        for index in range(self.time.size):
            relative_rotation = (
                self.base_body_rotation[index].T
                @ observed_body_rotation[index]
            )
            phi = so3_log(relative_rotation)
            relative_translation = self.base_body_rotation[index].T @ (
                observed_body_position[index] - self.base_body_position[index]
            )
            target[index, :3] = (
                so3_left_jacobian_inverse(phi) @ relative_translation
            )
            target[index, 3:] = phi
        coefficients = np.linalg.lstsq(
            self.basis.value,
            target,
            rcond=None,
        )[0]
        coefficients[:, :3] = np.clip(
            coefficients[:, :3], -translation_bound, translation_bound
        )
        coefficients[:, 3:] = np.clip(
            coefficients[:, 3:], -rotation_bound, rotation_bound
        )
        return coefficients

    def evaluate(
        self,
        coefficients: Sequence[float],
        parameters: VehicleParameters,
    ) -> AnalysisEvaluation:
        coefficient_matrix = self.reshape_coefficients(coefficients)
        correction = self.basis.value @ coefficient_matrix
        corrected_position = np.empty_like(self.base_body_position)
        corrected_rotation = np.empty_like(self.base_body_rotation)
        for index, value in enumerate(correction):
            rho = value[:3]
            phi = value[3:]
            corrected_position[index] = (
                self.base_body_position[index]
                + self.base_body_rotation[index]
                @ (so3_left_jacobian(phi) @ rho)
            )
            corrected_rotation[index] = (
                self.base_body_rotation[index] @ so3_exp(phi)
            )

        # The correction itself is a B-spline.  These interpolants provide a
        # continuous trajectory on R^3 x SO(3), whose first and second
        # derivatives are evaluated analytically by SciPy.
        position_spline = CubicSpline(
            self.time,
            corrected_position,
            axis=0,
            bc_type="natural",
        )
        rotation_spline = RotationSpline(
            self.time,
            Rotation.from_matrix(corrected_rotation),
        )
        body_velocity = np.asarray(position_spline(self.time, 1), dtype=float)
        body_acceleration = np.asarray(
            position_spline(self.time, 2), dtype=float
        )
        angular_velocity = np.asarray(
            rotation_spline(self.time, 1), dtype=float
        )
        angular_acceleration = np.asarray(
            rotation_spline(self.time, 2), dtype=float
        )

        pose_lever = (
            self.direct_problem.pose_sensor_position - parameters.cog_offset
        )
        sensor_position = corrected_position + np.einsum(
            "nij,j->ni", corrected_rotation, pose_lever
        )
        sensor_rotation = np.einsum(
            "nij,jk->nik",
            corrected_rotation,
            self.direct_problem.pose_body_to_sensor_rotation,
        )
        pose_residual = np.asarray(
            [
                strict.se3_log_error(
                    self.direct_problem.observations.sensor_position[index],
                    self.direct_problem.observed_sensor_rotation[index],
                    sensor_position[index],
                    sensor_rotation[index],
                )
                for index in range(self.time.size)
            ],
            dtype=float,
        )

        gravity_world = np.asarray((0.0, 0.0, -GRAVITY), dtype=float)
        plant = FullSixDofPlant(parameters, self.direct_problem.geometry)
        wrench_residual = np.empty((self.time.size, 6), dtype=float)
        for index, sample_time in enumerate(self.time):
            rigid = RigidBodyState(
                position=corrected_position[index],
                orientation_xyzw=matrix_to_quaternion(
                    corrected_rotation[index]
                ),
                linear_velocity=body_velocity[index],
                angular_velocity=angular_velocity[index],
            )
            actuator = ActuatorState(
                thrust=self.actuator_thrust[index],
                gimbal_angle=self.actuator_gimbal[index],
            )
            modeled = plant.total_body_wrench(
                float(sample_time), rigid, actuator
            )
            required_force = parameters.mass * (
                corrected_rotation[index].T
                @ (body_acceleration[index] - gravity_world)
            )
            required_torque = (
                parameters.inertia @ angular_acceleration[index]
                + np.cross(
                    angular_velocity[index],
                    parameters.inertia @ angular_velocity[index],
                )
            )
            wrench_residual[index, :3] = required_force - modeled[:3]
            wrench_residual[index, 3:] = required_torque - modeled[3:]

        smooth_second = self.basis.second @ coefficient_matrix
        arrays = (
            corrected_position,
            corrected_rotation,
            body_velocity,
            body_acceleration,
            angular_velocity,
            angular_acceleration,
            sensor_position,
            sensor_rotation,
            pose_residual,
            wrench_residual,
            smooth_second,
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError("analysis trajectory is non-finite")
        return AnalysisEvaluation(*arrays)

    def residual(
        self,
        coefficients: Sequence[float],
        parameters: VehicleParameters,
        include_smoothness: bool = True,
    ) -> np.ndarray:
        evaluation = self.evaluate(coefficients, parameters)
        sample_count = self.time.size
        pose = (
            np.einsum("ij,nj->ni", self.pose_factor, evaluation.pose_residual)
            / self.pose_scale
            / math.sqrt(sample_count)
        )
        # Natural cubic boundary acceleration is constrained by construction;
        # use interior samples for the dynamics term to avoid counting that
        # boundary condition as physical evidence.
        interior = slice(1, -1)
        dynamics_count = max(1, sample_count - 2)
        wrench_scale = np.asarray(
            (self.force_scale,) * 3 + (self.torque_scale,) * 3,
            dtype=float,
        )
        dynamics = (
            math.sqrt(self.lambda_dynamics)
            * evaluation.wrench_residual[interior]
            / wrench_scale
            / math.sqrt(dynamics_count)
        )
        blocks = [pose.ravel(), dynamics.ravel()]
        if include_smoothness:
            smooth_scale = np.asarray(
                (self.smooth_translation_scale,) * 3
                + (self.smooth_rotation_scale,) * 3,
                dtype=float,
            )
            smooth = (
                math.sqrt(self.lambda_smooth)
                * evaluation.smooth_second_derivative
                / smooth_scale
                / math.sqrt(sample_count)
            )
            blocks.append(smooth.ravel())
        return np.concatenate(blocks)


def open_uniform_spline_basis(
    time_axis: Sequence[float],
    coefficient_count: int,
    degree: int,
) -> SplineBasis:
    time_value = np.asarray(time_axis, dtype=float)
    if (
        time_value.ndim != 1
        or time_value.size < 3
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
        or degree < 2
        or coefficient_count <= degree
        or coefficient_count > time_value.size
    ):
        raise ValueError("spline basis settings are invalid")
    interior_count = coefficient_count - degree - 1
    if interior_count:
        interior = np.linspace(
            time_value[0], time_value[-1], interior_count + 2
        )[1:-1]
    else:
        interior = np.empty(0, dtype=float)
    knots = np.concatenate(
        (
            np.full(degree + 1, time_value[0]),
            interior,
            np.full(degree + 1, time_value[-1]),
        )
    )
    spline = BSpline(knots, np.eye(coefficient_count), degree, extrapolate=False)
    value = np.asarray(spline(time_value), dtype=float)
    first = np.asarray(spline.derivative(1)(time_value), dtype=float)
    second = np.asarray(spline.derivative(2)(time_value), dtype=float)
    return SplineBasis(knots, degree, value, first, second)


def _seed_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = payload.get("selection")
    if isinstance(selection, Mapping):
        return selection
    return payload


def load_parameter_seed(
    path: Optional[Path],
    fallback_delay: float,
    corrected_mass: Optional[float],
) -> ParameterSeed:
    source_path = None if path is None else path.expanduser().resolve()
    coordinate = np.zeros(strict.PHYSICAL_DIMENSION, dtype=float)
    delay = float(fallback_delay)
    source_kind = "nominal_fallback"
    if source_path is not None and source_path.is_file():
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
            selected = _seed_payload(raw)
            coordinate = np.asarray(selected["physical_coordinate"], dtype=float)
            delay = float(selected["delay_seconds"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "existing estimator result is not a compatible result.json: {}".format(
                    source_path
                )
            ) from error
        if (
            coordinate.shape != (strict.PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(coordinate))
            or not np.isfinite(delay)
            or delay < 0.0
        ):
            raise ValueError("existing estimator result contains invalid values")
        source_kind = "estimator_result"
    elif source_path is not None:
        print(
            "estimator result not found; using nominal parameters: {}".format(
                source_path
            ),
            flush=True,
        )

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
    return ParameterSeed(
        physical_coordinate=coordinate,
        delay_seconds=delay,
        source_kind=source_kind,
        source_path=source_path if source_kind == "estimator_result" else None,
        corrected_mass_kg=selected_mass,
        source_mass_kg=source_mass,
    )


def _decode_parameters(
    coordinate: Sequence[float],
    delay: float,
) -> VehicleParameters:
    parameterization = strict.FullyPhysicalInertiaParameterization(
        VehicleParameters.nominal()
    )
    return parameterization.decode(
        continuation._expand_coordinate(coordinate, delay)
    ).parameters


def _actuator_trajectory(
    direct_problem: baseline.DirectShootingProblem,
) -> tuple[np.ndarray, np.ndarray]:
    actuator_parameters = ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=0.0,
        delay=0.0,
    )
    state = direct_problem.initial_actuator_state
    sample_count = direct_problem.output_time.size
    thrust = np.empty((sample_count, 4), dtype=float)
    gimbal = np.empty((sample_count, 4), dtype=float)
    thrust[0] = state.thrust
    gimbal[0] = state.gimbal_angle
    output_index = 1
    half_step = 0.5 * direct_problem.integration_step
    for step_index, command in enumerate(direct_problem.commands):
        midpoint = advance_actuators(
            state, command, actuator_parameters, half_step
        )
        state = advance_actuators(
            midpoint, command, actuator_parameters, half_step
        )
        if (step_index + 1) % direct_problem.output_stride == 0:
            thrust[output_index] = state.thrust
            gimbal[output_index] = state.gimbal_angle
            output_index += 1
    if output_index != sample_count:
        raise RuntimeError("actuator and output grids disagree")
    return thrust, gimbal


def _make_bag_problem(
    specification: multi.BagSpecification,
    normalized_weight: float,
    flight: Any,
    seed: ParameterSeed,
    arguments: argparse.Namespace,
) -> BagProfileProblem:
    direct = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=seed.delay_seconds,
        prior_weight=0.0,
    )
    strict_problem = strict.MultipleShootingProblem(
        direct_problem=direct,
        delay=seed.delay_seconds,
        segment_duration=max(arguments.sample_step, 0.5),
        prior_weight=0.0,
        node_position_bound=np.inf,
        node_orientation_bound=np.inf,
        node_velocity_bound=np.inf,
        node_angular_velocity_bound=np.inf,
    )
    base_sensor_position, base_sensor_orientation, _residual = (
        strict_problem.full_rollout(seed.physical_coordinate)
    )
    parameters = _decode_parameters(
        seed.physical_coordinate, seed.delay_seconds
    )
    base_sensor_rotation = np.asarray(
        [quaternion_to_matrix(value) for value in base_sensor_orientation],
        dtype=float,
    )
    base_body_rotation = np.einsum(
        "nij,jk->nik",
        base_sensor_rotation,
        direct.pose_body_to_sensor_rotation.T,
    )
    pose_lever = direct.pose_sensor_position - parameters.cog_offset
    base_body_position = base_sensor_position - np.einsum(
        "nij,j->ni", base_body_rotation, pose_lever
    )
    coefficient_count = min(
        arguments.spline_knot_count,
        direct.output_time.size,
    )
    if coefficient_count <= arguments.spline_degree:
        raise ValueError("bag interval has too few samples for the spline")
    basis = open_uniform_spline_basis(
        direct.output_time,
        coefficient_count,
        arguments.spline_degree,
    )
    thrust, gimbal = _actuator_trajectory(direct)
    return BagProfileProblem(
        specification=specification,
        normalized_weight=float(normalized_weight),
        direct_problem=direct,
        strict_problem=strict_problem,
        basis=basis,
        base_body_position=base_body_position,
        base_body_rotation=base_body_rotation,
        actuator_thrust=thrust,
        actuator_gimbal=gimbal,
        pose_factor=strict.inertia_radius_se3_factor(VehicleParameters.nominal()),
        lambda_dynamics=arguments.lambda_dynamics,
        lambda_smooth=arguments.lambda_smooth,
        pose_scale=arguments.pose_scale,
        force_scale=arguments.force_residual_scale,
        torque_scale=arguments.torque_residual_scale,
        smooth_translation_scale=arguments.smooth_translation_scale,
        smooth_rotation_scale=arguments.smooth_rotation_scale,
    )


def _safe_profile_residual(
    problem: BagProfileProblem,
    coefficient: np.ndarray,
    parameters: VehicleParameters,
) -> np.ndarray:
    expected = problem.time.size * 6 + max(1, problem.time.size - 2) * 6
    expected += problem.time.size * 6
    try:
        residual = problem.residual(coefficient, parameters)
        if residual.size != expected or np.any(~np.isfinite(residual)):
            raise FloatingPointError("profile residual has invalid size or value")
        return residual
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(expected, 1.0e6 + np.linalg.norm(coefficient), dtype=float)


def _fit_trajectory(
    problem: BagProfileProblem,
    initial: np.ndarray,
    parameters: VehicleParameters,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    coefficient = problem.reshape_coefficients(initial)
    lower = np.empty(problem.coefficient_shape, dtype=float)
    upper = np.empty(problem.coefficient_shape, dtype=float)
    lower[:, :3] = -arguments.max_translation_correction
    upper[:, :3] = arguments.max_translation_correction
    lower[:, 3:] = -arguments.max_rotation_correction
    upper[:, 3:] = arguments.max_rotation_correction
    result = least_squares(
        lambda value: _safe_profile_residual(problem, value, parameters),
        coefficient.ravel(),
        bounds=(lower.ravel(), upper.ravel()),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss=arguments.loss,
        f_scale=arguments.robust_scale,
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.trajectory_max_nfev,
        verbose=0,
    )
    record = {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
    }
    return result.x.reshape(problem.coefficient_shape), record


def _theta_residual(
    variable: np.ndarray,
    mass_coordinate: float,
    problems: Sequence[BagProfileProblem],
    coefficients: Sequence[np.ndarray],
    seed_coordinate: np.ndarray,
    delay: float,
    prior_weight: float,
) -> np.ndarray:
    coordinate = np.concatenate(([mass_coordinate], np.asarray(variable, dtype=float)))
    try:
        parameters = _decode_parameters(coordinate, delay)
        blocks = []
        for problem, coefficient in zip(problems, coefficients):
            evaluation = problem.evaluate(coefficient, parameters)
            count = problem.time.size
            pose = (
                np.einsum("ij,nj->ni", problem.pose_factor, evaluation.pose_residual)
                / problem.pose_scale
                / math.sqrt(count)
            )
            wrench_scale = np.asarray(
                (problem.force_scale,) * 3 + (problem.torque_scale,) * 3,
                dtype=float,
            )
            dynamics = (
                math.sqrt(problem.lambda_dynamics)
                * evaluation.wrench_residual[1:-1]
                / wrench_scale
                / math.sqrt(max(1, count - 2))
            )
            root_weight = math.sqrt(problem.normalized_weight)
            blocks.extend((root_weight * pose.ravel(), root_weight * dynamics.ravel()))
        if prior_weight > 0.0:
            blocks.append(
                math.sqrt(prior_weight)
                * (coordinate[1:] - seed_coordinate[1:])
                / strict.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS[1:]
            )
        residual = np.concatenate(blocks)
        if np.any(~np.isfinite(residual)):
            raise FloatingPointError("theta residual is non-finite")
        return residual
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        size = sum(
            problem.time.size * 6 + max(1, problem.time.size - 2) * 6
            for problem in problems
        )
        if prior_weight > 0.0:
            size += strict.PHYSICAL_DIMENSION - 1
        return np.full(size, 1.0e6 + np.linalg.norm(variable), dtype=float)


def _update_parameters(
    coordinate: np.ndarray,
    problems: Sequence[BagProfileProblem],
    coefficients: Sequence[np.ndarray],
    seed_coordinate: np.ndarray,
    delay: float,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    scales = strict.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS[1:]
    lower = seed_coordinate[1:] - arguments.theta_bound_scale * scales
    upper = seed_coordinate[1:] + arguments.theta_bound_scale * scales
    initial = np.clip(coordinate[1:], lower, upper)
    result = least_squares(
        lambda value: _theta_residual(
            value,
            coordinate[0],
            problems,
            coefficients,
            seed_coordinate,
            delay,
            arguments.theta_prior_weight,
        ),
        initial,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss=arguments.loss,
        f_scale=arguments.robust_scale,
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.parameter_max_nfev,
        verbose=0,
    )
    updated = coordinate.copy()
    updated[1:] = result.x
    record = {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
    }
    return updated, record


def _pose_metrics(
    problem: BagProfileProblem,
    position: np.ndarray,
    rotation: np.ndarray,
) -> Mapping[str, float]:
    residual = np.asarray(
        [
            strict.se3_log_error(
                problem.direct_problem.observations.sensor_position[index],
                problem.direct_problem.observed_sensor_rotation[index],
                position[index],
                rotation[index],
            )
            for index in range(problem.time.size)
        ],
        dtype=float,
    )
    translation_norm = np.linalg.norm(residual[:, :3], axis=1)
    rotation_norm = np.linalg.norm(residual[:, 3:], axis=1)
    weighted = np.einsum("ij,nj->ni", problem.pose_factor, residual)
    return {
        "inertia_radius_loss_m2": 0.5 * float(np.mean(np.sum(weighted * weighted, axis=1))),
        "translation_rmse_m": float(np.sqrt(np.mean(translation_norm * translation_norm))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation_norm * rotation_norm))),
        "rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rotation_norm * rotation_norm)))),
        "translation_max_m": float(np.max(translation_norm)),
        "rotation_max_rad": float(np.max(rotation_norm)),
    }


def _wrench_metrics(residual: np.ndarray) -> Mapping[str, Any]:
    value = np.asarray(residual, dtype=float)
    interior = value[1:-1]
    return {
        "component_mean": np.mean(interior, axis=0),
        "component_rmse": np.sqrt(np.mean(interior * interior, axis=0)),
        "component_max_abs": np.max(np.abs(interior), axis=0),
        "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(interior[:, :3] ** 2, axis=1)))),
        "torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(interior[:, 3:] ** 2, axis=1)))),
    }


def _write_bag_pdf(
    path: Path,
    problem: BagProfileProblem,
    analysis: AnalysisEvaluation,
    free_position: np.ndarray,
    free_orientation: np.ndarray,
) -> None:
    observed = problem.direct_problem.observations
    relative_time = problem.time - problem.time[0]
    observed_rpy = baseline._rpy_series(observed.sensor_orientation_xyzw)
    analysis_quaternion = np.asarray(
        [matrix_to_quaternion(value) for value in analysis.sensor_rotation]
    )
    analysis_rpy = baseline._rpy_series(analysis_quaternion)
    free_rpy = baseline._rpy_series(free_orientation)
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        for label, value, color, style in (
            ("observed", observed.sensor_position, "#1e5abe", "-"),
            ("analysis", analysis.sensor_position, "#1e965f", ":"),
            ("free rollout", free_position, "#d2691e", "--"),
        ):
            axis.plot(
                value[:, 0],
                value[:, 1],
                value[:, 2],
                label=label,
                color=color,
                linestyle=style,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="best")
        axis.set_title("Analysis trajectory and correction-free rollout")
        pdf.savefig(figure)
        plt.close(figure)

        for title, observed_value, analysis_value, free_value, labels in (
            (
                "Sensor position",
                observed.sensor_position,
                analysis.sensor_position,
                free_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Sensor orientation",
                observed_rpy,
                analysis_rpy,
                free_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    observed_value[:, component],
                    label="observed",
                    color="#1e5abe",
                    linewidth=2.0,
                )
                axis.plot(
                    relative_time,
                    analysis_value[:, component],
                    label="analysis",
                    color="#1e965f",
                    linestyle=":",
                    linewidth=1.8,
                )
                axis.plot(
                    relative_time,
                    free_value[:, component],
                    label="free rollout",
                    color="#d2691e",
                    linestyle="--",
                    linewidth=1.4,
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes.ravel()):
            axis.plot(
                relative_time,
                analysis.wrench_residual[:, component],
                color="#8b4bb7",
            )
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            axis.set_ylabel(
                "{} [{}]".format(
                    WRENCH_COMPONENT_NAMES[component],
                    WRENCH_COMPONENT_UNITS[component],
                )
            )
            axis.grid(True, alpha=0.25)
        axes[0, 0].set_title("Required minus modeled body wrench")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        pdf.savefig(figure)
        plt.close(figure)


def _parameter_lines(
    seed: ParameterSeed,
    coordinate: np.ndarray,
    delay: float,
) -> list[str]:
    parameters = _decode_parameters(coordinate, delay)
    lines = [
        "Generalized-profiling shared parameters",
        "",
        "Seed source: {}".format(seed.source_kind),
        "Source mass [kg]: {:.12g}".format(seed.source_mass_kg),
        "Fixed corrected mass [kg]: {:.12g}".format(parameters.mass),
        "Command delay [s]: {:.12g}".format(delay),
        "Inertia [kg m^2]",
    ]
    lines.extend(
        "  [{: .12g}  {: .12g}  {: .12g}]".format(*row)
        for row in parameters.inertia
    )
    lines.extend(
        (
            "CoG offset [m]: [{:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.cog_offset
            ),
            "Force effectiveness: "
            "[{:.12g}, {:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.force_effectiveness
            ),
            "",
            "Physical coordinate",
        )
    )
    lines.extend(
        "  {:<43s} {: .12g}".format(name, value)
        for name, value in zip(strict.PHYSICAL_PARAMETER_NAMES, coordinate)
    )
    return lines


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit bag-local SE(3) correction splines with a rigid-body "
            "dynamics penalty, initialized by a multi-bag estimator result."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--estimator-result",
        type=Path,
        default=DEFAULT_ESTIMATOR_RESULT,
        help="multi-bag result.json; a missing file falls back to nominal",
    )
    parser.add_argument(
        "--corrected-mass",
        "--corrected-mass-kg",
        dest="corrected_mass",
        type=float,
        default=None,
        help="fixed corrected mass in kg, overriding the source result",
    )
    parser.add_argument(
        "--command-delay",
        type=float,
        default=None,
        help="override source/fallback command delay in seconds",
    )
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--spline-knot-count", type=int, default=14)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--lambda-dynamics", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=1.0e-3)
    parser.add_argument("--pose-scale", type=float, default=1.0)
    parser.add_argument("--force-residual-scale", type=float, default=10.0)
    parser.add_argument("--torque-residual-scale", type=float, default=1.0)
    parser.add_argument("--smooth-translation-scale", type=float, default=1.0)
    parser.add_argument("--smooth-rotation-scale", type=float, default=1.0)
    parser.add_argument("--max-translation-correction", type=float, default=2.0)
    parser.add_argument("--max-rotation-correction", type=float, default=1.5)
    parser.add_argument("--alternating-iterations", type=int, default=2)
    parser.add_argument("--trajectory-max-nfev", type=int, default=30)
    parser.add_argument("--parameter-max-nfev", type=int, default=20)
    parser.add_argument("--theta-prior-weight", type=float, default=0.1)
    parser.add_argument("--theta-bound-scale", type=float, default=3.0)
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="fit splines without updating inertia, CoG, or effectiveness",
    )
    parser.add_argument(
        "--zero-spline-initialization",
        action="store_true",
        help="start spline coefficients at zero instead of pose projection",
    )
    parser.add_argument("--loss", choices=("linear", "soft_l1", "huber"), default="soft_l1")
    parser.add_argument("--robust-scale", type=float, default=1.0)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.spline_knot_count,
        arguments.spline_degree,
        arguments.pose_scale,
        arguments.force_residual_scale,
        arguments.torque_residual_scale,
        arguments.smooth_translation_scale,
        arguments.smooth_rotation_scale,
        arguments.max_translation_correction,
        arguments.max_rotation_correction,
        arguments.alternating_iterations,
        arguments.trajectory_max_nfev,
        arguments.parameter_max_nfev,
        arguments.theta_bound_scale,
        arguments.robust_scale,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
    )
    if (
        any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or not np.isfinite(arguments.lambda_dynamics)
        or arguments.lambda_dynamics < 0.0
        or not np.isfinite(arguments.lambda_smooth)
        or arguments.lambda_smooth < 0.0
        or not np.isfinite(arguments.theta_prior_weight)
        or arguments.theta_prior_weight < 0.0
        or arguments.spline_degree < 2
        or arguments.spline_knot_count <= arguments.spline_degree
        or (
            arguments.command_delay is not None
            and (
                not np.isfinite(arguments.command_delay)
                or arguments.command_delay < 0.0
            )
        )
    ):
        raise SystemExit("generalized-profiling settings are invalid")


def run(arguments: argparse.Namespace) -> int:
    _validate_arguments(arguments)
    try:
        config = multi.load_multi_bag_config(arguments.config)
        seed = load_parameter_seed(
            arguments.estimator_result,
            config.initial_delay_seconds,
            arguments.corrected_mass,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.command_delay is not None:
        seed = ParameterSeed(
            physical_coordinate=seed.physical_coordinate,
            delay_seconds=float(arguments.command_delay),
            source_kind=seed.source_kind,
            source_path=seed.source_path,
            corrected_mass_kg=seed.corrected_mass_kg,
            source_mass_kg=seed.source_mass_kg,
        )
    print(
        "parameter seed: {}, mass={:.9g} kg, delay={:.6f} s".format(
            seed.source_kind, seed.corrected_mass_kg, seed.delay_seconds
        ),
        flush=True,
    )
    started = time.perf_counter()
    weights = multi._normalized_weights(config.bags)
    problems = []
    for index, (specification, weight) in enumerate(zip(config.bags, weights)):
        print(
            "loading bag {}/{} {}: {} [{:.3f}, {:.3f}] s".format(
                index + 1,
                len(config.bags),
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
        problems.append(
            _make_bag_problem(
                specification,
                float(weight),
                flight,
                seed,
                arguments,
            )
        )

    coordinate = seed.physical_coordinate.copy()
    parameters = _decode_parameters(coordinate, seed.delay_seconds)
    coefficients = [
        problem.initial_coefficients(
            parameters,
            not arguments.zero_spline_initialization,
            arguments.max_translation_correction,
            arguments.max_rotation_correction,
        )
        for problem in problems
    ]
    history = []
    for iteration in range(arguments.alternating_iterations):
        print(
            "profiling trajectories {}/{}".format(
                iteration + 1, arguments.alternating_iterations
            ),
            flush=True,
        )
        trajectory_records = []
        for bag_index, problem in enumerate(problems):
            coefficients[bag_index], record = _fit_trajectory(
                problem,
                coefficients[bag_index],
                parameters,
                arguments,
            )
            trajectory_records.append(dict(record, id=problem.specification.bag_id))
        iteration_record: dict[str, Any] = {
            "iteration": iteration,
            "trajectory_fits": trajectory_records,
            "input_physical_coordinate": coordinate.copy(),
        }
        if not arguments.trajectory_only:
            coordinate, theta_record = _update_parameters(
                coordinate,
                problems,
                coefficients,
                seed.physical_coordinate,
                seed.delay_seconds,
                arguments,
            )
            parameters = _decode_parameters(coordinate, seed.delay_seconds)
            iteration_record["parameter_update"] = theta_record
            iteration_record["output_physical_coordinate"] = coordinate.copy()
            print(
                "updated shared parameters (fixed mass {:.9g} kg)".format(
                    parameters.mass
                ),
                flush=True,
            )
        history.append(iteration_record)

    # The last outer update changed theta after the last inner solve.  Profile
    # once more so the reported analysis trajectories are minimizers for the
    # reported final parameters.
    if not arguments.trajectory_only:
        print("final trajectory profiling at updated parameters", flush=True)
        final_records = []
        for bag_index, problem in enumerate(problems):
            coefficients[bag_index], record = _fit_trajectory(
                problem,
                coefficients[bag_index],
                parameters,
                arguments,
            )
            final_records.append(dict(record, id=problem.specification.bag_id))
        history.append(
            {
                "iteration": "final_profile",
                "trajectory_fits": final_records,
                "physical_coordinate": coordinate.copy(),
            }
        )

    output_directory = arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    bags_directory = output_directory / "bags"
    bags_directory.mkdir(parents=True, exist_ok=True)
    bag_payloads = []
    bag_outputs: dict[str, Any] = {}
    weighted_analysis_loss = 0.0
    weighted_free_loss = 0.0
    for problem, coefficient in zip(problems, coefficients):
        analysis = problem.evaluate(coefficient, parameters)
        free_position, free_orientation, _ = (
            problem.strict_problem.full_rollout(coordinate)
        )
        free_rotation = np.asarray(
            [quaternion_to_matrix(value) for value in free_orientation], dtype=float
        )
        analysis_metrics = _pose_metrics(
            problem,
            analysis.sensor_position,
            analysis.sensor_rotation,
        )
        free_metrics = _pose_metrics(problem, free_position, free_rotation)
        weighted_analysis_loss += (
            problem.normalized_weight
            * analysis_metrics["inertia_radius_loss_m2"]
        )
        weighted_free_loss += (
            problem.normalized_weight
            * free_metrics["inertia_radius_loss_m2"]
        )
        bag_directory = bags_directory / problem.specification.bag_id
        bag_directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = bag_directory / "trajectory.pdf"
        arrays_path = bag_directory / "analysis.npz"
        result_path = bag_directory / "result.json"
        _write_bag_pdf(
            trajectory_path,
            problem,
            analysis,
            free_position,
            free_orientation,
        )
        np.savez_compressed(
            arrays_path,
            time=problem.time,
            spline_knots=problem.basis.knots,
            spline_coefficients=coefficient,
            analysis_body_position=analysis.body_position,
            analysis_body_rotation=analysis.body_rotation,
            analysis_body_velocity_world=analysis.body_velocity_world,
            analysis_body_acceleration_world=analysis.body_acceleration_world,
            analysis_angular_velocity_body=analysis.angular_velocity_body,
            analysis_angular_acceleration_body=analysis.angular_acceleration_body,
            analysis_sensor_position=analysis.sensor_position,
            analysis_sensor_rotation=analysis.sensor_rotation,
            free_sensor_position=free_position,
            free_sensor_orientation_xyzw=free_orientation,
            pose_residual_se3=analysis.pose_residual,
            required_minus_modeled_body_wrench=analysis.wrench_residual,
            actuator_thrust=problem.actuator_thrust,
            actuator_gimbal=problem.actuator_gimbal,
        )
        payload = {
            "schema": SCHEMA + "/bag-result",
            "id": problem.specification.bag_id,
            "source": {
                "path": str(problem.specification.path),
                "sha256": baseline._sha256(problem.specification.path),
                "requested_interval_seconds": [
                    problem.specification.start,
                    problem.specification.end,
                ],
                "raw_weight": problem.specification.weight,
                "normalized_weight": problem.normalized_weight,
            },
            "sample_count": int(problem.time.size),
            "spline": {
                "degree": problem.basis.degree,
                "coefficient_count": problem.basis.coefficient_count,
                "coefficient_shape": problem.coefficient_shape,
                "maximum_absolute_translation_coefficient_m": float(
                    np.max(np.abs(coefficient[:, :3]))
                ),
                "maximum_absolute_rotation_coefficient_rad": float(
                    np.max(np.abs(coefficient[:, 3:]))
                ),
            },
            "analysis_pose_metrics": analysis_metrics,
            "free_rollout_pose_metrics": free_metrics,
            "analysis_minus_free_loss_m2": (
                analysis_metrics["inertia_radius_loss_m2"]
                - free_metrics["inertia_radius_loss_m2"]
            ),
            "required_minus_modeled_body_wrench": _wrench_metrics(
                analysis.wrench_residual
            ),
            "outputs": {
                "trajectory_pdf": "trajectory.pdf",
                "analysis_npz": "analysis.npz",
            },
        }
        baseline._write_json(result_path, payload)
        bag_payloads.append(payload)
        relative = "bags/{}/".format(problem.specification.bag_id)
        bag_outputs[problem.specification.bag_id] = {
            "result_json": relative + "result.json",
            "trajectory_pdf": relative + "trajectory.pdf",
            "analysis_npz": relative + "analysis.npz",
        }

    parameters_path = output_directory / "parameters.txt"
    result_path = output_directory / "result.json"
    strict._write_text(
        parameters_path,
        _parameter_lines(seed, coordinate, seed.delay_seconds),
    )
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(arguments.config.expanduser().resolve()),
        "method": {
            "name": "multi_bag_generalized_profiling_se3_correction_spline",
            "analysis_trajectory": "T_bar(t) Exp(delta_xi(t)^)",
            "correction_basis": "open-uniform B-spline in se(3)",
            "continuous_derivatives": (
                "CubicSpline position and RotationSpline SO(3), "
                "analytic order 1 and 2"
            ),
            "dynamics_residual": "required minus modeled body wrench",
            "command_model": "strict causal ZOH with recorded gimbal rate limit",
            "mass_update": "fixed during parameter profiling",
            "parameter_update_enabled": not arguments.trajectory_only,
        },
        "initialization": {
            "kind": seed.source_kind,
            "estimator_result_path": None if seed.source_path is None else str(seed.source_path),
            "source_mass_kg": seed.source_mass_kg,
            "corrected_mass_kg": seed.corrected_mass_kg,
            "initial_physical_coordinate": seed.physical_coordinate,
            "command_delay_seconds": seed.delay_seconds,
            "missing_result_fallback": "nominal physical coordinate and config initial delay",
        },
        "settings": {
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
            "spline_knot_count_requested": arguments.spline_knot_count,
            "spline_degree": arguments.spline_degree,
            "lambda_dynamics": arguments.lambda_dynamics,
            "lambda_smooth": arguments.lambda_smooth,
            "pose_scale_m": arguments.pose_scale,
            "force_residual_scale_N": arguments.force_residual_scale,
            "torque_residual_scale_Nm": arguments.torque_residual_scale,
            "smooth_translation_scale_m_per_s2": arguments.smooth_translation_scale,
            "smooth_rotation_scale_rad_per_s2": arguments.smooth_rotation_scale,
            "alternating_iterations": arguments.alternating_iterations,
            "robust_loss": arguments.loss,
            "robust_scale": arguments.robust_scale,
        },
        "selection": {
            "physical_coordinate": coordinate,
            "parameters": strict._physical_payload(
                strict.FullyPhysicalInertiaParameterization(VehicleParameters.nominal()).decode(
                    continuation._expand_coordinate(coordinate, seed.delay_seconds)
                )
            ),
            "delay_seconds": seed.delay_seconds,
            "joint_analysis_inertia_radius_loss_m2": weighted_analysis_loss,
            "joint_free_rollout_inertia_radius_loss_m2": weighted_free_loss,
            "analysis_minus_free_loss_m2": weighted_analysis_loss - weighted_free_loss,
            "bags": bag_payloads,
        },
        "alternating_history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "result_json": "result.json",
            "parameters_text": "parameters.txt",
            "bags": bag_outputs,
        },
    }
    baseline._write_json(result_path, result)
    print(
        "joint analysis loss {:.9g} m^2; free-rollout loss {:.9g} m^2".format(
            weighted_analysis_loss, weighted_free_loss
        ),
        flush=True,
    )
    print("wrote {}".format(result_path), flush=True)
    print("wrote {}".format(parameters_path), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
