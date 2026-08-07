#!/usr/bin/env python3
"""Sobol multi-start extension of the deterministic recorded-control fit.

The estimator keeps the baseline's single, observation-reset-free, open-loop
trajectory objective.  It changes how that objective is initialized:

* a scrambled Sobol design explores bounded physical coordinates;
* an explicit Cholesky chart keeps inertia positive definite;
* mass, relative rotor effectiveness, and actuator time constants use logs;
* an optional plant-derived PID gain band can reject candidates before an
  expensive rollout;
* low-loss seeds are greedily separated in normalized parameter space; and
* independent bounded least-squares solves are run from at most sixteen seeds.

Each local solve uses exact forward sensitivities through the active actuator
branches, the RK4 rigid-body rollout, and every observation residual.  It does
not finite-difference the fifteen smooth coordinates.

The unchanged nominal-start deterministic baseline is also executed as an
explicit incumbent.  The final reported trajectory loss can therefore never
be worse than that baseline, even when its solution lies outside the Sobol
box or PID-retuning gate; constraint eligibility is reported separately.

Delay is part of the Sobol design but is held fixed inside each local solve.
Recorded commands are causal ZOH signals, so treating delay as an ordinary
smooth finite-difference coordinate would give a misleading local Jacobian.
The final comparison across seeds still selects delay from the configured
continuous bounded Sobol values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import deterministic_estimator as baseline
from grape_param_estim.controller_config import PID_GROUPS
from grape_param_estim.dynamics import (
    actuator_wrench_with_jacobian,
    advance_actuators_with_jacobian,
)
from grape_param_estim.geometry import (
    normalise_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    skew,
    so3_log,
    so3_right_jacobian_inverse,
)
from grape_param_estim.pid.proposal import (
    closed_loop_acceleration_response,
    group_compensation_scales,
)
from grape_param_estim.system import (
    GRAVITY,
    ActuatorParameters,
    ActuatorState,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/minimal-deterministic-sobol/v4"
OUTPUT_SUBDIRECTORY = "deterministic_sobol"
CURRENT_THRUST_TIME_CONSTANT = 0.01
CURRENT_GIMBAL_TIME_CONSTANT = 0.02
DELAY_INDEX = 15
SMOOTH_DIMENSION = DELAY_INDEX
SEARCH_DIMENSION = DELAY_INDEX + 1
SEARCH_PARAMETER_NAMES = (
    "log_mass_scale",
    "log_cholesky_xx_scale",
    "log_cholesky_yy_scale",
    "log_cholesky_zz_scale",
    "normalized_cholesky_yx_offset",
    "normalized_cholesky_zx_offset",
    "normalized_cholesky_zy_offset",
    "cog_offset_x_m",
    "cog_offset_y_m",
    "cog_offset_z_m",
    "force_effectiveness_contrast_1",
    "force_effectiveness_contrast_2",
    "force_effectiveness_contrast_3",
    "log_thrust_time_constant_scale",
    "log_gimbal_time_constant_scale",
    "command_delay_seconds",
)
_CHOLESKY_OFF_DIAGONALS = ((1, 0), (2, 0), (2, 1))


@dataclass(frozen=True)
class SearchBounds:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if (
            lower.shape != (SEARCH_DIMENSION,)
            or upper.shape != (SEARCH_DIMENSION,)
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(upper <= lower)
        ):
            raise ValueError("search bounds must be finite increasing 16-vectors")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def contains(self, coordinate: Sequence[float]) -> bool:
        value = np.asarray(coordinate, dtype=float)
        return bool(
            value.shape == (SEARCH_DIMENSION,)
            and np.all(np.isfinite(value))
            and np.all(value >= self.lower)
            and np.all(value <= self.upper)
        )

    def from_unit_cube(self, points: np.ndarray) -> np.ndarray:
        value = np.asarray(points, dtype=float)
        if (
            value.ndim != 2
            or value.shape[1] != SEARCH_DIMENSION
            or np.any(~np.isfinite(value))
            or np.any(value < 0.0)
            or np.any(value > 1.0)
        ):
            raise ValueError("Sobol points must lie in the 16-D unit cube")
        return self.lower[None, :] + value * self.width[None, :]

    def normalized(self, coordinate: Sequence[float]) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (SEARCH_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("coordinate must be a finite 16-vector")
        return (value - self.lower) / self.width

    def as_mapping(self) -> dict[str, list[float]]:
        return {
            name: [float(lower), float(upper)]
            for name, lower, upper in zip(
                SEARCH_PARAMETER_NAMES, self.lower, self.upper
            )
        }


@dataclass(frozen=True)
class DecodedSearchPoint:
    parameters: VehicleParameters
    actuator_parameters: ActuatorParameters
    delay: float
    inertia_principal_moments: np.ndarray
    inertia_triangle_margin: float


@dataclass(frozen=True)
class DecodedSearchJacobian:
    """Exact derivatives of physical values with respect to 15 smooth coordinates."""

    mass: np.ndarray
    inertia: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray
    thrust_time_constant: np.ndarray
    gimbal_time_constant: np.ndarray


class PhysicalSearchParameterization:
    """Bounded chart centered on the recorded controller's current plant."""

    def __init__(
        self,
        nominal: VehicleParameters,
        *,
        current_thrust_time_constant: float = CURRENT_THRUST_TIME_CONSTANT,
        current_gimbal_time_constant: float = CURRENT_GIMBAL_TIME_CONSTANT,
    ) -> None:
        if not isinstance(nominal, VehicleParameters):
            raise TypeError("nominal must be VehicleParameters")
        if current_thrust_time_constant <= 0.0 or current_gimbal_time_constant <= 0.0:
            raise ValueError("current actuator time constants must be positive")
        self.nominal = nominal
        self.nominal_cholesky = np.linalg.cholesky(nominal.inertia)
        self.current_thrust_time_constant = float(current_thrust_time_constant)
        self.current_gimbal_time_constant = float(current_gimbal_time_constant)

    def current_coordinate(self, delay: float) -> np.ndarray:
        selected_delay = float(delay)
        if not np.isfinite(selected_delay) or selected_delay < 0.0:
            raise ValueError("current delay must be finite and non-negative")
        result = np.zeros(SEARCH_DIMENSION, dtype=float)
        result[DELAY_INDEX] = selected_delay
        return result

    def decode(self, coordinate: Sequence[float]) -> DecodedSearchPoint:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (SEARCH_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("search coordinate must be a finite 16-vector")
        cholesky = self.nominal_cholesky.copy()
        cholesky[(0, 1, 2), (0, 1, 2)] *= np.exp(value[1:4])
        for local_index, (row, column) in enumerate(_CHOLESKY_OFF_DIAGONALS):
            scale = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            cholesky[row, column] += value[4 + local_index] * scale
        inertia = cholesky @ cholesky.T
        inertia = 0.5 * (inertia + inertia.T)
        principal = np.linalg.eigvalsh(inertia)
        triangle_margin = float(principal[0] + principal[1] - principal[2])
        log_effectiveness = (
            baseline.FORCE_EFFECTIVENESS_CONTRAST_BASIS @ value[10:13]
        )
        parameters = VehicleParameters(
            mass=self.nominal.mass * math.exp(float(value[0])),
            inertia=inertia,
            cog_offset=self.nominal.cog_offset + value[7:10],
            force_effectiveness=(
                self.nominal.force_effectiveness * np.exp(log_effectiveness)
            ),
            torque_effectiveness=self.nominal.torque_effectiveness,
            linear_drag=self.nominal.linear_drag,
            angular_drag=self.nominal.angular_drag,
        )
        actuator_parameters = ActuatorParameters(
            thrust_time_constant=(
                self.current_thrust_time_constant * math.exp(float(value[13]))
            ),
            gimbal_time_constant=(
                self.current_gimbal_time_constant * math.exp(float(value[14]))
            ),
            # Delay is applied explicitly to the recorded ZOH commands by the
            # baseline DirectShootingProblem constructor.
            delay=0.0,
        )
        return DecodedSearchPoint(
            parameters=parameters,
            actuator_parameters=actuator_parameters,
            delay=float(value[DELAY_INDEX]),
            inertia_principal_moments=principal,
            inertia_triangle_margin=triangle_margin,
        )

    def decode_with_jacobian(
        self, coordinate: Sequence[float]
    ) -> tuple[DecodedSearchPoint, DecodedSearchJacobian]:
        """Decode the physical chart and differentiate its smooth coordinates."""

        value = np.asarray(coordinate, dtype=float)
        decoded = self.decode(value)
        dimension = SMOOTH_DIMENSION

        mass = np.zeros(dimension, dtype=float)
        mass[0] = decoded.parameters.mass

        cholesky = self.nominal_cholesky.copy()
        cholesky[(0, 1, 2), (0, 1, 2)] *= np.exp(value[1:4])
        for local_index, (row, column) in enumerate(_CHOLESKY_OFF_DIAGONALS):
            scale = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            cholesky[row, column] += value[4 + local_index] * scale
        inertia = np.zeros((3, 3, dimension), dtype=float)
        for axis in range(3):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[axis, axis] = cholesky[axis, axis]
            inertia[:, :, 1 + axis] = (
                derivative @ cholesky.T
                + cholesky @ derivative.T
            )
        for local_index, (row, column) in enumerate(_CHOLESKY_OFF_DIAGONALS):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[row, column] = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            inertia[:, :, 4 + local_index] = (
                derivative @ cholesky.T
                + cholesky @ derivative.T
            )

        cog_offset = np.zeros((3, dimension), dtype=float)
        cog_offset[:, 7:10] = np.eye(3)

        force_effectiveness = np.zeros((4, dimension), dtype=float)
        force_effectiveness[:, 10:13] = (
            decoded.parameters.force_effectiveness[:, None]
            * baseline.FORCE_EFFECTIVENESS_CONTRAST_BASIS
        )

        thrust_time_constant = np.zeros(dimension, dtype=float)
        thrust_time_constant[13] = (
            decoded.actuator_parameters.thrust_time_constant
        )
        gimbal_time_constant = np.zeros(dimension, dtype=float)
        gimbal_time_constant[14] = (
            decoded.actuator_parameters.gimbal_time_constant
        )
        return decoded, DecodedSearchJacobian(
            mass=mass,
            inertia=inertia,
            cog_offset=cog_offset,
            force_effectiveness=force_effectiveness,
            thrust_time_constant=thrust_time_constant,
            gimbal_time_constant=gimbal_time_constant,
        )

    def encode(
        self,
        parameters: VehicleParameters,
        actuator_parameters: ActuatorParameters,
        delay: float,
    ) -> np.ndarray:
        """Map represented physical values back to the 16-D search chart."""

        if not isinstance(parameters, VehicleParameters) or not isinstance(
            actuator_parameters, ActuatorParameters
        ):
            raise TypeError("physical search encoding requires parameter objects")
        if not np.allclose(
            parameters.torque_effectiveness,
            self.nominal.torque_effectiveness,
            rtol=1.0e-12,
            atol=1.0e-15,
        ) or not np.allclose(
            parameters.linear_drag,
            self.nominal.linear_drag,
            rtol=1.0e-12,
            atol=1.0e-15,
        ) or not np.allclose(
            parameters.angular_drag,
            self.nominal.angular_drag,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("fixed torque-effectiveness and drag values disagree")
        cholesky = np.linalg.cholesky(parameters.inertia)
        result = np.empty(SEARCH_DIMENSION, dtype=float)
        result[0] = math.log(parameters.mass / self.nominal.mass)
        result[1:4] = np.log(
            np.diag(cholesky) / np.diag(self.nominal_cholesky)
        )
        for local_index, (row, column) in enumerate(_CHOLESKY_OFF_DIAGONALS):
            scale = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            result[4 + local_index] = (
                cholesky[row, column]
                - self.nominal_cholesky[row, column]
            ) / scale
        result[7:10] = parameters.cog_offset - self.nominal.cog_offset
        log_effectiveness = np.log(
            parameters.force_effectiveness
            / self.nominal.force_effectiveness
        )
        if not np.isclose(
            np.sum(log_effectiveness), 0.0, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("force effectiveness geometric mean is not represented")
        result[10:13] = (
            baseline.FORCE_EFFECTIVENESS_CONTRAST_BASIS.T
            @ log_effectiveness
        )
        result[13] = math.log(
            actuator_parameters.thrust_time_constant
            / self.current_thrust_time_constant
        )
        result[14] = math.log(
            actuator_parameters.gimbal_time_constant
            / self.current_gimbal_time_constant
        )
        result[DELAY_INDEX] = float(delay)
        if np.any(~np.isfinite(result)):
            raise ValueError("physical values are not representable in search chart")
        return result


@dataclass(frozen=True)
class PidGainGate:
    current: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    enabled: bool = True

    def __post_init__(self) -> None:
        current = np.asarray(self.current, dtype=float)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if (
            current.shape != (4, 3)
            or lower.shape != current.shape
            or upper.shape != current.shape
            or np.any(~np.isfinite(current))
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(current < 0.0)
            or np.any(lower < 0.0)
            or np.any(upper < lower)
            or np.any(current < lower)
            or np.any(current > upper)
        ):
            raise ValueError(
                "PID gain gate must contain valid current/lower/upper gains"
            )
        object.__setattr__(self, "current", current.copy())
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def disabled(cls, current: Sequence[Sequence[float]]) -> "PidGainGate":
        """Retain gain prediction for reporting without imposing a range."""

        values = np.asarray(current, dtype=float)
        return cls(values, values, values, enabled=False)

    @classmethod
    def from_scale_band(
        cls,
        current: Sequence[Sequence[float]],
        lower_scale: float,
        upper_scale: float,
    ) -> "PidGainGate":
        values = np.asarray(current, dtype=float)
        if (
            not np.isfinite(lower_scale)
            or not np.isfinite(upper_scale)
            or lower_scale <= 0.0
            or lower_scale > 1.0
            or upper_scale < 1.0
            or upper_scale <= lower_scale
        ):
            raise ValueError("PID scale band must bracket one with positive width")
        return cls(values, lower_scale * values, upper_scale * values)

    def evaluate(
        self,
        parameters: VehicleParameters,
        controller_nominal: VehicleParameters,
        geometry: Any,
    ) -> dict[str, Any]:
        response = closed_loop_acceleration_response(
            parameters, controller_nominal, geometry
        )
        scales = group_compensation_scales(response)
        gains = self.current * scales[:, None]
        if not self.enabled:
            return {
                "valid": True,
                "group_scales": scales,
                "gains": gains,
                "acceleration_response": response,
                "constraint_residual": np.zeros(24, dtype=float),
            }
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(self.current))
        valid = bool(
            np.all(gains >= self.lower - tolerance)
            and np.all(gains <= self.upper + tolerance)
        )
        span = self.upper - self.lower
        normalization = np.maximum(
            span,
            np.maximum(0.01 * np.abs(self.current), 1.0e-12),
        )
        lower_violation = np.maximum(self.lower - gains, 0.0) / normalization
        upper_violation = np.maximum(gains - self.upper, 0.0) / normalization
        return {
            "valid": valid,
            "group_scales": scales,
            "gains": gains,
            "acceleration_response": response,
            "constraint_residual": np.concatenate(
                (lower_violation.ravel(), upper_violation.ravel())
            ),
        }


def generate_sobol_coordinates(
    bounds: SearchBounds,
    *,
    power: int,
    seed: int,
    current_coordinate: Sequence[float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if power < 0:
        raise ValueError("Sobol power must be non-negative")
    current = np.asarray(current_coordinate, dtype=float)
    if not bounds.contains(current):
        raise ValueError("current coordinate must lie inside the search bounds")
    sampler = qmc.Sobol(d=SEARCH_DIMENSION, scramble=True, seed=int(seed))
    sobol = bounds.from_unit_cube(sampler.random_base2(power))
    coordinates = np.vstack((current, sobol))
    sources = ("current",) + tuple(
        "sobol_{:05d}".format(index) for index in range(sobol.shape[0])
    )
    return coordinates, sources


def select_diverse_candidate_indices(
    coordinates: np.ndarray,
    losses: np.ndarray,
    eligible_indices: Sequence[int],
    bounds: SearchBounds,
    *,
    count: int,
    minimum_distance: float,
    required_indices: Sequence[int] = (),
) -> tuple[int, ...]:
    values = np.asarray(coordinates, dtype=float)
    scores = np.asarray(losses, dtype=float)
    eligible = tuple(int(value) for value in eligible_indices)
    required = tuple(int(value) for value in required_indices)
    if (
        values.ndim != 2
        or values.shape[1] != SEARCH_DIMENSION
        or scores.shape != (values.shape[0],)
        or count < 1
        or not np.isfinite(minimum_distance)
        or minimum_distance < 0.0
        or any(index < 0 or index >= values.shape[0] for index in eligible)
        or any(index not in eligible for index in required)
        or len(set(required)) != len(required)
        or len(required) > count
    ):
        raise ValueError("diverse candidate selection inputs are invalid")
    ordered = sorted(eligible, key=lambda index: (scores[index], index))
    selected: list[int] = list(required)
    selected_normalized: list[np.ndarray] = [
        bounds.normalized(values[index]) for index in required
    ]
    for index in ordered:
        if len(selected) >= count:
            break
        if index in selected:
            continue
        normalized = bounds.normalized(values[index])
        if all(
            np.linalg.norm(normalized - previous) >= minimum_distance
            for previous in selected_normalized
        ):
            selected.append(index)
            selected_normalized.append(normalized)
            if len(selected) >= count:
                break
    return tuple(sorted(selected, key=lambda index: (scores[index], index)))


def _trajectory_residual(
    problem: baseline.DirectShootingProblem,
    simulation: baseline.Simulation,
) -> np.ndarray:
    errors = problem.error_blocks(simulation)
    count_scale = math.sqrt(problem.output_time.size)
    return np.concatenate(
        (
            (
                errors["position"]
                / problem.residual_scales["position_m"]
                / count_scale
            ).ravel(),
            (
                errors["orientation"]
                / problem.residual_scales["orientation_rad"]
                / count_scale
            ).ravel(),
            (
                errors["velocity"]
                / problem.residual_scales["velocity_m_per_s"]
                / count_scale
            ).ravel(),
            (
                errors["angular_velocity"]
                / problem.residual_scales["angular_velocity_rad_per_s"]
                / count_scale
            ).ravel(),
            (
                errors["specific_force"]
                / problem.residual_scales["specific_force_m_per_s2"]
                / count_scale
            ).ravel(),
        )
    )


def _trajectory_loss(residual: np.ndarray) -> float:
    value = np.asarray(residual, dtype=float)
    return 0.5 * float(value @ value)


def _normalise_quaternion_with_jacobian(
    quaternion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``normalise_quaternion`` and return its active-branch Jacobian."""

    raw = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(raw))
    if raw.shape != (4,) or not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("quaternion must have finite positive norm")
    sign = -1.0 if raw[3] < 0.0 else 1.0
    normalized = normalise_quaternion(raw)
    jacobian = sign * (
        np.eye(4) / norm - np.outer(raw, raw) / norm**3
    )
    return normalized, jacobian


def _quaternion_right_tangent_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Map a small body-right rotation vector to an ``xyzw`` perturbation."""

    value = np.asarray(quaternion, dtype=float)
    result = np.empty((4, 3), dtype=float)
    result[:3] = value[3] * np.eye(3) + skew(value[:3])
    result[3] = -value[:3]
    return result


def _actuator_step_with_sensitivity(
    state: ActuatorState,
    sensitivity: np.ndarray,
    command: Any,
    decoded: DecodedSearchPoint,
    parameter_jacobian: DecodedSearchJacobian,
    time_step: float,
) -> tuple[ActuatorState, np.ndarray]:
    """Advance one active actuator branch and propagate exact sensitivities."""

    value = np.asarray(sensitivity, dtype=float)
    if value.shape != (8, SMOOTH_DIMENSION,):
        raise ValueError("actuator sensitivity has the wrong shape")
    evaluation = advance_actuators_with_jacobian(
        state,
        command,
        decoded.actuator_parameters,
        time_step,
    )
    jacobian = evaluation.jacobian
    result = np.empty_like(value)
    result[:4] = jacobian.thrust_previous @ value[:4]
    result[4:] = jacobian.gimbal_previous @ value[4:]

    thrust_tau = decoded.actuator_parameters.thrust_time_constant
    thrust_target = np.clip(
        command.thrust,
        decoded.actuator_parameters.minimum_thrust,
        decoded.actuator_parameters.maximum_thrust,
    )
    thrust_fraction_tau = (
        -math.exp(-time_step / thrust_tau) * time_step / thrust_tau**2
    )
    result[:4] += np.outer(
        thrust_fraction_tau * (thrust_target - state.thrust),
        parameter_jacobian.thrust_time_constant,
    )

    gimbal_tau = decoded.actuator_parameters.gimbal_time_constant
    gimbal_target = np.clip(
        command.gimbal_angle,
        -decoded.actuator_parameters.maximum_gimbal_angle,
        decoded.actuator_parameters.maximum_gimbal_angle,
    )
    gimbal_fraction_tau = (
        -math.exp(-time_step / gimbal_tau) * time_step / gimbal_tau**2
    )
    rate_free = ~(
        evaluation.active_set["gimbal_rate_lower"]
        | evaluation.active_set["gimbal_rate_upper"]
    )
    angle_free = ~(
        evaluation.active_set["gimbal_angle_lower"]
        | evaluation.active_set["gimbal_angle_upper"]
    )
    active_derivative = (rate_free & angle_free).astype(float)
    result[4:] += np.outer(
        active_derivative
        * gimbal_fraction_tau
        * (gimbal_target - state.gimbal_angle),
        parameter_jacobian.gimbal_time_constant,
    )
    return evaluation.next_state, result


def _body_wrench_with_sensitivity(
    problem: baseline.DirectShootingProblem,
    decoded: DecodedSearchPoint,
    parameter_jacobian: DecodedSearchJacobian,
    rotation: np.ndarray,
    rotation_right_sensitivity: np.ndarray,
    linear_velocity: np.ndarray,
    linear_velocity_sensitivity: np.ndarray,
    angular_velocity: np.ndarray,
    angular_velocity_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the body wrench and its total smooth-coordinate derivative."""

    parameters = decoded.parameters
    wrench, jacobian = actuator_wrench_with_jacobian(
        actuators, parameters, problem.geometry
    )
    sensitivity = (
        jacobian.actual_thrust @ actuator_sensitivity[:4]
        + jacobian.actual_gimbal_angle @ actuator_sensitivity[4:]
        + jacobian.cog_offset @ parameter_jacobian.cog_offset
        + jacobian.force_effectiveness
        @ parameter_jacobian.force_effectiveness
    )
    body_velocity = rotation.T @ linear_velocity
    body_velocity_sensitivity = (
        skew(body_velocity) @ rotation_right_sensitivity
        + rotation.T @ linear_velocity_sensitivity
    )
    linear_drag = np.diag(parameters.linear_drag)
    angular_drag = np.diag(parameters.angular_drag)
    wrench[:3] -= parameters.linear_drag * body_velocity
    wrench[3:] -= parameters.angular_drag * angular_velocity
    sensitivity[:3] -= linear_drag @ body_velocity_sensitivity
    sensitivity[3:] -= angular_drag @ angular_velocity_sensitivity
    return wrench, sensitivity


def _rigid_derivative_with_sensitivity(
    problem: baseline.DirectShootingProblem,
    decoded: DecodedSearchPoint,
    parameter_jacobian: DecodedSearchJacobian,
    state_vector: np.ndarray,
    state_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous six-DoF derivative and exact forward sensitivity."""

    value = np.asarray(state_vector, dtype=float)
    sensitivity = np.asarray(state_sensitivity, dtype=float)
    if value.shape != (13,) or sensitivity.shape != (13, SMOOTH_DIMENSION):
        raise ValueError("rigid state or sensitivity has the wrong shape")
    quaternion, normalization = _normalise_quaternion_with_jacobian(
        value[3:7]
    )
    quaternion_sensitivity = normalization @ sensitivity[3:7]
    quaternion_tangent = _quaternion_right_tangent_matrix(quaternion)
    rotation_right_sensitivity = (
        2.0 * quaternion_tangent.T @ quaternion_sensitivity
    )
    rotation = quaternion_to_matrix(quaternion)
    linear_velocity = value[7:10]
    angular_velocity = value[10:13]
    linear_velocity_sensitivity = sensitivity[7:10]
    angular_velocity_sensitivity = sensitivity[10:13]
    wrench, wrench_sensitivity = _body_wrench_with_sensitivity(
        problem,
        decoded,
        parameter_jacobian,
        rotation,
        rotation_right_sensitivity,
        linear_velocity,
        linear_velocity_sensitivity,
        angular_velocity,
        angular_velocity_sensitivity,
        actuators,
        actuator_sensitivity,
    )

    pure_omega = np.concatenate((angular_velocity, np.asarray((0.0,))))
    quaternion_rate = 0.5 * quaternion_multiply(quaternion, pure_omega)
    quaternion_left_jacobian = np.empty((4, 4), dtype=float)
    quaternion_left_jacobian[:3, :3] = -skew(angular_velocity)
    quaternion_left_jacobian[:3, 3] = angular_velocity
    quaternion_left_jacobian[3, :3] = -angular_velocity
    quaternion_left_jacobian[3, 3] = 0.0
    quaternion_rate_sensitivity = 0.5 * (
        quaternion_left_jacobian @ quaternion_sensitivity
        + quaternion_tangent @ angular_velocity_sensitivity
    )

    parameters = decoded.parameters
    force = wrench[:3]
    force_sensitivity = wrench_sensitivity[:3]
    force_per_mass = force / parameters.mass
    linear_acceleration = (
        np.asarray((0.0, 0.0, -GRAVITY)) + rotation @ force_per_mass
    )
    linear_acceleration_sensitivity = (
        -rotation @ skew(force_per_mass) @ rotation_right_sensitivity
        + rotation
        @ (
            force_sensitivity / parameters.mass
            - np.outer(force, parameter_jacobian.mass)
            / parameters.mass**2
        )
    )

    inertia = parameters.inertia
    inertia_omega = inertia @ angular_velocity
    angular_acceleration = np.linalg.solve(
        inertia,
        wrench[3:] - np.cross(angular_velocity, inertia_omega),
    )
    inertia_omega_sensitivity = (
        np.einsum("ijk,j->ik", parameter_jacobian.inertia, angular_velocity)
        + inertia @ angular_velocity_sensitivity
    )
    angular_rhs_sensitivity = (
        wrench_sensitivity[3:]
        + skew(inertia_omega) @ angular_velocity_sensitivity
        - skew(angular_velocity) @ inertia_omega_sensitivity
    )
    inertia_alpha_sensitivity = np.einsum(
        "ijk,j->ik", parameter_jacobian.inertia, angular_acceleration
    )
    angular_acceleration_sensitivity = np.linalg.solve(
        inertia,
        angular_rhs_sensitivity - inertia_alpha_sensitivity,
    )
    derivative = np.concatenate(
        (
            linear_velocity,
            quaternion_rate,
            linear_acceleration,
            angular_acceleration,
        )
    )
    derivative_sensitivity = np.vstack(
        (
            linear_velocity_sensitivity,
            quaternion_rate_sensitivity,
            linear_acceleration_sensitivity,
            angular_acceleration_sensitivity,
        )
    )
    return derivative, derivative_sensitivity


def _rigid_step_with_sensitivity(
    problem: baseline.DirectShootingProblem,
    decoded: DecodedSearchPoint,
    parameter_jacobian: DecodedSearchJacobian,
    state: RigidBodyState,
    state_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
    time_step: float,
) -> tuple[RigidBodyState, np.ndarray]:
    """Match the production RK4 step while propagating forward sensitivity."""

    vector = state.as_vector()
    sensitivity = np.asarray(state_sensitivity, dtype=float)
    k1, j1 = _rigid_derivative_with_sensitivity(
        problem,
        decoded,
        parameter_jacobian,
        vector,
        sensitivity,
        actuators,
        actuator_sensitivity,
    )
    k2, j2 = _rigid_derivative_with_sensitivity(
        problem,
        decoded,
        parameter_jacobian,
        vector + 0.5 * time_step * k1,
        sensitivity + 0.5 * time_step * j1,
        actuators,
        actuator_sensitivity,
    )
    k3, j3 = _rigid_derivative_with_sensitivity(
        problem,
        decoded,
        parameter_jacobian,
        vector + 0.5 * time_step * k2,
        sensitivity + 0.5 * time_step * j2,
        actuators,
        actuator_sensitivity,
    )
    k4, j4 = _rigid_derivative_with_sensitivity(
        problem,
        decoded,
        parameter_jacobian,
        vector + time_step * k3,
        sensitivity + time_step * j3,
        actuators,
        actuator_sensitivity,
    )
    next_vector = vector + time_step / 6.0 * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )
    next_sensitivity = sensitivity + time_step / 6.0 * (
        j1 + 2.0 * j2 + 2.0 * j3 + j4
    )
    next_vector[3:7], normalization = _normalise_quaternion_with_jacobian(
        next_vector[3:7]
    )
    next_sensitivity[3:7] = normalization @ next_sensitivity[3:7]
    return RigidBodyState.from_vector(next_vector), next_sensitivity


def _simulate_trajectory_with_jacobian(
    problem: baseline.DirectShootingProblem,
    decoded: DecodedSearchPoint,
    parameter_jacobian: DecodedSearchJacobian,
) -> tuple[baseline.Simulation, np.ndarray, np.ndarray]:
    """Run one trajectory and analytically linearize all observation residuals."""

    parameters = decoded.parameters
    problem.actuator_parameters = decoded.actuator_parameters
    rigid = problem._initial_rigid_state(parameters)
    rigid_sensitivity = np.zeros((13, SMOOTH_DIMENSION), dtype=float)
    rigid_sensitivity[:3] = (
        problem.initial_body_rotation @ parameter_jacobian.cog_offset
    )
    rigid_sensitivity[7:10] = (
        problem.initial_body_rotation
        @ skew(problem.initial_omega_body)
        @ parameter_jacobian.cog_offset
    )
    actuators = problem.initial_actuator_state
    actuator_sensitivity = np.zeros((8, SMOOTH_DIMENSION), dtype=float)
    output_count = problem.output_time.size
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
    output_jacobian = {
        name: np.empty((output_count, 3, SMOOTH_DIMENSION), dtype=float)
        for name in (
            "position",
            "orientation",
            "velocity",
            "angular_velocity",
            "specific_force",
        )
    }

    def store(output_index: int) -> None:
        quaternion = rigid.orientation_xyzw
        rotation = quaternion_to_matrix(quaternion)
        quaternion_tangent = _quaternion_right_tangent_matrix(quaternion)
        rotation_right_sensitivity = (
            2.0 * quaternion_tangent.T @ rigid_sensitivity[3:7]
        )
        pose_lever = problem.pose_sensor_position - parameters.cog_offset
        velocity_lever = (
            problem.velocity_sensor_position - parameters.cog_offset
        )
        imu_lever = problem.imu_sensor_position - parameters.cog_offset
        wrench, wrench_sensitivity = _body_wrench_with_sensitivity(
            problem,
            decoded,
            parameter_jacobian,
            rotation,
            rotation_right_sensitivity,
            rigid.linear_velocity,
            rigid_sensitivity[7:10],
            rigid.angular_velocity,
            rigid_sensitivity[10:13],
            actuators,
            actuator_sensitivity,
        )
        inertia = parameters.inertia
        omega = rigid.angular_velocity
        omega_sensitivity = rigid_sensitivity[10:13]
        inertia_omega = inertia @ omega
        angular_acceleration = np.linalg.solve(
            inertia,
            wrench[3:] - np.cross(omega, inertia_omega),
        )
        inertia_omega_sensitivity = (
            np.einsum("ijk,j->ik", parameter_jacobian.inertia, omega)
            + inertia @ omega_sensitivity
        )
        angular_rhs_sensitivity = (
            wrench_sensitivity[3:]
            + skew(inertia_omega) @ omega_sensitivity
            - skew(omega) @ inertia_omega_sensitivity
        )
        angular_acceleration_sensitivity = np.linalg.solve(
            inertia,
            angular_rhs_sensitivity
            - np.einsum(
                "ijk,j->ik",
                parameter_jacobian.inertia,
                angular_acceleration,
            ),
        )
        force_per_mass_sensitivity = (
            wrench_sensitivity[:3] / parameters.mass
            - np.outer(wrench[:3], parameter_jacobian.mass)
            / parameters.mass**2
        )
        alpha_cross_lever_sensitivity = (
            -skew(imu_lever) @ angular_acceleration_sensitivity
            - skew(angular_acceleration) @ parameter_jacobian.cog_offset
        )
        omega_cross_lever = np.cross(omega, imu_lever)
        omega_cross_lever_sensitivity = (
            -skew(imu_lever) @ omega_sensitivity
            - skew(omega) @ parameter_jacobian.cog_offset
        )
        centripetal_sensitivity = (
            -skew(omega_cross_lever) @ omega_sensitivity
            + skew(omega) @ omega_cross_lever_sensitivity
        )
        specific_force_body = (
            wrench[:3] / parameters.mass
            + np.cross(angular_acceleration, imu_lever)
            + np.cross(omega, omega_cross_lever)
        )
        specific_force_body_sensitivity = (
            force_per_mass_sensitivity
            + alpha_cross_lever_sensitivity
            + centripetal_sensitivity
        )

        sensor_position = rigid.position + rotation @ pose_lever
        sensor_rotation = rotation @ problem.pose_body_to_sensor_rotation
        sensor_velocity = (
            rigid.linear_velocity
            + rotation @ np.cross(omega, velocity_lever)
        )
        arrays["sensor_position"][output_index] = sensor_position
        arrays["sensor_orientation_xyzw"][output_index] = (
            baseline.matrix_to_quaternion(sensor_rotation)
        )
        arrays["sensor_velocity_world"][output_index] = sensor_velocity
        arrays["angular_velocity_sensor"][output_index] = (
            problem.body_to_imu_rotation @ omega + problem.gyro_bias
        )
        arrays["specific_force_sensor"][output_index] = (
            problem.body_to_imu_rotation @ specific_force_body
            + problem.accelerometer_bias
        )
        arrays["cog_position"][output_index] = rigid.position
        arrays["cog_velocity_world"][output_index] = rigid.linear_velocity
        arrays["actuator_thrust"][output_index] = actuators.thrust
        arrays["actuator_gimbal"][output_index] = actuators.gimbal_angle

        output_jacobian["position"][output_index] = (
            rigid_sensitivity[:3]
            - rotation @ skew(pose_lever) @ rotation_right_sensitivity
            - rotation @ parameter_jacobian.cog_offset
        )
        observed_sensor_rotation = quaternion_to_matrix(
            problem.observations.sensor_orientation_xyzw[output_index]
        )
        orientation_error = so3_log(
            observed_sensor_rotation.T @ sensor_rotation
        )
        output_jacobian["orientation"][output_index] = (
            so3_right_jacobian_inverse(orientation_error)
            @ problem.pose_body_to_sensor_rotation.T
            @ rotation_right_sensitivity
        )
        velocity_cross_lever = np.cross(omega, velocity_lever)
        velocity_cross_lever_sensitivity = (
            -skew(velocity_lever) @ omega_sensitivity
            - skew(omega) @ parameter_jacobian.cog_offset
        )
        output_jacobian["velocity"][output_index] = (
            rigid_sensitivity[7:10]
            - rotation
            @ skew(velocity_cross_lever)
            @ rotation_right_sensitivity
            + rotation @ velocity_cross_lever_sensitivity
        )
        output_jacobian["angular_velocity"][output_index] = (
            problem.body_to_imu_rotation @ omega_sensitivity
        )
        output_jacobian["specific_force"][output_index] = (
            problem.body_to_imu_rotation
            @ specific_force_body_sensitivity
        )

    store(0)
    output_index = 1
    for step_index, command in enumerate(problem.commands):
        dt = problem.integration_step
        midpoint_actuators, midpoint_sensitivity = (
            _actuator_step_with_sensitivity(
                actuators,
                actuator_sensitivity,
                command,
                decoded,
                parameter_jacobian,
                0.5 * dt,
            )
        )
        rigid, rigid_sensitivity = _rigid_step_with_sensitivity(
            problem,
            decoded,
            parameter_jacobian,
            rigid,
            rigid_sensitivity,
            midpoint_actuators,
            midpoint_sensitivity,
            dt,
        )
        actuators, actuator_sensitivity = _actuator_step_with_sensitivity(
            midpoint_actuators,
            midpoint_sensitivity,
            command,
            decoded,
            parameter_jacobian,
            0.5 * dt,
        )
        if (step_index + 1) % problem.output_stride == 0:
            store(output_index)
            output_index += 1
    if output_index != output_count:
        raise RuntimeError("internal/output sensitivity grids disagree")
    simulation = baseline.Simulation(time=problem.output_time, **arrays)
    residual = _trajectory_residual(problem, simulation)
    count_scale = math.sqrt(problem.output_time.size)
    residual_jacobian = np.vstack(
        tuple(
            (
                output_jacobian[name]
                / problem.residual_scales[scale_name]
                / count_scale
            ).reshape(-1, SMOOTH_DIMENSION)
            for name, scale_name in (
                ("position", "position_m"),
                ("orientation", "orientation_rad"),
                ("velocity", "velocity_m_per_s"),
                ("angular_velocity", "angular_velocity_rad_per_s"),
                ("specific_force", "specific_force_m_per_s2"),
            )
        )
    )
    if np.any(~np.isfinite(residual_jacobian)):
        raise FloatingPointError("trajectory Jacobian is non-finite")
    return simulation, residual, residual_jacobian


def _join_smooth_coordinate(
    smooth: Sequence[float], fixed_delay: float
) -> np.ndarray:
    value = np.asarray(smooth, dtype=float)
    if value.shape != (SMOOTH_DIMENSION,) or np.any(~np.isfinite(value)):
        raise ValueError("smooth coordinate must be a finite 15-vector")
    return np.concatenate((value, np.asarray((float(fixed_delay),))))


class CandidateEvaluator:
    def __init__(
        self,
        *,
        flight: Any,
        parameterization: PhysicalSearchParameterization,
        pid_gate: PidGainGate,
        bounds: SearchBounds,
        sample_step: float,
        integration_step: float,
        prior_weight: float,
        pid_penalty_weight: float,
    ) -> None:
        self.flight = flight
        self.parameterization = parameterization
        self.pid_gate = pid_gate
        self.bounds = bounds
        self.sample_step = float(sample_step)
        self.integration_step = float(integration_step)
        self.prior_weight = float(prior_weight)
        self.pid_penalty_weight = float(pid_penalty_weight)
        self.geometry = baseline.GrapeGeometry.grape()
        self.prior_scales = np.asarray(
            (
                0.50,
                0.50,
                0.50,
                0.50,
                0.40,
                0.40,
                0.40,
                0.05,
                0.05,
                0.05,
                0.25,
                0.25,
                0.25,
                0.50,
                0.50,
            ),
            dtype=float,
        )

    def make_problem(self, delay: float) -> baseline.DirectShootingProblem:
        return baseline.DirectShootingProblem(
            flight=self.flight,
            sample_step=self.sample_step,
            integration_step=self.integration_step,
            command_delay=float(delay),
            prior_weight=self.prior_weight,
        )

    def physical_diagnostic(self, decoded: DecodedSearchPoint) -> dict[str, Any]:
        scale = max(float(np.trace(decoded.parameters.inertia)), 1.0e-15)
        normalized_margin = decoded.inertia_triangle_margin / scale
        return {
            "valid": bool(decoded.inertia_triangle_margin >= -1.0e-12 * scale),
            "inertia_principal_moments_kg_m2": decoded.inertia_principal_moments,
            "inertia_triangle_margin_kg_m2": decoded.inertia_triangle_margin,
            "normalized_inertia_triangle_margin": normalized_margin,
        }

    def pid_diagnostic(self, decoded: DecodedSearchPoint) -> dict[str, Any]:
        return self.pid_gate.evaluate(
            decoded.parameters,
            self.parameterization.nominal,
            self.geometry,
        )

    def simulate(
        self,
        problem: baseline.DirectShootingProblem,
        decoded: DecodedSearchPoint,
    ) -> baseline.Simulation:
        problem.actuator_parameters = decoded.actuator_parameters
        full_coordinate = problem.chart.encode(decoded.parameters)
        return problem.simulate_full_coordinates(full_coordinate)

    def evaluate(
        self,
        coordinate: Sequence[float],
        *,
        problem: Optional[baseline.DirectShootingProblem] = None,
    ) -> dict[str, Any]:
        value = np.asarray(coordinate, dtype=float)
        if not self.bounds.contains(value):
            return {"valid": False, "reason": "outside_search_bounds"}
        try:
            decoded = self.parameterization.decode(value)
            physical = self.physical_diagnostic(decoded)
            if not physical["valid"]:
                return {
                    "valid": False,
                    "reason": "inertia_triangle_inequality",
                    "physical": physical,
                }
            pid = self.pid_diagnostic(decoded)
            if not pid["valid"]:
                return {
                    "valid": False,
                    "reason": "pid_gain_out_of_range",
                    "physical": physical,
                    "pid": pid,
                }
            selected_problem = (
                self.make_problem(decoded.delay) if problem is None else problem
            )
            simulation = self.simulate(selected_problem, decoded)
            residual = _trajectory_residual(selected_problem, simulation)
            if np.any(~np.isfinite(residual)):
                raise FloatingPointError("trajectory residual is non-finite")
            return {
                "valid": True,
                "reason": "accepted",
                "decoded": decoded,
                "physical": physical,
                "pid": pid,
                "problem": selected_problem,
                "simulation": simulation,
                "trajectory_residual": residual,
                "trajectory_loss": _trajectory_loss(residual),
                "metrics": baseline._metrics(selected_problem, simulation),
            }
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as error:
            return {
                "valid": False,
                "reason": "numerical_failure",
                "detail": str(error),
            }

    def optimization_residual(
        self,
        problem: baseline.DirectShootingProblem,
        smooth_coordinate: Sequence[float],
        fixed_delay: float,
    ) -> np.ndarray:
        coordinate = _join_smooth_coordinate(smooth_coordinate, fixed_delay)
        residual_size = problem.output_time.size * 15 + SMOOTH_DIMENSION + 24 + 1
        try:
            decoded = self.parameterization.decode(coordinate)
            physical = self.physical_diagnostic(decoded)
            if not physical["valid"]:
                raise ValueError("local point violates inertia triangle inequality")
            pid = self.pid_diagnostic(decoded)
            if not pid["valid"]:
                raise ValueError("local point violates the PID gain gate")
            simulation = self.simulate(problem, decoded)
            trajectory = _trajectory_residual(problem, simulation)
            residual = np.concatenate(
                (
                    trajectory,
                    math.sqrt(self.prior_weight)
                    * np.asarray(smooth_coordinate, dtype=float)
                    / self.prior_scales,
                    self.pid_penalty_weight * pid["constraint_residual"],
                    np.zeros(1),
                )
            )
            if residual.shape != (residual_size,) or np.any(~np.isfinite(residual)):
                raise FloatingPointError("optimization residual is invalid")
            return residual
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            return np.full(
                residual_size,
                self.pid_penalty_weight
                * (100.0 + float(np.linalg.norm(smooth_coordinate))),
                dtype=float,
            )

    def optimization_residual_and_jacobian(
        self,
        problem: baseline.DirectShootingProblem,
        smooth_coordinate: Sequence[float],
        fixed_delay: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the local residual and its analytic forward-sensitivity Jacobian."""

        smooth = np.asarray(smooth_coordinate, dtype=float)
        coordinate = _join_smooth_coordinate(smooth, fixed_delay)
        residual_size = problem.output_time.size * 15 + SMOOTH_DIMENSION + 24 + 1
        try:
            decoded, parameter_jacobian = (
                self.parameterization.decode_with_jacobian(coordinate)
            )
            physical = self.physical_diagnostic(decoded)
            if not physical["valid"]:
                raise ValueError("local point violates inertia triangle inequality")
            pid = self.pid_diagnostic(decoded)
            if not pid["valid"]:
                raise ValueError("local point violates the PID gain gate")
            _simulation, trajectory, trajectory_jacobian = (
                _simulate_trajectory_with_jacobian(
                    problem,
                    decoded,
                    parameter_jacobian,
                )
            )
            residual = np.concatenate(
                (
                    trajectory,
                    math.sqrt(self.prior_weight)
                    * smooth
                    / self.prior_scales,
                    self.pid_penalty_weight * pid["constraint_residual"],
                    np.zeros(1),
                )
            )
            jacobian = np.vstack(
                (
                    trajectory_jacobian,
                    np.diag(
                        math.sqrt(self.prior_weight) / self.prior_scales
                    ),
                    np.zeros((24, SMOOTH_DIMENSION), dtype=float),
                    np.zeros((1, SMOOTH_DIMENSION), dtype=float),
                )
            )
            if (
                residual.shape != (residual_size,)
                or jacobian.shape != (residual_size, SMOOTH_DIMENSION)
                or np.any(~np.isfinite(residual))
                or np.any(~np.isfinite(jacobian))
            ):
                raise FloatingPointError(
                    "analytic optimization linearization is invalid"
                )
            return residual, jacobian
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            norm = float(np.linalg.norm(smooth))
            wall = self.pid_penalty_weight * (100.0 + norm)
            residual = np.full(residual_size, wall, dtype=float)
            gradient = (
                np.zeros(SMOOTH_DIMENSION, dtype=float)
                if norm <= np.finfo(float).eps
                else self.pid_penalty_weight * smooth / norm
            )
            return residual, np.tile(gradient, (residual_size, 1))


class _CachedLocalObjective:
    """Share one analytic rollout between SciPy's residual and Jacobian calls."""

    def __init__(
        self,
        evaluator: CandidateEvaluator,
        problem: baseline.DirectShootingProblem,
        fixed_delay: float,
    ) -> None:
        self.evaluator = evaluator
        self.problem = problem
        self.fixed_delay = float(fixed_delay)
        self.coordinate: Optional[np.ndarray] = None
        self.value: Optional[np.ndarray] = None
        self.jacobian_value: Optional[np.ndarray] = None
        self.linearization_count = 0

    def _evaluate(self, coordinate: Sequence[float]) -> None:
        value = np.asarray(coordinate, dtype=float)
        if (
            self.coordinate is not None
            and np.array_equal(value, self.coordinate)
        ):
            return
        residual, jacobian = self.evaluator.optimization_residual_and_jacobian(
            self.problem,
            value,
            self.fixed_delay,
        )
        self.coordinate = value.copy()
        self.value = residual
        self.jacobian_value = jacobian
        self.linearization_count += 1

    def residual(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.value is None:
            raise RuntimeError("local residual cache was not populated")
        return self.value

    def jacobian(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.jacobian_value is None:
            raise RuntimeError("local Jacobian cache was not populated")
        return self.jacobian_value


def _run_deterministic_baseline_incumbent(
    evaluator: CandidateEvaluator,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Run the unchanged nominal-start baseline path as a dominance incumbent."""

    print(
        "running deterministic baseline incumbent from nominal parameters",
        flush=True,
    )
    started = time.perf_counter()
    problem = evaluator.make_problem(arguments.command_delay)
    initial = np.zeros(baseline.ACTIVE_PARAMETER_DIMENSION, dtype=float)
    lower, upper = baseline.parameter_bounds()
    nominal_simulation = problem.simulate(initial)
    initializer_started = time.perf_counter()
    initializer_result = least_squares(
        lambda coordinates: problem.local_dynamics_residual(
            coordinates, nominal_simulation
        ),
        initial,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="soft_l1",
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
        max_nfev=arguments.baseline_initializer_max_nfev,
        verbose=0,
    )
    initializer_elapsed = time.perf_counter() - initializer_started
    refinement_started = time.perf_counter()
    refinement_result = least_squares(
        problem.residual,
        initializer_result.x,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="linear",
        ftol=1.0e-6,
        xtol=1.0e-6,
        gtol=1.0e-6,
        max_nfev=arguments.baseline_max_nfev,
        verbose=0,
    )
    refinement_elapsed = time.perf_counter() - refinement_started
    simulation = problem.simulate(refinement_result.x)
    trajectory_residual = _trajectory_residual(problem, simulation)
    full_coordinate = problem.full_coordinates(refinement_result.x)
    parameters = problem.chart.decode(full_coordinate)
    principal = np.linalg.eigvalsh(parameters.inertia)
    decoded = DecodedSearchPoint(
        parameters=parameters,
        actuator_parameters=problem.actuator_parameters,
        delay=float(arguments.command_delay),
        inertia_principal_moments=principal,
        inertia_triangle_margin=float(principal[0] + principal[1] - principal[2]),
    )
    coordinate = evaluator.parameterization.encode(
        parameters,
        problem.actuator_parameters,
        arguments.command_delay,
    )
    physical = evaluator.physical_diagnostic(decoded)
    pid = evaluator.pid_diagnostic(decoded)
    boundary = _boundary_diagnostic(
        coordinate,
        evaluator.bounds,
        arguments.boundary_proximity_fraction,
        physical=physical,
        pid=pid,
        pid_gate=evaluator.pid_gate,
        inertia_triangle_proximity_fraction=(
            arguments.inertia_triangle_proximity_fraction
        ),
    )
    record = {
        "source": "deterministic_baseline",
        "coordinate": coordinate,
        "active_coordinate": refinement_result.x,
        "trajectory_loss": _trajectory_loss(trajectory_residual),
        "metrics": baseline._metrics(problem, simulation),
        "decoded": decoded,
        "physical": physical,
        "pid": pid,
        "within_search_bounds": evaluator.bounds.contains(coordinate),
        "pid_valid": bool(pid["valid"]),
        "constraint_eligible": bool(
            evaluator.bounds.contains(coordinate)
            and physical["valid"]
            and pid["valid"]
        ),
        "boundary": boundary,
        "problem": problem,
        "simulation": simulation,
        "initializer": {
            "success": bool(initializer_result.success),
            "status": int(initializer_result.status),
            "message": str(initializer_result.message),
            "cost": float(initializer_result.cost),
            "optimality": float(initializer_result.optimality),
            "nfev": int(initializer_result.nfev),
            "njev": (
                None
                if initializer_result.njev is None
                else int(initializer_result.njev)
            ),
            "elapsed_seconds": initializer_elapsed,
        },
        "refinement": {
            "success": bool(refinement_result.success),
            "status": int(refinement_result.status),
            "message": str(refinement_result.message),
            "cost_with_prior": float(refinement_result.cost),
            "optimality": float(refinement_result.optimality),
            "nfev": int(refinement_result.nfev),
            "njev": (
                None
                if refinement_result.njev is None
                else int(refinement_result.njev)
            ),
            "elapsed_seconds": refinement_elapsed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "baseline incumbent: trajectory_loss={:.9g}, bounds={}, pid={}".format(
            record["trajectory_loss"],
            record["within_search_bounds"],
            record["pid_valid"],
        ),
        flush=True,
    )
    return record


def _baseline_incumbent_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strip runtime-only objects from the baseline dominance record."""

    return {
        key: value
        for key, value in record.items()
        if key not in {"decoded", "pid", "problem", "simulation"}
    }


def _screen_record(
    index: int,
    source: str,
    coordinate: np.ndarray,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": int(index),
        "source": str(source),
        "coordinate": np.asarray(coordinate, dtype=float),
        "accepted": bool(evaluation.get("valid", False)),
        "reason": str(evaluation.get("reason", "unknown")),
    }
    if "physical" in evaluation:
        result["physical"] = evaluation["physical"]
    if "pid" in evaluation:
        result["pid_group_scales"] = evaluation["pid"]["group_scales"]
        result["pid_gains"] = evaluation["pid"]["gains"]
    if evaluation.get("valid", False):
        result["trajectory_loss"] = evaluation["trajectory_loss"]
        result["metrics"] = evaluation["metrics"]
    if "detail" in evaluation:
        result["detail"] = evaluation["detail"]
    return result


def _boundary_diagnostic(
    coordinate: np.ndarray,
    bounds: SearchBounds,
    proximity_fraction: float,
    *,
    physical: Optional[Mapping[str, Any]] = None,
    pid: Optional[Mapping[str, Any]] = None,
    pid_gate: Optional[PidGainGate] = None,
    inertia_triangle_proximity_fraction: float = 1.0e-4,
) -> dict[str, Any]:
    normalized = bounds.normalized(coordinate)[:SMOOTH_DIMENSION]
    distance = np.minimum(normalized, 1.0 - normalized)
    hits = np.flatnonzero(distance <= proximity_fraction)
    coordinate_score = float(
        np.sum(
            np.maximum(proximity_fraction - distance, 0.0)
            / proximity_fraction
        )
    )
    inertia_margin = None
    inertia_near_boundary = False
    inertia_score = 0.0
    if physical is not None:
        inertia_margin = float(physical["normalized_inertia_triangle_margin"])
        inertia_near_boundary = bool(
            inertia_margin <= inertia_triangle_proximity_fraction
        )
        inertia_score = float(
            max(
                inertia_triangle_proximity_fraction - inertia_margin,
                0.0,
            )
            / inertia_triangle_proximity_fraction
        )
    pid_near_boundary_groups: list[str] = []
    pid_score = 0.0
    if pid is not None and pid_gate is not None and pid_gate.enabled:
        gains = np.asarray(pid["gains"], dtype=float)
        span = pid_gate.upper - pid_gate.lower
        for group_index, group_name in enumerate(PID_GROUPS):
            configured = span[group_index] > 0.0
            if not np.any(configured):
                continue
            lower_distance = (
                gains[group_index, configured]
                - pid_gate.lower[group_index, configured]
            ) / span[group_index, configured]
            upper_distance = (
                pid_gate.upper[group_index, configured]
                - gains[group_index, configured]
            ) / span[group_index, configured]
            group_distance = float(
                np.min(np.minimum(lower_distance, upper_distance))
            )
            if group_distance <= proximity_fraction:
                pid_near_boundary_groups.append(group_name)
                pid_score += max(
                    proximity_fraction - group_distance, 0.0
                ) / proximity_fraction
    coordinate_names = [
        SEARCH_PARAMETER_NAMES[int(index)] for index in hits
    ]
    physical_names = (
        ["inertia_triangle_inequality"] if inertia_near_boundary else []
    )
    all_names = coordinate_names + physical_names + [
        "pid_gain_{}".format(group) for group in pid_near_boundary_groups
    ]
    return {
        "proximity_fraction": proximity_fraction,
        "coordinate_near_boundary_names": coordinate_names,
        "inertia_triangle_proximity_fraction": (
            inertia_triangle_proximity_fraction
        ),
        "normalized_inertia_triangle_margin": inertia_margin,
        "inertia_triangle_near_boundary": inertia_near_boundary,
        "pid_near_boundary_groups": pid_near_boundary_groups,
        "near_boundary_names": all_names,
        "near_boundary_count": len(all_names),
        "proximity_score": coordinate_score + inertia_score + pid_score,
        "delay_excluded_because_fixed_within_local_solve": True,
    }


def _local_record(
    *,
    rank: int,
    seed_record: Mapping[str, Any],
    result: Any,
    coordinate: np.ndarray,
    evaluation: Mapping[str, Any],
    boundary: Mapping[str, Any],
    elapsed: float,
    analytic_linearization_count: int,
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "seed_index": int(seed_record["index"]),
        "seed_source": str(seed_record["source"]),
        "seed_coordinate": np.asarray(seed_record["coordinate"], dtype=float),
        "seed_trajectory_loss": float(seed_record["trajectory_loss"]),
        "fixed_delay_seconds": float(coordinate[DELAY_INDEX]),
        "coordinate": coordinate,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_cost_with_prior_and_constraints": float(result.cost),
        "optimizer_optimality": float(result.optimality),
        "optimizer_nfev": int(result.nfev),
        "optimizer_njev": None if result.njev is None else int(result.njev),
        "analytic_linearization_count": int(analytic_linearization_count),
        "optimizer_active_mask": np.asarray(result.active_mask, dtype=int),
        "elapsed_seconds": float(elapsed),
        "valid": bool(evaluation.get("valid", False)),
        "validation_reason": str(evaluation.get("reason", "unknown")),
        "trajectory_loss": (
            None
            if not evaluation.get("valid", False)
            else float(evaluation["trajectory_loss"])
        ),
        "metrics": evaluation.get("metrics"),
        "physical": evaluation.get("physical"),
        "pid_group_scales": (
            None
            if "pid" not in evaluation
            else evaluation["pid"]["group_scales"]
        ),
        "pid_gains": (
            None if "pid" not in evaluation else evaluation["pid"]["gains"]
        ),
        "boundary": dict(boundary),
    }


def choose_final_local_record(
    records: Sequence[Mapping[str, Any]],
    *,
    loss_tolerance_fraction: float,
) -> tuple[Optional[Mapping[str, Any]], dict[str, Any]]:
    if not np.isfinite(loss_tolerance_fraction) or loss_tolerance_fraction < 0.0:
        raise ValueError("selection loss tolerance must be non-negative")
    eligible = tuple(
        record
        for record in records
        if record.get("optimizer_success", False)
        and record.get("valid", False)
        and record.get("trajectory_loss") is not None
        and np.isfinite(float(record["trajectory_loss"]))
    )
    if not eligible:
        return None, {
            "reason": "no_converged_physically_valid_pid_valid_candidate",
            "eligible_count": 0,
        }
    absolute_best = min(eligible, key=lambda item: float(item["trajectory_loss"]))
    minimum_loss = float(absolute_best["trajectory_loss"])
    tolerance = loss_tolerance_fraction * max(
        abs(minimum_loss), np.finfo(float).eps
    )
    near_optimal = tuple(
        item
        for item in eligible
        if float(item["trajectory_loss"]) <= minimum_loss + tolerance
    )
    selected = min(
        near_optimal,
        key=lambda item: (
            int(item["boundary"]["near_boundary_count"]),
            float(item["boundary"]["proximity_score"]),
            float(item["trajectory_loss"]),
            int(item["rank"]),
        ),
    )
    return selected, {
        "reason": "fewest_boundary_contacts_within_near_minimum_loss_set",
        "eligible_count": len(eligible),
        "near_minimum_count": len(near_optimal),
        "minimum_trajectory_loss": minimum_loss,
        "loss_tolerance_fraction": loss_tolerance_fraction,
        "absolute_best_rank": int(absolute_best["rank"]),
        "selected_rank": int(selected["rank"]),
    }


def choose_with_baseline_incumbent(
    baseline_record: Mapping[str, Any],
    feasible_record: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], bool]:
    """Enforce that the final trajectory loss never exceeds the baseline."""

    baseline_loss = float(baseline_record["trajectory_loss"])
    if not np.isfinite(baseline_loss):
        raise ValueError("baseline incumbent loss must be finite")
    if feasible_record is None:
        return baseline_record, True
    feasible_loss = float(feasible_record["trajectory_loss"])
    if not np.isfinite(feasible_loss):
        raise ValueError("feasible candidate loss must be finite")
    if baseline_loss <= feasible_loss:
        return baseline_record, True
    return feasible_record, False


def _search_bounds(arguments: argparse.Namespace) -> SearchBounds:
    mass_min, mass_max = arguments.mass_scale_bounds
    diagonal_min, diagonal_max = arguments.inertia_cholesky_diagonal_scale_bounds
    thrust_min, thrust_max = arguments.thrust_time_constant_scale_bounds
    gimbal_min, gimbal_max = arguments.gimbal_time_constant_scale_bounds
    delay_min, delay_max = arguments.delay_bounds
    positive_pairs = (
        (mass_min, mass_max),
        (diagonal_min, diagonal_max),
        (thrust_min, thrust_max),
        (gimbal_min, gimbal_max),
    )
    if any(low <= 0.0 or high <= low for low, high in positive_pairs):
        raise ValueError("positive scale bounds must be increasing and positive")
    if (
        arguments.inertia_cholesky_offdiagonal_bound <= 0.0
        or arguments.cog_bound <= 0.0
        or arguments.force_effectiveness_contrast_bound <= 0.0
        or delay_min < 0.0
        or delay_max <= delay_min
    ):
        raise ValueError("Cholesky, CoG, effectiveness, or delay bounds are invalid")
    off = float(arguments.inertia_cholesky_offdiagonal_bound)
    cog = float(arguments.cog_bound)
    effectiveness = float(arguments.force_effectiveness_contrast_bound)
    lower = np.asarray(
        (
            math.log(mass_min),
            math.log(diagonal_min),
            math.log(diagonal_min),
            math.log(diagonal_min),
            -off,
            -off,
            -off,
            -cog,
            -cog,
            -cog,
            -effectiveness,
            -effectiveness,
            -effectiveness,
            math.log(thrust_min),
            math.log(gimbal_min),
            delay_min,
        )
    )
    upper = np.asarray(
        (
            math.log(mass_max),
            math.log(diagonal_max),
            math.log(diagonal_max),
            math.log(diagonal_max),
            off,
            off,
            off,
            cog,
            cog,
            cog,
            effectiveness,
            effectiveness,
            effectiveness,
            math.log(thrust_max),
            math.log(gimbal_max),
            delay_max,
        )
    )
    return SearchBounds(lower, upper)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sobol-screen physically valid deterministic recorded-control "
            "initial values, then refine diverse low-loss seeds locally."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--command-delay", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument(
        "--baseline-initializer-max-nfev",
        type=int,
        default=100,
        help="function-evaluation limit for the nominal baseline initializer",
    )
    parser.add_argument(
        "--baseline-max-nfev",
        type=int,
        default=25,
        help="function-evaluation limit for the exact baseline incumbent",
    )
    parser.add_argument(
        "--sobol-power",
        type=int,
        default=9,
        help="generate 2^POWER scrambled Sobol points in addition to current",
    )
    parser.add_argument("--sobol-seed", type=int, default=0)
    parser.add_argument(
        "--local-start-count",
        type=int,
        default=16,
        help="maximum number of diverse low-loss local least-squares starts",
    )
    parser.add_argument(
        "--minimum-seed-distance",
        type=float,
        default=0.20,
        help="minimum Euclidean distance in normalized 16-D search coordinates",
    )
    parser.add_argument(
        "--pid-gain-min-scale",
        type=float,
        default=None,
        help=(
            "optional lower allowed gain relative to each recorded current "
            "PID gain; specify together with --pid-gain-max-scale"
        ),
    )
    parser.add_argument(
        "--pid-gain-max-scale",
        type=float,
        default=None,
        help=(
            "optional upper allowed gain relative to each recorded current "
            "PID gain; specify together with --pid-gain-min-scale"
        ),
    )
    parser.add_argument(
        "--pid-constraint-penalty",
        type=float,
        default=100.0,
        help="hard residual-wall scale for invalid local trial points",
    )
    parser.add_argument(
        "--mass-scale-bounds",
        type=float,
        nargs=2,
        default=(0.85, 1.15),
        metavar=("MIN", "MAX"),
        help="mass bounds as positive scales of the current mass",
    )
    parser.add_argument(
        "--inertia-cholesky-diagonal-scale-bounds",
        type=float,
        nargs=2,
        default=(0.90, 1.10),
        metavar=("MIN", "MAX"),
        help="positive diagonal Cholesky bounds as current-value scales",
    )
    parser.add_argument(
        "--inertia-cholesky-offdiagonal-bound",
        type=float,
        default=0.05,
        help="symmetric bound on normalized lower-Cholesky offsets",
    )
    parser.add_argument(
        "--cog-bound",
        type=float,
        default=0.02,
        help="symmetric per-axis CoG offset bound in metres",
    )
    parser.add_argument(
        "--force-effectiveness-contrast-bound",
        type=float,
        default=0.10,
        help="symmetric bound on each zero-sum log-effectiveness contrast",
    )
    parser.add_argument(
        "--thrust-time-constant-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
        help="thrust time-constant bounds as current-value scales",
    )
    parser.add_argument(
        "--gimbal-time-constant-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
        help="gimbal time-constant bounds as current-value scales",
    )
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.08),
        metavar=("MIN", "MAX"),
        help="command-delay bounds in seconds",
    )
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=None,
        help=(
            "stop remaining local solves after a converged valid candidate "
            "reaches this trajectory loss"
        ),
    )
    parser.add_argument(
        "--early-stop-max-boundary-hits",
        type=int,
        default=0,
        help="maximum near-boundary constraint count for early acceptance",
    )
    parser.add_argument(
        "--boundary-proximity-fraction",
        type=float,
        default=0.02,
        help="normalized box/PID distance counted as near a boundary",
    )
    parser.add_argument(
        "--inertia-triangle-proximity-fraction",
        type=float,
        default=1.0e-4,
        help="trace-normalized inertia triangle margin counted as near-boundary",
    )
    parser.add_argument(
        "--selection-loss-tolerance-fraction",
        type=float,
        default=0.01,
        help="loss window in which fewer boundary contacts break the tie",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    finite = (
        arguments.start,
        arguments.end,
        arguments.sample_step,
        arguments.integration_step,
        arguments.command_delay,
        arguments.prior_weight,
        arguments.minimum_seed_distance,
        arguments.pid_constraint_penalty,
        arguments.boundary_proximity_fraction,
        arguments.inertia_triangle_proximity_fraction,
        arguments.selection_loss_tolerance_fraction,
    )
    if (
        any(not np.isfinite(value) for value in finite)
        or arguments.start >= arguments.end
        or arguments.sample_step <= 0.0
        or arguments.integration_step <= 0.0
        or arguments.command_delay < 0.0
        or arguments.prior_weight < 0.0
        or arguments.max_nfev < 1
        or arguments.baseline_initializer_max_nfev < 1
        or arguments.baseline_max_nfev < 1
        or arguments.sobol_power < 0
        or arguments.sobol_power > 20
        or arguments.local_start_count < 1
        or arguments.minimum_seed_distance < 0.0
        or arguments.pid_constraint_penalty <= 0.0
        or arguments.early_stop_max_boundary_hits < 0
        or not 0.0 < arguments.boundary_proximity_fraction < 0.5
        or arguments.inertia_triangle_proximity_fraction <= 0.0
        or arguments.selection_loss_tolerance_fraction < 0.0
        or (
            arguments.early_stop_loss is not None
            and (
                not np.isfinite(arguments.early_stop_loss)
                or arguments.early_stop_loss < 0.0
            )
        )
    ):
        raise SystemExit("Sobol deterministic settings are invalid")


def _physical_payload(decoded: DecodedSearchPoint) -> dict[str, Any]:
    result = baseline._physical_parameters(decoded.parameters)
    result.update(
        {
            "thrust_time_constant_seconds": (
                decoded.actuator_parameters.thrust_time_constant
            ),
            "gimbal_time_constant_seconds": (
                decoded.actuator_parameters.gimbal_time_constant
            ),
            "command_delay_seconds": decoded.delay,
        }
    )
    return result


def run(arguments: argparse.Namespace) -> int:
    _validate_arguments(arguments)
    try:
        bounds = _search_bounds(arguments)
        pid_gate_scale = (
            None
            if arguments.pid_gain_min_scale is None
            else float(arguments.pid_gain_min_scale),
            None
            if arguments.pid_gain_max_scale is None
            else float(arguments.pid_gain_max_scale),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bag = arguments.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit("bag does not exist: {}".format(bag))
    started = time.perf_counter()
    print(
        "loading {} [{:.3f}, {:.3f}] s".format(
            bag, arguments.start, arguments.end
        ),
        flush=True,
    )
    flight = baseline.load_flight_data(
        str(bag),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    nominal_parameters = VehicleParameters.nominal()
    parameterization = PhysicalSearchParameterization(nominal_parameters)
    current_coordinate = parameterization.current_coordinate(arguments.command_delay)
    if not bounds.contains(current_coordinate):
        raise SystemExit("current parameter coordinate is outside the search bounds")
    try:
        if pid_gate_scale == (None, None):
            pid_gate = PidGainGate.disabled(
                flight.controller_snapshot.gains
            )
        elif None not in pid_gate_scale:
            pid_gate = PidGainGate.from_scale_band(
                flight.controller_snapshot.gains,
                pid_gate_scale[0],
                pid_gate_scale[1],
            )
        else:
            raise ValueError(
                "PID gain minimum and maximum scales must be specified together"
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    evaluator = CandidateEvaluator(
        flight=flight,
        parameterization=parameterization,
        pid_gate=pid_gate,
        bounds=bounds,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        prior_weight=arguments.prior_weight,
        pid_penalty_weight=arguments.pid_constraint_penalty,
    )
    baseline_incumbent = _run_deterministic_baseline_incumbent(
        evaluator, arguments
    )
    coordinates, sources = generate_sobol_coordinates(
        bounds,
        power=arguments.sobol_power,
        seed=arguments.sobol_seed,
        current_coordinate=current_coordinate,
    )
    losses = np.full(coordinates.shape[0], np.inf, dtype=float)
    screen_records = []
    accepted_indices = []
    print(
        "screening {} current/Sobol candidates".format(coordinates.shape[0]),
        flush=True,
    )
    for index, (coordinate, source) in enumerate(zip(coordinates, sources)):
        evaluation = evaluator.evaluate(coordinate)
        record = _screen_record(index, source, coordinate, evaluation)
        screen_records.append(record)
        if evaluation.get("valid", False):
            losses[index] = float(evaluation["trajectory_loss"])
            accepted_indices.append(index)
        if (
            index == 0
            or (index + 1) % 16 == 0
            or index + 1 == coordinates.shape[0]
        ):
            print(
                "Sobol screen {}/{}: accepted={}, best_loss={}".format(
                    index + 1,
                    coordinates.shape[0],
                    len(accepted_indices),
                    (
                        "unavailable"
                        if not accepted_indices
                        else "{:.9g}".format(
                            min(losses[value] for value in accepted_indices)
                        )
                    ),
                ),
                flush=True,
            )
    selected_seed_indices = select_diverse_candidate_indices(
        coordinates,
        losses,
        accepted_indices,
        bounds,
        count=arguments.local_start_count,
        minimum_distance=arguments.minimum_seed_distance,
        required_indices=(0,),
    )
    selected_seed_set = set(selected_seed_indices)
    for record in screen_records:
        record["selected_as_local_seed"] = record["index"] in selected_seed_set
    local_records = []
    early_stop = None
    smooth_lower = bounds.lower[:SMOOTH_DIMENSION]
    smooth_upper = bounds.upper[:SMOOTH_DIMENSION]
    for rank, seed_index in enumerate(selected_seed_indices, start=1):
        seed_record = screen_records[seed_index]
        seed_coordinate = np.asarray(seed_record["coordinate"], dtype=float)
        fixed_delay = float(seed_coordinate[DELAY_INDEX])
        problem = evaluator.make_problem(fixed_delay)
        objective = _CachedLocalObjective(evaluator, problem, fixed_delay)
        local_started = time.perf_counter()
        print(
            "local solve {}/{} from {}: seed_loss={:.9g}, delay={:.6g}".format(
                rank,
                len(selected_seed_indices),
                seed_record["source"],
                seed_record["trajectory_loss"],
                fixed_delay,
            ),
            flush=True,
        )
        result = least_squares(
            objective.residual,
            seed_coordinate[:SMOOTH_DIMENSION],
            bounds=(smooth_lower, smooth_upper),
            method="trf",
            jac=objective.jacobian,
            x_scale="jac",
            loss="linear",
            ftol=1.0e-6,
            xtol=1.0e-6,
            gtol=1.0e-6,
            max_nfev=arguments.max_nfev,
            verbose=0,
        )
        coordinate = _join_smooth_coordinate(result.x, fixed_delay)
        evaluation = evaluator.evaluate(coordinate, problem=problem)
        boundary = _boundary_diagnostic(
            coordinate,
            bounds,
            arguments.boundary_proximity_fraction,
            physical=evaluation.get("physical"),
            pid=evaluation.get("pid"),
            pid_gate=pid_gate,
            inertia_triangle_proximity_fraction=(
                arguments.inertia_triangle_proximity_fraction
            ),
        )
        local_record = _local_record(
            rank=rank,
            seed_record=seed_record,
            result=result,
            coordinate=coordinate,
            evaluation=evaluation,
            boundary=boundary,
            elapsed=time.perf_counter() - local_started,
            analytic_linearization_count=objective.linearization_count,
        )
        local_records.append(local_record)
        print(
            "local solve {}: converged={}, valid={}, loss={}, boundaries={}".format(
                rank,
                local_record["optimizer_success"],
                local_record["valid"],
                (
                    "unavailable"
                    if local_record["trajectory_loss"] is None
                    else "{:.9g}".format(local_record["trajectory_loss"])
                ),
                local_record["boundary"]["near_boundary_count"],
            ),
            flush=True,
        )
        if (
            arguments.early_stop_loss is not None
            and local_record["optimizer_success"]
            and local_record["valid"]
            and local_record["trajectory_loss"] <= arguments.early_stop_loss
            and local_record["boundary"]["near_boundary_count"]
            <= arguments.early_stop_max_boundary_hits
        ):
            early_stop = {
                "triggered": True,
                "rank": rank,
                "trajectory_loss": local_record["trajectory_loss"],
                "threshold": arguments.early_stop_loss,
                "maximum_boundary_hits": arguments.early_stop_max_boundary_hits,
            }
            break
    if early_stop is None:
        early_stop = {
            "triggered": False,
            "threshold": arguments.early_stop_loss,
            "maximum_boundary_hits": arguments.early_stop_max_boundary_hits,
        }
    feasible_selected_record, selection_diagnostic = choose_final_local_record(
        local_records,
        loss_tolerance_fraction=arguments.selection_loss_tolerance_fraction,
    )
    feasible_loss = (
        None
        if feasible_selected_record is None
        else float(feasible_selected_record["trajectory_loss"])
    )
    baseline_loss = float(baseline_incumbent["trajectory_loss"])
    selected_record, baseline_selected = choose_with_baseline_incumbent(
        baseline_incumbent,
        feasible_selected_record,
    )
    selection_diagnostic.update(
        {
            "best_constraint_eligible_rank": (
                None
                if feasible_selected_record is None
                else int(feasible_selected_record["rank"])
            ),
            "best_constraint_eligible_trajectory_loss": feasible_loss,
            "deterministic_baseline_trajectory_loss": baseline_loss,
            "dominance_guard": (
                "select the deterministic baseline incumbent unless a "
                "constraint-eligible Sobol local result has no larger "
                "trajectory loss"
            ),
            "selected_source": (
                "deterministic_baseline"
                if baseline_selected
                else "sobol_local"
            ),
            "selected_rank": (
                None
                if baseline_selected
                else int(feasible_selected_record["rank"])
            ),
            "selected_trajectory_loss": float(selected_record["trajectory_loss"]),
        }
    )
    output = arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "trajectory.pdf"
    rejection_counts: dict[str, int] = {}
    for record in screen_records:
        if not record["accepted"]:
            reason = record["reason"]
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "path": str(bag),
            "sha256": baseline._sha256(bag),
            "requested_interval_seconds": [arguments.start, arguments.end],
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
        },
        "method": {
            "trajectory": (
                "single-shooting recorded-control open-loop without observation resets"
            ),
            "initial_design": "scrambled Sobol plus exact current coordinate",
            "baseline_incumbent": (
                "unchanged nominal-start deterministic baseline initializer "
                "and refinement retained as an unconditional loss incumbent"
            ),
            "inertia_parameterization": (
                "positive-diagonal lower Cholesky factor with normalized "
                "off-diagonal offsets"
            ),
            "positive_parameterization": (
                "log mass, zero-sum log force-effectiveness contrasts, and "
                "log actuator time constants"
            ),
            "delay": (
                "bounded Sobol coordinate fixed within each local solve "
                "because commands use causal ZOH"
            ),
            "local_optimizer": "scipy.optimize.least_squares",
            "local_jacobian": (
                "analytic forward sensitivities through the actuator model, "
                "RK4 rigid-body rollout, and observation residuals"
            ),
            "trajectory_loss": (
                "0.5 times the squared norm of the baseline-scaled position, "
                "orientation, velocity, gyro, and specific-force residuals; "
                "parameter prior and constraint wall excluded"
            ),
            "observation_resets": False,
            "latent_states": False,
            "q": None,
            "residual_wrench": None,
        },
        "search": {
            "parameter_names": SEARCH_PARAMETER_NAMES,
            "bounds": bounds.as_mapping(),
            "current_coordinate": current_coordinate,
            "sobol_power": arguments.sobol_power,
            "sobol_point_count": 2 ** arguments.sobol_power,
            "current_point_added": True,
            "seed": arguments.sobol_seed,
            "minimum_normalized_seed_distance": arguments.minimum_seed_distance,
            "requested_local_start_count": arguments.local_start_count,
            "selected_local_start_count": len(selected_seed_indices),
        },
        "pid_gate": {
            "enabled": pid_gate.enabled,
            "derivation": (
                "recorded current gains times the production group "
                "compensation scales from physical acceleration response"
            ),
            "current_gains": pid_gate.current,
            "lower_gains": pid_gate.lower,
            "upper_gains": pid_gate.upper,
            "lower_scale": pid_gate_scale[0],
            "upper_scale": pid_gate_scale[1],
            "hard_constraint_wall_weight": arguments.pid_constraint_penalty,
        },
        "screening": {
            "evaluated_count": len(screen_records),
            "accepted_count": len(accepted_indices),
            "rejected_count": len(screen_records) - len(accepted_indices),
            "rejection_counts": rejection_counts,
            "records": screen_records,
        },
        "local_optimization": {
            "maximum_function_evaluations_per_start": arguments.max_nfev,
            "prior_weight": arguments.prior_weight,
            "prior_scales": evaluator.prior_scales,
            "delay_fixed_within_each_solve": True,
            "jacobian": "analytic_forward_sensitivity",
            "completed_count": len(local_records),
            "early_stop": early_stop,
            "records": local_records,
        },
        "deterministic_baseline_incumbent": _baseline_incumbent_payload(
            baseline_incumbent
        ),
        "selection": selection_diagnostic,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {"result_json": "result.json", "trajectory_pdf": None},
    }
    result_code = 2
    if selected_record is not None:
        selected_coordinate = np.asarray(selected_record["coordinate"], dtype=float)
        current_evaluation = evaluator.evaluate(current_coordinate)
        if not current_evaluation.get("valid", False):
            raise RuntimeError("current nominal candidate failed deterministic replay")
        if baseline_selected:
            selected_evaluation = {
                "decoded": baseline_incumbent["decoded"],
                "pid": baseline_incumbent["pid"],
                "problem": baseline_incumbent["problem"],
                "simulation": baseline_incumbent["simulation"],
                "metrics": baseline_incumbent["metrics"],
            }
            selected_boundary = baseline_incumbent["boundary"]
            selected_within_bounds = bool(
                baseline_incumbent["within_search_bounds"]
            )
            selected_pid_valid = bool(baseline_incumbent["pid_valid"])
            selected_constraint_eligible = bool(
                baseline_incumbent["constraint_eligible"]
            )
        else:
            selected_evaluation = evaluator.evaluate(selected_coordinate)
            if not selected_evaluation.get("valid", False):
                raise RuntimeError(
                    "selected local candidate failed deterministic replay"
                )
            selected_boundary = selected_record["boundary"]
            selected_within_bounds = True
            selected_pid_valid = True
            selected_constraint_eligible = True
        selected_decoded = selected_evaluation["decoded"]
        payload["selection"].update(
            {
                "selected_coordinate": selected_coordinate,
                "selected_parameters": _physical_payload(selected_decoded),
                "selected_pid_group_scales": selected_evaluation["pid"][
                    "group_scales"
                ],
                "selected_pid_gains": selected_evaluation["pid"]["gains"],
                "selected_boundary": selected_boundary,
                "selected_within_search_bounds": selected_within_bounds,
                "selected_pid_valid": selected_pid_valid,
                "selected_constraint_eligible": selected_constraint_eligible,
            }
        )
        payload["recorded_control_open_loop_metrics"] = {
            "current": current_evaluation["metrics"],
            "selected": selected_evaluation["metrics"],
        }
        payload["outputs"]["trajectory_pdf"] = "trajectory.pdf"
        baseline._write_pdf(
            pdf_path,
            selected_evaluation["problem"],
            current_evaluation["simulation"],
            selected_evaluation["simulation"],
            current_evaluation["metrics"],
            selected_evaluation["metrics"],
        )
        result_code = 0
    elif pdf_path.is_file():
        # A failed rerun must not leave a previous trajectory looking current.
        pdf_path.unlink()
    baseline._write_json(output / "result.json", payload)
    print("wrote {}".format(output / "result.json"), flush=True)
    if selected_record is None:
        print(
            "no converged physically valid PID-valid local candidate was selected",
            flush=True,
        )
    else:
        if baseline_selected:
            print(
                "selected deterministic baseline incumbent with trajectory "
                "loss {:.9g}".format(selected_record["trajectory_loss"]),
                flush=True,
            )
        else:
            print(
                "selected local rank {} with trajectory loss {:.9g}".format(
                    selected_record["rank"], selected_record["trajectory_loss"]
                ),
                flush=True,
            )
        print("wrote {}".format(output / "trajectory.pdf"), flush=True)
    return result_code


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
