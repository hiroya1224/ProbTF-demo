#!/usr/bin/env python3
"""Strict multiple-shooting identification with an SE(3)-only data loss.

The recorded rotor and gimbal commands are treated as known inputs.  One set
of physical parameters is shared by every shooting segment.  Internal segment
states are numerical auxiliary variables and are tied together by an
augmented-Lagrangian continuity constraint.  The observation residual is only

    Log_SE(3)(T_observed^{-1} T_simulated),

with the translational part pushed to the Lie algebra through the inverse
SO(3) left Jacobian.  Velocity, gyro, and accelerometer samples are not part
of the observation loss.  They are used only to initialize shooting nodes.

The thirteen smooth physical coordinates are mass, a six-dimensional
Cholesky chart of the mass-distribution second moment, the three-dimensional
CoG offset, and three relative rotor-force-effectiveness contrasts.  The
inertia is reconstructed as ``J = trace(Sigma) I - Sigma`` with
``Sigma = L L.T``.  This guarantees both positive principal moments and their
triangle inequalities throughout optimization.  Recorded-command lag is
fixed outside the smooth solve because causal zero-order-hold lookup is not
smooth in lag.  The pose loss uses the nominal inertia radius:
``||rho||^2 + phi.T @ (J0 / m0) @ phi``.
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
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares

from . import deterministic_continuation_estimator as continuation
from . import deterministic_estimator as baseline
from . import deterministic_sobol_estimator as analytic
from grape_param_estim.dynamics import (
    FullSixDofPlant,
    actuator_wrench_with_jacobian,
    advance_actuators,
    advance_actuators_with_jacobian,
)
from grape_param_estim.geometry import (
    matrix_to_euler_xyz,
    matrix_to_quaternion,
    normalise_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    skew,
    so3_exp,
    so3_left_jacobian_inverse,
    so3_log,
    so3_right_jacobian,
    so3_right_jacobian_inverse,
)
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.system import (
    GRAVITY,
    ActuatorParameters,
    ActuatorState,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/minimal-deterministic-multiple-shooting/v6"
OUTPUT_SUBDIRECTORY = "deterministic_multiple_shooting"
PHYSICAL_DIMENSION = 13
NODE_DIMENSION = 20
PHYSICAL_PARAMETER_NAMES = (
    "log_mass_scale",
    "log_second_moment_cholesky_xx_scale",
    "log_second_moment_cholesky_yy_scale",
    "log_second_moment_cholesky_zz_scale",
    "normalized_second_moment_cholesky_yx_offset",
    "normalized_second_moment_cholesky_zx_offset",
    "normalized_second_moment_cholesky_zy_offset",
    "cog_offset_x_m",
    "cog_offset_y_m",
    "cog_offset_z_m",
    "force_effectiveness_contrast_1",
    "force_effectiveness_contrast_2",
    "force_effectiveness_contrast_3",
)
_CHOLESKY_OFF_DIAGONALS = ((1, 0), (2, 0), (2, 1))
BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS = np.asarray(
    (
        1.5,
        1.5,
        1.5,
        1.5,
        2.0,
        2.0,
        2.0,
        0.25,
        0.25,
        0.25,
        1.5,
        1.5,
        1.5,
    ),
    dtype=float,
)
NODE_PARAMETER_NAMES = (
    "cog_position_correction_x_m",
    "cog_position_correction_y_m",
    "cog_position_correction_z_m",
    "body_orientation_correction_x_rad",
    "body_orientation_correction_y_rad",
    "body_orientation_correction_z_rad",
    "cog_velocity_correction_x_m_per_s",
    "cog_velocity_correction_y_m_per_s",
    "cog_velocity_correction_z_m_per_s",
    "body_angular_velocity_correction_x_rad_per_s",
    "body_angular_velocity_correction_y_rad_per_s",
    "body_angular_velocity_correction_z_rad_per_s",
    "rotor_thrust_correction_1",
    "rotor_thrust_correction_2",
    "rotor_thrust_correction_3",
    "rotor_thrust_correction_4",
    "gimbal_angle_correction_1_rad",
    "gimbal_angle_correction_2_rad",
    "gimbal_angle_correction_3_rad",
    "gimbal_angle_correction_4_rad",
)


@dataclass(frozen=True)
class NodeReference:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    thrust: np.ndarray
    gimbal: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "position": (3,),
            "rotation": (3, 3),
            "linear_velocity": (3,),
            "angular_velocity": (3,),
            "thrust": (4,),
            "gimbal": (4,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError("{} has an invalid shape or value".format(name))
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class SegmentEvaluation:
    output_indices: np.ndarray
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    sensor_velocity_world: np.ndarray
    angular_velocity_sensor: np.ndarray
    specific_force_sensor: np.ndarray
    pose_residual: np.ndarray
    pose_jacobian: np.ndarray
    end_rigid: RigidBodyState
    end_actuator: ActuatorState
    end_rigid_sensitivity: np.ndarray
    end_actuator_sensitivity: np.ndarray


@dataclass(frozen=True)
class ProblemEvaluation:
    data_residual: np.ndarray
    data_jacobian: np.ndarray
    continuity_residual: np.ndarray
    continuity_jacobian: np.ndarray
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    sensor_velocity_world: np.ndarray
    angular_velocity_sensor: np.ndarray
    specific_force_sensor: np.ndarray
    decoded: analytic.DecodedSearchPoint


@dataclass(frozen=True)
class FixedDelaySolution:
    delay: float
    coordinate: np.ndarray
    optimizer_history: tuple[dict[str, Any], ...]
    evaluation: ProblemEvaluation
    full_rollout_position: np.ndarray
    full_rollout_orientation_xyzw: np.ndarray
    full_rollout_residual: np.ndarray
    elapsed_seconds: float


class FullyPhysicalInertiaParameterization:
    """Local physical chart whose inertia is fully physically consistent.

    The positive-definite second moment ``Sigma = L L.T`` is represented by
    six Cholesky coordinates centered at the nominal vehicle.  Mapping it to
    ``J = trace(Sigma) I - Sigma`` makes every represented inertia positive
    definite and gives it strict principal-moment triangle inequalities.
    """

    def __init__(self, nominal: VehicleParameters) -> None:
        if not isinstance(nominal, VehicleParameters):
            raise TypeError("nominal must be VehicleParameters")
        second_moment = (
            0.5 * float(np.trace(nominal.inertia)) * np.eye(3)
            - nominal.inertia
        )
        second_moment = 0.5 * (second_moment + second_moment.T)
        if np.any(np.linalg.eigvalsh(second_moment) <= 0.0):
            raise ValueError(
                "nominal inertia must satisfy strict triangle inequalities"
            )
        self.nominal = nominal
        self.nominal_second_moment = second_moment
        self.nominal_cholesky = np.linalg.cholesky(second_moment)

    def _cholesky(self, coordinate: np.ndarray) -> np.ndarray:
        cholesky = self.nominal_cholesky.copy()
        cholesky[(0, 1, 2), (0, 1, 2)] *= np.exp(coordinate[1:4])
        for local_index, (row, column) in enumerate(
            _CHOLESKY_OFF_DIAGONALS
        ):
            scale = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            cholesky[row, column] += coordinate[4 + local_index] * scale
        return cholesky

    def decode(
        self, coordinate: Sequence[float]
    ) -> analytic.DecodedSearchPoint:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (analytic.SEARCH_DIMENSION,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError("search coordinate must be a finite 16-vector")
        cholesky = self._cholesky(value)
        second_moment = cholesky @ cholesky.T
        inertia = np.trace(second_moment) * np.eye(3) - second_moment
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
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            delay=0.0,
        )
        return analytic.DecodedSearchPoint(
            parameters=parameters,
            actuator_parameters=actuator_parameters,
            delay=float(value[analytic.DELAY_INDEX]),
            inertia_principal_moments=principal,
            inertia_triangle_margin=triangle_margin,
        )

    def decode_with_jacobian(
        self, coordinate: Sequence[float]
    ) -> tuple[
        analytic.DecodedSearchPoint,
        analytic.DecodedSearchJacobian,
    ]:
        """Decode the chart and return its exact smooth Jacobian."""

        value = np.asarray(coordinate, dtype=float)
        decoded = self.decode(value)
        dimension = analytic.SMOOTH_DIMENSION

        mass = np.zeros(dimension, dtype=float)
        mass[0] = decoded.parameters.mass

        cholesky = self._cholesky(value)
        inertia = np.zeros((3, 3, dimension), dtype=float)

        def store_inertia_derivative(
            coordinate_index: int,
            cholesky_derivative: np.ndarray,
        ) -> None:
            second_moment_derivative = (
                cholesky_derivative @ cholesky.T
                + cholesky @ cholesky_derivative.T
            )
            inertia[:, :, coordinate_index] = (
                np.trace(second_moment_derivative) * np.eye(3)
                - second_moment_derivative
            )

        for axis in range(3):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[axis, axis] = cholesky[axis, axis]
            store_inertia_derivative(1 + axis, derivative)
        for local_index, (row, column) in enumerate(
            _CHOLESKY_OFF_DIAGONALS
        ):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[row, column] = math.sqrt(
                self.nominal_cholesky[row, row]
                * self.nominal_cholesky[column, column]
            )
            store_inertia_derivative(4 + local_index, derivative)

        cog_offset = np.zeros((3, dimension), dtype=float)
        cog_offset[:, 7:10] = np.eye(3)

        force_effectiveness = np.zeros((4, dimension), dtype=float)
        force_effectiveness[:, 10:13] = (
            decoded.parameters.force_effectiveness[:, None]
            * baseline.FORCE_EFFECTIVENESS_CONTRAST_BASIS
        )

        thrust_time_constant = np.zeros(dimension, dtype=float)
        gimbal_time_constant = np.zeros(dimension, dtype=float)
        return decoded, analytic.DecodedSearchJacobian(
            mass=mass,
            inertia=inertia,
            cog_offset=cog_offset,
            force_effectiveness=force_effectiveness,
            thrust_time_constant=thrust_time_constant,
            gimbal_time_constant=gimbal_time_constant,
        )


def segment_boundaries(
    sample_count: int,
    sample_step: float,
    segment_duration: float,
) -> np.ndarray:
    """Return output-grid indices delimiting all shooting segments."""

    if sample_count < 2 or sample_step <= 0.0 or segment_duration <= 0.0:
        raise ValueError("segment schedule inputs are invalid")
    intervals = max(1, int(round(segment_duration / sample_step)))
    boundaries = list(range(0, sample_count - 1, intervals))
    if boundaries[-1] != sample_count - 1:
        boundaries.append(sample_count - 1)
    result = np.asarray(boundaries, dtype=int)
    if result[0] != 0 or result[-1] != sample_count - 1:
        raise RuntimeError("segment schedule does not cover the output grid")
    return result


def se3_log_error(
    observed_position: Sequence[float],
    observed_rotation: np.ndarray,
    simulated_position: Sequence[float],
    simulated_rotation: np.ndarray,
) -> np.ndarray:
    """Return ``Log(T_obs^{-1} T_sim)`` as ``[rho, phi]``.

    ``phi`` is the principal SO(3) logarithm.  The relative translation is
    first expressed in the observed sensor frame, then mapped to the se(3)
    translational coordinate with ``J_l(phi)^{-1}``.
    """

    observed_p = np.asarray(observed_position, dtype=float)
    simulated_p = np.asarray(simulated_position, dtype=float)
    observed_r = np.asarray(observed_rotation, dtype=float)
    simulated_r = np.asarray(simulated_rotation, dtype=float)
    if (
        observed_p.shape != (3,)
        or simulated_p.shape != (3,)
        or observed_r.shape != (3, 3)
        or simulated_r.shape != (3, 3)
        or np.any(~np.isfinite(observed_p))
        or np.any(~np.isfinite(simulated_p))
        or np.any(~np.isfinite(observed_r))
        or np.any(~np.isfinite(simulated_r))
    ):
        raise ValueError("SE(3) error inputs must be finite poses")
    relative_rotation = observed_r.T @ simulated_r
    phi = so3_log(relative_rotation)
    relative_translation = observed_r.T @ (simulated_p - observed_p)
    rho = so3_left_jacobian_inverse(phi) @ relative_translation
    return np.concatenate((rho, phi))


def inertia_radius_se3_factor(nominal: VehicleParameters) -> np.ndarray:
    """Return ``W`` such that ``||W [rho, phi]||^2`` uses ``J0 / m0``.

    The per-sample pose quadratic is

        ||rho||^2 + phi.T @ (J0 / m0) @ phi,

    where ``J0`` and ``m0`` are fixed nominal values.  The upper-triangular
    Cholesky factor is used so that ``W.T @ W`` equals the desired metric.
    """

    if not isinstance(nominal, VehicleParameters):
        raise TypeError("nominal must be VehicleParameters")
    rotational_metric = np.asarray(nominal.inertia, dtype=float) / float(
        nominal.mass
    )
    rotational_metric = 0.5 * (rotational_metric + rotational_metric.T)
    rotational_factor = np.linalg.cholesky(rotational_metric).T
    factor = np.zeros((6, 6), dtype=float)
    factor[:3, :3] = np.eye(3)
    factor[3:, 3:] = rotational_factor
    return factor


def _rho_rotation_derivative(phi: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Differentiate ``J_l(phi)^{-1} translation`` with respect to ``phi``."""

    result = np.empty((3, 3), dtype=float)
    step = 1.0e-7 * max(1.0, float(np.linalg.norm(phi)))
    for axis in range(3):
        direction = np.zeros(3, dtype=float)
        direction[axis] = step
        plus = so3_left_jacobian_inverse(phi + direction) @ translation
        minus = so3_left_jacobian_inverse(phi - direction) @ translation
        result[:, axis] = (plus - minus) / (2.0 * step)
    return result


def _se3_log_error_with_jacobian(
    observed_position: np.ndarray,
    observed_rotation: np.ndarray,
    simulated_position: np.ndarray,
    simulated_rotation: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_right_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SE(3) log residual and its total-coordinate Jacobian."""

    relative_rotation = observed_rotation.T @ simulated_rotation
    phi = so3_log(relative_rotation)
    relative_translation = observed_rotation.T @ (
        simulated_position - observed_position
    )
    left_inverse = so3_left_jacobian_inverse(phi)
    rho = left_inverse @ relative_translation
    phi_jacobian = (
        so3_right_jacobian_inverse(phi) @ rotation_right_jacobian
    )
    translation_jacobian = observed_rotation.T @ position_jacobian
    rho_jacobian = (
        left_inverse @ translation_jacobian
        + _rho_rotation_derivative(phi, relative_translation) @ phi_jacobian
    )
    return (
        np.concatenate((rho, phi)),
        np.vstack((rho_jacobian, phi_jacobian)),
    )


def _physical_parameter_jacobian(
    parameterization: FullyPhysicalInertiaParameterization,
    physical_coordinate: Sequence[float],
    delay: float,
) -> tuple[analytic.DecodedSearchPoint, analytic.DecodedSearchJacobian]:
    expanded = continuation._expand_coordinate(physical_coordinate, delay)
    decoded, full_jacobian = parameterization.decode_with_jacobian(expanded)
    return decoded, analytic.DecodedSearchJacobian(
        mass=full_jacobian.mass[:PHYSICAL_DIMENSION],
        inertia=full_jacobian.inertia[:, :, :PHYSICAL_DIMENSION],
        cog_offset=full_jacobian.cog_offset[:, :PHYSICAL_DIMENSION],
        force_effectiveness=(
            full_jacobian.force_effectiveness[:, :PHYSICAL_DIMENSION]
        ),
        thrust_time_constant=np.zeros(PHYSICAL_DIMENSION, dtype=float),
        gimbal_time_constant=np.zeros(PHYSICAL_DIMENSION, dtype=float),
    )


def _extend_parameter_jacobian(
    source: analytic.DecodedSearchJacobian,
    dimension: int,
) -> analytic.DecodedSearchJacobian:
    source_dimension = int(np.asarray(source.mass).size)
    if source_dimension < PHYSICAL_DIMENSION or dimension < source_dimension:
        raise ValueError("extended derivative dimension is too small")

    def extend_vector(value: np.ndarray) -> np.ndarray:
        source_value = np.asarray(value, dtype=float)
        if source_value.shape != (source_dimension,):
            raise ValueError("parameter derivative vector has inconsistent width")
        result = np.zeros(dimension, dtype=float)
        result[:source_dimension] = source_value
        return result

    def extend_matrix(value: np.ndarray) -> np.ndarray:
        source_value = np.asarray(value, dtype=float)
        if source_value.ndim != 2 or source_value.shape[1] != source_dimension:
            raise ValueError("parameter derivative matrix has inconsistent width")
        result = np.zeros((source_value.shape[0], dimension), dtype=float)
        result[:, :source_dimension] = source_value
        return result

    source_inertia = np.asarray(source.inertia, dtype=float)
    if source_inertia.shape != (3, 3, source_dimension):
        raise ValueError("inertia derivative has inconsistent width")
    inertia = np.zeros((3, 3, dimension), dtype=float)
    inertia[:, :, :source_dimension] = source_inertia
    return analytic.DecodedSearchJacobian(
        mass=extend_vector(source.mass),
        inertia=inertia,
        cog_offset=extend_matrix(source.cog_offset),
        force_effectiveness=extend_matrix(source.force_effectiveness),
        thrust_time_constant=extend_vector(source.thrust_time_constant),
        gimbal_time_constant=extend_vector(source.gimbal_time_constant),
    )


def _decode_node(
    reference: NodeReference,
    correction: Sequence[float],
) -> tuple[RigidBodyState, ActuatorState, np.ndarray, np.ndarray]:
    value = np.asarray(correction, dtype=float)
    if value.shape != (NODE_DIMENSION,) or np.any(~np.isfinite(value)):
        raise ValueError("shooting-node correction must be finite and 20-D")
    rotation_vector = value[3:6]
    rotation = reference.rotation @ so3_exp(rotation_vector)
    quaternion = matrix_to_quaternion(rotation)
    rigid = RigidBodyState(
        position=reference.position + value[:3],
        orientation_xyzw=quaternion,
        linear_velocity=reference.linear_velocity + value[6:9],
        angular_velocity=reference.angular_velocity + value[9:12],
    )
    actuator = ActuatorState(
        thrust=reference.thrust + value[12:16],
        gimbal_angle=reference.gimbal + value[16:20],
    )
    rigid_jacobian = np.zeros((13, NODE_DIMENSION), dtype=float)
    rigid_jacobian[:3, :3] = np.eye(3)
    tangent = analytic._quaternion_right_tangent_matrix(quaternion)
    rigid_jacobian[3:7, 3:6] = (
        0.5 * tangent @ so3_right_jacobian(rotation_vector)
    )
    rigid_jacobian[7:10, 6:9] = np.eye(3)
    rigid_jacobian[10:13, 9:12] = np.eye(3)
    actuator_jacobian = np.zeros((8, NODE_DIMENSION), dtype=float)
    actuator_jacobian[:4, 12:16] = np.eye(4)
    actuator_jacobian[4:, 16:20] = np.eye(4)
    return rigid, actuator, rigid_jacobian, actuator_jacobian


def _encode_node(
    reference: NodeReference,
    rigid: RigidBodyState,
    actuator: ActuatorState,
) -> np.ndarray:
    rotation = quaternion_to_matrix(rigid.orientation_xyzw)
    result = np.empty(NODE_DIMENSION, dtype=float)
    result[:3] = rigid.position - reference.position
    result[3:6] = so3_log(reference.rotation.T @ rotation)
    result[6:9] = rigid.linear_velocity - reference.linear_velocity
    result[9:12] = rigid.angular_velocity - reference.angular_velocity
    result[12:16] = actuator.thrust - reference.thrust
    result[16:20] = actuator.gimbal_angle - reference.gimbal
    return result


def _actuator_step_with_sensitivity(
    state: ActuatorState,
    sensitivity: np.ndarray,
    command: Any,
    decoded: analytic.DecodedSearchPoint,
    parameter_jacobian: analytic.DecodedSearchJacobian,
    time_step: float,
    command_sensitivity: Optional[np.ndarray] = None,
) -> tuple[ActuatorState, np.ndarray]:
    value = np.asarray(sensitivity, dtype=float)
    dimension = value.shape[1]
    if value.shape[0] != 8:
        raise ValueError("actuator sensitivity must have eight rows")
    if command_sensitivity is None:
        command_derivative = np.zeros_like(value)
    else:
        command_derivative = np.asarray(command_sensitivity, dtype=float)
        if command_derivative.shape != value.shape:
            raise ValueError("command sensitivity must match actuator sensitivity")
    evaluation = advance_actuators_with_jacobian(
        state,
        command,
        decoded.actuator_parameters,
        time_step,
    )
    jacobian = evaluation.jacobian
    result = np.empty_like(value)
    result[:4] = (
        jacobian.thrust_previous @ value[:4]
        + jacobian.thrust_command @ command_derivative[:4]
    )
    result[4:] = (
        jacobian.gimbal_previous @ value[4:]
        + jacobian.gimbal_command @ command_derivative[4:]
    )

    thrust_tau_derivative = parameter_jacobian.thrust_time_constant
    thrust_tau = decoded.actuator_parameters.thrust_time_constant
    if thrust_tau > 0.0 and np.any(thrust_tau_derivative):
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
            thrust_tau_derivative,
        )

    gimbal_tau_derivative = parameter_jacobian.gimbal_time_constant
    gimbal_tau = decoded.actuator_parameters.gimbal_time_constant
    if gimbal_tau > 0.0 and np.any(gimbal_tau_derivative):
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
            gimbal_tau_derivative,
        )
    if result.shape != (8, dimension):
        raise RuntimeError("actuator sensitivity propagation changed shape")
    return evaluation.next_state, result


def _body_wrench_with_sensitivity(
    problem: baseline.DirectShootingProblem,
    decoded: analytic.DecodedSearchPoint,
    parameter_jacobian: analytic.DecodedSearchJacobian,
    rotation: np.ndarray,
    rotation_right_sensitivity: np.ndarray,
    linear_velocity: np.ndarray,
    linear_velocity_sensitivity: np.ndarray,
    angular_velocity: np.ndarray,
    angular_velocity_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = decoded.parameters
    wrench, jacobian = actuator_wrench_with_jacobian(
        actuators,
        parameters,
        problem.geometry,
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
    decoded: analytic.DecodedSearchPoint,
    parameter_jacobian: analytic.DecodedSearchJacobian,
    state_vector: np.ndarray,
    state_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(state_vector, dtype=float)
    sensitivity = np.asarray(state_sensitivity, dtype=float)
    if value.shape != (13,) or sensitivity.shape[0] != 13:
        raise ValueError("rigid state or sensitivity has the wrong shape")
    quaternion, normalization = analytic._normalise_quaternion_with_jacobian(
        value[3:7]
    )
    quaternion_sensitivity = normalization @ sensitivity[3:7]
    quaternion_tangent = analytic._quaternion_right_tangent_matrix(quaternion)
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
    force_per_mass = force / parameters.mass
    linear_acceleration = (
        np.asarray((0.0, 0.0, -GRAVITY)) + rotation @ force_per_mass
    )
    linear_acceleration_sensitivity = (
        -rotation @ skew(force_per_mass) @ rotation_right_sensitivity
        + rotation
        @ (
            wrench_sensitivity[:3] / parameters.mass
            - np.outer(force, parameter_jacobian.mass) / parameters.mass**2
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
    decoded: analytic.DecodedSearchPoint,
    parameter_jacobian: analytic.DecodedSearchJacobian,
    state: RigidBodyState,
    state_sensitivity: np.ndarray,
    actuators: ActuatorState,
    actuator_sensitivity: np.ndarray,
    time_step: float,
) -> tuple[RigidBodyState, np.ndarray]:
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
    next_vector[3:7], normalization = (
        analytic._normalise_quaternion_with_jacobian(next_vector[3:7])
    )
    next_sensitivity[3:7] = normalization @ next_sensitivity[3:7]
    return RigidBodyState.from_vector(next_vector), next_sensitivity


class MultipleShootingProblem:
    """One fixed-delay multiple-shooting inverse problem."""

    def __init__(
        self,
        *,
        direct_problem: baseline.DirectShootingProblem,
        delay: float,
        segment_duration: float,
        prior_weight: float,
        node_position_bound: float,
        node_orientation_bound: float,
        node_velocity_bound: float,
        node_angular_velocity_bound: float,
        global_dimension: int = PHYSICAL_DIMENSION,
    ) -> None:
        self.direct_problem = direct_problem
        self.delay = float(delay)
        self.prior_weight = float(prior_weight)
        self.global_dimension = int(global_dimension)
        if self.global_dimension < PHYSICAL_DIMENSION:
            raise ValueError("global dimension cannot omit physical coordinates")
        self.direct_problem.actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            delay=0.0,
        )
        nominal_parameters = VehicleParameters.nominal()
        self.parameterization = FullyPhysicalInertiaParameterization(
            nominal_parameters
        )
        self.pose_residual_factor = inertia_radius_se3_factor(
            nominal_parameters
        )
        self.boundaries = segment_boundaries(
            direct_problem.output_time.size,
            float(direct_problem.output_time[1] - direct_problem.output_time[0]),
            segment_duration,
        )
        self.segment_count = self.boundaries.size - 1
        self.node_count = self.segment_count - 1
        self.variable_dimension = (
            self.global_dimension + self.node_count * NODE_DIMENSION
        )
        self.pose_residual_dimension = direct_problem.output_time.size * 6
        self.data_residual_dimension = (
            self.pose_residual_dimension + PHYSICAL_DIMENSION
        )
        self.continuity_dimension = self.node_count * NODE_DIMENSION
        self.prior_scales = BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS.copy()
        self.continuity_scales = np.asarray(
            (
                0.05,
                0.05,
                0.05,
                0.10,
                0.10,
                0.10,
                0.20,
                0.20,
                0.20,
                0.20,
                0.20,
                0.20,
                1.0,
                1.0,
                1.0,
                1.0,
                0.05,
                0.05,
                0.05,
                0.05,
            ),
            dtype=float,
        )
        nominal_actuator_simulation = direct_problem.simulate(
            np.zeros(baseline.ACTIVE_PARAMETER_DIMENSION, dtype=float)
        )
        references: list[NodeReference] = []
        for output_index in self.boundaries[1:-1]:
            rotation = direct_problem.observed_body_rotation[output_index]
            omega = direct_problem.observed_omega_body[output_index]
            pose_lever = (
                direct_problem.pose_sensor_position
                - nominal_parameters.cog_offset
            )
            velocity_lever = (
                direct_problem.velocity_sensor_position
                - nominal_parameters.cog_offset
            )
            position = (
                direct_problem.observations.sensor_position[output_index]
                - rotation @ pose_lever
            )
            velocity = (
                direct_problem.observations.sensor_velocity_world[output_index]
                - rotation @ np.cross(omega, velocity_lever)
            )
            references.append(
                NodeReference(
                    position=position,
                    rotation=rotation,
                    linear_velocity=velocity,
                    angular_velocity=omega,
                    thrust=nominal_actuator_simulation.actuator_thrust[
                        output_index
                    ],
                    gimbal=nominal_actuator_simulation.actuator_gimbal[
                        output_index
                    ],
                )
            )
        self.node_references = tuple(references)
        self.node_bounds = np.asarray(
            (
                node_position_bound,
                node_position_bound,
                node_position_bound,
                node_orientation_bound,
                node_orientation_bound,
                node_orientation_bound,
                node_velocity_bound,
                node_velocity_bound,
                node_velocity_bound,
                node_angular_velocity_bound,
                node_angular_velocity_bound,
                node_angular_velocity_bound,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ),
            dtype=float,
        )

    def initial_coordinate(self) -> np.ndarray:
        return np.zeros(self.variable_dimension, dtype=float)

    def _decode_global_coordinate(
        self,
        global_coordinate: Sequence[float],
    ) -> tuple[
        analytic.DecodedSearchPoint,
        analytic.DecodedSearchJacobian,
    ]:
        return _physical_parameter_jacobian(
            self.parameterization,
            global_coordinate,
            self.delay,
        )

    def coordinate_delay(self, coordinate: Sequence[float]) -> float:
        del coordinate
        return self.delay

    def _command_with_sensitivity(
        self,
        step_index: int,
        local_dimension: int,
    ) -> tuple[Any, np.ndarray]:
        return (
            self.direct_problem.commands[step_index],
            np.zeros((8, local_dimension), dtype=float),
        )

    def split_coordinate(
        self, coordinate: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (self.variable_dimension,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError("multiple-shooting coordinate has the wrong shape")
        physical = value[: self.global_dimension]
        nodes = value[self.global_dimension :].reshape(
            self.node_count, NODE_DIMENSION
        )
        return physical, nodes

    def bounds(
        self,
        physical_lower: np.ndarray,
        physical_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = [np.asarray(physical_lower, dtype=float)]
        upper = [np.asarray(physical_upper, dtype=float)]
        actuator_parameters = self.direct_problem.actuator_parameters
        for reference in self.node_references:
            node_lower = -self.node_bounds.copy()
            node_upper = self.node_bounds.copy()
            node_lower[12:16] = (
                actuator_parameters.minimum_thrust - reference.thrust
            )
            node_upper[12:16] = (
                actuator_parameters.maximum_thrust - reference.thrust
            )
            node_lower[16:20] = (
                -actuator_parameters.maximum_gimbal_angle - reference.gimbal
            )
            node_upper[16:20] = (
                actuator_parameters.maximum_gimbal_angle - reference.gimbal
            )
            lower.append(node_lower)
            upper.append(node_upper)
        return np.concatenate(lower), np.concatenate(upper)

    def rebase_coordinate(
        self,
        previous: "MultipleShootingProblem",
        previous_coordinate: Sequence[float],
    ) -> np.ndarray:
        previous_physical, previous_nodes = previous.split_coordinate(
            previous_coordinate
        )
        if previous.node_count != self.node_count:
            raise ValueError("shooting schedules differ and cannot be rebased")
        result = self.initial_coordinate()
        if previous.global_dimension != self.global_dimension:
            raise ValueError("global coordinate dimensions differ")
        result[: self.global_dimension] = previous_physical
        for index in range(self.node_count):
            rigid, actuator, _, _ = _decode_node(
                previous.node_references[index],
                previous_nodes[index],
            )
            result[
                self.global_dimension
                + index * NODE_DIMENSION : self.global_dimension
                + (index + 1) * NODE_DIMENSION
            ] = _encode_node(self.node_references[index], rigid, actuator)
        return result

    def _initial_state_with_sensitivity(
        self,
        decoded: analytic.DecodedSearchPoint,
        physical_jacobian: analytic.DecodedSearchJacobian,
        segment_index: int,
        node_coordinate: Optional[np.ndarray],
    ) -> tuple[
        RigidBodyState,
        ActuatorState,
        np.ndarray,
        np.ndarray,
        analytic.DecodedSearchJacobian,
    ]:
        local_dimension = self.global_dimension + (
            NODE_DIMENSION if segment_index > 0 else 0
        )
        extended_jacobian = _extend_parameter_jacobian(
            physical_jacobian,
            local_dimension,
        )
        rigid_sensitivity = np.zeros((13, local_dimension), dtype=float)
        actuator_sensitivity = np.zeros((8, local_dimension), dtype=float)
        if segment_index == 0:
            rigid = self.direct_problem._initial_rigid_state(decoded.parameters)
            rigid_sensitivity[:3, : self.global_dimension] = (
                self.direct_problem.initial_body_rotation
                @ physical_jacobian.cog_offset
            )
            rigid_sensitivity[7:10, : self.global_dimension] = (
                self.direct_problem.initial_body_rotation
                @ skew(self.direct_problem.initial_omega_body)
                @ physical_jacobian.cog_offset
            )
            actuator = self.direct_problem.initial_actuator_state
        else:
            if node_coordinate is None:
                raise ValueError("internal segment requires a shooting node")
            rigid, actuator, node_rigid, node_actuator = _decode_node(
                self.node_references[segment_index - 1],
                node_coordinate,
            )
            rigid_sensitivity[:, self.global_dimension :] = node_rigid
            actuator_sensitivity[:, self.global_dimension :] = node_actuator
        return (
            rigid,
            actuator,
            rigid_sensitivity,
            actuator_sensitivity,
            extended_jacobian,
        )

    def _evaluate_segment(
        self,
        segment_index: int,
        decoded: analytic.DecodedSearchPoint,
        physical_jacobian: analytic.DecodedSearchJacobian,
        node_coordinate: Optional[np.ndarray],
    ) -> SegmentEvaluation:
        output_start = int(self.boundaries[segment_index])
        output_end = int(self.boundaries[segment_index + 1])
        internal_start = output_start * self.direct_problem.output_stride
        internal_end = output_end * self.direct_problem.output_stride
        (
            rigid,
            actuator,
            rigid_sensitivity,
            actuator_sensitivity,
            local_parameter_jacobian,
        ) = self._initial_state_with_sensitivity(
            decoded,
            physical_jacobian,
            segment_index,
            node_coordinate,
        )
        local_dimension = rigid_sensitivity.shape[1]
        output_indices = np.arange(output_start, output_end + 1, dtype=int)
        sensor_position = np.empty((output_indices.size, 3), dtype=float)
        sensor_orientation = np.empty((output_indices.size, 4), dtype=float)
        sensor_velocity = np.empty((output_indices.size, 3), dtype=float)
        angular_velocity_sensor = np.empty((output_indices.size, 3), dtype=float)
        specific_force_sensor = np.empty((output_indices.size, 3), dtype=float)
        pose_residual = np.empty((output_indices.size, 6), dtype=float)
        pose_jacobian = np.empty(
            (output_indices.size, 6, local_dimension), dtype=float
        )

        def store(local_index: int, output_index: int) -> None:
            quaternion = rigid.orientation_xyzw
            rotation = quaternion_to_matrix(quaternion)
            quaternion_tangent = analytic._quaternion_right_tangent_matrix(
                quaternion
            )
            rotation_right_sensitivity = (
                2.0 * quaternion_tangent.T @ rigid_sensitivity[3:7]
            )
            pose_lever = (
                self.direct_problem.pose_sensor_position
                - decoded.parameters.cog_offset
            )
            position = rigid.position + rotation @ pose_lever
            position_jacobian = (
                rigid_sensitivity[:3]
                - rotation @ skew(pose_lever) @ rotation_right_sensitivity
                - rotation @ local_parameter_jacobian.cog_offset
            )
            sensor_rotation = (
                rotation @ self.direct_problem.pose_body_to_sensor_rotation
            )
            sensor_rotation_right_sensitivity = (
                self.direct_problem.pose_body_to_sensor_rotation.T
                @ rotation_right_sensitivity
            )
            observed_position = (
                self.direct_problem.observations.sensor_position[output_index]
            )
            observed_rotation = (
                self.direct_problem.observed_sensor_rotation[output_index]
            )
            residual, jacobian = _se3_log_error_with_jacobian(
                observed_position,
                observed_rotation,
                position,
                sensor_rotation,
                position_jacobian,
                sensor_rotation_right_sensitivity,
            )
            sensor_position[local_index] = position
            sensor_orientation[local_index] = matrix_to_quaternion(
                sensor_rotation
            )
            velocity_lever = (
                self.direct_problem.velocity_sensor_position
                - decoded.parameters.cog_offset
            )
            sensor_velocity[local_index] = (
                rigid.linear_velocity
                + rotation
                @ np.cross(rigid.angular_velocity, velocity_lever)
            )
            angular_velocity_sensor[local_index] = (
                self.direct_problem.body_to_imu_rotation
                @ rigid.angular_velocity
                + self.direct_problem.gyro_bias
            )
            wrench = plant.total_body_wrench(
                float(self.direct_problem.output_time[output_index]),
                rigid,
                actuator,
            )
            inertia = decoded.parameters.inertia
            angular_acceleration = np.linalg.solve(
                inertia,
                wrench[3:]
                - np.cross(
                    rigid.angular_velocity,
                    inertia @ rigid.angular_velocity,
                ),
            )
            imu_lever = (
                self.direct_problem.imu_sensor_position
                - decoded.parameters.cog_offset
            )
            specific_force_body = (
                wrench[:3] / decoded.parameters.mass
                + np.cross(angular_acceleration, imu_lever)
                + np.cross(
                    rigid.angular_velocity,
                    np.cross(rigid.angular_velocity, imu_lever),
                )
            )
            specific_force_sensor[local_index] = (
                self.direct_problem.body_to_imu_rotation
                @ specific_force_body
                + self.direct_problem.accelerometer_bias
            )
            pose_residual[local_index] = residual
            pose_jacobian[local_index] = jacobian

        plant = FullSixDofPlant(
            decoded.parameters,
            self.direct_problem.geometry,
        )
        store(0, output_start)
        local_output_index = 1
        for step_index in range(internal_start, internal_end):
            command, command_sensitivity = self._command_with_sensitivity(
                step_index,
                local_dimension,
            )
            time_step = self.direct_problem.integration_step
            midpoint_actuator, midpoint_sensitivity = (
                _actuator_step_with_sensitivity(
                    actuator,
                    actuator_sensitivity,
                    command,
                    decoded,
                    local_parameter_jacobian,
                    0.5 * time_step,
                    command_sensitivity,
                )
            )
            rigid, rigid_sensitivity = _rigid_step_with_sensitivity(
                self.direct_problem,
                decoded,
                local_parameter_jacobian,
                rigid,
                rigid_sensitivity,
                midpoint_actuator,
                midpoint_sensitivity,
                time_step,
            )
            actuator, actuator_sensitivity = _actuator_step_with_sensitivity(
                midpoint_actuator,
                midpoint_sensitivity,
                command,
                decoded,
                local_parameter_jacobian,
                0.5 * time_step,
                command_sensitivity,
            )
            if (step_index + 1) % self.direct_problem.output_stride == 0:
                output_index = (step_index + 1) // self.direct_problem.output_stride
                store(local_output_index, output_index)
                local_output_index += 1
        if local_output_index != output_indices.size:
            raise RuntimeError("segment output grid is inconsistent")
        return SegmentEvaluation(
            output_indices=output_indices,
            sensor_position=sensor_position,
            sensor_orientation_xyzw=sensor_orientation,
            sensor_velocity_world=sensor_velocity,
            angular_velocity_sensor=angular_velocity_sensor,
            specific_force_sensor=specific_force_sensor,
            pose_residual=pose_residual,
            pose_jacobian=pose_jacobian,
            end_rigid=rigid,
            end_actuator=actuator,
            end_rigid_sensitivity=rigid_sensitivity,
            end_actuator_sensitivity=actuator_sensitivity,
        )

    def _continuity_block(
        self,
        segment_index: int,
        segment: SegmentEvaluation,
        next_node_coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        next_rigid, next_actuator, next_rigid_jacobian, next_actuator_jacobian = (
            _decode_node(
                self.node_references[segment_index],
                next_node_coordinate,
            )
        )
        global_jacobian = np.zeros(
            (NODE_DIMENSION, self.variable_dimension), dtype=float
        )
        local_columns = [np.arange(self.global_dimension, dtype=int)]
        if segment_index > 0:
            start = self.global_dimension + (segment_index - 1) * NODE_DIMENSION
            local_columns.append(np.arange(start, start + NODE_DIMENSION))
        local_columns_array = np.concatenate(local_columns)
        next_start = self.global_dimension + segment_index * NODE_DIMENSION
        next_columns = np.arange(next_start, next_start + NODE_DIMENSION)

        end_rigid = segment.end_rigid
        end_actuator = segment.end_actuator
        end_rotation = quaternion_to_matrix(end_rigid.orientation_xyzw)
        next_rotation = quaternion_to_matrix(next_rigid.orientation_xyzw)
        end_tangent = analytic._quaternion_right_tangent_matrix(
            end_rigid.orientation_xyzw
        )
        next_tangent = analytic._quaternion_right_tangent_matrix(
            next_rigid.orientation_xyzw
        )
        end_rotation_jacobian = (
            2.0 * end_tangent.T @ segment.end_rigid_sensitivity[3:7]
        )
        next_rotation_jacobian = (
            2.0 * next_tangent.T @ next_rigid_jacobian[3:7]
        )

        residual = np.empty(NODE_DIMENSION, dtype=float)
        residual[:3] = next_rigid.position - end_rigid.position
        global_jacobian[:3, local_columns_array] = (
            -segment.end_rigid_sensitivity[:3]
        )
        global_jacobian[:3, next_columns] = next_rigid_jacobian[:3]

        relative_rotation = end_rotation.T @ next_rotation
        rotation_residual = so3_log(relative_rotation)
        residual[3:6] = rotation_residual
        global_jacobian[3:6, local_columns_array] = (
            -so3_left_jacobian_inverse(rotation_residual)
            @ end_rotation_jacobian
        )
        global_jacobian[3:6, next_columns] = (
            so3_right_jacobian_inverse(rotation_residual)
            @ next_rotation_jacobian
        )

        residual[6:9] = (
            next_rigid.linear_velocity - end_rigid.linear_velocity
        )
        global_jacobian[6:9, local_columns_array] = (
            -segment.end_rigid_sensitivity[7:10]
        )
        global_jacobian[6:9, next_columns] = next_rigid_jacobian[7:10]

        rotated_next_omega = relative_rotation @ next_rigid.angular_velocity
        residual[9:12] = rotated_next_omega - end_rigid.angular_velocity
        global_jacobian[9:12, local_columns_array] = (
            skew(rotated_next_omega) @ end_rotation_jacobian
            - segment.end_rigid_sensitivity[10:13]
        )
        global_jacobian[9:12, next_columns] = (
            -relative_rotation
            @ skew(next_rigid.angular_velocity)
            @ next_rotation_jacobian
            + relative_rotation @ next_rigid_jacobian[10:13]
        )

        residual[12:16] = next_actuator.thrust - end_actuator.thrust
        global_jacobian[12:16, local_columns_array] = (
            -segment.end_actuator_sensitivity[:4]
        )
        global_jacobian[12:16, next_columns] = next_actuator_jacobian[:4]

        residual[16:20] = (
            next_actuator.gimbal_angle - end_actuator.gimbal_angle
        )
        global_jacobian[16:20, local_columns_array] = (
            -segment.end_actuator_sensitivity[4:]
        )
        global_jacobian[16:20, next_columns] = next_actuator_jacobian[4:]

        residual /= self.continuity_scales
        global_jacobian /= self.continuity_scales[:, None]
        return residual, global_jacobian

    def evaluate(
        self,
        coordinate: Sequence[float],
    ) -> ProblemEvaluation:
        global_coordinate, nodes = self.split_coordinate(coordinate)
        decoded, physical_jacobian = self._decode_global_coordinate(
            global_coordinate,
        )
        segment_evaluations: list[SegmentEvaluation] = []
        sensor_position = np.empty(
            (self.direct_problem.output_time.size, 3), dtype=float
        )
        sensor_orientation = np.empty(
            (self.direct_problem.output_time.size, 4), dtype=float
        )
        sensor_velocity = np.empty(
            (self.direct_problem.output_time.size, 3), dtype=float
        )
        angular_velocity_sensor = np.empty(
            (self.direct_problem.output_time.size, 3), dtype=float
        )
        specific_force_sensor = np.empty(
            (self.direct_problem.output_time.size, 3), dtype=float
        )
        pose_residual = np.empty(
            (self.direct_problem.output_time.size, 6), dtype=float
        )
        pose_jacobian = np.zeros(
            (
                self.direct_problem.output_time.size,
                6,
                self.variable_dimension,
            ),
            dtype=float,
        )
        for segment_index in range(self.segment_count):
            node_coordinate = (
                None if segment_index == 0 else nodes[segment_index - 1]
            )
            segment = self._evaluate_segment(
                segment_index,
                decoded,
                physical_jacobian,
                node_coordinate,
            )
            segment_evaluations.append(segment)
            local_columns = [np.arange(self.global_dimension, dtype=int)]
            if segment_index > 0:
                start = (
                    self.global_dimension
                    + (segment_index - 1) * NODE_DIMENSION
                )
                local_columns.append(np.arange(start, start + NODE_DIMENSION))
            columns = np.concatenate(local_columns)
            local_slice = slice(None) if segment_index == self.segment_count - 1 else slice(None, -1)
            indices = segment.output_indices[local_slice]
            sensor_position[indices] = segment.sensor_position[local_slice]
            sensor_orientation[indices] = segment.sensor_orientation_xyzw[local_slice]
            sensor_velocity[indices] = segment.sensor_velocity_world[local_slice]
            angular_velocity_sensor[indices] = (
                segment.angular_velocity_sensor[local_slice]
            )
            specific_force_sensor[indices] = (
                segment.specific_force_sensor[local_slice]
            )
            pose_residual[indices] = segment.pose_residual[local_slice]
            pose_jacobian[np.ix_(indices, np.arange(6), columns)] = (
                segment.pose_jacobian[local_slice]
            )

        count_scale = math.sqrt(self.direct_problem.output_time.size)
        normalized_pose = (
            np.einsum("ij,nj->ni", self.pose_residual_factor, pose_residual)
            / count_scale
        )
        normalized_pose_jacobian = (
            np.einsum(
                "ij,njk->nik",
                self.pose_residual_factor,
                pose_jacobian,
            )
            / count_scale
        )
        prior_residual = (
            math.sqrt(self.prior_weight)
            * global_coordinate[:PHYSICAL_DIMENSION]
            / self.prior_scales
        )
        prior_jacobian = np.zeros(
            (PHYSICAL_DIMENSION, self.variable_dimension), dtype=float
        )
        prior_jacobian[:, :PHYSICAL_DIMENSION] = np.diag(
            math.sqrt(self.prior_weight) / self.prior_scales
        )
        data_residual = np.concatenate(
            (normalized_pose.reshape(-1), prior_residual)
        )
        data_jacobian = np.vstack(
            (normalized_pose_jacobian.reshape(-1, self.variable_dimension), prior_jacobian)
        )

        continuity_residual = np.empty(self.continuity_dimension, dtype=float)
        continuity_jacobian = np.empty(
            (self.continuity_dimension, self.variable_dimension), dtype=float
        )
        for boundary_index in range(self.node_count):
            block, block_jacobian = self._continuity_block(
                boundary_index,
                segment_evaluations[boundary_index],
                nodes[boundary_index],
            )
            start = boundary_index * NODE_DIMENSION
            continuity_residual[start : start + NODE_DIMENSION] = block
            continuity_jacobian[start : start + NODE_DIMENSION] = block_jacobian
        if (
            np.any(~np.isfinite(data_residual))
            or np.any(~np.isfinite(data_jacobian))
            or np.any(~np.isfinite(continuity_residual))
            or np.any(~np.isfinite(continuity_jacobian))
        ):
            raise FloatingPointError("multiple-shooting evaluation is non-finite")
        return ProblemEvaluation(
            data_residual=data_residual,
            data_jacobian=data_jacobian,
            continuity_residual=continuity_residual,
            continuity_jacobian=continuity_jacobian,
            sensor_position=sensor_position,
            sensor_orientation_xyzw=sensor_orientation,
            sensor_velocity_world=sensor_velocity,
            angular_velocity_sensor=angular_velocity_sensor,
            specific_force_sensor=specific_force_sensor,
            decoded=decoded,
        )

    def full_rollout(
        self,
        global_coordinate: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        decoded, physical_jacobian = self._decode_global_coordinate(
            global_coordinate,
        )
        original_boundaries = self.boundaries
        original_segment_count = self.segment_count
        try:
            self.boundaries = np.asarray(
                (0, self.direct_problem.output_time.size - 1), dtype=int
            )
            self.segment_count = 1
            segment = self._evaluate_segment(
                0,
                decoded,
                physical_jacobian,
                None,
            )
        finally:
            self.boundaries = original_boundaries
            self.segment_count = original_segment_count
        residual = (
            np.einsum(
                "ij,nj->ni",
                self.pose_residual_factor,
                segment.pose_residual,
            )
            / math.sqrt(self.direct_problem.output_time.size)
        ).reshape(-1)
        return (
            segment.sensor_position,
            segment.sensor_orientation_xyzw,
            residual,
        )


class _CachedAugmentedObjective:
    def __init__(
        self,
        problem: MultipleShootingProblem,
        multipliers: np.ndarray,
        penalty: float,
    ) -> None:
        self.problem = problem
        self.multipliers = np.asarray(multipliers, dtype=float)
        self.penalty = float(penalty)
        self.coordinate: Optional[np.ndarray] = None
        self.evaluation: Optional[ProblemEvaluation] = None
        self.residual_value: Optional[np.ndarray] = None
        self.jacobian_value: Optional[np.ndarray] = None
        self.evaluation_count = 0
        self.best_data_cost = float("inf")
        self.last_progress_time = 0.0

    def _evaluate(self, coordinate: Sequence[float]) -> None:
        value = np.asarray(coordinate, dtype=float)
        if self.coordinate is not None and np.array_equal(value, self.coordinate):
            return
        self.evaluation_count += 1
        try:
            evaluation = self.problem.evaluate(value)
            root_penalty = math.sqrt(self.penalty)
            augmented_continuity = (
                root_penalty * evaluation.continuity_residual
                + self.multipliers / root_penalty
            )
            self.residual_value = np.concatenate(
                (evaluation.data_residual, augmented_continuity)
            )
            self.jacobian_value = np.vstack(
                (
                    evaluation.data_jacobian,
                    root_penalty * evaluation.continuity_jacobian,
                )
            )
            self.evaluation = evaluation
            data_cost = 0.5 * float(
                evaluation.data_residual @ evaluation.data_residual
            )
            continuity_max = (
                0.0
                if evaluation.continuity_residual.size == 0
                else float(np.max(np.abs(evaluation.continuity_residual)))
            )
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as error:
            total_dimension = (
                self.problem.data_residual_dimension
                + self.problem.continuity_dimension
            )
            residual = np.full(total_dimension, 1.0e3, dtype=float)
            jacobian = np.zeros(
                (total_dimension, self.problem.variable_dimension),
                dtype=float,
            )
            diagonal_count = min(total_dimension, value.size)
            residual[:diagonal_count] *= (
                1.0 + 0.01 * value[:diagonal_count] ** 2
            )
            jacobian[
                np.arange(diagonal_count), np.arange(diagonal_count)
            ] = 20.0 * value[:diagonal_count]
            self.residual_value = residual
            self.jacobian_value = jacobian
            self.evaluation = None
            data_cost = 0.5 * float(residual @ residual)
            continuity_max = float("inf")
            print(
                "  rejected divergent trial: {}".format(error),
                flush=True,
            )
        self.coordinate = value.copy()
        self.best_data_cost = min(self.best_data_cost, data_cost)
        now = time.monotonic()
        if (
            self.evaluation_count == 1
            or self.evaluation_count % 10 == 0
            or now - self.last_progress_time >= 5.0
        ):
            print(
                "  eval {:4d}: data={:.8g}, best={:.8g}, continuity_max={:.3e}".format(
                    self.evaluation_count,
                    data_cost,
                    self.best_data_cost,
                    continuity_max,
                ),
                flush=True,
            )
            self.last_progress_time = now

    def residual(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.residual_value is None:
            raise RuntimeError("augmented residual cache is empty")
        return self.residual_value

    def jacobian(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.jacobian_value is None:
            raise RuntimeError("augmented Jacobian cache is empty")
        return self.jacobian_value


def _solve_fixed_delay(
    problem: MultipleShootingProblem,
    initial_coordinate: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    arguments: argparse.Namespace,
) -> FixedDelaySolution:
    started = time.perf_counter()
    coordinate = np.clip(initial_coordinate, bounds[0], bounds[1])
    multipliers = np.zeros(problem.continuity_dimension, dtype=float)
    penalty = float(arguments.continuity_penalty_initial)
    previous_norm = float("inf")
    history: list[dict[str, Any]] = []
    final_evaluation: Optional[ProblemEvaluation] = None
    for outer_iteration in range(arguments.augmented_lagrangian_iterations):
        print(
            "delay {:.6f}s, augmented iteration {}/{}, penalty={:.3g}".format(
                problem.coordinate_delay(coordinate),
                outer_iteration + 1,
                arguments.augmented_lagrangian_iterations,
                penalty,
            ),
            flush=True,
        )
        objective = _CachedAugmentedObjective(problem, multipliers, penalty)
        result = least_squares(
            objective.residual,
            coordinate,
            jac=objective.jacobian,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            loss="linear",
            ftol=arguments.ftol,
            xtol=arguments.xtol,
            gtol=arguments.gtol,
            max_nfev=arguments.max_nfev,
            verbose=1,
        )
        coordinate = result.x.copy()
        final_evaluation = problem.evaluate(coordinate)
        continuity = final_evaluation.continuity_residual
        continuity_norm = float(np.linalg.norm(continuity))
        continuity_max = (
            0.0 if continuity.size == 0 else float(np.max(np.abs(continuity)))
        )
        data_cost = 0.5 * float(
            final_evaluation.data_residual @ final_evaluation.data_residual
        )
        history.append(
            {
                "outer_iteration": outer_iteration + 1,
                "penalty": penalty,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "data_cost": data_cost,
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "continuity_l2_normalized": continuity_norm,
                "continuity_max_normalized": continuity_max,
                "delay_seconds": problem.coordinate_delay(coordinate),
            }
        )
        print(
            "  data={:.9g}, continuity L2={:.3e}, max={:.3e}".format(
                data_cost,
                continuity_norm,
                continuity_max,
            ),
            flush=True,
        )
        if continuity_max <= arguments.continuity_tolerance:
            break
        multipliers = multipliers + penalty * continuity
        if continuity_norm > arguments.penalty_reduction_target * previous_norm:
            penalty = min(
                arguments.continuity_penalty_max,
                penalty * arguments.continuity_penalty_growth,
            )
        previous_norm = continuity_norm
    if final_evaluation is None:
        raise RuntimeError("fixed-delay solver did not evaluate a solution")
    physical, _ = problem.split_coordinate(coordinate)
    full_position, full_orientation, full_residual = problem.full_rollout(physical)
    return FixedDelaySolution(
        delay=problem.coordinate_delay(coordinate),
        coordinate=coordinate,
        optimizer_history=tuple(history),
        evaluation=final_evaluation,
        full_rollout_position=full_position,
        full_rollout_orientation_xyzw=full_orientation,
        full_rollout_residual=full_residual,
        elapsed_seconds=time.perf_counter() - started,
    )


def _pose_metrics(residual: np.ndarray) -> dict[str, float]:
    value = np.asarray(residual, dtype=float).reshape(-1, 6)
    translation = value[:, :3]
    rotation = value[:, 3:]
    translation_norm = np.linalg.norm(translation, axis=1)
    rotation_norm = np.linalg.norm(rotation, axis=1)
    return {
        "se3_translation_rmse_m": float(
            np.sqrt(np.mean(translation_norm * translation_norm))
        ),
        "se3_rotation_rmse_rad": float(
            np.sqrt(np.mean(rotation_norm * rotation_norm))
        ),
        "se3_rotation_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(rotation_norm * rotation_norm)))
        ),
        "se3_translation_max_m": float(np.max(translation_norm)),
        "se3_rotation_max_rad": float(np.max(rotation_norm)),
    }


def _unscaled_pose_residual(
    problem: MultipleShootingProblem,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            se3_log_error(
                problem.direct_problem.observations.sensor_position[index],
                problem.direct_problem.observed_sensor_rotation[index],
                position[index],
                quaternion_to_matrix(orientation_xyzw[index]),
            )
            for index in range(problem.direct_problem.output_time.size)
        ],
        dtype=float,
    )


def _stitched_simulation(
    problem: MultipleShootingProblem,
    evaluation: ProblemEvaluation,
) -> baseline.Simulation:
    """Expose a stitched multiple-shooting trajectory as a Simulation."""

    sample_count = problem.direct_problem.output_time.size
    return baseline.Simulation(
        time=problem.direct_problem.output_time,
        sensor_position=evaluation.sensor_position,
        sensor_orientation_xyzw=evaluation.sensor_orientation_xyzw,
        sensor_velocity_world=evaluation.sensor_velocity_world,
        angular_velocity_sensor=evaluation.angular_velocity_sensor,
        specific_force_sensor=evaluation.specific_force_sensor,
        cog_position=np.zeros((sample_count, 3), dtype=float),
        cog_velocity_world=np.zeros((sample_count, 3), dtype=float),
        actuator_thrust=np.zeros((sample_count, 4), dtype=float),
        actuator_gimbal=np.zeros((sample_count, 4), dtype=float),
    )


def _plot_stitched_vector_comparison(
    axes: Sequence[Any],
    time_axis: np.ndarray,
    target: np.ndarray,
    stitched: np.ndarray,
    labels: Sequence[str],
    ylabel: str,
) -> None:
    relative_time = time_axis - time_axis[0]
    for component, axis in enumerate(axes):
        axis.plot(
            relative_time,
            target[:, component],
            color="#1e5abe",
            linewidth=2.0,
            linestyle="-",
            label="target (observed rosbag)",
        )
        axis.plot(
            relative_time,
            stitched[:, component],
            color="#1e965f",
            linewidth=1.8,
            linestyle=":",
            label="stitched estimate",
        )
        axis.set_ylabel("{} {}".format(labels[component], ylabel))
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time since first common sample [s]")
    axes[0].legend(loc="best", fontsize=8)


def _parameter_summary_lines(
    problem: MultipleShootingProblem,
    solution: FixedDelaySolution,
    stitched_metrics: dict[str, Any],
    continuity_tolerance: float,
) -> list[str]:
    parameters = solution.evaluation.decoded.parameters
    inertia = parameters.inertia
    principal_moments = np.linalg.eigvalsh(inertia)
    physical_coordinate, _ = problem.split_coordinate(solution.coordinate)
    continuity = solution.evaluation.continuity_residual
    continuity_max = (
        0.0 if continuity.size == 0 else float(np.max(np.abs(continuity)))
    )
    stitched_loss = 0.5 * float(
        solution.evaluation.data_residual[: problem.pose_residual_dimension]
        @ solution.evaluation.data_residual[: problem.pose_residual_dimension]
    )
    full_rollout_loss = 0.5 * float(
        solution.full_rollout_residual @ solution.full_rollout_residual
    )
    soft_prior_cost = 0.5 * problem.prior_weight * float(
        (physical_coordinate / problem.prior_scales)
        @ (physical_coordinate / problem.prior_scales)
    )
    lines = [
        "Selected multiple-shooting estimate",
        "",
        "Decoded physical parameters",
        "  mass [kg]                  {:.12g}".format(parameters.mass),
        "  inertia [kg m^2]",
    ]
    lines.extend(
        "    [{: .12g}  {: .12g}  {: .12g}]".format(*row)
        for row in inertia
    )
    lines.extend(
        (
            "  principal inertia [kg m^2] [{:.12g}, {:.12g}, {:.12g}]".format(
                *principal_moments
            ),
            "  CoG offset [m]             [{:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.cog_offset
            ),
            "  rotor force effectiveness  [{:.12g}, {:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.force_effectiveness
            ),
            "  command delay [s]           {:.12g}".format(solution.delay),
            "",
            "Thirteen optimized smooth coordinates",
        )
    )
    lines.extend(
        "  {:<43s} {: .12g}".format(name, value)
        for name, value in zip(PHYSICAL_PARAMETER_NAMES, physical_coordinate)
    )
    lines.extend(
        (
            "",
            "Fit diagnostics",
            "  stitched inertia-radius loss [m^2]     {:.12g}".format(
                stitched_loss
            ),
            "  full-rollout inertia-radius loss [m^2] {:.12g}".format(
                full_rollout_loss
            ),
            "  broad soft-prior cost       {:.12g}".format(soft_prior_cost),
            "  position RMSE [m]           {:.12g}".format(
                stitched_metrics["position_rmse_m"]
            ),
            "  orientation RMSE [deg]      {:.12g}".format(
                stitched_metrics["orientation_angle_rmse_deg"]
            ),
            "  velocity RMSE [m/s]         {:.12g}".format(
                stitched_metrics["velocity_rmse_m_per_s"]
            ),
            "  angular velocity RMSE       {:.12g} rad/s".format(
                stitched_metrics["angular_velocity_rmse_rad_per_s"]
            ),
            "  specific force RMSE         {:.12g} m/s^2".format(
                stitched_metrics["specific_force_rmse_m_per_s2"]
            ),
            "  continuity max (normalized) {:.12g}".format(continuity_max),
            "  continuity tolerance        {:.12g}".format(
                continuity_tolerance
            ),
            "  continuity converged        {}".format(
                continuity_max <= continuity_tolerance
            ),
        )
    )
    return lines


def _write_text(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")
    temporary.replace(path)


def _write_pdf(
    path: Path,
    problem: MultipleShootingProblem,
    solution: FixedDelaySolution,
    stitched_metrics: dict[str, Any],
    parameter_lines: Sequence[str],
    continuity_tolerance: float,
) -> None:
    time_axis = problem.direct_problem.output_time
    target = problem.direct_problem.observations
    stitched = _stitched_simulation(problem, solution.evaluation)
    target_rpy = baseline._rpy_series(target.sensor_orientation_xyzw)
    stitched_rpy = baseline._rpy_series(stitched.sensor_orientation_xyzw)
    vector_specs = (
        (
            "Sensor position",
            target.sensor_position,
            stitched.sensor_position,
            ("x", "y", "z"),
            "[m]",
        ),
        (
            "Sensor orientation (display only; fit uses SO(3) log)",
            target_rpy,
            stitched_rpy,
            ("roll", "pitch", "yaw"),
            "[rad]",
        ),
        (
            "World-frame sensor velocity (diagnostic; excluded from fit)",
            target.sensor_velocity_world,
            stitched.sensor_velocity_world,
            ("vx", "vy", "vz"),
            "[m/s]",
        ),
        (
            "IMU angular velocity (diagnostic; excluded from fit)",
            target.angular_velocity_sensor,
            stitched.angular_velocity_sensor,
            ("wx", "wy", "wz"),
            "[rad/s]",
        ),
        (
            "IMU specific force (diagnostic; excluded from fit)",
            target.specific_force_sensor,
            stitched.specific_force_sensor,
            ("fx", "fy", "fz"),
            "[m/s²]",
        ),
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        figure.suptitle("Target and stitched multiple-shooting estimate", fontsize=15)
        grid = figure.add_gridspec(2, 2)
        axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
        for values, color, linestyle, linewidth, label in (
            (
                target.sensor_position,
                "#1e5abe",
                "-",
                2.5,
                "target (observed rosbag)",
            ),
            (
                stitched.sensor_position,
                "#1e965f",
                ":",
                1.8,
                "stitched estimate",
            ),
        ):
            axis_3d.plot(
                values[:, 0],
                values[:, 1],
                values[:, 2],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        axis_3d.set_xlabel("x [m]")
        axis_3d.set_ylabel("y [m]")
        axis_3d.set_zlabel("z [m]")
        axis_3d.set_title("Stitched trajectory")
        axis_3d.legend(loc="best", fontsize=8)

        metric_axis = figure.add_subplot(grid[0, 1])
        metric_axis.axis("off")
        continuity = solution.evaluation.continuity_residual
        continuity_max = (
            0.0
            if continuity.size == 0
            else float(np.max(np.abs(continuity)))
        )
        metric_lines = [
            "actual common support: {:.3f}–{:.3f} s".format(
                target.time[0], target.time[-1]
            ),
            "selected command delay: {:.6g} s".format(solution.delay),
            "",
            "metric                              stitched",
        ]
        for key, label in (
            ("position_rmse_m", "position RMSE [m]"),
            ("orientation_angle_rmse_deg", "orientation RMSE [deg]"),
            ("velocity_rmse_m_per_s", "velocity RMSE [m/s]"),
            (
                "angular_velocity_rmse_rad_per_s",
                "angular velocity RMSE [rad/s]",
            ),
            ("specific_force_rmse_m_per_s2", "specific force RMSE [m/s²]"),
        ):
            metric_lines.append(
                "{:<35s} {:>10.5g}".format(label, stitched_metrics[key])
            )
        metric_lines.extend(
            (
                "",
                "continuity max / tolerance:",
                "{:.4g} / {:.4g} ({})".format(
                    continuity_max,
                    continuity_tolerance,
                    "converged"
                    if continuity_max <= continuity_tolerance
                    else "not converged",
                ),
                "",
                "Velocity and IMU metrics are diagnostic only;",
                "the fit uses sensor position and orientation.",
            )
        )
        metric_axis.text(
            0.0,
            1.0,
            "\n".join(metric_lines),
            va="top",
            family="monospace",
            fontsize=8.5,
        )

        error_axis = figure.add_subplot(grid[1, 1])
        relative_time = target.time - target.time[0]
        error_axis.plot(
            relative_time,
            np.linalg.norm(
                stitched.sensor_position - target.sensor_position,
                axis=1,
            ),
            color="#1e965f",
            linestyle=":",
            linewidth=1.8,
            label="stitched position error",
        )
        error_axis.set_xlabel("time since first common sample [s]")
        error_axis.set_ylabel("position error norm [m]")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend(loc="best", fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)

        for title, target_value, stitched_value, labels, ylabel in vector_specs:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            figure.suptitle(title)
            _plot_stitched_vector_comparison(
                axes,
                time_axis,
                target_value,
                stitched_value,
                labels,
                ylabel,
            )
            pdf.savefig(figure)
            plt.close(figure)

        continuity_by_boundary = solution.evaluation.continuity_residual.reshape(
            -1, NODE_DIMENSION
        )
        figure, axis = plt.subplots(figsize=(11.0, 5.5))
        if continuity_by_boundary.size:
            axis.semilogy(
                np.arange(continuity_by_boundary.shape[0]),
                np.max(np.abs(continuity_by_boundary), axis=1),
                marker="o",
            )
        axis.axhline(
            continuity_tolerance,
            linestyle="--",
            label="continuity tolerance",
        )
        axis.set_xlabel("shooting boundary")
        axis.set_ylabel("max normalized continuity residual")
        axis.grid(True)
        axis.legend(loc="best")
        figure.suptitle("Shooting continuity diagnostic")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(
            0.02,
            0.98,
            "\n".join(parameter_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.4,
        )
        figure.suptitle("Final estimated parameters", fontsize=15)
        pdf.savefig(figure)
        plt.close(figure)


def _write_delay_profile_pdf(
    path: Path,
    ordered: Sequence[tuple[MultipleShootingProblem, FixedDelaySolution]],
    selected_delay: float,
    continuity_tolerance: float,
) -> None:
    delays_ms = np.asarray(
        [solution.delay * 1000.0 for _problem, solution in ordered],
        dtype=float,
    )
    full_losses = np.asarray(
        [_solution_full_rollout_loss(solution) for _problem, solution in ordered],
        dtype=float,
    )
    stitched_losses = np.asarray(
        [
            0.5
            * float(
                solution.evaluation.data_residual[: problem.pose_residual_dimension]
                @ solution.evaluation.data_residual[: problem.pose_residual_dimension]
            )
            for problem, solution in ordered
        ],
        dtype=float,
    )
    continuity = np.asarray(
        [
            0.0
            if solution.evaluation.continuity_residual.size == 0
            else float(
                np.max(np.abs(solution.evaluation.continuity_residual))
            )
            for _problem, solution in ordered
        ],
        dtype=float,
    )
    selected_index = int(
        np.argmin(np.abs(delays_ms - float(selected_delay) * 1000.0))
    )
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            delays_ms,
            full_losses,
            marker="o",
            label="full-rollout inertia-radius loss",
        )
        axes[0].plot(
            delays_ms,
            stitched_losses,
            marker=".",
            linestyle=":",
            label="stitched inertia-radius loss",
        )
        axes[0].scatter(
            [delays_ms[selected_index]],
            [full_losses[selected_index]],
            marker="*",
            s=180,
            color="#1e965f",
            label="selected",
            zorder=4,
        )
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        axes[1].semilogy(delays_ms, continuity, marker="o")
        axes[1].axhline(
            continuity_tolerance,
            linestyle="--",
            color="#555555",
            label="continuity tolerance",
        )
        axes[1].set_xlabel("recorded-command delay [ms]")
        axes[1].set_ylabel("max normalized continuity residual")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")
        figure.suptitle(
            "Fixed command-delay evaluation"
            if len(ordered) == 1
            else "Adaptive delay profile"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _solution_full_rollout_loss(solution: FixedDelaySolution) -> float:
    return 0.5 * float(
        solution.full_rollout_residual @ solution.full_rollout_residual
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit mass, fully physical second-moment/Cholesky inertia, CoG, "
            "relative force effectiveness at a fixed recorded-command lag "
            "by inertia-radius-normalized SE(3)-only multiple shooting."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=1.0,
        help=(
            "multiplier for the broad nominal-centered Gaussian soft prior; "
            "zero disables it without adding hard bounds"
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--augmented-lagrangian-iterations", type=int, default=10
    )
    parser.add_argument(
        "--continuity-penalty-initial", type=float, default=1.0
    )
    parser.add_argument(
        "--continuity-penalty-growth", type=float, default=10.0
    )
    parser.add_argument(
        "--continuity-penalty-max", type=float, default=1.0e6
    )
    parser.add_argument(
        "--penalty-reduction-target", type=float, default=0.50
    )
    parser.add_argument("--continuity-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument("--command-delay", type=float, default=0.16)
    parser.add_argument("--node-position-bound", type=float, default=2.0)
    parser.add_argument("--node-orientation-bound", type=float, default=1.5)
    parser.add_argument("--node-velocity-bound", type=float, default=5.0)
    parser.add_argument(
        "--node-angular-velocity-bound", type=float, default=10.0
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    return parser


def _validate_arguments(
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    finite_positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.segment_duration,
        arguments.max_nfev,
        arguments.augmented_lagrangian_iterations,
        arguments.continuity_penalty_initial,
        arguments.continuity_penalty_growth,
        arguments.continuity_penalty_max,
        arguments.penalty_reduction_target,
        arguments.continuity_tolerance,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.node_position_bound,
        arguments.node_orientation_bound,
        arguments.node_velocity_bound,
        arguments.node_angular_velocity_bound,
    )
    if (
        not np.isfinite(arguments.start)
        or not np.isfinite(arguments.end)
        or arguments.start >= arguments.end
        or any(not np.isfinite(value) or value <= 0.0 for value in finite_positive)
        or not np.isfinite(arguments.prior_weight)
        or arguments.prior_weight < 0.0
        or not np.isfinite(arguments.command_delay)
        or arguments.command_delay < 0.0
        or arguments.continuity_penalty_growth <= 1.0
        or not 0.0 < arguments.penalty_reduction_target < 1.0
    ):
        raise SystemExit("multiple-shooting settings are invalid")
    physical_lower = np.full(PHYSICAL_DIMENSION, -np.inf, dtype=float)
    physical_upper = np.full(PHYSICAL_DIMENSION, np.inf, dtype=float)
    return physical_lower, physical_upper, float(arguments.command_delay)


def _physical_payload(
    decoded: analytic.DecodedSearchPoint,
) -> dict[str, Any]:
    result = baseline._physical_parameters(decoded.parameters)
    result["command_delay_seconds"] = decoded.delay
    return result


def _solution_payload(
    problem: MultipleShootingProblem,
    solution: FixedDelaySolution,
    continuity_tolerance: float,
) -> dict[str, Any]:
    physical, _ = problem.split_coordinate(solution.coordinate)
    stitched_unscaled = _unscaled_pose_residual(
        problem,
        solution.evaluation.sensor_position,
        solution.evaluation.sensor_orientation_xyzw,
    )
    full_unscaled = _unscaled_pose_residual(
        problem,
        solution.full_rollout_position,
        solution.full_rollout_orientation_xyzw,
    )
    continuity = solution.evaluation.continuity_residual
    continuity_max = (
        0.0 if continuity.size == 0 else float(np.max(np.abs(continuity)))
    )
    soft_prior_cost = 0.5 * problem.prior_weight * float(
        (physical / problem.prior_scales)
        @ (physical / problem.prior_scales)
    )
    stitched_recorded_control_metrics = baseline._metrics(
        problem.direct_problem,
        _stitched_simulation(problem, solution.evaluation),
    )
    return {
        "delay_seconds": solution.delay,
        "physical_coordinate": physical,
        "stitched_inertia_radius_loss_m2": 0.5
        * float(
            solution.evaluation.data_residual[: problem.pose_residual_dimension]
            @ solution.evaluation.data_residual[: problem.pose_residual_dimension]
        ),
        "full_rollout_inertia_radius_loss_m2": 0.5
        * float(solution.full_rollout_residual @ solution.full_rollout_residual),
        "stitched_metrics": _pose_metrics(stitched_unscaled),
        "stitched_recorded_control_metrics": (
            stitched_recorded_control_metrics
        ),
        "full_rollout_metrics": _pose_metrics(full_unscaled),
        "soft_prior_cost": soft_prior_cost,
        "continuity_max_normalized": continuity_max,
        "continuity_l2_normalized": float(np.linalg.norm(continuity)),
        "continuity_converged": bool(
            continuity_max <= continuity_tolerance
        ),
        "parameters": _physical_payload(solution.evaluation.decoded),
        "optimizer_history": list(solution.optimizer_history),
        "elapsed_seconds": solution.elapsed_seconds,
    }


def run(arguments: argparse.Namespace) -> int:
    physical_lower, physical_upper, command_delay = _validate_arguments(
        arguments
    )
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
    flight = load_flight_data(
        str(bag),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    direct_problem = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=command_delay,
        prior_weight=arguments.prior_weight,
    )
    selected_problem = MultipleShootingProblem(
        direct_problem=direct_problem,
        delay=command_delay,
        segment_duration=arguments.segment_duration,
        prior_weight=arguments.prior_weight,
        node_position_bound=arguments.node_position_bound,
        node_orientation_bound=arguments.node_orientation_bound,
        node_velocity_bound=arguments.node_velocity_bound,
        node_angular_velocity_bound=arguments.node_angular_velocity_bound,
    )
    bounds = selected_problem.bounds(physical_lower, physical_upper)
    print(
        "solving fixed delay {:.6f}s ({:d} segments, {:d} variables; "
        "start=nominal)".format(
            command_delay,
            selected_problem.segment_count,
            selected_problem.variable_dimension,
        ),
        flush=True,
    )
    selected_solution = _solve_fixed_delay(
        selected_problem,
        selected_problem.initial_coordinate(),
        bounds,
        arguments,
    )
    nominal_problem = selected_problem
    nominal_position, nominal_orientation, nominal_residual = (
        nominal_problem.full_rollout(np.zeros(PHYSICAL_DIMENSION, dtype=float))
    )
    output_directory = (
        arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "result.json"
    pdf_path = output_directory / "trajectory.pdf"
    delay_profile_path = output_directory / "delay_profile.pdf"
    parameters_path = output_directory / "parameters.txt"
    ordered = [(selected_problem, selected_solution)]
    selected_payload = _solution_payload(
        selected_problem,
        selected_solution,
        arguments.continuity_tolerance,
    )
    stitched_metrics = selected_payload["stitched_recorded_control_metrics"]
    parameter_lines = _parameter_summary_lines(
        selected_problem,
        selected_solution,
        stitched_metrics,
        arguments.continuity_tolerance,
    )
    nominal_parameters = VehicleParameters.nominal()
    nominal_rotational_metric = (
        nominal_parameters.inertia / nominal_parameters.mass
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "path": str(bag),
            "sha256": baseline._sha256(bag),
            "requested_interval_seconds": [arguments.start, arguments.end],
            "fitted_common_support_seconds": [
                float(selected_problem.direct_problem.output_time[0]),
                float(selected_problem.direct_problem.output_time[-1]),
            ],
            "sample_count": int(
                selected_problem.direct_problem.output_time.size
            ),
        },
        "method": {
            "name": "augmented_lagrangian_multiple_shooting_se3_only",
            "data_loss": (
                "least-squares residual metric for each "
                "Log_SE3(T_observed^{-1} T_simulated) sample: ||rho||^2 + "
                "phi^T (J0 / m0) phi, with translation pushed through "
                "inverse SO(3) left Jacobian"
            ),
            "observation_terms": ["sensor_position", "sensor_orientation"],
            "excluded_observation_terms": [
                "velocity",
                "angular_velocity",
                "specific_force",
                "acceleration",
            ],
            "continuity_is_constraint_not_observation_loss": True,
            "segment_duration_requested_seconds": arguments.segment_duration,
            "segment_boundaries_output_indices": selected_problem.boundaries,
            "segment_count": selected_problem.segment_count,
            "internal_node_count": selected_problem.node_count,
            "variable_dimension": selected_problem.variable_dimension,
            "physical_dimension": PHYSICAL_DIMENSION,
            "node_dimension": NODE_DIMENSION,
            "physical_parameter_names": PHYSICAL_PARAMETER_NAMES,
            "inertia_parameterization": (
                "Sigma = L L^T; J = trace(Sigma) I - Sigma"
            ),
            "inertia_positive_definite_by_construction": True,
            "inertia_triangle_inequalities_by_construction": True,
            "physical_coordinate_box_bounds": None,
            "thrust_time_constant_model": None,
            "gimbal_time_constant_model": None,
            "actuator_command_response": (
                "instantaneous thrust; instantaneous gimbal target subject "
                "only to the configured gimbal rate and angle limits"
            ),
            "soft_prior": {
                "distribution": "independent Gaussian in physical coordinates",
                "mean": "nominal coordinate (all zeros)",
                "standard_deviations": {
                    name: float(scale)
                    for name, scale in zip(
                        PHYSICAL_PARAMETER_NAMES,
                        BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS,
                    )
                },
                "weight": arguments.prior_weight,
            },
            "node_parameter_names": NODE_PARAMETER_NAMES,
            "physical_jacobian": "analytic forward sensitivity",
            "node_jacobian": "analytic forward sensitivity",
            "lag_treatment": "fixed causal ZOH command lookup",
            "delay_search": None,
        },
        "fixed_parameters": {
            "command_delay_seconds": command_delay,
            "torque_effectiveness": [1.0] * 4,
            "linear_drag": [0.0] * 3,
            "angular_drag": [0.0] * 3,
        },
        "loss_metric": {
            "per_sample_pose_quadratic": (
                "||rho||^2 + phi^T (J0 / m0) phi"
            ),
            "residual_reduction": "divide by sqrt(sample count)",
            "reported_cost_convention": (
                "one half of the mean per-sample pose quadratic"
            ),
            "nominal_mass_kg": nominal_parameters.mass,
            "nominal_inertia_kg_m2": nominal_parameters.inertia,
            "rotation_metric_J0_over_m0_m2": nominal_rotational_metric,
            "se3_residual_factor": selected_problem.pose_residual_factor,
            "prior_weight": arguments.prior_weight,
            "soft_prior_standard_deviations": {
                name: float(scale)
                for name, scale in zip(
                    PHYSICAL_PARAMETER_NAMES,
                    BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS,
                )
            },
        },
        "delay_profile_seconds": [command_delay],
        "delay_results": [
            _solution_payload(
                problem,
                solution,
                arguments.continuity_tolerance,
            )
            for problem, solution in ordered
        ],
        "selection": selected_payload,
        "nominal_at_selected_delay": {
            "full_rollout_inertia_radius_loss_m2": 0.5
            * float(nominal_residual @ nominal_residual),
            "metrics": _pose_metrics(
                _unscaled_pose_residual(
                    nominal_problem,
                    nominal_position,
                    nominal_orientation,
                )
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "result_json": "result.json",
            "trajectory_pdf": "trajectory.pdf",
            "delay_profile_pdf": "delay_profile.pdf",
            "parameters_text": "parameters.txt",
        },
    }
    baseline._write_json(result_path, payload)
    _write_text(parameters_path, parameter_lines)
    _write_pdf(
        pdf_path,
        selected_problem,
        selected_solution,
        stitched_metrics,
        parameter_lines,
        arguments.continuity_tolerance,
    )
    _write_delay_profile_pdf(
        delay_profile_path,
        ordered,
        selected_solution.delay,
        arguments.continuity_tolerance,
    )
    print(
        "fixed delay {:.6f}s, full-rollout inertia-radius loss {:.9g}".format(
            selected_solution.delay,
            selected_payload["full_rollout_inertia_radius_loss_m2"],
        ),
        flush=True,
    )
    print("wrote {}".format(result_path), flush=True)
    print("wrote {}".format(pdf_path), flush=True)
    print("wrote {}".format(delay_profile_path), flush=True)
    print("wrote {}".format(parameters_path), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
