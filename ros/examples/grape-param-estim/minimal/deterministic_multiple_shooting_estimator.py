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
Cholesky inertia chart, the three-dimensional CoG offset, and three relative
rotor-force-effectiveness contrasts.  Recorded-command lag is profiled
outside the smooth solve because causal zero-order-hold lookup is not smooth
in lag.
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

import deterministic_continuation_estimator as continuation
import deterministic_estimator as baseline
import deterministic_sobol_estimator as analytic
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
    ActuatorState,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/minimal-deterministic-multiple-shooting/v1"
OUTPUT_SUBDIRECTORY = "deterministic_multiple_shooting"
PHYSICAL_DIMENSION = 13
NODE_DIMENSION = 20
PHYSICAL_PARAMETER_NAMES = analytic.SEARCH_PARAMETER_NAMES[:PHYSICAL_DIMENSION]
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
    parameterization: analytic.PhysicalSearchParameterization,
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
    if dimension < PHYSICAL_DIMENSION:
        raise ValueError("extended derivative dimension is too small")

    def extend_vector(value: np.ndarray) -> np.ndarray:
        result = np.zeros(dimension, dtype=float)
        result[:PHYSICAL_DIMENSION] = value
        return result

    def extend_matrix(value: np.ndarray) -> np.ndarray:
        result = np.zeros((value.shape[0], dimension), dtype=float)
        result[:, :PHYSICAL_DIMENSION] = value
        return result

    inertia = np.zeros((3, 3, dimension), dtype=float)
    inertia[:, :, :PHYSICAL_DIMENSION] = source.inertia
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
) -> tuple[ActuatorState, np.ndarray]:
    value = np.asarray(sensitivity, dtype=float)
    dimension = value.shape[1]
    if value.shape[0] != 8:
        raise ValueError("actuator sensitivity must have eight rows")
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
        translation_scale: float,
        rotation_scale: float,
        prior_weight: float,
        node_position_bound: float,
        node_orientation_bound: float,
        node_velocity_bound: float,
        node_angular_velocity_bound: float,
    ) -> None:
        self.direct_problem = direct_problem
        self.delay = float(delay)
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.prior_weight = float(prior_weight)
        self.parameterization = analytic.PhysicalSearchParameterization(
            VehicleParameters.nominal()
        )
        self.boundaries = segment_boundaries(
            direct_problem.output_time.size,
            float(direct_problem.output_time[1] - direct_problem.output_time[0]),
            segment_duration,
        )
        self.segment_count = self.boundaries.size - 1
        self.node_count = self.segment_count - 1
        self.variable_dimension = (
            PHYSICAL_DIMENSION + self.node_count * NODE_DIMENSION
        )
        self.pose_residual_dimension = direct_problem.output_time.size * 6
        self.data_residual_dimension = (
            self.pose_residual_dimension + PHYSICAL_DIMENSION
        )
        self.continuity_dimension = self.node_count * NODE_DIMENSION
        self.prior_scales = np.asarray(
            (
                0.50,
                0.80,
                0.80,
                0.80,
                0.40,
                0.40,
                0.40,
                0.05,
                0.05,
                0.05,
                0.25,
                0.25,
                0.25,
            ),
            dtype=float,
        )
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
        nominal_parameters = VehicleParameters.nominal()
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

    def split_coordinate(
        self, coordinate: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (self.variable_dimension,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError("multiple-shooting coordinate has the wrong shape")
        physical = value[:PHYSICAL_DIMENSION]
        nodes = value[PHYSICAL_DIMENSION:].reshape(
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
        result[:PHYSICAL_DIMENSION] = previous_physical
        for index in range(self.node_count):
            rigid, actuator, _, _ = _decode_node(
                previous.node_references[index],
                previous_nodes[index],
            )
            result[
                PHYSICAL_DIMENSION
                + index * NODE_DIMENSION : PHYSICAL_DIMENSION
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
        local_dimension = PHYSICAL_DIMENSION + (
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
            rigid_sensitivity[:3, :PHYSICAL_DIMENSION] = (
                self.direct_problem.initial_body_rotation
                @ physical_jacobian.cog_offset
            )
            rigid_sensitivity[7:10, :PHYSICAL_DIMENSION] = (
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
            rigid_sensitivity[:, PHYSICAL_DIMENSION:] = node_rigid
            actuator_sensitivity[:, PHYSICAL_DIMENSION:] = node_actuator
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
            pose_residual[local_index] = residual
            pose_jacobian[local_index] = jacobian

        store(0, output_start)
        local_output_index = 1
        for step_index in range(internal_start, internal_end):
            command = self.direct_problem.commands[step_index]
            time_step = self.direct_problem.integration_step
            midpoint_actuator, midpoint_sensitivity = (
                _actuator_step_with_sensitivity(
                    actuator,
                    actuator_sensitivity,
                    command,
                    decoded,
                    local_parameter_jacobian,
                    0.5 * time_step,
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
        local_columns = [np.arange(PHYSICAL_DIMENSION, dtype=int)]
        if segment_index > 0:
            start = PHYSICAL_DIMENSION + (segment_index - 1) * NODE_DIMENSION
            local_columns.append(np.arange(start, start + NODE_DIMENSION))
        local_columns_array = np.concatenate(local_columns)
        next_start = PHYSICAL_DIMENSION + segment_index * NODE_DIMENSION
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
        physical, nodes = self.split_coordinate(coordinate)
        decoded, physical_jacobian = _physical_parameter_jacobian(
            self.parameterization,
            physical,
            self.delay,
        )
        if decoded.inertia_triangle_margin <= 0.0:
            raise ValueError(
                "inertia principal moments violate the triangle inequality"
            )
        segment_evaluations: list[SegmentEvaluation] = []
        sensor_position = np.empty(
            (self.direct_problem.output_time.size, 3), dtype=float
        )
        sensor_orientation = np.empty(
            (self.direct_problem.output_time.size, 4), dtype=float
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
            local_columns = [np.arange(PHYSICAL_DIMENSION, dtype=int)]
            if segment_index > 0:
                start = (
                    PHYSICAL_DIMENSION
                    + (segment_index - 1) * NODE_DIMENSION
                )
                local_columns.append(np.arange(start, start + NODE_DIMENSION))
            columns = np.concatenate(local_columns)
            local_slice = slice(None) if segment_index == self.segment_count - 1 else slice(None, -1)
            indices = segment.output_indices[local_slice]
            sensor_position[indices] = segment.sensor_position[local_slice]
            sensor_orientation[indices] = segment.sensor_orientation_xyzw[local_slice]
            pose_residual[indices] = segment.pose_residual[local_slice]
            pose_jacobian[np.ix_(indices, np.arange(6), columns)] = (
                segment.pose_jacobian[local_slice]
            )

        count_scale = math.sqrt(self.direct_problem.output_time.size)
        scale = np.asarray(
            (
                self.translation_scale,
                self.translation_scale,
                self.translation_scale,
                self.rotation_scale,
                self.rotation_scale,
                self.rotation_scale,
            ),
            dtype=float,
        )
        normalized_pose = pose_residual / scale[None, :] / count_scale
        normalized_pose_jacobian = (
            pose_jacobian / scale[None, :, None] / count_scale
        )
        prior_residual = (
            math.sqrt(self.prior_weight) * physical / self.prior_scales
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
            decoded=decoded,
        )

    def full_rollout(
        self,
        physical_coordinate: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        decoded, physical_jacobian = _physical_parameter_jacobian(
            self.parameterization,
            physical_coordinate,
            self.delay,
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
        scale = np.asarray(
            (
                self.translation_scale,
                self.translation_scale,
                self.translation_scale,
                self.rotation_scale,
                self.rotation_scale,
                self.rotation_scale,
            ),
            dtype=float,
        )
        residual = (
            segment.pose_residual
            / scale[None, :]
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
                problem.delay,
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
        delay=problem.delay,
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


def _write_pdf(
    path: Path,
    problem: MultipleShootingProblem,
    nominal_position: np.ndarray,
    nominal_orientation: np.ndarray,
    solution: FixedDelaySolution,
    continuity_tolerance: float,
) -> None:
    time_axis = problem.direct_problem.output_time
    observed_position = problem.direct_problem.observations.sensor_position
    observed_rpy = np.unwrap(
        np.asarray(
            [
                matrix_to_euler_xyz(rotation)
                for rotation in problem.direct_problem.observed_sensor_rotation
            ]
        ),
        axis=0,
    )

    def rpy_series(quaternions: np.ndarray) -> np.ndarray:
        return np.unwrap(
            np.asarray(
                [
                    matrix_to_euler_xyz(quaternion_to_matrix(value))
                    for value in quaternions
                ]
            ),
            axis=0,
        )

    nominal_rpy = rpy_series(nominal_orientation)
    stitched_rpy = rpy_series(solution.evaluation.sensor_orientation_xyzw)
    full_rpy = rpy_series(solution.full_rollout_orientation_xyzw)
    labels = ("x", "y", "z")
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True)
        for axis, name, index in zip(axes, labels, range(3)):
            axis.plot(time_axis, observed_position[:, index], label="observed")
            axis.plot(time_axis, nominal_position[:, index], "--", label="nominal")
            axis.plot(
                time_axis,
                solution.evaluation.sensor_position[:, index],
                ":",
                label="multiple-shooting stitched",
            )
            axis.plot(
                time_axis,
                solution.full_rollout_position[:, index],
                "-.",
                label="selected full rollout",
            )
            axis.set_ylabel("{} [m]".format(name))
            axis.grid(True)
        axes[-1].set_xlabel("time [s]")
        axes[0].legend(loc="best")
        figure.suptitle("SE(3)-only multiple shooting: position")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True)
        for axis, name, index in zip(axes, ("roll", "pitch", "yaw"), range(3)):
            axis.plot(time_axis, observed_rpy[:, index], label="observed")
            axis.plot(time_axis, nominal_rpy[:, index], "--", label="nominal")
            axis.plot(time_axis, stitched_rpy[:, index], ":", label="stitched")
            axis.plot(time_axis, full_rpy[:, index], "-.", label="full rollout")
            axis.set_ylabel("{} [rad]".format(name))
            axis.grid(True)
        axes[-1].set_xlabel("time [s]")
        axes[0].legend(loc="best")
        figure.suptitle("SE(3)-only multiple shooting: orientation")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        continuity = solution.evaluation.continuity_residual.reshape(-1, NODE_DIMENSION)
        figure, axis = plt.subplots(figsize=(11.0, 5.5))
        if continuity.size:
            axis.semilogy(
                np.arange(continuity.shape[0]),
                np.max(np.abs(continuity), axis=1),
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


def _delay_grid(arguments: argparse.Namespace) -> np.ndarray:
    if arguments.delay_values is not None:
        values = np.asarray(arguments.delay_values, dtype=float)
        if values.ndim != 1 or values.size < 1:
            raise ValueError("delay-values must contain at least one value")
        values = np.unique(np.round(values, 12))
    else:
        values = continuation.inclusive_delay_grid(
            float(arguments.delay_bounds[0]),
            float(arguments.delay_bounds[1]),
            float(arguments.delay_step),
            required=(float(arguments.nominal_delay),),
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("delay grid contains invalid values")
    return values


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit mass, Cholesky inertia, CoG, relative force effectiveness, "
            "and recorded-command lag by SE(3)-only multiple shooting."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument("--translation-scale", type=float, default=0.05)
    parser.add_argument("--rotation-scale", type=float, default=0.10)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument(
        "--augmented-lagrangian-iterations", type=int, default=4
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
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.02),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--delay-step", type=float, default=0.01)
    parser.add_argument("--delay-values", type=float, nargs="+", default=None)
    parser.add_argument("--nominal-delay", type=float, default=0.01)
    parser.add_argument(
        "--mass-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--inertia-cholesky-diagonal-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--inertia-cholesky-offdiagonal-bound", type=float, default=0.8
    )
    parser.add_argument("--cog-bound", type=float, default=0.10)
    parser.add_argument(
        "--force-effectiveness-contrast-bound", type=float, default=0.60
    )
    parser.add_argument("--node-position-bound", type=float, default=2.0)
    parser.add_argument("--node-orientation-bound", type=float, default=1.5)
    parser.add_argument("--node-velocity-bound", type=float, default=5.0)
    parser.add_argument(
        "--node-angular-velocity-bound", type=float, default=10.0
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def _validate_arguments(
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.segment_duration,
        arguments.translation_scale,
        arguments.rotation_scale,
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
        or arguments.continuity_penalty_growth <= 1.0
        or not 0.0 < arguments.penalty_reduction_target < 1.0
    ):
        raise SystemExit("multiple-shooting settings are invalid")
    try:
        physical_bounds = continuation._physical_bounds(arguments)
        delays = _delay_grid(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not np.any(
        np.isclose(delays, arguments.nominal_delay, rtol=0.0, atol=1.0e-12)
    ):
        raise SystemExit("nominal delay must be present in the delay grid")
    return physical_bounds[0], physical_bounds[1], delays


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
    return {
        "delay_seconds": solution.delay,
        "physical_coordinate": physical,
        "stitched_se3_loss": 0.5
        * float(
            solution.evaluation.data_residual[: problem.pose_residual_dimension]
            @ solution.evaluation.data_residual[: problem.pose_residual_dimension]
        ),
        "full_rollout_se3_loss": 0.5
        * float(solution.full_rollout_residual @ solution.full_rollout_residual),
        "stitched_metrics": _pose_metrics(stitched_unscaled),
        "full_rollout_metrics": _pose_metrics(full_unscaled),
        "continuity_max_normalized": continuity_max,
        "continuity_l2_normalized": float(np.linalg.norm(continuity)),
        "continuity_converged": bool(
            continuity_max <= continuity_tolerance
        ),
        "parameters": analytic._physical_payload(solution.evaluation.decoded),
        "optimizer_history": list(solution.optimizer_history),
        "elapsed_seconds": solution.elapsed_seconds,
    }


def run(arguments: argparse.Namespace) -> int:
    physical_lower, physical_upper, delays = _validate_arguments(arguments)
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
    delay_order = continuation.branch_order(delays, arguments.nominal_delay)
    solved: dict[float, tuple[MultipleShootingProblem, FixedDelaySolution]] = {}
    anchor_problem: Optional[MultipleShootingProblem] = None
    anchor_solution: Optional[FixedDelaySolution] = None
    for branch_index, branch in enumerate(delay_order):
        previous_problem = anchor_problem
        previous_solution = anchor_solution
        for local_index, delay in enumerate(branch):
            key = round(float(delay), 12)
            if key in solved:
                previous_problem, previous_solution = solved[key]
                if local_index == 0:
                    anchor_problem, anchor_solution = solved[key]
                continue
            direct_problem = baseline.DirectShootingProblem(
                flight=flight,
                sample_step=arguments.sample_step,
                integration_step=arguments.integration_step,
                command_delay=float(delay),
                prior_weight=arguments.prior_weight,
            )
            problem = MultipleShootingProblem(
                direct_problem=direct_problem,
                delay=float(delay),
                segment_duration=arguments.segment_duration,
                translation_scale=arguments.translation_scale,
                rotation_scale=arguments.rotation_scale,
                prior_weight=arguments.prior_weight,
                node_position_bound=arguments.node_position_bound,
                node_orientation_bound=arguments.node_orientation_bound,
                node_velocity_bound=arguments.node_velocity_bound,
                node_angular_velocity_bound=(
                    arguments.node_angular_velocity_bound
                ),
            )
            bounds = problem.bounds(physical_lower, physical_upper)
            if previous_problem is None or previous_solution is None:
                initial = problem.initial_coordinate()
            else:
                initial = problem.rebase_coordinate(
                    previous_problem,
                    previous_solution.coordinate,
                )
            print(
                "solving delay {:.6f}s ({:d} segments, {:d} variables)".format(
                    delay,
                    problem.segment_count,
                    problem.variable_dimension,
                ),
                flush=True,
            )
            solution = _solve_fixed_delay(
                problem,
                initial,
                bounds,
                arguments,
            )
            solved[key] = (problem, solution)
            previous_problem, previous_solution = problem, solution
            if branch_index == 0 and local_index == 0:
                anchor_problem, anchor_solution = problem, solution
    if not solved:
        raise RuntimeError("no delay candidate was solved")
    selected_problem, selected_solution = min(
        solved.values(),
        key=lambda item: 0.5
        * float(
            item[1].full_rollout_residual @ item[1].full_rollout_residual
        ),
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
    ordered = sorted(solved.values(), key=lambda item: item[1].delay)
    selected_payload = _solution_payload(
        selected_problem,
        selected_solution,
        arguments.continuity_tolerance,
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
                "Log_SE3(T_observed^{-1} T_simulated), with translation "
                "pushed through inverse SO(3) left Jacobian"
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
            "node_parameter_names": NODE_PARAMETER_NAMES,
            "physical_jacobian": "analytic forward sensitivity",
            "node_jacobian": "analytic forward sensitivity",
            "lag_treatment": "external profile over causal ZOH command lookup",
        },
        "fixed_parameters": {
            "thrust_time_constant_seconds": analytic.CURRENT_THRUST_TIME_CONSTANT,
            "gimbal_time_constant_seconds": analytic.CURRENT_GIMBAL_TIME_CONSTANT,
            "torque_effectiveness": [1.0] * 4,
            "linear_drag": [0.0] * 3,
            "angular_drag": [0.0] * 3,
        },
        "loss_scales": {
            "se3_translation_m": arguments.translation_scale,
            "se3_rotation_rad": arguments.rotation_scale,
            "prior_weight": arguments.prior_weight,
        },
        "delay_profile_seconds": delays,
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
            "full_rollout_se3_loss": 0.5
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
        },
    }
    baseline._write_json(result_path, payload)
    _write_pdf(
        pdf_path,
        selected_problem,
        nominal_position,
        nominal_orientation,
        selected_solution,
        arguments.continuity_tolerance,
    )
    print(
        "selected delay {:.6f}s, full-rollout SE(3) loss {:.9g}".format(
            selected_solution.delay,
            selected_payload["full_rollout_se3_loss"],
        ),
        flush=True,
    )
    print("wrote {}".format(result_path), flush=True)
    print("wrote {}".format(pdf_path), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
