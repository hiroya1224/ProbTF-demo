"""Unwhitened analytic rigid-body wrench-balance residuals.

This module deliberately stops at the physical six-vector expressed in N and
Nm.  The statistical model for that vector (including Q and variable-dt
whitening) belongs to a separate boundary so changing its units cannot alter
the rigid-body or actuator derivatives implemented here.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.dynamics import actuator_wrench_with_jacobian
from grape_param_estim.geometry import (
    skew,
    so3_geodesic_midpoint_with_right_jacobians,
    so3_log,
)
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.system import (
    GRAVITY,
    ActuatorState,
    GrapeGeometry,
)


_MIDPOINT_BRANCH_WARNING_DISTANCE = 1.0e-5


def _immutable_array(value: np.ndarray, shape: Tuple[int, ...], name: str):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result


def _proper_rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must be orthonormal".format(name))
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must have determinant one".format(name))
    return result


@dataclass(frozen=True)
class DynamicsResidualJacobian:
    """Analytic blocks of one raw six-dimensional dynamics residual.

    Rotation columns are right-tangent endpoint perturbations.  Static
    parameter columns use the canonical 18-dimensional
    :class:`VehicleParameterChart` coordinates.
    """

    rotation_left: np.ndarray
    rotation_right: np.ndarray
    linear_velocity_left: np.ndarray
    linear_velocity_right: np.ndarray
    angular_velocity_left: np.ndarray
    angular_velocity_right: np.ndarray
    actuator_thrust_left: np.ndarray
    actuator_thrust_right: np.ndarray
    gimbal_angle_left: np.ndarray
    gimbal_angle_right: np.ndarray
    static_parameters: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("rotation_left", (6, 3)),
            ("rotation_right", (6, 3)),
            ("linear_velocity_left", (6, 3)),
            ("linear_velocity_right", (6, 3)),
            ("angular_velocity_left", (6, 3)),
            ("angular_velocity_right", (6, 3)),
            ("actuator_thrust_left", (6, 4)),
            ("actuator_thrust_right", (6, 4)),
            ("gimbal_angle_left", (6, 4)),
            ("gimbal_angle_right", (6, 4)),
            ("static_parameters", (6, PARAMETER_DIMENSION)),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape, name),
            )


@dataclass(frozen=True)
class DynamicsResidualEvaluation:
    """One physical wrench-balance residual and its analytic linearization."""

    residual: np.ndarray
    required_wrench: np.ndarray
    modeled_wrench: np.ndarray
    jacobian: DynamicsResidualJacobian
    branch_diagnostics: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        for name in ("residual", "required_wrench", "modeled_wrench"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), (6,), name),
            )
        if not isinstance(self.jacobian, DynamicsResidualJacobian):
            raise TypeError("jacobian must be a DynamicsResidualJacobian")
        copied = {}
        for name, diagnostic in self.branch_diagnostics.items():
            if type(name) is not str or not name:
                raise ValueError(
                    "branch diagnostic names must be non-empty strings"
                )
            value = np.asarray(diagnostic)
            if value.dtype != np.bool_ or value.ndim != 1:
                raise ValueError(
                    "branch diagnostics must be one-dimensional boolean arrays"
                )
            immutable = value.copy()
            immutable.setflags(write=False)
            copied[name] = immutable
        object.__setattr__(
            self, "branch_diagnostics", MappingProxyType(copied)
        )


def evaluate_raw_dynamics_residual(
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    linear_velocity_left: Sequence[float],
    linear_velocity_right: Sequence[float],
    angular_velocity_left: Sequence[float],
    angular_velocity_right: Sequence[float],
    actuator_thrust_left: Sequence[float],
    actuator_thrust_right: Sequence[float],
    gimbal_angle_left: Sequence[float],
    gimbal_angle_right: Sequence[float],
    time_step: float,
    parameter_chart: VehicleParameterChart,
    parameter_coordinates: Sequence[float],
    geometry: GrapeGeometry,
    gravity_world: Optional[Sequence[float]] = None,
) -> DynamicsResidualEvaluation:
    """Evaluate the unwhitened interval-average dynamics residual.

    The result is ``required - modeled`` in body coordinates, with force rows
    in N and torque rows in Nm.  Endpoint thrust and gimbal states are
    averaged arithmetically; the attitude midpoint follows the principal SO(3)
    geodesic.  Linear and angular drag are the fixed values carried by the
    parameter chart's decoded :class:`VehicleParameters`.
    """

    rotation0 = _proper_rotation(rotation_left, "rotation_left")
    rotation1 = _proper_rotation(rotation_right, "rotation_right")
    velocity0 = _finite_vector(
        linear_velocity_left, 3, "linear_velocity_left"
    )
    velocity1 = _finite_vector(
        linear_velocity_right, 3, "linear_velocity_right"
    )
    omega0 = _finite_vector(
        angular_velocity_left, 3, "angular_velocity_left"
    )
    omega1 = _finite_vector(
        angular_velocity_right, 3, "angular_velocity_right"
    )
    thrust0 = _finite_vector(
        actuator_thrust_left, 4, "actuator_thrust_left"
    )
    thrust1 = _finite_vector(
        actuator_thrust_right, 4, "actuator_thrust_right"
    )
    gimbal0 = _finite_vector(gimbal_angle_left, 4, "gimbal_angle_left")
    gimbal1 = _finite_vector(gimbal_angle_right, 4, "gimbal_angle_right")
    dt = float(time_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_step must be finite and positive")
    if not isinstance(parameter_chart, VehicleParameterChart):
        raise TypeError("parameter_chart must be a VehicleParameterChart")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be a GrapeGeometry")
    gravity = _finite_vector(
        (0.0, 0.0, -GRAVITY) if gravity_world is None else gravity_world,
        3,
        "gravity_world",
    )

    parameters, parameter_jacobian = parameter_chart.decode_with_jacobian(
        parameter_coordinates
    )
    midpoint_rotation, midpoint_rotation_left, midpoint_rotation_right = (
        so3_geodesic_midpoint_with_right_jacobians(rotation0, rotation1)
    )
    midpoint_velocity = 0.5 * (velocity0 + velocity1)
    midpoint_omega = 0.5 * (omega0 + omega1)
    midpoint_actuator = ActuatorState(
        thrust=0.5 * (thrust0 + thrust1),
        gimbal_angle=0.5 * (gimbal0 + gimbal1),
    )

    world_acceleration_minus_gravity = (
        (velocity1 - velocity0) / dt - gravity
    )
    body_acceleration_minus_gravity = (
        midpoint_rotation.T @ world_acceleration_minus_gravity
    )
    body_velocity = midpoint_rotation.T @ midpoint_velocity
    angular_acceleration = (omega1 - omega0) / dt

    required_force = parameters.mass * body_acceleration_minus_gravity
    inertia_omega = parameters.inertia @ midpoint_omega
    required_torque = (
        parameters.inertia @ angular_acceleration
        + np.cross(midpoint_omega, inertia_omega)
    )
    required_wrench = np.concatenate((required_force, required_torque))

    actuator_wrench, actuator_jacobian = actuator_wrench_with_jacobian(
        midpoint_actuator,
        parameters,
        geometry,
    )
    modeled_wrench = actuator_wrench.copy()
    modeled_wrench[:3] -= parameters.linear_drag * body_velocity
    modeled_wrench[3:] -= parameters.angular_drag * midpoint_omega
    residual = required_wrench - modeled_wrench

    linear_drag = np.diag(parameters.linear_drag)
    angular_drag = np.diag(parameters.angular_drag)
    midpoint_rotation_jacobian = np.zeros((6, 3), dtype=float)
    midpoint_rotation_jacobian[:3, :] = (
        parameters.mass * skew(body_acceleration_minus_gravity)
        + linear_drag @ skew(body_velocity)
    )

    velocity_left_jacobian = np.zeros((6, 3), dtype=float)
    velocity_right_jacobian = np.zeros((6, 3), dtype=float)
    velocity_left_jacobian[:3, :] = (
        -parameters.mass * midpoint_rotation.T / dt
        + 0.5 * linear_drag @ midpoint_rotation.T
    )
    velocity_right_jacobian[:3, :] = (
        parameters.mass * midpoint_rotation.T / dt
        + 0.5 * linear_drag @ midpoint_rotation.T
    )

    gyroscopic_jacobian = (
        skew(midpoint_omega) @ parameters.inertia
        - skew(inertia_omega)
    )
    omega_midpoint_jacobian = gyroscopic_jacobian + angular_drag
    omega_left_jacobian = np.zeros((6, 3), dtype=float)
    omega_right_jacobian = np.zeros((6, 3), dtype=float)
    omega_left_jacobian[3:, :] = (
        -parameters.inertia / dt + 0.5 * omega_midpoint_jacobian
    )
    omega_right_jacobian[3:, :] = (
        parameters.inertia / dt + 0.5 * omega_midpoint_jacobian
    )

    static_parameters_jacobian = np.zeros(
        (6, PARAMETER_DIMENSION), dtype=float
    )
    static_parameters_jacobian[:3, :] = np.outer(
        body_acceleration_minus_gravity,
        parameter_jacobian.mass,
    )
    for coordinate in range(PARAMETER_DIMENSION):
        inertia_derivative = parameter_jacobian.inertia[:, :, coordinate]
        static_parameters_jacobian[3:, coordinate] = (
            inertia_derivative @ angular_acceleration
            + np.cross(
                midpoint_omega,
                inertia_derivative @ midpoint_omega,
            )
        )
    static_parameters_jacobian -= (
        actuator_jacobian.cog_offset @ parameter_jacobian.cog_offset
        + actuator_jacobian.force_effectiveness
        @ parameter_jacobian.force_effectiveness
        + actuator_jacobian.torque_effectiveness
        @ parameter_jacobian.torque_effectiveness
    )

    jacobian = DynamicsResidualJacobian(
        rotation_left=(
            midpoint_rotation_jacobian @ midpoint_rotation_left
        ),
        rotation_right=(
            midpoint_rotation_jacobian @ midpoint_rotation_right
        ),
        linear_velocity_left=velocity_left_jacobian,
        linear_velocity_right=velocity_right_jacobian,
        angular_velocity_left=omega_left_jacobian,
        angular_velocity_right=omega_right_jacobian,
        actuator_thrust_left=-0.5 * actuator_jacobian.actual_thrust,
        actuator_thrust_right=-0.5 * actuator_jacobian.actual_thrust,
        gimbal_angle_left=-0.5 * actuator_jacobian.actual_gimbal_angle,
        gimbal_angle_right=-0.5 * actuator_jacobian.actual_gimbal_angle,
        static_parameters=static_parameters_jacobian,
    )
    relative_angle = float(np.linalg.norm(so3_log(rotation0.T @ rotation1)))
    return DynamicsResidualEvaluation(
        residual=residual,
        required_wrench=required_wrench,
        modeled_wrench=modeled_wrench,
        jacobian=jacobian,
        branch_diagnostics={
            "rotation_midpoint_log_near_pi": np.asarray(
                (
                    np.pi - relative_angle
                    <= _MIDPOINT_BRANCH_WARNING_DISTANCE,
                ),
                dtype=bool,
            )
        },
    )


__all__ = [
    "DynamicsResidualEvaluation",
    "DynamicsResidualJacobian",
    "evaluate_raw_dynamics_residual",
]
