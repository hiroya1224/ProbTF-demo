"""Forward rigid-body and effective command-response plant integration."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.dynamics import parameters_to_inertia
from grape_param_estim.plant.actuator import RealizedWrench
from grape_param_estim.plant.disturbance import (
    ConstantDisturbance,
    EffectiveAccelerationDisturbance,
)
from grape_param_estim.plant.parameters import (
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    PlantParameters,
)


def _readonly(values: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape {}".format(name, shape))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class PlantState:
    stamp: float
    position_world: np.ndarray
    velocity_world: np.ndarray
    orientation_xyzw: np.ndarray
    angular_velocity_body: np.ndarray
    linear_acceleration_world: np.ndarray = None
    angular_acceleration_body: np.ndarray = None

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("plant state stamp must be finite")
        quaternion = _readonly(
            self.orientation_xyzw, (4,), "orientation_xyzw"
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("orientation quaternion must be nonzero")
        quaternion = np.array(quaternion / norm, copy=True)
        quaternion.setflags(write=False)
        linear_acceleration = (
            np.zeros(3)
            if self.linear_acceleration_world is None
            else self.linear_acceleration_world
        )
        angular_acceleration = (
            np.zeros(3)
            if self.angular_acceleration_body is None
            else self.angular_acceleration_body
        )
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(
            self,
            "position_world",
            _readonly(self.position_world, (3,), "position_world"),
        )
        object.__setattr__(
            self,
            "velocity_world",
            _readonly(self.velocity_world, (3,), "velocity_world"),
        )
        object.__setattr__(self, "orientation_xyzw", quaternion)
        object.__setattr__(
            self,
            "angular_velocity_body",
            _readonly(
                self.angular_velocity_body, (3,), "angular_velocity_body"
            ),
        )
        object.__setattr__(
            self,
            "linear_acceleration_world",
            _readonly(
                linear_acceleration, (3,), "linear_acceleration_world"
            ),
        )
        object.__setattr__(
            self,
            "angular_acceleration_body",
            _readonly(
                angular_acceleration, (3,), "angular_acceleration_body"
            ),
        )

    @classmethod
    def from_vector(cls, values: Any, stamp: float = 0.0) -> "PlantState":
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (13,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "plant state vector must be [position,velocity,xyzw,omega]"
            )
        return cls(
            stamp=stamp,
            position_world=vector[:3],
            velocity_world=vector[3:6],
            orientation_xyzw=vector[6:10],
            angular_velocity_body=vector[10:13],
        )

    @property
    def vector(self) -> np.ndarray:
        result = np.concatenate(
            (
                self.position_world,
                self.velocity_world,
                self.orientation_xyzw,
                self.angular_velocity_body,
            )
        )
        result.setflags(write=False)
        return result


@runtime_checkable
class PlantBackend(Protocol):
    """Structural interface implemented by candidate plant simulators."""

    model_id: str

    def reset(self, initial_state: np.ndarray) -> None:
        ...

    def step(
        self,
        wrench: RealizedWrench,
        parameters: PlantParameters,
        dt: float,
    ) -> PlantState:
        ...


class RigidBodyPlantBackend:
    """Calibrated six-DoF rigid body with explicit body-wrench input."""

    model_id = "calibrated_rigid_body_forward_v1"

    def __init__(
        self,
        gravity_m_s2: float = 9.80665,
        linear_drag: float = 0.0,
        angular_drag: float = 0.0,
    ) -> None:
        gravity = float(gravity_m_s2)
        linear = float(linear_drag)
        angular = float(angular_drag)
        if (
            not np.isfinite(gravity)
            or gravity < 0.0
            or not np.isfinite(linear)
            or linear < 0.0
            or not np.isfinite(angular)
            or angular < 0.0
        ):
            raise ValueError("gravity and drag must be finite and non-negative")
        self.gravity_m_s2 = gravity
        self.linear_drag = linear
        self.angular_drag = angular
        self._state: Optional[PlantState] = None
        self._disturbance = ConstantDisturbance(
            np.zeros(3), np.zeros(3)
        )

    def set_disturbance(
        self, disturbance: ConstantDisturbance
    ) -> None:
        if not isinstance(disturbance, ConstantDisturbance):
            raise TypeError(
                "calibrated rigid-body disturbance must be "
                "ConstantDisturbance"
            )
        self._disturbance = disturbance

    def reset(self, initial_state: np.ndarray, stamp: float = 0.0) -> None:
        self._state = PlantState.from_vector(initial_state, stamp=stamp)

    @property
    def state(self) -> PlantState:
        if self._state is None:
            raise RuntimeError("plant backend has not been reset")
        return self._state

    def step(
        self,
        wrench: RealizedWrench,
        parameters: PlantParameters,
        dt: float,
    ) -> PlantState:
        if not isinstance(wrench, RealizedWrench):
            raise TypeError("wrench must be RealizedWrench")
        if not isinstance(parameters, PlantParameters):
            raise TypeError("parameters must be PlantParameters")
        if not parameters.calibrated_physical or not wrench.calibrated_wrench:
            raise ValueError(
                "physical rigid-body integration requires calibrated actuator wrench"
            )
        delta = float(dt)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("plant dt must be finite and positive")
        state = self.state
        mass = float(parameters.values[0])
        cog = parameters.values[1:4]
        inertia = parameters_to_inertia(parameters.values)
        rotation = Rotation.from_quat(state.orientation_xyzw)
        gravity = np.array([0.0, 0.0, -self.gravity_m_s2])
        force_world = rotation.apply(wrench.force_body)
        acceleration = (
            force_world / mass
            + gravity
            - self.linear_drag * state.velocity_world
            + self._disturbance.force_world / mass
        )
        torque_com = (
            wrench.torque_body
            + self._disturbance.torque_body
            - np.cross(cog, wrench.force_body)
        )
        omega = state.angular_velocity_body
        angular_acceleration = np.linalg.solve(
            inertia,
            torque_com
            - np.cross(omega, inertia @ omega)
            - self.angular_drag * omega,
        )
        new_position = (
            state.position_world
            + state.velocity_world * delta
            + 0.5 * acceleration * delta * delta
        )
        new_velocity = state.velocity_world + acceleration * delta
        omega_mid = omega + 0.5 * angular_acceleration * delta
        new_orientation = (
            rotation * Rotation.from_rotvec(omega_mid * delta)
        ).as_quat()
        new_omega = omega + angular_acceleration * delta
        self._state = PlantState(
            stamp=float(wrench.stamp),
            position_world=new_position,
            velocity_world=new_velocity,
            orientation_xyzw=new_orientation,
            angular_velocity_body=new_omega,
            linear_acceleration_world=acceleration,
            angular_acceleration_body=angular_acceleration,
        )
        return self._state


class EffectiveRigidBodyPlantBackend:
    """Uncalibrated command-to-motion model for current Grape bags."""

    model_id = "effective_closed_loop_forward_v1"

    def __init__(self, gravity_m_s2: float = 9.80665) -> None:
        gravity = float(gravity_m_s2)
        if not np.isfinite(gravity) or gravity < 0.0:
            raise ValueError("gravity must be finite and non-negative")
        self.gravity_m_s2 = gravity
        self._state: Optional[PlantState] = None
        self._disturbance = EffectiveAccelerationDisturbance(
            np.zeros(3), np.zeros(3)
        )

    def set_disturbance(
        self, disturbance: EffectiveAccelerationDisturbance
    ) -> None:
        if not isinstance(
            disturbance, EffectiveAccelerationDisturbance
        ):
            raise TypeError(
                "effective disturbance must be "
                "EffectiveAccelerationDisturbance"
            )
        self._disturbance = disturbance

    def reset(self, initial_state: np.ndarray, stamp: float = 0.0) -> None:
        self._state = PlantState.from_vector(initial_state, stamp=stamp)

    @property
    def state(self) -> PlantState:
        if self._state is None:
            raise RuntimeError("plant backend has not been reset")
        return self._state

    def step(
        self,
        wrench: RealizedWrench,
        parameters: PlantParameters,
        dt: float,
    ) -> PlantState:
        if not isinstance(wrench, RealizedWrench):
            raise TypeError("wrench must be RealizedWrench")
        if (
            not isinstance(parameters, PlantParameters)
            or parameters.model_id != EFFECTIVE_CLOSED_LOOP_MODEL_ID
        ):
            raise TypeError(
                "effective backend requires effective_closed_loop_v1 parameters"
            )
        delta = float(dt)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("plant dt must be finite and positive")
        values = parameters.values
        state = self.state
        rotation = Rotation.from_quat(state.orientation_xyzw)
        gravity = np.array([0.0, 0.0, -self.gravity_m_s2])
        force_specific_body = values[0] * wrench.force_body
        force_specific_body[2] += values[8]
        acceleration = (
            rotation.apply(force_specific_body)
            + gravity
            - values[6] * state.velocity_world
            + self._disturbance.linear_acceleration_world
        )
        coupling = np.cross(
            np.array([values[4], values[5], 0.0]), wrench.force_body
        )
        torque = wrench.torque_body + coupling
        angular_acceleration = values[1:4] * torque
        angular_acceleration[:2] += values[9:11]
        angular_acceleration -= values[7] * state.angular_velocity_body
        angular_acceleration += (
            self._disturbance.angular_acceleration_body
        )

        new_position = (
            state.position_world
            + state.velocity_world * delta
            + 0.5 * acceleration * delta * delta
        )
        new_velocity = state.velocity_world + acceleration * delta
        omega_mid = (
            state.angular_velocity_body
            + 0.5 * angular_acceleration * delta
        )
        new_orientation = (
            rotation * Rotation.from_rotvec(omega_mid * delta)
        ).as_quat()
        new_omega = state.angular_velocity_body + angular_acceleration * delta
        self._state = PlantState(
            stamp=float(wrench.stamp),
            position_world=new_position,
            velocity_world=new_velocity,
            orientation_xyzw=new_orientation,
            angular_velocity_body=new_omega,
            linear_acceleration_world=acceleration,
            angular_acceleration_body=angular_acceleration,
        )
        return self._state


__all__ = [
    "EffectiveRigidBodyPlantBackend",
    "PlantBackend",
    "PlantState",
    "RigidBodyPlantBackend",
]
