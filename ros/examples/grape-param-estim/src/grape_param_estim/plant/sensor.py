"""Observation predictions from forward plant state."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.plant.rigid_body import PlantState


def _readonly(values: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape {}".format(name, shape))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class PredictedObservation:
    stamp: float
    position_world: np.ndarray
    orientation_xyzw: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_body: np.ndarray
    specific_force_body: np.ndarray
    model_id: str

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("observation stamp must be finite")
        object.__setattr__(self, "stamp", stamp)
        for name, shape in (
            ("position_world", (3,)),
            ("orientation_xyzw", (4,)),
            ("velocity_world", (3,)),
            ("angular_velocity_body", (3,)),
            ("specific_force_body", (3,)),
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), shape, name)
            )


@runtime_checkable
class ObservationBackend(Protocol):
    """Structural interface implemented by plant observation models."""

    model_id: str

    def predict(
        self,
        state: PlantState,
        sensor_bias: Optional[np.ndarray] = None,
    ) -> PredictedObservation:
        ...


class RigidBodyObservationBackend:
    model_id = "rigid_body_pose_velocity_imu_observation_v1"

    def __init__(self, gravity_m_s2: float = 9.80665) -> None:
        gravity = float(gravity_m_s2)
        if not np.isfinite(gravity) or gravity < 0.0:
            raise ValueError("gravity must be finite and non-negative")
        self.gravity_m_s2 = gravity

    def predict(
        self, state: PlantState, sensor_bias: Optional[np.ndarray] = None
    ) -> PredictedObservation:
        if not isinstance(state, PlantState):
            raise TypeError("state must be PlantState")
        bias = (
            np.zeros(6)
            if sensor_bias is None
            else np.asarray(sensor_bias, dtype=float).reshape(-1)
        )
        if bias.shape != (6,) or not np.all(np.isfinite(bias)):
            raise ValueError("sensor_bias must contain accel and gyro biases")
        gravity = np.array([0.0, 0.0, -self.gravity_m_s2])
        specific = Rotation.from_quat(state.orientation_xyzw).inv().apply(
            state.linear_acceleration_world - gravity
        )
        return PredictedObservation(
            stamp=state.stamp,
            position_world=state.position_world,
            orientation_xyzw=state.orientation_xyzw,
            velocity_world=state.velocity_world,
            angular_velocity_body=state.angular_velocity_body + bias[3:],
            specific_force_body=specific + bias[:3],
            model_id=self.model_id,
        )


__all__ = [
    "ObservationBackend",
    "PredictedObservation",
    "RigidBodyObservationBackend",
]
