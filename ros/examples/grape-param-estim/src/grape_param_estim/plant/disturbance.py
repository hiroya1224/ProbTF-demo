"""Low-dimensional episode disturbance models."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstantDisturbance:
    force_world: np.ndarray
    torque_body: np.ndarray
    model_id: str = "constant_wrench_disturbance_v1"

    def __post_init__(self) -> None:
        for name in ("force_world", "torque_body"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError("{} must be a finite three-vector".format(name))
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    @classmethod
    def from_parameters(cls, values: Any) -> "ConstantDisturbance":
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("disturbance parameters must be [force,torque]")
        return cls(vector[:3], vector[3:])


@dataclass(frozen=True)
class EffectiveAccelerationDisturbance:
    """Constant nuisance in the effective model's calibrated coordinates."""

    linear_acceleration_world: np.ndarray
    angular_acceleration_body: np.ndarray
    model_id: str = "effective_constant_acceleration_disturbance_v1"

    def __post_init__(self) -> None:
        for name in (
            "linear_acceleration_world",
            "angular_acceleration_body",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "{} must be a finite three-vector".format(name)
                )
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    @classmethod
    def from_parameters(
        cls, values: Any
    ) -> "EffectiveAccelerationDisturbance":
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "effective disturbance parameters must be "
                "[linear_acceleration_world,angular_acceleration_body]"
            )
        return cls(vector[:3], vector[3:])


__all__ = [
    "ConstantDisturbance",
    "EffectiveAccelerationDisturbance",
]
