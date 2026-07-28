"""Separated static plant/actuator parameters and episode nuisance state."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.dynamics import (
    PARAMETER_NAMES as INERTIAL_PARAMETER_NAMES,
    validate_physical_parameters,
)


EFFECTIVE_CLOSED_LOOP_MODEL_ID = "effective_closed_loop_v1"
CALIBRATED_RIGID_BODY_MODEL_ID = "calibrated_rigid_body_v1"
ARTICULATED_GIMBALROTOR_MODEL_ID = "articulated_gimbalrotor_v1"

# Current bags do not independently calibrate command-to-wrench conversion.
# This profile therefore reports command-to-motion combinations rather than
# pretending that raw mass and thrust scale are separately identified.
EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES: Tuple[str, ...] = (
    "specific_thrust_authority",
    "roll_authority",
    "pitch_authority",
    "yaw_authority",
    "cog_coupling_x",
    "cog_coupling_y",
    "linear_drag",
    "angular_drag",
    "force_bias_z",
    "torque_bias_x",
    "torque_bias_y",
)
CALIBRATED_RIGID_BODY_PARAMETER_NAMES: Tuple[str, ...] = tuple(
    INERTIAL_PARAMETER_NAMES
)
ACTUATOR_PARAMETER_NAMES: Tuple[str, ...] = (
    "common_thrust_scale",
    "motor_time_constant",
    "command_delay",
    "gimbal_time_constant",
    "gimbal_angle_bias",
    "minimum_thrust",
    "maximum_thrust",
)


def _readonly_vector(
    values: Any, name: str, expected_size: Optional[int] = None
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if (
        array.size == 0
        or (expected_size is not None and array.size != expected_size)
        or not np.all(np.isfinite(array))
    ):
        suffix = (
            ""
            if expected_size is None
            else " with {} entries".format(expected_size)
        )
        raise ValueError("{} must be a non-empty finite vector{}".format(name, suffix))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _readonly_optional_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must be finite".format(name))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _names(values: Sequence[str], count: int, name: str) -> Tuple[str, ...]:
    normalized = tuple(str(item) for item in values)
    if (
        len(normalized) != count
        or any(not item for item in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(
            "{} must contain {} unique non-empty names".format(name, count)
        )
    return normalized


def _frozen_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return MappingProxyType({str(key): value for key, value in values.items()})


@dataclass(frozen=True)
class PlantParameters:
    """Static candidate plant parameters, excluding actuator/controller data."""

    model_id: str
    values: np.ndarray
    names: Tuple[str, ...]
    calibrated_physical: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = str(self.model_id)
        if model_id not in (
            EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            CALIBRATED_RIGID_BODY_MODEL_ID,
            ARTICULATED_GIMBALROTOR_MODEL_ID,
        ):
            raise ValueError("unsupported plant model profile: {}".format(model_id))
        values = _readonly_vector(self.values, "plant parameter values")
        names = _names(self.names, values.size, "plant parameter names")
        calibrated = self.calibrated_physical
        if type(calibrated) is not bool:
            raise TypeError("calibrated_physical must be a built-in bool")
        if model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID:
            if names != EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES:
                raise ValueError(
                    "effective_closed_loop_v1 requires its declared identifiable "
                    "parameter order"
                )
            if calibrated:
                raise ValueError(
                    "effective_closed_loop_v1 cannot be labelled physically calibrated"
                )
            if (
                values[0] <= 0.0
                or np.any(values[1:4] <= 0.0)
                or np.any(values[6:8] < 0.0)
            ):
                raise ValueError(
                    "effective authorities must be positive and drag non-negative"
                )
        elif model_id == CALIBRATED_RIGID_BODY_MODEL_ID:
            if names != CALIBRATED_RIGID_BODY_PARAMETER_NAMES:
                raise ValueError(
                    "calibrated_rigid_body_v1 requires inertial parameter order"
                )
            validate_physical_parameters(values)
            if not calibrated:
                raise ValueError(
                    "calibrated_rigid_body_v1 requires an actuator calibration gate"
                )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "metadata"))

    @classmethod
    def effective(cls, values: Any, **metadata: Any) -> "PlantParameters":
        return cls(
            model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            values=values,
            names=EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
            calibrated_physical=False,
            metadata=metadata,
        )

    @classmethod
    def calibrated_rigid_body(
        cls, values: Any, **metadata: Any
    ) -> "PlantParameters":
        return cls(
            model_id=CALIBRATED_RIGID_BODY_MODEL_ID,
            values=values,
            names=CALIBRATED_RIGID_BODY_PARAMETER_NAMES,
            calibrated_physical=True,
            metadata=metadata,
        )

    def as_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(
            {name: float(value) for name, value in zip(self.names, self.values)}
        )


@dataclass(frozen=True)
class ActuatorParameters:
    """Command-to-wrench parameters, kept separate from the candidate plant."""

    model_id: str
    values: np.ndarray
    names: Tuple[str, ...] = ACTUATOR_PARAMETER_NAMES
    calibrated_wrench: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = str(self.model_id)
        if not model_id:
            raise ValueError("actuator model_id is required")
        values = _readonly_vector(
            self.values, "actuator parameter values", len(ACTUATOR_PARAMETER_NAMES)
        )
        names = _names(self.names, values.size, "actuator parameter names")
        if names != ACTUATOR_PARAMETER_NAMES:
            raise ValueError("actuator parameters use a fixed versioned order")
        calibrated = self.calibrated_wrench
        if type(calibrated) is not bool:
            raise TypeError("calibrated_wrench must be a built-in bool")
        if (
            values[0] <= 0.0
            or values[1] < 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[5] < 0.0
            or values[6] <= values[5]
        ):
            raise ValueError(
                "actuator scale/rates/delay/bounds violate the model domain"
            )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self, "metadata", _frozen_mapping(self.metadata, "metadata")
        )

    @classmethod
    def first_order(
        cls,
        common_thrust_scale: float = 1.0,
        motor_time_constant: float = 0.0,
        command_delay: float = 0.0,
        gimbal_time_constant: float = 0.0,
        gimbal_angle_bias: float = 0.0,
        minimum_thrust: float = 0.0,
        maximum_thrust: float = 100.0,
        calibrated_wrench: bool = False,
        model_id: str = "first_order_gimbal_actuator_v1",
        **metadata: Any
    ) -> "ActuatorParameters":
        return cls(
            model_id=model_id,
            values=np.array(
                [
                    common_thrust_scale,
                    motor_time_constant,
                    command_delay,
                    gimbal_time_constant,
                    gimbal_angle_bias,
                    minimum_thrust,
                    maximum_thrust,
                ],
                dtype=float,
            ),
            calibrated_wrench=calibrated_wrench,
            metadata=metadata,
        )

    def value(self, name: str) -> float:
        try:
            index = self.names.index(str(name))
        except ValueError as exc:
            raise KeyError(name) from exc
        return float(self.values[index])


@dataclass(frozen=True)
class EpisodeNuisance:
    """Episode-specific initial state, disturbance, and sensor bias.

    Unknown controller state is represented by ``None``.  A closed-loop exact
    replay gate must reject it or explicitly marginalize it; this class never
    silently substitutes a zero state.
    """

    initial_plant_state: np.ndarray
    initial_actuator_state: np.ndarray
    disturbance_parameters: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    sensor_bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    controller_state: Optional[Any] = None
    state_sample_id: str = "initial_state_0"
    weight: float = 1.0
    disturbance_model_id: str = "none"

    def __post_init__(self) -> None:
        plant = _readonly_vector(self.initial_plant_state, "initial_plant_state")
        actuator = _readonly_optional_vector(
            self.initial_actuator_state, "initial_actuator_state"
        )
        disturbance = _readonly_optional_vector(
            self.disturbance_parameters, "disturbance_parameters"
        )
        disturbance_model_id = str(
            self.disturbance_model_id
        ).strip()
        bias = _readonly_optional_vector(self.sensor_bias, "sensor_bias")
        sample_id = str(self.state_sample_id)
        weight = float(self.weight)
        if not sample_id:
            raise ValueError("state_sample_id is required")
        if not disturbance_model_id:
            raise ValueError("disturbance_model_id is required")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("episode nuisance weight must be finite and positive")
        object.__setattr__(self, "initial_plant_state", plant)
        object.__setattr__(self, "initial_actuator_state", actuator)
        object.__setattr__(self, "disturbance_parameters", disturbance)
        object.__setattr__(
            self, "disturbance_model_id", disturbance_model_id
        )
        object.__setattr__(self, "sensor_bias", bias)
        object.__setattr__(self, "state_sample_id", sample_id)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class PlantHypothesis:
    """One weighted-posterior particle's static hardware hypothesis."""

    model_id: str
    plant_parameters: np.ndarray
    actuator_parameters: np.ndarray
    disturbance_parameters: np.ndarray
    plant_parameter_names: Tuple[str, ...] = ()
    actuator_parameter_names: Tuple[str, ...] = ACTUATOR_PARAMETER_NAMES
    derived_quantities: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = str(self.model_id)
        if not model_id:
            raise ValueError("plant hypothesis model_id is required")
        plant = _readonly_vector(self.plant_parameters, "plant_parameters")
        actuator = _readonly_optional_vector(
            self.actuator_parameters, "actuator_parameters"
        )
        disturbance = _readonly_optional_vector(
            self.disturbance_parameters, "disturbance_parameters"
        )
        plant_names = (
            tuple("plant_{}".format(index) for index in range(plant.size))
            if not self.plant_parameter_names
            else _names(
                self.plant_parameter_names, plant.size, "plant_parameter_names"
            )
        )
        actuator_names = (
            ()
            if actuator.size == 0 and not self.actuator_parameter_names
            else _names(
                self.actuator_parameter_names,
                actuator.size,
                "actuator_parameter_names",
            )
        )
        derived = {
            str(key): float(value)
            for key, value in self.derived_quantities.items()
        }
        if any(not np.isfinite(value) for value in derived.values()):
            raise ValueError("derived quantities must be finite")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "plant_parameters", plant)
        object.__setattr__(self, "actuator_parameters", actuator)
        object.__setattr__(self, "disturbance_parameters", disturbance)
        object.__setattr__(self, "plant_parameter_names", plant_names)
        object.__setattr__(self, "actuator_parameter_names", actuator_names)
        object.__setattr__(
            self, "derived_quantities", MappingProxyType(derived)
        )

    @property
    def vector(self) -> np.ndarray:
        result = np.concatenate(
            (
                self.plant_parameters,
                self.actuator_parameters,
                self.disturbance_parameters,
            )
        )
        result.setflags(write=False)
        return result


def effective_identifiable_quantities(
    plant_values: Any, actuator_values: Any
) -> Mapping[str, float]:
    """Return reportable command-to-motion combinations for uncalibrated data."""

    plant = _readonly_vector(
        plant_values,
        "effective plant values",
        len(EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES),
    )
    actuator = _readonly_vector(
        actuator_values,
        "effective actuator values",
        len(ACTUATOR_PARAMETER_NAMES),
    )
    # The product, rather than either raw factor, is invariant along the
    # command-scale/authority gauge direction.
    return MappingProxyType(
        {
            "vertical_command_acceleration_gain": float(
                plant[0] * actuator[0]
            ),
            "roll_command_angular_acceleration_gain": float(
                plant[1] * actuator[0]
            ),
            "pitch_command_angular_acceleration_gain": float(
                plant[2] * actuator[0]
            ),
            "yaw_command_angular_acceleration_gain": float(
                plant[3] * actuator[0]
            ),
            "command_delay_s": float(actuator[2]),
            "motor_time_constant_s": float(actuator[1]),
        }
    )


__all__ = [
    "ACTUATOR_PARAMETER_NAMES",
    "ARTICULATED_GIMBALROTOR_MODEL_ID",
    "CALIBRATED_RIGID_BODY_MODEL_ID",
    "CALIBRATED_RIGID_BODY_PARAMETER_NAMES",
    "EFFECTIVE_CLOSED_LOOP_MODEL_ID",
    "EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES",
    "ActuatorParameters",
    "EpisodeNuisance",
    "PlantHypothesis",
    "PlantParameters",
    "effective_identifiable_quantities",
]
