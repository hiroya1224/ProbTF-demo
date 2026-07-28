"""Explicit command-to-realized-wrench actuator backends."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from grape_param_estim.grape_geometry import (
    FIXED_GEOMETRY_PROFILE_SHA256,
    reconstruct_actuator_wrench,
)
from grape_param_estim.plant.parameters import ActuatorParameters


def _readonly(values: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape {}".format(name, shape))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class RealizedWrench:
    stamp: float
    force_body: np.ndarray
    torque_body: np.ndarray
    actuator_state: np.ndarray
    saturated: bool
    calibrated_wrench: bool
    model_id: str

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("wrench stamp must be finite")
        saturated = self.saturated
        calibrated = self.calibrated_wrench
        if type(saturated) is not bool or type(calibrated) is not bool:
            raise TypeError("wrench flags must be built-in bool values")
        state = np.asarray(self.actuator_state, dtype=float).reshape(-1)
        if not np.all(np.isfinite(state)):
            raise ValueError("actuator state must be finite")
        state_copy = np.array(state, copy=True)
        state_copy.setflags(write=False)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(
            self, "force_body", _readonly(self.force_body, (3,), "force_body")
        )
        object.__setattr__(
            self, "torque_body", _readonly(self.torque_body, (3,), "torque_body")
        )
        object.__setattr__(self, "actuator_state", state_copy)
        object.__setattr__(self, "model_id", str(self.model_id))

    @property
    def vector(self) -> np.ndarray:
        result = np.concatenate((self.force_body, self.torque_body))
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class ActuatorFeedback:
    """Particle-local actuator state available to a closed-loop controller."""

    base_thrust: np.ndarray
    gimbal_angle: np.ndarray
    generalized_wrench: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_thrust",
            _readonly(self.base_thrust, (4,), "base_thrust"),
        )
        object.__setattr__(
            self,
            "gimbal_angle",
            _readonly(self.gimbal_angle, (4,), "gimbal_angle"),
        )
        object.__setattr__(
            self,
            "generalized_wrench",
            _readonly(
                self.generalized_wrench,
                (6,),
                "generalized_wrench",
            ),
        )

    @property
    def actuator_state(self) -> np.ndarray:
        state = np.concatenate(
            (
                self.base_thrust,
                self.gimbal_angle,
                self.generalized_wrench,
            )
        )
        state.setflags(write=False)
        return state


@dataclass(frozen=True)
class ActuatorCalibrationIdentity:
    """Content identity required before a wrench is labelled calibrated."""

    artifact_sha256: str
    actuator_model_id: str
    schema: str = "grape_actuator_calibration/v1"

    def __post_init__(self) -> None:
        digest = str(self.artifact_sha256)
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(item not in "0123456789abcdef" for item in digest)
        ):
            raise ValueError(
                "actuator calibration artifact must be a lowercase SHA-256"
            )
        model_id = str(self.actuator_model_id).strip()
        schema = str(self.schema).strip()
        if not model_id or not schema:
            raise ValueError(
                "actuator calibration model and schema are required"
            )
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "actuator_model_id", model_id)
        object.__setattr__(self, "schema", schema)


@runtime_checkable
class ActuatorBackend(Protocol):
    """Structural interface implemented by command-to-wrench backends."""

    model_id: str

    def reset(self, initial_state: np.ndarray) -> None:
        ...

    def feedback(self) -> ActuatorFeedback:
        ...

    def step(
        self,
        command: Any,
        parameters: ActuatorParameters,
        dt: float,
    ) -> RealizedWrench:
        ...


class FirstOrderActuatorBackend:
    """Causal delay, exact-discretized first-order lag, bias, and saturation.

    ``ControllerCommand`` values remain controller-unit targets until this
    backend applies an explicit scale/calibration model.  No caller may
    bypass this layer and call a command a measured SI wrench.
    """

    model_id = "first_order_gimbal_actuator_v1"

    def __init__(
        self, geometry_profile_sha256: Optional[str] = None
    ) -> None:
        """Create the actuator mapping.

        ``None`` is retained only for legacy/unit callers.  Production
        assimilation passes the hash declared in its config and therefore
        cannot silently pick up a different set of module geometry constants.
        """

        if geometry_profile_sha256 is None:
            self.geometry_profile_sha256 = (
                FIXED_GEOMETRY_PROFILE_SHA256
            )
            self.geometry_profile_explicit = False
        else:
            digest = str(geometry_profile_sha256).lower()
            if digest != FIXED_GEOMETRY_PROFILE_SHA256:
                raise ValueError(
                    "actuator geometry profile does not match the "
                    "implemented wrench mapping"
                )
            self.geometry_profile_sha256 = digest
            self.geometry_profile_explicit = True
        self._queue: Deque[
            Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]
        ] = deque()
        self._thrust = np.zeros(4)
        self._gimbal = np.zeros(4)
        self._generalized = np.zeros(6)
        self._target_thrust = np.zeros(4)
        self._target_gimbal = np.zeros(4)
        self._target_generalized = np.zeros(6)
        self._last_command_stamp: Optional[float] = None
        self._last_command_values: Optional[
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        ] = None
        self._last_evaluation_stamp: Optional[float] = None

    def reset(self, initial_state: np.ndarray) -> None:
        values = np.asarray(initial_state, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("initial actuator state must be finite")
        if values.size not in (0, 8, 14):
            raise ValueError(
                "initial actuator state must be empty, [thrust,gimbal], or "
                "[thrust,gimbal,generalized]"
            )
        self._thrust = np.zeros(4)
        self._gimbal = np.zeros(4)
        self._generalized = np.zeros(6)
        if values.size >= 8:
            self._thrust[:] = values[:4]
            self._gimbal[:] = values[4:8]
        if values.size == 14:
            self._generalized[:] = values[8:14]
        self._target_thrust = np.array(self._thrust, copy=True)
        self._target_gimbal = np.array(self._gimbal, copy=True)
        self._target_generalized = np.array(
            self._generalized, copy=True
        )
        self._queue.clear()
        self._last_command_stamp = None
        self._last_command_values = None
        self._last_evaluation_stamp = None

    def feedback(self) -> ActuatorFeedback:
        """Return a detached view of this rollout's current actuator state."""

        return ActuatorFeedback(
            base_thrust=self._thrust,
            gimbal_angle=self._gimbal,
            generalized_wrench=self._generalized,
        )

    @staticmethod
    def _command_arrays(
        command: Any,
    ) -> Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        stamp = float(getattr(command, "stamp"))
        thrust = np.asarray(getattr(command, "base_thrust"), dtype=float).reshape(-1)
        gimbal = np.asarray(getattr(command, "gimbal_angle"), dtype=float).reshape(-1)
        generalized_value = getattr(command, "generalized_wrench", None)
        generalized = (
            None
            if generalized_value is None
            else np.asarray(generalized_value, dtype=float).reshape(-1)
        )
        if (
            not np.isfinite(stamp)
            or thrust.shape != (4,)
            or gimbal.shape != (4,)
            or not np.all(np.isfinite(thrust))
            or not np.all(np.isfinite(gimbal))
            or (
                generalized is not None
                and (
                    generalized.shape != (6,)
                    or not np.all(np.isfinite(generalized))
                )
            )
        ):
            raise ValueError("controller command has invalid actuator fields")
        return stamp, thrust.copy(), gimbal.copy(), (
            None if generalized is None else generalized.copy()
        )

    @staticmethod
    def _lag(current: np.ndarray, target: np.ndarray, dt: float, tau: float) -> np.ndarray:
        if tau <= np.finfo(float).eps:
            return np.array(target, copy=True)
        alpha = -np.expm1(-dt / tau)
        return current + alpha * (target - current)

    def _advance(
        self,
        duration: float,
        parameters: ActuatorParameters,
    ) -> bool:
        if duration <= 0.0:
            return False
        self._thrust = self._lag(
            self._thrust,
            self._target_thrust,
            duration,
            parameters.value("motor_time_constant"),
        )
        self._gimbal = self._lag(
            self._gimbal,
            self._target_gimbal
            + parameters.value("gimbal_angle_bias"),
            duration,
            parameters.value("gimbal_time_constant"),
        )
        self._generalized = self._lag(
            self._generalized,
            self._target_generalized,
            duration,
            parameters.value("motor_time_constant"),
        )
        lower = parameters.value("minimum_thrust")
        upper = parameters.value("maximum_thrust")
        clipped = np.clip(self._thrust, lower, upper)
        saturated = bool(np.any(clipped != self._thrust))
        self._thrust = clipped
        return saturated

    def step(
        self,
        command: Any,
        parameters: ActuatorParameters,
        dt: float,
        evaluation_stamp: Optional[float] = None,
    ) -> RealizedWrench:
        if not isinstance(parameters, ActuatorParameters):
            raise TypeError("parameters must be ActuatorParameters")
        delta = float(dt)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("actuator dt must be finite and non-negative")
        command_stamp, thrust, gimbal, generalized = self._command_arrays(
            command
        )
        stamp = (
            command_stamp + delta
            if evaluation_stamp is None
            else float(evaluation_stamp)
        )
        if not np.isfinite(stamp):
            raise ValueError("actuator evaluation stamp must be finite")
        if (
            self._last_evaluation_stamp is not None
            and stamp < self._last_evaluation_stamp - 1.0e-12
        ):
            raise ValueError(
                "actuator evaluation timestamps must be non-decreasing"
            )
        if command_stamp > stamp + 1.0e-12:
            raise ValueError("future controller command cannot drive actuator")
        interval_start = stamp - delta
        if (
            self._last_evaluation_stamp is not None
            and not np.isclose(
                interval_start,
                self._last_evaluation_stamp,
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "actuator dt must span consecutive evaluation timestamps"
            )
        if (
            self._last_command_stamp is not None
            and command_stamp < self._last_command_stamp - 1.0e-12
        ):
            raise ValueError("actuator command timestamps must be non-decreasing")
        is_new_command = (
            self._last_command_stamp is None
            or command_stamp > self._last_command_stamp + 1.0e-12
        )
        if not is_new_command:
            previous_thrust, previous_gimbal, previous_generalized = (
                self._last_command_values
            )
            same_generalized = (
                generalized is None
                and previous_generalized is None
            ) or (
                generalized is not None
                and previous_generalized is not None
                and np.array_equal(generalized, previous_generalized)
            )
            if (
                not np.array_equal(thrust, previous_thrust)
                or not np.array_equal(gimbal, previous_gimbal)
                or not same_generalized
            ):
                raise ValueError(
                    "one command timestamp cannot contain different targets"
                )
        else:
            self._queue.append(
                (command_stamp, thrust, gimbal, generalized)
            )
            self._last_command_stamp = command_stamp
            self._last_command_values = (
                np.array(thrust, copy=True),
                np.array(gimbal, copy=True),
                None
                if generalized is None
                else np.array(generalized, copy=True),
            )
        delay = parameters.value("command_delay")
        cursor = interval_start
        saturated = bool(getattr(command, "saturated", False))
        while (
            self._queue
            and self._queue[0][0] + delay <= stamp + 1.0e-12
        ):
            activation_stamp = self._queue[0][0] + delay
            if activation_stamp > cursor:
                saturated = (
                    self._advance(
                        min(activation_stamp, stamp) - cursor,
                        parameters,
                    )
                    or saturated
                )
                cursor = min(activation_stamp, stamp)
            (
                _,
                self._target_thrust,
                self._target_gimbal,
                target_generalized,
            ) = self._queue.popleft()
            self._target_generalized = (
                np.zeros(6)
                if target_generalized is None
                else target_generalized
            )
            cursor = max(cursor, activation_stamp)
        if stamp > cursor:
            saturated = (
                self._advance(stamp - cursor, parameters) or saturated
            )
        self._last_evaluation_stamp = stamp
        scale = parameters.value("common_thrust_scale")

        # Base thrust/gimbal channels are preferred because they keep the
        # controller/actuator boundary visible.  A generalized target is only
        # used when no base-thrust authority is present.
        if np.any(np.abs(self._thrust) > 0.0):
            vector = reconstruct_actuator_wrench(
                self._thrust * scale, self._gimbal
            )
        else:
            vector = self._generalized * scale
        state = self.feedback().actuator_state
        return RealizedWrench(
            stamp=stamp,
            force_body=vector[:3],
            torque_body=vector[3:],
            actuator_state=state,
            saturated=saturated,
            calibrated_wrench=parameters.calibrated_wrench,
            model_id=self.model_id,
        )


__all__ = [
    "ActuatorFeedback",
    "ActuatorCalibrationIdentity",
    "ActuatorBackend",
    "FirstOrderActuatorBackend",
    "RealizedWrench",
]
