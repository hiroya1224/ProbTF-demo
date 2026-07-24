"""Controller replay contracts for factual and counterfactual trajectories.

``VectorPidSurrogate`` mirrors the scalar clamp/integrator semantics of the
PC-side ``aerial_robot_control::PID`` class but is still explicitly a Python
surrogate.  It is never promoted to the exact replay oracle merely because a
unit test passes.  ``ExactControllerAdapter`` provides the same interface for
an externally built PC+MCU replay backend and carries a distinct backend ID.

Both teacher-forced and free-run paths call the controller from the beginning
of an episode.  Candidate gains therefore change the controller state and
command history; a counterfactual never reuses the recorded command as if it
were invariant to the candidate.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
AXIS_COUNT = len(AXIS_NAMES)
POSITION_CONTROL = 0
VELOCITY_CONTROL = 1
ACCELERATION_CONTROL = 2
_CONTROL_MODES = (POSITION_CONTROL, VELOCITY_CONTROL, ACCELERATION_CONTROL)


def _vector(values: Any, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(AXIS_COUNT, float(array))
    if array.shape != (AXIS_COUNT,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be finite scalar or six-vector".format(name))
    if positive and np.any(array <= 0.0):
        raise ValueError("{} must be strictly positive".format(name))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _matrix(values: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (count, AXIS_COUNT) or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape ({}, 6)".format(name, count))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _mask_matrix(
    values: Optional[Any], count: int, name: str, default: bool
) -> np.ndarray:
    if values is None:
        array = np.full((count, AXIS_COUNT), default, dtype=bool)
    else:
        array = np.asarray(values, dtype=bool)
    if array.shape != (count, AXIS_COUNT):
        raise ValueError("{} must have shape ({}, 6)".format(name, count))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _shortest_angle(error: float) -> float:
    return float((float(error) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class PidLimits:
    output: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    p_error: np.ndarray
    i_state: np.ndarray
    d_error: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "output",
            "p_term",
            "i_term",
            "d_term",
            "p_error",
            "i_state",
            "d_error",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name, positive=True))

    @classmethod
    def unbounded(cls, limit: float = 1.0e6) -> "PidLimits":
        values = np.full(AXIS_COUNT, float(limit))
        return cls(values, values, values, values, values, values, values)


@dataclass(frozen=True)
class ControllerParameters:
    p_gain: np.ndarray
    i_gain: np.ndarray
    d_gain: np.ndarray
    limits: PidLimits
    controller_mass: float = 1.0
    controller_inertia_diagonal: np.ndarray = None
    allocation_scale: np.ndarray = None
    thrust_scale: float = 1.0
    delay_compensation_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_gain", _vector(self.p_gain, "p_gain"))
        object.__setattr__(self, "i_gain", _vector(self.i_gain, "i_gain"))
        object.__setattr__(self, "d_gain", _vector(self.d_gain, "d_gain"))
        if not isinstance(self.limits, PidLimits):
            raise TypeError("limits must be PidLimits")
        mass = float(self.controller_mass)
        thrust_scale = float(self.thrust_scale)
        delay = float(self.delay_compensation_s)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("controller_mass must be finite and positive")
        if not np.isfinite(thrust_scale) or thrust_scale <= 0.0:
            raise ValueError("thrust_scale must be finite and positive")
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("delay_compensation_s must be finite and non-negative")
        inertia = (
            np.ones(3)
            if self.controller_inertia_diagonal is None
            else np.asarray(self.controller_inertia_diagonal, dtype=float)
        )
        if inertia.shape != (3,) or not np.all(np.isfinite(inertia)) or np.any(inertia <= 0.0):
            raise ValueError("controller_inertia_diagonal must be a positive three-vector")
        allocation = (
            np.ones(AXIS_COUNT)
            if self.allocation_scale is None
            else _vector(self.allocation_scale, "allocation_scale", positive=True)
        )
        inertia_copy = np.array(inertia, copy=True)
        inertia_copy.setflags(write=False)
        object.__setattr__(self, "controller_mass", mass)
        object.__setattr__(self, "controller_inertia_diagonal", inertia_copy)
        object.__setattr__(self, "allocation_scale", allocation)
        object.__setattr__(self, "thrust_scale", thrust_scale)
        object.__setattr__(self, "delay_compensation_s", delay)

    @property
    def generalized_inertia(self) -> np.ndarray:
        values = np.concatenate(
            (
                np.full(3, self.controller_mass),
                self.controller_inertia_diagonal,
            )
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class ControllerReference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))


@dataclass(frozen=True)
class ControllerFeedback:
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, "position"))
        object.__setattr__(self, "velocity", _vector(self.velocity, "velocity"))


@dataclass(frozen=True)
class ControllerStep:
    p_error: np.ndarray
    d_error: np.ndarray
    integral_state: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    feedforward_term: np.ndarray
    acceleration_command: np.ndarray
    generalized_wrench_command: np.ndarray
    term_saturated: np.ndarray
    output_saturated: np.ndarray
    mode: Any

    def __post_init__(self) -> None:
        for name in (
            "p_error",
            "d_error",
            "integral_state",
            "p_term",
            "i_term",
            "d_term",
            "feedforward_term",
            "acceleration_command",
            "generalized_wrench_command",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        for name in ("term_saturated", "output_saturated"):
            array = np.asarray(getattr(self, name), dtype=bool)
            if array.shape != (AXIS_COUNT,):
                raise ValueError("{} must be a six-vector".format(name))
            copy = np.array(array, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)


class VectorPidSurrogate:
    """Six independent PID axes with the deployed PC clamp semantics."""

    backend_id = "python_vector_pid_surrogate/v1"
    is_exact = False

    def __init__(
        self,
        parameters: ControllerParameters,
        initial_integral_state: Optional[np.ndarray] = None,
        yaw_axis: int = 5,
        nonnegative_integrator_axes: Sequence[int] = (2,),
    ) -> None:
        self.parameters = parameters
        self.yaw_axis = int(yaw_axis)
        if self.yaw_axis < 0 or self.yaw_axis >= AXIS_COUNT:
            raise ValueError("yaw_axis is out of range")
        axes = tuple(int(item) for item in nonnegative_integrator_axes)
        if any(item < 0 or item >= AXIS_COUNT for item in axes):
            raise ValueError("nonnegative integrator axis is out of range")
        self.nonnegative_integrator_axes = axes
        self.integral_state = (
            np.zeros(AXIS_COUNT)
            if initial_integral_state is None
            else np.array(_vector(initial_integral_state, "initial_integral_state"))
        )
        for axis in self.nonnegative_integrator_axes:
            self.integral_state[axis] = max(0.0, self.integral_state[axis])

    def reset(self) -> None:
        self.integral_state.fill(0.0)

    def set_parameters(self, parameters: ControllerParameters) -> None:
        if not isinstance(parameters, ControllerParameters):
            raise TypeError("parameters must be ControllerParameters")
        self.parameters = parameters

    def step(
        self,
        reference: ControllerReference,
        feedback: ControllerFeedback,
        delta: float,
        integration_enabled: Optional[np.ndarray] = None,
        control_mode: Optional[np.ndarray] = None,
        mode: Any = None,
    ) -> ControllerStep:
        dt = float(delta)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("delta must be finite and non-negative")
        integrate = (
            np.ones(AXIS_COUNT, dtype=bool)
            if integration_enabled is None
            else np.asarray(integration_enabled, dtype=bool)
        )
        modes = (
            np.full(AXIS_COUNT, POSITION_CONTROL, dtype=int)
            if control_mode is None
            else np.asarray(control_mode, dtype=int)
        )
        if integrate.shape != (AXIS_COUNT,) or modes.shape != (AXIS_COUNT,):
            raise ValueError("integration_enabled and control_mode must be six-vectors")
        if any(int(item) not in _CONTROL_MODES for item in modes):
            raise ValueError("control_mode contains an unsupported value")

        raw_p_error = reference.position - feedback.position
        raw_p_error[self.yaw_axis] = _shortest_angle(raw_p_error[self.yaw_axis])
        raw_d_error = reference.velocity - feedback.velocity
        raw_p_error = np.where(modes == POSITION_CONTROL, raw_p_error, 0.0)
        raw_d_error = np.where(modes == ACCELERATION_CONTROL, 0.0, raw_d_error)
        limits = self.parameters.limits
        p_error = np.clip(raw_p_error, -limits.p_error, limits.p_error)
        d_error = np.clip(raw_d_error, -limits.d_error, limits.d_error)
        increment = np.where(integrate, p_error * dt, 0.0)
        self.integral_state = np.clip(
            self.integral_state + increment,
            -limits.i_state,
            limits.i_state,
        )
        for axis in self.nonnegative_integrator_axes:
            self.integral_state[axis] = max(0.0, self.integral_state[axis])

        raw_p = p_error * self.parameters.p_gain
        raw_i = self.integral_state * self.parameters.i_gain
        raw_d = d_error * self.parameters.d_gain
        p_term = np.clip(raw_p, -limits.p_term, limits.p_term)
        i_term = np.clip(raw_i, -limits.i_term, limits.i_term)
        d_term = np.clip(raw_d, -limits.d_term, limits.d_term)
        feedforward = reference.acceleration
        raw_output = p_term + i_term + d_term + feedforward
        output = np.clip(raw_output, -limits.output, limits.output)
        term_saturated = (
            (raw_p != p_term) | (raw_i != i_term) | (raw_d != d_term)
        )
        output_saturated = raw_output != output
        wrench = (
            output
            * self.parameters.generalized_inertia
            * self.parameters.allocation_scale
            / self.parameters.thrust_scale
        )
        return ControllerStep(
            p_error=p_error,
            d_error=d_error,
            integral_state=self.integral_state.copy(),
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            feedforward_term=feedforward,
            acceleration_command=output,
            generalized_wrench_command=wrench,
            term_saturated=term_saturated,
            output_saturated=output_saturated,
            mode=mode,
        )


class ExactControllerAdapter:
    """Adapter for a separately built exact PC+MCU replay oracle."""

    is_exact = True

    def __init__(
        self,
        step_callback: Callable[..., ControllerStep],
        reset_callback: Callable[[], None],
        parameter_callback: Callable[[ControllerParameters], None],
        backend_id: str,
    ) -> None:
        if not callable(step_callback) or not callable(reset_callback) or not callable(parameter_callback):
            raise ValueError("exact adapter callbacks must be callable")
        if not backend_id or "surrogate" in backend_id.lower():
            raise ValueError("exact adapter needs an unambiguous backend_id")
        self._step_callback = step_callback
        self._reset_callback = reset_callback
        self._parameter_callback = parameter_callback
        self.backend_id = str(backend_id)

    def reset(self) -> None:
        self._reset_callback()

    def set_parameters(self, parameters: ControllerParameters) -> None:
        self._parameter_callback(parameters)

    def step(self, *args, **kwargs) -> ControllerStep:
        result = self._step_callback(*args, **kwargs)
        if not isinstance(result, ControllerStep):
            raise TypeError("exact replay callback must return ControllerStep")
        return result


@dataclass(frozen=True)
class ParameterChange:
    stamp: float
    parameters: ControllerParameters

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("parameter-change stamp must be finite")
        if not isinstance(self.parameters, ControllerParameters):
            raise TypeError("parameter-change value must be ControllerParameters")
        object.__setattr__(self, "stamp", stamp)


@dataclass(frozen=True)
class ReplayRequest:
    timestamps: np.ndarray
    reference_position: np.ndarray
    reference_velocity: np.ndarray
    reference_acceleration: np.ndarray
    actual_position: Optional[np.ndarray] = None
    actual_velocity: Optional[np.ndarray] = None
    control_mode: Optional[np.ndarray] = None
    integration_enabled: Optional[np.ndarray] = None
    reset_mask: Optional[np.ndarray] = None
    modes: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("timestamps must be finite, increasing, and contain at least two samples")
        count = times.size
        object.__setattr__(self, "timestamps", np.array(times, copy=True))
        for name in (
            "reference_position",
            "reference_velocity",
            "reference_acceleration",
        ):
            object.__setattr__(self, name, _matrix(getattr(self, name), count, name))
        for name in ("actual_position", "actual_velocity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _matrix(value, count, name))
        control_mode = (
            np.full((count, AXIS_COUNT), POSITION_CONTROL, dtype=int)
            if self.control_mode is None
            else np.asarray(self.control_mode, dtype=int)
        )
        if control_mode.shape != (count, AXIS_COUNT) or any(
            int(item) not in _CONTROL_MODES for item in control_mode.reshape(-1)
        ):
            raise ValueError("control_mode must have shape (N, 6) with supported values")
        object.__setattr__(
            self, "control_mode", np.array(control_mode, copy=True)
        )
        object.__setattr__(
            self,
            "integration_enabled",
            _mask_matrix(self.integration_enabled, count, "integration_enabled", True),
        )
        reset = (
            np.zeros(count, dtype=bool)
            if self.reset_mask is None
            else np.asarray(self.reset_mask, dtype=bool)
        )
        if reset.shape != (count,):
            raise ValueError("reset_mask must have shape (N,)")
        object.__setattr__(self, "reset_mask", np.array(reset, copy=True))
        modes = tuple([None] * count if self.modes is None else self.modes)
        if len(modes) != count:
            raise ValueError("modes must contain one value per timestamp")
        object.__setattr__(self, "modes", modes)

    @property
    def count(self) -> int:
        return int(self.timestamps.size)


@dataclass(frozen=True)
class ReplayResult:
    timestamps: np.ndarray
    feedback_position: np.ndarray
    feedback_velocity: np.ndarray
    p_error: np.ndarray
    d_error: np.ndarray
    integral_state: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    feedforward_term: np.ndarray
    acceleration_command: np.ndarray
    generalized_wrench_command: np.ndarray
    term_saturated: np.ndarray
    output_saturated: np.ndarray
    reset_applied: np.ndarray
    backend_id: str
    is_exact: bool
    replay_mode: str
    initial_integrator_known: bool


class ControllerReplay:
    """Chronological teacher-forced or closed-loop free-run replay."""

    def __init__(
        self,
        backend_factory: Callable[
            [ControllerParameters, Optional[np.ndarray]], Any
        ] = VectorPidSurrogate,
    ) -> None:
        if not callable(backend_factory):
            raise ValueError("backend_factory must be callable")
        self.backend_factory = backend_factory

    def run(
        self,
        request: ReplayRequest,
        parameters: ControllerParameters,
        replay_mode: str,
        initial_position: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
        initial_integral_state: Optional[np.ndarray] = None,
        parameter_changes: Sequence[ParameterChange] = (),
        plant_step: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray, float, int], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        reset_on_mode_change: bool = True,
        apply_delay_compensation: bool = True,
        plant_input: str = "acceleration_command",
    ) -> ReplayResult:
        if replay_mode not in ("teacher_forced", "free_run"):
            raise ValueError("replay_mode must be teacher_forced or free_run")
        if replay_mode == "teacher_forced" and (
            request.actual_position is None or request.actual_velocity is None
        ):
            raise ValueError("teacher_forced replay requires actual state arrays")
        if replay_mode == "free_run" and plant_step is None:
            raise ValueError("free_run replay requires plant_step")
        if plant_input not in (
            "acceleration_command",
            "generalized_wrench_command",
        ):
            raise ValueError(
                "plant_input must be acceleration_command or generalized_wrench_command"
            )
        backend = self.backend_factory(parameters, initial_integral_state)
        if not all(hasattr(backend, name) for name in ("step", "reset", "set_parameters")):
            raise TypeError("controller backend does not satisfy the replay contract")
        changes = tuple(sorted(parameter_changes, key=lambda item: item.stamp))
        change_index = 0
        count = request.count
        fields = {
            name: np.zeros((count, AXIS_COUNT))
            for name in (
                "feedback_position",
                "feedback_velocity",
                "p_error",
                "d_error",
                "integral_state",
                "p_term",
                "i_term",
                "d_term",
                "feedforward_term",
                "acceleration_command",
                "generalized_wrench_command",
            )
        }
        term_saturated = np.zeros((count, AXIS_COUNT), dtype=bool)
        output_saturated = np.zeros((count, AXIS_COUNT), dtype=bool)
        reset_applied = np.zeros(count, dtype=bool)
        if replay_mode == "free_run":
            position = _vector(
                request.reference_position[0] if initial_position is None else initial_position,
                "initial_position",
            ).copy()
            velocity = _vector(
                np.zeros(AXIS_COUNT) if initial_velocity is None else initial_velocity,
                "initial_velocity",
            ).copy()

        previous_mode = None
        for index in range(count):
            stamp = float(request.timestamps[index])
            while change_index < len(changes) and changes[change_index].stamp <= stamp:
                backend.set_parameters(changes[change_index].parameters)
                change_index += 1
            mode_changed = (
                index > 0
                and request.modes[index] != previous_mode
                and request.modes[index] is not None
            )
            if request.reset_mask[index] or (reset_on_mode_change and mode_changed):
                backend.reset()
                reset_applied[index] = True
            previous_mode = request.modes[index]
            if replay_mode == "teacher_forced":
                feedback = ControllerFeedback(
                    request.actual_position[index], request.actual_velocity[index]
                )
            else:
                feedback = ControllerFeedback(position, velocity)
            reference_stamp = (
                stamp + float(backend.parameters.delay_compensation_s)
                if apply_delay_compensation
                and hasattr(backend, "parameters")
                else stamp
            )

            def interpolated_reference(values):
                return np.asarray(
                    [
                        np.interp(
                            reference_stamp,
                            request.timestamps,
                            values[:, axis],
                        )
                        for axis in range(AXIS_COUNT)
                    ]
                )

            reference = ControllerReference(
                interpolated_reference(request.reference_position),
                interpolated_reference(request.reference_velocity),
                interpolated_reference(request.reference_acceleration),
            )
            delta = (
                0.0
                if index == 0
                else float(request.timestamps[index] - request.timestamps[index - 1])
            )
            step = backend.step(
                reference,
                feedback,
                delta,
                integration_enabled=request.integration_enabled[index],
                control_mode=request.control_mode[index],
                mode=request.modes[index],
            )
            fields["feedback_position"][index] = feedback.position
            fields["feedback_velocity"][index] = feedback.velocity
            for name in (
                "p_error",
                "d_error",
                "integral_state",
                "p_term",
                "i_term",
                "d_term",
                "feedforward_term",
                "acceleration_command",
                "generalized_wrench_command",
            ):
                fields[name][index] = getattr(step, name)
            term_saturated[index] = step.term_saturated
            output_saturated[index] = step.output_saturated
            if replay_mode == "free_run" and index + 1 < count:
                next_delta = float(
                    request.timestamps[index + 1] - request.timestamps[index]
                )
                position, velocity = plant_step(
                    position.copy(),
                    velocity.copy(),
                    getattr(step, plant_input).copy(),
                    next_delta,
                    index,
                )
                position = _vector(position, "plant position").copy()
                velocity = _vector(velocity, "plant velocity").copy()

        return ReplayResult(
            timestamps=np.array(request.timestamps, copy=True),
            term_saturated=term_saturated,
            output_saturated=output_saturated,
            reset_applied=reset_applied,
            backend_id=str(getattr(backend, "backend_id", type(backend).__name__)),
            is_exact=bool(getattr(backend, "is_exact", False)),
            replay_mode=replay_mode,
            initial_integrator_known=initial_integral_state is not None,
            **fields
        )


@dataclass(frozen=True)
class ReplayMetrics:
    normalized_rmse: np.ndarray
    normalized_maximum_error: np.ndarray
    event_agreement: float
    passed: bool
    rmse_threshold: float
    maximum_error_threshold: float
    event_agreement_threshold: float


def replay_metrics(
    predicted: np.ndarray,
    recorded: np.ndarray,
    predicted_events: Optional[np.ndarray] = None,
    recorded_events: Optional[np.ndarray] = None,
    scale_floor: float = 1.0e-9,
    rmse_threshold: float = 0.01,
    maximum_error_threshold: float = 0.03,
    event_agreement_threshold: float = 1.0,
) -> ReplayMetrics:
    """Compute the frozen factual replay gate from TODO V3."""

    first = np.asarray(predicted, dtype=float)
    second = np.asarray(recorded, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("predicted and recorded must be aligned finite matrices")
    centered = second - np.mean(second, axis=0)
    scale = np.maximum(
        np.maximum(np.sqrt(np.mean(centered * centered, axis=0)), np.ptp(second, axis=0)),
        float(scale_floor),
    )
    difference = first - second
    normalized_rmse = np.sqrt(np.mean(difference * difference, axis=0)) / scale
    normalized_maximum = np.max(np.abs(difference), axis=0) / scale
    if predicted_events is None and recorded_events is None:
        agreement = 1.0
    elif predicted_events is None or recorded_events is None:
        raise ValueError("both event arrays must be supplied together")
    else:
        event_first = np.asarray(predicted_events)
        event_second = np.asarray(recorded_events)
        if event_first.shape != event_second.shape:
            raise ValueError("event arrays must have matching shape")
        agreement = float(np.mean(event_first == event_second)) if event_first.size else 1.0
    passed = bool(
        np.all(normalized_rmse <= float(rmse_threshold))
        and np.all(normalized_maximum <= float(maximum_error_threshold))
        and agreement >= float(event_agreement_threshold)
    )
    return ReplayMetrics(
        normalized_rmse=normalized_rmse,
        normalized_maximum_error=normalized_maximum,
        event_agreement=agreement,
        passed=passed,
        rmse_threshold=float(rmse_threshold),
        maximum_error_threshold=float(maximum_error_threshold),
        event_agreement_threshold=float(event_agreement_threshold),
    )


__all__ = [
    "ACCELERATION_CONTROL",
    "AXIS_COUNT",
    "AXIS_NAMES",
    "ControllerFeedback",
    "ControllerParameters",
    "ControllerReference",
    "ControllerReplay",
    "ControllerStep",
    "ExactControllerAdapter",
    "POSITION_CONTROL",
    "ParameterChange",
    "PidLimits",
    "ReplayMetrics",
    "ReplayRequest",
    "ReplayResult",
    "VELOCITY_CONTROL",
    "VectorPidSurrogate",
    "replay_metrics",
]
