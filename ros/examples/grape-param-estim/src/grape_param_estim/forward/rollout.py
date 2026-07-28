"""Immutable forward-rollout inputs and results."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.plant.actuator import RealizedWrench
from grape_param_estim.plant.parameters import PlantHypothesis
from grape_param_estim.plant.rigid_body import PlantState
from grape_param_estim.plant.sensor import PredictedObservation


def _readonly(values: Any, shape_suffix: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim < len(shape_suffix)
        or array.shape[-len(shape_suffix) :] != shape_suffix
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            "{} must be finite with trailing shape {}".format(name, shape_suffix)
        )
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _times(values: Any, name: str, minimum_count: int = 1) -> np.ndarray:
    times = np.asarray(values, dtype=float).reshape(-1)
    if (
        times.size < minimum_count
        or not np.all(np.isfinite(times))
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError("{} must be finite and strictly increasing".format(name))
    copy = np.array(times, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class CommandSample:
    """Controller-unit command at one event time."""

    stamp: float
    base_thrust: np.ndarray
    gimbal_angle: np.ndarray
    generalized_wrench: Optional[np.ndarray] = None
    events: Tuple[int, ...] = ()
    saturated: bool = False

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("command stamp must be finite")
        saturated = self.saturated
        if type(saturated) is not bool:
            raise TypeError("command saturated must be a built-in bool")
        generalized = self.generalized_wrench
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(
            self, "base_thrust", _readonly(self.base_thrust, (4,), "base_thrust")
        )
        object.__setattr__(
            self, "gimbal_angle", _readonly(self.gimbal_angle, (4,), "gimbal_angle")
        )
        if generalized is not None:
            object.__setattr__(
                self,
                "generalized_wrench",
                _readonly(generalized, (6,), "generalized_wrench"),
            )
        object.__setattr__(
            self, "events", tuple(int(item) for item in self.events)
        )


@dataclass(frozen=True)
class RecordedCommandSeries:
    timestamps: np.ndarray
    base_thrust: np.ndarray
    gimbal_angle: np.ndarray
    generalized_wrench: Optional[np.ndarray] = None
    source_topic: str = "/gimbalrotor/four_axes/command"
    source_bag_sha256: str = ""

    def __post_init__(self) -> None:
        times = _times(self.timestamps, "recorded command timestamps")
        count = times.size
        thrust = _readonly(self.base_thrust, (4,), "base_thrust")
        gimbal = _readonly(self.gimbal_angle, (4,), "gimbal_angle")
        if thrust.shape != (count, 4) or gimbal.shape != (count, 4):
            raise ValueError("recorded command arrays must have shape (N, 4)")
        generalized = self.generalized_wrench
        if generalized is not None:
            generalized = _readonly(
                generalized, (6,), "generalized_wrench"
            )
            if generalized.shape != (count, 6):
                raise ValueError("generalized_wrench must have shape (N, 6)")
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "base_thrust", thrust)
        object.__setattr__(self, "gimbal_angle", gimbal)
        object.__setattr__(self, "generalized_wrench", generalized)
        object.__setattr__(self, "source_topic", str(self.source_topic))

    def causal_sample(self, stamp: float) -> CommandSample:
        value = float(stamp)
        if value < self.timestamps[0] - 1.0e-12:
            raise ValueError("no recorded command is available before rollout time")
        index = int(np.searchsorted(self.timestamps, value, side="right") - 1)
        return CommandSample(
            # Preserve the source event time.  The actuator's evaluation clock
            # is supplied separately so pure delay is measured from the
            # recorded command event rather than from every integration step.
            stamp=float(self.timestamps[index]),
            base_thrust=self.base_thrust[index],
            gimbal_angle=self.gimbal_angle[index],
            generalized_wrench=(
                None
                if self.generalized_wrench is None
                else self.generalized_wrench[index]
            ),
        )

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "timestamps": self.timestamps,
                "base_thrust": self.base_thrust,
                "gimbal_angle": self.gimbal_angle,
                "generalized_wrench": self.generalized_wrench,
                "source_topic": self.source_topic,
                "source_bag_sha256": self.source_bag_sha256,
            }
        )


@dataclass
class RolloutState:
    controller_state: Any
    actuator_state: np.ndarray
    plant_state: PlantState
    sensor_state: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    mode: str
    model_id: str
    hypothesis: PlantHypothesis
    integration_timestamps: np.ndarray
    plant_states: Tuple[PlantState, ...]
    commands: Tuple[Any, ...]
    realized_wrenches: Tuple[RealizedWrench, ...]
    predicted_observations: Tuple[PredictedObservation, ...]
    controller_tick_timestamps: np.ndarray
    likelihood_timestamps: np.ndarray
    events: Tuple[Mapping[str, Any], ...]
    used_recorded_commands: bool
    controller_fidelity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in (
            "open_loop_plant_identification",
            "closed_loop_plant_identification",
        ):
            raise ValueError("unsupported rollout mode")
        if not isinstance(self.hypothesis, PlantHypothesis):
            raise TypeError("hypothesis must be PlantHypothesis")
        used = self.used_recorded_commands
        if type(used) is not bool:
            raise TypeError("used_recorded_commands must be a built-in bool")
        if self.mode.startswith("open_loop") and not used:
            raise ValueError("open-loop rollout must record its command source")
        if self.mode.startswith("closed_loop") and used:
            raise ValueError("closed-loop rollout may not reuse recorded commands")
        integration = _times(
            self.integration_timestamps, "integration_timestamps"
        )
        controller = _times(
            self.controller_tick_timestamps,
            "controller_tick_timestamps",
            minimum_count=0,
        )
        likelihood = _times(
            self.likelihood_timestamps,
            "likelihood_timestamps",
            minimum_count=0,
        )
        if len(self.plant_states) != integration.size:
            raise ValueError("one plant state is required per integration time")
        if len(self.predicted_observations) != integration.size:
            raise ValueError("one prediction is required per integration time")
        if len(self.realized_wrenches) != max(0, integration.size - 1):
            raise ValueError("one realized wrench is required per integration step")
        for index, state in enumerate(self.plant_states):
            if not np.isclose(
                float(state.stamp),
                float(integration[index]),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError("plant state stamps must align with integration")
        for index, prediction in enumerate(self.predicted_observations):
            if not np.isclose(
                float(prediction.stamp),
                float(integration[index]),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "observation stamps must align with integration"
                )
        for index, wrench in enumerate(self.realized_wrenches, start=1):
            if not np.isclose(
                float(wrench.stamp),
                float(integration[index]),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError("wrench stamps must align with integration")
        events = tuple(
            MappingProxyType(dict(item)) for item in self.events
        )
        object.__setattr__(self, "integration_timestamps", integration)
        object.__setattr__(self, "controller_tick_timestamps", controller)
        object.__setattr__(self, "likelihood_timestamps", likelihood)
        object.__setattr__(self, "plant_states", tuple(self.plant_states))
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(
            self, "realized_wrenches", tuple(self.realized_wrenches)
        )
        object.__setattr__(
            self, "predicted_observations", tuple(self.predicted_observations)
        )
        object.__setattr__(self, "events", events)

    @property
    def positions(self) -> np.ndarray:
        result = np.stack([item.position_world for item in self.plant_states])
        result.setflags(write=False)
        return result

    @property
    def orientations_xyzw(self) -> np.ndarray:
        result = np.stack([item.orientation_xyzw for item in self.plant_states])
        result.setflags(write=False)
        return result

    @property
    def velocities(self) -> np.ndarray:
        result = np.stack([item.velocity_world for item in self.plant_states])
        result.setflags(write=False)
        return result

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "mode": self.mode,
                "model_id": self.model_id,
                "hypothesis": self.hypothesis.vector,
                "integration_timestamps": self.integration_timestamps,
                "positions": self.positions,
                "orientations_xyzw": self.orientations_xyzw,
                "velocities": self.velocities,
                "controller_tick_timestamps": self.controller_tick_timestamps,
                "likelihood_timestamps": self.likelihood_timestamps,
                "used_recorded_commands": self.used_recorded_commands,
                "controller_fidelity": self.controller_fidelity,
            }
        )


__all__ = [
    "CommandSample",
    "RecordedCommandSeries",
    "RolloutResult",
    "RolloutState",
]
