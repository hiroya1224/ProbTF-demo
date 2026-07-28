"""Recorded-command open-loop plant identification rollout."""

import inspect
from typing import Any, Callable, Optional, Tuple

import numpy as np

from grape_param_estim.forward.rollout import (
    CommandSample,
    RecordedCommandSeries,
    RolloutResult,
)
from grape_param_estim.plant.actuator import (
    ActuatorCalibrationIdentity,
    FirstOrderActuatorBackend,
)
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    CALIBRATED_RIGID_BODY_MODEL_ID,
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    ActuatorParameters,
    EpisodeNuisance,
    PlantHypothesis,
    PlantParameters,
)
from grape_param_estim.plant.disturbance import (
    ConstantDisturbance,
    EffectiveAccelerationDisturbance,
)
from grape_param_estim.plant.rigid_body import (
    EffectiveRigidBodyPlantBackend,
    RigidBodyPlantBackend,
)
from grape_param_estim.plant.sensor import RigidBodyObservationBackend


def _grid_values(grids: Any, name: str) -> np.ndarray:
    value = getattr(grids, name)
    return np.asarray(getattr(value, "timestamps", value), dtype=float).reshape(-1)


def _parameters(
    hypothesis: PlantHypothesis,
    actuator_calibration_identity: Optional[
        ActuatorCalibrationIdentity
    ] = None,
) -> Tuple[PlantParameters, ActuatorParameters]:
    if hypothesis.model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID:
        plant = PlantParameters.effective(hypothesis.plant_parameters)
        actuator = ActuatorParameters(
            model_id="uncalibrated_first_order_gimbal_actuator_v1",
            values=hypothesis.actuator_parameters,
            names=ACTUATOR_PARAMETER_NAMES,
            calibrated_wrench=False,
        )
    elif hypothesis.model_id == CALIBRATED_RIGID_BODY_MODEL_ID:
        if not isinstance(
            actuator_calibration_identity, ActuatorCalibrationIdentity
        ):
            raise ValueError(
                "calibrated rigid-body rollout requires an independent "
                "actuator calibration identity"
            )
        plant = PlantParameters.calibrated_rigid_body(
            hypothesis.plant_parameters
        )
        actuator = ActuatorParameters(
            model_id=actuator_calibration_identity.actuator_model_id,
            values=hypothesis.actuator_parameters,
            names=ACTUATOR_PARAMETER_NAMES,
            calibrated_wrench=True,
            metadata={
                "actuator_calibration_sha256": (
                    actuator_calibration_identity.artifact_sha256
                ),
                "actuator_calibration_schema": (
                    actuator_calibration_identity.schema
                ),
            },
        )
    else:
        raise ValueError(
            "open-loop rollout does not support model {}".format(
                hypothesis.model_id
            )
        )
    return plant, actuator


def _configure_disturbance(
    plant: Any,
    hypothesis: PlantHypothesis,
    nuisance: EpisodeNuisance,
) -> None:
    """Apply an episode-local disturbance in profile-explicit coordinates."""

    shared = np.asarray(hypothesis.disturbance_parameters, dtype=float)
    episode = np.asarray(nuisance.disturbance_parameters, dtype=float)
    if shared.size and np.any(shared != 0.0):
        raise ValueError(
            "disturbance is episode-specific and must not be stored in "
            "the shared plant hypothesis"
        )
    if episode.size == 0:
        episode = np.zeros(6)
    if episode.shape != (6,) or not np.all(np.isfinite(episode)):
        raise ValueError(
            "episode disturbance must contain six finite values"
        )
    expected_model_id = (
        "effective_constant_acceleration_disturbance_v1"
        if hypothesis.model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID
        else "constant_wrench_disturbance_v1"
    )
    if (
        np.any(episode != 0.0)
        and nuisance.disturbance_model_id != expected_model_id
    ):
        raise ValueError(
            "episode disturbance units/model do not match the plant profile"
        )
    if not hasattr(plant, "set_disturbance"):
        if np.any(episode != 0.0):
            raise TypeError(
                "plant backend cannot represent episode disturbance"
            )
        return
    disturbance = (
        EffectiveAccelerationDisturbance.from_parameters(episode)
        if hypothesis.model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID
        else ConstantDisturbance.from_parameters(episode)
    )
    plant.set_disturbance(disturbance)


def _reset_plant(plant: Any, initial: np.ndarray, stamp: float) -> None:
    """Reset a backend while retaining compatibility with the plan protocol."""

    parameters = inspect.signature(plant.reset).parameters
    if "stamp" in parameters:
        plant.reset(initial, stamp=float(stamp))
    else:
        plant.reset(initial)
        if not np.isclose(
            float(plant.state.stamp), float(stamp), rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "plant backend cannot represent the rollout start timestamp"
            )


def _step_actuator(
    actuator: Any,
    command: CommandSample,
    parameters: ActuatorParameters,
    dt: float,
    evaluation_stamp: float,
):
    """Advance an actuator without requiring extensions to §11.2's protocol."""

    step_parameters = inspect.signature(actuator.step).parameters
    if "evaluation_stamp" in step_parameters:
        return actuator.step(
            command,
            parameters,
            dt,
            evaluation_stamp=float(evaluation_stamp),
        )
    evaluated = CommandSample(
        stamp=float(evaluation_stamp),
        base_thrust=command.base_thrust,
        gimbal_angle=command.gimbal_angle,
        generalized_wrench=command.generalized_wrench,
        events=command.events,
        saturated=command.saturated,
    )
    return actuator.step(evaluated, parameters, dt)


class OpenLoopForwardModel:
    """Integrate recorded commands without constructing a controller backend."""

    mode = "open_loop_plant_identification"

    def __init__(
        self,
        actuator_factory: Callable[[], Any] = FirstOrderActuatorBackend,
        effective_plant_factory: Callable[[], Any] = EffectiveRigidBodyPlantBackend,
        calibrated_plant_factory: Callable[[], Any] = RigidBodyPlantBackend,
        observation_backend: Optional[Any] = None,
        actuator_calibration_identity: Optional[
            ActuatorCalibrationIdentity
        ] = None,
    ) -> None:
        for name, factory in (
            ("actuator_factory", actuator_factory),
            ("effective_plant_factory", effective_plant_factory),
            ("calibrated_plant_factory", calibrated_plant_factory),
        ):
            if not callable(factory):
                raise TypeError("{} must be callable".format(name))
        self.actuator_factory = actuator_factory
        self.effective_plant_factory = effective_plant_factory
        self.calibrated_plant_factory = calibrated_plant_factory
        if (
            actuator_calibration_identity is not None
            and not isinstance(
                actuator_calibration_identity,
                ActuatorCalibrationIdentity,
            )
        ):
            raise TypeError(
                "actuator_calibration_identity has the wrong type"
            )
        self.actuator_calibration_identity = (
            actuator_calibration_identity
        )
        self.observation_backend = (
            RigidBodyObservationBackend()
            if observation_backend is None
            else observation_backend
        )

    def run(
        self,
        commands: RecordedCommandSeries,
        hypothesis: PlantHypothesis,
        nuisance: EpisodeNuisance,
        grids: Any,
    ) -> RolloutResult:
        if not isinstance(commands, RecordedCommandSeries):
            raise TypeError("commands must be a RecordedCommandSeries")
        if not isinstance(hypothesis, PlantHypothesis):
            raise TypeError("hypothesis must be PlantHypothesis")
        if not isinstance(nuisance, EpisodeNuisance):
            raise TypeError("nuisance must be EpisodeNuisance")
        integration = _grid_values(grids, "plant_integration_grid")
        likelihood = _grid_values(grids, "likelihood_grid")
        if (
            integration.size < 2
            or np.any(np.diff(integration) <= 0.0)
            or integration[0] < commands.timestamps[0] - 1.0e-12
        ):
            raise ValueError("invalid integration grid for recorded commands")
        command_events = commands.timestamps[
            (commands.timestamps > integration[0] + 1.0e-12)
            & (commands.timestamps < integration[-1] - 1.0e-12)
        ]
        if command_events.size:
            # Command changes are integration events.  Including them avoids
            # applying a future ZOH command over the interval that precedes
            # its recorded timestamp.
            integration = np.unique(
                np.concatenate((integration, command_events))
            )

        plant_parameters, actuator_parameters = _parameters(
            hypothesis, self.actuator_calibration_identity
        )
        actuator = self.actuator_factory()
        plant = (
            self.effective_plant_factory()
            if hypothesis.model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID
            else self.calibrated_plant_factory()
        )
        _configure_disturbance(plant, hypothesis, nuisance)
        actuator.reset(nuisance.initial_actuator_state)
        initial = np.array(nuisance.initial_plant_state, copy=True)
        _reset_plant(plant, initial, float(integration[0]))

        states = [plant.state]
        predictions = [
            self.observation_backend.predict(
                plant.state, nuisance.sensor_bias
            )
        ]
        used_commands = []
        wrenches = []
        events = []
        for index in range(1, integration.size):
            stamp = float(integration[index])
            interval_start = float(integration[index - 1])
            delta = stamp - interval_start
            source_command = commands.causal_sample(interval_start)
            command = CommandSample(
                stamp=source_command.stamp,
                base_thrust=source_command.base_thrust,
                gimbal_angle=source_command.gimbal_angle,
                generalized_wrench=source_command.generalized_wrench,
                events=source_command.events,
                saturated=source_command.saturated,
            )
            realized = _step_actuator(
                actuator,
                command,
                actuator_parameters,
                delta,
                stamp,
            )
            state = plant.step(realized, plant_parameters, delta)
            used_commands.append(command)
            wrenches.append(realized)
            states.append(state)
            predictions.append(
                self.observation_backend.predict(state, nuisance.sensor_bias)
            )
            if realized.saturated:
                events.append(
                    {
                        "stamp": stamp,
                        "type": "actuator_saturation",
                        "integration_index": index,
                    }
                )

        return RolloutResult(
            mode=self.mode,
            model_id="{}/{}".format(self.mode, hypothesis.model_id),
            hypothesis=hypothesis,
            integration_timestamps=integration,
            plant_states=tuple(states),
            commands=tuple(used_commands),
            realized_wrenches=tuple(wrenches),
            predicted_observations=tuple(predictions),
            controller_tick_timestamps=np.empty(0),
            likelihood_timestamps=likelihood,
            events=tuple(events),
            used_recorded_commands=True,
            controller_fidelity=None,
        )


__all__ = ["OpenLoopForwardModel"]
