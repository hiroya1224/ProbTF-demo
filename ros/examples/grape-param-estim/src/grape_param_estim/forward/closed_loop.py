"""Exact-gated closed-loop controller/actuator/plant rollout."""

from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from typing import Any, Callable, Optional

import numpy as np

from grape_param_estim.controller.contracts import deep_thaw
from grape_param_estim.controller.replay_gate import (
    ExactClosedLoopGateReport,
)
from grape_param_estim.data.event_scheduler import EventGrid, EventScheduler
from grape_param_estim.forward.open_loop import (
    _configure_disturbance,
    _grid_values,
    _parameters,
    _reset_plant,
    _step_actuator,
)
from grape_param_estim.forward.rollout import CommandSample, RolloutResult
from grape_param_estim.plant.actuator import FirstOrderActuatorBackend
from grape_param_estim.plant.actuator import ActuatorCalibrationIdentity
from grape_param_estim.plant.parameters import (
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    EpisodeNuisance,
    PlantHypothesis,
)
from grape_param_estim.plant.rigid_body import (
    EffectiveRigidBodyPlantBackend,
    RigidBodyPlantBackend,
)
from grape_param_estim.plant.sensor import RigidBodyObservationBackend


class ClosedLoopGateError(RuntimeError):
    pass


_CLOSED_LOOP_EVENT_PRIORITY = (
    "plant_integration",
    "controller_tick",
    "likelihood",
)


def _fidelity(report: ExactClosedLoopGateReport) -> Optional[str]:
    return None if report.identity is None else report.identity.fidelity


def _closed_loop_scheduler(
    integration: np.ndarray,
    controller_ticks: np.ndarray,
    likelihood: np.ndarray,
) -> EventScheduler:
    """Build the single deterministic clock used by closed-loop rollout.

    A plant event precedes a controller event at the same timestamp so every
    non-initial controller tick sees feedback from the just-completed plant
    step.  The initial plant event only represents the already-reset state, so
    the first controller command is still available before the first interval.
    Likelihood events run last and never advance controller or plant state.
    """

    integration_events = set(float(value) for value in integration)
    scheduled_controller_ticks = tuple(
        float(value) for value in controller_ticks
    )
    if any(
        value not in integration_events
        for value in scheduled_controller_ticks
    ):
        raise ValueError(
            "controller ticks must be exact plant-integration events"
        )
    return EventScheduler(
        (
            EventGrid(
                "plant_integration",
                tuple(float(value) for value in integration),
            ),
            EventGrid("controller_tick", scheduled_controller_ticks),
            EventGrid(
                "likelihood",
                tuple(float(value) for value in likelihood),
            ),
        ),
        priority=_CLOSED_LOOP_EVENT_PRIORITY,
    )


def _command_from_output(output: Any, stamp: float) -> CommandSample:
    command = getattr(output, "command", None)
    if command is None:
        command = output
    base = getattr(command, "base_thrust", None)
    gimbal = getattr(command, "gimbal_angle", None)
    if base is None or gimbal is None:
        raise TypeError(
            "controller output must expose ControllerCommand base_thrust/gimbal_angle"
        )
    return CommandSample(
        stamp=stamp,
        base_thrust=base,
        gimbal_angle=gimbal,
        generalized_wrench=getattr(command, "generalized_wrench", None),
        events=tuple(getattr(output, "events", ())),
        saturated=bool(getattr(command, "saturated", False)),
    )


def _rotation_about_x(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        ),
        dtype=float,
    )


class _GimbalGeometryFeedback:
    """Map simulated one-DOF gimbals into the exact allocation geometry."""

    def __init__(self, snapshot: Any, initial_gimbal_angle: Any) -> None:
        nominal = getattr(snapshot, "nominal_geometry", None)
        if not isinstance(nominal, Mapping):
            raise TypeError(
                "closed-loop snapshot must expose nominal allocation geometry"
            )
        geometry = dict(deep_thaw(nominal))
        rotations = np.asarray(
            geometry.get("thrust_coordinate_rotations"),
            dtype=float,
        )
        initial = np.asarray(initial_gimbal_angle, dtype=float).reshape(-1)
        if (
            rotations.shape != (4, 3, 3)
            or initial.shape != (4,)
            or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(initial))
        ):
            raise ValueError(
                "closed-loop actuator feedback requires four finite "
                "one-DOF gimbals and four nominal thrust rotations"
            )
        self._geometry = geometry
        self._nominal_rotations = np.array(rotations, copy=True)
        self._initial_gimbal_angle = np.array(initial, copy=True)

    def allocation_geometry(self, gimbal_angle: Any) -> Mapping:
        current = np.asarray(gimbal_angle, dtype=float).reshape(-1)
        if current.shape != (4,) or not np.all(np.isfinite(current)):
            raise ValueError(
                "closed-loop actuator gimbal feedback must contain four "
                "finite angles"
            )
        rotations = np.stack(
            tuple(
                nominal
                @ _rotation_about_x(float(angle - initial))
                for nominal, angle, initial in zip(
                    self._nominal_rotations,
                    current,
                    self._initial_gimbal_angle,
                )
            )
        )
        geometry = dict(deep_thaw(self._geometry))
        geometry["thrust_coordinate_rotations"] = rotations.tolist()
        return geometry


def _actuator_feedback(actuator: Any) -> Any:
    callback = getattr(actuator, "feedback", None)
    if not callable(callback):
        raise TypeError(
            "closed-loop actuator backend must expose particle-local feedback"
        )
    result = callback()
    angles = np.asarray(
        getattr(result, "gimbal_angle", ()),
        dtype=float,
    ).reshape(-1)
    if angles.shape != (4,) or not np.all(np.isfinite(angles)):
        raise ValueError(
            "closed-loop actuator feedback must expose four finite gimbal angles"
        )
    return result


def _input_with_feedback(
    item: Any,
    prediction: Any,
    actuator_feedback: Any,
    allocation_geometry: Mapping,
    stamp: float,
    dt: float,
) -> Any:
    joint_positions = tuple(
        float(value) for value in actuator_feedback.gimbal_angle
    )
    changes = {
        "stamp": float(stamp),
        "dt": float(dt),
        "position": prediction.position_world,
        "velocity": prediction.velocity_world,
        "orientation": prediction.orientation_xyzw,
        "angular_velocity": prediction.angular_velocity_body,
        "joint_positions": joint_positions,
        "allocation_geometry": allocation_geometry,
    }
    # The contract uses an orientation matrix, while compatibility fixtures
    # may carry a quaternion field.  Populate only fields the selected
    # dataclass actually declares.
    if is_dataclass(item):
        fields = getattr(item, "__dataclass_fields__", {})
        missing = {
            "joint_positions",
            "allocation_geometry",
        }.difference(fields)
        if missing:
            raise TypeError(
                "closed-loop controller input cannot receive simulated "
                "actuator feedback: missing {}".format(
                    ", ".join(sorted(missing))
                )
            )
        selected = {key: value for key, value in changes.items() if key in fields}
        if "orientation" in fields or "current_rpy" in fields:
            from scipy.spatial.transform import Rotation

            rotation = Rotation.from_quat(
                prediction.orientation_xyzw
            )
            if "orientation" in fields:
                selected["orientation"] = rotation.as_matrix()
            if "current_rpy" in fields:
                selected["current_rpy"] = rotation.as_euler("xyz")
        return replace(item, **selected)
    callback = getattr(item, "with_feedback", None)
    if callable(callback):
        return callback(
            prediction,
            actuator_feedback=actuator_feedback,
            allocation_geometry=allocation_geometry,
            stamp=stamp,
            dt=dt,
        )
    raise TypeError("controller fixture input cannot receive simulated feedback")


class ClosedLoopForwardModel:
    """Roll out simulated feedback; recorded commands are explicitly rejected."""

    mode = "closed_loop_plant_identification"

    def __init__(
        self,
        controller_backend_factory: Callable[[], Any],
        factual_replay_report: ExactClosedLoopGateReport,
        actuator_factory: Callable[[], Any] = FirstOrderActuatorBackend,
        effective_plant_factory: Callable[[], Any] = EffectiveRigidBodyPlantBackend,
        calibrated_plant_factory: Callable[[], Any] = RigidBodyPlantBackend,
        observation_backend: Optional[Any] = None,
        actuator_calibration_identity: Optional[
            ActuatorCalibrationIdentity
        ] = None,
    ) -> None:
        if not callable(controller_backend_factory):
            raise TypeError("controller_backend_factory must be callable")
        if not isinstance(
            factual_replay_report, ExactClosedLoopGateReport
        ):
            raise TypeError(
                "factual_replay_report must be ExactClosedLoopGateReport"
            )
        if not factual_replay_report.passed:
            raise ClosedLoopGateError(
                "closed-loop rollout requires a passing factual replay gate"
            )
        fidelity = _fidelity(factual_replay_report)
        if fidelity not in ("pc_exact", "pc_mcu_exact"):
            raise ClosedLoopGateError(
                "closed-loop exact rollout requires pc_exact or pc_mcu_exact fidelity"
            )
        self.controller_backend_factory = controller_backend_factory
        self.factual_replay_report = factual_replay_report
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
        fixture: Any,
        snapshot: Any,
        hypothesis: PlantHypothesis,
        nuisance: EpisodeNuisance,
        grids: Any,
        recorded_commands: Any = None,
    ) -> RolloutResult:
        if recorded_commands is not None:
            raise ValueError(
                "closed-loop rollout must recompute commands from simulated feedback"
            )
        if nuisance.controller_state is None:
            raise ClosedLoopGateError(
                "unknown controller state may not be silently zero-initialized"
            )
        integration = _grid_values(grids, "plant_integration_grid")
        controller_ticks = _grid_values(grids, "controller_tick_grid")
        likelihood = _grid_values(grids, "likelihood_grid")
        if (
            integration.size < 2
            or controller_ticks.size == 0
            or np.any(np.diff(integration) <= 0.0)
            or np.any(np.diff(controller_ticks) <= 0.0)
        ):
            raise ValueError("closed-loop grids lack integration/controller ticks")
        if not np.all(np.isin(controller_ticks, integration)):
            raise ValueError(
                "controller ticks must be exact plant-integration events"
            )
        if controller_ticks[0] != integration[0]:
            raise ClosedLoopGateError(
                "integration must begin at the first controller tick"
            )

        plant_parameters, actuator_parameters = _parameters(
            hypothesis, self.actuator_calibration_identity
        )
        controller = self.controller_backend_factory()
        actuator = self.actuator_factory()
        plant = (
            self.effective_plant_factory()
            if hypothesis.model_id == EFFECTIVE_CLOSED_LOOP_MODEL_ID
            else self.calibrated_plant_factory()
        )
        _configure_disturbance(plant, hypothesis, nuisance)
        if not callable(getattr(controller, "reset", None)) or not callable(
            getattr(controller, "step", None)
        ):
            raise TypeError("controller backend does not satisfy the core contract")
        controller.reset(snapshot, nuisance.controller_state)
        actuator.reset(nuisance.initial_actuator_state)
        initial_actuator_feedback = _actuator_feedback(actuator)
        gimbal_geometry = _GimbalGeometryFeedback(
            snapshot,
            initial_actuator_feedback.gimbal_angle,
        )
        _reset_plant(
            plant,
            nuisance.initial_plant_state,
            float(integration[0]),
        )

        fixture_inputs = tuple(getattr(fixture, "controller_inputs"))
        if not fixture_inputs:
            raise ValueError("controller fixture has no inputs")
        fixture_times = np.asarray(
            [float(getattr(item, "stamp")) for item in fixture_inputs]
        )
        if (
            not np.all(np.isfinite(fixture_times))
            or np.any(np.diff(fixture_times) <= 0.0)
        ):
            raise ValueError(
                "controller fixture timestamps must be finite and increasing"
            )
        states = [plant.state]
        predictions = [
            self.observation_backend.predict(plant.state, nuisance.sensor_bias)
        ]
        commands = []
        wrenches = []
        events = []
        scheduler = _closed_loop_scheduler(
            integration, controller_ticks, likelihood
        )

        def controller_step(
            stamp: float, prediction: Any
        ) -> CommandSample:
            exact_indexes = np.flatnonzero(fixture_times == stamp)
            if exact_indexes.size != 1:
                raise ValueError(
                    "controller fixture must contain exactly one input at "
                    "tick {:.17g}".format(stamp)
                )
            fixture_index = int(exact_indexes[0])
            tick_dt = float(
                getattr(fixture_inputs[fixture_index], "dt")
            )
            if not np.isfinite(tick_dt) or tick_dt < 0.0:
                raise ValueError(
                    "controller fixture dt must be finite and non-negative"
                )
            actuator_feedback = _actuator_feedback(actuator)
            allocation_geometry = gimbal_geometry.allocation_geometry(
                actuator_feedback.gimbal_angle
            )
            core_input = _input_with_feedback(
                fixture_inputs[fixture_index],
                prediction,
                actuator_feedback,
                allocation_geometry,
                stamp,
                tick_dt,
            )
            output = controller.step(core_input)
            command = _command_from_output(output, stamp)
            commands.append(command)
            for event in getattr(output, "events", ()):
                events.append(
                    {
                        "stamp": stamp,
                        "type": "controller_event",
                        "code": int(event),
                    }
                )
            return command

        current_command = None
        previous_integration = float(integration[0])
        for scheduled_event in scheduler:
            stamp = float(scheduled_event.time)
            if scheduled_event.grid_name == "plant_integration":
                if scheduled_event.grid_index == 0:
                    continue
                if current_command is None:
                    raise ClosedLoopGateError(
                        "first controller tick must precede plant integration"
                    )
                delta = stamp - previous_integration
                realized = _step_actuator(
                    actuator,
                    current_command,
                    actuator_parameters,
                    delta,
                    stamp,
                )
                state = plant.step(realized, plant_parameters, delta)
                wrenches.append(realized)
                states.append(state)
                predictions.append(
                    self.observation_backend.predict(
                        state, nuisance.sensor_bias
                    )
                )
                previous_integration = stamp
            elif scheduled_event.grid_name == "controller_tick":
                current_command = controller_step(
                    stamp,
                    predictions[-1],
                )
            elif scheduled_event.grid_name == "likelihood":
                # Scoring consumes the returned prediction sequence.  Keeping
                # this event explicit prevents a likelihood/report rate from
                # ever becoming an accidental PID or integration clock.
                continue

        return RolloutResult(
            mode=self.mode,
            model_id="{}/{}".format(self.mode, hypothesis.model_id),
            hypothesis=hypothesis,
            integration_timestamps=integration,
            plant_states=tuple(states),
            commands=tuple(commands),
            realized_wrenches=tuple(wrenches),
            predicted_observations=tuple(predictions),
            controller_tick_timestamps=controller_ticks,
            likelihood_timestamps=likelihood,
            events=tuple(events),
            used_recorded_commands=False,
            controller_fidelity=_fidelity(self.factual_replay_report),
        )


__all__ = ["ClosedLoopForwardModel", "ClosedLoopGateError"]
