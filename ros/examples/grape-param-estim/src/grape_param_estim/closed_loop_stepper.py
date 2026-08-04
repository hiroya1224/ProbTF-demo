"""One-observation-interval stateful Grape closed-loop propagation."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import (
    FullSixDofPlant,
    _advance_plant_and_actuators,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    ReferenceState,
    RigidBodyState,
)
from grape_param_estim.timing import ZeroOrderHoldCommandHistory


def _finite_interval_discrepancy(
    value: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    if value is None:
        return None
    discrepancy = np.asarray(value, dtype=float)
    if discrepancy.shape != (6,) or np.any(~np.isfinite(discrepancy)):
        raise ValueError(
            "interval model discrepancy wrench must contain six finite values"
        )
    return discrepancy.copy()


@dataclass(frozen=True)
class ClosedLoopStepperState:
    """Dynamic state at one observation time.

    The delayed command history is causal auxiliary state owned by the
    corresponding :class:`ClosedLoopStepper`.
    """

    time: float
    rigid_body_state: RigidBodyState
    controller_state: ControllerState
    actuator_state: Optional[ActuatorState]

    def __post_init__(self) -> None:
        selected_time = float(self.time)
        if not np.isfinite(selected_time):
            raise ValueError("stepper state time must be finite")
        if not isinstance(self.rigid_body_state, RigidBodyState):
            raise TypeError("rigid_body_state must be RigidBodyState")
        if not isinstance(self.controller_state, ControllerState):
            raise TypeError("controller_state must be ControllerState")
        if self.actuator_state is not None and not isinstance(
            self.actuator_state, ActuatorState
        ):
            raise TypeError("actuator_state must be ActuatorState or None")
        object.__setattr__(self, "time", selected_time)


@dataclass(frozen=True)
class ClosedLoopStepSample:
    """The left-end sample emitted while one interval is propagated."""

    time: float
    rigid_body_state: RigidBodyState
    controller_state: ControllerState
    command: ActuatorCommand
    actuator_state: ActuatorState
    body_wrench: np.ndarray

    def __post_init__(self) -> None:
        selected_time = float(self.time)
        if not np.isfinite(selected_time):
            raise ValueError("closed-loop sample time must be finite")
        for name, expected in (
            ("rigid_body_state", RigidBodyState),
            ("controller_state", ControllerState),
            ("command", ActuatorCommand),
            ("actuator_state", ActuatorState),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError("{} has the wrong type".format(name))
        wrench = np.asarray(self.body_wrench, dtype=float)
        if wrench.shape != (6,) or np.any(~np.isfinite(wrench)):
            raise ValueError("body_wrench must contain six finite values")
        object.__setattr__(self, "time", selected_time)
        object.__setattr__(self, "body_wrench", wrench.copy())


class ClosedLoopStepper:
    """Advance one causal closed-loop forecast across time intervals.

    Command history remains continuous because actuator delay can span one or
    more integration intervals.
    """

    def __init__(
        self,
        controller: GrapeController,
        plant: FullSixDofPlant,
        actuator_parameters: ActuatorParameters,
        initial_state: ClosedLoopStepperState,
    ) -> None:
        if not isinstance(controller, GrapeController):
            raise TypeError("controller must be GrapeController")
        if not isinstance(plant, FullSixDofPlant):
            raise TypeError("plant must be FullSixDofPlant")
        if not isinstance(actuator_parameters, ActuatorParameters):
            raise TypeError("actuator_parameters must be ActuatorParameters")
        if not isinstance(initial_state, ClosedLoopStepperState):
            raise TypeError("initial_state must be ClosedLoopStepperState")
        self._controller = controller
        self._plant = plant
        self._actuator_parameters = actuator_parameters
        self._state = initial_state
        self._command_history = ZeroOrderHoldCommandHistory[ActuatorCommand](
            actuator_parameters.delay
        )
        self._terminal = False

    @property
    def state(self) -> ClosedLoopStepperState:
        return self._state

    @property
    def command_issue_times(self) -> np.ndarray:
        """Return a copy of every retained command issue time."""

        return self._command_history.issue_times

    @property
    def command_history_commands(self) -> Tuple[ActuatorCommand, ...]:
        """Return defensive copies of this forecast's causal commands."""

        return tuple(
            ActuatorCommand(
                command.thrust,
                command.gimbal_angle,
                command.virtual_force,
                command.desired_acceleration,
            )
            for command in self._command_history.values
        )

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def delayed_command_at(self, plant_time: float) -> ActuatorCommand:
        """Return the causally delayed, zero-order-held actuator command."""

        return self._command_history.value_at(plant_time)

    def advance_interval(
        self,
        end_time: float,
        reference: ReferenceState,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> ClosedLoopStepSample:
        """Emit the current sample and propagate to ``end_time``."""

        self._require_active()
        selected_end = float(end_time)
        if (
            not np.isfinite(selected_end)
            or selected_end <= self._state.time
        ):
            raise ValueError("interval end time must be finite and increasing")
        if not isinstance(reference, ReferenceState):
            raise TypeError("reference must be ReferenceState")
        discrepancy = _finite_interval_discrepancy(
            interval_model_discrepancy_wrench
        )
        time_step = selected_end - self._state.time
        sample, next_controller_state = self._issue_sample(
            reference, time_step, discrepancy
        )
        next_rigid_body, next_actuators = _advance_plant_and_actuators(
            start_time=sample.time,
            end_time=selected_end,
            state=sample.rigid_body_state,
            actuator_state=sample.actuator_state,
            command_history=self._command_history,
            actuator_parameters=self._actuator_parameters,
            plant=self._plant,
            interval_model_discrepancy_wrench=discrepancy,
        )
        self._state = ClosedLoopStepperState(
            time=selected_end,
            rigid_body_state=next_rigid_body,
            controller_state=next_controller_state,
            actuator_state=next_actuators,
        )
        return sample

    def terminal_sample(
        self,
        reference: ReferenceState,
        controller_time_step: float,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> ClosedLoopStepSample:
        """Emit the final sample without another plant transition."""

        self._require_active()
        selected_step = float(controller_time_step)
        if not np.isfinite(selected_step) or selected_step <= 0.0:
            raise ValueError("terminal controller time step must be positive")
        if not isinstance(reference, ReferenceState):
            raise TypeError("reference must be ReferenceState")
        discrepancy = _finite_interval_discrepancy(
            interval_model_discrepancy_wrench
        )
        sample, _unused_next_controller_state = self._issue_sample(
            reference, selected_step, discrepancy
        )
        # An initially absent actuator snapshot is initialised when the command
        # is issued, even if this is a one-sample terminal-only use.
        if self._state.actuator_state is None:
            self._state = ClosedLoopStepperState(
                time=self._state.time,
                rigid_body_state=self._state.rigid_body_state,
                controller_state=self._state.controller_state,
                actuator_state=sample.actuator_state,
            )
        self._terminal = True
        return sample

    def _require_active(self) -> None:
        if self._terminal:
            raise RuntimeError("closed-loop stepper is already terminal")

    def _issue_sample(
        self,
        reference: ReferenceState,
        time_step: float,
        interval_model_discrepancy_wrench: Optional[np.ndarray],
    ):
        current = self._state
        command, next_controller_state = self._controller.step(
            current.rigid_body_state,
            reference,
            current.controller_state,
            time_step,
            None
            if current.actuator_state is None
            else current.actuator_state.gimbal_angle,
        )
        self._command_history.append(current.time, command)
        actuator_state = current.actuator_state
        if actuator_state is None:
            limits = self._actuator_parameters
            actuator_state = ActuatorState(
                np.clip(
                    command.thrust,
                    limits.minimum_thrust,
                    limits.maximum_thrust,
                ),
                np.clip(
                    command.gimbal_angle,
                    -limits.maximum_gimbal_angle,
                    limits.maximum_gimbal_angle,
                ),
            )
        body_wrench = self._plant.total_body_wrench(
            current.time,
            current.rigid_body_state,
            actuator_state,
            interval_model_discrepancy_wrench,
        )
        return (
            ClosedLoopStepSample(
                time=current.time,
                rigid_body_state=current.rigid_body_state,
                controller_state=current.controller_state,
                command=command,
                actuator_state=actuator_state,
                body_wrench=body_wrench,
            ),
            next_controller_state,
        )


__all__ = [
    "ClosedLoopStepSample",
    "ClosedLoopStepper",
    "ClosedLoopStepperState",
]
