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


def _finite_interval_residual(
    value: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    if value is None:
        return None
    residual = np.asarray(value, dtype=float)
    if residual.shape != (6,) or np.any(~np.isfinite(residual)):
        raise ValueError(
            "interval residual wrench must contain six finite values"
        )
    return residual.copy()


@dataclass(frozen=True)
class ClosedLoopStepperState:
    """Dynamic state at one observation time.

    The delayed command history is causal auxiliary state owned by the
    corresponding :class:`ClosedLoopStepper`.  It is deliberately not part of
    the numeric state replaced by an EnKF analysis.
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
    """Advance one causal closed-loop member across observation intervals.

    One instance belongs to one ensemble member.  In particular, the command
    history must not be reset at observation boundaries because a continuous
    actuator delay can span one or more observation periods.
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
        """Return defensive copies of the member's causal commands."""

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

    def replace_dynamic_state(
        self,
        *,
        rigid_body_state: RigidBodyState,
        controller_state: ControllerState,
        actuator_state: ActuatorState,
    ) -> ClosedLoopStepperState:
        """Apply an analysis state without clearing prior issued commands.

        Time, plant parameters, actuator delay, and the complete causal command
        history remain unchanged.  This is the explicit boundary intended for
        an EnKF analysis update between two forecast intervals.
        """

        self._require_active()
        replacement = ClosedLoopStepperState(
            time=self._state.time,
            rigid_body_state=rigid_body_state,
            controller_state=controller_state,
            actuator_state=actuator_state,
        )
        self._state = replacement
        return replacement

    def replace_command_history(
        self, commands: Sequence[ActuatorCommand]
    ) -> None:
        """Replace command values while retaining their causal issue times.

        A deterministic ensemble analysis mixes members.  Delayed commands
        are auxiliary dynamic state and must receive the same member-space
        analysis before the next forecast interval.  The common issue-time
        grid and the currently analysed delay remain unchanged.
        """

        self._require_active()
        selected = tuple(commands)
        issue_times = self._command_history.issue_times
        if len(selected) != issue_times.size or any(
            not isinstance(command, ActuatorCommand)
            for command in selected
        ):
            raise ValueError(
                "commands must align with the retained command history"
            )
        self.replace_command_history_snapshot(issue_times, selected)

    def replace_command_history_snapshot(
        self,
        issue_times: Sequence[float],
        commands: Sequence[ActuatorCommand],
    ) -> None:
        """Replace the complete bounded causal-history snapshot.

        Spawned forecast workers receive the parent process's authoritative
        post-analysis history at every observation boundary.  Replacing both
        issue times and command values prevents a worker's older, untrimmed
        history from becoming hidden dynamic state.
        """

        self._require_active()
        selected_times = np.asarray(issue_times, dtype=float)
        selected_commands = tuple(commands)
        if (
            selected_times.ndim != 1
            or np.any(~np.isfinite(selected_times))
            or np.any(np.diff(selected_times) <= 0.0)
            or (
                selected_times.size
                and selected_times[-1] >= self._state.time
            )
            or len(selected_commands) != selected_times.size
            or any(
                not isinstance(command, ActuatorCommand)
                for command in selected_commands
            )
        ):
            raise ValueError(
                "history snapshot must contain increasing past issue times "
                "and aligned commands"
            )
        replacement = ZeroOrderHoldCommandHistory[ActuatorCommand](
            self._command_history.constant_delay
        )
        for time, command in zip(selected_times, selected_commands):
            replacement.append(
                float(time),
                ActuatorCommand(
                    command.thrust,
                    command.gimbal_angle,
                    command.virtual_force,
                    command.desired_acceleration,
                ),
            )
        self._command_history = replacement

    def accept_external_interval_advance(
        self,
        next_state: ClosedLoopStepperState,
        issued_command: ActuatorCommand,
    ) -> ClosedLoopStepperState:
        """Record one worker-computed interval in the parent-side history.

        The expensive plant propagation may run in another process, but the
        parent remains authoritative for bounded command history and the next
        boundary time.  The command is recorded at the current left endpoint
        exactly as :meth:`advance_interval` would have done.
        """

        self._require_active()
        if not isinstance(next_state, ClosedLoopStepperState):
            raise TypeError("next_state must be ClosedLoopStepperState")
        if not isinstance(issued_command, ActuatorCommand):
            raise TypeError("issued_command must be ActuatorCommand")
        if (
            next_state.time <= self._state.time
            or next_state.actuator_state is None
        ):
            raise ValueError(
                "external interval result must advance time with actuator state"
            )
        self._command_history.append(self._state.time, issued_command)
        self._state = ClosedLoopStepperState(
            time=next_state.time,
            rigid_body_state=next_state.rigid_body_state,
            controller_state=next_state.controller_state,
            actuator_state=next_state.actuator_state,
        )
        return self._state

    def trim_command_history(
        self, current_time: float, maximum_delay: float
    ) -> None:
        """Discard commands that cannot affect any future bounded delay.

        One predecessor at or before ``current_time - maximum_delay`` is
        retained because zero-order hold may still select it at the left edge
        of the admissible delay window.
        """

        self._require_active()
        selected_time = float(current_time)
        selected_maximum = float(maximum_delay)
        if (
            not np.isfinite(selected_time)
            or not np.isfinite(selected_maximum)
            or selected_maximum <= 0.0
            or selected_time != self._state.time
        ):
            raise ValueError(
                "current time/max delay must match state and be positive"
            )
        issue_times = self._command_history.issue_times
        if issue_times.size < 2:
            return
        threshold = selected_time - selected_maximum
        first = max(
            int(np.searchsorted(issue_times, threshold, side="right") - 1),
            0,
        )
        if first == 0:
            return
        commands = self.command_history_commands[first:]
        replacement = ZeroOrderHoldCommandHistory[ActuatorCommand](
            self._command_history.constant_delay
        )
        for time, command in zip(issue_times[first:], commands):
            replacement.append(float(time), command)
        self._command_history = replacement

    def replace_static_model(
        self,
        *,
        controller: GrapeController,
        plant: FullSixDofPlant,
        actuator_parameters: ActuatorParameters,
    ) -> None:
        """Apply updated static members without resetting causal history.

        An augmented-state filter can update vehicle parameters and constant
        delay at an observation boundary.  The controller and plant must then
        use those analysed values for the next interval, while every command
        already issued by this member remains available to ``u(t - delay)``.
        """

        self._require_active()
        if not isinstance(controller, GrapeController):
            raise TypeError("controller must be GrapeController")
        if not isinstance(plant, FullSixDofPlant):
            raise TypeError("plant must be FullSixDofPlant")
        if not isinstance(actuator_parameters, ActuatorParameters):
            raise TypeError(
                "actuator_parameters must be ActuatorParameters"
            )
        self._controller = controller
        self._plant = plant
        self._actuator_parameters = actuator_parameters
        self._command_history.constant_delay = actuator_parameters.delay

    def advance_interval(
        self,
        end_time: float,
        reference: ReferenceState,
        interval_residual_wrench: Optional[Sequence[float]] = None,
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
        residual = _finite_interval_residual(interval_residual_wrench)
        time_step = selected_end - self._state.time
        sample, next_controller_state = self._issue_sample(
            reference, time_step, residual
        )
        next_rigid_body, next_actuators = _advance_plant_and_actuators(
            start_time=sample.time,
            end_time=selected_end,
            state=sample.rigid_body_state,
            actuator_state=sample.actuator_state,
            command_history=self._command_history,
            actuator_parameters=self._actuator_parameters,
            plant=self._plant,
            interval_residual_wrench=residual,
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
        interval_residual_wrench: Optional[Sequence[float]] = None,
    ) -> ClosedLoopStepSample:
        """Emit the final sample without another plant transition."""

        self._require_active()
        selected_step = float(controller_time_step)
        if not np.isfinite(selected_step) or selected_step <= 0.0:
            raise ValueError("terminal controller time step must be positive")
        if not isinstance(reference, ReferenceState):
            raise TypeError("reference must be ReferenceState")
        residual = _finite_interval_residual(interval_residual_wrench)
        sample, _unused_next_controller_state = self._issue_sample(
            reference, selected_step, residual
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
        interval_residual_wrench: Optional[np.ndarray],
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
            interval_residual_wrench,
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
