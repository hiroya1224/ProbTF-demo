"""Persistent spawned workers for augmented member forecast intervals.

The parent process owns the authoritative, bounded command histories.  At
every observation boundary each worker receives the analysed vehicle model,
actuator delay, dynamic state, complete post-ETKF history snapshot, and
interval wrench.  It returns only the next dynamic state and the command
issued at the left endpoint.  No worker-local history can therefore become
undeclared filter state.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait as wait_connections
import signal
import traceback
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.closed_loop_stepper import (
    ClosedLoopStepper,
    ClosedLoopStepperState,
)
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.filter_state import GrapeFilterState
from grape_param_estim.progress import CancellationToken
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    GrapeGeometry,
    ReferenceState,
    VehicleParameters,
)


AUGMENTED_FORECAST_START_METHOD = "spawn"


class AugmentedForecastPoolError(RuntimeError):
    """A spawned augmented forecast worker failed its strict protocol."""


def _worker_main(
    connection,
    controller,
    geometry,
    initial_time,
    indexed_initial_members,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    steppers = {
        int(member): ClosedLoopStepper(
            controller=controller,
            plant=FullSixDofPlant(vehicle, geometry),
            actuator_parameters=actuator,
            initial_state=ClosedLoopStepperState(
                time=initial_time,
                rigid_body_state=state.rigid,
                controller_state=state.controller,
                actuator_state=state.actuator,
            ),
        )
        for member, state, vehicle, actuator in indexed_initial_members
    }
    try:
        while True:
            message = connection.recv()
            if not isinstance(message, tuple) or not message:
                raise RuntimeError("augmented forecast command is malformed")
            command = message[0]
            if command == "advance":
                _tag, start_time, end_time, reference, indexed_members = message
                advanced = []
                for (
                    member,
                    state,
                    vehicle,
                    actuator,
                    issue_times,
                    history_commands,
                    interval_wrench,
                ) in indexed_members:
                    stepper = steppers[int(member)]
                    if stepper.state.time != start_time:
                        raise RuntimeError(
                            "worker boundary time differs for member {}".format(
                                member
                            )
                        )
                    stepper.replace_static_model(
                        controller=controller,
                        plant=FullSixDofPlant(vehicle, geometry),
                        actuator_parameters=actuator,
                    )
                    stepper.replace_dynamic_state(
                        rigid_body_state=state.rigid,
                        controller_state=state.controller,
                        actuator_state=state.actuator,
                    )
                    stepper.replace_command_history_snapshot(
                        issue_times, history_commands
                    )
                    sample = stepper.advance_interval(
                        end_time, reference, interval_wrench
                    )
                    advanced.append(
                        (int(member), stepper.state, sample.command)
                    )
                connection.send(("advanced", tuple(advanced)))
            elif command == "close":
                connection.send(("closed",))
                return
            else:
                raise RuntimeError(
                    "unknown augmented forecast command {!r}".format(command)
                )
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(
                (
                    "error",
                    type(error).__name__,
                    str(error),
                    traceback.format_exc(),
                )
            )
        except BaseException:
            pass
    finally:
        connection.close()


def validated_forecast_worker_count(value, member_count: int) -> int:
    """Validate the public serial/parallel worker count and cap to members."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("forecast_workers must be an integer in [1, 256]")
    selected = int(value)
    if selected < 1 or selected > 256:
        raise ValueError("forecast_workers must be an integer in [1, 256]")
    if (
        isinstance(member_count, (bool, np.bool_))
        or not isinstance(member_count, (int, np.integer))
        or int(member_count) < 1
    ):
        raise ValueError("member_count must be a positive integer")
    return min(selected, member_count)


def _member_sequence(values, count: int, expected, name: str):
    selected = tuple(values)
    if len(selected) != count or any(
        not isinstance(value, expected) for value in selected
    ):
        raise ValueError(
            "{} must contain {} aligned {} values".format(
                name, count, expected.__name__
            )
        )
    return selected


class PersistentAugmentedForecastPool:
    """Keep stable member chunks alive across augmented forecast intervals."""

    def __init__(
        self,
        *,
        controller: GrapeController,
        geometry: GrapeGeometry,
        initial_time: float,
        initial_states: Sequence[GrapeFilterState],
        initial_vehicle_parameters: Sequence[VehicleParameters],
        initial_actuator_parameters: Sequence[ActuatorParameters],
        worker_count: int,
    ) -> None:
        if not isinstance(controller, GrapeController):
            raise TypeError("controller must be GrapeController")
        if not isinstance(geometry, GrapeGeometry):
            raise TypeError("geometry must be GrapeGeometry")
        selected_time = float(initial_time)
        if not np.isfinite(selected_time):
            raise ValueError("initial_time must be finite")
        states = tuple(initial_states)
        if not states or any(
            not isinstance(value, GrapeFilterState) for value in states
        ):
            raise ValueError(
                "initial_states must contain GrapeFilterState members"
            )
        member_count = len(states)
        vehicles = _member_sequence(
            initial_vehicle_parameters,
            member_count,
            VehicleParameters,
            "initial_vehicle_parameters",
        )
        actuators = _member_sequence(
            initial_actuator_parameters,
            member_count,
            ActuatorParameters,
            "initial_actuator_parameters",
        )
        workers = validated_forecast_worker_count(
            worker_count, member_count
        )
        if workers < 2:
            raise ValueError(
                "PersistentAugmentedForecastPool requires at least two workers"
            )
        self._member_count = member_count
        self._closed = False
        self._connections = []
        self._processes = []
        context = multiprocessing.get_context(
            AUGMENTED_FORECAST_START_METHOD
        )
        chunks = tuple(
            tuple(int(value) for value in chunk)
            for chunk in np.array_split(np.arange(member_count), workers)
            if len(chunk)
        )
        self._chunks = chunks
        try:
            for chunk in chunks:
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child_connection,
                        controller,
                        geometry,
                        selected_time,
                        tuple(
                            (
                                member,
                                states[member],
                                vehicles[member],
                                actuators[member],
                            )
                            for member in chunk
                        ),
                    ),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent_connection.close()
                    child_connection.close()
                    raise
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)
        except BaseException:
            self.abort()
            raise

    @property
    def worker_pids(self) -> Tuple[int, ...]:
        return tuple(
            int(process.pid)
            for process in self._processes
            if process.pid is not None
        )

    @property
    def worker_count(self) -> int:
        return len(self._processes)

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("augmented forecast pool is closed")

    def _receive_all(
        self,
        expected_tag: str,
        cancellation_token: Optional[CancellationToken],
    ):
        pending = set(self._connections)
        responses = []
        try:
            while pending:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                ready = wait_connections(tuple(pending), timeout=0.1)
                if not ready:
                    if any(
                        not process.is_alive()
                        for process in self._processes
                    ):
                        raise AugmentedForecastPoolError(
                            "augmented forecast worker exited unexpectedly"
                        )
                    continue
                for connection in ready:
                    try:
                        message = connection.recv()
                    except EOFError as error:
                        raise AugmentedForecastPoolError(
                            "augmented forecast worker closed its pipe"
                        ) from error
                    pending.remove(connection)
                    if not isinstance(message, tuple) or not message:
                        raise AugmentedForecastPoolError(
                            "augmented forecast response is malformed"
                        )
                    if message[0] == "error":
                        raise AugmentedForecastPoolError(
                            "{}: {}\n{}".format(
                                message[1], message[2], message[3]
                            )
                        )
                    if message[0] != expected_tag:
                        raise AugmentedForecastPoolError(
                            "expected {!r}, received {!r}".format(
                                expected_tag, message[0]
                            )
                        )
                    responses.append(
                        message[1:] if len(message) > 1 else ()
                    )
        except BaseException:
            self.abort()
            raise
        return tuple(responses)

    def advance_interval(
        self,
        *,
        start_time: float,
        end_time: float,
        reference: ReferenceState,
        analysis_states: Sequence[GrapeFilterState],
        vehicle_parameters: Sequence[VehicleParameters],
        actuator_parameters: Sequence[ActuatorParameters],
        command_issue_times: Sequence[Sequence[float]],
        command_histories: Sequence[Sequence[ActuatorCommand]],
        interval_wrench: np.ndarray,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Tuple[
        Tuple[ClosedLoopStepperState, ...], Tuple[ActuatorCommand, ...]
    ]:
        """Synchronise every member and advance one interval in stable order."""

        self._require_open()
        if cancellation_token is not None and not isinstance(
            cancellation_token, CancellationToken
        ):
            raise TypeError("cancellation_token must be CancellationToken")
        if cancellation_token is not None:
            try:
                cancellation_token.raise_if_cancelled()
            except BaseException:
                self.abort()
                raise
        selected_start = float(start_time)
        selected_end = float(end_time)
        if (
            not np.isfinite(selected_start)
            or not np.isfinite(selected_end)
            or selected_end <= selected_start
        ):
            raise ValueError("forecast interval times must be finite/increasing")
        if not isinstance(reference, ReferenceState):
            raise TypeError("reference must be ReferenceState")
        states = _member_sequence(
            analysis_states,
            self._member_count,
            GrapeFilterState,
            "analysis_states",
        )
        vehicles = _member_sequence(
            vehicle_parameters,
            self._member_count,
            VehicleParameters,
            "vehicle_parameters",
        )
        actuators = _member_sequence(
            actuator_parameters,
            self._member_count,
            ActuatorParameters,
            "actuator_parameters",
        )
        issue_times = tuple(
            np.asarray(value, dtype=float) for value in command_issue_times
        )
        histories = tuple(tuple(value) for value in command_histories)
        if len(issue_times) != self._member_count or len(histories) != (
            self._member_count
        ):
            raise ValueError("command history members are misaligned")
        for times, commands in zip(issue_times, histories):
            if (
                times.ndim != 1
                or np.any(~np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)
                or (times.size and times[-1] >= selected_start)
                or len(commands) != times.size
                or any(
                    not isinstance(command, ActuatorCommand)
                    for command in commands
                )
            ):
                raise ValueError("command history snapshots are invalid")
        if any(
            not np.array_equal(times, issue_times[0])
            for times in issue_times[1:]
        ):
            raise ValueError("command histories must share one issue-time grid")
        wrench = np.asarray(interval_wrench, dtype=float)
        if wrench.shape != (self._member_count, 6) or np.any(
            ~np.isfinite(wrench)
        ):
            raise ValueError("interval_wrench members are misaligned")

        try:
            for connection, chunk in zip(self._connections, self._chunks):
                connection.send(
                    (
                        "advance",
                        selected_start,
                        selected_end,
                        reference,
                        tuple(
                            (
                                member,
                                states[member],
                                vehicles[member],
                                actuators[member],
                                issue_times[member],
                                histories[member],
                                wrench[member],
                            )
                            for member in chunk
                        ),
                    )
                )
        except BaseException:
            self.abort()
            raise
        responses = self._receive_all("advanced", cancellation_token)
        by_member = {}
        try:
            for response in responses:
                if len(response) != 1:
                    raise AugmentedForecastPoolError(
                        "augmented worker response payload is malformed"
                    )
                for member, state, command in response[0]:
                    index = int(member)
                    if index in by_member:
                        raise AugmentedForecastPoolError(
                            "augmented worker returned a duplicate member"
                        )
                    if (
                        not isinstance(state, ClosedLoopStepperState)
                        or state.time != selected_end
                        or not isinstance(command, ActuatorCommand)
                    ):
                        raise AugmentedForecastPoolError(
                            "augmented worker returned invalid member values"
                        )
                    by_member[index] = (state, command)
            if set(by_member) != set(range(self._member_count)):
                raise AugmentedForecastPoolError(
                    "augmented forecast returned incomplete members"
                )
        except BaseException:
            self.abort()
            raise
        return (
            tuple(by_member[index][0] for index in range(self._member_count)),
            tuple(by_member[index][1] for index in range(self._member_count)),
        )

    def close(self) -> None:
        """Gracefully stop responsive workers and terminate stragglers."""

        if self._closed:
            return
        for connection in self._connections:
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        try:
            self._receive_all("closed", None)
        except BaseException:
            self.abort()
            return
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self._closed = True

    def abort(self) -> None:
        """Terminate all workers after cancellation or protocol failure."""

        if self._closed:
            return
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self._closed = True

    def __enter__(self) -> "PersistentAugmentedForecastPool":
        self._require_open()
        return self

    def __exit__(self, exception_type, _exception, _traceback) -> None:
        if exception_type is None:
            self.close()
        else:
            self.abort()

    def __del__(self):
        try:
            self.abort()
        except BaseException:
            pass


__all__ = [
    "AUGMENTED_FORECAST_START_METHOD",
    "AugmentedForecastPoolError",
    "PersistentAugmentedForecastPool",
    "validated_forecast_worker_count",
]
