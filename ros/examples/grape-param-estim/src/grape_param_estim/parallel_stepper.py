"""Persistent process workers for member-wise closed-loop intervals."""

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
from grape_param_estim.system import ActuatorParameters, ReferenceState


PARALLEL_STEPPER_START_METHOD = "spawn"


class ParallelStepperError(RuntimeError):
    """A persistent interval worker failed or violated its protocol."""


def _worker_main(
    connection,
    controller,
    plant,
    actuator_parameters,
    initial_time,
    indexed_initial_states,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    steppers = {
        int(member): ClosedLoopStepper(
            controller=controller,
            plant=plant,
            actuator_parameters=actuator_parameters,
            initial_state=ClosedLoopStepperState(
                time=initial_time,
                rigid_body_state=state.rigid,
                controller_state=state.controller,
                actuator_state=state.actuator,
            ),
        )
        for member, state in indexed_initial_states
    }
    try:
        while True:
            message = connection.recv()
            if not isinstance(message, tuple) or not message:
                raise RuntimeError("parallel stepper command is malformed")
            command = message[0]
            if command == "advance":
                _tag, end_time, reference, indexed_states = message
                advanced = []
                for member, state, interval_wrench in indexed_states:
                    stepper = steppers[int(member)]
                    stepper.replace_dynamic_state(
                        rigid_body_state=state.rigid,
                        controller_state=state.controller,
                        actuator_state=state.actuator,
                    )
                    stepper.advance_interval(
                        end_time, reference, interval_wrench
                    )
                    advanced.append((int(member), stepper.state))
                connection.send(("advanced", tuple(advanced)))
            elif command == "histories":
                connection.send(
                    (
                        "histories",
                        tuple(
                            (member, stepper.command_issue_times)
                            for member, stepper in sorted(steppers.items())
                        ),
                    )
                )
            elif command == "close":
                connection.send(("closed",))
                return
            else:
                raise RuntimeError(
                    "unknown parallel stepper command {!r}".format(command)
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


def _validated_worker_count(value, member_count: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("worker_count must be an integer in [1, 256]")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "worker_count must be an integer in [1, 256]"
        ) from error
    if result != value or result < 1 or result > 256:
        raise ValueError("worker_count must be an integer in [1, 256]")
    return min(result, member_count)


class PersistentParallelSteppers:
    """Keep one causal stepper per member in stable spawned worker chunks."""

    def __init__(
        self,
        *,
        controller: GrapeController,
        plant: FullSixDofPlant,
        actuator_parameters: ActuatorParameters,
        initial_time: float,
        initial_states: Sequence[GrapeFilterState],
        worker_count: int,
    ) -> None:
        if not isinstance(controller, GrapeController):
            raise TypeError("controller must be GrapeController")
        if not isinstance(plant, FullSixDofPlant):
            raise TypeError("plant must be FullSixDofPlant")
        if not isinstance(actuator_parameters, ActuatorParameters):
            raise TypeError(
                "actuator_parameters must be ActuatorParameters"
            )
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
        workers = _validated_worker_count(worker_count, len(states))
        if workers < 2:
            raise ValueError(
                "PersistentParallelSteppers requires at least two workers"
            )
        self._member_count = len(states)
        self._closed = False
        self._connections = []
        self._processes = []
        context = multiprocessing.get_context(PARALLEL_STEPPER_START_METHOD)
        chunks = tuple(
            tuple(int(value) for value in chunk)
            for chunk in np.array_split(np.arange(len(states)), workers)
            if len(chunk)
        )
        self._chunks = chunks
        try:
            for chunk in chunks:
                parent_connection, child_connection = context.Pipe(
                    duplex=True
                )
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child_connection,
                        controller,
                        plant,
                        actuator_parameters,
                        selected_time,
                        tuple((member, states[member]) for member in chunk),
                    ),
                    daemon=True,
                )
                process.start()
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
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("parallel steppers are closed")

    def _receive_all(
        self,
        expected_tag: str,
        cancellation_token: Optional[CancellationToken],
    ):
        pending = set(self._connections)
        results = []
        try:
            while pending:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                ready = wait_connections(tuple(pending), timeout=0.1)
                if not ready:
                    dead = [
                        process
                        for process in self._processes
                        if not process.is_alive()
                    ]
                    if dead:
                        raise ParallelStepperError(
                            "parallel stepper worker exited unexpectedly"
                        )
                    continue
                for connection in ready:
                    try:
                        message = connection.recv()
                    except EOFError as error:
                        raise ParallelStepperError(
                            "parallel stepper worker closed its pipe"
                        ) from error
                    pending.remove(connection)
                    if not isinstance(message, tuple) or not message:
                        raise ParallelStepperError(
                            "parallel stepper response is malformed"
                        )
                    if message[0] == "error":
                        raise ParallelStepperError(
                            "{}: {}\n{}".format(
                                message[1], message[2], message[3]
                            )
                        )
                    if message[0] != expected_tag:
                        raise ParallelStepperError(
                            "expected {!r}, received {!r}".format(
                                expected_tag, message[0]
                            )
                        )
                    results.append(message[1:] if len(message) > 1 else ())
        except BaseException:
            self.abort()
            raise
        return tuple(results)

    def advance_interval(
        self,
        *,
        end_time: float,
        reference: ReferenceState,
        analysis_states: Sequence[GrapeFilterState],
        interval_wrench: np.ndarray,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Tuple[ClosedLoopStepperState, ...]:
        """Apply analysis states and propagate all members once."""

        self._require_open()
        states = tuple(analysis_states)
        wrench = np.asarray(interval_wrench, dtype=float)
        if (
            len(states) != self._member_count
            or any(not isinstance(value, GrapeFilterState) for value in states)
            or wrench.shape != (self._member_count, 6)
            or np.any(~np.isfinite(wrench))
        ):
            raise ValueError("parallel interval members are misaligned")
        for connection, chunk in zip(self._connections, self._chunks):
            connection.send(
                (
                    "advance",
                    float(end_time),
                    reference,
                    tuple(
                        (int(member), states[int(member)], wrench[int(member)])
                        for member in chunk
                    ),
                )
            )
        responses = self._receive_all("advanced", cancellation_token)
        by_member = {}
        for response in responses:
            for member, state in response[0]:
                by_member[int(member)] = state
        if set(by_member) != set(range(self._member_count)):
            self.abort()
            raise ParallelStepperError(
                "parallel interval returned incomplete members"
            )
        return tuple(by_member[member] for member in range(self._member_count))

    def command_issue_times(
        self,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Return a detached issue-time history for every member."""

        self._require_open()
        for connection in self._connections:
            connection.send(("histories",))
        responses = self._receive_all("histories", cancellation_token)
        by_member = {}
        for response in responses:
            for member, times in response[0]:
                by_member[int(member)] = np.asarray(times, dtype=float)
        if set(by_member) != set(range(self._member_count)):
            self.abort()
            raise ParallelStepperError(
                "parallel histories returned incomplete members"
            )
        return tuple(
            by_member[member].copy() for member in range(self._member_count)
        )

    def close(self) -> None:
        """Close responsive workers, terminating only failed stragglers."""

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
                process.join()
        self._closed = True

    def abort(self) -> None:
        """Terminate every worker after cancellation or protocol failure."""

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
            process.join()
        self._closed = True

    def __enter__(self) -> "PersistentParallelSteppers":
        self._require_open()
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        if _exception_type is None:
            self.close()
        else:
            self.abort()

    def __del__(self):
        try:
            self.abort()
        except BaseException:
            pass


__all__ = [
    "PARALLEL_STEPPER_START_METHOD",
    "ParallelStepperError",
    "PersistentParallelSteppers",
]
