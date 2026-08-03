"""Typed progress, measured ETA, JSON Lines, and cancellation primitives.

This module has no ROS or Qt dependency.  Estimator loops call
``CancellationToken.raise_if_cancelled()`` at forecast boundaries and emit a
``ProgressEvent`` synchronously through a callback.  A worker process can use
``JsonlProgressWriter(sys.stdout)`` as that callback while sending ordinary
logs exclusively to stderr.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    IO,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np


PROGRESS_EVENT_SCHEMA = "grape-param-estim/progress-event/v1"


class ProgressValidationError(ValueError):
    """A progress event or tracker update violates the wire contract."""


class ProgressCancelled(RuntimeError):
    """Cooperative cancellation observed at a safe work boundary."""

    def __init__(self, reason: str):
        self.reason = str(reason)
        super().__init__("progress cancelled: {}".format(self.reason))


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ProgressValidationError("{} must be an integer".format(name))
    result = int(value)
    if result < minimum:
        raise ProgressValidationError(
            "{} must be at least {}".format(name, minimum)
        )
    return result


def _finite(value: Any, name: str, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProgressValidationError(
            "{} must be numeric".format(name)
        ) from error
    if not np.isfinite(result) or result < minimum:
        raise ProgressValidationError(
            "{} must be finite and at least {}".format(name, minimum)
        )
    return result


def _optional_integer(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, name)


@dataclass(frozen=True)
class ProgressEvent:
    """One versioned estimator-to-GUI progress event."""

    run_id: str
    stage_id: str
    stage_label: str
    completed_units: int
    total_units: int
    fraction: float
    elapsed_seconds: float
    eta_seconds: Optional[float]
    iteration: Optional[int] = None
    maximum_iterations: Optional[int] = None
    bag_id: Optional[str] = None
    member_id: Optional[int] = None
    message: str = ""
    schema: str = PROGRESS_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROGRESS_EVENT_SCHEMA:
            raise ProgressValidationError(
                "unsupported progress schema {!r}".format(self.schema)
            )
        for name in ("run_id", "stage_id", "stage_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProgressValidationError(
                    "{} must be a non-empty string".format(name)
                )
        if not isinstance(self.message, str):
            raise ProgressValidationError("message must be a string")
        completed = _integer(self.completed_units, "completed_units")
        total = _integer(self.total_units, "total_units", minimum=1)
        if completed > total:
            raise ProgressValidationError(
                "completed_units cannot exceed total_units"
            )
        fraction = _finite(self.fraction, "fraction")
        expected_fraction = float(completed) / float(total)
        if fraction > 1.0 or not np.isclose(
            fraction, expected_fraction, rtol=0.0, atol=5.0e-13
        ):
            raise ProgressValidationError(
                "fraction must equal completed_units / total_units"
            )
        elapsed = _finite(self.elapsed_seconds, "elapsed_seconds")
        eta = (
            None
            if self.eta_seconds is None
            else _finite(self.eta_seconds, "eta_seconds")
        )
        iteration = _optional_integer(self.iteration, "iteration")
        maximum = _optional_integer(
            self.maximum_iterations, "maximum_iterations"
        )
        if (iteration is None) != (maximum is None):
            raise ProgressValidationError(
                "iteration and maximum_iterations must appear together"
            )
        if maximum is not None and maximum < 1:
            raise ProgressValidationError(
                "maximum_iterations must be positive"
            )
        if iteration is not None and iteration > maximum:
            raise ProgressValidationError(
                "iteration cannot exceed maximum_iterations"
            )
        bag_id = self.bag_id
        if bag_id is not None and (
            not isinstance(bag_id, str) or not bag_id
        ):
            raise ProgressValidationError(
                "bag_id must be null or a non-empty string"
            )
        member_id = _optional_integer(self.member_id, "member_id")
        object.__setattr__(self, "completed_units", completed)
        object.__setattr__(self, "total_units", total)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "eta_seconds", eta)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "maximum_iterations", maximum)
        object.__setattr__(self, "member_id", member_id)

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete JSON wire representation."""

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "stage_label": self.stage_label,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "fraction": self.fraction,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "iteration": self.iteration,
            "maximum_iterations": self.maximum_iterations,
            "bag_id": self.bag_id,
            "member_id": self.member_id,
            "message": self.message,
        }

    def to_json(self) -> str:
        """Return one compact JSON value without a trailing newline."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProgressEvent":
        """Strictly parse one progress-event mapping."""

        if not isinstance(value, Mapping):
            raise ProgressValidationError("progress event must be an object")
        required = {
            "schema",
            "run_id",
            "stage_id",
            "stage_label",
            "completed_units",
            "total_units",
            "fraction",
            "elapsed_seconds",
            "eta_seconds",
            "iteration",
            "maximum_iterations",
            "bag_id",
            "member_id",
            "message",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise ProgressValidationError(
                "progress event keys differ from schema; missing={}, extra={}".format(
                    sorted(missing), sorted(extra)
                )
            )
        return cls(
            schema=value["schema"],
            run_id=value["run_id"],
            stage_id=value["stage_id"],
            stage_label=value["stage_label"],
            completed_units=value["completed_units"],
            total_units=value["total_units"],
            fraction=value["fraction"],
            elapsed_seconds=value["elapsed_seconds"],
            eta_seconds=value["eta_seconds"],
            iteration=value["iteration"],
            maximum_iterations=value["maximum_iterations"],
            bag_id=value["bag_id"],
            member_id=value["member_id"],
            message=value["message"],
        )

    @classmethod
    def from_json(cls, line: str) -> "ProgressEvent":
        """Parse exactly one JSONL line."""

        if not isinstance(line, str) or not line.strip():
            raise ProgressValidationError(
                "progress JSON line must be non-empty text"
            )
        try:
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ProgressValidationError(
                        "progress JSON contains non-finite {}".format(token)
                    )
                ),
            )
        except ProgressValidationError:
            raise
        except json.JSONDecodeError as error:
            raise ProgressValidationError(
                "invalid progress JSON: {}".format(error)
            ) from error
        return cls.from_dict(value)


ProgressCallback = Callable[[ProgressEvent], None]


class JsonlProgressWriter:
    """Synchronous callback that writes progress events and nothing else."""

    def __init__(self, stream: IO[str]):
        if not hasattr(stream, "write") or not hasattr(stream, "flush"):
            raise TypeError("stream must provide write() and flush()")
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, event: ProgressEvent) -> None:
        if not isinstance(event, ProgressEvent):
            raise TypeError("JsonlProgressWriter accepts ProgressEvent only")
        line = event.to_json() + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


class CancellationToken:
    """Thread-safe, cooperative cancellation with first-reason semantics."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "user_requested") -> bool:
        """Request cancellation; return true only for the first request."""

        if not isinstance(reason, str) or not reason:
            raise ProgressValidationError(
                "cancellation reason must be a non-empty string"
            )
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            return True

    def raise_if_cancelled(self) -> None:
        """Raise at a caller-selected safe work boundary."""

        if self._event.is_set():
            raise ProgressCancelled(self.reason)


class ProgressTracker:
    """Create monotonic events and switch from estimated to measured ETA.

    ``total_units`` is fixed for one tracker.  Producers should include known
    line-search and posterior-forecast work in the estimate before creating
    it.  Repeated events at the same completed count are allowed for stage
    messages; decreasing counts are rejected.
    """

    def __init__(
        self,
        run_id: str,
        total_units: int,
        callback: Optional[ProgressCallback] = None,
        cancellation_token: Optional[CancellationToken] = None,
        clock: Optional[Callable[[], float]] = None,
        eta_calibration_units: int = 16,
        measurement_window: int = 32,
        initial_seconds_per_unit: Optional[float] = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ProgressValidationError(
                "run_id must be a non-empty string"
            )
        self.run_id = run_id
        self.total_units = _integer(total_units, "total_units", minimum=1)
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback
        self.cancellation_token = (
            CancellationToken()
            if cancellation_token is None
            else cancellation_token
        )
        if not isinstance(self.cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be CancellationToken")
        self._clock = time.monotonic if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        self.eta_calibration_units = _integer(
            eta_calibration_units, "eta_calibration_units", minimum=1
        )
        window = _integer(
            measurement_window, "measurement_window", minimum=1
        )
        self.initial_seconds_per_unit = (
            None
            if initial_seconds_per_unit is None
            else _finite(
                initial_seconds_per_unit,
                "initial_seconds_per_unit",
                minimum=0.0,
            )
        )
        if self.initial_seconds_per_unit == 0.0:
            raise ProgressValidationError(
                "initial_seconds_per_unit must be positive"
            )
        self._measurements: Deque[Tuple[int, float]] = deque(maxlen=window)
        start = _finite(self._clock(), "clock")
        self._start_time = start
        self._last_time = start
        self._last_completed = 0
        self._last_event: Optional[ProgressEvent] = None
        self._lock = threading.Lock()

    @property
    def last_event(self) -> Optional[ProgressEvent]:
        with self._lock:
            return self._last_event

    def checkpoint(self) -> None:
        """Observe cancellation at a forecast/member/bag boundary."""

        self.cancellation_token.raise_if_cancelled()

    def cancel(self, reason: str = "user_requested") -> bool:
        return self.cancellation_token.cancel(reason)

    def _eta(self, completed: int) -> Optional[float]:
        remaining = self.total_units - completed
        if remaining == 0:
            return 0.0
        measured_units = sum(units for units, _seconds in self._measurements)
        measured_seconds = sum(
            seconds for _units, seconds in self._measurements
        )
        if (
            completed >= self.eta_calibration_units
            and measured_units > 0
            and measured_seconds > 0.0
        ):
            return float(remaining) * measured_seconds / float(measured_units)
        if self.initial_seconds_per_unit is not None:
            return float(remaining) * self.initial_seconds_per_unit
        return None

    def emit(
        self,
        completed_units: int,
        stage_id: str,
        stage_label: str,
        iteration: Optional[int] = None,
        maximum_iterations: Optional[int] = None,
        bag_id: Optional[str] = None,
        member_id: Optional[int] = None,
        message: str = "",
    ) -> ProgressEvent:
        """Emit one event; callback exceptions propagate to the producer."""

        self.checkpoint()
        completed = _integer(completed_units, "completed_units")
        if completed > self.total_units:
            raise ProgressValidationError(
                "completed_units cannot exceed total_units"
            )
        with self._lock:
            now = _finite(self._clock(), "clock")
            if now < self._last_time:
                raise ProgressValidationError("monotonic clock moved backwards")
            if completed < self._last_completed:
                raise ProgressValidationError(
                    "completed_units must be monotonic"
                )
            unit_delta = completed - self._last_completed
            time_delta = now - self._last_time
            if unit_delta > 0 and time_delta > 0.0:
                self._measurements.append((unit_delta, time_delta))
            elapsed = now - self._start_time
            event = ProgressEvent(
                run_id=self.run_id,
                stage_id=stage_id,
                stage_label=stage_label,
                completed_units=completed,
                total_units=self.total_units,
                fraction=float(completed) / float(self.total_units),
                elapsed_seconds=elapsed,
                eta_seconds=self._eta(completed),
                iteration=iteration,
                maximum_iterations=maximum_iterations,
                bag_id=bag_id,
                member_id=member_id,
                message=message,
            )
            self._last_completed = completed
            self._last_time = now
            self._last_event = event
        if self.callback is not None:
            self.callback(event)
        return event


def emit_progress(
    callback: Optional[ProgressCallback], event: ProgressEvent
) -> None:
    """Invoke an optional synchronous callback with type checking."""

    if not isinstance(event, ProgressEvent):
        raise TypeError("event must be ProgressEvent")
    if callback is not None:
        callback(event)


__all__ = [
    "CancellationToken",
    "JsonlProgressWriter",
    "PROGRESS_EVENT_SCHEMA",
    "ProgressCallback",
    "ProgressCancelled",
    "ProgressEvent",
    "ProgressTracker",
    "ProgressValidationError",
    "emit_progress",
]
