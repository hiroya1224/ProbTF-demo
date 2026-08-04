"""Strict JSONL progress events for the sparse batch estimator worker.

The worker reports two related notions of progress.  ``stage_*`` fields are
local to the current kind of work (for example one nonlinear solve), while
``completed_units``, ``total_units``, and ``fraction`` are the monotonic
overall run progress consumed by the GUI.  :class:`ProgressTracker` computes
the overall offset so estimator loops cannot accidentally make progress run
backwards when they move to the next stage.

This module deliberately has no ROS or Qt dependency and targets Python 3.8.
Worker stdout must contain only values written by :class:`JsonlProgressWriter`;
ordinary diagnostics belong on stderr.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from types import MappingProxyType
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


PROGRESS_EVENT_SCHEMA = "grape-param-estim/progress-event/v2"

STAGE_PREPARING_TRAJECTORY = "preparing_trajectory"
STAGE_OPTIMIZING_FULL_TRAJECTORY = "optimizing_full_trajectory"
STAGE_REFINING_CONSTANT_DELAY = "refining_constant_delay"
STAGE_UPDATING_MODEL_ERROR_COVARIANCE = (
    "updating_model_error_covariance"
)
STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY = (
    "computing_local_posterior_geometry"
)
STAGE_SAMPLING_PARAMETER_POSTERIOR = "sampling_parameter_posterior"
STAGE_WRITING_ARTIFACTS = "writing_artifacts"

STAGE_LABELS = MappingProxyType(
    {
        STAGE_PREPARING_TRAJECTORY: "Preparing trajectory",
        STAGE_OPTIMIZING_FULL_TRAJECTORY: "Optimizing full trajectory",
        STAGE_REFINING_CONSTANT_DELAY: "Refining constant delay",
        STAGE_UPDATING_MODEL_ERROR_COVARIANCE: (
            "Updating model-error covariance"
        ),
        STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY: (
            "Computing local posterior geometry"
        ),
        STAGE_SAMPLING_PARAMETER_POSTERIOR: (
            "Sampling parameter posterior"
        ),
        STAGE_WRITING_ARTIFACTS: "Writing artifacts",
    }
)


class ProgressValidationError(ValueError):
    """A progress event or tracker update violates the wire contract."""


class ProgressCancelled(RuntimeError):
    """Cooperative cancellation observed at a safe work boundary."""

    def __init__(self, reason: str):
        self.reason = str(reason)
        super().__init__("progress cancelled: {}".format(self.reason))


def stage_label(stage_id: str) -> str:
    """Return the fixed user-facing label for a supported stage ID."""

    if not isinstance(stage_id, str) or stage_id not in STAGE_LABELS:
        raise ProgressValidationError(
            "unsupported progress stage_id {!r}".format(stage_id)
        )
    return STAGE_LABELS[stage_id]


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
    if isinstance(value, bool):
        raise ProgressValidationError("{} must be numeric".format(name))
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


def _fraction(
    value: Any,
    name: str,
    completed: int,
    total: int,
) -> float:
    result = _finite(value, name)
    expected = float(completed) / float(total)
    if result > 1.0 or not np.isclose(
        result, expected, rtol=0.0, atol=5.0e-13
    ):
        raise ProgressValidationError(
            "{} must equal its completed units / total units".format(name)
        )
    return result


@dataclass(frozen=True)
class ProgressEvent:
    """One complete versioned worker-to-GUI progress event.

    The unprefixed unit fields are the overall run counter.  The explicit
    stage fields describe the current stage and may restart at zero when a new
    stage begins.  ``eta_seconds`` is the overall ETA and
    ``stage_eta_seconds`` is local to the current stage.
    """

    run_id: str
    stage_id: str
    stage_label: str
    stage_completed_units: int
    stage_total_units: int
    stage_fraction: float
    completed_units: int
    total_units: int
    fraction: float
    stage_elapsed_seconds: float
    stage_eta_seconds: Optional[float]
    elapsed_seconds: float
    eta_seconds: Optional[float]
    iteration: Optional[int] = None
    maximum_iterations: Optional[int] = None
    bag_id: Optional[str] = None
    sample_id: Optional[str] = None
    message: str = ""
    schema: str = PROGRESS_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROGRESS_EVENT_SCHEMA:
            raise ProgressValidationError(
                "unsupported progress schema {!r}".format(self.schema)
            )
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ProgressValidationError(
                "run_id must be a non-empty string"
            )
        expected_label = stage_label(self.stage_id)
        if self.stage_label != expected_label:
            raise ProgressValidationError(
                "stage_label for {!r} must be {!r}".format(
                    self.stage_id, expected_label
                )
            )
        if not isinstance(self.message, str):
            raise ProgressValidationError("message must be a string")

        stage_completed = _integer(
            self.stage_completed_units, "stage_completed_units"
        )
        stage_total = _integer(
            self.stage_total_units, "stage_total_units", minimum=1
        )
        if stage_completed > stage_total:
            raise ProgressValidationError(
                "stage_completed_units cannot exceed stage_total_units"
            )
        stage_fraction_value = _fraction(
            self.stage_fraction,
            "stage_fraction",
            stage_completed,
            stage_total,
        )

        completed = _integer(self.completed_units, "completed_units")
        total = _integer(self.total_units, "total_units", minimum=1)
        if completed > total:
            raise ProgressValidationError(
                "completed_units cannot exceed total_units"
            )
        fraction_value = _fraction(
            self.fraction, "fraction", completed, total
        )

        stage_elapsed = _finite(
            self.stage_elapsed_seconds, "stage_elapsed_seconds"
        )
        stage_eta = (
            None
            if self.stage_eta_seconds is None
            else _finite(self.stage_eta_seconds, "stage_eta_seconds")
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
        sample_id = self.sample_id
        if sample_id is not None and (
            not isinstance(sample_id, str) or not sample_id
        ):
            raise ProgressValidationError(
                "sample_id must be null or a non-empty string"
            )

        object.__setattr__(self, "stage_completed_units", stage_completed)
        object.__setattr__(self, "stage_total_units", stage_total)
        object.__setattr__(self, "stage_fraction", stage_fraction_value)
        object.__setattr__(self, "completed_units", completed)
        object.__setattr__(self, "total_units", total)
        object.__setattr__(self, "fraction", fraction_value)
        object.__setattr__(self, "stage_elapsed_seconds", stage_elapsed)
        object.__setattr__(self, "stage_eta_seconds", stage_eta)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "eta_seconds", eta)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "maximum_iterations", maximum)

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete JSON wire representation."""

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "stage_label": self.stage_label,
            "stage_completed_units": self.stage_completed_units,
            "stage_total_units": self.stage_total_units,
            "stage_fraction": self.stage_fraction,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "fraction": self.fraction,
            "stage_elapsed_seconds": self.stage_elapsed_seconds,
            "stage_eta_seconds": self.stage_eta_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "iteration": self.iteration,
            "maximum_iterations": self.maximum_iterations,
            "bag_id": self.bag_id,
            "sample_id": self.sample_id,
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
            "stage_completed_units",
            "stage_total_units",
            "stage_fraction",
            "completed_units",
            "total_units",
            "fraction",
            "stage_elapsed_seconds",
            "stage_eta_seconds",
            "elapsed_seconds",
            "eta_seconds",
            "iteration",
            "maximum_iterations",
            "bag_id",
            "sample_id",
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
            stage_completed_units=value["stage_completed_units"],
            stage_total_units=value["stage_total_units"],
            stage_fraction=value["stage_fraction"],
            completed_units=value["completed_units"],
            total_units=value["total_units"],
            fraction=value["fraction"],
            stage_elapsed_seconds=value["stage_elapsed_seconds"],
            stage_eta_seconds=value["stage_eta_seconds"],
            elapsed_seconds=value["elapsed_seconds"],
            eta_seconds=value["eta_seconds"],
            iteration=value["iteration"],
            maximum_iterations=value["maximum_iterations"],
            bag_id=value["bag_id"],
            sample_id=value["sample_id"],
            message=value["message"],
        )

    @classmethod
    def from_json(cls, line: str) -> "ProgressEvent":
        """Parse exactly one strict JSONL line."""

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


class StageProgress:
    """A stage-local counter whose events are offset into overall progress."""

    def __init__(
        self,
        tracker: "ProgressTracker",
        stage_id: str,
        total_units: int,
        overall_offset: int,
        start_time: float,
        measurement_window: int,
    ) -> None:
        self._tracker = tracker
        self.stage_id = stage_id
        self.stage_label = stage_label(stage_id)
        self.total_units = total_units
        self.overall_offset = overall_offset
        self._start_time = start_time
        self._last_time = start_time
        self._last_completed = 0
        self._last_event: Optional[ProgressEvent] = None
        self._measurements: Deque[Tuple[int, float]] = deque(
            maxlen=measurement_window
        )

    @property
    def completed_units(self) -> int:
        return self._last_completed

    @property
    def last_event(self) -> Optional[ProgressEvent]:
        return self._last_event

    def checkpoint(self) -> None:
        """Observe cancellation at a caller-selected safe boundary."""

        self._tracker.checkpoint()

    def emit(
        self,
        completed_units: int,
        *,
        iteration: Optional[int] = None,
        maximum_iterations: Optional[int] = None,
        bag_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        message: str = "",
    ) -> ProgressEvent:
        """Emit one stage-local update and its monotonic overall projection."""

        return self._tracker._emit_stage(
            self,
            completed_units,
            iteration=iteration,
            maximum_iterations=maximum_iterations,
            bag_id=bag_id,
            sample_id=sample_id,
            message=message,
        )

    def complete(
        self,
        *,
        iteration: Optional[int] = None,
        maximum_iterations: Optional[int] = None,
        bag_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        message: str = "",
    ) -> ProgressEvent:
        """Emit the final event for this stage."""

        return self.emit(
            self.total_units,
            iteration=iteration,
            maximum_iterations=maximum_iterations,
            bag_id=bag_id,
            sample_id=sample_id,
            message=message,
        )


class ProgressTracker:
    """Build ordered stages with measured overall and stage-local ETAs.

    ``overall_total_units`` is the sum of all stage invocations planned for
    the run.  A caller begins each invocation with :meth:`begin_stage`; the
    previous invocation must be complete before the next begins.  Reusing a
    stage ID is allowed, which is useful for alternating nonlinear, lag, and
    EM work while retaining one globally monotonic counter.
    """

    def __init__(
        self,
        run_id: str,
        overall_total_units: int,
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
        self.total_units = _integer(
            overall_total_units, "overall_total_units", minimum=1
        )
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
        self._measurement_window = _integer(
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
        self._measurements: Deque[Tuple[int, float]] = deque(
            maxlen=self._measurement_window
        )
        start = _finite(self._clock(), "clock")
        self._start_time = start
        self._last_time = start
        self._last_completed = 0
        self._last_event: Optional[ProgressEvent] = None
        self._active_stage: Optional[StageProgress] = None
        self._lock = threading.Lock()

    @property
    def last_event(self) -> Optional[ProgressEvent]:
        with self._lock:
            return self._last_event

    @property
    def completed_units(self) -> int:
        with self._lock:
            return self._last_completed

    def checkpoint(self) -> None:
        """Observe cancellation at a caller-selected safe work boundary."""

        self.cancellation_token.raise_if_cancelled()

    def cancel(self, reason: str = "user_requested") -> bool:
        return self.cancellation_token.cancel(reason)

    def begin_stage(
        self, stage_id: str, stage_total_units: int
    ) -> StageProgress:
        """Begin an ordered stage invocation without emitting an event."""

        self.checkpoint()
        stage_label(stage_id)
        total = _integer(
            stage_total_units, "stage_total_units", minimum=1
        )
        with self._lock:
            if (
                self._active_stage is not None
                and self._active_stage.completed_units
                != self._active_stage.total_units
            ):
                raise ProgressValidationError(
                    "the active stage must be complete before beginning another"
                )
            if self._last_completed + total > self.total_units:
                raise ProgressValidationError(
                    "stage work exceeds overall_total_units"
                )
            now = _finite(self._clock(), "clock")
            if now < self._last_time:
                raise ProgressValidationError("monotonic clock moved backwards")
            stage = StageProgress(
                tracker=self,
                stage_id=stage_id,
                total_units=total,
                overall_offset=self._last_completed,
                start_time=now,
                measurement_window=self._measurement_window,
            )
            self._active_stage = stage
            return stage

    def _eta(
        self,
        completed: int,
        total: int,
        measurements: Deque[Tuple[int, float]],
    ) -> Optional[float]:
        remaining = total - completed
        if remaining == 0:
            return 0.0
        measured_units = sum(units for units, _seconds in measurements)
        measured_seconds = sum(seconds for _units, seconds in measurements)
        if (
            completed >= self.eta_calibration_units
            and measured_units > 0
            and measured_seconds > 0.0
        ):
            return float(remaining) * measured_seconds / float(measured_units)
        if self.initial_seconds_per_unit is not None:
            return float(remaining) * self.initial_seconds_per_unit
        return None

    def _emit_stage(
        self,
        stage: StageProgress,
        completed_units: int,
        *,
        iteration: Optional[int],
        maximum_iterations: Optional[int],
        bag_id: Optional[str],
        sample_id: Optional[str],
        message: str,
    ) -> ProgressEvent:
        self.checkpoint()
        completed = _integer(completed_units, "stage_completed_units")
        if completed > stage.total_units:
            raise ProgressValidationError(
                "stage_completed_units cannot exceed stage_total_units"
            )
        with self._lock:
            if stage is not self._active_stage:
                raise ProgressValidationError(
                    "cannot emit from an inactive progress stage"
                )
            now = _finite(self._clock(), "clock")
            if now < self._last_time or now < stage._last_time:
                raise ProgressValidationError("monotonic clock moved backwards")
            if completed < stage._last_completed:
                raise ProgressValidationError(
                    "stage_completed_units must be monotonic"
                )

            unit_delta = completed - stage._last_completed
            time_delta = now - stage._last_time
            if unit_delta > 0 and time_delta > 0.0:
                measurement = (unit_delta, time_delta)
                stage._measurements.append(measurement)
                self._measurements.append(measurement)
            overall_completed = stage.overall_offset + completed
            if overall_completed < self._last_completed:
                raise ProgressValidationError(
                    "completed_units must be monotonic"
                )

            event = ProgressEvent(
                run_id=self.run_id,
                stage_id=stage.stage_id,
                stage_label=stage.stage_label,
                stage_completed_units=completed,
                stage_total_units=stage.total_units,
                stage_fraction=float(completed) / float(stage.total_units),
                completed_units=overall_completed,
                total_units=self.total_units,
                fraction=float(overall_completed) / float(self.total_units),
                stage_elapsed_seconds=now - stage._start_time,
                stage_eta_seconds=self._eta(
                    completed, stage.total_units, stage._measurements
                ),
                elapsed_seconds=now - self._start_time,
                eta_seconds=self._eta(
                    overall_completed, self.total_units, self._measurements
                ),
                iteration=iteration,
                maximum_iterations=maximum_iterations,
                bag_id=bag_id,
                sample_id=sample_id,
                message=message,
            )
            stage._last_completed = completed
            stage._last_time = now
            stage._last_event = event
            self._last_completed = overall_completed
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
    "STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY",
    "STAGE_LABELS",
    "STAGE_OPTIMIZING_FULL_TRAJECTORY",
    "STAGE_PREPARING_TRAJECTORY",
    "STAGE_REFINING_CONSTANT_DELAY",
    "STAGE_SAMPLING_PARAMETER_POSTERIOR",
    "STAGE_UPDATING_MODEL_ERROR_COVARIANCE",
    "STAGE_WRITING_ARTIFACTS",
    "StageProgress",
    "emit_progress",
    "stage_label",
]
