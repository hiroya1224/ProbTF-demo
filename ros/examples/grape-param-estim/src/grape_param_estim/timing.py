"""Continuous command-delay utilities for closed-loop forecasts."""

from dataclasses import dataclass
from typing import Generic, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np


CommandT = TypeVar("CommandT")
def validate_constant_delay(value: float) -> float:
    """Return a finite non-negative controller-to-plant delay in seconds."""

    delay = float(value)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("constant delay must be finite and non-negative")
    return delay


@dataclass
class ZeroOrderHoldCommandHistory(Generic[CommandT]):
    """Timestamped commands evaluated as ``u(t - delay)``.

    The first command is the causal pre-history value.  This matches a
    controller/actuator path that holds its most recently published command
    and also makes the initial condition explicit for sub-sample delays.
    """

    constant_delay: float

    def __post_init__(self) -> None:
        self.constant_delay = validate_constant_delay(self.constant_delay)
        self._entries: List[Tuple[float, CommandT]] = []

    def append(self, issued_time: float, command: CommandT) -> None:
        time = float(issued_time)
        if not np.isfinite(time):
            raise ValueError("command issue time must be finite")
        if self._entries and time <= self._entries[-1][0]:
            raise ValueError("command issue times must be strictly increasing")
        self._entries.append((time, command))

    @property
    def issue_times(self) -> np.ndarray:
        return np.asarray([entry[0] for entry in self._entries], dtype=float)

    @property
    def values(self) -> Tuple[CommandT, ...]:
        """Return the retained values in issue-time order."""

        return tuple(entry[1] for entry in self._entries)

    def value_at(self, plant_time: float) -> CommandT:
        if not self._entries:
            raise ValueError("command history is empty")
        query = float(plant_time)
        if not np.isfinite(query):
            raise ValueError("plant time must be finite")
        effective_time = query - self.constant_delay
        times = self.issue_times
        index = int(np.searchsorted(times, effective_time, side="right") - 1)
        return self._entries[max(index, 0)][1]

    def switch_times(self, start_time: float, end_time: float) -> np.ndarray:
        """Return delayed publication boundaries strictly inside an interval."""

        start = float(start_time)
        end = float(end_time)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("time interval must be finite and increasing")
        if not self._entries:
            return np.empty(0, dtype=float)
        delayed = self.issue_times[1:] + self.constant_delay
        tolerance = 1.0e-12
        return delayed[
            (delayed > start + tolerance) & (delayed < end - tolerance)
        ].copy()


def zero_order_hold_values(
    issue_times: Sequence[float],
    values: np.ndarray,
    query_times: Iterable[float],
    constant_delay: float,
) -> np.ndarray:
    """Vectorized zero-order-held lookup for numeric command previews/tests."""

    times = np.asarray(issue_times, dtype=float)
    commands = np.asarray(values)
    queries = np.asarray(tuple(query_times), dtype=float)
    delay = validate_constant_delay(constant_delay)
    if (
        times.ndim != 1
        or times.size == 0
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("issue_times must be finite and strictly increasing")
    if commands.shape[0] != times.size:
        raise ValueError("values must align with issue_times")
    if queries.ndim != 1 or np.any(~np.isfinite(queries)):
        raise ValueError("query_times must be a finite one-dimensional array")
    indices = np.searchsorted(times, queries - delay, side="right") - 1
    indices = np.clip(indices, 0, times.size - 1)
    return commands[indices].copy()


__all__ = [
    "ZeroOrderHoldCommandHistory",
    "validate_constant_delay",
    "zero_order_hold_values",
]
