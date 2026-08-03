"""Continuous command-delay utilities for closed-loop forecasts."""

from dataclasses import dataclass
from typing import Generic, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np


CommandT = TypeVar("CommandT")
DEFAULT_MAXIMUM_ESTIMATED_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class ConstantDelayChart:
    """One-dimensional continuous chart for a non-negative delay.

    The signed coordinate is retained in the raw control ensemble while its
    absolute value is the physical delay.  This keeps zero delay representable
    without quantizing to a simulation or publication period.
    """

    def decode(self, coordinate: float) -> float:
        value = float(coordinate)
        if not np.isfinite(value):
            raise ValueError("constant-delay coordinate must be finite")
        return abs(value)

    def encode(self, delay: float) -> float:
        return validate_constant_delay(delay)


@dataclass(frozen=True)
class BoundedDelayChart:
    """Smooth bijection from R to a physically bounded positive delay."""

    maximum_delay: float = DEFAULT_MAXIMUM_ESTIMATED_DELAY_SECONDS

    def __post_init__(self) -> None:
        maximum = float(self.maximum_delay)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum delay must be finite and positive")
        object.__setattr__(self, "maximum_delay", maximum)

    def decode(self, coordinate: float) -> float:
        value = float(coordinate)
        if not np.isfinite(value):
            raise ValueError("bounded-delay coordinate must be finite")
        if value >= 0.0:
            exponential = np.exp(-value)
            fraction = 1.0 / (1.0 + exponential)
        else:
            exponential = np.exp(value)
            fraction = exponential / (1.0 + exponential)
        return float(self.maximum_delay * fraction)

    def encode(self, delay: float) -> float:
        value = validate_constant_delay(delay)
        if not 0.0 < value < self.maximum_delay:
            raise ValueError(
                "delay must lie strictly between zero and maximum_delay"
            )
        fraction = value / self.maximum_delay
        return float(np.log(fraction) - np.log1p(-fraction))

    def coordinate_standard_deviation(
        self, delay: float, physical_standard_deviation: float
    ) -> float:
        """First-order conversion of a physical prior scale to chart units."""

        value = validate_constant_delay(delay)
        deviation = float(physical_standard_deviation)
        if (
            not 0.0 < value < self.maximum_delay
            or not np.isfinite(deviation)
            or deviation <= 0.0
        ):
            raise ValueError(
                "delay/deviation must be interior-positive and positive"
            )
        derivative = value * (1.0 - value / self.maximum_delay)
        return deviation / derivative


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
    "BoundedDelayChart",
    "ConstantDelayChart",
    "DEFAULT_MAXIMUM_ESTIMATED_DELAY_SECONDS",
    "ZeroOrderHoldCommandHistory",
    "validate_constant_delay",
    "zero_order_hold_values",
]
