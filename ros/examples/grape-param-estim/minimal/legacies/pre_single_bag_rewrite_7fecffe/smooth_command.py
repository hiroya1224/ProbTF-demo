#!/usr/bin/env python3
"""Differentiable local smoothing of a causal zero-order-hold command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CommandEvaluation:
    value: np.ndarray
    delay_derivative: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float)
        derivative = np.asarray(self.delay_derivative, dtype=float)
        if (
            value.ndim != 1
            or derivative.shape != value.shape
            or np.any(~np.isfinite(value))
            or np.any(~np.isfinite(derivative))
        ):
            raise ValueError("command evaluation must contain finite vectors")
        value = value.copy()
        derivative = derivative.copy()
        value.setflags(write=False)
        derivative.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "delay_derivative", derivative)


def _deduplicate_last(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse equal adjacent timestamps, retaining the final message."""

    retained = np.concatenate((times[1:] != times[:-1], np.asarray((True,))))
    return times[retained].copy(), values[retained].copy()


class QuinticSmoothZoh:
    """Differentiable surrogate of a causal ZOH command.

    The ZOH is written as its initial value plus command jumps.  Each Heaviside
    jump is replaced by a quintic smoothstep.  Transition supports are allowed
    to overlap deliberately, so broad continuation stages have a broad lag
    derivative.

    ``width_fraction`` is the transition half-width in units of the median
    recorded publish period.  It is not clipped to avoid overlap.
    """

    def __init__(self, times: Sequence[float], values: np.ndarray) -> None:
        sample_times = np.asarray(times, dtype=float)
        sample_values = np.asarray(values, dtype=float)
        if (
            sample_times.ndim != 1
            or sample_times.size < 1
            or sample_values.ndim != 2
            or sample_values.shape[0] != sample_times.size
            or sample_values.shape[1] < 1
            or np.any(~np.isfinite(sample_times))
            or np.any(~np.isfinite(sample_values))
            or np.any(np.diff(sample_times) < 0.0)
        ):
            raise ValueError("command history must be finite and time ordered")
        sample_times, sample_values = _deduplicate_last(sample_times, sample_values)
        if sample_times.size > 1 and np.any(np.diff(sample_times) <= 0.0):
            raise RuntimeError("deduplicated command times are not strict")
        self.times = sample_times
        self.values = sample_values
        self.dimension = sample_values.shape[1]
        self.median_period = (
            float(np.median(np.diff(sample_times))) if sample_times.size > 1 else 0.0
        )
        self.times.setflags(write=False)
        self.values.setflags(write=False)

    def transition_half_widths(self, width_fraction: float) -> np.ndarray:
        fraction = float(width_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("width fraction must be finite and positive")
        if self.times.size < 2:
            return np.empty(0, dtype=float)
        half_width = fraction * self.median_period
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError("transition half-width is invalid")
        return np.full(self.times.size - 1, half_width, dtype=float)

    def exact_zoh(self, time: float, delay: float) -> np.ndarray:
        query = float(time) - float(delay)
        if not np.isfinite(query):
            raise ValueError("command query must be finite")
        index = int(np.searchsorted(self.times, query, side="right") - 1)
        index = min(max(index, 0), self.times.size - 1)
        return self.values[index].copy()

    def evaluate(self, time: float, delay: float, width_fraction: float) -> CommandEvaluation:
        evaluation_time = float(time)
        lag = float(delay)
        if not np.isfinite(evaluation_time) or not np.isfinite(lag):
            raise ValueError("evaluation time and delay must be finite")
        if self.times.size < 2:
            return CommandEvaluation(
                value=self.values[0],
                delay_derivative=np.zeros(self.dimension, dtype=float),
            )
        fraction = float(width_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("width fraction must be finite and positive")
        half_width = fraction * self.median_period
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError("transition half-width is invalid")

        query = evaluation_time - lag
        transitions = self.times[1:]
        completed = int(np.searchsorted(transitions, query - half_width, side="right"))
        active_end = int(np.searchsorted(transitions, query + half_width, side="left"))
        value = self.values[completed].copy()
        delay_derivative = np.zeros(self.dimension, dtype=float)
        for local_index in range(completed, active_end):
            transition_index = local_index + 1
            transition_time = float(self.times[transition_index])
            q = (query - transition_time + half_width) / (2.0 * half_width)
            smooth = q**3 * (q * (6.0 * q - 15.0) + 10.0)
            smooth_derivative = 30.0 * q**2 * (1.0 - q) ** 2
            delta = self.values[transition_index] - self.values[transition_index - 1]
            value += delta * smooth
            delay_derivative += -delta * smooth_derivative / (2.0 * half_width)
        return CommandEvaluation(value=value, delay_derivative=delay_derivative)
