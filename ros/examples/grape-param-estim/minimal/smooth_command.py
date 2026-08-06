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
    """Causal ZOH with non-overlapping quintic transitions and lag gradient."""

    def __init__(
        self,
        times: Sequence[float],
        values: np.ndarray,
    ) -> None:
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
        sample_times, sample_values = _deduplicate_last(
            sample_times,
            sample_values,
        )
        if sample_times.size > 1 and np.any(np.diff(sample_times) <= 0.0):
            raise RuntimeError("deduplicated command times are not strict")
        self.times = sample_times
        self.values = sample_values
        self.dimension = sample_values.shape[1]
        if sample_times.size > 1:
            gaps = np.diff(sample_times)
            self.median_period = float(np.median(gaps))
            local_half_width_limit = np.empty(sample_times.size - 1, dtype=float)
            for local_index, transition_index in enumerate(
                range(1, sample_times.size)
            ):
                neighbors = [0.49 * gaps[transition_index - 1]]
                if transition_index < sample_times.size - 1:
                    neighbors.append(0.49 * gaps[transition_index])
                local_half_width_limit[local_index] = min(neighbors)
            self._local_half_width_limit = local_half_width_limit
        else:
            self.median_period = 0.0
            self._local_half_width_limit = np.empty(0, dtype=float)
        self.times.setflags(write=False)
        self.values.setflags(write=False)
        self._local_half_width_limit.setflags(write=False)

    def transition_half_widths(self, width_fraction: float) -> np.ndarray:
        fraction = float(width_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("width fraction must be finite and positive")
        if self.times.size < 2:
            return np.empty(0, dtype=float)
        global_half_width = fraction * self.median_period
        return np.minimum(global_half_width, self._local_half_width_limit)

    def exact_zoh(self, time: float, delay: float) -> np.ndarray:
        query = float(time) - float(delay)
        if not np.isfinite(query):
            raise ValueError("command query must be finite")
        index = int(np.searchsorted(self.times, query, side="right") - 1)
        index = min(max(index, 0), self.times.size - 1)
        return self.values[index].copy()

    def evaluate(
        self,
        time: float,
        delay: float,
        width_fraction: float,
    ) -> CommandEvaluation:
        evaluation_time = float(time)
        lag = float(delay)
        if not np.isfinite(evaluation_time) or not np.isfinite(lag):
            raise ValueError("evaluation time and delay must be finite")
        half_widths = self.transition_half_widths(width_fraction)
        if self.times.size < 2:
            return CommandEvaluation(
                value=self.values[0],
                delay_derivative=np.zeros(self.dimension, dtype=float),
            )

        query = evaluation_time - lag
        insertion = int(np.searchsorted(self.times[1:], query, side="left"))
        candidate_indices = []
        for local_index in (insertion - 1, insertion):
            if 0 <= local_index < half_widths.size:
                candidate_indices.append(local_index)
        for local_index in candidate_indices:
            transition_index = local_index + 1
            epsilon = float(half_widths[local_index])
            offset = query - float(self.times[transition_index])
            if -epsilon < offset < epsilon:
                q = (offset + epsilon) / (2.0 * epsilon)
                smooth = q**3 * (q * (6.0 * q - 15.0) + 10.0)
                smooth_derivative = 30.0 * q**2 * (1.0 - q) ** 2
                delta = (
                    self.values[transition_index]
                    - self.values[transition_index - 1]
                )
                return CommandEvaluation(
                    value=self.values[transition_index - 1] + delta * smooth,
                    delay_derivative=(
                        -delta * smooth_derivative / (2.0 * epsilon)
                    ),
                )

        return CommandEvaluation(
            value=self.exact_zoh(evaluation_time, lag),
            delay_derivative=np.zeros(self.dimension, dtype=float),
        )
