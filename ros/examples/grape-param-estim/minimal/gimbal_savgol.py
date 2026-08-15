#!/usr/bin/env python3
"""Centered irregular-time Savitzky--Golay smoothing for gimbal angles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


def unwrap_gimbal_angles(angle: np.ndarray) -> np.ndarray:
    """Unwrap each scalar joint trace independently."""

    value = np.asarray(angle, dtype=float)
    if value.ndim != 2 or value.shape[1] != 4 or np.any(~np.isfinite(value)):
        raise ValueError("gimbal angles must be a finite N x 4 array")
    return np.unwrap(value, axis=0)


@dataclass(frozen=True)
class GimbalSgEvaluation:
    time: np.ndarray
    angle: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        angle = np.asarray(self.angle, dtype=float)
        if (
            time.ndim != 1
            or angle.shape != (time.size, 4)
            or np.any(~np.isfinite(time))
            or np.any(~np.isfinite(angle))
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError("gimbal SG evaluation is invalid")
        object.__setattr__(self, "time", _readonly(time))
        object.__setattr__(self, "angle", _readonly(angle))


class IrregularSavitzkyGolayGimbal:
    """Local polynomial smoother on four asynchronous scalar channels.

    Every evaluation uses a centered time window.  The local time coordinate
    is scaled by the half-window for conditioning; the fitted intercept is the
    smoothed angle at the requested time.
    """

    def __init__(
        self,
        time_axis: Sequence[float],
        angle: np.ndarray,
        *,
        window_seconds: float,
        degree: int,
    ) -> None:
        time = np.asarray(time_axis, dtype=float)
        raw = unwrap_gimbal_angles(angle)
        window = float(window_seconds)
        polynomial_degree = int(degree)
        if (
            time.ndim != 1
            or time.size != raw.shape[0]
            or time.size < polynomial_degree + 1
            or np.any(~np.isfinite(time))
            or np.any(np.diff(time) <= 0.0)
            or not np.isfinite(window)
            or window <= 0.0
            or polynomial_degree < 0
        ):
            raise ValueError("irregular gimbal SG input is invalid")
        self.time = _readonly(time)
        self.raw_angle = _readonly(raw)
        self.window_seconds = window
        self.degree = polynomial_degree
        self.valid_start_time = float(time[0] + 0.5 * window)
        self.valid_end_time = float(time[-1] - 0.5 * window)
        if self.valid_end_time <= self.valid_start_time:
            raise ValueError("gimbal SG centered support is empty")

    def evaluate(self, query_time: Sequence[float]) -> GimbalSgEvaluation:
        query = np.asarray(query_time, dtype=float)
        if (
            query.ndim != 1
            or query.size < 1
            or np.any(~np.isfinite(query))
            or np.any(np.diff(query) <= 0.0)
            or query[0] < self.valid_start_time
            or query[-1] > self.valid_end_time
        ):
            raise ValueError("gimbal SG query is outside centered support")
        half = 0.5 * self.window_seconds
        result = np.empty((query.size, 4), dtype=float)
        powers = np.arange(self.degree + 1)
        for output_index, center in enumerate(query):
            left = int(np.searchsorted(self.time, center - half, side="left"))
            right = int(np.searchsorted(self.time, center + half, side="right"))
            if right - left < self.degree + 1:
                raise ValueError(
                    "gimbal SG window has fewer samples than polynomial terms"
                )
            local_time = (self.time[left:right] - center) / half
            design = local_time[:, None] ** powers[None, :]
            coefficients, _residual, rank, _singular = np.linalg.lstsq(
                design, self.raw_angle[left:right], rcond=None
            )
            if rank != self.degree + 1:
                raise ValueError("gimbal SG local polynomial is rank deficient")
            result[output_index] = coefficients[0]
        return GimbalSgEvaluation(query, result)


__all__ = (
    "GimbalSgEvaluation",
    "IrregularSavitzkyGolayGimbal",
    "unwrap_gimbal_angles",
)
