#!/usr/bin/env python3
"""Rotor-lag continuation schedules and exact strict-ZOH cell geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


def power_of_two_epsilon(depth: int) -> tuple[float, ...]:
    """Return ``epsilon_k = 2**(-k)`` for ``k = 0, ..., depth``."""

    maximum = int(depth)
    if maximum < 0:
        raise ValueError("continuation depth must be non-negative")
    return tuple(float(2.0 ** (-index)) for index in range(maximum + 1))


@dataclass(frozen=True)
class StrictLagCell:
    """One right-closed exact-ZOH equivalence cell ``(lower, upper]``."""

    index: int
    lower: float
    upper: float
    representative: float
    at_data_support_boundary: bool

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)


class StrictZohCellGrid:
    """Exact lag cells induced by evaluation and recorded command times."""

    def __init__(
        self,
        evaluation_time: Sequence[float],
        command_time: Sequence[float],
    ) -> None:
        evaluation = np.asarray(evaluation_time, dtype=float)
        command = np.asarray(command_time, dtype=float)
        if (
            evaluation.ndim != 1
            or evaluation.size < 1
            or command.ndim != 1
            or command.size < 2
            or np.any(~np.isfinite(evaluation))
            or np.any(~np.isfinite(command))
            or np.any(np.diff(evaluation) <= 0.0)
            or np.any(np.diff(command) <= 0.0)
        ):
            raise ValueError("strict-ZOH cell grid input is invalid")
        upper = float(evaluation[0] - command[0])
        if not np.isfinite(upper) or upper <= 0.0:
            raise ValueError("recorded command prehistory provides no lag support")
        differences = np.subtract.outer(evaluation, command).reshape(-1)
        interior = differences[(differences > 0.0) & (differences < upper)]
        candidates = np.sort(
            np.concatenate((np.asarray((0.0, upper)), interior))
        )
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(upper))
        retained = np.concatenate(
            (np.asarray((True,)), np.diff(candidates) > tolerance)
        )
        boundaries = candidates[retained]
        boundaries[0] = 0.0
        boundaries[-1] = upper
        if boundaries.size < 2 or np.any(np.diff(boundaries) <= 0.0):
            raise RuntimeError("strict-ZOH lag boundaries are invalid")
        self.evaluation_time = evaluation.copy()
        self.command_time = command.copy()
        self.boundaries = boundaries
        self.data_support_upper = upper

    @property
    def cell_count(self) -> int:
        return int(self.boundaries.size - 1)

    def cell(self, index: int) -> StrictLagCell:
        cell_index = int(index)
        if cell_index < 0 or cell_index >= self.cell_count:
            raise IndexError("strict-ZOH cell index is outside data support")
        lower = float(self.boundaries[cell_index])
        upper = float(self.boundaries[cell_index + 1])
        return StrictLagCell(
            index=cell_index,
            lower=lower,
            upper=upper,
            representative=0.5 * (lower + upper),
            at_data_support_boundary=cell_index == self.cell_count - 1,
        )

    def containing(self, lag_seconds: float) -> StrictLagCell:
        lag = float(lag_seconds)
        if (
            not np.isfinite(lag)
            or lag < 0.0
            or lag > self.data_support_upper
        ):
            raise ValueError("rotor lag lies outside recorded data support")
        if lag == 0.0:
            return self.cell(0)
        index = int(np.searchsorted(self.boundaries, lag, side="left") - 1)
        return self.cell(min(max(index, 0), self.cell_count - 1))

    def neighbor(
        self, cell: StrictLagCell, direction: int
    ) -> Optional[StrictLagCell]:
        step = int(direction)
        if step not in (-1, 1):
            raise ValueError("strict-ZOH neighbor direction must be -1 or +1")
        index = cell.index + step
        return None if index < 0 or index >= self.cell_count else self.cell(index)

    def command_indices(self, lag_seconds: float) -> np.ndarray:
        lag = float(lag_seconds)
        query = self.evaluation_time - lag
        index = np.searchsorted(self.command_time, query, side="right") - 1
        if np.any(index < 0):
            raise ValueError("rotor lag requests command before recorded prehistory")
        return index.astype(int, copy=False)


@dataclass(frozen=True)
class ProfiledCell(Generic[T]):
    cell: StrictLagCell
    cost: float
    payload: T


@dataclass(frozen=True)
class StrictProfileResult(Generic[T]):
    selected: ProfiledCell[T]
    evaluated: tuple[ProfiledCell[T], ...]
    final_neighbors: tuple[ProfiledCell[T], ...]


def local_strict_cell_descent(
    grid: StrictZohCellGrid,
    initial_lag: float,
    evaluator: Callable[[StrictLagCell], tuple[float, T]],
) -> StrictProfileResult[T]:
    """Profile adjacent exact cells until neither neighbor improves the cost."""

    cache: dict[int, ProfiledCell[T]] = {}

    def evaluate(cell: StrictLagCell) -> ProfiledCell[T]:
        if cell.index not in cache:
            cost, payload = evaluator(cell)
            value = float(cost)
            if not np.isfinite(value):
                raise FloatingPointError("strict-cell profile cost is non-finite")
            cache[cell.index] = ProfiledCell(cell, value, payload)
        return cache[cell.index]

    current = evaluate(grid.containing(initial_lag))
    for _iteration in range(grid.cell_count):
        candidates = [current]
        for direction in (-1, 1):
            neighbor = grid.neighbor(current.cell, direction)
            if neighbor is not None:
                candidates.append(evaluate(neighbor))
        best = min(candidates, key=lambda item: (item.cost, item.cell.index))
        if best.cell.index == current.cell.index:
            final_neighbors = tuple(
                item for item in candidates if item.cell.index != current.cell.index
            )
            return StrictProfileResult(
                selected=current,
                evaluated=tuple(cache[index] for index in sorted(cache)),
                final_neighbors=final_neighbors,
            )
        current = best
    raise RuntimeError("strict-ZOH local cell descent did not terminate")


__all__ = (
    "ProfiledCell",
    "StrictLagCell",
    "StrictProfileResult",
    "StrictZohCellGrid",
    "local_strict_cell_descent",
    "power_of_two_epsilon",
)
