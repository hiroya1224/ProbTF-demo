#!/usr/bin/env python3
"""Exact adaptive single-group PID survival maps for first-order Gimbalrotor.

For one selected PID group, the other three groups are held at the recorded
successful-flight gains.  The selected group is varied around the gain recorded
in the analyzed failure bag.  For every queried gain point and every sampled
plant, the full sampled closed-loop model is trimmed and analytically
linearized; no gain-affine surrogate is used.

The plotted scalar is the conditional survival fraction among plant samples
that are stable under the all-success controller.  A value of one is therefore
best.  The displayed boundary is the requested alpha survival level (0.95 by
default).

The three views are slices through the failure gain:

    PI: D = D_failure
    ID: P = P_failure
    DP: I = I_failure

Global search uses nested dyadic grids 3x3 -> 5x5 -> 9x9 -> 17x17 only until
the boundary is first detected.  From that point onward, only boundary cells
are split, until the local spacing reaches the configured final equivalent
grid size (33x33 by default).  If no boundary is found through 17x17, the
search stops there.  Every repeated gain point is cached exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import (  # noqa: E402
    COVARIANCE_NAMES,
    PID_GROUPS,
    actuator_parameters_from_estimate,
    draw_quotient_coordinates,
    load_estimate_json,
    quotient_to_scale_free_plants,
)
from grape_param_estim.controller import ControllerConfig  # noqa: E402
from grape_param_estim.controller_config import (  # noqa: E402
    PidGainConfiguration,
    apply_pid_gain_configuration,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    ScaleFreePlant,
    load_vehicle_model,
)
from gimbalrotor_pid_local_pole_validation import (  # noqa: E402
    _analyze_plant,
    decompose_thrust_delay,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


CONTOUR_SCHEMA = "grape-param-estim/first-order-lag-pid-group-survival/v3"
DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_GROUP = "roll_pitch"
DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 3.0
DEFAULT_MAX_GRID_SIZE = 17
DEFAULT_FINAL_BOUNDARY_GRID_SIZE = 33
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
PROJECTION_SPECS = (
    ("pi", 0, 1, 2, "P", "I", "D"),
    ("id", 1, 2, 0, "I", "D", "P"),
    ("dp", 2, 0, 1, "D", "P", "I"),
)

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class GainTriple:
    p: float
    i: float
    d: float

    def array(self) -> np.ndarray:
        result = np.asarray((self.p, self.i, self.d), dtype=float)
        if result.shape != (3,) or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("PID gains must be finite and strictly positive")
        return result


@dataclass(frozen=True)
class BreakEvaluation:
    log_ratio: np.ndarray
    stable_mask: np.ndarray
    caused_break_mask: np.ndarray
    caused_break_count: int
    caused_break_fraction: float
    unstable_count: int
    unstable_fraction: float


@dataclass(frozen=True)
class GridLevel:
    size: int
    new_point_count: int
    total_cached_point_count: int
    minimum_survival_fraction: float
    maximum_survival_fraction: float
    boundary_present: bool


@dataclass(frozen=True)
class LocalRefinementLevel:
    level: int
    equivalent_grid_size: int
    input_boundary_cell_count: int
    output_boundary_cell_count: int
    new_point_count: int
    total_cached_point_count: int


@dataclass(frozen=True)
class AdaptiveCell:
    x0: float
    x1: float
    y0: float
    y1: float
    values: tuple[float, float, float, float]

    @property
    def mean_value(self) -> float:
        return float(np.mean(np.asarray(self.values, dtype=float)))


@dataclass(frozen=True)
class ProjectionGrid:
    name: str
    first_axis: int
    second_axis: int
    hidden_axis: int
    hidden_gain: float
    axis_log_ratio: np.ndarray
    survival_fraction: np.ndarray
    global_levels: tuple[GridLevel, ...]
    boundary_first_seen_grid_size: Optional[int]
    local_refinement_levels: tuple[LocalRefinementLevel, ...]
    adaptive_cells: tuple[AdaptiveCell, ...]
    final_boundary_cells: tuple[AdaptiveCell, ...]
    boundary_segments_log_ratio: np.ndarray
    effective_local_equivalent_grid_size: int
    stop_reason: str


def _recorded_gains(estimate: Mapping[str, Any]) -> Mapping[str, GainTriple]:
    payload = estimate["controller"]["gains"]
    return {
        group: GainTriple(
            float(payload[group]["p_gain"]),
            float(payload[group]["i_gain"]),
            float(payload[group]["d_gain"]),
        )
        for group in PID_GROUPS
    }


def _configuration(gains: Mapping[str, GainTriple]) -> Any:
    values = np.asarray([gains[group].array() for group in PID_GROUPS], dtype=float)
    return apply_pid_gain_configuration(
        ControllerConfig.grape(), PidGainConfiguration(values)
    )


def _replace_group(
    gains: Mapping[str, GainTriple], group: str, triple: GainTriple
) -> Mapping[str, GainTriple]:
    result = dict(gains)
    result[group] = triple
    return result


def _inputs(
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_configuration: Any,
) -> Any:
    return SimpleNamespace(
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_configuration=controller_configuration,
    )


def _exact_matrix_with_configuration(
    *,
    plant: ScaleFreePlant,
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_dt: float,
    controller_configuration: Any,
) -> tuple[np.ndarray, bool]:
    result = _analyze_plant(
        scale_free=plant,
        inputs=_inputs(
            vehicle_model,
            actuator_parameters,
            controller_configuration,
        ),
        controller_dt=float(controller_dt),
        delay=decompose_thrust_delay(0.0, float(controller_dt)),
        fd_check=False,
        compute_eigenvalues=False,
    )
    matrix = result.get("jacobian")
    trim = result.get("trim")
    if matrix is None or trim is None or not bool(trim.equilibrium_valid):
        raise RuntimeError(
            "full 26-state pole Jacobian is unavailable at the requested gain"
        )
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (26, 26) or np.any(~np.isfinite(selected)):
        raise RuntimeError("first-order closed-loop Jacobian must be finite 26x26")
    near_kink = bool(trim.piecewise_linearization_near_kink) or bool(
        result.get("analytic_piecewise_near_kink", False)
    )
    return selected, near_kink


def _exact_matrix_chunk_task(
    task: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        indices,
        plants,
        vehicle_model,
        actuator_parameters,
        controller_dt,
        controller_configuration,
    ) = task
    selected_indices = np.asarray(indices, dtype=int)
    matrices = np.empty((selected_indices.size, 26, 26), dtype=float)
    near_kink = np.empty(selected_indices.size, dtype=bool)
    for local_index, plant in enumerate(plants):
        matrix, near = _exact_matrix_with_configuration(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            controller_configuration=controller_configuration,
        )
        matrices[local_index] = matrix
        near_kink[local_index] = near
    return selected_indices, matrices, near_kink


class ExactGroupGainEvaluator:
    """Exact per-gain Monte Carlo pole evaluator with batched eigensystems."""

    def __init__(
        self,
        *,
        plants: Sequence[ScaleFreePlant],
        hybrid_failure_gains: Mapping[str, GainTriple],
        success_gains: Mapping[str, GainTriple],
        group: str,
        vehicle_model: Any,
        actuator_parameters: Any,
        controller_dt: float,
        workers: int,
    ) -> None:
        self.plants = tuple(plants)
        self.hybrid_failure_gains = dict(hybrid_failure_gains)
        self.success_gains = dict(success_gains)
        self.group = str(group)
        self.base_gain = self.hybrid_failure_gains[self.group].array()
        self.success_gain = self.success_gains[self.group].array()
        self.vehicle_model = vehicle_model
        self.actuator_parameters = actuator_parameters
        self.controller_dt = float(controller_dt)
        self.workers = min(int(workers), len(self.plants))
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        self._executor = (
            None
            if self.workers <= 1
            else ProcessPoolExecutor(max_workers=self.workers)
        )
        self._radius_cache: dict[tuple[float, float, float], np.ndarray] = {}
        self._near_kink_cache: dict[tuple[float, float, float], np.ndarray] = {}
        self._break_cache: dict[tuple[float, float, float], BreakEvaluation] = {}
        self.success_log_ratio = np.log(self.success_gain / self.base_gain)
        self.success_log_radius = self.log_spectral_radius(self.success_log_ratio)
        self.success_stable_mask = self.success_log_radius < 0.0
        self.success_stable_count = int(np.count_nonzero(self.success_stable_mask))
        if self.success_stable_count <= 0:
            raise RuntimeError(
                "all-success controller has no stable plant sample; conditional survival is undefined"
            )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @property
    def sample_count(self) -> int:
        return len(self.plants)

    @property
    def evaluated_gain_point_count(self) -> int:
        return len(self._radius_cache)

    @property
    def gain_points_with_any_near_kink(self) -> int:
        return int(
            sum(bool(np.any(value)) for value in self._near_kink_cache.values())
        )

    @staticmethod
    def _key(log_ratio: Sequence[float]) -> tuple[float, float, float]:
        selected = np.asarray(log_ratio, dtype=float)
        if selected.shape != (3,) or np.any(~np.isfinite(selected)):
            raise ValueError("log gain ratio must be finite length three")
        rounded = np.round(selected, decimals=14)
        return tuple(float(value) for value in rounded)

    def gain_from_log_ratio(self, log_ratio: Sequence[float]) -> np.ndarray:
        selected = np.asarray(log_ratio, dtype=float)
        if selected.shape != (3,) or np.any(~np.isfinite(selected)):
            raise ValueError("log gain ratio must be finite length three")
        return self.base_gain * np.exp(selected)

    def _matrix_stack(self, log_ratio: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        gain = self.gain_from_log_ratio(log_ratio)
        gains = _replace_group(
            self.hybrid_failure_gains,
            self.group,
            GainTriple(*gain),
        )
        controller_configuration = _configuration(gains)
        sample_count = len(self.plants)
        # Send a small number of plant chunks to persistent workers.  This
        # avoids 512 IPC round trips at every gain point while preserving one
        # independently trimmed/linearized Jacobian per plant sample.
        chunk_count = min(sample_count, max(1, 2 * self.workers))
        index_chunks = [
            chunk
            for chunk in np.array_split(np.arange(sample_count, dtype=int), chunk_count)
            if chunk.size
        ]
        tasks = [
            (
                tuple(int(index) for index in chunk),
                tuple(self.plants[int(index)] for index in chunk),
                self.vehicle_model,
                self.actuator_parameters,
                self.controller_dt,
                controller_configuration,
            )
            for chunk in index_chunks
        ]
        if self._executor is None:
            rows = [_exact_matrix_chunk_task(task) for task in tasks]
        else:
            rows = list(self._executor.map(_exact_matrix_chunk_task, tasks))
        matrices = np.empty((sample_count, 26, 26), dtype=float)
        near_kink = np.empty(sample_count, dtype=bool)
        filled = np.zeros(sample_count, dtype=bool)
        for indices, chunk_matrices, chunk_near_kink in rows:
            matrices[indices] = chunk_matrices
            near_kink[indices] = chunk_near_kink
            filled[indices] = True
        if not np.all(filled):
            raise RuntimeError("exact plant sample ordering changed during evaluation")
        return matrices, near_kink

    def log_spectral_radius(self, log_ratio: Sequence[float]) -> np.ndarray:
        key = self._key(log_ratio)
        cached = self._radius_cache.get(key)
        if cached is not None:
            return cached.copy()
        matrices, near_kink = self._matrix_stack(key)
        # NumPy accepts a stack of square matrices.  All 512 eigensystems are
        # therefore dispatched in one call rather than a Python eigvals loop.
        eigenvalues = np.linalg.eigvals(matrices)
        radii = np.max(np.abs(eigenvalues), axis=1)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise FloatingPointError("closed-loop spectral radius became invalid")
        result = np.log(radii)
        self._radius_cache[key] = result.copy()
        self._near_kink_cache[key] = near_kink.copy()
        return result

    def break_evaluation(self, log_ratio: Sequence[float]) -> BreakEvaluation:
        key = self._key(log_ratio)
        cached = self._break_cache.get(key)
        if cached is not None:
            return BreakEvaluation(
                cached.log_ratio.copy(),
                cached.stable_mask.copy(),
                cached.caused_break_mask.copy(),
                cached.caused_break_count,
                cached.caused_break_fraction,
                cached.unstable_count,
                cached.unstable_fraction,
            )
        log_radius = self.log_spectral_radius(key)
        stable = log_radius < 0.0
        caused = self.success_stable_mask & ~stable
        result = BreakEvaluation(
            log_ratio=np.asarray(key, dtype=float),
            stable_mask=stable,
            caused_break_mask=caused,
            caused_break_count=int(np.count_nonzero(caused)),
            caused_break_fraction=float(np.mean(caused)),
            unstable_count=int(np.count_nonzero(~stable)),
            unstable_fraction=float(np.mean(~stable)),
        )
        self._break_cache[key] = result
        return result

    def survival_fraction(self, log_ratio: Sequence[float]) -> float:
        evaluation = self.break_evaluation(log_ratio)
        return float(
            1.0 - evaluation.caused_break_count / self.success_stable_count
        )


class SliceGridEvaluator:
    """Cached 2-D group slice evaluated in conditional survival fraction."""

    def __init__(
        self,
        group_evaluator: ExactGroupGainEvaluator,
        *,
        first_axis: int,
        second_axis: int,
        hidden_axis: int,
    ) -> None:
        self.group_evaluator = group_evaluator
        self.first_axis = int(first_axis)
        self.second_axis = int(second_axis)
        self.hidden_axis = int(hidden_axis)
        self._cache: dict[tuple[float, float], float] = {}

    @staticmethod
    def _visible_key(first: float, second: float) -> tuple[float, float]:
        rounded = np.round(np.asarray((first, second), dtype=float), decimals=14)
        return float(rounded[0]), float(rounded[1])

    def _full_coordinate(self, first: float, second: float) -> np.ndarray:
        coordinate = np.zeros(3, dtype=float)
        coordinate[self.first_axis] = float(first)
        coordinate[self.second_axis] = float(second)
        coordinate[self.hidden_axis] = 0.0
        return coordinate

    def evaluate(self, first: float, second: float) -> float:
        key = self._visible_key(first, second)
        cached = self._cache.get(key)
        if cached is not None:
            return float(cached)
        value = self.group_evaluator.survival_fraction(
            self._full_coordinate(*key)
        )
        self._cache[key] = float(value)
        return float(value)

    @property
    def cached_point_count(self) -> int:
        return len(self._cache)

    def regular_grid(
        self,
        lower: float,
        upper: float,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(float(lower), float(upper), int(size))
        field = np.empty((axis.size, axis.size), dtype=float)
        for first_index, first in enumerate(axis):
            for second_index, second in enumerate(axis):
                field[first_index, second_index] = self.evaluate(
                    float(first), float(second)
                )
        return axis, field


def _cell_brackets(values: Sequence[float], threshold: float) -> bool:
    selected = np.asarray(values, dtype=float)
    if selected.ndim != 1 or selected.size == 0 or np.any(~np.isfinite(selected)):
        raise ValueError("cell values must be finite and one-dimensional")
    low = float(np.min(selected))
    high = float(np.max(selected))
    return bool(low <= float(threshold) <= high and low < high)


def _boundary_present(field: np.ndarray, threshold: float) -> bool:
    value = np.asarray(field, dtype=float)
    if value.ndim != 2 or min(value.shape) < 2 or np.any(~np.isfinite(value)):
        raise ValueError("survival-fraction field must be finite 2-D")
    for first in range(value.shape[0] - 1):
        for second in range(value.shape[1] - 1):
            corners = (
                value[first, second],
                value[first + 1, second],
                value[first + 1, second + 1],
                value[first, second + 1],
            )
            if _cell_brackets(corners, threshold):
                return True
    return False


def _next_nested_size(size: int) -> int:
    return 2 * (int(size) - 1) + 1


def _grid_cells(axis: np.ndarray, field: np.ndarray) -> list[AdaptiveCell]:
    cells = []
    for first in range(axis.size - 1):
        for second in range(axis.size - 1):
            cells.append(
                AdaptiveCell(
                    float(axis[first]),
                    float(axis[first + 1]),
                    float(axis[second]),
                    float(axis[second + 1]),
                    (
                        float(field[first, second]),
                        float(field[first + 1, second]),
                        float(field[first + 1, second + 1]),
                        float(field[first, second + 1]),
                    ),
                )
            )
    return cells


def _cell_from_bounds(
    evaluator: SliceGridEvaluator,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> AdaptiveCell:
    return AdaptiveCell(
        float(x0),
        float(x1),
        float(y0),
        float(y1),
        (
            evaluator.evaluate(float(x0), float(y0)),
            evaluator.evaluate(float(x1), float(y0)),
            evaluator.evaluate(float(x1), float(y1)),
            evaluator.evaluate(float(x0), float(y1)),
        ),
    )


def _split_cell(
    evaluator: SliceGridEvaluator,
    cell: AdaptiveCell,
) -> tuple[AdaptiveCell, AdaptiveCell, AdaptiveCell, AdaptiveCell]:
    xm = 0.5 * (cell.x0 + cell.x1)
    ym = 0.5 * (cell.y0 + cell.y1)
    return (
        _cell_from_bounds(evaluator, cell.x0, xm, cell.y0, ym),
        _cell_from_bounds(evaluator, xm, cell.x1, cell.y0, ym),
        _cell_from_bounds(evaluator, xm, cell.x1, ym, cell.y1),
        _cell_from_bounds(evaluator, cell.x0, xm, ym, cell.y1),
    )


def _refine_boundary_to_target(
    evaluator: SliceGridEvaluator,
    base_cells: Sequence[AdaptiveCell],
    threshold: float,
    starting_grid_size: int,
    target_grid_size: int,
) -> tuple[
    tuple[AdaptiveCell, ...],
    tuple[AdaptiveCell, ...],
    tuple[LocalRefinementLevel, ...],
    int,
]:
    boundary = [cell for cell in base_cells if _cell_brackets(cell.values, threshold)]
    leaves = [cell for cell in base_cells if not _cell_brackets(cell.values, threshold)]
    diagnostics = []
    effective = int(starting_grid_size)
    level = 0
    current = boundary
    while current and effective < int(target_grid_size):
        before = evaluator.cached_point_count
        next_boundary: list[AdaptiveCell] = []
        for cell in current:
            for child in _split_cell(evaluator, cell):
                if _cell_brackets(child.values, threshold):
                    next_boundary.append(child)
                else:
                    leaves.append(child)
        after = evaluator.cached_point_count
        level += 1
        effective = _next_nested_size(effective)
        diagnostics.append(
            LocalRefinementLevel(
                level=level,
                equivalent_grid_size=effective,
                input_boundary_cell_count=len(current),
                output_boundary_cell_count=len(next_boundary),
                new_point_count=int(after - before),
                total_cached_point_count=int(after),
            )
        )
        current = next_boundary
    leaves.extend(current)
    return tuple(leaves), tuple(current), tuple(diagnostics), int(effective)


def _triangle_segment(
    points: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> Optional[np.ndarray]:
    intersections: list[np.ndarray] = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        va = float(values[first] - threshold)
        vb = float(values[second] - threshold)
        if va == 0.0:
            intersections.append(points[first])
        if vb == 0.0:
            intersections.append(points[second])
        if va * vb < 0.0:
            fraction = -va / (vb - va)
            intersections.append(
                points[first] + fraction * (points[second] - points[first])
            )
    unique: list[np.ndarray] = []
    for point in intersections:
        if not any(np.linalg.norm(point - other) <= 32.0 * _EPS for other in unique):
            unique.append(np.asarray(point, dtype=float))
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        best = None
        best_distance = -1.0
        for first in range(len(unique)):
            for second in range(first + 1, len(unique)):
                distance = float(np.linalg.norm(unique[first] - unique[second]))
                if distance > best_distance:
                    best_distance = distance
                    best = (unique[first], unique[second])
        assert best is not None
        return np.asarray(best, dtype=float)
    return np.asarray((unique[0], unique[1]), dtype=float)


def _boundary_segments_from_cells(
    cells: Sequence[AdaptiveCell],
    threshold: float,
) -> np.ndarray:
    """Piecewise-linear boundary using only already-evaluated cell corners."""

    segments: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for cell in cells:
        center = np.asarray(
            (0.5 * (cell.x0 + cell.x1), 0.5 * (cell.y0 + cell.y1)),
            dtype=float,
        )
        # Bilinear-center interpolation avoids an extra expensive gain query.
        center_value = float(np.mean(np.asarray(cell.values, dtype=float)))
        points = (
            np.asarray((cell.x0, cell.y0), dtype=float),
            np.asarray((cell.x1, cell.y0), dtype=float),
            np.asarray((cell.x1, cell.y1), dtype=float),
            np.asarray((cell.x0, cell.y1), dtype=float),
        )
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            triangle_points = np.asarray((points[first], points[second], center))
            triangle_values = np.asarray(
                (cell.values[first], cell.values[second], center_value),
                dtype=float,
            )
            segment = _triangle_segment(triangle_points, triangle_values, threshold)
            if segment is None:
                continue
            canonical = segment.copy()
            if tuple(canonical[1]) < tuple(canonical[0]):
                canonical = canonical[::-1]
            key = tuple(np.round(canonical.reshape(-1), decimals=13))
            if key in seen:
                continue
            seen.add(key)
            segments.append(segment)
    if not segments:
        return np.empty((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def _adaptive_projection_grid(
    evaluator: SliceGridEvaluator,
    *,
    name: str,
    first_axis: int,
    second_axis: int,
    hidden_axis: int,
    lower: float,
    upper: float,
    threshold: float,
    maximum_grid_size: int,
    final_boundary_grid_size: int,
) -> ProjectionGrid:
    levels: list[GridLevel] = []
    size = 3
    final_axis = None
    final_field = None
    boundary = False
    while True:
        before = evaluator.cached_point_count
        axis, field = evaluator.regular_grid(lower, upper, size)
        after = evaluator.cached_point_count
        boundary = _boundary_present(field, threshold)
        levels.append(
            GridLevel(
                size=int(size),
                new_point_count=int(after - before),
                total_cached_point_count=int(after),
                minimum_survival_fraction=float(np.min(field)),
                maximum_survival_fraction=float(np.max(field)),
                boundary_present=bool(boundary),
            )
        )
        final_axis = axis
        final_field = field
        # Once a boundary exists, global refinement stops immediately.
        if boundary or size >= int(maximum_grid_size):
            break
        size = _next_nested_size(size)

    assert final_axis is not None and final_field is not None
    base_cells = _grid_cells(final_axis, final_field)
    first_seen = int(final_axis.size) if boundary else None
    local_levels: tuple[LocalRefinementLevel, ...] = ()
    if boundary:
        adaptive_cells, final_boundary_cells, local_levels, effective = (
            _refine_boundary_to_target(
                evaluator,
                base_cells,
                threshold,
                int(final_axis.size),
                int(final_boundary_grid_size),
            )
        )
        segments = _boundary_segments_from_cells(final_boundary_cells, threshold)
        stop_reason = "boundary_found_then_refined_locally_to_target_spacing"
    else:
        adaptive_cells = tuple(base_cells)
        final_boundary_cells = ()
        effective = int(final_axis.size)
        segments = np.empty((0, 2, 2), dtype=float)
        stop_reason = "no_boundary_detected_through_global_17x17_search"

    hidden_gain = float(evaluator.group_evaluator.base_gain[hidden_axis])
    return ProjectionGrid(
        name=str(name),
        first_axis=int(first_axis),
        second_axis=int(second_axis),
        hidden_axis=int(hidden_axis),
        hidden_gain=hidden_gain,
        axis_log_ratio=np.asarray(final_axis, dtype=float),
        survival_fraction=np.asarray(final_field, dtype=float),
        global_levels=tuple(levels),
        boundary_first_seen_grid_size=first_seen,
        local_refinement_levels=local_levels,
        adaptive_cells=adaptive_cells,
        final_boundary_cells=tuple(final_boundary_cells),
        boundary_segments_log_ratio=segments,
        effective_local_equivalent_grid_size=int(effective),
        stop_reason=stop_reason,
    )


def _range_with_success(
    failure: GainTriple,
    success: GainTriple,
    scale_min: float,
    scale_max: float,
) -> tuple[float, float]:
    lower = float(scale_min)
    upper = float(scale_max)
    ratio = success.array() / failure.array()
    lower = min(lower, float(np.min(ratio)))
    upper = max(upper, float(np.max(ratio)))
    return math.log(lower), math.log(upper)


def _plot_projection(
    axis: Any,
    *,
    projection: ProjectionGrid,
    group_evaluator: ExactGroupGainEvaluator,
    failure_gain: GainTriple,
    success_gain: GainTriple,
    threshold: float,
    labels: Sequence[str],
) -> Any:
    first = projection.first_axis
    second = projection.second_axis
    base = group_evaluator.base_gain
    polygons = []
    values = []
    for cell in projection.adaptive_cells:
        x0 = base[first] * math.exp(cell.x0)
        x1 = base[first] * math.exp(cell.x1)
        y0 = base[second] * math.exp(cell.y0)
        y1 = base[second] * math.exp(cell.y1)
        polygons.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        values.append(cell.mean_value)
    collection = PolyCollection(
        polygons,
        array=np.asarray(values, dtype=float),
        cmap="RdYlGn",
        edgecolors=(0.0, 0.0, 0.0, 0.10),
        linewidths=0.25,
    )
    collection.set_clim(0.0, 1.0)
    axis.add_collection(collection)

    for index, segment in enumerate(projection.boundary_segments_log_ratio):
        axis.plot(
            base[first] * np.exp(segment[:, 0]),
            base[second] * np.exp(segment[:, 1]),
            color="black",
            linewidth=2.0,
            label=(f"{100.0 * threshold:.0f}% survival boundary" if index == 0 else None),
        )

    failure = failure_gain.array()
    success = success_gain.array()
    axis.plot(
        failure[first],
        failure[second],
        marker="x",
        markersize=9,
        markeredgewidth=2.0,
        linestyle="none",
        color="black",
        label="failure recorded",
    )
    axis.plot(
        success[first],
        success[second],
        marker="o",
        markersize=6,
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="black",
        label="success recorded",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(
        base[first] * math.exp(float(projection.axis_log_ratio[0])),
        base[first] * math.exp(float(projection.axis_log_ratio[-1])),
    )
    axis.set_ylim(
        base[second] * math.exp(float(projection.axis_log_ratio[0])),
        base[second] * math.exp(float(projection.axis_log_ratio[-1])),
    )
    axis.set_xlabel(labels[first])
    axis.set_ylabel(labels[second])
    hidden_label = labels[projection.hidden_axis]
    axis.set_title(
        "{} slice; {}={:.4g}; global {}x{}, local eq. {}".format(
            projection.name.upper(),
            hidden_label,
            projection.hidden_gain,
            projection.axis_log_ratio.size,
            projection.axis_log_ratio.size,
            projection.effective_local_equivalent_grid_size,
        )
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return collection


def _projection_payload(projection: ProjectionGrid) -> Mapping[str, Any]:
    adaptive_values = np.asarray(
        [value for cell in projection.adaptive_cells for value in cell.values],
        dtype=float,
    )
    return {
        "name": projection.name,
        "visible_axes": [projection.first_axis, projection.second_axis],
        "hidden_axis": projection.hidden_axis,
        "hidden_gain_fixed_at_failure_recorded": projection.hidden_gain,
        "global_grid_size_at_stop": int(projection.axis_log_ratio.size),
        "minimum_survival_fraction": float(np.min(adaptive_values)),
        "maximum_survival_fraction": float(np.max(adaptive_values)),
        "boundary_first_seen_global_grid_size": projection.boundary_first_seen_grid_size,
        "adaptive_leaf_cell_count": int(len(projection.adaptive_cells)),
        "final_boundary_cell_count": int(len(projection.final_boundary_cells)),
        "boundary_segment_count": int(projection.boundary_segments_log_ratio.shape[0]),
        "effective_local_equivalent_grid_size": projection.effective_local_equivalent_grid_size,
        "stop_reason": projection.stop_reason,
        "global_levels": [
            {
                "size": level.size,
                "new_point_count": level.new_point_count,
                "total_cached_point_count": level.total_cached_point_count,
                "minimum_survival_fraction": level.minimum_survival_fraction,
                "maximum_survival_fraction": level.maximum_survival_fraction,
                "boundary_present": level.boundary_present,
            }
            for level in projection.global_levels
        ],
        "local_refinement_levels": [
            {
                "level": level.level,
                "equivalent_grid_size": level.equivalent_grid_size,
                "input_boundary_cell_count": level.input_boundary_cell_count,
                "output_boundary_cell_count": level.output_boundary_cell_count,
                "new_point_count": level.new_point_count,
                "total_cached_point_count": level.total_cached_point_count,
            }
            for level in projection.local_refinement_levels
        ],
    }


def _adaptive_cell_array(cells: Sequence[AdaptiveCell]) -> np.ndarray:
    if not cells:
        return np.empty((0, 9), dtype=float)
    return np.asarray(
        [
            (
                cell.x0,
                cell.x1,
                cell.y0,
                cell.y1,
                *cell.values,
                cell.mean_value,
            )
            for cell in cells
        ],
        dtype=float,
    )


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    success_path = Path(arguments.success_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success_estimate = load_estimate_json(success_path)
    group = str(arguments.group)
    if group not in PID_GROUPS:
        raise ValueError("unknown PID group")

    failure_gains = _recorded_gains(estimate)
    success_gains = _recorded_gains(success_estimate)
    hybrid_failure_gains = dict(success_gains)
    hybrid_failure_gains[group] = failure_gains[group]
    failure_group_gain = failure_gains[group]
    success_group_gain = success_gains[group]
    lower, upper = _range_with_success(
        failure_group_gain,
        success_group_gain,
        float(arguments.scale_min),
        float(arguments.scale_max),
    )

    model_path = Path(estimate["input"]["vehicle_model"])
    vehicle_model = load_vehicle_model(model_path)
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
    quotient = draw_quotient_coordinates(
        estimate,
        str(arguments.covariance),
        int(arguments.samples),
        int(arguments.seed),
    )
    plants = quotient_to_scale_free_plants(
        estimate,
        quotient,
        vehicle_model.parameters,
        ScaleFreePlant,
    )

    group_evaluator = ExactGroupGainEvaluator(
        plants=plants,
        hybrid_failure_gains=hybrid_failure_gains,
        success_gains=success_gains,
        group=group,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        workers=int(arguments.workers),
    )
    try:
        failure_evaluation = group_evaluator.break_evaluation(np.zeros(3))
        success_evaluation = group_evaluator.break_evaluation(
            group_evaluator.success_log_ratio
        )
        threshold = float(arguments.alpha)

        projections = []
        for name, first, second, hidden, _first_label, _second_label, _hidden_label in PROJECTION_SPECS:
            slice_evaluator = SliceGridEvaluator(
                group_evaluator,
                first_axis=first,
                second_axis=second,
                hidden_axis=hidden,
            )
            projections.append(
                _adaptive_projection_grid(
                    slice_evaluator,
                    name=name,
                    first_axis=first,
                    second_axis=second,
                    hidden_axis=hidden,
                    lower=lower,
                    upper=upper,
                    threshold=threshold,
                    maximum_grid_size=int(arguments.max_grid_size),
                    final_boundary_grid_size=int(arguments.final_boundary_grid_size),
                )
            )

        output = (
            Path(arguments.output_dir).expanduser().resolve()
            if arguments.output_dir is not None
            else estimate_path.parent / "pid_gain_contour" / group
        )
        output.mkdir(parents=True, exist_ok=True)

        labels = (f"{group} P", f"{group} I", f"{group} D")
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
        color = None
        for axis, projection in zip(axes, projections):
            color = _plot_projection(
                axis,
                projection=projection,
                group_evaluator=group_evaluator,
                failure_gain=failure_group_gain,
                success_gain=success_group_gain,
                threshold=threshold,
                labels=labels,
            )
        assert color is not None
        figure.colorbar(color, ax=axes, shrink=0.92).set_label(
            "survival fraction among all-success-stable plant samples"
        )
        figure.suptitle(
            "{} / {} / single-group survival; {:.1f}% boundary".format(
                estimate["case_name"],
                group,
                100.0 * threshold,
            )
        )
        figure.savefig(output / "gain_contour.png", dpi=180)
        plt.close(figure)

        projection_arrays: dict[str, np.ndarray] = {}
        for projection in projections:
            projection_arrays[f"{projection.name}_axis_log_ratio"] = projection.axis_log_ratio
            projection_arrays[
                f"{projection.name}_survival_fraction"
            ] = projection.survival_fraction
            projection_arrays[
                f"{projection.name}_boundary_segments_log_ratio"
            ] = projection.boundary_segments_log_ratio
            projection_arrays[
                f"{projection.name}_adaptive_cells_log_ratio_and_survival"
            ] = _adaptive_cell_array(projection.adaptive_cells)
        np.savez_compressed(
            output / "gain_contour.npz",
            base_gain=group_evaluator.base_gain,
            success_gain=group_evaluator.success_gain,
            success_group_log_ratio=group_evaluator.success_log_ratio,
            success_baseline_stable_mask=group_evaluator.success_stable_mask,
            failure_group_stable_mask=failure_evaluation.stable_mask,
            caused_break_mask=failure_evaluation.caused_break_mask,
            projection_names=np.asarray([item.name for item in projections]),
            **projection_arrays,
        )

        baseline_stable_count = group_evaluator.success_stable_count
        conditional_failure_fraction = float(
            failure_evaluation.caused_break_count / baseline_stable_count
        )
        payload = {
            "schema": CONTOUR_SCHEMA,
            "source_commit": source_commit(_PROJECT_ROOT),
            "case_name": str(estimate["case_name"]),
            "group": group,
            "estimate_json": str(estimate_path),
            "success_json": str(success_path),
            "first_order_time_constant_seconds": float(
                estimate["actuator_model"]["thrust_time_constant_seconds"]
            ),
            "controller_dt_seconds": controller_dt,
            "sample_count": int(arguments.samples),
            "covariance": str(arguments.covariance),
            "seed": int(arguments.seed),
            "alpha_reference": threshold,
            "survival_boundary_fraction": threshold,
            "definition": {
                "baseline": "all four PID groups use success-flight recorded gains on this bag's plant distribution",
                "single_group_intervention": "only this group is varied; the other three groups stay at success-flight gains",
                "caused_break_sample": "stable under all-success baseline and unstable under the single-group intervention",
                "survival_fraction": "one minus caused-break count divided by the number of all-success-stable samples",
                "plot": "other three groups fixed at success gains; hidden selected-group gain fixed at its failure recorded value",
            },
            "method": {
                "closed_loop_model": "full_26_state_sampled_data_first_order_actuator",
                "linearization": "exact active-branch analytic Jacobian of the implemented sampled closed-loop map",
                "gain_surrogate": None,
                "eigensystems": "batched numpy.linalg.eigvals on the complete sample-by-26-by-26 Jacobian stack",
                "global_search": "nested 3x3 -> 5x5 -> 9x9 -> 17x17, stopping globally as soon as the survival boundary is detected",
                "local_refinement": "after first detection, only boundary cells are split dyadically until 33x33-equivalent spacing by default",
                "grid_reuse": "all repeated gain points are cached and never reevaluated",
                "boundary": "piecewise-linear alpha-survival segments from final locally refined cells",
                "heatmap": "adaptive leaf cells retain their survival values; only boundary neighborhoods become finer",
                "topology_limitation": "if no sampled cell brackets alpha through the 17x17 global search, sub-cell boundary islands are not certified absent",
            },
            "gain_domain": {
                "ratio_min": float(math.exp(lower)),
                "ratio_max": float(math.exp(upper)),
                "coordinate": "log(gain / this-bag recorded group gain)",
            },
            "recorded_failure_group_gain": {
                "p_gain": failure_group_gain.p,
                "i_gain": failure_group_gain.i,
                "d_gain": failure_group_gain.d,
            },
            "recorded_success_group_gain": {
                "p_gain": success_group_gain.p,
                "i_gain": success_group_gain.i,
                "d_gain": success_group_gain.d,
            },
            "all_success_baseline": {
                "stable_count": baseline_stable_count,
                "unstable_count": int(arguments.samples) - baseline_stable_count,
                "stable_fraction": float(np.mean(group_evaluator.success_stable_mask)),
            },
            "failure_gain_single_group_intervention": {
                "unstable_count": failure_evaluation.unstable_count,
                "unstable_fraction": failure_evaluation.unstable_fraction,
                "caused_break_count": failure_evaluation.caused_break_count,
                "caused_break_fraction_of_all_samples": failure_evaluation.caused_break_fraction,
                "caused_break_fraction_conditioned_on_success_baseline": conditional_failure_fraction,
                "survival_fraction_conditioned_on_success_baseline": 1.0 - conditional_failure_fraction,
                "counterdirection_rescued_count": int(
                    np.count_nonzero(
                        ~group_evaluator.success_stable_mask
                        & failure_evaluation.stable_mask
                    )
                ),
            },
            "success_gain_check": {
                "caused_break_count": success_evaluation.caused_break_count,
                "unstable_count": success_evaluation.unstable_count,
                "survival_fraction_conditioned_on_success_baseline": 1.0,
            },
            "analytic_active_branch_diagnostics": {
                "evaluated_gain_point_count": group_evaluator.evaluated_gain_point_count,
                "gain_points_with_any_sample_near_piecewise_kink": group_evaluator.gain_points_with_any_near_kink,
            },
            "projections": [_projection_payload(item) for item in projections],
            "files": {
                "npz": str(output / "gain_contour.npz"),
                "png": str(output / "gain_contour.png"),
            },
        }
        write_json(output / "gain_contour.json", payload)
        return payload
    finally:
        group_evaluator.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-json", type=Path, required=True)
    parser.add_argument("--success-json", type=Path, required=True)
    parser.add_argument("--group", choices=PID_GROUPS, default=DEFAULT_GROUP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--covariance",
        choices=COVARIANCE_NAMES,
        default="conservative_fusion",
    )
    parser.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    parser.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=DEFAULT_MAX_GRID_SIZE,
    )
    parser.add_argument(
        "--final-boundary-grid-size",
        type=int,
        default=DEFAULT_FINAL_BOUNDARY_GRID_SIZE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _is_dyadic_nested_size(value: int) -> bool:
    selected = int(value)
    if selected < 3:
        return False
    intervals = selected - 1
    return intervals > 0 and (intervals & (intervals - 1)) == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if not (0.0 < arguments.alpha < 1.0):
        raise SystemExit("--alpha must lie in (0,1)")
    if arguments.samples <= 0:
        raise SystemExit("--samples must be positive")
    if arguments.scale_min <= 0.0 or arguments.scale_max <= arguments.scale_min:
        raise SystemExit("gain scale bounds must satisfy 0 < min < max")
    if not _is_dyadic_nested_size(arguments.max_grid_size):
        raise SystemExit("--max-grid-size must be 2^k + 1 and at least three")
    if not _is_dyadic_nested_size(arguments.final_boundary_grid_size):
        raise SystemExit(
            "--final-boundary-grid-size must be 2^k + 1 and at least three"
        )
    if arguments.final_boundary_grid_size < arguments.max_grid_size:
        raise SystemExit(
            "--final-boundary-grid-size must be at least --max-grid-size"
        )
    if arguments.workers <= 0:
        raise SystemExit("--workers must be positive")
    payload = analyze(arguments)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
