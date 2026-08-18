#!/usr/bin/env python3
"""Adaptive group-only PID break maps for the first-order Gimbalrotor model.

The closed-loop model is the full 26-state sampled-data system used by the
first-order-lag pole validation.  For one selected PID gain group, the other
three groups are fixed at the recorded gains of the successful flight.  The
selected group is varied around the recorded gain of the analyzed failure bag.

For plant sample n, let F_n(K) be the full 26-state one-step Jacobian and let
rho_n(K) be its spectral radius.  The all-success controller defines a baseline
stable mask.  At a queried gain of one selected group, a sample is counted as
"caused to break by this group" exactly when

    rho_n(K_success) < 1  and  rho_n(K_group_only) >= 1.

Thus the plotted value is a directly interpretable fraction of the requested
plant samples, not the stability fraction of the whole failure controller.

The selected group's three 2-D views are slices through its recorded failure
point:

    PI: D = D_failure
    ID: P = P_failure
    DP: I = I_failure

The global raster is evaluated on nested dyadic grids

    3x3 -> 5x5 -> 9x9 -> 17x17

and every old point is reused exactly.  Global refinement continues through
the configured maximum grid even when a boundary appears early.  If the 5%
boundary is detected by that stage, only cells near the boundary are split
dyadically three more times by default.  The final boundary line is built from
those locally refined cells, giving eight times finer spacing than the 17x17
global grid without evaluating a dense 129x129 raster.

Before any raster evaluation, the expensive nonlinear one-step Jacobian is
audited for affine dependence on the selected raw P/I/D gains.  Samples that
pass the audit use cached 26x26 affine matrices.  Samples that do not pass fall
back to the original full 26-state evaluator at each queried point; they are
never discarded.
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
    NUMERICAL_SAMPLE_EXCEPTIONS,
    _analyze_plant,
    decompose_thrust_delay,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


CONTOUR_SCHEMA = "grape-param-estim/first-order-lag-pid-group-break/v2"
DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_GROUP = "roll_pitch"
DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 3.0
DEFAULT_AFFINE_BASIS_RATIO = 1.12
DEFAULT_MAX_GRID_SIZE = 17
DEFAULT_LOCAL_REFINEMENT_LEVELS = 3
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
PROJECTION_SPECS = (
    ("pi", 0, 1, 2, "P", "I", "D"),
    ("id", 1, 2, 0, "I", "D", "P"),
    ("dp", 2, 0, 1, "D", "P", "I"),
)

_EPS = np.finfo(float).eps
_AFFINE_AUDIT_TOLERANCE = float(4096.0 * _EPS ** (2.0 / 3.0))


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
class AffineSampleResult:
    index: int
    base_matrix: np.ndarray
    gain_derivatives: np.ndarray
    audit_relative_errors: np.ndarray
    affine_valid: bool
    piecewise_near_kink: bool


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
    minimum_break_fraction: float
    maximum_break_fraction: float
    boundary_present: bool


@dataclass(frozen=True)
class LocalRefinementLevel:
    level: int
    input_cell_count: int
    boundary_cell_count: int
    new_point_count: int
    total_cached_point_count: int


@dataclass(frozen=True)
class ProjectionGrid:
    name: str
    first_axis: int
    second_axis: int
    hidden_axis: int
    hidden_gain: float
    axis_log_ratio: np.ndarray
    break_fraction: np.ndarray
    levels: tuple[GridLevel, ...]
    boundary_first_seen_grid_size: Optional[int]
    initial_boundary_candidate_cell_count: int
    topology_probe_new_point_count: int
    topology_probe_detected_cell_count: int
    local_refinement_levels: tuple[LocalRefinementLevel, ...]
    boundary_segments_log_ratio: np.ndarray
    boundary_leaf_cell_count: int
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
    gains: Mapping[str, GainTriple],
) -> Any:
    return SimpleNamespace(
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_configuration=_configuration(gains),
    )


def _exact_matrix(
    *,
    plant: ScaleFreePlant,
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_dt: float,
    gains: Mapping[str, GainTriple],
) -> tuple[np.ndarray, bool]:
    result = _analyze_plant(
        scale_free=plant,
        inputs=_inputs(vehicle_model, actuator_parameters, gains),
        controller_dt=float(controller_dt),
        delay=decompose_thrust_delay(0.0, float(controller_dt)),
        fd_check=False,
    )
    matrix = result.get("jacobian")
    trim = result.get("trim")
    if matrix is None or trim is None or not bool(trim.equilibrium_valid):
        raise RuntimeError("full 26-state pole Jacobian is unavailable at the requested gain")
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (26, 26) or np.any(~np.isfinite(selected)):
        raise RuntimeError("first-order closed-loop Jacobian must be finite 26x26")
    return selected, bool(trim.piecewise_linearization_near_kink)


def _relative_matrix_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    numerator = float(np.linalg.norm(actual - predicted, ord="fro"))
    denominator = float(np.linalg.norm(actual, ord="fro"))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _audit_triples(
    base: GainTriple,
    success: GainTriple,
    scale_min: float,
    scale_max: float,
) -> tuple[GainTriple, GainTriple]:
    center = base.array()
    success_value = success.array()
    midpoint = math.sqrt(float(scale_min) * float(scale_max))
    first = GainTriple(*success_value)
    second = GainTriple(*(center * np.asarray((scale_max, midpoint, scale_min))))
    return first, second


def _build_affine_sample(task: tuple[Any, ...]) -> AffineSampleResult:
    (
        index,
        plant,
        vehicle_model,
        actuator_parameters,
        controller_dt,
        hybrid_failure_gains,
        group,
        success_group_gain,
        scale_min,
        scale_max,
        basis_ratio,
    ) = task
    base = hybrid_failure_gains[group]
    base_values = base.array()
    base_matrix, piecewise = _exact_matrix(
        plant=plant,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        gains=hybrid_failure_gains,
    )
    derivatives = np.empty((3, 26, 26), dtype=float)
    for axis in range(3):
        perturbed = base_values.copy()
        perturbed[axis] *= float(basis_ratio)
        delta = float(perturbed[axis] - base_values[axis])
        if delta == 0.0:
            raise RuntimeError("affine basis perturbation vanished")
        matrix, near_kink = _exact_matrix(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(
                hybrid_failure_gains,
                group,
                GainTriple(*perturbed),
            ),
        )
        derivatives[axis] = (matrix - base_matrix) / delta
        piecewise = piecewise or near_kink

    errors = []
    for triple in _audit_triples(base, success_group_gain, scale_min, scale_max):
        values = triple.array()
        actual, near_kink = _exact_matrix(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(hybrid_failure_gains, group, triple),
        )
        predicted = base_matrix + np.einsum(
            "j,jkl->kl", values - base_values, derivatives
        )
        errors.append(_relative_matrix_error(actual, predicted))
        piecewise = piecewise or near_kink
    audit = np.asarray(errors, dtype=float)
    valid = bool(
        np.all(np.isfinite(audit))
        and float(np.max(audit)) <= _AFFINE_AUDIT_TOLERANCE
    )
    return AffineSampleResult(
        index=int(index),
        base_matrix=base_matrix,
        gain_derivatives=derivatives,
        audit_relative_errors=audit,
        affine_valid=valid,
        piecewise_near_kink=piecewise,
    )


class GroupGainEvaluator:
    """Full 26-state spectral radii with audited affine matrix reuse."""

    def __init__(
        self,
        *,
        plants: Sequence[ScaleFreePlant],
        sample_results: Sequence[AffineSampleResult],
        hybrid_failure_gains: Mapping[str, GainTriple],
        success_gains: Mapping[str, GainTriple],
        group: str,
        vehicle_model: Any,
        actuator_parameters: Any,
        controller_dt: float,
    ) -> None:
        self.plants = tuple(plants)
        ordered = sorted(sample_results, key=lambda item: item.index)
        if len(ordered) != len(self.plants) or any(
            item.index != index for index, item in enumerate(ordered)
        ):
            raise ValueError("affine sample results do not align with plant samples")
        self.sample_results = tuple(ordered)
        self.base_matrix = np.asarray([item.base_matrix for item in ordered], dtype=float)
        self.derivative = np.asarray(
            [item.gain_derivatives for item in ordered], dtype=float
        )
        self.affine_valid = np.asarray([item.affine_valid for item in ordered], dtype=bool)
        self.hybrid_failure_gains = dict(hybrid_failure_gains)
        self.success_gains = dict(success_gains)
        self.group = str(group)
        self.base_gain = self.hybrid_failure_gains[self.group].array()
        self.success_gain = self.success_gains[self.group].array()
        self.vehicle_model = vehicle_model
        self.actuator_parameters = actuator_parameters
        self.controller_dt = float(controller_dt)
        self._radius_cache: dict[tuple[float, float, float], np.ndarray] = {}
        self._break_cache: dict[tuple[float, float, float], BreakEvaluation] = {}
        success_ratio = np.log(self.success_gain / self.base_gain)
        self.success_log_ratio = success_ratio
        self.success_log_radius = self.log_spectral_radius(success_ratio)
        self.success_stable_mask = self.success_log_radius < 0.0

    @property
    def fallback_count(self) -> int:
        return int(np.count_nonzero(~self.affine_valid))

    @property
    def sample_count(self) -> int:
        return len(self.plants)

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

    def _matrices(self, log_ratio: Sequence[float]) -> np.ndarray:
        gain = self.gain_from_log_ratio(log_ratio)
        delta = gain - self.base_gain
        matrices = self.base_matrix + np.einsum("j,njkl->nkl", delta, self.derivative)
        fallback = np.flatnonzero(~self.affine_valid)
        if fallback.size:
            triple = GainTriple(*gain)
            gains = _replace_group(self.hybrid_failure_gains, self.group, triple)
            for index in fallback:
                matrices[index], _near_kink = _exact_matrix(
                    plant=self.plants[int(index)],
                    vehicle_model=self.vehicle_model,
                    actuator_parameters=self.actuator_parameters,
                    controller_dt=self.controller_dt,
                    gains=gains,
                )
        return matrices

    def log_spectral_radius(self, log_ratio: Sequence[float]) -> np.ndarray:
        key = self._key(log_ratio)
        cached = self._radius_cache.get(key)
        if cached is not None:
            return cached.copy()
        matrices = self._matrices(key)
        eigenvalues = np.linalg.eigvals(matrices)
        radii = np.max(np.abs(eigenvalues), axis=1)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise FloatingPointError("closed-loop spectral radius became invalid")
        result = np.log(radii)
        self._radius_cache[key] = result.copy()
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


class SliceGridEvaluator:
    """Cached 2-D group slice with nested dyadic refinement."""

    def __init__(
        self,
        group_evaluator: GroupGainEvaluator,
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
        # The hidden gain remains exactly at the recorded failure value.
        coordinate[self.hidden_axis] = 0.0
        return coordinate

    def evaluate(self, first: float, second: float) -> float:
        key = self._visible_key(first, second)
        cached = self._cache.get(key)
        if cached is not None:
            return float(cached)
        value = self.group_evaluator.break_evaluation(
            self._full_coordinate(*key)
        ).caused_break_fraction
        self._cache[key] = float(value)
        return float(value)

    @property
    def cached_point_count(self) -> int:
        return len(self._cache)

    def regular_grid(self, lower: float, upper: float, size: int) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError("break-fraction field must be finite 2-D")
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


@dataclass(frozen=True)
class _BoundaryCell:
    x0: float
    x1: float
    y0: float
    y1: float
    values: tuple[float, float, float, float]


def _cell_from_bounds(
    evaluator: SliceGridEvaluator,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> _BoundaryCell:
    values = (
        evaluator.evaluate(float(x0), float(y0)),
        evaluator.evaluate(float(x1), float(y0)),
        evaluator.evaluate(float(x1), float(y1)),
        evaluator.evaluate(float(x0), float(y1)),
    )
    return _BoundaryCell(
        float(x0),
        float(x1),
        float(y0),
        float(y1),
        tuple(float(value) for value in values),
    )


def _global_candidate_cells(
    evaluator: SliceGridEvaluator,
    axis: np.ndarray,
    field: np.ndarray,
    threshold: float,
) -> tuple[list[_BoundaryCell], int, int]:
    """Find threshold cells and probe unbracketed cell centers once.

    The center probe is a small topology audit.  It can detect a compact island
    whose four global-grid corners all lie on the same side of the threshold.
    It does not mathematically certify that no still-smaller island exists.
    """

    before = evaluator.cached_point_count
    candidates: list[_BoundaryCell] = []
    center_detected = 0
    for first in range(axis.size - 1):
        for second in range(axis.size - 1):
            cell = _BoundaryCell(
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
            if _cell_brackets(cell.values, threshold):
                candidates.append(cell)
                continue
            center_value = evaluator.evaluate(
                0.5 * (cell.x0 + cell.x1),
                0.5 * (cell.y0 + cell.y1),
            )
            if _cell_brackets((*cell.values, center_value), threshold):
                candidates.append(cell)
                center_detected += 1
    after = evaluator.cached_point_count
    return candidates, int(after - before), int(center_detected)


def _split_boundary_cell(
    evaluator: SliceGridEvaluator,
    cell: _BoundaryCell,
    threshold: float,
) -> list[_BoundaryCell]:
    xm = 0.5 * (cell.x0 + cell.x1)
    ym = 0.5 * (cell.y0 + cell.y1)
    children = (
        _cell_from_bounds(evaluator, cell.x0, xm, cell.y0, ym),
        _cell_from_bounds(evaluator, xm, cell.x1, cell.y0, ym),
        _cell_from_bounds(evaluator, xm, cell.x1, ym, cell.y1),
        _cell_from_bounds(evaluator, cell.x0, xm, ym, cell.y1),
    )
    return [child for child in children if _cell_brackets(child.values, threshold)]


def _refine_boundary_cells(
    evaluator: SliceGridEvaluator,
    cells: Sequence[_BoundaryCell],
    threshold: float,
    levels: int,
) -> tuple[list[_BoundaryCell], tuple[LocalRefinementLevel, ...]]:
    current = list(cells)
    diagnostics: list[LocalRefinementLevel] = []
    for level in range(1, int(levels) + 1):
        if not current:
            break
        before = evaluator.cached_point_count
        refined: list[_BoundaryCell] = []
        for cell in current:
            refined.extend(_split_boundary_cell(evaluator, cell, threshold))
        after = evaluator.cached_point_count
        diagnostics.append(
            LocalRefinementLevel(
                level=int(level),
                input_cell_count=int(len(current)),
                boundary_cell_count=int(len(refined)),
                new_point_count=int(after - before),
                total_cached_point_count=int(after),
            )
        )
        current = refined
    return current, tuple(diagnostics)


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
    evaluator: SliceGridEvaluator,
    cells: Sequence[_BoundaryCell],
    threshold: float,
) -> np.ndarray:
    segments: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for cell in cells:
        center = np.asarray(
            (0.5 * (cell.x0 + cell.x1), 0.5 * (cell.y0 + cell.y1)),
            dtype=float,
        )
        center_value = evaluator.evaluate(float(center[0]), float(center[1]))
        points = (
            np.asarray((cell.x0, cell.y0), dtype=float),
            np.asarray((cell.x1, cell.y0), dtype=float),
            np.asarray((cell.x1, cell.y1), dtype=float),
            np.asarray((cell.x0, cell.y1), dtype=float),
        )
        values = cell.values
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            triangle_points = np.asarray((points[first], points[second], center))
            triangle_values = np.asarray((values[first], values[second], center_value))
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
    local_refinement_levels: int,
) -> ProjectionGrid:
    if maximum_grid_size < 3:
        raise ValueError("maximum grid size must be at least three")
    if (maximum_grid_size - 1) & (maximum_grid_size - 2):
        raise ValueError("maximum grid size must be 2^k + 1")
    if local_refinement_levels < 0:
        raise ValueError("local refinement levels must be non-negative")

    levels: list[GridLevel] = []
    size = 3
    final_axis = None
    final_field = None
    first_seen = None
    while True:
        before = evaluator.cached_point_count
        axis, field = evaluator.regular_grid(lower, upper, size)
        after = evaluator.cached_point_count
        boundary = _boundary_present(field, threshold)
        if boundary and first_seen is None:
            first_seen = int(size)
        levels.append(
            GridLevel(
                size=int(size),
                new_point_count=int(after - before),
                total_cached_point_count=int(after),
                minimum_break_fraction=float(np.min(field)),
                maximum_break_fraction=float(np.max(field)),
                boundary_present=bool(boundary),
            )
        )
        final_axis = axis
        final_field = field
        if size >= maximum_grid_size:
            break
        next_size = _next_nested_size(size)
        if next_size > maximum_grid_size:
            next_size = maximum_grid_size
        if next_size == size:
            break
        size = next_size

    assert final_axis is not None and final_field is not None
    candidates, probe_new, probe_detected = _global_candidate_cells(
        evaluator,
        np.asarray(final_axis, dtype=float),
        np.asarray(final_field, dtype=float),
        threshold,
    )
    if first_seen is None and candidates:
        first_seen = int(final_axis.size)

    leaf_cells: list[_BoundaryCell] = list(candidates)
    local_levels: tuple[LocalRefinementLevel, ...] = ()
    if leaf_cells and local_refinement_levels:
        leaf_cells, local_levels = _refine_boundary_cells(
            evaluator,
            leaf_cells,
            threshold,
            local_refinement_levels,
        )
    segments = _boundary_segments_from_cells(evaluator, leaf_cells, threshold)
    completed_local_levels = len(local_levels)
    effective = int((final_axis.size - 1) * (2 ** completed_local_levels) + 1)
    if candidates and segments.size:
        stop_reason = "boundary_refined_locally"
    elif candidates:
        stop_reason = "boundary_candidate_without_final_segment"
    else:
        stop_reason = "no_boundary_detected_through_global_and_center_probe"

    hidden_gain = float(evaluator.group_evaluator.base_gain[hidden_axis])
    return ProjectionGrid(
        name=str(name),
        first_axis=int(first_axis),
        second_axis=int(second_axis),
        hidden_axis=int(hidden_axis),
        hidden_gain=hidden_gain,
        axis_log_ratio=np.asarray(final_axis, dtype=float),
        break_fraction=np.asarray(final_field, dtype=float),
        levels=tuple(levels),
        boundary_first_seen_grid_size=first_seen,
        initial_boundary_candidate_cell_count=int(len(candidates)),
        topology_probe_new_point_count=int(probe_new),
        topology_probe_detected_cell_count=int(probe_detected),
        local_refinement_levels=local_levels,
        boundary_segments_log_ratio=segments,
        boundary_leaf_cell_count=int(len(leaf_cells)),
        effective_local_equivalent_grid_size=effective,
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
    group_evaluator: GroupGainEvaluator,
    failure_gain: GainTriple,
    success_gain: GainTriple,
    threshold: float,
    labels: Sequence[str],
) -> Any:
    log_axis = projection.axis_log_ratio
    first_gain = group_evaluator.base_gain[projection.first_axis] * np.exp(log_axis)
    second_gain = group_evaluator.base_gain[projection.second_axis] * np.exp(log_axis)
    x, y = np.meshgrid(first_gain, second_gain, indexing="ij")
    field = projection.break_fraction
    color = axis.pcolormesh(x, y, field, shading="auto", vmin=0.0, vmax=1.0)
    for segment in projection.boundary_segments_log_ratio:
        axis.plot(
            group_evaluator.base_gain[projection.first_axis] * np.exp(segment[:, 0]),
            group_evaluator.base_gain[projection.second_axis] * np.exp(segment[:, 1]),
            linewidth=2.2,
        )
    failure = failure_gain.array()
    success = success_gain.array()
    axis.plot(
        failure[projection.first_axis],
        failure[projection.second_axis],
        marker="x",
        markersize=9,
        markeredgewidth=2.0,
        label="failure recorded",
    )
    axis.plot(
        success[projection.first_axis],
        success[projection.second_axis],
        marker="o",
        markersize=6,
        label="success recorded",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(labels[projection.first_axis])
    axis.set_ylabel(labels[projection.second_axis])
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
    return color


def _projection_payload(projection: ProjectionGrid) -> Mapping[str, Any]:
    return {
        "name": projection.name,
        "visible_axes": [projection.first_axis, projection.second_axis],
        "hidden_axis": projection.hidden_axis,
        "hidden_gain_fixed_at_failure_recorded": projection.hidden_gain,
        "global_grid_size": int(projection.axis_log_ratio.size),
        "minimum_break_fraction": float(np.min(projection.break_fraction)),
        "maximum_break_fraction": float(np.max(projection.break_fraction)),
        "boundary_first_seen_global_grid_size": projection.boundary_first_seen_grid_size,
        "initial_boundary_candidate_cell_count": projection.initial_boundary_candidate_cell_count,
        "topology_center_probe_new_point_count": projection.topology_probe_new_point_count,
        "topology_center_probe_detected_cell_count": projection.topology_probe_detected_cell_count,
        "boundary_segment_count": int(projection.boundary_segments_log_ratio.shape[0]),
        "boundary_leaf_cell_count": projection.boundary_leaf_cell_count,
        "effective_local_equivalent_grid_size": projection.effective_local_equivalent_grid_size,
        "stop_reason": projection.stop_reason,
        "global_levels": [
            {
                "size": level.size,
                "new_point_count": level.new_point_count,
                "total_cached_point_count": level.total_cached_point_count,
                "minimum_break_fraction": level.minimum_break_fraction,
                "maximum_break_fraction": level.maximum_break_fraction,
                "boundary_present": level.boundary_present,
            }
            for level in projection.levels
        ],
        "local_refinement_levels": [
            {
                "level": level.level,
                "input_cell_count": level.input_cell_count,
                "boundary_cell_count": level.boundary_cell_count,
                "new_point_count": level.new_point_count,
                "total_cached_point_count": level.total_cached_point_count,
            }
            for level in projection.local_refinement_levels
        ],
    }


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
    # Other groups are always held at the successful-flight gains.  Only the
    # selected group is placed at this bag's failure gain at the affine origin.
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

    tasks = [
        (
            index,
            plant,
            vehicle_model,
            actuator_parameters,
            controller_dt,
            hybrid_failure_gains,
            group,
            success_group_gain,
            math.exp(lower),
            math.exp(upper),
            float(arguments.affine_basis_ratio),
        )
        for index, plant in enumerate(plants)
    ]
    workers = min(int(arguments.workers), len(tasks))
    if workers <= 1:
        sample_results = [_build_affine_sample(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            sample_results = list(executor.map(_build_affine_sample, tasks, chunksize=1))

    group_evaluator = GroupGainEvaluator(
        plants=plants,
        sample_results=sample_results,
        hybrid_failure_gains=hybrid_failure_gains,
        success_gains=success_gains,
        group=group,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
    )

    failure_evaluation = group_evaluator.break_evaluation(np.zeros(3))
    success_evaluation = group_evaluator.break_evaluation(
        group_evaluator.success_log_ratio
    )
    threshold = 1.0 - float(arguments.alpha)

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
                local_refinement_levels=int(arguments.local_refinement_levels),
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
        "fraction of all plant samples newly unstable vs all-success controller"
    )
    figure.suptitle(
        "{} / {} / group-only caused-break fraction; {:.1f}% boundary".format(
            estimate["case_name"],
            group,
            100.0 * threshold,
        )
    )
    figure.savefig(output / "gain_contour.png", dpi=180)
    plt.close(figure)

    audit_errors = np.asarray(
        [item.audit_relative_errors for item in sample_results], dtype=float
    )
    projection_arrays: dict[str, np.ndarray] = {}
    for projection in projections:
        projection_arrays[f"{projection.name}_axis_log_ratio"] = projection.axis_log_ratio
        projection_arrays[f"{projection.name}_break_fraction"] = projection.break_fraction
        projection_arrays[
            f"{projection.name}_boundary_segments_log_ratio"
        ] = projection.boundary_segments_log_ratio
    np.savez_compressed(
        output / "gain_contour.npz",
        base_gain=group_evaluator.base_gain,
        success_gain=group_evaluator.success_gain,
        success_group_log_ratio=group_evaluator.success_log_ratio,
        affine_valid=group_evaluator.affine_valid,
        affine_audit_relative_error=audit_errors,
        piecewise_near_kink=np.asarray(
            [item.piecewise_near_kink for item in sample_results], dtype=bool
        ),
        success_baseline_stable_mask=group_evaluator.success_stable_mask,
        failure_group_stable_mask=failure_evaluation.stable_mask,
        caused_break_mask=failure_evaluation.caused_break_mask,
        projection_names=np.asarray([item.name for item in projections]),
        **projection_arrays,
    )

    baseline_stable_count = int(np.count_nonzero(group_evaluator.success_stable_mask))
    conditional_denominator = max(baseline_stable_count, 1)
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
        "alpha_reference": float(arguments.alpha),
        "caused_break_boundary_fraction": threshold,
        "definition": {
            "baseline": "all four PID groups use success-flight recorded gains on this bag's plant distribution",
            "single_group_intervention": "only this group is replaced by the analyzed bag's recorded gain",
            "caused_break_sample": "stable under all-success baseline and unstable under the single-group intervention",
            "plot": "other three groups fixed at success gains; hidden selected-group gain fixed at its failure recorded value",
        },
        "method": {
            "closed_loop_model": "full_26_state_sampled_data_first_order_actuator",
            "raster": "nested dyadic global grid plus boundary-cell local refinement",
            "global_grid_sequence": "3x3 -> 5x5 -> 9x9 -> 17x17 by default; every global level is evaluated",
            "grid_reuse": "all repeated global and local points are cached and never reevaluated",
            "topology_probe": "after the final global grid, every unbracketed cell center is probed once before declaring the boundary absent",
            "local_boundary_refinement": "threshold candidate cells are split dyadically three additional times by default",
            "boundary": "piecewise-linear threshold segments from locally refined boundary cells",
            "no_newton_continuation": True,
            "affine_matrix_audit": "pre-grid audited raw-gain affine 26x26 Jacobian with exact full-evaluator fallback per failed sample",
            "topology_limitation": "the final cell-center probe reduces but does not mathematically certify absence of a sub-cell boundary island",
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
            "caused_break_fraction_conditioned_on_success_baseline": float(
                failure_evaluation.caused_break_count / conditional_denominator
            ),
            "counterdirection_rescued_count": int(
                np.count_nonzero(
                    ~group_evaluator.success_stable_mask & failure_evaluation.stable_mask
                )
            ),
        },
        "success_gain_check": {
            "caused_break_count": success_evaluation.caused_break_count,
            "unstable_count": success_evaluation.unstable_count,
        },
        "affine_matrix_audit": {
            "phase": "completed_before_grid_evaluation",
            "failure_policy": "sample-local exact full-evaluator fallback; no sample is discarded",
            "basis_ratio": float(arguments.affine_basis_ratio),
            "tolerance": _AFFINE_AUDIT_TOLERANCE,
            "valid_samples": int(np.count_nonzero(group_evaluator.affine_valid)),
            "fallback_samples": group_evaluator.fallback_count,
            "maximum_relative_frobenius_error": float(np.max(audit_errors)),
            "median_relative_frobenius_error": float(np.median(audit_errors)),
            "piecewise_near_kink_samples": int(
                np.count_nonzero([item.piecewise_near_kink for item in sample_results])
            ),
        },
        "projections": [_projection_payload(item) for item in projections],
        "files": {
            "npz": str(output / "gain_contour.npz"),
            "png": str(output / "gain_contour.png"),
        },
    }
    write_json(output / "gain_contour.json", payload)
    return payload


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
        "--affine-basis-ratio",
        type=float,
        default=DEFAULT_AFFINE_BASIS_RATIO,
    )
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=DEFAULT_MAX_GRID_SIZE,
    )
    parser.add_argument(
        "--local-refinement-levels",
        type=int,
        default=DEFAULT_LOCAL_REFINEMENT_LEVELS,
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
    if arguments.affine_basis_ratio <= 1.0:
        raise SystemExit("--affine-basis-ratio must exceed one")
    if not _is_dyadic_nested_size(arguments.max_grid_size):
        raise SystemExit("--max-grid-size must be 2^k + 1 and at least three")
    if arguments.local_refinement_levels < 0:
        raise SystemExit("--local-refinement-levels must be non-negative")
    if arguments.workers <= 0:
        raise SystemExit("--workers must be positive")
    payload = analyze(arguments)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
