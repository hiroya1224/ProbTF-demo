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

Global search uses nested dyadic grids 3x3 -> 5x5 -> 9x9 only until the
boundary is first detected.  From that point onward, only boundary cells are
split, until the local spacing reaches the configured final equivalent grid
size (33x33 by default).  If no boundary is found through 9x9, one exact
center probe is evaluated in every 9x9 cell.  Only cells made suspicious by
those probes are refined to 17x17-equivalent spacing, after which confirmed
boundary cells alone are refined to 33x33-equivalent spacing.  Every repeated
gain point is cached exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
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
    NUMERICAL_SAMPLE_EXCEPTIONS,
    _analyze_plant,
    decompose_thrust_delay,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


CONTOUR_SCHEMA = "grape-param-estim/first-order-lag-pid-group-survival/v5"
DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_GROUP = "roll_pitch"
DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 3.0
DEFAULT_MAX_GRID_SIZE = 9
DEFAULT_FINAL_BOUNDARY_GRID_SIZE = 33
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
PROJECTION_SPECS = (
    ("pi", 0, 1, 2, "P", "I", "D"),
    ("id", 1, 2, 0, "I", "D", "P"),
    ("dp", 2, 0, 1, "D", "P", "I"),
)
INTEGRAL_AXES_BY_GROUP = {
    "xy": (0, 1),
    "z": (2,),
    "roll_pitch": (3, 4),
    "yaw": (5,),
}

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
class GainPointEvaluation:
    log_ratio: np.ndarray
    log_spectral_radius: np.ndarray
    trim_vectors: np.ndarray
    trim_nfev: np.ndarray
    attempt_nfev: np.ndarray
    pole_valid_mask: np.ndarray
    stable_mask: np.ndarray
    piecewise_near_kink: np.ndarray
    warm_primary_success_mask: np.ndarray
    nearest_fallback_mask: np.ndarray
    generic_fallback_mask: np.ndarray
    cold_primary_mask: np.ndarray
    trim_wall_seconds: float
    analytic_jacobian_wall_seconds: float
    eigensystem_wall_seconds: float
    initial_step_infinity_norm: np.ndarray
    initial_unchanged: np.ndarray
    stage: str


@dataclass(frozen=True)
class GainPointRequest:
    """One deterministic gain evaluation prepared before a worker wave."""

    log_ratio: np.ndarray
    initial_trims: Optional[np.ndarray]
    nearest_trims: Optional[np.ndarray]
    stage: str


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
    input_cell_kind: str
    input_cell_count: int
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
class CenterProbeResult:
    coordinates: np.ndarray
    survival_fraction: np.ndarray
    suspicious_mask: np.ndarray


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
    center_probe: Optional[CenterProbeResult]
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
    initial_trim: Optional[np.ndarray],
    nearest_trim: Optional[np.ndarray],
) -> tuple[
    Optional[np.ndarray],
    bool,
    np.ndarray,
    int,
    np.ndarray,
    bool,
    bool,
    bool,
    bool,
    float,
    float,
    float,
    bool,
]:
    """Solve one exact sample, retrying continuation failures conservatively."""

    inputs = _inputs(vehicle_model, actuator_parameters, controller_configuration)
    delay = decompose_thrust_delay(0.0, float(controller_dt))
    attempt_nfev = np.full(3, -1, dtype=int)
    trim_seconds = 0.0
    jacobian_seconds = 0.0
    final_result: Optional[Mapping[str, Any]] = None
    warm_primary_success = False
    nearest_attempted = False
    generic_attempted = False
    cold_primary = False

    def attempt(initial: Optional[np.ndarray], slot: int) -> Optional[Mapping[str, Any]]:
        nonlocal trim_seconds, jacobian_seconds
        started = time.perf_counter()
        try:
            result = _analyze_plant(
                scale_free=plant,
                inputs=inputs,
                controller_dt=float(controller_dt),
                delay=delay,
                fd_check=False,
                compute_eigenvalues=False,
                initial_trim=initial,
            )
        except NUMERICAL_SAMPLE_EXCEPTIONS:
            trim_seconds += time.perf_counter() - started
            return None
        trim = result.get("trim")
        if trim is None:
            raise RuntimeError("hover-trim diagnostics are unavailable")
        attempt_nfev[slot] = int(trim.root_nfev)
        trim_seconds += float(result.get("trim_wall_seconds", 0.0))
        jacobian_seconds += float(
            result.get("analytic_jacobian_wall_seconds", 0.0)
        )
        return result

    selected_initial = None
    if initial_trim is not None:
        candidate = np.asarray(initial_trim, dtype=float)
        if candidate.shape == (10,) and np.all(np.isfinite(candidate)):
            selected_initial = candidate
    if selected_initial is not None:
        final_result = attempt(selected_initial, 0)
        warm_primary_success = bool(
            final_result is not None
            and final_result["trim"].equilibrium_valid
        )

    if not warm_primary_success and nearest_trim is not None:
        candidate = np.asarray(nearest_trim, dtype=float)
        if candidate.shape == (10,) and np.all(np.isfinite(candidate)):
            nearest_attempted = True
            final_result = attempt(candidate, 1)
            if final_result is not None and final_result["trim"].equilibrium_valid:
                selected_initial = candidate

    valid = bool(
        final_result is not None and final_result["trim"].equilibrium_valid
    )
    if not valid:
        if initial_trim is None and not nearest_attempted:
            cold_primary = True
        else:
            generic_attempted = True
        final_result = attempt(None, 0 if cold_primary else 2)

    if final_result is None:
        return (
            None,
            False,
            np.full(10, np.nan, dtype=float),
            -1,
            attempt_nfev,
            warm_primary_success,
            nearest_attempted,
            generic_attempted,
            cold_primary,
            float(trim_seconds),
            float(jacobian_seconds),
            float("nan"),
            False,
        )

    trim = final_result["trim"]
    near_kink = bool(trim.piecewise_linearization_near_kink) or bool(
        final_result.get("analytic_piecewise_near_kink", False)
    )
    if not bool(trim.equilibrium_valid):
        return (
            None,
            near_kink,
            np.full(10, np.nan, dtype=float),
            int(trim.root_nfev),
            attempt_nfev,
            warm_primary_success,
            nearest_attempted,
            generic_attempted,
            cold_primary,
            float(trim_seconds),
            float(jacobian_seconds),
            float(trim.root_initial_step_infinity_norm),
            bool(trim.root_initial_unchanged),
        )
    matrix = final_result.get("jacobian")
    if matrix is None:
        raise RuntimeError("valid hover trim has no full 26-state pole Jacobian")
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (26, 26) or np.any(~np.isfinite(selected)):
        raise RuntimeError("first-order closed-loop Jacobian must be finite 26x26")
    return (
        selected,
        near_kink,
        np.asarray(trim.trim_vector, dtype=float),
        int(trim.root_nfev),
        attempt_nfev,
        warm_primary_success,
        nearest_attempted,
        generic_attempted,
        cold_primary,
        float(trim_seconds),
        float(jacobian_seconds),
        float(trim.root_initial_step_infinity_norm),
        bool(trim.root_initial_unchanged),
    )


def _exact_matrix_chunk_task(
    task: tuple[Any, ...],
) -> tuple[Any, ...]:
    (
        indices,
        plants,
        vehicle_model,
        actuator_parameters,
        controller_dt,
        controller_configuration,
        initial_trims,
        nearest_trims,
    ) = task
    selected_indices = np.asarray(indices, dtype=int)
    matrices = np.empty((selected_indices.size, 26, 26), dtype=float)
    pole_valid = np.zeros(selected_indices.size, dtype=bool)
    near_kink = np.empty(selected_indices.size, dtype=bool)
    trim_vectors = np.full((selected_indices.size, 10), np.nan, dtype=float)
    trim_nfev = np.full(selected_indices.size, -1, dtype=int)
    attempt_nfev = np.full((selected_indices.size, 3), -1, dtype=int)
    warm_primary_success = np.zeros(selected_indices.size, dtype=bool)
    nearest_fallback = np.zeros(selected_indices.size, dtype=bool)
    generic_fallback = np.zeros(selected_indices.size, dtype=bool)
    cold_primary = np.zeros(selected_indices.size, dtype=bool)
    trim_seconds = np.zeros(selected_indices.size, dtype=float)
    jacobian_seconds = np.zeros(selected_indices.size, dtype=float)
    initial_step_infinity_norm = np.full(selected_indices.size, np.nan, dtype=float)
    initial_unchanged = np.zeros(selected_indices.size, dtype=bool)
    for local_index, plant in enumerate(plants):
        row_initial = None if initial_trims is None else initial_trims[local_index]
        row_nearest = None if nearest_trims is None else nearest_trims[local_index]
        (
            matrix,
            near,
            trim_vector,
            final_nfev,
            row_attempt_nfev,
            row_primary_success,
            row_nearest_fallback,
            row_generic_fallback,
            row_cold_primary,
            row_trim_seconds,
            row_jacobian_seconds,
            row_initial_step_infinity_norm,
            row_initial_unchanged,
        ) = _exact_matrix_with_configuration(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            controller_configuration=controller_configuration,
            initial_trim=row_initial,
            nearest_trim=row_nearest,
        )
        if matrix is None:
            matrices[local_index] = 0.0
        else:
            matrices[local_index] = matrix
            pole_valid[local_index] = True
        near_kink[local_index] = near
        trim_vectors[local_index] = trim_vector
        trim_nfev[local_index] = final_nfev
        attempt_nfev[local_index] = row_attempt_nfev
        warm_primary_success[local_index] = row_primary_success
        nearest_fallback[local_index] = row_nearest_fallback
        generic_fallback[local_index] = row_generic_fallback
        cold_primary[local_index] = row_cold_primary
        trim_seconds[local_index] = row_trim_seconds
        jacobian_seconds[local_index] = row_jacobian_seconds
        initial_step_infinity_norm[local_index] = row_initial_step_infinity_norm
        initial_unchanged[local_index] = row_initial_unchanged
    return (
        selected_indices,
        matrices,
        pole_valid,
        near_kink,
        trim_vectors,
        trim_nfev,
        attempt_nfev,
        warm_primary_success,
        nearest_fallback,
        generic_fallback,
        cold_primary,
        trim_seconds,
        jacobian_seconds,
        initial_step_infinity_norm,
        initial_unchanged,
    )


def _keyed_exact_matrix_chunk_task(
    task: tuple[tuple[float, float, float], tuple[Any, ...]],
) -> tuple[tuple[float, float, float], tuple[Any, ...]]:
    """Keep gain identity explicit when chunks from many gains interleave."""

    gain_key, chunk_task = task
    return gain_key, _exact_matrix_chunk_task(chunk_task)


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
        self._evaluation_cache: dict[
            tuple[float, float, float], GainPointEvaluation
        ] = {}
        self._break_cache: dict[tuple[float, float, float], BreakEvaluation] = {}
        self._scheduling_rows: list[Mapping[str, float]] = []
        self.success_log_ratio = np.log(self.success_gain / self.base_gain)
        success = self.evaluate(
            self.success_log_ratio, stage="all_success_baseline"
        )
        self.success_log_radius = success.log_spectral_radius.copy()
        self.success_pole_valid_mask = success.pole_valid_mask.copy()
        self.success_stable_mask = success.stable_mask.copy()
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
        return len(self._evaluation_cache)

    @property
    def gain_points_with_any_near_kink(self) -> int:
        return int(
            sum(
                bool(np.any(value.piecewise_near_kink))
                for value in self._evaluation_cache.values()
            )
        )

    @property
    def gain_points_with_any_unresolved_trim(self) -> int:
        return int(
            sum(
                bool(np.any(~value.pole_valid_mask))
                for value in self._evaluation_cache.values()
            )
        )

    @property
    def unresolved_trim_sample_evaluation_count(self) -> int:
        return int(
            sum(
                np.count_nonzero(~value.pole_valid_mask)
                for value in self._evaluation_cache.values()
            )
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

    def integral_gain_vector(self, log_ratio: Sequence[float]) -> np.ndarray:
        """Return the six-axis I gains used at one selected-group gain point."""

        selected_integral_gain = float(self.gain_from_log_ratio(log_ratio)[1])
        result = np.empty(6, dtype=float)
        for group, axes in INTEGRAL_AXES_BY_GROUP.items():
            integral_gain = (
                selected_integral_gain
                if group == self.group
                else float(self.hybrid_failure_gains[group].i)
            )
            result[np.asarray(axes, dtype=int)] = integral_gain
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("integral gains must be finite and strictly positive")
        return result

    def _matrix_tasks(
        self,
        log_ratio: Sequence[float],
        initial_trims: Optional[np.ndarray],
        nearest_trims: Optional[np.ndarray],
    ) -> list[tuple[Any, ...]]:
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
        return [
            (
                tuple(int(index) for index in chunk),
                tuple(self.plants[int(index)] for index in chunk),
                self.vehicle_model,
                self.actuator_parameters,
                self.controller_dt,
                controller_configuration,
                (
                    None
                    if initial_trims is None
                    else np.asarray(initial_trims, dtype=float)[chunk]
                ),
                (
                    None
                    if nearest_trims is None
                    else np.asarray(nearest_trims, dtype=float)[chunk]
                ),
            )
            for chunk in index_chunks
        ]

    def _matrix_stack(
        self,
        log_ratio: Sequence[float],
        initial_trims: Optional[np.ndarray],
        nearest_trims: Optional[np.ndarray],
    ) -> tuple[Any, ...]:
        tasks = self._matrix_tasks(log_ratio, initial_trims, nearest_trims)
        if self._executor is None:
            rows = [_exact_matrix_chunk_task(task) for task in tasks]
        else:
            rows = list(self._executor.map(_exact_matrix_chunk_task, tasks))
        return self._assemble_matrix_rows(rows)

    def _assemble_matrix_rows(
        self, rows: Sequence[tuple[Any, ...]]
    ) -> tuple[Any, ...]:
        sample_count = len(self.plants)
        matrices = np.empty((sample_count, 26, 26), dtype=float)
        pole_valid = np.empty(sample_count, dtype=bool)
        near_kink = np.empty(sample_count, dtype=bool)
        trim_vectors = np.full((sample_count, 10), np.nan, dtype=float)
        trim_nfev = np.full(sample_count, -1, dtype=int)
        attempt_nfev = np.full((sample_count, 3), -1, dtype=int)
        warm_primary_success = np.zeros(sample_count, dtype=bool)
        nearest_fallback = np.zeros(sample_count, dtype=bool)
        generic_fallback = np.zeros(sample_count, dtype=bool)
        cold_primary = np.zeros(sample_count, dtype=bool)
        trim_seconds = np.zeros(sample_count, dtype=float)
        jacobian_seconds = np.zeros(sample_count, dtype=float)
        initial_step_infinity_norm = np.full(sample_count, np.nan, dtype=float)
        initial_unchanged = np.zeros(sample_count, dtype=bool)
        filled = np.zeros(sample_count, dtype=bool)
        for row in rows:
            (
                indices,
                chunk_matrices,
                chunk_pole_valid,
                chunk_near_kink,
                chunk_trim_vectors,
                chunk_trim_nfev,
                chunk_attempt_nfev,
                chunk_primary_success,
                chunk_nearest_fallback,
                chunk_generic_fallback,
                chunk_cold_primary,
                chunk_trim_seconds,
                chunk_jacobian_seconds,
                chunk_initial_step_infinity_norm,
                chunk_initial_unchanged,
            ) = row
            matrices[indices] = chunk_matrices
            pole_valid[indices] = chunk_pole_valid
            near_kink[indices] = chunk_near_kink
            trim_vectors[indices] = chunk_trim_vectors
            trim_nfev[indices] = chunk_trim_nfev
            attempt_nfev[indices] = chunk_attempt_nfev
            warm_primary_success[indices] = chunk_primary_success
            nearest_fallback[indices] = chunk_nearest_fallback
            generic_fallback[indices] = chunk_generic_fallback
            cold_primary[indices] = chunk_cold_primary
            trim_seconds[indices] = chunk_trim_seconds
            jacobian_seconds[indices] = chunk_jacobian_seconds
            initial_step_infinity_norm[indices] = chunk_initial_step_infinity_norm
            initial_unchanged[indices] = chunk_initial_unchanged
            filled[indices] = True
        if not np.all(filled):
            raise RuntimeError("exact plant sample ordering changed during evaluation")
        return (
            matrices,
            pole_valid,
            near_kink,
            trim_vectors,
            trim_nfev,
            attempt_nfev,
            warm_primary_success,
            nearest_fallback,
            generic_fallback,
            cold_primary,
            trim_seconds,
            jacobian_seconds,
            initial_step_infinity_norm,
            initial_unchanged,
        )

    @staticmethod
    def _validate_trim_matrix(
        value: Optional[np.ndarray], sample_count: int, name: str
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        selected = np.asarray(value, dtype=float)
        if selected.shape != (sample_count, 10):
            raise ValueError(f"{name} must have shape (sample_count, 10)")
        return selected.copy()

    def evaluate(
        self,
        log_ratio: Sequence[float],
        *,
        initial_trims: Optional[np.ndarray] = None,
        nearest_trims: Optional[np.ndarray] = None,
        stage: str = "cold",
    ) -> GainPointEvaluation:
        key = self._key(log_ratio)
        cached = self._evaluation_cache.get(key)
        if cached is not None:
            return cached
        initial = self._validate_trim_matrix(
            initial_trims, self.sample_count, "initial_trims"
        )
        nearest = self._validate_trim_matrix(
            nearest_trims, self.sample_count, "nearest_trims"
        )
        wave_started = time.perf_counter()
        stack = self._matrix_stack(key, initial, nearest)
        queue_seconds = time.perf_counter() - wave_started
        result = self._gain_point_from_matrix_stack(key, stack, stage)
        total_seconds = time.perf_counter() - wave_started
        self._record_scheduling_wave(
            gain_count=1,
            task_count=min(self.sample_count, max(1, 2 * self.workers)),
            queue_seconds=queue_seconds,
            total_seconds=total_seconds,
            worker_seconds=float(np.sum(stack[10]) + np.sum(stack[11])),
            task_preparation_seconds=0.0,
            submission_seconds=0.0,
        )
        self._evaluation_cache[key] = result
        return result

    def _record_scheduling_wave(
        self,
        *,
        gain_count: int,
        task_count: int,
        queue_seconds: float,
        total_seconds: float,
        worker_seconds: float,
        task_preparation_seconds: float,
        submission_seconds: float,
    ) -> None:
        self._scheduling_rows.append(
            {
                "gain_count": float(gain_count),
                "task_count": float(task_count),
                "queue_seconds": float(queue_seconds),
                "total_seconds": float(total_seconds),
                "worker_seconds": float(worker_seconds),
                "task_preparation_seconds": float(task_preparation_seconds),
                "submission_seconds": float(submission_seconds),
            }
        )

    def _gain_point_from_matrix_stack(
        self,
        key: tuple[float, float, float],
        stack: tuple[Any, ...],
        stage: str,
    ) -> GainPointEvaluation:
        (
            matrices,
            pole_valid,
            near_kink,
            trim_vectors,
            trim_nfev,
            attempt_nfev,
            warm_primary_success,
            nearest_fallback,
            generic_fallback,
            cold_primary,
            trim_seconds,
            jacobian_seconds,
            initial_step_infinity_norm,
            initial_unchanged,
        ) = stack
        # NumPy accepts a stack of square matrices.  All 512 eigensystems are
        # therefore dispatched in one call rather than a Python eigvals loop.
        radii = np.full(self.sample_count, np.inf, dtype=float)
        eigensystem_started = time.perf_counter()
        if np.any(pole_valid):
            eigenvalues = np.linalg.eigvals(matrices[pole_valid])
            radii[pole_valid] = np.max(np.abs(eigenvalues), axis=1)
        eigensystem_seconds = time.perf_counter() - eigensystem_started
        if np.any(~np.isfinite(radii[pole_valid])) or np.any(
            radii[pole_valid] <= 0.0
        ):
            raise FloatingPointError("closed-loop spectral radius became invalid")
        log_radii = np.log(radii)
        return GainPointEvaluation(
            log_ratio=np.asarray(key, dtype=float),
            log_spectral_radius=log_radii,
            trim_vectors=trim_vectors,
            trim_nfev=trim_nfev,
            attempt_nfev=attempt_nfev,
            pole_valid_mask=pole_valid,
            stable_mask=log_radii < 0.0,
            piecewise_near_kink=near_kink,
            warm_primary_success_mask=warm_primary_success,
            nearest_fallback_mask=nearest_fallback,
            generic_fallback_mask=generic_fallback,
            cold_primary_mask=cold_primary,
            trim_wall_seconds=float(np.sum(trim_seconds)),
            analytic_jacobian_wall_seconds=float(np.sum(jacobian_seconds)),
            eigensystem_wall_seconds=float(eigensystem_seconds),
            initial_step_infinity_norm=initial_step_infinity_norm,
            initial_unchanged=initial_unchanged,
            stage=str(stage),
        )

    def evaluate_wave(
        self, requests: Sequence[GainPointRequest]
    ) -> Mapping[tuple[float, float, float], GainPointEvaluation]:
        """Evaluate one frozen-predictor refinement wave in a shared queue."""

        prepared: dict[
            tuple[float, float, float],
            tuple[Optional[np.ndarray], Optional[np.ndarray], str],
        ] = {}
        for request in requests:
            if not isinstance(request, GainPointRequest):
                raise TypeError("wave requests must be GainPointRequest values")
            key = self._key(request.log_ratio)
            if key in self._evaluation_cache:
                continue
            if key in prepared:
                raise ValueError("duplicate uncached gain in one refinement wave")
            prepared[key] = (
                self._validate_trim_matrix(
                    request.initial_trims, self.sample_count, "initial_trims"
                ),
                self._validate_trim_matrix(
                    request.nearest_trims, self.sample_count, "nearest_trims"
                ),
                str(request.stage),
            )

        wave_started = time.perf_counter()
        keyed_tasks = []
        expected_rows = {}
        for key in sorted(prepared):
            initial, nearest, _stage = prepared[key]
            tasks = self._matrix_tasks(key, initial, nearest)
            expected_rows[key] = len(tasks)
            keyed_tasks.extend((key, task) for task in tasks)
        task_count = len(keyed_tasks)
        task_preparation_seconds = time.perf_counter() - wave_started
        rows_by_gain: dict[
            tuple[float, float, float], list[tuple[Any, ...]]
        ] = {key: [] for key in prepared}
        wave_results: dict[
            tuple[float, float, float], GainPointEvaluation
        ] = {}
        worker_seconds = 0.0

        def accept(completed: tuple[tuple[float, float, float], tuple[Any, ...]]) -> None:
            nonlocal worker_seconds
            key, row = completed
            rows_by_gain[key].append(row)
            worker_seconds += float(np.sum(row[11])) + float(np.sum(row[12]))
            if len(rows_by_gain[key]) == expected_rows[key]:
                _initial, _nearest, stage = prepared[key]
                stack = self._assemble_matrix_rows(rows_by_gain.pop(key))
                wave_results[key] = self._gain_point_from_matrix_stack(
                    key, stack, stage
                )

        if self._executor is None:
            submission_seconds = 0.0
            for task in keyed_tasks:
                accept(_keyed_exact_matrix_chunk_task(task))
        else:
            submission_started = time.perf_counter()
            futures = [
                self._executor.submit(_keyed_exact_matrix_chunk_task, task)
                for task in keyed_tasks
            ]
            submission_seconds = time.perf_counter() - submission_started
            del keyed_tasks
            completed_futures = as_completed(futures)
            del futures
            for future in completed_futures:
                accept(future.result())
        queue_seconds = time.perf_counter() - wave_started
        if rows_by_gain:
            raise RuntimeError("refinement wave finished with incomplete gains")
        for key in sorted(wave_results):
            self._evaluation_cache[key] = wave_results[key]
        total_seconds = time.perf_counter() - wave_started
        if prepared:
            self._record_scheduling_wave(
                gain_count=len(prepared),
                task_count=task_count,
                queue_seconds=queue_seconds,
                total_seconds=total_seconds,
                worker_seconds=worker_seconds,
                task_preparation_seconds=task_preparation_seconds,
                submission_seconds=submission_seconds,
            )
        return {
            self._key(request.log_ratio): self._evaluation_cache[
                self._key(request.log_ratio)
            ]
            for request in requests
        }

    def log_spectral_radius(self, log_ratio: Sequence[float]) -> np.ndarray:
        return self.evaluate(log_ratio).log_spectral_radius.copy()

    def pole_valid_mask(self, log_ratio: Sequence[float]) -> np.ndarray:
        key = self._key(log_ratio)
        return self.evaluate(key).pole_valid_mask.copy()

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
        gain_point = self.evaluate(key)
        stable = gain_point.stable_mask
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

    def diagnostic_payload(self) -> Mapping[str, Any]:
        rows = tuple(self._evaluation_cache.values())

        def summarize(selected: Sequence[GainPointEvaluation]) -> Mapping[str, Any]:
            attempts = np.concatenate(
                [row.attempt_nfev[row.attempt_nfev >= 0] for row in selected]
            ) if selected else np.empty(0, dtype=int)
            nfev_one_displacements = np.concatenate(
                [
                    row.initial_step_infinity_norm[
                        (row.trim_nfev == 1)
                        & np.isfinite(row.initial_step_infinity_norm)
                    ]
                    for row in selected
                ]
            ) if selected else np.empty(0, dtype=float)
            nfev_one_unchanged_count = int(
                sum(
                    np.count_nonzero(
                        (row.trim_nfev == 1) & row.initial_unchanged
                    )
                    for row in selected
                )
            )
            return {
                "unique_gain_point_count": len(selected),
                "trim_solve_count": int(attempts.size),
                "warm_start_primary_success_count": int(
                    sum(np.count_nonzero(row.warm_primary_success_mask) for row in selected)
                ),
                "nearest_neighbor_fallback_count": int(
                    sum(np.count_nonzero(row.nearest_fallback_mask) for row in selected)
                ),
                "generic_cold_fallback_count": int(
                    sum(np.count_nonzero(row.generic_fallback_mask) for row in selected)
                ),
                "generic_cold_primary_count": int(
                    sum(np.count_nonzero(row.cold_primary_mask) for row in selected)
                ),
                "trim_nfev_mean": (None if attempts.size == 0 else float(np.mean(attempts))),
                "trim_nfev_median": (None if attempts.size == 0 else float(np.median(attempts))),
                "trim_nfev_max": (None if attempts.size == 0 else int(np.max(attempts))),
                "trim_nfev_one_count": int(nfev_one_displacements.size),
                "trim_nfev_one_initial_exactly_unchanged_count": nfev_one_unchanged_count,
                "trim_nfev_one_initial_exactly_unchanged_fraction": (
                    None
                    if nfev_one_displacements.size == 0
                    else float(
                        nfev_one_unchanged_count / nfev_one_displacements.size
                    )
                ),
                "trim_nfev_one_initial_step_infinity_norm_median": (
                    None
                    if nfev_one_displacements.size == 0
                    else float(np.median(nfev_one_displacements))
                ),
                "trim_nfev_one_initial_step_infinity_norm_max": (
                    None
                    if nfev_one_displacements.size == 0
                    else float(np.max(nfev_one_displacements))
                ),
                "trim_wall_seconds_sum_over_workers": float(
                    sum(row.trim_wall_seconds for row in selected)
                ),
                "analytic_jacobian_wall_seconds_sum_over_workers": float(
                    sum(row.analytic_jacobian_wall_seconds for row in selected)
                ),
                "batched_eigensystem_wall_seconds": float(
                    sum(row.eigensystem_wall_seconds for row in selected)
                ),
            }

        stages = sorted({row.stage for row in rows})
        result = dict(summarize(rows))
        result["by_stage"] = {
            stage: summarize([row for row in rows if row.stage == stage])
            for stage in stages
        }
        scheduling = tuple(self._scheduling_rows)
        queue_seconds = float(
            sum(row["queue_seconds"] for row in scheduling)
        )
        ideal_worker_seconds = float(
            sum(row["worker_seconds"] / self.workers for row in scheduling)
        )
        result["refinement_wave_scheduling"] = {
            "wave_count": len(scheduling),
            "multi_gain_wave_count": int(
                sum(row["gain_count"] > 1.0 for row in scheduling)
            ),
            "scheduled_gain_point_count": int(
                sum(row["gain_count"] for row in scheduling)
            ),
            "scheduled_plant_chunk_task_count": int(
                sum(row["task_count"] for row in scheduling)
            ),
            "maximum_gain_points_in_one_wave": int(
                max((row["gain_count"] for row in scheduling), default=0.0)
            ),
            "queue_and_collection_wall_seconds": queue_seconds,
            "total_wave_wall_seconds": float(
                sum(row["total_seconds"] for row in scheduling)
            ),
            "timed_worker_compute_seconds": float(
                sum(row["worker_seconds"] for row in scheduling)
            ),
            "task_preparation_wall_seconds": float(
                sum(row["task_preparation_seconds"] for row in scheduling)
            ),
            "executor_submission_wall_seconds": float(
                sum(row["submission_seconds"] for row in scheduling)
            ),
            "ideal_timed_worker_wall_seconds": ideal_worker_seconds,
            "queue_excess_over_ideal_timed_worker_seconds": float(
                max(0.0, queue_seconds - ideal_worker_seconds)
            ),
            "timed_worker_utilization_fraction": (
                None
                if queue_seconds <= 0.0
                else float(min(1.0, ideal_worker_seconds / queue_seconds))
            ),
            "per_gain_barriers_removed_within_waves": int(
                sum(max(0.0, row["gain_count"] - 1.0) for row in scheduling)
            ),
        }
        return result

    def cold_start_audit(
        self, log_ratios: Sequence[Sequence[float]]
    ) -> Mapping[str, Any]:
        rows = []
        for coordinate in log_ratios:
            key = self._key(coordinate)
            warm = self._evaluation_cache.get(key)
            if warm is None:
                raise ValueError("cold-start audit point has not been warm-solved")
            cold_stack = self._matrix_stack(key, None, None)
            matrices = cold_stack[0]
            cold_valid = np.asarray(cold_stack[1], dtype=bool)
            cold_trims = np.asarray(cold_stack[3], dtype=float)
            cold_radii = np.full(self.sample_count, np.inf, dtype=float)
            if np.any(cold_valid):
                eigenvalues = np.linalg.eigvals(matrices[cold_valid])
                cold_radii[cold_valid] = np.max(np.abs(eigenvalues), axis=1)
            cold_stable = cold_radii < 1.0
            common = cold_valid & warm.pole_valid_mask
            trim_delta = np.linalg.norm(
                cold_trims[common] - warm.trim_vectors[common],
                ord=np.inf,
                axis=1,
            ) if np.any(common) else np.empty(0, dtype=float)
            warm_radii = np.exp(warm.log_spectral_radius[common])
            radius_delta = np.abs(cold_radii[common] - warm_radii)
            classification_disagreement = int(
                np.count_nonzero(cold_stable != warm.stable_mask)
            )
            possible_branch = bool(
                classification_disagreement
                or (trim_delta.size and float(np.max(trim_delta)) > 1.0e-7)
            )
            rows.append(
                {
                    "log_gain_ratio": list(key),
                    "common_valid_sample_count": int(np.count_nonzero(common)),
                    "pole_valid_mask_disagreement_count": int(
                        np.count_nonzero(cold_valid != warm.pole_valid_mask)
                    ),
                    "stability_classification_disagreement_count": classification_disagreement,
                    "maximum_trim_vector_infinity_norm_difference": (
                        None if trim_delta.size == 0 else float(np.max(trim_delta))
                    ),
                    "maximum_spectral_radius_absolute_difference": (
                        None if radius_delta.size == 0 else float(np.max(radius_delta))
                    ),
                    "possible_multiple_trim_branch": possible_branch,
                }
            )
        return {
            "enabled": True,
            "equilibrium_acceptance": "both warm and cold paths use the unchanged strict equilibrium-validity check",
            "possible_multiple_trim_branch_detected": bool(
                any(row["possible_multiple_trim_branch"] for row in rows)
            ),
            "points": rows,
        }


class SliceGridEvaluator:
    """Cached 2-D group slice evaluated in conditional survival fraction."""

    def __init__(
        self,
        group_evaluator: ExactGroupGainEvaluator,
        *,
        first_axis: int,
        second_axis: int,
        hidden_axis: int,
        projection_name: str = "projection",
    ) -> None:
        self.group_evaluator = group_evaluator
        self.first_axis = int(first_axis)
        self.second_axis = int(second_axis)
        self.hidden_axis = int(hidden_axis)
        self.projection_name = str(projection_name)
        self._cache: dict[tuple[float, float], float] = {}
        self._trim_cache: dict[tuple[float, float], np.ndarray] = {}

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

    def integral_gain_vector(self, first: float, second: float) -> np.ndarray:
        key = self._visible_key(first, second)
        return self.group_evaluator.integral_gain_vector(
            self._full_coordinate(*key)
        )

    def evaluate(
        self,
        first: float,
        second: float,
        *,
        initial_trims: Optional[np.ndarray] = None,
        nearest_trims: Optional[np.ndarray] = None,
        stage: str = "cold",
    ) -> float:
        key = self._visible_key(first, second)
        cached = self._cache.get(key)
        if cached is not None:
            return float(cached)
        coordinate = self._full_coordinate(*key)
        exact_evaluate = getattr(self.group_evaluator, "evaluate", None)
        if callable(exact_evaluate):
            point = exact_evaluate(
                coordinate,
                initial_trims=initial_trims,
                nearest_trims=nearest_trims,
                stage=stage,
            )
            self._trim_cache[key] = np.asarray(point.trim_vectors, dtype=float).copy()
        value = self.group_evaluator.survival_fraction(coordinate)
        self._cache[key] = float(value)
        return float(value)

    def evaluate_wave(
        self,
        coordinates: Sequence[tuple[float, float]],
        *,
        initial_trims: Mapping[tuple[float, float], np.ndarray],
        nearest_trims: Mapping[
            tuple[float, float], Optional[np.ndarray]
        ],
        stage: str,
    ) -> None:
        """Evaluate all new visible coordinates with frozen predictors."""

        keys = sorted(
            {
                self._visible_key(*coordinate)
                for coordinate in coordinates
                if self._visible_key(*coordinate) not in self._cache
            }
        )
        exact_wave = getattr(self.group_evaluator, "evaluate_wave", None)
        if callable(exact_wave) and keys:
            requests = [
                GainPointRequest(
                    log_ratio=self._full_coordinate(*key),
                    initial_trims=initial_trims.get(key),
                    nearest_trims=nearest_trims.get(key),
                    stage=str(stage),
                )
                for key in keys
            ]
            evaluated = exact_wave(requests)
            for key, request in zip(keys, requests):
                point = evaluated[
                    self.group_evaluator._key(request.log_ratio)
                ]
                self._trim_cache[key] = np.asarray(
                    point.trim_vectors, dtype=float
                ).copy()
                value = self.group_evaluator.survival_fraction(
                    request.log_ratio
                )
                self._cache[key] = float(value)
            return
        for key in keys:
            self.evaluate(
                *key,
                initial_trims=initial_trims.get(key),
                nearest_trims=nearest_trims.get(key),
                stage=stage,
            )

    def trim_matrix(self, first: float, second: float) -> Optional[np.ndarray]:
        selected = self._trim_cache.get(self._visible_key(first, second))
        return None if selected is None else selected.copy()

    def nearest_trim_matrix(self, first: float, second: float) -> Optional[np.ndarray]:
        if not self._trim_cache:
            return None
        target = np.asarray(self._visible_key(first, second), dtype=float)
        ordered = sorted(
            self._trim_cache.items(),
            key=lambda item: (
                float(np.linalg.norm(np.asarray(item[0], dtype=float) - target)),
                item[0],
            ),
        )
        sample_count = next(iter(self._trim_cache.values())).shape[0]
        result = np.full((sample_count, 10), np.nan, dtype=float)
        unresolved = np.ones(sample_count, dtype=bool)
        for _coordinate, trims in ordered:
            valid = unresolved & np.all(np.isfinite(trims), axis=1)
            result[valid] = trims[valid]
            unresolved[valid] = False
            if not np.any(unresolved):
                break
        return result

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


def _bilinear_trim_predictor(
    *,
    x: float,
    y: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    corner_trims: Sequence[np.ndarray],
) -> np.ndarray:
    if not (x1 > x0 and y1 > y0) or len(corner_trims) != 4:
        raise ValueError("bilinear predictor requires an ordered cell and four corners")
    selected = [np.asarray(value, dtype=float) for value in corner_trims]
    if any(value.ndim != 2 or value.shape[1] != 10 for value in selected):
        raise ValueError("trim predictors must have shape (sample_count, 10)")
    if len({value.shape for value in selected}) != 1:
        raise ValueError("trim predictor corner shapes differ")
    a = (float(x) - float(x0)) / (float(x1) - float(x0))
    b = (float(y) - float(y0)) / (float(y1) - float(y0))
    weights = np.asarray(
        ((1.0 - a) * (1.0 - b), a * (1.0 - b), a * b, (1.0 - a) * b),
        dtype=float,
    )
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=64.0 * _EPS):
        raise RuntimeError("bilinear predictor weights do not sum to one")
    return sum(weight * value for weight, value in zip(weights, selected))


def _gain_scaled_bilinear_trim_predictor(
    *,
    x: float,
    y: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    corner_trims: Sequence[np.ndarray],
    corner_integral_gains: Sequence[np.ndarray],
    target_integral_gains: np.ndarray,
) -> np.ndarray:
    """Interpolate trim in ``K_I * integral_error`` coordinates.

    Integral components whose corner and target gains are exactly constant
    retain the raw bilinear result bit for bit.  This avoids introducing a
    multiply/divide round trip in constant-I slices such as DP.
    """

    result = _bilinear_trim_predictor(
        x=x,
        y=y,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        corner_trims=corner_trims,
    )
    selected_trims = [np.asarray(value, dtype=float) for value in corner_trims]
    selected_gains = [
        np.asarray(value, dtype=float) for value in corner_integral_gains
    ]
    target_gains = np.asarray(target_integral_gains, dtype=float)
    if len(selected_gains) != 4 or any(value.shape != (6,) for value in selected_gains):
        raise ValueError("corner integral gains must contain four length-six vectors")
    if target_gains.shape != (6,):
        raise ValueError("target integral gains must have length six")
    gain_stack = np.asarray(selected_gains, dtype=float)
    if (
        np.any(~np.isfinite(gain_stack))
        or np.any(gain_stack <= 0.0)
        or np.any(~np.isfinite(target_gains))
        or np.any(target_gains <= 0.0)
    ):
        raise ValueError("integral gains must be finite and strictly positive")

    a = (float(x) - float(x0)) / (float(x1) - float(x0))
    b = (float(y) - float(y0)) / (float(y1) - float(y0))
    weights = np.asarray(
        ((1.0 - a) * (1.0 - b), a * (1.0 - b), a * b, (1.0 - a) * b),
        dtype=float,
    )
    for component in range(6):
        if np.all(gain_stack[:, component] == target_gains[component]):
            continue
        scaled = [
            gains[component] * trim[:, component]
            for gains, trim in zip(selected_gains, selected_trims)
        ]
        result[:, component] = sum(
            weight * value for weight, value in zip(weights, scaled)
        ) / target_gains[component]
    return result


def _average_trim_predictors(predictors: Sequence[np.ndarray]) -> np.ndarray:
    if not predictors:
        raise ValueError("at least one trim predictor is required")
    stack = np.asarray(predictors, dtype=float)
    if stack.ndim != 3 or stack.shape[2] != 10:
        raise ValueError("trim predictor stack must be parent-by-sample-by-10")
    valid = np.all(np.isfinite(stack), axis=2)
    result = np.full(stack.shape[1:], np.nan, dtype=float)
    for sample in range(stack.shape[1]):
        available = stack[valid[:, sample], sample]
        if available.size:
            result[sample] = np.mean(available, axis=0)
    return result


def _cell_trim_predictor(
    evaluator: SliceGridEvaluator,
    cell: AdaptiveCell,
    x: float,
    y: float,
) -> Optional[np.ndarray]:
    corners = (
        evaluator.trim_matrix(cell.x0, cell.y0),
        evaluator.trim_matrix(cell.x1, cell.y0),
        evaluator.trim_matrix(cell.x1, cell.y1),
        evaluator.trim_matrix(cell.x0, cell.y1),
    )
    if any(value is None for value in corners):
        return None
    corner_coordinates = (
        (cell.x0, cell.y0),
        (cell.x1, cell.y0),
        (cell.x1, cell.y1),
        (cell.x0, cell.y1),
    )
    return _gain_scaled_bilinear_trim_predictor(
        x=x,
        y=y,
        x0=cell.x0,
        x1=cell.x1,
        y0=cell.y0,
        y1=cell.y1,
        corner_trims=[value for value in corners if value is not None],
        corner_integral_gains=[
            evaluator.integral_gain_vector(*coordinate)
            for coordinate in corner_coordinates
        ],
        target_integral_gains=evaluator.integral_gain_vector(x, y),
    )


def _evaluate_coordinates_from_parent_cells(
    evaluator: SliceGridEvaluator,
    coordinates: Sequence[tuple[float, float]],
    parent_cells: Sequence[AdaptiveCell],
    *,
    stage: str,
) -> None:
    contributions: dict[tuple[float, float], list[np.ndarray]] = {}
    tolerance = 128.0 * _EPS
    new_coordinates = [
        evaluator._visible_key(*coordinate)
        for coordinate in coordinates
        if evaluator._visible_key(*coordinate) not in evaluator._cache
    ]
    for coordinate in sorted(set(new_coordinates)):
        x, y = coordinate
        for cell in parent_cells:
            if (
                cell.x0 - tolerance <= x <= cell.x1 + tolerance
                and cell.y0 - tolerance <= y <= cell.y1 + tolerance
            ):
                predictor = _cell_trim_predictor(evaluator, cell, x, y)
                if predictor is not None:
                    contributions.setdefault(coordinate, []).append(predictor)
    predictors = {
        coordinate: _average_trim_predictors(values)
        for coordinate, values in contributions.items()
    }
    # Snapshot nearest-known fallbacks before solving this stage so results do
    # not depend on the order in which shared new vertices are visited.
    nearest = {
        coordinate: evaluator.nearest_trim_matrix(*coordinate)
        for coordinate in sorted(set(new_coordinates))
    }
    evaluator.evaluate_wave(
        sorted(set(new_coordinates)),
        initial_trims=predictors,
        nearest_trims=nearest,
        stage=stage,
    )


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


def _nested_regular_grid(
    evaluator: SliceGridEvaluator,
    lower: float,
    upper: float,
    size: int,
    *,
    previous_axis: Optional[np.ndarray],
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(float(lower), float(upper), int(size))
    coordinates = [
        (float(first), float(second)) for first in axis for second in axis
    ]
    if previous_axis is None:
        center = (float(axis[axis.size // 2]), float(axis[axis.size // 2]))
        anchor = evaluator.trim_matrix(*center)
        nearest = {
            evaluator._visible_key(*coordinate): evaluator.nearest_trim_matrix(*coordinate)
            for coordinate in coordinates
            if evaluator._visible_key(*coordinate) not in evaluator._cache
        }
        initial = {
            evaluator._visible_key(*coordinate): anchor
            for coordinate in coordinates
            if evaluator._visible_key(*coordinate) not in evaluator._cache
            and anchor is not None
        }
        evaluator.evaluate_wave(
            coordinates,
            initial_trims=initial,
            nearest_trims=nearest,
            stage=stage,
        )
    else:
        coarse = np.asarray(previous_axis, dtype=float)
        coarse_field = np.asarray(
            [
                [evaluator.evaluate(float(first), float(second)) for second in coarse]
                for first in coarse
            ],
            dtype=float,
        )
        _evaluate_coordinates_from_parent_cells(
            evaluator,
            coordinates,
            _grid_cells(coarse, coarse_field),
            stage=stage,
        )
    field = np.asarray(
        [
            [evaluator.evaluate(float(first), float(second)) for second in axis]
            for first in axis
        ],
        dtype=float,
    )
    return axis, field


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


def _refine_cells_one_level(
    evaluator: SliceGridEvaluator,
    cells: Sequence[AdaptiveCell],
    threshold: float,
    *,
    level: int,
    next_equivalent_grid_size: int,
    stage_prefix: str,
    input_cell_kind: str,
) -> tuple[
    tuple[AdaptiveCell, ...],
    tuple[AdaptiveCell, ...],
    LocalRefinementLevel,
]:
    """Split selected cells once and retain only confirmed boundary children."""

    before = evaluator.cached_point_count
    coordinates = []
    for cell in cells:
        xm = 0.5 * (cell.x0 + cell.x1)
        ym = 0.5 * (cell.y0 + cell.y1)
        coordinates.extend(
            (float(x), float(y))
            for x in (cell.x0, xm, cell.x1)
            for y in (cell.y0, ym, cell.y1)
        )
    _evaluate_coordinates_from_parent_cells(
        evaluator,
        coordinates,
        cells,
        stage=(
            f"{stage_prefix}:local_{level}_eq_{next_equivalent_grid_size}"
        ),
    )
    boundary: list[AdaptiveCell] = []
    leaves: list[AdaptiveCell] = []
    for cell in cells:
        for child in _split_cell(evaluator, cell):
            if _cell_brackets(child.values, threshold):
                boundary.append(child)
            else:
                leaves.append(child)
    after = evaluator.cached_point_count
    diagnostics = LocalRefinementLevel(
        level=int(level),
        equivalent_grid_size=int(next_equivalent_grid_size),
        input_cell_kind=str(input_cell_kind),
        input_cell_count=len(cells),
        output_boundary_cell_count=len(boundary),
        new_point_count=int(after - before),
        total_cached_point_count=int(after),
    )
    return tuple(leaves), tuple(boundary), diagnostics


def _refine_boundary_to_target(
    evaluator: SliceGridEvaluator,
    base_cells: Sequence[AdaptiveCell],
    threshold: float,
    starting_grid_size: int,
    target_grid_size: int,
    stage_prefix: str,
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
        next_effective = _next_nested_size(effective)
        level += 1
        new_leaves, next_boundary, row = _refine_cells_one_level(
            evaluator,
            current,
            threshold,
            level=level,
            next_equivalent_grid_size=next_effective,
            stage_prefix=stage_prefix,
            input_cell_kind="boundary",
        )
        leaves.extend(new_leaves)
        effective = next_effective
        diagnostics.append(row)
        current = list(next_boundary)
    leaves.extend(current)
    return tuple(leaves), tuple(current), tuple(diagnostics), int(effective)


def _probe_cell_centers(
    evaluator: SliceGridEvaluator,
    cells: Sequence[AdaptiveCell],
    threshold: float,
    *,
    stage: str,
) -> tuple[CenterProbeResult, tuple[AdaptiveCell, ...]]:
    """Exactly evaluate every cell center and identify hidden crossings."""

    coordinates = tuple(
        (
            0.5 * (cell.x0 + cell.x1),
            0.5 * (cell.y0 + cell.y1),
        )
        for cell in cells
    )
    _evaluate_coordinates_from_parent_cells(
        evaluator,
        coordinates,
        cells,
        stage=stage,
    )
    values = np.asarray(
        [evaluator.evaluate(*coordinate) for coordinate in coordinates],
        dtype=float,
    )
    suspicious = np.asarray(
        [
            not _cell_brackets(cell.values, threshold)
            and min(*cell.values, float(center)) <= float(threshold)
            <= max(*cell.values, float(center))
            for cell, center in zip(cells, values)
        ],
        dtype=bool,
    )
    return (
        CenterProbeResult(
            coordinates=np.asarray(coordinates, dtype=float),
            survival_fraction=values,
            suspicious_mask=suspicious,
        ),
        tuple(cell for cell, selected in zip(cells, suspicious) if selected),
    )


def _refine_suspicious_to_target(
    evaluator: SliceGridEvaluator,
    base_cells: Sequence[AdaptiveCell],
    suspicious_cells: Sequence[AdaptiveCell],
    threshold: float,
    starting_grid_size: int,
    target_grid_size: int,
    stage_prefix: str,
) -> tuple[
    tuple[AdaptiveCell, ...],
    tuple[AdaptiveCell, ...],
    tuple[LocalRefinementLevel, ...],
    int,
]:
    """Refine probe-selected cells once, then follow confirmed boundaries."""

    suspicious = set(suspicious_cells)
    leaves = [cell for cell in base_cells if cell not in suspicious]
    current = tuple(suspicious_cells)
    diagnostics = []
    effective = int(starting_grid_size)
    level = 0
    while current and effective < int(target_grid_size):
        level += 1
        next_effective = _next_nested_size(effective)
        new_leaves, next_boundary, row = _refine_cells_one_level(
            evaluator,
            current,
            threshold,
            level=level,
            next_equivalent_grid_size=next_effective,
            stage_prefix=stage_prefix,
            input_cell_kind=("suspicious" if level == 1 else "boundary"),
        )
        leaves.extend(new_leaves)
        current = next_boundary
        diagnostics.append(row)
        effective = next_effective
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
    if int(maximum_grid_size) != DEFAULT_MAX_GRID_SIZE:
        raise ValueError("global PID contour search size must be fixed at 9x9")
    levels: list[GridLevel] = []
    size = 3
    center = 0.5 * (float(lower) + float(upper))
    evaluator.evaluate(
        center,
        center,
        stage=f"{name}:anchor_1x1",
    )
    final_axis = None
    final_field = None
    boundary = False
    while True:
        before = evaluator.cached_point_count
        axis, field = _nested_regular_grid(
            evaluator,
            lower,
            upper,
            size,
            previous_axis=(None if final_axis is None else final_axis),
            stage=f"{name}:global_{size}x{size}",
        )
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
    center_probe: Optional[CenterProbeResult] = None
    if boundary:
        adaptive_cells, final_boundary_cells, local_levels, effective = (
            _refine_boundary_to_target(
                evaluator,
                base_cells,
                threshold,
                int(final_axis.size),
                int(final_boundary_grid_size),
                str(name),
            )
        )
        segments = _boundary_segments_from_cells(final_boundary_cells, threshold)
        stop_reason = (
            "boundary_found_on_global_grid_then_refined_locally_to_target_spacing"
        )
    else:
        center_probe, suspicious_cells = _probe_cell_centers(
            evaluator,
            base_cells,
            threshold,
            stage=f"{name}:center_probe_{final_axis.size}x{final_axis.size}_cells",
        )
        if suspicious_cells:
            adaptive_cells, final_boundary_cells, local_levels, effective = (
                _refine_suspicious_to_target(
                    evaluator,
                    base_cells,
                    suspicious_cells,
                    threshold,
                    int(final_axis.size),
                    int(final_boundary_grid_size),
                    str(name),
                )
            )
            segments = _boundary_segments_from_cells(
                final_boundary_cells, threshold
            )
            stop_reason = (
                "center_probe_detected_suspicious_cells_then_refined_"
                "locally_to_target_spacing"
            )
        else:
            adaptive_cells = tuple(base_cells)
            final_boundary_cells = ()
            effective = int(final_axis.size)
            segments = np.empty((0, 2, 2), dtype=float)
            stop_reason = (
                "no_boundary_detected_through_global_{}x{}_and_cell_center_probes"
            ).format(final_axis.size, final_axis.size)

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
        center_probe=center_probe,
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
    probe = projection.center_probe
    evaluated_values = (
        adaptive_values
        if probe is None
        else np.concatenate((adaptive_values, probe.survival_fraction))
    )
    return {
        "name": projection.name,
        "visible_axes": [projection.first_axis, projection.second_axis],
        "hidden_axis": projection.hidden_axis,
        "hidden_gain_fixed_at_failure_recorded": projection.hidden_gain,
        "global_grid_size_at_stop": int(projection.axis_log_ratio.size),
        "minimum_survival_fraction": float(np.min(evaluated_values)),
        "maximum_survival_fraction": float(np.max(evaluated_values)),
        "boundary_first_seen_global_grid_size": projection.boundary_first_seen_grid_size,
        "boundary_detection_source": (
            "global_grid"
            if projection.boundary_first_seen_grid_size is not None
            else (
                "center_probe"
                if probe is not None and np.any(probe.suspicious_mask)
                else None
            )
        ),
        "center_probe": {
            "performed": probe is not None,
            "evaluated_point_count": (
                0 if probe is None else int(probe.coordinates.shape[0])
            ),
            "suspicious_cell_count": (
                0 if probe is None else int(np.count_nonzero(probe.suspicious_mask))
            ),
            "minimum_survival_fraction": (
                None
                if probe is None
                else float(np.min(probe.survival_fraction))
            ),
            "maximum_survival_fraction": (
                None
                if probe is None
                else float(np.max(probe.survival_fraction))
            ),
        },
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
                "input_cell_kind": level.input_cell_kind,
                "input_cell_count": level.input_cell_count,
                "input_boundary_cell_count": (
                    level.input_cell_count
                    if level.input_cell_kind == "boundary"
                    else 0
                ),
                "input_suspicious_cell_count": (
                    level.input_cell_count
                    if level.input_cell_kind == "suspicious"
                    else 0
                ),
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
    analysis_started = time.perf_counter()
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
                projection_name=name,
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

        cold_audit: Mapping[str, Any] = {"enabled": False}
        if bool(arguments.trim_cold_audit):
            audit_points = []
            for projection_name, *_unused in PROJECTION_SPECS:
                candidates = sorted(
                    (
                        row.log_ratio
                        for row in group_evaluator._evaluation_cache.values()
                        if row.stage == f"{projection_name}:global_3x3"
                    ),
                    key=lambda value: tuple(value),
                )
                if candidates:
                    audit_points.append(candidates[0])
            cold_audit = group_evaluator.cold_start_audit(audit_points)

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
            "{} / {} / single-group survival; alpha={:.1f}%".format(
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
            probe = projection.center_probe
            projection_arrays[
                f"{projection.name}_center_probe_log_ratio"
            ] = (
                np.empty((0, 2), dtype=float)
                if probe is None
                else probe.coordinates
            )
            projection_arrays[
                f"{projection.name}_center_probe_survival_fraction"
            ] = (
                np.empty(0, dtype=float)
                if probe is None
                else probe.survival_fraction
            )
            projection_arrays[
                f"{projection.name}_center_probe_suspicious_mask"
            ] = (
                np.empty(0, dtype=bool)
                if probe is None
                else probe.suspicious_mask
            )
        np.savez_compressed(
            output / "gain_contour.npz",
            base_gain=group_evaluator.base_gain,
            success_gain=group_evaluator.success_gain,
            success_group_log_ratio=group_evaluator.success_log_ratio,
            success_baseline_pole_valid_mask=group_evaluator.success_pole_valid_mask,
            success_baseline_stable_mask=group_evaluator.success_stable_mask,
            failure_group_pole_valid_mask=group_evaluator.pole_valid_mask(np.zeros(3)),
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
                "unresolved_trim": "excluded from the all-success-stable conditioning set at baseline; treated as unstable under an intervention",
                "plot": "other three groups fixed at success gains; hidden selected-group gain fixed at its failure recorded value",
            },
            "method": {
                "closed_loop_model": "full_26_state_sampled_data_first_order_actuator",
                "linearization": "exact active-branch analytic Jacobian of the implemented sampled closed-loop map",
                "gain_surrogate": None,
                "eigensystems": "batched numpy.linalg.eigvals on the complete sample-by-26-by-26 Jacobian stack",
                "global_search": "nested 3x3 -> 5x5 -> 9x9, stopping globally as soon as the survival boundary is detected; no global 17x17 grid is constructed",
                "center_probe": "if 9x9 corners show no boundary, exactly evaluate one center per 9x9 cell and mark only cells where corners plus center bracket alpha",
                "local_refinement": "refine global boundary cells directly to 33x33-equivalent spacing; otherwise refine center-probe suspicious cells to 17x17-equivalent spacing and only confirmed boundary cells to 33x33-equivalent spacing",
                "grid_reuse": "all repeated gain points are cached and never reevaluated",
                "trim_continuation": "cold 1x1 center anchor; parent-cell bilinear continuation in gain-scaled PID-integral coordinates (K_I * integral_error), mapped back to raw integral state before the exact nonlinear trim solve, for nested global points, center probes, and local refinement",
                "refinement_scheduling": "freeze all predictors for one adaptive wave, then queue every gain-by-plant-chunk task before the next wave",
                "trim_fallback": "per sample: warm predictor, nearest already-converged visible gain point, then the original generic initialization",
                "boundary": "piecewise-linear alpha-survival segments retained as numeric diagnostics but not overlaid on the heatmap",
                "heatmap": "adaptive leaf cells retain their survival values; only boundary neighborhoods become finer",
                "topology_limitation": "If no boundary is detected through the global 9x9 grid and one center probe per cell, sub-cell boundary islands away from the cell center are not certified absent.",
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
                "pole_valid_count": int(
                    np.count_nonzero(group_evaluator.success_pole_valid_mask)
                ),
                "trim_unresolved_count": int(
                    np.count_nonzero(~group_evaluator.success_pole_valid_mask)
                ),
                "stable_count": baseline_stable_count,
                "unstable_count": int(arguments.samples) - baseline_stable_count,
                "stable_fraction": float(np.mean(group_evaluator.success_stable_mask)),
            },
            "failure_gain_single_group_intervention": {
                "pole_valid_count": int(
                    np.count_nonzero(group_evaluator.pole_valid_mask(np.zeros(3)))
                ),
                "trim_unresolved_count": int(
                    np.count_nonzero(~group_evaluator.pole_valid_mask(np.zeros(3)))
                ),
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
                "gain_points_with_any_unresolved_trim": group_evaluator.gain_points_with_any_unresolved_trim,
                "unresolved_trim_sample_evaluation_count": group_evaluator.unresolved_trim_sample_evaluation_count,
            },
            "trim_continuation_diagnostics": {
                **group_evaluator.diagnostic_payload(),
                "analysis_wall_seconds": float(time.perf_counter() - analysis_started),
            },
            "trim_cold_start_audit": cold_audit,
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
        help="fixed global nested-grid cap (must be 9)",
    )
    parser.add_argument(
        "--final-boundary-grid-size",
        type=int,
        default=DEFAULT_FINAL_BOUNDARY_GRID_SIZE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--trim-cold-audit",
        action="store_true",
        help="cold-solve deterministic anchor points and report branch equivalence",
    )
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
    if arguments.max_grid_size != DEFAULT_MAX_GRID_SIZE:
        raise SystemExit("--max-grid-size must be 9")
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
