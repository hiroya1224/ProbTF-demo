#!/usr/bin/env python3
"""Forward-only empirical robust pole-margin diagnostics for PID gains.

For a fixed deterministic set of plant samples from the failure-bag quotient
Gaussian, define

    delta(K) = 1 - max_i rho(F(theta_i, K)).

No FORM, probability contour, or gain surrogate is used.  The script probes
all 12 normalized PID gain directions at the recorded failure controller,
selects the two directions with largest local |d delta / d q_j|, and evaluates
an exact 2-D delta slice while holding the other ten gains at their recorded
failure values.
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
import time
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

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
    load_estimate_json,
    quotient_to_scale_free_plants,
)
from grape_param_estim.controller import (  # noqa: E402
    AllocationNearSingularError,
    ControllerConfig,
    ControllerJacobianError,
)
from grape_param_estim.controller_config import (  # noqa: E402
    PID_GAIN_NAMES,
    PidGainConfiguration,
    apply_pid_gain_configuration,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    ScaleFreePlant,
    load_vehicle_model,
)
from pid_gain_contour import _exact_matrix_with_configuration  # noqa: E402
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402

SCHEMA = "grape-param-estim/pid-forward-safe-margin-slice/v1"
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_SEED = 0
DEFAULT_PROBE_STEP = 0.05
DEFAULT_GRID_SIZE = 9
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)

# Only the supplied YAML limit_p / limit_i / limit_d numeric values are reused
# as default exploration caps.  They are PID-term output clamps in the actual
# controller implementation, not enforced gain bounds.
GAIN_CAPS = np.asarray(
    ((12.0, 12.0, 12.0), (25.0, 25.0, 25.0),
     (20.0, 20.0, 20.0), (20.0, 20.0, 20.0)),
    dtype=float,
)
GAIN_LABELS = tuple(
    f"{group}:{name[0].upper()}"
    for group in PID_GROUPS
    for name in PID_GAIN_NAMES
)


@dataclass(frozen=True)
class MarginEvaluation:
    q: np.ndarray
    gains: np.ndarray
    delta: float
    worst_spectral_radius: float
    worst_sample_index: Optional[int]
    pole_valid_mask: np.ndarray
    near_kink_mask: np.ndarray
    spectral_radius: np.ndarray
    trim_vectors: np.ndarray
    wall_seconds: float

    @property
    def safe_on_sample_set(self) -> bool:
        return bool(
            np.all(self.pole_valid_mask)
            and np.isfinite(self.delta)
            and self.delta > 0.0
        )


def _recorded_gain_matrix(estimate: Mapping[str, Any]) -> np.ndarray:
    payload = estimate["controller"]["gains"]
    return np.asarray(
        [[float(payload[g][n]) for n in PID_GAIN_NAMES] for g in PID_GROUPS],
        dtype=float,
    )


def _controller_configuration(gains: np.ndarray) -> Any:
    selected = np.asarray(gains, dtype=float)
    if selected.shape != (4, 3) or np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError("PID gain matrix must be finite, non-negative, and 4x3")
    return apply_pid_gain_configuration(
        ControllerConfig.grape(), PidGainConfiguration(selected)
    )


def _sobol_quotient_coordinates(
    estimate: Mapping[str, Any],
    covariance_name: str,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    distribution = estimate["plant_distribution"]
    center = np.asarray(distribution["quotient_coordinate"], dtype=float)
    covariance = np.asarray(distribution["covariances"][covariance_name], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    if center.shape != (13,) or covariance.shape != (13, 13):
        raise ValueError("first-order quotient distribution has invalid shape")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 13.0 * np.finfo(float).eps * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError("first-order quotient covariance is not PSD")
    active = eigenvalues > tolerance
    rank = int(np.count_nonzero(active))
    n = int(sample_count)
    if n <= 0:
        raise ValueError("sample_count must be positive")
    if rank == 0:
        return np.repeat(center[None, :], n, axis=0), {
            "effective_rank": 0,
            "eigenvalues": eigenvalues.tolist(),
            "psd_tolerance": float(tolerance),
        }

    sampler = qmc.Sobol(d=rank, scramble=True, seed=int(seed))
    if n & (n - 1) == 0:
        unit = sampler.random_base2(int(math.log2(n)))
    else:
        unit = sampler.random(n)
    lo = np.nextafter(0.0, 1.0)
    hi = np.nextafter(1.0, 0.0)
    standard = ndtri(np.clip(unit, lo, hi))
    factor = eigenvectors[:, active] * np.sqrt(eigenvalues[active])[None, :]
    quotient = center[None, :] + standard @ factor.T
    return quotient, {
        "effective_rank": rank,
        "eigenvalues": eigenvalues.tolist(),
        "psd_tolerance": float(tolerance),
        "method": "scrambled Sobol in positive-eigenvalue whitened coordinates, inverse-normal transformed",
    }


def _safe_exact_matrix_chunk_task(task: tuple[Any, ...]) -> tuple[Any, ...]:
    """Evaluate one plant chunk without exporting controller-chart exceptions.

    ControllerJacobianError and AllocationNearSingularError mean that the
    exact local pole evaluator is not defined for that particular plant/gain
    point.  They are converted to an unresolved sample inside the worker so
    multiprocessing never has to pickle those custom exception objects.
    """

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
    matrices = np.zeros((selected_indices.size, 26, 26), dtype=float)
    pole_valid = np.zeros(selected_indices.size, dtype=bool)
    near_kink = np.zeros(selected_indices.size, dtype=bool)
    trim_vectors = np.full((selected_indices.size, 10), np.nan, dtype=float)

    for local_index, plant in enumerate(plants):
        row_initial = (
            None if initial_trims is None else np.asarray(initial_trims)[local_index]
        )
        row_nearest = (
            None if nearest_trims is None else np.asarray(nearest_trims)[local_index]
        )
        try:
            result = _exact_matrix_with_configuration(
                plant=plant,
                vehicle_model=vehicle_model,
                actuator_parameters=actuator_parameters,
                controller_dt=float(controller_dt),
                controller_configuration=controller_configuration,
                initial_trim=row_initial,
                nearest_trim=row_nearest,
            )
        except (ControllerJacobianError, AllocationNearSingularError):
            continue

        matrix = result[0]
        near_kink[local_index] = bool(result[1])
        trim_vectors[local_index] = np.asarray(result[2], dtype=float)
        if matrix is not None:
            matrices[local_index] = np.asarray(matrix, dtype=float)
            pole_valid[local_index] = True

    return (
        selected_indices,
        matrices,
        pole_valid,
        near_kink,
        trim_vectors,
    )


class ForwardMarginEvaluator:
    """Exact full-12-gain evaluator over one fixed plant sample set."""

    def __init__(self, *, plants: Sequence[ScaleFreePlant], vehicle_model: Any,
                 actuator_parameters: Any, controller_dt: float, workers: int) -> None:
        self.plants = tuple(plants)
        self.vehicle_model = vehicle_model
        self.actuator_parameters = actuator_parameters
        self.controller_dt = float(controller_dt)
        self.workers = min(int(workers), len(self.plants))
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        self._executor = None if self.workers == 1 else ProcessPoolExecutor(max_workers=self.workers)
        self._cache: dict[tuple[float, ...], MarginEvaluation] = {}

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @staticmethod
    def _key(q: Sequence[float]) -> tuple[float, ...]:
        selected = np.asarray(q, dtype=float)
        if selected.shape != (12,) or np.any(~np.isfinite(selected)):
            raise ValueError("normalized gain coordinate must be finite length 12")
        if np.any(selected < 0.0) or np.any(selected > 1.0):
            raise ValueError("normalized gain coordinate must lie in [0,1]^12")
        # Exact in-process memoization must not merge nearby finite-difference
        # points.  The same deterministic floating-point coordinate produces
        # the same tuple without quantizing the search itself.
        return tuple(float(x) for x in selected)

    def _nearest_trims(self, q: np.ndarray) -> Optional[np.ndarray]:
        if not self._cache:
            return None
        nearest = min(
            self._cache.values(),
            key=lambda row: float(np.linalg.norm(row.q - q)),
        )
        return nearest.trim_vectors.copy()

    def evaluate(self, q: Sequence[float]) -> MarginEvaluation:
        key = self._key(q)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        selected_q = np.asarray(key, dtype=float)
        gains = selected_q.reshape(4, 3) * GAIN_CAPS
        configuration = _controller_configuration(gains)
        nearest = self._nearest_trims(selected_q)

        sample_count = len(self.plants)
        chunk_count = min(sample_count, max(1, 2 * self.workers))
        chunks = [c for c in np.array_split(np.arange(sample_count), chunk_count) if c.size]
        tasks = [
            (
                tuple(int(i) for i in chunk),
                tuple(self.plants[int(i)] for i in chunk),
                self.vehicle_model,
                self.actuator_parameters,
                self.controller_dt,
                configuration,
                None if nearest is None else nearest[chunk],
                None,
            )
            for chunk in chunks
        ]

        started = time.perf_counter()
        rows = (
            [_safe_exact_matrix_chunk_task(task) for task in tasks]
            if self._executor is None
            else list(self._executor.map(_safe_exact_matrix_chunk_task, tasks))
        )
        wall_seconds = time.perf_counter() - started

        matrices = np.zeros((sample_count, 26, 26), dtype=float)
        pole_valid = np.zeros(sample_count, dtype=bool)
        near_kink = np.zeros(sample_count, dtype=bool)
        trims = np.full((sample_count, 10), np.nan, dtype=float)
        filled = np.zeros(sample_count, dtype=bool)
        for row in rows:
            indices = np.asarray(row[0], dtype=int)
            matrices[indices] = np.asarray(row[1], dtype=float)
            pole_valid[indices] = np.asarray(row[2], dtype=bool)
            near_kink[indices] = np.asarray(row[3], dtype=bool)
            trims[indices] = np.asarray(row[4], dtype=float)
            filled[indices] = True
        if not np.all(filled):
            raise RuntimeError("plant sample ordering changed during exact evaluation")

        radii = np.full(sample_count, np.nan, dtype=float)
        if np.any(pole_valid):
            eig = np.linalg.eigvals(matrices[pole_valid])
            radii[pole_valid] = np.max(np.abs(eig), axis=1)
        if np.all(pole_valid):
            worst = int(np.argmax(radii))
            worst_radius = float(radii[worst])
            delta = float(1.0 - worst_radius)
        else:
            worst = None
            worst_radius = float("nan")
            delta = float("nan")

        result = MarginEvaluation(
            q=selected_q.copy(), gains=gains.copy(), delta=delta,
            worst_spectral_radius=worst_radius, worst_sample_index=worst,
            pole_valid_mask=pole_valid.copy(), near_kink_mask=near_kink.copy(),
            spectral_radius=radii.copy(), trim_vectors=trims.copy(),
            wall_seconds=float(wall_seconds),
        )
        self._cache[key] = result
        return result


def _forward_sensitivities(evaluator: ForwardMarginEvaluator, q0: np.ndarray, step: float):
    baseline = evaluator.evaluate(q0)
    derivatives = np.full(12, np.nan)
    minus_delta = np.full(12, np.nan)
    plus_delta = np.full(12, np.nan)
    for axis in range(12):
        qm = q0.copy(); qp = q0.copy()
        qm[axis] = max(0.0, q0[axis] - step)
        qp[axis] = min(1.0, q0[axis] + step)
        minus = baseline if qm[axis] == q0[axis] else evaluator.evaluate(qm)
        plus = baseline if qp[axis] == q0[axis] else evaluator.evaluate(qp)
        minus_delta[axis] = minus.delta
        plus_delta[axis] = plus.delta
        if np.isfinite(minus.delta) and np.isfinite(plus.delta):
            width = qp[axis] - qm[axis]
            if width > 0.0:
                derivatives[axis] = (plus.delta - minus.delta) / width
    return baseline, derivatives, minus_delta, plus_delta


def _select_axes(derivatives: np.ndarray) -> tuple[int, int]:
    finite = np.flatnonzero(np.isfinite(derivatives))
    if finite.size < 2:
        raise RuntimeError("fewer than two gain directions have finite forward sensitivities")
    order = finite[np.argsort(-np.abs(derivatives[finite]))]
    return int(order[0]), int(order[1])


def _grid_slice(evaluator: ForwardMarginEvaluator, q0: np.ndarray,
                first_axis: int, second_axis: int, grid_size: int):
    axis = np.linspace(0.0, 1.0, int(grid_size))
    delta = np.full((axis.size, axis.size), np.nan)
    resolved = np.zeros_like(delta)
    near_kink = np.zeros_like(delta)
    requests = []
    for i, first in enumerate(axis):
        for j, second in enumerate(axis):
            q = q0.copy(); q[first_axis] = first; q[second_axis] = second
            requests.append((float(np.linalg.norm(q - q0)), i, j, q))
    requests.sort(key=lambda item: item[0])
    for _distance, i, j, q in requests:
        row = evaluator.evaluate(q)
        delta[i, j] = row.delta
        resolved[i, j] = float(np.mean(row.pole_valid_mask))
        near_kink[i, j] = float(np.mean(row.near_kink_mask))
    return axis, delta, resolved, near_kink


def _row_payload(row: MarginEvaluation) -> Mapping[str, Any]:
    return {
        "delta": None if not np.isfinite(row.delta) else float(row.delta),
        "worst_spectral_radius": None if not np.isfinite(row.worst_spectral_radius) else float(row.worst_spectral_radius),
        "worst_sample_index": row.worst_sample_index,
        "pole_valid_count": int(np.count_nonzero(row.pole_valid_mask)),
        "pole_invalid_count": int(np.count_nonzero(~row.pole_valid_mask)),
        "near_piecewise_kink_count": int(np.count_nonzero(row.near_kink_mask)),
        "safe_on_fixed_sample_set": row.safe_on_sample_set,
    }


def _plot(output: Path, case_name: str, derivatives: np.ndarray,
          first_axis: int, second_axis: int, q_axis: np.ndarray,
          delta_grid: np.ndarray, failure_q: np.ndarray,
          success_q: Optional[np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)

    ax = axes[0]
    x = np.arange(12)
    ax.bar(x, derivatives)
    ax.axhline(0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(GAIN_LABELS, rotation=55, ha="right")
    ax.set_ylabel(r"local forward sensitivity $d\delta/dq_j$")
    ax.set_title("Failure-gain local forward probes")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    caps = GAIN_CAPS.reshape(-1)
    x_gain = q_axis * caps[first_axis]
    y_gain = q_axis * caps[second_axis]
    field = np.ma.masked_invalid(delta_grid.T)
    finite = delta_grid[np.isfinite(delta_grid)]
    if finite.size:
        low, high = float(np.min(finite)), float(np.max(finite))
        if low == high:
            span = max(abs(low), 1.0e-8)
            levels = np.linspace(low - span, high + span, 9)
        else:
            levels = np.linspace(low, high, 13)
        if low < 0.0 < high:
            levels = np.unique(np.sort(np.r_[levels, 0.0]))
        fill = ax.contourf(x_gain, y_gain, field, levels=levels)
        lines = ax.contour(x_gain, y_gain, field, levels=levels)
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.2e")
        fig.colorbar(fill, ax=ax).set_label(r"$\delta=1-\max_i\rho(F_i)$")
        if low <= 0.0 <= high:
            ax.contour(x_gain, y_gain, field, levels=[0.0], linewidths=2.0)

    failure_gain = failure_q * caps
    ax.scatter([failure_gain[first_axis]], [failure_gain[second_axis]], s=70, label="recorded failure")
    local = np.asarray((derivatives[first_axis], derivatives[second_axis]))
    if np.all(np.isfinite(local)) and np.linalg.norm(local) > 0.0:
        direction = local / np.linalg.norm(local)
        ax.arrow(
            failure_gain[first_axis], failure_gain[second_axis],
            0.12 * caps[first_axis] * direction[0],
            0.12 * caps[second_axis] * direction[1],
            head_width=0.025 * max(caps[first_axis], caps[second_axis]),
            length_includes_head=True,
        )
    if success_q is not None:
        success_gain = success_q * caps
        ax.scatter([success_gain[first_axis]], [success_gain[second_axis]], marker="x", s=80, label="success projection only")

    ax.set_xlabel(GAIN_LABELS[first_axis])
    ax.set_ylabel(GAIN_LABELS[second_axis])
    ax.set_xlim(0.0, caps[first_axis]); ax.set_ylim(0.0, caps[second_axis])
    ax.set_title("Exact empirical worst-pole margin slice")
    ax.legend(); ax.grid(True, alpha=0.2)
    fig.suptitle(f"{case_name}: forward-only sampled-plant pole margin diagnostics")
    fig.savefig(output / "safe_margin_slice.png", dpi=180)
    plt.close(fig)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success_path = None if arguments.success_json is None else Path(arguments.success_json).expanduser().resolve()
    success_estimate = None if success_path is None else load_estimate_json(success_path)

    failure_gains = _recorded_gain_matrix(estimate)
    caps = GAIN_CAPS.reshape(-1)
    failure_flat = failure_gains.reshape(-1)
    if np.any(failure_flat > caps):
        raise ValueError("recorded failure gains exceed the supplied default exploration caps")
    failure_q = failure_flat / caps

    success_q = None
    if success_estimate is not None:
        success_flat = _recorded_gain_matrix(success_estimate).reshape(-1)
        if np.all(success_flat <= caps):
            success_q = success_flat / caps

    vehicle_model = load_vehicle_model(Path(estimate["input"]["vehicle_model"]))
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
    quotient, sampling = _sobol_quotient_coordinates(
        estimate, str(arguments.covariance), int(arguments.samples), int(arguments.seed)
    )
    plants = quotient_to_scale_free_plants(
        estimate, quotient, vehicle_model.parameters, ScaleFreePlant
    )

    evaluator = ForwardMarginEvaluator(
        plants=plants, vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters, controller_dt=controller_dt,
        workers=int(arguments.workers),
    )
    try:
        baseline, derivatives, minus_delta, plus_delta = _forward_sensitivities(
            evaluator, failure_q, float(arguments.probe_step)
        )
        first_axis, second_axis = _select_axes(derivatives)
        q_axis, delta_grid, resolved_grid, kink_grid = _grid_slice(
            evaluator, failure_q, first_axis, second_axis, int(arguments.grid_size)
        )

        output = Path(arguments.output_dir).expanduser().resolve() if arguments.output_dir else estimate_path.parent / "pid_safe_margin_slices"
        output.mkdir(parents=True, exist_ok=True)
        _plot(output, str(estimate["case_name"]), derivatives, first_axis, second_axis,
              q_axis, delta_grid, failure_q, success_q)

        finite = np.isfinite(delta_grid)
        safe = finite & (delta_grid > 0.0)
        best = None
        if np.any(finite):
            flat_index = int(np.nanargmax(delta_grid))
            ij = np.unravel_index(flat_index, delta_grid.shape)
            best_q = failure_q.copy()
            best_q[first_axis] = q_axis[ij[0]]
            best_q[second_axis] = q_axis[ij[1]]
            best = {
                "grid_index": [int(ij[0]), int(ij[1])],
                "delta": float(delta_grid[ij]),
                "gain_matrix": (best_q.reshape(4, 3) * GAIN_CAPS).tolist(),
            }

        np.savez_compressed(
            output / "safe_margin_slice.npz",
            quotient_coordinates=quotient,
            failure_gain_matrix=failure_gains,
            gain_caps=GAIN_CAPS,
            failure_q=failure_q,
            local_derivative_d_delta_d_q=derivatives,
            probe_minus_delta=minus_delta,
            probe_plus_delta=plus_delta,
            selected_axes=np.asarray((first_axis, second_axis)),
            grid_q_axis=q_axis,
            grid_delta=delta_grid,
            grid_resolved_fraction=resolved_grid,
            grid_near_kink_fraction=kink_grid,
        )

        payload = {
            "schema": SCHEMA,
            "source_commit": source_commit(_PROJECT_ROOT),
            "case_name": str(estimate["case_name"]),
            "estimate_json": str(estimate_path),
            "success_json": None if success_path is None else str(success_path),
            "covariance": str(arguments.covariance),
            "sample_count": int(arguments.samples),
            "seed": int(arguments.seed),
            "definition": {
                "delta": "1 - maximum spectral radius over the fixed failure-bag plant sample set",
                "safe_on_fixed_sample_set": "delta > 0 and every sampled plant has a resolved hover-trim pole evaluation",
                "probability_interpretation": None,
                "success_controller_role": "optional comparison projection only; not used in plant sampling, axis selection, or margin evaluation",
            },
            "plant_sampling": sampling,
            "gain_coordinate": {
                "coordinate": "q_j = K_j / K_max_j in [0,1]",
                "groups": list(PID_GROUPS),
                "gain_order_within_group": ["P", "I", "D"],
                "caps": GAIN_CAPS.tolist(),
                "cap_semantics": "supplied YAML limit_p/limit_i/limit_d numeric values reused only as exploration caps",
            },
            "recorded_failure": {"gain_matrix": failure_gains.tolist(), **_row_payload(baseline)},
            "local_forward_probe": {
                "normalized_probe_step": float(arguments.probe_step),
                "labels": list(GAIN_LABELS),
                "d_delta_d_q": [None if not np.isfinite(x) else float(x) for x in derivatives],
                "minus_delta": [None if not np.isfinite(x) else float(x) for x in minus_delta],
                "plus_delta": [None if not np.isfinite(x) else float(x) for x in plus_delta],
                "selected_two_axes": [
                    {"flat_index": first_axis, "label": GAIN_LABELS[first_axis], "d_delta_d_q": float(derivatives[first_axis])},
                    {"flat_index": second_axis, "label": GAIN_LABELS[second_axis], "d_delta_d_q": float(derivatives[second_axis])},
                ],
            },
            "two_dimensional_exact_slice": {
                "grid_size": int(arguments.grid_size),
                "first_axis": GAIN_LABELS[first_axis],
                "second_axis": GAIN_LABELS[second_axis],
                "other_ten_gains": "fixed at the recorded failure controller",
                "finite_grid_point_count": int(np.count_nonzero(finite)),
                "unresolved_grid_point_count": int(delta_grid.size - np.count_nonzero(finite)),
                "safe_grid_point_count": int(np.count_nonzero(safe)),
                "minimum_delta": None if not np.any(finite) else float(np.nanmin(delta_grid)),
                "maximum_delta": None if not np.any(finite) else float(np.nanmax(delta_grid)),
                "best_evaluated_grid_point": best,
            },
            "diagnostic_only": "This slice is not a certified 12-D safe set or an optimized controller; it is a forward-evaluated direction diagnostic.",
            "elapsed_seconds": float(time.perf_counter() - started),
            "files": {
                "figure": str(output / "safe_margin_slice.png"),
                "npz": str(output / "safe_margin_slice.npz"),
            },
        }
        write_json(output / "safe_margin_slice.json", payload)
        return payload
    finally:
        evaluator.close()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--estimate-json", required=True)
    p.add_argument("--success-json")
    p.add_argument("--covariance", choices=COVARIANCE_NAMES, default="conservative_fusion")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--probe-step", type=float, default=DEFAULT_PROBE_STEP)
    p.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--output-dir")
    return p


def main() -> None:
    args = _parser().parse_args()
    if not (0.0 < args.probe_step <= 0.5):
        raise ValueError("--probe-step must lie in (0, 0.5]")
    if args.grid_size < 3:
        raise ValueError("--grid-size must be at least three")
    print(json.dumps(analyze(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
