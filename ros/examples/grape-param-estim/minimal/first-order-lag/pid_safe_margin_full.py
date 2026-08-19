#!/usr/bin/env python3
"""Forward-only robust pole-margin exploration over the full 12-D PID space.

For one fixed deterministic plant sample set from the failure-bag quotient
Gaussian, define

    delta(K) = 1 - max_i rho(F(theta_i, K)).

The full 12 PID gains are varied simultaneously in the normalized box

    q_j = K_j / K_max_j in [0, 1].

No FORM problem, gain-grid inversion, probability contour, or gain surrogate
is solved.  A scrambled Sobol design proposes gain vectors; every proposed
gain is checked by the exact hover-trim / sampled closed-loop pole evaluator
on the same fixed plant sample set.

After the 12-D scan, the script:
  * reports the best evaluated gain and every empirically safe gain,
  * reports the safe sampled gain nearest to the recorded failure controller,
  * summarizes the empirical safe gain cloud,
  * probes all 12 directions locally around the best 12-D point,
  * draws one exact 2-D delta contour through the two strongest local
    directions while the other ten gains are fixed at that best 12-D point.

The 2-D contour is therefore visualization after the full 12-D exploration;
it does not restrict the 12-D search.

Progress is shown with tqdm for the full gain scan, the local forward probes,
and the final exact 2-D contour evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc, spearmanr
from tqdm.auto import tqdm


_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import (  # noqa: E402
    COVARIANCE_NAMES,
    actuator_parameters_from_estimate,
    load_estimate_json,
    quotient_to_scale_free_plants,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    ScaleFreePlant,
    load_vehicle_model,
)
from pid_safe_margin_slices import (  # noqa: E402
    GAIN_CAPS,
    GAIN_LABELS,
    ForwardMarginEvaluator,
    MarginEvaluation,
    _recorded_gain_matrix,
    _row_payload,
    _sobol_quotient_coordinates,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


SCHEMA = "grape-param-estim/pid-forward-safe-margin-full/v1"
DEFAULT_PLANT_SAMPLE_COUNT = 128
DEFAULT_GAIN_SAMPLE_COUNT = 256
DEFAULT_SEED = 0
DEFAULT_GAIN_SEED = 1
DEFAULT_PROBE_STEP = 0.05
DEFAULT_GRID_SIZE = 9
DEFAULT_TOP_COUNT = 20
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


def _sobol_gain_coordinates(
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    n = int(sample_count)
    if n <= 0:
        raise ValueError("gain sample count must be positive")

    sampler = qmc.Sobol(d=12, scramble=True, seed=int(seed))
    if n & (n - 1) == 0:
        q = sampler.random_base2(int(math.log2(n)))
        balanced_power_of_two = True
    else:
        q = sampler.random(n)
        balanced_power_of_two = False

    selected = np.asarray(q, dtype=float)
    if selected.shape != (n, 12) or np.any(~np.isfinite(selected)):
        raise RuntimeError("Sobol gain design has an invalid shape")
    if np.any(selected < 0.0) or np.any(selected > 1.0):
        raise RuntimeError("Sobol gain design left the unit box")
    return selected, {
        "method": "scrambled Sobol in the full normalized 12-D PID gain box",
        "dimension": 12,
        "sample_count": n,
        "seed": int(seed),
        "power_of_two_balanced_design": bool(balanced_power_of_two),
    }


def _evaluate_full_scan(
    evaluator: ForwardMarginEvaluator,
    gain_q: np.ndarray,
    failure_q: np.ndarray,
    *,
    progress: bool,
) -> Mapping[str, Any]:
    selected = np.asarray(gain_q, dtype=float)
    if selected.ndim != 2 or selected.shape[1] != 12:
        raise ValueError("full gain design must be N-by-12")

    count = selected.shape[0]
    delta = np.full(count, np.nan, dtype=float)
    worst_radius = np.full(count, np.nan, dtype=float)
    worst_sample = np.full(count, -1, dtype=int)
    pole_valid_count = np.zeros(count, dtype=int)
    near_kink_count = np.zeros(count, dtype=int)
    wall_seconds = np.zeros(count, dtype=float)

    # Evaluation order only affects continuation warm starts, never the
    # mathematical gain design or stored result order.
    order = np.argsort(np.linalg.norm(selected - failure_q[None, :], axis=1))
    best_history = np.full(count, np.nan, dtype=float)
    best_so_far = -np.inf
    safe_so_far = 0
    unresolved_so_far = 0

    bar = tqdm(
        order,
        desc="12-D PID Sobol scan",
        unit="gain",
        dynamic_ncols=True,
        disable=not progress,
    )
    for step_index, row_index in enumerate(bar):
        row = evaluator.evaluate(selected[int(row_index)])
        delta[row_index] = row.delta
        worst_radius[row_index] = row.worst_spectral_radius
        worst_sample[row_index] = (
            -1 if row.worst_sample_index is None else int(row.worst_sample_index)
        )
        pole_valid_count[row_index] = int(np.count_nonzero(row.pole_valid_mask))
        near_kink_count[row_index] = int(np.count_nonzero(row.near_kink_mask))
        wall_seconds[row_index] = float(row.wall_seconds)

        if np.isfinite(row.delta):
            best_so_far = max(best_so_far, float(row.delta))
            if row.delta > 0.0:
                safe_so_far += 1
        else:
            unresolved_so_far += 1
        best_history[step_index] = (
            np.nan if not np.isfinite(best_so_far) else best_so_far
        )
        if progress:
            bar.set_postfix(
                best_delta=(
                    "n/a" if not np.isfinite(best_so_far) else f"{best_so_far:.3e}"
                ),
                safe=safe_so_far,
                unresolved=unresolved_so_far,
            )

    return {
        "delta": delta,
        "worst_spectral_radius": worst_radius,
        "worst_sample_index": worst_sample,
        "pole_valid_count": pole_valid_count,
        "near_kink_count": near_kink_count,
        "wall_seconds": wall_seconds,
        "evaluation_order": order.astype(int),
        "best_delta_history_in_evaluation_order": best_history,
    }


def _best_from_scan(
    scan_q: np.ndarray,
    scan_delta: np.ndarray,
    failure_q: np.ndarray,
    failure_row: MarginEvaluation,
) -> tuple[np.ndarray, float, str, Optional[int]]:
    candidates: list[tuple[float, np.ndarray, str, Optional[int]]] = []
    if np.isfinite(failure_row.delta):
        candidates.append(
            (
                float(failure_row.delta),
                np.asarray(failure_q, dtype=float).copy(),
                "recorded_failure",
                None,
            )
        )
    for index in np.flatnonzero(np.isfinite(scan_delta)):
        candidates.append(
            (
                float(scan_delta[index]),
                np.asarray(scan_q[index], dtype=float).copy(),
                "sobol_scan",
                int(index),
            )
        )
    if not candidates:
        raise RuntimeError("no finite full-PID gain evaluation is available")
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[0], best[2], best[3]


def _safe_cloud_summary(
    scan_q: np.ndarray,
    scan_delta: np.ndarray,
    failure_q: np.ndarray,
) -> tuple[Mapping[str, Any], np.ndarray, Optional[int], Optional[int]]:
    finite = np.isfinite(scan_delta)
    safe = finite & (scan_delta > 0.0)
    safe_indices = np.flatnonzero(safe)
    safe_q = np.asarray(scan_q[safe], dtype=float)

    best_safe_index: Optional[int] = None
    nearest_safe_index: Optional[int] = None
    if safe_indices.size:
        best_safe_index = int(
            safe_indices[np.argmax(scan_delta[safe_indices])]
        )
        distances = np.linalg.norm(
            scan_q[safe_indices] - failure_q[None, :], axis=1
        )
        nearest_safe_index = int(safe_indices[np.argmin(distances)])

    payload: dict[str, Any] = {
        "safe_gain_sample_count": int(safe_indices.size),
        "safe_gain_sample_fraction_of_scan": float(
            safe_indices.size / scan_q.shape[0]
        ),
        "fraction_semantics": (
            "fraction of the uniformly explored Sobol gain points that pass "
            "the fixed-plant-sample worst-pole test; not a plant stability probability"
        ),
        "best_safe_scan_index": best_safe_index,
        "nearest_safe_to_failure_scan_index": nearest_safe_index,
        "nearest_safe_to_failure_normalized_distance": (
            None
            if nearest_safe_index is None
            else float(
                np.linalg.norm(scan_q[nearest_safe_index] - failure_q)
            )
        ),
    }

    if safe_q.shape[0] == 0:
        payload["empirical_safe_cloud"] = None
        return payload, safe_q, best_safe_index, nearest_safe_index

    center = np.mean(safe_q, axis=0)
    cloud: dict[str, Any] = {
        "center_q": center.tolist(),
        "minimum_q": np.min(safe_q, axis=0).tolist(),
        "maximum_q": np.max(safe_q, axis=0).tolist(),
    }
    if safe_q.shape[0] >= 2:
        covariance = np.cov(safe_q, rowvar=False, bias=True)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        cloud["covariance_q"] = covariance.tolist()
        cloud["principal_variances"] = eigenvalues[order].tolist()
        cloud["principal_directions"] = eigenvectors[:, order].T.tolist()
    else:
        cloud["covariance_q"] = None
        cloud["principal_variances"] = None
        cloud["principal_directions"] = None
    payload["empirical_safe_cloud"] = cloud
    return payload, safe_q, best_safe_index, nearest_safe_index


def _scan_correlations(scan_q: np.ndarray, scan_delta: np.ndarray) -> np.ndarray:
    result = np.full(12, np.nan, dtype=float)
    finite = np.isfinite(scan_delta)
    if np.count_nonzero(finite) < 3:
        return result
    for axis in range(12):
        value = spearmanr(scan_q[finite, axis], scan_delta[finite])[0]
        if np.isfinite(value):
            result[axis] = float(value)
    return result


def _forward_probe(
    evaluator: ForwardMarginEvaluator,
    q0: np.ndarray,
    step: float,
    *,
    progress: bool,
) -> tuple[MarginEvaluation, np.ndarray, np.ndarray, np.ndarray]:
    baseline = evaluator.evaluate(q0)
    derivatives = np.full(12, np.nan, dtype=float)
    minus_delta = np.full(12, np.nan, dtype=float)
    plus_delta = np.full(12, np.nan, dtype=float)

    bar = tqdm(
        range(12),
        desc="Local 12-D forward probe",
        unit="axis",
        dynamic_ncols=True,
        disable=not progress,
    )
    for axis in bar:
        q_minus = q0.copy()
        q_plus = q0.copy()
        q_minus[axis] = max(0.0, q0[axis] - step)
        q_plus[axis] = min(1.0, q0[axis] + step)

        minus = (
            baseline
            if q_minus[axis] == q0[axis]
            else evaluator.evaluate(q_minus)
        )
        plus = (
            baseline
            if q_plus[axis] == q0[axis]
            else evaluator.evaluate(q_plus)
        )
        minus_delta[axis] = minus.delta
        plus_delta[axis] = plus.delta

        if np.isfinite(minus.delta) and np.isfinite(plus.delta):
            width = q_plus[axis] - q_minus[axis]
            if width > 0.0:
                derivatives[axis] = (plus.delta - minus.delta) / width
        elif np.isfinite(baseline.delta) and np.isfinite(plus.delta):
            width = q_plus[axis] - q0[axis]
            if width > 0.0:
                derivatives[axis] = (plus.delta - baseline.delta) / width
        elif np.isfinite(baseline.delta) and np.isfinite(minus.delta):
            width = q0[axis] - q_minus[axis]
            if width > 0.0:
                derivatives[axis] = (baseline.delta - minus.delta) / width

    return baseline, derivatives, minus_delta, plus_delta


def _select_visualization_axes(
    derivatives: np.ndarray,
    correlations: np.ndarray,
) -> tuple[int, int, str]:
    finite_derivative = np.flatnonzero(np.isfinite(derivatives))
    if finite_derivative.size >= 2:
        order = finite_derivative[
            np.argsort(-np.abs(derivatives[finite_derivative]))
        ]
        return int(order[0]), int(order[1]), "largest local |d delta / d q|"

    finite_correlation = np.flatnonzero(np.isfinite(correlations))
    if finite_correlation.size >= 2:
        order = finite_correlation[
            np.argsort(-np.abs(correlations[finite_correlation]))
        ]
        return int(order[0]), int(order[1]), "largest full-scan |Spearman correlation|"

    raise RuntimeError(
        "fewer than two gain directions have usable local or scan diagnostics"
    )


def _exact_slice(
    evaluator: ForwardMarginEvaluator,
    center_q: np.ndarray,
    first_axis: int,
    second_axis: int,
    grid_size: int,
    *,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, int(grid_size))
    delta = np.full((axis.size, axis.size), np.nan, dtype=float)
    resolved = np.zeros_like(delta)
    near_kink = np.zeros_like(delta)

    requests = []
    for i, first in enumerate(axis):
        for j, second in enumerate(axis):
            q = center_q.copy()
            q[first_axis] = float(first)
            q[second_axis] = float(second)
            requests.append(
                (float(np.linalg.norm(q - center_q)), i, j, q)
            )
    requests.sort(key=lambda item: item[0])

    bar = tqdm(
        requests,
        desc="Exact 2-D margin contour",
        unit="gain",
        dynamic_ncols=True,
        disable=not progress,
    )
    for _distance, i, j, q in bar:
        row = evaluator.evaluate(q)
        delta[i, j] = row.delta
        resolved[i, j] = float(np.mean(row.pole_valid_mask))
        near_kink[i, j] = float(np.mean(row.near_kink_mask))

    return axis, delta, resolved, near_kink


def _gain_point_payload(
    q: np.ndarray,
    delta: float,
    *,
    scan_index: Optional[int],
) -> Mapping[str, Any]:
    selected = np.asarray(q, dtype=float)
    return {
        "scan_index": scan_index,
        "q": selected.tolist(),
        "gain_matrix": (selected.reshape(4, 3) * GAIN_CAPS).tolist(),
        "delta": float(delta),
        "worst_spectral_radius": float(1.0 - delta),
    }


def _top_scan_points(
    scan_q: np.ndarray,
    scan_delta: np.ndarray,
    count: int,
) -> list[Mapping[str, Any]]:
    finite = np.flatnonzero(np.isfinite(scan_delta))
    if finite.size == 0:
        return []
    ordered = finite[np.argsort(-scan_delta[finite])]
    return [
        _gain_point_payload(
            scan_q[index],
            scan_delta[index],
            scan_index=int(index),
        )
        for index in ordered[: int(count)]
    ]


def _plot_overview(
    *,
    output: Path,
    case_name: str,
    scan_q: np.ndarray,
    scan_delta: np.ndarray,
    evaluation_order: np.ndarray,
    best_history: np.ndarray,
    failure_q: np.ndarray,
    best_q: np.ndarray,
    nearest_safe_q: Optional[np.ndarray],
    safe_q: np.ndarray,
    success_q: Optional[np.ndarray],
    derivatives: np.ndarray,
    first_axis: int,
    second_axis: int,
    q_axis: np.ndarray,
    delta_grid: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(15.5, 10.0), constrained_layout=True
    )

    axis = axes[0, 0]
    finite_delta = scan_delta[np.isfinite(scan_delta)]
    if finite_delta.size:
        axis.hist(finite_delta, bins=min(40, max(10, int(np.sqrt(finite_delta.size)))))
        axis.axvline(0.0, linewidth=1.5)
    axis.set_xlabel(r"$\delta = 1-\max_i \rho(F_i)$")
    axis.set_ylabel("gain sample count")
    axis.set_title("Full 12-D Sobol scan margin distribution")
    axis.grid(True, alpha=0.2)

    axis = axes[0, 1]
    axis.plot(np.arange(1, best_history.size + 1), best_history)
    axis.axhline(0.0, linewidth=1.5)
    axis.set_xlabel("evaluated gain points")
    axis.set_ylabel("best delta seen so far")
    axis.set_title("Forward scan progress")
    axis.grid(True, alpha=0.2)

    axis = axes[1, 0]
    coordinate = np.arange(12)
    if safe_q.shape[0] >= 2:
        axis.boxplot(
            [safe_q[:, index] for index in range(12)],
            positions=coordinate,
            widths=0.55,
        )
    elif safe_q.shape[0] == 1:
        axis.scatter(coordinate, safe_q[0], marker="s", label="only safe scan point")
    axis.plot(coordinate, failure_q, marker="o", label="recorded failure")
    axis.plot(coordinate, best_q, marker="^", label="best evaluated")
    if nearest_safe_q is not None:
        axis.plot(
            coordinate,
            nearest_safe_q,
            marker="v",
            label="nearest sampled safe",
        )
    if success_q is not None:
        axis.plot(
            coordinate,
            success_q,
            marker="x",
            linestyle="--",
            label="success comparison only",
        )
    axis.set_xticks(coordinate)
    axis.set_xticklabels(GAIN_LABELS, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(r"normalized gain $q_j=K_j/K_{j,\max}$")
    axis.set_title("Empirical safe gain cloud in all 12 coordinates")
    axis.legend()
    axis.grid(True, alpha=0.2)

    axis = axes[1, 1]
    caps = GAIN_CAPS.reshape(-1)
    x_gain = q_axis * caps[first_axis]
    y_gain = q_axis * caps[second_axis]
    field = np.ma.masked_invalid(delta_grid.T)
    finite = delta_grid[np.isfinite(delta_grid)]
    if finite.size:
        low = float(np.min(finite))
        high = float(np.max(finite))
        if low == high:
            span = max(abs(low), 1.0e-8)
            levels = np.linspace(low - span, high + span, 9)
        else:
            levels = np.linspace(low, high, 13)
        if low < 0.0 < high:
            levels = np.unique(np.sort(np.r_[levels, 0.0]))
        filled = axis.contourf(x_gain, y_gain, field, levels=levels)
        lines = axis.contour(x_gain, y_gain, field, levels=levels)
        axis.clabel(lines, inline=True, fontsize=7, fmt="%.2e")
        figure.colorbar(filled, ax=axis).set_label(
            r"$\delta = 1-\max_i \rho(F_i)$"
        )
        if low <= 0.0 <= high:
            axis.contour(
                x_gain, y_gain, field, levels=[0.0], linewidths=2.0
            )

    best_gain = best_q * caps
    failure_gain = failure_q * caps
    axis.scatter(
        [best_gain[first_axis]],
        [best_gain[second_axis]],
        marker="^",
        s=80,
        label="best 12-D point",
    )
    axis.scatter(
        [failure_gain[first_axis]],
        [failure_gain[second_axis]],
        marker="o",
        s=65,
        label="recorded failure projection",
    )
    if nearest_safe_q is not None:
        nearest_gain = nearest_safe_q * caps
        axis.scatter(
            [nearest_gain[first_axis]],
            [nearest_gain[second_axis]],
            marker="v",
            s=70,
            label="nearest sampled safe projection",
        )
    if success_q is not None:
        success_gain = success_q * caps
        axis.scatter(
            [success_gain[first_axis]],
            [success_gain[second_axis]],
            marker="x",
            s=75,
            label="success projection only",
        )

    local = np.asarray(
        (derivatives[first_axis], derivatives[second_axis]), dtype=float
    )
    if np.all(np.isfinite(local)) and np.linalg.norm(local) > 0.0:
        direction = local / np.linalg.norm(local)
        axis.arrow(
            best_gain[first_axis],
            best_gain[second_axis],
            0.10 * caps[first_axis] * direction[0],
            0.10 * caps[second_axis] * direction[1],
            head_width=0.02 * max(caps[first_axis], caps[second_axis]),
            length_includes_head=True,
        )

    axis.set_xlabel(GAIN_LABELS[first_axis])
    axis.set_ylabel(GAIN_LABELS[second_axis])
    axis.set_xlim(0.0, caps[first_axis])
    axis.set_ylim(0.0, caps[second_axis])
    axis.set_title("Exact delta contour through best 12-D point")
    axis.legend()
    axis.grid(True, alpha=0.2)

    figure.suptitle(
        f"{case_name}: full 12-D forward-only PID robust-margin exploration"
    )
    figure.savefig(output / "safe_margin_full.png", dpi=180)
    plt.close(figure)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    progress = not bool(arguments.no_progress)

    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success_path = (
        None
        if arguments.success_json is None
        else Path(arguments.success_json).expanduser().resolve()
    )
    success_estimate = (
        None if success_path is None else load_estimate_json(success_path)
    )

    failure_gains = _recorded_gain_matrix(estimate)
    caps = GAIN_CAPS.reshape(-1)
    failure_flat = failure_gains.reshape(-1)
    if np.any(failure_flat > caps):
        raise ValueError(
            "recorded failure gains exceed the supplied default exploration caps"
        )
    failure_q = failure_flat / caps

    success_q: Optional[np.ndarray] = None
    if success_estimate is not None:
        success_flat = _recorded_gain_matrix(success_estimate).reshape(-1)
        if np.all(success_flat <= caps):
            success_q = success_flat / caps

    vehicle_model = load_vehicle_model(
        Path(estimate["input"]["vehicle_model"])
    )
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(
        estimate["controller_timing"]["median_seconds"]
    )

    quotient, plant_sampling = _sobol_quotient_coordinates(
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
    gain_q, gain_sampling = _sobol_gain_coordinates(
        int(arguments.gain_samples),
        int(arguments.gain_seed),
    )

    evaluator = ForwardMarginEvaluator(
        plants=plants,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        workers=int(arguments.workers),
    )
    try:
        failure_row = evaluator.evaluate(failure_q)

        scan = _evaluate_full_scan(
            evaluator,
            gain_q,
            failure_q,
            progress=progress,
        )
        scan_delta = np.asarray(scan["delta"], dtype=float)
        correlations = _scan_correlations(gain_q, scan_delta)

        best_q, best_delta, best_source, best_scan_index = _best_from_scan(
            gain_q,
            scan_delta,
            failure_q,
            failure_row,
        )

        (
            safe_cloud_payload,
            safe_q,
            best_safe_index,
            nearest_safe_index,
        ) = _safe_cloud_summary(
            gain_q,
            scan_delta,
            failure_q,
        )

        (
            best_row,
            derivatives,
            minus_delta,
            plus_delta,
        ) = _forward_probe(
            evaluator,
            best_q,
            float(arguments.probe_step),
            progress=progress,
        )

        first_axis, second_axis, axis_selection_method = (
            _select_visualization_axes(derivatives, correlations)
        )
        (
            q_axis,
            delta_grid,
            resolved_grid,
            kink_grid,
        ) = _exact_slice(
            evaluator,
            best_q,
            first_axis,
            second_axis,
            int(arguments.grid_size),
            progress=progress,
        )

        # success is deliberately evaluated only after the 12-D search,
        # local probes, and contour construction so it cannot affect search
        # continuation or point selection.
        success_row: Optional[MarginEvaluation] = None
        if success_q is not None:
            success_row = evaluator.evaluate(success_q)

        output = (
            Path(arguments.output_dir).expanduser().resolve()
            if arguments.output_dir is not None
            else estimate_path.parent / "pid_safe_margin_full"
        )
        output.mkdir(parents=True, exist_ok=True)

        nearest_safe_q = (
            None
            if nearest_safe_index is None
            else gain_q[nearest_safe_index].copy()
        )
        _plot_overview(
            output=output,
            case_name=str(estimate["case_name"]),
            scan_q=gain_q,
            scan_delta=scan_delta,
            evaluation_order=np.asarray(scan["evaluation_order"], dtype=int),
            best_history=np.asarray(
                scan["best_delta_history_in_evaluation_order"], dtype=float
            ),
            failure_q=failure_q,
            best_q=best_q,
            nearest_safe_q=nearest_safe_q,
            safe_q=safe_q,
            success_q=success_q,
            derivatives=derivatives,
            first_axis=first_axis,
            second_axis=second_axis,
            q_axis=q_axis,
            delta_grid=delta_grid,
        )

        np.savez_compressed(
            output / "safe_margin_full.npz",
            quotient_coordinates=quotient,
            gain_caps=GAIN_CAPS,
            failure_gain_matrix=failure_gains,
            failure_q=failure_q,
            gain_scan_q=gain_q,
            gain_scan_delta=scan_delta,
            gain_scan_worst_spectral_radius=np.asarray(
                scan["worst_spectral_radius"], dtype=float
            ),
            gain_scan_worst_sample_index=np.asarray(
                scan["worst_sample_index"], dtype=int
            ),
            gain_scan_pole_valid_count=np.asarray(
                scan["pole_valid_count"], dtype=int
            ),
            gain_scan_near_kink_count=np.asarray(
                scan["near_kink_count"], dtype=int
            ),
            gain_scan_wall_seconds=np.asarray(
                scan["wall_seconds"], dtype=float
            ),
            gain_scan_evaluation_order=np.asarray(
                scan["evaluation_order"], dtype=int
            ),
            gain_scan_best_delta_history=np.asarray(
                scan["best_delta_history_in_evaluation_order"], dtype=float
            ),
            safe_scan_q=safe_q,
            best_q=best_q,
            local_derivative_d_delta_d_q=derivatives,
            local_probe_minus_delta=minus_delta,
            local_probe_plus_delta=plus_delta,
            full_scan_spearman_correlation=correlations,
            selected_axes=np.asarray((first_axis, second_axis), dtype=int),
            slice_q_axis=q_axis,
            slice_delta=delta_grid,
            slice_resolved_fraction=resolved_grid,
            slice_near_kink_fraction=kink_grid,
        )

        finite_scan = np.isfinite(scan_delta)
        safe_scan = finite_scan & (scan_delta > 0.0)
        finite_slice = np.isfinite(delta_grid)
        safe_slice = finite_slice & (delta_grid > 0.0)

        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "source_commit": source_commit(_PROJECT_ROOT),
            "case_name": str(estimate["case_name"]),
            "estimate_json": str(estimate_path),
            "success_json": (
                None if success_path is None else str(success_path)
            ),
            "covariance": str(arguments.covariance),
            "definition": {
                "delta": (
                    "1 - maximum spectral radius over the fixed failure-bag "
                    "plant sample set"
                ),
                "safe_on_fixed_sample_set": (
                    "delta > 0 and every sampled plant has a resolved "
                    "hover-trim pole evaluation"
                ),
                "probability_interpretation": None,
                "search_method": (
                    "forward-only full-12-D scrambled Sobol gain exploration"
                ),
                "success_controller_role": (
                    "optional comparison only; evaluated after the full search "
                    "so it cannot influence plant sampling, search point "
                    "selection, continuation, local-axis selection, or contour"
                ),
            },
            "plant_sampling": {
                **dict(plant_sampling),
                "sample_count": int(arguments.samples),
                "seed": int(arguments.seed),
            },
            "gain_sampling": gain_sampling,
            "gain_coordinate": {
                "coordinate": "q_j = K_j / K_max_j in [0,1]",
                "labels": list(GAIN_LABELS),
                "caps": GAIN_CAPS.tolist(),
                "cap_semantics": (
                    "supplied YAML limit_p/limit_i/limit_d numeric values "
                    "reused only as exploration caps"
                ),
            },
            "recorded_failure": {
                "gain_matrix": failure_gains.tolist(),
                **_row_payload(failure_row),
            },
            "full_12d_scan": {
                "finite_gain_point_count": int(np.count_nonzero(finite_scan)),
                "unresolved_gain_point_count": int(
                    gain_q.shape[0] - np.count_nonzero(finite_scan)
                ),
                "safe_gain_point_count": int(np.count_nonzero(safe_scan)),
                "minimum_delta": (
                    None
                    if not np.any(finite_scan)
                    else float(np.nanmin(scan_delta))
                ),
                "maximum_delta": (
                    None
                    if not np.any(finite_scan)
                    else float(np.nanmax(scan_delta))
                ),
                "best_evaluated": _gain_point_payload(
                    best_q,
                    best_delta,
                    scan_index=best_scan_index,
                ),
                "best_evaluated_source": best_source,
                "top_evaluated_points": _top_scan_points(
                    gain_q,
                    scan_delta,
                    int(arguments.top_count),
                ),
                "spearman_gain_delta_correlation": [
                    None if not np.isfinite(value) else float(value)
                    for value in correlations
                ],
                **safe_cloud_payload,
            },
            "best_point_local_forward_probe": {
                "normalized_probe_step": float(arguments.probe_step),
                "center": {
                    "q": best_q.tolist(),
                    "gain_matrix": (
                        best_q.reshape(4, 3) * GAIN_CAPS
                    ).tolist(),
                    **_row_payload(best_row),
                },
                "labels": list(GAIN_LABELS),
                "d_delta_d_q": [
                    None if not np.isfinite(value) else float(value)
                    for value in derivatives
                ],
                "minus_delta": [
                    None if not np.isfinite(value) else float(value)
                    for value in minus_delta
                ],
                "plus_delta": [
                    None if not np.isfinite(value) else float(value)
                    for value in plus_delta
                ],
                "visualization_axis_selection_method": axis_selection_method,
                "selected_two_axes": [
                    {
                        "flat_index": first_axis,
                        "label": GAIN_LABELS[first_axis],
                        "d_delta_d_q": (
                            None
                            if not np.isfinite(derivatives[first_axis])
                            else float(derivatives[first_axis])
                        ),
                    },
                    {
                        "flat_index": second_axis,
                        "label": GAIN_LABELS[second_axis],
                        "d_delta_d_q": (
                            None
                            if not np.isfinite(derivatives[second_axis])
                            else float(derivatives[second_axis])
                        ),
                    },
                ],
            },
            "exact_2d_slice_through_best_12d_point": {
                "grid_size": int(arguments.grid_size),
                "first_axis": GAIN_LABELS[first_axis],
                "second_axis": GAIN_LABELS[second_axis],
                "other_ten_gains": (
                    "fixed at the best point found by the full 12-D scan"
                ),
                "finite_grid_point_count": int(
                    np.count_nonzero(finite_slice)
                ),
                "unresolved_grid_point_count": int(
                    delta_grid.size - np.count_nonzero(finite_slice)
                ),
                "safe_grid_point_count": int(np.count_nonzero(safe_slice)),
                "minimum_delta": (
                    None
                    if not np.any(finite_slice)
                    else float(np.nanmin(delta_grid))
                ),
                "maximum_delta": (
                    None
                    if not np.any(finite_slice)
                    else float(np.nanmax(delta_grid))
                ),
            },
            "success_comparison_only": (
                None
                if success_row is None or success_q is None
                else {
                    "gain_matrix": (
                        success_q.reshape(4, 3) * GAIN_CAPS
                    ).tolist(),
                    **_row_payload(success_row),
                }
            ),
            "diagnostic_semantics": (
                "The Sobol scan and safe cloud are empirical forward "
                "evaluations over the configured gain box.  They are not a "
                "certified continuous 12-D safe set.  The exact 2-D contour is "
                "a visualization through the best sampled 12-D point, not the "
                "search space itself."
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
            "files": {
                "figure": str(output / "safe_margin_full.png"),
                "npz": str(output / "safe_margin_full.npz"),
            },
        }
        write_json(output / "safe_margin_full.json", payload)
        return payload
    finally:
        evaluator.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-json", required=True)
    parser.add_argument("--success-json")
    parser.add_argument(
        "--covariance",
        choices=COVARIANCE_NAMES,
        default="conservative_fusion",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_PLANT_SAMPLE_COUNT,
        help="fixed failure-bag plant sample count",
    )
    parser.add_argument(
        "--gain-samples",
        type=int,
        default=DEFAULT_GAIN_SAMPLE_COUNT,
        help="full-12-D Sobol PID gain sample count",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gain-seed", type=int, default=DEFAULT_GAIN_SEED)
    parser.add_argument(
        "--probe-step",
        type=float,
        default=DEFAULT_PROBE_STEP,
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
    )
    parser.add_argument(
        "--top-count",
        type=int,
        default=DEFAULT_TOP_COUNT,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if int(arguments.samples) <= 0:
        raise ValueError("--samples must be positive")
    if int(arguments.gain_samples) <= 0:
        raise ValueError("--gain-samples must be positive")
    if not (0.0 < float(arguments.probe_step) <= 0.5):
        raise ValueError("--probe-step must lie in (0, 0.5]")
    if int(arguments.grid_size) < 3:
        raise ValueError("--grid-size must be at least three")
    if int(arguments.top_count) <= 0:
        raise ValueError("--top-count must be positive")
    print(json.dumps(analyze(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
