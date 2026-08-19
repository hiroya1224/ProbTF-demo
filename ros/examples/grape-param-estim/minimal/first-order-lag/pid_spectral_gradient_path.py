#!/usr/bin/env python3
"""Gradient-guided forward path search using per-plant spectral-radius sensitivities.

At a PID gain vector q in the normalized 12-D gain box, this script keeps the
full paired plant response

    rho_i(q),  i = 1,...,N,

and estimates the N-by-12 Jacobian d rho_i / d q_j by finite differences.
The 24 +/- gain perturbations are evaluated as one multi-gain worker wave.

A soft maximum of the spectral radii is used only to construct a smooth search
margin and its gradient:

    R_tau(q) = tau log(mean(exp(rho_i(q)/tau))),
    delta_tau(q) = 1 - R_tau(q),
    grad delta_tau = - sum_i w_i grad rho_i,

where w_i are the softmax weights.  The underlying rho_i values, their sorted
order statistics, and the full per-plant Jacobian are retained in the output.

The path starts from a previously found safe gain (by default the best point
in pid_safe_margin_full/safe_margin_full.json).  At each accepted path point:

  1. build the finite-difference per-plant spectral-radius Jacobian,
  2. predict a step toward the recorded failure gains,
  3. if the linearized soft margin would cross zero, minimally correct that
     step along grad delta_tau,
  4. evaluate the predicted point exactly,
  5. if it is not soft-safe or does not move toward failure, halve the step and
     retry using the same local Jacobian.

There is no cloud of dozens of candidates at every path step and no FORM solve.
The only acceptance test is the exact forward pole evaluation.
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
from scipy.special import logsumexp
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
from pid_safe_margin_full import _sobol_gain_coordinates  # noqa: E402
from pid_safe_margin_slices import (  # noqa: E402
    GAIN_CAPS,
    GAIN_LABELS,
    ForwardMarginEvaluator,
    MarginEvaluation,
    _controller_configuration,
    _recorded_gain_matrix,
    _safe_exact_matrix_chunk_task,
    _sobol_quotient_coordinates,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


SCHEMA = "grape-param-estim/pid-spectral-gradient-path/v1"
DEFAULT_PLANT_SAMPLES = 128
DEFAULT_SEED_SCAN_SAMPLES = 256
DEFAULT_SEED = 0
DEFAULT_GAIN_SEED = 1
DEFAULT_TAU = 1.0e-4
DEFAULT_FD_STEP = 0.01
DEFAULT_PATH_STEP = 0.15
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


def _keyed_safe_chunk_task(task: tuple[int, tuple[Any, ...]]) -> tuple[int, tuple[Any, ...]]:
    key, payload = task
    return int(key), _safe_exact_matrix_chunk_task(payload)


class BatchedForwardMarginEvaluator(ForwardMarginEvaluator):
    """Forward evaluator that can submit many gain points in one worker wave."""

    def _assemble_rows(
        self,
        selected_q: np.ndarray,
        rows: Sequence[tuple[Any, ...]],
        wall_seconds: float,
    ) -> MarginEvaluation:
        sample_count = len(self.plants)
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
            raise RuntimeError("plant sample ordering changed during batched evaluation")

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

        gains = selected_q.reshape(4, 3) * GAIN_CAPS
        return MarginEvaluation(
            q=selected_q.copy(),
            gains=gains.copy(),
            delta=delta,
            worst_spectral_radius=worst_radius,
            worst_sample_index=worst,
            pole_valid_mask=pole_valid,
            near_kink_mask=near_kink,
            spectral_radius=radii,
            trim_vectors=trims,
            wall_seconds=float(wall_seconds),
        )

    def evaluate_many(
        self,
        q_values: Sequence[Sequence[float]],
        *,
        initial_trims: Optional[np.ndarray] = None,
        progress: bool = True,
        description: str = "Batched gain wave",
    ) -> list[MarginEvaluation]:
        q_matrix = np.asarray(q_values, dtype=float)
        if q_matrix.ndim != 2 or q_matrix.shape[1] != 12:
            raise ValueError("batched gain coordinates must be M-by-12")

        results: list[Optional[MarginEvaluation]] = [None] * q_matrix.shape[0]
        uncached: list[tuple[int, tuple[float, ...], np.ndarray]] = []
        for row_index, q in enumerate(q_matrix):
            key = self._key(q)
            cached = self._cache.get(key)
            if cached is not None:
                results[row_index] = cached
            else:
                uncached.append((row_index, key, np.asarray(key, dtype=float)))

        if uncached:
            sample_count = len(self.plants)
            chunk_count = min(sample_count, max(1, 2 * self.workers))
            chunks = [
                chunk
                for chunk in np.array_split(np.arange(sample_count, dtype=int), chunk_count)
                if chunk.size
            ]

            warm = None
            if initial_trims is not None:
                warm = np.asarray(initial_trims, dtype=float)
                if warm.shape != (sample_count, 10):
                    raise ValueError("initial_trims must have shape (sample_count, 10)")

            tasks: list[tuple[int, tuple[Any, ...]]] = []
            for row_index, _key, q in uncached:
                configuration = _controller_configuration(q.reshape(4, 3) * GAIN_CAPS)
                for chunk in chunks:
                    tasks.append(
                        (
                            row_index,
                            (
                                tuple(int(i) for i in chunk),
                                tuple(self.plants[int(i)] for i in chunk),
                                self.vehicle_model,
                                self.actuator_parameters,
                                self.controller_dt,
                                configuration,
                                None if warm is None else warm[chunk],
                                None,
                            ),
                        )
                    )

            started = time.perf_counter()
            if self._executor is None:
                iterator = (_keyed_safe_chunk_task(task) for task in tasks)
            else:
                iterator = self._executor.map(_keyed_safe_chunk_task, tasks)

            grouped: dict[int, list[tuple[Any, ...]]] = {
                row_index: [] for row_index, _key, _q in uncached
            }
            bar = tqdm(
                iterator,
                total=len(tasks),
                desc=description,
                unit="chunk",
                dynamic_ncols=True,
                leave=False,
                disable=not progress,
            )
            for row_index, row in bar:
                grouped[int(row_index)].append(row)
            total_wall = time.perf_counter() - started

            per_gain_wall = total_wall / max(1, len(uncached))
            for row_index, key, q in uncached:
                result = self._assemble_rows(q, grouped[row_index], per_gain_wall)
                self._cache[key] = result
                results[row_index] = result

        final = [row for row in results if row is not None]
        if len(final) != q_matrix.shape[0]:
            raise RuntimeError("batched gain evaluation lost a result")
        return final


def _softmax_summary(radii: np.ndarray, tau: float) -> Mapping[str, Any]:
    selected = np.asarray(radii, dtype=float)
    if selected.ndim != 1 or selected.size == 0 or np.any(~np.isfinite(selected)):
        raise ValueError("softmax radii must be one finite vector")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    scaled = selected / float(tau)
    log_normalizer = float(logsumexp(scaled))
    weights = np.exp(scaled - log_normalizer)
    smooth_max = float(tau * (log_normalizer - math.log(float(selected.size))))
    hard_max = float(np.max(selected))
    order = np.argsort(selected)
    return {
        "smooth_max": smooth_max,
        "delta_soft": float(1.0 - smooth_max),
        "hard_max": hard_max,
        "delta_hard": float(1.0 - hard_max),
        "stable_fraction": float(np.mean(selected < 1.0)),
        "weights": weights,
        "order": order,
        "sorted_radii": selected[order],
        "hard_minus_soft": float(hard_max - smooth_max),
        "theoretical_gap_bound": float(tau * math.log(float(selected.size))),
    }


def _finite_difference_jacobian(
    evaluator: BatchedForwardMarginEvaluator,
    q: np.ndarray,
    baseline: MarginEvaluation,
    fd_step: float,
    *,
    progress: bool,
) -> Mapping[str, Any]:
    if not np.all(baseline.pole_valid_mask):
        raise RuntimeError("finite-difference center has unresolved plant poles")

    q = np.asarray(q, dtype=float)
    if q.shape != (12,) or np.any(~np.isfinite(q)):
        raise ValueError("finite-difference center must be finite length 12")
    if not np.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError("finite-difference step must be finite and positive")

    baseline_rho = np.asarray(baseline.spectral_radius, dtype=float)
    sample_count = baseline_rho.size
    jacobian = np.full((sample_count, 12), np.nan, dtype=float)
    # 1 = central, 2 = baseline-to-plus, 3 = minus-to-baseline.
    difference_scheme = np.zeros((sample_count, 12), dtype=np.uint8)

    # Use symmetric points whenever q is in the box interior.  Close to a box
    # face, shorten both sides to the available symmetric displacement rather
    # than taking an off-center secant.  At the face itself, use the available
    # one-sided perturbation.
    minus_points: list[Optional[np.ndarray]] = [None] * 12
    plus_points: list[Optional[np.ndarray]] = [None] * 12
    for axis in range(12):
        lower_room = float(q[axis])
        upper_room = float(1.0 - q[axis])
        if lower_room > 0.0 and upper_room > 0.0:
            displacement = min(float(fd_step), lower_room, upper_room)
            minus_points[axis] = q.copy()
            plus_points[axis] = q.copy()
            minus_points[axis][axis] = q[axis] - displacement
            plus_points[axis][axis] = q[axis] + displacement
        elif lower_room > 0.0:
            minus_points[axis] = q.copy()
            minus_points[axis][axis] = max(0.0, q[axis] - fd_step)
        elif upper_room > 0.0:
            plus_points[axis] = q.copy()
            plus_points[axis][axis] = min(1.0, q[axis] + fd_step)

    minus_rows: list[Optional[MarginEvaluation]] = [None] * 12
    plus_rows: list[Optional[MarginEvaluation]] = [None] * 12
    refinement_count = np.zeros(12, dtype=int)

    while True:
        requests: list[np.ndarray] = []
        metadata: list[tuple[int, str]] = []
        for axis in range(12):
            if np.all(np.isfinite(jacobian[:, axis])):
                continue
            if minus_points[axis] is not None:
                requests.append(minus_points[axis])
                metadata.append((axis, "minus"))
            if plus_points[axis] is not None:
                requests.append(plus_points[axis])
                metadata.append((axis, "plus"))

        if not requests:
            unresolved = {
                GAIN_LABELS[axis]: np.flatnonzero(
                    ~np.isfinite(jacobian[:, axis])
                ).tolist()
                for axis in range(12)
                if np.any(~np.isfinite(jacobian[:, axis]))
            }
            raise RuntimeError(
                "no distinct floating-point gain perturbation resolves the "
                "finite difference for plant indices: "
                + json.dumps(unresolved, sort_keys=True)
            )

        rows = evaluator.evaluate_many(
            requests,
            initial_trims=baseline.trim_vectors,
            progress=progress,
            description=(
                "24-point rho finite differences"
                if not np.any(refinement_count)
                else "Refined rho finite differences"
            ),
        )
        for (axis, side), row in zip(metadata, rows):
            if side == "minus":
                minus_rows[axis] = row
            else:
                plus_rows[axis] = row

        for axis in range(12):
            unresolved = ~np.isfinite(jacobian[:, axis])
            if not np.any(unresolved):
                continue
            minus = minus_rows[axis]
            plus = plus_rows[axis]
            minus_valid = (
                np.zeros(sample_count, dtype=bool)
                if minus is None else np.asarray(minus.pole_valid_mask, dtype=bool)
            )
            plus_valid = (
                np.zeros(sample_count, dtype=bool)
                if plus is None else np.asarray(plus.pole_valid_mask, dtype=bool)
            )

            central = unresolved & minus_valid & plus_valid
            if np.any(central):
                width = float(
                    plus_points[axis][axis] - minus_points[axis][axis]
                )
                jacobian[central, axis] = (
                    plus.spectral_radius[central]
                    - minus.spectral_radius[central]
                ) / width
                difference_scheme[central, axis] = 1

            unresolved = ~np.isfinite(jacobian[:, axis])
            forward = unresolved & plus_valid
            if np.any(forward):
                width = float(plus_points[axis][axis] - q[axis])
                jacobian[forward, axis] = (
                    plus.spectral_radius[forward] - baseline_rho[forward]
                ) / width
                difference_scheme[forward, axis] = 2

            unresolved = ~np.isfinite(jacobian[:, axis])
            backward = unresolved & minus_valid
            if np.any(backward):
                width = float(q[axis] - minus_points[axis][axis])
                jacobian[backward, axis] = (
                    baseline_rho[backward] - minus.spectral_radius[backward]
                ) / width
                difference_scheme[backward, axis] = 3

        if np.all(np.isfinite(jacobian)):
            break

        # Only plants for which neither side resolved reach this point.  Move
        # both available perturbations geometrically toward the already
        # resolved baseline.  Termination is defined by floating-point
        # identity, not by an arbitrary retry count.
        for axis in range(12):
            if np.all(np.isfinite(jacobian[:, axis])):
                continue
            refinement_count[axis] += 1
            for collection in (minus_points, plus_points):
                point = collection[axis]
                if point is None:
                    continue
                refined = point.copy()
                refined[axis] = q[axis] + 0.5 * (point[axis] - q[axis])
                if refined[axis] == q[axis]:
                    collection[axis] = None
                else:
                    collection[axis] = refined
            minus_rows[axis] = None
            plus_rows[axis] = None

    coverage = np.mean(np.isfinite(jacobian), axis=0)

    return {
        "jacobian": jacobian,
        "coverage": coverage,
        "minus_q": tuple(
            None if point is None else point.copy()
            for point in minus_points
        ),
        "plus_q": tuple(
            None if point is None else point.copy()
            for point in plus_points
        ),
        "minus_rows": tuple(minus_rows),
        "plus_rows": tuple(plus_rows),
        "refinement_count": refinement_count,
        "difference_scheme": difference_scheme,
    }


def _soft_gradient(
    rho_jacobian: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = np.asarray(rho_jacobian, dtype=float)
    weights = np.asarray(weights, dtype=float)
    gradient = np.full(jacobian.shape[1], np.nan, dtype=float)
    effective_weight = np.zeros(jacobian.shape[1], dtype=float)
    for axis in range(jacobian.shape[1]):
        valid = np.isfinite(jacobian[:, axis])
        mass = float(np.sum(weights[valid]))
        effective_weight[axis] = mass
        if mass > 0.0:
            gradient[axis] = -float(
                np.sum(weights[valid] * jacobian[valid, axis]) / mass
            )
    return gradient, effective_weight


def _predict(
    q: np.ndarray,
    failure_q: np.ndarray,
    delta_soft: float,
    gradient: np.ndarray,
    step: float,
) -> np.ndarray:
    vector = failure_q - q
    distance = float(np.linalg.norm(vector))
    if distance == 0.0:
        return q.copy()
    direct = min(step, distance) * vector / distance

    finite = np.isfinite(gradient)
    g = np.zeros(12, dtype=float)
    g[finite] = gradient[finite]
    predicted_margin = float(delta_soft + np.dot(g, direct))
    proposal = direct.copy()
    norm_squared = float(np.dot(g, g))
    if predicted_margin < 0.0 and norm_squared > 0.0:
        proposal += (-predicted_margin / norm_squared) * g
        proposal_norm = float(np.linalg.norm(proposal))
        if proposal_norm > step:
            proposal *= step / proposal_norm

    return np.clip(q + proposal, 0.0, 1.0)


def _seed_candidates_from_artifacts(path: Path) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Load all available previously safe gain candidates, not just one best point."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[np.ndarray] = []
    sources: list[str] = []

    best = np.asarray(
        payload["full_12d_scan"]["best_evaluated"]["q"],
        dtype=float,
    )
    if best.shape == (12,) and np.all(np.isfinite(best)):
        candidates.append(best.copy())
        sources.append("json:best_evaluated")

    for row in payload["full_12d_scan"].get("top_evaluated_points", []):
        q = np.asarray(row.get("q"), dtype=float)
        if q.shape == (12,) and np.all(np.isfinite(q)):
            candidates.append(q.copy())
            sources.append("json:top_evaluated_points")

    npz_candidates = []
    configured_npz = payload.get("files", {}).get("npz")
    if configured_npz:
        npz_candidates.append(Path(configured_npz).expanduser())
    npz_candidates.append(path.with_name("safe_margin_full.npz"))

    used_npz = None
    for npz_path in npz_candidates:
        if not npz_path.exists():
            continue
        with np.load(npz_path) as archive:
            if "safe_scan_q" not in archive:
                continue
            safe = np.asarray(archive["safe_scan_q"], dtype=float)
        if safe.ndim == 2 and safe.shape[1] == 12 and np.all(np.isfinite(safe)):
            for q in safe:
                candidates.append(q.copy())
                sources.append("npz:safe_scan_q")
            used_npz = str(npz_path.resolve())
            break

    if not candidates:
        raise ValueError("seed artifacts contain no finite 12-D gain candidates")

    unique: list[np.ndarray] = []
    unique_sources: list[str] = []
    seen: set[tuple[float, ...]] = set()
    for q, source in zip(candidates, sources):
        key = tuple(float(value) for value in np.round(q, decimals=14))
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
        unique_sources.append(source)

    return np.asarray(unique, dtype=float), {
        "artifact_json": str(path),
        "artifact_npz": used_npz,
        "loaded_candidate_count": len(candidates),
        "unique_candidate_count": len(unique),
        "candidate_sources": unique_sources,
    }


def _select_revalidated_seed(
    evaluator: BatchedForwardMarginEvaluator,
    candidates: np.ndarray,
    failure_q: np.ndarray,
    failure_trims: np.ndarray,
    tau: float,
    *,
    progress: bool,
) -> tuple[Optional[np.ndarray], Mapping[str, Any]]:
    """Re-evaluate stored candidates under the current evaluator before reuse."""

    rows = evaluator.evaluate_many(
        candidates,
        initial_trims=failure_trims,
        progress=progress,
        description="Revalidate stored safe seeds",
    )

    best = None
    resolved_count = 0
    soft_safe_count = 0
    for index, (q, row) in enumerate(zip(candidates, rows)):
        if not np.all(row.pole_valid_mask):
            continue
        resolved_count += 1
        summary = _softmax_summary(row.spectral_radius, tau)
        if summary["delta_soft"] <= 0.0:
            continue
        soft_safe_count += 1
        distance = float(np.linalg.norm(q - failure_q))
        candidate = (
            float(summary["delta_soft"]),
            -distance,
            int(index),
            q.copy(),
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    diagnostics = {
        "candidate_count": int(candidates.shape[0]),
        "resolved_candidate_count": int(resolved_count),
        "soft_safe_candidate_count": int(soft_safe_count),
        "selected_candidate_index": None if best is None else int(best[2]),
    }
    return (None if best is None else best[3]), diagnostics


def _find_seed_fallback(
    evaluator: BatchedForwardMarginEvaluator,
    failure_q: np.ndarray,
    tau: float,
    sample_count: int,
    gain_seed: int,
    *,
    initial_trims: np.ndarray,
    progress: bool,
) -> np.ndarray:
    design, _info = _sobol_gain_coordinates(sample_count, gain_seed)
    rows = evaluator.evaluate_many(
        design,
        initial_trims=initial_trims,
        progress=progress,
        description="Fallback full-12-D seed scan",
    )
    best = None
    for q, row in zip(design, rows):
        if not np.all(row.pole_valid_mask):
            continue
        summary = _softmax_summary(row.spectral_radius, tau)
        if summary["delta_soft"] <= 0.0:
            continue
        distance = float(np.linalg.norm(q - failure_q))
        candidate = (summary["delta_soft"], -distance, q.copy())
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("fallback gain scan found no soft-safe seed")
    return best[2]


def _point_payload(q: np.ndarray, summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "q": np.asarray(q, dtype=float).tolist(),
        "gain_matrix": (np.asarray(q).reshape(4, 3) * GAIN_CAPS).tolist(),
        "delta_soft": float(summary["delta_soft"]),
        "delta_hard": float(summary["delta_hard"]),
        "smooth_max": float(summary["smooth_max"]),
        "hard_max": float(summary["hard_max"]),
        "stable_fraction": float(summary["stable_fraction"]),
        "hard_minus_soft": float(summary["hard_minus_soft"]),
    }


def _plot(
    output: Path,
    failure_q: np.ndarray,
    path_q: np.ndarray,
    path_soft: np.ndarray,
    path_hard: np.ndarray,
    path_distance: np.ndarray,
    path_sorted_rho: np.ndarray,
    failure_sorted_rho: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)

    axis = axes[0]
    axis.plot(path_distance, path_soft, marker="o", label="soft margin")
    axis.plot(path_distance, path_hard, marker="x", label="hard-max margin")
    axis.axhline(0.0)
    axis.set_xlabel("normalized 12-D distance to failure")
    axis.set_ylabel("margin")
    axis.set_title("Gradient-guided continuation")
    axis.legend()
    axis.grid(True, alpha=0.25)

    axis = axes[1]
    u = (np.arange(path_sorted_rho.shape[1]) + 1) / path_sorted_rho.shape[1]
    selected_indices = sorted(set((0, path_sorted_rho.shape[0] // 2, path_sorted_rho.shape[0] - 1)))
    for index in selected_indices:
        axis.plot(u, path_sorted_rho[index], label=f"path {index}")
    axis.plot(u, failure_sorted_rho, linestyle="--", label="failure")
    axis.axhline(1.0)
    axis.set_xlabel("empirical quantile")
    axis.set_ylabel("spectral radius")
    axis.set_title("Order-statistic curves retained along path")
    axis.legend()
    axis.grid(True, alpha=0.25)

    axis = axes[2]
    x = np.arange(12)
    axis.plot(x, failure_q, marker="o", label="failure")
    axis.plot(x, path_q[0], marker="s", label="seed")
    axis.plot(x, path_q[-1], marker="^", label="final")
    axis.set_xticks(x)
    axis.set_xticklabels(GAIN_LABELS, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("normalized gain q")
    axis.set_title("PID movement")
    axis.legend()
    axis.grid(True, alpha=0.25)

    figure.savefig(output / "spectral_gradient_path.png", dpi=180)
    plt.close(figure)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    progress = not bool(arguments.no_progress)
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)

    failure_gains = _recorded_gain_matrix(estimate)
    caps = GAIN_CAPS.reshape(-1)
    failure_q = failure_gains.reshape(-1) / caps
    if np.any(failure_q < 0.0) or np.any(failure_q > 1.0):
        raise ValueError("recorded failure gains lie outside the exploration caps")

    vehicle_model = load_vehicle_model(Path(estimate["input"]["vehicle_model"]))
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
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

    evaluator = BatchedForwardMarginEvaluator(
        plants=plants,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        workers=int(arguments.workers),
    )
    try:
        # Establish one resolved trim bundle first.  Stored safe points from
        # the previous scan may have depended on continuation, so they are
        # never trusted without re-evaluation under this evaluator.
        failure_seed_row = evaluator.evaluate(failure_q)
        if not np.all(failure_seed_row.pole_valid_mask):
            raise RuntimeError(
                "recorded failure gain itself has unresolved plant poles; "
                "there is no common resolved trim bundle for seed revalidation"
            )

        seed_path = (
            Path(arguments.seed_json).expanduser().resolve()
            if arguments.seed_json is not None
            else estimate_path.parent / "pid_safe_margin_full" / "safe_margin_full.json"
        )
        seed_revalidation: Mapping[str, Any] = {
            "artifact_used": False,
        }
        current_q = None

        if seed_path.exists():
            stored_candidates, artifact_info = _seed_candidates_from_artifacts(seed_path)
            current_q, validation_info = _select_revalidated_seed(
                evaluator,
                stored_candidates,
                failure_q,
                failure_seed_row.trim_vectors,
                float(arguments.tau),
                progress=progress,
            )
            seed_revalidation = {
                "artifact_used": True,
                **dict(artifact_info),
                **dict(validation_info),
            }

        if current_q is None:
            current_q = _find_seed_fallback(
                evaluator,
                failure_q,
                float(arguments.tau),
                int(arguments.seed_scan_samples),
                int(arguments.gain_seed),
                initial_trims=failure_seed_row.trim_vectors,
                progress=progress,
            )
            seed_source = "fallback Sobol scan revalidated with failure trims"
        else:
            seed_source = str(seed_path)

        seed_row = evaluator.evaluate(current_q)
        if not np.all(seed_row.pole_valid_mask):
            raise RuntimeError(
                "internal error: revalidated seed lost its resolved pole evaluation"
            )
        seed_summary = _softmax_summary(seed_row.spectral_radius, float(arguments.tau))
        if seed_summary["delta_soft"] <= 0.0:
            raise RuntimeError(
                "internal error: revalidated seed lost its positive soft margin"
            )

        path_q: list[np.ndarray] = []
        path_rho: list[np.ndarray] = []
        path_sorted: list[np.ndarray] = []
        path_order: list[np.ndarray] = []
        path_jacobian: list[np.ndarray] = []
        path_weights: list[np.ndarray] = []
        path_gradient: list[np.ndarray] = []
        path_gradient_weight: list[np.ndarray] = []
        path_soft: list[float] = []
        path_hard: list[float] = []
        path_stable_fraction: list[float] = []
        path_distance: list[float] = []
        path_step: list[float] = []
        path_corrections: list[int] = []

        step = float(arguments.path_step)
        path_bar = tqdm(
            desc="Spectral-gradient continuation",
            unit="accepted-step",
            dynamic_ncols=True,
            disable=not progress,
        )
        try:
            while True:
                baseline = evaluator.evaluate(current_q)
                if not np.all(baseline.pole_valid_mask):
                    raise RuntimeError("accepted path point has unresolved plant poles")
                summary = _softmax_summary(baseline.spectral_radius, float(arguments.tau))
                fd = _finite_difference_jacobian(
                    evaluator,
                    current_q,
                    baseline,
                    float(arguments.fd_step),
                    progress=progress,
                )
                gradient, gradient_weight = _soft_gradient(
                    fd["jacobian"], summary["weights"]
                )

                path_q.append(current_q.copy())
                path_rho.append(np.asarray(baseline.spectral_radius, dtype=float).copy())
                path_sorted.append(np.asarray(summary["sorted_radii"], dtype=float).copy())
                path_order.append(np.asarray(summary["order"], dtype=int).copy())
                path_jacobian.append(np.asarray(fd["jacobian"], dtype=float).copy())
                path_weights.append(np.asarray(summary["weights"], dtype=float).copy())
                path_gradient.append(gradient.copy())
                path_gradient_weight.append(gradient_weight.copy())
                path_soft.append(float(summary["delta_soft"]))
                path_hard.append(float(summary["delta_hard"]))
                path_stable_fraction.append(float(summary["stable_fraction"]))
                distance = float(np.linalg.norm(current_q - failure_q))
                path_distance.append(distance)
                path_step.append(step)

                if distance == 0.0:
                    path_corrections.append(0)
                    break

                trial_step = min(step, distance)
                corrections = 0
                accepted = False
                while True:
                    candidate_q = _predict(
                        current_q,
                        failure_q,
                        float(summary["delta_soft"]),
                        gradient,
                        trial_step,
                    )
                    if np.array_equal(candidate_q, current_q):
                        break
                    candidate_row = evaluator.evaluate(candidate_q)
                    if np.all(candidate_row.pole_valid_mask):
                        candidate_summary = _softmax_summary(
                            candidate_row.spectral_radius, float(arguments.tau)
                        )
                        candidate_distance = float(np.linalg.norm(candidate_q - failure_q))
                        if (
                            candidate_summary["delta_soft"] > 0.0
                            and candidate_distance < distance
                        ):
                            current_q = candidate_q
                            accepted = True
                            break
                    trial_step *= 0.5
                    corrections += 1

                path_corrections.append(corrections)
                if not accepted:
                    break

                path_bar.update(1)
                if progress:
                    path_bar.set_postfix(
                        distance=f"{np.linalg.norm(current_q-failure_q):.4f}",
                        delta=f"{candidate_summary['delta_soft']:.3e}",
                        retries=corrections,
                    )
        finally:
            path_bar.close()

        failure_row = evaluator.evaluate(failure_q)
        failure_summary = None
        if np.all(failure_row.pole_valid_mask):
            failure_summary = _softmax_summary(
                failure_row.spectral_radius, float(arguments.tau)
            )

        path_q_array = np.asarray(path_q, dtype=float)
        path_rho_array = np.asarray(path_rho, dtype=float)
        path_sorted_array = np.asarray(path_sorted, dtype=float)
        path_order_array = np.asarray(path_order, dtype=int)
        path_jacobian_array = np.asarray(path_jacobian, dtype=float)
        path_weights_array = np.asarray(path_weights, dtype=float)
        path_gradient_array = np.asarray(path_gradient, dtype=float)
        path_gradient_weight_array = np.asarray(path_gradient_weight, dtype=float)

        output = (
            Path(arguments.output_dir).expanduser().resolve()
            if arguments.output_dir is not None
            else estimate_path.parent / "pid_spectral_gradient_path"
        )
        output.mkdir(parents=True, exist_ok=True)

        failure_sorted = (
            np.sort(failure_row.spectral_radius)
            if np.all(failure_row.pole_valid_mask)
            else np.full(len(plants), np.nan)
        )
        _plot(
            output,
            failure_q,
            path_q_array,
            np.asarray(path_soft),
            np.asarray(path_hard),
            np.asarray(path_distance),
            path_sorted_array,
            failure_sorted,
        )

        np.savez_compressed(
            output / "spectral_gradient_path.npz",
            quotient_coordinates=quotient,
            failure_q=failure_q,
            gain_caps=GAIN_CAPS,
            path_q=path_q_array,
            path_rho=path_rho_array,
            path_sorted_rho=path_sorted_array,
            path_rho_order=path_order_array,
            path_rho_jacobian=path_jacobian_array,
            path_softmax_weights=path_weights_array,
            path_soft_gradient=path_gradient_array,
            path_soft_gradient_weight_coverage=path_gradient_weight_array,
            path_delta_soft=np.asarray(path_soft),
            path_delta_hard=np.asarray(path_hard),
            path_stable_fraction=np.asarray(path_stable_fraction),
            path_distance_to_failure=np.asarray(path_distance),
            path_nominal_step=np.asarray(path_step),
            path_exact_correction_halvings=np.asarray(path_corrections, dtype=int),
            failure_rho=np.asarray(failure_row.spectral_radius, dtype=float),
        )

        final_summary = _softmax_summary(path_rho_array[-1], float(arguments.tau))
        payload = {
            "schema": SCHEMA,
            "source_commit": source_commit(_PROJECT_ROOT),
            "case_name": str(estimate["case_name"]),
            "estimate_json": str(estimate_path),
            "plant_sampling": {
                **dict(plant_sampling),
                "sample_count": int(arguments.samples),
                "seed": int(arguments.seed),
            },
            "search": {
                "method": "finite-difference per-plant rho Jacobian + softmax-gradient predictor + exact forward corrector",
                "tau": float(arguments.tau),
                "softmax_hard_gap_bound": float(arguments.tau) * math.log(float(arguments.samples)),
                "fd_step_in_normalized_gain": float(arguments.fd_step),
                "nominal_path_step_in_normalized_gain": float(arguments.path_step),
                "seed_source": seed_source,
                "seed_revalidation": dict(seed_revalidation),
                "accepted_path_point_count": int(path_q_array.shape[0]),
                "total_exact_correction_halvings": int(np.sum(path_corrections)),
                "artificial_max_path_step_count": None,
                "stop_condition": "failure reached or exact-corrector step became numerically identical to current point after repeated halving",
            },
            "seed": _point_payload(path_q_array[0], _softmax_summary(path_rho_array[0], float(arguments.tau))),
            "final": {
                **_point_payload(path_q_array[-1], final_summary),
                "distance_to_failure": float(path_distance[-1]),
            },
            "failure": (
                None
                if failure_summary is None
                else _point_payload(failure_q, failure_summary)
            ),
            "retained_information": {
                "paired_rho": "rho_i(K) for the same ordered plant samples at every path point",
                "order_statistics": "sorted rho_(1)(K),...,rho_(N)(K) at every path point",
                "rho_gain_jacobian": "finite-difference d rho_i / d q_j, shape path x plant x 12",
                "softmax_weights": "weights over plant samples used only to aggregate the retained rho Jacobian into the search gradient",
            },
            "elapsed_seconds": float(time.perf_counter() - started),
            "files": {
                "figure": str(output / "spectral_gradient_path.png"),
                "npz": str(output / "spectral_gradient_path.npz"),
            },
        }
        write_json(output / "spectral_gradient_path.json", payload)
        return payload
    finally:
        evaluator.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-json", required=True)
    parser.add_argument("--seed-json")
    parser.add_argument("--covariance", choices=COVARIANCE_NAMES, default="conservative_fusion")
    parser.add_argument("--samples", type=int, default=DEFAULT_PLANT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gain-seed", type=int, default=DEFAULT_GAIN_SEED)
    parser.add_argument("--seed-scan-samples", type=int, default=DEFAULT_SEED_SCAN_SAMPLES)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--path-step", type=float, default=DEFAULT_PATH_STEP)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.samples <= 1:
        raise ValueError("--samples must be greater than one")
    if arguments.seed_scan_samples <= 0:
        raise ValueError("--seed-scan-samples must be positive")
    if not np.isfinite(arguments.tau) or arguments.tau <= 0.0:
        raise ValueError("--tau must be finite and positive")
    if not np.isfinite(arguments.fd_step) or arguments.fd_step <= 0.0:
        raise ValueError("--fd-step must be finite and positive")
    if not np.isfinite(arguments.path_step) or arguments.path_step <= 0.0:
        raise ValueError("--path-step must be finite and positive")
    print(json.dumps(analyze(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
