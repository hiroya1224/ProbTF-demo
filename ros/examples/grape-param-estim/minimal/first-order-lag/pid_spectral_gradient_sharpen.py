#!/usr/bin/env python3
"""Failure-started spectral-gradient continuation with automatic softmax sharpening.

The starting point is exactly the PID gain recorded in the failed bag.  No
previous safe-gain artifact, seed JSON, or saved safe cloud is read.  One
scrambled-Sobol sequence is mapped through the failure-bag quotient-space
Gaussian posterior and fixed for the entire run.  The search refines nested
prefixes of that sequence (16, 32, 64, ..., N_max); it never resamples plants.

For the fixed ordered plant samples theta_i and normalized gain q,

    rho_i(q) = spectral radius of the exact sampled closed-loop Jacobian,

and the full N-by-12 finite-difference Jacobian d rho_i / d q_j is retained.
For tau > 0,

    R_tau(q) = tau log(mean(exp(rho_i(q)/tau))),
    delta_tau(q) = 1 - R_tau(q),

and

    grad delta_tau = -sum_i w_i grad rho_i.

At each temperature:
  1. choose eta_tau = hard_margin_target + tau log(N),
  2. project normally onto delta_tau = eta_tau,
  3. move toward the failed gain in the tangent space of that level set,
  4. exact-forward correct back to the target level,
  5. repeat until the tangential displacement toward failure is below the
     finite-difference coordinate resolution.

Since delta_hard >= delta_tau - tau log(N), this soft target guarantees the
requested sampled hard margin at a resolved target point.

Then tau is halved.  The same rho_i and d rho_i/dq_j at the stage endpoint
immediately give the new softmax weights and the first sharpen correction.
Sharpening stops when

    tau log(N) / ||grad delta_tau|| <= fd_step,

so the worst-case softmax gap, converted to a local gain displacement, is no
larger than the coordinate scale used for the finite differences.

All acceptance/correction decisions use exact forward pole evaluations.
The linearized per-plant rho model is used only for predictors and for the
cheap 4 x 3 group contour diagnostic figure.

When a plant prefix grows, the same gain is first revalidated at the larger
Gaussian-QMC resolution.  Cached leading-plant evaluations are retained and
only the newly added suffix is evaluated.  All reported final values use an
exact forward evaluation over the full N_max prefix.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
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
from pid_safe_margin_slices import (  # noqa: E402
    GAIN_CAPS,
    GAIN_LABELS,
    _recorded_gain_matrix,
    _sobol_quotient_coordinates,
)
from pid_spectral_gradient_path import (  # noqa: E402
    BatchedForwardMarginEvaluator,
    _finite_difference_jacobian,
    _soft_gradient,
)
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


SCHEMA = "grape-param-estim/pid-spectral-gradient-sharpen/v4"
DEFAULT_PLANT_SAMPLES = 128
DEFAULT_INITIAL_PLANT_PREFIX = 16
DEFAULT_SEED = 0
DEFAULT_TAU = 1.0e-4
DEFAULT_FD_STEP = 0.01
DEFAULT_PATH_STEP = 0.15
DEFAULT_CONTOUR_GRID = 81
DEFAULT_HARD_MARGIN_TARGET = 5.0e-5
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)

GROUPS = ("xy", "z", "roll_pitch", "yaw")
PROJECTIONS = (
    ("PI", 0, 1),
    ("ID", 1, 2),
    ("DP", 2, 0),
)


def _nested_prefix_counts(
    maximum_sample_count: int,
    initial_sample_count: int = DEFAULT_INITIAL_PLANT_PREFIX,
) -> tuple[int, ...]:
    maximum = int(maximum_sample_count)
    initial = int(initial_sample_count)
    if maximum <= 0 or initial <= 0:
        raise ValueError("plant sample counts must be positive")
    if maximum <= initial:
        return (maximum,)

    counts: list[int] = []
    selected = initial
    while selected < maximum:
        counts.append(selected)
        selected *= 2
    if not counts or counts[-1] != maximum:
        counts.append(maximum)
    return tuple(counts)


def _pad_plant_prefix_arrays(
    values: Sequence[np.ndarray],
    maximum_sample_count: int,
    *,
    dtype: Any,
    fill_value: Any,
) -> np.ndarray:
    """Stack plant-indexed prefix arrays without inventing suffix values."""

    arrays = [np.asarray(value, dtype=dtype) for value in values]
    if not arrays:
        return np.empty((0, int(maximum_sample_count)), dtype=dtype)
    trailing_shape = arrays[0].shape[1:]
    result = np.full(
        (len(arrays), int(maximum_sample_count), *trailing_shape),
        fill_value,
        dtype=dtype,
    )
    for row_index, value in enumerate(arrays):
        if value.shape[1:] != trailing_shape:
            raise ValueError("plant-prefix arrays have inconsistent shapes")
        if value.shape[0] > int(maximum_sample_count):
            raise ValueError("plant-prefix array exceeds N_max")
        result[row_index, :value.shape[0]] = value
    return result


def _softmax_summary(radii: Sequence[float], tau: float) -> Mapping[str, Any]:
    rho = np.asarray(radii, dtype=float)
    if rho.ndim != 1 or rho.size == 0 or np.any(~np.isfinite(rho)):
        raise ValueError("softmax radii must be one finite vector")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    scaled = rho / float(tau)
    log_z = float(logsumexp(scaled))
    weights = np.exp(scaled - log_z)
    smooth_max = float(
        tau * (log_z - math.log(float(rho.size)))
    )
    hard_max = float(np.max(rho))
    order = np.argsort(rho)
    return {
        "smooth_max": smooth_max,
        "delta_soft": float(1.0 - smooth_max),
        "hard_max": hard_max,
        "delta_hard": float(1.0 - hard_max),
        "stable_fraction": float(np.mean(rho < 1.0)),
        "weights": weights,
        "order": order,
        "sorted_rho": rho[order],
        "hard_minus_soft": float(hard_max - smooth_max),
        "gap_bound": float(tau * math.log(float(rho.size))),
    }


def _soft_target_margin(
    tau: float,
    sample_count: int,
    hard_margin_target: float,
) -> float:
    return float(hard_margin_target) + float(tau) * math.log(float(sample_count))


def _target_residual(
    summary: Mapping[str, Any],
    soft_target_margin: float,
) -> float:
    return float(summary["delta_soft"]) - float(soft_target_margin)


def _target_level_status(
    summary: Mapping[str, Any],
    soft_target_margin: float,
    hard_margin_target: float,
    gradient: np.ndarray,
    fd_step: float,
) -> Mapping[str, Any]:
    residual = _target_residual(summary, soft_target_margin)
    gradient_norm = float(np.linalg.norm(np.asarray(gradient, dtype=float)))
    equivalent_gain_displacement = (
        float("inf")
        if gradient_norm == 0.0
        else abs(residual) / gradient_norm
    )
    hard_target_satisfied = bool(
        float(summary["delta_hard"]) >= float(hard_margin_target)
    )
    within_fd_resolution = bool(
        equivalent_gain_displacement <= float(fd_step)
    )
    return {
        "residual": float(residual),
        "equivalent_gain_displacement": float(
            equivalent_gain_displacement
        ),
        "within_fd_resolution": within_fd_resolution,
        "hard_target_satisfied": hard_target_satisfied,
        "resolved": bool(within_fd_resolution and hard_target_satisfied),
    }


def _point_payload(
    q: np.ndarray,
    summary: Mapping[str, Any],
    tau: float,
    *,
    hard_margin_target: float,
    sample_count: int,
) -> Mapping[str, Any]:
    selected = np.asarray(q, dtype=float)
    soft_target_margin = _soft_target_margin(
        tau,
        sample_count,
        hard_margin_target,
    )
    return {
        "q": selected.tolist(),
        "gain_matrix": (
            selected.reshape(4, 3) * GAIN_CAPS
        ).tolist(),
        "tau": float(tau),
        "hard_margin_target": float(hard_margin_target),
        "soft_target_margin": float(soft_target_margin),
        "target_residual": float(
            _target_residual(summary, soft_target_margin)
        ),
        "delta_soft": float(summary["delta_soft"]),
        "delta_hard": float(summary["delta_hard"]),
        "smooth_max": float(summary["smooth_max"]),
        "hard_max": float(summary["hard_max"]),
        "stable_fraction": float(summary["stable_fraction"]),
        "hard_minus_soft": float(summary["hard_minus_soft"]),
        "gap_bound": float(summary["gap_bound"]),
    }


def _gradient_bundle(
    evaluator: BatchedForwardMarginEvaluator,
    q: np.ndarray,
    tau: float,
    fd_step: float,
    *,
    progress: bool,
) -> Mapping[str, Any]:
    baseline = evaluator.evaluate(q)
    if not np.all(baseline.pole_valid_mask):
        raise RuntimeError(
            "accepted point has unresolved plant poles, so the rho Jacobian "
            "cannot be formed"
        )

    summary = _softmax_summary(
        baseline.spectral_radius,
        tau,
    )
    fd = _finite_difference_jacobian(
        evaluator,
        np.asarray(q, dtype=float),
        baseline,
        float(fd_step),
        progress=progress,
    )
    gradient, weight_coverage = _soft_gradient(
        fd["jacobian"],
        summary["weights"],
    )
    if not np.all(np.isfinite(gradient)):
        diagnostics = []
        jacobian = np.asarray(fd["jacobian"], dtype=float)
        for axis, label in enumerate(GAIN_LABELS):
            minus = fd["minus_rows"][axis]
            plus = fd["plus_rows"][axis]
            derivative = jacobian[:, axis]
            finite = derivative[np.isfinite(derivative)]
            diagnostics.append({
                "axis": int(axis),
                "label": str(label),
                "q": float(q[axis]),
                "minus_q": (
                    float(fd["minus_q"][axis][axis])
                    if fd["minus_q"][axis] is not None else float(q[axis])
                ),
                "plus_q": (
                    float(fd["plus_q"][axis][axis])
                    if fd["plus_q"][axis] is not None else float(q[axis])
                ),
                "minus_resolved": (
                    int(np.count_nonzero(minus.pole_valid_mask))
                    if minus is not None else int(baseline.pole_valid_mask.size)
                ),
                "plus_resolved": (
                    int(np.count_nonzero(plus.pole_valid_mask))
                    if plus is not None else int(baseline.pole_valid_mask.size)
                ),
                "finite_derivatives": int(finite.size),
                "derivative_min_median_max": (
                    None if finite.size == 0 else [
                        float(np.min(finite)),
                        float(np.median(finite)),
                        float(np.max(finite)),
                    ]
                ),
                "softmax_weight_coverage": float(weight_coverage[axis]),
                "minus_unresolved": (
                    [] if minus is None else
                    np.flatnonzero(~minus.pole_valid_mask).tolist()
                ),
                "plus_unresolved": (
                    [] if plus is None else
                    np.flatnonzero(~plus.pole_valid_mask).tolist()
                ),
            })
        raise RuntimeError(
            "soft-margin gradient is not finite in every PID coordinate:\n"
            + json.dumps({
                "baseline_resolved": int(np.count_nonzero(baseline.pole_valid_mask)),
                "baseline_trim_shape": list(baseline.trim_vectors.shape),
                "rho_min_mean_max": [
                    float(np.min(baseline.spectral_radius)),
                    float(np.mean(baseline.spectral_radius)),
                    float(np.max(baseline.spectral_radius)),
                ],
                "delta_soft": float(summary["delta_soft"]),
                "delta_hard": float(summary["delta_hard"]),
                "stable_fraction": float(summary["stable_fraction"]),
                "weights_sum": float(np.sum(summary["weights"])),
                "axes": diagnostics,
            }, indent=2)
        )

    return {
        "row": baseline,
        "summary": summary,
        "rho_jacobian": np.asarray(fd["jacobian"], dtype=float),
        "fd_coverage": np.asarray(fd["coverage"], dtype=float),
        "fd_refinement_count": np.asarray(
            fd["refinement_count"], dtype=int
        ),
        "fd_difference_scheme": np.asarray(
            fd["difference_scheme"], dtype=np.uint8
        ),
        "gradient": np.asarray(gradient, dtype=float),
        "gradient_weight_coverage": np.asarray(
            weight_coverage, dtype=float
        ),
    }


def _normal_newton_step(
    target_residual: float,
    gradient: np.ndarray,
) -> np.ndarray:
    g = np.asarray(gradient, dtype=float)
    norm_sq = float(np.dot(g, g))
    if norm_sq == 0.0:
        raise RuntimeError(
            "soft-margin gradient vanishes, so the local boundary normal is undefined"
        )
    return (-float(target_residual) / norm_sq) * g


def _tangent_direction(
    q: np.ndarray,
    failure_q: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    d = np.asarray(failure_q, dtype=float) - np.asarray(q, dtype=float)
    g = np.asarray(gradient, dtype=float)
    norm_sq = float(np.dot(g, g))
    if norm_sq == 0.0:
        raise RuntimeError(
            "soft-margin gradient vanishes, so the boundary tangent space is undefined"
        )
    return d - (float(np.dot(g, d)) / norm_sq) * g


def _exact_normal_projection(
    evaluator: BatchedForwardMarginEvaluator,
    q: np.ndarray,
    tau: float,
    soft_target_margin: float,
    hard_margin_target: float,
    fd_step: float,
    gradient: np.ndarray,
    *,
    progress: bool,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Correct q toward delta_tau=soft_target_margin along a local normal.

    The normal is held fixed within this corrector.  If the full nonlinear
    boundary bends enough that a fresh normal is needed, the outer continuation
    loop recomputes the N-by-12 rho Jacobian at the accepted point.
    """

    current = np.asarray(q, dtype=float).copy()
    g = np.asarray(gradient, dtype=float)
    norm_sq = float(np.dot(g, g))
    if norm_sq == 0.0:
        raise RuntimeError("cannot project with a zero soft-margin gradient")

    corrections = 0
    backtracks = 0
    while True:
        row = evaluator.evaluate(current)
        if not np.all(row.pole_valid_mask):
            return current, {}, {
                "resolved": False,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }

        summary = _softmax_summary(row.spectral_radius, tau)
        residual = _target_residual(summary, soft_target_margin)
        target_status = _target_level_status(
            summary,
            soft_target_margin,
            hard_margin_target,
            g,
            fd_step,
        )
        if bool(target_status["resolved"]):
            return current, summary, {
                "resolved": True,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
                "target_level_status": dict(target_status),
            }

        correction = (-residual / norm_sq) * g
        candidate = np.clip(current + correction, 0.0, 1.0)
        if np.array_equal(candidate, current):
            return current, summary, {
                "resolved": bool(target_status["resolved"]),
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
                "target_level_status": dict(target_status),
            }

        candidate_row = evaluator.evaluate(candidate)
        if np.all(candidate_row.pole_valid_mask):
            candidate_summary = _softmax_summary(
                candidate_row.spectral_radius,
                tau,
            )
            candidate_abs = abs(
                _target_residual(candidate_summary, soft_target_margin)
            )
            current_abs = abs(residual)
            if candidate_abs < current_abs:
                current = candidate
                corrections += 1
                continue

        # The Newton correction overshot/non-improved.  Backtrack on this
        # mathematically proposed correction until exact |delta| decreases.
        scale = 0.5
        accepted = False
        while True:
            backtracks += 1
            backtracked = np.clip(
                current + scale * correction,
                0.0,
                1.0,
            )
            if np.array_equal(backtracked, current):
                return current, summary, {
                    "resolved": False,
                    "normal_updates": int(corrections),
                    "backtracks": int(backtracks),
                    "target_level_status": dict(target_status),
                }
            backtracked_row = evaluator.evaluate(backtracked)
            if np.all(backtracked_row.pole_valid_mask):
                backtracked_summary = _softmax_summary(
                    backtracked_row.spectral_radius,
                    tau,
                )
                if abs(_target_residual(
                    backtracked_summary,
                    soft_target_margin,
                )) < abs(residual):
                    current = backtracked
                    corrections += 1
                    accepted = True
                    break
            scale *= 0.5

        if not accepted:
            return current, summary, {
                "resolved": False,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
                "target_level_status": dict(target_status),
            }


def _fresh_normal_projection(
    evaluator: BatchedForwardMarginEvaluator,
    q: np.ndarray,
    tau: float,
    soft_target_margin: float,
    hard_margin_target: float,
    fd_step: float,
    *,
    progress: bool,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Project with refreshed plantwise gain gradients until resolved."""

    current = np.asarray(q, dtype=float).copy()
    total_updates = 0
    total_backtracks = 0
    gradient_refreshes = 0
    while True:
        bundle = _gradient_bundle(
            evaluator,
            current,
            tau,
            fd_step,
            progress=progress,
        )
        summary = bundle["summary"]
        gradient = bundle["gradient"]
        target_status = _target_level_status(
            summary,
            soft_target_margin,
            hard_margin_target,
            gradient,
            fd_step,
        )
        if bool(target_status["resolved"]):
            return current, summary, {
                "resolved": True,
                "normal_updates": int(total_updates),
                "backtracks": int(total_backtracks),
                "gradient_refreshes": int(gradient_refreshes),
                "target_level_status": dict(target_status),
            }

        projected, projected_summary, projection = _exact_normal_projection(
            evaluator,
            current,
            tau,
            soft_target_margin,
            hard_margin_target,
            fd_step,
            gradient,
            progress=progress,
        )
        total_updates += int(projection["normal_updates"])
        total_backtracks += int(projection["backtracks"])
        if bool(projection["resolved"]):
            return projected, projected_summary, {
                "resolved": True,
                "normal_updates": int(total_updates),
                "backtracks": int(total_backtracks),
                "gradient_refreshes": int(gradient_refreshes),
                "target_level_status": dict(
                    projection["target_level_status"]
                ),
            }
        if np.array_equal(projected, current):
            return current, summary, {
                "resolved": False,
                "normal_updates": int(total_updates),
                "backtracks": int(total_backtracks),
                "gradient_refreshes": int(gradient_refreshes),
                "target_level_status": dict(target_status),
            }
        current = projected
        gradient_refreshes += 1


def _project_failure_to_boundary(
    evaluator: BatchedForwardMarginEvaluator,
    failure_q: np.ndarray,
    tau: float,
    soft_target_margin: float,
    hard_margin_target: float,
    fd_step: float,
    *,
    progress: bool,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Start at failure and Newton-project to the chosen soft level set."""

    current = np.asarray(failure_q, dtype=float).copy()
    total_corrections = 0
    total_backtracks = 0
    predictor_diagnostics = None
    failure_gradient = None
    failure_fd_coverage = None
    failure_gradient_weight_coverage = None
    failure_fd_scheme_counts = None
    terminal_target_status = None

    def diagnostics() -> Mapping[str, Any]:
        return {
            "normal_updates": int(total_corrections),
            "backtracks": int(total_backtracks),
            "first_predictor": predictor_diagnostics,
            "failure_gradient": failure_gradient,
            "failure_gradient_norm": (
                None if failure_gradient is None else
                float(np.linalg.norm(failure_gradient))
            ),
            "failure_fd_coverage": failure_fd_coverage,
            "failure_gradient_weight_coverage": (
                failure_gradient_weight_coverage
            ),
            "failure_fd_scheme_counts": failure_fd_scheme_counts,
            "target_level_status": terminal_target_status,
            "resolved": bool(
                terminal_target_status is not None
                and terminal_target_status["resolved"]
            ),
        }

    bar = tqdm(
        desc="Failure -> target soft level",
        unit="normal-update",
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        while True:
            bundle = _gradient_bundle(
                evaluator,
                current,
                tau,
                fd_step,
                progress=progress,
            )
            summary = bundle["summary"]
            gradient = bundle["gradient"]
            residual = _target_residual(summary, soft_target_margin)
            terminal_target_status = _target_level_status(
                summary,
                soft_target_margin,
                hard_margin_target,
                gradient,
                fd_step,
            )
            if failure_gradient is None:
                failure_gradient = np.asarray(
                    gradient,
                    dtype=float,
                ).tolist()
                failure_fd_coverage = np.asarray(
                    bundle["fd_coverage"],
                    dtype=float,
                ).tolist()
                failure_gradient_weight_coverage = np.asarray(
                    bundle["gradient_weight_coverage"],
                    dtype=float,
                ).tolist()
                schemes = np.asarray(
                    bundle["fd_difference_scheme"],
                    dtype=np.uint8,
                )
                refinements = np.asarray(
                    bundle["fd_refinement_count"],
                    dtype=int,
                )
                failure_fd_scheme_counts = [
                    {
                        "label": str(label),
                        "central": int(np.count_nonzero(
                            schemes[:, axis] == 1
                        )),
                        "forward": int(np.count_nonzero(
                            schemes[:, axis] == 2
                        )),
                        "backward": int(np.count_nonzero(
                            schemes[:, axis] == 3
                        )),
                        "refinements": int(refinements[axis]),
                    }
                    for axis, label in enumerate(GAIN_LABELS)
                ]

            if bool(terminal_target_status["resolved"]):
                return current, bundle, diagnostics()

            normal_step = _normal_newton_step(
                residual,
                gradient,
            )
            proposal = np.clip(
                current + normal_step,
                0.0,
                1.0,
            )
            if np.array_equal(proposal, current):
                return current, bundle, diagnostics()

            proposal_row = evaluator.evaluate(proposal)
            if np.all(proposal_row.pole_valid_mask):
                proposal_summary = _softmax_summary(
                    proposal_row.spectral_radius,
                    tau,
                )
                if predictor_diagnostics is None:
                    predictor_diagnostics = {
                        "step_norm": float(np.linalg.norm(normal_step)),
                        "linearized_residual": float(
                            residual + np.dot(gradient, normal_step)
                        ),
                        "exact_residual": float(
                            _target_residual(
                                proposal_summary,
                                soft_target_margin,
                            )
                        ),
                        "exact_residual_reduced": bool(
                            abs(_target_residual(
                                proposal_summary,
                                soft_target_margin,
                            )) < abs(residual)
                        ),
                    }
                if abs(_target_residual(
                    proposal_summary,
                    soft_target_margin,
                )) < abs(residual):
                    current = proposal
                    total_corrections += 1
                    bar.update(1)
                    if progress:
                        bar.set_postfix(
                            residual=(
                                f"{_target_residual(proposal_summary, soft_target_margin):.3e}"
                            )
                        )
                    continue

            # Newton predictor did not reduce exact residual: backtrack on the
            # same normal displacement until it does.
            displacement = proposal - current
            scale = 0.5
            while True:
                total_backtracks += 1
                candidate = np.clip(
                    current + scale * displacement,
                    0.0,
                    1.0,
                )
                if np.array_equal(candidate, current):
                    return current, bundle, diagnostics()
                candidate_row = evaluator.evaluate(candidate)
                if np.all(candidate_row.pole_valid_mask):
                    candidate_summary = _softmax_summary(
                        candidate_row.spectral_radius,
                        tau,
                    )
                    if abs(_target_residual(
                        candidate_summary,
                        soft_target_margin,
                    )) < abs(residual):
                        current = candidate
                        total_corrections += 1
                        bar.update(1)
                        break
                scale *= 0.5
    finally:
        bar.close()


def _walk_tangent_to_nearest_failure(
    evaluator: BatchedForwardMarginEvaluator,
    start_q: np.ndarray,
    failure_q: np.ndarray,
    tau: float,
    soft_target_margin: float,
    hard_margin_target: float,
    fd_step: float,
    path_step: float,
    *,
    progress: bool,
) -> tuple[np.ndarray, list[Mapping[str, Any]], Mapping[str, Any]]:
    """Follow a chosen delta_tau level while approaching the failed gain."""

    current = np.asarray(start_q, dtype=float).copy()
    history: list[Mapping[str, Any]] = []
    total_backtracks = 0
    total_normal_updates = 0
    total_normal_backtracks = 0

    bar = tqdm(
        desc=f"Tangent continuation tau={tau:.3e}",
        unit="accepted-step",
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        while True:
            bundle = _gradient_bundle(
                evaluator,
                current,
                tau,
                fd_step,
                progress=progress,
            )
            row = bundle["row"]
            summary = bundle["summary"]
            gradient = bundle["gradient"]

            # Tangent geometry is defined on the level set.  Finish the local
            # normal correction first, then rebuild J_rho at the corrected
            # point before constructing a tangent predictor.
            target_status = _target_level_status(
                summary,
                soft_target_margin,
                hard_margin_target,
                gradient,
                fd_step,
            )
            if not bool(target_status["resolved"]):
                projected, _projected_summary, projection = (
                    _fresh_normal_projection(
                        evaluator,
                        current,
                        tau,
                        soft_target_margin,
                        hard_margin_target,
                        fd_step,
                        progress=progress,
                    )
                )
                total_normal_updates += int(projection["normal_updates"])
                total_normal_backtracks += int(projection["backtracks"])
                if not bool(projection["resolved"]):
                    raise RuntimeError(
                        "normal correction from a tangent point became unresolved"
                    )
                if not np.array_equal(projected, current):
                    current = projected
                    continue

            tangent = _tangent_direction(
                current,
                failure_q,
                gradient,
            )
            tangent_norm = float(np.linalg.norm(tangent))
            distance = float(
                np.linalg.norm(current - failure_q)
            )

            history.append({
                "q": current.copy(),
                "rho": np.asarray(
                    row.spectral_radius,
                    dtype=float,
                ).copy(),
                "sorted_rho": np.asarray(
                    summary["sorted_rho"],
                    dtype=float,
                ).copy(),
                "rho_order": np.asarray(
                    summary["order"],
                    dtype=int,
                ).copy(),
                "rho_jacobian": np.asarray(
                    bundle["rho_jacobian"],
                    dtype=float,
                ).copy(),
                "weights": np.asarray(
                    summary["weights"],
                    dtype=float,
                ).copy(),
                "gradient": gradient.copy(),
                "fd_coverage": np.asarray(
                    bundle["fd_coverage"],
                    dtype=float,
                ).copy(),
                "fd_refinement_count": np.asarray(
                    bundle["fd_refinement_count"],
                    dtype=int,
                ).copy(),
                "fd_difference_scheme": np.asarray(
                    bundle["fd_difference_scheme"],
                    dtype=np.uint8,
                ).copy(),
                "gradient_weight_coverage": np.asarray(
                    bundle["gradient_weight_coverage"],
                    dtype=float,
                ).copy(),
                "delta_soft": float(
                    summary["delta_soft"]
                ),
                "soft_target_margin": float(soft_target_margin),
                "target_residual": float(
                    _target_residual(summary, soft_target_margin)
                ),
                "delta_hard": float(
                    summary["delta_hard"]
                ),
                "stable_fraction": float(
                    summary["stable_fraction"]
                ),
                "distance_to_failure": distance,
                "tangent_norm": tangent_norm,
            })

            # Within the coordinate resolution used to estimate J_rho, the
            # failed gain direction is locally normal to the boundary.
            if tangent_norm <= fd_step:
                return current, history, {
                    "total_backtracks": int(total_backtracks),
                    "total_normal_updates": int(total_normal_updates),
                    "total_normal_backtracks": int(total_normal_backtracks),
                    "terminal_tangent_norm": tangent_norm,
                    "terminal_reason": (
                        "tangent norm is within finite-difference resolution"
                    ),
                }

            step = min(
                float(path_step),
                tangent_norm,
            )
            direction = tangent / tangent_norm
            trial_scale = 1.0
            accepted = False

            while True:
                trial_step = trial_scale * step
                if trial_step <= fd_step:
                    return current, history, {
                        "total_backtracks": int(total_backtracks),
                        "total_normal_updates": int(total_normal_updates),
                        "total_normal_backtracks": int(
                            total_normal_backtracks
                        ),
                        "terminal_tangent_norm": tangent_norm,
                        "terminal_trial_step": float(trial_step),
                        "terminal_reason": (
                            "no distance-decreasing target-level tangent "
                            "step remains above finite-difference resolution"
                        ),
                    }
                predictor = np.clip(
                    current
                    + trial_step * direction,
                    0.0,
                    1.0,
                )
                if np.array_equal(predictor, current):
                    return current, history, {
                        "total_backtracks": int(total_backtracks),
                        "total_normal_updates": int(total_normal_updates),
                        "total_normal_backtracks": int(total_normal_backtracks),
                        "terminal_tangent_norm": tangent_norm,
                        "terminal_reason": (
                            "box-constrained tangent predictor cannot move"
                        ),
                    }

                # First correct back to the current local boundary using the
                # current normal; the next accepted point gets a fresh J_rho.
                predictor_row = evaluator.evaluate(predictor)
                if np.all(predictor_row.pole_valid_mask):
                    predictor_summary = _softmax_summary(
                        predictor_row.spectral_radius,
                        tau,
                    )
                    corrected, corrected_summary, correction = (
                        _exact_normal_projection(
                            evaluator,
                            predictor,
                            tau,
                            soft_target_margin,
                            hard_margin_target,
                            fd_step,
                            gradient,
                            progress=progress,
                        )
                    )
                    total_normal_updates += int(correction["normal_updates"])
                    total_normal_backtracks += int(correction["backtracks"])
                    corrected, corrected_summary, refreshed = (
                        _fresh_normal_projection(
                            evaluator,
                            corrected,
                            tau,
                            soft_target_margin,
                            hard_margin_target,
                            fd_step,
                            progress=progress,
                        )
                    )
                    total_normal_updates += int(refreshed["normal_updates"])
                    total_normal_backtracks += int(refreshed["backtracks"])
                    if bool(refreshed["resolved"]):
                        corrected_displacement = float(
                            np.linalg.norm(corrected - current)
                        )
                        new_distance = float(
                            np.linalg.norm(
                                corrected - failure_q
                            )
                        )
                        if corrected_displacement <= fd_step:
                            return current, history, {
                                "total_backtracks": int(total_backtracks),
                                "total_normal_updates": int(
                                    total_normal_updates
                                ),
                                "total_normal_backtracks": int(
                                    total_normal_backtracks
                                ),
                                "terminal_tangent_norm": tangent_norm,
                                "terminal_corrected_displacement": (
                                    corrected_displacement
                                ),
                                "terminal_reason": (
                                    "corrected target-level motion is within "
                                    "finite-difference resolution"
                                ),
                            }
                        if new_distance < distance:
                            # Accept the exact-forward corrected point.  Any
                            # residual left at floating-point resolution is
                            # revisited with the next fresh gradient bundle.
                            current = corrected
                            accepted = True
                            bar.update(1)
                            if progress:
                                bar.set_postfix(
                                    distance=f"{new_distance:.4f}",
                                    residual=(
                                        f"{_target_residual(corrected_summary, soft_target_margin):.2e}"
                                    ),
                                    tangent=f"{tangent_norm:.3e}",
                                )
                            break

                trial_scale *= 0.5
                total_backtracks += 1

            if not accepted:
                return current, history, {
                    "total_backtracks": int(total_backtracks),
                    "total_normal_updates": int(total_normal_updates),
                    "total_normal_backtracks": int(total_normal_backtracks),
                    "terminal_tangent_norm": tangent_norm,
                    "terminal_reason": (
                        "tangent correction found no accepted predictor"
                    ),
                }
    finally:
        bar.close()


def _linearized_group_contours(
    output: Path,
    final_q: np.ndarray,
    failure_q: np.ndarray,
    rho: np.ndarray,
    rho_jacobian: np.ndarray,
    hard_margin_target: float,
    fd_step: float,
    grid_size: int,
) -> tuple[plt.Figure, Mapping[str, np.ndarray]]:
    """Draw local hard-margin fields from the retained plantwise Jacobian."""

    q0 = np.asarray(final_q, dtype=float)
    qf = np.asarray(failure_q, dtype=float)
    rho0 = np.asarray(rho, dtype=float)
    jac = np.asarray(rho_jacobian, dtype=float)

    figure, axes = plt.subplots(
        4,
        3,
        figsize=(16.0, 17.0),
        constrained_layout=True,
    )

    panel_fields: list[Mapping[str, Any]] = []

    for group_index, group in enumerate(GROUPS):
        base = 3 * group_index
        for column, (projection_name, local_a, local_b) in enumerate(PROJECTIONS):
            axis = axes[group_index, column]
            a = base + local_a
            b = base + local_b

            span_a = max(abs(q0[a] - qf[a]), 2.0 * fd_step)
            span_b = max(abs(q0[b] - qf[b]), 2.0 * fd_step)
            a_min = max(0.0, min(q0[a], qf[a]) - span_a)
            a_max = min(1.0, max(q0[a], qf[a]) + span_a)
            b_min = max(0.0, min(q0[b], qf[b]) - span_b)
            b_max = min(1.0, max(q0[b], qf[b]) + span_b)

            qa = np.linspace(a_min, a_max, int(grid_size))
            qb = np.linspace(b_min, b_max, int(grid_size))
            da = qa[:, None] - q0[a]
            db = qb[None, :] - q0[b]

            predicted = (
                rho0[:, None, None]
                + jac[:, a, None, None] * da[None, :, :]
                + jac[:, b, None, None] * db[None, :, :]
            )

            hard_delta = 1.0 - np.max(
                predicted,
                axis=0,
            )

            panel_fields.append({
                "axis": axis,
                "group_index": group_index,
                "column": column,
                "axis_a": a,
                "axis_b": b,
                "qa": qa,
                "qb": qb,
                "hard_delta": hard_delta,
            })

    color_limit = max(
        float(np.max(np.abs(panel["hard_delta"])))
        for panel in panel_fields
    )
    if color_limit == 0.0:
        color_limit = float(np.finfo(float).eps)
    normalization = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    image = None
    contour_levels = (
        -float(hard_margin_target),
        0.0,
        float(hard_margin_target),
        2.0 * float(hard_margin_target),
    )
    contour_styles = (":", "--", "-", "-.")
    contour_widths = (1.0, 1.7, 2.2, 1.0)
    caps = GAIN_CAPS.reshape(-1)

    for panel in panel_fields:
        axis = panel["axis"]
        a = int(panel["axis_a"])
        b = int(panel["axis_b"])
        qa = np.asarray(panel["qa"], dtype=float)
        qb = np.asarray(panel["qb"], dtype=float)
        hard_delta = np.asarray(panel["hard_delta"], dtype=float)
        x = qa * caps[a]
        y = qb * caps[b]

        image = axis.pcolormesh(
            x,
            y,
            hard_delta.T,
            shading="auto",
            cmap="RdYlGn",
            norm=normalization,
            rasterized=True,
        )
        field_min = float(np.min(hard_delta))
        field_max = float(np.max(hard_delta))
        for level, linestyle, linewidth in zip(
            contour_levels,
            contour_styles,
            contour_widths,
        ):
            if field_min <= level <= field_max:
                axis.contour(
                    x,
                    y,
                    hard_delta.T,
                    levels=[level],
                    colors="black",
                    linestyles=linestyle,
                    linewidths=linewidth,
                )

        axis.scatter(
            [qf[a] * caps[a]],
            [qf[b] * caps[b]],
            marker="x",
            s=120,
            linewidths=2.4,
            c="red",
            zorder=6,
        )
        axis.scatter(
            [q0[a] * caps[a]],
            [q0[b] * caps[b]],
            marker="*",
            s=210,
            c="limegreen",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )
        axis.scatter(
            [q0[a] * caps[a]],
            [q0[b] * caps[b]],
            marker="o",
            s=62,
            facecolors="none",
            edgecolors="white",
            linewidths=1.8,
            zorder=8,
        )

        group_index = int(panel["group_index"])
        column = int(panel["column"])
        axis.set_xlabel(GAIN_LABELS[a])
        axis.set_ylabel(GAIN_LABELS[b])
        axis.set_title(f"{GROUPS[group_index]} {PROJECTIONS[column][0]}")
        axis.grid(True, alpha=0.18)

    if image is None:
        raise RuntimeError("local hard-margin contour grid has no panels")
    colorbar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(
        r"linearized sampled hard margin $1-\max_i\widehat{\rho}_i$"
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="x", color="red", linestyle="none",
            markersize=10, markeredgewidth=2.2, label="recorded failure",
        ),
        Line2D(
            [0], [0], marker="*", color="limegreen", markeredgecolor="black",
            linestyle="none", markersize=14, label="proposed gain",
        ),
        Line2D(
            [0], [0], marker="o", color="white", markerfacecolor="none",
            linestyle="none", markersize=7, markeredgewidth=1.6,
            label="local linearization center",
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.7,
            label="hard margin = 0",
        ),
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=2.2,
            label=f"hard margin = target ({hard_margin_target:.2g})",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        fontsize=8,
    )

    figure.suptitle(
        "Linearized local sampled hard-margin field "
        "(green = safer, red = unsafe)\n"
        "Contours are local first-order diagnostics, not exact global boundaries",
        fontsize=13,
    )
    figure.savefig(
        output / "spectral_gradient_group_contours.png",
        dpi=180,
    )
    contour_data = {
        "q_a": np.asarray(
            [panel["qa"] for panel in panel_fields],
            dtype=float,
        ).reshape(4, 3, grid_size),
        "q_b": np.asarray(
            [panel["qb"] for panel in panel_fields],
            dtype=float,
        ).reshape(4, 3, grid_size),
        "hard_margin": np.asarray(
            [panel["hard_delta"] for panel in panel_fields],
            dtype=float,
        ).reshape(4, 3, grid_size, grid_size),
        "axis_pairs": np.asarray(
            [
                (panel["axis_a"], panel["axis_b"])
                for panel in panel_fields
            ],
            dtype=int,
        ).reshape(4, 3, 2),
    }
    return figure, contour_data


def _plot_path(
    output: Path,
    failure_q: np.ndarray,
    stage_records: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.5),
        constrained_layout=True,
    )

    axis = axes[0]
    for stage_index, stage in enumerate(stage_records):
        history = stage["history"]
        distance = [
            float(row["distance_to_failure"])
            for row in history
        ]
        soft = [
            float(row["delta_soft"])
            for row in history
        ]
        hard = [
            float(row["delta_hard"])
            for row in history
        ]
        target = [
            float(stage["soft_target_margin"])
            for _row in history
        ]
        axis.plot(
            distance,
            soft,
            marker="o",
            label=(
                f"soft N={stage['sample_count']}, "
                f"tau={stage['tau']:.1e}"
            ),
        )
        axis.plot(
            distance,
            hard,
            linestyle="--",
            marker="x",
            alpha=0.6,
            label=(
                "hard-max margin"
                if stage_index == 0 else None
            ),
        )
        axis.plot(
            distance,
            target,
            linestyle="none",
            marker="s",
            markerfacecolor="none",
            label=(
                "soft target m + tau log(N)"
                if stage_index == 0 else None
            ),
        )
    axis.axhline(0.0)
    axis.set_xlabel("normalized 12-D distance to failure")
    axis.set_ylabel("margin")
    axis.set_title("Target-level continuation and sharpening")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.25)

    axis = axes[1]
    for stage in stage_records:
        history = stage["history"]
        if not history:
            continue
        selected = history[-1]["sorted_rho"]
        u = (
            np.arange(len(selected), dtype=float) + 1.0
        ) / len(selected)
        axis.plot(
            u,
            selected,
            label=(
                f"N={stage['sample_count']}, "
                f"tau={stage['tau']:.1e}"
            ),
        )
    axis.axhline(1.0)
    axis.set_xlabel("empirical quantile")
    axis.set_ylabel("spectral radius")
    axis.set_title("Order-statistic curve at each sharpened target level")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.25)

    axis = axes[2]
    x = np.arange(12)
    axis.plot(
        x,
        failure_q,
        marker="o",
        label="failure",
    )
    for stage in stage_records:
        if stage["history"]:
            axis.plot(
                x,
                stage["history"][-1]["q"],
                marker=".",
                label=(
                    f"N={stage['sample_count']}, "
                    f"tau={stage['tau']:.1e}"
                ),
            )
    axis.set_xticks(x)
    axis.set_xticklabels(
        GAIN_LABELS,
        rotation=55,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("normalized gain q")
    axis.set_title("PID displacement from failure")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.25)

    figure.savefig(
        output / "spectral_gradient_sharpen.png",
        dpi=180,
    )
    return figure


def _build_gain_update_table(
    failure_q: np.ndarray,
    proposal_q: np.ndarray,
) -> list[list[str]]:
    failure = np.asarray(failure_q, dtype=float).reshape(4, 3)
    proposal = np.asarray(proposal_q, dtype=float).reshape(4, 3)
    caps = np.asarray(GAIN_CAPS, dtype=float).reshape(4, 3)
    rows: list[list[str]] = []
    for group_index, group in enumerate(GROUPS):
        for local_index, gain_name in enumerate(("P", "I", "D")):
            failure_value = float(
                failure[group_index, local_index]
                * caps[group_index, local_index]
            )
            proposal_value = float(
                proposal[group_index, local_index]
                * caps[group_index, local_index]
            )
            delta = proposal_value - failure_value
            normalized_delta = float(
                proposal[group_index, local_index]
                - failure[group_index, local_index]
            )
            rows.append([
                group,
                gain_name,
                f"{failure_value:.7g}",
                f"{proposal_value:.7g}",
                f"{delta:+.7g}",
                f"{normalized_delta:+.7g}",
            ])
    return rows


def _build_pdf_report(
    output: Path,
    overview_figure: plt.Figure,
    contour_figure: plt.Figure,
    failure_q: np.ndarray,
    proposal_q: np.ndarray,
    hard_margin_target: float,
    sample_count: int,
    prefix_counts: Sequence[int],
    tau_initial: float,
    tau_final: float,
    final_soft_target_margin: float,
    final_summary: Mapping[str, Any],
) -> Path:
    """Write the overview, local field, and gain table as one PDF."""

    pdf_path = output / "spectral_gradient_hard_margin_report.pdf"
    table_rows = _build_gain_update_table(failure_q, proposal_q)
    distance = float(np.linalg.norm(
        np.asarray(proposal_q, dtype=float)
        - np.asarray(failure_q, dtype=float)
    ))
    target_residual = _target_residual(
        final_summary,
        final_soft_target_margin,
    )

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(overview_figure, bbox_inches="tight")
        pdf.savefig(contour_figure, bbox_inches="tight")

        table_figure, axis = plt.subplots(
            figsize=(11.69, 8.27),
            constrained_layout=True,
        )
        axis.axis("off")
        axis.set_title(
            "PID gain update: recorded failure to hard-margin proposal",
            fontsize=16,
            pad=14,
        )
        table = axis.table(
            cellText=table_rows,
            colLabels=(
                "Group",
                "Gain",
                "Failure",
                "Proposed",
                "Delta",
                "Delta / cap",
            ),
            cellLoc="right",
            colLoc="center",
            bbox=(0.04, 0.27, 0.92, 0.62),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.25)
        for (row_index, _column_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#d9e5f3")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f3f5f7")

        summary_lines = (
            f"Hard-margin target m: {hard_margin_target:.7g}",
            f"Soft target at final tau: {final_soft_target_margin:.7g}",
            f"Final sample count N_max: {sample_count}",
            "Nested Gaussian-QMC prefixes: "
            + " -> ".join(str(int(n)) for n in prefix_counts),
            f"Initial / final tau: {tau_initial:.7g} / {tau_final:.7g}",
            f"Exact final soft margin: {float(final_summary['delta_soft']):+.7g}",
            f"Exact final hard margin: {float(final_summary['delta_hard']):+.7g}",
            f"Final soft-level residual: {target_residual:+.3e}",
            f"Normalized 12-D distance from failure: {distance:.7g}",
        )
        axis.text(
            0.04,
            0.22,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            transform=axis.transAxes,
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": "#f7f7f7",
                "edgecolor": "#888888",
            },
        )
        axis.text(
            0.55,
            0.16,
            (
                "The proposal is evaluated with the exact sampled forward "
                "model. Contour fields on page 2 are local first-order "
                "diagnostics only."
            ),
            va="top",
            ha="left",
            fontsize=9,
            style="italic",
            wrap=True,
            transform=axis.transAxes,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#fff8dc",
                "edgecolor": "#c0a96b",
            },
        )
        pdf.savefig(table_figure, bbox_inches="tight")
        plt.close(table_figure)

        metadata = pdf.infodict()
        metadata["Title"] = "PID spectral-gradient hard-margin report"
        metadata["Subject"] = (
            "Failure-start continuation with sampled hard-margin target"
        )

    return pdf_path


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    progress = not bool(arguments.no_progress)
    hard_margin_target = float(arguments.hard_margin_target)

    estimate_path = Path(
        arguments.estimate_json
    ).expanduser().resolve()
    estimate = load_estimate_json(
        estimate_path
    )

    failure_gains = _recorded_gain_matrix(
        estimate
    )
    caps = GAIN_CAPS.reshape(-1)
    failure_q = failure_gains.reshape(-1) / caps
    if np.any(failure_q < 0.0) or np.any(failure_q > 1.0):
        raise ValueError(
            "recorded failure gains lie outside exploration caps"
        )

    vehicle_model = load_vehicle_model(
        Path(estimate["input"]["vehicle_model"])
    )
    actuator_parameters = actuator_parameters_from_estimate(
        estimate
    )
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

    evaluator = BatchedForwardMarginEvaluator(
        plants=plants,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        workers=int(arguments.workers),
    )

    try:
        prefix_counts = _nested_prefix_counts(
            int(arguments.samples),
            int(arguments.initial_prefix_samples),
        )
        current_q = failure_q.copy()
        stage_records: list[Mapping[str, Any]] = []
        prefix_records: list[Mapping[str, Any]] = []
        tau_values: list[float] = []
        initial_projection: Optional[Mapping[str, Any]] = None
        final_bundle: Mapping[str, Any]
        final_summary: Mapping[str, Any]
        tau = float(arguments.tau)

        for prefix_index, sample_count in enumerate(prefix_counts):
            evaluator.set_active_sample_count(sample_count)
            cache_before = dict(evaluator.cache_diagnostics())
            prefix_start_q = current_q.copy()
            revalidation_row = evaluator.evaluate(current_q)
            cache_after_revalidation = dict(
                evaluator.cache_diagnostics()
            )
            if not np.all(revalidation_row.pole_valid_mask):
                raise RuntimeError(
                    f"gain revalidation has unresolved plant poles at "
                    f"N={sample_count}"
                )

            tau = float(arguments.tau)
            soft_target_margin = _soft_target_margin(
                tau,
                sample_count,
                hard_margin_target,
            )
            revalidation_summary = _softmax_summary(
                revalidation_row.spectral_radius,
                tau,
            )
            if prefix_index == 0:
                current_q, _initial_bundle, prefix_projection = (
                    _project_failure_to_boundary(
                        evaluator,
                        current_q,
                        tau,
                        soft_target_margin,
                        hard_margin_target,
                        float(arguments.fd_step),
                        progress=progress,
                    )
                )
                initial_projection = dict(prefix_projection)
            else:
                current_q, _summary, prefix_projection = (
                    _fresh_normal_projection(
                        evaluator,
                        current_q,
                        tau,
                        soft_target_margin,
                        hard_margin_target,
                        float(arguments.fd_step),
                        progress=progress,
                    )
                )
            if not bool(prefix_projection["resolved"]):
                raise RuntimeError(
                    f"plant-prefix projection cannot resolve the "
                    f"hard-margin target level at N={sample_count}"
                )

            prefix_stage_start = len(stage_records)

            while True:
                soft_target_margin = _soft_target_margin(
                    tau,
                    sample_count,
                    hard_margin_target,
                )
                # Reproject with a fresh local rho Jacobian before tangent motion.
                boundary_bundle = _gradient_bundle(
                    evaluator,
                    current_q,
                    tau,
                    float(arguments.fd_step),
                    progress=progress,
                )
                boundary_summary = boundary_bundle["summary"]
                boundary_gradient = boundary_bundle["gradient"]
                stage_projection: Mapping[str, Any] = {
                    "resolved": True,
                    "normal_updates": 0,
                    "backtracks": 0,
                }

                boundary_status = _target_level_status(
                    boundary_summary,
                    soft_target_margin,
                    hard_margin_target,
                    boundary_gradient,
                    float(arguments.fd_step),
                )
                if not bool(boundary_status["resolved"]):
                    projected, _summary, stage_projection = (
                        _fresh_normal_projection(
                            evaluator,
                            current_q,
                            tau,
                            soft_target_margin,
                            hard_margin_target,
                            float(arguments.fd_step),
                            progress=progress,
                        )
                    )
                    if not bool(stage_projection["resolved"]):
                        raise RuntimeError(
                            "current tau stage cannot resolve its target "
                            f"level at N={sample_count}"
                        )
                    current_q = projected

                current_q, history, tangent_diagnostics = (
                    _walk_tangent_to_nearest_failure(
                        evaluator,
                        current_q,
                        failure_q,
                        tau,
                        soft_target_margin,
                        hard_margin_target,
                        float(arguments.fd_step),
                        float(arguments.path_step),
                        progress=progress,
                    )
                )

                final_bundle = _gradient_bundle(
                    evaluator,
                    current_q,
                    tau,
                    float(arguments.fd_step),
                    progress=progress,
                )
                final_summary = final_bundle["summary"]
                final_gradient = final_bundle["gradient"]
                gradient_norm = float(np.linalg.norm(final_gradient))
                if gradient_norm == 0.0:
                    sharpen_distance_bound = float("inf")
                else:
                    sharpen_distance_bound = float(
                        tau
                        * math.log(float(sample_count))
                        / gradient_norm
                    )

                stage_records.append({
                    "prefix_index": int(prefix_index),
                    "sample_count": int(sample_count),
                    "tau": float(tau),
                    "hard_margin_target": float(hard_margin_target),
                    "soft_target_margin": float(soft_target_margin),
                    "target_level_status": dict(
                        _target_level_status(
                            final_summary,
                            soft_target_margin,
                            hard_margin_target,
                            final_gradient,
                            float(arguments.fd_step),
                        )
                    ),
                    "gap_bound": float(
                        tau * math.log(float(sample_count))
                    ),
                    "gain_distance_gap_bound": sharpen_distance_bound,
                    "history": history,
                    "normal_projection": dict(stage_projection),
                    "tangent_diagnostics": dict(tangent_diagnostics),
                    "endpoint": {
                        **_point_payload(
                            current_q,
                            final_summary,
                            tau,
                            hard_margin_target=hard_margin_target,
                            sample_count=sample_count,
                        ),
                        "distance_to_failure": float(
                            np.linalg.norm(current_q - failure_q)
                        ),
                        "gradient_norm": gradient_norm,
                    },
                })
                tau_values.append(float(tau))

                if sharpen_distance_bound <= float(arguments.fd_step):
                    break

                tau *= 0.5
                sharper_soft_target_margin = _soft_target_margin(
                    tau,
                    sample_count,
                    hard_margin_target,
                )

                # Reuse the retained prefix rho_i and J_rho for the first
                # sharpen predictor at this same QMC resolution.
                sharper_summary = _softmax_summary(
                    final_bundle["row"].spectral_radius,
                    tau,
                )
                sharper_gradient, _coverage = _soft_gradient(
                    final_bundle["rho_jacobian"],
                    sharper_summary["weights"],
                )
                proposal = np.clip(
                    current_q
                    + _normal_newton_step(
                        _target_residual(
                            sharper_summary,
                            sharper_soft_target_margin,
                        ),
                        sharper_gradient,
                    ),
                    0.0,
                    1.0,
                )
                proposal_row = evaluator.evaluate(proposal)
                if np.all(proposal_row.pole_valid_mask):
                    current_q = proposal

            prefix_records.append({
                "prefix_index": int(prefix_index),
                "sample_count": int(sample_count),
                "start_q": prefix_start_q.tolist(),
                "start_revalidation": {
                    **_point_payload(
                        prefix_start_q,
                        revalidation_summary,
                        float(arguments.tau),
                        hard_margin_target=hard_margin_target,
                        sample_count=sample_count,
                    ),
                    "pole_valid_count": int(np.count_nonzero(
                        revalidation_row.pole_valid_mask
                    )),
                },
                "initial_projection": dict(prefix_projection),
                "stage_index_start": int(prefix_stage_start),
                "stage_index_stop": int(len(stage_records)),
                "final_tau": float(tau),
                "final": {
                    **_point_payload(
                        current_q,
                        final_summary,
                        tau,
                        hard_margin_target=hard_margin_target,
                        sample_count=sample_count,
                    ),
                    "distance_to_failure": float(
                        np.linalg.norm(current_q - failure_q)
                    ),
                },
                "cache_before": cache_before,
                "cache_after_revalidation": cache_after_revalidation,
                "revalidation_new_plant_evaluation_count": int(
                    cache_after_revalidation[
                        "new_plant_evaluation_count"
                    ]
                    - cache_before["new_plant_evaluation_count"]
                ),
                "revalidation_reused_plant_count": int(
                    cache_after_revalidation["cache_hit_count"]
                    - cache_before["cache_hit_count"]
                ),
                "cache_after": dict(evaluator.cache_diagnostics()),
            })

        if initial_projection is None:
            raise RuntimeError("nested plant refinement did not run")

        # Both proposal and failure are now exact at N_max.  Extending the
        # cached failure row evaluates only the previously unseen suffix.
        failure_row = evaluator.evaluate(failure_q)
        final_bundle = _gradient_bundle(
            evaluator,
            current_q,
            tau,
            float(arguments.fd_step),
            progress=progress,
        )
        final_summary = final_bundle["summary"]
        final_soft_target_margin = _soft_target_margin(
            tau,
            int(arguments.samples),
            hard_margin_target,
        )

        output = (
            Path(arguments.output_dir).expanduser().resolve()
            if arguments.output_dir is not None
            else estimate_path.parent
            / "pid_spectral_gradient_sharpen"
        )
        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        overview_figure = _plot_path(
            output,
            failure_q,
            stage_records,
        )
        contour_figure, contour_data = _linearized_group_contours(
            output,
            current_q,
            failure_q,
            final_bundle["row"].spectral_radius,
            final_bundle["rho_jacobian"],
            hard_margin_target,
            float(arguments.fd_step),
            int(arguments.contour_grid),
        )
        report_pdf = _build_pdf_report(
            output,
            overview_figure,
            contour_figure,
            failure_q,
            current_q,
            hard_margin_target,
            int(arguments.samples),
            prefix_counts,
            float(arguments.tau),
            float(tau),
            final_soft_target_margin,
            final_summary,
        )
        plt.close(overview_figure)
        plt.close(contour_figure)

        # Flatten stage history for numerical post-analysis.
        flat_history = [
            (stage_index, row_index, stage, row)
            for stage_index, stage in enumerate(stage_records)
            for row_index, row in enumerate(stage["history"])
        ]
        if flat_history:
            np.savez_compressed(
                output / "spectral_gradient_sharpen.npz",
                quotient_coordinates=quotient,
                gain_caps=GAIN_CAPS,
                failure_q=failure_q,
                stage_index=np.asarray(
                    [item[0] for item in flat_history],
                    dtype=int,
                ),
                stage_prefix_index=np.asarray(
                    [item[2]["prefix_index"] for item in flat_history],
                    dtype=int,
                ),
                stage_sample_count=np.asarray(
                    [item[2]["sample_count"] for item in flat_history],
                    dtype=int,
                ),
                stage_tau=np.asarray(
                    [item[2]["tau"] for item in flat_history],
                    dtype=float,
                ),
                stage_soft_target_margin=np.asarray(
                    [
                        item[2]["soft_target_margin"]
                        for item in flat_history
                    ],
                    dtype=float,
                ),
                path_q=np.asarray(
                    [item[3]["q"] for item in flat_history],
                    dtype=float,
                ),
                path_rho=_pad_plant_prefix_arrays(
                    [item[3]["rho"] for item in flat_history],
                    int(arguments.samples),
                    dtype=float,
                    fill_value=np.nan,
                ),
                path_sorted_rho=_pad_plant_prefix_arrays(
                    [item[3]["sorted_rho"] for item in flat_history],
                    int(arguments.samples),
                    dtype=float,
                    fill_value=np.nan,
                ),
                path_rho_order=_pad_plant_prefix_arrays(
                    [item[3]["rho_order"] for item in flat_history],
                    int(arguments.samples),
                    dtype=int,
                    fill_value=-1,
                ),
                path_rho_jacobian=_pad_plant_prefix_arrays(
                    [item[3]["rho_jacobian"] for item in flat_history],
                    int(arguments.samples),
                    dtype=float,
                    fill_value=np.nan,
                ),
                path_softmax_weights=_pad_plant_prefix_arrays(
                    [item[3]["weights"] for item in flat_history],
                    int(arguments.samples),
                    dtype=float,
                    fill_value=np.nan,
                ),
                path_soft_gradient=np.asarray(
                    [item[3]["gradient"] for item in flat_history],
                    dtype=float,
                ),
                path_fd_coverage=np.asarray(
                    [item[3]["fd_coverage"] for item in flat_history],
                    dtype=float,
                ),
                path_fd_refinement_count=np.asarray(
                    [
                        item[3]["fd_refinement_count"]
                        for item in flat_history
                    ],
                    dtype=int,
                ),
                path_fd_difference_scheme=_pad_plant_prefix_arrays(
                    [
                        item[3]["fd_difference_scheme"]
                        for item in flat_history
                    ],
                    int(arguments.samples),
                    dtype=np.uint8,
                    fill_value=0,
                ),
                path_gradient_weight_coverage=np.asarray(
                    [
                        item[3]["gradient_weight_coverage"]
                        for item in flat_history
                    ],
                    dtype=float,
                ),
                path_delta_soft=np.asarray(
                    [item[3]["delta_soft"] for item in flat_history],
                    dtype=float,
                ),
                path_target_residual=np.asarray(
                    [item[3]["target_residual"] for item in flat_history],
                    dtype=float,
                ),
                path_delta_hard=np.asarray(
                    [item[3]["delta_hard"] for item in flat_history],
                    dtype=float,
                ),
                path_stable_fraction=np.asarray(
                    [item[3]["stable_fraction"] for item in flat_history],
                    dtype=float,
                ),
                path_distance_to_failure=np.asarray(
                    [item[3]["distance_to_failure"] for item in flat_history],
                    dtype=float,
                ),
                final_q=current_q,
                hard_margin_target=np.asarray(
                    hard_margin_target,
                    dtype=float,
                ),
                final_soft_target_margin=np.asarray(
                    final_soft_target_margin,
                    dtype=float,
                ),
                final_target_residual=np.asarray(
                    _target_residual(
                        final_summary,
                        final_soft_target_margin,
                    ),
                    dtype=float,
                ),
                final_rho=np.asarray(
                    final_bundle["row"].spectral_radius,
                    dtype=float,
                ),
                final_rho_jacobian=np.asarray(
                    final_bundle["rho_jacobian"],
                    dtype=float,
                ),
                final_sorted_rho=np.asarray(
                    final_summary["sorted_rho"],
                    dtype=float,
                ),
                final_rho_order=np.asarray(
                    final_summary["order"],
                    dtype=int,
                ),
                final_softmax_weights=np.asarray(
                    final_summary["weights"],
                    dtype=float,
                ),
                final_soft_gradient=np.asarray(
                    final_bundle["gradient"],
                    dtype=float,
                ),
                final_fd_coverage=np.asarray(
                    final_bundle["fd_coverage"],
                    dtype=float,
                ),
                final_fd_refinement_count=np.asarray(
                    final_bundle["fd_refinement_count"],
                    dtype=int,
                ),
                final_fd_difference_scheme=np.asarray(
                    final_bundle["fd_difference_scheme"],
                    dtype=np.uint8,
                ),
                final_gradient_weight_coverage=np.asarray(
                    final_bundle["gradient_weight_coverage"],
                    dtype=float,
                ),
                contour_q_a=np.asarray(
                    contour_data["q_a"],
                    dtype=float,
                ),
                contour_q_b=np.asarray(
                    contour_data["q_b"],
                    dtype=float,
                ),
                contour_axis_pairs=np.asarray(
                    contour_data["axis_pairs"],
                    dtype=int,
                ),
                contour_linearized_hard_margin=np.asarray(
                    contour_data["hard_margin"],
                    dtype=float,
                ),
                prefix_sample_count=np.asarray(
                    prefix_counts,
                    dtype=int,
                ),
                prefix_start_q=np.asarray(
                    [row["start_q"] for row in prefix_records],
                    dtype=float,
                ),
                prefix_final_q=np.asarray(
                    [row["final"]["q"] for row in prefix_records],
                    dtype=float,
                ),
                prefix_start_hard_margin=np.asarray(
                    [
                        row["start_revalidation"]["delta_hard"]
                        for row in prefix_records
                    ],
                    dtype=float,
                ),
                prefix_final_hard_margin=np.asarray(
                    [row["final"]["delta_hard"] for row in prefix_records],
                    dtype=float,
                ),
                tau_values=np.asarray(
                    tau_values,
                    dtype=float,
                ),
            )

        payload = {
            "schema": SCHEMA,
            "source_commit": source_commit(
                _PROJECT_ROOT
            ),
            "case_name": str(
                estimate["case_name"]
            ),
            "estimate_json": str(
                estimate_path
            ),
            "plant_sampling": {
                **dict(plant_sampling),
                "distribution": (
                    "failure-bag quotient-space Gaussian posterior "
                    "theta ~ N(mu, Sigma)"
                ),
                "qmc_role": (
                    "scrambled Sobol supplies fixed Gaussian integration "
                    "points u; z=Phi^-1(u), theta=mu+Lz"
                ),
                "maximum_sample_count": int(
                    arguments.samples
                ),
                "sample_count": int(arguments.samples),
                "covariance": str(arguments.covariance),
                "nested_prefix_counts": [
                    int(value) for value in prefix_counts
                ],
                "initial_prefix_sample_count": int(
                    arguments.initial_prefix_samples
                ),
                "sequence_generation": (
                    "one N_max plant sequence generated once; no plant "
                    "resampling during search"
                ),
                "seed": int(
                    arguments.seed
                ),
            },
            "search": {
                "start": (
                    "exactly the PID gains recorded in the failed bag"
                ),
                "external_seed_artifact": None,
                "method": (
                    "nested Gaussian-QMC prefix refinement; within each "
                    "prefix: normal projection -> hard-margin-guaranteeing "
                    "soft-level tangent continuation -> tau sharpening"
                ),
                "hard_margin_target": float(hard_margin_target),
                "soft_target_rule": (
                    "delta_tau target = hard_margin_target + tau*log(N)"
                ),
                "rho_jacobian": (
                    "finite-difference per-plant d rho_i/dq_j, with all "
                    "gain perturbations submitted in batched worker waves"
                ),
                "initial_tau": float(
                    arguments.tau
                ),
                "final_tau": float(
                    tau
                ),
                "fd_step_in_normalized_gain": float(
                    arguments.fd_step
                ),
                "nominal_tangent_step_in_normalized_gain": float(
                    arguments.path_step
                ),
                "sharpen_rule": "tau <- tau / 2",
                "sharpen_stop": (
                    "tau*log(N)/||grad delta_tau|| <= fd_step"
                ),
                "in_process_cache": (
                    "exact gain evaluations retain their evaluated plant "
                    "prefix; increasing N evaluates only the new suffix for "
                    "that same gain"
                ),
                "nested_prefix_refinement": prefix_records,
                "evaluation_cache": dict(evaluator.cache_diagnostics()),
                "initial_failure_projection": dict(
                    initial_projection
                ),
            },
            "failure": {
                **_point_payload(
                    failure_q,
                    _softmax_summary(
                        failure_row.spectral_radius,
                        float(arguments.tau),
                    ),
                    float(arguments.tau),
                    hard_margin_target=hard_margin_target,
                    sample_count=int(arguments.samples),
                ),
                "pole_valid_count": int(np.count_nonzero(
                    failure_row.pole_valid_mask
                )),
                "pole_invalid_count": int(np.count_nonzero(
                    ~failure_row.pole_valid_mask
                )),
            },
            "stages": [
                {
                    "prefix_index": int(stage["prefix_index"]),
                    "sample_count": int(stage["sample_count"]),
                    "tau": float(stage["tau"]),
                    "hard_margin_target": float(
                        stage["hard_margin_target"]
                    ),
                    "soft_target_margin": float(
                        stage["soft_target_margin"]
                    ),
                    "target_level_status": dict(
                        stage["target_level_status"]
                    ),
                    "gap_bound": float(stage["gap_bound"]),
                    "gain_distance_gap_bound": float(
                        stage["gain_distance_gap_bound"]
                    ),
                    "accepted_path_point_count": int(
                        len(stage["history"])
                    ),
                    "normal_projection": dict(
                        stage["normal_projection"]
                    ),
                    "tangent_diagnostics": dict(
                        stage["tangent_diagnostics"]
                    ),
                    "endpoint": dict(
                        stage["endpoint"]
                    ),
                }
                for stage in stage_records
            ],
            "final": {
                **_point_payload(
                    current_q,
                    final_summary,
                    tau,
                    hard_margin_target=hard_margin_target,
                    sample_count=int(arguments.samples),
                ),
                "distance_to_failure": float(
                    np.linalg.norm(
                        current_q
                        - failure_q
                    )
                ),
                "gradient_norm": float(
                    np.linalg.norm(
                        final_bundle["gradient"]
                    )
                ),
                "target_residual": float(
                    _target_residual(
                        final_summary,
                        final_soft_target_margin,
                    )
                ),
                "hard_margin_target_satisfied": bool(
                    float(final_summary["delta_hard"])
                    >= hard_margin_target
                ),
                "target_level_status": dict(
                    _target_level_status(
                        final_summary,
                        final_soft_target_margin,
                        hard_margin_target,
                        final_bundle["gradient"],
                        float(arguments.fd_step),
                    )
                ),
            },
            "contours": {
                "figure": (
                    "4x3 group PI/ID/DP views of the linearized local "
                    "sampled hard-margin field"
                ),
                "background": (
                    "green is positive/safer hard margin; red is "
                    "negative/unsafe hard margin"
                ),
                "solid": "linearized hard-margin target contour",
                "dashed": "linearized zero hard-margin contour",
                "failure_marker": "large red X",
                "proposal_marker": "large green star",
                "linearization_marker": "small white open circle",
                "warning": (
                    "local first-order diagnostic, not an exact global boundary"
                ),
                "exact_grid_re_evaluation": False,
            },
            "retained_information": {
                "paired_rho": (
                    "rho_i(q) for the same ordered plant samples"
                ),
                "order_statistics": (
                    "sorted rho_(1)(q),...,rho_(N)(q)"
                ),
                "rho_gain_jacobian": (
                    "finite-difference d rho_i/dq_j"
                ),
                "finite_difference_scheme_codes": {
                    "1": "central",
                    "2": "baseline-to-plus one-sided",
                    "3": "minus-to-baseline one-sided",
                },
                "softmax_weights": (
                    "tau-dependent aggregation weights retained separately"
                ),
            },
            "elapsed_seconds": float(
                time.perf_counter()
                - started
            ),
            "files": {
                "json": str(
                    output
                    / "spectral_gradient_sharpen.json"
                ),
                "path_figure": str(
                    output
                    / "spectral_gradient_sharpen.png"
                ),
                "group_contours": str(
                    output
                    / "spectral_gradient_group_contours.png"
                ),
                "report_pdf": str(report_pdf),
                "npz": str(
                    output
                    / "spectral_gradient_sharpen.npz"
                ),
            },
        }
        write_json(
            output
            / "spectral_gradient_sharpen.json",
            payload,
        )
        return payload
    finally:
        evaluator.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--estimate-json",
        required=True,
    )
    parser.add_argument(
        "--covariance",
        choices=COVARIANCE_NAMES,
        default="conservative_fusion",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_PLANT_SAMPLES,
        help="Maximum Gaussian-QMC plant count N_max",
    )
    parser.add_argument(
        "--initial-prefix-samples",
        type=int,
        default=DEFAULT_INITIAL_PLANT_PREFIX,
        help=(
            "First nested Gaussian-QMC prefix; doubled until N_max "
            "(default: 16)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
    )
    parser.add_argument(
        "--fd-step",
        type=float,
        default=DEFAULT_FD_STEP,
    )
    parser.add_argument(
        "--path-step",
        type=float,
        default=DEFAULT_PATH_STEP,
    )
    parser.add_argument(
        "--contour-grid",
        type=int,
        default=DEFAULT_CONTOUR_GRID,
    )
    parser.add_argument(
        "--hard-margin-target",
        type=float,
        default=DEFAULT_HARD_MARGIN_TARGET,
        help=(
            "Target sampled hard stability margin m; the continuation uses "
            "delta_tau = m + tau*log(N)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    parser.add_argument(
        "--output-dir",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if int(arguments.samples) <= 1:
        raise ValueError(
            "--samples must be greater than one"
        )
    if int(arguments.initial_prefix_samples) <= 0:
        raise ValueError(
            "--initial-prefix-samples must be positive"
        )
    if (
        not np.isfinite(arguments.tau)
        or arguments.tau <= 0.0
    ):
        raise ValueError(
            "--tau must be finite and positive"
        )
    if (
        not np.isfinite(arguments.fd_step)
        or arguments.fd_step <= 0.0
    ):
        raise ValueError(
            "--fd-step must be finite and positive"
        )
    if (
        not np.isfinite(arguments.path_step)
        or arguments.path_step <= 0.0
    ):
        raise ValueError(
            "--path-step must be finite and positive"
        )
    if (
        not np.isfinite(arguments.hard_margin_target)
        or arguments.hard_margin_target <= 0.0
    ):
        raise ValueError(
            "--hard-margin-target must be finite and positive"
        )
    if int(arguments.contour_grid) < 3:
        raise ValueError(
            "--contour-grid must be at least three"
        )

    print(
        json.dumps(
            analyze(arguments),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
