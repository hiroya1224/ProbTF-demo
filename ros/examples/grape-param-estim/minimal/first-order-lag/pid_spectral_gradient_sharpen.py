#!/usr/bin/env python3
"""Failure-started spectral-gradient continuation with automatic softmax sharpening.

The starting point is exactly the PID gain recorded in the failed bag.  No
previous safe-gain artifact, seed JSON, or saved safe cloud is read.

For the fixed ordered plant samples theta_i and normalized gain q,

    rho_i(q) = spectral radius of the exact sampled closed-loop Jacobian,

and the full N-by-12 finite-difference Jacobian d rho_i / d q_j is retained.
For tau > 0,

    R_tau(q) = tau log(mean(exp(rho_i(q)/tau))),
    delta_tau(q) = 1 - R_tau(q),

and

    grad delta_tau = -sum_i w_i grad rho_i.

At each temperature:
  1. project the current point normally onto delta_tau = 0,
  2. move toward the failed gain in the tangent space of delta_tau = 0,
  3. exact-forward correct back to the boundary,
  4. repeat until the tangential displacement toward failure is below the
     finite-difference coordinate resolution.

Then tau is halved.  The same rho_i and d rho_i/dq_j at the stage endpoint
immediately give the new softmax weights and the first sharpen correction.
Sharpening stops when

    tau log(N) / ||grad delta_tau|| <= fd_step,

so the worst-case softmax gap, converted to a local gain displacement, is no
larger than the coordinate scale used for the finite differences.

All acceptance/correction decisions use exact forward pole evaluations.
The linearized per-plant rho model is used only for predictors and for the
cheap 4 x 3 group contour diagnostic figure.
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


SCHEMA = "grape-param-estim/pid-spectral-gradient-sharpen/v2"
DEFAULT_PLANT_SAMPLES = 128
DEFAULT_SEED = 0
DEFAULT_TAU = 1.0e-4
DEFAULT_FD_STEP = 0.01
DEFAULT_PATH_STEP = 0.15
DEFAULT_CONTOUR_GRID = 81
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)

GROUPS = ("xy", "z", "roll_pitch", "yaw")
PROJECTIONS = (
    ("PI", 0, 1),
    ("ID", 1, 2),
    ("DP", 2, 0),
)


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


def _point_payload(
    q: np.ndarray,
    summary: Mapping[str, Any],
    tau: float,
) -> Mapping[str, Any]:
    selected = np.asarray(q, dtype=float)
    return {
        "q": selected.tolist(),
        "gain_matrix": (
            selected.reshape(4, 3) * GAIN_CAPS
        ).tolist(),
        "tau": float(tau),
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
    delta_soft: float,
    gradient: np.ndarray,
) -> np.ndarray:
    g = np.asarray(gradient, dtype=float)
    norm_sq = float(np.dot(g, g))
    if norm_sq == 0.0:
        raise RuntimeError(
            "soft-margin gradient vanishes, so the local boundary normal is undefined"
        )
    return (-float(delta_soft) / norm_sq) * g


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
    gradient: np.ndarray,
    *,
    progress: bool,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Correct q toward delta_tau=0 along a supplied local normal.

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
    previous_abs_delta = None
    while True:
        row = evaluator.evaluate(current)
        if not np.all(row.pole_valid_mask):
            return current, {}, {
                "resolved": False,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }

        summary = _softmax_summary(row.spectral_radius, tau)
        delta = float(summary["delta_soft"])
        if delta == 0.0:
            return current, summary, {
                "resolved": True,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }

        correction = (-delta / norm_sq) * g
        candidate = np.clip(current + correction, 0.0, 1.0)
        if np.array_equal(candidate, current):
            return current, summary, {
                "resolved": True,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }

        candidate_row = evaluator.evaluate(candidate)
        if np.all(candidate_row.pole_valid_mask):
            candidate_summary = _softmax_summary(
                candidate_row.spectral_radius,
                tau,
            )
            candidate_abs = abs(
                float(candidate_summary["delta_soft"])
            )
            current_abs = abs(delta)
            if candidate_abs < current_abs:
                current = candidate
                corrections += 1
                previous_abs_delta = candidate_abs
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
                    "resolved": True,
                    "normal_updates": int(corrections),
                    "backtracks": int(backtracks),
                }
            backtracked_row = evaluator.evaluate(backtracked)
            if np.all(backtracked_row.pole_valid_mask):
                backtracked_summary = _softmax_summary(
                    backtracked_row.spectral_radius,
                    tau,
                )
                if abs(
                    float(backtracked_summary["delta_soft"])
                ) < abs(delta):
                    current = backtracked
                    corrections += 1
                    previous_abs_delta = abs(
                        float(backtracked_summary["delta_soft"])
                    )
                    accepted = True
                    break
            scale *= 0.5

        if not accepted:
            return current, summary, {
                "resolved": True,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }

        if (
            previous_abs_delta is not None
            and previous_abs_delta == 0.0
        ):
            row = evaluator.evaluate(current)
            return current, _softmax_summary(
                row.spectral_radius, tau
            ), {
                "resolved": True,
                "normal_updates": int(corrections),
                "backtracks": int(backtracks),
            }


def _project_failure_to_boundary(
    evaluator: BatchedForwardMarginEvaluator,
    failure_q: np.ndarray,
    tau: float,
    fd_step: float,
    *,
    progress: bool,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Start exactly at failure and Newton-project to the soft boundary."""

    current = np.asarray(failure_q, dtype=float).copy()
    total_corrections = 0
    total_backtracks = 0
    predictor_diagnostics = None
    failure_gradient = None
    failure_fd_coverage = None
    failure_gradient_weight_coverage = None
    failure_fd_scheme_counts = None

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
        }

    bar = tqdm(
        desc="Failure -> soft boundary",
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
            delta = float(summary["delta_soft"])
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

            if delta == 0.0:
                return current, bundle, diagnostics()

            normal_step = _normal_newton_step(
                delta,
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
                            delta + np.dot(gradient, normal_step)
                        ),
                        "exact_residual": float(
                            proposal_summary["delta_soft"]
                        ),
                        "exact_residual_reduced": bool(
                            abs(float(proposal_summary["delta_soft"]))
                            < abs(delta)
                        ),
                    }
                if abs(
                    float(proposal_summary["delta_soft"])
                ) < abs(delta):
                    current = proposal
                    total_corrections += 1
                    bar.update(1)
                    if progress:
                        bar.set_postfix(
                            delta=f"{proposal_summary['delta_soft']:.3e}"
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
                    if abs(
                        float(candidate_summary["delta_soft"])
                    ) < abs(delta):
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
    fd_step: float,
    path_step: float,
    *,
    progress: bool,
) -> tuple[np.ndarray, list[Mapping[str, Any]], Mapping[str, Any]]:
    """Follow delta_tau=0 while decreasing distance to the failed gain."""

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
            if float(summary["delta_soft"]) != 0.0:
                projected, _projected_summary, projection = (
                    _exact_normal_projection(
                        evaluator,
                        current,
                        tau,
                        gradient,
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
                }

            step = min(
                float(path_step),
                tangent_norm,
            )
            direction = tangent / tangent_norm
            trial_scale = 1.0
            accepted = False

            while True:
                predictor = np.clip(
                    current
                    + trial_scale * step * direction,
                    0.0,
                    1.0,
                )
                if np.array_equal(predictor, current):
                    return current, history, {
                        "total_backtracks": int(total_backtracks),
                        "total_normal_updates": int(total_normal_updates),
                        "total_normal_backtracks": int(total_normal_backtracks),
                        "terminal_tangent_norm": tangent_norm,
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
                            gradient,
                            progress=progress,
                        )
                    )
                    total_normal_updates += int(correction["normal_updates"])
                    total_normal_backtracks += int(correction["backtracks"])
                    if bool(correction["resolved"]):
                        new_distance = float(
                            np.linalg.norm(
                                corrected - failure_q
                            )
                        )
                        if new_distance < distance:
                            # Accept the exact-forward corrected point.  It
                            # need not have delta exactly zero; the next fresh
                            # gradient bundle reprojects as necessary.
                            current = corrected
                            accepted = True
                            bar.update(1)
                            if progress:
                                bar.set_postfix(
                                    distance=f"{new_distance:.4f}",
                                    delta=f"{corrected_summary['delta_soft']:.2e}",
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
                }
    finally:
        bar.close()


def _linearized_group_contours(
    output: Path,
    final_q: np.ndarray,
    failure_q: np.ndarray,
    rho: np.ndarray,
    rho_jacobian: np.ndarray,
    tau_values: Sequence[float],
    final_gradient: np.ndarray,
    fd_step: float,
    grid_size: int,
) -> None:
    """Draw cheap local PI/ID/DP soft-margin views for every PID group."""

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

            tau_sequence = list(dict.fromkeys(
                float(value) for value in tau_values
            ))
            for tau_index, tau in enumerate(tau_sequence):
                scaled = predicted / tau
                soft_max = tau * (
                    logsumexp(scaled, axis=0)
                    - math.log(float(rho0.size))
                )
                delta = 1.0 - soft_max
                linestyle = "-" if tau_index == len(tau_sequence) - 1 else ":"
                linewidth = 2.0 if tau_index == len(tau_sequence) - 1 else 1.0
                if np.nanmin(delta) <= 0.0 <= np.nanmax(delta):
                    axis.contour(
                        qa * GAIN_CAPS.reshape(-1)[a],
                        qb * GAIN_CAPS.reshape(-1)[b],
                        delta.T,
                        levels=[0.0],
                        linestyles=linestyle,
                        linewidths=linewidth,
                    )

            if np.nanmin(hard_delta) <= 0.0 <= np.nanmax(hard_delta):
                axis.contour(
                    qa * GAIN_CAPS.reshape(-1)[a],
                    qb * GAIN_CAPS.reshape(-1)[b],
                    hard_delta.T,
                    levels=[0.0],
                    linestyles="--",
                    linewidths=1.5,
                )

            axis.scatter(
                [q0[a] * GAIN_CAPS.reshape(-1)[a]],
                [q0[b] * GAIN_CAPS.reshape(-1)[b]],
                marker="^",
                s=55,
                label="final",
            )
            axis.scatter(
                [qf[a] * GAIN_CAPS.reshape(-1)[a]],
                [qf[b] * GAIN_CAPS.reshape(-1)[b]],
                marker="o",
                s=42,
                label="failure",
            )

            g2 = np.asarray(
                (final_gradient[a], final_gradient[b]),
                dtype=float,
            )
            if np.linalg.norm(g2) > 0.0:
                tangent2 = np.asarray(
                    (-g2[1], g2[0]),
                    dtype=float,
                )
                tangent2 /= np.linalg.norm(tangent2)
                center_x = q0[a] * GAIN_CAPS.reshape(-1)[a]
                center_y = q0[b] * GAIN_CAPS.reshape(-1)[b]
                dx = 0.12 * (a_max - a_min) * GAIN_CAPS.reshape(-1)[a] * tangent2[0]
                dy = 0.12 * (b_max - b_min) * GAIN_CAPS.reshape(-1)[b] * tangent2[1]
                axis.plot(
                    [center_x - dx, center_x + dx],
                    [center_y - dy, center_y + dy],
                    linestyle="-.",
                    linewidth=1.0,
                )

            axis.set_xlabel(GAIN_LABELS[a])
            axis.set_ylabel(GAIN_LABELS[b])
            axis.set_title(f"{group} {projection_name}")
            axis.grid(True, alpha=0.2)
            if group_index == 0 and column == 0:
                axis.legend(fontsize=8)

    figure.suptitle(
        "Local spectral-radius contours: soft boundaries sharpen toward hard max\n"
        "solid = final tau, dotted = earlier tau, dashed = linearized hard-max boundary",
        fontsize=13,
    )
    figure.savefig(
        output / "spectral_gradient_group_contours.png",
        dpi=180,
    )
    plt.close(figure)


def _plot_path(
    output: Path,
    failure_q: np.ndarray,
    stage_records: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.5),
        constrained_layout=True,
    )

    axis = axes[0]
    for stage in stage_records:
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
        axis.plot(
            distance,
            soft,
            marker="o",
            label=f"soft tau={stage['tau']:.1e}",
        )
        axis.plot(
            distance,
            hard,
            linestyle="--",
            marker="x",
            alpha=0.6,
            label=(
                "hard-max margin"
                if stage is stage_records[0] else None
            ),
        )
    axis.axhline(0.0)
    axis.set_xlabel("normalized 12-D distance to failure")
    axis.set_ylabel("margin")
    axis.set_title("Boundary continuation and sharpening")
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
            label=f"tau={stage['tau']:.1e}",
        )
    axis.axhline(1.0)
    axis.set_xlabel("empirical quantile")
    axis.set_ylabel("spectral radius")
    axis.set_title("Order-statistic curve at each sharpened boundary")
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
                label=f"tau={stage['tau']:.1e}",
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
    plt.close(figure)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    progress = not bool(arguments.no_progress)

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
        failure_row = evaluator.evaluate(
            failure_q
        )
        if not np.all(
            failure_row.pole_valid_mask
        ):
            raise RuntimeError(
                "recorded failure gain has unresolved plant poles; "
                "the failure-started rho Jacobian is undefined"
            )

        tau = float(arguments.tau)
        current_q, initial_bundle, initial_projection = (
            _project_failure_to_boundary(
                evaluator,
                failure_q,
                tau,
                float(arguments.fd_step),
                progress=progress,
            )
        )

        stage_records: list[Mapping[str, Any]] = []
        tau_values: list[float] = []

        while True:
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

            if float(boundary_summary["delta_soft"]) != 0.0:
                projected, _summary, stage_projection = _exact_normal_projection(
                    evaluator,
                    current_q,
                    tau,
                    boundary_gradient,
                    progress=progress,
                )
                current_q = projected

            current_q, history, tangent_diagnostics = (
                _walk_tangent_to_nearest_failure(
                    evaluator,
                    current_q,
                    failure_q,
                    tau,
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
            gradient_norm = float(
                np.linalg.norm(final_gradient)
            )
            if gradient_norm == 0.0:
                sharpen_distance_bound = float("inf")
            else:
                sharpen_distance_bound = float(
                    tau
                    * math.log(float(arguments.samples))
                    / gradient_norm
                )

            stage_records.append({
                "tau": float(tau),
                "gap_bound": float(
                    tau
                    * math.log(float(arguments.samples))
                ),
                "gain_distance_gap_bound": sharpen_distance_bound,
                "history": history,
                "normal_projection": dict(stage_projection),
                "tangent_diagnostics": dict(
                    tangent_diagnostics
                ),
                "endpoint": {
                    **_point_payload(
                        current_q,
                        final_summary,
                        tau,
                    ),
                    "distance_to_failure": float(
                        np.linalg.norm(
                            current_q
                            - failure_q
                        )
                    ),
                    "gradient_norm": gradient_norm,
                },
            })
            tau_values.append(
                float(tau)
            )

            if (
                sharpen_distance_bound
                <= float(arguments.fd_step)
            ):
                break

            tau *= 0.5

            # Sharpen at exactly the current q using the *same* retained
            # rho_i and J_rho first; no new finite differences are needed to
            # compute the new weights and first normal predictor.
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
                    float(
                        sharper_summary["delta_soft"]
                    ),
                    sharper_gradient,
                ),
                0.0,
                1.0,
            )
            proposal_row = evaluator.evaluate(
                proposal
            )
            if np.all(
                proposal_row.pole_valid_mask
            ):
                current_q = proposal
            else:
                # If the one-shot sharpen predictor leaves the resolved
                # branch, keep the current point.  The next stage immediately
                # recomputes its exact gradient and projects from there.
                current_q = current_q.copy()

        final_bundle = _gradient_bundle(
            evaluator,
            current_q,
            tau,
            float(arguments.fd_step),
            progress=progress,
        )
        final_summary = final_bundle["summary"]

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

        _plot_path(
            output,
            failure_q,
            stage_records,
        )
        _linearized_group_contours(
            output,
            current_q,
            failure_q,
            final_bundle["row"].spectral_radius,
            final_bundle["rho_jacobian"],
            tau_values,
            final_bundle["gradient"],
            float(arguments.fd_step),
            int(arguments.contour_grid),
        )

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
                stage_tau=np.asarray(
                    [item[2]["tau"] for item in flat_history],
                    dtype=float,
                ),
                path_q=np.asarray(
                    [item[3]["q"] for item in flat_history],
                    dtype=float,
                ),
                path_rho=np.asarray(
                    [item[3]["rho"] for item in flat_history],
                    dtype=float,
                ),
                path_sorted_rho=np.asarray(
                    [item[3]["sorted_rho"] for item in flat_history],
                    dtype=float,
                ),
                path_rho_order=np.asarray(
                    [item[3]["rho_order"] for item in flat_history],
                    dtype=int,
                ),
                path_rho_jacobian=np.asarray(
                    [item[3]["rho_jacobian"] for item in flat_history],
                    dtype=float,
                ),
                path_softmax_weights=np.asarray(
                    [item[3]["weights"] for item in flat_history],
                    dtype=float,
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
                path_fd_difference_scheme=np.asarray(
                    [
                        item[3]["fd_difference_scheme"]
                        for item in flat_history
                    ],
                    dtype=np.uint8,
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
                "sample_count": int(
                    arguments.samples
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
                    "failure normal projection -> soft-boundary tangent "
                    "continuation -> tau sharpening"
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
                    "only exact evaluations already computed during this "
                    "run are cached to avoid duplicate work"
                ),
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
                    "tau": float(stage["tau"]),
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
            },
            "contours": {
                "figure": (
                    "4x3 group PI/ID/DP views from the final per-plant "
                    "linear rho model"
                ),
                "solid": "final-tau soft boundary",
                "dotted": "earlier-tau soft boundaries",
                "dashed": "linearized hard-max boundary",
                "dash_dot": "final local tangent direction",
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
