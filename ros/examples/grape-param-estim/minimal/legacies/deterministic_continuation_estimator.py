#!/usr/bin/env python3
"""Continuation and one-dimensional delay-profile deterministic estimator.

The optimized model has fourteen coordinates: thirteen smooth physical
coordinates and one externally profiled recorded-command delay.  Thrust and
gimbal time constants remain fixed at their nominal values.  For every delay,
the physical solve follows the same local basin from short to long rollout
intervals.  Delay is never differentiated because the causal zero-order-hold
command lookup is piecewise constant in delay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from . import deterministic_estimator as baseline
from . import deterministic_sobol_estimator as analytic
from grape_param_estim.system import VehicleParameters


SCHEMA = "grape-param-estim/minimal-deterministic-continuation/v1"
OUTPUT_SUBDIRECTORY = "deterministic_continuation"
PHYSICAL_DIMENSION = 13
TOTAL_DIMENSION = PHYSICAL_DIMENSION + 1
PHYSICAL_PARAMETER_NAMES = analytic.SEARCH_PARAMETER_NAMES[:PHYSICAL_DIMENSION]
PARAMETER_NAMES = PHYSICAL_PARAMETER_NAMES + ("command_delay_seconds",)
FIXED_THRUST_TIME_CONSTANT = analytic.CURRENT_THRUST_TIME_CONSTANT
FIXED_GIMBAL_TIME_CONSTANT = analytic.CURRENT_GIMBAL_TIME_CONSTANT


def continuation_durations(
    interval_duration: float,
    requested: Sequence[float],
) -> np.ndarray:
    """Return increasing partial durations followed by the exact full span."""

    total = float(interval_duration)
    values = np.asarray(requested, dtype=float)
    if (
        not np.isfinite(total)
        or total <= 0.0
        or values.ndim != 1
        or values.size < 1
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("continuation durations must be finite and positive")
    partial = sorted({float(value) for value in values if value < total})
    return np.asarray(partial + [total], dtype=float)


def inclusive_delay_grid(
    lower: float,
    upper: float,
    step: float,
    *,
    required: Sequence[float] = (),
) -> np.ndarray:
    """Create a stable inclusive grid and insert explicitly required delays."""

    limits = np.asarray((lower, upper, step), dtype=float)
    required_values = np.asarray(required, dtype=float)
    if (
        np.any(~np.isfinite(limits))
        or lower < 0.0
        or upper <= lower
        or step <= 0.0
        or required_values.ndim != 1
        or np.any(~np.isfinite(required_values))
        or np.any(required_values < lower)
        or np.any(required_values > upper)
    ):
        raise ValueError("delay grid inputs are invalid")
    count = int(math.floor((upper - lower) / step + 1.0e-12))
    regular = lower + step * np.arange(count + 1, dtype=float)
    values = np.concatenate((regular, np.asarray((upper,)), required_values))
    rounded = np.round(values, decimals=12)
    return np.unique(np.clip(rounded, lower, upper))


def branch_order(grid: Sequence[float], anchor: float) -> tuple[tuple[float, ...], ...]:
    """Split a delay grid into adjacent warm-start branches from an anchor."""

    values = np.asarray(grid, dtype=float)
    selected_anchor = float(anchor)
    if (
        values.ndim != 1
        or values.size < 1
        or np.any(~np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
        or not np.any(np.isclose(values, selected_anchor, atol=1.0e-12, rtol=0.0))
    ):
        raise ValueError("delay branch inputs are invalid")
    anchor_value = float(
        values[np.flatnonzero(np.isclose(values, selected_anchor, atol=1.0e-12))[0]]
    )
    upper = tuple(float(value) for value in values if value > anchor_value)
    lower = tuple(float(value) for value in values[::-1] if value < anchor_value)
    return ((anchor_value,) + upper, (anchor_value,) + lower)


def _delay_key(delay: float) -> float:
    return round(float(delay), 12)


def _expand_coordinate(
    physical_coordinate: Sequence[float],
    delay: float,
) -> np.ndarray:
    physical = np.asarray(physical_coordinate, dtype=float)
    if (
        physical.shape != (PHYSICAL_DIMENSION,)
        or np.any(~np.isfinite(physical))
        or not np.isfinite(delay)
        or delay < 0.0
    ):
        raise ValueError("physical coordinate or delay is invalid")
    result = np.zeros(analytic.SEARCH_DIMENSION, dtype=float)
    result[:PHYSICAL_DIMENSION] = physical
    result[analytic.DELAY_INDEX] = float(delay)
    return result


def _physical_bounds(arguments: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    mass_min, mass_max = arguments.mass_scale_bounds
    diagonal_min, diagonal_max = (
        arguments.inertia_cholesky_diagonal_scale_bounds
    )
    off = float(arguments.inertia_cholesky_offdiagonal_bound)
    cog = float(arguments.cog_bound)
    effectiveness = float(arguments.force_effectiveness_contrast_bound)
    values = np.asarray(
        (
            mass_min,
            mass_max,
            diagonal_min,
            diagonal_max,
            off,
            cog,
            effectiveness,
        ),
        dtype=float,
    )
    if (
        np.any(~np.isfinite(values))
        or mass_min <= 0.0
        or mass_max <= mass_min
        or diagonal_min <= 0.0
        or diagonal_max <= diagonal_min
        or off <= 0.0
        or cog <= 0.0
        or effectiveness <= 0.0
    ):
        raise ValueError("physical parameter bounds are invalid")
    lower = np.asarray(
        (
            math.log(mass_min),
            math.log(diagonal_min),
            math.log(diagonal_min),
            math.log(diagonal_min),
            -off,
            -off,
            -off,
            -cog,
            -cog,
            -cog,
            -effectiveness,
            -effectiveness,
            -effectiveness,
        ),
        dtype=float,
    )
    upper = np.asarray(
        (
            math.log(mass_max),
            math.log(diagonal_max),
            math.log(diagonal_max),
            math.log(diagonal_max),
            off,
            off,
            off,
            cog,
            cog,
            cog,
            effectiveness,
            effectiveness,
            effectiveness,
        ),
        dtype=float,
    )
    return lower, upper


def _search_bounds(
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
    delay_bounds: Sequence[float],
) -> analytic.SearchBounds:
    delay_lower, delay_upper = (float(value) for value in delay_bounds)
    lower = np.concatenate(
        (
            physical_lower,
            np.asarray((-1.0, -1.0, delay_lower)),
        )
    )
    upper = np.concatenate(
        (
            physical_upper,
            np.asarray((1.0, 1.0, delay_upper)),
        )
    )
    return analytic.SearchBounds(lower, upper)


def _pid_gate(arguments: argparse.Namespace, current_gains: np.ndarray):
    minimum = arguments.pid_gain_min_scale
    maximum = arguments.pid_gain_max_scale
    if minimum is None and maximum is None:
        return analytic.PidGainGate.disabled(current_gains), (None, None)
    if minimum is None or maximum is None:
        raise ValueError(
            "PID gain minimum and maximum scales must be specified together"
        )
    return (
        analytic.PidGainGate.from_scale_band(current_gains, minimum, maximum),
        (float(minimum), float(maximum)),
    )


class _CachedPhysicalObjective:
    """Expose only the thirteen physical columns of the analytic Jacobian."""

    def __init__(
        self,
        evaluator: analytic.CandidateEvaluator,
        problem: baseline.DirectShootingProblem,
        delay: float,
    ) -> None:
        self.evaluator = evaluator
        self.problem = problem
        self.delay = float(delay)
        self.coordinate: Optional[np.ndarray] = None
        self.residual_value: Optional[np.ndarray] = None
        self.jacobian_value: Optional[np.ndarray] = None
        self.linearization_count = 0

    def _evaluate(self, coordinate: Sequence[float]) -> None:
        physical = np.asarray(coordinate, dtype=float)
        if self.coordinate is not None and np.array_equal(
            physical, self.coordinate
        ):
            return
        smooth = np.zeros(analytic.SMOOTH_DIMENSION, dtype=float)
        smooth[:PHYSICAL_DIMENSION] = physical
        residual, jacobian = self.evaluator.optimization_residual_and_jacobian(
            self.problem,
            smooth,
            self.delay,
        )
        self.coordinate = physical.copy()
        self.residual_value = residual
        self.jacobian_value = jacobian[:, :PHYSICAL_DIMENSION]
        self.linearization_count += 1

    def residual(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.residual_value is None:
            raise RuntimeError("physical residual cache was not populated")
        return self.residual_value

    def jacobian(self, coordinate: Sequence[float]) -> np.ndarray:
        self._evaluate(coordinate)
        if self.jacobian_value is None:
            raise RuntimeError("physical Jacobian cache was not populated")
        return self.jacobian_value


def _make_problem(
    flight: Any,
    delay: float,
    arguments: argparse.Namespace,
) -> baseline.DirectShootingProblem:
    return baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=float(delay),
        prior_weight=arguments.prior_weight,
    )


def _termination_category(result: Any) -> str:
    if result.success:
        return "converged_by_tolerance"
    if result.status == 0:
        return "function_evaluation_limit"
    return "optimizer_failure"


def _stage_record(
    *,
    duration: float,
    problem: baseline.DirectShootingProblem,
    start_coordinate: np.ndarray,
    start_evaluation: Mapping[str, Any],
    result: Any,
    selected_coordinate: np.ndarray,
    selected_evaluation: Mapping[str, Any],
    fallback: Optional[Mapping[str, Any]],
    objective: _CachedPhysicalObjective,
    max_nfev: int,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "duration_seconds": float(duration),
        "fitted_support_seconds": [
            float(problem.output_time[0]),
            float(problem.output_time[-1]),
        ],
        "sample_count": int(problem.output_time.size),
        "start_coordinate": start_coordinate,
        "start_trajectory_loss": (
            None
            if not start_evaluation.get("valid", False)
            else float(start_evaluation["trajectory_loss"])
        ),
        "coordinate": selected_coordinate,
        "trajectory_loss": float(selected_evaluation["trajectory_loss"]),
        "fallback": fallback,
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "termination_category": _termination_category(result),
            "max_nfev": int(max_nfev),
            "cost_with_prior_and_constraints": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "active_mask": np.asarray(result.active_mask, dtype=int),
            "analytic_linearization_count": objective.linearization_count,
            "elapsed_seconds": float(elapsed),
        },
    }


def _run_continuation(
    *,
    phase: str,
    delay: float,
    warm_start_delay: Optional[float],
    initial_coordinate: Sequence[float],
    stage_flights: Sequence[Any],
    durations: np.ndarray,
    evaluator: analytic.CandidateEvaluator,
    physical_bounds: tuple[np.ndarray, np.ndarray],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    coordinate = np.asarray(initial_coordinate, dtype=float).copy()
    stages: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for stage_index, (duration, flight) in enumerate(
        zip(durations, stage_flights), start=1
    ):
        problem = _make_problem(flight, delay, arguments)
        full_start = _expand_coordinate(coordinate, delay)
        start_evaluation = evaluator.evaluate(full_start, problem=problem)
        if not start_evaluation.get("valid", False):
            raise RuntimeError(
                "continuation warm start is invalid at delay {:.6g}: {}".format(
                    delay,
                    start_evaluation.get("reason", "unknown"),
                )
            )
        objective = _CachedPhysicalObjective(evaluator, problem, delay)
        stage_started = time.perf_counter()
        stage_max_nfev = (
            arguments.full_max_nfev
            if stage_index == len(durations)
            else arguments.max_nfev
        )
        result = least_squares(
            objective.residual,
            coordinate,
            bounds=physical_bounds,
            method="trf",
            jac=objective.jacobian,
            x_scale="jac",
            loss="linear",
            ftol=arguments.ftol,
            xtol=arguments.xtol,
            gtol=arguments.gtol,
            max_nfev=stage_max_nfev,
            verbose=0,
        )
        elapsed = time.perf_counter() - stage_started
        proposed_coordinate = np.asarray(result.x, dtype=float)
        proposed_evaluation = evaluator.evaluate(
            _expand_coordinate(proposed_coordinate, delay),
            problem=problem,
        )
        fallback = None
        if proposed_evaluation.get("valid", False):
            coordinate = proposed_coordinate
            selected_evaluation = proposed_evaluation
        else:
            selected_evaluation = start_evaluation
            fallback = {
                "used": True,
                "reason": proposed_evaluation.get("reason", "unknown"),
                "to": "stage_warm_start",
            }
        record = _stage_record(
            duration=duration,
            problem=problem,
            start_coordinate=np.asarray(full_start[:PHYSICAL_DIMENSION]),
            start_evaluation=start_evaluation,
            result=result,
            selected_coordinate=coordinate.copy(),
            selected_evaluation=selected_evaluation,
            fallback=fallback,
            objective=objective,
            max_nfev=stage_max_nfev,
            elapsed=elapsed,
        )
        stages.append(record)
        print(
            "{} delay={:.6f} stage {}/{} ({:.3f}s): loss={:.9g}, "
            "nfev={}, {}".format(
                phase,
                delay,
                stage_index,
                len(durations),
                duration,
                record["trajectory_loss"],
                result.nfev,
                record["optimizer"]["termination_category"],
            ),
            flush=True,
        )
    return {
        "phase": str(phase),
        "delay_seconds": float(delay),
        "warm_start_delay_seconds": warm_start_delay,
        "initial_coordinate": np.asarray(initial_coordinate, dtype=float),
        "coordinate": coordinate,
        "continuation_full_trajectory_loss": stages[-1]["trajectory_loss"],
        "stages": stages,
        "elapsed_seconds": time.perf_counter() - total_started,
    }


def _run_profile_branches(
    *,
    phase: str,
    grid: np.ndarray,
    anchor_delay: float,
    anchor_coordinate: np.ndarray,
    existing: dict[float, dict[str, Any]],
    stage_flights: Sequence[Any],
    durations: np.ndarray,
    evaluator: analytic.CandidateEvaluator,
    physical_bounds: tuple[np.ndarray, np.ndarray],
    arguments: argparse.Namespace,
) -> None:
    for branch_index, branch in enumerate(branch_order(grid, anchor_delay)):
        current_delay = float(branch[0])
        current_coordinate = anchor_coordinate.copy()
        anchor_record = existing.get(_delay_key(current_delay))
        if anchor_record is not None:
            current_coordinate = np.asarray(
                anchor_record["coordinate"], dtype=float
            ).copy()
        elif branch_index == 0:
            anchor_record = _run_continuation(
                phase=phase,
                delay=current_delay,
                warm_start_delay=None,
                initial_coordinate=current_coordinate,
                stage_flights=stage_flights,
                durations=durations,
                evaluator=evaluator,
                physical_bounds=physical_bounds,
                arguments=arguments,
            )
            existing[_delay_key(current_delay)] = anchor_record
            current_coordinate = np.asarray(
                anchor_record["coordinate"], dtype=float
            ).copy()
        for delay in branch[1:]:
            key = _delay_key(delay)
            if key in existing:
                current_delay = float(delay)
                current_coordinate = np.asarray(
                    existing[key]["coordinate"], dtype=float
                ).copy()
                continue
            record = _run_continuation(
                phase=phase,
                delay=float(delay),
                warm_start_delay=current_delay,
                initial_coordinate=current_coordinate,
                stage_flights=stage_flights,
                durations=durations,
                evaluator=evaluator,
                physical_bounds=physical_bounds,
                arguments=arguments,
            )
            existing[key] = record
            current_delay = float(delay)
            current_coordinate = np.asarray(record["coordinate"], dtype=float)


def _profile_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"replay_evaluation", "replay_problem"}
    }


def _write_delay_profile_pdf(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    selected_delay: float,
) -> None:
    ordered = sorted(records, key=lambda record: record["delay_seconds"])
    delays_ms = np.asarray(
        [1000.0 * float(record["delay_seconds"]) for record in ordered]
    )
    losses = np.asarray(
        [float(record["full_replay_trajectory_loss"]) for record in ordered]
    )
    phases = [str(record["phase"]) for record in ordered]
    figure, axis = baseline.plt.subplots(
        1,
        1,
        figsize=(11.7, 8.3),
        constrained_layout=True,
    )
    axis.plot(delays_ms, losses, color="#555555", linewidth=1.2)
    for phase, marker, color in (
        ("coarse", "o", "#1e5abe"),
        ("fine", "x", "#d2691e"),
    ):
        indices = [index for index, value in enumerate(phases) if value == phase]
        if indices:
            axis.scatter(
                delays_ms[indices],
                losses[indices],
                marker=marker,
                color=color,
                s=55,
                label="{} profile".format(phase),
                zorder=3,
            )
    selected_index = int(
        np.argmin(np.abs(delays_ms - 1000.0 * float(selected_delay)))
    )
    axis.scatter(
        [delays_ms[selected_index]],
        [losses[selected_index]],
        marker="*",
        color="#1e965f",
        s=180,
        label="selected",
        zorder=4,
    )
    axis.set_xlabel("recorded-command delay [ms]")
    axis.set_ylabel("full-trajectory loss")
    axis.set_title("Continuation delay profile")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    with baseline.PdfPages(path) as pdf:
        pdf.savefig(figure)
    baseline.plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit thirteen physical coordinates by trajectory-length "
            "continuation inside a coarse-to-fine delay profile search."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument(
        "--continuation-horizons",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
        metavar="SECONDS",
    )
    parser.add_argument("--max-nfev", type=int, default=35)
    parser.add_argument(
        "--full-max-nfev",
        type=int,
        default=80,
        help="function-evaluation limit for the final full-duration stage",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.08),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--nominal-delay", type=float, default=0.01)
    parser.add_argument("--coarse-delay-step", type=float, default=0.02)
    parser.add_argument("--fine-delay-step", type=float, default=0.0025)
    parser.add_argument(
        "--fine-delay-radius",
        type=float,
        default=None,
        help="half-width around the best coarse delay; defaults to coarse step",
    )
    parser.add_argument(
        "--mass-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--inertia-cholesky-diagonal-scale-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--inertia-cholesky-offdiagonal-bound", type=float, default=0.8
    )
    parser.add_argument("--cog-bound", type=float, default=0.10)
    parser.add_argument(
        "--force-effectiveness-contrast-bound", type=float, default=0.60
    )
    parser.add_argument("--pid-gain-min-scale", type=float, default=None)
    parser.add_argument("--pid-gain-max-scale", type=float, default=None)
    parser.add_argument("--pid-constraint-penalty", type=float, default=100.0)
    parser.add_argument(
        "--boundary-proximity-fraction", type=float, default=0.02
    )
    parser.add_argument(
        "--inertia-triangle-proximity-fraction",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    finite = (
        arguments.start,
        arguments.end,
        arguments.sample_step,
        arguments.integration_step,
        arguments.prior_weight,
        arguments.nominal_delay,
        arguments.coarse_delay_step,
        arguments.fine_delay_step,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.pid_constraint_penalty,
        arguments.boundary_proximity_fraction,
        arguments.inertia_triangle_proximity_fraction,
    )
    delay_bounds = np.asarray(arguments.delay_bounds, dtype=float)
    if (
        any(not np.isfinite(value) for value in finite)
        or arguments.start >= arguments.end
        or arguments.sample_step <= 0.0
        or arguments.integration_step <= 0.0
        or arguments.prior_weight < 0.0
        or arguments.max_nfev < 1
        or arguments.full_max_nfev < 1
        or arguments.nominal_delay < 0.0
        or arguments.coarse_delay_step <= 0.0
        or arguments.fine_delay_step <= 0.0
        or arguments.fine_delay_step >= arguments.coarse_delay_step
        or arguments.ftol <= 0.0
        or arguments.xtol <= 0.0
        or arguments.gtol <= 0.0
        or arguments.pid_constraint_penalty <= 0.0
        or delay_bounds.shape != (2,)
        or np.any(~np.isfinite(delay_bounds))
        or delay_bounds[0] < 0.0
        or delay_bounds[1] <= delay_bounds[0]
        or not delay_bounds[0] <= arguments.nominal_delay <= delay_bounds[1]
        or not 0.0 < arguments.boundary_proximity_fraction < 0.5
        or arguments.inertia_triangle_proximity_fraction <= 0.0
        or (
            arguments.fine_delay_radius is not None
            and (
                not np.isfinite(arguments.fine_delay_radius)
                or arguments.fine_delay_radius <= 0.0
            )
        )
    ):
        raise SystemExit("continuation/profile settings are invalid")
    try:
        durations = continuation_durations(
            arguments.end - arguments.start,
            arguments.continuation_horizons,
        )
        if durations[0] < 2.0 * arguments.sample_step:
            raise ValueError("first continuation horizon is too short")
        physical_bounds = _physical_bounds(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return physical_bounds


def run(arguments: argparse.Namespace) -> int:
    physical_bounds = _validate_arguments(arguments)
    bag = arguments.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit("bag does not exist: {}".format(bag))
    started = time.perf_counter()
    durations = continuation_durations(
        arguments.end - arguments.start,
        arguments.continuation_horizons,
    )
    print(
        "loading continuation horizons {} from {}".format(
            np.array2string(durations, precision=4),
            bag,
        ),
        flush=True,
    )
    stage_flights = [
        baseline.load_flight_data(
            str(bag),
            start_local=arguments.start,
            end_local=arguments.start + float(duration),
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        for duration in durations
    ]
    parameterization = analytic.PhysicalSearchParameterization(
        VehicleParameters.nominal(),
        current_thrust_time_constant=FIXED_THRUST_TIME_CONSTANT,
        current_gimbal_time_constant=FIXED_GIMBAL_TIME_CONSTANT,
    )
    try:
        pid_gate, pid_scales = _pid_gate(
            arguments,
            stage_flights[-1].controller_snapshot.gains,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bounds = _search_bounds(
        physical_bounds[0],
        physical_bounds[1],
        arguments.delay_bounds,
    )
    evaluator = analytic.CandidateEvaluator(
        flight=stage_flights[-1],
        parameterization=parameterization,
        pid_gate=pid_gate,
        bounds=bounds,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        prior_weight=arguments.prior_weight,
        pid_penalty_weight=arguments.pid_constraint_penalty,
    )
    coarse_grid = inclusive_delay_grid(
        arguments.delay_bounds[0],
        arguments.delay_bounds[1],
        arguments.coarse_delay_step,
        required=(arguments.nominal_delay,),
    )
    records: dict[float, dict[str, Any]] = {}
    nominal_coordinate = np.zeros(PHYSICAL_DIMENSION, dtype=float)
    print(
        "coarse delay grid [s]: {}".format(
            np.array2string(coarse_grid, precision=6)
        ),
        flush=True,
    )
    _run_profile_branches(
        phase="coarse",
        grid=coarse_grid,
        anchor_delay=arguments.nominal_delay,
        anchor_coordinate=nominal_coordinate,
        existing=records,
        stage_flights=stage_flights,
        durations=durations,
        evaluator=evaluator,
        physical_bounds=physical_bounds,
        arguments=arguments,
    )
    coarse_records = [
        records[_delay_key(delay)] for delay in coarse_grid
    ]
    best_coarse = min(
        coarse_records,
        key=lambda record: record["continuation_full_trajectory_loss"],
    )
    radius = (
        arguments.coarse_delay_step
        if arguments.fine_delay_radius is None
        else arguments.fine_delay_radius
    )
    fine_lower = max(
        float(arguments.delay_bounds[0]),
        float(best_coarse["delay_seconds"]) - radius,
    )
    fine_upper = min(
        float(arguments.delay_bounds[1]),
        float(best_coarse["delay_seconds"]) + radius,
    )
    fine_grid = inclusive_delay_grid(
        fine_lower,
        fine_upper,
        arguments.fine_delay_step,
        required=(best_coarse["delay_seconds"],),
    )
    print(
        "best coarse delay={:.6f}, fine grid [s]: {}".format(
            best_coarse["delay_seconds"],
            np.array2string(fine_grid, precision=6),
        ),
        flush=True,
    )
    _run_profile_branches(
        phase="fine",
        grid=fine_grid,
        anchor_delay=float(best_coarse["delay_seconds"]),
        anchor_coordinate=np.asarray(best_coarse["coordinate"], dtype=float),
        existing=records,
        stage_flights=stage_flights,
        durations=durations,
        evaluator=evaluator,
        physical_bounds=physical_bounds,
        arguments=arguments,
    )
    fine_keys = {_delay_key(delay) for delay in fine_grid}
    for key in fine_keys:
        if records[key]["phase"] != "coarse":
            records[key]["phase"] = "fine"

    full_flight = stage_flights[-1]
    valid_records: list[dict[str, Any]] = []
    for record in records.values():
        delay = float(record["delay_seconds"])
        problem = _make_problem(full_flight, delay, arguments)
        evaluation = evaluator.evaluate(
            _expand_coordinate(record["coordinate"], delay),
            problem=problem,
        )
        record["full_replay_valid"] = bool(evaluation.get("valid", False))
        record["full_replay_reason"] = str(
            evaluation.get("reason", "unknown")
        )
        record["full_replay_trajectory_loss"] = (
            None
            if not evaluation.get("valid", False)
            else float(evaluation["trajectory_loss"])
        )
        record["full_replay_metrics"] = evaluation.get("metrics")
        record["replay_evaluation"] = evaluation
        record["replay_problem"] = problem
        if evaluation.get("valid", False):
            valid_records.append(record)
    if not valid_records:
        raise RuntimeError("no valid full-trajectory profile result remains")
    selected = min(
        valid_records,
        key=lambda record: record["full_replay_trajectory_loss"],
    )
    selected_delay = float(selected["delay_seconds"])
    selected_evaluation = selected["replay_evaluation"]
    selected_problem = selected["replay_problem"]
    nominal_full_coordinate = _expand_coordinate(
        np.zeros(PHYSICAL_DIMENSION),
        selected_delay,
    )
    nominal_evaluation = evaluator.evaluate(
        nominal_full_coordinate,
        problem=selected_problem,
    )
    if not nominal_evaluation.get("valid", False):
        raise RuntimeError("nominal selected-delay replay is invalid")
    selected_coordinate = _expand_coordinate(
        selected["coordinate"],
        selected_delay,
    )
    selected_decoded = selected_evaluation["decoded"]
    selected_boundary = analytic._boundary_diagnostic(
        selected_coordinate,
        bounds,
        arguments.boundary_proximity_fraction,
        physical=selected_evaluation["physical"],
        pid=selected_evaluation["pid"],
        pid_gate=pid_gate,
        inertia_triangle_proximity_fraction=(
            arguments.inertia_triangle_proximity_fraction
        ),
    )

    output = arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    trajectory_pdf = output / "trajectory.pdf"
    delay_profile_pdf = output / "delay_profile.pdf"
    ordered_records = sorted(
        records.values(), key=lambda record: record["delay_seconds"]
    )
    elapsed = time.perf_counter() - started
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "path": str(bag),
            "sha256": baseline._sha256(bag),
            "requested_interval_seconds": [arguments.start, arguments.end],
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
        },
        "method": {
            "name": "trajectory_length_continuation_with_delay_profile",
            "total_dimension": TOTAL_DIMENSION,
            "inner_physical_dimension": PHYSICAL_DIMENSION,
            "physical_parameter_names": PHYSICAL_PARAMETER_NAMES,
            "profile_parameter": "command_delay_seconds",
            "initial_physical_coordinate": "exact nominal zero",
            "observation_resets": False,
            "delay_jacobian": None,
            "delay_search": "coarse grid followed by local fine grid",
            "physical_jacobian": (
                "analytic forward sensitivity through Cholesky/log chart, "
                "actuator active branches, RK4, and observation residuals"
            ),
            "continuation_durations_seconds": durations,
            "final_selection": (
                "minimum independently replayed full-trajectory loss over "
                "all coarse and fine delay candidates"
            ),
        },
        "fixed_parameters": {
            "thrust_time_constant_seconds": FIXED_THRUST_TIME_CONSTANT,
            "gimbal_time_constant_seconds": FIXED_GIMBAL_TIME_CONSTANT,
            "torque_effectiveness": [1.0] * 4,
            "linear_drag": [0.0] * 3,
            "angular_drag": [0.0] * 3,
        },
        "bounds": {
            "physical": {
                name: [float(lower), float(upper)]
                for name, lower, upper in zip(
                    PHYSICAL_PARAMETER_NAMES,
                    physical_bounds[0],
                    physical_bounds[1],
                )
            },
            "delay_seconds": [
                float(arguments.delay_bounds[0]),
                float(arguments.delay_bounds[1]),
            ],
        },
        "pid_gate": {
            "enabled": pid_gate.enabled,
            "lower_scale": pid_scales[0],
            "upper_scale": pid_scales[1],
            "current_gains": pid_gate.current,
            "lower_gains": pid_gate.lower,
            "upper_gains": pid_gate.upper,
        },
        "optimizer": {
            "name": "scipy.optimize.least_squares",
            "max_nfev_per_partial_stage": arguments.max_nfev,
            "max_nfev_for_full_stage": arguments.full_max_nfev,
            "ftol": arguments.ftol,
            "xtol": arguments.xtol,
            "gtol": arguments.gtol,
            "prior_weight": arguments.prior_weight,
        },
        "delay_profile": {
            "nominal_delay_seconds": arguments.nominal_delay,
            "coarse_step_seconds": arguments.coarse_delay_step,
            "coarse_grid_seconds": coarse_grid,
            "best_coarse_delay_seconds": best_coarse["delay_seconds"],
            "fine_step_seconds": arguments.fine_delay_step,
            "fine_radius_seconds": radius,
            "fine_grid_seconds": fine_grid,
            "records": [_profile_payload(record) for record in ordered_records],
        },
        "selection": {
            "selected_delay_seconds": selected_delay,
            "selected_phase": selected["phase"],
            "selected_physical_coordinate": selected["coordinate"],
            "selected_coordinate_14d": np.concatenate(
                (selected["coordinate"], np.asarray((selected_delay,)))
            ),
            "selected_trajectory_loss": selected[
                "full_replay_trajectory_loss"
            ],
            "selected_parameters": analytic._physical_payload(selected_decoded),
            "selected_pid_group_scales": selected_evaluation["pid"][
                "group_scales"
            ],
            "selected_pid_gains": selected_evaluation["pid"]["gains"],
            "selected_boundary": selected_boundary,
        },
        "recorded_control_open_loop_metrics": {
            "nominal_at_selected_delay": nominal_evaluation["metrics"],
            "selected": selected_evaluation["metrics"],
        },
        "elapsed_seconds": elapsed,
        "outputs": {
            "result_json": "result.json",
            "trajectory_pdf": "trajectory.pdf",
            "delay_profile_pdf": "delay_profile.pdf",
        },
    }
    baseline._write_pdf(
        trajectory_pdf,
        selected_problem,
        nominal_evaluation["simulation"],
        selected_evaluation["simulation"],
        nominal_evaluation["metrics"],
        selected_evaluation["metrics"],
    )
    _write_delay_profile_pdf(delay_profile_pdf, ordered_records, selected_delay)
    baseline._write_json(output / "result.json", payload)
    print(
        "selected delay={:.6f}s with full trajectory loss {:.9g}".format(
            selected_delay,
            selected["full_replay_trajectory_loss"],
        ),
        flush=True,
    )
    print("wrote {}".format(output / "result.json"), flush=True)
    print("wrote {}".format(trajectory_pdf), flush=True)
    print("wrote {}".format(delay_profile_pdf), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
