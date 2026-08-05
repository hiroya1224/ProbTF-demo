#!/usr/bin/env python3
"""Local parallel-tempering initializer for deterministic trajectory fitting.

The method deliberately does not scatter points across the full physical box.
It first runs the nominal-start deterministic baseline, builds a Gauss--Newton
proposal geometry at that result, calibrates a temperature ladder from local
loss increments, and explores a broad Gaussian neighbourhood with replica
exchange.  Every proposal costs one recorded-control forward rollout.  Only
the best point found across the history is refined with analytic
least-squares.

The fifteen smooth coordinates and analytic forward sensitivities are shared
with ``deterministic_sobol_estimator``.  Command delay remains fixed because a
causal ZOH command history is not differentiable with respect to delay.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

import deterministic_estimator as baseline
import deterministic_sobol_estimator as sobol
from grape_param_estim.system import VehicleParameters


SCHEMA = "grape-param-estim/minimal-deterministic-tempered/v1"
OUTPUT_SUBDIRECTORY = "deterministic_tempered"
DEFAULT_LOCAL_PRIOR_SCALES = (
    0.20,
    0.20,
    0.20,
    0.20,
    0.20,
    0.20,
    0.20,
    0.025,
    0.025,
    0.025,
    0.15,
    0.15,
    0.15,
    0.35,
    0.35,
)


def _local_bounds(
    scales: np.ndarray,
    radius: float,
    command_delay: float,
) -> sobol.SearchBounds:
    smooth_lower = -float(radius) * scales
    smooth_upper = float(radius) * scales
    delay_upper = max(0.08, 2.0 * float(command_delay) + 1.0e-6)
    return sobol.SearchBounds(
        np.concatenate((smooth_lower, np.asarray((0.0,)))),
        np.concatenate((smooth_upper, np.asarray((delay_upper,)))),
    )


def _pid_gate(arguments: argparse.Namespace, current_gains: np.ndarray):
    minimum = arguments.pid_gain_min_scale
    maximum = arguments.pid_gain_max_scale
    if minimum is None and maximum is None:
        return sobol.PidGainGate.disabled(current_gains), (None, None)
    if minimum is None or maximum is None:
        raise ValueError(
            "PID gain minimum and maximum scales must be specified together"
        )
    return (
        sobol.PidGainGate.from_scale_band(current_gains, minimum, maximum),
        (float(minimum), float(maximum)),
    )


def proposal_geometry_from_jacobian(
    jacobian: np.ndarray,
    local_scales: Sequence[float],
    *,
    damping: float,
    minimum_variance: float,
    maximum_variance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct a clipped ridge-aligned proposal square root from ``J``."""

    value = np.asarray(jacobian, dtype=float)
    scales = np.asarray(local_scales, dtype=float)
    if (
        value.ndim != 2
        or value.shape[1] != sobol.SMOOTH_DIMENSION
        or scales.shape != (sobol.SMOOTH_DIMENSION,)
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.isfinite(damping)
        or damping <= 0.0
        or not np.isfinite(minimum_variance)
        or not np.isfinite(maximum_variance)
        or minimum_variance <= 0.0
        or maximum_variance < minimum_variance
    ):
        raise ValueError("proposal geometry inputs are invalid")
    normalized_jacobian = value * scales[None, :]
    _left, singular_values, right_transpose = np.linalg.svd(
        normalized_jacobian,
        full_matrices=True,
    )
    raw_variances = np.full(
        sobol.SMOOTH_DIMENSION,
        1.0 / damping,
        dtype=float,
    )
    raw_variances[: singular_values.size] = 1.0 / (
        singular_values**2 + damping
    )
    variances = np.clip(
        raw_variances,
        minimum_variance,
        maximum_variance,
    )
    right = right_transpose.T
    normalized_root = right @ np.diag(np.sqrt(variances))
    root = scales[:, None] * normalized_root
    return root, {
        "singular_values": singular_values,
        "raw_normalized_variances": raw_variances,
        "clipped_normalized_variances": variances,
        "minimum_variance": minimum_variance,
        "maximum_variance": maximum_variance,
        "damping": damping,
        "largest_to_smallest_singular_value_ratio": float(
            singular_values[0]
            / max(singular_values[-1], np.finfo(float).eps)
        ),
    }


def calibrate_temperature_ladder(
    center_loss: float,
    pilot_losses: Sequence[float],
    *,
    replica_count: int,
    low_typical_uphill_acceptance: float,
    high_typical_uphill_acceptance: float,
    minimum_override: Optional[float] = None,
    maximum_override: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose loss-scale temperatures from positive pilot loss increments."""

    losses = np.asarray(pilot_losses, dtype=float)
    if (
        not np.isfinite(center_loss)
        or losses.ndim != 1
        or losses.size < 1
        or np.any(~np.isfinite(losses))
        or replica_count < 2
        or not 0.0 < low_typical_uphill_acceptance < 1.0
        or not 0.0 < high_typical_uphill_acceptance < 1.0
        or low_typical_uphill_acceptance
        >= high_typical_uphill_acceptance
    ):
        raise ValueError("temperature calibration inputs are invalid")
    increments = losses - float(center_loss)
    positive = increments[increments > 0.0]
    if positive.size:
        typical = float(np.median(positive))
        source = "median_positive_pilot_loss_increment"
    else:
        nonzero = np.abs(increments[np.abs(increments) > 0.0])
        typical = (
            float(np.median(nonzero))
            if nonzero.size
            else max(0.01 * abs(float(center_loss)), 1.0)
        )
        source = "fallback_absolute_pilot_loss_increment"
    minimum = typical / -math.log(low_typical_uphill_acceptance)
    maximum = typical / -math.log(high_typical_uphill_acceptance)
    if minimum_override is not None:
        minimum = float(minimum_override)
    if maximum_override is not None:
        maximum = float(maximum_override)
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError("temperature overrides must be positive and increasing")
    temperatures = np.geomspace(minimum, maximum, replica_count)
    return temperatures, {
        "typical_loss_increment": typical,
        "typical_increment_source": source,
        "positive_increment_count": int(positive.size),
        "low_typical_uphill_acceptance": low_typical_uphill_acceptance,
        "high_typical_uphill_acceptance": high_typical_uphill_acceptance,
        "minimum_temperature": minimum,
        "maximum_temperature": maximum,
    }


def replica_swap_log_acceptance(
    lower_loss: float,
    upper_loss: float,
    lower_temperature: float,
    upper_temperature: float,
) -> float:
    """Return the exact adjacent-replica exchange log acceptance ratio."""

    values = np.asarray(
        (lower_loss, upper_loss, lower_temperature, upper_temperature),
        dtype=float,
    )
    if (
        np.any(~np.isfinite(values))
        or lower_temperature <= 0.0
        or upper_temperature <= lower_temperature
    ):
        raise ValueError("replica exchange inputs are invalid")
    return float(
        (1.0 / lower_temperature - 1.0 / upper_temperature)
        * (lower_loss - upper_loss)
    )


def _local_prior_energy(
    coordinate: Sequence[float],
    scales: np.ndarray,
    strength: float,
) -> float:
    value = np.asarray(coordinate, dtype=float)
    return 0.5 * float(strength) * float(
        (value / scales) @ (value / scales)
    )


def _compact_evaluation(
    coordinate: np.ndarray,
    source: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "coordinate": np.asarray(coordinate, dtype=float).copy(),
        "source": str(source),
        "valid": bool(evaluation.get("valid", False)),
        "reason": str(evaluation.get("reason", "unknown")),
    }
    if evaluation.get("valid", False):
        record["trajectory_loss"] = float(evaluation["trajectory_loss"])
        record["normalized_inertia_triangle_margin"] = float(
            evaluation["physical"]["normalized_inertia_triangle_margin"]
        )
        record["pid_group_scales"] = evaluation["pid"]["group_scales"]
    elif "detail" in evaluation:
        record["detail"] = str(evaluation["detail"])
    return record


def _evaluate_batch(
    evaluator: sobol.CandidateEvaluator,
    problems: Sequence[baseline.DirectShootingProblem],
    coordinates: Sequence[np.ndarray],
    sources: Sequence[str],
    executor: Optional[ThreadPoolExecutor],
) -> list[dict[str, Any]]:
    if len(coordinates) != len(sources) or len(coordinates) > len(problems):
        raise ValueError("parallel evaluation batch has inconsistent sizes")

    def evaluate(index: int) -> dict[str, Any]:
        result = evaluator.evaluate(
            coordinates[index],
            problem=problems[index],
        )
        return _compact_evaluation(
            coordinates[index], sources[index], result
        )

    if executor is None:
        return [evaluate(index) for index in range(len(coordinates))]
    return list(executor.map(evaluate, range(len(coordinates))))


def _pilot_local_proposals(
    evaluator: sobol.CandidateEvaluator,
    center: np.ndarray,
    proposal_root: np.ndarray,
    center_loss: float,
    arguments: argparse.Namespace,
    rng: np.random.Generator,
    problems: Sequence[baseline.DirectShootingProblem],
    executor: Optional[ThreadPoolExecutor],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    valid: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    attempts = 0
    maximum_attempts = arguments.pilot_count * arguments.pilot_attempt_factor
    while len(valid) < arguments.pilot_count and attempts < maximum_attempts:
        batch_size = min(
            len(problems),
            maximum_attempts - attempts,
            arguments.pilot_count - len(valid) + len(problems) - 1,
        )
        candidates = [
            center
            + arguments.proposal_scale
            * (proposal_root @ rng.standard_normal(sobol.SMOOTH_DIMENSION))
            for _ in range(batch_size)
        ]
        full_coordinates = [
            sobol._join_smooth_coordinate(value, arguments.command_delay)
            for value in candidates
        ]
        records = _evaluate_batch(
            evaluator,
            problems,
            full_coordinates,
            ["pilot_{:05d}".format(attempts + index) for index in range(batch_size)],
            executor,
        )
        attempts += batch_size
        for record in records:
            if record["valid"]:
                valid.append(record)
                if len(valid) >= arguments.pilot_count:
                    break
            else:
                reason = record["reason"]
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    if len(valid) < arguments.pilot_count:
        raise RuntimeError(
            "pilot could not obtain {} valid local proposals in {} attempts".format(
                arguments.pilot_count,
                maximum_attempts,
            )
        )
    losses = [record["trajectory_loss"] for record in valid]
    print(
        "pilot: valid={}, attempts={}, best_loss={:.9g}, center_loss={:.9g}".format(
            len(valid),
            attempts,
            min(losses),
            center_loss,
        ),
        flush=True,
    )
    return valid, rejection_counts


def _top_diverse_records(
    records: Sequence[Mapping[str, Any]],
    scales: np.ndarray,
    *,
    count: int,
    minimum_distance: float,
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        (record for record in records if record.get("valid", False)),
        key=lambda record: float(record["trajectory_loss"]),
    )
    selected: list[Mapping[str, Any]] = []
    normalized: list[np.ndarray] = []
    for record in ordered:
        value = np.asarray(record["coordinate"], dtype=float)[
            : sobol.SMOOTH_DIMENSION
        ] / scales
        if all(
            np.linalg.norm(value - previous) >= minimum_distance
            for previous in normalized
        ):
            selected.append(record)
            normalized.append(value)
            if len(selected) >= count:
                break
    return selected


def _run_parallel_tempering(
    evaluator: sobol.CandidateEvaluator,
    center_record: Mapping[str, Any],
    pilot_records: Sequence[Mapping[str, Any]],
    proposal_root: np.ndarray,
    temperatures: np.ndarray,
    local_scales: np.ndarray,
    arguments: argparse.Namespace,
    rng: np.random.Generator,
    problems: Sequence[baseline.DirectShootingProblem],
    executor: Optional[ThreadPoolExecutor],
) -> dict[str, Any]:
    replica_count = temperatures.size
    initial_pool = sorted(
        [center_record] + list(pilot_records),
        key=lambda record: float(record["trajectory_loss"]),
    )
    pool_indices = np.rint(
        np.linspace(0, len(initial_pool) - 1, replica_count)
    ).astype(int)
    coordinates = np.asarray(
        [
            np.asarray(initial_pool[index]["coordinate"], dtype=float)[
                : sobol.SMOOTH_DIMENSION
            ]
            for index in pool_indices
        ]
    )
    losses = np.asarray(
        [float(initial_pool[index]["trajectory_loss"]) for index in pool_indices]
    )
    prior_energies = np.asarray(
        [
            _local_prior_energy(
                coordinate,
                local_scales,
                arguments.local_prior_strength,
            )
            for coordinate in coordinates
        ]
    )
    proposal_attempts = np.zeros(replica_count, dtype=int)
    proposal_accepts = np.zeros(replica_count, dtype=int)
    invalid_counts: dict[str, int] = {}
    swap_attempts = np.zeros(replica_count - 1, dtype=int)
    swap_accepts = np.zeros(replica_count - 1, dtype=int)
    loss_trace = [losses.copy()]
    accepted_trace = []
    swap_events: list[dict[str, Any]] = []
    evaluated_records: list[Mapping[str, Any]] = list(initial_pool)
    temperature_proposal_scales = arguments.proposal_scale * np.minimum(
        np.sqrt(temperatures / temperatures[0]),
        arguments.maximum_temperature_proposal_scale,
    )

    for sweep in range(arguments.sweeps):
        proposals = [
            coordinates[index]
            + temperature_proposal_scales[index]
            * (proposal_root @ rng.standard_normal(sobol.SMOOTH_DIMENSION))
            for index in range(replica_count)
        ]
        full_proposals = [
            sobol._join_smooth_coordinate(value, arguments.command_delay)
            for value in proposals
        ]
        records = _evaluate_batch(
            evaluator,
            problems,
            full_proposals,
            [
                "tempered_s{:04d}_r{:02d}".format(sweep, index)
                for index in range(replica_count)
            ],
            executor,
        )
        accepted = np.zeros(replica_count, dtype=bool)
        for index, record in enumerate(records):
            proposal_attempts[index] += 1
            if not record["valid"]:
                reason = record["reason"]
                invalid_counts[reason] = invalid_counts.get(reason, 0) + 1
                continue
            evaluated_records.append(record)
            candidate = proposals[index]
            candidate_loss = float(record["trajectory_loss"])
            candidate_prior = _local_prior_energy(
                candidate,
                local_scales,
                arguments.local_prior_strength,
            )
            log_acceptance = (
                -(candidate_loss - losses[index]) / temperatures[index]
                - (candidate_prior - prior_energies[index])
            )
            if math.log(rng.random()) < min(0.0, log_acceptance):
                coordinates[index] = candidate
                losses[index] = candidate_loss
                prior_energies[index] = candidate_prior
                proposal_accepts[index] += 1
                accepted[index] = True

        sweep_swaps = []
        if (sweep + 1) % arguments.exchange_interval == 0:
            phase = ((sweep + 1) // arguments.exchange_interval - 1) % 2
            for lower in range(phase, replica_count - 1, 2):
                upper = lower + 1
                swap_attempts[lower] += 1
                log_acceptance = replica_swap_log_acceptance(
                    losses[lower],
                    losses[upper],
                    temperatures[lower],
                    temperatures[upper],
                )
                swap_accepted = bool(
                    math.log(rng.random()) < min(0.0, log_acceptance)
                )
                if swap_accepted:
                    coordinates[[lower, upper]] = coordinates[[upper, lower]]
                    losses[[lower, upper]] = losses[[upper, lower]]
                    prior_energies[[lower, upper]] = prior_energies[
                        [upper, lower]
                    ]
                    swap_accepts[lower] += 1
                sweep_swaps.append(
                    {
                        "lower_replica": lower,
                        "upper_replica": upper,
                        "accepted": swap_accepted,
                        "log_acceptance": log_acceptance,
                    }
                )
            swap_events.append({"sweep": sweep + 1, "pairs": sweep_swaps})
        accepted_trace.append(accepted)
        loss_trace.append(losses.copy())
        if (
            sweep == 0
            or (sweep + 1) % 8 == 0
            or sweep + 1 == arguments.sweeps
        ):
            best = min(
                float(record["trajectory_loss"])
                for record in evaluated_records
                if record.get("valid", False)
            )
            print(
                "tempered sweep {}/{}: chain_best={:.9g}, "
                "cold_loss={:.9g}".format(
                    sweep + 1,
                    arguments.sweeps,
                    best,
                    losses[0],
                ),
                flush=True,
            )

    best_record = min(
        (record for record in evaluated_records if record.get("valid", False)),
        key=lambda record: float(record["trajectory_loss"]),
    )
    top_records = _top_diverse_records(
        evaluated_records,
        local_scales,
        count=arguments.saved_top_count,
        minimum_distance=arguments.saved_top_minimum_distance,
    )
    return {
        "best_record": best_record,
        "top_records": top_records,
        "evaluated_records": evaluated_records,
        "proposal_attempts": proposal_attempts,
        "proposal_accepts": proposal_accepts,
        "proposal_acceptance_rates": proposal_accepts
        / np.maximum(proposal_attempts, 1),
        "invalid_rejection_counts": invalid_counts,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "swap_acceptance_rates": swap_accepts / np.maximum(swap_attempts, 1),
        "loss_trace": np.asarray(loss_trace),
        "accepted_trace": np.asarray(accepted_trace),
        "swap_events": swap_events,
        "final_coordinates": coordinates,
        "final_losses": losses,
        "temperature_proposal_scales": temperature_proposal_scales,
    }


def _optimizer_record(result: Any, elapsed: float) -> dict[str, Any]:
    if result.success:
        category = "converged_by_tolerance"
    elif result.status == 0:
        category = "function_evaluation_limit"
    else:
        category = "optimizer_failure"
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "termination_category": category,
        "cost_with_prior_and_constraints": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "active_mask": np.asarray(result.active_mask, dtype=int),
        "elapsed_seconds": float(elapsed),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use local Jacobian-aligned parallel tempering to initialize one "
            "deterministic recorded-control trajectory refinement."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--command-delay", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--baseline-initializer-max-nfev", type=int, default=100)
    parser.add_argument("--baseline-max-nfev", type=int, default=25)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--replica-count", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=96)
    parser.add_argument("--exchange-interval", type=int, default=4)
    parser.add_argument("--pilot-count", type=int, default=48)
    parser.add_argument("--pilot-attempt-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="parallel forward-rollout workers",
    )
    parser.add_argument("--proposal-scale", type=float, default=1.0)
    parser.add_argument(
        "--maximum-temperature-proposal-scale", type=float, default=3.0
    )
    parser.add_argument("--proposal-damping", type=float, default=1.0e-6)
    parser.add_argument(
        "--proposal-minimum-normalized-variance",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--proposal-maximum-normalized-variance",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--local-prior-scales",
        type=float,
        nargs=sobol.SMOOTH_DIMENSION,
        default=DEFAULT_LOCAL_PRIOR_SCALES,
        metavar="SCALE",
    )
    parser.add_argument("--local-prior-radius", type=float, default=4.0)
    parser.add_argument("--local-prior-strength", type=float, default=1.0)
    parser.add_argument(
        "--low-typical-uphill-acceptance", type=float, default=0.02
    )
    parser.add_argument(
        "--high-typical-uphill-acceptance", type=float, default=0.30
    )
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    parser.add_argument("--saved-top-count", type=int, default=3)
    parser.add_argument(
        "--saved-top-minimum-distance", type=float, default=0.50
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
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> np.ndarray:
    scales = np.asarray(arguments.local_prior_scales, dtype=float)
    finite = (
        arguments.start,
        arguments.end,
        arguments.command_delay,
        arguments.prior_weight,
        arguments.pid_constraint_penalty,
        arguments.saved_top_minimum_distance,
        arguments.boundary_proximity_fraction,
        arguments.inertia_triangle_proximity_fraction,
    )
    finite_positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.replica_count,
        arguments.sweeps,
        arguments.exchange_interval,
        arguments.pilot_count,
        arguments.pilot_attempt_factor,
        arguments.workers,
        arguments.proposal_scale,
        arguments.maximum_temperature_proposal_scale,
        arguments.proposal_damping,
        arguments.proposal_minimum_normalized_variance,
        arguments.proposal_maximum_normalized_variance,
        arguments.local_prior_radius,
        arguments.local_prior_strength,
        arguments.saved_top_count,
    )
    if (
        any(not np.isfinite(value) for value in finite)
        or arguments.start >= arguments.end
        or arguments.command_delay < 0.0
        or arguments.prior_weight < 0.0
        or arguments.pid_constraint_penalty <= 0.0
        or arguments.baseline_initializer_max_nfev < 1
        or arguments.baseline_max_nfev < 1
        or arguments.max_nfev < 1
        or any(not np.isfinite(value) or value <= 0.0 for value in finite_positive)
        or int(arguments.replica_count) < 2
        or scales.shape != (sobol.SMOOTH_DIMENSION,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or arguments.proposal_maximum_normalized_variance
        < arguments.proposal_minimum_normalized_variance
        or not 0.0 < arguments.low_typical_uphill_acceptance < 1.0
        or not 0.0 < arguments.high_typical_uphill_acceptance < 1.0
        or arguments.low_typical_uphill_acceptance
        >= arguments.high_typical_uphill_acceptance
        or arguments.saved_top_minimum_distance < 0.0
        or not 0.0 < arguments.boundary_proximity_fraction < 0.5
        or arguments.inertia_triangle_proximity_fraction <= 0.0
    ):
        raise SystemExit("tempered deterministic settings are invalid")
    for value in (arguments.temperature_min, arguments.temperature_max):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise SystemExit("temperature overrides must be finite and positive")
    return scales


def run(arguments: argparse.Namespace) -> int:
    local_scales = _validate_arguments(arguments)
    bag = arguments.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit("bag does not exist: {}".format(bag))
    started = time.perf_counter()
    print(
        "loading {} [{:.3f}, {:.3f}] s".format(
            bag, arguments.start, arguments.end
        ),
        flush=True,
    )
    flight = baseline.load_flight_data(
        str(bag),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    try:
        pid_gate, pid_scales = _pid_gate(
            arguments, flight.controller_snapshot.gains
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bounds = _local_bounds(
        local_scales,
        arguments.local_prior_radius,
        arguments.command_delay,
    )
    parameterization = sobol.PhysicalSearchParameterization(
        VehicleParameters.nominal()
    )
    evaluator = sobol.CandidateEvaluator(
        flight=flight,
        parameterization=parameterization,
        pid_gate=pid_gate,
        bounds=bounds,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        prior_weight=arguments.prior_weight,
        pid_penalty_weight=arguments.pid_constraint_penalty,
    )
    evaluator.prior_scales = local_scales.copy()
    baseline_incumbent = sobol._run_deterministic_baseline_incumbent(
        evaluator, arguments
    )
    center = np.asarray(baseline_incumbent["coordinate"], dtype=float)
    center_problem = evaluator.make_problem(arguments.command_delay)
    center_evaluation = evaluator.evaluate(center, problem=center_problem)
    center_fallback = None
    if not center_evaluation.get("valid", False):
        nominal = parameterization.current_coordinate(arguments.command_delay)
        nominal_problem = evaluator.make_problem(arguments.command_delay)
        nominal_evaluation = evaluator.evaluate(nominal, problem=nominal_problem)
        if not nominal_evaluation.get("valid", False):
            raise RuntimeError("neither baseline nor nominal is a valid chain center")
        center_fallback = {
            "from": "deterministic_baseline",
            "to": "nominal",
            "reason": center_evaluation.get("reason", "unknown"),
        }
        center = nominal
        center_problem = nominal_problem
        center_evaluation = nominal_evaluation
    center_smooth = center[: sobol.SMOOTH_DIMENSION]
    _center_residual, center_jacobian = (
        evaluator.optimization_residual_and_jacobian(
            center_problem,
            center_smooth,
            arguments.command_delay,
        )
    )
    trajectory_rows = center_problem.output_time.size * 15
    proposal_root, proposal_diagnostic = proposal_geometry_from_jacobian(
        center_jacobian[:trajectory_rows],
        local_scales,
        damping=arguments.proposal_damping,
        minimum_variance=arguments.proposal_minimum_normalized_variance,
        maximum_variance=arguments.proposal_maximum_normalized_variance,
    )
    center_record = _compact_evaluation(
        center,
        "least_squares_center",
        center_evaluation,
    )
    rng = np.random.default_rng(arguments.seed)
    problem_count = max(arguments.replica_count, arguments.workers)
    problems = [
        evaluator.make_problem(arguments.command_delay)
        for _ in range(problem_count)
    ]
    executor = (
        None
        if arguments.workers == 1
        else ThreadPoolExecutor(max_workers=arguments.workers)
    )
    try:
        pilot_records, pilot_rejections = _pilot_local_proposals(
            evaluator,
            center_smooth,
            proposal_root,
            float(center_record["trajectory_loss"]),
            arguments,
            rng,
            problems,
            executor,
        )
        temperatures, temperature_diagnostic = calibrate_temperature_ladder(
            float(center_record["trajectory_loss"]),
            [record["trajectory_loss"] for record in pilot_records],
            replica_count=arguments.replica_count,
            low_typical_uphill_acceptance=(
                arguments.low_typical_uphill_acceptance
            ),
            high_typical_uphill_acceptance=(
                arguments.high_typical_uphill_acceptance
            ),
            minimum_override=arguments.temperature_min,
            maximum_override=arguments.temperature_max,
        )
        print(
            "temperature ladder: {}".format(
                np.array2string(temperatures, precision=5)
            ),
            flush=True,
        )
        tempered = _run_parallel_tempering(
            evaluator,
            center_record,
            pilot_records,
            proposal_root,
            temperatures,
            local_scales,
            arguments,
            rng,
            problems,
            executor,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    best_history = tempered["best_record"]
    best_coordinate = np.asarray(best_history["coordinate"], dtype=float)
    refinement_problem = evaluator.make_problem(arguments.command_delay)
    objective = sobol._CachedLocalObjective(
        evaluator,
        refinement_problem,
        arguments.command_delay,
    )
    refinement_started = time.perf_counter()
    refinement_result = least_squares(
        objective.residual,
        best_coordinate[: sobol.SMOOTH_DIMENSION],
        bounds=(
            bounds.lower[: sobol.SMOOTH_DIMENSION],
            bounds.upper[: sobol.SMOOTH_DIMENSION],
        ),
        method="trf",
        jac=objective.jacobian,
        x_scale="jac",
        loss="linear",
        ftol=1.0e-6,
        xtol=1.0e-6,
        gtol=1.0e-6,
        max_nfev=arguments.max_nfev,
        verbose=0,
    )
    refinement_elapsed = time.perf_counter() - refinement_started
    refined_coordinate = sobol._join_smooth_coordinate(
        refinement_result.x,
        arguments.command_delay,
    )
    refined_evaluation = evaluator.evaluate(
        refined_coordinate,
        problem=refinement_problem,
    )
    if not refined_evaluation.get("valid", False):
        raise RuntimeError("final tempered refinement is not physically valid")
    refined_loss = float(refined_evaluation["trajectory_loss"])
    print(
        "final refinement: history_loss={:.9g}, refined_loss={:.9g}, "
        "success={}".format(
            best_history["trajectory_loss"],
            refined_loss,
            refinement_result.success,
        ),
        flush=True,
    )

    history_evaluation = evaluator.evaluate(best_coordinate)
    candidates = (
        {
            "source": "deterministic_baseline",
            "trajectory_loss": float(baseline_incumbent["trajectory_loss"]),
            "coordinate": np.asarray(baseline_incumbent["coordinate"]),
            "evaluation": {
                "decoded": baseline_incumbent["decoded"],
                "pid": baseline_incumbent["pid"],
                "problem": baseline_incumbent["problem"],
                "simulation": baseline_incumbent["simulation"],
                "metrics": baseline_incumbent["metrics"],
            },
            "constraint_eligible": bool(
                baseline_incumbent["constraint_eligible"]
            ),
        },
        {
            "source": "tempered_history",
            "trajectory_loss": float(history_evaluation["trajectory_loss"]),
            "coordinate": best_coordinate,
            "evaluation": history_evaluation,
            "constraint_eligible": True,
        },
        {
            "source": "tempered_refinement",
            "trajectory_loss": refined_loss,
            "coordinate": refined_coordinate,
            "evaluation": refined_evaluation,
            "constraint_eligible": True,
        },
    )
    selected = min(candidates, key=lambda record: record["trajectory_loss"])
    selected_evaluation = selected["evaluation"]
    current_coordinate = parameterization.current_coordinate(
        arguments.command_delay
    )
    current_evaluation = evaluator.evaluate(current_coordinate)
    if not current_evaluation.get("valid", False):
        raise RuntimeError("nominal replay failed after tempered optimization")

    output = arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "trajectory.pdf"
    selected_coordinate = np.asarray(selected["coordinate"], dtype=float)
    selected_decoded = selected_evaluation["decoded"]
    selected_physical = evaluator.physical_diagnostic(selected_decoded)
    selected_boundary = sobol._boundary_diagnostic(
        selected_coordinate,
        bounds,
        arguments.boundary_proximity_fraction,
        physical=selected_physical,
        pid=selected_evaluation["pid"],
        pid_gate=pid_gate,
        inertia_triangle_proximity_fraction=(
            arguments.inertia_triangle_proximity_fraction
        ),
    )
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
            "name": "local_parallel_tempering_replica_exchange",
            "trajectory": (
                "single-shooting recorded-control open-loop without "
                "observation resets"
            ),
            "center": (
                "nominal-start deterministic baseline result, or nominal "
                "fallback when a configured hard gate rejects that result"
            ),
            "proposal_geometry": (
                "analytic trajectory Jacobian Gauss-Newton inverse in "
                "normalized coordinates with clipped eigenvalues"
            ),
            "local_prior": "zero-centered diagonal Gaussian",
            "temperature_calibration": (
                "median positive pilot trajectory-loss increment"
            ),
            "final_refinement": (
                "one bounded analytic-Jacobian least-squares solve from the "
                "best valid evaluated tempered point"
            ),
            "delay_fixed": True,
        },
        "local_parameterization": {
            "parameter_names": sobol.SEARCH_PARAMETER_NAMES,
            "prior_scales": local_scales,
            "prior_radius": arguments.local_prior_radius,
            "prior_strength": arguments.local_prior_strength,
            "bounds": bounds.as_mapping(),
            "fixed_command_delay_seconds": arguments.command_delay,
        },
        "pid_gate": {
            "enabled": pid_gate.enabled,
            "lower_scale": pid_scales[0],
            "upper_scale": pid_scales[1],
            "current_gains": pid_gate.current,
            "lower_gains": pid_gate.lower,
            "upper_gains": pid_gate.upper,
        },
        "deterministic_baseline_incumbent": (
            sobol._baseline_incumbent_payload(baseline_incumbent)
        ),
        "chain_center": {
            "coordinate": center,
            "trajectory_loss": center_record["trajectory_loss"],
            "fallback": center_fallback,
        },
        "proposal_geometry": proposal_diagnostic,
        "pilot": {
            "requested_valid_count": arguments.pilot_count,
            "valid_count": len(pilot_records),
            "rejection_counts": pilot_rejections,
            "losses": [record["trajectory_loss"] for record in pilot_records],
        },
        "temperature": {
            "values": temperatures,
            "calibration": temperature_diagnostic,
        },
        "parallel_tempering": {
            "replica_count": arguments.replica_count,
            "sweeps": arguments.sweeps,
            "forward_evaluation_count": (
                arguments.replica_count * arguments.sweeps
            ),
            "exchange_interval": arguments.exchange_interval,
            "workers": arguments.workers,
            "proposal_scale": arguments.proposal_scale,
            "temperature_proposal_scales": tempered[
                "temperature_proposal_scales"
            ],
            "proposal_attempts": tempered["proposal_attempts"],
            "proposal_accepts": tempered["proposal_accepts"],
            "proposal_acceptance_rates": tempered[
                "proposal_acceptance_rates"
            ],
            "invalid_rejection_counts": tempered[
                "invalid_rejection_counts"
            ],
            "swap_attempts": tempered["swap_attempts"],
            "swap_accepts": tempered["swap_accepts"],
            "swap_acceptance_rates": tempered["swap_acceptance_rates"],
            "loss_trace": tempered["loss_trace"],
            "accepted_trace": tempered["accepted_trace"],
            "swap_events": tempered["swap_events"],
            "final_coordinates": tempered["final_coordinates"],
            "final_losses": tempered["final_losses"],
            "best_history_record": best_history,
            "top_diverse_records": tempered["top_records"],
        },
        "final_refinement": {
            "start_coordinate": best_coordinate,
            "start_trajectory_loss": best_history["trajectory_loss"],
            "coordinate": refined_coordinate,
            "trajectory_loss": refined_loss,
            "analytic_linearization_count": objective.linearization_count,
            "optimizer": _optimizer_record(
                refinement_result, refinement_elapsed
            ),
        },
        "selection": {
            "policy": (
                "minimum trajectory loss among deterministic baseline, "
                "tempered history best, and final refinement"
            ),
            "candidate_losses": {
                record["source"]: record["trajectory_loss"]
                for record in candidates
            },
            "selected_source": selected["source"],
            "selected_trajectory_loss": selected["trajectory_loss"],
            "selected_coordinate": selected_coordinate,
            "selected_parameters": sobol._physical_payload(selected_decoded),
            "selected_pid_group_scales": selected_evaluation["pid"][
                "group_scales"
            ],
            "selected_pid_gains": selected_evaluation["pid"]["gains"],
            "selected_constraint_eligible": selected[
                "constraint_eligible"
            ],
            "selected_boundary": selected_boundary,
        },
        "recorded_control_open_loop_metrics": {
            "current": current_evaluation["metrics"],
            "selected": selected_evaluation["metrics"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "result_json": "result.json",
            "trajectory_pdf": "trajectory.pdf",
        },
    }
    baseline._write_pdf(
        pdf_path,
        selected_evaluation["problem"],
        current_evaluation["simulation"],
        selected_evaluation["simulation"],
        current_evaluation["metrics"],
        selected_evaluation["metrics"],
    )
    baseline._write_json(output / "result.json", payload)
    print("wrote {}".format(output / "result.json"), flush=True)
    print(
        "selected {} with trajectory loss {:.9g}".format(
            selected["source"], selected["trajectory_loss"]
        ),
        flush=True,
    )
    print("wrote {}".format(pdf_path), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
