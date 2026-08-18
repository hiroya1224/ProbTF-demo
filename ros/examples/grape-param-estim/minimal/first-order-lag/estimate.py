#!/usr/bin/env python3
"""Estimate one effective first-order rotor-thrust time constant and plant."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import (  # noqa: E402
    CASE_BAG_JSONS,
    CASE_OUTCOMES,
    ESTIMATE_SCHEMA,
    FirstOrderLagDynamicsProblem,
    controller_period_payload,
    controller_snapshot_payload,
    physical_point_payload,
    quotient_distribution_payload,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import ActuatorParameters  # noqa: E402
from single_bag_input import load_single_bag_input  # noqa: E402
from single_bag_savgol_core import (  # noqa: E402
    COMMON_SCALE_DIRECTION,
    PHYSICAL_DIMENSION,
    common_scale_quotient_basis,
    load_vehicle_model,
    prepare_single_bag_dataset,
)
from single_bag_savgol_covariance import parameter_covariances  # noqa: E402
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


DEFAULT_INITIAL_TAU_MULTIPLIERS = (1.0, 4.0, 16.0, 64.0)
DEFAULT_MAX_NFEV = 10000


def _resolve_case(arguments: argparse.Namespace) -> tuple[str, Path, str]:
    if arguments.case is not None:
        return (
            str(arguments.case),
            CASE_BAG_JSONS[str(arguments.case)],
            CASE_OUTCOMES[str(arguments.case)],
        )
    source = Path(arguments.bag_json).expanduser().resolve()
    return (
        str(arguments.case_name or source.stem),
        source,
        str(arguments.outcome or "unspecified"),
    )


def _actuator_parameters(arguments: argparse.Namespace) -> ActuatorParameters:
    # tau is estimated by FirstOrderLagDynamicsProblem itself.  The object is
    # still the source of actuator clamps and measured-gimbal replay settings.
    return ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=float(arguments.gimbal_time_constant),
        delay=0.0,
        minimum_thrust=float(arguments.minimum_thrust),
        maximum_thrust=float(arguments.maximum_thrust),
        maximum_gimbal_angle=float(arguments.maximum_gimbal_angle),
        maximum_gimbal_rate=float(arguments.maximum_gimbal_rate),
    )


def _safe_reduced_problem(problem: FirstOrderLagDynamicsProblem):
    basis = common_scale_quotient_basis()
    cache_x: Optional[np.ndarray] = None
    cache_residual: Optional[np.ndarray] = None
    cache_jacobian: Optional[np.ndarray] = None
    invalid_trial_count = 0
    invalid_types: set[str] = set()

    initial_full = np.concatenate((np.zeros(PHYSICAL_DIMENSION), np.asarray((math.log(0.1),))))
    initial_residual, _initial_jacobian, _payload = problem.global_residual_jacobian(
        initial_full, command_mode="strict", epsilon=None
    )
    penalty_component = min(
        math.sqrt(np.finfo(float).max / max(initial_residual.size, 1)) / 8.0,
        max(1.0, float(np.max(np.abs(initial_residual)))) * 1.0e6,
    )
    penalty_residual = np.full(initial_residual.shape, penalty_component)
    penalty_jacobian = np.zeros((initial_residual.size, 14), dtype=float)

    def evaluate(reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal cache_x, cache_residual, cache_jacobian
        nonlocal invalid_trial_count
        selected = np.asarray(reduced, dtype=float)
        if cache_x is not None and np.array_equal(selected, cache_x):
            assert cache_residual is not None and cache_jacobian is not None
            return cache_residual, cache_jacobian
        try:
            physical = basis @ selected[:13]
            full = np.concatenate((physical, selected[13:14]))
            residual, jacobian, _payload = problem.global_residual_jacobian(
                full, command_mode="strict", epsilon=None
            )
            reduced_jacobian = np.column_stack(
                (jacobian[:, :14] @ basis, jacobian[:, 14])
            )
            if (
                residual.ndim != 1
                or reduced_jacobian.shape != (residual.size, 14)
                or np.any(~np.isfinite(residual))
                or np.any(~np.isfinite(reduced_jacobian))
            ):
                raise FloatingPointError("non-finite first-order objective")
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as error:
            invalid_trial_count += 1
            invalid_types.add(type(error).__name__)
            residual = penalty_residual
            reduced_jacobian = penalty_jacobian
        cache_x = selected.copy()
        cache_residual = np.asarray(residual, dtype=float).copy()
        cache_jacobian = np.asarray(reduced_jacobian, dtype=float).copy()
        return cache_residual, cache_jacobian

    def residual(reduced: np.ndarray) -> np.ndarray:
        return evaluate(reduced)[0]

    def jacobian(reduced: np.ndarray) -> np.ndarray:
        return evaluate(reduced)[1]

    def diagnostics() -> Mapping[str, Any]:
        return {
            "invalid_trial_count": int(invalid_trial_count),
            "invalid_trial_exception_types": sorted(invalid_types),
        }

    return basis, residual, jacobian, diagnostics


def _fit_multistart(
    problem: FirstOrderLagDynamicsProblem,
    *,
    initial_taus: Sequence[float],
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[Any, list[Mapping[str, Any]], np.ndarray, Any]:
    basis, residual, jacobian, numerical_diagnostics = _safe_reduced_problem(problem)
    starts: list[Mapping[str, Any]] = []
    solutions: list[tuple[float, Any, np.ndarray]] = []
    for tau in initial_taus:
        value = float(tau)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("initial tau values must be finite and positive")
        initial = np.zeros(14, dtype=float)
        initial[13] = math.log(value)
        started = time.perf_counter()
        solved = least_squares(
            residual,
            initial,
            jac=jacobian,
            method="trf",
            max_nfev=int(max_nfev),
            ftol=float(ftol),
            xtol=float(xtol),
            gtol=float(gtol),
        )
        selected = np.asarray(solved.x, dtype=float)
        physical = basis @ selected[:13]
        log_tau = float(selected[13])
        evaluation = problem.evaluate_first_order(physical, log_tau)
        cost = float(evaluation.cost)
        row = {
            "initial_tau_seconds": value,
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": str(solved.message),
            "nfev": int(solved.nfev),
            "elapsed_seconds": float(time.perf_counter() - started),
            "final_tau_seconds": float(math.exp(log_tau)),
            "data_cost": cost,
        }
        starts.append(row)
        solutions.append((cost, solved, physical))
    successful = [item for item in solutions if bool(item[1].success)]
    candidates = successful if successful else solutions
    candidates.sort(key=lambda item: item[0])
    best_cost, best_solver, best_physical = candidates[0]
    del best_cost
    selected_tau = float(math.exp(float(np.asarray(best_solver.x)[13])))
    for row in starts:
        row["selected"] = bool(
            np.isclose(
                float(row["final_tau_seconds"]),
                selected_tau,
                rtol=0.0,
                atol=16.0 * np.finfo(float).eps * max(1.0, abs(selected_tau)),
            )
            and bool(row["success"]) == bool(best_solver.success)
        )
    return best_solver, starts, best_physical, numerical_diagnostics


def run_estimate(arguments: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    case_name, bag_json_path, outcome = _resolve_case(arguments)
    bag_input = load_single_bag_input(bag_json_path)
    output_dir = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir is not None
        else (_HERE / "outputs" / case_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = source_commit(_PROJECT_ROOT)
    started = time.perf_counter()
    stage = "load"
    try:
        model = load_vehicle_model(Path(arguments.vehicle_model))
        flight = load_flight_data(
            path=bag_input.bag_path,
            start_local=bag_input.start_seconds,
            end_local=bag_input.end_seconds,
            include_fc_specific_force=True,
            compute_sha256=not arguments.skip_bag_sha256,
            bag_id=case_name,
        )
        stage = "dataset"
        dataset = prepare_single_bag_dataset(
            flight=flight,
            window_seconds=float(arguments.sg_window),
            degree=int(arguments.sg_degree),
            covariance_mode=str(arguments.covariance_mode),
            geometric_correction=True,
        )
        actuator = _actuator_parameters(arguments)
        problem = FirstOrderLagDynamicsProblem(
            dataset,
            model,
            actuator,
            gimbal_source=str(arguments.gimbal_source),
            parameter_prior=None,
        )
        command_period = float(dataset.rotor_history.median_period)
        if not np.isfinite(command_period) or command_period <= 0.0:
            raise ValueError("rotor command history has no positive median period")
        if arguments.initial_tau is None:
            initial_taus = tuple(
                command_period * float(multiplier)
                for multiplier in arguments.initial_tau_multipliers
            )
        else:
            initial_taus = (float(arguments.initial_tau),)

        stage = "optimization"
        solved, starts, physical, numerical_diagnostics = _fit_multistart(
            problem,
            initial_taus=initial_taus,
            max_nfev=arguments.max_nfev,
            ftol=arguments.ftol,
            xtol=arguments.xtol,
            gtol=arguments.gtol,
        )
        log_tau = float(np.asarray(solved.x, dtype=float)[13])
        tau = float(math.exp(log_tau))
        final = problem.evaluate_first_order(physical, log_tau)

        stage = "postfit_uncertainty"
        uncertainty = parameter_covariances(
            final.acceleration_jacobian,
            dataset.covariance,
            COMMON_SCALE_DIRECTION,
            additional_residual_covariance=None,
            uncentered_residual=final.acceleration_residual,
        )
        timing = controller_period_payload(flight)
        controller = controller_snapshot_payload(flight)
        payload: Mapping[str, Any] = {
            "schema": ESTIMATE_SCHEMA,
            "status": "completed",
            "source_commit": revision,
            "case_name": case_name,
            "flight_outcome": outcome,
            "input": {
                "bag_json": str(Path(bag_json_path).expanduser().resolve()),
                "bag_path": bag_input.bag_path,
                "bag_interval_seconds": [
                    float(bag_input.start_seconds),
                    float(bag_input.end_seconds),
                ],
                "vehicle_model": str(Path(arguments.vehicle_model).expanduser().resolve()),
                "sg_window_seconds": float(arguments.sg_window),
                "sg_degree": int(arguments.sg_degree),
                "covariance_mode": str(arguments.covariance_mode),
                "gimbal_source": str(arguments.gimbal_source),
            },
            "actuator_model": {
                "kind": "zero_order_hold_command_then_first_order_thrust",
                "pure_delay_seconds": 0.0,
                "thrust_time_constant_seconds": tau,
                "optimized_coordinate": "log_thrust_time_constant",
                "gimbal_time_constant_seconds": float(arguments.gimbal_time_constant),
                "minimum_thrust": float(arguments.minimum_thrust),
                "maximum_thrust": float(arguments.maximum_thrust),
                "maximum_gimbal_angle": float(arguments.maximum_gimbal_angle),
                "maximum_gimbal_rate": float(arguments.maximum_gimbal_rate),
                "initialization": {
                    "command_period_seconds": command_period,
                    "initial_tau_seconds": list(initial_taus),
                    "multistart_results": starts,
                },
            },
            "optimization": {
                "success": bool(solved.success),
                "any_multistart_success": bool(any(row["success"] for row in starts)),
                "selected_from_successful_multistarts_when_available": True,
                "status": int(solved.status),
                "message": str(solved.message),
                "nfev": int(solved.nfev),
                "data_cost": float(final.cost),
                "residual_rms": float(
                    np.sqrt(np.mean(final.acceleration_residual**2))
                ),
                "numerical_diagnostics": dict(numerical_diagnostics()),
            },
            "point_estimate": physical_point_payload(final.parameters),
            "plant_distribution": quotient_distribution_payload(
                physical, uncertainty
            ),
            "controller": controller,
            "controller_timing": timing,
            "provenance": {
                "bag_sha256": str(flight.provenance.bag_sha256),
                "bag_size_bytes": int(flight.provenance.bag_size_bytes),
                "adapter_revision": flight.provenance.adapter_revision,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json(output_dir / "estimate.json", payload)
        write_json(
            output_dir / "status.json",
            {
                "schema": ESTIMATE_SCHEMA + "-status",
                "status": "completed",
                "case_name": case_name,
                "source_commit": revision,
                "estimate_json": str(output_dir / "estimate.json"),
            },
        )
        return output_dir, payload
    except Exception as error:
        failure = {
            "schema": ESTIMATE_SCHEMA + "-status",
            "status": "failed",
            "source_commit": revision,
            "case_name": case_name,
            "failure_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json(output_dir / "status.json", failure)
        return output_dir, failure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=tuple(CASE_BAG_JSONS))
    source.add_argument("--bag-json", type=Path)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        default=_MINIMAL / "grape_vehicle_model.json",
    )
    parser.add_argument("--skip-bag-sha256", action="store_true")
    parser.add_argument("--sg-window", type=float, default=1.0)
    parser.add_argument("--sg-degree", type=int, default=5)
    parser.add_argument("--covariance-mode", default="identity")
    parser.add_argument(
        "--gimbal-source",
        choices=("measured_sg", "measured_linear", "command_replay"),
        default="measured_sg",
    )
    parser.add_argument("--initial-tau", type=float, default=None)
    parser.add_argument(
        "--initial-tau-multipliers",
        type=float,
        nargs="+",
        default=DEFAULT_INITIAL_TAU_MULTIPLIERS,
        help="used only when --initial-tau is omitted; multiplied by command period",
    )
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--ftol", type=float, default=float(np.sqrt(np.finfo(float).eps)))
    parser.add_argument("--xtol", type=float, default=float(np.sqrt(np.finfo(float).eps)))
    parser.add_argument("--gimbal-time-constant", type=float, default=0.0)
    parser.add_argument("--minimum-thrust", type=float, default=1.5)
    parser.add_argument("--maximum-thrust", type=float, default=27.6145)
    parser.add_argument("--maximum-gimbal-angle", type=float, default=3.14)
    parser.add_argument("--maximum-gimbal-rate", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    directory, payload = run_estimate(arguments)
    print(directory)
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
