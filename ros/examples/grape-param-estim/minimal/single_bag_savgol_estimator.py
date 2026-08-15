#!/usr/bin/env python3
"""CLI for prior-free geometric-SG estimation of one rosbag."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import ActuatorParameters  # noqa: E402
from single_bag_savgol_core import (  # noqa: E402
    BASE_PLAN_COMMIT,
    DEFAULT_SMOOTH_MAX_NFEV,
    DEFAULT_STRICT_MAX_NFEV,
    EstimatorConfig,
    LmSettings,
    SingleBagDynamicsProblem,
    estimate_single_bag,
    load_vehicle_model,
    prepare_single_bag_dataset,
)
from single_bag_input import load_single_bag_input  # noqa: E402
from single_bag_savgol_reports import (  # noqa: E402
    output_run_directory,
    source_commit,
    write_completed_case,
    write_failure_report_pdf,
    write_json,
)
from single_bag_wrench_replay import fit_external_wrench_replay  # noqa: E402


def _arguments_payload(arguments: argparse.Namespace) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in vars(arguments).items()
        if not key.startswith("_")
    }
    payload["base_plan_commit"] = BASE_PLAN_COMMIT
    return payload


def resolve_bag_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    """Resolve either direct CLI fields or the minimal single-bag JSON."""

    values = vars(arguments).copy()
    bag_json = values.get("bag_json")
    if bag_json is not None:
        resolved = bool(values.get("_bag_json_resolved", False))
        if resolved:
            return argparse.Namespace(**values)
        if values.get("bag") is not None:
            raise ValueError("--bag and --bag-json cannot be used together")
        if values.get("bag_start") is not None or values.get("bag_end") is not None:
            raise ValueError("--bag-start/--bag-end cannot be used with --bag-json")
        bag_input = load_single_bag_input(bag_json)
        values["bag"] = bag_input.bag_path
        values["bag_start"] = bag_input.start_seconds
        values["bag_end"] = bag_input.end_seconds
        if values.get("bag_id") is None:
            values["bag_id"] = Path(bag_json).expanduser().stem
        values["_bag_json_resolved"] = True
    else:
        if values.get("bag") is None:
            raise ValueError("one of --bag or --bag-json is required")
        if values.get("bag_start") is None or values.get("bag_end") is None:
            raise ValueError("--bag requires --bag-start and --bag-end")
        start = float(values["bag_start"])
        end = float(values["bag_end"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("bag interval must be finite with start < end")
        values["bag"] = Path(values["bag"]).expanduser().resolve()
        values["bag_start"] = start
        values["bag_end"] = end
    return argparse.Namespace(**values)


def actuator_parameters_from_arguments(
    arguments: argparse.Namespace,
) -> ActuatorParameters:
    return ActuatorParameters(
        thrust_time_constant=float(arguments.thrust_time_constant),
        gimbal_time_constant=float(arguments.gimbal_time_constant),
        delay=0.0,
        minimum_thrust=float(arguments.minimum_thrust),
        maximum_thrust=float(arguments.maximum_thrust),
        maximum_gimbal_angle=float(arguments.maximum_gimbal_angle),
        maximum_gimbal_rate=float(arguments.maximum_gimbal_rate),
    )


def estimator_config_from_arguments(
    arguments: argparse.Namespace,
) -> EstimatorConfig:
    return EstimatorConfig(
        covariance_mode=arguments.covariance_mode,
        geometric_correction=not arguments.naive_so3_derivatives,
        gimbal_source=arguments.gimbal_source,
        lag_mode=arguments.lag_mode,
        initial_rotor_lag=arguments.initial_rotor_lag,
        initial_rotor_lag_multiplier=arguments.initial_rotor_lag_multiplier,
        fixed_rotor_lag=arguments.fixed_rotor_lag,
        lag_continuation_depth=arguments.lag_continuation_depth,
        lag_continuation_schedule=(
            None
            if arguments.lag_continuation_schedule is None
            else tuple(arguments.lag_continuation_schedule)
        ),
        lag_continuation_enabled=not arguments.disable_lag_continuation,
        smooth_max_nfev=arguments.smooth_max_nfev,
        strict_max_nfev=arguments.strict_max_nfev,
        kkt_enabled=not arguments.disable_kkt,
        solver_type=arguments.solver_type,
        initial_physical_coordinate=np.asarray(arguments.initial_coordinate),
        scale_initial_offset=arguments.scale_initial_offset,
        lm=LmSettings(
            initial_damping=arguments.lm_initial_damping,
            initial_trust_radius=arguments.lm_initial_trust_radius,
            maximum_trust_radius=arguments.lm_maximum_trust_radius,
            minimum_trust_radius=arguments.lm_minimum_trust_radius,
            acceptance_ratio=arguments.lm_acceptance_ratio,
            gtol=arguments.gtol,
            ftol=arguments.ftol,
            xtol=arguments.xtol,
        ),
    )


def run_estimator(
    arguments: argparse.Namespace,
    *,
    case_name: str = "default",
    output_directory: Optional[Path] = None,
) -> tuple[Path, Mapping[str, Any]]:
    """Execute one case.  Ablation runner reuses this exact pipeline."""

    arguments = resolve_bag_arguments(arguments)
    revision = source_commit(_PROJECT_ROOT)
    directory = (
        output_run_directory(
            arguments.output_root,
            "default",
            arguments.run_id,
            commit=revision,
        )
        if output_directory is None
        else Path(output_directory)
    )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage = "arguments"
    try:
        config = estimator_config_from_arguments(arguments)
        actuator_parameters = actuator_parameters_from_arguments(arguments)
        stage = "vehicle_model"
        model = load_vehicle_model(arguments.vehicle_model)
        stage = "single_bag_load"
        flight = load_flight_data(
            path=str(arguments.bag),
            start_local=float(arguments.bag_start),
            end_local=float(arguments.bag_end),
            include_fc_specific_force=True,
            compute_sha256=not arguments.skip_bag_sha256,
            bag_id=arguments.bag_id,
        )
        stage = "geometric_sg_covariance"
        dataset = prepare_single_bag_dataset(
            flight=flight,
            window_seconds=float(arguments.sg_window),
            degree=int(arguments.sg_degree),
            covariance_mode=config.covariance_mode,
            geometric_correction=config.geometric_correction,
        )
        problem = SingleBagDynamicsProblem(
            dataset,
            model,
            actuator_parameters,
            gimbal_source=config.gimbal_source,
        )
        stage = "parameter_estimation"
        result = estimate_single_bag(problem, config)
        stage = "standardized_wrench_replay"
        # Even when replay is disabled as a *case algorithm*, the plan requires
        # a frozen-result standardized replay for every completed report.
        replay = fit_external_wrench_replay(
            dataset=dataset, model=model, evaluation=result.evaluation
        )
        stage = "report"
        payload = write_completed_case(
            directory,
            case_name=case_name,
            source_revision=revision,
            arguments=_arguments_payload(arguments),
            dataset=dataset,
            model=model,
            result=result,
            replay=replay,
        )
        return directory, payload
    except Exception as error:
        elapsed = time.perf_counter() - started
        failure = {
            "status": "failed",
            "case_name": case_name,
            "source_commit": revision,
            "base_plan_commit": BASE_PLAN_COMMIT,
            "failure_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "elapsed_seconds": elapsed,
            "traceback": traceback.format_exc(),
        }
        write_json(directory / "status.json", failure)
        write_json(directory / "result.json", failure)
        write_json(directory / "arguments.json", _arguments_payload(arguments))
        write_json(directory / "timing.json", {"elapsed_seconds": elapsed})
        write_failure_report_pdf(
            directory / "report.pdf",
            case_name=case_name,
            failure_stage=stage,
            exception_type=type(error).__name__,
            message=str(error),
        )
        return directory, failure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    bag_source = parser.add_mutually_exclusive_group(required=True)
    bag_source.add_argument("--bag", type=Path)
    bag_source.add_argument(
        "--bag-json",
        type=Path,
        help="JSON containing only bag_path/start_seconds/end_seconds",
    )
    parser.add_argument("--bag-id", default=None)
    parser.add_argument("--bag-start", type=float, default=None)
    parser.add_argument("--bag-end", type=float, default=None)
    parser.add_argument("--skip-bag-sha256", action="store_true")
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--sg-window", type=float, default=1.0)
    parser.add_argument("--sg-degree", type=int, default=5)
    parser.add_argument(
        "--covariance-mode",
        choices=(
            "full",
            "identity",
            "diagonal",
            "block_s_alpha",
            "full_no_R_uncertainty_in_s",
            "full_no_position_rotation_cross",
            "global_full",
        ),
        default="identity",
    )
    parser.add_argument("--naive-so3-derivatives", action="store_true")
    parser.add_argument(
        "--gimbal-source",
        choices=("measured_sg", "measured_linear", "command_replay"),
        default="measured_sg",
    )
    parser.add_argument(
        "--lag-mode",
        choices=("estimated", "zero", "fixed"),
        default="estimated",
    )
    parser.add_argument("--initial-rotor-lag", type=float, default=None)
    parser.add_argument(
        "--initial-rotor-lag-multiplier", type=float, default=1.0
    )
    parser.add_argument("--fixed-rotor-lag", type=float, default=None)
    parser.add_argument(
        "--lag-continuation-depth",
        type=int,
        default=9,
        help="maximum k in epsilon_k=2^-k (inclusive)",
    )
    parser.add_argument(
        "--lag-continuation-schedule",
        type=float,
        nargs="+",
        default=None,
        help="explicit epsilon schedule for the legacy-schedule ablation",
    )
    parser.add_argument("--disable-lag-continuation", action="store_true")
    parser.add_argument(
        "--smooth-max-nfev", type=int, default=DEFAULT_SMOOTH_MAX_NFEV
    )
    parser.add_argument(
        "--strict-max-nfev", type=int, default=DEFAULT_STRICT_MAX_NFEV
    )
    parser.add_argument("--disable-kkt", action="store_true")
    parser.add_argument(
        "--solver-type",
        choices=("custom_kkt_lm", "standard_least_squares"),
        default="custom_kkt_lm",
    )
    parser.add_argument(
        "--initial-coordinate", type=float, nargs=14, default=np.zeros(14)
    )
    parser.add_argument("--scale-initial-offset", type=float, default=0.0)
    parser.add_argument("--lm-initial-damping", type=float, default=1.0e-3)
    parser.add_argument("--lm-initial-trust-radius", type=float, default=1.0)
    parser.add_argument("--lm-maximum-trust-radius", type=float, default=8.0)
    parser.add_argument("--lm-minimum-trust-radius", type=float, default=1.0e-10)
    parser.add_argument("--lm-acceptance-ratio", type=float, default=1.0e-4)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--ftol", type=float, default=float(np.sqrt(np.finfo(float).eps)))
    parser.add_argument("--xtol", type=float, default=float(np.sqrt(np.finfo(float).eps)))
    defaults = ActuatorParameters()
    parser.add_argument(
        "--thrust-time-constant", type=float, default=defaults.thrust_time_constant
    )
    parser.add_argument(
        "--gimbal-time-constant", type=float, default=defaults.gimbal_time_constant
    )
    parser.add_argument("--minimum-thrust", type=float, default=defaults.minimum_thrust)
    parser.add_argument("--maximum-thrust", type=float, default=defaults.maximum_thrust)
    parser.add_argument(
        "--maximum-gimbal-angle", type=float, default=defaults.maximum_gimbal_angle
    )
    parser.add_argument(
        "--maximum-gimbal-rate", type=float, default=defaults.maximum_gimbal_rate
    )
    parser.add_argument("--output-root", type=Path, default=_HERE / "outputs")
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    directory, payload = run_estimator(arguments)
    print(directory)
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
