#!/usr/bin/env python3
"""Regenerate standard first-order-lag artifacts from completed estimate JSON."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import CASE_BAG_JSONS, FirstOrderLagDynamicsProblem, load_estimate_json  # noqa: E402
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import ActuatorParameters  # noqa: E402
from reports import write_standard_outputs  # noqa: E402
from single_bag_savgol_core import load_vehicle_model, prepare_single_bag_dataset  # noqa: E402
from single_bag_savgol_reports import write_json  # noqa: E402


def _actuator_parameters(estimate: Mapping[str, Any]) -> ActuatorParameters:
    model = estimate["actuator_model"]
    return ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=float(model["gimbal_time_constant_seconds"]),
        delay=0.0,
        minimum_thrust=float(model["minimum_thrust"]),
        maximum_thrust=float(model["maximum_thrust"]),
        maximum_gimbal_angle=float(model["maximum_gimbal_angle"]),
        maximum_gimbal_rate=float(model["maximum_gimbal_rate"]),
    )


def _synthetic_arguments(estimate: Mapping[str, Any], source: Path) -> Mapping[str, Any]:
    actuator = estimate["actuator_model"]
    input_payload = estimate["input"]
    return {
        "reconstructed_from_estimate_json": str(source),
        "case": estimate["case_name"],
        "bag_json": input_payload["bag_json"],
        "vehicle_model": input_payload["vehicle_model"],
        "sg_window": input_payload["sg_window_seconds"],
        "sg_degree": input_payload["sg_degree"],
        "covariance_mode": input_payload["covariance_mode"],
        "gimbal_source": input_payload["gimbal_source"],
        "initial_tau": actuator["initialization"].get("initial_tau_seconds"),
        "gimbal_time_constant": actuator["gimbal_time_constant_seconds"],
        "minimum_thrust": actuator["minimum_thrust"],
        "maximum_thrust": actuator["maximum_thrust"],
        "maximum_gimbal_angle": actuator["maximum_gimbal_angle"],
        "maximum_gimbal_rate": actuator["maximum_gimbal_rate"],
    }


def render_estimate(path: Path) -> Path:
    source = Path(path).expanduser().resolve()
    estimate = load_estimate_json(source)
    input_payload = estimate["input"]
    interval = np.asarray(input_payload["bag_interval_seconds"], dtype=float)
    if interval.shape != (2,) or np.any(~np.isfinite(interval)) or interval[1] <= interval[0]:
        raise ValueError("estimate JSON bag interval is invalid")
    model = load_vehicle_model(Path(input_payload["vehicle_model"]))
    flight = load_flight_data(
        path=str(input_payload["bag_path"]),
        start_local=float(interval[0]),
        end_local=float(interval[1]),
        include_fc_specific_force=True,
        compute_sha256=False,
        bag_id=str(estimate["case_name"]),
    )
    dataset = prepare_single_bag_dataset(
        flight=flight,
        window_seconds=float(input_payload["sg_window_seconds"]),
        degree=int(input_payload["sg_degree"]),
        covariance_mode=str(input_payload["covariance_mode"]),
        geometric_correction=True,
    )
    problem = FirstOrderLagDynamicsProblem(
        dataset,
        model,
        _actuator_parameters(estimate),
        gimbal_source=str(input_payload["gimbal_source"]),
        parameter_prior=None,
    )
    physical = np.asarray(
        estimate["plant_distribution"]["physical_chart_coordinate"], dtype=float
    )
    if physical.shape != (14,) or np.any(~np.isfinite(physical)):
        raise ValueError("estimate JSON physical coordinate is invalid")
    tau = float(estimate["actuator_model"]["thrust_time_constant_seconds"])
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("estimate JSON time constant is invalid")
    evaluation = problem.evaluate_first_order(physical, math.log(tau))
    initialization = estimate["actuator_model"]["initialization"]
    solver_runs = initialization.get("multistart_results", ())
    output_dir = source.parent
    files = write_standard_outputs(
        output_dir,
        estimate=estimate,
        arguments=_synthetic_arguments(estimate, source),
        dataset=dataset,
        model=model,
        evaluation=evaluation,
        solver_runs=solver_runs,
    )
    write_json(
        output_dir / "status.json",
        {
            "schema": str(estimate["schema"]) + "-status",
            "status": "completed",
            "case_name": estimate["case_name"],
            "source_commit": estimate["source_commit"],
            "estimate_json": str(source),
            "standard_outputs": list(files),
        },
    )
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--estimate-json", type=Path)
    source.add_argument("--case", choices=tuple(CASE_BAG_JSONS))
    parser.add_argument(
        "--all",
        action="store_true",
        help="render failure1, failure2, and success; default when no source is specified",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.estimate_json is not None:
        paths = (Path(arguments.estimate_json),)
    elif arguments.case is not None:
        paths = (_HERE / "outputs" / str(arguments.case) / "estimate.json",)
    else:
        paths = tuple(_HERE / "outputs" / case / "estimate.json" for case in CASE_BAG_JSONS)
    for path in paths:
        print(render_estimate(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
