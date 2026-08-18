#!/usr/bin/env python3
"""Run the local-pole validation for all current bag/covariance/delay cases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from gimbalrotor_pid_local_pole_validation import (  # noqa: E402
    COVARIANCE_MODES,
    DEFAULT_FD_CHECK_SAMPLES,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DELAY_MODES,
    PLAN_BASE_COMMIT,
    analyze_case,
    source_commit,
    write_json,
    write_outputs,
)


THREE_BAG_SCHEMA = "grape-param-estim/gimbalrotor-pid-local-poles-three-bag/v1"
DEFAULT_COVARIANCE_MODES = ("conservative_fusion", "overlap_corrected")
DEFAULT_DELAY_MODES = ("fitted_thrust_delay", "zero_thrust_delay")
_ESTIMATOR_ROOT = (
    _HERE
    / "outputs"
    / "916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1"
    / "prior_ablation"
)
_STATIC_ROOT = (
    _HERE
    / "outputs"
    / "585db5ba8a236232d85f2097615cf64b7eb76ff0"
    / "gimbalrotor_pid_postprocess"
)


CASE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "failure1": {
        "outcome": "crashed",
        "estimator": _ESTIMATOR_ROOT
        / "single_rosbag_1_nominal_pseudo_conditioning_production_20260817"
        / "cases"
        / "prior_free",
        "static": _STATIC_ROOT
        / "single_rosbag_1_prior_free_static_pid_production_20260817"
        / "pid_gain_postprocess.json",
        "bag_json": _HERE / "bag_jsons" / "single_rosbag_1.json",
    },
    "failure2": {
        "outcome": "crashed",
        "estimator": _ESTIMATOR_ROOT
        / "single_rosbag_2_nominal_pseudo_conditioning_production_20260817"
        / "cases"
        / "prior_free",
        "static": _STATIC_ROOT
        / "single_rosbag_2_prior_free_static_pid_production_20260817"
        / "pid_gain_postprocess.json",
        "bag_json": _HERE / "bag_jsons" / "single_rosbag_2.json",
    },
    "success": {
        "outcome": "successful",
        "estimator": _ESTIMATOR_ROOT
        / "single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817"
        / "cases"
        / "prior_free",
        "static": _STATIC_ROOT
        / "single_rosbag_succeeded_prior_free_static_pid_production_20260817"
        / "pid_gain_postprocess.json",
        "bag_json": _HERE / "bag_jsons" / "single_rosbag_succeeded.json",
    },
}


def _case_row(name: str, report: Mapping[str, Any]) -> Mapping[str, Any]:
    radius = report["stability_distribution"]["spectral_radius"]
    gains = report["controller"]["gains"]["roll_pitch"]
    distribution = report["stability_distribution"]
    return {
        "case": name,
        "actual_outcome": report["flight_outcome"],
        "covariance_mode": report["plant_distribution"]["covariance_mode"],
        "delay_mode": report["delay_model"]["mode"],
        "recorded_roll_pitch_pid": dict(gains),
        "fitted_thrust_delay_seconds": report["delay_model"]["fitted_rotor_lag_seconds"],
        "controller_dt_seconds": report["controller_timing"]["selected_controller_dt_seconds"],
        "requested_samples": distribution["requested_samples"],
        "pole_valid_samples": distribution["pole_valid_samples"],
        "stable_fraction": distribution["stable_fraction_among_pole_valid"],
        "center_spectral_radius": report["center_result"]["spectral_radius"],
        "spectral_radius": radius,
        "median_unstable_pole_count": distribution["unstable_pole_count_median"],
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    comparisons: list[Mapping[str, Any]] = []
    lookup = {
        (row["case"], row["covariance_mode"], row["delay_mode"]): row
        for row in rows
    }
    for case in CASE_DEFINITIONS:
        for covariance in DEFAULT_COVARIANCE_MODES:
            fitted = lookup.get((case, covariance, "fitted_thrust_delay"))
            zero = lookup.get((case, covariance, "zero_thrust_delay"))
            if fitted is None or zero is None:
                continue
            fitted_radius = fitted["spectral_radius"]
            zero_radius = zero["spectral_radius"]
            comparisons.append(
                {
                    "case": case,
                    "covariance_mode": covariance,
                    "fitted_delay_stable_fraction": fitted["stable_fraction"],
                    "zero_delay_stable_fraction": zero["stable_fraction"],
                    "stable_fraction_difference_fitted_minus_zero": (
                        None
                        if fitted["stable_fraction"] is None or zero["stable_fraction"] is None
                        else fitted["stable_fraction"] - zero["stable_fraction"]
                    ),
                    "fitted_delay_median_spectral_radius": (
                        None if fitted_radius is None else fitted_radius["q50"]
                    ),
                    "zero_delay_median_spectral_radius": (
                        None if zero_radius is None else zero_radius["q50"]
                    ),
                    "median_spectral_radius_difference_fitted_minus_zero": (
                        None
                        if fitted_radius is None or zero_radius is None
                        else fitted_radius["q50"] - zero_radius["q50"]
                    ),
                }
            )
    return {
        "schema": THREE_BAG_SCHEMA,
        "source_commit": source_commit(),
        "plan_base_commit": PLAN_BASE_COMMIT,
        "case_count": len(rows),
        "cases_are_averaged": False,
        "rows": list(rows),
        "fitted_versus_zero_delay": comparisons,
    }


def _value(value: Any) -> str:
    return "—" if value is None else "{:.8g}".format(float(value))


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Three-bag local sampled-data pole validation",
        "",
        "Each row is an independent bag distribution; bags are never averaged.",
        "",
        (
            "| case | outcome | covariance | delay | roll/pitch P/I/D | fitted lag [s] | "
            "dt [s] | pole valid | stable fraction | center radius | median radius | "
            "radius 16–84% | radius 2.5–97.5% | median unstable poles |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        pid = row["recorded_roll_pitch_pid"]
        radius = row["spectral_radius"]
        lines.append(
            "| {case} | {outcome} | {covariance} | {delay} | {p:.6g}/{i:.6g}/{d:.6g} | "
            "{lag:.8g} | {dt:.8g} | {valid}/{requested} | {stable} | {center} | "
            "{median} | {range68} | {range95} | {unstable} |".format(
                case=row["case"],
                outcome=row["actual_outcome"],
                covariance=row["covariance_mode"],
                delay=row["delay_mode"],
                p=pid["p_gain"],
                i=pid["i_gain"],
                d=pid["d_gain"],
                lag=row["fitted_thrust_delay_seconds"],
                dt=row["controller_dt_seconds"],
                valid=row["pole_valid_samples"],
                requested=row["requested_samples"],
                stable=_value(row["stable_fraction"]),
                center=_value(row["center_spectral_radius"]),
                median="—" if radius is None else _value(radius["q50"]),
                range68="—" if radius is None else "[{:.8g}, {:.8g}]".format(radius["q16"], radius["q84"]),
                range95="—" if radius is None else "[{:.8g}, {:.8g}]".format(radius["q025"], radius["q975"]),
                unstable=_value(row["median_unstable_pole_count"]),
            )
        )
    lines.extend(
        (
            "",
            "## Fitted thrust delay versus zero delay",
            "",
            "| case | covariance | stable fraction fitted | stable fraction zero | difference | median radius fitted | median radius zero | difference |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for item in summary["fitted_versus_zero_delay"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                item["case"],
                item["covariance_mode"],
                _value(item["fitted_delay_stable_fraction"]),
                _value(item["zero_delay_stable_fraction"]),
                _value(item["stable_fraction_difference_fitted_minus_zero"]),
                _value(item["fitted_delay_median_spectral_radius"]),
                _value(item["zero_delay_median_spectral_radius"]),
                _value(item["median_spectral_radius_difference_fitted_minus_zero"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-yaml", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fd-check-samples", type=int, default=DEFAULT_FD_CHECK_SAMPLES)
    parser.add_argument("--controller-dt", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-mode", action="append", choices=COVARIANCE_MODES,
        help="Repeat to restrict the default two covariance modes.",
    )
    parser.add_argument(
        "--delay-mode", action="append", choices=DELAY_MODES,
        help="Repeat to restrict the default fitted/zero delay modes.",
    )
    parser.add_argument(
        "--case", action="append", choices=tuple(CASE_DEFINITIONS),
        help="Repeat to restrict the default three bags.",
    )
    return parser


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(arguments.output_dir).expanduser().resolve()
    covariance_modes = tuple(arguments.covariance_mode or DEFAULT_COVARIANCE_MODES)
    delay_modes = tuple(arguments.delay_mode or DEFAULT_DELAY_MODES)
    cases = tuple(arguments.case or CASE_DEFINITIONS)
    rows: list[Mapping[str, Any]] = []
    for name in cases:
        definition = CASE_DEFINITIONS[name]
        estimator = Path(definition["estimator"])
        for covariance_mode in covariance_modes:
            for delay_mode in delay_modes:
                report, arrays, status = analyze_case(
                    result_path=estimator / "result.json",
                    arrays_path=estimator / "arrays.npz",
                    static_postprocess_path=Path(definition["static"]),
                    arguments_path=estimator / "arguments.json",
                    bag_json_path=Path(definition["bag_json"]),
                    controller_yaml_path=arguments.controller_yaml,
                    vehicle_model_path=arguments.vehicle_model,
                    covariance_mode=covariance_mode,
                    sample_count=arguments.samples,
                    seed=arguments.seed,
                    delay_mode=delay_mode,
                    controller_dt_override=arguments.controller_dt,
                    fd_check_samples=arguments.fd_check_samples,
                    flight_outcome=str(definition["outcome"]),
                )
                case_output = output / covariance_mode / delay_mode / name
                write_outputs(case_output, report, arrays, status)
                rows.append(_case_row(name, report))
    summary = build_summary(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "three_bag_local_pole_summary.json", summary)
    (output / "three_bag_local_pole_summary.md").write_text(
        render_summary_markdown(summary), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    execute(build_argument_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
