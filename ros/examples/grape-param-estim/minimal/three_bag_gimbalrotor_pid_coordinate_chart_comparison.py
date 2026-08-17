#!/usr/bin/env python3
"""Run the coordinate-chart PID sensitivity comparison for several bags.

The intended production use is the current three-bag set (failure1, failure2,
success), but the CLI accepts any named set.  Every input is evaluated in both
the original estimator quotient chart and the estimate-centered scale-free SPD
chart at 0.5, 1, and 2 covariance sigma by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller_config import PID_GROUPS  # noqa: E402
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    PostprocessInputError,
    PostprocessNumericalError,
    load_estimator_result,
    load_vehicle_model,
)
from gimbalrotor_pid_postprocess_sensitivity import (  # noqa: E402
    COORDINATE_MODES,
    COVARIANCE_MODES,
    DEFAULT_DERIVATIVE_SIGMA_FRACTION,
    SENSITIVITY_SCHEMA,
    analyze_eigen_directions,
    analyze_monte_carlo,
    build_report,
    load_sensitivity_artifacts,
    render_markdown,
    source_commit,
    write_json,
)


COMPARISON_SCHEMA = (
    "grape-param-estim/gimbalrotor-pid-coordinate-chart-comparison/v1"
)
DEFAULT_COVARIANCE_MODES = (
    "conservative_fusion",
    "overlap_corrected",
)
DEFAULT_SIGMA_MULTIPLES = (0.5, 1.0, 2.0)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use NAME=/path/to/result.json"
        )
    name, raw_path = value.split("=", 1)
    selected_name = name.strip()
    if not selected_name:
        raise argparse.ArgumentTypeError("input name cannot be empty")
    return selected_name, Path(raw_path)


def _sigma_directory(value: float) -> str:
    text = "{:g}".format(float(value)).replace("-", "m").replace(".", "p")
    return "sigma_{}".format(text)


def _write_report_bundle(
    directory: Path, report: Mapping[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "pid_gain_sensitivity.json", report)
    markdown = render_markdown(report)
    (directory / "pid_gain_sensitivity.md").write_text(
        markdown, encoding="utf-8"
    )
    eigen = report["center_and_eigen_sensitivity"]["eigen_sampling"]
    mc = report["monte_carlo"]
    write_json(
        directory / "status.json",
        {
            "schema": SENSITIVITY_SCHEMA + "/status/v1",
            "status": "completed",
            "source_commit": report["source_commit"],
            "covariance_mode": report["input"]["covariance_mode"],
            "coordinate_mode": report["input"]["coordinate_mode"],
            "finite_sigma_multiple": eigen["sigma_multiple"],
            "valid_eigen_sample_count": eigen["valid_sample_count"],
            "invalid_eigen_sample_count": eigen["invalid_sample_count"],
            "valid_local_derivative_direction_count": (
                eigen["derivative_valid_direction_count"]
            ),
            "monte_carlo_enabled": bool(mc.get("enabled", False)),
            "monte_carlo_valid_sample_count": mc.get(
                "valid_sample_count"
            ),
            "monte_carlo_invalid_sample_count": mc.get(
                "invalid_sample_count"
            ),
        },
    )


def _relative_difference(first: Optional[float], second: Optional[float]):
    if first is None or second is None:
        return None
    denominator = max(abs(float(first)), abs(float(second)))
    if denominator == 0.0:
        return 0.0
    return abs(float(first) - float(second)) / denominator


def build_comparison(
    reports: Mapping[str, Mapping[str, Mapping[str, Mapping[float, Mapping[str, Any]]]]]
) -> Mapping[str, Any]:
    covariance_summary: dict[str, Any] = {}
    for covariance_mode, bags in reports.items():
        bag_summary: dict[str, Any] = {}
        for bag_name, coordinate_reports in bags.items():
            per_group = {}
            reference_sigma = sorted(
                coordinate_reports["estimator_quotient"]
            )[0]
            estimator_one = coordinate_reports["estimator_quotient"][
                reference_sigma
            ]["center_and_eigen_sensitivity"]
            centered_one = coordinate_reports[
                "centered_scale_free_spd"
            ][reference_sigma]["center_and_eigen_sensitivity"]
            for group in PID_GROUPS:
                estimator_local = estimator_one["group_summary"][group][
                    "linearized_one_sigma"
                ]
                centered_local = centered_one["group_summary"][group][
                    "linearized_one_sigma"
                ]
                finite = {}
                for sigma in sorted(
                    coordinate_reports["estimator_quotient"]
                ):
                    finite[str(sigma)] = {}
                    for coordinate_mode in COORDINATE_MODES:
                        analysis = coordinate_reports[coordinate_mode][
                            sigma
                        ]["center_and_eigen_sensitivity"]
                        item = analysis["group_summary"][group]
                        finite[str(sigma)][coordinate_mode] = {
                            "finite_secant_one_sigma": item[
                                "finite_secant_one_sigma"
                            ],
                            "finite_to_local_sigma_ratio": item[
                                "finite_to_local_sigma_ratio"
                            ],
                            "sigma_point_min": item["sigma_point_min"],
                            "sigma_point_max": item["sigma_point_max"],
                            "finite_secant_complete": item[
                                "finite_secant_complete"
                            ],
                        }
                per_group[group] = {
                    "center_scale": estimator_one["group_summary"][group][
                        "center_scale"
                    ],
                    "local_one_sigma_estimator_quotient": (
                        estimator_local
                    ),
                    "local_one_sigma_centered_scale_free_spd": (
                        centered_local
                    ),
                    "local_coordinate_relative_difference": (
                        _relative_difference(
                            estimator_local, centered_local
                        )
                    ),
                    "finite_sigma_comparison": finite,
                }
            validity = {}
            for sigma in sorted(
                coordinate_reports["estimator_quotient"]
            ):
                validity[str(sigma)] = {}
                for coordinate_mode in COORDINATE_MODES:
                    analysis = coordinate_reports[coordinate_mode][
                        sigma
                    ]["center_and_eigen_sensitivity"]
                    validity[str(sigma)][coordinate_mode] = {
                        "valid_sample_count": analysis["eigen_sampling"][
                            "valid_sample_count"
                        ],
                        "invalid_sample_count": analysis[
                            "eigen_sampling"
                        ]["invalid_sample_count"],
                        "valid_direction_pair_count": analysis[
                            "eigen_sampling"
                        ]["valid_direction_pair_count"],
                    }
            transform = centered_one["coordinate_transform"]
            bag_summary[bag_name] = {
                "groups": per_group,
                "finite_sampling_validity": validity,
                "centered_chart_pushforward_condition_number": (
                    transform["condition_number"]
                ),
                "centered_chart_pushforward_singular_values": (
                    transform["singular_values"]
                ),
            }
        covariance_summary[covariance_mode] = bag_summary
    return {
        "schema": COMPARISON_SCHEMA,
        "interpretation": {
            "purpose": (
                "compare finite nonlinear sensitivity in the original "
                "estimator quotient chart and an estimate-centered "
                "scale-free SPD chart"
            ),
            "local_invariance_check": (
                "the infinitesimal propagated one-sigma should agree "
                "between coordinate charts up to numerical differentiation "
                "and covariance push-forward error"
            ),
            "straightening_diagnostic": (
                "for finite excursions, compare finite_secant_one_sigma / "
                "infinitesimal_one_sigma across 0.5, 1, and 2 sigma; "
                "values staying near one indicate a more nearly linear map "
                "over that excursion"
            ),
            "automatic_preference": False,
        },
        "covariance_modes": covariance_summary,
    }


def _fmt(value: Any, fmt: str = ".6g") -> str:
    if value is None:
        return "incomplete"
    return format(float(value), fmt)


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Gimbalrotor PID coordinate-chart comparison",
        "",
        "The same fitted plants and local estimator covariance are propagated",
        "through two coordinate charts:",
        "",
        "- `estimator_quotient`: the existing common-scale quotient chart;",
        "- `centered_scale_free_spd`: an estimate-centered chart on",
        "  `SPD(3) x R^3 x R_+^4` using the scale-free second moment, CoG,",
        "  and log force-over-mass ratios.",
        "",
        "The infinitesimal one-sigma result is a coordinate-invariance sanity",
        "check. The finite 0.5/1/2-sigma secants are the chart-curvature test.",
        "No threshold automatically declares either chart superior.",
        "",
    ]
    for covariance_mode, bags in comparison["covariance_modes"].items():
        lines.extend(
            (
                "## Covariance: `{}`".format(covariance_mode),
                "",
                "### Infinitesimal coordinate-invariance check",
                "",
                (
                    "| bag | group | center | estimator quotient sigma | "
                    "centered SPD sigma | relative difference |"
                ),
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for bag_name, bag in bags.items():
            for group in PID_GROUPS:
                item = bag["groups"][group]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} |".format(
                        bag_name,
                        group,
                        _fmt(item["center_scale"]),
                        _fmt(
                            item[
                                "local_one_sigma_estimator_quotient"
                            ]
                        ),
                        _fmt(
                            item[
                                "local_one_sigma_centered_scale_free_spd"
                            ]
                        ),
                        (
                            "incomplete"
                            if item[
                                "local_coordinate_relative_difference"
                            ]
                            is None
                            else "{:.3%}".format(
                                item[
                                    "local_coordinate_relative_difference"
                                ]
                            )
                        ),
                    )
                )
        lines.extend(
            (
                "",
                "### Finite-sigma nonlinearity",
                "",
                (
                    "| bag | group | sigma | estimator finite/local | "
                    "centered SPD finite/local | estimator envelope | "
                    "centered SPD envelope |"
                ),
                "|---|---|---:|---:|---:|---|---|",
            )
        )
        for bag_name, bag in bags.items():
            for group in PID_GROUPS:
                item = bag["groups"][group]
                for sigma, modes in item[
                    "finite_sigma_comparison"
                ].items():
                    estimator = modes["estimator_quotient"]
                    centered = modes["centered_scale_free_spd"]
                    lines.append(
                        "| {} | {} | {} | {} | {} | [{}, {}] | [{}, {}] |".format(
                            bag_name,
                            group,
                            sigma,
                            _fmt(
                                estimator[
                                    "finite_to_local_sigma_ratio"
                                ]
                            ),
                            _fmt(
                                centered[
                                    "finite_to_local_sigma_ratio"
                                ]
                            ),
                            _fmt(estimator["sigma_point_min"]),
                            _fmt(estimator["sigma_point_max"]),
                            _fmt(centered["sigma_point_min"]),
                            _fmt(centered["sigma_point_max"]),
                        )
                    )
        lines.extend(
            (
                "",
                "### Finite-sample validity",
                "",
                "| bag | sigma | coordinate | valid / 27 | invalid |",
                "|---|---:|---|---:|---:|",
            )
        )
        for bag_name, bag in bags.items():
            for sigma, modes in bag[
                "finite_sampling_validity"
            ].items():
                for coordinate_mode in COORDINATE_MODES:
                    value = modes[coordinate_mode]
                    lines.append(
                        "| {} | {} | {} | {} / 27 | {} |".format(
                            bag_name,
                            sigma,
                            coordinate_mode,
                            value["valid_sample_count"],
                            value["invalid_sample_count"],
                        )
                    )
        lines.extend(
            (
                "",
                "A large allocation condition number or loss of the source",
                "threshold rank is not an invalidation criterion in these runs.",
                "Only an actually non-finite or mathematically undefined",
                "floating-point calculation is marked invalid.",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def run(
    *,
    inputs: Sequence[tuple[str, Path]],
    vehicle_model_path: Path,
    output_dir: Path,
    covariance_modes: Sequence[str],
    sigma_multiples: Sequence[float],
    derivative_sigma_fraction: float,
    monte_carlo_samples: int,
    seed: int,
) -> Mapping[str, Any]:
    model = load_vehicle_model(vehicle_model_path)
    revision = source_commit()
    all_reports: dict[str, Any] = {}
    for covariance_mode in covariance_modes:
        covariance_reports: dict[str, Any] = {}
        for bag_name, result_path in inputs:
            result = load_estimator_result(result_path)
            arrays_path = (
                Path(result_path).expanduser().resolve().parent
                / "arrays.npz"
            )
            artifacts = load_sensitivity_artifacts(
                arrays_path, covariance_mode=covariance_mode
            )
            bag_reports: dict[str, Any] = {}
            for coordinate_mode in COORDINATE_MODES:
                coordinate_reports: dict[float, Any] = {}
                monte_carlo = None
                for sigma in sigma_multiples:
                    analysis = analyze_eigen_directions(
                        result=result,
                        artifacts=artifacts,
                        model=model,
                        sigma_multiple=sigma,
                        coordinate_mode=coordinate_mode,
                        derivative_sigma_fraction=(
                            derivative_sigma_fraction
                        ),
                    )
                    if monte_carlo is None:
                        monte_carlo = analyze_monte_carlo(
                            result=result,
                            artifacts=artifacts,
                            model=model,
                            sample_count=monte_carlo_samples,
                            seed=seed,
                            characteristic_length_m=analysis[
                                "characteristic_length_m"
                            ],
                            coordinate_mode=coordinate_mode,
                        )
                    report = build_report(
                        revision=revision,
                        result=result,
                        artifacts=artifacts,
                        model=model,
                        eigen_analysis=analysis,
                        monte_carlo=monte_carlo,
                    )
                    coordinate_reports[float(sigma)] = report
                    destination = (
                        output_dir
                        / covariance_mode
                        / bag_name
                        / coordinate_mode
                        / _sigma_directory(sigma)
                    )
                    _write_report_bundle(destination, report)
                bag_reports[coordinate_mode] = coordinate_reports
            covariance_reports[bag_name] = bag_reports
        all_reports[covariance_mode] = covariance_reports
    comparison = build_comparison(all_reports)
    payload = {
        "schema": COMPARISON_SCHEMA,
        "source_commit": revision,
        "input": {
            "vehicle_model_json": str(
                Path(vehicle_model_path).expanduser().resolve()
            ),
            "bags": {
                name: str(Path(path).expanduser().resolve())
                for name, path in inputs
            },
            "covariance_modes": list(covariance_modes),
            "coordinate_modes": list(COORDINATE_MODES),
            "sigma_multiples": [float(v) for v in sigma_multiples],
            "derivative_sigma_fraction": derivative_sigma_fraction,
            "monte_carlo_samples": int(monte_carlo_samples),
            "seed": int(seed),
        },
        "comparison": comparison,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "coordinate_chart_comparison.json", payload)
    markdown = render_comparison_markdown(comparison)
    (output_dir / "coordinate_chart_comparison.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown, end="")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare estimator-quotient and centered scale-free SPD "
            "sensitivity charts across the current three-bag style dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=_parse_named_path,
        action="append",
        required=True,
        help="Named estimator result: NAME=/path/to/result.json",
    )
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-mode",
        choices=COVARIANCE_MODES,
        action="append",
        dest="covariance_modes",
    )
    parser.add_argument(
        "--sigma-multiple",
        type=float,
        action="append",
        dest="sigma_multiples",
    )
    parser.add_argument(
        "--derivative-sigma-fraction",
        type=float,
        default=DEFAULT_DERIVATIVE_SIGMA_FRACTION,
    )
    parser.add_argument("--monte-carlo-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    covariance_modes = (
        tuple(arguments.covariance_modes)
        if arguments.covariance_modes
        else DEFAULT_COVARIANCE_MODES
    )
    sigma_multiples = (
        tuple(arguments.sigma_multiples)
        if arguments.sigma_multiples
        else DEFAULT_SIGMA_MULTIPLES
    )
    try:
        run(
            inputs=tuple(arguments.input),
            vehicle_model_path=arguments.vehicle_model,
            output_dir=arguments.output_dir.expanduser().resolve(),
            covariance_modes=covariance_modes,
            sigma_multiples=sigma_multiples,
            derivative_sigma_fraction=(
                arguments.derivative_sigma_fraction
            ),
            monte_carlo_samples=arguments.monte_carlo_samples,
            seed=arguments.seed,
        )
    except PostprocessInputError as error:
        print("input error: {}".format(error), file=sys.stderr)
        return 2
    except PostprocessNumericalError as error:
        print("numerical error: {}".format(error), file=sys.stderr)
        return 3
    except OSError as error:
        print("output error: {}".format(error), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
