#!/usr/bin/env python3
"""Run production Monte Carlo PID postprocessing for the three current bags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller_config import PID_GAIN_NAMES, PID_GROUPS  # noqa: E402
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    PostprocessInputError,
    PostprocessNumericalError,
)
from gimbalrotor_pid_monte_carlo_postprocess import (  # noqa: E402
    COVARIANCE_MODES,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    render_markdown as render_case_markdown,
    sample_pid_gain_distribution,
    write_outputs,
)
from gimbalrotor_pid_postprocess_sensitivity import (  # noqa: E402
    source_commit,
    write_json,
)


THREE_BAG_SCHEMA = (
    "grape-param-estim/gimbalrotor-pid-monte-carlo-three-bag/v1"
)
DEFAULT_COVARIANCE_MODES = ("conservative_fusion", "overlap_corrected")


def _parse_named_path(value: str, option: str) -> tuple[str, Path]:
    text = str(value)
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "{} must use NAME=PATH syntax".format(option)
        )
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "{} must use non-empty NAME=PATH syntax".format(option)
        )
    return name, Path(raw_path).expanduser().resolve()


def _named_mapping(values: Sequence[str], option: str) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, path = _parse_named_path(value, option)
        if name in result:
            raise PostprocessInputError(
                "duplicate case name {!r} in {}".format(name, option)
            )
        result[name] = path
    return result


def build_summary(cases: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Mapping[str, Any]:
    summary: dict[str, Any] = {}
    for covariance_mode, mode_cases in cases.items():
        case_summary: dict[str, Any] = {}
        for name, report in mode_cases.items():
            groups: dict[str, Any] = {}
            for group in PID_GROUPS:
                scale = report["gain_scale_distribution"][group]
                proposal = report["pid_gain_proposal"][group]
                groups[group] = {
                    "point_scale": report["center"]["scales"][group],
                    "scale_quantiles": dict(scale["quantiles"]),
                    "scale_standard_deviation": scale["standard_deviation"],
                    "nonpositive_fraction": scale["nonpositive_fraction"],
                    "gains": {
                        gain: dict(proposal["gains"][gain])
                        for gain in PID_GAIN_NAMES
                    },
                }
            case_summary[name] = {
                "sampling": dict(report["sampling"]),
                "warnings": list(report["warnings"]),
                "groups": groups,
                "joint_gain_scale_distribution": dict(
                    report["joint_gain_scale_distribution"]
                ),
            }
        summary[covariance_mode] = case_summary
    return {
        "schema": THREE_BAG_SCHEMA,
        "source_commit": source_commit(),
        "coordinate_mode": "estimator_quotient",
        "covariance_modes": list(cases),
        "cases": summary,
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Three-bag Gimbalrotor PID Monte Carlo proposals",
        "",
        "Each estimator Gaussian is sampled in its native common-scale quotient",
        "coordinate and propagated through the same nonlinear static PID map.",
        "No Gaussian is fitted to the output PID scale distribution.",
        "",
    ]
    for covariance_mode, cases in summary["cases"].items():
        lines.extend(
            (
                "## Covariance: `{}`".format(covariance_mode),
                "",
                "### Gain-scale proposals",
                "",
                (
                    "| case | group | point | median | 16–84% | "
                    "2.5–97.5% | std | nonpositive | valid samples |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for name, case in cases.items():
            for group in PID_GROUPS:
                item = case["groups"][group]
                q = item["scale_quantiles"]
                lines.append(
                    "| {} | {} | {:.8g} | {:.8g} | [{:.8g}, {:.8g}] | [{:.8g}, {:.8g}] | {:.8g} | {:.3%} | {}/{} |".format(
                        name,
                        group,
                        item["point_scale"],
                        q["q50"],
                        q["q16"],
                        q["q84"],
                        q["q025"],
                        q["q975"],
                        item["scale_standard_deviation"],
                        item["nonpositive_fraction"],
                        case["sampling"]["valid_count"],
                        case["sampling"]["requested_count"],
                    )
                )
        lines.extend(("", "### Median PID proposals", ""))
        for name, case in cases.items():
            lines.extend(
                (
                    "#### {}".format(name),
                    "",
                    "| group | gain | recorded | median | 16–84% | 2.5–97.5% |",
                    "|---|---|---:|---:|---:|---:|",
                )
            )
            for group in PID_GROUPS:
                for gain in PID_GAIN_NAMES:
                    item = case["groups"][group]["gains"][gain]
                    lines.append(
                        "| {} | {} | {:.8g} | {:.8g} | [{:.8g}, {:.8g}] | [{:.8g}, {:.8g}] |".format(
                            group,
                            gain,
                            item["recorded"],
                            item["median"],
                            item["range_68"][0],
                            item["range_68"][1],
                            item["range_95"][0],
                            item["range_95"][1],
                        )
                    )
            lines.append("")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run native-Gaussian Monte Carlo PID postprocessing for multiple "
            "named cases, normally failure1/failure2/success."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Named estimator result as NAME=PATH; repeat per case.",
    )
    parser.add_argument(
        "--static-postprocess",
        action="append",
        required=True,
        help="Named static pid_gain_postprocess.json as NAME=PATH; repeat per case.",
    )
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-mode",
        action="append",
        choices=COVARIANCE_MODES,
        help=(
            "Covariance mode; repeat to run several. Defaults to "
            "conservative_fusion and overlap_corrected."
        ),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--characteristic-length", type=float)
    return parser


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    results = _named_mapping(arguments.result, "--result")
    statics = _named_mapping(arguments.static_postprocess, "--static-postprocess")
    if set(results) != set(statics):
        raise PostprocessInputError(
            "--result and --static-postprocess must contain the same case names"
        )
    modes = tuple(arguments.covariance_mode or DEFAULT_COVARIANCE_MODES)
    output = Path(arguments.output_dir).expanduser().resolve()
    all_reports: dict[str, dict[str, Mapping[str, Any]]] = {}
    for covariance_mode in modes:
        mode_reports: dict[str, Mapping[str, Any]] = {}
        for name, result_path in results.items():
            arrays_path = result_path.parent / "arrays.npz"
            report, samples = sample_pid_gain_distribution(
                result_path=result_path,
                arrays_path=arrays_path,
                static_postprocess_path=statics[name],
                vehicle_model_path=arguments.vehicle_model,
                covariance_mode=covariance_mode,
                sample_count=arguments.samples,
                seed=arguments.seed,
                characteristic_length_override=arguments.characteristic_length,
            )
            case_output = output / covariance_mode / name
            write_outputs(output_dir=case_output, report=report, samples=samples)
            mode_reports[name] = report
        all_reports[covariance_mode] = mode_reports
    summary = build_summary(all_reports)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "three_bag_pid_monte_carlo_summary.json", summary)
    markdown = render_summary_markdown(summary)
    (output / "three_bag_pid_monte_carlo_summary.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown, end="")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        execute(arguments)
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
