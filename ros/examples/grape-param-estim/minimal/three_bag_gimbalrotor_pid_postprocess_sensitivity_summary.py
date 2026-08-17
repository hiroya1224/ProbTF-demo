#!/usr/bin/env python3
"""Three-bag within-vs-between summary for PID sensitivity reports."""

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
from gimbalrotor_pid_postprocess_sensitivity import (  # noqa: E402
    SENSITIVITY_SCHEMA,
    write_json,
)


SUMMARY_SCHEMA = (
    "grape-param-estim/"
    "gimbalrotor-pid-postprocess-sensitivity-three-bag/v1"
)


def _load(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "sensitivity report cannot be read: {}".format(source)
        ) from error
    if not isinstance(payload, dict) or payload.get("schema") != SENSITIVITY_SCHEMA:
        raise ValueError(
            "input is not a {} report: {}".format(
                SENSITIVITY_SCHEMA, source
            )
        )
    return payload


def parse_named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use LABEL=PATH"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label or not path.strip():
        raise argparse.ArgumentTypeError(
            "--input must use non-empty LABEL=PATH"
        )
    return label, Path(path)


def build_summary(
    named_reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if len(named_reports) < 2:
        raise ValueError("at least two sensitivity reports are required")
    labels = [str(label) for label, _report in named_reports]
    if len(set(labels)) != len(labels):
        raise ValueError("sensitivity report labels must be unique")
    rows = {}
    for label, report in named_reports:
        analysis = report["center_and_eigen_sensitivity"]
        rows[label] = {
            "covariance_mode": report["input"]["covariance_mode"],
            "estimator_result_json": report["input"][
                "estimator_result_json"
            ],
            "groups": {
                group: {
                    "center_scale": float(
                        analysis["group_summary"][group]["center_scale"]
                    ),
                    "within_local_linear_one_sigma": float(
                        analysis["group_summary"][group][
                            "linearized_one_sigma"
                        ]
                    ),
                    "relative_within_local_linear_one_sigma": (
                        analysis["group_summary"][group][
                            "relative_linearized_one_sigma"
                        ]
                    ),
                    "eigen_point_min": float(
                        analysis["group_summary"][group][
                            "sigma_point_min"
                        ]
                    ),
                    "eigen_point_max": float(
                        analysis["group_summary"][group][
                            "sigma_point_max"
                        ]
                    ),
                }
                for group in PID_GROUPS
            },
        }
    comparison = {}
    for group in PID_GROUPS:
        centers = np.asarray(
            [rows[label]["groups"][group]["center_scale"] for label in labels],
            dtype=float,
        )
        within = np.asarray(
            [
                rows[label]["groups"][group][
                    "within_local_linear_one_sigma"
                ]
                for label in labels
            ],
            dtype=float,
        )
        between_std = float(np.std(centers, ddof=0))
        within_rms = float(np.sqrt(np.mean(within**2)))
        comparison[group] = {
            "center_min": float(np.min(centers)),
            "center_max": float(np.max(centers)),
            "center_mean": float(np.mean(centers)),
            "between_bag_center_standard_deviation": between_std,
            "within_bag_local_sigma_rms": within_rms,
            "between_to_within_std_ratio": (
                between_std / within_rms if within_rms > 0.0 else None
            ),
            "note": (
                "descriptive comparison only; with this small bag count "
                "this is not a random-effects inference"
            ),
        }
    covariance_modes = sorted(
        {rows[label]["covariance_mode"] for label in labels}
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "bag_labels": labels,
        "covariance_modes": covariance_modes,
        "per_bag": rows,
        "within_vs_between": comparison,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    labels = summary["bag_labels"]
    lines = [
        "# Gimbalrotor PID sensitivity: within-bag vs between-bag",
        "",
        "This comparison separates local sensitivity around each identified plant",
        "from variation of the point estimates across bags. It is descriptive;",
        "with only a few bags it is not a random-effects inference.",
        "",
        "## Per-bag local sensitivity",
        "",
        "| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        for group in PID_GROUPS:
            item = summary["per_bag"][label]["groups"][group]
            relative = item["relative_within_local_linear_one_sigma"]
            lines.append(
                "| {} | {} | {:.8g} | {:.8g} | {:.3%} | {:.8g} | {:.8g} |".format(
                    label,
                    group,
                    item["center_scale"],
                    item["within_local_linear_one_sigma"],
                    0.0 if relative is None else relative,
                    item["eigen_point_min"],
                    item["eigen_point_max"],
                )
            )
    lines.extend(
        (
            "",
            "## Within-vs-between scale",
            "",
            "| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for group in PID_GROUPS:
        item = summary["within_vs_between"][group]
        ratio = item["between_to_within_std_ratio"]
        lines.append(
            "| {} | {:.8g} | {:.8g} | {:.8g} | {:.8g} | {} |".format(
                group,
                item["center_min"],
                item["center_max"],
                item["between_bag_center_standard_deviation"],
                item["within_bag_local_sigma_rms"],
                "undefined" if ratio is None else "{:.6g}".format(ratio),
            )
        )
    lines.extend(
        (
            "",
            "Interpretation guide:",
            "",
            "- `within-bag sigma RMS >> between-bag std`: the gain correction is",
            "  locally weakly determined by each fitted plant.",
            "- `between-bag std >> within-bag sigma RMS`: each fit may be locally",
            "  sharp, while different bags identify different effective plants.",
            "- Comparable values indicate that both effects matter.",
            "",
            "Do not average the bag-specific gain proposals from this table.",
            "",
        )
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local PID-correction sensitivity with between-bag "
            "variation."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=parse_named_report,
        required=True,
        help="Repeat as LABEL=path/to/pid_gain_sensitivity.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    named = [(label, _load(path)) for label, path in arguments.input]
    summary = build_summary(named)
    directory = arguments.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "gimbalrotor_pid_sensitivity_three_bag.json",
        summary,
    )
    markdown = render_markdown(summary)
    (directory / "gimbalrotor_pid_sensitivity_three_bag.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown, end="")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        execute(arguments)
    except (ValueError, OSError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
