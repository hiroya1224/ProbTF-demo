#!/usr/bin/env python3
"""Aggregate three completed static Gimbalrotor PID proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller_config import (  # noqa: E402
    PID_GAIN_NAMES,
    PID_GROUPS,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    POSTPROCESS_SCHEMA,
    PostprocessInputError,
)


SUMMARY_SCHEMA = (
    "grape-param-estim/gimbalrotor-pid-postprocess-three-bag/v1"
)
EXPECTED_LABELS = ("failure1", "failure2", "success")


def source_commit(repository_root: Path = _PROJECT_ROOT) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessInputError(
            "{} cannot be read: {}".format(label, source)
        ) from error
    if not isinstance(payload, Mapping):
        raise PostprocessInputError("{} must contain an object".format(label))
    return payload


def load_completed_report(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    report = _read_mapping(source, "PID postprocess report")
    if report.get("schema") != POSTPROCESS_SCHEMA:
        raise PostprocessInputError("PID postprocess schema is unsupported")
    status = _read_mapping(source.parent / "status.json", "PID status")
    if status.get("status") != "completed":
        raise PostprocessInputError("PID postprocess is not completed")
    if status.get("source_commit") != report.get("source_commit"):
        raise PostprocessInputError("PID report/status source commits differ")
    return report


def build_three_bag_summary(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    summary_source_commit: Optional[str] = None,
) -> Mapping[str, Any]:
    labels = tuple(reports)
    if set(labels) != set(EXPECTED_LABELS) or len(labels) != len(
        EXPECTED_LABELS
    ):
        raise PostprocessInputError(
            "reports must contain exactly failure1, failure2, and success"
        )
    commits = {str(reports[label].get("source_commit")) for label in labels}
    if len(commits) != 1:
        raise PostprocessInputError(
            "three PID reports must use one source commit"
        )
    report_commit = next(iter(commits))
    revision = (
        report_commit
        if summary_source_commit is None
        else str(summary_source_commit)
    )
    if revision != report_commit:
        raise PostprocessInputError(
            "summary source commit must match the PID report source commit"
        )
    bag_rows = {}
    scale_values = {group: [] for group in PID_GROUPS}
    for label in EXPECTED_LABELS:
        report = reports[label]
        if report.get("schema") != POSTPROCESS_SCHEMA:
            raise PostprocessInputError(
                "{} report schema is unsupported".format(label)
            )
        groups = report.get("gain_groups")
        overall = report.get("overall")
        source = report.get("input")
        if not all(isinstance(value, Mapping) for value in (groups, overall, source)):
            raise PostprocessInputError(
                "{} report is missing required sections".format(label)
            )
        group_rows = {}
        for group in PID_GROUPS:
            value = groups.get(group)
            if not isinstance(value, Mapping):
                raise PostprocessInputError(
                    "{} report is missing group {}".format(label, group)
                )
            scale = float(value["scale"])
            if not np.isfinite(scale) or scale <= 0.0:
                raise PostprocessInputError(
                    "{} group {} scale is invalid".format(label, group)
                )
            scale_values[group].append(scale)
            group_rows[group] = {
                "scale": scale,
                "old": dict(value["old"]),
                "proposed": dict(value["proposed"]),
            }
        bag_rows[label] = {
            "estimator_result_json": source["estimator_result_json"],
            "estimator_source_commit": source["estimator_source_commit"],
            "bag_json": source["bag_json"],
            "bag_path": source["bag_path"],
            "bag_interval_seconds": list(source["bag_interval_seconds"]),
            "controller_yaml": source["controller_yaml"],
            "controller_yaml_sha256": source["controller_yaml_sha256"],
            "controller_gain_source": source["controller_gain_source"],
            "controller_gain_snapshot": dict(
                report["controller_gain_snapshot"]
            ),
            "gain_groups": group_rows,
            "error_before_frobenius": float(
                overall["error_before_frobenius"]
            ),
            "error_after_frobenius": float(
                overall["error_after_frobenius"]
            ),
            "off_diagonal_coupling_ratio": float(
                overall["off_diagonal_coupling_ratio"]
            ),
            "proposal_status": str(overall["proposal_status"]),
            "warnings": list(overall["warnings"]),
        }
    controller_hashes = {
        row["controller_yaml_sha256"] for row in bag_rows.values()
    }
    if len(controller_hashes) != 1:
        raise PostprocessInputError(
            "three PID reports must use one controller YAML"
        )
    statistics = {}
    for group in PID_GROUPS:
        values = np.asarray(scale_values[group], dtype=float)
        statistics[group] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "source_commit": revision,
        "method": "diagnostic_comparison_without_automatic_averaging",
        "bag_order": list(EXPECTED_LABELS),
        "bags": bag_rows,
        "scale_statistics": statistics,
        "deployment_yaml_generated": False,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gimbalrotor PID static postprocess: three-bag comparison",
        "",
        "Source commit: `{}`".format(summary["source_commit"]),
        "",
        "No mean deployment YAML is generated. These are diagnostic, "
        "bag-specific proposals.",
        "",
        "| bag | xy scale | z scale | roll_pitch scale | yaw scale | "
        "error before | error after | coupling | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in EXPECTED_LABELS:
        row = summary["bags"][label]
        groups = row["gain_groups"]
        lines.append(
            "| {} | {:.9g} | {:.9g} | {:.9g} | {:.9g} | {:.9g} | "
            "{:.9g} | {:.9g} | {} |".format(
                label,
                groups["xy"]["scale"],
                groups["z"]["scale"],
                groups["roll_pitch"]["scale"],
                groups["yaw"]["scale"],
                row["error_before_frobenius"],
                row["error_after_frobenius"],
                row["off_diagonal_coupling_ratio"],
                row["proposal_status"],
            )
        )
    lines.extend(
        (
            "",
            "## Scale spread",
            "",
            "| group | min | max | mean | standard deviation |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for group in PID_GROUPS:
        stats = summary["scale_statistics"][group]
        lines.append(
            "| {} | {:.9g} | {:.9g} | {:.9g} | {:.9g} |".format(
                group,
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["standard_deviation"],
            )
        )
    for label in EXPECTED_LABELS:
        lines.extend(("", "## {} gains".format(label), ""))
        lines.extend(
            (
                "| group | scale | P old -> proposed | I old -> proposed | "
                "D old -> proposed |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for group in PID_GROUPS:
            value = summary["bags"][label]["gain_groups"][group]
            old = value["old"]
            proposed = value["proposed"]
            lines.append(
                "| {} | {:.9g} | {:.9g} -> {:.9g} | {:.9g} -> {:.9g} | "
                "{:.9g} -> {:.9g} |".format(
                    group,
                    value["scale"],
                    old["p_gain"],
                    proposed["p_gain"],
                    old["i_gain"],
                    proposed["i_gain"],
                    old["d_gain"],
                    proposed["d_gain"],
                )
            )
    return "\n".join(lines) + "\n"


def _parse_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must have LABEL=PATH form")
    label, path = value.split("=", 1)
    if label not in EXPECTED_LABELS or not path:
        raise argparse.ArgumentTypeError(
            "LABEL must be failure1, failure2, or success"
        )
    return label, Path(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exactly three completed static PID proposals."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input,
        required=True,
        help="failure1=PATH, failure2=PATH, or success=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        paths = dict(arguments.input)
        if len(paths) != 3:
            raise PostprocessInputError(
                "provide each of failure1, failure2, and success exactly once"
            )
        reports = {
            label: load_completed_report(paths[label])
            for label in EXPECTED_LABELS
        }
        summary = build_three_bag_summary(
            reports, summary_source_commit=source_commit()
        )
        directory = arguments.output_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "gimbalrotor_pid_postprocess_three_bag.json",
            summary,
        )
        (directory / "gimbalrotor_pid_postprocess_three_bag.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        write_json(
            directory / "status.json",
            {
                "schema": SUMMARY_SCHEMA + "/status/v1",
                "status": "completed",
                "source_commit": summary["source_commit"],
                "deployment_yaml_generated": False,
            },
        )
    except (PostprocessInputError, OSError, KeyError, TypeError, ValueError) as error:
        print("summary error: {}".format(error), file=sys.stderr)
        return 2
    print(render_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
