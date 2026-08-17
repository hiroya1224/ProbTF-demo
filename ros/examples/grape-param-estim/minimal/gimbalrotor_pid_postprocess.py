#!/usr/bin/env python3
"""CLI for the static Gimbalrotor PID gain postprocessor."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import yaml


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
    DEFAULT_LARGE_SCALE_MAX,
    DEFAULT_LARGE_SCALE_MIN,
    DEFAULT_STRONG_COUPLING_THRESHOLD,
    POSTPROCESS_SCHEMA,
    PostprocessInputError,
    PostprocessNumericalError,
    apply_gain_corrections_to_yaml,
    build_report,
    compute_static_pid_proposal,
    load_bag_provenance,
    load_controller_yaml,
    load_estimator_result,
    load_vehicle_model,
)


def source_commit(repository_root: Path = _PROJECT_ROOT) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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


def _without_gain_leaves(document: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = deepcopy(dict(document))
    controller = copied["controller"]
    for group in PID_GROUPS:
        for gain in PID_GAIN_NAMES:
            del controller[group][gain]
    return copied


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def terminal_summary(report: Mapping[str, Any]) -> str:
    source = report["input"]
    plant = report["scale_free_plant"]
    lines = [
        "Gimbalrotor PID static-effectiveness postprocess",
        "",
        "plant result: {}".format(source["estimator_result_json"]),
        "bag: {}".format(source["bag_path"]),
        "bag interval: [{:.6g}, {:.6g}] s".format(
            *source["bag_interval_seconds"]
        ),
        "estimator source commit: {}".format(
            source["estimator_source_commit"]
        ),
        "controller YAML: {}".format(source["controller_yaml"]),
        "rotor lag: {:.9g} s".format(plant["rotor_lag_seconds"]),
        "characteristic length: {:.9g} m".format(
            report["linearization"]["characteristic_length_m"]
        ),
        "",
        "group         scale      P old -> new       I old -> new       D old -> new",
    ]
    for group in PID_GROUPS:
        gain = report["gain_groups"][group]
        old = gain["old"]
        proposed = gain["proposed"]
        lines.append(
            "{:<13} {:>9.6g}  {:>8.6g} -> {:<9.6g} "
            "{:>8.6g} -> {:<9.6g} {:>8.6g} -> {:<9.6g}".format(
                group,
                gain["scale"],
                old["p_gain"],
                proposed["p_gain"],
                old["i_gain"],
                proposed["i_gain"],
                old["d_gain"],
                proposed["d_gain"],
            )
        )
    overall = report["overall"]
    lines.extend(
        (
            "",
            "Hbar error: before {:.9g} -> after {:.9g}".format(
                overall["error_before_frobenius"],
                overall["error_after_frobenius"],
            ),
            "coupling ratio: {:.9g}".format(
                overall["off_diagonal_coupling_ratio"]
            ),
            "proposal status: {}".format(overall["proposal_status"]),
            "warnings:",
        )
    )
    lines.extend("  - {}".format(value) for value in overall["warnings"])
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Propose four static Gimbalrotor PID gain-group scales from one "
            "scale-free estimator result."
        )
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--bag-json", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--controller-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-non-prior-free-result", action="store_true")
    parser.add_argument("--allow-point-estimate-only", action="store_true")
    parser.add_argument("--characteristic-length", type=float)
    parser.add_argument(
        "--large-scale-min", type=float, default=DEFAULT_LARGE_SCALE_MIN
    )
    parser.add_argument(
        "--large-scale-max", type=float, default=DEFAULT_LARGE_SCALE_MAX
    )
    parser.add_argument(
        "--strong-coupling-threshold",
        type=float,
        default=DEFAULT_STRONG_COUPLING_THRESHOLD,
    )
    return parser


def _failure_payload(
    arguments: argparse.Namespace,
    revision: str,
    stage: str,
    error: BaseException,
) -> Mapping[str, Any]:
    return {
        "schema": POSTPROCESS_SCHEMA + "/status/v1",
        "status": "failed",
        "source_commit": revision,
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "input": {
            "estimator_result_json": str(arguments.result),
            "bag_json": str(arguments.bag_json),
            "vehicle_model_json": str(arguments.vehicle_model),
            "controller_yaml": str(arguments.controller_yaml),
        },
    }


def _best_effort_failure_status(
    arguments: argparse.Namespace,
    revision: str,
    stage: str,
    error: BaseException,
) -> None:
    try:
        directory = arguments.output_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "status.json",
            _failure_payload(arguments, revision, stage, error),
        )
    except Exception:
        pass


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    revision = source_commit()
    stage = "input_validation"
    try:
        result = load_estimator_result(
            arguments.result,
            allow_non_prior_free_result=(
                arguments.allow_non_prior_free_result
            ),
            allow_point_estimate_only=arguments.allow_point_estimate_only,
        )
        bag = load_bag_provenance(arguments.bag_json)
        model = load_vehicle_model(arguments.vehicle_model)
        controller = load_controller_yaml(arguments.controller_yaml)
        stage = "static_allocation"
        proposal = compute_static_pid_proposal(
            result,
            model,
            controller,
            characteristic_length_override=arguments.characteristic_length,
            large_scale_min=arguments.large_scale_min,
            large_scale_max=arguments.large_scale_max,
            strong_coupling_threshold=arguments.strong_coupling_threshold,
        )
        report = build_report(
            source_commit=revision,
            result=result,
            bag=bag,
            model=model,
            controller=controller,
            proposal=proposal,
        )
        full_yaml, overlay_yaml = apply_gain_corrections_to_yaml(
            controller, proposal.corrections
        )
        if _without_gain_leaves(controller.document) != _without_gain_leaves(
            full_yaml
        ):
            raise PostprocessNumericalError(
                "full proposal YAML changed a non-gain field"
            )
        stage = "output_write"
        directory = arguments.output_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "pid_gain_postprocess.json", report)
        _write_yaml(directory / "pid_gain_overlay.yaml", overlay_yaml)
        _write_yaml(
            directory / "GimbalrotorControl.pid-proposal.yaml", full_yaml
        )
        status = {
            "schema": POSTPROCESS_SCHEMA + "/status/v1",
            "status": "completed",
            "source_commit": revision,
            "proposal_status": proposal.proposal_status,
            "warnings": list(proposal.warnings),
            "estimator_case_name": result.case_name,
            "estimator_source_commit": result.source_commit,
        }
        write_json(directory / "status.json", status)
        print(terminal_summary(report), end="")
        return report
    except (PostprocessInputError, PostprocessNumericalError, OSError) as error:
        _best_effort_failure_status(arguments, revision, stage, error)
        raise


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
