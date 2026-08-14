#!/usr/bin/env python3
"""Failure-isolated fixed and configured ablations for one rosbag."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Optional, Sequence


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from single_bag_savgol_core import BASE_PLAN_COMMIT  # noqa: E402
from single_bag_savgol_estimator import (  # noqa: E402
    build_argument_parser as build_estimator_argument_parser,
    resolve_bag_arguments,
    run_estimator,
)
from single_bag_savgol_reports import (  # noqa: E402
    json_sanitize,
    output_run_directory,
    source_commit,
    write_failure_report_pdf,
    write_json,
)


FIXED_CASE_NAMES = (
    "default_full_covariance",
    "cov_identity",
    "cov_diagonal",
    "cov_block_s_alpha",
    "cov_full_no_R_uncertainty_in_s",
    "cov_full_no_position_rotation_cross",
    "cov_global_full",
    "kkt_on_scale_negative",
    "kkt_on_scale_zero",
    "kkt_on_scale_positive",
    "kkt_off_scale_negative",
    "kkt_off_scale_zero",
    "kkt_off_scale_positive",
    "lag_zero",
    "lag_fixed",
    "lag_common_estimated",
    "lag_split_estimated",
    "lag_split_strict_only",
    "actuator_stateful",
    "actuator_direct_command",
    "so3_geometric_correction",
    "so3_naive_rotation_vector_derivatives",
    "solver_custom_kkt_lm",
    "solver_standard_gauge_least_squares",
    "jacobian_analytic",
    "jacobian_finite_difference",
    "external_wrench_raw_only",
    "external_wrench_trajectory_fitted",
    "naive_all",
)


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError("case name is empty after sanitization")
    return result


def _clone_arguments(
    base: argparse.Namespace, overrides: Mapping[str, Any]
) -> argparse.Namespace:
    values = deepcopy(vars(base))
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError("unknown case override(s): {}".format(", ".join(unknown)))
    values.update(overrides)
    values["run_id"] = None
    return argparse.Namespace(**values)


def _scale_offsets(arguments: argparse.Namespace) -> tuple[float, float, float]:
    values = arguments.kkt_scale_offsets
    if values is None:
        raise ValueError(
            "KKT scale-offset cases require explicit --kkt-scale-offsets NEG ZERO POS"
        )
    negative, zero, positive = map(float, values)
    if not negative < zero < positive:
        raise ValueError("KKT scale offsets must be strictly increasing")
    return negative, zero, positive


def fixed_case_overrides(
    name: str, arguments: argparse.Namespace
) -> dict[str, Any]:
    if name not in FIXED_CASE_NAMES:
        raise ValueError("unknown fixed case: {}".format(name))
    covariance = {
        "default_full_covariance": "full",
        "cov_identity": "identity",
        "cov_diagonal": "diagonal",
        "cov_block_s_alpha": "block_s_alpha",
        "cov_full_no_R_uncertainty_in_s": "full_no_R_uncertainty_in_s",
        "cov_full_no_position_rotation_cross": "full_no_position_rotation_cross",
        "cov_global_full": "global_full",
    }
    if name in covariance:
        return {"covariance_mode": covariance[name]}
    if name.startswith("kkt_"):
        negative, zero, positive = _scale_offsets(arguments)
        offset = (
            negative
            if name.endswith("negative")
            else positive if name.endswith("positive") else zero
        )
        return {
            "disable_kkt": name.startswith("kkt_off"),
            "solver_type": "custom_kkt_lm",
            "scale_initial_offset": offset,
        }
    direct: dict[str, dict[str, Any]] = {
        "lag_zero": {"lag_mode": "zero"},
        "lag_fixed": {"lag_mode": "fixed"},
        "lag_common_estimated": {"lag_mode": "common_estimated"},
        "lag_split_estimated": {"lag_mode": "split_estimated"},
        "lag_split_strict_only": {"lag_mode": "split_strict_only"},
        "actuator_stateful": {"actuator_propagation": "stateful"},
        "actuator_direct_command": {"actuator_propagation": "direct_command"},
        "so3_geometric_correction": {"naive_so3_derivatives": False},
        "so3_naive_rotation_vector_derivatives": {
            "naive_so3_derivatives": True
        },
        "solver_custom_kkt_lm": {
            "solver_type": "custom_kkt_lm",
            "disable_kkt": False,
        },
        "solver_standard_gauge_least_squares": {
            "solver_type": "standard_least_squares"
        },
        "jacobian_analytic": {"jacobian_mode": "analytic"},
        "jacobian_finite_difference": {"jacobian_mode": "finite_difference"},
        # Replay is post-fit in both cases so reports retain the standardized
        # raw-vs-fitted comparison.  The case name records whether fitted replay
        # is part of the case algorithm or evaluation-only.
        "external_wrench_raw_only": {"disable_replay": True},
        "external_wrench_trajectory_fitted": {"disable_replay": False},
        "naive_all": {
            "covariance_mode": "identity",
            "disable_kkt": True,
            "solver_type": "custom_kkt_lm",
            "lag_mode": "zero",
            "actuator_propagation": "direct_command",
            "naive_so3_derivatives": True,
            "jacobian_mode": "finite_difference",
            "disable_replay": False,
        },
    }
    return direct[name]


def _load_sweep_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("sweep config cannot be read: {}".format(path)) from error
    if not isinstance(value, dict):
        raise ValueError("sweep config root must be an object")
    return value


def sweep_cases(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Expand only user-supplied values; no alternate scientific value is invented."""

    result: list[tuple[str, dict[str, Any]]] = []

    def members(key: str) -> list[Any]:
        value = config.get(key, [])
        if not isinstance(value, list):
            raise ValueError("{} sweep must be a list".format(key))
        return value

    scalar = {
        "sg_windows": "sg_window",
        "sg_degrees": "sg_degree",
    }
    for key, argument_name in scalar.items():
        for index, value in enumerate(members(key)):
            result.append(
                (
                    "sweep__{}__{:03d}".format(key, index),
                    {argument_name: value},
                )
            )
    for index, value in enumerate(members("lag_initials")):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("each lag_initials member must be [rotor, gimbal]")
        result.append(
            (
                "sweep__lag_initials__{:03d}".format(index),
                {"initial_rotor_lag": value[0], "initial_gimbal_lag": value[1]},
            )
        )
    for index, value in enumerate(members("lag_bounds")):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("each lag_bounds member must be [lower, upper]")
        result.append(
            (
                "sweep__lag_bounds__{:03d}".format(index),
                {"lag_bounds": value},
            )
        )
    for index, value in enumerate(members("smooth_schedules")):
        if not isinstance(value, list) or not value:
            raise ValueError("each smooth schedule must be a non-empty list")
        result.append(
            (
                "sweep__smooth_schedule__{:03d}".format(index),
                {"smooth_width_schedule": value},
            )
        )
    mapping_groups = {
        "lm_settings": {
            "initial_damping": "lm_initial_damping",
            "initial_trust_radius": "lm_initial_trust_radius",
            "maximum_trust_radius": "lm_maximum_trust_radius",
            "minimum_trust_radius": "lm_minimum_trust_radius",
            "acceptance_ratio": "lm_acceptance_ratio",
            "gtol": "gtol",
            "ftol": "ftol",
            "xtol": "xtol",
        },
        "termination_limits": {
            "smooth_max_nfev": "smooth_max_nfev",
            "strict_max_nfev": "strict_max_nfev",
            "strict_alternations": "strict_alternations",
        },
        "actuator_settings": {
            "minimum_thrust": "minimum_thrust",
            "maximum_thrust": "maximum_thrust",
            "maximum_gimbal_angle": "maximum_gimbal_angle",
            "maximum_gimbal_rate": "maximum_gimbal_rate",
        },
        "actuator_time_constants": {
            "thrust": "thrust_time_constant",
            "gimbal": "gimbal_time_constant",
        },
        "segments": {"start": "bag_start", "end": "bag_end"},
    }
    for group, key_mapping in mapping_groups.items():
        for index, value in enumerate(members(group)):
            if not isinstance(value, dict):
                raise ValueError("{} members must be objects".format(group))
            unknown = sorted(set(value) - set(key_mapping))
            if unknown:
                raise ValueError(
                    "unknown {} keys: {}".format(group, ", ".join(unknown))
                )
            result.append(
                (
                    "sweep__{}__{:03d}".format(group, index),
                    {key_mapping[key]: item for key, item in value.items()},
                )
            )
    for index, value in enumerate(members("initial_coordinates")):
        if not isinstance(value, list) or len(value) != 14:
            raise ValueError("initial chart coordinates must contain 14 values")
        result.append(
            (
                "sweep__initial_coordinate__{:03d}".format(index),
                {"initial_coordinate": value},
            )
        )
    return result


def agent_candidate_cases(
    config: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    cases = config.get("agent_candidates", [])
    if not isinstance(cases, list):
        raise ValueError("agent_candidates must be a list")
    result = []
    for value in cases:
        if not isinstance(value, dict):
            raise ValueError("agent candidate must be an object")
        required = (
            "case_name",
            "changed_from_default",
            "expected_property",
            "reason_for_expectation",
            "overrides",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(
                "agent candidate is missing: {}".format(", ".join(missing))
            )
        name = "agent_candidate__{}".format(_safe_name(value["case_name"]))
        overrides = value["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError("agent candidate overrides must be an object")
        proposal = {
            "case_name": name,
            "changed_from_default": value["changed_from_default"],
            "expected_property": value["expected_property"],
            "reason_for_expectation": value["reason_for_expectation"],
            "cheat_guard_acknowledged": True,
        }
        result.append((name, overrides, proposal))
    return result


def run_case_sequence(
    *,
    run_directory: Path,
    case_names: Sequence[str],
    source_revision: str,
    executor: Callable[[str, Path], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run every case under a top-level exception boundary."""

    summaries: list[dict[str, Any]] = []
    for case_name in case_names:
        safe = _safe_name(case_name)
        directory = run_directory / "cases" / safe
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            payload = dict(executor(case_name, directory))
        except Exception as error:  # This boundary must never stop later cases.
            payload = {
                "status": "failed",
                "case_name": case_name,
                "source_commit": source_revision,
                "base_plan_commit": BASE_PLAN_COMMIT,
                "failure_stage": "ablation_case_top_level",
                "exception_type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "traceback": traceback.format_exc(),
            }
            write_json(directory / "status.json", payload)
            write_json(directory / "result.json", payload)
            write_json(directory / "arguments.json", {})
            write_json(
                directory / "timing.json",
                {"elapsed_seconds": payload["elapsed_seconds"]},
            )
            write_failure_report_pdf(
                directory / "report.pdf",
                case_name=case_name,
                failure_stage="ablation_case_top_level",
                exception_type=type(error).__name__,
                message=str(error),
            )
        summaries.append(
            {
                "case_name": case_name,
                "status": payload.get("status", "failed"),
                "failure_reason": payload.get("message"),
                "elapsed_seconds": payload.get(
                    "elapsed_seconds", time.perf_counter() - started
                ),
                "point_estimate": payload.get("parameters"),
                "common_evaluation": payload.get("common_evaluation"),
                "ridge": payload.get("ridge"),
                "uncertainty": payload.get("uncertainty"),
            }
        )
    return summaries


def _case_summary(case_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "status": payload.get("status", "failed"),
        "failure_reason": payload.get("message"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "point_estimate": payload.get("parameters"),
        "common_evaluation": payload.get("common_evaluation"),
        "ridge": payload.get("ridge"),
        "uncertainty": payload.get("uncertainty"),
    }


def _record_case_failure(
    *,
    case_name: str,
    directory: Path,
    source_revision: str,
    error: BaseException,
    started: float,
) -> dict[str, Any]:
    payload = {
        "status": "failed",
        "case_name": case_name,
        "source_commit": source_revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "failure_stage": "ablation_case_top_level",
        "exception_type": type(error).__name__,
        "message": str(error),
        "elapsed_seconds": time.perf_counter() - started,
        "traceback": traceback.format_exc(),
    }
    write_json(directory / "status.json", payload)
    write_json(directory / "result.json", payload)
    write_json(directory / "arguments.json", {})
    write_json(
        directory / "timing.json",
        {"elapsed_seconds": payload["elapsed_seconds"]},
    )
    write_failure_report_pdf(
        directory / "report.pdf",
        case_name=case_name,
        failure_stage="ablation_case_top_level",
        exception_type=type(error).__name__,
        message=str(error),
    )
    return payload


def _run_estimator_case_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one case in a spawned process and persist its terminal state."""

    case_name = str(task["case_name"])
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        override_error = task.get("override_error")
        if override_error:
            raise ValueError(str(override_error))
        proposal = task.get("proposal")
        if proposal is not None:
            write_json(directory / "proposal.json", proposal)
        base = argparse.Namespace(**dict(task["base_arguments"]))
        case_arguments = _clone_arguments(base, dict(task["overrides"]))
        _directory, payload = run_estimator(
            case_arguments,
            case_name=case_name,
            output_directory=directory,
        )
        return dict(payload)
    except Exception as error:  # Each worker must leave a terminal record.
        return _record_case_failure(
            case_name=case_name,
            directory=directory,
            source_revision=str(task["source_revision"]),
            error=error,
            started=started,
        )


def _load_terminal_case(directory: Path) -> Optional[dict[str, Any]]:
    required = (
        directory / "status.json",
        directory / "result.json",
        directory / "arguments.json",
        directory / "timing.json",
        directory / "report.pdf",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        status = json.loads(required[0].read_text(encoding="utf-8"))
        payload = json.loads(required[1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(status, dict) or not isinstance(payload, dict):
        return None
    terminal_status = status.get("status")
    if terminal_status not in ("completed", "failed"):
        return None
    if payload.get("status") != terminal_status:
        return None
    if terminal_status == "completed" and not (directory / "arrays.npz").is_file():
        return None
    return payload


def run_estimator_case_tasks(
    *,
    run_directory: Path,
    case_names: Sequence[str],
    source_revision: str,
    base_arguments: argparse.Namespace,
    override_lookup: Mapping[str, Mapping[str, Any]],
    override_failures: Mapping[str, Exception],
    proposal_lookup: Mapping[str, Mapping[str, Any]],
    case_workers: int,
    resume_existing: bool,
) -> list[dict[str, Any]]:
    """Run independent estimator cases, optionally resuming terminal outputs."""

    payloads: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    base_values = deepcopy(vars(base_arguments))
    for case_name in case_names:
        directory = run_directory / "cases" / _safe_name(case_name)
        if resume_existing:
            terminal = _load_terminal_case(directory)
            if terminal is not None:
                payloads[case_name] = terminal
                continue
        tasks.append(
            {
                "case_name": case_name,
                "directory": directory,
                "source_revision": source_revision,
                "base_arguments": base_values,
                "overrides": dict(override_lookup.get(case_name, {})),
                "override_error": (
                    str(override_failures[case_name])
                    if case_name in override_failures
                    else None
                ),
                "proposal": proposal_lookup.get(case_name),
            }
        )

    if case_workers == 1:
        for task in tasks:
            payloads[str(task["case_name"])] = _run_estimator_case_task(task)
    elif tasks:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=case_workers, mp_context=context
        ) as pool:
            futures = {
                pool.submit(_run_estimator_case_task, task): task for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                case_name = str(task["case_name"])
                try:
                    payloads[case_name] = future.result()
                except BaseException as error:
                    directory = Path(task["directory"])
                    directory.mkdir(parents=True, exist_ok=True)
                    payloads[case_name] = _record_case_failure(
                        case_name=case_name,
                        directory=directory,
                        source_revision=source_revision,
                        error=error,
                        started=time.perf_counter(),
                    )
    return [_case_summary(name, payloads[name]) for name in case_names]


def build_argument_parser() -> argparse.ArgumentParser:
    base = build_estimator_argument_parser()
    base.description = __doc__
    base.add_argument(
        "--cases",
        nargs="+",
        choices=("all",) + FIXED_CASE_NAMES,
        default=("all",),
    )
    base.add_argument("--kkt-scale-offsets", type=float, nargs=3, default=None)
    base.add_argument("--sweep-config", type=Path, default=None)
    base.add_argument("--ablation-run-id", default=None)
    base.add_argument("--case-workers", type=int, default=1)
    base.add_argument("--resume-existing", action="store_true")
    return base


def run_ablation(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    arguments = resolve_bag_arguments(arguments)
    if arguments.case_workers < 1:
        raise ValueError("--case-workers must be at least 1")
    if arguments.resume_existing and not arguments.ablation_run_id:
        raise ValueError("--resume-existing requires --ablation-run-id")
    revision = source_commit(_HERE.parent)
    selected = (
        list(FIXED_CASE_NAMES)
        if "all" in arguments.cases
        else list(dict.fromkeys(arguments.cases))
    )
    config = _load_sweep_config(arguments.sweep_config)
    sweeps = sweep_cases(config)
    candidates = agent_candidate_cases(config)
    if candidates and not selected:
        raise ValueError("agent candidates require fixed cases to run first")
    all_names = selected + [name for name, _ in sweeps] + [
        name for name, _, _ in candidates
    ]
    manifest = {
        "source_commit": revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "bag_path": arguments.bag,
        "bag_start": arguments.bag_start,
        "bag_end": arguments.bag_end,
        "case_names": all_names,
        "fixed_case_names": selected,
        "sweep_config": config,
        "case_workers": arguments.case_workers,
    }
    if arguments.resume_existing:
        run_directory = (
            Path(arguments.output_root)
            / revision
            / "ablation"
            / str(arguments.ablation_run_id)
        )
        if not run_directory.is_dir():
            raise ValueError(
                "ablation run directory does not exist: {}".format(run_directory)
            )
        manifest_path = run_directory / "manifest.json"
        try:
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "existing ablation manifest cannot be read: {}".format(
                    manifest_path
                )
            ) from error
        expected_manifest = json_sanitize(manifest)
        for key in (
            "source_commit",
            "base_plan_commit",
            "bag_path",
            "bag_start",
            "bag_end",
            "case_names",
            "fixed_case_names",
            "sweep_config",
        ):
            if existing_manifest.get(key) != expected_manifest.get(key):
                raise ValueError(
                    "existing ablation manifest mismatch for {}".format(key)
                )
    else:
        run_directory = output_run_directory(
            arguments.output_root,
            "ablation",
            arguments.ablation_run_id,
            commit=revision,
        )
    write_json(run_directory / "manifest.json", manifest)
    override_lookup: dict[str, dict[str, Any]] = {}
    override_failures: dict[str, Exception] = {}
    for name in selected:
        try:
            override_lookup[name] = fixed_case_overrides(name, arguments)
        except Exception as error:  # Recorded as a case failure, not root failure.
            override_failures[name] = error
    override_lookup.update(dict(sweeps))
    proposal_lookup = {name: proposal for name, _, proposal in candidates}
    override_lookup.update({name: override for name, override, _ in candidates})

    summaries = run_estimator_case_tasks(
        run_directory=run_directory,
        case_names=all_names,
        source_revision=revision,
        base_arguments=arguments,
        override_lookup=override_lookup,
        override_failures=override_failures,
        proposal_lookup=proposal_lookup,
        case_workers=arguments.case_workers,
        resume_existing=arguments.resume_existing,
    )
    summary = {
        "source_commit": revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "status": "completed",
        "case_count": len(summaries),
        "completed_count": sum(item["status"] == "completed" for item in summaries),
        "failed_count": sum(item["status"] != "completed" for item in summaries),
        "cases": summaries,
    }
    write_json(run_directory / "summary.json", summary)
    return run_directory, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    directory, _summary = run_ablation(arguments)
    print(directory)
    # Per-case failures are experiment results, not a runner failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
