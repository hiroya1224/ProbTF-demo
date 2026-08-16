#!/usr/bin/env python3
"""Run a fixed manifest of optional parameter-prior cases on one bag."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Optional, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-prior-ablation-matplotlib")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from single_bag_parameter_prior import (  # noqa: E402
    QUOTIENT_COMPONENT_LABELS,
    QUOTIENT_COMPONENT_UNITS,
)
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


PRIOR_ABLATION_SCHEMA = "grape-param-estim/prior-ablation/v1"
PRIOR_FREE_CASE = "prior_free"
VALID_POINT_STATUSES = ("completed", "point_estimate_completed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("case name is empty after sanitization")
    return result


def load_ablation_manifest(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("prior-ablation manifest cannot be read: {}".format(source)) from error
    if not isinstance(raw, dict) or raw.get("schema") != PRIOR_ABLATION_SCHEMA:
        raise ValueError("prior-ablation schema must be {}".format(PRIOR_ABLATION_SCHEMA))
    if raw.get("include_prior_free_baseline") is not True:
        raise ValueError("primary prior ablation requires a prior-free baseline")
    name = raw.get("name")
    cases = raw.get("cases")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("prior-ablation name must be a non-empty string")
    if not isinstance(cases, list) or not cases:
        raise ValueError("prior-ablation cases must be a non-empty list")
    resolved = []
    names = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict) or set(item) != {"case_name", "prior_json"}:
            raise ValueError("manifest case {} requires only case_name and prior_json".format(index))
        case_name = item["case_name"]
        prior_json = item["prior_json"]
        if not isinstance(case_name, str) or not case_name.strip():
            raise ValueError("manifest case name must be a non-empty string")
        if case_name == PRIOR_FREE_CASE:
            raise ValueError("prior_free is reserved for the baseline")
        if not isinstance(prior_json, str) or not prior_json.strip():
            raise ValueError("manifest prior_json must be a non-empty string")
        prior_path = (source.parent / prior_json).resolve()
        if not prior_path.is_file():
            raise ValueError("manifest prior JSON does not exist: {}".format(prior_path))
        names.append(case_name)
        resolved.append(
            {
                "case_name": case_name,
                "prior_json": prior_path,
                "prior_json_manifest_value": prior_json,
                "prior_json_sha256": _sha256(prior_path),
            }
        )
    if len(set(names)) != len(names):
        raise ValueError("prior-ablation case names must be unique")
    return {
        "schema": PRIOR_ABLATION_SCHEMA,
        "name": name.strip(),
        "source_path": source,
        "source_sha256": _sha256(source),
        "include_prior_free_baseline": True,
        "cases": resolved,
    }


def quotient_vector_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    scale_free = payload["parameters"]["scale_free"]
    inertia = np.asarray(scale_free["inertia_over_mass_m2"], dtype=float)
    cog = np.asarray(scale_free["cog_position_body_m"], dtype=float)
    force = np.asarray(scale_free["force_effectiveness_over_mass"], dtype=float)
    value = np.concatenate(
        (
            inertia[[0, 1, 2], [0, 1, 2]],
            inertia[[0, 0, 1], [1, 2, 2]],
            cog,
            force,
        )
    )
    if value.shape != (13,) or np.any(~np.isfinite(value)):
        raise ValueError("case result has an invalid scale-free parameter vector")
    return value


def prior_vectors_from_payload(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    error = np.full(13, np.nan)
    standardized = np.full(13, np.nan)
    target = np.full(13, np.nan)
    achieved = np.full(13, np.nan)
    prior = payload.get("prior", {})
    for factor in prior.get("factor_evaluations", []):
        indices = np.asarray(factor["quotient_indices"], dtype=int)
        error[indices] = np.asarray(factor["physical_error"], dtype=float)
        standardized[indices] = np.asarray(
            factor["standardized_residual"], dtype=float
        )
        target[indices] = np.asarray(factor["physical_target"], dtype=float)
        achieved[indices] = np.asarray(factor["physical_value"], dtype=float)
    return error, standardized, target, achieved


def _case_arguments(
    base: argparse.Namespace, prior_json: Optional[Path]
) -> argparse.Namespace:
    values = deepcopy(vars(base))
    values["prior_json"] = prior_json
    values["run_id"] = None
    return argparse.Namespace(**values)


def _record_worker_failure(task: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    failure = {
        "status": "failed",
        "overall_case_status": "failed",
        "optimization_status": "failed",
        "postfit_uncertainty_status": "not_run",
        "case_name": task["case_name"],
        "source_commit": task["source_commit"],
        "base_plan_commit": BASE_PLAN_COMMIT,
        "failure_stage": "prior_ablation_case_top_level",
        "exception_type": type(error).__name__,
        "message": str(error),
        "elapsed_seconds": 0.0,
        "traceback": traceback.format_exc(),
    }
    write_json(directory / "status.json", failure)
    write_json(directory / "result.json", failure)
    write_json(directory / "arguments.json", {})
    write_json(directory / "timing.json", {"elapsed_seconds": 0.0})
    write_failure_report_pdf(
        directory / "report.pdf",
        case_name=str(task["case_name"]),
        failure_stage="prior_ablation_case_top_level",
        exception_type=type(error).__name__,
        message=str(error),
    )
    return failure


def _run_case_task(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        base = argparse.Namespace(**dict(task["base_arguments"]))
        arguments = _case_arguments(
            base,
            None if task["prior_json"] is None else Path(task["prior_json"]),
        )
        _directory, payload = run_estimator(
            arguments,
            case_name=str(task["case_name"]),
            output_directory=Path(task["directory"]),
        )
        return dict(payload)
    except BaseException as error:
        failure = _record_worker_failure(task, error)
        failure["elapsed_seconds"] = time.perf_counter() - started
        write_json(Path(task["directory"]) / "status.json", failure)
        write_json(Path(task["directory"]) / "result.json", failure)
        return failure


def _load_terminal_case(directory: Path) -> Optional[dict[str, Any]]:
    required = tuple(
        directory / name
        for name in ("status.json", "result.json", "arguments.json", "timing.json", "report.pdf")
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        status = json.loads(required[0].read_text(encoding="utf-8"))
        result = json.loads(required[1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    terminal = status.get("status")
    if terminal not in VALID_POINT_STATUSES + ("failed",):
        return None
    if result.get("status") != terminal:
        return None
    if terminal in VALID_POINT_STATUSES and not (directory / "arrays.npz").is_file():
        return None
    return result


def _run_tasks(
    tasks: Sequence[Mapping[str, Any]], *, workers: int, resume_existing: bool
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        if resume_existing:
            payload = _load_terminal_case(Path(task["directory"]))
            if payload is not None:
                payloads[str(task["case_name"])] = payload
                continue
        pending.append(task)
    if workers == 1:
        for task in pending:
            payloads[str(task["case_name"])] = _run_case_task(task)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(_run_case_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    payloads[str(task["case_name"])] = future.result()
                except BaseException as error:
                    payloads[str(task["case_name"])] = _record_worker_failure(task, error)
    return payloads


def build_summary(
    *,
    source_revision: str,
    manifest: Mapping[str, Any],
    bag_id: str,
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cases = [
        {
            "case_name": PRIOR_FREE_CASE,
            "prior_json": None,
            "prior_json_sha256": None,
        }
    ] + list(manifest["cases"])
    names = [str(item["case_name"]) for item in cases]
    count = len(names)
    x = np.full((count, 13), np.nan)
    data_cost = np.full(count, np.nan)
    prior_cost = np.full(count, np.nan)
    total_cost = np.full(count, np.nan)
    lag = np.full(count, np.nan)
    target_error = np.full((count, 13), np.nan)
    standardized = np.full((count, 13), np.nan)
    target = np.full((count, 13), np.nan)
    achieved = np.full((count, 13), np.nan)
    optimization_status = []
    postfit_status = []
    overall_status = []
    case_records = []
    for index, item in enumerate(cases):
        name = names[index]
        payload = payloads[name]
        overall = str(payload.get("overall_case_status", payload.get("status", "failed")))
        optimization = str(payload.get("optimization_status", "failed"))
        postfit = str(payload.get("postfit_uncertainty_status", "not_run"))
        overall_status.append(overall)
        optimization_status.append(optimization)
        postfit_status.append(postfit)
        if overall in VALID_POINT_STATUSES and optimization == "completed":
            x[index] = quotient_vector_from_payload(payload)
            objective = payload["optimization_objective"]
            data_cost[index] = float(objective["data_objective_sum"])
            prior_cost[index] = float(objective["prior_objective_sum"])
            total_cost[index] = float(objective["total_objective_sum"])
            lag[index] = float(payload["parameters"]["rotor_lag_seconds"])
            if name != PRIOR_FREE_CASE:
                target_error[index], standardized[index], target[index], achieved[index] = prior_vectors_from_payload(payload)
        case_records.append(
            {
                "case_name": name,
                "prior_config_path": item.get("prior_json"),
                "prior_config_sha256": item.get("prior_json_sha256"),
                "overall_case_status": overall,
                "optimization_status": optimization,
                "postfit_uncertainty_status": postfit,
                "data_objective": data_cost[index],
                "prior_objective": prior_cost[index],
                "total_objective": total_cost[index],
                "rotor_lag_seconds": lag[index],
                "x": x[index],
                "prior_target": target[index],
                "prior_achieved_value": achieved[index],
                "prior_target_error": target_error[index],
                "prior_standardized_residual": standardized[index],
                "failure_stage": payload.get("failure_stage"),
                "failure_reason": payload.get("failure_reason", payload.get("message")),
            }
        )
    baseline = x[0]
    delta_x = x - baseline
    delta_data = data_cost - data_cost[0]
    arrays = {
        "case_names": np.asarray(names, dtype="U"),
        "prior_config_paths": np.asarray(
            ["" if item.get("prior_json") is None else str(item["prior_json"]) for item in cases],
            dtype="U",
        ),
        "prior_config_sha256": np.asarray(
            ["" if item.get("prior_json_sha256") is None else str(item["prior_json_sha256"]) for item in cases],
            dtype="U",
        ),
        "quotient_component_labels": np.asarray(QUOTIENT_COMPONENT_LABELS, dtype="U"),
        "quotient_component_units": np.asarray(QUOTIENT_COMPONENT_UNITS, dtype="U"),
        "x_prior_free": baseline,
        "x_per_case": x,
        "delta_x_per_case": delta_x,
        "data_objective_prior_free": np.asarray(data_cost[0]),
        "data_objective_per_case": data_cost,
        "delta_data_objective_per_case": delta_data,
        "prior_objective_per_case": prior_cost,
        "total_objective_per_case": total_cost,
        "rotor_lag_prior_free": np.asarray(lag[0]),
        "rotor_lag_per_case": lag,
        "prior_target_per_case": target,
        "prior_achieved_value_per_case": achieved,
        "prior_target_error_per_case": target_error,
        "prior_standardized_residual_per_case": standardized,
        "optimization_status_per_case": np.asarray(optimization_status, dtype="U"),
        "postfit_uncertainty_status_per_case": np.asarray(postfit_status, dtype="U"),
        "overall_case_status_per_case": np.asarray(overall_status, dtype="U"),
    }
    summary = {
        "schema": "grape-param-estim/prior-ablation-summary/v1",
        "status": "completed",
        "source_commit": source_revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "bag_id": bag_id,
        "manifest": {
            "schema": manifest["schema"],
            "name": manifest["name"],
            "source_path": manifest["source_path"],
            "source_sha256": manifest["source_sha256"],
        },
        "case_count": count,
        "point_estimate_count": sum(item in VALID_POINT_STATUSES for item in overall_status),
        "failed_count": sum(item == "failed" for item in overall_status),
        "postfit_uncertainty_failed_count": sum(item == "failed" for item in postfit_status),
        "quotient_component_labels": QUOTIENT_COMPONENT_LABELS,
        "quotient_component_units": QUOTIENT_COMPONENT_UNITS,
        "case_names": arrays["case_names"],
        "prior_config_paths": arrays["prior_config_paths"],
        "prior_config_sha256": arrays["prior_config_sha256"],
        "x_prior_free": baseline,
        "x_per_case": x,
        "delta_x_per_case": delta_x,
        "data_objective_prior_free": data_cost[0],
        "data_objective_per_case": data_cost,
        "delta_data_objective_per_case": delta_data,
        "prior_objective_per_case": prior_cost,
        "total_objective_per_case": total_cost,
        "rotor_lag_prior_free": lag[0],
        "rotor_lag_per_case": lag,
        "prior_target_per_case": target,
        "prior_achieved_value_per_case": achieved,
        "prior_target_error_per_case": target_error,
        "prior_standardized_residual_per_case": standardized,
        "optimization_status_per_case": optimization_status,
        "postfit_uncertainty_status_per_case": postfit_status,
        "overall_case_status_per_case": overall_status,
        "cases": case_records,
    }
    return summary, arrays


def _heatmap(
    pdf: PdfPages,
    *,
    values: np.ndarray,
    case_names: Sequence[str],
    labels: Sequence[str],
    title: str,
    colorbar_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 8.5))
    finite = np.asarray(values)[np.isfinite(values)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, np.finfo(float).tiny)
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_yticks(np.arange(len(case_names)))
    axis.set_yticklabels(case_names, fontsize=7)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def write_ablation_pdf(path: Path, summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    names = [str(value) for value in arrays["case_names"]]
    records = summary["cases"]
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = [
            "case | target -> achieved | standardized residual | delta data loss | prior loss | lag [s] | opt/postfit",
            "-" * 130,
        ]
        for record in records:
            indices = np.flatnonzero(np.isfinite(np.asarray(record["prior_target"], dtype=float)))
            if indices.size:
                target_text = ", ".join(
                    "{}:{:.4g}->{:.4g} ({:+.3g} std)".format(
                        QUOTIENT_COMPONENT_LABELS[index],
                        record["prior_target"][index],
                        record["prior_achieved_value"][index],
                        record["prior_standardized_residual"][index],
                    )
                    for index in indices
                )
            else:
                target_text = "prior-free"
            lines.append(
                "{} | {} | dL={:+.4g} Lp={:.4g} lag={:.5g} | {}/{}".format(
                    record["case_name"],
                    target_text,
                    record["data_objective"] - summary["data_objective_prior_free"],
                    record["prior_objective"],
                    record["rotor_lag_seconds"],
                    record["optimization_status"],
                    record["postfit_uncertainty_status"],
                )
            )
        axis.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace", fontsize=5.8)
        figure.suptitle("{}: prior-ablation case overview".format(summary["bag_id"]))
        pdf.savefig(figure)
        plt.close(figure)

        delta = np.asarray(arrays["delta_x_per_case"])
        _heatmap(pdf, values=1e3 * delta[:, 6:9], case_names=names, labels=("CoG_x", "CoG_y", "CoG_z"), title="CoG compensation response", colorbar_label="delta CoG [mm]")
        _heatmap(pdf, values=delta[:, :6], case_names=names, labels=QUOTIENT_COMPONENT_LABELS[:6], title="Inertia-over-mass compensation response", colorbar_label="delta J/m [m^2]")
        _heatmap(pdf, values=delta[:, 9:13], case_names=names, labels=QUOTIENT_COMPONENT_LABELS[9:13], title="Force-over-mass compensation response", colorbar_label="delta f/m [kg^-1]")

        figure, axis = plt.subplots(figsize=(11.0, 8.5))
        value = np.asarray(arrays["delta_data_objective_per_case"])
        axis.barh(np.arange(len(names)), value)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("delta data objective")
        axis.set_title("Data-fit price of pseudo-conditioning")
        axis.grid(True, axis="x", alpha=0.3)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        _heatmap(pdf, values=np.asarray(arrays["prior_standardized_residual_per_case"]), case_names=names, labels=QUOTIENT_COMPONENT_LABELS, title="Final standardized prior residual", colorbar_label="target error / prior std")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = build_estimator_argument_parser()
    parser.description = __doc__
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ablation-run-id", default=None)
    parser.add_argument("--case-workers", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def run_prior_ablation(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    arguments = resolve_bag_arguments(arguments)
    if arguments.prior_json is not None:
        raise ValueError("--prior-json belongs to manifest cases and cannot be supplied at runner level")
    if arguments.case_workers < 1:
        raise ValueError("--case-workers must be at least 1")
    if arguments.resume_existing and not arguments.ablation_run_id:
        raise ValueError("--resume-existing requires --ablation-run-id")
    manifest = load_ablation_manifest(arguments.manifest)
    revision = source_commit(_HERE.parent)
    if arguments.resume_existing:
        run_directory = Path(arguments.output_root) / revision / "prior_ablation" / str(arguments.ablation_run_id)
        if not run_directory.is_dir():
            raise ValueError("prior-ablation run directory does not exist: {}".format(run_directory))
    else:
        run_directory = output_run_directory(
            arguments.output_root,
            "prior_ablation",
            arguments.ablation_run_id,
            commit=revision,
        )
    cases = [{"case_name": PRIOR_FREE_CASE, "prior_json": None, "prior_json_sha256": None}] + list(manifest["cases"])
    experiment = {
        "schema": PRIOR_ABLATION_SCHEMA,
        "source_commit": revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "bag_id": arguments.bag_id,
        "bag_path": arguments.bag,
        "bag_start": arguments.bag_start,
        "bag_end": arguments.bag_end,
        "manifest_path": manifest["source_path"],
        "manifest_sha256": manifest["source_sha256"],
        "case_names": [item["case_name"] for item in cases],
        "case_workers": arguments.case_workers,
        "same_default_initialization_for_every_case": True,
    }
    manifest_output = run_directory / "manifest.json"
    if arguments.resume_existing:
        existing = json.loads(manifest_output.read_text(encoding="utf-8"))
        expected = json_sanitize(experiment)
        for key in ("schema", "source_commit", "bag_id", "bag_path", "bag_start", "bag_end", "manifest_sha256", "case_names"):
            if existing.get(key) != expected.get(key):
                raise ValueError("existing prior-ablation manifest mismatch for {}".format(key))
    write_json(manifest_output, experiment)
    base_values = deepcopy(vars(arguments))
    tasks = [
        {
            "case_name": item["case_name"],
            "prior_json": item["prior_json"],
            "directory": run_directory / "cases" / _safe_name(str(item["case_name"])),
            "source_commit": revision,
            "base_arguments": base_values,
        }
        for item in cases
    ]
    started = time.perf_counter()
    payloads = _run_tasks(tasks, workers=arguments.case_workers, resume_existing=arguments.resume_existing)
    summary, arrays = build_summary(
        source_revision=revision,
        manifest=manifest,
        bag_id=str(arguments.bag_id),
        payloads=payloads,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(run_directory / "prior_ablation.json", summary)
    np.savez_compressed(run_directory / "prior_ablation.npz", **arrays)
    write_ablation_pdf(run_directory / "prior_ablation.pdf", summary, arrays)
    write_json(
        run_directory / "status.json",
        {
            "status": "completed",
            "source_commit": revision,
            "bag_id": arguments.bag_id,
            "case_count": summary["case_count"],
            "point_estimate_count": summary["point_estimate_count"],
            "failed_count": summary["failed_count"],
            "postfit_uncertainty_failed_count": summary["postfit_uncertainty_failed_count"],
        },
    )
    write_json(run_directory / "arguments.json", vars(arguments))
    write_json(run_directory / "timing.json", {"elapsed_seconds": summary["elapsed_seconds"]})
    return run_directory, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    directory, _summary = run_prior_ablation(arguments)
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
