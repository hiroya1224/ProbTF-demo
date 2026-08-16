#!/usr/bin/env python3
"""Compare original and CoG-reseed-refined distributions across three bags."""

from __future__ import annotations

import argparse
import json
import os
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-cog-reseed-matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from single_bag_cog_prior_reseed_refinement import (  # noqa: E402
    BASELINE_CASE_SOURCE_COMMIT,
    BASELINE_REPOSITORY_COMMIT,
    BASELINE_RUN_BY_BAG_ID,
    CONDITIONING_SOURCE_COVARIANCE_NAME,
    REFINEMENT_INTERPRETATION,
)
from single_bag_cross_bag_consensus import (  # noqa: E402
    _cross_evaluate,
    _load_case,
    _pairwise_distance,
    _problem_from_case,
)
from single_bag_savgol_core import load_vehicle_model  # noqa: E402
from single_bag_savgol_reports import (  # noqa: E402
    output_run_directory,
    source_commit,
    write_json,
)


_BAG_ORDER = (
    "single_rosbag_1",
    "single_rosbag_2",
    "single_rosbag_succeeded",
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(label))
    return value


def _load_refinement_directory(directory: Path) -> dict[str, Any]:
    top = Path(directory).expanduser().resolve()
    status = _read_object(top / "status.json", "refinement status")
    comparison = _read_object(top / "comparison.json", "refinement comparison")
    conditioning = _read_object(top / "conditioning.json", "conditioning")
    if status.get("status") != "completed" or comparison.get("status") != "completed":
        raise ValueError("refinement input is not completed: {}".format(top))
    bag_id = str(status["bag_id"])
    if bag_id not in BASELINE_RUN_BY_BAG_ID:
        raise ValueError("unexpected refinement bag id: {}".format(bag_id))
    baseline_directory = Path(status["baseline_case_directory"]).resolve()
    expected_baseline = (
        _HERE
        / "outputs"
        / BASELINE_CASE_SOURCE_COMMIT
        / "default"
        / BASELINE_RUN_BY_BAG_ID[bag_id]
    ).resolve()
    if baseline_directory != expected_baseline:
        raise ValueError("refinement references the wrong immutable baseline")
    refined_directory = Path(status["refined_directory"]).resolve()
    if refined_directory != (top / "refined").resolve():
        raise ValueError("refined output is outside its top-level run")
    if comparison.get("refinement_prior_role") != "none":
        raise ValueError("refinement unexpectedly records an active prior")
    if not np.array_equal(
        np.asarray(conditioning.get("cog_prior_std_m"), dtype=float),
        np.full(3, 0.001),
    ):
        raise ValueError("refinement did not use the exact production 1 mm prior")
    if conditioning.get("conditioning_prior_role") != "initialization_only":
        raise ValueError("conditioning prior role is not initialization-only")
    if conditioning.get("conditioned_seed_is_finite") is not True:
        raise ValueError("conditioned seed diagnostic is not finite")
    return {
        "directory": top,
        "bag_id": bag_id,
        "status": status,
        "comparison": comparison,
        "conditioning": conditioning,
        "original_case": _load_case(baseline_directory),
        "refined_case": _load_case(refined_directory),
    }


def _stage_arrays(
    refinements: Sequence[Mapping[str, Any]], stage: str
) -> dict[str, np.ndarray]:
    values = [item["comparison"][stage] for item in refinements]
    return {
        "objective": np.asarray(
            [item["strict_identity_objective_sum"] for item in values],
            dtype=float,
        ),
        "rotor_lag": np.asarray(
            [
                item.get(
                    "rotor_lag_seconds",
                    item.get("evaluation_rotor_lag_seconds"),
                )
                for item in values
            ],
            dtype=float,
        ),
        "cog": np.stack(
            [np.asarray(item["cog_position_body_m"]) for item in values]
        ),
        "force_effectiveness": np.stack(
            [np.asarray(item["force_effectiveness"]) for item in values]
        ),
        "inertia_over_mass": np.stack(
            [np.asarray(item["inertia_over_mass_m2"]) for item in values]
        ),
        "force_over_mass": np.stack(
            [np.asarray(item["force_effectiveness_over_mass"]) for item in values]
        ),
    }


def _physical_distance(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "inertia_over_mass_frobenius_m2": float(
            np.linalg.norm(
                np.asarray(first["inertia_over_mass_m2"])
                - np.asarray(second["inertia_over_mass_m2"])
            )
        ),
        "force_effectiveness_over_mass_l2": float(
            np.linalg.norm(
                np.asarray(first["force_effectiveness_over_mass"])
                - np.asarray(second["force_effectiveness_over_mass"])
            )
        ),
        "cog_l2_m": float(
            np.linalg.norm(
                np.asarray(first["cog_position_body_m"])
                - np.asarray(second["cog_position_body_m"])
            )
        ),
    }


def _comparison_rows(
    refinements: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for label, item in zip(("failure1", "failure2", "success"), refinements):
        comparison = item["comparison"]
        original = comparison["original"]
        conditioned = comparison["conditioned"]
        refined = comparison["refined"]
        row = {
                "case": label,
                "bag_id": item["bag_id"],
                "L_original": original["strict_identity_objective_sum"],
                "L_conditioned_seed_at_original_lag": conditioned[
                    "strict_identity_objective_sum"
                ],
                "L_refined": refined["strict_identity_objective_sum"],
                "Delta_L_refined_minus_original": comparison[
                    "delta_L_refined_minus_original"
                ],
                "relative_Delta_L_refined_minus_original": comparison[
                    "relative_delta_L_refined_minus_original"
                ],
                "rotor_lag_original": original["rotor_lag_seconds"],
                "rotor_lag_refined": refined["rotor_lag_seconds"],
                "CoG_original_m": original["cog_position_body_m"],
                "CoG_conditioned_m": conditioned["cog_position_body_m"],
                "CoG_refined_m": refined["cog_position_body_m"],
                "force_effectiveness_original": original[
                    "force_effectiveness"
                ],
                "force_effectiveness_conditioned": conditioned[
                    "force_effectiveness"
                ],
                "force_effectiveness_refined": refined[
                    "force_effectiveness"
                ],
                "J_over_m_original": original["inertia_over_mass_m2"],
                "J_over_m_conditioned": conditioned[
                    "inertia_over_mass_m2"
                ],
                "J_over_m_refined": refined["inertia_over_mass_m2"],
                "force_over_mass_original": original[
                    "force_effectiveness_over_mass"
                ],
                "force_over_mass_conditioned": conditioned[
                    "force_effectiveness_over_mass"
                ],
                "force_over_mass_refined": refined[
                    "force_effectiveness_over_mass"
                ],
            }
        for stage_name, stage in (
            ("original", original),
            ("conditioned", conditioned),
            ("refined", refined),
        ):
            cog = np.asarray(stage["cog_position_body_m"])
            force = np.asarray(stage["force_effectiveness"])
            for axis, value in zip(("x", "y", "z"), cog):
                row["CoG_{}_{}_m".format(stage_name, axis)] = float(value)
            for rotor, value in enumerate(force, start=1):
                row[
                    "force_effectiveness_{}_{}".format(stage_name, rotor)
                ] = float(value)
        rows.append(row)
    return rows


def _format_array(value: Any, precision: int = 8) -> str:
    return np.array2string(
        np.asarray(value, dtype=float),
        precision=precision,
        separator=", ",
        suppress_small=False,
        max_line_width=140,
    )


def _human_summary_text(
    *,
    rows: Sequence[Mapping[str, Any]],
    original_distance: np.ndarray,
    refined_distance: np.ndarray,
    original_cross_cost: np.ndarray,
    original_cross_delta: np.ndarray,
    refined_cross_cost: np.ndarray,
    refined_cross_delta: np.ndarray,
    success_comparisons: Mapping[str, Any],
) -> str:
    labels = ("failure1", "failure2", "success")
    lines = [
        "# Three-bag CoG-prior reseed refinement summary",
        "",
        REFINEMENT_INTERPRETATION,
        "",
        "## Objectives and rotor lag",
        "",
        "| case | L original | L conditioned seed | L refined | Delta L | lag original [s] | lag refined [s] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in zip(labels, rows):
        lines.append(
            "| {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | {:.12g} |".format(
                label,
                row["L_original"],
                row["L_conditioned_seed_at_original_lag"],
                row["L_refined"],
                row["Delta_L_refined_minus_original"],
                row["rotor_lag_original"],
                row["rotor_lag_refined"],
            )
        )
    lines.extend(("", "## Scale-free physical parameters", ""))
    for label, row in zip(labels, rows):
        lines.extend(
            (
                "### {}".format(label),
                "",
                "```text",
                "CoG original    [m] = {}".format(_format_array(row["CoG_original_m"])),
                "CoG conditioned [m] = {}".format(_format_array(row["CoG_conditioned_m"])),
                "CoG refined     [m] = {}".format(_format_array(row["CoG_refined_m"])),
                "force original      = {}".format(_format_array(row["force_effectiveness_original"])),
                "force conditioned   = {}".format(_format_array(row["force_effectiveness_conditioned"])),
                "force refined       = {}".format(_format_array(row["force_effectiveness_refined"])),
                "J/m original    [m^2] =\n{}".format(_format_array(row["J_over_m_original"])),
                "J/m conditioned [m^2] =\n{}".format(_format_array(row["J_over_m_conditioned"])),
                "J/m refined     [m^2] =\n{}".format(_format_array(row["J_over_m_refined"])),
                "```",
                "",
            )
        )

    def append_matrix(title: str, matrix: np.ndarray) -> None:
        lines.extend(("### {}".format(title), "", "```text"))
        lines.append("rows=target, columns=source: {}".format(", ".join(labels)))
        lines.append(_format_array(matrix))
        lines.extend(("```", ""))

    lines.extend(("## Distribution and cross-evaluation matrices", ""))
    append_matrix("Original conservative quotient distance", original_distance)
    append_matrix("Refined conservative quotient distance", refined_distance)
    append_matrix("Original absolute cross cost", original_cross_cost)
    append_matrix("Original delta cross cost", original_cross_delta)
    append_matrix("Refined absolute cross cost", refined_cross_cost)
    append_matrix("Refined delta cross cost", refined_cross_delta)
    lines.extend(("## Failure-to-success comparisons", "", "```json"))
    lines.append(json.dumps(success_comparisons, indent=2, sort_keys=True))
    lines.extend(("```", ""))
    return "\n".join(lines)


def _annotated_matrix(axis: Any, matrix: np.ndarray, labels: Sequence[str], title: str) -> None:
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                "{:.4g}".format(matrix[row, column]),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    axis.figure.colorbar(image, ax=axis, shrink=0.75)


def _write_report(
    path: Path,
    *,
    bag_ids: Sequence[str],
    stages: Mapping[str, Mapping[str, np.ndarray]],
    original_distance: np.ndarray,
    refined_distance: np.ndarray,
    original_cross_cost: np.ndarray,
    original_cross_delta: np.ndarray,
    refined_cross_cost: np.ndarray,
    refined_cross_delta: np.ndarray,
    success_comparisons: Mapping[str, Any],
) -> None:
    short_labels = ("failure1", "failure2", "success")
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
        index = np.arange(3)
        width = 0.24
        for stage_index, (stage_name, color) in enumerate(
            zip(("original", "conditioned", "refined"), ("C0", "C1", "C2"))
        ):
            stage = stages[stage_name]
            axes[0, 0].bar(
                index + (stage_index - 1) * width,
                stage["objective"],
                width,
                label=stage_name,
                color=color,
            )
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xticks(index, short_labels)
        axes[0, 0].set_ylabel("identity objective")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        for stage_name, marker in (("original", "o"), ("refined", "s")):
            axes[0, 1].plot(
                index,
                stages[stage_name]["rotor_lag"],
                marker=marker,
                label=stage_name,
            )
        axes[0, 1].set_xticks(index, short_labels)
        axes[0, 1].set_ylabel("rotor lag [s]")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        components = ("x", "y", "z")
        for bag_index, label in enumerate(short_labels):
            axes[1, 0].plot(
                components,
                stages["original"]["cog"][bag_index] * 1000.0,
                "o--",
                alpha=0.65,
                label="{} original".format(label),
            )
            axes[1, 0].plot(
                components,
                stages["refined"]["cog"][bag_index] * 1000.0,
                "s-",
                label="{} refined".format(label),
            )
        axes[1, 0].set_ylabel("CoG [mm]")
        axes[1, 0].legend(fontsize=6, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)

        for bag_index, label in enumerate(short_labels):
            axes[1, 1].plot(
                np.arange(4),
                stages["original"]["force_over_mass"][bag_index],
                "o--",
                alpha=0.65,
                label="{} original".format(label),
            )
            axes[1, 1].plot(
                np.arange(4),
                stages["refined"]["force_over_mass"][bag_index],
                "s-",
                label="{} refined".format(label),
            )
        axes[1, 1].set_xticks(np.arange(4), ("f1/m", "f2/m", "f3/m", "f4/m"))
        axes[1, 1].set_ylabel("force effectiveness / mass")
        axes[1, 1].legend(fontsize=6, ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        figure.suptitle("CoG reseed refinement: three-bag physical summary")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = [
            "Exact scale-free physical values",
            "",
            "Rows below are original -> conditioned seed -> refined.",
            "",
        ]
        for bag_index, bag_id in enumerate(bag_ids):
            lines.extend(
                (
                    bag_id,
                    "  CoG [m] = {} -> {} -> {}".format(
                        _format_array(stages["original"]["cog"][bag_index], 6),
                        _format_array(stages["conditioned"]["cog"][bag_index], 6),
                        _format_array(stages["refined"]["cog"][bag_index], 6),
                    ),
                    "  f/m = {} -> {} -> {}".format(
                        _format_array(stages["original"]["force_over_mass"][bag_index], 6),
                        _format_array(stages["conditioned"]["force_over_mass"][bag_index], 6),
                        _format_array(stages["refined"]["force_over_mass"][bag_index], 6),
                    ),
                    "  J/m original:\n{}".format(
                        _format_array(stages["original"]["inertia_over_mass"][bag_index], 6)
                    ),
                    "  J/m conditioned:\n{}".format(
                        _format_array(stages["conditioned"]["inertia_over_mass"][bag_index], 6)
                    ),
                    "  J/m refined:\n{}".format(
                        _format_array(stages["refined"]["inertia_over_mass"][bag_index], 6)
                    ),
                    "",
                )
            )
        axis.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=6.3,
        )
        figure.suptitle("Original / conditioned / refined physical audit")
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
        _annotated_matrix(
            axes[0], original_distance, short_labels, "original conservative distance"
        )
        _annotated_matrix(
            axes[1], refined_distance, short_labels, "refined conservative distance"
        )
        _annotated_matrix(
            axes[2],
            refined_distance - original_distance,
            short_labels,
            "refined - original distance",
        )
        figure.suptitle("Original versus refined quotient-distribution distance")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
        for axis, matrix, title in (
            (axes[0, 0], original_cross_cost, "original absolute cross cost"),
            (axes[0, 1], original_cross_delta, "original delta cost"),
            (axes[1, 0], refined_cross_cost, "refined absolute cross cost"),
            (axes[1, 1], refined_cross_delta, "refined delta cost"),
        ):
            _annotated_matrix(axis, matrix, short_labels, title)
        figure.suptitle("Profiled strict cross-evaluation (rows target, columns source)")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        for failure_name in ("failure1", "failure2"):
            figure = plt.figure(figsize=(11.0, 8.5))
            axis = figure.add_subplot(111)
            axis.axis("off")
            lines = [
                "{} comparisons to successful-bag solutions".format(
                    failure_name
                ),
                "",
                REFINEMENT_INTERPRETATION,
                "",
            ]
            for key, value in success_comparisons.items():
                if not key.startswith(failure_name + "_"):
                    continue
                lines.append(key)
                lines.append(json.dumps(value, indent=2, sort_keys=True))
                lines.append("")
            axis.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                family="monospace",
                fontsize=7,
            )
            figure.suptitle("Scientific comparison audit")
            pdf.savefig(figure)
            plt.close(figure)


def run_consensus(arguments: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    revision = source_commit(_PROJECT_ROOT)
    directory = output_run_directory(
        arguments.output_root,
        "cog_prior_reseed_refinement_consensus",
        arguments.run_id,
        commit=revision,
    )
    started = time.perf_counter()
    stage = "input_validation"
    try:
        loaded = [
            _load_refinement_directory(path)
            for path in arguments.refinement_directory
        ]
        if len(loaded) != 3:
            raise ValueError("exactly three refinement directories are required")
        by_bag = {item["bag_id"]: item for item in loaded}
        if set(by_bag) != set(_BAG_ORDER):
            raise ValueError("refinement inputs must be the exact three production bags")
        refinements = [by_bag[bag_id] for bag_id in _BAG_ORDER]
        for item in refinements:
            if item["status"].get("source_commit") != revision:
                raise ValueError("refinement source commit differs from consensus")

        stage = "distribution_comparison"
        original_cases = [item["original_case"] for item in refinements]
        refined_cases = [item["refined_case"] for item in refinements]
        original_coordinate = np.stack(
            [case["quotient_coordinate"] for case in original_cases]
        )
        refined_coordinate = np.stack(
            [case["quotient_coordinate"] for case in refined_cases]
        )
        original_covariance = np.stack(
            [
                case["quotient_covariance_conservative_fusion"]
                for case in original_cases
            ]
        )
        refined_covariance = np.stack(
            [
                case["quotient_covariance_conservative_fusion"]
                for case in refined_cases
            ]
        )
        original_squared, original_distance = _pairwise_distance(
            original_coordinate, original_covariance
        )
        refined_squared, refined_distance = _pairwise_distance(
            refined_coordinate, refined_covariance
        )

        stage = "cross_evaluation"
        model = load_vehicle_model(arguments.vehicle_model)
        problems = [_problem_from_case(case, model) for case in original_cases]
        original_cross_cost, original_cross_delta, original_cross_lag = (
            _cross_evaluate(original_cases, problems)
        )
        refined_cross_cost, refined_cross_delta, refined_cross_lag = (
            _cross_evaluate(refined_cases, problems)
        )

        stage = "summary"
        stages = {
            name: _stage_arrays(refinements, name)
            for name in ("original", "conditioned", "refined")
        }
        rows = _comparison_rows(refinements)
        original_success = refinements[2]["comparison"]["original"]
        refined_success = refinements[2]["comparison"]["refined"]
        success_comparisons: dict[str, Any] = {}
        for failure_index, failure_name in enumerate(("failure1", "failure2")):
            failure_original = refinements[failure_index]["comparison"]["original"]
            failure_refined = refinements[failure_index]["comparison"]["refined"]
            success_comparisons[
                "{}_original_vs_original_success".format(failure_name)
            ] = {
                "scale_free_physical_distance": _physical_distance(
                    failure_original, original_success
                ),
                "original_conservative_quotient_distance": float(
                    original_distance[failure_index, 2]
                ),
                "original_success_on_failure_absolute_cost": float(
                    original_cross_cost[failure_index, 2]
                ),
                "original_success_on_failure_delta_cost": float(
                    original_cross_delta[failure_index, 2]
                ),
                "original_failure_on_success_absolute_cost": float(
                    original_cross_cost[2, failure_index]
                ),
                "original_failure_on_success_delta_cost": float(
                    original_cross_delta[2, failure_index]
                ),
            }
            success_comparisons[
                "{}_refined_vs_original_success".format(failure_name)
            ] = {
                "scale_free_physical_distance": _physical_distance(
                    failure_refined, original_success
                ),
                "original_success_on_failure_absolute_cost": float(
                    original_cross_cost[failure_index, 2]
                ),
                "original_success_on_failure_delta_cost": float(
                    original_cross_delta[failure_index, 2]
                ),
                "refined_failure_on_success_absolute_cost": float(
                    refined_cross_cost[2, failure_index]
                ),
                "refined_failure_on_success_delta_cost": float(
                    refined_cross_delta[2, failure_index]
                ),
            }
            success_comparisons[
                "{}_refined_vs_refined_success".format(failure_name)
            ] = {
                "scale_free_physical_distance": _physical_distance(
                    failure_refined, refined_success
                ),
                "refined_conservative_quotient_distance": float(
                    refined_distance[failure_index, 2]
                ),
                "refined_failure_on_success_absolute_cost": float(
                    refined_cross_cost[2, failure_index]
                ),
                "refined_failure_on_success_delta_cost": float(
                    refined_cross_delta[2, failure_index]
                ),
            }

        metadata = {
            "source_commit": revision,
            "baseline_repository_commit": BASELINE_REPOSITORY_COMMIT,
            "baseline_case_source_commit": BASELINE_CASE_SOURCE_COMMIT,
            "conditioning_source_covariance_name": (
                CONDITIONING_SOURCE_COVARIANCE_NAME
            ),
            "conditioning_prior_role": "initialization_only",
            "refinement_prior_role": "none",
            "refinement_all_parameters_free": True,
            "refinement_lag_reestimated": True,
            "refinement_existing_estimator_reused": True,
        }
        payload = {
            **metadata,
            "status": "completed",
            "bag_ids": _BAG_ORDER,
            "refinement_directories": [item["directory"] for item in refinements],
            "summary_rows": rows,
            "original_pairwise_distance_squared_conservative_fusion": (
                original_squared
            ),
            "original_pairwise_distance_conservative_fusion": original_distance,
            "refined_pairwise_distance_squared_conservative_fusion": (
                refined_squared
            ),
            "refined_pairwise_distance_conservative_fusion": refined_distance,
            "original_cross_evaluation": {
                "cost": original_cross_cost,
                "delta_cost": original_cross_delta,
                "profiled_rotor_lag_seconds": original_cross_lag,
            },
            "refined_cross_evaluation": {
                "cost": refined_cross_cost,
                "delta_cost": refined_cross_delta,
                "profiled_rotor_lag_seconds": refined_cross_lag,
            },
            "success_comparisons": success_comparisons,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(directory / "result.json", payload)
        write_json(
            directory / "summary.json",
            {
                **metadata,
                "status": "completed",
                "rows": rows,
                "success_comparisons": success_comparisons,
            },
        )
        (directory / "summary.md").write_text(
            _human_summary_text(
                rows=rows,
                original_distance=original_distance,
                refined_distance=refined_distance,
                original_cross_cost=original_cross_cost,
                original_cross_delta=original_cross_delta,
                refined_cross_cost=refined_cross_cost,
                refined_cross_delta=refined_cross_delta,
                success_comparisons=success_comparisons,
            ),
            encoding="utf-8",
        )
        write_json(
            directory / "arguments.json",
            {
                **metadata,
                "refinement_directory": list(arguments.refinement_directory),
                "vehicle_model": arguments.vehicle_model,
                "output_root": arguments.output_root,
                "run_id": arguments.run_id,
            },
        )
        np.savez_compressed(
            directory / "arrays.npz",
            original_objective=stages["original"]["objective"],
            conditioned_objective=stages["conditioned"]["objective"],
            refined_objective=stages["refined"]["objective"],
            original_rotor_lag=stages["original"]["rotor_lag"],
            refined_rotor_lag=stages["refined"]["rotor_lag"],
            original_cog=stages["original"]["cog"],
            conditioned_cog=stages["conditioned"]["cog"],
            refined_cog=stages["refined"]["cog"],
            original_force_effectiveness=stages["original"][
                "force_effectiveness"
            ],
            conditioned_force_effectiveness=stages["conditioned"][
                "force_effectiveness"
            ],
            refined_force_effectiveness=stages["refined"][
                "force_effectiveness"
            ],
            original_inertia_over_mass=stages["original"][
                "inertia_over_mass"
            ],
            conditioned_inertia_over_mass=stages["conditioned"][
                "inertia_over_mass"
            ],
            refined_inertia_over_mass=stages["refined"][
                "inertia_over_mass"
            ],
            original_force_over_mass=stages["original"]["force_over_mass"],
            conditioned_force_over_mass=stages["conditioned"][
                "force_over_mass"
            ],
            refined_force_over_mass=stages["refined"]["force_over_mass"],
            original_pairwise_distance_squared_conservative_fusion=(
                original_squared
            ),
            original_pairwise_distance_conservative_fusion=original_distance,
            refined_pairwise_distance_squared_conservative_fusion=(
                refined_squared
            ),
            refined_pairwise_distance_conservative_fusion=refined_distance,
            original_cross_evaluation_cost=original_cross_cost,
            original_cross_evaluation_delta_cost=original_cross_delta,
            original_cross_evaluation_profiled_rotor_lag_seconds=(
                original_cross_lag
            ),
            refined_cross_evaluation_cost=refined_cross_cost,
            refined_cross_evaluation_delta_cost=refined_cross_delta,
            refined_cross_evaluation_profiled_rotor_lag_seconds=(
                refined_cross_lag
            ),
        )
        _write_report(
            directory / "report.pdf",
            bag_ids=_BAG_ORDER,
            stages=stages,
            original_distance=original_distance,
            refined_distance=refined_distance,
            original_cross_cost=original_cross_cost,
            original_cross_delta=original_cross_delta,
            refined_cross_cost=refined_cross_cost,
            refined_cross_delta=refined_cross_delta,
            success_comparisons=success_comparisons,
        )
        write_json(
            directory / "status.json",
            {
                **metadata,
                "status": "completed",
                "elapsed_seconds": payload["elapsed_seconds"],
            },
        )
        write_json(
            directory / "timing.json",
            {"elapsed_seconds": payload["elapsed_seconds"]},
        )
        return directory, payload
    except Exception as error:
        elapsed = time.perf_counter() - started
        failure = {
            "status": "failed",
            "source_commit": revision,
            "baseline_repository_commit": BASELINE_REPOSITORY_COMMIT,
            "baseline_case_source_commit": BASELINE_CASE_SOURCE_COMMIT,
            "conditioning_source_covariance_name": (
                CONDITIONING_SOURCE_COVARIANCE_NAME
            ),
            "conditioning_prior_role": "initialization_only",
            "refinement_prior_role": "none",
            "refinement_all_parameters_free": True,
            "refinement_lag_reestimated": True,
            "refinement_existing_estimator_reused": True,
            "failure_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        write_json(directory / "status.json", failure)
        write_json(directory / "timing.json", {"elapsed_seconds": elapsed})
        return directory, failure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refinement-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=_HERE / "outputs")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    directory, payload = run_consensus(build_argument_parser().parse_args(argv))
    print(directory)
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
