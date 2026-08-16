#!/usr/bin/env python3
"""Gauge-quotient and cross-evaluation diagnostics for three single-bag fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from rotor_lag import local_strict_cell_descent  # noqa: E402
from single_bag_savgol_core import (  # noqa: E402
    BASE_PLAN_COMMIT,
    SingleBagDynamicsProblem,
    load_vehicle_model,
    prepare_single_bag_dataset,
)
from single_bag_savgol_estimator import (  # noqa: E402
    actuator_parameters_from_arguments,
)
from single_bag_savgol_covariance import (  # noqa: E402
    machine_pseudoinverse_symmetric,
)
from single_bag_savgol_reports import (  # noqa: E402
    output_run_directory,
    source_commit,
    write_json,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-cross-bag-matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read JSON: {}".format(path)) from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _load_case(directory: Path) -> dict[str, Any]:
    case = directory.expanduser().resolve()
    result = _read_object(case / "result.json")
    arguments = _read_object(case / "arguments.json")
    if result.get("status") != "completed":
        raise ValueError("consensus input case is not completed: {}".format(case))
    with np.load(case / "arrays.npz") as arrays:
        quotient_basis = np.asarray(arrays["quotient_basis"]).copy()
        quotient_coordinate = np.asarray(arrays["quotient_coordinate"]).copy()
        quotient_covariance_overlap = np.asarray(
            arrays["quotient_covariance_overlap_corrected"]
        ).copy()
        quotient_covariance_wrench = np.asarray(
            arrays["quotient_covariance_wrench_corrected"]
        ).copy()
        quotient_covariance_conservative = np.asarray(
            arrays["quotient_covariance_conservative_fusion"]
        ).copy()
    parameters = result["parameters"]
    estimated = parameters["estimated"]
    return {
        "directory": case,
        "result": result,
        "arguments": arguments,
        "bag_id": str(arguments.get("bag_id") or case.parent.parent.name),
        "coordinate": np.asarray(parameters["chart_coordinate"], dtype=float),
        "rotor_lag": float(parameters["rotor_lag_seconds"]),
        "quotient_basis": quotient_basis,
        "quotient_coordinate": quotient_coordinate,
        "quotient_covariance": quotient_covariance_overlap,
        "quotient_covariance_overlap_corrected": (
            quotient_covariance_overlap
        ),
        "quotient_covariance_wrench_corrected": quotient_covariance_wrench,
        "quotient_covariance_conservative_fusion": (
            quotient_covariance_conservative
        ),
        "mass": float(estimated["mass_kg"]),
        "inertia": np.asarray(estimated["inertia_kg_m2"], dtype=float),
        "force_effectiveness": np.asarray(
            estimated["force_effectiveness"], dtype=float
        ),
        "cog": np.asarray(estimated["cog_position_body_m"], dtype=float),
    }


def _problem_from_case(case: dict[str, Any], model: Any) -> SingleBagDynamicsProblem:
    values = dict(case["arguments"])
    arguments = argparse.Namespace(**values)
    flight = load_flight_data(
        path=str(values["bag"]),
        start_local=float(values["bag_start"]),
        end_local=float(values["bag_end"]),
        include_fc_specific_force=True,
        compute_sha256=False,
        bag_id=values.get("bag_id"),
    )
    dataset = prepare_single_bag_dataset(
        flight=flight,
        window_seconds=float(values["sg_window"]),
        degree=int(values["sg_degree"]),
        covariance_mode=str(values["covariance_mode"]),
        geometric_correction=not bool(values["naive_so3_derivatives"]),
    )
    return SingleBagDynamicsProblem(
        dataset,
        model,
        actuator_parameters_from_arguments(arguments),
        gimbal_source=str(values["gimbal_source"]),
    )


def _validated_psd_inverse(
    value: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, int]:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("{} must be one finite square matrix".format(label))
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            "{} is materially indefinite (minimum eigenvalue {})".format(
                label, float(eigenvalues[0])
            )
        )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    return machine_pseudoinverse_symmetric(matrix), eigenvalues, rank


def _squared_distance(
    first_coordinate: np.ndarray,
    first_covariance: np.ndarray,
    second_coordinate: np.ndarray,
    second_covariance: np.ndarray,
    label: str,
) -> float:
    delta = np.asarray(first_coordinate) - np.asarray(second_coordinate)
    inverse, _eigenvalues, _rank = _validated_psd_inverse(
        np.asarray(first_covariance) + np.asarray(second_covariance), label
    )
    value = float(delta @ inverse @ delta)
    if value < 0.0:
        raise RuntimeError(
            "{} produced a negative squared distance {}".format(label, value)
        )
    return value


def _pairwise_distance(
    coordinate: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = coordinate.shape[0]
    squared = np.zeros((count, count))
    for first in range(count):
        for second in range(first + 1, count):
            value = _squared_distance(
                coordinate[first],
                covariance[first],
                coordinate[second],
                covariance[second],
                "pairwise covariance ({}, {})".format(first, second),
            )
            squared[first, second] = squared[second, first] = value
    return squared, np.sqrt(squared)


def fuse_quotient_gaussians(
    coordinate: np.ndarray, covariance: np.ndarray
) -> dict[str, np.ndarray | int]:
    """Form the zero-jitter Gaussian product using machine-rank inverses."""

    locations = np.asarray(coordinate, dtype=float)
    covariances = np.asarray(covariance, dtype=float)
    if (
        locations.ndim != 2
        or covariances.shape
        != (locations.shape[0], locations.shape[1], locations.shape[1])
        or np.any(~np.isfinite(locations))
        or np.any(~np.isfinite(covariances))
    ):
        raise ValueError("Gaussian fusion inputs are invalid")
    precision = np.empty_like(covariances)
    information = np.zeros(locations.shape[1], dtype=float)
    for index in range(locations.shape[0]):
        precision[index], _eigenvalues, _rank = _validated_psd_inverse(
            covariances[index], "fusion covariance {}".format(index)
        )
        information += precision[index] @ locations[index]
    fused_precision = 0.5 * (
        np.sum(precision, axis=0) + np.sum(precision, axis=0).T
    )
    fused_covariance, precision_eigenvalues, rank = _validated_psd_inverse(
        fused_precision, "fused quotient precision"
    )
    scale = float(np.max(np.abs(precision_eigenvalues)))
    tolerance = fused_precision.shape[0] * np.finfo(float).eps * scale
    eigenvalues, eigenvectors = np.linalg.eigh(fused_precision)
    unresolved = eigenvectors[:, eigenvalues <= tolerance]
    fused_coordinate = fused_covariance @ information
    return {
        "per_bag_precision": precision,
        "precision": fused_precision,
        "covariance": fused_covariance,
        "coordinate": fused_coordinate,
        "rank": rank,
        "unresolved_directions": unresolved,
    }


def _cross_evaluate(
    cases: Sequence[dict[str, Any]], problems: Sequence[SingleBagDynamicsProblem]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(cases)
    cost = np.empty((count, count))
    lag = np.empty((count, count))
    for target, problem in enumerate(problems):
        initial_lag = float(cases[target]["rotor_lag"])
        for source, case in enumerate(cases):
            coordinate = np.asarray(case["coordinate"])

            def evaluate(cell):
                evaluation = problem.evaluate_physical(
                    coordinate, cell.representative, command_mode="strict"
                )
                return evaluation.cost, cell.representative

            profile = local_strict_cell_descent(
                problem.strict_lag_grid, initial_lag, evaluate
            )
            cost[target, source] = profile.selected.cost
            lag[target, source] = profile.selected.cell.representative
    delta = cost - np.diag(cost)[:, None]
    return cost, delta, lag


def _write_report(
    path: Path,
    bag_ids: Sequence[str],
    pairwise_distance_overlap: np.ndarray,
    pairwise_distance_wrench: np.ndarray,
    pairwise_distance_conservative: np.ndarray,
    cross_cost: np.ndarray,
    cross_delta: np.ndarray,
    inertia_over_mass: np.ndarray,
    force_over_mass: np.ndarray,
    cog: np.ndarray,
    fused: dict[str, np.ndarray | int],
    bag_to_fused_distance: np.ndarray,
) -> None:
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
        for axis, matrix, title in (
            (
                axes[0],
                pairwise_distance_overlap,
                "pairwise quotient distance: SG overlap",
            ),
            (axes[1], cross_delta, "cross-evaluation Delta L_ij"),
        ):
            image = axis.imshow(matrix, cmap="viridis")
            axis.set_xticks(range(len(bag_ids)), bag_ids, rotation=30, ha="right")
            axis.set_yticks(range(len(bag_ids)), bag_ids)
            axis.set_title(title)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(column, row, "{:.3g}".format(matrix[row, column]), ha="center", va="center", color="white")
            figure.colorbar(image, ax=axis, shrink=0.8)
        figure.suptitle("Three-bag scale-quotient consensus")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
        for axis, matrix, title in zip(
            axes,
            (
                pairwise_distance_overlap,
                pairwise_distance_wrench,
                pairwise_distance_conservative,
            ),
            (
                "SG overlap",
                "centered excess wrench",
                "conservative fusion",
            ),
        ):
            image = axis.imshow(matrix, cmap="viridis")
            axis.set_xticks(
                range(len(bag_ids)), bag_ids, rotation=30, ha="right"
            )
            axis.set_yticks(range(len(bag_ids)), bag_ids)
            axis.set_title(title)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(
                        column,
                        row,
                        "{:.3g}".format(matrix[row, column]),
                        ha="center",
                        va="center",
                        color="white",
                    )
            figure.colorbar(image, ax=axis, shrink=0.75)
        figure.suptitle("Three-bag quotient covariance distances")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = ["Gauge-invariant physical quantities", ""]
        for index, bag_id in enumerate(bag_ids):
            lines.extend(
                (
                    bag_id,
                    "  J/m = {}".format(np.array2string(inertia_over_mass[index], precision=6)),
                    "  f/m = {}".format(np.array2string(force_over_mass[index], precision=6)),
                    "  c   = {}".format(np.array2string(cog[index], precision=6)),
                    "  cross costs = {}".format(np.array2string(cross_cost[index], precision=6)),
                    "",
                )
            )
        axis.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8)
        figure.suptitle("Three-bag physical and cross-evaluation audit")
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        covariance_eigenvalues = np.linalg.eigvalsh(
            np.asarray(fused["covariance"])
        )
        lines = [
            "Conservative quotient Gaussian product",
            "",
            "fused precision rank: {} / 13".format(fused["rank"]),
            "unresolved fused direction count: {}".format(
                np.asarray(fused["unresolved_directions"]).shape[1]
            ),
            "fused coordinate:",
            np.array2string(np.asarray(fused["coordinate"]), precision=7),
            "",
            "fused covariance eigenvalues:",
            np.array2string(covariance_eigenvalues, precision=7),
            "",
            "bag-to-fused distance:",
            np.array2string(bag_to_fused_distance, precision=7),
            "",
            (
                "Interpretation: conservative local Gaussian uncertainty for "
                "fusion; not a calibrated generative covariance."
            ),
        ]
        axis.text(
            0.03,
            0.97,
            "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=8,
        )
        figure.suptitle("Three-bag conservative fusion diagnostic")
        pdf.savefig(figure)
        plt.close(figure)


def run_consensus(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    cases = [_load_case(Path(value)) for value in arguments.case_directory]
    if len(cases) != 3:
        raise ValueError("cross-bag consensus requires exactly three case directories")
    basis = np.stack([case["quotient_basis"] for case in cases])
    if not np.allclose(basis, basis[0], rtol=0.0, atol=2e-13):
        raise ValueError("single-bag outputs do not share one quotient basis")
    coordinate = np.stack([case["quotient_coordinate"] for case in cases])
    covariance_overlap = np.stack(
        [case["quotient_covariance_overlap_corrected"] for case in cases]
    )
    covariance_wrench = np.stack(
        [case["quotient_covariance_wrench_corrected"] for case in cases]
    )
    covariance_conservative = np.stack(
        [case["quotient_covariance_conservative_fusion"] for case in cases]
    )
    squared_overlap, distance_overlap = _pairwise_distance(
        coordinate, covariance_overlap
    )
    squared_wrench, distance_wrench = _pairwise_distance(
        coordinate, covariance_wrench
    )
    squared_conservative, distance_conservative = _pairwise_distance(
        coordinate, covariance_conservative
    )
    distance_ratio = np.full_like(distance_overlap, np.nan)
    np.divide(
        distance_conservative,
        distance_overlap,
        out=distance_ratio,
        where=distance_overlap > 0.0,
    )
    fused = fuse_quotient_gaussians(coordinate, covariance_conservative)
    bag_to_fused_squared = np.asarray(
        [
            _squared_distance(
                coordinate[index],
                covariance_conservative[index],
                np.asarray(fused["coordinate"]),
                np.asarray(fused["covariance"]),
                "bag {} to fused covariance".format(index),
            )
            for index in range(len(cases))
        ]
    )
    bag_to_fused_distance = np.sqrt(bag_to_fused_squared)
    model = load_vehicle_model(arguments.vehicle_model)
    problems = [_problem_from_case(case, model) for case in cases]
    cross_cost, cross_delta, cross_lag = _cross_evaluate(cases, problems)
    inertia_over_mass = np.stack(
        [case["inertia"] / case["mass"] for case in cases]
    )
    force_over_mass = np.stack(
        [case["force_effectiveness"] / case["mass"] for case in cases]
    )
    cog = np.stack([case["cog"] for case in cases])
    revision = source_commit(_PROJECT_ROOT)
    directory = output_run_directory(
        arguments.output_root,
        "consensus",
        arguments.run_id,
        commit=revision,
    )
    bag_ids = [case["bag_id"] for case in cases]
    payload = {
        "status": "completed",
        "source_commit": revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "bag_ids": bag_ids,
        "case_directories": [case["directory"] for case in cases],
        "scale_free": {
            "inertia_over_mass_m2": inertia_over_mass,
            "force_effectiveness_over_mass": force_over_mass,
            "cog_position_body_m": cog,
        },
        "quotient": {
            "basis": basis[0],
            "coordinate": coordinate,
            "covariance_overlap_corrected": covariance_overlap,
            "covariance_wrench_corrected": covariance_wrench,
            "covariance_conservative_fusion": covariance_conservative,
            "pairwise_distance_squared": squared_overlap,
            "pairwise_distance": distance_overlap,
            "pairwise_distance_squared_overlap_corrected": squared_overlap,
            "pairwise_distance_overlap_corrected": distance_overlap,
            "pairwise_distance_squared_wrench_corrected": squared_wrench,
            "pairwise_distance_wrench_corrected": distance_wrench,
            "pairwise_distance_squared_conservative_fusion": (
                squared_conservative
            ),
            "pairwise_distance_conservative_fusion": distance_conservative,
            "conservative_to_overlap_distance_ratio": distance_ratio,
            "fused_precision_conservative_fusion": fused["precision"],
            "fused_covariance_conservative_fusion": fused["covariance"],
            "fused_coordinate_conservative_fusion": fused["coordinate"],
            "fused_precision_rank_conservative_fusion": fused["rank"],
            "fused_unresolved_directions_conservative_fusion": (
                fused["unresolved_directions"]
            ),
            "fused_quotient_precision_conservative_fusion": (
                fused["precision"]
            ),
            "fused_quotient_covariance_conservative_fusion": (
                fused["covariance"]
            ),
            "fused_quotient_coordinate_conservative_fusion": (
                fused["coordinate"]
            ),
            "bag_to_fused_distance_squared": bag_to_fused_squared,
            "bag_to_fused_distance": bag_to_fused_distance,
        },
        "cross_evaluation": {
            "cost": cross_cost,
            "delta_cost": cross_delta,
            "profiled_rotor_lag_seconds": cross_lag,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(directory / "result.json", payload)
    write_json(
        directory / "status.json",
        {
            "status": "completed",
            "source_commit": revision,
            "base_plan_commit": BASE_PLAN_COMMIT,
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    write_json(
        directory / "arguments.json",
        {
            "case_directory": list(arguments.case_directory),
            "vehicle_model": arguments.vehicle_model,
            "run_id": arguments.run_id,
        },
    )
    write_json(
        directory / "timing.json",
        {"elapsed_seconds": payload["elapsed_seconds"]},
    )
    np.savez_compressed(
        directory / "arrays.npz",
        quotient_basis=basis[0],
        quotient_coordinate=coordinate,
        quotient_covariance_overlap_corrected=covariance_overlap,
        quotient_covariance_wrench_corrected=covariance_wrench,
        quotient_covariance_conservative_fusion=covariance_conservative,
        pairwise_distance_squared=squared_overlap,
        pairwise_distance=distance_overlap,
        pairwise_distance_squared_overlap_corrected=squared_overlap,
        pairwise_distance_overlap_corrected=distance_overlap,
        pairwise_distance_squared_wrench_corrected=squared_wrench,
        pairwise_distance_wrench_corrected=distance_wrench,
        pairwise_distance_squared_conservative_fusion=squared_conservative,
        pairwise_distance_conservative_fusion=distance_conservative,
        conservative_to_overlap_distance_ratio=distance_ratio,
        fused_quotient_precision_conservative_fusion=fused["precision"],
        fused_quotient_covariance_conservative_fusion=fused["covariance"],
        fused_quotient_coordinate_conservative_fusion=fused["coordinate"],
        fused_quotient_precision_rank_conservative_fusion=np.asarray(
            fused["rank"]
        ),
        fused_quotient_unresolved_directions_conservative_fusion=fused[
            "unresolved_directions"
        ],
        bag_to_fused_distance_squared=bag_to_fused_squared,
        bag_to_fused_distance=bag_to_fused_distance,
        cross_evaluation_cost=cross_cost,
        cross_evaluation_delta_cost=cross_delta,
        cross_evaluation_profiled_rotor_lag_seconds=cross_lag,
        inertia_over_mass=inertia_over_mass,
        force_effectiveness_over_mass=force_over_mass,
        cog_position_body_m=cog,
    )
    _write_report(
        directory / "report.pdf",
        bag_ids,
        distance_overlap,
        distance_wrench,
        distance_conservative,
        cross_cost,
        cross_delta,
        inertia_over_mass,
        force_over_mass,
        cog,
        fused,
        bag_to_fused_distance,
    )
    return directory, payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-directory", type=Path, action="append", required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=_HERE / "outputs")
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    directory, _payload = run_consensus(build_argument_parser().parse_args(argv))
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
