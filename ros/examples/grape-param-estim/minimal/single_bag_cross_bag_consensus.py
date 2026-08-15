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
        quotient_covariance = np.asarray(
            arrays["quotient_covariance_overlap_corrected"]
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
        "quotient_covariance": quotient_covariance,
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


def _pairwise_distance(
    coordinate: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = coordinate.shape[0]
    squared = np.zeros((count, count))
    for first in range(count):
        for second in range(first + 1, count):
            delta = coordinate[first] - coordinate[second]
            combined = covariance[first] + covariance[second]
            value = float(delta @ np.linalg.pinv(combined) @ delta)
            squared[first, second] = squared[second, first] = max(0.0, value)
    return squared, np.sqrt(squared)


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
    pairwise_distance: np.ndarray,
    cross_cost: np.ndarray,
    cross_delta: np.ndarray,
    inertia_over_mass: np.ndarray,
    force_over_mass: np.ndarray,
    cog: np.ndarray,
) -> None:
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
        for axis, matrix, title in (
            (axes[0], pairwise_distance, "pairwise quotient distance d_ij"),
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


def run_consensus(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    cases = [_load_case(Path(value)) for value in arguments.case_directory]
    if len(cases) != 3:
        raise ValueError("cross-bag consensus requires exactly three case directories")
    basis = np.stack([case["quotient_basis"] for case in cases])
    if not np.allclose(basis, basis[0], rtol=0.0, atol=2e-13):
        raise ValueError("single-bag outputs do not share one quotient basis")
    coordinate = np.stack([case["quotient_coordinate"] for case in cases])
    covariance = np.stack([case["quotient_covariance"] for case in cases])
    squared_distance, distance = _pairwise_distance(coordinate, covariance)
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
            "covariance_overlap_corrected": covariance,
            "pairwise_distance_squared": squared_distance,
            "pairwise_distance": distance,
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
        quotient_covariance_overlap_corrected=covariance,
        pairwise_distance_squared=squared_distance,
        pairwise_distance=distance,
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
        distance,
        cross_cost,
        cross_delta,
        inertia_over_mass,
        force_over_mass,
        cog,
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
