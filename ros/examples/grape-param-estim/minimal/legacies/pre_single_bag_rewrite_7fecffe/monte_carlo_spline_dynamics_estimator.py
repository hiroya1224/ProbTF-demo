#!/usr/bin/env python3
"""Monte Carlo pose-uncertainty propagation through spline-dynamics estimation.

For one selected bag this script runs the nominal deterministic spline-dynamics
solve once, freezes the selected spline knot spacing and command lag, then
repeats only:

  noisy pose samples -> fixed-spacing quintic spline -> fixed-lag 13-D
  physical-parameter solve.

The external-wrench reconstruction and report generation are intentionally not
run for Monte Carlo samples.

Expected base repository commit:
    9d113ea754e2f9b4520a148fcb134e8781d6a8e6
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

import deterministic_spline_dynamics_estimator as deterministic
from spline_trajectory import PoseSplineSelection, fit_pose_spline_fixed
from grape_param_estim.real_rosbag import load_flight_data


EXPECTED_BASE_COMMIT = "9d113ea754e2f9b4520a148fcb134e8781d6a8e6"
SCHEMA = "grape-param-estim/monte-carlo-spline-dynamics/v1"
OUTPUT_SUBDIRECTORY = "monte_carlo_spline_dynamics"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class _FallbackProgress:
    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.count = 0
        self.postfix = ""

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self.count:
            print("", flush=True)
        return False

    def set_postfix_str(self, text: str) -> None:
        self.postfix = str(text)

    def update(self, amount: int = 1) -> None:
        self.count += int(amount)
        suffix = (" " + self.postfix) if self.postfix else ""
        print(
            "\rMonte Carlo {}/{}{}".format(self.count, self.total, suffix),
            end="",
            flush=True,
        )


def _progress(total: int):
    if _tqdm is None:
        return _FallbackProgress(total)
    return _tqdm(
        total=total,
        desc="Monte Carlo",
        unit="trajectory",
        dynamic_ncols=True,
        smoothing=0.1,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(
            deterministic._json_sanitize(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(
            deterministic._json_sanitize(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "{}_pid{}".format(stamp, os.getpid())


def _bag_seed_component(bag_id: str) -> int:
    digest = hashlib.sha256(bag_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sample_rng(base_seed: int, bag_id: str, sample_index: int):
    seed_words = (
        int(base_seed),
        _bag_seed_component(bag_id),
        int(sample_index),
    )
    generator = np.random.default_rng(np.random.SeedSequence(seed_words))
    return generator, seed_words


def _vehicle_parameter_payload(parameters: Any) -> dict[str, Any]:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return {
        "mass_kg": float(parameters.mass),
        "inertia_kg_m2": inertia,
        "inertia_principal_moments_kg_m2": np.linalg.eigvalsh(inertia),
        "cog_offset_m": np.asarray(parameters.cog_offset, dtype=float),
        "force_effectiveness": np.asarray(
            parameters.force_effectiveness, dtype=float
        ),
        "torque_effectiveness": np.asarray(
            parameters.torque_effectiveness, dtype=float
        ),
        "linear_drag": np.asarray(parameters.linear_drag, dtype=float),
        "angular_drag": np.asarray(parameters.angular_drag, dtype=float),
    }


def _solution_payload(solution: Any) -> dict[str, Any]:
    return {
        "physical_parameter_names": tuple(
            deterministic.strict.PHYSICAL_PARAMETER_NAMES
        ),
        "physical_coordinate": np.asarray(
            solution.physical_coordinate, dtype=float
        ),
        "delay_seconds": float(solution.delay_seconds),
        "objective_cost": float(deterministic._solution_cost(solution)),
        "data_loss": float(solution.evaluation.data_loss),
        "prior_cost": float(solution.evaluation.prior_cost),
        "optimizer": dict(solution.optimizer),
        "parameters": _vehicle_parameter_payload(
            solution.evaluation.decoded.parameters
        ),
    }


def _select_only_bag(config: Any):
    specifications = tuple(config.multi_bag.bags)
    if len(specifications) != 1:
        available = ", ".join(
            specification.bag_id for specification in specifications
        )
        raise SystemExit(
            "Monte Carlo config must contain exactly one bag; "
            "found {} ({})".format(
                len(specifications),
                available or "no bag IDs",
            )
        )
    return specifications[0]


def _estimate_nominal_solution(
    bag: Any,
    arguments: argparse.Namespace,
    initial_delay: float,
) -> tuple[Any, dict[str, Any]]:
    """Run the deterministic spline-dynamics lag search once."""

    problem = deterministic.SplineDynamicsProblem((bag,), arguments.prior_weight)
    initial_physical = np.zeros(
        deterministic.strict.PHYSICAL_DIMENSION, dtype=float
    )
    physical_lower, physical_upper = deterministic._physical_bounds(
        initial_physical, arguments.physical_bound_scale
    )
    smooth_lower = np.concatenate(
        (
            physical_lower,
            np.asarray((arguments.delay_bounds[0],), dtype=float),
        )
    )
    smooth_upper = np.concatenate(
        (
            physical_upper,
            np.asarray((arguments.delay_bounds[1],), dtype=float),
        )
    )
    coordinate = np.concatenate(
        (initial_physical, np.asarray((initial_delay,), dtype=float))
    )
    coordinate = np.minimum(np.maximum(coordinate, smooth_lower), smooth_upper)

    smooth_stages = []
    for stage_index, width in enumerate(arguments.smoothstep_width_fractions):
        coordinate, evaluation, optimizer = deterministic._solve_smooth(
            problem,
            coordinate,
            float(width),
            smooth_lower,
            smooth_upper,
            arguments,
        )
        smooth_stages.append(
            {
                "stage_index": stage_index,
                "width_fraction": float(width),
                "physical_coordinate": coordinate[
                    : deterministic.strict.PHYSICAL_DIMENSION
                ].copy(),
                "delay_seconds": float(
                    coordinate[deterministic.DELAY_INDEX]
                ),
                "objective_cost": 0.5
                * float(evaluation.residual @ evaluation.residual),
                "data_loss": float(evaluation.data_loss),
                "prior_cost": float(evaluation.prior_cost),
                "optimizer": dict(optimizer),
            }
        )

    smooth_physical = coordinate[
        : deterministic.strict.PHYSICAL_DIMENSION
    ].copy()
    smooth_delay = float(coordinate[deterministic.DELAY_INDEX])
    candidate_delays = deterministic.smooth.zoh_polish_delays(
        smooth_delay,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.delay_bounds,
    )
    screening_costs = np.asarray(
        [
            0.5
            * float(
                (
                    evaluation
                    := problem.evaluate_strict(smooth_physical, float(delay))
                ).residual
                @ evaluation.residual
            )
            for delay in candidate_delays
        ],
        dtype=float,
    )
    top_count = min(arguments.zoh_polish_top_k, candidate_delays.size)
    top_indices = np.argsort(screening_costs, kind="stable")[:top_count]

    strict_solutions = []
    refined_candidates = []
    for rank, candidate_index in enumerate(top_indices):
        delay = float(candidate_delays[candidate_index])
        solution = deterministic._solve_strict(
            problem,
            smooth_physical,
            delay,
            physical_lower,
            physical_upper,
            arguments,
        )
        strict_solutions.append(solution)
        refined_candidates.append(
            {
                "rank": rank,
                "delay_seconds": delay,
                "screening_cost": float(screening_costs[candidate_index]),
                "refined": _solution_payload(solution),
            }
        )

    if not strict_solutions:
        raise RuntimeError("nominal strict-ZOH polish produced no solution")
    selected = min(strict_solutions, key=deterministic._solution_cost)
    return selected, {
        "smooth_stages": smooth_stages,
        "candidate_delays_seconds": candidate_delays,
        "screening_costs": screening_costs,
        "refined_candidates": refined_candidates,
        "selected": _solution_payload(selected),
    }


def _perturb_pose_observations(
    *,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
    position_sigma_m: float,
    rotation_sigma_rad: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    position_value = np.asarray(position, dtype=float)
    orientation_value = np.asarray(orientation_xyzw, dtype=float)
    position_noise = rng.normal(
        0.0, position_sigma_m, size=position_value.shape
    )
    rotation_vector_noise = rng.normal(
        0.0, rotation_sigma_rad, size=(orientation_value.shape[0], 3)
    )

    observed_rotation = Rotation.from_quat(orientation_value).as_matrix()
    local_noise_rotation = Rotation.from_rotvec(rotation_vector_noise).as_matrix()
    noisy_rotation = np.einsum(
        "nij,njk->nik", observed_rotation, local_noise_rotation
    )
    noisy_orientation = Rotation.from_matrix(noisy_rotation).as_quat()

    diagnostics = {
        "position_noise_vector_rms_m": float(
            np.sqrt(np.mean(np.sum(position_noise * position_noise, axis=1)))
        ),
        "position_noise_component_rms_m": float(
            np.sqrt(np.mean(position_noise * position_noise))
        ),
        "rotation_noise_vector_rms_rad": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        rotation_vector_noise * rotation_vector_noise, axis=1
                    )
                )
            )
        ),
        "rotation_noise_component_rms_rad": float(
            np.sqrt(np.mean(rotation_vector_noise * rotation_vector_noise))
        ),
    }
    return (
        position_value + position_noise,
        np.asarray(noisy_orientation, dtype=float),
        diagnostics,
    )


def _bag_with_noisy_pose_spline(
    nominal_bag: Any,
    noisy_position: np.ndarray,
    noisy_orientation_xyzw: np.ndarray,
) -> tuple[Any, dict[str, float]]:
    """Refit spline coefficients only, keeping nominal model selection fixed."""

    direct = nominal_bag.direct_problem
    spacing = float(nominal_bag.spline_selection.selected_spacing_seconds)
    spline = fit_pose_spline_fixed(
        time_axis=direct.output_time,
        sensor_position=noisy_position,
        sensor_orientation_xyzw=noisy_orientation_xyzw,
        body_to_pose_sensor_rotation=direct.pose_body_to_sensor_rotation,
        knot_spacing_seconds=spacing,
    )
    selection = PoseSplineSelection(
        spline=spline,
        selected_spacing_seconds=spacing,
        candidates=tuple(),
        fit_position_rmse_m=float("nan"),
        fit_orientation_rmse_rad=float("nan"),
        fit_metric_rmse_m=float("nan"),
    )
    collocation = spline.evaluate(nominal_bag.collocation_time)
    diagnostics = {
        "maximum_acceleration_m_per_s2": float(
            np.max(
                np.linalg.norm(
                    collocation.sensor_acceleration_world, axis=1
                )
            )
        ),
        "maximum_angular_acceleration_rad_per_s2": float(
            np.max(
                np.linalg.norm(
                    collocation.body_angular_acceleration, axis=1
                )
            )
        ),
    }
    return (
        replace(
            nominal_bag,
            spline_selection=selection,
            collocation=collocation,
        ),
        diagnostics,
    )


def _vector_summary(
    values: np.ndarray,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("summary values must contain samples")
    count, dimension = array.shape
    if names is not None and len(tuple(names)) != dimension:
        raise ValueError("summary names have the wrong dimension")
    covariance = (
        np.cov(array, rowvar=False, ddof=1)
        if count >= 2
        else np.zeros((dimension, dimension), dtype=float)
    )
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    result = {
        "sample_count": int(count),
        "mean": np.mean(array, axis=0),
        "standard_deviation": (
            np.std(array, axis=0, ddof=1)
            if count >= 2
            else np.zeros(dimension, dtype=float)
        ),
        "covariance": covariance,
        "quantile_0_025": np.quantile(array, 0.025, axis=0),
        "median": np.quantile(array, 0.5, axis=0),
        "quantile_0_975": np.quantile(array, 0.975, axis=0),
    }
    if names is not None:
        result["names"] = tuple(names)
    return result


def _running_summary(
    successful_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not successful_records:
        return {"successful_sample_count": 0}
    coordinate = np.asarray(
        [record["physical_coordinate"] for record in successful_records],
        dtype=float,
    )
    mass = np.asarray(
        [record["parameters"]["mass_kg"] for record in successful_records],
        dtype=float,
    )
    cog = np.asarray(
        [record["parameters"]["cog_offset_m"] for record in successful_records],
        dtype=float,
    )
    force_effectiveness = np.asarray(
        [
            record["parameters"]["force_effectiveness"]
            for record in successful_records
        ],
        dtype=float,
    )
    principal_moments = np.asarray(
        [
            record["parameters"]["inertia_principal_moments_kg_m2"]
            for record in successful_records
        ],
        dtype=float,
    )
    return {
        "successful_sample_count": len(successful_records),
        "physical_coordinate": _vector_summary(
            coordinate, deterministic.strict.PHYSICAL_PARAMETER_NAMES
        ),
        "mass_kg": _vector_summary(mass, ("mass_kg",)),
        "cog_offset_m": _vector_summary(cog, ("x", "y", "z")),
        "force_effectiveness": _vector_summary(
            force_effectiveness,
            ("rotor_1", "rotor_2", "rotor_3", "rotor_4"),
        ),
        "inertia_principal_moments_kg_m2": _vector_summary(
            principal_moments,
            ("principal_1", "principal_2", "principal_3"),
        ),
    }


def _checkpoint_payload(
    *,
    arguments: argparse.Namespace,
    run_id: str,
    run_directory: Path,
    nominal_payload: Mapping[str, Any],
    completed_count: int,
    successful_records: Sequence[Mapping[str, Any]],
    failure_count: int,
    started_wall_time: str,
    started_perf: float,
    status: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "expected_base_commit": EXPECTED_BASE_COMMIT,
        "run_id": run_id,
        "bag_id": arguments.bag_id,
        "patterns_requested": int(arguments.patterns),
        "patterns_completed": int(completed_count),
        "successful_patterns": len(successful_records),
        "failed_patterns": int(failure_count),
        "position_sigma_m_per_axis": float(arguments.position_sigma_m),
        "rotation_sigma_deg_per_axis": float(arguments.rotation_sigma_deg),
        "seed": int(arguments.seed),
        "started_utc": started_wall_time,
        "elapsed_seconds": float(time.perf_counter() - started_perf),
        "run_directory": run_directory,
        "nominal": nominal_payload,
        "running_summary": _running_summary(successful_records),
        "files": {
            "samples_jsonl": "samples.jsonl",
            "checkpoint_json": "checkpoint.json",
            "summary_json": "summary.json",
            "samples_npz": "samples.npz",
        },
    }


def _write_final_npz(
    path: Path,
    successful_records: Sequence[Mapping[str, Any]],
) -> None:
    if not successful_records:
        np.savez_compressed(
            path,
            sample_index=np.empty(0, dtype=int),
            physical_coordinate=np.empty(
                (0, deterministic.strict.PHYSICAL_DIMENSION), dtype=float
            ),
        )
        return
    np.savez_compressed(
        path,
        sample_index=np.asarray(
            [record["sample_index"] for record in successful_records],
            dtype=int,
        ),
        physical_coordinate=np.asarray(
            [record["physical_coordinate"] for record in successful_records],
            dtype=float,
        ),
        mass_kg=np.asarray(
            [record["parameters"]["mass_kg"] for record in successful_records],
            dtype=float,
        ),
        inertia_kg_m2=np.asarray(
            [record["parameters"]["inertia_kg_m2"] for record in successful_records],
            dtype=float,
        ),
        inertia_principal_moments_kg_m2=np.asarray(
            [
                record["parameters"]["inertia_principal_moments_kg_m2"]
                for record in successful_records
            ],
            dtype=float,
        ),
        cog_offset_m=np.asarray(
            [record["parameters"]["cog_offset_m"] for record in successful_records],
            dtype=float,
        ),
        force_effectiveness=np.asarray(
            [
                record["parameters"]["force_effectiveness"]
                for record in successful_records
            ],
            dtype=float,
        ),
        objective_cost=np.asarray(
            [record["objective_cost"] for record in successful_records],
            dtype=float,
        ),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = deterministic.create_argument_parser()
    parser.description = (
        "Monte Carlo propagation of pose-observation uncertainty through "
        "the deterministic spline-dynamics parameter estimator. One bag is "
        "selected from the JSON config per process."
    )
    parser.add_argument(
        "--patterns",
        type=int,
        required=True,
        help="Number of Monte Carlo noisy trajectories; required.",
    )
    parser.add_argument(
        "--position-sigma-m",
        type=float,
        default=0.05,
        help=(
            "Independent Gaussian position sigma per x/y/z observation "
            "component [m]. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=10.0,
        help=(
            "Independent Gaussian local rotation-vector sigma per x/y/z "
            "component [deg]. Default: 10."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base random seed. Bag ID and pattern index are mixed into it "
            "for reproducible independent streams."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional output run ID. Default is UTC timestamp + PID, so "
            "simultaneous processes do not collide."
        ),
    )
    return parser


def _validate_monte_carlo_arguments(arguments: argparse.Namespace) -> None:
    values = np.asarray(
        (arguments.position_sigma_m, arguments.rotation_sigma_deg),
        dtype=float,
    )
    if (
        arguments.patterns <= 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or arguments.seed < 0
    ):
        raise SystemExit(
            "patterns must be positive; sigmas and seed must be finite/nonnegative"
        )
    if arguments.run_id is not None and not _SAFE_RUN_ID.fullmatch(
        arguments.run_id
    ):
        raise SystemExit(
            "--run-id must start with an alphanumeric and contain only "
            "letters, digits, '.', '_' or '-'"
        )


def run(arguments: argparse.Namespace) -> int:
    _validate_monte_carlo_arguments(arguments)
    try:
        config = deterministic.load_spline_config(arguments.config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    specification = _select_only_bag(config)
    # Keep the resolved ID on the Namespace because the rest of this script
    # already uses arguments.bag_id for checkpoint metadata.
    arguments.bag_id = specification.bag_id
    initial_delay = (
        config.multi_bag.initial_delay_seconds
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    deterministic._validate_arguments(arguments, config, initial_delay)

    run_id = _default_run_id() if arguments.run_id is None else arguments.run_id
    run_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
        / specification.bag_id
        / run_id
    )
    if run_directory.exists():
        raise SystemExit(
            "run output already exists; choose another --run-id: {}".format(
                run_directory
            )
        )
    run_directory.mkdir(parents=True, exist_ok=False)
    started_wall_time = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    static_metadata = {
        "schema": SCHEMA,
        "status": "initializing",
        "expected_base_commit": EXPECTED_BASE_COMMIT,
        "run_id": run_id,
        "config": arguments.config.expanduser().resolve(),
        "bag": {
            "id": specification.bag_id,
            "path": specification.path,
            "start_seconds": float(specification.start),
            "end_seconds": float(specification.end),
            "config_weight_ignored_for_single_bag_mc": float(
                specification.weight
            ),
        },
        "patterns_requested": int(arguments.patterns),
        "noise_model": {
            "position": (
                "independent additive Gaussian noise at every estimator pose "
                "observation; equal x/y/z sigma"
            ),
            "position_sigma_m_per_axis": float(arguments.position_sigma_m),
            "rotation": (
                "independent right-multiplicative SO(3) rotation-vector "
                "Gaussian noise at every estimator pose observation; equal "
                "local x/y/z sigma"
            ),
            "rotation_sigma_deg_per_axis": float(arguments.rotation_sigma_deg),
            "temporal_correlation": "none",
        },
        "seed": int(arguments.seed),
        "started_utc": started_wall_time,
        "algorithm": {
            "lag": (
                "estimated once from the nominal trajectory by the same "
                "smooth-continuation + strict-ZOH polish as the deterministic "
                "estimator, then frozen for every Monte Carlo pattern"
            ),
            "spline_knot_spacing": (
                "selected once by nominal blocked CV, then frozen; each "
                "Monte Carlo pattern refits only spline coefficients"
            ),
            "physical_parameters": (
                "13-D strict-ZOH least-squares solve at fixed lag for each "
                "perturbed spline, initialized from the nominal solution"
            ),
            "external_wrench_reconstruction": False,
            "forward_rollout_reports": False,
        },
    }
    _atomic_write_json(run_directory / "run.json", static_metadata)

    print("Monte Carlo output: {}".format(run_directory), flush=True)
    print(
        "loading {}: {} [{:.3f}, {:.3f}] s".format(
            specification.bag_id,
            specification.path,
            specification.start,
            specification.end,
        ),
        flush=True,
    )
    flight = load_flight_data(
        str(specification.path),
        start_local=specification.start,
        end_local=specification.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )

    print("nominal spline selection + one-time lag estimation", flush=True)
    nominal_bag = deterministic._build_bag_data(
        specification,
        1.0,
        flight,
        initial_delay,
        config.spline,
        arguments,
    )
    nominal_solution, nominal_search = _estimate_nominal_solution(
        nominal_bag, arguments, initial_delay
    )
    nominal_payload = {
        "selected_knot_spacing_seconds": float(
            nominal_bag.spline_selection.selected_spacing_seconds
        ),
        "collocation_time_bounds_seconds": (
            float(nominal_bag.collocation_time[0]),
            float(nominal_bag.collocation_time[-1]),
        ),
        "observation_count": int(nominal_bag.direct_problem.output_time.size),
        "collocation_count": int(nominal_bag.collocation_time.size),
        "lag_search": nominal_search,
    }
    _atomic_write_json(run_directory / "nominal.json", nominal_payload)
    print(
        "fixed lag={:.6f}s, knot spacing={:.6g}s, nominal mass={:.6g} kg".format(
            nominal_solution.delay_seconds,
            nominal_bag.spline_selection.selected_spacing_seconds,
            nominal_solution.evaluation.decoded.parameters.mass,
        ),
        flush=True,
    )

    initial_physical = np.zeros(
        deterministic.strict.PHYSICAL_DIMENSION, dtype=float
    )
    physical_lower, physical_upper = deterministic._physical_bounds(
        initial_physical, arguments.physical_bound_scale
    )
    physical_start = np.asarray(
        nominal_solution.physical_coordinate, dtype=float
    ).copy()
    fixed_delay = float(nominal_solution.delay_seconds)
    original_observations = nominal_bag.direct_problem.observations
    rotation_sigma_rad = math.radians(arguments.rotation_sigma_deg)

    successful_records: list[dict[str, Any]] = []
    failure_count = 0
    completed_count = 0
    samples_path = run_directory / "samples.jsonl"
    checkpoint_path = run_directory / "checkpoint.json"
    _atomic_write_json(
        checkpoint_path,
        _checkpoint_payload(
            arguments=arguments,
            run_id=run_id,
            run_directory=run_directory,
            nominal_payload=nominal_payload,
            completed_count=0,
            successful_records=successful_records,
            failure_count=0,
            started_wall_time=started_wall_time,
            started_perf=started_perf,
            status="running",
        ),
    )

    with samples_path.open("a", encoding="utf-8") as stream:
        with _progress(arguments.patterns) as progress:
            for sample_index in range(arguments.patterns):
                sample_started = time.perf_counter()
                rng, seed_words = _sample_rng(
                    arguments.seed, specification.bag_id, sample_index
                )
                try:
                    (
                        noisy_position,
                        noisy_orientation,
                        noise_diagnostics,
                    ) = _perturb_pose_observations(
                        position=original_observations.sensor_position,
                        orientation_xyzw=(
                            original_observations.sensor_orientation_xyzw
                        ),
                        position_sigma_m=arguments.position_sigma_m,
                        rotation_sigma_rad=rotation_sigma_rad,
                        rng=rng,
                    )
                    noisy_bag, spline_diagnostics = _bag_with_noisy_pose_spline(
                        nominal_bag, noisy_position, noisy_orientation
                    )
                    sample_problem = deterministic.SplineDynamicsProblem(
                        (noisy_bag,), arguments.prior_weight
                    )
                    solution = deterministic._solve_strict(
                        sample_problem,
                        physical_start,
                        fixed_delay,
                        physical_lower,
                        physical_upper,
                        arguments,
                    )
                    record = {
                        "sample_index": int(sample_index),
                        "status": "success",
                        "seed_words": seed_words,
                        "elapsed_seconds": float(
                            time.perf_counter() - sample_started
                        ),
                        "noise_diagnostics": noise_diagnostics,
                        "spline_diagnostics": spline_diagnostics,
                        **_solution_payload(solution),
                    }
                    successful_records.append(record)
                    progress.set_postfix_str(
                        "mass={:.3f}kg nfev={} ok={} fail={}".format(
                            record["parameters"]["mass_kg"],
                            record["optimizer"]["nfev"],
                            len(successful_records),
                            failure_count,
                        )
                    )
                except Exception as error:
                    failure_count += 1
                    record = {
                        "sample_index": int(sample_index),
                        "status": "failed",
                        "seed_words": seed_words,
                        "elapsed_seconds": float(
                            time.perf_counter() - sample_started
                        ),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                    progress.set_postfix_str(
                        "FAILED {} ok={} fail={}".format(
                            type(error).__name__,
                            len(successful_records),
                            failure_count,
                        )
                    )

                _append_jsonl(stream, record)
                completed_count += 1
                _atomic_write_json(
                    checkpoint_path,
                    _checkpoint_payload(
                        arguments=arguments,
                        run_id=run_id,
                        run_directory=run_directory,
                        nominal_payload=nominal_payload,
                        completed_count=completed_count,
                        successful_records=successful_records,
                        failure_count=failure_count,
                        started_wall_time=started_wall_time,
                        started_perf=started_perf,
                        status="running",
                    ),
                )
                progress.update(1)

    final_payload = _checkpoint_payload(
        arguments=arguments,
        run_id=run_id,
        run_directory=run_directory,
        nominal_payload=nominal_payload,
        completed_count=completed_count,
        successful_records=successful_records,
        failure_count=failure_count,
        started_wall_time=started_wall_time,
        started_perf=started_perf,
        status="completed",
    )
    final_payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(run_directory / "summary.json", final_payload)
    _atomic_write_json(checkpoint_path, final_payload)
    _write_final_npz(run_directory / "samples.npz", successful_records)

    print(
        "completed: {} success, {} failed; summary: {}".format(
            len(successful_records),
            failure_count,
            run_directory / "summary.json",
        ),
        flush=True,
    )
    return 0 if successful_records else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)
    return run(arguments)


if __name__ == "__main__":
    sys.exit(main())
