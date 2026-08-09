#!/usr/bin/env python3
"""Confidence/ridge analysis for the geometric Savitzky--Golay dynamics estimator.

The deterministic point estimate uses every valid raw mocap center.  For the
local Gaussian information factor, this script conservatively selects centers
whose complete W-second SG windows do not overlap.  Thus one raw pose sample is
not counted repeatedly merely because neighboring local-polynomial windows
share it.  The remaining non-overlapping residual-wrench samples retain the
same first-layer iid Gaussian model as spline_dynamics_confidence.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import deterministic_savgol_dynamics_estimator as deterministic
import spline_dynamics_confidence as legacy
from grape_param_estim.system import GRAVITY


SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v1"
OUTPUT_SUBDIRECTORY = "savgol_dynamics_confidence"

# The mature confidence helper functions refer to their module-global
# deterministic backend.  Point them at the SG backend for this process only.
legacy.deterministic = deterministic
legacy.load_flight_data = deterministic.load_flight_data
legacy.GRAVITY = GRAVITY


def _sanitize(value: Any) -> Any:
    return deterministic._json_sanitize(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _nonoverlapping_window_indices(
    time_axis: Sequence[float],
    window_seconds: float,
) -> np.ndarray:
    """Greedy maximum-cardinality ordered subset of disjoint equal windows."""

    time_value = np.asarray(time_axis, dtype=float)
    window = float(window_seconds)
    if (
        time_value.ndim != 1
        or time_value.size < 1
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
        or not np.isfinite(window)
        or window <= 0.0
    ):
        raise ValueError("confidence window selection inputs are invalid")

    selected: list[int] = []
    previous_right = -math.inf
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(time_value))),
        window,
    )
    half = 0.5 * window
    for index, center in enumerate(time_value):
        left = float(center - half)
        right = float(center + half)
        # Strict separation also avoids sharing a raw sample exactly on a
        # common boundary when timestamps happen to land there.
        if left > previous_right + tolerance:
            selected.append(index)
            previous_right = right
    result = np.asarray(selected, dtype=int)
    if result.size < 2:
        raise ValueError(
            "W={:.12g}s leaves only {} non-overlapping confidence window(s); "
            "at least two are required to estimate the 6x6 residual-wrench "
            "covariance".format(window, result.size)
        )
    return result


def _translation_covariance_summary(bag: Any) -> dict[str, Any]:
    covariance = np.asarray(
        bag.collocation.sensor_acceleration_world_covariance,
        dtype=float,
    )
    finite = np.all(np.isfinite(covariance), axis=(1, 2))
    if np.any(finite):
        diagonal = np.diagonal(
            covariance[finite],
            axis1=1,
            axis2=2,
        )
        std = np.sqrt(np.maximum(diagonal, 0.0))
        return {
            "available_fraction": float(np.mean(finite)),
            "mean_acceleration_std_xyz_m_per_s2": np.mean(std, axis=0),
            "median_acceleration_std_xyz_m_per_s2": np.median(std, axis=0),
            "maximum_acceleration_std_xyz_m_per_s2": np.max(std, axis=0),
            "used_directly_in_this_confidence_factor": False,
            "note": (
                "The local translation covariance is preserved as a diagnostic. "
                "This v1 confidence factor instead prevents SG overlap double "
                "counting by using disjoint windows; a later full correlated "
                "R3/SO3 observation model can use these covariances explicitly."
            ),
        }
    return {
        "available_fraction": 0.0,
        "used_directly_in_this_confidence_factor": False,
        "note": (
            "No local empirical covariance is available, usually because the "
            "window has exactly degree+1 samples and zero residual degrees of "
            "freedom."
        ),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = deterministic.create_argument_parser()
    parser.description = (
        "Analyze local parameter confidence/ridges for one degree-5 geometric "
        "Savitzky-Golay dynamics bag."
    )
    parser.add_argument(
        "--deterministic-result",
        type=Path,
        default=None,
        help=(
            "Optional result.json from deterministic_savgol_dynamics_estimator. "
            "When supplied, reuse its selected physical coordinate and lag "
            "instead of re-running the parameter optimizer."
        ),
    )
    return parser


def run(arguments: argparse.Namespace) -> int:
    if deterministic.PHYSICAL_DIMENSION != 14:
        raise SystemExit("SG confidence expects the 14-D physical coordinate")

    # SG estimator semantics: zero command-lag initialization unless explicitly
    # overridden.  Do this before config handling so the config's historical
    # initial_delay_seconds cannot silently re-enter.
    if arguments.initial_delay is None:
        arguments.initial_delay = 0.0
    deterministic._ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)
    deterministic._ACTIVE_BAGS.clear()
    arguments.spline_cv_folds = 0
    arguments.maximum_spline_acceleration = math.inf
    arguments.maximum_spline_angular_acceleration = math.inf

    try:
        config = deterministic.load_spline_config(arguments.config)
        vehicle_model = deterministic.load_vehicle_model(
            arguments.vehicle_model_json
        )
        parameter_prior = deterministic.load_parameter_prior(
            arguments.prior_json
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    specification = legacy._only_bag(config)
    initial_delay = float(arguments.initial_delay)
    deterministic._validate_arguments(arguments, config, initial_delay)

    started = time.perf_counter()
    print(
        "loading raw pose {}: {} [{:.3f}, {:.3f}] s".format(
            specification.bag_id,
            specification.path,
            specification.start,
            specification.end,
        ),
        flush=True,
    )
    flight = deterministic.load_flight_data(
        str(specification.path),
        start_local=specification.start,
        end_local=specification.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    try:
        bag = deterministic._build_bag_data(
            specification,
            1.0,
            flight,
            initial_delay,
            config.spline,
            arguments,
            vehicle_model.parameters,
            vehicle_model.geometry,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(
        "SG W={:.6g}s; full deterministic support [{:.6f}, {:.6f}]s, {} raw centers"
        .format(
            arguments.window_seconds,
            bag.collocation_time[0],
            bag.collocation_time[-1],
            bag.collocation_time.size,
        ),
        flush=True,
    )

    if arguments.deterministic_result is None:
        selected, optimizer_history = legacy._estimate_solution(
            bag,
            arguments,
            initial_delay,
            vehicle_model.parameters,
            parameter_prior,
        )
    else:
        result_path = arguments.deterministic_result.expanduser().resolve()
        try:
            deterministic_result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            deterministic_window = float(
                deterministic_result["settings"]["window_seconds"]
            )
            selection = deterministic_result["selection"]
            physical_coordinate = np.asarray(
                selection["physical_coordinate"],
                dtype=float,
            )
            delay_seconds = float(selection["delay_seconds"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                "deterministic SG result cannot be read: {}".format(result_path)
            ) from error
        if not np.isclose(
            deterministic_window,
            float(arguments.window_seconds),
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise SystemExit(
                "deterministic-result W={}s does not match requested W={}s".format(
                    deterministic_window, arguments.window_seconds
                )
            )
        if physical_coordinate.shape != (deterministic.PHYSICAL_DIMENSION,):
            raise SystemExit("deterministic-result physical coordinate is not 14-D")
        problem = deterministic.SplineDynamicsProblem(
            (bag,),
            vehicle_model.parameters,
            parameter_prior,
        )
        selected_evaluation = problem.evaluate_strict(
            physical_coordinate,
            delay_seconds,
        )
        selected = deterministic.DynamicsSolution(
            physical_coordinate=physical_coordinate.copy(),
            delay_seconds=delay_seconds,
            evaluation=selected_evaluation,
            optimizer={
                "source": "precomputed_deterministic_result",
                "result_json": str(result_path),
            },
        )
        optimizer_history = {
            "source": "precomputed_deterministic_result",
            "result_json": str(result_path),
        }
        print(
            "reused deterministic parameter result {}".format(result_path),
            flush=True,
        )
    evaluation = selected.evaluation.bag_evaluations[0]
    print(
        "selected delay {:.6f}s; mass {:.6g} kg".format(
            selected.delay_seconds,
            selected.evaluation.decoded.parameters.mass,
        ),
        flush=True,
    )

    scales = legacy._reference_scales(vehicle_model.parameters)
    raw_per_dimensionless = legacy._parameter_raw_per_dimensionless(
        scales["length_m"]
    )
    wrench_scale = legacy._wrench_dimensionless_scale(
        scales["force_N"],
        scales["torque_Nm"],
    )

    wrench_raw_full = np.asarray(
        evaluation.residual_body_wrench,
        dtype=float,
    )
    wrench_jacobian_raw_full = np.asarray(
        evaluation.residual_wrench_jacobian,
        dtype=float,
    )
    expected_shape = (
        wrench_raw_full.shape[0],
        6,
        deterministic.PHYSICAL_DIMENSION,
    )
    if wrench_jacobian_raw_full.shape != expected_shape:
        raise RuntimeError(
            "residual-wrench Jacobian has unexpected shape {}".format(
                wrench_jacobian_raw_full.shape
            )
        )

    confidence_indices = _nonoverlapping_window_indices(
        bag.collocation_time,
        float(arguments.window_seconds),
    )
    confidence_time = np.asarray(
        bag.collocation_time[confidence_indices],
        dtype=float,
    )
    wrench_raw = wrench_raw_full[confidence_indices]
    wrench_jacobian_raw = wrench_jacobian_raw_full[confidence_indices]
    print(
        "confidence likelihood uses {} disjoint SG windows ({} full deterministic centers)"
        .format(confidence_indices.size, bag.collocation_time.size),
        flush=True,
    )

    wrench_dimensionless = wrench_raw * wrench_scale[None, :]
    jacobian_dimensionless = (
        wrench_jacobian_raw
        * wrench_scale[None, :, None]
        * raw_per_dimensionless[None, None, :]
    )

    wrench_mean_dimensionless = np.mean(wrench_dimensionless, axis=0)
    wrench_covariance_dimensionless = np.cov(
        wrench_dimensionless,
        rowvar=False,
        ddof=1,
    )
    whitener, covariance_diagnostics = legacy._pseudo_whitener(
        wrench_covariance_dimensionless
    )
    centered = wrench_dimensionless - wrench_mean_dimensionless[None, :]
    whitened_residual = np.einsum(
        "ij,nj->ni",
        whitener,
        centered,
    )
    whitened_jacobian = np.einsum(
        "ij,njk->nik",
        whitener,
        jacobian_dimensionless,
    )
    stacked_jacobian = whitened_jacobian.reshape(
        -1,
        deterministic.PHYSICAL_DIMENSION,
    )
    data_information = stacked_jacobian.T @ stacked_jacobian
    data_information_vector = -(
        stacked_jacobian.T @ whitened_residual.reshape(-1)
    )

    svd = legacy._svd_payload(
        stacked_jacobian,
        deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    posterior = legacy._posterior_payload(
        data_information=data_information,
        data_information_vector=data_information_vector,
        selected=selected,
        raw_per_dimensionless=raw_per_dimensionless,
        reference_parameters=vehicle_model.parameters,
        parameter_prior=parameter_prior,
        names=deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    prefix = legacy._prefix_information(
        confidence_time,
        whitened_jacobian,
        posterior["prior_covariance_dimensionless"],
    )
    reconstruction = legacy._trajectory_reconstruction(
        bag,
        selected,
        evaluation,
        arguments,
        initial_delay,
        vehicle_model.parameters,
    )
    angular_excitation = legacy._angular_excitation_payload(bag)
    translation_covariance = _translation_covariance_summary(bag)

    wrench_mean_raw = np.mean(wrench_raw, axis=0)
    wrench_covariance_raw = np.cov(wrench_raw, rowvar=False, ddof=1)
    mahalanobis_squared = np.sum(
        whitened_residual * whitened_residual,
        axis=1,
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
        / specification.bag_id
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    vt = np.asarray(
        [
            mode["dimensionless_direction"]
            for mode in reversed(svd["weak_directions"])
        ],
        dtype=float,
    )
    likelihood_payload = {
        "schema": "grape-param-estim/parameter-likelihood/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "id": specification.bag_id,
            "path": specification.path,
            "start_seconds": specification.start,
            "end_seconds": specification.end,
        },
        "vehicle_model": {
            "source_path": vehicle_model.source_path,
            "reference_parameters": legacy._parameter_payload(
                vehicle_model.parameters
            ),
        },
        "savgol_observation_model": {
            "degree": 5,
            "window_seconds": float(arguments.window_seconds),
            "deterministic_raw_center_count": int(bag.collocation_time.size),
            "likelihood_disjoint_window_count": int(confidence_indices.size),
            "likelihood_center_times_seconds": confidence_time,
            "selection_rule": (
                "earliest-center greedy subset whose complete centered W-second "
                "raw-pose windows are pairwise disjoint"
            ),
        },
        "linearization_point": {
            "coordinate_names": deterministic.PHYSICAL_PARAMETER_NAMES,
            "raw_coordinate": selected.physical_coordinate,
            "dimensionless_coordinate": (
                selected.physical_coordinate / raw_per_dimensionless
            ),
            "physical_vector_order": deterministic.PHYSICAL_VALUE_NAMES,
            "physical_vector": deterministic.physical_parameter_vector(
                selected.evaluation.decoded.parameters
            ),
            "delay_seconds": float(selected.delay_seconds),
        },
        "local_gaussian_likelihood": {
            "coordinate": "dimensionless delta from linearization_point",
            "information_matrix": data_information,
            "information_vector": data_information_vector,
            "definition": (
                "-log L(delta) = 0.5 delta^T Lambda delta - eta^T delta + constant"
            ),
        },
        "identifiability": {
            "singular_values": svd["singular_values_descending"],
            "Vt": vt,
            "numerical_rank": svd["numerical_rank"],
            "nullity": svd["nullity"],
        },
        "external_wrench_model": {
            "mean_dimensionless": wrench_mean_dimensionless,
            "covariance_dimensionless": wrench_covariance_dimensionless,
            "iid_scope": "pairwise-disjoint SG windows only",
        },
    }
    posterior_payload = {
        "schema": "grape-param-estim/parameter-posterior/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vehicle_model_path": vehicle_model.source_path,
        "prior": {
            "source_path": parameter_prior.source_path,
            "input": parameter_prior.raw,
            "vector_order": deterministic.PHYSICAL_VALUE_NAMES,
            "mean": parameter_prior.mean,
            "std": parameter_prior.std,
        },
        "posterior_coordinate": {
            "coordinate_names": deterministic.PHYSICAL_PARAMETER_NAMES,
            "mean_dimensionless": posterior["posterior_mean_dimensionless"],
            "covariance_dimensionless": posterior[
                "posterior_covariance_dimensionless"
            ],
            "precision_dimensionless": posterior[
                "posterior_precision_dimensionless"
            ],
        },
        "posterior_physical": posterior["posterior_physical"],
        "data_identifiability": likelihood_payload["identifiability"],
        "savgol_observation_model": likelihood_payload[
            "savgol_observation_model"
        ],
    }
    _write_json(
        output_directory / "parameter_likelihood.json",
        likelihood_payload,
    )
    _write_json(
        output_directory / "parameter_posterior.json",
        posterior_payload,
    )

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vehicle_model_path": vehicle_model.source_path,
        "parameter_prior_path": parameter_prior.source_path,
        "bag": {
            "id": specification.bag_id,
            "path": specification.path,
            "start_seconds": specification.start,
            "end_seconds": specification.end,
            "deterministic_raw_center_count": int(bag.collocation_time.size),
            "confidence_disjoint_window_count": int(confidence_indices.size),
            "confidence_center_time_seconds": confidence_time,
            "window_seconds": float(arguments.window_seconds),
            "polynomial_degree": 5,
        },
        "assumptions": {
            "pose_front_end": (
                "raw timestamped pose; degree-5 local polynomial in R3 and "
                "geometric Savitzky-Golay on SO3"
            ),
            "SG_overlap_handling": (
                "the deterministic estimate uses every valid raw center; the "
                "confidence factor uses only pairwise-disjoint complete SG "
                "windows so shared raw pose samples are not counted as "
                "independent likelihood samples"
            ),
            "external_wrench": (
                "the remaining disjoint-window residual-wrench samples are "
                "treated as iid Gaussian with nonzero empirical mean and 6x6 "
                "empirical covariance"
            ),
            "pose_measurement_covariance": (
                "translation local-LS covariance is reported diagnostically; "
                "a full correlated R3/SO3 measurement model is not yet fused "
                "into the wrench likelihood"
            ),
            "information_accumulation": (
                "sample information matrices are summed, not averaged"
            ),
        },
        "savgol_translation_covariance_diagnostic": translation_covariance,
        "nondimensionalization": {
            "reference_scales": scales,
            "parameter_raw_per_dimensionless": raw_per_dimensionless,
            "wrench_raw_to_dimensionless_diagonal": wrench_scale,
            "parameter_names": deterministic.PHYSICAL_PARAMETER_NAMES,
        },
        "deterministic_solution": {
            "physical_coordinate": selected.physical_coordinate,
            "delay_seconds": selected.delay_seconds,
            "objective_cost": float(deterministic._solution_cost(selected)),
            "parameters": legacy._parameter_payload(
                selected.evaluation.decoded.parameters
            ),
            "optimizer_history": optimizer_history,
        },
        "external_wrench_model": {
            "mean_raw_body_wrench": wrench_mean_raw,
            "covariance_raw_body_wrench": wrench_covariance_raw,
            "mean_dimensionless": wrench_mean_dimensionless,
            "covariance_dimensionless": wrench_covariance_dimensionless,
            "covariance_diagnostics": covariance_diagnostics,
            "mahalanobis_squared_per_sample": mahalanobis_squared,
            "mahalanobis_squared_mean": float(np.mean(mahalanobis_squared)),
            "mahalanobis_squared_median": float(np.median(mahalanobis_squared)),
            "confidence_sample_indices_in_full_deterministic_support": (
                confidence_indices
            ),
        },
        "data_information": {
            "matrix_dimensionless": data_information,
            "svd": svd,
        },
        "angular_excitation_diagnostic": angular_excitation,
        "information_vs_duration": prefix,
        "prior_and_local_posterior": posterior,
        "trajectory_reconstruction_check": reconstruction,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_directory / "confidence.json", payload)
    legacy._write_pdf(
        output_directory / "confidence.pdf",
        time_axis=confidence_time,
        wrench_dimensionless=wrench_dimensionless,
        wrench_mean=wrench_mean_dimensionless,
        wrench_covariance=wrench_covariance_dimensionless,
        svd=svd,
        prefix=prefix,
        posterior=posterior,
        reconstruction=reconstruction,
        angular_excitation=angular_excitation,
        names=deterministic.PHYSICAL_PARAMETER_NAMES,
    )

    print(
        "SG confidence analysis written to {}".format(output_directory),
        flush=True,
    )
    replay_metrics = reconstruction["external_wrench_replay"]["metrics"]
    parameter_metrics = reconstruction["parameter_only"]["metrics"]
    print(
        "trajectory check: parameter-only RMSE {:.6g} m / {:.6g} deg; "
        "with external-wrench replay {:.6g} m / {:.6g} deg".format(
            parameter_metrics["position_rmse_m"],
            parameter_metrics["orientation_angle_rmse_deg"],
            replay_metrics["position_rmse_m"],
            replay_metrics["orientation_angle_rmse_deg"],
        ),
        flush=True,
    )
    print(
        "data-only ridge check: weakest relative information {:.3e}".format(
            svd["weakest_relative_information_strength"]
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
