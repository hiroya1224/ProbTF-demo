#!/usr/bin/env python3
"""Confidence/ridge analysis for the geometric Savitzky--Golay dynamics estimator.

The deterministic point estimate and the local Gaussian information factor both
use every valid centered raw-mocap SG evaluation time.  The residual-wrench
samples retain the same first-layer iid Gaussian model as
``spline_dynamics_confidence.py``.
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


SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v3"
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
                "The local translation covariance is preserved as a diagnostic "
                "and is not used directly in the current residual-wrench likelihood."
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



def _residual_parameter_diagnostics(
    *,
    wrench_raw: np.ndarray,
    wrench_dimensionless: np.ndarray,
    jacobian_dimensionless: np.ndarray,
    wrench_covariance_dimensionless: np.ndarray,
    wrench_scale: np.ndarray,
    raw_per_dimensionless: np.ndarray,
    selected: Any,
    reference_parameters: Any,
) -> dict[str, Any]:
    wrench = np.asarray(wrench_dimensionless, dtype=float)
    jacobian = np.asarray(jacobian_dimensionless, dtype=float)
    count = wrench.shape[0]
    matrix = jacobian.reshape(-1, deterministic.PHYSICAL_DIMENSION)
    vector = wrench.reshape(-1)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * (float(singular[0]) if singular.size else 0.0)
    )
    positive = singular > tolerance
    rank = int(np.count_nonzero(positive))
    if rank:
        pseudoinverse = (
            vt[positive].T
            @ np.diag(1.0 / singular[positive])
            @ u[:, positive].T
        )
    else:
        pseudoinverse = np.zeros(
            (deterministic.PHYSICAL_DIMENSION, vector.size), dtype=float
        )

    best_fit_delta_dimensionless = -(pseudoinverse @ vector)
    parameter_change = matrix @ best_fit_delta_dimensionless
    remaining = vector + parameter_change
    absorbed = vector - remaining
    total_energy = float(vector @ vector)
    remaining_energy = float(remaining @ remaining)
    absorbed_fraction = (
        0.0 if total_energy <= 0.0 else 1.0 - remaining_energy / total_energy
    )
    remaining_series = remaining.reshape(count, 6)
    absorbed_series = absorbed.reshape(count, 6)

    mean_dimensionless = np.mean(wrench, axis=0)
    repeated_mean = np.tile(mean_dimensionless, count)
    mean_induced_bias_dimensionless = -(pseudoinverse @ repeated_mean)

    covariance = np.asarray(wrench_covariance_dimensionless, dtype=float)
    blocks = pseudoinverse.reshape(
        deterministic.PHYSICAL_DIMENSION, count, 6
    ).transpose(1, 0, 2)
    parameter_covariance_dimensionless = np.einsum(
        "nai,ij,nbj->ab",
        blocks,
        covariance,
        blocks,
    )
    parameter_covariance_dimensionless = 0.5 * (
        parameter_covariance_dimensionless
        + parameter_covariance_dimensionless.T
    )
    parameter_second_moment_dimensionless = (
        parameter_covariance_dimensionless
        + np.outer(
            mean_induced_bias_dimensionless,
            mean_induced_bias_dimensionless,
        )
    )

    raw_scale = np.asarray(raw_per_dimensionless, dtype=float)
    best_fit_delta_raw = best_fit_delta_dimensionless * raw_scale
    mean_induced_bias_raw = mean_induced_bias_dimensionless * raw_scale
    parameter_covariance_raw = (
        raw_scale[:, None]
        * parameter_covariance_dimensionless
        * raw_scale[None, :]
    )
    parameter_second_moment_raw = (
        raw_scale[:, None]
        * parameter_second_moment_dimensionless
        * raw_scale[None, :]
    )

    parameterization = deterministic.SplinePhysicalParameterization(
        reference_parameters
    )
    _decoded, parameter_jacobian = parameterization.decode_with_jacobian(
        selected.physical_coordinate,
        selected.delay_seconds,
    )
    physical_jacobian = deterministic.physical_parameter_jacobian(
        parameter_jacobian
    )
    physical_best_fit_correction = physical_jacobian @ best_fit_delta_raw
    physical_mean_induced_bias = physical_jacobian @ mean_induced_bias_raw
    physical_covariance = (
        physical_jacobian
        @ parameter_covariance_raw
        @ physical_jacobian.T
    )
    physical_covariance = 0.5 * (physical_covariance + physical_covariance.T)
    physical_second_moment = (
        physical_jacobian
        @ parameter_second_moment_raw
        @ physical_jacobian.T
    )
    physical_second_moment = 0.5 * (
        physical_second_moment + physical_second_moment.T
    )

    wrench_scale_value = np.asarray(wrench_scale, dtype=float)
    remaining_raw = remaining_series / wrench_scale_value[None, :]
    absorbed_raw = absorbed_series / wrench_scale_value[None, :]
    second_moment_raw = (
        np.asarray(wrench_raw, dtype=float).T
        @ np.asarray(wrench_raw, dtype=float)
        / count
    )
    second_moment_dimensionless = wrench.T @ wrench / count
    remaining_second_moment_dimensionless = (
        remaining_series.T @ remaining_series / count
    )

    per_component_fraction = []
    for component in range(6):
        denominator = float(np.sum(wrench[:, component] ** 2))
        numerator = float(np.sum(remaining_series[:, component] ** 2))
        per_component_fraction.append(
            None if denominator <= 0.0 else 1.0 - numerator / denominator
        )

    return {
        "linearized_model": (
            "w_bar(theta + delta) ~= w_bar(theta) + J_bar delta_bar"
        ),
        "metric": (
            "same reference force/torque scaling as deterministic SG second-moment objective"
        ),
        "sample_count": int(count),
        "objective_jacobian_singular_values": singular,
        "objective_jacobian_numerical_rank": rank,
        "objective_jacobian_numerical_tolerance": tolerance,
        "absorbability": {
            "total_second_moment_energy": total_energy / count,
            "remaining_second_moment_energy_after_best_local_parameter_correction": (
                remaining_energy / count
            ),
            "absorbable_fraction": absorbed_fraction,
            "irreducible_fraction": 1.0 - absorbed_fraction,
            "per_wrench_component_absorbable_fraction": per_component_fraction,
            "best_fit_parameter_correction_dimensionless": (
                best_fit_delta_dimensionless
            ),
            "best_fit_parameter_correction_raw_coordinate": best_fit_delta_raw,
            "best_fit_physical_parameter_correction": (
                physical_best_fit_correction
            ),
            "absorbed_wrench_dimensionless": absorbed_series,
            "remaining_wrench_dimensionless": remaining_series,
            "absorbed_wrench_raw": absorbed_raw,
            "remaining_wrench_raw": remaining_raw,
            "remaining_wrench_raw_mean": np.mean(remaining_raw, axis=0),
            "remaining_wrench_raw_rms": np.sqrt(
                np.mean(remaining_raw * remaining_raw, axis=0)
            ),
        },
        "residual_wrench_second_moment": {
            "raw": second_moment_raw,
            "dimensionless": second_moment_dimensionless,
            "remaining_dimensionless_after_best_local_parameter_correction": (
                remaining_second_moment_dimensionless
            ),
        },
        "residual_implied_parameter_error": {
            "assumption": (
                "current iid residual-wrench model with empirical nonzero mean and covariance"
            ),
            "mean_induced_bias_dimensionless": mean_induced_bias_dimensionless,
            "covariance_dimensionless": parameter_covariance_dimensionless,
            "second_moment_dimensionless": parameter_second_moment_dimensionless,
            "std_dimensionless": np.sqrt(
                np.maximum(0.0, np.diag(parameter_covariance_dimensionless))
            ),
            "mean_induced_bias_raw_coordinate": mean_induced_bias_raw,
            "covariance_raw_coordinate": parameter_covariance_raw,
            "second_moment_raw_coordinate": parameter_second_moment_raw,
            "std_raw_coordinate": np.sqrt(
                np.maximum(0.0, np.diag(parameter_covariance_raw))
            ),
            "physical_vector_order": deterministic.PHYSICAL_VALUE_NAMES,
            "physical_mean_induced_bias": physical_mean_induced_bias,
            "physical_covariance": physical_covariance,
            "physical_second_moment": physical_second_moment,
            "physical_std": np.sqrt(
                np.maximum(0.0, np.diag(physical_covariance))
            ),
        },
    }


def _matrix_lines(name: str, matrix: np.ndarray) -> list[str]:
    value = np.asarray(matrix, dtype=float)
    lines = [name]
    for row in value:
        lines.append(
            "  [" + ", ".join("{: .8g}".format(float(item)) for item in row) + "]"
        )
    return lines


def _residual_parameter_diagnostic_lines(
    diagnostics: Mapping[str, Any],
) -> list[str]:
    absorb = diagnostics["absorbability"]
    error = diagnostics["residual_implied_parameter_error"]
    lines = [
        "Residual-wrench / parameter diagnostic",
        "",
        "Linearized model: {}".format(diagnostics["linearized_model"]),
        "Metric: {}".format(diagnostics["metric"]),
        "Sample count: {}".format(diagnostics["sample_count"]),
        "Objective Jacobian rank: {}".format(
            diagnostics["objective_jacobian_numerical_rank"]
        ),
        "Absorbable residual second-moment fraction: {:.12g}".format(
            absorb["absorbable_fraction"]
        ),
        "Irreducible residual second-moment fraction: {:.12g}".format(
            absorb["irreducible_fraction"]
        ),
        "Per-component absorbable fractions: {}".format(
            absorb["per_wrench_component_absorbable_fraction"]
        ),
        "Remaining raw wrench mean [Fx,Fy,Fz,Mx,My,Mz]: {}".format(
            np.asarray(absorb["remaining_wrench_raw_mean"])
        ),
        "Remaining raw wrench RMS [Fx,Fy,Fz,Mx,My,Mz]: {}".format(
            np.asarray(absorb["remaining_wrench_raw_rms"])
        ),
        "",
        "Best local correction for the realized residual",
    ]
    for name, value in zip(
        deterministic.PHYSICAL_PARAMETER_NAMES,
        absorb["best_fit_parameter_correction_raw_coordinate"],
    ):
        lines.append("  {:52s} {:+.10g}".format(name, float(value)))
    lines.extend(["", "Mean-induced parameter bias under current iid residual model"])
    for name, value, std in zip(
        deterministic.PHYSICAL_PARAMETER_NAMES,
        error["mean_induced_bias_raw_coordinate"],
        error["std_raw_coordinate"],
    ):
        lines.append(
            "  {:52s} bias={:+.10g}  std={:.10g}".format(
                name, float(value), float(std)
            )
        )
    lines.extend(["", "Physical parameter bias/std"])
    for name, value, std in zip(
        deterministic.PHYSICAL_VALUE_NAMES,
        error["physical_mean_induced_bias"],
        error["physical_std"],
    ):
        lines.append(
            "  {:36s} bias={:+.10g}  std={:.10g}".format(
                name, float(value), float(std)
            )
        )
    lines.extend([""])
    lines.extend(
        _matrix_lines(
            "Parameter covariance in raw 14-D coordinate",
            error["covariance_raw_coordinate"],
        )
    )
    lines.extend([""])
    lines.extend(
        _matrix_lines(
            "Parameter second moment in raw 14-D coordinate",
            error["second_moment_raw_coordinate"],
        )
    )
    lines.extend([""])
    lines.extend(
        _matrix_lines(
            "Physical parameter covariance",
            error["physical_covariance"],
        )
    )
    lines.extend([""])
    lines.extend(
        _matrix_lines(
            "Physical parameter second moment",
            error["physical_second_moment"],
        )
    )
    return lines


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

    confidence_time = np.asarray(bag.collocation_time, dtype=float)
    wrench_raw = wrench_raw_full
    wrench_jacobian_raw = wrench_jacobian_raw_full
    print(
        "confidence likelihood uses {} residual-wrench samples (all valid SG centers)"
        .format(confidence_time.size),
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
    posterior["numerical_linear_algebra"] = {
        "precision_inverse": "Moore-Penrose pseudoinverse",
        "posterior_mean_solver": "pseudoinverse times information vector",
        "reason": (
            "data/prior information may be rank deficient because ridge directions "
            "are part of the intended model; ordinary matrix inversion is not used"
        ),
    }
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
    residual_parameter_diagnostics = _residual_parameter_diagnostics(
        wrench_raw=wrench_raw,
        wrench_dimensionless=wrench_dimensionless,
        jacobian_dimensionless=jacobian_dimensionless,
        wrench_covariance_dimensionless=wrench_covariance_dimensionless,
        wrench_scale=wrench_scale,
        raw_per_dimensionless=raw_per_dimensionless,
        selected=selected,
        reference_parameters=vehicle_model.parameters,
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
            "raw_pose_sample_count": int(np.asarray(flight.pose.times).size),
            "valid_center_count": int(confidence_time.size),
            "residual_wrench_sample_count": int(wrench_raw.shape[0]),
            "likelihood_center_times_seconds": confidence_time,
            "selection_rule": "all valid centered SG evaluation times",
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
            "second_moment_dimensionless": (
                residual_parameter_diagnostics["residual_wrench_second_moment"][
                    "dimensionless"
                ]
            ),
            "iid_scope": "all valid centered SG evaluation times",
        },
        "residual_parameter_absorbability": {
            "absorbable_fraction": residual_parameter_diagnostics["absorbability"][
                "absorbable_fraction"
            ],
            "irreducible_fraction": residual_parameter_diagnostics["absorbability"][
                "irreducible_fraction"
            ],
            "best_fit_parameter_correction_raw_coordinate": (
                residual_parameter_diagnostics["absorbability"][
                    "best_fit_parameter_correction_raw_coordinate"
                ]
            ),
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
        "residual_implied_parameter_error": (
            residual_parameter_diagnostics["residual_implied_parameter_error"]
        ),
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
            "raw_pose_sample_count": int(np.asarray(flight.pose.times).size),
            "valid_center_count": int(confidence_time.size),
            "residual_wrench_sample_count": int(wrench_raw.shape[0]),
            "confidence_center_time_seconds": confidence_time,
            "window_seconds": float(arguments.window_seconds),
            "polynomial_degree": 5,
        },
        "assumptions": {
            "pose_front_end": (
                "raw timestamped pose; degree-5 local polynomial in R3 and "
                "geometric Savitzky-Golay on SO3"
            ),
            "residual_wrench_sampling": (
                "every valid centered SG evaluation time is used; there is no "
                "confidence-specific temporal subsampling"
            ),
            "external_wrench": (
                "all residual-wrench samples are treated as iid Gaussian with "
                "nonzero empirical mean and 6x6 empirical covariance"
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
            "sample_count": int(wrench_raw.shape[0]),
            "sample_time_seconds": confidence_time,
        },
        "residual_parameter_diagnostics": residual_parameter_diagnostics,
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
    diagnostic_lines = _residual_parameter_diagnostic_lines(
        residual_parameter_diagnostics
    )
    deterministic.strict._write_text(
        output_directory / "residual_parameter_diagnostics.txt",
        diagnostic_lines,
    )
    deterministic._write_parameters_pdf(
        output_directory / "residual_parameter_diagnostics.pdf",
        diagnostic_lines,
    )
    payload["outputs"] = {
        "residual_parameter_diagnostics_txt": "residual_parameter_diagnostics.txt",
        "residual_parameter_diagnostics_pdf": "residual_parameter_diagnostics.pdf",
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
    print(
        "residual absorbability: {:.3%} locally absorbable by the 14-D parameter chart".format(
            residual_parameter_diagnostics["absorbability"]["absorbable_fraction"]
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
