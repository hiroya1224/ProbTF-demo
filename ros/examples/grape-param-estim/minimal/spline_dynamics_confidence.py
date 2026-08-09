#!/usr/bin/env python3
"""Local confidence/ridge analysis for deterministic spline dynamics.

This script intentionally starts from one deterministic pose-spline estimate
and analyzes *one bag*.  It does not use the Monte Carlo pose perturbation
layer.

Model used by this first confidence layer
-----------------------------------------
1. The deterministic spline-dynamics estimator provides a MAP-like physical
   parameter point, spline-implied residual body wrench, and its analytic
   parameter Jacobian.
2. Rigid-body quantities are nondimensionalized by fixed vehicle-model reference
   scales:
       M* = vehicle-model reference mass
       L* = sqrt(trace(J*) / (3 M*))
       T* = sqrt(L* / g)
   so dimensionless gravity has magnitude one.
3. At the deterministic point, the dimensionless residual wrench sequence is
   summarized by its empirical nonzero mean and sample covariance:
       W_k iid~ N(mu_w, Sigma_w).
   Temporal correlation is deliberately ignored in this first model.
4. The parameter-to-wrench Jacobian is whitened with Sigma_w and SVD is used
   to expose data-constrained and ridge directions without hand-designed
   drone-specific ridge coordinates.
5. The data information matrix is kept separate from the externally supplied
   Gaussian physical-parameter prior. The posterior precision is their sum.
6. Prefix information spectra are stored without dividing by sample count, so
   longer bags accumulate information under the iid approximation.

The deterministic estimator must use the same explicit vehicle-model and
Gaussian-prior JSON inputs; no embedded nominal model is used.
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

import deterministic_spline_dynamics_estimator as deterministic
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.system import GRAVITY


SCHEMA = "grape-param-estim/spline-dynamics-confidence/v2"
OUTPUT_SUBDIRECTORY = "spline_dynamics_confidence"


def _sanitize(value: Any) -> Any:
    return deterministic._json_sanitize(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _only_bag(config: Any):
    bags = tuple(config.multi_bag.bags)
    if len(bags) != 1:
        raise SystemExit(
            "confidence analysis requires exactly one bag in --config; found {}"
            .format(len(bags))
        )
    return bags[0]


def _parameter_payload(parameters: Any) -> dict[str, Any]:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return {
        "mass_kg": float(parameters.mass),
        "inertia_kg_m2": inertia,
        "inertia_principal_moments_kg_m2": np.linalg.eigvalsh(inertia),
        "cog_position_body_m": np.asarray(parameters.cog_offset, dtype=float),
        "force_effectiveness": np.asarray(
            parameters.force_effectiveness, dtype=float
        ),
        "torque_effectiveness": np.asarray(
            parameters.torque_effectiveness, dtype=float
        ),
    }


def _estimate_solution(
    bag: Any,
    arguments: argparse.Namespace,
    initial_delay: float,
    reference_parameters: Any,
    parameter_prior: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run the same smooth-lag -> strict-ZOH physical solve as the estimator."""

    problem = deterministic.SplineDynamicsProblem(
        (bag,),
        reference_parameters,
        parameter_prior,
    )
    initial_physical = np.zeros(
        deterministic.PHYSICAL_DIMENSION,
        dtype=float,
    )
    physical_lower, physical_upper = deterministic._physical_bounds(
        initial_physical,
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
        (
            initial_physical,
            np.asarray((initial_delay,), dtype=float),
        )
    )
    coordinate = np.minimum(
        np.maximum(coordinate, smooth_lower),
        smooth_upper,
    )

    smooth_history: list[dict[str, Any]] = []
    for stage_index, width in enumerate(arguments.smoothstep_width_fractions):
        print(
            "smoothstep {}/{}: width_fraction={:.6g}".format(
                stage_index + 1,
                len(arguments.smoothstep_width_fractions),
                width,
            ),
            flush=True,
        )
        coordinate, evaluation, optimizer = deterministic._solve_smooth(
            problem,
            coordinate,
            float(width),
            smooth_lower,
            smooth_upper,
            arguments,
        )
        cost = 0.5 * float(evaluation.residual @ evaluation.residual)
        print(
            "  cost={:.9g}, delay={:.6f}s, nfev={}".format(
                cost,
                coordinate[deterministic.DELAY_INDEX],
                optimizer["nfev"],
            ),
            flush=True,
        )
        smooth_history.append(
            {
                "stage_index": stage_index,
                "width_fraction": float(width),
                "objective_cost": cost,
                "physical_coordinate": coordinate[
                    : deterministic.PHYSICAL_DIMENSION
                ],
                "delay_seconds": float(
                    coordinate[deterministic.DELAY_INDEX]
                ),
                "optimizer": optimizer,
            }
        )

    smooth_physical = coordinate[
        : deterministic.PHYSICAL_DIMENSION
    ].copy()
    smooth_delay = float(coordinate[deterministic.DELAY_INDEX])
    candidate_delays = deterministic.smooth.zoh_polish_delays(
        smooth_delay,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.delay_bounds,
    )
    screening_costs = []
    for delay in candidate_delays:
        evaluation = problem.evaluate_strict(
            smooth_physical,
            float(delay),
        )
        screening_costs.append(
            0.5 * float(evaluation.residual @ evaluation.residual)
        )
    screening_costs = np.asarray(screening_costs, dtype=float)

    top_count = min(
        arguments.zoh_polish_top_k,
        candidate_delays.size,
    )
    top_indices = np.argsort(
        screening_costs,
        kind="stable",
    )[:top_count]
    strict_solutions = []
    strict_history = []
    for rank, candidate_index in enumerate(top_indices):
        delay = float(candidate_delays[candidate_index])
        print(
            "strict-ZOH {}/{}: delay={:.6f}s".format(
                rank + 1,
                top_count,
                delay,
            ),
            flush=True,
        )
        solution = deterministic._solve_strict(
            problem,
            smooth_physical,
            delay,
            physical_lower,
            physical_upper,
            arguments,
        )
        strict_solutions.append(solution)
        strict_history.append(
            {
                "rank": rank,
                "delay_seconds": delay,
                "screening_cost": float(
                    screening_costs[candidate_index]
                ),
                "refined_cost": float(
                    deterministic._solution_cost(solution)
                ),
                "physical_coordinate": solution.physical_coordinate,
                "optimizer": solution.optimizer,
            }
        )

    if not strict_solutions:
        raise RuntimeError("strict-ZOH polish produced no solution")
    selected = min(
        strict_solutions,
        key=deterministic._solution_cost,
    )
    return selected, {
        "smooth_history": smooth_history,
        "candidate_delays_seconds": candidate_delays,
        "screening_costs": screening_costs,
        "strict_history": strict_history,
    }


def _reference_scales(
    reference_parameters: Any,
) -> dict[str, float]:
    mass_scale = float(reference_parameters.mass)
    length_scale = math.sqrt(
        float(np.trace(reference_parameters.inertia))
        / (3.0 * mass_scale)
    )
    if not (
        np.isfinite(mass_scale)
        and mass_scale > 0.0
        and np.isfinite(length_scale)
        and length_scale > 0.0
        and np.isfinite(GRAVITY)
        and GRAVITY > 0.0
    ):
        raise ValueError("vehicle-model nondimensionalization scales are invalid")
    time_scale = math.sqrt(length_scale / float(GRAVITY))
    force_scale = mass_scale * length_scale / time_scale**2
    torque_scale = (
        mass_scale * length_scale**2 / time_scale**2
    )
    return {
        "mass_kg": mass_scale,
        "length_m": length_scale,
        "time_s": time_scale,
        "force_N": force_scale,
        "torque_Nm": torque_scale,
        "dimensionless_gravity": (
            float(GRAVITY) * time_scale**2 / length_scale
        ),
    }


def _parameter_raw_per_dimensionless(
    length_scale: float,
) -> np.ndarray:
    """q_raw = S q_bar for the estimator's 14 physical coordinates."""

    scale = np.ones(
        deterministic.PHYSICAL_DIMENSION,
        dtype=float,
    )
    # mass/inertia log coordinates and effectiveness log coordinates are
    # already dimensionless.  CoG offsets are lengths.
    scale[7:10] = float(length_scale)
    return scale


def _wrench_dimensionless_scale(
    force_scale: float,
    torque_scale: float,
) -> np.ndarray:
    return np.asarray(
        (
            1.0 / force_scale,
            1.0 / force_scale,
            1.0 / force_scale,
            1.0 / torque_scale,
            1.0 / torque_scale,
            1.0 / torque_scale,
        ),
        dtype=float,
    )


def _pseudo_whitener(
    covariance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(
        0.0,
        float(np.max(eigenvalues)),
    )
    tolerance = (
        max(covariance.shape)
        * np.finfo(float).eps
        * max(1.0, largest)
    )
    positive = eigenvalues > tolerance
    inverse_sqrt = np.zeros_like(eigenvalues)
    inverse_sqrt[positive] = 1.0 / np.sqrt(
        eigenvalues[positive]
    )
    whitener = (
        eigenvectors
        @ np.diag(inverse_sqrt)
        @ eigenvectors.T
    )
    return whitener, {
        "eigenvalues": eigenvalues,
        "numerical_rank": int(np.count_nonzero(positive)),
        "numerical_tolerance": tolerance,
        "uses_pseudoinverse_for_singular_directions": True,
    }


def _svd_payload(
    stacked_jacobian: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    """Data-only ridge analysis.

    ``relative_information_strength`` has the intentionally simple meaning

        1   : as strongly constrained as the best local parameter direction
        ~0  : essentially a ridge / weakly identifiable direction.

    No prior enters this SVD.
    """

    matrix = np.asarray(stacked_jacobian, dtype=float)
    _u, singular_values, vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    information = singular_values * singular_values
    strongest = float(information[0]) if information.size else 0.0
    relative_information = (
        information / strongest
        if strongest > 0.0
        else np.zeros_like(information)
    )

    if singular_values.size:
        tolerance = (
            max(matrix.shape)
            * np.finfo(float).eps
            * float(singular_values[0])
        )
    else:
        tolerance = 0.0
    rank = int(np.count_nonzero(singular_values > tolerance))

    directions = []
    for index in range(vt.shape[0] - 1, -1, -1):
        # Keep the signed right-singular vector exactly as returned by SVD.
        # The complete vector has the standard global +/- ambiguity, while
        # relative signs between components are meaningful.
        vector = np.asarray(vt[index], dtype=float)
        order = np.argsort(
            np.abs(vector),
            kind="stable",
        )[::-1]
        directions.append(
            {
                "singular_index": int(index),
                "singular_value": float(singular_values[index]),
                "information_eigenvalue": float(information[index]),
                "relative_information_strength": float(
                    relative_information[index]
                ),
                "dimensionless_direction": vector,
                "interpretation": (
                    "Signed right-singular vector returned by SVD. The whole "
                    "vector may be multiplied by -1 without changing the mode; "
                    "relative signs between components are meaningful."
                ),
                "dominant_components": [
                    {
                        "name": str(names[column]),
                        "coefficient": float(vector[column]),
                        "absolute_coefficient": float(abs(vector[column])),
                    }
                    for column in order[: min(8, len(order))]
                ],
            }
        )

    weakest_relative = (
        float(relative_information[-1])
        if relative_information.size
        else 0.0
    )
    return {
        "singular_values_descending": singular_values,
        "information_eigenvalues_descending": information,
        "relative_information_strength_descending": relative_information,
        "strongest_direction_is_one": True,
        "weakest_relative_information_strength": weakest_relative,
        "numerical_rank": rank,
        "nullity": int(matrix.shape[1] - rank),
        "numerical_tolerance": tolerance,
        # Smallest singular direction first.
        "weak_directions": directions,
    }



def _symmetric_covariance_sqrt(
    covariance: np.ndarray,
) -> np.ndarray:
    value = np.asarray(covariance, dtype=float)
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    tolerance = (
        value.shape[0]
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if np.any(eigenvalues < -tolerance):
        raise ValueError("covariance is not positive semidefinite")
    return (
        eigenvectors
        @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        @ eigenvectors.T
    )


def _prefix_information(
    time_axis: np.ndarray,
    sample_jacobian: np.ndarray,
    prior_covariance_dimensionless: np.ndarray,
) -> dict[str, Any]:
    """Accumulate data information and compare it with the supplied prior."""

    time_value = np.asarray(time_axis, dtype=float)
    jacobian = np.asarray(sample_jacobian, dtype=float)
    prior_covariance = np.asarray(
        prior_covariance_dimensionless,
        dtype=float,
    )
    dimension = jacobian.shape[2]
    if prior_covariance.shape != (dimension, dimension):
        raise ValueError("prefix prior covariance has the wrong dimension")
    prior_covariance_sqrt = _symmetric_covariance_sqrt(
        prior_covariance
    )

    cumulative = np.zeros((dimension, dimension), dtype=float)
    eigenvalue_history = np.empty(
        (time_value.size, dimension), dtype=float
    )
    relative_history = np.empty(
        (time_value.size, dimension), dtype=float
    )
    generalized_information_history = np.empty(
        (time_value.size, dimension), dtype=float
    )
    remaining_history = np.empty(
        (time_value.size, dimension), dtype=float
    )
    numerical_rank_history = np.empty(time_value.size, dtype=int)

    for index in range(time_value.size):
        block = jacobian[index]
        cumulative += block.T @ block
        symmetric = 0.5 * (cumulative + cumulative.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]
        eigenvalue_history[index] = eigenvalues

        largest = max(0.0, float(eigenvalues[0]))
        relative_history[index] = (
            eigenvalues / largest
            if largest > 0.0
            else np.zeros(dimension, dtype=float)
        )
        tolerance = (
            dimension
            * np.finfo(float).eps
            * max(1.0, largest)
        )
        numerical_rank_history[index] = int(
            np.count_nonzero(eigenvalues > tolerance)
        )

        prior_whitened = (
            prior_covariance_sqrt
            @ symmetric
            @ prior_covariance_sqrt
        )
        generalized = np.linalg.eigvalsh(
            0.5 * (prior_whitened + prior_whitened.T)
        )[::-1]
        generalized = np.maximum(generalized, 0.0)
        generalized_information_history[index] = generalized
        remaining_history[index] = 1.0 / np.sqrt(1.0 + generalized)

    return {
        "time_seconds_from_analysis_start": time_value - time_value[0],
        "information_eigenvalues_descending": eigenvalue_history,
        "relative_information_strength_descending": relative_history,
        "numerical_rank": numerical_rank_history,
        "prior_normalized_data_information_descending": (
            generalized_information_history
        ),
        "remaining_std_fraction_by_mode": remaining_history,
        "remaining_std_fraction_strongest_mode": remaining_history[:, 0],
        "remaining_std_fraction_median_mode": np.median(
            remaining_history,
            axis=1,
        ),
        "remaining_std_fraction_weakest_mode": remaining_history[:, -1],
        "remaining_std_fraction_interpretation": (
            "1 means the bag did not reduce prior uncertainty; "
            "0 means the data dominate the prior along that generalized mode."
        ),
    }



def _physical_distribution_payload(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    z95 = 1.959963984540054
    lower = mean - z95 * std
    upper = mean + z95 * std
    return {
        "vector_order": deterministic.PHYSICAL_VALUE_NAMES,
        "mean": mean,
        "covariance": covariance,
        "std": std,
        "interval_95": np.column_stack((lower, upper)),
    }


def _posterior_payload(
    *,
    data_information: np.ndarray,
    data_information_vector: np.ndarray,
    selected: Any,
    raw_per_dimensionless: np.ndarray,
    reference_parameters: Any,
    parameter_prior: Any,
    names: Sequence[str],
) -> dict[str, Any]:
    """Fuse the local data likelihood with the external physical Gaussian prior."""

    data_information = np.asarray(data_information, dtype=float)
    data_information_vector = np.asarray(
        data_information_vector,
        dtype=float,
    )
    raw_scale = np.asarray(raw_per_dimensionless, dtype=float)

    parameterization = deterministic.SplinePhysicalParameterization(
        reference_parameters
    )
    decoded, jacobian_raw = parameterization.decode_with_jacobian(
        selected.physical_coordinate,
        selected.delay_seconds,
    )
    physical_reference = deterministic.physical_parameter_vector(
        decoded.parameters
    )
    physical_jacobian_raw = deterministic.physical_parameter_jacobian(
        jacobian_raw
    )
    physical_jacobian_dimensionless = (
        physical_jacobian_raw * raw_scale[None, :]
    )

    inverse_variance = 1.0 / (parameter_prior.std**2)
    prior_precision = (
        physical_jacobian_dimensionless.T
        @ (
            inverse_variance[:, None]
            * physical_jacobian_dimensionless
        )
    )
    prior_information_vector = (
        physical_jacobian_dimensionless.T
        @ (
            inverse_variance
            * (parameter_prior.mean - physical_reference)
        )
    )
    prior_precision = 0.5 * (
        prior_precision + prior_precision.T
    )
    prior_covariance = np.linalg.pinv(prior_precision, hermitian=True)
    prior_covariance = 0.5 * (
        prior_covariance + prior_covariance.T
    )

    posterior_precision = data_information + prior_precision
    posterior_precision = 0.5 * (
        posterior_precision + posterior_precision.T
    )
    posterior_information_vector = (
        data_information_vector + prior_information_vector
    )
    posterior_covariance = np.linalg.pinv(posterior_precision, hermitian=True)
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    posterior_delta_mean = (
        posterior_covariance @ posterior_information_vector
    )

    selected_dimensionless = (
        selected.physical_coordinate / raw_scale
    )
    posterior_mean_dimensionless = (
        selected_dimensionless + posterior_delta_mean
    )
    posterior_physical_mean = (
        physical_reference
        + physical_jacobian_dimensionless @ posterior_delta_mean
    )
    posterior_physical_covariance = (
        physical_jacobian_dimensionless
        @ posterior_covariance
        @ physical_jacobian_dimensionless.T
    )
    posterior_physical_covariance = 0.5 * (
        posterior_physical_covariance
        + posterior_physical_covariance.T
    )

    prior_covariance_sqrt = _symmetric_covariance_sqrt(
        prior_covariance
    )
    prior_whitened_information = (
        prior_covariance_sqrt
        @ data_information
        @ prior_covariance_sqrt
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5
        * (
            prior_whitened_information
            + prior_whitened_information.T
        )
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    remaining_std = 1.0 / np.sqrt(1.0 + eigenvalues)

    generalized_modes = []
    for rank_from_weakest, index in enumerate(
        range(eigenvalues.size - 1, -1, -1)
    ):
        direction = (
            prior_covariance_sqrt @ eigenvectors[:, index]
        )
        norm = np.linalg.norm(direction)
        if norm > 0.0:
            direction = direction / norm
        component_order = np.argsort(
            np.abs(direction),
            kind="stable",
        )[::-1]
        generalized_modes.append(
            {
                "rank_from_weakest": rank_from_weakest + 1,
                "data_to_prior_precision_ratio": float(
                    eigenvalues[index]
                ),
                "remaining_prior_std_fraction": float(
                    remaining_std[index]
                ),
                "dimensionless_parameter_direction": direction,
                "dominant_components": [
                    {
                        "name": str(names[column]),
                        "coefficient": float(direction[column]),
                    }
                    for column in component_order[
                        : min(8, len(component_order))
                    ]
                ],
            }
        )

    return {
        "selected_coordinate_raw": selected.physical_coordinate,
        "selected_coordinate_dimensionless": selected_dimensionless,
        "data_information_dimensionless": data_information,
        "data_information_vector_dimensionless": (
            data_information_vector
        ),
        "prior_precision_dimensionless": prior_precision,
        "prior_covariance_dimensionless": prior_covariance,
        "prior_information_vector_dimensionless": (
            prior_information_vector
        ),
        "posterior_precision_dimensionless": posterior_precision,
        "posterior_covariance_dimensionless": posterior_covariance,
        "posterior_information_vector_dimensionless": (
            posterior_information_vector
        ),
        "posterior_delta_mean_dimensionless": posterior_delta_mean,
        "posterior_mean_dimensionless": posterior_mean_dimensionless,
        "physical_linearization_point": physical_reference,
        "physical_jacobian_dimensionless": (
            physical_jacobian_dimensionless
        ),
        "prior_physical": _physical_distribution_payload(
            parameter_prior.mean,
            np.diag(parameter_prior.std**2),
        ),
        "posterior_physical": _physical_distribution_payload(
            posterior_physical_mean,
            posterior_physical_covariance,
        ),
        "prior_normalized_confidence": {
            "data_to_prior_precision_ratio_descending": eigenvalues,
            "remaining_prior_std_fraction_descending_modes": (
                remaining_std
            ),
            "weak_modes": generalized_modes,
            "interpretation": (
                "The data-only information is compared against the supplied "
                "Gaussian prior. Ridge information itself remains in data SVD."
            ),
        },
    }



def _trajectory_reconstruction(
    bag: Any,
    selected: Any,
    dynamics_evaluation: Any,
    arguments: argparse.Namespace,
    initial_delay: float,
    reference_parameters: Any,
) -> dict[str, Any]:
    """Validate that the 14-D parameterization still reproduces the trajectory.

    Three rollouts are kept distinct:
    - reference: reference physical parameters;
    - parameter_only: selected 14-D parameters with no external wrench;
    - external_wrench_replay: selected 14-D parameters held fixed while the
      existing external-wrench replay is refined.

    The last one answers whether the complete deterministic reconstruction
    pipeline remains intact after adding the fourth rotor-effectiveness degree
    of freedom.
    """

    observations = bag.direct_problem.observations

    reference_rollout = deterministic.forward_rollout(
        bag,
        np.zeros(
            deterministic.PHYSICAL_DIMENSION,
            dtype=float,
        ),
        initial_delay,
        reference_parameters,
    )
    parameter_rollout = deterministic.forward_rollout(
        bag,
        selected.physical_coordinate,
        selected.delay_seconds,
        reference_parameters,
    )

    print(
        "validating 14-D trajectory reconstruction with external-wrench replay",
        flush=True,
    )
    (
        wrench_problem,
        wrench_evaluation,
        wrench_optimizer,
    ) = deterministic._solve_wrench_replay(
        bag,
        selected.physical_coordinate,
        selected.delay_seconds,
        dynamics_evaluation,
        arguments,
        reference_parameters,
    )
    replay_rollout = wrench_evaluation.simulation
    replay_observations = deterministic._observations_at_times(
        observations,
        replay_rollout.time,
    )

    reference_metrics = deterministic._pose_metrics(
        observations,
        reference_rollout,
    )
    parameter_metrics = deterministic._pose_metrics(
        observations,
        parameter_rollout,
    )
    replay_metrics = deterministic._pose_metrics(
        replay_observations,
        replay_rollout,
    )

    print(
        "  pose RMSE: reference {:.6g} m / {:.6g} deg; "
        "14-D parameter-only {:.6g} m / {:.6g} deg; "
        "14-D + replay {:.6g} m / {:.6g} deg".format(
            reference_metrics["position_rmse_m"],
            reference_metrics["orientation_angle_rmse_deg"],
            parameter_metrics["position_rmse_m"],
            parameter_metrics["orientation_angle_rmse_deg"],
            replay_metrics["position_rmse_m"],
            replay_metrics["orientation_angle_rmse_deg"],
        ),
        flush=True,
    )

    selected_parameter_payload = _parameter_payload(
        selected.evaluation.decoded.parameters
    )
    selected_parameter_payload["delay_seconds"] = float(
        selected.delay_seconds
    )

    return {
        "selected_physical_parameters": selected_parameter_payload,
        "meaning": {
            "reference": (
                "Vehicle-model reference parameters and initial lag; no external wrench."
            ),
            "parameter_only": (
                "Selected 14-D physical parameters and selected lag; "
                "no external wrench."
            ),
            "external_wrench_replay": (
                "Selected 14-D physical parameters and lag are fixed; only the "
                "piecewise-linear external wrench is refined. This is the direct "
                "check that the pre-existing reconstruction pipeline still works "
                "after adding the fourth rotor-effectiveness degree of freedom."
            ),
        },
        "reference": {
            "metrics": reference_metrics,
            "time": reference_rollout.time,
            "sensor_position": reference_rollout.sensor_position,
            "sensor_orientation_xyzw": (
                reference_rollout.sensor_orientation_xyzw
            ),
        },
        "parameter_only": {
            "metrics": parameter_metrics,
            "time": parameter_rollout.time,
            "sensor_position": parameter_rollout.sensor_position,
            "sensor_orientation_xyzw": (
                parameter_rollout.sensor_orientation_xyzw
            ),
        },
        "observed": {
            "time": observations.time,
            "sensor_position": observations.sensor_position,
            "sensor_orientation_xyzw": (
                observations.sensor_orientation_xyzw
            ),
        },
        "external_wrench_replay": {
            "metrics": replay_metrics,
            "time": replay_rollout.time,
            "observed_position_on_replay_support": (
                replay_observations.sensor_position
            ),
            "observed_orientation_xyzw_on_replay_support": (
                replay_observations.sensor_orientation_xyzw
            ),
            "sensor_position": replay_rollout.sensor_position,
            "sensor_orientation_xyzw": (
                replay_rollout.sensor_orientation_xyzw
            ),
            "optimizer": wrench_optimizer,
            "knot_time": wrench_evaluation.knot_time,
            "body_wrench_coefficients": (
                wrench_evaluation.coefficients
            ),
            "initial_body_wrench_coefficients": (
                wrench_problem.initial_coefficients
            ),
        },
    }


def _angular_excitation_payload(bag: Any) -> dict[str, Any]:
    omega = np.asarray(
        bag.collocation.body_angular_velocity,
        dtype=float,
    )
    alpha = np.asarray(
        bag.collocation.body_angular_acceleration,
        dtype=float,
    )
    if omega.ndim != 2 or omega.shape[1] != 3:
        raise ValueError("angular velocity has unexpected shape")
    if alpha.shape != omega.shape:
        raise ValueError("angular acceleration has unexpected shape")
    return {
        "axis_names": ("body_x", "body_y", "body_z"),
        "angular_velocity_rms_rad_per_s": np.sqrt(
            np.mean(omega * omega, axis=0)
        ),
        "angular_acceleration_rms_rad_per_s2": np.sqrt(
            np.mean(alpha * alpha, axis=0)
        ),
        "angular_velocity_peak_rad_per_s": np.max(
            np.abs(omega), axis=0
        ),
        "angular_acceleration_peak_rad_per_s2": np.max(
            np.abs(alpha), axis=0
        ),
        "interpretation": (
            "These are only intuitive excitation diagnostics. "
            "Actual parameter identifiability is determined by the full "
            "wrench Jacobian/SVD, including Euler cross-coupling terms."
        ),
    }



def _continuous_rpy_degrees(
    orientation_xyzw: np.ndarray,
) -> np.ndarray:
    """Return continuous roll/pitch/yaw curves for plotting only.

    scipy's lower-case ``xyz`` convention is the usual extrinsic XYZ
    representation, equivalent to the common intrinsic ZYX yaw-pitch-roll
    decomposition. Euler angles are used only for human-readable diagnostics;
    orientation errors remain computed on SO(3).
    """

    from scipy.spatial.transform import Rotation as SciPyRotation

    quaternion = np.asarray(orientation_xyzw, dtype=float)
    rpy_rad = SciPyRotation.from_quat(quaternion).as_euler(
        "xyz",
        degrees=False,
    )
    return np.degrees(np.unwrap(rpy_rad, axis=0))



def _write_pdf(
    path: Path,
    *,
    time_axis: np.ndarray,
    wrench_dimensionless: np.ndarray,
    wrench_mean: np.ndarray,
    wrench_covariance: np.ndarray,
    svd: Mapping[str, Any],
    prefix: Mapping[str, Any],
    posterior: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    angular_excitation: Mapping[str, Any],
    names: Sequence[str],
) -> None:
    """Write only plots whose numerical meaning is explicit in the labels."""

    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    relative_time = np.asarray(
        time_axis,
        dtype=float,
    ) - float(time_axis[0])
    wrench_std = np.sqrt(
        np.maximum(
            0.0,
            np.diag(
                np.asarray(
                    wrench_covariance,
                    dtype=float,
                )
            ),
        )
    )

    observed = reconstruction["observed"]
    parameter_only = reconstruction["parameter_only"]
    replay = reconstruction["external_wrench_replay"]

    with PdfPages(path) as pdf:
        # ------------------------------------------------------------------
        # 1. Fine reconstruction view.
        #
        # Do NOT overlay reference or parameter-only open-loop rollouts here.
        # They can diverge by tens of metres and would destroy the scale needed
        # to inspect the sub-millimetre replay reconstruction.
        # ------------------------------------------------------------------
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        labels = ("x [m]", "y [m]", "z [m]")
        replay_observed_time = np.asarray(
            replay["time"],
            dtype=float,
        )
        replay_observed_position = np.asarray(
            replay["observed_position_on_replay_support"],
            dtype=float,
        )
        replay_position = np.asarray(
            replay["sensor_position"],
            dtype=float,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                replay_observed_time,
                replay_observed_position[:, component],
                label="observed",
            )
            axis.plot(
                replay_observed_time,
                replay_position[:, component],
                label="14-D parameters + external-wrench replay",
            )
            axis.set_ylabel(labels[component])
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Fine pose reproduction: position"
        )
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("record-local time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        replay_observed_orientation = np.asarray(
            replay["observed_orientation_xyzw_on_replay_support"],
            dtype=float,
        )
        replay_orientation = np.asarray(
            replay["sensor_orientation_xyzw"],
            dtype=float,
        )
        observed_rpy_deg = _continuous_rpy_degrees(
            replay_observed_orientation
        )
        replay_rpy_deg = _continuous_rpy_degrees(
            replay_orientation
        )
        # Align the independently unwrapped plotting branches at the first
        # sample. This changes only the displayed Euler branch, not SO(3).
        for component in range(3):
            replay_rpy_deg[:, component] += (
                360.0
                * np.round(
                    (
                        observed_rpy_deg[0, component]
                        - replay_rpy_deg[0, component]
                    )
                    / 360.0
                )
            )

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        rpy_labels = (
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        )
        for component, axis in enumerate(axes):
            axis.plot(
                replay_observed_time,
                observed_rpy_deg[:, component],
                label="observed",
            )
            axis.plot(
                replay_observed_time,
                replay_rpy_deg[:, component],
                label="14-D parameters + external-wrench replay",
            )
            axis.set_ylabel(rpy_labels[component])
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Fine pose reproduction: roll / pitch / yaw"
        )
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("record-local time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        replay_position_error = np.linalg.norm(
            replay_position - replay_observed_position,
            axis=1,
        )
        replay_orientation_error = np.degrees(
            np.linalg.norm(
                deterministic._orientation_errors(
                    np.asarray(
                        replay[
                            "observed_orientation_xyzw_on_replay_support"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        replay["sensor_orientation_xyzw"],
                        dtype=float,
                    ),
                ),
                axis=1,
            )
        )

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            replay_observed_time,
            replay_position_error,
        )
        axes[0].set_ylabel("position error norm [m]")
        axes[0].set_title(
            "Fine replay error after the 14-D change — lower is better"
        )
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(
            replay_observed_time,
            replay_orientation_error,
        )
        axes[1].set_ylabel("orientation error angle [deg]")
        axes[1].set_xlabel("record-local time [s]")
        axes[1].grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        # Parameter-only open-loop rollout on its own scale, with the observed
        # pose explicitly overlaid.
        observed_time = np.asarray(observed["time"], dtype=float)
        observed_position = np.asarray(
            observed["sensor_position"], dtype=float
        )
        parameter_time = np.asarray(
            parameter_only["time"], dtype=float
        )
        parameter_position = np.asarray(
            parameter_only["sensor_position"], dtype=float
        )

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                observed_time,
                observed_position[:, component],
                label="observed",
            )
            axis.plot(
                parameter_time,
                parameter_position[:, component],
                label="parameter-only open-loop rollout",
            )
            axis.set_ylabel(("x [m]", "y [m]", "z [m]")[component])
            axis.grid(True, alpha=0.25)
        axes[0].set_title("Parameter-only open-loop pose: position")
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("record-local time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        observed_parameter_rpy_deg = _continuous_rpy_degrees(
            np.asarray(
                observed["sensor_orientation_xyzw"], dtype=float
            )
        )
        parameter_rpy_deg = _continuous_rpy_degrees(
            np.asarray(
                parameter_only["sensor_orientation_xyzw"], dtype=float
            )
        )
        for component in range(3):
            parameter_rpy_deg[:, component] += (
                360.0
                * np.round(
                    (
                        observed_parameter_rpy_deg[0, component]
                        - parameter_rpy_deg[0, component]
                    )
                    / 360.0
                )
            )

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                observed_time,
                observed_parameter_rpy_deg[:, component],
                label="observed",
            )
            axis.plot(
                parameter_time,
                parameter_rpy_deg[:, component],
                label="parameter-only open-loop rollout",
            )
            axis.set_ylabel(
                ("roll [deg]", "pitch [deg]", "yaw [deg]")[component]
            )
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Parameter-only open-loop pose: roll / pitch / yaw"
        )
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("record-local time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        parameter_position_error = np.linalg.norm(
            np.asarray(
                parameter_only["sensor_position"],
                dtype=float,
            )
            - np.asarray(
                observed["sensor_position"],
                dtype=float,
            ),
            axis=1,
        )
        parameter_orientation_error = np.degrees(
            np.linalg.norm(
                deterministic._orientation_errors(
                    np.asarray(
                        observed["sensor_orientation_xyzw"],
                        dtype=float,
                    ),
                    np.asarray(
                        parameter_only["sensor_orientation_xyzw"],
                        dtype=float,
                    ),
                ),
                axis=1,
            )
        )
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            np.asarray(parameter_only["time"]),
            parameter_position_error,
        )
        axes[0].set_ylabel("position error norm [m]")
        axes[0].set_title(
            "Parameter-only open-loop rollout (separate diagnostic)"
        )
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(
            np.asarray(parameter_only["time"]),
            parameter_orientation_error,
        )
        axes[1].set_ylabel("orientation error angle [deg]")
        axes[1].set_xlabel("record-local time [s]")
        axes[1].grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        # ------------------------------------------------------------------
        # 2. External-wrench variability.  The band directly shows what
        #    "not confident in the disturbance" means under the iid Gaussian
        #    approximation.
        # ------------------------------------------------------------------
        for start, title, labels in (
            (
                0,
                "Dimensionless residual force: empirical mean ± 1 std",
                ("Fx", "Fy", "Fz"),
            ),
            (
                3,
                "Dimensionless residual torque: empirical mean ± 1 std",
                ("Mx", "My", "Mz"),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for local, axis in enumerate(axes):
                component = start + local
                mean = float(wrench_mean[component])
                std = float(wrench_std[component])
                axis.plot(
                    relative_time,
                    wrench_dimensionless[:, component],
                    label="residual wrench",
                )
                axis.axhline(
                    mean,
                    linestyle="--",
                    label="empirical mean",
                )
                axis.fill_between(
                    relative_time,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    label="mean ± 1 std",
                )
                axis.set_ylabel(labels[local])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                title
                + "\nWide band = disturbance varies more, so matching "
                "parameter directions receive weaker evidence"
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time from analysis start [s]")
            pdf.savefig(figure)
            plt.close(figure)

        # ------------------------------------------------------------------
        # 3. Literal SVD output: A = U diag(s) V^T.
        # ------------------------------------------------------------------
        singular_values = np.asarray(
            svd["singular_values_descending"], dtype=float
        )
        mode_index = np.arange(1, singular_values.size + 1)
        figure, axis = plt.subplots(
            figsize=(11.7, 8.3), constrained_layout=True
        )
        axis.semilogy(
            mode_index,
            np.maximum(singular_values, np.finfo(float).tiny),
            marker="o",
        )
        axis.set_xticks(mode_index)
        axis.set_xlabel("SVD mode i (largest s_i -> smallest s_i)")
        axis.set_ylabel("singular value s_i")
        axis.set_title(
            "Singular values S of the whitened dimensionless Jacobian"
        )
        axis.grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        # _svd_payload stores modes from smallest s_i to largest s_i.
        # Reverse only that storage order to reconstruct the ordinary V^T.
        vt_rows = np.asarray(
            [
                mode["dimensionless_direction"]
                for mode in reversed(svd["weak_directions"])
            ],
            dtype=float,
        )
        figure, axis = plt.subplots(
            figsize=(11.7, 8.3), constrained_layout=True
        )
        image = axis.imshow(
            vt_rows,
            aspect="auto",
            interpolation="nearest",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axis.set_xticks(
            np.arange(len(names)), names, rotation=70, ha="right"
        )
        axis.set_yticks(
            np.arange(vt_rows.shape[0]),
            [
                "v{}  s={:.3g}".format(i + 1, singular_values[i])
                for i in range(vt_rows.shape[0])
            ],
        )
        axis.set_xlabel("original dimensionless estimator coordinate")
        axis.set_ylabel("right-singular vector v_i")
        axis.set_title(
            "Signed right-singular vectors V^T\n"
            "A complete row may flip sign; relative signs inside the row matter"
        )
        figure.colorbar(
            image, ax=axis, label="signed component of v_i"
        )
        pdf.savefig(figure)
        plt.close(figure)

        # Decoded physical parameter values are shown separately from SVD.
        decoded = reconstruction["selected_physical_parameters"]
        inertia = np.asarray(decoded["inertia_kg_m2"], dtype=float)
        principal = np.asarray(
            decoded["inertia_principal_moments_kg_m2"], dtype=float
        )
        cog = np.asarray(decoded["cog_position_body_m"], dtype=float)
        effectiveness = np.asarray(
            decoded["force_effectiveness"], dtype=float
        )
        physical_text = "\n".join(
            (
                "Decoded physical parameters at the selected estimate",
                "",
                "mass [kg]",
                "  {:.12g}".format(decoded["mass_kg"]),
                "",
                "inertia J [kg m^2]",
                "  [{: .8g}  {: .8g}  {: .8g}]".format(*inertia[0]),
                "  [{: .8g}  {: .8g}  {: .8g}]".format(*inertia[1]),
                "  [{: .8g}  {: .8g}  {: .8g}]".format(*inertia[2]),
                "",
                "principal moments [kg m^2]",
                "  [{: .8g}, {: .8g}, {: .8g}]".format(*principal),
                "",
                "CoG position in body frame [m]",
                "  [{: .8g}, {: .8g}, {: .8g}]".format(*cog),
                "",
                "rotor force effectiveness",
                "  [{: .8g}, {: .8g}, {: .8g}, {: .8g}]".format(
                    *effectiveness
                ),
                "",
                "selected delay [s]",
                "  {:.12g}".format(decoded["delay_seconds"]),
            )
        )
        figure, axis = plt.subplots(
            figsize=(11.7, 8.3), constrained_layout=True
        )
        axis.axis("off")
        axis.text(
            0.02,
            0.98,
            physical_text,
            va="top",
            ha="left",
            family="monospace",
            transform=axis.transAxes,
        )
        pdf.savefig(figure)
        plt.close(figure)

        # Simple axis-wise angular-motion diagnostics.  These do not replace
        # the full SVD; they help judge hypotheses such as insufficient yaw
        # excitation.
        axis_names = tuple(angular_excitation["axis_names"])
        omega_rms = np.asarray(
            angular_excitation["angular_velocity_rms_rad_per_s"],
            dtype=float,
        )
        alpha_rms = np.asarray(
            angular_excitation[
                "angular_acceleration_rms_rad_per_s2"
            ],
            dtype=float,
        )
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axes[0].bar(axis_names, omega_rms)
        axes[0].set_ylabel("RMS angular velocity [rad/s]")
        axes[0].set_title(
            "Axis-wise rotational excitation diagnostic"
        )
        axes[0].grid(True, axis="y", alpha=0.25)
        axes[1].bar(axis_names, alpha_rms)
        axes[1].set_ylabel("RMS angular acceleration [rad/s^2]")
        axes[1].set_xlabel(
            "Low body-z values can support a yaw-excitation hypothesis, "
            "but identifiability is decided by the full Jacobian."
        )
        axes[1].grid(True, axis="y", alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        # ------------------------------------------------------------------
        # 4. Confidence relative to the chosen proper prior.  This scale has
        #    a direct statistical meaning, unlike raw Hessian eigenvalues.
        # ------------------------------------------------------------------
        prior_confidence = posterior.get(
            "prior_normalized_confidence"
        )
        if prior_confidence is not None:
            remaining = np.asarray(
                prior_confidence[
                    "remaining_prior_std_fraction_descending_modes"
                ],
                dtype=float,
            )
            figure, axis = plt.subplots(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axis.bar(
                np.arange(1, remaining.size + 1),
                remaining,
            )
            axis.set_ylim(0.0, 1.05)
            axis.set_xticks(
                np.arange(1, remaining.size + 1)
            )
            axis.set_xlabel(
                "prior-normalized local combination "
                "(1 = best learned, {} = least learned)".format(
                    remaining.size
                )
            )
            axis.set_ylabel(
                "fraction of prior standard deviation remaining\n"
                "(0 = strong uncertainty reduction; "
                "1 = prior essentially unchanged)"
            )
            axis.set_title(
                "How much uncertainty remains in every local combination?"
            )
            axis.grid(True, axis="y", alpha=0.25)
            pdf.savefig(figure)
            plt.close(figure)

        if "remaining_std_fraction_weakest_mode" in prefix:
            prefix_time = np.asarray(
                prefix["time_seconds_from_analysis_start"],
                dtype=float,
            )
            figure, axis = plt.subplots(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axis.plot(
                prefix_time,
                np.asarray(
                    prefix[
                        "remaining_std_fraction_weakest_mode"
                    ],
                    dtype=float,
                ),
                label="least learned mode",
            )
            axis.plot(
                prefix_time,
                np.asarray(
                    prefix[
                        "remaining_std_fraction_median_mode"
                    ],
                    dtype=float,
                ),
                label="median mode",
            )
            axis.plot(
                prefix_time,
                np.asarray(
                    prefix[
                        "remaining_std_fraction_strongest_mode"
                    ],
                    dtype=float,
                ),
                label="best learned mode",
            )
            axis.set_ylim(0.0, 1.05)
            axis.set_xlabel("observed duration [s]")
            axis.set_ylabel(
                "fraction of prior std remaining\n"
                "(1 = no learning; 0 = strong learning)"
            )
            axis.set_title(
                "Confidence accumulation as the rosbag becomes longer"
            )
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")
            pdf.savefig(figure)
            plt.close(figure)



def create_argument_parser() -> argparse.ArgumentParser:
    parser = deterministic.create_argument_parser()
    parser.description = (
        "Analyze local parameter confidence and ridge directions from one "
        "deterministic spline-dynamics bag."
    )
    return parser


def run(arguments: argparse.Namespace) -> int:
    if not hasattr(deterministic, "PHYSICAL_PARAMETER_NAMES"):
        raise SystemExit(
            "deterministic_spline_dynamics_estimator.py is not patched for "
            "independent rotor-effectiveness coordinates; run "
            "apply_spline_dynamics_confidence_patch.py first"
        )
    if deterministic.PHYSICAL_DIMENSION != 14:
        raise SystemExit(
            "confidence script expects the patched 14-D physical coordinate"
        )

    try:
        config = deterministic.load_spline_config(
            arguments.config
        )
        vehicle_model = deterministic.load_vehicle_model(
            arguments.vehicle_model_json
        )
        parameter_prior = deterministic.load_parameter_prior(
            arguments.prior_json
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    specification = _only_bag(config)
    initial_delay = (
        config.multi_bag.initial_delay_seconds
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    deterministic._validate_arguments(
        arguments,
        config,
        initial_delay,
    )

    started = time.perf_counter()
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
    print(
        "selected spline knot spacing {:.6g}s; support [{:.6f}, {:.6f}]s"
        .format(
            bag.spline_selection.selected_spacing_seconds,
            bag.collocation_time[0],
            bag.collocation_time[-1],
        ),
        flush=True,
    )

    selected, optimizer_history = _estimate_solution(
        bag,
        arguments,
        initial_delay,
        vehicle_model.parameters,
        parameter_prior,
    )
    evaluation = selected.evaluation.bag_evaluations[0]
    print(
        "selected delay {:.6f}s; mass {:.6g} kg".format(
            selected.delay_seconds,
            selected.evaluation.decoded.parameters.mass,
        ),
        flush=True,
    )

    scales = _reference_scales(vehicle_model.parameters)
    raw_per_dimensionless = (
        _parameter_raw_per_dimensionless(
            scales["length_m"]
        )
    )
    wrench_scale = _wrench_dimensionless_scale(
        scales["force_N"],
        scales["torque_Nm"],
    )

    wrench_raw = np.asarray(
        evaluation.residual_body_wrench,
        dtype=float,
    )
    wrench_jacobian_raw = np.asarray(
        evaluation.residual_wrench_jacobian,
        dtype=float,
    )
    if wrench_jacobian_raw.shape != (
        wrench_raw.shape[0],
        6,
        deterministic.PHYSICAL_DIMENSION,
    ):
        raise RuntimeError(
            "residual-wrench Jacobian has unexpected shape {}"
            .format(wrench_jacobian_raw.shape)
        )

    wrench_dimensionless = (
        wrench_raw * wrench_scale[None, :]
    )
    # d(w_bar)/d(q_bar) =
    # diag(wrench_scale) d(w_raw)/d(q_raw) diag(raw_per_dimensionless).
    jacobian_dimensionless = (
        wrench_jacobian_raw
        * wrench_scale[None, :, None]
        * raw_per_dimensionless[None, None, :]
    )

    wrench_mean_dimensionless = np.mean(
        wrench_dimensionless,
        axis=0,
    )
    wrench_covariance_dimensionless = np.cov(
        wrench_dimensionless,
        rowvar=False,
        ddof=1,
    )
    whitener, covariance_diagnostics = _pseudo_whitener(
        wrench_covariance_dimensionless
    )

    centered = (
        wrench_dimensionless
        - wrench_mean_dimensionless[None, :]
    )
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
    data_information = (
        stacked_jacobian.T @ stacked_jacobian
    )
    data_information_vector = -(
        stacked_jacobian.T @ whitened_residual.reshape(-1)
    )

    svd = _svd_payload(
        stacked_jacobian,
        deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    posterior = _posterior_payload(
        data_information=data_information,
        data_information_vector=data_information_vector,
        selected=selected,
        raw_per_dimensionless=raw_per_dimensionless,
        reference_parameters=vehicle_model.parameters,
        parameter_prior=parameter_prior,
        names=deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    prefix = _prefix_information(
        bag.collocation_time,
        whitened_jacobian,
        posterior["prior_covariance_dimensionless"],
    )
    reconstruction = _trajectory_reconstruction(
        bag,
        selected,
        evaluation,
        arguments,
        initial_delay,
        vehicle_model.parameters,
    )
    angular_excitation = _angular_excitation_payload(bag)

    wrench_mean_raw = np.mean(
        wrench_raw,
        axis=0,
    )
    wrench_covariance_raw = np.cov(
        wrench_raw,
        rowvar=False,
        ddof=1,
    )
    mahalanobis_squared = np.sum(
        whitened_residual * whitened_residual,
        axis=1,
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
        / specification.bag_id
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "reference_parameters": _parameter_payload(
                vehicle_model.parameters
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
                "-log L(delta) = 0.5 delta^T Lambda delta "
                "- eta^T delta + constant"
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
            "covariance_dimensionless": (
                wrench_covariance_dimensionless
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
            "covariance_dimensionless": (
                posterior["posterior_covariance_dimensionless"]
            ),
            "precision_dimensionless": (
                posterior["posterior_precision_dimensionless"]
            ),
        },
        "posterior_physical": posterior["posterior_physical"],
        "data_identifiability": likelihood_payload["identifiability"],
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
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "vehicle_model_path": vehicle_model.source_path,
        "parameter_prior_path": parameter_prior.source_path,
        "bag": {
            "id": specification.bag_id,
            "path": specification.path,
            "start_seconds": specification.start,
            "end_seconds": specification.end,
            "collocation_count": int(
                bag.collocation_time.size
            ),
            "collocation_time_seconds": bag.collocation_time,
            "selected_knot_spacing_seconds": float(
                bag.spline_selection.selected_spacing_seconds
            ),
        },
        "assumptions": {
            "pose_uncertainty": "not modeled in this analysis",
            "external_wrench": (
                "collocation samples are treated as iid Gaussian with "
                "nonzero empirical mean and empirical 6x6 covariance"
            ),
            "external_wrench_source": (
                "Confidence uses spline-implied residual_body_wrench at the "
                "selected deterministic parameter point. Refined replay wrench "
                "is used separately only for trajectory-reconstruction validation."
            ),
            "information_accumulation": (
                "sample information matrices are summed, not averaged"
            ),
            "local_parameter_combination_definition": (
                "right singular vectors of the whitened dimensionless "
                "residual-wrench Jacobian; large singular values are best "
                "identified combinations and small singular values are poorly "
                "identified/ridge combinations"
            ),
        },
        "nondimensionalization": {
            "reference_scales": scales,
            "parameter_raw_per_dimensionless": raw_per_dimensionless,
            "wrench_raw_to_dimensionless_diagonal": wrench_scale,
            "parameter_names": deterministic.PHYSICAL_PARAMETER_NAMES,
        },
        "deterministic_solution": {
            "physical_coordinate": selected.physical_coordinate,
            "delay_seconds": selected.delay_seconds,
            "objective_cost": float(
                deterministic._solution_cost(selected)
            ),
            "parameters": _parameter_payload(
                selected.evaluation.decoded.parameters
            ),
            "optimizer_history": optimizer_history,
        },
        "external_wrench_model": {
            "mean_raw_body_wrench": wrench_mean_raw,
            "covariance_raw_body_wrench": wrench_covariance_raw,
            "mean_dimensionless": wrench_mean_dimensionless,
            "covariance_dimensionless": (
                wrench_covariance_dimensionless
            ),
            "covariance_diagnostics": covariance_diagnostics,
            "mahalanobis_squared_per_sample": mahalanobis_squared,
            "mahalanobis_squared_mean": float(
                np.mean(mahalanobis_squared)
            ),
            "mahalanobis_squared_median": float(
                np.median(mahalanobis_squared)
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
        "elapsed_seconds": float(
            time.perf_counter() - started
        ),
    }
    json_path = output_directory / "confidence.json"
    pdf_path = output_directory / "confidence.pdf"
    _write_json(json_path, payload)
    _write_pdf(
        pdf_path,
        time_axis=bag.collocation_time,
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
        "confidence analysis written to {}".format(
            output_directory
        ),
        flush=True,
    )
    replay_metrics = reconstruction["external_wrench_replay"]["metrics"]
    parameter_metrics = reconstruction["parameter_only"]["metrics"]
    print(
        "14-D trajectory check: parameter-only RMSE {:.6g} m / {:.6g} deg; "
        "with external-wrench replay {:.6g} m / {:.6g} deg".format(
            parameter_metrics["position_rmse_m"],
            parameter_metrics["orientation_angle_rmse_deg"],
            replay_metrics["position_rmse_m"],
            replay_metrics["orientation_angle_rmse_deg"],
        ),
        flush=True,
    )
    print(
        "data-only ridge check: weakest relative information {:.3e} "
        "(1 = strongest direction; near 0 = ridge)".format(
            svd["weakest_relative_information_strength"]
        ),
        flush=True,
    )
    prior_confidence = posterior.get("prior_normalized_confidence")
    if prior_confidence is not None:
        remaining = np.asarray(
            prior_confidence[
                "remaining_prior_std_fraction_descending_modes"
            ],
            dtype=float,
        )
        print(
            "confidence across local parameter combinations: "
            "best retains {:.1f}% of prior std; "
            "least retains {:.1f}%".format(
                100.0 * float(remaining[0]),
                100.0 * float(remaining[-1]),
            ),
            flush=True,
        )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
