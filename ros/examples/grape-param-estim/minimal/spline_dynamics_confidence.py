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
2. Rigid-body quantities are nondimensionalized by fixed nominal reference
   scales:
       M* = nominal mass
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
5. The data information matrix is kept separate from the broad proper prior.
   The posterior precision is their sum.
6. Prefix information spectra are stored without dividing by sample count, so
   longer bags accumulate information under the iid approximation.

Run after applying ``apply_spline_dynamics_confidence_patch.py`` to
``deterministic_spline_dynamics_estimator.py``.
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
from grape_param_estim.system import GRAVITY, VehicleParameters


SCHEMA = "grape-param-estim/spline-dynamics-confidence/v1"
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
        "cog_offset_m": np.asarray(parameters.cog_offset, dtype=float),
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
) -> tuple[Any, dict[str, Any]]:
    """Run the same smooth-lag -> strict-ZOH physical solve as the estimator."""

    problem = deterministic.SplineDynamicsProblem(
        (bag,),
        arguments.prior_weight,
    )
    initial_physical = np.zeros(
        deterministic.PHYSICAL_DIMENSION,
        dtype=float,
    )
    physical_lower, physical_upper = deterministic._physical_bounds(
        initial_physical,
        arguments.physical_bound_scale,
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


def _reference_scales() -> dict[str, float]:
    nominal = VehicleParameters.nominal()
    mass_scale = float(nominal.mass)
    length_scale = math.sqrt(
        float(np.trace(nominal.inertia))
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
        raise ValueError("nominal nondimensionalization scales are invalid")
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
        vector = vt[index]
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
                    "Large absolute coefficients identify parameters that can "
                    "move together along this local weak/ridge direction. "
                    "The coefficient sign gives the relative direction of motion."
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



def _prefix_information(
    time_axis: np.ndarray,
    sample_jacobian: np.ndarray,
    prior_standard_deviation_dimensionless: np.ndarray,
    prior_weight: float,
) -> dict[str, Any]:
    """Track interpretable confidence accumulation as the bag becomes longer.

    For every prefix, data information is measured relative to the proper
    parameter prior.  If ``g`` is an eigenvalue of the prior-whitened data
    information, the local posterior standard deviation along that generalized
    mode is reduced by

        remaining_std_fraction = 1 / sqrt(1 + g).

    Hence:
        1 -> this bag has taught us essentially nothing beyond the prior;
        0 -> data strongly constrain that mode relative to the prior.
    """

    time_value = np.asarray(time_axis, dtype=float)
    jacobian = np.asarray(sample_jacobian, dtype=float)
    prior_std = np.asarray(
        prior_standard_deviation_dimensionless,
        dtype=float,
    )
    dimension = jacobian.shape[2]
    if prior_std.shape != (dimension,):
        raise ValueError("prefix prior scale has the wrong dimension")

    cumulative = np.zeros(
        (dimension, dimension),
        dtype=float,
    )
    eigenvalue_history = np.empty(
        (time_value.size, dimension),
        dtype=float,
    )
    relative_history = np.empty(
        (time_value.size, dimension),
        dtype=float,
    )
    numerical_rank_history = np.empty(
        time_value.size,
        dtype=int,
    )

    if prior_weight > 0.0:
        effective_prior_std = prior_std / math.sqrt(float(prior_weight))
        prior_covariance_sqrt = np.diag(effective_prior_std)
        generalized_information_history = np.empty(
            (time_value.size, dimension),
            dtype=float,
        )
        remaining_history = np.empty(
            (time_value.size, dimension),
            dtype=float,
        )
    else:
        effective_prior_std = None
        prior_covariance_sqrt = None
        generalized_information_history = None
        remaining_history = None

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

        if prior_covariance_sqrt is not None:
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
            remaining_history[index] = (
                1.0 / np.sqrt(1.0 + generalized)
            )

    payload: dict[str, Any] = {
        "time_seconds_from_analysis_start": (
            time_value - time_value[0]
        ),
        "information_eigenvalues_descending": eigenvalue_history,
        "relative_information_strength_descending": relative_history,
        "numerical_rank": numerical_rank_history,
        "relative_information_interpretation": (
            "1 means as informative as the strongest direction at that prefix; "
            "values near 0 indicate weak/ridge directions."
        ),
    }

    if remaining_history is not None:
        # generalized eigenvalues are descending, so remaining fractions are
        # ascending: [strongest learned, ..., weakest learned].
        payload.update(
            {
                "prior_weight": float(prior_weight),
                "effective_prior_standard_deviation_dimensionless": (
                    effective_prior_std
                ),
                "prior_normalized_data_information_descending": (
                    generalized_information_history
                ),
                "remaining_std_fraction_by_mode": remaining_history,
                "remaining_std_fraction_strongest_mode": (
                    remaining_history[:, 0]
                ),
                "remaining_std_fraction_median_mode": np.median(
                    remaining_history,
                    axis=1,
                ),
                "remaining_std_fraction_weakest_mode": (
                    remaining_history[:, -1]
                ),
                "remaining_std_fraction_interpretation": (
                    "1 means the bag did not reduce the prior uncertainty; "
                    "0 means the data dominate the prior along that generalized mode."
                ),
            }
        )
    return payload



def _posterior_payload(
    *,
    data_information: np.ndarray,
    selected_coordinate: np.ndarray,
    raw_per_dimensionless: np.ndarray,
    prior_raw_standard_deviation: np.ndarray,
    prior_weight: float,
    names: Sequence[str],
) -> dict[str, Any]:
    data_information = np.asarray(
        data_information,
        dtype=float,
    )
    raw_per_dimensionless = np.asarray(
        raw_per_dimensionless,
        dtype=float,
    )
    selected_raw = np.asarray(
        selected_coordinate,
        dtype=float,
    )
    prior_raw = np.asarray(
        prior_raw_standard_deviation,
        dtype=float,
    )

    selected_dimensionless = selected_raw / raw_per_dimensionless
    prior_dimensionless = prior_raw / raw_per_dimensionless
    prior_precision = (
        float(prior_weight)
        * np.diag(1.0 / prior_dimensionless**2)
    )
    posterior_precision = data_information + prior_precision
    posterior_covariance = np.linalg.pinv(
        posterior_precision,
        hermitian=True,
    )
    raw_scale_matrix = np.diag(raw_per_dimensionless)
    posterior_covariance_raw = (
        raw_scale_matrix
        @ posterior_covariance
        @ raw_scale_matrix
    )

    payload: dict[str, Any] = {
        "selected_coordinate_raw": selected_raw,
        "selected_coordinate_dimensionless": selected_dimensionless,
        "prior_weight": float(prior_weight),
        "prior_standard_deviation_raw": prior_raw,
        "prior_standard_deviation_dimensionless": prior_dimensionless,
        "prior_precision_dimensionless": prior_precision,
        "data_information_dimensionless": data_information,
        "posterior_precision_dimensionless": posterior_precision,
        "posterior_covariance_dimensionless": posterior_covariance,
        "posterior_standard_deviation_dimensionless": np.sqrt(
            np.maximum(0.0, np.diag(posterior_covariance))
        ),
        "posterior_covariance_raw_coordinate": posterior_covariance_raw,
        "posterior_standard_deviation_raw_coordinate": np.sqrt(
            np.maximum(0.0, np.diag(posterior_covariance_raw))
        ),
    }

    if prior_weight > 0.0:
        effective_prior_std = (
            prior_dimensionless / math.sqrt(float(prior_weight))
        )
        prior_covariance_sqrt = np.diag(effective_prior_std)
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
        # Store weakest-learning modes first because those are the ones that
        # matter for "I am not confident".
        for rank_from_weakest, index in enumerate(
            range(eigenvalues.size - 1, -1, -1)
        ):
            direction = eigenvectors[:, index]
            physical_direction = (
                prior_covariance_sqrt @ direction
            )
            norm = np.linalg.norm(physical_direction)
            if norm > 0.0:
                physical_direction = physical_direction / norm
            component_order = np.argsort(
                np.abs(physical_direction),
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
                    "dimensionless_parameter_direction": (
                        physical_direction
                    ),
                    "dominant_components": [
                        {
                            "name": str(names[column]),
                            "coefficient": float(
                                physical_direction[column]
                            ),
                        }
                        for column in component_order[
                            : min(8, len(component_order))
                        ]
                    ],
                }
            )

        marginal_prior_std = effective_prior_std
        marginal_posterior_std = np.sqrt(
            np.maximum(0.0, np.diag(posterior_covariance))
        )
        marginal_remaining_fraction = (
            marginal_posterior_std / marginal_prior_std
        )
        payload["prior_normalized_confidence"] = {
            "data_to_prior_precision_ratio_descending": eigenvalues,
            "remaining_prior_std_fraction_descending_modes": remaining_std,
            "interpretation": (
                "data_to_prior_precision_ratio > 1 means the data add more "
                "precision than the prior along that generalized mode. "
                "remaining_prior_std_fraction = 1 means no learning beyond "
                "the prior; values near 0 mean strong data constraint."
            ),
            "weak_modes": generalized_modes,
            "marginal_remaining_prior_std_fraction": (
                marginal_remaining_fraction
            ),
        }

    return payload



def _trajectory_reconstruction(
    bag: Any,
    selected: Any,
    dynamics_evaluation: Any,
    arguments: argparse.Namespace,
    initial_delay: float,
) -> dict[str, Any]:
    """Validate that the 14-D parameterization still reproduces the trajectory.

    Three rollouts are kept distinct:
    - nominal: nominal physical parameters;
    - parameter_only: selected 14-D parameters with no external wrench;
    - external_wrench_replay: selected 14-D parameters held fixed while the
      existing external-wrench replay is refined.

    The last one answers whether the complete deterministic reconstruction
    pipeline remains intact after adding the fourth rotor-effectiveness degree
    of freedom.
    """

    observations = bag.direct_problem.observations

    nominal_rollout = deterministic.forward_rollout(
        bag,
        np.zeros(
            deterministic.PHYSICAL_DIMENSION,
            dtype=float,
        ),
        initial_delay,
    )
    parameter_rollout = deterministic.forward_rollout(
        bag,
        selected.physical_coordinate,
        selected.delay_seconds,
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
    )
    replay_rollout = wrench_evaluation.simulation
    replay_observations = deterministic._observations_at_times(
        observations,
        replay_rollout.time,
    )

    nominal_metrics = deterministic._pose_metrics(
        observations,
        nominal_rollout,
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
        "  pose RMSE: nominal {:.6g} m / {:.6g} deg; "
        "14-D parameter-only {:.6g} m / {:.6g} deg; "
        "14-D + replay {:.6g} m / {:.6g} deg".format(
            nominal_metrics["position_rmse_m"],
            nominal_metrics["orientation_angle_rmse_deg"],
            parameter_metrics["position_rmse_m"],
            parameter_metrics["orientation_angle_rmse_deg"],
            replay_metrics["position_rmse_m"],
            replay_metrics["orientation_angle_rmse_deg"],
        ),
        flush=True,
    )

    return {
        "meaning": {
            "nominal": (
                "Nominal physical parameters and initial lag; no external wrench."
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
        "nominal": {
            "metrics": nominal_metrics,
            "time": nominal_rollout.time,
            "sensor_position": nominal_rollout.sensor_position,
            "sensor_orientation_xyzw": (
                nominal_rollout.sensor_orientation_xyzw
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
    nominal = reconstruction["nominal"]

    with PdfPages(path) as pdf:
        # ------------------------------------------------------------------
        # 1. Most important sanity check: does the 14-D solution still
        #    reconstruct the measured trajectory?
        # ------------------------------------------------------------------
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        labels = ("x [m]", "y [m]", "z [m]")
        for component, axis in enumerate(axes):
            axis.plot(
                np.asarray(observed["time"]),
                np.asarray(observed["sensor_position"])[:, component],
                label="observed",
            )
            axis.plot(
                np.asarray(parameter_only["time"]),
                np.asarray(parameter_only["sensor_position"])[:, component],
                label="14-D parameters only",
            )
            axis.plot(
                np.asarray(replay["time"]),
                np.asarray(replay["sensor_position"])[:, component],
                label="14-D + external-wrench replay",
            )
            axis.set_ylabel(labels[component])
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Trajectory reproduction after adding independent rotor effectiveness"
        )
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("record-local time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        observed_position = np.asarray(
            observed["sensor_position"],
            dtype=float,
        )
        nominal_position_error = np.linalg.norm(
            np.asarray(nominal["sensor_position"], dtype=float)
            - observed_position,
            axis=1,
        )
        parameter_position_error = np.linalg.norm(
            np.asarray(parameter_only["sensor_position"], dtype=float)
            - observed_position,
            axis=1,
        )
        replay_position_error = np.linalg.norm(
            np.asarray(replay["sensor_position"], dtype=float)
            - np.asarray(
                replay["observed_position_on_replay_support"],
                dtype=float,
            ),
            axis=1,
        )

        nominal_orientation_error = np.degrees(
            np.linalg.norm(
                deterministic._orientation_errors(
                    np.asarray(
                        observed["sensor_orientation_xyzw"],
                        dtype=float,
                    ),
                    np.asarray(
                        nominal["sensor_orientation_xyzw"],
                        dtype=float,
                    ),
                ),
                axis=1,
            )
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
            constrained_layout=True,
        )
        axes[0].plot(
            np.asarray(nominal["time"]),
            nominal_position_error,
            label="nominal",
        )
        axes[0].plot(
            np.asarray(parameter_only["time"]),
            parameter_position_error,
            label="14-D parameters only",
        )
        axes[0].plot(
            np.asarray(replay["time"]),
            replay_position_error,
            label="14-D + replay",
        )
        axes[0].set_ylabel("position error norm [m]")
        axes[0].set_title(
            "Trajectory error: lower is better; replay verifies the full pipeline"
        )
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(
            np.asarray(nominal["time"]),
            nominal_orientation_error,
            label="nominal",
        )
        axes[1].plot(
            np.asarray(parameter_only["time"]),
            parameter_orientation_error,
            label="14-D parameters only",
        )
        axes[1].plot(
            np.asarray(replay["time"]),
            replay_orientation_error,
            label="14-D + replay",
        )
        axes[1].set_ylabel("orientation error angle [deg]")
        axes[1].set_xlabel("record-local time [s]")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")
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
        # 3. Data-only ridge geometry.  Normalize away the arbitrary overall
        #    scale so the graph has one simple interpretation.
        # ------------------------------------------------------------------
        relative_information = np.asarray(
            svd["relative_information_strength_descending"],
            dtype=float,
        )
        figure, axis = plt.subplots(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis.semilogy(
            np.arange(1, relative_information.size + 1),
            np.maximum(
                relative_information,
                np.finfo(float).tiny,
            ),
            marker="o",
        )
        axis.set_xlabel(
            "data-information mode (strongest → weakest)"
        )
        axis.set_ylabel(
            "relative information strength\n"
            "(1 = strongest; near 0 = ridge)"
        )
        axis.set_title(
            "Data-only identifiability spectrum\n"
            "Small values mean the trajectory scarcely changes along that "
            "parameter combination"
        )
        axis.grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        # Weakest three data-only ridge directions, each as an ordinary bar
        # plot instead of an unlabeled heatmap.
        weak = list(svd["weak_directions"])
        for rank, mode in enumerate(weak[:3], start=1):
            direction = np.asarray(
                mode["dimensionless_direction"],
                dtype=float,
            )
            figure, axis = plt.subplots(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            y = np.arange(len(names))
            axis.barh(y, direction)
            axis.set_yticks(y, names)
            axis.axvline(0.0, linewidth=1.0)
            axis.set_xlabel(
                "coefficient in dimensionless ridge direction\n"
                "(large |value| = participates strongly; sign = coupled motion)"
            )
            axis.set_title(
                "Ridge direction {}: relative information = {:.3e}\n"
                "Changing these parameters together produces little change "
                "in the wrench likelihood".format(
                    rank,
                    mode["relative_information_strength"],
                )
            )
            axis.grid(True, axis="x", alpha=0.25)
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
            # Reverse so the weakest learned mode appears first.
            remaining_weak_to_strong = remaining[::-1]
            figure, axis = plt.subplots(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axis.bar(
                np.arange(1, remaining.size + 1),
                remaining_weak_to_strong,
            )
            axis.set_ylim(0.0, 1.05)
            axis.set_xlabel(
                "prior-normalized mode (weakest learned → strongest learned)"
            )
            axis.set_ylabel(
                "fraction of prior standard deviation remaining\n"
                "(1 = bag adds no confidence; 0 = strongly learned)"
            )
            axis.set_title(
                "How much uncertainty remains after this bag?"
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
    )
    evaluation = selected.evaluation.bag_evaluations[0]
    print(
        "selected delay {:.6f}s; mass {:.6g} kg".format(
            selected.delay_seconds,
            selected.evaluation.decoded.parameters.mass,
        ),
        flush=True,
    )

    scales = _reference_scales()
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

    svd = _svd_payload(
        stacked_jacobian,
        deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    prior_dimensionless = (
        deterministic.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS
        / raw_per_dimensionless
    )
    prefix = _prefix_information(
        bag.collocation_time,
        whitened_jacobian,
        prior_dimensionless,
        arguments.prior_weight,
    )
    posterior = _posterior_payload(
        data_information=data_information,
        selected_coordinate=selected.physical_coordinate,
        raw_per_dimensionless=raw_per_dimensionless,
        prior_raw_standard_deviation=(
            deterministic.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS
        ),
        prior_weight=arguments.prior_weight,
        names=deterministic.PHYSICAL_PARAMETER_NAMES,
    )
    reconstruction = _trajectory_reconstruction(
        bag,
        selected,
        evaluation,
        arguments,
        initial_delay,
    )

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
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
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
            "ridge_definition": (
                "right singular directions of the whitened dimensionless "
                "residual-wrench Jacobian"
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
            "confidence relative to prior: least-learned mode retains "
            "{:.1f}% of prior std; best-learned mode retains {:.1f}%".format(
                100.0 * float(remaining[-1]),
                100.0 * float(remaining[0]),
            ),
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
