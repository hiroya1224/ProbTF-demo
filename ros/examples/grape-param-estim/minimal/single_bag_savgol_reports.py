#!/usr/bin/env python3
"""Commit-namespaced JSON/NPZ/PDF reporting for single-bag estimation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-single-bag-matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

try:  # noqa: E402
    from .single_bag_savgol_core import (
        BASE_PLAN_COMMIT,
        EstimationResult,
        SingleBagDataset,
        VehicleModelInput,
        physical_parameter_vector,
    )
    from .single_bag_wrench_replay import WrenchReplayResult
except ImportError:  # pragma: no cover
    from single_bag_savgol_core import (  # type: ignore
        BASE_PLAN_COMMIT,
        EstimationResult,
        SingleBagDataset,
        VehicleModelInput,
        physical_parameter_vector,
    )
    from single_bag_wrench_replay import WrenchReplayResult  # type: ignore


WRENCH_LINE_STYLES = {
    "raw_sg_inverse_dynamics": "-",
    "trajectory_fitted_external": "--",
    "additional": ":",
    "additional_2": "-.",
}
DIAGNOSTIC_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _quantile_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "minimum": None,
            "median": None,
            "maximum": None,
            "quantiles": {},
        }
    quantiles = np.quantile(finite, DIAGNOSTIC_QUANTILES)
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "quantiles": {
            "{:g}".format(level): float(item)
            for level, item in zip(DIAGNOSTIC_QUANTILES, quantiles)
        },
    }


def _covariance_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    eigenvalues = np.asarray(value["sigma_z_eigenvalues"])
    gains = np.asarray(value["whitening_gain"])
    ranks = np.asarray(value["sigma_z_machine_rank"], dtype=int)
    unique, counts = np.unique(ranks, return_counts=True)
    return {
        "sigma_z_eigenvalue_summary": {
            "overall": _quantile_summary(eigenvalues),
            "per_eigenmode": [
                _quantile_summary(eigenvalues[:, index])
                for index in range(eigenvalues.shape[1])
            ],
        },
        "sigma_z_rank_summary": {
            "minimum": int(np.min(ranks)),
            "median": float(np.median(ranks)),
            "maximum": int(np.max(ranks)),
            "counts": {
                str(int(rank)): int(count)
                for rank, count in zip(unique, counts)
            },
        },
        "whitening_gain_summary": {
            "overall": _quantile_summary(gains),
            "per_eigenmode": [
                _quantile_summary(gains[:, index])
                for index in range(gains.shape[1])
            ],
        },
        "mahalanobis_contribution_summary": _quantile_summary(
            np.asarray(value["mahalanobis_contribution_per_time"])
        ),
    }


def json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_sanitize(value), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def source_commit(repository_root: Optional[Path] = None) -> str:
    root = _repository_root() if repository_root is None else repository_root
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "{}-{}".format(prefix, stamp)


def output_run_directory(
    output_root: Path,
    category: str,
    run_id: Optional[str] = None,
    *,
    commit: Optional[str] = None,
) -> Path:
    revision = source_commit() if commit is None else str(commit)
    directory = Path(output_root) / revision / category / (
        new_run_id() if run_id is None else run_id
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _parameter_payload(
    model: VehicleModelInput, result: EstimationResult
) -> dict[str, Any]:
    nominal = model.parameters
    estimated = result.evaluation.parameters
    nominal_vector = physical_parameter_vector(nominal)
    estimated_vector = physical_parameter_vector(estimated)
    return {
        "nominal": {
            "mass_kg": nominal.mass,
            "inertia_kg_m2": nominal.inertia,
            "cog_position_body_m": nominal.cog_offset,
            "force_effectiveness": nominal.force_effectiveness,
        },
        "estimated": {
            "mass_kg": estimated.mass,
            "inertia_kg_m2": estimated.inertia,
            "cog_position_body_m": estimated.cog_offset,
            "force_effectiveness": estimated.force_effectiveness,
        },
        "physical_vector_nominal": nominal_vector,
        "physical_vector_estimated": estimated_vector,
        "physical_vector_delta": estimated_vector - nominal_vector,
        "chart_coordinate": result.physical_coordinate,
        "rotor_lag_seconds": result.rotor_lag_seconds,
        "rotor_lag_strict_cell_seconds": {
            "lower_exclusive": result.diagnostics["lag"][
                "strict_lag_cell_lower_seconds"
            ],
            "upper_inclusive": result.diagnostics["lag"][
                "strict_lag_cell_upper_seconds"
            ],
            "representative": result.diagnostics["lag"][
                "strict_lag_cell_representative_seconds"
            ],
        },
        "final_smooth_rotor_lag_seconds": (
            result.final_smooth_rotor_lag_seconds
        ),
        "scale_free": {
            "inertia_over_mass_m2": estimated.inertia / estimated.mass,
            "force_effectiveness_over_mass": (
                estimated.force_effectiveness / estimated.mass
            ),
            "cog_position_body_m": estimated.cog_offset,
        },
    }


def common_metrics(
    result: EstimationResult, replay: Optional[WrenchReplayResult]
) -> dict[str, Any]:
    reference = result.reference_evaluation
    acceleration = reference.acceleration_residual
    cross = result.diagnostics.get("metric_cross_evaluation", {}).get(
        "estimated", {}
    )
    metrics: dict[str, Any] = {
        "reference_objective_sum": reference.cost,
        "full_covariance_objective_sum": cross.get(
            "full_covariance_objective_sum", reference.cost
        ),
        "identity_objective_sum": cross.get(
            "identity_objective_sum", 0.5 * float(np.sum(acceleration**2))
        ),
        "specific_acceleration_rmse_m_per_s2": float(
            np.sqrt(np.mean(np.sum(acceleration[:, :3] ** 2, axis=1)))
        ),
        "angular_acceleration_rmse_rad_per_s2": float(
            np.sqrt(np.mean(np.sum(acceleration[:, 3:] ** 2, axis=1)))
        ),
        "raw_residual_force_rmse_n": float(
            np.sqrt(
                np.mean(
                    np.sum(result.evaluation.raw_residual_wrench[:, :3] ** 2, axis=1)
                )
            )
        ),
        "raw_residual_torque_rmse_nm": float(
            np.sqrt(
                np.mean(
                    np.sum(result.evaluation.raw_residual_wrench[:, 3:] ** 2, axis=1)
                )
            )
        ),
    }
    if replay is not None:
        metrics.update(
            {
                "fitted_external_force_rmse_n": float(
                    np.sqrt(
                        np.mean(
                            np.sum(replay.fitted_external_wrench[:, :3] ** 2, axis=1)
                        )
                    )
                ),
                "fitted_external_torque_rmse_nm": float(
                    np.sqrt(
                        np.mean(
                            np.sum(replay.fitted_external_wrench[:, 3:] ** 2, axis=1)
                        )
                    )
                ),
            }
        )
    return metrics


def result_payload(
    *,
    case_name: str,
    source_revision: str,
    model: VehicleModelInput,
    result: EstimationResult,
    replay: Optional[WrenchReplayResult],
) -> dict[str, Any]:
    diagnostics = result.diagnostics
    covariance = diagnostics.get("covariance", {})
    closure = diagnostics.get("closure", {})
    overlap = diagnostics.get("overlap_correction", {})
    diagnostic_payload = {
        "covariance": _covariance_summary(covariance),
        "reference_full_covariance": _covariance_summary(
            diagnostics.get("reference_full_covariance", covariance)
        ),
        "metric_cross_evaluation": diagnostics.get(
            "metric_cross_evaluation", {}
        ),
        "lag": diagnostics.get("lag", {}),
        "continuation": diagnostics.get("continuation", {}),
        "gimbal": diagnostics.get("gimbal", {}),
        "quotient": diagnostics.get("quotient", {}),
        "closure": {
            key: value
            for key, value in closure.items()
            if not key.endswith("_error")
        },
        "inertia": diagnostics.get("inertia", {}),
        "residual_wrench": diagnostics.get("residual_wrench", {}),
        "overlap_correction": {
            "cross_time_covariance_model": overlap.get(
                "cross_time_covariance_model"
            ),
            "uncertainty_variance_naive_in_ridge_basis": overlap.get(
                "uncertainty_variance_naive_in_ridge_basis"
            ),
            "uncertainty_variance_overlap_in_ridge_basis": overlap.get(
                "uncertainty_variance_overlap_in_ridge_basis"
            ),
            "uncertainty_variance_inflation_in_ridge_basis": overlap.get(
                "uncertainty_variance_inflation_in_ridge_basis"
            ),
            "variance_inflation_in_ridge_basis": overlap.get(
                "uncertainty_variance_inflation_in_ridge_basis"
            ),
        },
    }
    payload = {
        "status": "completed" if result.success else "failed",
        "case_name": case_name,
        "source_commit": source_revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "success": result.success,
        "solver_status": result.status,
        "solver_message": result.message,
        "elapsed_seconds": result.elapsed_seconds,
        "nfev": result.total_nfev,
        "parameters": _parameter_payload(model, result),
        "common_evaluation": common_metrics(result, replay),
        "actuator_active_set_counts": (
            result.evaluation.actuator_history.active_set_counts
        ),
        "strict_final_evaluation": result.evaluation.actuator_history.strict_final,
        "stages": result.stages,
        "diagnostics": diagnostic_payload,
        "ridge": {
            key: value
            for key, value in result.ridge.items()
            if key not in ("raw_whitened_jacobian", "raw_acceleration_jacobian")
        },
        "uncertainty": {
            "parameter_covariance_naive": result.uncertainty.naive,
            "parameter_covariance_overlap_corrected": (
                result.uncertainty.overlap_corrected
            ),
            "parameter_covariance_wrench_corrected": (
                result.uncertainty.wrench_corrected
            ),
        },
    }
    if not result.success:
        payload.update(
            {
                "failure_stage": result.first_failure_stage,
                "exception_type": "OptimizerNonSuccess",
                "failure_reason": result.message,
                "first_failure_stage": result.first_failure_stage,
                "first_failure_status": result.first_failure_status,
                "first_failure_message": result.first_failure_message,
            }
        )
    return payload


def arrays_payload(
    dataset: SingleBagDataset,
    result: EstimationResult,
    replay: Optional[WrenchReplayResult],
) -> dict[str, np.ndarray]:
    sg = dataset.sg
    covariance = dataset.covariance
    evaluation = result.evaluation
    diagnostics = result.diagnostics
    covariance_diagnostics = diagnostics["covariance"]
    reference_covariance_diagnostics = diagnostics["reference_full_covariance"]
    ridge_diagnostics = diagnostics["ridge"]
    overlap = diagnostics["overlap_correction"]
    closure = diagnostics["closure"]
    residual_wrench = result.residual_wrench_uncertainty
    lag_table = diagnostics["lag"]["candidate_table"]
    lag_rotor: list[float] = []
    lag_cost: list[float] = []
    lag_selected: list[bool] = []
    for row in lag_table:
        lag_rotor.append(float(row["representative_seconds"]))
        lag_cost.append(float(row["strict_cost"]))
        lag_selected.append(bool(row["selected"]))
    continuation = diagnostics["continuation"]
    quotient = diagnostics["quotient"]
    lag = diagnostics["lag"]
    arrays: dict[str, np.ndarray] = {
        "sg_time": np.asarray(dataset.time),
        "sg_R": np.asarray(sg.body_rotation),
        "sg_omega": np.asarray(sg.body_angular_velocity),
        "sg_a_S": np.asarray(sg.sensor_acceleration_world),
        "sg_alpha": np.asarray(sg.body_angular_acceleration),
        "sg_s": np.asarray(covariance.z[:, :3]),
        "local_Omega": np.asarray(covariance.local_omega),
        "local_Sigma_xi": np.asarray(covariance.local_sigma_xi),
        "local_Sigma_z": np.asarray(covariance.local_sigma_z),
        "actual_thrust_history": np.asarray(
            evaluation.actuator_history.actual_thrust
        ),
        "actual_gimbal_history": np.asarray(
            evaluation.actuator_history.actual_gimbal
        ),
        "gimbal_raw_time": np.asarray(dataset.gimbal_raw_time),
        "gimbal_raw_angle": np.asarray(dataset.gimbal_raw_angle),
        "gimbal_sg_angle": np.asarray(dataset.gimbal_sg_angle),
        "gimbal_command_replay_angle": np.asarray(
            diagnostics["gimbal"].get(
                "command_replay_angle", dataset.gimbal_sg_angle
            )
        ),
        "modeled_wrench": np.asarray(evaluation.modeled_wrench),
        "required_wrench": np.asarray(evaluation.required_wrench),
        "raw_residual_wrench": np.asarray(evaluation.raw_residual_wrench),
        "modeled_wrench_nominal_mass_gauge": np.asarray(
            residual_wrench.modeled_wrench
        ),
        "required_wrench_nominal_mass_gauge": np.asarray(
            residual_wrench.required_wrench
        ),
        "residual_wrench_nominal_mass_gauge": np.asarray(
            residual_wrench.wrench
        ),
        "residual_wrench_mass_gauge_scale": np.asarray(
            residual_wrench.mass_gauge_scale
        ),
        "residual_wrench_fixed_mass_kg": np.asarray(
            residual_wrench.fixed_mass_kg
        ),
        "residual_wrench_centered": np.asarray(
            residual_wrench.centered_wrench
        ),
        "residual_wrench_mean": np.asarray(residual_wrench.mean),
        "residual_wrench_total_empirical_covariance": np.asarray(
            residual_wrench.empirical_covariance
        ),
        "residual_wrench_total_empirical_std": np.asarray(
            residual_wrench.empirical_std
        ),
        "residual_wrench_total_empirical_correlation": np.asarray(
            residual_wrench.empirical_correlation
        ),
        "residual_wrench_sg_covariance_per_time": np.asarray(
            residual_wrench.sg_covariance_per_time
        ),
        "residual_wrench_sg_covariance_mean": np.asarray(
            residual_wrench.sg_covariance_mean
        ),
        "residual_wrench_excess_covariance_raw": np.asarray(
            residual_wrench.excess_covariance_raw
        ),
        "residual_wrench_excess_covariance_raw_eigenvalues": np.asarray(
            residual_wrench.excess_covariance_raw_eigenvalues
        ),
        "residual_wrench_model_discrepancy_covariance": np.asarray(
            residual_wrench.model_discrepancy_covariance
        ),
        "residual_wrench_model_discrepancy_eigenvalues": np.asarray(
            residual_wrench.model_discrepancy_eigenvalues
        ),
        "residual_wrench_model_discrepancy_std": np.asarray(
            residual_wrench.model_discrepancy_std
        ),
        "residual_wrench_model_discrepancy_correlation": np.asarray(
            residual_wrench.model_discrepancy_correlation
        ),
        "residual_acceleration_model_discrepancy_covariance": np.asarray(
            residual_wrench.acceleration_model_discrepancy_covariance
        ),
        "residual_acceleration_model_discrepancy_eigenvalues": np.asarray(
            residual_wrench.acceleration_model_discrepancy_eigenvalues
        ),
        "residual_acceleration_model_discrepancy_std": np.asarray(
            residual_wrench.acceleration_model_discrepancy_std
        ),
        "residual_acceleration_model_discrepancy_correlation": np.asarray(
            residual_wrench.acceleration_model_discrepancy_correlation
        ),
        "residual_acceleration": np.asarray(evaluation.acceleration_residual),
        "whitened_residual": np.asarray(evaluation.whitened_residual),
        "raw_physical_jacobian": np.asarray(evaluation.acceleration_jacobian),
        "whitened_physical_jacobian": np.asarray(evaluation.whitened_jacobian),
        "raw_jacobian": np.asarray(evaluation.acceleration_jacobian),
        "whitened_jacobian": np.asarray(
            result.ridge["raw_whitened_jacobian"]
        ),
        "raw_whitened_jacobian": np.asarray(
            result.ridge["raw_whitened_jacobian"]
        ),
        "singular_values": np.asarray(result.ridge["singular_values"]),
        "right_singular_vectors": np.asarray(
            result.ridge["right_singular_vectors"]
        ),
        "sigma_z_eigenvalues": np.asarray(
            covariance_diagnostics["sigma_z_eigenvalues"]
        ),
        "sigma_z_machine_rank": np.asarray(
            covariance_diagnostics["sigma_z_machine_rank"]
        ),
        "sigma_z_retained_condition_number": np.asarray(
            covariance_diagnostics["sigma_z_retained_condition_number"]
        ),
        "whitening_gain": np.asarray(
            covariance_diagnostics["whitening_gain"]
        ),
        "mahalanobis_contribution_per_time": np.asarray(
            covariance_diagnostics["mahalanobis_contribution_per_time"]
        ),
        "covariance_eigenmode_residual": np.asarray(
            covariance_diagnostics["covariance_eigenmode_residual"]
        ),
        "covariance_eigenmode_mahalanobis_contribution": np.asarray(
            covariance_diagnostics[
                "covariance_eigenmode_mahalanobis_contribution"
            ]
        ),
        "reference_full_sigma_z_eigenvalues": np.asarray(
            reference_covariance_diagnostics["sigma_z_eigenvalues"]
        ),
        "reference_full_whitening_gain": np.asarray(
            reference_covariance_diagnostics["whitening_gain"]
        ),
        "reference_full_mahalanobis_contribution_per_time": np.asarray(
            reference_covariance_diagnostics[
                "mahalanobis_contribution_per_time"
            ]
        ),
        "unwhitened_physical_jacobian_singular_values": np.asarray(
            result.ridge["unwhitened_diagnostic_singular_values"]
        ),
        "unwhitened_physical_jacobian_right_singular_vectors": np.asarray(
            result.ridge["unwhitened_diagnostic_right_singular_vectors"]
        ),
        "whitened_physical_jacobian_singular_values": np.asarray(
            result.ridge["whitened_singular_values"]
        ),
        "whitened_physical_jacobian_right_singular_vectors": np.asarray(
            result.ridge["whitened_right_singular_vectors"]
        ),
        "parameter_displacement_ridge_coordinates": np.asarray(
            ridge_diagnostics["parameter_displacement_ridge_coordinates"]
        ),
        "parameter_displacement_ridge_energy_fraction": np.asarray(
            ridge_diagnostics[
                "parameter_displacement_ridge_energy_fraction"
            ]
        ),
        "uncertainty_variance_naive_in_ridge_basis": np.asarray(
            overlap["uncertainty_variance_naive_in_ridge_basis"]
        ),
        "uncertainty_variance_overlap_in_ridge_basis": np.asarray(
            overlap["uncertainty_variance_overlap_in_ridge_basis"]
        ),
        "uncertainty_variance_inflation_in_ridge_basis": np.asarray(
            overlap["uncertainty_variance_inflation_in_ridge_basis"]
        ),
        "force_acceleration_closure_error": np.asarray(
            closure["force_acceleration_closure_error"]
        ),
        "torque_acceleration_closure_error": np.asarray(
            closure["torque_acceleration_closure_error"]
        ),
        "lag_candidate_rotor_seconds": np.asarray(lag_rotor),
        "lag_candidate_strict_cost": np.asarray(lag_cost),
        "lag_candidate_selected": np.asarray(lag_selected, dtype=bool),
        "lag_continuation_epsilon": np.asarray(continuation["epsilon"]),
        "lag_continuation_rotor_lag": np.asarray(
            continuation["rotor_lag_seconds"]
        ),
        "lag_continuation_smooth_cost": np.asarray(
            continuation["smooth_cost"]
        ),
        "lag_continuation_strict_cost": np.asarray(
            continuation["strict_cost"]
        ),
        "lag_continuation_command_max_error": np.asarray(
            continuation["command_max_error"]
        ),
        "lag_continuation_absolute_cost_difference": np.asarray(
            continuation["absolute_cost_difference"]
        ),
        "lag_continuation_rotor_lag_step": np.asarray(
            continuation["rotor_lag_step"]
        ),
        "lag_continuation_physical_step_norm": np.asarray(
            continuation["physical_step_norm"]
        ),
        "lag_continuation_lag_jacobian_norm": np.asarray(
            continuation["lag_jacobian_norm"]
        ),
        "lag_continuation_physical_coordinate": np.asarray(
            continuation["physical_coordinate"]
        ),
        "strict_lag_cell_lower": np.asarray(
            lag["strict_lag_cell_lower_seconds"]
        ),
        "strict_lag_cell_upper": np.asarray(
            lag["strict_lag_cell_upper_seconds"]
        ),
        "strict_lag_cell_representative": np.asarray(
            lag["strict_lag_cell_representative_seconds"]
        ),
        "strict_lag_neighbor_cells": np.asarray(
            [
                (row["cell_lower_seconds"], row["cell_upper_seconds"])
                for row in lag["neighbor_cells"]
            ],
            dtype=float,
        ).reshape(-1, 2),
        "strict_lag_neighbor_costs": np.asarray(
            [row["strict_cost"] for row in lag["neighbor_cells"]], dtype=float
        ),
        "quotient_basis": np.asarray(quotient["basis"]),
        "quotient_coordinate": np.asarray(quotient["coordinate"]),
        "quotient_jtj": np.asarray(quotient["jtj"]),
        "quotient_covariance_naive": np.asarray(quotient["covariance_naive"]),
        "quotient_covariance_overlap_corrected": np.asarray(
            quotient["covariance_overlap_corrected"]
        ),
        "quotient_covariance_wrench_corrected": np.asarray(
            quotient["covariance_wrench_corrected"]
        ),
        "parameter_covariance_naive": np.asarray(result.uncertainty.naive),
        "parameter_covariance_overlap_corrected": np.asarray(
            result.uncertainty.overlap_corrected
        ),
        "parameter_covariance_wrench_corrected": np.asarray(
            result.uncertainty.wrench_corrected
        ),
        "parameter_sandwich_middle_sg": np.asarray(
            result.uncertainty.sandwich_middle
        ),
        "parameter_sandwich_middle_wrench": np.asarray(
            result.uncertainty.sandwich_middle_wrench
        ),
        "parameter_sandwich_middle_total": np.asarray(
            result.uncertainty.sandwich_middle_total
        ),
        "measured_gyro": np.asarray(dataset.measured_gyro),
        "measured_specific_force": np.asarray(dataset.measured_specific_force),
    }
    if replay is not None:
        arrays.update(
            {
                "fitted_external_wrench": np.asarray(
                    replay.fitted_external_wrench
                ),
                "reconstructed_cog_position": np.asarray(
                    replay.reconstructed_cog_position
                ),
                "reconstructed_body_orientation_xyzw": np.asarray(
                    replay.reconstructed_body_orientation_xyzw
                ),
                "reconstructed_cog_velocity": np.asarray(
                    replay.reconstructed_cog_velocity
                ),
                "reconstructed_body_angular_velocity": np.asarray(
                    replay.reconstructed_body_angular_velocity
                ),
                "reconstructed_sensor_position": np.asarray(
                    replay.reconstructed_sensor_position
                ),
                "reconstructed_sensor_orientation_xyzw": np.asarray(
                    replay.reconstructed_sensor_orientation_xyzw
                ),
                "free_cog_position": np.asarray(replay.free_cog_position),
                "free_body_orientation_xyzw": np.asarray(
                    replay.free_body_orientation_xyzw
                ),
                "free_cog_velocity": np.asarray(replay.free_cog_velocity),
                "free_body_angular_velocity": np.asarray(
                    replay.free_body_angular_velocity
                ),
                "free_sensor_position": np.asarray(replay.free_sensor_position),
                "free_sensor_orientation_xyzw": np.asarray(
                    replay.free_sensor_orientation_xyzw
                ),
                "predicted_gyro": np.asarray(replay.predicted_gyro),
                "predicted_specific_force": np.asarray(
                    replay.predicted_specific_force
                ),
            }
        )
    return arrays


def _residual_wrench_history_figure(
    *,
    case_name: str,
    time_axis: np.ndarray,
    result: EstimationResult,
) -> plt.Figure:
    residual = result.residual_wrench_uncertainty
    wrench = np.asarray(residual.wrench)
    mean = np.asarray(residual.mean)
    standard_deviation = np.asarray(residual.empirical_std)
    names = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    units = ("N", "N", "N", "Nm", "Nm", "Nm")
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
    for component, axis in enumerate(axes.flat):
        axis.plot(time_axis, wrench[:, component], lw=0.9, label="raw residual")
        axis.axhline(mean[component], color="tab:orange", lw=1.2, label="mean")
        axis.fill_between(
            time_axis,
            mean[component] - standard_deviation[component],
            mean[component] + standard_deviation[component],
            color="tab:orange",
            alpha=0.18,
            label="mean +/- 1 sigma total",
        )
        axis.set_ylabel("{} [{}]".format(names[component], units[component]))
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    force_rms = float(np.sqrt(np.mean(np.sum(wrench[:, :3] ** 2, axis=1))))
    torque_rms = float(np.sqrt(np.mean(np.sum(wrench[:, 3:] ** 2, axis=1))))
    figure.suptitle(
        (
            "{}: raw Newton--Euler residual wrench (not trajectory-fitted external wrench)\n"
            "mass gauge fixed to nominal mass = {:.9g} kg"
        ).format(case_name, residual.fixed_mass_kg)
    )
    figure.text(
        0.5,
        0.012,
        (
            "vector RMS about zero: force {:.5g} N, torque {:.5g} Nm; "
            "component std about mean: force {}, torque {}"
        ).format(
            force_rms,
            torque_rms,
            np.array2string(standard_deviation[:3], precision=4),
            np.array2string(standard_deviation[3:], precision=4),
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.93))
    return figure


def _residual_wrench_covariance_figure(
    *, case_name: str, result: EstimationResult
) -> plt.Figure:
    residual = result.residual_wrench_uncertainty
    labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    columns = (
        (
            "empirical total",
            np.asarray(residual.empirical_covariance),
            np.asarray(residual.empirical_correlation),
        ),
        (
            "mean full-SG prediction",
            np.asarray(residual.sg_covariance_mean),
            np.asarray(residual.sg_correlation),
        ),
        (
            "PSD excess model discrepancy",
            np.asarray(residual.model_discrepancy_covariance),
            np.asarray(residual.model_discrepancy_correlation),
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.2))
    for column, (title, covariance, correlation) in enumerate(columns):
        covariance_image = axes[0, column].imshow(covariance, cmap="coolwarm")
        correlation_image = axes[1, column].imshow(
            correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0
        )
        axes[0, column].set_title("{} covariance".format(title), fontsize=9)
        axes[1, column].set_title("{} correlation".format(title), fontsize=9)
        figure.colorbar(
            covariance_image,
            ax=axes[0, column],
            fraction=0.046,
            pad=0.04,
            label="component-unit product",
        )
        figure.colorbar(
            correlation_image,
            ax=axes[1, column],
            fraction=0.046,
            pad=0.04,
        )
        for row in range(2):
            axes[row, column].set_xticks(np.arange(6))
            axes[row, column].set_yticks(np.arange(6))
            axes[row, column].set_xticklabels(labels)
            axes[row, column].set_yticklabels(labels)
    figure.suptitle(
        (
            "{}: residual-wrench covariance decomposition in nominal-mass gauge\n"
            "force [N], torque [Nm]; raw excess eigenvalues {}"
        ).format(
            case_name,
            np.array2string(
                residual.excess_covariance_raw_eigenvalues, precision=3
            ),
        )
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def write_residual_wrench_pdf(
    path: Path,
    *,
    case_name: str,
    dataset: SingleBagDataset,
    result: EstimationResult,
) -> None:
    """Write the two-page standalone residual-wrench scientific report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.asarray(dataset.time) - float(dataset.time[0])
    with PdfPages(path) as pdf:
        for figure in (
            _residual_wrench_history_figure(
                case_name=case_name, time_axis=time_axis, result=result
            ),
            _residual_wrench_covariance_figure(
                case_name=case_name, result=result
            ),
        ):
            pdf.savefig(figure)
            plt.close(figure)


def write_report_pdf(
    path: Path,
    *,
    case_name: str,
    dataset: SingleBagDataset,
    model: VehicleModelInput,
    result: EstimationResult,
    replay: Optional[WrenchReplayResult],
) -> None:
    """Write the twelve-page measured-gimbal/one-lag diagnostic report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.asarray(dataset.time) - float(dataset.time[0])
    labels = "xyz"
    with PdfPages(path) as pdf:
        # Page 1 -- SG trajectory.
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
        trajectory_arrays = (
            np.asarray(dataset.reference_sg.sensor_position),
            np.asarray(dataset.reference_sg.body_angular_velocity),
            np.asarray(dataset.reference_sg.body_angular_acceleration),
        )
        ylabels = ("pose position [m]", "body omega [rad/s]", "body alpha [rad/s^2]")
        for axis, values, ylabel in zip(axes, trajectory_arrays, ylabels):
            for component in range(3):
                axis.plot(time_axis, values[:, component], label=labels[component])
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes[0].legend(loc="best", ncol=3)
        axes[-1].set_xlabel("time [s]")
        figure.suptitle("{}: SG trajectory and derived motion".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Pages 2--3 -- first-class residual-wrench diagnostics.
        for figure in (
            _residual_wrench_history_figure(
                case_name=case_name, time_axis=time_axis, result=result
            ),
            _residual_wrench_covariance_figure(
                case_name=case_name, result=result
            ),
        ):
            pdf.savefig(figure)
            plt.close(figure)

        # Page 4 -- acceleration objective.
        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        observed = np.asarray(dataset.covariance.z)
        predicted = np.column_stack(
            (
                result.evaluation.predicted_specific_acceleration,
                result.evaluation.predicted_angular_acceleration,
            )
        )
        for column in range(2):
            for component in range(3):
                axis = axes[component, column]
                index = 3 * column + component
                axis.plot(time_axis, observed[:, index], label="SG observed")
                axis.plot(time_axis, predicted[:, index], "--", label="model")
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.3)
        axes[0, 0].set_title("specific acceleration [m/s^2]")
        axes[0, 1].set_title("angular acceleration [rad/s^2]")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 1].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle("{}: acceleration residual objective".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 3 -- measured gimbal audit.
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        raw_time = np.asarray(dataset.gimbal_raw_time) - float(dataset.time[0])
        replay_angle = np.asarray(
            result.diagnostics["gimbal"]["command_replay_angle"]
        )
        rmse = np.asarray(
            result.diagnostics["gimbal"]["command_replay_rmse_rad"]
        )
        maximum = np.asarray(
            result.diagnostics["gimbal"]["command_replay_max_abs_error_rad"]
        )
        for joint, axis in enumerate(axes.flat):
            axis.plot(raw_time, dataset.gimbal_raw_angle[:, joint], ".", ms=2, label="raw")
            axis.plot(time_axis, dataset.gimbal_sg_angle[:, joint], "-", label="SG")
            axis.plot(time_axis, replay_angle[:, joint], "--", label="command replay")
            axis.set_title(
                "gimbal {}: RMSE={:.4g}, max={:.4g} rad".format(
                    joint + 1, rmse[joint], maximum[joint]
                )
            )
            axis.set_ylabel("angle [rad]")
            axis.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle(
            "{}: gimbal measurement audit (objective={})".format(
                case_name, result.diagnostics["gimbal"]["objective_source"]
            )
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 4 -- IMU comparison.
        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        measured = (dataset.measured_gyro, dataset.measured_specific_force)
        predicted_imu = (
            (np.full_like(measured[0], np.nan), np.full_like(measured[1], np.nan))
            if replay is None
            else (replay.predicted_gyro, replay.predicted_specific_force)
        )
        for column in range(2):
            for component in range(3):
                axis = axes[component, column]
                axis.plot(time_axis, measured[column][:, component], label="measured")
                axis.plot(time_axis, predicted_imu[column][:, component], "--", label="post-fit")
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.3)
        axes[0, 0].set_title("gyro [rad/s]")
        axes[0, 1].set_title("specific force [m/s^2]")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 1].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle("{}: IMU comparison (diagnostic only)".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 5 -- power-of-two continuation.
        continuation = result.diagnostics["continuation"]
        epsilon = np.asarray(continuation["epsilon"])
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
        if epsilon.size:
            index = np.arange(epsilon.size)
            axes[0].plot(index, continuation["rotor_lag_seconds"], "o-")
            axes[1].plot(index, continuation["smooth_cost"], "o-", label="smooth")
            axes[1].plot(index, continuation["strict_cost"], "s--", label="strict at same point")
            gap = np.asarray(continuation["absolute_cost_difference"])
            command_error = np.asarray(continuation["command_max_error"])
            axes[2].semilogy(index, np.maximum(gap, np.finfo(float).tiny), "o-", label="|L_eps-L_0|")
            axes[2].semilogy(index, np.maximum(command_error, np.finfo(float).tiny), "s--", label="max thrust error")
            axes[1].legend(loc="best")
            axes[2].legend(loc="best")
            axes[-1].set_xticks(index)
            axes[-1].set_xticklabels(["{:.3g}".format(value) for value in epsilon])
        else:
            for axis in axes:
                axis.text(0.5, 0.5, "continuation disabled for this case", ha="center", va="center", transform=axis.transAxes)
        axes[0].set_ylabel("rotor lag [s]")
        axes[1].set_ylabel("objective")
        axes[2].set_ylabel("convergence error")
        axes[2].set_xlabel("epsilon = 2^-k")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.suptitle("{}: smooth-to-strict continuation".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 6 -- exact strict lag cells.
        lag = result.diagnostics["lag"]
        table = lag["candidate_table"]
        figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0))
        if table:
            representative = np.asarray([row["representative_seconds"] for row in table])
            costs = np.asarray([row["strict_cost"] for row in table])
            selected = np.asarray([row["selected"] for row in table], dtype=bool)
            axes[0].plot(representative, costs, "o-")
            axes[0].plot(representative[selected], costs[selected], "r*", ms=14, label="selected")
            for row_index, row in enumerate(table):
                color = "tab:red" if row["selected"] else "tab:blue"
                axes[1].plot(
                    [row["cell_lower_seconds"], row["cell_upper_seconds"]],
                    [row_index, row_index],
                    color=color,
                    lw=4,
                )
            axes[0].legend(loc="best")
        axes[0].set_xlabel("cell representative lag [s]")
        axes[0].set_ylabel("profiled strict cost")
        axes[1].set_xlabel("strict cell interval (lower, upper] [s]")
        axes[1].set_ylabel("evaluated cell")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.text(
            0.5,
            0.01,
            "selected ({:.9g}, {:.9g}] s; final smooth {:.9g} s; data boundary={}".format(
                lag["strict_lag_cell_lower_seconds"],
                lag["strict_lag_cell_upper_seconds"],
                lag["final_smooth_rotor_lag_seconds"],
                lag["lag_reached_data_support_boundary"],
            ),
            ha="center",
        )
        figure.suptitle("{}: exact strict-ZOH lag cells".format(case_name))
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        pdf.savefig(figure)
        plt.close(figure)

        # Page 7 -- physical and scale-free parameters.
        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        nominal = model.parameters
        estimated = result.evaluation.parameters
        lines = [
            "Case: {}".format(case_name),
            "status: {} ({})".format(result.status, result.message),
            "",
            "mass [kg]: nominal {:.9g}; estimated {:.9g}".format(nominal.mass, estimated.mass),
            "inertia [kg m^2]:",
            np.array2string(estimated.inertia, precision=8),
            "CoG body [m]: {}".format(np.array2string(estimated.cog_offset, precision=8)),
            "force effectiveness: {}".format(np.array2string(estimated.force_effectiveness, precision=8)),
            "",
            "scale-free J/m [m^2]:",
            np.array2string(estimated.inertia / estimated.mass, precision=8),
            "scale-free f/m: {}".format(np.array2string(estimated.force_effectiveness / estimated.mass, precision=8)),
            "",
            "rotor lag cell: ({:.9g}, {:.9g}] s".format(
                lag["strict_lag_cell_lower_seconds"], lag["strict_lag_cell_upper_seconds"]
            ),
            "representative: {:.9g} s".format(result.rotor_lag_seconds),
        ]
        axis.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=9)
        figure.suptitle("{}: parameter estimate".format(case_name))
        pdf.savefig(figure)
        plt.close(figure)

        # Page 8 -- ridge spectrum.
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
        white = np.asarray(result.ridge["whitened_singular_values"])
        raw = np.asarray(result.ridge["unwhitened_diagnostic_singular_values"])
        direction = np.arange(white.size)
        axes[0, 0].semilogy(direction, np.maximum(white, np.finfo(float).tiny), "o-", label="optimization")
        axes[0, 0].semilogy(direction, np.maximum(raw, np.finfo(float).tiny), "s--", label="unwhitened")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 1].bar(direction, result.diagnostics["ridge"]["parameter_displacement_ridge_coordinates"])
        axes[1, 0].bar(direction, result.diagnostics["ridge"]["parameter_displacement_ridge_energy_fraction"])
        axes[1, 1].bar(direction, result.diagnostics["overlap_correction"]["uncertainty_variance_inflation_in_ridge_basis"])
        axes[0, 0].set_ylabel("singular value")
        axes[0, 1].set_ylabel("V^T delta q")
        axes[1, 0].set_ylabel("displacement fraction")
        axes[1, 1].set_ylabel("overlap / naive variance")
        for axis in axes.flat:
            axis.set_xlabel("ridge direction")
            axis.grid(True, alpha=0.3)
        figure.suptitle(
            "{}: ridge spectrum (rank {}, nullity {}, ||Jv||={:.3g})".format(
                case_name,
                result.ridge["machine_numerical_rank"],
                result.ridge["nullity"],
                result.ridge["j_v_scale_norm"],
            )
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 9 -- covariance and uncertainty.
        covariance = result.diagnostics["covariance"]
        reference_covariance = result.diagnostics["reference_full_covariance"]
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
        eigenvalues = np.asarray(reference_covariance["sigma_z_eigenvalues"])
        gains = np.asarray(covariance["whitening_gain"])
        for mode in range(eigenvalues.shape[1]):
            axes[0].semilogy(time_axis, np.maximum(eigenvalues[:, mode], np.finfo(float).tiny))
            axes[1].semilogy(time_axis, np.maximum(gains[:, mode], np.finfo(float).tiny))
        axes[2].plot(
            time_axis,
            covariance["mahalanobis_contribution_per_time"],
            label="optimization metric",
        )
        axes[2].plot(
            time_axis,
            reference_covariance["mahalanobis_contribution_per_time"],
            "--",
            label="reference full covariance",
        )
        axes[0].set_ylabel("full Sigma_z eigenvalue")
        axes[1].set_ylabel("optimization whitening gain")
        axes[2].set_ylabel("Mahalanobis contribution")
        axes[2].set_xlabel("time [s]")
        axes[2].legend(loc="best", fontsize=8)
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.suptitle("{}: optimization vs reference covariance".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        # Page 10 -- wrench and Newton--Euler closure.
        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        raw_wrench = np.asarray(result.evaluation.raw_residual_wrench)
        fitted = np.full_like(raw_wrench, np.nan) if replay is None else replay.fitted_external_wrench
        for component, axis in enumerate(axes.flat):
            axis.plot(time_axis, raw_wrench[:, component], label="raw residual")
            axis.plot(time_axis, fitted[:, component], "--", label="trajectory-fitted")
            axis.set_ylabel(("F", "F", "F", "T", "T", "T")[component] + labels[component % 3])
            axis.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        closure = result.diagnostics["closure"]
        figure.suptitle(
            "{}: wrench / closure (force max {:.3g}, torque max {:.3g})".format(
                case_name,
                closure["force_acceleration_closure_error_max_abs"],
                closure["torque_acceleration_closure_error_max_abs"],
            )
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def write_failure_report_pdf(
    path: Path,
    *,
    case_name: str,
    failure_stage: str,
    exception_type: str,
    message: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(10.0, 7.0))
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(
            0.05,
            0.95,
            "status: failed\ncase: {}\nfailure stage: {}\nexception: {}\nmessage: {}".format(
                case_name, failure_stage, exception_type, message
            ),
            va="top",
            family="monospace",
        )
        pdf.savefig(figure)
        plt.close(figure)


def write_completed_case(
    directory: Path,
    *,
    case_name: str,
    source_revision: str,
    arguments: Mapping[str, Any],
    dataset: SingleBagDataset,
    model: VehicleModelInput,
    result: EstimationResult,
    replay: Optional[WrenchReplayResult],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = result_payload(
        case_name=case_name,
        source_revision=source_revision,
        model=model,
        result=result,
        replay=replay,
    )
    status = {
        "status": payload["status"],
        "case_name": case_name,
        "source_commit": source_revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "message": result.message,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if not result.success:
        status.update(
            {
                "failure_stage": result.first_failure_stage,
                "exception_type": "OptimizerNonSuccess",
                "failure_reason": result.message,
                "first_failure_stage": result.first_failure_stage,
                "first_failure_status": result.first_failure_status,
                "first_failure_message": result.first_failure_message,
            }
        )
    write_json(directory / "status.json", status)
    write_json(directory / "result.json", payload)
    write_json(directory / "arguments.json", dict(arguments))
    write_json(
        directory / "timing.json",
        {
            "elapsed_seconds": result.elapsed_seconds,
            "nfev": result.total_nfev,
            "stages": result.stages,
        },
    )
    np.savez_compressed(directory / "arrays.npz", **arrays_payload(dataset, result, replay))
    write_report_pdf(
        directory / "report.pdf",
        case_name=case_name,
        dataset=dataset,
        model=model,
        result=result,
        replay=replay,
    )
    write_residual_wrench_pdf(
        directory / "residual_wrench.pdf",
        case_name=case_name,
        dataset=dataset,
        result=result,
    )
    return payload
