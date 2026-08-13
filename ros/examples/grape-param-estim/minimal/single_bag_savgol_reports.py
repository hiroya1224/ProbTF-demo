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
        "gimbal_lag_seconds": result.gimbal_lag_seconds,
    }


def common_metrics(
    result: EstimationResult, replay: Optional[WrenchReplayResult]
) -> dict[str, Any]:
    reference = result.reference_evaluation
    acceleration = reference.acceleration_residual
    metrics: dict[str, Any] = {
        "reference_objective_sum": reference.cost,
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
        },
    }
    if not result.success:
        payload.update(
            {
                "failure_stage": "parameter_estimation",
                "exception_type": "OptimizerNonSuccess",
                "failure_reason": result.message,
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
        "modeled_wrench": np.asarray(evaluation.modeled_wrench),
        "required_wrench": np.asarray(evaluation.required_wrench),
        "raw_residual_wrench": np.asarray(evaluation.raw_residual_wrench),
        "residual_acceleration": np.asarray(evaluation.acceleration_residual),
        "whitened_residual": np.asarray(evaluation.whitened_residual),
        "raw_physical_jacobian": np.asarray(evaluation.acceleration_jacobian),
        "whitened_physical_jacobian": np.asarray(evaluation.whitened_jacobian),
        "raw_jacobian": np.asarray(result.ridge["raw_acceleration_jacobian"]),
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
        "parameter_covariance_naive": np.asarray(result.uncertainty.naive),
        "parameter_covariance_overlap_corrected": np.asarray(
            result.uncertainty.overlap_corrected
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


def write_report_pdf(
    path: Path,
    *,
    case_name: str,
    dataset: SingleBagDataset,
    model: VehicleModelInput,
    result: EstimationResult,
    replay: Optional[WrenchReplayResult],
) -> None:
    """Write the standardized four-section report for one completed case."""

    path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.asarray(dataset.time) - float(dataset.time[0])
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
        observed = dataset.reference_sg.sensor_position
        reconstructed = (
            np.full_like(observed, np.nan)
            if replay is None
            else replay.reconstructed_sensor_position
        )
        for component, axis in enumerate(axes):
            axis.plot(
                time_axis,
                observed[:, component],
                "-",
                label="observed trajectory",
            )
            axis.plot(
                time_axis,
                reconstructed[:, component],
                "--",
                label="estimated + fitted-wrench reconstruction",
            )
            axis.set_ylabel("{} [m]".format("xyz"[component]))
            axis.grid(True, alpha=0.3)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("time [s]")
        figure.suptitle("{}: trajectory".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        raw = result.evaluation.raw_residual_wrench
        fitted = np.full_like(raw, np.nan) if replay is None else replay.fitted_external_wrench
        for component, axis in enumerate(axes.flat):
            axis.plot(
                time_axis,
                raw[:, component],
                linestyle=WRENCH_LINE_STYLES["raw_sg_inverse_dynamics"],
                label="raw SG inverse-dynamics residual",
            )
            axis.plot(
                time_axis,
                fitted[:, component],
                linestyle=WRENCH_LINE_STYLES["trajectory_fitted_external"],
                label="trajectory-fitted external wrench",
            )
            unit = "N" if component < 3 else "N m"
            label = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[component]
            axis.set_ylabel("{} [{}]".format(label, unit))
            axis.grid(True, alpha=0.3)
        axes.flat[0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle("{}: residual wrench".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        measured = (dataset.measured_gyro, dataset.measured_specific_force)
        predicted = (
            (np.full_like(measured[0], np.nan), np.full_like(measured[1], np.nan))
            if replay is None
            else (replay.predicted_gyro, replay.predicted_specific_force)
        )
        for column in range(2):
            for component in range(3):
                axis = axes[component, column]
                axis.plot(time_axis, measured[column][:, component], "-", label="measured")
                axis.plot(time_axis, predicted[column][:, component], "--", label="predicted")
                unit = "rad/s" if column == 0 else "m/s^2"
                axis.set_ylabel("{} [{}]".format("xyz"[component], unit))
                axis.grid(True, alpha=0.3)
        axes[0, 0].set_title("gyro")
        axes[0, 1].set_title("specific force")
        axes[0, 0].legend(loc="best")
        axes[0, 1].legend(loc="best")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle("{}: measured sensor comparison".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        nominal = model.parameters
        estimated = result.evaluation.parameters
        lines = [
            "Case: {}".format(case_name),
            "status: {} ({})".format(result.status, result.message),
            "",
            "mass [kg]       nominal {: .9g}   estimated {: .9g}   delta {: .9g}".format(
                nominal.mass, estimated.mass, estimated.mass - nominal.mass
            ),
            "",
            "inertia [kg m^2] (nominal -> estimated; delta)",
        ]
        for row in range(3):
            lines.append(
                "  {} -> {}   d={}".format(
                    np.array2string(nominal.inertia[row], precision=7),
                    np.array2string(estimated.inertia[row], precision=7),
                    np.array2string(
                        estimated.inertia[row] - nominal.inertia[row], precision=7
                    ),
                )
            )
        lines.extend(
            (
                "",
                "CoG [m]       {} -> {}   d={}".format(
                    np.array2string(nominal.cog_offset, precision=7),
                    np.array2string(estimated.cog_offset, precision=7),
                    np.array2string(estimated.cog_offset - nominal.cog_offset, precision=7),
                ),
                "rotor force effectiveness",
                "  {} -> {}   d={}".format(
                    np.array2string(nominal.force_effectiveness, precision=7),
                    np.array2string(estimated.force_effectiveness, precision=7),
                    np.array2string(
                        estimated.force_effectiveness - nominal.force_effectiveness,
                        precision=7,
                    ),
                ),
                "",
                "rotor lag [s] {: .9g}".format(result.rotor_lag_seconds),
                "gimbal lag [s] {: .9g}".format(result.gimbal_lag_seconds),
            )
        )
        axis.text(
            0.03,
            0.97,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        figure.suptitle("{}: nominal -> estimated parameters".format(case_name))
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
                "failure_stage": "parameter_estimation",
                "exception_type": "OptimizerNonSuccess",
                "failure_reason": result.message,
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
    return payload
