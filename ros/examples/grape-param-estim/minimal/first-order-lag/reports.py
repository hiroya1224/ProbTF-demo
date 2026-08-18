#!/usr/bin/env python3
"""Standard diagnostic artifacts for the isolated first-order-lag estimator."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import lombscargle

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-first-order-lag-matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from single_bag_savgol_reports import write_json  # noqa: E402


STANDARD_OUTPUT_FILENAMES = (
    "arguments.json",
    "arrays.npz",
    "estimate.json",
    "report.pdf",
    "residual_wrench.pdf",
    "residual_wrench_summary.json",
    "result.json",
    "status.json",
    "timing.json",
)

WRENCH_LABELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
WRENCH_UNITS = ("N", "N", "N", "N m", "N m", "N m")
ACCELERATION_LABELS = ("sx", "sy", "sz", "alphax", "alphay", "alphaz")
ACCELERATION_UNITS = (
    "m/s^2",
    "m/s^2",
    "m/s^2",
    "rad/s^2",
    "rad/s^2",
    "rad/s^2",
)
DEFAULT_SPECTRUM_MIN_HZ = 0.05
DEFAULT_SPECTRUM_MAX_HZ = 5.0
DEFAULT_SPECTRUM_SIZE = 3000


def _plain_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in arguments.items() if not str(key).startswith("_")}


def _linear_detrend(time: np.ndarray, value: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    y = np.asarray(value, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or t.size < 2:
        raise ValueError("detrend inputs must be aligned one-dimensional arrays")
    centered = t - float(np.mean(t))
    design = np.column_stack((np.ones_like(centered), centered))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficient


def wrench_lomb_scargle(
    time: Sequence[float],
    residual_wrench: np.ndarray,
    *,
    minimum_hz: float = DEFAULT_SPECTRUM_MIN_HZ,
    maximum_hz: float = DEFAULT_SPECTRUM_MAX_HZ,
    size: int = DEFAULT_SPECTRUM_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    wrench = np.asarray(residual_wrench, dtype=float)
    if (
        t.ndim != 1
        or wrench.shape != (t.size, 6)
        or t.size < 3
        or np.any(~np.isfinite(t))
        or np.any(~np.isfinite(wrench))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("residual wrench spectrum inputs are invalid")
    low = float(minimum_hz)
    high = float(maximum_hz)
    count = int(size)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low or count < 2:
        raise ValueError("spectrum grid is invalid")
    frequency = np.linspace(low, high, count)
    angular = 2.0 * np.pi * frequency
    relative = t - t[0]
    power = np.zeros((count, 6), dtype=float)
    peaks = np.full(6, np.nan, dtype=float)
    for component in range(6):
        selected = _linear_detrend(relative, wrench[:, component])
        scale = float(np.linalg.norm(selected))
        if scale == 0.0:
            continue
        spectrum = lombscargle(relative, selected, angular, normalize=True)
        spectrum = np.asarray(spectrum, dtype=float)
        if np.any(~np.isfinite(spectrum)):
            raise FloatingPointError("Lomb-Scargle spectrum became non-finite")
        power[:, component] = spectrum
        peaks[component] = float(frequency[int(np.argmax(spectrum))])
    return frequency, power, peaks


def residual_wrench_summary(time: Sequence[float], residual_wrench: np.ndarray) -> Mapping[str, Any]:
    t = np.asarray(time, dtype=float)
    wrench = np.asarray(residual_wrench, dtype=float)
    frequency, power, peaks = wrench_lomb_scargle(t, wrench)
    del frequency, power
    rms = np.sqrt(np.mean(wrench**2, axis=0))
    mean = np.mean(wrench, axis=0)
    return {
        "component_labels": list(WRENCH_LABELS),
        "component_units": list(WRENCH_UNITS),
        "mean": mean.tolist(),
        "rms": rms.tolist(),
        "force_vector_rms_n": float(np.sqrt(np.mean(np.sum(wrench[:, :3] ** 2, axis=1)))),
        "torque_vector_rms_nm": float(np.sqrt(np.mean(np.sum(wrench[:, 3:] ** 2, axis=1)))),
        "lomb_scargle": {
            "minimum_hz": DEFAULT_SPECTRUM_MIN_HZ,
            "maximum_hz": DEFAULT_SPECTRUM_MAX_HZ,
            "grid_size": DEFAULT_SPECTRUM_SIZE,
            "linear_trend_removed": True,
            "peak_frequency_hz": peaks.tolist(),
        },
    }


def _arrays_payload(
    *,
    estimate: Mapping[str, Any],
    dataset: Any,
    evaluation: Any,
) -> Mapping[str, np.ndarray]:
    time = np.asarray(dataset.time, dtype=float)
    relative = time - float(time[0])
    observed = np.asarray(dataset.covariance.z, dtype=float)
    predicted = np.column_stack(
        (
            np.asarray(evaluation.predicted_specific_acceleration, dtype=float),
            np.asarray(evaluation.predicted_angular_acceleration, dtype=float),
        )
    )
    command_time = np.asarray(dataset.rotor_history.times, dtype=float)
    command = np.asarray(dataset.rotor_history.values, dtype=float)
    distribution = estimate["plant_distribution"]
    covariance = distribution["covariances"]
    return {
        "time": time,
        "time_relative": relative,
        "observed_generalized_acceleration": observed,
        "predicted_generalized_acceleration": predicted,
        "acceleration_residual": np.asarray(evaluation.acceleration_residual, dtype=float),
        "raw_residual_wrench": np.asarray(evaluation.raw_residual_wrench, dtype=float),
        "modeled_wrench": np.asarray(evaluation.modeled_wrench, dtype=float),
        "required_wrench": np.asarray(evaluation.required_wrench, dtype=float),
        "rotor_command_time": command_time,
        "rotor_command_time_relative": command_time - float(time[0]),
        "rotor_command_thrust": command,
        "actual_thrust": np.asarray(evaluation.actuator_history.actual_thrust, dtype=float),
        "actual_thrust_log_tau_jacobian": np.asarray(
            evaluation.actuator_history.actual_thrust_lag_jacobian, dtype=float
        ),
        "actual_gimbal": np.asarray(evaluation.actuator_history.actual_gimbal, dtype=float),
        "physical_chart_coordinate": np.asarray(
            distribution["physical_chart_coordinate"], dtype=float
        ),
        "quotient_basis": np.asarray(distribution["quotient_basis"], dtype=float),
        "quotient_coordinate": np.asarray(distribution["quotient_coordinate"], dtype=float),
        "covariance_naive": np.asarray(covariance["naive"], dtype=float),
        "covariance_overlap_corrected": np.asarray(
            covariance["overlap_corrected"], dtype=float
        ),
        "covariance_conservative_fusion": np.asarray(
            covariance["conservative_fusion"], dtype=float
        ),
        "thrust_time_constant_seconds": np.asarray(
            float(estimate["actuator_model"]["thrust_time_constant_seconds"])
        ),
        "sensor_position": np.asarray(dataset.reference_sg.sensor_position, dtype=float),
        "body_angular_velocity": np.asarray(
            dataset.reference_sg.body_angular_velocity, dtype=float
        ),
        "body_angular_acceleration": np.asarray(
            dataset.reference_sg.body_angular_acceleration, dtype=float
        ),
        "measured_gyro": np.asarray(dataset.measured_gyro, dtype=float),
        "measured_specific_force": np.asarray(dataset.measured_specific_force, dtype=float),
    }


def _wrench_history_figure(case_name: str, time: np.ndarray, wrench: np.ndarray) -> plt.Figure:
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    for component, axis in enumerate(axes.flat):
        axis.plot(time, wrench[:, component])
        axis.set_ylabel("{} [{}]".format(WRENCH_LABELS[component], WRENCH_UNITS[component]))
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    figure.suptitle("{}: first-order-model residual wrench".format(case_name))
    figure.tight_layout()
    return figure


def _wrench_spectrum_figure(
    case_name: str,
    frequency: np.ndarray,
    power: np.ndarray,
    peaks: np.ndarray,
) -> plt.Figure:
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    for component, axis in enumerate(axes.flat):
        axis.plot(frequency, power[:, component])
        if np.isfinite(peaks[component]):
            axis.axvline(peaks[component], linestyle="--", linewidth=1.0)
            axis.set_title("peak {:.4g} Hz".format(peaks[component]))
        axis.set_ylabel(WRENCH_LABELS[component])
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("frequency [Hz]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    figure.suptitle("{}: residual-wrench Lomb-Scargle spectrum".format(case_name))
    figure.tight_layout()
    return figure


def write_residual_wrench_pdf(path: Path, *, case_name: str, dataset: Any, evaluation: Any) -> None:
    relative = np.asarray(dataset.time, dtype=float) - float(dataset.time[0])
    wrench = np.asarray(evaluation.raw_residual_wrench, dtype=float)
    frequency, power, peaks = wrench_lomb_scargle(dataset.time, wrench)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for figure in (
            _wrench_history_figure(case_name, relative, wrench),
            _wrench_spectrum_figure(case_name, frequency, power, peaks),
        ):
            pdf.savefig(figure)
            plt.close(figure)


def write_report_pdf(
    path: Path,
    *,
    case_name: str,
    estimate: Mapping[str, Any],
    dataset: Any,
    model: Any,
    evaluation: Any,
) -> None:
    relative = np.asarray(dataset.time, dtype=float) - float(dataset.time[0])
    command_time = np.asarray(dataset.rotor_history.times, dtype=float) - float(dataset.time[0])
    command = np.asarray(dataset.rotor_history.values, dtype=float)
    actual = np.asarray(evaluation.actuator_history.actual_thrust, dtype=float)
    observed = np.asarray(dataset.covariance.z, dtype=float)
    predicted = np.column_stack(
        (
            np.asarray(evaluation.predicted_specific_acceleration, dtype=float),
            np.asarray(evaluation.predicted_angular_acceleration, dtype=float),
        )
    )
    wrench = np.asarray(evaluation.raw_residual_wrench, dtype=float)
    frequency, power, peaks = wrench_lomb_scargle(dataset.time, wrench)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        for rotor, axis in enumerate(axes.flat):
            axis.step(command_time, command[:, rotor], where="post", label="issued ZOH")
            axis.plot(relative, actual[:, rotor], label="first-order response")
            axis.set_ylabel("thrust {} [N]".format(rotor + 1))
            axis.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle(
            "{}: command and fitted first-order thrust, tau={:.6g} s".format(
                case_name, float(estimate["actuator_model"]["thrust_time_constant_seconds"])
            )
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = _wrench_history_figure(case_name, relative, wrench)
        pdf.savefig(figure)
        plt.close(figure)
        figure = _wrench_spectrum_figure(case_name, frequency, power, peaks)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
        for component, axis in enumerate(axes.flat):
            axis.plot(relative, observed[:, component], label="SG observed")
            axis.plot(relative, predicted[:, component], "--", label="model")
            axis.set_ylabel(
                "{} [{}]".format(ACCELERATION_LABELS[component], ACCELERATION_UNITS[component])
            )
            axis.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle("{}: acceleration objective".format(case_name))
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        estimated = estimate["point_estimate"]
        nominal = model.parameters
        initialization = estimate["actuator_model"]["initialization"]
        lines = [
            "case: {}".format(case_name),
            "outcome metadata: {}".format(estimate.get("flight_outcome", "unspecified")),
            "",
            "first-order thrust time constant [s]: {:.12g}".format(
                estimate["actuator_model"]["thrust_time_constant_seconds"]
            ),
            "pure delay [s]: {:.12g}".format(estimate["actuator_model"]["pure_delay_seconds"]),
            "initial tau [s]: {}".format(initialization.get("initial_tau_seconds")),
            "controller command median period [s]: {:.12g}".format(
                estimate["controller_timing"]["median_seconds"]
            ),
            "",
            "data objective: {:.12g}".format(estimate["optimization"]["data_cost"]),
            "residual RMS: {:.12g}".format(estimate["optimization"]["residual_rms"]),
            "nfev: {}".format(estimate["optimization"]["nfev"]),
            "",
            "mass [kg]: nominal {:.12g}; estimated {:.12g}".format(
                nominal.mass, estimated["mass_kg"]
            ),
            "CoG body [m]: {}".format(
                np.array2string(np.asarray(estimated["cog_position_body_m"]), precision=8)
            ),
            "force effectiveness: {}".format(
                np.array2string(np.asarray(estimated["force_effectiveness"]), precision=8)
            ),
            "J/m [m^2]:",
            np.array2string(
                np.asarray(estimated["scale_free"]["inertia_over_mass_m2"]), precision=8
            ),
            "",
            "residual-wrench peak frequencies [Hz]:",
            np.array2string(peaks, precision=6),
        ]
        axis.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        figure.suptitle("{}: first-order-lag estimate summary".format(case_name))
        pdf.savefig(figure)
        plt.close(figure)


def write_failure_report_pdf(path: Path, *, failure: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(10.0, 7.0))
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(
            0.05,
            0.95,
            "status: failed\ncase: {}\nstage: {}\nexception: {}\nmessage: {}".format(
                failure.get("case_name"),
                failure.get("failure_stage"),
                failure.get("exception_type"),
                failure.get("message"),
            ),
            va="top",
            family="monospace",
        )
        pdf.savefig(figure)
        plt.close(figure)


def write_standard_outputs(
    directory: Path,
    *,
    estimate: Mapping[str, Any],
    arguments: Mapping[str, Any],
    dataset: Any,
    model: Any,
    evaluation: Any,
    solver_runs: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    case_name = str(estimate["case_name"])
    write_json(directory / "result.json", dict(estimate))
    write_json(directory / "arguments.json", _plain_arguments(arguments))
    write_json(
        directory / "timing.json",
        {
            "elapsed_seconds": float(estimate["elapsed_seconds"]),
            "nfev": int(estimate["optimization"]["nfev"]),
            "solver_runs": list(solver_runs),
            "initial_tau_seconds": estimate["actuator_model"]["initialization"].get(
                "initial_tau_seconds"
            ),
            "final_tau_seconds": float(
                estimate["actuator_model"]["thrust_time_constant_seconds"]
            ),
        },
    )
    np.savez_compressed(
        directory / "arrays.npz",
        **_arrays_payload(estimate=estimate, dataset=dataset, evaluation=evaluation),
    )
    summary = residual_wrench_summary(dataset.time, evaluation.raw_residual_wrench)
    write_json(directory / "residual_wrench_summary.json", summary)
    write_residual_wrench_pdf(
        directory / "residual_wrench.pdf",
        case_name=case_name,
        dataset=dataset,
        evaluation=evaluation,
    )
    write_report_pdf(
        directory / "report.pdf",
        case_name=case_name,
        estimate=estimate,
        dataset=dataset,
        model=model,
        evaluation=evaluation,
    )
    return STANDARD_OUTPUT_FILENAMES


def write_failure_outputs(
    directory: Path,
    *,
    failure: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "result.json", dict(failure))
    write_json(directory / "arguments.json", _plain_arguments(arguments))
    write_json(
        directory / "timing.json",
        {"elapsed_seconds": float(failure.get("elapsed_seconds", 0.0))},
    )
    write_failure_report_pdf(directory / "report.pdf", failure=failure)
