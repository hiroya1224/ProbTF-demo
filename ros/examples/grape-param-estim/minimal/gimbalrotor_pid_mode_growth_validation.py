#!/usr/bin/env python3
"""One-shot validation of predicted unstable pole growth against recorded attitude.

This script intentionally lives outside the production local-pole analysis.
For each selected failed flight it:

1. recomputes the center-plant fitted-delay closed-loop Jacobian;
2. selects the dominant oscillatory unstable eigenpair;
3. extracts a phase-invariant orientation direction from its right eigenvector;
4. projects the recorded SO(3) attitude error onto that direction; and
5. fits one exponentially growing/decaying sinusoid directly to that scalar
   signal, comparing its growth rate and frequency with the pole prediction.

The fit is diagnostic only.  It does not feed back into parameter estimation,
pole classification, or PID gain selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gimbalrotor_pid_local_pole_validation as local_poles  # noqa: E402
import three_bag_gimbalrotor_pid_local_pole_validation as three_bag  # noqa: E402
from grape_param_estim.geometry import (  # noqa: E402
    euler_xyz_to_matrix,
    quaternion_to_matrix,
    so3_log,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402


SCHEMA = "grape-param-estim/gimbalrotor-pid-mode-growth-validation/v1"
VALIDATION_BASE_COMMIT = "e640e9d3d80fbd028afd7782a075214ceee9195a"
DEFAULT_CASES = ("failure1", "failure2")
DEFAULT_FREQUENCY_MIN_HZ = 0.2
DEFAULT_FREQUENCY_MAX_HZ = 2.0
DEFAULT_FREQUENCY_GRID_SIZE = 4000


@dataclass(frozen=True)
class PredictedMode:
    eigenvalue: complex
    right_eigenvector: np.ndarray
    orientation_axis: np.ndarray
    orientation_subspace_singular_values: np.ndarray
    growth_rate_per_second: float
    frequency_hz: float
    doubling_time_seconds: Optional[float]

    def __post_init__(self) -> None:
        vector = np.asarray(self.right_eigenvector, dtype=complex)
        axis = np.asarray(self.orientation_axis, dtype=float)
        singular = np.asarray(
            self.orientation_subspace_singular_values, dtype=float
        )
        if (
            vector.ndim != 1
            or axis.shape != (3,)
            or singular.shape != (2,)
            or np.any(~np.isfinite(vector))
            or np.any(~np.isfinite(axis))
            or np.any(~np.isfinite(singular))
            or not np.isclose(np.linalg.norm(axis), 1.0)
        ):
            raise ValueError("predicted mode arrays are invalid")
        object.__setattr__(self, "right_eigenvector", vector.copy())
        object.__setattr__(self, "orientation_axis", axis.copy())
        object.__setattr__(
            self, "orientation_subspace_singular_values", singular.copy()
        )


@dataclass(frozen=True)
class DampedSineFit:
    growth_rate_per_second: float
    frequency_hz: float
    intercept: float
    slope: float
    cosine_coefficient: float
    sine_coefficient: float
    amplitude_at_start: float
    doubling_time_seconds: Optional[float]
    residual_rms: float
    r_squared: float
    initial_frequency_hz: float
    optimizer_status: int
    optimizer_message: str


def _read_json(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("{} must contain a JSON object".format(source))
    return value


def _doubling_time(growth_rate: float) -> Optional[float]:
    value = float(growth_rate)
    if not np.isfinite(value):
        raise ValueError("growth rate must be finite")
    return None if value <= 0.0 else float(math.log(2.0) / value)


def _dominant_oscillatory_mode(
    jacobian: np.ndarray,
    controller_dt: float,
) -> PredictedMode:
    matrix = np.asarray(jacobian, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("closed-loop Jacobian must be finite and square")
    dt = float(controller_dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("controller_dt must be finite and positive")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    imag_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    imag_tolerance = (
        64.0 * np.finfo(float).eps * imag_scale
    )
    positive_imag = np.flatnonzero(eigenvalues.imag > imag_tolerance)
    if positive_imag.size == 0:
        raise RuntimeError("closed-loop center has no oscillatory eigenpair")

    unstable = positive_imag[np.abs(eigenvalues[positive_imag]) > 1.0]
    candidates = unstable if unstable.size else positive_imag
    magnitudes = np.abs(eigenvalues[candidates])
    selected_index = int(candidates[int(np.argmax(magnitudes))])
    value = complex(eigenvalues[selected_index])
    vector = np.asarray(eigenvectors[:, selected_index], dtype=complex)

    orientation = vector[3:6]
    plane = np.column_stack((orientation.real, orientation.imag))
    left, singular, _right = np.linalg.svd(plane, full_matrices=False)
    if singular.size < 1 or singular[0] <= np.finfo(float).tiny:
        raise RuntimeError(
            "selected oscillatory mode has no observable orientation component"
        )
    if singular.size == 1:
        singular = np.asarray((singular[0], 0.0))
    axis = left[:, 0]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis

    radius = float(abs(value))
    angle = float(abs(np.angle(value)))
    growth = float(math.log(radius) / dt)
    frequency = float(angle / (2.0 * math.pi * dt))
    return PredictedMode(
        eigenvalue=value,
        right_eigenvector=vector,
        orientation_axis=axis,
        orientation_subspace_singular_values=singular[:2],
        growth_rate_per_second=growth,
        frequency_hz=frequency,
        doubling_time_seconds=_doubling_time(growth),
    )


def _causal_reference_indices(
    reference_times: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    if (
        reference.ndim != 1
        or query.ndim != 1
        or reference.size == 0
        or query.size == 0
        or np.any(~np.isfinite(reference))
        or np.any(~np.isfinite(query))
        or np.any(np.diff(reference) <= 0.0)
        or np.any(np.diff(query) <= 0.0)
    ):
        raise ValueError("reference/query times must be finite and increasing")
    indices = np.searchsorted(reference, query, side="right") - 1
    return indices


def _recorded_orientation_error(
    inputs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flight = load_flight_data(
        path=inputs.bag.bag_path,
        start_local=inputs.bag.start_seconds,
        end_local=inputs.bag.end_seconds,
        include_fc_specific_force=False,
        compute_sha256=False,
        bag_id=inputs.bag.source_path.stem,
    )
    pose_times = np.asarray(flight.pose.times, dtype=float)
    orientations = np.asarray(flight.pose.orientations_xyzw, dtype=float)
    reference_times = np.asarray(flight.reference.times, dtype=float)
    reference_rpy = np.asarray(flight.reference.rpy, dtype=float)

    indices = _causal_reference_indices(reference_times, pose_times)
    valid = indices >= 0
    if np.count_nonzero(valid) < 8:
        raise RuntimeError(
            "too few pose samples have a causal controller reference"
        )
    times = pose_times[valid]
    orientations = orientations[valid]
    indices = indices[valid]

    error = np.empty((times.size, 3), dtype=float)
    for row, (quaternion, reference_index) in enumerate(
        zip(orientations, indices)
    ):
        actual = quaternion_to_matrix(quaternion)
        reference = euler_xyz_to_matrix(reference_rpy[int(reference_index)])
        error[row] = so3_log(reference.T @ actual)

    if np.any(~np.isfinite(error)):
        raise RuntimeError("recorded SO(3) attitude error became non-finite")
    return times, error, reference_rpy[indices]


def _linear_detrend(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    centered = t - t[0]
    design = np.column_stack((np.ones_like(centered), centered))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficient


def _lomb_frequency_initialization(
    time: np.ndarray,
    signal: np.ndarray,
    frequency_min_hz: float,
    frequency_max_hz: float,
    grid_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    lower = float(frequency_min_hz)
    upper = float(frequency_max_hz)
    count = int(grid_size)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
        or count < 32
    ):
        raise ValueError("frequency search range/grid is invalid")
    frequencies = np.linspace(lower, upper, count)
    detrended = _linear_detrend(t, y)
    angular = 2.0 * math.pi * frequencies
    power = lombscargle(
        t - t[0],
        detrended,
        angular,
        precenter=True,
        normalize=True,
    )
    if np.any(~np.isfinite(power)):
        raise RuntimeError("Lomb-Scargle spectrum became non-finite")
    selected = int(np.argmax(power))
    return float(frequencies[selected]), frequencies, power


def _damped_sine_design(
    relative_time: np.ndarray,
    growth_rate: float,
    frequency_hz: float,
) -> np.ndarray:
    t = np.asarray(relative_time, dtype=float)
    exponent = float(growth_rate) * t
    envelope = np.exp(exponent)
    phase = 2.0 * math.pi * float(frequency_hz) * t
    return np.column_stack(
        (
            np.ones_like(t),
            t,
            envelope * np.cos(phase),
            envelope * np.sin(phase),
        )
    )


def fit_damped_sine(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_grid_size: int,
) -> tuple[DampedSineFit, np.ndarray, np.ndarray, np.ndarray]:
    """Fit trend + exp(sigma*t)*(a*cos(2*pi*f*t)+b*sin(...)).

    Frequency is searched only inside the explicit diagnostic band.  The
    growth-rate bounds are numerical representability bounds: they keep
    ``exp(sigma*t)`` finite and are not an engineering stability criterion.
    """

    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    if (
        t.ndim != 1
        or y.shape != t.shape
        or t.size < 16
        or np.any(~np.isfinite(t))
        or np.any(~np.isfinite(y))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("fit time/signal must be finite and aligned")
    relative = t - t[0]
    duration = float(relative[-1])
    if duration <= 0.0:
        raise ValueError("fit window must have positive duration")

    initial_frequency, frequency_grid, power = (
        _lomb_frequency_initialization(
            t,
            y,
            frequency_min_hz,
            frequency_max_hz,
            frequency_grid_size,
        )
    )

    initial_design = _damped_sine_design(
        relative, 0.0, initial_frequency
    )
    initial_linear, *_ = np.linalg.lstsq(initial_design, y, rcond=None)
    initial = np.concatenate(
        (
            np.asarray((0.0, math.log(initial_frequency))),
            initial_linear,
        )
    )

    numerical_growth_limit = 600.0 / duration
    lower = np.asarray(
        (
            -numerical_growth_limit,
            math.log(float(frequency_min_hz)),
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        )
    )
    upper = np.asarray(
        (
            numerical_growth_limit,
            math.log(float(frequency_max_hz)),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        )
    )

    def residual(parameter: np.ndarray) -> np.ndarray:
        sigma = float(parameter[0])
        frequency = float(math.exp(float(parameter[1])))
        design = _damped_sine_design(relative, sigma, frequency)
        return design @ parameter[2:] - y

    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        ftol=np.sqrt(np.finfo(float).eps),
        xtol=np.sqrt(np.finfo(float).eps),
        gtol=np.sqrt(np.finfo(float).eps),
        max_nfev=4000,
    )
    parameter = np.asarray(solved.x, dtype=float)
    if np.any(~np.isfinite(parameter)):
        raise RuntimeError("damped-sine fit returned non-finite parameters")

    sigma = float(parameter[0])
    frequency = float(math.exp(parameter[1]))
    design = _damped_sine_design(relative, sigma, frequency)
    fitted = design @ parameter[2:]
    residual_vector = fitted - y
    sse = float(residual_vector @ residual_vector)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r_squared = float(1.0 - sse / sst) if sst > 0.0 else float("nan")
    amplitude = float(math.hypot(parameter[4], parameter[5]))

    fit = DampedSineFit(
        growth_rate_per_second=sigma,
        frequency_hz=frequency,
        intercept=float(parameter[2]),
        slope=float(parameter[3]),
        cosine_coefficient=float(parameter[4]),
        sine_coefficient=float(parameter[5]),
        amplitude_at_start=amplitude,
        doubling_time_seconds=_doubling_time(sigma),
        residual_rms=float(np.sqrt(np.mean(residual_vector * residual_vector))),
        r_squared=r_squared,
        initial_frequency_hz=initial_frequency,
        optimizer_status=int(solved.status),
        optimizer_message=str(solved.message),
    )
    observed_envelope = amplitude * np.exp(sigma * relative)
    return fit, fitted, observed_envelope, np.column_stack(
        (frequency_grid, power)
    )


def _fit_window(
    time: np.ndarray,
    signal: np.ndarray,
    error: np.ndarray,
    *,
    fit_start_seconds: Optional[float],
    fit_end_seconds: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    rotation_error = np.asarray(error, dtype=float)
    relative = t - t[0]
    start = (
        float(relative[0])
        if fit_start_seconds is None
        else float(fit_start_seconds)
    )
    end = (
        float(relative[-1])
        if fit_end_seconds is None
        else float(fit_end_seconds)
    )
    if (
        not np.isfinite(start)
        or not np.isfinite(end)
        or start < relative[0]
        or end > relative[-1]
        or end <= start
    ):
        raise ValueError("fit window lies outside recorded pose support")
    mask = (relative >= start) & (relative <= end)
    if np.count_nonzero(mask) < 16:
        raise ValueError("fit window contains too few pose samples")
    return t[mask], relative[mask], y[mask], rotation_error[mask]


def _json_mode(mode: PredictedMode) -> Mapping[str, Any]:
    value = mode.eigenvalue
    return {
        "eigenvalue": {
            "real": float(value.real),
            "imag": float(value.imag),
            "magnitude": float(abs(value)),
            "angle_rad": float(np.angle(value)),
        },
        "growth_rate_per_second": mode.growth_rate_per_second,
        "frequency_hz": mode.frequency_hz,
        "doubling_time_seconds": mode.doubling_time_seconds,
        "orientation_axis_xyz": mode.orientation_axis.tolist(),
        "orientation_subspace_singular_values": (
            mode.orientation_subspace_singular_values.tolist()
        ),
        "right_eigenvector_orientation_real": (
            mode.right_eigenvector[3:6].real.tolist()
        ),
        "right_eigenvector_orientation_imag": (
            mode.right_eigenvector[3:6].imag.tolist()
        ),
        "right_eigenvector_omega_real": (
            mode.right_eigenvector[9:12].real.tolist()
        ),
        "right_eigenvector_omega_imag": (
            mode.right_eigenvector[9:12].imag.tolist()
        ),
    }


def _json_fit(fit: DampedSineFit) -> Mapping[str, Any]:
    return {
        "growth_rate_per_second": fit.growth_rate_per_second,
        "frequency_hz": fit.frequency_hz,
        "doubling_time_seconds": fit.doubling_time_seconds,
        "intercept_rad": fit.intercept,
        "slope_rad_per_second": fit.slope,
        "cosine_coefficient_rad": fit.cosine_coefficient,
        "sine_coefficient_rad": fit.sine_coefficient,
        "amplitude_at_start_rad": fit.amplitude_at_start,
        "residual_rms_rad": fit.residual_rms,
        "r_squared": fit.r_squared,
        "initial_frequency_hz_from_lomb_scargle": fit.initial_frequency_hz,
        "optimizer_status": fit.optimizer_status,
        "optimizer_message": fit.optimizer_message,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    predicted = report["predicted_mode"]
    observed = report["observed_fit"]
    comparison = report["comparison"]
    lines = [
        "# Gimbalrotor unstable-mode growth validation",
        "",
        "- Case: `{}`".format(report["case"]),
        "- Actual outcome: `{}`".format(report["actual_outcome"]),
        "- Fit window: [{:.6g}, {:.6g}] s relative to first valid pose sample".format(
            report["fit_window"]["start_seconds"],
            report["fit_window"]["end_seconds"],
        ),
        "",
        "## Pole prediction",
        "",
        "- discrete pole: `{:.9g} {:+.9g}i`".format(
            predicted["eigenvalue"]["real"],
            predicted["eigenvalue"]["imag"],
        ),
        "- |z|: `{:.9g}`".format(predicted["eigenvalue"]["magnitude"]),
        "- growth sigma: `{:.9g} 1/s`".format(
            predicted["growth_rate_per_second"]
        ),
        "- frequency: `{:.9g} Hz`".format(predicted["frequency_hz"]),
        "- doubling time: `{}` s".format(
            "—"
            if predicted["doubling_time_seconds"] is None
            else "{:.9g}".format(predicted["doubling_time_seconds"])
        ),
        "- matched orientation axis xyz: `{}`".format(
            np.asarray(predicted["orientation_axis_xyz"])
        ),
        "",
        "## Recorded attitude fit",
        "",
        "- growth sigma: `{:.9g} 1/s`".format(
            observed["growth_rate_per_second"]
        ),
        "- frequency: `{:.9g} Hz`".format(observed["frequency_hz"]),
        "- doubling time: `{}` s".format(
            "—"
            if observed["doubling_time_seconds"] is None
            else "{:.9g}".format(observed["doubling_time_seconds"])
        ),
        "- fit R^2: `{:.9g}`".format(observed["r_squared"]),
        "- residual RMS: `{:.9g} rad`".format(observed["residual_rms_rad"]),
        "",
        "## Comparison",
        "",
        "- observed - predicted growth: `{:.9g} 1/s`".format(
            comparison["growth_rate_difference_per_second"]
        ),
        "- observed / predicted growth: `{}`".format(
            "—"
            if comparison["growth_rate_ratio"] is None
            else "{:.9g}".format(comparison["growth_rate_ratio"])
        ),
        "- observed - predicted frequency: `{:.9g} Hz`".format(
            comparison["frequency_difference_hz"]
        ),
        "- observed / predicted frequency: `{:.9g}`".format(
            comparison["frequency_ratio"]
        ),
        "",
        "The flight label is metadata only; it does not enter the fit.",
    ]
    return "\n".join(lines) + "\n"


def _plot_case(
    *,
    output_path: Path,
    relative_time: np.ndarray,
    rotation_error: np.ndarray,
    modal_signal: np.ndarray,
    fit_mask: np.ndarray,
    fitted_signal: np.ndarray,
    observed_envelope: np.ndarray,
    predicted_envelope: np.ndarray,
    fit: DampedSineFit,
    predicted: PredictedMode,
    spectrum: np.ndarray,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(
        relative_time,
        np.degrees(rotation_error[:, 0]),
        label="rotation error x",
    )
    axes[0].plot(
        relative_time,
        np.degrees(rotation_error[:, 1]),
        label="rotation error y",
    )
    axes[0].plot(
        relative_time,
        np.degrees(rotation_error[:, 2]),
        label="rotation error z",
    )
    axes[0].plot(
        relative_time,
        np.degrees(modal_signal),
        label="matched modal-direction signal",
    )
    axes[0].set_ylabel("rotation error [deg]")
    axes[0].legend()
    axes[0].grid(True)

    fit_time = relative_time[fit_mask]
    axes[1].plot(
        fit_time,
        np.degrees(modal_signal[fit_mask]),
        label="recorded matched signal",
    )
    axes[1].plot(
        fit_time,
        np.degrees(fitted_signal),
        label="fitted damped sinusoid",
    )
    trend = fit.intercept + fit.slope * (fit_time - fit_time[0])
    axes[1].plot(
        fit_time,
        np.degrees(trend + observed_envelope),
        label="observed-fit + envelope",
    )
    axes[1].plot(
        fit_time,
        np.degrees(trend - observed_envelope),
        label="observed-fit - envelope",
    )
    axes[1].plot(
        fit_time,
        np.degrees(trend + predicted_envelope),
        linestyle="--",
        label="pole-predicted + envelope",
    )
    axes[1].plot(
        fit_time,
        np.degrees(trend - predicted_envelope),
        linestyle="--",
        label="pole-predicted - envelope",
    )
    axes[1].set_ylabel("matched signal [deg]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(spectrum[:, 0], spectrum[:, 1], label="Lomb-Scargle")
    axes[2].axvline(
        predicted.frequency_hz,
        linestyle="--",
        label="pole frequency",
    )
    axes[2].axvline(
        fit.frequency_hz,
        linestyle=":",
        label="fitted frequency",
    )
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel("normalized power")
    axes[2].legend()
    axes[2].grid(True)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_case(
    case_name: str,
    *,
    controller_yaml: Path,
    vehicle_model: Path,
    output_dir: Path,
    fit_start_seconds: Optional[float],
    fit_end_seconds: Optional[float],
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_grid_size: int,
    controller_dt_override: Optional[float],
) -> Mapping[str, Any]:
    if case_name not in three_bag.CASE_DEFINITIONS:
        raise KeyError("unknown case {!r}".format(case_name))
    definition = three_bag.CASE_DEFINITIONS[case_name]
    estimator = Path(definition["estimator"])

    # The center plant does not depend on the covariance construction.  One
    # production covariance mode is loaded only because the existing validated
    # input adapter owns the quotient center.
    inputs = local_poles.load_case_inputs(
        result_path=estimator / "result.json",
        arrays_path=estimator / "arrays.npz",
        static_postprocess_path=Path(definition["static"]),
        arguments_path=estimator / "arguments.json",
        bag_json_path=Path(definition["bag_json"]),
        controller_yaml_path=controller_yaml,
        vehicle_model_path=vehicle_model,
        covariance_mode="conservative_fusion",
    )
    timing = local_poles.controller_timing_from_bag(inputs.bag)
    controller_dt = (
        float(timing["median_seconds"])
        if controller_dt_override is None
        else float(controller_dt_override)
    )
    if not np.isfinite(controller_dt) or controller_dt <= 0.0:
        raise ValueError("controller_dt must be finite and positive")

    fitted_delay = float(inputs.result.plant.rotor_lag_seconds)
    delay = local_poles.decompose_thrust_delay(
        fitted_delay, controller_dt
    )
    center = local_poles._analyze_plant(  # pylint: disable=protected-access
        scale_free=inputs.sampling_coordinates.center_plant,
        inputs=inputs,
        controller_dt=controller_dt,
        delay=delay,
        fd_check=False,
    )
    if center["jacobian"] is None or center["eigenvalues"] is None:
        raise RuntimeError("center hover equilibrium has no valid pole Jacobian")
    predicted = _dominant_oscillatory_mode(
        center["jacobian"], controller_dt
    )

    time, rotation_error, _reference_rpy = _recorded_orientation_error(
        inputs
    )
    relative_time = time - time[0]
    modal_signal = rotation_error @ predicted.orientation_axis

    fit_time, fit_relative, fit_signal, fit_error = _fit_window(
        time,
        modal_signal,
        rotation_error,
        fit_start_seconds=fit_start_seconds,
        fit_end_seconds=fit_end_seconds,
    )
    fit, fitted_signal, observed_envelope, spectrum = fit_damped_sine(
        fit_time,
        fit_signal,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        frequency_grid_size=frequency_grid_size,
    )
    relative_fit_time = fit_time - fit_time[0]
    predicted_envelope = (
        fit.amplitude_at_start
        * np.exp(predicted.growth_rate_per_second * relative_fit_time)
    )

    fit_mask = (time >= fit_time[0]) & (time <= fit_time[-1])
    if np.count_nonzero(fit_mask) != fit_time.size:
        raise RuntimeError("fit mask no longer aligns with selected pose samples")

    predicted_json = _json_mode(predicted)
    observed_json = _json_fit(fit)
    growth_ratio = (
        None
        if predicted.growth_rate_per_second == 0.0
        else fit.growth_rate_per_second
        / predicted.growth_rate_per_second
    )
    frequency_ratio = fit.frequency_hz / predicted.frequency_hz
    report: Mapping[str, Any] = {
        "schema": SCHEMA,
        "source_commit": local_poles.source_commit(),
        "validation_base_commit": VALIDATION_BASE_COMMIT,
        "case": case_name,
        "actual_outcome": str(definition["outcome"]),
        "scientific_role": (
            "one-shot post-hoc validation; no feedback to estimator, "
            "pole classifier, or PID proposal"
        ),
        "inputs": {
            "estimator_result": str(estimator / "result.json"),
            "static_postprocess": str(definition["static"]),
            "bag_json": str(definition["bag_json"]),
            "bag_path": inputs.bag.bag_path,
            "bag_interval_seconds": [
                inputs.bag.start_seconds,
                inputs.bag.end_seconds,
            ],
            "controller_yaml": str(Path(controller_yaml).expanduser().resolve()),
            "vehicle_model": str(Path(vehicle_model).expanduser().resolve()),
        },
        "pole_model": {
            "delay_mode": "fitted_thrust_delay",
            "fitted_thrust_delay_seconds": fitted_delay,
            "controller_dt_seconds": controller_dt,
            "controller_dt_source": (
                "recorded_thrust_command_median"
                if controller_dt_override is None
                else "explicit_override"
            ),
            "center_equilibrium_valid": bool(
                center["trim"].equilibrium_valid
            ),
            "center_one_step_trim_defect_norm": float(
                center["trim"].one_step_defect_norm
            ),
        },
        "predicted_mode": predicted_json,
        "recorded_signal": {
            "definition": (
                "q(t) = u^T Log(R_ref(t)^T R_pose(t)); u is the principal "
                "orientation direction of the dominant oscillatory center "
                "right eigenvector"
            ),
            "pose_sample_count": int(time.size),
            "orientation_axis_xyz": predicted.orientation_axis.tolist(),
        },
        "fit_window": {
            "start_seconds": float(fit_relative[0]),
            "end_seconds": float(fit_relative[-1]),
            "duration_seconds": float(fit_relative[-1] - fit_relative[0]),
            "sample_count": int(fit_time.size),
            "time_origin": "first valid pose sample in selected bag interval",
        },
        "fit_method": {
            "model": (
                "c0 + c1*t + exp(sigma*t)*(a*cos(2*pi*f*t) + "
                "b*sin(2*pi*f*t))"
            ),
            "frequency_initialization": "Lomb-Scargle peak",
            "frequency_search_min_hz": float(frequency_min_hz),
            "frequency_search_max_hz": float(frequency_max_hz),
            "frequency_grid_size": int(frequency_grid_size),
            "growth_bounds": (
                "numerical representability only: +/- 600/window_duration"
            ),
            "flight_label_used_in_fit": False,
            "predicted_pole_used_as_fit_initialization": False,
        },
        "observed_fit": observed_json,
        "comparison": {
            "growth_rate_difference_per_second": float(
                fit.growth_rate_per_second
                - predicted.growth_rate_per_second
            ),
            "growth_rate_ratio": (
                None if growth_ratio is None else float(growth_ratio)
            ),
            "frequency_difference_hz": float(
                fit.frequency_hz - predicted.frequency_hz
            ),
            "frequency_ratio": float(frequency_ratio),
        },
    }

    case_output = Path(output_dir).expanduser().resolve() / case_name
    case_output.mkdir(parents=True, exist_ok=True)
    local_poles.write_json(
        case_output / "mode_growth_validation.json", report
    )
    (case_output / "mode_growth_validation.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    np.savez_compressed(
        case_output / "mode_growth_validation.npz",
        time_seconds=time,
        relative_time_seconds=relative_time,
        rotation_error_xyz_rad=rotation_error,
        orientation_axis_xyz=predicted.orientation_axis,
        modal_signal_rad=modal_signal,
        fit_time_seconds=fit_time,
        fit_relative_time_seconds=relative_fit_time,
        fit_rotation_error_xyz_rad=fit_error,
        fit_modal_signal_rad=fit_signal,
        fitted_signal_rad=fitted_signal,
        observed_fit_envelope_rad=observed_envelope,
        pole_predicted_envelope_rad=predicted_envelope,
        lomb_frequency_hz=spectrum[:, 0],
        lomb_power=spectrum[:, 1],
        dominant_eigenvalue=np.asarray((predicted.eigenvalue,), dtype=complex),
        dominant_right_eigenvector=predicted.right_eigenvector,
    )
    _plot_case(
        output_path=case_output / "mode_growth_validation.png",
        relative_time=relative_time,
        rotation_error=rotation_error,
        modal_signal=modal_signal,
        fit_mask=fit_mask,
        fitted_signal=fitted_signal,
        observed_envelope=observed_envelope,
        predicted_envelope=predicted_envelope,
        fit=fit,
        predicted=predicted,
        spectrum=spectrum,
    )
    return report


def _summary(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = []
    for report in reports:
        predicted = report["predicted_mode"]
        observed = report["observed_fit"]
        comparison = report["comparison"]
        rows.append(
            {
                "case": report["case"],
                "actual_outcome": report["actual_outcome"],
                "predicted_growth_rate_per_second": (
                    predicted["growth_rate_per_second"]
                ),
                "observed_growth_rate_per_second": (
                    observed["growth_rate_per_second"]
                ),
                "growth_rate_ratio": comparison["growth_rate_ratio"],
                "predicted_doubling_time_seconds": (
                    predicted["doubling_time_seconds"]
                ),
                "observed_doubling_time_seconds": (
                    observed["doubling_time_seconds"]
                ),
                "predicted_frequency_hz": predicted["frequency_hz"],
                "observed_frequency_hz": observed["frequency_hz"],
                "frequency_ratio": comparison["frequency_ratio"],
                "fit_r_squared": observed["r_squared"],
            }
        )
    return {
        "schema": SCHEMA + "-summary",
        "source_commit": local_poles.source_commit(),
        "validation_base_commit": VALIDATION_BASE_COMMIT,
        "rows": rows,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gimbalrotor unstable-mode growth validation summary",
        "",
        "| case | outcome | pole sigma [1/s] | observed sigma [1/s] | ratio | pole doubling [s] | observed doubling [s] | pole f [Hz] | observed f [Hz] | ratio | fit R^2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value: Any) -> str:
        if value is None:
            return "—"
        return "{:.8g}".format(float(value))

    for row in summary["rows"]:
        lines.append(
            "| {case} | {outcome} | {ps} | {os} | {gr} | {pd} | {od} | {pf} | {of} | {fr} | {r2} |".format(
                case=row["case"],
                outcome=row["actual_outcome"],
                ps=number(row["predicted_growth_rate_per_second"]),
                os=number(row["observed_growth_rate_per_second"]),
                gr=number(row["growth_rate_ratio"]),
                pd=number(row["predicted_doubling_time_seconds"]),
                od=number(row["observed_doubling_time_seconds"]),
                pf=number(row["predicted_frequency_hz"]),
                of=number(row["observed_frequency_hz"]),
                fr=number(row["frequency_ratio"]),
                r2=number(row["fit_r_squared"]),
            )
        )
    lines.extend(
        (
            "",
            "This is a post-hoc validation.  The observed fit is not used by the estimator or local-pole calculation.",
        )
    )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-yaml",
        type=Path,
        default=Path(
            "/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/"
            "config/grape/GimbalrotorControl.yaml"
        ),
    )
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        default=_HERE / "grape_vehicle_model.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(three_bag.CASE_DEFINITIONS),
        help="Repeat to select cases; default is failure1 and failure2.",
    )
    parser.add_argument(
        "--fit-start",
        type=float,
        default=None,
        help="Seconds after the first valid pose sample; default uses full support.",
    )
    parser.add_argument(
        "--fit-end",
        type=float,
        default=None,
        help="Seconds after the first valid pose sample; default uses full support.",
    )
    parser.add_argument(
        "--frequency-min",
        type=float,
        default=DEFAULT_FREQUENCY_MIN_HZ,
    )
    parser.add_argument(
        "--frequency-max",
        type=float,
        default=DEFAULT_FREQUENCY_MAX_HZ,
    )
    parser.add_argument(
        "--frequency-grid-size",
        type=int,
        default=DEFAULT_FREQUENCY_GRID_SIZE,
    )
    parser.add_argument("--controller-dt", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "outputs" / "mode_growth_validation",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    cases = tuple(arguments.case or DEFAULT_CASES)
    reports = [
        analyze_case(
            case,
            controller_yaml=arguments.controller_yaml,
            vehicle_model=arguments.vehicle_model,
            output_dir=arguments.output_dir,
            fit_start_seconds=arguments.fit_start,
            fit_end_seconds=arguments.fit_end,
            frequency_min_hz=arguments.frequency_min,
            frequency_max_hz=arguments.frequency_max,
            frequency_grid_size=arguments.frequency_grid_size,
            controller_dt_override=arguments.controller_dt,
        )
        for case in cases
    ]
    summary = _summary(reports)
    output = Path(arguments.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    local_poles.write_json(output / "mode_growth_validation_summary.json", summary)
    (output / "mode_growth_validation_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
