#!/usr/bin/env python3
"""Deterministic rigid-body parameter estimation from geometric SG pose derivatives.

This module deliberately reuses the mature downstream dynamics, analytic
parameter Jacobians, strict-ZOH refinement, external-wrench replay, and report
generation from ``deterministic_spline_dynamics_estimator.py``.  Only the
pose-to-kinematics front end is replaced.

For parameter estimation:
- raw mocap pose timestamps are used directly (no pose resampling);
- a degree-5 local polynomial is fit over a physical time window W;
- translation is fit in R^3;
- rotation uses the geometric SO(3) Savitzky-Golay construction of
  Jongeneel & Saccon (IROS 2022);
- only raw pose times with a centered full W-second window are used as
  Newton-Euler evaluation times.

The old spline estimator remains untouched for ablation/reference purposes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import deterministic_spline_dynamics_estimator as base
import savgol_trajectory as sg


SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v1"
OUTPUT_SUBDIRECTORY = "deterministic_savgol_dynamics"
DATA_DICTIONARY_SOURCE = Path(__file__).resolve().with_name(
    "deterministic_savgol_dynamics_data_dictionary.md"
)

# Re-export the physical model API so the confidence layer can later use this
# module as a drop-in deterministic backend.
PHYSICAL_DIMENSION = base.PHYSICAL_DIMENSION
GLOBAL_DIMENSION = base.GLOBAL_DIMENSION
DELAY_INDEX = base.DELAY_INDEX
PHYSICAL_PARAMETER_NAMES = base.PHYSICAL_PARAMETER_NAMES
PHYSICAL_VALUE_NAMES = base.PHYSICAL_VALUE_NAMES
SplineDynamicsProblem = base.SplineDynamicsProblem
BagDynamicsEvaluation = base.BagDynamicsEvaluation
JointDynamicsEvaluation = base.JointDynamicsEvaluation
DynamicsSolution = base.DynamicsSolution
SplinePhysicalParameterization = base.SplinePhysicalParameterization
GaussianPhysicalPrior = base.GaussianPhysicalPrior
VehicleModelInput = base.VehicleModelInput
physical_parameter_vector = base.physical_parameter_vector
physical_parameter_jacobian = base.physical_parameter_jacobian
load_vehicle_model = base.load_vehicle_model
load_parameter_prior = base.load_parameter_prior
_physical_bounds = base._physical_bounds
_solve_smooth = base._solve_smooth
_solve_strict = base._solve_strict
_solution_cost = base._solution_cost
_json_sanitize = base._json_sanitize
smooth = base.smooth
baseline = base.baseline
strict = base.strict
load_flight_data = base.load_flight_data


def __getattr__(name: str) -> Any:
    """Forward mature downstream helpers to the existing spline backend.

    Only the pose-kinematics front end is replaced in this module.  Keeping
    helper lookup transparent lets the confidence layer reuse the existing
    replay/report functions without copying them.
    """

    return getattr(base, name)


_ACTIVE_WINDOW_SECONDS: Optional[float] = None
_ACTIVE_BAGS: dict[str, base.BagSplineData] = {}
_ACTIVE_RAW_WRENCH_STATS: dict[str, dict[str, Any]] = {}
_ORIGINAL_PARAMETER_LINES = base._parameter_lines


def _window() -> float:
    if _ACTIVE_WINDOW_SECONDS is None:
        raise RuntimeError("Savitzky-Golay window is not configured")
    return float(_ACTIVE_WINDOW_SECONDS)


def _compatibility_config(path: Path) -> base.SplineEstimatorConfig:
    """Load only the bag configuration; spline settings are intentionally inert."""

    multi_config = base.multi.load_multi_bag_config(path.expanduser().resolve())
    # base.run still serializes these historical fields before the JSON is
    # rewritten below.  NaN is used deliberately instead of inventing values.
    settings = base.SplineSettings(
        knot_spacing_candidates_seconds=(_window(),),
        collocation_step_seconds=math.nan,
        boundary_exclusion_knot_spans_each_side=0.0,
        cross_validation_block_seconds=math.nan,
    )
    return base.SplineEstimatorConfig(
        multi_bag=multi_config,
        spline=settings,
    )


def load_spline_config(path: Path) -> base.SplineEstimatorConfig:
    """Compatibility name used by the existing confidence/dynamics code."""

    return _compatibility_config(path)


def _validate_arguments(
    arguments: argparse.Namespace,
    config: base.SplineEstimatorConfig,
    initial_delay: float,
) -> None:
    del config
    positive = (
        arguments.window_seconds,
        arguments.sample_step,
        arguments.integration_step,
        arguments.smooth_max_nfev,
        arguments.strict_max_nfev,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.zoh_polish_top_k,
    )
    ratio = arguments.sample_step / arguments.integration_step
    bounds = np.asarray(arguments.delay_bounds, dtype=float)
    widths = np.asarray(arguments.smoothstep_width_fractions, dtype=float)
    if (
        any(
            not np.isfinite(value) or value <= 0.0
            for value in positive
        )
        or not np.isclose(
            ratio,
            round(ratio),
            atol=1.0e-12,
            rtol=0.0,
        )
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or not bounds[0] <= initial_delay <= bounds[1]
        or widths.ndim != 1
        or widths.size == 0
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit("Savitzky-Golay dynamics settings are invalid")


def _build_bag_data(
    specification: Any,
    normalized_weight: float,
    flight: Any,
    initial_delay: float,
    settings: Any,
    arguments: argparse.Namespace,
    reference_parameters: Any,
    geometry: Any,
) -> base.BagSplineData:
    del settings

    direct = base.baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=initial_delay,
        prior_weight=0.0,
        reference_parameters=reference_parameters,
        geometry=geometry,
    )

    raw_time = np.asarray(flight.pose.times, dtype=float)
    if not sg.window_is_feasible(
        raw_time,
        arguments.window_seconds,
        degree=sg.POLYNOMIAL_DEGREE,
    ):
        minimum = sg.minimum_feasible_window_seconds(
            raw_time,
            degree=sg.POLYNOMIAL_DEGREE,
        )
        raise ValueError(
            "W={:.12g}s is too short for degree-5 local polynomial on bag {}; "
            "minimum data-supported W is {:.12g}s".format(
                arguments.window_seconds,
                specification.bag_id,
                minimum,
            )
        )

    selection = sg.select_pose_spline(
        time_axis=raw_time,
        sensor_position=np.asarray(flight.pose.positions, dtype=float),
        sensor_orientation_xyzw=np.asarray(
            flight.pose.orientations_xyzw,
            dtype=float,
        ),
        body_to_pose_sensor_rotation=direct.pose_body_to_sensor_rotation,
        knot_spacing_candidates_seconds=(float(arguments.window_seconds),),
        rotational_metric=(
            reference_parameters.inertia / reference_parameters.mass
        ),
    )

    # The parameter loss uses every raw pose timestamp for which the complete
    # centered W-second window exists.  The DirectShootingProblem support is
    # intersected only because its extrinsics/actuator model are reused
    # downstream; no direct.output_time sample is used to fit SG.
    support_start = float(direct.output_time[0])
    support_end = float(direct.output_time[-1])

    # Delay search queries u(t - tau).  Exclude any initial interval that
    # cannot be covered at the largest allowed delay instead of relying on
    # invalid ZOH extrapolation.
    maximum_delay = float(arguments.delay_bounds[1])
    support_start = max(
        support_start,
        float(flight.rotor_command.all_times[0]) + maximum_delay,
        float(flight.gimbal_command.all_times[0]) + maximum_delay,
    )

    collocation_time = selection.spline.centered_raw_times(
        support_start=support_start,
        support_end=support_end,
    )
    collocation = selection.spline.evaluate(collocation_time)
    initial_gimbal = base.baseline._linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((collocation_time[0],), dtype=float),
    )[0]

    bag = base.BagSplineData(
        specification=specification,
        normalized_weight=float(normalized_weight),
        flight=flight,
        direct_problem=direct,
        spline_selection=selection,
        collocation=collocation,
        rotor_history=base.QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        ),
        gimbal_history=base.QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        ),
        initial_gimbal=initial_gimbal,
        # Historical field retained only for ABI compatibility.  It has no
        # meaning in the SG estimator and is removed from emitted JSON.
        boundary_exclusion_knot_spans_each_side=0.0,
    )
    _ACTIVE_BAGS[str(specification.bag_id)] = bag
    return bag


def _timestamp_interval_statistics(time_axis: Sequence[float]) -> dict[str, Any]:
    value = np.asarray(time_axis, dtype=float)
    differences = np.diff(value)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return {
            "sample_count": int(value.size),
            "positive_interval_count": 0,
            "minimum_seconds": None,
            "median_seconds": None,
            "mean_seconds": None,
            "maximum_seconds": None,
            "median_frequency_hz": None,
        }
    median = float(np.median(positive))
    return {
        "sample_count": int(value.size),
        "positive_interval_count": int(positive.size),
        "minimum_seconds": float(np.min(positive)),
        "median_seconds": median,
        "mean_seconds": float(np.mean(positive)),
        "maximum_seconds": float(np.max(positive)),
        "median_frequency_hz": float(1.0 / median),
    }


def _parameter_lines(
    selected: Any,
    initial_delay: float,
    bags: Sequence[Any],
    bag_payloads: Sequence[Mapping[str, Any]],
    reference_parameters: Any,
) -> list[str]:
    lines = _ORIGINAL_PARAMETER_LINES(
        selected,
        initial_delay,
        bags,
        bag_payloads,
        reference_parameters,
    )
    lines = [
        line.replace(
            "Deterministic pose-spline dynamics estimator",
            "Deterministic geometric Savitzky-Golay dynamics estimator",
        )
        for line in lines
    ]
    lines.extend(["", "Recorded command timestamp intervals (data-derived)"])
    for bag in bags:
        rotor = _timestamp_interval_statistics(
            bag.flight.rotor_command.all_times
        )
        gimbal = _timestamp_interval_statistics(
            bag.flight.gimbal_command.all_times
        )
        lines.append("  Bag {}".format(bag.specification.bag_id))
        for name, statistics in (("rotor", rotor), ("gimbal", gimbal)):
            median = statistics["median_seconds"]
            frequency = statistics["median_frequency_hz"]
            if median is None:
                lines.append("    {}: no positive timestamp interval".format(name))
            else:
                lines.append(
                    "    {}: median dt={:.9g} s ({:.6g} Hz), min={:.9g} s, max={:.9g} s".format(
                        name,
                        median,
                        frequency,
                        statistics["minimum_seconds"],
                        statistics["maximum_seconds"],
                    )
                )
    return lines


class _RewritingStdout:
    """Keep base.run's useful progress output while removing spline wording."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, value: str) -> int:
        text = value.replace(
            "loading and fitting pose spline",
            "loading raw pose for geometric Savitzky-Golay",
        )
        text = text.replace(
            "selected strict lag",
            "selected strict lag",
        )
        pattern = re.compile(
            r"selected knot spacing ([0-9eE+.\-]+)s from .*?; parameter support "
            r"\[([0-9eE+.\-]+), ([0-9eE+.\-]+)\]s after excluding "
            r"[0-9eE+.\-]+ knot spans per side"
        )
        text = pattern.sub(
            r"SG window \1s; centered raw-pose parameter support "
            r"[\2, \3]s",
            text,
        )
        text = text.replace(
            "spline-dynamics",
            "Savitzky-Golay dynamics",
        )
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())


def _write_savgol_fit_pdf(path: Path, bag: base.BagSplineData) -> None:
    """Preserve the old fit/derivative diagnostics and add SG-specific ones."""

    raw_time = np.asarray(bag.flight.pose.times, dtype=float)
    raw_position = np.asarray(bag.flight.pose.positions, dtype=float)
    raw_orientation = np.asarray(
        bag.flight.pose.orientations_xyzw,
        dtype=float,
    )
    trajectory = bag.spline_selection.spline
    filtered = trajectory.evaluate(raw_time)
    filtered_orientation = trajectory.sensor_orientation_xyzw(raw_time)

    raw_rpy = base.baseline._rpy_series(raw_orientation)
    filtered_rpy = base.baseline._rpy_series(filtered_orientation)
    position_error = filtered.sensor_position - raw_position
    orientation_error = base._orientation_errors(
        raw_orientation,
        filtered_orientation,
    )
    relative_time = raw_time - raw_time[0]
    collocation = bag.collocation
    collocation_relative = collocation.time - raw_time[0]

    with base.PdfPages(path) as pdf:
        for title, reference, estimate, labels in (
            (
                "Raw mocap and local-polynomial position",
                raw_position,
                filtered.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Raw mocap and geometric-SG orientation",
                raw_rpy,
                filtered_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = base.plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    label="raw mocap",
                )
                axis.plot(
                    relative_time,
                    estimate[:, component],
                    linestyle="--",
                    label="degree-5 local polynomial",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; W={:.6g}s".format(title, trajectory.window_seconds)
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time from raw-pose start [s]")
            pdf.savefig(figure)
            base.plt.close(figure)

        value = np.column_stack((position_error, orientation_error))
        labels = (
            "dx [m]",
            "dy [m]",
            "dz [m]",
            "dRx [rad]",
            "dRy [rad]",
            "dRz [rad]",
        )
        figure, axes = base.plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes.ravel()):
            axis.plot(relative_time, value[:, component])
            axis.set_ylabel(labels[component])
            axis.grid(True, alpha=0.25)
        axes[0, 0].set_title("Local-polynomial pose residual")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        pdf.savefig(figure)
        base.plt.close(figure)

        for title, value, labels in (
            (
                "Translational derivatives at raw centered evaluation times",
                np.column_stack(
                    (
                        collocation.sensor_velocity_world,
                        collocation.sensor_acceleration_world,
                    )
                ),
                (
                    "vx [m/s]",
                    "vy [m/s]",
                    "vz [m/s]",
                    "ax [m/s2]",
                    "ay [m/s2]",
                    "az [m/s2]",
                ),
            ),
            (
                "Geometric-SG rotational derivatives (body frame)",
                np.column_stack(
                    (
                        collocation.body_angular_velocity,
                        collocation.body_angular_acceleration,
                    )
                ),
                (
                    "wx [rad/s]",
                    "wy [rad/s]",
                    "wz [rad/s]",
                    "alphax [rad/s2]",
                    "alphay [rad/s2]",
                    "alphaz [rad/s2]",
                ),
            ),
        ):
            figure, axes = base.plt.subplots(
                3,
                2,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes.ravel()):
                axis.plot(
                    collocation_relative,
                    value[:, component],
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0, 0].set_title(title)
            axes[-1, 0].set_xlabel("time [s]")
            axes[-1, 1].set_xlabel("time [s]")
            pdf.savefig(figure)
            base.plt.close(figure)

        acceleration_variance = np.diagonal(
            collocation.sensor_acceleration_world_covariance,
            axis1=1,
            axis2=2,
        )
        acceleration_std = np.sqrt(
            np.maximum(acceleration_variance, 0.0)
        )
        figure, axes = base.plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                collocation_relative,
                acceleration_std[:, component],
            )
            axis.set_ylabel(
                ("ax", "ay", "az")[component] + " local std [m/s2]"
            )
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Translation second-derivative covariance from local LS residuals"
        )
        axes[-1].set_xlabel("time [s]")
        pdf.savefig(figure)
        base.plt.close(figure)

        figure, axes = base.plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            collocation_relative,
            collocation.window_sample_count,
        )
        axes[0].axhline(
            sg.MINIMUM_WINDOW_POINTS,
            linestyle="--",
            label="minimum required",
        )
        axes[0].set_ylabel("samples/window")
        axes[0].legend(loc="best")
        axes[1].semilogy(
            collocation_relative,
            collocation.position_fit_condition_number,
        )
        axes[1].set_ylabel("cond(position LS)")
        axes[2].semilogy(
            collocation_relative,
            collocation.rotation_fit_condition_number,
        )
        axes[2].set_ylabel("cond(SO3 LS)")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Savitzky-Golay window diagnostics; W={:.6g}s".format(
                trajectory.window_seconds
            )
        )
        pdf.savefig(figure)
        base.plt.close(figure)


def _write_raw_residual_wrench_pdf(
    path: Path,
    time_axis: np.ndarray,
    body_wrench: np.ndarray,
    window_seconds: float,
) -> None:
    time_value = np.asarray(time_axis, dtype=float)
    wrench = np.asarray(body_wrench, dtype=float)
    if (
        time_value.ndim != 1
        or wrench.shape != (time_value.size, 6)
        or time_value.size < 1
    ):
        raise ValueError("raw residual-wrench diagnostic has invalid shape")
    relative = time_value - time_value[0]
    names = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
    units = ("N", "N", "N", "N m", "N m", "N m")
    with base.PdfPages(path) as pdf:
        for offset, title in (
            (0, "Raw inverse-dynamics residual body force"),
            (3, "Raw inverse-dynamics residual body torque"),
        ):
            figure, axes = base.plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(relative, wrench[:, component])
                axis.axhline(0.0, linewidth=0.7, alpha=0.5)
                axis.set_ylabel(
                    "{} [{}]".format(names[component], units[component])
                )
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; degree-5 SG W={:.6g}s; no replay-grid resampling".format(
                    title, window_seconds
                )
            )
            axes[-1].set_xlabel("time from first SG dynamics center [s]")
            pdf.savefig(figure)
            base.plt.close(figure)


def _rewrite_npz(
    bag_directory: Path,
    bag: base.BagSplineData,
) -> None:
    old_path = bag_directory / "spline_dynamics.npz"
    if not old_path.is_file():
        return
    with np.load(old_path, allow_pickle=False) as archive:
        values = {}
        for name in archive.files:
            new_name = (
                "savgol_" + name[len("spline_") :]
                if name.startswith("spline_")
                else name
            )
            values[new_name] = archive[name]

    collocation = bag.collocation
    values.update(
        {
            "raw_pose_time": np.asarray(
                bag.flight.pose.times,
                dtype=float,
            ),
            "raw_pose_sensor_position": np.asarray(
                bag.flight.pose.positions,
                dtype=float,
            ),
            "raw_pose_sensor_orientation_xyzw": np.asarray(
                bag.flight.pose.orientations_xyzw,
                dtype=float,
            ),
            "savgol_window_seconds": np.asarray(
                bag.spline_selection.spline.window_seconds,
                dtype=float,
            ),
            "savgol_polynomial_degree": np.asarray(
                sg.POLYNOMIAL_DEGREE,
                dtype=int,
            ),
            "savgol_window_sample_count": (
                collocation.window_sample_count
            ),
            "savgol_position_fit_condition_number": (
                collocation.position_fit_condition_number
            ),
            "savgol_rotation_fit_condition_number": (
                collocation.rotation_fit_condition_number
            ),
            "savgol_sensor_position_covariance": (
                collocation.sensor_position_covariance
            ),
            "savgol_sensor_velocity_world_covariance": (
                collocation.sensor_velocity_world_covariance
            ),
            "savgol_sensor_acceleration_world_covariance": (
                collocation.sensor_acceleration_world_covariance
            ),
        }
    )
    raw_wrench_time = np.asarray(
        values["raw_inferred_external_body_wrench_time"], dtype=float
    )
    raw_wrench = np.asarray(
        values["raw_inferred_external_body_wrench"], dtype=float
    )
    statistics = {
        "definition": "required_body_wrench - modeled_body_wrench at raw centered SG evaluation times",
        "sample_count": int(raw_wrench.shape[0]),
        "mean": np.mean(raw_wrench, axis=0),
        "std": np.std(raw_wrench, axis=0, ddof=1) if raw_wrench.shape[0] > 1 else np.full(6, np.nan),
        "rms": np.sqrt(np.mean(raw_wrench * raw_wrench, axis=0)),
        "time_interval_seconds": (float(raw_wrench_time[0]), float(raw_wrench_time[-1])),
        "uses_replay_optimization": False,
        "uses_legacy_uniform_rollout_grid": False,
    }
    _ACTIVE_RAW_WRENCH_STATS[str(bag.specification.bag_id)] = statistics
    _write_raw_residual_wrench_pdf(
        bag_directory / "raw_residual_wrench.pdf",
        raw_wrench_time,
        raw_wrench,
        float(bag.spline_selection.spline.window_seconds),
    )

    new_path = bag_directory / "savgol_dynamics.npz"
    np.savez_compressed(new_path, **values)
    old_path.unlink()


def _rewrite_diagnostics(
    diagnostics: dict[str, Any],
    bag: base.BagSplineData,
) -> dict[str, Any]:
    old = diagnostics.pop("spline", {})
    trajectory = bag.spline_selection.spline
    candidate = bag.spline_selection.candidates[0]
    diagnostics.pop("collocation_count", None)
    diagnostics["evaluation_count"] = int(
        bag.collocation_time.size
    )
    diagnostics["savgol"] = {
        "degree": sg.POLYNOMIAL_DEGREE,
        "window_seconds": float(trajectory.window_seconds),
        "window_definition": (
            "physical time width; all raw mocap pose samples inside each "
            "local window are used"
        ),
        "translation": "degree-5 local least-squares polynomial in R^3",
        "rotation": (
            "degree-5 geometric Savitzky-Golay local polynomial on SO(3) "
            "(Jongeneel-Saccon IROS 2022)"
        ),
        "raw_pose_resampled_before_fit": False,
        "parameter_loss_evaluation_times": (
            "all raw pose timestamps with a complete centered W-second "
            "window, intersected with actuator/dynamics support"
        ),
        "minimum_required_points": sg.MINIMUM_WINDOW_POINTS,
        "minimum_window_sample_count": int(
            candidate.minimum_window_sample_count
        ),
        "maximum_window_sample_count": int(
            candidate.maximum_window_sample_count
        ),
        "raw_fit_interval_seconds": old.get(
            "fit_interval_seconds",
            np.asarray(
                (trajectory.start_time, trajectory.end_time),
                dtype=float,
            ),
        ),
        "parameter_estimation_interval_seconds": old.get(
            "parameter_estimation_interval_seconds",
            np.asarray(
                (
                    bag.collocation_time[0],
                    bag.collocation_time[-1],
                ),
                dtype=float,
            ),
        ),
        "actual_boundary_exclusion_seconds_start_end": old.get(
            "actual_boundary_exclusion_seconds_start_end",
        ),
        "centered_window_only_for_parameter_loss": True,
        "one_sided_shifted_full_window_used_only_for_edge_diagnostics": True,
        "fit_metrics": old.get("fit_metrics"),
        "translation_covariance": (
            "local ordinary-least-squares covariance estimated from position "
            "fit residuals when residual degrees of freedom are positive"
        ),
        "window_diagnostics": sg.candidate_payload(candidate),
    }
    rotor_timing = _timestamp_interval_statistics(
        bag.flight.rotor_command.all_times
    )
    gimbal_timing = _timestamp_interval_statistics(
        bag.flight.gimbal_command.all_times
    )
    diagnostics["command_timestamp_intervals"] = {
        "rotor": rotor_timing,
        "gimbal": gimbal_timing,
        "selected_lag_seconds": None,
        "note": (
            "Publish/update intervals are measured directly from rosbag "
            "timestamps; no command period is hard-coded into the SG front end."
        ),
    }
    raw_wrench_statistics = _ACTIVE_RAW_WRENCH_STATS.get(
        str(bag.specification.bag_id)
    )
    if raw_wrench_statistics is not None:
        diagnostics["raw_inverse_dynamics_residual_wrench_statistics"] = (
            raw_wrench_statistics
        )
    inferred = diagnostics.get("inferred_external_wrench")
    if isinstance(inferred, dict):
        definition = inferred.get("definition")
        if isinstance(definition, str):
            inferred["definition"] = definition.replace(
                "observed pose spline",
                "geometric Savitzky-Golay pose estimate",
            )
    return diagnostics


def _rewrite_json_outputs(output_directory: Path, initial_delay: float) -> None:
    root_path = output_directory / "result.json"
    if not root_path.is_file():
        raise RuntimeError("base estimator did not produce result.json")

    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["schema"] = SCHEMA
    root["method"] = {
        "name": "deterministic_savgol_dynamics",
        "description": (
            "raw-pose degree-5 local-polynomial gradient matching with "
            "geometric Savitzky-Golay filtering on SO(3), shared physical "
            "parameters, and command lag"
        ),
        "uses_multiple_shooting_nodes": False,
        "uses_continuity_constraints": False,
        "uses_augmented_lagrangian": False,
        "sensor_channels_in_parameter_loss": False,
        "pose_role": (
            "raw mocap pose supplies local degree-5 R3/SO3 polynomial fits"
        ),
        "raw_pose_resampled_before_local_polynomial_fit": False,
        "polynomial_degree": sg.POLYNOMIAL_DEGREE,
        "window_seconds": _window(),
        "command_mode_during_search": "quintic smoothstep ZOH",
        "command_mode_final": "strict ZOH",
    }
    old_settings = root.get("settings", {})
    root["settings"] = {
        "window_seconds": _window(),
        "polynomial_degree": sg.POLYNOMIAL_DEGREE,
        "minimum_required_pose_samples_per_window": (
            sg.MINIMUM_WINDOW_POINTS
        ),
        "sample_step_seconds_for_rollout_diagnostics_only": old_settings.get(
            "sample_step_seconds"
        ),
        "integration_step_seconds": old_settings.get(
            "integration_step_seconds"
        ),
        "physical_coordinate_bounds": old_settings.get(
            "physical_coordinate_bounds"
        ),
        "delay_bounds_seconds": old_settings.get(
            "delay_bounds_seconds"
        ),
        "smoothstep_width_fractions": old_settings.get(
            "smoothstep_width_fractions"
        ),
        "pose_resampling_used_in_parameter_loss": False,
        "collocation_grid_used_in_parameter_loss": False,
    }
    initial = root.setdefault("initial_estimate", {})
    initial["delay_seconds"] = float(initial_delay)
    initial["delay_default"] = (
        "zero unless --initial-delay is explicitly supplied"
    )

    selected_lag = float(root.get("selection", {}).get("delay_seconds", math.nan))
    for diagnostics in root.get("bag_diagnostics", []):
        bag_id = str(diagnostics.get("id"))
        bag = _ACTIVE_BAGS.get(bag_id)
        if bag is not None:
            _rewrite_diagnostics(diagnostics, bag)
            timing = diagnostics.get("command_timestamp_intervals")
            if isinstance(timing, dict):
                timing["selected_lag_seconds"] = selected_lag
                for channel in ("rotor", "gimbal"):
                    block = timing.get(channel)
                    if isinstance(block, dict):
                        median = block.get("median_seconds")
                        block["selected_lag_over_median_interval"] = (
                            None
                            if median is None or not np.isfinite(selected_lag)
                            else float(selected_lag / float(median))
                        )

    outputs = root.setdefault("outputs", {})
    bag_outputs = outputs.get("bags", {})
    for bag_id, entry in bag_outputs.items():
        if isinstance(entry, dict):
            entry.pop("spline_dynamics_npz", None)
            entry["savgol_dynamics_npz"] = (
                "bags/{}/savgol_dynamics.npz".format(bag_id)
            )
            entry["savgol_fit_pdf"] = (
                "bags/{}/savgol_fit.pdf".format(bag_id)
            )
            entry["raw_residual_wrench_pdf"] = (
                "bags/{}/raw_residual_wrench.pdf".format(bag_id)
            )

    root_path.write_text(
        json.dumps(
            _json_sanitize(root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for bag_id, bag in _ACTIVE_BAGS.items():
        bag_directory = output_directory / "bags" / bag_id
        result_path = bag_directory / "result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["schema"] = SCHEMA + "/bag-result"
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                _rewrite_diagnostics(diagnostics, bag)
                timing = diagnostics.get("command_timestamp_intervals")
                if isinstance(timing, dict):
                    selected_lag = float(payload.get("shared_delay_seconds", math.nan))
                    timing["selected_lag_seconds"] = selected_lag
                    for channel in ("rotor", "gimbal"):
                        block = timing.get(channel)
                        if isinstance(block, dict):
                            median = block.get("median_seconds")
                            block["selected_lag_over_median_interval"] = (
                                None
                                if median is None or not np.isfinite(selected_lag)
                                else float(selected_lag / float(median))
                            )
            outputs = payload.setdefault("outputs", {})
            outputs.pop("spline_dynamics_npz", None)
            outputs["savgol_dynamics_npz"] = "savgol_dynamics.npz"
            outputs["savgol_fit_pdf"] = "savgol_fit.pdf"
            outputs["raw_residual_wrench_pdf"] = "raw_residual_wrench.pdf"
            result_path.write_text(
                json.dumps(
                    _json_sanitize(payload),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate shared rigid-body parameters from raw mocap pose using "
            "degree-5 local polynomial / geometric Savitzky-Golay derivatives."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--vehicle-model-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        required=True,
        help=(
            "Physical SG window width W in seconds. All raw pose samples in "
            "each window are used; degree is fixed at 5."
        ),
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=0.05,
        help=(
            "Uniform output step used only by legacy forward-rollout "
            "diagnostics; it does not resample the SG parameter-loss input."
        ),
    )
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--smooth-max-nfev", type=int, default=60)
    parser.add_argument("--strict-max-nfev", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.20),
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=None,
        help="Initial command lag. Default is exactly 0 s.",
    )
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(0.50, 0.20, 0.05),
    )
    parser.add_argument(
        "--zoh-polish-radius",
        type=float,
        default=0.004,
    )
    parser.add_argument(
        "--zoh-polish-step",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--zoh-polish-top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def run(arguments: argparse.Namespace) -> int:
    global _ACTIVE_WINDOW_SECONDS
    _ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)
    _ACTIVE_BAGS.clear()
    _ACTIVE_RAW_WRENCH_STATS.clear()

    # Required by base.run's old report serializer but intentionally unused
    # by the SG front end.
    arguments.spline_cv_folds = 0
    arguments.maximum_spline_acceleration = math.inf
    arguments.maximum_spline_angular_acceleration = math.inf

    # Command-lag initialization is deliberately zero by default.
    if arguments.initial_delay is None:
        arguments.initial_delay = 0.0

    # Patch only this process.  The existing spline file on disk is untouched.
    base.SCHEMA = SCHEMA
    base.OUTPUT_SUBDIRECTORY = OUTPUT_SUBDIRECTORY
    base.DATA_DICTIONARY_SOURCE = DATA_DICTIONARY_SOURCE
    base.load_spline_config = load_spline_config
    base._build_bag_data = _build_bag_data
    base._validate_arguments = _validate_arguments
    base.candidate_payload = sg.candidate_payload
    base._parameter_lines = _parameter_lines

    original_stdout = sys.stdout
    sys.stdout = _RewritingStdout(original_stdout)
    try:
        try:
            result = base.run(arguments)
        except ValueError as error:
            # Window feasibility and local-polynomial conditioning failures
            # should stop before optimization with a concise diagnostic rather
            # than a long traceback from the compatibility backend.
            raise SystemExit(str(error)) from error
    finally:
        sys.stdout = original_stdout

    output_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
    )
    for bag_id, bag in _ACTIVE_BAGS.items():
        bag_directory = output_directory / "bags" / bag_id
        _write_savgol_fit_pdf(
            bag_directory / "savgol_fit.pdf",
            bag,
        )
        _rewrite_npz(bag_directory, bag)

    _rewrite_json_outputs(output_directory, float(arguments.initial_delay))
    print(
        "Savitzky-Golay reports written to {}".format(output_directory),
        flush=True,
    )
    return int(result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
