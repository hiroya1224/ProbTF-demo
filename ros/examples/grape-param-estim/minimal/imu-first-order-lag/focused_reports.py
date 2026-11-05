"""Write only the requested sensor, trajectory and applied-torque figures.

The caller supplies the reconstruction.  This module neither fits a model nor
changes an orientation law, trajectory, covariance or historical publication.
Every numerical curve and band used by the plots is returned for archival.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from imu_reports import (PdfPages, first_order_measurement_response, plt,
                         rotations, rpy_standard_deviation)


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def _sensor_taus(taus):
    if isinstance(taus, Mapping):
        values = [taus["gyro_time_constant_seconds"],
                  taus["accel_time_constant_seconds"]]
    else:
        values = _array(taus, (3,), "taus [thrust, gyro, accel]")[1:]
    values = _array(values, (2,), "gyro/accel time constants")
    if np.any(values < 0):
        raise ValueError("sensor time constants must be nonnegative")
    return values


def _position_std(branch, count, label):
    # Prefer the saved rectangular factor, when available, to avoid forming a
    # covariance merely to recover its diagonal.  No eigenvalue operation or
    # display-only adjustment is applied to either representation.
    if "position_covariance_factor_fc" in branch:
        factor = np.asarray(branch["position_covariance_factor_fc"], dtype=float)
        if factor.ndim != 3 or factor.shape[:2] != (count, 3) or not np.isfinite(factor).all():
            raise ValueError(f"{label}: position factor must have finite shape (N, 3, K)")
        return np.sqrt(np.sum(factor**2, axis=2))
    covariance = _array(branch["covariance_position_fc"], (count, 3, 3),
                        f"{label}: position covariance")
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    if np.any(variance < 0):
        raise ValueError(f"{label}: position variance is negative")
    return np.sqrt(variance)


def _trajectory_arrays(trajectories, time):
    count = len(time)
    result = {}
    for label, branch in trajectories.items():
        if "time" in branch and not np.array_equal(np.asarray(branch["time"]), time):
            raise ValueError(f"{label}: trajectory must use the original native IMU clock")
        position = _array(branch["position_fc_world"], (count, 3), f"{label}: position")
        quaternion = _array(branch["quaternion_wxyz"], (count, 4), f"{label}: quaternion")
        factor = branch.get("rotation_covariance_factor")
        covariance = branch.get("covariance_rotation")
        if factor is not None:
            factor = np.asarray(factor, dtype=float)
            if factor.ndim != 3 or factor.shape[:2] != (count, 3) or not np.isfinite(factor).all():
                raise ValueError(f"{label}: rotation factor must have finite shape (N, 3, K)")
        elif covariance is None:
            raise ValueError(f"{label}: a saved rotation covariance or factor is required")
        if covariance is not None:
            covariance = _array(covariance, (count, 3, 3), f"{label}: rotation covariance")
        angles = np.rad2deg(np.unwrap(rotations(quaternion).as_euler("xyz"), axis=0))
        angle_std = rpy_standard_deviation(quaternion, covariance, factor)
        if not np.isfinite(angle_std).all():
            raise ValueError(f"{label}: Euler uncertainty is non-finite")
        result[str(label)] = {
            "position_fc_world": position,
            "position_standard_deviation": _position_std(branch, count, label),
            "quaternion_wxyz": quaternion,
            "rpy_degrees": angles,
            "rpy_standard_deviation_degrees": angle_std,
        }
    return result


def _reference_arrays(reference, data, time):
    """Keep original mocap rows inside support; never interpolate the reference."""
    if reference is None:
        keys = ("mocap_time", "mocap_position", "mocap_quaternion_xyzw")
        if not all(key in data for key in keys):
            return {}, {"available": False}
        reference = dict(zip(("time", "position_world", "quaternion_xyzw"),
                             (data[key] for key in keys)))
    stamp = np.asarray(reference["time"], dtype=float)
    if stamp.ndim != 1 or not np.isfinite(stamp).all():
        raise ValueError("reference requires finite original mocap timestamps")
    count = len(stamp)
    position = _array(reference["position_world"], (count, 3), "mocap reference position")
    quaternion = _array(reference["quaternion_xyzw"], (count, 4), "mocap reference quaternion")
    index = np.flatnonzero((stamp >= time[0]) & (stamp <= time[-1]))
    stamp, position, quaternion = stamp[index], position[index], quaternion[index]
    angles = (np.rad2deg(np.unwrap(rotations(quaternion[:, [3, 0, 1, 2]]).as_euler("xyz"), axis=0))
              if len(index) else np.empty((0, 3)))
    arrays = {"time": stamp, "elapsed_seconds": stamp - time[0],
              "source_index": index, "position_world": position,
              "quaternion_xyzw": quaternion, "rpy_degrees": angles}
    metadata = {"available": bool(len(index)), "original_count": count,
                "displayed_count": len(index),
                "clock": "original mocap timestamps cropped to native IMU support; no interpolation or extrapolation",
                "role": "read-only black reference; never used to update supplied reconstruction",
                "uncertainty": "no reference covariance supplied or fabricated"}
    return arrays, metadata


def write_requested_figures(output, case, data, common_model, taus, trajectories=None,
                            reference=None, *, show_model_uncertainty=False):
    """Write three common figures and optionally the two trajectory figures.

    ``taus`` is either ``[tau_thrust, tau_gyro, tau_accel]`` or the existing
    ``estimate['time_constants_seconds']`` mapping.  ``common_model`` is an
    actual ``replay['model']`` branch, including the applied residual wrench.
    Optional ``trajectories`` maps display labels to caller-created replay
    branches; the writer does not invent a second orientation-conditioned
    model replay or silently substitute IMU dead reckoning.

    ``reference`` optionally supplies original mocap ``time``,
    ``position_world`` and ``quaternion_xyzw`` arrays.  If omitted, the same
    arrays are taken from the corresponding ``data['mocap_*']`` fields when
    present.  Only original rows inside IMU support are drawn, without a
    fabricated reference uncertainty.  Euler display branches may differ by
    integer full turns; the native quaternion and position means never change.

    Return ``arrays``, ``metadata`` and relative ``files``.  Array keys for
    trajectory ``i`` start with ``trajectory_i__``; metadata retains its label.
    Sensor responses condition on the supplied raw initial reading, use exact
    ZOH physical inputs on native intervals, and do not include uncertainty
    bands. By default model position covariance bands are omitted from the
    figure only; their original numerical arrays remain in the result.
    Optional position bands retain the supplied propagated covariance upper
    bound; attitude bands use the supplied posterior-law local approximation.
    Neither is claimed to be a 95 percent coverage guarantee.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    time = np.asarray(data["time"], dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("figures require finite, strictly increasing native IMU timestamps")
    if not np.array_equal(np.asarray(common_model["time"]), time):
        raise ValueError("common model must use the original native IMU clock")
    count = len(time)
    elapsed = time - time[0]
    sensor_taus = _sensor_taus(taus)
    arrays = {"time": time.copy(), "elapsed_seconds": elapsed}
    for sensor, physical_key, tau in zip(
            ("gyro", "accel"),
            ("predicted_gyro_body", "predicted_specific_force_body"), sensor_taus):
        raw = _array(data[sensor], (count, 3), f"raw {sensor}")
        physical = _array(common_model[physical_key], (count, 3), f"model physical {sensor}")
        arrays[f"{sensor}_raw"] = raw
        arrays[f"{sensor}_physical_model_plus_applied_residual"] = physical
        arrays[f"{sensor}_forward_lag_response"] = first_order_measurement_response(
            time, physical, float(tau), raw[0])
    wrench = _array(common_model["applied_residual_wrench_body"], (count, 6),
                    "actually applied residual wrench")
    arrays["applied_residual_torque_body_cog"] = wrench[:, 3:]
    if trajectories is not None and not trajectories:
        raise ValueError("trajectories must be None or contain at least one supplied branch")
    tracks = {} if trajectories is None else _trajectory_arrays(trajectories, time)
    reference_arrays, reference_metadata = _reference_arrays(reference, data, time)
    arrays.update({"reference__" + key: value for key, value in reference_arrays.items()})
    trajectory_labels = []
    for index, (label, values) in enumerate(tracks.items()):
        prefix = f"trajectory_{index}__"
        turns = (360 * np.round((reference_arrays["rpy_degrees"][0]
                                - values["rpy_degrees"][0]) / 360)
                 if reference_metadata["available"] else np.zeros(3))
        # Display branch alignment changes only an integer number of turns.
        # Do not mutate either the reconstruction or its native quaternions.
        values["rpy_degrees"] = values["rpy_degrees"] + turns
        values["rpy_integer_turn_offset_degrees"] = turns
        trajectory_labels.append({"label": label, "array_prefix": prefix})
        arrays.update({prefix + key: value for key, value in values.items()})

    files = []
    figure_types = []
    model_color, raw_color = "#2b8c49", "#555555"
    track_colors = plt.get_cmap("tab10")
    settings = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                "figure.titlesize": 16, "legend.fontsize": 10, "lines.linewidth": 1.45}
    with plt.rc_context(settings), PdfPages(destination / "requested_results.pdf") as combined:
        def save(figure, stem):
            figure.tight_layout(rect=(0, .055, 1, .945))
            try:
                combined.savefig(figure)
                for extension in ("png", "pdf"):
                    name = f"{stem}.{extension}"
                    figure.savefig(destination / name, dpi=180)
                    files.append(name)
            finally:
                plt.close(figure)
            figure_types.append(stem)

        def setup(axes):
            for axis in np.asarray(axes).flat:
                axis.grid(alpha=.23)
                axis.set_xlabel("time since first native IMU sample [s]")
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, 5))

        for sensor, unit in (("gyro", "rad/s"), ("accel", "m/s²")):
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            setup(axes)
            fig.suptitle(f"{case} — {sensor}: raw observation and reconstructed sensor response")
            for coordinate, axis in enumerate(axes):
                axis.plot(elapsed, arrays[f"{sensor}_raw"][:, coordinate],
                          color=raw_color, alpha=.65, label="raw sensor")
                axis.plot(elapsed, arrays[f"{sensor}_forward_lag_response"][:, coordinate],
                          color=model_color, label="model + applied residual, with sensor lag")
                axis.set_ylabel(f"{'xyz'[coordinate]} [{unit}]")
                axis.legend(loc="best")
            tau = sensor_taus[0 if sensor == "gyro" else 1]
            fig.text(.055, .018,
                     f"τ = {tau:.6g} s; forward response τ ẏ + y = physical model input. "
                     "Native-interval ZOH; initialized by the first raw reading.", fontsize=10)
            save(fig, f"{sensor}_reconstruction")

        if tracks:
            for kind, value_key, std_key, unit, coordinate_labels, stem in (
                    ("FC position", "position_fc_world", "position_standard_deviation", "m",
                     ("x", "y", "z"), "position_reconstruction"),
                    ("attitude", "rpy_degrees", "rpy_standard_deviation_degrees", "deg",
                     ("roll", "pitch", "yaw"), "attitude_reconstruction")):
                columns = 2 if kind == 'attitude' or show_model_uncertainty else 1
                fig, axes = plt.subplots(3, columns, figsize=(15 if columns == 2 else 12, 9),
                                         sharex=True, squeeze=False)
                setup(axes)
                fig.suptitle(f"{case} — {kind}: supplied reconstructions and uncertainty")
                axes[0, 0].set_title("Mean positions" if kind == "FC position" else "Representative attitudes")
                if columns == 2:
                    axes[0, 1].set_title("Same estimates with full ±2σ range")
                if reference_metadata["available"]:
                    reference_value = reference_arrays["position_world" if kind == "FC position" else "rpy_degrees"]
                    for coordinate in range(3):
                        for column in range(columns):
                            axes[coordinate, column].plot(reference_arrays["elapsed_seconds"],
                                                          reference_value[:, coordinate], color="black",
                                                          linewidth=1.8, label="mocap reference")
                for index, (label, values) in enumerate(tracks.items()):
                    color = track_colors(index % 10)
                    mean, std = values[value_key], values[std_key]
                    for coordinate in range(3):
                        for column in range(columns):
                            axis = axes[coordinate, column]
                            axis.plot(elapsed, mean[:, coordinate], color=color, label=label)
                            if column == 1:
                                axis.fill_between(elapsed, mean[:, coordinate] - 2 * std[:, coordinate],
                                                  mean[:, coordinate] + 2 * std[:, coordinate],
                                                  color=color, alpha=.18)
                            axis.set_ylabel(f"{coordinate_labels[coordinate]} [{unit}]")
                for axis in axes.flat:
                    axis.legend(loc="best")
                explanation = (
                    "Model position covariance bands omitted by request; computed covariance data retained.\n"
                    if columns == 1 else
                    ("Position: propagated covariance upper bound.\n" if kind == 'FC position' else
                     "Attitude: IMU posterior-law local Euler approximation.\n")
                    + "±2σ is not a 95% guarantee. Right column retains complete bands; ")
                explanation += ("black reference uses original mocap stamps."
                    if reference_metadata['available'] else "no mocap reference supplied.")
                fig.text(.055, .012, explanation, fontsize=10)
                save(fig, stem)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        setup(axes)
        fig.suptitle(f"{case} — actually applied residual torque")
        for coordinate, axis in enumerate(axes):
            axis.plot(elapsed, arrays["applied_residual_torque_body_cog"][:, coordinate],
                      color=model_color, label="added residual torque mean")
            axis.set_ylabel(f"T{'xyz'[coordinate]} [N m]")
            axis.legend(loc="best")
        fig.text(.055, .018,
                 "Body coordinates, about the center of gravity; the same torque mean added to reconstruction dynamics.",
                 fontsize=10)
        save(fig, "applied_residual_torque")
    files.append("requested_results.pdf")
    metadata = {
        "schema": "grape-param-estim/imu-first-order-lag-focused-figures/v3",
        "case": str(case), "figure_count": len(figure_types), "figure_types": figure_types,
        "files": files, "native_count": count,
        "first_native_time": float(time[0]), "last_native_time": float(time[-1]),
        "clock": "original native IMU timestamps; elapsed display subtracts only the first timestamp",
        "gyro_time_constant_seconds": float(sensor_taus[0]),
        "accel_time_constant_seconds": float(sensor_taus[1]),
        "sensor_response": "actual model + applied residual physical inputs, forward first-order lag; exact native-interval ZOH",
        "sensor_response_initial_condition": "conditioned on first raw reading; no sensor uncertainty band requested",
        "sensor_units": {"gyro": "rad/s", "accel": "m/s^2"},
        "applied_torque": "actual replay applied_residual_wrench_body columns 3:6; body coordinates about CoG; N m",
        "torque_uncertainty": "no band requested; no replacement uncertainty computed",
        "trajectories": trajectory_labels,
        "trajectory_interpretation": "caller-supplied reconstructions; writer performs no model replay, orientation update or substitution",
        "trajectory_uncertainty": "unaltered supplied marginal covariance/factor; IMU attitude ±2 standard deviations retained; model position bands omitted by default, numerical arrays retained; not a 95 percent guarantee",
        "model_position_uncertainty_shown": bool(show_model_uncertainty and tracks),
        "computed_covariance_data_retained": True,
        "position_is_upper_bound": bool(tracks),
        "position_uncertainty": "propagated covariance upper bound, retaining supplied parameter cross and declared unknown pose/history bounds",
        "attitude_uncertainty": "native posterior-law local right-rotation covariance proxy; Euler first-order transformation",
        "attitude_display": "unwrapped xyz Euler degrees, initially aligned only by integer full turns to the first displayed reference; native quaternions unchanged",
        "reference": reference_metadata,
        "estimator_changed": False, "monte_carlo": False,
    }
    return {"arrays": arrays, "metadata": metadata, "files": files}
