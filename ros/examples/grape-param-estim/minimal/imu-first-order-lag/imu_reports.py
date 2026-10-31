"""Read-only reports for the separate native-IMU first-order-lag experiment.

All uncertainty below is taken from the estimated law or propagated rectangular
factors.  Plotting never changes those laws, clips eigenvalues, or draws samples.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-imu-first-order-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
WRENCH_LABELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
COLORS = {"mocap": "black", "GA": "tab:blue", "GA + magnetic output": "tab:orange",
          "model + residual wrench": "tab:green", "IMU dead reckoning": "tab:purple"}


def rotations(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def orientation_comparison(mocap_time, mocap_quaternion_xyzw, estimate_time,
                           estimate_quaternion_wxyz):
    """Interpolate reconstruction to original supported mocap stamps only."""
    mt, et = np.asarray(mocap_time), np.asarray(estimate_time)
    mask = (mt >= et[0]) & (mt <= et[-1])
    index = np.flatnonzero(mask)
    if not index.size:
        raise ValueError("no original mocap timestamps lie inside reconstruction support")
    selected = mt[index]
    reference = Rotation.from_quat(np.asarray(mocap_quaternion_xyzw)[index])
    reconstructed = Slerp(et, rotations(estimate_quaternion_wxyz))(selected)
    error = np.rad2deg((reference.inv() * reconstructed).magnitude())
    return {"time": selected, "mocap_index": index, "mask": mask,
            "so3_error_deg": error,
            "statistics": {"count": int(index.size), "rms_deg": float(np.sqrt(np.mean(error**2))),
                           "maximum_deg": float(np.max(error)), "final_deg": float(error[-1]),
                           "first_time": float(selected[0]), "last_time": float(selected[-1]),
                           "clock": "original mocap header timestamps inside IMU support; reconstruction Slerp; no extrapolation"}}


def rpy_standard_deviation(q_wxyz, right_covariance, right_factor=None):
    """Linearized xyz Euler standard deviation from body/right angle covariance."""
    angles = rotations(q_wxyz).as_euler("xyz")
    result = np.empty_like(angles)
    for k, (roll, pitch, _) in enumerate(angles):
        sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
        # body angular increment = E * xyz Euler increment
        e = np.array([[1., 0., -sp], [0., cr, sr * cp], [0., -sr, cr * cp]])
        j = np.linalg.solve(e, np.eye(3))
        if right_factor is None:
            covariance = j @ np.asarray(right_covariance)[k] @ j.T
            result[k] = np.sqrt(np.diag(covariance))
        else:
            factor = j @ np.asarray(right_factor)[k]
            result[k] = np.sqrt(np.sum(factor**2, axis=1))
    return np.rad2deg(result)


def first_order_measurement_response(time, physical_input, tau, initial_measurement):
    """tau*y_dot+y=x, exact ZOH x over each native interval; raw initial y."""
    t, x = np.asarray(time), np.asarray(physical_input)
    response = np.empty_like(x, dtype=float)
    response[0] = initial_measurement
    if float(tau) < 0:
        raise ValueError("sensor time constant must be nonnegative")
    for k in range(1, len(t)):
        if tau == 0:
            response[k] = x[k]
        else:
            alpha = np.exp(-(t[k] - t[k - 1]) / tau)
            response[k] = alpha * response[k - 1] + (1 - alpha) * x[k - 1]
    return response


def covariance_tiles(size, block=9):
    return [(slice(i, min(i + block, size)), slice(j, min(j + block, size)))
            for i in range(0, size, block) for j in range(0, size, block)]


def covariance_wireframes(mean, factor, sigma=2., count=61):
    """Three deterministic great circles of the factor's covariance ellipsoid."""
    u, singular, _ = np.linalg.svd(np.asarray(factor), full_matrices=False)
    basis = u * (sigma * singular)[None, :]
    angle = np.linspace(0., 2 * np.pi, count)
    circles = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        coordinate = np.zeros((3, count))
        coordinate[a], coordinate[b] = np.cos(angle), np.sin(angle)
        circles.append(np.asarray(mean)[:, None] + basis @ coordinate)
    return circles


def applied_wrench_distribution(model_replay, native_time):
    """Read the exact applied wrench law; never substitute a display residual."""
    time = np.asarray(model_replay["time"])
    if not np.array_equal(time, np.asarray(native_time)):
        raise ValueError("applied residual-wrench report requires the original replay clock")
    mean = np.asarray(model_replay["applied_residual_wrench_body"])
    factor = np.asarray(model_replay["applied_residual_wrench_covariance_factor"])
    if mean.shape != (len(time), 6) or factor.ndim != 3 or factor.shape[:2] != mean.shape:
        raise ValueError("applied residual-wrench mean/factor do not share six-dimensional native rows")
    return mean, factor, np.sqrt(np.sum(factor**2, axis=2))


def _lomb(time, wrench):
    # Reuse the original spectrum helper only. No old estimator/replay is run.
    spec = importlib.util.spec_from_file_location("imu_readonly_original_reports", HERE.parent / "first-order-lag/reports.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.wrench_lomb_scargle(time, wrench)


def _units(label):
    if label == "mass_kg":
        return "kg"
    if label.startswith("inertia_"):
        return "kg m^2"
    if label.startswith("cog_"):
        return "m"
    if label.endswith("seconds"):
        return "s"
    return "dimensionless"


def _save_parameter_csv(output, distribution):
    labels = distribution["physical_si_labels"]
    value = np.asarray(distribution["physical_si_value"])
    covariance = np.asarray(distribution["covariance_physical_si"])
    if value.shape != (17,) or covariance.shape != (17, 17) or len(labels) != 17:
        raise ValueError("report requires the complete actual 17-parameter distribution")
    with (output / "parameter_values_si.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["index", "parameter", "unit", "estimated_value", "standard_deviation"])
        for k, label in enumerate(labels):
            writer.writerow([k + 1, label, _units(label), format(value[k], ".17g"),
                             format(np.sqrt(covariance[k, k]), ".17g")])
    with (output / "parameter_covariance_si.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["row_parameter / column_parameter", *labels])
        for label, row in zip(labels, covariance):
            writer.writerow([label, *[format(x, ".17g") for x in row]])
    return labels, value, covariance


def write_reports(output, data, fit, orientation, replay):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    case = str(fit["estimate"]["case_name"])
    fa, estimate = fit["arrays"], fit["estimate"]
    t = np.asarray(data["time"])
    relative = t - t[0]
    mt = np.asarray(data["mocap_time"])
    original_mocap = (mt >= t[0]) & (mt <= t[-1])
    mr = mt[original_mocap] - t[0]
    mocap_q = np.asarray(data["mocap_quaternion_xyzw"])
    mocap_rpy = np.rad2deg(np.unwrap(Rotation.from_quat(mocap_q[original_mocap]).as_euler("xyz"), axis=0))
    mocap_position = np.asarray(data["mocap_position"])[original_mocap]
    tracks = {"GA": (orientation["q_recursive_wxyz"], orientation["covariance_recursive"]),
              "GA + magnetic output": (orientation["q_output_wxyz"], orientation["covariance_output"]),
              "model + residual wrench": (replay["model"]["quaternion_wxyz"], replay["model"]["covariance_rotation"])}
    comparisons = {label: orientation_comparison(mt, mocap_q, t, q) for label, (q, _) in tracks.items()}
    distribution = estimate["parameter_distribution"]
    labels, value, covariance = _save_parameter_csv(output, distribution)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8,
                         "axes.labelsize": 10, "figure.titlesize": 14})
    pdfs = {name: PdfPages(output / name) for name in
            ("report.pdf", "report_body.pdf", "parameter_appendix.pdf", "residual_wrench.pdf",
             "sensor_reconstruction.pdf", "trajectory_replay.pdf")}
    pages = 0
    files = list(pdfs) + ["parameter_values_si.csv", "parameter_covariance_si.csv"]

    def save(fig, stem=None, companion=None, appendix=False):
        nonlocal pages
        fig.suptitle(case + " — " + fig._suptitle.get_text() if fig._suptitle else case)
        fig.tight_layout(rect=(0, .035, 1, .95))
        pdfs["report.pdf"].savefig(fig)
        pdfs["parameter_appendix.pdf" if appendix else "report_body.pdf"].savefig(fig)
        pages += 1
        if companion:
            pdfs[companion].savefig(fig)
        if stem:
            for extension in ("png", "svg"):
                name = stem + "." + extension
                fig.savefig(output / name, dpi=160)
                files.append(name)
        plt.close(fig)

    def axes_grid(rows, cols, title):
        fig, axs = plt.subplots(rows, cols, figsize=(11, 8.5), squeeze=False)
        fig.suptitle(title)
        for ax in axs.flat:
            ax.grid(alpha=.25)
            ax.set_xlabel("time since first native IMU sample [s]")
        return fig, axs

    try:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Native-IMU reconstruction: method and interpretation")
        text = [
            "Separate experiment: existing mocap identification and outputs are unchanged.",
            "Identification uses native gyro / specific force with fitted thrust, gyro and accel first-order lags.",
            "Orientation: gyro prediction + acceleration update recur; magnetic factors affect output only.",
            "Only initial orientation / position and initial velocity use mocap. Later mocap is read-only evaluation.",
            "Comparison uses original mocap timestamps inside IMU support; reconstructed attitudes are Slerped.",
            "Native timestamps and firmware first-or-changed magnetic freshness proxy are retained.",
            "Static sensor covariance = trace(unbiased static covariance)/3 times identity; no bias subtraction.",
            "Offline local polynomial derivatives and their overlap covariance are included in the lag fit.",
            "Uncertainty: Bingham-derived right-angle covariance and deterministic local Gaussian propagation.",
            "Shared acceleration/attitude information uses declared conservative covariance bounds.",
            "These algebraic bounds do not guarantee coverage of systematic acceleration or magnetic bias.",
            "Magnetic output can be worse on these data; no mocap-based gating, tuning or recursive feedback.",
            "Model reconstruction adds the native-IMU residual wrench time series to the physical dynamics.",
            "No Monte Carlo, artificial display variance, eigenvalue clipping or uncertainty-based sample selection.",
            "Mass / inertia / thrust scale use a common-scale gauge; chip lever arm is fixed, not calibrated.",
            "All 17 actual SI estimates and the complete joint covariance are appended at the end.",
            f"Native IMU support: {t[0]:.6f} to {t[-1]:.6f} bag seconds; N = {len(t)}.",
            "Individual PDF appendices and full-precision covariance CSV contain the same distribution."]
        fig.text(.055, .88, "\n\n".join(text), va="top", fontsize=10)
        save(fig)

        fig, axs = axes_grid(2, 2, "Rotor commands and fitted thrust dynamics")
        for k, ax in enumerate(axs.flat):
            ax.step(np.asarray(data["rotor_command_time"]) - t[0], np.asarray(data["rotor_command"])[:, k],
                    where="post", label="sent command", alpha=.65)
            ax.plot(relative, np.asarray(fa["actual_thrust"])[:, k], label="fitted first-order thrust")
            ax.set_title(f"rotor {k + 1}")
            ax.set_ylabel("thrust [N]")
            ax.legend()
        save(fig, "commands_and_thrust")

        fig, axs = axes_grid(3, 2, "Generalized IMU acceleration fit")
        for k in range(3):
            for col, observed, predicted, unit in (
                    (0, "corrected_specific_force", "predicted_specific_force", "m/s²"),
                    (1, "corrected_angular_acceleration", "predicted_angular_acceleration", "rad/s²")):
                ax = axs[k, col]
                ax.plot(relative, fa[observed][:, k], label="lag-corrected measurement")
                ax.plot(relative, fa[predicted][:, k], label="identified dynamics", alpha=.8)
                ax.set_ylabel("xyz"[k] + " [" + unit + "]")
                ax.legend()
        save(fig, "generalized_acceleration")

        residual = np.asarray(fa["raw_residual_wrench"])
        valid = np.asarray(fa["fit_valid_mask"], dtype=bool)
        fig, axs = axes_grid(3, 2, "Residual wrench at native measurement timestamps")
        for k, ax in enumerate(axs.flat):
            ax.plot(relative[valid], residual[valid, k])
            ax.set_ylabel(WRENCH_LABELS[k] + (" [N]" if k < 3 else " [N m]"))
        save(fig, "residual_wrench_history", "residual_wrench.pdf")
        frequency, power, peaks = _lomb(t[valid], residual[valid])
        fig, axs = axes_grid(3, 2, "Residual wrench spectrum (linear detrend; native irregular clock)")
        for k, ax in enumerate(axs.flat):
            ax.plot(frequency, power[:, k])
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel(WRENCH_LABELS[k] + " normalized power")
        save(fig, "residual_wrench_spectrum", "residual_wrench.pdf")

        applied_wrench, applied_factor, applied_std = applied_wrench_distribution(replay["model"], t)
        fig, axs = axes_grid(3, 2, "Applied residual wrench in reconstructed dynamics (body frame, about CoG)")
        for k, ax in enumerate(axs.flat):
            ax.fill_between(relative, applied_wrench[:, k] - 2 * applied_std[:, k],
                            applied_wrench[:, k] + 2 * applied_std[:, k],
                            color=COLORS["model + residual wrench"], alpha=.2, label="applied law ±2σ")
            ax.plot(relative, applied_wrench[:, k], color=COLORS["model + residual wrench"],
                    label="added residual-wrench mean")
            ax.set_ylabel(WRENCH_LABELS[k] + (" [N]" if k < 3 else " [N m]"))
            ax.legend()
        fig.text(.04, .012, "The native-IMU residual wrench is added to the dynamics; this is not a model-only free replay.", fontsize=9)
        save(fig, "applied_residual_wrench", "residual_wrench.pdf")

        # Model response is a measurement-state replay, not a new sensor fit.
        sensor_responses = {}
        for signal, tau_key, predicted, unit in (
                ("gyro", "gyro_time_constant_seconds", "predicted_gyro_body", "rad/s"),
                ("accel", "accel_time_constant_seconds", "predicted_specific_force_body", "m/s²")):
            physical = np.asarray(replay["model"][predicted])
            response = first_order_measurement_response(t, physical,
                float(estimate["time_constants_seconds"][tau_key]), np.asarray(data[signal])[0])
            sensor_responses[signal] = response
            fig, axs = axes_grid(3, 1, signal + ": raw, inverse-lag correction and model measurement response")
            for k, ax in enumerate(axs[:, 0]):
                ax.plot(relative, data[signal][:, k], label="raw native sensor", alpha=.5)
                ax.plot(relative, orientation["corrected_" + signal][:, k], label="inverse-lag corrected")
                ax.plot(relative, physical[:, k], label="model + residual wrench physical input", alpha=.8)
                ax.plot(relative, response[:, k], label="model first-order sensor response", linestyle="--")
                ax.set_ylabel("xyz"[k] + " [" + unit + "]")
                ax.legend()
            save(fig, signal + "_reconstruction", "sensor_reconstruction.pdf")
        np.savez_compressed(output / "sensor_measurement_response.npz", time=t,
                            gyro=sensor_responses["gyro"], accel=sensor_responses["accel"])
        files.append("sensor_measurement_response.npz")

        fig, axs = axes_grid(3, 1, "Attitude reconstruction against native mocap pose stream")
        for k, ax in enumerate(axs[:, 0]):
            ax.plot(mr, mocap_rpy[:, k], label="mocap", color="black", linewidth=1.7)
            for name, (q, _) in tracks.items():
                rpy = np.rad2deg(np.unwrap(rotations(q).as_euler("xyz"), axis=0))
                # Only add complete turns for a comparable Euler branch; law unchanged.
                rpy += 360 * np.round((mocap_rpy[0] - rpy[0]) / 360)
                ax.plot(relative, rpy[:, k], label=name, color=COLORS[name], alpha=.85)
            ax.set_ylabel(("roll", "pitch", "yaw")[k] + " [deg]")
            ax.legend()
        save(fig, "attitude_comparison", "trajectory_replay.pdf")
        fig, axs = axes_grid(1, 1, "SO(3) attitude errors at original mocap timestamps")
        ax = axs[0, 0]
        for name, comparison in comparisons.items():
            stats = comparison["statistics"]
            ax.plot(comparison["time"] - t[0], comparison["so3_error_deg"], color=COLORS[name],
                    label=f"{name}: RMS {stats['rms_deg']:.3f}°, final {stats['final_deg']:.3f}°")
        ax.set_ylabel("geodesic attitude error [deg]")
        ax.legend()
        save(fig, "attitude_error", "trajectory_replay.pdf")

        # Each track has its own native-law covariance; no truth-based rescaling.
        for name, (q, p) in tracks.items():
            fig, axs = axes_grid(3, 1, name + ": local attitude mean and ±2σ uncertainty")
            rpy = np.rad2deg(np.unwrap(rotations(q).as_euler("xyz"), axis=0))
            rpy += 360 * np.round((mocap_rpy[0] - rpy[0]) / 360)
            factor = replay["model"].get("rotation_covariance_factor") if name == "model + residual wrench" else None
            std = rpy_standard_deviation(q, p, factor)
            for k, ax in enumerate(axs[:, 0]):
                ax.fill_between(relative, rpy[:, k] - 2 * std[:, k], rpy[:, k] + 2 * std[:, k],
                                color=COLORS[name], alpha=.2, label="local Gaussian ±2σ")
                ax.plot(relative, rpy[:, k], color=COLORS[name], label=name)
                ax.plot(mr, mocap_rpy[:, k], color="black", label="mocap")
                ax.set_ylabel(("roll", "pitch", "yaw")[k] + " [deg]")
                ax.legend()
            save(fig, "attitude_uncertainty_" + ("model" if name == "model + residual wrench" else "ga" if name == "GA" else "magnetic"),
                 "trajectory_replay.pdf")

        def position_figure(uncertainty):
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Position reconstruction " + ("with full propagated ±2σ spread" if uncertainty else "(mean trajectories)"))
            axs = [fig.add_subplot(2, 2, k + 1) for k in range(3)]
            ax3 = fig.add_subplot(2, 2, 4, projection="3d")
            for k, ax in enumerate(axs):
                ax.plot(mr, mocap_position[:, k], color="black", label="mocap")
                ax.set_xlabel("time since first native IMU sample [s]")
                ax.set_ylabel("xyz"[k] + " FC position [m]")
                ax.grid(alpha=.25)
            ax3.plot(*mocap_position.T, color="black", label="mocap")
            for key, name in (("model", "model + residual wrench"), ("imu_dead_reckoning", "IMU dead reckoning")):
                branch, color = replay[key], COLORS[name]
                position, factor = branch["position_fc_world"], branch["position_covariance_factor_fc"]
                std = np.sqrt(np.sum(factor**2, axis=2))
                for k, ax in enumerate(axs):
                    ax.plot(relative, position[:, k], color=color, label=name)
                    if uncertainty:
                        ax.fill_between(relative, position[:, k] - 2 * std[:, k],
                                        position[:, k] + 2 * std[:, k], color=color, alpha=.18)
                ax3.plot(*position.T, color=color, label=name)
                if uncertainty:
                    for index in np.unique(np.linspace(0, len(t) - 1, 10).astype(int)):
                        for circle in covariance_wireframes(position[index], factor[index]):
                            ax3.plot(*circle, color=color, alpha=.25, linewidth=.5)
            for ax in axs + [ax3]:
                ax.legend()
            ax3.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
            fig.text(.04, .012, "FC origin; local Gaussian linearization. Ellipsoids: deterministic 2σ factor-SVD wireframes, not samples."
                     if uncertainty else "Residual-wrench model replay and IMU integration are not reset to later mocap.", fontsize=9)
            return fig
        save(position_figure(False), "position_reconstruction", "trajectory_replay.pdf")
        save(position_figure(True), "position_uncertainty", "trajectory_replay.pdf")

        appendix_start = pages + 1
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle("Actual estimated SI values and joint covariance coordinate order")
        ax.axis("off")
        rows = [[str(k + 1), label, _units(label), f"{value[k]:.9e}", f"{np.sqrt(covariance[k,k]):.9e}"]
                for k, label in enumerate(labels)]
        table = ax.table(cellText=rows, colLabels=["#", "parameter", "unit", "estimate", "std. deviation"],
                         colWidths=[.035, .39, .1, .22, .22], loc="upper center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1., 1.6)
        fig.text(.045, .085, "Common-scale gauge: " + distribution["common_scale_gauge"], fontsize=9)
        fig.text(.045, .058, "Covariance units are products of row/column units. All cross terms and weak modes are retained.", fontsize=9)
        fig.text(.045, .032, "Full-precision values: parameter_values_si.csv; full 17×17 covariance: parameter_covariance_si.csv.", fontsize=9)
        save(fig, appendix=True)
        for rs, cs in covariance_tiles(17):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.suptitle(f"Full SI covariance: rows {rs.start+1}–{rs.stop}, columns {cs.start+1}–{cs.stop}")
            ax.axis("off")
            table = ax.table(cellText=[[f"{x:.6e}" for x in row] for row in covariance[rs, cs]],
                             rowLabels=[str(i + 1) for i in range(rs.start, rs.stop)],
                             colLabels=[str(i + 1) for i in range(cs.start, cs.stop)],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1., 2.8)
            fig.text(.05, .085, "Indices refer to the preceding SI parameter table; Cij has units unit(i) × unit(j).", fontsize=10)
            fig.text(.05, .055, "Actual joint covariance, including all sensor-lag cross covariance; no eigenvalue clipping.", fontsize=10)
            save(fig, appendix=True)
    finally:
        for pdf in pdfs.values():
            pdf.close()

    statistics = {"attitude_comparison": {name: item["statistics"] for name, item in comparisons.items()},
                  "residual_wrench_rms": np.sqrt(np.mean(residual[valid]**2, axis=0)).tolist(),
                  "applied_residual_wrench_rms": np.sqrt(np.mean(applied_wrench**2, axis=0)).tolist(),
                  "applied_residual_wrench_clock": "same original IMU/replay timestamps; interpolation in dynamics uses this same mean and uncertainty law",
                  "residual_wrench_peak_frequency_hz": [None if not np.isfinite(x) else float(x) for x in peaks],
                  "mocap_comparison_clock": "original supported mocap stamps; Slerp reconstruction only; no extrapolation",
                  "parameter_count": 17, "covariance_entry_count": 289,
                  "covariance_approximation": "local Gaussian; conservative bounds for modeled shared terms, not systematic-bias coverage"}
    result = {"files": files, "statistics": statistics, "appendix_start_page": appendix_start,
              "report_page_count": pages, "body_page_count": appendix_start - 1,
              "parameter_appendix_page_count": pages - appendix_start + 1,
              "monte_carlo": False, "mocap_used_for_parameter_identification": False,
              "model_replay_includes_residual_wrench": True}
    (output / "residual_wrench_summary.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    files.append("residual_wrench_summary.json")
    return result
