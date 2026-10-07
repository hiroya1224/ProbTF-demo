"""Scenario validation, causal bag sampling, and independent evaluation.

Simulator states and load values are used only here for evaluation; the
controller continues to see its usual reference and IMU topics.
"""
import csv
import json
import math
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np


VECTOR_TOPICS = {
    "/deflecomp/theta_eq_hat": "prediction",
    "/deflecomp/kp_est": "kp_est",
    "/deflecomp/kp_exec": "kp_exec",
    "/deflecomp/kp_exec_target": "kp_target",
    "/deflecomp/kp_cov_diag": "kp_cov",
}
JOINT_TOPICS = {
    "/ref/joint_states": "reference",
    "/cmd/joint_states": "command",
    "/equil/joint_states": "actual",
}
EVENT_TOPIC = "/deflecomp_demo/event"
MARKER_TOPIC = "/deflecomp_demo/markers"


def validate_scenario(config, robot=None):
    names = config["joint_names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("joint_names must be nonempty and unique")
    if not config.get("load_frame"):
        raise ValueError("load_frame is required")
    for key in ("rate_hz", "analysis_rate_hz"):
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    criteria = config["criteria"]
    for key in ("prediction_joint_rms_deg", "stable_log_k_range",
                "execution_log_k_gap", "window_s", "max_sample_gap_s"):
        if not math.isfinite(float(criteria[key])) or float(criteria[key]) <= 0:
            raise ValueError(f"criteria.{key} must be positive and finite")
    if "prediction_tip_mm" in criteria:
        if not math.isfinite(float(criteria["prediction_tip_mm"])) or float(criteria["prediction_tip_mm"]) <= 0:
            raise ValueError("criteria.prediction_tip_mm must be positive and finite")
    phases = config["phases"]
    if not phases or len({phase["name"] for phase in phases}) != len(phases):
        raise ValueError("phases must be nonempty with unique names")
    for phase in phases:
        q = np.radians(np.asarray(phase["target_deg"], dtype=float))
        if q.shape != (len(names),) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid target in {phase['name']}")
        for key in ("move_s", "hold_s", "mass_kg"):
            if not math.isfinite(float(phase[key])) or float(phase[key]) < 0:
                raise ValueError(f"invalid {key} in {phase['name']}")
        if not isinstance(phase.get("check_convergence", True), bool):
            raise ValueError(f"check_convergence must be boolean in {phase['name']}")
        if phase.get("check_convergence", True) and float(phase["hold_s"]) < float(criteria["window_s"]):
            raise ValueError(f"hold too short for evaluation in {phase['name']}")
        if "hold_until" in phase:
            if phase["hold_until"] != "prediction_stable" or not phase.get("check_convergence", True):
                raise ValueError(f"invalid hold_until in {phase['name']}")
            minimum = float(phase.get("min_hold_s", criteria["window_s"]))
            if not math.isfinite(minimum) or minimum < float(criteria["window_s"]) or minimum > float(phase["hold_s"]):
                raise ValueError(f"invalid min_hold_s in {phase['name']}")
        if robot is not None:
            if list(robot.model_joint_names) != names:
                raise ValueError(f"scenario joints differ from model: {robot.model_joint_names}")
            if not robot.has_frame(config["load_frame"]):
                raise ValueError(f"unknown load frame: {config['load_frame']}")
            if np.any(q < robot.model.lowerPositionLimit) or np.any(q > robot.model.upperPositionLimit):
                raise ValueError(f"target outside URDF limits: {phase['name']}")
    # Load trials must hold the same reference as the preceding no-load trial.
    for before, after in zip(phases, phases[1:]):
        if after["mass_kg"] != before["mass_kg"]:
            if after["move_s"] != 0 or after["target_deg"] != before["target_deg"]:
                raise ValueError("a load change must keep the previous reference fixed")


def interpolate_reference(start, target, elapsed, duration):
    if duration <= 0:
        return np.asarray(target, dtype=float).copy()
    u = np.clip(float(elapsed) / float(duration), 0.0, 1.0)
    blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return np.asarray(start) + blend * (np.asarray(target) - np.asarray(start))


def parse_gate(text):
    return dict(part.split("=", 1) for part in text.split() if "=" in part)


def pose_error(predicted, actual):
    distance_mm = 1000.0 * float(np.linalg.norm(predicted.translation - actual.translation))
    rotation = predicted.rotation.T @ actual.rotation
    angle_deg = math.degrees(math.acos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    return distance_mm, angle_deg


def evaluate_snapshot(latest, robot, stamp, start_stamp, max_age=0.5):
    required = ["reference", "command", "actual", "prediction", "kp_est", "kp_exec", "kp_cov"]
    if any(key not in latest for key in required):
        return None
    ages = [float(stamp) - latest[key][0] for key in required]
    if "gate" in latest:
        ages.append(float(stamp) - latest["gate"][0])
    age = max(ages)
    if age > max_age or min(ages) < -1e-6:
        return None
    values = {key: np.asarray(latest[key][1], dtype=float) for key in required}
    if any(v.shape != (robot.nv,) or not np.all(np.isfinite(v)) for v in values.values()):
        return None
    if np.any(values["kp_est"] <= 0) or np.any(values["kp_exec"] <= 0):
        return None
    q_ref, q_cmd, q_actual, q_prediction = (values[key] for key in required[:4])
    tip_ref = robot.fk_pose(q_ref)
    tip_actual = robot.fk_pose(q_actual)
    tip_prediction = robot.fk_pose(q_prediction)
    pred_mm, pred_deg = pose_error(tip_prediction, tip_actual)
    track_mm, track_deg = pose_error(tip_ref, tip_actual)
    row = {
        "time_s": float(stamp - start_stamp),
        "stamp": float(stamp),
        "max_age_s": float(age),
        "prediction_joint_rms_deg": float(np.degrees(np.sqrt(np.mean((q_prediction - q_actual) ** 2)))),
        "tracking_joint_rms_deg": float(np.degrees(np.sqrt(np.mean((q_ref - q_actual) ** 2)))),
        "command_offset_rms_deg": float(np.degrees(np.sqrt(np.mean((q_cmd - q_ref) ** 2)))),
        "prediction_tip_mm": pred_mm,
        "prediction_tip_deg": pred_deg,
        "tracking_tip_mm": track_mm,
        "tracking_tip_deg": track_deg,
        "execution_log_k_gap": float(np.max(np.abs(np.log(values["kp_exec"] / values["kp_est"])))),
    }
    for key, vector in values.items():
        for index, value in enumerate(vector):
            row[f"{key}_{index}"] = float(value)
    gate = parse_gate(latest.get("gate", (0, ""))[1])
    row["gate_ready"] = int(gate.get("reason") == "ready")
    row["update_applied"] = int(gate.get("est_update_applied", "0"))
    row["laplace_dx_max_abs"] = float(gate.get("laplace_dx_max_abs", "0"))
    return row


def prediction_stable_window(rows, criteria):
    """Check sustained prediction accuracy independently of parameter drift."""
    if len(rows) < 2:
        return False
    times = np.asarray([row["time_s"] for row in rows])
    if times[-1] - times[0] < float(criteria["window_s"]) - 1e-9:
        return False
    if np.max(np.diff(times)) > float(criteria["max_sample_gap_s"]):
        return False
    if max(row["prediction_joint_rms_deg"] for row in rows) > criteria["prediction_joint_rms_deg"]:
        return False
    if "prediction_tip_mm" in criteria:
        if any(not math.isfinite(row.get("prediction_tip_mm", math.nan))
               or row["prediction_tip_mm"] > criteria["prediction_tip_mm"] for row in rows):
            return False
    return True


class PredictionConvergenceMonitor:
    """A fresh post-step window for the experiment's stopping rule only."""
    def __init__(self, criteria, start_time_s, min_hold_s, confirmation_s=0.0):
        self.criteria = dict(criteria, window_s=criteria["window_s"] + confirmation_s)
        self.start_time_s = start_time_s
        self.min_hold_s = min_hold_s
        self.rows = []

    def observe(self, row):
        if row is None or row["time_s"] < self.start_time_s:
            return
        if self.rows and row["time_s"] <= self.rows[-1]["time_s"]:
            return
        self.rows.append(row)
        left = row["time_s"] - self.criteria["window_s"]
        while len(self.rows) > 1 and self.rows[1]["time_s"] <= left:
            self.rows.pop(0)

    def converged(self, time_s):
        if not self.rows or time_s - self.start_time_s < self.min_hold_s:
            return False
        age = time_s - self.rows[-1]["time_s"]
        return (0 <= age <= self.criteria["max_sample_gap_s"]
                and prediction_stable_window(self.rows, self.criteria))


def stable_window(rows, criteria, joint_count):
    """Also require an open estimator gate and stable K_est/K_exec."""
    if not prediction_stable_window(rows, criteria):
        return False
    if not any(row["gate_ready"] for row in rows):
        return False
    if max(row["execution_log_k_gap"] for row in rows) > criteria["execution_log_k_gap"]:
        return False
    log_k = np.log([[row[f"kp_est_{i}"] for i in range(joint_count)] for row in rows])
    return bool(np.max(np.ptp(log_k, axis=0)) <= criteria["stable_log_k_range"])


def phase_summary(rows, events, criteria, joint_count, gates=None):
    holds = [event for event in events if event["kind"] == "hold"]
    summaries = []
    end = rows[-1]["time_s"] if rows else 0.0
    for event in holds:
        phase_end = next((e["time_s"] for e in events
                          if e["kind"] in ("phase", "finished") and e["time_s"] > event["time_s"]), end)
        samples = [row for row in rows if event["time_s"] <= row["time_s"] <= phase_end]
        sample_times = [row["time_s"] for row in samples]
        convergence, prediction_convergence = None, None
        def window_at(index):
            left = sample_times[index] - criteria["window_s"]
            # Include the preceding sample to cover the full requested duration.
            begin = max(0, bisect_right(sample_times, left) - 1)
            return samples[begin:index + 1]
        for index, sample in enumerate(samples):
            window = window_at(index)
            if prediction_convergence is None and prediction_stable_window(window, criteria):
                prediction_convergence = sample["time_s"] - event["time_s"]
            if convergence is None and stable_window(window, criteria, joint_count):
                convergence = sample["time_s"] - event["time_s"]
            if convergence is not None and prediction_convergence is not None:
                break
        tail = window_at(len(samples) - 1) if samples else []
        last = samples[-1] if samples else None
        tail_fresh = bool(last and phase_end - last["time_s"] <= criteria["max_sample_gap_s"])
        gate_samples = [row for row in (gates if gates is not None else samples)
                        if event["time_s"] <= row["time_s"] < phase_end]
        update_times = [row["time_s"] - event["time_s"] for row in gate_samples if row["update_applied"]]
        summary = {
            "phase": event["phase"], "mass_kg": event["mass_kg"],
            "hold_start_s": event["time_s"], "hold_end_s": phase_end,
            "samples": len(samples), "first_stable_window_s": convergence,
            "stable_at_end": tail_fresh and stable_window(tail, criteria, joint_count),
            "first_prediction_stable_window_s": prediction_convergence,
            "prediction_stable_at_end": tail_fresh and prediction_stable_window(tail, criteria),
            "updates": sum(row["update_applied"] for row in gate_samples),
            "first_update_s": min(update_times) if update_times else None,
            "gate_reason_counts": dict(Counter(row.get("gate_reason", "unknown") for row in gate_samples)),
            "skip_reason_counts": dict(Counter(row.get("skip_reason", "unknown") for row in gate_samples)),
            "peak_prediction_joint_rms_deg": max((row["prediction_joint_rms_deg"] for row in samples), default=None),
            "hold_termination": next((e["kind"] for e in events if e["phase"] == event["phase"]
                                      and e["kind"] in ("converged", "hold_timeout")), "duration"),
        }
        for key in ("prediction_joint_rms_deg", "prediction_tip_mm", "prediction_tip_deg",
                    "tracking_tip_mm", "tracking_tip_deg", "execution_log_k_gap"):
            summary[f"tail_median_{key}"] = float(np.median([row[key] for row in tail])) if tail else None
        if last:
            summary["final_kp_est"] = [last[f"kp_est_{i}"] for i in range(joint_count)]
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError("no valid samples to export")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_recording(directory, bag_path=None, output=None):
    import rosbag
    import yaml
    from deflecomp_core.robot.pinocchio_robot import RobotArm

    directory = Path(directory)
    output = Path(output) if output else directory
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["scenario"]
    robot = RobotArm(str(directory / "model.urdf"), tip_link=config["load_frame"])
    validate_scenario(config, robot)
    start_stamp = manifest["start_stamp"]
    latest, rows, events, gates = {}, [], [], []
    next_sample = start_stamp
    count, invalid = 0, 0
    topics = list(VECTOR_TOPICS) + list(JOINT_TOPICS) + [EVENT_TOPIC, "/deflecomp/estimation_gate_status"]
    with rosbag.Bag(str(bag_path or directory / "experiment.bag")) as bag:
        topic_info = bag.get_type_and_topic_info().topics
        missing = [topic for topic in topics + ["/imu", "/tf", "/tf_static", "/deflecomp_sim/external_wrench"]
                   if topic not in topic_info]
        if missing:
            raise ValueError(f"missing required bag topics: {missing}")
        bag_duration = bag.get_end_time() - bag.get_start_time()
        recorded_topics = {topic: {"messages": info.message_count,
                                   "mean_hz_over_bag": info.message_count / max(bag_duration, 1e-9)}
                           for topic, info in topic_info.items()}
        for topic, msg, t in bag.read_messages(topics=topics):
            stamp = t.to_sec()
            if topic == EVENT_TOPIC:
                events.append(json.loads(msg.data))
            elif topic in JOINT_TOPICS:
                mapping = dict(zip(msg.name, msg.position))
                if all(name in mapping for name in robot.model_joint_names):
                    latest[JOINT_TOPICS[topic]] = (stamp, [mapping[name] for name in robot.model_joint_names])
            elif topic in VECTOR_TOPICS:
                latest[VECTOR_TOPICS[topic]] = (stamp, msg.data)
            else:
                latest["gate"] = (stamp, msg.data)
                gate = parse_gate(msg.data)
                gates.append({"time_s": stamp - start_stamp,
                              "gate_reason": gate.get("reason", "unknown"),
                              "update_applied": int(gate.get("est_update_applied", "0")),
                              "skip_reason": gate.get("est_update_skipped_reason", "unknown"),
                              "laplace_dx_max_abs": float(gate.get("laplace_dx_max_abs", "0"))})
            # Evaluate at prediction receipt time using only already-recorded
            # samples. No interpolation across movement or load discontinuities.
            if topic == "/deflecomp/theta_eq_hat" and stamp >= next_sample:
                count += 1
                next_sample = stamp + 1.0 / config["analysis_rate_hz"]
                row = evaluate_snapshot(latest, robot, stamp, start_stamp,
                                        config["criteria"]["max_sample_gap_s"])
                if row is None:
                    invalid += 1
                else:
                    rows.append(row)
    if not events or not rows:
        raise ValueError("bag has no scenario events or valid evaluation samples")
    write_csv(output / "samples.csv", rows)
    write_csv(output / "estimation_gates.csv", gates)
    (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    summaries = phase_summary(rows, events, config["criteria"], robot.nv, gates)
    phases_by_name = {phase["name"]: phase for phase in config["phases"]}
    for summary in summaries:
        summary["convergence_checked"] = phases_by_name[summary["phase"]].get("check_convergence", True)
    checked = [summary for summary in summaries if summary["convergence_checked"]]
    expected = [phase["name"] for phase in config["phases"]]
    completed = manifest.get("completed", False) and [p["phase"] for p in summaries] == expected
    report = {
        "completed": completed, "all_holds_stable_at_end": completed and bool(checked) and all(p["stable_at_end"] for p in checked),
        "all_predictions_stable_at_end": completed and bool(checked) and all(p["prediction_stable_at_end"] for p in checked),
        "criteria": config["criteria"], "valid_samples": len(rows),
        "discarded_stale_or_incomplete_samples": invalid,
        "sample_attempts": count, "phases": summaries,
        "recorded_topics": recorded_topics,
        "interpretation": [
            "K_est is a pose- and load-dependent effective stiffness, not material identification.",
            "Full tip/joint errors are evaluated against simulator truth, never supplied to the estimator.",
            "Prediction convergence and K_est/K_exec stability are reported separately under the recorded criteria.",
            "Brief holds marked check_convergence=false are observations, excluded from the overall convergence result.",
            "Time is ROS receipt/wall time; Python timer integration is not a deterministic physical clock.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    parameters = yaml.safe_load((directory / "parameters.yaml").read_text())
    kp_true = parameters.get("deflecomp_sim", {}).get("kp_true", [])
    plot_recording(output, rows, events, config, kp_true)
    if (directory / "rviz.mp4").is_file():
        import shutil
        import subprocess
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "2",
                            "-i", str(directory / "rviz.mp4"), "-frames:v", "1", str(output / "preview.png")],
                           check=True, timeout=15, stdout=subprocess.DEVNULL)
    return report


def plot_recording(output, rows, events, config, kp_true):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([row["time_s"] for row in rows])
    n = len(config["joint_names"])
    def values(key):
        return np.asarray([row[key] for row in rows])
    def mark(ax):
        for event in events:
            if event["kind"] == "phase":
                ax.axvline(event["time_s"], color="0.6", lw=0.7, alpha=0.6)
        load = next((event["time_s"] for event in events if event["kind"] == "phase" and event["mass_kg"] > 0), None)
        unload = next((event["time_s"] for event in events if load is not None and event["kind"] == "phase" and event["time_s"] > load and event["mass_kg"] == 0), None)
        if load is not None:
            ax.axvspan(load, unload if unload is not None else t[-1], color="#e69f00", alpha=0.12)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, t[-1])
    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=160)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
    fig.suptitle("Tip load experiment: prediction, tracking, and estimation")
    axes[0].plot(t, values("prediction_tip_mm"), label="Predicted tip vs plant", color="#0072b2")
    axes[0].plot(t, values("tracking_tip_mm"), label="Reference tip vs plant", color="#cc79a7")
    axes[0].set_ylabel("Tip position error [mm]")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, values("prediction_joint_rms_deg"), label="Prediction vs plant")
    axes[1].plot(t, values("tracking_joint_rms_deg"), label="Reference vs plant", alpha=0.7)
    axes[1].axhline(config["criteria"]["prediction_joint_rms_deg"], color="k", ls=":", label="Prediction criterion")
    axes[1].set_ylabel("Joint RMS error [deg]")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, values("prediction_tip_deg"), label="Predicted orientation vs plant")
    axes[2].plot(t, values("tracking_tip_deg"), label="Reference orientation vs plant", alpha=0.7)
    axes[2].set_ylabel("Tip orientation error [deg]")
    axes[2].legend(loc="upper right")
    axes[3].plot(t, values("execution_log_k_gap"), label="max |log(K_exec / K_est)|")
    axes[3].plot(t, values("laplace_dx_max_abs"), label="Estimator batch log-K step", alpha=0.7)
    axes[3].set_ylabel("Log-K difference")
    axes[3].legend(loc="upper right")
    phase_events = [event for event in events if event["kind"] == "phase"]
    et = [event["time_s"] for event in phase_events] + [t[-1]]
    masses = [event["mass_kg"] for event in phase_events]
    axes[4].step(et, masses + [masses[-1]], where="post", label="Tip load [kg equivalent]", color="#e69f00")
    axes[4].plot(t, values("gate_ready"), label="Timing gate ready", alpha=0.4, lw=0.8)
    axes[4].set_ylabel("Load / gate")
    axes[4].set_xlabel("Time from scenario start [s]")
    axes[4].legend(loc="upper right")
    for ax in axes:
        mark(ax)
    for event in phase_events:
        axes[0].text(event["time_s"] + 0.3, 0.98, event["phase"], rotation=90,
                     va="top", fontsize=7, transform=axes[0].get_xaxis_transform())
    save(fig, "overview")

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.3 * n), sharex=True, constrained_layout=True)
    for index, ax in enumerate(np.atleast_1d(axes)):
        estimate = values(f"kp_est_{index}")
        std = np.sqrt(np.maximum(values(f"kp_cov_{index}"), 0))
        ax.fill_between(t, estimate * np.exp(-2 * std), estimate * np.exp(2 * std),
                        color="#0072b2", alpha=0.15, label="K_est log-space +/-2 sigma")
        ax.plot(t, estimate, label="K_est", color="#0072b2")
        ax.plot(t, values(f"kp_exec_{index}"), label="K_exec", color="#d55e00")
        if len(kp_true) == n:
            ax.axhline(kp_true[index], ls=":", color="black", label="K_true (evaluation only)")
        ax.set_yscale("log")
        ax.set_ylabel(config["joint_names"][index] + "\nK [N m/rad]")
        mark(ax)
        if index == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time [s]; shaded interval = applied load")
    save(fig, "stiffness")

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.3 * n), sharex=True, constrained_layout=True)
    for index, ax in enumerate(np.atleast_1d(axes)):
        for key, label, color, style in (("reference", "Reference", "#009e73", "-"),
                ("actual", "Plant", "#cc79a7", "-"),
                ("prediction", "Predicted equilibrium", "#0072b2", "--"),
                ("command", "Spring setpoint", "#d55e00", ":")):
            ax.plot(t, np.degrees(values(f"{key}_{index}")), label=label, color=color, ls=style, lw=1)
        ax.set_ylabel(config["joint_names"][index] + "\nAngle [deg]")
        mark(ax)
        if index == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time [s]; shaded interval = applied load")
    save(fig, "joint_states")
