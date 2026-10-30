"""Native IMU/issued-command input, with mocap confined to initial conditions.

The rosbag import is isolated in ``load_case``. Measurement, record and source
clocks are retained separately. No on-board fused quaternion or ongoing mocap
attitude is admitted to the IMU observation arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
CASE_JSONS = {
    "failure1": "single_rosbag_1.json", "failure2": "single_rosbag_2.json",
    "success": "single_rosbag_succeeded.json",
}
TOPICS = {
    "imu": "/gimbalrotor/imu", "converted": "/gimbalrotor/sensor_plugin/imu1/ros_converted",
    "mocap": "/gimbalrotor/mocap/pose", "state": "/gimbalrotor/flight_state",
    "rotor": "/gimbalrotor/four_axes/command", "joint": "/gimbalrotor/joint_states",
    "gimbal_command": "/gimbalrotor/gimbals_ctrl", "pid": "/gimbalrotor/debug/pose/pid",
    "static": "/tf_static",
}
SENSOR_POSITION = np.array([-.0172999968682441, -.00110000084294132, .05706099896])
JOINTS = ("gimbal1", "gimbal2", "gimbal3", "gimbal4")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isotropic_noise(samples):
    """Unbiased sample covariance -> trace(S)/3 I, never covariance of mean."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("noise calibration requires at least two finite 3-vectors")
    covariance = np.cov(values, rowvar=False, ddof=1)
    return np.eye(3) * float(np.trace(covariance) / 3), covariance


def changed_magnetometer_mask(values):
    """Firmware-compatible value-change proxy; no physical fresh stamp exists."""
    values = np.asarray(values, dtype=float)
    return np.r_[True, np.any(values[1:] != values[:-1], axis=1)]


def initial_conditions(time, position, quaternion_xyzw, source_index, initial_time):
    """One q/p bracket at t0 and a strictly-past 1 s initial velocity fit."""
    time, position, quaternion = map(np.asarray, (time, position, quaternion_xyzw))
    t0 = float(initial_time)
    right = int(np.searchsorted(time, t0, side="left"))
    if right == len(time) or (right == 0 and time[right] != t0):
        raise ValueError("initial IMU stamp is outside mocap support")
    left = right if time[right] == t0 else right - 1
    indices = np.array([left, right], dtype=int)
    if left == right:
        q0, p0 = quaternion[left], position[left]
    else:
        fraction = (t0 - time[left]) / (time[right] - time[left])
        p0 = (1 - fraction) * position[left] + fraction * position[right]
        q0 = Slerp(time[indices], Rotation.from_quat(quaternion[indices]))([t0]).as_quat()[0]
    past = (time < t0) & (time >= t0 - 1.)
    if np.count_nonzero(past) < 3:
        raise ValueError("initial velocity requires three strictly-past mocap samples")
    dt = time[past] - t0
    design = np.column_stack([np.ones(len(dt)), dt])
    coefficients = np.linalg.lstsq(design, position[past], rcond=None)[0]
    error = position[past] - design @ coefficients
    residual_covariance = error.T @ error / (len(dt) - 2)
    slope_variance_scale = np.linalg.solve(design.T @ design, np.eye(2))[1, 1]
    return {
        "initial_quaternion_wxyz": np.asarray(q0)[[3, 0, 1, 2]],
        "initial_position": np.asarray(p0).copy(), "initial_velocity": coefficients[1],
        "initial_velocity_covariance": residual_covariance * slope_variance_scale,
        "initial_mocap_bracket_source_indices": np.asarray(source_index)[indices],
        "initial_velocity_source_indices": np.asarray(source_index)[past],
        "initial_mocap_bracket_time": time[indices].copy(),
        "initial_velocity_source_time": time[past].copy(),
    }


def _strict(rows, name):
    stamps = np.array([row["stamp_ns"] for row in rows], dtype=np.int64)
    if len(stamps) == 0 or np.any(np.diff(stamps) <= 0):
        raise ValueError(f"{name} must retain strictly increasing source stamps")
    return stamps


def _read_messages(bag):
    """Raw fields only; retain original per-topic index and absolute ns clocks."""
    by_topic = {topic: name for name, topic in TOPICS.items()}
    streams = {name: [] for name in TOPICS}
    for selected_index, (topic, message, record) in enumerate(bag.read_messages(topics=list(by_topic))):
        name = by_topic[topic]
        stamp = message.stamp if name == "imu" else message.header.stamp if name in ("converted", "mocap", "joint") else record
        row = {"stamp_ns": stamp.to_nsec(), "record_ns": record.to_nsec(),
               "source_index": len(streams[name]), "selected_record_index": selected_index}
        if name == "imu":
            row.update(gyro=list(message.gyro), accel=list(message.acc), mag=list(message.mag))
        elif name == "converted":
            row["frame"] = message.header.frame_id
        elif name == "mocap":
            p, q = message.pose.position, message.pose.orientation
            row.update(position=[p.x, p.y, p.z], quaternion=[q.x, q.y, q.z, q.w], frame=message.header.frame_id)
        elif name == "state":
            row["state"] = int(message.data)
        elif name == "rotor":
            row["value"] = list(message.base_thrust)
        elif name == "joint":
            lookup = dict(zip(message.name, message.position))
            row["value"] = [float(lookup[key]) for key in JOINTS]
        elif name == "gimbal_command":
            row["value"] = list(message.position)
        elif name == "static":
            row["transforms"] = []
            for tf in message.transforms:
                p, q = tf.transform.translation, tf.transform.rotation
                row["transforms"].append((tf.header.frame_id, tf.child_frame_id,
                                           [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
        streams[name].append(row)
    return streams


def _verify_extrinsics(streams):
    frames = sorted(set(row["frame"] for row in streams["converted"]))
    if not frames or any(frame.split("/")[-1] != "fc" for frame in frames):
        raise ValueError("converted IMU is not expressed in the audited FC frame")
    matches = [tf for row in streams["static"] for tf in row["transforms"]
               if tf[0].split("/")[-1] == "main_body" and tf[1].split("/")[-1] == "fc"]
    if not matches:
        raise ValueError("main_body to FC static transform is absent")
    for _, _, translation, quaternion in matches:
        if not np.allclose(translation, SENSOR_POSITION, rtol=0, atol=1e-12):
            raise ValueError("FC translation disagrees with audited robot geometry")
        if not np.allclose(Rotation.from_quat(quaternion).as_matrix(), np.eye(3), rtol=0, atol=1e-12):
            raise ValueError("FC axes are not aligned with body axes")
    return {"body_frame": matches[0][0], "imu_frame": frames[0],
            "source": "bag /tf_static main_body->FC, identity rotation; native fields copied directly to converted IMU",
            "physical_imu_lever_arm_calibrated": False,
            "sensor_position_model": "nominal FC origin; physical accelerometer-chip offset is not independently calibrated"}


def build_case_data(case, config, streams, bag_origin_ns, bag_metadata):
    """Pure construction seam: all noninitial mocap remains evaluation-only."""
    origin = int(bag_origin_ns)
    local = lambda rows: (_strict(rows, "stream") - origin).astype(float) * 1e-9
    imu = streams["imu"]
    all_time = local(imu)
    mask = (all_time >= float(config["start_seconds"])) & (all_time <= float(config["end_seconds"]))
    selected = [row for row, keep in zip(imu, mask) if keep]
    if not selected:
        raise ValueError("requested interval has no original IMU samples")
    values = {key: np.asarray([row[key] for row in imu], dtype=float) for key in ("gyro", "accel", "mag")}
    if any(value.shape != (len(imu), 3) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("IMU observations must be finite native 3-vectors")
    mag_observed_all = changed_magnetometer_mask(values["mag"])
    states = streams["state"]
    if not states or states[0]["state"] != 0:
        raise ValueError("recorded preflight must begin in ARM_OFF")
    end_arm_off = next(row["record_ns"] for row in states if row["state"] != 0)
    imu_stamp = np.asarray([row["stamp_ns"] for row in imu], dtype=np.int64)
    static = (imu_stamp >= states[0]["record_ns"]) & (imu_stamp < end_arm_off)
    static_mag = static & mag_observed_all
    covariances, raw_covariances = {}, {}
    for key in values:
        use = static_mag if key == "mag" else static
        covariances[key], raw_covariances[key] = isotropic_noise(values[key][use])
    pose = streams["mocap"]
    if any(row["frame"].split("/")[-1] != "world" for row in pose):
        raise ValueError("mocap initial/evaluation poses are not in the world frame")
    pose_time = local(pose)
    pose_position = np.asarray([row["position"] for row in pose], dtype=float)
    pose_quaternion = np.asarray([row["quaternion"] for row in pose], dtype=float)
    pose_index = np.asarray([row["source_index"] for row in pose], dtype=np.int64)
    time = all_time[mask]
    initial = initial_conditions(pose_time, pose_position, pose_quaternion, pose_index, time[0])
    # Separate evaluation mask. These arrays never participate in calibration,
    # gravity direction updates, gyro propagation or the subsequent plant fit.
    evaluation = (pose_time >= float(config["start_seconds"])) & (pose_time <= float(config["end_seconds"]))
    result = {"time": time, **{key: value[mask].copy() for key, value in values.items()},
              "mag_observed": mag_observed_all[mask],
              "imu_stamp_abs": imu_stamp[mask].astype(float) * 1e-9,
              "imu_stamp_ns": imu_stamp[mask],
              "imu_bag_time_abs": np.asarray([row["record_ns"] for row in selected], dtype=float) * 1e-9,
              "imu_bag_time_ns": np.asarray([row["record_ns"] for row in selected], dtype=np.int64),
              "imu_source_index": np.asarray([row["source_index"] for row in selected], dtype=np.int64),
              "imu_selected_record_index": np.asarray([row["selected_record_index"] for row in selected], dtype=np.int64),
              "calibration_time": all_time[static],
              "calibration_gyro": values["gyro"][static], "calibration_accel": values["accel"][static],
              "calibration_mag": values["mag"][static], "calibration_mag_observed": mag_observed_all[static],
              "gyro_noise_covariance": covariances["gyro"], "accel_noise_covariance": covariances["accel"],
              "mag_noise_covariance": covariances["mag"],
              "mocap_time": pose_time[evaluation], "mocap_position": pose_position[evaluation],
              "mocap_quaternion_xyzw": pose_quaternion[evaluation], "mocap_source_index": pose_index[evaluation],
              "gyro_sensor_position_in_body": SENSOR_POSITION.copy(),
              "body_to_gyro_sensor_rotation": np.eye(3), **initial}
    # Include every causal pre-window command. Actual-joint support includes
    # one post-window sample for interpolation, never invented held observations.
    command_metadata = {}
    for stream, key, value_key in (("rotor", "rotor_command", "rotor_command"),
                                   ("gimbal_command", "gimbal_command", "gimbal_command"),
                                   ("joint", "gimbal", "gimbal_angle")):
        rows = streams[stream]
        source_time = local(rows)
        stop = int(np.searchsorted(source_time, time[-1], side="right"))
        if stream == "joint":
            stop = min(stop + 1, len(source_time))
        take = slice(0, stop)
        if stop == 0 or source_time[0] > time[0]:
            raise ValueError(f"{stream} has no causal initial support")
        result[key + "_time"] = source_time[take]
        result[value_key] = np.asarray([row["value"] for row in rows], dtype=float)[take]
        result[key + "_source_index"] = np.asarray([row["source_index"] for row in rows], dtype=np.int64)[take]
        result[key + "_bag_time_ns"] = np.asarray([row["record_ns"] for row in rows], dtype=np.int64)[take]
        command_metadata[stream] = {"count": stop, "pre_window_count": int(np.count_nonzero(source_time[take] < time[0])),
                                    "clock": "header" if stream == "joint" else "bag record"}
    pid_time = local(streams["pid"])
    pid_window = pid_time[(pid_time >= config["start_seconds"]) & (pid_time <= config["end_seconds"])]
    result["controller_dt"] = float(np.median(np.diff(pid_window)))
    first_mag = np.flatnonzero(np.linalg.norm(result["mag"], axis=1) > 0)[0]
    # This is a declared initial world-reference construction, not ongoing mocap.
    result["initial_magnetic_reference_world"] = Rotation.from_quat(initial["initial_quaternion_wxyz"][[1, 2, 3, 0]]).apply(result["mag"][first_mag])
    result["metadata"] = {"schema": "grape-param-estim/imu-only-native-input/v1", "case": case,
        **bag_metadata, "requested_window": [config["start_seconds"], config["end_seconds"]],
        "time_origin_ns": origin, "measurement_clock": "native spinal/Imu.stamp",
        "units": {"time": "s", "gyro": "rad/s", "accel": "m/s^2 specific force",
                  "mag": "microtesla", "rotor_command": "N issued base_thrust", "gimbal_angle": "rad"},
        "state_frames": {"initial_quaternion_wxyz": "FC/body axes to world; axes aligned by verified static TF",
                         "initial_position": "FC sensor origin in world, not CoG/body origin",
                         "initial_velocity": "FC sensor-origin velocity in world, not CoG/body-origin velocity",
                         "mocap_position": "FC sensor origin in world; evaluation only",
                         "sensor_position_in_body": SENSOR_POSITION.tolist()},
        "source_index": "original per-topic message index; selected_record_index is within the explicitly selected topic set",
        "extrinsics": _verify_extrinsics(streams), "command_history": command_metadata,
        "initial_alignment": {"time": float(time[0]), "mocap_bracket_time": initial["initial_mocap_bracket_time"].tolist(),
                              "mocap_bracket_source_indices": initial["initial_mocap_bracket_source_indices"].tolist(),
                              "velocity_regression": "strictly past 1-second mocap prefix; initial condition only",
                              "velocity_prefix_count": len(initial["initial_velocity_source_indices"]),
                              "mocap_use_after_initialization": "evaluation only"},
        "calibration": {"interval": "initial contiguous ARM_OFF, header-selected within record-time state interval",
                        "interval_local": [(states[0]["record_ns"] - origin) * 1e-9, (end_arm_off - origin) * 1e-9],
                        "imu_count": int(static.sum()), "mag_change_count": int(static_mag.sum()),
                        "covariance": "trace(unbiased sample covariance)/3 * I; no division by N; no bias subtraction",
                        "raw_covariances": {key: value.tolist() for key, value in raw_covariances.items()},
                        "means_not_subtracted": {key: values[key][static_mag if key == "mag" else static].mean(0).tolist() for key in values}},
        "magnetometer": {"available": True, "units": "microtesla (native firmware driver convention)",
                         "observation_mask": "first full-stream value or vector change; firmware-compatible proxy, not an independent physical timestamp",
                         "initial_world_reference_source_index": int(result["imu_source_index"][first_mag]),
                         "onboard_fused_quaternion_used": False},
        "observation_resampling": False}
    return result


def load_case(case):
    import rosbag  # Only the I/O boundary needs ROS.
    source = HERE.parent / "bag_jsons" / CASE_JSONS[case]
    config = json.loads(source.read_text())
    path = Path(config["bag_path"])
    with rosbag.Bag(str(path), "r") as bag:
        origin = next(bag.read_messages(raw=True))[2].to_nsec()
        streams = _read_messages(bag)
        metadata = {"bag_path": str(path), "bag_size_bytes": path.stat().st_size,
                    "bag_sha256": sha256(path), "case_json": str(source), "case_json_sha256": sha256(source)}
    return build_case_data(case, config, streams, origin, metadata)


def save_case(case, output):
    result = load_case(case)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{case}.npz", **{key: value for key, value in result.items() if key != "metadata"})
    (output / f"{case}.json").write_text(json.dumps(result["metadata"], indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_JSONS))
    parser.add_argument("--output", type=Path, default=Path("/tmp/imu-first-order-lag-inputs"))
    args = parser.parse_args()
    for case in (tuple(CASE_JSONS) if args.case is None else (args.case,)):
        data = save_case(case, args.output)
        print(json.dumps({"case": case, "samples": len(data["time"]), "mag_observed": int(data["mag_observed"].sum()), "output": str(args.output / f"{case}.npz")}), flush=True)
