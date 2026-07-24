"""Reproducible real-bag vertical slice for Grape counterfactual diagnosis.

This adapter deliberately stops before recommendation when the deployed
PC/MCU replay oracle, its bag-derived conformance fixture, or held-out
probability calibration is unavailable.  It still produces the common
desired/nominal/actual trajectory view and a low-dimensional effective
response posterior from immutable bag evidence.
"""

from dataclasses import asdict, dataclass
import csv
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from .effective_response import (
    EffectiveResponseFitConfig,
    EffectiveResponsePosterior,
    LowDimensionalEffectiveResponse,
    TrajectoryTransitionBatch,
    fit_effective_response,
)
from .episode import message_event_time, sha256_file, stable_hash
from .state_smoother import (
    SmootherConfig,
    TrajectoryObservations,
    TrajectoryPosterior,
    smooth_trajectory,
)


SCHEMA = "grape_real_bag_vertical_slice/v1"
WORKFLOW_STATUS = "EXPERIMENTAL"
AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NOT_FINITE"
        return "UNBOUNDED" if value > 0.0 else "NEGATIVE_UNBOUNDED"
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _source_commit(repository: Optional[Any] = None) -> str:
    root = (
        Path(repository).resolve()
        if repository is not None
        else Path(__file__).resolve().parents[5]
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit + ("+dirty" if status else "")
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def load_vertical_slice_config(path: Any) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping) or config.get("schema") != SCHEMA:
        raise ValueError("unsupported vertical-slice config schema")
    required_topics = {
        "mocap_pose",
        "imu",
        "controller_pid",
        "four_axis_command",
        "roll_pitch_gain",
        "flight_state",
    }
    topics = config.get("topics", {})
    if set(topics) != required_topics or any(
        not str(value).startswith("/") for value in topics.values()
    ):
        raise ValueError("vertical-slice topic map is incomplete")
    episodes = tuple(config.get("episodes", ()))
    identifiers = [str(item.get("episode_id", "")) for item in episodes]
    if (
        len(episodes) != 3
        or len(set(identifiers)) != 3
        or set(identifiers)
        != {"20260612-04", "20260612-07", "20260612-08"}
    ):
        raise ValueError("vertical slice must declare bags 4, 7, and 8")
    for item in episodes:
        digest = str(item.get("source_bag_sha256", ""))
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or float(item.get("start_offset_s", -1.0)) < 0.0
            or float(item.get("duration_s", 0.0)) <= 0.0
        ):
            raise ValueError("vertical-slice episode provenance is invalid")
    if config.get("exact_controller", {}).get("status") != "ORACLE_UNAVAILABLE":
        raise ValueError(
            "repository config cannot claim an unavailable exact oracle"
        )
    conventions = config.get("conventions", {})
    expected_frames = conventions.get("expected_header_frames", {})
    if (
        conventions.get("world_frame") != "ENU"
        or conventions.get("body_frame") != "FLU"
        or conventions.get("quaternion_order") != "xyzw"
        or set(expected_frames)
        != {"mocap_pose", "imu", "controller_pid"}
        or not conventions.get("units")
    ):
        raise ValueError("vertical-slice frame/unit conventions are incomplete")
    output = dict(config)
    output["episodes"] = [dict(item) for item in episodes]
    output["config_sha256"] = stable_hash(config)
    return output


def _event_seconds(message: Any, record_time: Any) -> float:
    event, _ = message_event_time(message)
    return float(record_time.to_sec()) if event is None else float(event)


def _pid_scalar(axis: Any, field: str) -> float:
    value = getattr(axis, field)
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("empty PID field")
        value = value[0]
    return float(value)


def _pid_vectors(message: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [getattr(message, name) for name in AXES]
    desired_position = np.asarray(
        [_pid_scalar(item, "target_p") for item in values], dtype=float
    )
    desired_velocity = np.asarray(
        [_pid_scalar(item, "target_d") for item in values], dtype=float
    )
    nominal_acceleration = np.asarray(
        [_pid_scalar(item, "total") for item in values], dtype=float
    )
    if not np.all(
        np.isfinite(
            np.concatenate(
                (desired_position, desired_velocity, nominal_acceleration)
            )
        )
    ):
        raise ValueError("PID vectors must be finite")
    return desired_position, desired_velocity, nominal_acceleration


def _deduplicate(rows: Sequence[Tuple[float, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty(0), np.empty((0, 0))
    ordered = sorted(rows, key=lambda item: item[0])
    unique: Dict[float, np.ndarray] = {}
    for stamp, value in ordered:
        unique[float(stamp)] = np.asarray(value, dtype=float)
    times = np.asarray(sorted(unique), dtype=float)
    values = np.asarray([unique[stamp] for stamp in times], dtype=float)
    return times, values


@dataclass(frozen=True)
class BagIntervalData:
    episode_id: str
    stratum: str
    bag_path: str
    source_bag_sha256: str
    bag_start_time: float
    interval_start_offset_s: float
    interval_end_offset_s: float
    mocap_times: np.ndarray
    mocap_positions: np.ndarray
    mocap_quaternions: np.ndarray
    imu_times: np.ndarray
    accelerometer: np.ndarray
    gyro: np.ndarray
    pid_times: np.ndarray
    desired_position_euler: np.ndarray
    desired_velocity: np.ndarray
    nominal_acceleration: np.ndarray
    four_axis_thrust: np.ndarray
    roll_pitch_gain: Mapping[str, float]
    flight_states: Tuple[int, ...]
    topic_counts: Mapping[str, int]
    header_record_offset_median_s: Mapping[str, Optional[float]]
    observed_header_frames: Mapping[str, Tuple[str, ...]]


def read_bag_interval(
    bag_path: Any,
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
) -> BagIntervalData:
    """Read only the declared real-bag slice without modifying the bag."""

    try:
        import genpy
        import rosbag
    except ImportError as exc:  # pragma: no cover - ROS installation boundary
        raise RuntimeError("real-bag adapter requires ROS 1 rosbag/genpy") from exc

    path = Path(bag_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    expected_hash = str(episode["source_bag_sha256"])
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError("source bag SHA-256 does not match frozen config")
    topics = dict(config["topics"])
    selected = tuple(topics.values())
    buffers: Dict[str, list] = {
        name: []
        for name in (
            "mocap",
            "imu",
            "pid",
            "four_axis",
            "flight_state",
        )
    }
    offsets: Dict[str, list] = {name: [] for name in ("mocap", "imu", "pid")}
    counts = {value: 0 for value in selected}
    observed_frames: Dict[str, set] = {
        "mocap_pose": set(),
        "imu": set(),
        "controller_pid": set(),
    }
    gain = {}
    with rosbag.Bag(str(path), "r") as bag:
        bag_start = float(bag.get_start_time())
        start = bag_start + float(episode["start_offset_s"])
        end = start + float(episode["duration_s"])
        read_start = genpy.Time.from_sec(max(bag_start, start - 1.0))
        read_end = genpy.Time.from_sec(end + 1.0)
        for topic, message, record_time in bag.read_messages(
            topics=selected,
            start_time=read_start,
            end_time=read_end,
        ):
            counts[topic] += 1
            event = _event_seconds(message, record_time)
            relative = event - bag_start
            if topic == topics["roll_pitch_gain"]:
                continue
            if not float(episode["start_offset_s"]) <= relative <= (
                float(episode["start_offset_s"]) + float(episode["duration_s"])
            ):
                continue
            record_seconds = float(record_time.to_sec())
            if topic == topics["mocap_pose"]:
                observed_frames["mocap_pose"].add(
                    str(message.header.frame_id)
                )
                position = message.pose.position
                orientation = message.pose.orientation
                buffers["mocap"].append(
                    (
                        relative,
                        np.array(
                            [
                                position.x,
                                position.y,
                                position.z,
                                orientation.x,
                                orientation.y,
                                orientation.z,
                                orientation.w,
                            ]
                        ),
                    )
                )
                offsets["mocap"].append(record_seconds - event)
            elif topic == topics["imu"]:
                observed_frames["imu"].add(str(message.header.frame_id))
                acceleration = message.linear_acceleration
                angular_velocity = message.angular_velocity
                buffers["imu"].append(
                    (
                        relative,
                        np.array(
                            [
                                acceleration.x,
                                acceleration.y,
                                acceleration.z,
                                angular_velocity.x,
                                angular_velocity.y,
                                angular_velocity.z,
                            ]
                        ),
                    )
                )
                offsets["imu"].append(record_seconds - event)
            elif topic == topics["controller_pid"]:
                observed_frames["controller_pid"].add(
                    str(message.header.frame_id)
                )
                desired_position, desired_velocity, nominal = _pid_vectors(
                    message
                )
                buffers["pid"].append(
                    (
                        relative,
                        np.concatenate(
                            (desired_position, desired_velocity, nominal)
                        ),
                    )
                )
                offsets["pid"].append(record_seconds - event)
            elif topic == topics["four_axis_command"]:
                values = np.asarray(message.base_thrust, dtype=float)
                if values.shape == (4,) and np.all(np.isfinite(values)):
                    buffers["four_axis"].append((relative, values))
            elif topic == topics["flight_state"]:
                buffers["flight_state"].append(
                    (relative, np.array([int(message.data)], dtype=float))
                )

        gain_end = genpy.Time.from_sec(end)
        for _, message, _ in bag.read_messages(
            topics=[topics["roll_pitch_gain"]],
            end_time=gain_end,
        ):
            gain.update(
                {
                    str(item.name): float(item.value)
                    for item in message.doubles
                }
            )

    mocap_times, mocap_values = _deduplicate(buffers["mocap"])
    imu_times, imu_values = _deduplicate(buffers["imu"])
    pid_times, pid_values = _deduplicate(buffers["pid"])
    _, four_axis = _deduplicate(buffers["four_axis"])
    _, flight_state = _deduplicate(buffers["flight_state"])
    if (
        mocap_times.size < 3
        or imu_times.size < 3
        or pid_times.size < 3
        or mocap_values.shape[1:] != (7,)
        or imu_values.shape[1:] != (6,)
        or pid_values.shape[1:] != (18,)
    ):
        raise ValueError("declared interval lacks required mocap/IMU/PID data")
    expected_frames = config["conventions"]["expected_header_frames"]
    for role, frames in observed_frames.items():
        if not frames or not frames.issubset(set(expected_frames[role])):
            raise ValueError(
                "{} header frame(s) {} do not match frozen conventions".format(
                    role, sorted(frames)
                )
            )
    return BagIntervalData(
        episode_id=str(episode["episode_id"]),
        stratum=str(episode["stratum"]),
        bag_path=str(path),
        source_bag_sha256=expected_hash,
        bag_start_time=bag_start,
        interval_start_offset_s=float(episode["start_offset_s"]),
        interval_end_offset_s=float(episode["start_offset_s"])
        + float(episode["duration_s"]),
        mocap_times=mocap_times,
        mocap_positions=mocap_values[:, :3],
        mocap_quaternions=mocap_values[:, 3:],
        imu_times=imu_times,
        accelerometer=imu_values[:, :3],
        gyro=imu_values[:, 3:],
        pid_times=pid_times,
        desired_position_euler=pid_values[:, :6],
        desired_velocity=pid_values[:, 6:12],
        nominal_acceleration=pid_values[:, 12:],
        four_axis_thrust=four_axis,
        roll_pitch_gain=gain,
        flight_states=tuple(int(item) for item in flight_state.reshape(-1)),
        topic_counts=counts,
        header_record_offset_median_s={
            key: (
                None
                if not values
                else float(np.median(np.asarray(values, dtype=float)))
            )
            for key, values in offsets.items()
        },
        observed_header_frames={
            key: tuple(sorted(values))
            for key, values in observed_frames.items()
        },
    )


def _interp(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.column_stack(
        [np.interp(query, times, array[:, index]) for index in range(array.shape[1])]
    )


def _interp_rotations(
    times: np.ndarray, quaternions: np.ndarray, query: np.ndarray
) -> np.ndarray:
    rotations = Slerp(times, Rotation.from_quat(quaternions))(query)
    return rotations.as_rotvec()


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(probability, cumulative, values[order]))


def _parameter_summary(
    posterior: EffectiveResponsePosterior, probability: float
) -> Mapping[str, Mapping[str, float]]:
    weights = np.asarray(posterior.weights)
    fields = {
        "roll_effectiveness": np.asarray(
            [item.effectiveness[3, 3] for item in posterior.samples]
        ),
        "pitch_effectiveness": np.asarray(
            [item.effectiveness[4, 4] for item in posterior.samples]
        ),
        "roll_from_pitch_cross_coupling": np.asarray(
            [item.effectiveness[3, 4] for item in posterior.samples]
        ),
        "pitch_from_roll_cross_coupling": np.asarray(
            [item.effectiveness[4, 3] for item in posterior.samples]
        ),
        "roll_delay_s": np.asarray(
            [item.delay_s[3] for item in posterior.samples]
        ),
        "pitch_delay_s": np.asarray(
            [item.delay_s[4] for item in posterior.samples]
        ),
    }
    tail = 0.5 * (1.0 - float(probability))
    return {
        name: {
            "mean": float(np.dot(weights, values)),
            "lower": _weighted_quantile(values, weights, tail),
            "upper": _weighted_quantile(values, weights, 1.0 - tail),
        }
        for name, values in fields.items()
    }


def analyze_interval(
    data: BagIntervalData,
    config: Mapping[str, Any],
    source_commit: str,
) -> Tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    smoother_values = config["smoother"]
    seed = int(config["seed"])
    smoother = smooth_trajectory(
        TrajectoryObservations(
            mocap_times=data.mocap_times,
            mocap_positions_world=data.mocap_positions,
            mocap_quaternions_xyzw=data.mocap_quaternions,
            imu_times=data.imu_times,
            accelerometer_body=data.accelerometer,
            gyro_body=data.gyro,
        ),
        SmootherConfig(
            trajectory_sample_count=int(config["trajectory_sample_count"]),
            seed=seed,
            mocap_position_sigma=float(
                smoother_values["mocap_position_sigma"]
            ),
            mocap_orientation_sigma=float(
                smoother_values["mocap_orientation_sigma_rad"]
            ),
            accelerometer_noise_sigma=float(
                smoother_values["accelerometer_noise_sigma"]
            ),
            gyro_noise_sigma=float(
                smoother_values["gyro_noise_sigma_rad_s"]
            ),
            mocap_nis_gate=float(smoother_values["mocap_nis_gate"]),
        ),
    )
    start = max(
        data.interval_start_offset_s,
        float(smoother.timestamps[0]),
        float(data.pid_times[0]),
    )
    end = min(
        data.interval_end_offset_s,
        float(smoother.timestamps[-1]),
        float(data.pid_times[-1]),
    )
    rate = float(config["sample_rate_hz"])
    count = int(np.floor((end - start) * rate)) + 1
    if count < 20:
        raise ValueError("common smoother/PID interval is too short")
    times = start + np.arange(count) / rate
    actual_position = np.column_stack(
        (
            _interp(smoother.timestamps, smoother.position_world, times),
            _interp_rotations(
                smoother.timestamps, smoother.quaternion_xyzw, times
            ),
        )
    )
    actual_velocity = np.column_stack(
        (
            _interp(smoother.timestamps, smoother.velocity_world, times),
            _interp(
                smoother.timestamps,
                smoother.angular_velocity_body,
                times,
            ),
        )
    )
    desired_euler = _interp(
        data.pid_times, data.desired_position_euler, times
    )
    desired_position = np.array(desired_euler, copy=True)
    desired_position[:, 3:] = Rotation.from_euler(
        "xyz", desired_euler[:, 3:]
    ).as_rotvec()
    desired_velocity = _interp(
        data.pid_times, data.desired_velocity, times
    )
    command = _interp(data.pid_times, data.nominal_acceleration, times)

    covariance_diagonal = np.column_stack(
        [
            np.interp(
                times,
                smoother.timestamps,
                smoother.covariance[:, index, index],
            )
            for index in range(smoother.covariance.shape[1])
        ]
    )
    position_std = np.sqrt(
        np.maximum(
            np.column_stack(
                (covariance_diagonal[:, :3], covariance_diagonal[:, 6:9])
            ),
            0.0,
        )
    )
    velocity_std = np.sqrt(
        np.maximum(
            np.column_stack(
                (
                    covariance_diagonal[:, 3:6],
                    np.full(
                        (count, 3),
                        float(smoother_values["gyro_noise_sigma_rad_s"])
                        ** 2,
                    ),
                )
            ),
            0.0,
        )
    )

    nominal_position = np.empty_like(actual_position)
    nominal_velocity = np.empty_like(actual_velocity)
    nominal_position[0] = actual_position[0]
    nominal_velocity[0] = actual_velocity[0]
    equilibrium_command = np.median(command, axis=0)
    local_command = command - equilibrium_command
    nominal_rotation = Rotation.from_rotvec(actual_position[0, 3:])
    for index, delta in enumerate(np.diff(times), start=1):
        acceleration = local_command[index - 1]
        nominal_position[index, :3] = (
            nominal_position[index - 1]
            [:3]
            + nominal_velocity[index - 1, :3] * delta
            + 0.5 * acceleration[:3] * delta * delta
        )
        rotation_increment = (
            nominal_velocity[index - 1, 3:] * delta
            + 0.5 * acceleration[3:] * delta * delta
        )
        nominal_rotation = (
            nominal_rotation * Rotation.from_rotvec(rotation_increment)
        )
        nominal_position[index, 3:] = nominal_rotation.as_rotvec()
        nominal_velocity[index] = (
            nominal_velocity[index - 1] + acceleration * delta
        )

    batches = []
    for sample_index, sample_id in enumerate(smoother.sample_ids):
        sample_position = np.column_stack(
            (
                _interp(
                    smoother.timestamps,
                    smoother.sample_position_world[sample_index],
                    times,
                ),
                _interp_rotations(
                    smoother.timestamps,
                    smoother.sample_quaternion_xyzw[sample_index],
                    times,
                ),
            )
        )
        sample_velocity = np.column_stack(
            (
                _interp(
                    smoother.timestamps,
                    smoother.sample_velocity_world[sample_index],
                    times,
                ),
                _interp(
                    smoother.timestamps,
                    smoother.sample_angular_velocity_body[sample_index],
                    times,
                ),
            )
        )
        batches.append(
            TrajectoryTransitionBatch(
                timestamps=times,
                generalized_position=sample_position,
                generalized_velocity=sample_velocity,
                commands=command,
                episode_id=data.episode_id,
                trajectory_sample_id=int(sample_id),
                trajectory_weight=float(
                    smoother.sample_weights[sample_index]
                ),
            )
        )
    response_values = config["effective_response"]
    response = fit_effective_response(
        batches,
        EffectiveResponseFitConfig(
            delay_grid_s=np.asarray(response_values["delay_grid_s"]),
            time_constant_grid_s=np.asarray(
                response_values["time_constant_grid_s"]
            ),
            posterior_sample_count=int(
                response_values["posterior_sample_count"]
            ),
            em_iterations=int(response_values["em_iterations"]),
            position_sigma=float(response_values["position_sigma"]),
            velocity_sigma=float(response_values["velocity_sigma"]),
            seed=seed,
        ),
    )

    nominal_actual_residual = np.array(
        actual_position - nominal_position, copy=True
    )
    nominal_actual_residual[:, 3:] = (
        Rotation.from_rotvec(nominal_position[:, 3:]).inv()
        * Rotation.from_rotvec(actual_position[:, 3:])
    ).as_rotvec()
    desired_actual_residual = np.array(
        actual_position - desired_position, copy=True
    )
    desired_actual_residual[:, 3:] = (
        Rotation.from_rotvec(desired_position[:, 3:]).inv()
        * Rotation.from_rotvec(actual_position[:, 3:])
    ).as_rotvec()
    actual_acceleration = np.gradient(actual_velocity, times, axis=0)
    acceleration_mismatch = actual_acceleration - command
    input_hash = stable_hash(
        {
            "mocap_times": data.mocap_times,
            "mocap_positions": data.mocap_positions,
            "mocap_quaternions": data.mocap_quaternions,
            "imu_times": data.imu_times,
            "accelerometer": data.accelerometer,
            "gyro": data.gyro,
            "pid_times": data.pid_times,
            "desired_position": data.desired_position_euler,
            "desired_velocity": data.desired_velocity,
            "nominal_acceleration": data.nominal_acceleration,
        }
    )
    gates = {
        "exact_controller_replay": {
            "passed": False,
            "status": "ORACLE_UNAVAILABLE",
            "reason": "deployed PC/MCU exact replay backend is not connected",
        },
        "bag_derived_exact_fixture": {
            "passed": False,
            "status": "BAG_DERIVED_FIXTURE_UNAVAILABLE",
        },
        "probability_calibration": {
            "passed": False,
            "status": "NO_COMPLETE_12_FOLD_SELECTION_RESULT",
        },
        "joint_state_parameter_dependence": {
            "passed": False,
            "status": "MODULAR_TRAJECTORY_MIXTURE_ONLY",
        },
        "controller_integrator_state": {
            "passed": False,
            "status": "NOT_RECORDED_OR_LATENTLY_INFERRED",
        },
        "recommendation": {
            "passed": False,
            "status": WORKFLOW_STATUS,
        },
    }
    run_content = {
        "schema": SCHEMA,
        "episode_id": data.episode_id,
        "source_bag_sha256": data.source_bag_sha256,
        "input_slice_sha256": input_hash,
        "config_sha256": config["config_sha256"],
        "source_commit": source_commit,
        "model_id": LowDimensionalEffectiveResponse.model_id,
        "seed": seed,
        "interval": [start, end],
        "gates": gates,
    }
    run_id = stable_hash(run_content)[:20]
    credible = float(config["credible_probability"])
    summary = {
        **run_content,
        "run_id": run_id,
        "workflow_status": WORKFLOW_STATUS,
        "recommendation_available": False,
        "stratum": data.stratum,
        "source_bag": data.bag_path,
        "bag_start_record_time": data.bag_start_time,
        "interval_start_offset_s": start,
        "interval_end_offset_s": end,
        "sample_rate_hz": rate,
        "trajectory_point_count": count,
        "source_topic_counts": data.topic_counts,
        "header_record_offset_median_s": (
            data.header_record_offset_median_s
        ),
        "roll_pitch_gain_at_interval": data.roll_pitch_gain,
        "roll_pitch_gain_interpretation": (
            "last update at or before interval end; this slice does not "
            "evaluate the full bag-8 gain sweep"
        ),
        "flight_states_observed": sorted(set(data.flight_states)),
        "frame_and_unit_conventions": config["conventions"],
        "observed_header_frames": data.observed_header_frames,
        "frame_unit_gate_passed": True,
        "four_axis_thrust": {
            "sample_count": int(data.four_axis_thrust.shape[0]),
            "mean": (
                []
                if not data.four_axis_thrust.size
                else np.mean(data.four_axis_thrust, axis=0)
            ),
        },
        "smoother": {
            "backend": "error_state_ekf_rts",
            "is_smoothed": smoother.is_smoothed,
            "trajectory_sample_count": smoother.sample_count,
            "mocap_update_count": int(np.count_nonzero(smoother.mocap_used)),
            "mocap_rejection_count": int(
                np.count_nonzero(smoother.mocap_rejected)
            ),
            "sampling_approximation": smoother.sampling_approximation,
            "mean_position_std_m": float(np.mean(position_std[:, :3])),
            "mean_orientation_std_rad": float(np.mean(position_std[:, 3:])),
        },
        "effective_response": {
            "model_id": LowDimensionalEffectiveResponse.model_id,
            "posterior_approximation": response.approximation,
            "log_evidence": response.log_evidence,
            "identifiable": response.identifiability.identifiable,
            "design_rank": response.identifiability.design_rank,
            "parameter_count": response.identifiability.parameter_count,
            "condition_number": response.identifiability.condition_number,
            "fit_diagnostics": response.fit_diagnostics,
            "credible_probability": credible,
            "roll_pitch_summary": _parameter_summary(response, credible),
        },
        "trajectory_diagnostics": {
            "desired_actual_position_rms_m": float(
                np.sqrt(np.mean(desired_actual_residual[:, :3] ** 2))
            ),
            "desired_actual_attitude_rms_rad": float(
                np.sqrt(np.mean(desired_actual_residual[:, 3:] ** 2))
            ),
            "nominal_actual_position_rms_m": float(
                np.sqrt(np.mean(nominal_actual_residual[:, :3] ** 2))
            ),
            "nominal_actual_attitude_rms_rad": float(
                np.sqrt(np.mean(nominal_actual_residual[:, 3:] ** 2))
            ),
            "diagnostic_acceleration_mismatch_rms": np.sqrt(
                np.mean(acceleration_mismatch**2, axis=0)
            ),
            "acceleration_derivative_use": (
                "diagnostic_only_not_an_independent_likelihood_observation"
            ),
            "nominal_definition": (
                "equilibrium-centered recorded_PoseControlPid.total with "
                "unit-gain local translation and SO(3) rotation integration "
                "from actual initial state; approximation_not_exact_PC_MCU_replay"
            ),
            "nominal_equilibrium_command": equilibrium_command,
        },
        "gates": gates,
    }
    arrays = {
        "time_offset_s": times,
        "desired_position": desired_position,
        "desired_velocity": desired_velocity,
        "nominal_position": nominal_position,
        "nominal_velocity": nominal_velocity,
        "actual_position_mean": actual_position,
        "actual_velocity_mean": actual_velocity,
        "actual_position_std": position_std,
        "actual_velocity_std": velocity_std,
        "nominal_acceleration_command": command,
        "nominal_actual_log_residual": nominal_actual_residual,
        "desired_actual_log_residual": desired_actual_residual,
    }
    return summary, arrays


def _candidate_rows(config: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    grid = config["candidate_grid"]
    return [
        {
            "candidate_id": "p{:g}_d{:g}_a{:g}".format(p_gain, d_gain, scale),
            "roll_pitch_p": float(p_gain),
            "roll_pitch_d": float(d_gain),
            "allocation_scale": float(scale),
            "success_probability": "",
            "credible_lower": "",
            "credible_upper": "",
            "support": "NOT_CLASSIFIED",
            "exact_evaluation_status": "ORACLE_UNAVAILABLE",
            "recommendable": False,
        }
        for p_gain, d_gain, scale in product(
            grid["roll_pitch_p"],
            grid["roll_pitch_d"],
            grid["allocation_scale"],
        )
    ]


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    diagnostics = summary["trajectory_diagnostics"]
    response = summary["effective_response"]
    lines = [
        "# Grape real-bag vertical slice: {}".format(summary["episode_id"]),
        "",
        "Workflow status: `{}`. Recommendation available: `false`.".format(
            summary["workflow_status"]
        ),
        "",
        "- Run ID: `{}`".format(summary["run_id"]),
        "- Source bag SHA-256: `{}`".format(summary["source_bag_sha256"]),
        "- Interval: `{:.3f}`–`{:.3f}` s from bag start".format(
            summary["interval_start_offset_s"],
            summary["interval_end_offset_s"],
        ),
        "- Source commit: `{}`".format(summary["source_commit"]),
        "- Effective response: `{}`".format(response["model_id"]),
        "",
        "## Trajectory diagnostics",
        "",
        "- desired→actual position RMS: `{:.6g} m`".format(
            diagnostics["desired_actual_position_rms_m"]
        ),
        "- desired→actual attitude RMS: `{:.6g} rad`".format(
            diagnostics["desired_actual_attitude_rms_rad"]
        ),
        "- nominal approximation→actual position RMS: `{:.6g} m`".format(
            diagnostics["nominal_actual_position_rms_m"]
        ),
        "- nominal approximation→actual attitude RMS: `{:.6g} rad`".format(
            diagnostics["nominal_actual_attitude_rms_rad"]
        ),
        "",
        "The nominal curve is an equilibrium-centered local integration of "
        "recorded `PoseControlPid.total`; it is not the deployed PC/MCU "
        "replay oracle. Numerical acceleration is reported only as a "
        "diagnostic and never enters the likelihood.",
        "",
        "## Effective-response posterior",
        "",
        "- Identifiable at the conditional design level: `{}` "
        "(rank `{}/{}`, condition `{}`)".format(
            response["identifiable"],
            response["design_rank"],
            response["parameter_count"],
            response["condition_number"],
        ),
    ]
    for name, interval in response["roll_pitch_summary"].items():
        lines.append(
            "- `{}`: mean `{:.6g}`, interval `[{:.6g}, {:.6g}]`".format(
                name,
                interval["mean"],
                interval["lower"],
                interval["upper"],
            )
        )
    lines.extend(
        [
        "",
        "## Recommendation gates",
        "",
        ]
    )
    for name, gate in summary["gates"].items():
        lines.append(
            "- `{}`: `{}` ({})".format(
                name, gate["passed"], gate["status"]
            )
        )
    lines.extend(
        [
            "",
            "The candidate CSV is therefore an unevaluated common grid. It "
            "contains no success-probability or support claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vertical_slice_artifact(
    output_root: Any,
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> Path:
    root = Path(output_root).expanduser().resolve()
    destination = root / str(summary["episode_id"]) / str(summary["run_id"])
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}.".format(summary["run_id"]),
            dir=str(destination.parent),
        )
    )
    try:
        with (staging / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(
                _plain(summary),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        columns = ["time_offset_s"]
        for name, values in arrays.items():
            if name == "time_offset_s":
                continue
            columns.extend(
                "{}_{}".format(name, axis)
                for axis in AXES[: np.asarray(values).shape[1]]
            )
        with (staging / "trajectory.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for row in range(len(arrays["time_offset_s"])):
                values = [arrays["time_offset_s"][row]]
                for name, array in arrays.items():
                    if name != "time_offset_s":
                        values.extend(np.asarray(array)[row])
                writer.writerow([float(item) for item in values])
        candidates = _candidate_rows(config)
        with (staging / "candidate_grid.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
        _write_report(staging / "REPORT.md", summary)
        os.rename(str(staging), str(destination))
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise
    return destination


def analyze_bag(
    bag_path: Any,
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Any,
    source_commit: Optional[str] = None,
) -> Path:
    data = read_bag_interval(bag_path, episode, config)
    summary, arrays = analyze_interval(
        data, config, source_commit or _source_commit()
    )
    return write_vertical_slice_artifact(
        output_root, summary, arrays, config
    )


__all__ = [
    "AXES",
    "BagIntervalData",
    "SCHEMA",
    "WORKFLOW_STATUS",
    "analyze_bag",
    "analyze_interval",
    "load_vertical_slice_config",
    "read_bag_interval",
    "write_vertical_slice_artifact",
]
