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
_TRAJECTORY_EVIDENCE_FIELDS = (
    "time_offset_s",
    "desired_position",
    "desired_velocity",
    "nominal_position",
    "nominal_velocity",
    "actual_position_mean",
    "actual_velocity_mean",
    "actual_position_std",
    "actual_velocity_std",
    "nominal_acceleration_command",
    "nominal_actual_log_residual",
    "desired_actual_log_residual",
    "actual_position_samples",
    "actual_velocity_samples",
    "nominal_position_samples",
    "nominal_velocity_samples",
    "actual_sample_ids",
    "actual_sample_weights",
)


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


def _trajectory_evidence_sha256(
    arrays: Mapping[str, np.ndarray]
) -> str:
    missing = [
        name for name in _TRAJECTORY_EVIDENCE_FIELDS if name not in arrays
    ]
    if missing:
        raise ValueError(
            "trajectory evidence is missing {}".format(", ".join(missing))
        )
    return stable_hash(
        {
            name: np.asarray(arrays[name])
            for name in _TRAJECTORY_EVIDENCE_FIELDS
        }
    )


def _source_commit(
    repository: Optional[Any] = None, explicit: Optional[str] = None
) -> str:
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
        if status:
            raise RuntimeError(
                "vertical-slice artifacts require a clean source tree"
            )
        if explicit is not None and str(explicit) != commit:
            raise ValueError(
                "explicit source commit does not match clean HEAD"
            )
        return commit
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(
            "vertical-slice artifacts require a verifiable git revision"
        )


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


def _input_slice_sha256(data: BagIntervalData) -> str:
    """Bind every bag-derived value used by analysis or diagnostics."""

    if not isinstance(data, BagIntervalData):
        raise TypeError("input slice hashing requires BagIntervalData")
    return stable_hash(
        {
            "episode_id": data.episode_id,
            "stratum": data.stratum,
            "source_bag_sha256": data.source_bag_sha256,
            "bag_start_time": data.bag_start_time,
            "interval": [
                data.interval_start_offset_s,
                data.interval_end_offset_s,
            ],
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
            "four_axis_thrust": data.four_axis_thrust,
            "roll_pitch_gain": data.roll_pitch_gain,
            "flight_states": data.flight_states,
            "topic_counts": data.topic_counts,
            "header_record_offset_median_s": (
                data.header_record_offset_median_s
            ),
            "observed_header_frames": data.observed_header_frames,
        }
    )


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


def _integrate_nominal_trajectory(
    times: np.ndarray,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    local_command: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the diagnostic local model from one coherent initial state."""

    stamps = np.asarray(times, dtype=float).reshape(-1)
    position_zero = np.asarray(initial_position, dtype=float).reshape(-1)
    velocity_zero = np.asarray(initial_velocity, dtype=float).reshape(-1)
    command = np.asarray(local_command, dtype=float)
    if (
        stamps.size < 2
        or position_zero.shape != (6,)
        or velocity_zero.shape != (6,)
        or command.shape != (stamps.size, 6)
        or not np.all(np.isfinite(stamps))
        or not np.all(np.isfinite(position_zero))
        or not np.all(np.isfinite(velocity_zero))
        or not np.all(np.isfinite(command))
        or np.any(np.diff(stamps) <= 0.0)
    ):
        raise ValueError("nominal integration inputs are invalid")
    position = np.empty((stamps.size, 6), dtype=float)
    velocity = np.empty_like(position)
    position[0] = position_zero
    velocity[0] = velocity_zero
    rotation = Rotation.from_rotvec(position_zero[3:])
    for index, delta in enumerate(np.diff(stamps), start=1):
        acceleration = command[index - 1]
        position[index, :3] = (
            position[index - 1, :3]
            + velocity[index - 1, :3] * delta
            + 0.5 * acceleration[:3] * delta * delta
        )
        rotation_increment = (
            velocity[index - 1, 3:] * delta
            + 0.5 * acceleration[3:] * delta * delta
        )
        rotation = rotation * Rotation.from_rotvec(rotation_increment)
        position[index, 3:] = rotation.as_rotvec()
        velocity[index] = velocity[index - 1] + acceleration * delta
    return position, velocity


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

    equilibrium_command = np.median(command, axis=0)
    local_command = command - equilibrium_command
    nominal_position, nominal_velocity = _integrate_nominal_trajectory(
        times,
        actual_position[0],
        actual_velocity[0],
        local_command,
    )

    batches = []
    actual_position_samples = []
    actual_velocity_samples = []
    nominal_position_samples = []
    nominal_velocity_samples = []
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
        sample_nominal_position, sample_nominal_velocity = (
            _integrate_nominal_trajectory(
                times,
                sample_position[0],
                sample_velocity[0],
                local_command,
            )
        )
        actual_position_samples.append(sample_position)
        actual_velocity_samples.append(sample_velocity)
        nominal_position_samples.append(sample_nominal_position)
        nominal_velocity_samples.append(sample_nominal_velocity)
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
    input_hash = _input_slice_sha256(data)
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
        "actual_position_samples": np.asarray(actual_position_samples),
        "actual_velocity_samples": np.asarray(actual_velocity_samples),
        "nominal_position_samples": np.asarray(nominal_position_samples),
        "nominal_velocity_samples": np.asarray(nominal_velocity_samples),
        "actual_sample_ids": np.asarray(
            smoother.sample_ids, dtype=np.int64
        ),
        "actual_sample_weights": np.asarray(
            smoother.sample_weights, dtype=float
        ),
    }
    trajectory_hash = _trajectory_evidence_sha256(arrays)
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
        "trajectory_evidence_sha256": trajectory_hash,
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


def _se3_log_residual(
    reference_position_rotvec: np.ndarray,
    actual_position_rotvec: np.ndarray,
) -> np.ndarray:
    """Return Log(T_reference^-1 T_actual) as translation/rotation coordinates."""

    reference = np.asarray(reference_position_rotvec, dtype=float).reshape(-1)
    actual = np.asarray(actual_position_rotvec, dtype=float).reshape(-1)
    if (
        reference.shape != (6,)
        or actual.shape != (6,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(actual))
    ):
        raise ValueError("SE(3) residual inputs must be finite 6-vectors")
    reference_rotation = Rotation.from_rotvec(reference[3:])
    relative_rotation = reference_rotation.inv() * Rotation.from_rotvec(
        actual[3:]
    )
    rotation_vector = relative_rotation.as_rotvec()
    relative_translation = reference_rotation.inv().apply(
        actual[:3] - reference[:3]
    )
    angle = float(np.linalg.norm(rotation_vector))
    cross = np.asarray(
        [
            [0.0, -rotation_vector[2], rotation_vector[1]],
            [rotation_vector[2], 0.0, -rotation_vector[0]],
            [-rotation_vector[1], rotation_vector[0], 0.0],
        ]
    )
    if angle < 1.0e-8:
        coefficient = 1.0 / 12.0 + angle * angle / 720.0
    else:
        coefficient = (
            1.0 - 0.5 * angle / np.tan(0.5 * angle)
        ) / (angle * angle)
    inverse_left_jacobian = (
        np.eye(3)
        - 0.5 * cross
        + coefficient * np.matmul(cross, cross)
    )
    return np.concatenate(
        (inverse_left_jacobian.dot(relative_translation), rotation_vector)
    )


def _se3_log_mean_transform_coordinates(log_coordinates: np.ndarray) -> np.ndarray:
    """Map an SE(3) log-coordinate mean back to translation/rotvec form."""

    coordinates = np.asarray(log_coordinates, dtype=float).reshape(-1)
    if coordinates.shape != (6,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("SE(3) mean coordinates must be a finite 6-vector")
    translation_coordinates = coordinates[:3]
    rotation_vector = coordinates[3:]
    angle = float(np.linalg.norm(rotation_vector))
    cross = np.asarray(
        [
            [0.0, -rotation_vector[2], rotation_vector[1]],
            [rotation_vector[2], 0.0, -rotation_vector[0]],
            [-rotation_vector[1], rotation_vector[0], 0.0],
        ]
    )
    if angle < 1.0e-8:
        first = 0.5 - angle * angle / 24.0
        second = 1.0 / 6.0 - angle * angle / 120.0
    else:
        first = (1.0 - np.cos(angle)) / (angle * angle)
        second = (angle - np.sin(angle)) / (angle**3)
    left_jacobian = (
        np.eye(3)
        + first * cross
        + second * np.matmul(cross, cross)
    )
    return np.concatenate(
        (left_jacobian.dot(translation_coordinates), rotation_vector)
    )


def _vertical_slice_message_inputs(
    summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Mapping[str, Any]:
    """Validate and normalize the factual slice used by ROS message records."""

    if summary.get("schema") != SCHEMA:
        raise ValueError("unsupported vertical-slice summary schema")
    if (
        summary.get("workflow_status") != WORKFLOW_STATUS
        or summary.get("recommendation_available") is not False
    ):
        raise ValueError(
            "analysis messages require the non-recommendable EXPERIMENTAL slice"
        )
    for gate_name in (
        "exact_controller_replay",
        "probability_calibration",
        "recommendation",
    ):
        gate = summary.get("gates", {}).get(gate_name, {})
        if gate.get("passed") is not False:
            raise ValueError(
                "{} must fail before factual-only records are emitted".format(
                    gate_name
                )
            )
    conventions = summary.get("frame_and_unit_conventions", {})
    if (
        summary.get("frame_unit_gate_passed") is not True
        or conventions.get("world_frame") != "ENU"
        or conventions.get("body_frame") != "FLU"
        or "world"
        not in conventions.get("expected_header_frames", {}).get(
            "mocap_pose", ()
        )
    ):
        raise ValueError("analysis message frame/unit provenance is invalid")
    for name in (
        "source_bag_sha256",
        "input_slice_sha256",
        "trajectory_evidence_sha256",
        "config_sha256",
    ):
        digest = str(summary.get(name, ""))
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    run_content = {
        "schema": summary["schema"],
        "episode_id": summary["episode_id"],
        "source_bag_sha256": summary["source_bag_sha256"],
        "input_slice_sha256": summary["input_slice_sha256"],
        "trajectory_evidence_sha256": summary[
            "trajectory_evidence_sha256"
        ],
        "config_sha256": summary["config_sha256"],
        "source_commit": summary["source_commit"],
        "model_id": summary["model_id"],
        "seed": summary["seed"],
        "interval": summary["interval"],
        "gates": summary["gates"],
    }
    if stable_hash(run_content)[:20] != str(summary.get("run_id", "")):
        raise ValueError("vertical-slice run_id does not match its provenance")
    if not str(summary.get("source_commit", "")):
        raise ValueError("vertical-slice source_commit must not be empty")
    topics = tuple(sorted(summary.get("source_topic_counts", {})))
    if len(topics) != 6 or any(not item.startswith("/") for item in topics):
        raise ValueError("vertical-slice source topics are incomplete")

    times = np.asarray(arrays.get("time_offset_s"), dtype=float).reshape(-1)
    desired_position = np.asarray(
        arrays.get("desired_position"), dtype=float
    )
    desired_velocity = np.asarray(
        arrays.get("desired_velocity"), dtype=float
    )
    nominal_position = np.asarray(
        arrays.get("nominal_position"), dtype=float
    )
    nominal_velocity = np.asarray(
        arrays.get("nominal_velocity"), dtype=float
    )
    actual_position = np.asarray(
        arrays.get("actual_position_mean"), dtype=float
    )
    actual_velocity = np.asarray(
        arrays.get("actual_velocity_mean"), dtype=float
    )
    actual_position_samples = np.asarray(
        arrays.get("actual_position_samples"), dtype=float
    )
    actual_velocity_samples = np.asarray(
        arrays.get("actual_velocity_samples"), dtype=float
    )
    nominal_position_samples = np.asarray(
        arrays.get("nominal_position_samples"), dtype=float
    )
    nominal_velocity_samples = np.asarray(
        arrays.get("nominal_velocity_samples"), dtype=float
    )
    sample_ids = np.asarray(
        arrays.get("actual_sample_ids"), dtype=np.int64
    ).reshape(-1)
    sample_weights = np.asarray(
        arrays.get("actual_sample_weights"), dtype=float
    ).reshape(-1)
    point_count = times.size
    sample_count = sample_ids.size
    if (
        point_count < 2
        or np.any(np.diff(times) <= 0.0)
        or any(
            item.shape != (point_count, 6)
            for item in (
                desired_position,
                desired_velocity,
                nominal_position,
                nominal_velocity,
                actual_position,
                actual_velocity,
            )
        )
        or any(
            item.shape != (sample_count, point_count, 6)
            for item in (
                actual_position_samples,
                actual_velocity_samples,
                nominal_position_samples,
                nominal_velocity_samples,
            )
        )
        or sample_weights.shape != (sample_count,)
        or sample_count < 1
        or len(set(int(item) for item in sample_ids)) != sample_count
        or np.any(sample_ids < 0)
        or np.any(sample_weights < 0.0)
        or not np.isclose(np.sum(sample_weights), 1.0, atol=1.0e-10)
        or not all(
            np.all(np.isfinite(item))
            for item in (
                times,
                desired_position,
                desired_velocity,
                nominal_position,
                nominal_velocity,
                actual_position,
                actual_velocity,
                actual_position_samples,
                actual_velocity_samples,
                nominal_position_samples,
                nominal_velocity_samples,
                sample_weights,
            )
        )
    ):
        raise ValueError("vertical-slice trajectory samples are invalid")
    if (
        _trajectory_evidence_sha256(arrays)
        != summary["trajectory_evidence_sha256"]
    ):
        raise ValueError(
            "trajectory evidence does not match trajectory_evidence_sha256"
        )
    interval_start = float(summary["interval_start_offset_s"])
    interval_end = float(summary["interval_end_offset_s"])
    sample_period = 1.0 / float(summary["sample_rate_hz"])
    if (
        not np.isclose(times[0], interval_start, atol=1.0e-9)
        or times[-1] > interval_end + 1.0e-9
        or interval_end - times[-1] >= sample_period + 1.0e-9
        or not np.allclose(
            np.asarray(summary["interval"], dtype=float),
            [interval_start, interval_end],
            atol=1.0e-9,
        )
    ):
        raise ValueError("trajectory timestamps do not match source interval")
    bag_start = float(summary["bag_start_record_time"])
    if not np.isfinite(bag_start):
        raise ValueError("bag_start_record_time must be finite")
    return {
        "times": times,
        "desired_position": desired_position,
        "desired_velocity": desired_velocity,
        "nominal_position": nominal_position,
        "nominal_velocity": nominal_velocity,
        "actual_position": actual_position,
        "actual_velocity": actual_velocity,
        "actual_position_samples": actual_position_samples,
        "actual_velocity_samples": actual_velocity_samples,
        "nominal_position_samples": nominal_position_samples,
        "nominal_velocity_samples": nominal_velocity_samples,
        "sample_ids": sample_ids,
        "sample_weights": sample_weights,
        "topics": topics,
        "absolute_time_ns": np.rint(
            (bag_start + times) * 1.0e9
        ).astype(np.int64),
        "source_interval_ns": np.rint(
            (bag_start + np.asarray([interval_start, interval_end]))
            * 1.0e9
        ).astype(np.int64),
    }


def build_vertical_slice_analysis_records(
    summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Tuple[Any, ...]:
    """Build factual Grape trajectory/mismatch records for a derived bag.

    This deliberately emits no CounterfactualCandidate while the exact
    controller and calibration gates remain unavailable.
    """

    try:
        import genpy
        from geometry_msgs.msg import Transform, Twist
        from grape_param_estim.msg import ModelMismatch, TrajectoryParticleSet
        from probtf_msgs.msg import Provenance
    except ImportError as error:  # pragma: no cover - ROS integration only.
        raise RuntimeError(
            "Grape analysis message generation requires built ROS 1 messages"
        ) from error
    from .artifacts import AnalysisBagRecord

    values = _vertical_slice_message_inputs(summary, arrays)
    sample_ids = [int(item) for item in values["sample_ids"]]
    sample_weights = [float(item) for item in values["sample_weights"]]
    absolute_time_ns = [int(item) for item in values["absolute_time_ns"]]
    source_interval_ns = [
        int(item) for item in values["source_interval_ns"]
    ]

    def ros_time(nanoseconds: int):
        seconds, remainder = divmod(int(nanoseconds), 1_000_000_000)
        return genpy.Time(seconds, remainder)

    stamps = [ros_time(item) for item in absolute_time_ns]
    interval_start = ros_time(source_interval_ns[0])
    interval_end = ros_time(source_interval_ns[1])
    trajectory_end = stamps[-1]

    def transform(position_rotvec: np.ndarray):
        pose = np.asarray(position_rotvec, dtype=float)
        quaternion = Rotation.from_rotvec(pose[3:]).as_quat()
        message = Transform()
        message.translation.x = float(pose[0])
        message.translation.y = float(pose[1])
        message.translation.z = float(pose[2])
        message.rotation.x = float(quaternion[0])
        message.rotation.y = float(quaternion[1])
        message.rotation.z = float(quaternion[2])
        message.rotation.w = float(quaternion[3])
        return message

    def twist(velocity: np.ndarray):
        state = np.asarray(velocity, dtype=float)
        message = Twist()
        message.linear.x = float(state[0])
        message.linear.y = float(state[1])
        message.linear.z = float(state[2])
        message.angular.x = float(state[3])
        message.angular.y = float(state[4])
        message.angular.z = float(state[5])
        return message

    def provenance(kind: str, model_version: str):
        message = Provenance()
        message.source_ids = [
            "source_bag_sha256:{}".format(summary["source_bag_sha256"]),
            "input_slice_sha256:{}".format(summary["input_slice_sha256"]),
            "trajectory_evidence_sha256:{}".format(
                summary["trajectory_evidence_sha256"]
            ),
            "episode_id:{}".format(summary["episode_id"]),
            "run_id:{}".format(summary["run_id"]),
        ]
        message.derived_from_edge_ids = []
        message.method = "{}/matched_trajectory_message_builder".format(SCHEMA)
        message.detail = json.dumps(
            {
                "kind": kind,
                "normalized_dataset_sha256_semantics": (
                    "input_slice_sha256 over normalized adapter evidence"
                ),
                "trajectory_evidence_sha256": summary[
                    "trajectory_evidence_sha256"
                ],
                "effective_response_model_id": summary["model_id"],
                "model_version": model_version,
                "run_id": summary["run_id"],
                "workflow_status": WORKFLOW_STATUS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return message

    def fill_common(
        message: Any, kind: str, model_version: str
    ) -> None:
        message.source_bag_sha256 = [summary["source_bag_sha256"]]
        message.normalized_dataset_sha256 = [summary["input_slice_sha256"]]
        message.source_topics = list(values["topics"])
        message.source_interval_start = interval_start
        message.source_interval_end = interval_end
        message.config_sha256 = summary["config_sha256"]
        message.source_commit = summary["source_commit"]
        message.model_version = model_version
        message.seed = int(summary["seed"])
        message.causal = False
        message.prefix_cutoff = genpy.Time()
        message.provenance = provenance(kind, model_version)

    def trajectory_record(
        kind: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        ids: Sequence[int],
        weights: Sequence[float],
        approximation: str,
        model_version: str,
    ) -> AnalysisBagRecord:
        position_samples = np.asarray(positions, dtype=float)
        velocity_samples = np.asarray(velocities, dtype=float)
        message = TrajectoryParticleSet()
        message.header.stamp = trajectory_end
        message.header.frame_id = "world"
        message.run_id = summary["run_id"]
        message.trajectory_id = "{}:{}".format(summary["run_id"], kind)
        message.trajectory_length = len(stamps)
        message.sample_ids = [int(item) for item in ids]
        message.sample_weights = [float(item) for item in weights]
        message.stamps = stamps
        message.transforms = [
            transform(position_samples[sample_index, time_index])
            for sample_index in range(position_samples.shape[0])
            for time_index in range(position_samples.shape[1])
        ]
        message.twists = [
            twist(velocity_samples[sample_index, time_index])
            for sample_index in range(velocity_samples.shape[0])
            for time_index in range(velocity_samples.shape[1])
        ]
        message.candidate_id = ""
        message.candidate_parameter_names = []
        message.candidate_parameters = []
        message.approximation = approximation
        fill_common(message, kind, model_version)
        return AnalysisBagRecord(
            "/analysis/grape_param_estim/trajectory/{}".format(kind),
            message,
            absolute_time_ns[-1],
        )

    desired_positions = values["desired_position"][None, :, :]
    desired_velocities = values["desired_velocity"][None, :, :]
    records = [
        trajectory_record(
            "desired",
            desired_positions,
            desired_velocities,
            [0],
            [1.0],
            "recorded_controller_target_interpolated",
            "recorded_PoseControlPid_target/interpolated/v1",
        ),
        trajectory_record(
            "nominal",
            values["nominal_position_samples"],
            values["nominal_velocity_samples"],
            sample_ids,
            sample_weights,
            (
                "equilibrium_centered_recorded_PoseControlPid.total;"
                "same_sample_initial_state;nominal_not_exact_pc_mcu_replay"
            ),
            "equilibrium_centered_local_PID_total_integrator/v1",
        ),
        trajectory_record(
            "actual_posterior",
            values["actual_position_samples"],
            values["actual_velocity_samples"],
            sample_ids,
            sample_weights,
            "offline_RTS_coherent_trajectory_samples:{}".format(
                summary["smoother"]["sampling_approximation"]
            ),
            "error_state_EKF_RTS/v1",
        ),
    ]
    probability = float(
        summary["effective_response"]["credible_probability"]
    )
    if not 0.0 < probability < 1.0:
        raise ValueError("credible probability must lie in (0, 1)")
    lower_probability = 0.5 * (1.0 - probability)
    upper_probability = 1.0 - lower_probability

    def weighted_summary(samples: np.ndarray):
        sample_values = np.asarray(samples, dtype=float)
        weights = values["sample_weights"]
        mean = np.average(sample_values, axis=0, weights=weights)
        centered = sample_values - mean
        covariance = np.matmul(
            (centered * weights[:, None]).T,
            centered,
        )
        lower = np.asarray(
            [
                _weighted_quantile(
                    sample_values[:, axis], weights, lower_probability
                )
                for axis in range(6)
            ]
        )
        upper = np.asarray(
            [
                _weighted_quantile(
                    sample_values[:, axis], weights, upper_probability
                )
                for axis in range(6)
            ]
        )
        return mean, lower, upper, covariance

    for time_index, record_time_ns in enumerate(absolute_time_ns):
        tracking_samples = np.asarray(
            [
                _se3_log_residual(
                    values["desired_position"][time_index],
                    values["actual_position_samples"][
                        sample_index, time_index
                    ],
                )
                for sample_index in range(len(sample_ids))
            ]
        )
        model_samples = np.asarray(
            [
                _se3_log_residual(
                    values["nominal_position_samples"][
                        sample_index, time_index
                    ],
                    values["actual_position_samples"][
                        sample_index, time_index
                    ],
                )
                for sample_index in range(len(sample_ids))
            ]
        )
        tracking_mean, tracking_lower, tracking_upper, _ = (
            weighted_summary(tracking_samples)
        )
        model_mean, model_lower, model_upper, model_covariance = (
            weighted_summary(model_samples)
        )
        mismatch = ModelMismatch()
        mismatch.header.seq = time_index
        mismatch.header.stamp = stamps[time_index]
        mismatch.header.frame_id = "world"
        mismatch.run_id = summary["run_id"]
        mismatch.sample_set_id = "{}:actual_posterior".format(
            summary["run_id"]
        )
        mismatch.desired = transform(values["desired_position"][time_index])
        mismatch.nominal = transform(values["nominal_position"][time_index])
        mismatch.actual_posterior_mean = transform(
            values["actual_position"][time_index]
        )
        mismatch.nominal_to_actual_mean = transform(
            _se3_log_mean_transform_coordinates(model_mean)
        )
        mismatch.tracking_residual_mean = tracking_mean.tolist()
        mismatch.tracking_residual_lower = tracking_lower.tolist()
        mismatch.tracking_residual_upper = tracking_upper.tolist()
        mismatch.model_residual_mean = model_mean.tolist()
        mismatch.model_residual_lower = model_lower.tolist()
        mismatch.model_residual_upper = model_upper.tolist()
        mismatch.model_residual_covariance = model_covariance.reshape(
            -1
        ).tolist()
        mismatch.trajectory_sample_count = len(sample_ids)
        mismatch.diagnostics = [
            WORKFLOW_STATUS,
            "recommendation_available=false",
            "nominal_not_exact_pc_mcu_replay",
            "offline_RTS_noncausal",
            "matched_sample_SE3_log_residual",
        ]
        fill_common(
            mismatch,
            "model_mismatch",
            "matched_sample_SE3_log/v1",
        )
        records.append(
            AnalysisBagRecord(
                "/analysis/grape_param_estim/model_mismatch",
                mismatch,
                record_time_ns,
            )
        )
    return tuple(records)


def materialize_vertical_slice_analysis_bag(
    source_bag: Any,
    output_bag: Any,
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    """Merge factual vertical-slice records into a new immutable-source bag."""

    from .artifacts import merge_analysis_bag

    destination = Path(output_bag).expanduser().resolve()
    sidecar = destination.with_suffix(".json")
    if sidecar.exists():
        raise FileExistsError(str(sidecar))
    records = build_vertical_slice_analysis_records(summary, arrays)
    merged = merge_analysis_bag(
        source_bag,
        destination,
        records,
        expected_source_sha256=str(summary["source_bag_sha256"]),
    )
    topic_counts: Dict[str, int] = {}
    message_types: Dict[str, str] = {}
    for record in records:
        topic_counts[record.topic] = topic_counts.get(record.topic, 0) + 1
        message_types[record.topic] = str(
            getattr(record.message, "_type", type(record.message).__name__)
        )
    metadata = {
        **dict(merged),
        "schema": "{}/analysis_bag_manifest/v1".format(SCHEMA),
        "run_id": summary["run_id"],
        "config_sha256": summary["config_sha256"],
        "input_slice_sha256": summary["input_slice_sha256"],
        "trajectory_evidence_sha256": summary[
            "trajectory_evidence_sha256"
        ],
        "source_commit": summary["source_commit"],
        "workflow_status": summary["workflow_status"],
        "recommendation_available": False,
        "analysis_topic_counts": topic_counts,
        "analysis_message_types": message_types,
        "analysis_metadata": str(sidecar),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(sidecar.name),
        suffix=".tmp",
        dir=str(sidecar.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _plain(metadata),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        if sidecar.exists():
            raise FileExistsError(str(sidecar))
        os.rename(temporary_name, str(sidecar))
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return metadata


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
    _vertical_slice_message_inputs(summary, arrays)
    config_payload = {
        key: value
        for key, value in config.items()
        if key != "config_sha256"
    }
    if (
        stable_hash(config_payload) != config.get("config_sha256")
        or config.get("config_sha256") != summary.get("config_sha256")
    ):
        raise ValueError(
            "vertical-slice config does not match config_sha256 provenance"
        )
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
        point_count = len(arrays["time_offset_s"])
        tabular_arrays = [
            (name, np.asarray(values))
            for name, values in arrays.items()
            if name != "time_offset_s"
            and np.asarray(values).ndim == 2
            and np.asarray(values).shape[0] == point_count
        ]
        columns = ["time_offset_s"]
        for name, values in tabular_arrays:
            columns.extend(
                "{}_{}".format(name, axis)
                for axis in AXES[: values.shape[1]]
            )
        with (staging / "trajectory.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            for row in range(point_count):
                values = [arrays["time_offset_s"][row]]
                for _, array in tabular_arrays:
                    values.extend(array[row])
                writer.writerow([float(item) for item in values])
        candidates = _candidate_rows(config)
        with (staging / "candidate_grid.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=tuple(candidates[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(candidates)
        np.savez_compressed(
            str(staging / "trajectory_particles.npz"),
            schema=np.asarray(SCHEMA),
            trajectory_evidence_sha256=np.asarray(
                summary["trajectory_evidence_sha256"]
            ),
            **{
                name: np.asarray(arrays[name])
                for name in _TRAJECTORY_EVIDENCE_FIELDS
            }
        )
        _write_report(staging / "REPORT.md", summary)
        files = {}
        for path in sorted(staging.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        with (staging / "artifact_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "schema": "{}/artifact_manifest/v1".format(SCHEMA),
                    "run_id": summary["run_id"],
                    "trajectory_evidence_sha256": summary[
                        "trajectory_evidence_sha256"
                    ],
                    "files": files,
                },
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
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
    analysis_bag_path: Optional[Any] = None,
) -> Path:
    verified_commit = _source_commit(explicit=source_commit)
    data = read_bag_interval(bag_path, episode, config)
    summary, arrays = analyze_interval(
        data, config, verified_commit
    )
    destination = write_vertical_slice_artifact(
        output_root, summary, arrays, config
    )
    if analysis_bag_path is not None:
        materialize_vertical_slice_analysis_bag(
            bag_path,
            analysis_bag_path,
            summary,
            arrays,
        )
    return destination


def analyze_configured_bags(
    *,
    bag_root: Any,
    episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Any,
    source_commit: Optional[str] = None,
    analysis_bag_root: Optional[Any] = None,
) -> Tuple[Path, ...]:
    """Validate the clean revision once, then write several bag artifacts."""

    verified_commit = _source_commit(explicit=source_commit)
    root = Path(bag_root).expanduser().resolve()
    bag_output_root = (
        None
        if analysis_bag_root is None
        else Path(analysis_bag_root).expanduser().resolve()
    )
    destinations = []
    for episode in episodes:
        data = read_bag_interval(root / episode["bag"], episode, config)
        summary, arrays = analyze_interval(data, config, verified_commit)
        destination = write_vertical_slice_artifact(
            output_root, summary, arrays, config
        )
        destinations.append(destination)
        if bag_output_root is not None:
            materialize_vertical_slice_analysis_bag(
                root / episode["bag"],
                bag_output_root
                / str(summary["episode_id"])
                / "{}.analysis.bag".format(summary["run_id"]),
                summary,
                arrays,
            )
    return tuple(destinations)


__all__ = [
    "AXES",
    "BagIntervalData",
    "SCHEMA",
    "WORKFLOW_STATUS",
    "analyze_bag",
    "analyze_configured_bags",
    "analyze_interval",
    "build_vertical_slice_analysis_records",
    "load_vertical_slice_config",
    "materialize_vertical_slice_analysis_bag",
    "read_bag_interval",
    "write_vertical_slice_artifact",
]
