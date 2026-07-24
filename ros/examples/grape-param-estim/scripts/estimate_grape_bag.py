#!/usr/bin/env python3
"""Estimate Grape inertial parameters directly from a ROS 1 bag.

No ROS master, simulated clock, or ``rosbag play`` process is required.  The
script reads event-time-aligned observations, runs a tempered resample-move
particle filter, and record-time-merges the original and derived messages into
a temporary analysis bag before an atomic rename.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import genpy
import numpy as np
import rosbag
import rospkg
import yaml
from geometry_msgs.msg import WrenchStamped
from probtf_msgs.msg import (
    ApproximationInfo,
    BinghamOrientation,
    ConditionalGaussianTranslation,
    ProbabilisticTransformComponent,
    ProbabilisticTransformStamped,
    Provenance,
)
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Header

from grape_param_estim.dynamics import PARAMETER_NAMES, predict_wrench
from grape_param_estim.grape_geometry import reconstruct_actuator_wrench
from grape_param_estim.kinematics import KinematicsConfig, estimate_kinematics
from grape_param_estim.msg import (
    EstimatorDiagnostics,
    InertialParameterEstimate,
    ParameterParticleSet,
)
from grape_param_estim.particle_filter import (
    ObservationBatch,
    ParameterBounds,
    ParticleFilterConfig,
    StaticParameterParticleFilter,
)


def _default_config_path() -> Path:
    return Path(rospkg.RosPack().get_path("grape_param_estim")) / "config" / "estimator.yaml"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bag", required=True, type=Path)
    parser.add_argument("--output-bag", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--particle-count", type=int, default=None)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="atomically replace an existing output bag")
    parser.add_argument("__name", type=str, default=None)
    parser.add_argument("__log", type=str, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> tuple:
    text = path.read_text(encoding="utf-8")
    mapping = yaml.safe_load(text)
    if not isinstance(mapping, dict):
        raise ValueError("estimator config must contain a YAML mapping.")
    return mapping, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event_time(message, record_time: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        value = float(stamp.to_sec())
        if np.isfinite(value) and value > 0.0:
            return value
    return float(record_time)


def _pose_fields(message) -> tuple:
    pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    quaternion = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=float,
    )
    return position, quaternion


def _strictly_increasing_last(times, *arrays):
    order = np.argsort(times, kind="stable")
    sorted_times = np.asarray(times, dtype=float)[order]
    sorted_arrays = [np.asarray(value)[order] for value in arrays]
    keep = np.ones(sorted_times.size, dtype=bool)
    if sorted_times.size > 1:
        keep[:-1] = np.diff(sorted_times) > 1.0e-9
    return (sorted_times[keep],) + tuple(value[keep] for value in sorted_arrays)


def _read_selected_data(bag_path: Path, config: dict, start_offset: float, duration: float) -> dict:
    topics = config["topics"]
    selected = {
        topics["mocap_pose"],
        topics["calibrated_actuator_wrench"],
        topics["joint_states"],
        topics["rotor_command"],
        topics["flight_state"],
    }
    data = {
        "pose_event": [],
        "pose_record": [],
        "position": [],
        "quaternion": [],
        "pose_frame": [],
        "wrench_event": [],
        "wrench_record": [],
        "wrench": [],
        "wrench_frame": [],
        "joint_event": [],
        "joint_record": [],
        "joint": [],
        "command_event": [],
        "command_record": [],
        "command": [],
        "flight_event": [],
        "flight_state": [],
        "ground_reference_z": None,
    }
    with rosbag.Bag(str(bag_path), "r") as bag:
        bag_start = float(bag.get_start_time())
        real_bag_config = config.get("real_bag", {})
        allowed_states = tuple(real_bag_config.get("allowed_flight_states", (5,)))
        takeoff_state = int(real_bag_config.get("takeoff_state", 3))
        ground_reference_duration = float(
            real_bag_config.get("ground_reference_duration_s", 0.0)
        )
        if (
            takeoff_state in allowed_states
            and np.isfinite(ground_reference_duration)
            and ground_reference_duration > 0.0
        ):
            ground_values = []
            for _, message, _ in bag.read_messages(
                topics=(topics["mocap_pose"],),
                start_time=genpy.Time.from_sec(bag_start),
                end_time=genpy.Time.from_sec(
                    bag_start + ground_reference_duration
                ),
            ):
                try:
                    position, _ = _pose_fields(message)
                except (AttributeError, TypeError, ValueError):
                    continue
                if np.isfinite(position[2]):
                    ground_values.append(float(position[2]))
            if ground_values:
                data["ground_reference_z"] = float(
                    np.median(np.asarray(ground_values, dtype=float))
                )
        start = bag_start + max(0.0, float(start_offset))
        stop = None if duration <= 0.0 else start + float(duration)
        start_time = genpy.Time.from_sec(start)
        end_time = None if stop is None else genpy.Time.from_sec(stop)
        for topic, message, stamp in bag.read_messages(
            topics=selected, start_time=start_time, end_time=end_time
        ):
            record = float(stamp.to_sec())
            if topic == topics["mocap_pose"]:
                try:
                    position, quaternion = _pose_fields(message)
                except (AttributeError, TypeError, ValueError):
                    continue
                if np.all(np.isfinite(position)) and np.all(np.isfinite(quaternion)):
                    data["pose_event"].append(_event_time(message, record))
                    data["pose_record"].append(record)
                    data["position"].append(position)
                    data["quaternion"].append(quaternion)
                    data["pose_frame"].append(str(message.header.frame_id))
            elif topic == topics["calibrated_actuator_wrench"]:
                wrench = np.array(
                    [
                        message.wrench.force.x,
                        message.wrench.force.y,
                        message.wrench.force.z,
                        message.wrench.torque.x,
                        message.wrench.torque.y,
                        message.wrench.torque.z,
                    ],
                    dtype=float,
                )
                if np.all(np.isfinite(wrench)):
                    data["wrench_event"].append(_event_time(message, record))
                    data["wrench_record"].append(record)
                    data["wrench"].append(wrench)
                    data["wrench_frame"].append(str(message.header.frame_id))
            elif topic == topics["joint_states"]:
                positions = dict(zip(message.name, message.position))
                if all(name in positions for name in ("gimbal1", "gimbal2", "gimbal3", "gimbal4")):
                    values = np.array([positions["gimbal{}".format(index)] for index in range(1, 5)])
                    if np.all(np.isfinite(values)):
                        data["joint_event"].append(_event_time(message, record))
                        data["joint_record"].append(record)
                        data["joint"].append(values)
            elif topic == topics["rotor_command"]:
                values = np.asarray(getattr(message, "base_thrust", ()), dtype=float)
                if values.shape == (4,) and np.all(np.isfinite(values)):
                    data["command_event"].append(record)
                    data["command_record"].append(record)
                    data["command"].append(values)
            elif topic == topics["flight_state"]:
                data["flight_event"].append(record)
                data["flight_state"].append(int(message.data))
    if len(data["pose_event"]) < 20:
        raise ValueError("input bag does not contain enough configured mocap pose samples.")
    pose_frames = {value.lstrip("/") for value in data["pose_frame"]}
    if pose_frames != {"world"}:
        raise ValueError(
            "configured mocap poses must be world_from_fc; unsupported parent frames: {}".format(
                sorted(pose_frames)
            )
        )
    wrench_frames = {value.lstrip("/") for value in data["wrench_frame"]}
    if data["wrench_event"] and wrench_frames != {"gimbalrotor/fc"}:
        raise ValueError(
            "calibrated actuator wrench must be expressed at gimbalrotor/fc; got {}".format(
                sorted(wrench_frames)
            )
        )
    return data


def _flight_state_valid_mask(data: dict, grid: np.ndarray, config: dict) -> np.ndarray:
    if not data["flight_event"]:
        return np.ones(grid.size, dtype=bool)

    state_times, states = _strictly_increasing_last(
        data["flight_event"], data["flight_state"]
    )
    state_index = np.searchsorted(state_times, grid, side="right") - 1
    state_available = state_index >= 0
    state_index = np.clip(state_index, 0, len(state_times) - 1)
    selected_states = np.asarray(states[state_index], dtype=int)

    real_bag_config = config.get("real_bag", {})
    allowed_values = real_bag_config.get("allowed_flight_states", (5,))
    if not isinstance(allowed_values, (list, tuple)) or not allowed_values:
        raise ValueError("real_bag.allowed_flight_states must be a non-empty list")
    allowed_states = []
    for value in allowed_values:
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError(
                "real_bag.allowed_flight_states must contain uint8 integers"
            )
        allowed_states.append(value)

    valid = state_available & np.isin(selected_states, allowed_states)
    takeoff_state = int(real_bag_config.get("takeoff_state", 3))
    takeoff = selected_states == takeoff_state
    if takeoff_state not in allowed_states or not np.any(takeoff):
        return valid

    clearance = float(
        real_bag_config.get("minimum_takeoff_clearance_m", np.inf)
    )
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError(
            "real_bag.minimum_takeoff_clearance_m must be finite and non-negative"
        )
    ground_reference = data.get("ground_reference_z")
    if ground_reference is None or not np.isfinite(float(ground_reference)):
        return valid & ~takeoff

    pose_times, positions = _strictly_increasing_last(
        data["pose_event"], data["position"]
    )
    position_z = np.interp(
        grid,
        pose_times,
        positions[:, 2],
        left=np.nan,
        right=np.nan,
    )
    airborne_takeoff = (
        np.isfinite(position_z)
        & (position_z - float(ground_reference) >= clearance)
    )
    return valid & (~takeoff | airborne_takeoff)


def _resample_pose(data: dict, config: dict) -> tuple:
    times, records, positions, quaternions = _strictly_increasing_last(
        data["pose_event"], data["pose_record"], data["position"], data["quaternion"]
    )
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    valid = quaternion_norm > 1.0e-9
    times, records, positions, quaternions = (
        times[valid],
        records[valid],
        positions[valid],
        quaternions[valid] / quaternion_norm[valid, None],
    )
    for index in range(1, quaternions.shape[0]):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    synchronization = config["synchronization"]
    rate = float(synchronization["resample_rate_hz"])
    warmup = float(synchronization.get("warmup_s", 0.0))
    start = times[0] + warmup
    stop = times[-1] - 1.0 / rate
    if stop <= start:
        raise ValueError("configured time window leaves no pose data.")
    grid = start + np.arange(int(np.floor((stop - start) * rate)) + 1) / rate
    positions_grid = np.column_stack(
        [np.interp(grid, times, positions[:, column]) for column in range(3)]
    )
    relative_times = times - times[0]
    grid_relative = grid - times[0]
    quaternions_grid = Slerp(relative_times, Rotation.from_quat(quaternions))(
        grid_relative
    ).as_quat()
    records_grid = np.interp(grid, times, records)
    pose_age = _nearest_age(times, grid)
    pose_valid = pose_age <= float(synchronization["max_pose_age_s"])
    return grid, records_grid, positions_grid, quaternions_grid, pose_valid


def _nearest_age(sample_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sample_times, query, side="left")
    left = np.clip(right - 1, 0, sample_times.size - 1)
    right = np.clip(right, 0, sample_times.size - 1)
    return np.minimum(np.abs(query - sample_times[left]), np.abs(sample_times[right] - query))


def _synchronized_wrench(data: dict, grid: np.ndarray, config: dict) -> tuple:
    synchronization = config["synchronization"]
    if len(data["wrench_event"]) >= 10:
        times, values = _strictly_increasing_last(data["wrench_event"], data["wrench"])
        wrench = np.column_stack(
            [np.interp(grid, times, values[:, column], left=np.nan, right=np.nan) for column in range(6)]
        )
        age = _nearest_age(times, grid)
        valid = np.all(np.isfinite(wrench), axis=1) & (
            age <= float(synchronization.get("max_wrench_age_s", 0.03))
        )
        return wrench, valid, "calibrated_wrench"

    if len(data["command_event"]) < 2 or len(data["joint_event"]) < 2:
        raise ValueError(
            "input bag has neither calibrated actuator wrench nor enough command/joint samples."
        )
    command_times, commands = _strictly_increasing_last(data["command_event"], data["command"])
    joint_times, joints = _strictly_increasing_last(data["joint_event"], data["joint"])
    command_index = np.searchsorted(command_times, grid, side="right") - 1
    command_valid = command_index >= 0
    command_index = np.clip(command_index, 0, len(command_times) - 1)
    held_commands = commands[command_index]
    command_age = grid - command_times[command_index]
    joint_grid = np.column_stack(
        [np.interp(grid, joint_times, joints[:, column]) for column in range(4)]
    )
    joint_age = _nearest_age(joint_times, grid)
    valid = (
        command_valid
        & (command_age >= -1.0e-6)
        & (command_age <= float(synchronization["max_command_age_s"]))
        & (joint_age <= float(synchronization["max_joint_age_s"]))
        & np.all(np.isfinite(joint_grid), axis=1)
    )
    if data["flight_event"]:
        valid &= _flight_state_valid_mask(data, grid, config)
    wrench = reconstruct_actuator_wrench(held_commands, joint_grid)
    return wrench, valid, "command_as_force_effective"


def _make_header(stamp: float, frame_id: str = "gimbalrotor/fc") -> Header:
    header = Header()
    header.stamp = genpy.Time.from_sec(float(stamp))
    header.frame_id = frame_id
    return header


def _provenance(source_bag: Path, mode: str, config_hash: str) -> Provenance:
    message = Provenance()
    message.source_ids = [
        "rosbag:{}".format(source_bag.name),
        "observation:{}".format(mode),
        "config:sha256:{}".format(config_hash),
    ]
    message.method = "tempered_resample_move_particle_filter"
    message.detail = (
        "URDF nominal values are not estimator inputs; calibrated_wrench removes the mass/thrust "
        "scale gauge, while command_as_force_effective reports only effective parameters."
    )
    return message


def _approximation(kind: int, detail: str, source: str) -> ApproximationInfo:
    message = ApproximationInfo()
    message.kind = kind
    message.lossy = True
    message.detail = detail
    message.source = source
    message.has_error_bound = False
    message.error_bound = 0.0
    return message


def _estimate_message(stamp, source_bag, model, summary, update, seed, provenance):
    message = InertialParameterEstimate()
    message.header = _make_header(stamp)
    message.source_bag = str(source_bag)
    message.model = model
    message.parameter_names = list(PARAMETER_NAMES)
    message.mean = summary.mean.tolist()
    message.map = summary.map.tolist()
    message.std = summary.standard_deviation.tolist()
    message.lower_95 = summary.lower_95.tolist()
    message.upper_95 = summary.upper_95.tolist()
    message.covariance = summary.covariance.reshape(-1).tolist()
    message.particle_count = int(update["particle_count"])
    message.effective_sample_size = summary.effective_sample_size
    message.update_index = update["result"].update_index
    message.observation_count = update["result"].observation_count
    message.resampled = update["result"].resampled
    message.log_likelihood = summary.log_evidence
    message.seed = int(seed)
    message.provenance = provenance
    message.approximation = _approximation(
        ApproximationInfo.MONTE_CARLO,
        "weighted particle posterior summarized by moments and marginal quantiles",
        "grape_param_estim",
    )
    return message


def _particle_message(stamp, particle_filter, stride):
    message = ParameterParticleSet()
    message.header = _make_header(stamp)
    message.parameter_names = list(PARAMETER_NAMES)
    retained_count = max(
        1,
        int(np.ceil(particle_filter.particles.shape[0] / max(1, int(stride)))),
    )
    cumulative = np.cumsum(particle_filter.weights)
    cumulative[-1] = 1.0
    positions = (np.arange(retained_count) + 0.5) / retained_count
    indices = np.searchsorted(cumulative, positions, side="left")
    values = particle_filter.particles[indices]
    # Systematic weighted decimation represents multiplicity through repeated
    # indices, so each retained row has equal normalized mass.
    weights = np.full(retained_count, 1.0 / retained_count)
    message.particle_count = len(indices)
    message.stride = len(PARAMETER_NAMES)
    message.values = values.reshape(-1).tolist()
    message.normalized_weight = weights.tolist()
    return message


def _wrench_message(stamp: float, values: np.ndarray) -> WrenchStamped:
    message = WrenchStamped()
    message.header = _make_header(stamp)
    message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = map(float, values[:3])
    message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = map(float, values[3:])
    return message


def _probtf_cog_message(stamp, mean, covariance, provenance):
    message = ProbabilisticTransformStamped()
    message.header = _make_header(stamp)
    message.child_frame_id = "grape_param_estim/estimated_cog"
    message.edge_id = "grape_param_estim_fc_to_cog"
    message.authority = "grape_param_estim"
    message.is_static = False
    message.representative_kind = ProbabilisticTransformStamped.REPRESENTATIVE_MOMENT
    message.representative.translation.x = float(mean[1])
    message.representative.translation.y = float(mean[2])
    message.representative.translation.z = float(mean[3])
    message.representative.rotation.w = 1.0
    component = ProbabilisticTransformComponent()
    component.component_id = "inertial_parameter_posterior_moment"
    component.weight = 1.0
    component.orientation.kind = BinghamOrientation.DIRAC
    component.orientation.inverse_concentration = 0.0
    # JMAA-normalized trace-zero Dirac shape for mode [w,x,y,z]=[1,0,0,0].
    component.orientation.shape_upper_wxyz = [
        1.5,
        0.0,
        0.0,
        0.0,
        -0.5,
        0.0,
        0.0,
        -0.5,
        0.0,
        -0.5,
    ]
    component.orientation.reference_quaternion.w = 1.0
    component.translation.mean_at_reference.x = float(mean[1])
    component.translation.mean_at_reference.y = float(mean[2])
    component.translation.mean_at_reference.z = float(mean[3])
    cog_covariance = covariance[1:4, 1:4]
    component.translation.residual_covariance_upper = [
        float(cog_covariance[0, 0]),
        float(cog_covariance[0, 1]),
        float(cog_covariance[0, 2]),
        float(cog_covariance[1, 1]),
        float(cog_covariance[1, 2]),
        float(cog_covariance[2, 2]),
    ]
    component.translation.rotation_coupling = [0.0] * 27
    component.approximation = _approximation(
        ApproximationInfo.MOMENT_SUMMARY,
        "CoG translation marginal induced from the shared inertial-parameter particles",
        "grape_param_estim",
    )
    component.provenance = provenance
    message.components = [component]
    message.approximation = component.approximation
    message.provenance = provenance
    return message


def _diagnostic_message(
    event_stamp,
    record_stamp,
    result,
    residual,
    sigma,
    rank,
    condition,
    particle_count,
    gate_reason=None,
):
    message = EstimatorDiagnostics()
    message.header = _make_header(event_stamp)
    message.update_index = result.update_index
    message.observation_count = result.observation_count
    message.source_record_time = genpy.Time.from_sec(float(record_stamp))
    message.event_time = genpy.Time.from_sec(float(event_stamp))
    message.ess_before = result.ess_before
    message.ess_after = result.ess_after
    message.resampled = result.resampled
    message.normalized_innovation_squared = float(np.sum((residual / sigma) ** 2))
    message.force_residual_norm = float(np.linalg.norm(residual[:3]))
    message.torque_residual_norm = float(np.linalg.norm(residual[3:]))
    message.log_likelihood_increment = result.log_evidence_increment
    message.mcmc_accepted = result.mcmc_accepted
    message.mcmc_proposed = result.mcmc_proposed
    message.excitation_rank = rank
    message.excitation_condition = condition
    if gate_reason is not None:
        message.gate_reason = str(gate_reason)
    elif rank < len(PARAMETER_NAMES):
        message.gate_reason = "updated_rank_deficient"
    elif not np.isfinite(condition):
        message.gate_reason = "updated_singular"
    else:
        message.gate_reason = "updated"
    return message


def _observation_batches(selected: np.ndarray, batch_size: int) -> list:
    """Partition selected indices without discarding a final partial batch."""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("particle_filter.batch_size must be positive.")
    indices = np.asarray(selected, dtype=int).reshape(-1)
    return [
        indices[offset : offset + size]
        for offset in range(0, indices.size, size)
    ]


def _analysis_events(input_path: Path, config: dict, config_hash: str, args) -> list:
    data = _read_selected_data(input_path, config, args.start_offset, args.duration)
    grid, record_grid, position, quaternion, pose_valid = _resample_pose(data, config)
    wrench, wrench_valid, mode = _synchronized_wrench(data, grid, config)
    observation_config = config["observation"]
    sync_config = config["synchronization"]
    kinematics = estimate_kinematics(
        grid,
        position,
        quaternion,
        config=KinematicsConfig(
            window_length=int(sync_config["savgol_window"]),
            polynomial_order=int(sync_config["savgol_polynomial_order"]),
            position_sigma=float(observation_config["mocap_position_sigma_m"]),
            orientation_sigma=float(np.deg2rad(observation_config["mocap_orientation_sigma_deg"])),
        ),
    )
    valid = (
        kinematics.valid_mask
        & pose_valid
        & wrench_valid
        & np.all(np.isfinite(wrench), axis=1)
    )
    estimation_stride = int(sync_config.get("estimation_stride", 5))
    selected = np.flatnonzero(valid)[:: max(1, estimation_stride)]
    if selected.size < 15:
        raise ValueError("fewer than 15 synchronized, derivative-valid observations remain.")

    parameter_config = config["parameters"]
    if tuple(parameter_config["names"]) != tuple(PARAMETER_NAMES):
        raise ValueError("config parameter names/order do not match the dynamics API.")
    bounds_config = parameter_config["bounded_uniform"]
    bounds = ParameterBounds(bounds_config["lower"], bounds_config["upper"])
    pf_config = config["particle_filter"]
    seed = int(config.get("seed", 7) if args.seed is None else args.seed)
    particle_count = int(
        pf_config["particle_count"] if args.particle_count is None else args.particle_count
    )
    particle_filter = StaticParameterParticleFilter(
        bounds,
        ParticleFilterConfig(
            particle_count=particle_count,
            resample_ess_fraction=float(pf_config["resample_ess_fraction"]),
            tempering_ess_fraction=float(pf_config.get("tempering_ess_fraction", 0.7)),
            mcmc_steps=int(pf_config["mcmc_steps"]),
            local_move_scale=float(pf_config.get("local_move_scale", 0.35)),
            prior_move_probability=float(pf_config.get("prior_move_probability", 0.03)),
            student_t_degrees_of_freedom=float(
                observation_config["student_t_degrees_of_freedom"]
            ),
            seed=seed,
        ),
    )
    force_sigma = np.full(3, float(observation_config["force_residual_sigma_n"]))
    torque_sigma = np.full(3, float(observation_config["torque_residual_sigma_nm"]))
    batch_size = int(pf_config.get("batch_size", 1))
    output_every = int(pf_config.get("output_every_observations", batch_size))
    if output_every <= 0:
        raise ValueError(
            "particle_filter.output_every_observations must be positive."
        )
    observation_batches = _observation_batches(selected, batch_size)
    provenance = _provenance(input_path, mode, config_hash)
    output_config = config["output"]
    events = []
    last_output_count = -output_every
    for batch_index, indices in enumerate(observation_batches):
        batch = ObservationBatch(
            specific_acceleration=kinematics.specific_acceleration_body[indices],
            angular_velocity=kinematics.angular_velocity_body[indices],
            angular_acceleration=kinematics.angular_acceleration_body[indices],
            actuator_wrench=wrench[indices],
            force_sigma=force_sigma,
            torque_sigma=torque_sigma,
        )
        pre_update_mean = np.average(
            particle_filter.particles, axis=0, weights=particle_filter.weights
        )
        pre_update_ess = float(1.0 / np.dot(particle_filter.weights, particle_filter.weights))
        rank, condition = particle_filter.excitation_metrics(
            pre_update_mean, pending_batch=batch
        )
        gating = config.get("gating", {})
        minimum_rank = int(gating.get("minimum_excitation_rank", 1))
        maximum_condition = float(gating.get("maximum_excitation_condition", np.inf))
        should_gate = rank < minimum_rank or (
            rank == len(PARAMETER_NAMES) and condition > maximum_condition
        )
        if should_gate:
            stamp = float(grid[indices[-1]])
            record_stamp = float(record_grid[indices[-1]])
            predicted = predict_wrench(
                pre_update_mean,
                kinematics.specific_acceleration_body[indices[-1]],
                kinematics.angular_velocity_body[indices[-1]],
                kinematics.angular_acceleration_body[indices[-1]],
            )
            residual = wrench[indices[-1]] - predicted
            # A gated batch is not counted as evidence; retain current filter
            # indices and ESS in a diagnostic-only record.
            class _GatedResult:
                update_index = particle_filter.update_index
                observation_count = particle_filter.observation_count
                ess_before = pre_update_ess
                ess_after = pre_update_ess
                resampled = False
                log_evidence_increment = 0.0
                mcmc_accepted = 0
                mcmc_proposed = 0

            events.append(
                (
                    record_stamp,
                    output_config["diagnostics_topic"],
                    _diagnostic_message(
                        stamp,
                        record_stamp,
                        _GatedResult(),
                        residual,
                        np.concatenate((force_sigma, torque_sigma)),
                        rank,
                        condition,
                        particle_count,
                        gate_reason="gated_insufficient_excitation",
                    ),
                )
            )
            continue
        result = particle_filter.update(batch)
        another_batch_remains = batch_index + 1 < len(observation_batches)
        if (
            result.observation_count - last_output_count < output_every
            and another_batch_remains
        ):
            continue
        last_output_count = result.observation_count
        summary = particle_filter.posterior_summary()
        stamp = float(grid[indices[-1]])
        record_stamp = float(record_grid[indices[-1]])
        predicted = predict_wrench(
            summary.mean,
            kinematics.specific_acceleration_body[indices[-1]],
            kinematics.angular_velocity_body[indices[-1]],
            kinematics.angular_acceleration_body[indices[-1]],
        )
        residual = wrench[indices[-1]] - predicted
        rank, condition = particle_filter.excitation_metrics(summary.mean)
        update_bundle = {"result": result, "particle_count": particle_count}
        estimate_message = _estimate_message(
            stamp,
            input_path,
            "{}:{}".format(config["model"], mode),
            summary,
            update_bundle,
            seed,
            provenance,
        )
        events.extend(
            [
                (record_stamp, output_config["estimate_topic"], estimate_message),
                (
                    record_stamp,
                    output_config["particles_topic"],
                    _particle_message(stamp, particle_filter, pf_config["particle_output_stride"]),
                ),
                (
                    record_stamp,
                    output_config["diagnostics_topic"],
                    _diagnostic_message(
                        stamp,
                        record_stamp,
                        result,
                        residual,
                        np.concatenate((force_sigma, torque_sigma)),
                        rank,
                        condition,
                        particle_count,
                    ),
                ),
                (
                    record_stamp,
                    output_config["predicted_wrench_topic"],
                    _wrench_message(stamp, predicted),
                ),
                (
                    record_stamp,
                    output_config["residual_wrench_topic"],
                    _wrench_message(stamp, residual),
                ),
                (
                    record_stamp,
                    output_config["probtf_cog_topic"],
                    _probtf_cog_message(stamp, summary.mean, summary.covariance, provenance),
                ),
            ]
        )
        print(
            "update={:d} observations={:d} ESS={:.1f}/{:d} mass={:.4f} rank={:d} mode={}".format(
                result.update_index,
                result.observation_count,
                summary.effective_sample_size,
                particle_count,
                summary.mean[0],
                rank,
                mode,
            ),
            flush=True,
        )
    if not events:
        raise RuntimeError("particle filter produced no analysis snapshots.")
    return sorted(events, key=lambda item: item[0])


def _write_analysis_bag(input_path: Path, output_path: Path, events: list, force: bool) -> None:
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved:
        raise ValueError("input and output bag paths must differ.")
    if output_path.exists() and not force:
        raise FileExistsError("output bag already exists; pass --force to replace it atomically.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(output_path.name), suffix=".bag", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        # Merge in record-time order.  Appending past-time chunks to a copied
        # bag makes the bag header/index end time regress and is unreliable in
        # rosbag play/Foxglove.  Passing each original connection header keeps
        # caller/latching/type metadata (including /tf_static) while preserving
        # every original message and record timestamp.
        event_index = 0
        original_connections = {}
        with rosbag.Bag(str(input_path), "r") as source:
            with rosbag.Bag(
                str(temporary),
                "w",
                compression=source.compression,
            ) as destination:
                for topic, message, stamp, connection_header in source.read_messages(
                    return_connection_header=True
                ):
                    source_seconds = float(stamp.to_sec())
                    while event_index < len(events) and events[event_index][0] <= source_seconds:
                        record_time, derived_topic, derived_message = events[event_index]
                        destination.write(
                            derived_topic,
                            derived_message,
                            t=genpy.Time.from_sec(float(record_time)),
                        )
                        event_index += 1
                    # rosbag.Bag.write normally collapses all publishers on a
                    # topic into one connection even when a connection header
                    # is supplied.  Select/create the matching private
                    # connection explicitly so two latched /tf_static
                    # publishers retain late-subscriber semantics.
                    header_key = tuple(
                        sorted(
                            (
                                str(key),
                                bytes(value)
                                if isinstance(value, (bytes, bytearray))
                                else str(value),
                            )
                            for key, value in connection_header.items()
                        )
                    )
                    connection_key = (topic, header_key)
                    if connection_key in original_connections:
                        destination._topic_connections[topic] = original_connections[
                            connection_key
                        ]
                    else:
                        destination._topic_connections.pop(topic, None)
                    destination.write(
                        topic,
                        message,
                        t=stamp,
                        connection_header=connection_header,
                    )
                    original_connections.setdefault(
                        connection_key, destination._topic_connections[topic]
                    )
                while event_index < len(events):
                    record_time, derived_topic, derived_message = events[event_index]
                    destination.write(
                        derived_topic,
                        derived_message,
                        t=genpy.Time.from_sec(float(record_time)),
                    )
                    event_index += 1
        os.replace(str(temporary), str(output_path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = _parse_arguments()
    if not np.isfinite(args.start_offset) or args.start_offset < 0.0:
        raise ValueError("--start-offset must be finite and non-negative.")
    if not np.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be finite and non-negative (zero means all data).")
    if args.seed is not None and not 0 <= args.seed <= np.iinfo(np.uint64).max:
        raise ValueError("--seed must fit in uint64.")
    if args.particle_count is not None and args.particle_count < 32:
        raise ValueError("--particle-count must be at least 32.")
    config_path = _default_config_path() if args.config is None else args.config
    if not args.input_bag.is_file():
        raise FileNotFoundError("input bag does not exist: {}".format(args.input_bag))
    config, config_hash = _load_config(config_path)
    events = _analysis_events(args.input_bag, config, config_hash, args)
    _write_analysis_bag(args.input_bag, args.output_bag, events, args.force)
    print(
        json.dumps(
            {
                "input_bag": str(args.input_bag),
                "output_bag": str(args.output_bag),
                "analysis_messages": len(events),
                "config_sha256": config_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
