#!/usr/bin/env python3
"""Generate a deterministic, fully excited synthetic Grape ROS bag.

The script is deliberately an offline command: importing it or running it
does not initialize a ROS node and never contacts a ROS master.  Ground truth
is written for post-run evaluation only; the estimator must not subscribe to
or read that topic.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, WrenchStamped
from probtf_msgs.msg import ApproximationInfo, Provenance
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from grape_param_estim.dynamics import (
    PARAMETER_NAMES,
    predict_wrench,
    validate_physical_parameters,
)
from grape_param_estim.kinematics import KinematicsConfig, estimate_kinematics
from grape_param_estim.msg import InertialParameterEstimate


MOCAP_TOPIC = "/gimbalrotor/mocap/pose"
ACTUATOR_WRENCH_TOPIC = "/grape_param_estim/input/actuator_wrench"
GROUND_TRUTH_TOPIC = "/grape_param_estim/ground_truth"
METADATA_TOPIC = "/grape_param_estim/metadata"

DEFAULT_DURATION = 24.0
DEFAULT_RATE = 50.0
START_TIME = 1.0
POSITION_NOISE_SIGMA_M = 0.01
ORIENTATION_NOISE_SIGMA_RAD = float(np.deg2rad(1.0))
FORCE_NOISE_SIGMA_N = 0.02
TORQUE_NOISE_SIGMA_NM = 0.002
KINEMATICS_WINDOW_LENGTH = 51
KINEMATICS_POLYNOMIAL_ORDER = 3


class GenerationError(RuntimeError):
    """Raised for a user-facing generator configuration error."""


def _default_truth_config() -> Path:
    source_candidate = Path(__file__).resolve().parents[1] / "config" / "sanity_truth.yaml"
    if source_candidate.is_file():
        return source_candidate
    try:
        import rospkg
    except ImportError as exc:
        raise GenerationError(
            "cannot locate config/sanity_truth.yaml; pass --truth-config explicitly"
        ) from exc
    try:
        package_path = Path(rospkg.RosPack().get_path("grape_param_estim"))
        return package_path / "config" / "sanity_truth.yaml"
    except rospkg.ResourceNotFound as exc:
        raise GenerationError(
            "cannot locate config/sanity_truth.yaml; pass --truth-config explicitly"
        ) from exc


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a six-DOF synthetic bag for Grape parameter-estimation "
            "sanity checks."
        )
    )
    parser.add_argument("--output-bag", required=True, help="Output ROS bag path.")
    parser.add_argument("--seed", type=int, default=7, help="Non-negative random seed.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Trajectory duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help="Pose and wrench rate in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--truth-config",
        help="Evaluation-only truth YAML (default: package config/sanity_truth.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing output bag.",
    )
    return parser.parse_args(argv)


def _load_truth(path: Path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise GenerationError("failed to read truth config {}: {}".format(path, exc)) from exc
    if not isinstance(document, dict):
        raise GenerationError("truth config must contain a YAML mapping")

    names = tuple(str(value) for value in document.get("parameter_names", ()))
    if names != tuple(PARAMETER_NAMES):
        raise GenerationError(
            "truth parameter_names must exactly match core order: {}".format(
                list(PARAMETER_NAMES)
            )
        )
    try:
        parameters = np.asarray(document["parameters"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerationError("truth config parameters must be a numeric length-10 array") from exc
    if parameters.shape != (len(PARAMETER_NAMES),):
        raise GenerationError("truth config parameters must contain exactly 10 values")
    try:
        parameters = np.asarray(validate_physical_parameters(parameters), dtype=float)
    except (TypeError, ValueError) as exc:
        raise GenerationError("truth parameters are not physically valid: {}".format(exc)) from exc

    model = str(document.get("model", "synthetic_grape_inertial_parameters"))
    source = str(document.get("source", path))
    return model, source, parameters


def _multisine_pose(relative_time: np.ndarray):
    """Return smooth world-from-fc position and quaternion samples."""

    t = np.asarray(relative_time, dtype=float)
    two_pi_t = 2.0 * np.pi * t
    position = np.column_stack(
        (
            0.18 * np.sin(0.23 * two_pi_t)
            + 0.055 * np.sin(0.71 * two_pi_t + 0.4),
            0.16 * np.sin(0.29 * two_pi_t + 0.7)
            + 0.050 * np.sin(0.79 * two_pi_t + 1.1),
            1.20
            + 0.12 * np.sin(0.19 * two_pi_t + 1.3)
            + 0.045 * np.sin(0.83 * two_pi_t + 0.2),
        )
    )
    euler_xyz = np.column_stack(
        (
            0.30 * np.sin(0.31 * two_pi_t + 0.1)
            + 0.075 * np.sin(0.97 * two_pi_t + 0.8),
            0.27 * np.sin(0.37 * two_pi_t + 0.9)
            + 0.070 * np.sin(1.07 * two_pi_t + 0.3),
            0.35 * np.sin(0.41 * two_pi_t + 0.5)
            + 0.090 * np.sin(0.89 * two_pi_t + 1.4),
        )
    )
    quaternion_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return position, quaternion_xyzw


def _noisy_pose(position, quaternion_xyzw, rng):
    noisy_position = position + rng.normal(
        0.0, POSITION_NOISE_SIGMA_M, size=position.shape
    )
    tangent_noise = rng.normal(
        0.0,
        ORIENTATION_NOISE_SIGMA_RAD,
        size=(quaternion_xyzw.shape[0], 3),
    )
    noisy_rotation = Rotation.from_quat(quaternion_xyzw) * Rotation.from_rotvec(
        tangent_noise
    )
    return noisy_position, noisy_rotation.as_quat()


def _excitation_metrics(parameters, specific, omega, alpha):
    """Numerically measure local ten-parameter excitation at the truth."""

    columns = []
    for index in range(len(PARAMETER_NAMES)):
        step = 1.0e-6 * max(1.0, abs(float(parameters[index])))
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += step
        minus[index] -= step
        forward = predict_wrench(plus, specific, omega, alpha)
        backward = predict_wrench(minus, specific, omega, alpha)
        columns.append(((forward - backward) / (2.0 * step)).reshape(-1))
    jacobian = np.column_stack(columns)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = singular_values[0] * max(jacobian.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    return rank, condition, singular_values


def _header(message, sequence, stamp, frame_id):
    message.header.seq = int(sequence)
    message.header.stamp = stamp
    message.header.frame_id = frame_id


def _pose_message(sequence, stamp, position, quaternion_xyzw):
    message = PoseStamped()
    _header(message, sequence, stamp, "world")
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (
        float(value) for value in position
    )
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = (float(value) for value in quaternion_xyzw)
    return message


def _wrench_message(sequence, stamp, wrench):
    message = WrenchStamped()
    _header(message, sequence, stamp, "gimbalrotor/fc")
    message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = (
        float(value) for value in wrench[:3]
    )
    message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = (
        float(value) for value in wrench[3:]
    )
    return message


def _truth_message(output_path, model, source, parameters, seed, stamp, count):
    message = InertialParameterEstimate()
    _header(message, 0, stamp, "gimbalrotor/fc")
    message.source_bag = str(output_path)
    message.model = model
    message.parameter_names = list(PARAMETER_NAMES)
    values = [float(value) for value in parameters]
    message.mean = values
    message.map = values
    message.std = [0.0] * len(values)
    message.lower_95 = values
    message.upper_95 = values
    message.covariance = [0.0] * (len(values) * len(values))
    message.particle_count = 1
    message.effective_sample_size = 1.0
    message.update_index = 0
    message.observation_count = int(count)
    message.resampled = False
    message.log_likelihood = 0.0
    message.seed = int(seed)
    message.provenance = Provenance(
        source_ids=[source],
        derived_from_edge_ids=[],
        method="synthetic_generator_ground_truth",
        detail="Evaluation-only truth; forbidden as estimator input.",
    )
    message.approximation = ApproximationInfo(
        kind=ApproximationInfo.EXACT,
        lossy=False,
        detail="Exact configured synthetic truth.",
        source="generate_sanity_bag.py",
        has_error_bound=True,
        error_bound=0.0,
    )
    return message


def _validate_cli(args, output_path):
    if args.seed < 0 or args.seed > np.iinfo(np.uint64).max:
        raise GenerationError("--seed must be in the uint64 range")
    if not np.isfinite(args.duration) or args.duration < 2.0:
        raise GenerationError("--duration must be finite and at least 2 seconds")
    if not np.isfinite(args.rate) or args.rate < 20.0:
        raise GenerationError("--rate must be finite and at least 20 Hz")
    if output_path.exists() and not args.force:
        raise GenerationError(
            "output already exists: {} (use --force to replace it)".format(output_path)
        )
    if output_path.exists() and not output_path.is_file():
        raise GenerationError(
            "output path exists and is not a regular file: {}".format(output_path)
        )
    if not output_path.parent.is_dir():
        raise GenerationError(
            "output parent directory does not exist: {}".format(output_path.parent)
        )


def generate(args):
    output_path = Path(args.output_bag).expanduser().resolve()
    _validate_cli(args, output_path)
    truth_path = (
        Path(args.truth_config).expanduser().resolve()
        if args.truth_config
        else _default_truth_config().resolve()
    )
    model, source, truth = _load_truth(truth_path)
    rng = np.random.default_rng(args.seed)

    sample_count = int(np.floor(args.duration * args.rate)) + 1
    relative_time = np.arange(sample_count, dtype=float) / float(args.rate)
    position, quaternion_xyzw = _multisine_pose(relative_time)
    noisy_position, noisy_quaternion_xyzw = _noisy_pose(
        position, quaternion_xyzw, rng
    )

    kinematics = estimate_kinematics(
        relative_time,
        position,
        quaternion_xyzw,
        KinematicsConfig(
            window_length=KINEMATICS_WINDOW_LENGTH,
            polynomial_order=KINEMATICS_POLYNOMIAL_ORDER,
            position_sigma=POSITION_NOISE_SIGMA_M,
            orientation_sigma=ORIENTATION_NOISE_SIGMA_RAD,
        ),
    )
    valid_indices = np.flatnonzero(kinematics.valid_mask)
    if valid_indices.size == 0:
        raise GenerationError("kinematics produced no valid samples")
    specific = kinematics.specific_acceleration_body[valid_indices]
    omega = kinematics.angular_velocity_body[valid_indices]
    alpha = kinematics.angular_acceleration_body[valid_indices]
    calibrated_wrench = predict_wrench(truth, specific, omega, alpha)
    wrench_sigma = np.array(
        [FORCE_NOISE_SIGMA_N] * 3 + [TORQUE_NOISE_SIGMA_NM] * 3,
        dtype=float,
    )
    measured_wrench = calibrated_wrench + rng.normal(
        0.0, wrench_sigma, size=calibrated_wrench.shape
    )

    excitation_rank, excitation_condition, singular_values = _excitation_metrics(
        truth, specific, omega, alpha
    )
    if excitation_rank != len(PARAMETER_NAMES):
        raise GenerationError(
            "synthetic trajectory is not full rank: {} of {}".format(
                excitation_rank, len(PARAMETER_NAMES)
            )
        )

    absolute_time = START_TIME + relative_time
    first_stamp = rospy.Time.from_sec(float(absolute_time[0]))
    metadata = {
        "schema": "grape_param_estim.synthetic.v1",
        "generator": "generate_sanity_bag.py",
        "seed": int(args.seed),
        "duration_s": float(relative_time[-1]),
        "rate_hz": float(args.rate),
        "sample_count": int(sample_count),
        "valid_wrench_count": int(valid_indices.size),
        "frames": {"parent": "world", "body": "gimbalrotor/fc"},
        "pose_convention": "world_from_body quaternion xyzw",
        "derivative_filter": {
            "kind": "Savitzky-Golay",
            "window_length": KINEMATICS_WINDOW_LENGTH,
            "polynomial_order": KINEMATICS_POLYNOMIAL_ORDER,
        },
        "truth_config": str(truth_path),
        "truth_model": model,
        "truth_source": source,
        "truth_is_estimator_input": False,
        "noise": {
            "position_sigma_m": POSITION_NOISE_SIGMA_M,
            "orientation_tangent_sigma_rad": ORIENTATION_NOISE_SIGMA_RAD,
            "force_sigma_n": FORCE_NOISE_SIGMA_N,
            "torque_sigma_nm": TORQUE_NOISE_SIGMA_NM,
        },
        "excitation": {
            "local_parameter_rank": excitation_rank,
            "condition": excitation_condition,
            "singular_values": [float(value) for value in singular_values],
        },
        "topics": {
            "mocap": MOCAP_TOPIC,
            "calibrated_actuator_wrench": ACTUATOR_WRENCH_TOPIC,
            "ground_truth_evaluation_only": GROUND_TRUTH_TOPIC,
            "metadata": METADATA_TOPIC,
        },
    }

    valid_to_wrench = {
        int(sample_index): measured_wrench[wrench_index]
        for wrench_index, sample_index in enumerate(valid_indices)
    }
    with rosbag.Bag(str(output_path), "w") as bag:
        bag.write(
            GROUND_TRUTH_TOPIC,
            _truth_message(
                output_path,
                model,
                source,
                truth,
                args.seed,
                first_stamp,
                valid_indices.size,
            ),
            first_stamp,
        )
        bag.write(
            METADATA_TOPIC,
            String(data=json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            first_stamp,
        )
        for index, seconds in enumerate(absolute_time):
            stamp = rospy.Time.from_sec(float(seconds))
            bag.write(
                MOCAP_TOPIC,
                _pose_message(
                    index,
                    stamp,
                    noisy_position[index],
                    noisy_quaternion_xyzw[index],
                ),
                stamp,
            )
            if index in valid_to_wrench:
                bag.write(
                    ACTUATOR_WRENCH_TOPIC,
                    _wrench_message(index, stamp, valid_to_wrench[index]),
                    stamp,
                )

    summary = {
        "output_bag": str(output_path),
        "samples": sample_count,
        "wrench_samples": int(valid_indices.size),
        "excitation_rank": excitation_rank,
        "excitation_condition": excitation_condition,
        "seed": int(args.seed),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def main(argv=None):
    try:
        args = _parse_args(argv)
        generate(args)
    except (GenerationError, OSError, rosbag.ROSBagException, ValueError) as exc:
        print("generate_sanity_bag.py: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
