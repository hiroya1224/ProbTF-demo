"""Independent native-belief and fixed-box geometry checks on recorded trials."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from probtf.distributions import DistributionStatus, OrientationKind
from probtf.geometry import quat_to_rotmat
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.box_geometry import BoxCollisionModel, evaluate_box_grasp
from symaware_grasp.grasp_trial import EVENT_TOPIC, GRIPPER_JOINT_NAMES, gripper_opening
from symaware_grasp_ros import record_from_app_message


OBJECT_TOPIC = "/symaware_grasp/object_belief"
TARGET_TOPIC = "/symaware_grasp/grasp_targets"
IK_TOPIC = "/symaware_grasp/symmetry_aware_ik_result"


def _component_check(record, expected_count, name):
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError(name + " has a non-OK native transform distribution")
    if len(record.distribution.components) != expected_count or len(normalized.components) != expected_count:
        raise ValueError(name + " does not retain the expected {} positive-weight components".format(expected_count))
    if any(item.component.orientation.kind is not OrientationKind.FINITE_BINGHAM for item in normalized.components):
        raise ValueError(name + " must retain finite Bingham orientations")
    return {
        "component_count": len(normalized.components),
        "component_ids": [item.component.component_id for item in normalized.components],
        "normalized_weights": [float(item.weight) for item in normalized.components],
        "mode_quaternions_wxyz": [item.component.orientation.mode_wxyz.tolist() for item in normalized.components],
    }


def _target_mode_check(check, expected_modes):
    recorded_modes = np.asarray(check["mode_quaternions_wxyz"])
    similarities = np.abs(recorded_modes @ expected_modes.T)
    matches = np.argmax(similarities, axis=1)
    if len(set(matches.tolist())) != len(expected_modes) or np.any(np.max(similarities, axis=1) < 1.0 - 1e-8):
        raise ValueError("native target component modes do not match the manifest target quaternions")


def _square_box_mode_check(dimensions, object_rotation, modes):
    """Check upright square-prism side normals without trusting mode metadata."""

    dimensions = np.asarray(dimensions, dtype=float)
    rotation = np.asarray(object_rotation, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise ValueError("physical_box size must contain three positive finite dimensions")
    if not np.isclose(dimensions[0], dimensions[1], rtol=0.0, atol=1e-9) or dimensions[2] <= dimensions[0]:
        raise ValueError("box comparison requires a square footprint and a longer vertical dimension")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("physical_box rotation must be a finite proper rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("physical_box rotation must be a finite proper rotation")
    if not np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
        raise ValueError("the physical box must have an upward z axis")
    if modes.ndim != 2 or modes.shape[1] != 4 or len(modes) not in (1, 4) or not np.all(np.isfinite(modes)):
        raise ValueError("box modes must contain one or four finite wxyz quaternions")
    if not np.allclose(np.linalg.norm(modes, axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("box modes must be unit quaternions")
    mode_rotations = np.asarray([quat_to_rotmat(mode) for mode in modes])
    if not np.allclose(mode_rotations[:, :, 2], rotation[:, 2], rtol=0.0, atol=1e-8):
        raise ValueError("all box target modes must preserve the same upward z axis")
    outward_normals = np.array([rotation[:, 0], rotation[:, 1], -rotation[:, 0], -rotation[:, 1]])
    similarities = mode_rotations[:, :, 0] @ outward_normals.T
    matches = np.argmax(similarities, axis=1)
    if np.any(np.max(similarities, axis=1) < 1.0 - 1e-8):
        raise ValueError("target x axes must be outward normals of square-box lateral faces")
    if len(modes) == 4 and set(matches.tolist()) != {0, 1, 2, 3}:
        raise ValueError("four C4 modes must cover all four distinct lateral face normals")
    if len(modes) == 1 and matches[0] != 0:
        raise ValueError("the single box mode must use the reference face normal")
    return {
        "square_footprint": True, "long_vertical_dimension": True,
        "common_upward_z": True, "mode_x_face_indices": matches.tolist(),
        "mode_count": len(modes), "selected_group": "C4" if len(modes) == 4 else "single_reference",
    }


def analyze_box_trial(directory, output=None):
    """Write sample metrics and certify the last second of the recorded hold.

    Clearance is computed against manifest physical_box, independently of a
    representative belief component. Only observed configurations are checked;
    contact dynamics and continuous motion between observations are not certified.
    """
    import rosbag

    directory = Path(directory)
    output = Path(output) if output is not None else directory
    manifest = json.loads((directory / "manifest.json").read_text())
    distribution = manifest["distribution"]
    if distribution not in ("mixture", "single_pose"):
        raise ValueError("box distribution must be mixture or single_pose")
    expected_count = 4 if distribution == "mixture" else 1
    expected_modes = np.asarray(manifest["target_quaternions"], dtype=float)
    if expected_modes.shape != (expected_count, 4) or not np.all(np.isfinite(expected_modes)):
        raise ValueError("manifest target_quaternions must contain the expected wxyz modes")
    norms = np.linalg.norm(expected_modes, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("manifest target quaternions must be unit quaternions")
    expected_modes = expected_modes / norms[:, None]
    start = float(manifest["start_stamp"])
    end = float(manifest["end_stamp"]) if "end_stamp" in manifest else float("inf")
    closed_opening = float(manifest["closed_opening_m"])
    if not math.isfinite(start) or not math.isfinite(closed_opening):
        raise ValueError("start_stamp and closed_opening_m must be finite")
    if "end_stamp" in manifest and (not math.isfinite(end) or end < start):
        raise ValueError("end_stamp must be finite and not precede start_stamp")
    if not isinstance(manifest["completed"], bool):
        raise ValueError("manifest completed must be a boolean")
    physical = manifest["physical_box"]
    dimensions = np.asarray(physical["size"], dtype=float)
    position = np.asarray(physical["position"], dtype=float)
    rotation = np.asarray(physical["rotation"], dtype=float)
    square_geometry = _square_box_mode_check(dimensions, rotation, expected_modes)
    model = directory / manifest["model"]
    model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if "model_sha256" in manifest and model_digest != manifest["model_sha256"]:
        raise ValueError("recorded collision model does not match its manifest hash")
    collision = BoxCollisionModel(model, dimensions=dimensions, object_rotation=rotation)
    robot = ToyArm6DOF()
    rows, events, results = [], [], []
    stage, target = "startup", None
    object_check, target_check = None, None
    object_messages, target_messages = 0, 0
    previous_stamp, previous_theta = None, None
    required_topics = ["/joint_states", EVENT_TOPIC, OBJECT_TOPIC, TARGET_TOPIC, IK_TOPIC]
    topics = required_topics + ["/target_joint_states"]
    with rosbag.Bag(str(directory / "experiment.bag")) as bag:
        recorded = bag.get_type_and_topic_info().topics
        missing = [topic for topic in required_topics if topic not in recorded]
        if missing:
            raise ValueError("bag is missing required box trial topics: " + ", ".join(missing))
        for topic, message, stamp in bag.read_messages(topics=topics):
            stamp = float(stamp.to_sec())
            if topic == EVENT_TOPIC:
                event = json.loads(message.data)
                events.append(event)
                stage = event["stage"]
            elif topic == OBJECT_TOPIC:
                object_check = _component_check(record_from_app_message(message), expected_count, "object belief")
                _square_box_mode_check(dimensions, rotation, object_check["mode_quaternions_wxyz"])
                object_messages += 1
            elif topic == TARGET_TOPIC:
                if len(message.targets) != 1:
                    raise ValueError("box comparison must retain exactly one grasp candidate")
                target_check = _component_check(record_from_app_message(message.targets[0]), expected_count, "grasp candidate")
                _target_mode_check(target_check, expected_modes)
                target_check["candidate_count"] = 1
                target_check["grasp_id"] = str(message.targets[0].grasp_id)
                target_messages += 1
            elif topic == "/target_joint_states":
                mapping = dict(zip(message.name, message.position))
                if all(name in mapping for name in robot.joint_names):
                    target = [float(mapping[name]) for name in robot.joint_names]
            elif topic == IK_TOPIC:
                result = {key: getattr(message, key) for key in (
                    "solver_name", "grasp_id", "success", "total_cost", "position_cost",
                    "orientation_cost", "motion_cost", "joint_limit_cost",
                )}
                result["theta_solution"] = list(message.theta_solution)
                results.append(result)
            elif topic == "/joint_states" and start <= stamp <= end and stage != "finished":
                mapping = dict(zip(message.name, message.position))
                if not all(name in mapping for name in robot.joint_names + GRIPPER_JOINT_NAMES):
                    raise ValueError("recorded joint state lacks an arm or gripper joint")
                theta = np.array([mapping[name] for name in robot.joint_names], dtype=float)
                left, right = (float(mapping[name]) for name in GRIPPER_JOINT_NAMES)
                if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > 1e-8:
                    raise ValueError("recorded jaws must have equal finite joint positions")
                opening = gripper_opening(0.5 * (left + right))
                row = {"time_s": stamp - start, "stage": stage, "opening_m": opening}
                row.update(evaluate_box_grasp(
                    theta, opening, position, expected_modes, robot=robot,
                    dimensions=dimensions, object_rotation=rotation,
                ))
                gap, pair = collision.clearance(theta, opening, position)
                row.update(box_clearance_lower_bound_mm=1000.0 * gap, closest_collision_pair=pair)
                row["joint_speed_rad_s"] = (
                    float(np.max(np.abs(theta - previous_theta))) / (stamp - previous_stamp)
                    if previous_stamp is not None and stamp > previous_stamp else 0.0
                )
                for index, value in enumerate(theta):
                    row["joint_{}".format(index + 1)] = float(value)
                    row["target_{}".format(index + 1)] = float(target[index]) if target is not None else float(value)
                previous_stamp, previous_theta = stamp, theta.copy()
                rows.append(row)
    if not rows or not results or object_check is None or target_check is None:
        raise ValueError("bag has no measured motion, IK result, native object belief or native grasp candidate")
    hold = [row for row in rows if row["stage"] == "hold_grasp"]
    if not hold:
        raise ValueError("bag has no final box grasp observation")
    hold_observation_s = hold[-1]["time_s"] - hold[0]["time_s"]
    tail = [row for row in hold if row["time_s"] >= hold[-1]["time_s"] - 1.0]
    final = tail[-1]
    smallest = min(rows, key=lambda row: row["box_clearance_lower_bound_mm"])
    # The comparison requires at least 0.5 mm even if a manifest sets less.
    minimum_required = float(manifest.get("minimum_required_clearance_mm", 0.5))
    if not math.isfinite(minimum_required):
        raise ValueError("minimum_required_clearance_mm must be finite")
    minimum_required = max(0.5, minimum_required)
    collision_free = smallest["box_clearance_lower_bound_mm"] >= minimum_required
    reached = bool(
        manifest["completed"] and collision_free and hold_observation_s >= 1.0
        and any(result["success"] for result in results)
        and max(row["tcp_object_distance_mm"] for row in tail) <= 2.0
        and max(row["nearest_mode_error_deg"] for row in tail) <= 2.0
        and max(abs(row["gripper_opening_mm"] - 1000.0 * closed_opening) for row in tail) <= 0.2
    )
    summary = {
        "completed": manifest["completed"], "kinematic_grasp_reached": reached,
        "samples": len(rows), "ik_results": results, "final": final,
        "minimum_box_clearance_lower_bound_mm": smallest["box_clearance_lower_bound_mm"],
        "closest_collision_pair": smallest["closest_collision_pair"],
        "collision_free": bool(collision_free), "minimum_required_clearance_mm": minimum_required,
        "distribution": distribution, "object_yaw_pi": manifest.get("object_yaw_pi"),
        "physical_box": physical, "model_sha256": model_digest,
        "square_box_mode_check": square_geometry,
        "native_belief_checks": {
            "object": object_check, "grasp_candidate": target_check,
            "object_messages_checked": object_messages, "target_messages_checked": target_messages,
        },
        "hold_observation_s": hold_observation_s, "final_window_s": 1.0,
        "final_window_samples": len(tail), "events": events,
        "interpretation": "Recorded kinematic approach and jaw closure; contact, friction, holding force and unsampled configurations are not certified.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "events.json").write_text(json.dumps(events, indent=2, allow_nan=False) + "\n")
    return summary
