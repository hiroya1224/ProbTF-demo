"""Read actual ROS bags and independently check native laws and box metrics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rosbag
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from probtf.geometry import quat_to_rotmat, rpy_to_quat
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.box_geometry import box_symmetry_quaternions, make_box_record
from symaware_grasp.box_trial_analysis import analyze_box_trial, _square_box_mode_check
from symaware_grasp.grasp_targets import compose_grasp_targets
from symaware_grasp.grasp_trial import GRIPPER_JOINT_NAMES, animated_gripper_urdf, gripper_joint_position
from symaware_grasp.models import GraspCandidate
from symaware_grasp.msg import IKResult
from symaware_grasp_ros import grasp_targets_to_msg, object_belief_to_msg


def _trial(tmp_path, distribution="mixture", candidate_count=1, object_distribution=None,
           target_distribution=None, final_roll=0.0, box_side=0.100, early_roll=0.06):
    robot = ToyArm6DOF()
    position, quaternion, _ = robot.forward_kinematics(np.zeros(6))
    rotation = quat_to_rotmat(quaternion)
    modes = box_symmetry_quaternions(quaternion)
    if distribution == "single_pose":
        modes = modes[:1]
    package = Path(__import__("symaware_grasp.grasp_trial", fromlist=["__file__"]).__file__).parents[2]
    model = tmp_path / "animated_gripper.urdf"
    model.write_text(animated_gripper_urdf((package / "urdf/simple_six_dof_arm.urdf").read_text()))
    manifest = {
        "distribution": distribution, "completed": True, "start_stamp": 100.0, "end_stamp": 102.7,
        "model": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "physical_box": {"size": [box_side, box_side, 0.180], "position": position.tolist(), "rotation": rotation.tolist()},
        "target_quaternions": modes.tolist(), "closed_opening_m": 0.106,
        "minimum_required_clearance_mm": 0.5,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    object_record = make_box_record(
        distribution=object_distribution or distribution, position=position,
        mode_rpy_deg=(0.0, 0.0, 0.0), dimensions=(box_side, box_side, 0.180), stamp=100.0,
    )
    target_record = make_box_record(
        distribution=target_distribution or distribution, position=position,
        mode_rpy_deg=(0.0, 0.0, 0.0), dimensions=(box_side, box_side, 0.180), stamp=100.0,
    )
    candidate = GraspCandidate("box_center", np.zeros(3), quaternion, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    records = compose_grasp_targets(target_record, [candidate])
    target_message = grasp_targets_to_msg(records, [candidate], "demo_box")
    if candidate_count == 2:
        target_message.targets.append(target_message.targets[0])
    result = IKResult()
    result.solver_name = "symmetry_aware_pointwise"
    result.grasp_id = "box_center"
    result.success = True
    result.theta_solution = [0.0] * 6
    result.total_cost, result.position_cost, result.orientation_cost = 0.2, 0.1, -8.0
    result.motion_cost, result.joint_limit_cost = 0.23, 0.04

    def event(stage, time_s):
        return String(data=json.dumps({"stage": stage, "time_s": time_s, "stamp": 100.0 + time_s, "details": {}}))

    def joint(roll, opening=0.106):
        message = JointState()
        message.name = list(robot.joint_names) + GRIPPER_JOINT_NAMES
        message.position = [0.0] * 5 + [roll] + [gripper_joint_position(opening)] * 2
        return message

    with rosbag.Bag(str(tmp_path / "experiment.bag"), "w") as bag:
        for topic, message, time_s in (
            ("/symaware_grasp/object_belief", object_belief_to_msg(object_record, "demo_box"), -0.2),
            ("/symaware_grasp/grasp_targets", target_message, -0.1),
            ("/symaware_grasp/symmetry_aware_ik_result", result, 0.0),
            ("/symaware_grasp/trial_event", event("observe_box", 0.0), 0.0),
            ("/joint_states", joint(0.0), 0.05),
            ("/symaware_grasp/trial_event", event("hold_grasp", 0.1), 0.1),
            ("/joint_states", joint(early_roll, 0.185), 0.2),
            ("/joint_states", joint(final_roll), 1.5),
            ("/joint_states", joint(final_roll), 2.0),
            ("/joint_states", joint(final_roll), 2.5),
            ("/symaware_grasp/trial_event", event("finished", 2.6), 2.6),
            ("/joint_states", joint(0.4), 2.65),
            ("/joint_states", joint(0.4), 3.0),
        ):
            bag.write(topic, message, t=rospy.Time.from_sec(100.0 + time_s))
    return manifest


@pytest.mark.parametrize("distribution, expected_count", [("mixture", 4), ("single_pose", 1)])
def test_real_bag_retains_candidate_components_and_certifies_only_final_hold(tmp_path, distribution, expected_count):
    _trial(tmp_path, distribution)
    summary = analyze_box_trial(tmp_path)
    assert summary["completed"] and summary["kinematic_grasp_reached"]
    assert summary["samples"] == 5 and summary["final_window_samples"] == 3
    assert summary["ik_results"][0]["motion_cost"] == pytest.approx(0.23)
    assert summary["ik_results"][0]["theta_solution"] == [0.0] * 6
    checks = summary["native_belief_checks"]
    assert checks["object"]["component_count"] == expected_count
    assert checks["grasp_candidate"]["component_count"] == expected_count
    assert checks["grasp_candidate"]["candidate_count"] == 1
    assert summary["square_box_mode_check"]["common_upward_z"]
    assert summary["square_box_mode_check"]["mode_x_face_indices"] == list(range(expected_count))
    assert summary["final"]["stage"] == "hold_grasp"
    assert summary["final"]["time_s"] == pytest.approx(2.5)
    with (tmp_path / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["stage"] == "hold_grasp"
    assert float(rows[-1]["time_s"]) == summary["final"]["time_s"]
    assert float(rows[1]["nearest_mode_error_deg"]) > 2.0
    assert max(float(row["time_s"]) for row in rows) <= 2.5
    assert summary["events"][-1]["stage"] == "finished"


@pytest.mark.parametrize("kind", ["object", "target", "candidate_count"])
def test_native_bag_validation_rejects_collapsed_law_or_changed_candidate_budget(tmp_path, kind):
    _trial(tmp_path, candidate_count=2 if kind == "candidate_count" else 1,
           object_distribution="single_pose" if kind == "object" else None,
           target_distribution="single_pose" if kind == "target" else None)
    with pytest.raises(ValueError, match="components|exactly one"):
        analyze_box_trial(tmp_path)


def test_nearest_mode_error_uses_full_rotation_even_when_tcp_reaches_center(tmp_path):
    _trial(tmp_path, distribution="single_pose", final_roll=0.06)
    summary = analyze_box_trial(tmp_path)
    assert summary["final"]["tcp_object_distance_mm"] < 1e-10
    assert summary["final"]["nearest_mode_error_deg"] > 3.0
    assert not summary["kinematic_grasp_reached"]


def test_all_observed_box_clearances_are_required(tmp_path):
    _trial(tmp_path, box_side=0.1064, early_roll=0.0)
    summary = analyze_box_trial(tmp_path)
    assert summary["final"]["tcp_object_distance_mm"] < 1e-10
    assert summary["final"]["nearest_mode_error_deg"] < 1e-10
    assert summary["minimum_box_clearance_lower_bound_mm"] < 0.5
    assert not summary["collision_free"] and not summary["kinematic_grasp_reached"]


def test_collision_model_hash_is_verified_before_bag_analysis(tmp_path):
    _trial(tmp_path)
    with (tmp_path / "animated_gripper.urdf").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="manifest hash"):
        analyze_box_trial(tmp_path)


@pytest.mark.parametrize("yaw", [0.0, np.radians(67.5), np.radians(123.0)])
def test_square_box_modes_preserve_z_and_cover_four_outward_face_normals(yaw):
    reference = rpy_to_quat(0.0, 0.0, yaw)
    rotation = quat_to_rotmat(reference)
    modes = box_symmetry_quaternions(reference)
    check = _square_box_mode_check([0.100, 0.100, 0.180], rotation, modes)
    assert check["mode_x_face_indices"] == [0, 1, 2, 3]
    # Quaternion signs cannot change physical face normals.
    assert _square_box_mode_check([0.100, 0.100, 0.180], rotation, -modes) == check


def test_manifest_only_d2_modes_are_rejected_independently_of_native_record_match(tmp_path):
    manifest = _trial(tmp_path)
    manifest["target_quaternions"] = np.eye(4).tolist()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same upward z"):
        analyze_box_trial(tmp_path)


@pytest.mark.parametrize("dimensions", [[0.100, 0.080, 0.180], [0.100, 0.100, 0.100]])
def test_c4_comparison_requires_square_footprint_and_long_vertical_axis(dimensions):
    modes = box_symmetry_quaternions([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="square footprint"):
        _square_box_mode_check(dimensions, np.eye(3), modes)


def test_duplicate_face_normals_and_diagonal_faces_are_rejected():
    modes = box_symmetry_quaternions([1.0, 0.0, 0.0, 0.0])
    duplicate = modes.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="four distinct"):
        _square_box_mode_check([0.100, 0.100, 0.180], np.eye(3), duplicate)
    diagonal = box_symmetry_quaternions(rpy_to_quat(0.0, 0.0, np.pi / 4.0))
    with pytest.raises(ValueError, match="outward normals"):
        _square_box_mode_check([0.100, 0.100, 0.180], np.eye(3), diagonal)
