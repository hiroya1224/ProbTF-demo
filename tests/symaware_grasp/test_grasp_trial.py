from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.beliefs import make_transform_record, representative_component
from symaware_grasp.geometry_utils import axially_symmetric_bingham_parameter
from probtf.geometry import rpy_to_quat
from symaware_grasp.grasp_collision import BottleCollisionModel, box_cylinder_gap
from symaware_grasp.grasp_trial import (
    GRIPPER_JOINT_NAMES, animated_gripper_urdf, evaluate_grasp,
    gripper_joint_position, gripper_opening, plan_approach,
    solve_cartesian_pose,
)

PACKAGE = Path(__file__).resolve().parents[2] / "ros/examples/symaware_grasp"
GRASP_THETA = np.array([-0.1772881612, -1.1079135965, 1.4852550422,
                       -0.2362947816, -0.3870982875, 0.2193207126])


def test_animated_fingers_keep_the_original_arm_and_tcp():
    text = (PACKAGE / "urdf/simple_six_dof_arm.urdf").read_text()
    original, animated = ET.fromstring(text), ET.fromstring(animated_gripper_urdf(text))
    for joint in original.findall("joint"):
        new = animated.find("./joint[@name='{}']".format(joint.get("name")))
        assert ET.tostring(joint) == ET.tostring(new)
    assert len(animated.findall("./joint[@type='revolute']")) == 6
    assert len(animated.findall("./joint[@type='prismatic']")) == 2
    palm = animated.find("./link[@name='tool0']")
    assert [v.get("name") for v in palm.findall("visual")] == ["gripper_palm"]
    for name in GRIPPER_JOINT_NAMES:
        joint = animated.find("./joint[@name='{}']".format(name))
        assert joint.find("parent").get("link") == "tool0"


def test_pad_faces_match_the_commanded_full_opening():
    model = ET.fromstring(animated_gripper_urdf((PACKAGE / "urdf/simple_six_dof_arm.urdf").read_text()))
    for opening in (0.185, 0.102, 0.100):
        q = gripper_joint_position(opening)
        faces = []
        for side, sign in (("left", 1), ("right", -1)):
            visual = model.find("./link[@name='gripper_{}_link']/visual[@name='gripper_{}_pad']".format(side, side))
            y = float(visual.find("origin").get("xyz").split()[1])
            width = float(visual.find("geometry/box").get("size").split()[1])
            faces.append(y + sign*q - sign*width/2)
        assert faces[0]-faces[1] == pytest.approx(opening)
        assert gripper_opening(q) == pytest.approx(opening)


@pytest.mark.parametrize("opening", [np.nan, np.inf, -0.1, 0.09, 0.23])
def test_invalid_gripper_openings_are_rejected(opening):
    with pytest.raises(ValueError):
        gripper_joint_position(opening)


def test_cartesian_approach_uses_the_selected_grasp_orientation():
    robot = ToyArm6DOF()
    goal = robot.forward_transform(GRASP_THETA)
    pregrasp, waypoints = plan_approach(robot, GRASP_THETA)
    expected_start = goal[:3, 3]-0.12*goal[:3, 0]
    np.testing.assert_allclose(robot.forward_transform(pregrasp)[:3, 3], expected_start, atol=1e-5)
    for theta, u in zip(waypoints, np.linspace(0, 1, len(waypoints))):
        pose = robot.forward_transform(theta)
        np.testing.assert_allclose(pose[:3, 3], expected_start+u*0.12*goal[:3, 0], atol=1e-5)
        np.testing.assert_allclose(pose[:3, :3], goal[:3, :3], atol=1e-5)
        assert robot.within_limits(theta)


def test_unreachable_approach_is_rejected():
    with pytest.raises(RuntimeError, match="not reachable"):
        solve_cartesian_pose(ToyArm6DOF(), [10, 0, 0], np.eye(3), GRASP_THETA)


def test_grasp_metrics_measure_alignment_without_restricting_free_yaw():
    robot = ToyArm6DOF()
    pose = robot.forward_transform(GRASP_THETA)
    values = evaluate_grasp(GRASP_THETA, 0.102, pose[:3, 3], [0, 0, 1])
    assert values["tcp_object_distance_mm"] == pytest.approx(0)
    assert values["axis_tilt_deg"] < 0.01
    assert values["gripper_clearance_mm"] == pytest.approx(2)


def test_corrected_mount_and_closed_pads_clear_the_bottle():
    xml = animated_gripper_urdf((PACKAGE / "urdf/simple_six_dof_arm.urdf").read_text())
    model = BottleCollisionModel(xml)
    gap, pair = model.clearance(GRASP_THETA, .106, [.78, -.12, .46])
    assert gap == pytest.approx(.003, abs=2e-5)
    assert "pad_collision" in pair
    # Recreate the original long flange: it intersected the bottle by 50 mm.
    root = ET.fromstring(xml)
    collision = root.find("./link[@name='link_6']/collision")
    collision.find("origin").set("xyz", "0.05 0 0")
    collision.find("geometry/cylinder").set("length", "0.10")
    gap, pair = BottleCollisionModel(ET.tostring(root)).clearance(GRASP_THETA, .106, [.78, -.12, .46])
    assert gap < -.049
    assert pair.startswith("link_6:")


def test_direct_transfer_interferes_but_elevated_transfer_clears_bottle():
    model = BottleCollisionModel(animated_gripper_urdf((PACKAGE / "urdf/simple_six_dof_arm.urdf").read_text()))
    constrained_theta = [-.434914899, -.430443983, .408423198, -1.569462356, -1.612191727, 1.592722706]
    pregrasp, approach = plan_approach(ToyArm6DOF(), constrained_theta)
    initial = [.6, -1, 1.2, .2, -.5, .8]
    with pytest.raises(RuntimeError, match="insufficient bottle clearance"):
        model.validate_path([initial, pregrasp], .185, [.78, -.12, .46])
    elevated = [.3, -2, 1.5, 0, .5, 0]
    assert model.validate_path([initial, elevated, pregrasp], .185, [.78, -.12, .46])["clearance_lower_bound_mm"] > 10
    assert model.validate_path(approach, .185, [.78, -.12, .46])["clearance_lower_bound_mm"] > 19


def test_separating_axis_gap_rejects_overlap_and_certifies_rotated_clearance():
    rotation = np.eye(3)
    assert box_cylinder_gap(np.zeros(3), rotation, np.ones(3)*.01, np.zeros(3), [0, 0, 1], .05, .09) < 0
    assert box_cylinder_gap(np.array([0, .064, 0]), rotation, [.03, .01, .04], np.zeros(3), [0, 0, 1], .05, .09) == pytest.approx(.004)


@pytest.mark.parametrize("yaw_concentration", [0, 5000])
def test_bottle_reference_frame_is_identical_for_both_distribution_conditions(yaw_concentration):
    reference = rpy_to_quat(0, 0, 3*np.pi/8)
    parameter = axially_symmetric_bingham_parameter(reference, [0, 0, 1], [5000, 5000, yaw_concentration])
    record = make_transform_record("base_link", "bottle", 0, "bottle", "test", [.78, -.12, .46],
        np.zeros((3, 3)), parameter, reference, "test")
    _, _, representative = representative_component(record)
    assert abs(float(representative @ reference)) == pytest.approx(1)
    assert abs(float(record.representative.rotation_wxyz @ reference)) == pytest.approx(1)
