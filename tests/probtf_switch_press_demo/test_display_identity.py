"""Check the actual ROS publication path against the frozen IK inputs."""
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rospy

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_right_matrix, rpy_to_quat
from probtf_ros import transform_distribution_from_msg
from probtf_switch_press_demo.control import GoalLikelihood, Prediction
from probtf_switch_press_demo.model import BODY_FRAMES, COMPARISON_METHOD, Model
from probtf_switch_press_demo.ros_display import Display, build_belief_messages
from probtf_switch_press_demo.scene import load_config, transform, write_robot


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


def _wire(message):
    buffer = BytesIO()
    message.serialize(buffer)
    return type(message)().deserialize(buffer.getvalue())


def _bytes(message):
    buffer = BytesIO()
    message.serialize(buffer)
    return buffer.getvalue()


def _sensor_record(stamp):
    components = []
    for index, (weight, mean, angles) in enumerate((
            (.3, [.02, -.03, .6], [.1, -.2, .3]),
            (.7, [.04, .01, .64], [-.2, .25, -.3]))):
        mode = rpy_to_quat(*angles)
        basis = quat_right_matrix(mode)
        orientation = BinghamOrientation.from_parameter_matrix(
            basis @ np.diag([0., -37., -61., -94.]) @ basis.T, mode)
        coupling = np.zeros((3, 9))
        coupling[0, 0], coupling[1, 4], coupling[2, 8] = .03, -.02, .01
        components.append(TransformComponent("image_branch_" + str(index), weight, orientation,
            ConditionalGaussianTranslation(mean, np.diag([.002, .003, .004])**2, coupling)))
    return TransformDistributionStamped("camera_optical_frame", "switch_marker", stamp,
        "native_image_marker", "actual_image_detector", TransformDistribution(tuple(components)))


@pytest.fixture
def case(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    model = Model.from_config(urdf, rod_length=config["rod"]["length"],
        hand_camera=transform(config["camera"]["translation"], config["camera"]["rpy"]))
    observed = np.array(config["initial_command"])
    stamp = 12.345
    sensor = _sensor_record(stamp)
    goal = GoalLikelihood(model, sensor, observed, config["marker"]["goal_offset"],
                          config["marker"]["goal_orientation_wxyz"])
    command = observed + [.014, -.015, .043, -.007, -.014, 0.]
    hand = model.hand(command, stamp)
    score, diagnostics = goal.objective.evaluate(hand)
    prediction = Prediction(command-observed, command, hand, score, diagnostics)
    prediction.records = model.display_records(command, stamp, hand_summary=hand)
    trial = SimpleNamespace(belief_records=prediction.records + (goal.record,),
        observation_preview=dict(marker_record=goal.marker_record, goal_record=goal.record,
                                 stamp=stamp, source_command=observed, sensor_record=sensor),
        pose=observed+.7, command=command.copy(), raw_image=None,
        phase="tracking", cycle=2, ik_iteration=3, ik_pending=True)
    return SimpleNamespace(model=model, observed=observed, stamp=stamp, sensor=sensor,
        goal=goal, prediction=prediction, trial=trial)


def _assert_record_payload(expected, message):
    actual = transform_distribution_from_msg(_wire(message))
    assert actual.parent_frame_id == expected.parent_frame_id
    assert actual.child_frame_id == expected.child_frame_id
    assert actual.stamp == pytest.approx(expected.stamp, abs=1e-9)
    assert actual.provenance.method == expected.provenance.method
    assert actual.provenance.detail == expected.provenance.detail
    assert len(actual.distribution.components) == len(expected.distribution.components)
    for source, published in zip(expected.distribution.components, actual.distribution.components):
        assert published.component_id == source.component_id
        assert published.raw_weight == source.raw_weight
        assert published.orientation.kind is source.orientation.kind
        np.testing.assert_array_equal(published.orientation.shape_matrix, source.orientation.shape_matrix)
        assert published.orientation.inverse_concentration == source.orientation.inverse_concentration
        np.testing.assert_array_equal(published.translation.mean_at_reference, source.translation.mean_at_reference)
        np.testing.assert_array_equal(published.translation.residual_covariance, source.translation.residual_covariance)
        np.testing.assert_array_equal(published.translation.rotation_coupling, source.translation.rotation_coupling)
        assert published.provenance.detail == source.provenance.detail
    return actual


def test_ros_wire_payload_keeps_the_objective_hand_goal_and_all_root_laws(case):
    snapshot, marker, goal = build_belief_messages(case.trial)
    snapshot = _wire(snapshot)
    assert snapshot.header.frame_id == "base_link"
    assert snapshot.header.stamp.to_sec() == pytest.approx(case.stamp, abs=1e-9)
    assert tuple(item.child_frame_id for item in snapshot.transforms) == BODY_FRAMES + ("switch_goal",)
    for native, message in zip(case.trial.belief_records, snapshot.transforms):
        _assert_record_payload(native, message)
    hand = _assert_record_payload(case.prediction.records[-1], snapshot.transforms[-2])
    component = hand.distribution.components[0]
    assert hand.provenance.method == COMPARISON_METHOD
    assert len(hand.distribution.components) == 1
    np.testing.assert_array_equal(component.translation.mean_at_reference, case.prediction.hand.mean)
    np.testing.assert_array_equal(component.translation.residual_covariance, case.prediction.hand.covariance)
    np.testing.assert_allclose(component.orientation.backend_parameter_matrix(),
                               case.prediction.hand.orientation_parameter, atol=1e-12, rtol=1e-14)
    hand_context = json.loads(hand.provenance.detail)
    assert hand_context["command"] == list(case.prediction.command)
    for key in ("source_id", "source_fingerprint", "revision", "stamp", "frame_id"):
        assert hand_context[key] == case.prediction.hand.metadata[key]
    target = _assert_record_payload(case.goal.record, goal)
    _assert_record_payload(case.goal.marker_record, marker)
    np.testing.assert_array_equal([item.raw_weight for item in target.distribution.components], case.goal.summary.weights)
    for item, orientation in zip(target.distribution.components, case.goal.summary.orientations):
        np.testing.assert_array_equal(item.translation.mean_at_reference, case.goal.summary.mean)
        np.testing.assert_array_equal(item.translation.residual_covariance, case.goal.summary.covariance)
        np.testing.assert_array_equal(item.orientation.shape_matrix, orientation.shape_matrix)
    context = json.loads(target.provenance.detail)
    assert context["observation_command"] == list(case.observed)
    assert context["observation_stamp"] == case.stamp


def test_goal_and_marker_payloads_stay_fixed_while_future_hand_command_changes(case):
    before = build_belief_messages(case.trial)
    command = case.prediction.command + [.03, -.02, .01, 0., .01, 0.]
    hand = case.model.hand(command, case.stamp)
    case.trial.belief_records = case.model.display_records(command, case.stamp, hand_summary=hand) + (case.goal.record,)
    after = build_belief_messages(case.trial)
    assert _bytes(before[1]) == _bytes(after[1])
    assert _bytes(before[2]) == _bytes(after[2])
    assert _bytes(before[0].transforms[-2]) != _bytes(after[0].transforms[-2])
    assert case.goal.objective.goal is case.goal.summary
    assert case.goal.objective.evaluate(hand)[0] != pytest.approx(case.prediction.score, abs=1e-8)


def test_actual_display_publishes_same_laws_despite_new_physical_pose_and_heartbeat(case, monkeypatch):
    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(_wire(message))

    display = Display.__new__(Display)
    display.trial = case.trial
    for name in ("joints", "rigid_joints", "markers", "predictions", "status", "probtf",
                 "marker_belief", "marker_display", "goal_belief"):
        setattr(display, name, Publisher())
    from visualization_msgs.msg import MarkerArray
    display.scene_markers = lambda: MarkerArray()
    display.prediction_markers = lambda: MarkerArray()
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(99.)))
    display.publish()
    first = display.probtf.messages[-1]
    case.trial.pose = np.array([2., -2., 1., -1., .5, .2])
    case.trial.command = case.observed + .4
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(199.)))
    display.publish()
    assert _bytes(first) == _bytes(display.probtf.messages[-1])
    assert display.joints.messages[-1].header.stamp.to_sec() == 199.
    assert display.probtf.messages[-1].header.stamp.to_sec() == pytest.approx(case.stamp, abs=1e-9)
    _assert_record_payload(case.goal.record, display.goal_belief.messages[-1])
    _assert_record_payload(case.goal.marker_record, display.marker_display.messages[-1])
    _assert_record_payload(case.sensor, display.marker_belief.messages[-1])


def test_initial_display_uses_same_model_source_without_an_observation_or_physical_pose(case):
    records = case.model.display_records(case.observed, case.stamp)
    initial = SimpleNamespace(belief_records=records, observation_preview=None)
    snapshot, marker, goal = build_belief_messages(initial)
    assert marker is None and goal is None
    assert tuple(item.child_frame_id for item in snapshot.transforms) == BODY_FRAMES
    for native, message in zip(records, snapshot.transforms):
        actual = _assert_record_payload(native, message)
        assert json.loads(actual.provenance.detail)["command"] == list(case.observed)
