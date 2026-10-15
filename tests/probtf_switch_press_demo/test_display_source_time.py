"""Publication is a frozen-law view; a heartbeat is never a new observation."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

rospy = pytest.importorskip("rospy")

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.provenance import TransformProvenance
from probtf.joint import JointTransformLaw
from probtf.candidates import IndependentTransformLaw
from probtf_switch_press_demo.beliefs import BODY_FRAMES, finite_pose_record
import probtf_switch_press_demo.ros_display as ros_display
from probtf_switch_press_demo.ros_display import Display


class Collector:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(deepcopy(message))


def test_repeated_display_heartbeats_preserve_all_four_source_laws(monkeypatch):
    source_stamp = 12.25
    metadata = dict(source_law_id="shared_body", source_revision=7,
                    query_id="command_4", order=2)
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[:, 0, 3] = [.1, .2]
    def body_record(parent, child):
        return finite_pose_record(parent, child, source_stamp, poses,
                                  ("node_0", "node_1"), (.3, .7), metadata=metadata)
    roots = tuple(body_record("base_link", "root_predicted/" + name) for name in BODY_FRAMES)
    hand = body_record("observation_hand", "predicted/rod_tip")
    component = TransformComponent("image_mode", 1., BinghamOrientation.dirac([1, 0, 0, 0]),
        ConditionalGaussianTranslation(np.array([.2, .1, .4]), np.eye(3) * .001, np.zeros((3, 9))))
    marker = TransformDistributionStamped("camera", "marker", source_stamp, "camera_marker", "test",
        TransformDistribution((component,)), provenance=TransformProvenance(
            method="native_detector", detail=json.dumps(dict(observation_id="pixels_4"))))
    goal = TransformDistributionStamped("observation_hand", "goal", source_stamp, "observed_goal", "test",
        marker.distribution, provenance=marker.provenance)
    prediction = SimpleNamespace(root_records=roots, hand_record=hand)
    trial = SimpleNamespace(pose=np.zeros(6), command=np.zeros(6), phase="tracking", cycle=0,
        ik_iteration=2, ik_pending=True, raw_image=None, hand_camera=np.eye(4),
        observation_preview=dict(marker_record=marker, goal_record=goal, hand=np.eye(4)),
        preview=dict(prediction=prediction, goal_record=goal, hand=np.eye(4)))
    display = Display.__new__(Display)
    law = object()
    trial.controller = SimpleNamespace(candidates=SimpleNamespace(law=law))
    display.descriptor_law = law
    display.trial = trial
    for name in ("joints", "rigid_joints", "markers", "predictions", "status", "marker_belief",
                 "goal_belief", "probtf", "marker_display", "hand_display", "goal_display"):
        setattr(display, name, Collector())
    display.scene_markers = lambda: None
    display.prediction_markers = lambda: None
    display.root_prediction = display.task_prediction = display.observed_record = None
    stamps = iter((100., 101.))
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(next(stamps))))
    display.publish()
    display.publish()
    assert [message.header.stamp.to_sec() for message in display.probtf.messages] == [100., 101.]
    for index in range(2):
        messages = list(display.probtf.messages[index].transforms)
        assert len(messages) == len(BODY_FRAMES)
        messages += [getattr(display, name).messages[index]
                     for name in ("hand_display", "marker_display", "goal_display")]
        for message in messages:
            assert message.header.stamp.to_sec() == source_stamp
            detail = json.loads(message.provenance.detail)
            assert detail["display_result_kind"] == "frozen_law_query"
            assert detail["source_observation_stamp"] == source_stamp
            assert detail["temporal_prediction"] is False
        root_metadata = json.loads(messages[0].provenance.detail)
        assert all(root_metadata[key] == value for key, value in metadata.items())
        assert json.loads(display.marker_display.messages[index].provenance.detail)["observation_id"] == "pixels_4"
    assert roots[0].stamp == hand.stamp == marker.stamp == goal.stamp == source_stamp
    assert "display_result_kind" not in json.loads(roots[0].provenance.detail)


def test_governing_body_law_is_latched_once_without_replacing_continuous_source(monkeypatch, tmp_path):
    published = {}
    def publisher(topic, message_type, **options):
        collector = Collector()
        collector.message_type, collector.options = message_type, options
        published[topic] = collector
        return collector
    monkeypatch.setattr(rospy, "Publisher", publisher)
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(100.)))
    monkeypatch.setattr(ros_display, "StaticTransformBroadcaster",
                        lambda: SimpleNamespace(sendTransform=lambda _message: None))
    monkeypatch.setattr(ros_display.cv2, "imread", lambda *_args: np.zeros((10, 10), dtype=np.uint8))
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([-4., -3., -2., 0.]))
    component = TransformComponent("free_pose", 1., orientation,
        ConditionalGaussianTranslation([.1, .2, .3], np.eye(3)*.001, np.zeros((3, 9))))
    edge = TransformDistributionStamped("base_link", "link_1", 0., "initial_link_1", "test",
                                        TransformDistribution((component,)))
    source = IndependentTransformLaw((edge,), True, "shared_body", 7, 2, 21, (-.15,))
    def generator(*_args):
        raise AssertionError("descriptor publication must not evaluate quadrature")
    law = JointTransformLaw(source, generator, source.frame_names, "model_sha256_123")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='test'><joint name='joint_1' type='revolute'>"
                    "<child link='link_1'/><axis xyz='0 0 1'/></joint></robot>")
    trial = SimpleNamespace(urdf=urdf, config=dict(base_translation=[0., 0., 0.],
        rod=dict(length=.325, radius=.008)), controller=SimpleNamespace(candidates=SimpleNamespace(law=law)),
        pose=np.zeros(6), command=np.zeros(6), phase="initial", cycle=0, ik_iteration=0,
        ik_pending=False, raw_image=None, observation_preview=None, preview=None)
    display = Display(trial, tmp_path / "tag.png")
    display.scene_markers = lambda: None
    display.prediction_markers = lambda: None
    display.publish()
    display.publish()
    publisher = published["/switch_press/body_law"]
    assert publisher.options == dict(queue_size=1, latch=True)
    assert len(publisher.messages) == 1
    descriptor = json.loads(publisher.messages[0].data)
    assert descriptor["source_law"] == json.loads(json.dumps(source.descriptor()))
    assert descriptor["source_law"]["kind"] == "independent_native_transforms"
    assert descriptor["source_law"]["law_id"] == "shared_body"
    assert descriptor["source_law"]["revision"] == 7
    source_component = descriptor["source_law"]["edges"][0]["components"][0]
    assert source_component["orientation"]["kind"] == "finite_bingham"
    np.testing.assert_allclose(source_component["orientation"]["parameter_matrix"],
                               orientation.backend_parameter_matrix())
    assert "node_ids" not in descriptor["source_law"]
    assert "initial_local_transforms" not in descriptor["source_law"]
    assert descriptor["model_id"] == law.model_id
    assert descriptor["frame_names"] == list(source.frame_names)
    generator_info = descriptor["generator_definition"]
    assert generator_info["urdf"]["sha256"] == hashlib.sha256(urdf.read_bytes()).hexdigest()
    assert generator_info["rod"]["tip_translation_m"] == [.325, 0., 0.]
    assert generator_info["reference_command"] == [-.15]
    assert generator_info["joint_axes"] == [dict(joint="joint_1", child="link_1", axis=[0., 0., 1.])]
    assert generator_info["model"] == "configured_local_transform_laws_with_known_joint_right_action"
    assert "equilibrium" not in generator_info
    assert generator_info["unresolved_generator_status"] == "UNAVAILABLE"
    assert "stamp" not in descriptor
    # Replay may reload an edited native law. Re-publish its full descriptor
    # exactly once, with its own source revision and unchanged source stamp.
    updated_source = replace(source, revision=8)
    trial.controller.candidates.law = JointTransformLaw(
        updated_source, generator, source.frame_names, law.model_id)
    display.publish()
    display.publish()
    assert len(publisher.messages) == 2
    updated = json.loads(publisher.messages[-1].data)
    assert updated["source_law"]["revision"] == 8
    assert updated["source_law"]["edges"][0]["stamp"] == 0.
