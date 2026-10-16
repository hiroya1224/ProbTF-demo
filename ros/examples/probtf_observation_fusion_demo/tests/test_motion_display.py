"""Physical references use their recorded frame, including on replay hold."""

from types import SimpleNamespace

import cv2
import numpy as np
import rospy
from scipy.spatial.transform import Rotation

from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


class Capture:
    def __init__(self):
        self.messages = None

    def sendTransform(self, messages):
        self.messages = messages

    def publish(self, message):
        self.messages = message


def _pose(message):
    result = np.eye(4)
    value = message.transform
    result[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    result[:3, :3] = Rotation.from_quat([
        value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w]).as_matrix()
    return result


def test_replay_robot_tf_and_physical_path_do_not_use_future_frame(tmp_path, monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    first = scene.step(0.0, 1792080000.0)
    second = scene.step(1.0, 1792080001.0, command=scene.initial_command + .02)
    # Preparing the whole movie leaves FK at a later physical state.  Replaying
    # the first frame must still draw its original physical tip and no future path.
    scene.monitor.reached = True
    display = MotionDisplay.__new__(MotionDisplay)
    display.scene, display.case, display.sensor_names = scene, "no_sensors", ()
    display.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
    display.transforms, display.publisher = Capture(), Capture()
    display.marker_id = 0
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(1234.5)))
    snapshot = SimpleNamespace(observation_ids=(), stamp=first.stamp)
    display.publish(first, 0, snapshot, 0)

    sent = {message.child_frame_id: message for message in display.transforms.messages}
    for prefix, poses in (("", first.link_poses), ("rigid/", first.rigid_link_poses)):
        for name, expected in poses.items():
            if name == "base_link":
                continue
            message = sent[prefix + name]
            assert message.header.frame_id == "base_link"
            np.testing.assert_allclose(_pose(message), expected, atol=1e-14)
    assert not np.allclose(first.link_poses["rod_tip"], second.link_poses["rod_tip"])
    assert not np.allclose(first.link_poses["rod_tip"], first.rigid_link_poses["rod_tip"])
    markers = display.publisher.messages.markers
    assert not any(marker.ns == "actual_path" for marker in markers)
    actual_tip = next(marker for marker in markers if marker.ns == "actual_tip")
    np.testing.assert_allclose([actual_tip.pose.position.x, actual_tip.pose.position.y,
                               actual_tip.pose.position.z], first.tip, atol=1e-14)
    # The final monitor cannot turn the lamp green in an earlier movie frame.
    lamp = next(marker for marker in markers if marker.ns == "scene" and marker.id == 2)
    assert lamp.color.g < 0.2
    assert not any(marker.ns.startswith(("gaussian/", "axes/")) for marker in markers)

    display.publish(second, 1, SimpleNamespace(observation_ids=(), stamp=second.stamp), 1)
    path = next(marker for marker in display.publisher.messages.markers if marker.ns == "actual_path")
    np.testing.assert_allclose([[point.x, point.y, point.z] for point in path.points],
                               [first.tip, second.tip], atol=1e-14)
