"""The actual executed search intervals alone have orange reference trails."""

from copy import deepcopy
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_switch_press_demo.ros_display import Display as SwitchDisplay


TEAL = (.02, .65, .75, 1.)
ORANGE = (1., .45, 0., 1.)


class Capture:
    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message

    def sendTransform(self, message):
        self.message = message


def _wire(message):
    stream = io.BytesIO()
    message.serialize(stream)
    return MarkerArray().deserialize(stream.getvalue())


def _coordinates(points):
    return np.array([[point.x, point.y, point.z] for point in points])


def _rgba(colors):
    return np.array([[color.r, color.g, color.b, color.a] for color in colors])


def _path(message):
    return next(marker for marker in message.markers if marker.ns == 'actual_path')


def _publish(display, frames, index):
    frame = frames[index]
    display.publish(frame, index, SimpleNamespace(observation_ids=(), stamp=frame.stamp), index)
    return _wire(display.publisher.message)


@pytest.fixture
def recorded_search(tmp_path, monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    # action_mode is chosen after this image; command_interval_mode belongs
    # to the motion that arrived here. The reacquisition row intentionally
    # differs, to catch recoloring from the next requested action.
    history = (
        ('reaching', 'reaching'),
        ('braking', 'reaching'),
        ('settling', 'braking'),
        ('searching', 'settling'),
        ('searching', 'searching'),
        ('settling', 'searching'),
        ('settling', 'settling'),
        ('searching', 'settling'),
        ('searching', 'searching'),
        ('reaching', 'searching'),
        ('reaching', 'reaching'),
    )
    frames = []
    for index, (action, interval) in enumerate(history):
        command = scene.initial_command + .001*index*np.array([1., -.5, .2, 0., 0., 0.])
        frame = scene.step(index*.05, 1792080000.+index*.05, command=command)
        scene.trajectory[-1].update(action_mode=action, command_interval_mode=interval,
            search_amplitude_rad=.02 if index < 7 else .03,
            search_completed_roundtrips=0 if index < 5 else 1)
        frames.append(frame)
    display = MotionDisplay.__new__(MotionDisplay)
    display.scene, display.case, display.sensor_names = scene, 'marker', ('marker',)
    display.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
    display.transforms, display.publisher = Capture(), Capture()
    display.marker_id = 0
    monkeypatch.setattr(rospy.Time, 'now', staticmethod(lambda: rospy.Time.from_sec(8000.)))
    return scene, display, frames


def test_actual_ros_wire_has_exact_executed_search_segments_and_no_boundary_bleed(recorded_search):
    scene, display, frames = recorded_search
    original = deepcopy(scene.trajectory)
    output = _publish(display, frames, len(frames)-1)
    path = _path(output)
    assert path.type == Marker.LINE_LIST
    assert path.action == Marker.ADD
    assert path.header.frame_id == 'base_link'
    assert path.header.stamp == rospy.Time.from_sec(8000.)
    np.testing.assert_allclose([path.scale.x, path.scale.y, path.scale.z], [.003, 0., 0.], atol=0.)
    assert len(path.points) == len(path.colors) == 2*(len(frames)-1)
    for index, row in enumerate(scene.trajectory[1:]):
        np.testing.assert_array_equal(_coordinates(path.points[index*2:index*2+2]),
                                      [scene.trajectory[index]['tip'], row['tip']])
        expected = ORANGE if row['command_interval_mode'] == 'searching' else TEAL
        # ROS ColorRGBA uses float32, so compare the declared values in that
        # same wire representation. Each segment has one color at both ends.
        np.testing.assert_array_equal(_rgba(path.colors[index*2:index*2+2]),
                                      np.array([expected, expected], dtype=np.float32))
    colors = _rgba(path.colors).reshape(-1, 2, 4)
    np.testing.assert_array_equal(colors[:, 0], colors[:, 1])
    assert scene.trajectory == original

    # The physical board replaces the old tag artwork. All other shared
    # switch geometry and sizes remain identical.
    baseline = _wire(SwitchDisplay.scene_markers(display))
    base = _path(baseline)
    assert (path.ns, path.id, path.pose, path.scale) == (base.ns, base.id, base.pose, base.scale)
    actual_by_key = {(marker.ns, marker.id): marker for marker in output.markers}
    for expected in baseline.markers:
        if expected.ns not in ('actual_path', 'status', 'tag'):
            assert expected == actual_by_key[(expected.ns, expected.id)]


@pytest.mark.parametrize('has_plan_source', [False, True])
def test_reached_hold_status_and_executed_path_use_the_same_physical_record(recorded_search, has_plan_source):
    scene, display, frames = recorded_search
    source = (dict(source_stamp=frames[-1].stamp, source_command=frames[-1].command.tolist(),
        source_version=2, body_source_state_ids=['retained_law'], goal_source_state_ids=['retained_law'])
        if has_plan_source else None)
    scene.trajectory[-1].update(action_mode='reached_hold', command_interval_mode='reached_hold',
        reaching_phase='reached_hold', reaching_phase_source=source)
    message = _publish(display, frames, len(frames)-1)
    status = next(marker for marker in message.markers if marker.ns == 'status')
    assert 'Phase: REACHED | COMMAND HOLD' in status.text
    assert 'Action: REACHED | COMMAND HOLD' in status.text
    assert scene.trajectory[-1]['reaching_phase_source'] == source
    path = _path(message)
    np.testing.assert_array_equal(_rgba(path.colors[-2:]), np.array([TEAL]*2, dtype=np.float32))


def test_history_colors_do_not_follow_current_or_future_actions_and_replay_has_no_future_points(recorded_search):
    scene, display, frames = recorded_search
    searching = _path(_publish(display, frames, 4))
    settled = _path(_publish(display, frames, 6))
    expanded = _path(_publish(display, frames, 8))
    reacquired = _path(_publish(display, frames, 9))
    np.testing.assert_array_equal(_coordinates(reacquired.points[:len(searching.points)]),
                                  _coordinates(searching.points))
    np.testing.assert_array_equal(_rgba(reacquired.colors[:len(searching.colors)]), _rgba(searching.colors))
    np.testing.assert_array_equal(_rgba(expanded.colors[:len(settled.colors)]), _rgba(settled.colors))
    np.testing.assert_array_equal(_rgba(reacquired.colors[:len(expanded.colors)]), _rgba(expanded.colors))
    np.testing.assert_array_equal(_rgba(settled.colors[-2:]), np.array([TEAL]*2, dtype=np.float32))
    np.testing.assert_array_equal(_rgba(expanded.colors[-2:]), np.array([ORANGE]*2, dtype=np.float32))
    np.testing.assert_array_equal(_rgba(reacquired.colors[-2:]), np.array([ORANGE]*2, dtype=np.float32))
    # Replaying an earlier frame after the complete history was prepared must
    # show only the intervals that had actually executed by that frame.
    earlier = _path(_publish(display, frames, 2))
    assert len(earlier.points) == 4
    np.testing.assert_array_equal(_coordinates(earlier.points),
                                  [scene.trajectory[0]['tip'], scene.trajectory[1]['tip'],
                                   scene.trajectory[1]['tip'], scene.trajectory[2]['tip']])
    np.testing.assert_array_equal(_rgba(earlier.colors), np.array([TEAL]*4, dtype=np.float32))


def test_action_hud_and_replay_reset_use_same_frame_without_retained_orange_strokes(recorded_search):
    scene, display, frames = recorded_search
    output = _publish(display, frames, 4)
    assert next(marker for marker in output.markers if marker.ns == 'status').text.endswith('Action: MARKER SEARCH')
    output = _publish(display, frames, 6)
    assert next(marker for marker in output.markers if marker.ns == 'status').text.endswith('Action: MARKER SEARCH | SETTLING')
    output = _publish(display, frames, 8)
    assert next(marker for marker in output.markers if marker.ns == 'status').text.endswith('Action: MARKER SEARCH')
    output = _publish(display, frames, 9)
    assert next(marker for marker in output.markers if marker.ns == 'status').text.endswith('Action: REACHING')
    assert any(color.r == 1. for color in _path(output).colors)
    # DELETEALL remains first on every publication, including reset/first
    # frame where no actual_path marker is emitted at all.
    scene.reset()
    first = scene.step(0., 1792080000.)
    output = _publish(display, [first], 0)
    assert output.markers[0].action == Marker.DELETEALL
    assert not any(marker.ns == 'actual_path' for marker in output.markers)
    assert not any(marker.ns == 'status' and 'Action:' in marker.text for marker in output.markers)


def test_legacy_history_is_teal_and_invalid_interval_metadata_is_rejected(recorded_search):
    scene, display, frames = recorded_search
    for row in scene.trajectory:
        row.pop('action_mode')
        row.pop('command_interval_mode')
    output = _publish(display, frames, 2)
    np.testing.assert_array_equal(_rgba(_path(output).colors), np.array([TEAL]*4, dtype=np.float32))
    scene.trajectory[1]['command_interval_mode'] = 'unrecorded-search'
    with pytest.raises(ValueError, match='Unknown physical path command interval mode'):
        _publish(display, frames, 2)


def test_path_color_adapter_rejects_points_from_another_trajectory(recorded_search):
    scene, display, frames = recorded_search
    path = _path(_publish(display, frames, 2))
    with pytest.raises(AssertionError, match='points and recorded command intervals differ'):
        MotionDisplay._color_action_path(path, scene.trajectory[:2])


def test_finished_experiment_labels_the_last_search_action_and_preserves_its_executed_orange_history(recorded_search):
    _, display, frames = recorded_search
    active = _path(_publish(display, frames, 8))
    frame = frames[8]
    display.publish(frame, 8, SimpleNamespace(observation_ids=(), stamp=frame.stamp), 8,
        execution={'execution_mode': 'live', 'actual_update_hz': 1.3,
                   'simulated_step_s': .05, 'experiment_finished': True})
    finished = _wire(display.publisher.message)
    status = next(marker for marker in finished.markers if marker.ns == 'status').text
    assert 'FINISHED' in status
    assert status.endswith('Last action: MARKER SEARCH')
    path = _path(finished)
    np.testing.assert_array_equal(_coordinates(path.points), _coordinates(active.points))
    np.testing.assert_array_equal(_rgba(path.colors), _rgba(active.colors))
    assert any(color.r == 1. for color in path.colors)
