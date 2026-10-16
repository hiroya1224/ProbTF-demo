"""Motion may not reuse a distribution from an earlier sent command/time."""

from dataclasses import replace
import io
import json
from pathlib import Path

import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo import motion_node
from probtf_observation_fusion_demo.display import verify_wire_identity
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


CONFIG = Path(__file__).resolve().parents[1] / 'config'


@pytest.fixture(scope='module')
def motion_updates(tmp_path_factory):
    """Use the actual physics, direct prior, core queries and ROS serialization."""
    output = tmp_path_factory.mktemp('same_time_motion') / 'run'
    original_scene = motion_node.MovingSwitchScene

    class ShortMotion(MovingSwitchScene):
        def __init__(self, output_dir):
            super().__init__(output_dir)
            # Three controller-selected commands suffice to expose the former
            # 0.5 s stale-estimate hold without replaying the whole scene.
            self.duration = .1

    motion_node.MovingSwitchScene = ShortMotion
    try:
        prepared = motion_node.prepare_motion(CONFIG / 'imu.yaml', output)
    finally:
        motion_node.MovingSwitchScene = original_scene
    return output, prepared


def test_every_physical_tick_queries_its_own_command_and_core_snapshot(motion_updates):
    output, (_, frames, updates, mapping, manifest) = motion_updates
    assert len(frames) == len(updates) == 3
    assert mapping == [0, 1, 2]
    assert manifest['inference_period_s'] == manifest['frame_period_s'] == .05
    assert manifest['physical_estimate_time_offset_s'] == 0.
    assert manifest['sensors_drive_commands'] is True
    assert manifest['same_motion_sensor_estimation_comparison'] is False
    versions = []
    records = [json.loads(line) for line in (output / 'estimation.jsonl').read_text().splitlines()]
    for index, (frame, update, record) in enumerate(zip(frames, updates, records)):
        snapshot, _, image = motion_node.read_update(update)
        motion_node.require_current_update(frame, snapshot, image, index, mapping[index])
        assert snapshot.command == tuple(frame.command)
        assert snapshot.stamp == frame.stamp
        assert record['moving_input_family']['frame_index'] == index
        assert record['moving_input_family']['time'] == frame.time
        published = json.loads((output / 'estimate_{:03d}.json'.format(index)).read_text())
        verify_wire_identity(snapshot, motion_node.read_update(update)[1],
                             seed=published['render_seed'], point_size=published['point_size'])
        assert published['wire_identity_verified'] is True
        assert published['command'] == frame.command.tolist()
        assert published['stamp'] == frame.stamp
        assert snapshot.observation_ids == ('imu_link_3:frame:{:05d}'.format(index),)
        for point in snapshot.points:
            assert point.query.command == tuple(frame.command)
            assert point.query.stamp == frame.stamp
        versions.append(snapshot.version)
    assert np.all(np.diff(versions) > 0)
    assert updates[0][0].command != updates[1][0].command
    sent = json.loads((output / 'sent_commands.json').read_text())
    for index, record in enumerate(records[:-1]):
        assert record['controller']['next_command'] == sent[index + 1]['command']


def test_playback_rejects_stale_command_or_observation_before_publishing(motion_updates):
    _, (_, frames, updates, _, manifest) = motion_updates
    first, later = frames[:2]
    snapshot, _, image = motion_node.read_update(updates[0])
    # The previous implementation accepted this exact combination while the
    # physical robot advanced through the following nine ticks.
    with pytest.raises(AssertionError, match='own inference'):
        motion_node.require_current_update(later, snapshot, image, 1, 0)
    with pytest.raises(AssertionError, match='not simultaneous'):
        motion_node.require_current_update(later, snapshot, image, 1, 1)
    with pytest.raises(AssertionError, match='observation'):
        motion_node.require_current_update(
            first, replace(snapshot, observation_ids=('imu:frame:00001',)), image, 0, 0)
    # Constructor validation precedes ROS publishers/timers and must reject an
    # old two-rate replay map even before any frames are sent.
    with pytest.raises(AssertionError, match='one same-time'):
        motion_node.MotionPlayback(None, frames, updates, [0, 0, 2], manifest, 'test', False)


def test_frame_matching_never_recentres_distribution_on_physical_truth(motion_updates):
    _, (_, frames, updates, _, _) = motion_updates
    frame = frames[1]
    snapshot, _, image = motion_node.read_update(updates[1])
    altered_truth = replace(frame, joints=frame.joints + 1.,
        tip=frame.tip + 10., link_poses={}, rigid_link_poses={})
    # Only explicitly shared time/command metadata defines correspondence.
    # Neither a drawing pose nor a realized joint angle may recenter queries.
    motion_node.require_current_update(altered_truth, snapshot, image, 1, 1)
    assert any(np.linalg.norm(point.query.mean - altered_truth.tip) > 1.
               for point in snapshot.points if point.frame_label == 'hand')


def test_playback_forwards_exact_wire_and_catches_up_without_dropping_ticks(
        motion_updates, monkeypatch):
    """A late callback must publish every due same-time source frame."""
    output, (scene, frames, updates, mapping, manifest) = motion_updates
    publishers = {}
    displayed = []
    clock = [100.0]

    class Publisher:
        def __init__(self, topic, data_class, **settings):
            self.data_class, self.settings = data_class, settings
            self.payloads = []
            publishers[topic] = self

        def publish(self, message):
            stream = io.BytesIO()
            message.serialize(stream)
            self.payloads.append(stream.getvalue())

    class Display:
        def __init__(self, *_args):
            pass

        def publish(self, frame, index, snapshot, update_index):
            assert frame.stamp == snapshot.stamp
            assert tuple(frame.command) == snapshot.command
            displayed.append((index, update_index))

    monkeypatch.setattr(motion_node.rospy, 'Publisher', Publisher)
    monkeypatch.setattr(motion_node.rospy, 'Service', lambda *_args: None)
    monkeypatch.setattr(motion_node.rospy, 'Timer', lambda *_args: None)
    monkeypatch.setattr(motion_node.rospy, 'on_shutdown', lambda *_args: None)
    monkeypatch.setattr(motion_node, 'MotionDisplay', Display)
    monkeypatch.setattr(motion_node.time, 'monotonic', lambda: clock[0])
    def forbidden(_update):
        raise AssertionError('Playback must forward the verified raw marker bytes.')
    monkeypatch.setattr(motion_node, 'read_update', forbidden)
    playback = motion_node.MotionPlayback(
        scene, frames, updates, mapping, manifest, 'imu', False)
    try:
        playback.started = 100.0
        clock[0] = 100.12  # The old wall-index selection skipped frame 1.
        playback.advance(None)
        assert displayed == [(0, 0), (1, 1), (2, 2)]
        wire = publishers['/observation_fusion/distributions']
        assert wire.data_class is MarkerArray
        assert wire.settings['queue_size'] >= 3
        assert wire.payloads == [update[1].read_bytes() for update in updates]
        snapshots = publishers['/observation_fusion/snapshot'].payloads
        images = publishers['/observation_fusion/detection_image'].payloads
        motion = publishers['/observation_fusion/motion_metadata'].payloads
        for index, (snapshot_bytes, image_bytes, motion_bytes) in enumerate(
                zip(snapshots, images, motion)):
            snapshot = json.loads(String().deserialize(snapshot_bytes).data)
            actual_image = Image().deserialize(image_bytes)
            actual_motion = json.loads(String().deserialize(motion_bytes).data)
            saved = json.loads((output/'estimate_{:03d}.json'.format(index)).read_text())
            for extra in ('render_seed', 'point_size', 'display_roundoff_tolerance',
                          'wire_identity_verified'):
                saved.pop(extra)
            assert snapshot == saved
            assert actual_motion['sample_index'] == actual_motion['estimate_index'] == index
            assert actual_motion['estimate_stamp'] == actual_motion['physical_stamp'] == snapshot['stamp']
            assert actual_motion['command'] == snapshot['command']
            assert actual_image.header.stamp.to_sec() == snapshot['stamp']
            expected_image = motion_node.read_image(updates[index][0], updates[index][2])
            assert actual_image.data == expected_image.data
            assert all(query['source_stamp'] == snapshot['stamp']
                       and query['source_command'] == snapshot['command']
                       for query in snapshot['queries'])
        # Replay retains the same wire and pixels, and its lookahead remains
        # bounded instead of decoding every full-resolution frame at once.
        clock[0] = 101.0
        playback.replay(None)
        assert wire.payloads[-1] == updates[0][1].read_bytes()
        assert len(playback.image_futures) <= 2
    finally:
        playback.stop_prefetch()
