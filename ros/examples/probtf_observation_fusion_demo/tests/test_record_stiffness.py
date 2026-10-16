"""The video recorder must reject unrelated stiffness publications or HUDs."""

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.display import snapshot_record
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo import motion_node
from test_motion_update import motion_updates


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stiffness_video_recorder',
                                            PACKAGE/'scripts/record_four_cases.py')
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)


def _string_round_trip(data):
    stream = io.BytesIO()
    String(json.dumps(data)).serialize(stream)
    return json.loads(String().deserialize(stream.getvalue()).data)


@pytest.fixture(scope='module')
def subscribed_stiffness_run(motion_updates, tmp_path_factory):
    output, (_, frames, updates, _, manifest) = motion_updates
    run = tmp_path_factory.mktemp('received_stiffness')
    (run/'node').symlink_to(output, target_is_directory=True)
    snapshots, motion, wire, images, stiffness, hud = [], [], [], [], [], []
    replay_wall = 10.0
    for index, (frame, update) in enumerate(zip(frames, updates)):
        snapshot, markers, image = motion_node.read_update(update)
        record = snapshot_record(snapshot)
        wall = replay_wall+.05*index
        snapshots.append({'wall_elapsed_s': wall, 'data': _string_round_trip(record)})
        stiffness.append({'wall_elapsed_s': wall,
                          'data': _string_round_trip(snapshot.stiffness_record)})
        stream = io.BytesIO()
        markers.serialize(stream)
        wire.append({'wall_elapsed_s': wall,
                     'sha256': hashlib.sha256(stream.getvalue()).hexdigest()})
        images.append({'wall_elapsed_s': wall, 'stamp': snapshot.stamp,
                       'sha256': hashlib.sha256(image.data).hexdigest()})
        marker = MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)
        stream = io.BytesIO()
        MarkerArray([marker]).serialize(stream)
        actual = MarkerArray().deserialize(stream.getvalue()).markers[0]
        hud.append({'stamp': actual.header.stamp.to_sec(), 'text': actual.text})
        motion.append({'wall_elapsed_s': wall,
                       'data': dict(manifest, sample_index=index,
                                    scene_elapsed=frame.time,
                                    finished=index == len(frames)-1)})
    return run, snapshots, motion, wire, images, replay_wall, stiffness, hud


def test_recorder_accepts_actual_saved_same_snapshot_stiffness_and_hud(subscribed_stiffness_run):
    accepted = RECORDER.verify_received(*subscribed_stiffness_run)
    assert accepted['published_stiffness_identity_verified']
    assert accepted['rviz_stiffness_hud_identity_verified']
    assert accepted['stiffness_updates_without_failure']
    assert accepted['unique_snapshots'] == accepted['unique_images'] == 3
    assert accepted['actual_moving_image_fps'] >= 19.9


def test_recorder_rejects_changed_stiffness_parameter_on_actual_topic(subscribed_stiffness_run):
    arguments = list(subscribed_stiffness_run)
    arguments[6] = deepcopy(arguments[6])
    arguments[6][1]['data']['posterior']['K_median'][0] += 1.
    with pytest.raises(AssertionError, match='Published stiffness differs'):
        RECORDER.verify_received(*arguments)


def test_recorder_rejects_missing_stiffness_update(subscribed_stiffness_run):
    arguments = list(subscribed_stiffness_run)
    arguments[6] = arguments[6][:-1]
    with pytest.raises(AssertionError, match='Published stiffness differs'):
        RECORDER.verify_received(*arguments)


@pytest.mark.parametrize('change', ['text', 'stamp', 'missing'])
def test_recorder_rejects_wrong_or_missing_actual_rviz_stiffness_hud(subscribed_stiffness_run, change):
    arguments = list(subscribed_stiffness_run)
    arguments[7] = deepcopy(arguments[7])
    if change == 'text':
        arguments[7][1]['text'] += '\nUnrelated display values'
    elif change == 'stamp':
        arguments[7][1]['stamp'] += .001
    else:
        arguments[7].pop()
    with pytest.raises(AssertionError, match='Actual RViz stiffness text differs'):
        RECORDER.verify_received(*arguments)
