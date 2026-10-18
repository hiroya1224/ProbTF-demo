"""Diagnostic truth remains acquisition-aligned and changes one central law."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from probtf.geometry import quat_to_rotmat, rotation_vector_to_quaternion
from probtf_observation_fusion_demo import sensors, moment_observation
from uncertainty_ablation_observations import (
    install_oracle_marker_patch, prepare_oracle_marker_observation,
)


COMMON = dict(camera=dict(matrix=[[600., 0., 480.], [0., 600., 360.], [0., 0., 1.]],
                          distortion=[], width=960, height=720))


def pose(translation, rotation_vector):
    value = np.eye(4)
    value[:3, 3] = translation
    value[:3, :3] = quat_to_rotmat(rotation_vector_to_quaternion(rotation_vector))
    return value


def capture(stamp=1003.25):
    return dict(event_id='image:00000123', stamp=stamp,
        command=[.1, -.2, .3, -.4, .5, -.6],
        camera_pose=pose([.3, .2, .7], [.2, -.3, .1]).tolist(),
        marker_pose=pose([1.1, -.1, .8], [1.4, .1, -1.5]).tolist())


def prepared_fixture(image, data=None):
    data = capture() if data is None else data
    return prepare_oracle_marker_observation(COMMON, image, data['stamp'],
        camera_pose=data['camera_pose'], marker_pose=data['marker_pose'],
        observation_id=data['event_id'], command=data['command'],
        capture_stamp=data['stamp'])


def test_oracle_is_exact_captured_camera_relative_mode_with_native_covariance():
    image = np.zeros((4, 5), dtype=np.uint8)
    data = capture()
    prepared = prepared_fixture(image, data)
    expected = np.linalg.solve(data['camera_pose'], data['marker_pose'])
    assert prepared.image is image
    assert prepared.stamp == data['stamp']
    assert len(prepared.detected.record.distribution.components) == 1
    component = prepared.detected.record.distribution.components[0]
    np.testing.assert_allclose(component.translation.mean_at_reference, expected[:3, 3], atol=1e-15)
    np.testing.assert_allclose(quat_to_rotmat(component.orientation.mode_wxyz), expected[:3, :3], atol=1e-14)
    law, _ = sensors._marker_mode_chart(component, expected[:3, :3])
    np.testing.assert_allclose(law.covariance, np.diag([1e-8]*3+[1e-6]*3), rtol=1e-9, atol=1e-14)
    assert prepared.detected.log_masses == (0.,)
    assert prepared.detected.weights == (1.,)
    assert prepared.detected.seed_indices == (0,)
    assert prepared.board_fusion['original_image_observation_id'] == data['event_id']
    assert prepared.board_fusion['source_command'] == data['command']
    assert prepared.board_fusion['physical_truth_used_for_inference'] is True
    assert prepared.board_fusion['central_image_observation_bypassed'] is True


def test_oracle_uses_no_random_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('The diagnostic law must not sample candidates.')
    for name in ('default_rng', 'normal', 'random', 'multivariate_normal'):
        monkeypatch.setattr(np.random, name, forbidden)
    prepared_fixture(np.zeros((3, 4), dtype=np.uint8))


def test_oracle_rejects_retimed_truth():
    data = capture()
    with pytest.raises(ValueError, match='same original acquisition stamp'):
        prepare_oracle_marker_observation(COMMON, np.zeros((3, 4)), data['stamp']+.1,
            camera_pose=data['camera_pose'], marker_pose=data['marker_pose'],
            observation_id=data['event_id'], command=data['command'], capture_stamp=data['stamp'])


def test_process_patch_retains_missingness_and_restores_all_imports(tmp_path, monkeypatch):
    data = capture()
    truth = tmp_path/'diagnostic_marker_truth'
    truth.mkdir()
    (truth/(data['event_id']+'.json')).write_text(json.dumps(data))
    image = np.zeros((3, 4), dtype=np.uint8)
    base = replace(prepared_fixture(image), detected=None,
                   board_fusion=dict(available=False, reason='baseline detector missing'))
    calls = []
    def baseline(common, pixels, stamp):
        calls.append((pixels, stamp))
        return base
    monkeypatch.setattr(sensors, 'prepare_marker_observation', baseline)
    old_moment = moment_observation.prepare_marker_observation
    restore = install_oracle_marker_patch(tmp_path)
    try:
        result = moment_observation.prepare_marker_observation(COMMON, image, data['stamp'])
        assert result.detected is None
        assert result.board_fusion['intervention_applied'] is False
        assert result.board_fusion['baseline_detection_required'] is True
        assert len(calls) == 1
    finally:
        restore()
    assert sensors.prepare_marker_observation is baseline
    assert moment_observation.prepare_marker_observation is old_moment


@pytest.mark.parametrize('require_detected', [True, False])
def test_process_patch_reaches_fixed_native_moment_observation(tmp_path, monkeypatch, require_detected):
    data = capture()
    truth = tmp_path/'diagnostic_marker_truth'
    truth.mkdir()
    (truth/(data['event_id']+'.json')).write_text(json.dumps(data))
    image = np.zeros((3, 4), dtype=np.uint8)
    base = prepared_fixture(image)
    if not require_detected:
        base = replace(base, detected=None)
    monkeypatch.setattr(sensors, 'prepare_marker_observation', lambda *_: base)
    restore = install_oracle_marker_patch(tmp_path, require_detected=require_detected)
    try:
        result = moment_observation.prepare_observation(COMMON,
            dict(sensor='marker', sensor_id='marker_camera', observation_id=data['event_id']),
            dict(camera_marker=(('camera_inverse', True), ('marker', False))), image, data['stamp'])
        assert len(result.components) == 1
        native = result.components[0]
        assert native.observation_id == data['event_id']
        expected = np.linalg.solve(data['camera_pose'], data['marker_pose'])
        np.testing.assert_allclose(native.observed_mean[:3], expected[:3, 3], atol=1e-14)
        np.testing.assert_allclose(native.observed_mean[3:], 0., atol=1e-14)
        np.testing.assert_allclose(native.covariance, np.diag([1e-8]*3+[1e-6]*3), rtol=1e-9, atol=1e-14)
        board = result.prepared_marker.board_fusion
        assert board['intervention_applied'] is True
        assert board['baseline_detection_required'] is require_detected
        assert board['baseline_marker_available'] is require_detected
    finally:
        restore()


def test_process_patch_refuses_nearest_available_truth(tmp_path, monkeypatch):
    data = capture()
    truth = tmp_path/'diagnostic_marker_truth'
    truth.mkdir()
    (truth/(data['event_id']+'.json')).write_text(json.dumps(data))
    image = np.zeros((3, 4), dtype=np.uint8)
    base = prepared_fixture(image)
    monkeypatch.setattr(sensors, 'prepare_marker_observation', lambda *_: base)
    restore = install_oracle_marker_patch(tmp_path)
    try:
        with pytest.raises(ValueError, match='No exact original acquisition truth'):
            moment_observation.prepare_marker_observation(COMMON, image, data['stamp']+.01)
    finally:
        restore()


def test_oracle_record_passes_real_fresh_goal_construction(tmp_path):
    from probtf_observation_fusion_demo.fresh_goal import fresh_marker_goals
    from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
    from probtf_observation_fusion_demo.moving_inference import MovingInference

    scene = MovingSwitchScene(tmp_path/'scene')
    frame = scene.step(0., stamp=1792080000., command=scene.initial_command,
                       render_camera=False)
    inference = MovingInference(scene.robot_description, ())
    result = inference.update(scene.sensor_inputs(frame), 0)
    canonical = result.stages[-1].components[0].state
    image = np.zeros((3, 4), dtype=np.uint8)
    prepared = prepare_oracle_marker_observation(inference.common, image, frame.stamp,
        camera_pose=frame.camera_pose, marker_pose=scene.marker,
        observation_id='image:00000000', command=frame.command,
        capture_stamp=frame.stamp)
    posterior = SimpleNamespace(observation_id='image:00000000', sensor_id='camera',
        query=SimpleNamespace(stamp=canonical.stamp, command=canonical.command,
                              state_id=canonical.state_id))
    estimate = sensors.LocalEstimate(posterior, {}, detector_result=prepared.detected,
                                     board_fusion=prepared.board_fusion)
    supplied = replace(result, local_estimates=(estimate,))
    goals, record = fresh_marker_goals(supplied, canonical)
    assert len(goals) == 1
    assert record['applied'] is True
    assert record['raw_marker_count'] == 0
    assert record['native_component_count'] == 1
    assert record['marker_board_fusion']['physical_truth_received'] is True
    assert record['marker_board_fusion']['central_distribution']['stamp'] == frame.stamp
