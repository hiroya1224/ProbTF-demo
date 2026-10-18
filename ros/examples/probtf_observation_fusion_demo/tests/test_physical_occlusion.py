"""Static physical depth changes pixels before the existing exposure law."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rospkg

from probtf_observation_fusion_demo.board_camera import BoardCamera, make_camera
from probtf_observation_fusion_demo.physical_occlusion import marker_occluder_settings
from probtf_switch_press_demo.vision import PlanarCamera


@pytest.fixture
def camera_inputs():
    texture = Path(rospkg.RosPack().get_path('prob_artag_detector'))/'samples/apriltag_36h11_id_021.png'
    board = dict(enabled=True, rows=3, columns=3, marker_size=.072, pitch=.09,
                 ids=[[17, 18, 19], [20, 21, 22], [23, 24, 25]])
    settings = dict(width=160, height=120, focal_length=120., noise_std=2.,
                    blur_sigma=.8, exposure_time=.1, motion_blur_samples=3)
    return texture, board, settings


def camera_pose(translation):
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = translation
    return pose


def test_disabled_factory_is_original_camera_with_identical_noise_and_exposure_bytes(camera_inputs):
    texture, board, settings = camera_inputs
    reference = BoardCamera(texture, board, settings, 21)
    disabled = make_camera(texture, board, dict(settings,
        marker_occluder=dict(enabled=False)), 21, marker_size=.18)
    assert type(disabled) is BoardCamera
    first, last = camera_pose([.1, -.22, .72]), camera_pose([.06, -.20, .36])
    history = [(0., first), (.1, last)]
    np.testing.assert_array_equal(disabled.render(last, np.eye(4), history),
                                  reference.render(last, np.eye(4), history))
    assert disabled.diagnostics == reference.diagnostics
    plain = make_camera(texture, dict(enabled=False), settings, 21, marker_size=.18)
    assert type(plain) is PlanarCamera


def test_sheet_behind_board_does_not_cover_board_and_sheet_behind_camera_is_clipped(camera_inputs):
    texture, board, settings = camera_inputs
    base = BoardCamera(texture, board, settings, 21)
    pose = camera_pose([0., 0., .6])
    for z in (-.1, .8):
        configured = dict(settings, marker_occluder=dict(enabled=True,
            center=[0., 0., z], size=[.15, .15]))
        occluded = make_camera(texture, board, configured, 21)
        expected = base.sharp_image(pose, np.eye(4))
        actual = occluded.sharp_image(pose, np.eye(4))
        np.testing.assert_array_equal(actual, expected)
        assert occluded._occlusion_samples[-1]['occluded_board_pixels'] == 0


def test_one_fixed_sheet_covers_more_board_when_camera_physically_approaches(camera_inputs):
    texture, board, settings = camera_inputs
    camera = make_camera(texture, board, dict(settings, marker_occluder=dict(enabled=True)), 21)
    geometry = deepcopy(camera.occluder_config)
    far, near = camera_pose([.111, -.2245, .7237]), camera_pose([.0633, -.2006, .3596])
    camera.sharp_image(far, np.eye(4))
    distant = camera._occlusion_samples[-1]
    image = camera.sharp_image(near, np.eye(4))
    close = camera._occlusion_samples[-1]
    assert 0. < distant['occluded_board_fraction'] < .8
    assert close['occluded_board_fraction'] > .95
    assert close['occluded_board_fraction'] > distant['occluded_board_fraction']
    assert camera.occluder_config == geometry
    assert np.count_nonzero(image == geometry['intensity']) > close['occluded_board_pixels']


def test_each_exposure_is_occluded_before_single_gaussian_blur_and_noise_draw(camera_inputs, monkeypatch):
    texture, board, settings = camera_inputs
    camera = make_camera(texture, board, dict(settings, marker_occluder=dict(enabled=True)), 21)
    first, last = camera_pose([.111, -.2245, .7237]), camera_pose([.0633, -.2006, .3596])
    poses = []
    for alpha in np.linspace(0., 1., settings['motion_blur_samples']):
        pose = first.copy()
        pose[:3, 3] = (1.-alpha)*first[:3, 3]+alpha*last[:3, 3]
        poses.append(pose)
    sharp = [camera.sharp_image(pose, np.eye(4)) for pose in poses]
    expected = np.mean(sharp, axis=0).astype(np.float32)
    expected = cv2.GaussianBlur(expected, (0, 0), settings['blur_sigma'])
    expected += np.random.default_rng(21).normal(0., settings['noise_std'], expected.shape).astype(np.float32)
    expected = np.clip(np.rint(expected), 0, 255).astype(np.uint8)
    calls = dict(blur=0, noise=0)
    blur, rng = cv2.GaussianBlur, np.random.default_rng(21)
    def one_blur(*args, **kwargs):
        calls['blur'] += 1
        return blur(*args, **kwargs)
    def one_noise(*args, **kwargs):
        calls['noise'] += 1
        return rng.normal(*args, **kwargs)
    monkeypatch.setattr(cv2, 'GaussianBlur', one_blur)
    camera.rng = SimpleNamespace(normal=one_noise)
    image = camera.render(last, np.eye(4), [(0., first), (.1, last)])
    np.testing.assert_array_equal(image, expected)
    assert calls == dict(blur=1, noise=1)
    diagnostic = camera.diagnostics['marker_occluder']
    assert diagnostic['exposure_samples'] == settings['motion_blur_samples']
    assert len(diagnostic['exposure_occluded_board_fractions']) == settings['motion_blur_samples']
    assert diagnostic['occluded_board_fraction'] == pytest.approx(np.mean(
        diagnostic['exposure_occluded_board_fractions']))
    assert camera.diagnostics['marker_board']
    assert camera.diagnostics['image_noise_applications'] == 1
    assert not diagnostic['distance_trigger_used'] and not diagnostic['sensor_events_disabled']


def test_enabled_single_marker_camera_has_same_physical_depth_compositing(camera_inputs):
    texture, _, settings = camera_inputs
    camera = make_camera(texture, dict(enabled=False), dict(settings,
        marker_occluder=dict(enabled=True, center=[0., 0., .2], size=[.2, .2])), 21,
        marker_size=.18)
    assert isinstance(camera, PlanarCamera) and not isinstance(camera, BoardCamera)
    camera.render(camera_pose([0., 0., .6]), np.eye(4))
    assert camera.diagnostics['marker_occluder']['occluded_board_fraction'] > .95


@pytest.mark.parametrize('invalid', (dict(size=[0., .1]), dict(center=[0., np.nan, .2]),
                                    dict(intensity=256), dict(enabled='false')))
def test_invalid_physical_geometry_is_explicit(invalid):
    with pytest.raises(ValueError):
        marker_occluder_settings(dict(marker_occluder=invalid))
