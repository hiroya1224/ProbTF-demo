"""Native physical pixels agree with the original vectorized pixel tests."""

from pathlib import Path

import numpy as np
import pytest
import rospkg
from scipy.spatial.transform import Rotation

from probtf_observation_fusion_demo import physical_occlusion
from probtf_observation_fusion_demo.board_camera import make_camera
from probtf_observation_fusion_demo.physical_camera_native import occlusion_mask


def numpy_reference(rays, rotation, origin, center, lower, upper, near, board, bounds):
    """The original implementation, including division and exact edge tests."""
    x0, y0, x1, y1 = bounds
    ray_x, ray_y, ray_z = (value[y0:y1, x0:x1] for value in rays)
    directions = [rotation[i, 0]*ray_x+rotation[i, 1]*ray_y+rotation[i, 2]*ray_z
                  for i in range(3)]
    denominator = directions[2]
    depth = np.zeros_like(denominator)
    np.divide(center[2]-origin[2], denominator, out=depth, where=denominator != 0.)
    hit_x, hit_y = origin[0]+depth*directions[0], origin[1]+depth*directions[1]
    sheet = ((denominator != 0.) & (depth > near)
        & (hit_x >= lower[0]) & (hit_x <= upper[0])
        & (hit_y >= lower[1]) & (hit_y <= upper[1]))
    board_depth = np.zeros_like(denominator)
    np.divide(-origin[2], denominator, out=board_depth, where=denominator != 0.)
    return sheet & (~board[y0:y1, x0:x1].astype(bool) | (depth < board_depth))


@pytest.mark.parametrize('rotation_vector', ([0., 0., 0.], [.0, .3, 0.],
    [1.2, -.4, .7], [0., np.pi/2., 0.]))
@pytest.mark.parametrize('sheet_z', (-.2, 0., .2))
def test_exact_edges_depth_order_parallel_rays_and_roi_match(rotation_vector, sheet_z):
    edges = np.array([-1., np.nextafter(-.5, -np.inf), -.5,
        np.nextafter(-.5, np.inf), 0., np.nextafter(.5, -np.inf),
        .5, np.nextafter(.5, np.inf), 1.])
    x, y = np.meshgrid(edges, edges)
    rays = (x, y, np.ones_like(x))
    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    origin = np.array([0., 0., -1.])
    board = ((np.indices(x.shape).sum(axis=0) % 3) != 0).astype(np.uint8)
    arguments = (rays, rotation, origin, [0., 0., sheet_z], [-.5, -.5],
                 [.5, .5], .02, board)
    for bounds in ((0, 0, 9, 9), (1, 2, 8, 7)):
        np.testing.assert_array_equal(occlusion_mask(*arguments, bounds),
                                      numpy_reference(*arguments, bounds))


def test_exact_near_plane_and_zero_denominator_match_without_epsilon():
    x, y = np.meshgrid(np.linspace(-1., 1., 7), np.linspace(-1., 1., 5))
    z = np.ones_like(x)
    z[:, 3] = 0.
    board = np.zeros(x.shape, np.uint8)
    for depth in (.02, np.nextafter(.02, np.inf), np.nextafter(.02, -np.inf)):
        arguments = ((x, y, z), np.eye(3), np.zeros(3), [0., 0., depth],
                     [-1., -1.], [1., 1.], .02, board, (0, 0, 7, 5))
        np.testing.assert_array_equal(occlusion_mask(*arguments), numpy_reference(*arguments))


def test_final_exposure_blur_noise_pixels_and_diagnostics_are_byte_identical(monkeypatch):
    texture = Path(rospkg.RosPack().get_path('prob_artag_detector'))/'samples/apriltag_36h11_id_021.png'
    board = dict(enabled=True, rows=3, columns=3, marker_size=.072, pitch=.09,
                 ids=[[17, 18, 19], [20, 21, 22], [23, 24, 25]])
    settings = dict(width=320, height=240, focal_length=240., noise_std=2.,
        blur_sigma=.8, exposure_time=.1, motion_blur_samples=7,
        marker_occluder=dict(enabled=True))
    reference = make_camera(texture, board, settings, 32)
    native = make_camera(texture, board, settings, 32)
    first, last = np.eye(4), np.eye(4)
    first[:3, :3] = np.diag([1., -1., -1.])
    last[:3, :3] = Rotation.from_rotvec([.03, -.02, .01]).as_matrix() @ first[:3, :3]
    first[:3, 3], last[:3, 3] = [.111, -.2245, .7237], [.0633, -.2006, .3596]
    history = [(0., first), (.1, last)]
    for _ in range(3):
        monkeypatch.setattr(physical_occlusion, 'occlusion_mask', numpy_reference)
        expected = reference.render(last, np.eye(4), history)
        expected_diagnostics = reference.diagnostics.copy()
        monkeypatch.setattr(physical_occlusion, 'occlusion_mask', occlusion_mask)
        actual = native.render(last, np.eye(4), history)
        np.testing.assert_array_equal(actual, expected)
        assert native.diagnostics == expected_diagnostics


def test_invalid_buffer_geometry_is_not_a_silent_empty_mask():
    rays = (np.ones((2, 3)),)*3
    with pytest.raises(ValueError, match='rectangle'):
        occlusion_mask(rays, np.eye(3), np.zeros(3), [0., 0., .2],
                       [-1., -1.], [1., 1.], .02, np.zeros((2, 3), np.uint8), (0, 0, 4, 2))
    with pytest.raises(ValueError, match='finite'):
        occlusion_mask(rays, np.eye(3), np.array([0., 0., np.nan]), [0., 0., .2],
                       [-1., -1.], [1., 1.], .02, np.zeros((2, 3), np.uint8), (0, 0, 3, 2))
