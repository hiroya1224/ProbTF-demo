"""Independent scalar checks for the batched collision and path calculations."""
from pathlib import Path

import numpy as np
import pytest

from probtf_switch_press_demo.body import EquilibriumBody, clearance, motion_samples, shape_displacement
from probtf_switch_press_demo.scene import apply, load_config, transform, write_robot
from symaware_grasp.box_geometry import box_box_gap


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def body_scene(tmp_path):
    config = load_config(ROOT / "ros/examples/probtf_switch_press_demo/config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "collision_arm.urdf", config)
    return EquilibriumBody(urdf, config["plant"]["stiffness"]), config


def scalar_clearance(boxes, marker_pose, obstacles):
    smallest, pair = float("inf"), ""
    for link, pose, half in boxes:
        for obstacle in obstacles:
            center = apply(marker_pose, obstacle["center"])
            other_half = np.asarray(obstacle["size"]) / 2
            bound = np.linalg.norm(pose[:3, 3] - center) - np.linalg.norm(half) - np.linalg.norm(other_half)
            gap = (float(bound) if bound > .06 else
                   box_box_gap(pose[:3, 3], pose[:3, :3], half,
                               center, marker_pose[:3, :3], other_half))
            if gap < smallest:
                smallest, pair = gap, link + "/" + obstacle["name"]
    return smallest, pair


def spectral_displacement(first, second):
    return max(np.linalg.norm(a[:3, 3] - b[:3, 3]) +
               np.linalg.norm(a[:3, :3] - b[:3, :3], ord=2) * np.linalg.norm(half)
               for (_name, a, half), (_other, b, _half) in zip(first, second))


@pytest.mark.parametrize("pose", [np.zeros(6), np.array([-.15, -1.85, 2.35, 0., -1.25, 0.]),
                                   np.array([.7, -.8, 1.2, -.4, .6, .3])])
def test_pinocchio_collision_boxes_match_original_urdf_forward_kinematics(body_scene, pose):
    body, _config = body_scene
    frames = body.geometry.transforms(pose, .185)
    for reference in (None, transform([.3, -.4, .2], [.2, -.5, .7])):
        actual = body.boxes(pose, reference)
        assert len(actual) == len(body.geometry.bodies)
        for (actual_link, actual_pose, actual_half), (link, _name, local, half) in zip(actual, body.geometry.bodies):
            expected = frames[link] @ local
            if reference is not None:
                expected = reference @ expected
            assert actual_link == link
            np.testing.assert_allclose(actual_pose, expected, rtol=0., atol=2e-15)
            np.testing.assert_array_equal(actual_half, half)


@pytest.mark.parametrize("marker, local_pose, expected_sign", [
    (np.eye(4), transform([.2, 0., 0.]), 0),                  # Exact parallel contact.
    (np.eye(4), transform([.15, 0., 0.]), -1),               # Overlap.
    (np.eye(4), transform([.205, 0., 0.]), 1),               # Near separation, SAT path.
    (np.eye(4), transform([2., 0., 0.]), 1),                # Distant sphere certificate.
    (transform([.4, -.2, .8], [.3, -.5, .7]), transform([.2, 0., 0.]), 0),
    (transform([.4, -.2, .8], [.3, -.5, .7]), transform([.205, 0., 0.], [1e-13, 0., 0.]), 1),
    (transform([.4, -.2, .8], [.3, -.5, .7]), transform([.08, .03, .01], [.4, -.7, .2]), -1),
])
def test_batched_clearance_matches_scalar_sat_for_contact_and_rotated_boxes(marker, local_pose, expected_sign):
    boxes = [("rod", marker @ local_pose, np.array([.1, .05, .02]))]
    obstacles = [dict(name="panel", center=[0., 0., 0.], size=[.2, .4, .06])]
    expected = scalar_clearance(boxes, marker, obstacles)
    actual = clearance(boxes, marker, obstacles)
    assert actual[1] == expected[1] == "rod/panel"
    assert actual[0] == pytest.approx(expected[0], abs=2e-14)
    assert np.sign(actual[0]) == expected_sign


def test_batched_clearance_checks_every_body_obstacle_pair_and_keeps_first_tie():
    marker = transform([.3, -.4, .6], [.5, .2, -.8])
    boxes = [(name, marker @ transform(center, rotation), np.asarray(half))
             for name, center, rotation, half in (
                 ("upper_arm", [1.8, .4, .6], [.1, -.4, .8], [.3, .04, .06]),
                 ("rod", [.025, -.14, .09], [.7, .2, -.3], [.008, .008, .16]),
                 ("camera", [-.02, .13, .04], [.2, -.3, .1], [.035, .02, .015]))]
    obstacles = [dict(name="panel", center=[0., 0., -.025], size=[.64, .72, .05]),
                 dict(name="button", center=[0., -.24, .015], size=[.05, .05, .03]),
                 dict(name="lamp", center=[.11, -.24, .018], size=[.045, .045, .036])]
    expected = scalar_clearance(boxes, marker, obstacles)
    actual = clearance(boxes, marker, obstacles)
    assert actual[0] == pytest.approx(expected[0], abs=2e-14)
    assert actual[1] == expected[1]
    duplicate_boxes = [boxes[1], ("same_rod", boxes[1][1].copy(), boxes[1][2].copy())]
    duplicate_obstacles = [obstacles[0], dict(obstacles[0], name="same_panel")]
    assert clearance(duplicate_boxes, marker, duplicate_obstacles)[1] == "rod/panel"


@pytest.mark.parametrize("rotation", [(0., 0., 0.), (.1, -.3, .8), (np.pi, 0., 0.), (1e-12, 0., 0.)])
def test_shape_displacement_matches_original_spectral_norm_for_so3(rotation):
    first = [("rod", transform([.1, .2, .3], [.3, -.1, .2]), np.array([.16, .008, .008])),
             ("camera", transform([-.2, .1, .8], [-.4, .2, -.6]), np.array([.035, .02, .015]))]
    movement = transform([.02, -.01, .005], rotation)
    second = [(name, movement @ pose, half) for name, pose, half in first]
    assert shape_displacement(first, second) == pytest.approx(spectral_displacement(first, second), abs=2e-15)
    assert shape_displacement(first, first) == 0.


def original_motion_samples(body, start, end, warm, settings):
    """Original recursive algorithm, deliberately recomputing endpoint boxes."""
    start, end = np.asarray(start), np.asarray(end)
    count = max(1, int(np.ceil(np.max(np.abs(end - start)) / settings["joint_resolution"])))
    pose = np.array(warm, copy=True)
    yield start.copy(), pose.copy()

    def segment(c0, q0, c1, q1, depth=0):
        if spectral_displacement(body.boxes(q0), body.boxes(q1)) <= settings["spatial_resolution"]:
            yield c1.copy(), q1.copy()
            return
        assert depth < 12
        middle = (c0 + c1) / 2
        middle_pose = body.equilibrium(middle, q0)
        yield from segment(c0, q0, middle, middle_pose, depth + 1)
        yield from segment(middle, middle_pose, c1, q1, depth + 1)

    previous = start
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        command = start + fraction * (end - start)
        next_pose = body.equilibrium(command, pose)
        yield from segment(previous, pose, command, next_pose)
        previous, pose = command, next_pose


def test_cached_motion_endpoints_preserve_equilibrium_path_and_spatial_bound(body_scene):
    body, config = body_scene
    start = np.asarray(config["initial_command"])
    end = start + np.array([.009, -.012, .006, .003, 0., 0.])
    warm = body.equilibrium(start)

    class CountedBody:
        def __init__(self):
            self.box_calls = 0

        def equilibrium(self, *args):
            return body.equilibrium(*args)

        def boxes(self, *args):
            self.box_calls += 1
            return body.boxes(*args)

    old_body, new_body = CountedBody(), CountedBody()
    expected = list(original_motion_samples(old_body, start, end, warm, config["motion"]))
    actual = list(motion_samples(new_body, start, end, warm, config["motion"]))
    assert len(actual) == len(expected) > 2
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
    # Each pose's collision boxes can be retained through all recursive uses.
    assert new_body.box_calls <= len(actual)
    assert new_body.box_calls < old_body.box_calls
    for (_c0, q0), (_c1, q1) in zip(actual, actual[1:]):
        assert spectral_displacement(body.boxes(q0), body.boxes(q1)) <= config["motion"]["spatial_resolution"]
