import numpy as np

from prob_artag_renderer.annotations import annotate_tags
from prob_artag_renderer.camera_model import CameraModel
from prob_artag_renderer.config import load_config
from prob_artag_renderer.coordinates import invert_transform, projected_edge_lengths
from prob_artag_renderer.scene_sampler import SCENARIOS, SceneSampler


def _angle_deg(tag, T_W_C):
    T_C_M = invert_transform(T_W_C).dot(tag.T_W_M)
    normal = T_C_M[:3, 2]
    toward_camera = -T_C_M[:3, 3]
    toward_camera /= np.linalg.norm(toward_camera)
    return np.rad2deg(np.arccos(np.clip(np.dot(normal, toward_camera), -1.0, 1.0)))


def test_all_scenarios_are_valid_and_deterministic():
    config = load_config()
    camera = CameraModel.from_dict(config["camera"])
    first = SceneSampler(camera, config, seed=123)
    second = SceneSampler(camera, config, seed=123)
    for scenario in SCENARIOS:
        a = first.sample(scenario)
        b = second.sample(scenario)
        assert len(a.tags) == len(b.tags)
        assert [tag.marker_id for tag in a.tags] == [tag.marker_id for tag in b.tags]
        for tag_a, tag_b in zip(a.tags, b.tags):
            np.testing.assert_array_equal(tag_a.T_W_M, tag_b.T_W_M)
        annotation = annotate_tags(0, scenario, 123, camera, a.T_W_C, a.tags, {})
        for tag in annotation.tags:
            assert tag.front_facing
            assert np.all(tag.corners_depth_m > 0)
            assert np.all(tag.corners_px[:, 0] >= 30)
            assert np.all(tag.corners_px[:, 0] < camera.width - 30)
            assert np.all(tag.corners_px[:, 1] >= 30)
            assert np.all(tag.corners_px[:, 1] < camera.height - 30)
            edge = np.mean(projected_edge_lengths(tag.corners_px))
            assert 24 <= edge <= 400
            if scenario == "small":
                assert edge <= 60
        angles = [_angle_deg(tag, a.T_W_C) for tag in a.tags]
        if scenario == "frontal":
            assert all(0 <= angle <= 10 for angle in angles)
        elif scenario == "oblique":
            assert all(45 <= angle <= 70 for angle in angles)
    assert len(first.sample("multi_tag").tags) in (3, 4, 5)
    assert len(first.sample("occluded").occluders) == 1


def test_camera_sequence_keeps_world_tags_fixed_and_moves_smoothly():
    config = load_config()
    camera = CameraModel.from_dict(config["camera"])
    sequence = SceneSampler(camera, config, seed=11).sample_sequence("multi_tag", 10)
    assert [frame.frame_id for frame in sequence] == list(range(10))
    for frame in sequence[1:]:
        for initial, current in zip(sequence[0].tags, frame.tags):
            np.testing.assert_array_equal(initial.T_W_M, current.T_W_M)
    positions = np.array([frame.T_W_C[:3, 3] for frame in sequence])
    steps = np.diff(positions, axis=0)
    np.testing.assert_allclose(steps, np.repeat(steps[:1], 9, axis=0), atol=1e-12)
    assert np.linalg.norm(positions[-1] - positions[0]) > 0.01
