"""Shared physical model with explicit commands and an observation-only boundary."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene, SensorInputs


def test_switch_scene_never_loads_or_replays_successful_commands(tmp_path, monkeypatch):
    original_open = Path.open

    def reject_recordings(path, *args, **kwargs):
        if path.suffix == '.csv':
            raise AssertionError('Physical scene must not load recorded successful commands.')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', reject_recordings)
    first = MovingSwitchScene(tmp_path / "first")
    second = MovingSwitchScene(tmp_path / "second")
    original = Path(__file__).resolve().parents[2] / "probtf_switch_press_demo/config/default.yaml"
    original = yaml.safe_load(original.read_text(encoding="utf-8"))
    for key in ("seed", "base_translation", "rod", "camera", "marker", "obstacles",
                "plant", "initial_command", "reach_tolerance"):
        assert first.config[key] == original[key]
    for key in ('joint_resolution', 'spatial_resolution', 'clearance', 'frame_period'):
        assert first.config['motion'][key] == original['motion'][key]
    times = (0.0, 9.0, 18.0, 27.0, first.duration)
    for time in times:
        a, b = first.step(time, 1000.0 + time), second.step(time, 1000.0 + time)
        np.testing.assert_array_equal(a.command, b.command)
        np.testing.assert_array_equal(a.joints, b.joints)
        np.testing.assert_array_equal(a.gravity_in_sensor, b.gravity_in_sensor)
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.command, first.initial_command)
        assert not a.collided
    assert not a.reached
    assert np.linalg.norm(a.tip - first.goal) > first.config['reach_tolerance']
    assert np.linalg.norm(a.tip - a.rigid_tip) > .2
    np.testing.assert_allclose(first.trajectory[-1]['physical_error_m'],
                               first.trajectory[0]['physical_error_m'], atol=1e-12)


def test_physics_integrates_from_the_last_actual_sent_command(tmp_path, monkeypatch):
    from probtf_observation_fusion_demo import moving_scene
    scene = MovingSwitchScene(tmp_path)
    original_samples = moving_scene.motion_samples
    segments = []

    def record_samples(body, start, end, warm, settings):
        segments.append((np.array(start), np.array(end)))
        yield from original_samples(body, start, end, warm, settings)

    monkeypatch.setattr(moving_scene, 'motion_samples', record_samples)
    scene.step(0., 1000.)
    first_command = scene.initial_command + .001
    scene.step(.05, 1000.05, command=first_command)
    second_command = scene.initial_command - .002
    second = scene.step(.1, 1000.1, command=second_command)
    np.testing.assert_array_equal(segments[0][0], scene.initial_command)
    np.testing.assert_array_equal(segments[0][1], first_command)
    np.testing.assert_array_equal(segments[1][0], first_command)
    np.testing.assert_array_equal(segments[1][1], second_command)
    np.testing.assert_array_equal(second.command, second_command)
    held = scene.step(.15, 1000.15)
    np.testing.assert_array_equal(held.command, second_command)
    scene.reset()
    reset = scene.step(0., 1000.)
    np.testing.assert_array_equal(reset.command, scene.initial_command)


def test_estimator_sensor_input_excludes_simulator_state(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0.0, 1000.0)
    observation = scene.sensor_inputs(frame)
    assert {item.name for item in fields(SensorInputs)} == {
        "time", "stamp", "command", "image", "gravity_in_sensor",
        "imu_target_link", "imu_sigma_rad"}
    assert observation.command is frame.command
    assert observation.image is frame.image
    assert observation.gravity_in_sensor is frame.gravity_in_sensor
    assert observation.stamp == frame.stamp
    assert observation.imu_target_link == "link_3"
    assert not observation.command.flags.writeable
    assert not observation.image.flags.writeable
    np.testing.assert_allclose(np.linalg.norm(observation.gravity_in_sensor), 1.0)
