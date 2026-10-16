"""Physical replay and the observation-only boundary of the moving comparison."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene, SensorInputs


def test_switch_scene_is_shared_and_commands_reproduce_flexible_reach(tmp_path):
    first = MovingSwitchScene(tmp_path / "first")
    second = MovingSwitchScene(tmp_path / "second")
    original = Path(__file__).resolve().parents[2] / "probtf_switch_press_demo/config/default.yaml"
    original = yaml.safe_load(original.read_text(encoding="utf-8"))
    for key in ("seed", "base_translation", "rod", "camera", "marker", "obstacles",
                "plant", "initial_command", "motion", "reach_tolerance"):
        assert first.config[key] == original[key]
    times = (0.0, 9.0, 18.0, 27.0, first.duration)
    for time in times:
        a, b = first.step(time, 1000.0 + time), second.step(time, 1000.0 + time)
        np.testing.assert_array_equal(a.command, b.command)
        np.testing.assert_array_equal(a.joints, b.joints)
        np.testing.assert_array_equal(a.gravity_in_sensor, b.gravity_in_sensor)
        np.testing.assert_array_equal(a.image, b.image)
        assert not a.collided
    assert a.reached
    assert np.linalg.norm(a.tip - first.goal) < 0.006
    # The two rendered robot models have different physical consequences.
    # A rigid robot at the same command does not accidentally share the result.
    assert np.linalg.norm(a.rigid_tip - first.goal) > 0.6
    assert np.linalg.norm(a.tip - a.rigid_tip) > 0.6
    assert a.maximum_deflection_rad > 0.3
    assert first.trajectory[-1]["physical_error_m"] < first.trajectory[0]["physical_error_m"] / 50.0


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
