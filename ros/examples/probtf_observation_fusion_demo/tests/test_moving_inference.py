"""Moving commands retain the common prior, sensor boundary and joint laws."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.spherical_law.first_order import JointTransformState, point_query
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def moving_inputs(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp("moving_inference"))
    frame = scene.step(0.0, 1000.0)
    cases = {name: tuple(yaml.safe_load((CONFIG / (name + ".yaml")).read_text())["observations"])
             for name in ("no_sensors", "imu", "marker", "imu_marker")}
    return scene, frame, cases


def test_moving_cases_share_exact_prior_and_core_paths(moving_inputs):
    scene, frame, cases = moving_inputs
    inputs = scene.sensor_inputs(frame)
    results = {name: MovingInference(scene.robot_description, observations).update(inputs, 0)
               for name, observations in cases.items()}
    common = results["no_sensors"].prior
    for result in results.values():
        assert result.prior.state_id == common.state_id
        assert result.prior.stamp == frame.stamp
        assert result.prior.command == tuple(frame.command)
        assert result.prior.edge_ids == common.edge_ids
        assert result.hand_path[-3:] == (("tool0", False), ("press_rod", False), ("rod_tip", False))
        assert result.camera_path[-2:] == (("tool0", False), ("camera_optical_frame", False))
        np.testing.assert_array_equal(result.prior.covariance, common.covariance)
        assert result.records["moving_input_family"]["physical_truth_received"] is False
    assert results["no_sensors"].stages[-1].components[0].state is common
    assert [results[name].records["valid_observation_count"] for name in cases] == [0, 1, 1, 2]
    # The same actual local camera/IMU distributions feed standalone and fusion.
    assert results["imu"].local_estimates[0].diagnostics == results["imu_marker"].local_estimates[0].diagnostics
    assert results["marker"].local_estimates[0].diagnostics == results["imu_marker"].local_estimates[1].diagnostics


def test_command_family_uses_known_geometry_and_retains_posterior_cross_blocks(moving_inputs):
    scene, frame, cases = moving_inputs
    inference = MovingInference(scene.robot_description, cases["imu_marker"])
    result = inference.update(scene.sensor_inputs(frame), 0)
    # Its modal path is FK of the given command family, not of the sprung arm.
    hand = point_query(result.prior, result.hand_path)
    np.testing.assert_allclose(hand.mode, frame.rigid_tip, atol=1e-12)
    assert np.linalg.norm(hand.mode - frame.tip) > 0.2
    posterior = result.stages[-1].components[0].state
    diagonal = np.zeros_like(posterior.covariance)
    for i in range(len(posterior.transforms)):
        diagonal[6*i:6*i+6, 6*i:6*i+6] = posterior.covariance[6*i:6*i+6, 6*i:6*i+6]
    assert np.linalg.norm(posterior.covariance-diagonal) > 1e-4
    artificial = JointTransformState(posterior.transforms, diagonal,
        version=posterior.version, stamp=posterior.stamp, command=posterior.command)
    exact_query = point_query(posterior, result.hand_path)
    removed = point_query(artificial, result.hand_path)
    assert np.linalg.norm(exact_query.covariance-removed.covariance) > 1e-5


def test_same_sensor_inputs_have_no_hidden_physics_or_random_inference(moving_inputs, monkeypatch):
    scene, frame, cases = moving_inputs
    inputs = scene.sensor_inputs(frame)
    expected = MovingInference(scene.robot_description, cases["imu_marker"]).update(inputs, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not draw transform or propagation samples.")
    for name in ("default_rng", "normal", "random", "randn", "multivariate_normal", "RandomState"):
        monkeypatch.setattr(np.random, name, forbidden)
    # Change physical masses/stiffness/poses outside the only estimator input.
    changed_urdf = scene.robot_description.replace('value="0.275"', 'value="99.0"')
    actual = MovingInference(changed_urdf, cases["imu_marker"]).update(inputs, 0)
    assert actual.records == expected.records


def test_frame_updates_retain_and_transport_the_previous_joint_posterior(moving_inputs):
    scene, frame, cases = moving_inputs
    inference = MovingInference(scene.robot_description, cases["imu"])
    inputs = scene.sensor_inputs(frame)
    first = inference.update(inputs, 0)
    assert inference.update(inputs, 0) is first
    next_command = np.asarray(inputs.command).copy()
    next_command[1] += 0.02
    later = replace(inputs, time=0.5, stamp=inputs.stamp+0.5, command=next_command)
    second = inference.update(later, 1)
    assert second.prior.command == tuple(next_command)
    assert second.prior.stamp == later.stamp
    assert second.prior.version > first.stages[-1].components[0].state.version
    assert first.records["observation_ids"] != second.records["observation_ids"]
    from probtf_observation_fusion_demo.moving_control import command_state
    previous = first.stages[-1].components[0]
    expected = command_state(previous.state, next_command, inference.body_ids, inference.actions)
    np.testing.assert_array_equal(second.prior.covariance, expected.covariance)
    for actual_edge, expected_edge in zip(second.prior.transforms, expected.transforms):
        np.testing.assert_array_equal(actual_edge.translation, expected_edge.translation)
        np.testing.assert_array_equal(actual_edge.rotation_wxyz, expected_edge.rotation_wxyz)
    assert second.records['prediction'][0]['source_state_id'] == previous.state.state_id
    assert not second.records['moving_input_family']['initialization']
    with pytest.raises(ValueError, match="indices must increase"):
        inference.update(inputs, 0)


def test_imu_leaves_downstream_relative_candidates_uncertain_and_unchanged(moving_inputs):
    scene, frame, cases = moving_inputs
    result = MovingInference(scene.robot_description, cases["imu"]).update(scene.sensor_inputs(frame), 0)
    updated = result.stages[-1].components[0].state
    for name in ("link_4", "link_5", "link_6"):
        index = result.prior.edge_ids.index(name)
        # Reexpression uses maximum-eigenvalue-zero gauge; B(A+cI)=B(A).
        before, after = result.prior.transforms[index].bingham, updated.transforms[index].bingham
        np.testing.assert_allclose(after-np.trace(after)*np.eye(4)/4,
                                   before-np.trace(before)*np.eye(4)/4, atol=1e-10)
        np.testing.assert_allclose(updated.covariance[6*index:6*index+6, 6*index:6*index+6],
                                   result.prior.covariance[6*index:6*index+6, 6*index:6*index+6], atol=1e-14)
        assert np.linalg.eigvalsh(updated.covariance[6*index+3:6*index+6, 6*index+3:6*index+6])[0] > 0.
