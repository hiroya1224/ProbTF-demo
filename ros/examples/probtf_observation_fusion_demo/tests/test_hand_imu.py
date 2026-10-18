"""A hand-mounted gravity sensor observes every upstream transform candidate."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import roslaunch
import yaml

from probtf_observation_fusion_demo.body_native import reexpress_state, point_query
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def attachments(tmp_path_factory):
    root = tmp_path_factory.mktemp('hand_imu_attachment')
    intermediate = MovingSwitchScene(root/'intermediate')
    hand = MovingSwitchScene(root/'hand', imu_target_link='tool0')
    frames = tuple(scene.step(0., 1000.) for scene in (intermediate, hand))
    return intermediate, hand, frames


def _expected_gravity(scene, frame, target):
    # Simulator truth is used only to audit generation of this synthetic
    # sensor reading; it is never supplied to MovingInference.
    gravity = frame.link_poses[target][:3, :3].T @ np.array([0., 0., -1.])
    seed = np.random.SeedSequence([int(scene.config['seed']),
                                  int(round(frame.time*1e6)), 1701])
    noise = np.random.default_rng(seed).normal(0., scene.imu_sigma_rad, 3)
    gravity += noise-gravity*np.dot(gravity, noise)
    return gravity/np.linalg.norm(gravity)


def test_hand_attachment_changes_only_the_physical_sensor_frame(attachments):
    intermediate, hand, (original, mounted) = attachments
    assert intermediate.imu_target_link == 'link_3'
    assert hand.imu_target_link == 'tool0'
    assert intermediate.config == hand.config
    assert intermediate.config_sha256 == hand.config_sha256
    assert intermediate.robot_description == hand.robot_description
    assert hand.config['imu']['target_link'] == 'link_3'
    assert intermediate.imu_sigma_rad == hand.imu_sigma_rad
    for name in ('initial_command', 'marker', 'goal', 'tip_offset', 'hand_camera'):
        np.testing.assert_array_equal(getattr(intermediate, name), getattr(hand, name))
    for name in ('command', 'joints', 'tip', 'rigid_tip', 'camera_pose', 'image'):
        np.testing.assert_array_equal(getattr(original, name), getattr(mounted, name))
    np.testing.assert_array_equal(original.gravity_in_sensor,
                                  _expected_gravity(intermediate, original, 'link_3'))
    np.testing.assert_array_equal(mounted.gravity_in_sensor,
                                  _expected_gravity(hand, mounted, 'tool0'))
    assert np.linalg.norm(original.gravity_in_sensor-mounted.gravity_in_sensor) > .1
    inputs = hand.sensor_inputs(mounted)
    assert inputs.imu_target_link == 'tool0'
    assert inputs.imu_sigma_rad == intermediate.imu_sigma_rad
    assert not hasattr(inputs, 'joints') and not hasattr(inputs, 'link_poses')


def test_hand_gravity_updates_all_six_local_laws_and_the_same_display_distribution(
        attachments, monkeypatch):
    intermediate, hand, (original, mounted) = attachments
    observations = yaml.safe_load((PACKAGE/'config/imu_hand.yaml').read_text())['observations']
    original_observations = yaml.safe_load((PACKAGE/'config/imu.yaml').read_text())['observations']
    reference = MovingInference(intermediate.robot_description, original_observations)
    reference_prior = reference.prior_at(original.command, original.stamp, 0)
    inference = MovingInference(hand.robot_description, observations)
    inputs = hand.sensor_inputs(mounted)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Hand-IMU inference and display must not draw Monte Carlo samples.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    first = inference.update(inputs, 0)
    assert first.prior.state_id == reference_prior.state_id
    np.testing.assert_array_equal(first.prior.covariance, reference_prior.covariance)
    assert first.local_estimates[0].diagnostics['target_link'] == 'tool0'
    assert first.local_estimates[0].diagnostics['measurement_variance'] == inputs.imu_sigma_rad**2
    previous = first.stages[-1].components[0].state
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    np.testing.assert_array_equal(second.prior.covariance, previous.covariance)
    assert second.records['prediction'][0]['source_state_id'] == previous.state_id
    assert second.records['observation_ids'] != first.records['observation_ids']
    current = second.stages[-1].components[0].state
    for edge in inference.body_ids:
        index = current.edge_ids.index(edge)
        rotation = slice(6*index+3, 6*index+6)
        assert np.trace(previous.covariance[rotation, rotation]) < np.trace(
            first.prior.covariance[rotation, rotation])
        assert np.trace(current.covariance[rotation, rotation]) < np.trace(
            previous.covariance[rotation, rotation])
    first_index, last_index = (current.edge_ids.index(edge) for edge in ('link_1', 'link_6'))
    assert np.linalg.norm(current.covariance[
        6*first_index:6*first_index+6, 6*last_index:6*last_index+6]) > 1e-6
    for edge in ('tool0', 'press_rod', 'rod_tip', 'camera_optical_frame'):
        index = current.edge_ids.index(edge)
        np.testing.assert_array_equal(current.covariance[6*index:6*index+6], 0.)

    stage = second.stages[-1]
    actual = {component.label: component for component in stage.components}
    # Native keep/reset children remain the change-model evidence; the
    # declared projected numerator is the body actually used and displayed.
    assert [branch.weight for branch in second.frozen_body_branches] == [.75, .25]
    numerator = second.projected_body_numerator
    assert numerator is not None
    assert numerator.reference_state.state_id == second.prior.state_id
    assert len(numerator.posterior_components) == len(stage.components) == 1
    for raw, canonical in zip(numerator.posterior_components,
                              second.frozen_body_posterior.components):
        np.testing.assert_array_equal(raw.distribution.mean, canonical.distribution.mean)
        np.testing.assert_array_equal(raw.distribution.covariance, canonical.distribution.covariance)
        displayed = actual[raw.label]
        rebuilt = reexpress_state(numerator.reference_state, raw.distribution.mean,
            raw.distribution.covariance, version=displayed.state.version,
            component_id=displayed.state.component_id)
        assert rebuilt.state_id == displayed.state.state_id
        np.testing.assert_array_equal(rebuilt.covariance, displayed.state.covariance)
        assert displayed.weight == raw.weight and displayed.log_weight == raw.log_weight
    snapshot = build_snapshot(second, stage, inference.common)
    verify_adapter_identity(snapshot, stage)
    observed = [point for point in snapshot.points if point.axis == 'observed']
    assert {point.frame_label for point in observed} == {'IMU gravity at tool0'}
    axis = inference.common['display']['axis_length']
    for point in observed:
        source = actual[point.component_id]
        expected = point_query(source.state, second.link_paths['tool0'],
            axis*inputs.gravity_in_sensor/np.linalg.norm(inputs.gravity_in_sensor),
            point.query.query_id, second.root_frame)
        np.testing.assert_array_equal(point.query.mean, expected.mean)
        np.testing.assert_array_equal(point.query.covariance, expected.covariance)
    settings = inference.common['display']
    markers = make_markers(snapshot, settings['seed'], settings['point_size'])
    assert verify_wire_identity(snapshot, markers, seed=settings['seed'],
                                point_size=settings['point_size'])


def test_unavailable_imu_attachment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='IMU attachment'):
        MovingSwitchScene(tmp_path, imu_target_link='not_a_robot_link')


def test_hand_launch_selects_async_motion_and_preserves_the_requested_rviz_file(tmp_path):
    custom_rviz = str(tmp_path/'unchanged_user_view.rviz')
    config = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(str(PACKAGE/'launch/imu_hand.launch'), config,
        argv=['rviz_config:='+custom_rviz], verbose=False)
    node = next(node for node in config.nodes if node.name == 'observation_fusion_demo')
    assert node.type == 'observation_fusion_motion.py'
    assert config.params['/observation_fusion_demo/execution_mode'].value == 'async'
    assert config.params['/observation_fusion_demo/imu_target_link'].value == 'tool0'
    assert config.params['/observation_fusion_demo/observations_config'].value == str(
        PACKAGE/'config/imu_hand.yaml')
    view = next(node for node in config.nodes if node.name == 'observation_fusion_rviz')
    assert view.args == '-d '+custom_rviz
