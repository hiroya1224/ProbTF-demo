"""A detected stationary switch remains stationary when the body alone updates."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, point_query, reexpress_state,
)
from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.body_projection import (
    project_update, retain_goal_components,
)
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene

from test_body_projection import _result, common_state


PACKAGE = Path(__file__).resolve().parents[1]


def test_schmidt_retention_keeps_the_entire_marker_law_and_contracts_cross_blocks(common_state):
    projection = project_update(_result(common_state))
    prior = replace(projection.state, version=4, stamp=1000.05)
    sources = tuple(replace(item, state=replace(item.state, version=4, stamp=1000.05))
                    for item in projection.goal_components)
    mean = prior.mean.copy()
    mean[3], mean[15], mean[60], mean[64] = .02, -.03, .1, .05
    updated = Gaussian(mean, prior.covariance*.7)
    current = reexpress_state(prior, updated.mean, updated.covariance,
                              version=6, component_id='internal:6')
    retained = retain_goal_components(sources, prior, updated, current)
    # Independent closed-form right-chart derivative; scalar contraction gives
    # the cross update .7*G*C without a second inverse-based implementation.
    chart = np.eye(60)
    for index in range(10):
        vector = mean[6*index+3:6*index+6]
        theta = np.linalg.norm(vector)
        if theta:
            unit = vector/theta
            cross = np.array([[0., -unit[2], unit[1]], [unit[2], 0., -unit[0]],
                              [-unit[1], unit[0], 0.]])
            block = (np.eye(3)-((1.-np.cos(theta))/theta)*cross
                     +(1.-np.sin(theta)/theta)*(cross @ cross))
            chart[6*index+3:6*index+6, 6*index+3:6*index+6] = block
    for source, actual in zip(sources, retained):
        old, state = source.state, actual.state
        assert (actual.label, actual.weight, actual.log_weight) == (
            source.label, source.weight, source.log_weight)
        assert state.transforms[-1] is old.transforms[-1]
        assert all(edge is expected for edge, expected in zip(state.transforms[:-1], current.transforms[:-1]))
        np.testing.assert_array_equal(state.covariance[60:, 60:], old.covariance[60:, 60:])
        np.testing.assert_array_equal(state.covariance[:60, :60], current.covariance[:60, :60])
        np.testing.assert_allclose(state.covariance[:60, 60:],
                                   .7*chart @ old.covariance[:60, 60:], atol=2e-18, rtol=2e-12)
        assert np.linalg.norm(state.covariance[:60, 60:]) > 0.
        covariance_support(state.covariance)
        for point in ([0., 0., 0.], [0., -.24, .045], [.075, -.24, .045]):
            before = point_query(old, (('marker', False),), point)
            after = point_query(state, (('marker', False),), point)
            np.testing.assert_array_equal(after.mean, before.mean)
            np.testing.assert_allclose(after.covariance, before.covariance, atol=1e-18, rtol=1e-13)
        assert state.version == current.version
        assert state.stamp == current.stamp and state.command == current.command


def test_cut_can_hold_a_fixed_goal_even_with_perfect_prior_body_marker_correlation():
    body = BinghamTransform('body', [0., 0., 0.], rotation_wxyz=[1., 0., 0., 0.],
                            translation_covariance=np.diag([1., 0., 0.]))
    marker = BinghamTransform('marker', [1., 0., 0.], rotation_wxyz=[1., 0., 0., 0.],
                              translation_covariance=np.diag([1., 0., 0.]))
    covariance = np.zeros((12, 12))
    covariance[np.ix_([0, 6], [0, 6])] = 1.
    prior = JointTransformState((body, marker), covariance, stamp=1., command=(0.,))
    raw_mean = np.zeros(12)
    raw_mean[[0, 6]] = .1
    raw = Gaussian(raw_mean, .5*covariance)
    internal = reexpress_state(prior, raw.mean, raw.covariance, version=2)
    source = WeightedState(prior.component_id, 1., prior, 0.)
    retained = retain_goal_components((source,), prior, raw, internal)[0].state
    assert retained.transforms[0] is internal.transforms[0]
    assert retained.transforms[1] is marker
    np.testing.assert_array_equal(retained.covariance[np.ix_([0, 6], [0, 6])],
                                  [[.5, .5], [.5, 1.]])
    covariance_support(retained.covariance)


def test_image_loss_retains_underflowed_modes_and_empty_updates_are_identity(common_state):
    projection = project_update(_result(common_state, weights=(1., 0.), logs=(0., -1000.)))
    prior = projection.state
    retained = retain_goal_components(projection.goal_components, prior,
        Gaussian(prior.mean, prior.covariance), prior)
    assert all(actual is source for actual, source in zip(retained, projection.goal_components))
    assert retained[1].weight == 0. and retained[1].log_weight == -1000.
    # A missing image must not become an unreported covariance inflation step.
    grown = Gaussian(prior.mean, prior.covariance*1.01)
    with pytest.raises(ValueError, match='body covariance contraction'):
        retain_goal_components(retained, prior, grown, prior)


@pytest.fixture(scope='module')
def marker_and_imu(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('fixed_switch_goal'), imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    observations = yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations']
    return scene, inputs, observations


def test_detected_goal_capture_is_unchanged_by_missing_images_commands_and_imu(marker_and_imu, monkeypatch):
    scene, inputs, observations = marker_and_imu
    inference = MovingInference(scene.robot_description, observations)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Fixed-goal inference must not use Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    first = inference.update(inputs, 0)
    assert first.records['fixed_goal_memory_event'] == 'marker_observation_update'
    memory = deepcopy(first.records['fixed_goal_memory'])
    assert memory['available'] and len(memory['goal_components']) >= 2
    initial_goals = first.stages[-1].goal_components
    initial_body = first.stages[-1].components[0].state
    assert memory['observation_stamp'] == 1000.
    assert memory['observation_command'] == list(inputs.command)
    assert len(memory['observation_ids']) == 1
    command = inputs.command.copy()
    current = first
    for index in range(1, 5):
        command = command+np.array([.001, -.002, .001, 0., .001, -.001])
        missing = replace(inputs, image=np.full_like(inputs.image, 255),
            command=command, stamp=1000.+.05*index, time=.05*index)
        current = inference.update(missing, index)
        assert current.records['fixed_goal_memory_event'] == 'image_missing_hold'
        assert current.records['fixed_goal_memory'] == memory
        assert len(current.stages[-1].components) == 1
        for previous, actual in zip(initial_goals, current.stages[-1].goal_components):
            assert actual.weight == previous.weight and actual.log_weight == previous.log_weight
            assert actual.state.transforms[-1] is previous.state.transforms[-1]
            np.testing.assert_array_equal(actual.state.covariance[60:, 60:],
                                          previous.state.covariance[60:, 60:])
            np.testing.assert_array_equal(actual.state.covariance[:60, :60],
                current.stages[-1].components[0].state.covariance[:60, :60])
        assert inference.update(missing, index) is current
    assert not np.array_equal(initial_body.covariance[:60, :60],
        current.stages[-1].components[0].state.covariance[:60, :60])
    # A new valid image is a new capture event rather than an image-loss update.
    reacquired = inference.update(replace(inputs, stamp=1000.25, time=.25), 5)
    updated_memory = reacquired.records['fixed_goal_memory']
    assert reacquired.records['fixed_goal_memory_event'] == 'marker_observation_update'
    assert updated_memory['observation_stamp'] == 1000.25
    assert updated_memory['observation_version'] > memory['observation_version']
    assert updated_memory['observation_ids'] != memory['observation_ids']


def test_missing_image_before_detection_is_labelled_as_unobserved_prior(marker_and_imu):
    scene, inputs, observations = marker_and_imu
    inference = MovingInference(scene.robot_description, observations)
    result = inference.update(replace(inputs, image=np.full_like(inputs.image, 255)), 0)
    assert result.records['fixed_goal_memory_event'] == 'unobserved_prior'
    assert result.records['fixed_goal_memory']['available'] is False
    assert inference.fixed_goal_memory is None
