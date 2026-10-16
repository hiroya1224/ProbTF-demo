"""One internal state, full joint moments, and unpruned conditional goals."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from probtf.spherical_law.first_order import reexpress_state
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.body_projection import project_update, retain_goal_components
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


@pytest.fixture(scope='module')
def common_state(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('joint_projection'))
    inference = MovingInference(scene.robot_description, ())
    return inference.prior_at(scene.initial_command, 1000., 0)


def _result(reference, weights=(.3, .7), logs=None, *, identity=False):
    raw, displayed = [], []
    for index, weight in enumerate(weights):
        mean = np.zeros(reference.mean.size)
        covariance = reference.covariance.copy()
        if not identity:
            mean[3], mean[15] = ((.012, -.007) if index == 0 else (-.009, .013))
            mean[60], mean[64] = ((.03, -.005) if index == 0 else (-.01, .008))
            covariance *= .7 if index == 0 else .8
            covariance[3, 33] = covariance[33, 3] = .15*np.sqrt(covariance[3, 3]*covariance[33, 33])
            covariance[3, 60] = covariance[60, 3] = .05*np.sqrt(covariance[3, 3]*covariance[60, 60])
        log = float(np.log(weight)) if logs is None else logs[index]
        label = reference.component_id if identity else f'raw:{index}'
        child = PosteriorComponent(label, Gaussian(mean, covariance),
                                   weight, log_weight=log)
        actual = reference if identity else reexpress_state(reference, mean, covariance,
            version=reference.version+2, component_id=child.label)
        raw.append(child)
        displayed.append(WeightedState(child.label, weight, actual, log))
    branch = FrozenBodyBranch(reference.component_id, 1., reference,
        Gaussian(reference.mean, reference.covariance), tuple(raw), 0.)
    return SimpleNamespace(frozen_body_branches=(branch,), marker_path=(('marker', False),),
        stages=(Stage('fused', tuple(displayed)),))


def test_internal_full_joint_moments_include_between_and_all_cross_covariances(common_state):
    source = _result(common_state)
    projection = project_update(source)
    children = source.frozen_body_branches[0].posterior_components
    mean = sum(child.weight*child.distribution.mean for child in children)
    within = sum(child.weight*child.distribution.covariance for child in children)
    between = sum(child.weight*np.outer(child.distribution.mean-mean,
        child.distribution.mean-mean) for child in children)
    np.testing.assert_allclose(projection.posterior.mean, mean, atol=1e-17)
    np.testing.assert_allclose(projection.posterior.covariance, within+between, atol=1e-17)
    np.testing.assert_allclose(projection.record['raw_within_covariance'], within, atol=1e-17)
    np.testing.assert_allclose(projection.record['raw_between_covariance'], between, atol=1e-17)
    assert projection.posterior.mean.size == 66
    assert projection.posterior.covariance[3, 33] != 0.
    assert projection.posterior.covariance[3, 60] != 0.
    assert projection.state.covariance[3, 60] != 0.
    expected = reexpress_state(common_state, mean, within+between,
        version=projection.state.version, component_id=projection.state.component_id)
    np.testing.assert_allclose(expected.covariance, projection.state.covariance, atol=1e-17)
    assert projection.state.component_id == 'internal:3'
    assert projection.record['source_component_count'] == 2


def test_every_goal_retains_its_conditional_marker_law_and_the_common_body(common_state):
    source = _result(common_state)
    projection = project_update(source)
    for raw, goal, details in zip(source.frozen_body_branches[0].posterior_components,
                                  projection.goal_components,
                                  projection.record['conditional_transformations']):
        p = raw.distribution.covariance
        regression = p[60:, :60] @ covariance_support(p[:60, :60]).inverse
        residual = p[60:, 60:]-regression @ p[:60, 60:]
        body_covariance = projection.posterior.covariance[:60, :60]
        mean = np.array(details['retargeted_raw_mean'])
        covariance = np.array(details['retargeted_raw_covariance'])
        np.testing.assert_allclose(mean[60:], raw.distribution.mean[60:]+regression @ (
            projection.posterior.mean[:60]-raw.distribution.mean[:60]), atol=1e-16)
        np.testing.assert_array_equal(mean[:60], projection.posterior.mean[:60])
        np.testing.assert_array_equal(covariance[:60, :60], body_covariance)
        np.testing.assert_allclose(covariance[:60, 60:], body_covariance @ regression.T, atol=1e-16)
        np.testing.assert_allclose(covariance[60:, 60:],
            residual+regression @ body_covariance @ regression.T, atol=1e-16)
        np.testing.assert_array_equal(goal.state.covariance[:60, :60], projection.state.covariance[:60, :60])
        for internal_edge, goal_edge in zip(projection.state.transforms[:10], goal.state.transforms[:10]):
            np.testing.assert_array_equal(internal_edge.translation, goal_edge.translation)
            if internal_edge.bingham is not None:
                np.testing.assert_array_equal(internal_edge.bingham, goal_edge.bingham)
        assert goal.label == raw.label
        assert goal.weight == raw.weight and goal.log_weight == raw.log_weight
    assert projection.record['original_goal_marginal_unchanged_claimed'] is False


def test_single_component_and_empty_observations_keep_actual_state_identity(common_state):
    for identity in (False, True):
        source = _result(common_state, weights=(1.,), identity=identity)
        projection = project_update(source)
        assert projection.state is source.stages[-1].components[0].state
        assert projection.posterior is source.frozen_body_branches[0].posterior_components[0].distribution
        assert projection.goal_components[0] is source.stages[-1].components[0]
        if identity:
            assert projection.state is common_state
        assert projection.record['single_component_identity']


def test_underflowed_positive_mode_and_fixed_attachments_survive_without_mc(common_state, monkeypatch):
    source = _result(common_state, weights=(1., 0.), logs=(0., -1000.))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Projection must use analytic moments rather than Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    projection = project_update(source)
    assert len(projection.goal_components) == 2
    assert projection.goal_components[1].weight == 0.
    assert projection.goal_components[1].log_weight == -1000.
    assert [item['log_weight'] for item in projection.record['sources']] == [0., -1000.]
    assert not projection.record['component_pruning_used']
    for edge in ('tool0', 'press_rod', 'rod_tip', 'camera_optical_frame'):
        i = projection.state.edge_ids.index(edge)
        np.testing.assert_array_equal(projection.state.covariance[6*i:6*i+6], 0.)
        for goal in projection.goal_components:
            np.testing.assert_array_equal(goal.state.covariance[6*i:6*i+6], 0.)


def test_image_dropout_preserves_all_previous_modes_and_applies_only_updated_body(common_state):
    projection = project_update(_result(common_state))
    prior = replace(projection.state, version=4, stamp=1000.05)
    goals = tuple(replace(component, state=replace(component.state, version=4, stamp=1000.05))
                  for component in projection.goal_components)
    unchanged = Gaussian(prior.mean, prior.covariance)
    retained = retain_goal_components(goals, prior, unchanged, prior)
    assert all(actual is expected for actual, expected in zip(retained, goals))
    mean = np.zeros(66)
    mean[3], mean[15] = .002, -.001
    updated = Gaussian(mean, prior.covariance*.9)
    effective = reexpress_state(prior, updated.mean, updated.covariance,
        version=6, component_id='internal:6')
    retained = retain_goal_components(goals, prior, updated, effective)
    assert len(retained) == len(goals)
    for old, current in zip(goals, retained):
        assert (current.label, current.weight, current.log_weight) == (old.label, old.weight, old.log_weight)
        np.testing.assert_array_equal(current.state.covariance[:60, :60], effective.covariance[:60, :60])
        assert current.state.version == effective.version
        assert current.state.stamp == effective.stamp and current.state.command == effective.command
    # A missing image does not invent a new marker observation from the
    # internal marker mean; only the updated body enters each retained E|B.
    altered_mean = updated.mean.copy()
    altered_mean[60:] = 100.
    altered = retain_goal_components(goals, prior, Gaussian(altered_mean, updated.covariance), effective)
    assert [item.state.state_id for item in altered] == [item.state.state_id for item in retained]


def test_foreign_raw_parameters_and_noncommon_retained_body_are_rejected(common_state):
    source = _result(common_state)
    wrong_covariance = source.stages[-1].components[0].state.covariance.copy()
    wrong_covariance[3, 9] += 1e-6
    wrong_covariance[9, 3] += 1e-6
    foreign = replace(source.stages[-1].components[0], state=replace(
        source.stages[-1].components[0].state, covariance=wrong_covariance))
    wrong = SimpleNamespace(frozen_body_branches=source.frozen_body_branches,
        marker_path=source.marker_path, stages=(Stage('fused', (foreign,)+source.stages[-1].components[1:]),))
    with pytest.raises(ValueError, match='do not reexpress'):
        project_update(wrong)
    projection = project_update(source)
    covariance = projection.goal_components[0].state.covariance.copy()
    covariance[3, 9] += 1e-6
    covariance[9, 3] += 1e-6
    bad_goal = replace(projection.goal_components[0], state=replace(
        projection.goal_components[0].state, covariance=covariance))
    with pytest.raises(ValueError, match='bit-identical body marginals'):
        retain_goal_components((bad_goal,)+projection.goal_components[1:],
            projection.state, Gaussian(projection.state.mean, projection.state.covariance), projection.state)
