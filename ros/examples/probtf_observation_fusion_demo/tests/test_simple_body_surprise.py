"""Body-only change hypotheses retain the shared joint law and K ratio."""

import numpy as np
import pytest
from scipy.special import logsumexp

from probtf_observation_fusion_demo.bayes_factor import common_chart_gaussian
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from probtf_observation_fusion_demo.model import component_log_weight
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.simple_inference import SimpleInference
from test_simple_inference import acquired, events


def test_body_surprise_defaults_to_quarter_probability_and_has_explicit_switch(acquired):
    description, _, templates = acquired
    default = SimpleInference(description, templates)
    assert default.surprise_enabled and default.change_probability == .25
    assert default.surprise_settings['event_probability_unit'] == 'per_effective_sensor_event'
    disabled = SimpleInference(description, templates, body_surprise_enabled=False)
    assert not disabled.surprise_enabled and disabled.change_probability == .25
    enabled = SimpleInference(description, templates, body_surprise_enabled=True)
    assert enabled.surprise_enabled and enabled.change_probability == .25
    assert enabled.settings['body_bayes_factor_surprise'] == enabled.settings['bayes_factor_surprise']


def test_real_observation_hypotheses_have_correct_odds_and_one_full_joint_canonical_law(acquired, events, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No Monte Carlo is used by body surprise or its moment projection.')
    for name in ('normal', 'random', 'default_rng', 'multivariate_normal'):
        monkeypatch.setattr(np.random, name, forbidden)
    model = SimpleInference(acquired[0], acquired[2])
    initial = model.update_sensor_event(events('joint:initial', 'joint', frame_index=0))
    assert not initial.records['bayes_factor_surprise']['applied']
    result = model.update_sensor_event(events('image:valid', 'image', frame_index=0))
    surprise = result.records['bayes_factor_surprise']
    assert surprise['enabled'] and surprise['applied']
    assert surprise['change_probability'] == .25
    assert surprise['prior_change_odds'] == pytest.approx(1./3.)
    posterior_log_odds = (surprise['posterior_change_log_probability']
                          -surprise['posterior_keep_log_probability'])
    assert posterior_log_odds == pytest.approx(np.log(.25/.75)+surprise['log_bayes_factor'], abs=2.e-9)
    assert surprise['probability_unit'] == 'per_effective_sensor_event'
    assert len(result.stages[-1].components) == 1 and not result.stages[-1].goal_components
    assert len(result.frozen_body_branches) == 2
    assert len(result.frozen_body_posterior.components) == 1
    assert result.projected_body_numerator is not None
    assert result.records['simple_inference']['body_BFS_enabled']
    target = result.prior
    children = [(child, common_chart_gaussian(branch.reference_state, child.distribution, target))
                for branch in result.frozen_body_branches for child in branch.posterior_components]
    weights = np.exp([component_log_weight(child) for child, _ in children])
    assert logsumexp([component_log_weight(child) for child, _ in children]) == pytest.approx(0., abs=1.e-12)
    mean = sum(weight*law.mean for weight, (_, law) in zip(weights, children))
    within = sum(weight*law.covariance for weight, (_, law) in zip(weights, children))
    between = sum(weight*np.outer(law.mean-mean, law.mean-mean)
                  for weight, (_, law) in zip(weights, children))
    canonical = result.frozen_body_posterior.components[0].distribution
    np.testing.assert_allclose(canonical.mean, mean, atol=2.e-15, rtol=2.e-13)
    np.testing.assert_allclose(canonical.covariance, within+between, atol=2.e-15, rtol=2.e-13)
    assert result.records['internal_projection']['common_chart_materialization']
    assert not result.records['bayes_factor_transport']['reset_law_restricted_to_keep_prior_support']
    current = result.stages[-1].components[0].state
    snapshot = build_snapshot(result, result.stages[-1], model.common)
    assert {point.query.state_id for point in snapshot.points} == {current.state_id}
    assert model.components[0].state is current
    assert result.records['candidate_measure']['shape_source_state_ids'] == [current.state_id]
    assert result.records['candidate_measure']['bayes_factor_hypothesis_scale']['reset_kernel_preserves_total_mass']


def test_reset_keeps_environment_marginal_and_K_uses_actual_transition_denominator(acquired, events):
    model = SimpleInference(acquired[0], acquired[2])
    image = model.update_sensor_event(events('image:map', 'image', frame_index=0))
    result = model.update_sensor_event(events('imu:next', 'imu', stamp=1000.02, frame_index=1))
    assert result.records['bayes_factor_surprise']['applied']
    keep, reset = result.frozen_body_branches
    marker_index = result.prior.edge_ids.index(model.marker.edge_id)
    environment = np.arange(6*marker_index, 6*marker_index+6)
    body = np.array([i for i in range(result.prior.mean.size) if i not in environment])
    np.testing.assert_array_equal(keep.prior.covariance[np.ix_(environment, environment)],
                                  reset.prior.covariance[np.ix_(environment, environment)])
    np.testing.assert_array_equal(keep.reference_state.transforms[marker_index].translation,
                                  reset.reference_state.transforms[marker_index].translation)
    np.testing.assert_array_equal(keep.reference_state.transforms[marker_index].bingham,
                                  reset.reference_state.transforms[marker_index].bingham)
    assert np.linalg.norm(keep.prior.covariance[np.ix_(body, environment)]) > 0.
    np.testing.assert_array_equal(reset.prior.covariance[np.ix_(body, environment)],
                                  np.zeros((len(body), len(environment))))
    for branch in (keep, reset):
        for name in model.body_ids:
            index = branch.reference_state.edge_ids.index(name)
            np.testing.assert_array_equal(branch.prior.covariance[6*index:6*index+3],
                                          np.zeros((3, result.prior.mean.size)))
    numerator = result.projected_body_numerator
    assert numerator.reference_state is keep.reference_state
    assert numerator.prior is keep.prior
    ratio = JointRotationDensityRatio(result.frozen_body_branches,
                                      model.body_ids, posterior_branch=numerator)
    assert ratio.branches == result.frozen_body_branches
    assert ratio.posterior_branch is numerator
    assert [branch.weight for branch in ratio.branches] == [.75, .25]
    assert len(ratio.marginals[0][1]) == 1 and not ratio.marginals[1][1]
    child = numerator.posterior_components[0]
    assert child.weight == 1. and child.log_weight == 0.
    assert child.distribution is result.frozen_body_posterior.components[0].distribution
    assert result.records['fixed_goal_memory']['observation_stamp'] == image.prior.stamp
    assert not result.records['internal_projection']['conditional_goals_constructed']


def test_missing_disabled_and_duplicate_events_do_not_apply_extra_change(acquired, events, monkeypatch):
    from probtf_observation_fusion_demo import moving_inference
    model = SimpleInference(acquired[0], acquired[2])
    first = model.update_sensor_event(events('image:map', 'image', frame_index=0))
    previous = first.stages[-1].components[0].state
    missing_event = events('image:missing', 'image', stamp=1000.1, frame_index=1,
                           image=np.full_like(acquired[1].image, 255))
    missing = model.update_sensor_event(missing_event)
    assert not missing.records['bayes_factor_surprise']['applied']
    assert missing.records['bayes_factor_surprise']['posterior_change_probability'] is None
    assert len(missing.frozen_body_branches) == 1 and missing.projected_body_numerator is None
    np.testing.assert_array_equal(missing.stages[-1].components[0].state.covariance, previous.covariance)
    mass = model.candidate_measure.log_total_mass
    assert model.update_sensor_event(missing_event) is missing
    assert model.candidate_measure.log_total_mass == mass
    # The marker-only stream still carries IMU acquisition records, but an
    # unenabled sensor event has no body likelihood and no change update.
    marker_only = tuple(item for item in acquired[2] if item['sensor'] == 'marker')
    disabled_sensor = SimpleInference(acquired[0], marker_only)
    imu = disabled_sensor.update_sensor_event(events('imu:disabled', 'imu', stamp=1000.2, frame_index=2))
    assert not imu.records['bayes_factor_surprise']['applied']
    assert not imu.records['observation_ids']
    def forbidden(*args, **kwargs):
        raise AssertionError('A disabled body-BFS option must not construct transition priors.')
    monkeypatch.setattr(moving_inference, 'transition_priors', forbidden)
    disabled = SimpleInference(acquired[0], acquired[2], body_surprise_enabled=False)
    ordinary = disabled.update_sensor_event(events('image:ordinary', 'image', stamp=1000.3, frame_index=3))
    assert not ordinary.records['bayes_factor_surprise']['enabled']
    assert not ordinary.records['bayes_factor_surprise']['applied']
    assert len(ordinary.frozen_body_branches) == 1
    assert ordinary.projected_body_numerator is None
