"""Declared batch hazard, full joint ratio and continuous historical K law."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent, FusedMixture, condition_gaussian
from probtf_observation_fusion_demo.joint_batch_stiffness import (
    aggregate_event_hazard, make_joint_batch_stiffness_context,
)
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
from legacy_marker import legacy_marker_scene

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


@pytest.mark.parametrize('rho,count,expected', [(0., 100, 0.), (1., 0, 0.),
    (1., 3, 1.), (.1, 3, .271), (.25, 3, .578125)])
def test_batch_hazard_is_at_least_one_change_not_one_arbitrary_event(rho, count, expected):
    assert aggregate_event_hazard(rho, count) == pytest.approx(expected, abs=2e-16)


def test_small_hazard_preserves_probability_without_subtractive_cancellation():
    assert aggregate_event_hazard(1e-20, 100) == pytest.approx(1e-18, rel=1e-14)
    for probability, count in ((-.1, 2), (float('nan'), 1), (.1, -1), (.1, True)):
        with pytest.raises(ValueError):
            aggregate_event_hazard(probability, count)


@pytest.fixture(scope='module')
def arm(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('joint_batch_stiffness'))
    frame = scene.step(0., 1000.)
    inference = MovingInference(scene.robot_description, (), config_path=scene.legacy_inference_config)
    source = inference.prior_at(frame.command, frame.stamp, 0)
    covariance = source.covariance.copy()
    covariance[4, 10] = covariance[10, 4] = .2*np.sqrt(covariance[4, 4]*covariance[10, 10])
    source = reexpress_state(source, source.mean, covariance)
    return scene, inference, source


def batch_inputs(state, tag='a', empty=False):
    # Three actual factors of this analytic test jointly condition a single
    # full Gaussian; no intermediate single-state projection is performed.
    ids = () if empty else tuple(tag+':'+str(i) for i in range(3))
    prior = Gaussian(state.mean, state.covariance)
    h = np.zeros((3, state.mean.size)); h[:, 10] = 1.
    posterior = (prior if empty else condition_gaussian(prior,
        np.array([.001, -.002, .004]), np.eye(3)*.0007, h)[0])
    child = PosteriorComponent('joint:'+tag, posterior)
    displayed = reexpress_state(state, posterior.mean, posterior.covariance,
        version=state.version+1, component_id='joint:'+tag)
    branch = FrozenBodyBranch(state.component_id, 1., state, prior, (child,))
    result = SimpleNamespace(prior=state,
        frozen_body_prior=prior,
        frozen_body_posterior=FusedMixture((child,), (), ids, state.version, state.stamp, state.command),
        frozen_body_branches=(branch,), projected_body_numerator=branch,
        stages=(Stage('joint_batch', (WeightedState(child.label, 1., displayed),)),),
        records=dict(observation_ids=list(ids)))
    events = [dict(event_id=tag+':joint', kind='joint', stamp=state.stamp-.4,
        sequence=0, command=state.command, payload_hash=tag+':joint-hash')]
    for i, identity in enumerate(ids):
        events.append(dict(event_id=identity, kind='imu' if i != 1 else 'image',
            stamp=state.stamp-.3+.1*i, sequence=i+1, command=state.command,
            payload_hash=identity+':raw-hash'))
    events.append(dict(event_id=tag+':missing-image', kind='image', stamp=state.stamp,
        sequence=4, command=state.command, payload_hash=tag+':missing-hash'))
    ledger = dict(kind='actual test command ledger', source_stamp=state.stamp-.4,
        target_stamp=state.stamp, steps=[])
    return result, events, ledger


def identifier(arm):
    scene, inference, _ = arm
    return StiffnessIdentifier(scene.urdf, inference.body_ids, dict(
        prior_config_path=str(REPO/'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'),
        bayes_factor_surprise=dict(enabled=True, change_probability=.10,
            reset_measure='initial_log_stiffness', event_probability_unit='per_effective_sensor_event',
            evidence_method='laplace')))


def test_context_retains_all_raw_events_and_ledger_but_counts_only_effective_sensor_factors(arm):
    result, events, ledger = batch_inputs(arm[2])
    context = make_joint_batch_stiffness_context(result, events=events, command_ledger=ledger,
        probability=.1, batch_id='batch:a')
    record = context.record()
    assert record['acquired_event_count'] == 5
    assert record['effective_sensor_event_count'] == 3
    assert record['command_ledger'] == ledger
    assert record['stiffness_change_probability_batch'] == pytest.approx(.271)
    assert 'latent within-batch change times are not retained' in record['reset_time_approximation']
    assert not record['exact_sequential_hazard_recursion_preserved']
    assert not record['parameter_density_projection_applied']
    assert not record['intermediate_observation_Laplace_updates']
    ledger['steps'].append('later mutation')
    events[0]['command'] = [999.]*6
    assert context.record()['command_ledger']['steps'] == []
    assert context.record()['acquired_events'][0]['command'] == list(result.prior.command)


def test_full_joint_batch_adds_one_factor_and_one_hazard_not_one_per_sensor(arm, monkeypatch):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2])
    integral_calls = []
    original = estimate.stiffness_bfs.integrator
    def integral(density):
        integral_calls.append(density)
        return original(density)
    estimate.stiffness_bfs.integrator = integral
    # update_batch must not hide intermediate proposal solves.
    with monkeypatch.context() as patch:
        patch.setattr(estimate, 'command_proposal', lambda _: pytest.fail('intermediate command proposal'))
        record = estimate.update_batch(result, events=events, command_ledger=ledger, batch_id='batch:a')
    assert record['status'] == 'updated_density_factor', record.get('failure_reason')
    assert len(estimate.factors) == len(estimate.stiffness_bfs.updates) == 1
    assert estimate.factors[0].observation_ids == tuple(result.records['observation_ids'])
    assert estimate.stiffness_bfs.updates[0].rho0 == pytest.approx(.271)
    assert len(integral_calls) == 1  # first Keep and Reset are identical integrands
    assert record['bayes_factor_surprise']['identical_keep_reset_evidence_reused']
    assert record['bayes_factor_surprise']['metadata']['joint_batch']['acquired_event_count'] == 5
    # An actual body cross remains in the ratio; scalar per-link substitutes
    # would not reproduce these full frozen parameters.
    marginal = record['body_marginal']['branches'][0]
    assert marginal['prior_covariance'][4][10] != 0.
    assert marginal['components'][0]['covariance'][4][10] != 0.
    assert estimate.command_proposal(result) is not None
    again = estimate.update_batch(result, events=events, command_ledger=ledger, batch_id='batch:a')
    assert again is record and len(integral_calls) == 1


def test_following_batch_retains_every_old_continuous_factor_and_changes_no_reset_prior(arm):
    estimate = identifier(arm)
    first, events, ledger = batch_inputs(arm[2])
    a = estimate.update_batch(first, events=events, command_ledger=ledger, batch_id='batch:a')
    assert a['posterior'] is not None
    old_factor, reset = estimate.factors[0], estimate.stiffness_bfs.reset
    source = replace(first.stages[-1].components[0].state, stamp=first.prior.stamp+1.)
    second, events, ledger = batch_inputs(source, 'b')
    b = estimate.update_batch(second, events=events, command_ledger=ledger, batch_id='batch:b')
    assert b['posterior'] is not None, b.get('failure_reason')
    assert estimate.factors[0] is old_factor
    assert len(estimate.factors) == len(estimate.stiffness_bfs.updates) == 2
    assert estimate.stiffness_bfs.reset is reset
    assert all(u.reset is reset for u in estimate.stiffness_bfs.updates)
    assert estimate.observation_ids == frozenset(first.records['observation_ids']+second.records['observation_ids'])
    assert b['posterior']['global_moments'] is None
    assert not b['posterior']['bayes_factor_surprise']['factor_pruning_applied']
    assert b['joint_batch']['comparison_source_command'] == list(source.command)


def test_batch_identity_source_or_acquisition_order_change_cannot_silently_reuse_update(arm):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2])
    estimate.update_batch(result, events=events, command_ledger=ledger, batch_id='batch:a')
    with pytest.raises(ValueError, match='identity changed'):
        estimate.update_batch(result, events=events, command_ledger=dict(ledger, altered=True), batch_id='batch:a')
    with pytest.raises(ValueError, match='acquisition stamp and sequence order'):
        make_joint_batch_stiffness_context(result, events=events[::-1], command_ledger=ledger, probability=.1)
    context = make_joint_batch_stiffness_context(result, events=events, command_ledger=ledger, probability=.1)
    with pytest.raises(ValueError, match='exact frozen body comparison source'):
        context.validate_result(SimpleNamespace(prior=replace(result.prior, stamp=result.prior.stamp+1.), records=result.records))


def test_missing_observations_keep_identity_and_do_not_apply_batch_hazard(arm):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2], empty=True)
    record = estimate.update_batch(result, events=events, command_ledger=ledger, batch_id='empty')
    assert record['status'] == 'identity_density_ratio'
    assert not record['update_applied']
    assert record['joint_batch']['effective_sensor_event_count'] == 0
    assert record['joint_batch']['stiffness_change_probability_batch'] == 0.
    assert not estimate.factors and not estimate.stiffness_bfs.updates
