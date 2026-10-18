"""A K batch retains the mechanical query at every acquisition command."""
from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent, condition_gaussian
from probtf_observation_fusion_demo.model import DemoResult, FrozenBodyBranch
from probtf_observation_fusion_demo.async_stiffness import (
    PhysicalActionEngine, capture_direct_stiffness_action, capture_stiffness_packet,
    reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.stiffness_bayes_factor import update_hazard_log_probabilities
from probtf_observation_fusion_demo.stiffness_identification import (
    ChartedBodyDensityRatio, FrozenStiffnessFactor, FrozenTimeIndexedStiffnessFactor,
)
import probtf_observation_fusion_demo.stiffness_identification as stiffness_module
from probtf_observation_fusion_demo.joint_batch_fusion import add_log_evidence_parts
from test_joint_batch_stiffness import arm, batch_inputs, identifier


def prefix_specifications(state, result, events):
    prior = Gaussian(state.mean, state.covariance)
    specs = []
    for index, identity in enumerate(result.records['observation_ids']):
        event = next(item for item in events if item['event_id'] == identity)
        reference = replace(state, stamp=event['stamp'], command=tuple(event['command']))
        h = np.zeros((1, state.mean.size)); h[0, 10] = 1.
        posterior = condition_gaussian(prior, np.array([.003+.001*index]),
            np.array([[.0007]]), h)[0]
        child = PosteriorComponent('prefix:'+identity, posterior)
        branch = FrozenBodyBranch(reference.component_id, 1., reference, prior, (child,))
        specs.append(dict(observation_id=identity, source_stamp=event['stamp'],
            source_command=list(event['command']), branches=(branch,), posterior_branch=None,
            provenance=dict(prefix_index=index, raw_hash=event['payload_hash'])))
        prior = posterior
    return specs


def test_prefix_bias_is_kept_without_reexpression_and_evidence_runs_once(arm, monkeypatch):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2])
    specs = prefix_specifications(arm[2], result, events)
    # The second prior is the preceding full Gaussian, not a recentered
    # approximation and not the initial reference covariance.
    assert specs[1]['branches'][0].prior.mean[10] != 0.
    assert not np.array_equal(specs[1]['branches'][0].prior.covariance,
                              specs[1]['branches'][0].reference_state.covariance)
    integrals = []
    actual = estimate.stiffness_bfs.integrator
    def count(query):
        integrals.append(query)
        return actual(query)
    estimate.stiffness_bfs.integrator = count
    monkeypatch.setattr(estimate, 'command_proposal',
                        lambda _: pytest.fail('No per-observation command proposal is allowed.'))
    record = estimate.update_batch(result, events=events, command_ledger=ledger,
        batch_id='time-indexed:a', event_body_ratios=specs)
    assert record['update_applied'], record.get('failure_reason')
    assert len(estimate.factors) == len(estimate.stiffness_bfs.updates) == 1
    factor = estimate.factors[0]
    assert isinstance(factor, FrozenTimeIndexedStiffnessFactor)
    assert len(factor.parts) == 3 and len(integrals) == 1
    assert [part.states[0].stamp for part in factor.parts] == [item['source_stamp'] for item in specs]
    assert factor.observation_ids == tuple(result.records['observation_ids'])
    np.testing.assert_array_equal(factor.parts[1].ratio.branches[0].prior.mean,
                                  specs[1]['branches'][0].prior.mean)
    assert not record['stiffness_batch_factorization']['endpoint_body_ratio_used_as_K_factor']
    assert not record['time_indexed_body_factors'][0]['body_marginal']['repeated_full_matrix_JSON_expansion']
    assert record['time_indexed_body_factors'][0]['prefix_source']['raw_hash'] == events[1]['payload_hash']
    assert estimate.update_batch(result, events=events, command_ledger=ledger,
        batch_id='time-indexed:a', event_body_ratios=specs) is record
    assert len(integrals) == 1


@pytest.mark.parametrize('change', ['command', 'stamp', 'missing', 'reverse'])
def test_time_indexed_sources_cannot_be_retimed_or_omitted(arm, change):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2])
    specs = prefix_specifications(arm[2], result, events)
    if change == 'command':
        specs[0]['source_command'][1] += .01
    elif change == 'stamp':
        specs[0]['source_stamp'] += .01
    elif change == 'missing':
        specs.pop()
    else:
        specs.reverse()
    with pytest.raises(ValueError, match='acquisition|effective'):
        estimate.update_batch(result, events=events, command_ledger=ledger,
            batch_id='bad-source', event_body_ratios=specs)
    assert not estimate.factors and not estimate.stiffness_bfs.updates


def test_production_joint_batch_cannot_silently_use_endpoint_only_ratio(arm):
    estimate = identifier(arm)
    result, events, ledger = batch_inputs(arm[2])
    result.records['joint_batch'] = dict(batch_id='real-production-batch')
    with pytest.raises(ValueError, match='endpoint-only fallback is forbidden'):
        estimate.update_batch(result, events=events, command_ledger=ledger,
                              batch_id='real-production-batch')
    # Calling the central public method must not bypass the batch entry guard.
    with pytest.raises(ValueError, match='endpoint-only fallback is forbidden'):
        estimate.update(result)
    assert not estimate.factors and not estimate.stiffness_bfs.updates


def test_newer_endpoint_cannot_hide_an_acquisition_before_the_retained_frontier(arm):
    estimate = identifier(arm)
    first, events, ledger = batch_inputs(arm[2])
    estimate.update_batch(first, events=events, command_ledger=ledger, batch_id='first',
        event_body_ratios=prefix_specifications(arm[2], first, events))
    source = replace(arm[2], stamp=arm[2].stamp+1.)
    second, events, ledger = batch_inputs(source, 'late')
    # Raw order within this batch remains sorted and its endpoint is newer,
    # but the first newly named sensor reading belongs before the old frontier.
    events[0]['stamp'] = arm[2].stamp-.4
    events[1]['stamp'] = arm[2].stamp-.3
    specs = prefix_specifications(source, second, events)
    with pytest.raises(ValueError, match='explicit checkpoint replay is required'):
        estimate.update_batch(second, events=events, command_ledger=ledger,
            batch_id='late', event_body_ratios=specs)
    assert len(estimate.factors) == len(estimate.stiffness_bfs.updates) == 1


def test_small_finite_global_mixture_mass_update_is_not_an_exact_identity(arm):
    estimate = identifier(arm)
    source = arm[2]
    branches = []
    for index, displacement in enumerate((-.03, .03)):
        mean = source.mean.copy(); mean[10] = displacement
        prior = Gaussian(mean, source.covariance)
        weight = .5+(2e-13 if index else -2e-13)
        child = PosteriorComponent('child:'+str(index), prior, weight,
                                   log_weight=math.log(weight))
        branches.append(FrozenBodyBranch('parent:'+str(index), .5, source,
            prior, (child,), math.log(.5)))
    ratio = ChartedBodyDensityRatio(branches, estimate.body_edge_ids)
    assert not ratio.identity
    value = np.zeros(36); value[10] = .04
    jacobian = np.zeros((36, 6)); jacobian[10, 1] = 1.
    log_ratio, score = ratio.evaluate(((value, jacobian), (value, jacobian)))
    assert log_ratio != 0. and score[1] != 0.


def test_large_common_log_density_scale_preserves_small_mass_and_ratio_score(arm, monkeypatch):
    estimate = identifier(arm)
    source = arm[2]
    branches, components = [], []
    for index, weight in enumerate((.17, .83)):
        mean = source.mean.copy(); mean[10] = .03*index
        prior = Gaussian(mean, source.covariance)
        posterior_mean = mean.copy(); posterior_mean[10] -= .004
        child = PosteriorComponent('child:'+str(index), Gaussian(posterior_mean,
            .8*source.covariance), weight, log_weight=math.log(weight))
        branches.append(FrozenBodyBranch('parent:'+str(index), .5, source,
            prior, (child,), math.log(.5)))
        components.append(child)
    ratio = ChartedBodyDensityRatio(branches, estimate.body_edge_ids)
    value, jacobian = np.zeros(36), np.zeros((36, 6)); jacobian[10, 1] = 1.
    expected = ratio.evaluate(((value, jacobian), (value, jacobian)))
    mixture_before = stiffness_module._mixture_density(components, source.mean)
    actual_density = stiffness_module._density_parts
    def shifted(*args, **kwargs):
        parts, score = actual_density(*args, **kwargs)
        return add_log_evidence_parts(parts, (-1e20,)), score
    monkeypatch.setattr(stiffness_module, '_density_parts', shifted)
    shifted_ratio = ratio.evaluate(((value, jacobian), (value, jacobian)))
    shifted_mixture = stiffness_module._mixture_density(components, source.mean)
    assert shifted_ratio[0] == pytest.approx(expected[0], abs=2e-14)
    np.testing.assert_allclose(shifted_ratio[1], expected[1], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(shifted_mixture[1], mixture_before[1], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(shifted_mixture[3], mixture_before[3], rtol=2e-14, atol=2e-14)


def test_two_batch_full_history_properness_and_reanchoring_preserve_each_raw_factor(arm):
    estimate = identifier(arm)
    results = []
    for index in range(2):
        source = replace(arm[2], stamp=arm[2].stamp+index,
            command=tuple(np.asarray(arm[2].command)+np.array([0., .03*index, -.02*index, 0., 0., 0.])))
        result, events, ledger = batch_inputs(source, 'batch:'+str(index))
        specs = prefix_specifications(source, result, events)
        record = estimate.update_batch(result, events=events, command_ledger=ledger,
            batch_id='batch:'+str(index), event_body_ratios=specs)
        assert record['update_applied'], record.get('failure_reason')
        results.append(result)
        if not index:
            first = estimate.factors[0]
            first_parameters = [(part.factor_id, part.ratio.branches[0].prior.mean.tobytes(),
                part.ratio.branches[0].prior.covariance.tobytes()) for part in first.parts]
    assert len(estimate.factors) == len(estimate.stiffness_bfs.updates) == 2
    assert estimate.factors[0] is first
    assert first_parameters == [(part.factor_id, part.ratio.branches[0].prior.mean.tobytes(),
        part.ratio.branches[0].prior.covariance.tobytes()) for part in first.parts]
    assert len(estimate.observation_ids) == 6
    for factor in estimate.factors:
        assert factor.properness['log_factor_lower_bound'] == sum(
            part.properness['log_factor_lower_bound'] for part in factor.parts)
        assert factor.properness['log_factor_upper_bound'] == sum(
            part.properness['log_factor_upper_bound'] for part in factor.parts)
        # The full body cross remains in each unprojected prefix Gaussian.
        assert all(part.ratio.branches[0].prior.covariance[4, 10] != 0. for part in factor.parts)
    x = estimate.initial_components[0].distribution.mean+np.array([0., .02, -.03, .01, .04, -.02])
    actual = estimate.log_density(x)
    value, score = estimate.stiffness_bfs.initial(x)
    # Independently reconstruct the continuous Keep/Reset recurrence with all
    # six time-specific factors. Batch boundaries must not prune the history.
    for update, factor in zip(estimate.stiffness_bfs.updates, estimate.factors):
        keep, reset = update_hazard_log_probabilities(update)
        reset_value, reset_score = update.reset.evaluate(x)
        terms = np.array([keep+value, reset+update.keep_log_mass+reset_value-update.reset.log_mass])
        top = np.max(terms)
        mixed = top+np.log(np.exp(terms-top).sum())
        weights = np.exp(terms-mixed)
        value, score = mixed, weights[0]*score+weights[1]*reset_score
        parts = [estimate._factor_evaluation(part, x) for part in factor.parts]
        contribution = sum(item[0] for item in parts)
        assert factor.properness['log_factor_lower_bound'] <= contribution <= factor.properness['log_factor_upper_bound']
        value += contribution
        score += sum((item[1] for item in parts), np.zeros(6))
    assert actual[0] == pytest.approx(value, abs=1e-10)
    np.testing.assert_allclose(actual[1], score, rtol=1e-12, atol=1e-10)
    source_result = results[-1]
    proposal = estimate.command_proposal(source_result)
    assert proposal is not None
    source_result.records['stiffness'] = estimate.last_record
    _, inference, _ = arm
    result = DemoResult(source_result.prior, source_result.stages,
        np.empty((0, 0), np.uint8), np.empty((0, 0), np.uint8),
        inference.paths['links'], inference.paths['hand'], inference.paths['camera'],
        inference.paths['marker'], inference.paths['camera_marker'], inference.common['root_frame'],
        (), (), source_result.records)
    packet = capture_stiffness_packet(result)
    engine = PhysicalActionEngine(arm[0].urdf, inference.body_ids, estimate.settings)
    action = capture_direct_stiffness_action(engine, result, packet)
    future_stamp = result.prior.stamp+.2
    future_body = tuple(replace(item, state=replace(item.state, stamp=future_stamp))
                        for item in result.stages[-1].components)
    future = replace(result, prior=replace(result.prior, stamp=future_stamp),
        stages=(replace(result.stages[-1], components=future_body),))
    view, _ = reanchor_retained_stiffness(engine, future, action)
    assert view.records['stiffness']['captured_source_stamp'] == result.prior.stamp
    assert view.records['stiffness']['source_stamp'] == future_stamp
    assert packet.record['source_stamp'] == result.prior.stamp
    assert packet.record['posterior']['factor_ids'] == [factor.factor_id for factor in estimate.factors]
    assert estimate.factors[0] is first
    repeated = estimate.log_density(x)
    assert repeated[0] == actual[0]
    np.testing.assert_array_equal(repeated[1], actual[1])


def test_real_arm_queries_each_q_and_its_K_sensitivity_in_one_native_batch(arm, monkeypatch):
    estimate = identifier(arm)
    source = arm[2]
    commands = (np.asarray(source.command), np.asarray(source.command)
                +np.array([0., .25, -.20, .04, .15, .08]))
    parts = []
    for index, command in enumerate(commands):
        reference = replace(source, stamp=source.stamp+index,
            command=tuple(command), component_id='acquisition:'+str(index))
        prior = Gaussian(reference.mean, reference.covariance)
        mean = reference.mean.copy(); mean[10] += .006*(index+1)
        child = PosteriorComponent('child:'+str(index), Gaussian(mean, .85*prior.covariance))
        branch = FrozenBodyBranch(reference.component_id, 1., reference, prior, (child,))
        ratio = ChartedBodyDensityRatio((branch,), estimate.body_edge_ids)
        estimate._freeze_branch(reference, {})
        parts.append(FrozenStiffnessFactor('atomic:'+str(index), ('obs:'+str(index),),
            ratio, (reference,), estimate._branch_anchor,
            dict(verified=True, log_factor_lower_bound=-1000., log_factor_upper_bound=1000.)))
    batch = FrozenTimeIndexedStiffnessFactor(hashlib.sha256(b'actual-q-sensitive').hexdigest(),
        ('obs:0', 'obs:1'), tuple(parts), dict(verified=True))
    seen = []
    native = estimate._auxiliary_native.evaluate_batch
    def capture(logk, command, *args, **kwargs):
        seen.append(np.asarray(command).copy())
        return native(logk, command, *args, **kwargs)
    monkeypatch.setattr(estimate._auxiliary_native, 'evaluate_batch', capture)
    x = estimate.initial_components[0].distribution.mean+np.array([0., .01, -.015, .02, -.01, .01])
    value, gradient = estimate._factor_evaluation(batch, x)
    assert len(seen) == 1 and seen[0].shape == (2, 6)
    np.testing.assert_array_equal(seen[0], commands)
    atomic = [estimate._factor_evaluation(part, x) for part in parts]
    assert value == sum(item[0] for item in atomic)
    np.testing.assert_array_equal(gradient, sum((item[1] for item in atomic), np.zeros(6)))
    # This differentiates the mechanical equilibrium at both q_i. It is not
    # a finite set of candidate K values or a Monte Carlo integration.
    step = 1e-5
    for coordinate in (1, 2, 4):
        direction = np.eye(6)[coordinate]*step
        plus = estimate._factor_evaluation(batch, x+direction)[0]
        minus = estimate._factor_evaluation(batch, x-direction)[0]
        assert gradient[coordinate] == pytest.approx((plus-minus)/(2.*step), rel=2e-5, abs=2e-5)
    # Replacing both commands by the final command changes the likelihood
    # and its derivative. A single END FK cannot supply these two factors.
    wrong_value, wrong_gradient = 0., np.zeros(6)
    for part in parts:
        reference = replace(part.states[0], command=tuple(commands[-1]))
        estimate._branch_anchor = None
        prediction = estimate._prediction(x, reference)
        contribution, derivative = part.ratio.evaluate((prediction,))
        wrong_value += contribution; wrong_gradient += derivative
    assert abs(value-wrong_value) > 1e-4
    assert np.linalg.norm(gradient-wrong_gradient) > 1e-4
