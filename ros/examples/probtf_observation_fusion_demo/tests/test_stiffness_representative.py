"""Continuous-density K selection retains information and evaluates every move."""

from types import SimpleNamespace

import numpy as np

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


def context(log_density, *, settings=None):
    identifier = object.__new__(StiffnessIdentifier)
    identifier.settings = dict(settings or {})
    identifier.initial_components = (PosteriorComponent('initial', Gaussian(
        np.zeros(2), np.eye(2))),)
    identifier.components = identifier.initial_components
    identifier.factors = (SimpleNamespace(factor_id='factor:1'),)
    identifier._branch_anchor = None
    calls, predictions = [], []

    def density(x):
        calls.append(np.array(x, copy=True))
        return log_density(x)

    def prediction(x, state):
        predictions.append((np.array(x, copy=True), state.command))
        equilibrium = np.array(state.command)+.1*np.exp(-np.asarray(x))
        return None, None, equilibrium, np.zeros(2, dtype=bool)

    identifier.log_density = density
    identifier._prediction = prediction
    result = next_result(identifier, 1.)
    return identifier, result, calls, predictions


def next_result(identifier, stamp, command=(0., 0.)):
    prior = SimpleNamespace(command=command, stamp=stamp, state_id='prior:'+str(stamp))
    component = SimpleNamespace(state=SimpleNamespace(state_id='body:'+str(stamp)), log_weight=0.)
    identifier.last_record = dict(source_command=list(command), source_stamp=stamp,
        source_prior_state_id=prior.state_id,
        body_posterior_state_ids=[component.state.state_id],
        body_posterior_log_weights=[0.], posterior=dict(density_id='density:'+str(len(identifier.factors))))
    return SimpleNamespace(prior=prior, stages=(SimpleNamespace(components=(component,)),))


def test_coupled_starts_escape_joint_stop_plateau_and_accepted_step_is_evaluated(monkeypatch):
    def density(x):
        response = np.maximum(x-.5, 0.)
        residual = response-.75
        return -float(residual@residual), -2.*residual*(x > .5)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('K representatives must not sample the density.')

    identifier, result, calls, _ = context(density)
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    initial_record = identifier.last_record['posterior']
    initial_factors = identifier.factors
    proposal = identifier.command_proposal(result)
    selected = np.array(proposal['selected_logK'])
    old_starts = (np.zeros(2), np.array([1., 0.]), np.array([-1., 0.]),
                  np.array([0., 1.]), np.array([0., -1.]))
    assert proposal['selected_log_unnormalized_density'] > max(density(x)[0] for x in old_starts)
    np.testing.assert_array_equal(identifier.last_record['proposal_selection']['base_logK'], [1., 1.])
    assert np.all(selected > .5)
    assert any(np.array_equal(x, selected) for x in calls)
    assert proposal['selected_log_unnormalized_density'] == density(selected)[0]
    selection = identifier.last_record['proposal_selection']
    assert selection['iterations'] == 1 and not selection['convergence_claimed']
    assert selection['density_evaluation_count'] == len(calls)
    assert selection['selected_density_evaluated'] and not selection['global_mode_claimed']
    for trial in selection['line_search']:
        if trial['accepted']:
            assert trial['log_unnormalized_density'] >= trial['armijo_bound']
    assert identifier.factors is initial_factors
    assert identifier.last_record['posterior'] is initial_record


def test_unchanged_density_keeps_representative_exactly_and_only_resolves_current_equilibrium():
    def density(x):
        residual = x-1.25
        return -float(residual@residual), -2.*residual

    identifier, result, calls, predictions = context(density)
    first = identifier.command_proposal(result)
    count = len(calls)
    predictions.clear()
    moved = next_result(identifier, 2., command=(.01, -.02))
    second = identifier.command_proposal(moved)
    assert len(calls) == count
    assert len(predictions) == 1 and predictions[0][1] == moved.prior.command
    np.testing.assert_array_equal(second['selected_logK'], first['selected_logK'])
    assert second['selected_log_unnormalized_density'] == first['selected_log_unnormalized_density']
    selection = identifier.last_record['proposal_selection']
    assert selection['retained_representative_bitwise']
    assert selection['density_evaluation_count'] == 0
    assert selection['retained_unchanged_density']
    assert second['source_command'] == list(moved.prior.command)
    np.testing.assert_array_equal(second['auxiliary_equilibrium'],
        np.array(moved.prior.command)+.1*np.exp(-np.array(first['selected_logK'])))


def test_new_factor_reoptimizes_from_previous_evaluated_representative():
    target = np.array([1.25, 1.25])

    def density(x):
        residual = x-target
        return -float(residual@residual), -2.*residual

    identifier, result, calls, _ = context(density)
    first = identifier.command_proposal(result)
    identifier.factors += (SimpleNamespace(factor_id='factor:2'),)
    target[:] = [1.8, 1.3]
    calls.clear()
    moved = next_result(identifier, 2.)
    second = identifier.command_proposal(moved)
    assert any(np.array_equal(x, first['selected_logK']) for x in calls)
    assert second['selected_log_unnormalized_density'] >= density(np.array(first['selected_logK']))[0]
    selection = identifier.last_record['proposal_selection']
    assert not selection['retained_unchanged_density']
    assert selection['factor_ids'] == ['factor:1', 'factor:2']
    assert selection['candidate_count'] == len(identifier._proposal_candidates)+1
    # New evidence updates the complete density/score at x, even when x is
    # already the best start and no costly new nonlinear trial is needed.
    assert selection['local_refinement_skipped']
    assert selection['line_search'] == []
    np.testing.assert_array_equal(second['selected_logK'], first['selected_logK'])
    np.testing.assert_array_equal(selection['selected_density_score'],
                                  density(np.array(first['selected_logK']))[1])
    assert len(calls) == len(identifier._proposal_candidates)+1


def test_a_better_fixed_start_after_new_evidence_triggers_local_refinement():
    target = np.array([.2, .2])

    def density(x):
        residual = x-target
        return -float(residual@residual), -2.*residual

    identifier, result, _, _ = context(density)
    first = identifier.command_proposal(result)
    identifier.factors += (SimpleNamespace(factor_id='factor:2'),)
    target[:] = [1.4, 1.4]
    moved = next_result(identifier, 2.)
    second = identifier.command_proposal(moved)
    selection = identifier.last_record['proposal_selection']
    assert not selection['local_refinement_skipped']
    assert selection['local_refinement_trigger'].startswith('a prior-derived start improves')
    assert selection['iterations'] == 1 and selection['line_search']
    np.testing.assert_array_equal(selection['base_logK'], [1., 1.])
    assert second['selected_log_unnormalized_density'] >= density(np.ones(2))[0]
    assert second['selected_log_unnormalized_density'] > density(np.array(first['selected_logK']))[0]


def test_exhausted_line_search_retains_an_evaluated_start_and_reports_failure():
    def density(x):
        residual = x-.2
        return -float(residual@residual), -2.*residual

    identifier, result, _, _ = context(density,
        settings=dict(representative_max_backtracks=0))
    proposal = identifier.command_proposal(result)
    np.testing.assert_array_equal(proposal['selected_logK'], [0., 0.])
    assert not proposal['density_improvement_claimed']
    selection = identifier.last_record['proposal_selection']
    assert not selection['line_search'][0]['accepted']
    assert selection['termination_reason'].startswith('Armijo line search exhausted')
    assert not selection['convergence_claimed']
    assert proposal['selected_log_unnormalized_density'] == density(np.zeros(2))[0]


def test_configured_iteration_limit_and_exact_score_tolerance_are_respected():
    def density(x):
        residual = x-np.ones(2)
        return -float(residual@residual), -2.*residual

    identifier, result, _, _ = context(density,
        settings=dict(optimizer_max_iterations=2, representative_max_iterations=5,
                      optimizer_gradient_tolerance=1e-9))
    proposal = identifier.command_proposal(result)
    assert proposal['selected_log_unnormalized_density'] == 0.
    selection = identifier.last_record['proposal_selection']
    assert selection['maximum_iterations'] == 2
    assert selection['iterations'] == 0 and selection['convergence_claimed']
