"""Continuous-density K selection retains information and evaluates every move."""

from types import SimpleNamespace

import numpy as np
import pytest

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


def context(log_density, *, settings=None, dimension=2):
    identifier = object.__new__(StiffnessIdentifier)
    identifier.settings = dict(settings or {})
    identifier.initial_components = (PosteriorComponent('initial', Gaussian(
        np.zeros(dimension), np.eye(dimension))),)
    identifier.components = identifier.initial_components
    identifier.factors = (SimpleNamespace(factor_id='factor:1'),)
    identifier._branch_anchor = None
    # This fixture substitutes a finite equilibrium response for mechanics;
    # it does not perform or claim a real auxiliary numerical audit.
    identifier.last_equilibrium_audit = None
    calls, predictions = [], []

    def density(x):
        calls.append(np.array(x, copy=True))
        return log_density(x)

    def prediction(x, state):
        predictions.append((np.array(x, copy=True), state.command))
        equilibrium = np.array(state.command)+.1*np.exp(-np.asarray(x))
        return None, None, equilibrium, np.zeros(dimension, dtype=bool)

    identifier.log_density = density
    identifier._prediction = prediction
    result = next_result(identifier, 1., command=(0.,)*dimension)
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
    # Winning against fixed starts is not stationarity of the new density.
    # The previous point remains the best start but must still follow its score.
    np.testing.assert_array_equal(selection['base_logK'], first['selected_logK'])
    np.testing.assert_array_equal(selection['base_density_score'],
                                  density(np.array(first['selected_logK']))[1])
    assert np.max(np.abs(selection['base_density_score'])) > selection['gradient_tolerance']
    assert not selection['local_refinement_skipped']
    assert selection['iterations'] == 1 and selection['line_search']
    assert not np.array_equal(second['selected_logK'], first['selected_logK'])
    assert not np.array_equal(second['selected_K'], first['selected_K'])
    assert second['selected_log_unnormalized_density'] > density(np.array(first['selected_logK']))[0]
    np.testing.assert_array_equal(selection['selected_density_score'],
                                  density(np.array(second['selected_logK']))[1])
    assert selection['density_evaluation_count'] == len(calls)
    assert len(calls) > len(identifier._proposal_candidates)+1


def test_new_measurement_moves_previous_best_K_and_improves_full_prior_plus_likelihood():
    measurements = [.3]
    variance = .25

    def full_density(x):
        # A normalized unit Gaussian prior and all normalized observation
        # likelihoods: the new measurement changes the continuous posterior.
        residuals = x[0]-np.array(measurements)
        value = -.5*(np.log(2*np.pi)+x[0]**2)
        value -= .5*np.sum(np.log(2*np.pi*variance)+residuals**2/variance)
        score = -x[0]-np.sum(residuals)/variance
        return float(value), np.array([score])

    identifier, result, _, _ = context(full_density, dimension=1)
    prior, factors = identifier.initial_components, identifier.factors
    first = identifier.command_proposal(result)
    measurements.append(.5)
    identifier.factors += (SimpleNamespace(factor_id='factor:2'),)
    complete_factors = identifier.factors
    next_observation = next_result(identifier, 2., command=(0.,))
    second = identifier.command_proposal(next_observation)
    selection = identifier.last_record['proposal_selection']
    np.testing.assert_array_equal(selection['base_logK'], first['selected_logK'])
    assert selection['base_candidate_index'] == selection['candidate_count']-1
    assert not selection['local_refinement_skipped']
    assert selection['iterations'] == 1 and selection['line_search'][-1]['accepted']
    assert second['selected_K'][0] > first['selected_K'][0]
    assert second['selected_log_unnormalized_density'] > full_density(np.array(first['selected_logK']))[0]
    assert second['selected_log_unnormalized_density'] == full_density(np.array(second['selected_logK']))[0]
    analytic_mode = sum(measurements)/(variance+len(measurements))
    assert abs(second['selected_logK'][0]-analytic_mode) < abs(first['selected_logK'][0]-analytic_mode)
    assert identifier.initial_components is prior
    assert identifier.factors is complete_factors and identifier.factors[:1] == factors
    assert second['density_id'] == identifier.last_record['posterior']['density_id']
    assert second['source_command'] == list(next_observation.prior.command)
    assert selection['factor_ids'] == ['factor:1', 'factor:2']


def test_new_factor_at_stationary_previous_point_stops_by_score_not_fixed_start_rank():
    precision = [1.]
    identifier, result, _, predictions = context(
        lambda x: (-.5*precision[0]*float(x@x), -precision[0]*x))
    first = identifier.command_proposal(result)
    identifier.factors += (SimpleNamespace(factor_id='factor:2'),)
    precision[0] = 2.
    predictions.clear()
    second = identifier.command_proposal(next_result(identifier, 2.))
    selection = identifier.last_record['proposal_selection']
    assert not selection['retained_unchanged_density']
    assert not selection['local_refinement_skipped']
    assert selection['convergence_claimed'] and selection['iterations'] == 0
    assert selection['termination_reason'] == 'full-density score satisfies configured tolerance'
    assert selection['line_search'] == []
    assert len(predictions) == 1
    np.testing.assert_array_equal(second['selected_K'], first['selected_K'])


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
    assert selection['local_refinement_trigger'] == 'new density factor; refine the best evaluated stable start'
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


def _dense_seed_reference(identifier, result):
    """Independent old eager ordering; no line search for these seed tests.

    Resolve every start's current equilibrium before ranking. An exactly
    tied stable previous representative overrides the first stable maximum.
    """
    previous = getattr(identifier, '_proposal_representative', None)
    candidates = list(identifier._proposal_candidates)
    if previous is not None:
        candidates.append(previous['logK'])
    evaluated, cache = [], {}
    for index, x in enumerate(candidates):
        key = x.tobytes()
        try:
            if key not in cache:
                density, score = identifier.log_density(x)
                if not np.isfinite(density) or not np.all(np.isfinite(score)):
                    raise ValueError('Invalid density')
                saved = identifier._branch_anchor
                try:
                    identifier._branch_anchor = None
                    _, _, equilibrium, active = identifier._prediction(x, result.prior)
                finally:
                    identifier._branch_anchor = saved
                cache[key] = dict(logK=x.copy(), density=float(density),
                    score=np.array(score, copy=True), equilibrium=np.array(equilibrium, copy=True),
                    active=np.array(active, copy=True))
            evaluated.append((cache[key], index))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    if not evaluated:
        return None
    best = max(evaluated, key=lambda item:item[0]['density'])
    old = next((item for item in evaluated if previous is not None
                and item[1] == len(candidates)-1), None)
    if old is not None and best[0]['density'] <= old[0]['density']:
        return old
    return best


@pytest.mark.parametrize('case', ['unique', 'ties', 'invalid_density',
    'unstable_highest', 'all_unstable', 'previous_tied', 'previous_unstable'])
def test_density_ranked_equilibria_match_dense_selection_bitwise(case):
    dimension = 6
    centre = np.zeros(dimension)
    previous_x = np.full(dimension, .2)
    candidates = [centre]
    for axis in np.eye(dimension):
        candidates.extend((axis, -axis))
    candidates.extend((np.ones(dimension), -np.ones(dimension)))

    def density(x):
        if case == 'invalid_density' and (np.array_equal(x, centre) or x[0] == 1.):
            return np.nan, np.zeros(dimension)
        if case in ('ties', 'previous_tied', 'previous_unstable'):
            return 7.25, np.zeros(dimension)
        return -float(x@x), -2.*x

    def make():
        identifier, result, calls, predictions = context(density,
            settings=dict(representative_max_iterations=0), dimension=dimension)
        identifier._proposal_candidates = tuple(item.copy() for item in candidates)
        identifier._branch_anchor = object()
        original_prediction = identifier._prediction
        def prediction(x, state):
            # Production isolates every current-command stability query from
            # the retained factor's frozen continuation anchor.
            assert identifier._branch_anchor is None
            value = original_prediction(x, state)
            if (case == 'all_unstable'
                    or case == 'unstable_highest' and np.array_equal(x, centre)
                    or case == 'previous_unstable' and np.array_equal(x, previous_x)):
                raise ValueError('No current stable equilibrium at this candidate')
            return value
        identifier._prediction = prediction
        if case in ('previous_tied', 'previous_unstable'):
            identifier._proposal_representative = dict(factor_ids=('old-factor',),
                logK=previous_x.copy(), density=0., score=np.zeros(dimension),
                candidate_index=2, step_scale=.5)
        return identifier, result, calls, predictions

    dense, dense_result, dense_calls, dense_predictions = make()
    expected = _dense_seed_reference(dense, dense_result)
    lazy, result, calls, predictions = make()
    original_factors = lazy.factors
    original_prior = lazy.initial_components
    original_anchor = lazy._branch_anchor
    proposal = lazy.command_proposal(result)
    selection = lazy.last_record['proposal_selection']
    assert lazy.factors is original_factors and lazy.initial_components is original_prior
    assert lazy._branch_anchor is original_anchor
    assert selection['density_evaluation_count'] == len(calls) == len(dense_calls)
    assert selection['current_equilibrium_evaluation_count'] == len(predictions)
    assert selection['density_ranked_count'] == len(candidates)+(case in ('previous_tied', 'previous_unstable'))-sum(
        not np.isfinite(density(x)[0]) for x in candidates)
    if expected is None:
        assert proposal is None and lazy.last_record['proposal_status'] == 'unavailable'
        assert len(predictions) == len(dense_predictions) == 15
        assert len(selection['unavailable_candidates']) == 15
        return
    item, index = expected
    assert proposal['selected_candidate_index'] == selection['base_candidate_index'] == index
    np.testing.assert_array_equal(proposal['selected_logK'], item['logK'])
    np.testing.assert_array_equal(proposal['selected_K'], np.exp(item['logK']))
    np.testing.assert_array_equal(selection['selected_density_score'], item['score'])
    np.testing.assert_array_equal(proposal['auxiliary_equilibrium'], item['equilibrium'])
    np.testing.assert_array_equal(proposal['auxiliary_active_joint_limits'], item['active'])
    np.testing.assert_array_equal(proposal['command'],
        2.*np.array(result.prior.command)-item['equilibrium'])
    assert proposal['selected_log_unnormalized_density'] == item['density']
    assert selection['lower_ranked_equilibrium_checks_skipped'] > 0
    assert len(predictions) == (2 if case in ('unstable_highest', 'previous_unstable') else 1)
    if case == 'unique':
        assert len(dense_predictions) == 15 and len(predictions) == 1
    if case == 'previous_tied':
        assert selection['local_refinement_skipped']
        assert selection['refinement_skip_reason'] == 'configured zero representative-refinement iteration budget'
        assert len(dense_predictions) == 16 and len(predictions) == 1


def test_duplicate_previous_start_keeps_old_tie_precedence_without_an_extra_solve():
    identifier, result, calls, predictions = context(lambda x:(0., np.zeros(2)),
        settings=dict(representative_max_iterations=0))
    identifier._proposal_candidates = (np.zeros(2), np.ones(2))
    identifier._proposal_representative = dict(factor_ids=('old-factor',),
        logK=np.ones(2), density=0., score=np.zeros(2), candidate_index=0, step_scale=.5)
    proposal = identifier.command_proposal(result)
    assert proposal['selected_candidate_index'] == 2
    np.testing.assert_array_equal(proposal['selected_logK'], np.ones(2))
    assert len(calls) == 2 and len(predictions) == 1
    selection = identifier.last_record['proposal_selection']
    assert selection['local_refinement_skipped']
    assert selection['refinement_skip_reason'] == 'configured zero representative-refinement iteration budget'
    assert selection['density_ranked_count'] == 3
    assert selection['current_equilibrium_candidate_checks'] == 1
