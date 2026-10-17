"""Analytic one-dimensional oracles for the continuous stiffness recursion."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    ContinuousStiffnessBFS, LogIntegral, ResetMeasure,
)


def normal_density(mean=0., variance=1., log_scale=0.):
    def evaluate(x):
        value = np.asarray(x)[0]
        return (float(norm.logpdf(value, mean, math.sqrt(variance))+log_scale),
                np.array([-(value-mean)/variance]))
    return evaluate


def integrate_1d(density):
    # The runtime module deliberately does not choose this or any other rule.
    value, error = quad(lambda x: math.exp(density(np.array([x]))[0]),
                        -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
    return LogIntegral(math.log(value), 'deterministic 1-D QUADPACK test oracle',
                       diagnostics=dict(absolute_error_estimate=error))


def make_filter(*, rho=.1, initial_mean=0., initial_variance=1.):
    initial = normal_density(initial_mean, initial_variance)
    reset = ResetMeasure('initial-logK-law', initial,
                         description='explicitly selected initial continuous law')
    return ContinuousStiffnessBFS(initial, initial_source_id='initial-logK-law',
        rho0=rho, reset=reset, integrator=integrate_1d)


def gaussian_product(mean, variance, observed, noise_variance):
    evidence = float(norm.pdf(observed, mean, math.sqrt(variance+noise_variance)))
    posterior_variance = 1./(1./variance+1./noise_variance)
    posterior_mean = posterior_variance*(mean/variance+observed/noise_variance)
    return evidence, posterior_mean, posterior_variance


def test_recursive_density_and_score_match_all_analytic_reset_histories():
    filter = make_filter()
    components = [(1., 0., 1.)]
    cumulative_log_mass = 0.
    for event, (observed, variance) in enumerate(((.7, .8), (-.4, .6), (1.2, .9))):
        keep_evidence = sum(weight*gaussian_product(mean, cov, observed, variance)[0]
                            for weight, mean, cov in components)
        reset_evidence = gaussian_product(0., 1., observed, variance)[0]
        mixture_evidence = .9*keep_evidence+.1*reset_evidence
        # Independent expansion has one component for every possible latest
        # reset time. None of these components is selected or discarded.
        next_components = []
        for weight, mean, cov in components:
            evidence, mean, cov = gaussian_product(mean, cov, observed, variance)
            next_components.append((.9*weight*evidence/mixture_evidence, mean, cov))
        evidence, mean, cov = gaussian_product(0., 1., observed, variance)
        next_components.append((.1*evidence/mixture_evidence, mean, cov))
        record = filter.advance(str(event), ('obs-'+str(event),),
                                normal_density(observed, variance))
        assert record['rho0'] == .1
        assert record['prior_reset_odds'] == pytest.approx(1./9.)
        assert record['keep_evidence']['log_value'] == pytest.approx(math.log(keep_evidence), abs=2e-12)
        assert record['reset_evidence']['log_value'] == pytest.approx(math.log(reset_evidence), abs=2e-12)
        assert record['log_bayes_factor_reset_over_keep'] == pytest.approx(
            math.log(reset_evidence/keep_evidence), abs=2e-12)
        assert record['posterior_reset_mass'] == pytest.approx(
            .1*reset_evidence/mixture_evidence, abs=2e-12)
        assert record['hypothesis_scale']['keep_log_total_mass'] == pytest.approx(cumulative_log_mass, abs=3e-12)
        assert record['hypothesis_scale']['reset_log_total_mass'] == pytest.approx(cumulative_log_mass, abs=3e-12)
        cumulative_log_mass += math.log(mixture_evidence)
        assert record['posterior_log_mass'] == pytest.approx(cumulative_log_mass, abs=3e-12)
        components = next_components
        for x in (-2., -.3, .6, 2.):
            terms = np.array([math.log(weight)+norm.logpdf(x, mean, math.sqrt(cov))
                              for weight, mean, cov in components])
            normalized_log_density = float(logsumexp(terms))
            expected_score = sum(weight*(-(x-mean)/cov)
                for weight, (_, mean, cov) in zip(np.exp(terms-normalized_log_density), components))
            value, score = filter.evaluate([x])
            assert value == pytest.approx(normalized_log_density+cumulative_log_mass, abs=4e-12)
            np.testing.assert_allclose(score, [expected_score], atol=3e-12, rtol=1e-12)
        assert record['retained_event_count'] == event+1
    assert len(filter.updates) == 3
    assert filter.record()['factor_ids'] == ['0', '1', '2']
    assert not filter.record()['density_projection_applied']
    assert not filter.record()['factor_pruning_applied']


def test_zero_rho_is_the_literal_ordinary_product_without_any_integrals():
    initial = normal_density(.2, 1.3, log_scale=3.2)
    def forbidden(*args):
        raise AssertionError('rho=0 must need neither evidence nor reset law')
    filter = ContinuousStiffnessBFS(initial, initial_source_id='scaled-initial',
        initial_log_mass=3.2, rho0=0., integrator=forbidden)
    factors = [normal_density(.4, .7), normal_density(-.2, .9)]
    for index, factor in enumerate(factors):
        record = filter.advance(str(index), ('obs'+str(index),), factor)
        assert not record['hazard_applied']
        assert record['keep_evidence'] is record['reset_evidence'] is None
    for x in (-1., 0., 2.):
        expected_value, expected_score = initial(np.array([x]))
        for factor in factors:
            value, score = factor(np.array([x]))
            expected_value += value
            expected_score = expected_score+score
        actual_value, actual_score = filter.evaluate([x])
        assert actual_value == expected_value
        np.testing.assert_array_equal(actual_score, expected_score)


def test_branch_scale_invariance_and_reset_measure_is_explicit():
    baseline = make_filter()
    scaled_initial = normal_density(log_scale=7.)
    scaled_reset = ResetMeasure('scaled-reset', normal_density(log_scale=-4.), log_mass=-4.)
    scaled = ContinuousStiffnessBFS(scaled_initial, initial_source_id='scaled-initial',
        initial_log_mass=7., reset=scaled_reset, integrator=integrate_1d)
    for index, factor in enumerate((normal_density(.6, .5), normal_density(-.3, .8))):
        left = baseline.advance(str(index), (str(index),), factor)
        right = scaled.advance(str(index), (str(index),), factor)
        assert left['log_bayes_factor_reset_over_keep'] == pytest.approx(
            right['log_bayes_factor_reset_over_keep'], abs=3e-12)
        assert left['posterior_reset_mass'] == pytest.approx(right['posterior_reset_mass'], abs=1e-12)
        for x in (-1., .2, 1.):
            actual_value, actual_score = scaled.evaluate([x])
            expected_value, expected_score = baseline.evaluate([x])
            assert actual_value-scaled.log_mass == pytest.approx(expected_value-baseline.log_mass, abs=3e-12)
            np.testing.assert_allclose(actual_score, expected_score, atol=3e-12)
        assert scaled.log_mass-baseline.log_mass == pytest.approx(7., abs=3e-12)
    undeclared = ContinuousStiffnessBFS(normal_density(), initial_source_id='base')
    with pytest.raises(ValueError, match='explicitly declared reset'):
        undeclared.advance('factor', ('obs',), normal_density())


def test_no_observations_and_same_event_replay_are_bitwise_identity():
    filter = make_filter()
    def forbidden(*args):
        raise AssertionError('No observation/replay must not evaluate a factor or evidence.')
    before = filter.record()
    record = filter.advance('not-observed', (), forbidden, integrator=forbidden)
    assert not record['update_applied'] and not record['hazard_applied']
    assert filter.record() == before
    first = filter.advance('factor', ('image:1',), normal_density(.8, .5), event_id='event:1')
    updates = filter.updates
    assert filter.advance('factor', ('image:1',), forbidden,
                          event_id='event:1', integrator=forbidden) is first
    assert filter.updates is updates
    with pytest.raises(ValueError, match='cannot be counted again'):
        filter.advance('different-factor', ('image:1',), forbidden)
    with pytest.raises(ValueError, match='changed its contents'):
        filter.advance('different-factor', ('imu:1',), forbidden, event_id='event:1')
    with pytest.raises(ValueError, match='duplicate'):
        filter.advance('different-factor', ('imu:1', 'imu:1'), forbidden)


def test_separate_sensor_events_at_the_same_capture_time_are_accepted_once():
    filter = make_filter()
    for sensor in ('imu', 'image'):
        record = filter.advance(sensor, (sensor+':1000',), normal_density(.3, 1.),
            metadata=dict(sensor=sensor, stamp=1000.))
        assert record['metadata']['stamp'] == 1000.
    assert len(filter.updates) == 2
    assert filter.observation_ids == frozenset(('imu:1000', 'image:1000'))


def test_failure_has_no_partial_update_and_can_retry_the_same_pending_event():
    filter = make_filter()
    initial = filter.record()
    def unavailable(_):
        raise ValueError('evidence calculation unavailable')
    with pytest.raises(ValueError, match='unavailable'):
        filter.advance('pending', ('image:1',), normal_density(), integrator=unavailable)
    assert filter.record() == initial
    assert not filter.observation_ids
    record = filter.advance('pending', ('image:1',), normal_density())
    assert record['update_applied'] and record['retained_event_count'] == 1


def test_numerical_integral_must_explicitly_report_its_method_and_error():
    filter = make_filter()
    with pytest.raises(ValueError, match='LogIntegral'):
        filter.advance('bad', ('bad',), normal_density(), integrator=lambda _:0.)
    for value in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match='positive and finite'):
            LogIntegral(value, 'invalid-test-integral')
    no_method = ContinuousStiffnessBFS(normal_density(), initial_source_id='base',
        reset=ResetMeasure('reset', normal_density()))
    with pytest.raises(ValueError, match='explicitly selected evidence integrator'):
        no_method.advance('factor', ('obs',), normal_density())
    record = filter.advance('okay', ('okay',), normal_density())
    assert not record['keep_evidence']['exact']
    assert record['keep_evidence']['diagnostics']['absolute_error_estimate'] >= 0.


def test_identity_factor_is_not_a_missing_event_and_still_applies_the_hazard():
    filter = make_filter()
    filter.advance('informative', ('obs1',), normal_density(1.1, .3))
    before = lambda x: filter.evaluate([x])
    # Freeze independent values before mutating the recurrence.
    old_mass = filter.log_mass
    old = {x:before(x) for x in (-1., 0., 1.)}
    def identity(x):
        return 0., np.zeros_like(x)
    def forbidden(*args):
        raise AssertionError('An explicitly exact identity factor has evidence one.')
    record = filter.advance('identity', ('obs2',), identity,
                            identity_factor=True, integrator=forbidden)
    assert record['hazard_applied'] and record['posterior_reset_mass'] == pytest.approx(.1)
    assert record['keep_evidence']['exact'] and record['posterior_log_mass'] == old_mass
    for x, (old_value, _) in old.items():
        expected = logsumexp([math.log(.9)+old_value,
                              math.log(.1)+normal_density()(np.array([x]))[0]+old_mass])
        assert filter.evaluate([x])[0] == pytest.approx(expected, abs=1e-12)


def test_transition_from_ordinary_product_recovers_its_scale_explicitly():
    filter = make_filter(rho=0.)
    filter.advance('old-product', ('old',), normal_density(.6, .4))
    assert filter.log_mass is None
    record = filter.advance('new-bfs', ('new',), normal_density(-.1, .7), rho0=.1)
    assert record['recovered_keep_scale'] is not None
    expected_mass = norm.logpdf(.6, 0., math.sqrt(1.4))
    assert record['keep_log_mass'] == pytest.approx(expected_mass, abs=1e-12)


def test_long_history_and_tiny_log_branches_survive_without_recursion_or_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Continuous BFS must not use Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    filter = make_filter(rho=0.)
    def identity(x):
        return 0., np.zeros_like(x)
    for index in range(1100):
        filter.advance(str(index), (str(index),), identity, identity_factor=True)
    assert len(filter.updates) == 1100
    np.testing.assert_array_equal(filter.evaluate([.2])[1], [-.2])
    assert filter.evaluate([.2])[0] == normal_density()(np.array([.2]))[0]
    # A finite reset log mass must remain represented even if its posterior
    # ordinary mass underflows to zero. The function is never branch-pruned.
    tiny = make_filter()
    tiny.reset = ResetMeasure('different-rare-reset', normal_density(50., 1.))
    results = iter((LogIntegral(0., 'explicit test evidence'),
                    LogIntegral(-1000., 'explicit test evidence')))
    record = tiny.advance('rare-reset', ('rare',), identity,
                          integrator=lambda _:next(results))
    assert record['posterior_reset_mass'] == 0.
    assert record['log_bayes_factor_reset_over_keep'] == -1000.
    assert tiny.updates[0].rho0 == .1
    assert tiny.updates[0].reset is tiny.reset


def test_identical_initial_keep_and_reset_share_one_integral_without_dropping_hypotheses():
    filter = make_filter()
    calls = []
    def integrate(query):
        calls.append(query)
        return integrate_1d(query)
    record = filter.advance('first', ('first',), normal_density(.5, .7), integrator=integrate)
    assert len(calls) == 1
    assert record['identical_keep_reset_evidence_reused']
    assert record['posterior_reset_mass'] == pytest.approx(.1)
    assert filter.updates[0].rho0 == .1


@pytest.mark.parametrize('rho', [-.1, 1.1, np.nan, np.inf])
def test_invalid_hazard_is_rejected(rho):
    with pytest.raises(ValueError, match='probability'):
        make_filter(rho=rho)
