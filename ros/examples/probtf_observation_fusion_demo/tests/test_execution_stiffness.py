"""Execution damping preserves the estimate and the accepted filter epoch."""

from copy import deepcopy
import json

import numpy as np
import pytest

from probtf_observation_fusion_demo.execution_stiffness import LogStiffnessDamper


def test_first_estimate_initializes_exactly_with_independent_json_records():
    estimate = np.array([2.3, 100.01])
    original = estimate.copy()
    damping = LogStiffnessDamper({})
    record = damping.advance(estimate, 1234.5)
    np.testing.assert_array_equal(record['execution_K'], original)
    np.testing.assert_array_equal(record['estimated_K'], original)
    assert record['initialized'] and record['initialized_this_step']
    assert record['previous_execution_logK'] is record['previous_stamp'] is None
    assert record['actual_dt'] == record['alpha'] == 0.
    np.testing.assert_array_equal(record['applied_delta_logK'], [0., 0.])
    assert record['kp_exec_tau'] == .5 and record['max_log_kp_exec_step'] == .05
    json.dumps(record, allow_nan=False)
    estimate[:] = 999.
    record['execution_K'][0] = 888.
    record['execution_logK'][0] = 777.
    retained = damping.advance(original, 1234.5)
    np.testing.assert_array_equal(retained['execution_K'], original)
    assert not retained['initialized_this_step']


def test_uncapped_log_lag_matches_closed_form_and_uses_actual_elapsed_time():
    damping = LogStiffnessDamper({'kp_exec_tau': .5, 'max_log_kp_exec_step': 0.})
    first = damping.advance([10.], 100.)
    record = damping.advance([100.], 100.2)
    expected_alpha = 1.-np.exp(-.2/.5)
    expected_log = np.log(100.)+(np.log(10.)-np.log(100.))*np.exp(-.2/.5)
    assert record['actual_dt'] == pytest.approx(.2)
    assert record['alpha'] == pytest.approx(expected_alpha)
    np.testing.assert_allclose(record['execution_logK'], [expected_log], atol=2e-14, rtol=0.)
    np.testing.assert_allclose(record['execution_K'], [np.exp(expected_log)], atol=2e-13, rtol=0.)
    assert 10. < record['execution_K'][0] < 100.
    assert record['previous_stamp'] == first['stamp']
    np.testing.assert_array_equal(record['raw_delta_logK'], record['applied_delta_logK'])


def test_log_step_cap_limits_both_directions_per_update_without_overshoot():
    damping = LogStiffnessDamper({})
    damping.advance([1., 1.], 0.)
    first = damping.advance([1e6, 1e-6], 10.)
    np.testing.assert_array_equal(first['applied_delta_logK'], [.05, -.05])
    np.testing.assert_allclose(first['execution_K'], np.exp([.05, -.05]), rtol=1e-15)
    second = damping.advance([1e6, 1e-6], 10.01)
    np.testing.assert_array_equal(second['applied_delta_logK'], [.05, -.05])
    np.testing.assert_allclose(second['execution_logK'], [.1, -.1], rtol=0., atol=1e-15)
    # The cap is per update, not multiplied by a nominal control rate or dt.
    assert second['actual_dt'] == pytest.approx(.01)


def test_same_estimate_continues_to_be_followed_without_any_new_estimate():
    damping = LogStiffnessDamper({})
    first = damping.advance([10.], 0.)
    values = [first['execution_K'][0]]
    for index in range(1, 11):
        record = damping.advance([100.], .1*index)
        values.append(record['execution_K'][0])
        assert record['estimated_K'] == [100.]
        assert 0. < record['applied_delta_logK'][0] <= .05
    assert np.all(np.diff(values) > 0.) and values[-1] < 100.


@pytest.mark.parametrize('settings', ({}, {'kp_exec_tau': 0., 'max_log_kp_exec_step': 0.}))
def test_zero_dt_preserves_execution_exactly_even_when_the_target_jumps(settings):
    damping = LogStiffnessDamper(settings)
    original = damping.advance([3.7, 9.1], 4.)
    record = damping.advance([300., .1], 4.)
    assert record['estimated_K'] == [300., .1]
    assert record['actual_dt'] == record['alpha'] == 0.
    np.testing.assert_array_equal(record['execution_K'], original['execution_K'])
    np.testing.assert_array_equal(record['execution_logK'], original['execution_logK'])
    np.testing.assert_array_equal(record['applied_delta_logK'], [0., 0.])


def test_zero_tau_and_zero_cap_follow_exact_target_at_positive_dt():
    damping = LogStiffnessDamper({'kp_exec_tau': 0., 'max_log_kp_exec_step': 0.})
    damping.advance([1., 2.], 0.)
    target = [9.3, .17]
    record = damping.advance(target, .01)
    assert record['alpha'] == 1.
    np.testing.assert_array_equal(record['execution_K'], target)


def test_accepted_record_restoration_prevents_unaccepted_proposals_consuming_time_or_state():
    settings = {'kp_exec_tau': .5, 'max_log_kp_exec_step': .05}
    accepted = LogStiffnessDamper(settings).advance([10., 20.], 0.)
    saved = deepcopy(accepted)
    candidate = LogStiffnessDamper(settings, previous=accepted).advance([100., 2.], .2)
    retried = LogStiffnessDamper(settings, previous=accepted).advance([100., 2.], .2)
    assert candidate == retried and accepted == saved
    # Only committing that candidate makes its filter stamp the next origin.
    next_epoch = LogStiffnessDamper(settings, previous=candidate).advance([100., 2.], .5)
    assert next_epoch['actual_dt'] == pytest.approx(.3)
    np.testing.assert_array_equal(next_epoch['previous_execution_logK'], candidate['execution_logK'])
    assert np.all(np.abs(next_epoch['applied_delta_logK']) <= .05)
    minimal = {'stamp': candidate['stamp'], 'execution_logK': candidate['execution_logK']}
    restored = LogStiffnessDamper(settings, previous=minimal).advance([100., 2.], .5)
    np.testing.assert_array_equal(restored['execution_logK'], next_epoch['execution_logK'])


def test_restoration_copies_the_accepted_record_and_uncapped_lag_composes_in_time():
    settings = {'kp_exec_tau': .5, 'max_log_kp_exec_step': 0.}
    initial = LogStiffnessDamper(settings).advance([10.], 0.)
    restored = LogStiffnessDamper(settings, previous=initial)
    initial['execution_logK'][0] += 5.
    restored.advance([100.], .1)
    split = restored.advance([100.], .3)
    whole = LogStiffnessDamper(settings)
    whole.advance([10.], 0.)
    one_step = whole.advance([100.], .3)
    np.testing.assert_allclose(split['execution_logK'], one_step['execution_logK'], atol=1e-14, rtol=0.)


@pytest.mark.parametrize('value', ([0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.],
    [], 1., [[1., 2.]], ['invalid', 1.], [1.]))
def test_invalid_estimated_K_is_explicit_and_does_not_advance_state(value):
    damping = LogStiffnessDamper({})
    accepted = damping.advance([10., 20.], 1.)
    with pytest.raises(ValueError, match='Estimated K'):
        damping.advance(value, 2.)
    actual = damping.advance([100., 2.], 3.)
    expected = LogStiffnessDamper({}, previous=accepted).advance([100., 2.], 3.)
    assert actual == expected


@pytest.mark.parametrize('stamp', (np.nan, np.inf, -np.inf, .9, 'invalid'))
def test_invalid_or_decreasing_stamp_does_not_advance_state(stamp):
    damping = LogStiffnessDamper({})
    accepted = damping.advance([10.], 1.)
    with pytest.raises(ValueError, match='stamp'):
        damping.advance([100.], stamp)
    assert damping.advance([100.], 2.) == LogStiffnessDamper({}, previous=accepted).advance([100.], 2.)


@pytest.mark.parametrize('setting', (
    {'kp_exec_tau': -1.}, {'kp_exec_tau': np.nan}, {'kp_exec_tau': np.inf},
    {'max_log_kp_exec_step': -1.}, {'max_log_kp_exec_step': np.nan},
    {'max_log_kp_exec_step': np.inf}, {'kp_exec_tau': 'invalid'}))
def test_invalid_filter_settings_are_rejected(setting):
    with pytest.raises(ValueError, match='finite nonnegative'):
        LogStiffnessDamper(setting)


@pytest.mark.parametrize('previous', (
    {}, {'stamp': 0.}, {'execution_logK': [0.]},
    {'stamp': np.nan, 'execution_logK': [0.]},
    {'stamp': 0., 'execution_logK': [np.inf]},
    {'stamp': 0., 'execution_logK': []},
    {'stamp': 0., 'execution_logK': [[0.]]},
    {'stamp': 0., 'execution_logK': [1000.]},
    {'stamp': 0., 'execution_logK': [-1000.]},
    {'stamp': 0., 'execution_logK': [0.], 'execution_K': [-1.]},
    {'stamp': 0., 'execution_logK': [0.], 'execution_K': [2.]},
    {'stamp': 0., 'execution_logK': [0.], 'execution_K': [1., 1.]},
))
def test_invalid_previous_record_is_not_silently_reinitialized(previous):
    with pytest.raises(ValueError):
        LogStiffnessDamper({}, previous=previous)


def test_filter_uses_no_randomness(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Execution stiffness damping must be deterministic.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    initial = LogStiffnessDamper({}).advance([10., 20.], 0.)
    LogStiffnessDamper({}, previous=initial).advance([100., 2.], .1)
