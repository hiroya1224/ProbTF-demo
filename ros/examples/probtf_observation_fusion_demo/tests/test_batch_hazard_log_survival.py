"""A rounded ordinary rho=1 cannot erase a finite-survival branch."""
import json
import math
import numpy as np
import pytest

from probtf_observation_fusion_demo.joint_batch_stiffness import aggregate_event_hazard_logs
from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    ContinuousStiffnessBFS, LocalLaplaceEvidence, ResetMeasure,
)


def normal(mean):
    def evaluate(x):
        delta = np.asarray(x)-mean
        return float(-.5*delta@delta-.5*math.log(2.*math.pi)), -delta
    return evaluate


def test_ordinary_one_keeps_finite_survival_log_for_many_actual_event_trials():
    rho, keep, reset = aggregate_event_hazard_logs(.1, 420)
    assert rho == 1.
    assert keep == 420*math.log1p(-.1)
    assert math.isfinite(keep) and math.exp(keep) > 0.
    assert reset == 0.
    json.dumps(dict(rho=rho, keep=keep, reset=reset), allow_nan=False)


def test_finite_survival_branch_can_dominate_where_reset_density_is_small():
    initial, reset_density = normal(0.), normal(100.)
    bfs = ContinuousStiffnessBFS(initial, initial_source_id='keep',
        reset=ResetMeasure('different reset', reset_density))
    rho, keep, reset = aggregate_event_hazard_logs(.1, 420)
    record = bfs.advance('identity', ('all effective batch observations',),
        lambda x: (0., np.zeros_like(x)), identity_factor=True,
        rho0=rho, log_keep_probability=keep, log_reset_probability=reset)
    value, score = bfs.evaluate(np.array([0.]))
    assert value == pytest.approx(keep-.5*math.log(2.*math.pi), abs=1e-12)
    assert value > reset_density(np.array([0.]))[0]+4000.
    np.testing.assert_allclose(score, [0.], atol=1e-100)
    assert record['rho0'] == 1. and record['log_keep_prior_probability'] == keep
    assert record['posterior_keep_log_probability'] == keep
    assert record['authoritative_log_probabilities']
    json.dumps(record, allow_nan=False)


def test_true_event_probability_one_is_still_an_exact_reset():
    rho, keep, reset = aggregate_event_hazard_logs(1., 420)
    assert (rho, keep, reset) == (1., None, 0.)
    reset_density = normal(100.)
    bfs = ContinuousStiffnessBFS(normal(0.), initial_source_id='keep',
        reset=ResetMeasure('reset', reset_density))
    record = bfs.advance('identity', ('obs',), lambda x: (0., np.zeros_like(x)),
        identity_factor=True, rho0=rho,
        log_keep_probability=keep, log_reset_probability=reset)
    assert bfs.evaluate(np.array([0.]))[0] == reset_density(np.array([0.]))[0]
    assert record['log_keep_prior_probability'] is None
    assert record['posterior_keep_log_probability'] is None
    json.dumps(record, allow_nan=False)


def test_nonidentity_integrates_both_hypotheses_despite_ordinary_rho_one():
    calls = []
    rule = LocalLaplaceEvidence(lambda: (np.array([0.]), np.array([100.])))
    def integrate(density):
        calls.append(density)
        return rule(density)
    bfs = ContinuousStiffnessBFS(normal(0.), initial_source_id='keep',
        reset=ResetMeasure('reset', normal(100.)), integrator=integrate)
    rho, keep, reset = aggregate_event_hazard_logs(.1, 420)
    record = bfs.advance('factor', ('batch observed factor',), normal(0.),
        rho0=rho, log_keep_probability=keep, log_reset_probability=reset)
    assert len(calls) == 2
    assert record['keep_evidence'] is not None and record['reset_evidence'] is not None
    assert record['posterior_keep_log_probability'] == pytest.approx(0., abs=1e-12)
    assert math.isfinite(record['posterior_reset_log_probability'])
    assert record['posterior_reset_log_probability'] < -2000.
    assert bfs.evaluate(np.array([0.]))[0] > -60.
    json.dumps(record, allow_nan=False)


def test_survival_log_is_part_of_immutable_density_identity():
    ids = []
    for keep in (-40., -50.):
        initial = normal(0.)
        bfs = ContinuousStiffnessBFS(initial, initial_source_id='same',
            reset=ResetMeasure('same', initial))
        bfs.advance('factor', ('same-observation',), lambda x: (0., np.zeros_like(x)),
            identity_factor=True, rho0=1., log_keep_probability=keep, log_reset_probability=0.)
        ids.append(bfs.density_id)
    assert ids[0] != ids[1]
