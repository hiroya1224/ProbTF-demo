"""A failed local mass can retry declared starts without fabricating curvature."""
from types import SimpleNamespace

import numpy as np
import pytest

import probtf_observation_fusion_demo.stiffness_bayes_factor as module
from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    LocalLaplaceEvidence, StiffnessEvidenceUnavailable,
)


def test_finite_wolfe_failure_with_indefinite_curvature_retries_same_density(monkeypatch):
    actual = module.minimize
    optimizer_calls = []
    density_queries = []
    def density(x):
        density_queries.append(float(x[0]))
        return -.25*(x[0]*x[0]-1.)**2, np.array([x[0]-x[0]**3])
    def minimize(fun, seed, **kwargs):
        optimizer_calls.append(np.asarray(seed).copy())
        if len(optimizer_calls) == 1:
            # A finite better line trial can fail Wolfe's curvature condition.
            # The returned iterate is not a point where a Laplace mass exists.
            assert fun(np.array([1.]))[0] < fun(seed)[0]
            return SimpleNamespace(x=np.asarray(seed).copy(), success=False,
                message='Desired error not necessarily achieved due to precision loss.', nit=0)
        return actual(fun, seed, **kwargs)
    monkeypatch.setattr(module, 'minimize', minimize)
    rule = LocalLaplaceEvidence(lambda:(np.array([.3]), np.array([2.])))
    result = rule(density)
    recovery = result.diagnostics['mass_operation_recovery']
    assert recovery['applied'] and recovery['accepted_original_seed_index'] == 1
    assert not recovery['stationarity_is_acceptance_gate']
    assert not recovery['curvature_regularization_added']
    assert recovery['failed_attempts'][0]['diagnostics']['Hessian_eigenvalues'][0] < 0.
    assert result.diagnostics['positive_definiteness_verified']
    assert result.diagnostics['mode'][0] == pytest.approx(1., abs=1e-6)
    np.testing.assert_array_equal(optimizer_calls, [[.3], [2.]])
    assert .3 in density_queries and 2. in density_queries


def test_positive_finite_mass_keeps_original_nonstationary_path(monkeypatch):
    def minimize(fun, seed, **kwargs):
        return SimpleNamespace(x=np.array([3.]), success=False,
            message='finite optimizer iteration budget', nit=1)
    monkeypatch.setattr(module, 'minimize', minimize)
    density = lambda x:(-.5*float(x@x), -np.asarray(x))
    result = LocalLaplaceEvidence(lambda:(np.array([0.]), np.array([2.])))(density)
    assert result.diagnostics['mode'] == [3.]
    assert result.diagnostics['finite_nonstationary_point_accepted']
    assert 'mass_operation_recovery' not in result.diagnostics


def test_no_declared_start_has_valid_mass_still_reports_actual_failed_operation(monkeypatch):
    monkeypatch.setattr(module, 'minimize', lambda fun, seed, **kwargs:
        SimpleNamespace(x=np.asarray(seed).copy(), success=False, message='finite stop', nit=0))
    rule = LocalLaplaceEvidence(lambda:(np.array([0.]), np.array([.1])))
    with pytest.raises(StiffnessEvidenceUnavailable, match='Matrix is not positive definite') as caught:
        rule(lambda x:(.5*float(x@x), np.asarray(x)))
    recovery = caught.value.diagnostics['mass_operation_recovery']
    assert recovery['all_declared_starts_exhausted']
    assert len(recovery['failed_attempts']) == 2
    assert not recovery['curvature_regularization_added']
