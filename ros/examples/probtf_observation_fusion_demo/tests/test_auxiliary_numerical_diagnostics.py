"""Finite auxiliary audit misses remain usable, without altering any matrix."""

from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


def _auxiliary(*, hessian=1., residual=1e-5):
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    active = np.array([False])
    identifier.sensitivity = SimpleNamespace(last_active_set=active,
        equilibrium_jacobians=lambda *args: (np.array([[hessian]]), np.array([[2.]])),
        equilibrium_active_set=lambda *args: active.copy())
    identifier.robot = SimpleNamespace(tau_gravity=lambda theta: np.array([residual]),
        model=SimpleNamespace(lowerPositionLimit=np.array([-10.]), upperPositionLimit=np.array([10.])))
    identifier.spring = SimpleNamespace(torque=lambda *args: np.zeros(1))
    identifier.solver = SimpleNamespace(_V_total=lambda *args: 0.)
    return identifier


def test_finite_KKT_miss_returns_actual_iterate_and_honest_nonconvergence():
    identifier = _auxiliary(residual=2.5e-5)
    theta, H, J, active = identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))
    np.testing.assert_array_equal(theta, [.2])
    np.testing.assert_array_equal(H, [[1.]])
    np.testing.assert_array_equal(J, [[2.]])
    assert not active.any()
    audit = identifier.last_equilibrium_audit
    assert audit['KKT_residual_inf_nm'] == 2.5e-5
    assert not audit['KKT_tolerance_satisfied'] and not audit['refinement_converged']
    assert audit['used_finite_auxiliary_result'] and not audit['audit_is_fatal']
    assert audit['refinement_termination'] == 'no_further_residual_and_potential_reduction'
    assert identifier.numerical_diagnostics()['equilibrium_summary']['residual_warning_count'] == 1


def test_finite_negative_curvature_is_diagnostic_and_Hessian_is_not_regularized():
    identifier = _auxiliary(hessian=-2., residual=0.)
    theta, H, J, _ = identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))
    np.testing.assert_array_equal(theta, [.2])
    np.testing.assert_array_equal(H, [[-2.]])
    assert np.all(np.isfinite(np.linalg.solve(H, J)))
    audit = identifier.last_equilibrium_audit
    assert audit['Hessian_eigenvalues'] == [-2.]
    assert not audit['positive_curvature'] and not audit['Hessian_modified']
    assert not audit['physical_stability_guaranteed']
    assert identifier.numerical_diagnostics()['equilibrium_summary']['nonpositive_curvature_count'] == 1


@pytest.mark.parametrize('hessian,residual', [(float('nan'), 0.), (1., float('inf'))])
def test_actual_nonfinite_auxiliary_arithmetic_remains_an_error(hessian, residual):
    identifier = _auxiliary(hessian=hessian, residual=residual)
    with pytest.raises(ValueError, match='nonfinite'):
        identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))


def test_actual_unsolved_linear_system_remains_an_error():
    identifier = _auxiliary(hessian=0., residual=1e-5)
    with pytest.raises(np.linalg.LinAlgError):
        identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))


@pytest.mark.parametrize('result', ['exception', 'nonfinite'])
def test_diagnostic_eigendecomposition_failure_preserves_finite_auxiliary_result(monkeypatch, result):
    identifier = _auxiliary(hessian=2., residual=0.)

    def diagnostic_eigenvalues(matrix):
        if result == 'exception':
            raise np.linalg.LinAlgError('diagnostic eigensolver failed')
        return np.array([float('inf')])

    monkeypatch.setattr(np.linalg, 'eigvalsh', diagnostic_eigenvalues)
    theta, H, J, _ = identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))
    np.testing.assert_array_equal(theta, [.2])
    np.testing.assert_array_equal(H, [[2.]])
    np.testing.assert_array_equal(np.linalg.solve(H, J), [[1.]])
    audit = identifier.last_equilibrium_audit
    assert audit['Hessian_eigenvalues'] is None and audit['positive_curvature'] is None
    assert audit['curvature_audit_error'] and not audit['audit_is_fatal']
    assert identifier.numerical_diagnostics()['equilibrium_summary']['curvature_audit_failure_count'] == 1


def test_diagnostic_symmetrization_does_not_overflow_finite_Hessian():
    identifier = _auxiliary(hessian=1e308, residual=0.)
    with np.errstate(over='raise'):
        _, H, J, _ = identifier._refine_equilibrium(np.array([.2]), np.array([.1]), np.array([5.]))
    np.testing.assert_array_equal(H, [[1e308]])
    assert np.all(np.isfinite(np.linalg.solve(H, J)))
    assert identifier.last_equilibrium_audit['Hessian_eigenvalues'] == [1e308]


def test_long_finite_continuation_uses_bounded_work_without_clipping_K_or_anchor():
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier._branch_anchor = ('source', np.array([0.]), np.array([.2]))
    state = SimpleNamespace(state_id='source', command=(.1,))
    queries = []

    def refine(initial, command, stiffness):
        queries.append((np.array(initial), np.array(stiffness)))
        return np.array(initial), np.eye(1), np.eye(1), np.array([False])

    identifier._refine_equilibrium = refine
    x = np.array([101.])
    identifier._branch_equilibrium(x, state)
    assert len(queries) == 1000
    np.testing.assert_array_equal(queries[-1][1], np.exp(x))
    assert all(np.array_equal(initial, [.2]) for initial, _ in queries)
    audit = identifier.last_continuation_audit
    assert audit['requested_stages'] == 1010 and audit['used_stages'] == 1000
    assert audit['work_budget_saturated']
    assert audit['used_maximum_logK_step'] == pytest.approx(.101)
    assert not audit['stiffness_candidate_clipped'] and not audit['anchor_changed']
