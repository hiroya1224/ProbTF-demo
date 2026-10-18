"""Support, normalization and prior-once contracts of nonlinear conditioning."""
import numpy as np
import pytest

from probtf.spherical_law.fusion import Gaussian, condition_gaussian
from probtf_observation_fusion_demo.nonlinear_conditioning import condition_nonlinear


def linear_likelihood(matrix, observed):
    def evaluate(value):
        return matrix@value-observed, matrix
    return evaluate


def test_one_step_affine_limit_matches_full_cross_gaussian_conditioning_and_evidence():
    prior=Gaussian(np.array([.2,-.4,.1,.3]),np.array([
        [1.,.2,.1,-.1],[.2,.7,-.1,.1],[.1,-.1,.6,.15],[-.1,.1,.15,.8]]))
    matrix=np.array([[1.,.1,-.3,.2],[.4,-.2,.7,.5]])
    observed=np.array([.8,-.5]);noise=np.array([[.3,.09],[.09,.15]])
    expected,evidence=condition_gaussian(prior,observed,noise,matrix)
    actual=condition_nonlinear(prior,linear_likelihood(matrix,observed),noise,max_iterations=1)
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=2e-15,rtol=2e-14)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=2e-15,rtol=2e-14)
    assert actual.log_evidence==pytest.approx(evidence,abs=4e-14)
    assert actual.diagnostics['completed_iterations']==1
    assert actual.diagnostics['stationarity_verified']


def test_rank_deficient_prior_retains_exact_dirac_coordinates_and_all_cross():
    root=np.array([[1.,0.],[.3,.5],[0.,0.],[.1,-.2]])
    prior=Gaussian(np.array([.2,-.1,7.,.4]),root@root.T)
    matrix=np.array([[1.,.2,2.,-.4],[.3,-.1,1.,.7]])
    observed=np.array([13.7,7.5]);noise=np.array([[.2,.04],[.04,.3]])
    expected,evidence=condition_gaussian(prior,observed,noise,matrix)
    actual=condition_nonlinear(prior,linear_likelihood(matrix,observed),noise,max_iterations=1)
    assert actual.diagnostics['prior_support_rank']==2
    assert actual.distribution.mean[2]==7.
    assert np.array_equal(actual.distribution.covariance[2],np.zeros(4))
    assert np.array_equal(actual.distribution.covariance[:,2],np.zeros(4))
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=2e-14,rtol=2e-14)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=2e-14,rtol=2e-14)
    assert actual.log_evidence==pytest.approx(evidence,abs=5e-14)


def test_rank_zero_prior_evidence_is_the_actual_fixed_candidate_likelihood():
    prior=Gaussian(np.array([.3,-.4]),np.zeros((2,2)))
    matrix=np.array([[.5,2.]])
    residual=matrix@prior.mean-np.array([.2]);noise=np.array([[.15]])
    actual=condition_nonlinear(prior,linear_likelihood(matrix,np.array([.2])),noise)
    expected=-.5*float(residual@np.linalg.solve(noise,residual))-.5*np.log(2.*np.pi*.15)
    np.testing.assert_array_equal(actual.distribution.mean,prior.mean)
    np.testing.assert_array_equal(actual.distribution.covariance,prior.covariance)
    assert actual.log_evidence==pytest.approx(expected)
    assert actual.diagnostics['prior_support_rank']==0


def test_iterations_keep_original_prior_and_do_not_raise_the_likelihood_to_a_power():
    prior=Gaussian(np.array([0.]),np.array([[1.]]));noise=np.array([[.04]])
    seen=[]
    def evaluate(value):
        assert not value.flags.writeable
        seen.append(value.copy())
        return value-.7,np.ones((1,1))
    first=condition_nonlinear(prior,evaluate,noise,max_iterations=1)
    repeated=condition_nonlinear(prior,evaluate,noise,max_iterations=20)
    np.testing.assert_array_equal(prior.mean,[0.]);np.testing.assert_array_equal(prior.covariance,[[1.]])
    np.testing.assert_allclose(repeated.distribution.mean,first.distribution.mean,atol=2e-15)
    np.testing.assert_allclose(repeated.distribution.covariance,first.distribution.covariance,atol=2e-15)
    assert repeated.log_evidence==pytest.approx(first.log_evidence,abs=2e-15)
    assert repeated.diagnostics['original_prior_held_fixed']
    assert not repeated.diagnostics['repeated_observation_multiplication']
    assert repeated.distribution.covariance[0,0]==pytest.approx(1./26.)


def test_true_nonlinear_objective_is_used_during_backtracking_and_at_final_mean():
    prior=Gaussian(np.zeros(1),np.ones((1,1)));noise=np.array([[.01]])
    def evaluate(value):
        return np.exp(value)-10.,np.exp(value).reshape(1,1)
    actual=condition_nonlinear(prior,evaluate,noise,max_iterations=40)
    value=actual.distribution.mean
    objective=.5*float(value@value)+.5*float((np.exp(value)-10.)**2/.01)
    assert actual.diagnostics['final_objective']==pytest.approx(objective)
    assert actual.diagnostics['final_objective']<actual.diagnostics['initial_objective']
    assert any(row['accepted'] and row['scale']<1. for row in actual.diagnostics['line_search'])
    assert all(b['objective']<=a['objective'] for a,b in zip(
        actual.diagnostics['objective_history'],actual.diagnostics['objective_history'][1:]))
    assert not actual.diagnostics['exact_nonlinear_evidence_claimed']


@pytest.mark.parametrize('work,backtracks',[(0,20),(1,0)])
def test_finite_budget_or_line_search_nonconvergence_returns_the_finite_proxy(work,backtracks):
    prior=Gaussian(np.zeros(1),np.ones((1,1)))
    actual=condition_nonlinear(prior,lambda x:(np.exp(x)-10.,np.exp(x).reshape(1,1)),
        np.array([[.01]]),max_iterations=work,max_backtracks=backtracks)
    assert np.isfinite(actual.log_evidence)
    assert np.isfinite(actual.distribution.mean).all()
    assert np.isfinite(actual.distribution.covariance).all()
    assert actual.diagnostics['finite_nonstationary_iterate_used']
    assert actual.diagnostics['numerical_quality_is_stop_gate'] is False


def test_nonfinite_observation_or_required_improper_covariance_is_a_real_failure():
    prior=Gaussian(np.zeros(1),np.ones((1,1)))
    with pytest.raises(ValueError,match='nonfinite residual'):
        condition_nonlinear(prior,lambda x:(np.array([np.nan]),np.ones((1,1))),np.eye(1))
    with pytest.raises(np.linalg.LinAlgError):
        condition_nonlinear(prior,lambda x:(x,np.eye(1)),np.zeros((1,1)))


def test_unavailable_trial_can_backtrack_to_a_valid_finite_update():
    prior=Gaussian(np.zeros(1),np.ones((1,1)))
    def evaluate(value):
        if value[0]>.3:
            return np.array([np.nan]),np.ones((1,1))
        return value-.8,np.ones((1,1))
    actual=condition_nonlinear(prior,evaluate,np.eye(1),max_iterations=1)
    assert actual.distribution.mean[0]==pytest.approx(.2)
    assert any('actual_trial_error'in row for row in actual.diagnostics['line_search'])
    assert actual.diagnostics['final_objective']<actual.diagnostics['initial_objective']
