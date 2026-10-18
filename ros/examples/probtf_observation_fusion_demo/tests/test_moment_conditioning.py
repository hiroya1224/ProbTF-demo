"""One conditioning step from propagated mean/covariance/full cross."""
import numpy as np
import pytest
from probtf.spherical_law.fusion import Gaussian,condition_gaussian
from probtf_observation_fusion_demo.moment_conditioning import condition_moments


def condition(prior,H,observed,R,correction=None):
    predicted=H@prior.mean
    if correction is not None:predicted=predicted+correction
    return condition_moments(prior,predicted,H@prior.covariance@H.T,
        prior.covariance@H.T,H,observed,R)


def test_full_cross_correlated_noise_and_evidence_match_core_analytic_conditional():
    prior=Gaussian(np.array([.2,-.4,.1,.3]),np.array([
        [1.,.2,.1,-.1],[.2,.7,-.1,.1],[.1,-.1,.6,.15],[-.1,.1,.15,.8]]))
    H=np.array([[1.,.1,-.3,.2],[.4,-.2,.7,.5]])
    observed=np.array([.8,-.5]);R=np.array([[.3,.09],[.09,.15]])
    expected,evidence=condition_gaussian(prior,observed,R,H)
    actual=condition(prior,H,observed,R)
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=2e-15,rtol=2e-14)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=2e-15,rtol=2e-14)
    assert actual.log_evidence==pytest.approx(evidence,abs=4e-14)
    assert actual.diagnostics['optimization_iterations']==0
    assert not actual.diagnostics['point_optimization_used']


def test_propagated_mean_correction_changes_conditional_and_evidence_without_recentring():
    prior=Gaussian(np.array([.2,-.1]),np.array([[.3,.1],[.1,.4]]))
    H=np.array([[1.,.7]]);R=np.array([[.04]]);observed=np.array([.5]);correction=np.array([-.06])
    actual=condition(prior,H,observed,R,correction)
    expected,evidence=condition_gaussian(prior,observed-correction,R,H)
    modal=condition(prior,H,observed,R)
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=1e-15)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=1e-15)
    assert actual.log_evidence==pytest.approx(evidence,abs=1e-14)
    assert not np.array_equal(actual.distribution.mean,modal.distribution.mean)
    assert actual.log_evidence!=modal.log_evidence
    assert actual.diagnostics['core_O_P_mean_used']


def test_rank_deficient_prior_keeps_exact_dirac_and_all_remaining_cross():
    root=np.array([[1.,0.],[.3,.5],[0.,0.],[.1,-.2]])
    prior=Gaussian(np.array([.2,-.1,7.,.4]),root@root.T)
    H=np.array([[1.,.2,2.,-.4],[.3,-.1,1.,.7]])
    observed=np.array([13.7,7.5]);R=np.array([[.2,.04],[.04,.3]])
    expected,evidence=condition_gaussian(prior,observed,R,H)
    actual=condition(prior,H,observed,R)
    assert actual.diagnostics['prior_support_rank']==2
    assert actual.distribution.mean[2]==7.
    np.testing.assert_array_equal(actual.distribution.covariance[2],np.zeros(4))
    np.testing.assert_array_equal(actual.distribution.covariance[:,2],np.zeros(4))
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=2e-14)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=2e-14)
    assert actual.log_evidence==pytest.approx(evidence,abs=5e-14)


def test_rank_zero_and_singular_innovation_use_actual_support_without_floors():
    prior=Gaussian(np.array([.3,-.4]),np.zeros((2,2)));H=np.array([[.5,2.]])
    observed=np.array([.2]);R=np.array([[.15]])
    actual=condition(prior,H,observed,R)
    expected,evidence=condition_gaussian(prior,observed,R,H)
    np.testing.assert_array_equal(actual.distribution.mean,prior.mean)
    np.testing.assert_array_equal(actual.distribution.covariance,prior.covariance)
    assert actual.log_evidence==pytest.approx(evidence)
    # An exact observation on a supported subspace is not regularized.
    prior=Gaussian(np.zeros(2),np.eye(2));H=np.array([[1.,0.],[0.,0.]])
    exact=condition(prior,H,np.array([.2,0.]),np.zeros((2,2)))
    np.testing.assert_array_equal(exact.distribution.covariance,np.diag([0.,1.]))
    assert exact.diagnostics['innovation_support_rank']==1


def test_synchronous_sensor_cross_is_not_replaced_by_independent_predicted_blocks():
    prior=Gaussian(np.array([.2]),np.array([[.7]]));H=np.ones((2,1))
    observed=np.array([.5,-.1]);R=np.diag([.03,.04])
    actual=condition(prior,H,observed,R)
    expected,evidence=condition_gaussian(prior,observed,R,H)
    assert actual.diagnostics['innovation_covariance'][0][1]==.7
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=1e-14)
    assert actual.log_evidence==pytest.approx(evidence,abs=1e-13)
    independent_variance=.7+np.diag(R)
    independent_log=-.5*np.sum(np.log(2*np.pi*independent_variance)+(observed-.2)**2/independent_variance)
    assert abs(actual.log_evidence-independent_log)>.1


def test_real_nonfinite_invalid_covariance_and_unavailable_support_remain_failures():
    prior=Gaussian(np.zeros(1),np.ones((1,1)));H=np.ones((1,1))
    with pytest.raises(ValueError,match='finite moments'):
        condition_moments(prior,np.array([np.nan]),np.eye(1),H,H,np.zeros(1),np.eye(1))
    with pytest.raises(ValueError):
        condition(prior,H,np.zeros(1),-np.eye(1))
    with pytest.raises(ValueError):
        condition(Gaussian(np.zeros(1),np.zeros((1,1))),H,np.ones(1),np.zeros((1,1)))
