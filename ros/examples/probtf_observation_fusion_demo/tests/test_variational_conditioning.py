"""Independent Gaussian/KL/expectation oracles for full-parameter updates."""
import math
import numpy as np
import pytest

from probtf.spherical_law.fusion import Gaussian,condition_gaussian
from probtf_observation_fusion_demo.joint_batch_fusion import log_evidence_parts_difference
from probtf_observation_fusion_demo.variational_conditioning import (
    condition_variational,ExpectedLikelihoodEvaluation,_WhitenedObjective)


def affine_evaluator(H,y,R,*,offset=None,common_constant=0.,seen=None):
    Q=np.linalg.inv(R);b=np.zeros(y.size)if offset is None else offset
    normalizer=.5*(np.linalg.slogdet(R)[1]+y.size*np.log(2*np.pi))
    def evaluate(mean,covariance):
        assert not mean.flags.writeable and not covariance.flags.writeable
        if seen is not None:seen.append((mean.copy(),covariance.copy()))
        residual=H@mean+b-y
        quadratic=.5*float(residual@Q@residual)
        trace=.5*float(np.trace(Q@H@covariance@H.T))
        terms=(quadratic,trace,normalizer,common_constant)
        return ExpectedLikelihoodEvaluation(math.fsum(terms),H.T@Q@residual,
            .5*H.T@Q@H,diagnostics={'same_observation':True},
            expected_negative_log_likelihood_terms=terms)
    return evaluate


def test_anisotropic_correlated_prior_full_posterior_and_affine_evidence_match_core():
    prior=Gaussian(np.array([.2,-.4,.1,.3]),np.array([
        [1.,.2,.1,-.1],[.2,.7,-.1,.1],[.1,-.1,.6,.15],[-.1,.1,.15,.8]]))
    H=np.array([[1.,.1,-.3,.2],[.4,-.2,.7,.5]])
    y=np.array([.8,-.5]);R=np.array([[.3,.09],[.09,.15]]);b=np.array([.06,-.04])
    expected,evidence=condition_gaussian(prior,y-b,R,H)
    actual=condition_variational(prior,affine_evaluator(H,y,R,offset=b),
        max_iterations=300,gradient_tolerance=1e-10)
    np.testing.assert_allclose(actual.distribution.mean,expected.mean,atol=3e-9,rtol=2e-9)
    np.testing.assert_allclose(actual.distribution.covariance,expected.covariance,atol=3e-9,rtol=2e-9)
    assert actual.log_evidence==pytest.approx(evidence,abs=2e-12)
    assert actual.diagnostics['parameter_count']==4+10
    assert actual.diagnostics['full_joint_cross_covariance_optimized']
    assert not actual.diagnostics['point_MAP_used']


def test_full_mean_and_cholesky_gradient_match_independent_KL_finite_differences():
    prior=Gaussian(np.array([.2,-.1,.3]),np.array([[.8,.13,-.08],[.13,.4,.1],[-.08,.1,.6]]))
    H=np.array([[1.,.3,-.1],[.2,-.4,.8]]);y=np.array([.4,-.3]);R=np.array([[.12,.03],[.03,.2]])
    evaluator=affine_evaluator(H,y,R)
    objective=_WhitenedObjective(prior,evaluator)
    x=np.linspace(-.13,.11,objective.parameter_count)
    actual=objective.evaluate(x)
    inverse=np.linalg.inv(prior.covariance);logdet=np.linalg.slogdet(prior.covariance)[1]
    def independent_cost(parameters):
        distribution=objective.evaluate(parameters)['distribution']
        difference=distribution.mean-prior.mean
        KL=.5*(np.trace(inverse@distribution.covariance)+difference@inverse@difference
            -prior.mean.size+logdet-np.linalg.slogdet(distribution.covariance)[1])
        return KL+evaluator(distribution.mean,distribution.covariance).expected_negative_log_likelihood
    basis=np.eye(x.size);h=1e-6
    numerical=np.array([(independent_cost(x+h*unit)-independent_cost(x-h*unit))/(2*h)for unit in basis])
    np.testing.assert_allclose(actual['gradient'],numerical,atol=4e-9,rtol=3e-8)
    # All off-diagonal Cholesky coordinates participate, not only variances.
    assert np.count_nonzero(np.abs(actual['gradient'][3:][~objective.diagonal])>1e-6)==3


def test_supported_prior_preserves_exact_Dirac_coordinates_and_matches_affine_oracle():
    B=np.array([[1.,0.],[.3,.5],[0.,0.],[.1,-.2]])
    prior=Gaussian(np.array([.2,-.1,7.,.4]),B@B.T)
    H=np.array([[1.,.2,2.,-.4],[.3,-.1,1.,.7]])
    y=np.array([13.7,7.5]);R=np.array([[.2,.04],[.04,.3]])
    expected,evidence=condition_gaussian(prior,y,R,H)
    result=condition_variational(prior,affine_evaluator(H,y,R),max_iterations=300,gradient_tolerance=1e-10)
    assert result.distribution.mean[2]==7.
    np.testing.assert_array_equal(result.distribution.covariance[2],np.zeros(4))
    np.testing.assert_array_equal(result.distribution.covariance[:,2],np.zeros(4))
    np.testing.assert_allclose(result.distribution.mean,expected.mean,atol=2e-8)
    np.testing.assert_allclose(result.distribution.covariance,expected.covariance,atol=2e-8)
    assert result.log_evidence==pytest.approx(evidence,abs=2e-12)
    assert result.diagnostics['prior_support_rank']==2


def test_rank_zero_and_constant_likelihood_retain_the_original_exact_prior():
    prior=Gaussian(np.array([.3,-.4]),np.zeros((2,2)))
    H=np.array([[.5,2.]]);y=np.array([.2]);R=np.array([[.15]])
    evaluator=affine_evaluator(H,y,R);result=condition_variational(prior,evaluator)
    assert result.distribution is prior
    assert result.log_evidence==-evaluator(prior.mean,prior.covariance).expected_negative_log_likelihood
    assert result.diagnostics['completed_iterations']==0
    full=Gaussian(np.array([-.2,.3]),np.array([[.4,.1],[.1,.7]]))
    constant=condition_variational(full,lambda m,P:ExpectedLikelihoodEvaluation(2.,np.zeros(2),np.zeros((2,2))))
    assert constant.distribution is full and constant.log_evidence==-2.
    assert constant.diagnostics['final_KL']==0.


def test_original_prior_and_each_new_observation_are_used_once_across_sequential_updates():
    prior=Gaussian(np.array([.1,-.2]),np.array([[.8,.2],[.2,.5]]))
    before_mean=prior.mean.copy();before_covariance=prior.covariance.copy()
    H1=np.array([[1.,.3]]);y1=np.array([.8]);R1=np.array([[.12]])
    H2=np.array([[.4,-.9]]);y2=np.array([-.1]);R2=np.array([[.08]])
    first=condition_variational(prior,affine_evaluator(H1,y1,R1),max_iterations=200,gradient_tolerance=1e-10)
    repeated=condition_variational(prior,affine_evaluator(H1,y1,R1),max_iterations=200,gradient_tolerance=1e-10)
    np.testing.assert_array_equal(first.distribution.mean,repeated.distribution.mean)
    np.testing.assert_array_equal(first.distribution.covariance,repeated.distribution.covariance)
    second=condition_variational(first.distribution,affine_evaluator(H2,y2,R2),max_iterations=200,gradient_tolerance=1e-10)
    exact_first,z1=condition_gaussian(prior,y1,R1,H1)
    expected,z2=condition_gaussian(exact_first,y2,R2,H2)
    _,actual_prior_z2=condition_gaussian(first.distribution,y2,R2,H2)
    np.testing.assert_allclose(second.distribution.mean,expected.mean,atol=2e-8)
    np.testing.assert_allclose(second.distribution.covariance,expected.covariance,atol=2e-8)
    assert first.log_evidence==pytest.approx(z1,abs=2e-12)
    assert second.log_evidence==pytest.approx(actual_prior_z2,abs=2e-12)
    # The second prior is the first finite numerical optimum, whose small
    # parameter error also perturbs the next predictive evidence.
    assert first.log_evidence+second.log_evidence==pytest.approx(z1+z2,abs=2e-9)
    np.testing.assert_array_equal(prior.mean,before_mean);np.testing.assert_array_equal(prior.covariance,before_covariance)


def test_exact_nonlinear_moments_are_recomputed_for_mean_and_covariance_trials():
    prior=Gaussian(np.array([.2]),np.array([[.4]]));y=.5;variance=.04;seen=[]
    def squared_signal(mean,covariance):
        m=float(mean[0]);P=float(covariance[0,0]);seen.append((m,P))
        predicted=m*m+P;S=2*P*P+4*m*m*P
        quadratic=.5*(predicted-y)**2/variance;trace=.5*S/variance;normalizer=.5*np.log(2*np.pi*variance)
        return ExpectedLikelihoodEvaluation(quadratic+trace+normalizer,
            np.array([2*m*(m*m+3*P-y)/variance]),
            np.array([[(3*m*m+3*P-y)/variance]]),
            diagnostics={'predicted_mean':predicted,'predicted_covariance':S},
            expected_negative_log_likelihood_terms=(quadratic,trace,normalizer))
    result=condition_variational(prior,squared_signal,max_iterations=100,gradient_tolerance=1e-9)
    assert len(seen)>1 and len({m for m,P in seen})>1 and len({P for m,P in seen})>1
    assert result.diagnostics['final_objective']<result.diagnostics['initial_objective']
    assert not result.diagnostics['strict_exact_model_lower_bound_claimed']
    assert not result.diagnostics['exact_nonlinear_evidence_claimed']


@pytest.mark.parametrize('iterations',[0,1])
def test_finite_budget_returns_valid_full_Gaussian_and_does_not_prevent_next_update(iterations):
    prior=Gaussian(np.zeros(2),np.array([[1.,.2],[.2,.7]]))
    result=condition_variational(prior,affine_evaluator(np.array([[1.,.4]]),np.array([2.]),np.array([[.01]])),
        max_iterations=iterations,gradient_tolerance=1e-12)
    assert np.isfinite(result.log_evidence)
    assert result.diagnostics['finite_nonstationary_iterate_used']
    assert not result.diagnostics['numerical_quality_is_stop_gate']
    if iterations==0:assert result.distribution is prior
    next_result=condition_variational(result.distribution,
        affine_evaluator(np.array([[.1,1.]]),np.array([.2]),np.array([[.1]])))
    assert np.isfinite(next_result.log_evidence)


def test_actual_initial_nonfinite_is_failure_but_later_unavailable_trials_keep_finite_result():
    prior=Gaussian(np.zeros(1),np.eye(1))
    with pytest.raises(ValueError,match='nonfinite'):
        condition_variational(prior,lambda m,P:ExpectedLikelihoodEvaluation(np.nan,np.zeros(1),np.zeros((1,1))))
    affine=affine_evaluator(np.eye(1),np.array([.8]),np.array([[.1]]))
    def limited(mean,covariance):
        if mean[0]>.3:return ExpectedLikelihoodEvaluation(np.nan,np.array([np.nan]),np.array([[np.nan]]))
        return affine(mean,covariance)
    result=condition_variational(prior,limited,max_iterations=5)
    assert result.diagnostics['invalid_trial_count']>0
    assert np.isfinite(result.log_evidence)and np.all(np.isfinite(result.distribution.covariance))
    assert all(row['finite_candidate_retained']and 'nonfinite'in row['reason']
               for row in result.diagnostics['invalid_trials'])
    assert result.log_evidence!=0.


def test_large_common_likelihood_constant_preserves_optimized_law_and_small_evidence_parts():
    prior=Gaussian(np.array([.2,-.3]),np.array([[.8,.13],[.13,.4]]))
    H=np.array([[1.,.2],[.3,-.7]]);y=np.array([.7,-.1]);R=np.array([[.2,.03],[.03,.1]])
    base=condition_variational(prior,affine_evaluator(H,y,R),max_iterations=100,gradient_tolerance=1e-9)
    shifted=condition_variational(prior,affine_evaluator(H,y,R,common_constant=1e20),max_iterations=100,gradient_tolerance=1e-9)
    np.testing.assert_allclose(shifted.distribution.mean,base.distribution.mean,atol=2e-9)
    np.testing.assert_allclose(shifted.distribution.covariance,base.distribution.covariance,atol=2e-9)
    assert log_evidence_parts_difference(shifted.log_evidence_parts,(-1e20,0.))==pytest.approx(base.log_evidence,abs=2e-11)


@pytest.mark.parametrize('common_constant',[0.,1e20])
def test_affine_analytic_initial_proposal_is_compared_using_the_original_prior_ELBO(common_constant):
    # An anisotropic prior makes replacing p0 by the seed readily visible.
    prior=Gaussian(np.array([.2,-.3]),np.array([[.8,.13],[.13,.4]]))
    H=np.array([[1.,.2],[.3,-.7]]);y=np.array([.7,-.1]);R=np.array([[.2,.03],[.03,.1]])
    seed,evidence=condition_gaussian(prior,y,R,H)
    result=condition_variational(prior,affine_evaluator(H,y,R,common_constant=common_constant),
        initial_distribution=seed)
    np.testing.assert_allclose(result.distribution.mean,seed.mean,atol=2e-15)
    np.testing.assert_allclose(result.distribution.covariance,seed.covariance,atol=2e-15)
    assert result.diagnostics['initial_distribution']['selected']
    assert not result.diagnostics['initial_distribution']['used_as_new_prior']
    assert result.diagnostics['completed_iterations']==0
    assert result.diagnostics['stationarity_verified']
    assert result.diagnostics['final_KL']>0.
    assert log_evidence_parts_difference(result.log_evidence_parts,(-common_constant,0.))==pytest.approx(evidence,abs=3e-13)


def test_worse_nonlinear_initial_proposal_is_rejected_then_original_prior_is_optimized():
    prior=Gaussian(np.array([.2]),np.array([[.4]]));seed=Gaussian(np.array([3.]),np.array([[.4]]))
    def evaluator(mean,covariance):
        m=float(mean[0]);P=float(covariance[0,0]);y=.5;R=.04
        mu=m*m+P;S=2*P*P+4*m*m*P
        normalizer=.5*np.log(2*np.pi*R)
        terms=(.5*(mu-y)**2/R,.5*S/R,normalizer)
        return ExpectedLikelihoodEvaluation(math.fsum(terms),
            np.array([2*m*(m*m+3*P-y)/R]),np.array([[(3*m*m+3*P-y)/R]]),
            expected_negative_log_likelihood_terms=terms)
    result=condition_variational(prior,evaluator,initial_distribution=seed,max_iterations=2)
    proposal=result.diagnostics['initial_distribution']
    assert proposal['considered'] and not proposal['selected']
    assert proposal['objective']>result.diagnostics['initial_objective']
    assert result.diagnostics['optimization_start_is_original_prior']
    assert result.diagnostics['final_objective']<result.diagnostics['initial_objective']


def test_zero_budget_does_not_evaluate_or_adopt_an_initial_proposal():
    prior=Gaussian(np.zeros(1),np.eye(1));H=np.eye(1);y=np.ones(1);R=np.array([[.1]])
    seed,_=condition_gaussian(prior,y,R,H);seen=[]
    result=condition_variational(prior,affine_evaluator(H,y,R,seen=seen),
        initial_distribution=seed,max_iterations=0)
    assert result.distribution is prior and len(seen)==1
    assert not result.diagnostics['initial_distribution']['considered']
    assert result.diagnostics['finite_nonstationary_iterate_used']
    fixed=Gaussian(np.array([.2]),np.zeros((1,1)));fixed_seen=[]
    fixed_result=condition_variational(fixed,affine_evaluator(H,y,R,seen=fixed_seen),initial_distribution=seed)
    assert fixed_result.distribution is fixed and len(fixed_seen)==1
    assert not fixed_result.diagnostics['initial_distribution']['considered']


def test_unavailable_optional_initial_proposal_is_recorded_without_losing_original_prior():
    prior=Gaussian(np.zeros(1),np.eye(1));singular=Gaussian(np.zeros(1),np.zeros((1,1)))
    result=condition_variational(prior,affine_evaluator(np.eye(1),np.array([.2]),np.array([[.1]])),
        initial_distribution=singular)
    proposal=result.diagnostics['initial_distribution']
    assert proposal['considered']and not proposal['selected']
    assert 'unavailable optional initial proposal'in proposal['status']
    assert proposal['reason']
    assert result.diagnostics['optimization_start_is_original_prior']
    assert np.isfinite(result.log_evidence)
