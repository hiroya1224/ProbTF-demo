"""Full-Gaussian variational conditioning with one immutable original prior.

An evaluator returns E_q[-log likelihood] INCLUDING its native Gaussian
normalizer and its full mean/covariance derivatives. These expectations use
the declared propagated-moment proxy. Consequently -F is an approximate
moment-ELBO score, not a guaranteed bound for an exact nonlinear model.
"""
from dataclasses import dataclass,field
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support,require_in_support,require_same_support
from .joint_batch_fusion import add_log_evidence_parts,log_evidence_parts_difference


@dataclass(frozen=True)
class ExpectedLikelihoodEvaluation:
    """Full symmetric Frobenius covariance-gradient convention.

For a symmetric perturbation changing both P_ij and P_ji by epsilon,
the differential contribution is 2*gradient_covariance[i,j]*epsilon.
Optional terms are additive NLL pieces (e.g. quadratic/trace/normalizer),
retained separately so a large common scale cannot erase smaller terms.
"""
    expected_negative_log_likelihood: float
    gradient_mean: np.ndarray
    gradient_covariance: np.ndarray
    diagnostics: dict=field(default_factory=dict)
    expected_negative_log_likelihood_terms: object=None


@dataclass(frozen=True)
class VariationalConditioning:
    distribution: Gaussian
    log_evidence: float
    log_evidence_parts: tuple
    diagnostics: dict


def _get(value,name,default=None):
    return value.get(name,default)if isinstance(value,Mapping)else getattr(value,name,default)


def _expm1_minus_argument(value):
    # Stable evaluation of the same analytic function near zero, not clipping
    # log Cholesky coordinates or adding a covariance regularizer.
    value=np.asarray(value,dtype=float)
    out=np.expm1(value)-value
    small=np.abs(value)<1e-3
    z=value[small]
    out[small]=z*z*(.5+z*(1./6.+z*(1./24.+z*(1./120.+z/720.))))
    return out


class _WhitenedObjective:
    """One fixed prior support; all full-covariance candidates share it."""
    def __init__(self,prior,evaluate):
        self.prior=prior;self.evaluate_likelihood=evaluate
        support=covariance_support(prior.covariance,'fixed variational original prior')
        self.support=support
        self.root=support.basis*np.sqrt(support.eigenvalues)
        self.inverse_root=(support.basis/np.sqrt(support.eigenvalues)).T
        self.rank=self.root.shape[1];self.dimension=prior.mean.size
        self.lower=np.tril_indices(self.rank);self.diagonal=self.lower[0]==self.lower[1]
        self.parameter_count=self.rank+len(self.lower[0])
        self.prior_sha256=hashlib.sha256(prior.mean.tobytes()+prior.covariance.tobytes()).hexdigest()

    def parameters_from_distribution(self,distribution):
        """An optional initial proposal, expressed in the ORIGINAL prior chart."""
        if (distribution.mean.shape!=self.prior.mean.shape or
                distribution.covariance.shape!=self.prior.covariance.shape):
            raise ValueError('An initial Gaussian proposal differs from its original prior dimensions.')
        require_same_support(self.support,distribution.support,'initial variational proposal')
        difference=distribution.mean-self.prior.mean
        require_in_support(difference,self.support,'initial variational proposal mean shift')
        a=self.inverse_root@difference
        V=self.inverse_root@distribution.covariance@self.inverse_root.T
        L=np.linalg.cholesky(.5*V+.5*V.T)
        packed=L[self.lower].copy()
        packed[self.diagonal]=np.log(np.diag(L))
        return np.r_[a,packed]

    def evaluate(self,parameters):
        x=np.array(parameters,dtype=float,copy=True)
        if x.shape!=(self.parameter_count,)or not np.all(np.isfinite(x)):
            raise ValueError('Variational Gaussian parameters must be finite with their fixed support dimensions.')
        r=self.rank;a=x[:r];packed=x[r:];logs=packed[self.diagonal]
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            L=np.zeros((r,r));L[self.lower]=packed
            diagonal=np.exp(logs)
            if np.any(diagonal==0.):
                raise FloatingPointError('A variational Cholesky diagonal underflowed to zero.')
            L[np.diag_indices(r)]=diagonal
            at_prior=bool(np.all(x==0.))
            if at_prior:
                # The white identity IS the original prior. Avoid replacing
                # that exact identity with an eigensystem round-trip.
                mean=np.array(self.prior.mean,copy=True)
                covariance=np.array(self.prior.covariance,copy=True)
            else:
                mean=self.prior.mean+self.root@a
                transformed=self.root@L
                covariance=transformed@transformed.T
                covariance=.5*covariance+.5*covariance.T
            mean.setflags(write=False);covariance.setflags(write=False)
            raw=self.evaluate_likelihood(mean,covariance)
            nll=float(_get(raw,'expected_negative_log_likelihood'))
            gm=np.asarray(_get(raw,'gradient_mean'),dtype=float)
            gP=np.asarray(_get(raw,'gradient_covariance'),dtype=float)
            if gm.shape!=(self.dimension,)or gP.shape!=(self.dimension,self.dimension):
                raise ValueError('Expected-likelihood derivatives differ from the original joint Gaussian dimensions.')
            if not math.isfinite(nll)or not np.all(np.isfinite(gm))or not np.all(np.isfinite(gP)):
                raise ValueError('Expected likelihood or its full Gaussian derivatives are nonfinite.')
            skew=float(np.max(np.abs(gP-gP.T),initial=0.))
            symmetric=.5*gP+.5*gP.T
            off_diagonal=packed[~self.diagonal]
            kl=.5*float(a@a+off_diagonal@off_diagonal+_expm1_minus_argument(2.*logs).sum())
            projected=self.root.T@symmetric@self.root
            inverse_transpose=(solve_triangular(L.T,np.eye(r),lower=False,check_finite=False)
                               if r else np.empty((0,0)))
            gradient_a=a+self.root.T@gm
            gradient_L=L-inverse_transpose+2.*projected@L
            gradient_packed=gradient_L[self.lower].copy()
            gradient_packed[self.diagonal]*=diagonal
            gradient=np.r_[gradient_a,gradient_packed]
        if not math.isfinite(kl)or not np.all(np.isfinite(gradient)):
            raise ValueError('Variational KL or full Cholesky derivatives are nonfinite.')
        terms=_get(raw,'expected_negative_log_likelihood_terms')
        terms=(nll,)if terms is None else tuple(float(value)for value in terms)
        if not terms or not all(math.isfinite(value)for value in terms):
            raise ValueError('Expected-likelihood normalizer terms must be finite and nonempty.')
        parts=add_log_evidence_parts((0.,0.),(kl,)+terms)
        if not all(math.isfinite(value)for value in parts):
            raise ValueError('The variational objective is nonfinite.')
        objective=math.fsum(parts)
        distribution=self.prior if at_prior else Gaussian(mean,covariance)
        return dict(parameters=x,distribution=distribution,kl=kl,nll=nll,
            nll_terms=terms,nll_terms_scalar_discrepancy=math.fsum(terms)-nll,
            objective=objective,objective_parts=parts,gradient=gradient,
            gradient_inf=float(np.max(np.abs(gradient),initial=0.)),
            covariance_gradient_maximum_skew=skew,
            evaluator_diagnostics=deepcopy(_get(raw,'diagnostics',{})or {}))


def condition_variational(prior,evaluate,*,max_iterations=24,
                          gradient_tolerance=1e-6,max_line_search=20,
                          initial_distribution=None):
    """Minimize KL(q||fixed p0)+E_q[-log L] over the full Gaussian family.

``evaluate(mean,covariance)`` must return the data contract documented by
ExpectedLikelihoodEvaluation (an object or mapping is accepted). Mean and
covariance are in the ORIGINAL prior chart, with fixed support whitening
B=U sqrt(D). The callback must recompute the SAME declared propagated-moment
likelihood for every candidate. It must not return a point-MAP loss or use a
trial posterior as a new prior. Native/parent weights are applied outside.
An optional Gaussian is only a starting proposal: compare its SAME objective
with the original-prior candidate, then optimize the better finite point.
The KL reference never becomes this proposal. Zero budget/rank zero retains
the original prior without evaluating a starting proposal.

Finite nonstationarity or a solver work limit returns the best finite q,
with truthful diagnostics. A truly undefined initial objective fails. Later
undefined trials are recorded as unavailable (+inf with undefined gradient)
so a line search may shorten them; they never become zero information.
"""
    if (isinstance(max_iterations,bool)or int(max_iterations)!=max_iterations or max_iterations<0
            or isinstance(max_line_search,bool)or int(max_line_search)!=max_line_search or max_line_search<1
            or not math.isfinite(gradient_tolerance)or gradient_tolerance<=0.):
        raise ValueError('Variational conditioning needs valid finite work settings.')
    objective=_WhitenedObjective(prior,evaluate)
    initial=objective.evaluate(np.zeros(objective.parameter_count))
    best=initial;last=initial;valid_calls=1;invalid_trials=[]
    history=[dict(evaluation=0,objective=initial['objective'],kl=initial['kl'],
        expected_negative_log_likelihood=initial['nll'],gradient_inf=initial['gradient_inf'],best=True)]
    initial_parts=initial['objective_parts']
    seed_record=dict(provided=initial_distribution is not None,considered=False,
        selected=False,proposal_only=True,used_as_new_prior=False,status='not provided')
    if initial_distribution is not None:
        if objective.rank==0 or int(max_iterations)==0:
            seed_record['status']='not evaluated: rank-zero prior or zero iteration budget'
        else:
            seed_record['considered']=True
            try:
                seed_parameters=objective.parameters_from_distribution(initial_distribution)
                seed=objective.evaluate(seed_parameters)
                valid_calls+=1
                chosen=(seed['objective_parts']<best['objective_parts']or
                    (seed['objective_parts']==best['objective_parts']and seed['gradient_inf']<best['gradient_inf']))
                seed_record.update(selected=chosen,status='evaluated on same fixed-prior objective',
                    objective=seed['objective'],objective_parts=list(seed['objective_parts']),
                    gradient_inf=seed['gradient_inf'])
                history.append(dict(evaluation=valid_calls-1,objective=seed['objective'],kl=seed['kl'],
                    expected_negative_log_likelihood=seed['nll'],gradient_inf=seed['gradient_inf'],
                    best=chosen,initial_proposal=True))
                if chosen:best=seed;last=seed
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError)as error:
                seed_record.update(status='unavailable optional initial proposal; original prior candidate retained',
                    reason=type(error).__name__+': '+str(error))
    start=best

    def value_gradient(parameters):
        nonlocal best,last,valid_calls
        if last is not None and np.array_equal(parameters,last['parameters']):
            current=last
        else:
            try:
                current=objective.evaluate(parameters)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError)as error:
                invalid_trials.append(dict(reason=type(error).__name__+': '+str(error),
                    parameter_sha256=hashlib.sha256(np.asarray(parameters,dtype=float).tobytes()).hexdigest(),
                    finite_candidate_retained=True))
                # Undefined derivatives are represented as undefined, never
                # as a fabricated zero gradient of an unavailable candidate.
                return math.inf,np.full(objective.parameter_count,np.nan)
            valid_calls+=1;last=current
            improved=(current['objective_parts']<best['objective_parts']or
                (current['objective_parts']==best['objective_parts']and current['gradient_inf']<best['gradient_inf']))
            if improved:best=current
            history.append(dict(evaluation=valid_calls-1,objective=current['objective'],
                kl=current['kl'],expected_negative_log_likelihood=current['nll'],
                gradient_inf=current['gradient_inf'],best=improved))
        # Subtract a fixed objective scale, preserving its addition residual.
        # This changes neither the optimized q nor its reported evidence.
        centered=log_evidence_parts_difference(current['objective_parts'],initial_parts)
        return centered,current['gradient'].copy()

    solver_result=None;solver_error=None;recorded_warnings=[]
    if objective.rank==0:
        reason='rank-zero original prior: exact Dirac candidate retained'
    elif int(max_iterations)==0:
        reason='configured zero iteration budget: finite original-prior candidate retained'
    elif start['gradient_inf']<=gradient_tolerance:
        reason='starting full Gaussian parameter gradient satisfies configured tolerance'
    else:
        reason='L-BFGS-B returned the best finite full Gaussian candidate'
        with warnings.catch_warnings(record=True)as seen:
            warnings.simplefilter('always')
            try:
                solver_result=minimize(value_gradient,start['parameters'],jac=True,method='L-BFGS-B',
                    options=dict(maxiter=int(max_iterations),maxls=int(max_line_search),
                        gtol=float(gradient_tolerance),ftol=0.,maxcor=10))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError)as error:
                solver_error=type(error).__name__+': '+str(error)
                reason='optimizer arithmetic unavailable: best finite full Gaussian candidate retained'
            recorded_warnings=[str(item.message)for item in seen]
    log_parts=(-best['objective_parts'][0],-best['objective_parts'][1])
    evidence=math.fsum(log_parts)
    score=best['gradient_inf']
    diagnostics=dict(method='full-Gaussian fixed-prior propagated-moment variational conditioning',
        objective_definition='KL(q||original p0)+expected native Gaussian negative log likelihood including normalizer',
        evidence_method='negative optimized moment-ELBO proxy objective; not a Laplace normalizer',
        exact_nonlinear_evidence_claimed=False,strict_exact_model_lower_bound_claimed=False,
        affine_limit='exact Gaussian posterior/evidence at the variational optimum; finite optimization error recorded',
        original_prior_held_fixed=True,original_prior_numeric_sha256=objective.prior_sha256,
        observation_added_once=True,repeated_observation_multiplication=False,
        optimized_variables='whitened Gaussian mean and every lower-triangular Cholesky entry; diagonal logarithms',
        covariance_gradient_convention='full symmetric Frobenius; paired off-diagonal perturbation contributes twice',
        prior_support_rank=objective.rank,joint_dimension=objective.dimension,
        parameter_count=objective.parameter_count,full_joint_cross_covariance_optimized=True,
        whitening='B=U sqrt(D), P0=B B.T; fixed original support, not an orthonormal basis alone',
        covariance_regularization_added=False,covariance_clip_applied=False,monte_carlo_used=False,
        point_MAP_used=False,Gauss_Newton_used=False,body_Laplace_evidence_used=False,
        initial_objective=initial['objective'],final_objective=best['objective'],
        initial_distribution=seed_record,optimization_start_objective=start['objective'],
        optimization_start_is_original_prior=bool(np.all(start['parameters']==0.)),
        final_objective_parts=list(best['objective_parts']),final_KL=best['kl'],
        final_expected_negative_log_likelihood=best['nll'],
        final_expected_negative_log_likelihood_terms=list(best['nll_terms']),
        likelihood_terms_scalar_discrepancy=best['nll_terms_scalar_discrepancy'],
        final_gradient_inf=score,gradient_tolerance=float(gradient_tolerance),
        stationarity_verified=score<=gradient_tolerance,
        finite_nonstationary_iterate_used=score>gradient_tolerance,
        numerical_quality_is_stop_gate=False,max_iterations=int(max_iterations),
        max_line_search=int(max_line_search),valid_evaluator_calls=valid_calls,
        invalid_trial_count=len(invalid_trials),invalid_trials=invalid_trials,
        optimizer_method='unbounded L-BFGS-B; no parameter clipping',
        optimizer_success=None if solver_result is None else bool(solver_result.success),
        optimizer_status=None if solver_result is None else int(solver_result.status),
        optimizer_message=None if solver_result is None else str(solver_result.message),
        completed_iterations=0 if solver_result is None else int(solver_result.nit),
        optimizer_exception=solver_error,optimizer_warnings=recorded_warnings,
        termination_reason=reason,objective_history=history,
        optimization_centering='subtract fixed initial objective scale with retained addition residual',
        source_evaluator_diagnostics=best['evaluator_diagnostics'],
        log_evidence=evidence,log_evidence_parts=list(log_parts))
    return VariationalConditioning(best['distribution'],evidence,log_parts,diagnostics)
