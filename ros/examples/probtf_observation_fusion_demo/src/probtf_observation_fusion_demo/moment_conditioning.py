"""One analytic joint conditioning step from the unchanged core moments.

The observation-space mean includes the core's Bingham-derived O(P)
correction. Covariance and full body/observation cross covariance are the
actual propagated quantities. No candidate-point optimization is performed.
"""
from dataclasses import dataclass
import math
import numpy as np

from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support, require_in_support
from .joint_batch_fusion import add_log_evidence_parts


@dataclass(frozen=True)
class MomentConditioning:
    distribution: Gaussian
    log_evidence: float
    log_evidence_parts: tuple
    diagnostics: dict


def condition_moments(prior, predicted_mean, predicted_covariance,
                      body_observation_cross_covariance, jacobian,
                      observed_mean, observation_covariance):
    """Condition once using mu_y, S_y, C_zy and the same frozen joint prior.

The present core has S_y=H P H.T and C_zy=P H.T, including all cross
blocks. The Joseph expression below is therefore the joint Gaussian
conditional covariance, evaluated as a sum of actual covariance terms.
The innovation and evidence use the supplied propagated S_y, while the gain
uses the supplied full C_zy. Differences in these core identities are
reported as numerical diagnostics, never as tolerance-based stop gates.
"""
    mean=np.asarray(prior.mean);covariance=np.asarray(prior.covariance)
    mu=np.asarray(predicted_mean,dtype=float);S=np.asarray(predicted_covariance,dtype=float)
    C=np.asarray(body_observation_cross_covariance,dtype=float);H=np.asarray(jacobian,dtype=float)
    y=np.asarray(observed_mean,dtype=float);R=np.asarray(observation_covariance,dtype=float)
    n=mean.size;d=mu.size
    if (mu.ndim!=1 or y.shape!=(d,) or S.shape!=(d,d) or R.shape!=(d,d)
            or C.shape!=(n,d) or H.shape!=(d,n)):
        raise ValueError('Propagated observation, full cross, and joint prior dimensions differ.')
    if not all(np.all(np.isfinite(value))for value in (mu,S,C,H,y,R)):
        raise ValueError('Propagated observation conditioning needs finite moments and native likelihood parameters.')
    # These are the symmetric covariance tensors, not an eigenvalue floor.
    S=.5*S+.5*S.T;R=.5*R+.5*R.T
    covariance_support(S,'propagated observation covariance')
    covariance_support(R,'native observation covariance')
    innovation=S+R
    support=covariance_support(innovation,'propagated observation innovation')
    residual=y-mu
    require_in_support(residual,support,'propagated observation residual')
    gain=C@support.inverse
    posterior_mean=mean+gain@residual
    remaining=np.eye(n)-gain@H
    posterior_covariance=remaining@covariance@remaining.T+gain@R@gain.T
    posterior_covariance=.5*posterior_covariance+.5*posterior_covariance.T
    posterior=Gaussian(posterior_mean,posterior_covariance)
    normalizer=-.5*(support.rank*math.log(2.*math.pi)+support.log_pdet)
    quadratic=-.5*float(residual@support.inverse@residual)
    parts=add_log_evidence_parts((0.,0.),(normalizer,quadratic))
    evidence=math.fsum(parts)
    if not math.isfinite(evidence):
        raise ValueError('Propagated Gaussian overlap evidence is nonfinite.')
    implied_S=H@covariance@H.T;implied_C=covariance@H.T
    local_mean=mu+S@support.inverse@residual
    local_covariance=S-S@support.inverse@S
    diagnostics=dict(method='one analytic Gaussian conditional from propagated core moments',
        original_prior_held_fixed=True,observation_added_once=True,
        repeated_observation_multiplication=False,point_optimization_used=False,
        optimization_iterations=0,gauss_newton_used=False,laplace_body_evidence_used=False,
        core_O_P_mean_used=True,full_body_observation_cross_used=True,
        joint_dimension=n,observation_dimension=d,innovation_support_rank=support.rank,
        prior_support_rank=prior.support.rank,innovation_residual=residual.tolist(),
        innovation_covariance=innovation.tolist(),gain=gain.tolist(),
        propagated_covariance_vs_HPH_max_abs=float(np.max(np.abs(S-implied_S),initial=0.)),
        propagated_cross_vs_PH_max_abs=float(np.max(np.abs(C-implied_C),initial=0.)),
        local_conditional_mean=local_mean.tolist(),
        local_conditional_covariance=(.5*local_covariance+.5*local_covariance.T).tolist(),
        local_marginal_mean_closure_error=(mu+H@(posterior.mean-mean)-local_mean).tolist(),
        local_marginal_covariance_closure_max_abs=float(np.max(np.abs(H@posterior.covariance@H.T-local_covariance),initial=0.)),
        covariance_method='Joseph form of the same joint Gaussian conditional; no jitter or clipping',
        evidence_method='normalized Gaussian overlap from propagated mean and full innovation covariance',
        evidence_normalizer=normalizer,evidence_quadratic=quadratic,
        log_evidence=evidence,log_evidence_parts=list(parts),
        monte_carlo_used=False,numerical_quality_is_stop_gate=False)
    return MomentConditioning(posterior,evidence,parts,diagnostics)
