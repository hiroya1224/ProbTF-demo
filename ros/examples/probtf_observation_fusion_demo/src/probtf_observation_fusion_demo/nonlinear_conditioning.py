"""Prior-once support-space nonlinear Gaussian conditioning.

The returned proxy uses the finite best nonlinear iterate as its mean and
the Gauss--Newton precision there as its covariance. Its explicitly reported
GN-Laplace evidence is an approximation, not a nonlinear moment integral.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support


@dataclass(frozen=True)
class NonlinearConditioning:
    distribution: Gaussian
    log_evidence: float
    log_evidence_parts: tuple
    diagnostics: dict


def condition_nonlinear(prior, residual_and_jacobian, observation_covariance, *,
                        max_iterations=40, gradient_tolerance=1e-8,
                        max_backtracks=20, armijo=1e-4):
    """Condition one fixed joint prior on one fixed nonlinear residual family.

``residual_and_jacobian(x)`` evaluates the candidate in the ORIGINAL joint
chart and returns residual[d], derivative[d,n]. The covariance is the native
proper observation covariance, with a single fixed observation base measure.
All optimization iterates evaluate the same prior and the same observation.
No posterior is fed back as an extra independent prior or observation.
"""
    if (isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 0
            or isinstance(max_backtracks, bool) or int(max_backtracks) != max_backtracks or max_backtracks < 0
            or not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.
            or not math.isfinite(armijo) or not 0. < armijo < .5):
        raise ValueError('Nonlinear conditioning needs valid work limits and optimizer parameters.')
    mean, covariance = np.asarray(prior.mean), np.asarray(prior.covariance)
    support = covariance_support(covariance, 'frozen nonlinear joint prior')
    root = support.basis*np.sqrt(support.eigenvalues)
    rank, dimension = root.shape[1], mean.size
    noise = np.array(observation_covariance, dtype=float, copy=True)
    if (noise.ndim != 2 or noise.shape[0] != noise.shape[1] or not noise.shape[0]
            or not np.all(np.isfinite(noise))):
        raise ValueError('The native observation covariance must be a finite nonempty square matrix.')
    # Proper SPD is the raw likelihood contract, not a numerical quality gate.
    # Do not replace failed likelihood covariance by a floor or pseudoinverse.
    noise_skew = float(np.max(np.abs(noise-noise.T), initial=0.))
    noise = .5*noise+.5*noise.T
    noise_root = np.linalg.cholesky(noise)
    log_noise_det = 2.*float(np.log(np.diag(noise_root)).sum())
    observation_dimension = noise.shape[0]
    initial_hash = hashlib.sha256(mean.tobytes()+covariance.tobytes()).hexdigest()
    calls = 0

    def evaluate(whitened):
        nonlocal calls
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            candidate = np.array(mean+root@whitened, copy=True)
            candidate.setflags(write=False)
            residual, jacobian = residual_and_jacobian(candidate)
            residual, jacobian = np.asarray(residual, dtype=float), np.asarray(jacobian, dtype=float)
            if residual.shape != (observation_dimension,) or jacobian.shape != (observation_dimension, dimension):
                raise ValueError('The nonlinear residual/Jacobian differs from its fixed observation/prior dimensions.')
            if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
                raise ValueError('The nonlinear observation returned a nonfinite residual or derivative.')
            weighted = solve_triangular(noise_root, residual, lower=True, check_finite=False)
            sensitivity = solve_triangular(noise_root, jacobian@root, lower=True, check_finite=False)
            objective = .5*float(whitened@whitened+weighted@weighted)
            gradient = whitened+sensitivity.T@weighted
            metric = np.eye(rank)+sensitivity.T@sensitivity
        if (not math.isfinite(objective) or not np.all(np.isfinite(gradient))
                or not np.all(np.isfinite(metric))):
            raise ValueError('The nonlinear conditioning objective or GN system is nonfinite.')
        calls += 1
        return dict(xi=whitened.copy(), candidate=candidate, residual=residual.copy(),
            objective=objective, gradient=gradient, metric=metric)

    current = evaluate(np.zeros(rank))
    initial_objective = current['objective']
    history, trials, completed = [], [], 0
    termination = 'configured finite iteration budget'
    for iteration in range(int(max_iterations)):
        score = float(np.max(np.abs(current['gradient']), initial=0.))
        if score <= gradient_tolerance:
            termination = 'finite objective score satisfies optimizer tolerance'
            break
        factor = np.linalg.cholesky(current['metric'])
        direction = -cho_solve((factor, True), current['gradient'], check_finite=False)
        slope = float(current['gradient']@direction)
        accepted = None
        for backtrack in range(int(max_backtracks)+1):
            scale = .5**backtrack
            record = dict(iteration=iteration, backtrack=backtrack, scale=scale, accepted=False)
            try:
                trial = evaluate(current['xi']+scale*direction)
                bound = current['objective']+armijo*scale*slope
                record.update(objective=trial['objective'], armijo_bound=bound)
                if trial['objective'] <= bound:
                    accepted = trial
                    record['accepted'] = True
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                # An unavailable trial can be shortened without discarding
                # the original observation or the current finite iterate.
                record['actual_trial_error'] = type(error).__name__+': '+str(error)
            trials.append(record)
            if accepted is not None:
                break
        completed += 1
        if accepted is None:
            termination = 'line search returned the retained finite best iterate'
            break
        current = accepted
        history.append(dict(iteration=iteration+1, objective=current['objective'],
            gradient_inf=float(np.max(np.abs(current['gradient']), initial=0.))))

    # The GN precision is positive definite by I + A.T A. A real failed
    # factorization is not converted into a synthetic covariance/evidence.
    factor = np.linalg.cholesky(current['metric'])
    whitened_covariance = (cho_solve((factor, True), np.eye(rank), check_finite=False)
                          if rank else np.empty((0,0)))
    posterior_covariance = root@whitened_covariance@root.T
    posterior_covariance = .5*posterior_covariance+.5*posterior_covariance.T
    posterior = Gaussian(current['candidate'], posterior_covariance)
    log_metric_det = 2.*float(np.log(np.diag(factor)).sum())
    evidence_parts = (-current['objective'], -.5*log_noise_det,
        -.5*log_metric_det, -.5*observation_dimension*math.log(2.*math.pi))
    evidence = math.fsum(evidence_parts)
    if not math.isfinite(evidence):
        raise ValueError('The finite-iterate GN-Laplace evidence is nonfinite.')
    score = float(np.max(np.abs(current['gradient']), initial=0.))
    diagnostics = dict(method='fixed-prior finite-iterate Gauss-Newton/Laplace proxy',
        prior_support_rank=rank, joint_dimension=dimension, observation_dimension=observation_dimension,
        prior_numeric_sha256=initial_hash, original_prior_held_fixed=True,
        observation_covariance_held_fixed=True, repeated_observation_multiplication=False,
        observation_covariance_maximum_skew=noise_skew,
        observation_covariance_symmetric_part_used=True,
        candidate_mean_is_finite_modal_geometry=True, core_O_P_mean_added_to_point_likelihood=False,
        exact_nonlinear_moments_claimed=False, exact_nonlinear_evidence_claimed=False,
        exact_nonlinear_Hessian_used=False, covariance_method='L (I + J_whitened.T J_whitened)^-1 L.T',
        evidence_method='GN-Laplace normalizer at finite best nonlinear iterate; no completed-square shift',
        support_regularization_added=False, covariance_clip_applied=False, monte_carlo_used=False,
        initial_objective=initial_objective, final_objective=current['objective'],
        final_original_chart_mean=posterior.mean.tolist(), final_whitened_mean=current['xi'].tolist(),
        final_native_residual=current['residual'].tolist(), final_gradient_inf=score,
        stationarity_verified=score <= gradient_tolerance,
        finite_nonstationary_iterate_used=score > gradient_tolerance,
        numerical_quality_is_stop_gate=False, max_iterations=int(max_iterations),
        completed_iterations=completed, gradient_tolerance=gradient_tolerance,
        termination_reason=termination, evaluator_calls=calls,
        objective_history=history, line_search=trials,
        logdet_observation_covariance=log_noise_det, logdet_GN_precision=log_metric_det,
        log_evidence_parts=list(evidence_parts), log_evidence=evidence)
    return NonlinearConditioning(posterior, evidence, evidence_parts, diagnostics)
