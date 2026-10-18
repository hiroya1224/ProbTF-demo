"""One recursive log-K Gaussian, updated by the current rotation factor only.

The existing joint-rotation factor and auxiliary mechanics are reused. Past
factor functions are not retained or evaluated. The local posterior Gaussian
projection computes no evidence and is not an exact nonlinear Bayes update.
"""

import hashlib
import math
import time

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from .stiffness_identification import StiffnessIdentifier, _atomic_stiffness_factors


class GaussianProjectionUnavailable(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def project_gaussian_factor(prior, factor, *, max_iterations=5,
                            gradient_tolerance=1.e-6, curvature_step=1.e-4,
                            maximum_line_search=8, prior_seed_scan=True):
    """Fit a local Gaussian to N(x; prior) times one differentiable factor.

    Initialize from finite deterministic prior-coordinate probes, then
    optimize the mean in prior-whitened coordinates. At the best finite
    evaluated point, invert the symmetric local negative-log curvature.
    This is a finite-iterate posterior Laplace projection, not Laplace
    evidence. A finite optimizer nonconvergence is diagnostic only.
    """
    if (isinstance(max_iterations, bool) or int(max_iterations) != max_iterations
            or max_iterations < 0 or gradient_tolerance <= 0. or curvature_step <= 0.
            or maximum_line_search < 1 or not np.isfinite(gradient_tolerance)
            or not np.isfinite(curvature_step)):
        raise ValueError('The simple Gaussian projection needs valid finite numerical settings.')
    diagnostics = dict(method='prior-whitened finite MAP iteration and local posterior Gaussian curvature',
        evidence_computed=False, monte_carlo_used=False, prior_fixed_during_update=True,
        maximum_iterations=int(max_iterations), curvature_step=float(curvature_step),
        finite_nonconvergence_is_fatal=False, covariance_regularization_added=False,
        old_mean=prior.mean.tolist(), old_covariance=prior.covariance.tolist())
    try:
        scale = np.linalg.cholesky(prior.covariance)
        dimension = prior.mean.size
        cache, best = {}, [None]
        def objective(z):
            z = np.asarray(z, dtype=float)
            if z.shape != (dimension,) or not np.all(np.isfinite(z)):
                raise ValueError('A simple Gaussian posterior trial is nonfinite or has the wrong dimension.')
            key = z.tobytes()
            if key not in cache:
                x = prior.mean+scale @ z
                log_factor, score = factor(x)
                score = np.asarray(score, dtype=float)
                if not np.isfinite(log_factor) or score.shape != x.shape or not np.all(np.isfinite(score)):
                    raise ValueError('The current joint-rotation factor returned a nonfinite value or score.')
                value, gradient = .5*float(z @ z)-float(log_factor), z-scale.T @ score
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError('The simple Gaussian posterior target is nonfinite.')
                cache[key] = float(value), gradient.copy()
                if best[0] is None or value < best[0][0]:
                    best[0] = (float(value), z.copy(), gradient.copy())
            return cache[key]
        zero = np.zeros(dimension)
        centre_value = objective(zero)[0]
        scan_started = time.perf_counter()
        seeds = [dict(label='prior_mean', whitened=zero.tolist(),
                      target_negative_log=centre_value)]
        unavailable_seeds = []
        if prior_seed_scan and max_iterations:
            directions = [(f'Cholesky_column_{index}', direction)
                          for index, direction in enumerate(np.eye(dimension))]
            directions.append(('Cholesky_column_sum', np.ones(dimension)))
            for label, direction in directions:
                for sign in (1., -1.):
                    point = sign*direction
                    name = ('plus_' if sign > 0. else 'minus_')+label
                    try:
                        value, _ = objective(point)
                        seeds.append(dict(label=name, whitened=point.tolist(),
                                          target_negative_log=float(value)))
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                        unavailable_seeds.append(dict(label=name, whitened=point.tolist(),
                            unavailable_reason=type(error).__name__+': '+str(error)))
        initial = best[0][1].copy()
        selected_seed = min(seeds, key=lambda row: row['target_negative_log'])
        diagnostics['prior_seed_scan'] = dict(
            enabled=bool(prior_seed_scan and max_iterations),
            definition='prior mean plus/minus each Cholesky column and plus/minus their coherent sum; optimization initialization only, not quadrature or posterior samples',
            original_prior_and_current_factor_fixed=True, probes=seeds,
            unavailable_probes=unavailable_seeds,
            selected_seed_label=selected_seed['label'],
            initial_whitened=initial.tolist(),
            work_seconds=time.perf_counter()-scan_started)
        if max_iterations:
            try:
                result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                    options=dict(maxiter=int(max_iterations), maxls=int(maximum_line_search),
                                 ftol=0., gtol=float(gradient_tolerance)))
                diagnostics.update(optimizer_success=bool(result.success),
                    optimizer_status=int(result.status), optimizer_message=str(result.message),
                    optimizer_iterations=int(result.nit))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                # A failed trial does not erase an earlier finite point. The
                # covariance calculation below decides actual availability.
                diagnostics.update(optimizer_success=False, optimizer_message=str(error),
                                   optimizer_trial_unavailable=True)
        else:
            diagnostics.update(optimizer_success=False, optimizer_iterations=0,
                optimizer_message='explicit zero mean-iteration budget; local covariance still evaluated')
        selected_value, centre, score = best[0]
        # Freeze the selected mean before curvature probes: probes are
        # derivatives, not additional independent observations or samples.
        hessian = np.empty((dimension, dimension))
        for index, direction in enumerate(np.eye(dimension)):
            plus = objective(centre+curvature_step*direction)[1]
            minus = objective(centre-curvature_step*direction)[1]
            hessian[:, index] = (plus-minus)/(2.*curvature_step)
        if not np.all(np.isfinite(hessian)):
            raise ValueError('The local posterior Gaussian curvature is nonfinite.')
        diagnostics.update(target_negative_log_at_selected=float(selected_value),
            selected_whitened_score=score.tolist(), selected_score_inf=float(np.max(np.abs(score), initial=0.)),
            stationarity_tolerance_met=bool(np.max(np.abs(score), initial=0.) <= gradient_tolerance),
            curvature_skew_inf=float(np.max(np.abs(hessian-hessian.T), initial=0.)),
            candidate_mean=(prior.mean+scale @ centre).tolist(),
            posterior_curvature_whitened=(.5*hessian+.5*hessian.T).tolist(),
            factor_evaluation_count=len(cache))
        # This symmetric part represents the same quadratic form. No
        # eigenvalue floor, ridge, damping covariance, or jitter is added.
        hessian = .5*hessian+.5*hessian.T
        root = np.linalg.cholesky(hessian)
        covariance = scale @ cho_solve((root, True), np.eye(dimension)) @ scale.T
        covariance = .5*covariance+.5*covariance.T
        np.linalg.cholesky(covariance)  # Required for the next normalized Gaussian.
        posterior = Gaussian(prior.mean+scale @ centre, covariance)
        diagnostics.update(projection_available=True, new_mean=posterior.mean.tolist(),
            new_covariance=posterior.covariance.tolist(),
            approximation='one local normalized Gaussian at a finite posterior target iterate; no evidence or global mode claim')
        return posterior, diagnostics
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        diagnostics.update(projection_available=False,
            unavailable_reason=type(error).__name__+': '+str(error), prior_retained=True)
        raise GaussianProjectionUnavailable(diagnostics['unavailable_reason'], diagnostics) from error


class SimpleStiffnessIdentifier(StiffnessIdentifier):
    """Keep one full log-K Gaussian and one current observation's metadata."""

    def __init__(self, urdf_path, body_edge_ids, settings):
        settings = dict(settings, matching_target='joint_rotation',
                        bayes_factor_surprise=dict(enabled=False, change_probability=0.))
        super().__init__(urdf_path, body_edge_ids, settings)
        if len(self.components) != 1:
            raise ValueError('The simple stiffness reference requires one initial log-K Gaussian.')
        self._simple_distribution = self.components[0].distribution
        self.simple_projection_count = 0

    def _density_record(self):
        distribution = getattr(self, '_simple_distribution', self.components[0].distribution)
        digest = hashlib.sha256(b'ProbTF simple normalized logK Gaussian v1')
        digest.update(distribution.mean.tobytes()+distribution.covariance.tobytes())
        return dict(density_id=digest.hexdigest(), matching_target='joint_rotation',
            representation='single normalized full-covariance Gaussian in log K',
            mean=distribution.mean.tolist(), covariance=distribution.covariance.tolist(),
            components=[dict(label='simple_logK', weight=1., log_weight=0.,
                x_est=distribution.mean.tolist(), P_est=distribution.covariance.tolist())],
            factor_ids=[], retained_factor_count=0,
            all_history_factors_reevaluated=False, bayes_factor_surprise_enabled=False,
            evidence_computed=False, normalization_constant=1.,
            normalization_definition='the retained Gaussian is normalized; normalization of the pre-projection nonlinear target is not evaluated',
            parameterization='x = log(K / (1 N m/rad))', K_units='N m/rad',
            approximation='recursive local posterior Gaussian projection; historical factors replaced by the current mean and full covariance')

    def _freeze_branch(self, state, record):
        super()._freeze_branch(state, record)
        record['auxiliary_branch_selection']['policy'] = (
            'current retained log-K Gaussian mean numerical anchor; fixed straight-logK continuation during this one factor')

    def log_density(self, x):
        distribution = self._simple_distribution
        x = np.asarray(x, dtype=float)
        if x.shape != distribution.mean.shape or not np.all(np.isfinite(x)):
            raise ValueError('A simple log-K density query must be finite and dimensionally consistent.')
        root = np.linalg.cholesky(distribution.covariance)
        error = x-distribution.mean
        inverse_error = cho_solve((root, True), error)
        logdet = 2.*float(np.sum(np.log(np.diag(root))))
        return (-.5*(x.size*math.log(2.*math.pi)+logdet+float(error @ inverse_error)),
                -inverse_error)

    def _discard_current_factors(self):
        self.factors = ()
        self.factor_cache.clear()
        self._physical_prediction_cache.clear()
        self._physical_prediction_query = None
        self._branch_anchor = None

    def update(self, result, *, batch_context=None, event_body_ratios=None):
        previous = self.last_record
        prior_record = self._density_record()
        # Reuse source/chart/acquisition/duplicate checks and the unchanged
        # joint-rotation factor constructor. No BFS/evidence runs in this call.
        record = super().update(result, batch_context=batch_context,
                                event_body_ratios=event_body_ratios)
        if record is previous:
            return record
        projections, failures = [], []
        updated = False
        try:
            for factor in tuple(_atomic_stiffness_factors(self.factors)):
                if factor.ratio.identity:
                    continue
                try:
                    posterior, diagnostic = project_gaussian_factor(self._simple_distribution,
                        lambda x, current=factor: self._factor_evaluation(current, x),
                        max_iterations=int(self.settings.get('simple_gaussian_max_iterations', 5)),
                        gradient_tolerance=float(self.settings.get('optimizer_gradient_tolerance', 1.e-6)),
                        curvature_step=float(self.settings.get('simple_gaussian_curvature_step', 1.e-4)),
                        maximum_line_search=int(self.settings.get('simple_gaussian_max_line_search', 8)),
                        prior_seed_scan=bool(self.settings.get('simple_gaussian_prior_seed_scan', True)))
                    self._simple_distribution = posterior
                    self.components = (PosteriorComponent('simple_logK', posterior),)
                    self.simple_projection_count += 1
                    updated = True
                    diagnostic.update(factor_id=factor.factor_id,
                        observation_ids=list(factor.observation_ids),
                        source_stamp=factor.states[0].stamp,
                        source_command=list(factor.states[0].command))
                    projections.append(diagnostic)
                except GaussianProjectionUnavailable as error:
                    failures.append(dict(error.diagnostics, factor_id=factor.factor_id,
                        observation_ids=list(factor.observation_ids),
                        source_stamp=factor.states[0].stamp,
                        source_command=list(factor.states[0].command)))
        finally:
            self._discard_current_factors()
        record.update(simple_reference=True, prior=prior_record,
            posterior=self._density_record(), update_applied=updated,
            bayes_factor_surprise_enabled=False, evidence_computed=False,
            cumulative_factors_retained=False, retained_factor_count=0,
            approximation='current log-K Gaussian times this acquisition joint-rotation ratio, projected to one local full-covariance Gaussian',
            simple_gaussian_projection=projections, simple_projection_failures=failures,
            simple_projection_count=self.simple_projection_count,
            stiffness_posterior_defined=True, command_proposal=None,
            body_reconstructed_from_stiffness=False, physical_truth_received=False,
            body_marginal=dict(record.get('body_marginal') or {}, edge_ids=list(self.body_edge_ids)))
        if updated:
            record['status'] = ('updated_single_gaussian_with_unavailable_factors' if failures
                                else 'updated_single_gaussian')
        elif failures or record.get('status') == 'failed_density_factor':
            record.update(status='single_gaussian_projection_unavailable', prior_retained=True,
                retained_prior=self._density_record(), new_observation_projection_applied=False)
        return record

    def command_proposal(self, result):
        record = self.last_record
        state = result.prior
        body = result.stages[-1].components
        if (record is None or record['source_stamp'] != state.stamp
                or tuple(record['source_command']) != tuple(state.command)
                or record['source_prior_state_id'] != state.state_id
                or record['body_posterior_state_ids'] != [item.state.state_id for item in body]):
            raise ValueError('A simple stiffness representative must use its current recorded body source.')
        if record.get('command_proposal') is not None:
            return record['command_proposal']
        x = self._simple_distribution.mean
        try:
            physical = self._physical_prediction(x, state)
            density, _ = self.log_density(x)
            stiffness = np.exp(x)
            proposal = dict(command=list(state.command), source_command=list(state.command),
                source_stamp=state.stamp, source_prior_state_id=state.state_id,
                body_posterior_state_ids=record['body_posterior_state_ids'],
                density_id=record['posterior']['density_id'], selected_logK=x.tolist(),
                selected_K=stiffness.tolist(), selected_candidate_index=0,
                selected_log_unnormalized_density=density,
                auxiliary_equilibrium=physical['theta'].tolist(),
                auxiliary_active_joint_limits=physical['active'].tolist(),
                auxiliary_equilibrium_numerical_audit=physical['audit'],
                representative_from_prior=self.simple_projection_count == 0,
                method='mode of the retained single log-K Gaussian; no extra command displacement is selected here',
                acceptance='the separate selector evaluates the same joint-law expected squared reaching error',
                global_mode_claimed=False, evidence_computed=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
            record.update(proposal_status='unavailable', proposal_failure_reason=str(error),
                          command_proposal=None)
            return None
        record.update(command_proposal=proposal, proposal_status='single_gaussian_mode',
            representative_from_prior=proposal['representative_from_prior'],
            feedback_to_body=False, feedback_to_body_or_control=True,
            feedback_to_control='incremental command action of the observed body; Gaussian representative only')
        return proposal
