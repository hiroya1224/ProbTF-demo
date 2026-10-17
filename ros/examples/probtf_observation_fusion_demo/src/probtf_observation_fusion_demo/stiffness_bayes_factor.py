"""Continuous log-K Keep/Reset recursion, with an explicit evidence rule.

The density remains a function containing every supplied factor. Numerical
integration is a separate operation: a Gaussian used by an evidence rule is
never installed as the stiffness law. No integration method is selected by
this module. In particular the finite starts used to select K_est are not
quadrature points.

For positive rho, reset is scaled to the (possibly numerically estimated)
retained keep mass. The result is the unnormalized measure
L_j [(1-rho) f_keep + rho M_keep f_reset / M_reset]. Its mass accumulates
the normalized mixed evidence on top of M_keep. At rho=0 the expression is
literally L_j f_keep, preserving the ordinary product and its scale without
requiring an integral. An evidence rule states its approximation in LogIntegral.
"""

from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import OptimizeResult, minimize


@dataclass(frozen=True)
class LogIntegral:
    """An explicitly supplied integral of a density, not of its score."""

    log_value: float
    method: str
    exact: bool = False
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.log_value):
            raise ValueError('A stiffness evidence integral must be positive and finite.')
        if not self.method:
            raise ValueError('An evidence integral must identify its calculation method.')

    def record(self):
        return dict(log_value=float(self.log_value), method=self.method,
                    exact=bool(self.exact), diagnostics=dict(self.diagnostics))


@dataclass(frozen=True)
class ResetMeasure:
    """A continuous reset measure and its explicitly declared mass/identity."""

    source_id: str
    evaluate: object
    log_mass: float = 0.
    description: str = 'explicit reset measure'

    def __post_init__(self):
        if not self.source_id or not callable(self.evaluate):
            raise ValueError('A reset measure needs an identity and a density/score function.')
        if not np.isfinite(self.log_mass):
            raise ValueError('The reset measure must have a declared positive finite mass.')


class StiffnessEvidenceUnavailable(ValueError):
    """A numerical evidence calculation failed; no posterior is fabricated."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class LocalLaplaceEvidence:
    """Explicit local mass approximation, separate from the retained K law.

    Rank deterministic numerical starts, find one stationary local mode with
    the complete score, and validate its unmodified Hessian at two difference
    steps. No mode/covariance is substituted for the continuous density. This
    does not integrate all modes or guarantee the total evidence's accuracy.
    Seed/evaluation failures are recorded, never converted to zero density.
    """

    def _continue_available_bfgs(self, query, accepted, diagnostics):
        """Continue the same mode search after an unavailable line trial.

        SciPy does not expose its line search when a density query raises.
        Reconstruct its inverse-BFGS approximation from completed iterates,
        then backtrack explicitly. An unavailable query has no assigned
        function value or gradient and cannot become an accepted iterate.
        """
        point = np.array(accepted[0], copy=True)
        value, score = query(point)
        inverse = np.eye(point.size)
        history = diagnostics['availability_recovery']
        history.update(accepted_iterations_before_failure=len(accepted)-1,
            line_search='Armijo ascent of the unchanged log density; unavailable trials are rejected without a value',
            maximum_backtracks=64, iterations=[])

        def update_inverse(matrix, displacement, old_score, new_score):
            # BFGS is applied to minus the log density. Retain the previous
            # approximation if this finite secant has no positive curvature;
            # no Hessian regularization or modification of the law is made.
            difference = old_score-new_score
            curvature = float(displacement @ difference)
            if not np.isfinite(curvature) or curvature <= 0.:
                return matrix, False
            factor = np.eye(point.size)-np.outer(displacement, difference)/curvature
            result = factor @ matrix @ factor.T+np.outer(displacement, displacement)/curvature
            if not np.all(np.isfinite(result)):
                return matrix, False
            return result, True

        for next_point in accepted[1:]:
            next_value, next_score = query(next_point)
            inverse, _ = update_inverse(inverse, next_point-point, score, next_score)
            point, value, score = np.array(next_point, copy=True), next_value, next_score
        completed = len(accepted)-1
        failed_trial = diagnostics['unavailable_trials'][-1]
        resume_direction = np.asarray(failed_trial['logK'])-point
        resume_failed_line = True
        message = 'Availability-aware BFGS reached the configured iteration limit.'
        success = False
        for iteration in range(completed, self.max_iterations):
            if np.max(np.abs(score)) <= self.gradient_tolerance:
                success, message = True, 'Full-score tolerance reached after availability-aware BFGS.'
                break
            # First retry exactly the failed SciPy line at a shorter step.
            # Subsequent accepted iterations use the reconstructed BFGS
            # direction, with the same explicit backtracking treatment.
            direction = resume_direction if resume_failed_line else inverse @ score
            slope = float(score @ direction)
            if not np.isfinite(slope) or slope <= 0. or not np.all(np.isfinite(direction)):
                message = 'Availability-aware BFGS has no finite ascent direction.'
                break
            step, found = 1., False
            row = dict(iteration=iteration, source_logK=point.tolist(),
                source_log_density=float(value), source_score_inf=float(np.max(np.abs(score))),
                direction=direction.tolist(), trials=[])
            if resume_failed_line:
                row['resumes_unavailable_scipy_line'] = True
                row['trials'].append(dict(step=1., logK=failed_trial['logK'],
                    available=False, reason=failed_trial['reason'], accepted=False,
                    already_evaluated_by_scipy=True))
                step = .5
                resume_failed_line = False
            for backtrack in range(64):
                trial = point+step*direction
                if np.array_equal(trial, point):
                    row['resolution_exhausted'] = True
                    break
                trial_record = dict(step=float(step), logK=trial.tolist())
                try:
                    trial_value, trial_score = query(trial)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    trial_record.update(available=False, reason=str(error), accepted=False)
                    diagnostics['unavailable_trials'].append(dict(
                        phase='availability_backtrack', iteration=iteration,
                        step=float(step), logK=trial.tolist(), reason=str(error)))
                else:
                    sufficient = bool(trial_value >= value+1e-4*step*slope)
                    trial_record.update(available=True, log_density=float(trial_value),
                        Armijo_log_density=float(value+1e-4*step*slope), accepted=sufficient)
                    if sufficient:
                        inverse, updated = update_inverse(inverse, trial-point, score, trial_score)
                        row['inverse_BFGS_updated'] = updated
                        point, value, score = trial, trial_value, trial_score
                        found = True
                row['trials'].append(trial_record)
                if found:
                    break
                step *= .5
            row['accepted'] = found
            history['iterations'].append(row)
            if not found:
                message = 'Availability-aware BFGS could not find an evaluable Armijo step.'
                break
            completed += 1
        if np.max(np.abs(score)) <= self.gradient_tolerance:
            success, message = True, 'Full-score tolerance reached after availability-aware BFGS.'
        return OptimizeResult(x=point, success=success, message=message, nit=completed)

    def __init__(self, seeds, *, max_iterations=80, gradient_tolerance=1e-6,
                 hessian_step=1e-4, hessian_relative_tolerance=1e-3,
                 hessian_absolute_tolerance=1e-5, max_hessian_refinements=8):
        self.seeds = seeds
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.hessian_step = float(hessian_step)
        self.hessian_relative_tolerance = float(hessian_relative_tolerance)
        self.hessian_absolute_tolerance = float(hessian_absolute_tolerance)
        try:
            self.max_hessian_refinements = int(max_hessian_refinements)
        except (ValueError, TypeError, OverflowError):
            raise ValueError('Laplace evidence Hessian refinements must be a nonnegative integer.')
        if (isinstance(max_hessian_refinements, (bool, np.bool_))
                or self.max_hessian_refinements != max_hessian_refinements
                or self.max_hessian_refinements < 0):
            raise ValueError('Laplace evidence Hessian refinements must be a nonnegative integer.')
        if (not callable(seeds) or self.max_iterations <= 0
                or not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.
                or not np.isfinite(self.hessian_step) or self.hessian_step <= 0.
                or not np.isfinite(self.hessian_relative_tolerance) or self.hessian_relative_tolerance < 0.
                or not np.isfinite(self.hessian_absolute_tolerance) or self.hessian_absolute_tolerance < 0.):
            raise ValueError('Laplace evidence requires valid mode/Hessian numerical settings.')

    def __call__(self, density):
        started = time.perf_counter()
        diagnostics = dict(approximation='single stationary local-mode Laplace mass only; no continuous-law replacement',
            global_mass_accuracy_guaranteed=False, all_modes_integrated=False,
            mode_method='deterministic-start ranking followed by BFGS with the complete density score',
            maximum_iterations=self.max_iterations, gradient_tolerance=self.gradient_tolerance,
            Hessian_steps=[self.hessian_step, .5*self.hessian_step],
            configured_Hessian_steps=[self.hessian_step, .5*self.hessian_step],
            max_hessian_refinements=self.max_hessian_refinements,
            adaptive_Hessian_steps_enabled=self.max_hessian_refinements > 0,
            Hessian_refinement_count=0, Hessian_attempts=[],
            Hessian_relative_tolerance=self.hessian_relative_tolerance,
            Hessian_absolute_tolerance=self.hessian_absolute_tolerance,
            unavailable_starts=[], unavailable_trials=[],
            availability_recovery=dict(applied=False),
            evaluations=0, density_projection_applied=False)
        cache = {}
        def query(x):
            x = np.asarray(x, dtype=float)
            key = x.tobytes()
            if key not in cache:
                diagnostics['evaluations'] += 1
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    value, score = ContinuousStiffnessBFS._evaluate(density, x)
                cache[key] = (value, np.array(score, copy=True))
            return cache[key]
        def fail(message):
            diagnostics['seconds'] = time.perf_counter()-started
            raise StiffnessEvidenceUnavailable(message, diagnostics)
        ranked = []
        seen = set()
        for index, seed in enumerate(self.seeds()):
            seed = np.asarray(seed, dtype=float)
            if seed.ndim != 1 or not seed.size or not np.all(np.isfinite(seed)):
                fail('Laplace evidence received an invalid deterministic mode start.')
            if seed.tobytes() in seen:
                continue
            seen.add(seed.tobytes())
            try:
                value, _ = query(seed)
                ranked.append((value, index, seed.copy()))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                diagnostics['unavailable_starts'].append(dict(index=index, reason=str(error)))
        diagnostics['ranked_start_count'] = len(ranked)
        if not ranked:
            fail('Laplace evidence has no evaluable deterministic mode start.')
        _, index, seed = max(ranked, key=lambda item:item[0])
        diagnostics.update(selected_start_index=index, selected_start=seed.tolist())
        accepted = [seed.copy()]
        def objective(x):
            try:
                value, score = query(x)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                diagnostics['unavailable_trials'].append(dict(
                    phase='scipy_BFGS', logK=np.asarray(x).tolist(), reason=str(error)))
                raise
            return -value, -score
        def completed_iteration(x):
            accepted.append(np.array(x, copy=True))
        try:
            optimum = minimize(objective, seed, jac=True, method='BFGS',
                callback=completed_iteration,
                options=dict(maxiter=self.max_iterations, gtol=self.gradient_tolerance))
            value, score = query(optimum.x)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            if not diagnostics['unavailable_trials']:
                fail('Laplace evidence mode evaluation is unavailable: '+str(error))
            diagnostics['availability_recovery'].update(applied=True, trigger=str(error),
                source_logK=accepted[-1].tolist())
            diagnostics['mode_method'] += '; availability-aware BFGS continuation after an unavailable trial'
            try:
                optimum = self._continue_available_bfgs(query, accepted, diagnostics)
                value, score = query(optimum.x)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as recovery_error:
                fail('Laplace evidence mode evaluation is unavailable: '+str(recovery_error))
        score_norm = float(np.max(np.abs(score)))
        diagnostics.update(optimizer_success=bool(optimum.success), optimizer_message=str(optimum.message),
            optimizer_iterations=int(optimum.nit), mode=optimum.x.tolist(),
            mode_log_density=value, mode_score=score.tolist(), mode_score_inf=score_norm)
        mode = np.array(optimum.x, copy=True)
        diagnostics['stationarity_polish'] = []
        if (not optimum.success and self.gradient_tolerance < score_norm
                <= 100.*self.gradient_tolerance):
            # Near a stationary point BFGS can stop on rounded objective
            # changes while the explicitly evaluated score still resolves a
            # Newton correction. Solve that same score, without loosening its
            # tolerance or modifying a Hessian to force positive curvature.
            for _ in range(3):
                try:
                    matrix = np.column_stack([-(query(mode+self.hessian_step*axis)[1]
                        -query(mode-self.hessian_step*axis)[1])/(2.*self.hessian_step)
                        for axis in np.eye(mode.size)])
                    symmetric = .5*(matrix+matrix.T)
                    np.linalg.cholesky(symmetric)
                    direction = np.linalg.solve(symmetric, score)
                    trial_value, trial_score = query(mode+direction)
                    trial_norm = float(np.max(np.abs(trial_score)))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    break
                # This is a root solve for the score, followed below by
                # stationarity, SPD and two-step Hessian checks. Small
                # differences of a log density built from large cancelling
                # body-density terms cannot provide an objective monotonicity
                # test at this precision. Its final magnitude does not bound
                # that cancellation error. Do not loosen the score tolerance
                # or add a fictitious density-based acceptance guarantee.
                accepted = trial_norm < score_norm
                diagnostics['stationarity_polish'].append(dict(
                    score_inf_before=score_norm, score_inf_after=trial_norm,
                    log_density_before=value, log_density_after=trial_value,
                    objective_monotonicity_required=False,
                    acceptance='score residual decrease; unmodified SPD and two-step Hessian verified at the final mode',
                    accepted=bool(accepted)))
                if not accepted:
                    break
                mode += direction
                value, score, score_norm = trial_value, trial_score, trial_norm
                if score_norm <= self.gradient_tolerance:
                    break
            diagnostics.update(mode=mode.tolist(), mode_log_density=value,
                mode_score=score.tolist(), mode_score_inf=score_norm)
        if score_norm > self.gradient_tolerance:
            fail('Laplace evidence mode did not satisfy the configured full-score tolerance.')
        hessian_cache = {}
        for refinement in range(self.max_hessian_refinements+1):
            coarse = self.hessian_step*(.5**refinement)
            steps = [coarse, .5*coarse]
            attempt = dict(refinement=refinement, steps=steps,
                reused_previous_fine=coarse in hessian_cache)
            diagnostics['Hessian_attempts'].append(attempt)
            diagnostics.update(Hessian_steps=steps, Hessian_refinement_count=refinement)
            hessians = []
            try:
                for step in steps:
                    if step not in hessian_cache:
                        hessian_cache[step] = np.column_stack([-(query(mode+step*axis)[1]
                            -query(mode-step*axis)[1])/(2.*step)
                            for axis in np.eye(mode.size)])
                    hessians.append(hessian_cache[step])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                attempt.update(available=False, reason=str(error), precision_checks_passed=False)
                fail('Laplace evidence Hessian evaluation is unavailable: '+str(error))
            asymmetry = [float(np.max(np.abs(item-item.T))) for item in hessians]
            difference = float(np.max(np.abs(hessians[0]-hessians[1])))
            stable = bool(np.allclose(hessians[0], hessians[1],
                rtol=self.hessian_relative_tolerance, atol=self.hessian_absolute_tolerance))
            symmetric = [bool(np.allclose(matrix, matrix.T,
                rtol=self.hessian_relative_tolerance, atol=self.hessian_absolute_tolerance))
                for matrix in hessians]
            reason = ('Laplace evidence Hessian is unstable across the two score-difference steps.'
                if not stable else 'Laplace evidence score Hessian is not numerically symmetric.'
                if not all(symmetric) else None)
            attempt.update(available=True, Hessians=[matrix.tolist() for matrix in hessians],
                raw_asymmetry_max=asymmetry, step_difference_max=difference,
                two_step_stability_verified=stable, symmetry_verified=symmetric,
                precision_checks_passed=reason is None, reason=reason)
            diagnostics.update(Hessian_raw_asymmetry_max=asymmetry,
                Hessian_step_difference_max=difference)
            if reason is None:
                break
            if refinement == self.max_hessian_refinements:
                fail(reason)
            # Re-evaluate the same full score with both widths halved. The
            # preceding fine matrix becomes the next coarse matrix. Neither
            # tolerances nor the density, mode, or curvature are modified.
        # Symmetrization removes roundoff only after the symmetry check. No
        # eigenvalue clipping, variance cap or positive diagonal is added.
        symmetric = .5*(hessians[1]+hessians[1].T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        diagnostics['Hessian_eigenvalues'] = eigenvalues.tolist()
        try:
            cholesky = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError:
            fail('Laplace evidence requires a positive definite unmodified mode Hessian.')
        condition = float(eigenvalues[-1]/eigenvalues[0])
        diagnostics['Hessian_condition_number'] = condition
        if not np.isfinite(condition) or condition*np.finfo(float).eps >= 1.:
            fail('Laplace evidence mode Hessian is numerically unresolved at float precision.')
        log_determinant = float(2.*np.log(np.diag(cholesky)).sum())
        log_integral = float(value+.5*mode.size*math.log(2.*math.pi)-.5*log_determinant)
        diagnostics.update(Hessian=symmetric.tolist(), Hessian_log_determinant=log_determinant,
            seconds=time.perf_counter()-started, stationarity_verified=True,
            positive_definiteness_verified=True, two_step_stability_verified=True,
            support_regularization_added=False)
        return LogIntegral(log_integral, 'local stationary-mode Laplace evidence',
                           exact=False, diagnostics=diagnostics)


@dataclass(frozen=True)
class _Update:
    event_id: str
    factor_id: str
    observation_ids: tuple
    factor: object
    rho0: float
    keep_log_mass: object
    reset: object
    record: dict


class ContinuousStiffnessBFS:
    """Retain an exact expression; approximate only explicitly supplied masses.

    ``integrator`` is callable as ``integrator(density_and_score)`` and must
    return LogIntegral. It receives the actual continuous function. The
    caller owns acquisition/event persistence: a failed integral raises
    before this object mutates, and the caller must keep that pending event.
    Missing observations and exact repeated events do not apply a hazard.
    Different sensor events may have the same acquisition time; observation
    identities, rather than a strictly increasing timestamp, prevent reuse.
    """

    def __init__(self, initial, *, initial_source_id, initial_log_mass=0.,
                 rho0=.10, reset=None, integrator=None):
        if not callable(initial) or not initial_source_id:
            raise ValueError('A continuous initial density and its source identity are required.')
        self._validate_rho(rho0)
        if not np.isfinite(initial_log_mass):
            raise ValueError('The initial measure must have positive finite declared mass.')
        self.initial = initial
        self.initial_source_id = str(initial_source_id)
        self.initial_log_mass = float(initial_log_mass)
        self.log_mass = float(initial_log_mass)
        self.rho0 = float(rho0)
        self.reset = reset
        self.integrator = integrator
        self.updates = ()
        self.observation_ids = frozenset()
        self._events = {}

    @staticmethod
    def _validate_rho(rho):
        if not np.isfinite(rho) or not 0. <= rho <= 1.:
            raise ValueError('The stiffness reset prior probability must lie in [0, 1].')

    @staticmethod
    def _evaluate(function, x):
        value, score = function(x)
        score = np.asarray(score, dtype=float)
        if score.shape != x.shape or not np.all(np.isfinite(score)) or not np.isfinite(value):
            raise ValueError('A continuous stiffness density/factor query must have finite value and score.')
        return float(value), score

    def evaluate(self, x):
        """Unnormalized log density and full recursive score, without pruning."""
        x = np.asarray(x, dtype=float)
        value, score = self._evaluate(self.initial, x)
        reset_cache = {}
        # An iterative recurrence avoids a Python recursion limit after many
        # asynchronous observations; all events and factors remain retained.
        for update in self.updates:
            if update.rho0:
                reset = update.reset
                key = (reset.source_id, id(reset.evaluate))
                if key not in reset_cache:
                    reset_cache[key] = self._evaluate(reset.evaluate, x)
                reset_value, reset_score = reset_cache[key]
                if update.rho0 == 1.:
                    value, score = reset_value-reset.log_mass+update.keep_log_mass, reset_score
                else:
                    terms = np.array([math.log1p(-update.rho0)+value,
                        math.log(update.rho0)+reset_value-reset.log_mass+update.keep_log_mass])
                    value = float(logsumexp(terms))
                    responsibility = np.exp(terms-value)
                    score = responsibility[0]*score+responsibility[1]*reset_score
            contribution, derivative = self._evaluate(update.factor, x)
            value += contribution
            score = score+derivative
        return float(value), score

    log_density = evaluate

    @staticmethod
    def _integral(integrator, density):
        if integrator is None:
            raise ValueError('Positive-rho stiffness BFS requires an explicitly selected evidence integrator.')
        result = integrator(density)
        if not isinstance(result, LogIntegral):
            raise ValueError('The evidence integrator must return LogIntegral with method/approximation metadata.')
        return result

    def advance(self, factor_id, observation_ids, factor, *, event_id=None,
                rho0=None, reset=None, integrator=None, identity_factor=False,
                metadata=None):
        """Apply one fresh valid observation event, retaining both hypotheses.

        ``identity_factor`` is a declared exact L=1 optimization. It still
        applies the hazard when the event contains a valid observation. An
        empty observation list performs neither a factor nor a transition.
        """
        ids = tuple(str(item) for item in observation_ids)
        if not ids:
            return dict(status='no_observation_identity', update_applied=False,
                        hazard_applied=False, density_id=self.density_id)
        if len(ids) != len(set(ids)):
            raise ValueError('An event cannot contain duplicate stiffness observation IDs.')
        if not callable(factor) or not factor_id:
            raise ValueError('A retained stiffness factor needs a function and source identity.')
        rho = self.rho0 if rho0 is None else float(rho0)
        self._validate_rho(rho)
        candidate_reset = self.reset if reset is None else reset
        if rho and not isinstance(candidate_reset, ResetMeasure):
            raise ValueError('Positive-rho stiffness BFS requires an explicitly declared reset measure.')
        event_id = str(factor_id if event_id is None else event_id)
        signature = (str(factor_id), ids, rho,
                     None if not rho else candidate_reset.source_id, bool(identity_factor))
        if event_id in self._events:
            previous_signature, previous_record = self._events[event_id]
            if previous_signature != signature:
                raise ValueError('An already used stiffness event identity changed its contents.')
            return previous_record
        repeated = self.observation_ids.intersection(ids)
        if repeated:
            raise ValueError('A stiffness observation cannot be counted again: '+repr(sorted(repeated)))
        rule = self.integrator if integrator is None else integrator
        keep_mass = self.log_mass
        keep_integral = reset_integral = recovered_scale = None
        shared_evidence = False
        log_bf = posterior_reset = mixed_evidence = None
        if not rho:
            # No mass/evidence approximation can alter the rho=0 product.
            new_mass = keep_mass if identity_factor else None
        else:
            if keep_mass is None:
                recovered_scale = self._integral(rule, self.evaluate)
                keep_mass = recovered_scale.log_value
            if identity_factor:
                keep_integral = LogIntegral(0., 'declared exact identity factor', exact=True)
                reset_integral = keep_integral
            else:
                if rho < 1.:
                    def keep_query(x):
                        x = np.asarray(x, dtype=float)
                        value, score = self.evaluate(x)
                        contribution, derivative = self._evaluate(factor, x)
                        return value-keep_mass+contribution, score+derivative
                    keep_integral = self._integral(rule, keep_query)
                shared_evidence = (rho < 1. and not self.updates
                    and candidate_reset.evaluate is self.initial
                    and candidate_reset.log_mass == keep_mass)
                if shared_evidence:
                    # Identical integrands need one actual integral. Both
                    # hypotheses and their distinct prior masses remain.
                    reset_integral = keep_integral
                else:
                    def reset_query(x):
                        x = np.asarray(x, dtype=float)
                        value, score = self._evaluate(candidate_reset.evaluate, x)
                        contribution, derivative = self._evaluate(factor, x)
                        return value-candidate_reset.log_mass+contribution, score+derivative
                    reset_integral = self._integral(rule, reset_query)
            if rho == 1.:
                mixed_evidence = reset_integral.log_value
                posterior_reset = 1.
            else:
                terms = [math.log1p(-rho)+keep_integral.log_value,
                         math.log(rho)+reset_integral.log_value]
                mixed_evidence = float(logsumexp(terms))
                posterior_reset = float(math.exp(terms[1]-mixed_evidence))
                log_bf = float(reset_integral.log_value-keep_integral.log_value)
            if identity_factor:
                mixed_evidence, posterior_reset = 0., rho
            new_mass = keep_mass+mixed_evidence
        record = dict(status='updated_continuous_keep_reset_density' if rho else
                      'updated_continuous_factor_product', update_applied=True,
            event_id=event_id, factor_id=str(factor_id), observation_ids=list(ids),
            hazard_applied=bool(rho), rho0=rho,
            prior_reset_odds=(None if rho == 1. else rho/(1.-rho)),
            keep_log_mass=keep_mass,
            reset_source_id=(None if not rho else candidate_reset.source_id),
            reset_description=(None if not rho else candidate_reset.description),
            reset_log_mass=(None if not rho else candidate_reset.log_mass),
            hypothesis_scale=(None if not rho else dict(
                convention='retain keep candidate mass and scale reset to that same mass',
                keep_log_total_mass=keep_mass, reset_log_total_mass=keep_mass,
                reset_input_log_total_mass=candidate_reset.log_mass,
                reset_log_scale=keep_mass-candidate_reset.log_mass)),
            keep_evidence=None if keep_integral is None else keep_integral.record(),
            reset_evidence=None if reset_integral is None else reset_integral.record(),
            recovered_keep_scale=None if recovered_scale is None else recovered_scale.record(),
            log_bayes_factor_reset_over_keep=log_bf,
            posterior_reset_mass=posterior_reset,
            log_mixed_evidence=mixed_evidence,
            posterior_log_mass=new_mass,
            identity_factor=bool(identity_factor), metadata=dict(metadata or {}),
            identical_keep_reset_evidence_reused=shared_evidence,
            density_projection_applied=False, factor_pruning_applied=False,
            integration_semantics='only branch mass/evidence; retained law is the full continuous expression')
        update = _Update(event_id, str(factor_id), ids, factor, rho,
                         keep_mass, candidate_reset if rho else None, record)
        self.updates += (update,)
        self.log_mass = new_mass
        self.observation_ids = self.observation_ids.union(ids)
        record['density_id'] = self.density_id
        record['retained_event_count'] = len(self.updates)
        self._events[event_id] = (signature, record)
        return record

    @property
    def density_id(self):
        identity = hashlib.sha256(self.initial_source_id.encode())
        for update in self.updates:
            identity.update(repr((update.event_id, update.factor_id, update.observation_ids,
                update.rho0, update.keep_log_mass,
                None if update.reset is None else update.reset.source_id,
                None if update.reset is None else update.reset.log_mass)).encode())
        return identity.hexdigest()

    def record(self):
        return dict(density_id=self.density_id,
            representation='continuous factor/Keep-Reset logsumexp recurrence; no stiffness moment projection',
            initial_source_id=self.initial_source_id, initial_log_mass=self.initial_log_mass,
            normalization_log_mass=self.log_mass,
            retained_event_count=len(self.updates),
            factor_ids=[item.factor_id for item in self.updates],
            updates=[item.record for item in self.updates],
            global_moments=None, density_projection_applied=False,
            factor_pruning_applied=False, monte_carlo_used=False)
