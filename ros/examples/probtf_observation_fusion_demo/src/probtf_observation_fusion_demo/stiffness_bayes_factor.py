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
from contextlib import nullcontext
import hashlib
import math
import time

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import OptimizeResult, minimize


def _runtime_stage(owner, name, **context):
    monitor = getattr(owner, 'runtime_monitor', None)
    return nullcontext() if monitor is None else monitor.stage(name, **context)


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
    """Local-mode Laplace formula at a finite optimizer iterate.

    Rank deterministic starts and run BFGS with the complete score. A finite
    result need not meet the optimizer's stopping tolerance: that quality
    check is diagnostic, not a condition for updating the retained law. Use
    one central score-difference width for the former local-mode Laplace
    formula. Stationarity is not required. The formula omits the linear
    score term, so it is not the exact integral of a quadratic log-density
    expansion at a nonstationary iterate.
    Actual evaluation/factorization failures remain failures; no jitter,
    clipped curvature, or fabricated mass is supplied. No mode/covariance is
    substituted for the continuous density. This does not integrate all
    modes or guarantee the total evidence's accuracy.
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
        """Retry declared numerical starts only after required mass arithmetic fails.

        A finite point with an evaluable positive Hessian keeps the original
        path, regardless of optimizer flags or score residual.  A failed
        Cholesky is not a mass: try the remaining already declared starts
        against the same full density, without changing curvature or law.
        """
        started = time.perf_counter()
        seeds, cache = tuple(self.seeds()), {}
        failures = []
        try:
            with _runtime_stage(self, 'stiffness.laplace.attempt',
                    start_policy='rank declared starts', seed_count=len(seeds)):
                return self._integrate_seed_pool(density, seeds, cache)
        except StiffnessEvidenceUnavailable as initial_error:
            initial = initial_error.diagnostics
            if initial.get('failed_operation') != 'Cholesky factorization / Laplace log mass':
                raise
            failures.append(dict(source='initial ranked mode search', diagnostics=dict(initial)))
            selected = np.asarray(initial.get('selected_start', ()), dtype=float)
            seen = {selected.tobytes()}
            for index, seed in enumerate(seeds):
                seed = np.asarray(seed, dtype=float)
                if seed.tobytes() in seen:
                    continue
                seen.add(seed.tobytes())
                try:
                    with _runtime_stage(self, 'stiffness.laplace.attempt',
                            start_policy='declared restart after unavailable mass',
                            original_seed_index=index, seed_count=1):
                        result = self._integrate_seed_pool(density, (seed,), cache)
                except StiffnessEvidenceUnavailable as error:
                    failures.append(dict(source='declared deterministic restart',
                        original_seed_index=index, diagnostics=dict(error.diagnostics)))
                    continue
                diagnostics = result.diagnostics
                diagnostics.update(selected_start_index=index,
                    evaluations=diagnostics['evaluations']+sum(
                        attempt['diagnostics'].get('evaluations', 0) for attempt in failures),
                    seconds=time.perf_counter()-started,
                    mode_method=diagnostics['mode_method']+'; declared-start restart after an unavailable Laplace mass',
                    mass_operation_recovery=dict(applied=True,
                        trigger='actual Cholesky / Laplace mass failure, not optimizer status or stationarity audit',
                        failed_attempts=failures, accepted_original_seed_index=index,
                        same_density_and_score=True, same_hessian_width=True,
                        same_laplace_formula=True, curvature_regularization_added=False,
                        optimizer_status_is_acceptance_gate=False,
                        stationarity_is_acceptance_gate=False,
                        deterministic_starts_are_parameter_measure=False))
                return result
            initial['mass_operation_recovery'] = dict(applied=True,
                trigger='actual Cholesky / Laplace mass failure',
                failed_attempts=failures, all_declared_starts_exhausted=True,
                same_density_and_score=True, curvature_regularization_added=False)
            initial['seconds'] = time.perf_counter()-started
            raise StiffnessEvidenceUnavailable(
                str(initial_error)+'; no evaluable mass after every declared numerical start; '
                'no curvature regularization or fabricated mass was supplied.', initial) from initial_error

    def _integrate_seed_pool(self, density, seeds, cache):
        started = time.perf_counter()
        # The former two-width rule used its fine (half-configured) width
        # in the mass formula. Retain that numerical choice without running
        # its coarse-width comparison or adaptive quality audit.
        actual_hessian_step = .5*self.hessian_step
        diagnostics = dict(approximation='local-mode Laplace formula evaluated at a finite optimizer iterate; stationarity not required; no continuous-law replacement',
            linear_score_term_in_mass_formula=False,
            approximation_note='The former local-mode Laplace formula is retained; it is not the exact quadratic log-density integral at a nonstationary iterate.',
            global_mass_accuracy_guaranteed=False, all_modes_integrated=False,
            mode_method='deterministic-start ranking followed by BFGS with the complete density score',
            maximum_iterations=self.max_iterations, gradient_tolerance=self.gradient_tolerance,
            Hessian_step=actual_hessian_step, Hessian_steps=[actual_hessian_step],
            configured_hessian_step=self.hessian_step,
            configured_Hessian_steps=[self.hessian_step, actual_hessian_step],
            max_hessian_refinements=self.max_hessian_refinements,
            adaptive_Hessian_steps_enabled=False,
            Hessian_refinement_count=0, Hessian_attempts=[],
            Hessian_relative_tolerance=self.hessian_relative_tolerance,
            Hessian_absolute_tolerance=self.hessian_absolute_tolerance,
            deprecated_numerical_audit_settings=dict(
                hessian_relative_tolerance=self.hessian_relative_tolerance,
                hessian_absolute_tolerance=self.hessian_absolute_tolerance,
                max_hessian_refinements=self.max_hessian_refinements,
                configured_hessian_step_role='legacy coarse width; mass curvature uses half this width as before',
                effect='accepted for configuration compatibility; no second-width evaluation or acceptance gate'),
            numerical_quality_checks_are_update_gates=False,
            two_step_stability_checked=False, two_step_stability_verified=None,
            unavailable_starts=[], unavailable_trials=[],
            availability_recovery=dict(applied=False),
            evaluations=0, density_projection_applied=False)
        def query(x):
            x = np.asarray(x, dtype=float)
            if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
                raise ValueError('A Laplace density query needs finite log-stiffness coordinates.')
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
        with _runtime_stage(self, 'stiffness.laplace.seed_ranking', seed_count=len(seeds)):
            for index, seed in enumerate(seeds):
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
        with _runtime_stage(self, 'stiffness.laplace.mode_optimization',
                selected_start_index=index, dimension=seed.size,
                maximum_iterations=self.max_iterations):
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
        diagnostics.update(stationarity_verified=bool(score_norm <= self.gradient_tolerance),
            finite_nonstationary_point_accepted=bool(score_norm > self.gradient_tolerance),
            optimizer_failure_is_fatal=False,
            stationarity_polish_status='not performed; finite optimizer result is used without a residual acceptance gate')
        attempt = dict(steps=[actual_hessian_step], precision_checks_required=False)
        diagnostics['Hessian_attempts'].append(attempt)
        try:
            with _runtime_stage(self, 'stiffness.laplace.hessian',
                    dimension=mode.size, step=actual_hessian_step,
                    mode_score_inf=score_norm), np.errstate(over='raise', invalid='raise', divide='raise'):
                matrix = np.column_stack([-(query(mode+actual_hessian_step*axis)[1]
                    -query(mode-actual_hessian_step*axis)[1])/(2.*actual_hessian_step)
                    for axis in np.eye(mode.size)])
                # A scalar quadratic form uses the symmetric part. This is
                # an arithmetic average, not curvature clipping or jitter.
                symmetric = .5*matrix+.5*matrix.T
            if not np.all(np.isfinite(symmetric)):
                raise FloatingPointError('The calculated Laplace Hessian contains NaN or infinity.')
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            attempt.update(available=False, reason=str(error))
            diagnostics['failed_operation'] = 'score-difference Hessian evaluation'
            fail('Laplace evidence Hessian evaluation is unavailable: '+str(error))
        # Cheap quality diagnostics never reject a finite expansion. A second
        # width is deliberately not evaluated in the updating path.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            asymmetry = float(np.max(np.abs(matrix-matrix.T)))
            symmetry_verified = bool(np.allclose(matrix, matrix.T,
                rtol=self.hessian_relative_tolerance, atol=self.hessian_absolute_tolerance))
        attempt.update(available=True, Hessians=[matrix.tolist()],
            raw_asymmetry_max=[asymmetry if np.isfinite(asymmetry) else None],
            symmetry_verified=[symmetry_verified], reason=None)
        diagnostics.update(Hessian=symmetric.tolist(),
            Hessian_raw_asymmetry_max=attempt['raw_asymmetry_max'],
            Hessian_symmetry_verified=symmetry_verified,
            Hessian_step_difference_max=None,
            Hessian_symmetrization='arithmetic mean of the raw score Jacobian and its transpose; no eigenvalue modification')
        try:
            eigenvalues = np.linalg.eigvalsh(symmetric)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                condition = float(np.max(np.abs(eigenvalues))/np.min(np.abs(eigenvalues)))
            diagnostics.update(Hessian_eigenvalues=eigenvalues.tolist() if np.all(np.isfinite(eigenvalues)) else None,
                Hessian_condition_number=condition if np.isfinite(condition) else None,
                Hessian_condition_number_finite=bool(np.isfinite(condition)))
        except np.linalg.LinAlgError as error:
            diagnostics.update(Hessian_eigenvalues=None, Hessian_condition_number=None,
                Hessian_spectral_diagnostic_reason=str(error))
        try:
            with _runtime_stage(self, 'stiffness.laplace.mass_factorization',
                    dimension=mode.size, evaluations=diagnostics['evaluations']):
                cholesky = np.linalg.cholesky(symmetric)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    log_determinant = float(2.*np.log(np.diag(cholesky)).sum())
                    log_integral = float(value+.5*mode.size*math.log(2.*math.pi)-.5*log_determinant)
                if not np.isfinite(log_integral):
                    raise FloatingPointError('The calculated Laplace log mass is not finite.')
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            diagnostics.update(failed_operation='Cholesky factorization / Laplace log mass',
                positive_definiteness_verified=False)
            fail('Laplace evidence factorization or mass calculation failed: '+str(error))
        diagnostics.update(Hessian_log_determinant=log_determinant,
            seconds=time.perf_counter()-started, positive_definiteness_verified=True,
            support_regularization_added=False)
        return LogIntegral(log_integral, 'finite-iterate local-mode Laplace evidence',
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
    log_keep_probability: object = None
    log_reset_probability: object = None


def hazard_log_probabilities(rho, log_keep_probability=None, log_reset_probability=None):
    """Optional authoritative logs preserve finite batch survival mass."""
    if log_keep_probability is None and log_reset_probability is None:
        return (-np.inf if rho == 1. else math.log1p(-rho),
                -np.inf if rho == 0. else math.log(rho))
    keep = -np.inf if log_keep_probability is None else float(log_keep_probability)
    reset = -np.inf if log_reset_probability is None else float(log_reset_probability)
    if (np.isnan(keep) or np.isnan(reset) or keep > 0. or reset > 0.
            or not np.isfinite(logsumexp([keep, reset]))
            or abs(float(logsumexp([keep, reset]))) > 1e-12):
        raise ValueError('Keep/Reset log prior probabilities must be normalized finite-or-zero masses.')
    return keep, reset


def update_hazard_log_probabilities(update):
    keep = getattr(update, 'log_keep_probability', None)
    reset = getattr(update, 'log_reset_probability', None)
    if keep is None and reset is None:
        return hazard_log_probabilities(update.rho0)
    # Frozen updates were validated once at advance(), not at every density
    # query. Keeping the stored logs also avoids reconstructing 1-rho.
    return (-np.inf if keep is None else keep, -np.inf if reset is None else reset)


def _json_log_probability(value):
    return None if value == -np.inf else float(value)


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
        # Optional arithmetic batching. It returns the same ordered factor
        # values/scores; the retained recurrence and per-event hazards below
        # are unchanged. No candidate or observation is pruned by this hook.
        self.batch_factor_evaluator = None

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
        batch = getattr(self, 'batch_factor_evaluator', None)
        evaluated = None if batch is None else tuple(batch(x, self.updates))
        if evaluated is not None and len(evaluated) != len(self.updates):
            raise ValueError('A stiffness factor batch must retain every ordered update.')
        # An iterative recurrence avoids a Python recursion limit after many
        # asynchronous observations; all events and factors remain retained.
        for index, update in enumerate(self.updates):
            log_keep, log_reset = update_hazard_log_probabilities(update)
            if log_reset != -np.inf:
                reset = update.reset
                key = (reset.source_id, id(reset.evaluate))
                if key not in reset_cache:
                    reset_cache[key] = self._evaluate(reset.evaluate, x)
                reset_value, reset_score = reset_cache[key]
                if log_keep == -np.inf:
                    value, score = reset_value-reset.log_mass+update.keep_log_mass, reset_score
                else:
                    terms = np.array([log_keep+value,
                        log_reset+reset_value-reset.log_mass+update.keep_log_mass])
                    value = float(logsumexp(terms))
                    responsibility = np.exp(terms-value)
                    score = responsibility[0]*score+responsibility[1]*reset_score
            if evaluated is None:
                contribution, derivative = self._evaluate(update.factor, x)
            else:
                contribution, derivative = evaluated[index]
                derivative = np.asarray(derivative, dtype=float)
                if (derivative.shape != x.shape or not np.isfinite(contribution)
                        or not np.all(np.isfinite(derivative))):
                    raise ValueError('A batched stiffness factor must have finite value and score.')
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
                metadata=None, log_keep_probability=None, log_reset_probability=None):
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
        explicit_logs = log_keep_probability is not None or log_reset_probability is not None
        log_keep, log_reset = hazard_log_probabilities(rho,
            log_keep_probability, log_reset_probability)
        has_reset, has_keep = log_reset != -np.inf, log_keep != -np.inf
        candidate_reset = self.reset if reset is None else reset
        if has_reset and not isinstance(candidate_reset, ResetMeasure):
            raise ValueError('Positive-rho stiffness BFS requires an explicitly declared reset measure.')
        event_id = str(factor_id if event_id is None else event_id)
        signature = (str(factor_id), ids, rho,
                     None if not has_reset else candidate_reset.source_id, bool(identity_factor))
        if explicit_logs:
            signature += (log_keep, log_reset)
        if event_id in self._events:
            previous_signature, previous_record = self._events[event_id]
            if previous_signature != signature:
                raise ValueError('An already used stiffness event identity changed its contents.')
            return previous_record
        repeated = self.observation_ids.intersection(ids)
        if repeated:
            raise ValueError('A stiffness observation cannot be counted again: '+repr(sorted(repeated)))
        rule = self.integrator if integrator is None else integrator
        if isinstance(rule, LocalLaplaceEvidence):
            rule.runtime_monitor = getattr(self, 'runtime_monitor', None)
        timing_context = dict(event_id=event_id, factor_id=str(factor_id),
            observation_count=len(ids), retained_event_count=len(self.updates))
        keep_mass = self.log_mass
        keep_integral = reset_integral = recovered_scale = None
        shared_evidence = False
        log_bf = posterior_reset = mixed_evidence = None
        posterior_keep_log = posterior_reset_log = None
        if not has_reset:
            # No mass/evidence approximation can alter the rho=0 product.
            new_mass = keep_mass if identity_factor else None
            posterior_keep_log, posterior_reset_log = 0., -np.inf
        else:
            if keep_mass is None:
                with _runtime_stage(self, 'stiffness.evidence.keep_scale', **timing_context):
                    recovered_scale = self._integral(rule, self.evaluate)
                keep_mass = recovered_scale.log_value
            if identity_factor:
                keep_integral = LogIntegral(0., 'declared exact identity factor', exact=True)
                reset_integral = keep_integral
            else:
                if has_keep:
                    def keep_query(x):
                        x = np.asarray(x, dtype=float)
                        value, score = self.evaluate(x)
                        contribution, derivative = self._evaluate(factor, x)
                        return value-keep_mass+contribution, score+derivative
                    with _runtime_stage(self, 'stiffness.evidence.keep', **timing_context):
                        keep_integral = self._integral(rule, keep_query)
                shared_evidence = (has_keep and not self.updates
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
                    with _runtime_stage(self, 'stiffness.evidence.reset', **timing_context):
                        reset_integral = self._integral(rule, reset_query)
            if not has_keep:
                mixed_evidence = reset_integral.log_value
                posterior_reset = 1.
                posterior_keep_log, posterior_reset_log = -np.inf, 0.
            else:
                terms = [log_keep+keep_integral.log_value,
                         log_reset+reset_integral.log_value]
                mixed_evidence = float(logsumexp(terms))
                posterior_reset = float(math.exp(terms[1]-mixed_evidence))
                posterior_keep_log, posterior_reset_log = terms[0]-mixed_evidence, terms[1]-mixed_evidence
                log_bf = float(reset_integral.log_value-keep_integral.log_value)
            if identity_factor:
                mixed_evidence, posterior_reset = 0., rho
                posterior_keep_log, posterior_reset_log = log_keep, log_reset
            new_mass = keep_mass+mixed_evidence
        log_odds = None if not has_reset or not has_keep else log_reset-log_keep
        ordinary_odds = (None if not has_keep or (log_odds is not None and
                         log_odds > math.log(np.finfo(float).max))
                         else 0. if not has_reset else
                         math.exp(log_odds) if explicit_logs else rho/(1.-rho))
        record = dict(status='updated_continuous_keep_reset_density' if has_reset else
                      'updated_continuous_factor_product', update_applied=True,
            event_id=event_id, factor_id=str(factor_id), observation_ids=list(ids),
            hazard_applied=has_reset, rho0=rho,
            prior_reset_odds=ordinary_odds, prior_log_reset_odds=log_odds,
            log_keep_prior_probability=_json_log_probability(log_keep),
            log_reset_prior_probability=_json_log_probability(log_reset),
            authoritative_log_probabilities=explicit_logs,
            zero_mass_log_representation='null means negative infinity',
            posterior_keep_log_probability=_json_log_probability(posterior_keep_log),
            posterior_reset_log_probability=_json_log_probability(posterior_reset_log),
            keep_log_mass=keep_mass,
            reset_source_id=(None if not has_reset else candidate_reset.source_id),
            reset_description=(None if not has_reset else candidate_reset.description),
            reset_log_mass=(None if not has_reset else candidate_reset.log_mass),
            hypothesis_scale=(None if not has_reset else dict(
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
                         keep_mass, candidate_reset if has_reset else None, record,
                         log_keep if explicit_logs else None, log_reset if explicit_logs else None)
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
            if getattr(update, 'log_keep_probability', None) is not None:
                identity.update(repr((update.log_keep_probability, update.log_reset_probability)).encode())
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
