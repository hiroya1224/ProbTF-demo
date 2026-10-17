"""Deterministic execution-only damping of a positive stiffness estimate.

The estimated density is never modified. The coordinator advances this
signal at every motion tick, including while inference or IK is busy. Each
IK calculation captures one value and freezes it for that calculation.
Previous filter records support deterministic restoration and replay. The
per-update cap is in log K, matching deflecomp, rather than a limit per second.
"""

import numpy as np


def _vector(value, name, *, positive=False):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name+' must be a finite nonempty vector.') from error
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(name+' must be a finite nonempty vector.')
    if positive and np.any(result <= 0.):
        raise ValueError(name+' must contain strictly positive stiffnesses.')
    return result.copy()


def _stamp(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('A stiffness filter stamp must be finite.') from error
    if not np.isfinite(result):
        raise ValueError('A stiffness filter stamp must be finite.')
    return result


def _exp(log_value):
    try:
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            value = np.exp(log_value)
    except FloatingPointError as error:
        raise ValueError('Execution log K must represent finite positive stiffnesses.') from error
    if not np.all(np.isfinite(value)) or np.any(value <= 0.):
        raise ValueError('Execution log K must represent finite positive stiffnesses.')
    return value


class LogStiffnessDamper:
    """First-order log-K lag and an independent per-update log-step cap.

    Settings use deflecomp's ``kp_exec_tau`` and ``max_log_kp_exec_step``.
    Zero tau removes the lag; zero maximum step removes the cap. A zero dt
    always preserves execution K. The first estimate initializes it directly.
    ``previous`` restores a filter record's ``stamp`` and ``execution_logK``;
    it need not belong to an adopted IK plan. A supplied ``execution_K`` is
    checked and preserved.
    All returned values are independent JSON-compatible numbers and lists.
    """

    def __init__(self, settings, previous=None):
        settings = dict(settings or {})
        try:
            self.tau = float(settings.get('kp_exec_tau', .5))
            self.max_log_step = float(settings.get('max_log_kp_exec_step', .05))
        except (TypeError, ValueError) as error:
            raise ValueError('Execution stiffness settings must be finite nonnegative scalars.') from error
        if (not np.isfinite(self.tau) or self.tau < 0.
                or not np.isfinite(self.max_log_step) or self.max_log_step < 0.):
            raise ValueError('Execution stiffness settings must be finite nonnegative scalars.')
        self._stamp = self._log_execution = self._execution = None
        if previous is not None:
            if not isinstance(previous, dict) or not {'stamp', 'execution_logK'}.issubset(previous):
                raise ValueError('A previous execution record requires stamp and execution_logK.')
            restored_stamp = _stamp(previous['stamp'])
            restored_log = _vector(previous['execution_logK'], 'Previous execution log K')
            restored = _exp(restored_log)
            if 'execution_K' in previous:
                supplied = _vector(previous['execution_K'], 'Previous execution K', positive=True)
                if supplied.shape != restored.shape or not np.allclose(
                        supplied, restored, rtol=1e-12, atol=0.):
                    raise ValueError('Previous execution K and execution_logK do not agree.')
                restored = supplied
            self._stamp, self._log_execution, self._execution = restored_stamp, restored_log, restored

    def advance(self, estimated_K, stamp):
        """Advance once; invalid input leaves the current filter state intact."""
        estimated = _vector(estimated_K, 'Estimated K', positive=True)
        stamp = _stamp(stamp)
        estimated_log = np.log(estimated)
        initialize = self._log_execution is None
        previous_stamp = self._stamp
        previous_log = None if initialize else self._log_execution.copy()
        if initialize:
            dt, alpha = 0., 0.
            raw_delta = applied_delta = np.zeros_like(estimated)
            execution_log, execution = estimated_log.copy(), estimated.copy()
        else:
            if estimated.shape != self._log_execution.shape:
                raise ValueError('Estimated K shape must match the execution stiffness vector.')
            dt = stamp-self._stamp
            if not np.isfinite(dt) or dt < 0.:
                raise ValueError('Execution stiffness stamps must be nondecreasing with finite dt.')
            alpha = (0. if dt == 0. else 1. if self.tau == 0.
                     else float(-np.expm1(-dt/self.tau)))
            raw_delta = alpha*(estimated_log-previous_log)
            applied_delta = (np.clip(raw_delta, -self.max_log_step, self.max_log_step)
                             if self.max_log_step > 0. else raw_delta.copy())
            execution_log = previous_log+applied_delta
            if np.array_equal(execution_log, previous_log):
                execution = self._execution.copy()
            elif np.array_equal(execution_log, estimated_log):
                execution = estimated.copy()
            else:
                execution = _exp(execution_log)
        record = dict(stamp=stamp, previous_stamp=previous_stamp,
            estimated_K=estimated.tolist(), execution_K=execution.tolist(),
            estimated_logK=estimated_log.tolist(), execution_logK=execution_log.tolist(),
            previous_execution_logK=None if previous_log is None else previous_log.tolist(),
            actual_dt=dt, alpha=alpha, raw_delta_logK=raw_delta.tolist(),
            applied_delta_logK=applied_delta.tolist(), kp_exec_tau=self.tau,
            max_log_kp_exec_step=self.max_log_step,
            initialized=True, initialized_this_step=initialize)
        self._stamp, self._log_execution, self._execution = stamp, execution_log.copy(), execution.copy()
        return record
