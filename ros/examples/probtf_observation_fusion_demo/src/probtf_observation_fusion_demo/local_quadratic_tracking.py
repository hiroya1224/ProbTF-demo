"""Exact small affine dynamics for a frozen local command-gradient model.

The model is g(u)=g_s+H_s(u-u_s).  It is an execution approximation, not a
replacement of the candidate law or a claim that H_s is the exact objective
Hessian.  No body propagation or observation evaluation is performed here.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm


def _vector(value, name, shape=None):
    result = np.asarray(value, dtype=float)
    if (result.ndim != 1 or not result.size or
            (shape is not None and result.shape != shape) or
            not np.all(np.isfinite(result))):
        raise ValueError('Local quadratic %s must be a finite matching vector.' % name)
    return result.copy()


def _interval(dt, eta, gamma):
    dt, eta, gamma = float(dt), float(eta), float(gamma)
    if (not all(math.isfinite(x) for x in (dt, eta, gamma)) or dt < 0.
            or eta <= 0. or gamma < 0.):
        raise ValueError('Local quadratic dt/gamma must be nonnegative and eta positive, all finite.')
    return dt, eta, gamma


@dataclass(frozen=True)
class LocalQuadraticIntervalCheck:
    violation: object
    bound_verified: bool
    evaluated_interior_points: int
    unresolved_interval_count: int
    comparison_bound_unavailable_count: int

    def record(self):
        return dict(method='floating-point analytic displacement comparison, with deterministic midpoint refinement near bounds',
            bound_verified=self.bound_verified,
            evaluated_interior_points=self.evaluated_interior_points,
            unresolved_interval_count=self.unresolved_interval_count,
            comparison_bound_unavailable_count=self.comparison_bound_unavailable_count,
            formal_interval_arithmetic=False,
            unresolved_comparison_is_a_stop_gate=False,
            violation_is_an_evaluated_trajectory_point=self.violation is not None)


@dataclass(frozen=True, init=False, eq=False)
class FrozenLocalQuadraticModel:
    """Copy one source command, complete gradient and curvature atomically.

    A symmetric positive-definite H gives a unique finite equilibrium and a
    damped stable linear system when gamma>0. Spectral diagnostics never
    modify H or reject finite arithmetic. The producer owns the declared
    positive-definite approximation and its regularization metadata.
    """

    def __init__(self, source_command, gradient, curvature):
        object.__setattr__(self, '_source', _vector(source_command, 'source command'))
        object.__setattr__(self, '_gradient', _vector(gradient, 'source gradient', self._source.shape))
        object.__setattr__(self, '_curvature', np.asarray(curvature, dtype=float).copy())
        n = self._source.size
        if self._curvature.shape != (n, n) or not np.all(np.isfinite(self._curvature)):
            raise ValueError('Local quadratic curvature must be a finite matching square matrix.')
        for value in (self._source, self._gradient, self._curvature):
            value.setflags(write=False)
        object.__setattr__(self, 'dimension', n)
        object.__setattr__(self, 'is_zero', not np.any(self._curvature))
        # Upward rounding is used only in the inexpensive path-bound estimate.
        try:
            with np.errstate(all='ignore'):
                bound = max(float(np.nextafter(
                    math.fsum(abs(float(x)) for x in row), np.inf)) for row in self._curvature)
        except OverflowError:
            # This norm is only a path-comparison diagnostic. A finite H may
            # still have an overflowing row sum and an evaluable eta*H.
            bound = math.inf
        object.__setattr__(self, '_curvature_inf', bound)

    def gradient_at(self, command):
        command = _vector(command, 'evaluation command', self._source.shape)
        with np.errstate(all='ignore'):
            gradient = self._gradient+self._curvature @ (command-self._source)
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError('Local quadratic gradient contains NaN or infinity.')
        return gradient

    def advance(self, command, velocity, dt, eta, gamma):
        """Integrate from the current state; source time is never replayed.

        A displacement state starting at zero avoids subtracting a possibly
        huge stationary command H^-1 g when curvature is small. The augmented
        matrix exponential includes the affine forcing without cancellation.
        """
        command = _vector(command, 'command', self._source.shape)
        velocity = _vector(velocity, 'velocity', self._source.shape)
        dt, eta, gamma = _interval(dt, eta, gamma)
        if self.is_zero:
            # The same function, arithmetic order and small-gamma coefficients
            # preserve the former held-gradient path bit for bit.
            from .running_gradient import step_running_gradient
            return step_running_gradient(command, velocity, self._gradient, dt, eta, gamma)
        if dt == 0.:
            return command, velocity
        gradient = self.gradient_at(command)
        n = self.dimension
        system = np.zeros((2*n+1, 2*n+1))
        system[:n, n:2*n] = np.eye(n)
        system[n:2*n, :n] = -eta*self._curvature
        system[n:2*n, n:2*n] = -gamma*np.eye(n)
        system[n:2*n, -1] = -eta*gradient
        state = np.concatenate((np.zeros(n), velocity, [1.]))
        with np.errstate(all='ignore'):
            result = expm(system*dt) @ state
            next_command, next_velocity = command+result[:n], result[n:2*n]
        if not np.all(np.isfinite(next_command)) or not np.all(np.isfinite(next_velocity)):
            raise FloatingPointError('Local quadratic integration produced NaN or infinity.')
        return next_command, next_velocity

    def _displacement_bound(self, command, velocity, dt, eta, gamma):
        """Norm comparison d''+gamma*d'-eta*||H||inf*d=eta*||g(u0)||inf.

        The comparison is a sufficient arithmetic enclosure, not the physical
        path. Failure to establish it can only request more path evaluation.
        """
        force = float(np.max(np.abs(self.gradient_at(command))))
        speed = float(np.max(np.abs(velocity)))
        with np.errstate(all='ignore'):
            kappa = float(np.nextafter(eta*self._curvature_inf, np.inf))
            forcing = float(np.nextafter(eta*force, np.inf))
            system = np.array([[0., 1., 0.], [kappa, -gamma, forcing], [0., 0., 0.]])
            radius = float((expm(system*dt) @ np.array([0., speed, 1.]))[0])
        if not math.isfinite(radius) or radius < 0.:
            return math.inf
        # A floating comparison has no formal interval-arithmetic guarantee.
        # A roundoff margin makes it conservative in the ordinary small-step
        # regime; it never changes the actual state or any acceptance rule.
        with np.errstate(all='ignore'):
            return float(np.nextafter(radius*(1.+64.*np.finfo(float).eps), np.inf))

    def check_joint_interval(self, command, velocity, dt, eta, gamma,
                             lower, upper, *, endpoint=None,
                             maximum_intervals=128, maximum_depth=18):
        """Look for a real path violation, never stop on an uncertain bound.

        Far from a joint limit the norm comparison needs no interior matrix
        exponential. Close to a limit, deterministic refinement evaluates the
        same exact affine trajectory. A finite refinement budget can leave an
        unresolved enclosure; this is reported and execution continues. No
        root-completeness or formal continuous-path guarantee is claimed there.
        """
        command = _vector(command, 'command', self._source.shape)
        velocity = _vector(velocity, 'velocity', self._source.shape)
        lower = _vector(lower, 'joint lower', self._source.shape)
        upper = _vector(upper, 'joint upper', self._source.shape)
        dt, eta, gamma = _interval(dt, eta, gamma)
        final, final_velocity = (self.advance(command, velocity, dt, eta, gamma)
                                 if endpoint is None else endpoint)
        pending = [(0., command, velocity, dt, final, final_velocity, 0)]
        visited = interior = unresolved = unavailable = 0
        while pending:
            left, point, speed, right, end, end_speed, depth = pending.pop()
            for stamp, value in ((left, point), (right, end)):
                bad = np.flatnonzero((value < lower) | (value > upper))
                if bad.size:
                    return LocalQuadraticIntervalCheck((int(bad[0]), stamp, value.copy()),
                        False, interior, unresolved, unavailable)
            if right == left:
                continue
            if visited >= maximum_intervals:
                unresolved += 1
                continue
            visited += 1
            try:
                radius = self._displacement_bound(point, speed, right-left, eta, gamma)
            except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
                radius = math.inf
            if not math.isfinite(radius):
                unavailable += 1
            if np.all(point-radius >= lower) and np.all(point+radius <= upper):
                continue
            middle = .5*(left+right)
            if depth >= maximum_depth or middle == left or middle == right:
                unresolved += 1
                continue
            halfway, half_speed = self.advance(point, speed, middle-left, eta, gamma)
            interior += 1
            bad = np.flatnonzero((halfway < lower) | (halfway > upper))
            if bad.size:
                return LocalQuadraticIntervalCheck((int(bad[0]), middle, halfway.copy()),
                    False, interior, unresolved, unavailable)
            pending.append((middle, halfway, half_speed, right, end, end_speed, depth+1))
            pending.append((left, point, speed, middle, halfway, half_speed, depth+1))
        return LocalQuadraticIntervalCheck(None, unresolved == 0, interior, unresolved, unavailable)
