"""Six-point Gaussian cubature through controlled rigid-body rotation.

The only replacement of Chapter 4's time predictor is the map from the
initial body angular velocity to a right Hamilton rotation increment.  All
quaternions use wxyz and map body vectors into world coordinates.  The
Gaussian cubature is deterministic, with six positive weights of 1/6; it is
not Monte Carlo and is not exact for the nonlinear flow.

Within a prediction interval the six initial angular velocities are fixed.
Each trajectory jointly integrates its angular velocity and quaternion, so
their dependence is retained.  A caller that replaces angular velocity at a
new acquisition by an independent Gaussian must declare that Chapter 4
closure; this module never silently re-cubatures an intermediate state.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
CORE = HERE.parents[3] / 'core/probtf_core/src'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
from probtf.bingham import quaternion_product_second_moment


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite array of shape ' + str(shape))
    return array


def quaternion_multiply(left, right):
    """Broadcast the Hamilton product, retaining the input wxyz convention."""
    a, b = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    if a.shape[-1] != 4:
        raise ValueError('Hamilton quaternion arrays must have last dimension 4')
    scalar = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    vector = (a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
              + np.cross(a[..., 1:], b[..., 1:]))
    return np.concatenate((scalar, vector), axis=-1)


def gaussian_six_points(mean_omega, covariance_omega):
    """Return Chapter 4's +/-sqrt(3) Cholesky columns and six equal weights.

    The covariance is that of one acquired angular velocity in (rad/s)^2,
    not a continuous-time spectral density.  Positive-definite covariances
    use Cholesky without jitter or flooring.  The exactly deterministic zero
    covariance is supported as six coincident points with the same weights.
    """
    mean = _finite(mean_omega, (3,), 'mean_omega')
    covariance = _finite(covariance_omega, (3, 3), 'covariance_omega')
    covariance = .5 * (covariance + covariance.T)
    root = (np.zeros((3, 3)) if not np.any(covariance)
            else np.linalg.cholesky(covariance))
    offsets = np.sqrt(3.) * root.T
    points = np.concatenate((mean[None, :] + offsets, mean[None, :] - offsets))
    return points, np.full(6, 1. / 6.)


def _constant_rate_increment(omega, elapsed):
    vectors = float(elapsed) * np.asarray(omega)
    angles = np.linalg.norm(vectors, axis=-1)
    # sin(angle/2)/angle = sinc(angle/(2*pi))/2, including angle=0.
    scales = .5 * np.sinc(angles / (2. * np.pi))
    return np.concatenate((np.cos(.5 * angles)[:, None],
                           vectors * scales[:, None]), axis=1)


def integrate_six_point_increment(mean_omega, covariance_omega, times,
                                 torque, inertia, *, mode='nonlinear',
                                 max_step=.002):
    """Integrate six initial conditions against one fixed body torque history.

    Parameters
    ----------
    times : increasing vector
        Output times on the same clock accepted by ``torque``.  The first is
        the interval's start, with identity rotation increment.
    torque : callable(t)->(3,), or a constant (3,) array
        Actual actuator torque plus the declared body-CoG residual torque.
        Every cubature trajectory uses this same known input history.
    inertia : (3,3)
        One fixed symmetric positive-definite body-CoG inertia tensor.
    mode : nonlinear | linear_forced | constant
        Nonlinear uses J*omega_dot=tau-omega cross(J*omega) and q_dot =
        q*[0,omega]/2 with joint RK4 stages.  Constant is the original
        held-angular-velocity comparator, which intentionally ignores torque.
        Linear_forced retains the identical torque input but omits only the
        omega cross(J*omega) term, isolating the Euler nonlinearity's effect.

    Intermediate output times do not introduce new Gaussian noise.  Quaternion
    normalization corrects finite-step integrator norm drift only; it does
    not modify an angular-velocity covariance or its quadrature weights.
    """
    grid = np.asarray(times, float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0.)):
        raise ValueError('times must be a nonempty finite strictly increasing vector')
    tensor = _finite(inertia, (3, 3), 'inertia')
    tensor = .5 * (tensor + tensor.T)
    np.linalg.cholesky(tensor)
    inverse = np.linalg.solve(tensor, np.eye(3))
    step_limit = float(max_step)
    if not np.isfinite(step_limit) or step_limit <= 0.:
        raise ValueError('max_step must be positive and finite')
    if mode not in ('nonlinear', 'linear_forced', 'constant'):
        raise ValueError('mode must be nonlinear, linear_forced or constant')
    nodes, weights = gaussian_six_points(mean_omega, covariance_omega)
    if callable(torque):
        torque_function = torque
    else:
        fixed_torque = _finite(torque, (3,), 'torque')
        torque_function = lambda _time: fixed_torque

    angular = np.empty((len(grid), 6, 3))
    increments = np.empty((len(grid), 6, 4))
    angular[0] = nodes
    increments[0] = np.array([1., 0., 0., 0.])
    max_norm_error = 0.
    count = 0

    def derivative(time, state):
        omega, quaternion = state[:, :3], state[:, 3:]
        applied = _finite(torque_function(float(time)), (3,), 'torque(time)')
        momentum = omega @ tensor.T
        gyroscopic = np.cross(omega, momentum) if mode == 'nonlinear' else np.zeros_like(omega)
        acceleration = (applied - gyroscopic) @ inverse.T
        pure_omega = np.concatenate((np.zeros((6, 1)), omega), axis=1)
        quaternion_rate = .5 * quaternion_multiply(quaternion, pure_omega)
        return np.concatenate((acceleration, quaternion_rate), axis=1)

    if mode == 'constant':
        angular[:] = nodes
        for index, time in enumerate(grid):
            increments[index] = _constant_rate_increment(nodes, time-grid[0])
    else:
        state = np.concatenate((nodes, increments[0]), axis=1)
        for output in range(1, len(grid)):
            lower, upper = grid[output-1], grid[output]
            steps = max(1, int(np.ceil((upper-lower)/step_limit)))
            dt = (upper-lower)/steps
            for substep in range(steps):
                time = lower + substep*dt
                first = derivative(time, state)
                second = derivative(time+.5*dt, state+.5*dt*first)
                third = derivative(time+.5*dt, state+.5*dt*second)
                fourth = derivative(time+dt, state+dt*third)
                state = state + dt/6.*(first+2.*second+2.*third+fourth)
                norms = np.linalg.norm(state[:, 3:], axis=1)
                if not np.isfinite(state).all() or not np.isfinite(norms).all() or np.any(norms == 0.):
                    raise FloatingPointError('nonlinear rigid-body integration produced a non-finite or zero quaternion')
                max_norm_error = max(max_norm_error, float(np.max(np.abs(norms-1.))))
                state[:, 3:] /= norms[:, None]
                count += 1
            angular[output], increments[output] = state[:, :3], state[:, 3:]
    moments = np.einsum('s,tsi,tsj->tij', weights, increments, increments)
    mean_angular = np.einsum('s,tsi->ti', weights, angular)
    centered = angular - mean_angular[:, None, :]
    covariance_angular = np.einsum('s,tsi,tsj->tij', weights, centered, centered)
    # q*q^T is antipodal invariant.  Its covariance with omega describes a
    # retained part of the within-interval attitude/rate dependence.
    cross = np.einsum('s,tsi,tsj,tsk->tijk', weights, increments, increments, centered)
    return dict(times=grid.copy(), weights=weights, initial_omega_nodes=nodes,
                omega=angular, delta=increments, M2delta=moments,
                mean_omega=mean_angular, covariance_omega=covariance_angular,
                covariance_delta_second_omega=cross, mode=mode,
                nonlinear_rk4_steps=count, max_step=step_limit,
                maximum_quaternion_norm_correction=max_norm_error,
                quaternion_order='wxyz', increment_action='right',
                angular_velocity_units='rad/s', covariance_is_sample_variance=True,
                repeated_noise_in_substeps=False, monte_carlo=False)


def propagate_orientation_second_moments(initial_second_moment, increment_moments):
    """Use the existing core's independent quaternion-product moment formula.

    Initial orientation and the interval's angular-velocity law are assumed
    independent.  The caller must not use this factorization for a dependent
    initial joint law without stating the approximation.
    """
    initial = _finite(initial_second_moment, (4, 4), 'initial_second_moment')
    increments = np.asarray(increment_moments, float)
    if increments.ndim != 3 or increments.shape[1:] != (4, 4):
        raise ValueError('increment_moments must have shape (time,4,4)')
    return np.stack([quaternion_product_second_moment(initial, moment)
                     for moment in increments])


def compose_product_cubature(parent_quaternions, parent_weights,
                             increment_quaternions, increment_weights):
    """Enumerate a finite product law for plotting its nonlinear Euler moments.

    This adds no quadrature points and performs no sampling.  For three
    acquired six-point increments and deterministic Q0 there are 6^3=216
    weighted products, exactly the finite law whose second moment the core
    formula computes.  Retain the frozen interval-start parents at every
    intermediate time; do not compose the same increment more than once.
    """
    parents, increments = np.asarray(parent_quaternions, float), np.asarray(increment_quaternions, float)
    left_weights, right_weights = np.asarray(parent_weights, float), np.asarray(increment_weights, float)
    if (parents.ndim != 2 or parents.shape[1] != 4 or increments.ndim != 2 or increments.shape[1] != 4
            or left_weights.shape != (len(parents),) or right_weights.shape != (len(increments),)):
        raise ValueError('finite product-law arrays have inconsistent shapes')
    products = quaternion_multiply(parents[:, None, :], increments[None, :, :]).reshape(-1, 4)
    weights = (left_weights[:, None]*right_weights[None, :]).reshape(-1)
    return products, weights


def weighted_rpy_summary(quaternions, weights, *, reference_rpy=None):
    """Exact Euler first/two moments of the supplied finite law in one chart.

    Euler angles are unwrapped by multiples of 2*pi about ``reference_rpy``.
    If omitted, the quaternion second-moment mode chooses the chart.  The
    band formed by +/-2 standard deviations is a descriptive moment band,
    not a claim of Gaussian RPY density or exact 95% coverage.  Nonlinear
    six-point flow approximation is still present in the underlying law.
    """
    points, mass = np.asarray(quaternions, float), np.asarray(weights, float)
    if points.ndim != 2 or points.shape[1] != 4 or mass.shape != (len(points),):
        raise ValueError('quaternion/weight shapes are inconsistent')
    if (not np.isfinite(points).all() or not np.isfinite(mass).all()
            or np.any(mass < 0.) or mass.sum() <= 0.):
        raise ValueError('finite positive-mass quaternion law is required')
    mass = mass / mass.sum()
    second = np.einsum('s,si,sj->ij', mass, points, points)
    if reference_rpy is None:
        _eigenvalues, eigenvectors = np.linalg.eigh(second)
        mode = eigenvectors[:, -1]
        reference = Rotation.from_quat(mode[[1, 2, 3, 0]]).as_euler('xyz')
    else:
        reference = _finite(reference_rpy, (3,), 'reference_rpy')
    angles = Rotation.from_quat(points[:, [1, 2, 3, 0]]).as_euler('xyz')
    unwrapped = reference + (angles-reference+np.pi) % (2.*np.pi)-np.pi
    mean = np.einsum('s,si->i', mass, unwrapped)
    centered = unwrapped-mean
    covariance = np.einsum('s,si,sj->ij', mass, centered, centered)
    return dict(mean=mean, covariance=covariance, second_moment=second,
                chart_reference=reference.copy(), unwrapped_points=unwrapped,
                weights=mass, band_interpretation='mean +/- 2 weighted standard deviations; no coverage guarantee')
