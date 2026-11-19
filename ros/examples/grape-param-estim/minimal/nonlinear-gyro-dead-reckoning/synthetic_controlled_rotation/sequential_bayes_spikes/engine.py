"""Causal Bayes updates of a controlled nonlinear rotation law.

The unknown is the *same* initial body angular velocity W0 throughout the
experiment.  Known J, known torque, and the once-given initial orientation
push each W0 into a valid quaternion and angular velocity at every time.
An observation multiplies the weights of that joint law; it does not
replace angular velocity by a new independent Gaussian or discard the
orientation/rate dependence.

A fixed, deterministic tensor Gauss--Hermite rule approximates the initial
three-dimensional Gaussian integral.  Sequential likelihood multiplication
is exact for that finite rule, apart from floating arithmetic.  It is not
Monte Carlo and not a Gaussian projection of the posterior.  Quadrature
refinement and finite RK4 error remain numerical approximations.  A fixed
rule can under-resolve a very concentrated or far-tail posterior; that is a
reason to report/refine integration, not to reject a finite observation.
"""
from pathlib import Path
import importlib.util
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
NONLINEAR = EXPERIMENT.parent
if str(NONLINEAR) not in sys.path:
    sys.path.insert(0, str(NONLINEAR))
from propagation import quaternion_multiply
from probtf.bingham import rotation_first_moment

# This file is itself named engine.py.  Import the unchanged display helper
# under a distinct name rather than accidentally importing ourselves.
_spec = importlib.util.spec_from_file_location(
    '_sequential_bayes_existing_display', NONLINEAR/'rate_comparison_5hz/engine.py')
_display = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_display)
euler_proxy = _display.euler_proxy


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name+' must be finite with shape '+str(shape))
    return result


def gaussian_tensor_points(mean, covariance, order):
    """Return positive tensor Gauss--Hermite nodes for N(mean,covariance).

    ``order`` is the number of abscissae per coordinate, not a sample count.
    There are order**3 deterministic nodes.  No random generator, covariance
    floor, jitter, or state-dependent acceptance criterion is used.
    """
    mean = _finite(mean, (3,), 'initial angular-rate mean')
    covariance = _finite(covariance, (3, 3), 'initial angular-rate covariance')
    if int(order) != order or int(order) < 2:
        raise ValueError('Gauss--Hermite order must be an integer >= 2')
    root = np.linalg.cholesky(.5*(covariance+covariance.T))
    abscissae, mass = hermgauss(int(order))
    coordinates = np.stack(np.meshgrid(abscissae, abscissae, abscissae,
                                      indexing='ij'), axis=-1).reshape(-1, 3)
    weights = np.einsum('i,j,k->ijk', mass, mass, mass).reshape(-1)/np.pi**1.5
    weights /= weights.sum()
    nodes = mean+np.sqrt(2.)*coordinates@root.T
    if (not np.isfinite(nodes).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0.)):
        raise FloatingPointError('Non-finite nodes or non-positive tensor quadrature weights')
    return nodes, weights


def bayes_log_weight_update(log_weights, omega_nodes, observation, covariance):
    """Multiply a joint-law prior by ONE Gaussian gyro likelihood.

    Returns normalized posterior log masses and the log predictive density.
    Prior normalization is accounted for, so the returned evidence is an
    increment rather than an accumulated unnormalized likelihood.  Tiny
    masses remain represented in the log domain; no node is pruned.
    """
    omega = np.asarray(omega_nodes, float)
    if omega.ndim != 2 or omega.shape[1] != 3 or not np.isfinite(omega).all():
        raise ValueError('omega_nodes must be a finite (nodes,3) array')
    prior = _finite(log_weights, (len(omega),), 'prior log masses')
    measured = _finite(observation, (3,), 'gyro observation')
    covariance = _finite(covariance, (3, 3), 'gyro covariance')
    root = np.linalg.cholesky(.5*(covariance+covariance.T))
    residual = measured-omega
    whitened = np.linalg.solve(root, residual.T).T
    likelihood = -.5*np.sum(whitened**2, axis=1)
    likelihood -= 1.5*np.log(2.*np.pi)+np.log(np.diag(root)).sum()
    combined = prior+likelihood
    normalizer = float(logsumexp(combined))
    prior_normalizer = float(logsumexp(prior))
    if not np.isfinite(normalizer):
        raise FloatingPointError('Gaussian observation has no finite log normalizer')
    return combined-normalizer, normalizer-prior_normalizer


def _weighted_mean_covariance(values, weights):
    mean = weights@values
    centered = values-mean
    covariance = (centered.T*weights)@centered
    return mean, .5*(covariance+covariance.T)


def summarize_joint(quaternion_wxyz, omega, weights, initial_nodes=None):
    """Moments of ONE weighted joint law, retaining all rotation/rate cross.

    Every quaternion is a valid physical rotation.  The 9 rotation entries
    are used as random-variable coordinates only; a Gaussian in ambient
    matrix coordinates is never used as an inference state.  RPY is the
    unchanged Euler/delta-method display proxy from the same moments.
    """
    angular = np.asarray(omega, float)
    if angular.ndim != 2 or angular.shape[1] != 3 or not np.isfinite(angular).all():
        raise ValueError('omega must be a finite (nodes,3) array')
    quaternion = _finite(quaternion_wxyz, (len(angular), 4), 'quaternion nodes')
    mass = _finite(weights, (len(angular),), 'joint masses')
    if np.any(mass < 0.) or not mass.sum() > 0.:
        raise ValueError('Joint masses must be nonnegative and have positive total')
    mass = mass/mass.sum()
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    vector = rotation.transpose(0, 2, 1).reshape(len(rotation), 9)
    empirical_mean = mass@vector
    moment = (quaternion.T*mass)@quaternion
    moment = .5*(moment+moment.T)
    core_mean = rotation_first_moment(moment)
    mean_vector = core_mean.reshape(9, order='F')
    rotation_centered = vector-mean_vector
    angular_mean, angular_covariance = _weighted_mean_covariance(angular, mass)
    angular_centered = angular-angular_mean
    covariance = (rotation_centered.T*mass)@rotation_centered
    covariance = .5*(covariance+covariance.T)
    cross = (rotation_centered.T*mass)@angular_centered
    rpy, rpy_covariance, _ = euler_proxy(core_mean, covariance)
    result = dict(
        quaternion_second_moment=moment,
        rotation_mean=core_mean,
        rotation_covariance_colmajor=covariance,
        omega_mean_rad_s=angular_mean,
        omega_covariance_rad2_s2=angular_covariance,
        rotation_omega_covariance_colmajor=cross,
        rpy_proxy_rad=rpy,
        rpy_covariance_delta_rad2=rpy_covariance,
        core_vs_rotation_mean_max_abs=np.asarray(np.max(np.abs(mean_vector-empirical_mean))))
    if initial_nodes is not None:
        initial = _finite(initial_nodes, (len(angular), 3), 'initial angular-rate nodes')
        initial_mean, initial_covariance = _weighted_mean_covariance(initial, mass)
        result.update(initial_omega_mean_rad_s=initial_mean,
                      initial_omega_covariance_rad2_s2=initial_covariance)
    return result


def predict_joint(data, torque_profile, sigma_deg_s=5., order=31,
                  output_dt=.01, max_step=.002, observation_sets=None,
                  *, mode='nonlinear'):
    """Propagate one shared controlled flow; Bayes-update each observation case.

    Parameters use the existing selected-input schema: ``time=[start,end]``,
    ``initial_quaternion_wxyz``, ``sparse_gyro_time``, and ``sparse_gyro``.
    ``torque_profile`` supplies known ``inertia`` and callable ``torque``.
    ``observation_sets`` maps case names to (acquisitions,3) arrays.  All
    cases share the initial observation and therefore the same initial
    Gaussian N(y0,R); the initial reading is not multiplied a second time.

    Results map case names to numerical arrays, plus ``events`` and
    ``metadata`` records.  An event is recorded twice at the same original
    timestamp: event_side=-1 before and +1 after the current likelihood.
    Ordinary outputs have side=0 and acquisition_index=-1.  The initial
    record is side=+1.  Earlier online outputs are never retrospectively
    replaced by later observation information.

    No process noise or independently refreshed input law is introduced.
    Initial uncertainty plus posterior likelihoods determine all reported
    covariances and cross covariances.  The linear_forced mode is provided
    for analytical Gaussian-update verification, not the principal result.
    """
    bounds = _finite(data['time'], (2,), 'experiment bounds')
    start, end = bounds
    observations = np.asarray(data['sparse_gyro_time'], float)
    measured = np.asarray(data['sparse_gyro'], float)
    if (observations.ndim != 1 or not len(observations)
            or not np.isfinite(observations).all()
            or measured.shape != (len(observations), 3) or not np.isfinite(measured).all()
            or np.any(np.diff(observations) <= 0.) or observations[0] != start
            or observations[-1] > end or end <= start):
        raise ValueError('Increasing finite observation times must start at experiment start')
    sigma, output_dt, max_step = float(sigma_deg_s), float(output_dt), float(max_step)
    if not all(np.isfinite(x) and x > 0. for x in (sigma, output_dt, max_step)):
        raise ValueError('Positive finite sigma, output_dt and max_step are required')
    if mode not in ('nonlinear', 'linear_forced'):
        raise ValueError('mode must be nonlinear or linear_forced')
    quaternion0 = _finite(data['initial_quaternion_wxyz'], (4,), 'initial quaternion')
    if np.linalg.norm(quaternion0) == 0.:
        raise ValueError('Initial quaternion must be nonzero')
    quaternion0 = quaternion0/np.linalg.norm(quaternion0)
    tensor = _finite(torque_profile['inertia'], (3, 3), 'inertia')
    tensor = .5*(tensor+tensor.T)
    np.linalg.cholesky(tensor)
    inverse = np.linalg.solve(tensor, np.eye(3))
    torque = torque_profile['torque']
    if not callable(torque):
        fixed_torque = _finite(torque, (3,), 'known torque')
        torque = lambda _stamp: fixed_torque
    covariance = np.eye(3)*np.deg2rad(sigma)**2
    nodes, prior_mass = gaussian_tensor_points(measured[0], covariance, order)
    initial_log_mass = np.log(prior_mass)
    cases = {'default': measured} if observation_sets is None else dict(observation_sets)
    if not cases:
        raise ValueError('At least one observation case is required')
    for name, values in cases.items():
        values = _finite(values, measured.shape, 'observations for '+str(name))
        if not np.array_equal(values[0], measured[0]):
            raise ValueError('All cases must share the same first observation')
        cases[name] = values
    grid = np.unique(np.r_[np.linspace(start, end, int(round((end-start)/output_dt))+1),
                           observations])
    event_at = {float(stamp): index for index, stamp in enumerate(observations)}
    state = np.column_stack((nodes, np.broadcast_to(quaternion0, (len(nodes), 4))))
    zero_scalar = np.zeros((len(nodes), 1))
    fields = ('time', 'event_side', 'acquisition_index',
              'quaternion_second_moment', 'rotation_mean', 'rotation_covariance_colmajor',
              'omega_mean_rad_s', 'omega_covariance_rad2_s2',
              'rotation_omega_covariance_colmajor', 'rpy_proxy_rad',
              'rpy_covariance_delta_rad2', 'initial_omega_mean_rad_s',
              'initial_omega_covariance_rad2_s2', 'core_vs_rotation_mean_max_abs')
    outputs = {name: {key: [] for key in fields} for name in cases}
    logs = {name: initial_log_mass.copy() for name in cases}
    events = {name: [] for name in cases}
    event_logs = {name: [] for name in cases}
    event_prior_logs = {name: [] for name in cases}
    latent = {}

    def update_latent(name):
        mass = np.exp(logs[name])
        mass /= mass.sum()
        latent[name] = _weighted_mean_covariance(nodes, mass)
        return mass

    for name in cases:
        update_latent(name)

    def append(name, stamp, side, acquisition):
        mass = np.exp(logs[name])
        moment = summarize_joint(state[:, 3:], state[:, :3], mass)
        moment.update(time=float(stamp), event_side=int(side), acquisition_index=int(acquisition),
                      initial_omega_mean_rad_s=latent[name][0],
                      initial_omega_covariance_rad2_s2=latent[name][1])
        for key in fields:
            outputs[name][key].append(np.asarray(moment[key]).copy())
        return moment

    for name in cases:
        append(name, start, 1, 0)
        event_prior_logs[name].append(logs[name].copy())
        event_logs[name].append(logs[name].copy())
        events[name].append(dict(acquisition_index=0, time=float(start), initialization=True,
                                 likelihood_applications=0,
                                 log_predictive_evidence=None,
                                 first_reading_used_once=True))

    step_count = 0
    norm_correction = 0.

    def derivative(stamp, values):
        angular, quaternion = values[:, :3], values[:, 3:]
        applied = _finite(torque(float(stamp)), (3,), 'known torque at RK4 stage')
        momentum = angular@tensor.T
        gyroscopic = (np.cross(angular, momentum) if mode == 'nonlinear'
                       else np.zeros_like(angular))
        acceleration = (applied-gyroscopic)@inverse.T
        quaternion_rate = .5*quaternion_multiply(
            quaternion, np.concatenate((zero_scalar, angular), axis=1))
        return np.column_stack((acceleration, quaternion_rate))

    for row in range(1, len(grid)):
        lower, upper = grid[row-1], grid[row]
        count = max(1, int(np.ceil((upper-lower)/max_step)))
        dt = (upper-lower)/count
        for substep in range(count):
            stamp = lower+substep*dt
            k1 = derivative(stamp, state)
            k2 = derivative(stamp+.5*dt, state+.5*dt*k1)
            k3 = derivative(stamp+.5*dt, state+.5*dt*k2)
            k4 = derivative(stamp+dt, state+dt*k3)
            state += dt/6.*(k1+2.*k2+2.*k3+k4)
            norms = np.linalg.norm(state[:, 3:], axis=1)
            if not np.isfinite(state).all() or not np.isfinite(norms).all() or np.any(norms == 0.):
                raise FloatingPointError('Controlled joint RK4 produced a non-finite or zero quaternion')
            norm_correction = max(norm_correction, float(np.max(np.abs(norms-1.))))
            state[:, 3:] /= norms[:, None]
            step_count += 1
        acquisition = event_at.get(float(upper))
        if acquisition is None:
            for name in cases:
                append(name, upper, 0, -1)
            continue
        for name, values in cases.items():
            before = append(name, upper, -1, acquisition)
            event_prior_logs[name].append(logs[name].copy())
            logs[name], evidence = bayes_log_weight_update(
                logs[name], state[:, :3], values[acquisition], covariance)
            mass = update_latent(name)
            after = append(name, upper, 1, acquisition)
            event_logs[name].append(logs[name].copy())
            events[name].append(dict(
                acquisition_index=int(acquisition), time=float(upper), initialization=False,
                likelihood_applications=1, log_predictive_evidence=float(evidence),
                innovation_rad_s=(values[acquisition]-before['omega_mean_rad_s']).tolist(),
                prior_omega_mean_rad_s=before['omega_mean_rad_s'].tolist(),
                posterior_omega_mean_rad_s=after['omega_mean_rad_s'].tolist(),
                prior_omega_covariance_rad2_s2=before['omega_covariance_rad2_s2'].tolist(),
                posterior_omega_covariance_rad2_s2=after['omega_covariance_rad2_s2'].tolist(),
                minimum_quadrature_mass=float(np.min(mass)),
                maximum_quadrature_mass=float(np.max(mass)),
                inverse_squared_mass_sum=float(1./np.dot(mass, mass)),
                zero_mass_from_float_exp_count=int(np.count_nonzero(mass == 0.)),
                quadrature_concentration_is_not_an_acceptance_gate=True))

    result = {}
    for name in cases:
        arrays = {key: np.stack(values) for key, values in outputs[name].items()}
        arrays['rpy_proxy_rad'] = np.unwrap(arrays['rpy_proxy_rad'], axis=0)
        arrays.update(
            initial_omega_nodes_rad_s=nodes.copy(),
            initial_quadrature_weights=prior_mass.copy(),
            event_time=observations.copy(),
            event_log_weights=np.stack(event_logs[name]),
            event_prior_log_weights=np.stack(event_prior_logs[name]),
            measurement_covariance_rad2_s2=covariance.copy())
        arrays['events'] = events[name]
        arrays['metadata'] = dict(
            schema='grape/sequential-joint-bayes-rotation/v1',
            posterior='initial Gaussian times all acquired Gaussian likelihood factors, normalized at each acquisition',
            latent_variable='same initial body angular velocity W0; known initial orientation, J and torque',
            first_reading='N(y0,R) initialization once; equivalent to first-reading posterior under a diffuse initial rate prior',
            noise_model='independent per-reading isotropic Gaussian; no process noise or bias state',
            propagation='same deterministic controlled joint quaternion/angular-rate RK4 map for every node',
            nonlinear_mode=mode, quadrature='fixed tensor Gauss-Hermite over initial 3D Gaussian',
            quadrature_order_per_axis=int(order), quadrature_nodes=len(nodes),
            quadrature_refinement_is_numerical_integration_not_a_state_change=True,
            gaussian_posterior_projection=False, monte_carlo=False,
            resampling=False, node_pruning=False, covariance_floor=False,
            repeated_noise_in_substeps=False, previous_posterior_discarded=False,
            orientation_rate_dependence='same weighted physical joint candidates; all 9x3 cross entries retained',
            rpy='existing f(E[R]) and delta-method covariance, display only; not an exact Gaussian or 95 percent region',
            known_inertia_body_kg_m2=tensor.tolist(), assumed_sigma_deg_s=sigma,
            max_step_seconds=max_step, rk4_steps=step_count,
            maximum_quaternion_norm_correction=norm_correction,
            quaternion_order='wxyz', rotation='body to world',
            covariance_column_order='vec(R) column major; covariance9x9 and cross9x3',
            later_truth_or_spike_labels_used=False,
            finite_quadrature_diagnostic_does_not_stop_updates=True)
        result[name] = arrays
    return result
