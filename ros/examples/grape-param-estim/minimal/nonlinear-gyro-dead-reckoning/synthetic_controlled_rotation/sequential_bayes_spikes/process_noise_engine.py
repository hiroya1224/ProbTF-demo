"""Markov angular process noise with conditional quaternion/rotation moments.

Only omega is numerically quadrature-discretized. At each omega node we
retain conditional quaternion second moments and full centered rotation
covariance. Because the known body-torque omega dynamics do not depend on
attitude, these conditional statistics close under right quaternion action,
Gaussian omega convolution and Gaussian gyro conditioning. No Gaussian or
Bingham posterior is substituted for the joint law. Requadrature is a
deterministic numerical approximation of the omega integral, diagnosed at
every step; no Monte Carlo, covariance floor or state rejection is used.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_existing = _load('_process_existing_joint_moments', 'engine.py')
_native = _load('_process_existing_native_flow', 'native_flow.py')
gaussian_tensor_points = _existing.gaussian_tensor_points
rotation_first_moment = _existing.rotation_first_moment
euler_proxy = _existing.euler_proxy
ControlledFlow = _native.ControlledFlow


def _rotation_forms():
    """Read-only linear map M=E[qqT] -> vec(E[R]) from public core calls."""
    basis = np.eye(4)
    diagonal = [rotation_first_moment(np.outer(q, q)).reshape(9, order='F')
                for q in basis]
    forms = np.zeros((9, 4, 4))
    for a in range(4):
        forms[:, a, a] = diagonal[a]
        for b in range(a+1, 4):
            q = (basis[a]+basis[b])/np.sqrt(2.)
            cross = rotation_first_moment(np.outer(q, q)).reshape(9, order='F')
            forms[:, a, b] = forms[:, b, a] = cross-.5*(diagonal[a]+diagonal[b])
    return forms


_ROTATION_FORMS = _rotation_forms()


def conditional_rotation_means(quaternion_moments):
    return np.einsum('kab,nab->nk', _ROTATION_FORMS, quaternion_moments)


def _normalized(weights):
    weights = np.asarray(weights, float)
    if (weights.ndim != 1 or not np.isfinite(weights).all()
            or np.any(weights < 0.) or not weights.sum() > 0.):
        raise ValueError('Finite nonnegative mixture weights with positive mass required')
    return weights/weights.sum()


def _log_gaussian(values, mean, covariance):
    root = np.linalg.cholesky(covariance)
    difference = np.asarray(values)-mean
    whitened = np.linalg.solve(root, difference.reshape(-1, 3).T).T
    result = -.5*np.sum(whitened**2, axis=-1)
    result -= 1.5*np.log(2*np.pi)+np.log(np.diag(root)).sum()
    return result.reshape(difference.shape[:-1])


def transport_conditionals(quaternion_moments, rotation_covariances, increments_wxyz):
    """Apply q'=q Hamilton delta to each conditional law, preserving all C_R."""
    delta = np.asarray(increments_wxyz, float)
    w, x, y, z = delta.T
    right = np.stack((np.column_stack((w, -x, -y, -z)),
                      np.column_stack((x, w, z, -y)),
                      np.column_stack((y, -z, w, x)),
                      np.column_stack((z, y, -x, w))), axis=1)
    moments = right@quaternion_moments@right.swapaxes(1, 2)
    rotation = Rotation.from_quat(delta[:, [1, 2, 3, 0]]).as_matrix()
    action = np.einsum('nij,ab->niajb', rotation.swapaxes(1, 2), np.eye(3)).reshape(-1, 9, 9)
    covariance = action@rotation_covariances@action.swapaxes(1, 2)
    return .5*(moments+moments.swapaxes(1, 2)), .5*(covariance+covariance.swapaxes(1, 2))


def mixture_moments(weights, omega_means, omega_covariance,
                    quaternion_moments, rotation_covariances):
    """Exact joint moments of the finite mixture with a common local omega C."""
    mass = _normalized(weights)
    omega = np.asarray(omega_means, float)
    means_r = conditional_rotation_means(quaternion_moments)
    moment = np.einsum('n,nab->ab', mass, quaternion_moments)
    moment = .5*(moment+moment.T)
    mean_r = rotation_first_moment(moment)
    vector_r = mean_r.reshape(9, order='F')
    centered_r = means_r-vector_r
    mean_omega = mass@omega
    centered_omega = omega-mean_omega
    covariance_omega = np.asarray(omega_covariance)+(centered_omega.T*mass)@centered_omega
    covariance_r = np.einsum('n,nab->ab', mass, rotation_covariances)
    covariance_r += (centered_r.T*mass)@centered_r
    covariance_r = .5*(covariance_r+covariance_r.T)
    cross = (centered_r.T*mass)@centered_omega
    rpy, rpy_covariance, _ = euler_proxy(mean_r, covariance_r)
    return dict(quaternion_second_moment=moment, rotation_mean=mean_r,
        rotation_covariance_colmajor=covariance_r, omega_mean_rad_s=mean_omega,
        omega_covariance_rad2_s2=.5*(covariance_omega+covariance_omega.T),
        rotation_omega_covariance_colmajor=cross, rpy_proxy_rad=rpy,
        rpy_covariance_delta_rad2=rpy_covariance,
        core_vs_rotation_mean_max_abs=np.asarray(np.max(np.abs(mass@means_r-vector_r))))


def condition_gyro(weights, omega_means, process_covariance, measurement,
                   measurement_covariance):
    """Analytic conditioning of each N(omega; f_i,Q) component by one gyro."""
    mass = _normalized(weights)
    q = np.asarray(process_covariance, float)
    r = np.asarray(measurement_covariance, float)
    combined = q+r
    gain = np.linalg.solve(combined, q.T).T
    with np.errstate(divide='ignore'):
        log_mass = np.log(mass)
    evidence = _log_gaussian(np.asarray(measurement)[None, :], omega_means, combined)
    log_normalizer = float(logsumexp(log_mass+evidence))
    if not np.isfinite(log_normalizer):
        raise FloatingPointError('Undefined Gaussian-mixture observation normalizer')
    updated_mass = np.exp(log_mass+evidence-log_normalizer)
    updated_means = omega_means+(np.asarray(measurement)-omega_means)@gain.T
    # Joseph form keeps the physical covariance positive without cancellation,
    # eigenvalue clipping, floors or jitter; it equals Q-Q(Q+R)^-1Q.
    residual_gain = np.eye(3)-gain
    updated_covariance = residual_gain@q@residual_gain.T+gain@r@gain.T
    return updated_mass, updated_means, .5*(updated_covariance+updated_covariance.T), log_normalizer


def regrid_joint_mixture(weights, omega_means, omega_covariance,
                         quaternion_moments, rotation_covariances, order,
                         *, proposal_mean=None, proposal_covariance=None):
    """Nyström integration of a mixture; its Gaussian proposal is divided out."""
    mass = _normalized(weights)
    exact = mixture_moments(mass, omega_means, omega_covariance,
                             quaternion_moments, rotation_covariances)
    center = (exact['omega_mean_rad_s'] if proposal_mean is None
              else np.asarray(proposal_mean, float))
    proposal_covariance = (exact['omega_covariance_rad2_s2'] if proposal_covariance is None
                           else np.asarray(proposal_covariance, float))
    nodes, base_weights = gaussian_tensor_points(center, proposal_covariance, order)
    with np.errstate(divide='ignore'):
        log_mass = np.log(mass)
    log_components = _log_gaussian(nodes[:, None, :], omega_means[None, :, :], omega_covariance)
    log_components += log_mass[None, :]
    log_density = logsumexp(log_components, axis=1)
    log_importance = np.log(base_weights)+log_density-_log_gaussian(nodes, center, proposal_covariance)
    log_normalizer = float(logsumexp(log_importance))
    if not np.isfinite(log_normalizer):
        raise FloatingPointError('Undefined mixture quadrature normalizer')
    quadrature_weights = np.exp(log_importance-log_normalizer)
    responsibilities = np.exp(log_components-log_density[:, None])
    moments = (responsibilities@np.asarray(quaternion_moments).reshape(len(mass), 16)).reshape(-1, 4, 4)
    means_r = conditional_rotation_means(quaternion_moments)
    conditional_mean_r = conditional_rotation_means(moments)
    # Law of total covariance in centered form: positive within-component
    # covariance plus a centered Gram matrix. Never raw E[RR']-E[R]E[R]'.
    covariances = (responsibilities@np.asarray(rotation_covariances).reshape(len(mass), 81)).reshape(-1, 9, 9)
    for lower in range(0, len(nodes), 64):
        upper = min(lower+64, len(nodes))
        centered = means_r[None, :, :]-conditional_mean_r[lower:upper, None, :]
        covariances[lower:upper] += np.einsum('ji,jia,jib->jab', responsibilities[lower:upper], centered, centered,
                                             optimize=True)
    covariances = .5*(covariances+covariances.swapaxes(1, 2))
    approx = mixture_moments(quadrature_weights, nodes, np.zeros((3, 3)), moments, covariances)
    diagnostic = dict(log_density_normalizer=log_normalizer,
        density_integral_minus_one=float(np.expm1(log_normalizer)),
        maximum_mixture_responsibility_sum_error=float(np.max(abs(responsibilities.sum(axis=1)-1.))),
        zero_mass_from_float_exp_count=int(np.count_nonzero(quadrature_weights == 0.)),
        finite_diagnostic_only=True, observation_rejected=False,
        gaussian_proposal_is_not_posterior=True)
    for key in ('omega_mean_rad_s', 'omega_covariance_rad2_s2', 'quaternion_second_moment',
                'rotation_mean', 'rotation_covariance_colmajor', 'rotation_omega_covariance_colmajor'):
        diagnostic['maximum_'+key+'_difference'] = float(np.max(abs(approx[key]-exact[key])))
    return dict(nodes=nodes, weights=quadrature_weights, quaternion_moments=moments,
        rotation_covariances=covariances, proposal_mean=center.copy(),
        proposal_covariance=proposal_covariance.copy(), diagnostics=diagnostic,
        log_unnormalized_weights=log_importance.copy())


def _fit_parent_proposal(flow, mean, covariance, observation, observation_covariance):
    """Place parent-omega integration nodes, using this observation only.

    A Gaussian approximation is used ONLY to place points. The actual
    continuous mixture density divided by this proposal enters all weights.
    """
    prior_root = np.linalg.cholesky(covariance)
    observation_root = np.linalg.cholesky(observation_covariance)
    times = np.array([flow.start, flow.end])
    def residual(value):
        end = flow.evaluate(np.asarray(value)[None, :], times)[-1, 0, :3]
        return np.r_[np.linalg.solve(prior_root, value-mean),
                     np.linalg.solve(observation_root, end-observation)]
    result = least_squares(residual, mean, jac='3-point', max_nfev=24,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        raise FloatingPointError('Non-finite parent quadrature proposal')
    precision = result.jac.T@result.jac
    root_precision = np.linalg.cholesky(.5*(precision+precision.T))
    inverse_root = np.linalg.solve(root_precision, np.eye(3))
    proposal_covariance = inverse_root.T@inverse_root
    return result.x.copy(), proposal_covariance, dict(
        placement='observation-informed parent omega proposal only; actual mixture/proposal ratio retained',
        cost=float(result.cost), optimality=float(result.optimality), nfev=int(result.nfev),
        success=bool(result.success), status=int(result.status),
        finite_unconverged_proposal_does_not_reject_observation=True)


def predict_process_noise(data, torque_profile, measurement_sigma_deg_s=5.,
        process_sigma_rad_s=3*np.pi/180, order=5, mode='nonlinear', max_step=.002):
    """Add one isotropic omega kick at each selected interval END, then observe.

    The process sigma is per selected interval, not a continuous-time spectral
    density and not per RK4 substep. Instantaneously q is unchanged by the
    endpoint kick; its next physical interval uses the resulting omega law.
    """
    begun = time.monotonic()
    stamps = np.asarray(data['sparse_gyro_time'], float)
    measured = np.asarray(data['sparse_gyro'], float)
    bounds = np.asarray(data['time'], float)
    sigma = float(measurement_sigma_deg_s)
    process_sigma = float(process_sigma_rad_s)
    if (bounds.shape != (2,) or stamps.ndim != 1 or len(stamps) < 2
            or measured.shape != (len(stamps), 3) or not np.isfinite(stamps).all()
            or not np.isfinite(measured).all() or np.any(np.diff(stamps) <= 0.)
            or stamps[0] != bounds[0] or stamps[-1] != bounds[1]
            or not np.isfinite(sigma) or sigma <= 0.
            or not np.isfinite(process_sigma) or process_sigma <= 0.):
        raise ValueError('Finite full experiment clock and positive measurement/process sigma required')
    measurement_covariance = np.eye(3)*np.deg2rad(sigma)**2
    process_covariance = np.eye(3)*process_sigma**2
    q0 = np.asarray(data['initial_quaternion_wxyz'], float)
    if q0.shape != (4,) or not np.isfinite(q0).all() or np.linalg.norm(q0) == 0.:
        raise ValueError('Finite nonzero initial quaternion required')
    q0 = q0/np.linalg.norm(q0)
    # The retained state is a continuous omega Gaussian MIXTURE with
    # conditional orientation laws; quadrature nodes are not its support.
    mass = np.ones(1)
    mixture_means = measured[:1].copy()
    mixture_covariance = measurement_covariance.copy()
    moments = np.outer(q0, q0)[None, :, :]
    rotation_covariances = np.zeros((1, 9, 9))
    output = {}; diagnostics = []; events = []
    def append(index, side, statistics):
        statistics = dict(statistics, time=np.asarray(stamps[index]), event_side=np.asarray(side),
            acquisition_index=np.asarray(index),
            target_likelihood_factor_count=np.asarray(index if side > 0 else index-1))
        for key, value in statistics.items():
            output.setdefault(key, []).append(np.asarray(value).copy())
    append(0, 1, mixture_moments(mass, mixture_means, mixture_covariance,
                                 moments, rotation_covariances))
    total_steps = 0; maximum_norm = 0.; last_progress = 0
    for index in range(1, len(stamps)):
        lower, upper = stamps[index-1:index+1]
        flow = ControlledFlow(torque_profile, np.array([1., 0., 0., 0.]), lower, upper,
                              max_step=max_step, mode=mode)
        previous = mixture_moments(mass, mixture_means, mixture_covariance, moments, rotation_covariances)
        # Prior prediction uses only the inherited law. No current gyro
        # value influences this published pre-observation output.
        prior_rule = regrid_joint_mixture(mass, mixture_means, mixture_covariance,
                                           moments, rotation_covariances, order)
        prior_end = flow.evaluate(prior_rule['nodes'], np.array([lower, upper]))[-1]
        prior_moments, prior_covariances = transport_conditionals(
            prior_rule['quaternion_moments'], prior_rule['rotation_covariances'], prior_end[:, 3:])
        prior = mixture_moments(prior_rule['weights'], prior_end[:, :3], process_covariance,
                                 prior_moments, prior_covariances)
        append(index, -1, prior)
        # A large observation may place the useful parent-omega region far
        # outside a prior-centered finite GH grid. Move the integration
        # proposal, keeping the actual continuous mixture/proposal ratio.
        proposal_mean, proposal_covariance, placement = _fit_parent_proposal(flow,
            previous['omega_mean_rad_s'], previous['omega_covariance_rad2_s2'],
            measured[index], process_covariance+measurement_covariance)
        posterior_rule = regrid_joint_mixture(mass, mixture_means, mixture_covariance,
            moments, rotation_covariances, order, proposal_mean=proposal_mean,
            proposal_covariance=proposal_covariance)
        posterior_end = flow.evaluate(posterior_rule['nodes'], np.array([lower, upper]))[-1]
        propagated_omega = posterior_end[:, :3]
        propagated_moments, propagated_covariances = transport_conditionals(
            posterior_rule['quaternion_moments'], posterior_rule['rotation_covariances'], posterior_end[:, 3:])
        updated_mass, updated_means, updated_covariance, evidence = condition_gyro(
            posterior_rule['weights'], propagated_omega, process_covariance,
            measured[index], measurement_covariance)
        evidence += posterior_rule['diagnostics']['log_density_normalizer']
        posterior = mixture_moments(updated_mass, updated_means, updated_covariance,
                                     propagated_moments, propagated_covariances)
        append(index, 1, posterior)
        events.append(dict(acquisition_index=index, time=float(upper), log_predictive_evidence=float(evidence),
            process_noise_application_count=index, likelihood_applications=1,
            process_sigma_rad_s=process_sigma, measurement_sigma_deg_s=sigma))
        diagnostics.append(dict(acquisition_index=index, time=float(upper),
            **prior_rule['diagnostics'], posterior_parent_proposal=placement,
            posterior_parent_log_prior_mass_quadrature=posterior_rule['diagnostics']['log_density_normalizer'],
            posterior_parent_proposal_mean_rad_s=proposal_mean.tolist(),
            posterior_parent_proposal_covariance_rad2_s2=proposal_covariance.tolist()))
        mass, mixture_means, mixture_covariance = updated_mass, updated_means, updated_covariance
        moments, rotation_covariances = propagated_moments, propagated_covariances
        total_steps += flow.total_steps
        maximum_norm = max(maximum_norm, flow.maximum_norm_correction)
        if int(upper-stamps[0]) >= last_progress+2:
            last_progress = int(upper-stamps[0])
            print(f'Process sigma {process_sigma:g} {mode} GH {order}: '
                  f't={upper:g}s, wall={time.monotonic()-begun:.1f}s', flush=True)
    final_rule = regrid_joint_mixture(mass, mixture_means, mixture_covariance,
                                      moments, rotation_covariances, order)
    arrays = {key: np.stack(value) for key, value in output.items()}
    arrays['rpy_proxy_rad'] = np.unwrap(arrays['rpy_proxy_rad'], axis=0)
    arrays.update(measurement_covariance_rad2_s2=measurement_covariance,
        process_covariance_rad2_s2=process_covariance,
        process_noise_reference_interval_seconds=np.asarray(np.median(np.diff(stamps))),
        process_noise_application_count=np.asarray(len(stamps)-1),
        event_time=stamps.copy(),
        final_omega_nodes_rad_s=final_rule['nodes'], final_quadrature_weights=final_rule['weights'],
        final_conditional_quaternion_second_moment=final_rule['quaternion_moments'],
        final_conditional_rotation_covariance_colmajor=final_rule['rotation_covariances'],
        final_posterior_component_weights=updated_mass,
        final_posterior_component_omega_means_rad_s=updated_means,
        final_posterior_common_omega_covariance_rad2_s2=updated_covariance,
        final_posterior_conditional_quaternion_second_moment=propagated_moments,
        final_posterior_conditional_rotation_covariance_colmajor=propagated_covariances,
        numerical_diagnostics=diagnostics, events=events, final_quadrature_diagnostic=final_rule['diagnostics'],
        metadata=dict(schema='grape/conditional-orientation-process-noise/v1', mode=mode,
            process_sigma_rad_s=process_sigma, process_sigma_deg_s=float(np.rad2deg(process_sigma)),
            process_scope='independent isotropic body-omega increment at each selected interval END; q unchanged instantaneously',
            process_noise_is_continuous_time_psd=False, process_noise_per_rk4_substep=False,
            process_noise_application_count=len(stamps)-1, measurement_count=len(stamps),
            quadrature_order_per_axis=int(order), quadrature_node_count=int(order)**3,
            posterior='Markov transition plus gyro Bayes update at each acquired time; all past information retained recursively',
            conditional_statistics='E[qqT|omega], E[R|omega] and full Cov(vecR|omega); all rotation/omega dependence retained in omega-conditioned functions',
            observation_update='analytic Gaussian component conditioning; mixture weights normalized by exact component evidence',
            numerical_approximation='deterministic Gaussian-importance Nystrom quadrature in current omega only',
            finite_mixture_tail_resolution_is_not_guaranteed=True,
            numerical_tail_scope='A finite convolution-mixture quadrature can under-resolve far tails after large or sign-reversing innovations; order refinement and independent linear-Kalman differences must be reported',
            analytic_linear_gaussian_limit_has_finite_quadrature_error=True,
            finite_numerical_errors_do_not_reject_observations_or_stop_updates=True,
            retained_state='continuous Gaussian mixture in omega with conditional quaternion/rotation moments; not a Dirac omega-node law',
            observation_informed_parent_proposal=True,
            gaussian_proposal_is_not_posterior=True, independent_q_omega_assumption=False,
            earlier_online_outputs_rewritten=False, initial_reading_used_once=True,
            monte_carlo=False, covariance_floor=False, node_pruning=False, process_samples_drawn=False,
            maximum_quaternion_norm_correction=maximum_norm, native_rk4_steps=total_steps,
            included_physical_source_sha256=hashlib.sha256((HERE/'native_flow.cpp').read_bytes()).hexdigest(),
            engine_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            wall_seconds=time.monotonic()-begun))
    return arrays
