"""Two deterministic uncertain replays from an IMU-only physical fit.

The model trajectory has no measured-attitude reset and no fitted disturbance
mean.  Its rectangular factors propagate all identified quotient/log-lag modes.
The IMU trajectory integrates the measured specific force using the separate
Bingham output; temporal error correlations are bounded, not discarded.
Only the explicitly supplied initial conditions can originate from mocap.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent, HERE.parent.parent / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from grape_param_estim.dynamics import (FullSixDofPlant,
    _normalise_quaternion_with_jacobian, _quaternion_right_tangent_lift,
    _quaternion_right_tangent_projection)
from grape_param_estim.geometry import quaternion_to_matrix, skew
from grape_param_estim.system import ActuatorState, GRAVITY
from imu_model import first_order_zoh_thrust_history


def _factor(covariance):
    """PSD square root without a relative cutoff, jitter or clipping."""
    matrix = np.asarray(covariance, float)
    if np.array_equal(matrix, np.zeros_like(matrix)):
        return np.zeros_like(matrix)
    if np.array_equal(matrix, np.diag(np.diag(matrix))):
        diagonal = np.diag(matrix)
        if np.any(diagonal < 0):
            raise ValueError('negative covariance diagonal')
        return np.diag(np.sqrt(diagonal))
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(.5 * (matrix + matrix.T))
        if np.any(values < 0):
            raise ValueError('a supplied covariance has a negative eigenvalue')
        return vectors * np.sqrt(values)[None, :]


def _gram(factor):
    return factor @ factor.T


def _signal_covariances(data, fit):
    """Fixed-lag sensor covariance, keeping all local derivative cross terms."""
    problem = fit.get('problem')
    if problem is not None:
        signals = (problem.gyro, problem.accel)
    else:
        from imu_signal import local_polynomial_signal
        options = fit.get('estimate', {}).get('method', {}).get('options', {})
        signals = tuple(local_polynomial_signal(data['time'], data[key], data[noise],
            window_seconds=float(options.get('sg_window_seconds', 1.)),
            degree=int(options.get('sg_degree', 5)), max_derivative=2)
            for key, noise in (('gyro', 'gyro_noise_covariance'),
                               ('accel', 'accel_noise_covariance')))
    cg, ca = (np.asarray(value['covariances']) for value in signals)
    tg, ta = np.asarray(fit['taus'])[1:]
    omega = cg[0, 0] + tg * (cg[0, 1] + cg[1, 0]) + tg**2 * cg[1, 1]
    alpha = cg[1, 1] + tg * (cg[1, 2] + cg[2, 1]) + tg**2 * cg[2, 2]
    cross = cg[0, 1] + tg * (cg[0, 2] + cg[1, 1]) + tg**2 * cg[1, 2]
    force = ca[0, 0] + ta * (ca[0, 1] + ca[1, 0]) + ta**2 * ca[1, 1]
    return {'omega': omega, 'alpha': alpha, 'omega_alpha': cross, 'force': force}


def _initial(data, fit, orientation, noise):
    pars = fit['parameters']
    arrays = fit['arrays']
    q_wxyz = np.asarray(data['initial_quaternion_wxyz'], float)
    q, _ = _normalise_quaternion_with_jacobian(q_wxyz[[1, 2, 3, 0]])
    rotation = quaternion_to_matrix(q)
    lever = np.asarray(data['gyro_sensor_position_in_body']) - pars.cog_offset
    omega = np.asarray(arrays['corrected_gyro'])[0]
    x = np.r_[np.asarray(data['initial_position']) - rotation @ lever, q,
        np.asarray(data['initial_velocity']) - rotation @ np.cross(omega, lever), omega]
    _, pj = fit['chart'].decode_with_jacobian(fit['physical_coordinate'])
    basis = np.asarray(fit['joint_basis'])[:14, :13]
    dr = np.zeros((3, 16)); dr[:, :13] = -pj.cog_offset @ basis
    domega = np.zeros((3, 16))
    domega[:, 14] = fit['taus'][1] * arrays['gyro_derivative'][0]
    sensitivity = np.zeros((13, 16))
    sensitivity[:3] = -rotation @ dr
    sensitivity[7:10] = -rotation @ (skew(omega) @ dr - skew(lever) @ domega)
    sensitivity[10:13] = domega
    initial = np.zeros((13, 12))
    initial[:3, :3] = np.eye(3)
    initial[:3, 3:6] = rotation @ skew(lever)
    initial[3:7, 3:6] = _quaternion_right_tangent_lift(q)
    initial[7:10, 6:9] = np.eye(3)
    initial[7:10, 3:6] = rotation @ skew(np.cross(omega, lever))
    initial[7:10, 9:12] = rotation @ skew(lever)
    initial[10:13, 9:12] = np.eye(3)
    initial_factor = np.zeros((12, 9))
    initial_factor[3:6, :3] = _factor(orientation['covariance_output'][0])
    initial_factor[6:9, 3:6] = _factor(data['initial_velocity_covariance'])
    initial_factor[9:12, 6:9] = _factor(noise['omega'][0])
    return x, sensitivity, initial, initial_factor, lever, dr, pj, basis


def _fc_output(x, sensitivity, initial, wrench, lever, dr, factors):
    rotation = quaternion_to_matrix(x[3:7]); omega = x[10:13]
    projection = _quaternion_right_tangent_projection(x[3:7])
    output = np.zeros((12, 13))
    output[:3, :3] = np.eye(3)
    output[:3, 3:7] = -rotation @ skew(lever) @ projection
    output[3:6, 3:7] = projection
    output[6:9, 7:10] = np.eye(3)
    output[6:9, 3:7] = -rotation @ skew(np.cross(omega, lever)) @ projection
    output[6:9, 10:13] = -rotation @ skew(lever)
    output[9:12, 10:13] = np.eye(3)
    direct = np.zeros((12, 16))
    direct[:3] = rotation @ dr
    direct[6:9] = rotation @ skew(omega) @ dr
    parameter_factor, initial_factor, wrench_factor = factors
    factor = np.sqrt(3.) * np.concatenate(((output @ sensitivity + direct) @ parameter_factor,
        output @ initial @ initial_factor, output @ wrench @ wrench_factor), axis=1)
    return (x[:3] + rotation @ lever,
        x[7:10] + rotation @ np.cross(omega, lever), factor)


def _model_replay(data, fit, orientation, noise, options):
    times = np.asarray(data['time'], float)
    pars, geometry = fit['parameters'], fit['model'].geometry
    plant = FullSixDofPlant(pars, geometry)
    x, s, initial, initial_factor, lever, dr, pj, basis = _initial(data, fit, orientation, noise)
    residual = np.asarray(fit['arrays']['raw_residual_wrench'])
    # A persistent, zero-mean body-wrench error: uncentred moment preserves
    # systematic mismatch as uncertainty; its sample mean never drives x.
    wrench_factor = np.linalg.qr(residual / np.sqrt(len(residual)), mode='reduced')[1].T
    param_factor = np.asarray(fit['parameter_covariance_factor_reduced'])
    factors = (param_factor, initial_factor, wrench_factor)
    u = np.zeros((13, 6))
    dm = pj.mass @ basis
    d_inertia = np.einsum('abp,pq->abq', pj.inertia, basis)
    maximum_step = float(options.get('integration_max_step_seconds', .005))
    if not np.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError('integration_max_step_seconds must be positive and finite')
    subdivisions = [max(1, int(np.ceil(dt / maximum_step))) for dt in np.diff(times)]
    nodes = [times[0]]
    native_indices = [0]
    for t0, t1, count in zip(times[:-1], times[1:], subdivisions):
        nodes.extend(np.linspace(t0, t1, count + 1)[1:])
        native_indices.append(len(nodes) - 1)
    nodes = np.asarray(nodes)
    stages = np.unique(np.r_[nodes, .5 * (nodes[:-1] + nodes[1:])])
    fit_options = getattr(fit.get('problem'), 'options',
        fit.get('estimate', {}).get('method', {}).get('options', {}))
    history = first_order_zoh_thrust_history(command_times=data['rotor_command_time'],
        command_values=data['rotor_command'], evaluation_times=stages,
        time_constant=float(fit['taus'][0]),
        minimum_thrust=float(fit_options.get('minimum_thrust', 1.5)),
        maximum_thrust=float(fit_options.get('maximum_thrust', 27.6145)))
    gimbal = np.column_stack([np.interp(stages, data['gimbal_time'], np.asarray(data['gimbal_angle'])[:, i])
                             for i in range(4)])
    if times[0] < np.asarray(data['gimbal_time'])[0] or times[-1] > np.asarray(data['gimbal_time'])[-1]:
        raise ValueError('model replay exceeds measured gimbal support')
    stage_bundles = []
    for thrust, angles, dthrust in zip(history.thrust, gimbal, history.log_tau_jacobian):
        actuator = ActuatorState(thrust, angles)
        evaluation = plant.actuator_wrench_evaluation(actuator)
        jac = evaluation.jacobian
        physical_wrench = (jac.cog_offset @ pj.cog_offset
            + jac.force_effectiveness @ pj.force_effectiveness) @ basis
        stage_bundles.append((actuator, evaluation, physical_wrench, dthrust))

    def rhs(stamp, value):
        state, ps, ix, wx = value
        stage = int(np.searchsorted(stages, stamp))
        actuator, evaluation, d_wrench, d_thrust = stage_bundles[stage]
        dx, a, thrust_j, _ = plant.derivative_with_jacobian(stamp, state, actuator,
            precomputed_actuator_wrench=evaluation)
        # The shared plant returns a canonical (w >= 0) quaternion rate.
        # An RK4 *intermediate* raw stage can be in the opposite hemisphere;
        # transport its rate/Jacobian back to that stage's ambient sign.  The
        # full step is canonicalised only after combining its four stages.
        # R and all physical derivatives are sign invariant.  This adapter
        # changes no shared plant/core code or physical model.
        if state[6] < 0.:
            dx[3:7] *= -1.
            a[3:7] *= -1.
            thrust_j[3:7] *= -1.
        rot = quaternion_to_matrix(state[3:7]); omega = state[10:13]
        total_force = evaluation.body_wrench[:3] - pars.linear_drag * (rot.T @ state[7:10])
        b = np.zeros((13, 16))
        b[7:10, :13] = rot @ (d_wrench[:3] / pars.mass
            - total_force[:, None] * dm[None, :] / pars.mass**2)
        inertia_alpha = np.einsum('abp,b->ap', d_inertia, dx[10:13])
        inertia_omega = np.einsum('abp,b->ap', d_inertia, omega)
        b[10:13, :13] = np.linalg.solve(pars.inertia,
            d_wrench[3:] - inertia_alpha - skew(omega) @ inertia_omega)
        b[:, 13] = thrust_j @ d_thrust
        l = np.zeros((13, 6)); l[7:10, :3] = rot / pars.mass
        l[10:13, 3:] = np.linalg.solve(pars.inertia, np.eye(3))
        return dx, a @ ps + b, a @ ix, a @ wx + l

    def rk4(t0, t1, value):
        dt = t1 - t0; mid = .5 * (t0 + t1)
        k1 = rhs(t0, value)
        k2 = rhs(mid, tuple(v + .5 * dt * k for v, k in zip(value, k1)))
        k3 = rhs(mid, tuple(v + .5 * dt * k for v, k in zip(value, k2)))
        k4 = rhs(t1, tuple(v + dt * k for v, k in zip(value, k3)))
        result = tuple(v + dt / 6. * (a + 2*b + 2*c + d)
            for v, a, b, c, d in zip(value, k1, k2, k3, k4))
        q, nq = _normalise_quaternion_with_jacobian(result[0][3:7])
        result[0][3:7] = q
        for derivative in result[1:]:
            derivative[3:7] = nq @ derivative[3:7]
        return result

    n = len(times); width = sum(value.shape[1] for value in factors)
    out = {'time': times.copy(), 'position_fc_world': np.empty((n, 3)),
        'quaternion_wxyz': np.empty((n, 4)), 'velocity_fc_world': np.empty((n, 3)),
        'angular_velocity_body': np.empty((n, 3)), 'position_cog_world': np.empty((n, 3)),
        'velocity_cog_world': np.empty((n, 3)), 'covariance_state': np.empty((n, 12, 12)),
        'state_covariance_factor': np.empty((n, 12, width)),
        'predicted_gyro_body': np.empty((n, 3)), 'predicted_specific_force_body': np.empty((n, 3))}
    value = (x, s, initial, u)
    next_native = 0
    for j, stamp in enumerate(nodes):
        if j:
            value = rk4(nodes[j-1], stamp, value)
        if j != native_indices[next_native]:
            continue
        x, s, initial, u = value
        k = next_native
        p, v, factor = _fc_output(x, s, initial, u, lever, dr, factors)
        out['position_fc_world'][k], out['velocity_fc_world'][k] = p, v
        out['quaternion_wxyz'][k] = x[[6, 3, 4, 5]]
        out['angular_velocity_body'][k] = out['predicted_gyro_body'][k] = x[10:13]
        out['position_cog_world'][k], out['velocity_cog_world'][k] = x[:3], x[7:10]
        out['state_covariance_factor'][k] = factor
        out['covariance_state'][k] = _gram(factor)
        dx = rhs(stamp, value)[0]
        rot = quaternion_to_matrix(x[3:7])
        out['predicted_specific_force_body'][k] = (rot.T @ (dx[7:10] + np.array([0., 0., GRAVITY]))
            + np.cross(dx[10:13], lever) + np.cross(x[10:13], np.cross(x[10:13], lever)))
        next_native += 1
        if next_native == n:
            break
    out['covariance_position_fc'] = out['covariance_state'][:, :3, :3]
    out['covariance_rotation'] = out['covariance_state'][:, 3:6, 3:6]
    out['position_covariance_factor_fc'] = out['state_covariance_factor'][:, :3]
    out['rotation_covariance_factor'] = out['state_covariance_factor'][:, 3:6]
    out['persistent_wrench_second_moment'] = _gram(wrench_factor)
    return out


def _integrate_linear_acceleration(times, acceleration, p0, v0):
    position = np.empty_like(acceleration); velocity = np.empty_like(acceleration)
    position[0], velocity[0] = p0, v0
    for k, dt in enumerate(np.diff(times)):
        position[k+1] = (position[k] + dt * velocity[k]
            + dt**2 * (acceleration[k] / 3. + acceleration[k+1] / 6.))
        velocity[k+1] = velocity[k] + .5 * dt * (acceleration[k] + acceleration[k+1])
    return position, velocity


def _imu_replay(data, fit, orientation, noise):
    times = np.asarray(data['time']); n = len(times)
    arrays = fit['arrays']; pars = fit['parameters']
    q = np.asarray(orientation['q_output_wxyz'])
    rot = np.asarray([quaternion_to_matrix(value[[1, 2, 3, 0]]) for value in q])
    omega = np.asarray(arrays['corrected_gyro'])
    alpha = np.asarray(arrays['corrected_angular_acceleration'])
    force = np.asarray(arrays['corrected_specific_force'])
    lever = np.asarray(data['gyro_sensor_position_in_body']) - pars.cog_offset
    body_cog_force = force - np.cross(alpha, lever) - np.cross(omega, np.cross(omega, lever))
    gravity = np.array([0., 0., -GRAVITY])
    acceleration_cog = np.einsum('nij,nj->ni', rot, body_cog_force) + gravity
    acceleration_fc = np.einsum('nij,nj->ni', rot, force) + gravity
    p0 = np.asarray(data['initial_position']); v0 = np.asarray(data['initial_velocity'])
    pc0 = p0 - rot[0] @ lever
    vc0 = v0 - rot[0] @ np.cross(omega[0], lever)
    pc, vc = _integrate_linear_acceleration(times, acceleration_cog, pc0, vc0)
    lever_world = np.einsum('nij,j->ni', rot, lever)
    p = pc + lever_world
    v = vc + np.einsum('nij,nj->ni', rot, np.cross(omega, lever))
    p_direct, v_direct = _integrate_linear_acceleration(times, acceleration_fc, p0, v0)
    _, pj = fit['chart'].decode_with_jacobian(fit['physical_coordinate'])
    dr = np.zeros((3, 16)); dr[:, :13] = -pj.cog_offset @ np.asarray(fit['joint_basis'])[:14, :13]
    lp = np.asarray(fit['parameter_covariance_factor_reduced'])
    covariance_accel = np.empty((n, 3, 3)); accel_sigma = np.empty(n)
    rotation_cov = np.asarray(orientation['covariance_output'])
    for k in range(n):
        j_omega = skew(np.cross(omega[k], lever)) + skew(omega[k]) @ skew(lever)
        j_alpha = skew(lever)
        gyro_cov = (j_omega @ noise['omega'][k] @ j_omega.T
            + j_alpha @ noise['alpha'][k] @ j_alpha.T
            + j_omega @ noise['omega_alpha'][k] @ j_alpha.T
            + j_alpha @ noise['omega_alpha'][k].T @ j_omega.T)
        # Accel and gyro raw-noise correlation is unknown: 2-sum bound.
        measurement = 2. * (noise['force'][k] + gyro_cov)
        j_lever = -skew(alpha[k]) - skew(omega[k]) @ skew(omega[k])
        param = j_lever @ dr
        param[:, 14] += (j_omega @ (fit['taus'][1] * arrays['gyro_derivative'][k])
            + j_alpha @ (fit['taus'][1] * arrays['gyro_second_derivative'][k]))
        param[:, 15] += fit['taus'][2] * arrays['accel_derivative'][k]
        param_factor = rot[k] @ param @ lp
        jrot = -rot[k] @ skew(body_cog_force[k])
        # Data are shared across fit, orientation and force inversion. This
        # matrix Cauchy bound permits arbitrary cross-correlation of all 3.
        covariance_accel[k] = 3. * (rot[k] @ measurement @ rot[k].T
            + jrot @ rotation_cov[k] @ jrot.T + _gram(param_factor))
        accel_sigma[k] = np.sqrt(np.linalg.norm(covariance_accel[k], ord=2))
    elapsed = times - times[0]
    # Minkowski bounds keep arbitrary temporal correlations, including
    # overlapping SG windows, rather than accumulating independent noise.
    integrated_p, integrated_v = _integrate_linear_acceleration(times,
        np.repeat(accel_sigma[:, None], 3, axis=1), np.zeros(3), np.zeros(3))
    sigma_p_acc, sigma_v_acc = integrated_p[:, 0], integrated_v[:, 0]
    sigma_v0 = np.linalg.norm(_factor(data['initial_velocity_covariance']), ord=2)
    omega0_factor = _factor(noise['omega'][0])
    r_factor = dr @ lp
    # Unknown-time orientation covariance plus lever/model uncertainty.
    rot_sigma = np.sqrt(np.linalg.norm(rotation_cov, ord=2, axis=(1, 2)))
    lever_parameter_sigma = np.linalg.norm(r_factor, ord=2)
    offset_sigma = np.linalg.norm(lever) * (rot_sigma + rot_sigma[0]) + 2. * lever_parameter_sigma
    domega0 = np.zeros((3, 16))
    domega0[:, 14] = fit['taus'][1] * arrays['gyro_derivative'][0]
    velocity_lever_param0 = skew(omega[0]) @ dr - skew(lever) @ domega0
    vc_offset_sigma = (np.linalg.norm(np.cross(omega[0], lever)) * rot_sigma[0]
        + np.linalg.norm(skew(lever) @ omega0_factor, ord=2)
        + np.linalg.norm(velocity_lever_param0 @ lp, ord=2))
    sigma_p = sigma_p_acc + elapsed * (sigma_v0 + vc_offset_sigma) + offset_sigma
    sigma_v = sigma_v_acc + sigma_v0 + vc_offset_sigma
    for k in range(n):
        domega = np.zeros((3, 16))
        domega[:, 14] = fit['taus'][1] * arrays['gyro_derivative'][k]
        velocity_lever_param = skew(omega[k]) @ dr - skew(lever) @ domega
        sigma_v[k] += (np.linalg.norm(np.cross(omega[k], lever)) * rot_sigma[k]
            + np.linalg.norm(skew(lever) @ _factor(noise['omega'][k]), ord=2)
            + np.linalg.norm(velocity_lever_param @ lp, ord=2))
    # The same initial variable appears with opposite signs at t0: cancel it
    # exactly, before applying separate-time conservative bounds at t>0.
    sigma_p[0] = 0.; sigma_v[0] = sigma_v0
    factor = np.zeros((n, 12, 12))
    for k in range(n):
        factor[k, :3, :3] = np.eye(3) * sigma_p[k]
        factor[k, 3:6, 3:6] = _factor(rotation_cov[k])
        factor[k, 6:9, 6:9] = np.eye(3) * sigma_v[k]
        factor[k, 9:12, 9:12] = _factor(orientation['corrected_gyro_covariance'][k])
    # Unknown cross between the four output blocks: 4-sum matrix bound.
    factor *= 2.
    covariance = factor @ np.swapaxes(factor, 1, 2)
    return {'time': times.copy(), 'position_fc_world': p, 'quaternion_wxyz': q.copy(),
        'velocity_fc_world': v, 'angular_velocity_body': omega.copy(),
        'position_cog_world': pc, 'velocity_cog_world': vc,
        'direct_sensor_position_world': p_direct, 'direct_sensor_velocity_world': v_direct,
        'world_acceleration_cog': acceleration_cog, 'world_acceleration_fc': acceleration_fc,
        'world_acceleration_covariance_bound': covariance_accel,
        'covariance_state': covariance, 'state_covariance_factor': factor,
        'covariance_position_fc': covariance[:, :3, :3], 'covariance_rotation': covariance[:, 3:6, 3:6],
        'position_covariance_factor_fc': factor[:, :3], 'rotation_covariance_factor': factor[:, 3:6],
        'predicted_gyro_body': omega.copy(), 'predicted_specific_force_body': force.copy()}


def replay(data, fit_result, orientation_result, options=None):
    """Replay native IMU times without consulting any ongoing mocap arrays."""
    times = np.asarray(data['time'])
    if np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('replay requires finite, strictly increasing native timestamps')
    for result in (fit_result['arrays'], orientation_result):
        if not np.array_equal(times, result['time']):
            raise ValueError('fit/orientation replay clocks differ from the raw IMU clock')
    noise = _signal_covariances(data, fit_result)
    model = _model_replay(data, fit_result, orientation_result, noise, options or {})
    imu = _imu_replay(data, fit_result, orientation_result, noise)
    metadata = {'schema': 'grape-param-estim/imu-first-order-lag-replay/v1',
        'ongoing_mocap_used': False, 'monte_carlo_used': False,
        'initial_condition_frame': 'FC nominal sensor-origin position/velocity in world; body-to-world wxyz quaternion',
        'state_covariance_order': ['position_FC_world', 'right_rotation_body', 'velocity_FC_world', 'angular_velocity_body'],
        'model_integration': 'canonical full-six-DoF ambient-13 RK4 with exact thrust-history values at all stages; intermediate quaternion-rate/Jacobian sign transported to raw stage hemisphere, full-step normalization differentiated',
        'model_parameter_uncertainty': 'all 16 quotient/log-tau modes propagated as rectangular Gram factor; first-order sensitivity approximation',
        'model_discrepancy_mean': [0.] * 6,
        'model_discrepancy_uncertainty': 'persistent whole-window zero-mean body wrench, uncentred second moment of IMU-fit residual wrench',
        'model_unknown_cross_bound': '3*(parameter + initial-state + persistent-wrench propagated covariance)',
        'initial_state_source_cross_assumption': 'initial rotation, past-prefix velocity and fixed-tau gyro noise blocks independent; parameter/source cross then bounded by factor 3',
        'imu_mean': 'lever-corrected CoG specific force integrated piecewise-linearly; FC reconstructed with Bingham output attitude; direct sensor-origin integration also saved',
        'imu_acceleration_bound': '3*(2*(accel + joint gyro/alpha local noise) + orientation + full-parameter); unknown-cross matrix Cauchy bounds',
        'imu_temporal_bound': 'spectral-norm isotropic majorant with Minkowski integration, permitting arbitrary SG-overlap/time correlations',
        'imu_state_bound': '4 times block-diagonal marginal majorants permits arbitrary output-block correlations; not an exact posterior',
        'imu_initial_offset_bound': 'initial FC lever cancellation exact at t0; later-time unknown-correlation bound can be discontinuous at t0',
        'predicted_sensor_columns': 'underlying physical gyro/specific-force inputs before forward sensor lag',
        'sensor_extrinsic_assumption': 'known nominal FC frame/position, not independently calibrated physical chip lever',
        'integration_max_step_seconds': float((options or {}).get('integration_max_step_seconds', .005))}
    return {'model': model, 'imu_dead_reckoning': imu, 'metadata': metadata}


def save_replay(result, directory):
    """Store small deterministic replay arrays separately from historical data."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    arrays = {f'{branch}__{key}': value for branch in ('model', 'imu_dead_reckoning')
              for key, value in result[branch].items()}
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    (directory / 'metadata.json').write_text(json.dumps(result['metadata'], indent=2) + '\n')
