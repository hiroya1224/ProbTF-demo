"""Exactly scaled body rates and a consistent integrated attitude trajectory.

Only the prescribed physical motion is scaled. Sensor noise, bias, clocks,
inertia and observation covariance remain the original experiment's values.
The Euler angles are obtained from the new quaternion path: dividing the old
Euler angles would not divide the coupled body angular velocity exactly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from trajectory import trajectory, generate_dataset, INERTIA

DEFAULT_ANGULAR_RATE_SCALE = 1./75.
RTOL = 1e-11
ATOL = 1e-13


def _scale(value):
    scale = float(value)
    if not np.isfinite(scale) or scale < 0.:
        raise ValueError('Angular-rate scale must be finite and nonnegative')
    return scale


@lru_cache(maxsize=8)
def _quaternion_solution(end_time, scale):
    q0_xyzw = trajectory(0., 'three_frequency')['quaternion_xyzw']
    q0 = q0_xyzw[[3, 0, 1, 2]]

    def derivative(stamp, quaternion):
        omega = trajectory(stamp, 'three_frequency')['omega_body_rad_s']*scale
        scalar, vector = quaternion[0], quaternion[1:]
        return .5*np.r_[-vector@omega, scalar*omega+np.cross(vector, omega)]

    solution = solve_ivp(derivative, (0., end_time), q0, method='DOP853',
                         dense_output=True, rtol=RTOL, atol=ATOL)
    if not solution.success or solution.sol is None or not np.isfinite(solution.y).all():
        raise FloatingPointError('Quaternion trajectory integration failed: '+solution.message)
    return solution


def _euler_derivatives(rpy, omega, alpha):
    roll, pitch = rpy[:, 0], rpy[:, 1]
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    mapping = np.zeros((len(rpy), 3, 3))
    mapping[:, 0, 0] = 1.
    mapping[:, 0, 2] = -sp
    mapping[:, 1, 1], mapping[:, 1, 2] = cr, sr*cp
    mapping[:, 2, 1], mapping[:, 2, 2] = -sr, cr*cp
    velocity = np.linalg.solve(mapping, omega)
    d_roll, d_pitch = np.zeros_like(mapping), np.zeros_like(mapping)
    d_roll[:, 1, 1], d_roll[:, 1, 2] = -sr, cr*cp
    d_roll[:, 2, 1], d_roll[:, 2, 2] = -cr, -sr*cp
    d_pitch[:, 0, 2] = -cp
    d_pitch[:, 1, 2], d_pitch[:, 2, 2] = -sr*sp, -cr*sp
    derivative = (d_roll*velocity[:, 0, None, None]
                  +d_pitch*velocity[:, 1, None, None])
    acceleration = np.linalg.solve(mapping, alpha-np.einsum('nij,nj->ni', derivative, velocity))
    return velocity, acceleration


def scaled_trajectory(time, angular_rate_scale=DEFAULT_ANGULAR_RATE_SCALE):
    """Evaluate omega_new(t)=scale*omega_old(t), with its consistent pose.

    Old means the fixed three-frequency profile at exactly the same time.
    Alpha scales linearly; omega cross Jomega scales quadratically. The
    starting pose is unchanged. Quaternion integration is deterministic
    DOP853 with finite numerical error, not a closed-form attitude solution.
    """
    stamps = np.asarray(time, float)
    if not np.isfinite(stamps).all() or np.any(stamps < 0.):
        raise ValueError('Scaled motion times must be finite and nonnegative')
    scale = _scale(angular_rate_scale)
    flat = stamps.reshape(-1)
    source = trajectory(flat, 'three_frequency')
    omega = source['omega_body_rad_s']*scale
    alpha = source['alpha_body_rad_s2']*scale
    # A fixed common support lets independently evaluated references use the
    # very same dense solution throughout the default five-second interval.
    horizon = max(5., float(np.max(flat)) if len(flat) else 5.)
    solution = _quaternion_solution(horizon, scale)
    quaternion_wxyz = solution.sol(flat).T
    norms = np.linalg.norm(quaternion_wxyz, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.):
        raise FloatingPointError('Invalid integrated quaternion')
    quaternion_wxyz = quaternion_wxyz/norms[:, None]
    quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
    quaternion_xyzw[flat == 0.] = trajectory(0., 'three_frequency')['quaternion_xyzw']
    rpy = Rotation.from_quat(quaternion_xyzw).as_euler('xyz')
    rpy_velocity, rpy_acceleration = _euler_derivatives(rpy, omega, alpha)
    gyroscopic = np.cross(omega, omega@INERTIA.T)
    torque = alpha@INERTIA.T+gyroscopic

    def restore(value):
        return value.reshape(stamps.shape+(value.shape[-1],))

    return dict(time=stamps.copy(), quaternion_xyzw=restore(quaternion_xyzw),
        rpy_rad=restore(rpy), rpy_velocity_rad_s=restore(rpy_velocity),
        rpy_acceleration_rad_s2=restore(rpy_acceleration), omega_body_rad_s=restore(omega),
        alpha_body_rad_s2=restore(alpha), torque_body_nm=restore(torque),
        gyroscopic_torque_nm=restore(gyroscopic))


def generate_scaled_dataset(angular_rate_scale=DEFAULT_ANGULAR_RATE_SCALE,
                            duration=5., sensor_rate=200., seed=20261113):
    """Return a scaled physical motion with the unchanged sensor realization."""
    scale = _scale(angular_rate_scale)
    data, original_metadata = generate_dataset(duration, sensor_rate, seed,
                                               motion_profile='three_frequency')
    native = scaled_trajectory(data['imu_time'], scale)
    reference = scaled_trajectory(data['reference_time'], scale)
    control = scaled_trajectory(data['torque_time'], scale)
    data['angular_rate_scale'] = np.asarray(scale)
    for suffix in ('amplitude_deg', 'frequency_hz', 'phase_rad'):
        data['source_motion_'+suffix] = data.pop('motion_'+suffix)
    data['gyro_true_body_rad_s'] = native['omega_body_rad_s']
    data['gyro'] = native['omega_body_rad_s']+data['gyro_bias_rad_s']+data['gyro_realized_noise_rad_s']
    data['reference_quaternion_xyzw'] = reference['quaternion_xyzw']
    data['reference_rpy_rad'] = reference['rpy_rad']
    data['reference_rpy_velocity_rad_s'] = reference['rpy_velocity_rad_s']
    data['reference_rpy_acceleration_rad_s2'] = reference['rpy_acceleration_rad_s2']
    data['reference_omega_body_rad_s'] = reference['omega_body_rad_s']
    data['reference_alpha_body_rad_s2'] = reference['alpha_body_rad_s2']
    data['reference_gyroscopic_torque_body_nm'] = reference['gyroscopic_torque_nm']
    data['torque_body_nm'] = control['torque_body_nm']
    metadata = dict(original_metadata)
    coefficients = {key: metadata.pop(key) for key in
                    ('rpy_amplitude_deg', 'rpy_frequency_hz', 'rpy_phase_rad')}
    metadata.update(schema='grape/synthetic-rate-scaled-controlled-rotation-input/v1',
        angular_rate_scale=scale,
        source_motion=dict(profile='three_frequency', **coefficients),
        motion='body omega(t) and alpha(t) scaled from the saved three-frequency profile at the same time; quaternion path reintegrated from unchanged initial pose',
        kinematics='omega_new=scale*omega_old and alpha_new=scale*alpha_old exactly in body coordinates; Euler rates/accelerations derived from the new pose and body kinematics',
        source_coefficients_are_new_euler_amplitudes=False,
        rpy_path='obtained from integrated quaternion; not original Euler angles divided by the scale factor',
        quaternion_integration=dict(method='scipy.solve_ivp DOP853, dense output',
            rtol=RTOL, atol=ATOL, finite_numerical_integration=True,
            normalization='remove finite quaternion norm drift only; no state covariance adjustment'),
        scaling_scope='physical motion only; sensor noise, bias, native clocks, inertia, seed and assumed reconstruction sigma unchanged',
        linear_inertial_torque_scale=scale, gyroscopic_torque_scale=scale*scale)
    return data, metadata
