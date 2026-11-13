"""Smooth synthetic rotation, its exact kinematics, and noisy gyro generation.

The prescribed RPY curve is a kinematic test motion, not a full aerodynamic
flight simulation.  The known body-CoG torque is obtained once by inverse
Euler dynamics during generation.  Downstream reconstruction receives that
saved input and sparse noisy gyro readings, not the subsequent truth poses.
Random draws are confined to sensor-data generation, never state inference.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


INERTIA = np.diag([.045, .070, .100])  # kg m^2, three distinct principal moments
GYRO_NOISE_SIGMA_DEG_S = .2
GYRO_BIAS_DEG_S = np.array([.03, -.02, .04])
# Rows correspond to roll, pitch, yaw; columns are two smooth sine terms.
AMPLITUDE_DEG = np.array([[35., 6.], [25., 5.], [50., 10.]])
FREQUENCY_HZ = np.array([[.42, .13], [.31, .19], [.37, .11]])
PHASE_RAD = np.array([[.2, 0.], [-.4, .6], [.5, -.2]])
TORQUE_RATE_HZ = 1000.

MOTION_PROFILES = {
    'two_frequency': dict(amplitude_deg=AMPLITUDE_DEG,
                          frequency_hz=FREQUENCY_HZ, phase_rad=PHASE_RAD),
    # Three substantial, distinct frequencies per axis, fixed before replay.
    'three_frequency': dict(
        amplitude_deg=np.array([[23., 15., 9.], [17., 10., 7.], [32., 21., 12.]]),
        frequency_hz=np.array([[.29, .73, 1.17], [.37, .83, 1.31], [.23, .61, 1.03]]),
        phase_rad=np.array([[.2, 1.1, -.7], [-.4, .8, 1.7], [.5, -1.2, .4]])),
}


def _motion_parameters(motion_profile):
    try:
        return MOTION_PROFILES[motion_profile]
    except (KeyError, TypeError):
        raise ValueError('Unknown motion profile: '+str(motion_profile)) from None


def trajectory(time, motion_profile='two_frequency'):
    """Evaluate analytic R=Rz(yaw) Ry(pitch) Rx(roll) and body derivatives.

    Input may be a scalar or an array.  Every vector-valued result has shape
    ``np.shape(time)+(dimension,)``.  Quaternion order is xyzw, Hamilton;
    its rotation maps body vectors to world vectors.  Omega is expressed in
    the body frame and is not the coordinate derivative of RPY. The original
    two-frequency motion remains the default for earlier-result replay.
    """
    stamps = np.asarray(time, dtype=float)
    if not np.isfinite(stamps).all():
        raise ValueError('Synthetic trajectory times must be finite')
    flat = stamps.reshape(-1)
    parameters = _motion_parameters(motion_profile)
    amplitude = np.deg2rad(parameters['amplitude_deg'])
    frequency = 2.*np.pi*parameters['frequency_hz']
    phase = flat[:, None, None]*frequency[None, :, :] + parameters['phase_rad'][None, :, :]
    rpy = np.sum(amplitude[None, :, :]*np.sin(phase), axis=-1)
    velocity = np.sum((amplitude*frequency)[None, :, :]*np.cos(phase), axis=-1)
    acceleration = np.sum(-(amplitude*frequency**2)[None, :, :]*np.sin(phase), axis=-1)
    roll, pitch, _yaw = rpy.T
    roll_dot, pitch_dot, yaw_dot = velocity.T
    roll_ddot, pitch_ddot, yaw_ddot = acceleration.T
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    omega = np.column_stack((
        roll_dot-yaw_dot*sp,
        pitch_dot*cr+yaw_dot*sr*cp,
        -pitch_dot*sr+yaw_dot*cr*cp))
    alpha = np.column_stack((
        roll_ddot-yaw_ddot*sp-yaw_dot*cp*pitch_dot,
        pitch_ddot*cr-pitch_dot*sr*roll_dot+yaw_ddot*sr*cp
        +yaw_dot*cr*roll_dot*cp-yaw_dot*sr*sp*pitch_dot,
        -pitch_ddot*sr-pitch_dot*cr*roll_dot+yaw_ddot*cr*cp
        -yaw_dot*sr*roll_dot*cp-yaw_dot*cr*sp*pitch_dot))
    gyroscopic = np.cross(omega, omega@INERTIA.T)
    torque = alpha@INERTIA.T+gyroscopic
    quaternion = Rotation.from_euler('xyz', rpy).as_quat()
    shape = stamps.shape
    def restore(value):
        return value.reshape(shape+(value.shape[-1],))
    return dict(time=stamps.copy(), rpy_rad=restore(rpy),
        rpy_velocity_rad_s=restore(velocity), rpy_acceleration_rad_s2=restore(acceleration),
        quaternion_xyzw=restore(quaternion), omega_body_rad_s=restore(omega),
        alpha_body_rad_s2=restore(alpha), torque_body_nm=restore(torque),
        gyroscopic_torque_nm=restore(gyroscopic))


def _native_grid(duration, rate):
    count = int(np.floor(duration*rate))
    return np.arange(count+1, dtype=float)/rate


def generate_dataset(duration=5., sensor_rate=200., seed=20261113,
                     motion_profile='two_frequency'):
    """Generate one saved realization shared by all acquisition/noise cases.

    The fixed 0.2 deg/s isotropic generation noise differs from the assumed
    input variances in the reconstruction sweep.  Bias is a separate fixed
    vector, not hidden in white-noise covariance.  The generator's covariance,
    bias and intermediate truth are provenance/reference fields, not inputs
    to a reconstruction's observation mean or assumed covariance.
    """
    duration, sensor_rate = float(duration), float(sensor_rate)
    if (not np.isfinite(duration) or duration <= 0.
            or not np.isfinite(sensor_rate) or sensor_rate <= 0.):
        raise ValueError('Positive finite duration and sensor rate required')
    parameters = _motion_parameters(motion_profile)
    nominal_imu_time = _native_grid(duration, sensor_rate)
    stamps_ns = np.rint(nominal_imu_time*1e9).astype(np.int64)
    imu_time = stamps_ns.astype(float)*1e-9
    if len(imu_time) < 2:
        raise ValueError('Generate at least two gyro readings')
    reference_time = np.unique(np.r_[imu_time, duration])
    torque_time = np.unique(np.r_[_native_grid(duration, TORQUE_RATE_HZ), duration])
    native = trajectory(imu_time, motion_profile)
    reference = trajectory(reference_time, motion_profile)
    control = trajectory(torque_time, motion_profile)
    covariance = np.eye(3)*np.deg2rad(GYRO_NOISE_SIGMA_DEG_S)**2
    bias = np.deg2rad(GYRO_BIAS_DEG_S)
    generator = np.random.default_rng(int(seed))
    noise = generator.normal(0., np.deg2rad(GYRO_NOISE_SIGMA_DEG_S),
                             size=native['omega_body_rad_s'].shape)
    gyro = native['omega_body_rad_s']+bias+noise
    indices = np.arange(len(imu_time), dtype=np.int64)
    q0 = trajectory(0., motion_profile)['quaternion_xyzw'][[3, 0, 1, 2]]
    data = dict(time=np.array([0., duration]), imu_time=imu_time,
        motion_profile=np.asarray(motion_profile),
        motion_amplitude_deg=parameters['amplitude_deg'].copy(),
        motion_frequency_hz=parameters['frequency_hz'].copy(),
        motion_phase_rad=parameters['phase_rad'].copy(),
        imu_source_index=indices, imu_stamp_ns=stamps_ns, imu_record_ns=stamps_ns.copy(),
        gyro=gyro, gyro_true_body_rad_s=native['omega_body_rad_s'],
        gyro_realized_noise_rad_s=noise, gyro_bias_rad_s=bias,
        generator_gyro_covariance_rad2_s2=covariance,
        initial_quaternion_wxyz=q0, inertia_body_kg_m2=INERTIA.copy(),
        reference_time=reference_time,
        reference_quaternion_xyzw=reference['quaternion_xyzw'],
        reference_rpy_rad=reference['rpy_rad'],
        reference_rpy_velocity_rad_s=reference['rpy_velocity_rad_s'],
        reference_rpy_acceleration_rad_s2=reference['rpy_acceleration_rad_s2'],
        reference_omega_body_rad_s=reference['omega_body_rad_s'],
        reference_alpha_body_rad_s2=reference['alpha_body_rad_s2'],
        reference_gyroscopic_torque_body_nm=reference['gyroscopic_torque_nm'],
        torque_time=torque_time, torque_body_nm=control['torque_body_nm'])
    metadata = dict(schema='grape/synthetic-controlled-rotation-input/v1',
        duration_seconds=duration, source_clock='synthetic seconds from t=0; ns clocks saved without physical transport delay',
        sensor_rate_hz=sensor_rate, sensor_count=len(imu_time),
        requested_window_seconds=[0., duration], actual_window_seconds=[0., duration],
        motion='prescribed smooth multi-frequency RPY motion; kinematic orientation experiment, not a full six-DoF flight/aircraft simulation',
        rotation='Rz(yaw) Ry(pitch) Rx(roll), body-to-world, Hamilton quaternion xyzw',
        motion_profile=motion_profile, frequency_components_per_axis=int(parameters['frequency_hz'].shape[1]),
        rpy_amplitude_deg=parameters['amplitude_deg'].tolist(),
        rpy_frequency_hz=parameters['frequency_hz'].tolist(),
        rpy_phase_rad=parameters['phase_rad'].tolist(),
        kinematics='analytic first/second derivatives of the prescribed RPY curve; exact body omega and alpha via RzRyRx rate mapping',
        inertia_body_kg_m2=INERTIA.tolist(),
        inertia_scope='known fixed physical tensor shared by generator and both comparison methods; no parameter fit',
        control='body-CoG torque tau=J alpha+omega cross(J omega), generated once from prescribed motion; saved on 1kHz grid and supplied as known input to both methods',
        control_rate_hz=TORQUE_RATE_HZ,
        torque_interpolation='reconstruction uses saved known torque with piecewise-linear interpolation; finite input interpolation error',
        gyro_observation='body omega + fixed bias + independent isotropic Gaussian sensor noise',
        gyro_noise_generation_sigma_deg_s=GYRO_NOISE_SIGMA_DEG_S,
        gyro_noise_generation_covariance_rad2_s2=covariance.tolist(),
        gyro_bias_generation_deg_s=GYRO_BIAS_DEG_S.tolist(),
        gyro_bias_is_white_noise=False,
        seed=int(seed), random_generator='numpy default_rng PCG64, generation only',
        random_sensor_noise_generation=True, monte_carlo_state_inference=False,
        generator_noise_covariance_or_bias_supplied_to_estimator=False,
        shared_known_physical_parameters=['inertia_body_kg_m2'],
        initial_pose='known prescribed pose once at t=0, shared by all cases',
        later_reference_attitudes_supplied_to_estimator=False,
        gyro_source_index=f'original {sensor_rate:g}Hz sequence index, never renumbered after sparse selection',
        realized_measurements_and_noise_saved=True)
    return data, metadata
