"""Scale the saved smooth body omega, preserving clocks and sensor disturbances.

This is a new known-input synthetic experiment. Scaling Euler angles would
not scale body omega exactly. Instead, prescribe omega3(t)=3 omega1(t),
integrate its quaternion kinematics, and regenerate the consistent torque.
No random draws or estimator changes are involved in this transformation.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trajectory = load_module('_omega3_base_trajectory', HERE.parent/'trajectory.py')
shared = load_module('_omega3_input_shared', HERE/'run.py')


def generate(directory=HERE/'frequent_publication', scale=3., acquisition_rate_hz=10.,
             spike_scale=1.):
    """Return source, sensor, acquisitions, estimator-only data, physics, metadata.

    Reference poses are numerically integrated with DOP853 using the analytic
    base angular velocity. Saved torque remains on the original 1 kHz grid;
    reconstruction uses the existing piecewise-linear torque/RK4 pipeline.
    """
    directory = Path(directory)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError('A finite positive angular-velocity scale is required')
    if not np.isfinite(acquisition_rate_hz) or acquisition_rate_hz <= 0.:
        raise ValueError('A finite positive acquisition rate is required')
    spike_scale = float(spike_scale)
    if not np.isfinite(spike_scale) or spike_scale <= 0.:
        raise ValueError('A finite positive spike-amplitude scale is required')
    original = dict(np.load(directory/'source_input.npz', allow_pickle=False))
    sensor = dict(np.load(directory/'sensor_input.npz', allow_pickle=False))
    parameters = {name: original['motion_'+name].copy() for name in
                  ('amplitude_deg', 'frequency_hz', 'phase_rad', 'offset_deg')}
    profile_name = str(original['motion_profile'])
    def motion(stamp):
        return trajectory.trajectory(stamp, profile_name, parameters)
    def quaternion_rhs(stamp, quaternion):
        omega = scale*motion(stamp)['omega_body_rad_s']
        vector, scalar = quaternion[:3], quaternion[3]
        return .5*np.r_[scalar*omega+np.cross(vector, omega), -np.dot(vector, omega)]
    q0 = original['initial_quaternion_wxyz'][[1, 2, 3, 0]]
    solution = solve_ivp(quaternion_rhs, original['time'], q0, method='DOP853',
        rtol=1e-11, atol=1e-13, dense_output=True)
    if not solution.success:
        raise RuntimeError('Synthetic quaternion integration failed: '+solution.message)
    quaternion = solution.sol(original['reference_time']).T
    if not np.isfinite(quaternion).all():
        raise FloatingPointError('Nonfinite generated quaternion')

    # Drop the old prescribed-RPY derivatives: the new prescribed variable is
    # body omega, so those derivatives would describe a different trajectory.
    source = {key: value.copy() for key, value in original.items()
              if key not in ('reference_rpy_velocity_rad_s',
                             'reference_rpy_acceleration_rad_s2')
              and not key.startswith('motion_')}
    source.update({'base_motion_'+name: value for name, value in parameters.items()})
    source.update(base_motion_profile=original['motion_profile'].copy(),
        motion_profile=np.asarray('body_omega_scaled'), body_omega_scale=np.asarray(scale),
        gyro_true_body_rad_s=scale*original['gyro_true_body_rad_s'],
        reference_omega_body_rad_s=scale*original['reference_omega_body_rad_s'],
        reference_alpha_body_rad_s2=scale*original['reference_alpha_body_rad_s2'],
        reference_gyroscopic_torque_body_nm=scale**2*original['reference_gyroscopic_torque_body_nm'],
        reference_quaternion_xyzw=quaternion,
        reference_rpy_rad=Rotation.from_quat(quaternion).as_euler('xyz'))
    control = motion(original['torque_time'])
    omega = scale*control['omega_body_rad_s']
    alpha = scale*control['alpha_body_rad_s2']
    inertia = source['inertia_body_kg_m2']
    source['torque_body_nm'] = alpha@inertia.T+np.cross(omega, omega@inertia.T)
    # Reuse each saved disturbance component, rather than subtracting two
    # measured arrays and introducing a new floating-point noise realization.
    source['gyro'] = (source['gyro_true_body_rad_s']+source['gyro_bias_rad_s']
                      +source['gyro_realized_noise_rad_s'])
    sensor['gyro_white'] = source['gyro'].copy()
    for key in ('spike_vector_rad_s', 'event_vector_body_rad_s', 'event_magnitude_deg_s'):
        sensor[key] = spike_scale*sensor[key]
    sensor['gyro_spike'] = sensor['gyro_white']+sensor['spike_vector_rad_s']

    desired = np.arange(*source['time'], 1./acquisition_rate_hz)
    desired = np.r_[desired, source['time'][1]]
    requested = np.rint(desired*1e9).astype(np.int64)
    index = np.searchsorted(sensor['stamp_ns'], requested)
    acquired = {key: sensor[key][index].copy() for key in
                ('time', 'stamp_ns', 'record_ns', 'source_index', 'gyro_spike',
                 'spike_mask', 'spike_vector_rad_s')}
    acquired.update(native_array_index=index, requested_stamp_ns=requested)
    selected = dict(time=source['time'].copy(),
        initial_quaternion_wxyz=source['initial_quaternion_wxyz'].copy(),
        sparse_gyro_time=acquired['time'].copy(), sparse_gyro=acquired['gyro_spike'].copy())
    physical = dict(inertia=inertia.copy(),
        torque=shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
    unchanged_source_keys = ('time', 'imu_time', 'imu_source_index', 'imu_stamp_ns',
        'imu_record_ns', 'gyro_realized_noise_rad_s', 'gyro_bias_rad_s',
        'generator_gyro_covariance_rad2_s2', 'initial_quaternion_wxyz',
        'inertia_body_kg_m2', 'reference_time', 'torque_time')
    old_sensor = dict(np.load(directory/'sensor_input.npz', allow_pickle=False))
    metadata = dict(schema='grape/body-omega-scaled-synthetic-input/v1',
        body_omega_scale=scale, alpha_scale=scale, gyroscopic_torque_scale=scale**2,
        spike_scale=spike_scale,
        spike_scaling='Multiply saved per-pulse magnitudes/vectors and their native vector sum; retain every event time/width/direction and acquisition index',
        spike_timing_rate_width_or_direction_changed=False,
        single_pulse_magnitude_range_deg_s=(spike_scale*np.array([20.,60.])).tolist(),
        time_or_frequency_rescaled=False, rpy_amplitudes_scaled=False,
        quaternion='Hamilton xyzw, body-to-world; integrate qdot=q Hamilton (0,omega3)/2 from the same q0',
        torque='J alpha3 + omega3 cross(J omega3), saved at unchanged 1 kHz timestamps',
        base_motion_parameters=parameters,
        quaternion_truth_integration=dict(method='scipy solve_ivp DOP853',rtol=1e-11,
            atol=1e-13, function_evaluations=solution.nfev,
            maximum_norm_error=float(np.max(abs(np.linalg.norm(quaternion, axis=1)-1.)))),
        base_file_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('source_input.npz', 'sensor_input.npz')},
        unchanged_source_arrays={key: bool(np.array_equal(source[key], original[key]))
                                 for key in unchanged_source_keys},
        unchanged_sensor_disturbances_and_clocks={key: bool(np.array_equal(value, old_sensor[key]))
            for key, value in sensor.items() if key not in ('gyro_white', 'gyro_spike')},
        native_reading_count=len(sensor['time']), acquisition_count=len(acquired['time']),
        native_rate_hz=200., acquisition_rate_hz=acquisition_rate_hz,
        generation_white_sigma_deg_s=2., measurement_assumed_sigma_deg_s=5.,
        fixed_bias_and_white_noise_and_pulse_timing_and_direction_reused=True,
        new_random_draws=False, future_truth_or_noise_supplied_to_estimator=False,
        estimator_data_fields=list(selected), process_covariance_rad2_s2=np.zeros((3,3)),
        old_omega_std_deg_s=np.rad2deg(np.std(original['reference_omega_body_rad_s'],axis=0)),
        new_omega_std_deg_s=np.rad2deg(np.std(source['reference_omega_body_rad_s'],axis=0)))
    return source, sensor, acquired, selected, physical, metadata
