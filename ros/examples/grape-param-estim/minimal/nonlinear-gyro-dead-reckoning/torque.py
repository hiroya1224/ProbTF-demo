"""Known-control body torque and a strictly past, frozen residual mean.

The evaluation gyro and all noninitial mocap are deliberately absent from
this module's input accesses.  The prediction is conditional on the nominal
physical model, frozen thrust lag and past residual mean.  The empirical
past residual scatter is saved as a diagnostic, not added to the omega-law
covariance or described as a posterior for future unknown torque.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MINIMAL = HERE.parent
for directory in (MINIMAL, MINIMAL.parent / 'src', MINIMAL / 'imu-first-order-lag'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from grape_param_estim.gimbalrotor_pid_postprocess import load_vehicle_model
from imu_signal import local_polynomial_signal

_path = MINIMAL / 'first-order-lag/core.py'
_spec = importlib.util.spec_from_file_location('nonlinear_gyro_existing_thrust_lag', _path)
_thrust = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _thrust
_spec.loader.exec_module(_thrust)
first_order_zoh_thrust_history = _thrust.first_order_zoh_thrust_history

DEFAULT_THRUST_LAG_SOURCE = (MINIMAL / 'imu-first-order-lag/publication/'
                             'imu_native_20261031/success/estimate.json')
MINIMUM_THRUST = 1.5
MAXIMUM_THRUST = 27.6145


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def nominal_actuator_wrench(model, thrust, gimbal):
    """Vectorized copy of the existing nominal actuator-wrench formula.

    Torques are about the nominal CoG and expressed in the common body
    frame.  Rotor arm geometry, reaction moment and thrust-origin motion
    under measured gimbal angles are all retained.
    """
    thrust, gimbal = np.asarray(thrust, float), np.asarray(gimbal, float)
    if (thrust.shape != gimbal.shape or thrust.ndim != 2 or thrust.shape[1] != 4
            or not np.isfinite(thrust).all() or not np.isfinite(gimbal).all()):
        raise ValueError('Finite Nx4 thrust and measured-gimbal arrays are required')
    parameters, geometry = model.parameters, model.body_geometry
    yaw = np.asarray(geometry.arm_yaws)
    direction = np.stack((np.sin(yaw)[None, :] * np.sin(gimbal),
                          -np.cos(yaw)[None, :] * np.sin(gimbal),
                          np.cos(gimbal)), axis=-1)
    origins = np.asarray(geometry.rotor_origins)[None, :, :] + float(geometry.thrust_offset) * (
        direction - np.array([0., 0., 1.]))
    force_rotor = (thrust * np.asarray(parameters.force_effectiveness)[None, :])[:, :, None] * direction
    reaction = (np.asarray(parameters.torque_effectiveness)
                * np.asarray(geometry.rotor_directions) * float(geometry.moment_force_rate))
    torque_rotor = np.cross(origins - np.asarray(parameters.cog_offset), force_rotor)
    torque_rotor += reaction[None, :, None] * force_rotor
    return np.concatenate((force_rotor.sum(axis=1), torque_rotor.sum(axis=1)), axis=1)


def _time_weights(time):
    stamps = np.asarray(time, float)
    if len(stamps) < 2 or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0.):
        raise ValueError('Past calibration needs at least two increasing finite timestamps')
    dt = np.diff(stamps)
    weights = np.r_[dt[0] / 2., (dt[:-1] + dt[1:]) / 2., dt[-1] / 2.]
    return weights / (stamps[-1] - stamps[0])


class TorqueProfile:
    """Piecewise-linear known body torque with one frozen residual mean."""
    def __init__(self, time, torque, residual_mean=None):
        self.time = np.asarray(time, float).copy()
        self.torque = np.asarray(torque, float).copy()
        self.residual_mean = (np.zeros(3) if residual_mean is None
                              else np.asarray(residual_mean, float).copy())
        if (self.time.ndim != 1 or len(self.time) < 2
                or self.torque.shape != (len(self.time), 3)
                or self.residual_mean.shape != (3,)
                or not np.isfinite(self.time).all() or not np.isfinite(self.torque).all()
                or not np.isfinite(self.residual_mean).all()
                or np.any(np.diff(self.time) <= 0.)):
            raise ValueError('A finite increasing torque profile and finite residual mean are required')

    def __call__(self, time):
        query = np.asarray(time, float)
        if (not np.isfinite(query).all() or np.any(query < self.time[0])
                or np.any(query > self.time[-1])):
            raise ValueError('Torque query is outside the recorded profile support')
        flat = query.reshape(-1)
        value = np.column_stack([np.interp(flat, self.time, self.torque[:, j]) for j in range(3)])
        value += self.residual_mean
        return value.reshape(query.shape + (3,))


def build_torque_profile(data, model_path, thrust_tau=None, *, profile_step_seconds=.001):
    """Prepare nominal actuator torque plus a causal calibration residual.

    Only ``past_calibration_gyro`` from strictly before the evaluation start
    is read.  Its local polynomial is offline within that already-completed
    prefix.  Raw gyro is treated as physical angular velocity there: this
    experiment does not invert the gyro's sensor lag using hidden dense
    evaluation measurements.
    """
    model_path = Path(model_path).resolve()
    model = load_vehicle_model(model_path)
    inertia = np.asarray(model.parameters.inertia, float).copy()
    # Validate the given full SPD matrix without changing its eigenvalues.
    np.linalg.cholesky(inertia)
    tau_source = None
    if thrust_tau is None:
        tau_source = DEFAULT_THRUST_LAG_SOURCE.resolve()
        prior = json.loads(tau_source.read_text())
        thrust_tau = float(prior['time_constants_seconds']['thrust_time_constant_seconds'])
    tau, step = float(thrust_tau), float(profile_step_seconds)
    if not np.isfinite(tau) or tau <= 0. or not np.isfinite(step) or step <= 0.:
        raise ValueError('Positive finite thrust lag and profile spacing are required')
    time = np.asarray(data['time'], float)
    past_time = np.asarray(data['past_calibration_gyro_time'], float)
    past_gyro = np.asarray(data['past_calibration_gyro'], float)
    if (time.shape != (2,) or not np.isfinite(time).all() or time[1] <= time[0]
            or past_time.ndim != 1 or len(past_time) < 4
            or not np.isfinite(past_time).all() or np.any(np.diff(past_time) <= 0.)
            or past_time[-1] >= time[0]):
        raise ValueError('Calibration must finish strictly before the finite prediction interval')
    signal = local_polynomial_signal(past_time, past_gyro,
        np.asarray(data['gyro_noise_covariance'], float), window_seconds=.08,
        degree=3, max_derivative=1)
    first = float(past_time[0]); last = float(time[1])
    uniform = np.linspace(first, last, max(1, int(np.ceil((last - first) / step))) + 1)
    profile_time = np.unique(np.r_[uniform, past_time, time])
    command_time = np.asarray(data['rotor_command_time'], float)
    command = np.asarray(data['rotor_command'], float)
    gimbal_time = np.asarray(data['gimbal_time'], float)
    gimbal_angle = np.asarray(data['gimbal_angle'], float)
    if (profile_time[0] < gimbal_time[0] or profile_time[-1] > gimbal_time[-1]
            or gimbal_angle.shape != (len(gimbal_time), 4)
            or np.any(np.diff(gimbal_time) <= 0.) or not np.isfinite(gimbal_angle).all()):
        raise ValueError('Measured gimbal history does not support the torque interval')
    history = first_order_zoh_thrust_history(command_times=command_time,
        command_values=command, evaluation_times=profile_time, time_constant=tau,
        minimum_thrust=MINIMUM_THRUST, maximum_thrust=MAXIMUM_THRUST)
    gimbal = np.column_stack([np.interp(profile_time, gimbal_time, gimbal_angle[:, j]) for j in range(4)])
    wrench = nominal_actuator_wrench(model, history.thrust, gimbal)
    calibration_index = np.searchsorted(profile_time, past_time)
    omega, alpha = signal['smooth'], signal['derivative']
    required = alpha @ inertia.T + np.cross(omega, omega @ inertia.T)
    past_residual = required - wrench[calibration_index, 3:]
    weights = _time_weights(past_time)
    residual_mean = weights @ past_residual
    centred = past_residual - residual_mean
    residual_scatter = np.einsum('n,ni,nj->ij', weights, centred, centred)
    arrays = dict(torque_profile_time=profile_time,
        torque_actuator_body=wrench[:, 3:], torque_applied_body=wrench[:, 3:] + residual_mean,
        actuator_wrench_body=wrench, profile_thrust=history.thrust,
        profile_gimbal_angle=gimbal, residual_mean_body=residual_mean,
        calibration_time=past_time.copy(), calibration_gyro_smooth=omega,
        calibration_angular_acceleration=alpha, calibration_required_torque_body=required,
        calibration_actuator_torque_body=wrench[calibration_index, 3:],
        calibration_residual_torque_body=past_residual, calibration_time_weights=weights,
        calibration_residual_scatter_body=residual_scatter,
        nominal_inertia_body=inertia)
    metadata = dict(schema='grape/nonlinear-gyro-causal-torque/v1',
        model_path=str(model_path), model_sha256=_sha(model_path),
        nominal_inertia_body=inertia.tolist(), inertia_diagonalized=False,
        physical_parameters='entire nominal grape_vehicle_model.json; no fitted success physical values',
        thrust_tau_seconds=tau, thrust_tau_source=('explicit fixed argument' if tau_source is None
            else dict(path=str(tau_source), sha256=_sha(tau_source),
                      field='time_constants_seconds.thrust_time_constant_seconds',
                      role='previously identified value from another record, frozen before evaluation; physical parameters not read')),
        thrust_model='existing exact ZOH-input first-order response over complete supplied command prefix',
        thrust_bounds_newton=[MINIMUM_THRUST, MAXIMUM_THRUST],
        thrust_bound_role='existing actuator saturation model; not covariance clipping',
        thrust_active_set_counts=dict(history.active_set_counts),
        gimbal_model='actual measured gimbal angles, linear interpolation on original header clock',
        actuator_frame='body frame, torque about nominal CoG; full rotor geometry and reaction moment retained',
        torque_profile_interpolation='piecewise linear from actuator torque evaluated on the bounded-spacing grid plus all calibration times; finite interpolation approximation',
        requested_profile_step_seconds=step,
        maximum_profile_step_seconds=float(np.max(np.diff(profile_time))),
        residual_model='constant time-weighted mean of strictly past calibration required-minus-actuator torque',
        residual_mean_body_nm=residual_mean.tolist(),
        calibration_time_seconds=[float(past_time[0]), float(past_time[-1])],
        calibration_measurement_count=len(past_time),
        calibration_signal='raw gyro treated as physical omega; local polynomial .08s degree3 within completed past prefix only',
        gyro_sensor_lag_inverted=False, calibration_future_relative_to_prediction_used=False,
        calibration_covariance_role='individual-reading isotropic gyro covariance; used by existing derivative diagnostic',
        residual_scatter_role='time-weighted centred empirical torque scatter, not a posterior or covariance of the estimated mean; saved diagnostic only',
        band_conditioning='nominal J, known control profile, frozen thrust tau and frozen past residual mean; only requested omega candidate uncertainty is propagated',
        evaluation_dense_gyro_used=False, ongoing_mocap_used=False,
        input_sha256={key:_array_sha(data[key]) for key in ('rotor_command_time', 'rotor_command',
            'gimbal_time', 'gimbal_angle', 'past_calibration_gyro_time',
            'past_calibration_gyro', 'gyro_noise_covariance')},
        implementation_sha256=_sha(Path(__file__)))
    return dict(torque=TorqueProfile(profile_time, wrench[:, 3:], residual_mean),
        actuator_torque=TorqueProfile(profile_time, wrench[:, 3:]),
        inertia=inertia, residual_mean=residual_mean, arrays=arrays, metadata=metadata)
