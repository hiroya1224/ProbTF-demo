"""Native ProbTF Bingham IMU recurrence with output-only magnetic fusion.

The carried law contains gyro prediction and acceleration evidence only.
The magnetic law is constructed for that timestamp's output and never becomes
the next prior. All sensor clocks remain the original extracted IMU clocks.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time as wall_time

import numpy as np

from imu_signal import local_polynomial_signal, corrected_lag_signal

ROOT = Path(__file__).resolve().parents[5]
CORE = ROOT / 'ros/core/probtf_core/src'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
ORIENTATION_SCRIPTS = ROOT / 'ros/examples/probtf_orientation_demo/scripts'
spec = importlib.util.spec_from_file_location('imu_lag_existing_moment_inverse',
                                             ORIENTATION_SCRIPTS / 'bingham_moment_inverse.py')
moment_inverse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moment_inverse)

from probtf.bingham import (
    bingham_second_moment, bingham_fourth_moment, rotation_first_moment,
    rotation_vector_second_moment, canonical_bingham_parameter,
)
from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import bingham_rotation_coordinates
from probtf_estimators.orientation_imu import gravity_bingham_evidence, magnetic_bingham_evidence


def _isotropic_sensor_covariance(value):
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError('A finite individual-reading sensor covariance is required.')
    variance = float(np.trace(covariance) / 3.)
    if variance < 0:
        raise ValueError('Sensor noise variance cannot be negative.')
    return np.eye(3) * variance


def gravity_body_moments(parameter, gravity_world, integration_steps=120):
    """Query E[R.T g] and Cov[R.T g] from the same native Bingham law."""
    second = bingham_second_moment(parameter, integration_steps=integration_steps)
    fourth = bingham_fourth_moment(parameter, integration_steps=integration_steps)
    mean_rotation = rotation_first_moment(second)
    rotation_second = rotation_vector_second_moment(fourth)
    action = np.kron(np.eye(3), np.asarray(gravity_world).reshape(1, 3))
    mean = mean_rotation.T @ gravity_world
    covariance = action @ rotation_second @ action.T - np.outer(mean, mean)
    return mean, .5 * (covariance + covariance.T)


def conservative_acceleration_covariance(sensor_covariance, gravity_mean,
                                         gravity_covariance, acceleration):
    """An explicit data-dependent dynamic-acceleration uncertainty envelope.

    The innovation outer product models unresolved translational acceleration
    as a zero-mean second-moment envelope. Shared-data cross covariance with
    attitude is not inferred: twice the sum bounds unknown cross terms in the
    linearized marginal-error model. This is a conservative approximation,
    not an exact Bayesian treatment of the same acceleration used twice.

    The largest eigenvalue times I dominates the full anisotropic envelope.
    trace(C)/3*I would not preserve that Loewner upper bound. Static sensor
    noise itself remains the user's trace(S)/3*I model.
    """
    residual = np.asarray(acceleration) - np.asarray(gravity_mean)
    dynamic = np.outer(residual, residual)
    effective = 2. * (np.asarray(sensor_covariance) + gravity_covariance + dynamic)
    effective = .5 * (effective + effective.T)
    isotropic_variance = float(np.linalg.eigvalsh(effective)[-1])
    return effective, np.eye(3) * isotropic_variance, dynamic


def _continuous_mode(parameter, previous):
    quaternion, covariance = bingham_rotation_coordinates(parameter)
    quaternion = np.array(quaternion)
    if quaternion @ previous < 0:
        quaternion *= -1
    return quaternion, covariance


def estimate_orientation(data, gyro_tau, accel_tau, lag_covariance_2x2=None, options=None):
    """Estimate both laws without using any later mocap orientation.

    ``lag_covariance_2x2`` is covariance of the physical time constants in
    seconds squared, ordered gyro then acceleration. It is not log-tau
    covariance. ``options`` controls explicit offline local-polynomial and
    existing native moment-inverse settings; it is never selected from mocap.
    """
    options = {} if options is None else dict(options)
    window = float(options.get('window_seconds', .08))
    degree = int(options.get('degree', 3))
    integration_steps = int(options.get('integration_steps', 120))
    concentration = float(options.get('initial_concentration', 100.))
    gravity_magnitude = float(options.get('gravity_magnitude', 9.80665))
    if not np.isfinite(concentration) or concentration <= 0 or not np.isfinite(gravity_magnitude) or gravity_magnitude <= 0:
        raise ValueError('The finite initial Bingham concentration and gravity magnitude must be positive.')
    stamps = np.asarray(data['time'], dtype=float)
    initial = np.asarray(data['initial_quaternion_wxyz'], dtype=float)
    if initial.shape != (4,) or not np.isfinite(initial).all() or not np.linalg.norm(initial) > 0:
        raise ValueError('A finite initial quaternion from the initial mocap alignment is required.')
    initial = initial / np.linalg.norm(initial)
    lag_covariance = (np.zeros((2, 2)) if lag_covariance_2x2 is None
                      else np.asarray(lag_covariance_2x2, dtype=float))
    if lag_covariance.shape != (2, 2) or not np.isfinite(lag_covariance).all():
        raise ValueError('The two physical sensor time constants need a finite 2x2 covariance.')
    sensor_covariances = {name: _isotropic_sensor_covariance(data[name + '_noise_covariance'])
                          for name in ('gyro', 'accel', 'mag')}
    signals = {name: local_polynomial_signal(stamps, data[name], sensor_covariances[name],
                                             window_seconds=window, degree=degree)
               for name in ('gyro', 'accel')}
    gyro = corrected_lag_signal(signals['gyro'], gyro_tau, lag_covariance[0, 0])
    accel = corrected_lag_signal(signals['accel'], accel_tau, lag_covariance[1, 1])
    n = len(stamps)
    magnetic = np.asarray(data.get('mag', np.zeros((n, 3))), dtype=float)
    observed = np.asarray(data.get('mag_observed', np.zeros(n, dtype=bool)), dtype=bool)
    if magnetic.shape != (n, 3) or observed.shape != (n,):
        raise ValueError('Magnetic values and their actual-new-observation mask must share the native IMU rows.')
    magnetic_valid = observed & np.isfinite(magnetic).all(axis=1) & (np.linalg.norm(magnetic, axis=1) > 0)
    reference = np.asarray(data.get('initial_magnetic_reference_world', np.zeros(3)), dtype=float)
    reference_available = (reference.shape == (3,) and np.isfinite(reference).all()
                           and np.linalg.norm(reference) > 0)
    if not reference_available and magnetic_valid[0]:
        reference = quat_to_rotmat(initial) @ magnetic[0]
        reference_available = True
    initial_parameter = concentration * (np.outer(initial, initial) - np.eye(4))
    recursive = np.empty((n, 4, 4)); output = np.empty_like(recursive)
    q_recursive = np.empty((n, 4)); q_output = np.empty_like(q_recursive)
    covariance_recursive = np.empty((n, 3, 3)); covariance_output = np.empty_like(covariance_recursive)
    effective_accel = np.zeros((n, 3, 3)); isotropic_accel = np.zeros_like(effective_accel)
    dynamic_accel = np.zeros_like(effective_accel)
    gravity_mean = np.zeros((n, 3)); gravity_covariance = np.zeros_like(effective_accel)
    accel_weight = np.zeros(n); mag_weight = np.zeros(n); moment_residual = np.zeros(n)
    accel_updated = np.zeros(n, dtype=bool); mag_updated = np.zeros(n, dtype=bool)
    state = canonical_bingham_parameter(initial_parameter)
    gravity_world = np.array([0., 0., gravity_magnitude])
    started = wall_time.monotonic()
    progress = options.get('progress')
    for k in range(n):
        if k:
            state, moment_residual[k] = moment_inverse.propagate(
                state, gyro['mean'][k], float(stamps[k] - stamps[k - 1]),
                gyro['covariance'][k], integration_steps=integration_steps)
            gravity_mean[k], gravity_covariance[k] = gravity_body_moments(
                state, gravity_world, integration_steps)
            effective_accel[k], isotropic_accel[k], dynamic_accel[k] = conservative_acceleration_covariance(
                accel['covariance'][k], gravity_mean[k], gravity_covariance[k], accel['mean'][k])
            variance = float(isotropic_accel[k, 0, 0])
            magnitude = float(np.linalg.norm(accel['mean'][k]))
            if magnitude > 0 and variance > 0:
                accel_weight[k] = gravity_magnitude * magnitude / variance
                state = canonical_bingham_parameter(state + gravity_bingham_evidence(
                    gravity_world, accel['mean'][k], accel_weight[k]))
                accel_updated[k] = True
        recursive[k] = state
        q_recursive[k], covariance_recursive[k] = _continuous_mode(
            state, initial if k == 0 else q_recursive[k - 1])
        if k == 0:
            q_recursive[k] = initial  # The initial mode is known analytically.
        displayed = state
        if k and reference_available and magnetic_valid[k]:
            # Reference was formed from one initial reading, not a noise-free
            # world field. Both individual-reading noise covariances enter.
            variance = 2. * float(sensor_covariances['mag'][0, 0])
            if variance > 0:
                mag_weight[k] = float(np.linalg.norm(reference) * np.linalg.norm(magnetic[k]) / variance)
                displayed = canonical_bingham_parameter(state + magnetic_bingham_evidence(
                    reference, magnetic[k], mag_weight[k]))
                mag_updated[k] = True
        output[k] = displayed
        q_output[k], covariance_output[k] = _continuous_mode(
            displayed, initial if k == 0 else q_output[k - 1])
        if k == 0:
            q_output[k] = initial
        if progress is not None and (k + 1 == n or (k + 1) % 200 == 0):
            progress(k + 1, n, wall_time.monotonic() - started)
    cross = lag_covariance[0, 1] * np.einsum('ni,nj->nij', signals['gyro']['derivative'], signals['accel']['derivative'])
    return dict(time=stamps.copy(), A_recursive=recursive, A_output=output,
                q_recursive_wxyz=q_recursive, q_output_wxyz=q_output,
                covariance_recursive=covariance_recursive, covariance_output=covariance_output,
                right_rotation_covariance_recursive=covariance_recursive,
                right_rotation_covariance_output=covariance_output,
                corrected_gyro=gyro['mean'], corrected_accel=accel['mean'],
                corrected_gyro_covariance=gyro['covariance'], corrected_accel_covariance=accel['covariance'],
                corrected_gyro_accel_cross_covariance=cross,
                gyro_tau_sensitivity=signals['gyro']['derivative'], accel_tau_sensitivity=signals['accel']['derivative'],
                effective_accel_covariance=effective_accel,
                effective_accel_isotropic_covariance=isotropic_accel,
                dynamic_accel_second_moment=dynamic_accel,
                predicted_gravity_body_mean=gravity_mean, predicted_gravity_body_covariance=gravity_covariance,
                acceleration_concentration=accel_weight, magnetic_concentration=mag_weight,
                acceleration_updated=accel_updated, magnetic_updated=mag_updated,
                moment_fit_residual=moment_residual, lag_covariance_2x2=lag_covariance.copy(),
                metadata=dict(method='native ProbTF Bingham gyro+acceleration recurrence; magnetic fusion only in output',
                    sensor_lag_model='x = local_polynomial(y) + tau * local_polynomial_derivative(y)',
                    sensor_lag_window_seconds=window, sensor_lag_polynomial_degree=degree,
                    sensor_lag_offline_centered_windows=True, integration_steps=integration_steps,
                    initial_concentration=concentration, initial_quaternion_wxyz=initial.tolist(),
                    initial_observation_updates_skipped=True,
                    only_initial_mocap_used=True, magnetic_reference_available=bool(reference_available),
                    magnetic_reference_world=reference.tolist() if reference_available else None,
                    magnetic_observation_policy='new-observation mask only; output factors never carried forward',
                    magnetic_noise='current individual-reading sensor variance plus initial-reference individual-reading variance',
                    gyro_prediction_convention='body-frame right increment, right-end native angular velocity',
                    acceleration_covariance='2*(corrected_sensor_covariance + predicted_gravity_covariance + innovation_outer_product)',
                    acceleration_covariance_isotropization='lambda_max(C_eff)*I Loewner upper envelope; static sensor noise remains trace(S)/3*I',
                    shared_acceleration_attitude_information='conservative data-dependent envelope; not exact Bayes with shared observations',
                    orientation_covariance='native Bingham-derived high-concentration right rotation coordinates, same A law',
                    propagation='existing deterministic cubature and native quaternion-product moments; no Monte Carlo',
                    estimated_sensor_lag_covariance_units='physical tau seconds squared, order gyro/acceleration',
                    wall_seconds=wall_time.monotonic() - started))
