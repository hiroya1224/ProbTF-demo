"""Independent deterministic checks for the synthetic controlled rotation.

These diagnostics do not change the primary noisy measurements, fitted law,
or candidate covariance.  The zero-noise reconstruction is a separate unit
experiment, never substituted for the requested biased/noisy experiment.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for directory in (HERE, PARENT/'rate_comparison_5hz'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from trajectory import trajectory, INERTIA
from engine import predict_factorized


def _scores(prediction, motion_profile='two_frequency', *, evaluate=None):
    reference = (trajectory(prediction['time'], motion_profile=motion_profile)
                 if evaluate is None else evaluate(prediction['time']))
    original = Rotation.from_quat(reference['quaternion_xyzw'])
    estimated = Rotation.from_euler('xyz', prediction['rpy_mean_rad'])
    error = (original.inv()*estimated).magnitude()
    return dict(geodesic_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(error**2)))),
                geodesic_max_deg=float(np.rad2deg(np.max(error))),
                geodesic_final_deg=float(np.rad2deg(error[-1])))


def audit_dataset(data, profile, *, rates=(.5, 2., 5.)):
    """Return read-only numerical/kinematic diagnostics as JSON-safe values.

    ``profile`` has the same fixed recorded input callable and fixed inertia
    supplied to the main estimator.  Analytic reference values are used only
    to check this construction and to generate ideal observations for the
    explicitly separate zero-noise/no-bias unit experiment below. The
    recorded ``motion_profile`` selects the prescribed analytic reference;
    old saved datasets without that key retain the two-frequency profile.
    When ``angular_rate_scale`` is present, the scaled body-rate law and its
    numerically integrated attitude are used consistently instead.
    ``rates`` limits only this separate diagnostic, never a primary update.
    """
    start, end = np.asarray(data['time'], float)
    motion_profile = str(np.asarray(data.get('motion_profile', 'two_frequency')).item())
    angular_rate_scale = float(np.asarray(data.get('angular_rate_scale', 1.)).item())
    if 'angular_rate_scale' in data:
        from scaled_motion import scaled_trajectory
        evaluate = lambda stamps: scaled_trajectory(stamps, angular_rate_scale=angular_rate_scale)
    else:
        evaluate = lambda stamps: trajectory(stamps, motion_profile=motion_profile)
    step = 1e-5
    times = np.linspace(start+2*step, end-2*step, 101)
    middle = evaluate(times)
    before = evaluate(times-step)
    after = evaluate(times+step)
    rotation = Rotation.from_quat(middle['quaternion_xyzw']).as_matrix()
    matrix_dot = (Rotation.from_quat(after['quaternion_xyzw']).as_matrix()
                  - Rotation.from_quat(before['quaternion_xyzw']).as_matrix())/(2*step)
    body_hat = np.einsum('tji,tjk->tik', rotation, matrix_dot)
    omega_numerical = .5*np.column_stack((body_hat[:, 2, 1]-body_hat[:, 1, 2],
                                         body_hat[:, 0, 2]-body_hat[:, 2, 0],
                                         body_hat[:, 1, 0]-body_hat[:, 0, 1]))
    alpha_numerical = (after['omega_body_rad_s']-before['omega_body_rad_s'])/(2*step)
    required = (middle['alpha_body_rad_s2']@INERTIA.T
                + np.cross(middle['omega_body_rad_s'], middle['omega_body_rad_s']@INERTIA.T))
    dense = evaluate(np.linspace(start, end, 1001))
    gyroscopic_rms = np.sqrt(np.mean(dense['gyroscopic_torque_nm']**2, axis=0))
    inertial_rms = np.sqrt(np.mean((dense['alpha_body_rad_s2']@INERTIA.T)**2, axis=0))
    control_times = np.asarray(data['torque_time'], float)
    control_midpoints = .5*(control_times[:-1]+control_times[1:])
    profile_error = (profile['torque'](control_midpoints)
                     -evaluate(control_midpoints)['torque_body_nm'])
    second = np.trace(INERTIA)/2*np.eye(3)-INERTIA
    result = dict(schema='grape/synthetic-controlled-rotation-audit/v1',
        role='read-only construction and integration checks; no state acceptance gate',
        motion_profile=motion_profile,
        angular_rate_scale=angular_rate_scale,
        analytic_body_omega_vs_rotation_derivative_max_abs_rad_s=float(np.max(np.abs(
            omega_numerical-middle['omega_body_rad_s']))),
        analytic_body_alpha_vs_omega_derivative_max_abs_rad_s2=float(np.max(np.abs(
            alpha_numerical-middle['alpha_body_rad_s2']))),
        torque_equation_max_abs_residual_nm=float(np.max(np.abs(required-middle['torque_body_nm']))),
        torque_profile_vs_analytic_max_abs_nm=float(np.max(np.abs(profile_error))),
        torque_profile_vs_analytic_rms_axis_nm=np.sqrt(np.mean(profile_error**2, axis=0)).tolist(),
        torque_profile_check='all saved input-cell midpoints, not just stored control nodes',
        torque_profile_check_count=len(control_midpoints),
        gyroscopic_torque_rms_axis_nm=gyroscopic_rms.tolist(),
        inertial_torque_rms_axis_nm=inertial_rms.tolist(),
        gyroscopic_over_inertial_rms_axis=(gyroscopic_rms/inertial_rms).tolist(),
        principal_inertias_kg_m2=np.linalg.eigvalsh(INERTIA).tolist(),
        physical_second_moment_eigenvalues_kg_m2=np.linalg.eigvalsh(second).tolist(),
        trajectory_changed_using_comparison_results=False,
        estimation_monte_carlo=False)
    if 'angular_rate_scale' in data:
        unscaled = trajectory(times, motion_profile=motion_profile)
        source_dense = trajectory(np.linspace(start, end, 1001), motion_profile=motion_profile)
        result['scaled_motion_consistency'] = dict(
            operation='body omega(t) and alpha(t) scaled at unchanged physical times; attitude is reintegrated',
            omega_vs_scaled_source_max_abs_rad_s=float(np.max(np.abs(
                middle['omega_body_rad_s']-angular_rate_scale*unscaled['omega_body_rad_s']))),
            alpha_vs_scaled_source_max_abs_rad_s2=float(np.max(np.abs(
                middle['alpha_body_rad_s2']-angular_rate_scale*unscaled['alpha_body_rad_s2']))),
            gyroscopic_vs_squared_scale_max_abs_nm=float(np.max(np.abs(
                middle['gyroscopic_torque_nm']-angular_rate_scale**2*unscaled['gyroscopic_torque_nm']))),
            source_peak_axis_body_omega_deg_s=np.rad2deg(np.max(np.abs(source_dense['omega_body_rad_s']), axis=0)).tolist(),
            scaled_peak_axis_body_omega_deg_s=np.rad2deg(np.max(np.abs(dense['omega_body_rad_s']), axis=0)).tolist())
    rate_results = {}
    native = np.asarray(data['imu_time'], float)
    native_ns = np.asarray(data.get('imu_stamp_ns', np.rint(native*1e9)), np.int64)
    requested_rates = tuple(float(rate) for rate in rates)
    for rate in requested_rates:
        desired = start+np.arange(int(np.ceil((end-start)*rate)))/rate
        selected_index = np.searchsorted(native_ns, np.rint(desired*1e9).astype(np.int64), side='left')
        selected = native[selected_index]
        ideal = dict(time=np.asarray([start, end]), sparse_gyro_time=selected,
                     sparse_gyro=evaluate(selected)['omega_body_rad_s'],
                     initial_quaternion_wxyz=np.asarray(data['initial_quaternion_wxyz']))
        methods = {}
        for mode in ('nonlinear', 'linear_forced'):
            prediction = predict_factorized(ideal, profile, 0., mode, output_dt=.02, max_step=.002)
            methods[mode] = _scores(prediction, motion_profile=motion_profile, evaluate=evaluate)
        rate_results[str(rate)] = dict(observation_count=len(selected),
            acquisition_source_index=np.asarray(data.get('imu_source_index', np.arange(len(native))))[selected_index].tolist(),
            acquisition_times_seconds=selected.tolist(), methods=methods)
    result['separate_zero_noise_no_bias_unit_experiment'] = dict(
        generated_measurements='analytic omega at original synthetic acquisition times, covariance exactly zero',
        purpose='verify consistency of matched input and nonlinear integration; not the primary noisy reconstruction',
        primary_noisy_dataset_or_law_replaced=False,
        requested_rates_hz=list(requested_rates), rates=rate_results)
    return result
