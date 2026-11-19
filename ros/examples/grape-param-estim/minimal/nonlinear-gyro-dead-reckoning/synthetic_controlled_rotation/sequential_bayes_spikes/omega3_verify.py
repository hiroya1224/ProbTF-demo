#!/usr/bin/env python3
"""Independent post-run diagnostics for exact body-omega x3 with Q=0.

Finite differences are recorded, never used to reject observations or stop
subsequent filter updates. This module does not rerun a test suite, change a
candidate distribution, regenerate sensor noise, or call an older verifier's
run function with historical test-count claims.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
MODELS = ('nonlinear', 'linear_forced')


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load('_omega3_verification_helpers', 'rate_comparison_verify.py')
sha = helpers.sha
load_npz = helpers.load_npz
write_json = helpers.write_json
difference = helpers.difference
finite_max = helpers.finite_max


def refresh_manifest(output):
    """Include completed files only; cache trees are not publications."""
    return helpers.refresh_manifest(Path(output))


def source_checks(source, sensor, acquired, original, original_sensor, config):
    scale = float(config['body_omega_scale'])
    spike_scale = float(config.get('spike_scale', 1.))
    spike_amplitude_keys = ('spike_vector_rad_s', 'event_vector_body_rad_s',
                            'event_magnitude_deg_s')
    scaled = {name: difference(source[name], scale*original[name]) for name in (
        'gyro_true_body_rad_s', 'reference_omega_body_rad_s', 'reference_alpha_body_rad_s2')}
    scaled['reference_gyroscopic_torque_body_nm'] = difference(
        source['reference_gyroscopic_torque_body_nm'],
        scale**2*original['reference_gyroscopic_torque_body_nm'])
    preserved_keys = ('time', 'imu_time', 'imu_source_index', 'imu_stamp_ns', 'imu_record_ns',
        'gyro_realized_noise_rad_s', 'gyro_bias_rad_s', 'generator_gyro_covariance_rad2_s2',
        'initial_quaternion_wxyz', 'inertia_body_kg_m2', 'reference_time', 'torque_time')
    preserved = {key: bool(np.array_equal(source[key], original[key])) for key in preserved_keys}
    sensor_preserved = {key: bool(np.array_equal(value, original_sensor[key]))
        for key, value in sensor.items()
        if key not in ('gyro_white', 'gyro_spike', *spike_amplitude_keys)}
    spike_amplitude_checks = {key: difference(sensor[key], spike_scale*original_sensor[key])
                             for key in spike_amplitude_keys}
    # Reconstruct all pulse overlaps from their event vectors, independent of
    # the saved native spike vector. Halving pulse heights must not remove
    # events, move endpoints, or alter the acquisition schedule.
    pulse_sum = np.zeros_like(sensor['spike_vector_rad_s'])
    active_count = np.zeros(len(pulse_sum), dtype=np.int64)
    for start, stop, vector in zip(sensor['event_native_start_array_index'],
            sensor['event_native_stop_array_index_exclusive'], sensor['event_vector_body_rad_s']):
        pulse_sum[int(start):int(stop)] += vector
        active_count[int(start):int(stop)] += 1
    native_rate = 1./np.median(np.diff(sensor['time']))
    rate = float(config['acquisition_rate_hz'])
    stride = int(round(native_rate/rate))
    expected_index = np.arange(0, len(sensor['time']), stride)
    index = np.asarray(acquired['native_array_index'], int)
    acquired_equal = {key: bool(np.array_equal(acquired[key], sensor[key][index])) for key in (
        'time', 'stamp_ns', 'record_ns', 'source_index', 'gyro_spike', 'spike_mask', 'spike_vector_rad_s')}
    inertia = source['inertia_body_kg_m2']
    analytic = load('_omega3_verification_base_motion', '../trajectory.py')
    parameters = {key: original['motion_'+key] for key in (
        'amplitude_deg', 'frequency_hz', 'phase_rad', 'offset_deg')}
    base = analytic.trajectory(original['torque_time'], str(original['motion_profile']), parameters)
    omega = scale*base['omega_body_rad_s']
    alpha = scale*base['alpha_body_rad_s2']
    gyroscopic = np.cross(omega, omega@inertia.T)
    expected_torque = alpha@inertia.T+gyroscopic
    baseline_gyroscopic = np.cross(base['omega_body_rad_s'], base['omega_body_rad_s']@inertia.T)
    return dict(body_omega_scale=scale, expected_scale=3., spike_scale=spike_scale,
        scaling_checks=scaled, unchanged_source_arrays=preserved,
        unchanged_native_disturbances_and_clocks=sensor_preserved,
        spike_amplitude_vs_declared_scale=spike_amplitude_checks,
        spike_vector_vs_sum_of_saved_pulses=difference(sensor['spike_vector_rad_s'], pulse_sum),
        pulse_active_count_exact=bool(np.array_equal(sensor['active_spike_count'], active_count)),
        pulse_mask_exact=bool(np.array_equal(sensor['spike_mask'], active_count>0)),
        pulse_event_count=len(sensor['event_vector_body_rad_s']),
        pulse_event_count_unchanged=len(sensor['event_vector_body_rad_s'])==len(original_sensor['event_vector_body_rad_s']),
        time_or_frequency_rescaled=False,
        base_frequency_coefficients_unchanged=bool(np.array_equal(
            source['base_motion_frequency_hz'], original['motion_frequency_hz'])),
        gyro_white_from_new_truth_plus_original_bias_and_noise=difference(sensor['gyro_white'],
            source['gyro_true_body_rad_s']+original['gyro_bias_rad_s']+original['gyro_realized_noise_rad_s']),
        gyro_spike_from_new_white_plus_scaled_original_pulses=difference(sensor['gyro_spike'],
            sensor['gyro_white']+spike_scale*original_sensor['spike_vector_rad_s']),
        nominal_native_rate_hz=native_rate, requested_acquisition_rate_hz=rate,
        native_index_stride=stride, expected_acquisition_count=len(expected_index),
        selected_native_index_exact=bool(np.array_equal(index, expected_index)),
        selected_values_and_clocks_exact=acquired_equal,
        selected_first_time_seconds=float(acquired['time'][0]),
        selected_last_time_seconds=float(acquired['time'][-1]),
        new_gyro_and_alpha_are_body_coordinates=True,
        torque_from_new_body_omega_and_alpha=difference(source['torque_body_nm'], expected_torque),
        torque_relation_to_original=difference(source['torque_body_nm'],
            scale*original['torque_body_nm']+(scale**2-scale)*baseline_gyroscopic),
        gyroscopic_term_at_control_times_vs_9x_original=difference(gyroscopic, 9.*baseline_gyroscopic),
        original_noise_white_covariance=difference(source['generator_gyro_covariance_rad2_s2'],
            np.eye(3)*np.deg2rad(2.)**2),
        source_reference_quaternion_maximum_norm_error=finite_max(
            np.linalg.norm(source['reference_quaternion_xyzw'], axis=1)-1.))


def dynamics_checks(source, config, physical, engine):
    """Read-only truth consistency and RK4 refinement, outside both filters."""
    times = source['reference_time']
    initial = source['reference_omega_body_rad_s'][:1]
    q0 = source['initial_quaternion_wxyz']
    normal_step = float(config['max_step_seconds'])
    states = {}
    reports = {}
    reference = Rotation.from_quat(source['reference_quaternion_xyzw'])
    for label, step in (('configured_step', normal_step), ('half_step', normal_step/2.)):
        flow = engine._adaptive.ControlledFlow(physical, q0, *source['time'],
            max_step=step, mode='nonlinear')
        state = flow.evaluate(initial, times)[:, 0]
        states[label] = state
        attitude_error = (reference.inv()*Rotation.from_quat(state[:, [4, 5, 6, 3]])).magnitude()
        reports[label] = dict(maximum_step_seconds=step,
            maximum_body_omega_difference_deg_s=finite_max(np.rad2deg(
                state[:, :3]-source['reference_omega_body_rad_s'])),
            maximum_orientation_difference_deg=finite_max(np.rad2deg(attitude_error)),
            maximum_quaternion_norm_correction=float(flow.maximum_norm_correction))
    comparison = (Rotation.from_quat(states['configured_step'][:, [4, 5, 6, 3]]).inv()
        *Rotation.from_quat(states['half_step'][:, [4, 5, 6, 3]])).magnitude()
    return dict(scope='Generator/known-input consistency only; no reference data are passed to either Bayesian filter',
        reference_initial_omega_used_only_for_this_separate_check=True,
        individual_steps=reports,
        configured_vs_half_step_maximum_omega_difference_deg_s=finite_max(np.rad2deg(
            states['configured_step'][:, :3]-states['half_step'][:, :3])),
        configured_vs_half_step_maximum_orientation_difference_deg=finite_max(np.rad2deg(comparison)),
        finite_differences_are_diagnostics_not_acceptance_gates=True)


def plot_checks(output, source, sensor, acquired, cases):
    record_path = output/'omega3_plotting.json'
    data_path = output/'omega3_plotted_data.npz'
    if not record_path.exists() or not data_path.exists():
        return dict(available=False, reason='Saved plot record/data not present yet')
    record = json.loads(record_path.read_text())
    data = load_npz(data_path)
    settings = record['settings']
    checks = dict(available=True,
        renderer_hash_matches_current_source=record['renderer_sha256']==sha(HERE/'omega3_plotting.py'),
        native_count_matches=settings['native_reading_count']==len(sensor['time']),
        acquisition_count_matches=settings['selected_acquisition_count']==len(acquired['time']),
        process_covariance_display=record['settings'].get('model_process_covariance'),
        all_native_samples_drawn=record.get('all_native_samples_drawn'),
        all_saved_moment_rows_drawn=record.get('all_saved_moment_rows_drawn'),
        smoothing=record.get('smoothing'), covariance_adjustment=record.get('covariance_adjustment'),
        models={})
    for mode, case in cases.items():
        stamp = case['time']
        truth_omega = np.column_stack([np.interp(stamp, source['reference_time'],
            source['reference_omega_body_rad_s'][:, axis]) for axis in range(3)])
        truth_rpy = np.column_stack([np.interp(stamp, source['reference_time'],
            source['reference_rpy_rad'][:, axis]) for axis in range(3)])
        rpy_error = np.rad2deg(np.arctan2(np.sin(case['rpy_proxy_rad']-truth_rpy),
            np.cos(case['rpy_proxy_rad']-truth_rpy)))
        omega_error = np.rad2deg(case['omega_mean_rad_s']-truth_omega)
        so3_error = np.rad2deg((Rotation.from_euler('xyz', truth_rpy).inv()
            *Rotation.from_euler('xyz', case['rpy_proxy_rad'])).magnitude())
        rpy_width = 2*np.rad2deg(np.sqrt(np.diagonal(case['rpy_covariance_delta_rad2'], axis1=1, axis2=2)))
        omega_width = 2*np.rad2deg(np.sqrt(np.diagonal(case['omega_covariance_rad2_s2'], axis1=1, axis2=2)))
        checks['models'][mode] = dict(
            saved_time_exact=bool(np.array_equal(data[mode+'_time'], stamp)),
            plotted_omega_two_sigma_vs_saved_covariance=difference(data[mode+'_omega_two_sigma_deg_s'], omega_width),
            plotted_rpy_two_sigma_vs_saved_covariance=difference(data[mode+'_rpy_two_sigma_deg'], rpy_width),
            plotted_omega_error_uses_new_truth=difference(data[mode+'_omega_error_deg_s'], omega_error),
            plotted_rpy_error_uses_new_truth=difference(data[mode+'_rpy_error_deg'], rpy_error),
            plotted_so3_error_uses_new_truth=difference(data[mode+'_so3_proxy_error_deg'], so3_error),
            stored_moment_row_count_matches=settings['stored_moment_rows'][mode]==len(stamp))
    return checks


def run(args):
    begun = time.monotonic()
    output = args.output.resolve()
    baseline = args.baseline.resolve()
    config = json.loads((output/'config.json').read_text())
    source, sensor, acquired = [load_npz(output/name) for name in (
        'source_input.npz', 'sensor_input.npz', 'acquisitions.npz')]
    original, original_sensor = [load_npz(baseline/name) for name in (
        'source_input.npz', 'sensor_input.npz')]
    summary = json.loads((output/'summary.json').read_text())
    cases = {mode: load_npz(output/f'bayes_{mode}.npz') for mode in MODELS}
    metadata = summary['engine_metadata']
    engine = load('_omega3_independent_q0_engine', 'full_rate_engine.py')
    shared = load('_omega3_independent_torque', 'run.py')
    physical = dict(inertia=source['inertia_body_kg_m2'],
        torque=shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
    baseline_hashes = summary.get('baseline_input_file_sha256', {})
    baseline_current = {name: sha(baseline/name) for name in baseline_hashes}
    input_hashes = summary.get('input_file_sha256', {})
    input_current = {name: sha(output/name) for name in input_hashes}
    expected_count = int(round((source['time'][1]-source['time'][0])*config['acquisition_rate_hz']))+1
    report = dict(schema='grape/omega3-q0-independent-postrun-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), post_run_only=True,
        observations_or_filter_state_changed=False,
        finite_differences_are_diagnostics_not_observation_gates=True,
        verifier_sha256=sha(__file__),
        source_scaling_and_disturbances=source_checks(source, sensor, acquired, original, original_sensor, config),
        baseline_input_hashes=dict(recorded=baseline_hashes, current=baseline_current,
            unchanged_since_generation=baseline_hashes==baseline_current if baseline_hashes else None),
        new_input_hashes=dict(recorded=input_hashes, current=input_current,
            unchanged_since_run_start=input_hashes==input_current if input_hashes else None),
        declared_process_covariance_rad2_s2=config.get('process_covariance_rad2_s2'),
        declared_process_covariance_exactly_zero=bool(np.array_equal(
            np.asarray(config.get('process_covariance_rad2_s2')), np.zeros((3,3)))),
        numerical_test_provenance=dict(tests_run_by_this_verifier=False,
            counts_or_status_not_inferred_from_historical_runs=True),
        code_sha256={path.name: sha(path) for pattern in ('omega3*.py', 'full_rate*.py', 'full_rate*.cpp')
            for path in sorted(HERE.glob(pattern))},
        unchanged_included_physics_sha256=sha(HERE/'native_flow.cpp'), models={})
    for mode, case in cases.items():
        print(f'Verify omega x3 Q=0: {mode}, {len(acquired["time"])} selected readings', flush=True)
        item = dict(structure=helpers.structure(case, acquired, expected_count),
            covariance=helpers.covariance_diagnostics(case),
            recorded_engine_metadata=metadata[mode],
            declared_measurement_covariance_comparison=difference(case['measurement_covariance_rad2_s2'],
                np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2),
            q0_process_noise_metadata=metadata[mode].get('process_noise', 'not separately recorded'),
            independent_original_prior_and_all_factors=helpers.full_target_check(
                case, source, acquired, config, physical, mode, metadata[mode], engine))
        if 'process_covariance_rad2_s2' in case:
            item['saved_process_covariance_comparison'] = difference(case['process_covariance_rad2_s2'], np.zeros((3,3)))
        if mode=='linear_forced':
            item['analytic_linear_angular_reference'] = helpers.analytic_linear(
                case, source, acquired, config, physical, engine)
        report['models'][mode] = item
    report['both_models_same_measurement_covariance'] = bool(np.array_equal(
        cases['nonlinear']['measurement_covariance_rad2_s2'], cases['linear_forced']['measurement_covariance_rad2_s2']))
    report['known_input_nonlinear_dynamics'] = dynamics_checks(source, config, physical, engine)
    report['plotted_saved_numeric_identity'] = plot_checks(output, source, sensor, acquired, cases)
    files = [path for path in output.rglob('*') if path.is_file()
             and 'cache' not in path.relative_to(output).parts]
    report['publication_files'] = dict(maximum_bytes=max((path.stat().st_size for path in files), default=0),
        all_files_below_50000000_bytes=all(path.stat().st_size<50_000_000 for path in files),
        intermediate_cache_excluded=True)
    if args.test_evidence is not None:
        report['numerical_test_provenance']['supplied_evidence'] = json.loads(args.test_evidence.read_text())
        report['numerical_test_provenance']['supplied_evidence_sha256'] = sha(args.test_evidence)
    report['wall_seconds'] = time.monotonic()-begun
    write_json(output/'verification.json', report)
    refresh_manifest(output)
    print('Wrote '+str(output/'verification.json'), flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=HERE/'omega3_q0_publication')
    parser.add_argument('--baseline', type=Path, default=HERE/'frequent_publication')
    parser.add_argument('--test-evidence', type=Path,
        help='Optional actual test execution record supplied by the caller; no test counts are assumed')
    run(parser.parse_args())
