#!/usr/bin/env python3
"""Compare the existing six filters on a separately saved synthetic flight."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

import bingham_moment_inverse
import compare_feedback_orientation as comparison
import plot_success_yaw_diagnostics as diagnostics
import static_vector_noise
from synthetic_flight_data import generate_flight


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assumed_sensor_noise(settings):
    """Construct the estimator's noise inputs without reading generated noise."""
    noise = {}
    for sensor, variance in settings['estimation_noise_variances'].items():
        noise[sensor] = dict(isotropic_variance=variance,
            isotropic_covariance=(variance*np.eye(3)).tolist(),
            source='configured estimator assumption; not calibrated from this realization')
        if sensor != 'angular_velocity':
            noise[sensor]['reference_magnitude'] = settings['reference_magnitudes'][sensor]
    return noise


def estimate_measurements(measurements, initial, reference, noise, progress=None):
    """Only readings, initial orientation, fixed references and assumed noise enter."""
    acceleration_weights = static_vector_noise.concentration_from_vector_noise(
        measurements['acceleration_m_s2'], noise['acceleration'])
    magnetic_weights = static_vector_noise.concentration_from_vector_noise(
        measurements['magnetic_raw'], noise['magnetic'])
    result = comparison.compare_six(measurements['gyro_rad_s'], measurements['acceleration_m_s2'],
        measurements['magnetic_raw'], measurements['measurement_stamps'], initial, reference,
        gyro_covariance=np.asarray(noise['angular_velocity']['isotropic_covariance']),
        acceleration_concentration=acceleration_weights, magnetic_concentration=magnetic_weights,
        propagation=bingham_moment_inverse.propagate, progress=progress)
    return result, acceleration_weights, magnetic_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--sample-rate', type=float, default=200.)
    parser.add_argument('--dataset', type=Path,
                        help='Reuse input_data.npz and simulation_settings.json in this directory.')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.dataset is None:
        data, settings = generate_flight(args.duration, args.sample_rate)
        np.savez_compressed(output/'input_data.npz', **data)
        (output/'simulation_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    else:
        data = dict(np.load(args.dataset/'input_data.npz', allow_pickle=False))
        settings = json.loads((args.dataset/'simulation_settings.json').read_text())
        if output.resolve() != args.dataset.resolve():
            for name in ('input_data.npz', 'simulation_settings.json'):
                shutil.copy2(args.dataset/name, output/name)
    times = data['measurement_stamps']
    truth = data['truth_wxyz']
    initial = truth[0].copy()
    reference = np.asarray(settings['magnetic_reference_world_unit'])
    noise = assumed_sensor_noise(settings)
    measurements = {key: data[key] for key in (
        'measurement_stamps', 'gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw')}
    (parameters, modes, carried, residuals), acc_weights, mag_weights = estimate_measurements(
        measurements, initial, reference, noise,
        progress=lambda k, total, seconds: print(
            'synthetic propagation {}/{}; {:.1f}s'.format(k, total, seconds), flush=True))
    axis_errors, errors, stats = comparison.evaluate(modes, truth)
    truth_rpy = np.rad2deg(np.unwrap(comparison.original.rotations_from_wxyz(truth).as_euler('xyz'), axis=0))
    # These two legacy names allow the unchanged comparison math to read its
    # reference arrays. They contain synthetic truth, never mocap measurements.
    arrays = dict(time_seconds=data['time_seconds'], measurement_stamps=times,
        measurement_stamp_ns=np.rint(times*1e9).astype(np.int64),
        gyro_rad_s=data['gyro_rad_s'], acceleration_m_s2=data['acceleration_m_s2'],
        magnetic_raw=data['magnetic_raw'], truth_wxyz=truth,
        mocap_wxyz=truth, mocap_rpy_deg=truth_rpy,
        position_time_seconds=times, position_measurement_stamps=times, position_m=data['position_m'],
        display_parameter=parameters, mode_wxyz=modes, carried_parameter=carried,
        moment_fit_residual=residuals, rpy_deg=comparison.display_rpy(modes, truth_rpy),
        rpy_error_deg=axis_errors, error_deg=errors,
        acceleration_direction_concentration=acc_weights, magnetic_direction_concentration=mag_weights)
    # The comparison helper needs only this one reference quaternion. Future
    # truth is deliberately unavailable to its propagation/update loop.
    madgwick_inputs = dict(measurements, mocap_wxyz=initial.reshape(1, 4))
    madgwick_modes = comparison.madgwick.run_madgwick(madgwick_inputs, reference, .1)[:, 1]
    paired_modes = np.concatenate((modes[:, comparison.ALL_CASES], madgwick_modes[:, None]), axis=1)
    paired_axis, paired_errors, paired_stats = comparison.evaluate(paired_modes, truth)
    paired = {key: arrays[key] for key in ('time_seconds', 'measurement_stamps',
        'measurement_stamp_ns', 'gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw',
        'mocap_wxyz', 'mocap_rpy_deg', 'truth_wxyz', 'acceleration_direction_concentration',
        'magnetic_direction_concentration')}
    paired.update(mode_wxyz=paired_modes, rpy_deg=comparison.display_rpy(paired_modes, truth_rpy),
                  rpy_error_deg=paired_axis, error_deg=paired_errors)
    paired_output = output/'madgwick_comparison'
    paired_output.mkdir(exist_ok=True)
    np.savez_compressed(output/'timeseries.npz', **arrays)
    np.savez_compressed(paired_output/'timeseries.npz', **paired)
    comparison.CASE_LABELS = tuple(label.replace('（', ' (').replace('）', ')')
                                   for label in comparison.CASE_LABELS)
    comparison.MADGWICK_LABEL = comparison.MADGWICK_LABEL.replace('（', ' (').replace('）', ')')
    source_names = ['compare_synthetic_flight.py', 'synthetic_flight_data.py', 'bingham_moment_inverse.py',
                    'compare_feedback_orientation.py', 'compare_rolling_bag.py',
                    'compare_madgwick_rolling.py', 'static_vector_noise.py']
    metadata = dict(synthetic_data=True, orientation_reference_label='人工真値',
        orientation_reference_description='known synthetic body-to-world quaternion at each sensor timestamp',
        legacy_reference_array_names='mocap_wxyz and mocap_rpy_deg contain synthetic truth, not measured mocap',
        dataset_file='input_data.npz', dataset_sha256=sha256(output/'input_data.npz'),
        settings_file='simulation_settings.json', settings_sha256=sha256(output/'simulation_settings.json'),
        assumed_sensor_noise=noise, initial_quaternion_wxyz=initial.tolist(),
        initial_bingham_concentration=100., integration_steps=120,
        reference_magnetic_world_unit=reference.tolist(), gyro_bias_removed=False,
        gyro_sample_covariance_rad2_s2=noise['angular_velocity']['isotropic_covariance'],
        evaluated_imu_samples=len(times), position_samples=len(times),
        duration_seconds=float(times[-1]-times[0]),
        truth_in_estimator='initial quaternion and fixed world reference vectors only; no later pose or generating covariance',
        physical_acceleration_model='R.T (position_second_derivative - gravity_world)',
        estimation_acceleration_model='existing world +z gravity-direction likelihood; no translation compensation',
        maximum_translation_acceleration_m_s2=float(np.linalg.norm(data['acceleration_world_m_s2'], axis=1).max()),
        moment_inverse='same scaled Bingham moment equations and analytic Jacobian in all three groups',
        maximum_moment_fit_residual=float(residuals.max()),
        statistics=dict(zip(comparison.CASE_IDS, stats)),
        case_labels=dict(zip(comparison.CASE_IDS, comparison.CASE_LABELS)),
        output_and_retained_evidence_coefficients=comparison.COEFFICIENTS.tolist(),
        carried_group_ids=comparison.GROUP_IDS, case_carried_groups=dict(zip(comparison.CASE_IDS, comparison.CASE_GROUPS)),
        coordinate_colors=dict(zip(('x', 'y', 'z'), comparison.COORDINATE_COLORS)),
        source_files_sha256={name: sha256(Path(__file__).parent/name) for name in source_names},
        direction_concentration_statistics={sensor: dict(minimum=float(values.min()),
            median=float(np.median(values)), maximum=float(values.max()))
            for sensor, values in [('acceleration', acc_weights), ('magnetic', mag_weights)]})
    comparison.plot_orientation(arrays, output, reference_label='人工真値')
    comparison.plot_errors(arrays, output, comparison.CASE_LABELS,
                           comparison.CASE_COLORS, comparison.CASE_STYLES, 'axis_errors')
    metadata['raw_plot_limits'] = comparison.plot_raw_measurements(arrays, output)
    labels = [comparison.CASE_LABELS[i] for i in comparison.ALL_CASES]+[comparison.MADGWICK_LABEL]
    comparison.plot_errors(paired, paired_output, labels,
        [comparison.CASE_COLORS[i] for i in comparison.ALL_CASES]+[comparison.MADGWICK_COLOR],
        [comparison.CASE_STYLES[i] for i in comparison.ALL_CASES]+[':'], 'method_comparison')
    (output/'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    paired_metadata = dict(metadata, statistics=dict(zip(
        [comparison.CASE_IDS[i] for i in comparison.ALL_CASES]+['madgwick_marg'], paired_stats)),
        case_labels=dict(zip([comparison.CASE_IDS[i] for i in comparison.ALL_CASES]+['madgwick_marg'], labels)),
        madgwick_beta=.1)
    (paired_output/'summary.json').write_text(json.dumps(paired_metadata, ensure_ascii=False, indent=2)+'\n')
    diagnostics.create_diagnostics(output, output)
    print(json.dumps(dict(output=str(output), samples=len(times), statistics=metadata['statistics'],
                          wall_seconds=time.monotonic()-started), indent=2), flush=True)


if __name__ == '__main__':
    main()
