#!/usr/bin/env python3
"""Plot recorded yaw motion and signed tracking residuals without filtering."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'success_yaw_comparison', HERE / 'compare_feedback_orientation.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_label(value):
    # Keep the existing condition identities; apply the current thesis typography.
    return value.replace('（', ' (').replace('）', ')')


def signed_yaw_error_deg(mode_wxyz, truth_wxyz):
    """Extrinsic xyz Euler yaw estimate minus reference, in [-180, 180)."""
    estimate = comparison.original.rotations_from_wxyz(mode_wxyz).as_euler(
        'xyz', degrees=True)[..., 2]
    reference = comparison.original.rotations_from_wxyz(truth_wxyz).as_euler(
        'xyz', degrees=True)[..., 2]
    return (estimate - reference + 180.0) % 360.0 - 180.0


def series_statistics(values, times):
    values = np.asarray(values, dtype=float)
    centered_time = times - times.mean()
    slope = float(centered_time @ (values - values.mean()) /
                  (centered_time @ centered_time))
    return dict(first_deg=float(values[0]), last_deg=float(values[-1]),
                mean_deg=float(values.mean()),
                absolute_rms_deg=float(np.sqrt(np.mean(values * values))),
                minimum_deg=float(values.min()), maximum_deg=float(values.max()),
                maximum_absolute_deg=float(np.abs(values).max()),
                end_minus_start_deg=float(values[-1] - values[0]),
                ordinary_least_squares_slope_deg_per_second=slope,
                signed_wrap_boundary_crossings=int(
                    np.count_nonzero(np.abs(np.diff(values)) > 180.0)))


def motion_statistics(arrays, reference='mocap quaternion spherical interpolation at the original IMU stamps'):
    rotations = comparison.original.rotations_from_wxyz(arrays['mocap_wxyz'])
    rpy = np.rad2deg(np.unwrap(rotations.as_euler('xyz'), axis=0))
    records = {}
    for component, name in enumerate(('roll', 'pitch', 'yaw')):
        values = rpy[:, component]
        records[name] = dict(
            first_deg=float(values[0]), last_deg=float(values[-1]),
            minimum_deg=float(values.min()), maximum_deg=float(values.max()),
            range_deg=float(np.ptp(values)),
            end_minus_start_deg=float(values[-1] - values[0]),
            accumulated_absolute_euler_motion_deg=float(np.abs(np.diff(values)).sum()))
    increments = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    gyro = arrays['gyro_rad_s']
    return dict(
        reference=reference,
        rpy_convention='extrinsic xyz; component traces unwrapped in time',
        rpy=records,
        accumulated_rotation_increment_deg=float(increments.sum()),
        maximum_rotation_increment_deg=float(increments.max()),
        measured_gyro_z_rad_per_second=dict(
            minimum=float(gyro[:, 2].min()), maximum=float(gyro[:, 2].max()),
            rms=float(np.sqrt(np.mean(gyro[:, 2] ** 2)))),
        interpretation=(
            'Recorded yaw excursion describes physical motion. Signed yaw residual '
            'describes tracking error; its endpoint change or fitted slope alone '
            'does not identify bias drift, sensor calibration error, or model error.'))


def create_diagnostics(input_directory, output_directory):
    paths = (input_directory / 'timeseries.npz',
             input_directory / 'madgwick_comparison' / 'timeseries.npz')
    arrays, marginal = [dict(np.load(path, allow_pickle=False)) for path in paths]
    times = arrays['time_seconds']
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('At least two strictly increasing measurement times are required.')
    for key in ('time_seconds', 'measurement_stamps', 'mocap_wxyz'):
        if not np.array_equal(arrays[key], marginal[key]):
            raise ValueError('The two saved comparisons use different ' + key + '.')
    if arrays['mode_wxyz'].shape != (len(times), 6, 4):
        raise ValueError('The source must contain the existing six Bingham conditions.')
    if marginal['mode_wxyz'].shape != (len(times), 4, 4):
        raise ValueError('The paired source must contain three Bingham conditions and Madgwick.')
    summary_path = input_directory / 'summary.json'
    paired_summary_path = input_directory / 'madgwick_comparison' / 'summary.json'
    summary = json.loads(summary_path.read_text())
    paired_summary = json.loads(paired_summary_path.read_text())
    reference_label = summary.get('orientation_reference_label', 'モーションキャプチャ')
    if list(summary['case_labels']) != list(comparison.CASE_IDS):
        raise ValueError('The saved six condition identities differ from the comparison.')
    if list(paired_summary['case_labels'])[-1] != 'madgwick_marg':
        raise ValueError('The final paired condition must be the nine-axis Madgwick filter.')
    modes = np.concatenate((arrays['mode_wxyz'], marginal['mode_wxyz'][:, -1:]), axis=1)
    if not np.isfinite(modes).all() or not np.isfinite(arrays['mocap_wxyz']).all():
        raise ValueError('Saved orientation values must be finite.')
    residuals = np.column_stack([
        signed_yaw_error_deg(modes[:, case], arrays['mocap_wxyz'])
        for case in range(7)])
    labels = list(comparison.CASE_LABELS) + [comparison.MADGWICK_LABEL]
    ids = list(comparison.CASE_IDS) + ['madgwick_marg']
    colors = list(comparison.CASE_COLORS) + [comparison.MADGWICK_COLOR]
    styles = list(comparison.CASE_STYLES) + [':']
    markers = list(comparison.CASE_MARKERS) + ['*']
    reference_rpy = np.rad2deg(np.unwrap(
        comparison.original.rotations_from_wxyz(arrays['mocap_wxyz']).as_euler('xyz'), axis=0))
    displayed_rpy = comparison.display_rpy(modes, reference_rpy)
    plt = comparison.configure_plotting()
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(11.4, 8.8))
    for case, label in enumerate(labels):
        for axis, values in zip(axes, (displayed_rpy[:, case, 2], residuals[:, case])):
            comparison.plot_condition(axis, times, values, case, colors[case],
                                      styles[case], markers[case], display_label(label))
    axes[0].plot(times, reference_rpy[:, 2], color='black', linestyle='--',
                 linewidth=1.5, label=reference_label)
    axes[1].axhline(0.0, color='#777777', linewidth=0.8)
    axes[0].set_ylabel('ヨー角 [deg]')
    axes[1].set_ylabel('符号付きヨー誤差 [deg]')
    axes[1].set_xlabel('窓開始からの経過時間 [s]')
    for axis in axes:
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.02),
                   frameon=False, fontsize=9.5, handlelength=4.3)
    figure.tight_layout()
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison.original.save_figure(figure, output_directory, 'yaw_motion_and_signed_error')
    plt.close(figure)
    report = dict(
        samples=len(times), displayed_start_seconds=float(times[0]),
        displayed_end_seconds=float(times[-1]),
        duration_seconds=float(times[-1] - times[0]),
        first_measurement_stamp=float(arrays['measurement_stamps'][0]),
        last_measurement_stamp=float(arrays['measurement_stamps'][-1]),
        signed_yaw_error_definition=(
            '(extrinsic-xyz Euler yaw(estimate) - Euler yaw(mocap) + 180) '
            'mod 360 - 180 degrees; range [-180, 180)'),
        error_evaluation='all saved IMU-aligned samples; equal-weight statistics; no smoothing or decimation',
        trend_diagnostic=(
            'Ordinary least-squares slope of the signed wrapped residual against '
            'measurement time. No trend line is added to the plot. A nonzero '
            'slope is not a separate estimate of gyro bias or causal drift. '
            'Wrap crossings, if present, make endpoint changes and linear slopes branch-dependent.'),
        mocap_motion=motion_statistics(arrays, summary.get('orientation_reference_description',
            'mocap quaternion spherical interpolation at the original IMU stamps')),
        statistics={name: dict(label=display_label(label),
                               **series_statistics(residuals[:, case], times))
                    for case, (name, label) in enumerate(zip(ids, labels))},
        source_files_sha256={str(path): sha256(path) for path in paths},
        source_summaries_sha256={str(path): sha256(path)
                                 for path in (summary_path, paired_summary_path)},
        helper_sha256={name: sha256(HERE / name) for name in (
            'compare_feedback_orientation.py', 'compare_rolling_bag.py')},
        plot_script_sha256=sha256(Path(__file__)),
        plot_all_samples=True, plotted_trend_line=False,
        coordinate_reference='same world-to-body convention and timestamps as the saved six-condition inference',
        figure_files=['yaw_motion_and_signed_error.' + suffix
                      for suffix in ('png', 'pdf', 'svg')])
    if summary.get('synthetic_data', False):
        report['truth_motion'] = report.pop('mocap_motion')
        report['coordinate_reference'] = 'synthetic body-to-world truth at the original sensor timestamps'
        report['signed_yaw_error_definition'] = report['signed_yaw_error_definition'].replace(
            'yaw(mocap)', 'yaw(synthetic truth)')
    (output_directory / 'drift_diagnostics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = create_diagnostics(args.input, args.output)
    print(json.dumps(dict(samples=report['samples'], duration_seconds=report['duration_seconds'],
                          output=str(args.output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
