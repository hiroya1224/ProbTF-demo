#!/usr/bin/env python3
"""Compare the nine-axis Bingham fusion with plain nine-axis Madgwick.

The normalized-gradient update follows Madgwick's 2010 report and the author's
2011 MATLAB equations.  Beta=0.1 matches the author's distributed example.
The only extra orientation operation is a fixed heading-frame conversion based
on the original prefix reference.  No evaluated mocap enters either filter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def repository_root():
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / 'ros/core/probtf_core/src/probtf_estimators').is_dir():
            return candidate
    raise RuntimeError('Run this analysis in a ProbTF-demo checkout.')


ROOT = repository_root()
sys.path.insert(0, str(ROOT / 'ros/examples/probtf_orientation_demo/scripts'))
import compare_rolling_bag as original

CASE_IDS = ('nine_axis', 'madgwick_marg')
CASE_LABELS = ('9軸の全情報の融合', 'Madgwick フィルタ（9軸）')
CASE_COLORS = (original.CASE_COLORS[4], '#000000')
CASE_STYLES = ('-', '--')
REPORT_URL = 'https://x-io.co.uk/downloads/madgwick_internal_report.pdf'
MATLAB_URL = 'https://x-io.co.uk/downloads/madgwick_algorithm_matlab.zip'


def quaternion_product(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([lw*rw-lx*rx-ly*ry-lz*rz,
                     lw*rx+lx*rw+ly*rz-lz*ry,
                     lw*ry-lx*rz+ly*rw+lz*rx,
                     lw*rz+lx*ry-ly*rx+lz*rw])


def direction_residual_jacobian(q, direction, observation):
    """R(q).T direction - observation, with the direction held fixed in J.

    The usual unit-quaternion polynomial is used off the sphere to take the
    same ambient derivative as Madgwick's published gradient step.
    """
    w, x, y, z = q
    dx, dy, dz = direction
    prediction = np.array([
        dx*(1-2*y*y-2*z*z) + 2*dy*(x*y+w*z) + 2*dz*(x*z-w*y),
        2*dx*(x*y-w*z) + dy*(1-2*x*x-2*z*z) + 2*dz*(y*z+w*x),
        2*dx*(x*z+w*y) + 2*dy*(y*z-w*x) + dz*(1-2*x*x-2*y*y),
    ])
    jacobian = np.array([
        [2*dy*z-2*dz*y, 2*dy*y+2*dz*z,
         -4*dx*y+2*dy*x-2*dz*w, -4*dx*z+2*dy*w+2*dz*x],
        [-2*dx*z+2*dz*x, 2*dx*y-4*dy*x+2*dz*w,
         2*dx*x+2*dz*z, -2*dx*w-4*dy*z+2*dz*y],
        [2*dx*y-2*dy*x, 2*dx*z-2*dy*w-4*dz*x,
         2*dx*w+2*dy*z-4*dz*y, 2*dx*x+2*dy*y],
    ])
    return prediction - observation, jacobian


def madgwick_update(q, gyro, acceleration, dt, magnetic=None, beta=0.1):
    """Forward-Euler quaternion update with fixed normalized-gradient gain."""
    q = np.asarray(q, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    if q.shape != (4,) or gyro.shape != (3,) or acceleration.shape != (3,):
        raise ValueError('Expected quaternion (4,) and gyro/acceleration (3,).')
    if not np.all(np.isfinite(np.r_[q, gyro, acceleration, dt, beta])):
        raise ValueError('Inputs must be finite.')
    if dt < 0 or beta < 0 or np.linalg.norm(q) <= 1e-12:
        raise ValueError('dt/beta must be non-negative and q non-zero.')
    q = q / np.linalg.norm(q)
    derivative = 0.5 * quaternion_product(q, np.r_[0.0, gyro])
    acceleration_norm = np.linalg.norm(acceleration)
    if acceleration_norm > 1e-12 and beta > 0:
        residual, jacobian = direction_residual_jacobian(
            q, [0.0, 0.0, 1.0], acceleration/acceleration_norm)
        gradient = jacobian.T @ residual
        if magnetic is not None:
            magnetic = np.asarray(magnetic, dtype=float)
            if magnetic.shape != (3,) or not np.all(np.isfinite(magnetic)):
                raise ValueError('magnetic must be a finite vector (3,).')
            magnetic_norm = np.linalg.norm(magnetic)
            if magnetic_norm > 1e-12:
                observed = magnetic/magnetic_norm
                conjugate = q * np.array([1.0, -1.0, -1.0, -1.0])
                field = quaternion_product(quaternion_product(q, np.r_[0.0, observed]), conjugate)[1:]
                direction = [np.hypot(field[0], field[1]), 0.0, field[2]]
                residual, jacobian = direction_residual_jacobian(q, direction, observed)
                gradient += jacobian.T @ residual
        magnitude = np.linalg.norm(gradient)
        # The published step is undefined at an exact optimum; its limit is
        # handled without dividing by zero or introducing a correction.
        if magnitude > 1e-12:
            derivative -= beta * gradient/magnitude
    result = q + dt*derivative
    return result/np.linalg.norm(result)


def heading_frame(reference):
    reference = np.asarray(reference, dtype=float)
    if np.hypot(reference[0], reference[1]) <= 1e-12:
        raise ValueError('Magnetic north needs a nonzero horizontal component.')
    heading = np.arctan2(reference[1], reference[0])
    return Rotation.from_euler('z', heading), float(heading)


def run_madgwick(arrays, magnetic_reference, beta=0.1):
    """Use only the initial orientation, prefix field and live sensor samples."""
    world_from_north, _ = heading_frame(magnetic_reference)
    initial_world = original.rotations_from_wxyz(arrays['mocap_wxyz'][0])
    initial_north = original.wxyz_from_rotations(world_from_north.inv()*initial_world)
    timestamps = arrays['measurement_stamps']
    if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Measurement stamps must be nonempty and strictly increasing.')
    result = np.empty((len(timestamps), 2, 4))
    states = [initial_north.copy(), initial_north.copy()]
    result[0] = arrays['mocap_wxyz'][0]
    for k in range(1, len(timestamps)):
        interval = timestamps[k]-timestamps[k-1]
        for case in (0, 1):
            states[case] = madgwick_update(
                states[case], arrays['gyro_rad_s'][k], arrays['acceleration_m_s2'][k], interval,
                arrays['magnetic_raw'][k] if case else None, beta)
            north_attitude = original.rotations_from_wxyz(states[case])
            result[k, case] = original.wxyz_from_rotations(world_from_north*north_attitude)
    for case in (0, 1):
        result[:, case] = original.continuous_quaternions(result[:, case])
    return result


def statistics(modes, truth, times):
    axis_error, combined = original.rotation_error_series(modes, truth)
    first_minute = times <= 60.0
    records = []
    for case in range(modes.shape[1]):
        item = {'rms_deg': float(np.sqrt(np.mean(combined[:, case]**2))),
                'rpy_rms_deg': np.sqrt(np.mean(axis_error[:, case]**2, axis=0)).tolist(),
                'first_60_seconds_rms_deg': float(np.sqrt(np.mean(combined[first_minute, case]**2))),
                'first_60_seconds_yaw_rms_deg': float(np.sqrt(np.mean(axis_error[first_minute, case, 2]**2))),
                'final_deg': float(combined[-1, case])}
        records.append(item)
    return records, axis_error, combined


def magnetic_heading_control(arrays, reference):
    """Mocap-tilt-compensated compass: an input diagnostic, not a filter."""
    observed = arrays['magnetic_raw']
    observed = observed / np.linalg.norm(observed, axis=1)[:, None]
    field_world = original.rotations_from_wxyz(arrays['mocap_wxyz']).apply(observed)
    field_heading = np.arctan2(field_world[:, 1], field_world[:, 0])
    _, prefix_heading = heading_frame(reference)
    heading_change = (np.rad2deg(field_heading-prefix_heading)+180.0) % 360.0 - 180.0
    # Rotating the measured field by the reference roll/pitch alone gives the
    # same compass estimate: reference heading minus this field-heading change.
    compass_yaw = arrays['mocap_rpy_deg'][:, 2]-heading_change
    return field_world, compass_yaw, np.abs(heading_change)


def plot_comparison(arrays, output):
    """Four rows, one column, and exactly the two nine-axis estimates."""
    plt = original.configure_matplotlib()
    times = arrays['time_seconds']
    for logarithmic in (False, True):
        figure, axes = plt.subplots(4, 1, sharex=True, figsize=(8.1, 8.4))
        for component, axis in enumerate(axes):
            values = arrays['rpy_error_deg'][:, :, component] if component < 3 else arrays['error_deg']
            if logarithmic:
                axis.set_yscale('log', nonpositive='mask')
            for case in range(2):
                plotted = np.ma.masked_less_equal(values[:, case], 0.0) if logarithmic else values[:, case]
                axis.plot(times, plotted, color=CASE_COLORS[case], ls=CASE_STYLES[case],
                          lw=1.0, label=CASE_LABELS[case])
            axis.set_ylabel(('ロール誤差', 'ピッチ誤差', 'ヨー誤差', '3軸の合成誤差')[component]+' / deg')
            if not logarithmic:
                axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.22)
            axis.set_xlim(times[0], times[-1])
        axes[0].legend(ncol=1, loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False)
        axes[-1].set_xlabel('経過時間 / s')
        figure.tight_layout()
        original.save_figure(figure, output, 'method_comparison_semilog' if logarithmic else 'method_comparison')
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_five_conditions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--render-only', action='store_true', help='Select the two nine-axis estimates from saved comparison results.')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Use a separate output folder.')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = dict(np.load(args.input/'timeseries.npz'))
    metadata = json.loads((args.input/'summary.json').read_text())
    input_keys = ('time_seconds', 'measurement_stamps', 'gyro_rad_s', 'acceleration_m_s2',
                  'magnetic_raw', 'mocap_wxyz', 'mocap_rpy_deg')
    arrays = {key: base[key] for key in input_keys}
    if args.render_only:
        saved_ids = list(metadata['case_labels'])
        if any(case not in saved_ids for case in CASE_IDS):
            raise ValueError('--render-only needs saved nine_axis and madgwick_marg estimates.')
        selected = [saved_ids.index(case) for case in CASE_IDS]
        for key in ('mode_wxyz', 'rpy_deg', 'rpy_error_deg', 'error_deg'):
            arrays[key] = base[key][:, selected].copy()
        beta = metadata['madgwick_beta']
        inference_sha = metadata.get('inference_script_sha256', metadata.get('script_sha256'))
    else:
        if list(metadata['case_labels']) != list(original.CASE_IDS):
            raise ValueError('Fresh inference needs the five-condition Bingham result.')
        reference = np.array(metadata['reference_magnetic_world_unit'])
        # The second returned sequence is the original plain MARG update.
        madgwick = run_madgwick(base, reference, args.beta)[:, 1]
        modes = np.stack((base['mode_wxyz'][:, 4], madgwick), axis=1)
        records, axis_error, combined = statistics(modes, base['mocap_wxyz'], base['time_seconds'])
        rpy = np.stack([original.rotations_from_wxyz(modes[:, case]).as_euler('xyz', degrees=True)
                        for case in range(2)], axis=1)
        rpy += 360*np.round((base['mocap_rpy_deg'][:, None]-rpy)/360)
        arrays.update(mode_wxyz=modes, rpy_deg=rpy, rpy_error_deg=axis_error, error_deg=combined)
        beta = args.beta
        inference_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records, verified_axes, verified_combined = statistics(
        arrays['mode_wxyz'], arrays['mocap_wxyz'], arrays['time_seconds'])
    np.testing.assert_allclose(arrays['rpy_error_deg'], verified_axes, atol=1e-12)
    np.testing.assert_allclose(arrays['error_deg'], verified_combined, atol=1e-12)
    heading = metadata.get('world_from_magnetic_north_heading_deg')
    if heading is None:
        _, angle = heading_frame(metadata['reference_magnetic_world_unit'])
        heading = float(np.rad2deg(angle))
    report = {
        'bag_path': metadata['bag_path'], 'evaluated_imu_samples': len(arrays['time_seconds']),
        'duration_seconds': metadata['duration_seconds'], 'case_labels': dict(zip(CASE_IDS, CASE_LABELS)),
        'statistics': dict(zip(CASE_IDS, records)), 'madgwick_beta': beta,
        'madgwick_reference': REPORT_URL, 'equation_validation_source': MATLAB_URL,
        'madgwick_algorithm': 'normalized-gradient MARG equations; forward Euler; normalize output; no bias adaptation, gating, smoothing or gain schedule',
        'gyro_sample_convention': metadata.get('gyro_sample_convention', metadata.get('sample_convention')),
        'initialization': 'same initial mocap orientation; no evaluated reference orientation enters the recursive filters',
        'world_from_magnetic_north_heading_deg': heading,
        'heading_frame_source': 'fixed magnetic reference from the initial 1 s / 200 samples',
        'input_timeseries_sha256': hashlib.sha256((args.input/'timeseries.npz').read_bytes()).hexdigest(),
        'input_summary_sha256': hashlib.sha256((args.input/'summary.json').read_bytes()).hexdigest(),
        'inference_script_sha256': inference_sha,
        'evaluation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selected_from_saved_estimates': args.render_only,
        'figure_layout': 'four rows, one column, two nine-axis curves; linear and semilogarithmic y axes',
    }
    np.savez_compressed(args.output/'timeseries.npz', **arrays)
    plot_comparison(arrays, args.output)
    report['wall_seconds'] = time.monotonic()-started
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['statistics'], ensure_ascii=False, indent=2), flush=True)
    print('completed: '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
