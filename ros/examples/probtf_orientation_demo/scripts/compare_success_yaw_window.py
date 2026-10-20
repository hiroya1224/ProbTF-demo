#!/usr/bin/env python3
"""Replay the existing six filters from the PID success bag, then crop a window.

Only the bag adapter and output window differ from the existing comparison.
The reference-only window selection is supplied as an immutable JSON record.
No estimator is initialized again at the beginning of the displayed window.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import compare_feedback_orientation as comparison
import static_vector_noise
import bingham_moment_inverse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def display_label(label):
    return label.replace('（', ' (').replace('）', ')')


def read_native_measurements(path):
    import rosbag
    imu, mocap, frames = [], [], set()
    imu_ns, record_ns = [], []
    with rosbag.Bag(str(path), 'r') as bag:
        bag_start = bag.get_start_time()
        for topic, msg, stamp in bag.read_messages(topics=[
                '/gimbalrotor/imu', '/gimbalrotor/mocap/pose']):
            if topic == '/gimbalrotor/imu':
                imu.append([stamp.to_sec(), msg.stamp.to_sec(),
                            *msg.gyro, *msg.acc, *msg.mag])
                imu_ns.append(msg.stamp.to_nsec())
                record_ns.append(stamp.to_nsec())
            else:
                q, p = msg.pose.orientation, msg.pose.position
                mocap.append([stamp.to_sec(), msg.header.stamp.to_sec(),
                              q.w, q.x, q.y, q.z, p.x, p.y, p.z])
                frames.add(msg.header.frame_id)
    imu, mocap = np.asarray(imu), np.asarray(mocap)
    if (imu.ndim != 2 or mocap.ndim != 2 or len(imu) < 2 or len(mocap) < 2
            or not np.isfinite(imu).all() or not np.isfinite(mocap).all()
            or np.any(np.diff(imu[:, 1]) <= 0) or np.any(np.diff(mocap[:, 1]) <= 0)
            or frames != {'world'}):
        raise ValueError('The native IMU and world mocap require finite, increasing measurement streams.')
    return imu, mocap, np.asarray(imu_ns, dtype=np.int64), np.asarray(record_ns, dtype=np.int64), bag_start


def crop_window(arrays, start_stamp, end_stamp):
    """Retain every original sample in the requested measurement-time window."""
    keep = (arrays['measurement_stamps'] >= start_stamp) & (arrays['measurement_stamps'] <= end_stamp)
    position_keep = ((arrays['position_measurement_stamps'] >= start_stamp)
                     & (arrays['position_measurement_stamps'] <= end_stamp))
    if np.count_nonzero(keep) < 2:
        raise ValueError('The display window requires at least two measured IMU samples.')
    result = {key: value[position_keep if key.startswith('position_') else keep].copy()
              for key, value in arrays.items()}
    result['time_seconds'] = result['measurement_stamps']-start_stamp
    result['position_time_seconds'] = result['position_measurement_stamps']-start_stamp
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-json', type=Path, required=True)
    parser.add_argument('--bag', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selection_json.read_text())
    bag = args.bag or Path(selected['bag_path'])
    actual_hash = sha256(bag)
    if actual_hash != selected['bag_sha256']:
        raise ValueError('The selected PID success bag hash changed.')
    window = selected['selection']['selected_bag_elapsed_interval_s']
    origin = selected['bag_start_ns']*1e-9
    start_stamp, end_stamp = (origin+float(t) for t in window)
    args.output.mkdir(parents=True, exist_ok=True)
    paired = args.output/'madgwick_comparison'
    paired.mkdir(exist_ok=True)
    started = time.monotonic()
    imu, mocap, imu_ns, record_ns, bag_start = read_native_measurements(bag)
    if abs(bag_start-origin) > 1e-6:
        raise ValueError('The selected bag clock origin differs from the source bag.')
    prefix, reference, metadata = comparison.prepare_measurements(imu, mocap)
    calibration_start = max(imu[0, 1], mocap[0, 1])
    calibration_end = calibration_start+metadata['calibration_seconds']
    calibration = imu[(imu[:, 1] >= calibration_start) & (imu[:, 1] < calibration_end)]
    noise = dict(
        model='stationary Gaussian sensor noise; full gyro covariance and isotropic vector-direction evidence',
        angular_velocity=static_vector_noise.estimate_vector_noise(calibration[:, 2:5]),
        acceleration=static_vector_noise.estimate_isotropic_vector_noise(calibration[:, 5:8]),
        magnetic=static_vector_noise.estimate_isotropic_vector_noise(calibration[:, 8:11]),
        calibration_start_stamp=calibration_start, calibration_end_stamp_exclusive=calibration_end,
        calibration_bag_elapsed_interval_s=[calibration_start-origin, calibration_end-origin],
        calibration_actual_measurement_extent_s=[float(calibration[0, 1]), float(calibration[-1, 1])],
        bag_sha256=actual_hash, calibration_before_filter_initialization=True,
        reference_directions='existing world +z specific force and prefix-only fixed magnetic direction',
        mean_subtracted_from_observations=False, calibration_parameter_uncertainty_propagated=False,
        concentration='norm(static_mean)*norm(current_raw_vector)/isotropic_variance',
        temporal_independence='existing filter assumption; measured lag-one correlations are retained')
    metadata['vector_noise_calibration'] = noise
    noise['angular_velocity'].update(
        units='rad/s', covariance_units='rad^2/s^2',
        covariance_used='full 3x3 sample covariance; no isotropic approximation',
        propagation='existing gyro increment quaternion second moment; dt applied inside the increment map',
        bias_removed=False)
    metadata['gyro_sample_covariance_rad2_s2'] = noise['angular_velocity']['sample_covariance']
    (args.output/'static_noise_calibration.json').write_text(json.dumps(noise, ensure_ascii=False, indent=2)+'\n')
    metadata.pop('acceleration_direction_concentration')
    metadata.pop('magnetic_direction_concentration')
    # Keep the complete recurrence from its original calibration/initial state
    # to the window end. Reference interpolation still has both end brackets.
    n = int(np.searchsorted(prefix['measurement_stamps'], end_stamp, side='right'))
    position_keep = prefix['position_measurement_stamps'] <= end_stamp
    prefix = {key: value[position_keep if key.startswith('position_') else slice(0, n)]
              for key, value in prefix.items()}
    prefix_ns_keep = (imu[:, 1] >= prefix['measurement_stamps'][0]) & (imu[:, 1] <= prefix['measurement_stamps'][-1])
    prefix['measurement_stamp_ns'] = imu_ns[prefix_ns_keep]
    prefix['bag_record_stamp_ns'] = record_ns[prefix_ns_keep]
    prefix['bag_elapsed_seconds'] = prefix['measurement_stamps']-origin
    prefix['recurrence_elapsed_seconds'] = prefix['time_seconds'].copy()
    prefix['acceleration_direction_concentration'] = static_vector_noise.concentration_from_vector_noise(
        prefix['acceleration_m_s2'], noise['acceleration'])
    prefix['magnetic_direction_concentration'] = static_vector_noise.concentration_from_vector_noise(
        prefix['magnetic_raw'], noise['magnetic'])
    if len(prefix['measurement_stamp_ns']) != n:
        raise ValueError('The exact source stamps lost correspondence with prepared measurements.')
    print('prefix samples={}, bag elapsed {:.6f}..{:.6f}; display {}..{} s'.format(
        n, prefix['bag_elapsed_seconds'][0], prefix['bag_elapsed_seconds'][-1], *window), flush=True)
    parameters, modes, carried, residuals = comparison.compare_six(
        prefix['gyro_rad_s'], prefix['acceleration_m_s2'], prefix['magnetic_raw'],
        prefix['measurement_stamps'], np.asarray(metadata['initial_quaternion_wxyz']), reference,
        acceleration_concentration=prefix['acceleration_direction_concentration'],
        magnetic_concentration=prefix['magnetic_direction_concentration'],
        gyro_covariance=np.asarray(metadata['gyro_sample_covariance_rad2_s2']),
        propagation=bingham_moment_inverse.propagate,
        progress=lambda k, total, seconds: print('propagation {}/{}; {:.1f}s'.format(k, total, seconds), flush=True))
    axis_errors, errors, prefix_stats = comparison.evaluate(modes, prefix['mocap_wxyz'])
    prefix.update(display_parameter=parameters, mode_wxyz=modes, carried_parameter=carried,
        moment_fit_residual=residuals, rpy_deg=comparison.display_rpy(modes, prefix['mocap_rpy_deg']),
        rpy_error_deg=axis_errors, error_deg=errors)
    madgwick_modes = comparison.madgwick.run_madgwick(prefix, reference, .1)[:, 1]
    window_arrays = crop_window(prefix, start_stamp, end_stamp)
    keep = (prefix['measurement_stamps'] >= start_stamp) & (prefix['measurement_stamps'] <= end_stamp)
    comparison_modes = np.concatenate((modes[keep][:, comparison.ALL_CASES], madgwick_modes[keep, None]), axis=1)
    paired_arrays = {key: window_arrays[key] for key in (
        'time_seconds', 'measurement_stamps', 'measurement_stamp_ns', 'bag_record_stamp_ns',
        'bag_elapsed_seconds', 'recurrence_elapsed_seconds', 'gyro_rad_s',
        'acceleration_m_s2', 'magnetic_raw', 'mocap_wxyz', 'mocap_rpy_deg',
        'acceleration_direction_concentration', 'magnetic_direction_concentration')}
    paired_axis_errors, paired_errors, paired_stats = comparison.evaluate(comparison_modes, window_arrays['mocap_wxyz'])
    paired_arrays.update(mode_wxyz=comparison_modes,
        rpy_deg=comparison.display_rpy(comparison_modes, window_arrays['mocap_rpy_deg']),
        rpy_error_deg=paired_axis_errors, error_deg=paired_errors)
    # Preserve the completed inference before plotting/provenance I/O.
    np.savez_compressed(args.output/'timeseries.npz', **window_arrays)
    np.savez_compressed(paired/'timeseries.npz', **paired_arrays)
    _, _, window_stats = comparison.evaluate(window_arrays['mode_wxyz'], window_arrays['mocap_wxyz'])
    # Labels change typography only. IDs, coefficients and distributions remain
    # the same six conditions as the existing comparison and renderer.
    comparison.CASE_LABELS = tuple(display_label(label) for label in comparison.CASE_LABELS)
    comparison.MADGWICK_LABEL = display_label(comparison.MADGWICK_LABEL)
    labels = [comparison.CASE_LABELS[i] for i in comparison.ALL_CASES]+[comparison.MADGWICK_LABEL]
    colors = [comparison.CASE_COLORS[i] for i in comparison.ALL_CASES]+[comparison.MADGWICK_COLOR]
    styles = [comparison.CASE_STYLES[i] for i in comparison.ALL_CASES]+[':']
    source_scripts = [Path(__file__).resolve(), *(Path(__file__).resolve().parent/name for name in (
        'compare_feedback_orientation.py', 'compare_rolling_bag.py', 'compare_madgwick_rolling.py',
        'static_vector_noise.py', 'bingham_moment_inverse.py'))]
    metadata['direction_concentration_statistics'] = {
        sensor: dict(minimum=float(window_arrays[sensor+'_direction_concentration'].min()),
            median=float(np.median(window_arrays[sensor+'_direction_concentration'])),
            maximum=float(window_arrays[sensor+'_direction_concentration'].max()))
        for sensor in ('acceleration', 'magnetic')}
    metadata['moment_inverse'] = 'scaled Bingham moment equations with the unchanged core moments and analytic Jacobian; same inverse in all three groups'
    metadata.update(bag_path=str(bag.resolve()), bag_sha256=actual_hash,
        bag_size_bytes=bag.stat().st_size, bag_start_ns=selected['bag_start_ns'],
        window_selection=selected['selection'], selection_sha256=sha256(args.selection_json),
        evaluated_imu_samples=len(window_arrays['time_seconds']), position_samples=len(window_arrays['position_m']),
        duration_seconds=float(window_arrays['time_seconds'][-1]-window_arrays['time_seconds'][0]),
        requested_display_duration_seconds=float(window[1]-window[0]),
        window_time_zero_stamp=start_stamp, prefix_propagated_imu_samples=n,
        prefix_measurement_origin=metadata['measurement_time_origin'],
        prefix_last_measurement_stamp=float(prefix['measurement_stamps'][-1]),
        reset_at_window_start=False, gyro_bias_removed=False,
        prefix_statistics=dict(zip(comparison.CASE_IDS, prefix_stats)),
        statistics=dict(zip(comparison.CASE_IDS, window_stats)),
        maximum_moment_fit_residual=float(residuals.max()),
        case_labels=dict(zip(comparison.CASE_IDS, comparison.CASE_LABELS)),
        output_and_retained_evidence_coefficients=comparison.COEFFICIENTS.tolist(),
        carried_group_ids=comparison.GROUP_IDS,
        case_carried_groups=dict(zip(comparison.CASE_IDS, comparison.CASE_GROUPS)),
        prediction_convention='right-end angular velocity; all prefix measurements; no window reset',
        input_topics={'imu': '/gimbalrotor/imu', 'mocap': '/gimbalrotor/mocap/pose'},
        imu_message_type='spinal/Imu', imu_frame='gimbalrotor/fc', mocap_position_frame='world',
        body_to_sensor_rotation='identity from the recorded main_body -> fc static transform',
        native_firmware_quaternion_used=False, plot_style=['science', 'no-latex'],
        coordinate_colors=dict(zip(('x', 'y', 'z'), comparison.COORDINATE_COLORS)),
        plot_colors=dict(zip(comparison.CASE_IDS, comparison.CASE_COLORS)),
        plot_styles=dict(zip(comparison.CASE_IDS, comparison.CASE_STYLES)),
        source_files_sha256={str(path.relative_to(comparison.original.ROOT)): sha256(path) for path in source_scripts})
    comparison.plot_orientation(window_arrays, args.output)
    comparison.plot_errors(window_arrays, args.output, comparison.CASE_LABELS,
        comparison.CASE_COLORS, comparison.CASE_STYLES, 'axis_errors')
    metadata['raw_plot_limits'] = comparison.plot_raw_measurements(window_arrays, args.output)
    comparison.plot_errors(paired_arrays, paired, labels, colors, styles, 'method_comparison')
    paired_metadata = dict(metadata, madgwick_beta=.1,
        world_from_magnetic_north_heading_deg=float(np.rad2deg(comparison.madgwick.heading_frame(reference)[1])),
        statistics=dict(zip([comparison.CASE_IDS[i] for i in comparison.ALL_CASES]+['madgwick_marg'], paired_stats)),
        case_labels=dict(zip([comparison.CASE_IDS[i] for i in comparison.ALL_CASES]+['madgwick_marg'], labels)))
    for output, arrays, report in ((args.output, window_arrays, metadata), (paired, paired_arrays, paired_metadata)):
        (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (args.output/'selection.json').write_text(args.selection_json.read_text())
    (args.output/'static_noise_calibration.json').write_text(json.dumps(noise, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), prefix_samples=n,
        window_samples=len(window_arrays['time_seconds']), statistics=metadata['statistics'],
        madgwick=paired_metadata['statistics']['madgwick_marg'],
        wall_seconds=time.monotonic()-started), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
