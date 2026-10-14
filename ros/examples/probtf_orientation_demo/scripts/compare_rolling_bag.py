#!/usr/bin/env python3
"""Compare five gyro, accelerometer, and magnetometer orientation estimators.

All angular-velocity samples are propagated.  The calculation uses the
ProbTF core's cubature, quaternion-product moments, Bingham moment matching,
and gravity/magnetic vector evidence.  The deprecated ROS package is never imported.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/probtf-orientation-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def repository_root():
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    if os.environ.get('PROBTF_DEMO_ROOT'):
        candidates.insert(0, Path(os.environ['PROBTF_DEMO_ROOT']))
    for root in candidates:
        if (root / 'ros/core/probtf_core/src/probtf_estimators').is_dir():
            return root
    raise RuntimeError('Run this script in a ProbTF-demo checkout.')


ROOT = repository_root()
sys.path.insert(0, str(ROOT / 'ros/core/probtf_core/src'))

from probtf.bingham import (
    bingham_mode, bingham_second_moment, canonical_bingham_parameter,
    match_bingham_to_second_moment, quaternion_product_second_moment,
)
from probtf_estimators.orientation_imu import (
    delta_quaternion_second_moment, gravity_bingham_evidence, magnetic_bingham_evidence,
)

CASE_IDS = ('gyro_only', 'gyro_acceleration', 'gyro_magnetic',
            'magnetic_acceleration', 'nine_axis')
CASE_LABELS = ('角速度の時間発展のみ', '角速度と加速度の融合', '角速度と地磁気の融合',
               '地磁気と加速度の融合', '9軸の全情報の融合')
CASE_COLORS = ('#0072B2', '#009E73', '#E69F00', '#CC79A7', '#D55E00')
PROPAGATED_CASES = (0, 1, 2, 4)
SPECIFIC_FORCE_REFERENCE_WORLD = np.array([0.0, 0.0, 1.0])
REFERENCE_LABEL = 'モーションキャプチャ'


def rotations_from_wxyz(quaternions):
    return Rotation.from_quat(np.asarray(quaternions)[..., [1, 2, 3, 0]])


def wxyz_from_rotations(rotations):
    return rotations.as_quat()[..., [3, 0, 1, 2]]


def continuous_quaternions(quaternions):
    output = np.asarray(quaternions, dtype=float).copy()
    for k in range(1, len(output)):
        if output[k-1] @ output[k] < 0.0:
            output[k] *= -1.0
    return output


def rotation_error_series(modes, truth):
    """Absolute wrapped Euler-component differences and SO(3) angle, in degrees.

    Euler-component differences compare the actual RPY traces, rather than
    decomposing the relative rotation.  The combined error is the magnitude
    of that relative rotation, not the Euclidean norm of the three differences.
    """
    truth_rotation = rotations_from_wxyz(truth)
    reference_rpy = truth_rotation.as_euler('xyz', degrees=True)
    axis_errors = []
    combined_errors = []
    for case in range(modes.shape[1]):
        estimate = rotations_from_wxyz(modes[:, case])
        difference = estimate.as_euler('xyz', degrees=True) - reference_rpy
        axis_errors.append(np.abs((difference + 180.0) % 360.0 - 180.0))
        combined_errors.append(np.rad2deg((estimate.inv() * truth_rotation).magnitude()))
    return np.stack(axis_errors, axis=1), np.stack(combined_errors, axis=1)


def update_error_evaluation(arrays, metadata):
    if arrays['mode_wxyz'].shape[1] != len(CASE_IDS):
        raise ValueError('This script requires five-condition data; re-run the bag calculation in a new output directory.')
    axis_errors, combined_errors = rotation_error_series(
        arrays['mode_wxyz'], arrays['mocap_wxyz'],
    )
    arrays['rpy_error_deg'] = axis_errors
    arrays['error_deg'] = combined_errors
    metadata['case_labels'] = dict(zip(CASE_IDS, CASE_LABELS))
    metadata['rpy_error_definition'] = (
        'absolute extrinsic-xyz Euler component difference, wrapped to [-180, 180) degrees; '
        'estimate minus mocap; no decomposition of the relative rotation'
    )
    metadata['combined_error_definition'] = (
        'SO(3) relative-rotation angle in degrees, 2*acos(abs(q_ref dot q_mode)); '
        'not the Euclidean norm of RPY differences'
    )
    metadata['statistics'] = {}
    for case, name in enumerate(CASE_IDS):
        error = combined_errors[:, case]
        metadata['statistics'][name] = {
            'rms_deg': float(np.sqrt(np.mean(error**2))),
            'median_deg': float(np.median(error)), 'p95_deg': float(np.quantile(error, 0.95)),
            'max_deg': float(error.max()), 'final_deg': float(error[-1]),
            'rpy_errors': {
                component: {
                    'rms_deg': float(np.sqrt(np.mean(axis_errors[:, case, i]**2))),
                    'median_deg': float(np.median(axis_errors[:, case, i])),
                    'p95_deg': float(np.quantile(axis_errors[:, case, i], 0.95)),
                    'max_deg': float(axis_errors[:, case, i].max()),
                    'final_deg': float(axis_errors[-1, case, i]),
                }
                for i, component in enumerate(('roll', 'pitch', 'yaw'))
            },
        }


def read_bag(path):
    import rosbag
    imu, mocap = [], []
    with rosbag.Bag(str(path), 'r') as bag:
        for topic, msg, stamp in bag.read_messages(
            topics=['/delta/imu', '/delta/mocap/pose'],
        ):
            if topic == '/delta/imu':
                imu.append([stamp.to_sec(), msg.stamp.to_sec(),
                            *msg.gyro_data, *msg.acc_data, *msg.mag_data])
            else:
                q = msg.pose.orientation
                mocap.append([stamp.to_sec(), msg.header.stamp.to_sec(),
                              q.w, q.x, q.y, q.z])
    imu, mocap = np.asarray(imu, dtype=float), np.asarray(mocap, dtype=float)
    if len(imu) < 2 or len(mocap) < 2:
        raise ValueError('Both /delta/imu and /delta/mocap/pose are required.')
    if not np.all(np.isfinite(imu)) or not np.all(np.isfinite(mocap)):
        raise ValueError('The input streams contain non-finite values.')
    if np.any(np.diff(imu[:, 1]) <= 0.0) or np.any(np.diff(mocap[:, 1]) <= 0.0):
        raise ValueError('Measurement timestamps must strictly increase.')
    return imu, mocap


def prepare_streams(imu, mocap, calibration_seconds=1.0, duration=None):
    overlap_start = max(imu[0, 1], mocap[0, 1])
    overlap_end = min(imu[-1, 1], mocap[-1, 1])
    interpolation = Slerp(mocap[:, 1], rotations_from_wxyz(mocap[:, 2:6]))
    calibration = imu[(imu[:, 1] >= overlap_start)
                      & (imu[:, 1] < overlap_start + calibration_seconds)]
    if len(calibration) < 20:
        raise ValueError('At least 20 calibration samples are required.')
    norm = np.linalg.norm(calibration[:, 8:11], axis=1)
    if np.any(norm <= 0.0):
        raise ValueError('The calibration contains zero magnetic vectors.')
    calibration_rotations = interpolation(calibration[:, 1])
    fields_world = calibration_rotations.apply(calibration[:, 8:11] / norm[:, None])
    magnetic_reference = fields_world.mean(axis=0)
    magnetic_reference /= np.linalg.norm(magnetic_reference)
    acceleration_mean_world = calibration_rotations.apply(calibration[:, 5:8]).mean(axis=0)
    selected = imu[(imu[:, 1] >= overlap_start + calibration_seconds)
                   & (imu[:, 1] <= overlap_end)]
    if duration is not None:
        selected = selected[selected[:, 1] <= selected[0, 1] + duration]
    truth = interpolation(selected[:, 1])
    calibration_motion = np.rad2deg(
        (calibration_rotations[0].inv() * calibration_rotations).magnitude()
    )
    metadata = {
        'input_imu_count': int(len(imu)), 'input_mocap_count': int(len(mocap)),
        'calibration_samples': int(len(calibration)),
        'calibration_seconds': calibration_seconds,
        'calibration_mocap_motion_max_deg': float(calibration_motion.max()),
        'calibration_gyro_mean_rad_s': calibration[:, 2:5].mean(axis=0).tolist(),
        'calibration_gyro_std_rad_s': calibration[:, 2:5].std(axis=0).tolist(),
        'gyro_bias_removed': False,
        'accelerometer_bias_removed': False,
        'calibration_acceleration_mean_m_s2': calibration[:, 5:8].mean(axis=0).tolist(),
        'calibration_acceleration_mean_world_m_s2': acceleration_mean_world.tolist(),
        'calibration_acceleration_world_z_angle_deg': float(np.rad2deg(np.arccos(np.clip(
            acceleration_mean_world[2]/np.linalg.norm(acceleration_mean_world), -1.0, 1.0)))),
        'reference_magnetic_world_unit': magnetic_reference.tolist(),
        'initial_quaternion_wxyz': wxyz_from_rotations(truth[0]).tolist(),
        'measurement_time_origin': float(selected[0, 1]),
        'mocap_reference_usage': 'initial orientation and prefix-only fixed magnetic reference',
        'alignment': 'header timestamps; quaternion spherical interpolation; no extrapolation',
        'imu_latency_quantiles_seconds': np.quantile(imu[:, 0]-imu[:, 1], [0, .5, 1]).tolist(),
        'mocap_latency_quantiles_seconds': np.quantile(mocap[:, 0]-mocap[:, 1], [0, .5, 1]).tolist(),
        'maximum_mocap_gap_seconds': float(np.diff(mocap[:, 1]).max()),
    }
    return selected, wxyz_from_rotations(truth), magnetic_reference, metadata


def propagate(parameter, gyro, dt, covariance, integration_steps=120):
    if dt == 0.0 or (not np.any(gyro) and not np.any(covariance)):
        return canonical_bingham_parameter(parameter), 0.0
    current_moment = bingham_second_moment(parameter, integration_steps=integration_steps)
    delta_moment = delta_quaternion_second_moment(gyro, dt, covariance)
    target = quaternion_product_second_moment(current_moment, delta_moment)
    # Use the public core solver with the preceding concentration as its
    # starting point.  This changes the numerical initialization, not the
    # propagation or observation model.
    predicted = match_bingham_to_second_moment(
        target, integration_steps=integration_steps,
        initial_eigenvalues=np.linalg.eigvalsh(parameter), max_iterations=80,
    )
    fitted_moment = bingham_second_moment(predicted, integration_steps=integration_steps)
    residual = float(np.linalg.norm(target-fitted_moment))
    if residual > 2e-5:
        fallback = match_bingham_to_second_moment(
            target, integration_steps=integration_steps, max_iterations=160,
        )
        fallback_residual = float(np.linalg.norm(
            target-bingham_second_moment(fallback, integration_steps=integration_steps)
        ))
        if fallback_residual < residual:
            predicted, residual = fallback, fallback_residual
    if not np.isfinite(residual) or residual > 1e-3:
        raise FloatingPointError('Bingham moment matching did not resolve the target: {}'.format(residual))
    return canonical_bingham_parameter(predicted), residual


def compare_conditions(gyro, acceleration, magnetic, timestamps, initial_quaternion,
                       magnetic_reference, initial_concentration=100.0,
                       gyro_variance=1e-4, acceleration_concentration=20.0, magnetic_concentration=20.0,
                       integration_steps=120, progress=None):
    initial = initial_concentration * (
        np.outer(initial_quaternion, initial_quaternion) - np.eye(4)
    )
    states = {case: initial.copy() for case in PROPAGATED_CASES}
    covariance = np.eye(3) * gyro_variance
    count = len(timestamps)
    display = np.empty((count, len(CASE_IDS), 4, 4))
    residuals = np.zeros((count, len(PROPAGATED_CASES)))
    started, last_progress = time.monotonic(), time.monotonic()
    for k in range(count):
        if k:
            dt = timestamps[k]-timestamps[k-1]
            if dt <= 0.0:
                raise ValueError('The propagation timestamps must strictly increase.')
            # The right-end angular velocity is held over this interval,
            # matching the ROS orientation filter's sample convention.
            for column, case in enumerate(PROPAGATED_CASES):
                states[case], residuals[k, column] = propagate(
                    states[case], gyro[k], dt, covariance, integration_steps,
                )
        gravity_evidence = gravity_bingham_evidence(
            SPECIFIC_FORCE_REFERENCE_WORLD, acceleration[k], acceleration_concentration,
        )
        magnetic_evidence = magnetic_bingham_evidence(
            magnetic_reference, magnetic[k], magnetic_concentration,
        )
        states[1] = canonical_bingham_parameter(states[1] + gravity_evidence)
        states[2] = canonical_bingham_parameter(states[2] + magnetic_evidence)
        # No gyro or historical posterior enters this observation-only case.
        display[k, 3] = canonical_bingham_parameter(gravity_evidence + magnetic_evidence)
        states[4] = canonical_bingham_parameter(states[4] + gravity_evidence + magnetic_evidence)
        for case in PROPAGATED_CASES:
            display[k, case] = states[case]
        if progress and time.monotonic()-last_progress >= 10.0:
            progress(k+1, count, time.monotonic()-started)
            last_progress = time.monotonic()
    modes = np.stack([
        continuous_quaternions(np.array([bingham_mode(A) for A in display[:, case]]))
        for case in range(len(CASE_IDS))
    ], axis=1)
    for case in range(len(CASE_IDS)):
        if modes[0, case] @ initial_quaternion < 0.0:
            modes[:, case] *= -1.0
    # The exact initial gyro-only mode is known analytically.  Keeping it
    # avoids an artificial nonzero identity error from eigensolver roundoff.
    modes[0, 0] = initial_quaternion
    return display, modes, residuals


def metrics_and_arrays(imu, truth, modes, display, residuals, reference, metadata, args):
    truth = continuous_quaternions(truth)
    if truth[0] @ modes[0, 0] < 0.0:
        truth *= -1.0
    truth_rotation = rotations_from_wxyz(truth)
    principal_rpy = np.stack([
        rotations_from_wxyz(modes[:, case]).as_euler('xyz')
        for case in range(len(CASE_IDS))
    ], axis=1)
    reference_radians = np.unwrap(truth_rotation.as_euler('xyz'), axis=0)
    # This is a display-only choice of the Euler-angle branch.  Mocap is
    # never passed to the time recursion after its common initialization.
    difference = (principal_rpy-reference_radians[:,None,:]+np.pi) % (2*np.pi)-np.pi
    rpy = np.rad2deg(reference_radians[:,None,:]+difference)
    reference_rpy = np.rad2deg(reference_radians)
    norms = np.linalg.norm(imu[:, 8:11], axis=1)
    unit_fields = imu[:, 8:11] / norms[:, None]
    field_residual = np.rad2deg(np.arccos(np.clip(
        truth_rotation.apply(unit_fields) @ reference, -1.0, 1.0,
    )))
    metadata.update({
        'evaluated_imu_samples': len(imu), 'duration_seconds': float(imu[-1, 1]-imu[0, 1]),
        'initial_bingham_concentration': args.initial_concentration,
        'gyro_sample_covariance_rad2_s2': (np.eye(3)*args.gyro_variance).tolist(),
        'magnetic_direction_concentration': args.magnetic_concentration,
        'acceleration_direction_concentration': args.acceleration_concentration,
        'integration_steps': args.integration_steps,
        'gravity_evidence_case_ids': [CASE_IDS[i] for i in (1, 3, 4)],
        'reference_specific_force_world_unit': SPECIFIC_FORCE_REFERENCE_WORLD.tolist(),
        'acceleration_model': 'normalized specific force aligned with mocap-world +z; translational acceleration not removed; no norm gating or smoothing',
        'propagation_case_ids': [CASE_IDS[i] for i in PROPAGATED_CASES],
        'observation_only_case': 'magnetic_acceleration: current gravity and magnetic evidence with a uniform prior, independently at each sample',
        'posterior_reused_as_next_prior': [CASE_IDS[i] for i in (1, 2, 4)],
        'sample_convention': 'right-end angular velocity; interval from measurement timestamps',
        'moment_fit_residual_frobenius_max': float(residuals.max()),
        'moment_fit_residual_frobenius_p95': float(np.quantile(residuals, .95)),
        'magnetic_norm_min_max_raw_units': [float(norms.min()), float(norms.max())],
        'magnetic_direction_residual_deg': {
            'rms': float(np.sqrt(np.mean(field_residual**2))),
            'p95': float(np.quantile(field_residual, .95)), 'max': float(field_residual.max()),
        },
        'rpy_convention': 'extrinsic xyz; mocap unwrapped; each estimate uses the nearest mocap component branch for display only',
        'quaternion_convention': 'wxyz, adjacent signs aligned, common initial sign',
    })
    arrays = dict(
        time_seconds=imu[:, 1]-imu[0, 1], measurement_stamps=imu[:, 1],
        gyro_rad_s=imu[:, 2:5], acceleration_m_s2=imu[:, 5:8], magnetic_raw=imu[:, 8:11],
        mocap_wxyz=truth, mode_wxyz=modes, display_parameter=display,
        rpy_deg=rpy, mocap_rpy_deg=reference_rpy,
        magnetic_norm=norms, magnetic_direction_residual_deg=field_residual,
        moment_fit_residual=residuals,
    )
    update_error_evaluation(arrays, metadata)
    return metadata, arrays


def configure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'IPAexGothic', 'font.size': 10,
        'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    })
    return plt


def save_figure(figure, output, name):
    for extension in ('pdf', 'svg', 'png'):
        destination = output / (name+'.'+extension)
        figure.savefig(destination, dpi=180, bbox_inches='tight')
        if extension == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')


def plot_results(arrays, output):
    plt = configure_matplotlib()
    t = arrays['time_seconds']
    for name, values, reference, components, units in (
        ('rpy', arrays['rpy_deg'], arrays['mocap_rpy_deg'], ('roll', 'pitch', 'yaw'), 'deg'),
        ('quaternion', arrays['mode_wxyz'], arrays['mocap_wxyz'], ('w', 'x', 'y', 'z'), ''),
    ):
        figure, axes = plt.subplots(len(components), 1, sharex=True,
                                   figsize=(8.1, 2.0*len(components)))
        for component, axis in enumerate(axes):
            for case in range(len(CASE_IDS)):
                axis.plot(t, values[:, case, component], color=CASE_COLORS[case],
                          lw=1.0, label=CASE_LABELS[case], alpha=.9)
            axis.plot(t, reference[:, component], color='#222222', ls='--', lw=1.0,
                      label=REFERENCE_LABEL)
            axis.set_ylabel(components[component]+(' / '+units if units else ''))
            axis.grid(alpha=.22)
            axis.set_xlim(t[0], t[-1])
        axes[0].legend(ncol=1, loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False)
        axes[-1].set_xlabel('経過時間 / s')
        figure.tight_layout()
        save_figure(figure, output, name)
        plt.close(figure)
    for logarithmic in (False, True):
        figure, axes = plt.subplots(4, 1, sharex=True, figsize=(8.1, 8.4))
        for component, axis in enumerate(axes):
            values = arrays['rpy_error_deg'][:, :, component] if component < 3 else arrays['error_deg']
            if logarithmic:
                axis.set_yscale('log', nonpositive='mask')
            for case in range(len(CASE_IDS)):
                plotted = np.ma.masked_less_equal(values[:, case], 0.0) if logarithmic else values[:, case]
                axis.plot(t, plotted, color=CASE_COLORS[case], lw=1.0, label=CASE_LABELS[case])
            axis.set_ylabel(('ロール誤差', 'ピッチ誤差', 'ヨー誤差', '3軸の合成誤差')[component]+' / deg')
            if not logarithmic:
                axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.22)
            axis.set_xlim(t[0], t[-1])
        axes[0].legend(ncol=1, loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False)
        axes[-1].set_xlabel('経過時間 / s')
        figure.tight_layout()
        save_figure(figure, output, 'axis_errors_semilog' if logarithmic else 'axis_errors')
        plt.close(figure)


def write_csv(arrays, destination):
    headers = ['time_s']
    columns = [arrays['time_seconds']]
    for name, q in [('mocap', arrays['mocap_wxyz'])] + [
        (name, arrays['mode_wxyz'][:, i]) for i, name in enumerate(CASE_IDS)
    ]:
        for i, component in enumerate('wxyz'):
            headers.append(name+'_q'+component)
            columns.append(q[:, i])
    for name, rpy in [('mocap', arrays['mocap_rpy_deg'])] + [
        (name, arrays['rpy_deg'][:, i]) for i, name in enumerate(CASE_IDS)
    ]:
        for i, component in enumerate(('roll', 'pitch', 'yaw')):
            headers.append(name+'_'+component+'_deg')
            columns.append(rpy[:, i])
    for i, name in enumerate(CASE_IDS):
        for component, label in enumerate(('roll', 'pitch', 'yaw')):
            headers.append(name+'_'+label+'_error_deg')
            columns.append(arrays['rpy_error_deg'][:, i, component])
        headers.append(name+'_rotation_error_deg')
        columns.append(arrays['error_deg'][:, i])
    headers += ['magnetic_direction_residual_deg', 'magnetic_norm_raw_units']
    columns += [arrays['magnetic_direction_residual_deg'], arrays['magnetic_norm']]
    with destination.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(np.column_stack(columns))


def render_video(arrays, output, fps=8.0, samples=180):
    plt = configure_matplotlib()
    sys.path.insert(0, str(ROOT / 'third_party/BinghamNLL/src'))
    from bingham.visualize.SO3s import draw_so3s
    from probtf.distributions import BinghamOrientation
    from probtf.probability.sampling import sample_bingham_orientation
    t = arrays['time_seconds']
    frame_times = np.arange(0.0, t[-1]+.5/fps, 1.0/fps)
    indices = np.clip(np.searchsorted(t, frame_times), 0, len(t)-1)
    figure = plt.figure(figsize=(16, 13.2), dpi=100)
    grid = figure.add_gridspec(3, 3, height_ratios=(3.6, 3.6, 1.6),
                              left=0.06, right=0.96, top=0.89, bottom=0.06, hspace=0.35)
    axes = [figure.add_subplot(grid[i//3, i%3], projection='3d') for i in range(6)]
    # Five distributions and mocap in reading order above a common timeline.
    timeline = figure.add_subplot(grid[2, :])
    for case in range(len(CASE_IDS)):
        timeline.plot(t, arrays['error_deg'][:, case], color=CASE_COLORS[case],
                      lw=1.0, label=CASE_LABELS[case])
    cursor = timeline.axvline(0.0, color='#222222', lw=1.0)
    timeline.set_xlim(0.0, t[-1])
    timeline.set_ylim(0.0, max(10.0, float(arrays['error_deg'].max())*1.05))
    timeline.set_xlabel('経過時間 / s')
    timeline.set_ylabel('3軸の合成誤差 / deg')
    timeline.grid(alpha=.2)
    timeline.legend(ncol=len(CASE_IDS), loc='upper left', fontsize=9, frameon=False)
    clock = figure.text(.5, .985, '', ha='center', va='top', fontsize=14)
    description = figure.text(.5, .955,
        '点群：Bingham分布の軸端点　実線の軸：分布のモード　黒破線：モーションキャプチャ',
        ha='center', va='top', fontsize=10)
    width, height = figure.canvas.get_width_height()
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '{}x{}'.format(width,height),
               '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output/'bingham_comparison.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    generator = np.random.default_rng()
    saved = set()
    targets = np.linspace(0.0, t[-1], 5)
    snapshot_frames = {int(np.argmin(abs(frame_times-value))) for value in targets}
    try:
        for frame, k in enumerate(indices):
            truth_rotation = rotations_from_wxyz(arrays['mocap_wxyz'][k]).as_matrix()
            for case, axis in enumerate(axes):
                axis.clear()
                if case < len(CASE_IDS):
                    orientation = BinghamOrientation.from_parameter_matrix(arrays['display_parameter'][k,case])
                    q = sample_bingham_orientation(orientation, samples, rng=generator)
                    rotations = rotations_from_wxyz(q).as_matrix()
                    mode = rotations_from_wxyz(arrays['mode_wxyz'][k,case]).as_matrix()
                    draw_so3s(axis, rotations, axes=mode, axes_gt=None,
                              point_of_view=np.array([1.4,-1.5,1.0]), distance=9)
                    # A line break preserves the wording while fitting each sphere.
                    title = CASE_LABELS[case].replace('（', '\n（')
                    axis.set_title(title+'\n誤差 {:.2f} deg'.format(arrays['error_deg'][k,case]),
                                   color=CASE_COLORS[case], fontsize=10, pad=12)
                else:
                    axis.plot([], [], [], label=REFERENCE_LABEL)
                    draw_so3s(axis, None, axes=truth_rotation, axes_gt=None,
                              point_of_view=np.array([1.4,-1.5,1.0]), distance=9)
                    axis.set_title(REFERENCE_LABEL, fontsize=12, pad=18)
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()
                if case < len(CASE_IDS):
                    for component in range(3):
                        tip = truth_rotation[:,component]
                        axis.plot([0,tip[0]], [0,tip[1]], [0,tip[2]],
                                  color='black', ls='--', lw=2.0)
                axis.set_xlim(-1.05,1.05)
                axis.set_ylim(-1.05,1.05)
                axis.set_zlim(-1.05,1.05)
                axis.set_xlabel('x')
                axis.set_ylabel('y')
                axis.set_zlabel('z')
            cursor.set_xdata([t[k],t[k]])
            clock.set_text('t = {:.2f} s'.format(t[k]))
            figure.canvas.draw()
            process.stdin.write(figure.canvas.tostring_rgb())
            if frame in snapshot_frames and frame not in saved:
                figure.savefig(output/'bingham_snapshot_{:03d}.png'.format(len(saved)),dpi=100)
                saved.add(frame)
            if frame % max(1,int(fps*5)) == 0:
                print('video frame {}/{} (t={:.2f}s)'.format(frame+1,len(indices),t[k]),flush=True)
    finally:
        process.stdin.close()
        result = process.wait()
        plt.close(figure)
    if result:
        raise RuntimeError('ffmpeg failed with exit status {}'.format(result))
    return {'frames': len(indices), 'fps': fps, 'width': width, 'height': height,
            'visualizer': 'third_party/BinghamNLL/src/bingham/visualize/SO3s.py:draw_so3s',
            'samples_per_distribution_per_frame': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path,
                        default=Path.home()/'rosbags/20250310_delta_linear_rolling_1_2025-03-10-16-18-36.bag')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-seconds', type=float, default=1.0)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--initial-concentration', type=float, default=100.0)
    parser.add_argument('--gyro-variance', type=float, default=1e-4)
    parser.add_argument('--acceleration-concentration', type=float, default=20.0)
    parser.add_argument('--magnetic-concentration', type=float, default=20.0)
    parser.add_argument('--integration-steps', type=int, default=120)
    parser.add_argument('--video-fps', type=float, default=8.0)
    parser.add_argument('--video-samples', type=int, default=180)
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.render_only:
        with np.load(args.output/'timeseries.npz') as archive:
            arrays = {key:archive[key] for key in archive.files}
        metadata = json.loads((args.output/'summary.json').read_text())
        if 'video' in metadata:
            metadata.setdefault('video_script_sha256', metadata.get('evaluation_script_sha256', metadata.get('script_sha256')))
        update_error_evaluation(arrays, metadata)
    else:
        imu, mocap = read_bag(args.bag)
        imu, truth, reference, metadata = prepare_streams(
            imu, mocap, args.calibration_seconds, args.duration,
        )
        print('calibration: {} samples; evaluation: {} samples, {:.2f} seconds'.format(
            metadata['calibration_samples'],len(imu),imu[-1,1]-imu[0,1]),flush=True)
        display, modes, residuals = compare_conditions(
            imu[:,2:5], imu[:,5:8], imu[:,8:11], imu[:,1], truth[0], reference,
            initial_concentration=args.initial_concentration,
            gyro_variance=args.gyro_variance,
            acceleration_concentration=args.acceleration_concentration,
            magnetic_concentration=args.magnetic_concentration,
            integration_steps=args.integration_steps,
            progress=lambda done,total,elapsed: print(
                'propagation {}/{}; {:.1f}s elapsed'.format(done,total,elapsed),flush=True),
        )
        metadata, arrays = metrics_and_arrays(
            imu,truth,modes,display,residuals,reference,metadata,args,
        )
        metadata['bag_path'] = str(args.bag.resolve())
        metadata['bag_size_bytes'] = args.bag.stat().st_size
        metadata['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Keep the original propagation-script hash when re-evaluating cached states.
    metadata['evaluation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['evaluation_from_saved_states'] = args.render_only
    np.savez_compressed(args.output/'timeseries.npz', **arrays)
    write_csv(arrays,args.output/'timeseries.csv')
    (args.output/'summary.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metadata['statistics'],indent=2),flush=True)
    plot_results(arrays,args.output)
    if not args.skip_video:
        metadata['video'] = render_video(arrays,args.output,args.video_fps,args.video_samples)
        metadata['video_script_sha256'] = metadata['evaluation_script_sha256']
    metadata['wall_seconds'] = time.monotonic()-started
    (args.output/'summary.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print('completed: '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
