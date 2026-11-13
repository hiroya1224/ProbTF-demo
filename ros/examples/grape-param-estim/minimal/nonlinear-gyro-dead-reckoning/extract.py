#!/usr/bin/env python3
"""Extract the fixed successful bag-8 prefix for sparse-gyro prediction.

This module only reads original clocks and values. It does not integrate an
orientation, adjust a model, add sensor noise, or fit anything to mocap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
MINIMAL = HERE.parent
BAG = Path('/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/'
           '20260612_grape_hovering_8_2026-06-12-17-46-41.bag')
MODEL = MINIMAL / 'grape_vehicle_model.json'
OUTPUT = Path('/tmp/nonlinear-gyro-dead-reckoning-20261113')
GROUPS = ('xy', 'z', 'roll_pitch', 'yaw')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _helpers():
    path = MINIMAL / 'imu-first-order-lag/imu_data.py'
    spec = importlib.util.spec_from_file_location('nonlinear_existing_imu_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(bag_path=BAG, start_seconds=50., duration_seconds=5.,
            gyro_period_seconds=2., calibration_window=(40., 45.)):
    import genpy
    import rosbag

    helper = _helpers()
    start, duration, period = map(float, (start_seconds, duration_seconds,
                                        gyro_period_seconds))
    if start < 0. or duration <= 0. or period <= 0.:
        raise ValueError('A nonnegative start and positive duration/gyro period are required')
    topics = dict(imu=helper.TOPICS['imu'], converted=helper.TOPICS['converted'],
                  mocap=helper.TOPICS['mocap'], state=helper.TOPICS['state'],
                  rotor=helper.TOPICS['rotor'], joint=helper.TOPICS['joint'],
                  gimbal_command=helper.TOPICS['gimbal_command'], static='/tf_static')
    for group in GROUPS:
        topics['gain_' + group] = '/gimbalrotor/controller/' + group + '/parameter_updates'
    inverse = {topic: name for name, topic in topics.items()}
    streams = {name: [] for name in topics}
    with rosbag.Bag(str(bag_path), 'r') as bag:
        origin = next(bag.read_messages(raw=True))[2].to_nsec()
        end_record = genpy.Time.from_sec(origin * 1e-9 + start + duration + .2)
        for topic, message, record in bag.read_messages(topics=list(inverse),
                                                       end_time=end_record):
            name = inverse[topic]
            stamp = (message.stamp if name == 'imu' else message.header.stamp
                     if name in ('converted', 'mocap', 'joint') else record)
            row = dict(stamp_ns=stamp.to_nsec(), record_ns=record.to_nsec(),
                       source_index=len(streams[name]))
            if name == 'imu':
                row['gyro'] = list(message.gyro)
            elif name == 'converted':
                row['frame'] = message.header.frame_id
            elif name == 'mocap':
                q, p = message.pose.orientation, message.pose.position
                row.update(quaternion=[q.x, q.y, q.z, q.w],
                           position=[p.x, p.y, p.z], frame=message.header.frame_id)
            elif name == 'state':
                row['state'] = int(message.data)
            elif name == 'rotor':
                row['value'] = list(message.base_thrust)
            elif name == 'gimbal_command':
                row['value'] = list(message.position)
            elif name == 'joint':
                lookup = dict(zip(message.name, message.position))
                row['value'] = [float(lookup[j]) for j in helper.JOINTS]
            elif name == 'static':
                row['transforms'] = []
                for transform in message.transforms:
                    q, p = transform.transform.rotation, transform.transform.translation
                    row['transforms'].append((transform.header.frame_id,
                        transform.child_frame_id, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
            elif name.startswith('gain_'):
                row['doubles'] = {p.name: float(p.value) for p in message.doubles}
                row['bools'] = {p.name: bool(p.value) for p in message.bools}
            streams[name].append(row)

    local = lambda rows: (helper._strict(rows, 'source') - origin).astype(float) * 1e-9
    imu_time = local(streams['imu'])
    gyro = np.asarray([r['gyro'] for r in streams['imu']], float)
    desired = start + np.arange(0., duration, period)
    sparse_index = np.searchsorted(imu_time, desired, side='left')
    if np.any(sparse_index >= len(imu_time)) or len(np.unique(sparse_index)) != len(sparse_index):
        raise ValueError('Requested sparse observations are not supported by distinct native readings')
    t0 = float(imu_time[sparse_index[0]])
    dense = (imu_time >= t0) & (imu_time <= t0 + duration)
    pose_time = local(streams['mocap'])
    if any(row['frame'].split('/')[-1] != 'world' for row in streams['mocap']):
        raise ValueError('Mocap reference is not expressed in world')
    quaternion = np.asarray([r['quaternion'] for r in streams['mocap']], float)
    position = np.asarray([r['position'] for r in streams['mocap']], float)
    if not pose_time[0] <= t0 < t0 + duration <= pose_time[-1]:
        raise ValueError('Mocap reference does not bracket the requested prediction interval')
    right = int(np.searchsorted(pose_time, t0, side='left'))
    bracket = np.unique([right if pose_time[right] == t0 else right - 1, right])
    q0 = (quaternion[bracket[0]] if len(bracket) == 1 else
          Slerp(pose_time[bracket], Rotation.from_quat(quaternion[bracket]))([t0]).as_quat()[0])
    eval_pose = (pose_time >= t0) & (pose_time <= t0 + duration)
    # Include the initial alignment bracket and one endpoint bracket. All
    # noninitial mocap remains a read-only reference stream.
    reference_index = np.arange(max(0, bracket[0]),
        min(len(pose_time), int(np.searchsorted(pose_time, t0 + duration, side='right')) + 1))
    state = streams['state']
    if not state or state[0]['state'] != 0:
        raise ValueError('The calibration prefix does not start with ARM_OFF')
    static_end_ns = next(r['record_ns'] for r in state if r['state'] != 0)
    stamps = np.asarray([r['stamp_ns'] for r in streams['imu']], dtype=np.int64)
    static = (stamps >= state[0]['record_ns']) & (stamps < static_end_ns)
    covariance, raw_covariance = helper.isotropic_noise(gyro[static])
    past = ((imu_time >= float(calibration_window[0]))
            & (imu_time <= float(calibration_window[1])))
    if not np.any(past) or imu_time[past][-1] >= t0:
        raise ValueError('Residual calibration must contain original strictly-past gyro observations')
    data = dict(time=np.array([t0, t0 + duration]),
        requested_sparse_gyro_time=desired,
        imu_time=imu_time, imu_stamp_ns=stamps,
        imu_record_ns=np.asarray([r['record_ns'] for r in streams['imu']], np.int64),
        imu_source_index=np.arange(len(imu_time), dtype=np.int64), gyro=gyro,
        dense_imu_time=imu_time[dense], dense_gyro=gyro[dense],
        dense_imu_source_index=np.flatnonzero(dense),
        sparse_gyro_time=imu_time[sparse_index], sparse_gyro=gyro[sparse_index],
        sparse_gyro_stamp_ns=stamps[sparse_index],
        sparse_gyro_record_ns=np.asarray([streams['imu'][i]['record_ns'] for i in sparse_index], np.int64),
        sparse_gyro_source_index=sparse_index,
        calibration_time=imu_time[static], calibration_gyro=gyro[static],
        calibration_source_index=np.flatnonzero(static),
        past_calibration_gyro_time=imu_time[past], past_calibration_gyro=gyro[past],
        past_calibration_source_index=np.flatnonzero(past),
        gyro_noise_covariance=covariance,
        gyro_noise_covariance_raw_diagnostic=raw_covariance,
        mocap_time=pose_time[reference_index],
        mocap_quaternion_xyzw=quaternion[reference_index],
        mocap_position=position[reference_index],
        mocap_source_index=reference_index,
        mocap_stamp_ns=np.asarray([streams['mocap'][i]['stamp_ns'] for i in reference_index], np.int64),
        mocap_record_ns=np.asarray([streams['mocap'][i]['record_ns'] for i in reference_index], np.int64),
        initial_quaternion_wxyz=q0[[3, 0, 1, 2]],
        initial_mocap_bracket_time=pose_time[bracket],
        initial_mocap_bracket_source_index=bracket,
        gyro_sensor_position_in_body=helper.SENSOR_POSITION.copy(),
        body_to_gyro_sensor_rotation=np.eye(3))
    command_metadata = {}
    for source, key, values in (('rotor', 'rotor_command', 'rotor_command'),
                                 ('joint', 'gimbal', 'gimbal_angle'),
                                 ('gimbal_command', 'gimbal_command', 'gimbal_command')):
        rows = streams[source]
        clock = local(rows)
        stop = int(np.searchsorted(clock, t0 + duration, side='right'))
        if source == 'joint':
            stop = min(len(clock), stop + 1)
        data[key + '_time'] = clock[:stop]
        data[values] = np.asarray([r['value'] for r in rows], float)[:stop]
        data[key + '_source_index'] = np.arange(stop, dtype=np.int64)
        data[key + '_stamp_ns'] = np.asarray([r['stamp_ns'] for r in rows[:stop]], np.int64)
        data[key + '_record_ns'] = np.asarray([r['record_ns'] for r in rows[:stop]], np.int64)
        command_metadata[source] = dict(count=stop,
            clock='header' if source == 'joint' else 'bag record',
            pre_evaluation_count=int(np.count_nonzero(clock[:stop] < t0)))
    model = json.loads(MODEL.read_text())
    data['nominal_inertia'] = np.asarray(model['inertia_kg_m2'])
    data['nominal_cog_position_body'] = np.asarray(model['cog_position_body_m'])
    metadata = dict(schema='grape/nonlinear-sparse-gyro-input/v1',
        bag_path=str(Path(bag_path).resolve()), bag_size_bytes=Path(bag_path).stat().st_size,
        bag_sha256_computed=False, time_origin_ns=int(origin),
        requested_window_seconds=[start, start + duration],
        actual_window_seconds=[t0, t0 + duration],
        requested_gyro_rate_hz=1. / period,
        sparse_selection='first original native gyro header stamp at or after each requested bag-local epoch; no averaging',
        sparse_requested_epochs=desired.tolist(), sparse_actual_epochs=imu_time[sparse_index].tolist(),
        sparse_source_indices=sparse_index.tolist(),
        actual_sparse_intervals_seconds=np.diff(imu_time[sparse_index]).tolist(),
        source_clock='native spinal/Imu.stamp and mocap/joint header stamps; issued rotor/gimbal commands use bag record time',
        units=dict(time='s from exact first bag record ns', gyro='rad/s',
            rotor_command='N base_thrust', gimbal_angle='rad', quaternion='xyzw except initial wxyz'),
        source_indices='original per-topic prefix indices; not renumbered after extraction',
        extrinsics=helper._verify_extrinsics(streams),
        native_gyro_body_alignment='spinal/Imu.gyro copied directly into FC/baselink omega; static main_body->FC rotation is identity',
        initial_mocap_alignment=dict(time=t0, bracket_times=pose_time[bracket].tolist(),
            bracket_source_indices=bracket.tolist(), use='one initial quaternion only'),
        mocap_role_after_initialization='read-only orientation and height reference, never predictor/torque calibration input',
        dense_gyro_role='past calibration 40–45 s, and read-only high-rate reference; evaluation predictor may read sparse_gyro only',
        stationary_noise_diagnostic=dict(start_seconds=float(imu_time[static][0]),
            end_seconds=float(imu_time[static][-1]), count=int(static.sum()),
            formula='trace(unbiased sample covariance)/3 * I; no division by N and no bias subtraction',
            means_not_subtracted=gyro[static].mean(0).tolist()),
        past_residual_calibration_window=list(map(float, calibration_window)),
        command_history=command_metadata,
        recorded_startup_gains={g:streams['gain_' + g][0] for g in GROUPS},
        model_path=str(MODEL), model_sha256=sha(MODEL),
        extraction_code_sha256=sha(Path(__file__)),
        helper_code_sha256=sha(MINIMAL / 'imu-first-order-lag/imu_data.py'),
        full_bag_or_video_copied=False, synthetic_noise_added=False,
        identified_model_from_another_bag_used=False)
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=BAG)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--start', type=float, default=50.)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--gyro-period', type=float, default=2.)
    args = parser.parse_args()
    arrays, metadata = extract(args.bag, args.start, args.duration, args.gyro_period)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'input.npz', **arrays)
    metadata['arrays_sha256'] = sha(args.output / 'input.npz')
    (args.output / 'input.json').write_text(json.dumps(metadata, indent=2,
        ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), arrays=len(arrays),
        sparse_time=arrays['sparse_gyro_time'].tolist(),
        sparse_gyro=arrays['sparse_gyro'].tolist(),
        nominal_inertia=arrays['nominal_inertia'].tolist()), indent=2))


if __name__ == '__main__':
    main()
