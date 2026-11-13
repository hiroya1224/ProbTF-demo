#!/usr/bin/env python3
"""Smooth artificial attitude, noisy gyro, and matched linear/nonlinear dynamics."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
from trajectory import generate_dataset
from plotting import render_case, render_overview, render_generated_motion
from audit import audit_dataset
for directory in (PARENT, PARENT/'rate_comparison_5hz'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
from engine import predict_factorized
from torque import TorqueProfile

RATES = (.5, 2., 5.)
SIGMAS = (.1, .5, 1., 2., 2.5, 5.)
MODES = ('nonlinear', 'linear_forced')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=encode, ensure_ascii=False,
                                   indent=2, allow_nan=False)+'\n')


def label(value):
    return format(value, 'g').replace('.', 'p')


def select_gyro(data, rate):
    start, end = data['time']
    desired = np.arange(start, end, 1./float(rate))
    desired_ns = np.rint(desired*1e9).astype(np.int64)
    indices = np.searchsorted(data['imu_stamp_ns'], desired_ns)
    selected = dict(time=data['time'].copy(), initial_quaternion_wxyz=data['initial_quaternion_wxyz'].copy(),
                    sparse_gyro_time=data['imu_time'][indices].copy(), sparse_gyro=data['gyro'][indices].copy())
    evidence = dict(requested_stamp_ns=desired_ns, stamp_ns=data['imu_stamp_ns'][indices],
                    time=selected['sparse_gyro_time'], gyro=selected['sparse_gyro'],
                    source_index=data['imu_source_index'][indices])
    return selected, evidence


def score(data, prediction):
    common = np.linspace(*data['time'], 501)
    center = np.column_stack([np.interp(common, prediction['time'], prediction['rpy_mean_rad'][:, a]) for a in range(3)])
    covariance = np.stack([np.column_stack([
        np.interp(common, prediction['time'], prediction['rpy_covariance_rad2'][:, a, b]) for b in range(3)]) for a in range(3)], axis=1)
    reference = Slerp(data['reference_time'], Rotation.from_quat(data['reference_quaternion_xyzw']))(common)
    rpy_reference = np.unwrap(reference.as_euler('xyz'), axis=0)
    errors = (center-rpy_reference+np.pi)%(2*np.pi)-np.pi
    geodesic = (reference.inv()*Rotation.from_euler('xyz', center)).magnitude()
    arrays = dict(time=common, rpy_proxy_rad=center, rpy_covariance_delta_rad2=covariance,
                  reference_rpy_rad=rpy_reference, error_rpy_rad=errors, error_geodesic_rad=geodesic)
    metric = dict(so3_proxy_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(geodesic**2)))),
                  so3_proxy_max_deg=float(np.rad2deg(np.max(geodesic))),
                  so3_proxy_final_deg=float(np.rad2deg(geodesic[-1])),
                  rpy_proxy_rmse_deg=np.rad2deg(np.sqrt(np.mean(errors**2, axis=0))),
                  final_rpy_two_sigma_deg=2*np.rad2deg(np.sqrt(np.diag(covariance[-1]))),
                  score_count=len(common),
                  exact_moment_internal_difference=float(np.max(prediction['core_vs_rotation_mean_max_abs'])))
    return arrays, metric


def write_index(output, records, args):
    lines = ['# 合成gyro：線形・非線形回転動力学の比較', '',
             '[sigma=2.5 deg/sの一覧図](rate_overview_sigma_2p5.pdf) / '
             '[生成した運動と入力](generated_motion.pdf) / [CSV](comparison_table.csv) / [方法](../../README.md)', '',
             '共通の既知J・既知入力トルク・同じgyroを使用。線形側はEuler式のジャイロ項だけを省く。',
             '生成ノイズと仮定する候補分散を分離し、真値を予測へ戻さない。帯は同じ回転モーメントからの局所±2sigma表示。', '',
             '| 仮定するsigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |', '|---:|---|---|---|']
    for sigma in args.sigmas:
        lines.append('| '+format(sigma, 'g')+' | '+' | '.join(
            f'[PDF](rate_{label(rate)}_hz/sigma_{label(sigma)}_deg_s/comparison.pdf)' for rate in args.rates)+' |')
    lines.extend(['', 'sigma=2.5 deg/s、501共通時刻での姿勢中心のSO(3) RMSE [deg]：', '',
                  '| gyro Hz | 取得数 | 非線形Euler | ジャイロ項なし |', '|---:|---:|---:|---:|'])
    for row in records:
        if row['assumed_sigma_deg_s'] == 2.5:
            lines.append(f"| {row['rate_hz']:g} | {row['gyro_count']} | "+' | '.join(
                f"{row['methods'][mode]['so3_proxy_rmse_deg']:.4f}" for mode in MODES)+' |')
    lines.extend(['', 'ノイズなし・biasなしの理想再生と数値整合は[audit.json](audit.json)に別保存。'
                  '主結果の測定列は変更しない。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    begun = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data, metadata = generate_dataset(duration=args.duration, sensor_rate=args.sensor_rate, seed=args.seed)
    np.savez_compressed(output/'input.npz', **data)
    write_json(output/'input.json', metadata)
    # Predictor receives only the declared control, J, q0 and selected gyro.
    profile = dict(inertia=data['inertia_body_kg_m2'].copy(),
                   torque=TorqueProfile(data['torque_time'], data['torque_body_nm']))
    render_generated_motion(output, data)
    print('Checking analytic kinematics and ideal matched dynamics', flush=True)
    audit = audit_dataset(data, profile)
    write_json(output/'audit.json', audit)
    records, overview = [], {}
    for rate in args.rates:
        selected, evidence = select_gyro(data, rate)
        directory = output/('rate_'+label(rate)+'_hz')
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/'acquisitions.npz', **evidence)
        for sigma in args.sigmas:
            print('Synthetic comparison:', rate, 'Hz, sigma', sigma, 'deg/s', flush=True)
            destination = directory/('sigma_'+label(sigma)+'_deg_s')
            destination.mkdir(exist_ok=True)
            curves, metrics = {}, {}
            for mode in MODES:
                prediction = predict_factorized(selected, profile, sigma, mode, output_dt=.02, max_step=args.max_step)
                curves[mode], metrics[mode] = score(data, prediction)
                prediction['rpy_proxy_rad'] = prediction.pop('rpy_mean_rad')
                prediction['rpy_covariance_delta_rad2'] = prediction.pop('rpy_covariance_rad2')
                np.savez_compressed(destination/(mode+'.npz'), **prediction)
                np.savez_compressed(destination/(mode+'_plot.npz'), **curves[mode])
            record = dict(rate_hz=rate, assumed_sigma_deg_s=sigma, gyro_count=len(evidence['time']),
                          actual_acquisition_times=evidence['time'], methods=metrics,
                          covariance_per_reading_rad2_s2=np.eye(3)*np.deg2rad(sigma)**2,
                          same_realized_gyro_for_all_sigmas=True)
            write_json(destination/'metrics.json', record)
            render_case(destination, rate, sigma, curves, evidence['time'])
            if sigma == 2.5: overview[rate] = curves
            records.append(record)
    if overview: render_overview(output, overview)
    summary = dict(schema='grape/synthetic-controlled-rotation-comparison/v1',
        generation=metadata, duration_seconds=args.duration, rates_hz=args.rates, assumed_sigmas_deg_s=args.sigmas,
        results=records, audit=audit, known_inertia=data['inertia_body_kg_m2'],
        initial_attitude='known q0 once; no later truth corrections',
        predictor_data_keys=['time', 'initial_quaternion_wxyz', 'sparse_gyro_time', 'sparse_gyro'],
        known_control='recorded 1 kHz torque generated from prescribed motion by inverse nonlinear Euler; identical in both methods',
        linear_definition='only omega cross Jomega omitted from angular-rate dynamics; identical nonlinear quaternion kinematics',
        law='independent six-point gyro increment factors; unchanged core quaternion second moments and full rotation covariance',
        display='Euler center f(E[R]) and delta-method covariance; conditional local 2sigma, not guaranteed 95% coverage',
        no_monte_carlo_in_estimation=True, random_numbers_only_in_sensor_generation=True,
        bias_estimated_or_subtracted=False, wall_seconds=time.monotonic()-begun,
        source_sha256={p.name: sha(p) for p in HERE.glob('*.py')},
        shared_source_sha256={str(p.relative_to(PARENT)): sha(p) for p in
            (PARENT/'propagation.py', PARENT/'torque.py', PARENT/'rate_comparison_5hz/engine.py')})
    write_json(output/'summary.json', summary)
    with (output/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['rate_hz', 'assumed_sigma_deg_s', 'method', 'gyro_count',
                         'so3_proxy_rmse_deg', 'so3_proxy_max_deg', 'so3_proxy_final_deg',
                         'roll_rmse_deg', 'pitch_rmse_deg', 'yaw_rmse_deg',
                         'roll_final_2sigma_deg', 'pitch_final_2sigma_deg', 'yaw_final_2sigma_deg'])
        for row in records:
            for mode, metric in row['methods'].items():
                writer.writerow([row['rate_hz'], row['assumed_sigma_deg_s'], mode, row['gyro_count'],
                    metric['so3_proxy_rmse_deg'], metric['so3_proxy_max_deg'], metric['so3_proxy_final_deg'],
                    *metric['rpy_proxy_rmse_deg'], *metric['final_rpy_two_sigma_deg']])
    write_index(output, records, args)
    print('Completed', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-controlled-rotation-20261113/results'))
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--sensor-rate', type=float, default=200.)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--rates', type=float, nargs='+', default=list(RATES))
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS))
    parser.add_argument('--max-step', type=float, default=.002)
    run(parser.parse_args())
