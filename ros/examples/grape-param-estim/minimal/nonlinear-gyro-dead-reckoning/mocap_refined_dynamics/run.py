#!/usr/bin/env python3
"""Nominal-start mocap dynamics refinement and held-out gyro rate/noise sweep."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for directory in (HERE, PARENT, PARENT/'rate_comparison_5hz'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from kinematics import prepare_kinematics
from fit import fit_rotational_model, build_refined_profile
from engine import predict_factorized
from plotting import render_case, render_overview, render_dynamics
from report import render_report
from verification import verify_unused_reference_invariance
from identifiability import audit_fit
from numeric_audit import audit_numerics
from error_budget import build_drift_report

_spec = importlib.util.spec_from_file_location('mocap_refined_existing_rate_helpers', PARENT/'rate_comparison_5hz/run.py')
_rates = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _rates
_spec.loader.exec_module(_rates)
acquisitions = _rates.acquisitions
common_statistics = _rates.common_statistics
write_json = _rates.write_json
label = _rates.label
RATES = (.5, 2., 5.)
SIGMAS = (.1, .5, 1., 2., 2.5, 5.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def clip_training_inputs(data, kinematics):
    """The optimizer receives no evaluation-time sensor or control arrays."""
    end = float(kinematics['time'][-1])
    result = {'time': np.asarray(data['time']).copy()}
    for prefix, value_name in (('rotor_command', 'rotor_command'), ('gimbal', 'gimbal_angle')):
        stamps = np.asarray(data[prefix+'_time'])
        # One right bracket is needed only for measured-angle interpolation.
        count = np.searchsorted(stamps, end, side='right') + (prefix == 'gimbal')
        result[prefix+'_time'] = stamps[:count].copy()
        result[value_name] = np.asarray(data[value_name])[:count].copy()
    return result


def numeric_fields(record):
    return {key: np.asarray(value) for key, value in record.items()
            if isinstance(value, (np.ndarray, int, float, np.number))}


def derivative_check(data, kinematics, profiles):
    """Read-only diagnostic AFTER fit; never supplied to the predictor."""
    result = numeric_fields(kinematics)
    times = np.asarray(kinematics['time'])
    omega = kinematics['omega_body_rad_s']
    alpha = kinematics['alpha_body_rad_s2']
    result['gyro_reference_rad_s'] = np.column_stack([
        np.interp(times, data['imu_time'], data['gyro'][:, dimension]) for dimension in range(3)])
    metrics = {}
    for name, profile in profiles.items():
        inertia = profile['inertia']
        gyroscopic = np.cross(omega, omega@inertia.T)
        required = alpha@inertia.T + gyroscopic
        applied = profile['torque'](times)
        predicted_alpha = np.linalg.solve(inertia, (applied-gyroscopic).T).T
        error = predicted_alpha-alpha
        result[name+'_gyroscopic_torque_nm'] = gyroscopic
        result[name+'_required_torque_nm'] = required
        result[name+'_applied_torque_nm'] = applied
        result[name+'_predicted_alpha_rad_s2'] = predicted_alpha
        metrics[name] = dict(alpha_rmse_axis_rad_s2=np.sqrt(np.mean(error**2, axis=0)),
            alpha_vector_rmse_rad_s2=np.sqrt(np.mean(np.sum(error**2, axis=1))),
            torque_rmse_axis_nm=np.sqrt(np.mean((applied-required)**2, axis=0)),
            gyroscopic_rms_axis_nm=np.sqrt(np.mean(gyroscopic**2, axis=0)),
            inertial_rms_axis_nm=np.sqrt(np.mean((alpha@inertia.T)**2, axis=0)))
    return result, metrics


def write_index(output, summary):
    lines = ['# mocapでrefineした非線形回転モデル', '',
        '公称値を初期値にした最小二乗。調整は40〜45秒、予測評価は50〜55秒。',
        '元の推定・旧図を変更せず、非線形Euler項と指令トルクを含めて比較した。', '',
        '[周波数一覧図（sigma=2.5 deg/s）](rate_overview_sigma_2p5.pdf) / '
        '[動力学の当てはまり](dynamics_fit.pdf) / [パラメータ表](fit_report.pdf) / '
        '[全条件CSV](comparison_table.csv) / [結果と制約](RESULTS.md) / [方法](../../README.md)', '',
        '| sigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |', '|---:|---|---|---|']
    for sigma in summary['sigmas_deg_s']:
        links = [f'[PDF](rate_{label(rate)}_hz/sigma_{label(sigma)}_deg_s/comparison.pdf)' for rate in summary['rates_hz']]
        lines.append('| '+format(sigma, 'g')+' | '+' | '.join(links)+' |')
    lines.extend(['', 'sigma=2.5 deg/sでのSO(3) RMSE [deg]。全条件で同じ501時刻を評価。', '',
                  '| Hz | 取得数 | 公称初期モデル | refine後 |', '|---:|---:|---:|---:|'])
    for record in summary['results']:
        if record['sigma_deg_s'] == 2.5:
            lines.append('| '+str(record['rate_hz'])+' | '+str(record['original_gyro_count'])+' | '+
                ' | '.join(f"{record['methods'][name]['geodesic_rmse_deg']:.3f}" for name in ('nominal', 'refined'))+' |')
    lines.extend(['', '帯は固定したモデル・固定した過去余剰平均を条件とする角速度不確かさだけ。',
                  '全回転モーメントを保存し、RPY中心・帯だけにdelta-method表示近似を使う。',
                  '評価中のmocap・高密度gyroから余剰トルクを算出して入力へ戻していない。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.input, allow_pickle=False))
    metadata = json.loads(args.input.with_suffix('.json').read_text())
    mocap_data = dict(np.load(args.mocap_input, allow_pickle=False))
    train_mask = (mocap_data['mocap_time'] >= args.training_start) & (mocap_data['mocap_time'] <= args.training_end)
    train_times = mocap_data['mocap_time'][train_mask]
    if not len(train_times) or args.training_end + args.sg_window/2 >= data['time'][0]:
        raise ValueError('Training including its centered SG support must precede prediction')
    training = prepare_kinematics(mocap_data, train_times, args.sg_window, 3)
    train_inputs = clip_training_inputs(data, training)
    np.savez_compressed(args.output/'training_kinematics.npz', **numeric_fields(training))
    print('Fitting nominal-start nonlinear Euler model:', len(train_times), 'mocap rows', flush=True)
    fitted = fit_rotational_model(train_inputs, training, args.model,
                                 initial_tau=args.initial_tau, max_nfev=args.max_nfev)
    write_json(args.output/'fit.json', fitted['parameters'])
    np.savez_compressed(args.output/'fit_arrays.npz', **fitted['arrays'])
    identifiability = audit_fit(train_inputs, training, fitted)
    write_json(args.output/'identifiability.json', identifiability['metadata'])
    np.savez_compressed(args.output/'identifiability.npz', **identifiability['arrays'])
    print('Fit complete; building frozen torque profiles', flush=True)
    profiles = dict(refined=build_refined_profile(data, fitted),
                    nominal=build_refined_profile(data, fitted, use_initial=True))
    for name, profile in profiles.items():
        np.savez_compressed(args.output/(name+'_torque_profile.npz'), **profile['arrays'])
        write_json(args.output/(name+'_torque_profile.json'), profile['metadata'])

    # Independent held-out reference is read only after all parameter fitting.
    eval_mask = (mocap_data['mocap_time'] >= data['time'][0]) & (mocap_data['mocap_time'] <= data['time'][1])
    evaluation = prepare_kinematics(mocap_data, mocap_data['mocap_time'][eval_mask], args.sg_window, 3)
    checks, check_metrics = {}, {}
    for name, kin in (('training', training), ('heldout', evaluation)):
        checks[name], check_metrics[name] = derivative_check(data, kin, profiles)
        np.savez_compressed(args.output/(name+'_dynamics_check.npz'), **checks[name])
    write_json(args.output/'dynamics_check.json', check_metrics)
    render_dynamics(args.output, checks)

    records, overview = [], {}
    data_flow_check = None
    numeric_check = None
    for rate in args.rates:
        selected, evidence = acquisitions(data, metadata, rate)
        directory = args.output/('rate_'+label(rate)+'_hz')
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/'acquisitions.npz', **evidence)
        for sigma in args.sigmas:
            print('Prediction:', rate, 'Hz, sigma', sigma, 'deg/s', flush=True)
            destination = directory/('sigma_'+label(sigma)+'_deg_s')
            destination.mkdir(exist_ok=True)
            curves, metrics = {}, {}
            for name in ('refined', 'nominal'):
                profile = profiles['nominal' if name == 'nominal' else 'refined']
                mode = 'nonlinear'
                prediction = predict_factorized(selected, profile, sigma, mode, output_dt=.02, max_step=args.max_step)
                if name == 'refined' and rate == 5 and sigma == 2.5:
                    data_flow_check = verify_unused_reference_invariance(selected, profile, sigma, prediction,
                                                                        predict_factorized, args.max_step)
                    write_json(args.output/'data_flow_verification.json', data_flow_check)
                    numeric_check = audit_numerics(data, selected, fitted, profile, prediction,
                                                   predict_factorized, sigma, args.max_step)
                    write_json(args.output/'numerical_sensitivity.json', numeric_check)
                curves[name], metrics[name] = common_statistics(data, prediction)
                prediction['rpy_proxy_rad'] = prediction.pop('rpy_mean_rad')
                prediction['rpy_covariance_delta_rad2'] = prediction.pop('rpy_covariance_rad2')
                np.savez_compressed(destination/(name+'.npz'), **prediction)
                np.savez_compressed(destination/(name+'_plot.npz'), **curves[name])
            record = dict(rate_hz=rate, sigma_deg_s=sigma, original_gyro_count=len(evidence['time']),
                actual_observation_times=evidence['time'], methods=metrics,
                covariance_rad2_s2=np.eye(3)*np.deg2rad(sigma)**2,
                synthetic_sensor_noise_added=False, same_measurements_all_sigmas=True)
            write_json(destination/'metrics.json', record)
            render_case(destination, rate, sigma, curves, evidence['time'])
            records.append(record)
            if sigma == 2.5:
                overview[rate] = curves
    if overview:
        render_overview(args.output, overview)
    summary = dict(schema='grape/mocap-refined-nonlinear-dynamics/v1',
        training_interval=[args.training_start, args.training_end],
        evaluation_interval=data['time'], sg_window_seconds=args.sg_window, sg_degree=3,
        inputs={name: dict(path=str(path.resolve()), sha256=sha(path))
                for name, path in (('archived_controls_and_gyro', args.input), ('archived_mocap', args.mocap_input), ('nominal_model', args.model))},
        training_input_sha256={key: array_hash(value) for key, value in train_inputs.items()},
        fit_kinematics_sha256={key: array_hash(training[key]) for key in ('time', 'omega_body_rad_s', 'alpha_body_rad_s2')},
        rates_hz=args.rates, sigmas_deg_s=args.sigmas, results=records,
        derivative_checks=check_metrics, fit=fitted['parameters'],
        identifiability=identifiability['metadata'], data_flow_verification=data_flow_check,
        numerical_sensitivity=numeric_check,
        nonlinear_gyroscopic_term_retained=True,
        evaluation_mocap_in_predictor=False, dense_evaluation_gyro_in_predictor=False,
        baseline='same nominal-start parameters and initial tau, with its own strictly past mocap residual mean; previous other-bag tau not reused',
        torque='recorded issued rotor thrust, measured gimbal, refined first-order actuator response and frozen training residual mean',
        candidate_law='independent six-point gyro increment factors, exact quaternion second and rotation first/full covariance moments',
        displayed_rpy='f(E[R]) and local delta-method covariance; not exact Euler moments or guaranteed coverage',
        bands='conditional on point physical parameters and frozen past torque mean; no parameter covariance or unknown future torque uncertainty included',
        no_monte_carlo=True, wall_seconds=time.monotonic()-started,
        source_sha256={p.name: sha(p) for p in HERE.glob('*.py')},
        shared_source_sha256={str(p.relative_to(PARENT)): sha(p) for p in
            (PARENT/'propagation.py', PARENT/'torque.py', PARENT/'rate_comparison_5hz/engine.py',
             PARENT/'rate_comparison_5hz/run.py')})
    write_json(args.output/'summary.json', summary)
    with (args.output/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['rate_hz', 'sigma_deg_s', 'method', 'gyro_count', 'so3_rmse_deg', 'so3_max_deg', 'so3_final_deg',
                         'roll_rmse_deg', 'pitch_rmse_deg', 'yaw_rmse_deg', 'roll_final_2sigma_deg', 'pitch_final_2sigma_deg', 'yaw_final_2sigma_deg'])
        for record in records:
            for name, metric in record['methods'].items():
                writer.writerow([record['rate_hz'], record['sigma_deg_s'], name, record['original_gyro_count'],
                    metric['geodesic_rmse_deg'], metric['geodesic_max_deg'], metric['geodesic_final_deg'],
                    *metric['rpy_proxy_rmse_deg'], *metric['final_rpy_two_sigma_delta_deg']])
    write_index(args.output, summary)
    render_report(args.output)
    build_drift_report(args.output, mocap_input=args.mocap_input)
    print('Completed', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PARENT/'publication/bag8_50_55s/input.npz')
    parser.add_argument('--mocap-input', type=Path, default=PARENT/'gyro_mocap_sg_check/publication/bag8_50_55s/input.npz')
    parser.add_argument('--model', type=Path, default=PARENT.parent/'grape_vehicle_model.json')
    parser.add_argument('--output', type=Path, default=Path('/tmp/mocap-refined-dynamics-20261113/results'))
    parser.add_argument('--training-start', type=float, default=40.)
    parser.add_argument('--training-end', type=float, default=45.)
    parser.add_argument('--sg-window', type=float, default=.32)
    parser.add_argument('--initial-tau', type=float)
    parser.add_argument('--max-nfev', type=int, default=300)
    parser.add_argument('--max-step', type=float, default=.002)
    parser.add_argument('--rates', type=float, nargs='+', default=list(RATES))
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS))
    run(parser.parse_args())
