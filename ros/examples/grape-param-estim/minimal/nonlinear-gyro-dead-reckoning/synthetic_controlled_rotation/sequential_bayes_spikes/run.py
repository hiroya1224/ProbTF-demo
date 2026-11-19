#!/usr/bin/env python3
"""Causal joint Bayesian rotation/rate inference with a saved spike stress test."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
SHARED = EXPERIMENT.parent
sys.path.insert(0, str(SHARED))
from torque import TorqueProfile


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


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_and_corrupt(source, config):
    """One seeded perturbation realization, with no estimation-dependent choice.

    Random timing means selection among acquired noninitial sensor instants.
    This conditions on observing the requested number of spikes. It is not
    an estimate of a natural native-sensor outlier frequency.
    """
    begin, end = source['time']
    requested_ns = np.rint(np.arange(begin, end, 1./config['rate_hz'])*1e9).astype(np.int64)
    index = np.searchsorted(source['imu_stamp_ns'], requested_ns)
    white = source['gyro'][index].copy()
    rng = np.random.default_rng(int(config['spike_seed']))
    count = int(config['spike_count'])
    if count < 0 or count > len(index)-1:
        raise ValueError('Spike count must fit the noninitial acquisitions')
    chosen = np.sort(rng.choice(np.arange(1, len(index)), size=count, replace=False))
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0.):
        raise FloatingPointError('Generated zero spike direction')
    lower, upper = config['spike_magnitude_range_deg_s']
    magnitudes = rng.uniform(lower, upper, size=count)
    perturbation = np.zeros_like(white)
    perturbation[chosen] = np.deg2rad(magnitudes)[:, None]*directions/norms[:, None]
    mask = np.zeros(len(index), dtype=bool)
    mask[chosen] = True
    corrupted = white+perturbation
    acquisitions = dict(time=source['imu_time'][index].copy(),
        requested_stamp_ns=requested_ns, stamp_ns=source['imu_stamp_ns'][index].copy(),
        source_index=source['imu_source_index'][index].copy(),
        gyro_white=white, gyro_spike=corrupted,
        spike_mask=mask, spike_vector_rad_s=perturbation)
    native_mask = np.zeros(len(source['imu_time']), dtype=bool)
    native_mask[index[chosen]] = True
    native_spike = np.zeros_like(source['gyro'])
    native_spike[index] = perturbation
    sensor = dict(time=source['imu_time'].copy(), stamp_ns=source['imu_stamp_ns'].copy(),
        source_index=source['imu_source_index'].copy(), gyro_white=source['gyro'].copy(),
        gyro_spike=source['gyro']+native_spike,
        spike_mask=native_mask, spike_vector_rad_s=native_spike)
    predictor = dict(time=source['time'].copy(),
        initial_quaternion_wxyz=source['initial_quaternion_wxyz'].copy(),
        sparse_gyro_time=acquisitions['time'].copy(), sparse_gyro=white.copy())
    return predictor, acquisitions, sensor


def score(source, case):
    """Score online outputs only; right-hand posterior is used at acquisition."""
    stamps = np.linspace(*source['time'], 501)
    def interp(name):
        values = case[name]
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        result = np.column_stack([np.interp(stamps, case['time'], flat[:, i])
                                  for i in range(flat.shape[1])])
        return result.reshape((len(stamps),)+shape)
    rpy = interp('rpy_proxy_rad')
    omega = interp('omega_mean_rad_s')
    truth_rpy = np.column_stack([np.interp(stamps, source['reference_time'],
                      source['reference_rpy_rad'][:, i]) for i in range(3)])
    truth_omega = np.column_stack([np.interp(stamps, source['reference_time'],
                      source['reference_omega_body_rad_s'][:, i]) for i in range(3)])
    errors = (Rotation.from_euler('xyz', truth_rpy).inv()
              *Rotation.from_euler('xyz', rpy)).magnitude()
    omega_error = omega-truth_omega
    return dict(so3_proxy_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
        so3_proxy_max_deg=float(np.rad2deg(errors.max())),
        so3_proxy_final_deg=float(np.rad2deg(errors[-1])),
        omega_rmse_deg_s=np.rad2deg(np.sqrt(np.mean(omega_error**2, axis=0))),
        omega_final_error_deg_s=np.rad2deg(omega_error[-1]),
        final_omega_two_sigma_deg_s=2*np.rad2deg(np.sqrt(np.diag(case['omega_covariance_rad2_s2'][-1]))),
        final_rpy_two_sigma_deg=2*np.rad2deg(np.sqrt(np.diag(case['rpy_covariance_delta_rad2'][-1]))),
        score_count=len(stamps)), dict(error_time=stamps, error_geodesic_rad=errors,
                                       error_omega_rad_s=omega_error)


def refinement(coarse, fine):
    """Numerical integration diagnostics, never an observation acceptance gate."""
    assert np.array_equal(coarse['time'], fine['time'])
    mean_difference = fine['omega_mean_rad_s']-coarse['omega_mean_rad_s']
    covariance_difference = (fine['omega_covariance_rad2_s2']
                             -coarse['omega_covariance_rad2_s2'])
    rpy_difference = (fine['rpy_proxy_rad']-coarse['rpy_proxy_rad']+np.pi)%(2*np.pi)-np.pi
    return dict(max_omega_mean_component_difference_deg_s=float(np.max(np.abs(np.rad2deg(mean_difference)))),
        max_omega_covariance_component_difference_deg2_s2=float(np.max(np.abs(covariance_difference))*(180/np.pi)**2),
        max_rpy_proxy_component_difference_deg=float(np.max(np.abs(np.rad2deg(rpy_difference)))),
        max_rotation_covariance_component_difference=float(np.max(np.abs(
            fine['rotation_covariance_colmajor']-coarse['rotation_covariance_colmajor']))),
        finite_diagnostic_only=True)


def save_case(path, case):
    arrays = {key: value for key, value in case.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)


def reference_event_diagnostics(source, cases, events):
    """Read-only evaluation; these reference values never reach the filter."""
    for name, records in events.items():
        case = cases[name]
        for record in records:
            if record['initialization']:
                continue
            stamp = record['time']
            rows = np.flatnonzero(case['time']==stamp)
            truth = np.array([np.interp(stamp, source['reference_time'],
                               source['reference_rpy_rad'][:, axis]) for axis in range(3)])
            for side, label in ((-1,'prior'),(1,'posterior')):
                row = rows[case['event_side'][rows]==side][0]
                error = (Rotation.from_euler('xyz', truth).inv()
                         *Rotation.from_euler('xyz', case['rpy_proxy_rad'][row])).magnitude()
                record[label+'_so3_proxy_error_deg_reference_only'] = float(np.rad2deg(error))
                record[label+'_rpy_proxy_deg_reference_only'] = np.rad2deg(case['rpy_proxy_rad'][row])
    return events


def report(output, config, acquisitions, metrics, convergence, events):
    contaminated = acquisitions['time'][acquisitions['spike_mask']]
    if len(contaminated):
        last = float(contaminated[-1])
        remaining = int(np.count_nonzero(acquisitions['time']>last))
        recovery_note = f'最後のスパイクは{last:g}秒で、その後の正常な新規観測は{remaining}件。'
        if not remaining:
            recovery_note += 'スパイク後の再観測による回復まで評価した結果ではない。'
    else:
        recovery_note = 'この設定ではスパイクを追加していない。'
    lines = ['# 時系列Bayes更新とスパイク観測', '',
        '[姿勢・角速度の比較](attitude_and_gyro.pdf) / [PNG](attitude_and_gyro.png) / '
        '[観測と幅の診断](bayes_diagnostics.pdf) / [旧方式との比較](before_after.pdf) / '
        '[比較動画・音声なし](sequential_bayes_spike_comparison.mp4)', '',
        f'5秒、{config["rate_hz"]:g} Hz取得、観測の仮定σ={config["assumed_sigma_deg_s"]:g} deg/s。既知の同じ慣性・入力トルク、'
        '元の3周波数軌道と白色ノイズ実現値を使用した。', '',
        f'今回のスパイクは、初期化後の取得{len(acquisitions["time"])-1}件から'
        f'{int(acquisitions["spike_mask"].sum())}件を一様に乱数選択する条件付きストレス試験。'
        '自然なスパイク発生率の推定ではない。乱数seed、実際の時刻・方向・振幅を保存し、'
        '推定側にはスパイクのフラグも補正値も渡していない。', '',
        '| スパイク時刻 [s] | 加算角速度 [deg/s] | ベクトル長 [deg/s] |',
        '|---:|---|---:|']
    for i in np.flatnonzero(acquisitions['spike_mask']):
        value = np.rad2deg(acquisitions['spike_vector_rad_s'][i])
        lines.append(f'| {acquisitions["time"][i]:g} | '+', '.join(f'{v:.4f}' for v in value)
                     +f' | {np.linalg.norm(value):.4f} |')
    lines.extend(['', '| 方式・入力 | 姿勢中心RMSE [deg] | 終端誤差 [deg] | ω RMSE xyz [deg/s] |',
                  '|---|---:|---:|---|'])
    names = dict(white='Bayes・白色のみ', spike='Bayes・スパイクあり',
                 reset_white='旧独立置換・白色のみ', reset_spike='旧独立置換・スパイクあり')
    for name, metric in metrics.items():
        lines.append(f'| {names[name]} | {metric["so3_proxy_rmse_deg"]:.5f} | '
            f'{metric["so3_proxy_final_deg"]:.5f} | '
            +', '.join(f'{v:.5f}' for v in metric['omega_rmse_deg_s'])+' |')
    lines.extend(['', 'このBayes更新は、同じ初期角速度の候補へ、取得時刻順に観測尤度を掛ける。'
        '観測のたびに候補を測定値中心へ置換せず、過去の尤度を保持する。'
        '同じ候補が作る姿勢・角速度と、その交差共分散を保持する。', '',
        'スパイク用のロバスト尤度、外れ値除去、BFS、分散の追加・拡縮は導入していない。'
        'Gaussian観測仮定とスパイクの不整合が、そのまま結果に現れる。', '',
        '## 今回起きたこと', '',
        f'白色のみの姿勢中心RMSEは{metrics["white"]["so3_proxy_rmse_deg"]:.4f}度、'
        f'スパイクありでは{metrics["spike"]["so3_proxy_rmse_deg"]:.4f}度となった。'
        '角速度の観測が姿勢と依存しているため、今回のBayes更新は現在の姿勢の重みも変える。'
        'Gaussian観測モデルはスパイクを異常として区別せず、整合する初期角速度候補へ重みを移す。'
        'その結果、角速度だけでなく現在の姿勢にも大きい誤りが生じた。過去に記録した出力を遡って修正した結果ではない。', '',
        '| スパイク観測時刻 [s] | 姿勢中心誤差・直前→直後 [deg] | ωの±2σ半幅xyz・直前→直後 [deg/s] |',
        '|---:|---|---|'])
    for record in events['spike']:
        if record['initialization'] or not np.any(acquisitions['spike_mask'] & (acquisitions['time']==record['time'])):
            continue
        before = record['prior_so3_proxy_error_deg_reference_only']
        after = record['posterior_so3_proxy_error_deg_reference_only']
        widths = [2*np.rad2deg(np.sqrt(np.diag(record[label+'_omega_covariance_rad2_s2'])))
                  for label in ('prior','posterior')]
        text = ' → '.join('('+', '.join(f'{v:.3f}' for v in width)+')' for width in widths)
        lines.append(f'| {record["time"]:g} | {before:.4f} → {after:.4f} | {text} |')
    lines.extend(['', 'この試行では誤った観測を取り込んでも幅が縮んだ。小さい事後共分散は、'
        '仮定したモデルの下での集中を表しており、観測が正しいことの保証ではない。'
        'スパイク向けの処理は無断で追加していない。', '',
        recovery_note, '',
        '旧独立置換との比較には、旧6点則と新しいtensor Gauss–Hermite則の差も含まれる。'
        'Bayes更新だけを変更した厳密なアブレーションとは呼ばない。'
        '白色のみ／スパイクありの主比較は、同じ新しい求積則で計算している。', '',
        '姿勢帯は同じ回転モーメントからの局所delta-method ±2σであり、'
        '保証付き95%領域ではない。', '',
        '## 決定論的積分の精細化', '',
        '| 次数比較 | 入力 | 最大ω平均差 [deg/s] | 最大ω共分散差 [(deg/s)²] | 最大RPY中心差 [deg] |',
        '|---|---|---:|---:|---:|'])
    for label, cases in convergence.items():
        for name, item in cases.items():
            lines.append(f'| {label} | {name} | {item["max_omega_mean_component_difference_deg_s"]:.6g} | '
                         f'{item["max_omega_covariance_component_difference_deg2_s2"]:.6g} | '
                         f'{item["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '有限求積の差を記録したもので、品質閾値による観測の棄却には用いない。'
        'この固定事前中心の格子は今回の3観測に対するもの。'
        '大量の観測で事後が細くなる場合、同じ次数で正確に積分できるとは限らない。', '',
        '[観測前後の全数値](events.json) / [設定・数値要約](summary.json) / '
        '[方法と再現](../README.md)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    source_dir = (args.config.parent/config['source_directory']).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha(p) for p in source_dir.glob('*.npz')}
    with np.load(source_dir/'input.npz', allow_pickle=False) as bundle:
        source = {key: bundle[key].copy() for key in bundle.files}
    selected, acquisitions, sensor = select_and_corrupt(source, config)
    np.savez_compressed(output/'acquisitions.npz', **acquisitions)
    np.savez_compressed(output/'sensor_input.npz', **sensor)
    write_json(output/'config.json', config)
    profile = dict(inertia=source['inertia_body_kg_m2'].copy(),
                   torque=TorqueProfile(source['torque_time'], source['torque_body_nm']))
    bayes = module_from_path('synthetic_sequential_bayes_engine', HERE/'engine.py')
    old = module_from_path('synthetic_prior_reset_engine', SHARED/'rate_comparison_5hz/engine.py')
    observations = {name: acquisitions['gyro_'+name] for name in ('white', 'spike')}
    all_metrics, convergence, timing = {}, {}, {}
    previous = None
    for order in config['quadrature_orders']:
        start = time.monotonic()
        print('Sequential Bayes quadrature order', order, flush=True)
        cases = bayes.predict_joint(selected, profile, config['assumed_sigma_deg_s'],
            order=order, output_dt=config['output_dt_seconds'], max_step=config['max_step_seconds'],
            observation_sets=observations)
        timing[str(order)] = time.monotonic()-start
        if previous is not None:
            convergence[f'{previous_order} → {order}'] = {
                name: refinement(previous[name], cases[name]) for name in observations}
        previous, previous_order = cases, order
    events = reference_event_diagnostics(source, cases,
        {name: case.get('events', []) for name, case in cases.items()})
    metadata = {name: case.get('metadata', {}) for name, case in cases.items()}
    for name, case in cases.items():
        all_metrics[name], errors = score(source, case)
        case.update(errors)
        # Keep online full-covariance outputs, not frame caches or all flow nodes.
        save_case(output/('bayes_'+name+'.npz'), case)
    for name in observations:
        print('Original independent-reset comparator', name, flush=True)
        old_data = dict(selected, sparse_gyro=observations[name])
        value = old.predict_factorized(old_data, profile, config['assumed_sigma_deg_s'],
                'nonlinear', output_dt=config['output_dt_seconds'], max_step=config['max_step_seconds'])
        value['rpy_proxy_rad'] = value.pop('rpy_mean_rad')
        value['rpy_covariance_delta_rad2'] = value.pop('rpy_covariance_rad2')
        all_metrics['reset_'+name], errors = score(source, value)
        value.update(errors)
        save_case(output/('reset_'+name+'.npz'), value)
        cases['reset_'+name] = value
    write_json(output/'events.json', events)
    write_json(output/'quadrature_refinement.json', convergence)
    plot = module_from_path('synthetic_sequential_bayes_plotting', HERE/'plotting.py')
    plot.render(output, source, acquisitions, cases, diagnostics=dict(events=events))
    summary = dict(schema='grape/synthetic-sequential-bayes-spikes/v1', config=config,
        metrics=all_metrics, quadrature_refinement=convergence,
        quadrature_wall_seconds=timing, wall_seconds=time.monotonic()-begun,
        source_directory=str(source_dir), source_numeric_sha256=source_hashes,
        source_numeric_unchanged=source_hashes=={p.name: sha(p) for p in source_dir.glob('*.npz')},
        predictor_data_keys=list(selected), observation_data_keys=['white gyro', 'spike gyro'],
        spike_flags_or_truth_supplied_to_estimator=False,
        same_white_noise_realization=True, initial_gyro_used_once=True,
        first_gyro_initialization='N(y0,R): first-reading posterior under diffuse angular-rate prior; no additional y0 likelihood',
        bayes_target='N(w0;y0,R) times product over acquired k>=1 of N(yk; omega(tk;w0),R)',
        process_noise_added=False, bias_estimated_or_subtracted=False,
        robustification_or_outlier_rejection=False, monte_carlo_estimation=False,
        posterior_representation='same latent-initial-rate density; deterministic tensor Gauss-Hermite integration; no Gaussian posterior reset',
        engine_metadata=metadata,
        source_sha256={p.name: sha(p) for p in HERE.glob('*.py')},
        original_physics='same known J, torque interpolation, joint nonlinear Euler/quaternion RK4; body-to-world Hamilton wxyz',
        display='RPY center f(E[R]) and delta-method covariance from same full rotation moments; not exact Euler Gaussian or guaranteed 95% region')
    write_json(output/'summary.json', summary)
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['case','so3_proxy_rmse_deg','so3_proxy_max_deg','so3_proxy_final_deg',
                         'omega_x_rmse_deg_s','omega_y_rmse_deg_s','omega_z_rmse_deg_s'])
        for name, values in all_metrics.items():
            writer.writerow([name,values['so3_proxy_rmse_deg'],values['so3_proxy_max_deg'],
                             values['so3_proxy_final_deg'],*values['omega_rmse_deg_s']])
    report(output, config, acquisitions, all_metrics, convergence, events)
    files=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json']
    write_json(output/'manifest.json', dict(files=files,
        bytes_excluding_manifest=sum(p['bytes'] for p in files),
        old_results_modified=False, intermediate_cache_included=False))
    print('Completed', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-sequential-bayes-spikes-20261120'))
    run(parser.parse_args())
