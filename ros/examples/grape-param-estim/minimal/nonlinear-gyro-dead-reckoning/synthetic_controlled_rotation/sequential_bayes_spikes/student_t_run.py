#!/usr/bin/env python3
"""Student-t only: nonlinear Euler versus omitted gyroscopic torque term."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = load('_student_run_shared', 'run.py')


def prepare_inputs(directory, config):
    """Change only saved spike amplitudes; preserve motion/noise/clocks."""
    directory = Path(directory)
    selection = load('_student_input_selection', 'rate_comparison_run.py')
    source, sensor, acquired, selected, physical = selection.saved_inputs(
        directory, config['acquisition_rate_hz'])
    baseline_config = json.loads((directory/'config.json').read_text())
    baseline_scale = float(baseline_config.get('spike_scale', 1.))
    target_scale = float(config['spike_scale'])
    if (not np.isfinite(baseline_scale) or baseline_scale <= 0.
            or not np.isfinite(target_scale) or target_scale <= 0.):
        raise ValueError('Finite positive source and target spike-amplitude scales required')
    ratio = target_scale/baseline_scale
    amplitude_fields = ('spike_vector_rad_s', 'event_vector_body_rad_s', 'event_magnitude_deg_s')
    if ratio != 1.:
        for name in amplitude_fields:
            sensor[name] = ratio*sensor[name]
        sensor['gyro_spike'] = sensor['gyro_white']+sensor['spike_vector_rad_s']
        indices = acquired['native_array_index']
        acquired['gyro_spike'] = sensor['gyro_spike'][indices].copy()
        acquired['spike_vector_rad_s'] = sensor['spike_vector_rad_s'][indices].copy()
        selected['sparse_gyro'] = acquired['gyro_spike'].copy()
    transformation = dict(baseline_spike_scale=baseline_scale, target_spike_scale=target_scale,
        applied_amplitude_multiplier=ratio, scaled_sensor_fields=list(amplitude_fields),
        measured_gyro_rule='unchanged gyro_white + scaled saved spike_vector_rad_s',
        event_times_rate_widths_directions_masks_and_counts_changed=False,
        motion_torque_white_noise_bias_or_clocks_changed=False, new_random_draws=False)
    return source, sensor, acquired, selected, physical, transformation


def write_report(output):
    metrics = json.loads((output/'student_t_metrics.json').read_text())
    config = json.loads((output/'config.json').read_text())
    lines = ['# Student-t観測：ジャイロ項あり／なし', '',
        '[姿勢・角速度と±2σ](attitude_and_gyro.pdf) ／ '
        '[復元誤差](reconstruction_errors.pdf) ／ '
        '[定式化と再現手順](../STUDENT_T.md)', '',
        f'両方の観測尤度を3変量Student-t、自由度ν={config["student_df"]:g}とした。'
        'Student-tを使わない方式は今回の図に含めない。'
        f'Q=0・body角速度3倍・スパイク振幅{config["spike_scale"]:g}倍・20秒・10 Hzとし、'
        '保存済みのスパイクを元の振幅（単独パルス20〜60 deg/s）へ戻した。'
        '発生時刻・頻度・幅・方向・白色noise・真値運動・トルクは保持し、'
        '途中の真値・スパイクフラグ・生成biasを推定へ渡さない。', '',
        '観測の周辺共分散は $R=(5\\pi/180)^2I_3$ のまま。'
        'Student-tの尺度行列は $\\Sigma=(\\nu-2)R/\\nu$ とし、ν=4ではR/2。'
        '初期候補は従来と同じ $\\mathcal N(W_0;y_0,R)$ で、初回gyroを一度だけ初期化に使う。'
        'その後の200観測は、すべてStudent-t尤度で取得順に一度ずつ更新する。', '',
        'Gamma精度を解析的に周辺化したStudent-t密度を直接評価し、'
        '既存の決定論的求積で全候補と全交差を伝播する。'
        'PichéらのStudent-t観測モデルを使うが、同論文の変分Gaussian状態近似へ置換する実装ではない。'
        'Gaussianは初期候補と数値積分用proposalであり、各観測のStudent-t尤度をGaussianに戻していない。', '',
        '| Student-t方式 | 姿勢中心RMSE [deg] | 終端姿勢誤差 [deg] | ω RMSE xyz [deg/s] |',
        '|---|---:|---:|---|']
    for mode, label in (('nonlinear', '非線形Euler'), ('linear_forced', 'ジャイロ項なし')):
        item = metrics['results'][mode]
        lines.append(f'| {label} | {item["so3_proxy_rmse_deg"]:.6f} | '
            f'{item["so3_proxy_final_deg"]:.6f} | '
            +', '.join(f'{value:.6f}' for value in item['omega_rmse_deg_s'])+' |')
    lines.extend(['', f'共通する{metrics["evaluation_count"]}個の保存済み観測直後で評価。'
        '推定列の補間や平滑化はしない。RPYの中心と帯は同じ回転モーメントからの局所表示近似。'
        '観測前後401行、四元数二次モーメント、全回転共分散、角速度共分散と交差を保存する。', '',
        '[全評価値](student_t_metrics.json)、[求積の有限差](quadrature_refinement.json)、'
        '[独立検証](verification.json)、[実行条件・入力とsource hash](summary.json)。'
        '有限な求積差やproposal未収束を観測棄却・更新停止へ使わない。', '',
        '参照：Piché, Särkkä, Hartikainen (2012), '
        '[Recursive Outlier-Robust Filtering and Smoothing for Nonlinear Systems Using the Multivariate Student-t Distribution]'
        '(https://users.aalto.fi/~ssarkka/pub/studentFiltering_MLSP.pdf), '
        '§2.1–2.3。ν=4は実行前に同論文の推奨値に固定した。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    if np.any(np.asarray(config['process_covariance_rad2_s2']) != 0):
        raise ValueError('This comparison keeps Q=0')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, sensor, acquired, selected, physical, transformation = prepare_inputs(args.input, config)
    input_names = ('source_input.npz', 'sensor_input.npz', 'acquisitions.npz')
    input_hashes = {name: shared.sha(args.input/name) for name in input_names}
    shutil.copyfile(args.input/'source_input.npz', output/'source_input.npz')
    np.savez_compressed(output/'sensor_input.npz', **sensor)
    np.savez_compressed(output/'acquisitions.npz', **acquired)
    shared.write_json(output/'config.json', config)
    if not np.array_equal(acquired['gyro_spike'], np.load(output/'acquisitions.npz')['gyro_spike']):
        raise ValueError('Saved acquisitions must match the unchanged input clock')
    engine = load('_student_full_rate_engine', 'full_rate_engine.py')
    cache = output/'cache'
    cache.mkdir(exist_ok=True)
    models = ('nonlinear', 'linear_forced')
    previous = None
    convergence, timings = {}, {}
    for order in config['quadrature_orders']:
        started = time.monotonic()
        def predict(mode):
            print(f'Student-t df={config["student_df"]:g}: {mode}, GH {order}, {len(acquired["time"])} readings', flush=True)
            case = engine.predict_full_rate(selected, physical,
                sigma_deg_s=config['assumed_sigma_deg_s'], student_df=config['student_df'],
                order=order, max_step=config['max_step_seconds'], mode=mode,
                refresh_seconds=config['refresh_seconds'],
                reanchor_displacement_sigma=config['reanchor_displacement_sigma'],
                dense_startup_seconds=config['dense_startup_seconds'])
            shared.save_case(cache/f'order{order}_{mode}.npz', case)
            shared.write_json(cache/f'order{order}_{mode}_refresh.json', case['refresh_records'])
            shared.write_json(cache/f'order{order}_{mode}_metadata.json', case['metadata'])
            print(f'FINISHED Student-t {mode} GH {order}', flush=True)
            return case
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(predict, mode): mode for mode in models}
            cases = {futures[future]: future.result() for future in as_completed(futures)}
        cases = {mode: cases[mode] for mode in models}
        timings[str(order)] = time.monotonic()-started
        if previous is not None:
            convergence[f'{previous_order} -> {order}'] = {
                mode: shared.refinement(previous[mode], cases[mode]) for mode in models}
        previous, previous_order = cases, order
    engine_metadata = {}
    for mode in models:
        engine_metadata[mode] = cases[mode]['metadata']
        shared.save_case(output/f'bayes_{mode}.npz', cases[mode])
    plotting = load('_student_only_plots', 'student_t_plotting.py')
    plotting.render(output, source, acquired, sensor, cases, metadata={'config': config})
    summary = dict(schema='grape/student-t-only-rotation-comparison/v1', config=config,
        observation_distribution='multivariate Student-t for every acquired j>=1 factor',
        initial_law='Same initial Gaussian N(W0;y0,R) as the previous experiment; y0 initializes once',
        student_scale_rad2_s2=cases['nonlinear']['observation_student_scale_rad2_s2'],
        observation_covariance_rad2_s2=cases['nonlinear']['measurement_covariance_rad2_s2'],
        marginal_observation_covariance_preserved=True,
        input_directory_relative_to_runner=os.path.relpath(args.input.resolve(), HERE),
        input_file_sha256=input_hashes,
        saved_input_file_sha256={name: shared.sha(output/name) for name in input_names},
        copied_input_files_bitwise_identical=all(shared.sha(output/name)==input_hashes[name] for name in input_names),
        source_input_bitwise_identical=shared.sha(output/'source_input.npz')==input_hashes['source_input.npz'],
        input_transformation=transformation,
        input_files_unchanged_after_inference=all(shared.sha(args.input/name)==input_hashes[name] for name in input_names),
        native_reading_count=len(sensor['time']), observation_count=len(acquired['time']),
        selected_spiked_reading_count=int(np.count_nonzero(acquired['spike_mask'])),
        every_selected_sample_used=True, process_covariance_rad2_s2=np.zeros((3,3)),
        model_difference='omega cross(J omega) only; identical Hamilton quaternion kinematics',
        compared_models=list(models), gaussian_observation_comparator_included=False,
        inference='Direct analytically marginalized Student-t likelihood; deterministic importance quadrature over initial W0',
        variational_gaussian_state_replacement=False, monte_carlo_inference=False,
        source_sha256={path.name: shared.sha(path) for pattern in
            ('student_t*', 'full_rate*', 'native_flow*', 'run.py', 'omega3_plotting.py')
            for path in HERE.glob(pattern) if path.suffix in ('.py', '.cpp', '.json')},
        engine_metadata=engine_metadata, quadrature_refinement=convergence,
        inference_wall_seconds=timings, wall_seconds=time.monotonic()-begun)
    shared.write_json(output/'summary.json', summary)
    shared.write_json(output/'quadrature_refinement.json', convergence)
    compact = load('_student_publication_storage', 'compact_rate_publication.py')
    compact.compact(output)
    write_report(output)
    manifest = load('_student_manifest', 'rate_comparison_verify.py')
    manifest.refresh_manifest(output)
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'omega3_q0_publication')
    parser.add_argument('--config', type=Path, default=HERE/'student_t_config.json')
    parser.add_argument('--output', type=Path, default=HERE/'student_t_publication')
    run(parser.parse_args())
