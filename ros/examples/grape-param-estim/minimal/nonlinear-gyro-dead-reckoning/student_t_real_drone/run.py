#!/usr/bin/env python3
"""Frozen real-flight Student-t comparison, separate from synthetic results."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import json
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SHARED = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load('_real_student_run_input', HERE/'input.py')
shared = load('_real_student_shared_reports', SHARED/'run.py')


def label(value):
    return format(value, 'g').replace('.', 'p')


def preserved_hashes(include_sparse_real=False):
    roots = [SHARED/'student_t_publication',
             BASE/'mocap_refined_dynamics/publication/bag8_40_45_fit_50_55_predict']
    if include_sparse_real:
        roots.append(HERE/'publication/bag8_50_55s')
    return {str(path.relative_to(BASE)):shared.sha(path) for root in roots
            for path in sorted(root.rglob('*')) if path.is_file()
            and 'cache' not in path.relative_to(root).parts}


def write_index(output, config, records):
    if config.get('acquisition_mode') == 'full_native':
        return write_full_index(output, config, records)
    primary = f'rate_{label(config["primary_rate_hz"])}_hz/sigma_{label(config["primary_sigma_deg_s"])}_deg_s'
    lines = ['# 実ドローン：Student-t観測による二モデル比較', '',
        f'[主図：5 Hz・σ=2.5 deg/s]({primary}/attitude_and_gyro.pdf) ／ '
        f'[同条件の誤差]({primary}/reconstruction_errors.pdf) ／ [方法](../../README.md)', '',
        '一日目#8の同じ5秒区間、bag時刻50.000988619〜55.000988619秒。'
        '40〜45秒でrefineした慣性・アクチュエータモデルと過去余剰トルク平均を凍結した。'
        '新たなnoiseや人工スパイクは加えず、原gyroの取得時刻・値を使う。'
        'Student-tはν=4、Q=0。ジャイロ項あり／なしだけを比較する。', '',
        '初期姿勢は保存済みの一度だけのmocap整合。評価中のmocap・未取得gyroを'
        '推定やトルクの更新へ渡さない。モデルの再同定、時刻ずれ・bias補正はしない。', '',
        '| Hz | 周辺σ [deg/s] | gyro数 | 項あり姿勢RMSE [deg] | 項なし姿勢RMSE [deg] | 図 |',
        '|---:|---:|---:|---:|---:|---|']
    for record in sorted(records, key=lambda row:(row['sigma_deg_s'],row['rate_hz'])):
        result = record['metrics']['results']
        lines.append(f'| {record["rate_hz"]:g} | {record["sigma_deg_s"]:g} | '
            f'{record["observation_count"]} | {result["nonlinear"]["so3_proxy_rmse_deg"]:.6f} | '
            f'{result["linear_forced"]["so3_proxy_rmse_deg"]:.6f} | '
            f'[PDF]({record["directory"]}/attitude_and_gyro.pdf) |')
    lines.extend(['', 'すべて同じ501時刻で評価した。最後の取得後も固定終端まで同じ事後候補を'
        '物理伝播し、水平補間で埋めない。四元数・角速度の全共同モーメントと観測直前／直後を保存する。'
        'RPY中心と帯は回転モーメントに対する局所表示近似である。', '',
        '角速度の誤差欄は生gyroとの比較であり、角速度真値との誤差ではない。'
        '図のSG角速度は過去に保存したmocap微分の読取専用参照。'
        '固定モデルと過去余剰平均に残る実機との不一致をStudent-tが解消すると保証しない。', '',
        '[全結果CSV](comparison_table.csv) ／ [条件・source hash](summary.json) ／ '
        '[独立検証](verification.json) ／ [結果の解釈](RESULTS.md) ／ '
        '[合成結果と旧実測結果の保存確認](preserved_sources.json)。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def write_full_index(output, config, records):
    lines = ['# 実ドローン：原gyro全件によるStudent-t比較', '',
        '同じbag #8の50.000988619〜55.000988619秒に含まれる原gyroを、'
        '間引きも補間もせず全件使用する。最初の一件を初期角速度候補へ一度だけ使い、'
        'その後の各観測を取得順にStudent-t尤度として一度ずつ融合する。', '',
        '慣性・実指令／gimbal由来トルク・40〜45秒の余剰トルク平均は前の実測比較と同じ凍結値。'
        'Q=0、ν=4、周辺σ=2.5／5 deg/sを維持する。途中mocap、noise追加、biasや時計の補正、'
        '再同定は使わない。合成結果と0.5／2／5／10 Hz結果は変更しない。', '',
        '| 周辺σ [deg/s] | 原gyro数 | 項あり姿勢RMSE [deg] | 項なし姿勢RMSE [deg] | 図 |',
        '|---:|---:|---:|---:|---|']
    for record in sorted(records, key=lambda row:row['sigma_deg_s']):
        result = record['metrics']['results']
        lines.append(f'| {record["sigma_deg_s"]:g} | {record["observation_count"]} | '
            f'{result["nonlinear"]["so3_proxy_rmse_deg"]:.6f} | '
            f'{result["linear_forced"]["so3_proxy_rmse_deg"]:.6f} | '
            f'[姿勢・gyro]({record["directory"]}/attitude_and_gyro.pdf) ／ '
            f'[誤差]({record["directory"]}/reconstruction_errors.pdf) |')
    lines.extend(['', '観測直前／直後と、同じ501時刻の予測を保存する。最後の取得後も固定終端まで'
        '同じ事後候補を伝播する。角速度の全共分散と姿勢との交差を保持し、帯は同じ保存量の±2σ。'
        'RPYは回転モーメントからの局所表示近似。生gyroは雑音を含む測定参照であり真値ではない。', '',
        '[結果の解釈](RESULTS.md) ／ [全結果CSV](comparison_table.csv) ／ '
        '[取得時計・条件・hash](summary.json) ／ [独立検証](verification.json) ／ '
        '[旧結果の保存確認](preserved_sources.json)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    full_native = config.get('acquisition_mode') == 'full_native'
    if np.any(np.asarray(config['process_covariance_rad2_s2']) != 0):
        raise ValueError('This comparison keeps the declared Q=0 model')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preserved_before = preserved_hashes(include_sparse_real=full_native)
    shared.write_json(output/'config.json', config)
    dense = load('_real_student_dense', HERE/('full_dense_engine.py' if full_native else 'dense_engine.py'))
    plotting = load('_real_student_plots', HERE/'plotting.py')
    compact = load('_real_student_compact', SHARED/'compact_rate_publication.py')
    manifest = load('_real_student_manifest', SHARED/'rate_comparison_verify.py')
    rates = ([inputs.load_full_inputs()[-1]['native_median_rate_hz']] if full_native else config['rates_hz'])
    jobs = [(rate,sigma) for sigma in config['sigmas_deg_s'] for rate in rates]
    primary = (rates[0] if full_native else config['primary_rate_hz'],config['primary_sigma_deg_s'])
    jobs = [primary]+[job for job in jobs if job != primary]
    records = []
    frozen_sources = None
    for rate, sigma in jobs:
        source, acquired, selected, physical, provenance = (inputs.load_full_inputs() if full_native
            else inputs.load_inputs(rate))
        if frozen_sources is None:
            frozen_sources = provenance['source_file_sha256']
            np.savez_compressed(output/'reference_input.npz', **source)
            original = dict(np.load(inputs.REFINED/'refined_torque_profile.npz'))
            np.savez_compressed(output/'frozen_physical_input.npz',
                inertia=original['inertia_body'], torque_time=original['torque_profile_time'],
                torque_actuator_body=original['torque_actuator_body'],
                torque_applied_body=original['torque_applied_body'],
                residual_mean_body=original['residual_mean_body'])
        rate_directory = output/('rate_full_native' if full_native else f'rate_{label(rate)}_hz')
        directory = rate_directory/f'sigma_{label(sigma)}_deg_s'
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory/'cache'
        cache.mkdir(exist_ok=True)
        np.savez_compressed(directory/'acquisitions.npz', **acquired)
        if full_native:
            np.savez_compressed(rate_directory/'acquisitions.npz', **acquired)
        case_config = dict(config, rate_hz=rate, sigma_deg_s=sigma)
        shared.write_json(directory/'config.json', case_config)
        shared.write_json(directory/'input.json', provenance)
        previous = None
        convergence, timings = {}, {}
        for order in config['quadrature_orders']:
            started = time.monotonic()
            def predict(mode):
                print(f'Real bag8 Student-t: {rate:g} Hz / sigma {sigma:g}, {mode}, GH {order}', flush=True)
                predictor = dense.predict_full_dense if full_native else dense.predict_dense
                case = predictor(selected, physical, sigma_deg_s=sigma,
                    order=order, max_step=config['max_step_seconds'], mode=mode,
                    student_df=config['student_df'], output_dt=config['output_dt_seconds'],
                    refresh_seconds=config['refresh_seconds'],
                    reanchor_displacement_sigma=config['reanchor_displacement_sigma'],
                    dense_startup_seconds=config['dense_startup_seconds'])
                shared.save_case(cache/f'order{order}_{mode}.npz', case)
                shared.write_json(cache/f'order{order}_{mode}_refresh.json', case['refresh_records'])
                shared.write_json(cache/f'order{order}_{mode}_metadata.json', case['metadata'])
                shared.write_json(cache/f'order{order}_{mode}_dense.json', case['dense_interval_records'])
                print(f'FINISHED {rate:g} Hz / sigma {sigma:g} {mode} GH {order}', flush=True)
                return case
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {pool.submit(predict, mode):mode for mode in ('nonlinear','linear_forced')}
                cases = {pending[future]:future.result() for future in as_completed(pending)}
            cases = {mode:cases[mode] for mode in ('nonlinear','linear_forced')}
            timings[str(order)] = time.monotonic()-started
            if previous is not None:
                convergence[f'{previous_order} -> {order}'] = {
                    mode:shared.refinement(previous[mode],cases[mode]) for mode in cases}
            previous, previous_order = cases, order
        for mode, case in cases.items():
            shared.save_case(directory/f'bayes_{mode}.npz', case)
        plotting.render_case(directory, source, acquired, cases, metadata={'config':case_config})
        metrics = json.loads((directory/'metrics.json').read_text())
        case_summary = dict(schema='grape/real-student-case/v1', config=case_config,
            input=provenance, engine_metadata={mode:case['metadata'] for mode,case in cases.items()},
            quadrature_refinement=convergence, inference_wall_seconds=timings,
            dense_interval_records={mode:case['dense_interval_records'] for mode,case in cases.items()},
            model_difference='omega cross(J omega) only; same refined inertia, frozen torque and quaternion kinematics',
            state_model='Q=0, initial N(W0;y0,R) times all acquired Student-t likelihoods once',
            initial_reading_used_once=True, gaussian_observation_comparator_included=False,
            finite_numerical_differences_are_diagnostics_not_gates=True)
        shared.write_json(directory/'summary.json', case_summary)
        shared.write_json(directory/'quadrature_refinement.json', convergence)
        compact.compact(directory)
        manifest.refresh_manifest(directory)
        records.append(dict(rate_hz=rate,sigma_deg_s=sigma,observation_count=len(acquired['time']),
            directory=str(directory.relative_to(output)), metrics=metrics,
            prediction_tail_seconds=provenance['prediction_tail_seconds']))
        print('PUBLISHED '+str(directory), flush=True)
    preserved_after = preserved_hashes(include_sparse_real=full_native)
    preservation = dict(before_sha256=preserved_before, after_sha256=preserved_after,
        every_preserved_file_unchanged=preserved_before==preserved_after,
        frozen_synthetic_commit='5bdd9072',
        note='Synthetic full-spike Student-t, former real refined and (for full_native) sparse real publications were read only; caches excluded')
    shared.write_json(output/'preserved_sources.json', preservation)
    summary = dict(schema='grape/real-flight-student-dynamics-comparison/v1', config=config,
        original_source_sha256=frozen_sources, records=records,
        preservation=preservation['every_preserved_file_unchanged'],
        initial_mocap_only_for_inference=True, refit_performed=False,
        additional_noise_spikes_bias_or_time_shift_correction=False,
        process_noise=False, variational_gaussian_state_replacement=False, monte_carlo=False,
        source_sha256={str(path.relative_to(BASE)):shared.sha(path)
            for path in sorted(HERE.glob('*.py'))},
        shared_source_sha256={str(path.relative_to(BASE)):shared.sha(path)
            for name in ('full_rate_engine.py','full_rate_flow.py','full_rate_flow.cpp',
                         'student_t_observation.py','native_flow.py','native_flow.cpp')
            for path in [SHARED/name]},
        wall_seconds=time.monotonic()-begun)
    shared.write_json(output/'summary.json', summary)
    with (output/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['rate_hz','sigma_deg_s','model','observation_count','so3_rmse_deg','so3_final_deg'])
        for record in records:
            for mode,item in record['metrics']['results'].items():
                writer.writerow([record['rate_hz'],record['sigma_deg_s'],mode,
                    record['observation_count'],item['so3_proxy_rmse_deg'],item['so3_proxy_final_deg']])
    write_index(output, config, records)
    manifest.refresh_manifest(output)
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path, default=HERE/'publication/bag8_50_55s')
    run(parser.parse_args())
