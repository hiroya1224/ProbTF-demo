#!/usr/bin/env python3
"""Compare Q=0 dynamics on the same-time, threefold body angular velocity."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
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


shared = load('_omega3_run_shared', 'run.py')


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    if np.any(np.asarray(config['process_covariance_rad2_s2']) != 0):
        raise ValueError('This runner implements the requested deterministic Q=0 model')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generator = load('_omega3_input_generator', 'omega3_input.py')
    source, sensor, acquired, selected, physical, generation = generator.generate(
        args.input, config['body_omega_scale'], config['acquisition_rate_hz'])
    shared.write_json(output/'config.json', config)
    shared.write_json(output/'generation.json', generation)
    np.savez_compressed(output/'source_input.npz', **source)
    np.savez_compressed(output/'sensor_input.npz', **sensor)
    np.savez_compressed(output/'acquisitions.npz', **acquired)
    engine = load('_omega3_q0_engine', 'full_rate_engine.py')
    cache = output/'cache'
    cache.mkdir(exist_ok=True)
    models = ('nonlinear', 'linear_forced')
    convergence, timings = {}, {}
    previous = None
    for order in config['quadrature_orders']:
        before = time.monotonic()
        def predict(mode):
            print(f'Q=0, body omega x3: {mode}, GH {order}, {len(acquired["time"])} observations', flush=True)
            result = engine.predict_full_rate(selected, physical,
                sigma_deg_s=config['assumed_sigma_deg_s'], order=order,
                max_step=config['max_step_seconds'], mode=mode,
                refresh_seconds=config['refresh_seconds'],
                reanchor_displacement_sigma=config['reanchor_displacement_sigma'],
                dense_startup_seconds=config['dense_startup_seconds'])
            shared.save_case(cache/f'order{order}_{mode}.npz', result)
            shared.write_json(cache/f'order{order}_{mode}_refresh.json', result['refresh_records'])
            shared.write_json(cache/f'order{order}_{mode}_metadata.json', result['metadata'])
            metric, _ = shared.score(source, result)
            print(f'FINISHED {mode} GH {order}: attitude RMSE {metric["so3_proxy_rmse_deg"]:.6f} deg (501 display-grid times)', flush=True)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(predict, mode): mode for mode in models}
            cases = {futures[future]: future.result() for future in as_completed(futures)}
        cases = {mode: cases[mode] for mode in models}
        timings[str(order)] = time.monotonic()-before
        if previous is not None:
            convergence[f'{previous_order} -> {order}'] = {
                mode: shared.refinement(previous[mode], cases[mode]) for mode in models}
        previous, previous_order = cases, order
    metrics, engine_metadata = {}, {}
    for mode in models:
        metrics[mode], errors = shared.score(source, cases[mode])
        cases[mode].update(errors)
        engine_metadata[mode] = cases[mode]['metadata']
        shared.save_case(output/f'bayes_{mode}.npz', cases[mode])
    plotter = load('_omega3_saved_plotter', 'omega3_plotting.py')
    plotter.render(output, source, acquired, sensor, cases,
                   metadata=dict(config=config, input_generation='generation.json'))
    summary = dict(schema='grape/omega3-q0-comparison/v1', config=config,
        observation_count=len(acquired['time']), acquisition_rate_hz=config['acquisition_rate_hz'],
        native_reading_count=len(sensor['time']),
        selected_spiked_reading_count=int(np.count_nonzero(acquired['spike_mask'])),
        every_selected_sample_used=True, first_reading_initializes_once=True,
        process_covariance_rad2_s2=np.zeros((3,3)),
        estimator='full_rate_engine.predict_full_rate: deterministic Q=0 physical flow of the initial candidate law',
        update='initial N(W0;y0,R) times each of the 200 subsequent acquired likelihoods exactly once',
        model_difference='omega cross(J omega) only; shared quaternion kinematics',
        compared_models=list(models), metrics_common_501_times=metrics,
        primary_metrics='omega3_metrics.json: 201 identical saved posterior timestamps, no estimate interpolation',
        engine_metadata=engine_metadata, quadrature_refinement=convergence,
        inference_wall_seconds=timings, generation='generation.json',
        baseline_input_file_sha256=generation['base_file_sha256'],
        input_file_sha256={name: shared.sha(output/name) for name in
                          ('source_input.npz', 'sensor_input.npz', 'acquisitions.npz')},
        source_sha256={path.name: shared.sha(path) for pattern in
            ('omega3*', 'full_rate*', 'native_flow*', 'run.py', 'rate_comparison_plotting.py')
            for path in HERE.glob(pattern) if path.suffix in ('.py', '.cpp', '.json')},
        old_input_files_unchanged=generation['base_file_sha256']=={
            name: shared.sha(args.input/name) for name in generation['base_file_sha256']},
        monte_carlo_inference=False, covariance_rescaling=False,
        wall_seconds=time.monotonic()-started)
    shared.write_json(output/'summary.json', summary)
    shared.write_json(output/'quadrature_refinement.json', convergence)
    # Same retained moments; omit only reconstructible intermediate node weights.
    compact = load('_omega3_publication_compact', 'compact_rate_publication.py')
    compact.compact(output)
    write_report(output)
    verifier = load('_omega3_manifest', 'rate_comparison_verify.py')
    verifier.refresh_manifest(output)
    print('Completed '+str(output), flush=True)


def write_report(output):
    metrics = json.loads((output/'omega3_metrics.json').read_text())
    generation = json.loads((output/'generation.json').read_text())
    lines = ['# Q=0・body角速度3倍の結果', '',
        '[姿勢・角速度・±2σ](attitude_and_gyro.pdf) ／ '
        '[二モデルの誤差](reconstruction_errors.pdf) ／ '
        '[数式・再現手順](../OMEGA3_Q0.md)', '',
        '元の実飛行に合わせた滑らかなbody角速度を、時間軸・周波数を変えず各時刻で3倍にした。'
        '同じ初期姿勢から四元数を積分し直し、角加速度は3倍、ジャイロ項は9倍、'
        'トルクは $J\\alpha_3+\\omega_3\\times J\\omega_3$ として再生成した。', '',
        '20秒・10 Hz取得・201件の観測、生成白色σ=2 deg/s、観測Rのσ=5 deg/s。'
        'bias・スパイクの実現値・慣性・初期姿勢は従来と同じ。新たな乱数は引かない。'
        '推定のプロセスノイズはQ=0。全取得尤度を取得順に一度ずつ使い、前回の候補を捨てない。', '',
        'body角速度の標準偏差xyzは元の `'
        +', '.join(f'{value:.3f}' for value in generation['old_omega_std_deg_s'])
        +'` から `'+', '.join(f'{value:.3f}' for value in generation['new_omega_std_deg_s'])
        +'` deg/sになった。', '',
        '| モデル | 姿勢中心RMSE [deg] | 終端姿勢誤差 [deg] | ω RMSE xyz [deg/s] |',
        '|---|---:|---:|---|']
    for mode, label in (('nonlinear', '非線形Euler'), ('linear_forced', 'ジャイロ項なし')):
        item = metrics['results'][mode]
        lines.append(f'| {label} | {item["so3_proxy_rmse_deg"]:.6f} | '
            f'{item["so3_proxy_final_deg"]:.6f} | '
            +', '.join(f'{value:.6f}' for value in item['omega_rmse_deg_s'])+' |')
    lines.extend(['', f'共通する{metrics["evaluation_count"]}個の保存済み観測直後で評価した。'
        '推定列を補間しない。途中の真値は読取専用の評価参照であり、推定へ渡さない。'
        '[全評価値](omega3_metrics.json)。', '',
        'Q=0でも、スパイクを独立Gaussianとみなす観測モデルの不一致は残る。'
        'そのため、正しい非線形モデルがこの一つのノイズ実現値で必ず小さい姿勢誤差になるとは限らない。'
        '結果を見てR・ノイズ・seed・入力を選び直していない。', '',
        '上段RPY・中段ω平均と±2σ・下段ωの2σ半幅は同じ保存済み共同候補から描画する。'
        '観測前後401行、全回転共分散、角速度共分散と交差を保持する。'
        'RPY中心と帯は局所表示近似であり、保証付き95%領域ではない。', '',
        '[求積7→11の数値差](quadrature_refinement.json)、'
        '[有限proposal診断](proposal_refresh_diagnostics.json)、'
        '[入力生成・雑音同一性](generation.json)、'
        '[独立検証](verification.json)、[入力とsource hash](summary.json)を保存する。', '',
        '元の速度のQ=0とQ>0の結果はそのまま保持する。'
        '生bag・中間計算キャッシュは追加しない。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'frequent_publication')
    parser.add_argument('--config', type=Path, default=HERE/'omega3_config.json')
    parser.add_argument('--output', type=Path, default=HERE/'omega3_q0_publication')
    run(parser.parse_args())
