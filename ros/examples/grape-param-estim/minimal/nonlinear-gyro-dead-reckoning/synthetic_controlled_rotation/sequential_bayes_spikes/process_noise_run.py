#!/usr/bin/env python3
"""Model-side angular process noise, same recorded 10 Hz gyro and R."""
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
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


base = load('_process_noise_shared_inputs', 'rate_comparison_run.py')
shared = base.shared
MODELS = ('nonlinear', 'linear_forced')


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, sensor, acquired, selected, physical = base.saved_inputs(
        args.input, config['acquisition_rate_hz'])
    source_hashes = {name: shared.sha(args.input/name)
                     for name in ('source_input.npz', 'sensor_input.npz')}
    baseline_hashes = {mode: shared.sha(args.zero_process/f'bayes_{mode}.npz') for mode in MODELS}
    np.savez_compressed(output/'acquisitions.npz', **acquired)
    shared.write_json(output/'config.json', config)
    engine = load('_process_noise_filter', 'process_noise_engine.py')
    previous = None
    convergence, timings = {}, {}
    for order in config['quadrature_orders']:
        before = time.monotonic()
        def predict(mode):
            print(f'Process sigma=3 deg/s: {mode}, quadrature order {order}', flush=True)
            case = engine.predict_process_noise(selected, physical,
                measurement_sigma_deg_s=config['measurement_sigma_deg_s'],
                process_sigma_rad_s=config['process_sigma_rad_s'],
                order=order, mode=mode, max_step=config['max_step_seconds'])
            cache = output/'cache'
            cache.mkdir(exist_ok=True)
            shared.save_case(cache/f'order{order}_{mode}.npz', case)
            shared.write_json(cache/f'order{order}_{mode}_metadata.json', case.get('metadata', {}))
            metric, _ = shared.score(source, case)
            print(f'FINISHED {mode}, order {order}: attitude RMSE {metric["so3_proxy_rmse_deg"]:.6f} deg', flush=True)
            return case
        cases = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(predict, mode): mode for mode in MODELS}
            for future in as_completed(futures):
                cases[futures[future]] = future.result()
        cases = {mode: cases[mode] for mode in MODELS}
        timings[str(order)] = time.monotonic()-before
        if previous is not None:
            convergence[f'{previous_order} -> {order}'] = {
                mode: shared.refinement(previous[mode], cases[mode]) for mode in MODELS}
        previous, previous_order = cases, order
    metrics, metadata = {}, {}
    for mode in MODELS:
        metrics[mode], errors = shared.score(source, cases[mode])
        cases[mode].update(errors)
        metadata[mode] = cases[mode].get('metadata', {})
        shared.save_case(output/f'bayes_{mode}.npz', cases[mode])
    baseline = {mode: dict(np.load(args.zero_process/f'bayes_{mode}.npz')) for mode in MODELS}
    plotter = load('_process_noise_saved_figures', 'process_noise_plotting.py')
    plotter.render(output, source, acquired, sensor, cases, baseline,
        metadata=dict(config=config, engine_metadata=metadata))
    summary = dict(schema='grape/isotropic-angular-process-noise/v1', config=config,
        observation_count=len(acquired['time']),
        process_increment_count=len(acquired['time'])-1,
        process_covariance_rad2_s2=np.eye(3)*config['process_sigma_rad_s']**2,
        measurement_covariance_rad2_s2=np.eye(3)*np.deg2rad(config['measurement_sigma_deg_s'])**2,
        process_noise_scope='inference-model uncertainty only; original physical truth and sensor readings unchanged',
        gyro_updates='one analytic Gaussian-component Bayes update per acquired reading; initial reading used once',
        posterior='continuous Gaussian mixture in current angular velocity with conditional quaternion second moments and rotation covariance; numerical quadrature of transition integrals, not a Gaussian posterior replacement',
        metrics_common_501_times=metrics, quadrature_refinement=convergence,
        inference_wall_seconds=timings, engine_metadata=metadata,
        input_directory='frequent_publication', zero_process_directory='rate_10hz_publication',
        input_sha256=source_hashes, zero_process_sha256=baseline_hashes,
        input_files_unchanged=source_hashes=={name: shared.sha(args.input/name) for name in source_hashes},
        zero_process_files_unchanged=baseline_hashes=={mode: shared.sha(args.zero_process/f'bayes_{mode}.npz') for mode in MODELS},
        source_sha256={f.name: shared.sha(f) for f in HERE.glob('process_noise*.py')},
        monte_carlo=False, sensor_or_physical_data_regenerated=False,
        wall_seconds=time.monotonic()-started)
    shared.write_json(output/'summary.json', summary)
    shared.write_json(output/'quadrature_refinement.json', convergence)
    write_index(output)
    manifest(output)
    print('Completed', output, flush=True)


def manifest(output):
    files = [dict(path=f.name, bytes=f.stat().st_size, sha256=shared.sha(f))
             for f in sorted(output.iterdir()) if f.is_file() and f.name!='manifest.json']
    shared.write_json(output/'manifest.json', dict(files=files,
        bytes_excluding_manifest=sum(f['bytes'] for f in files),
        intermediate_cache_included=False))


def write_index(output):
    (output/'INDEX.md').write_text('''# 等方プロセスノイズσ=3 deg/sを加えた10 Hz比較

[姿勢・角速度・実際の±2σ](attitude_and_gyro.pdf) /
[Q=0との比較](process_noise_comparison.pdf) /
[数式・再現手順](../PROCESS_NOISE.md) /
[数値](summary.json)

推定モデルの各0.1秒の予測終端で、角速度へ独立Gaussianプロセスノイズを加える。
Qω=(3π/180)²I₃ rad²/s²。観測R=(5π/180)²I₃、gyroの原値・スパイク・既知入力・初期姿勢は同じ。
これは推定モデルの不確かさで、人工真値や測定列へ乱数を追加したものではない。

新しい角速度の不確かさは次の区間で姿勢へ伝播する。現在姿勢・角速度の依存を保持してBayes更新する。
原200 Hz測定の表示と、実際に使用する10 Hz・201観測を区別する。

角速度密度を単一Gaussianへ置換せず、Gaussian混合の遷移積分を決定論的求積で評価する。
条件付き姿勢モーメントを輸送・混合し、姿勢／角速度の全交差を保持する。
有限求積・トルク補間・RK4とRPYの局所表示近似を区別し、有限の診断値で観測更新を止めない。
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'frequent_publication')
    parser.add_argument('--zero-process', type=Path, default=HERE/'rate_10hz_publication')
    parser.add_argument('--config', type=Path, default=HERE/'process_noise_config.json')
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-process-noise-comparison-20261120'))
    run(parser.parse_args())
