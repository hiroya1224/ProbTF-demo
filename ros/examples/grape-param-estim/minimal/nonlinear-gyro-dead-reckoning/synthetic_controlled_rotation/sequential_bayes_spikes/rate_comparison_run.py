#!/usr/bin/env python3
"""Compare both models at 10 Hz, reusing the same saved 200 Hz sensor input."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
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


shared = load('_all_native_gyro_shared', 'run.py')


def saved_inputs(directory, acquisition_rate_hz=10.):
    """Select original readings at requested times, without regeneration."""
    source = dict(np.load(directory/'source_input.npz'))
    sensor = dict(np.load(directory/'sensor_input.npz'))
    desired = np.arange(source['time'][0], source['time'][1]+0.5/acquisition_rate_hz,
                        1./acquisition_rate_hz)
    requested = np.rint(desired*1e9).astype(np.int64)
    index = np.searchsorted(sensor['stamp_ns'], requested)
    acquired = dict(time=sensor['time'][index].copy(), stamp_ns=sensor['stamp_ns'][index].copy(),
        record_ns=sensor['record_ns'][index].copy(), source_index=sensor['source_index'][index].copy(),
        native_array_index=index, requested_stamp_ns=requested,
        gyro_spike=sensor['gyro_spike'][index].copy(), spike_mask=sensor['spike_mask'][index].copy(),
        spike_vector_rad_s=sensor['spike_vector_rad_s'][index].copy())
    selected = dict(time=source['time'].copy(),
        initial_quaternion_wxyz=source['initial_quaternion_wxyz'].copy(),
        sparse_gyro_time=acquired['time'].copy(), sparse_gyro=acquired['gyro_spike'].copy())
    physical = dict(inertia=source['inertia_body_kg_m2'].copy(),
        torque=shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
    return source, sensor, acquired, selected, physical


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, sensor, acquired, selected, physical = saved_inputs(args.input, config['acquisition_rate_hz'])
    inputs_hash = {name: shared.sha(args.input/name) for name in (
        'source_input.npz', 'sensor_input.npz', 'bayes_nonlinear.npz', 'bayes_linear_forced.npz')}
    shared.write_json(output/'config.json', config)
    np.savez_compressed(output/'acquisitions.npz', **acquired)
    engine = load('_all_native_gyro_engine', 'full_rate_engine.py')
    prior_cases = None
    convergence, timings = {}, {}
    models = ('nonlinear', 'linear_forced')
    for order in config['quadrature_orders']:
        cases = {}
        before = time.monotonic()
        def predict(mode):
            print(f'{config["acquisition_rate_hz"]:g} Hz gyro: {mode}, GH {order}, {len(acquired["time"])} observations', flush=True)
            result = engine.predict_full_rate(selected, physical,
                sigma_deg_s=config['assumed_sigma_deg_s'], order=order,
                max_step=config['max_step_seconds'], mode=mode,
                refresh_seconds=config['refresh_seconds'],
                reanchor_displacement_sigma=config['reanchor_displacement_sigma'],
                dense_startup_seconds=config['dense_startup_seconds'])
            cache = output/'cache'
            cache.mkdir(exist_ok=True)
            shared.save_case(cache/f'order{order}_{mode}.npz', result)
            shared.write_json(cache/f'order{order}_{mode}_refresh.json', result['refresh_records'])
            shared.write_json(cache/f'order{order}_{mode}_metadata.json', result['metadata'])
            metric, _ = shared.score(source, result)
            print(f'FINISHED {mode} GH {order}: attitude RMSE {metric["so3_proxy_rmse_deg"]:.6f} deg', flush=True)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(predict, mode):mode for mode in models}
            for future in as_completed(futures):
                cases[futures[future]] = future.result()
        cases = {mode: cases[mode] for mode in models}
        timings[str(order)] = time.monotonic()-before
        if prior_cases is not None:
            convergence[f'{prior_order} -> {order}'] = {
                mode: shared.refinement(prior_cases[mode], cases[mode]) for mode in models}
        prior_cases, prior_order = cases, order
    metrics, engine_metadata = {}, {}
    for mode in models:
        metrics[mode], errors = shared.score(source, cases[mode])
        cases[mode].update(errors)
        engine_metadata[mode] = cases[mode].get('metadata', {})
        shared.save_case(output/f'bayes_{mode}.npz', cases[mode])
    one_hz = {mode: dict(np.load(args.input/f'bayes_{mode}.npz')) for mode in models}
    plotter = load('_selected_rate_plotter', 'rate_comparison_plotting.py')
    plotter.render(output, source, acquired, sensor, cases, one_hz,
        metadata=dict(observations_count=len(acquired['time']), rate_hz=config['acquisition_rate_hz'], config=config))
    summary = dict(schema='grape/selected-rate-gyro-dynamics-comparison/v1',
        config=config, observation_count=len(acquired['time']),
        acquisition_rate_hz=config['acquisition_rate_hz'], every_selected_sample_used=True,
        sensor_native_rate_hz=200., native_reading_count=len(sensor['time']),
        native_spiked_observation_count=int(np.count_nonzero(acquired['spike_mask'])),
        first_reading_used_as_initial_law_once=True,
        update='causal prefix of the original initial Gaussian times each acquired likelihood exactly once',
        numerical_proposal_refresh_is_not_an_observation_update_schedule=True,
        compared_models=list(models), model_difference='omega cross(J omega) only; shared quaternion kinematics',
        metrics_common_501_times=metrics, metrics_scope='Same 501 evaluation times as existing 1 Hz report; figure report additionally uses exact shared stored timestamps',
        quadrature_refinement=convergence, inference_wall_seconds=timings,
        engine_metadata=engine_metadata,
        input_directory_relative_to_runner=os.path.relpath(args.input.resolve(), HERE),
        input_file_sha256=inputs_hash,
        source_sha256={f.name: shared.sha(f) for pattern in ('full_rate*', 'rate_comparison*')
                       for f in HERE.glob(pattern) if f.suffix in ('.py', '.cpp')},
        original_input_files_unchanged=inputs_hash == {name: shared.sha(args.input/name) for name in inputs_hash},
        generated_noise_or_motion_recomputed=False, monte_carlo_inference=False,
        wall_seconds=time.monotonic()-started)
    shared.write_json(output/'summary.json', summary)
    shared.write_json(output/'quadrature_refinement.json', convergence)
    write_report(output, summary)
    files = [dict(path=f.name, bytes=f.stat().st_size, sha256=shared.sha(f))
             for f in sorted(output.iterdir()) if f.is_file() and f.name!='manifest.json']
    shared.write_json(output/'manifest.json', dict(files=files,
        bytes_excluding_manifest=sum(f['bytes'] for f in files),
        raw_bag_or_intermediate_cache_included=False))
    print('Completed', output, flush=True)


def write_report(output, summary):
    lines = ['# 10 Hz gyroを使う比較', '',
        '[姿勢・角速度と±2σ](attitude_and_gyro.pdf) / '
        '[1 Hzとの比較](rate_comparison_errors.pdf) / [全数値](summary.json)', '',
        '実飛行の規模に合わせた既存の20秒・200 Hz入力を再生成せずに読み、'
        '0〜20秒から0.1秒ごとの201件を原値のまま取得した。比較する二モデルの測定・原時計・慣性・既知トルク・'
        '初期姿勢・観測σ=5 deg/sは同一。生成白色σ=2 deg/sと頻発スパイクも元の実現値のまま。', '',
        r'$\pi_{m,k}(w)\propto\mathcal N(w;y_0,R)'
        r'\prod_{j=1}^{k}\mathcal N(y_j;\omega_m(t_j;w),R)$。', '',
        '全観測で取得順に尤度を一度ずつ掛ける。初回は初期法則にだけ使い、再融合しない。'
        '途中の真値・スパイクフラグは推定入力にしない。', '',
        '| モデル | 姿勢中心RMSE [deg] | 終端誤差 [deg] | ω RMSE xyz [deg/s] |',
        '|---|---:|---:|---|']
    for mode, label in (('nonlinear', '非線形Euler'), ('linear_forced', 'ジャイロ項なし')):
        metric = summary['metrics_common_501_times'][mode]
        lines.append(f'| {label} | {metric["so3_proxy_rmse_deg"]:.6f} | '
            f'{metric["so3_proxy_final_deg"]:.6f} | '
            + ', '.join(f'{value:.6f}' for value in metric['omega_rmse_deg_s'])+' |')
    lines.extend(['', '上表は旧1 Hzレポートと同じ501個の評価時刻での比較。'
        '別の周波数比較図の統計は、保存された共通時刻の観測直後を使う。'
        'RPY中心と帯は同じ回転モーメントからの局所表示近似であり、保証付き95%領域ではない。', '',
        '## 数値積分と観測更新の区別', '',
        '積分点のGaussian proposalは時間区間ごとに再配置する。'
        'これは求積点の配置だけを変える処理で、更新済みの分布をGaussianへ置換しない。'
        '各時刻の重みには元の初期事前と全取得尤度を使い、proposal密度を除く。'
        '選択した10 Hzの各測定で直前・直後の共同モーメントを計算する。'
        '必要な再配置は計算を追加するだけで、有限な診断値を観測棄却や停止条件にしない。', '',
        '過去の全尤度を新しい求積点で評価する際は、同じC++ RK4で履歴を伝播しながら'
        '対数尤度を集約する。巨大な全履歴×全求積点の配列は保持しない。'
        '過去に出力した列へ未来の観測を戻さない。', '',
        'スパイクを独立Gaussian誤差とみなす仮定は変えていない。'
        '実際のパルスには時間相関があるので、観測数に応じて狭まる共分散を'
        'そのまま現実の誤差を保証する幅とは解釈できない。ロバスト化・外れ値除去・'
        'BFS・過程雑音・共分散floorは追加していない。', '',
        '元の1 Hz結果は保持する。[実装と再現](../RATE_10HZ.md)。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'frequent_publication')
    parser.add_argument('--config', type=Path, default=HERE/'rate_10hz_config.json')
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-10hz-comparison-20261120'))
    run(parser.parse_args())
