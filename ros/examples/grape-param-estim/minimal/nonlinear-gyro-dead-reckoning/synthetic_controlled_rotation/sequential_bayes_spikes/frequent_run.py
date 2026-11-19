#!/usr/bin/env python3
"""1 Hz causal Bayes updates of a 200 Hz gyro with frequent Poisson spikes."""
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = load_module('_frequent_bayes_shared_reporting', HERE/'run.py')
generator = load_module('_frequent_bayes_input', HERE/'frequent_spikes_input.py')
write_json = shared.write_json
sha = shared.sha


def record_report(output, config, generation, acquisitions, metrics, convergence, events):
    active = float(config['spike_active_until_seconds'])
    times = acquisitions['time']
    contaminated = acquisitions['spike_mask']
    captured = times[contaminated]
    native_count = int(generation['realized_spike_count'])
    lines = ['# 1 Hz更新・高頻度スパイク・白色ノイズσ=2 deg/s', '',
        '[比較動画](frequent_spikes_comparison.mp4) / [姿勢・角速度](attitude_and_gyro.pdf) / '
        '[生gyroの全体と拡大](native_gyro_full_and_zoom.pdf) / '
        '[ノイズ成分](sensor_noise_components.pdf) / [観測更新の診断](bayes_diagnostics.pdf)', '',
        f'実験は{config["duration_seconds"]:g}秒。生gyroは{config["sensor_rate_hz"]:g} Hz、'
        f'Bayes更新は{config["rate_hz"]:g} Hz。生成する白色ノイズの標準偏差は'
        f'**{config["white_sigma_deg_s"]:g} deg/s**。推定のGaussian観測モデルは従来の'
        f'σ={config["assumed_sigma_deg_s"]:g} deg/sで、生成側と区別している。', '',
        f'0〜{active:g}秒は強度{config["spike_rate_hz"]:g}回/秒のPoisson過程で'
        f'{native_count}回のスパイクが発生した。開始時刻は更新周期とは独立に生成し、'
        f'幅は{config["spike_width_range_seconds"][0]:g}〜{config["spike_width_range_seconds"][1]:g}秒、'
        f'単独パルスのベクトル長は{config["spike_magnitude_range_deg_s"][0]:g}〜'
        f'{config["spike_magnitude_range_deg_s"][1]:g} deg/s。重なるパルスはベクトル和となる。', '',
        f'{active:g}〜{config["duration_seconds"]:g}秒はスパイクを止めた回復観察区間で、'
        '白色ノイズは同じσのまま残す。これは事前に定めた入力条件であり、'
        '推定結果やseedを見て正常区間を切り出したものではない。', '',
        f'生データ上のスパイク{native_count}回のうち、1 Hzの取得と重なったのは'
        f'{int(contaminated.sum())}件：'+(', '.join(f'{v:g}' for v in captured) or 'なし')+'秒。'
        '取得しなかった200 Hz測定値を追加の独立観測として融合していない。', '',
        '| 条件 | 姿勢中心RMSE [deg] | 終端誤差 [deg] | ω RMSE xyz [deg/s] |',
        '|---|---:|---:|---|']
    for name, label in (('white','白色ノイズのみ'),('spike','白色＋頻発スパイク')):
        item = metrics[name]
        lines.append(f'| {label} | {item["so3_proxy_rmse_deg"]:.5f} | '
            f'{item["so3_proxy_final_deg"]:.5f} | '
            +', '.join(f'{v:.5f}' for v in item['omega_rmse_deg_s'])+' |')
    lines.extend(['', f'{config["duration_seconds"]:g}秒終端の姿勢中心誤差は、白色のみで{metrics["white"]["so3_proxy_final_deg"]:.3f}度、'
        f'スパイクありで{metrics["spike"]["so3_proxy_final_deg"]:.3f}度。'
        '正常観測で誤差が下がる場面はあるが、この試行では回復観察区間の最後にも誤差が残った。'
        '回復するように観測尤度や共分散を調整した結果ではない。', '',
        '## スパイク取得と、その後の正常な更新', '',
        '通常のGaussian尤度をそのまま使い、スパイクを知らせたり外れ値として除去したりしていない。'
        '同じ初期角速度候補に全観測の尤度を取得順に掛け、姿勢・角速度の依存と全交差共分散を保持する。', '',
        '| 時刻 [s] | スパイク取得 | 加算ベクトル xyz [deg/s] | 姿勢誤差 直前→直後 [deg] |',
        '|---:|---|---|---|'])
    for event in events['spike']:
        if event['initialization']:
            continue
        k = event['acquisition_index']
        vector = np.rad2deg(acquisitions['spike_vector_rad_s'][k])
        before = event['prior_so3_proxy_error_deg_reference_only']
        after = event['posterior_so3_proxy_error_deg_reference_only']
        lines.append(f'| {event["time"]:g} | '+('あり' if contaminated[k] else 'なし')+' | '
            +', '.join(f'{v:.3f}' for v in vector)+f' | {before:.4f} → {after:.4f} |')
    lines.extend(['', 'この表の真値比較は推定完了後の評価専用で、フィルタへ戻していない。'
        '過去のオンライン出力を後の観測で書き換える平滑化も行わない。', '',
        '## 数値積分とモデルの範囲', '',
        '観測数を増やすと固定した初期格子では事後の細さを解像しにくいため、'
        '同じ「初期Gaussian × 全観測尤度」の積分を適応Gauss–Hermite重要度求積で評価する。'
        '提案Gaussianは積分用の座標と配分であり、Bayes事後をGaussianへ置換する状態更新ではない。'
        'モンテカルロは使わない。次数間の有限差は診断として保存し、観測の棄却条件には使わない。', '',
        '| 次数比較 | 条件 | 最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |',
        '|---|---|---:|---:|'])
    for label, cases in convergence.items():
        for name, item in cases.items():
            lines.append(f'| {label} | {name} | {item["max_omega_mean_component_difference_deg_s"]:.6g} | '
                f'{item["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '非線形Euler式と四元数を同時RK4で積分する。既知の慣性Jと入力トルクは生成側と一致しており、'
        '未知の機体モデル誤差を含む試験とは区別する。既存ProbTF coreと旧結果は変更していない。', '',
        '表示する姿勢中心はf(E[R])、帯は同じ全回転共分散からの局所delta-method ±2σ。'
        '厳密なGaussian姿勢分布や保証付き95%領域とは扱わない。', '',
        '[全イベント](spike_events.json) / [観測前後](events.json) / '
        '[入力の生成定義](generation.json) / [数値・設定](summary.json) / '
        '[再現手順](../FREQUENT_SPIKES.md)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old_dir = HERE.parent/'publication/three_frequency_0p5hz_sigma5'
    original = {f.name: sha(f) for f in old_dir.glob('*.npz')}
    source, selected, acquisitions, sensor, pulse_events, generation = generator.generate(config)
    np.savez_compressed(output/'source_input.npz', **source)
    np.savez_compressed(output/'sensor_input.npz', **sensor)
    np.savez_compressed(output/'acquisitions.npz', **acquisitions)
    write_json(output/'config.json', config)
    write_json(output/'spike_events.json', pulse_events)
    write_json(output/'generation.json', generation)
    profile = dict(inertia=source['inertia_body_kg_m2'],
        torque=shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
    engine = load_module('_frequent_adaptive_bayes_engine', HERE/'adaptive_engine.py')
    observations = {name: acquisitions['gyro_'+name] for name in ('white','spike')}
    previous = None
    convergence, durations = {}, {}
    for order in config['quadrature_orders']:
        print('Frequent-spike adaptive Bayes order', order, flush=True)
        before = time.monotonic()
        cases = engine.predict_adaptive(selected, profile,
            sigma_deg_s=config['assumed_sigma_deg_s'], order=order,
            output_dt=config['output_dt_seconds'], max_step=config['max_step_seconds'],
            observation_sets=observations)
        durations[str(order)] = time.monotonic()-before
        if previous is not None:
            convergence[f'{previous_order} → {order}'] = {
                name: shared.refinement(previous[name], cases[name]) for name in observations}
        previous, previous_order = cases, order
    events = shared.reference_event_diagnostics(source, cases,
        {name: case.get('events',[]) for name, case in cases.items()})
    metadata = {name: case.get('metadata',{}) for name, case in cases.items()}
    metrics = {}
    for name, case in cases.items():
        metrics[name], errors = shared.score(source, case)
        case.update(errors)
        shared.save_case(output/('bayes_'+name+'.npz'), case)
    write_json(output/'events.json', events)
    write_json(output/'quadrature_refinement.json', convergence)
    plot = load_module('_frequent_bayes_plotting', HERE/'frequent_plotting.py')
    plot.render(output, source, acquisitions, sensor, cases,
        metadata=dict(config=config,generation=generation,events=events))
    standardized = source['gyro_realized_noise_rad_s']
    summary = dict(schema='grape/frequent-poisson-spikes-1hz-bayes/v1',config=config,
        generation=generation,metrics=metrics,quadrature_refinement=convergence,
        inference_wall_seconds=durations,wall_seconds=time.monotonic()-started,
        actual_white_noise_std_deg_s=np.rad2deg(np.std(standardized,axis=0,ddof=1)),
        estimator_noise_covariance_rad2_s2=np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2,
        posterior_target='initial N(w0;y0,R) times all acquired k>=1 Gaussian gyro likelihoods; no process noise, reset, or posterior Gaussian replacement',
        observation_input_keys=list(selected),observation_sets=['white gyro values','spiked gyro values'],
        later_truth_or_spike_flags_supplied_to_estimator=False,
        posterior_moments_and_visualization_use_same_law=True,
        monte_carlo_estimation=False,randomness_only_in_sensor_generation=True,
        source_sha256={f.name:sha(f) for f in HERE.glob('*.py')},
        native_source_sha256={f.name:sha(f) for f in HERE.glob('*.cpp')},
        original_numeric_source_sha256=original,
        original_results_unchanged=original=={f.name:sha(f) for f in old_dir.glob('*.npz')},
        engine_metadata=metadata)
    write_json(output/'summary.json', summary)
    record_report(output,config,generation,acquisitions,metrics,convergence,events)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['case','so3_proxy_rmse_deg','so3_proxy_final_deg','omega_x_rmse_deg_s','omega_y_rmse_deg_s','omega_z_rmse_deg_s'])
        for name,item in metrics.items():
            writer.writerow([name,item['so3_proxy_rmse_deg'],item['so3_proxy_final_deg'],*item['omega_rmse_deg_s']])
    files=[dict(path=str(f.relative_to(output)),bytes=f.stat().st_size,sha256=sha(f))
        for f in sorted(output.rglob('*')) if f.is_file() and f.name!='manifest.json']
    write_json(output/'manifest.json',dict(files=files,
        bytes_excluding_manifest=sum(f['bytes'] for f in files),
        compiled_binary_or_intermediate_frames_included=False))
    print('Completed',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'frequent_config.json')
    parser.add_argument('--output',type=Path,default=Path('/tmp/synthetic-frequent-spikes-1hz-20261120'))
    run(parser.parse_args())
