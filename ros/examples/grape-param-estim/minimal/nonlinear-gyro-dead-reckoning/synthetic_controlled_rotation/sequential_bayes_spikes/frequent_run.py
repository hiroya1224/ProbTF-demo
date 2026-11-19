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


MODELS = ('nonlinear', 'linear_forced')
MODEL_LABELS = {'nonlinear': '非線形Euler', 'linear_forced': 'ジャイロ項なし'}
OBSOLETE_FILES = ('bayes_white.npz', 'bayes_spike.npz', 'white_case.pdf',
    'white_case.png', 'spike_case.pdf', 'spike_case.png',
    'sensor_noise_components.pdf', 'sensor_noise_components.png')


def predict_models(engine, selected, profile, acquisitions, config, order):
    """Compare dynamics only: both receive the identical contaminated readings."""
    observed = np.asarray(acquisitions['gyro_spike'], float)
    model_data = dict(selected, sparse_gyro=observed.copy())
    result = {}
    for mode in MODELS:
        result[mode] = engine.predict_adaptive(model_data, profile,
            sigma_deg_s=config['assumed_sigma_deg_s'], order=order,
            output_dt=config['output_dt_seconds'], max_step=config['max_step_seconds'],
            observation_sets={'shared': observed.copy()}, mode=mode)['shared']
    return result


def record_report(output, config, generation, acquisitions, metrics, convergence, events):
    active = float(config['spike_active_until_seconds'])
    captured = acquisitions['time'][acquisitions['spike_mask']]
    lines = ['# 同じスパイク入りgyro：非線形Eulerとジャイロ項なしの比較', '',
        '[比較動画](frequent_spikes_comparison.mp4) / [姿勢・角速度](attitude_and_gyro.pdf) / '
        '[共通の生gyro](native_gyro_full_and_zoom.pdf) / [角速度の±2σと半幅](gyro_uncertainty.pdf) / '
        '[観測更新の診断](bayes_diagnostics.pdf)', '',
        '**両モデルが受け取る測定列は同一。どちらにも白色ノイズσ=2 deg/sと頻発スパイクが入る。** '
        '比較の違いは角速度の非線形ジャイロ項だけで、四元数運動学とBayes更新は共通。', '',
        r'非線形：$\dot\omega=J^{-1}(\tau-\omega\times(J\omega))$。', '',
        r'ジャイロ項なし：$\dot\omega=J^{-1}\tau$。両者で $\dot q=\tfrac12q\odot(0,\omega)$ を使う。', '',
        f'{config["duration_seconds"]:g}秒、200 Hzの生データから1 Hzで同じ20件を取得。'
        f'生成白色σ={config["white_sigma_deg_s"]:g} deg/s、推定の観測σ={config["assumed_sigma_deg_s"]:g} deg/s。'
        'J、既知トルク、初期姿勢、初回gyroと初期候補も同じ。', '',
        f'0〜{active:g}秒はPoisson強度{config["spike_rate_hz"]:g}回/秒で'
        f'{generation["realized_spike_count"]}回のスパイクが発生。取得と重なったのは'
        +', '.join(f'{v:g}' for v in captured)+'秒。'
        f'{active:g}秒以降は入力側でスパイクを止め、白色ノイズを残した正常な更新を続ける。', '',
        '| 動力学モデル | 姿勢中心RMSE [deg] | 終端誤差 [deg] | ω RMSE xyz [deg/s] |',
        '|---|---:|---:|---|']
    for name in MODELS:
        item = metrics[name]
        lines.append(f'| {MODEL_LABELS[name]} | {item["so3_proxy_rmse_deg"]:.5f} | '
            f'{item["so3_proxy_final_deg"]:.5f} | '+', '.join(f'{v:.5f}' for v in item['omega_rmse_deg_s'])+' |')
    lines.extend(['', '## 角速度も推定する共同状態', '',
        '状態はbody角速度ωと単位四元数q（Hamilton、wxyz順）。トルクτはbody重心まわりの物理トルクで、'
        'Iの逆はトルク項にも掛かる。各候補のωとqを同時RK4積分し、方位はE[qqᵀ]としても保持する。'
        '回転行列は同じqから統計・表示用に導出する量で、方位状態を回転行列へ置き換えない。', '',
        'ωの全3×3共分散と、方位との依存を保持している。±2σの帯は保存した共分散の実値であり、'
        '角速度曲線の大きい振幅に対して細いため、境界線・独立した半幅図・動画中の数値でも明示する。', '',
        '| モデル | 初期±2σ半幅xyz [deg/s] | 終端±2σ半幅xyz [deg/s] |',
        '|---|---|---|'])
    for name in MODELS:
        final=', '.join(f'{v:.5f}' for v in metrics[name]['final_omega_two_sigma_deg_s'])
        initial=', '.join([f'{2*config["assumed_sigma_deg_s"]:.5f}']*3)
        lines.append(f'| {MODEL_LABELS[name]} | {initial} | {final} |')
    lines.extend(['', 'モデル誤差を観測雑音として自動で増やす処理は入れていない。'
        'したがって、ジャイロ項を省いたモデルの予測平均が大きく外れても、仮定したモデル内の共分散は縮み得る。', '',
        '## 各観測の前後', '',
        '| 時刻 [s] | スパイク取得 | 非線形の姿勢誤差 直前→直後 [deg] | ジャイロ項なし 直前→直後 [deg] |',
        '|---:|---|---|---|'])
    paired = {mode: {r['acquisition_index']:r for r in events[mode]} for mode in MODELS}
    for k, stamp in enumerate(acquisitions['time'][1:],1):
        changes=[]
        for mode in MODELS:
            item=paired[mode][k]
            changes.append(f'{item["prior_so3_proxy_error_deg_reference_only"]:.4f} → '
                           f'{item["posterior_so3_proxy_error_deg_reference_only"]:.4f}')
        lines.append(f'| {stamp:g} | '+('あり' if acquisitions['spike_mask'][k] else 'なし')+' | '+' | '.join(changes)+' |')
    lines.extend(['', '真値比較は評価専用。モデル・更新へ途中の真値やスパイクフラグを渡していない。'
        '両モデルとも同じ初期角速度の候補へ全取得尤度を一度ずつ掛け、姿勢・角速度の依存と全交差共分散を保持する。'
        '外れ値除去・ロバスト化・BFS・過程雑音は加えない。', '',
        '## 決定論的求積の精細化', '',
        '| 次数比較 | モデル | 最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |',
        '|---|---|---:|---:|'])
    for label, models in convergence.items():
        for mode, item in models.items():
            lines.append(f'| {label} | {MODEL_LABELS[mode]} | {item["max_omega_mean_component_difference_deg_s"]:.6g} | '
                f'{item["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '同じ初期Gaussian×全観測尤度を、適応Gauss–Hermite重要度求積で評価する。'
        'proposalは数値積分用であり、事後状態へ代入しない。モンテカルロなし。'
        '有限な積分差を観測の棄却条件に使わない。', '',
        'RPY中心はf(E[R])、帯は同じ回転全共分散からの局所delta-method ±2σで、保証付き95%領域ではない。'
        '生成側と一致するJ・トルクを既知入力として与えた合成回転であり、未知機体の同定実験とは区別する。', '',
        '[全スパイク](spike_events.json) / [観測前後](events.json) / [設定・数値](summary.json) / '
        '[再現手順](../FREQUENT_SPIKES.md)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in OBSOLETE_FILES:
        obsolete = output/name
        if obsolete.exists():
            obsolete.unlink()
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
    previous = None
    convergence, durations = {}, {}
    for order in config['quadrature_orders']:
        print('Frequent-spike adaptive Bayes order', order, flush=True)
        before = time.monotonic()
        cases = predict_models(engine, selected, profile, acquisitions, config, order)
        durations[str(order)] = time.monotonic()-before
        if previous is not None:
            convergence[f'{previous_order} → {order}'] = {
                name: shared.refinement(previous[name], cases[name]) for name in MODELS}
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
    summary = dict(schema='grape/spiked-gyro-dynamics-comparison/v2',config=config,
        generation=generation,metrics=metrics,quadrature_refinement=convergence,
        inference_wall_seconds=durations,wall_seconds=time.monotonic()-started,
        actual_white_noise_std_deg_s=np.rad2deg(np.std(standardized,axis=0,ddof=1)),
        estimator_noise_covariance_rad2_s2=np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2,
        posterior_target='initial N(w0;y0,R) times all acquired k>=1 Gaussian gyro likelihoods; no process noise, reset, or posterior Gaussian replacement',
        observation_input_keys=list(selected),observation_sets='same spike-contaminated gyro values for both models',
        compared_models=list(MODELS),
        model_difference='only omega cross(J omega) omitted in linear_forced; identical nonlinear quaternion kinematics',
        same_measurement_sha256=hashlib.sha256(np.ascontiguousarray(acquisitions['gyro_spike']).tobytes()).hexdigest(),
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
    parser.add_argument('--output',type=Path,default=Path('/tmp/synthetic-spiked-model-comparison-20261120'))
    run(parser.parse_args())
