#!/usr/bin/env python3
"""Real native gyro, Student-t observations and angular process sigma 1 deg/s."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import time

import numpy as np

import run as old


HERE = Path(__file__).resolve().parent
MODELS = ('nonlinear', 'linear_forced')


def protected_hashes():
    result = old.preserved_hashes(include_sparse_real=True)
    root = HERE/'publication/bag8_full_gyro'
    result.update({str(path.relative_to(old.BASE)):old.shared.sha(path)
        for path in sorted(root.rglob('*')) if path.is_file()
        and 'cache' not in path.relative_to(root).parts})
    return result


def write_report(output):
    summary = json.loads((output/'summary.json').read_text())
    baseline = json.loads((HERE/'publication/bag8_full_gyro/summary.json').read_text())
    lines = ['# 実ドローン：全原gyro＋角速度プロセスσ=1 deg/s', '',
        '同じ#8の50.000988619〜55.000988619秒・原gyro全1001件（約200 Hz）を使用。'
        'Student-t ν=4、観測の周辺σ=2.5／5 deg/s、慣性・実指令／gimbal由来トルク・'
        '過去余剰トルク平均と初期姿勢を維持し、非線形Euler／ジャイロ項なしを比較した。', '',
        '各原取得の直前に独立な角速度増分を一度加える。'
        '$Q_\\omega=(\\pi/180)^2I_3=0.0003046174197867086I_3$ [rad²/s²]。'
        '約0.005秒ごとの増分であり、連続時間PSDや角加速度ノイズではない。'
        '不等間隔でも一取得区間あたり同じQを使う。最初の観測と最後の取得後の尾部には追加しない。', '',
        '| 観測σ [deg/s] | Q | 非線形：姿勢RMSE [deg] | 項なし：姿勢RMSE [deg] | 非線形：終端 [deg] | 項なし：終端 [deg] |',
        '|---:|---|---:|---:|---:|---:|']
    for record in summary['records']:
        sigma = record['sigma_deg_s']
        zero = next(item for item in baseline['records'] if item['sigma_deg_s'] == sigma)
        for row, label in ((zero,'0'), (record,'σ=1 deg/s')):
            nl, linear = (row['metrics']['results'][name] for name in MODELS)
            lines.append(f'| {sigma:g} | {label} | {nl["so3_proxy_rmse_deg"]:.6f} | '
                f'{linear["so3_proxy_rmse_deg"]:.6f} | {nl["so3_proxy_final_deg"]:.6f} | '
                f'{linear["so3_proxy_final_deg"]:.6f} |')
    lines.extend(['', '姿勢誤差は同じ501時刻でのmocapとのSO(3)測地距離。'
        '角速度は雑音を含む生gyroとの比較であり、角速度真値との誤差ではない。'
        'Q=0の旧結果は未収束の箇所を含む参考値としてそのまま併記し、再計算していない。', '',
        '## 図', ''])
    for record in summary['records']:
        relative = record['directory']
        lines.append(f'- 観測σ={record["sigma_deg_s"]:g} deg/s：'
            f'[姿勢・gyro・±2σ]({relative}/attitude_and_gyro.pdf) ／ '
            f'[誤差]({relative}/reconstruction_errors.pdf)')
    lines.extend(['', '## 保持する法則', '',
        r'$$p_k^-(q,\omega)=\int p_{k-1}^+(dq_0,d\omega_0)\,\delta_{q_0\odot\Delta q_k(\omega_0)}(dq)\,\mathcal N(\omega;f_k(\omega_0),Q_\omega),$$', '',
        r'$$p_k^+(q,\omega)\propto p_k^-(q,\omega)\,t_4(y_k;\omega,\tfrac12R).$$', '',
        '区間内のEuler式と四元数式を同時にRK4で積分する。差はω×Jω項の有無だけ。'
        '区間終端の独立な角速度増分は、その瞬間の姿勢を直接変更せず、次区間の姿勢伝播へ作用する。', '',
        '初回は従来どおり一度だけ $N(\\omega_0;y_0,R)$ で初期化する。以後は前回の共同候補を'
        '予測・条件付けして持ち越す。Q=0の初期W0候補へ共分散だけ足す処理や、観測ごとに'
        '角速度を独立なGaussianへ置き換える処理ではない。', '',
        '## 決定論的積分と近似', '',
        '角速度の遷移Gaussian混合にStudent-t因子を掛けた連続密度を積分対象とし、時刻間は'
        '決定論的Nyström求積の重みと条件付き姿勢モーメントで持ち越す。連続密度関数そのものの'
        '全履歴厳密保持ではない。Gaussian proposalは積分点の配置だけに使い、実際の密度を'
        'proposalで割って重みを計算する。Student-t事後をGaussianへ置換しない。', '',
        '角速度を条件とする四元数二次モーメント・回転の全9×9共分散を輸送・融合し、'
        '角速度の全3×3共分散と姿勢との交差を保持する。観測因子は角速度だけに依存するため、'
        '角速度を固定した姿勢の条件付き法則はその因子によって変わらない。'
        '一方、その条件付き法則を混合する角速度の重みは観測で変化する。', '',
        '遷移・条件付けの式と有限求積誤差は区別する。積分点を状態の物理的な粒子とは扱わず、'
        'Monte Carloや共分散floor、外れ値除去、有限な精度診断による観測停止を使わない。'
        '求積の再配置を繰り返す有限近似の誤差は時系列で蓄積し得る。', '',
        '| 観測σ | モデル | 求積変更 | 最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |',
        '|---:|---|---|---:|---:|'])
    for record in summary['records']:
        detail = json.loads((output/record['directory']/'summary.json').read_text())
        for refinement, modes in detail['quadrature_refinement'].items():
            for mode, item in modes.items():
                lines.append(f'| {record["sigma_deg_s"]:g} | {mode} | {refinement} | '
                    f'{item["max_omega_mean_component_difference_deg_s"]:.6g} | '
                    f'{item["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '図は細かい側の同じ保存モーメントから描く。RPY中心と帯は回転モーメントの'
        '局所表示近似で、帯は同じ共分散の±2σ。推定値を補間・平滑化せず、描画用の分散変更もしない。'
        'モデルパラメータ不確かさ全体を含む保証付き領域とは説明しない。', '',
        '途中mocapは描画・評価専用で、姿勢初期整合以外には推定へ渡さない。'
        '原gyro・指令・トルクへ人工noiseやspikeを追加せず、biasや時刻ずれも補正しない。'
        '観測は全1001件、初回以外の観測因子とプロセス増分は各1000件。', '',
        '[全条件・数値・source hash](summary.json) ／ [独立検証](verification.json) ／ '
        '[旧結果保存確認](preserved_sources.json)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))
    manifest = old.load('_real_student_process_report_manifest', old.SHARED/'rate_comparison_verify.py')
    manifest.refresh_manifest(output)


def run(args):
    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    q = np.eye(3)*np.deg2rad(config['process_sigma_deg_s'])**2
    config['process_covariance_rad2_s2'] = q.tolist()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    old.shared.write_json(output/'config.json', config)
    engine = old.load('_real_student_process_engine', HERE/'student_t_process_engine.py')
    plotter = old.load('_real_student_process_plots', HERE/'plotting.py')
    source, acquired, selected, physical, provenance = old.inputs.load_full_inputs()
    np.savez_compressed(output/'reference_input.npz', **source)
    original = dict(np.load(old.inputs.REFINED/'refined_torque_profile.npz', allow_pickle=False))
    np.savez_compressed(output/'frozen_physical_input.npz', inertia=original['inertia_body'],
        torque_time=original['torque_profile_time'], torque_actuator_body=original['torque_actuator_body'],
        torque_applied_body=original['torque_applied_body'], residual_mean_body=original['residual_mean_body'])
    rate_directory = output/'rate_full_native'
    rate_directory.mkdir(exist_ok=True)
    np.savez_compressed(rate_directory/'acquisitions.npz', **acquired)
    records = []
    for sigma in config['sigmas_deg_s']:
        directory = rate_directory/f'sigma_{old.label(sigma)}_deg_s'
        cache = directory/'cache'
        cache.mkdir(parents=True, exist_ok=True)
        case_config = dict(config, sigma_deg_s=sigma, rate_hz=provenance['native_median_rate_hz'],
            process_reference_interval_seconds=float(np.median(np.diff(acquired['time']))))
        old.shared.write_json(directory/'config.json', case_config)
        old.shared.write_json(directory/'input.json', provenance)
        np.savez_compressed(directory/'acquisitions.npz', **acquired)
        previous = None
        refinement, timings = {}, {}
        for order in config['quadrature_orders']:
            started = time.monotonic()
            def predict(mode):
                print(f'Real full-native Student-t / Q sigma 1 / R sigma {sigma:g}: {mode} GH {order}', flush=True)
                case = engine.predict_process_student(selected, physical, sigma_deg_s=sigma,
                    process_sigma_deg_s=config['process_sigma_deg_s'], student_df=config['student_df'],
                    order=order, mode=mode, max_step=config['max_step_seconds'],
                    output_dt=config['output_dt_seconds'])
                old.shared.save_case(cache/f'order{order}_{mode}.npz', case)
                old.shared.write_json(cache/f'order{order}_{mode}_metadata.json', case['metadata'])
                print(f'FINISHED R sigma {sigma:g} {mode} GH {order}', flush=True)
                return case
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {pool.submit(predict, mode):mode for mode in MODELS}
                cases = {pending[future]:future.result() for future in as_completed(pending)}
            cases = {mode:cases[mode] for mode in MODELS}
            timings[str(order)] = time.monotonic()-started
            if previous is not None:
                refinement[f'{previous_order} -> {order}'] = {
                    mode:old.shared.refinement(previous[mode],cases[mode]) for mode in MODELS}
            previous, previous_order = cases, order
        for mode, case in cases.items():
            old.shared.save_case(directory/f'bayes_{mode}.npz', case)
            old.shared.write_json(directory/f'diagnostics_{mode}.json', case.get('diagnostics', []))
        plotter.render_case(directory, source, acquired, cases, metadata={'config':case_config})
        metrics = json.loads((directory/'metrics.json').read_text())
        old.shared.write_json(directory/'summary.json', dict(config=case_config, input=provenance,
            engine_metadata={mode:case['metadata'] for mode,case in cases.items()},
            quadrature_refinement=refinement, inference_wall_seconds=timings,
            initial_reading_used_once=True, observation_count=len(acquired['time']),
            process_increment_count=len(acquired['time'])-1,
            shared_state='Sequential deterministic quadrature of current joint density with conditional quaternion and full rotation moments',
            finite_numerical_differences_are_diagnostics_not_gates=True))
        old.shared.write_json(directory/'quadrature_refinement.json', refinement)
        records.append(dict(sigma_deg_s=sigma,rate_hz=provenance['native_median_rate_hz'],
            observation_count=len(acquired['time']),directory=str(directory.relative_to(output)),metrics=metrics))
        print('PUBLISHED '+str(directory), flush=True)
    after = protected_hashes()
    old.shared.write_json(output/'preserved_sources.json', dict(before_sha256=before,after_sha256=after,
        every_preserved_file_unchanged=before==after,
        note='Original synthetic, refined, sparse real and full-native Q=0 publications are read only; caches excluded'))
    old.shared.write_json(output/'summary.json', dict(schema='grape/real-full-native-student-process/v1',
        config=config,records=records,input=provenance,observation_count=len(acquired['time']),
        process_increment_count=len(acquired['time'])-1,
        preservation=before==after,monte_carlo=False,model_refitted=False,
        sensor_values_noise_bias_or_clock_changed=False,
        source_sha256={str(path.relative_to(old.BASE)):old.shared.sha(path) for path in sorted(HERE.glob('*.py'))},
        wall_seconds=time.monotonic()-begun))
    with (output/'comparison_table.csv').open('w', newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['sigma_deg_s','process_sigma_deg_s','model','observation_count','so3_rmse_deg','so3_final_deg'])
        for record in records:
            for mode, metric in record['metrics']['results'].items():
                writer.writerow([record['sigma_deg_s'],config['process_sigma_deg_s'],mode,
                    record['observation_count'],metric['so3_proxy_rmse_deg'],metric['so3_proxy_final_deg']])
    write_report(output)
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'process_config.json')
    parser.add_argument('--output',type=Path,default=HERE/'publication/bag8_full_gyro_q1')
    run(parser.parse_args())
