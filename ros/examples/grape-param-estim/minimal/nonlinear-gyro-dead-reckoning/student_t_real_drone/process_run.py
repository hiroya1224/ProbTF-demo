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


def protected_hashes(include_full_q1=False):
    result = old.preserved_hashes(include_sparse_real=True)
    root = HERE/'publication/bag8_full_gyro'
    result.update({str(path.relative_to(old.BASE)):old.shared.sha(path)
        for path in sorted(root.rglob('*')) if path.is_file()
        and 'cache' not in path.relative_to(root).parts})
    if include_full_q1:
        root = HERE/'publication/bag8_full_gyro_q1'
        result.update({str(path.relative_to(old.BASE)):old.shared.sha(path)
            for path in sorted(root.rglob('*')) if path.is_file()
            and 'cache' not in path.relative_to(root).parts})
    return result


def write_report(output):
    summary = json.loads((output/'summary.json').read_text())
    if summary['config'].get('acquisition_mode') == 'selected_rate':
        return write_rate_report(output, summary)
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


def write_rate_report(output, summary):
    sparse_zero = json.loads((HERE/'publication/bag8_50_55s/summary.json').read_text())
    full_zero = json.loads((HERE/'publication/bag8_full_gyro/summary.json').read_text())
    full_q1 = json.loads((HERE/'publication/bag8_full_gyro_q1/summary.json').read_text())
    lines = ['# 原gyroを間引いたStudent-t＋Qσ=1 deg/s比較', '',
        '以前と同じ原gyroの0.5／2／5／10 Hz取得で、非線形Euler／ジャイロ項なしを比較した。'
        '同じbag #8の50.000988619〜55.000988619秒、同じ初期姿勢・J・実指令／gimbal由来トルク・'
        '過去余剰トルク平均を使用。ν=4、観測Rの周辺σ=2.5／5 deg/sを維持した。', '',
        r'角速度プロセス増分は、取得した観測間隔ごとに $Q_\omega=(\pi/180)^2I_3$ [rad²/s²] を一度加える。',
        '間引いて使わない原gyro時刻にはQも観測因子も追加しない。初回と最後の観測後の尾部にはQを追加しない。'
        '取得頻度に応じたdt倍率は掛けていない。したがって一秒あたりの累積過程分散は取得頻度によって異なり、'
        '同一の連続時間拡散モデルで観測数だけを変えた比較とは区別する。', '',
        '## 同じ501時刻での姿勢RMSE [deg]', '',
        '| 観測σ [deg/s] | 取得 | 観測数 | Q1：非線形 | Q1：項なし | Q0：非線形 | Q0：項なし |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for sigma in summary['config']['sigmas_deg_s']:
        rows = sorted((r for r in summary['records'] if r['sigma_deg_s']==sigma),key=lambda r:r['rate_hz'])
        rows += [next(r for r in full_q1['records'] if r['sigma_deg_s']==sigma)]
        for row in rows:
            native = row not in summary['records']
            references = full_zero['records'] if native else sparse_zero['records']
            zero = next(r for r in references if r['sigma_deg_s']==sigma and (native or r['rate_hz']==row['rate_hz']))
            q_values = row['metrics']['results']
            z_values = zero['metrics']['results']
            frequency = '全原レート（約200 Hz）' if native else f'{row["rate_hz"]:g} Hz'
            lines.append(f'| {sigma:g} | {frequency} | {row["observation_count"]} | '
                f'{q_values["nonlinear"]["so3_proxy_rmse_deg"]:.6f} | '
                f'{q_values["linear_forced"]["so3_proxy_rmse_deg"]:.6f} | '
                f'{z_values["nonlinear"]["so3_proxy_rmse_deg"]:.6f} | '
                f'{z_values["linear_forced"]["so3_proxy_rmse_deg"]:.6f} |')
    lines.extend(['', '**間引き条件では7→11点でも大きな求積差が残る箇所がある。以下の結果は有限求積の'
        '実行結果として保存し、収束済みの復元精度や非線形モデルの優位とは主張しない。** '
        '各条件の差は末尾の表とquadrature_refinement.jsonへ記録する。', '',
        'Q1はσ=1 deg/sを意味し、単位行列Qそのものではない。Q0と全原レートの旧保存値は'
        '変更せず併記した。Q0には以前記録した未収束箇所があるため、収束済みの真値とは扱わない。'
        '途中mocapは同じ501時刻での評価・表示専用。角速度の誤差は雑音を含む生gyroとの差である。', '',
        '## 図と検証', ''])
    for row in sorted(summary['records'],key=lambda r:(r['sigma_deg_s'],r['rate_hz'])):
        directory = row['directory']
        lines.append(f'- {row["rate_hz"]:g} Hz、観測σ={row["sigma_deg_s"]:g} deg/s：'
            f'[姿勢・gyro・±2σ]({directory}/attitude_and_gyro.pdf) ／ [誤差]({directory}/reconstruction_errors.pdf)')
    lines.extend(['', '[全原レートQ1の図と更新式](../bag8_full_gyro_q1/INDEX.md)。今回も現在の共同候補を'
        'Markov予測・Student-t条件付けで持ち越す同じengineを使い、初期角速度候補だけのQ0法則へ'
        '共分散を加算する処理や、毎回独立Gaussianへ置換する処理は行わない。', '',
        '時刻間は連続ターゲットに対する決定論的Nyström求積と、角速度を条件にした四元数二次モーメント・'
        '回転の全共分散を保持する有限近似。Gaussian proposalは配置用で密度比を重みに残す。'
        'Monte Carlo、共分散floor、外れ値棄却、有限診断による更新停止は使わない。', '',
        '| 観測σ | Hz | モデル | 求積変更 | 最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |',
        '|---:|---:|---|---|---:|---:|'])
    for row in sorted(summary['records'],key=lambda r:(r['sigma_deg_s'],r['rate_hz'])):
        detail = json.loads((output/row['directory']/'summary.json').read_text())
        for refinement, modes in detail['quadrature_refinement'].items():
            for mode, item in modes.items():
                lines.append(f'| {row["sigma_deg_s"]:g} | {row["rate_hz"]:g} | {mode} | {refinement} | '
                    f'{item["max_omega_mean_component_difference_deg_s"]:.6g} | '
                    f'{item["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '図の帯は同じ保存共同モーメントの±2σで、RPYは局所表示近似。'
        '推定値の補間・平滑化や表示用の分散変更はしない。有限求積差は診断であり精度保証ではない。'
        '原値・header/record ns・元indexを保持し、未来の観測・未取得gyroを予測へ入れない。', '',
        '[条件・数値](summary.json) ／ [独立検証](verification.json) ／ '
        '[全原レートと旧結果の保存hash](preserved_sources.json)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))
    manifest = old.load('_real_student_selected_process_manifest', old.SHARED/'rate_comparison_verify.py')
    manifest.refresh_manifest(output)


def run(args):
    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    selected_rate = config.get('acquisition_mode') == 'selected_rate'
    q = np.eye(3)*np.deg2rad(config['process_sigma_deg_s'])**2
    config['process_covariance_rad2_s2'] = q.tolist()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = protected_hashes(include_full_q1=selected_rate)
    old.shared.write_json(output/'config.json', config)
    engine = old.load('_real_student_process_engine', HERE/'student_t_process_engine.py')
    plotter = old.load('_real_student_process_plots', HERE/'plotting.py')
    source, acquired, selected, physical, provenance = old.inputs.load_full_inputs()
    np.savez_compressed(output/'reference_input.npz', **source)
    original = dict(np.load(old.inputs.REFINED/'refined_torque_profile.npz', allow_pickle=False))
    np.savez_compressed(output/'frozen_physical_input.npz', inertia=original['inertia_body'],
        torque_time=original['torque_profile_time'], torque_actuator_body=original['torque_actuator_body'],
        torque_applied_body=original['torque_applied_body'], residual_mean_body=original['residual_mean_body'])
    records = []
    rates = config['rates_hz'] if selected_rate else [None]
    jobs = [(rate,sigma) for sigma in config['sigmas_deg_s'] for rate in rates]
    primary = (config.get('primary_rate_hz'),config['primary_sigma_deg_s'])
    if primary in jobs:
        jobs = [primary]+[job for job in jobs if job != primary]
    for rate, sigma in jobs:
        source, acquired, selected, physical, provenance = (old.inputs.load_inputs(rate)
            if selected_rate else old.inputs.load_full_inputs())
        if selected_rate:
            provenance.update(acquisition_mode='selected_rate',
                full_window_native_record_count=len(source['gyro_time']),
                process_interval_semantics='Same angular-velocity increment covariance per selected acquisition; no time rescaling')
        rate_directory = output/(f'rate_{old.label(rate)}_hz' if selected_rate else 'rate_full_native')
        rate_directory.mkdir(exist_ok=True)
        np.savez_compressed(rate_directory/'acquisitions.npz', **acquired)
        directory = rate_directory/f'sigma_{old.label(sigma)}_deg_s'
        cache = directory/'cache'
        cache.mkdir(parents=True, exist_ok=True)
        rate_value = rate if selected_rate else provenance['native_median_rate_hz']
        case_config = dict(config, sigma_deg_s=sigma, rate_hz=rate_value,
            process_reference_interval_seconds=float(np.median(np.diff(acquired['time']))))
        old.shared.write_json(directory/'config.json', case_config)
        old.shared.write_json(directory/'input.json', provenance)
        np.savez_compressed(directory/'acquisitions.npz', **acquired)
        previous = None
        refinement, timings = {}, {}
        for order in config['quadrature_orders']:
            started = time.monotonic()
            def predict(mode):
                print(f'Real {rate_value:g} Hz Student-t / Q sigma 1 / R sigma {sigma:g}: {mode} GH {order}', flush=True)
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
        records.append(dict(sigma_deg_s=sigma,rate_hz=rate_value,
            observation_count=len(acquired['time']),directory=str(directory.relative_to(output)),metrics=metrics))
        print('PUBLISHED '+str(directory), flush=True)
    after = protected_hashes(include_full_q1=selected_rate)
    old.shared.write_json(output/'preserved_sources.json', dict(before_sha256=before,after_sha256=after,
        every_preserved_file_unchanged=before==after,
        note='Original synthetic, refined, sparse real and full-native Q=0 (and Q=1 for selected_rate) publications are read only; caches excluded'))
    old.shared.write_json(output/'summary.json', dict(schema='grape/real-full-native-student-process/v1',
        config=config,records=records,input=provenance,
        observation_count=None if selected_rate else len(acquired['time']),
        process_increment_count=None if selected_rate else len(acquired['time'])-1,
        observation_count_by_rate_hz={format(record['rate_hz'],'g'):record['observation_count'] for record in records},
        preservation=before==after,monte_carlo=False,model_refitted=False,
        sensor_values_noise_bias_or_clock_changed=False,
        source_sha256={str(path.relative_to(old.BASE)):old.shared.sha(path) for path in sorted(HERE.glob('*.py'))},
        wall_seconds=time.monotonic()-begun))
    with (output/'comparison_table.csv').open('w', newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['rate_hz','sigma_deg_s','process_sigma_deg_s','model','observation_count','so3_rmse_deg','so3_final_deg'])
        for record in records:
            for mode, metric in record['metrics']['results'].items():
                writer.writerow([record['rate_hz'],record['sigma_deg_s'],config['process_sigma_deg_s'],mode,
                    record['observation_count'],metric['so3_proxy_rmse_deg'],metric['so3_proxy_final_deg']])
    write_report(output)
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'process_config.json')
    parser.add_argument('--output',type=Path,default=HERE/'publication/bag8_full_gyro_q1')
    run(parser.parse_args())
