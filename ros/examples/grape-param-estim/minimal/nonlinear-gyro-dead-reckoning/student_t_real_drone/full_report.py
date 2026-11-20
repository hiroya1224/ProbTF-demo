#!/usr/bin/env python3
"""Describe saved full-native results without rerunning or modifying inference."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

import run


HERE = Path(__file__).resolve().parent
MODELS = ('nonlinear', 'linear_forced')
LABELS = {'nonlinear': '非線形Euler', 'linear_forced': 'ジャイロ項なし'}


def render(output):
    output = Path(output).resolve()
    summary = json.loads((output/'summary.json').read_text())
    previous = json.loads((HERE/'publication/bag8_50_55s/summary.json').read_text())
    first = output/summary['records'][0]['directory']
    provenance = json.loads((first/'input.json').read_text())
    count = provenance['observation_count']
    lines = ['# 原gyro全件の結果', '',
        f'同じ5秒の原gyro **{count}件**を全件使用した。原時計の中央値で'
        f'約{provenance["native_median_rate_hz"]:.6f} Hz。'
        f'初回一件を初期候補に使い、残り{count-1}件をStudent-t尤度として各一度融合した。'
        '再標本化、観測の平均化、bias・時刻補正、モデルの再同定は行っていない。', '',
        '## 同じ条件での取得数比較', '',
        '同じ501時刻でmocapに対する姿勢中心のSO(3)測地距離を評価した。'
        '慣性・トルク・過去余剰平均・初期姿勢・ν=4・Q=0・周辺Rは共通。', '',
        '| 周辺σ [deg/s] | 取得 | 件数 | 非線形RMSE [deg] | 項なしRMSE [deg] | 非線形終端 [deg] | 項なし終端 [deg] |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for full in summary['records']:
        sigma = full['sigma_deg_s']
        old = [row for row in previous['records'] if row['sigma_deg_s'] == sigma]
        for row in sorted(old, key=lambda row:row['rate_hz'])+[full]:
            result = row['metrics']['results']
            frequency = ('全原レート' if row is full else f'{row["rate_hz"]:g} Hz')
            nl, linear = (result[name] for name in MODELS)
            lines.append(f'| {sigma:g} | {frequency} | {row["observation_count"]} | '
                f'{nl["so3_proxy_rmse_deg"]:.6f} | {linear["so3_proxy_rmse_deg"]:.6f} | '
                f'{nl["so3_proxy_final_deg"]:.6f} | {linear["so3_proxy_final_deg"]:.6f} |')
    lines.extend(['', '## 更新している法則と解釈', '',
        r'$$\pi_k(w)\propto\mathcal N(w;y_0,R)\prod_{j=1}^{k}t_4(y_j;\omega(t_j;w),\tfrac12 R).$$', '',
        'ここでwは初期角速度であり、現在の角速度と姿勢は同じ候補wから凍結力学で伝播する。'
        'Q=0なので、1001件を使っても任意の現在角速度へ毎回置き換えられるわけではない。'
        '過去の尤度を失わず、各取得までの情報を使った事後の共同モーメントを表示している。', '',
        '全件化は独立な観測因子を増やす変更であり、固定モデルの系統誤差や未知の時間相関を'
        '修正する変更ではない。隣接gyroの測定誤差を独立とする既存の仮定も維持している。'
        'Student-tはモデルとずれた観測を相対的に弱く評価し得るため、取得数を増やせば'
        'mocapへの姿勢誤差が単調に減るとは限らない。今回の比較だけでこの要因を一意に同定したとは言わない。', '',
        '途中mocapは評価専用。SG角速度の点線も読取専用。元のgyroの値は実測値であり、'
        '追加の白色noise・spikeは入れていない。角速度誤差は生gyroとの差で、角速度真値との誤差ではない。', '',
        '## 数値精度と表示', '',
        '| 周辺σ | モデル | 7→11点：最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |',
        '|---:|---|---:|---:|'])
    for row in summary['records']:
        detail = json.loads((output/row['directory']/'summary.json').read_text())
        for mode in MODELS:
            residual = detail['quadrature_refinement']['7 -> 11'][mode]
            lines.append(f'| {row["sigma_deg_s"]:g} | {LABELS[mode]} | '
                f'{residual["max_omega_mean_component_difference_deg_s"]:.6g} | '
                f'{residual["max_rpy_proxy_component_difference_deg"]:.6g} |')
    lines.extend(['', '各軸11点の決定論的求積の結果を図示した。表は全保存時刻・全軸の最大差であり、'
        '連続密度に対する収束や精度保証ではない。有限な数値診断による観測棄却・更新停止はしていない。'
        'Gaussian proposalは積分点の配置用で、事後分布そのものへの置換ではない。', '',
        '**現状は跳びが残り、数値的にも未収束の箇所がある。高精度な復元や、非線形モデルの優位を'
        '示す結果とは扱わない。** 全件化でσ=2.5の5 Hz比較よりRMSEは小さくなった一方、'
        'σ=5では10 Hzの結果より大きい。取得数の増加だけで問題が解決したとは言えない。', '',
        '観測前後2001行は既存engineの値を保持し、それ以外の予測時刻を追加した。'
        f'最後の観測後{provenance["prediction_tail_seconds"]*1e6:.3f} μsも同じ候補を物理伝播する。'
        '密な追加予測は、同じ因果的proposalを共有する区間で計算をまとめた。'
        '区間分割の有限RK4差を[dense_grouping_audit.json](dense_grouping_audit.json)へ記録した。', '',
        '姿勢・角速度の全共同モーメントと全交差を保存する。図の±2σはその同じ共分散。'
        'RPYは回転モーメントからの局所表示近似であり、保証された95%領域ではない。'
        'モデルパラメータと未知の余剰トルクの不確かさを含めた帯ではない。', '',
        '[図の一覧](INDEX.md) ／ [SciPyによる全1000尤度と全共同モーメントの独立照合](verification.json) ／ '
        '[以前の合成・実測結果のhash保存確認](preserved_sources.json)', ''])
    audit_path = output/'quadrature_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines.extend(['## 跳び付近の追加数値診断（σ=2.5）', '',
            '7点と11点で最もRPY差が大きかった保存行について、11点側が実際に使った同じproposalを'
            '固定し、7／11／17／25点で同じ過去尤度を積分した。観測更新やproposalを再実行せず、'
            '公開した推定値も変更していない。', '',
            '| モデル | bag時刻 [s] | 到着済み尤度数 | 同じproposalの11→25点：最大RPY差 [deg] | log正規化積分の差 |',
            '|---|---:|---:|---:|---:|'])
        for mode in MODELS:
            row = audit['models'][mode]['selected_rows'][0]
            by_order = {item['order']:item for item in row['same_GH11_proposal_evaluations']}
            delta = np.asarray(by_order[25]['moments']['rpy_proxy_rad'])-by_order[11]['moments']['rpy_proxy_rad']
            maximum = float(np.max(np.abs(np.rad2deg(np.arctan2(np.sin(delta),np.cos(delta))))))
            lines.append(f'| {LABELS[mode]} | {row["time"]:.9f} | {row["included_noninitial_factor_count"]} | '
                f'{maximum:.6f} | {by_order[25]["log_normalizer"]-by_order[11]["log_normalizer"]:.6f} |')
        lines.extend(['', '7点／11点で選ばれるproposal領域の差に加え、同じproposal内にも求積差が残る。'
            'したがって、跳びを物理モデルの不一致だけで説明することはできない。'
            'この局所診断は分離した他のモードまで十分に積分できたという証明でもない。'
            '独立検証の全尤度・モーメント一致は、保存した有限求積内の整合性確認であり、'
            'この収束問題とは区別する。[全数値](quadrature_audit.json)。', ''])
    (output/'RESULTS.md').write_text('\n'.join(lines))
    sources = [Path(__file__), output/'summary.json',
        HERE/'publication/bag8_50_55s/summary.json']
    if audit_path.exists():
        sources.append(audit_path)
    record = dict(schema='grape/full-native-result-report/v1',
        inference_rerun=False, candidate_or_covariance_changed=False,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        result_sha256=hashlib.sha256((output/'RESULTS.md').read_bytes()).hexdigest(),
        semantics='Postprocessing/report code hashes supplement the source snapshot recorded at the end of the primary inference run')
    (output/'report_generation.json').write_text(json.dumps(record,indent=2)+'\n')
    manifest = run.load('_real_full_report_manifest', run.SHARED/'rate_comparison_verify.py')
    manifest.refresh_manifest(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=HERE/'publication/bag8_full_gyro')
    render(parser.parse_args().output)
