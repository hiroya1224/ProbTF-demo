"""Small human-readable parameter/result report from the saved experiment."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def parameter_rows(fit):
    initial, refined = fit['initial'], fit['refined']
    rows = []
    for name, (i, j) in zip(('Jxx', 'Jyy', 'Jzz', 'Jxy', 'Jxz', 'Jyz'),
                           ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        rows.append((name, 'kg m2', initial['inertia_body_kg_m2'][i][j], refined['inertia_body_kg_m2'][i][j]))
    for i, name in enumerate(('CoG x', 'CoG y', 'CoG z')):
        rows.append((name, 'm', initial['cog_offset_body_m'][i], refined['cog_offset_body_m'][i]))
    for i in range(4):
        rows.append((f'Rotor {i} thrust multiplier', '1', initial['force_effectiveness'][i], refined['force_effectiveness'][i]))
    rows.append(('Thrust time constant', 's', initial['thrust_tau_seconds'], refined['thrust_tau_seconds']))
    return rows


def render_report(output):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    fit = summary['fit']
    audit = summary['identifiability']
    train_start, train_end = summary['training_interval']
    eval_start, eval_end = summary['evaluation_interval']
    rows = parameter_rows(fit)
    selected = [r for r in summary['results'] if r['sigma_deg_s'] == 2.5]
    lines = ['# mocapでrefineした回転モデル：結果', '',
        f'調整に使ったmocapはbag #8の{train_start:g}〜{train_end:g}秒、予測評価は{eval_start:.6f}〜{eval_end:.6f}秒。SG窓{summary["sg_window_seconds"]:g}秒・3次。',
        'body角速度は正規化四元数の一次微分、角加速度は二次微分から求めた。', '',
        '最小二乗の式は、時刻の台形積分重みをwとして', '',
        r'\[\min_\theta\sum_n w_n\|J_\theta\dot\omega_n+\omega_n\times(J_\theta\omega_n)-\tau_{\mathrm{act},\theta}(t_n)\|^2.\]', '',
        '慣性traceを公称値に固定し、慣性形状5・重心3・推力係数4・推力時定数1の13座標を、公称値から1回の探索で調整した。',
        '時定数には公称値がないため、原指令周期の中央値を数値初期値にした。反動トルク比とロータ幾何は固定。', '',
        '| パラメータ | 単位 | 初期値 | 調整後 |', '|---|---|---:|---:|']
    lines.extend(f'| {name} | {unit} | {a:.8g} | {b:.8g} |' for name, unit, a, b in rows)
    lines.extend(['', f"全慣性のtraceは{fit['fixed_trace_inertia_kg_m2']:.12g} kg m²。共通の慣性／トルク倍率は回転式だけでは同定できない。",
        f"log chartのJacobianはrank {fit['jacobian_rank']}/13、条件数{fit['condition_number']:.4g}。直接のtrace-free慣性座標では局所rank {audit['direct_physical_residual_jacobian_rank']}/13。全特異値と基底を保存した。",
        f"C=trace(J)/2 I-Jの最小固有値は{audit['physical_second_moment_eigenvalues_kg_m2'][0]:.5g} kg m²。慣性が平面剛体の境界に近づき、log chartの感度が縮退した。このrank差を追加の物理パラメータ非同定とは解釈しない。",
        '公称値からの変化が大きく、低励起ホバリングの有効局所モデルとして扱う。物理的な真値を一意に復元したという結果ではない。',
        f"回転式だけの同定であり、学習区間の平均推力zは{audit['training_mean_actuator_force_body_n'][2]:.4f} N、公称質量の重量は{audit['nominal_mass_weight_n']:.4f} N。並進力の整合は課していない。係数を校正済みの実推力や6自由度モデルの同定値と扱わない。", '',
        '学習トルク残差RMS [Nm]（余剰平均を加える前）：', '',
        '| 軸 | 初期 | 調整後 |', '|---|---:|---:|'])
    for i, axis in enumerate('xyz'):
        lines.append(f"| {axis} | {fit['initial_training_residual_rms_nm'][i]:.6f} | {fit['refined_training_residual_rms_nm'][i]:.6f} |")
    lines.extend(['', '学習残差の時間重み付き平均だけを固定して、両モデルの予測へそれぞれ加算した。',
        f"調整後の固定余剰平均は{fit['refined_residual_mean_body_nm']} Nm。評価区間からは求めていない。", '',
        'sigma=2.5 deg/s、SO(3)姿勢RMSE [deg]（同じ501時刻）：', '',
        '| gyro Hz | 取得数 | 公称初期モデル | 調整後の非線形モデル |', '|---:|---:|---:|---:|'])
    for record in selected:
        lines.append('| '+str(record['rate_hz'])+' | '+str(record['original_gyro_count'])+' | '+
                     ' | '.join(f"{record['methods'][name]['geodesic_rmse_deg']:.4f}" for name in ('nominal', 'refined'))+' |')
    lines.extend(['', '学習の当てはまりと予測精度は別に示した。モデルrefineで改善しても、評価区間の誤差はなお大きく残っている。',
        '今回のトルク最小二乗は、角加速度誤差をJ^T Jで重み付けする。Jも調整するので、この重み自体も変化する。トルク残差が減ったことを姿勢予測の適合と同一視しない。',
        '[ずれの分解](DRIFT_ANALYSIS.md)に、評価区間の平均角加速度誤差、各取得区間のトルク誤差・非線形項・gyro誤差の寄与を保存した。',
        '評価区間のmocapと高密度gyroは参照専用。初期姿勢を一度整合した後、各取得時刻の原gyroだけを予測へ入れた。',
        '角速度の不確かさは各取得で独立な等方Gaussian。指定sigmaを変えても測定値そのものは変えず、乱数は追加しない。',
        '6点の決定論的求積、非線形RK4、指令応答の有限時間格子、RPYのdelta-method表示近似がある。',
        '帯は固定したJ・推力係数・余剰平均に条件付けた±2sigmaで、モデル誤差や将来の余剰トルク変動を包含する保証ではない。',
        'SG窓が重なるため残差には時間相関がある。保存したGauss–Newton曲率共分散は局所診断であり、校正済みの物理パラメータ事後ではない。',
        'null方向の不確かさをゼロと解釈しない。推定の共分散floor・jitter・枝棄却・Monte Carloは使っていない。', '',
        '[全18条件](INDEX.md) / [一覧図](rate_overview_sigma_2p5.pdf) / [動力学照合](dynamics_fit.pdf) / [数値刻み・補間の照合](numerical_sensitivity.json)', ''])
    (output/'RESULTS.md').write_text('\n'.join(lines))

    with PdfPages(output/'fit_report.pdf') as pdf:
        fig = plt.figure(figsize=(10, 11))
        fig.text(.07, .95, 'Mocap-refined nonlinear Euler model', fontsize=18)
        fig.text(.07, .91, f'Train: bag8 {train_start:g}-{train_end:g} s; held-out prediction: {eval_start:.3f}-{eval_end:.3f} s\n'
                 f'Centered SG {summary["sg_window_seconds"]:g} s / degree 3; normalized quaternion first/second derivatives\n'
                 'Ordinary time-weighted torque LS; nonlinear omega x J omega retained\n'
                 'Fixed nominal trace(J), rotor geometry and reaction ratio; 13 fitted coordinates', fontsize=10, linespacing=1.6, va='top')
        axis = fig.add_axes([.06, .28, .88, .52])
        axis.axis('off')
        table = axis.table(cellText=[[name, unit, f'{a:.7g}', f'{b:.7g}'] for name, unit, a, b in rows],
                           colLabels=['Parameter', 'Unit', 'Nominal / numerical start', 'Refined'],
                           cellLoc='left', colWidths=[.36, .12, .26, .26], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        fig.text(.07, .25, f"trace(J) = {fit['fixed_trace_inertia_kg_m2']:.10g} kg m2 (scale convention)\n"
                 f"Log-chart rank {fit['jacobian_rank']}/13; direct physical local rank {audit['direct_physical_residual_jacobian_rank']}/13\n"
                 f"Near planar-inertia boundary: min eigenvalue(C)={audit['physical_second_moment_eigenvalues_kg_m2'][0]:.3g}\n"
                 'Rotational-only effective fit; translational force balance is not constrained.\n'
                 'Thrust tau seed is the median command period, not a physical nominal time constant.\n'
                 'Curvature covariance is unreliable near the boundary; SG errors are correlated.',
                 fontsize=10, linespacing=1.7, va='top')
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw=dict(height_ratios=[1, 1.6]))
        singular = np.asarray(fit['jacobian_singular_values'])
        axes[0].semilogy(np.arange(1, len(singular)+1), singular, 'o-')
        axes[0].set_xlabel('Jacobian singular-value index')
        axes[0].set_ylabel('Singular value')
        axes[0].grid(alpha=.3)
        axes[1].axis('off')
        table_rows = [[f"{r['rate_hz']:g}", str(r['original_gyro_count']),
                      *[f"{r['methods'][name]['geodesic_rmse_deg']:.4f}" for name in ('nominal', 'refined')]] for r in selected]
        table = axes[1].table(cellText=table_rows,
            colLabels=['Hz', 'Readings', 'Nominal nonlinear', 'Refined nonlinear'],
            bbox=[0, .48, 1, .45], cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        axes[1].text(0, .3, 'Held-out SO(3) RMSE [deg], sigma=2.5 deg/s; same 501 score times.\n'
                     'Training residual mean is added and frozen before evaluation.\n'
                     'Same original gyro values at all noises; no synthetic draws.\n'
                     'Only sparse gyro and the initial mocap pose enter the predictor.\n'
                     'Bands condition on the fitted physical point, not unknown future model error.',
                     transform=axes[1].transAxes, fontsize=10, linespacing=1.7)
        fig.suptitle('Identification strength and held-out prediction', fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .94))
        pdf.savefig(fig)
        plt.close(fig)
