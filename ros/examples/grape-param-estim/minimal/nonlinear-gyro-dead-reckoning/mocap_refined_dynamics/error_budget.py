#!/usr/bin/env python3
"""Read-only torque/gyroscopic error budget of saved nonlinear predictions.

Mocap derivatives are diagnostic references only.  This script never changes
a parameter, torque profile, prediction, observation clock or candidate law.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from kinematics import prepare_kinematics


def _weights(time):
    difference = np.diff(time)
    return np.r_[difference[0]/2., (difference[:-1]+difference[1:])/2.,
                 difference[-1]/2.] / (time[-1]-time[0])


def _rms(weights, values):
    return np.sqrt(weights @ (values*values))


def _weighted_correlation(weights, first, second):
    x = first-weights @ first
    y = second-weights @ second
    numerator = weights @ (x*y)
    denominator = np.sqrt((weights @ (x*x)) * (weights @ (y*y)))
    return np.divide(numerator, denominator,
                     out=np.full(3, np.nan), where=denominator > 0.)


def analyse_saved(results, mocap_input, sigma_label='2p5'):
    """Decompose mean body-rate error, retaining all six saved trajectories.

    e = E[omega_pred]-omega_SG; g(w)=w cross(Jw).
    e_dot = J^-1(tau_model-J*alpha_SG-g(omega_SG))
            -J^-1(E[g(omega_pred)]-g(omega_SG))
            +(alpha_SG-d/dt omega_SG).

    The last term records finite-window SG derivative inconsistency.  Stored
    output-step trapezoid errors are reported separately; no residual is
    silently attributed to the physical model or the nonlinear term.
    """
    results, mocap_input = Path(results), Path(mocap_input)
    model = json.loads((results/'fit.json').read_text())
    inertia = np.asarray(model['refined']['inertia_body_kg_m2'], float)
    inverse = np.linalg.solve(inertia, np.eye(3))
    reference = dict(np.load(results/'heldout_dynamics_check.npz'))
    profile = dict(np.load(results/'refined_torque_profile.npz'))
    mocap = dict(np.load(mocap_input))
    weights = _weights(reference['time'])
    torque_error = (reference['refined_applied_torque_nm']
                    -reference['refined_required_torque_nm'])
    alpha_error = (reference['refined_predicted_alpha_rad_s2']
                   -reference['alpha_body_rad_s2'])
    inertia_eigenvalues = np.linalg.eigvalsh(inertia)
    fit_arrays = dict(np.load(results/'fit_arrays.npz'))
    training_weights = fit_arrays['training_time_weights']
    training_omega = fit_arrays['training_omega_body_rad_s']
    training_alpha = fit_arrays['training_alpha_body_rad_s2']
    initial_inertia_eigenvalues = np.linalg.eigvalsh(np.asarray(
        model['initial']['inertia_body_kg_m2']))
    training_gyroscopic = np.cross(training_omega, training_omega @ inertia.T)
    training_applied = (fit_arrays['refined_training_actuator_torque']
                        +fit_arrays['refined_residual_mean_body'])
    training_model_alpha_error = np.linalg.solve(
        inertia, (training_applied-training_gyroscopic).T).T-training_alpha
    global_statistics = dict(
        torque_error_sign='model applied minus mocap-SG required, body CoG',
        torque_error_mean_nm=(weights @ torque_error).tolist(),
        torque_error_rms_nm=_rms(weights, torque_error).tolist(),
        alpha_error_mean_rad_s2=(weights @ alpha_error).tolist(),
        alpha_error_rms_rad_s2=_rms(weights, alpha_error).tolist(),
        alpha_model_vs_sg_time_weighted_correlation=_weighted_correlation(
            weights, reference['refined_predicted_alpha_rad_s2'],
            reference['alpha_body_rad_s2']).tolist(),
        torque_equals_inertia_times_alpha_error_max_abs_nm=float(np.max(np.abs(
            torque_error-alpha_error @ inertia.T))),
        torque_least_squares_metric_on_alpha_error=(inertia.T @ inertia).tolist(),
        torque_least_squares_metric_eigenvalues=(inertia_eigenvalues**2).tolist(),
        torque_least_squares_metric_max_min_ratio=float((
            inertia_eigenvalues[-1]/inertia_eigenvalues[0])**2),
        initial_min_inertia_eigenvalue_kg_m2=float(initial_inertia_eigenvalues[0]),
        refined_min_inertia_eigenvalue_kg_m2=float(inertia_eigenvalues[0]),
        initial_vs_refined_min_acceleration_metric_ratio=float((
            initial_inertia_eigenvalues[0]/inertia_eigenvalues[0])**2),
        training_refined_alpha_vector_rmse_rad_s2=float(np.sqrt(training_weights @
            np.sum(training_model_alpha_error**2, axis=1))),
        training_zero_alpha_vector_rmse_rad_s2=float(np.sqrt(training_weights @
            np.sum(training_alpha**2, axis=1))),
        least_squares_interpretation=('Torque residual r_tau=J r_alpha, so the fitted '
            'torque objective weights acceleration errors by J.T J. An improvement '
            'in this training objective is not a guarantee of unbiased held-out '
            'angular acceleration or small integrated orientation error.'))
    numeric = dict(reference_time=reference['time'],
                   reference_omega_rad_s=reference['omega_body_rad_s'],
                   reference_alpha_rad_s2=reference['alpha_body_rad_s2'],
                   applied_torque_nm=reference['refined_applied_torque_nm'],
                   required_torque_nm=reference['refined_required_torque_nm'],
                   model_alpha_at_sg_omega_rad_s2=reference['refined_predicted_alpha_rad_s2'])
    rates = {}
    for rate in ('0p5', '2', '5'):
        prediction_path = results/f'rate_{rate}_hz'/f'sigma_{sigma_label}_deg_s/refined.npz'
        if not prediction_path.exists():
            continue
        prediction = dict(np.load(prediction_path))
        records = []
        block = 0
        while f'block_{block}_time' in prediction:
            prefix = f'block_{block}_'
            time = prediction[prefix+'time']
            nodes = prediction[prefix+'omega']
            mass = prediction[prefix+'weights']
            sg = prepare_kinematics(mocap, time,
                                    window_seconds=float(reference['window_seconds']),
                                    degree=int(reference['degree']))
            omega_sg, alpha_sg = sg['omega_body_rad_s'], sg['alpha_body_rad_s2']
            torque = np.column_stack([np.interp(time, profile['torque_profile_time'],
                profile['torque_applied_body'][:, axis]) for axis in range(3)])
            gyroscopic_sg = np.cross(omega_sg, omega_sg @ inertia.T)
            gyroscopic_pred = np.einsum('s,tsi->ti', mass,
                                         np.cross(nodes, nodes @ inertia.T))
            mean_omega = np.einsum('s,tsi->ti', mass, nodes)
            error = mean_omega-omega_sg
            forcing = (torque-alpha_sg @ inertia.T-gyroscopic_sg) @ inverse.T
            feedback = -(gyroscopic_pred-gyroscopic_sg) @ inverse.T
            forcing_integral = cumulative_trapezoid(forcing, time, axis=0, initial=0.)
            feedback_integral = cumulative_trapezoid(feedback, time, axis=0, initial=0.)
            sg_closure = (cumulative_trapezoid(alpha_sg, time, axis=0, initial=0.)
                          -(omega_sg-omega_sg[0]))
            numerical_closure = (error-error[0]-forcing_integral
                                 -feedback_integral-sg_closure)
            saved = dict(time=time, mean_rate_error_rad_s=error,
                         forcing_acceleration_error_rad_s2=forcing,
                         nonlinear_feedback_error_rad_s2=feedback,
                         integrated_torque_mismatch_rad_s=forcing_integral,
                         integrated_nonlinear_feedback_rad_s=feedback_integral,
                         sg_derivative_closure_rad_s=sg_closure,
                         numerical_integral_closure_rad_s=numerical_closure)
            for key, value in saved.items():
                numeric[f'rate_{rate}_block_{block}_{key}'] = value
            records.append(dict(
                interval_seconds=time[[0, -1]].tolist(),
                initial_gyro_minus_sg_deg_s=np.rad2deg(error[0]).tolist(),
                final_predicted_mean_minus_sg_deg_s=np.rad2deg(error[-1]).tolist(),
                integrated_torque_mismatch_deg_s=np.rad2deg(forcing_integral[-1]).tolist(),
                integrated_nonlinear_feedback_deg_s=np.rad2deg(feedback_integral[-1]).tolist(),
                sg_derivative_closure_deg_s=np.rad2deg(sg_closure[-1]).tolist(),
                numerical_integral_closure_deg_s=np.rad2deg(numerical_closure[-1]).tolist()))
            block += 1
        def vector_rms(field):
            return float(np.sqrt(np.mean(np.sum(np.asarray(
                [record[field] for record in records])**2, axis=1))))
        rates[rate] = dict(block_count=block, blocks=records,
            initial_rate_error_vector_rms_deg_s=vector_rms('initial_gyro_minus_sg_deg_s'),
            final_rate_error_vector_rms_deg_s=vector_rms('final_predicted_mean_minus_sg_deg_s'),
            integrated_torque_mismatch_vector_rms_deg_s=vector_rms('integrated_torque_mismatch_deg_s'),
            integrated_nonlinear_feedback_vector_rms_deg_s=vector_rms('integrated_nonlinear_feedback_deg_s'),
            maximum_endpoint_numerical_closure_deg_s=float(np.max(np.abs([
                record['numerical_integral_closure_deg_s'] for record in records]))))
    metadata = dict(schema='grape/refined-nonlinear-omega-error-budget/v1',
        sigma_deg_s=float(sigma_label.replace('p', '.')),
        definition='e=E[omega_pred]-omega_mocap_SG, body components',
        derivative_identity=('e_dot=J^-1(tau_model-J alpha_SG-g(omega_SG)) '
            '-J^-1(E[g(omega_pred)]-g(omega_SG)) +(alpha_SG-d/dt omega_SG)'),
        gyroscopic_definition='g(w)=w cross(J w)',
        nonlinear_expectation='all six saved angular-rate trajectories and their original weights',
        diagnostic_mocap_window_seconds=float(reference['window_seconds']),
        diagnostic_sg_degree=int(reference['degree']),
        integration='trapezoid on saved output timestamps; closure saved explicitly',
        sg_closure=('A local SG derivative fit does not define a globally exact derivative '
            'of the moving-window fitted omega curve; its integrated discrepancy is separated.'),
        global_statistics=global_statistics, rates=rates,
        missing_saved_conditions=[rate for rate in ('0p5', '2', '5') if rate not in rates],
        missing_condition_policy=('Only existing saved sigma=2.5 predictions are diagnosed. '
            'Missing requested diagnostic artifacts are recorded, not a fit/state rejection '
            'and do not stop any subsequent update or prediction.'),
        limitation=('A diagnostic against mocap SG, not a measured external torque, a proof '
            'of gyro truth, or a fit/prediction input. Model mismatch includes unmodeled '
            'actuation, residual variation and finite SG derivative error.'),
        parameters_or_predictions_modified=False,
        evaluation_reference_fed_back_to_prediction=False)
    return metadata, numeric


def build_drift_report(results, mocap_input=None):
    """Save diagnostics after all nonlinear prediction files are complete."""
    results = Path(results)
    if mocap_input is None:
        mocap_input = HERE.parent/'gyro_mocap_sg_check/publication/bag8_50_55s/input.npz'
    metadata, arrays = analyse_saved(results, mocap_input)
    (results/'drift_budget.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(results/'drift_budget.npz', **arrays)
    render(results, metadata, arrays)
    statistics = metadata['global_statistics']
    def vector(values):
        return '['+', '.join(f'{value:.6g}' for value in values)+']'
    lines = ['# 非線形回転モデルの予測がずれる原因', '',
        '以下は保存済みの非線形予測をmocapのSG微分と比較した読取専用診断です。'
        '評価区間の参照を同定・トルク・予測へ戻していません。', '',
        '## 参照姿勢・角速度が正しくてもモデルの角加速度が一致しない', '',
        '評価区間50〜55秒のbody・重心まわりのトルク誤差（モデル−SG必要量）は、', '',
        '- 時間加重平均 '+vector(statistics['torque_error_mean_nm'])+' Nm',
        '- 時間加重RMS '+vector(statistics['torque_error_rms_nm'])+' Nm', '',
        'です。SGで求めた角速度を運動方程式へそのまま代入しても、角加速度誤差は、', '',
        '- 時間加重平均 '+vector(statistics['alpha_error_mean_rad_s2'])+' rad/s²',
        '- 時間加重RMS '+vector(statistics['alpha_error_rms_rad_s2'])+' rad/s²',
        '- モデル角加速度とSG角加速度の相関 '+vector(
            statistics['alpha_model_vs_sg_time_weighted_correlation']), '',
        'です。初期ジャイロの誤差や分布幅だけでは説明できず、固定したトルク予測の不一致が'
        '次の取得まで積分されます。再取得で角速度候補を置き換えても、既に生じた姿勢誤差は戻りません。', '',
        '## 非線形項とトルクの不一致を分離する', '',
        r'\(g(w)=w\times Jw\)、\(e=\mathbb E[\omega_{\rm pred}]-\omega_{\rm SG}\) とすると、', '',
        r'\[\dot e=J^{-1}(\tau_{\rm model}-J\alpha_{\rm SG}-g(\omega_{\rm SG}))'
        r'-J^{-1}(\mathbb E[g(\omega_{\rm pred})]-g(\omega_{\rm SG}))'
        r'+(\alpha_{\rm SG}-\tfrac{d}{dt}\omega_{\rm SG}).\]', '',
        '第1項が参照軌道上でのトルク予測誤差、第2項が予測角速度が参照から離れたことに伴う'
        '非線形フィードバックです。最後は移動窓SGの角速度曲線と、別に求めた角加速度の積分不整合です。'
        '6本の保存軌道を元の重みで使い、各項を取得区間ごとに積分しました。', '',
        '| 取得Hz | 初期角速度誤差 | 区間末の角速度誤差 | トルク不一致の積分 | 非線形フィードバックの積分 |',
        '|---:|---:|---:|---:|---:|']
    for rate in metadata['rates']:
        item = metadata['rates'][rate]
        fields = ('initial_rate_error_vector_rms_deg_s', 'final_rate_error_vector_rms_deg_s',
                  'integrated_torque_mismatch_vector_rms_deg_s',
                  'integrated_nonlinear_feedback_vector_rms_deg_s')
        lines.append('| '+rate.replace('p', '.')+' | '+' | '.join(
            f'{item[field]:.4f}' for field in fields)+' |')
    lines.extend(['', '各列は区間ごとの終端ベクトル量のRMS、単位はdeg/sです。'
        'トルク不一致が主成分であり、非線形項はその後の誤差の方向と大きさを変えています。'
        '非線形項を除けば正しくなる、という結論ではありません。', '',
        '## トルク最小二乗と軌道誤差は同じ規準ではない', '',
        r'\(r_\tau=Jr_\alpha\) なので、同定したトルク二乗誤差は'
        r'\(r_\alpha^\top J^\top J r_\alpha\) です。', '',
        '今回の '+r'\(J^\top J\)'+' の固有値は '+vector(
            statistics['torque_least_squares_metric_eigenvalues'])+'、最大／最小比は '
        f"{statistics['torque_least_squares_metric_max_min_ratio']:.4f}"+' です。'
        'この同定目的の改善は、未使用区間での角加速度の不偏性や姿勢の長時間積分精度を保証しません。', '',
        'さらに、最小二乗でJそのものも動かしているため、角加速度誤差に対する重みも同時に変わります。'
        '今回、Jの最小固有値は '
        f"{statistics['initial_min_inertia_eigenvalue_kg_m2']:.6g}"+' から '
        f"{statistics['refined_min_inertia_eigenvalue_kg_m2']:.6g}"+' kg m²へ変わり、'
        '最小固有値方向に対応する二乗重みは '
        f"{statistics['initial_vs_refined_min_acceleration_metric_ratio']:.4f}"+' 分の1になっています。'
        'これは同じ角加速度誤差でもトルク二乗誤差を小さくできる方向があることを意味します。'
        '方向自体も回転しているため、固定のbody軸すべてをこの比率で軽くしたという意味ではありません。', '',
        '訓練区間の角加速度ベクトルRMSEを直接評価すると、refine後＋凍結した過去余剰平均は '
        f"{statistics['training_refined_alpha_vector_rmse_rad_s2']:.6g}"+' rad/s²であり、'
        '角加速度を常に零と置いた診断対照の '
        f"{statistics['training_zero_alpha_vector_rmse_rad_s2']:.6g}"+' rad/s²を下回っていません。'
        'トルク目的関数の改善だけを、回転ダイナミクスの予測精度が十分になった証拠とは扱えません。'
        'この零角加速度は原因確認用の数値であり、予測方式の変更や採用ではありません。', '',
        '[トルク・角加速度と誤差分解の図](nonlinear_error_budget.pdf)、'
        '[全区間の数値](drift_budget.json)、[描画数値](drift_budget.npz)を保存しています。', '',
        'これはSGを参照にしたモデル不一致の診断です。外力の真値やジャイロの真値を測ったものではなく、'
        'モデル化されていないアクチュエータ応答・時間変動する余剰トルク・SGの微分誤差を個別に同定したものでもありません。', ''])
    if metadata['missing_saved_conditions']:
        lines.extend(['診断対象sigma=2.5 deg/sの保存結果が存在しない取得Hz: '+
            ', '.join(rate.replace('p', '.') for rate in metadata['missing_saved_conditions'])+
            '。該当する区間別診断だけを未生成として記録しました。', ''])
    (results/'DRIFT_ANALYSIS.md').write_text('\n'.join(lines))
    return metadata


def render(output, metadata, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output/'nonlinear_error_budget.pdf') as pdf:
        figure, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        time = arrays['reference_time']
        for axis in range(3):
            axes[axis, 0].plot(time, arrays['required_torque_nm'][:, axis], 'k', label='Mocap-SG required torque')
            axes[axis, 0].plot(time, arrays['applied_torque_nm'][:, axis], color='#1769aa', label='Refined model + frozen past residual')
            axes[axis, 1].plot(time, arrays['reference_alpha_rad_s2'][:, axis], 'k', label='Mocap SG alpha')
            axes[axis, 1].plot(time, arrays['model_alpha_at_sg_omega_rad_s2'][:, axis], color='#1769aa', label='Model alpha at mocap SG omega')
            axes[axis, 0].set_ylabel('xyz'[axis]+' torque [Nm]')
            axes[axis, 1].set_ylabel('xyz'[axis]+' alpha [rad/s2]')
            for plot in axes[axis]:
                plot.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        for plot in axes[-1]:
            plot.set_xlabel('Original bag time [s]')
        figure.suptitle('Why the refined nonlinear prediction drifts: held-out torque/acceleration mismatch')
        figure.tight_layout(rect=(0, 0, 1, .95))
        pdf.savefig(figure)
        figure.savefig(output/'heldout_torque_error.png', dpi=150)
        plt.close(figure)

        available_rates = list(metadata['rates'])
        if not available_rates:
            return
        figure, axes = plt.subplots(len(available_rates), 3,
                                   figsize=(14, 3.3*len(available_rates)), sharex=True, squeeze=False)
        for row, rate in enumerate(available_rates):
            for block in range(metadata['rates'][rate]['block_count']):
                prefix = f'rate_{rate}_block_{block}_'
                time = arrays[prefix+'time']
                error = np.rad2deg(arrays[prefix+'mean_rate_error_rad_s'])
                forcing = np.rad2deg(arrays[prefix+'integrated_torque_mismatch_rad_s'])
                feedback = np.rad2deg(arrays[prefix+'integrated_nonlinear_feedback_rad_s'])
                sg = np.rad2deg(arrays[prefix+'sg_derivative_closure_rad_s'])
                for axis in range(3):
                    plot = axes[row, axis]
                    first = block == 0
                    plot.plot(time, error[:, axis], 'k', lw=1.4,
                              label='Mean predicted omega error' if first else None)
                    plot.plot(time, forcing[:, axis]+error[0, axis], color='#ba4141', lw=1.2,
                              label='Initial error + torque mismatch integral' if first else None)
                    plot.plot(time, feedback[:, axis], color='#8e44ad', lw=1.,
                              label='Nonlinear feedback integral' if first else None)
                    plot.plot(time, forcing[:, axis]+feedback[:, axis]+sg[:, axis]+error[0, axis],
                              color='#1c9b75', ls=':', lw=1.2,
                              label='Full budget including SG closure' if first else None)
            for axis, plot in enumerate(axes[row]):
                plot.set_title(rate.replace('p', '.')+' Hz / '+'xyz'[axis])
                plot.set_ylabel('Body angular-rate error [deg/s]')
                plot.grid(alpha=.25)
                if row == len(available_rates)-1:
                    plot.set_xlabel('Original bag time [s]')
        axes[0, 0].legend(fontsize=7)
        figure.suptitle('Within-acquisition nonlinear error budget / sigma 2.5 deg/s\n'
                         'Rate is reacquired at each boundary; attitude is not reset')
        figure.tight_layout(rect=(0, 0, 1, .94))
        pdf.savefig(figure)
        figure.savefig(output/'within_block_error_budget.png', dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--mocap-input', type=Path, default=(HERE.parent/
        'gyro_mocap_sg_check/publication/bag8_50_55s/input.npz'))
    arguments = parser.parse_args()
    build_drift_report(arguments.results, arguments.mocap_input)
    print(arguments.results/'DRIFT_ANALYSIS.md')


if __name__ == '__main__':
    main()
