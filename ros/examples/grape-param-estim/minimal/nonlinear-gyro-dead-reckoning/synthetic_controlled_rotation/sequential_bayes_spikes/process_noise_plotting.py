"""Plot saved 10 Hz joint moments with model angular-rate process noise.

The generator and observation covariance are unchanged. This is reporting
only: no filtering, resampling, covariance adjustment, or smoothing occurs.
"""
from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
from matplotlib.lines import Line2D


HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_process_plot_saved_helpers', HERE/'rate_comparison_plotting.py')
saved=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(saved)
plt=saved.plt
MODELS=saved.MODELS
COLORS=saved.COLORS
LABELS=saved.LABELS
STYLES=saved.STYLES


def _find(metadata, key):
    if not isinstance(metadata, dict):
        return None
    if key in metadata:
        return metadata[key]
    for value in metadata.values():
        result=_find(value, key)
        if result is not None:
            return result
    return None


def _noise(cases, metadata):
    process=None
    for key in ('process_covariance_rad2_s2', 'process_omega_covariance_rad2_s2',
                'process_noise_covariance_rad2_s2'):
        if key in cases['nonlinear']:
            process=np.asarray(cases['nonlinear'][key])
            break
        value=_find(metadata, key)
        if value is not None:
            process=np.asarray(value)
            break
    if process is None:
        sigma=_find(metadata, 'process_sigma_deg_s')
        if sigma is None:
            raise ValueError('Saved process covariance or explicit process_sigma_deg_s metadata is required')
        process=np.diag(np.broadcast_to(np.deg2rad(sigma), (3,))**2)
    if process.shape!=(3, 3):
        raise ValueError('Expected the saved per-interval 3x3 process covariance')
    observed=np.asarray(cases['nonlinear']['measurement_covariance_rad2_s2'])
    return dict(process_covariance_rad2_s2=process.tolist(),
        process_sigma_xyz_deg_s=np.rad2deg(np.sqrt(np.diag(process))).tolist(),
        process_noise_reference_interval_seconds=float(np.asarray(cases['nonlinear'].get(
            'process_noise_reference_interval_seconds', .1)).item()),
        observation_covariance_rad2_s2=observed.tolist(),
        observation_sigma_xyz_deg_s=np.rad2deg(np.sqrt(np.diag(observed))).tolist(),
        process_noise_scope='Model angular-velocity transition assumption; not extra noise added to generated sensor data')


def _noise_label(settings):
    q=', '.join(f'{v:g}' for v in settings['process_sigma_xyz_deg_s'])
    r=', '.join(f'{v:g}' for v in settings['observation_sigma_xyz_deg_s'])
    return f'Model Q omega sigma xyz=({q}) deg/s per {settings["process_noise_reference_interval_seconds"]:g} s; observation R sigma xyz=({r}) deg/s'


def _main(output, source, acquired, sensor, cases, settings):
    fig, axes=plt.subplots(3, 3, figsize=(15.5, 12), sharex=True,
        gridspec_kw={'height_ratios':[1, 1, .65]})
    for j in range(3):
        top, omega, width=axes[:, j]
        top.plot(source['reference_time'], np.rad2deg(source['reference_rpy_rad'][:, j]),
            color='black', lw=1.7, zorder=4)
        omega.plot(sensor['time'], np.rad2deg(sensor['gyro_spike'][:, j]),
            color=saved.RAW, lw=.45, alpha=.85, zorder=1)
        omega.plot(source['reference_time'], np.rad2deg(source['reference_omega_body_rad_s'][:, j]),
            color='black', lw=1.4, zorder=3)
        omega.scatter(acquired['time'], np.rad2deg(acquired['gyro_spike'][:, j]), s=10,
            facecolors='white', edgecolors='.25', linewidths=.6, zorder=2)
        for name in MODELS:
            case=cases[name]
            for axis, mean_key, cov_key in (
                (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                (omega, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean=np.rad2deg(case[mean_key][:, j])
                half=saved._halfwidth(case[cov_key])[:, j]
                axis.fill_between(case['time'], mean-half, mean+half, color=COLORS[name],
                    alpha=.3 if axis is omega else .16, lw=0, zorder=4)
                axis.plot(case['time'], mean, color=COLORS[name], ls=STYLES[name], lw=1.5, zorder=6)
                if axis is omega:
                    for sign in (-1, 1):
                        axis.plot(case['time'], mean+sign*half, color=COLORS[name],
                            ls=STYLES[name], lw=.7, zorder=7)
            half=saved._halfwidth(case['omega_covariance_rad2_s2'])[:, j]
            width.plot(case['time'], half, color=COLORS[name], ls=STYLES[name], lw=1.5)
            for side, fill in ((-1, 'white'), (1, COLORS[name])):
                rows=case['event_side']==side
                width.scatter(case['time'][rows], half[rows], s=6,
                    facecolors=fill, edgecolors=COLORS[name], linewidths=.45, zorder=5)
        top.set_title(('Roll', 'Pitch', 'Yaw')[j], fontsize=13)
        omega.set_title('omega '+'xyz'[j]+' mean +/-2sigma', fontsize=12)
        width.set_title('omega '+'xyz'[j]+': actual 2sigma half-width', fontsize=12)
        for row, axis in enumerate((top, omega, width)):
            saved._decorate(axis, settings['data_range_seconds'], xlabel=row==2)
            axis.set_ylabel('deg' if row==0 else 'deg/s')
    fig.suptitle(f'10 Hz gyro updates with model process noise / two dynamics models', fontsize=15, y=.993)
    fig.legend(handles=saved._legends(rate=settings['nominal_rate_hz']), loc='upper center',
        bbox_to_anchor=(.5, .955), ncol=2, frameon=False, fontsize=10)
    fig.text(.035, .014, _noise_label(settings)+'.\n'
        f'All {settings["native_reading_count"]:,} native raw readings are shown; only {settings["selected_acquisition_count"]} selected readings update each model. '
        'The source gyro, bias, white noise and spikes are unchanged.\n'
        'Process noise is a transition-model assumption, not newly generated sensor noise. Both models use the same readings, J, torque and quaternion kinematics.\n'
        'All saved before/after moments are drawn. Width markers: open=prior, filled=posterior. RPY bands are a local display approximation; no covariance rescaling.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .88))
    return saved._save(fig, output, 'attitude_and_gyro')


def _comparison_data(source, cases, zero_noise_cases):
    raw, metric=saved._comparison_data(source, cases, zero_noise_cases)
    groups={'selected_rate':'process_noise', 'one_hz':'zero_process'}
    arrays={}
    for key, value in raw.items():
        for old, new in groups.items():
            if key.startswith(old+'_'):
                key=new+key[len(old):]
                break
        arrays[key]=value
    metric['results']={new:metric['results'][old] for old, new in groups.items()}
    return arrays, metric


def _compare(output, arrays, cases, zero_noise_cases, settings):
    fig, axes=plt.subplots(3, 3, figsize=(15.5, 12), sharex=True,
        gridspec_kw={'height_ratios':[1, 1, .75]})
    time=arrays['time']
    for j in range(3):
        for row, key in enumerate(('rpy_error_deg', 'omega_error_deg_s')):
            axis=axes[row, j]
            axis.axhline(0, color='black', lw=.7, alpha=.6)
            for name in MODELS:
                for group in ('zero_process', 'process_noise'):
                    axis.plot(time, arrays[group+'_'+name+'_'+key][:, j], color=COLORS[name],
                        ls=STYLES[name] if group=='process_noise' else ':',
                        lw=1.5 if group=='process_noise' else 1.1)
            axis.set_title((('Roll', 'Pitch', 'Yaw') if row==0 else
                ('omega x', 'omega y', 'omega z'))[j]+' error')
            axis.set_ylabel('deg' if row==0 else 'deg/s')
        width=axes[2, j]
        for name in MODELS:
            for group in ('zero_process', 'process_noise'):
                width.plot(time, arrays[group+'_'+name+'_omega_two_sigma_deg_s'][:, j],
                    color=COLORS[name], ls=STYLES[name] if group=='process_noise' else ':',
                    lw=1.5 if group=='process_noise' else 1.1)
        width.set_title('omega '+'xyz'[j]+' actual posterior 2sigma half-width')
        width.set_ylabel('deg/s')
        for row in range(3):
            saved._decorate(axes[row, j], (time[0], time[-1]), xlabel=row==2)
    q=settings['process_sigma_xyz_deg_s']
    q_label=f'{q[0]:g}' if q[0]==q[1]==q[2] else '('+', '.join(f'{v:g}' for v in q)+')'
    handles=[Line2D([], [], color=COLORS[name], ls=STYLES[name] if noise else ':',
        lw=1.7 if noise else 1.1, label=LABELS[name]+(f' / model Q sigma={q_label} deg/s' if noise else ' / Q=0'))
        for name in MODELS for noise in (True, False)]
    fig.suptitle('Same 10 Hz spiky readings: model angular-rate process noise versus Q=0', fontsize=15)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.text(.035, .016, _noise_label(settings)+'.\n'
        f'All four curves are evaluated at exactly {len(time)} common saved timestamps, using the posterior at duplicate timestamps. No estimate interpolation.\n'
        'Top: wrapped RPY error. Middle: body-rate error. Bottom: actual saved posterior omega 2sigma half-width; not a covariance multiplier.\n'
        'Synthetic reference is read-only. Observation covariance and sensor measurements are unchanged; process noise is only a model assumption.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .88))
    return saved._save(fig, output, 'process_noise_comparison')


def _report(output, settings, metrics):
    lines=['# 10 Hz gyro：角速度のモデル過程雑音を入れた比較', '',
        '[姿勢・角速度・共分散](attitude_and_gyro.pdf) / '
        '[Q=0との同時刻比較](process_noise_comparison.pdf)', '',
        f'200 Hzの原測定{settings["native_reading_count"]:,}件を変更せず、'
        f'同じ10 Hzの{settings["selected_acquisition_count"]}件を両モデルに使った。'
        '初回測定は初期化に一度だけ使い、残る観測は各時刻で一度ずつ更新に用いる。', '',
        '実際に使った観測σ xyz [deg/s] = '+str(settings['observation_sigma_xyz_deg_s'])+'、'
        'モデル過程σ xyz [deg/s] = '+str(settings['process_sigma_xyz_deg_s'])+'。'
        f'基準区間は{settings["process_noise_reference_interval_seconds"]:g}秒で、今回の指定は等方σ=3 deg/s、'
        'Q=(3π/180)²I₃ [rad²/s²]である。これは生成した人工真値や生gyroに雑音を追加し直したことを意味しない。', '',
        '各区間の終端で角速度に独立な不定増分を加えるモデルであり、その瞬間に方位を跳ばさない。'
        '変更された角速度は次の区間の回転時間発展に作用する。連続時間の白色トルク拡散とは区別する。', '',
        '## 同一の保存時刻での評価', '',
        f'全{metrics["common_saved_timestamp_count"]}共通時刻の最後の事後行を選び、等しい重みでRMSEを計算する。'
        '推定値は補間・平滑化せず、真値比較は評価に限る。', '',
        '| モデル過程雑音 | 動力学 | 姿勢中心SO(3) RMSE [deg] | RPY RMSE xyz [deg] | ω RMSE xyz [deg/s] | 終端ω 2σ半幅 xyz [deg/s] |',
        '|---|---|---:|---|---|---|']
    for group, label in (('zero_process', 'Q=0'), ('process_noise', 'モデル過程雑音あり')):
        for name in MODELS:
            value=metrics['results'][group][name]
            lines.append(f'| {label} | {LABELS[name]} | {value["so3_proxy_rmse_deg"]:.6f} | '
                +' | '.join(', '.join(f'{v:.6f}' for v in value[key]) for key in
                    ('rpy_rmse_deg', 'omega_rmse_deg_s', 'final_omega_two_sigma_deg_s'))+' |')
    lines.extend(['', '方位は単位四元数、角速度はbody座標の共同状態である。'
        '同じ保存候補から得た全角速度共分散と方位との交差を使い、描画のための分散操作は行わない。'
        'RPY帯は局所表示近似で、保証付き95%領域ではない。', '',
        'この文書と図は保存済みモーメントの比較だけであり、推定の再実行、Monte Carlo、'
        '観測棄却、BFS、共分散floor、平滑化を追加しない。過程雑音を含む推定法の具体的な近似は実行側の記録を参照する。', ''])
    (output/'PROCESS_NOISE_COMPARISON.md').write_text('\n'.join(lines))
    return 'PROCESS_NOISE_COMPARISON.md'


def render(output, source, acquisitions, sensor, cases, zero_noise_cases, metadata=None):
    """Render saved process-noise cases against existing Q=0 10 Hz cases."""
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for models in (cases, zero_noise_cases):
        if set(models)!=set(MODELS):
            raise ValueError('Expected only nonlinear and linear_forced models')
    settings=saved._readings(source, acquisitions, sensor, cases)
    settings.update(_noise(cases, metadata or {}))
    if not settings['selected_readings_match_native_values_times_indices']:
        raise ValueError('Selected observations must match saved native values, times and indices')
    arrays, metrics=_comparison_data(source, cases, zero_noise_cases)
    files=_main(output, source, acquisitions, sensor, cases, settings)
    files+=_compare(output, arrays, cases, zero_noise_cases, settings)
    files.append(_report(output, settings, metrics))
    np.savez_compressed(output/'process_noise_metrics.npz', **arrays)
    (output/'process_noise_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    files+=['process_noise_metrics.npz', 'process_noise_metrics.json']
    record=dict(schema='grape/process-noise-saved-moment-plots/v1',
        renderer_sha256=saved._sha(__file__), helper_sha256=saved._sha(HERE/'rate_comparison_plotting.py'),
        settings=settings, files=files, comparison_groups=['zero_process', 'process_noise'],
        comparison_evaluation=metrics['evaluation'],
        all_native_samples_drawn=True, all_saved_moment_rows_drawn=True,
        averaging=False, downsampling=False, smoothing=False, covariance_adjustment=False,
        no_estimator_rerun=True, generated_sensor_data_unchanged=True,
        reference='Read-only synthetic reference; never returned to estimator', metadata=metadata or {})
    (output/'process_noise_plotting.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--zero-noise', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Existing frequent_publication source/sensor directory')
    args=parser.parse_args()
    cases={name:saved._load(args.output, 'bayes_'+name+'.npz') for name in MODELS}
    baseline={name:saved._load(args.zero_noise, 'bayes_'+name+'.npz') for name in MODELS}
    metadata=json.loads((args.output/'config.json').read_text()) if (args.output/'config.json').exists() else {}
    result=render(args.output, saved._load(args.source, 'source_input.npz'),
        saved._load(args.output, 'acquisitions.npz'), saved._load(args.source, 'sensor_input.npz'),
        cases, baseline, metadata=metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))
