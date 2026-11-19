"""Independent figures from saved selected-rate and 1 Hz Bayesian moments.

This module does not run an estimator, smooth curves, alter covariances, or
discard native measurements. Dense observation event lines are omitted from
the overview; a fixed first-second view shows the actual saved event sides.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation


plt.rcParams['path.simplify'] = False
MODELS = ('nonlinear', 'linear_forced')
COLORS = {'nonlinear': '#1769aa', 'linear_forced': '#ba4141'}
LABELS = {'nonlinear': 'Nonlinear Euler', 'linear_forced': 'No gyroscopic term'}
STYLES = {'nonlinear': '-', 'linear_forced': '--'}
RAW = '#738b88'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _halfwidth(covariance):
    return 2*np.rad2deg(np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)))


def _reference(source, times, key):
    return np.column_stack([np.interp(times, source['reference_time'],
        source[key][:, j]) for j in range(3)])


def _last_rows(case):
    """One causal output per saved timestamp: posterior wins over prior."""
    clock = np.rint(np.asarray(case['time'])*1e9).astype(np.int64)
    stamps, reverse = np.unique(clock[::-1], return_index=True)
    return stamps, len(clock)-1-reverse


def _readings(source, acquired, sensor, cases):
    native_index=np.asarray(sensor['source_index'])
    selected=np.searchsorted(native_index, acquired['source_index'])
    inside=np.all(selected<len(native_index))
    same = (inside and np.array_equal(acquired['time'], sensor['time'][selected])
        and np.array_equal(acquired['source_index'], native_index[selected])
        and np.array_equal(acquired['gyro_spike'], sensor['gyro_spike'][selected]))
    return dict(native_reading_count=len(sensor['time']),
        selected_acquisition_count=len(acquired['time']),
        selected_readings_match_native_values_times_indices=bool(same),
        nominal_rate_hz=float(1/np.median(np.diff(acquired['time']))),
        data_range_seconds=[float(sensor['time'][0]), float(sensor['time'][-1])],
        stored_moment_rows={name: len(cases[name]['time']) for name in MODELS},
        stored_prior_rows={name: int(np.count_nonzero(cases[name]['event_side']==-1))
            for name in MODELS},
        stored_posterior_rows={name: int(np.count_nonzero(cases[name]['event_side']==1))
            for name in MODELS},
        final_recorded_likelihood_factor_count={name:
            int(np.asarray(cases[name]['target_likelihood_factor_count'])[-1])
            if 'target_likelihood_factor_count' in cases[name] else None
            for name in MODELS},
        acquired_spike_readings=int(np.count_nonzero(acquired['spike_mask'])),
        observed_gyro_sha256=hashlib.sha256(np.ascontiguousarray(acquired['gyro_spike']).tobytes()).hexdigest(),
        initialization='First selected reading initializes the initial law once; subsequent selected readings are likelihood updates',
        profile=str(np.asarray(source.get('motion_profile', '')).item()))


def _decorate(axis, limits, *, xlabel=False):
    axis.set_xlim(*limits)
    axis.grid(alpha=.22)
    axis.tick_params(labelsize=9)
    if limits[0]<15<limits[1]:
        axis.axvspan(15, limits[1], color='.85', alpha=.3, zorder=0)
        axis.axvline(15, color='.4', ls=':', lw=.8)
        axis.text((15+limits[1])/2, .98, 'spike-free tail',
            transform=axis.get_xaxis_transform(), ha='center', va='top',
            fontsize=8, color='.35')
    if xlabel:
        axis.set_xlabel('Data time [s]')


def _save(figure, output, stem):
    names = [stem+'.pdf', stem+'.png']
    figure.savefig(output/names[0])
    figure.savefig(output/names[1], dpi=160)
    plt.close(figure)
    return names


def _legends(*, comparison=False, rate=10):
    if comparison:
        return [Line2D([], [], color=COLORS[name], ls=STYLES[name] if full else ':',
            lw=1.8 if full else 1.3,
            label=LABELS[name]+(f' / {rate:g} Hz' if full else ' / 1 Hz'))
            for name in MODELS for full in (True, False)]
    return [Line2D([], [], color='black', lw=1.7, label='Synthetic truth'),
        Line2D([], [], color=RAW, lw=.7, label='Raw gyro: every native sample')]+[
        Line2D([], [], color=COLORS[name], ls=STYLES[name], lw=1.7,
            label=LABELS[name]) for name in MODELS]+[
        Line2D([], [], color='.2', marker='o', mfc='white', ls='', markersize=4,
            label=f'Selected {rate:g} Hz readings')]


def _overview(output, source, acquired, sensor, cases, settings):
    fig, axes=plt.subplots(3, 3, figsize=(15.5, 12), sharex=True,
        gridspec_kw={'height_ratios':[1, 1, .65]})
    limits=settings['data_range_seconds']
    for j in range(3):
        top, middle, width=axes[:, j]
        top.plot(source['reference_time'], np.rad2deg(source['reference_rpy_rad'][:, j]),
            color='black', lw=1.7, zorder=4)
        middle.plot(sensor['time'], np.rad2deg(sensor['gyro_spike'][:, j]),
            color=RAW, lw=.45, alpha=.85, zorder=1)
        middle.plot(source['reference_time'], np.rad2deg(source['reference_omega_body_rad_s'][:, j]),
            color='black', lw=1.4, zorder=3)
        middle.scatter(acquired['time'], np.rad2deg(acquired['gyro_spike'][:, j]),
            s=10, facecolors='white', edgecolors='.25', linewidths=.6, zorder=2)
        for name in MODELS:
            case=cases[name]
            for axis, mean_key, covariance_key in (
                (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                (middle, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean=np.rad2deg(case[mean_key][:, j])
                half=_halfwidth(case[covariance_key])[:, j]
                axis.fill_between(case['time'], mean-half, mean+half,
                    color=COLORS[name], alpha=.30 if axis is middle else .16, lw=0, zorder=4)
                axis.plot(case['time'], mean, color=COLORS[name], ls=STYLES[name], lw=1.5, zorder=6)
                if axis is middle:
                    for sign in (-1, 1):
                        axis.plot(case['time'], mean+sign*half, color=COLORS[name],
                            ls=STYLES[name], lw=.7, zorder=7)
            width.plot(case['time'], _halfwidth(case['omega_covariance_rad2_s2'])[:, j],
                color=COLORS[name], ls=STYLES[name], lw=1.5)
        for row, axis in enumerate((top, middle, width)):
            axis.set_ylabel('deg' if row==0 else 'deg/s')
            _decorate(axis, limits, xlabel=row==2)
        top.set_title(('Roll', 'Pitch', 'Yaw')[j], fontsize=13)
        middle.set_title('omega '+'xyz'[j]+' mean +/-2sigma', fontsize=12)
        width.set_title('omega '+'xyz'[j]+': actual 2sigma half-width', fontsize=12)
    fig.suptitle(f'Flight-scaled synthetic motion: {settings["nominal_rate_hz"]:g} Hz updates / two dynamics models',
        fontsize=15, y=.993)
    fig.legend(handles=_legends(rate=settings['nominal_rate_hz']), loc='upper center', bbox_to_anchor=(.5, .955),
        ncol=2, frameon=False, fontsize=10)
    fig.text(.035, .016,
        f'All {settings["native_reading_count"]:,} native readings are shown; only {settings["selected_acquisition_count"]} selected readings at {settings["nominal_rate_hz"]:g} Hz update the estimator. '
        'Same spiky measurements, J, torque, initial law and quaternion kinematics in both models.\n'
        'Middle row: actual saved omega mean +/-2sigma, with boundaries. Bottom row: the SAME half-width in deg/s; no variance rescaling.\n'
        'All saved moment rows are drawn, including recorded before/after states. Dense event lines are omitted; the fixed first-second figure shows event sides.\n'
        'RPY uses the saved local display approximation, not a guaranteed 95% region. Synthetic truth is read-only; no smoothing or outlier rejection.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .89))
    return _save(fig, output, 'attitude_and_gyro')


def _uncertainty_zoom(output, source, acquired, sensor, cases, settings):
    fig, axes=plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
    limits=(float(sensor['time'][0]), min(float(sensor['time'][-1]), float(sensor['time'][0])+1))
    native=(sensor['time']>=limits[0])&(sensor['time']<=limits[1])
    reference=(source['reference_time']>=limits[0])&(source['reference_time']<=limits[1])
    for j in range(3):
        axes[0, j].plot(sensor['time'][native], np.rad2deg(sensor['gyro_spike'][native, j]),
            color=RAW, lw=.65, zorder=1)
        axes[0, j].plot(source['reference_time'][reference],
            np.rad2deg(source['reference_omega_body_rad_s'][reference, j]), color='black', lw=1.5, zorder=3)
        selected=(acquired['time']>=limits[0])&(acquired['time']<=limits[1])
        axes[0, j].scatter(acquired['time'][selected],
            np.rad2deg(acquired['gyro_spike'][selected, j]), s=18,
            facecolors='white', edgecolors='.25', linewidths=.7, zorder=2)
        for name in MODELS:
            case=cases[name]
            rows=(case['time']>=limits[0])&(case['time']<=limits[1])
            time=case['time'][rows]
            half=_halfwidth(case['omega_covariance_rad2_s2'])[rows, j]
            mean=np.rad2deg(case['omega_mean_rad_s'][rows, j])
            axes[0, j].fill_between(time, mean-half, mean+half,
                color=COLORS[name], alpha=.25, lw=0, zorder=4)
            for curve in (mean-half, mean+half):
                axes[0, j].plot(time, curve, color=COLORS[name], ls=STYLES[name], lw=.8, zorder=5)
            axes[0, j].plot(time, mean, color=COLORS[name], ls=STYLES[name], lw=1.4, zorder=6)
            axes[1, j].plot(time, half, color=COLORS[name], ls=STYLES[name], lw=1.2)
            for side, fill in ((-1, 'white'), (1, COLORS[name])):
                event=case['event_side'][rows]==side
                axes[1, j].scatter(time[event], half[event], s=7, facecolors=fill,
                    edgecolors=COLORS[name], lw=.45, zorder=6)
        axes[0, j].set_title('omega '+'xyz'[j]+' +/-2sigma')
        axes[1, j].set_title('omega '+'xyz'[j]+' 2sigma half-width')
        for row in range(2):
            axes[row, j].set_ylabel('deg/s')
            _decorate(axes[row, j], limits, xlabel=row==1)
    fig.suptitle('Fixed first-second view: saved angular-rate uncertainty before and after updates', fontsize=15)
    fig.legend(handles=_legends(rate=settings['nominal_rate_hz']), loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, frameon=False)
    fig.text(.04, .025,
        'This is a fixed display window, not a change to data or inference. Every native reading inside the window is shown.\n'
        'Width markers show only stored event rows: open = prior, filled = posterior. Means and bands use the same saved full-covariance moments.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .85))
    return _save(fig, output, 'gyro_uncertainty_first_second')


def _comparison_data(source, cases, one_hz_cases):
    records={(group, name): _last_rows(values[name])
        for group, values in (('selected_rate', cases), ('one_hz', one_hz_cases)) for name in MODELS}
    common=None
    for stamps, _ in records.values():
        common=stamps if common is None else np.intersect1d(common, stamps)
    if len(common)<2:
        raise ValueError('Need at least two common saved timestamps to compare rates without interpolating estimates')
    times=common.astype(float)/1e9
    true_rpy=_reference(source, times, 'reference_rpy_rad')
    true_omega=_reference(source, times, 'reference_omega_body_rad_s')
    truth=Rotation.from_euler('xyz', true_rpy)
    arrays={'time':times}
    metrics={}
    for group, values in (('selected_rate', cases), ('one_hz', one_hz_cases)):
        metrics[group]={}
        for name in MODELS:
            stamps, rows=records[group, name]
            selected=rows[np.searchsorted(stamps, common)]
            case=values[name]
            error_rpy=np.rad2deg(np.arctan2(np.sin(case['rpy_proxy_rad'][selected]-true_rpy),
                np.cos(case['rpy_proxy_rad'][selected]-true_rpy)))
            error_omega=np.rad2deg(case['omega_mean_rad_s'][selected]-true_omega)
            so3=np.rad2deg((truth.inv()*Rotation.from_euler('xyz', case['rpy_proxy_rad'][selected])).magnitude())
            prefix=group+'_'+name
            arrays[prefix+'_rpy_error_deg']=error_rpy
            arrays[prefix+'_omega_error_deg_s']=error_omega
            arrays[prefix+'_so3_error_deg']=so3
            arrays[prefix+'_omega_two_sigma_deg_s']=_halfwidth(case['omega_covariance_rad2_s2'][selected])
            metrics[group][name]=dict(so3_proxy_rmse_deg=float(np.sqrt(np.mean(so3**2))),
                so3_proxy_final_deg=float(so3[-1]),
                rpy_rmse_deg=np.sqrt(np.mean(error_rpy**2, axis=0)).tolist(),
                omega_rmse_deg_s=np.sqrt(np.mean(error_omega**2, axis=0)).tolist(),
                final_omega_two_sigma_deg_s=arrays[prefix+'_omega_two_sigma_deg_s'][-1].tolist())
    return arrays, dict(common_saved_timestamp_count=len(times),
        evaluation_range_seconds=[float(times[0]), float(times[-1])],
        evaluation='Equal-weight RMSE on the identical common saved timestamps; last/posterior row at duplicate times. No estimate interpolation.',
        timestamp_matching='Original elapsed seconds rounded to integer ns for equality matching only',
        reference='Synthetic reference interpolated to shared timestamps for read-only evaluation; never estimator input',
        results=metrics)


def _rate_comparison(output, arrays, metrics, settings):
    fig, axes=plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    time=arrays['time']
    for j in range(3):
        for row, key in enumerate(('rpy_error_deg', 'omega_error_deg_s')):
            axis=axes[row, j]
            axis.axhline(0, color='black', lw=.8, alpha=.6)
            for name in MODELS:
                for group in ('one_hz', 'selected_rate'):
                    error=arrays[group+'_'+name+'_'+key][:, j]
                    axis.plot(time, error, color=COLORS[name],
                        ls=STYLES[name] if group=='selected_rate' else ':',
                        lw=1.6 if group=='selected_rate' else 1.1,
                        alpha=1 if group=='selected_rate' else .85)
            axis.set_title((('Roll', 'Pitch', 'Yaw') if row==0 else
                ('omega x', 'omega y', 'omega z'))[j]+' error')
            axis.set_ylabel('deg' if row==0 else 'deg/s')
            _decorate(axis, (time[0], time[-1]), xlabel=row==1)
    fig.suptitle(f'Same spiky sensor realization: 1 Hz versus {settings["nominal_rate_hz"]:g} Hz', fontsize=15)
    fig.legend(handles=_legends(comparison=True, rate=settings['nominal_rate_hz']), loc='upper center',
        bbox_to_anchor=(.5, .955), ncol=2, frameon=False, fontsize=10)
    fig.text(.04, .025,
        f'All four curves use the identical {len(time)} common saved timestamps and the posterior row at each duplicate timestamp.\n'
        'Wrapped RPY differences and body-rate errors are read-only diagnostics. No smoothing, interpolation of estimates, or noise-only counterfactual.\n'
        'The models differ only by omega cross(J omega); the rates differ only in which saved sensor readings enter the observation likelihood.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .86))
    return _save(fig, output, 'rate_comparison_errors')


def _report(output, settings, metrics):
    rate=settings['nominal_rate_hz']
    lines=[f'# {rate:g} Hz観測を使った別実験：1 Hzとの比較', '',
        '[全20秒の姿勢・角速度・2σ](attitude_and_gyro.pdf) / '
        '[最初1秒の観測前後](gyro_uncertainty_first_second.pdf) / '
        '[同時刻での1 Hzとの誤差比較](rate_comparison_errors.pdf)', '',
        f'200 Hzの原gyroは{settings["native_reading_count"]:,}件、そのうち{rate:g} Hzで取得した'
        f'{settings["selected_acquisition_count"]:,}件を推定に使う。選択した原値・原時刻・indexの完全一致：'
        f'{settings["selected_readings_match_native_values_times_indices"]}。'
        '未取得の生測定は表示にだけ使い、推定の尤度へ入れない。', '',
        '非線形Eulerとジャイロ項なしは同じスパイク入り測定を使う。1 Hz版は別結果として保持する。'
        '初回gyroは初期法則の構成に一度だけ用い、以後の観測を尤度として使う。'
        '全体図では全原測定と全保存モーメントを描き、密集する観測時刻の縦線は省いた。', '',
        '終端で記録された尤度因子数：'+', '.join(
            LABELS[name]+' = '+str(settings['final_recorded_likelihood_factor_count'][name])
            for name in MODELS)+'。この数は初期化に一度だけ使った最初の観測を含めない。', '',
        '## 同じ時刻で評価したRMSE', '',
        f'四つの推定列に共通する{metrics["common_saved_timestamp_count"]}時刻を使用する。'
        '同時刻の観測直前・直後が両方ある場合は最後の事後行を選ぶ。'
        '推定値の補間や平滑化は行わない。以下はこの共通時刻に等しい重みを与えたRMSEである。', '',
        '| 観測取得 | モデル | SO(3)姿勢中心RMSE [deg] | RPY RMSE xyz [deg] | ω RMSE xyz [deg/s] |',
        '|---|---|---:|---|---|']
    for group, group_label in (('one_hz', '1 Hz'), ('selected_rate', f'{rate:g} Hz')):
        for name in MODELS:
            item=metrics['results'][group][name]
            lines.append(f'| {group_label} | {LABELS[name]} | {item["so3_proxy_rmse_deg"]:.6f} | '
                +', '.join(f'{v:.6f}' for v in item['rpy_rmse_deg'])+' | '
                +', '.join(f'{v:.6f}' for v in item['omega_rmse_deg_s'])+' |')
    lines.extend(['', 'SO(3)誤差は保存RPY中心から得た回転の相対回転角。'
        'RPYの成分誤差は±πへ折り返した値であり、SO(3)誤差とは異なる。'
        '真値比較は評価専用で、推定・共分散・観測の選択へ戻さない。', '',
        '角速度帯は保存した全3×3共分散の対角から求めた±2σで、方位との全交差も元データに残る。'
        '表示のための分散操作はしない。RPY帯は局所delta-methodによる表示近似で、保証付き95%領域ではない。', '',
        'このモジュールは保存済み数値の描画・比較だけを行う。'
        '推定の再実行、Monte Carlo、外れ値除去、BFS、共分散floor、平滑化を追加しない。', ''])
    name=f'COMPARISON_1HZ_VS_{rate:g}HZ.md'
    (output/name).write_text('\n'.join(lines))
    return name


def render(output, source, acquisitions, sensor, cases, one_hz_cases, metadata=None):
    """Render only saved arrays. `cases` and `one_hz_cases` share model keys."""
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for values in (cases, one_hz_cases):
        if set(values)!=set(MODELS):
            raise ValueError('Expected only nonlinear and linear_forced models')
    settings=_readings(source, acquisitions, sensor, cases)
    if not settings['selected_readings_match_native_values_times_indices']:
        raise ValueError('Each selected acquisition must exactly match its native gyro value, timestamp and source index')
    arrays, metrics=_comparison_data(source, cases, one_hz_cases)
    files=[]
    files+=_overview(output, source, acquisitions, sensor, cases, settings)
    files+=_uncertainty_zoom(output, source, acquisitions, sensor, cases, settings)
    files+=_rate_comparison(output, arrays, metrics, settings)
    files.append(_report(output, settings, metrics))
    np.savez_compressed(output/'rate_comparison_metrics.npz', **arrays)
    (output/'rate_comparison_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    files+=['rate_comparison_metrics.npz', 'rate_comparison_metrics.json']
    record=dict(schema='grape/selected-rate-saved-moment-figures/v1',
        renderer_sha256=_sha(__file__), settings=settings, files=files,
        model_labels=LABELS, all_native_samples_drawn=True, all_saved_moment_rows_drawn=True,
        averaging=False, downsampling=False, smoothing=False, covariance_adjustment=False,
        comparison_evaluation=metrics['evaluation'],
        moment_storage_metadata=metadata or {},
        fixed_zoom_seconds=[float(sensor['time'][0]), float(sensor['time'][0])+1],
        dense_observation_vlines_omitted=True,
        bands='Actual saved +/-2sigma; RPY is a local display approximation, not a guaranteed 95% region',
        no_estimator_rerun=True, no_movie_requested=True)
    (output/'rate_comparison_plotting.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return record


def _load(directory, filename):
    with np.load(directory/filename, allow_pickle=False) as values:
        return {key:values[key].copy() for key in values.files}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Existing selected-rate numerical output; figures are written here')
    parser.add_argument('--one-hz', type=Path, required=True, help='Existing 1 Hz reference output')
    args=parser.parse_args()
    source_directory=(args.output if (args.output/'source_input.npz').is_file() else args.one_hz)
    sensor_directory=(args.output if (args.output/'sensor_input.npz').is_file() else args.one_hz)
    source=_load(source_directory, 'source_input.npz')
    acquisitions=_load(args.output, 'acquisitions.npz')
    sensor=_load(sensor_directory, 'sensor_input.npz')
    cases={name:_load(args.output, 'bayes_'+name+'.npz') for name in MODELS}
    reference={name:_load(args.one_hz, 'bayes_'+name+'.npz') for name in MODELS}
    result=render(args.output, source, acquisitions, sensor, cases, reference)
    print(json.dumps(result, indent=2, ensure_ascii=False))
