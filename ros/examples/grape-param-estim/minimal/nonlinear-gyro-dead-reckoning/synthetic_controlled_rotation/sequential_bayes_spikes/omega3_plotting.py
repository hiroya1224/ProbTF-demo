"""Saved-moment plots for body omega x3 at unchanged timing, with Q=0.

Only the new source truth is used. This module does not estimate states,
change covariance, smooth measurements, or compare against the old motion.
"""
from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
from scipy.spatial.transform import Rotation


HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_omega3_saved_plot_helpers', HERE/'rate_comparison_plotting.py')
saved=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(saved)
plt=saved.plt
MODELS=saved.MODELS
COLORS=saved.COLORS
LABELS=saved.LABELS
STYLES=saved.STYLES


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
        omega.scatter(acquired['time'], np.rad2deg(acquired['gyro_spike'][:, j]),
            s=10, facecolors='white', edgecolors='.25', linewidths=.6, zorder=2)
        for name in MODELS:
            case=cases[name]
            for axis, mean_key, covariance_key in (
                (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                (omega, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean=np.rad2deg(case[mean_key][:, j])
                half=saved._halfwidth(case[covariance_key])[:, j]
                axis.fill_between(case['time'], mean-half, mean+half, color=COLORS[name],
                    alpha=.30 if axis is omega else .16, lw=0, zorder=4)
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
    fig.suptitle('Body omega x3 / Q=0 / '
        f'spike amplitude x{settings["spike_scale"]:g} / two dynamics models',
        fontsize=15, y=.993)
    fig.legend(handles=saved._legends(rate=settings['nominal_rate_hz']), loc='upper center',
        bbox_to_anchor=(.5, .955), ncol=2, frameon=False, fontsize=10)
    fig.text(.035, .014,
        f'All {settings["native_reading_count"]:,} native gyro values are shown; only {settings["selected_acquisition_count"]} selected readings at {settings["nominal_rate_hz"]:g} Hz update each model. Q=0.\n'
        'Body omega(t) is 3x the previous calibrated profile at the SAME times; the new quaternion and torque are computed consistently. RPY angles are not multiplied by 3.\n'
        f'Spike amplitude x{settings["spike_scale"]:g} relative to the original saved pulses; onset times, rate and widths unchanged. White sigma remains {settings["sensor_white_sigma_deg_s"]:g} deg/s; observation sigma remains {settings["measurement_assumed_sigma_deg_s"]:g} deg/s.\n'
        'The two models use the same new truth, spiky readings, initial law, J and torque. Only omega cross(J omega) is omitted in the red model.\n'
        'Bands are actual saved +/-2sigma; bottom row is the SAME omega half-width. Open/filled markers=prior/posterior. RPY is a local display approximation; no covariance rescaling.',
        fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .88))
    return saved._save(fig, output, 'attitude_and_gyro')


def _diagnostics(source, cases):
    arrays={}
    metrics={}
    common=None
    rows_by_name={}
    for name in MODELS:
        clock, rows=saved._last_rows(cases[name])
        rows_by_name[name]=(clock, rows)
        common=clock if common is None else np.intersect1d(common, clock)
    common_time=common.astype(float)/1e9
    arrays['metric_time']=common_time
    for name in MODELS:
        case=cases[name]
        time=np.asarray(case['time'])
        true_rpy=saved._reference(source, time, 'reference_rpy_rad')
        true_omega=saved._reference(source, time, 'reference_omega_body_rad_s')
        rpy_error=np.rad2deg(np.arctan2(np.sin(case['rpy_proxy_rad']-true_rpy),
            np.cos(case['rpy_proxy_rad']-true_rpy)))
        omega_error=np.rad2deg(case['omega_mean_rad_s']-true_omega)
        so3=np.rad2deg((Rotation.from_euler('xyz', true_rpy).inv()
            *Rotation.from_euler('xyz', case['rpy_proxy_rad'])).magnitude())
        arrays[name+'_time']=time
        arrays[name+'_rpy_error_deg']=rpy_error
        arrays[name+'_omega_error_deg_s']=omega_error
        arrays[name+'_so3_proxy_error_deg']=so3
        arrays[name+'_rpy_two_sigma_deg']=saved._halfwidth(case['rpy_covariance_delta_rad2'])
        arrays[name+'_omega_two_sigma_deg_s']=saved._halfwidth(case['omega_covariance_rad2_s2'])
        clock, rows=rows_by_name[name]
        selected=rows[np.searchsorted(clock, common)]
        arrays[name+'_metric_row']=selected
        metrics[name]=dict(so3_proxy_rmse_deg=float(np.sqrt(np.mean(so3[selected]**2))),
            so3_proxy_final_deg=float(so3[selected][-1]),
            rpy_rmse_deg=np.sqrt(np.mean(rpy_error[selected]**2, axis=0)).tolist(),
            omega_rmse_deg_s=np.sqrt(np.mean(omega_error[selected]**2, axis=0)).tolist(),
            final_omega_two_sigma_deg_s=arrays[name+'_omega_two_sigma_deg_s'][selected][-1].tolist())
    return arrays, dict(evaluation_count=len(common_time),
        evaluation_range_seconds=[float(common_time[0]), float(common_time[-1])],
        evaluation='Equal-weight errors on identical saved timestamps; last/posterior row at duplicate times. No estimate interpolation.',
        timestamp_matching='Elapsed seconds rounded to integer ns for equality matching only',
        reference='NEW omega-x3 synthetic truth only; read-only diagnostic, never fed to either estimator',
        results=metrics)


def _errors(output, arrays, settings):
    fig, axes=plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    for j in range(3):
        for row, error_key, width_key in (
            (0, 'rpy_error_deg', 'rpy_two_sigma_deg'),
            (1, 'omega_error_deg_s', 'omega_two_sigma_deg_s')):
            axis=axes[row, j]
            axis.axhline(0, color='black', lw=.8, alpha=.6)
            for name in MODELS:
                time=arrays[name+'_time']
                mean=arrays[name+'_'+error_key][:, j]
                half=arrays[name+'_'+width_key][:, j]
                axis.fill_between(time, mean-half, mean+half,
                    color=COLORS[name], alpha=.16, lw=0)
                axis.plot(time, mean, color=COLORS[name], ls=STYLES[name], lw=1.5)
            axis.set_title((('Roll', 'Pitch', 'Yaw') if row==0 else
                ('omega x', 'omega y', 'omega z'))[j]+' error +/-2sigma')
            axis.set_ylabel('deg' if row==0 else 'deg/s')
            saved._decorate(axis, settings['data_range_seconds'], xlabel=row==1)
    handles=[saved.Line2D([], [], color=COLORS[name], ls=STYLES[name], lw=1.7,
        label=LABELS[name]) for name in MODELS]
    fig.suptitle('Body omega x3 / Q=0 / '
        f'spike amplitude x{settings["spike_scale"]:g}: reconstruction errors', fontsize=15)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.text(.04, .025,
        'Top: wrapped component RPY error. Bottom: body angular-rate error. Both models use the same new source truth and same selected noisy measurements.\n'
        f'Spike amplitude x{settings["spike_scale"]:g}; onset times, rate and widths unchanged. White sigma remains {settings["sensor_white_sigma_deg_s"]:g} deg/s; observation sigma remains {settings["measurement_assumed_sigma_deg_s"]:g} deg/s.\n'
        'Bands use the SAME saved +/-2sigma as the main figure, with every saved prior/posterior row; no estimate smoothing or covariance adjustment.\n'
        'Reference subtraction is evaluation only. RPY bands are local display approximations, not guaranteed 95% regions. Q=0; observation sigma remains 5 deg/s.',
        fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .86))
    return saved._save(fig, output, 'reconstruction_errors')


def render(output, source, acquired, sensor, cases, metadata=None):
    """Write figures and metrics from the new saved omega-x3 experiment only."""
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if set(cases)!=set(MODELS):
        raise ValueError('Expected nonlinear and linear_forced cases only')
    settings=saved._readings(source, acquired, sensor, cases)
    if not settings['selected_readings_match_native_values_times_indices']:
        raise ValueError('Selected readings must exactly match new native source values, times and indices')
    settings['model_process_covariance']='Q=0'
    settings['body_rate_multiplier']=3
    settings['temporal_frequency_change']=False
    if metadata is None and (output/'config.json').exists():
        metadata={'config':json.loads((output/'config.json').read_text())}
    config=(metadata or {}).get('config', {})
    settings['spike_scale']=float(config.get('spike_scale', 1.))
    settings['spike_scale_reference']='Amplitude of original saved pulses; original onset times, rate and widths'
    settings['sensor_white_sigma_deg_s']=float(np.rad2deg(np.sqrt(
        np.trace(source['generator_gyro_covariance_rad2_s2'])/3.)))
    settings['measurement_assumed_sigma_deg_s']=float(config.get('assumed_sigma_deg_s', 5.))
    arrays, metrics=_diagnostics(source, cases)
    files=_main(output, source, acquired, sensor, cases, settings)
    files+=_errors(output, arrays, settings)
    np.savez_compressed(output/'omega3_plotted_data.npz', **arrays)
    (output/'omega3_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    files+=['omega3_plotted_data.npz', 'omega3_metrics.json']
    record=dict(schema='grape/omega3-q0-saved-moment-plots/v1',
        renderer_sha256=saved._sha(__file__), helper_sha256=saved._sha(HERE/'rate_comparison_plotting.py'),
        settings=settings, files=files, metrics_evaluation=metrics['evaluation'],
        all_native_samples_drawn=True, all_saved_moment_rows_drawn=True,
        averaging=False, downsampling=False, smoothing=False, covariance_adjustment=False,
        no_estimator_rerun=True, reference='New omega-x3 truth only; not earlier motion', metadata=metadata or {})
    (output/'omega3_plotting.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    result=render(args.output, saved._load(args.output, 'source_input.npz'),
        saved._load(args.output, 'acquisitions.npz'), saved._load(args.output, 'sensor_input.npz'),
        {name:saved._load(args.output, 'bayes_'+name+'.npz') for name in MODELS})
    print(json.dumps(result, indent=2, ensure_ascii=False))
