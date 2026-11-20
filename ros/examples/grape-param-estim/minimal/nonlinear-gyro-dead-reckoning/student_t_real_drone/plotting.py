"""Real-flight Student-t figures drawn only from saved joint moments.

Mocap is a read-only attitude reference. The gyro comparison uses recorded
measurements, not a truth angular velocity. No state estimation, covariance
adjustment or estimate interpolation is performed by this renderer.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MODELS = ('nonlinear', 'linear_forced')
COLORS = {'nonlinear': '#1769aa', 'linear_forced': '#ba4141'}
STYLES = {'nonlinear': '-', 'linear_forced': '--'}
LABELS = {'nonlinear': 'Student-t / nonlinear Euler',
          'linear_forced': 'Student-t / no gyroscopic term'}
RAW = '#738b88'
plt.rcParams['path.simplify'] = False


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _halfwidth(covariance):
    return 2*np.rad2deg(np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)))


def _save(figure, output, stem):
    names = [stem+'.pdf', stem+'.png']
    figure.savefig(output/names[0])
    figure.savefig(output/names[1], dpi=160)
    plt.close(figure)
    return names


def _decorate(axis, duration, xlabel=False):
    axis.set_xlim(0., duration)
    axis.grid(alpha=.22)
    axis.tick_params(labelsize=9)
    if xlabel:
        axis.set_xlabel('Elapsed time [s]')


def _acquired_gyro(acquired):
    for key in ('gyro', 'gyro_body_rad_s', 'gyro_spike', 'sparse_gyro'):
        if key in acquired:
            return np.asarray(acquired[key], float)
    raise KeyError('Acquisitions must contain the original body gyro values')


def _reference(source, stamps):
    """Use quaternion SLERP, never interpolate reference Euler angles."""
    rotations = Slerp(np.asarray(source['reference_time'], float),
        Rotation.from_quat(source['reference_quaternion_xyzw']))(np.asarray(stamps, float))
    return rotations, rotations.as_euler('xyz')


def _metric_rows(case, stamps):
    """Select saved 501-grid rows; a posterior wins at a repeated clock."""
    clock = np.rint(np.asarray(case['time'])*1e9).astype(np.int64)
    requested = np.rint(np.asarray(stamps)*1e9).astype(np.int64)
    sides = np.asarray(case['event_side'])
    chosen = {}
    for row, stamp in enumerate(clock):
        if int(stamp) not in chosen or sides[row] >= sides[chosen[int(stamp)]]:
            chosen[int(stamp)] = row
    missing = [int(stamp) for stamp in requested if int(stamp) not in chosen]
    if missing:
        raise ValueError('The fixed evaluation clock must be present in saved dense outputs; no estimate interpolation is allowed')
    return np.asarray([chosen[int(stamp)] for stamp in requested], dtype=int)


def _diagnostics(source, cases):
    start, end = np.asarray(source['time'], float)
    metric_time = np.linspace(start, end, 501)
    arrays = {'metric_time': metric_time}
    metrics = {}
    for name in MODELS:
        case = cases[name]
        stamps = np.asarray(case['time'], float)
        reference_rotations, reference_rpy = _reference(source, stamps)
        raw_reference = np.column_stack([np.interp(stamps, source['gyro_time'],
            source['gyro_body_rad_s'][:, axis]) for axis in range(3)])
        difference = np.asarray(case['rpy_proxy_rad'])-reference_rpy
        rpy_error = np.rad2deg(np.arctan2(np.sin(difference), np.cos(difference)))
        omega_error = np.rad2deg(case['omega_mean_rad_s']-raw_reference)
        geodesic = np.rad2deg((reference_rotations.inv()
            *Rotation.from_euler('xyz', case['rpy_proxy_rad'])).magnitude())
        rows = _metric_rows(case, metric_time)
        arrays[name+'_time'] = stamps.copy()
        arrays[name+'_rpy_error_deg'] = rpy_error
        arrays[name+'_omega_vs_raw_gyro_error_deg_s'] = omega_error
        arrays[name+'_so3_proxy_error_deg'] = geodesic
        arrays[name+'_rpy_two_sigma_deg'] = _halfwidth(case['rpy_covariance_delta_rad2'])
        arrays[name+'_omega_two_sigma_deg_s'] = _halfwidth(case['omega_covariance_rad2_s2'])
        arrays[name+'_metric_row'] = rows
        metrics[name] = dict(so3_proxy_rmse_deg=float(np.sqrt(np.mean(geodesic[rows]**2))),
            so3_proxy_final_deg=float(geodesic[rows][-1]),
            rpy_rmse_deg=np.sqrt(np.mean(rpy_error[rows]**2, axis=0)).tolist(),
            omega_vs_raw_gyro_rmse_deg_s=np.sqrt(np.mean(omega_error[rows]**2, axis=0)).tolist(),
            final_omega_two_sigma_deg_s=arrays[name+'_omega_two_sigma_deg_s'][rows][-1].tolist(),
            final_rpy_two_sigma_deg=arrays[name+'_rpy_two_sigma_deg'][rows][-1].tolist())
    return arrays, dict(evaluation_count=501,
        evaluation_range_bag_seconds=[float(start), float(end)],
        evaluation='Same 501 fixed times for every acquisition rate; exact saved grid rows, posterior at duplicate clock; no estimate interpolation',
        timestamp_matching='Rounded integer ns for equality matching only',
        attitude_reference='Recorded mocap quaternion SLERP, read-only evaluation reference',
        omega_reference='Linearly interpolated recorded raw gyro for evaluation only; a noisy measurement, not angular-velocity truth',
        results=metrics)


def _settings(source, acquired, cases, metadata):
    config = dict((metadata or {}).get('config', metadata or {}))
    start, end = np.asarray(source['time'], float)
    process = cases['nonlinear'].get('process_covariance_rad2_s2')
    if process is None:
        process = config.get('process_covariance_rad2_s2')
    if process is None:
        process = np.eye(3)*np.deg2rad(float(config.get('process_sigma_deg_s', 0.)))**2
    process = np.asarray(process, float)
    interval_kind = ('selected' if config.get('acquisition_mode') == 'selected_rate'
        or len(acquired['time']) < len(source['gyro_time']) else 'native')
    return dict(start_bag_seconds=float(start), end_bag_seconds=float(end),
        duration_seconds=float(end-start),
        acquisition_rate_hz=float(config.get('rate_hz', config.get('acquisition_rate_hz',
            1/np.median(np.diff(acquired['time']))))),
        sigma_deg_s=float(config.get('sigma_deg_s', config.get('assumed_sigma_deg_s', 5.))),
        student_df=float(config.get('student_df', 4.)),
        native_reading_count=len(source['gyro_time']),
        acquired_reading_count=len(acquired['time']),
        process_covariance_rad2_s2=process.tolist(),
        process_sigma_xyz_deg_s=np.rad2deg(np.sqrt(np.diag(process))).tolist(),
        process_noise_enabled=bool(np.any(process != 0.)),
        process_noise_interval_kind=interval_kind,
        process_noise_interval_median_seconds=float(np.median(np.diff(acquired['time']))),
        process_noise_scope=f'Independent body angular-velocity increment at each {interval_kind} acquisition endpoint, before its observation; not per RK4 step or sensor noise',
        stored_rows={name: len(cases[name]['time']) for name in MODELS},
        stored_event_rows={name: {str(side): int(np.count_nonzero(cases[name]['event_side']==side))
            for side in (-1, 0, 1)} for name in MODELS},
        initial_law='N(W0;y0,R), initialized from first acquired gyro once',
        observation_distribution='Multivariate Student-t for each acquired j>=1 factor',
        gaussian_observation_comparator_included=False)


def _process_label(settings):
    if not settings['process_noise_enabled']:
        return 'Q=0'
    sigma = settings['process_sigma_xyz_deg_s']
    label = f'{sigma[0]:g}' if sigma[0] == sigma[1] == sigma[2] else '('+', '.join(f'{value:g}' for value in sigma)+')'
    return (f'Q omega sigma={label} deg/s per {settings.get("process_noise_interval_kind", "native")} interval '
        f'(~{settings["process_noise_interval_median_seconds"]:g} s)')


def _main(output, source, acquired, cases, settings):
    start = settings['start_bag_seconds']
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12), sharex=True,
        gridspec_kw={'height_ratios': [1, 1, .65]})
    _, reference_rpy = _reference(source, source['reference_time'])
    sg = 'omega_reference_time' in source and 'omega_reference_body_rad_s' in source
    for axis in range(3):
        top, middle, width = axes[:, axis]
        top.plot(source['reference_time']-start, np.rad2deg(reference_rpy[:, axis]),
            color='black', lw=1.7, zorder=4)
        middle.plot(source['gyro_time']-start, np.rad2deg(source['gyro_body_rad_s'][:, axis]),
            color=RAW, lw=.65, alpha=.85, zorder=1)
        if sg:
            middle.plot(source['omega_reference_time']-start,
                np.rad2deg(source['omega_reference_body_rad_s'][:, axis]),
                color='black', ls=':', lw=1.25, zorder=3)
        middle.scatter(np.asarray(acquired['time'])-start,
            np.rad2deg(_acquired_gyro(acquired)[:, axis]), s=12,
            facecolors='white', edgecolors='.25', linewidths=.65, zorder=2)
        for name in MODELS:
            case = cases[name]
            elapsed = case['time']-start
            for panel, mean_key, covariance_key in (
                (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                (middle, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean = np.rad2deg(case[mean_key][:, axis])
                half = _halfwidth(case[covariance_key])[:, axis]
                panel.fill_between(elapsed, mean-half, mean+half, color=COLORS[name],
                    alpha=.30 if panel is middle else .16, lw=0, zorder=4)
                panel.plot(elapsed, mean, color=COLORS[name], ls=STYLES[name], lw=1.5, zorder=6)
                if panel is middle:
                    for sign in (-1, 1):
                        panel.plot(elapsed, mean+sign*half, color=COLORS[name],
                            ls=STYLES[name], lw=.7, zorder=7)
            half = _halfwidth(case['omega_covariance_rad2_s2'])[:, axis]
            width.plot(elapsed, half, color=COLORS[name], ls=STYLES[name], lw=1.5)
            for side, fill in ((-1, 'white'), (1, COLORS[name])):
                rows = case['event_side']==side
                width.scatter(elapsed[rows], half[rows], s=9,
                    facecolors=fill, edgecolors=COLORS[name], linewidths=.45, zorder=5)
        top.set_title(('Roll', 'Pitch', 'Yaw')[axis], fontsize=13)
        middle.set_title('omega '+'xyz'[axis]+' mean +/-2sigma', fontsize=12)
        width.set_title('omega '+'xyz'[axis]+': actual 2sigma half-width', fontsize=12)
        for row, panel in enumerate((top, middle, width)):
            _decorate(panel, settings['duration_seconds'], xlabel=row==2)
            panel.set_ylabel('deg' if row==0 else 'deg/s')
    handles = [Line2D([], [], color='black', lw=1.7, label='Recorded mocap attitude'),
        Line2D([], [], color=RAW, lw=.8, label='Recorded raw gyro: measurement')]
    if sg:
        handles.append(Line2D([], [], color='black', ls=':', lw=1.2,
            label='Mocap SG angular rate: offline reference'))
    handles.extend(Line2D([], [], color=COLORS[name], ls=STYLES[name], lw=1.7,
        label=LABELS[name]) for name in MODELS)
    reading_label = (f'All {settings["acquired_reading_count"]} native gyro readings'
        if settings['process_noise_enabled'] and settings['acquired_reading_count'] == settings['native_reading_count']
        else f'Selected {settings["acquisition_rate_hz"]:g} Hz readings')
    handles.append(Line2D([], [], color='.2', marker='o', mfc='white', ls='', markersize=4,
        label=reading_label))
    title = (f'Real flight bag #8 / Student-t nu={settings["student_df"]:g} / '
        f'{settings["acquisition_rate_hz"]:g} Hz / marginal sigma={settings["sigma_deg_s"]:g} deg/s')
    if settings['process_noise_enabled']:
        title = (f'Real flight bag #8 / Student-t nu={settings["student_df"]:g} / {reading_label}\n'
            f'R marginal sigma={settings["sigma_deg_s"]:g} deg/s / {_process_label(settings)}')
    fig.suptitle(title, fontsize=15, y=.993)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955),
        ncol=2, frameon=False, fontsize=10)
    fig.text(.035, .012,
        f'Bag time {start:.9f}--{settings["end_bag_seconds"]:.9f} s. Both models use the SAME {settings["acquired_reading_count"]} acquired raw readings, J, known torque and initial attitude; {_process_label(settings)}.\n'
        'Frozen dynamics were fitted on bag time 40--45 s, with the frozen past residual-torque mean. No held-out mocap or unacquired gyro enters the predictor.\n'
        f'Student-t nu={settings["student_df"]:g}; R is marginal covariance, Sigma=(nu-2)/nu R is its scale. Initial N(W0;y0,R) is used once. Only the gyroscopic term differs.\n'
        'Dense curves and the final post-observation tail are propagated candidates, not held measurements. Bands are the SAME saved +/-2sigma; no covariance adjustment.\n'
        'Mocap / optional SG are read-only references; raw gyro is a noisy measurement. Open/filled width markers=prior/posterior. RPY bands are local display approximations.',
        fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .86))
    return _save(fig, output, 'attitude_and_gyro')


def _errors(output, arrays, settings):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    start = settings['start_bag_seconds']
    for axis in range(3):
        for row, error_key, width_key in (
            (0, 'rpy_error_deg', 'rpy_two_sigma_deg'),
            (1, 'omega_vs_raw_gyro_error_deg_s', 'omega_two_sigma_deg_s')):
            panel = axes[row, axis]
            panel.axhline(0., color='black', lw=.8, alpha=.6)
            for name in MODELS:
                elapsed = arrays[name+'_time']-start
                mean = arrays[name+'_'+error_key][:, axis]
                half = arrays[name+'_'+width_key][:, axis]
                panel.fill_between(elapsed, mean-half, mean+half, color=COLORS[name], alpha=.16, lw=0)
                panel.plot(elapsed, mean, color=COLORS[name], ls=STYLES[name], lw=1.5)
            title = ('Roll', 'Pitch', 'Yaw')[axis]+' error vs mocap' if row==0 else 'omega '+'xyz'[axis]+' difference vs raw gyro'
            panel.set_title(title, fontsize=12)
            panel.set_ylabel('deg' if row==0 else 'deg/s')
            _decorate(panel, settings['duration_seconds'], xlabel=row==1)
    handles = [Line2D([], [], color=COLORS[name], ls=STYLES[name], lw=1.7,
        label=LABELS[name]) for name in MODELS]
    title = (f'Real flight Student-t / {settings["acquisition_rate_hz"]:g} Hz / '
        f'marginal sigma={settings["sigma_deg_s"]:g} deg/s')
    if settings['process_noise_enabled']:
        title += '\n'+_process_label(settings)
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=handles, loc='upper center',
        bbox_to_anchor=(.5, .92 if settings['process_noise_enabled'] else .955), ncol=2, frameon=False)
    fig.text(.04, .025,
        'Top: wrapped RPY component error against recorded mocap quaternion SLERP. Bottom: difference against interpolated raw gyro, NOT angular-velocity truth.\n'
        'Reference interpolation is read-only evaluation. Saved estimates are not interpolated; all dense, prior, posterior and terminal-tail moments are plotted.\n'
        'Bands are the SAME saved +/-2sigma as in the main figure; RPY bands are local display approximations, not guaranteed 95% regions.\n'
        f'Both curves use Student-t observations and {_process_label(settings)}. Frozen model, residual torque, source clocks and first-reading initialization are shared.', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .86))
    return _save(fig, output, 'reconstruction_errors')


def render_case(output, source, acquired, cases, metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if set(cases) != set(MODELS):
        raise ValueError('Expected the two Student-t dynamics models only')
    settings = _settings(source, acquired, cases, metadata)
    arrays, metrics = _diagnostics(source, cases)
    files = _main(output, source, acquired, cases, settings)
    files += _errors(output, arrays, settings)
    np.savez_compressed(output/'plotted_data.npz', **arrays)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    record = dict(schema='grape/real-flight-student-t-saved-moment-plots/v1',
        renderer_sha256=_sha(__file__), settings=settings, files=files,
        metrics_evaluation=metrics['evaluation'],
        all_saved_moment_rows_drawn=True, every_native_gyro_drawn=True,
        averaging=False, estimate_interpolation=False, covariance_adjustment=False,
        no_estimator_rerun=True, metadata=metadata or {})
    (output/'plotting.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return metrics
