"""Figures and a compact movie from the saved sequential-Bayes moments.

This module does no filtering, propagation, covariance fitting or smoothing.
It converts the same saved means/covariances to display units and keeps the
two sides of an acquisition at their original common timestamp.  RPY bands
remain the estimator's declared local Euler-moment display approximation.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation


COLORS = {'white': '#1769aa', 'spike': '#d56b12'}
LABELS = {'white': 'Bayes: white noise', 'spike': 'Bayes: white + spike'}
SPIKE = '#ba252a'
ORDER = ('white', 'spike')


def _settings(acquisitions, cases):
    intervals = np.diff(np.asarray(acquisitions['time'], float))
    rate = float(1./np.median(intervals)) if len(intervals) else None
    regular = bool(len(intervals) and np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-12))
    rate_label = (f'{rate:g} Hz acquired gyro' if regular else
                  f'median acquired rate {rate:g} Hz' if rate is not None else 'One initial gyro reading')
    covariance = cases['white'].get('measurement_covariance_rad2_s2')
    if covariance is None:
        sigma = None
        noise_label = 'assumed Gaussian gyro likelihood'
    else:
        covariance = np.asarray(covariance)
        deviation = np.rad2deg(np.sqrt(np.diag(covariance)))
        isotropic = np.allclose(covariance, np.eye(3)*covariance[0, 0], rtol=1e-10, atol=0.)
        sigma = float(deviation[0]) if isotropic else deviation.tolist()
        noise_label = (f'assumed sigma = {sigma:g} deg/s' if isotropic else
                       'assumed xyz std = ('+', '.join(f'{value:g}' for value in deviation)+') deg/s')
    return dict(rate_hz=rate, regular_acquisition=regular, assumed_sigma_deg_s=sigma,
                rate_label=rate_label, noise_label=noise_label,
                spike_count=int(np.count_nonzero(acquisitions['spike_mask'])))


def _halfwidth(covariance):
    # No max(..., 0), clipping, floor, smoothing or saturation: these are the
    # actual saved diagonal moments, expressed in the plot's physical units.
    return 2.*np.rad2deg(np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)))


def _reference(reference, times, key):
    return np.column_stack([np.interp(times, reference['reference_time'],
        reference[key][:, dimension]) for dimension in range(3)])


def _error(reference, case):
    truth = _reference(reference, case['time'], 'reference_rpy_rad')
    actual = Rotation.from_euler('xyz', truth)
    center = Rotation.from_euler('xyz', case['rpy_proxy_rad'])
    return np.rad2deg((actual.inv()*center).magnitude())


def _innovations(acquisitions, case, name):
    """Observed gyro minus the saved pre-observation predictive rate mean."""
    times = np.asarray(case['time'])
    sides = np.asarray(case.get('event_side', np.zeros(len(times), dtype=int)))
    values, used = [], []
    for stamp, observation in zip(acquisitions['time'][1:], acquisitions['gyro_'+name][1:]):
        matches = np.flatnonzero((times == stamp) & (sides == -1))
        if not len(matches):
            matches = np.flatnonzero(times == stamp)
        if not len(matches):
            raise ValueError('A saved prediction at an acquisition is required for the innovation plot')
        difference = observation - case['omega_mean_rad_s'][matches[0]]
        values.append(np.rad2deg(difference))
        used.append(stamp)
    return np.asarray(used), np.asarray(values).reshape(-1, 3)


def _event_lines(axis, acquisitions, alpha=.45):
    for stamp, spike in zip(acquisitions['time'], acquisitions['spike_mask']):
        axis.axvline(stamp, color=SPIKE if spike else '.6',
                     lw=1.2 if spike else .7, ls=':' if spike else '-',
                     alpha=.65 if spike else alpha, zorder=0)


def _legend():
    return [Line2D([], [], color='black', lw=2., label='Synthetic truth'),
        *[Line2D([], [], color=COLORS[name], lw=1.7, label=LABELS[name]) for name in ORDER],
        Line2D([], [], color='.45', marker='x', ls='', label='Acquired gyro (white)'),
        Line2D([], [], color=SPIKE, marker='x', ls='', label='Contaminated gyro')]


def _comparison(output, reference, acquisitions, cases, settings):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    truth_time = reference['reference_time']
    mask = np.asarray(acquisitions['spike_mask'], bool)
    for dimension in range(3):
        top, bottom = axes[:, dimension]
        top.plot(truth_time, np.rad2deg(reference['reference_rpy_rad'][:, dimension]),
                 color='black', lw=1.9, zorder=4)
        bottom.plot(truth_time, np.rad2deg(reference['reference_omega_body_rad_s'][:, dimension]),
                    color='black', lw=1.9, zorder=4)
        for name in ORDER:
            case, color = cases[name], COLORS[name]
            for axis, value_key, covariance_key in (
                (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                (bottom, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean = np.rad2deg(case[value_key][:, dimension])
                half = _halfwidth(case[covariance_key])[:, dimension]
                axis.fill_between(case['time'], mean-half, mean+half,
                                  color=color, alpha=.17, lw=0, zorder=1)
                axis.plot(case['time'], mean, color=color, lw=1.5, zorder=5)
        bottom.scatter(acquisitions['time'], np.rad2deg(acquisitions['gyro_white'][:, dimension]),
                       color='.45', marker='x', s=36, linewidths=1.3, zorder=7)
        bottom.scatter(acquisitions['time'][mask],
                       np.rad2deg(acquisitions['gyro_spike'][mask, dimension]),
                       color=SPIKE, marker='x', s=62, linewidths=1.8, zorder=8)
        top.set_title(('Roll', 'Pitch', 'Yaw')[dimension], fontsize=14)
        bottom.set_title(r'$\omega_'+ 'xyz'[dimension] + '$', fontsize=14)
        top.set_ylabel('deg', fontsize=12)
        bottom.set_ylabel('deg/s', fontsize=12)
        bottom.set_xlabel('Data time [s]', fontsize=12)
        for axis in (top, bottom):
            _event_lines(axis, acquisitions)
            axis.grid(alpha=.22)
            axis.tick_params(labelsize=10)
            axis.set_xlim(float(truth_time[0]), float(truth_time[-1]))
    figure.suptitle('Sequential Bayesian update / '+settings['rate_label']+' / '+settings['noise_label'],
                   y=.992, fontsize=15)
    figure.legend(handles=_legend(), loc='upper center', bbox_to_anchor=(.5, .954),
                  ncol=3, frameon=False, fontsize=10.5)
    figure.text(.035, .015,
        'Shading: saved +/-2sigma. RPY uses local Euler-moment proxies; no guaranteed 95% interpretation.\n'
        'Red dotted lines: acquired samples randomly selected for contamination in this stress test. '
        'The estimator is not told the spike mask.', fontsize=9.5)
    figure.tight_layout(rect=(0, .065, 1, .88))
    for suffix, kwargs in (('pdf', {}), ('png', {'dpi':160})):
        figure.savefig(output/('attitude_and_gyro.'+suffix), **kwargs)
    plt.close(figure)


def _diagnostics(output, reference, acquisitions, cases):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    errors = {}
    for dimension, axis in enumerate(axes[0]):
        for name in ORDER:
            case = cases[name]
            width = _halfwidth(case['omega_covariance_rad2_s2'])[:, dimension]
            axis.plot(case['time'], width, color=COLORS[name], lw=1.6, label=LABELS[name])
            sides = np.asarray(case.get('event_side', np.zeros(len(width), dtype=int)))
            for side, fill in ((-1, 'none'), (1, COLORS[name])):
                selected = sides == side
                axis.scatter(case['time'][selected], width[selected], s=28,
                    facecolors=fill, edgecolors=COLORS[name], zorder=5)
        axis.set_title(r'Saved rate half-width: $2\sigma_{\omega_'+ 'xyz'[dimension] + '}$')
        axis.set_ylabel('deg/s')
    for name in ORDER:
        case, color = cases[name], COLORS[name]
        errors[name] = _error(reference, case)
        axes[1, 0].plot(case['time'], errors[name], color=color, lw=1.6, label=LABELS[name])
        stamps, innovation = _innovations(acquisitions, case, name)
        axes[1, 1].plot(stamps, np.linalg.norm(innovation, axis=1), 'o', color=color,
                        markersize=6)
        cross = np.asarray(case['rotation_omega_covariance_colmajor'])
        axes[1, 2].plot(case['time'], np.linalg.norm(cross, axis=(1, 2)), color=color, lw=1.6)
    axes[1, 0].set_title('Attitude-center SO(3) error')
    axes[1, 0].set_ylabel('deg')
    axes[1, 1].set_title('Before-update gyro innovation norm')
    axes[1, 1].set_ylabel('deg/s')
    axes[1, 2].set_title(r'Preserved $\|\operatorname{Cov}(\mathrm{vec}R,\omega)\|_F$')
    axes[1, 2].set_ylabel('rad/s')
    for axis in axes.flat:
        _event_lines(axis, acquisitions)
        axis.set_xlabel('Data time [s]')
        axis.set_xlim(float(reference['reference_time'][0]), float(reference['reference_time'][-1]))
        axis.grid(alpha=.23)
    figure.suptitle('What the Gaussian-likelihood Bayes update does with an unmodelled spike', fontsize=15, y=.994)
    figure.legend(handles=_legend()[1:3], loc='upper center', bbox_to_anchor=(.5, .95),
                  ncol=2, frameon=False, fontsize=11)
    figure.text(.04, .015,
        'Hollow circles: before observation. Filled circles: after observation (initial posterior included).\n'
        'A smaller posterior covariance is confidence under the assumed likelihood, not proof that a spiked measurement is correct.',
        fontsize=10)
    figure.tight_layout(rect=(0, .065, 1, .9))
    for suffix, kwargs in (('pdf', {}), ('png', {'dpi':160})):
        figure.savefig(output/('bayes_diagnostics.'+suffix), **kwargs)
    plt.close(figure)
    return errors


def _before_after(output, reference, acquisitions, cases):
    """Expose the old reset and the new posterior update on the same spike."""
    if 'reset_spike' not in cases:
        return []
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    methods = (('reset_spike', '#ba4141', '--', 'Old: independent gyro reset'),
               ('spike', COLORS['spike'], '-', 'New: sequential Bayes'))
    for dimension in range(3):
        for row, (value_key, covariance_key, truth_key) in enumerate((
                ('rpy_proxy_rad', 'rpy_covariance_delta_rad2', 'reference_rpy_rad'),
                ('omega_mean_rad_s', 'omega_covariance_rad2_s2', 'reference_omega_body_rad_s'))):
            axis = axes[row, dimension]
            axis.plot(reference['reference_time'], np.rad2deg(reference[truth_key][:, dimension]),
                      color='black', lw=1.9, label='Synthetic truth', zorder=4)
            for name, color, style, label in methods:
                case = cases[name]
                mean = np.rad2deg(case[value_key][:, dimension])
                half = _halfwidth(case[covariance_key])[:, dimension]
                axis.fill_between(case['time'], mean-half, mean+half,
                                  color=color, alpha=.15, lw=0, zorder=1)
                axis.plot(case['time'], mean, color=color, ls=style, lw=1.5,
                          label=label, zorder=5)
            if row:
                axis.scatter(acquisitions['time'], np.rad2deg(acquisitions['gyro_spike'][:, dimension]),
                             marker='x', color='.45', s=36, zorder=7)
                mask = np.asarray(acquisitions['spike_mask'], bool)
                axis.scatter(acquisitions['time'][mask],
                    np.rad2deg(acquisitions['gyro_spike'][mask, dimension]),
                    marker='x', color=SPIKE, s=62, linewidths=1.8, zorder=8)
                axis.set_xlabel('Data time [s]')
            axis.set_title((('Roll','Pitch','Yaw') if row == 0 else
                            ('omega x','omega y','omega z'))[dimension], fontsize=13)
            axis.set_ylabel('deg' if row == 0 else 'deg/s')
            _event_lines(axis, acquisitions)
            axis.grid(alpha=.23)
            axis.set_xlim(float(reference['reference_time'][0]), float(reference['reference_time'][-1]))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95),
                  ncol=3, frameon=False, fontsize=11)
    figure.suptitle('Same contaminated measurements: old independent reset versus sequential Bayes',
                   fontsize=15, y=.994)
    figure.text(.04, .015,
        'Shading: saved +/-2sigma; RPY bands are local display proxies.\n'
        'Both methods use the same initial pose, J, known torque and acquired gyro. '
        'The old calculation uses its original six-point law.\n'
        'The new calculation uses deterministic tensor quadrature and a cumulative observation likelihood.', fontsize=9.5)
    figure.tight_layout(rect=(0, .065, 1, .9))
    for suffix, kwargs in (('pdf', {}), ('png', {'dpi':160})):
        figure.savefig(output/('before_after.'+suffix), **kwargs)
    plt.close(figure)
    return ['before_after.pdf', 'before_after.png']


def _schedule(acquisitions, lower, upper, fps):
    """Slow data time and pause at the two sides of an instantaneous update."""
    schedule = []
    event_times = list(np.asarray(acquisitions['time'], float)[1:])
    begin = lower
    for finish in event_times+[upper]:
        frames = max(1, round(20.*(finish-begin)/(upper-lower)*fps))
        schedule.extend((float(t), 'prediction') for t in np.linspace(begin, finish, frames, endpoint=False))
        if finish in event_times:
            schedule.extend([(finish, 'prior')]*max(1, round(.6*fps)))
            schedule.extend([(finish, 'posterior')]*max(1, round(.6*fps)))
        begin = finish
    schedule.extend([(upper, 'prediction')]*fps)
    return schedule


def _visible(case, stamp, phase):
    times = np.asarray(case['time'])
    at = times == stamp
    if phase == 'prior':
        side = np.asarray(case.get('event_side', np.zeros(len(times), dtype=int)))
        at &= side != 1
    return (times < stamp) | at


def _band_vertices(time, mean, halfwidth):
    if not len(time):
        return np.empty((0, 2))
    return np.column_stack((np.r_[time, time[::-1]],
        np.r_[mean-halfwidth, (mean+halfwidth)[::-1]]))


def _movie(output, reference, acquisitions, cases, settings, fps=12):
    """960x540 time reveal of the SAME curves; no invented particles or poses."""
    width, height = 960, 540
    figure, axes = plt.subplots(2, 3, figsize=(width/100, height/100), dpi=100)
    figure.subplots_adjust(left=.065, right=.985, bottom=.17, top=.82, wspace=.28, hspace=.40)
    count = settings['spike_count']
    figure.suptitle(f'Sequential Bayes: white noise vs {count} acquired spike'+('s' if count != 1 else ''),
                   fontsize=14, y=.982)
    figure.legend(handles=_legend()[:3]+_legend()[4:], loc='upper center',
                  bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=8)
    figure.text(.05, .058,
        'Bands: saved +/-2sigma. RPY: local moment display approximation. '
        'No spike detection or rejection in the estimator.', fontsize=8.2)
    stamp_text = figure.text(.05, .015, '', fontsize=10, color='#334b65', animated=True)
    phase_text = figure.text(.97, .015, '', fontsize=10, ha='right', animated=True)
    dynamic, layers = [stamp_text, phase_text], []
    source_time = np.asarray(reference['reference_time'])
    lower, upper = float(source_time[0]), float(source_time[-1])
    spike_mask = np.asarray(acquisitions['spike_mask'], bool)
    cursor_lines = []
    for dimension in range(3):
        for row, (value_key, covariance_key, truth_key) in enumerate((
                ('rpy_proxy_rad', 'rpy_covariance_delta_rad2', 'reference_rpy_rad'),
                ('omega_mean_rad_s', 'omega_covariance_rad2_s2', 'reference_omega_body_rad_s'))):
            axis = axes[row, dimension]
            truth = np.rad2deg(reference[truth_key][:, dimension])
            line, = axis.plot([], [], color='black', lw=1.5, animated=True, zorder=4)
            dynamic.append(line)
            layers.append(('truth', line, source_time, truth))
            limits = [truth]
            for name in ORDER:
                case = cases[name]
                mean = np.rad2deg(case[value_key][:, dimension])
                half = _halfwidth(case[covariance_key])[:, dimension]
                limits.extend((mean-half, mean+half))
                band = PolyCollection([], facecolors=COLORS[name], edgecolors='none',
                                      alpha=.18, animated=True, zorder=1)
                axis.add_collection(band)
                line, = axis.plot([], [], color=COLORS[name], lw=1.3, animated=True, zorder=5)
                endpoint, = axis.plot([], [], color=COLORS[name], marker='o', ls='', ms=3.5,
                                       animated=True, zorder=6)
                dynamic.extend((band, line, endpoint))
                layers.append(('case', (band, line, endpoint), case, mean, half))
            if row:
                normal = np.rad2deg(acquisitions['gyro_white'][:, dimension])
                spike = np.rad2deg(acquisitions['gyro_spike'][spike_mask, dimension])
                normal_line, = axis.plot([], [], color='.5', marker='x', ls='', ms=5,
                                          animated=True, zorder=6)
                spike_line, = axis.plot([], [], color=SPIKE, marker='x', ls='', ms=6,
                                         mew=1.5, animated=True, zorder=7)
                dynamic.extend((normal_line, spike_line))
                layers.append(('observations', normal_line, acquisitions['time'], normal))
                layers.append(('observations', spike_line, acquisitions['time'][spike_mask], spike))
                limits.extend((normal, spike))
            low = min(np.min(value) for value in limits if len(value))
            high = max(np.max(value) for value in limits if len(value))
            margin = .08*(high-low) if high != low else 1.
            axis.set_ylim(low-margin, high+margin)
            axis.set_xlim(lower, upper)
            _event_lines(axis, acquisitions, alpha=.3)
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=8)
            axis.set_ylabel('deg' if row == 0 else 'deg/s', fontsize=9)
            axis.set_title((('roll','pitch','yaw') if row == 0 else
                           ('omega x','omega y','omega z'))[dimension], fontsize=10)
            if row:
                axis.set_xlabel('Data time [s]', fontsize=9)
            cursor = axis.axvline(lower, color='.35', lw=.8, alpha=.55, animated=True)
            dynamic.append(cursor)
            cursor_lines.append(cursor)
    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(figure.bbox)
    schedule = _schedule(acquisitions, lower, upper, fps)
    movie = output/'sequential_bayes_spike_comparison.mp4'
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-maxrate', '1400k',
        '-bufsize', '2800k', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(movie)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for stamp, phase in schedule:
            figure.canvas.restore_region(background)
            for layer in layers:
                kind = layer[0]
                if kind in ('truth', 'observations'):
                    _, line, times, value = layer
                    selected = times <= stamp
                    if phase == 'prior' and kind == 'observations':
                        selected &= times < stamp
                    line.set_data(times[selected], value[selected])
                else:
                    _, (band, line, endpoint), case, mean, half = layer
                    selected = _visible(case, stamp, phase)
                    times = case['time'][selected]
                    line.set_data(times, mean[selected])
                    endpoint.set_data(times[-1:], mean[selected][-1:])
                    band.set_verts([_band_vertices(times, mean[selected], half[selected])])
            stamp_text.set_text(f'Data time = {stamp:.3f} s  |  '+settings['rate_label']+' / '+settings['noise_label'])
            phase_text.set_text({'prior':'Before observation', 'posterior':'After Bayes update',
                                 'prediction':'Initial posterior' if stamp == lower else 'Nonlinear prediction'}[phase])
            for cursor in cursor_lines:
                cursor.set_xdata([stamp, stamp])
            for artist in dynamic:
                if artist.get_visible():
                    figure.draw_artist(artist)
            rgb = np.asarray(figure.canvas.buffer_rgba())[...,:3]
            process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        status = process.wait()
        if status:
            raise RuntimeError('ffmpeg failed: '+error)
    except BaseException:
        if process.poll() is None:
            process.kill()
        raise
    finally:
        plt.close(figure)
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
        '-show_format','-of','json',str(movie)], text=True))
    video = next(item for item in info['streams'] if item['codec_type'] == 'video')
    if (video['width'], video['height']) != (960, 540) or movie.stat().st_size >= 50_000_000:
        raise RuntimeError('The encoded movie does not meet the requested dimension/size bound')
    return dict(file=movie.name, bytes=movie.stat().st_size,
        sha256=hashlib.sha256(movie.read_bytes()).hexdigest(), duration_seconds=float(info['format']['duration']),
        width=video['width'], height=video['height'], fps=fps, frames=len(schedule),
        animation='Saved moments revealed in acquisition order; same-time prior and posterior each paused for about 0.6 s; no interpolation of a filtering update',
        event_hold_seconds_per_side=round(.6*fps)/fps,
        narration=False)


def render(output, reference, acquisitions, cases, diagnostics=None):
    """Save all figures and a data-backed MP4, returning display provenance.

    ``cases`` must contain ``white`` and ``spike`` dictionaries with the exact
    persisted estimator arrays; source arrays are not modified.  Additional
    cases may be supplied but are not silently folded into these two laws.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings(acquisitions, cases)
    _comparison(output, reference, acquisitions, cases, settings)
    _diagnostics(output, reference, acquisitions, cases)
    before_after = _before_after(output, reference, acquisitions, cases)
    movie = _movie(output, reference, acquisitions, cases, settings)
    description = dict(schema='grape/sequential-bayes-spike-plot/v1',
        methods=list(ORDER), reference='Synthetic truth; read-only evaluation',
        settings=settings,
        display_operations=['radian-to-degree conversion', '2*sqrt(saved diagonal covariance)',
            'SO(3) geodesic error of saved Euler display center',
            'observed gyro minus saved pre-update rate mean',
            'Frobenius norm of saved full rotation/rate cross covariance'],
        covariance_adjustment=False, smoothing=False, monte_carlo=False,
        duplicated_before_after_timestamps_preserved=True,
        rpy_bands='Saved local Euler moment proxy covariance; not guaranteed 95% coverage',
        movie=movie,
        files=['attitude_and_gyro.pdf','attitude_and_gyro.png','bayes_diagnostics.pdf',
               'bayes_diagnostics.png', movie['file']]+before_after)
    (output/'plotting.json').write_text(json.dumps(description, ensure_ascii=False, indent=2)+'\n')
    return description
