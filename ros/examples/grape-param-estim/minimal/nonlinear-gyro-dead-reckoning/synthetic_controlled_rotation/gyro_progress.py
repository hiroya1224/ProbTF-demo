#!/usr/bin/env python3
"""Show saved angular-rate means/full covariances on both acquisition sides.

This postprocessor does not integrate dynamics or update a distribution.
It exposes the saved six-point laws, including the otherwise omitted left
limit at every observation, and retains the original inference artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from gyro_progress_data import extract_progress
from plotting import COLORS, LABELS

HERE = Path(__file__).resolve().parent
METHODS = ('linear_forced', 'nonlinear')
HEADER = 'Current acquisition step replaces rate by a fresh Gaussian; it is not Bayesian conditioning.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def label(value):
    return format(value, 'g').replace('.', 'p')


def prepare_curves(publication, case, acquisitions, data):
    curves, metadata = {}, {}
    for method in METHODS:
        source = case/(method+'.npz')
        progress = extract_progress(dict(np.load(source)), acquisitions,
                                    float(json.loads((case/'metrics.json').read_text())['assumed_sigma_deg_s']))
        curve = progress['arrays']
        curve['reference_omega_rad_s'] = np.column_stack([
            np.interp(curve['time'], data['imu_time'], data['gyro_true_body_rad_s'][:, axis]) for axis in range(3)])
        np.savez_compressed(case/(method+'_gyro_progress.npz'), **curve)
        curves[method] = curve
        metadata[method] = dict(metadata=progress['metadata'], events=progress['events'],
                                source_prediction=str(source.relative_to(publication)), source_sha256=sha(source))
    write_json(case/'gyro_progress_events.json', metadata)
    return curves, metadata


def acquisition_lines(axis, acquisitions):
    for stamp in acquisitions['time']:
        axis.axvline(stamp, color='.6', lw=.6, alpha=.5, zorder=0)
    axis.grid(alpha=.22)
    axis.tick_params(labelsize=10)


def plot_axis(axis, dimension, curves, acquisitions, events, kind):
    base = curves['nonlinear']
    reference = base['reference_omega_rad_s'][:, dimension]
    if kind == 'width':
        for method in METHODS:
            curve = curves[method]
            width = 2*np.rad2deg(np.sqrt(curve['covariance_rad2_s2'][:, dimension, dimension]))
            axis.plot(curve['time'], width, color=COLORS[method],
                      ls='-' if method == 'nonlinear' else '--', lw=1.5, label=LABELS[method])
        # Preserve both limits; open = before, filled = after acquisition.
        before = [e for e in events['nonlinear']['events'] if e['before_two_sigma_deg_s'] is not None]
        after = events['nonlinear']['events']
        axis.scatter([e['time_seconds'] for e in before], [e['before_two_sigma_deg_s'][dimension] for e in before],
                     facecolors='white', edgecolors=COLORS['nonlinear'], s=23, zorder=5, label='Before acquisition')
        axis.scatter([e['time_seconds'] for e in after], [e['after_two_sigma_deg_s'][dimension] for e in after],
                     color=COLORS['nonlinear'], s=15, zorder=6, label='After acquisition')
        axis.set_ylabel('2sigma half-width [deg/s]', fontsize=10)
    else:
        axis.plot(base['time'], np.rad2deg(reference) if kind == 'mean' else np.zeros(len(reference)),
                  color='black', lw=1.6, label='Synthetic truth' if kind == 'mean' else 'Zero error')
        for method in METHODS:
            curve = curves[method]
            center = np.rad2deg(curve['mean_rad_s'][:, dimension])
            if kind == 'error': center -= np.rad2deg(curve['reference_omega_rad_s'][:, dimension])
            width = 2*np.rad2deg(np.sqrt(curve['covariance_rad2_s2'][:, dimension, dimension]))
            axis.fill_between(curve['time'], center-width, center+width,
                              color=COLORS[method], alpha=.19 if method == 'nonlinear' else .12, lw=0)
            axis.plot(curve['time'], center, color=COLORS[method],
                      ls='-' if method == 'nonlinear' else '--', lw=1.4, label=LABELS[method])
        observed = np.rad2deg(acquisitions['gyro'][:, dimension])
        if kind == 'error':
            observed -= np.rad2deg(np.interp(acquisitions['time'], base['time'], reference))
        axis.scatter(acquisitions['time'], observed, color='#d68113', marker='x', s=27,
                     linewidths=1.2, label='Acquired gyro', zorder=7)
        axis.set_ylabel('xyz'[dimension]+(' omega [deg/s]' if kind == 'mean' else ' omega error [deg/s]'), fontsize=10)
    acquisition_lines(axis, acquisitions)


def render_case(case, rate, sigma, curves, acquisitions, events):
    with PdfPages(case/'gyro_progress.pdf') as pdf:
        for kind, title in (('mean', 'Body angular-rate mean and +/-2sigma'),
                            ('width', 'Actual uncertainty width before/after acquisitions'),
                            ('error', 'Body angular-rate error and +/-2sigma')):
            figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
            for dimension, axis in enumerate(axes):
                plot_axis(axis, dimension, curves, acquisitions, events, kind)
                if kind == 'width': axis.set_title('xyz'[dimension]+' axis', fontsize=11)
            axes[0].legend(fontsize=9, loc='best')
            axes[-1].set_xlabel('Time [s]', fontsize=11)
            figure.suptitle(f'{rate:g} Hz / assumed sigma {sigma:g} deg/s: {title}', fontsize=13)
            figure.text(.055, .015, HEADER+'\nBoth saved limits are retained at the same timestamp; no interpolation across acquisitions.', fontsize=8)
            figure.tight_layout(rect=(0, .055, 1, .95))
            pdf.savefig(figure)
            if kind in ('mean', 'width'):
                figure.savefig(case/('gyro_progress.png' if kind == 'mean' else 'gyro_width.png'), dpi=160)
            plt.close(figure)


def render_overview(publication, overview, sigma=2.5):
    suffix = label(sigma)
    with PdfPages(publication/f'gyro_progress_sigma_{suffix}.pdf') as pdf:
        for kind in ('mean', 'width'):
            figure, axes = plt.subplots(len(overview), 3, figsize=(15, 3.5*len(overview)), squeeze=False, sharex=True)
            for row, (rate, entry) in enumerate(overview.items()):
                for dimension, axis in enumerate(axes[row]):
                    plot_axis(axis, dimension, entry['curves'], entry['acquisitions'], entry['events'], kind)
                    axis.set_title(f'{rate:g} Hz / '+'xyz'[dimension]+' axis', fontsize=11)
                    if row == len(overview)-1: axis.set_xlabel('Time [s]', fontsize=11)
            axes[0, 0].legend(fontsize=8, loc='best')
            title = 'Body angular-rate mean and +/-2sigma' if kind == 'mean' else 'Actual +/-2sigma half-width: acquisition waists and widenings'
            figure.suptitle(f'{title} / assumed sigma {sigma:g} deg/s', fontsize=13)
            figure.text(.045, .015, HEADER+' Width panels use their actual data ranges.', fontsize=9)
            figure.tight_layout(rect=(0, .04, 1, .95))
            pdf.savefig(figure)
            figure.savefig(publication/f'gyro_{"progress" if kind == "mean" else "width"}_sigma_{suffix}.png', dpi=170)
            plt.close(figure)


def add_index(publication, summary):
    header = '## 角速度の途中経過'
    index = publication/'INDEX.md'
    text = index.read_text().split('\n'+header)[0].rstrip()
    lines = [text, '', header, '', '[平均・±2σと幅の一覧PDF](gyro_progress_sigma_2p5.pdf) / '
             '[幅の一覧PNG](gyro_width_sigma_2p5.png)', '',
             '観測の直前・直後を同じ時刻に両方表示。現在は新しい独立Gaussianへの置換であり、Bayes条件付けではない。'
             '非線形では狭まる場合と広がる場合があり、線形側の角速度分散は一定。', '',
             '| 仮定sigma [deg/s] | '+ ' | '.join(f'{rate:g} Hz' for rate in summary['rates_hz'])+' |',
             '|---:|'+ '|'.join('---' for _ in summary['rates_hz'])+'|']
    for sigma in summary['assumed_sigmas_deg_s']:
        lines.append('| '+format(sigma, 'g')+' | '+' | '.join(
            f'[gyro PDF](rate_{label(rate)}_hz/sigma_{label(sigma)}_deg_s/gyro_progress.pdf)' for rate in summary['rates_hz'])+' |')
    index.write_text('\n'.join(lines)+'\n')


def add_gyro_progress(publication):
    publication = Path(publication).resolve()
    summary = json.loads((publication/'summary.json').read_text())
    data = dict(np.load(publication/'input.npz'))
    # These are the inference artifacts; none are rewritten by this script.
    immutable = [publication/'input.npz', publication/'input.json', publication/'summary.json',
                 publication/'comparison_table.csv']
    for record in summary['results']:
        case = publication/f'rate_{label(record["rate_hz"])}_hz'/f'sigma_{label(record["assumed_sigma_deg_s"])}_deg_s'
        immutable.extend(case/(method+'.npz') for method in METHODS)
    hashes_before = {str(p.relative_to(publication)): sha(p) for p in immutable}
    records, overview = [], {}
    for record in summary['results']:
        rate, sigma = record['rate_hz'], record['assumed_sigma_deg_s']
        case = publication/f'rate_{label(rate)}_hz'/f'sigma_{label(sigma)}_deg_s'
        acquisitions = dict(np.load(case.parent/'acquisitions.npz'))
        curves, events = prepare_curves(publication, case, acquisitions, data)
        print('Plot gyro progress:', rate, 'Hz, sigma', sigma, flush=True)
        render_case(case, rate, sigma, curves, acquisitions, events)
        records.append(dict(rate_hz=rate, sigma_deg_s=sigma, methods=events))
        if sigma == 2.5:
            overview[rate] = dict(curves=curves, events=events, acquisitions=acquisitions)
    if overview: render_overview(publication, overview)
    hashes_after = {str(p.relative_to(publication)): sha(p) for p in immutable}
    postprocessing = dict(schema='grape/synthetic-gyro-progress-display/v1',
        operation='display saved rate law with both sides of each acquisition; no propagation or observation update rerun',
        source_sha256={name: sha(HERE/name) for name in ('gyro_progress.py', 'gyro_progress_data.py')},
        numerical_inputs_sha256=hashes_before,
        numerical_inputs_bitwise_unchanged=hashes_before == hashes_after,
        current_update='fresh independent Gaussian replacement, not conditional Bayesian fusion', cases=records)
    write_json(publication/'gyro_progress_manifest.json', postprocessing)
    add_index(publication, summary)
    manifest_path = publication/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['postprocessing'] = 'gyro_progress_manifest.json records the added display and unchanged original inference artifacts'
    manifest['files'] = [dict(path=str(p.relative_to(publication)), bytes=p.stat().st_size, sha256=sha(p))
                         for p in sorted(publication.rglob('*')) if p.is_file() and p != manifest_path]
    manifest['bytes_excluding_manifest'] = sum(p['bytes'] for p in manifest['files'])
    write_json(manifest_path, manifest)
    print('Saved angular-rate plots; original inference unchanged:', hashes_before == hashes_after, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, default=HERE/'publication/smooth_rotation_5s')
    add_gyro_progress(parser.parse_args().publication)
