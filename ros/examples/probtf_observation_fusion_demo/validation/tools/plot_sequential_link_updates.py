#!/usr/bin/env python3
"""Plot actual per-observation, same-command body-law update magnitudes.

Consumes the summary JSON from benchmark_sequential_inference.py. The native
and Python curves share the original raw acquisitions and executed ledger;
these are observation-update displacements, not physical tracking errors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(summary, metric, link=None):
    origin = summary['events'][0]['stamp']
    times, values = [], []
    for event in summary['events']:
        updates = event['same_time_command_update_displacements']
        updates = [update for update in updates if link is None or update['link'] == link]
        if updates:
            times.append(event['stamp']-origin)
            values.append(max(update[metric] for update in updates))
    return np.asarray(times), np.asarray(values)


def maximum(summary, metric, after_initial=False):
    origin = summary['events'][0]['stamp']
    values = [(update, event) for event in summary['events']
              if not after_initial or event['stamp'] > origin
              for update in event['same_time_command_update_displacements']]
    update, event = max(values, key=lambda pair: pair[0][metric])
    bfs = event.get('bayes_factor_surprise') or {}
    return dict(value=update[metric], link=update['link'], event_id=event['event_id'],
        elapsed_s=event['stamp']-origin,
        posterior_reset_probability=bfs.get('posterior_change_probability'),
        posterior_reset_log_probability=bfs.get('posterior_change_log_probability'))


def main(directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='row')
    fig.subplots_adjust(top=.85, bottom=.15, hspace=.37, wspace=.21)
    figure_metadata = dict(schema='sequential_link_update_plot_v1', cases={},
        script_sha256=digest(Path(__file__)),
        source_data_directory=str(directory.resolve()),
        definition='After transporting the previous candidate through the actual ledger to the new acquisition command and time, compare each link query before and after that one observation.',
        scale='Symmetric logarithmic axes retain zero and show the large initial corrections.',
        limits=['Short recorded raw prefixes only; not a new closed-loop execution.',
                'K evidence, representative selection, IPC and live contention are excluded.',
                'The plotted change is not a physical pose error or an accuracy guarantee.',
                'No update is discarded, clipped or stopped by a plotted threshold.'])
    palette = plt.get_cmap('tab10')
    legend_handles = None
    for row, case in enumerate(('marker', 'imu_marker')):
        native_path = directory/(case+'_native')/'summary.json'
        reference_path = directory/(case+'_python')/'summary.json'
        native = json.loads(native_path.read_text())
        reference = json.loads(reference_path.read_text())
        links = sorted({update['link'] for event in native['events']
                        for update in event['same_time_command_update_displacements']})
        origin = native['events'][0]['stamp']
        native_source = native['source_sha256']
        reference_source = reference['source_sha256']
        measurements = {}
        for column, metric, factor, label, linthresh in (
                (0, 'mode_displacement_m', 1000., 'Modal position change [mm]', .001),
                (1, 'modal_angle_rad', 1., 'Modal orientation change [rad]', .00001)):
            axis = axes[row, column]
            handles = []
            for index, link in enumerate(links):
                t, values = observations(native, metric, link)
                line, = axis.plot(t, factor*values, color=palette(index),
                                  linewidth=.85, alpha=.66, label=link)
                handles.append(line)
            t, values = observations(native, metric)
            maximum_line, = axis.plot(t, factor*values, color='black', linewidth=1.5,
                                      label='native: maximum over links')
            ref_t, ref_values = observations(reference, metric)
            reference_line, = axis.plot(ref_t, factor*ref_values, color='red',
                linestyle='--', linewidth=1.0, label='Python: maximum over links')
            axis.scatter(t[t == 0.], factor*values[t == 0.], s=35, color='black',
                         marker='o', zorder=4)
            axis.set_yscale('symlog', linthresh=linthresh)
            axis.set_ylim(bottom=0.)
            axis.set_xlim(left=-.025, right=max(t)*1.025)
            axis.set_ylabel(label)
            axis.grid(True, which='both', alpha=.2)
            axis.set_xlabel('Original acquisition time from initial command [s]')
            metric_max = maximum(native, metric)
            metric_later_max = maximum(native, metric, after_initial=True)
            measurements[metric] = dict(all_times=metric_max, after_initial_time=metric_later_max)
            axis.text(.985, .97,
                'initial max: %.3f %s\nafter initial time: %.3f %s' % (
                    factor*max(values[t == 0.]), 'mm' if column == 0 else 'rad',
                    factor*metric_later_max['value'], 'mm' if column == 0 else 'rad'),
                ha='right', va='top', transform=axis.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            if legend_handles is None:
                legend_handles = handles+[maximum_line, reference_line]
        axes[row, 0].set_title('%s: %d sensor events in %.3f s' % (
            case, len(native['events'])-1, native['events'][-1]['stamp']-origin), loc='left')
        axes[row, 1].set_title('Same acquisition command; initial corrections included', loc='left')
        figure_metadata['cases'][case] = dict(
            native_summary_sha256=digest(native_path),
            python_summary_sha256=digest(reference_path),
            native_source_unchanged=native['source_unchanged'],
            python_source_unchanged=reference['source_unchanged'],
            source_hashes_match=native_source == reference_source,
            source_sha256=native_source, native_backend=native['backend'],
            original_raw_input_run=native['input_run'],
            acquisition_span_s=native['acquisition_span_s'],
            native_error=native['error'], python_error=reference['error'],
            maxima=measurements)
    fig.suptitle('Sequential body-candidate changes caused by individual observations\n'
                 'Native arithmetic with the original Python reference overlay', fontsize=13)
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               bbox_to_anchor=(.5, .045), frameon=False, fontsize=8)
    fig.text(.5, .012,
        'Symmetric-log scales include zero. These are short raw replays, not physical error or closed-loop success measurements.',
        ha='center', fontsize=9)
    output = directory/'sequential_link_updates.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    figure_metadata['figure_sha256'] = digest(output)
    (directory/'sequential_link_updates_summary.json').write_text(
        json.dumps(figure_metadata, indent=2, allow_nan=False)+'\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', type=Path, required=True)
    main(parser.parse_args().comparison_root)
