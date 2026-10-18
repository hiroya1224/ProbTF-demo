#!/usr/bin/env python3
"""Plot saved, same-command observation corrections from a compact audit.

The input deliberately contains no physical truth or raw image payloads.
Acquisition time is not the later display/adoption time. Finite-iteration
updates processed after the motion cutoff remain in the recorded series.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source, destination):
    record = json.loads(source.read_text())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    figure.subplots_adjust(top=.87, bottom=.18, hspace=.42, wspace=.25)
    for row, (case, data) in enumerate(record['cases'].items()):
        events = data['events']
        for column, (metric, factor, label) in enumerate((
                ('maximum_link_mode_displacement_m', 1000., 'Position correction [mm]'),
                ('maximum_link_modal_angle_rad', 180./3.141592653589793,
                 'Orientation correction [degrees]'))):
            axis = axes[row, column]
            axis.plot([e['elapsed_s'] for e in events],
                      [factor*e[metric] for e in events], linewidth=.8,
                      color='tab:blue', alpha=.75)
            for kind, marker, color in (('image', 'o', 'tab:orange'),
                                        ('imu', '.', 'tab:blue')):
                selected = [e for e in events if e['event_kind'] == kind]
                if not selected:
                    continue
                axis.scatter([e['elapsed_s'] for e in selected],
                             [factor*e[metric] for e in selected],
                             s=12 if kind == 'image' else 7,
                             marker=marker, color=color, label=kind, zorder=3)
            later = [e for e in events if not e['initial_time']]
            maximum = max(later, key=lambda e: e[metric])
            axis.text(.98, .97, 'max after t=0: %.1f at %.2f s' % (
                factor*maximum[metric], maximum['elapsed_s']),
                transform=axis.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            axis.set_title(case + ': maximum over robot links 1–6', loc='left', fontsize=10)
            axis.set_ylabel(label)
            axis.set_xlabel('Original acquisition time [s]')
            axis.set_yscale('symlog', linthresh=.05)
            axis.set_ylim(bottom=0.)
            axis.set_xlim(-1., 92.)
            axis.grid(True, which='both', alpha=.2)
    axes[1, 0].legend(loc='upper right', bbox_to_anchor=(1., .88), frameon=False)
    figure.suptitle('Fixed-prior nonlinear observation updates: large corrections remain', fontsize=13)
    figure.text(.5, .055,
        'Before and after each observation at exactly the same command/time; initial observations included.\n'
        'Includes late updates processed after motion. These are not physical pose errors or display times.',
        ha='center', fontsize=9)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    print(json.dumps(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        figure=str(destination), figure_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    main(args.source, args.destination)
