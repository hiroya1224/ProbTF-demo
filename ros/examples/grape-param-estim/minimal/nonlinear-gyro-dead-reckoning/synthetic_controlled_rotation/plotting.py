"""Plots for matched-input synthetic nonlinear/linear angular dynamics.

The linear comparator omits omega x J omega from the angular equation only.
Both methods retain the same quaternion kinematics and known input torque.
RPY centers and bands are the saved Euler moment proxies; plotting performs
no distribution fitting and never changes a propagated covariance.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

COLORS = dict(nonlinear='#1769aa', linear_forced='#ba4141')
LABELS = dict(nonlinear='Nonlinear Euler', linear_forced='Linear angular dynamics')
ALPHAS = dict(nonlinear=.18, linear_forced=.12)
METHODS = ('linear_forced', 'nonlinear')
ANGLE_NAMES = ('Roll', 'Pitch', 'Yaw')


def _style(axis):
    axis.grid(alpha=.25)
    axis.tick_params(labelsize=10)


def _plot_rpy(axis, dimension, curves, elapsed):
    base = curves['nonlinear']
    axis.plot(elapsed, np.rad2deg(base['reference_rpy_rad'][:, dimension]),
              color='black', lw=1.8, label='Synthetic truth', zorder=4)
    for method in METHODS:
        curve = curves[method]
        center = np.rad2deg(curve['rpy_proxy_rad'][:, dimension])
        width = 2*np.rad2deg(np.sqrt(curve['rpy_covariance_delta_rad2'][:, dimension, dimension]))
        axis.fill_between(elapsed, center-width, center+width,
                          color=COLORS[method], alpha=ALPHAS[method], lw=0)
        axis.plot(elapsed, center, color=COLORS[method],
                  lw=1.7 if method == 'nonlinear' else 1.4,
                  ls='-' if method == 'nonlinear' else '--',
                  label=LABELS[method], zorder=5)
    _style(axis)


def render_case(destination, rate, sigma, curves, acquisition_times):
    """Write comparison.pdf (2 pages) and rpy.png from saved plot moments."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = curves['nonlinear']
    elapsed = base['time']-base['time'][0]
    with PdfPages(destination/'comparison.pdf') as pdf:
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for dimension, axis in enumerate(axes):
            for stamp in acquisition_times:
                axis.axvline(stamp-base['time'][0], color='.65', lw=.6, alpha=.4, zorder=0)
            _plot_rpy(axis, dimension, curves, elapsed)
            axis.set_ylabel(ANGLE_NAMES[dimension]+' [deg]', fontsize=11)
        axes[0].legend(fontsize=9, loc='best')
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        figure.suptitle(f'Synthetic smooth rotation / gyro {rate:g} Hz / sigma {sigma:g} deg/s per reading\n'
                       'Same known torque; linear angular dynamics omit only omega x J omega', fontsize=13)
        figure.text(.055, .017, 'Shading: local +/-2sigma Euler moment proxy; no Gaussian coverage guarantee. Gray lines: gyro acquisitions.', fontsize=9)
        figure.tight_layout(rect=(0, .04, 1, .93))
        figure.savefig(destination/'rpy.png', dpi=160)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for dimension, axis in enumerate(axes[:3]):
            for method in METHODS:
                curve = curves[method]
                width = 2*np.rad2deg(np.sqrt(curve['rpy_covariance_delta_rad2'][:, dimension, dimension]))
                axis.plot(elapsed, width, color=COLORS[method],
                          ls='-' if method == 'nonlinear' else '--', label=LABELS[method])
            axis.set_ylabel(ANGLE_NAMES[dimension]+' 2sigma [deg]', fontsize=11)
            _style(axis)
        for method in METHODS:
            axes[-1].plot(elapsed, np.rad2deg(curves[method]['error_geodesic_rad']),
                          color=COLORS[method], ls='-' if method == 'nonlinear' else '--', label=LABELS[method])
        axes[-1].set_ylabel('SO(3) error [deg]', fontsize=11)
        _style(axes[-1])
        axes[0].legend(fontsize=9)
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        figure.suptitle(f'{rate:g} Hz / sigma {sigma:g} deg/s: propagated widths and synthetic-truth error', fontsize=13)
        figure.tight_layout(rect=(0, .02, 1, .95))
        pdf.savefig(figure)
        plt.close(figure)


def render_overview(destination, overview, sigma=2.5):
    """Compare all available acquisition rates on common synthetic truth."""
    destination = Path(destination)
    figure, axes = plt.subplots(len(overview), 3, figsize=(15, 3.5*len(overview)),
                                sharex=True, squeeze=False)
    for row, (rate, curves) in enumerate(overview.items()):
        base = curves['nonlinear']
        elapsed = base['time']-base['time'][0]
        for dimension, axis in enumerate(axes[row]):
            _plot_rpy(axis, dimension, curves, elapsed)
            axis.set_title(f'{rate:g} Hz / '+ANGLE_NAMES[dimension], fontsize=12)
            axis.set_ylabel('deg', fontsize=11)
            if row == len(overview)-1:
                axis.set_xlabel('Time [s]', fontsize=11)
    axes[0, 0].legend(fontsize=9, loc='best')
    figure.suptitle(f'Synthetic smooth rotation / assumed gyro sigma {sigma:g} deg/s\n'
                   'Same known input torque: nonlinear Euler versus omitted gyroscopic term; local 2sigma bands', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94))
    suffix = format(sigma, 'g').replace('.', 'p')
    figure.savefig(destination/f'rate_overview_sigma_{suffix}.pdf')
    figure.savefig(destination/f'rate_overview_sigma_{suffix}.png', dpi=170)
    plt.close(figure)


def render_generated_motion(destination, data):
    """Show the saved prescribed motion, actual noisy gyro and known torque."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    reference_time = np.asarray(data['reference_time'])
    imu_time = np.asarray(data['imu_time'])
    torque_time = np.asarray(data['torque_time'])
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for dimension in range(3):
        angle, gyro, torque = axes[dimension]
        angle.plot(reference_time, np.rad2deg(data['reference_rpy_rad'][:, dimension]),
                   'k', lw=1.6, label='Synthetic truth')
        angle.set_ylabel(ANGLE_NAMES[dimension]+' [deg]', fontsize=11)
        gyro.plot(imu_time, np.rad2deg(data['gyro'][:, dimension]), color='.5',
                  alpha=.65, lw=.8, label='Generated gyro with bias/noise')
        gyro.plot(imu_time, np.rad2deg(data['gyro_true_body_rad_s'][:, dimension]),
                  color='black', lw=1.2, label='True body angular velocity')
        gyro.set_ylabel('xyz'[dimension]+' omega [deg/s]', fontsize=11)
        torque.plot(torque_time, data['torque_body_nm'][:, dimension],
                    color='#1769aa', lw=1.5, label='Known input torque')
        torque.plot(reference_time, data['reference_gyroscopic_torque_body_nm'][:, dimension],
                    color='#8e44ad', ls='--', lw=1.3, label='omega x J omega')
        torque.set_ylabel('xyz'[dimension]+' torque [Nm]', fontsize=11)
        for axis in axes[dimension]:
            _style(axis)
    axes[0, 0].set_title('Prescribed smooth orientation', fontsize=12)
    axes[0, 1].set_title('Generated body gyro', fontsize=12)
    axes[0, 2].set_title('Matched inverse-dynamics input', fontsize=12)
    for axis in axes[0]:
        axis.legend(fontsize=9, loc='best')
    for axis in axes[-1]:
        axis.set_xlabel('Time [s]', fontsize=11)
    figure.suptitle('Synthetic rotation inputs: same inertia and known torque for both methods\n'
                   'Prescribed kinematic motion; not a six-DoF flight simulation', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(destination/'generated_motion.pdf')
    figure.savefig(destination/'generated_motion.png', dpi=170)
    plt.close(figure)
