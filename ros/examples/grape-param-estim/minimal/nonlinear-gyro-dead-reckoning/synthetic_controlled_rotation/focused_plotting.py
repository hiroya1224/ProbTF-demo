"""One six-panel attitude/rate figure from the saved candidate moments.

No propagation, fitting, smoothing, or covariance adjustment is performed.
Duplicated angular-rate timestamps retain the two sides of each acquisition.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'nonlinear': '#1769aa', 'linear_forced': '#ba4141'}
LABELS = {'nonlinear': 'Nonlinear Euler',
          'linear_forced': 'Linear angular dynamics'}
METHODS = ('linear_forced', 'nonlinear')


def render_focused(destination, rpy_curves, gyro_curves, acquisitions,
                   rate=.5, sigma=5., angular_rate_scale=1.):
    """Save the requested RPY-over-body-omega 2-by-3 PDF and PNG.

    RPY curves contain time, reference_rpy_rad, rpy_proxy_rad and
    rpy_covariance_delta_rad2. Rate curves contain time, mean_rad_s,
    covariance_rad2_s2 and reference_omega_rad_s. All values are already
    computed from the same inference law; only unit conversion is applied.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    rpy_base, gyro_base = rpy_curves['nonlinear'], gyro_curves['nonlinear']
    observation_times = np.asarray(acquisitions['time'])

    for dimension, name in enumerate(('Roll', 'Pitch', 'Yaw')):
        upper, lower = axes[:, dimension]
        upper.plot(rpy_base['time'], np.rad2deg(rpy_base['reference_rpy_rad'][:, dimension]),
                   color='black', lw=2.1, label='Synthetic truth', zorder=4)
        lower.plot(gyro_base['time'], np.rad2deg(gyro_base['reference_omega_rad_s'][:, dimension]),
                   color='black', lw=2.1, label='Synthetic truth', zorder=4)

        for method in METHODS:
            attitude = rpy_curves[method]
            angular = gyro_curves[method]
            angle_mean = np.rad2deg(attitude['rpy_proxy_rad'][:, dimension])
            angle_halfwidth = 2.*np.rad2deg(np.sqrt(
                attitude['rpy_covariance_delta_rad2'][:, dimension, dimension]))
            rate_mean = np.rad2deg(angular['mean_rad_s'][:, dimension])
            rate_halfwidth = 2.*np.rad2deg(np.sqrt(
                angular['covariance_rad2_s2'][:, dimension, dimension]))
            color, line = COLORS[method], '-' if method == 'nonlinear' else '--'
            alpha = .2 if method == 'nonlinear' else .13
            upper.fill_between(attitude['time'], angle_mean-angle_halfwidth,
                               angle_mean+angle_halfwidth, color=color,
                               alpha=alpha, lw=0, zorder=2)
            lower.fill_between(angular['time'], rate_mean-rate_halfwidth,
                               rate_mean+rate_halfwidth, color=color,
                               alpha=alpha, lw=0, zorder=2)
            upper.plot(attitude['time'], angle_mean, color=color, ls=line,
                       lw=1.5, label=LABELS[method], zorder=5)
            # Keeping duplicate acquisition timestamps draws the actual
            # instantaneous replacement, rather than an interpolated waist.
            lower.plot(angular['time'], rate_mean, color=color, ls=line,
                       lw=1.5, label=LABELS[method], zorder=5)

        lower.scatter(observation_times, np.rad2deg(acquisitions['gyro'][:, dimension]),
                      color='#d68113', marker='x', s=36, linewidths=1.4,
                      label='Acquired gyro', zorder=7)
        upper.set_title(name, fontsize=14)
        lower.set_title(r'$\omega_'+ 'xyz'[dimension] + '$', fontsize=14)
        upper.set_ylabel('deg', fontsize=12)
        lower.set_ylabel('deg/s', fontsize=12)
        lower.set_xlabel('Time [s]', fontsize=12)
        for axis in (upper, lower):
            for stamp in observation_times:
                axis.axvline(stamp, color='.6', alpha=.5, lw=.7, zorder=0)
            axis.grid(alpha=.23)
            axis.tick_params(labelsize=11)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945),
                  ncol=4, fontsize=11, frameon=False)
    motion_label = ('smooth three-frequency motion' if angular_rate_scale == 1. else
                    f'three-frequency source; body rate x {angular_rate_scale:.6g}')
    figure.suptitle(f'{rate:g} Hz gyro / assumed sigma = {sigma:g} deg/s / {motion_label}',
                   fontsize=15, y=.995)
    figure.text(.045, .012,
                'Shading: +/-2sigma; RPY uses local Euler moment proxies. Gray lines: gyro acquisitions. '
                'Rate bands retain both sides of each acquisition.', fontsize=10)
    figure.tight_layout(rect=(0, .045, 1, .9))
    figure.savefig(destination/'attitude_and_gyro.pdf')
    figure.savefig(destination/'attitude_and_gyro.png', dpi=180)
    plt.close(figure)
