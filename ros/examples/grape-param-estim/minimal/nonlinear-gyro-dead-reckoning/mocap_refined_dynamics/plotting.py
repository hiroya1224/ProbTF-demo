"""Figures for the held-out, mocap-refined nonlinear Euler experiment."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

COLORS = dict(refined='#1769aa', nominal='#ba4141')
LABELS = dict(refined='Refined nonlinear Euler', nominal='Nominal-start nonlinear Euler')


def render_case(destination, rate, sigma, curves, acquisition_times):
    destination = Path(destination)
    base = curves['refined']
    elapsed = base['time'] - base['time'][0]
    with PdfPages(destination / 'comparison.pdf') as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
        for dimension, axis in enumerate(axes):
            axis.plot(elapsed, np.rad2deg(base['reference_rpy_rad'][:, dimension]),
                      'k', lw=1.7, label='Held-out mocap reference (initial pose only)')
            for method in ('nominal', 'refined'):
                curve = curves[method]
                center = np.rad2deg(curve['rpy_proxy_rad'][:, dimension])
                width = 2 * np.rad2deg(np.sqrt(curve['rpy_covariance_delta_rad2'][:, dimension, dimension]))
                if method == 'refined':
                    axis.fill_between(elapsed, center-width, center+width, color=COLORS[method], alpha=.2)
                axis.plot(elapsed, center, color=COLORS[method], lw=1.7 if method == 'refined' else 1.1,
                          ls='-' if method == 'refined' else '--', label=LABELS[method])
            for stamp in acquisition_times:
                axis.axvline(stamp-base['time'][0], color='.7', lw=.5, alpha=.5)
            axis.set_ylabel(('Roll', 'Pitch', 'Yaw')[dimension]+' [deg]')
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('Time from common initialization [s]')
        fig.suptitle(f'{rate:g} Hz / angular-rate sigma {sigma:g} deg/s per reading\n'
                     'Mocap-refined model; held-out prediction; gyroscopic term retained', fontsize=12)
        fig.text(.07, .013, 'Euler center f(E[R]) and local +/-2sigma are display approximations; model parameters and past torque mean are fixed.', fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .93))
        fig.savefig(destination/'rpy.png', dpi=160)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(10.5, 9), sharex=True)
        for dimension, axis in enumerate(axes[:3]):
            for method, curve in curves.items():
                width = 2*np.rad2deg(np.sqrt(curve['rpy_covariance_delta_rad2'][:, dimension, dimension]))
                axis.plot(elapsed, width, color=COLORS[method], label=LABELS[method])
            axis.set_ylabel(('Roll', 'Pitch', 'Yaw')[dimension]+' 2sigma [deg]')
            axis.grid(alpha=.25)
        for method, curve in curves.items():
            axes[3].plot(elapsed, np.rad2deg(curve['error_geodesic_rad']), color=COLORS[method], label=LABELS[method])
        axes[3].set_ylabel('SO(3) error [deg]')
        axes[3].grid(alpha=.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{rate:g} Hz / sigma {sigma:g} deg/s: conditional widths and held-out mocap error')
        fig.tight_layout(rect=(0, .02, 1, .95))
        pdf.savefig(fig)
        plt.close(fig)


def render_overview(destination, overview, sigma=2.5):
    fig, axes = plt.subplots(len(overview), 3, figsize=(14, 3.3*len(overview)), sharex=True, squeeze=False)
    for row, (rate, curves) in enumerate(overview.items()):
        base = curves['refined']
        elapsed = base['time']-base['time'][0]
        for dimension, axis in enumerate(axes[row]):
            axis.plot(elapsed, np.rad2deg(base['reference_rpy_rad'][:, dimension]), 'k', lw=1.4, label='Mocap')
            for method in ('nominal', 'refined'):
                curve = curves[method]
                center = np.rad2deg(curve['rpy_proxy_rad'][:, dimension])
                if method == 'refined':
                    width = 2*np.rad2deg(np.sqrt(curve['rpy_covariance_delta_rad2'][:, dimension, dimension]))
                    axis.fill_between(elapsed, center-width, center+width, color=COLORS[method], alpha=.2)
                axis.plot(elapsed, center, color=COLORS[method], lw=1.5 if method == 'refined' else 1,
                          ls='-' if method == 'refined' else '--', label=LABELS[method])
            axis.set_title(f'{rate:g} Hz / '+('Roll', 'Pitch', 'Yaw')[dimension])
            axis.set_ylabel('deg')
            axis.grid(alpha=.25)
            if row == len(overview)-1:
                axis.set_xlabel('Time [s]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Mocap-refined nonlinear dynamics / sigma={sigma:g} deg/s / Euler moment proxy and local 2sigma')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(Path(destination)/'rate_overview_sigma_2p5.pdf')
    fig.savefig(Path(destination)/'rate_overview_sigma_2p5.png', dpi=170)
    plt.close(fig)


def render_dynamics(destination, sections):
    """Each section contains independent kinematic references and model torques."""
    with PdfPages(Path(destination)/'dynamics_fit.pdf') as pdf:
        for name, section in sections.items():
            time = section['time']
            fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
            for dimension in range(3):
                left, right = axes[dimension]
                left.plot(time, section['omega_body_rad_s'][:, dimension], 'k', lw=1.5, label='Mocap SG body omega')
                left.plot(time, section['gyro_reference_rad_s'][:, dimension], color='.5', lw=.7, alpha=.65, label='Native gyro reference')
                right.plot(time, section['alpha_body_rad_s2'][:, dimension], 'k', lw=1.5, label='Mocap SG body alpha')
                for method in ('nominal', 'refined'):
                    right.plot(time, section[method+'_predicted_alpha_rad_s2'][:, dimension], color=COLORS[method],
                               lw=1, label=LABELS[method]+' + past torque mean')
                left.set_ylabel(('x', 'y', 'z')[dimension]+' omega [rad/s]')
                right.set_ylabel(('x', 'y', 'z')[dimension]+' alpha [rad/s2]')
                for axis in (left, right):
                    axis.grid(alpha=.25)
            axes[0, 0].legend(fontsize=8)
            axes[0, 1].legend(fontsize=8)
            for axis in axes[-1]:
                axis.set_xlabel('Original bag clock [s]')
            fig.suptitle(name+': derivative-space check; held-out derivatives never enter prediction')
            fig.tight_layout(rect=(0, 0, 1, .95))
            pdf.savefig(fig)
            fig.savefig(Path(destination)/(name+'_derivatives.png'), dpi=150)
            plt.close(fig)

            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            for dimension, axis in enumerate(axes):
                axis.plot(time, section['refined_required_torque_nm'][:, dimension], 'k', label='J alpha + omega x J omega')
                axis.plot(time, section['refined_applied_torque_nm'][:, dimension], color=COLORS['refined'], label='Actuator + frozen past mean')
                axis.plot(time, section['refined_gyroscopic_torque_nm'][:, dimension], color='#8e44ad', label='omega x J omega')
                axis.set_ylabel(('x', 'y', 'z')[dimension]+' torque [Nm]')
                axis.grid(alpha=.25)
            axes[0].legend(fontsize=8)
            axes[-1].set_xlabel('Original bag clock [s]')
            fig.suptitle(name+': nonlinear Euler torque balance (reference-based diagnostic only)')
            fig.tight_layout(rect=(0, 0, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
