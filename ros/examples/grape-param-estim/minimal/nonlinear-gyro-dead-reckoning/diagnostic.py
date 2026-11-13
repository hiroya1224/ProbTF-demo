#!/usr/bin/env python3
"""Read-only check of nominal torque against the held-out dense gyro.

The dense evaluation gyro is used here only AFTER prediction, to explain
discrepancy.  This is not a physical torque measurement and is not a source
of a corrective wrench for the sparse-gyro predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from torque import build_torque_profile, local_polynomial_signal, _time_weights

HERE = Path(__file__).resolve().parent
OUTPUT = Path('/tmp/nonlinear-gyro-dead-reckoning-20261113')


def build_diagnostic(data, control):
    """Compare torques at the observed, held-out angular-rate trajectory."""
    time = np.asarray(data['dense_imu_time'], float)
    gyro = np.asarray(data['dense_gyro'], float)
    signal = local_polynomial_signal(time, gyro,
        np.asarray(data['gyro_noise_covariance'], float),
        window_seconds=.08, degree=3, max_derivative=1)
    inertia = np.asarray(control['inertia'], float)
    omega = signal['smooth']
    alpha = signal['derivative']
    gyro_term = np.cross(omega, omega @ inertia.T)
    implied = alpha @ inertia.T + gyro_term
    applied = control['torque'](time)
    actuator = control['actuator_torque'](time)
    mismatch = applied - implied
    alpha_mismatch = np.linalg.solve(inertia, mismatch.T).T
    integrated = np.zeros_like(alpha_mismatch)
    integrated[1:] = np.cumsum(.5 * np.diff(time)[:, None]
        * (alpha_mismatch[:-1] + alpha_mismatch[1:]), axis=0)
    weights = _time_weights(time)
    arrays = dict(time=time, elapsed=time-time[0], dense_gyro_raw=gyro,
        dense_gyro_smooth=omega, dense_gyro_derivative=alpha,
        nominal_inertia_body=inertia,
        nominal_gyro_coupling_torque=gyro_term,
        heldout_gyro_implied_nominal_torque=implied,
        command_actuator_torque=actuator, applied_prediction_torque=applied,
        applied_minus_heldout_implied_torque=mismatch,
        acceleration_mismatch_at_observed_omega=alpha_mismatch,
        integrated_acceleration_mismatch=integrated,
        time_weights=weights)
    metadata = dict(schema='grape/nonlinear-gyro-read-only-input-model-diagnostic/v1',
        used_by_predictor=False, heldout_dense_gyro_used=True,
        nominal_torque_equation='J*domega/dt + omega cross (J*omega)',
        physical_truth_claimed=False,
        interpretation='gyro-implied torque under the same fixed nominal inertia and raw-gyro approximation; not measured true external torque',
        gyro_model='raw gyro treated as physical omega, no sensor-lag inversion',
        derivative='existing irregular-clock local polynomial .08s degree3; offline within held-out evaluation data only',
        prediction_control_model_changed=False, residual_recalibrated=False,
        feedback_to_prediction=False,
        original_time_seconds=[float(time[0]), float(time[-1])],
        sample_count=len(time),
        units=dict(torque='N m, body frame about nominal CoG',
                   angular_velocity='rad/s', angular_acceleration='rad/s^2'),
        mean_applied_torque_nm=(weights @ applied).tolist(),
        mean_implied_torque_nm=(weights @ implied).tolist(),
        mean_torque_mismatch_nm=(weights @ mismatch).tolist(),
        rms_torque_mismatch_nm=np.sqrt(weights @ (mismatch**2)).tolist(),
        mean_acceleration_mismatch_rad_s2=(weights @ alpha_mismatch).tolist(),
        integrated_acceleration_mismatch_endpoint_rad_s=integrated[-1].tolist(),
        integrated_mismatch_interpretation='integral of J^-1 torque mismatch evaluated along observed omega; not a separate free nonlinear omega trajectory or a control correction',
        sparse_observation_input_not_modified=True,
        torque_provenance=control['metadata'],
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(arrays=arrays, metadata=metadata)


def save_diagnostic(result, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    a, m = result['arrays'], result['metadata']
    np.savez_compressed(output/'input_model_diagnostic.npz', **a)
    (output/'input_model_diagnostic.json').write_text(json.dumps(m, indent=2,
        allow_nan=False)+'\n')
    fig, axes = plt.subplots(3, 1, figsize=(11., 8.), sharex=True)
    for j, axis in enumerate(axes):
        axis.plot(a['elapsed'], a['applied_prediction_torque'][:, j],
            color='#c03c35', linewidth=1.5, label='Command model + frozen past residual')
        axis.plot(a['elapsed'], a['heldout_gyro_implied_nominal_torque'][:, j],
            color='#34495e', linewidth=1.2, label='Held-out gyro implied, nominal J')
        axis.set_ylabel(('x', 'y', 'z')[j]+' torque [N m]')
        axis.grid(True, alpha=.25)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[-1].set_xlabel('Time from selected gyro initialization [s]')
    fig.suptitle('Read-only nominal torque diagnostic using held-out dense gyro', fontsize=14)
    fig.text(.06, .015,
        'Diagnostic only: raw gyro treated as omega; finite polynomial derivative; nominal J.\n'
        'These are not measured true torques. Held-out values never enter the sparse-gyro prediction.',
        fontsize=10)
    fig.tight_layout(rect=(0., .065, 1., .96))
    fig.savefig(output/'input_model_diagnostic.png', dpi=200)
    plt.close(fig)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUTPUT/'input.npz')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model', type=Path, default=HERE.parent/'grape_vehicle_model.json')
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {key:source[key] for key in source.files}
    control = build_torque_profile(data, args.model)
    result = build_diagnostic(data, control)
    metadata = save_diagnostic(result, args.output)
    print(json.dumps({k:metadata[k] for k in ('sample_count', 'mean_applied_torque_nm',
        'mean_implied_torque_nm', 'mean_torque_mismatch_nm', 'rms_torque_mismatch_nm',
        'mean_acceleration_mismatch_rad_s2',
        'integrated_acceleration_mismatch_endpoint_rad_s')}, indent=2))


if __name__ == '__main__':
    main()
