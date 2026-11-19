#!/usr/bin/env python3
"""Post-run diagnostics of saved flight-scaled results; never filter gates."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def verify(output):
    shared = module('_flight_verify_shared', 'run.py')
    generator = module('_flight_verify_generator', 'frequent_spikes_input.py')
    native = module('_flight_verify_flow', 'native_flow.py')
    source = dict(np.load(output/'source_input.npz'))
    sensor = dict(np.load(output/'sensor_input.npz'))
    acquired = dict(np.load(output/'acquisitions.npz'))
    config = json.loads((output/'config.json').read_text())
    cases = {name: dict(np.load(output/f'bayes_{name}.npz'))
             for name in ('nonlinear', 'linear_forced')}
    # Compare sensor ingredients against the unchanged earlier motion. No
    # reference value, flag or result from this check is fed to a predictor.
    old_source, _, old_acquired, old_sensor, old_events, _ = generator.generate()
    diagnostics = {}
    for name, case in cases.items():
        rotation = case['rotation_covariance_colmajor']
        omega = case['omega_covariance_rad2_s2']
        cross = case['rotation_omega_covariance_colmajor']
        joint = np.concatenate((np.concatenate((rotation, cross), axis=2),
            np.concatenate((cross.swapaxes(1, 2), omega), axis=2)), axis=1)
        diagnostics[name] = dict(
            omega_covariance_shape=list(omega.shape),
            rotation_omega_cross_shape=list(cross.shape),
            all_joint_moments_finite=bool(np.isfinite(joint).all()),
            minimum_joint_covariance_eigenvalue_diagnostic=float(np.linalg.eigvalsh(joint).min()),
            initial_omega_two_sigma_halfwidth_deg_s=2*np.rad2deg(np.sqrt(np.diag(omega[0]))),
            final_omega_two_sigma_halfwidth_deg_s=2*np.rad2deg(np.sqrt(np.diag(omega[-1]))),
            maximum_core_vs_direct_rotation_mean_difference=float(case['core_vs_rotation_mean_max_abs'].max()))
    profile = dict(inertia=source['inertia_body_kg_m2'],
        torque=shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
    # Linear omega(t)=w0+A(t). Reuse the final posterior's exact RK4 query
    # clock to keep interpolation/integration differences separate.
    start, end = source['time']
    grid = np.unique(np.r_[np.linspace(start, end,
        int(round((end-start)/config['output_dt_seconds']))+1), acquired['time']])
    query = np.unique(np.r_[start, acquired['time'], grid[grid >= acquired['time'][-1]]])
    flow = native.ControlledFlow(profile, source['initial_quaternion_wxyz'], start, end,
        max_step=config['max_step_seconds'], mode='linear_forced')
    shift = flow.evaluate(np.zeros((1, 3)), query)[:, 0, :3]
    initial_mean = np.mean(acquired['gyro_spike']-shift[np.searchsorted(query, acquired['time'])], axis=0)
    final_mean = initial_mean+shift[-1]
    expected_covariance = cases['linear_forced']['measurement_covariance_rad2_s2']/len(acquired['time'])
    # A separate, read-only generator consistency check uses the one exact
    # initial truth value; it does not modify or rerun either Bayesian law.
    truth_flow = native.ControlledFlow(profile, source['initial_quaternion_wxyz'], start, end,
        max_step=config['max_step_seconds'], mode='nonlinear')
    reconstructed = truth_flow.evaluate(source['reference_omega_body_rad_s'][:1], source['reference_time'])[:, 0]
    attitude_error = (Rotation.from_quat(source['reference_quaternion_xyzw']).inv()
        *Rotation.from_quat(reconstructed[:, [4, 5, 6, 3]])).magnitude()
    movie = output/'frequent_spikes_comparison.mp4'
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration', '-of', 'json', str(movie)], text=True))['streams'][0]
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(movie), '-f', 'null', '-'],
        capture_output=True, text=True)
    same_initial = all(np.array_equal(cases['nonlinear'][key][0], cases['linear_forced'][key][0])
        for key in ('quaternion_second_moment', 'omega_mean_rad_s', 'omega_covariance_rad2_s2'))
    result = dict(schema='grape/flight-scaled-rotation-verification/v1',
        scope='post-run diagnostics only; none are observation acceptance or early-stop criteria',
        compared_models=list(cases),
        shared_spiked_measurement_sha256=hashlib.sha256(acquired['gyro_spike'].tobytes()).hexdigest(),
        initial_joint_moments_identical=same_initial,
        white_noise_realization_unchanged=bool(np.array_equal(source['gyro_realized_noise_rad_s'], old_source['gyro_realized_noise_rad_s'])),
        white_covariance_and_bias_unchanged=all(np.array_equal(source[k], old_source[k])
            for k in ('generator_gyro_covariance_rad2_s2', 'gyro_bias_rad_s')),
        spike_events_unchanged=json.loads((output/'spike_events.json').read_text()) == old_events,
        spike_vectors_masks_and_acquisition_clock_unchanged=all(np.array_equal(sensor[k], old_sensor[k])
            for k in ('spike_vector_rad_s', 'spike_mask', 'time', 'source_index'))
            and np.array_equal(acquired['time'], old_acquired['time']),
        physical_motion_std_deg_s=np.rad2deg(source['gyro_true_body_rad_s']).std(axis=0),
        physical_motion_max_abs_deg_s=np.abs(np.rad2deg(source['gyro_true_body_rad_s'])).max(axis=0),
        model_difference='Only omega cross(J omega); same quaternion kinematics, J, torque, readings and observation covariance',
        linear_model_analytic_posterior_check=dict(observed_count=len(acquired['time']),
            final_mean_max_difference_rad_s=float(np.max(np.abs(final_mean-cases['linear_forced']['omega_mean_rad_s'][-1]))),
            final_covariance_max_difference_rad2_s2=float(np.max(np.abs(expected_covariance-cases['linear_forced']['omega_covariance_rad2_s2'][-1])))),
        generation_dynamics_check_reference_only=dict(
            max_body_rate_error_deg_s=float(np.max(np.abs(np.rad2deg(reconstructed[:, :3]-source['reference_omega_body_rad_s'])))),
            max_attitude_error_deg=float(np.rad2deg(attitude_error.max())),
            oracle_values_supplied_to_bayesian_filters=False),
        numeric_diagnostics=diagnostics,
        omega_uncertainty_plotted_without_rescaling=True,
        no_white_only_comparison_output=not any((output/name).exists() for name in ('bayes_white.npz', 'bayes_spike.npz', 'white_case.pdf')),
        movie=dict(file=movie.name, sha256=shared.sha(movie), bytes=movie.stat().st_size,
            width=int(probe['width']), height=int(probe['height']), duration_seconds=float(probe['duration']),
            complete_decode=decode.returncode == 0))
    shared.write_json(output/'verification.json', result)
    files=[dict(path=str(f.relative_to(output)), bytes=f.stat().st_size, sha256=shared.sha(f))
        for f in sorted(output.rglob('*')) if f.is_file() and f.name!='manifest.json']
    shared.write_json(output/'manifest.json', dict(files=files,
        bytes_excluding_manifest=sum(f['bytes'] for f in files),
        compiled_binary_or_intermediate_frames_included=False))
    print(json.dumps(result, default=lambda v: v.tolist(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    verify(parser.parse_args().output)
