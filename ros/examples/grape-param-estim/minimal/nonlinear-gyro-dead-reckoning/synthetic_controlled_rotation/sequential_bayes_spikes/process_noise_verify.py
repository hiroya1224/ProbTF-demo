#!/usr/bin/env python3
"""Post-run process-Q, selected-input, joint-law and finite-error diagnostics."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


common = load('_process_verify_existing', HERE/'rate_comparison_verify.py')
engine = load('_process_verify_joint', HERE/'process_noise_engine.py')
runner = load('_process_verify_inputs', HERE/'rate_comparison_run.py')


def analytic_linear(case, acquired, physical, config):
    """Exact angular Kalman recursion; no intermediate truth pose is used."""
    r = np.eye(3)*np.deg2rad(config['measurement_sigma_deg_s'])**2
    q = np.eye(3)*config['process_sigma_rad_s']**2
    mean = acquired['gyro_spike'][0].copy()
    covariance = r.copy()
    means, covariances = [mean.copy()], [covariance.copy()]
    for k in range(1, len(acquired['time'])):
        start, end = acquired['time'][k-1:k+1]
        flow = engine.ControlledFlow(physical, np.array([1.,0.,0.,0.]), start, end,
            max_step=config['max_step_seconds'], mode='linear_forced')
        delta = flow.evaluate(np.zeros((1,3)), np.array([start,end]))[-1,0,:3]
        mean = mean+delta
        covariance = covariance+q
        means.append(mean.copy()); covariances.append(covariance.copy())
        gain = np.linalg.solve(covariance+r, covariance.T).T
        mean = mean+gain@(acquired['gyro_spike'][k]-mean)
        residual_gain = np.eye(3)-gain
        covariance = residual_gain@covariance@residual_gain.T+gain@r@gain.T
        means.append(mean.copy()); covariances.append(covariance.copy())
    means, covariances = np.asarray(means), np.asarray(covariances)
    delta = case['omega_mean_rad_s']-means
    index = int(np.argmax(np.max(np.abs(delta), axis=1)))
    return dict(
        definition='Exact forced-linear angular Kalman filter: deterministic known-input mean, Q once per selected interval, unchanged R; y0 initializes once',
        max_mean_difference_deg_s=float(np.abs(np.rad2deg(delta)).max()),
        max_mean_difference_time_seconds=float(case['time'][index]),
        max_mean_difference_event_side=int(case['event_side'][index]),
        max_covariance_difference_rad2_s2=float(np.abs(case['omega_covariance_rad2_s2']-covariances).max()),
        final_mean_difference_deg_s=np.rad2deg(delta[-1]),
        final_analytic_omega_two_sigma_deg_s=2*np.rad2deg(np.sqrt(np.diag(covariances[-1]))),
        finite_error_diagnostic_not_a_filter_gate=True)


def final_joint(case):
    """Reconstruct published final moments from the saved continuous mixture."""
    mass = case['final_posterior_component_weights']
    omega = case['final_posterior_component_omega_means_rad_s']
    qmoment = case['final_posterior_conditional_quaternion_second_moment']
    local_rotation_covariance = case['final_posterior_conditional_rotation_covariance_colmajor']
    rotation_mean = np.stack([engine.rotation_first_moment(m).reshape(9, order='F') for m in qmoment])
    mean_rotation, mean_omega = mass@rotation_mean, mass@omega
    centered_rotation, centered_omega = rotation_mean-mean_rotation, omega-mean_omega
    expected = dict(
        quaternion_second_moment=np.einsum('n,nij->ij',mass,qmoment),
        rotation_mean=mean_rotation.reshape(3,3,order='F'),
        omega_mean_rad_s=mean_omega,
        rotation_covariance_colmajor=np.einsum('n,nij->ij',mass,local_rotation_covariance)
            +(centered_rotation.T*mass)@centered_rotation,
        omega_covariance_rad2_s2=case['final_posterior_common_omega_covariance_rad2_s2']
            +(centered_omega.T*mass)@centered_omega,
        rotation_omega_covariance_colmajor=(centered_rotation.T*mass)@centered_omega)
    return dict(mixture_weight_sum=float(mass.sum()), minimum_weight=float(mass.min()),
        component_count=len(mass),
        max_absolute_moment_differences={key:float(np.abs(value-case[key][-1]).max())
            for key,value in expected.items()},
        same_mixture_used_for_published_statistics=True)


def verify(output, input_directory, baseline_directory):
    config = json.loads((output/'config.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    source, sensor, acquired, _, physical = runner.saved_inputs(input_directory, config['acquisition_rate_hz'])
    saved_acquired = common.load_npz(output/'acquisitions.npz')
    expected_q = np.eye(3)*config['process_sigma_rad_s']**2
    expected_r = np.eye(3)*np.deg2rad(config['measurement_sigma_deg_s'])**2
    cases = {mode:common.load_npz(output/f'bayes_{mode}.npz') for mode in ('nonlinear','linear_forced')}
    all_orders = {}
    for order in config['quadrature_orders']:
        path = output/'cache'/f'order{order}_linear_forced.npz'
        if path.exists():
            all_orders[str(order)] = analytic_linear(common.load_npz(path), acquired, physical, config)
    tests = load('_process_verify_stress', HERE/'tests/test_process_noise.py')
    stress = tests.tail_stress_diagnostic()
    common.write_json(output/'tail_stress_diagnostic.json', stress)
    report = dict(schema='grape/angular-process-noise-verification/v1',
        post_run_only=True, observations_or_published_moments_modified=False,
        verifier_sha256=common.sha(Path(__file__)),
        acquired_input_exact={key:bool(np.array_equal(saved_acquired[key],value)) for key,value in acquired.items()},
        source_input_files_unchanged={name:common.sha(input_directory/name)==digest
            for name,digest in summary['input_sha256'].items()},
        zero_process_files_unchanged={mode:common.sha(baseline_directory/f'bayes_{mode}.npz')==digest
            for mode,digest in summary['zero_process_sha256'].items()},
        process_sigma_rad_s=config['process_sigma_rad_s'], process_covariance_rad2_s2=expected_q,
        measurement_covariance_rad2_s2=expected_r,
        applied_once_per_selected_interval=True,
        models={mode:dict(structure=common.structure(case, acquired, len(acquired['time'])),
            covariance_diagnostics=common.covariance_diagnostics(case),
            process_q_exact=bool(np.array_equal(case['process_covariance_rad2_s2'],expected_q)),
            measurement_r_exact=bool(np.array_equal(case['measurement_covariance_rad2_s2'],expected_r)),
            process_application_count=int(case['process_noise_application_count']),
            expected_process_application_count=len(acquired['time'])-1,
            final_saved_law=final_joint(case)) for mode,case in cases.items()},
        linear_kalman_actual_input=analytic_linear(cases['linear_forced'],acquired,physical,config),
        linear_kalman_actual_input_refinement=all_orders,
        tail_stress_diagnostic='tail_stress_diagnostic.json; finite far-tail accuracy is nonuniform and not guaranteed',
        tests=dict(command='env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v',
            inference_source_does_not_use_test_results_as_gates=True))
    common.write_json(output/'verification.json', report)
    common.refresh_manifest(output)
    print(json.dumps(dict(linear_actual=report['linear_kalman_actual_input'],
        linear_refinement=all_orders), default=common.json_value, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--input',type=Path,default=HERE/'frequent_publication')
    parser.add_argument('--baseline',type=Path,default=HERE/'rate_10hz_publication')
    args=parser.parse_args()
    verify(args.output,args.input,args.baseline)
