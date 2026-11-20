#!/usr/bin/env python3
"""Independent post-run diagnostics of the Student-t-only comparison.

The target is the original first-reading Gaussian times every subsequent
multivariate Student-t likelihood. Checks never change a candidate law or
gate an observation. Finite integration differences are recorded as numbers.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parent
MODELS = ('nonlinear', 'linear_forced')


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load('_student_verification_helpers', 'rate_comparison_verify.py')
sha, load_npz, write_json = helpers.sha, helpers.load_npz, helpers.write_json
difference = helpers.difference


def full_target_check(case, source, acquired, config, physical, mode, metadata, engine):
    """Re-evaluate all likelihoods with SciPy, independently of native scores."""
    anchor = int(case['refresh_acquisition_index'][-1])
    order = int(metadata['quadrature_order_per_axis'])
    center = case['refresh_proposal_mean_rad_s'][-1]
    proposal_covariance = case['refresh_proposal_covariance_rad2_s2'][-1]
    nodes, quadrature = engine.gaussian_tensor_points(center, proposal_covariance, order)
    covariance = case['measurement_covariance_rad2_s2']
    degrees = float(config['student_df'])
    scale = (degrees-2.)/degrees*covariance
    flow = engine._adaptive.ControlledFlow(physical, source['initial_quaternion_wxyz'],
        *source['time'], max_step=config['max_step_seconds'], mode=mode)
    states = flow.evaluate(nodes, acquired['time'])
    raw = (np.log(quadrature)
        +multivariate_normal.logpdf(nodes, mean=acquired['gyro_spike'][0], cov=covariance)
        -multivariate_normal.logpdf(nodes, mean=center, cov=proposal_covariance))
    prior_at_anchor = None
    posterior_at_anchor = None
    for index in range(len(acquired['time'])):
        if index==anchor:
            prior_at_anchor = raw-logsumexp(raw)
        if index:
            raw += multivariate_t.logpdf(states[index,:,:3]-acquired['gyro_spike'][index],
                loc=np.zeros(3), shape=scale, df=degrees)
        if index==anchor:
            posterior_at_anchor = raw-logsumexp(raw)
    final_logs = raw-logsumexp(raw)
    mass = np.exp(final_logs)
    moments = helpers.independent_joint_moments(states[-1], mass)
    saved_anchor = (case['last_refresh_log_weights'] if 'last_refresh_log_weights' in case
                    else case['refresh_log_weights'][-1])
    saved_prior = (case['last_refresh_prior_log_weights'] if 'last_refresh_prior_log_weights' in case
                   else case['refresh_prior_log_weights'][-1])
    return dict(reference='scipy.stats.multivariate_t.logpdf; independently accumulated original prior times every acquired j>=1 likelihood divided by proposal',
        degrees_of_freedom=degrees, observation_marginal_covariance_rad2_s2=covariance,
        student_scale_rad2_s2=scale, initial_law='N(W0;y0,R), first reading once',
        final_student_likelihood_count=len(acquired['time'])-1,
        saved_refresh_acquisition_index=anchor, oracle_or_spike_labels_used=False,
        final_normalized_weight_sum=float(mass.sum()),
        initial_gaussian_prior_used_once=True, gaussian_observation_likelihoods_after_initialization=0,
        last_refresh_prior_log_weight_difference=difference(saved_prior,prior_at_anchor),
        last_refresh_posterior_log_weight_difference=difference(saved_anchor,posterior_at_anchor),
        final_joint_moment_differences={name:difference(case[name][-1], value)
                                        for name,value in moments.items()},
        finite_differences_are_diagnostics_not_filter_gates=True)


def plot_checks(output, source, sensor, acquired, cases):
    record_path, data_path = output/'student_t_plotting.json', output/'student_t_plotted_data.npz'
    if not record_path.exists() or not data_path.exists():
        return dict(available=False, reason='Saved plot data/record not present')
    record = json.loads(record_path.read_text())
    data = load_npz(data_path)
    result = dict(available=True, models={},
        renderer_sha256_matches_current_source=record.get('renderer_sha256')==sha(HERE/'student_t_plotting.py'),
        comparison_contains_only_student_t_models=set(cases)==set(MODELS),
        gaussian_observation_case_drawn=False)
    for mode, case in cases.items():
        stamp = case['time']
        truth_rpy = np.column_stack([np.interp(stamp, source['reference_time'], source['reference_rpy_rad'][:,axis])
                                    for axis in range(3)])
        truth_omega = np.column_stack([np.interp(stamp, source['reference_time'], source['reference_omega_body_rad_s'][:,axis])
                                      for axis in range(3)])
        rpy_error = np.rad2deg(np.arctan2(np.sin(case['rpy_proxy_rad']-truth_rpy),
                                        np.cos(case['rpy_proxy_rad']-truth_rpy)))
        omega_error = np.rad2deg(case['omega_mean_rad_s']-truth_omega)
        so3_error = np.rad2deg((Rotation.from_euler('xyz',truth_rpy).inv()
            *Rotation.from_euler('xyz',case['rpy_proxy_rad'])).magnitude())
        result['models'][mode] = dict(
            saved_time_exact=bool(np.array_equal(data[mode+'_time'], stamp)),
            rpy_two_sigma_vs_saved_covariance=difference(data[mode+'_rpy_two_sigma_deg'],
                2*np.rad2deg(np.sqrt(np.diagonal(case['rpy_covariance_delta_rad2'],axis1=1,axis2=2)))),
            omega_two_sigma_vs_saved_covariance=difference(data[mode+'_omega_two_sigma_deg_s'],
                2*np.rad2deg(np.sqrt(np.diagonal(case['omega_covariance_rad2_s2'],axis1=1,axis2=2)))),
            plotted_rpy_error=difference(data[mode+'_rpy_error_deg'],rpy_error),
            plotted_omega_error=difference(data[mode+'_omega_error_deg_s'],omega_error),
            plotted_so3_error=difference(data[mode+'_so3_proxy_error_deg'],so3_error))
    return result


def run(args):
    begun = time.monotonic()
    output, baseline = args.output.resolve(), args.baseline.resolve()
    config = json.loads((output/'config.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    source, sensor, acquired = [load_npz(output/name) for name in
                               ('source_input.npz','sensor_input.npz','acquisitions.npz')]
    cases = {mode:load_npz(output/f'bayes_{mode}.npz') for mode in MODELS}
    engine = load('_student_verification_engine','full_rate_engine.py')
    shared = load('_student_verification_torque','run.py')
    physical = dict(inertia=source['inertia_body_kg_m2'],
                    torque=shared.TorqueProfile(source['torque_time'],source['torque_body_nm']))
    names = ('source_input.npz','sensor_input.npz','acquisitions.npz')
    report = dict(schema='grape/student-t-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), post_run_only=True,
        observations_or_filter_state_changed=False,
        finite_differences_are_diagnostics_not_observation_gates=True,
        verifier_sha256=sha(__file__),
        inputs_bitwise_equal_to_preserved_half_spike_input={
            name:sha(output/name)==sha(baseline/name) for name in names},
        input_sha256={name:sha(output/name) for name in names},
        degrees_of_freedom=config['student_df'],
        declared_process_covariance_exactly_zero=bool(np.array_equal(
            np.asarray(config['process_covariance_rad2_s2']),np.zeros((3,3)))),
        numerical_test_provenance=dict(tests_run_by_this_verifier=False),
        code_sha256={path.name:sha(path) for pattern in ('student_t*.py','full_rate*.py','full_rate*.cpp')
            for path in sorted(HERE.glob(pattern))}, models={})
    count = int(round(np.diff(source['time'])[0]*config['acquisition_rate_hz']))+1
    for mode, case in cases.items():
        print(f'Verify Student-t {mode}: {len(acquired["time"])} observations',flush=True)
        item = dict(structure=helpers.structure(case,acquired,count),
            covariance=helpers.covariance_diagnostics(case),
            recorded_engine_metadata=summary['engine_metadata'][mode],
            observation_covariance_vs_config=difference(case['measurement_covariance_rad2_s2'],
                np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2),
            saved_student_degrees_vs_config=difference(case['observation_student_df'],
                np.asarray(config['student_df'])),
            saved_student_scale_vs_declared_marginal_covariance=difference(
                case['observation_student_scale_rad2_s2'],
                (config['student_df']-2.)/config['student_df']*case['measurement_covariance_rad2_s2']),
            independent_original_prior_and_all_student_factors=full_target_check(
                case,source,acquired,config,physical,mode,summary['engine_metadata'][mode],engine))
        report['models'][mode] = item
    report['plotted_saved_numeric_identity'] = plot_checks(output,source,sensor,acquired,cases)
    files = [path for path in output.rglob('*') if path.is_file()
             and 'cache' not in path.relative_to(output).parts]
    report['publication_files'] = dict(maximum_bytes=max((path.stat().st_size for path in files),default=0),
        all_files_below_50000000_bytes=all(path.stat().st_size<50_000_000 for path in files),
        cache_excluded=True)
    if args.test_evidence is not None:
        report['numerical_test_provenance']['supplied_evidence'] = json.loads(args.test_evidence.read_text())
        report['numerical_test_provenance']['supplied_evidence_sha256'] = sha(args.test_evidence)
    report['wall_seconds'] = time.monotonic()-begun
    write_json(output/'verification.json',report)
    helpers.refresh_manifest(output)
    print('Wrote '+str(output/'verification.json'),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=HERE/'student_t_publication')
    parser.add_argument('--baseline',type=Path,default=HERE/'omega3_q0_publication')
    parser.add_argument('--test-evidence',type=Path)
    run(parser.parse_args())
