#!/usr/bin/env python3
"""Post-run diagnostics for every native gyro in the held-out real interval.

This script reads completed estimates. Its numerical diagnostics cannot
reject observations, stop inference, or alter a candidate distribution.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEQ = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sparse = load('_real_full_student_existing_checks', HERE/'verify.py')
helpers = sparse.helpers
sha, load_npz, write_json = helpers.sha, helpers.load_npz, helpers.write_json
difference = helpers.difference
MODELS = sparse.MODELS


def native_acquisition_checks(acquired, raw):
    index = np.flatnonzero((raw['imu_time'] >= raw['time'][0])
                          & (raw['imu_time'] <= raw['time'][1]))
    expected = dict(time=raw['imu_time'][index], gyro=raw['gyro'][index],
        header_stamp_ns=raw['imu_stamp_ns'][index], record_stamp_ns=raw['imu_record_ns'][index],
        native_array_index=index, original_source_index=raw['imu_source_index'][index])
    for alias in ('source_index', 'archive_array_index'):
        if alias in acquired:
            expected[alias] = index
    checks = {key: bool(np.array_equal(acquired[key], value))
              for key, value in expected.items()}
    return dict(selection='All original imu_time in the closed fixed physical interval',
        expected_native_count=len(index), actual_count=len(acquired['time']),
        original_values_indices_and_clocks_exact=checks,
        all_original_native_readings_used=all(checks.values()),
        selected_archive_array_indices=index,
        physical_interval_seconds=raw['time'],
        rate_computed_from_native_clock_hz=float((len(index)-1)/(acquired['time'][-1]-acquired['time'][0])),
        observed_interval_seconds=np.diff(acquired['time']),
        no_resampling_or_regular_grid_value_interpolation=True,
        final_prediction_tail_seconds=float(raw['time'][1]-acquired['time'][-1]),
        artificial_sensor_noise_or_spikes_added=False,
        bias_scale_or_clock_correction_added=False)


def protected_files_against_head():
    """Every tracked file in the earlier publications must remain unchanged."""
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    protected = (SEQ/'student_t_publication',
        HERE/'publication/bag8_50_55s',
        sparse.REFINED)
    result = {}
    for folder in protected:
        relative = folder.relative_to(root)
        names = subprocess.check_output(['git', 'ls-files', '--', str(relative)], text=True).splitlines()
        files = {}
        for name in names:
            before = subprocess.check_output(['git', 'show', 'HEAD:'+name])
            current = root/name
            files[name] = dict(head_sha256=hashlib.sha256(before).hexdigest(),
                current_sha256=sha(current), unchanged=before == current.read_bytes())
        result[str(relative)] = dict(tracked_file_count=len(files),
            all_tracked_files_unchanged=all(item['unchanged'] for item in files.values()), files=files)
    return dict(reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        all_protected_publications_unchanged=all(item['all_tracked_files_unchanged'] for item in result.values()),
        publications=result)


def independent_target_check(case, acquired, raw, physical, config, metadata, mode, engine):
    """Re-evaluate every prefix factor with SciPy, outside the update engine."""
    last_refresh = len(case['refresh_acquisition_index'])-1
    anchor = int(case['refresh_acquisition_index'][last_refresh])
    order = int(metadata['quadrature_order_per_axis'])
    center = case['refresh_proposal_mean_rad_s'][last_refresh]
    proposal = case['refresh_proposal_covariance_rad2_s2'][last_refresh]
    nodes, quadrature = engine.gaussian_tensor_points(center, proposal, order)
    last = len(acquired['time'])-1
    tail = case['time'][(case['event_side'] == 0) & (case['acquisition_index'] == last)]
    query = np.r_[acquired['time'], tail if len(tail) else raw['time'][1:]]
    flow = engine._adaptive.ControlledFlow(physical, raw['initial_quaternion_wxyz'],
        *raw['time'], max_step=config['max_step_seconds'], mode=mode)
    states = flow.evaluate(nodes, query)
    covariance = case['measurement_covariance_rad2_s2']
    nu = float(config['student_df'])
    scale = (nu-2.)/nu*covariance
    logs = (np.log(quadrature)
        +multivariate_normal.logpdf(nodes, mean=acquired['gyro'][0], cov=covariance)
        -multivariate_normal.logpdf(nodes, mean=center, cov=proposal))
    prefix_prior = logs-logsumexp(logs) if anchor == 0 else None
    prefix_posterior = prefix_prior.copy() if anchor == 0 else None
    for index in range(1, len(acquired['time'])):
        if index == anchor:
            prefix_prior = logs-logsumexp(logs)
        logs += multivariate_t.logpdf(states[index, :, :3]-acquired['gyro'][index],
            loc=np.zeros(3), shape=scale, df=nu)
        if index == anchor:
            prefix_posterior = logs-logsumexp(logs)
    logs -= logsumexp(logs)
    mass = np.exp(logs)
    moments = helpers.independent_joint_moments(states[-1], mass)
    saved_post = (case['last_refresh_log_weights'] if 'last_refresh_log_weights' in case
                  else case['refresh_log_weights'][-1])
    saved_prior = (case['last_refresh_prior_log_weights'] if 'last_refresh_prior_log_weights' in case
                   else case['refresh_prior_log_weights'][-1])
    return dict(reference='Native physical flow and independent scipy.stats.multivariate_t factors for every native gyro j>=1',
        acquired_count=len(acquired['time']), actual_student_likelihood_count=last,
        initial_gaussian_used_once=True, proposal_density_divided_out=True,
        last_refresh_acquisition_index=anchor,
        independently_recomputed_refresh_prior_logweight_difference=difference(saved_prior, prefix_prior),
        independently_recomputed_refresh_posterior_logweight_difference=difference(saved_post, prefix_posterior),
        degrees_of_freedom=nu, marginal_covariance_rad2_s2=covariance, scale_rad2_s2=scale,
        propagation_query_includes_all_postfinal_dense_times=True,
        physical_end_time=float(query[-1]), normalized_weight_sum=float(mass.sum()),
        final_joint_moment_differences={name: difference(case[name][-1], value)
            for name, value in moments.items()},
        final_state_is_propagated_not_held=True,
        mocap_or_gyro_outside_the_fixed_interval_used_in_target=False,
        finite_differences_are_diagnostics_not_filter_gates=True)


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    raw = load_npz(sparse.ARCHIVE/'input.npz')
    source = load_npz(output/'reference_input.npz')
    torque = load_npz(sparse.REFINED/'refined_torque_profile.npz')
    engine = load('_real_full_student_independent_engine', SEQ/'full_rate_engine.py')
    shared = load('_real_full_student_frozen_torque', SEQ/'run.py')
    physical = dict(inertia=torque['inertia_body'],
        torque=shared.TorqueProfile(torque['torque_profile_time'], torque['torque_applied_body']))
    root_config = json.loads((output/'config.json').read_text())
    frozen = load_npz(output/'frozen_physical_input.npz')
    report = dict(schema='grape/real-drone-full-native-student-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), post_run_only=True,
        observations_or_filter_state_changed=False,
        finite_differences_are_diagnostics_not_observation_gates=True,
        source_sha256=sha(__file__), input_archive_sha256=sha(sparse.ARCHIVE/'input.npz'),
        refined_profile_sha256=sha(sparse.REFINED/'refined_torque_profile.npz'),
        frozen_physical_input=dict(inertia_vs_archived_refined=difference(frozen['inertia'], torque['inertia_body']),
            torque_time_vs_archived=difference(frozen['torque_time'], torque['torque_profile_time']),
            actuator_torque_vs_archived=difference(frozen['torque_actuator_body'], torque['torque_actuator_body']),
            applied_torque_vs_archived=difference(frozen['torque_applied_body'], torque['torque_applied_body']),
            residual_mean_vs_archived=difference(frozen['residual_mean_body'], torque['residual_mean_body'])),
        preserved_publications=protected_files_against_head(), cases={})
    if (output/'preserved_sources.json').exists():
        report['runner_preservation_record'] = json.loads((output/'preserved_sources.json').read_text())
    for directory in sorted(output.glob('rate_full_native/sigma_*_deg_s')):
        key = str(directory.relative_to(output))
        config = sparse.case_config(directory, root_config)
        acquired_path = directory/'acquisitions.npz'
        if not acquired_path.exists():
            acquired_path = directory.parent/'acquisitions.npz'
        acquired = load_npz(acquired_path)
        summary = json.loads((directory/'summary.json').read_text())
        cases = {mode: load_npz(directory/f'bayes_{mode}.npz') for mode in MODELS}
        item = dict(acquisitions=native_acquisition_checks(acquired, raw),
            case_declares_no_process_noise=bool(np.all(np.asarray(config['process_covariance_rad2_s2']) == 0.)),
            models={})
        for mode, case in cases.items():
            nu = float(config['student_df'])
            covariance = np.eye(3)*np.deg2rad(float(config['sigma_deg_s']))**2
            structure = sparse.structure(case, acquired, raw['time'])
            structure.update(expected_prior_rows=len(acquired['time'])-1,
                actual_prior_rows=int(np.count_nonzero(case['event_side'] == -1)),
                expected_posterior_rows=len(acquired['time']),
                actual_posterior_rows=int(np.count_nonzero(case['event_side'] == 1)))
            item['models'][mode] = dict(structure=structure,
                covariance=helpers.covariance_diagnostics(case),
                declared_student_parameters=dict(degrees=difference(case['observation_student_df'], np.asarray(nu)),
                    marginal_covariance=difference(case['measurement_covariance_rad2_s2'], covariance),
                    scale=difference(case['observation_student_scale_rad2_s2'], (nu-2.)/nu*covariance)),
                full_student_target=independent_target_check(case, acquired, raw, physical, config,
                    summary['engine_metadata'][mode], mode, engine))
        item['plots'] = sparse.plot_checks(directory, source, cases)
        report['cases'][key] = item
        print('Verified '+key, flush=True)
    report['case_count'] = len(report['cases'])
    if args.test_evidence is not None:
        report['supplied_actual_test_evidence'] = json.loads(args.test_evidence.read_text())
        report['supplied_actual_test_evidence_sha256'] = sha(args.test_evidence)
    report['wall_seconds'] = time.monotonic()-started
    write_json(output/'verification.json', report)
    helpers.refresh_manifest(output)
    print('Wrote '+str(output/'verification.json'), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=HERE/'publication/bag8_full_gyro')
    parser.add_argument('--test-evidence', type=Path)
    run(parser.parse_args())
