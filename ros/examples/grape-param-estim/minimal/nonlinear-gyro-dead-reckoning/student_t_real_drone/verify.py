#!/usr/bin/env python3
"""Read-only verification of Student-t reconstruction on held-out real flight.

Every finite numerical difference is recorded. No diagnostic changes a
candidate distribution, rejects a measurement, or gates a later update.
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
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEQ = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'
ARCHIVE = BASE/'publication/bag8_50_55s'
REFINED = BASE/'mocap_refined_dynamics/publication/bag8_40_45_fit_50_55_predict'
MODELS = ('nonlinear', 'linear_forced')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load('_real_student_verification_helpers', SEQ/'rate_comparison_verify.py')
sha, load_npz, write_json = helpers.sha, helpers.load_npz, helpers.write_json
difference = helpers.difference


def acquisition_checks(acquired, rate_hz, raw, metadata):
    requested_start, requested_end = metadata['requested_window_seconds']
    requested = requested_start+np.arange(0., requested_end-requested_start, 1./rate_hz)
    indices = np.searchsorted(raw['imu_time'], requested, side='left')
    expected = dict(time=raw['imu_time'][indices], gyro=raw['gyro'][indices],
        header_stamp_ns=raw['imu_stamp_ns'][indices], record_stamp_ns=raw['imu_record_ns'][indices])
    checks = {name: bool(np.array_equal(acquired[name], value)) for name, value in expected.items()}
    for name in ('source_index', 'archive_array_index', 'native_array_index'):
        if name in acquired:
            checks[name] = bool(np.array_equal(acquired[name], indices))
    if 'original_source_index' in acquired:
        checks['original_source_index'] = bool(np.array_equal(
            acquired['original_source_index'], raw['imu_source_index'][indices]))
    return dict(requested_acquisition_rate_hz=rate_hz, expected_count=len(indices),
        actual_count=len(acquired['time']), original_values_and_clocks_exact=checks,
        requested_bag_times=requested, selected_archive_array_indices=indices,
        physical_interval_seconds=raw['time'], initial_acquisition_matches_start=bool(
            acquired['time'][0]==raw['time'][0]),
        final_prediction_tail_seconds=float(raw['time'][1]-acquired['time'][-1]),
        synthetic_sensor_noise_added=False, bias_scale_or_clock_correction_added=False)


def structure(case, acquired, bounds):
    count = len(acquired['time'])
    index = np.r_[0, np.repeat(np.arange(1, count), 2)]
    sides = np.r_[1, np.tile([-1, 1], count-1)]
    observed_rows = case['event_side'] != 0
    actual_index = np.asarray(case['acquisition_index'], int)
    expected_factors = np.where(case['event_side']<0, actual_index-1, actual_index)
    grid = np.linspace(*bounds, 501)
    saved_ns = np.rint(case['time']*1e9).astype(np.int64)
    wanted_ns = np.rint(grid*1e9).astype(np.int64)
    return dict(actual_saved_rows=len(case['time']), expected_observation_rows=2*count-1,
        actual_observation_rows=int(np.count_nonzero(observed_rows)),
        actual_dense_rows=int(np.count_nonzero(~observed_rows)),
        observation_times_exact=bool(np.array_equal(case['time'][observed_rows], acquired['time'][index])),
        observation_sides_exact=bool(np.array_equal(case['event_side'][observed_rows], sides)),
        observation_acquisition_indices_exact=bool(np.array_equal(actual_index[observed_rows], index)),
        all_likelihood_factor_counts_exact=bool(np.array_equal(
            case['target_likelihood_factor_count'], expected_factors)),
        final_time_equals_fixed_physical_end=bool(case['time'][-1]==bounds[1]),
        final_likelihood_count=int(case['target_likelihood_factor_count'][-1]),
        expected_final_likelihood_count=count-1,
        saved_501_grid_present_without_estimate_interpolation=bool(np.isin(wanted_ns, saved_ns).all()),
        clock_matching='Integer-ns equality for identifying the fixed grid; no estimate interpolation',
        tail_dense_row_count=int(np.count_nonzero((case['event_side']==0)&(actual_index==count-1))))


def final_target_check(case, acquired, raw, profile, config, metadata, mode, engine):
    """SciPy Student-t factors and independently converted quaternion moments."""
    last = len(acquired['time'])-1
    refresh = len(case['refresh_acquisition_index'])-1
    order = int(metadata['quadrature_order_per_axis'])
    center = case['refresh_proposal_mean_rad_s'][refresh]
    proposal = case['refresh_proposal_covariance_rad2_s2'][refresh]
    nodes, quadrature = engine.gaussian_tensor_points(center, proposal, order)
    tail = case['time'][(case['event_side']==0)&(case['acquisition_index']==last)]
    query = np.r_[acquired['time'], tail]
    if not len(tail):
        query = np.r_[acquired['time'], raw['time'][1]]
    flow = engine._adaptive.ControlledFlow(profile, raw['initial_quaternion_wxyz'],
        *raw['time'], max_step=config['max_step_seconds'], mode=mode)
    states = flow.evaluate(nodes, query)
    covariance = case['measurement_covariance_rad2_s2']
    nu = float(config['student_df'])
    scale = (nu-2.)/nu*covariance
    logs = (np.log(quadrature)
        +multivariate_normal.logpdf(nodes, mean=acquired['gyro'][0], cov=covariance)
        -multivariate_normal.logpdf(nodes, mean=center, cov=proposal))
    for index in range(1, len(acquired['time'])):
        logs += multivariate_t.logpdf(states[index, :, :3]-acquired['gyro'][index],
            loc=np.zeros(3), shape=scale, df=nu)
    logs -= logsumexp(logs)
    mass = np.exp(logs)
    moments = helpers.independent_joint_moments(states[-1], mass)
    return dict(reference='Independent native quaternion flow plus scipy.stats.multivariate_t factors for ALL selected j>=1 observations',
        actual_student_likelihood_count=last, initial_gaussian_used_once=True,
        proposal_density_divided_out=True, degrees_of_freedom=nu,
        marginal_covariance_rad2_s2=covariance, scale_rad2_s2=scale,
        propagation_query_includes_all_postfinal_dense_times=True,
        physical_end_time=float(query[-1]), normalized_weight_sum=float(mass.sum()),
        final_joint_moment_differences={name: difference(case[name][-1], value)
            for name, value in moments.items()},
        final_state_is_propagated_not_held=True,
        finite_differences_are_diagnostics_not_filter_gates=True,
        mocap_or_unacquired_gyro_used_in_target=False)


def plot_checks(directory, source, cases):
    path = directory/'plotted_data.npz'
    if not path.exists():
        return dict(available=False)
    plotted = load_npz(path)
    reference = Slerp(source['reference_time'], Rotation.from_quat(source['reference_quaternion_xyzw']))
    record = json.loads((directory/'plotting.json').read_text())
    stored_metrics = json.loads((directory/'metrics.json').read_text())
    result = dict(available=True, models={},
        recorded_settings=record['settings'],
        renderer_sha256_matches=record['renderer_sha256']==sha(HERE/'plotting.py'))
    for mode, case in cases.items():
        stamp = case['time']
        rotations = reference(stamp)
        reference_rpy = rotations.as_euler('xyz')
        rpy_error = np.rad2deg(np.arctan2(np.sin(case['rpy_proxy_rad']-reference_rpy),
                                        np.cos(case['rpy_proxy_rad']-reference_rpy)))
        gyro = np.column_stack([np.interp(stamp, source['gyro_time'], source['gyro_body_rad_s'][:, axis]) for axis in range(3)])
        expected = dict(time=stamp,
            rpy_two_sigma_deg=2*np.rad2deg(np.sqrt(np.diagonal(case['rpy_covariance_delta_rad2'],axis1=1,axis2=2))),
            omega_two_sigma_deg_s=2*np.rad2deg(np.sqrt(np.diagonal(case['omega_covariance_rad2_s2'],axis1=1,axis2=2))),
            rpy_error_deg=rpy_error,
            omega_vs_raw_gyro_error_deg_s=np.rad2deg(case['omega_mean_rad_s']-gyro),
            so3_proxy_error_deg=np.rad2deg((rotations.inv()*Rotation.from_euler('xyz',case['rpy_proxy_rad'])).magnitude()))
        checked = {name:difference(plotted[mode+'_'+name], value)
            for name,value in expected.items() if mode+'_'+name in plotted}
        rows = np.asarray(plotted[mode+'_metric_row'],int)
        metric_time = np.linspace(*source['time'],501)
        selected_time = np.rint(stamp[rows]*1e9).astype(np.int64)
        expected_time = np.rint(metric_time*1e9).astype(np.int64)
        checked.update(metric_count=len(rows),metric_times_match_fixed_501_grid=bool(np.array_equal(
            selected_time,expected_time)),metric_rows_are_not_observation_priors=bool(
            np.all(case['event_side'][rows]>=0)),
            recorded_geodesic_rmse_difference_deg=difference(np.asarray(
                stored_metrics['results'][mode]['so3_proxy_rmse_deg']),np.asarray(
                np.sqrt(np.mean(expected['so3_proxy_error_deg'][rows]**2)))))
        result['models'][mode] = checked
    result.update(mocap_comparison='SLERP of original measured quaternion, read-only evaluation',
        gyro_error_reference='Raw gyro interpolation for evaluation only; not truth angular velocity',
        saved_covariance_adjusted=False)
    return result


def synthetic_preservation():
    """Compare every tracked precious synthetic result to the current HEAD."""
    root = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    relative = (SEQ/'student_t_publication').relative_to(root)
    names = subprocess.check_output(['git','ls-files','--',str(relative)],text=True).splitlines()
    files = {}
    for name in names:
        before = subprocess.check_output(['git','show','HEAD:'+name])
        current = root/name
        files[name] = dict(head_sha256=hashlib.sha256(before).hexdigest(),
            current_sha256=sha(current), unchanged=before==current.read_bytes())
    return dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_file_count=len(names), all_tracked_synthetic_results_unchanged=all(
            item['unchanged'] for item in files.values()), files=files)


def case_config(directory, root_config):
    config = dict(root_config)
    if (directory/'config.json').exists():
        config.update(json.loads((directory/'config.json').read_text()))
    return config


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    raw = load_npz(ARCHIVE/'input.npz')
    source = load_npz(output/'reference_input.npz')
    raw_metadata = json.loads((ARCHIVE/'input.json').read_text())
    torque = load_npz(REFINED/'refined_torque_profile.npz')
    engine = load('_real_student_independent_engine', SEQ/'full_rate_engine.py')
    shared = load('_real_student_torque', SEQ/'run.py')
    profile = dict(inertia=torque['inertia_body'],
        torque=shared.TorqueProfile(torque['torque_profile_time'], torque['torque_applied_body']))
    root_config = json.loads((output/'config.json').read_text())
    frozen = load_npz(output/'frozen_physical_input.npz')
    report = dict(schema='grape/real-drone-student-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), post_run_only=True,
        observations_or_filter_state_changed=False,
        finite_differences_are_diagnostics_not_observation_gates=True,
        source_sha256=sha(__file__), input_archive_sha256=sha(ARCHIVE/'input.npz'),
        refined_profile_sha256=sha(REFINED/'refined_torque_profile.npz'),
        frozen_physical_input=dict(inertia_vs_archived_refined=difference(
            frozen['inertia'],torque['inertia_body']),
            torque_time_vs_archived=difference(frozen['torque_time'],torque['torque_profile_time']),
            applied_torque_vs_archived=difference(frozen['torque_applied_body'],torque['torque_applied_body']),
            residual_mean_vs_archived=difference(frozen['residual_mean_body'],torque['residual_mean_body'])),
        preserved_synthetic_results=synthetic_preservation(), cases={})
    for directory in sorted(output.glob('rate_*_hz/sigma_*_deg_s')):
        key = str(directory.relative_to(output))
        config = case_config(directory, root_config)
        rate = float(config.get('rate_hz', directory.parent.name[5:-3].replace('p','.')))
        acquired_path = directory/'acquisitions.npz'
        if not acquired_path.exists():
            acquired_path = directory.parent/'acquisitions.npz'
        acquired = load_npz(acquired_path)
        summary = json.loads((directory/'summary.json').read_text())
        cases = {mode:load_npz(directory/f'bayes_{mode}.npz') for mode in MODELS}
        item = dict(acquisitions=acquisition_checks(acquired,rate,raw,raw_metadata),models={})
        for mode, case in cases.items():
            sigma = float(config['sigma_deg_s'])
            nu = float(config['student_df'])
            covariance = np.eye(3)*np.deg2rad(sigma)**2
            item['models'][mode] = dict(structure=structure(case,acquired,raw['time']),
                covariance=helpers.covariance_diagnostics(case),
                declared_student_parameters=dict(degrees=difference(case['observation_student_df'],np.asarray(nu)),
                    marginal_covariance=difference(case['measurement_covariance_rad2_s2'],covariance),
                    scale=difference(case['observation_student_scale_rad2_s2'],(nu-2.)/nu*covariance)),
                full_student_target=final_target_check(case,acquired,raw,profile,config,
                    summary['engine_metadata'][mode],mode,engine))
        item['plots'] = plot_checks(directory,source,cases)
        report['cases'][key] = item
        print('Verified '+key,flush=True)
    report['case_count'] = len(report['cases'])
    if args.test_evidence is not None:
        report['supplied_actual_test_evidence'] = json.loads(args.test_evidence.read_text())
        report['supplied_actual_test_evidence_sha256'] = sha(args.test_evidence)
    report['wall_seconds'] = time.monotonic()-started
    write_json(output/'verification.json',report)
    helpers.refresh_manifest(output)
    print('Wrote '+str(output/'verification.json'),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=HERE/'publication/bag8_50_55s')
    parser.add_argument('--test-evidence',type=Path)
    run(parser.parse_args())
