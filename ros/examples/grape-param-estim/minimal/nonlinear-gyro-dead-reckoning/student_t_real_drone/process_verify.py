#!/usr/bin/env python3
"""Post-run checks of native gyro, process increments and joint Student-t law.

All diagnostics are read-only. A finite numerical discrepancy is recorded,
never used to discard a reading, gate inference, or adjust a covariance.
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
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEQ = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'
MODELS = ('nonlinear', 'linear_forced')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = load('_real_process_verifier_full_helpers', HERE/'full_verify.py')
inputs = load('_real_process_verifier_saved_inputs', HERE/'input.py')
sparse = full.sparse
helpers = full.helpers
sha, load_npz, write_json = helpers.sha, helpers.load_npz, helpers.write_json
difference = helpers.difference


def quaternion_moment_rotation(moment):
    """Independent Hamilton wxyz -> body-to-world E[R] polynomial."""
    moment = np.asarray(moment)
    ww, xx, yy, zz = (moment[..., k, k] for k in range(4))
    wx, wy, wz = (moment[..., 0, k] for k in (1, 2, 3))
    xy, xz, yz = moment[..., 1, 2], moment[..., 1, 3], moment[..., 2, 3]
    return np.stack((ww+xx-yy-zz, 2*(xy-wz), 2*(xz+wy),
                     2*(xy+wz), ww-xx+yy-zz, 2*(yz-wx),
                     2*(xz-wy), 2*(yz+wx), ww-xx-yy+zz), axis=-1).reshape(
                         moment.shape[:-2]+(3, 3))


def independent_joint(mass, omega, quaternion_moments, conditional_rotation_covariance):
    """Total covariance of omega-node conditional orientation laws."""
    mass = np.asarray(mass, float)
    omega = np.asarray(omega, float)
    local_means = quaternion_moment_rotation(quaternion_moments)
    vectors = local_means.transpose(0, 2, 1).reshape(-1, 9)
    vector_mean, omega_mean = mass@vectors, mass@omega
    centered_r, centered_omega = vectors-vector_mean, omega-omega_mean
    moment = np.einsum('n,nij->ij', mass, quaternion_moments)
    return dict(quaternion_second_moment=moment,
        rotation_mean=vector_mean.reshape(3, 3, order='F'),
        rotation_covariance_colmajor=np.einsum('n,nij->ij', mass, conditional_rotation_covariance)
            +(centered_r.T*mass)@centered_r,
        omega_mean_rad_s=omega_mean,
        omega_covariance_rad2_s2=(centered_omega.T*mass)@centered_omega,
        rotation_omega_covariance_colmajor=(centered_r.T*mass)@centered_omega)


def independent_density_rule(component_weights, component_means, component_covariance,
        component_qmoments, component_rotation_covariance, nodes, quadrature_weights,
        proposal_mean, proposal_covariance, measurement, measurement_covariance, student_df):
    """SciPy Gaussian-mixture × Student-t target / saved proposal, no fitting."""
    component_weights = np.asarray(component_weights, float)
    nodes = np.asarray(nodes, float)
    with np.errstate(divide='ignore'):
        log_component_weights = np.log(component_weights)
        log_base = np.log(quadrature_weights)
    component_logs = multivariate_normal.logpdf(
        nodes[:, None, :]-np.asarray(component_means)[None, :, :],
        mean=np.zeros(3), cov=component_covariance)+log_component_weights[None, :]
    log_density = logsumexp(component_logs, axis=1)
    responsibilities = np.exp(component_logs-log_density[:, None])
    if measurement is not None:
        nu = float(student_df)
        scale = (nu-2.)/nu*np.asarray(measurement_covariance)
        observation = multivariate_t.logpdf(nodes-np.asarray(measurement),
            loc=np.zeros(3), shape=scale, df=nu)
    else:
        observation = np.zeros(len(nodes))
    log_weight = (log_base+log_density+observation
        -multivariate_normal.logpdf(nodes, mean=proposal_mean, cov=proposal_covariance))
    log_normalizer = float(logsumexp(log_weight))
    posterior_weights = np.exp(log_weight-log_normalizer)
    qmoments = np.einsum('ni,ijk->njk', responsibilities, component_qmoments)
    component_r = quaternion_moment_rotation(component_qmoments).transpose(0,2,1).reshape(-1,9)
    node_r = quaternion_moment_rotation(qmoments).transpose(0,2,1).reshape(-1,9)
    covariance = np.einsum('ni,ijk->njk', responsibilities, component_rotation_covariance)
    for lower in range(0, len(nodes), 64):
        upper = min(lower+64, len(nodes))
        centered = component_r[None, :, :]-node_r[lower:upper, None, :]
        covariance[lower:upper] += np.einsum('ni,nij,nik->njk',
            responsibilities[lower:upper], centered, centered, optimize=True)
    return dict(weights=posterior_weights, quaternion_moments=qmoments,
        rotation_covariances=covariance, log_normalizer=log_normalizer,
        moments=independent_joint(posterior_weights, nodes, qmoments, covariance),
        responsibility_sum_error=float(np.max(np.abs(responsibilities.sum(axis=1)-1.))))


def selected_acquisition_checks(acquired, rate_hz, raw, metadata):
    """Original first-at/after native reading for each legacy requested time."""
    start,end = metadata['requested_window_seconds']
    requested = start+np.arange(0.,end-start,1./float(rate_hz))
    index = np.searchsorted(raw['imu_time'],requested,side='left')
    expected = dict(requested_time=requested,time=raw['imu_time'][index],gyro=raw['gyro'][index],
        header_stamp_ns=raw['imu_stamp_ns'][index],record_stamp_ns=raw['imu_record_ns'][index],
        original_source_index=raw['imu_source_index'][index],
        requested_bag_stamp_ns=np.rint(requested*1e9).astype(np.int64))
    for alias in ('source_index','archive_array_index','native_array_index'):
        if alias in acquired:
            expected[alias] = index
    checks = {key:bool(np.array_equal(acquired[key],value)) for key,value in expected.items()}
    _,historical = inputs.historical.acquisitions(raw,metadata,float(rate_hz))
    legacy = {key:bool(np.array_equal(acquired[key],value)) for key,value in historical.items()}
    native = (raw['imu_time']>=raw['time'][0])&(raw['imu_time']<=raw['time'][1])
    return dict(selection='First original native gyro at or after each legacy requested bag time; no value interpolation',
        acquisition_mode='selected_rate',requested_acquisition_rate_hz=float(rate_hz),
        expected_count=len(index),actual_count=len(acquired['time']),
        all_selected_native_readings_exact=all(checks.values()) and all(legacy.values()),
        original_values_indices_and_clocks_exact=checks,historical_selector_evidence_exact=legacy,
        requested_bag_times=requested,selected_archive_array_indices=index,
        original_window_native_record_count=int(np.count_nonzero(native)),
        physical_interval_seconds=raw['time'],
        selected_readings_inside_fixed_window=bool(np.all((raw['imu_time'][index]>=raw['time'][0])
            &(raw['imu_time'][index]<=raw['time'][1]))),
        gyro_values_resampled_or_interpolated=False,
        final_prediction_tail_seconds=float(raw['time'][1]-acquired['time'][-1]),
        artificial_sensor_noise_or_spikes_added=False,bias_scale_or_clock_correction_added=False)


def acquisition_checks(acquired,config,raw,metadata):
    if config.get('acquisition_mode','full_native')=='selected_rate':
        return selected_acquisition_checks(acquired,float(config['rate_hz']),raw,metadata)
    return full.native_acquisition_checks(acquired,raw)


def protected_files_against_head(current_output=None):
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    protected = (SEQ/'student_t_publication', HERE/'publication/bag8_50_55s',
        HERE/'publication/bag8_full_gyro', HERE/'publication/bag8_full_gyro_q1', sparse.REFINED)
    records = {}
    for folder in protected:
        if current_output is not None and folder.resolve()==Path(current_output).resolve():
            continue
        relative = folder.relative_to(root)
        names = subprocess.check_output(['git','ls-files','--',str(relative)],text=True).splitlines()
        files = {}
        for name in names:
            before = subprocess.check_output(['git','show','HEAD:'+name])
            current = root/name
            files[name] = dict(head_sha256=hashlib.sha256(before).hexdigest(),
                current_sha256=sha(current), unchanged=before==current.read_bytes())
        records[str(relative)] = dict(tracked_file_count=len(files),
            all_tracked_files_unchanged=all(item['unchanged'] for item in files.values()), files=files)
    return dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        current_verification_output_excluded_from_protected_roots=(str(current_output)
            if current_output is not None else None),
        all_protected_publications_unchanged=all(row['all_tracked_files_unchanged'] for row in records.values()),
        publications=records)


def process_application_checks(case, acquired, config):
    expected_q = np.eye(3)*np.deg2rad(float(config['process_sigma_deg_s']))**2
    count = len(acquired['time'])-1
    result = dict(process_covariance_difference=difference(case['process_covariance_rad2_s2'], expected_q),
        expected_process_increment_count=count,
        actual_process_increment_count=int(case['process_noise_application_count']),
        expected_sigma_deg_s=float(config['process_sigma_deg_s']),
        process_is_per_interval_not_continuous_psd=True,
        process_sigma_is_not_rescaled_by_acquisition_period=bool(np.array_equal(
            case['process_covariance_rad2_s2'],expected_q)),
        check_scope='Declared transition model plus saved increment/row ledger; RK4-step invariance also checked by the engine unit tests')
    if 'process_application_count' in case:
        expected = np.asarray(case['acquisition_index'], int)
        result['row_process_application_count_difference'] = difference(
            case['process_application_count'], expected)
    if 'process_increment_times' in case:
        result['increment_times_exact'] = bool(np.array_equal(case['process_increment_times'], acquired['time'][1:]))
    rows_exact = bool('process_application_count' in case and np.array_equal(
        case['process_application_count'],case['acquisition_index']))
    result.update(row_process_application_counts_exact=rows_exact,
        applied_at_native_acquisition_endpoints_only=bool(result.get('increment_times_exact',False)
            and rows_exact and int(case['process_noise_application_count'])==count),
        no_extra_process_increment_for_dense_queries_or_last_tail=bool(rows_exact
            and np.all(case['process_application_count']<=count)
            and int(case['process_noise_application_count'])==count))
    return result


def transport_independent(qmoments, rotation_covariance, increments):
    """Right quaternion multiplication and vec_F(R D) Kronecker action."""
    w,x,y,z = np.asarray(increments).T
    right = np.stack((np.column_stack((w,-x,-y,-z)),np.column_stack((x,w,z,-y)),
        np.column_stack((y,-z,w,x)),np.column_stack((z,y,-x,w))),axis=1)
    qnext = right@qmoments@right.transpose(0,2,1)
    matrices = Rotation.from_quat(increments[:,[1,2,3,0]]).as_matrix()
    action = np.stack([np.kron(matrix.T,np.eye(3)) for matrix in matrices])
    return qnext,action@rotation_covariance@action.transpose(0,2,1)


def last_observation_rule(case, acquired, config, physical, mode):
    """Reconstruct the last Student-t target and physical tail independently."""
    expected_r = np.eye(3)*np.deg2rad(float(config['sigma_deg_s']))**2
    rule = independent_density_rule(case['final_prior_component_weights'],
        case['final_prior_component_omega_means_rad_s'],
        case['final_prior_common_omega_covariance_rad2_s2'],
        case['final_prior_conditional_quaternion_second_moment'],
        case['final_prior_conditional_rotation_covariance_colmajor'],
        case['final_quadrature_nodes_rad_s'],case['final_quadrature_base_weights'],
        case['final_proposal_mean_rad_s'],case['final_proposal_covariance_rad2_s2'],
        case['final_measurement_rad_s'],expected_r,float(config['student_df']))
    rows = np.flatnonzero((case['event_side']==1)&(case['acquisition_index']==len(acquired['time'])-1))
    row = int(rows[-1])
    report = dict(reference='Independent SciPy Gaussian transition-mixture density and multivariate Student-t likelihood divided by the saved Gaussian proposal',
        first_gyro_initializes_once=True,
        native_gyro_count=len(acquired['time']),recursive_noninitial_likelihood_count=len(acquired['time'])-1,
        saved_last_measurement_exact=bool(np.array_equal(case['final_measurement_rad_s'],acquired['gyro'][-1])),
        saved_last_observation_time_exact=bool(case['final_posterior_time']==acquired['time'][-1]),
        mixture_covariance_vs_process_q=difference(case['final_prior_common_omega_covariance_rad2_s2'],
            case['process_covariance_rad2_s2']),
        independent_weight_sum=float(rule['weights'].sum()),
        responsibility_sum_error=rule['responsibility_sum_error'],
        log_predictive_density=float(rule['log_normalizer']),
        posterior_weight_difference=difference(case['final_quadrature_posterior_weights'],rule['weights']),
        conditional_quaternion_moment_difference=difference(case['final_conditional_quaternion_second_moment'],
            rule['quaternion_moments']),
        conditional_rotation_covariance_difference=difference(
            case['final_conditional_rotation_covariance_colmajor'],rule['rotation_covariances']),
        last_observation_joint_moment_differences={key:difference(case[key][row],value)
            for key,value in rule['moments'].items()},
        quadrature_convergence_proven=False,
        numerical_scope='Equality within the saved finite integration rule; not proof of continuous-density convergence')
    tail = case['time'][(case['event_side']==0)&(case['acquisition_index']==len(acquired['time'])-1)]
    report['physical_tail'] = dict(saved_tail_count=len(tail),process_increments_added=0)
    if len(tail):
        native = load('_real_process_verify_tail_flow',SEQ/'native_flow.py')
        flow = native.ControlledFlow(physical,np.array([1.,0.,0.,0.]),
            float(acquired['time'][-1]),float(case['time'][-1]),
            max_step=float(config['max_step_seconds']),mode=mode)
        states = flow.evaluate(case['final_quadrature_nodes_rad_s'],
            np.r_[acquired['time'][-1],tail])
        qmoments,covariance = transport_independent(rule['quaternion_moments'],
            rule['rotation_covariances'],states[-1,:,3:])
        moments = independent_joint(rule['weights'],states[-1,:,:3],qmoments,covariance)
        report['physical_tail'].update(physical_end=float(tail[-1]),
            duration_seconds=float(tail[-1]-acquired['time'][-1]),
            final_joint_moment_differences={key:difference(case[key][-1],value)
                for key,value in moments.items()},
            source='Same native physical RK4, independent conditional right action and total joint moments')
    return report


def run(args):
    begun = time.monotonic()
    output = args.output.resolve()
    raw = load_npz(sparse.ARCHIVE/'input.npz')
    raw_metadata = json.loads((sparse.ARCHIVE/'input.json').read_text())
    source = load_npz(output/'reference_input.npz')
    original = load_npz(sparse.REFINED/'refined_torque_profile.npz')
    frozen = load_npz(output/'frozen_physical_input.npz')
    physical = dict(inertia=frozen['inertia'],torque=SimpleNamespace(
        time=frozen['torque_time'],torque=frozen['torque_applied_body']))
    root_config = json.loads((output/'config.json').read_text())
    report = dict(schema='grape/real-drone-student-process-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), post_run_only=True,
        observations_or_filter_state_changed=False,
        finite_differences_are_diagnostics_not_observation_gates=True,
        source_sha256=sha(__file__), input_archive_sha256=sha(sparse.ARCHIVE/'input.npz'),
        refined_profile_sha256=sha(sparse.REFINED/'refined_torque_profile.npz'),
        frozen_physical_input=dict(
            inertia_vs_archived_refined=difference(frozen['inertia'],original['inertia_body']),
            torque_time_vs_archived=difference(frozen['torque_time'],original['torque_profile_time']),
            actuator_torque_vs_archived=difference(frozen['torque_actuator_body'],original['torque_actuator_body']),
            applied_torque_vs_archived=difference(frozen['torque_applied_body'],original['torque_applied_body']),
            residual_mean_vs_archived=difference(frozen['residual_mean_body'],original['residual_mean_body'])),
        protected_publications=protected_files_against_head(output), cases={})
    for directory in sorted(output.glob('rate_*/sigma_*_deg_s')):
        key = str(directory.relative_to(output))
        config = sparse.case_config(directory,root_config)
        acquired = load_npz(directory/'acquisitions.npz')
        cases = {mode:load_npz(directory/f'bayes_{mode}.npz') for mode in MODELS}
        item = dict(acquisitions=acquisition_checks(acquired,config,raw,raw_metadata), models={})
        for mode, case in cases.items():
            nu = float(config['student_df'])
            covariance = np.eye(3)*np.deg2rad(float(config['sigma_deg_s']))**2
            item['models'][mode] = dict(structure=sparse.structure(case,acquired,raw['time']),
                covariance=helpers.covariance_diagnostics(case),
                process=process_application_checks(case,acquired,config),
                declared_student_parameters=dict(
                    degrees=difference(case['observation_student_df'],np.asarray(nu)),
                    marginal_covariance=difference(case['measurement_covariance_rad2_s2'],covariance),
                    scale=difference(case['observation_student_scale_rad2_s2'],(nu-2.)/nu*covariance)),
                last_observation_density=last_observation_rule(case,acquired,config,physical,mode))
        item['plots'] = sparse.plot_checks(directory,source,cases)
        report['cases'][key] = item
        print('Verified '+key,flush=True)
    report['case_count'] = len(report['cases'])
    if args.test_evidence is not None:
        report['supplied_actual_test_evidence'] = json.loads(args.test_evidence.read_text())
        report['supplied_actual_test_evidence_sha256'] = sha(args.test_evidence)
    report['wall_seconds'] = time.monotonic()-begun
    write_json(output/'verification.json',report)
    helpers.refresh_manifest(output)
    print('Wrote '+str(output/'verification.json'),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=HERE/'publication/bag8_full_gyro_q1')
    parser.add_argument('--test-evidence',type=Path)
    run(parser.parse_args())
