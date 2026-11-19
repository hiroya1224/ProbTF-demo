#!/usr/bin/env python3
"""Post-run checks of selected native gyro readings and their Bayes target.

This diagnostic never accepts/rejects an observation or changes a filter.
Finite integration differences and covariance eigenvalues are reported as
numbers, without a quality threshold. No subsequent truth is required.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
MODELS = ('nonlinear','linear_forced')
TEST_COMMAND = ('env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest '
    'discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/'
    'synthetic_controlled_rotation/sequential_bayes_spikes/tests -v')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value))


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,
                               default=json_value,allow_nan=False)+'\n')


def finite_max(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else None


def difference(actual, expected):
    actual,expected = np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape:
        return dict(actual_shape=actual.shape,expected_shape=expected.shape,shape_matches=False)
    delta = actual-expected
    return dict(shape_matches=True,maximum_absolute_finite_difference=finite_max(delta),
                nonfinite_difference_count=int(np.count_nonzero(~np.isfinite(delta))))


def structure(case, acquired, expected_count):
    count = len(acquired['time'])
    index = np.r_[0,np.repeat(np.arange(1,count),2)]
    sides = np.r_[1,np.tile([-1,1],count-1)]
    factors = np.where(sides==1,index,np.maximum(0,index-1))
    return dict(selected_observation_count=count,expected_selected_count=expected_count,
        requested_selection_count_matches=(count==expected_count),
        expected_saved_rows=2*count-1,actual_saved_rows=len(case['time']),
        expected_prior_rows=count-1,actual_prior_rows=int(np.count_nonzero(case['event_side']==-1)),
        expected_posterior_rows=count,actual_posterior_rows=int(np.count_nonzero(case['event_side']==1)),
        time_rows_exact=bool(np.array_equal(case['time'],acquired['time'][index])),
        event_side_rows_exact=bool(np.array_equal(case['event_side'],sides)),
        acquisition_indices_exact=bool(np.array_equal(case['acquisition_index'],index)),
        likelihood_factor_count_all_rows_exact=bool(np.array_equal(case['target_likelihood_factor_count'],factors)),
        first_reading_likelihood_count=int(case['target_likelihood_factor_count'][0]),
        final_likelihood_count=int(case['target_likelihood_factor_count'][-1]),
        expected_final_likelihood_count=count-1,
        event_times_exact=bool(np.array_equal(case['event_time'],acquired['time'])))


def covariance_diagnostics(case):
    shapes = {'quaternion_second_moment':(4,4),'rotation_mean':(3,3),
        'rotation_covariance_colmajor':(9,9),'omega_mean_rad_s':(3,),
        'omega_covariance_rad2_s2':(3,3),'rotation_omega_covariance_colmajor':(9,3)}
    rows = len(case['time'])
    fields = {name:dict(shape=case[name].shape,expected_shape=(rows,*shape),
        shape_matches=(case[name].shape==(rows,*shape)),
        finite=bool(np.isfinite(case[name]).all()),
        nonfinite_count=int(np.count_nonzero(~np.isfinite(case[name]))))
        for name,shape in shapes.items()}
    result = dict(fields=fields,numerical_eigenvalues_are_diagnostics_not_acceptance_gates=True)
    if not all(item['shape_matches'] and item['finite'] for item in fields.values()):
        result['covariance_diagnostics_skipped']='Missing expected finite array shapes'
        return result
    rotation = case['rotation_covariance_colmajor']
    omega = case['omega_covariance_rad2_s2']
    cross = case['rotation_omega_covariance_colmajor']
    joint = np.concatenate((np.concatenate((rotation,cross),axis=2),
                            np.concatenate((cross.transpose(0,2,1),omega),axis=2)),axis=1)
    moment = case['quaternion_second_moment']
    result.update(minimum_omega_diagonal_rad2_s2=float(np.diagonal(omega,axis1=1,axis2=2).min()),
        minimum_omega_eigenvalue_rad2_s2=float(np.linalg.eigvalsh(omega).min()),
        minimum_joint_rotation_omega_eigenvalue=float(np.linalg.eigvalsh(joint).min()),
        maximum_joint_covariance_asymmetry=finite_max(joint-joint.transpose(0,2,1)),
        maximum_quaternion_second_moment_trace_error=finite_max(np.trace(moment,axis1=1,axis2=2)-1.),
        minimum_quaternion_second_moment_eigenvalue=float(np.linalg.eigvalsh(moment).min()),
        final_omega_covariance_rad2_s2=omega[-1],
        final_omega_two_sigma_deg_s=2*np.rad2deg(np.sqrt(np.diag(omega[-1]))))
    return result


def analytic_linear(case, source, acquired, config, physical, engine):
    # For linear_forced, omega(t;w)=w+a(t). This flow evaluates only the
    # known J/torque input with w=0; it uses no intermediate reference pose.
    flow = engine._adaptive.ControlledFlow(physical,source['initial_quaternion_wxyz'],
        *source['time'],max_step=config['max_step_seconds'],mode='linear_forced')
    offset = flow.evaluate(np.zeros((1,3)),acquired['time'])[:,0,:3]
    initial_readings = acquired['gyro_spike']-offset
    sums = np.vstack((np.zeros(3),np.cumsum(initial_readings,axis=0)))
    index = np.asarray(case['acquisition_index'],int)
    included = np.where(case['event_side']==1,index+1,index)
    mean = sums[included]/included[:,None]+offset[index]
    observation_covariance = np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2
    covariance = observation_covariance[None,:,:]/included[:,None,None]
    mean_delta = case['omega_mean_rad_s']-mean
    covariance_delta = case['omega_covariance_rad2_s2']-covariance
    denominator = np.linalg.norm(covariance,axis=(1,2))
    return dict(reference='Exact forced-linear Gaussian posterior with ALL selected readings; a(t) is known-input zero-W0 flow',
        oracle_or_generated_bias_used=False,
        expected_mean='mean_{j included}(y_j-a(t_j))+a(t)',
        expected_covariance='R / number_of_included_readings; y0 initializes once',
        maximum_all_time_mean_difference_rad_s=finite_max(mean_delta),
        maximum_all_time_mean_difference_deg_s=finite_max(np.rad2deg(mean_delta)),
        per_component_maximum_mean_difference_deg_s=np.max(abs(np.rad2deg(mean_delta)),axis=0),
        maximum_all_time_covariance_difference_rad2_s2=finite_max(covariance_delta),
        maximum_relative_covariance_frobenius_difference=finite_max(
            np.linalg.norm(covariance_delta,axis=(1,2))/denominator),
        final_analytic_mean_rad_s=mean[-1],final_analytic_covariance_rad2_s2=covariance[-1],
        final_mean_difference_rad_s=mean_delta[-1],final_covariance_difference_rad2_s2=covariance_delta[-1],
        finite_differences_are_reported_without_quality_gate=True)


def independent_joint_moments(state, mass):
    omega,quaternion = state[:,:3],state[:,3:]
    rotations = Rotation.from_quat(quaternion[:,[1,2,3,0]]).as_matrix()
    vector = rotations.transpose(0,2,1).reshape(len(state),9)
    rotation_mean = np.einsum('n,nij->ij',mass,rotations)
    omega_mean = mass@omega
    centered_rotation = vector-mass@vector
    centered_omega = omega-omega_mean
    return dict(quaternion_second_moment=np.einsum('n,ni,nj->ij',mass,quaternion,quaternion),
        rotation_mean=rotation_mean,omega_mean_rad_s=omega_mean,
        rotation_covariance_colmajor=np.einsum('n,ni,nj->ij',mass,centered_rotation,centered_rotation),
        omega_covariance_rad2_s2=np.einsum('n,ni,nj->ij',mass,centered_omega,centered_omega),
        rotation_omega_covariance_colmajor=np.einsum('n,ni,nj->ij',mass,centered_rotation,centered_omega))


def full_target_check(case, source, acquired, config, physical, mode, metadata, engine):
    index = int(case['refresh_acquisition_index'][-1])
    finish = len(acquired['time'])-1
    order = int(metadata['quadrature_order_per_axis'])
    center = case['refresh_proposal_mean_rad_s'][-1]
    covariance = case['refresh_proposal_covariance_rad2_s2'][-1]
    nodes,quadrature = engine.gaussian_tensor_points(center,covariance,order)
    flow = engine.FullRateFlow(physical,source['initial_quaternion_wxyz'],*source['time'],
                               max_step=config['max_step_seconds'],mode=mode)
    measurement_covariance = case['measurement_covariance_rad2_s2']
    past,states = flow.prefix_and_block(nodes,acquired['time'],acquired['gyro_spike'],
                                        index,finish,measurement_covariance)
    raw = (np.log(quadrature)
        +engine.gaussian_log_density(nodes,acquired['gyro_spike'][0],measurement_covariance)
        -engine.gaussian_log_density(nodes,center,covariance)+past)
    if index:
        raw += engine.gaussian_log_density(states[0,:,:3],acquired['gyro_spike'][index],measurement_covariance)
    anchor = raw-logsumexp(raw)
    for acquired_index in range(index+1,finish+1):
        raw += engine.gaussian_log_density(states[acquired_index-index,:,:3],
                                           acquired['gyro_spike'][acquired_index],measurement_covariance)
    final_logs = raw-logsumexp(raw)
    mass = np.exp(final_logs)
    moments = independent_joint_moments(states[-1],mass)
    saved_anchor_logs = (case['last_refresh_log_weights'] if 'last_refresh_log_weights' in case
                         else case['refresh_log_weights'][-1])
    return dict(last_refresh_acquisition_index=index,last_refresh_time_seconds=float(acquired['time'][index]),
        original_prior_used_once=True,reevaluated_likelihood_factors=finish,
        future_truth_bias_or_spike_labels_used=False,
        last_refresh_normalized_log_weight_comparison=difference(saved_anchor_logs,anchor),
        final_posterior_weight_sum=float(mass.sum()),final_float_underflow_zero_weight_count=int(np.count_nonzero(mass==0.)),
        final_saved_moments_vs_independent_all_factor_moments={
            name:difference(case[name][-1],value) for name,value in moments.items()},
        source_of_rotations='Same native quaternion candidates, converted independently with scipy Rotation',
        numerical_differences_are_diagnostics_not_filter_gates=True)


def plot_checks(output, cases, acquired, sensor):
    path = output/'rate_comparison_plotting.json'
    if not path.exists():
        return dict(available=False,reason='Final figures have not been rendered')
    record = json.loads(path.read_text())
    settings = record['settings']
    result = dict(available=True,recorded_native_count=settings['native_reading_count'],
        recorded_acquisition_count=settings['selected_acquisition_count'],
        native_and_acquisition_counts_match=(settings['native_reading_count']==len(sensor['time'])
            and settings['selected_acquisition_count']==len(acquired['time'])),
        all_native_samples_drawn=record.get('all_native_samples_drawn'),
        all_saved_moment_rows_drawn=record.get('all_saved_moment_rows_drawn'),
        smoothing=record.get('smoothing'),covariance_adjustment=record.get('covariance_adjustment'))
    result['models'] = {mode:dict(
        moment_rows_match=(settings['stored_moment_rows'][mode]==len(case['time'])),
        prior_rows_match=(settings['stored_prior_rows'][mode]==int(np.count_nonzero(case['event_side']==-1))),
        posterior_rows_match=(settings['stored_posterior_rows'][mode]==int(np.count_nonzero(case['event_side']==1))),
        final_factor_count_matches=(settings['final_recorded_likelihood_factor_count'][mode]
                                   ==int(case['target_likelihood_factor_count'][-1])))
        for mode,case in cases.items()}
    return result


def refresh_manifest(output):
    files = [dict(path=str(path.relative_to(output)),bytes=path.stat().st_size,sha256=sha(path))
        for path in sorted(output.rglob('*')) if path.is_file()
        and 'cache' not in path.relative_to(output).parts and path.name!='manifest.json']
    write_json(output/'manifest.json',dict(files=files,
        bytes_excluding_manifest=sum(item['bytes'] for item in files),
        raw_bag_or_intermediate_cache_included=False))
    return files


def run(args):
    started = time.monotonic()
    output,input_directory = args.output.resolve(),args.input.resolve()
    config = json.loads((output/'config.json').read_text())
    source = load_npz(input_directory/'source_input.npz')
    sensor = load_npz(input_directory/'sensor_input.npz')
    acquired = load_npz(output/'acquisitions.npz')
    summary_path = output/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cases,metadata = {},{}
    for mode in MODELS:
        if args.cache_order is None:
            cases[mode] = load_npz(output/f'bayes_{mode}.npz')
            metadata[mode] = summary['engine_metadata'][mode]
        else:
            stem = output/'cache'/f'order{args.cache_order}_{mode}'
            cases[mode] = load_npz(stem.with_suffix('.npz'))
            metadata[mode] = json.loads(stem.with_name(stem.name+'_metadata.json').read_text())
    engine = load_module('_full_rate_verification_engine','full_rate_engine.py')
    shared = load_module('_full_rate_verification_shared','run.py')
    physical = dict(inertia=source['inertia_body_kg_m2'],
        torque=shared.TorqueProfile(source['torque_time'],source['torque_body_nm']))
    native_indices = np.asarray(acquired['native_array_index'],int)
    rate = float(config.get('acquisition_rate_hz',config.get('rate_hz',10.)))
    native_rate = 1./np.median(np.diff(sensor['time']))
    stride = int(round(native_rate/rate))
    expected_indices = np.arange(0,len(sensor['time']),stride)
    equality = {name:bool(np.array_equal(acquired[name],sensor[name][native_indices]))
        for name in ('time','stamp_ns','record_ns','source_index','gyro_spike','spike_mask','spike_vector_rad_s')}
    equality['native_array_index_selection_exact'] = bool(np.array_equal(native_indices,expected_indices))
    saved_hashes = summary.get('input_file_sha256',{})
    current_hashes = {name:sha(input_directory/name) for name in saved_hashes}
    report = dict(schema='grape/selected-rate-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='final publication' if args.cache_order is None else f'completed cache order {args.cache_order}',
        post_run_only=True,filter_state_or_observations_modified=False,
        finite_numerical_differences_do_not_gate_filter_updates=True,
        verifier_sha256=sha(__file__),
        input_file_equality=equality,
        selected_input=dict(native_recording_count=len(sensor['time']),
            requested_acquisition_rate_hz=rate,native_rate_hz=native_rate,
            native_index_stride=stride,selected_observation_count=len(native_indices),
            expected_selected_count=len(expected_indices),
            all_native_readings_included=bool(len(native_indices)==len(sensor['time'])),
            selected_native_indices=native_indices,
            selection='Fixed stride through unchanged saved native readings, including t=0 and t=20; no interpolation or regeneration'),
        input_value_sha256=hashlib.sha256(np.ascontiguousarray(acquired['gyro_spike']).tobytes()).hexdigest(),
        original_1hz_files=dict(initial_hashes_recorded_in_run_summary=saved_hashes,
            current_hashes=current_hashes,
            unchanged_since_run_start=(saved_hashes==current_hashes) if saved_hashes else None,
            no_initial_hash_claim_for_other_files=True),
        original_1hz_config=json.loads((input_directory/'config.json').read_text()) if (input_directory/'config.json').exists() else None,
        source_sha256={path.name:sha(path) for pattern in ('full_rate*','rate_comparison*')
            for path in sorted(HERE.glob(pattern))
            if path.suffix in ('.py','.cpp')},
        unchanged_included_physics_sha256=sha(HERE/'native_flow.cpp'),
        tests=dict(command=TEST_COMMAND,passed=36,exit_code=0,wall_seconds=1.372,
            provenance='Execution reported by root agent for completed session 72726; this verifier does not rerun the suite'),
        models={})
    for mode in MODELS:
        print(f'Verify {mode}: {len(acquired["time"])} selected original readings at {rate:g} Hz',flush=True)
        case = cases[mode]
        item = dict(structure=structure(case,acquired,len(expected_indices)),covariance=covariance_diagnostics(case),
            measurement_covariance_vs_declared_config=difference(
                case['measurement_covariance_rad2_s2'],
                np.eye(3)*np.deg2rad(config['assumed_sigma_deg_s'])**2),
            recorded_engine_metadata=metadata[mode],
            recorded_native_source_hash_matches=(metadata[mode]['native_source_sha256']==sha(HERE/'full_rate_flow.cpp')),
            recorded_included_physics_hash_matches=(metadata[mode]['included_physical_source_sha256']==sha(HERE/'native_flow.cpp')))
        if mode=='linear_forced':
            item['analytic_all_reading_reference'] = analytic_linear(case,source,acquired,config,physical,engine)
        if not args.skip_refresh_weights:
            item['independent_full_target_check'] = full_target_check(case,source,acquired,config,physical,mode,metadata[mode],engine)
        report['models'][mode] = item
    report['both_models_same_observation_covariance'] = bool(np.array_equal(
        cases['nonlinear']['measurement_covariance_rad2_s2'],cases['linear_forced']['measurement_covariance_rad2_s2']))
    report['plots'] = plot_checks(output,cases,acquired,sensor)
    files = [dict(path=str(path.relative_to(output)),bytes=path.stat().st_size)
        for path in output.rglob('*') if path.is_file() and 'cache' not in path.relative_to(output).parts]
    report['file_sizes'] = dict(maximum_bytes=max(item['bytes'] for item in files),
        all_publication_files_below_50000000_bytes=all(item['bytes']<50_000_000 for item in files),
        files=sorted(files,key=lambda item:item['path']),cache_excluded=True)
    report['wall_seconds'] = time.monotonic()-started
    destination = (output/'verification.json' if args.cache_order is None
                   else output/'cache'/f'verification_order{args.cache_order}.json')
    write_json(destination,report)
    if args.cache_order is None:
        refresh_manifest(output)
    print('Wrote '+str(destination),flush=True)
    for mode,item in report['models'].items():
        print(mode,json.dumps(item['structure'],default=json_value),flush=True)
        if 'analytic_all_reading_reference' in item:
            print('Linear analytic maximum mean difference [deg/s]:',
                item['analytic_all_reading_reference']['maximum_all_time_mean_difference_deg_s'],flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='Completed selected-rate output directory')
    parser.add_argument('--input',type=Path,default=HERE/'frequent_publication',
                        help='Unchanged original 1 Hz publication containing saved native sensor inputs')
    parser.add_argument('--cache-order',type=int,
                        help='Diagnose this completed cache order before final publication is ready')
    parser.add_argument('--skip-refresh-weights',action='store_true',
                        help='Skip the extra full-history model-flow diagnostic')
    run(parser.parse_args())
