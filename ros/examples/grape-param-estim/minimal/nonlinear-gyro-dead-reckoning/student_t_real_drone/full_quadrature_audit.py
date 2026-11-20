#!/usr/bin/env python3
"""Read-only quadrature diagnostics for the completed native-gyro cases.

This does not fit proposals or rerun inference. At selected saved times it
integrates the same causal target under the saved GH11 numerical proposal at
several tensor orders. Finite differences remain diagnostics, never gates.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load('_real_full_audit_input', HERE/'input.py')
dense = load('_real_full_audit_engine', HERE/'full_dense_engine.py')
plotting = load('_real_full_audit_plots', HERE/'plotting.py')
MOMENTS = ('quaternion_second_moment', 'rotation_mean', 'rotation_covariance_colmajor',
    'omega_mean_rad_s', 'omega_covariance_rad2_s2', 'rotation_omega_covariance_colmajor',
    'rpy_proxy_rad', 'rpy_covariance_delta_rad2', 'initial_omega_mean_rad_s',
    'initial_omega_covariance_rad2_s2')


def saved_proposal_parameters(case, row):
    # The engine publishes a prior row BEFORE breaking to refresh at that
    # acquisition. Hence prior i uses the last refresh <=i-1 (its factor
    # count), while posterior i uses the last refresh <=i. GH11 baseline
    # reconstruction below independently checks this choice against saved
    # moments instead of assuming that metadata alone establishes agreement.
    count = int(case['target_likelihood_factor_count'][row])
    refresh = int(np.searchsorted(case['refresh_acquisition_index'], count, side='right')-1)
    covariance = case['refresh_proposal_covariance_rad2_s2'][refresh]
    return dict(refresh_acquisition_index=int(case['refresh_acquisition_index'][refresh]),
        center_rad_s=case['refresh_proposal_mean_rad_s'][refresh].tolist(),
        covariance_rad2_s2=covariance.tolist(),
        center_deg_s=np.rad2deg(case['refresh_proposal_mean_rad_s'][refresh]).tolist(),
        standard_deviation_deg_s=np.rad2deg(np.sqrt(np.diag(covariance))).tolist())


def reevaluate(data, physical, case, row, order, mode, max_step=.002):
    """Integrate prior times all arrived likelihoods / saved causal proposal."""
    stamp = float(case['time'][row])
    count = int(case['target_likelihood_factor_count'][row])
    refresh = int(np.searchsorted(case['refresh_acquisition_index'], count, side='right')-1)
    anchor = int(case['refresh_acquisition_index'][refresh])
    center = case['refresh_proposal_mean_rad_s'][refresh]
    proposal_covariance = case['refresh_proposal_covariance_rad2_s2'][refresh]
    marginal = case['measurement_covariance_rad2_s2']
    nodes, quadrature = dense.engine.gaussian_tensor_points(center, proposal_covariance, order)
    acquired = np.asarray(data['sparse_gyro_time'])
    observed = np.asarray(data['sparse_gyro'])
    flow = dense.engine.FullRateFlow(physical, data['initial_quaternion_wxyz'],
                                   *data['time'], max_step=max_step, mode=mode)
    if stamp == acquired[count]:
        # The latest factor is evaluated once at its physically propagated
        # state. Native prefix j<count omits that factor until this step.
        query = acquired[:count+1]
        prefix = count
    else:
        query = np.r_[acquired[:count+1], stamp]
        prefix = count+1
    past, states = flow.prefix_and_block(nodes, query, observed, prefix=prefix,
        finish=len(query)-1, covariance=marginal, student_df=4.)
    raw = (np.log(quadrature)+dense.engine.gaussian_log_density(nodes, observed[0], marginal)
        -dense.engine.gaussian_log_density(nodes, center, proposal_covariance)+past)
    state = states[-1]
    if prefix == count and count:
        raw += dense.engine.student_t_log_density(state[:, :3], observed[count], marginal, 4.)
    normalizer = float(logsumexp(raw))
    mass = np.exp(raw-normalizer)
    moment = dense.engine.summarize_joint(state[:, 3:], state[:, :3], mass, initial_nodes=nodes)
    return dict(order=int(order), node_count=len(nodes), likelihood_factor_count=count,
        saved_proposal_refresh_acquisition_index=anchor,
        saved_proposal_mean_rad_s=center.tolist(),
        saved_proposal_covariance_rad2_s2=proposal_covariance.tolist(),
        log_normalizer=normalizer, numerical_node_effective_count=float(1/np.sum(mass**2)),
        numerical_node_maximum_mass=float(np.max(mass)),
        moments={key:np.asarray(moment[key]).tolist() for key in MOMENTS},
        maximum_absolute_difference_from_saved_GH11={key:float(np.max(np.abs(
            np.asarray(moment[key])-case[key][row]))) for key in MOMENTS},
        native_step_count=int(flow.total_steps))


def run(args):
    begun = time.monotonic()
    source, acquired, data, physical, provenance = inputs.load_full_inputs()
    cache = args.directory/'cache'
    cases = {order:{mode:dict(np.load(cache/f'order{order}_{mode}.npz', allow_pickle=False))
        for mode in plotting.MODELS} for order in (7, 11)}
    report = dict(schema='grape/full-native-causal-quadrature-audit/v1',
        directory=str(args.directory.resolve()), observation_count=len(acquired['time']),
        time_bounds=data['time'].tolist(), proposal_reoptimized=False,
        full_engine_rerun=False, finite_differences_are_diagnostics_not_gates=True,
        covariance_or_posterior_modified=False,
        limitations='Higher tensor orders under one saved proposal test local quadrature only. They do not establish that this proposal resolves all separated modes or that the complete online sequence has converged.',
        numerical_node_effective_count_definition='1 / sum(normalized quadrature-importance mass squared); numerical concentration diagnostic, not a count of Monte Carlo samples.',
        fixed_501_metrics={}, models={})
    for order in (7, 11):
        _, metrics = plotting._diagnostics(source, cases[order])
        report['fixed_501_metrics'][str(order)] = metrics
    for mode in plotting.MODELS:
        coarse, fine = cases[7][mode], cases[11][mode]
        if not np.array_equal(coarse['time'], fine['time']):
            raise ValueError('This direct saved-row comparison requires matching physical clocks')
        difference = (fine['rpy_proxy_rad']-coarse['rpy_proxy_rad']+np.pi)%(2*np.pi)-np.pi
        worst, axis = np.unravel_index(np.argmax(abs(difference)), difference.shape)
        rotation_difference = np.rad2deg((Rotation.from_euler('xyz', coarse['rpy_proxy_rad']).inv()
            *Rotation.from_euler('xyz', fine['rpy_proxy_rad'])).magnitude())
        records = []
        for label, row in (('worst_rpy_row', int(worst)), ('final_row', len(fine['time'])-1)):
            entry = dict(label=label, row=row, time=float(fine['time'][row]),
                event_side=int(fine['event_side'][row]),
                acquisition_index=int(fine['acquisition_index'][row]),
                included_noninitial_factor_count=int(fine['target_likelihood_factor_count'][row]),
                GH7_rpy_proxy_deg=np.rad2deg(coarse['rpy_proxy_rad'][row]).tolist(),
                GH11_rpy_proxy_deg=np.rad2deg(fine['rpy_proxy_rad'][row]).tolist(),
                saved_proposals_by_online_order={'7':saved_proposal_parameters(coarse, row),
                                                '11':saved_proposal_parameters(fine, row)},
                GH7_GH11_so3_proxy_difference_deg=float(rotation_difference[row]),
                GH7_GH11_joint_moment_maximum_absolute_differences={key:float(np.max(np.abs(
                    fine[key][row]-coarse[key][row]))) for key in MOMENTS},
                same_GH11_proposal_evaluations=[])
            for order in args.orders:
                record = reevaluate(data, physical, fine, row, order, mode)
                entry['same_GH11_proposal_evaluations'].append(record)
                print(mode, label, 'GH', order, 'logZ', record['log_normalizer'],
                    'ESS', record['numerical_node_effective_count'],
                    'omega', record['moments']['omega_mean_rad_s'], flush=True)
            finest = entry['same_GH11_proposal_evaluations'][-1]
            baseline = [record for record in entry['same_GH11_proposal_evaluations']
                        if record['order'] == 11]
            if baseline:
                entry['GH11_baseline_reconstruction_maximum_absolute_differences'] = \
                    baseline[0]['maximum_absolute_difference_from_saved_GH11']
            for record in entry['same_GH11_proposal_evaluations']:
                record['difference_from_highest_requested_order'] = dict(
                    log_normalizer=record['log_normalizer']-finest['log_normalizer'],
                    moment_maximum_absolute={key:float(np.max(np.abs(
                        np.asarray(record['moments'][key])-np.asarray(finest['moments'][key]))))
                        for key in MOMENTS})
            records.append(entry)
        report['models'][mode] = dict(maximum_saved_rpy_component_difference_deg=float(
                np.rad2deg(abs(difference[worst, axis]))), worst_component=('roll','pitch','yaw')[axis],
            maximum_saved_so3_proxy_difference_deg=float(rotation_difference.max()),
            maximum_saved_omega_component_difference_deg_s=float(np.rad2deg(np.max(np.abs(
                fine['omega_mean_rad_s']-coarse['omega_mean_rad_s'])))),
            GH7_proposal_refresh_count=len(coarse['refresh_acquisition_index']),
            GH11_proposal_refresh_count=len(fine['refresh_acquisition_index']),
            selected_rows=records)
    report['wall_seconds'] = time.monotonic()-begun
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['case_sha256'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(cache.glob('order*_*.npz'))}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
        default=HERE/'publication/bag8_full_gyro/rate_full_native/sigma_2p5_deg_s')
    parser.add_argument('--output', type=Path, default=Path('/tmp/probtf-full-quadrature-audit.json'))
    parser.add_argument('--orders', nargs='+', type=int, default=[7, 11, 17, 25])
    run(parser.parse_args())
