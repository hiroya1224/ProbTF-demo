#!/usr/bin/env python3
"""Compare deterministic quadrature orders at already saved attitude laws.

This is a fixed-state numerical check. It never reruns the recursive estimator,
refines an estimated mode, changes an outcome, or uses a true attitude to place
integration nodes. Agreement of a few orders does not prove global coverage of
broad or multimodal densities or convergence of the whole recursive filter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'ros/core/probtf_core/src'))

import numpy as np

from translational_attitude_law import ExponentialAttitudeLaw

METHODS = ('joint_marg', 'joint_imu', 'gravity_marg')


def _indices(count, stride, maximum, include_indices=()):
    selected = np.unique(np.r_[np.arange(0, count, stride), count - 1])
    if maximum and len(selected) > maximum:
        selected = selected[np.linspace(0, len(selected) - 1, maximum, dtype=int)]
    additional=np.asarray([i for i in include_indices if i<count],dtype=int)
    return np.unique(np.r_[selected,additional])


def _pitch(integral):
    pitch = np.arcsin(np.clip(-integral['R'][:, 2, 0], -1., 1.))
    mean = float(integral['weights'] @ pitch)
    variance = float(integral['weights'] @ (pitch - mean)**2)
    return dict(pitch_mean_deg=float(np.rad2deg(mean)),
                pitch_std_deg=float(np.rad2deg(np.sqrt(variance))))


def _difference(value, reference, gravity_norm2, magnetic_norm2):
    result = {'logZ_difference_absolute':abs(float(value['logZ'])-float(reference['logZ'])),
              'relative_logZ_difference_absolute':abs(float(value['relative_logZ'])-float(reference['relative_logZ']))}
    for key in ('Mqq', 'Ngg', 'Nbb'):
        difference = value[key] - reference[key]
        result[key + '_frobenius'] = float(np.linalg.norm(difference))
        result[key + '_maximum_absolute_entry'] = float(np.max(np.abs(difference)))
    result['Ngg_normalized_frobenius'] = result['Ngg_frobenius'] / gravity_norm2
    result['Nbb_normalized_frobenius'] = result['Nbb_frobenius'] / magnetic_norm2
    current_pitch, reference_pitch = _pitch(value), _pitch(reference)
    result['pitch_mean_difference_deg'] = abs(current_pitch['pitch_mean_deg'] - reference_pitch['pitch_mean_deg'])
    result['pitch_std_difference_deg'] = abs(current_pitch['pitch_std_deg'] - reference_pitch['pitch_std_deg'])
    return result


def _aggregate(rows):
    comparisons = {}
    names = sorted({name for row in rows for name in row['differences_from_highest_order']})
    for name in names:
        available = [row['differences_from_highest_order'][name] for row in rows
                     if name in row['differences_from_highest_order']]
        comparisons[name] = {key: max(value[key] for value in available)
                             for key in available[0]} if available else {}
    return dict(checked_saved_laws=len(rows), maximum_differences=comparisons)


def validate(run_root, orders=(7, 9, 11), stride=200, max_states=0, cases=None, global_refinement=2, include_indices=()):
    run_root = Path(run_root).resolve()
    orders = tuple(sorted(set(int(value) for value in orders)))
    if len(orders) < 2 or any(value < 3 for value in orders):
        raise ValueError('At least two distinct quadrature orders >=3 are required')
    if stride < 1 or max_states < 0 or global_refinement < 1:
        raise ValueError('stride/global_refinement must be positive and max_states nonnegative')
    if any(int(i)!=i or i<0 for i in include_indices):
        raise ValueError('Explicit saved-state indices must be nonnegative integers')
    files = sorted(run_root.rglob('evaluated_timeseries.npz'))
    source = HERE / 'translational_attitude_law.py'
    result = dict(schema='translational-attitude-fixed-state-quadrature-check/v1',
        scope='same saved probability law and saved reference; integration order only',
        estimator_rerun=False, estimated_modes_refined=False,
        recursive_filter_convergence_claimed=False,
        global_multimodal_coverage_proved=False,
        limitations='Finite local GH and global Haar multiple-importance quadrature can miss sub-grid remote peaks or broad features; matching orders at fixed saved states is not proof of full recursive convergence.',
        run_root=str(run_root), quadrature_orders=list(orders), reference_order=max(orders),
        sampling='deterministic uniform-in-time saved-state selection; no Monte Carlo',
        stride=int(stride), max_states_per_case=int(max_states),
        additionally_requested_saved_indices=list(include_indices),
        global_refinement=int(global_refinement),
        global_refinement_scope='same GH order, double radial/angle Haar resolution at lowest and highest GH orders',
        law_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases={}, failures=[], rows=[])
    started = time.monotonic()
    for archive in files:
        case = archive.parent.name
        if cases and case not in cases:
            continue
        settings_path = archive.parent / 'simulation_settings.json'
        if not settings_path.exists():
            result['failures'].append(dict(case=case, source=str(archive),
                reason='missing sibling simulation_settings.json'))
            continue
        settings = json.loads(settings_path.read_text())
        g = np.asarray(settings['gravity_world_m_s2'], float)
        b = np.asarray(settings['magnetic_field_world'], float)
        identifier = str(archive.parent.relative_to(run_root))
        case_rows = []
        failures_before = len(result['failures'])
        with np.load(archive, allow_pickle=False) as compressed:
            arrays = {key: compressed[key] for key in ('time_seconds','A','Wg','Wb','mode_wxyz')}
            for key in ('projection_moment_residual','projection_initial_moment_residual'):
                if key in compressed:arrays[key]=compressed[key]
            count = len(arrays['time_seconds'])
            selected = _indices(count, stride, max_states, include_indices)
            for index in selected:
                stamp = float(arrays['time_seconds'][index])
                for method_index, method in enumerate(METHODS):
                    identity = dict(case=case, source=str(archive), index=int(index),
                                    time_seconds=stamp, method=method)
                    try:
                        law = ExponentialAttitudeLaw(arrays['A'][index, method_index],
                            arrays['Wg'][index, method_index], arrays['Wb'][index, method_index], g, b)
                        reference = np.asarray(arrays['mode_wxyz'][index, method_index], float)
                        # Recompute the exact Hessian at the SAME saved reference.
                        # No fallback is used: a non-SPD required proposal is
                        # a recorded validation failure, not a variance change.
                        proposal = law.proposal_covariance(reference)
                        integrals = {order: law.integrate(reference, proposal, order=order)
                                     for order in orders}
                        if any(not np.isfinite(value[key]).all() for value in integrals.values()
                               for key in ('weights','Mqq','Ngg','Nbb','R')):
                            raise FloatingPointError('Non-finite quadrature output')
                        baseline = integrals[max(orders)]
                        global_checks={}
                        for check_order in sorted(set((min(orders),max(orders)))):
                            original=integrals[check_order]
                            refined=law.integrate(reference,proposal,order=check_order,
                                global_radial=original['global_radial_order']*global_refinement,
                                global_angles=original['global_angle_order']*global_refinement)
                            if any(not np.isfinite(refined[key]).all() for key in ('weights','Mqq','Ngg','Nbb','R')):
                                raise FloatingPointError('Non-finite refined global quadrature output')
                            global_checks[str(check_order)]=dict(
                                original_global_radial=original['global_radial_order'],
                                original_global_angles=original['global_angle_order'],
                                refined_global_radial=refined['global_radial_order'],
                                refined_global_angles=refined['global_angle_order'],
                                differences=_difference(original,refined,float(g@g),float(b@b)))
                        row = dict(identity,
                            global_resolution_checks=global_checks,
                            integration_rule_by_order={str(order):{key:value[key] for key in
                                ('global_radial_order','global_angle_order','local_mixture_weight','quadrature_nodes')}
                                for order,value in integrals.items()},
                            proposal_eigenvalues=np.linalg.eigvalsh(proposal).tolist(),
                            posterior_pitch_by_order={str(order): _pitch(value)
                                                      for order, value in integrals.items()},
                            differences_from_highest_order={str(order): _difference(value, baseline,
                                float(g @ g), float(b @ b)) for order, value in integrals.items()
                                if order != max(orders)})
                        for check_order,check in global_checks.items():
                            row['differences_from_highest_order']['global_refinement_at_GH_'+check_order]=check['differences']
                        for key in ('projection_moment_residual','projection_initial_moment_residual'):
                            if key in arrays:row['saved_'+key]=float(arrays[key][index,method_index])
                        case_rows.append(row)
                    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                        result['failures'].append(dict(identity,
                            reason=type(error).__name__ + ': ' + str(error)))
        result['rows'].extend(case_rows)
        result['cases'][identifier] = dict(case=case, source=str(archive),
            source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            saved_samples=count, selected_state_indices=selected.tolist(),
            failures=len(result['failures'])-failures_before,
            aggregate=_aggregate(case_rows),
            methods={method: _aggregate([row for row in case_rows if row['method']==method])
                     for method in METHODS})
    result['aggregate'] = _aggregate(result['rows'])
    result['aggregate_by_method']={method:_aggregate([row for row in result['rows']
        if row['method']==method]) for method in METHODS}
    result['elapsed_seconds'] = time.monotonic() - started
    result['status'] = ('no_completed_saved_laws_found' if not result['cases'] else
                        'fixed_state_check_complete_with_failures' if result['failures'] else
                        'fixed_state_check_complete')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=200,
                        help='Saved-state stride; 200 is one second for 200 Hz results')
    parser.add_argument('--max-states', type=int, default=0,
                        help='Optional uniform-in-time cap per case (0 means no cap)')
    parser.add_argument('--orders', type=int, nargs='+', default=[7, 9, 11])
    parser.add_argument('--cases', nargs='+', default=None)
    parser.add_argument('--global-refinement',type=int,default=2)
    parser.add_argument('--include-indices',type=int,nargs='*',default=[],
                        help='Additional original saved indices, included after the uniform cap; indices beyond an archive are ignored')
    args = parser.parse_args()
    result = validate(args.run_root,args.orders,args.stride,args.max_states,args.cases,args.global_refinement,args.include_indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:result[key] for key in ('status','aggregate','elapsed_seconds')}, indent=2))
    print('Validation failures:', len(result['failures']))
    print('Saved fixed-state quadrature check:', args.output)


if __name__ == '__main__':
    main()
