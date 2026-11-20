"""Dense causal predictions grouped by an unchanged numerical proposal.

Observation rows belong to the existing full-rate engine and are retained
bit for bit. Added predictions reuse its causal proposal until the next
refresh, evaluating the original initial law times every arrived likelihood.
Inserting physical query times changes only finite RK4 segmentation, not the
candidate law, observations, or any covariance convention.
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_real_drone_dense_reference', HERE/'dense_engine.py')
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
engine = reference.engine
ROW_FIELDS = reference.ROW_FIELDS


def predict_full_dense(data, physical, sigma_deg_s=5., order=11, max_step=.002,
                       mode='nonlinear', student_df=4., output_dt=.01,
                       refresh_seconds=.02, **engine_options):
    """Keep full-rate updates and add physical predictions through the end.

    ``event_side=0`` marks a prediction with no new reading. Its index and
    likelihood count identify the last reading included, excluding the first
    reading already used once to initialize N(W0; y0, R). Gaussian proposals
    are integration measures only, never replacements for the posterior.
    """
    begun = time.monotonic()
    output_dt = float(output_dt)
    if not np.isfinite(output_dt) or output_dt <= 0.:
        raise ValueError('A finite positive dense output interval is required')
    original = engine.predict_full_rate(data, physical, sigma_deg_s=sigma_deg_s,
        student_df=student_df, order=order, max_step=max_step, mode=mode,
        refresh_seconds=refresh_seconds, **engine_options)
    start, end = np.asarray(data['time'], float)
    acquired = np.asarray(data['sparse_gyro_time'], float)
    observed = np.asarray(data['sparse_gyro'], float)
    covariance = original['measurement_covariance_rad2_s2']
    count = int(np.floor((end-start)/output_dt))
    grid = start+np.arange(count+1, dtype=float)*output_dt
    grid = np.unique(np.r_[grid[grid <= end], end])
    stamps = grid[~np.isin(grid, acquired)]
    last_acquisition = np.searchsorted(acquired, stamps, side='right')-1
    refresh_indices = original['refresh_acquisition_index']
    group_for_stamp = np.searchsorted(refresh_indices, last_acquisition, side='right')-1
    posterior_rows = np.flatnonzero(original['event_side'] == 1)
    extra = {key: [] for key in ROW_FIELDS}
    flow = engine.FullRateFlow(physical, data['initial_quaternion_wxyz'],
                               start, end, max_step=max_step, mode=mode)
    groups = []
    intervals = []
    log_observation = (lambda values, mean: engine.gaussian_log_density(values, mean, covariance)) \
        if student_df is None else (lambda values, mean: engine.student_t_log_density(
            values, mean, covariance, student_df))
    for refresh in np.unique(group_for_stamp):
        selected = group_for_stamp == refresh
        added_times = stamps[selected]
        used_indices = last_acquisition[selected]
        anchor = int(refresh_indices[refresh])
        last_needed = int(used_indices[-1])
        center = original['refresh_proposal_mean_rad_s'][refresh]
        proposal_covariance = original['refresh_proposal_covariance_rad2_s2'][refresh]
        nodes, quadrature = engine.gaussian_tensor_points(center, proposal_covariance, order)
        future_observation_times = acquired[anchor+1:last_needed+1]
        after_anchor = np.unique(np.r_[future_observation_times, added_times])
        query = np.r_[acquired[:anchor+1], after_anchor]
        past, states = flow.prefix_and_block(nodes, query, observed,
            prefix=anchor+1, finish=len(query)-1, covariance=covariance,
            student_df=student_df)
        logs = (np.log(quadrature)
            +engine.gaussian_log_density(nodes, observed[0], covariance)
            -engine.gaussian_log_density(nodes, center, proposal_covariance)+past)
        normalizer = float(logsumexp(logs))
        if not np.isfinite(normalizer):
            raise FloatingPointError('Undefined grouped dense posterior normalizer')
        logs -= normalizer
        included = anchor
        last_rpy = original['rpy_proxy_rad'][posterior_rows[anchor]].copy()
        for stamp, state in zip(after_anchor, states):
            next_index = included+1
            if next_index <= last_needed and stamp == acquired[next_index]:
                logs += log_observation(state[:, :3], observed[next_index])
                normalizer = float(logsumexp(logs))
                if not np.isfinite(normalizer):
                    raise FloatingPointError('Undefined grouped dense observation normalizer')
                logs -= normalizer
                included = next_index
                # Match the original causal angle branch without altering its
                # moment or observation output in any way.
                last_rpy = original['rpy_proxy_rad'][posterior_rows[included]].copy()
                continue
            moment = engine.summarize_joint(state[:, 3:], state[:, :3], np.exp(logs),
                                            initial_nodes=nodes)
            angle = moment['rpy_proxy_rad']
            angle = angle+2.*np.pi*np.round((last_rpy-angle)/(2.*np.pi))
            moment['rpy_proxy_rad'] = angle
            last_rpy = angle
            moment.update(time=float(stamp), event_side=0, acquisition_index=included,
                          target_likelihood_factor_count=included)
            for key in ROW_FIELDS:
                extra[key].append(np.asarray(moment[key]).copy())
        groups.append(dict(reused_refresh_acquisition_index=anchor,
            last_included_acquisition_index=last_needed,
            first_prediction_time=float(added_times[0]),
            last_prediction_time=float(added_times[-1]), prediction_count=len(added_times),
            native_prefix_evaluations=1))
        for index in np.unique(used_indices):
            interval_stamps = added_times[used_indices == index]
            intervals.append(dict(last_included_acquisition_index=int(index),
                last_included_acquisition_time=float(acquired[index]),
                reused_refresh_acquisition_index=anchor,
                first_prediction_time=float(interval_stamps[0]),
                last_prediction_time=float(interval_stamps[-1]),
                prediction_count=len(interval_stamps),
                included_noninitial_likelihood_count=int(index)))
    result = dict(original)
    old_count = len(original['time'])
    added_count = len(extra['time'])
    if added_count:
        permutation = np.argsort(np.r_[original['time'], np.stack(extra['time'])], kind='stable')
        for key in ROW_FIELDS:
            combined = np.concatenate((original[key], np.stack(extra[key])), axis=0)
            result[key] = combined[permutation]
        result['observation_row_indices'] = np.argsort(permutation)[:old_count]
    else:
        result['observation_row_indices'] = np.arange(old_count)
    result['dense_output_grid_time'] = grid
    result['dense_interval_records'] = intervals
    result['dense_group_records'] = groups
    result['metadata'] = dict(original['metadata'])
    result['metadata'].update(dense_output=dict(output_dt_seconds=output_dt,
        physical_start_time=float(start), physical_end_time=float(end),
        grid_count=len(grid), added_prediction_count=added_count,
        original_observation_row_count=old_count,
        original_observation_rows_retained_exactly=True,
        event_side_meaning={'-1': 'original observation prior',
                            '0': 'physical prediction without a new observation',
                            '1': 'original observation posterior'},
        final_acquisition_time=float(acquired[-1]),
        final_unobserved_prediction_seconds=float(end-acquired[-1]),
        posterior_target='original N(W0;y0,R) times every likelihood through last included acquisition, divided by the same causal numerical proposal',
        future_observations_used=False, interpolation_or_flat_tail_hold=False,
        grouped_by_causal_proposal=True, native_prefix_evaluations=len(groups),
        numerical_integration_note='Dense query times split RK4 intervals after the same causal proposal anchor; finite RK4 differences are not changes to the posterior law or saved observation rows.',
        native_step_count=int(flow.total_steps),
        maximum_quaternion_norm_correction=float(flow.maximum_norm_correction)),
        dense_wrapper_wall_seconds=time.monotonic()-begun)
    return result
