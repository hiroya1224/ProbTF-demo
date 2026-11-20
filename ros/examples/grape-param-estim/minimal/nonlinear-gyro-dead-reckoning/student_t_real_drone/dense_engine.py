"""Causal dense physical prediction around the existing Student-t updates.

The existing engine owns all observation updates. Its saved pre/post rows are
retained exactly. Additional times, including the unobserved final tail, are
push-forwards of the same initial-rate law conditioned only on readings that
have already arrived. No interpolation, new observation, process noise, or
Gaussian posterior replacement is introduced here.
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'synthetic_controlled_rotation/sequential_bayes_spikes'
spec = importlib.util.spec_from_file_location('_real_drone_existing_student_engine',
                                            SHARED/'full_rate_engine.py')
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)

ROW_FIELDS = ('time', 'event_side', 'acquisition_index', 'quaternion_second_moment',
    'rotation_mean', 'rotation_covariance_colmajor', 'omega_mean_rad_s',
    'omega_covariance_rad2_s2', 'rotation_omega_covariance_colmajor', 'rpy_proxy_rad',
    'rpy_covariance_delta_rad2', 'initial_omega_mean_rad_s',
    'initial_omega_covariance_rad2_s2', 'core_vs_rotation_mean_max_abs',
    'target_likelihood_factor_count')


def predict_dense(data, physical, sigma_deg_s=5., order=11, max_step=.002,
                  mode='nonlinear', student_df=4., output_dt=.01,
                  refresh_seconds=.02, **engine_options):
    """Return unchanged observation rows plus propagated dense/tail moments.

    ``event_side=0`` identifies an added prediction, ``-1/+1`` retain the
    original observation prior/posterior. At a prediction, acquisition_index
    and target_likelihood_factor_count identify the latest included reading.
    The first reading initializes the original Gaussian exactly once.
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
    extra = {key: [] for key in ROW_FIELDS}
    flow = engine.FullRateFlow(physical, data['initial_quaternion_wxyz'],
                               start, end, max_step=max_step, mode=mode)
    refresh_indices = original['refresh_acquisition_index']
    posterior_rows = np.flatnonzero(original['event_side'] == 1)
    interval_records = []
    for index, begin in enumerate(acquired):
        last = index == len(acquired)-1
        finish = end if last else acquired[index+1]
        stamps = grid[(grid > begin) & ((grid <= finish) if last else (grid < finish))]
        # In particular, a final observation exactly at end needs no tail.
        if not len(stamps):
            continue
        refresh = int(np.searchsorted(refresh_indices, index, side='right')-1)
        center = original['refresh_proposal_mean_rad_s'][refresh]
        proposal_covariance = original['refresh_proposal_covariance_rad2_s2'][refresh]
        nodes, quadrature = engine.gaussian_tensor_points(center, proposal_covariance, order)
        # Prefix rows are precisely genuine past acquisitions. Every added
        # physical query follows the last included acquisition; it never
        # occupies a likelihood slot or receives a future measurement value.
        query = np.r_[acquired[:index+1], stamps]
        likelihood, states = flow.prefix_and_block(nodes, query, observed,
            prefix=index+1, finish=len(query)-1, covariance=covariance,
            student_df=student_df)
        raw = (np.log(quadrature)
            +engine.gaussian_log_density(nodes, observed[0], covariance)
            -engine.gaussian_log_density(nodes, center, proposal_covariance)+likelihood)
        normalizer = float(logsumexp(raw))
        if not np.isfinite(normalizer):
            raise FloatingPointError('Undefined dense posterior normalizer')
        mass = np.exp(raw-normalizer)
        last_rpy = original['rpy_proxy_rad'][posterior_rows[index]].copy()
        for stamp, state in zip(stamps, states):
            moment = engine.summarize_joint(state[:, 3:], state[:, :3], mass,
                                            initial_nodes=nodes)
            # Only choose the angle branch from the preceding causal value.
            # All original observation rows remain untouched below.
            angle = moment['rpy_proxy_rad']
            angle = angle+2.*np.pi*np.round((last_rpy-angle)/(2.*np.pi))
            moment['rpy_proxy_rad'] = angle
            last_rpy = angle
            moment.update(time=float(stamp), event_side=0, acquisition_index=index,
                          target_likelihood_factor_count=index)
            for key in ROW_FIELDS:
                extra[key].append(np.asarray(moment[key]).copy())
        interval_records.append(dict(last_included_acquisition_index=index,
            last_included_acquisition_time=float(begin),
            reused_refresh_acquisition_index=int(refresh_indices[refresh]),
            first_prediction_time=float(stamps[0]), last_prediction_time=float(stamps[-1]),
            prediction_count=len(stamps), included_noninitial_likelihood_count=index))
    result = dict(original)
    old_count = len(original['time'])
    added_count = len(extra['time'])
    if added_count:
        merged_time = np.r_[original['time'], np.stack(extra['time'])]
        permutation = np.argsort(merged_time, kind='stable')
        for key in ROW_FIELDS:
            combined = np.concatenate((original[key], np.stack(extra[key])), axis=0)
            result[key] = combined[permutation]
        inverse_permutation = np.argsort(permutation)
        result['observation_row_indices'] = inverse_permutation[:old_count]
    else:
        result['observation_row_indices'] = np.arange(old_count)
    result['dense_output_grid_time'] = grid
    result['dense_interval_records'] = interval_records
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
        native_step_count=int(flow.total_steps),
        maximum_quaternion_norm_correction=float(flow.maximum_norm_correction)),
        dense_wrapper_wall_seconds=time.monotonic()-begun)
    return result
