"""Read saved six-point angular-rate laws, including both acquisition sides.

This module only exposes already propagated moments. It does not sample,
refit, interpolate across an observation, or change a candidate covariance.
The current experiment replaces angular rate by a fresh independent Gaussian
at each acquisition; that operation is not a conditional Bayesian fusion.
"""
from __future__ import annotations

import numpy as np


def _axis_width_ratio(after, before):
    """JSON-safe ratios; an exactly zero prior width has no finite ratio."""
    return [float(a / b) if b > 0. else None for a, b in zip(after, before)]


def _halfwidth_deg_s(covariance):
    return 2. * np.rad2deg(np.sqrt(np.diag(covariance)))


def extract_progress(prediction, acquisitions, sigma_deg_s):
    """Return full moments and pre/post metadata from saved increment factors.

    Result ``arrays`` contains ``time``, ``mean_rad_s``, the full
    ``covariance_rad2_s2``, ``block_index``, ``stage`` and
    ``right_continuous_mask``. Observation boundaries occur twice at exactly
    the same timestamp: previous prediction first, new input second. Stage
    codes are 0=interior/final, 1=before acquisition, 2=after acquisition.

    ``events`` is a JSON-safe list including the initial acquisition (whose
    prior is None). Width ratios below/above one mean actual narrowing/
    widening respectively; the data is never altered to create a waist.
    ``metadata`` records differences from canonical saved right-continuous
    moments as diagnostics, without making a new numerical acceptance gate.
    """
    sigma = float(sigma_deg_s)
    stamps = np.asarray(acquisitions['time'], float)
    gyro = np.asarray(acquisitions['gyro'], float)
    if (stamps.ndim != 1 or not len(stamps) or gyro.shape != (len(stamps), 3)
            or not np.isfinite(stamps).all() or not np.isfinite(gyro).all()
            or np.any(np.diff(stamps) <= 0.) or not np.isfinite(sigma) or sigma < 0.):
        raise ValueError('Finite increasing acquisitions and nonnegative sigma required')

    blocks = []
    index = 0
    while f'block_{index}_time' in prediction:
        prefix = f'block_{index}_'
        time = np.asarray(prediction[prefix+'time'], float)
        nodes = np.asarray(prediction[prefix+'omega'], float)
        weights = np.asarray(prediction[prefix+'weights'], float)
        if (time.ndim != 1 or len(time) < 2 or nodes.ndim != 3
                or nodes.shape[0] != len(time) or nodes.shape[2] != 3
                or weights.shape != (nodes.shape[1],)
                or not all(np.isfinite(x).all() for x in (time, nodes, weights))
                or np.any(np.diff(time) <= 0.) or np.any(weights < 0.)):
            raise ValueError('Saved block must contain increasing times and weighted rate trajectories')
        mean = np.einsum('s,tsi->ti', weights, nodes)
        centered = nodes - mean[:, None, :]
        covariance = np.einsum('s,tsi,tsj->tij', weights, centered, centered)
        blocks.append(dict(time=time, mean=mean, covariance=covariance))
        index += 1
    if len(blocks) != len(stamps):
        raise ValueError('Each saved acquisition must have exactly one prediction block')

    parts = {name: [] for name in ('time', 'mean_rad_s', 'covariance_rad2_s2',
                                  'block_index', 'stage', 'right_continuous_mask')}
    events = []
    post_covariance = np.eye(3) * np.deg2rad(sigma)**2
    for block, values in enumerate(blocks):
        if values['time'][0] != stamps[block]:
            raise ValueError('Saved block start and acquisition clock differ')
        if block and blocks[block-1]['time'][-1] != values['time'][0]:
            raise ValueError('Pre/post acquisition timestamps must match exactly')
        count = len(values['time'])
        stage = np.zeros(count, dtype=np.int8)
        stage[0] = 2
        keep = np.ones(count, dtype=bool)
        if block < len(blocks)-1:
            stage[-1] = 1
            keep[-1] = False
        for name, value in dict(time=values['time'], mean_rad_s=values['mean'],
                covariance_rad2_s2=values['covariance'],
                block_index=np.full(count, block, dtype=np.int64), stage=stage,
                right_continuous_mask=keep).items():
            parts[name].append(value)

        previous = blocks[block-1] if block else None
        before_mean = previous['mean'][-1] if previous is not None else None
        before_covariance = previous['covariance'][-1] if previous is not None else None
        after_mean = values['mean'][0]
        after_covariance = values['covariance'][0]
        before_width = _halfwidth_deg_s(before_covariance) if previous is not None else None
        after_width = _halfwidth_deg_s(after_covariance)
        event = dict(observation_index=block, time_seconds=float(stamps[block]),
            acquired_gyro_rad_s=gyro[block].tolist(),
            before_mean_rad_s=None if before_mean is None else before_mean.tolist(),
            before_covariance_rad2_s2=None if before_covariance is None else before_covariance.tolist(),
            after_mean_rad_s=after_mean.tolist(),
            after_covariance_rad2_s2=after_covariance.tolist(),
            before_two_sigma_deg_s=None if before_width is None else before_width.tolist(),
            after_two_sigma_deg_s=after_width.tolist(),
            post_over_pre_width_ratio=None if before_width is None else _axis_width_ratio(after_width, before_width),
            post_minus_assumed_covariance_max_abs_rad2_s2=float(np.max(np.abs(after_covariance-post_covariance))),
            post_mean_minus_acquisition_max_abs_rad_s=float(np.max(np.abs(after_mean-gyro[block]))))
        for source_key in ('source_index', 'stamp_ns', 'requested_stamp_ns'):
            if source_key in acquisitions:
                event[source_key] = int(np.asarray(acquisitions[source_key])[block])
        events.append(event)

    arrays = {name: np.concatenate(value) for name, value in parts.items()}
    keep = arrays['right_continuous_mask']
    saved_time = np.asarray(prediction['time'], float)
    saved_mean = np.asarray(prediction['omega_mean_rad_s'], float)
    saved_covariance = np.asarray(prediction['omega_covariance_rad2_s2'], float)
    if (arrays['time'][keep].shape != saved_time.shape
            or saved_mean.shape != arrays['mean_rad_s'][keep].shape
            or saved_covariance.shape != arrays['covariance_rad2_s2'][keep].shape):
        raise ValueError('Canonical and blockwise saved arrays have inconsistent shapes')
    metadata = dict(
        source='original saved six-point angular-rate trajectories and unchanged weights',
        operation='weighted mean and centered full covariance, with both sides of every acquisition retained',
        moment_units=dict(time='s', mean='rad/s', covariance='(rad/s)^2'),
        stage_codes={'0': 'interior or final prediction', '1': 'before acquisition', '2': 'after acquisition'},
        observation_count=len(stamps), prediction_blocks=len(blocks),
        retained_pre_observation_endpoints=len(blocks)-1,
        equal_timestamp_boundary_pairs=len(blocks)-1,
        angular_rate_update='fresh independent Gaussian centered at the acquired gyro with covariance sigma^2 I; not Bayesian conditioning of prior rate',
        covariance_reset_sigma_deg_s=sigma,
        linear_forced_covariance='constant sigma^2 I between acquisitions because its angular-rate sensitivity is identity',
        nonlinear_covariance='nonlinear dynamics can increase or decrease different marginal widths; a new observation need not narrow every axis',
        moments_changed=False, interpolation_across_observations=False,
        monte_carlo=False, covariance_floor_or_clip=False,
        canonical_time_bitwise_equal=bool(np.array_equal(arrays['time'][keep], saved_time)),
        canonical_mean_max_abs_difference_rad_s=float(np.max(np.abs(arrays['mean_rad_s'][keep]-saved_mean))),
        canonical_covariance_max_abs_difference_rad2_s2=float(np.max(np.abs(arrays['covariance_rad2_s2'][keep]-saved_covariance))))
    return dict(arrays=arrays, events=events, metadata=metadata)
