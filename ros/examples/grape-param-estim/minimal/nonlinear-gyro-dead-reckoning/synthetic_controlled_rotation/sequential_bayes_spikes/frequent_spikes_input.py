"""Native-clock Poisson spike pulses, independent of gyro acquisition times.

Randomness belongs only to sensor generation.  The reconstruction receives
the selected gyro values, not the event process, mask, bias, or truth.  The
earlier acquired-one-spike stress experiment is kept separate and unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_frequent_spikes_original_trajectory', HERE.parent/'trajectory.py')
_trajectory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trajectory)


DEFAULTS = dict(
    duration_seconds=20., sensor_rate_hz=200., rate_hz=1.,
    white_seed=20261113, white_sigma_deg_s=2., spike_seed=20261120,
    spike_rate_hz=10., spike_width_range_seconds=[.03, .08],
    spike_magnitude_range_deg_s=[20., 60.], spike_active_until_seconds=15.,
    motion_profile='three_frequency')


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.:
        raise ValueError(name+' must be positive and finite')
    return value


def _range(value, name, *, positive):
    value = np.asarray(value, dtype=float)
    if (value.shape != (2,) or not np.isfinite(value).all()
            or value[1] < value[0] or value[0] < 0.
            or (positive and value[0] == 0.)):
        raise ValueError(name+' must be an ordered finite nonnegative pair')
    return value


def generate(config=None):
    """Return source, selected inputs, acquisitions, sensor, events, metadata.

    ``spike_rate_hz`` is a homogeneous Poisson arrival intensity.  Pulse
    onsets are generated in continuous time over the declared active phase,
    independently of the 1 Hz observation grid.  The first native sample at
    or after onset is affected; samples at/after onset+width are excluded.
    Pulses are rectangular, with an independent isotropic direction and a
    uniformly distributed vector length.  Simultaneous pulse vectors add.

    No event is forced onto an acquisition or regenerated to obtain a
    particular reconstruction result.  The declared intervention ends at
    spike_active_until_seconds, clipping any crossing pulse and creating a
    clean recovery tail.  This is a predeclared test schedule, not selection
    of a lucky tail from the realized randomness.  The initial reading is
    naturally unspiked because arrival times are > 0.
    """
    settings = dict(DEFAULTS)
    if config is not None:
        settings.update(config)
    duration = _positive(settings['duration_seconds'], 'duration_seconds')
    native_rate = _positive(settings['sensor_rate_hz'], 'sensor_rate_hz')
    acquisition_rate = _positive(settings['rate_hz'], 'rate_hz')
    arrival_rate = _positive(settings['spike_rate_hz'], 'spike_rate_hz')
    white_sigma = _positive(settings['white_sigma_deg_s'], 'white_sigma_deg_s')
    active_until = _positive(settings['spike_active_until_seconds'],
                             'spike_active_until_seconds')
    if active_until > duration:
        raise ValueError('spike_active_until_seconds must not exceed duration_seconds')
    widths = _range(settings['spike_width_range_seconds'],
                    'spike_width_range_seconds', positive=True)
    magnitudes = _range(settings['spike_magnitude_range_deg_s'],
                        'spike_magnitude_range_deg_s', positive=False)
    parameters_file = settings.get('motion_parameters_file')
    motion_parameters = None
    if parameters_file is not None:
        parameters_path = (HERE/parameters_file).resolve()
        motion_parameters = json.loads(parameters_path.read_text())
    source, generation = _trajectory.generate_dataset(
        duration=duration, sensor_rate=native_rate,
        seed=int(settings['white_seed']), motion_profile=settings['motion_profile'],
        motion_parameters=motion_parameters)
    if parameters_file is not None:
        generation['motion_calibration'] = dict(
            parameters_file=str(parameters_file),
            parameters_file_sha256=hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
            purpose='actual-flight amplitude/frequency calibration of the synthetic generator only; not an online observation or estimator parameter',
            motion_parameters_supplied_to_estimator=False)
    # Reuse the existing seeded standardized Gaussian realization, changing
    # its requested physical sigma only in this new experiment.  The old
    # stored measurements and old generator defaults are not overwritten.
    original_sigma = float(generation['gyro_noise_generation_sigma_deg_s'])
    source['gyro_realized_noise_rad_s'] = (source['gyro_realized_noise_rad_s']
                                          *(white_sigma/original_sigma))
    source['gyro'] = (source['gyro_true_body_rad_s']+source['gyro_bias_rad_s']
                      +source['gyro_realized_noise_rad_s'])
    source['generator_gyro_covariance_rad2_s2'] = np.eye(3)*np.deg2rad(white_sigma)**2
    generation = dict(generation,
        gyro_noise_generation_sigma_deg_s=white_sigma,
        gyro_noise_generation_covariance_rad2_s2=source['generator_gyro_covariance_rad2_s2'].tolist(),
        standardized_white_realization='same seed and original standardized Gaussian draws; scaled to requested new physical sigma',
        standardized_white_original_sigma_deg_s=original_sigma,
        old_stored_raw_gyro_unchanged=True)
    time = source['imu_time']
    native_count = len(time)
    # Distinct deterministic streams make the arrival process independent
    # of width/direction/amplitude generation and all acquisition choices.
    streams = np.random.SeedSequence(int(settings['spike_seed'])).spawn(4)
    arrival_rng, width_rng, magnitude_rng, direction_rng = (
        np.random.default_rng(stream) for stream in streams)
    onsets = []
    stamp = float(arrival_rng.exponential(1./arrival_rate))
    while stamp < active_until:
        onsets.append(stamp)
        stamp += float(arrival_rng.exponential(1./arrival_rate))
    onsets = np.asarray(onsets, dtype=float)
    pulse_count = len(onsets)
    pulse_widths = width_rng.uniform(*widths, size=pulse_count)
    applied_stops = np.minimum(onsets+pulse_widths, active_until)
    pulse_magnitudes = magnitude_rng.uniform(*magnitudes, size=pulse_count)
    directions = direction_rng.normal(size=(pulse_count, 3))
    norm = np.linalg.norm(directions, axis=1)
    if np.any(norm == 0.):
        raise FloatingPointError('Generated an undefined spike direction')
    directions /= norm[:, None]
    vectors = directions*np.deg2rad(pulse_magnitudes)[:, None]
    starts = np.searchsorted(time, onsets, side='left')
    stops = np.searchsorted(time, applied_stops, side='left')
    perturbation = np.zeros((native_count, 3))
    active_count = np.zeros(native_count, dtype=np.int64)
    records = []
    for event, (onset, width, lower, upper, vector) in enumerate(
            zip(onsets, pulse_widths, starts, stops, vectors)):
        lower, upper = int(lower), int(upper)
        perturbation[lower:upper] += vector
        active_count[lower:upper] += 1
        records.append(dict(
            event_index=event,
            requested_onset_seconds=float(onset),
            requested_stop_seconds=float(onset+width),
            applied_stop_seconds=float(applied_stops[event]),
            width_seconds=float(width),
            applied_width_seconds=float(applied_stops[event]-onset),
            first_native_array_index=lower,
            native_stop_array_index_exclusive=upper,
            first_native_source_index=(int(source['imu_source_index'][lower])
                                       if lower < native_count else None),
            last_native_source_index=(int(source['imu_source_index'][upper-1])
                                      if upper > lower else None),
            first_native_stamp_ns=(int(source['imu_stamp_ns'][lower])
                                   if lower < native_count else None),
            last_native_stamp_ns=(int(source['imu_stamp_ns'][upper-1])
                                  if upper > lower else None),
            native_sample_count=upper-lower,
            direction_body=directions[event].tolist(),
            magnitude_deg_s=float(pulse_magnitudes[event]),
            vector_body_rad_s=vector.tolist()))
    mask = active_count > 0
    sensor = dict(
        time=time.copy(), stamp_ns=source['imu_stamp_ns'].copy(),
        record_ns=source['imu_record_ns'].copy(),
        source_index=source['imu_source_index'].copy(),
        gyro_white=source['gyro'].copy(),
        gyro_spike=source['gyro']+perturbation,
        spike_mask=mask, spike_vector_rad_s=perturbation,
        active_spike_count=active_count,
        event_requested_onset_seconds=onsets,
        event_requested_stop_seconds=onsets+pulse_widths,
        event_applied_stop_seconds=applied_stops,
        event_native_start_array_index=starts,
        event_native_stop_array_index_exclusive=stops,
        event_width_seconds=pulse_widths,
        event_direction_body=directions,
        event_magnitude_deg_s=pulse_magnitudes,
        event_vector_body_rad_s=vectors)
    desired = np.arange(0., duration, 1./acquisition_rate)
    requested_ns = np.rint(desired*1e9).astype(np.int64)
    selected_index = np.searchsorted(source['imu_stamp_ns'], requested_ns)
    acquisitions = dict(
        time=time[selected_index].copy(), requested_stamp_ns=requested_ns,
        stamp_ns=source['imu_stamp_ns'][selected_index].copy(),
        record_ns=source['imu_record_ns'][selected_index].copy(),
        source_index=source['imu_source_index'][selected_index].copy(),
        native_array_index=selected_index.copy(),
        gyro_white=sensor['gyro_white'][selected_index].copy(),
        gyro_spike=sensor['gyro_spike'][selected_index].copy(),
        spike_mask=mask[selected_index].copy(),
        spike_vector_rad_s=perturbation[selected_index].copy(),
        active_spike_count=active_count[selected_index].copy())
    selected = dict(
        time=source['time'].copy(),
        initial_quaternion_wxyz=source['initial_quaternion_wxyz'].copy(),
        sparse_gyro_time=acquisitions['time'].copy(),
        sparse_gyro=acquisitions['gyro_white'].copy())
    hit_count = int(np.count_nonzero(acquisitions['spike_mask']))
    metadata = dict(
        schema='grape/synthetic-native-poisson-spikes/v1',
        generation=generation,
        duration_seconds=duration,
        native_rate_hz=native_rate,
        acquisition_rate_hz=acquisition_rate,
        spike_seed=int(settings['spike_seed']),
        white_seed=int(settings['white_seed']),
        white_sigma_deg_s=white_sigma,
        spike_process='homogeneous Poisson pulse onsets in continuous sensor time',
        spike_expected_arrivals_per_second=arrival_rate,
        realized_spike_count=pulse_count,
        realized_arrivals_per_second=pulse_count/active_until,
        spike_active_until_seconds=active_until,
        spike_active_interval_seconds=[0., active_until],
        clean_recovery_tail_seconds=duration-active_until,
        clean_tail_is_a_predeclared_intervention=True,
        pulse_shape='rectangular over [onset,min(onset+width,active_until)); overlapping vectors sum',
        native_discretization='first native sample at/after onset, excluding samples at/after requested stop',
        spike_width_range_seconds=widths.tolist(),
        spike_magnitude_range_deg_s=magnitudes.tolist(),
        direction_law='isotropic unit direction from a normalized 3D standard Gaussian, generation only',
        arrival_process_independent_of_acquisition=True,
        forced_acquired_spike_count=False,
        forced_clean_recovery_interval=bool(active_until < duration),
        seeds_reselected_from_estimation_results=False,
        native_spiked_sample_count=int(np.count_nonzero(mask)),
        native_spiked_fraction=float(np.mean(mask)),
        acquired_observation_count=len(selected_index),
        acquired_spiked_count=hit_count,
        acquired_clean_count=len(selected_index)-hit_count,
        acquired_spike_times=acquisitions['time'][acquisitions['spike_mask']].tolist(),
        acquired_clean_times=acquisitions['time'][~acquisitions['spike_mask']].tolist(),
        initial_reading_unspiked=bool(not mask[0]),
        white_noise_realization_shared_between_cases=True,
        native_spikes_not_supplied_to_estimator=True,
        random_numbers_used_for_sensor_generation_only=True,
        estimator_process_noise_added=False)
    return source, selected, acquisitions, sensor, records, metadata
