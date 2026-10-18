"""Deterministic time coverage when explicitly thinning a pending frontier.

Selection uses acquisition metadata only. It never reads an image, gravity,
posterior, likelihood or physical truth. The default processes every event.
An explicit per-sensor capacity chooses observations across the entire claimed
time range; it does not replace the frontier with its first or last N events.
All unselected raw events and every executed command interval are retained.
"""
import math
from numbers import Integral


def _uniform_time_indices(events, count):
    if count >= len(events):
        return tuple(range(len(events))), ()
    first, last = float(events[0].stamp), float(events[-1].stamp)
    if first == last:
        # Immutable acquisition order supplies coverage when clocks coincide.
        indices = tuple(int((i + .5) * len(events) / count) for i in range(count))
        return indices, ()
    targets = tuple(first + (i + .5) * (last - first) / count for i in range(count))
    indices, previous = [], -1
    for i, target in enumerate(targets):
        # Reserve enough distinct later acquisitions. Ordered nearest-midpoint
        # selection remains valid for irregular or repeated acquisition times.
        stop = len(events) - (count - i - 1)
        index = min(range(previous + 1, stop),
                    key=lambda j: (abs(float(events[j].stamp) - target), events[j].sequence))
        indices.append(index)
        previous = index
    return tuple(indices), targets


def select_observations(events, enabled_sensors, options=None):
    """Select a chronological subset without changing the immutable raw batch.

``mode=all`` is the default. ``uniform_time`` requires explicit positive
``max_events_per_sensor`` capacities keyed by ``imu`` and/or ``image``.
Unspecified sensor types retain all acquisitions. Command notifications and
disabled sensors are not counted as statistical observations.
"""
    events, enabled = tuple(events), set(enabled_sensors)
    options = dict(options or {})
    mode = options.get('mode', 'all')
    if mode not in ('all', 'uniform_time'):
        raise ValueError('Unknown observation selection mode: ' + str(mode))
    capacities = dict(options.get('max_events_per_sensor', {}))
    if any(kind not in ('imu', 'image') or isinstance(value, bool)
           or not isinstance(value, Integral) or value < 1
           for kind, value in capacities.items()):
        raise ValueError('Observation capacities must be positive imu/image event counts.')
    if any(not math.isfinite(float(event.stamp)) for event in events):
        raise ValueError('Observation selection needs finite acquisition times.')
    keys = [(event.stamp, event.sequence) for event in events]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError('Observation selection needs immutable acquisition order.')
    selected, per_sensor = set(), {}
    for kind, sensor in (('imu', 'imu'), ('image', 'marker')):
        eligible = tuple(event for event in events if event.kind == kind and sensor in enabled)
        count = (min(len(eligible), int(capacities.get(kind, len(eligible))))
                 if mode == 'uniform_time' else len(eligible))
        indices, targets = _uniform_time_indices(eligible, count) if count else ((), ())
        retained = tuple(eligible[index] for index in indices)
        selected.update(event.event_id for event in retained)
        per_sensor[kind] = dict(available_count=len(eligible), selected_count=len(retained),
            available_time_range=(None if not eligible else [eligible[0].stamp, eligible[-1].stamp]),
            selected_time_range=(None if not retained else [retained[0].stamp, retained[-1].stamp]),
            target_bin_midpoints=list(targets),
            selected_event_ids=[event.event_id for event in retained],
            selected_stamps=[event.stamp for event in retained],
            max_events=capacities.get(kind) if mode == 'uniform_time' else None)
    eligible_ids = {event.event_id for event in events
                    if {'imu': 'imu', 'image': 'marker'}.get(event.kind) in enabled}
    omitted = eligible_ids - selected
    return dict(mode=mode,
        selection_rule=('all enabled acquired observations' if mode == 'all' else
                        'ordered nearest equal-duration-bin midpoint over the full per-sensor frontier'),
        selected_event_ids=[event.event_id for event in events if event.event_id in selected],
        unselected_event_ids=[event.event_id for event in events if event.event_id in omitted],
        per_sensor=per_sensor, thinning_applied=bool(omitted),
        unselected_reason='temporal_thinning' if omitted else None,
        selection_uses_observed_values=False, original_raw_retained=True,
        original_time_command_sequence_retained=True, actual_command_ledger_omitted=False,
        bfs_or_K_update_for_unselected_observations=False)
