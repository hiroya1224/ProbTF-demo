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
    span = last-first
    if math.isfinite(span):
        relative = tuple((float(event.stamp)-first)/span for event in events)
        interpolate = lambda fraction: first+fraction*span
    else:
        # Opposite, finite extreme timestamps can overflow their difference.
        # A common positive rescaling preserves the same least-squares problem.
        scale = max(abs(first), abs(last))
        left, right = first/scale, last/scale
        relative = tuple((float(event.stamp)/scale-left)/(right-left) for event in events)
        interpolate = lambda fraction: (1.-fraction)*first+fraction*last
    if count == 1:
        index = min(range(len(events)),
                    key=lambda j: (abs(relative[j]-.5), events[j].sequence))
        return (index,), (interpolate(.5),)

    fractions = tuple(i/(count-1) for i in range(count))
    targets = (first,)+tuple(interpolate(value) for value in fractions[1:-1])+(last,)
    # Fix both endpoint acquisitions. Midpoint targets could choose two almost
    # simultaneous central observations while omitting both ends of a frontier.
    # For the remaining strictly ordered indices j_i, minimize
    #   sum_i ((t[j_i]-first)/(last-first)-i/(count-1))**2.
    # A common scaling of all time residuals does not change the minimizer.
    # Prefix minima evaluate the ordered assignment in O(count*len(events));
    # unlike forward nearest-neighbor selection, an early choice cannot consume
    # a cluster needed to distribute the later targets. Exact ties retain the
    # first predecessor in immutable acquisition order.
    size = len(events)
    previous = [math.inf]*size
    previous[0] = 0.
    parents = [[None]*size for _ in range(count)]
    for stage in range(1, count):
        current = [math.inf]*size
        best_cost, best_index = math.inf, None
        for index in range(1, size):
            if previous[index-1] < best_cost:
                best_cost, best_index = previous[index-1], index-1
            permitted = (index == size-1 if stage == count-1 else
                         stage <= index <= size-count+stage)
            if permitted and best_index is not None:
                current[index] = best_cost+(relative[index]-fractions[stage])**2
                parents[stage][index] = best_index
        previous = current
    indices = [size-1]
    for stage in range(count-1, 0, -1):
        indices.append(parents[stage][indices[-1]])
    return tuple(reversed(indices)), targets


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
            target_times=list(targets),
            endpoint_times_included=(len(retained) >= 2 and bool(eligible) and
                retained[0].stamp == eligible[0].stamp and retained[-1].stamp == eligible[-1].stamp),
            identical_timestamp_sequence_quantiles=(bool(eligible) and count < len(eligible)
                and eligible[0].stamp == eligible[-1].stamp),
            selected_event_ids=[event.event_id for event in retained],
            selected_stamps=[event.stamp for event in retained],
            max_events=capacities.get(kind) if mode == 'uniform_time' else None)
    eligible_ids = {event.event_id for event in events
                    if {'imu': 'imu', 'image': 'marker'}.get(event.kind) in enabled}
    omitted = eligible_ids - selected
    return dict(mode=mode,
        selection_algorithm=('all_v1' if mode == 'all' else
                             'endpoint_constrained_ordered_least_squares_v2'),
        selection_rule=('all enabled acquired observations' if mode == 'all' else
                        'fixed endpoint times and globally least-squares ordered unique equal-time targets; singleton central time; identical-clock sequence quantiles'),
        selected_event_ids=[event.event_id for event in events if event.event_id in selected],
        unselected_event_ids=[event.event_id for event in events if event.event_id in omitted],
        per_sensor=per_sensor, thinning_applied=bool(omitted),
        unselected_reason='temporal_thinning' if omitted else None,
        selection_uses_observed_values=False, original_raw_retained=True,
        original_time_command_sequence_retained=True, actual_command_ledger_omitted=False,
        bfs_or_K_update_for_unselected_observations=False)
