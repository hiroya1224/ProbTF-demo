"""Chronological event replay helpers; batching changes scheduling only.

Every selected sensor is fused against the preceding canonical posterior.
These helpers do not construct batch-start likelihoods or aggregate hazards.
"""
from .async_state import RecordedCommandPath


def validate_claimed_event(event, metadata, worker_id):
    if (event.sequence != metadata.sequence or event.kind != metadata.kind
            or event.stamp != metadata.stamp or event.command != metadata.command
            or event.payload_hash != metadata.payload_hash
            or event.payload_json != metadata.payload_json
            or event.status != 'processing' or event.worker_id != worker_id):
        raise ValueError('An immutable replay observation changed after claim.')


def selected_sensor_event_ids(events, enabled_sensors, selection=None):
    """Validate an explicit policy decision; the default includes every sensor."""
    events = tuple(events)
    if (len({event.event_id for event in events}) != len(events)
            or any((left.stamp, left.sequence) >= (right.stamp, right.sequence)
                   for left, right in zip(events, events[1:]))):
        raise ValueError('A replay frontier must preserve chronological acquisition and sequence order.')
    eligible = tuple(event.event_id for event in events
        if {'imu': 'imu', 'image': 'marker'}.get(event.kind) in enabled_sensors)
    if selection is None:
        return frozenset(eligible)
    selected = tuple(selection['selected_event_ids'])
    if (len(set(selected)) != len(selected)
            or any(identity not in eligible for identity in selected)
            or tuple(identity for identity in eligible if identity in selected) != selected):
        raise ValueError('Temporal observation selection must retain original identities and order.')
    return frozenset(selected)


def command_path_to_event(previous, previous_index, event, indexed_steps, inference):
    """Use the full actual suffix, including intervals across omitted raw events."""
    index = int(event.payload['frame_index'])
    if previous is None:
        if event.kind != 'joint':
            raise ValueError('Chronological replay must initialize from a joint notification.')
        return None, ()
    if previous_index is None or index < previous_index:
        raise ValueError('A replay acquisition precedes its retained body source.')
    if any(interval not in indexed_steps for interval in range(previous_index, index)):
        raise ValueError('Chronological replay is missing an actually executed command interval.')
    steps = tuple(indexed_steps[interval] for interval in range(previous_index, index))
    path = RecordedCommandPath(previous, steps, body_ids=inference.body_ids,
                               actions=inference.actions)
    if path.target_stamp != event.stamp or path.target_command != event.command:
        raise ValueError('A replay acquisition differs from its actual ledger endpoint.')
    return path, steps


def record_sequential_source(result, previous, event, *, batch_id, batch_offset):
    """Keep the last effective observation clock through missing observations."""
    observed = tuple(result.records.get('observation_ids', ()))
    if observed not in ((), (event.event_id,)):
        raise ValueError('One sequential update must retain exactly its own observation identity.')
    old = {} if previous is None else previous.records.get('sequential_source', {})
    effective_stamp = event.stamp if observed else old.get('last_effective_observation_stamp')
    effective_id = event.event_id if observed else old.get('last_effective_observation_id')
    result.records['sequential_source'] = dict(
        acquisition_event_id=event.event_id, acquisition_sequence=event.sequence,
        acquisition_stamp=event.stamp, acquisition_command=list(event.command),
        last_effective_observation_stamp=effective_stamp,
        last_effective_observation_id=effective_id,
        endpoint_prediction_stamp=result.prior.stamp,
        endpoint_command=list(result.prior.command),
        prior_state_ids=[branch.reference_state.state_id
                         for branch in result.frozen_body_branches],
        terminal_body_state_ids=[item.state.state_id
                                 for item in result.stages[-1].components],
        replay_batch_id=batch_id, replay_batch_offset=batch_offset,
        batch_is_scheduling_only=True, hazard_aggregation_applied=False,
        likelihood_constructed_from_previous_canonical=True,
        physical_truth_received=False, monte_carlo_used=False)
