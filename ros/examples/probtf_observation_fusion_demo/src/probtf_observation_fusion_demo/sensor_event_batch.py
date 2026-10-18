"""Immutable acquisition-frontier batches; events retain their original order.

Batching changes scheduling and IPC, not the per-event filtering operator.
Metadata-only SensorEvents deliberately cannot stand in for raw observations.
"""

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class SensorEventBatch:
    snapshot_id: str
    worker_id: str
    frontier_stamp: float
    frontier_sequence: int
    claimed_wall_stamp: float
    events: tuple = ()
    reason: str = 'no_events_through_frontier'
    blocked_event_id: object = None
    blocked_event_status: object = None
    blocked_event_stamp: object = None
    blocked_event_sequence: object = None

    def __post_init__(self):
        if not self.snapshot_id or not self.worker_id:
            raise ValueError('A batch needs its snapshot and worker identities.')
        if (not math.isfinite(self.frontier_stamp) or not math.isfinite(self.claimed_wall_stamp)
                or self.frontier_sequence < 0):
            raise ValueError('A batch needs a finite acquisition frontier and claim time.')
        events = tuple(self.events)
        previous = None
        for event in events:
            key = (event.stamp, event.sequence)
            if (previous is not None and key <= previous or event.stamp > self.frontier_stamp
                    or event.sequence > self.frontier_sequence):
                raise ValueError('Batch events must form an ordered prefix within its immutable frontier.')
            if event.status != 'processing' or event.worker_id != self.worker_id:
                raise ValueError('Every batch event must be claimed by the same worker.')
            previous = key
        object.__setattr__(self, 'events', events)

    def __len__(self):
        return len(self.events)

    def __bool__(self):
        return bool(self.events)

    @property
    def event_ids(self):
        return tuple(event.event_id for event in self.events)

    @property
    def batch_id(self):
        value = dict(snapshot_id=self.snapshot_id, worker_id=self.worker_id,
            frontier_stamp=self.frontier_stamp, frontier_sequence=self.frontier_sequence,
            events=[(event.event_id, event.sequence, event.payload_hash) for event in self.events])
        return hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def record(self):
        return dict(batch_id=self.batch_id, snapshot_id=self.snapshot_id, worker_id=self.worker_id,
            frontier_stamp=self.frontier_stamp, frontier_sequence=self.frontier_sequence,
            claimed_wall_stamp=self.claimed_wall_stamp, reason=self.reason,
            event_count=len(self.events), event_ids=list(self.event_ids),
            events=[dict(event_id=event.event_id, sequence=event.sequence, kind=event.kind,
                stamp=event.stamp, command=list(event.command), payload_hash=event.payload_hash,
                payload_loaded=event.payload_loaded,
                capture_index=event.payload_metadata.get('frame_index')) for event in self.events],
            blocked_event=(None if self.blocked_event_id is None else dict(
                event_id=self.blocked_event_id, status=self.blocked_event_status,
                stamp=self.blocked_event_stamp, sequence=self.blocked_event_sequence)),
            ordering='acquisition stamp, then immutable insertion sequence',
            newest_only_selection=False, per_event_updates_preserved=True)


def event_requests(batch, ledger_steps, *, previous_frame_index=None):
    """Build ordered worker requests with every actually executed interval.

``ledger_steps`` can be the full ledger or just its required suffix. No
future command is generated. At initialization the first event is the same
initialization point as in the one-event path, so it needs no earlier action.
The returned tuple contains the existing (event, frame_index, steps) shape;
events can be metadata-only and must then be loaded by ID inside the worker.
"""
    by_index = {}
    for step in ledger_steps:
        if step.index in by_index:
            raise ValueError('A batch ledger has a duplicate actual interval.')
        by_index[step.index] = step
    source = previous_frame_index
    requests = []
    previous_stamp = previous_command = None
    for event in batch.events:
        index = event.payload_metadata.get('frame_index')
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('A batch event requires its acquired physical frame_index.')
        if source is None:
            source = index
        if index < source:
            raise ValueError('A batch cannot move backwards through the executed ledger.')
        required = range(source, index)
        if any(number not in by_index for number in required):
            raise ValueError('A batch ledger is missing an actually executed interval.')
        steps = tuple(by_index[number] for number in required)
        for left, right in zip(steps, steps[1:]):
            if left.end_stamp != right.start_stamp or left.end_command != right.start_command:
                raise ValueError('A batch ledger has a gap or mismatched command.')
        if steps:
            if (steps[-1].end_stamp != event.stamp or steps[-1].end_command != event.command
                    or previous_stamp is not None and (
                        steps[0].start_stamp != previous_stamp
                        or steps[0].start_command != previous_command)):
                raise ValueError('A batch event and its actual ledger endpoint disagree.')
        elif previous_stamp is not None and (event.stamp != previous_stamp or event.command != previous_command):
            raise ValueError('Same-frame batch events must share the actual command and stamp.')
        requests.append((event, index, steps))
        source, previous_stamp, previous_command = index, event.stamp, event.command
    return tuple(requests)


def ledger_range_record(request):
    event, index, steps = request
    return dict(event_id=event.event_id, event_sequence=event.sequence,
        capture_index=index, source_capture_index=index-len(steps),
        actual_interval_count=len(steps), actual_interval_indices=[step.index for step in steps],
        selected_K_density_ids=[step.density_id for step in steps],
        actual_interval_K_and_provenance_retained=True)
