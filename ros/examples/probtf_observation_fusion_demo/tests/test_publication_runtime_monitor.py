"""Publication timing separates receipt from service without changing payloads."""
import json

import pytest

from test_async_publication import packet_for, render_context, wire
from test_measure_display_provenance import acquired, measured_source


def progress(renderer):
    return [json.loads(line) for line in
        (renderer.output/'runtime/publication_progress.jsonl').read_text().splitlines()]


def test_receipt_raw_publish_and_save_intervals_preserve_original_sources(render_context, acquired,
                                                                        monkeypatch):
    renderer, clock, control, source = render_context
    original_raw = renderer._publish_raw_events

    def slow_raw(events):
        original_raw(events)
        # Time spent in raw publication belongs to the historical delay,
        # but must not be counted as time waiting to receive this packet.
        clock[0] += 2.

    monkeypatch.setattr(renderer, '_publish_raw_events', slow_raw)
    pixels = acquired[1].image
    raw = (dict(kind='joint', stamp=source.prior.stamp, command=source.prior.command,
                payload={'command_velocity': [0.]*6}),)
    packet = packet_for(control, raw=raw, preview=(0, pixels, source.prior.stamp))
    ack = renderer.process(packet, received_monotonic_s=10., received_wall_stamp=1234.)
    assert ack['queue_received_delay_s'] == 1.
    assert ack['queue_delay_s'] == 3.
    assert ack['queue_received_monotonic_s'] == 10.
    assert ack['queue_received_wall_stamp'] == 1234.
    assert wire(renderer.distributions.messages[0]) == control['markers_wire']
    assert renderer.images.messages[0].data == pixels.tobytes()
    assert renderer.current['snapshot'] is control['snapshot']
    manifest = json.loads((renderer.output/'motion_manifest.json').read_text())
    assert manifest['physical_stamp'] == packet.frame.stamp
    assert manifest['displayed_snapshot']['stamp'] == control['snapshot'].stamp
    assert manifest['publication_process']['queue_delay_s'] == 3.
    assert manifest['publication_process']['queue_received_delay_s'] == 1.
    rows = progress(renderer)
    finished = [row for row in rows if row['event'] == 'stage_finished']
    assert [row['stage'] for row in finished] == [
        'preview_publish', 'raw_publish', 'snapshot_wire_publish', 'snapshot_json_publish',
        'stiffness_json_publish', 'scene_publish', 'motion_metadata_publish',
        'manifest_json_save', 'publication_record_save']
    assert all(row['outcome'] == 'completed' and row['duration_s'] >= 0.
               and row['cpu_s'] >= 0. for row in finished)
    assert all(row['context']['physical_stamp'] == packet.frame.stamp for row in finished)
    receipt = next(row for row in rows if row.get('name') == 'queue_received')
    assert receipt['context']['queue_received_monotonic_s'] == 10.
    assert receipt['context']['queue_received_wall_stamp'] == 1234.
    assert receipt['context']['queue_received_delay_s'] == 1.
    assert rows[-1]['name'] == 'packet_completed'


def test_real_publication_failure_is_timed_and_still_propagates(render_context, monkeypatch):
    renderer, _, control, _ = render_context

    def fail(*args, **kwargs):
        raise RuntimeError('scene publisher did not complete')

    monkeypatch.setattr(renderer.display, 'publish', fail)
    with pytest.raises(RuntimeError, match='scene publisher did not complete'):
        renderer.process(packet_for(control))
    rows = progress(renderer)
    failed = [row for row in rows if row.get('outcome') == 'failed']
    assert len(failed) == 1
    assert failed[0]['stage'] == 'scene_publish'
    assert failed[0]['error'] == 'RuntimeError: scene publisher did not complete'
    assert failed[0]['duration_s'] >= 0.
    assert not any(row.get('name') == 'packet_completed' for row in rows)
    assert not (renderer.output/'publication.jsonl').exists()
    current = json.loads((renderer.output/'runtime/publication_current.json').read_text())
    assert current['outcome'] == 'failed'
    assert current['active_stage'] is None
