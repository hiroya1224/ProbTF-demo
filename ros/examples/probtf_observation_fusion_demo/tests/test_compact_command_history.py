"""Compact journal references change records, never transported candidates."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import struct

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_state import (
    ActionStep, RecordedCommandPath, predict_result,
)
from probtf_observation_fusion_demo.display import make_markers, snapshot_record
from probtf_observation_fusion_demo.node import build_snapshot

from test_async_state import steps_for
from test_measure_display_provenance import acquired, measured_source


INITIAL_HEAD = hashlib.sha256(b'ProbTF actual-command ledger v1').hexdigest()


def reference_for(steps, *, start_index=None, source_head=INITIAL_HEAD):
    head = source_head
    for step in steps:
        payload = json.dumps(step.record(), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()
        head = hashlib.sha256(bytes.fromhex(head)+payload).hexdigest()
    begin = steps[0].index if steps else (0 if start_index is None else start_index)
    return dict(path='commands.jsonl', start_index=begin,
        end_index_exclusive=begin+len(steps), step_count=len(steps),
        source_prefix_chain_sha256=source_head, prefix_chain_sha256=head)


def wire(message):
    output = io.BytesIO()
    message.serialize(output)
    return output.getvalue()


def test_compact_and_legacy_prediction_are_bitwise_same_law_measure_sensor_and_wire(measured_source):
    inference, source = measured_source
    steps = steps_for(source)
    reference = reference_for(steps)
    legacy = predict_result(source, steps, steps[-1].end_stamp, 3,
                            inference.body_ids, inference.actions)
    compact = predict_result(source, steps, steps[-1].end_stamp, 3,
                             inference.body_ids, inference.actions, ledger_reference=reference)
    for left, right in zip(legacy.stages[-1].components+legacy.stages[-1].goal_components,
                           compact.stages[-1].components+compact.stages[-1].goal_components):
        assert left.state.state_id == right.state.state_id
        assert left.weight == right.weight and left.log_weight == right.log_weight
        np.testing.assert_array_equal(left.state.covariance, right.state.covariance)
        for a, b in zip(left.state.transforms, right.state.transforms):
            if a.bingham is not None:
                np.testing.assert_array_equal(a.bingham, b.bingham)
    assert legacy.records['candidate_measure'] == compact.records['candidate_measure']
    assert struct.pack('!d', legacy.records['candidate_measure']['log_total_mass']) == struct.pack(
        '!d', compact.records['candidate_measure']['log_total_mass'])
    assert legacy.records['sensor_event'] == compact.records['sensor_event'] == source.records['sensor_event']
    full_path = legacy.records['async_prediction']['command_path']
    compact_path = compact.records['async_prediction']['command_path']
    assert full_path['steps'] == [step.record() for step in steps]
    assert 'steps' not in compact_path
    assert compact_path['step_count'] == len(steps)
    assert compact_path['first_interval_index'] == 0 and compact_path['last_interval_index'] == 2
    assert compact_path['ledger_reference'] == reference
    for key in ('joint_increment', 'source_stamp', 'source_command', 'source_prior_state_id',
                'body_posterior_state_ids', 'goal_source_state_ids', 'body_posterior_versions',
                'goal_source_versions', 'target_stamp', 'target_command', 'axes'):
        assert full_path[key] == compact_path[key]
    a = build_snapshot(legacy, legacy.stages[-1], inference.common)
    b = build_snapshot(compact, compact.stages[-1], inference.common)
    assert snapshot_record(a) == snapshot_record(b)
    settings = inference.common['display']
    full_wire = wire(make_markers(a, settings['seed'], settings['point_size']))
    compact_wire = wire(make_markers(b, settings['seed'], settings['point_size']))
    assert full_wire == compact_wire
    assert wire(MarkerArray().deserialize(compact_wire)) == full_wire


def test_compact_prediction_never_reexpands_historical_steps(measured_source, monkeypatch):
    inference, source = measured_source
    steps = steps_for(source)
    reference = reference_for(steps)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Compact prediction must not serialize every historical ActionStep again.')
    monkeypatch.setattr(ActionStep, 'record', forbidden)
    result = predict_result(source, steps, steps[-1].end_stamp, 3,
                            inference.body_ids, inference.actions, ledger_reference=reference)
    assert result.records['async_prediction']['command_path']['step_count'] == 3
    assert result.records['candidate_measure']['log_total_mass'] == source.records['candidate_measure']['log_total_mass']


def test_compact_reference_is_copied_and_selection_record_uses_same_actual_endpoint(measured_source):
    inference, source = measured_source
    steps = steps_for(source)
    path = RecordedCommandPath(source, steps, body_ids=inference.body_ids, actions=inference.actions)
    reference = reference_for(steps)
    record = path.selection_record(path.target_command, ledger_reference=reference)
    reference['path'] = 'changed.jsonl'
    assert record['ledger_reference']['path'] == 'commands.jsonl'
    assert record['command'] == list(path.target_command)
    assert record['command_jacobian'] is None
    assert 'steps' not in record


@pytest.mark.parametrize('field,value,message', [
    ('path', '', 'path'), ('path', 3, 'path'),
    ('start_index', True, 'integers'), ('step_count', 3., 'integers'),
    ('step_count', 2, 'range'), ('start_index', 1, 'range'),
    ('end_index_exclusive', 4, 'range'),
    ('prefix_chain_sha256', 'not a hash', 'SHA256'),
    ('source_prefix_chain_sha256', 'z'*64, 'SHA256'),
    ('source_prefix_chain_sha256', '0'*64, 'empty prefix'),
])
def test_compact_reference_rejects_metadata_and_action_range_mismatch(measured_source, field, value, message):
    inference, source = measured_source
    steps = steps_for(source)
    reference = reference_for(steps)
    reference[field] = value
    with pytest.raises(ValueError, match=message):
        predict_result(source, steps, steps[-1].end_stamp, 3,
                       inference.body_ids, inference.actions, ledger_reference=reference)


def test_compact_reference_rejects_wrong_target_physical_frame(measured_source):
    inference, source = measured_source
    steps = steps_for(source)
    with pytest.raises(ValueError, match='target physical frame index'):
        predict_result(source, steps, steps[-1].end_stamp, 4,
                       inference.body_ids, inference.actions, ledger_reference=reference_for(steps))


def test_empty_compact_range_retains_source_and_same_prefix_head(measured_source):
    inference, source = measured_source
    reference = reference_for(())
    result = predict_result(source, (), source.prior.stamp, 0,
                            inference.body_ids, inference.actions, ledger_reference=reference)
    record = result.records['async_prediction']['command_path']
    assert record['step_count'] == 0
    assert record['first_interval_index'] is record['last_interval_index'] is None
    assert record['ledger_reference']['source_prefix_chain_sha256'] == record['ledger_reference']['prefix_chain_sha256']
    assert record['joint_increment'] == [0.]*6
    reference['prefix_chain_sha256'] = '1'*64
    with pytest.raises(ValueError, match='empty ledger range'):
        predict_result(source, (), source.prior.stamp, 0,
                       inference.body_ids, inference.actions, ledger_reference=reference)
