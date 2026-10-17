"""Finite real-worker runs over the durable asynchronous event/command stores."""
import hashlib
import json
from queue import Queue

import numpy as np

from probtf_observation_fusion_demo.async_workers import control_worker, event_inference_worker
from probtf_observation_fusion_demo.async_stiffness import FrozenKPacket
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
from test_async_execution import _advance_ticks, _new_session


def test_real_inference_worker_processes_joint_IMU_and_image_at_one_stamp(tmp_path):
    current, _, _, _ = _new_session(tmp_path/'acquisition', case='imu_marker', duration=.1)
    try:
        events = current.store.events()
        assert [event.kind for event in events] == ['joint', 'imu', 'image']
        assert len({event.stamp for event in events}) == 1
        requests, replies = Queue(), Queue()
        for event in events:
            requests.put((event, 0, ()))
        requests.put(None)
        output = tmp_path/'worker'; output.mkdir()
        event_inference_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(output),
            'per_effective_sensor_event')
        results = [replies.get_nowait() for _ in events]
        assert all(not reply.get('failed') for reply in results), results
        assert [reply['event_id'] for reply in results] == [event.event_id for event in events]
        assert [reply['event_sequence'] for reply in results] == [event.sequence for event in events]
        assert [reply['sensor_kind'] for reply in results] == ['joint', 'imu', 'image']
        assert [reply['capture_index'] for reply in results] == [0, 0, 0]
        assert results[0]['packet'] is None
        assert all(isinstance(reply['packet'], FrozenKPacket) for reply in results[1:])
        records = [json.loads(line) for line in (output/'estimation.jsonl').read_text().splitlines()]
        assert len(records) == 3
        previous_ids = None
        for event, reply, record in zip(events, results, records):
            assert record['sensor_event']['event_id'] == event.event_id
            assert record['sensor_event']['kind'] == event.kind
            scheduling = record['async_estimation']
            assert scheduling['command_action_count'] == 0
            assert not scheduling['newest_only_selection']
            assert scheduling['posterior_retained']
            assert not scheduling['physical_truth_received']
            assert not scheduling['monte_carlo_used']
            assert reply['result'].records['sensor_event'] == record['sensor_event']
            ids = tuple(item.state.state_id for item in reply['result'].stages[-1].components)
            if previous_ids is not None:
                assert ids != previous_ids
            previous_ids = ids
        assert records[0]['observation_ids'] == []
        assert records[1]['observation_ids'] and records[2]['observation_ids']
        assert set(records[1]['observation_ids']).isdisjoint(records[2]['observation_ids'])
        assert records[1]['bayes_factor_surprise']['applied']
        assert records[2]['bayes_factor_surprise']['applied']
    finally:
        current.close()


def test_initial_undefined_stiffness_posterior_fails_even_without_retained_factors(tmp_path, monkeypatch):
    current, _, _, _ = _new_session(tmp_path/'acquisition', duration=.1)
    try:
        event = current.store.get('imu:00000000')
        observed = {}
        reason = 'initial stiffness Bayes-factor posterior is undefined'
        def undefined_update(identifier, result):
            assert not identifier.factors
            assert result.records['sensor_event']['event_id'] == event.event_id
            assert result.records['observation_ids']
            observed['body_source_state_ids'] = [
                item.state.state_id for item in result.stages[-1].components]
            return dict(stiffness_posterior_defined=False, posterior=None,
                proposal_status='undefined_stiffness_posterior',
                proposal_failure_reason=reason)
        def undefined_proposal(identifier, _result):
            assert not identifier.factors
            observed['proposal_called'] = True
            return None
        monkeypatch.setattr(StiffnessIdentifier, 'update', undefined_update)
        monkeypatch.setattr(StiffnessIdentifier, 'command_proposal', undefined_proposal)
        requests, replies = Queue(), Queue()
        requests.put((event, 0, ())); requests.put(None)
        output = tmp_path/'worker'; output.mkdir()
        event_inference_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(output),
            'per_effective_sensor_event')
        failure = replies.get_nowait()
        assert failure['failed'] and failure['worker'] == 'inference'
        assert reason in failure['traceback']
        assert 'result' not in failure and 'packet' not in failure and replies.empty()
        assert observed['proposal_called']
        records = [json.loads(line) for line in (output/'estimation_failure.jsonl').read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record['event_id'] == event.event_id and record['event_sequence'] == event.sequence
        assert record['sensor_kind'] == 'imu' and record['capture_index'] == 0
        assert record['stiffness']['stiffness_posterior_defined'] is False
        assert record['stiffness']['proposal_failure_reason'] == reason
        assert record['body_source_state_ids'] == observed['body_source_state_ids']
        assert not (output/'estimation.jsonl').exists()
    finally:
        current.close()


def test_persistent_control_worker_reconstructs_every_actual_ledger_delta(tmp_path):
    current, clock, initial, _ = _new_session(tmp_path/'acquisition', duration=.1)
    try:
        _advance_ticks(current, clock, 4)
        steps = tuple(current.actions)
        assert len(steps) == 4 and [step.index for step in steps] == list(range(4))
        requests, replies = Queue(), Queue()
        common = dict(result=initial, capture_index=0, packet=None, action_packet=None,
            velocity=np.zeros(6), event_id='joint:00000000', ledger_source_index=0)
        requests.put(dict(common, request_id=0, index=0, stamp=initial.prior.stamp,
            ledger_delta=(), ledger_target_index=0))
        requests.put(dict(common, request_id=1, index=4, stamp=current.frame.stamp,
            ledger_delta=steps, ledger_target_index=4))
        # A second completed computation may refer to the same physical frame
        # (for example while the final sensor backlog drains).
        requests.put(dict(common, request_id=2, index=4, stamp=current.frame.stamp,
            ledger_delta=(), ledger_target_index=4))
        requests.put(None)
        output = current.output
        control_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(output), .5)
        first, caught_up, same_frame = [replies.get_nowait() for _ in range(3)]
        assert all(not reply.get('failed') for reply in (first, caught_up, same_frame))
        assert first['request_id'] == 0 and caught_up['request_id'] == 1
        assert caught_up['diagnostics']['source_stamp'] == current.frame.stamp
        assert caught_up['diagnostics']['source_command'] == list(current.command)
        assert caught_up['packet'] is None
        records = [json.loads(line) for line in (output/'control.jsonl').read_text().splitlines()]
        assert [record['request_id'] for record in records] == [0, 1, 2]
        assert len({record['artifact_prefix'] for record in records}) == 3
        assert [record['event_id'] for record in records] == ['joint:00000000']*3
        for record, expected_count in zip(records, (0, 4, 4)):
            path = record['prediction']['command_path']
            assert path.get('step_count', len(path.get('steps', ()))) == expected_count
            if 'steps' in path:
                assert len(path['steps']) == expected_count
            else:
                reference = path['ledger_reference']
                assert (output/reference['path']).is_file()
                digest = hashlib.sha256(b'ProbTF actual-command ledger v1').digest()
                for step in steps[:expected_count]:
                    encoded = json.dumps(step.record(), sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()
                    digest = hashlib.sha256(digest+encoded).digest()
                assert reference['prefix_chain_sha256'] == digest.hex()
                assert reference['start_index'] == 0
                assert reference['end_index_exclusive'] == expected_count
        assert records[1]['prediction']['command_path']['physical_truth_received'] is False
        assert caught_up['snapshot'].command == tuple(current.command)
        assert caught_up['snapshot'].stamp == current.frame.stamp
        assert same_frame['diagnostics']['source_command'] == caught_up['diagnostics']['source_command']
        for record in records:
            assert (output/record['artifact_prefix']).with_suffix('.markers.bin').is_file()
    finally:
        current.close()


def test_control_worker_rejects_missing_actual_ledger_intervals(tmp_path):
    current, _, initial, _ = _new_session(tmp_path/'acquisition', duration=.1)
    try:
        requests, replies = Queue(), Queue()
        requests.put(dict(result=initial, capture_index=0, packet=None,
            request_id=0, index=1, stamp=initial.prior.stamp+.01,
            ledger_delta=(), ledger_source_index=0, ledger_target_index=1,
            velocity=np.zeros(6)))
        requests.put(None)
        output = tmp_path/'worker'; output.mkdir()
        control_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(output), .5)
        error = replies.get_nowait()
        assert error['failed'] and error['worker'] == 'control'
        assert 'actual source/target frames' in error['traceback']
        assert not (output/'control.jsonl').exists()
    finally:
        current.close()
