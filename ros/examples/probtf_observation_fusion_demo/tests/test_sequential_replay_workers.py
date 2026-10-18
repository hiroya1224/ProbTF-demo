"""Scheduling frontiers do not replace chronological sensor Bayes updates."""
from dataclasses import replace
import json
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pytest

from probtf_observation_fusion_demo import async_workers
from probtf_observation_fusion_demo.async_artifacts import (
    capture_candidate_identity, identity_hash, load_stiffness_job,
)
from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.sequential_replay import (
    command_path_to_event, selected_sensor_event_ids, validate_claimed_event,
)

from test_joint_batch_inference import acquired


def run_replay(tmp_path, acquired, selection=None):
    description, inputs, templates = acquired
    output = tmp_path/'output'
    output.mkdir()
    urdf = tmp_path/'robot.urdf'
    urdf.write_text(description)
    database = output/'events.sqlite3'
    commands = [tuple(np.asarray(inputs.command) + index*np.array(
        [.0002, 0., -.0001, 0., 0., 0.])) for index in range(5)]
    stamps = [inputs.stamp+.05*index for index in range(5)]
    ledger = tuple(ActionStep(index, stamps[index], stamps[index+1],
        commands[index], commands[index+1], (.00015, 0., -.00007, 0., 0., 0.),
        density_id='captured-K:'+str(index), selected_K=(100.,)*6)
        for index in range(4))
    requests, replies, K_requests = Queue(), Queue(), Queue()
    thread = Thread(target=async_workers.event_inference_worker,
        args=(requests, replies, description, str(urdf), templates,
              str(output), 'per_effective_sensor_event', K_requests), daemon=True)
    with SensorEventStore(database) as store:
        def append(identity, kind, index, *, missing=False):
            payload = dict(time=.05*index, frame_index=index)
            if kind == 'joint':
                payload['command_velocity'] = [0.]*6
            elif kind == 'imu':
                payload.update(gravity_in_sensor=inputs.gravity_in_sensor,
                    imu_target_link=inputs.imu_target_link, imu_sigma_rad=inputs.imu_sigma_rad)
            else:
                payload.update(image=np.zeros_like(inputs.image) if missing else inputs.image,
                               image_stamp=stamps[index])
            return store.append(identity, kind, stamps[index], commands[index], payload)

        append('joint:0', 'joint', 0)
        append('imu:0', 'imu', 0)
        append('image:0', 'image', 0)
        append('joint:1', 'joint', 1)
        append('imu:1', 'imu', 1)
        append('image:2', 'image', 2)
        append('image:3:missing', 'image', 3, missing=True)
        append('joint:4', 'joint', 4)
        thread.start()

        def run(initial):
            batch = store.claim_batch('inference', max_stamp=stamps[-1],
                snapshot_id='initial' if initial else 'all-ready',
                initial_command_only=initial)
            request = dict(kind='sensor_event_batch', batch=batch, database=str(database),
                           ledger_steps=() if initial else ledger)
            if not initial and selection is not None:
                request['observation_selection'] = dict(selected_event_ids=selection)
            requests.put(request)  # The default must be sequential, no flag required.
            messages = []
            while True:
                message = replies.get(timeout=45.)
                assert not message.get('failed'), message
                messages.append(message)
                if message.get('kind') == 'inference_batch_completed':
                    return messages

        try:
            initial = run(True)
            observed = run(False)
        finally:
            requests.put(None)
            thread.join(timeout=10.)
            assert not thread.is_alive()
        events = tuple(store.get(event.event_id) for event in store.events())
    archives = []
    while not K_requests.empty():
        archives.append(K_requests.get_nowait())
    return dict(description=description, urdf=urdf, templates=templates,
        output=output, initial=initial, observed=observed, archives=archives,
        events=events, ledger=ledger, stamps=stamps)


def test_frontier_replays_each_sensor_against_previous_canonical_without_waiting_for_K(tmp_path, acquired):
    value = run_replay(tmp_path, acquired)
    ready = [message for message in value['observed']
             if message.get('kind') == 'inference_body_ready']
    expected = ['imu:0', 'image:0', 'imu:1', 'image:2', 'image:3:missing']
    assert [message['event_id'] for message in ready] == expected
    assert [message['assimilated_event_ids'] for message in ready] == [(identity,) for identity in expected]
    assert all(message['packet'] is None for message in ready)
    assert [reference['job_sequence'] for reference in value['archives']] == list(range(5))
    assert [message['stiffness_job_sequence'] for message in ready] == [1, 2, 3, 4, None]
    assert ready[-1]['K_processed'] and not ready[-1]['stiffness_pending']
    assert ready[-1]['result'].records['sequential_source']['last_effective_observation_stamp'] == value['stamps'][2]
    assert ready[-1]['result'].records['fixed_goal_memory']['observation_stamp'] == value['stamps'][2]
    assert ready[-1]['result'].prior.stamp == value['stamps'][3]
    assert all(message['body_update_seconds'] > 0. for message in ready)

    # An independent ordinary recurrence must produce the exact same law,
    # including cross covariance and each fresh image-time goal.
    model = MovingInference(value['description'], value['templates'])
    expected_by_id = {message['event_id']: message for message in ready}
    previous = None
    index = None
    for event in value['events']:
        if event.event_id != 'joint:0' and event.event_id not in expected_by_id:
            continue
        endpoint = event.payload['frame_index']
        steps = () if previous is None else value['ledger'][index:endpoint]
        path = (None if previous is None else RecordedCommandPath(previous, steps,
                body_ids=model.body_ids, actions=model.actions))
        result = model.update_sensor_event(event, event_index=endpoint,
            command_map=path, probability_unit='per_effective_sensor_event')
        if event.event_id in expected_by_id:
            actual = expected_by_id[event.event_id]['result']
            left = result.stages[-1].components[0].state
            right = actual.stages[-1].components[0].state
            assert left.state_id == right.state_id
            np.testing.assert_array_equal(left.covariance, right.covariance)
            assert [component.state.state_id for component in result.stages[-1].goal_components] == [
                component.state.state_id for component in actual.stages[-1].goal_components]
        previous, index = result, endpoint

    for reference, identity in zip(value['archives'], ['joint:0']+expected[:-1]):
        job = load_stiffness_job(reference)
        assert job['update_mode'] == 'sequential_sensor_event'
        assert [event['event_id'] for event in job['events']] == [identity]
        result = job['result']
        assert 'joint_batch' not in result.records
        assert not result.time_resolved_body_ratios
        assert result.frozen_body_prior is not None
        assert result.frozen_body_posterior is not None
        assert result.image.size == result.detection_image.size == 0
        assert capture_candidate_identity(result) == job['source_candidate_identity']
        assert identity_hash(job['source_candidate_identity']) == job['source_candidate_id']
        if identity != 'joint:0':
            assert result.records['observation_ids'] == [identity]
            assert result.records['bayes_factor_surprise']['change_probability'] == .25
            assert len(result.frozen_body_branches) == 2
    assert all(event.status == 'completed' for event in value['events'])
    record = value['observed'][-1]['batch_record']
    assert record['stiffness_job_sequences'] == [1, 2, 3, 4]
    assert record['body_ready_already_handed_off']
    assert not record['hazard_aggregation_applied']
    assert not record['joint_body_batch']
    assert record['body_update_seconds_by_sensor_kind']['imu']['count'] == 2
    assert record['body_update_seconds_by_sensor_kind']['image']['count'] == 3


def test_thinned_events_do_not_update_body_BFS_or_K_but_actual_ledger_is_complete(tmp_path, acquired):
    value = run_replay(tmp_path, acquired, selection=['image:0', 'image:2'])
    ready = [message for message in value['observed']
             if message.get('kind') == 'inference_body_ready']
    assert [message['event_id'] for message in ready] == ['image:0', 'image:2']
    jobs = [load_stiffness_job(reference) for reference in value['archives']]
    assert [job['events'][0]['event_id'] for job in jobs] == ['joint:0', 'image:0', 'image:2']
    assert [step['index'] for step in jobs[-1]['command_ledger']] == [0, 1]
    assert [step['density_id'] for step in jobs[-1]['command_ledger']] == ['captured-K:0', 'captured-K:1']
    assert jobs[-1]['result'].prior.command == value['events'][5].command
    assert jobs[-1]['result'].records['bayes_factor_surprise']['change_probability'] == .25
    for event in value['events']:
        if event.event_id in ('imu:0', 'imu:1', 'image:3:missing'):
            assert event.result_metadata['reason'] == 'temporal_thinning'
            assert not event.result_metadata['observation_fused']
            assert not event.result_metadata['body_update_applied']
            assert event.result_metadata['stiffness_job_sequence'] is None
    assert value['observed'][-1]['batch_record']['completed_event_ids'] == [event.event_id for event in value['events'][1:]]


def test_independent_K_worker_uses_single_event_hazard_and_original_body_laws(tmp_path, acquired, monkeypatch):
    value = run_replay(tmp_path, acquired, selection=['imu:0', 'image:0'])
    # The old joint-batch routine must never be called for these archives.
    monkeypatch.setattr(async_workers.StiffnessIdentifier, 'update_batch',
        lambda *args, **kwargs: pytest.fail('A sequential K event cannot aggregate its hazard'))
    requests, replies = Queue(), Queue()
    for reference in value['archives']:
        requests.put(reference)
    requests.put(None)
    async_workers.stiffness_worker(requests, replies, value['description'], str(value['urdf']),
                                  value['templates'], str(value['output']))
    results = [replies.get_nowait() for _ in value['archives']]
    assert all(message['kind'] == 'stiffness_ready' for message in results), results
    assert [message['job_sequence'] for message in results] == [0, 1, 2]
    assert results[0]['packet'] is None
    assert [message['stiffness_record']['observation_ids'] for message in results[1:]] == [['imu:0'], ['image:0']]
    assert all('joint_batch' not in message['stiffness_record'] for message in results)
    assert results[-1]['packet'] is not None


def test_replay_selection_claim_and_actual_endpoint_validation(tmp_path, acquired):
    description, inputs, templates = acquired
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        event = store.append('imu:0', 'imu', inputs.stamp, inputs.command,
            dict(frame_index=0, gravity_in_sensor=inputs.gravity_in_sensor))
        batch = store.claim_batch('inference', max_stamp=inputs.stamp, snapshot_id='one')
        metadata = batch.events[0]
        claimed = store.get(event.event_id)
        validate_claimed_event(claimed, metadata, 'inference')
        with pytest.raises(ValueError, match='changed after claim'):
            validate_claimed_event(replace(claimed, kind='image'), metadata, 'inference')
        with pytest.raises(ValueError, match='identities and order'):
            selected_sensor_event_ids(batch.events, {'imu'}, {'selected_event_ids': ['foreign']})
        with pytest.raises(ValueError, match='identities and order'):
            selected_sensor_event_ids(batch.events, {'imu'}, {'selected_event_ids': ['imu:0', 'imu:0']})
        with pytest.raises(ValueError, match='chronological acquisition'):
            selected_sensor_event_ids((replace(metadata, event_id='later',
                sequence=metadata.sequence+1), metadata), {'imu'})
        model = MovingInference(description, templates)
        with pytest.raises(ValueError, match='initialize from a joint'):
            command_path_to_event(None, None, claimed, {}, model)
