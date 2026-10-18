"""Readiness is independent of K and rendering; immutable sources survive."""
from dataclasses import replace
import json
import pickle
from pathlib import Path
from queue import Queue
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo import async_workers
from probtf_observation_fusion_demo.async_artifacts import (
    archive_stiffness_job, capture_candidate_identity, identity_hash, load_stiffness_job,
)
from probtf_observation_fusion_demo.display import verify_wire_identity
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore

from test_joint_batch_inference import acquired, batch_problem


@pytest.fixture
def deferred_body(tmp_path, acquired):
    description, inputs, templates = acquired
    urdf = tmp_path/'robot.urdf'
    urdf.write_text(description)
    output = tmp_path/'output'
    output.mkdir()
    database = output/'events.sqlite3'
    jobs, replies, K_jobs = Queue(), Queue(), Queue()
    thread = Thread(target=async_workers.event_inference_worker,args=(jobs,replies,
        description,str(urdf),templates,str(output),'per_effective_sensor_event',K_jobs),daemon=True)
    with SensorEventStore(database) as store:
        store.append('joint:0','joint',inputs.stamp,inputs.command,
            dict(frame_index=0,time=0.,command_velocity=[0.]*6))
        store.append('imu:0','imu',inputs.stamp,inputs.command,
            dict(frame_index=0,time=0.,gravity_in_sensor=inputs.gravity_in_sensor,
                imu_target_link=inputs.imu_target_link,imu_sigma_rad=inputs.imu_sigma_rad))
        store.append('image:0','image',inputs.stamp,inputs.command,
            dict(frame_index=0,time=0.,image_stamp=inputs.stamp,image=inputs.image))
        thread.start()

        def run_batch(initial):
            batch = store.claim_batch('inference',max_stamp=inputs.stamp,
                snapshot_id='initial' if initial else 'all-sensors',initial_command_only=initial)
            jobs.put(dict(kind='sensor_event_batch',batch=batch,database=str(database),
                ledger_steps=(),joint_body_batch=True))
            messages = []
            while True:
                reply = replies.get(timeout=30.)
                assert not reply.get('failed'),reply
                messages.append(reply)
                if reply.get('kind') == 'inference_batch_completed':
                    return messages
        try:
            initial = run_batch(True)
            observed = run_batch(False)
            yield dict(description=description,urdf=urdf,templates=templates,output=output,
                initial=initial,observed=observed,K_jobs=K_jobs,store=store)
        finally:
            jobs.put(None)
            thread.join(timeout=10.)
            assert not thread.is_alive()


def test_new_body_and_every_ratio_are_ready_before_K_worker_runs(deferred_body):
    value = deferred_body
    assert not any(row.get('kind') == 'inference_body_ready' for row in value['initial'])
    initial = value['initial'][-1]
    assert initial['final_message']['result'].records['observation_ids'] == []
    assert initial['final_message']['packet'] is None
    assert initial['final_message']['stiffness_job_sequence'] == 0
    messages = value['observed']
    assert messages[0]['kind'] == 'inference_body_ready'
    ready = messages[0]
    assert ready['result'].records['observation_ids'] == ['imu:0','image:0']
    assert ready['packet'] is None and ready['stiffness_pending']
    assert ready['body_processed'] and not ready['K_processed']
    assert ready['stiffness_job_sequence'] == 1
    assert messages[-1]['batch_record']['body_ready_already_handed_off']
    assert value['store'].stats()['counts'] == {'completed':3}
    for event in value['store'].events():
        assert event.result_metadata['body_processed']
        assert event.result_metadata['K_processed'] is False
    first,second = value['K_jobs'].get_nowait(),value['K_jobs'].get_nowait()
    assert [first['job_sequence'],second['job_sequence']] == [0,1]
    job = load_stiffness_job(second)
    assert job['result'].image.size == job['result'].detection_image.size == 0
    assert job['result'].records['observation_ids'] == ['imu:0','image:0']
    assert [row['observation_id'] for row in job['result'].time_resolved_body_ratios] == ['imu:0','image:0']
    assert [row['event_id'] for row in job['events']] == ['imu:0','image:0']
    assert all(row['stamp'] == ready['result'].prior.stamp for row in job['events'])
    assert all('image' not in row and 'payload' not in row for row in job['events'])
    assert job['source_candidate_id'] == ready['body_candidate_id']
    assert identity_hash(capture_candidate_identity(job['result'])) == ready['body_candidate_id']


def complete_K(value):
    jobs = value['K_jobs']
    jobs.put(None)
    replies = Queue()
    async_workers.stiffness_worker(jobs,replies,value['description'],str(value['urdf']),
        value['templates'],str(value['output']))
    first,second = replies.get_nowait(),replies.get_nowait()
    assert first['kind'] == second['kind'] == 'stiffness_ready',(first,second)
    assert first['job_sequence'] == 0 and first['initialize_only'] and first['packet'] is None
    assert second['job_sequence'] == 1 and second['packet'] is not None
    assert second['stiffness_record']['observation_ids'] == ['imu:0','image:0']
    return second


def test_gradient_ready_precedes_all_artifacts_and_cached_K_preserves_identity(deferred_body):
    value = deferred_body
    estimated = complete_K(value)
    body = value['observed'][0]['result']
    jobs,replies,artifacts = Queue(),Queue(),Queue()
    thread = Thread(target=async_workers.control_worker,args=(jobs,replies,value['description'],
        str(value['urdf']),value['templates'],str(value['output']),.5,'running_gradient',
        {'enabled':True,'weight':2000000.},artifacts),daemon=True)
    request = dict(result=body,packet=estimated['packet'],action_packet=None,
        steps=(),stamp=body.prior.stamp,index=0,capture_index=0,
        request_id=1,event_id='image:0',snapshot_update_count=2,velocity=np.zeros(6))
    thread.start()
    try:
        jobs.put(request)
        first = replies.get(timeout=30.)
        assert first['kind'] == 'control_gradient_ready',first
        assert first['gradient_evaluation']['status'] == 'ok'
        assert 'snapshot' not in first and 'markers_wire' not in first and 'packet' not in first
        assert len(pickle.dumps(first,protocol=pickle.HIGHEST_PROTOCOL)) < 256*1024
        assert not (value['output']/'control.jsonl').exists()
        assert not list(value['output'].glob('*.markers.bin'))
        # The consumer can request another gradient without running rendering.
        jobs.put(dict(request,request_id=2,packet=None,
            parameter_packet_id=estimated['packet'].packet_id,
            action_packet_ref=first['action_packet_ref']))
        second = replies.get(timeout=30.)
        assert second['kind'] == 'control_gradient_ready',second
        assert second['gradient_evaluation']['selected_K'] == first['gradient_evaluation']['selected_K']
        assert second['candidate_identity']['body'] == first['candidate_identity']['body']
    finally:
        jobs.put(None)
        thread.join(timeout=15.)
        assert not thread.is_alive()
    artifacts.put(None)
    rendered = Queue()
    async_workers.artifact_worker(artifacts,rendered,value['description'],str(value['urdf']),
        value['templates'],str(value['output']))
    for ready in (first,second):
        artifact = rendered.get_nowait()
        assert artifact['kind'] == 'control_artifacts_ready',artifact
        assert artifact['candidate_identity'] == ready['candidate_identity']
        assert artifact['candidate_id'] == ready['candidate_id']
        from visualization_msgs.msg import MarkerArray
        message = MarkerArray().deserialize(artifact['markers_wire'])
        from probtf_observation_fusion_demo.moving_inference import MovingInference
        display = MovingInference(value['description'],value['templates']).common['display']
        assert verify_wire_identity(artifact['snapshot'],message,seed=display['seed'],point_size=display['point_size'])


def test_numeric_cross_covariance_hash_detects_same_ID_but_different_law(deferred_body):
    result = deferred_body['observed'][0]['result']
    expected = capture_candidate_identity(result)
    stage = result.stages[-1]
    component = stage.components[0]
    state = component.state
    changed = state.covariance.copy()
    changed[0,6] += 1e-10
    changed[6,0] += 1e-10
    # Deliberately stale state_id simulates a broken/mislabeled handoff.
    def stale_id_proxy(original):
        covariance = original.covariance.copy()
        covariance[0,6] += 1e-10
        covariance[6,0] += 1e-10
        return SimpleNamespace(covariance=covariance,**{name:getattr(original,name)for name in (
            'state_id','mean','transforms','component_id','edge_ids','command','stamp','version')})
    impostor = stale_id_proxy(state)
    altered = replace(result,stages=(replace(stage,
        components=(replace(component,state=impostor),),
        goal_components=tuple(replace(item,state=stale_id_proxy(item.state))
            for item in stage.goal_components)),))
    actual = capture_candidate_identity(altered)
    assert actual['body'][0]['state_id'] == expected['body'][0]['state_id']
    assert actual['body'][0]['numeric_law_sha256'] != expected['body'][0]['numeric_law_sha256']
    assert identity_hash(actual) != identity_hash(expected)


def test_failed_K_archive_is_retained_and_later_FIFO_jobs_are_not_skipped(deferred_body,monkeypatch):
    value = deferred_body
    first,second = value['K_jobs'].get_nowait(),value['K_jobs'].get_nowait()
    path = Path(first['path'])
    path.write_bytes(path.read_bytes()+b'broken')
    jobs,replies = Queue(),Queue()
    jobs.put(first);jobs.put(second);jobs.put(None)
    async_workers.stiffness_worker(jobs,replies,value['description'],str(value['urdf']),
        value['templates'],str(value['output']))
    failed,blocked = replies.get_nowait(),replies.get_nowait()
    assert failed['kind'] == 'stiffness_failed'
    assert not failed['K_processed'] and failed['old_K_retained']
    assert 'payload changed' in failed['failure_reason']
    assert blocked['kind'] == 'stiffness_blocked'
    assert blocked['blocked_by_job_sequence'] == 0
    assert blocked['job_sequence'] == 1 and not blocked['K_processed']
    assert Path(second['path']).exists()
    assert load_stiffness_job(second)['result'].records['observation_ids'] == ['imu:0','image:0']
    assert not (value['output']/'stiffness.jsonl').exists()


def test_archive_preserves_distinct_acquisition_commands_and_all_cross_blocks(batch_problem,tmp_path):
    model,events,ledger,_ = batch_problem
    result = model.update_sensor_batch(events,ledger[:3],batch_id='distinct-acquisitions')
    identity = capture_candidate_identity(result)
    metadata = tuple(dict(event_id=event.event_id,kind=event.kind,sequence=event.sequence,
        stamp=event.stamp,command=event.command,payload_hash=event.payload_hash)for event in events)
    job = dict(job_sequence=4,batch_id='distinct-acquisitions',initialize_only=False,
        result=replace(result,image=np.empty((0,0)),detection_image=np.empty((0,0)),
            sensor_posteriors=(),local_estimates=()),events=metadata,
        command_ledger=result.records['joint_batch']['actual_ledger'],
        source_candidate_identity=identity,source_candidate_id=identity_hash(identity))
    restored = load_stiffness_job(archive_stiffness_job(tmp_path,job))
    ratios = restored['result'].time_resolved_body_ratios
    assert [item['source_command']for item in ratios] == [events[i].command for i in (1,2,3)]
    assert all(item['source_command'] != result.prior.command for item in ratios)
    assert restored['command_ledger'] == [step.record()for step in ledger[:3]]
    for before,after in zip(result.time_resolved_body_ratios,ratios):
        assert before['source_stamp'] == after['source_stamp']
        assert before['provenance']['payload_hash'] == after['provenance']['payload_hash']
        for old,new in zip(before['branches'],after['branches']):
            assert old.reference_state.state_id == new.reference_state.state_id
            assert old.log_weight == new.log_weight
            np.testing.assert_array_equal(old.prior.mean,new.prior.mean)
            np.testing.assert_array_equal(old.prior.covariance,new.prior.covariance)
            for left,right in zip(old.posterior_components,new.posterior_components):
                assert left.log_weight == right.log_weight
                np.testing.assert_array_equal(left.distribution.mean,right.distribution.mean)
                np.testing.assert_array_equal(left.distribution.covariance,right.distribution.covariance)
