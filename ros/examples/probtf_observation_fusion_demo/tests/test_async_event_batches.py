"""Whole-frontier scheduling and an actual numerical batch handoff."""

from pathlib import Path
import json
import time
from threading import Thread

import pytest

from probtf_observation_fusion_demo.async_motion import AsyncMotionSession
from probtf_observation_fusion_demo.async_workers import event_inference_worker

from test_async_execution import Context, LocalQueue, _complete_acquisitions, _complete_control


PACKAGE = Path(__file__).resolve().parents[1]


def _session(tmp_path,case='imu_marker'):
    clock = [100.]
    current = AsyncMotionSession(PACKAGE/('config/'+case+'.yaml'), tmp_path/'run',
        clock=lambda: clock[0], context=Context(), duration=.1,
        control_mode='running_gradient')
    _complete_acquisitions(current)
    return current, clock


def _dispatch(current):
    for _ in range(4):
        current.submit_estimate()
        if current.inference_busy:
            return current.estimate_requests.get_nowait()
        _complete_acquisitions(current)
    raise AssertionError('The finite acquisition batch was not dispatched.')


@pytest.mark.parametrize('case,observation_ids,status',[
    ('marker',['image:00000000'],'ok'),
    ('imu_marker',['imu:00000000','image:00000000'],'ok'),
    ('imu',['imu:00000000'],'waiting_for_goal')])
def test_initial_prior_is_the_estimator_law_until_start_then_every_raw_observation_fuses(
        tmp_path,case,observation_ids,status):
    from probtf.spherical_law.first_order import point_query
    import numpy as np
    current,_=_session(tmp_path,case)
    requests=LocalQueue()
    worker=Thread(target=event_inference_worker,args=(requests,current.estimate_replies,
        current.scene.robot_description,str(current.scene.urdf),current.templates,
        str(current.output),'per_effective_sensor_event'),daemon=True)
    worker.start()
    def finish(request):
        requests.put(request)
        deadline=time.monotonic()+10.
        while current.inference_busy and time.monotonic()<deadline:
            current.poll()
            time.sleep(.005)
        assert not current.inference_busy
    def control():
        current.submit_control()
        reply=_complete_control(current)
        current.control_replies.put(reply)
        current.poll()
        return reply
    try:
        first=_dispatch(current)
        assert first['batch'].reason=='initial_command_prior_only'
        assert [event.kind for event in first['batch'].events]==['joint']
        assert all(not event.payload_loaded for event in first['batch'].events)
        finish(first)
        prior=current.latest_result
        assert prior.records['observation_ids']==[]
        assert current.latest_packet is None
        assert current.store.stats()['counts']=={'completed':1,'pending':2}
        assert current.last_assimilated_event=='joint:00000000'
        before=control()
        assert before['gradient_evaluation']['status']=='waiting_for_goal'
        declared=current.inference_description.prior_at(current.command,current.frame.stamp,0)
        for path in current.inference_description.paths['links'].values():
            actual=point_query(prior.stages[-1].components[0].state,path,[0.,0.,0.])
            expected=point_query(declared,path,[0.,0.,0.])
            np.testing.assert_array_equal(actual.mean,expected.mean)
            np.testing.assert_array_equal(actual.covariance,expected.covariance)
        initial_ids={item.query.state_id for item in before['snapshot'].points if item.frame_label=='link_1'}
        raw_before=[(event.event_id,event.stamp,event.command,event.payload_hash,event.attempts)
                    for event in current.store.events() if event.status=='pending']
        for _ in range(5):
            current.submit_estimate()
            assert not current.database_selection_pending and not current.inference_busy
        assert current.manifest()['startup_phase']=='initial_prior'
        assert not current.manifest()['observation_updates_enabled']
        assert raw_before==[(event.event_id,event.stamp,event.command,event.payload_hash,event.attempts)
                            for event in current.store.events() if event.status=='pending']
        assert current.start()
        second=_dispatch(current)
        assert [event.kind for event in second['batch'].events]==['imu','image']
        finish(second)
        assert current.completed_batches==2
        assert current.store.stats()['counts']=={'completed':3}
        assert current.latest_result.records['observation_ids']==observation_ids
        assert current.latest_packet is not None
        after=control()
        assert after['gradient_evaluation']['status']==status
        assert initial_ids!={item.query.state_id for item in after['snapshot'].points if item.frame_label=='link_1'}
        assert current.current_control['snapshot'] is after['snapshot']
        assert current.last_completed_batch['completed_event_ids']==list(second['batch'].event_ids)
        assert current.last_completed_batch['all_claimed_events_completed']
        for event in second['batch'].events:
            timing=current.store.get(event.event_id).result_metadata
            if event.event_id in observation_ids:
                assert timing['inference_seconds'] is None and timing['batch_work_seconds']>0.
            else:
                assert not timing['observation_fused']
                assert timing['reason']=='sensor not enabled in this comparison case'
    finally:
        requests.put(None)
        worker.join(timeout=10.)
        current.close()
        assert not worker.is_alive()


def test_checkpoint_from_another_batch_cannot_advance_the_frontier(tmp_path):
    current, _ = _session(tmp_path)
    try:
        request = _dispatch(current)
        event = request['batch'].events[0]
        current.estimate_replies.put(dict(kind='inference_batch_progress',
            batch_id='different_batch', batch_offset=0, event_id=event.event_id))
        with pytest.raises(ValueError, match='outside the claimed'):
            current.poll()
        assert current.batch_progress_count == 0
        assert current.latest_result is None
    finally:
        current.close()


def test_empty_frontier_retries_when_a_reserved_raw_payload_arrives(tmp_path):
    current, _ = _session(tmp_path)
    try:
        current.selected_sensor_batch = None
        current.database_empty_acquisition_watermark = (
            current.acquisition_submitted, current.acquisition_completed)
        current.submit_estimate()
        assert current.acquisition_requests.empty()
        # Receipt changes readiness even without a newer acquisition request.
        current.acquisition_completed += 1
        current.submit_estimate()
        job = current.acquisition_requests.get_nowait()
        assert job['operation'] == 'select_batch'
        assert current.database_selection_pending
    finally:
        current.close()


def test_changed_claimed_sensor_kind_is_rejected_without_completing_or_dropping_raw(tmp_path):
    current,_ = _session(tmp_path)
    try:
        request = _dispatch(current)
        original = request['batch'].events[0]
        # The raw command notification still has the same ID, time, q and
        # payload hash. A changed sensor kind must not be silently accepted
        # just because its fields happen to be valid for a disabled sensor.
        current.store.connection.execute('UPDATE events SET kind=? WHERE event_id=?',
                                         ('imu',original.event_id))
        current.store.connection.commit()
        requests = LocalQueue(); requests.put(request); requests.put(None)
        event_inference_worker(requests,current.estimate_replies,
            current.scene.robot_description,str(current.scene.urdf),
            tuple(item for item in current.templates if item['sensor']=='marker'),
            str(current.output),'per_effective_sensor_event')
        with pytest.raises(RuntimeError,match='immutable joint-batch acquisition changed'):
            current.poll()
        assert current.latest_result is None
        assert current.store.get(original.event_id).status == 'failed'
        assert current.store.stats()['total'] == 3
        assert not current.store.stats()['counts'].get('completed',0)
        for event in request['batch'].events:
            assert current.store.get(event.event_id).payload_loaded
    finally:
        current.close()


def test_required_stiffness_failure_preserves_input_and_actual_arithmetic_reason(tmp_path,monkeypatch):
    from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
    reason='required evidence Hessian Cholesky is not positive definite'
    failure=dict(stiffness_posterior_defined=False,failure_reason=reason,
        proposal_failure_reason='no valid posterior',
        evidence=dict(operation='Cholesky',input_logK=[1.,2.,3.,4.,5.,6.]))
    monkeypatch.setattr(StiffnessIdentifier,'update_batch',lambda *args,**kwargs:failure)
    monkeypatch.setattr(StiffnessIdentifier,'command_proposal',lambda *args:None)
    current,_=_session(tmp_path)
    try:
        request=_dispatch(current)
        requests=LocalQueue(); requests.put(request); requests.put(None)
        event_inference_worker(requests,current.estimate_replies,
            current.scene.robot_description,str(current.scene.urdf),
            current.templates,str(current.output),'per_effective_sensor_event')
        with pytest.raises(RuntimeError,match=reason):
            current.poll()
        saved=json.loads((current.output/'estimation_failure.jsonl').read_text())
        assert saved['stiffness']==failure
        assert saved['event_ids']==list(request['batch'].event_ids)
        assert current.latest_result is None and not current.completed_batches
        assert current.store.stats()['total']==3
        assert not current.store.stats()['counts'].get('completed',0)
        assert current.store.get(request['batch'].event_ids[0]).status=='failed'
        assert all(current.store.get(event.event_id).payload_loaded for event in request['batch'].events)
    finally:
        current.close()
