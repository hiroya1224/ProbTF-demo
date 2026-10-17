"""Lossless sensor-event scheduling and direct-K asynchronous execution."""
import io
import json
from pathlib import Path
import pickle
from queue import Empty, Queue

import numpy as np
import pytest
import yaml
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_motion import AsyncMotionSession, _freeze_arrays
from probtf_observation_fusion_demo.async_state import RecordedCommandPath, predict_result
from probtf_observation_fusion_demo.async_stiffness import FrozenActionKPacket, capture_stiffness_packet
from probtf_observation_fusion_demo.async_workers import _handoff_result, control_worker
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.marker_search import marker_observed
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier

PACKAGE = Path(__file__).resolve().parents[1]


class FakeProcess:
    def __init__(self, target, args, **_kwargs):
        self.target, self.args = target, args
        self.exitcode = None
        self.running = False
        self.terminated = False
    def start(self):
        self.running = True
    def is_alive(self):
        return self.running
    def join(self, **_kwargs):
        incoming = self.args[0]
        with incoming.mutex:
            stopped = any(item is None for item in incoming.queue)
        if stopped:
            self.running, self.exitcode = False, 0
    def terminate(self):
        self.running, self.terminated, self.exitcode = False, True, -15


class LocalQueue(Queue):
    def cancel_join_thread(self):
        pass
    def close(self):
        pass


class Context:
    Queue = LocalQueue
    Process = FakeProcess


def _complete_acquisitions(current):
    """Run the actual acquisition worker on this finite batch and real DB."""
    jobs = LocalQueue()
    count = processed = 0
    while True:
        try:
            request = current.acquisition_requests.get_nowait()
            jobs.put(request)
            processed += 1
            count += request['kind'] in ('joint', 'imu', 'image')
        except Empty:
            break
    if processed:
        jobs.put(None)
        process = current.workers[2]
        process.target(jobs, *process.args[1:])
        current.collect_archives()
    return count


def _select_next_estimate(current):
    """Deliver the asynchronous DB selection/ACK before inference can start."""
    for _ in range(4):
        current.submit_estimate()
        if current.inference_busy:
            return True
        if current.acquisition_requests.empty():
            return False
        _complete_acquisitions(current)
    raise AssertionError('Finite DB handshake did not dispatch its selected event.')


def _complete_estimate(current, previous_result, identifier, *, acknowledge=True):
    """Complete exactly the claimed oldest event with the actual estimator."""
    assert _select_next_estimate(current)
    event, index, steps = current.estimate_requests.get_nowait()
    inference = current._test_inference
    path = (None if previous_result is None else RecordedCommandPath(previous_result, steps,
        body_ids=inference.body_ids, actions=inference.actions))
    result = inference.update_sensor_event(event, event_index=index, command_map=path,
        probability_unit='per_effective_sensor_event')
    result.records['stiffness'] = identifier.update(result)
    proposal = identifier.command_proposal(result)
    packet = None if proposal is None else capture_stiffness_packet(result)
    transferred = _handoff_result(result)
    current.estimate_replies.put(dict(result=transferred, packet=packet,
        event_id=event.event_id, event_sequence=event.sequence, sensor_kind=event.kind,
        capture_index=index, capture_stamp=event.stamp,
        marker_observed=marker_observed(result), work_seconds=0.))
    current.poll()
    if acknowledge:
        _complete_acquisitions(current)
    return transferred, packet, index


def _complete_control(current):
    """The actual control worker owns its separate mechanics and serialization."""
    request = current.control_requests.get_nowait()
    if 'ledger_delta' in request:
        ledger = getattr(current, '_test_control_ledger', [])
        for step in request['ledger_delta']:
            assert step.index == len(ledger)
            ledger.append(step)
        current._test_control_ledger = ledger
        assert request['ledger_target_index'] == len(ledger)
        # This finite synchronous worker starts afresh; give it the identical
        # complete prefix that its persistent counterpart already retained.
        request = dict(request, ledger_delta=tuple(ledger))
    requests, replies = LocalQueue(), LocalQueue()
    requests.put(request); requests.put(None)
    control_worker(requests, replies, current.scene.robot_description, str(current.scene.urdf),
        current.templates, str(current.output), current.horizon, current.control_mode,
        current.visibility_options)
    reply = replies.get_nowait()
    assert not reply.get('failed'), reply
    return reply


def _new_session(tmp_path, case='imu', duration=1., control_mode='single_step_ik', visibility_factor_options=None):
    clock = [100.]
    current = AsyncMotionSession(PACKAGE/('config/'+case+'.yaml'), tmp_path/'run',
        clock=lambda: clock[0], context=Context(), duration=duration, control_mode=control_mode,
        visibility_factor_options=visibility_factor_options)
    current.probability_unit = 'per_effective_sensor_event'
    current._test_inference = MovingInference(current.scene.robot_description, current.templates)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(current.scene.urdf, current.inference_description.body_ids, settings)
    _complete_acquisitions(current)
    current.submit_estimate()
    result, _, _ = _complete_estimate(current, None, identifier)
    assert current.start()
    return current, clock, result, identifier


@pytest.fixture
def session(tmp_path):
    value = _new_session(tmp_path)
    yield value
    value[0].close()


def _advance_ticks(current, clock, ticks, *, hz=120.):
    for index in range(1, ticks+1):
        clock[0] = 100.+index/hz
        current.advance()


def _drain_estimates(current, previous, identifier):
    for _ in range(200):
        if _select_next_estimate(current):
            previous, _, _ = _complete_estimate(current, previous, identifier)
        elif current.store.stats()['earliest_unfinished'] is None:
            return previous
    raise AssertionError('Finite acquired event set did not drain.')


def test_120hz_physics_and_independent_40_30_12hz_acquisition_continue_while_workers_are_busy(session):
    current, clock, result, identifier = session
    current.submit_control()
    plan = _complete_control(current)
    current.control_replies.put(plan); current.poll()
    assert _select_next_estimate(current)
    factor_ids = tuple(x.factor_id for x in identifier.factors)
    _advance_ticks(current, clock, 120, hz=120.)
    assert current.motion_finished and not current.estimation_finished
    assert current.frame.time == 1.
    assert len(current.actions) == 120
    assert current.inference_busy and current.control_busy
    assert current.assimilated == ['joint:00000000']
    assert current.latest_result is result
    assert tuple(x.factor_id for x in identifier.factors) == factor_ids
    assert current.period == pytest.approx(1/120.)
    assert current.rates == {'joint': 40., 'imu': 30., 'image': 12.}
    assert len(current.capture_indices) == 13
    # Rendering/database writes were not allowed to run during the 120 ticks.
    assert current.acquisition_completed == 3
    assert current.acquisition_submitted == 85
    assert _complete_acquisitions(current) == 82
    events = current.store.events()
    assert {kind: sum(e.kind == kind for e in events) for kind in current.rates} == {
        'joint': 41, 'imu': 31, 'image': 13}
    assert [(e.stamp, e.sequence) for e in events] == sorted((e.stamp, e.sequence) for e in events)
    assert current.acquisition_completed == current.acquisition_submitted == 85
    assert current.manifest()['all_captured_sensor_data_archived']
    assert current.manifest()['target_physics_hz'] == 120.
    previous_velocity = np.zeros(6)
    for step, row in zip(current.actions, _jsonlines(current.output/'commands.jsonl')):
        dt = step.end_stamp-step.start_stamp
        velocity = np.asarray(row['velocity'])
        assert np.max(np.abs(velocity)) <= current.servo.speed+1e-10
        assert np.max(np.abs(velocity-previous_velocity)) <= current.servo.acceleration*dt+1e-7
        previous_velocity = velocity
    for event in events:
        assert not {'camera_pose', 'marker_pose', 'equilibrium', 'stiffness'} & set(event.payload)
        if event.kind == 'joint':
            assert set(event.payload) == {'time', 'frame_index', 'command_velocity'}
        if event.kind == 'image':
            assert (current.output/event.payload['image_path']).is_file()


def test_physics_does_not_write_the_DB_and_next_selection_waits_for_completion_ACK(session, monkeypatch):
    current, clock, initial, identifier = session
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The physical coordinator must not perform blocking DB writes.')
    monkeypatch.setattr(current.store, 'claim_next', forbidden)
    monkeypatch.setattr(current.store, 'complete', forbidden)
    assert _select_next_estimate(current)
    result, _, _ = _complete_estimate(current, initial, identifier, acknowledge=False)
    assert current.database_ack_pending == 'imu:00000000'
    assert not current.inference_busy and current.latest_result is result
    assert current.store.get('imu:00000000').status == 'processing'
    for _ in range(3):
        current.submit_estimate()
    assert not current.database_selection_pending
    with current.acquisition_requests.mutex:
        operations = list(current.acquisition_requests.queue)
    assert len(operations) == 1 and operations[0]['operation'] == 'complete'
    archived = current.acquisition_completed
    _advance_ticks(current, clock, 6)
    assert len(current.actions) == 6
    assert current.database_ack_pending == 'imu:00000000'
    assert current.acquisition_completed == archived
    raw_count = _complete_acquisitions(current)
    assert raw_count == 3
    assert current.acquisition_completed == archived+raw_count
    assert current.database_ack_pending is None
    assert current.store.get('imu:00000000').status == 'completed'
    assert _select_next_estimate(current)
    selected, _, _ = current.estimate_requests.get_nowait()
    assert selected.kind == 'imu' and selected.stamp > result.prior.stamp
    assert current.acquisition_completed == archived+raw_count


@pytest.mark.parametrize('reply, message', [
    (dict(kind='database_selection', event=None), 'Unexpected database selection'),
    (dict(kind='database_completed', event_id='wrong-event'), 'another event'),
])
def test_unsolicited_DB_replies_do_not_count_as_archived_sensor_events(session, reply, message):
    current = session[0]
    before = current.acquisition_completed
    current.acquisition_replies.put(reply)
    with pytest.raises(ValueError, match=message):
        current.collect_archives()
    assert current.acquisition_completed == before


def _jsonlines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row]


def test_oldest_events_are_processed_without_latest_only_selection_or_omissions(tmp_path):
    current, clock, result, identifier = _new_session(tmp_path, duration=.2)
    try:
        _advance_ticks(current, clock, 24)
        _complete_acquisitions(current)
        result = _drain_estimates(current, result, identifier)
        events = current.store.events()
        selected = _jsonlines(current.output/'assimilation_selection.jsonl')
        assert current.estimation_finished and not current.closed
        assert all(e.status == 'completed' for e in events)
        expected = [e.event_id for e in events if e.kind == 'imu' or e.event_id == 'joint:00000000']
        assert current.assimilated == expected
        assert [x['event_id'] for x in selected] == expected
        assert all(x['archived_not_assimilated_capture_indices'] == [] for x in selected)
        assert all(x['posterior_reset'] is False for x in selected)
        assert [x['selected_stamp'] for x in selected] == sorted(x['selected_stamp'] for x in selected)
        for event in events:
            if event.kind == 'joint' and event.event_id != 'joint:00000000':
                assert event.result_metadata['observation_fused'] is False
                assert event.result_metadata['reason'] == 'command notification'
            if event.kind == 'image':
                assert event.result_metadata['observation_fused'] is False
                assert event.result_metadata['reason'] == 'sensor not enabled in this comparison case'
        assert result.records['sensor_event']['kind'] == 'imu'
    finally:
        current.close()


def test_same_frame_IMU_and_image_are_distinct_ordered_updates(tmp_path):
    current, _, result, identifier = _new_session(tmp_path, case='imu_marker', duration=.1)
    try:
        assert current.assimilated == ['joint:00000000']
        current.submit_estimate()
        imu, _, imu_index = _complete_estimate(current, result, identifier)
        current.submit_estimate()
        image, _, image_index = _complete_estimate(current, imu, identifier)
        assert imu_index == image_index == 0
        assert imu.prior.stamp == image.prior.stamp
        assert imu.prior.command == image.prior.command
        assert current.assimilated == ['joint:00000000', 'imu:00000000', 'image:00000000']
        assert imu.records['sensor_event']['kind'] == 'imu'
        assert image.records['sensor_event']['kind'] == 'image'
        assert imu.records['sensor_event']['sequence'] < image.records['sensor_event']['sequence']
        assert imu.records['observation_ids'] and image.records['observation_ids']
        assert set(imu.records['observation_ids']).isdisjoint(image.records['observation_ids'])
        assert len(identifier.factors) == 2
        assert imu.stages[-1].components[0].state.state_id != image.stages[-1].components[0].state.state_id
        assert not image.records['sensor_event']['simultaneous_sensor_wait']
    finally:
        current.close()


def test_completed_old_law_is_adopted_atomically_without_rolling_back_new_inference(session):
    current, clock, original, identifier = session
    assert current.latest_packet is None
    current.submit_control()
    old_plan = _complete_control(current)
    assert old_plan['packet'] is None
    current.submit_estimate()
    updated, raw, _ = _complete_estimate(current, original, identifier)
    assert raw is not None
    assert current.current_control is None
    # An inference reply cannot pre-adopt K under an in-flight rigid plan.
    assert current.engine.packet is None
    current.control_replies.put(old_plan); current.poll()
    assert current.current_control is old_plan
    assert current.latest_result is updated and current.latest_packet is raw
    assert current.engine.packet is None
    clock[0] = 100.05; current.advance()
    assert current.actions[-1].selected_K is None
    # The next request uses the latest result and its exact estimated K.
    direct = _complete_control(current)
    assert isinstance(direct['packet'], FrozenActionKPacket)
    assert direct['packet'].estimated_packet is raw
    current.control_replies.put(direct); current.poll()
    clock[0] = 100.10; current.advance()
    np.testing.assert_array_equal(current.actions[-1].selected_K, raw.proposal['selected_K'])
    assert current.actions[-1].provenance['action_packet_id'] == direct['packet'].packet_id
    assert current.latest_result is updated and current.latest_packet is raw
    assert current.current_control['snapshot'].stiffness_record['command_proposal'] == raw.proposal
    assert 'execution_stiffness' not in current.current_control['snapshot'].stiffness_record


def test_marker_only_archives_IMU_but_updates_once_for_the_distinct_image_event(tmp_path):
    current, _, initial, identifier = _new_session(tmp_path, case='marker', duration=.1)
    try:
        current.submit_estimate()
        result, packet, index = _complete_estimate(current, initial, identifier)
        assert index == 0
        assert current.assimilated == ['joint:00000000', 'image:00000000']
        assert result.records['sensor_event']['kind'] == 'image'
        assert packet is not None and len(identifier.factors) == 1
        skipped = current.store.get('imu:00000000')
        assert skipped.status == 'completed'
        assert np.asarray(skipped.payload['gravity_in_sensor']).shape == (3,)
        assert skipped.result_metadata['observation_fused'] is False
        assert skipped.result_metadata['reason'] == 'sensor not enabled in this comparison case'
        assert current.store.get('image:00000000').result_metadata['observation_fused']
    finally:
        current.close()


def test_duplicate_control_reply_cannot_change_the_active_bundle(session):
    current, _, _, _ = session
    current.submit_control(); plan = _complete_control(current)
    current.control_replies.put(plan); current.poll()
    expected = current.current_control
    current.control_replies.put(dict(plan, velocity=-np.ones(6)))
    with pytest.raises(ValueError, match='outstanding frozen-law request'):
        current.poll()
    assert current.current_control is expected
    np.testing.assert_array_equal(current.desired_velocity, plan['velocity'])


def test_mismatched_control_reply_does_not_complete_the_outstanding_request(session):
    current = session[0]
    current.submit_control(); plan = _complete_control(current)
    request_id = current.pending_control_request
    current.control_replies.put(dict(plan, request_id=request_id+1))
    with pytest.raises(ValueError, match='outstanding frozen-law request'):
        current.poll()
    assert current.current_control is None and current.control_busy
    assert current.pending_control_request == request_id
    current.control_replies.put(plan); current.poll()
    assert current.current_control is plan and not current.control_busy


def test_the_claimed_inference_event_is_not_replaceable_by_another_ID(session):
    current, _, result, _ = session
    assert _select_next_estimate(current)
    event, index, _ = current.estimate_requests.get_nowait()
    current.estimate_replies.put(dict(event_id='other-event', result=result, packet=None,
        sensor_kind=event.kind, capture_index=index, marker_observed=False, work_seconds=0.))
    with pytest.raises(ValueError, match='claimed acquisition event'):
        current.poll()
    assert current.store.get(event.event_id).status == 'processing'


def test_existing_half_second_source_clock_reference_holds_its_finite_endpoint(session):
    current, _, _, _ = session
    current.submit_control(); plan = _complete_control(current)
    current.control_replies.put(plan); current.poll()
    diagnostics = plan['diagnostics']
    assert diagnostics['duration'] == .5
    assert diagnostics['ik_update_count'] == 1 and diagnostics['future_ik_updates'] == 0
    source, target = np.array(diagnostics['source_command']), np.array(diagnostics['next_command'])
    _, middle = current._reaching_plan_velocity(.05, diagnostics['source_stamp']+.25)
    np.testing.assert_allclose(middle['reference_command'], source+.5*(target-source), atol=1e-14)
    requested, late = current._reaching_plan_velocity(.05, diagnostics['source_stamp']+1.)
    np.testing.assert_array_equal(late['reference_command'], target)
    np.testing.assert_array_equal(requested, (target-current.command)/.05)
    assert late['past_selected_duration']
    assert np.linalg.norm(requested) > 0.
    assert 'trajectory_commands' not in diagnostics and 'rollout_steps' not in diagnostics


def test_motion_end_drain_and_close_are_distinct_and_pending_work_remains_recoverable(tmp_path):
    current, clock, result, identifier = _new_session(tmp_path, duration=.1)
    _advance_ticks(current, clock, 13)
    _complete_acquisitions(current)
    assert current.motion_finished and not current.estimation_finished and not current.closed
    assert current.frame.time == pytest.approx(current.scene.duration, abs=1e-6)
    assert current.manifest()['motion_terminal_tolerance_s'] == 1e-6
    stamp, action_count = current.frame.stamp, len(current.actions)
    _drain_estimates(current, result, identifier)
    assert current.estimation_finished and not current.closed
    clock[0] = 101.; current.advance()
    assert current.frame.stamp == stamp and len(current.actions) == action_count
    path = current.output/'observations.sqlite3'
    count = current.store.stats()['total']
    current.close(); current.close()
    with SensorEventStore(path) as reopened:
        assert reopened.stats()['total'] == count
        assert reopened.stats()['earliest_unfinished'] is None

    other, _, _, _ = _new_session(tmp_path/'pending', duration=.1)
    path = other.output/'observations.sqlite3'
    assert other.store.stats()['earliest_unfinished'] is not None
    other.close()
    with SensorEventStore(path) as reopened:
        assert reopened.stats()['earliest_unfinished'] is not None
        assert not all(e.status == 'completed' for e in reopened.events())


def test_missed_deadlines_are_reported_without_interpolating_fabricated_observations(session):
    current, clock, _, _ = session
    clock[0] = 101.; current.advance()
    _complete_acquisitions(current)
    assert len(current.actions) == 1
    assert current.missed_acquisition_deadlines == {'joint': 39, 'imu': 29, 'image': 11}
    events = current.store.events()
    assert len(events) == 6
    assert sorted({e.stamp for e in events}) == [1792080000., 1792080001.]
    assert not current.manifest()['SG_interpolation_used']


def test_display_and_control_share_the_same_caught_up_law_after_a_delay(session):
    current, clock, result, _ = session
    _advance_ticks(current, clock, 4)
    inference = current.inference_description
    predicted = predict_result(result, tuple(current.actions), target_stamp=current.frame.stamp,
        frame_index=4, body_ids=inference.body_ids, actions=inference.actions)
    snapshot = build_snapshot(predicted, predicted.stages[-1], inference.common)
    display = inference.common['display']
    message = make_markers(snapshot, display['seed'], display['point_size'])
    buffer = io.BytesIO(); message.serialize(buffer)
    assert verify_wire_identity(snapshot, MarkerArray().deserialize(buffer.getvalue()),
        seed=display['seed'], point_size=display['point_size'])
    assert snapshot.command == tuple(current.command) and snapshot.stamp == current.frame.stamp
    assert snapshot.observation_ids == ()


def test_process_serialization_restores_readonly_law_arrays_without_changing_values(session):
    result = session[2]
    restored = _freeze_arrays(pickle.loads(pickle.dumps(result)))
    source, target = result.stages[-1].components[0].state, restored.stages[-1].components[0].state
    np.testing.assert_array_equal(source.covariance, target.covariance)
    assert source.state_id == target.state_id
    assert not target.covariance.flags.writeable and not target.transforms[0].bingham.flags.writeable


def test_acquisition_failure_is_explicit_and_reserved_image_is_not_skipped(session, monkeypatch):
    current = session[0]
    event = current.store.reserve('image:failed', 'image', current.frame.stamp+.01,
        current.command)
    current.acquisition_replies.put(dict(failed=True, traceback='test image encoder failure'))
    with pytest.raises(RuntimeError, match='image encoder failure'):
        current.collect_archives()
    assert current.failure['failed']
    assert current.store.get(event.event_id).status == 'awaiting_payload'


def test_update_rates_exclude_waiting_for_the_user_to_start_motion(session):
    current = session[0]
    current.control_times = [1., 100., 100.5, 101.]
    current.estimate_times = [2., 100.25, 100.75, 101.25]
    current.started = 100.
    assert current.manifest()['actual_control_hz'] == 2.
    assert current.manifest()['actual_estimate_hz'] == 2.
