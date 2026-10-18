"""Finite endpoint execution preserves original clocks and actual K intervals."""

import json
from pathlib import Path
import time

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.display import verify_wire_identity
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.simple_inference import SimpleInference
from probtf_observation_fusion_demo.simple_motion import SimpleMotionSession
from test_async_execution import Context
from test_simple_inference import acquired


CONFIG = Path(__file__).resolve().parents[1]/'config'
EPOCH = 1792080000.


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


class DeclaredActionEngine:
    """Two declared test action maps; no physical pose or true K is queried."""
    def __init__(self):
        self.packet = 'K0'

    def set(self, packet):
        self.packet = packet

    def increment(self, before, after):
        scale = .8 if self.packet == 'K0' else 1.1
        return dict(joint_increment=(scale*(after-before)).tolist(), density_id=self.packet,
            selected_K=[120. if self.packet == 'K0' else 140.]*len(before),
            captured_body_posterior_versions=[3], captured_body_posterior_state_ids=['observed_joint_law'],
            captured_source_stamp=EPOCH, physical_truth_received=False)


def plan_reply(target, *, source_command, packet='K1', duration=.5, selection_performed=True):
    return dict(kind='plan', request_id=2, last_source_index=0, packet=packet,
        processed=[], duration=duration, q_next=np.array(target), work_seconds=.2,
        selection_performed=selection_performed, source_command=list(source_command),
        diagnostics=dict(source_stamp=EPOCH, source_command=list(source_command),
            source_version=3, body_source_state_ids=['observed_joint_law'],
            goal_source_state_ids=['observed_joint_law'], goal_observed=True))


def test_old_finite_reference_reaches_and_holds_endpoint_before_selected_packet_adoption(tmp_path):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'marker.yaml', tmp_path/'timing',
        context=Context(), clock=lambda: clock[0], duration=3., body_surprise_enabled=False)
    try:
        session.engine = DeclaredActionEngine()
        initial = session.command.copy()
        old_target = initial.copy()
        old_target[0] += .02
        new_target = initial.copy()
        new_target[0] -= .01
        session.started = clock[0]
        session.plan = dict(start=initial.copy(), target=old_target, start_time=0.,
            end_time=1., duration=1., source={}, goal_observed=True, request_id=1)
        clock[0] = 100.25
        session.advance()
        expected_boundary = initial+.25*(old_target-initial)
        np.testing.assert_allclose(session.command, expected_boundary, atol=1e-14)
        first = session.actions[0]
        assert first.start_stamp == EPOCH and first.end_stamp == EPOCH+.25
        assert first.density_id == 'K0'
        np.testing.assert_allclose(first.joint_increment, .8*(expected_boundary-initial), atol=1e-14)
        frozen_record = first.record()

        clock[0] = 101.
        session.advance()
        np.testing.assert_allclose(session.command, old_target, atol=1e-14)
        assert session.engine.packet == 'K0'
        # Selection is evaluated at this held endpoint.  Until its reply is
        # adopted, the old K also owns the intervening zero-command interval.
        session.control_busy = True
        session.control_replies.put(plan_reply(new_target, source_command=old_target))
        clock[0] = 101.1
        session.advance()
        np.testing.assert_array_equal(session.command, old_target)
        assert session.actions[2].density_id == 'K0'
        np.testing.assert_array_equal(session.actions[2].joint_increment, np.zeros(6))
        assert session.engine.packet == 'K1'
        assert session.control_busy
        assert session.current_display is None
        assert session.plan['start_time'] == pytest.approx(1.1)
        np.testing.assert_array_equal(session.plan['start'], old_target)

        clock[0] = 101.35
        session.advance()
        expected_halfway = old_target+.5*(new_target-old_target)
        np.testing.assert_allclose(session.command, expected_halfway, atol=1e-14)
        assert session.actions[3].density_id == 'K1'
        np.testing.assert_allclose(session.actions[3].joint_increment,
            1.1*(expected_halfway-old_target), atol=1e-14)
        assert first.record() == frozen_record
        # Display completion acknowledges the same request after execution has
        # already begun; rendering does not delay finite-plan adoption.
        display = dict(kind='display', request_id=2, diagnostics={})
        session.control_replies.put(display)
        session.poll()
        assert session.current_display is display
        assert not session.control_busy
        np.testing.assert_array_equal(session.command, expected_halfway)

        clock[0] = 102.
        session.advance()
        np.testing.assert_allclose(session.command, new_target, atol=1e-14)
        clock[0] = 102.6
        session.advance()
        np.testing.assert_array_equal(session.command, new_target)
        np.testing.assert_array_equal(session.actions[-1].joint_increment, np.zeros(6))
        np.testing.assert_array_equal(session.velocity, np.zeros(6))
        assert 'velocity' not in vars(session)
        assert first.record() == frozen_record
        saved = json_rows(session.output/'commands.jsonl')
        assert saved[0]['density_id'] == 'K0'
        assert saved[2]['density_id'] == 'K0'
        assert saved[3]['density_id'] == 'K1'
        for previous, current in zip(saved, saved[1:]):
            assert previous['end_stamp'] == current['start_stamp']
            assert previous['end_command'] == current['start_command']
    finally:
        session.close()


def test_stale_selected_command_is_rejected_before_replacing_active_K(tmp_path):
    session = SimpleMotionSession(CONFIG/'marker.yaml', tmp_path/'stale', context=Context(),
        duration=1., body_surprise_enabled=False)
    try:
        session.engine = DeclaredActionEngine()
        stale = session.command.copy()
        stale[0] += .01
        session.control_replies.put(plan_reply(stale, source_command=stale))
        with pytest.raises(ValueError, match='held current command'):
            session.poll()
        assert session.engine.packet == 'K0'
        assert session.plan is None
    finally:
        session.close()


def test_control_uses_completed_body_before_its_batch_finishes_and_does_not_replace_moving_K(tmp_path, acquired):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'marker.yaml', tmp_path/'moving_updates',
        context=Context(), clock=lambda: clock[0], duration=2., body_surprise_enabled=False)
    try:
        session.engine = DeclaredActionEngine()
        initial = session.command.copy()
        target = initial.copy()
        target[0] += .02
        session.started = clock[0]
        session.plan = dict(start=initial.copy(), target=target, start_time=0., end_time=1.,
            duration=1., source={}, goal_observed=True, request_id=1)
        with SensorEventStore(session.output/'completed_body_event.sqlite3') as store:
            event = store.append('image:completed_inside_batch', 'image', EPOCH, initial,
                dict(time=0., frame_index=0, image=acquired[1].image, image_stamp=EPOCH))
        body = session.inference_description.update_sensor_event(event, event_index=0)
        assert body.records['simple_inference']['goal_observed']
        session.inference_busy = True
        session.replies.put(dict(kind='body', result=body, packet=None, capture_index=0,
            goal_observed=True))
        clock[0] = 100.25
        session.advance()
        active = session.plan
        assert session.latest_body is body
        assert session.latest_body_index == 0
        assert session.inference_busy
        assert session.last_assimilated_capture is None
        assert session.processed_events == 0
        assert session.control_requests.empty()
        assert session.plan is active
        assert session.engine.packet == 'K0'
        assert not (session.output/'simple_plans.jsonl').exists()
        clock[0] = 101.
        session.advance()
        # The body owner is deliberately still busy with the SAME batch.
        # Its completed law is nevertheless enough for the independent
        # selector to transport every actual old-K interval to this endpoint.
        assert session.inference_busy and session.last_assimilated_capture is None
        request = session.control_requests.get_nowait()
        assert request['select'] is True
        assert request['result'] is body
        assert request['capture_index'] == 0
        assert request['index'] == len(session.frames)-1
        assert request['stamp'] == session.frame.stamp
        assert request['steps'] == tuple(session.actions)
        assert all(step.density_id == 'K0' for step in request['steps'])
        assert session.control_busy
        np.testing.assert_array_equal(session.command, target)
        clock[0] = 101.3
        session.advance()
        np.testing.assert_array_equal(session.command, target)
        assert session.control_requests.empty()
    finally:
        session.close()


@pytest.mark.parametrize('override', [None, False, True])
def test_body_surprise_override_reaches_actual_recurrence_and_selection_constructors(tmp_path, monkeypatch, override):
    constructed = []
    original = SimpleInference.__init__

    def construct(self, *args, **kwargs):
        original(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(SimpleInference, '__init__', construct)
    session = SimpleMotionSession(CONFIG/'marker.yaml', tmp_path/'worker_options',
        duration=.2, context=Context(), body_surprise_enabled=override)
    try:
        # Exercise each real target up to its request loop. A worker whose
        # constructor loses the override is observable even though this fake
        # multiprocessing context does not start background computation.
        for worker in session.workers[1:]:
            worker.args[0].put(None)
            worker.target(*worker.args)
        assert len(constructed) == 3
        expected = True if override is None else override
        assert all(inference.surprise_enabled is expected for inference in constructed)
        assert all(inference.settings['body_bayes_factor_surprise']['enabled'] is expected
            for inference in constructed)
        if expected:
            assert all(inference.change_probability == .25 for inference in constructed)
        assert session.replies.empty() and session.control_replies.empty()
    finally:
        session.close()


@pytest.mark.parametrize('case', ['marker', 'imu_marker'])
@pytest.mark.parametrize('body_surprise_enabled', [False, True])
def test_short_real_spawned_loop_archives_and_processes_original_events_in_order(
        tmp_path, case, body_surprise_enabled):
    session = SimpleMotionSession(CONFIG/(case+'.yaml'), tmp_path/case, duration=.2,
        body_surprise_enabled=body_surprise_enabled)
    deadline = time.monotonic()+20.
    try:
        while not session.estimation_finished and time.monotonic() < deadline:
            session.advance()
            if session.started is None:
                session.start()
            time.sleep(.005)
        assert session.motion_finished, session.manifest()
        assert session.estimation_finished, session.manifest()
        assert session.frame.time == pytest.approx(.2)
        assert session.acquisition_completed == session.acquisition_submitted
        assert 'velocity' not in vars(session)
        manifest = session.manifest()
        assert manifest['all_captured_sensor_data_archived']
        assert not manifest['physical_truth_used_for_inference']
        assert not manifest['cumulative_K_factor_reevaluation']
        assert manifest['body_bfs_enabled'] is body_surprise_enabled
        assert not manifest['K_bfs_enabled']

        updates = json_rows(session.output/'simple_updates.jsonl')
        with SensorEventStore(session.output/'observations.sqlite3') as store:
            events = store.events()
            assert store.stats()['earliest_unfinished'] is None
        assert len(updates) == len(events) == session.processed_events
        assert len({row['event_id'] for row in updates}) == len(updates)
        original = {event.event_id: event for event in events}
        order = [(row['acquisition_stamp'], row['sequence']) for row in updates]
        assert order == sorted(order)
        for row in updates:
            event = original[row['event_id']]
            assert row['acquisition_stamp'] == event.stamp
            assert row['acquisition_command'] == list(event.command)
            assert row['sequence'] == event.sequence
            assert row['frame_index'] == event.payload['frame_index']
            assert row['simple_inference']['state_id'] == row['body_state_ids'][0]
            assert row['simple_inference']['body_BFS_enabled'] is body_surprise_enabled
            if row['stiffness'] is None:
                assert not row['observation_ids']
                assert row['K_update_skipped_no_observation']
            else:
                assert not row['stiffness']['physical_truth_received']
                assert row['stiffness']['matching_target'] == 'joint_rotation'
                assert row['stiffness']['body_posterior_state_ids'] == row['body_state_ids']
                assert not row['stiffness'].get('bayes_factor_surprise_enabled', False)
            if row['observation_ids']:
                assert row['observation_selected']
            if row['observation_eligible'] and not row['observation_selected']:
                assert not row['observation_ids']
                assert row['stiffness'] is None
        observed = [row for row in updates if row['observation_ids']]
        assert observed
        assert all(row['body_surprise']['enabled'] is body_surprise_enabled for row in observed)
        assert any(row['body_surprise']['applied'] for row in observed) is body_surprise_enabled
        kinds = {row['kind'] for row in observed}
        assert 'image' in kinds
        if case == 'imu_marker':
            assert 'imu' in kinds
        initial = updates[0]
        assert initial['kind'] == 'joint'
        assert not initial['observation_ids']
        assert not initial['simple_inference']['goal_observed']
        assert any(row['simple_inference']['goal_observed'] for row in updates)
        frozen = session.latest_packet.record
        assert any(row['body_state_ids'] == frozen['body_posterior_state_ids']
            and row['acquisition_stamp'] == frozen['source_stamp']
            and row['acquisition_command'] == frozen['source_command'] for row in updates)
        display = session.current_display
        snapshot = display['snapshot']
        ids = display['diagnostics']['body_source_state_ids']
        assert ids == display['diagnostics']['goal_source_state_ids']
        assert ids == snapshot.candidate_measure['shape_source_state_ids']
        assert {point.query.state_id for point in snapshot.points} == set(ids)
        assert snapshot.surprise_record['enabled'] is body_surprise_enabled
        actual_markers = MarkerArray().deserialize(display['markers_wire'])
        settings = session.inference_description.common['display']
        verify_wire_identity(snapshot, actual_markers, seed=settings['seed'],
            point_size=settings['point_size'])
        for row in json_rows(session.output/'simple_plans.jsonl'):
            diagnostics = row['diagnostics']
            assert row['start_command'] == diagnostics['source_command']
            assert diagnostics['body_source_state_ids'] == diagnostics['goal_source_state_ids']
            assert not diagnostics['body_reconstructed_from_stiffness']
            if diagnostics['goal_observed']:
                assert diagnostics['S_after'] <= diagnostics['S_before']
        assert all(step.start_command == previous.end_command
            for previous, step in zip(session.actions, session.actions[1:]))
        # Finite source time, source command and interval K are saved together.
        assert len(json_rows(session.output/'commands.jsonl')) == len(session.actions)
    finally:
        session.close()
