"""Only a physically maintained success becomes a confirmed command hold."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.simple_motion import SimpleMotionSession
from test_async_execution import Context
from test_simple_motion import plan_reply


CONFIG = Path(__file__).resolve().parents[1]/'config'
EPOCH = 1792080000.


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def observed_body(identity):
    return SimpleNamespace(stages=(SimpleNamespace(components=(SimpleNamespace(
        state=SimpleNamespace(state_id=identity)),)),),
        records={'simple_inference': {'goal_observed': True}})


def attach_reach_notification(session, monkeypatch, *, reached=lambda time: time == .25):
    """Inject only the existing success notification, never a truth estimate."""
    original = session.scene.step

    def step(time, *args, **kwargs):
        frame = original(time, *args, **kwargs)
        value = bool(reached(time))
        session.scene.trajectory[-1]['reached'] = value
        return replace(frame, reached=value)

    monkeypatch.setattr(session.scene, 'step', step)


def active_plan(session):
    start = session.command.copy()
    target = start.copy()
    target[0] += .02
    session.plan = dict(start=start, target=target, start_time=0., end_time=1.,
        duration=1., source={}, goal_observed=True, request_id=1)
    return start, target


def tick(session, clock, elapsed):
    clock[0] = 100.+elapsed
    session.advance()


def complete_display(session, request_id=1):
    display = dict(kind='display', request_id=request_id, capture_index=0, index=0,
        event_id='joint:test', snapshot=None, diagnostics={'original_observation_stamp': EPOCH})
    session.control_replies.put(display)
    session.poll()
    return display


def test_single_ON_holds_sent_command_for_settling_then_OFF_resumes_selection(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'single_on',
        duration=5., context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch)
        start, target = active_plan(session)
        assert session.reach_trial is session.reach_stop is None
        assert np.linalg.norm(session.velocity) > 0.
        session.started = clock[0]
        tick(session, clock, .25)
        stopped = start+.25*(target-start)
        np.testing.assert_allclose(session.command, stopped, atol=1e-14)
        assert session.frame.reached
        assert session.reach_stop is None
        assert session.reach_trial['time'] == pytest.approx(.25)
        assert session.reach_trial['stamp'] == pytest.approx(EPOCH+.25)
        np.testing.assert_array_equal(session.reach_trial['command'], session.command)
        np.testing.assert_array_equal(session.velocity, np.zeros(6))
        reached_row = rows(session.output/'physical_trajectory.jsonl')[-1]
        assert reached_row['command_interval_mode'] == 'reaching'
        assert reached_row['action_mode'] == 'reached_settling'
        acquisitions_at_reach = session.acquisition_submitted
        assert session.manifest()['command_selection_paused_for_reach']
        assert not session.manifest()['command_selection_stopped_on_reach']

        for elapsed in (.5, 1., 2.249):
            tick(session, clock, elapsed)
            assert not session.frame.reached
            assert session.reach_trial is not None
            assert session.reach_stop is None
            np.testing.assert_array_equal(session.command, stopped)
            np.testing.assert_array_equal(session.velocity, np.zeros(6))
            np.testing.assert_array_equal(session.actions[-1].joint_increment, np.zeros(6))
            assert session.scene.trajectory[-1]['command_interval_mode'] == 'reached_settling'
            assert session.scene.trajectory[-1]['action_mode'] == 'reached_settling'
        assert session.acquisition_submitted > acquisitions_at_reach
        assert not session.motion_finished

        session.latest_body, session.latest_body_index = observed_body('after_trial'), 0
        tick(session, clock, 2.25)
        assert session.reach_trial is session.reach_stop is None
        assert not session.frame.reached
        request = session.control_requests.get_nowait()
        assert request['select'] is True
        np.testing.assert_array_equal(session.command, stopped)
        assert [row['kind'] for row in rows(session.output/'reach_hold_events.jsonl')] == [
            'trial_started', 'trial_released']
        manifest = session.manifest()
        assert not manifest['command_selection_stopped_on_reach']
        assert not manifest['command_selection_paused_for_reach']
        assert not manifest['physical_truth_used_for_inference']
    finally:
        session.close()


def test_continuous_ON_confirms_after_two_seconds_and_OFF_reopens_control(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'confirmed',
        duration=5., context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch, reached=lambda time: .25 <= time <= 2.25)
        active_plan(session)
        session.started = clock[0]
        tick(session, clock, .25)
        stopped = session.command.copy()
        for elapsed in (.75, 1.5, 2.249):
            tick(session, clock, elapsed)
            assert session.reach_stop is None
            assert session.reach_trial is not None
        tick(session, clock, 2.25)
        assert session.reach_trial is None
        assert session.reach_stop['confirmed_time'] == pytest.approx(2.25)
        assert session.reach_stop['confirmed_stamp'] == pytest.approx(EPOCH+2.25)
        assert session.reach_stop['time'] == pytest.approx(.25)
        np.testing.assert_array_equal(session.reach_stop['command'], stopped)
        np.testing.assert_array_equal(session.command, stopped)
        assert session.frame.reached
        assert session.scene.trajectory[-1]['command_interval_mode'] == 'reached_settling'
        assert session.scene.trajectory[-1]['action_mode'] == 'reached_hold'
        assert session.manifest()['command_selection_stopped_on_reach']

        session.latest_body, session.latest_body_index = observed_body('after_confirmed'), 0
        tick(session, clock, 2.5)
        assert not session.frame.reached, 'The lamp remains the physical result, never a latched display.'
        assert session.reach_trial is session.reach_stop is None
        assert session.scene.trajectory[-1]['command_interval_mode'] == 'reached_hold'
        assert session.control_requests.get_nowait()['select'] is True
        assert [row['kind'] for row in rows(session.output/'reach_hold_events.jsonl')] == [
            'trial_started', 'confirmed', 'hold_released']
    finally:
        session.close()


def test_ON_duration_is_continuous_not_accumulated_across_OFF(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'interrupted_on',
        duration=6., context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch,
            reached=lambda time: time == .25 or time >= 1.5)
        active_plan(session)
        session.started = clock[0]
        tick(session, clock, .25)
        tick(session, clock, .75)
        assert session.reach_trial['on_since_time'] is None
        tick(session, clock, 1.5)
        assert session.reach_trial['on_since_time'] == pytest.approx(1.5)
        tick(session, clock, 2.25)
        assert session.reach_trial is not None and session.reach_stop is None
        tick(session, clock, 3.499)
        assert session.reach_stop is None
        tick(session, clock, 3.5)
        assert session.reach_trial is None
        assert session.reach_stop['confirmed_time'] == pytest.approx(3.5)
    finally:
        session.close()


@pytest.mark.parametrize('arrival_phase', ['same_tick', 'confirmed', 'released'])
def test_retired_plan_cannot_change_command_or_K_even_after_trial_release(
        tmp_path, monkeypatch, arrival_phase):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/arrival_phase,
        duration=5., context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch,
            reached=(lambda time: .25 <= time <= 2.25) if arrival_phase == 'confirmed'
            else (lambda time: time == .25))
        start, _ = active_plan(session)
        session.started = clock[0]
        session.control_counter = 7
        pending_target = start.copy()
        pending_target[1] += .2
        message = plan_reply(pending_target, source_command=start,
            packet=object(), selection_performed=True)
        message['request_id'] = 7
        session.control_busy = True
        if arrival_phase == 'same_tick':
            session.control_replies.put(message)
        tick(session, clock, .25)
        stopped = session.command.copy()
        assert session.retired_control_through >= 7
        if arrival_phase != 'same_tick':
            tick(session, clock, 2.25)
            assert (session.reach_stop is not None) == (arrival_phase == 'confirmed')
            session.control_replies.put(message)
            session.poll()
        assert session.engine.packet is None
        assert session.plan is None
        np.testing.assert_array_equal(session.command, stopped)
        assert not (session.output/'simple_plans.jsonl').exists()
        dispositions = rows(session.output/'simple_plan_dispositions.jsonl')
        assert len(dispositions) == 1
        assert dispositions[0]['request_id'] == message['request_id']
        assert not dispositions[0]['adopted']
        assert dispositions[0]['reason']
        display = complete_display(session, message['request_id'])
        assert session.current_display is display
        assert not session.control_busy

        if arrival_phase == 'released':
            # A new selection belongs to the resumed execution and remains
            # admissible. It must start at the command actually being held.
            new_target = stopped.copy()
            new_target[0] -= .01
            new = plan_reply(new_target, source_command=stopped, packet=None)
            new['request_id'] = 8
            session.control_replies.put(new)
            session.poll()
            assert session.plan['request_id'] == 8
            tick(session, clock, 2.5)
            np.testing.assert_allclose(session.command, stopped+.5*(new_target-stopped), atol=1e-14)
    finally:
        session.close()


def test_settling_and_confirmed_hold_request_display_only_once_per_body_version(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'display_only', duration=5.,
        context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch, reached=lambda time: time >= .25)
        active_plan(session)
        session.started = clock[0]
        first_body = observed_body('joint_body:first')
        session.latest_body, session.latest_body_index = first_body, 0
        tick(session, clock, .25)
        stopped = session.command.copy()
        request = session.control_requests.get_nowait()
        assert request['select'] is False
        assert request['result'] is first_body
        assert request['stamp'] == session.frame.stamp
        assert request['steps'] == tuple(session.actions)
        assert not any(key in request for key in ('tip', 'joints', 'physical_pose', 'true_K'))
        complete_display(session, request['request_id'])
        tick(session, clock, .5)
        assert session.control_requests.empty()
        tick(session, clock, 2.25)
        assert session.reach_stop is not None
        assert session.control_requests.empty(), 'Confirmation alone is not a new body version.'

        second_body = observed_body('joint_body:second')
        session.replies.put(dict(kind='body', result=second_body, packet=None, capture_index=1,
            goal_observed=True))
        tick(session, clock, 2.5)
        refreshed = session.control_requests.get_nowait()
        assert refreshed['result'] is second_body
        assert refreshed['select'] is False
        assert refreshed['capture_index'] == 1
        assert refreshed['steps'] == tuple(session.actions[1:])
        assert session.latest_body is second_body
        np.testing.assert_array_equal(session.command, stopped)
        np.testing.assert_array_equal(session.velocity, np.zeros(6))
    finally:
        session.close()


def test_initial_ON_begins_trial_without_pretending_it_is_confirmed(tmp_path, monkeypatch):
    original = MovingSwitchScene.step

    def step(scene, *args, **kwargs):
        frame = original(scene, *args, **kwargs)
        scene.trajectory[-1]['reached'] = True
        return replace(frame, reached=True)

    monkeypatch.setattr(MovingSwitchScene, 'step', step)
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'initial_reach', duration=5.,
        context=Context(), body_surprise_enabled=False)
    try:
        assert session.frame.reached
        assert session.reach_stop is None
        assert session.reach_trial['time'] == 0.
        assert session.reach_trial['stamp'] == EPOCH
        np.testing.assert_array_equal(session.reach_trial['command'], session.command)
        assert session.scene.trajectory[-1]['action_mode'] == 'reached_settling'
        assert session.scene.trajectory[-1]['command_interval_mode'] == 'waiting_for_goal'
        assert session.acquisition_submitted > 0
        assert not session.motion_finished
        session.latest_body = observed_body('initial_reached_law')
        session.latest_body_index = 0
        session._submit_control()
        assert session.control_requests.get_nowait()['select'] is False
        manifest = session.manifest()
        assert manifest['command_selection_paused_for_reach']
        assert not manifest['command_selection_stopped_on_reach']
        assert manifest['reach_trial']['time'] == 0.
    finally:
        session.close()
