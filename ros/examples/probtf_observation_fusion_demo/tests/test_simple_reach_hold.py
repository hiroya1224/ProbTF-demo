"""The first physical success ends command selection, not sensing or physics."""

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


def attach_reach_notification(session, monkeypatch, *, first_time=.25):
    """Inject only the existing success notification, never a truth estimate."""
    original = session.scene.step

    def step(time, *args, **kwargs):
        frame = original(time, *args, **kwargs)
        reached = time == first_time
        session.scene.trajectory[-1]['reached'] = reached
        return replace(frame, reached=reached)

    monkeypatch.setattr(session.scene, 'step', step)


def active_plan(session):
    start = session.command.copy()
    target = start.copy()
    target[0] += .02
    session.plan = dict(start=start, target=target, start_time=0., end_time=1.,
        duration=1., source={}, goal_observed=True, request_id=1)
    return start, target


def test_first_reach_latches_sent_command_while_physics_and_acquisition_continue(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'first_reach',
        duration=2., context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch)
        start, target = active_plan(session)
        assert session.reach_stop is None
        assert not session.manifest()['command_selection_stopped_on_reach']
        assert np.linalg.norm(session.velocity) > 0.
        session.started = clock[0]
        clock[0] = 100.25
        session.advance()
        stopped = start+.25*(target-start)
        np.testing.assert_allclose(session.command, stopped, atol=1e-14)
        assert session.frame.reached
        assert session.reach_stop['time'] == pytest.approx(.25)
        assert session.reach_stop['stamp'] == pytest.approx(EPOCH+.25)
        np.testing.assert_array_equal(session.reach_stop['command'], session.command)
        saved_stop = json.loads(json.dumps(session.reach_stop))
        reached_row = rows(session.output/'physical_trajectory.jsonl')[-1]
        assert reached_row['command_interval_mode'] == 'reaching'
        assert reached_row['action_mode'] == 'reached_hold'
        np.testing.assert_array_equal(session.velocity, np.zeros(6))

        # The ongoing finite plan would otherwise advance to its old endpoint.
        # Physical state can leave the tolerance again; the sent position must
        # stay at the very first reached command rather than re-enable control.
        acquisitions_at_reach = session.acquisition_submitted
        for elapsed in (.5, 1., 1.5):
            clock[0] = 100.+elapsed
            session.advance()
            assert session.frame.time == pytest.approx(elapsed)
            assert not session.frame.reached
            np.testing.assert_array_equal(session.command, session.reach_stop['command'])
            np.testing.assert_array_equal(session.velocity, np.zeros(6))
            np.testing.assert_array_equal(session.actions[-1].joint_increment, np.zeros(6))
            assert session.scene.trajectory[-1]['command_interval_mode'] == 'reached_hold'
            assert session.scene.trajectory[-1]['action_mode'] == 'reached_hold'
        assert session.acquisition_submitted > acquisitions_at_reach
        assert len(session.frames) == 5
        assert session.reach_stop == saved_stop
        assert len(rows(session.output/'reach_stop.jsonl')) == 1
        assert not session.motion_finished
        manifest = session.manifest()
        assert manifest['command_selection_stopped_on_reach']
        assert manifest['reach_stop'] == saved_stop
        assert not manifest['physical_truth_used_for_inference']
    finally:
        session.close()


@pytest.mark.parametrize('reply_before_advance', [False, True])
def test_plan_pending_at_reach_or_arriving_later_cannot_change_position_or_K(
        tmp_path, monkeypatch, reply_before_advance):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'pending', duration=2.,
        context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch)
        start, target = active_plan(session)
        session.started = clock[0]
        pending_target = start.copy()
        pending_target[1] += .2
        # Its source deliberately differs from the now-reached command. A
        # retired result is classified first, not raised as a new stale plan.
        message = plan_reply(pending_target, source_command=start,
            packet=object(), selection_performed=True)
        session.control_busy = True
        if reply_before_advance:
            session.control_replies.put(message)
        clock[0] = 100.25
        session.advance()
        if not reply_before_advance:
            session.control_replies.put(message)
            session.poll()
        stopped = session.command.copy()
        assert session.engine.packet is None
        clock[0] = 100.75
        session.advance()
        np.testing.assert_array_equal(session.command, stopped)
        assert not (session.output/'simple_plans.jsonl').exists()
        dispositions = rows(session.output/'simple_plan_dispositions.jsonl')
        assert len(dispositions) == 1
        assert dispositions[0]['request_id'] == message['request_id']
        assert 'reach' in dispositions[0]['reason']
        assert 'hold' in dispositions[0]['reason']

        # The same request's completed distribution may still refresh RViz.
        display = dict(kind='display', request_id=message['request_id'],
            snapshot=None, diagnostics={'original_observation_stamp': EPOCH})
        session.control_replies.put(display)
        session.poll()
        assert session.current_display is display
        assert not session.control_busy
        np.testing.assert_array_equal(session.command, stopped)
    finally:
        session.close()


def test_reached_session_requests_only_new_body_versions_for_display(tmp_path, monkeypatch):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'display_only', duration=2.,
        context=Context(), clock=lambda: clock[0], body_surprise_enabled=False)
    try:
        attach_reach_notification(session, monkeypatch)
        active_plan(session)
        session.started = clock[0]
        first_body = observed_body('joint_body:first')
        session.latest_body, session.latest_body_index = first_body, 0
        clock[0] = 100.25
        session.advance()
        request = session.control_requests.get_nowait()
        assert request['select'] is False
        assert request['result'] is first_body
        assert request['stamp'] == session.frame.stamp
        assert request['steps'] == tuple(session.actions)
        assert not any(key in request for key in ('tip', 'joints', 'physical_pose', 'true_K'))

        session.control_replies.put(dict(kind='display', request_id=request['request_id'],
            snapshot=None, diagnostics={'original_observation_stamp': EPOCH}))
        session.poll()
        clock[0] = 100.5
        session.advance()
        assert session.control_requests.empty(), 'Unchanged body must not trigger repeated display work.'

        second_body = observed_body('joint_body:second')
        session.replies.put(dict(kind='body', result=second_body, packet=None, capture_index=1,
            goal_observed=True))
        clock[0] = 100.75
        session.advance()
        refreshed = session.control_requests.get_nowait()
        assert refreshed['result'] is second_body
        assert refreshed['select'] is False
        assert refreshed['capture_index'] == 1
        assert refreshed['steps'] == tuple(session.actions[1:])
        assert session.latest_body is second_body
        np.testing.assert_array_equal(session.command, session.reach_stop['command'])
    finally:
        session.close()


def test_already_reached_initial_frame_is_latched_before_any_plan(tmp_path, monkeypatch):
    original = MovingSwitchScene.step

    def step(scene, *args, **kwargs):
        frame = original(scene, *args, **kwargs)
        scene.trajectory[-1]['reached'] = True
        return replace(frame, reached=True)

    monkeypatch.setattr(MovingSwitchScene, 'step', step)
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'initial_reach', duration=1.,
        context=Context(), body_surprise_enabled=False)
    try:
        assert session.frame.reached
        assert session.reach_stop['time'] == 0.
        assert session.reach_stop['stamp'] == EPOCH
        np.testing.assert_array_equal(session.reach_stop['command'], session.command)
        assert session.scene.trajectory[-1]['action_mode'] == 'reached_hold'
        assert session.scene.trajectory[-1]['command_interval_mode'] == 'waiting_for_goal'
        assert session.acquisition_submitted > 0
        assert not session.motion_finished
        session.latest_body = observed_body('initial_reached_law')
        session.latest_body_index = 0
        session._submit_control()
        assert session.control_requests.get_nowait()['select'] is False
        manifest = session.manifest()
        assert manifest['command_selection_stopped_on_reach']
        assert manifest['reach_stop']['time'] == 0.
    finally:
        session.close()
