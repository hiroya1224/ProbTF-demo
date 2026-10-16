"""Motion advances while inference is busy; every actual action is retained."""
from dataclasses import replace
import io
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_motion import AsyncMotionSession, _freeze_arrays
from probtf_observation_fusion_demo.async_state import RecordedCommandPath, predict_result
from probtf_observation_fusion_demo.async_stiffness import (
    PhysicalActionEngine, capture_stiffness_packet, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_control import MovingController
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


PACKAGE = Path(__file__).resolve().parents[1]


class FakeProcess:
    exitcode = None
    def __init__(self, **kwargs):
        pass
    def start(self):
        pass
    def is_alive(self):
        return True
    def join(self, **kwargs):
        pass
    def terminate(self):
        pass


class LocalQueue(Queue):
    def cancel_join_thread(self):
        pass
    def close(self):
        pass


class Context:
    Queue = LocalQueue
    Process = FakeProcess


@pytest.fixture
def session(tmp_path):
    clock = [100.]
    current = AsyncMotionSession(PACKAGE/'config/imu.yaml', tmp_path/'run',
        clock=lambda: clock[0], context=Context(), duration=1.)
    inputs, index, _ = current.estimate_requests.get_nowait()
    result = current.inference_description.update(inputs, index)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(current.scene.urdf, current.inference_description.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    identifier.command_proposal(result)
    packet = capture_stiffness_packet(result)
    current.estimate_replies.put(dict(result=result, packet=packet, capture_index=0))
    current.poll()
    assert current.start()
    current.desired_velocity[:] = [.02, -.01, .005, .01, -.005, .005]
    yield current, clock, result, identifier
    current.close()


def test_twenty_hz_motion_and_fifteen_hz_archiving_continue_during_blocked_inference(session):
    current, clock, result, identifier = session
    original_ids = tuple(identifier.factors)
    for index in range(1, 21):
        clock[0] = 100.+index*.05
        current.advance()
        # No further inference reply is supplied: expensive work is blocked.
        if index >= 2:
            assert current.inference_busy
    assert current.motion_finished
    assert current.frame.time == 1.
    assert len(current.actions) == 20
    assert len(current.capture_indices) == 16  # includes t=0, ends at t=1
    assert np.linalg.norm(current.command-current.scene.initial_command) > .015
    assert current.assimilated == [0]
    assert tuple(identifier.factors) == original_ids
    assert current.latest_result is result
    rows = [json.loads(line) for line in (current.output/'commands.jsonl').read_text().splitlines()]
    previous_velocity = np.zeros(6)
    for row in rows:
        step = row['step']
        dt = step['end_stamp']-step['start_stamp']
        velocity = np.asarray(row['velocity'])
        assert np.max(np.abs(velocity)) <= current.servo.speed+1e-10
        assert np.max(np.abs(velocity-previous_velocity)) <= current.servo.acceleration*dt+1e-7
        assert step['density_id'] == current.latest_packet.proposal['density_id']
        previous_velocity = velocity
    captures = [json.loads(line) for line in (current.output/'captures.jsonl').read_text().splitlines()]
    assert len(captures) == 16
    # Acquisition does not wait for the archive thread. The fake clock
    # advanced faster than PNG writes, so verify every completed archive at
    # the end; a failed writer must propagate its actual exception.
    for job in current.archive_jobs:
        job.result(timeout=5.)
    current.collect_archives()
    assert all((current.output/row['image']).exists() for row in captures)
    assert all(row['observation_assimilated_at_capture'] is False for row in captures)


def test_latest_capture_selection_keeps_posterior_and_full_intervening_action_path(session):
    current, clock, result, _ = session
    for index in range(1, 21):
        clock[0] = 100.+index*.05
        current.advance()
    # Consume the request already sent at t=.1. Simulate its completion with
    # a real core update, then choose the newest archived capture at t=1.
    inputs, index, steps = current.estimate_requests.get_nowait()
    path = RecordedCommandPath(result, steps, body_ids=current.inference_description.body_ids,
                               actions=current.inference_description.actions)
    updated = current.inference_description.update(inputs, index, command_map=path)
    current.latest_result = updated
    current.last_assimilated_capture = index
    current.inference_busy = False
    current.submit_estimate()
    latest_inputs, selected, remaining = current.estimate_requests.get_nowait()
    assert selected == 20
    assert latest_inputs.stamp == current.frame.stamp
    assert len(remaining) == 20-index
    assert remaining[0].start_stamp == updated.prior.stamp
    assert remaining[-1].end_stamp == current.frame.stamp
    records = [json.loads(line) for line in (current.output/'assimilation_selection.jsonl').read_text().splitlines()]
    assert records[-1]['selected_capture_index'] == 20
    assert records[-1]['archived_not_assimilated_capture_indices']
    assert records[-1]['posterior_reset'] is False
    assert current.latest_result is updated


def test_display_and_control_use_the_same_caught_up_law_after_an_inference_delay(session):
    current, clock, result, _ = session
    for index in range(1, 5):
        clock[0] = 100.+index*.05
        current.advance()
    inference = current.inference_description
    predicted = predict_result(result, tuple(current.actions), target_stamp=current.frame.stamp,
        frame_index=4, body_ids=inference.body_ids, actions=inference.actions)
    snapshot = build_snapshot(predicted, predicted.stages[-1], inference.common)
    display = inference.common['display']
    markers = make_markers(snapshot, display['seed'], display['point_size'])
    wire = io.BytesIO()
    markers.serialize(wire)
    verify_wire_identity(snapshot, MarkerArray().deserialize(wire.getvalue()),
                         seed=display['seed'], point_size=display['point_size'])
    assert snapshot.command == tuple(current.command)
    assert snapshot.stamp == current.frame.stamp
    assert snapshot.observation_ids == ()  # prediction is no new observation
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'hand'} == {
        predicted.stages[-1].components[0].state.state_id}


def test_process_serialization_restores_readonly_law_arrays_without_changing_values(session):
    import pickle
    _, _, result, _ = session
    restored = _freeze_arrays(pickle.loads(pickle.dumps(result)))
    source = result.stages[-1].components[0].state
    target = restored.stages[-1].components[0].state
    np.testing.assert_array_equal(source.covariance, target.covariance)
    assert source.state_id == target.state_id
    assert not target.covariance.flags.writeable
    assert not target.transforms[0].bingham.flags.writeable


def test_async_display_rejects_unlabelled_or_mismatched_delayed_sources(session, monkeypatch):
    current, _, result, _ = session
    snapshot = build_snapshot(result, result.stages[-1], current.inference_description.common)
    display = MotionDisplay.__new__(MotionDisplay)
    bad = dict(execution_mode='async', physical_stamp=current.frame.stamp,
        command=current.command.tolist(), displayed_prediction_stamp=snapshot.stamp,
        displayed_prediction_command=list(snapshot.command), displayed_prediction_age_s=.2)
    with pytest.raises(AssertionError, match='source clocks'):
        display.publish(current.frame, 0, snapshot, 0, execution=bad)


def test_update_rates_exclude_waiting_for_the_user_to_start_motion(session):
    current, _, _, _ = session
    # Bootstrap work finished long before the start service. Subsequent
    # independent workers both complete at 2 Hz while the body is moving.
    current.control_times = [1., 100., 100.5, 101.]
    current.estimate_times = [2., 100.25, 100.75, 101.25]
    current.started = 100.
    record = current.manifest()
    assert record['actual_control_hz'] == 2.
    assert record['actual_estimate_hz'] == 2.


def _complete_estimate(current, previous_result, identifier):
    """A real delayed update, delivered through the coordinator's reply queue."""
    inputs, index, steps = current.estimate_requests.get_nowait()
    inference = current.inference_description
    path = RecordedCommandPath(previous_result, steps,
        body_ids=inference.body_ids, actions=inference.actions)
    result = inference.update(inputs, index, command_map=path)
    result.records['stiffness'] = identifier.update(result)
    assert identifier.command_proposal(result) is not None
    packet = capture_stiffness_packet(result)
    current.estimate_replies.put(dict(result=result, packet=packet, capture_index=index))
    current.poll()
    return result, packet, index


def _complete_control(current):
    """Use a separate mechanics workspace, as the actual control process does."""
    request = current.control_requests.get_nowait()
    inference = current.inference_description
    predicted = predict_result(request['result'], request['steps'],
        target_stamp=request['stamp'], frame_index=request['index'],
        body_ids=inference.body_ids, actions=inference.actions)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    engine = PhysicalActionEngine(current.scene.urdf, inference.body_ids, settings)
    view, mapping = reanchor_retained_stiffness(engine, predicted, request['packet'])
    control = inference.settings['control']
    controller = MovingController(inference, current.scene.robot_description,
        control['goal_offset'], control['goal_orientation_wxyz'], control,
        integration_steps=control['integration_steps'])
    controller.velocity = np.array(request['velocity'], copy=True)
    _, diagnostics = controller.step(view, current.horizon, command_map=mapping)
    snapshot = replace(build_snapshot(view, view.stages[-1], inference.common),
        stiffness_record=view.records['stiffness'])
    display = inference.common['display']
    markers = make_markers(snapshot, display['seed'], display['point_size'])
    verify_wire_identity(snapshot, markers, seed=display['seed'], point_size=display['point_size'])
    return dict(index=request['index'], capture_index=request['capture_index'],
        packet=request['packet'], velocity=controller.velocity.copy(),
        snapshot=snapshot, markers=markers, diagnostics=diagnostics, work_seconds=0.)


def test_faster_inference_cannot_starve_completed_IK_or_roll_back_the_latest_posterior(session):
    current, clock, initial, identifier = session
    initial_packet = current.latest_packet
    current.submit_control()
    delayed_first_control = _complete_control(current)
    # Two real inference results complete before the first IK reply arrives.
    for index in (1, 2):
        clock[0] = 100.+.05*index
        current.advance()
    second, _, second_index = _complete_estimate(current, initial, identifier)
    clock[0] = 100.15
    current.advance()
    third, third_packet, third_index = _complete_estimate(current, second, identifier)
    assert third_index > second_index > delayed_first_control['capture_index']
    assert current.current_control is None
    current.control_replies.put(delayed_first_control)
    current.poll()
    assert current.current_control is delayed_first_control
    assert current.current_control['capture_index'] == 0
    assert current.engine.packet.packet_id == initial_packet.packet_id
    np.testing.assert_array_equal(current.desired_velocity, delayed_first_control['velocity'])
    assert current.latest_result is third
    assert current.latest_packet is third_packet
    assert current.last_assimilated_capture == third_index
    assert len(current.control_times) == 1

    # The next IK starts with that newest law. Another observation overtakes
    # it too; its completion must still supply a usable atomic active plan.
    current.submit_control()
    delayed_third_control = _complete_control(current)
    assert delayed_third_control['capture_index'] == third_index
    clock[0] = 100.20
    current.advance()
    fourth, fourth_packet, fourth_index = _complete_estimate(current, third, identifier)
    assert fourth_index > third_index
    # An inference reply alone must not switch K under an already active plan.
    assert current.engine.packet.packet_id == initial_packet.packet_id
    current.control_replies.put(delayed_third_control)
    current.poll()
    assert current.current_control is delayed_third_control
    assert current.engine.packet.packet_id == third_packet.packet_id
    assert current.latest_packet is fourth_packet
    assert current.latest_result is fourth
    assert current.last_assimilated_capture == fourth_index
    assert len(current.control_times) == 2
    np.testing.assert_array_equal(current.desired_velocity, delayed_third_control['velocity'])
    assert current.current_control['snapshot'].stiffness_record['captured_packet_id'] == third_packet.packet_id
    assert current.current_control['diagnostics']['stiffness_command_correction'][
        'captured_parameter_packet']['packet_id'] == third_packet.packet_id
    clock[0] = 100.25
    current.advance()
    assert current.actions[-1].provenance['packet_id'] == third_packet.packet_id
    assert current.actions[-1].density_id == third_packet.proposal['density_id']
    np.testing.assert_array_equal(current.actions[-1].selected_K, third_packet.proposal['selected_K'])


def test_an_older_control_reply_cannot_replace_the_active_law_K_or_velocity(session):
    current, clock, initial, identifier = session
    current.submit_control()
    initial_control = _complete_control(current)
    current.control_replies.put(initial_control)
    current.poll()
    for index in (1, 2):
        clock[0] = 100.+.05*index
        current.advance()
    updated, packet, capture_index = _complete_estimate(current, initial, identifier)
    # Finish the older in-flight IK; the next submission can use the update.
    old_inflight = _complete_control(current)
    current.control_replies.put(old_inflight)
    current.poll()
    current.submit_control()
    active = _complete_control(current)
    assert active['capture_index'] == capture_index
    current.control_replies.put(active)
    current.poll()
    expected_velocity = current.desired_velocity.copy()
    expected_control_count = len(current.control_times)
    assert current.engine.packet.packet_id == packet.packet_id
    # A duplicate/late reply from the older capture has different numerical
    # values. It cannot roll back any member of the active source bundle.
    late = dict(initial_control, velocity=-np.ones(6))
    current.control_replies.put(late)
    current.poll()
    assert current.current_control is active
    assert current.current_control['snapshot'] is active['snapshot']
    assert current.engine.packet.packet_id == packet.packet_id
    np.testing.assert_array_equal(current.desired_velocity, expected_velocity)
    assert len(current.control_times) == expected_control_count
    assert current.latest_result is updated
    assert current.latest_packet is packet


def test_expired_completed_plan_brakes_to_rest_when_neither_worker_replies(session):
    current,clock,_,_ = session
    current.submit_control()
    completed = _complete_control(current)
    assert np.linalg.norm(completed['velocity']) > 1e-6
    current.control_replies.put(completed)
    current.poll()
    committed = current.control_times[-1]
    previous_velocity = current.velocity.copy()
    expired_velocities = []
    stopped_position = None
    for index in range(1,21):
        clock[0] = 100.+.05*index
        current.advance()
        dt = current.actions[-1].end_stamp-current.actions[-1].start_stamp
        assert np.max(np.abs(current.velocity-previous_velocity)) <= current.servo.acceleration*dt+1e-7
        assert np.max(np.abs(current.velocity)) <= current.servo.speed+1e-10
        if clock[0] > committed+current.horizon:
            expired_velocities.append(current.velocity.copy())
            assert np.all(np.abs(current.velocity) <= np.abs(previous_velocity)+1e-10)
            if np.max(np.abs(current.velocity)) < 1e-12:
                if stopped_position is None:
                    stopped_position = current.command.copy()
                np.testing.assert_array_equal(current.command,stopped_position)
        previous_velocity = current.velocity.copy()
    assert len(expired_velocities) >= 3
    assert stopped_position is not None
    np.testing.assert_array_equal(current.velocity,np.zeros(6))
    assert current.inference_busy and current.control_busy
    assert len(current.control_times) == 1
    assert current.manifest()['ik_plan_expired']
    assert np.all(current.command >= current.servo.lower)
    assert np.all(current.command <= current.servo.upper)


def test_reacquisition_waits_for_the_new_goal_mixture_to_be_committed_with_IK(tmp_path):
    clock = [100.]
    current = AsyncMotionSession(PACKAGE/'config/marker.yaml',tmp_path/'marker_resume',
        clock=lambda:clock[0],context=Context(),duration=1.)
    try:
        inputs,index,_ = current.estimate_requests.get_nowait()
        initial = current.inference_description.update(inputs,index)
        settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
        identifier = StiffnessIdentifier(current.scene.urdf,
            current.inference_description.body_ids,settings)
        initial.records['stiffness'] = identifier.update(initial)
        assert identifier.command_proposal(initial) is not None
        current.estimate_replies.put(dict(result=initial,
            packet=capture_stiffness_packet(initial),capture_index=0))
        current.poll()
        assert current.start()
        current.submit_control()
        old_plan = _complete_control(current)
        current.control_replies.put(old_plan)
        current.poll()
        assert old_plan['snapshot'].goal_memory['available']

        # Inject an image-loss event at the initial capture, then let the
        # actual stop/settle policy run. Drain image jobs before overriding
        # this test's visibility event, so thread timing cannot change it.
        current.last_marker_missing_capture = 0
        current.latest_marker_observed = False
        for index in range(1,8):
            for job in current.archive_jobs:
                job.result()
            current.collect_archives()
            current.marker_preview_visible = False
            clock[0] = 100.+.05*index
            current.advance()
        assert current.last_action_mode != 'reaching'
        assert current.current_control is old_plan

        # A genuine later marker likelihood updates all goal branches, but
        # that result alone cannot resume an older active reaching plan.
        fresh,_,fresh_index = _complete_estimate(current,initial,identifier)
        assert fresh_index > current.last_marker_missing_capture
        assert current.latest_marker_observed
        assert fresh.records['fixed_goal_memory']['observation_stamp'] > initial.prior.stamp
        detected = next(item.detector_result for item in fresh.local_estimates
                        if item.detector_result is not None)
        native = detected.record.distribution.components
        assert len(fresh.stages[-1].goal_components) == len(native)
        assert [goal.weight for goal in fresh.stages[-1].goal_components] == list(detected.weights)
        np.testing.assert_array_equal([goal.log_weight for goal in fresh.stages[-1].goal_components],
            np.asarray(detected.log_masses)-logsumexp(detected.log_masses))
        assert all(goal.label.endswith(mode.component_id)
                   for goal,mode in zip(fresh.stages[-1].goal_components,native))
        for job in current.archive_jobs:
            job.result()
        current.collect_archives()
        current.marker_preview_visible = True
        clock[0] = 100.40
        current.advance()
        assert current.last_action_mode != 'reaching'
        assert current.current_control is old_plan

        # Discard the simulated unfinished old request, then finish a real
        # full-objective IK on the newly fused mixture and commit it.
        while not current.control_requests.empty():
            current.control_requests.get_nowait()
        current.control_busy = False
        current.submit_control()
        fresh_plan = _complete_control(current)
        assert fresh_plan['capture_index'] == fresh_index
        remembered = fresh_plan['snapshot'].goal_memory['goal_components']
        assert len(remembered) == len(native)
        assert [goal['weight'] for goal in remembered] == list(detected.weights)
        np.testing.assert_array_equal([goal['log_weight'] for goal in remembered],
            np.asarray(detected.log_masses)-logsumexp(detected.log_masses))
        current.control_replies.put(fresh_plan)
        current.poll()
        for job in current.archive_jobs:
            job.result()
        current.collect_archives()
        current.marker_preview_visible = True
        clock[0] = 100.45
        current.advance()
        assert current.last_action_mode == 'reaching'
        assert current.current_control is fresh_plan
        assert fresh_plan['snapshot'].goal_memory['observation_stamp'] == fresh.prior.stamp
        assert current.engine.packet.packet_id == fresh_plan['packet'].packet_id
    finally:
        current.close()


def test_close_flushes_archives_and_is_idempotent_for_workers_and_queues(session,monkeypatch):
    current,_,_,_ = session
    calls = []
    for index,worker in enumerate(current.workers):
        monkeypatch.setattr(worker,'join',lambda index=index,**kwargs:calls.append(('join',index)))
        monkeypatch.setattr(worker,'terminate',lambda index=index:calls.append(('terminate',index)))
    queues = (current.estimate_requests,current.estimate_replies,
              current.control_requests,current.control_replies)
    for index,queue in enumerate(queues):
        monkeypatch.setattr(queue,'close',lambda index=index:calls.append(('close_queue',index)))
        monkeypatch.setattr(queue,'cancel_join_thread',
                            lambda index=index:calls.append(('cancel_queue_join',index)))
    current.close()
    first_calls = tuple(calls)
    assert current.closed
    assert all(job.done() and job.exception() is None for job in current.archive_jobs)
    assert all((current.output/'capture_{:05d}.png'.format(index)).exists()
               for index in current.capture_indices)
    assert current.estimate_requests.get_nowait() is None
    assert current.control_requests.get_nowait() is None
    assert [call for call in calls if call[0] == 'close_queue'] == [
        ('close_queue',index) for index in range(4)]
    current.close()
    assert tuple(calls) == first_calls
    assert current.estimate_requests.empty() and current.control_requests.empty()
    with pytest.raises(RuntimeError,match='shutdown'):
        current.archive_executor.submit(lambda:None)
