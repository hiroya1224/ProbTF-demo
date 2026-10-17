"""Actual async workers keep one task/view law through loss and reacquisition."""

from copy import deepcopy

import numpy as np
import pytest

from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor

from test_async_execution import (
    _new_session, _complete_acquisitions, _complete_estimate, _complete_control,
    _advance_ticks,
)


def observed_session(tmp_path, case):
    current, clock, previous, identifier = _new_session(tmp_path, case=case, duration=.3,
        control_mode='running_gradient',
        visibility_factor_options=dict(enabled=True, weight=20., minimum_norm_m=1e-9))
    for _ in range(4):
        if previous.records.get('fixed_goal_memory', {}).get('available', False):
            break
        previous, _, _ = _complete_estimate(current, previous, identifier)
    assert previous.records['fixed_goal_memory']['available']
    current.submit_control()
    reply = _complete_control(current)
    assert reply['gradient_evaluation']['status'] == 'ok', reply['gradient_evaluation']
    current.control_replies.put(reply)
    current.poll()
    assert current.last_gradient_adoption['accepted']
    return current, clock, previous, identifier, reply


def assert_common_source(evaluation):
    factor = evaluation['visibility_factor']
    assert factor['status'] == 'ok'
    source = factor['source_ids']
    assert source['body_state_ids'] == evaluation['body_state_ids']
    assert source['goal_state_ids'] == evaluation['goal_state_ids']
    assert source['parameter_density_id'] == evaluation['parameter_density_id']
    assert source['selected_K'] == evaluation['selected_K']
    assert source['source_command'] == evaluation['source_command']
    assert source['evaluation_command'] == evaluation['source_command']
    assert source['source_stamp'] == evaluation['source_stamp']
    assert source['body_observation_source_stamp'] == evaluation['body_observation_source_stamp']
    assert source['goal_observation_source_stamp'] == evaluation['goal_observation_source_stamp']
    assert len(factor['per_mode_record']) == len(evaluation['source_branches'])
    for mode, task in zip(factor['per_mode_record'], evaluation['source_branches']):
        assert mode['source_state_id'] == task['state_id']
        assert mode['source_command'] == task['query_command'] == evaluation['source_command']
        assert mode['source_stamp'] == task['query_stamp'] == evaluation['source_stamp']
        assert mode['source_version'] == task['query_version']
        assert mode['source_frame'] == 'camera_optical_frame'
    np.testing.assert_array_equal(evaluation['gradient'], evaluation['selected_gradient'])
    np.testing.assert_allclose(evaluation['selected_gradient'],
        np.asarray(evaluation['task_gradient'])+factor['weight']*np.asarray(evaluation['view_gradient']),
        rtol=0., atol=0.)


@pytest.mark.parametrize('case', ['marker', 'imu_marker'])
def test_active_view_worker_and_accepted_packet_share_complete_task_source(tmp_path, case):
    current, _, _, _, reply = observed_session(tmp_path, case)
    try:
        evaluation = reply['gradient_evaluation']
        assert_common_source(evaluation)
        packet = current.gradient_tracker.last_packet
        assert packet.as_dict()['selection_factors'] == evaluation['selection_factors']
        assert packet.source_stamp == reply['snapshot'].stamp
        assert list(packet.source_command) == list(reply['snapshot'].command)
        assert tuple(packet.goal_state_ids) == tuple(evaluation['goal_state_ids'])
        assert tuple(packet.body_state_ids) == tuple(evaluation['body_state_ids'])
        assert current.engine.packet is reply['packet']
        assert current.last_gradient_adoption['state_before'] == current.last_gradient_adoption['state_after']
    finally:
        current.close()


def test_held_marker_view_updates_with_IMU_during_loss_and_new_gradient_keeps_u_v(tmp_path):
    current, clock, previous, identifier, initial = observed_session(tmp_path, 'imu_marker')
    try:
        original_memory = deepcopy(previous.records['fixed_goal_memory'])
        original_mean = np.array(initial['gradient_evaluation']['visibility_factor']['per_mode_record'][0]['position_mean'])
        original_body = initial['gradient_evaluation']['body_state_ids']
        original_packet = current.gradient_tracker.last_packet
        current.marker_preview_visible = False
        current.latest_marker_observed = False
        _advance_ticks(current, clock, 4)
        _complete_acquisitions(current)
        updated = previous
        for _ in range(5):
            updated, latest_K, _ = _complete_estimate(current, updated, identifier)
            if updated.records['sensor_event']['kind'] == 'imu':
                break
        assert updated.records['sensor_event']['kind'] == 'imu'
        assert updated.records['observation_ids']
        assert updated.prior.stamp > original_memory['observation_stamp']
        assert updated.records['fixed_goal_memory'] == original_memory
        # Deliver the already in-flight old-law result, then request the
        # completed IMU law at the actual current execution point.
        delayed = _complete_control(current)
        current.control_replies.put(delayed)
        current.poll()
        current.submit_control()
        fresh = _complete_control(current)
        evaluation = fresh['gradient_evaluation']
        assert evaluation['status'] == 'ok', evaluation
        assert_common_source(evaluation)
        assert evaluation['goal_observation_source_stamp'] == original_memory['observation_stamp']
        assert evaluation['body_observation_source_stamp'] == updated.prior.stamp
        assert evaluation['body_state_ids'] != original_body
        assert evaluation['parameter_density_id'] == latest_K.proposal['density_id']
        changed_mean = evaluation['visibility_factor']['per_mode_record'][0]['position_mean']
        assert not np.array_equal(changed_mean, original_mean)
        u_before, v_before = current.command.copy(), current.velocity.copy()
        current.control_replies.put(fresh)
        current.poll()
        np.testing.assert_array_equal(current.command, u_before)
        np.testing.assert_array_equal(current.velocity, v_before)
        assert current.last_gradient_adoption['state_before'] == current.last_gradient_adoption['state_after']
        assert current.gradient_tracker.last_packet is not original_packet
        clock[0] = 100.+5/120.
        current.advance()
        assert current.last_action_mode == 'reaching'
        assert current.actions[-1].density_id == latest_K.proposal['density_id']
        assert np.linalg.norm(current.command-u_before) > 0.
        assert current.gradient_tracker.last_packet.selection_factors['visibility_factor']['status'] == 'ok'
    finally:
        current.close()


def test_reacquired_image_updates_once_and_replaying_it_cannot_duplicate_BFS_or_K_factor(tmp_path):
    current, clock, previous, identifier, _ = observed_session(tmp_path, 'imu_marker')
    try:
        old_goal_stamp = previous.records['fixed_goal_memory']['observation_stamp']
        current.marker_preview_visible = False
        current.latest_marker_observed = False
        _advance_ticks(current, clock, 12)
        _complete_acquisitions(current)
        updated = previous
        for _ in range(12):
            updated, _, capture_index = _complete_estimate(current, updated, identifier)
            if updated.records['sensor_event']['kind'] == 'image':
                break
        assert updated.records['sensor_event']['kind'] == 'image'
        assert updated.records['fixed_goal_memory']['observation_stamp'] > old_goal_stamp
        assert updated.records['fixed_goal_memory_event'] == 'marker_observation_update'
        event_id = updated.records['sensor_event']['event_id']
        ids = tuple(current.assimilated)
        assert ids.count(event_id) == 1
        original_result = current._test_inference.last_result
        original_density_record = identifier.last_record
        factor_ids = tuple(item.factor_id for item in identifier.factors)
        replay = current._test_inference.update_sensor_event(current.store.get(event_id),
            event_index=capture_index, probability_unit='per_effective_sensor_event')
        assert replay is original_result
        assert identifier.update(replay) is original_density_record
        assert tuple(item.factor_id for item in identifier.factors) == factor_ids
        assert tuple(current.assimilated) == ids
        assert current.store.get(event_id).attempts == 1
        assert replay.records['fixed_goal_memory']['observation_stamp'] == updated.records['fixed_goal_memory']['observation_stamp']
    finally:
        current.close()


def test_unavailable_active_view_is_rejected_without_changing_accepted_task_law_K_or_motion_state(tmp_path, monkeypatch):
    current, clock, _, _, original = observed_session(tmp_path, 'marker')
    try:
        _advance_ticks(current, clock, 2)
        before_packet, before_engine = current.gradient_tracker.last_packet, current.engine.packet
        before_u, before_v = current.command.copy(), current.velocity.copy()
        def unavailable(self, components, queries, *, source_ids=None):
            return dict(enabled=True, weight=self.weight, status='unavailable',
                reason='test: a retained mode has degenerate camera position',
                penalty=None, weighted_penalty=None, log_factor=None,
                source_ids=source_ids, per_mode_record=[])
        monkeypatch.setattr(VisibilityFactor, 'evaluate', unavailable)
        failed = _complete_control(current)
        evaluation = failed['gradient_evaluation']
        assert evaluation['task_status'] == 'ok'
        assert evaluation['task_gradient'] is not None
        assert evaluation['visibility_factor']['status'] == 'unavailable'
        assert evaluation['status'] == 'failed'
        assert evaluation['gradient'] is None
        current.control_replies.put(failed)
        current.poll()
        assert not current.last_gradient_adoption['accepted']
        assert current.gradient_tracker.last_packet is before_packet
        assert current.engine.packet is before_engine
        assert current.current_control is original
        np.testing.assert_array_equal(current.command, before_u)
        np.testing.assert_array_equal(current.velocity, before_v)
        clock[0] = 100.+3/120.
        current.advance()
        assert current.last_action_mode == 'reaching'
        assert current.actions[-1].provenance['action_packet_id'] == before_engine.packet_id
    finally:
        current.close()
