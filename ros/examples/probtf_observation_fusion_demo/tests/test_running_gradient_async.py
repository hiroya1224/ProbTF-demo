"""The actual worker, retained law, actuator and publication use one gradient."""
import io
import json

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.display import verify_wire_identity
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_control import MovingController
from probtf_observation_fusion_demo.running_gradient import step_running_gradient
from probtf_observation_fusion_demo.event_motion import _gradient_identity

from test_async_execution import (
    _new_session, _complete_acquisitions, _complete_estimate, _complete_control,
    _advance_ticks, _jsonlines,
)


def _forbid_endpoint(*_args, **_kwargs):
    raise AssertionError('Running gradient must not call endpoint IK, QP or servo clipping.')


def test_zero_mass_log_weight_keeps_its_law_and_has_a_strict_JSON_identity():
    law=dict(weights=[1.,0.],log_weights=[0.,-np.inf])
    identity=json.dumps(_gradient_identity(law),sort_keys=True,allow_nan=False)
    assert json.loads(identity)['log_weights'][1]=={'nonfinite_float_identity':'-inf'}
    assert law['log_weights'][1]==-np.inf and law['weights'][1]==0.


def _observed_marker_session(tmp_path, monkeypatch, case='marker'):
    current, clock, prior, identifier = _new_session(
        tmp_path, case=case, duration=1., control_mode='running_gradient')
    # IMU and image are separate events, even when acquired at the same frame.
    while not prior.records.get('fixed_goal_memory', {}).get('available', False):
        prior, _, _ = _complete_estimate(current, prior, identifier)
    monkeypatch.setattr(MovingController, 'reaching_step', _forbid_endpoint)
    monkeypatch.setattr(MovingController, '_command_bounds', _forbid_endpoint)
    current.submit_control()
    reply = _complete_control(current)
    assert reply['gradient_evaluation']['status']=='ok', reply['gradient_evaluation']
    current.control_replies.put(reply); current.poll()
    assert current.last_gradient_adoption['accepted']
    return current, clock, prior, identifier, reply


def test_no_observed_goal_is_a_declared_hold_not_a_fabricated_J(tmp_path, monkeypatch):
    current, clock, _, _ = _new_session(
        tmp_path, case='marker', control_mode='running_gradient')
    try:
        monkeypatch.setattr(MovingController, 'gradient_at_command', _forbid_endpoint)
        monkeypatch.setattr(MovingController, 'reaching_step', _forbid_endpoint)
        monkeypatch.setattr(MovingController, '_command_bounds', _forbid_endpoint)
        before=current.command.copy()
        current.submit_control(); reply=_complete_control(current)
        assert reply['gradient_evaluation']['status']=='waiting_for_goal'
        assert reply['gradient_evaluation']['gradient'] is None
        assert reply['gradient_evaluation']['objective_at_source'] is None
        current.control_replies.put(reply); current.poll()
        assert not current.last_gradient_adoption['accepted']
        _advance_ticks(current, clock, 120)
        np.testing.assert_array_equal(current.command,before)
        np.testing.assert_array_equal(current.velocity,np.zeros(6))
        assert current.gradient_tracker.last_packet is None
        assert current.period==pytest.approx(1/120.)
        _complete_acquisitions(current)
        events=current.store.events()
        assert {kind:sum(e.kind==kind for e in events) for kind in current.rates}=={
            'joint':41,'imu':31,'image':13}
        assert MotionDisplay._reaching_phase_label(current.scene.trajectory[-1])=='WAITING FOR OBSERVED GOAL'
    finally:
        current.close()


@pytest.mark.parametrize('case',['marker','imu_marker'])
def test_retained_goal_runs_during_image_loss_and_busy_workers_with_exact_dynamics(tmp_path, monkeypatch, case):
    current,clock,_,_,reply=_observed_marker_session(tmp_path,monkeypatch,case)
    try:
        source=current.command.copy()
        packet=current.gradient_tracker.last_packet
        current.marker_preview_visible=False
        current.latest_marker_observed=False
        _advance_ticks(current,clock,12)
        expected_u,expected_v=step_running_gradient(source,np.zeros(6),packet.gradient,
            current.frame.time,current.gradient_tracker.eta,current.gradient_tracker.gamma)
        np.testing.assert_allclose(current.command,expected_u,atol=1e-14,rtol=0.)
        np.testing.assert_allclose(current.velocity,expected_v,atol=1e-14,rtol=0.)
        assert np.linalg.norm(current.command-source)>0.
        assert current.control_busy and (current.inference_busy or current.database_selection_pending)
        assert all(row['behavior']['mode']=='reaching' for row in _jsonlines(current.output/'commands.jsonl'))
        assert current.gradient_tracker.last_packet is packet
        assert current.engine.packet is reply['packet']
        for action in current.actions:
            np.testing.assert_array_equal(action.selected_K,packet.selected_K)
            assert action.density_id==packet.parameter_density_id
        assert current.manifest()['ik_plan_horizon_s'] is None
        assert not current.manifest()['velocity_clipping_enabled']
        assert current.manifest()['gradient_source_stamp']==packet.source_stamp
        wire=MarkerArray().deserialize(reply['markers_wire'])
        display=current.inference_description.common['display']
        assert verify_wire_identity(reply['snapshot'],wire,seed=display['seed'],point_size=display['point_size'])
        assert {p.query.state_id for p in reply['snapshot'].points if p.frame_label=='hand'}==set(packet.body_state_ids)
        assert {p.query.state_id for p in reply['snapshot'].points if p.frame_label=='switch_goal'}==set(packet.goal_state_ids)
        assert reply['diagnostics']['ik_update_count']==0
        assert reply['diagnostics']['endpoint_selection_performed'] is False
    finally:
        current.close()


def test_delayed_gradient_adoption_preserves_actual_state_and_does_not_rewrite_intervals(tmp_path,monkeypatch):
    current,clock,_,_,_= _observed_marker_session(tmp_path,monkeypatch)
    try:
        _advance_ticks(current,clock,4)
        delayed=_complete_control(current)
        prior_actions=tuple(action.record() for action in current.actions)
        # Finish more physical ticks before delivering this immutable result.
        for tick in range(5,9):
            clock[0]=100.+tick/120.; current.advance()
        before_u,before_v=current.command.copy(),current.velocity.copy()
        current.control_replies.put(delayed); current.poll()
        np.testing.assert_array_equal(current.command,before_u)
        np.testing.assert_array_equal(current.velocity,before_v)
        np.testing.assert_array_equal(current.gradient_tracker.position,before_u)
        np.testing.assert_array_equal(current.gradient_tracker.velocity,before_v)
        assert tuple(action.record() for action in current.actions[:4])==prior_actions
        adoption=current.last_gradient_adoption
        assert adoption['accepted'] and adoption['gradient_source_age_s']>0.
        assert adoption['state_before']==adoption['state_after']
        assert delayed['snapshot'].stamp<current.frame.stamp
        assert delayed['snapshot'].stamp==current.gradient_tracker.last_packet.source_stamp
        assert current.engine.packet is delayed['packet']
    finally:
        current.close()


def test_failed_evaluation_is_logged_rejected_and_keeps_the_accepted_law_K_and_gradient(tmp_path,monkeypatch):
    current,clock,_,_,original=_observed_marker_session(tmp_path,monkeypatch)
    try:
        _advance_ticks(current,clock,2)
        failed=_complete_control(current)
        failed=dict(failed,gradient_evaluation=dict(failed['gradient_evaluation'],
            status='failed',reason='test: both finite-difference sides unavailable',gradient=None))
        old_gradient=current.gradient_tracker.last_packet
        old_engine=current.engine.packet
        current.control_replies.put(failed); current.poll()
        assert not current.last_gradient_adoption['accepted']
        assert current.gradient_tracker.last_packet is old_gradient
        assert current.current_control is original and current.engine.packet is old_engine
        assert current.last_gradient_adoption['evaluation_reason'].startswith('test:')
        clock[0]=100.+3/120.; current.advance()
        assert current.actions[-1].provenance['action_packet_id']==old_engine.packet_id
    finally:
        current.close()


def test_IMU_only_update_and_new_K_drive_the_held_image_goal_without_sensor_synchronization(tmp_path,monkeypatch):
    current,clock,previous,identifier,_=_observed_marker_session(tmp_path,monkeypatch,'imu_marker')
    try:
        old_goal_stamp=previous.records['fixed_goal_memory']['observation_stamp']
        _advance_ticks(current,clock,4)
        _complete_acquisitions(current)
        updated,new_K,_=_complete_estimate(current,previous,identifier)
        assert updated.records['sensor_event']['kind']=='imu'
        assert updated.prior.stamp>old_goal_stamp
        assert updated.records['fixed_goal_memory']['observation_stamp']==old_goal_stamp
        # Finish the already outstanding evaluation, then request the new law.
        delayed=_complete_control(current)
        current.control_replies.put(delayed); current.poll()
        current.submit_control(); fresh=_complete_control(current)
        assert fresh['gradient_evaluation']['status']=='ok'
        evaluation=fresh['gradient_evaluation']
        assert evaluation['body_observation_source_stamp']==updated.prior.stamp
        assert evaluation['goal_observation_source_stamp']==old_goal_stamp
        assert set(evaluation['observation_source_stamps'])=={updated.prior.stamp,old_goal_stamp}
        assert evaluation['parameter_density_id']==new_K.proposal['density_id']
        np.testing.assert_array_equal(evaluation['selected_K'],new_K.proposal['selected_K'])
        before_u,before_v=current.command.copy(),current.velocity.copy()
        current.marker_preview_visible=False; current.latest_marker_observed=False
        current.control_replies.put(fresh); current.poll()
        np.testing.assert_array_equal(current.command,before_u)
        np.testing.assert_array_equal(current.velocity,before_v)
        clock[0]=100.+5/120.; current.advance()
        assert current.last_action_mode=='reaching'
        assert current.actions[-1].density_id==new_K.proposal['density_id']
        assert current.current_control is fresh
    finally:
        current.close()


def test_unavailable_K_branch_is_a_rejected_provenance_packet_not_a_worker_crash(tmp_path,monkeypatch):
    current,clock,_,_,original=_observed_marker_session(tmp_path,monkeypatch)
    try:
        _advance_ticks(current,clock,2)
        def unavailable(*_args,**_kwargs):
            raise ValueError('test: captured stable branch unavailable')
        monkeypatch.setattr('probtf_observation_fusion_demo.async_workers.capture_direct_stiffness_action',unavailable)
        failed=_complete_control(current)
        evaluation=failed['gradient_evaluation']
        assert evaluation['status']=='failed' and evaluation['gradient'] is None
        assert evaluation['body_state_ids'] and evaluation['goal_state_ids']
        assert evaluation['command_interval_id']==failed['index']
        old_gradient=current.gradient_tracker.last_packet
        current.control_replies.put(failed); current.poll()
        assert not current.last_gradient_adoption['accepted']
        assert current.current_control is original
        assert current.gradient_tracker.last_packet is old_gradient
        assert (current.output/'gradient_failures.jsonl').is_file()
    finally:
        current.close()
