"""A newly received force must never rewrite an already elapsed interval."""
import json
import numpy as np

from probtf_observation_fusion_demo.running_gradient import GradientPacket, step_running_gradient
from test_async_execution import _new_session


def _message(session, gradient, request_id):
    result=session.latest_result
    command=list(session.command)
    diagnostics=dict(reaching_phase='running_gradient',source_stamp=session.frame.stamp,
        source_command=command,source_version=result.prior.version,
        body_source_state_ids=[c.state.state_id for c in result.stages[-1].components],
        goal_source_state_ids=[])
    evaluation=dict(gradient=list(gradient),source_command=command,
        source_stamp=session.frame.stamp,body_state_ids=diagnostics['body_source_state_ids'],
        goal_state_ids=[],parameter_density_id=None,selected_K=None,
        objective_at_source=1.,computation_seconds=.1,finite_difference_diagnostics={},
        status='ok',snapshot_update_count=len(session.assimilated))
    return dict(index=len(session.frames)-1,request_id=request_id,event_id='diagnostic',
        capture_index=session.last_assimilated_capture,control_mode='running_gradient',
        gradient_evaluation=evaluation,diagnostics=diagnostics,packet=None,velocity=None)


def test_old_packet_integrates_until_next_boundary_and_received_packet_preserves_uv(tmp_path):
    session,clock,_,_=_new_session(tmp_path,control_mode='running_gradient',duration=2.)
    try:
        old=np.array([4.,0.,0.,0.,0.,0.])
        first=_message(session,old,0)
        session.current_control=first
        session.gradient_tracker.accept(GradientPacket(**first['gradient_evaluation'],packet_id='old'),
                                        session.frame.stamp,len(session.assimilated))
        start=session.command.copy()
        new=_message(session,-old,1)
        session.control_busy=True
        session.pending_control_request=1
        clock[0]=100.5
        session.control_replies.put(new)
        session.poll()
        assert session.gradient_tracker.last_packet.packet_id=='old'
        assert len(session.pending_control_activations)==1
        clock[0]=101.
        session.advance()
        expected_u,expected_v=step_running_gradient(start,np.zeros(6),old,1.,1e-6,8.)
        np.testing.assert_array_equal(session.command,expected_u)
        np.testing.assert_array_equal(session.velocity,expected_v)
        adoption=session.last_gradient_adoption
        assert adoption['adoption_stamp']==1792080001.
        assert adoption['received_motion_elapsed_s']==.5
        assert adoption['state_before']==adoption['state_after']
        first_interval=json.loads((session.output/'commands.jsonl').read_text().splitlines()[-1])
        assert first_interval['reaching_execution']['gradient_packet_id']=='old'
        clock[0]=101.1
        session.advance()
        second_interval=json.loads((session.output/'commands.jsonl').read_text().splitlines()[-1])
        assert second_interval['reaching_execution']['gradient_packet_id']==adoption['gradient_packet']['packet_id']
    finally:
        session.close()
