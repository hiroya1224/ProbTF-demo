"""Isolation checks for diagnostic truth substitutions, not production tests."""
import numpy as np
import pytest

from uncertainty_ablation_states import intervene_control_result,oracle_K_packet
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_control import control_components


@pytest.fixture
def initial(tmp_path):
    scene=MovingSwitchScene(tmp_path/'scene')
    frame=scene.step(0.,stamp=1792080000.,command=scene.initial_command,render_camera=False)
    inference=MovingInference(scene.robot_description,())
    result=inference.update(scene.sensor_inputs(frame),0)
    truth=dict(stamp=frame.stamp,command=frame.command.tolist(),
        link_poses={key:value.tolist()for key,value in frame.link_poses.items()},
        marker_pose=scene.marker.tolist(),true_K=scene.config['plant']['stiffness'])
    return result,inference,truth


def test_baseline_is_original_object(initial):
    result,inference,truth=initial
    assert intervene_control_result(result,inference,truth,'baseline')is result
    assert intervene_control_result(result,inference,truth,'oracle_marker')is result
    assert oracle_K_packet(None,result,None,truth)is None


def test_goal_changes_only_task_goal_not_body(initial):
    result,inference,truth=initial
    before=result.stages[-1].components[0].state
    changed=intervene_control_result(result,inference,truth,'oracle_goal')
    body,goals=control_components(changed)
    assert body[0].state is before
    goal=goals[0].state
    index=goal.edge_ids.index('marker');part=slice(6*index,6*index+6)
    assert np.array_equal(goal.transforms[index].translation,np.array(truth['marker_pose'])[:3,3])
    assert np.array_equal(goal.covariance[part,part],np.diag([1e-8]*3+[1e-6]*3))
    assert np.count_nonzero(goal.covariance[part,:6*index])==0
    assert changed.records['fixed_goal_memory']['available']
    assert result.records.get('fixed_goal_memory',{}).get('available',False)is False
    assert 'diagnostic_truth_intervention'not in result.records


def test_body_corrects_local_centers_and_scales_full_covariance(initial):
    result,inference,truth=initial
    before=result.stages[-1].components[0].state
    changed=intervene_control_result(result,inference,truth,'oracle_body')
    body,_=control_components(changed);after=body[0].state
    scales=np.ones(before.mean.size)
    parent='base_link'
    from probtf.geometry import quat_to_rotmat
    for name in inference.body_ids:
        index=after.edge_ids.index(name);scales[6*index:6*index+6]=.01
        expected=np.linalg.solve(np.asarray(truth['link_poses'][parent]),np.asarray(truth['link_poses'][name]))
        assert np.allclose(after.transforms[index].translation,expected[:3,3],atol=1e-15)
        assert np.allclose(quat_to_rotmat(after.transforms[index].rotation_wxyz),expected[:3,:3],atol=1e-14)
        parent=name
    assert np.array_equal(after.covariance,scales[:,None]*before.covariance*scales[None,:])
    assert after.transforms[-1]is before.transforms[-1]
    assert result.stages[-1].components[0].state is before


def test_capture_stamp_must_match(initial):
    result,inference,truth=initial
    truth=dict(truth,stamp=truth['stamp']+.01)
    with pytest.raises(ValueError,match='exact control-capture'):
        intervene_control_result(result,inference,truth,'oracle_body')


def test_body_goal_combination_shares_exact_modified_body(initial):
    result,inference,truth=initial
    changed=intervene_control_result(result,inference,truth,'oracle_body_goal')
    body,goals=control_components(changed)
    assert body[0].state.state_id!=result.stages[-1].components[0].state.state_id
    assert changed.records['fixed_goal_memory']['available']
    assert changed.records['diagnostic_truth_intervention']['body_sd_scale']==.01
    assert np.array_equal(body[0].state.covariance[:-6,:-6],goals[0].state.covariance[:-6,:-6])
    for left,right in zip(body[0].state.transforms[:-1],goals[0].state.transforms[:-1]):
        assert np.array_equal(left.translation,right.translation)
        assert np.array_equal(left.rotation_wxyz,right.rotation_wxyz)
        if left.bingham is not None:assert np.array_equal(left.bingham,right.bingham)


def test_body_oracle_scales_nonzero_cross_and_preserves_biased_goal(initial):
    """Center/width oracle does not also correct the retained root goal.

    Initial laws happen to be independent. Insert a valid correlated joint
    law so the isolation assertion actually exercises cross congruence.
    """
    from dataclasses import replace
    from probtf.spherical_law.first_order import JointTransformState

    result,inference,truth=initial
    last=result.stages[-1]
    source=last.components[0].state
    covariance=np.array(source.covariance,copy=True)
    body_index=source.edge_ids.index(inference.body_ids[0])
    marker_index=source.edge_ids.index(inference.marker.edge_id)
    a,b=6*body_index+3,6*marker_index
    covariance[a,b]=covariance[b,a]=.25*np.sqrt(covariance[a,a]*covariance[b,b])
    assert covariance[a,b]!=0.
    assert np.min(np.linalg.eigvalsh(covariance))>=0.
    correlated=JointTransformState(source.transforms,covariance,
        version=source.version,stamp=source.stamp,command=source.command,
        component_id=source.component_id)
    supplied=replace(result,stages=result.stages[:-1]+(replace(last,
        components=(replace(last.components[0],state=correlated),),
        goal_components=()),))
    changed=intervene_control_result(supplied,inference,truth,'oracle_body')
    body,goals=control_components(changed)
    after=body[0].state
    assert after.covariance[a,b]==.01*covariance[a,b]
    marker_part=slice(6*marker_index,6*marker_index+6)
    assert np.array_equal(after.covariance[marker_part,marker_part],
                          covariance[marker_part,marker_part])
    assert after.transforms[marker_index]is source.transforms[marker_index]
    assert np.array_equal(goals[0].state.transforms[marker_index].translation,
                          source.transforms[marker_index].translation)
    assert np.array_equal(goals[0].state.covariance[marker_part,marker_part],
                          covariance[marker_part,marker_part])
    assert np.array_equal(correlated.covariance,covariance)


def test_known_K_packet_has_its_own_provenance_and_real_equilibrium(initial,tmp_path):
    import hashlib,json
    from pathlib import Path
    import yaml
    from probtf_observation_fusion_demo.async_stiffness import PhysicalActionEngine,FrozenKPacket
    result,inference,truth=initial
    urdf=tmp_path/'robot.urdf'
    # The generated description is the same known geometry used by inference.
    import rospkg
    package=Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    scene=MovingSwitchScene(tmp_path/'second_scene')
    urdf.write_text(scene.robot_description)
    settings=yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
    engine=PhysicalActionEngine(urdf,tuple(inference.body_ids),settings)
    record=engine.identifier.update(result)
    old_density=record.get('posterior')or record.get('retained_prior')
    record['command_proposal']=dict(density_id=old_density['density_id'],
        source_command=list(result.prior.command),source_stamp=result.prior.stamp,
        selected_K=[20.]*6,selected_logK=np.log([20.]*6).tolist(),
        auxiliary_equilibrium=list(result.prior.command))
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
    original=FrozenKPacket(encoded,hashlib.sha256(encoded.encode()).hexdigest())
    replacement=oracle_K_packet(engine,result,original,truth)
    assert original.record_json==encoded
    assert replacement.proposal['selected_K']==truth['true_K']
    assert replacement.proposal['density_id']!=old_density['density_id']
    assert replacement.record['posterior']['estimated_density_claimed']is False
    engine.set(replacement)
    theta=np.asarray(replacement.proposal['auxiliary_equilibrium'])
    command=np.asarray(result.prior.command)
    residual=engine.identifier.robot.tau_gravity(theta)+engine.identifier.spring.torque(
        theta,command,np.asarray(truth['true_K']))
    assert np.max(np.abs(residual))<1e-7
