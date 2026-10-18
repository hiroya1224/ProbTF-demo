"""All-event joint batches publish one terminal law with original clocks."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.body_native import point_query
from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath, predict_result
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore


CONFIG = Path(__file__).resolve().parents[1]/'config'


@pytest.fixture(scope='module')
def acquired(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('joint_batch_scene'),imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0.,1000.))
    templates = yaml.safe_load((CONFIG/'imu_marker.yaml').read_text())['observations']
    return scene.robot_description,inputs,templates


@pytest.fixture
def batch_problem(tmp_path,acquired):
    description,inputs,templates = acquired
    store = SensorEventStore(tmp_path/'events.sqlite3')
    q0 = np.asarray(inputs.command)
    commands = [q0+index*np.array([.002,0.,-.001,0.,0.,0.]) for index in range(5)]
    stamps = [1000.+.05*index for index in range(5)]
    ledger = tuple(ActionStep(index,stamps[index],stamps[index+1],tuple(commands[index]),
        tuple(commands[index+1]),(.0015,0.,-.0007,0.,0.,0.),
        density_id='recorded-K:'+str(index),selected_K=(100.,110.,120.,130.,140.,150.))
        for index in range(4))
    def event(identifier,kind,index,**extra):
        payload = dict(time=.05*index,frame_index=index)
        if kind == 'image':
            payload.update(image=inputs.image,image_stamp=stamps[index])
        elif kind == 'imu':
            payload.update(gravity_in_sensor=inputs.gravity_in_sensor,
                imu_target_link=inputs.imu_target_link,imu_sigma_rad=inputs.imu_sigma_rad)
        payload.update(extra)
        return store.append(identifier,kind,stamps[index],commands[index],payload)
    events = (event('joint:start','joint',0),event('image:first','image',0),
              event('imu:middle','imu',1),event('image:last','image',2),
              event('joint:end','joint',3))
    model = MovingInference(description,templates)
    yield model,events,ledger,event
    store.close()


def test_joint_batch_keeps_every_event_and_projects_only_once(batch_problem,monkeypatch):
    import probtf_observation_fusion_demo.moving_inference as moving_module
    model,events,ledger,_ = batch_problem
    calls = []
    original = moving_module.project_update
    def projection(result, **options):
        calls.append(result)
        return original(result, **options)
    monkeypatch.setattr(moving_module,'project_update',projection)
    result = model.update_sensor_batch(events,ledger[:3],batch_id='all-acquired')
    record = result.records['joint_batch']
    assert len(calls) == 1
    assert record['raw_event_ids'] == [event.event_id for event in events]
    assert record['effective_event_ids'] == ['image:first','imu:middle','image:last']
    assert result.records['valid_observation_count'] == 3
    assert set(model._sensor_events) == {event.event_id for event in events}
    assert record['intermediate_projection_count'] == 0
    assert record['terminal_projection_count'] == 1
    assert record['old_per_event_recursion_equivalent'] is False
    assert len(result.stages[-1].components) == 1
    assert result.records['bayes_factor_surprise']['effective_event_count'] == 3
    assert result.records['bayes_factor_surprise']['change_probability'] == pytest.approx(1.-.75**3)
    for parent in record['parent_fusions']:
        assert parent['observation_ids'] == ['image:first','imu:middle','image:last']
        assert parent['all_cross_covariances_retained']
        assert not parent['component_pruning_used']


def test_start_to_end_ledger_and_all_factor_acquisition_clocks_are_unchanged(batch_problem):
    model,events,ledger,_ = batch_problem
    result = model.update_sensor_batch(events,ledger[:3],batch_id='ledger')
    record = result.records['joint_batch']
    assert record['source_stamp'] == 1000.
    assert record['endpoint_stamp'] == 1000.15
    assert record['actual_ledger'] == [step.record() for step in ledger[:3]]
    assert [row['density_id'] for row in record['actual_ledger']] == ['recorded-K:0','recorded-K:1','recorded-K:2']
    originals = {event.event_id:event for event in events}
    for parent in record['likelihood_factors']:
        assert len(parent) == 3
        for factor in parent:
            event = originals[factor['observation_id']]
            assert factor['common_factor_stamp'] == 1000.
            assert factor['acquisition_stamp'] == event.stamp
            assert factor['acquisition_command'] == list(event.command)
            assert factor['acquisition_event']['payload_hash'] == event.payload_hash
    body = result.stages[-1].components[0].state
    assert body.stamp == events[-1].stamp and body.command == events[-1].command
    assert model.candidate_measure.components[0].state is body
    assert result.records['candidate_measure']['shape_source_state_ids'] == [body.state_id]
    block_diagonal = np.zeros_like(body.covariance)
    for index in range(len(body.transforms)):
        block_diagonal[6*index:6*index+6,6*index:6*index+6] = body.covariance[6*index:6*index+6,6*index:6*index+6]
    assert np.linalg.norm(body.covariance-block_diagonal) > 1e-5
    # Propagation uses this same full covariance, never a display-only body.
    hand = point_query(body,result.hand_path)
    np.testing.assert_allclose(hand.covariance,hand.jacobian @ body.covariance @ hand.jacobian.T,atol=1e-12)


def test_goal_uses_latest_image_capture_clock_then_actual_suffix_to_endpoint(batch_problem):
    model,events,ledger,_ = batch_problem
    result = model.update_sensor_batch(events,ledger[:3],batch_id='goal-clock')
    memory = result.records['fixed_goal_memory']
    assert memory['observation_ids'] == ['image:last']
    assert memory['observation_stamp'] == events[-2].stamp
    assert memory['observation_command'] == list(events[-2].command)
    assert result.records['fixed_goal_memory_event'] == 'marker_observation_update'
    assert result.stages[-1].goal_components
    for goal in result.stages[-1].goal_components:
        assert goal.state.stamp == events[-1].stamp
        assert goal.state.command == events[-1].command
    actual_ids = [goal.state.state_id for goal in result.stages[-1].goal_components]
    assert [goal['state_id'] for goal in result.records['stages'][-1]['goal_components']] == actual_ids
    assert result.records['internal_projection']['goal_state_ids'] == actual_ids
    assert result.records['internal_projection']['goal_source_state_ids'] == actual_ids
    assert set(result.records['observation_ids']) == {'image:first','imu:middle','image:last'}


def test_missing_later_image_retains_root_goal_and_does_not_add_evidence(batch_problem):
    model,events,ledger,event = batch_problem
    first = model.update_sensor_batch(events,ledger[:3],batch_id='observed')
    memory = deepcopy(first.records['fixed_goal_memory'])
    mass = model.candidate_measure.log_total_mass
    old_goals = first.stages[-1].goal_components
    missing = event('image:missing','image',4,image=np.full_like(first.image,255))
    second = model.update_sensor_batch((missing,),ledger[3:],batch_id='missing')
    assert second.records['valid_observation_count'] == 0
    assert second.records['joint_batch']['effective_event_ids'] == []
    assert model.candidate_measure.log_total_mass == mass
    assert second.records['fixed_goal_memory'] == memory
    assert not second.records['bayes_factor_surprise']['applied']
    assert second.records['joint_batch_source']['last_effective_observation_stamp'] == events[-2].stamp
    path=RecordedCommandPath(first,ledger[3:],body_ids=model.body_ids,actions=model.actions)
    expected_body=path.push(first.stages[-1].components[0].state,missing.command,model.body_ids,model.actions)
    actual_body=second.stages[-1].components[0].state
    np.testing.assert_array_equal(actual_body.covariance,expected_body.covariance)
    for expected,actual in zip(expected_body.transforms,actual_body.transforms):
        np.testing.assert_array_equal(actual.translation,expected.translation)
        np.testing.assert_array_equal(actual.rotation_wxyz,expected.rotation_wxyz)
        np.testing.assert_array_equal(actual.bingham,expected.bingham)
    for before,after in zip(old_goals,second.stages[-1].goal_components):
        a = before.state.edge_ids.index(model.marker.edge_id)
        b = after.state.edge_ids.index(model.marker.edge_id)
        np.testing.assert_array_equal(before.state.transforms[a].translation,after.state.transforms[b].translation)
        np.testing.assert_array_equal(before.state.transforms[a].bingham,after.state.transforms[b].bingham)
        assert after.log_weight == before.log_weight
        expected=path.push(before.state,missing.command,model.body_ids,model.actions)
        # All body/goal cross blocks survive the same actual action, along
        # with the unchanged root-goal marginal. No new sensor was fused.
        np.testing.assert_allclose(after.state.covariance,expected.covariance,rtol=2e-12,atol=1e-14)


def test_replay_and_missing_ledger_are_explicit_structural_errors(batch_problem):
    model,events,ledger,_ = batch_problem
    with pytest.raises(ValueError,match='missing|contiguous|endpoint'):
        model.update_sensor_batch(events,(ledger[0],ledger[2]),batch_id='gap')
    assert model.last_result is None
    model.update_sensor_batch(events,ledger[:3],batch_id='valid')
    with pytest.raises(ValueError,match='checkpoint replay'):
        model.update_sensor_batch(events,ledger[:3],batch_id='duplicate')


def test_disabled_sensor_notifications_are_archived_without_duplicate_hazards(batch_problem):
    model,events,ledger,_ = batch_problem
    model.templates = tuple(template for template in model.templates if template['sensor'] == 'marker')
    result = model.update_sensor_batch(events,ledger[:3],batch_id='marker-only')
    assert result.records['joint_batch']['raw_event_ids'] == [event.event_id for event in events]
    assert result.records['joint_batch']['effective_event_ids'] == ['image:first','image:last']
    assert result.records['valid_observation_count'] == 2
    assert result.records['bayes_factor_surprise']['effective_event_count'] == 2
    assert result.records['bayes_factor_surprise']['change_probability'] == pytest.approx(1.-.75**2)
    assert set(model._sensor_events) == {event.event_id for event in events}


def test_joint_batch_inference_uses_no_monte_carlo(batch_problem,monkeypatch):
    model,events,ledger,_ = batch_problem
    def forbidden(*args,**kwargs):
        raise AssertionError('Joint-batch inference must not sample candidate transforms.')
    for name in ('default_rng','normal','random','randn','multivariate_normal','RandomState'):
        monkeypatch.setattr(np.random,name,forbidden)
    result = model.update_sensor_batch(events,ledger[:3],batch_id='deterministic')
    assert result.records['joint_batch']['monte_carlo_used'] is False
    assert all(not parent['sampling_used'] for parent in result.records['joint_batch']['parent_fusions'])


def test_completed_batch_handoff_prediction_control_and_actual_wire_share_one_law(batch_problem,acquired):
    from probtf.geometry import quat_to_rotmat
    from probtf_observation_fusion_demo.async_workers import _handoff_result
    from probtf_observation_fusion_demo.display import make_markers,verify_wire_identity
    from probtf_observation_fusion_demo.moving_control import MovingController
    from probtf_observation_fusion_demo.node import build_snapshot
    model,events,ledger,_ = batch_problem
    completed = model.update_sensor_batch(events,ledger[:3],batch_id='law-boundary')
    # The asynchronous process handoff strips raw likelihood intermediates,
    # and actual executed motion then changes the common candidate law.
    handed = _handoff_result(completed)
    current = predict_result(handed,ledger[3:],ledger[3].end_stamp,4,model.body_ids,model.actions)
    control = model.settings['control']
    controller = MovingController(model,acquired[0],control['goal_offset'],
        control['goal_orientation_wxyz'],control,integration_steps=120)
    evaluation = controller.gradient_at_command(current).record
    assert evaluation['status'] == 'ok',evaluation.get('reason')
    stage = current.stages[-1]
    snapshot = build_snapshot(current,stage,model.common)
    settings = model.common['display']
    actual = make_markers(snapshot,settings['seed'],settings['point_size'])
    assert verify_wire_identity(snapshot,actual,seed=settings['seed'],point_size=settings['point_size'])
    assert evaluation['body_state_ids'] == [item.state.state_id for item in stage.components]
    assert evaluation['goal_state_ids'] == [item.state.state_id for item in stage.goal_components]
    assert current.records['candidate_measure']['shape_source_state_ids'] == evaluation['body_state_ids']
    assert current.records['fixed_goal_memory']['observation_stamp'] == events[-2].stamp
    assert snapshot.stamp == ledger[3].end_stamp > events[-2].stamp
    body = {item.label:item for item in stage.components}
    goals = {item.label:item for item in stage.goal_components}
    paths = dict(current.link_paths,hand=current.hand_path,camera=current.camera_path,marker=current.marker_path)
    axis = float(settings['axis_length'])
    endpoints = dict(origin=np.zeros(3),x=[axis,0.,0.],y=[0.,axis,0.],z=[0.,0.,axis])
    for point in snapshot.points:
        source = (goals if point.frame_label in ('marker','switch_goal') else body)[point.component_id]
        if point.frame_label == 'switch_goal':
            path = current.marker_path
            endpoint = np.asarray(control['goal_offset'])+quat_to_rotmat(
                control['goal_orientation_wxyz']) @ endpoints[point.axis]
        else:
            path,endpoint = paths[point.frame_label],endpoints[point.axis]
        independent = point_query(source.state,path,endpoint,frame=current.root_frame)
        assert point.query.state_id == independent.state_id
        np.testing.assert_array_equal(point.query.mean,independent.mean)
        np.testing.assert_array_equal(point.query.covariance,independent.covariance)
        assert point.log_weight == source.log_weight


def test_stiffness_ratios_keep_each_acquisition_command_instead_of_the_endpoint(batch_problem):
    from scipy.special import logsumexp
    from probtf_observation_fusion_demo.async_workers import _handoff_result
    model,events,ledger,_ = batch_problem
    result = model.update_sensor_batch(events,ledger[:3],batch_id='per-command')
    specifications = result.time_resolved_body_ratios
    assert [item['observation_id'] for item in specifications] == ['image:first','imu:middle','image:last']
    original = {event.event_id:event for event in events}
    for item in specifications:
        event = original[item['observation_id']]
        assert item['source_stamp'] == event.stamp
        assert item['source_command'] == event.command
        assert item['source_command'] != result.prior.command
        assert item['provenance']['payload_hash'] == event.payload_hash
        assert not item['provenance']['terminal_command_substituted']
        assert not item['provenance']['intermediate_body_projection_applied']
        assert logsumexp([branch.log_weight for branch in item['branches']]) == pytest.approx(0.,abs=1e-12)
        assert logsumexp([child.log_weight for branch in item['branches']
                          for child in branch.posterior_components]) == pytest.approx(0.,abs=1e-12)
        for branch in item['branches']:
            assert branch.reference_state.stamp == event.stamp
            assert branch.reference_state.command == event.command
            assert branch.reference_state.component_id == branch.label
            assert branch.prior.covariance.shape == result.prior.covariance.shape
    assert any(np.linalg.norm(branch.prior.mean) > 1e-5
               for item in specifications[1:] for branch in item['branches'])
    # Historical computational charts stay in the estimator. IK/display get
    # only the same installed final canonical body and its proper source.
    handoff = _handoff_result(result)
    assert not handoff.time_resolved_body_ratios
    assert handoff.stages[-1].components[0].state is result.stages[-1].components[0].state
    predicted=predict_result(result,(),target_stamp=result.prior.stamp,frame_index=3,
        body_ids=model.body_ids,actions=model.actions)
    assert not predicted.time_resolved_body_ratios
    assert result.time_resolved_body_ratios is specifications
    assert predicted.records['joint_batch_source']==result.records['joint_batch_source']


def test_common_extreme_evidence_keeps_final_and_every_acquisition_law(batch_problem,acquired,monkeypatch):
    """A common density scale cannot erase Keep/Reset or native masses."""
    import probtf_observation_fusion_demo.joint_batch_inference as module
    from probtf_observation_fusion_demo.joint_batch_fusion import add_log_evidence_parts
    model,events,ledger,_=batch_problem
    baseline=model.update_sensor_batch(events,ledger[:3],batch_id='common-scale')
    original=module.fuse_gaussian_likelihood_factors_with_prefixes
    def scaled(*args,**kwargs):
        history=original(*args,**kwargs)
        def scale(item,field):
            parts=add_log_evidence_parts(getattr(item,field+'_parts'),(-1e20,))
            return replace(item,**{field:float(sum(parts)),field+'_parts':parts})
        return replace(history,fusion=scale(history.fusion,'log_evidence'),
            prefixes=tuple(scale(prefix,'cumulative_log_evidence') for prefix in history.prefixes))
    monkeypatch.setattr(module,'fuse_gaussian_likelihood_factors_with_prefixes',scaled)
    changed=MovingInference(acquired[0],acquired[2]).update_sensor_batch(
        events,ledger[:3],batch_id='common-scale')
    np.testing.assert_allclose(changed.stages[-1].components[0].state.mean,
        baseline.stages[-1].components[0].state.mean,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(changed.stages[-1].components[0].state.covariance,
        baseline.stages[-1].components[0].state.covariance,rtol=0.,atol=1e-12)
    assert changed.records['bayes_factor_surprise']['posterior_change_probability']==pytest.approx(
        baseline.records['bayes_factor_surprise']['posterior_change_probability'],abs=1e-12)
    for left,right in zip(changed.time_resolved_body_ratios,baseline.time_resolved_body_ratios):
        np.testing.assert_allclose([branch.log_weight for branch in left['branches']],
            [branch.log_weight for branch in right['branches']],rtol=0.,atol=1e-12)
        for lbranch,rbranch in zip(left['branches'],right['branches']):
            np.testing.assert_allclose([child.log_weight for child in lbranch.posterior_components],
                [child.log_weight for child in rbranch.posterior_components],rtol=0.,atol=1e-12)


def test_parent_evidence_normalization_cannot_truncate_a_parent_or_evidence():
    from probtf_observation_fusion_demo.time_resolved_body_ratios import normalized_parent_evidence
    with pytest.raises(ValueError,match='complete evidence'):
        normalized_parent_evidence([np.log(.17),np.log(.83)],[-1e20]*3,[(-1e20,0.)]*2)
    with pytest.raises(ValueError,match='complete evidence parts'):
        normalized_parent_evidence([np.log(.17),np.log(.83)],[-1e20]*2,[(-1e20,0.)]*3)


def test_intermediate_frame_index_must_match_its_own_actual_ledger_endpoint(batch_problem):
    model,events,ledger,_=batch_problem
    middle=events[2]
    payload=dict(middle.payload,frame_index=2)
    from probtf_observation_fusion_demo.sensor_event_store import _encode
    encoded,blobs,payload_hash=_encode(payload)
    corrupted=replace(middle,payload_json=encoded,blobs=blobs,payload_hash=payload_hash)
    # Its stamp and command still name frame 1; the last frame 3 is valid.
    # The original implementation only checked the final frame index.
    with pytest.raises(ValueError,match='frame_index.*ledger endpoint'):
        model.update_sensor_batch(events[:2]+(corrupted,)+events[3:],ledger[:3],batch_id='wrong-middle-frame')
    assert model.last_result is None and not model._sensor_events


def test_same_frame_sensors_can_cross_batches_and_later_batches_keep_the_exact_source(batch_problem):
    model,events,ledger,_=batch_problem
    results=[]
    groups=((events[:1],()),(events[1:2],()),(events[2:3],ledger[:1]),(events[3:],ledger[1:3]))
    for index,(group,steps) in enumerate(groups):
        previous=tuple(item.state.state_id for item in model.components)
        result=model.update_sensor_batch(group,steps,batch_id='split:'+str(index));results.append(result)
        assert result.records['joint_batch']['retained_input_state_ids']==list(previous)
        assert result.records['joint_batch']['raw_event_ids']==[event.event_id for event in group]
        assert result.prior.stamp==group[-1].stamp and result.prior.command==group[-1].command
        for parent in result.records['joint_batch']['likelihood_factors']:
            for factor in parent:
                event=next(event for event in group if event.event_id==factor['observation_id'])
                assert factor['acquisition_stamp']==event.stamp
                assert factor['acquisition_command']==list(event.command)
    assert results[0].prior.stamp==results[1].prior.stamp==events[0].stamp
    assert results[0].records['valid_observation_count']==0
    assert results[1].records['observation_ids']==['image:first']
    assert results[2].records['observation_ids']==['imu:middle']
    assert model.fixed_goal_memory['observation_ids']==['image:last']
    assert set(model._sensor_events)=={event.event_id for event in events}


def test_cross_batch_order_rejects_unseen_older_event_without_changing_retained_law(batch_problem):
    model,events,ledger,event=batch_problem
    first=model.update_sensor_batch(events[:2],(),batch_id='first-image')
    original_components=model.components;original_memory=model.fixed_goal_memory
    forged=replace(events[1],event_id='image:late-unseen',sequence=events[0].sequence)
    with pytest.raises(ValueError,match='retained batch cursor.*checkpoint replay'):
        model.update_sensor_batch((forged,),(),batch_id='late')
    assert model.last_result is first
    assert model.components is original_components and model.fixed_goal_memory is original_memory
    assert forged.event_id not in model._sensor_events
