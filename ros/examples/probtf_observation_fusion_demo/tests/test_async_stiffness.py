"""Async consumers retain K provenance and never share estimator workspaces."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_control import command_state
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier

from probtf_observation_fusion_demo.async_stiffness import (
    FrozenKPacket, PhysicalActionEngine, capture_stiffness_packet, reanchor_retained_stiffness,
)


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('async_stiffness'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    templates = yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations']
    inference = MovingInference(scene.robot_description, templates)
    result = inference.update(inputs, 0)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    assert identifier.command_proposal(result) is not None
    packet = capture_stiffness_packet(result)
    return scene, inference, result, identifier, settings, packet


def engine_for(observed):
    scene, inference, _, _, settings, packet = observed
    engine = PhysicalActionEngine(scene.urdf, inference.body_ids, settings)
    engine.set(packet)
    return engine


def caught_up_result(observed, engine):
    _, inference, result, _, _, _ = observed
    current = np.array(result.prior.command)+np.array([.003, -.002, .001, .002, -.001, .001])
    action = engine.increment(result.prior.command, current)
    def advance(item):
        future = command_state(item.state, current, inference.body_ids, inference.actions,
                               joint_increment=action['joint_increment'])
        future = replace(future, stamp=1000.3, version=future.version+10)
        return replace(item, state=future)
    stage = replace(result.stages[-1], sensor_ids=(), components=tuple(map(advance,result.stages[-1].components)),
        goal_components=tuple(map(advance,result.stages[-1].goal_components)))
    return replace(result, prior=stage.components[0].state,
        stages=result.stages[:-1]+(stage,), sensor_posteriors=(), local_estimates=(), records=dict(result.records,
            observation_ids=[], prediction_only=True))


def test_packet_has_no_mutable_alias_to_the_estimator_record(observed):
    _, _, result, _, _, packet = observed
    actual = json.dumps(result.records['stiffness'], sort_keys=True)
    copy = packet.record
    copy['source_stamp'] = -1.
    copy['command_proposal']['selected_K'][0] = -1.
    assert packet.record['source_stamp'] == result.prior.stamp
    assert packet.proposal['selected_K'][0] > 0.
    assert json.dumps(result.records['stiffness'], sort_keys=True) == actual
    with pytest.raises(FrozenInstanceError):
        packet.packet_id = 'different'


def test_reanchor_uses_same_K_and_density_but_honest_current_prediction_metadata(observed):
    _, _, result, estimator, _, packet = observed
    estimator_record = estimator.last_record
    estimator_factor_ids = tuple(item.factor_id for item in estimator.factors)
    engine = engine_for(observed)
    current = caught_up_result(observed, engine)
    view, mapping = reanchor_retained_stiffness(engine, current)
    record = view.records['stiffness']
    assert engine.identifier is not estimator
    assert engine.identifier.robot is not estimator.robot
    assert engine.identifier.robot.data is not estimator.robot.data
    assert estimator.last_record is estimator_record
    assert tuple(item.factor_id for item in estimator.factors) == estimator_factor_ids
    assert record['status'] == 'prediction_with_retained_parameters'
    assert not record['update_applied'] and record['posterior'] is None
    assert record['prior'] == record['retained_prior'] == packet.record['posterior']
    assert record['source_stamp'] == 1000.3
    assert record['source_command'] == list(current.prior.command)
    assert record['body_posterior_state_ids'] == [current.stages[-1].components[0].state.state_id]
    assert record['captured_source_stamp'] == result.prior.stamp
    assert record['captured_source_command'] == list(result.prior.command)
    assert record['captured_body_posterior_state_ids'] == [result.stages[-1].components[0].state.state_id]
    assert record['observation_ids'] == []
    assert record['captured_observation_ids'] == result.records['observation_ids']
    assert record['information_rank'] is None
    np.testing.assert_array_equal(mapping.stiffness, packet.proposal['selected_K'])
    np.testing.assert_array_equal(record['command_proposal']['selected_logK'], packet.proposal['selected_logK'])
    assert record['command_proposal']['density_id'] == packet.proposal['density_id']
    assert record['proposal_selection']['density_evaluation_count'] == 0
    assert engine.identifier.factors == ()
    mapping.validate_result(view)
    with pytest.raises(ValueError, match='different|modified'):
        mapping.validate_result(result)
    # The primary body remains the caller's prediction, with every cross block.
    assert view.stages[-1].components[0].state is current.stages[-1].components[0].state
    assert view.stages[-1].goal_components == current.stages[-1].goal_components


def test_reanchored_view_passes_existing_display_and_actual_wire_checks(observed):
    _, inference, _, _, _, _ = observed
    engine = engine_for(observed)
    current = caught_up_result(observed, engine)
    view, mapping = reanchor_retained_stiffness(engine, current)
    snapshot = build_snapshot(view, view.stages[-1], inference.common)
    verify_adapter_identity(snapshot, view.stages[-1])
    marker = MotionDisplay.stiffness_marker(snapshot, view.records['stiffness'])
    assert 'prediction_with_retained_parameters' in marker.text
    assert 'IK correction K' in marker.text
    settings = inference.common['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    assert verify_wire_identity(snapshot,message,seed=settings['seed'],point_size=settings['point_size'])
    assert snapshot.command == tuple(mapping.source_command)


def test_increment_matches_independent_exact_equilibria_and_carries_provenance(observed):
    _, inference, result, estimator, _, packet = observed
    engine = engine_for(observed)
    q0 = np.array(result.prior.command)+.001*np.arange(6)
    q1 = q0+.0003*np.arange(6)[::-1]
    action = engine.increment(q0,q1)
    stiffness = np.array(packet.proposal['selected_K'])
    captured = np.array(packet.proposal['auxiliary_equilibrium'])
    left = estimator._refine_equilibrium(captured,q0,stiffness)[0]
    right,hessian,_,active = estimator._refine_equilibrium(captured,q1,stiffness)
    np.testing.assert_array_equal(action['joint_increment'],right-left)
    rhs = np.diag(stiffness);rhs[active] = 0.
    np.testing.assert_array_equal(action['command_jacobian'],np.linalg.solve(hessian,rhs))
    assert action['captured_source_stamp'] == result.prior.stamp
    assert action['density_id'] == packet.proposal['density_id']
    assert action['captured_observation_ids'] == result.records['observation_ids']
    assert not action['parameter_density_updated'] and not action['physical_truth_received']
    json.dumps(action,allow_nan=False)
    # A fixed anchor makes ordering irrelevant and increments telescope.
    reverse = engine.increment(q1,q0)
    np.testing.assert_array_equal(reverse['joint_increment'],-np.array(action['joint_increment']))
    np.testing.assert_array_equal(engine.increment(q0,q0)['joint_increment'],np.zeros(6))
    assert engine.identifier.body_edge_ids == tuple(inference.body_ids)


def test_two_engines_do_not_share_model_cache_and_packet_tampering_is_rejected(observed):
    _, _, result, _, _, packet = observed
    first,second = engine_for(observed),engine_for(observed)
    assert first.identifier.robot is not second.identifier.robot
    q = np.array(result.prior.command)+.001
    first.anchor(q)
    assert tuple(q) in first._cache and tuple(q) not in second._cache
    before = first.provenance()
    with pytest.raises(ValueError,match='hash'):
        first.set(FrozenKPacket(packet.record_json, 'wrong'))
    assert first.provenance() == before


def test_no_packet_and_foreign_capture_sources_fail_explicitly(observed):
    scene,inference,result,_,settings,_ = observed
    engine = PhysicalActionEngine(scene.urdf,inference.body_ids,settings)
    with pytest.raises(ValueError,match='captured K packet'):
        engine.increment(result.prior.command,result.prior.command)
    with pytest.raises(ValueError,match='captured K packet'):
        reanchor_retained_stiffness(engine,result)
    record = dict(result.records['stiffness'],source_stamp=result.prior.stamp+1.)
    foreign = replace(result,records=dict(result.records,stiffness=record))
    with pytest.raises(ValueError,match='provenance'):
        capture_stiffness_packet(foreign)


def test_reanchor_and_prediction_do_not_use_MC_or_reselect_stiffness(observed,monkeypatch):
    engine = engine_for(observed)
    def forbidden(*args,**kwargs):
        raise AssertionError('A retained K action must neither sample nor update its parameter density.')
    monkeypatch.setattr(engine.identifier,'command_proposal',forbidden)
    monkeypatch.setattr(engine.identifier,'update',forbidden)
    monkeypatch.setattr(engine.identifier,'log_density',forbidden)
    for name in ('default_rng','normal','random','randn','multivariate_normal','RandomState'):
        monkeypatch.setattr(np.random,name,forbidden)
    current = caught_up_result(observed,engine)
    view,mapping = reanchor_retained_stiffness(engine,current)
    assert mapping.record()['density_id'] == engine.packet.proposal['density_id']
    assert not view.records['stiffness']['update_applied']


def test_reanchored_map_acts_on_all_current_body_and_goal_cross_covariances(observed):
    _,inference,_,_,_,packet = observed
    engine = engine_for(observed)
    current = caught_up_result(observed,engine)
    view,mapping = reanchor_retained_stiffness(engine,current)
    query = np.array(view.prior.command)+.0002*np.arange(6)
    action = engine.increment(mapping.source_command,query)
    delta = np.array(action['joint_increment'])
    for source in view.stages[-1].components+view.stages[-1].goal_components:
        actual = mapping.push(source.state,query,inference.body_ids,inference.actions)
        expected = command_state(source.state,query,inference.body_ids,inference.actions,
                                 joint_increment=delta)
        assert actual.state_id == expected.state_id
        np.testing.assert_array_equal(actual.covariance,expected.covariance)
        marker = actual.edge_ids.index('marker')
        np.testing.assert_array_equal(actual.transforms[marker].translation,
                                      source.state.transforms[marker].translation)
    assert packet.proposal['selected_K'] == mapping.record()['selected_K']
    # Catch-up already cached the current anchor; an honest record reports
    # zero new anchor solves rather than inventing one per view publication.
    assert view.records['stiffness']['proposal_selection']['current_equilibrium_evaluation_count'] == 0


def test_IK_and_history_share_exact_captured_anchor_theta_increment_and_G(observed):
    _,inference,_,_,_,_ = observed
    history_engine = engine_for(observed)
    control_engine = engine_for(observed)
    current = caught_up_result(observed,history_engine)
    view,mapping = reanchor_retained_stiffness(control_engine,current)
    commands = [np.array(view.prior.command)+scale*np.arange(6)
                for scale in (.003,-.002,.001)]
    # Opposite evaluation orders must not change a branch or a numerical bit.
    history = {tuple(q):history_engine.increment(view.prior.command,q) for q in commands}
    for q in reversed(commands):
        actual = mapping.selection_record(q)
        expected = history[tuple(q)]
        np.testing.assert_array_equal(actual['auxiliary_equilibrium'],expected['query_equilibrium'])
        np.testing.assert_array_equal(actual['joint_increment'],expected['joint_increment'])
        np.testing.assert_array_equal(actual['command_jacobian'],expected['command_jacobian'])
        np.testing.assert_array_equal(mapping.jacobian(q),expected['command_jacobian'])
        for source in view.stages[-1].components+view.stages[-1].goal_components:
            predicted = command_state(source.state,q,inference.body_ids,inference.actions,
                                      joint_increment=expected['joint_increment'])
            selected = mapping.push(source.state,q,inference.body_ids,inference.actions)
            assert predicted.state_id == selected.state_id
            np.testing.assert_array_equal(predicted.covariance,selected.covariance)
    assert mapping.record()['branch_policy'] == control_engine.provenance()['branch_policy']
    assert mapping.record()['captured_parameter_packet']['packet_id'] == history_engine.packet.packet_id


def test_new_K_epoch_rejects_old_map_and_reanchors_to_the_new_completed_source(observed):
    scene,_,_,_,settings,_ = observed
    engine = engine_for(observed)
    old_current = caught_up_result(observed,engine)
    _,old_mapping = reanchor_retained_stiffness(engine,old_current)
    old_K = old_mapping.stiffness.copy()
    frame = scene.step(.05,1000.05)
    templates = yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations']
    inference = MovingInference(scene.robot_description,templates)
    result = inference.update(scene.sensor_inputs(frame),1)
    estimator = StiffnessIdentifier(scene.urdf,inference.body_ids,settings)
    result.records['stiffness'] = estimator.update(result)
    assert estimator.command_proposal(result) is not None
    packet = capture_stiffness_packet(result)
    engine.set(packet)
    with pytest.raises(ValueError,match='epoch'):
        old_mapping.jacobian(old_mapping.source_command)
    np.testing.assert_array_equal(old_mapping.stiffness,old_K)
    new_context = (scene,inference,result,estimator,settings,packet)
    current = caught_up_result(new_context,engine)
    view,mapping = reanchor_retained_stiffness(engine,current)
    action = engine.increment(view.prior.command,np.array(view.prior.command)+.001)
    selected = mapping.selection_record(np.array(view.prior.command)+.001)
    np.testing.assert_array_equal(action['joint_increment'],selected['joint_increment'])
    np.testing.assert_array_equal(action['command_jacobian'],selected['command_jacobian'])
    np.testing.assert_array_equal(mapping.stiffness,packet.proposal['selected_K'])
    assert view.records['stiffness']['captured_source_stamp'] == 1000.05
    assert mapping.record()['density_id'] == packet.proposal['density_id']
