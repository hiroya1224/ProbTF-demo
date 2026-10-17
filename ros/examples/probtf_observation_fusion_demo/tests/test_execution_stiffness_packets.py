"""Execution K is a frozen control sidecar of the unchanged estimated law."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath
from probtf_observation_fusion_demo.async_stiffness import (
    FrozenExecutionKPacket, PhysicalActionEngine, capture_stiffness_packet,
    execution_stiffness_packet, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.execution_stiffness import LogStiffnessDamper
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity

from test_async_stiffness import observed, engine_for, caught_up_result


DAMPING = {'kp_exec_tau': .5, 'max_log_kp_exec_step': .05}


def evaluated_alternative_estimate(observed):
    """A different evaluated representative of the actual fixture density.

    This is a numerical test candidate, not a new observation or a claim to
    have found its density's mode. Its full density/score, K, equilibrium,
    active set and compensation command are evaluated consistently.
    """
    _, _, result, estimator, _, raw = observed
    record = raw.record
    proposal = record['command_proposal']
    x = np.asarray(proposal['selected_logK'])+np.array([.6, -.4, .5, -.3, .4, -.5])
    stiffness = np.exp(x)
    density, score = estimator.log_density(x)
    engine = engine_for(observed)
    command = np.asarray(result.prior.command)
    equilibrium, _, _, active = engine.identifier._refine_equilibrium(
        np.asarray(proposal['auxiliary_equilibrium']), command, stiffness)
    proposal.update(selected_K=stiffness.tolist(), selected_logK=x.tolist(),
        selected_log_unnormalized_density=density,
        base_candidate_log_unnormalized_density=density,
        auxiliary_equilibrium=equilibrium.tolist(),
        auxiliary_active_joint_limits=active.tolist(),
        command=(2.*command-equilibrium).tolist(), density_improvement_claimed=False,
        method='test fixture: explicitly evaluated alternative continuous-density representative')
    selection = record['proposal_selection']
    selection.update(base_logK=x.tolist(), base_log_unnormalized_density=density,
        selected_log_unnormalized_density=density, base_density_score=score.tolist(),
        selected_density_score=score.tolist(), score_step=np.zeros_like(x).tolist(),
        density_improvement_claimed=False, convergence_claimed=False)
    alternative = replace(result, records=dict(result.records, stiffness=record))
    return capture_stiffness_packet(alternative)


@pytest.fixture(scope='module')
def execution_context(observed):
    _, _, result, _, _, raw = observed
    original = json.dumps(result.records['stiffness'], sort_keys=True)
    engine = engine_for(observed)
    damper = LogStiffnessDamper(DAMPING)
    first_sample = damper.advance(raw.proposal['selected_K'], result.prior.stamp)
    first = execution_stiffness_packet(engine, result, raw, first_sample)
    target = evaluated_alternative_estimate(observed)
    current = caught_up_result(observed, engine)
    samples = [damper.advance(target.proposal['selected_K'], result.prior.stamp+.05*i)
               for i in range(1, 7)]
    second = execution_stiffness_packet(engine, current, target, samples[-1], previous=first)
    assert json.dumps(result.records['stiffness'], sort_keys=True) == original
    assert not np.array_equal(second.execution['execution_K'], target.proposal['selected_K'])
    return current, target, first, second, samples, damper


def execution_engine(observed, packet):
    scene, inference, _, _, settings, _ = observed
    engine = PhysicalActionEngine(scene.urdf, inference.body_ids, settings)
    engine.set(packet)
    return engine


def encoded_packet(packet, record):
    """Rehash a semantic tamper so validation must inspect its actual values."""
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256((packet.estimated_packet.packet_id+'\n'+encoded).encode()).hexdigest()
    return replace(packet, execution_json=encoded, packet_id=digest)


def test_filter_samples_advance_between_IK_requests_but_a_packet_remains_frozen(observed, execution_context):
    current, target, first, second, samples, damper = execution_context
    np.testing.assert_array_equal(first.execution['execution_K'], observed[-1].proposal['selected_K'])
    assert first.execution['initialized_this_step']
    assert second.record == target.record
    assert second.proposal == target.proposal
    assert second.execution['previous_execution_packet_id'] == first.packet_id
    for index, sample in enumerate(samples):
        assert sample['actual_dt'] == pytest.approx(.05)
        assert np.max(np.abs(sample['applied_delta_logK'])) <= .05
        if index:
            assert not np.array_equal(sample['execution_K'], samples[index-1]['execution_K'])
    assert second.execution['filter_sample']['stamp'] == current.prior.stamp
    assert second.execution['filter_sample']['execution_K'] == samples[-1]['execution_K']
    before = second.execution_json
    later = damper.advance(target.proposal['selected_K'], current.prior.stamp+.05)
    assert later['execution_K'] != second.execution['execution_K']
    assert second.execution_json == before
    assert second.execution['timebase'].startswith('physical simulated time')
    assert 'every coordinator motion tick' in second.execution['timebase']


def test_execution_packet_preserves_original_density_and_has_no_mutable_aliases(observed, execution_context):
    _, target, _, packet, _, _ = execution_context
    _, _, result, estimator, _, raw = observed
    original = json.dumps(result.records['stiffness'], sort_keys=True)
    factors = tuple(item.factor_id for item in estimator.factors)
    assert isinstance(packet, FrozenExecutionKPacket)
    assert packet.estimated_packet is target
    assert packet.record == target.record
    assert packet.proposal == target.proposal
    estimate, proposal, execution = packet.record, packet.proposal, packet.execution
    estimate['posterior']['factor_ids'].clear()
    proposal['selected_K'][0] = -1.
    execution['execution_K'][0] = -1.
    execution['filter_sample']['execution_K'][0] = -2.
    assert packet.record == target.record
    assert all(value > 0. for value in packet.execution['execution_K'])
    assert packet.execution['filter_sample']['execution_K'] == packet.execution['execution_K']
    with pytest.raises(FrozenInstanceError):
        packet.packet_id = 'modified'
    with pytest.raises(FrozenInstanceError):
        packet.estimated_packet = raw
    assert json.dumps(result.records['stiffness'], sort_keys=True) == original
    assert tuple(item.factor_id for item in estimator.factors) == factors
    assert packet.execution['parameter_density_updated'] is False
    assert packet.execution['body_reconstructed_from_stiffness'] is False
    assert packet.execution['physical_truth_received'] is False
    assert packet.execution['monte_carlo_used'] is False


def test_damped_equilibrium_and_action_use_execution_K_with_original_estimate_provenance(observed, execution_context):
    current, target, _, packet, _, _ = execution_context
    engine = execution_engine(observed, packet)
    execution = packet.execution
    np.testing.assert_array_equal(engine.stiffness, execution['execution_K'])
    np.testing.assert_array_equal(engine.captured_command, current.prior.command)
    np.testing.assert_array_equal(engine.captured_equilibrium, execution['auxiliary_equilibrium'])
    assert execution['anchor_prior_state_id'] == current.prior.state_id
    assert execution['anchor_body_state_ids'] == [item.state.state_id for item in current.stages[-1].components]
    assert execution['estimated_source_stamp'] == target.record['source_stamp']
    assert execution['estimated_source_command'] == target.record['source_command']
    equilibrium = engine.captured_equilibrium
    residual = engine.identifier.robot.tau_gravity(equilibrium)+engine.identifier.spring.torque(
        equilibrium, engine.captured_command, engine.stiffness)
    residual[np.asarray(execution['auxiliary_active_joint_limits'])] = 0.
    assert np.max(np.abs(residual)) <= 1e-9
    provenance = engine.provenance()
    assert provenance['estimated_K'] == target.proposal['selected_K']
    assert provenance['selected_K'] == execution['execution_K']
    assert provenance['estimated_packet_id'] == target.packet_id
    assert provenance['execution_packet_id'] == packet.packet_id
    assert provenance['execution_stiffness'] == execution
    with pytest.raises(ValueError):
        engine.stiffness[0] = 1.
    identity = engine.increment(current.prior.command, current.prior.command)
    np.testing.assert_array_equal(identity['joint_increment'], np.zeros(6))


def test_reanchor_keeps_estimated_proposal_verbatim_and_separates_execution_proposal(observed, execution_context):
    current, target, _, packet, _, _ = execution_context
    estimator = observed[3]
    estimator_record = estimator.last_record
    original = deepcopy(current.records['stiffness'])
    engine = execution_engine(observed, packet)
    view, mapping = reanchor_retained_stiffness(engine, current)
    record = view.records['stiffness']
    assert record['command_proposal'] == target.proposal
    assert record['execution_command_proposal'] != target.proposal
    assert record['execution_command_proposal']['selected_K'] == packet.execution['execution_K']
    assert record['execution_command_proposal']['selected_logK'] == packet.execution['execution_logK']
    assert 'selected_log_unnormalized_density' not in record['execution_command_proposal']
    assert record['execution_command_proposal']['density_improvement_claimed'] is False
    assert record['prior'] == record['retained_prior'] == target.record['posterior']
    assert record['posterior'] is None and record['update_applied'] is False
    assert record['proposal_selection']['density_evaluation_count'] == 0
    assert record['captured_estimate_packet_id'] == target.packet_id
    assert record['captured_packet_id'] == packet.packet_id
    assert record['execution_stiffness'] == packet.execution
    assert current.records['stiffness'] == original
    assert estimator.last_record is estimator_record
    assert view.prior is current.prior
    assert view.stages is current.stages
    for item in view.stages[-1].components+view.stages[-1].goal_components:
        assert mapping.push(item.state, view.prior.command, observed[1].body_ids,
                            observed[1].actions) is item.state
    mapping.validate_result(view)
    with pytest.raises(ValueError, match='execution parameter view'):
        capture_stiffness_packet(view)


def test_IK_and_actual_action_ledger_use_same_execution_epoch_and_full_joint_law(observed, execution_context):
    current, _, _, packet, _, _ = execution_context
    scene, inference, *_ = observed
    control_engine = execution_engine(observed, packet)
    history_engine = execution_engine(observed, packet)
    view, mapping = reanchor_retained_stiffness(control_engine, current)
    settings = inference.settings['control']
    controller = MovingController(inference, scene.robot_description,
        settings['goal_offset'], settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])
    command, diagnostics = controller.step(view, .25, command_map=mapping)
    selected = mapping.selection_record(command)
    assert diagnostics['gradient_evaluation_count'] == 1
    assert diagnostics['selected_command_action'] == selected
    action = history_engine.increment(view.prior.command, command)
    np.testing.assert_array_equal(selected['joint_increment'], action['joint_increment'])
    np.testing.assert_array_equal(selected['command_jacobian'], action['command_jacobian'])
    assert action['selected_K'] == packet.execution['execution_K']
    assert action['execution_packet_id'] == packet.packet_id
    step = ActionStep(index=1, start_stamp=view.prior.stamp,
        end_stamp=view.prior.stamp+.25, start_command=view.prior.command,
        end_command=command, joint_increment=action['joint_increment'],
        density_id=action['density_id'], selected_K=action['selected_K'],
        anchor_command=action['numerical_branch_anchor_command'],
        anchor_equilibrium=action['numerical_branch_anchor'],
        source_state_ids=tuple(action['captured_body_posterior_state_ids']),
        source_stamp=action['captured_source_stamp'], provenance=action)
    path = RecordedCommandPath(view, (step,), body_ids=inference.body_ids,
                               actions=inference.actions)
    assert path.record()['steps'][0]['provenance']['execution_packet_id'] == packet.packet_id
    for source in view.stages[-1].components+view.stages[-1].goal_components:
        selected_state = mapping.push(source.state, command, inference.body_ids, inference.actions)
        actual_state = path.push(source.state, command, inference.body_ids, inference.actions)
        expected = command_state(source.state, command, inference.body_ids, inference.actions,
                                 joint_increment=action['joint_increment'])
        assert selected_state.state_id == actual_state.state_id == expected.state_id
        np.testing.assert_array_equal(actual_state.covariance, selected_state.covariance)
        np.testing.assert_array_equal(actual_state.mean, selected_state.mean)
        marker = source.state.edge_ids.index('marker')
        np.testing.assert_array_equal(actual_state.transforms[marker].translation,
                                      source.state.transforms[marker].translation)


def test_actual_serialized_display_uses_same_law_and_publishes_both_K_values(observed, execution_context):
    current, target, _, packet, _, _ = execution_context
    inference = observed[1]
    engine = execution_engine(observed, packet)
    view, mapping = reanchor_retained_stiffness(engine, current)
    snapshot = build_snapshot(view, view.stages[-1], inference.common)
    verify_adapter_identity(snapshot, view.stages[-1])
    record = view.records['stiffness']
    hud = MotionDisplay.stiffness_marker(snapshot, record)
    values = lambda vector: ', '.join('{:.4g}'.format(value) for value in vector)
    assert 'Estimated K [Nm/rad]: '+values(target.proposal['selected_K']) in hud.text
    assert 'Execution IK K [Nm/rad]: '+values(packet.execution['execution_K']) in hud.text
    assert 'cap=0.05/motion tick' in hud.text
    display = inference.common['display']
    message = make_markers(snapshot, display['seed'], display['point_size'])
    buffer = io.BytesIO(); message.serialize(buffer)
    actual = MarkerArray().deserialize(buffer.getvalue())
    assert verify_wire_identity(snapshot, actual, seed=display['seed'], point_size=display['point_size'])
    # The scene HUD is published separately from the candidate-law array.
    hud_buffer = io.BytesIO(); MarkerArray([hud]).serialize(hud_buffer)
    actual_hud = MarkerArray().deserialize(hud_buffer.getvalue())
    assert actual_hud.markers[0].text == hud.text
    assert actual_hud.markers[0].ns == 'estimated_stiffness'
    np.testing.assert_array_equal(actual_hud.markers[0].scale.__getstate__(),
                                  hud.scale.__getstate__())
    assert snapshot.command == tuple(mapping.source_command)
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'hand'} == {
        item.state.state_id for item in view.stages[-1].components}
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'switch_goal'} == {
        item.state.state_id for item in view.stages[-1].goal_components}
    mismatched = deepcopy(record)
    mismatched['execution_stiffness']['execution_K'][0] *= 1.01
    with pytest.raises(ValueError, match='estimated/execution K'):
        MotionDisplay.stiffness_marker(snapshot, mismatched)


@pytest.mark.parametrize('tamper', ['packet_hash', 'raw_hash', 'execution_value', 'sample_value',
                                  'estimated_value', 'anchor'])
def test_tampered_packets_fail_before_changing_accepted_engine(observed, execution_context, tamper):
    _, _, first, packet, _, _ = execution_context
    engine = execution_engine(observed, first)
    before = engine.provenance()
    if tamper == 'packet_hash':
        bad = replace(packet, packet_id='modified')
    elif tamper == 'raw_hash':
        raw = replace(packet.estimated_packet, record_json=packet.estimated_packet.record_json+' ')
        bad = replace(packet, estimated_packet=raw)
    else:
        record = packet.execution
        if tamper == 'execution_value':
            record['execution_K'][0] *= 1.01
            record['execution_logK'][0] = float(np.log(record['execution_K'][0]))
        elif tamper == 'sample_value':
            record['filter_sample']['execution_K'][0] *= 1.01
        elif tamper == 'estimated_value':
            record['estimated_K'][0] *= 1.01
        else:
            record['auxiliary_equilibrium'][1] += .1
        bad = encoded_packet(packet, record)
    with pytest.raises(ValueError):
        engine.set(bad)
    assert engine.provenance() == before


@pytest.mark.parametrize('tamper', ['time', 'target', 'sample_hash'])
def test_IK_capture_cannot_retime_or_retarget_a_filter_sample(observed, execution_context, tamper):
    current, target, first, packet, samples, _ = execution_context
    sample = deepcopy(samples[-1])
    if tamper == 'time':
        sample['stamp'] += .01
    elif tamper == 'target':
        sample['estimated_K'][0] *= 1.01
    else:
        sample = deepcopy(packet.execution['filter_sample'])
        sample['execution_K'][0] *= 1.01
    with pytest.raises(ValueError, match='sample|filter'):
        execution_stiffness_packet(engine_for(observed), current, target, sample, previous=first)


def test_new_execution_epoch_rejects_old_map_and_never_updates_or_samples_a_density(observed, execution_context, monkeypatch):
    current, target, first, second, samples, _ = execution_context
    engine = execution_engine(observed, first)
    _, old_map = reanchor_retained_stiffness(engine, current)
    def forbidden(*args, **kwargs):
        raise AssertionError('Execution interpolation cannot update, select or sample a parameter density.')
    for name in ('update', 'command_proposal', 'log_density'):
        monkeypatch.setattr(engine.identifier, name, forbidden)
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    frozen = execution_stiffness_packet(engine, current, target, samples[-1], previous=first)
    assert frozen.packet_id == second.packet_id
    view, mapping = reanchor_retained_stiffness(engine, current, frozen)
    with pytest.raises(ValueError, match='epoch'):
        old_map.jacobian(current.prior.command)
    query = np.asarray(view.prior.command)+.0002*np.arange(6)
    selection = mapping.selection_record(query)
    assert mapping.record()['selected_K'] == frozen.execution['execution_K']
    np.testing.assert_array_equal(selection['joint_increment'],
                                  engine.increment(view.prior.command, query)['joint_increment'])
    assert engine.identifier.factors == ()
    assert view.records['stiffness']['proposal_selection']['density_evaluation_count'] == 0
