"""Direct K_est actions freeze the original estimate and current branch."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import pickle

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath
from probtf_observation_fusion_demo.async_stiffness import (
    FrozenActionKPacket, PhysicalActionEngine, capture_direct_stiffness_action,
    capture_stiffness_packet, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.moving_control import MovingController
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity

from test_async_stiffness import observed, engine_for, caught_up_result


def _alternative_representative(observed):
    """An explicitly evaluated alternative of this fixture's same density.

    This test input is a numerical representative, not an extra observation
    or a claim of density improvement. No inference state is overwritten.
    """
    _, _, result, estimator, _, raw = observed
    record = raw.record
    proposal = record['command_proposal']
    x = np.array(proposal['selected_logK'])+np.array([.2, -.15, .12, -.1, .15, -.18])
    stiffness = np.exp(x)
    density, score = estimator.log_density(x)
    engine = engine_for(observed)
    command = np.asarray(result.prior.command)
    theta, _, _, active = engine.identifier._refine_equilibrium(
        np.asarray(proposal['auxiliary_equilibrium']), command, stiffness)
    proposal.update(selected_K=stiffness.tolist(), selected_logK=x.tolist(),
        selected_log_unnormalized_density=density,
        base_candidate_log_unnormalized_density=density,
        auxiliary_equilibrium=theta.tolist(),
        auxiliary_active_joint_limits=active.tolist(),
        command=(2.*command-theta).tolist(), density_improvement_claimed=False,
        method='test input: separately evaluated continuous-density representative')
    record['proposal_selection'].update(base_logK=x.tolist(),
        base_log_unnormalized_density=density, selected_log_unnormalized_density=density,
        base_density_score=score.tolist(), selected_density_score=score.tolist(),
        score_step=np.zeros_like(x).tolist(), density_improvement_claimed=False,
        convergence_claimed=False)
    return capture_stiffness_packet(replace(result,
        records=dict(result.records, stiffness=record)))


@pytest.fixture(scope='module')
def direct_context(observed):
    raw = observed[-1]
    engine = engine_for(observed)
    first = capture_direct_stiffness_action(engine, observed[2], raw)
    engine.set(first)
    current = caught_up_result(observed, engine)
    target = _alternative_representative(observed)
    second = capture_direct_stiffness_action(engine, current, target, previous=first)
    return current, target, first, second


def _engine(observed, packet):
    engine = PhysicalActionEngine(observed[0].urdf, observed[1].body_ids, observed[4])
    engine.set(packet)
    return engine


def _tamper(packet, action):
    encoded = json.dumps(action, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(('direct-action\n'+packet.estimated_packet.packet_id+'\n'+encoded).encode()).hexdigest()
    return replace(packet, action_json=encoded, packet_id=digest)


def test_direct_packet_keeps_K_bits_and_original_density_proposal_without_aliases(observed, direct_context):
    current, raw, first, packet = direct_context
    source_before = json.dumps(current.records['stiffness'], sort_keys=True)
    assert isinstance(packet, FrozenActionKPacket)
    assert packet.estimated_packet is raw
    assert packet.record == raw.record and packet.proposal == raw.proposal
    assert pickle.loads(pickle.dumps(packet)) == packet
    np.testing.assert_array_equal(packet.action['selected_K'], raw.proposal['selected_K'])
    assert np.array(packet.action['selected_K']).tobytes() == np.array(raw.proposal['selected_K']).tobytes()
    assert packet.action['selected_K'] != first.action['selected_K']
    assert not any(key.startswith('execution_') for key in packet.action)
    for copy in (packet.record, packet.proposal, packet.action):
        if 'selected_K' in copy:
            copy['selected_K'][0] = -1.
        else:
            copy['command_proposal']['selected_K'][0] = -1.
    assert packet.record == raw.record and packet.proposal == raw.proposal
    assert all(x > 0. for x in packet.action['selected_K'])
    assert json.dumps(current.records['stiffness'], sort_keys=True) == source_before
    with pytest.raises(FrozenInstanceError):
        packet.action_json = '{}'


def test_anchor_continues_previous_stable_branch_then_solves_exact_new_K(observed, direct_context):
    current, raw, first, packet = direct_context
    previous_engine = _engine(observed, first)
    command = np.array(current.prior.command)
    initial = previous_engine.anchor(command)['equilibrium']
    expected, _, _, active = previous_engine.identifier._refine_equilibrium(
        np.array(initial), command, np.array(raw.proposal['selected_K']))
    action = packet.action
    np.testing.assert_array_equal(action['branch_initial_equilibrium'], initial)
    np.testing.assert_array_equal(action['branch_initial_K'], first.action['selected_K'])
    np.testing.assert_array_equal(action['auxiliary_equilibrium'], expected)
    np.testing.assert_array_equal(action['auxiliary_active_joint_limits'], active)
    assert action['previous_action_packet_id'] == first.packet_id
    assert action['branch_initial_packet_id'] == first.packet_id
    assert action['anchor_command'] == list(current.prior.command)
    assert action['anchor_stamp'] == current.prior.stamp
    assert action['anchor_prior_state_id'] == current.prior.state_id
    assert action['anchor_body_state_ids'] == [x.state.state_id for x in current.stages[-1].components]
    assert action['anchor_goal_state_ids'] == [x.state.state_id for x in current.stages[-1].goal_components]
    assert action['estimated_source_stamp'] == raw.record['source_stamp']
    engine = _engine(observed, packet)
    np.testing.assert_array_equal(engine.stiffness, raw.proposal['selected_K'])
    residual = engine.identifier.robot.tau_gravity(expected)+engine.identifier.spring.torque(
        expected, command, engine.stiffness)
    residual[active] = 0.
    assert np.max(np.abs(residual)) <= 1e-9
    np.testing.assert_array_equal(engine.increment(command, command)['joint_increment'], np.zeros(6))


def test_each_trial_starts_from_the_same_direct_anchor_and_is_order_independent(observed, direct_context, monkeypatch):
    current, _, _, packet = direct_context
    left, right = _engine(observed, packet), _engine(observed, packet)
    starts = []
    original = left.identifier._refine_equilibrium
    def recorded(initial, command, stiffness):
        starts.append(np.array(initial, copy=True))
        return original(initial, command, stiffness)
    monkeypatch.setattr(left.identifier, '_refine_equilibrium', recorded)
    commands = [np.array(current.prior.command)+scale*np.arange(6) for scale in (.002, -.001, .003)]
    expected = {tuple(q): right.increment(current.prior.command, q) for q in reversed(commands)}
    for q in commands:
        actual = left.increment(current.prior.command, q)
        np.testing.assert_array_equal(actual['joint_increment'], expected[tuple(q)]['joint_increment'])
        np.testing.assert_array_equal(actual['command_jacobian'], expected[tuple(q)]['command_jacobian'])
        assert actual['action_packet_id'] == packet.packet_id
        assert actual['estimated_packet_id'] == packet.estimated_packet.packet_id
        assert 'execution_stiffness' not in actual
    assert len(starts) == len(commands)
    for initial in starts:
        np.testing.assert_array_equal(initial, packet.action['auxiliary_equilibrium'])


def test_reanchor_preserves_raw_proposal_and_map_uses_the_same_current_action_proposal(observed, direct_context):
    current, raw, _, packet = direct_context
    before = deepcopy(current.records)
    estimator_record = observed[3].last_record
    engine = _engine(observed, packet)
    view, mapping = reanchor_retained_stiffness(engine, current)
    record = view.records['stiffness']
    assert record['command_proposal'] == raw.proposal
    assert record['action_command_proposal'] is mapping._source_proposal
    assert record['action_command_proposal']['source_command'] == list(current.prior.command)
    assert record['action_command_proposal']['source_stamp'] == current.prior.stamp
    assert record['action_command_proposal']['selected_K'] == raw.proposal['selected_K']
    assert record['action_stiffness'] == packet.action
    assert record['captured_estimate_packet_id'] == raw.packet_id
    assert record['captured_packet_id'] == packet.packet_id
    assert 'execution_stiffness' not in record and 'execution_command_proposal' not in record
    assert record['posterior'] is None and not record['update_applied']
    assert record['prior'] == record['retained_prior'] == raw.record['posterior']
    assert record['proposal_selection']['density_evaluation_count'] == 0
    assert view.prior is current.prior and view.stages is current.stages
    assert current.records == before and observed[3].last_record is estimator_record
    mapping.validate_result(view)
    with pytest.raises(ValueError, match='action parameter view'):
        capture_stiffness_packet(view)


def test_IK_ledger_and_actual_display_share_the_direct_action_and_all_cross_covariances(observed, direct_context):
    current, raw, _, packet = direct_context
    scene, inference = observed[:2]
    control_engine, history_engine = _engine(observed, packet), _engine(observed, packet)
    view, mapping = reanchor_retained_stiffness(control_engine, current)
    control = inference.settings['control']
    controller = MovingController(inference, scene.robot_description,
        control['goal_offset'], control['goal_orientation_wxyz'], control)
    command, diagnostics = controller.step(view, .25, command_map=mapping)
    selection = mapping.selection_record(command)
    action = history_engine.increment(view.prior.command, command)
    assert diagnostics['selected_command_action'] == selection
    assert diagnostics['gradient_evaluation_count'] == 1
    np.testing.assert_array_equal(selection['joint_increment'], action['joint_increment'])
    assert action['selected_K'] == raw.proposal['selected_K']
    step = ActionStep(index=0, start_stamp=view.prior.stamp, end_stamp=view.prior.stamp+.25,
        start_command=view.prior.command, end_command=command,
        joint_increment=action['joint_increment'], density_id=action['density_id'],
        selected_K=action['selected_K'], anchor_command=action['numerical_branch_anchor_command'],
        anchor_equilibrium=action['numerical_branch_anchor'],
        source_state_ids=tuple(action['captured_body_posterior_state_ids']),
        source_stamp=action['captured_source_stamp'], provenance=action)
    path = RecordedCommandPath(view, (step,), body_ids=inference.body_ids, actions=inference.actions)
    for item in view.stages[-1].components+view.stages[-1].goal_components:
        selected = mapping.push(item.state, command, inference.body_ids, inference.actions)
        advanced = path.push(item.state, command, inference.body_ids, inference.actions)
        assert selected.state_id == advanced.state_id
        np.testing.assert_array_equal(selected.covariance, advanced.covariance)
        np.testing.assert_array_equal(selected.mean, advanced.mean)
    snapshot = build_snapshot(view, view.stages[-1], inference.common)
    verify_adapter_identity(snapshot, view.stages[-1])
    display = inference.common['display']
    message = make_markers(snapshot, display['seed'], display['point_size'])
    buffer = io.BytesIO(); message.serialize(buffer)
    actual_wire = MarkerArray().deserialize(buffer.getvalue())
    assert verify_wire_identity(snapshot, actual_wire, seed=display['seed'], point_size=display['point_size'])


def test_new_direct_epoch_cannot_rewrite_an_old_ledger_or_use_an_old_map(observed, direct_context):
    current, _, first, second = direct_context
    inference = observed[1]
    engine = _engine(observed, first)
    view, old_map = reanchor_retained_stiffness(engine, observed[2])
    command = np.array(view.prior.command)+.001*np.arange(6)
    action = engine.increment(view.prior.command, command)
    step = ActionStep(index=0, start_stamp=view.prior.stamp, end_stamp=view.prior.stamp+.1,
        start_command=view.prior.command, end_command=command,
        joint_increment=action['joint_increment'], density_id=action['density_id'],
        selected_K=action['selected_K'], source_stamp=action['captured_source_stamp'], provenance=action)
    path = RecordedCommandPath(view, (step,), body_ids=inference.body_ids, actions=inference.actions)
    before = deepcopy(step.record())
    source = view.stages[-1].components[0].state
    expected = path.push(source, command, inference.body_ids, inference.actions)
    engine.set(second)
    assert step.record() == before
    actual = path.push(source, command, inference.body_ids, inference.actions)
    assert actual.state_id == expected.state_id
    assert step.record()['selected_K'] == first.action['selected_K']
    with pytest.raises(ValueError, match='epoch'):
        old_map.jacobian(command)
    new_view, new_map = reanchor_retained_stiffness(engine, current)
    np.testing.assert_array_equal(new_map.stiffness, second.proposal['selected_K'])
    assert new_view.records['stiffness']['action_stiffness'] == second.action


@pytest.mark.parametrize('tamper', ['hash', 'raw_hash', 'K', 'anchor', 'active', 'past_anchor'])
def test_direct_packet_tampering_cannot_change_the_accepted_engine(observed, direct_context, tamper):
    _, _, first, second = direct_context
    engine = _engine(observed, first)
    before = engine.provenance()
    if tamper == 'hash':
        bad = replace(second, packet_id='wrong')
    elif tamper == 'raw_hash':
        bad = replace(second, estimated_packet=replace(second.estimated_packet,
            record_json=second.estimated_packet.record_json+' '))
    else:
        action = second.action
        if tamper == 'K':
            action['selected_K'][0] *= 1.01
        elif tamper == 'anchor':
            action['auxiliary_equilibrium'][1] += .1
        elif tamper == 'active':
            action['auxiliary_active_joint_limits'][0] = not action['auxiliary_active_joint_limits'][0]
        else:
            action['anchor_stamp'] = second.record['source_stamp']-.1
        bad = _tamper(second, action)
    with pytest.raises(ValueError):
        engine.set(bad)
    assert engine.provenance() == before


def test_capture_and_reanchor_do_not_evaluate_update_or_sample_a_parameter_density(observed, direct_context, monkeypatch):
    current, raw, first, second = direct_context
    engine = _engine(observed, first)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('A direct K action must not update, select or sample the density.')
    for method in ('update', 'command_proposal', 'log_density'):
        monkeypatch.setattr(engine.identifier, method, forbidden)
    for method in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, method, forbidden)
    packet = capture_direct_stiffness_action(engine, current, raw, previous=first)
    assert packet == second
    view, mapping = reanchor_retained_stiffness(engine, current, packet)
    assert mapping.record()['selected_K'] == raw.proposal['selected_K']
    assert engine.identifier.factors == ()
    assert view.records['stiffness']['proposal_selection']['density_evaluation_count'] == 0


def test_direct_view_rejects_a_different_body_at_the_same_capture_time(observed, direct_context):
    current, _, _, packet = direct_context
    engine = _engine(observed, packet)
    changed = replace(current, prior=replace(current.prior, component_id='different-current-body'))
    with pytest.raises(ValueError, match='different current body source'):
        reanchor_retained_stiffness(engine, changed)
