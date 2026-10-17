"""Running gradients evaluate the complete frozen relative law, not an endpoint.

The density oracle independently uses SciPy's normalized Gaussian and the
core Bingham normalizer. Synthetic mixtures isolate otherwise easily hidden
weight, normalizer and physical-position-bound effects.
"""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from probtf_observation_fusion_demo.async_stiffness import (
    capture_direct_stiffness_action, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.moving_control import MovingController

from test_async_stiffness import observed, engine_for
from test_goal_mixture_criterion import _problem, _oracle


def test_independent_complete_density_and_centered_gradient_keep_every_goal_mode():
    outputs = []
    for options in ({}, {'weights': (.75, .25)}, {'goal_angles': (-.22, .35)},
                    {'underflow': True, 'goal_angles': (.08, .08)}):
        inference, result, controller = _problem(**options)
        source = np.asarray(result.prior.command)
        packet = controller.gradient_at_command(result, source)
        record = packet.record
        assert packet.status == 'ok', record['reason']
        actual, branches = controller.objective_at_command(result, source)
        expected = -_oracle(inference, result, controller, source)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(record['objective_at_source'], expected, atol=1e-10)
        # A distinct, smaller oracle stencil checks the complete gradient.
        step = controller.gradient_step*.5
        derivative = (-_oracle(inference, result, controller, source+[step])[0]
                      +_oracle(inference, result, controller, source-[step])[0])/(2.*step)
        np.testing.assert_allclose(record['gradient'], [derivative], rtol=1e-5, atol=1e-6)
        assert record['goal_log_weights'] == [item.log_weight for item in result.stages[-1].components]
        assert len(branches) == len(record['source_branches']) == 2
        for source_branch, branch in zip(result.stages[-1].components, record['source_branches']):
            assert branch['parent_state_id'] == source_branch.state.state_id
            assert branch['log_weight'] == source_branch.log_weight
            assert np.isfinite(branch['log_position_pdet'])
            assert np.isfinite(branch['log_bingham_normalizer'])
            np.testing.assert_allclose(np.array(branch['query_within'])+branch['query_between'],
                                       branch['query_covariance'], atol=1e-12)
        outputs.append(record['gradient'][0])
    assert abs(outputs[0]-outputs[1]) > 1e-3
    assert abs(outputs[0]-outputs[2]) > 1e-3


def test_repeated_frozen_gradient_is_identical_and_has_no_mutable_alias_or_endpoint_state():
    _, result, controller = _problem()
    original_velocity = np.array([.17])
    controller.velocity = original_velocity.copy()
    first = controller.gradient_at_command(result)
    second = controller.gradient_at_command(result)
    np.testing.assert_array_equal(first.gradient, second.gradient)
    assert first.record['objective_at_source'] == second.record['objective_at_source']
    assert first.record['finite_difference_diagnostics'] == second.record['finite_difference_diagnostics']
    assert first.record['endpoint_selection_performed'] is False
    np.testing.assert_array_equal(controller.velocity, original_velocity)
    copy = first.record
    copy['gradient'][0] = -999.
    copy['source_branches'][0]['query_covariance'][0][0] = -999.
    assert first.record['gradient'][0] != -999.
    assert first.record['source_branches'][0]['query_covariance'][0][0] >= 0.
    with pytest.raises(FrozenInstanceError):
        first.record_json = '{}'


@pytest.mark.parametrize('boundary', ['lower', 'upper'])
def test_only_physical_joint_positions_select_one_sided_differences(boundary):
    inference, result, controller = _problem()
    command = np.asarray(result.prior.command)
    getattr(controller, boundary)[:] = command
    # Old short-command velocity/acceleration boxes are deliberately tiny;
    # neither bounds the full-J differentiation domain.
    controller.speed = controller.acceleration = 1e-20
    packet = controller.gradient_at_command(result)
    assert packet.status == 'ok', packet.record['reason']
    diagnostics = packet.record['finite_difference_diagnostics']
    coordinate = diagnostics['coordinates'][0]
    direction = 1. if boundary == 'lower' else -1.
    side = 'forward' if direction > 0. else 'backward'
    assert coordinate['method'] == 'one-sided '+side+' derivative'
    step = coordinate[side]['step']
    assert abs(step) > 1e-6
    expected = (-_oracle(inference, result, controller, command+[step])[0]
                +_oracle(inference, result, controller, command)[0])/step
    np.testing.assert_allclose(packet.gradient, [expected], rtol=1e-8, atol=1e-7)
    assert diagnostics['objective_evaluation_count'] == 2


def test_failed_probe_uses_available_opposite_side_and_reports_the_actual_failure(monkeypatch):
    _, result, controller = _problem()
    original = controller._query
    source = result.prior.command[0]

    def restricted(state):
        if state.command[0] > source:
            raise ValueError('test stable branch unavailable on forward side')
        return original(state)

    monkeypatch.setattr(controller, '_query', restricted)
    packet = controller.gradient_at_command(result)
    assert packet.status == 'ok', packet.record['reason']
    coordinate = packet.record['finite_difference_diagnostics']['coordinates'][0]
    assert coordinate['forward']['status'] == 'evaluation_failed'
    assert 'stable branch unavailable' in coordinate['forward']['reason']
    assert coordinate['method'] == 'one-sided backward derivative'
    assert np.isfinite(packet.gradient[0])


def test_no_evaluable_side_and_nonfinite_objective_fail_without_fake_zero_gradient(monkeypatch):
    _, result, controller = _problem()
    original = controller._query
    source = result.prior.command[0]

    def restricted(state):
        if state.command[0] != source:
            raise ValueError('test no stable branch on either side')
        return original(state)

    monkeypatch.setattr(controller, '_query', restricted)
    failed = controller.gradient_at_command(result)
    assert failed.status == 'failed' and failed.gradient is None
    assert failed.record['objective_at_source'] is not None
    assert 'No finite objective difference' in failed.record['reason']
    attempts = failed.record['finite_difference_diagnostics']['evaluations']
    assert sum(item['status'] == 'failed' for item in attempts) == 2

    monkeypatch.setattr(controller, '_evaluate_command_objective',
                        lambda *_args: (float('nan'), [], ()))
    invalid = controller.gradient_at_command(result)
    assert invalid.status == 'failed' and invalid.gradient is None
    assert invalid.record['objective_at_source'] is None
    assert 'not finite' in invalid.record['reason']


def test_wrong_command_is_not_silently_relabelled_or_clipped():
    _, result, controller = _problem()
    packet = controller.gradient_at_command(result, np.asarray(result.prior.command)+.001)
    assert packet.status == 'failed' and packet.gradient is None
    assert 'transport history first' in packet.record['reason']
    assert packet.record['finite_difference_diagnostics']['objective_evaluation_count'] == 0
    with pytest.raises(ValueError, match='physical joint positions'):
        controller.objective_at_command(result, [controller.upper[0]+1.])


def test_full_K_action_and_every_joint_cross_remain_frozen_for_all_probes(observed, monkeypatch):
    scene, inference, result, _, _, raw = observed
    engine = engine_for(observed)
    captured = capture_direct_stiffness_action(engine, result, raw)
    engine.set(captured)
    result, mapping = reanchor_retained_stiffness(engine, result)
    controller = MovingController(inference, scene.robot_description,
        [0., -.24, .045], [.5, -.5, .5, .5],
        dict(max_joint_velocity=.048, max_joint_acceleration=.3, damping=.025))
    source_records = json.dumps(result.records, sort_keys=True)
    covariances = [item.state.covariance.copy() for item in result.stages[-1].goal_components]
    seen_anchors = []
    refine = engine.identifier._refine_equilibrium

    def capture(initial, command, stiffness):
        seen_anchors.append((np.array(initial), np.array(stiffness)))
        return refine(initial, command, stiffness)

    monkeypatch.setattr(engine.identifier, '_refine_equilibrium', capture)
    packet = controller.gradient_at_command(result, command_map=mapping)
    assert packet.status == 'ok', packet.record['reason']
    record = packet.record
    assert record['selected_K'] == raw.proposal['selected_K']
    assert record['parameter_density_id'] == raw.proposal['density_id']
    assert record['command_action']['captured_parameter_packet']['action_packet_id'] == captured.packet_id
    assert record['body_state_ids'] == [item.state.state_id for item in result.stages[-1].components]
    assert record['goal_state_ids'] == [item.state.state_id for item in result.stages[-1].goal_components]
    assert len(seen_anchors) == 2*len(result.prior.command)
    for anchor, stiffness in seen_anchors:
        np.testing.assert_array_equal(anchor, captured.action['auxiliary_equilibrium'])
        np.testing.assert_array_equal(stiffness, captured.action['selected_K'])
    for before, item in zip(covariances, result.stages[-1].goal_components):
        np.testing.assert_array_equal(before, item.state.covariance)
    assert json.dumps(result.records, sort_keys=True) == source_records
    # All evaluations retain body/environment cross blocks before the core
    # relative query; an independent-marginal replacement changes J.
    source = result.stages[-1].goal_components[0].state
    marker = source.edge_ids.index('marker')
    body_rows = np.array([6*source.edge_ids.index(name)+offset
        for name in inference.body_ids for offset in range(6)])
    assert np.linalg.norm(source.covariance[np.ix_(body_rows, np.arange(6*marker, 6*marker+6))]) > 0.
