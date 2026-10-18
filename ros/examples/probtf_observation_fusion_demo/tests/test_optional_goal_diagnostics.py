"""Unused conditional-goal diagnostics cannot gate an unchanged body/task law."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo import body_projection, moving_inference
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.model import Stage
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from test_body_projection import _result, common_state
from test_moving_control import controller


def _forbidden_diagnostic(*unused, **unused_keywords):
    # Reproduce a failure in the OPTIONAL diagnostic arithmetic, without
    # changing the valid raw joint Gaussian or inventing a zero residual.
    raise ValueError('unused conditional Schur diagnostic is unavailable')


@pytest.mark.parametrize('underflow', [False, True])
def test_omitting_conditional_goals_preserves_every_joint_moment_and_source(
        common_state, monkeypatch, underflow):
    source = (_result(common_state, weights=(1., 0.), logs=(0., -1000.))
              if underflow else _result(common_state))
    baseline = body_projection.project_update(source)
    assert baseline.goal_components
    monkeypatch.setattr(body_projection, '_conditional_retarget', _forbidden_diagnostic)
    actual = body_projection.project_update(source, construct_conditional_goals=False)
    assert actual.goal_components == ()
    assert actual.state.state_id == baseline.state.state_id
    np.testing.assert_array_equal(actual.posterior.mean, baseline.posterior.mean)
    np.testing.assert_array_equal(actual.posterior.covariance, baseline.posterior.covariance)
    np.testing.assert_array_equal(actual.state.covariance, baseline.state.covariance)
    for key in ('sources', 'raw_source_state_ids', 'raw_mean', 'raw_covariance',
                'raw_within_covariance', 'raw_between_covariance',
                'source_prior_state_id', 'source_global_log_normalizer'):
        assert actual.record[key] == baseline.record[key]
    assert actual.record['conditional_goals_constructed'] is False
    assert actual.record['conditional_goal_omission_reason']
    assert actual.record['conditional_transformations'] == []
    assert actual.record['component_pruning_used'] is False
    if underflow:
        assert [entry['log_weight'] for entry in actual.record['sources']] == [0., -1000.]
    # The required/default conditional-goal path still reports the original
    # arithmetic failure; this is selective dead-work elimination, not an
    # exception handler, a PSD tolerance change, or a zero-information law.
    with pytest.raises(ValueError, match='unused conditional Schur'):
        body_projection.project_update(source)


def test_omission_does_not_bypass_the_actual_raw_joint_provenance_checks(common_state):
    source = _result(common_state)
    displayed = source.stages[-1].components
    foreign = replace(displayed[0], state=replace(displayed[0].state,
                                                 component_id='foreign-source'))
    wrong = SimpleNamespace(frozen_body_branches=source.frozen_body_branches,
        marker_path=source.marker_path, stages=(Stage('fused', (foreign,)+displayed[1:]),))
    with pytest.raises(ValueError, match='provenance differs'):
        body_projection.project_update(wrong, construct_conditional_goals=False)


@pytest.fixture(scope='module')
def capture(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('optional_goal_diagnostics'))
    frame = scene.step(0., 1000.)
    templates = yaml.safe_load((Path(__file__).resolve().parents[1]
                               /'config/imu_marker.yaml').read_text())['observations']
    return scene, scene.sensor_inputs(frame), templates


@pytest.mark.parametrize('retained_goal', [False, True])
def test_fresh_and_retained_task_goals_control_and_wire_are_identical_without_diagnostics(
        capture, monkeypatch, retained_goal):
    scene, inputs, templates = capture
    packets = [inputs]
    if retained_goal:
        packets.append(replace(inputs, image=np.full_like(inputs.image, 255),
                               stamp=1000.05, time=.05))
    original = body_projection.project_update
    baseline_inference = MovingInference(scene.robot_description, templates)

    def force_diagnostics(result, *, construct_conditional_goals=True):
        return original(result, construct_conditional_goals=True)

    # Construct an independently evaluated reference with the historical
    # extra diagnostics. They do not enter either task-goal policy.
    with monkeypatch.context() as patch:
        patch.setattr(moving_inference, 'project_update', force_diagnostics)
        for index, packet in enumerate(packets):
            baseline = baseline_inference.update(packet, index)

    actual_inference = MovingInference(scene.robot_description, templates)
    requested = []

    def tracked_projection(result, *, construct_conditional_goals=True):
        requested.append(construct_conditional_goals)
        return original(result, construct_conditional_goals=construct_conditional_goals)

    with monkeypatch.context() as patch:
        patch.setattr(body_projection, '_conditional_retarget', _forbidden_diagnostic)
        patch.setattr(moving_inference, 'project_update', tracked_projection)
        for index, packet in enumerate(packets):
            actual = actual_inference.update(packet, index)
    assert requested == [False]*len(packets)
    assert actual.prior.state_id == baseline.prior.state_id
    actual_stage, baseline_stage = actual.stages[-1], baseline.stages[-1]
    assert [item.state.state_id for item in actual_stage.components] == [
        item.state.state_id for item in baseline_stage.components]
    assert [(item.label, item.weight, item.log_weight, item.state.state_id)
            for item in actual_stage.goal_components] == [
        (item.label, item.weight, item.log_weight, item.state.state_id)
            for item in baseline_stage.goal_components]
    assert actual.records['observation_ids'] == baseline.records['observation_ids']
    assert actual.records['variational_observation_update'] == baseline.records[
        'variational_observation_update']
    assert actual.records['fixed_goal_memory'] == baseline.records['fixed_goal_memory']
    assert actual.records['internal_projection']['conditional_goals_constructed'] is False
    assert actual.records['internal_projection']['conditional_goal_diagnostics_used_for_task'] is False
    assert actual.records['internal_projection']['diagnostic_conditional_goal_state_ids'] == []
    for a, b in zip(actual.frozen_body_posterior.components,
                    baseline.frozen_body_posterior.components):
        np.testing.assert_array_equal(a.distribution.mean, b.distribution.mean)
        np.testing.assert_array_equal(a.distribution.covariance, b.distribution.covariance)
    if retained_goal:
        assert actual.records['fixed_goal_memory_event'] == 'image_missing_hold'
    else:
        assert actual.records['fresh_goal_construction'] == baseline.records['fresh_goal_construction']

    actual_command, actual_control = controller(scene, actual_inference).step(actual, .05)
    baseline_command, baseline_control = controller(scene, baseline_inference).step(baseline, .05)
    np.testing.assert_array_equal(actual_command, baseline_command)
    for key in ('body_source_state_ids', 'goal_source_state_ids', 'source_weights',
                'source_log_weights', 'root_goal_candidates'):
        assert actual_control[key] == baseline_control[key]
    wires = []
    for result, inference in ((actual, actual_inference), (baseline, baseline_inference)):
        snapshot = build_snapshot(result, result.stages[-1], inference.common)
        verify_adapter_identity(snapshot, result.stages[-1])
        settings = inference.common['display']
        message = make_markers(snapshot, settings['seed'], settings['point_size'])
        wires.append(verify_wire_identity(snapshot, message, seed=settings['seed'],
                                          point_size=settings['point_size']))
    assert wires[0] == wires[1]
