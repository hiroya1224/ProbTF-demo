"""The estimated-K command map acts on the observed full body distribution."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.geometry import axis_angle_to_quat, quat_right_matrix, quat_to_rotmat
from probtf_observation_fusion_demo.moving_control import MovingController
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def corrected(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('stiffness_corrected_control'),
                              imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    # Historical combined-packet source for command-law compatibility.
    settings['bayes_factor_surprise'] = {'enabled': False}
    contexts = {}
    for name in ('imu_hand', 'imu_marker'):
        observations = yaml.safe_load((PACKAGE/'config'/f'{name}.yaml').read_text())['observations']
        inference = MovingInference(scene.robot_description, observations)
        result = inference.update(inputs, 0)
        identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
        result.records['stiffness'] = identifier.update(result)
        proposal = identifier.command_proposal(result)
        assert proposal is not None
        contexts[name] = inference, result, identifier, proposal
    return scene, inputs, contexts


def test_fixed_auxiliary_command_jacobian_matches_independent_equilibrium_differences(corrected):
    _, _, contexts = corrected
    inference, result, identifier, proposal = contexts['imu_hand']
    mapping = StiffnessCommandMap(identifier, result, proposal)
    command = np.array(result.prior.command)
    analytic = mapping.jacobian(command)
    numeric = np.zeros_like(analytic)
    step = 1e-5
    anchor, stiffness = np.array(proposal['auxiliary_equilibrium']), np.array(proposal['selected_K'])
    for joint in range(len(command)):
        increment = step*np.eye(len(command))[joint]
        plus = identifier._refine_equilibrium(anchor, command+increment, stiffness)[0]
        minus = identifier._refine_equilibrium(anchor, command-increment, stiffness)[0]
        numeric[:, joint] = (plus-minus)/(2.*step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)
    assert np.linalg.norm(analytic-np.eye(len(command))) > 1e-3
    assert not analytic.flags.writeable
    assert mapping.record()['physical_truth_received'] is False
    assert mapping.record()['body_reconstructed_from_stiffness'] is False
    for component in result.stages[-1].components:
        assert mapping.push(component.state, command, inference.body_ids,
                            inference.actions) is component.state


def test_corrected_future_law_preserves_all_covariance_and_the_unknown_marker(corrected, monkeypatch):
    _, _, contexts = corrected
    inference, result, identifier, proposal = contexts['imu_marker']
    mapping = StiffnessCommandMap(identifier, result, proposal)
    command = np.array(result.prior.command)
    trial = command + np.array([.0004, -.0003, .0002, -.0001, .0003, -.0002])
    theta = identifier._refine_equilibrium(np.array(proposal['auxiliary_equilibrium']),
        trial, np.array(proposal['selected_K']))[0]
    equilibrium_increment = theta-np.array(proposal['auxiliary_equilibrium'])
    assert np.linalg.norm(equilibrium_increment-(trial-command)) > 1e-7

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Corrected control must not sample or reconstruct a physical body.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    for component in result.stages[-1].components:
        source = component.state
        before_covariance = source.covariance.copy()
        future = mapping.push(source, trial, inference.body_ids, inference.actions)
        chart = np.eye(source.mean.size)
        for name, action in zip(inference.body_ids, inference.actions):
            index = source.edge_ids.index(name)
            delta = axis_angle_to_quat(action['axis'], equilibrium_increment[int(action['joint_index'])])
            chart[6*index+3:6*index+6, 6*index+3:6*index+6] = quat_to_rotmat(delta).T
            right = quat_right_matrix(delta)
            np.testing.assert_allclose(future.transforms[index].bingham,
                right@source.transforms[index].bingham@right.T, atol=1e-12)
            np.testing.assert_array_equal(future.transforms[index].translation,
                                          source.transforms[index].translation)
        np.testing.assert_allclose(future.covariance, chart@source.covariance@chart.T,
                                   atol=1e-15, rtol=1e-12)
        marker = source.edge_ids.index('marker')
        assert future.transforms[marker] is source.transforms[marker]
        assert np.linalg.norm(future.covariance[:36, 6*marker:6*marker+6]) > 1e-6
        np.testing.assert_array_equal(source.covariance, before_covariance)
        assert future.command == tuple(trial)
        assert future.stamp == source.stamp and future.version == source.version
        assert future.component_id == source.component_id


def test_trial_equilibria_are_order_independent_and_invalid_branches_do_not_fall_back(corrected, monkeypatch):
    _, _, contexts = corrected
    inference, result, identifier, proposal = contexts['imu_hand']
    forward = StiffnessCommandMap(identifier, result, proposal)
    reverse = StiffnessCommandMap(identifier, result, proposal)
    source = result.stages[-1].components[0].state
    command = np.array(source.command)
    trials = [command+np.eye(len(command))[joint]*.0003 for joint in range(len(command))]
    first = {tuple(q): forward.push(source, q, inference.body_ids, inference.actions) for q in trials}
    for q in reversed(trials):
        state = reverse.push(source, q, inference.body_ids, inference.actions)
        assert state.state_id == first[tuple(q)].state_id
        np.testing.assert_array_equal(reverse.jacobian(q), forward.jacobian(q))
    saved = json.dumps(forward.record(), sort_keys=True)
    copied = forward.record()
    copied['selected_K'][0] += 1.
    assert json.dumps(forward.record(), sort_keys=True) == saved
    def unavailable(*_args, **_kwargs):
        raise ValueError('A declared auxiliary stable branch is unavailable.')
    monkeypatch.setattr(identifier, '_refine_equilibrium', unavailable)
    with pytest.raises(ValueError, match='stable branch is unavailable'):
        forward.push(source, command+np.full(6, .0004), inference.body_ids, inference.actions)
    assert json.dumps(forward.record(), sort_keys=True) == saved


def test_stale_or_foreign_body_and_density_sources_are_rejected(corrected):
    _, _, contexts = corrected
    inference, result, identifier, proposal = contexts['imu_hand']
    mapping = StiffnessCommandMap(identifier, result, proposal)
    mapping.validate_result(result)
    with pytest.raises(ValueError, match='same recorded'):
        StiffnessCommandMap(identifier, result, dict(proposal))
    foreign = replace(result.prior, stamp=result.prior.stamp+.05)
    with pytest.raises(ValueError, match='frozen source'):
        mapping.push(foreign, result.prior.command, inference.body_ids, inference.actions)
    with pytest.raises(ValueError, match='different or modified'):
        mapping.validate_result(replace(result, prior=foreign))
    changed_stage = replace(result.stages[-1], components=(replace(
        result.stages[-1].components[0], log_weight=-.1),))
    with pytest.raises(ValueError, match='different or modified'):
        mapping.validate_result(replace(result, stages=result.stages[:-1]+(changed_stage,)))
    record = result.records['stiffness']
    correct_versions = record['body_posterior_versions']
    try:
        record['body_posterior_versions'] = [version+1 for version in correct_versions]
        with pytest.raises(ValueError, match='different body update'):
            StiffnessCommandMap(identifier, result, proposal)
        with pytest.raises(ValueError, match='different or modified'):
            mapping.validate_result(result)
    finally:
        record['body_posterior_versions'] = correct_versions
    mapping.validate_result(result)


def test_one_complete_objective_step_uses_the_same_corrected_candidates(corrected):
    scene, _, contexts = corrected
    inference, result, identifier, proposal = contexts['imu_marker']
    mapping = StiffnessCommandMap(identifier, result, proposal)
    settings = inference.settings['control']
    selector = MovingController(inference, scene.robot_description,
        settings['goal_offset'], settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])
    selected, diagnostics = selector.step(result, .05, command_map=mapping)
    increment = selected-np.array(result.prior.command)
    assert np.max(np.abs(increment)) <= settings['max_joint_velocity']*.05+1e-12
    assert np.max(np.abs(increment/.05)) <= settings['max_joint_acceleration']*.05+1e-12
    assert diagnostics['gradient_evaluation_count'] == 1
    stage = result.stages[-1]
    goals = stage.goal_components or stage.components
    assert diagnostics['source_state_ids'] == [c.state.state_id for c in goals]
    assert diagnostics['body_source_state_ids'] == [c.state.state_id for c in stage.components]
    assert diagnostics['stiffness_command_correction'] == mapping.record()
    assert -diagnostics['log_expected_density'] <= diagnostics['feasible_baseline_objective']+1e-12
    for component, branch in zip(goals, diagnostics['selected_branches']):
        future = mapping.push(component.state, selected, inference.body_ids, inference.actions)
        query = selector._query(future)
        np.testing.assert_array_equal(query.mean, branch['query_mean'])
        np.testing.assert_array_equal(query.covariance, branch['query_covariance'])
        assert branch['state_id'] == future.state_id
        assert branch['query_command'] == selected.tolist()
        assert branch['parent_state_id'] == component.state.state_id


def test_selected_corrected_law_is_the_next_prior_and_published_snapshot_source(corrected):
    scene, inputs, contexts = corrected
    original, result, identifier, proposal = contexts['imu_marker']
    inference = MovingInference(scene.robot_description, original.templates)
    reproduced = inference.update(inputs, 0)
    assert [item.state.state_id for item in reproduced.stages[-1].components] == [
        item.state.state_id for item in result.stages[-1].components]
    mapping = StiffnessCommandMap(identifier, result, proposal)
    settings = inference.settings['control']
    controller = MovingController(inference, scene.robot_description,
        settings['goal_offset'], settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])
    selected, diagnostics = controller.step(result, .05, command_map=mapping)
    selection = mapping.selection_record(selected)
    saved = json.dumps(selection, sort_keys=True)
    assert selection['command'] == selected.tolist()
    np.testing.assert_array_equal(selection['command_jacobian'], mapping.jacobian(selected))
    later = replace(inputs, stamp=inputs.stamp+.05, time=.05, command=selected,
                    image=np.full_like(inputs.image, 255))
    updated = inference.update(later, 1, command_map=mapping)
    # A fresh IMU packet invokes the declared change prior. Its continuation
    # law is still precisely the selected K-corrected law, with prior mass
    # .75; the additional direct-family reset has prior mass .25.
    assert updated.records['bayes_factor_surprise']['applied']
    assert len(updated.stages[0].components) == 2
    assert [item.weight for item in updated.stages[0].components] == [.75, .25]
    np.testing.assert_array_equal([item.log_weight for item in updated.stages[0].components],
                                  np.log([.75, .25]))
    for source, prior, selected_body_id in zip(result.stages[-1].components,
            updated.stages[0].components, diagnostics['selected_body_state_ids']):
        expected = mapping.push(source.state, selected, inference.body_ids, inference.actions)
        assert selected_body_id == expected.state_id
        np.testing.assert_array_equal(prior.state.covariance, expected.covariance)
        for actual, predicted in zip(prior.state.transforms, expected.transforms):
            np.testing.assert_array_equal(actual.translation, predicted.translation)
            if predicted.bingham is not None:
                np.testing.assert_array_equal(actual.bingham, predicted.bingham)
        assert prior.state.command == tuple(selected)
        assert prior.state.stamp == later.stamp and prior.state.version > expected.version
        assert prior.weight == .75*source.weight
        assert prior.log_weight == np.log(.75)+source.log_weight
    for source, branch in zip(result.stages[-1].goal_components,
                             diagnostics['selected_branches']):
        expected = mapping.push(source.state, selected, inference.body_ids, inference.actions)
        assert branch['state_id'] == expected.state_id
    snapshot = build_snapshot(updated, updated.stages[-1], inference.common)
    verify_adapter_identity(snapshot, updated.stages[-1])
    display = inference.common['display']
    message = make_markers(snapshot, display['seed'], display['point_size'])
    assert verify_wire_identity(snapshot, message, seed=display['seed'], point_size=display['point_size'])
    assert snapshot.command == tuple(selected) and snapshot.stamp == later.stamp
    assert json.dumps(selection, sort_keys=True) == saved
