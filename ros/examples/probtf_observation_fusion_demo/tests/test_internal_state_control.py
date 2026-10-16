"""A single retained body never replaces the observed goal mixture in IK."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf.geometry import axis_angle_to_quat, quat_right_matrix, quat_to_rotmat
from probtf.spherical_law.first_order import (BinghamTransform, JointTransformState,
    bingham_from_rotation_coordinates)
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.moving_control import command_state, control_components
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier

from test_goal_mixture_criterion import _problem, _oracle


PACKAGE = Path(__file__).resolve().parents[1]


def _internal_and_goals(result):
    goals = result.stages[-1].components
    internal = WeightedState('internal', 1., replace(
        goals[0].state, component_id='internal'), 0.)
    stage = SimpleNamespace(label='fused', components=(internal,),
        sensor_ids=result.stages[-1].sensor_ids, goal_components=goals)
    return internal, stage


@pytest.mark.parametrize('underflow', [False, True])
def test_full_goal_density_and_gradient_keep_all_modes_with_one_internal_body(underflow, monkeypatch):
    inference, original, selector = _problem(underflow=underflow)
    internal, stage = _internal_and_goals(original)
    result = SimpleNamespace(**dict(vars(original), stages=(stage,)))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The internal approximation and goal criterion must be analytic.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    selected, diagnostics = selector.step(result, .05)
    expected = _oracle(inference, original, selector, selected)[0]
    np.testing.assert_allclose(diagnostics['log_expected_density'], expected, atol=1e-10)
    assert diagnostics['body_source_state_ids'] == [internal.state.state_id]
    assert diagnostics['body_source_weights'] == [1.]
    assert diagnostics['body_source_log_weights'] == [0.]
    goal_ids = [item.state.state_id for item in stage.goal_components]
    assert diagnostics['source_state_ids'] == diagnostics['goal_source_state_ids'] == goal_ids
    assert diagnostics['source_weights'] == [item.weight for item in stage.goal_components]
    assert diagnostics['source_log_weights'] == [item.log_weight for item in stage.goal_components]
    assert diagnostics['goal_parent_internal_state_id'] == internal.state.state_id
    assert [item['parent_state_id'] for item in diagnostics['selected_branches']] == goal_ids
    assert diagnostics['gradient_evaluation_count'] == 1
    baseline = np.array(diagnostics['gradient_at_command'])
    difference = diagnostics['gradient_difference_steps'][0]
    expected_gradient = -(_oracle(inference, original, selector, baseline+[difference])[0]
                         -_oracle(inference, original, selector, baseline)[0])/difference
    np.testing.assert_allclose(diagnostics['objective_gradient'], [expected_gradient], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('changed', ['mean', 'covariance', 'version', 'stamp', 'command'])
def test_goal_branch_cannot_substitute_a_different_body_or_time(changed):
    _, original, _ = _problem()
    _, stage = _internal_and_goals(original)
    candidate = stage.goal_components[1]
    state = candidate.state
    if changed == 'mean':
        body = replace(state.transforms[0], translation=state.transforms[0].translation+[.01, 0., 0.],
                       rotation_wxyz=None)
        state = replace(state, transforms=(body,)+state.transforms[1:])
    elif changed == 'covariance':
        covariance = state.covariance.copy()
        covariance[:3, :3] += .0001*np.eye(3)
        body = replace(state.transforms[0], translation_covariance=covariance[:3, :3],
                       rotation_wxyz=None)
        state = replace(state, transforms=(body,)+state.transforms[1:], covariance=covariance)
    else:
        value = {'version': state.version+1, 'stamp': state.stamp+.05,
                 'command': (state.command[0]+.01,)}[changed]
        state = replace(state, **{changed: value})
    stage.goal_components = (stage.goal_components[0], replace(candidate, state=state))
    result = SimpleNamespace(**dict(vars(original), stages=(stage,)))
    with pytest.raises(ValueError, match='same internal body marginal'):
        control_components(result)


@pytest.fixture(scope='module')
def corrected_goal_problem(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('internal_goal_control'),
                              imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    observations = yaml.safe_load((PACKAGE/'config/imu_hand.yaml').read_text())['observations']
    inference = MovingInference(scene.robot_description, observations)
    result = inference.update(inputs, 0)
    internal = result.stages[-1].components[0]
    state = internal.state
    marker_index = state.edge_ids.index('marker')
    goals = []
    for index, weight in enumerate((.3, .7)):
        transforms = list(state.transforms)
        transforms[marker_index] = replace(transforms[marker_index],
            translation=transforms[marker_index].translation+[0., .02*(2*index-1), 0.],
            rotation_wxyz=None)
        label = 'observed_goal_'+str(index)
        goals.append(WeightedState(label, weight, replace(state,
            transforms=tuple(transforms), component_id=label), float(np.log(weight))))
    stage = SimpleNamespace(label='fused', components=(internal,),
        sensor_ids=result.stages[-1].sensor_ids, goal_components=tuple(goals))
    result = replace(result, stages=result.stages[:-1]+(stage,))
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    proposal = identifier.command_proposal(result)
    assert proposal is not None
    return scene, inference, result, identifier, proposal


def test_fixed_K_map_authorizes_goal_laws_with_the_same_internal_body(corrected_goal_problem):
    _, inference, result, identifier, proposal = corrected_goal_problem
    mapping = StiffnessCommandMap(identifier, result, proposal)
    internal, goals = control_components(result)
    mapping.validate_result(result)
    record = mapping.record()
    assert record['body_posterior_state_ids'] == [internal[0].state.state_id]
    assert record['goal_source_state_ids'] == [item.state.state_id for item in goals]
    assert record['goal_source_weights'] == [.3, .7]
    assert record['goal_parent_internal_state_id'] == internal[0].state.state_id
    trial = np.asarray(result.prior.command)+[.0002, -.0001, .0001, -.0001, .0002, .0001]
    future_body = mapping.push(internal[0].state, trial, inference.body_ids, inference.actions)
    marker_index = future_body.edge_ids.index('marker')
    body = [coordinate for index in range(len(future_body.transforms)) if index != marker_index
            for coordinate in range(6*index, 6*index+6)]
    for goal in goals:
        future = mapping.push(goal.state, trial, inference.body_ids, inference.actions)
        np.testing.assert_array_equal(future.covariance[np.ix_(body, body)],
                                      future_body.covariance[np.ix_(body, body)])
        assert future.transforms[marker_index] is goal.state.transforms[marker_index]
        for index, edge in enumerate(future_body.transforms):
            if index != marker_index:
                np.testing.assert_array_equal(future.transforms[index].rotation_wxyz, edge.rotation_wxyz)
                np.testing.assert_array_equal(future.transforms[index].translation, edge.translation)
        assert mapping.push(goal.state, result.prior.command,
                            inference.body_ids, inference.actions) is goal.state
    foreign = replace(goals[0].state, component_id='unregistered_goal')
    with pytest.raises(ValueError, match='frozen source'):
        mapping.push(foreign, trial, inference.body_ids, inference.actions)
    changed = SimpleNamespace(**dict(vars(result.stages[-1]), goal_components=tuple(
        replace(item, weight=1.-item.weight, log_weight=float(np.log(1.-item.weight)))
        for item in goals)))
    with pytest.raises(ValueError, match='different or modified'):
        mapping.validate_result(replace(result, stages=result.stages[:-1]+(changed,)))
    stiffness_record = result.records['stiffness']
    source_ids = stiffness_record['body_posterior_state_ids']
    try:
        stiffness_record['body_posterior_state_ids'] = [item.state.state_id for item in goals]
        with pytest.raises(ValueError, match='different or modified'):
            mapping.validate_result(result)
    finally:
        stiffness_record['body_posterior_state_ids'] = source_ids
    mapping.validate_result(result)


def test_K_proposal_uses_internal_provenance_while_complete_criterion_uses_goal_modes(corrected_goal_problem):
    scene, inference, result, identifier, proposal = corrected_goal_problem
    mapping = StiffnessCommandMap(identifier, result, proposal)
    settings = inference.settings['control']
    from probtf_observation_fusion_demo.moving_control import MovingController
    selector = MovingController(inference, scene.robot_description,
        settings['goal_offset'], settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])
    _, diagnostics = selector.step(result, .05, stiffness_proposal=proposal, command_map=mapping)
    assert diagnostics['body_source_state_ids'] == proposal['body_posterior_state_ids']
    assert diagnostics['source_state_ids'] == [item.state.state_id
        for item in result.stages[-1].goal_components]
    assert len(diagnostics['selected_branches']) == 2
    assert diagnostics['stiffness_initial_proposal']['source'] is proposal
    assert diagnostics['selected_command_action']['body_posterior_state_ids'] == proposal['body_posterior_state_ids']
    assert diagnostics['selected_command_action']['goal_source_state_ids'] == diagnostics['source_state_ids']


@pytest.mark.parametrize('direction', [-1., 1.])
def test_repeated_limit_seeking_commands_keep_room_to_brake_for_601_cycles(direction, monkeypatch):
    """An objective beyond either limit must brake before the feasible set vanishes."""
    inference, result, selector = _problem()
    # Isolate the velocity/position interface with an always outward objective.
    # Its full gradient is exact and nonzero; no geometric solver can supply a
    # lucky inward turn that conceals an infeasible limit approach.
    selector._query = lambda state: SimpleNamespace(command=state.command, state_id=state.state_id,
        mean=np.r_[selector.goal_offset, np.zeros(3)], covariance=np.zeros((6, 6)))
    selector._step_metric = lambda *_args: np.eye(1)
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._relative_density',
        lambda query, *_args: (direction*query.command[0], {}))
    dt = .05
    previous_velocity = selector.velocity.copy()
    braking_seen = False
    for index in range(601):
        current = np.array(result.prior.command)
        selected, diagnostics = selector.step(result, dt)
        velocity = (selected-current)/dt
        assert np.all(selected >= selector.lower) and np.all(selected <= selector.upper)
        assert np.max(np.abs(velocity)) <= selector.speed+1e-12
        assert np.max(np.abs(velocity-previous_velocity)) <= selector.acceleration*dt+1e-12
        assert diagnostics['gradient_evaluation_count'] == 1
        if direction*(velocity[0]-previous_velocity[0]) < -1e-9:
            braking_seen = True
            assert direction*current[0] < 3.
        # Independently simulate future maximum braking. Every selected state
        # must admit this continuation without crossing a joint limit.
        stopped_position, stopped_velocity = selected.copy(), velocity.copy()
        for _ in range(20):
            stopped_velocity = np.sign(stopped_velocity)*np.maximum(
                np.abs(stopped_velocity)-selector.acceleration*dt, 0.)
            stopped_position += dt*stopped_velocity
            assert np.all(stopped_position >= selector.lower-1e-12)
            assert np.all(stopped_position <= selector.upper+1e-12)
        assert np.all(stopped_velocity == 0.)
        stage = result.stages[-1]
        next_components = tuple(replace(item, state=replace(item.state, command=tuple(selected)))
                                for item in stage.components)
        result = SimpleNamespace(**dict(vars(result), prior=next_components[0].state,
            stages=(replace(stage, components=next_components),)))
        previous_velocity = velocity
    assert braking_seen
    assert abs(selected[0]-direction*3.) < 1e-9
    assert abs(selector.velocity[0]) < 1e-9


def test_concentrated_Bingham_command_actions_keep_symmetric_parameters_and_full_covariance():
    covariance = np.diag([3e-5, 5e-5, 8e-5])
    parameter = bingham_from_rotation_coordinates(
        axis_angle_to_quat([1., 2., -3.], .78), covariance)
    body = BinghamTransform('body', [0., 0., 0.], parameter, .0002*np.eye(3))
    marker = BinghamTransform('marker', [.3, -.2, .1],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], .003*np.eye(3)), .0003*np.eye(3))
    state = JointTransformState.from_independent((body, marker), command=(0.,), stamp=1.)
    full = state.covariance.copy()
    full[3:6, 6:9] = np.diag([1e-6, -2e-6, 3e-6])
    full[6:9, 3:6] = full[3:6, 6:9].T
    state = replace(state, covariance=full)
    actions = ({'joint_index': 0, 'axis': [1., -2., .5]},)
    probes = np.array([[1., 2., 3., 4.], [-3., 2., 1., -.5],
                       [0., 1., -.3, 2.], [1., 0., 0., 0.]])
    probes /= np.linalg.norm(probes, axis=1)[:, None]
    for increment in (.013, -.17, .61, -.33, 1.1):
        current = state
        command = np.asarray(current.command)+increment
        future = command_state(current, command, ('body',), actions)
        quaternion = axis_angle_to_quat(actions[0]['axis'], command[0]-current.command[0])
        right = quat_right_matrix(quaternion)
        raw_parameter = right@current.transforms[0].bingham@right.T
        np.testing.assert_array_equal(future.transforms[0].bingham, future.transforms[0].bingham.T)
        for probe in probes:
            np.testing.assert_allclose(probe@future.transforms[0].bingham@probe,
                                      probe@raw_parameter@probe, rtol=2e-14, atol=1e-11)
        chart = np.eye(current.mean.size)
        chart[3:6, 3:6] = quat_to_rotmat(quaternion).T
        np.testing.assert_array_equal(future.covariance, chart@current.covariance@chart.T)
        assert future.transforms[1] is current.transforms[1]
        assert command_state(future, future.command, ('body',), actions) is future
        restored = command_state(future, current.command, ('body',), actions)
        np.testing.assert_allclose(restored.transforms[0].bingham,
                                  current.transforms[0].bingham, rtol=1e-13, atol=1e-10)
        np.testing.assert_allclose(restored.covariance, current.covariance, rtol=1e-13, atol=1e-18)
        state = future
