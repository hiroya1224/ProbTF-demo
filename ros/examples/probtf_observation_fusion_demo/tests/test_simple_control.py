"""Finite reaching commands use one correlated body/environment measure."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    conditional_total_covariance,
)
from probtf_observation_fusion_demo import simple_control
from probtf_observation_fusion_demo.body_native import point_query
from probtf_observation_fusion_demo.model import Stage, WeightedState
from probtf_observation_fusion_demo.moving_control import command_state
from probtf_observation_fusion_demo.simple_control import SimpleSelector, joint_reach_score
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from test_stiffness_corrected_control import corrected


@pytest.fixture
def joint_model():
    covariance = np.zeros((24, 24))
    for index, diagonal in ((0, [.005, .008, .01]), (1, [.007, .006, .009]),
                            (3, [.008, .007, .006])):
        covariance[6*index+3:6*index+6, 6*index+3:6*index+6] = np.diag(diagonal)
    covariance[18:21, 18:21] = np.diag([.002, .002, .001])
    covariance[5, 19] = covariance[19, 5] = .3*np.sqrt(covariance[5, 5]*covariance[19, 19])
    transforms = []
    for name, index, translation in (('link_1', 0, [0., 0., 0.]),
            ('link_2', 1, [.8, 0., 0.]), ('marker', 3, [.8, .4, 0.])):
        transforms.append(BinghamTransform(name, np.asarray(translation),
            bingham=bingham_from_rotation_coordinates([1., 0., 0., 0.],
                covariance[6*index+3:6*index+6, 6*index+3:6*index+6]),
            translation_covariance=covariance[6*index:6*index+3, 6*index:6*index+3]))
    transforms.insert(2, BinghamTransform('tool', np.array([.45, 0., 0.]),
        rotation_wxyz=np.array([1., 0., 0., 0.])))
    state = JointTransformState(tuple(transforms), covariance, version=5, stamp=2.,
        command=(0., 0.), component_id='shared_body_environment')
    inference = SimpleNamespace(reference_command=(0., 0.), body_ids=('link_1', 'link_2'),
        actions=({'joint_index': 0, 'axis': [0., 0., 1.]},
                 {'joint_index': 1, 'axis': [0., 0., 1.]}),
        paths=dict(hand=(('link_1', False), ('link_2', False), ('tool', False)),
            marker=(('marker', False),)),
        settings=dict(root_frame='base_link', control=dict(goal_offset=[0., 0., 0.])),
        goal_observed=True)
    description = '<robot name="two_joints">'+''.join(
        '<joint name="j{0}" type="revolute"><parent link="{1}"/><child link="link_{0}"/>'
        '<axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>'.format(
            index, 'base_link' if index == 1 else 'link_1') for index in (1, 2))+'</robot>'
    result = SimpleNamespace(prior=state,
        stages=(Stage('canonical', (WeightedState('canonical', 1., state),)),),
        records=dict(simple_inference=dict(goal_observed=True)))
    selector = SimpleSelector(inference, description, {})
    return inference, state, result, selector


def test_root_frame_score_equals_full_joint_difference_moments_and_includes_environment_cross(joint_model):
    inference, state, _, selector = joint_model
    score, record = selector.score(state)
    hand = point_query(state, inference.paths['hand'])
    goal = point_query(state, inference.paths['marker'])
    covariance, _, _ = conditional_total_covariance(state.covariance,
        hand.jacobian-goal.jacobian)
    expected = float(np.linalg.norm(hand.mean-goal.mean)**2+np.trace(covariance))
    assert score == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose(record['difference_covariance'], covariance, atol=1e-13)
    independent = float(np.linalg.norm(hand.mean-goal.mean)**2
        +np.trace(hand.covariance+goal.covariance))
    assert abs(score-independent) > 1e-4
    assert record['hand_state_id'] == record['goal_state_id'] == state.state_id
    assert not np.array_equal(hand.mean, hand.mode)


def test_finite_euclidean_step_decreases_only_shared_score_and_keeps_source_law(joint_model, monkeypatch):
    inference, state, result, selector = joint_model
    def forbidden(*args, **kwargs):
        raise AssertionError('No Monte Carlo is used by finite command differentiation.')
    for name in ('normal', 'random', 'randn', 'default_rng', 'multivariate_normal'):
        monkeypatch.setattr(np.random, name, forbidden)
    before = state.covariance.tobytes()
    q, record = selector.select(result)
    assert record['status'] == 'selected_decreasing_finite_step'
    assert record['S_after'] < record['S_before']
    assert 0. < np.linalg.norm(q-np.asarray(state.command)) <= .02+1e-14
    assert np.all(q >= selector.lower) and np.all(q <= selector.upper)
    assert not hasattr(selector, 'velocity')
    assert record['body_source_state_ids'] == record['goal_source_state_ids'] == [state.state_id]
    assert not record['independent_velocity_state']
    assert record['command_action'] == 'declared_right_joint_action_no_selected_K'
    future = command_state(state, q, inference.body_ids, inference.actions)
    assert selector.score(future)[0] == record['S_after']
    assert state.covariance.tobytes() == before


def test_all_trials_act_on_the_frozen_estimated_body_and_use_supplied_equilibrium_increment(joint_model):
    inference, state, result, selector = joint_model
    class Map:
        def __init__(self):
            self.sources = []
            self.validated = []
        def validate_result(self, value):
            self.validated.append(value)
        def record(self):
            return dict(selected_K=[10., 20.], prediction_method='declared synthetic equilibrium increment')
        def push(self, source, command, body_ids, actions):
            self.sources.append(source)
            return command_state(source, command, body_ids, actions,
                joint_increment=.6*(command-np.asarray(source.command)))
        def selection_record(self, command):
            return dict(joint_increment=(.6*(command-np.asarray(state.command))).tolist())
    mapping = Map()
    q, record = selector.select(result, command_map=mapping)
    assert mapping.validated == [result]
    assert len(mapping.sources) >= 5 and all(source is state for source in mapping.sources)
    expected = command_state(state, q, inference.body_ids, inference.actions,
        joint_increment=.6*(q-np.asarray(state.command)))
    assert selector.score(expected)[0] == record['S_after']
    assert record['command_map']['selected_K'] == [10., 20.]
    assert not record['body_reconstructed_from_stiffness']


def test_no_marker_hold_does_not_query_unknown_goal_then_missing_marker_retains_shared_goal(joint_model, monkeypatch):
    _, state, result, selector = joint_model
    def forbidden(*args, **kwargs):
        raise AssertionError('An unobserved landmark prior must not become a reaching target.')
    with monkeypatch.context() as patch:
        patch.setattr(simple_control, 'point_query', forbidden)
        q, record = selector.select(result, goal_observed=False)
    np.testing.assert_array_equal(q, state.command)
    assert record['status'] == 'holding_before_first_valid_marker'
    assert record['S_before'] is None
    result.records['marker_detected'] = False
    q, record = selector.select(result)
    assert record['goal_observed'] and record['S_before'] is not None
    assert record['goal_source_state_ids'] == [state.state_id]


def test_direction_derivative_and_returned_score_include_candidate_covariance(joint_model):
    inference, state, result, selector = joint_model
    _, record = selector.select(result)
    expected = []
    for direction in np.eye(2):
        plus = command_state(state, 1e-5*direction, inference.body_ids, inference.actions)
        minus = command_state(state, -1e-5*direction, inference.body_ids, inference.actions)
        expected.append((selector.score(plus)[0]-selector.score(minus)[0])/2e-5)
    np.testing.assert_allclose(record['gradient'], expected, rtol=2e-7, atol=2e-8)
    assert record['source_score']['difference_covariance'] != record['selected_score']['difference_covariance']


def test_joint_limits_are_used_without_velocity_or_curvature_model(joint_model):
    inference, state, result, selector = joint_model
    command = np.array([selector.upper[0], 0.])
    at_limit = command_state(state, command, inference.body_ids, inference.actions)
    result = SimpleNamespace(prior=at_limit,
        stages=(Stage('canonical', (WeightedState('canonical', 1., at_limit),)),),
        records=result.records)
    q, record = selector.select(result)
    assert np.all(q >= selector.lower) and np.all(q <= selector.upper)
    assert record['gradient_difference_spans_rad'][0] == pytest.approx(selector.difference_step)
    assert record['S_after'] <= record['S_before']


def test_real_stiffness_command_map_preserves_observed_body_for_simple_trial_selection(corrected):
    scene, _, contexts = corrected
    inference, source, estimate, proposal = contexts['imu_marker']
    stage = replace(source.stages[-1], goal_components=())
    result = replace(source, stages=source.stages[:-1]+(stage,))
    mapping = StiffnessCommandMap(estimate, result, proposal)
    selector = SimpleSelector(inference, scene.robot_description,
        dict(goal_offset=inference.settings['control']['goal_offset'], max_step_rad=.005))
    before = result.stages[-1].components[0].state
    original_covariance = before.covariance.tobytes()
    q, record = selector.select(result, command_map=mapping, goal_observed=True)
    expected = mapping.push(before, q, inference.body_ids, inference.actions)
    assert record['S_after'] == selector.score(expected)[0]
    assert record['S_after'] <= record['S_before']
    assert before.covariance.tobytes() == original_covariance
    assert record['command_map']['selected_K'] == proposal['selected_K']
    assert record['selected_command_action']['source_command'] == list(before.command)
    assert record['selected_command_action']['body_posterior_state_ids'] == [before.state_id]
    assert not record['body_reconstructed_from_stiffness']
