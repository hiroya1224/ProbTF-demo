"""Complete-density IK has command bounds, with no separate s criterion."""

from types import SimpleNamespace

import numpy as np
import pytest

from probtf.geometry import axis_angle_to_quat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    pose_query,
)
from probtf_observation_fusion_demo.model import Stage, WeightedState
from probtf_observation_fusion_demo.moving_control import MovingController, command_state


def _problem(goal_angle=.2):
    body = BinghamTransform('body', [0., 0., 0.],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], .002*np.eye(3)),
        .0001*np.eye(3))
    tip = BinghamTransform('tip', [1., 0., 0.], rotation_wxyz=[1., 0., 0., 0.])
    marker = BinghamTransform('marker', [0., 0., 0.], rotation_wxyz=[1., 0., 0., 0.])
    state = JointTransformState.from_independent((body, tip, marker),
        command=(0.,), stamp=1., component_id='observed')
    inference = SimpleNamespace(body_ids=('body',),
        actions=({'joint_index': 0, 'axis': [0., 0., 1.]},),
        paths={'hand': ('body', 'tip'), 'marker': (('marker', False),)},
        settings={'root_frame': 'base'})
    path = (('marker', True), ('body', False), ('tip', False))
    goal_position = pose_query(state, path).mean[:3]
    urdf = ('<robot name="one"><joint name="one" type="revolute">'
            '<limit lower="-3" upper="3"/></joint></robot>')
    selector = MovingController(inference, urdf, goal_position,
        axis_angle_to_quat([0., 0., 1.], goal_angle),
        {'max_joint_velocity': .2, 'max_joint_acceleration': .6, 'damping': .025})
    result = SimpleNamespace(prior=state,
        stages=(Stage('fused', (WeightedState('observed', 1., state),)),),
        records={'observation_ids': []})
    return inference, result, selector


def _squared_position_error(inference, result, selector, command):
    future = command_state(result.prior, command, inference.body_ids, inference.actions)
    query = pose_query(future, selector.relative_path)
    error = query.mean[:3]-selector.goal_offset
    return float(error @ error+np.trace(query.covariance[:3, :3]))


def test_real_full_density_gradient_may_improve_orientation_while_position_s_increases(monkeypatch):
    inference, result, selector = _problem()
    original_covariance = result.prior.covariance.copy()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The complete-density update must not sample body candidates.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    selected, diagnostics = selector.step(result, .5)
    assert -diagnostics['log_expected_density'] < diagnostics['braking_baseline_objective']
    assert diagnostics['line_search_accepted']
    assert diagnostics['gradient_evaluation_count'] == 1
    # Real core propagation and the real normalized Gaussian/Bingham density:
    # the source position already matches; correcting orientation moves it.
    assert selected[0] > .01
    assert _squared_position_error(inference, result, selector, selected) > (
        _squared_position_error(inference, result, selector, result.prior.command))
    np.testing.assert_array_equal(result.prior.covariance, original_covariance)
    assert 'position_progress' not in diagnostics
    assert 'position_descent_correction' not in diagnostics
    for trial in diagnostics['line_search']:
        assert 'position_error_limit' not in trial
        assert 'position_progress_feasible' not in trial


@pytest.mark.parametrize('kind', ['visual', 'stiffness'])
def test_initial_candidate_is_accepted_by_J_even_when_position_s_increases(kind):
    inference, result, selector = _problem()
    seed = np.array([.04])
    assert _squared_position_error(inference, result, selector, seed) > (
        _squared_position_error(inference, result, selector, result.prior.command))
    if kind == 'visual':
        arguments = {'initial_command': seed}
        record_name = 'approach_initial_proposal'
    else:
        proposal = dict(command=seed.tolist(), source_command=list(result.prior.command),
            source_stamp=result.prior.stamp, source_prior_state_id=result.prior.state_id,
            body_posterior_state_ids=[result.prior.state_id], density_id='retained-K')
        result.records['stiffness'] = dict(posterior={'density_id': 'retained-K'},
                                           command_proposal=proposal)
        arguments = {'stiffness_proposal': proposal}
        record_name = 'stiffness_initial_proposal'
    _, diagnostics = selector.step(result, .5, **arguments)
    assert diagnostics[record_name]['accepted_as_initial_point']
    np.testing.assert_array_equal(diagnostics['gradient_at_command'], seed)
    assert diagnostics['feasible_baseline_objective'] < diagnostics['braking_baseline_objective']
    assert diagnostics['gradient_evaluation_count'] == 1
    assert 'position_error_limit' not in diagnostics[record_name]


def test_J_worse_seed_is_still_rejected():
    _, result, selector = _problem(goal_angle=0.)
    _, diagnostics = selector.step(result, .5, initial_command=np.array([.04]))
    proposal = diagnostics['approach_initial_proposal']
    assert not proposal['accepted_as_initial_point']
    assert proposal['proposed_objective'] > proposal['comparison_objective']
    np.testing.assert_array_equal(diagnostics['gradient_at_command'], result.prior.command)


@pytest.mark.parametrize('seed', [None, .04])
def test_all_Armijo_rejections_keep_the_validated_initial_candidate(monkeypatch, seed):
    _, result, selector = _problem()
    initial = 0. if seed is None else seed
    # A deterministic optimizer fixture: the finite-difference neighborhood
    # descends, but every available line-search scale raises the exact J.
    def log_density(query, *_args):
        command = query.command[0]
        cost = -command if command <= initial+2e-5 else 1.-initial
        return -cost, {}
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._relative_density', log_density)
    monkeypatch.setattr(selector, '_step_metric', lambda *_args: np.eye(1))
    arguments = {} if seed is None else {'initial_command': np.array([seed])}
    selected, diagnostics = selector.step(result, .5, **arguments)
    np.testing.assert_allclose(selected, [initial], atol=1e-12)
    assert diagnostics['quadratic_step']['solver_success']
    assert not diagnostics['line_search_accepted']
    assert len(diagnostics['line_search']) == 9
    assert all(trial['objective'] > trial['armijo_bound'] for trial in diagnostics['line_search'])
    assert diagnostics['gradient_evaluation_count'] == 1
    if seed is None:
        assert diagnostics['selection_reason'].startswith('feasible braking baseline')
    else:
        assert diagnostics['selection_reason'].startswith('validated complete-criterion initial proposal')


def test_failed_quadratic_returns_braking_baseline_even_after_a_J_improving_seed(monkeypatch):
    _, result, selector = _problem()
    def failed(_metric, _gradient, baseline, _lower, _upper):
        return np.zeros_like(baseline), dict(solver_success=False,
            constraints_satisfied=True, accepted=False, fallback=None)
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._bounded_gradient_direction', failed)
    selected, diagnostics = selector.step(result, .5, initial_command=np.array([.04]))
    assert diagnostics['approach_initial_proposal']['accepted_as_initial_point']
    np.testing.assert_array_equal(selected, result.prior.command)
    assert diagnostics['quadratic_step']['fallback'] == 'maximal braking baseline'
    assert diagnostics['gradient_evaluation_count'] == 1
