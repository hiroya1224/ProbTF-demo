"""Exact J/s acceptance after correcting a rejected tangent direction."""

from types import SimpleNamespace

import numpy as np

from probtf.spherical_law.first_order import BinghamTransform, JointTransformState
from probtf_observation_fusion_demo.model import Stage, WeightedState
from probtf_observation_fusion_demo.moving_control import (
    MovingController, _position_interior_directions,
)


def test_interior_correction_keeps_box_and_strictly_descends_both_gradients():
    direction = np.array([.5, 0.])
    gradient = np.array([-1., 0.])
    progress = np.array([0., 2.])
    originals = [item.copy() for item in (direction, gradient, progress)]
    candidates = list(_position_interior_directions(direction, gradient, progress,
        np.zeros(2), np.full(2, -.5), np.full(2, .5)))
    assert len(candidates) == 1
    corrected, record = candidates[0]
    assert record['axis'] == 1
    assert gradient @ corrected < 0. and progress @ corrected < 0.
    assert np.all(corrected >= -.5) and np.all(corrected <= .5)
    for actual, expected in zip((direction, gradient, progress), originals):
        np.testing.assert_array_equal(actual, expected)
    assert not list(_position_interior_directions(np.zeros(1), [-1.], [1.],
        np.zeros(1), [-.5], [.5]))


def test_tangent_rejection_can_move_inside_exact_unchanged_position_constraint(monkeypatch):
    # Optimizer-only curved constraint: J=-x, s=x^2+(1+y)^2+constant.
    # The local QP is tangent near y=0. Every original trial down to 1/256
    # violates exact s; a strict interior direction must be checked instead.
    edges = [BinghamTransform(name, [0., 0., 0.], -100.*np.diag([0., 1., 1., 1.]),
                              1e-5*np.eye(3)) for name in ('x', 'y')]
    state = JointTransformState.from_independent(edges, command=(0., 0.), stamp=1.)
    inference = SimpleNamespace(body_ids=('x', 'y'),
        actions=({'joint_index': 0, 'axis': [0., 0., 1.]},
                 {'joint_index': 1, 'axis': [0., 0., 1.]}),
        paths={'hand': ('x', 'y'), 'marker': (('x', False),)},
        settings={'root_frame': 'base'})
    urdf = '<robot name="test">'+''.join(
        '<joint name="j{}" type="revolute"><limit lower="-3" upper="3"/></joint>'.format(i)
        for i in range(2))+'</robot>'
    class CurvedConstraintController(MovingController):
        def _query(self, current):
            x, y = current.command
            return SimpleNamespace(mean=np.array([x, 1.+y, 0., 0., 0., 0.]),
                covariance=1e-5*np.eye(6), state_id=current.state_id)
        def _step_metric(self, *args):
            return np.eye(2)
        def _selection_sources(self, *args):
            return {}
    controller = CurvedConstraintController(inference, urdf, [0., 0., 0.], [1., 0., 0., 0.],
        {'max_joint_velocity': 1., 'max_joint_acceleration': 1., 'gradient_max_backtracks': 3})
    result = SimpleNamespace(prior=state,
        stages=(Stage('fused', (WeightedState('source', 1., state),)),), records={'observation_ids': []})
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._relative_density',
                        lambda query, *_: (float(query.mean[0]), {}))
    command, diagnostics = controller.step(result, 1.)
    assert diagnostics['maximum_line_search_backtracks'] >= 8
    original = [item for item in diagnostics['line_search'] if item['direction_kind'] == 'coupled quadratic']
    assert len(original) >= 9
    assert all(not item['position_progress_feasible'] for item in original)
    assert diagnostics['position_descent_correction']['applied']
    assert diagnostics['position_descent_correction']['additional_gradient_evaluations'] == 0
    assert diagnostics['line_search_accepted']
    assert command[0] > .01 and command[1] < -.01
    assert command[0]**2+(1.+command[1])**2 <= 1.+1e-12
    assert -diagnostics['log_expected_density'] < diagnostics['feasible_baseline_objective']
    assert diagnostics['position_progress']['constraint_satisfied']
    assert diagnostics['gradient_evaluation_count'] == 1
