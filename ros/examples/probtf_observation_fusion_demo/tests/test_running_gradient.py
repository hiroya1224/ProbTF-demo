"""Mathematical tracking, asynchronous continuity and explicit failure tests."""

from dataclasses import FrozenInstanceError
import math
import pickle

import numpy as np
import pytest
from scipy.linalg import expm

from probtf_observation_fusion_demo.running_gradient import (
    GradientPacket, RunningGradientTracker, stable_damped_coefficients,
    step_running_gradient)


def packet(gradient=(1., -2.), command=(0., 0.), stamp=0., **changes):
    values = dict(gradient=gradient, source_command=command, source_stamp=stamp,
        body_state_ids=['body_7'], goal_state_ids=['goal_3'], parameter_density_id='density_4',
        selected_K=[100., 200.], objective_at_source=2., computation_seconds=.3,
        finite_difference_diagnostics={'steps': np.array([.001, .001]),
                                       'nested': {'scheme': ['central', 'forward']}},
        snapshot_update_count=4, command_interval_id=17, observation_source_stamps=[-.1],
        packet_id='gradient_8')
    values.update(changes)
    return GradientPacket(**values)


def tracker(**changes):
    options = dict(eta=.6, gamma=2., joint_lower=[-100., -100.], joint_upper=[100., 100.])
    options.update(changes)
    return RunningGradientTracker([0., 0.], **options)


@pytest.mark.parametrize('gamma,dt', [(0., .1), (2., .1), (1e-15, .3),
                                      (1e-6, 2e-3), (1., 100.), (1e4, 1.)])
def test_constant_input_exact_solution_matches_independent_matrix_exponential(gamma, dt):
    # The augmented linear system independently integrates u, v and constant g.
    eta, initial, velocity, gradient = .7, np.array([.3, -.8]), np.array([.2, .1]), np.array([1., -2.])
    system = np.array([[0., 1., 0.], [0., -gamma, -eta], [0., 0., 0.]])
    propagator = expm(system*dt)
    expected = np.array([propagator@np.array([u, v, g])
                         for u, v, g in zip(initial, velocity, gradient)])
    actual_u, actual_v = step_running_gradient(initial, velocity, gradient, dt, eta, gamma)
    np.testing.assert_allclose(actual_u, expected[:, 0], rtol=2e-13, atol=3e-13)
    np.testing.assert_allclose(actual_v, expected[:, 1], rtol=2e-13, atol=3e-13)


def test_tiny_damping_limit_retains_quadratic_displacement():
    for gamma in (0., 1e-300, 1e-20, 1e-12):
        decay, a_coeff, b_coeff = stable_damped_coefficients(gamma, .025)
        assert math.isclose(decay, 1., abs_tol=3e-14)
        assert math.isclose(a_coeff, .025, rel_tol=2e-14)
        assert math.isclose(b_coeff, .5*.025**2, rel_tol=2e-14)


def test_held_gradient_is_independent_of_irregular_tick_partition():
    initial, velocity, gradient = [1., -1.], [.3, -.4], [2., 3.]
    whole = step_running_gradient(initial, velocity, gradient, .7, .02, .7)
    current = (initial, velocity)
    for dt in (.001, .125, .2, .374):
        current = step_running_gradient(*current, gradient, dt, .02, .7)
    np.testing.assert_allclose(current, whole, atol=4e-15)


def test_fixed_quadratic_converges_and_matches_known_feedback_system():
    # At 1 ms a sample-and-hold gradient approximates the known continuously
    # evaluated damped quadratic. This is an error bound, not an exact identity.
    eta, gamma, target, dt, duration = 1.2, 1.8, np.array([.7, -.4]), .001, 12.
    state = tracker(eta=eta, gamma=gamma)
    system = np.array([[0., 1.], [-eta, -gamma]])
    expected = np.array([expm(system*duration)@np.array([-goal, 0.])
                         for goal in target])
    for index in range(int(duration/dt)):
        state.accept(packet(state.position-target, state.position, index*dt), index*dt)
        result = state.advance(dt)
        assert not result.record['halted']
    np.testing.assert_allclose(state.position, target+expected[:, 0], atol=5e-6)
    np.testing.assert_allclose(state.velocity, expected[:, 1], atol=8e-6)
    assert np.linalg.norm(state.position-target) < 3e-5


def test_ideal_continuous_quadratic_energy_decreases_distinct_from_delayed_input():
    eta, gamma, displacement, velocity = .7, 1.3, 1.4, -.2
    system = np.array([[0., 1.], [-eta, -gamma]])
    energies = []
    for stamp in np.linspace(0., 8., 81):
        delta, speed = expm(system*stamp)@np.array([displacement, velocity])
        energies.append(.5*eta*delta**2+.5*speed**2)
    assert np.all(np.diff(energies) <= 1e-14)
    # Delayed stale input need not lower a fixed quadratic energy each tick.
    u, v = step_running_gradient([0.], [0.], [2.], .2, eta, gamma)
    assert .5*eta*u[0]**2+.5*v[0]**2 > 0.


def test_goal_jump_changes_acceleration_without_resetting_command_or_velocity():
    state = tracker()
    state.accept(packet(), 0.)
    state.advance(.2)
    before_u, before_v = state.position, state.velocity
    old_acceleration = -state.gamma*before_v-state.eta*state.gradient
    adopted = state.accept(packet((-10., 8.), state.position, .2, packet_id='jump'), .2)
    assert adopted['accepted']
    np.testing.assert_array_equal(state.position, before_u)
    np.testing.assert_array_equal(state.velocity, before_v)
    assert np.linalg.norm(-state.gamma*state.velocity-state.eta*state.gradient-old_acceleration) > 1.
    tiny = state.advance(1e-7)
    np.testing.assert_allclose((np.array(tiny.command)-before_u)/1e-7, before_v, atol=4e-7)
    assert np.max(np.abs(np.array(tiny.velocity)-before_v)) < 1e-6


def test_no_result_waits_without_fabricating_a_zero_gradient():
    state = tracker()
    for dt in (.1, .2, .4):
        result = state.advance(dt)
        assert result.record['status'] == 'waiting_for_initial_gradient'
        assert result.record['gradient'] is None
        assert result.record['gradient_packet'] is None
    np.testing.assert_array_equal(state.position, [0., 0.])
    assert state.gradient is None


def test_async_hold_keeps_original_provenance_and_exposes_age_and_motion_distance():
    state = tracker()
    source = packet(stamp=0.)
    assert state.accept(source, .3, snapshot_update_count=7)['accepted']
    result = state.advance(2., stamp=2.3, snapshot_update_count=10)
    assert result.record['status'] == 'tracking'
    assert result.record['gradient_source_age_s'] == 2.3
    assert result.record['gradient_snapshot_update_lag'] == 6
    # At the next tick the distance is measured from the packet evaluation point.
    result = state.advance(.1, stamp=2.4, snapshot_update_count=11)
    assert result.record['gradient_source_command_distance_l2'] > 0.
    assert result.record['gradient_packet']['source_stamp'] == 0.
    assert result.record['gradient_packet']['selected_K'] == [100., 200.]
    assert result.record['gradient_packet']['command_interval_id'] == 17
    assert result.record['gradient_distance_evaluated_at_command'] == list(result.command)
    assert result.record['gradient_source_command_distance_l2'] == np.linalg.norm(result.command)
    assert state.last_packet is source


def test_packet_is_deeply_immutable_pickleable_and_has_no_input_alias():
    gradient = np.array([1., -2.])
    metadata = {'steps': [1., 2.], 'nested': {'x': [3.]}}
    source = packet(gradient=gradient, finite_difference_diagnostics=metadata)
    gradient[0] = 99.
    metadata['nested']['x'][0] = 88.
    assert source.gradient == (1., -2.)
    assert source.finite_difference_diagnostics['nested']['x'] == (3.,)
    with pytest.raises(FrozenInstanceError):
        source.source_stamp = 8.
    with pytest.raises(TypeError):
        source.finite_difference_diagnostics['new'] = 8.
    revived = pickle.loads(pickle.dumps(source))
    assert revived.as_dict() == source.as_dict()
    output = source.as_dict()
    output['finite_difference_diagnostics']['nested']['x'][0] = -1.
    assert source.finite_difference_diagnostics['nested']['x'] == (3.,)


@pytest.mark.parametrize('changes,reason', [
    ({'gradient': [np.nan, 1.]}, 'invalid_packet'),
    ({'gradient': [np.inf, 1.]}, 'invalid_packet'),
    ({'gradient': None, 'status': 'failed', 'reason': 'branch failure'}, 'evaluation_failed'),
    ({'source_stamp': 2.}, 'source_time_is_in_future'),
    ({'objective_at_source': np.nan}, 'invalid_packet'),
])
def test_bad_result_is_explicitly_rejected_and_never_replaces_last_good_input(changes, reason):
    state = tracker()
    accepted = packet()
    state.accept(accepted, 0.)
    state.advance(.1)
    before_u, before_v = state.position, state.velocity
    record = state.accept(packet(**changes), .1)
    assert not record['accepted']
    assert record['reason'] == reason
    assert state.last_packet is accepted
    np.testing.assert_array_equal(state.position, before_u)
    np.testing.assert_array_equal(state.velocity, before_v)
    assert not state.halted


def test_age_threshold_is_a_diagnostic_and_never_stops_finite_updates():
    unconstrained = tracker()
    assert unconstrained.accept(packet(), 100.)['accepted']
    assert unconstrained.advance(.01, stamp=100.01).record['status'] == 'tracking'
    bounded = tracker(max_source_age=.5)
    adoption=bounded.accept(packet(), 1.)
    assert adoption['accepted']
    assert adoption['numerical_diagnostic_warning']=='source_age_exceeded'
    assert bounded.accept(packet(), .1)['accepted']
    bounded.advance(.2)
    safe = bounded.position
    result = bounded.advance(.4)
    assert result.record['numerical_diagnostic_warning']=='source_age_exceeded'
    assert result.record['status']=='tracking' and not result.record['stop_requested']
    assert not bounded.halted
    assert np.linalg.norm(np.asarray(result.command)-safe)>0.


@pytest.mark.parametrize('options,updates,expected', [
    ({'max_command_distance': .001}, None, 'source_command_distance_exceeded'),
    ({'max_snapshot_update_lag': 1}, 7, 'snapshot_update_lag_exceeded'),
])
def test_optional_distance_and_update_lag_thresholds_are_observable(options, updates, expected):
    state = tracker(**options)
    assert state.accept(packet(), 0.)['accepted']
    state.advance(.1)
    continued = state.advance(.01, snapshot_update_count=updates)
    assert continued.record['numerical_diagnostic_warning'] == expected
    assert continued.record['status']=='tracking' and not state.halted


def test_joint_limit_stops_at_last_safe_position_and_records_unclipped_rejected_command():
    state = tracker(eta=10., gamma=0., joint_lower=[-.1, -.1], joint_upper=[.1, .1])
    state.accept(packet(), 0.)
    result = state.advance(1.)
    assert result.record['status'] == 'fail_safe_stop'
    assert result.record['reason'] == 'joint_position_out_of_range'
    assert result.record['proposed_command'] == [-5., 10.]
    assert result.record['last_safe_command'] == [0., 0.]
    np.testing.assert_array_equal(result.command, [0., 0.])
    assert state.advance(.1).record['status'] == 'halted'
    assert not state.accept(packet(), 0.)['accepted']


def test_joint_limits_check_interior_turning_point_even_if_endpoint_returns_inside():
    # u(t)=t-t^2, returning to zero at t=1 but reaching +.25 in between.
    state = tracker(eta=1., gamma=0., joint_lower=[-.1, -.1], joint_upper=[.2, .2],
                    initial_velocity=[1., 0.])
    state.accept(packet([2., 0.]), 0.)
    stopped = state.advance(1.)
    assert stopped.record['reason'] == 'joint_position_out_of_range_inside_interval'
    assert stopped.record['violating_interval_offset_s'] == .5
    assert stopped.record['proposed_command'][0] == .25
    assert stopped.record['last_safe_velocity'] == [1., 0.]


def test_damped_turning_point_is_checked_before_returning_to_an_in_range_endpoint():
    state = tracker(eta=1., gamma=2., joint_lower=[-.2, -.2], joint_upper=[.15, .2],
                    initial_velocity=[1., 0.])
    state.accept(packet([2., 0.]), 0.)
    endpoint, _ = step_running_gradient([0., 0.], [1., 0.], [2., 0.], 1., 1., 2.)
    assert -.2 < endpoint[0] < .15
    stopped = state.advance(1.)
    assert stopped.record['reason'] == 'joint_position_out_of_range_inside_interval'
    assert stopped.record['violating_interval_offset_s'] == math.log(2.)/2.
    assert stopped.record['proposed_command'][0] > .15


def test_numerical_failure_is_a_stop_notification_with_no_nonfinite_command():
    state = tracker(eta=1e300, gamma=0.)
    state.accept(packet([1e300, 0.]), 0.)
    stopped = state.advance(1.)
    assert stopped.record['reason'] == 'nonfinite_or_invalid_integration'
    assert stopped.record['stop_requested']
    assert np.all(np.isfinite(stopped.command))
    assert np.all(np.isfinite(stopped.velocity))
