"""Frozen affine feedback, asynchronous continuity and actual path bounds."""
import math
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from probtf_observation_fusion_demo.local_quadratic_tracking import FrozenLocalQuadraticModel
from probtf_observation_fusion_demo.running_gradient import step_running_gradient
from test_running_gradient import packet, tracker


def test_coupled_undamped_solution_matches_independent_normal_mode_formula():
    angle = .37
    basis = np.array([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
    eigenvalues = np.array([.4, 7.])
    hessian = (basis*eigenvalues) @ basis.T
    source, target = np.array([.3, -.7]), np.array([-.2, .4])
    initial, velocity, eta, duration = np.array([.9, -.1]), np.array([-.3, .2]), .8, 2.3
    model = FrozenLocalQuadraticModel(source, hessian @ (source-target), hessian)
    omega = np.sqrt(eta*eigenvalues)
    offset, speed = basis.T @ (initial-target), basis.T @ velocity
    expected_position = target+basis @ (np.cos(omega*duration)*offset+
        np.sin(omega*duration)*speed/omega)
    expected_velocity = basis @ (-omega*np.sin(omega*duration)*offset+
                                 np.cos(omega*duration)*speed)
    actual = model.advance(initial, velocity, duration, eta, 0.)
    np.testing.assert_allclose(actual[0], expected_position, atol=2e-15)
    np.testing.assert_allclose(actual[1], expected_velocity, atol=3e-15)


@pytest.mark.parametrize('gamma', [0., 1e-15, 2., 20.])
def test_actual_interval_partition_does_not_change_fixed_local_model(gamma):
    model = FrozenLocalQuadraticModel([.2, -.1], [2., -3.], [[3., .4], [.4, 1.]])
    initial = (np.array([.7, -.5]), np.array([.1, .2]))
    whole = model.advance(*initial, .83, .9, gamma)
    state = initial
    for dt in (.001, .007, .023, .099, .2, .5):
        state = model.advance(*state, dt, .9, gamma)
    np.testing.assert_allclose(state, whole, atol=3e-15, rtol=4e-15)


def test_critical_damping_uses_the_finite_exact_limit():
    eta, gamma, duration, target = 1., 2., 1.7, .3
    u0, v0 = .8, -.2
    model = FrozenLocalQuadraticModel([0.], [-target], [[1.]])
    y0 = u0-target
    y = math.exp(-duration)*(y0+(v0+y0)*duration)
    v = math.exp(-duration)*(v0-(v0+y0)*duration)
    result = model.advance([u0], [v0], duration, eta, gamma)
    np.testing.assert_allclose(result[0], [target+y], atol=4e-16)
    np.testing.assert_allclose(result[1], [v], atol=4e-16)


def test_tiny_curvature_does_not_subtract_a_huge_equilibrium_command():
    model = FrozenLocalQuadraticModel([0., 0.], [1., -2.], np.eye(2)*1e-250)
    actual = model.advance([.3, -.4], [.1, .2], .01, 1., 3.)
    expected = step_running_gradient([.3, -.4], [.1, .2], [1., -2.], .01, 1., 3.)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-17)
    assert np.all(np.isfinite(actual))


def test_one_packet_with_SPD_curvature_has_a_finite_asymptotic_equilibrium():
    source = np.array([.1, -.1]); curvature = np.array([[4., 1.], [1., 2.]])
    gradient = np.array([1., -2.]); target = source-np.linalg.solve(curvature, gradient)
    state = tracker(eta=.7, gamma=2.)
    assert state.accept(packet(gradient, source, curvature=curvature), 0.)['accepted']
    # No further observations, gradients, source commands or K packets arrive.
    result = state.advance(80.)
    assert result.record['status'] == 'tracking'
    np.testing.assert_allclose(state.position, target, atol=3e-14)
    np.testing.assert_allclose(state.velocity, [0., 0.], atol=2e-14)
    assert result.record['gradient_packet']['source_command'] == source.tolist()
    assert result.record['execution_gradient_model'] == 'local_quadratic'


def test_fixed_positive_quadratic_energy_decreases_without_new_packets():
    curvature = np.array([[2., .7], [.7, 4.]])
    source, gradient = np.array([.2, -.1]), np.array([.3, -.6])
    target = source-np.linalg.solve(curvature, gradient)
    model = FrozenLocalQuadraticModel(source, gradient, curvature)
    u, v = np.array([1., -.9]), np.array([-.2, .3]); eta = .6
    energy = lambda: .5*float(v@v)+.5*eta*float((u-target)@curvature@(u-target))
    previous = energy()
    for dt in (.001, .02, .04, .1, .5, 1., 2., 8.):
        u, v = model.advance(u, v, dt, eta, 1.1)
        assert energy() <= previous+2e-15
        previous = energy()


def test_new_model_keeps_position_velocity_and_uses_current_displacement_from_source():
    state = tracker()
    state.accept(packet(curvature=[[3., .2], [.2, 2.]]), 0.)
    state.advance(.23)
    command, velocity = state.position, state.velocity
    source, gradient, curvature = np.array([.4, -.3]), np.array([-2., 1.]), np.array([[2., .3], [.3, 4.]])
    new_packet = packet(gradient, source, stamp=.1, curvature=curvature,
                        local_model_method='test frozen quadratic')
    adoption = state.accept(new_packet, .23)
    assert adoption['accepted']
    np.testing.assert_array_equal(state.position, command)
    np.testing.assert_array_equal(state.velocity, velocity)
    np.testing.assert_allclose(state.gradient, gradient+curvature @ (command-source), atol=0.)
    zero = state.advance(0.)
    np.testing.assert_array_equal(zero.command, command)
    np.testing.assert_array_equal(zero.velocity, velocity)
    # Elapsed time since observation is not retrospectively integrated.
    independent = FrozenLocalQuadraticModel(source, gradient, curvature)
    expected = independent.advance(command, velocity, .07, state.eta, state.gamma)
    result = state.advance(.07)
    np.testing.assert_array_equal(result.command, expected[0])
    np.testing.assert_array_equal(result.velocity, expected[1])
    np.testing.assert_array_equal(result.record['source_gradient'], gradient)


def test_none_and_exact_zero_curvature_preserve_legacy_arithmetic_bitwise():
    legacy, explicit_zero = tracker(), tracker()
    legacy.accept(packet(), 0.)
    zero = packet(curvature=np.zeros((2, 2)), local_model_method='zero comparison')
    assert explicit_zero.accept(zero, 0.)['accepted']
    for dt in (.000001, .0047, .13, .9):
        a, b = legacy.advance(dt), explicit_zero.advance(dt)
        np.testing.assert_array_equal(a.command, b.command)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        assert a.record['execution_gradient_model'] == b.record['execution_gradient_model'] == 'held_gradient'


@pytest.mark.parametrize('curvature', [[[1.]], [[1., 0.], [0., np.nan]]])
def test_bad_curvature_never_replaces_last_good_packet(curvature):
    state = tracker();good = packet(curvature=np.eye(2))
    state.accept(good, 0.);state.advance(.1)
    command, velocity = state.position, state.velocity
    rejected = state.accept(packet(curvature=curvature), .1)
    assert not rejected['accepted'] and rejected['reason'] == 'invalid_packet'
    assert state.last_packet is good
    np.testing.assert_array_equal(state.position, command)
    np.testing.assert_array_equal(state.velocity, velocity)


@pytest.mark.parametrize('field,value', [('source_stamp', .2), ('source_command', [1., 0.]),
    ('body_state_ids', ['another body']), ('goal_state_ids', ['another goal']),
    ('parameter_density_id', 'another density'), ('selected_K', [1., 2.])])
def test_curvature_cannot_claim_a_different_source_than_its_gradient(field, value):
    state = tracker()
    rejected = state.accept(packet(curvature=np.eye(2),
        local_model_diagnostics={field:value}), .3)
    assert not rejected['accepted'] and rejected['reason'] == 'invalid_packet'
    assert field in rejected['validation_error']


def test_curvature_and_metadata_are_immutable_and_pickleable():
    curvature = np.eye(2)
    original = packet(curvature=curvature,
        local_model_diagnostics={'regularization': {'added':1e-6}})
    curvature[0, 0] = 99.
    revived = pickle.loads(pickle.dumps(original))
    assert revived.as_dict() == original.as_dict()
    assert original.curvature == ((1., 0.), (0., 1.))
    mutable = original.as_dict();mutable['curvature'][0][0] = 77.
    assert original.curvature[0][0] == 1.


def test_curved_path_excursion_is_detected_even_when_both_endpoints_are_inside():
    # u=sin(2*pi*t)/(2*pi) returns to zero at t=1 but exceeds +.1.
    state = tracker(eta=1., gamma=0., initial_velocity=[1., 0.],
        joint_lower=[-.1, -.2], joint_upper=[.1, .2])
    state.accept(packet([0., 0.], curvature=np.diag([(2.*math.pi)**2, 1.])), 0.)
    result = state.advance(1.)
    assert result.record['reason'] == 'joint_position_out_of_range_inside_interval'
    assert result.record['proposed_command'][0] > .1
    np.testing.assert_array_equal(result.command, [0., 0.])


def test_unresolved_path_comparison_is_diagnostic_and_does_not_stop_finite_motion():
    model = FrozenLocalQuadraticModel([0.], [0.], [[1.]])
    check = model.check_joint_interval([0.], [0.], .1, 1., 1., [0.], [1.],
                                      maximum_intervals=0)
    assert check.violation is None and not check.bound_verified
    assert check.unresolved_interval_count == 1
    assert not check.record()['unresolved_comparison_is_a_stop_gate']


def test_unresolved_bound_at_initial_limit_keeps_finite_inward_tracking():
    state = tracker(joint_lower=[0., -100.], joint_upper=[10., 100.])
    state.accept(packet([-1., 0.], curvature=np.eye(2)), 0.)
    result = state.advance(.01)
    assert result.record['status'] == 'tracking' and not state.halted
    assert result.command[0] > 0.
    assert not result.record['local_model_joint_interval_check']['unresolved_comparison_is_a_stop_gate']


def test_frozen_local_model_has_no_input_alias():
    source, gradient, curvature = np.zeros(2), np.ones(2), np.eye(2)
    model = FrozenLocalQuadraticModel(source, gradient, curvature)
    source[:], gradient[:], curvature[:] = 99., 99., 99.
    np.testing.assert_array_equal(model.gradient_at([0., 0.]), [1., 1.])
    with pytest.raises(FrozenInstanceError):
        model.is_zero = True


def test_nonpositive_curvature_diagnostic_does_not_rewrite_a_finite_affine_model():
    # The production producer supplies SPD H. This finite input still defines
    # a linear system; this layer must not eigenvalue-clip or use an audit gate.
    model = FrozenLocalQuadraticModel([0.], [1.], [[-1.]])
    position, velocity = model.advance([0.], [0.], .1, 1., 0.)
    np.testing.assert_allclose(position, [1.-math.cosh(.1)], atol=1e-16)
    np.testing.assert_allclose(velocity, [-math.sinh(.1)], atol=1e-16)
