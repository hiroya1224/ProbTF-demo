"""Independent analytic checks of the declared final comparison family."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.distributions import BinghamOrientation
import probtf_switch_press_demo.objective as objective_module
from probtf_switch_press_demo.objective import PoseOverlapObjective


LOG_AREA = np.log(2*np.pi**2)


def pose(mean=(0., 0., 0.), covariance=None, parameter=None, **extra):
    return SimpleNamespace(mean=np.asarray(mean, dtype=float),
        covariance=np.eye(3)*.03 if covariance is None else np.asarray(covariance, dtype=float),
        orientation_parameter=np.zeros((4, 4)) if parameter is None else parameter,
        metadata=dict(frame_id="base_link", observation_stamp=2., command=(.1, .2)), **extra)


def goal(mean=(0., 0., 0.), covariance=None, parameters=None, weights=None, **extra):
    parameters = (np.zeros((4, 4)),) if parameters is None else tuple(parameters)
    return SimpleNamespace(mean=np.asarray(mean, dtype=float),
        covariance=np.eye(3)*.02 if covariance is None else np.asarray(covariance, dtype=float),
        orientation_parameters=parameters, weights=(1.,) if weights is None else weights,
        metadata=dict(frame_id="base_link", observation_stamp=2., command=(.1, .2)), **extra)


def paired_parameter(a, b):
    return np.diag([a, a, b, b])


def paired_log_z(a, b):
    # On S3, q0^2+q1^2 is exactly Uniform[0,1]. This one-dimensional
    # integral is an independent formula, not the implementation's solver.
    gap = abs(a-b)
    return LOG_AREA+a if gap == 0. else LOG_AREA+max(a, b)+np.log(-np.expm1(-gap))-np.log(gap)


def test_anisotropic_gaussian_overlap_and_uniform_sphere_normalization():
    rotation = np.array([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    hand = pose([.12, -.03, .07], rotation @ np.diag([.003, .012, .027]) @ rotation.T)
    target = goal([.08, -.01, .11], np.array([[.014, .002, -.001], [.002, .021, .003], [-.001, .003, .009]]))
    objective = PoseOverlapObjective(target)
    score, diagnostics = objective.evaluate(hand)
    expected = multivariate_normal.logpdf(hand.mean, target.mean, hand.covariance+target.covariance)-LOG_AREA
    assert score == pytest.approx(expected, abs=2e-10)
    assert diagnostics["log_orientation_overlap"] == pytest.approx(-LOG_AREA, abs=2e-10)
    assert objective.goal is target
    assert diagnostics["goal_source"] == target.metadata


def test_finite_bingham_overlap_matches_independent_normalizer_ratio():
    hand = pose(parameter=paired_parameter(0., -4.))
    target = goal(parameters=[paired_parameter(0., -7.)])
    score, diagnostics = PoseOverlapObjective(target).evaluate(hand)
    expected_orientation = paired_log_z(0., -11.)-paired_log_z(0., -4.)-paired_log_z(0., -7.)
    expected_position = multivariate_normal.logpdf(hand.mean, target.mean, hand.covariance+target.covariance)
    assert score == pytest.approx(expected_position+expected_orientation, abs=2e-9)
    assert diagnostics["log_hand_normalizer"] == pytest.approx(paired_log_z(0., -4.), abs=2e-9)


def test_mixture_overlaps_are_summed_before_logarithm():
    hand = pose(parameter=paired_parameter(0., -4.))
    target = goal(parameters=[paired_parameter(0., -7.), paired_parameter(-7., 0.)], weights=(.2, .8))
    score, diagnostics = PoseOverlapObjective(target).evaluate(hand)
    common = paired_log_z(0., -4.)+paired_log_z(0., -7.)
    component_logs = np.array([paired_log_z(0., -11.), paired_log_z(-7., -4.)])-common
    expected = logsumexp(np.log(target.weights)+component_logs)
    assert diagnostics["log_orientation_overlap"] == pytest.approx(expected, abs=2e-9)
    assert abs(expected-np.dot(target.weights, component_logs)) > .1
    assert score == pytest.approx(diagnostics["log_position_overlap"]+expected, abs=2e-9)
    assert sum(diagnostics["orientation_component_responsibilities"]) == pytest.approx(1.)


def test_independent_parameter_gauge_shifts_do_not_change_overlap():
    hand_parameter = paired_parameter(0., -4.)
    goal_parameters = [paired_parameter(0., -7.), paired_parameter(-7., 0.)]
    base = PoseOverlapObjective(goal(parameters=goal_parameters, weights=(.3, .7))).evaluate(pose(parameter=hand_parameter))[0]
    shifted = PoseOverlapObjective(goal(parameters=[goal_parameters[0]+81*np.eye(4),
        goal_parameters[1]-23*np.eye(4)], weights=(.3, .7))).evaluate(pose(parameter=hand_parameter+107*np.eye(4)))[0]
    assert shifted == pytest.approx(base, abs=2e-10)


def test_command_dependent_covariance_keeps_its_normalization_and_gradient():
    target = goal(covariance=np.eye(3)*.017)
    objective = PoseOverlapObjective(target)

    def hand_at(command):
        q = command[0]
        return pose([q, 0., 0.], np.eye(3)*(.011+.07*q*q))

    q = .13
    gradient = objective.finite_difference_gradient(hand_at, np.array([q]), step=1e-6)[0]
    variance = .011+.017+.07*q*q
    expected = -3*.07*q/variance-q/variance+.07*q**3/variance**2
    assert gradient == pytest.approx(expected, abs=2e-8)
    narrow, broad = pose(covariance=np.eye(3)*.01), pose(covariance=np.eye(3)*.09)
    narrow_score = objective.evaluate(narrow)[0]
    broad_score = objective.evaluate(broad)[0]
    assert broad_score-narrow_score == pytest.approx(-1.5*np.log((.09+.017)/(.01+.017)), abs=2e-10)


def test_command_dependent_hand_bingham_normalizer_is_not_dropped():
    objective = PoseOverlapObjective(goal(parameters=[paired_parameter(0., -7.)]))

    def hand_at(command):
        return pose(parameter=paired_parameter(0., -command[0]))

    kappa = 4.
    expected = (1/np.expm1(kappa+7)-1/(kappa+7))-(1/np.expm1(kappa)-1/kappa)
    gradient = objective.finite_difference_gradient(hand_at, np.array([kappa]), step=1e-5)[0]
    assert gradient == pytest.approx(expected, abs=2e-8)


def test_orientation_preference_changes_even_when_position_is_unchanged():
    target_parameter = np.diag([0., -20., -35., -50.])
    objective = PoseOverlapObjective(goal(parameters=[target_parameter]))
    aligned = pose(parameter=np.diag([0., -8., -13., -21.]))
    turned = pose(parameter=np.diag([-13., -21., 0., -8.]))
    assert objective.evaluate(aligned)[0] > objective.evaluate(turned)[0]+1.


def test_normalizer_cache_uses_exact_spectral_invariance(monkeypatch):
    objective_module._diagonal_log_normalizer.cache_clear()
    original = objective_module.bingham_log_normalizer
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(objective_module, "bingham_log_normalizer", count)
    hand = pose(parameter=paired_parameter(0., -4.))
    target = goal(parameters=[paired_parameter(0., -7.)])
    score = PoseOverlapObjective(target).evaluate(hand)[0]
    first_calls = len(calls)
    permutation = np.eye(4)[[2, 0, 3, 1]]
    moved_hand = pose(parameter=permutation @ hand.orientation_parameter @ permutation.T+11*np.eye(4))
    moved_goal = goal(parameters=[permutation @ target.orientation_parameters[0] @ permutation.T-9*np.eye(4)])
    assert PoseOverlapObjective(moved_goal).evaluate(moved_hand)[0] == pytest.approx(score, abs=2e-10)
    assert len(calls) == first_calls == 3
    objective_module._diagonal_log_normalizer.cache_clear()


def test_existing_normalizer_refinement_is_reported_without_claiming_an_error_bound():
    objective = PoseOverlapObjective(goal(parameters=[np.diag([0., -110., -170., -230.])]), integration_steps=240)
    diagnostics = objective.refinement_diagnostics(pose(parameter=np.diag([0., -45., -80., -130.])), refined_steps=480)
    assert diagnostics["absolute_log_overlap_change"] < 1e-7
    assert diagnostics["error_bound"] is None


def test_invalid_laws_and_frame_mismatch_are_rejected_without_regularization():
    with pytest.raises(ValueError, match="normalized"):
        PoseOverlapObjective(goal(parameters=[np.zeros((4, 4))]*2, weights=(1., 1.)))
    with pytest.raises(ValueError, match="semidefinite"):
        PoseOverlapObjective(goal(covariance=np.diag([.01, -.01, .01])))
    objective = PoseOverlapObjective(goal(covariance=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="positive definite"):
        objective.evaluate(pose(covariance=np.zeros((3, 3))))
    hand = pose()
    hand.metadata["frame_id"] = "camera"
    with pytest.raises(ValueError, match="same comparison frame"):
        PoseOverlapObjective(goal()).evaluate(hand)


def test_dirac_limit_uses_the_other_density_without_inventing_a_width():
    hand = pose(parameter=np.zeros((4, 4)))
    hand.orientation_parameter = None
    hand.orientation = BinghamOrientation.dirac([1., 0., 0., 0.])
    target = goal(parameters=[paired_parameter(0., -7.)])
    _score, diagnostics = PoseOverlapObjective(target).evaluate(hand)
    assert diagnostics["log_orientation_overlap"] == pytest.approx(-paired_log_z(0., -7.), abs=2e-9)
    point_target = goal(parameters=[None], orientations=(BinghamOrientation.dirac([1., 0., 0., 0.]),))
    with pytest.raises(ValueError, match="Dirac--Dirac"):
        PoseOverlapObjective(point_target).evaluate(hand)
