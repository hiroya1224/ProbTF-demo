"""Coupled IK steps respect position progress without changing candidate laws."""

import numpy as np
import pytest

from probtf_observation_fusion_demo.moving_control import _bounded_gradient_direction


def _solve(metric, gradient, lower, upper, *, baseline=None, progress=None):
    gradient = np.asarray(gradient, dtype=float)
    baseline = np.zeros_like(gradient) if baseline is None else np.asarray(baseline, dtype=float)
    progress = np.zeros_like(gradient) if progress is None else np.asarray(progress, dtype=float)
    direction, record = _bounded_gradient_direction(np.asarray(metric, dtype=float),
        gradient, baseline, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), progress)
    assert np.all(baseline+direction >= np.asarray(lower)-1e-11)
    assert np.all(baseline+direction <= np.asarray(upper)+1e-11)
    assert gradient @ direction <= 1e-11
    assert progress @ direction <= 1e-11
    return direction, record


def test_box_quadratic_retains_joint_coupling_at_an_active_limit():
    # Once x reaches .5, the coupled optimum has y=-3*x/4. Clipping
    # the unbounded Newton solution separately would instead give y=-.5.
    direction, record = _solve([[4., 3.], [3., 4.]], [-8., 0.], [-.5, -.5], [.5, .5])
    np.testing.assert_allclose(direction, [.5, -.375], atol=2e-7)
    assert record['accepted']
    assert not np.allclose(direction, [.5, -.5])


def test_position_progress_halfspace_changes_the_coupled_box_optimum():
    direction, record = _solve([[4., 3.], [3., 4.]], [-8., 0.], [-.5, -.5], [.5, .5],
                                progress=[1., 1.])
    np.testing.assert_allclose(direction, [.5, -.5], atol=2e-7)
    assert record['accepted']
    assert abs(record['linearized_position_change']) < 1e-10


def test_progress_blocks_a_density_improvement_that_only_moves_away():
    direction, record = _solve([[1.]], [-1.], [-.5], [.5], progress=[1.])
    np.testing.assert_allclose(direction, [0.], atol=1e-11)
    assert record['objective_slope'] >= -1e-11


def test_mandatory_braking_baseline_is_the_origin_of_the_local_step():
    direction, record = _solve([[2., 0.], [0., 3.]], [-1., 2.], [.01, -.03], [.04, .02],
        baseline=[.01, -.02], progress=[-1., 1.])
    np.testing.assert_allclose(direction, [.03, -.01], atol=2e-7)
    assert record['accepted']


def test_a_locked_coordinate_remains_locked_in_the_coupled_problem():
    direction, record = _solve([[4., 3.], [3., 4.]], [-8., -1.], [0., -.5], [0., .5])
    np.testing.assert_allclose(direction, [0., .25], atol=2e-7)
    assert record['accepted']


@pytest.mark.parametrize('objective_scale', [1., 1e6])
def test_objective_units_do_not_change_the_bounded_minimizer(objective_scale):
    direction, _ = _solve(objective_scale*np.array([[4., 3.], [3., 4.]]),
        objective_scale*np.array([-8., 0.]), [-.5, -.5], [.5, .5])
    np.testing.assert_allclose(direction, [.5, -.375], atol=2e-7)


def test_recorded_imu_marker_divergence_is_excluded_by_the_progress_constraint(monkeypatch):
    # Frozen large-board source ccb633159ae05dff91f5b9c7672ce863c131c3eb33d8940411033f5237848d01,
    # IMU+marker at 27.8003578186 s. Metric and full-J gradient are the actual
    # logged values; the position-expectation derivative was recomputed from
    # the exact recorded joint law and the same frozen estimated-K action.
    # The old elementwise-clipped direction increased s from .0057512923
    # to .0089935286 m^2 despite improving J from 10252.60 to 6687.30.
    metric = np.array([
        [984432.8017840417, 229015.93693560443, 187613.92970088412, 205659.3795686655, 82888.23524947569, 93782.83535739427],
        [229015.93693560438, 1286285.96646128, 796036.9526925656, 39000.30667499321, 527669.9691492007, 20554.24107649445],
        [187613.92970088404, 796036.9526925656, 702346.3551857498, 29015.93143336895, 472271.2965277531, 13682.778890874863],
        [205659.37956866543, 39000.30667499322, 29015.931433368958, 363685.54759036325, 15391.037426236227, 341854.492841552],
        [82888.23524947565, 527669.9691492008, 472271.29652775323, 15391.037426236226, 338951.05955628847, 10616.799096037788],
        [93782.83535739426, 20554.24107649445, 13682.778890874863, 341854.49284155195, 10616.799096037787, 334352.9851966778]])
    gradient = np.array([-43251.99332597549, -27798.16021393344, 81646.34529384784,
                          73617.06219016924, 23279.825705867548, -10202.528497757157])
    progress = np.array([.11510181958333114, -.016996107234448776, .02003960110821129,
                         .011154473609966546, .011621829187555275, .00001142503357545732])
    old_direction = np.clip(-np.linalg.solve(metric, gradient), -.024, .024)
    assert progress @ old_direction > .0018
    before = [value.copy() for value in (metric, gradient, progress)]
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The six-variable deterministic quadratic must not sample candidates.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    direction, record = _solve(metric, gradient, [-.024]*6, [.024]*6, progress=progress)
    assert record['accepted']
    assert gradient @ direction < -100.
    assert not np.allclose(direction, old_direction)
    for actual, expected in zip((metric, gradient, progress), before):
        np.testing.assert_array_equal(actual, expected)
