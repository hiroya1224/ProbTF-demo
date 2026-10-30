from pathlib import Path
import sys

import numpy as np

AUTOMATION = Path(__file__).resolve().parents[1]
if str(AUTOMATION) not in sys.path:
    sys.path.insert(0, str(AUTOMATION))
import evaluate_rest_pid_reference as comparison

FAILURE = np.array([[3., .1, 1.], [5., 1., 2.5], [20., 1., 8.], [4., 1., 2.]])
SUCCESS = np.array([[4., .1, 2.], [5., 1., 2.5], [13., 1., 20.], [6., 1., 2.]])


def test_failure_point_with_success_rest_changes_only_the_selected_triple():
    failure, success = FAILURE.copy(), SUCCESS.copy()
    actual = comparison.controller_for_point(failure, success, "roll_pitch", "failure", "success")
    np.testing.assert_array_equal(actual, [[4., .1, 2.], [5., 1., 2.5], [20., 1., 8.], [6., 1., 2.]])
    np.testing.assert_array_equal(failure, FAILURE)
    np.testing.assert_array_equal(success, SUCCESS)


def test_success_point_with_failure_rest_does_not_restore_other_success_gains():
    actual = comparison.controller_for_point(FAILURE, SUCCESS, "roll_pitch", "success", "failure")
    np.testing.assert_array_equal(actual, [[3., .1, 1.], [5., 1., 2.5], [13., 1., 20.], [4., 1., 2.]])


def test_coincident_z_points_have_identical_controllers_within_each_context():
    for context, expected in (("success", SUCCESS), ("failure", FAILURE)):
        first = comparison.controller_for_point(FAILURE, SUCCESS, "z", "failure", context)
        second = comparison.controller_for_point(FAILURE, SUCCESS, "z", "success", context)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, expected)
    unique = {tuple(comparison.controller_for_point(FAILURE, SUCCESS, group, point, context).ravel())
              for group in comparison.GROUPS for point in comparison.POINTS for context in comparison.CONTEXTS}
    assert len(unique) == 8


def test_replay_compares_original_positions_and_nonfinite_validity():
    same = comparison.replay_diagnostic([.9, None, 1.1], [.9, None, 1.1])
    assert same == {"same_ordered_radii": True, "same_validity": True, "maximum_finite_absolute_error": 0.}
    swapped = comparison.replay_diagnostic([.9, 1.1, None], [1.1, .9, None])
    assert not swapped["same_ordered_radii"] and swapped["same_validity"]
