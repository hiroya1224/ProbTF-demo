from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

from legacies import deterministic_estimator as baseline  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_multi_bag_multiple_shooting_estimator as estimator,
)
from legacies import deterministic_multiple_shooting_estimator as strict  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_smooth_lag_multiple_shooting_estimator as smooth,
)
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402


class MultiBagConfigTests(unittest.TestCase):
    def test_relative_paths_weights_and_initial_delay_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            linked_bag = root / "flight.bag"
            linked_bag.symlink_to(baseline.DEFAULT_BAG.resolve())
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "bags": [
                            {
                                "id": "flight_1",
                                "path": "flight.bag",
                                "start": 19.0,
                                "end": 24.0,
                                "weight": 2.5,
                            }
                        ],
                        "initial_delay_seconds": 0.03,
                    }
                ),
                encoding="utf-8",
            )
            config = estimator.load_multi_bag_config(config_path)
        self.assertEqual(len(config.bags), 1)
        self.assertEqual(config.bags[0].bag_id, "flight_1")
        self.assertEqual(config.bags[0].path, baseline.DEFAULT_BAG.resolve())
        self.assertAlmostEqual(config.bags[0].weight, 2.5)
        self.assertAlmostEqual(config.initial_delay_seconds, 0.03)

    def test_duplicate_or_unsafe_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "bags": [
                            {
                                "id": "../same",
                                "path": str(baseline.DEFAULT_BAG),
                                "start": 19.0,
                                "end": 24.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                estimator.load_multi_bag_config(config_path)

    def test_existing_method_with_config_routes_to_multi_bag_estimator(self):
        with patch.object(estimator, "main", return_value=0) as selected_main:
            status = entrypoint.main(
                (
                    "--method",
                    "deterministic_multiple_shooting",
                    "--config",
                    "config.json",
                )
            )
        self.assertEqual(status, 0)
        selected_main.assert_called_once_with(["--config", "config.json"])


class _LinearSyntheticProblem:
    global_dimension = estimator.GLOBAL_DIMENSION
    variable_dimension = estimator.GLOBAL_DIMENSION
    pose_residual_dimension = 2
    data_residual_dimension = 2 + strict.PHYSICAL_DIMENSION
    continuity_dimension = 0
    prior_weight = 0.0
    prior_scales = np.ones(strict.PHYSICAL_DIMENSION)
    delay = 0.0

    def __init__(self, mass_truth, delay_truth, scale):
        self.mass_truth = float(mass_truth)
        self.delay_truth = float(delay_truth)
        self.scale = float(scale)

    def initial_coordinate(self):
        value = np.zeros(self.global_dimension)
        value[estimator.DELAY_INDEX] = 0.01
        return value

    def bounds(self, lower, upper):
        return np.asarray(lower), np.asarray(upper)

    def evaluate(self, coordinate):
        value = np.asarray(coordinate, dtype=float)
        residual = np.zeros(self.data_residual_dimension)
        residual[0] = self.scale * (value[0] - self.mass_truth)
        residual[1] = (value[estimator.DELAY_INDEX] - self.delay_truth) / self.scale
        jacobian = np.zeros((self.data_residual_dimension, self.global_dimension))
        jacobian[0, 0] = self.scale
        jacobian[1, estimator.DELAY_INDEX] = 1.0 / self.scale
        empty = np.empty(0)
        return strict.ProblemEvaluation(
            data_residual=residual,
            data_jacobian=jacobian,
            continuity_residual=empty,
            continuity_jacobian=np.empty((0, self.global_dimension)),
            sensor_position=np.zeros((1, 3)),
            sensor_orientation_xyzw=np.asarray(((0.0, 0.0, 0.0, 1.0),)),
            sensor_velocity_world=np.zeros((1, 3)),
            angular_velocity_sensor=np.zeros((1, 3)),
            specific_force_sensor=np.zeros((1, 3)),
            decoded=SimpleNamespace(parameters=SimpleNamespace()),
        )

    def full_rollout(self, coordinate):
        evaluation = self.evaluate(coordinate)
        return (
            np.zeros((1, 3)),
            np.asarray(((0.0, 0.0, 0.0, 1.0),)),
            evaluation.data_residual[:2],
        )


class SyntheticJointRecoveryTests(unittest.TestCase):
    def _joint(self, reverse=False):
        specifications = (
            estimator.BagSpecification("a", Path("a.bag"), 0.0, 1.0, 1.0),
            estimator.BagSpecification("b", Path("b.bag"), 0.0, 1.0, 3.0),
        )
        problems = (
            _LinearSyntheticProblem(0.4, 0.06, 1.0),
            _LinearSyntheticProblem(0.4, 0.06, 2.0),
        )
        if reverse:
            specifications = tuple(reversed(specifications))
            problems = tuple(reversed(problems))
        return estimator.JointMultipleShootingProblem(
            estimator._make_blocks(specifications, problems)
        )

    def test_two_synthetic_bags_recover_shared_parameters(self):
        problem = self._joint()
        initial = problem.initial_coordinate()
        result = least_squares(
            lambda value: problem.evaluate(value).data_residual,
            initial,
            jac=lambda value: problem.evaluate(value).data_jacobian,
        )
        self.assertAlmostEqual(result.x[0], 0.4, places=7)
        self.assertAlmostEqual(result.x[estimator.DELAY_INDEX], 0.06, places=7)

    def test_bag_order_does_not_change_shared_solution(self):
        solutions = []
        for reverse in (False, True):
            problem = self._joint(reverse)
            result = least_squares(
                lambda value: problem.evaluate(value).data_residual,
                problem.initial_coordinate(),
                jac=lambda value: problem.evaluate(value).data_jacobian,
            )
            solutions.append(result.x[: estimator.GLOBAL_DIMENSION])
        np.testing.assert_allclose(solutions[0], solutions[1], atol=1.0e-12)


@unittest.skipUnless(baseline.DEFAULT_BAG.is_file(), "sample rosbag unavailable")
class RecordedFlightJointProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flight = load_flight_data(
            str(baseline.DEFAULT_BAG),
            start_local=19.0,
            end_local=19.2,
            include_fc_specific_force=True,
            compute_sha256=False,
        )

    def _problem(self):
        return smooth.SmoothLagMultipleShootingProblem(
            flight=self.flight,
            sample_step=0.05,
            integration_step=0.025,
            initial_delay=0.01,
            width_fraction=0.5,
            segment_duration=0.1,
            body_displacement_scale=1.0,
            prior_weight=1.0,
            node_position_bound=2.0,
            node_orientation_bound=1.5,
            node_velocity_bound=5.0,
            node_angular_velocity_bound=10.0,
        )

    def _specification(self, bag_id, weight=1.0):
        return estimator.BagSpecification(
            bag_id,
            baseline.DEFAULT_BAG,
            19.0,
            19.2,
            weight,
        )

    def test_one_bag_joint_evaluation_matches_single_bag_problem(self):
        single = self._problem()
        joint = estimator.JointMultipleShootingProblem(
            (
                estimator.BagShootingBlock(
                    self._specification("single"),
                    1.0,
                    single,
                ),
            )
        )
        single_evaluation = single.evaluate(single.initial_coordinate())
        joint_evaluation = joint.evaluate(joint.initial_coordinate())
        np.testing.assert_allclose(
            joint_evaluation.data_residual,
            single_evaluation.data_residual,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            joint_evaluation.data_jacobian,
            single_evaluation.data_jacobian,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            joint_evaluation.continuity_residual,
            single_evaluation.continuity_residual,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            joint_evaluation.continuity_jacobian,
            single_evaluation.continuity_jacobian,
            atol=1.0e-14,
        )

    def test_soft_prior_is_present_once_for_two_bags(self):
        problems = (self._problem(), self._problem())
        specifications = (
            self._specification("first"),
            self._specification("second"),
        )
        joint = estimator.JointMultipleShootingProblem(
            estimator._make_blocks(specifications, problems)
        )
        coordinate = joint.initial_coordinate()
        coordinate[0] = 0.3
        evaluation = joint.evaluate(coordinate)
        self.assertEqual(
            evaluation.data_residual.size,
            2 * problems[0].pose_residual_dimension
            + strict.PHYSICAL_DIMENSION,
        )
        expected_prior = np.zeros(strict.PHYSICAL_DIMENSION)
        expected_prior[0] = 0.3 / joint.prior_scales[0]
        np.testing.assert_allclose(
            evaluation.data_residual[-strict.PHYSICAL_DIMENSION :],
            expected_prior,
            atol=1.0e-14,
        )

    def test_other_bag_node_columns_are_exactly_zero(self):
        problems = (self._problem(), self._problem())
        joint = estimator.JointMultipleShootingProblem(
            estimator._make_blocks(
                (self._specification("first"), self._specification("second")),
                problems,
            )
        )
        evaluation = joint.evaluate(joint.initial_coordinate())
        first_rows = slice(0, problems[0].pose_residual_dimension)
        second_rows = slice(
            problems[0].pose_residual_dimension,
            problems[0].pose_residual_dimension
            + problems[1].pose_residual_dimension,
        )
        np.testing.assert_array_equal(
            evaluation.data_jacobian[first_rows, joint.node_slices[1]],
            0.0,
        )
        np.testing.assert_array_equal(
            evaluation.data_jacobian[second_rows, joint.node_slices[0]],
            0.0,
        )

    def test_sequential_projection_restores_each_bag_exactly(self):
        problems = (self._problem(), self._problem())
        joint = estimator.JointMultipleShootingProblem(
            estimator._make_blocks(
                (self._specification("first"), self._specification("second")),
                problems,
            )
        )
        global_coordinate = joint.initial_coordinate()[
            : estimator.GLOBAL_DIMENSION
        ].copy()
        global_coordinate[0] = 0.05
        global_coordinate[estimator.DELAY_INDEX] = 0.012
        coordinate = joint.continuous_coordinate(global_coordinate)
        evaluation = joint.evaluate(coordinate)
        self.assertLess(
            float(np.max(np.abs(evaluation.continuity_residual))),
            1.0e-11,
        )
        rollouts = joint.full_rollouts(global_coordinate)
        for bag_evaluation, rollout, problem in zip(
            evaluation.bag_evaluations,
            rollouts,
            problems,
        ):
            np.testing.assert_allclose(
                bag_evaluation.data_residual[: problem.pose_residual_dimension],
                rollout.residual,
                rtol=0.0,
                atol=1.0e-11,
            )

    def test_shared_and_bag_local_jacobian_columns_match_central_difference(self):
        problems = (self._problem(), self._problem())
        joint = estimator.JointMultipleShootingProblem(
            estimator._make_blocks(
                (self._specification("first"), self._specification("second")),
                problems,
            )
        )
        coordinate = joint.initial_coordinate()
        evaluation = joint.evaluate(coordinate)
        analytic_matrix = np.vstack(
            (evaluation.data_jacobian, evaluation.continuity_jacobian)
        )

        def residual(value):
            current = joint.evaluate(value)
            return np.concatenate(
                (current.data_residual, current.continuity_residual)
            )

        columns = (0, estimator.DELAY_INDEX, joint.node_slices[0].start)
        step = 1.0e-7
        for column in columns:
            positive = coordinate.copy()
            negative = coordinate.copy()
            positive[column] += step
            negative[column] -= step
            numerical = (residual(positive) - residual(negative)) / (2.0 * step)
            np.testing.assert_allclose(
                analytic_matrix[:, column],
                numerical,
                rtol=5.0e-4,
                atol=3.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
