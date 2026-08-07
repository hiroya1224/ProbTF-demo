from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_estimator as baseline  # noqa: E402
import deterministic_multi_bag_multiple_shooting_estimator as multi  # noqa: E402
import deterministic_multiple_shooting_estimator as strict  # noqa: E402
import deterministic_spline_dynamics_estimator as estimator  # noqa: E402
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.dynamics import FullSixDofPlant  # noqa: E402
from grape_param_estim.geometry import matrix_to_quaternion  # noqa: E402
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    ActuatorState,
    GRAVITY,
    RigidBodyState,
    VehicleParameters,
)
from smooth_command import QuinticSmoothZoh  # noqa: E402
from spline_trajectory import (  # noqa: E402
    PoseSplineEvaluation,
    fit_pose_spline_fixed,
)


class PoseSplineAnalyticTests(unittest.TestCase):
    def test_polynomial_position_derivatives_are_analytic(self):
        time = np.linspace(0.0, 2.0, 81)
        position = np.column_stack(
            (
                1.0 + 2.0 * time - 0.5 * time**2 + 0.2 * time**3,
                -0.3 + 0.4 * time**2 - 0.1 * time**3,
                0.7 - 0.2 * time + 0.05 * time**3,
            )
        )
        quaternion = np.tile((0.0, 0.0, 0.0, 1.0), (time.size, 1))
        spline = fit_pose_spline_fixed(
            time_axis=time,
            sensor_position=position,
            sensor_orientation_xyzw=quaternion,
            body_to_pose_sensor_rotation=np.eye(3),
            knot_spacing_seconds=0.05,
        )
        query = np.linspace(0.05, 1.95, 73)
        value = spline.evaluate(query)
        expected_velocity = np.column_stack(
            (
                2.0 - query + 0.6 * query**2,
                0.8 * query - 0.3 * query**2,
                -0.2 + 0.15 * query**2,
            )
        )
        expected_acceleration = np.column_stack(
            (
                -1.0 + 1.2 * query,
                0.8 - 0.6 * query,
                0.3 * query,
            )
        )
        np.testing.assert_allclose(value.sensor_position, np.column_stack((
            1.0 + 2.0 * query - 0.5 * query**2 + 0.2 * query**3,
            -0.3 + 0.4 * query**2 - 0.1 * query**3,
            0.7 - 0.2 * query + 0.05 * query**3,
        )), atol=2.0e-12)
        np.testing.assert_allclose(
            value.sensor_velocity_world, expected_velocity, atol=2.0e-11
        )
        np.testing.assert_allclose(
            value.sensor_acceleration_world, expected_acceleration, atol=2.0e-10
        )

    def test_composed_rotation_returns_body_omega_and_alpha(self):
        time = np.linspace(0.0, 1.2, 241)
        yaw_rate = 0.7
        roll_rate = -0.45
        matrices = np.asarray(
            [
                Rotation.from_rotvec((0.0, 0.0, yaw_rate * value)).as_matrix()
                @ Rotation.from_rotvec((roll_rate * value, 0.0, 0.0)).as_matrix()
                for value in time
            ]
        )
        quaternion = np.asarray(
            [matrix_to_quaternion(value) for value in matrices]
        )
        spline = fit_pose_spline_fixed(
            time_axis=time,
            sensor_position=np.zeros((time.size, 3)),
            sensor_orientation_xyzw=quaternion,
            body_to_pose_sensor_rotation=np.eye(3),
            knot_spacing_seconds=0.005,
        )
        query = np.linspace(0.05, 1.15, 101)
        evaluation = spline.evaluate(query)
        angle = roll_rate * query
        expected_omega = np.column_stack(
            (
                np.full(query.size, roll_rate),
                yaw_rate * np.sin(angle),
                yaw_rate * np.cos(angle),
            )
        )
        expected_alpha = np.column_stack(
            (
                np.zeros(query.size),
                yaw_rate * roll_rate * np.cos(angle),
                -yaw_rate * roll_rate * np.sin(angle),
            )
        )
        np.testing.assert_allclose(
            evaluation.body_angular_velocity,
            expected_omega,
            rtol=2.0e-4,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            evaluation.body_angular_acceleration,
            expected_alpha,
            rtol=3.0e-3,
            atol=3.0e-4,
        )

    def test_sensor_pose_to_cog_acceleration_matches_exact_kinematics(self):
        time = np.linspace(0.0, 1.0, 1001)
        angle = 0.4 * time + 0.1 * time**2
        omega_z = 0.4 + 0.2 * time
        alpha_z = np.full(time.size, 0.2)
        rotation = Rotation.from_rotvec(
            np.column_stack((np.zeros(time.size), np.zeros(time.size), angle))
        ).as_matrix()
        omega = np.column_stack(
            (np.zeros(time.size), np.zeros(time.size), omega_z)
        )
        alpha = np.column_stack(
            (np.zeros(time.size), np.zeros(time.size), alpha_z)
        )
        cog_position = np.column_stack(
            (0.2 * time**3, -0.3 * time**2, 0.1 * time)
        )
        cog_velocity = np.column_stack(
            (0.6 * time**2, -0.6 * time, np.full(time.size, 0.1))
        )
        cog_acceleration = np.column_stack(
            (1.2 * time, np.full(time.size, -0.6), np.zeros(time.size))
        )
        lever = np.asarray((0.13, -0.04, 0.07))
        sensor_position = cog_position + np.einsum("nij,j->ni", rotation, lever)
        sensor_velocity = cog_velocity + np.einsum(
            "nij,nj->ni", rotation, np.cross(omega, lever)
        )
        sensor_acceleration = cog_acceleration + np.einsum(
            "nij,nj->ni",
            rotation,
            np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever)),
        )
        evaluation = PoseSplineEvaluation(
            time=time,
            sensor_position=sensor_position,
            sensor_velocity_world=sensor_velocity,
            sensor_acceleration_world=sensor_acceleration,
            body_rotation=rotation,
            body_angular_velocity=omega,
            body_angular_acceleration=alpha,
        )
        recovered = estimator.cog_kinematics_from_pose_spline(
            evaluation, lever, np.zeros(3)
        )
        np.testing.assert_allclose(recovered[0], cog_position, atol=2.0e-16)
        np.testing.assert_allclose(recovered[1], cog_velocity, atol=3.0e-16)
        np.testing.assert_allclose(recovered[2], cog_acceleration, atol=4.0e-16)
        finite_difference = np.gradient(
            np.gradient(recovered[0], time, axis=0, edge_order=2),
            time,
            axis=0,
            edge_order=2,
        )
        np.testing.assert_allclose(
            finite_difference[2:-2], cog_acceleration[2:-2], atol=3.0e-6
        )


class ConfigurationAndLagTests(unittest.TestCase):
    def test_missing_result_falls_back_to_nominal_with_mass_override(self):
        with tempfile.TemporaryDirectory() as directory:
            seed = estimator.load_initial_estimate(
                Path(directory) / "missing.json", 0.03, 3.2
            )
        self.assertEqual(seed.source_kind, "nominal_fallback")
        self.assertAlmostEqual(seed.delay_seconds, 0.03)
        self.assertAlmostEqual(seed.selected_mass_kg, 3.2)
        self.assertAlmostEqual(
            seed.physical_coordinate[0],
            math.log(3.2 / VehicleParameters.nominal().mass),
        )

    def test_entrypoint_routes_spline_dynamics_method(self):
        with patch.object(estimator, "main", return_value=0) as selected_main:
            status = entrypoint.main(
                (
                    "--method",
                    "deterministic_spline_dynamics",
                    "--config",
                    "config.json",
                )
            )
        self.assertEqual(status, 0)
        selected_main.assert_called_once_with(["--config", "config.json"])

    def test_spline_dynamics_is_default_entrypoint_method(self):
        self.assertEqual(entrypoint.DEFAULT_METHOD, "deterministic_spline_dynamics")
        with patch.object(estimator, "main", return_value=0) as selected_main:
            status = entrypoint.main(("--config", "config.json"))
        self.assertEqual(status, 0)
        selected_main.assert_called_once_with(["--config", "config.json"])

    def test_smooth_and_strict_zoh_search_recover_known_lag(self):
        sample_time = np.arange(0.0, 2.01, 0.05)
        values = np.sin(9.0 * sample_time)[:, None]
        history = QuinticSmoothZoh(sample_time, values)
        query = np.arange(0.25, 1.75, 0.011)
        truth = 0.037
        width = 0.3
        observed_smooth = np.asarray(
            [history.evaluate(value, truth, width).value for value in query]
        ).ravel()

        def residual(delay):
            return np.asarray(
                [history.evaluate(value, delay[0], width).value for value in query]
            ).ravel() - observed_smooth

        smooth_result = least_squares(residual, (0.02,), bounds=(0.0, 0.1))
        self.assertAlmostEqual(smooth_result.x[0], truth, places=7)
        strict_observed = np.asarray(
            [history.exact_zoh(value, truth) for value in query]
        ).ravel()
        candidates = estimator.smooth.zoh_polish_delays(
            smooth_result.x[0], 0.004, 0.001, (0.0, 0.1)
        )
        costs = [
            np.sum(
                (
                    np.asarray([history.exact_zoh(value, delay) for value in query]).ravel()
                    - strict_observed
                )
                ** 2
            )
            for delay in candidates
        ]
        selected = candidates[int(np.argmin(costs))]
        self.assertLessEqual(abs(selected - truth), 0.001)


@unittest.skipUnless(baseline.DEFAULT_BAG.is_file(), "sample rosbag unavailable")
class RecordedAndSyntheticDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flight = load_flight_data(
            str(baseline.DEFAULT_BAG),
            start_local=19.0,
            end_local=19.4,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        cls.arguments = estimator.create_argument_parser().parse_args(
            (
                "--config",
                "unused.json",
                "--sample-step",
                "0.05",
                "--integration-step",
                "0.025",
                "--spline-cv-folds",
                "3",
            )
        )
        cls.specification = multi.BagSpecification(
            "short", baseline.DEFAULT_BAG, 19.0, 19.4, 1.0
        )
        cls.settings = estimator.SplineSettings((0.1, 0.2), 0.02)
        cls.bag = estimator._build_bag_data(
            cls.specification,
            1.0,
            cls.flight,
            0.02,
            cls.settings,
            cls.arguments,
        )

    @classmethod
    def _synthetic_bag(cls, bag_id, thrust, truth_mass_coordinate):
        base = cls.bag
        time = base.collocation_time
        nominal = VehicleParameters.nominal()
        coordinate = np.zeros(strict.PHYSICAL_DIMENSION)
        coordinate[0] = truth_mass_coordinate
        parameterization = strict.FullyPhysicalInertiaParameterization(nominal)
        decoded = parameterization.decode(
            estimator.continuation._expand_coordinate(coordinate, 0.02)
        )
        parameters = decoded.parameters
        actuator = ActuatorState(
            thrust=np.full(4, thrust), gimbal_angle=np.zeros(4)
        )
        rigid = RigidBodyState(
            position=np.zeros(3),
            orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        wrench = FullSixDofPlant(
            parameters, base.direct_problem.geometry
        ).total_body_wrench(float(time[0]), rigid, actuator)
        body_alpha = np.linalg.solve(parameters.inertia, wrench[3:])
        cog_acceleration = np.asarray((0.0, 0.0, -GRAVITY)) + wrench[:3] / parameters.mass
        lever = base.direct_problem.pose_sensor_position - parameters.cog_offset
        sensor_acceleration = cog_acceleration + np.cross(body_alpha, lever)
        count = time.size
        collocation = PoseSplineEvaluation(
            time=time.copy(),
            sensor_position=np.zeros((count, 3)),
            sensor_velocity_world=np.zeros((count, 3)),
            sensor_acceleration_world=np.tile(sensor_acceleration, (count, 1)),
            body_rotation=np.tile(np.eye(3), (count, 1, 1)),
            body_angular_velocity=np.zeros((count, 3)),
            body_angular_acceleration=np.tile(body_alpha, (count, 1)),
        )
        command_time = np.asarray((time[0] - 1.0, time[-1] + 1.0))
        rotor = QuinticSmoothZoh(command_time, np.full((2, 4), thrust))
        gimbal = QuinticSmoothZoh(command_time, np.zeros((2, 4)))
        specification = multi.BagSpecification(
            bag_id, baseline.DEFAULT_BAG, 19.0, 19.4, 1.0
        )
        return estimator.BagSplineData(
            specification=specification,
            normalized_weight=1.0,
            flight=base.flight,
            direct_problem=base.direct_problem,
            spline_selection=base.spline_selection,
            collocation=collocation,
            rotor_history=rotor,
            gimbal_history=gimbal,
            initial_gimbal=np.zeros(4),
        )

    def test_same_model_synthetic_pose_has_zero_residual_wrench(self):
        bag = self._synthetic_bag("perfect", 5.5, 0.0)
        problem = estimator.SplineDynamicsProblem((bag,), prior_weight=0.0)
        evaluation = problem.evaluate_strict(
            np.zeros(strict.PHYSICAL_DIMENSION), 0.02
        )
        np.testing.assert_allclose(
            evaluation.bag_evaluations[0].residual_body_wrench,
            0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            evaluation.bag_evaluations[0].acceleration_residual,
            0.0,
            atol=2.0e-12,
        )

    def test_synthetic_multi_bag_recovers_shared_mass(self):
        truth = 0.18
        first = self._synthetic_bag("first", 4.8, truth)
        second = self._synthetic_bag("second", 6.1, truth)
        first = estimator.BagSplineData(
            **{**first.__dict__, "normalized_weight": 0.5}
        )
        second = estimator.BagSplineData(
            **{**second.__dict__, "normalized_weight": 0.5}
        )
        problem = estimator.SplineDynamicsProblem((first, second), prior_weight=0.0)

        def evaluate(value):
            coordinate = np.zeros(strict.PHYSICAL_DIMENSION)
            coordinate[0] = value[0]
            return problem.evaluate_strict(coordinate, 0.02).residual

        result = least_squares(evaluate, (0.0,), diff_step=1.0e-5)
        self.assertAlmostEqual(result.x[0], truth, places=8)

    def test_dynamics_residual_analytic_jacobian_matches_finite_difference(self):
        problem = estimator.SplineDynamicsProblem((self.bag,), prior_weight=0.3)
        coordinate = np.zeros(estimator.GLOBAL_DIMENSION)
        coordinate[estimator.DELAY_INDEX] = 0.03
        analytic = problem.evaluate_smooth(coordinate, 0.25)
        numerical = np.empty_like(analytic.jacobian)
        for column in range(estimator.GLOBAL_DIMENSION):
            step = 2.0e-6 if column == estimator.DELAY_INDEX else 2.0e-7
            direction = np.zeros(estimator.GLOBAL_DIMENSION)
            direction[column] = step
            plus = problem.evaluate_smooth(coordinate + direction, 0.25).residual
            minus = problem.evaluate_smooth(coordinate - direction, 0.25).residual
            numerical[:, column] = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            analytic.jacobian, numerical, rtol=2.0e-4, atol=2.0e-5
        )

    def test_forward_rollout_reproduces_synthetic_truth(self):
        coordinate = np.linspace(-0.02, 0.02, strict.PHYSICAL_DIMENSION)
        truth = estimator.forward_rollout(self.bag, coordinate, 0.031)
        repeated = estimator.forward_rollout(self.bag, coordinate, 0.031)
        np.testing.assert_allclose(
            repeated.sensor_position, truth.sensor_position, atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            repeated.sensor_orientation_xyzw,
            truth.sensor_orientation_xyzw,
            atol=0.0,
            rtol=0.0,
        )

    def test_trajectory_report_prioritizes_complete_observed_estimated_comparison(
        self,
    ):
        rollout = estimator.forward_rollout(
            self.bag,
            np.zeros(strict.PHYSICAL_DIMENSION),
            0.02,
        )

        class CapturePdf:
            def __init__(self):
                self.saved = []

            def __enter__(self):
                return self

            def __exit__(self, *_arguments):
                return False

            def savefig(self, figure):
                self.saved.append(figure)

        capture = CapturePdf()
        with patch.object(estimator, "PdfPages", return_value=capture):
            estimator._write_trajectory_pdf(
                Path("unused.pdf"), self.bag, rollout, rollout
            )
        self.assertEqual(len(capture.saved), 9)
        primary_title = capture.saved[0].axes[0].get_title()
        self.assertIn("observed vs estimated", primary_title)
        position_title = capture.saved[1].axes[0].get_title()
        orientation_title = capture.saved[2].axes[0].get_title()
        self.assertIn("observed vs estimated", position_title)
        self.assertIn("observed vs estimated", orientation_title)

    def test_changing_imu_observations_does_not_change_estimator(self):
        problem = estimator.SplineDynamicsProblem((self.bag,), prior_weight=0.5)
        coordinate = np.zeros(estimator.GLOBAL_DIMENSION)
        coordinate[estimator.DELAY_INDEX] = 0.02
        before = problem.evaluate_smooth(coordinate, 0.3)
        direct = self.bag.direct_problem
        original = direct.observations
        changed = baseline.Observations(
            time=original.time,
            sensor_position=original.sensor_position,
            sensor_orientation_xyzw=original.sensor_orientation_xyzw,
            sensor_velocity_world=original.sensor_velocity_world + 100.0,
            angular_velocity_sensor=original.angular_velocity_sensor - 200.0,
            specific_force_sensor=original.specific_force_sensor + 300.0,
        )
        direct.observations = changed
        try:
            after = problem.evaluate_smooth(coordinate, 0.3)
        finally:
            direct.observations = original
        np.testing.assert_array_equal(after.residual, before.residual)
        np.testing.assert_array_equal(after.jacobian, before.jacobian)

    def test_bag_order_does_not_change_joint_objective_or_normal_equations(self):
        first = estimator.BagSplineData(
            **{**self.bag.__dict__, "normalized_weight": 0.4}
        )
        second_base = self._synthetic_bag("synthetic", 5.2, 0.1)
        second = estimator.BagSplineData(
            **{**second_base.__dict__, "normalized_weight": 0.6}
        )
        coordinate = np.zeros(estimator.GLOBAL_DIMENSION)
        coordinate[estimator.DELAY_INDEX] = 0.025
        forward = estimator.SplineDynamicsProblem(
            (first, second), 0.7
        ).evaluate_smooth(coordinate, 0.3)
        reverse = estimator.SplineDynamicsProblem(
            (second, first), 0.7
        ).evaluate_smooth(coordinate, 0.3)
        self.assertAlmostEqual(forward.data_loss, reverse.data_loss, places=14)
        self.assertAlmostEqual(forward.prior_cost, reverse.prior_cost, places=14)
        np.testing.assert_allclose(
            forward.jacobian.T @ forward.jacobian,
            reverse.jacobian.T @ reverse.jacobian,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            forward.jacobian.T @ forward.residual,
            reverse.jacobian.T @ reverse.residual,
            atol=2.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
