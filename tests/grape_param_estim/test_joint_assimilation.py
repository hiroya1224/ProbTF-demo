import unittest
from pathlib import Path
import tempfile

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.joint_assimilation import (
    ACTUATOR_STATE_DIMENSION,
    SHARED_STATIC_DIMENSION,
    JointBagProblem,
    JointIEnKSConfig,
    JointWeakConstraintIEnKSQ,
    JointWeakConstraintPrior,
    JointWeakConstraintProblem,
    assimilate_joint_flights,
    assimilation_run_manifest,
    initial_prior_forecast_manifest,
    prepare_joint_flight,
    write_joint_assimilation_payloads,
)
from grape_param_estim.artifact_io import (
    begin_bundle,
    load_assimilation_run,
    mark_bundle_complete,
)
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.strong_constraint import PARAMETER_OFFSET
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.synthetic import full_six_dof_reference
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    PoseObservations,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)
from grape_param_estim.weak_constraint import WeakConstraintProblem
from grape_param_estim.real_rosbag import build_real_flight_episode
try:
    from .test_flight_inspection import _fake_arrays
except ImportError:  # nosetests imports this directory as top-level modules.
    from test_flight_inspection import _fake_arrays


def _synthetic_joint_bag(
    bag_id,
    phase,
    position_bias,
    initial_pose_delta,
    initial_velocity_delta,
    truth_parameters,
    truth_delay,
    truth_residual_wrench,
):
    """Build one observation-reset-free full closed-loop joint episode."""

    times = np.linspace(0.0, 0.16, 5)
    references = tuple(
        ReferenceState(
            position=value.position + np.asarray(position_bias, dtype=float),
            linear_velocity=value.linear_velocity,
            linear_acceleration=value.linear_acceleration,
            rpy=value.rpy,
            angular_velocity=value.angular_velocity,
            angular_acceleration=value.angular_acceleration,
        )
        for value in full_six_dof_reference(times + float(phase))
    )
    pose_delta = np.asarray(initial_pose_delta, dtype=float)
    velocity_delta = np.asarray(initial_velocity_delta, dtype=float)
    initial_reference = references[0]
    initial_state = RigidBodyState(
        position=initial_reference.position + pose_delta[:3],
        orientation_xyzw=matrix_to_quaternion(
            euler_xyz_to_matrix(initial_reference.rpy + pose_delta[3:])
        ),
        linear_velocity=(
            initial_reference.linear_velocity + velocity_delta[:3]
        ),
        angular_velocity=(
            initial_reference.angular_velocity + velocity_delta[3:]
        ),
    )
    controller_configuration = ControllerConfig.grape()
    controller_state = initial_controller_state(
        controller_configuration, trim_hover=True
    )
    geometry = GrapeGeometry.grape()
    controller_parameters = VehicleParameters.nominal()
    known_actuator_dynamics = ActuatorParameters(
        thrust_time_constant=0.025,
        gimbal_time_constant=0.035,
        delay=0.0,
    )

    def controller():
        return GrapeController(
            controller_configuration,
            controller_parameters,
            geometry,
            articulated_model=GrapeArticulatedModel(),
        )

    nominal = simulate_closed_loop(
        times=times,
        references=references,
        initial_state=initial_state,
        initial_controller_state=controller_state,
        controller=controller(),
        plant=FullSixDofPlant(controller_parameters, geometry),
        actuator_parameters=known_actuator_dynamics,
    )
    truth = simulate_closed_loop(
        times=times,
        references=references,
        initial_state=initial_state,
        initial_controller_state=controller_state,
        controller=controller(),
        plant=FullSixDofPlant(truth_parameters, geometry),
        actuator_parameters=ActuatorParameters(
            thrust_time_constant=known_actuator_dynamics.thrust_time_constant,
            gimbal_time_constant=known_actuator_dynamics.gimbal_time_constant,
            delay=truth_delay,
        ),
        interval_residual_wrench=truth_residual_wrench,
    )
    observations = PoseObservations(
        times=times,
        position=truth.position,
        orientation_xyzw=truth.orientation_xyzw,
        translation_covariance=np.eye(3) * 0.004**2,
        rotation_covariance=np.eye(3) * np.deg2rad(0.25) ** 2,
    )
    strong = StrongConstraintProblem(
        references=references,
        observations=observations,
        nominal_trajectory=nominal,
        initial_state_anchor=initial_state,
        initial_controller_anchor=controller_state,
        controller_configuration=controller_configuration,
        controller_parameters=controller_parameters,
        geometry=geometry,
        actuator_parameters=known_actuator_dynamics,
        parameter_chart=VehicleParameterChart(controller_parameters),
    )
    process = GaussMarkovWrenchProcess(
        times=times[:-1],
        stationary_standard_deviation=np.asarray(
            (0.08, 0.08, 0.08, 0.006, 0.006, 0.006)
        ),
        correlation_time=0.12,
    )
    return JointBagProblem(
        bag_id, WeakConstraintProblem(strong, process), "same-hardware"
    ), truth


class JointAssimilationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chart = VehicleParameterChart(VehicleParameters.nominal())
        truth_coordinates = np.zeros(18)
        truth_coordinates[0] = 0.035
        truth_coordinates[1:4] = np.asarray((0.025, -0.015, 0.020))
        truth_coordinates[7:10] = np.asarray((0.004, -0.003, 0.002))
        truth_coordinates[10:14] = np.asarray((-0.03, 0.02, -0.01, 0.015))
        truth_coordinates[14:18] = np.asarray((0.02, -0.015, 0.01, -0.02))
        cls.shared_truth_coordinates = truth_coordinates.copy()
        cls.shared_truth_parameters = chart.decode(truth_coordinates)
        cls.shared_truth_delay = 0.017
        cls.first_truth_residual = np.asarray(
            (
                (0.030, -0.010, 0.020, 0.0020, 0.0000, -0.0010),
                (0.020, 0.015, 0.010, 0.0010, -0.0020, 0.0000),
                (-0.010, 0.020, 0.025, 0.0000, -0.0010, 0.0015),
                (-0.020, 0.005, 0.015, -0.0010, 0.0010, 0.0020),
            )
        )
        cls.second_truth_residual = np.asarray(
            (
                (-0.015, 0.025, -0.010, -0.0010, 0.0015, 0.0000),
                (0.010, 0.030, -0.020, 0.0000, 0.0020, -0.0015),
                (0.025, -0.015, -0.005, 0.0015, 0.0000, -0.0020),
                (0.015, -0.025, 0.010, 0.0020, -0.0010, -0.0010),
            )
        )
        cls.first, cls.first_truth = _synthetic_joint_bag(
            "bag_a",
            phase=0.0,
            position_bias=(0.0, 0.0, 0.0),
            initial_pose_delta=(
                0.010,
                -0.004,
                0.006,
                0.004,
                -0.003,
                0.002,
            ),
            initial_velocity_delta=(
                0.015,
                -0.010,
                0.005,
                0.003,
                0.000,
                -0.002,
            ),
            truth_parameters=cls.shared_truth_parameters,
            truth_delay=cls.shared_truth_delay,
            truth_residual_wrench=cls.first_truth_residual,
        )
        cls.second, cls.second_truth = _synthetic_joint_bag(
            "bag_b",
            phase=0.37,
            position_bias=(0.12, -0.08, 0.05),
            initial_pose_delta=(
                -0.012,
                0.008,
                -0.005,
                -0.003,
                0.005,
                -0.004,
            ),
            initial_velocity_delta=(
                -0.020,
                0.012,
                -0.004,
                -0.002,
                0.004,
                0.003,
            ),
            truth_parameters=cls.shared_truth_parameters,
            truth_delay=cls.shared_truth_delay,
            truth_residual_wrench=cls.second_truth_residual,
        )

    def test_synthetic_episodes_share_only_physical_truth_and_delay(self):
        problem = JointWeakConstraintProblem((self.second, self.first))
        truth_control = np.zeros(problem.control_dimension)
        truth_control[problem.layout.shared_parameter_slice] = (
            self.shared_truth_coordinates
        )
        truth_control[problem.layout.shared_delay_index] = (
            self.shared_truth_delay
        )
        decoded = problem.decode_control(truth_control)

        self.assertAlmostEqual(
            decoded.constant_delay, self.shared_truth_delay
        )
        self.assertAlmostEqual(
            decoded.parameters.mass, self.shared_truth_parameters.mass
        )
        np.testing.assert_allclose(
            decoded.parameters.inertia, self.shared_truth_parameters.inertia
        )
        first_strong = self.first.problem.strong_problem
        second_strong = self.second.problem.strong_problem
        self.assertFalse(
            np.allclose(
                first_strong.initial_state_anchor.as_vector(),
                second_strong.initial_state_anchor.as_vector(),
            )
        )
        self.assertFalse(
            np.allclose(
                np.asarray(
                    [value.position for value in first_strong.references]
                ),
                np.asarray(
                    [value.position for value in second_strong.references]
                ),
            )
        )
        self.assertFalse(
            np.allclose(
                self.first_truth_residual, self.second_truth_residual
            )
        )
        self.assertFalse(
            np.allclose(self.first_truth.position, self.second_truth.position)
        )

    def test_layout_shares_only_physical_parameters_and_delay(self):
        problem = JointWeakConstraintProblem((self.second, self.first))
        local_dimension = (
            PARAMETER_OFFSET
            + ACTUATOR_STATE_DIMENSION
            + self.first.problem.wrench_process.innovation_dimension
        )
        self.assertEqual(
            problem.control_dimension,
            SHARED_STATIC_DIMENSION + 2 * local_dimension,
        )
        self.assertEqual(
            tuple(value.bag_id for value in problem.bags),
            ("bag_a", "bag_b"),
        )

    def test_tau_is_shared_but_local_innovation_forecasts_are_independent(self):
        problem = JointWeakConstraintProblem((self.first, self.second))
        center = np.zeros(problem.control_dimension)
        center[problem.layout.shared_delay_index] = -0.017
        first_layout = problem.layout.for_bag("bag_a")
        second_layout = problem.layout.for_bag("bag_b")
        first_changed = center.copy()
        first_changed[first_layout.innovation_slice.start] = 0.75
        second_changed = center.copy()
        second_changed[second_layout.innovation_slice.start + 1] = -0.75

        decoded = problem.decode_control(first_changed)
        self.assertEqual(decoded.constant_delay, 0.017)
        baseline_paths = problem.forecast(center)
        first_paths = problem.forecast(first_changed)
        second_paths = problem.forecast(second_changed)
        self.assertGreater(
            np.linalg.norm(first_paths[0].position - baseline_paths[0].position),
            1.0e-10,
        )
        np.testing.assert_allclose(
            first_paths[1].position,
            baseline_paths[1].position,
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            np.linalg.norm(
                second_paths[1].position - baseline_paths[1].position
            ),
            1.0e-10,
        )
        np.testing.assert_allclose(
            second_paths[0].position,
            baseline_paths[0].position,
            rtol=0.0,
            atol=0.0,
        )

    def test_joint_residual_concatenates_pose_only_windows(self):
        problem = JointWeakConstraintProblem((self.first, self.second))
        control = np.zeros(problem.control_dimension)
        trajectories = problem.forecast(control)
        residual = problem.residual(trajectories)
        samples = self.first.problem.strong_problem.observations.times.size
        self.assertEqual(residual.shape, (2 * samples * 6,))

    def test_bag_order_is_canonical_and_second_bag_adds_static_information(self):
        forward = JointWeakConstraintProblem((self.first, self.second))
        reverse = JointWeakConstraintProblem((self.second, self.first))
        forward_prior = JointWeakConstraintPrior.grape(forward).ensemble(5, 31)
        reverse_prior = JointWeakConstraintPrior.grape(reverse).ensemble(5, 31)
        np.testing.assert_array_equal(forward_prior, reverse_prior)
        np.testing.assert_allclose(
            forward.forecast_residual_batch(forward_prior),
            reverse.forecast_residual_batch(reverse_prior),
        )

        first_only = JointWeakConstraintProblem((self.first,))
        second_only = JointWeakConstraintProblem((self.second,))

        def static_response(selected):
            center = np.zeros(selected.control_dimension)
            shifted = center.copy()
            shifted[selected.layout.shared_parameter_slice.start] = 0.04
            return selected.residual(
                selected.forecast(shifted)
            ) - selected.residual(selected.forecast(center))

        first_delta = static_response(first_only)
        second_delta = static_response(second_only)
        two_center = np.zeros(forward.control_dimension)
        two_shifted = two_center.copy()
        two_shifted[forward.layout.shared_parameter_slice.start] = 0.04
        two_delta = forward.residual(
            forward.forecast(two_shifted)
        ) - forward.residual(forward.forecast(two_center))

        first_size = first_delta.size
        np.testing.assert_allclose(two_delta[:first_size], first_delta)
        np.testing.assert_allclose(two_delta[first_size:], second_delta)
        first_information = float(np.dot(first_delta, first_delta))
        second_information = float(np.dot(second_delta, second_delta))
        self.assertGreater(first_information, 0.0)
        self.assertGreater(second_information, 0.0)
        self.assertAlmostEqual(
            float(np.dot(two_delta, two_delta)),
            first_information + second_information,
            places=8,
        )

    def test_fit_is_order_invariant_and_preserves_raw_member_alignment(self):
        forward = JointWeakConstraintProblem((self.first, self.second))
        reverse = JointWeakConstraintProblem((self.second, self.first))
        configuration = JointIEnKSConfig(
            ensemble_size=6, maximum_iterations=1, seed=5
        )
        solver = JointWeakConstraintIEnKSQ(configuration)
        forward_posterior = solver.fit(
            forward, JointWeakConstraintPrior.grape(forward)
        )
        reverse_posterior = solver.fit(
            reverse, JointWeakConstraintPrior.grape(reverse)
        )
        np.testing.assert_allclose(
            forward_posterior.control_ensemble,
            reverse_posterior.control_ensemble,
            rtol=0.0,
            atol=1.0e-12,
        )
        posterior = forward_posterior
        np.testing.assert_array_equal(
            posterior.member_id,
            posterior.shared_parameter_ensemble.member_id,
        )
        self.assertEqual(
            posterior.shared_parameter_ensemble.constant_delay.shape, (6,)
        )
        self.assertTrue(
            np.all(posterior.shared_parameter_ensemble.constant_delay >= 0.0)
        )
        expected_ridge = np.concatenate(
            (
                self.first.problem.parameter_chart.ridge_direction(),
                np.asarray((0.0,)),
            )
        )
        expected_ridge /= np.linalg.norm(expected_ridge)
        np.testing.assert_allclose(
            posterior.ridge.expected_direction, expected_ridge
        )
        self.assertAlmostEqual(
            posterior.ridge.expected_variance,
            float(
                expected_ridge
                @ posterior.ridge.covariance
                @ expected_ridge
            ),
        )
        for bag_index, bag in enumerate(posterior.bags):
            np.testing.assert_array_equal(bag.member_id, posterior.member_id)
            self.assertEqual(len(bag.trajectory_ensemble), 6)
            layout = forward.layout.for_bag(bag.bag_id)
            np.testing.assert_array_equal(
                bag.innovation_ensemble,
                posterior.control_ensemble[:, layout.innovation_slice],
            )
            bag_problem = forward.bags[bag_index].problem
            self.assertEqual(forward.bags[bag_index].bag_id, bag.bag_id)
            decoded_residual = np.asarray(
                [
                    bag_problem.wrench_process.decode(value)
                    for value in bag.innovation_ensemble
                ]
            )
            np.testing.assert_allclose(
                bag.residual_wrench_ensemble, decoded_residual
            )

        replay = forward.forecast(posterior.control_ensemble[0])
        for bag_index, bag in enumerate(posterior.bags):
            np.testing.assert_allclose(
                bag.trajectory_ensemble[0].position,
                replay[bag_index].position,
            )

    def test_different_configuration_fingerprints_are_blocked(self):
        incompatible = JointBagProblem(
            "bag_c", self.first.problem, "different-hardware"
        )
        with self.assertRaises(ValueError):
            JointWeakConstraintProblem((self.first, incompatible))

    def test_real_episode_run_bundle_round_trips_all_raw_member_paths(self):
        episode = build_real_flight_episode(
            _fake_arrays(),
            sample_period=0.2,
            start_local=3.0,
            end_local=4.4,
            window_state=None,
        )
        prepared = prepare_joint_flight(
            "bag_a",
            episode,
            "same-hardware",
            maximum_knots=2,
            initial_delay=0.01,
        )
        result = assimilate_joint_flights(
            (prepared,),
            configuration=JointIEnKSConfig(
                ensemble_size=5, maximum_iterations=1, seed=19
            ),
            delay_mean=0.01,
            delay_standard_deviation=0.004,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            begin_bundle(
                root,
                assimilation_run_manifest(
                    run_id="joint-round-trip",
                    request_path="requests/joint-round-trip.json",
                    request_fingerprint="sha256:test",
                    project_request_fingerprint="sha256:" + "a" * 64,
                    selected_intervals={
                        "bag_a": (
                            episode.window_start_local_time,
                            episode.window_end_local_time,
                        )
                    },
                    configuration_fingerprint="same-hardware",
                    member_count=5,
                    estimator_revision="test",
                ),
            )
            write_joint_assimilation_payloads(str(root), result)
            mark_bundle_complete(
                root,
                {
                    "termination_reason": result.posterior.termination_reason,
                    "converged": result.posterior.converged,
                    "initial_prior_forecast": initial_prior_forecast_manifest(
                        result.posterior
                    ),
                },
            )
            loaded = load_assimilation_run(root)
            np.testing.assert_array_equal(
                loaded.shared_posterior["member_id"], np.arange(5)
            )
            self.assertEqual(
                loaded.bags["bag_a"]["posterior_position"].shape[0], 5
            )
            self.assertEqual(
                loaded.bags["bag_a"]["residual_wrench_knot"].shape,
                (5, 2, 6),
            )
            np.testing.assert_array_equal(
                loaded.diagnostics["initial_prior_member_id"], np.arange(5)
            )
            self.assertEqual(
                loaded.diagnostics[
                    "effective_prior_control_ensemble"
                ].shape,
                loaded.diagnostics[
                    "requested_prior_control_ensemble"
                ].shape,
            )


if __name__ == "__main__":
    unittest.main()
