import unittest

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.factors.dynamics import (
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    body_wrench_statistical_residual,
)
from grape_param_estim.batch.lag_profile import (
    LagObjectiveResult,
    LagProfileSettings,
    optimize_lag_profile,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.lm import LMSettings, solve_batch_map
from grape_param_estim.batch.problem import BatchProblem
from grape_param_estim.batch.ridge import analyze_reduced_hessian
from grape_param_estim.batch.sparse_solver import solve_scaled_lm_step
from grape_param_estim.batch.state import BatchState, StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.synthetic_batch import (
    generate_perfect_model_batch_trajectory,
    simulate_delayed_zoh_first_order,
)


_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)
_GRAVITY_WORLD = np.asarray((0.0, 0.0, -9.80665))
_DYNAMICS_Q_DEFINITION = DiagonalQDefinition(
    residual_quantity=BODY_WRENCH_QUANTITY,
    component_names=("x", "y", "z", "roll", "pitch", "yaw"),
    component_units=("N",) * 3 + ("N*m",) * 3,
    interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
)


def _single_knot_layout(bag_ids):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag_id in bag_ids:
        keys.extend(
            VariableKey(kind, bag_id=bag_id, knot_index=0)
            for kind in _KNOT_KINDS
        )
    return VariableLayout(tuple(keys))


def _static_marker_state(coordinates):
    layout = _single_knot_layout(("synthetic-delay",))
    values = {}
    for key in layout.variable_keys:
        if key.kind is VariableKind.STATIC_PARAMETERS:
            values[key] = np.asarray(coordinates, dtype=float)
        elif key.kind is VariableKind.ORIENTATION_TANGENT:
            values[key] = np.eye(3)
        else:
            values[key] = np.zeros(key.dimension)
    return BatchState(layout, values)


def _commands_reproducing_profile(times, target, delay, time_constant):
    """Invert exact first-order segments so the delayed response hits target."""

    commands = np.empty_like(target)
    commands[0] = target[0]
    for index, time_step in enumerate(np.diff(times)):
        before_switch = commands[index] + np.exp(
            -delay / time_constant
        ) * (target[index] - commands[index])
        decay = np.exp(-(time_step - delay) / time_constant)
        commands[index + 1] = (
            target[index + 1] - decay * before_switch
        ) / (1.0 - decay)
    event_times = np.concatenate(
        (
            np.asarray((times[0] - 1.0,), dtype=float),
            np.asarray(times[:-1], dtype=float),
        )
    )
    return event_times, commands


def _dynamics_system(trajectory, actuator_thrust, coordinates):
    residuals = []
    jacobians = []
    for index, time_step in enumerate(trajectory.time_step):
        evaluation = evaluate_raw_dynamics_residual(
            rotation_left=trajectory.rotation[index],
            rotation_right=trajectory.rotation[index + 1],
            linear_velocity_left=trajectory.linear_velocity[index],
            linear_velocity_right=trajectory.linear_velocity[index + 1],
            angular_velocity_left=trajectory.angular_velocity[index],
            angular_velocity_right=trajectory.angular_velocity[index + 1],
            actuator_thrust_left=actuator_thrust[index],
            actuator_thrust_right=actuator_thrust[index + 1],
            gimbal_angle_left=trajectory.gimbal_angle[index],
            gimbal_angle_right=trajectory.gimbal_angle[index + 1],
            time_step=time_step,
            parameter_chart=trajectory.parameter_chart,
            parameter_coordinates=coordinates,
            geometry=trajectory.geometry,
            gravity_world=_GRAVITY_WORLD,
        )
        statistical = body_wrench_statistical_residual(
            "synthetic-ridge-delay",
            index,
            evaluation,
            _DYNAMICS_Q_DEFINITION,
        )
        residuals.append(statistical.residual)
        jacobians.append(statistical.jacobian.static_parameters)
    return np.concatenate(residuals), np.vstack(jacobians)


class JointRidgeAndContinuousDelaySyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = generate_perfect_model_batch_trajectory(
            interval_count=24,
            seed=630,
        )
        cls.truth_delay = 0.0073
        cls.time_constant = 0.0105
        cls.event_times, cls.command_values = _commands_reproducing_profile(
            cls.trajectory.times,
            cls.trajectory.actuator_thrust,
            cls.truth_delay,
            cls.time_constant,
        )
        reproduced = simulate_delayed_zoh_first_order(
            cls.trajectory.times,
            cls.event_times,
            cls.command_values,
            delay=cls.truth_delay,
            time_constant=cls.time_constant,
            initial_state=cls.trajectory.actuator_thrust[0],
        )
        np.testing.assert_allclose(
            reproduced,
            cls.trajectory.actuator_thrust,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    @classmethod
    def _system_at_delay(cls, delay, coordinates=None):
        thrust = simulate_delayed_zoh_first_order(
            cls.trajectory.times,
            cls.event_times,
            cls.command_values,
            delay=delay,
            time_constant=cls.time_constant,
            initial_state=cls.trajectory.actuator_thrust[0],
        )
        selected = (
            cls.trajectory.truth_parameter_coordinates
            if coordinates is None
            else coordinates
        )
        return _dynamics_system(cls.trajectory, thrust, selected)

    def test_subsample_delay_profile_and_exact_scale_ridge_coexist(self):
        truth_coordinates = self.trajectory.truth_parameter_coordinates

        def evaluator(lag, _warm_start):
            residual, _ = self._system_at_delay(lag)
            return LagObjectiveResult(
                objective=0.5 * float(residual @ residual),
                converged=True,
                state=_static_marker_state(truth_coordinates),
                inner_iterations=1,
                termination_reason="synthetic_production_dynamics_factor",
            )

        profile = optimize_lag_profile(
            evaluator,
            LagProfileSettings(
                minimum_lag=0.001,
                maximum_lag=0.014,
                coarse_grid_points=9,
                refinement_tolerance=5.0e-7,
                maximum_refinement_evaluations=38,
            ),
            initial_warm_start=_static_marker_state(truth_coordinates),
        )
        self.assertLess(
            self.truth_delay,
            float(np.min(self.trajectory.time_step)),
        )
        self.assertAlmostEqual(profile.best_lag, self.truth_delay, delta=1.2e-6)
        self.assertLess(profile.best_objective, 2.0e-8)

        ridge = self.trajectory.parameter_chart.ridge_direction()
        ridge /= np.linalg.norm(ridge)
        truth_ridge_residual_norms = []
        for displacement in (-0.55, -0.2, 0.0, 0.35, 0.7):
            residual, _ = self._system_at_delay(
                self.truth_delay,
                truth_coordinates + displacement * ridge,
            )
            truth_ridge_residual_norms.append(
                np.linalg.norm(residual, ord=np.inf)
            )
        self.assertLess(
            max(truth_ridge_residual_norms),
            5.0e-10,
        )
        self.assertGreater(np.ptp(self.trajectory.actuator_thrust), 2.0)
        self.assertGreater(np.ptp(self.trajectory.gimbal_angle), 0.1)

        truth_residual, static_jacobian = self._system_at_delay(
            self.truth_delay
        )
        epsilon = 2.0e-6
        plus_residual, _ = self._system_at_delay(self.truth_delay + epsilon)
        minus_residual, _ = self._system_at_delay(self.truth_delay - epsilon)
        delay_column = (plus_residual - minus_residual) / (2.0 * epsilon)
        joint_jacobian = np.column_stack((static_jacobian, delay_column))
        static_geometry = analyze_reduced_hessian(
            static_jacobian.T @ static_jacobian,
            relative_rank_tolerance=2.0e-9,
        )
        joint_hessian = joint_jacobian.T @ joint_jacobian
        joint_eigenvalues, joint_eigenvectors = np.linalg.eigh(
            0.5 * (joint_hessian + joint_hessian.T)
        )
        joint_rank = int(
            np.count_nonzero(
                joint_eigenvalues
                > 2.0e-9 * float(np.max(joint_eigenvalues))
            )
        )
        expected_null = np.concatenate((ridge, np.zeros(1)))
        numerical_null = joint_eigenvectors[:, 0]
        self.assertEqual(static_geometry.effective_rank, 17)
        self.assertGreater(
            abs(float(static_geometry.ridge_directions[0].vector @ ridge)),
            1.0 - 2.0e-9,
        )
        self.assertEqual(joint_rank, 18)
        self.assertGreater(
            abs(float(numerical_null @ expected_null)),
            1.0 - 2.0e-9,
        )
        self.assertLess(
            np.linalg.norm(static_jacobian @ ridge, ord=np.inf),
            2.0e-12,
        )
        self.assertLess(np.linalg.norm(truth_residual, ord=np.inf), 3.0e-10)


def _factor(residual, blocks):
    value = np.asarray(residual, dtype=float)
    return FactorEvaluation(
        residual=value,
        jacobian_blocks=tuple(
            JacobianBlock(key, jacobian) for key, jacobian in blocks
        ),
        squared_error=float(value @ value),
        active_set={},
    )


def _permutation_problem(input_bag_ids):
    layout = _single_knot_layout(input_bag_ids)
    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    static_truth = np.linspace(-0.08, 0.09, 18)
    bag_seeds = {"bag-alpha": 91, "bag-mu": 207, "bag-zeta": 518}
    bag_truth = {
        bag_id: np.random.RandomState(seed).normal(scale=0.12, size=3)
        for bag_id, seed in bag_seeds.items()
    }

    def evaluate(state):
        static = state.value(static_key)
        factors = [
            _factor(
                0.23 * (static - static_truth),
                ((static_key, 0.23 * np.eye(18)),),
            )
        ]
        for bag_id in input_bag_ids:
            generator = np.random.RandomState(bag_seeds[bag_id])
            position_key = VariableKey(
                VariableKind.POSITION,
                bag_id=bag_id,
                knot_index=0,
            )
            position = state.value(position_key)
            shared_matrix = generator.normal(scale=0.18, size=(7, 18))
            local_matrix = generator.normal(scale=0.35, size=(7, 3))
            target = (
                shared_matrix @ static_truth
                + local_matrix @ bag_truth[bag_id]
            )
            factors.append(
                _factor(
                    shared_matrix @ static + local_matrix @ position - target,
                    (
                        (static_key, shared_matrix),
                        (position_key, local_matrix),
                    ),
                )
            )
            factors.append(
                _factor(
                    0.31 * (position - bag_truth[bag_id]),
                    ((position_key, 0.31 * np.eye(3)),),
                )
            )
            for kind in _KNOT_KINDS:
                if kind in (
                    VariableKind.POSITION,
                    VariableKind.ORIENTATION_TANGENT,
                ):
                    continue
                key = VariableKey(kind, bag_id=bag_id, knot_index=0)
                value = state.value(key)
                factors.append(
                    _factor(
                        0.27 * value,
                        ((key, 0.27 * np.eye(key.dimension)),),
                    )
                )
            orientation_key = VariableKey(
                VariableKind.ORIENTATION_TANGENT,
                bag_id=bag_id,
                knot_index=0,
            )
            factors.append(
                _factor(
                    np.zeros(3),
                    ((orientation_key, np.eye(3)),),
                )
            )
        return tuple(factors)

    problem = BatchProblem(layout, StateScaling.unit(), evaluate)
    initial_values = {}
    for index, key in enumerate(layout.variable_keys):
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            initial_values[key] = np.eye(3)
        else:
            initial_values[key] = np.full(
                key.dimension,
                0.17 - 0.004 * index,
            )
    return problem, BatchState(layout, initial_values)


class MultiBagInputPermutationIntegrationTests(unittest.TestCase):
    def test_iterated_smoother_matches_sparse_lm_nonlinear_solution(self):
        problem, initial = _permutation_problem(
            ("bag-alpha", "bag-mu", "bag-zeta")
        )
        common = dict(
            maximum_iterations=32,
            initial_damping=0.04,
            gradient_tolerance=2.0e-10,
            scaled_step_tolerance=2.0e-10,
            relative_objective_tolerance=1.0e-12,
        )
        sparse = solve_batch_map(
            problem,
            initial,
            LMSettings(method="sparse_lm", **common),
        )
        ieks = solve_batch_map(
            problem,
            initial,
            LMSettings(method="ieks", **common),
        )

        self.assertTrue(sparse.converged)
        self.assertTrue(ieks.converged)
        self.assertAlmostEqual(ieks.objective, sparse.objective, places=18)
        self.assertGreater(len(ieks.iterations), 1)
        for key in problem.layout.variable_keys:
            np.testing.assert_allclose(
                ieks.state.value(key),
                sparse.state.value(key),
                rtol=3.0e-10,
                atol=3.0e-11,
            )

    def test_actual_assembly_schur_and_lm_are_input_order_invariant(self):
        permutations = (
            ("bag-alpha", "bag-mu", "bag-zeta"),
            ("bag-zeta", "bag-alpha", "bag-mu"),
            ("bag-mu", "bag-zeta", "bag-alpha"),
        )
        runs = []
        settings = LMSettings(
            maximum_iterations=32,
            initial_damping=0.04,
            gradient_tolerance=2.0e-10,
            scaled_step_tolerance=2.0e-10,
            relative_objective_tolerance=1.0e-12,
        )
        for permutation in permutations:
            problem, initial = _permutation_problem(permutation)
            linearization = problem.linearize(initial).sparse
            schur = solve_scaled_lm_step(
                linearization,
                problem.coordinate_scale,
                damping=0.13,
            )
            solved = solve_batch_map(problem, initial, settings)
            runs.append((problem, linearization, schur, solved))

        reference_problem, reference_linear, reference_schur, reference = runs[0]
        self.assertEqual(
            reference_problem.layout.bag_ids,
            tuple(sorted(permutations[0])),
        )
        self.assertTrue(reference.converged)
        self.assertGreater(len(reference.iterations), 0)
        for problem, linearization, schur, solved in runs[1:]:
            self.assertEqual(problem.layout, reference_problem.layout)
            self.assertAlmostEqual(
                linearization.objective,
                reference_linear.objective,
                places=14,
            )
            np.testing.assert_allclose(
                schur.reduced_hessian,
                reference_schur.reduced_hessian,
                rtol=3.0e-14,
                atol=3.0e-15,
            )
            np.testing.assert_allclose(
                schur.reduced_rhs,
                reference_schur.reduced_rhs,
                rtol=3.0e-14,
                atol=3.0e-15,
            )
            self.assertTrue(solved.converged)
            self.assertAlmostEqual(solved.objective, reference.objective, places=19)
            for key in reference_problem.layout.variable_keys:
                np.testing.assert_allclose(
                    solved.state.value(key),
                    reference.state.value(key),
                    rtol=2.0e-11,
                    atol=2.0e-12,
                )
            self.assertEqual(
                tuple(item.local_dimension for item in schur.bag_diagnostics),
                tuple(
                    item.local_dimension
                    for item in reference_schur.bag_diagnostics
                ),
            )


if __name__ == "__main__":
    unittest.main()
