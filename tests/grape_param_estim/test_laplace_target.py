import unittest

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.batch.problem import BatchProblem
from grape_param_estim.batch.state import BatchState, StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_log
from grape_param_estim.posterior.delayed_acceptance import PosteriorPoint
from grape_param_estim.posterior.laplace_target import (
    ConditionalTrajectoryWarmStart,
    FixedDelayLaplaceProblem,
    LaplaceMarginalTarget,
    LaplaceTargetModelFailure,
    factorize_bag_local_hessian,
)


KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _factor(residual, blocks):
    residual = np.asarray(residual, dtype=float)
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=tuple(blocks),
        squared_error=float(residual @ residual),
        active_set={},
    )


class LinearGaussianFactory:
    def __init__(
        self,
        *,
        initial_position=-3.0,
        include_static_prior=True,
        static_offset=0.0,
        delay_offset=0.0,
        singular=False,
    ):
        self.initial_position = float(initial_position)
        self.include_static_prior = include_static_prior
        self.static_offset = float(static_offset)
        self.delay_offset = float(delay_offset)
        self.singular = singular
        keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
        for bag_id in ("bag-a", "bag-b"):
            keys.extend(
                VariableKey(kind, bag_id=bag_id, knot_index=0)
                for kind in KNOT_KINDS
            )
        self.layout = VariableLayout(tuple(keys))

    @staticmethod
    def _local_vector(state, bag_id):
        parts = []
        for kind in KNOT_KINDS:
            value = state.knot_value(bag_id, 0, kind)
            parts.append(
                so3_log(value)
                if kind is VariableKind.ORIENTATION_TANGENT
                else value
            )
        return np.concatenate(tuple(parts))

    def __call__(self, point):
        coordinate = point.static_coordinate.copy()
        coordinate[0] += self.static_offset
        values = {
            VariableKey(VariableKind.STATIC_PARAMETERS): coordinate,
        }
        for bag_id in self.layout.bag_ids:
            for kind in KNOT_KINDS:
                key = VariableKey(kind, bag_id=bag_id, knot_index=0)
                if kind is VariableKind.ORIENTATION_TANGENT:
                    value = np.eye(3)
                else:
                    value = np.zeros(key.dimension)
                    if kind is VariableKind.POSITION:
                        value[0] = self.initial_position
                values[key] = value
        initial = BatchState(self.layout, values)

        def evaluator(state):
            static = state.value(self.layout.variable_keys[0])
            prior_residual = static / 2.0
            prior_jacobian = np.eye(18) / 2.0
            factors = [
                _factor(
                    prior_residual,
                    (
                        JacobianBlock(
                            self.layout.variable_keys[0], prior_jacobian
                        ),
                    ),
                )
            ]
            for bag_index, bag_id in enumerate(self.layout.bag_ids):
                local = self._local_vector(state, bag_id)
                coefficient = 0.1 * (bag_index + 1)
                scale = np.ones(local.size)
                scale[0] = np.exp(
                    coefficient * static[1]
                    + (bag_index + 1) * point.delay
                )
                if self.singular and bag_id == "bag-b":
                    scale[-1] = 0.0
                target = np.zeros(local.size)
                target[0] = static[0] + (bag_index + 1) * point.delay
                difference = local - target
                residual = scale * difference
                static_jacobian = np.zeros((local.size, 18))
                static_jacobian[0, 0] = -scale[0]
                static_jacobian[0, 1] = (
                    coefficient * scale[0] * difference[0]
                )
                blocks = [
                    JacobianBlock(
                        self.layout.variable_keys[0], static_jacobian
                    )
                ]
                offset = 0
                for kind in KNOT_KINDS:
                    key = VariableKey(kind, bag_id=bag_id, knot_index=0)
                    block = np.zeros((local.size, key.dimension))
                    local_scale = scale[offset : offset + key.dimension]
                    block[
                        np.arange(offset, offset + key.dimension),
                        np.arange(key.dimension),
                    ] = local_scale
                    blocks.append(JacobianBlock(key, block))
                    offset += key.dimension
                factors.append(_factor(residual, blocks))
            return tuple(factors)

        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            evaluator,
        )
        return FixedDelayLaplaceProblem(
            fixed_delay=point.delay + self.delay_offset,
            problem=problem,
            initial_state=initial,
            graph_objective_includes_static_prior=self.include_static_prior,
        )


def _point(c0=0.4, c1=-0.2, delay=0.03):
    coordinate = np.zeros(18)
    coordinate[0] = c0
    coordinate[1] = c1
    return PosteriorPoint(coordinate, delay)


def _settings(**changes):
    values = dict(
        maximum_iterations=8,
        initial_damping=0.0,
        minimum_damping=0.0,
        maximum_damping=1.0e8,
        gradient_tolerance=1.0e-10,
        scaled_step_tolerance=1.0e-12,
        relative_objective_tolerance=1.0e-12,
    )
    values.update(changes)
    return LMSettings(**values)


class LaplaceMarginalTargetTests(unittest.TestCase):
    def test_matches_linear_gaussian_profile_and_dense_logdet_oracle(self):
        factory = LinearGaussianFactory()
        target = LaplaceMarginalTarget(
            factory,
            lambda delay: -0.5 * (delay / 0.05) ** 2,
            _settings(),
        )
        point = _point()
        result = target(point)
        self.assertTrue(result.successful, result.failure_reason)
        expected_objective = 0.5 * np.sum((point.static_coordinate / 2.0) ** 2)
        expected_logdet = 0.0
        for bag_index in range(2):
            expected_logdet += 2.0 * (
                0.1 * (bag_index + 1) * point.static_coordinate[1]
                + (bag_index + 1) * point.delay
            )
        expected_prior = -0.5 * (point.delay / 0.05) ** 2
        self.assertAlmostEqual(result.graph_objective, expected_objective, 12)
        self.assertAlmostEqual(result.local_log_determinant, expected_logdet, 11)
        self.assertAlmostEqual(
            result.log_density,
            expected_prior - expected_objective - 0.5 * expected_logdet,
            11,
        )

        product = factory(point)
        final = product.problem.linearize(result.warm_start.state).sparse
        sparse_logdet = factorize_bag_local_hessian(final)
        dense_logdet = 0.0
        for bag_id in final.layout.bag_ids:
            local_slice = final.layout.bag_slice(bag_id)
            sign, value = np.linalg.slogdet(
                final.hessian[local_slice, local_slice].toarray()
            )
            self.assertEqual(sign, 1.0)
            dense_logdet += value
        self.assertAlmostEqual(sparse_logdet.value, dense_logdet, 11)
        self.assertEqual(
            tuple(item.bag_id for item in sparse_logdet.bags),
            ("bag-a", "bag-b"),
        )

    def test_warm_start_rebases_only_local_state_across_delay_graphs(self):
        factory = LinearGaussianFactory(initial_position=100.0)
        target = LaplaceMarginalTarget(factory, lambda delay: 0.0, _settings())
        first = target(_point(delay=0.01))
        self.assertTrue(first.successful, first.failure_reason)
        self.assertIsInstance(first.warm_start, ConditionalTrajectoryWarmStart)
        second_coordinate = _point(c0=0.7, c1=0.3).static_coordinate.copy()
        second_coordinate[2] = -0.0
        second_point = PosteriorPoint(second_coordinate, 0.04)
        second = target(second_point, first.warm_start)
        self.assertTrue(second.successful, second.failure_reason)
        np.testing.assert_array_equal(
            second.warm_start.state.value(
                second.warm_start.state.layout.variable_keys[0]
            ).view(np.uint64),
            second_point.static_coordinate.view(np.uint64),
        )
        for bag_index, bag_id in enumerate(("bag-a", "bag-b")):
            expected = (
                second_point.static_coordinate[0]
                + (bag_index + 1) * second_point.delay
            )
            actual = second.warm_start.state.knot_value(
                bag_id, 0, VariableKind.POSITION
            )[0]
            self.assertAlmostEqual(actual, expected, 12)

    def test_rejects_factory_static_delay_and_prior_contract_mismatches(self):
        cases = (
            (
                LinearGaussianFactory(static_offset=np.spacing(0.4)),
                "factory_static_coordinate_mismatch",
            ),
            (
                LinearGaussianFactory(delay_offset=np.spacing(0.03)),
                "factory_delay_mismatch",
            ),
            (
                LinearGaussianFactory(include_static_prior=False),
                "graph_objective_missing_static_prior",
            ),
        )
        for factory, reason in cases:
            result = LaplaceMarginalTarget(
                factory, lambda delay: 0.0, _settings()
            )(_point())
            self.assertFalse(result.successful)
            self.assertEqual(result.failure_reason, reason)
            self.assertEqual(result.log_density, float("-inf"))

    def test_nonconvergence_singular_nonfinite_and_model_failures_are_explicit(self):
        nonconverged = LaplaceMarginalTarget(
            LinearGaussianFactory(),
            lambda delay: 0.0,
            _settings(maximum_iterations=1, initial_damping=100.0),
        )(_point())
        self.assertFalse(nonconverged.successful)
        self.assertEqual(
            nonconverged.failure_reason,
            "lm_nonconverged:maximum_iterations",
        )

        singular = LaplaceMarginalTarget(
            LinearGaussianFactory(singular=True),
            lambda delay: 0.0,
            _settings(),
        )(_point())
        self.assertFalse(singular.successful)
        self.assertEqual(
            singular.failure_reason,
            "local_hessian_factorization_failure",
        )

        nonfinite = LaplaceMarginalTarget(
            LinearGaussianFactory(), lambda delay: np.nan, _settings()
        )(_point())
        self.assertFalse(nonfinite.successful)
        self.assertEqual(nonfinite.failure_reason, "delay_prior_nonfinite")

        def failing_factory(point):
            raise LaplaceTargetModelFailure("invalid_chart")

        model_failure = LaplaceMarginalTarget(
            failing_factory, lambda delay: 0.0, _settings()
        )(_point())
        self.assertFalse(model_failure.successful)
        self.assertEqual(
            model_failure.failure_reason,
            "problem_factory_model_failure:invalid_chart",
        )


if __name__ == "__main__":
    unittest.main()
