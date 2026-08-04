import unittest

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.ieks import (
    solve_scaled_conditional_ieks_step,
    solve_scaled_ieks_step,
    validate_ieks_topology,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.linearize import assemble_sparse_linearization
from grape_param_estim.batch.lm import LMSettings, NonlinearSolverMethod
from grape_param_estim.batch.variables import VariableKey, VariableKind


_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _layout(bag_ids=("bag-a", "bag-b"), knot_count=3, biases=True):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag_id in bag_ids:
        if biases:
            keys.extend(
                (
                    VariableKey(VariableKind.GYRO_BIAS, bag_id=bag_id),
                    VariableKey(
                        VariableKind.ACCELEROMETER_BIAS,
                        bag_id=bag_id,
                    ),
                )
            )
        for knot_index in range(knot_count):
            keys.extend(
                VariableKey(
                    kind,
                    bag_id=bag_id,
                    knot_index=knot_index,
                )
                for kind in _KNOT_KINDS
            )
    return VariableLayout(tuple(keys))


def _factor(residual, blocks):
    selected = np.asarray(residual, dtype=float)
    return FactorEvaluation(
        residual=selected,
        jacobian_blocks=tuple(
            JacobianBlock(key, value) for key, value in blocks
        ),
        squared_error=float(selected @ selected),
        active_set={},
    )


def _markov_linearization(layout, seed=709):
    generator = np.random.RandomState(seed)
    factors = []
    for key in layout.variable_keys:
        residual = generator.normal(scale=0.2, size=key.dimension)
        factors.append(
            _factor(
                residual,
                (
                    (
                        key,
                        (0.7 + generator.uniform())
                        * np.eye(key.dimension),
                    ),
                ),
            )
        )

    static = VariableKey(VariableKind.STATIC_PARAMETERS)
    for bag_id in layout.bag_ids:
        knot_count = 1 + max(
            key.knot_index
            for key in layout.variable_keys
            if key.bag_id == bag_id and key.knot_index is not None
        )
        bias_keys = tuple(
            key
            for key in layout.variable_keys
            if key.bag_id == bag_id and key.knot_index is None
        )
        for knot_index in range(knot_count - 1):
            rows = 13
            keys = (static,) + bias_keys
            keys += tuple(
                VariableKey(
                    kind,
                    bag_id=bag_id,
                    knot_index=selected_knot,
                )
                for selected_knot in (knot_index, knot_index + 1)
                for kind in _KNOT_KINDS
            )
            factors.append(
                _factor(
                    generator.normal(scale=0.08, size=rows),
                    tuple(
                        (
                            key,
                            generator.normal(
                                scale=0.06,
                                size=(rows, key.dimension),
                            ),
                        )
                        for key in keys
                    ),
                )
            )
    return assemble_sparse_linearization(layout, tuple(factors))


def _dense_step(linearization, scale, damping, optimize_shared):
    hessian = linearization.hessian.toarray()
    scaled_hessian = scale[:, None] * hessian * scale[None, :]
    scaled_gradient = scale * linearization.gradient
    expected = np.zeros(scale.size)
    first = 0 if optimize_shared else linearization.layout.shared_slice.stop
    selected = np.arange(first, scale.size)
    expected[selected] = np.linalg.solve(
        scaled_hessian[np.ix_(selected, selected)]
        + damping * np.eye(selected.size),
        -scaled_gradient[selected],
    )
    return expected


class IeksInformationSmootherTests(unittest.TestCase):
    def test_joint_forward_backward_pass_matches_dense_system(self):
        layout = _layout()
        linearization = _markov_linearization(layout)
        generator = np.random.RandomState(812)
        scale = np.exp(
            generator.normal(scale=0.25, size=layout.total_dimension)
        )
        damping = 0.19

        result = solve_scaled_ieks_step(
            linearization, scale, damping
        )
        expected = _dense_step(
            linearization, scale, damping, optimize_shared=True
        )

        np.testing.assert_allclose(
            result.scaled_delta, expected, rtol=3.0e-10, atol=3.0e-11
        )
        np.testing.assert_allclose(result.delta, scale * expected)
        self.assertEqual(len(result.bag_diagnostics), 2)
        self.assertTrue(
            all(value.knot_count == 3 for value in result.bag_diagnostics)
        )
        self.assertTrue(
            all(
                value.forward_factorizations == value.backward_steps == 3
                for value in result.bag_diagnostics
            )
        )

    def test_conditional_pass_matches_dense_system_and_fixes_parameters(self):
        layout = _layout()
        linearization = _markov_linearization(layout, seed=914)
        scale = np.linspace(0.65, 1.45, layout.total_dimension)
        damping = 0.08

        result = solve_scaled_conditional_ieks_step(
            linearization, scale, damping
        )
        expected = _dense_step(
            linearization, scale, damping, optimize_shared=False
        )

        np.testing.assert_array_equal(
            result.delta[layout.shared_slice], np.zeros(18)
        )
        np.testing.assert_allclose(
            result.scaled_delta, expected, rtol=3.0e-10, atol=3.0e-11
        )

    def test_nonadjacent_factor_is_rejected_before_smoothing(self):
        layout = _layout(("bag-a",), knot_count=3)
        linearization = _markov_linearization(layout)
        left = VariableKey(
            VariableKind.POSITION, bag_id="bag-a", knot_index=0
        )
        right = VariableKey(
            VariableKind.POSITION, bag_id="bag-a", knot_index=2
        )
        unsupported = _factor(
            np.ones(2),
            (
                (left, np.ones((2, 3))),
                (right, np.ones((2, 3))),
            ),
        )
        factors = tuple(
            FactorEvaluation(
                residual=np.zeros(key.dimension),
                jacobian_blocks=(JacobianBlock(key, np.eye(key.dimension)),),
                squared_error=0.0,
                active_set={},
            )
            for key in layout.variable_keys
        ) + (unsupported,)
        invalid = assemble_sparse_linearization(layout, factors)

        with self.assertRaisesRegex(ValueError, "adjacent knots"):
            validate_ieks_topology(invalid)
        with self.assertRaisesRegex(ValueError, "adjacent knots"):
            solve_scaled_ieks_step(
                invalid, np.ones(layout.total_dimension), 0.1
            )

    def test_solver_setting_selects_ieks_explicitly(self):
        settings = LMSettings(method="ieks", maximum_iterations=7)
        self.assertIs(settings.method, NonlinearSolverMethod.IEKS)
        self.assertEqual(settings.maximum_iterations, 7)
        with self.assertRaisesRegex(ValueError, "sparse_lm or ieks"):
            LMSettings(method="filter")


if __name__ == "__main__":
    unittest.main()
