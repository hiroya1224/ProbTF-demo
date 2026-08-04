import unittest

import numpy as np

from grape_param_estim.batch.factors.prior import (
    evaluate_orientation_prior_factor,
    evaluate_vector_prior_factor,
)
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_exp


class BatchPriorFactorTests(unittest.TestCase):
    def test_vector_prior_supports_static_bias_and_knot_blocks(self):
        keys = (
            VariableKey(VariableKind.STATIC_PARAMETERS),
            VariableKey(VariableKind.GYRO_BIAS, bag_id="bag-a"),
            VariableKey(
                VariableKind.CONTROLLER_INTEGRAL,
                bag_id="bag-a",
                knot_index=0,
            ),
        )
        for key in keys:
            with self.subTest(kind=key.kind.value):
                value = np.linspace(-0.2, 0.3, key.dimension)
                mean = np.linspace(0.1, -0.1, key.dimension)
                whitening = np.diag(
                    np.linspace(0.8, 1.4, key.dimension)
                )
                factor = evaluate_vector_prior_factor(
                    key,
                    value,
                    mean,
                    whitening,
                )
                np.testing.assert_allclose(
                    factor.residual,
                    whitening @ (value - mean),
                    atol=1.0e-15,
                )
                np.testing.assert_array_equal(
                    factor.jacobian_blocks[0].value, whitening
                )

    def test_orientation_prior_jacobian_matches_right_difference(self):
        key = VariableKey(
            VariableKind.ORIENTATION_TANGENT,
            bag_id="bag-a",
            knot_index=0,
        )
        mean = so3_exp((0.3, -0.1, 0.2))
        rotation = mean @ so3_exp((0.2, -0.15, 0.08))
        whitening = np.asarray(
            ((1.5, 0.1, 0.0), (0.0, 1.2, -0.1), (0.0, 0.0, 0.9))
        )
        factor = evaluate_orientation_prior_factor(
            key,
            rotation,
            mean,
            whitening,
        )
        step = 1.0e-7
        numerical = np.empty((3, 3), dtype=float)
        for coordinate in range(3):
            direction = np.zeros(3, dtype=float)
            direction[coordinate] = step
            plus = evaluate_orientation_prior_factor(
                key,
                rotation @ so3_exp(direction),
                mean,
                whitening,
            ).residual
            minus = evaluate_orientation_prior_factor(
                key,
                rotation @ so3_exp(-direction),
                mean,
                whitening,
            ).residual
            numerical[:, coordinate] = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            factor.jacobian_blocks[0].value,
            numerical,
            rtol=4.0e-8,
            atol=4.0e-9,
        )
        self.assertFalse(factor.active_set["rotation_log_near_pi"][0])

    def test_wrong_key_shapes_and_singular_weighting_are_rejected(self):
        orientation_key = VariableKey(
            VariableKind.ORIENTATION_TANGENT,
            bag_id="bag-a",
            knot_index=0,
        )
        position_key = VariableKey(
            VariableKind.POSITION,
            bag_id="bag-a",
            knot_index=0,
        )
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            evaluate_vector_prior_factor(
                orientation_key,
                np.zeros(3),
                np.zeros(3),
                np.eye(3),
            )
        with self.assertRaisesRegex(ValueError, "orientation"):
            evaluate_orientation_prior_factor(
                position_key,
                np.eye(3),
                np.eye(3),
                np.eye(3),
            )
        with self.assertRaisesRegex(ValueError, "full rank"):
            evaluate_vector_prior_factor(
                position_key,
                np.zeros(3),
                np.zeros(3),
                np.zeros((3, 3)),
            )


if __name__ == "__main__":
    unittest.main()
