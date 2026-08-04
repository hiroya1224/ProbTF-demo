from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.batch.factors.dynamics import (
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
    body_wrench_statistical_residual,
    diagonal_q_log_normalization,
    evaluate_dynamics_factor,
    specific_acceleration_statistical_residual,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.geometry import so3_exp
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import GrapeGeometry, VehicleParameters


def _definition(quantity, interval_model):
    return DiagonalQDefinition(
        residual_quantity=quantity,
        component_names=("x", "y", "z", "roll", "pitch", "yaw"),
        component_units=("u",) * 6,
        interval_model=interval_model,
    )


class BatchDynamicsFactorTests(unittest.TestCase):
    def setUp(self):
        nominal = replace(
            VehicleParameters.nominal(),
            cog_offset=np.asarray((0.01, -0.02, 0.015)),
        )
        self.chart = VehicleParameterChart(nominal)
        self.coordinates = np.asarray(
            (
                0.06, 0.02, -0.01, 0.03, 0.01, -0.008, 0.006,
                0.004, -0.003, 0.005, 0.03, -0.02, 0.01, -0.04,
                0.02, -0.01, 0.03, -0.02,
            )
        )
        self.values = {
            "rotation_left": so3_exp((0.13, -0.09, 0.16)),
            "rotation_right": so3_exp((0.13, -0.09, 0.16))
            @ so3_exp((0.06, -0.04, 0.03)),
            "linear_velocity_left": np.asarray((0.2, -0.1, 0.05)),
            "linear_velocity_right": np.asarray((0.24, -0.08, 0.02)),
            "angular_velocity_left": np.asarray((0.3, -0.2, 0.15)),
            "angular_velocity_right": np.asarray((0.28, -0.16, 0.19)),
            "actuator_thrust_left": np.asarray((5.8, 6.2, 5.9, 6.4)),
            "actuator_thrust_right": np.asarray((6.0, 6.1, 6.2, 6.3)),
            "gimbal_angle_left": np.asarray((0.1, -0.12, 0.08, -0.09)),
            "gimbal_angle_right": np.asarray((0.11, -0.1, 0.06, -0.13)),
            "time_step": 0.025,
            "parameter_chart": self.chart,
            "parameter_coordinates": self.coordinates,
            "geometry": GrapeGeometry.grape(),
        }

    def _raw(self, coordinates=None):
        values = dict(self.values)
        if coordinates is not None:
            values["parameter_coordinates"] = coordinates
        return evaluate_raw_dynamics_residual(**values)

    def test_body_wrench_factor_uses_spectral_density_dt_whitening(self):
        definition = _definition(
            BODY_WRENCH_QUANTITY,
            QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
        )
        raw = self._raw()
        statistical = body_wrench_statistical_residual(
            "bag-a", 4, raw, definition
        )
        q = np.asarray((2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        factor = evaluate_dynamics_factor(
            statistical, q, self.values["time_step"]
        )
        whitening = np.diag(
            np.sqrt(self.values["time_step"] / q)
        )
        np.testing.assert_allclose(factor.residual, whitening @ raw.residual)
        self.assertEqual(
            tuple(block.variable_key.kind for block in factor.jacobian_blocks),
            (
                VariableKind.STATIC_PARAMETERS,
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.LINEAR_VELOCITY,
                VariableKind.LINEAR_VELOCITY,
                VariableKind.ANGULAR_VELOCITY,
                VariableKind.ANGULAR_VELOCITY,
                VariableKind.ACTUATOR_THRUST,
                VariableKind.ACTUATOR_THRUST,
                VariableKind.GIMBAL_ANGLE,
                VariableKind.GIMBAL_ANGLE,
            ),
        )
        for block, raw_block in zip(
            factor.jacobian_blocks, statistical.jacobian_blocks
        ):
            np.testing.assert_allclose(
                block.value, whitening @ raw_block.value
            )

    def test_specific_static_jacobian_matches_test_only_difference(self):
        definition = _definition(
            SPECIFIC_ACCELERATION_QUANTITY,
            QIntervalModel.FIXED_INTERVAL_COVARIANCE,
        )

        def evaluate(coordinates):
            raw = self._raw(coordinates)
            return specific_acceleration_statistical_residual(
                "bag-a",
                4,
                raw,
                definition,
                self.chart,
                coordinates,
            )

        result = evaluate(self.coordinates)
        numerical = np.empty((6, 18))
        step = 1.0e-7
        for coordinate in range(18):
            direction = np.zeros(18)
            direction[coordinate] = step
            numerical[:, coordinate] = (
                evaluate(self.coordinates + direction).residual
                - evaluate(self.coordinates - direction).residual
            ) / (2.0 * step)
        np.testing.assert_allclose(
            result.jacobian.static_parameters,
            numerical,
            rtol=2.0e-5,
            atol=3.0e-7,
        )

    def test_coordinate_choice_exposes_common_scale_likelihood_difference(self):
        body_definition = _definition(
            BODY_WRENCH_QUANTITY,
            QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
        )
        specific_definition = _definition(
            SPECIFIC_ACCELERATION_QUANTITY,
            QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
        )
        ridge = self.chart.ridge_direction()
        shifted_coordinates = self.coordinates + 0.4 * ridge
        raw0 = self._raw(self.coordinates)
        raw1 = self._raw(shifted_coordinates)
        body0 = body_wrench_statistical_residual(
            "bag-a", 4, raw0, body_definition
        )
        body1 = body_wrench_statistical_residual(
            "bag-a", 4, raw1, body_definition
        )
        specific0 = specific_acceleration_statistical_residual(
            "bag-a",
            4,
            raw0,
            specific_definition,
            self.chart,
            self.coordinates,
        )
        specific1 = specific_acceleration_statistical_residual(
            "bag-a",
            4,
            raw1,
            specific_definition,
            self.chart,
            shifted_coordinates,
        )
        self.assertFalse(np.allclose(body0.residual, body1.residual))
        np.testing.assert_allclose(
            specific0.residual,
            specific1.residual,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            specific0.jacobian.static_parameters @ ridge,
            np.zeros(6),
            atol=2.0e-13,
        )

    def test_q_gaussian_normalization_matches_interval_definition(self):
        q = np.asarray((2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        dt = np.asarray((0.01, 0.03))
        spectral = diagonal_q_log_normalization(
            q,
            dt,
            _definition(
                BODY_WRENCH_QUANTITY,
                QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
            ),
        )
        expected = 0.5 * np.sum(
            np.log(2.0 * np.pi * q[None, :] / dt[:, None])
        )
        self.assertAlmostEqual(spectral, expected)
        fixed = diagonal_q_log_normalization(
            q,
            dt,
            _definition(
                BODY_WRENCH_QUANTITY,
                QIntervalModel.FIXED_INTERVAL_COVARIANCE,
            ),
        )
        expected_fixed = 0.5 * np.sum(
            np.log(np.broadcast_to(2.0 * np.pi * q, (2, 6)))
        )
        self.assertAlmostEqual(fixed, expected_fixed)

    def test_mismatched_q_quantity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "body_wrench"):
            body_wrench_statistical_residual(
                "bag-a",
                0,
                self._raw(),
                _definition(
                    SPECIFIC_ACCELERATION_QUANTITY,
                    QIntervalModel.FIXED_INTERVAL_COVARIANCE,
                ),
            )


if __name__ == "__main__":
    unittest.main()
