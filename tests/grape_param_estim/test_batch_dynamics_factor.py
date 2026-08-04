from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.batch.factors.dynamics import (
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_COMPONENT_NAMES,
    BODY_WRENCH_COMPONENT_UNITS,
    BODY_WRENCH_QUANTITY,
    body_wrench_statistical_residual,
    diagonal_q_log_normalization,
    evaluate_dynamics_factor,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.geometry import so3_exp
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import GrapeGeometry, VehicleParameters


def _definition():
    return DiagonalQDefinition(
        residual_quantity=BODY_WRENCH_QUANTITY,
        component_names=BODY_WRENCH_COMPONENT_NAMES,
        component_units=BODY_WRENCH_COMPONENT_UNITS,
        interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
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
        definition = _definition()
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

    def test_body_wrench_static_jacobian_matches_test_only_difference(self):
        definition = _definition()

        def evaluate(coordinates):
            raw = self._raw(coordinates)
            return body_wrench_statistical_residual(
                "bag-a",
                4,
                raw,
                definition,
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

    def test_body_wrench_residual_has_physical_common_scale_units(self):
        body_definition = _definition()
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
        np.testing.assert_allclose(
            body1.residual,
            np.exp(0.4) * body0.residual,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            body0.jacobian.static_parameters @ ridge,
            body0.residual,
            rtol=3.0e-13,
            atol=3.0e-13,
        )

    def test_q_gaussian_normalization_uses_q_over_variable_dt(self):
        q = np.asarray((2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        dt = np.asarray((0.01, 0.03))
        spectral = diagonal_q_log_normalization(
            q,
            dt,
            _definition(),
        )
        expected = 0.5 * np.sum(
            np.log(2.0 * np.pi * q[None, :] / dt[:, None])
        )
        self.assertAlmostEqual(spectral, expected)
    def test_non_body_wrench_or_non_continuous_definition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "body_wrench"):
            DiagonalQDefinition(
                residual_quantity="specific_acceleration",
                component_names=BODY_WRENCH_COMPONENT_NAMES,
                component_units=BODY_WRENCH_COMPONENT_UNITS,
                interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
            )
        with self.assertRaisesRegex(TypeError, "QIntervalModel"):
            DiagonalQDefinition(
                residual_quantity=BODY_WRENCH_QUANTITY,
                component_names=BODY_WRENCH_COMPONENT_NAMES,
                component_units=BODY_WRENCH_COMPONENT_UNITS,
                interval_model="fixed_interval_covariance",
            )


if __name__ == "__main__":
    unittest.main()
