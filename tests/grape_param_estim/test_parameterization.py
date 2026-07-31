from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import euler_xyz_to_matrix, matrix_to_quaternion
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.system import (
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class VehicleParameterChartTests(unittest.TestCase):
    def setUp(self):
        self.nominal = VehicleParameters.nominal()
        self.chart = VehicleParameterChart(self.nominal)

    def test_zero_is_nominal_and_dimension_is_fixed(self):
        self.assertEqual(PARAMETER_DIMENSION, 18)
        decoded = self.chart.decode(np.zeros(PARAMETER_DIMENSION))

        self.assertEqual(decoded.mass, self.nominal.mass)
        np.testing.assert_allclose(decoded.inertia, self.nominal.inertia)
        np.testing.assert_allclose(decoded.cog_offset, self.nominal.cog_offset)
        np.testing.assert_allclose(
            decoded.force_effectiveness, self.nominal.force_effectiveness
        )
        np.testing.assert_allclose(
            decoded.torque_effectiveness, self.nominal.torque_effectiveness
        )
        np.testing.assert_allclose(
            self.chart.encode(self.nominal), np.zeros(PARAMETER_DIMENSION),
            atol=1.0e-14,
        )

    def test_forward_inverse_round_trip_includes_full_inertia(self):
        coordinates = np.asarray(
            (
                0.17,
                0.11,
                -0.08,
                0.05,
                0.035,
                -0.027,
                0.019,
                0.014,
                -0.009,
                0.006,
                -0.13,
                0.07,
                0.03,
                -0.05,
                0.09,
                -0.04,
                0.12,
                -0.06,
            )
        )
        parameters = self.chart.decode(coordinates)
        recovered = self.chart.encode(parameters)
        reconstructed = self.chart.decode(recovered)

        np.testing.assert_allclose(recovered, coordinates, atol=2.0e-14)
        self.assertAlmostEqual(reconstructed.mass, parameters.mass)
        np.testing.assert_allclose(
            reconstructed.inertia, parameters.inertia, atol=2.0e-16
        )
        np.testing.assert_allclose(
            reconstructed.cog_offset, parameters.cog_offset
        )
        np.testing.assert_allclose(
            reconstructed.force_effectiveness,
            parameters.force_effectiveness,
        )
        np.testing.assert_allclose(
            reconstructed.torque_effectiveness,
            parameters.torque_effectiveness,
        )
        self.assertGreater(
            np.max(
                np.abs(
                    parameters.inertia
                    - np.diag(np.diag(parameters.inertia))
                )
            ),
            0.0,
        )

    def test_every_finite_chart_sample_has_positive_physical_parameters(self):
        generator = np.random.RandomState(12)
        for _ in range(50):
            coordinates = generator.normal(0.0, 0.8, PARAMETER_DIMENSION)
            parameters = self.chart.decode(coordinates)
            self.assertGreater(parameters.mass, 0.0)
            self.assertTrue(
                np.all(parameters.force_effectiveness > 0.0)
            )
            self.assertTrue(
                np.all(parameters.torque_effectiveness > 0.0)
            )
            np.testing.assert_allclose(
                parameters.inertia, parameters.inertia.T, atol=0.0
            )
            self.assertTrue(
                np.all(np.linalg.eigvalsh(parameters.inertia) > 0.0)
            )

    def test_ridge_direction_is_an_exact_common_dynamics_scale(self):
        base = np.asarray(
            (
                0.08,
                0.09,
                -0.04,
                0.02,
                0.025,
                -0.017,
                0.013,
                0.010,
                -0.006,
                0.004,
                -0.07,
                0.04,
                0.02,
                -0.03,
                0.06,
                -0.02,
                0.03,
                -0.05,
            )
        )
        scale_log = 0.37
        direction = self.chart.ridge_direction()
        original = self.chart.decode(base)
        scaled = self.chart.decode(base + scale_log * direction)
        factor = np.exp(scale_log)

        self.assertEqual(direction.shape, (PARAMETER_DIMENSION,))
        self.assertAlmostEqual(scaled.mass, factor * original.mass)
        np.testing.assert_allclose(
            scaled.inertia,
            factor * original.inertia,
            rtol=1.0e-13,
            atol=1.0e-16,
        )
        np.testing.assert_allclose(
            scaled.force_effectiveness,
            factor * original.force_effectiveness,
        )
        np.testing.assert_allclose(scaled.cog_offset, original.cog_offset)
        np.testing.assert_allclose(
            scaled.torque_effectiveness, original.torque_effectiveness
        )

        state = RigidBodyState(
            position=(0.2, -0.1, 0.9),
            orientation_xyzw=matrix_to_quaternion(
                euler_xyz_to_matrix((0.17, -0.11, 0.23))
            ),
            linear_velocity=(0.4, -0.2, 0.1),
            angular_velocity=(0.3, -0.25, 0.18),
        )
        actuators = ActuatorState(
            thrust=(5.3, 6.1, 5.8, 5.5),
            gimbal_angle=(0.12, -0.08, 0.09, -0.05),
        )
        geometry = GrapeGeometry.grape()
        original_derivative = FullSixDofPlant(
            original, geometry
        ).derivative(0.4, state.as_vector(), actuators)
        scaled_derivative = FullSixDofPlant(
            scaled, geometry
        ).derivative(0.4, state.as_vector(), actuators)
        np.testing.assert_allclose(
            scaled_derivative, original_derivative, atol=3.0e-14
        )

    def test_shape_finite_type_and_unrepresented_drag_are_rejected(self):
        for invalid in (
            np.zeros(PARAMETER_DIMENSION - 1),
            np.zeros(PARAMETER_DIMENSION + 1),
            np.full(PARAMETER_DIMENSION, np.nan),
        ):
            with self.assertRaisesRegex(ValueError, "18 finite values"):
                self.chart.decode(invalid)
        with self.assertRaises(TypeError):
            VehicleParameterChart(object())
        with self.assertRaises(TypeError):
            self.chart.encode(object())

        changed_drag = replace(
            self.nominal, linear_drag=np.asarray((0.1, 0.0, 0.0))
        )
        with self.assertRaisesRegex(ValueError, "drag"):
            self.chart.encode(changed_drag)


if __name__ == "__main__":
    unittest.main()
