from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import euler_xyz_to_matrix, matrix_to_quaternion
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
    VehicleParameterJacobian,
)
from grape_param_estim.system import (
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


def _inertia_coordinates(matrix):
    return np.asarray(
        (
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 2],
        ),
        dtype=float,
    )


def _central_difference_jacobian(chart, coordinates, step=2.0e-4):
    mass = np.zeros(PARAMETER_DIMENSION, dtype=float)
    inertia = np.zeros((3, 3, PARAMETER_DIMENSION), dtype=float)
    cog_offset = np.zeros((3, PARAMETER_DIMENSION), dtype=float)
    force_effectiveness = np.zeros((4, PARAMETER_DIMENSION), dtype=float)
    torque_effectiveness = np.zeros((4, PARAMETER_DIMENSION), dtype=float)
    stencil = ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0))

    for column in range(PARAMETER_DIMENSION):
        direction = np.zeros(PARAMETER_DIMENSION, dtype=float)
        direction[column] = step
        for offset, coefficient in stencil:
            parameters = chart.decode(coordinates + offset * direction)
            mass[column] += coefficient * parameters.mass
            inertia[:, :, column] += coefficient * parameters.inertia
            cog_offset[:, column] += coefficient * parameters.cog_offset
            force_effectiveness[:, column] += (
                coefficient * parameters.force_effectiveness
            )
            torque_effectiveness[:, column] += (
                coefficient * parameters.torque_effectiveness
            )

    denominator = 12.0 * step
    return VehicleParameterJacobian(
        mass=mass / denominator,
        inertia=inertia / denominator,
        cog_offset=cog_offset / denominator,
        force_effectiveness=force_effectiveness / denominator,
        torque_effectiveness=torque_effectiveness / denominator,
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

    def assert_jacobian_matches_central_difference(self, coordinates):
        decoded, analytic = self.chart.decode_with_jacobian(coordinates)
        reference = _central_difference_jacobian(self.chart, coordinates)

        expected = self.chart.decode(coordinates)
        self.assertEqual(decoded.mass, expected.mass)
        np.testing.assert_array_equal(decoded.inertia, expected.inertia)
        np.testing.assert_array_equal(decoded.cog_offset, expected.cog_offset)
        np.testing.assert_array_equal(
            decoded.force_effectiveness, expected.force_effectiveness
        )
        np.testing.assert_array_equal(
            decoded.torque_effectiveness, expected.torque_effectiveness
        )
        np.testing.assert_allclose(
            analytic.mass, reference.mass, rtol=2.0e-10, atol=2.0e-11
        )
        np.testing.assert_allclose(
            analytic.inertia,
            reference.inertia,
            rtol=3.0e-9,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            analytic.cog_offset,
            reference.cog_offset,
            rtol=0.0,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            analytic.force_effectiveness,
            reference.force_effectiveness,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            analytic.torque_effectiveness,
            reference.torque_effectiveness,
            rtol=2.0e-10,
            atol=2.0e-11,
        )

    def test_analytic_jacobian_matches_random_central_differences(self):
        generator = np.random.RandomState(314159)
        for _ in range(8):
            coordinates = generator.normal(
                0.0, 0.35, PARAMETER_DIMENSION
            )
            self.assert_jacobian_matches_central_difference(coordinates)

    def test_inertia_jacobian_handles_repeated_and_near_eigenvalues(self):
        orthogonal, _ = np.linalg.qr(
            np.asarray(
                (
                    (1.0, 2.0, -1.0),
                    (-2.0, 1.0, 0.5),
                    (0.7, -0.4, 1.0),
                )
            )
        )
        cases = [np.zeros((3, 3), dtype=float)]
        for eigenvalues in (
            (0.21, 0.21, -0.13),
            (0.21, 0.21 + 1.0e-11, -0.13),
        ):
            cases.append(
                orthogonal @ np.diag(eigenvalues) @ orthogonal.T
            )

        for relative_log_inertia in cases:
            coordinates = np.zeros(PARAMETER_DIMENSION, dtype=float)
            coordinates[0] = 0.09
            coordinates[1:7] = _inertia_coordinates(
                relative_log_inertia
            )
            coordinates[7:10] = (0.01, -0.02, 0.03)
            coordinates[10:18] = np.linspace(-0.12, 0.16, 8)
            self.assert_jacobian_matches_central_difference(coordinates)

    def test_decode_with_jacobian_has_strict_finite_shapes(self):
        _, jacobian = self.chart.decode_with_jacobian(
            np.zeros(PARAMETER_DIMENSION)
        )
        self.assertIsInstance(jacobian, VehicleParameterJacobian)
        expected_shapes = {
            "mass": (PARAMETER_DIMENSION,),
            "inertia": (3, 3, PARAMETER_DIMENSION),
            "cog_offset": (3, PARAMETER_DIMENSION),
            "force_effectiveness": (4, PARAMETER_DIMENSION),
            "torque_effectiveness": (4, PARAMETER_DIMENSION),
        }
        for name, shape in expected_shapes.items():
            value = getattr(jacobian, name)
            self.assertEqual(value.shape, shape)
            self.assertTrue(np.all(np.isfinite(value)))

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

        decoded, jacobian = self.chart.decode_with_jacobian(base)
        self.assertEqual(decoded.mass, original.mass)
        self.assertAlmostEqual(jacobian.mass @ direction, original.mass)
        np.testing.assert_allclose(
            np.tensordot(jacobian.inertia, direction, axes=(2, 0)),
            original.inertia,
            rtol=1.0e-13,
            atol=1.0e-16,
        )
        np.testing.assert_allclose(
            jacobian.cog_offset @ direction, np.zeros(3)
        )
        np.testing.assert_allclose(
            jacobian.force_effectiveness @ direction,
            original.force_effectiveness,
        )
        np.testing.assert_allclose(
            jacobian.torque_effectiveness @ direction, np.zeros(4)
        )

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
            with self.assertRaisesRegex(ValueError, "18 finite values"):
                self.chart.decode_with_jacobian(invalid)
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
