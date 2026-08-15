from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from _support import synthetic_problem_parts
from savgol_trajectory import (
    GeometricSavitzkyGolayPose,
    left_jacobian_with_directional_derivative,
    skew,
)
from single_bag_savgol_covariance import (
    GRAVITY_WORLD,
    build_sg_covariance,
    generalized_acceleration_from_xi,
    propagation_jacobian,
)


class SavgolCovarianceTests(unittest.TestCase):
    def test_import_and_polynomial_exactness(self):
        time = np.linspace(-1.0, 1.0, 101)
        position = np.column_stack(
            (
                1.0 + 2.0 * time + 3.0 * time**2,
                time**3,
                -2.0 + 0.5 * time,
            )
        )
        orientation = np.zeros((time.size, 4))
        orientation[:, 3] = 1.0
        trajectory = GeometricSavitzkyGolayPose(
            time_axis=time,
            sensor_position=position,
            sensor_orientation_xyzw=orientation,
            pose_sensor_to_body_rotation=np.eye(3),
            window_seconds=0.5,
            degree=3,
        )
        query = np.asarray((-0.5, -0.1, 0.3, 0.6))
        value = trajectory.evaluate(query, centered=True)
        expected_acceleration = np.column_stack(
            (np.full(query.size, 6.0), 6.0 * query, np.zeros(query.size))
        )
        self.assertTrue(
            np.allclose(
                value.sensor_acceleration_world,
                expected_acceleration,
                atol=2e-11,
            )
        )
        self.assertTrue(
            all(
                window.translation.pseudoinverse.ndim == 2
                for window in value.local_windows
            )
        )

    def test_so3_rotation_frame_sanity(self):
        time = np.linspace(-1.0, 1.0, 121)
        theta = 0.3 * time + 0.2 * time**2
        rotations = Rotation.from_rotvec(
            np.column_stack((0 * time, 0 * time, theta))
        )
        trajectory = GeometricSavitzkyGolayPose(
            time_axis=time,
            sensor_position=np.zeros((time.size, 3)),
            sensor_orientation_xyzw=rotations.as_quat(),
            pose_sensor_to_body_rotation=np.eye(3),
            window_seconds=0.4,
            degree=3,
        )
        query = np.asarray((-0.4, 0.0, 0.35))
        value = trajectory.evaluate(query, centered=True)
        expected_omega = 0.3 + 0.4 * query
        self.assertTrue(
            np.allclose(value.body_angular_velocity[:, 2], expected_omega, atol=2e-10)
        )
        self.assertTrue(
            np.allclose(value.body_angular_acceleration[:, 2], 0.4, atol=2e-10)
        )
        expected_rotation = Rotation.from_rotvec(
            np.column_stack(
                (0 * query, 0 * query, 0.3 * query + 0.2 * query**2)
            )
        ).as_matrix()
        self.assertTrue(np.allclose(value.body_rotation, expected_rotation, atol=2e-10))

    def test_hover_and_free_fall_specific_force(self):
        xi = np.zeros(12)
        hover = generalized_acceleration_from_xi(xi, np.eye(3))
        self.assertTrue(np.isclose(np.linalg.norm(hover[:3]), 9.80665))
        xi[:3] = GRAVITY_WORLD
        free_fall = generalized_acceleration_from_xi(xi, np.eye(3))
        self.assertTrue(np.allclose(free_fall[:3], 0.0))

    def test_covariance_shapes_symmetry_and_ablation_blocks(self):
        dataset, _model, _actuator = synthetic_problem_parts()
        sg = dataset.sg
        modes = (
            "full",
            "identity",
            "diagonal",
            "block_s_alpha",
            "full_no_R_uncertainty_in_s",
            "full_no_position_rotation_cross",
            "global_full",
        )
        values = {
            mode: build_sg_covariance(sg, degree=3, mode=mode) for mode in modes
        }
        count = sg.time.size
        for value in values.values():
            self.assertEqual(value.local_omega.shape, (count, 6, 6))
            self.assertEqual(value.local_sigma_xi.shape, (count, 12, 12))
            self.assertEqual(value.local_sigma_z.shape, (count, 6, 6))
            self.assertTrue(
                np.allclose(
                    value.local_sigma_z,
                    value.local_sigma_z.transpose(0, 2, 1),
                )
            )
        self.assertTrue(
            np.allclose(values["identity"].local_sigma_z, np.eye(6)[None])
        )
        diagonal = values["diagonal"].local_sigma_z
        self.assertTrue(np.allclose(diagonal, diagonal * np.eye(6)[None]))
        block = values["block_s_alpha"].local_sigma_z
        self.assertTrue(np.allclose(block[:, :3, 3:], 0.0))
        cross_zero = values["full_no_position_rotation_cross"].local_omega
        self.assertTrue(np.allclose(cross_zero[:, :3, 3:], 0.0))
        global_covariance = values["global_full"].local_sigma_z
        self.assertTrue(np.allclose(global_covariance, global_covariance[0]))

    def test_so3_directional_derivatives_closed_form_symmetry_bilinearity(self):
        eta = np.asarray((0.2, -0.1, 0.3))
        zeta = np.asarray((-0.4, 0.5, 0.1))
        jacobian, first, second = left_jacobian_with_directional_derivative(
            np.zeros(3), eta, zeta
        )
        self.assertTrue(np.array_equal(jacobian, np.eye(3)))
        self.assertTrue(np.allclose(first, 0.5 * skew(eta)))
        self.assertTrue(
            np.allclose(
                second,
                (skew(eta) @ skew(zeta) + skew(zeta) @ skew(eta)) / 6.0,
            )
        )

        phi = np.asarray((0.1, -0.05, 0.08))
        eta_two = np.asarray((0.07, 0.03, -0.02))
        zeta_two = np.asarray((-0.06, 0.04, 0.09))
        _, _, combined_first = left_jacobian_with_directional_derivative(
            phi, eta + eta_two, zeta
        )
        _, _, separate_first_a = left_jacobian_with_directional_derivative(
            phi, eta, zeta
        )
        _, _, separate_first_b = left_jacobian_with_directional_derivative(
            phi, eta_two, zeta
        )
        self.assertTrue(
            np.allclose(combined_first, separate_first_a + separate_first_b)
        )
        _, _, combined_second = left_jacobian_with_directional_derivative(
            phi, eta, zeta + zeta_two
        )
        _, _, separate_second_a = left_jacobian_with_directional_derivative(
            phi, eta, zeta
        )
        _, _, separate_second_b = left_jacobian_with_directional_derivative(
            phi, eta, zeta_two
        )
        self.assertTrue(
            np.allclose(combined_second, separate_second_a + separate_second_b)
        )
        _, _, reversed_directions = left_jacobian_with_directional_derivative(
            phi, zeta, eta
        )
        self.assertTrue(np.allclose(separate_first_a, reversed_directions))

    def test_covariance_propagation_zero_rotation_special_case(self):
        acceleration = np.asarray((0.3, -0.4, 9.5))
        rho2 = np.asarray((-0.4, 0.2, 0.1))
        xi = np.concatenate(
            (acceleration, np.zeros(3), np.zeros(3), rho2)
        )
        jacobian = propagation_jacobian(xi, np.eye(3))
        expected = np.zeros((6, 12))
        expected[:3, :3] = np.eye(3)
        expected[:3, 3:6] = skew(acceleration - GRAVITY_WORLD)
        expected[3:, 3:6] = 0.5 * skew(rho2)
        expected[3:, 9:12] = np.eye(3)
        self.assertTrue(np.allclose(jacobian, expected, atol=2e-15))

        no_geometric = propagation_jacobian(
            xi, np.eye(3), geometric_correction=False
        )
        expected[3:, 3:6] = skew(rho2)
        self.assertTrue(np.allclose(no_geometric, expected, atol=2e-15))


if __name__ == "__main__":
    unittest.main()
