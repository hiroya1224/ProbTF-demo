import math
import unittest

import numpy as np

from grape_param_estim.dynamics import (
    PARAMETER_NAMES,
    parameters_to_inertia,
    physical_parameter_mask,
    predict_wrench,
)
from grape_param_estim.urdf_inertia import composite_inertia_from_urdf


def _parameters(mass=2.0, center=(0.1, -0.2, 0.3), inertia=None):
    if inertia is None:
        inertia = np.diag([1.0, 1.2, 1.5])
    inertia = np.asarray(inertia, dtype=float)
    return np.array(
        [
            mass,
            center[0],
            center[1],
            center[2],
            inertia[0, 0],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 1],
            inertia[1, 2],
            inertia[2, 2],
        ],
        dtype=float,
    )


_MINIMAL_URDF = """
<robot name="two_link">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>
  <link name="payload">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.15"/>
    </inertial>
  </link>
  <joint name="payload_yaw" type="revolute">
    <parent link="base"/>
    <child link="payload"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


class DynamicsTest(unittest.TestCase):
    def test_parameter_order_and_symmetric_inertia_conversion(self):
        self.assertEqual(
            PARAMETER_NAMES,
            (
                "mass",
                "cog_x",
                "cog_y",
                "cog_z",
                "inertia_xx",
                "inertia_xy",
                "inertia_xz",
                "inertia_yy",
                "inertia_yz",
                "inertia_zz",
            ),
        )
        parameters = np.array(
            [2.0, 0.1, 0.2, 0.3, 1.0, 0.1, -0.2, 1.5, 0.3, 1.8]
        )
        np.testing.assert_allclose(
            parameters_to_inertia(parameters),
            [[1.0, 0.1, -0.2], [0.1, 1.5, 0.3], [-0.2, 0.3, 1.8]],
        )

    def test_physical_mask_rejects_each_required_invalidity(self):
        valid = _parameters()
        invalid_mass = valid.copy()
        invalid_mass[0] = 0.0
        invalid_spd = _parameters(inertia=np.diag([1.0, -0.1, 1.0]))
        invalid_triangle = _parameters(inertia=np.diag([1.0, 1.0, 2.1]))
        invalid_finite = valid.copy()
        invalid_finite[2] = np.nan
        particles = np.stack(
            [valid, invalid_mass, invalid_spd, invalid_triangle, invalid_finite]
        )
        np.testing.assert_array_equal(
            physical_parameter_mask(particles),
            [True, False, False, False, False],
        )
        self.assertTrue(bool(physical_parameter_mask(valid)))

    def test_predict_wrench_matches_equations_and_vectorizes(self):
        parameters = _parameters()
        specific = np.array([0.4, -1.1, 9.2])
        omega = np.array([0.3, -0.2, 0.5])
        alpha = np.array([-0.7, 0.6, 0.2])

        mass = parameters[0]
        center = parameters[1:4]
        first_moment = mass * center
        inertia_com = parameters_to_inertia(parameters)
        inertia_origin = inertia_com + mass * (
            float(center @ center) * np.eye(3) - np.outer(center, center)
        )
        expected_force = (
            mass * specific
            + np.cross(alpha, first_moment)
            + np.cross(omega, np.cross(omega, first_moment))
        )
        expected_torque = (
            inertia_origin @ alpha
            + np.cross(omega, inertia_origin @ omega)
            + np.cross(first_moment, specific)
        )
        expected = np.hstack((expected_force, expected_torque))

        np.testing.assert_allclose(
            predict_wrench(parameters, specific, omega, alpha),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )

        second = _parameters(mass=3.0, center=(-0.05, 0.1, 0.2))
        batch = predict_wrench(np.stack((parameters, second)), specific, omega, alpha)
        self.assertEqual(batch.shape, (2, 6))
        np.testing.assert_allclose(batch[0], expected, rtol=1e-13, atol=1e-13)
        self.assertTrue(np.all(np.isfinite(batch)))

    def test_predict_wrench_rejects_nonphysical_or_nonfinite_inputs(self):
        invalid = _parameters(inertia=np.diag([1.0, 1.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "non-physical"):
            predict_wrench(invalid, np.zeros(3), np.zeros(3), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_wrench(
                _parameters(),
                [np.nan, 0.0, 0.0],
                np.zeros(3),
                np.zeros(3),
            )

    def test_urdf_composite_inertia_follows_joint_configuration(self):
        result = composite_inertia_from_urdf(
            _MINIMAL_URDF,
            q={"payload_yaw": math.pi / 2.0},
            base_link="base",
        )
        expected_center = np.array([1.0 / 3.0, 1.0 / 6.0, 0.0])
        expected_inertia = np.array(
            [
                [0.57 - 1.0 / 12.0, -0.5 + 1.0 / 6.0, 0.0],
                [-0.5 + 1.0 / 6.0, 1.4 - 1.0 / 3.0, 0.0],
                [0.0, 0.0, 1.8 - 5.0 / 12.0],
            ]
        )
        self.assertEqual(result.base_link, "base")
        self.assertAlmostEqual(result.mass, 3.0)
        np.testing.assert_allclose(result.center_of_mass, expected_center, atol=1e-14)
        np.testing.assert_allclose(result.inertia_com, expected_inertia, atol=1e-14)
        self.assertEqual(result.reachable_links, ("base", "payload"))
        self.assertEqual(result.parameters.shape, (10,))
        self.assertTrue(bool(physical_parameter_mask(result.parameters)))

    def test_urdf_defaults_joint_positions_to_zero(self):
        result = composite_inertia_from_urdf(_MINIMAL_URDF)
        np.testing.assert_allclose(
            result.center_of_mass,
            [0.5, 0.0, 0.0],
            atol=1e-14,
        )

    def test_urdf_internal_base_reexpresses_the_whole_robot(self):
        angle = math.pi / 2.0
        root_result = composite_inertia_from_urdf(
            _MINIMAL_URDF,
            q={"payload_yaw": angle},
            base_link="base",
        )
        payload_result = composite_inertia_from_urdf(
            _MINIMAL_URDF,
            q={"payload_yaw": angle},
            base_link="payload",
        )

        rotation_base_payload = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation_base_payload = np.array([1.0, 0.0, 0.0])
        expected_center = rotation_base_payload.T @ (
            root_result.center_of_mass - translation_base_payload
        )
        expected_inertia = (
            rotation_base_payload.T
            @ root_result.inertia_com
            @ rotation_base_payload
        )

        self.assertAlmostEqual(payload_result.mass, root_result.mass)
        np.testing.assert_allclose(
            payload_result.center_of_mass,
            expected_center,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            payload_result.inertia_com,
            expected_inertia,
            atol=1e-14,
        )
        self.assertEqual(payload_result.reachable_links, ("base", "payload"))


if __name__ == "__main__":
    unittest.main()
