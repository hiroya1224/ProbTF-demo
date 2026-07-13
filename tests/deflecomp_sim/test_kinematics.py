import unittest
from pathlib import Path

import numpy as np

from deflecomp_core.observation.imu_frame_config import identity_imu_frame_config
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.external_wrench import (
    external_force_arrow_points,
    frame_wrench_in_world,
    generalized_external_wrench,
)
from deflecomp_sim.sensor_simulator import build_imu_kinematic_samples


class KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        urdf = (
            cls.root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_description"
            / "urdf"
            / "simple6r.urdf"
        )
        cls.robot = RobotArm(str(urdf))

    def test_imu_samples_are_computed_without_ros_messages(self):
        zeros = np.zeros(self.robot.nv, dtype=float)
        config = identity_imu_frame_config(self.robot.tip_link_name)

        samples = build_imu_kinematic_samples(self.robot, [config], zeros, zeros, zeros)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].frame_id, self.robot.tip_link_name)
        self.assertTrue(np.all(np.isfinite(samples[0].linear_acceleration)))
        self.assertAlmostEqual(np.linalg.norm(samples[0].orientation_xyzw), 1.0)

    def test_external_wrench_maps_to_finite_joint_torque(self):
        zeros = np.zeros(self.robot.nv, dtype=float)
        joint_torque = generalized_external_wrench(
            self.robot,
            zeros,
            self.robot.tip_link_name,
            force=[1.0, 0.0, 0.0],
            torque=[0.0, 0.0, 0.0],
        )

        self.assertEqual(joint_torque.shape, (self.robot.nv,))
        self.assertTrue(np.all(np.isfinite(joint_torque)))

    def test_world_force_remains_world_resolved_at_rotated_target_frame(self):
        q = np.array([0.2, -0.4, 0.3, 0.1, -0.2, 0.5], dtype=float)
        expected_force = np.array([0.0, 0.0, -4.905], dtype=float)

        point, force_world, torque_world = frame_wrench_in_world(
            self.robot,
            q,
            self.robot.tip_link_name,
            force=expected_force,
            torque=[0.0, 0.0, 0.0],
            reference_frame="world",
        )

        self.assertEqual(point.shape, (3,))
        np.testing.assert_allclose(force_world, expected_force, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(torque_world, np.zeros(3), atol=0.0, rtol=0.0)

    def test_external_force_arrow_points_in_force_direction_from_application_point(self):
        application_point = np.array([0.1, -0.2, 0.8], dtype=float)
        force_world = np.array([0.0, 0.0, -4.905], dtype=float)

        start, end = external_force_arrow_points(
            application_point,
            force_world,
            scale=0.05,
        )

        np.testing.assert_allclose(start, application_point, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            end - start,
            0.05 * force_world,
            atol=1.0e-15,
            rtol=0.0,
        )
        self.assertLess(end[2], start[2])

    def test_rviz_uses_fixed_frame_external_force_marker(self):
        rviz_config = (
            self.root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_description"
            / "rviz"
            / "deflecomp.rviz"
        ).read_text()
        sim_config = (
            self.root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_sim"
            / "config"
            / "sim_params.yaml"
        ).read_text()

        marker_topic = "/deflecomp_sim/external_wrench_marker"
        self.assertIn(f"Marker Topic: {marker_topic}", rviz_config)
        self.assertIn(f"external_wrench_marker_topic: {marker_topic}", sim_config)
        self.assertNotIn("Class: rviz/WrenchStamped", rviz_config)


if __name__ == "__main__":
    unittest.main()
