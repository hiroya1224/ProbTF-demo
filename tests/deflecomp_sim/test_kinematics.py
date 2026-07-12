import unittest
from pathlib import Path

import numpy as np

from deflecomp_core.observation.imu_frame_config import identity_imu_frame_config
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.external_wrench import generalized_external_wrench
from deflecomp_sim.sensor_simulator import build_imu_kinematic_samples


class KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        urdf = (
            root
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


if __name__ == "__main__":
    unittest.main()
