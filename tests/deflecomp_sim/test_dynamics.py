import unittest
from pathlib import Path

import numpy as np

from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.dynamic_simulator import DynamicParams, FlexibleJointSimulator


class DynamicSimulatorTest(unittest.TestCase):
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
        cls.urdf = str(urdf)

    def make_simulator(self, **overrides):
        robot = RobotArm(self.urdf)
        n = robot.nv
        values = {
            "K": np.ones(n, dtype=float) * 100.0,
            "D": np.ones(n, dtype=float) * 10.0,
            "use_pinv": False,
            "limit_position_low": robot.model.lowerPositionLimit,
            "limit_position_high": robot.model.upperPositionLimit,
            "integrator": "semi_implicit",
            "ref_tau": 0.0,
            "ref_max_vel": 0.0,
            "eq_mode": "dynamic",
        }
        values.update(overrides)
        simulator = FlexibleJointSimulator(
            robot=robot,
            params=DynamicParams(**values),
            spring_model=LinearSpringModel(),
        )
        return robot, simulator

    def test_implicit_damping_converges_for_stiff_light_joint_dynamics(self):
        robot, simulator = self.make_simulator()
        q_ref = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3], dtype=float)
        equilibrium = EquilibriumSolver(robot, simulator.spring_model).solve(
            theta_cmd=q_ref,
            kp_vec=simulator.K,
            theta_init=q_ref,
        )

        for _ in range(2000):
            q, qd = simulator.step(dt=0.001, q_ref=q_ref)

        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(np.isfinite(qd)))
        self.assertLess(np.linalg.norm(q - equilibrium), 1e-3)
        self.assertLess(np.linalg.norm(qd), 1e-3)

    def test_position_limit_cancels_only_outward_velocity(self):
        robot, simulator = self.make_simulator(
            K=np.ones(6, dtype=float),
            D=np.ones(6, dtype=float),
        )
        upper = robot.model.upperPositionLimit.copy()
        simulator.reset(q=upper, qd=np.ones(robot.nv, dtype=float))

        q, qd = simulator.step(dt=0.001, q_ref=upper)

        np.testing.assert_allclose(q, upper, atol=0.0)
        np.testing.assert_allclose(qd, np.zeros(robot.nv), atol=0.0)

    def test_velocity_limit_bounds_the_integrated_position_step(self):
        velocity_limit = np.ones(6, dtype=float) * 4.0
        robot, simulator = self.make_simulator(
            K=np.ones(6, dtype=float) * 500.0,
            D=np.zeros(6, dtype=float),
            limit_velocity=velocity_limit,
        )
        q_before = simulator.q.copy()
        dt = 0.001

        q, qd = simulator.step(
            dt=dt,
            q_ref=np.array([2.0, 2.0, 2.0, -2.0, -1.0, 2.0], dtype=float),
        )

        step_velocity = (q - q_before) / dt
        self.assertTrue(np.any(np.isclose(np.abs(qd), velocity_limit, atol=1.0e-12)))
        self.assertTrue(np.all(np.abs(step_velocity) <= velocity_limit + 1.0e-12))
        np.testing.assert_allclose(qd, step_velocity, atol=1.0e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
