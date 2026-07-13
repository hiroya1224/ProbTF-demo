import unittest
from pathlib import Path

import numpy as np

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import PeriodicSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIMPLE6R_URDF = (
    REPOSITORY_ROOT
    / "ros"
    / "examples"
    / "deflecomp"
    / "deflecomp_description"
    / "urdf"
    / "simple6r.urdf"
)
FIXED_BASE_PENDULUM_URDF = REPOSITORY_ROOT / "tests" / "fixtures" / "fixed_base_pendulum.urdf"
YAMAGUCHI_URDF = (
    REPOSITORY_ROOT.parent
    / "nejineji-urdfs"
    / "yamaguchi_arm_nejineji"
    / "urdf"
    / "yamaguchi_6axis_arm_nejineji.urdf"
)


class _SingularCouplingRobot:
    nv = 2

    def d_tau_gravity(self, theta):
        del theta
        return np.array([[0.0, 10.0], [0.0, 0.0]], dtype=float)


class _SingularCouplingSpring:
    def stiffness_diag(self, theta, theta_cmd, kp_vec):
        del theta, theta_cmd, kp_vec
        return np.array([0.0, 1.0], dtype=float)

    def log_stiffness_jacobian_diag(self, theta, theta_cmd, kp_vec):
        del theta, theta_cmd, kp_vec
        return np.array([1.0, 0.0], dtype=float)


class _ForcedActiveSensitivity(SensitivityCalculator):
    def equilibrium_active_set(self, theta_eq, theta_cmd, kp_vec):
        del theta_eq, theta_cmd, kp_vec
        return np.array([False, True], dtype=bool)


class InverseStaticsConsistencyTests(unittest.TestCase):
    def test_pinocchio_potential_gradient_with_massive_fixed_base(self):
        """The fixed-base inertia must not scale the movable-link COM energy."""
        robot = RobotArm(
            str(FIXED_BASE_PENDULUM_URDF),
            tip_link="pendulum_link",
            base_link="base_link",
        )
        theta = np.array([0.37], dtype=float)
        epsilon = 1.0e-7
        gradient_fd = (
            robot.potential_gravity(theta + epsilon)
            - robot.potential_gravity(theta - epsilon)
        ) / (2.0 * epsilon)

        self.assertGreater(abs(float(robot.tau_gravity(theta)[0])), 1.0)
        self.assertAlmostEqual(
            float(gradient_fd),
            float(robot.tau_gravity(theta)[0]),
            delta=2.0e-8,
        )

    def test_singular_free_block_cannot_move_an_active_joint(self):
        sensitivity = _ForcedActiveSensitivity(
            robot=_SingularCouplingRobot(),
            spring_model=_SingularCouplingSpring(),
        )
        zeros = np.zeros(2, dtype=float)
        ones = np.ones(2, dtype=float)

        j_q, j_x = sensitivity.equilibrium_jacobians(zeros, zeros, ones)
        theta_x = -np.linalg.pinv(j_q, rcond=1.0e-12) @ j_x

        np.testing.assert_allclose(j_q[:, 1], np.array([0.0, 1.0]), atol=0.0)
        np.testing.assert_allclose(theta_x[1, :], 0.0, atol=0.0)

    def test_gravity_must_be_evaluated_at_the_desired_pose(self):
        robot = RobotArm(str(SIMPLE6R_URDF))
        spring = PeriodicSpringModel()
        solver = EquilibriumSolver(robot, spring)
        theta_ref = np.array([0.20, -0.40, 0.35, -0.25, 0.30, -0.15], dtype=float)
        kp_vec = np.full(robot.nv, 50.0, dtype=float)

        tau_ref = robot.tau_gravity(theta_ref)
        theta_cmd = spring.theta_cmd_from_theta_ref(tau_ref, theta_ref, kp_vec)
        residual = robot.tau_gravity(theta_ref) + spring.torque(
            theta_ref,
            theta_cmd,
            kp_vec,
        )
        theta_eq = solver.solve(theta_cmd, kp_vec, theta_init=theta_ref)

        self.assertLess(np.linalg.norm(residual), 1.0e-12)
        self.assertLess(np.linalg.norm(theta_eq - theta_ref), 1.0e-6)

        # A path-dependent projected pose is not interchangeable with the
        # known target pose: anchoring the command at theta_ref while using a
        # different pose's gravity breaks the inverse-statics equation.
        theta_gravity_wrong = theta_ref + 0.1 * np.array(
            [0.0, 0.35, -0.25, 0.20, -0.15, 0.10],
            dtype=float,
        )
        theta_cmd_wrong = spring.theta_cmd_from_theta_ref(
            robot.tau_gravity(theta_gravity_wrong),
            theta_ref,
            kp_vec,
        )
        wrong_residual = robot.tau_gravity(theta_ref) + spring.torque(
            theta_ref,
            theta_cmd_wrong,
            kp_vec,
        )
        theta_eq_wrong = solver.solve(theta_cmd_wrong, kp_vec, theta_init=theta_ref)

        self.assertGreater(np.linalg.norm(wrong_residual), 1.0e-3)
        self.assertGreater(np.linalg.norm(theta_eq_wrong - theta_ref), 1.0e-3)


@unittest.skipUnless(YAMAGUCHI_URDF.exists(), f"Yamaguchi URDF not found: {YAMAGUCHI_URDF}")
class YamaguchiEquilibriumConstraintTests(unittest.TestCase):
    def setUp(self):
        self.robot = RobotArm(str(YAMAGUCHI_URDF))
        self.spring = PeriodicSpringModel()
        self.solver = EquilibriumSolver(
            self.robot,
            self.spring,
            EquilibriumConfig(
                maxiter=300,
                n_lambda=10,
                ftol=1.0e-14,
                refine=True,
                refine_maxiter=300,
                refine_tol=1.0e-12,
            ),
        )

    def test_pinocchio_potential_gradient_matches_generalized_gravity(self):
        theta = np.array([0.20, -0.30, 0.25, -0.15, -0.20, 0.12], dtype=float)
        epsilon = 1.0e-7
        gradient_fd = np.zeros(self.robot.nv, dtype=float)
        for index in range(self.robot.nv):
            delta = np.zeros(self.robot.nv, dtype=float)
            delta[index] = epsilon
            gradient_fd[index] = (
                self.robot.potential_gravity(theta + delta)
                - self.robot.potential_gravity(theta - delta)
            ) / (2.0 * epsilon)

        np.testing.assert_allclose(
            gradient_fd,
            self.robot.tau_gravity(theta),
            rtol=1.0e-6,
            atol=2.0e-9,
        )

    def test_bound_equilibrium_satisfies_kkt_and_active_set_sensitivity(self):
        # module4_joint1 (index 4) has an upper limit of zero.  A positive
        # command holds it firmly against that limit, so changing K[4] changes
        # only the reaction torque, not the equilibrium pose.
        theta_cmd = np.array([0.10, -0.15, 0.12, 0.05, 0.10, 0.08], dtype=float)
        kp_vec = np.array([5.0, 5.0, 5.0, 10.0, 20.0, 20.0], dtype=float)
        theta_eq = self.solver.solve(theta_cmd=theta_cmd, kp_vec=kp_vec)
        gradient = self.robot.tau_gravity(theta_eq) + self.spring.torque(
            theta_eq,
            theta_cmd,
            kp_vec,
        )

        active_index = 4
        self.assertAlmostEqual(theta_eq[active_index], 0.0, places=10)
        self.assertLess(gradient[active_index], -1.0)
        free = np.ones(self.robot.nv, dtype=bool)
        free[active_index] = False
        self.assertLess(np.linalg.norm(gradient[free], ord=np.inf), 1.0e-7)

        sensitivity = SensitivityCalculator(self.robot, self.spring)
        j_q, j_x = sensitivity.equilibrium_jacobians(theta_eq, theta_cmd, kp_vec)
        theta_x = -np.linalg.solve(j_q, j_x)
        self.assertTrue(sensitivity.last_active_set[active_index])
        np.testing.assert_allclose(theta_x[:, active_index], 0.0, atol=1.0e-12)

        epsilon = 1.0e-2
        kp_plus = kp_vec.copy()
        kp_minus = kp_vec.copy()
        kp_plus[active_index] *= np.exp(epsilon)
        kp_minus[active_index] *= np.exp(-epsilon)
        theta_plus = self.solver.solve(
            theta_cmd,
            kp_plus,
            theta_init=theta_eq,
            lambdas=np.array([0.0]),
        )
        theta_minus = self.solver.solve(
            theta_cmd,
            kp_minus,
            theta_init=theta_eq,
            lambdas=np.array([0.0]),
        )
        theta_x_fd = (theta_plus - theta_minus) / (2.0 * epsilon)
        np.testing.assert_allclose(theta_x_fd, theta_x[:, active_index], atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
