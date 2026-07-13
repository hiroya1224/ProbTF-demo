import unittest
from pathlib import Path

import numpy as np

from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import PeriodicSpringModel
from deflecomp_core.observation.bingham import BinghamUtils
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.robot.pinocchio_robot import RobotArm


class FakeRobot:
    def __init__(self, frame_quaternions, frame_jacobians):
        self.frame_quaternions = frame_quaternions
        self.frame_jacobians = frame_jacobians

    def frame_quaternion_wxyz_base(self, theta, fid):
        del theta
        return self.frame_quaternions[fid].copy()

    def frame_angular_jacobian_base(self, theta, fid):
        del theta
        return self.frame_jacobians[fid].copy()


class FakeSensitivity:
    def __init__(self, robot, J_q, J_x):
        self.robot = robot
        self.J_q = J_q
        self.J_x = J_x

    def equilibrium_jacobians(self, theta_eq, theta_cmd, kp_vec):
        del theta_eq, theta_cmd, kp_vec
        return self.J_q.copy(), self.J_x.copy()


class FakeSolver:
    def __init__(self, theta_eq):
        self.theta_eq = theta_eq
        self.solve_calls = 0

    def solve(self, theta_cmd, kp_vec, theta_init=None):
        del theta_cmd, kp_vec, theta_init
        self.solve_calls += 1
        return self.theta_eq.copy()


class LogStiffnessSolver:
    """Small exact model used to exercise nonlinear update acceptance."""

    def __init__(self, size, gain=1.0):
        self.size = int(size)
        self.gain = float(gain)
        self.solve_calls = 0

    def solve(self, theta_cmd, kp_vec, theta_init=None):
        del theta_cmd, theta_init
        self.solve_calls += 1
        return self.gain * np.log(np.asarray(kp_vec, dtype=float))


class RotationVectorRobot:
    def frame_quaternion_wxyz_base(self, theta, fid):
        del fid
        half_angle = 0.5 * float(theta[0])
        return np.array([np.cos(half_angle), np.sin(half_angle), 0.0, 0.0], dtype=float)

    def frame_angular_jacobian_base(self, theta, fid):
        del theta, fid
        jacobian = np.zeros((3, 3), dtype=float)
        jacobian[0, 0] = 1.0
        return jacobian


def make_estimator(
    z=None,
    J_w=None,
    J_q=None,
    J_x=None,
    x0=None,
    P0=None,
    Q=None,
    **kwargs,
):
    n = 3 if x0 is None else x0.size
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if z is None else np.asarray(z, dtype=float)
    J_w = np.eye(n) if J_w is None else np.asarray(J_w, dtype=float)
    J_q = np.eye(n) if J_q is None else np.asarray(J_q, dtype=float)
    J_x = np.eye(n) if J_x is None else np.asarray(J_x, dtype=float)
    x0 = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.eye(n, dtype=float) if P0 is None else np.asarray(P0, dtype=float)
    Q = np.zeros((n, n), dtype=float) if Q is None else np.asarray(Q, dtype=float)
    theta_eq = np.zeros(J_q.shape[0], dtype=float)
    robot = FakeRobot({0: z}, {0: J_w})
    solver = FakeSolver(theta_eq)
    sensitivity = FakeSensitivity(robot, J_q, J_x)
    estimator = MultiFrameStiffnessWEKF(
        x0=x0,
        P0=P0,
        Q=Q,
        solver=solver,
        sensitivity=sensitivity,
        **kwargs,
    )
    return estimator, solver


class StiffnessWEKFLaplaceTests(unittest.TestCase):
    def test_clone_preserves_active_set_sensitivity_tolerances(self):
        repository_root = Path(__file__).resolve().parents[2]
        urdf = (
            repository_root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_description"
            / "urdf"
            / "simple6r.urdf"
        )
        robot = RobotArm(str(urdf), base_link="link1")
        spring_model = PeriodicSpringModel()
        solver = EquilibriumSolver(robot, spring_model, EquilibriumConfig())
        sensitivity = SensitivityCalculator(
            robot,
            spring_model,
            active_set_tol=3.0e-7,
            kkt_tol=7.0e-6,
        )
        estimator = MultiFrameStiffnessWEKF(
            x0=np.zeros(robot.nv),
            P0=np.eye(robot.nv),
            Q=np.zeros((robot.nv, robot.nv)),
            solver=solver,
            sensitivity=sensitivity,
        )

        clone = estimator.clone_for_evaluation()

        self.assertEqual(clone.sensitivity.active_set_tol, sensitivity.active_set_tol)
        self.assertEqual(clone.sensitivity.kkt_tol, sensitivity.kkt_tol)
        self.assertEqual(clone.max_log_kp_update_step, estimator.max_log_kp_update_step)
        self.assertEqual(clone.max_equilibrium_pose_jump, estimator.max_equilibrium_pose_jump)

    def test_relative_quaternion_jacobian_matches_robot_finite_difference(self):
        repository_root = Path(__file__).resolve().parents[2]
        urdf = (
            repository_root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_description"
            / "urdf"
            / "simple6r.urdf"
        )
        # A moving, rotated base frame exercises both subtraction of the base
        # angular velocity and the WORLD-to-base coordinate conversion.
        robot = RobotArm(str(urdf), base_link="link1")
        fid = robot.get_frame_id(robot.tip_link_name)
        theta = np.array([0.31, -0.27, 0.42, -0.19, 0.36, -0.24], dtype=float)
        z_nominal = robot.frame_quaternion_wxyz_base(theta, fid)
        jacobian_fd = np.zeros((4, robot.nv), dtype=float)
        epsilon = 1.0e-7
        for joint_index in range(robot.nv):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[joint_index] += epsilon
            theta_minus[joint_index] -= epsilon
            z_plus = robot.frame_quaternion_wxyz_base(theta_plus, fid)
            z_minus = robot.frame_quaternion_wxyz_base(theta_minus, fid)
            if float(z_plus @ z_nominal) < 0.0:
                z_plus = -z_plus
            if float(z_minus @ z_nominal) < 0.0:
                z_minus = -z_minus
            jacobian_fd[:, joint_index] = (z_plus - z_minus) / (2.0 * epsilon)

        jacobian_analytic = 0.5 * (
            BinghamUtils.spatial_qmat_from_quat_wxyz(z_nominal)
            @ robot.frame_angular_jacobian_base(theta, fid)
        )
        relative_error = np.linalg.norm(jacobian_analytic - jacobian_fd) / max(
            np.linalg.norm(jacobian_fd),
            1.0e-12,
        )

        self.assertLess(relative_error, 1.0e-6)

    def test_real_robot_likelihood_gradient_and_information_match_finite_difference(self):
        repository_root = Path(__file__).resolve().parents[2]
        urdf = (
            repository_root
            / "ros"
            / "examples"
            / "deflecomp"
            / "deflecomp_description"
            / "urdf"
            / "simple6r.urdf"
        )
        robot = RobotArm(str(urdf), base_link="link1")
        fid = robot.get_frame_id(robot.tip_link_name)
        theta = np.array([0.31, -0.27, 0.42, -0.19, 0.36, -0.24], dtype=float)
        theta_x = np.array(
            [
                [0.12, -0.03, 0.00, 0.01, 0.00, 0.00],
                [0.00, 0.16, -0.02, 0.00, 0.01, 0.00],
                [0.01, 0.00, 0.14, -0.02, 0.00, 0.01],
                [0.00, 0.02, 0.00, 0.11, -0.01, 0.00],
                [0.00, 0.00, 0.01, 0.00, 0.13, -0.02],
                [-0.01, 0.00, 0.00, 0.01, 0.00, 0.10],
            ],
            dtype=float,
        )
        factor = np.array(
            [
                [0.8, -0.1, 0.2, 0.0],
                [0.1, 1.1, -0.2, 0.1],
                [-0.2, 0.0, 0.9, 0.3],
                [0.0, -0.1, 0.2, 1.2],
            ],
            dtype=float,
        )
        A = -(factor.T @ factor)
        estimator, _ = make_estimator(
            x0=np.zeros(6, dtype=float),
            J_w=np.eye(6, dtype=float),
            J_q=np.eye(6, dtype=float),
            J_x=np.eye(6, dtype=float),
        )
        estimator.robot = robot

        term = estimator._compute_frame_laplace_term(theta, fid, A, theta_x)
        z_nominal = robot.frame_quaternion_wxyz_base(theta, fid)
        epsilon = 1.0e-6
        gradient_fd = np.zeros(6, dtype=float)
        z_x_fd = np.zeros((4, 6), dtype=float)
        for state_index in range(6):
            delta = np.zeros(6, dtype=float)
            delta[state_index] = epsilon
            z_plus = robot.frame_quaternion_wxyz_base(theta - theta_x @ delta, fid)
            z_minus = robot.frame_quaternion_wxyz_base(theta + theta_x @ delta, fid)
            if float(z_plus @ z_nominal) < 0.0:
                z_plus = -z_plus
            if float(z_minus @ z_nominal) < 0.0:
                z_minus = -z_minus
            ell_plus = float(z_plus.T @ A @ z_plus)
            ell_minus = float(z_minus.T @ A @ z_minus)
            gradient_fd[state_index] = (ell_plus - ell_minus) / (2.0 * epsilon)
            z_x_fd[:, state_index] = (z_plus - z_minus) / (2.0 * epsilon)
        information_fd = -2.0 * (z_x_fd.T @ A @ z_x_fd)

        np.testing.assert_allclose(term["gradient"], gradient_fd, rtol=2.0e-5, atol=1.0e-8)
        np.testing.assert_allclose(
            term["information"],
            information_fd,
            rtol=2.0e-5,
            atol=1.0e-8,
        )

    def test_local_laplace_terms_match_linear_model_finite_difference(self):
        z = np.array([0.91, 0.22, -0.28, 0.21], dtype=float)
        z = z / np.linalg.norm(z)
        J_w = np.array(
            [
                [0.7, -0.2, 0.1],
                [0.1, 0.5, -0.3],
                [-0.4, 0.2, 0.6],
            ],
            dtype=float,
        )
        J_q = np.array(
            [
                [1.8, 0.2, -0.1],
                [0.1, 1.5, 0.3],
                [-0.2, 0.1, 1.7],
            ],
            dtype=float,
        )
        J_x = np.diag([0.4, -0.7, 0.5])
        A = -np.array(
            [
                [0.4, -0.1, 0.2, 0.0],
                [-0.1, 1.2, 0.1, -0.2],
                [0.2, 0.1, 0.8, 0.3],
                [0.0, -0.2, 0.3, 1.5],
            ],
            dtype=float,
        )
        A = 0.5 * (A + A.T)
        estimator, _ = make_estimator(z=z, J_w=J_w, J_q=J_q, J_x=J_x)

        terms = estimator._compute_local_laplace_terms(
            theta_eq=np.zeros(3),
            J_q=J_q,
            J_x=J_x,
            A_map={0: A},
        )
        grad = terms["gradient"]
        info = terms["information"]

        theta_x = np.linalg.pinv(J_q, rcond=1e-12) @ J_x
        M = BinghamUtils.spatial_qmat_from_quat_wxyz(z) @ (J_w @ theta_x)

        def ell(delta_x):
            z_local = z - 0.5 * (M @ delta_x)
            return float(z_local.T @ (A @ z_local))

        direction = np.array([0.3, -0.4, 0.5], dtype=float)
        eps_grad = 1e-6
        eps_hess = 1e-3
        first_diff = (ell(eps_grad * direction) - ell(-eps_grad * direction)) / (2.0 * eps_grad)
        second_diff = (
            ell(eps_hess * direction) - 2.0 * ell(np.zeros(3)) + ell(-eps_hess * direction)
        ) / (eps_hess * eps_hess)

        self.assertTrue(np.allclose(info, info.T, atol=1e-12))
        self.assertTrue(np.isclose(first_diff, float(grad.T @ direction), rtol=1e-6, atol=1e-8))
        self.assertTrue(np.isclose(second_diff, -float(direction.T @ info @ direction), rtol=2e-4, atol=1e-8))

    def test_observable_subspace_clips_small_negative_information_eigenvalues(self):
        estimator, _ = make_estimator(observability_abs=1e-12)
        info = np.diag([-1e-12, 0.0, 2.0])

        eigvals, _, keep, raw_eigvals, min_raw_eig = estimator._observable_subspace(info)

        self.assertLess(min_raw_eig, 0.0)
        self.assertTrue(np.all(eigvals >= 0.0))
        self.assertLess(raw_eigvals[0], 0.0)
        self.assertEqual(np.count_nonzero(keep), 1)

    def test_prior_whitening_exposes_weak_mode_after_dominant_mode_contracts(self):
        estimator, _ = make_estimator(observability_rcond=1.0e-4, observability_abs=1.0e-12)
        information = np.diag([1.0e-5, 1.0, 0.0])

        initial = estimator._prior_whitened_observable_subspace(
            information,
            np.eye(3),
        )
        after_dominant_contraction = estimator._prior_whitened_observable_subspace(
            information,
            np.diag([1.0, 1.0e-4, 1.0]),
        )

        self.assertEqual(np.count_nonzero(initial[2]), 1)
        self.assertEqual(np.count_nonzero(after_dominant_contraction[2]), 2)

    def test_zero_information_skips_mean_update_and_keeps_process_noise(self):
        Q = np.eye(3) * 0.25
        estimator, solver = make_estimator(Q=Q)
        x_before = estimator.x.copy()
        P_before = estimator.P.copy()
        theta_cmd = np.zeros(3)

        estimator.update_with_multi(theta_cmd, {0: np.zeros((4, 4))}, theta_init_eq_pred=None)

        self.assertEqual(solver.solve_calls, 1)
        self.assertTrue(np.allclose(estimator.x, x_before))
        self.assertTrue(np.allclose(estimator.P, P_before + Q))
        self.assertEqual(estimator.last_debug["laplace_update_skipped_reason"], "no_observable_information")

    def test_nonzero_gradient_with_zero_information_is_safely_skipped(self):
        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        estimator, _ = make_estimator()
        theta_cmd = np.zeros(3)

        estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None)

        self.assertGreater(estimator.last_debug["laplace_grad_norm"], 0.0)
        self.assertEqual(estimator.last_debug["laplace_update_skipped_reason"], "no_observable_information")
        self.assertTrue(np.allclose(estimator.x, np.zeros(3)))

    def test_zero_gradient_nonzero_information_only_shrinks_covariance(self):
        A = np.diag([0.0, -2.0, -3.0, -4.0])
        estimator, _ = make_estimator(P0=np.eye(3) * 2.0, laplace_jitter=0.0)
        theta_cmd = np.zeros(3)
        P_before = estimator.P.copy()

        estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None)

        self.assertTrue(np.allclose(estimator.last_update_step, np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(estimator.x, np.zeros(3), atol=1e-12))
        self.assertTrue(np.all(np.diag(estimator.P) < np.diag(P_before)))
        self.assertNotIn("laplace_update_skipped_reason", estimator.last_debug)

    def test_estimator_keeps_kp_bounds(self):
        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = 2.0
        A[1, 0] = 2.0
        A[1, 1] = -0.1
        robot = RotationVectorRobot()
        solver = LogStiffnessSolver(3)
        # theta_eq = x, hence -pinv(J_q) J_x = I requires J_x = -I.
        sensitivity = FakeSensitivity(robot, np.eye(3), -np.eye(3))
        estimator = MultiFrameStiffnessWEKF(
            x0=np.zeros(3),
            P0=np.eye(3) * 10.0,
            Q=np.zeros((3, 3)),
            solver=solver,
            sensitivity=sensitivity,
            laplace_jitter=0.0,
        )
        theta_cmd = np.zeros(3)

        result = estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None, kp_lim=(0.1, 500.0))

        self.assertTrue(result.update_applied)
        self.assertTrue(np.all(estimator.kp_hat <= 500.0 + 1e-12))
        self.assertTrue(np.all(estimator.kp_hat >= 0.1 - 1e-12))
        np.testing.assert_allclose(result.theta_eq, estimator.x, atol=1.0e-12)
        self.assertLess(result.debug["laplace_step_scale"], 1.0)
        self.assertFalse(result.debug["laplace_backtracking_trials"][0]["accepted"])
        self.assertGreaterEqual(
            result.debug["laplace_log_likelihood_after"] + 1.0e-12,
            result.debug["laplace_log_likelihood_before"],
        )

    def test_yamaguchi_exact_backtracking_and_weak_k5_mode_regression(self):
        repository_root = Path(__file__).resolve().parents[2]
        urdf = (
            repository_root.parent
            / "nejineji-urdfs"
            / "yamaguchi_arm_nejineji"
            / "urdf"
            / "yamaguchi_6axis_arm_nejineji.urdf"
        )
        if not urdf.exists():
            self.skipTest(f"Yamaguchi regression URDF is not installed: {urdf}")

        robot = RobotArm(str(urdf), tip_link="module4_link2", base_link="base_link")
        spring_model = PeriodicSpringModel()
        solver = EquilibriumSolver(robot, spring_model, EquilibriumConfig())
        sensitivity = SensitivityCalculator(robot, spring_model)
        theta_cmd = np.array(
            [-0.16713734, 1.14925658, -0.15910085, 0.66640165, -0.40405263, -1.19423781],
            dtype=float,
        )
        kp_true = np.array([5.0, 5.0, 5.0, 10.0, 20.0, 20.0], dtype=float)
        theta_true = solver.solve(theta_cmd, kp_true, theta_cmd)
        frame_names = [
            "module1_link1",
            "module2_link1",
            "module3_link1",
            "module4_link2",
            "module5_d405_link",
        ]
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=float)
        observations = [
            FrameImuObservation(
                frame_name=frame_name,
                gravity_dir=robot.gravity_dir_in_frame(
                    theta_true,
                    gravity_world,
                    robot.get_frame_id(frame_name),
                ),
            )
            for frame_name in frame_names
        ]
        A_map = ImuObservationBuilder(robot, parameter_A=1000.0).build_A_map(observations)

        kp_min, kp_max = 1.0, 500.0
        initial_log_kp = np.full(6, np.log(np.sqrt(kp_min * kp_max)), dtype=float)
        initial_std = np.log(kp_max / kp_min) / 4.0
        estimator = MultiFrameStiffnessWEKF(
            x0=initial_log_kp,
            P0=np.eye(6) * initial_std**2,
            Q=np.zeros((6, 6)),
            solver=solver,
            sensitivity=sensitivity,
            observability_rcond=1.0e-4,
            observability_abs=1.0e-10,
        )
        theta_initial = solver.solve(theta_cmd, estimator.kp_est, theta_cmd)
        initial_pose_error = float(np.linalg.norm(theta_initial - theta_true))

        ranks = []
        k5_after_first = None
        for update_index in range(16):
            result = estimator.update_with_multi(
                theta_cmd_sent=theta_cmd,
                A_map=A_map,
                theta_init_eq_pred=theta_initial if update_index == 0 else estimator.last_theta_eq,
                kp_lim=(kp_min, kp_max),
            )
            ranks.append(result.obs_rank)
            self.assertGreaterEqual(
                result.debug["laplace_log_likelihood_after"] + 1.0e-9,
                result.debug["laplace_log_likelihood_before"],
            )
            if update_index == 0:
                k5_after_first = float(result.kp_est[4])
                self.assertLess(result.debug["laplace_step_scale"], 1.0)
                self.assertFalse(result.debug["laplace_backtracking_trials"][0]["accepted"])

        self.assertIsNotNone(k5_after_first)
        # The raw per-sample relative cutoff leaves this mode below threshold,
        # whereas prior whitening exposes it after the dominant modes contract.
        self.assertGreater(ranks[-1], ranks[0])
        self.assertLess(abs(float(estimator.kp_est[4]) - kp_true[4]), abs(k5_after_first - kp_true[4]))
        final_pose_error = float(np.linalg.norm(estimator.last_theta_eq - theta_true))
        self.assertGreater(initial_pose_error, 0.10)
        self.assertLess(final_pose_error, initial_pose_error)
        self.assertLess(final_pose_error, 0.05)

    def test_yamaguchi_large_laplace_step_is_confined_to_local_pose_branch(self):
        repository_root = Path(__file__).resolve().parents[2]
        urdf = (
            repository_root.parent
            / "nejineji-urdfs"
            / "yamaguchi_arm_nejineji"
            / "urdf"
            / "yamaguchi_6axis_arm_nejineji.urdf"
        )
        if not urdf.exists():
            self.skipTest(f"Yamaguchi regression URDF is not installed: {urdf}")

        robot = RobotArm(str(urdf), tip_link="module4_link2", base_link="base_link")
        spring_model = PeriodicSpringModel()
        solver = EquilibriumSolver(robot, spring_model, EquilibriumConfig())
        sensitivity = SensitivityCalculator(robot, spring_model)
        theta_cmd = np.array(
            [-1.26429437, -1.71550493, -0.24465812, 1.85605893, -0.27998728, 0.93715279],
            dtype=float,
        )
        kp_true = np.array([5.0, 5.0, 5.0, 10.0, 20.0, 20.0], dtype=float)
        theta_true = solver.solve(theta_cmd, kp_true, theta_cmd)
        frame_names = [
            "module1_link1",
            "module2_link1",
            "module3_link1",
            "module4_link2",
            "module5_d405_link",
        ]
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=float)
        observations = [
            FrameImuObservation(
                frame_name=frame_name,
                gravity_dir=robot.gravity_dir_in_frame(
                    theta_true,
                    gravity_world,
                    robot.get_frame_id(frame_name),
                ),
            )
            for frame_name in frame_names
        ]
        A_map = ImuObservationBuilder(robot, parameter_A=1000.0).build_A_map(observations)

        kp_min, kp_max = 1.0, 500.0
        initial_log_kp = np.full(6, np.log(np.sqrt(kp_min * kp_max)), dtype=float)
        initial_std = np.log(kp_max / kp_min) / 4.0
        estimator = MultiFrameStiffnessWEKF(
            x0=initial_log_kp,
            P0=np.eye(6) * initial_std**2,
            Q=np.zeros((6, 6)),
            solver=solver,
            sensitivity=sensitivity,
            observability_rcond=1.0e-4,
            observability_abs=1.0e-10,
        )
        theta_initial = solver.solve(theta_cmd, estimator.kp_est, theta_cmd)
        initial_pose_error = float(np.linalg.norm(theta_initial - theta_true))
        self.assertGreater(initial_pose_error, 0.2)

        def gravity_angle_rms(theta):
            angles = []
            for frame_name in frame_names:
                frame_id = robot.get_frame_id(frame_name)
                gravity_predicted = robot.gravity_dir_in_frame(
                    theta,
                    gravity_world,
                    frame_id,
                )
                gravity_observed = robot.gravity_dir_in_frame(
                    theta_true,
                    gravity_world,
                    frame_id,
                )
                cosine = float(np.clip(gravity_predicted @ gravity_observed, -1.0, 1.0))
                angles.append(np.arccos(cosine))
            return float(np.sqrt(np.mean(np.square(angles))))

        initial_gravity_angle_rms = gravity_angle_rms(theta_initial)

        # Isolate the branch guard from the log-stiffness trust region.  The
        # formerly accepted alpha=0.125 candidate jumps module1_joint1 to its
        # lower stop; with the mean-step guard disabled, the pose guard must
        # still reject that branch and find a local candidate.
        branch_guard_estimator = MultiFrameStiffnessWEKF(
            x0=initial_log_kp,
            P0=np.eye(6) * initial_std**2,
            Q=np.zeros((6, 6)),
            solver=solver,
            sensitivity=sensitivity,
            observability_rcond=1.0e-4,
            observability_abs=1.0e-10,
            max_log_kp_update_step=np.inf,
        )
        branch_result = branch_guard_estimator.update_with_multi(
            theta_cmd_sent=theta_cmd,
            A_map=A_map,
            theta_init_eq_pred=theta_initial,
            kp_lim=(kp_min, kp_max),
        )
        branch_trial_reasons = [
            trial["reason"] for trial in branch_result.debug["laplace_backtracking_trials"]
        ]
        self.assertIn("equilibrium_pose_jump_exceeded", branch_trial_reasons)
        self.assertLessEqual(
            branch_result.debug["laplace_equilibrium_pose_jump_norm"],
            branch_guard_estimator.max_equilibrium_pose_jump + 1.0e-12,
        )
        self.assertLess(float(np.linalg.norm(branch_result.theta_eq - theta_true)), initial_pose_error)

        first_pose_error = None
        first_trial_reasons = []
        pose_error_after_eight = None
        gravity_angle_rms_after_eight = None
        for update_index in range(20):
            x_before = estimator.x.copy()
            result = estimator.update_with_multi(
                theta_cmd_sent=theta_cmd,
                A_map=A_map,
                theta_init_eq_pred=theta_initial if update_index == 0 else estimator.last_theta_eq,
                kp_lim=(kp_min, kp_max),
            )
            self.assertTrue(result.update_applied)
            self.assertLessEqual(
                float(np.max(np.abs(result.x_est - x_before))),
                estimator.max_log_kp_update_step + 1.0e-12,
            )
            self.assertLessEqual(
                result.debug["laplace_equilibrium_pose_jump_norm"],
                estimator.max_equilibrium_pose_jump + 1.0e-12,
            )
            self.assertGreaterEqual(
                result.debug["laplace_log_likelihood_after"] + 1.0e-9,
                result.debug["laplace_log_likelihood_before"],
            )
            if update_index == 0:
                first_pose_error = float(np.linalg.norm(result.theta_eq - theta_true))
                first_trial_reasons = [
                    trial["reason"] for trial in result.debug["laplace_backtracking_trials"]
                ]
            if update_index == 7:
                pose_error_after_eight = float(np.linalg.norm(result.theta_eq - theta_true))
                gravity_angle_rms_after_eight = gravity_angle_rms(result.theta_eq)

        self.assertIn("log_kp_trust_region_exceeded", first_trial_reasons)
        self.assertIsNotNone(first_pose_error)
        self.assertLess(first_pose_error, initial_pose_error)
        self.assertIsNotNone(pose_error_after_eight)
        self.assertIsNotNone(gravity_angle_rms_after_eight)
        self.assertLess(pose_error_after_eight, initial_pose_error)
        self.assertLess(gravity_angle_rms_after_eight, initial_gravity_angle_rms)
        self.assertLess(float(np.linalg.norm(estimator.last_theta_eq - theta_true)), 0.05)
        self.assertTrue(np.all(estimator.kp_est > kp_min + 1.0e-6))
        self.assertTrue(np.all(estimator.kp_est < kp_max - 1.0e-6))

    def test_particle_correction_moment_matches_zealot_and_pursuit_mixture(self):
        x0 = np.array([0.0, 0.0], dtype=float)
        P0 = np.diag([0.01, 0.04])
        estimator, _ = make_estimator(x0=x0, P0=P0)

        estimator.apply_particle_correction(
            x_new=np.array([1.0, 5.0], dtype=float),
            active_indices=np.array([0], dtype=int),
            reset_std=0.2,
            pursuit_mixture_weight=0.25,
            theta_eq=np.ones(2, dtype=float),
            kp_lim=(0.001, 500.0),
        )

        x_expected = np.array([0.25, 0.0], dtype=float)
        P_pursuit = np.diag([0.04, 0.04])
        dz = x0 - x_expected
        dp = np.array([1.0, 0.0], dtype=float) - x_expected
        P_expected = 0.75 * (P0 + np.outer(dz, dz)) + 0.25 * (P_pursuit + np.outer(dp, dp))

        self.assertTrue(np.allclose(estimator.x, x_expected))
        self.assertTrue(np.allclose(estimator.P, P_expected))
        self.assertIsNone(estimator.last_theta_eq)
        self.assertEqual(estimator.last_debug["particle_correction_pursuit_mixture_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
