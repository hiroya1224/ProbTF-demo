#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import Optional, List

import rospkg
from ikpy.chain import Chain as IkChain

from deflecomp_core.control.feedforward import CommandGenerator
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import LinearSpringModel, PeriodicSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_examples.ik.planner import RikGaikPlanner
from deflecomp_examples.utils.viz import Visualizer
from deflecomp_sim.sensor_simulator import SyntheticObservationBuilder


def resolve_default_urdf() -> str:
    return os.path.join(rospkg.RosPack().get_path("deflecomp_description"), "urdf", "simple6r.urdf")


def make_theta_ref_via_rik(
    urdf_path: str,
    T_target_se3: pin.SE3,
    n_steps: int,
    tip_link: str = "link6",
    base_link: str = "base_link",
) -> np.ndarray:
    robot = RobotArm(urdf_path, tip_link=tip_link, base_link=base_link)
    chain = IkChain.from_urdf_file(urdf_path, base_elements=[base_link], symbolic=False)
    planner = RikGaikPlanner(chain)

    theta_init = np.zeros(robot.nv, dtype=float)
    theta_goal = planner.rik_solve(T_target_se3, theta_init, max_iter=1200)
    theta_ref_path = RikGaikPlanner.make_theta_ref_path(theta_init, theta_goal, n_steps=n_steps)
    return theta_ref_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=resolve_default_urdf())
    parser.add_argument("--spring-model", choices=["linear", "periodic"], default="periodic")
    args = parser.parse_args()

    rng = np.random.default_rng(3)
    g_base = np.array([0.0, 0.0, -9.81], dtype=float)
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    parameter_A = 100.0
    n_ref_steps = 50
    obs_frames: List[str] = ["link6", "link1"]

    urdf_path = os.path.abspath(args.urdf)
    spring_model = PeriodicSpringModel() if args.spring_model == "periodic" else LinearSpringModel()
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")

    # ----- Fixed target: random theta -> equilibrium (with kp_true) -> FK pose -----
    theta_rand = np.array(
        [rng.uniform(lo, hi) for lo, hi in zip(robot_sim.model.lowerPositionLimit, robot_sim.model.upperPositionLimit)],
        dtype=float,
    )
    solver = EquilibriumSolver(robot_sim, spring_model, EquilibriumConfig(maxiter=80))
    theta_eq_for_target = solver.solve(theta_cmd=theta_rand, kp_vec=kp_true, theta_init=theta_rand)
    T_target_se3 = robot_sim.fk_pose(theta_eq_for_target)

    # ----- Build theta_ref path via one-shot IK to target (RIK), then linear interpolation -----
    theta_ref_path = make_theta_ref_via_rik(urdf_path, T_target_se3, n_steps=n_ref_steps)

    # ----- Weird EKF initialization -----
    x0 = np.log(np.ones(robot_est.nv, dtype=float) * 50.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q = np.eye(robot_est.nv) * 1e-3
    estimator_solver = EquilibriumSolver(robot_est, spring_model, EquilibriumConfig(maxiter=80))
    sensitivity = SensitivityCalculator(robot_est, spring_model)
    wekf = MultiFrameStiffnessWEKF(x0, P0, Q, estimator_solver, sensitivity, eps_def=1e-6)

    command_generator = CommandGenerator(robot_est, spring_model)
    obs_builder = SyntheticObservationBuilder(robot_sim, spring_model, g_world=g_base, parameter_A=parameter_A)
    theta_ws_true: Optional[np.ndarray] = None
    theta_eq_pred_prev: Optional[np.ndarray] = None
    theta_cmd_final: Optional[np.ndarray] = None

    # ----- Main loop along theta_ref path (matching the original sample flow) -----
    for k in range(theta_ref_path.shape[0]):
        theta_ref_k = theta_ref_path[k]
        kp_hat = np.exp(wekf.x)

        # GaIK (inverse statics): theta_cmd = theta_ref + K^{-1} tau_g(theta_ref)
        theta_cmd_k = command_generator.theta_cmd_from_theta_ref(theta_ref_k, kp_hat)
        theta_cmd_final = theta_cmd_k

        # Observations from simulated "truth" robot at equilibrium(theta_cmd, kp_true)
        A_map, theta_eq_true_k = obs_builder.build_A_multi(
            theta_cmd=theta_cmd_k,
            kp_true=kp_true,                  # <- truth stiffness for building A_f (reproduces original sample)
            frame_names=obs_frames,
            newton_iter_true=60,
            theta_ws_true=theta_ws_true,
        )
        theta_ws_true = theta_eq_true_k

        # EKF update using all frames; warm start with previous predicted equilibrium (or current ref)
        theta_eq_used = wekf.update_with_multi(
            theta_cmd=theta_cmd_k,
            A_map=A_map,
            theta_init_eq_pred=(theta_eq_pred_prev if theta_eq_pred_prev is not None else theta_ref_k),
        )
        theta_eq_pred_prev = theta_eq_used

        if (k + 1) % max(1, n_ref_steps // 4) == 0:
            print(f"[{k+1}/{n_ref_steps}] Kp_hat =", np.exp(wekf.x))

    # ----- Visualization (same “triplet” view as before) -----
    viz = Visualizer()
    viz.show_triplet(
        robot_sim=robot_sim,
        robot_est=robot_est,
        theta_cmd_final=theta_cmd_final,
        kp_true=kp_true,
        kp_est=np.maximum(np.exp(wekf.x), 1e-8),
        T_target_se3=T_target_se3,
        spring_model=spring_model,
        title="Triplet: rigid vs true-gravity vs est-gravity (multi-frame)",
    )

    # ----- Report pose error wrt target -----
    theta_eq_true_final = solver.solve(theta_cmd=theta_cmd_final, kp_vec=kp_true, theta_init=theta_cmd_final)
    E_final = robot_sim.fk_pose(theta_eq_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))

    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(wekf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)

    # To avoid figures lingering in some environments
    plt.show()


if __name__ == "__main__":
    main()
