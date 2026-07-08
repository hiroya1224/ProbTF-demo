from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deflecomp_core.control.feedforward import CommandGenerator, lowpass_theta_cmd
from deflecomp_core.estimator.delay_rls import CommandDelayRLS
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.robot.pinocchio_robot import RobotArm


@dataclass
class CompensationStepResult:
    theta_cmd: np.ndarray
    theta_cmd_raw: np.ndarray
    theta_eq_hat: np.ndarray
    kp_hat: np.ndarray
    tau_hat: np.ndarray
    debug: Dict[str, Any]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


class DeflectionCompensator:
    def __init__(
        self,
        robot: RobotArm,
        spring_model,
        stiffness_estimator: MultiFrameStiffnessWEKF,
        delay_estimator: Optional[CommandDelayRLS] = None,
        command_generator: Optional[CommandGenerator] = None,
        equilibrium_solver: Optional[EquilibriumSolver] = None,
        observation_builder: Optional[ImuObservationBuilder] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.robot = robot
        self.spring_model = spring_model
        self.stiffness_estimator = stiffness_estimator
        self.delay_estimator = delay_estimator
        self.command_generator = command_generator or CommandGenerator(robot=robot, spring_model=spring_model)
        self.equilibrium_solver = equilibrium_solver or EquilibriumSolver(robot=robot, spring_model=spring_model)
        self.observation_builder = observation_builder or ImuObservationBuilder(robot=robot)
        self.config = config or {}

        self.last_theta_cmd: Optional[np.ndarray] = None
        self.last_theta_eq: Optional[np.ndarray] = None
        self.last_stamp: Optional[float] = None
        self.last_theta_ref: Optional[np.ndarray] = None
        self.theta_ref_for_feedforward: Optional[np.ndarray] = None

    def _observable_joint_basis(
        self,
        theta: np.ndarray,
        imu_observations: Sequence[FrameImuObservation],
    ):
        n = theta.size
        info = np.zeros((n, n), dtype=float)
        g_world = getattr(self.observation_builder, "g_world", np.array([0.0, 0.0, -1.0], dtype=float))
        seen_frames = set()
        for observation in imu_observations:
            frame_name = observation.frame_name
            if frame_name in seen_frames or not self.robot.has_frame(frame_name):
                continue
            seen_frames.add(frame_name)
            fid = self.robot.get_frame_id(frame_name)
            J_g = self.robot.gravity_dir_jacobian_in_frame(theta=theta, g_world=g_world, fid=fid)
            info += J_g.T @ J_g

        eigvals, eigvecs = np.linalg.eigh(0.5 * (info + info.T))
        lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
        abs_threshold = float(self.config.get("feedforward_observability_abs", 1e-10))
        rel_threshold = float(self.config.get("feedforward_observability_rcond", 1e-4)) * max(lam_max, 0.0)
        keep = eigvals > max(abs_threshold, rel_threshold)
        return eigvecs[:, keep], eigvals, int(np.count_nonzero(keep))

    def _theta_ref_for_gravity(
        self,
        theta_ref: np.ndarray,
        imu_observations: Optional[Sequence[FrameImuObservation]],
        debug: Dict[str, Any],
    ) -> np.ndarray:
        if not _as_bool(self.config.get("project_unobservable_feedforward", True)):
            self.last_theta_ref = theta_ref.copy()
            self.theta_ref_for_feedforward = theta_ref.copy()
            debug["feedforward_observable_rank"] = theta_ref.size
            return theta_ref

        if self.last_theta_ref is None or self.theta_ref_for_feedforward is None:
            self.last_theta_ref = theta_ref.copy()
            self.theta_ref_for_feedforward = theta_ref.copy()
            debug["feedforward_observable_rank"] = theta_ref.size
            return theta_ref

        if not imu_observations:
            self.last_theta_ref = theta_ref.copy()
            debug["feedforward_observable_rank"] = 0
            debug["feedforward_observable_eigvals"] = np.zeros(theta_ref.size, dtype=float)
            return self.theta_ref_for_feedforward.copy()

        U_obs, eigvals, rank = self._observable_joint_basis(theta_ref, imu_observations)
        delta_ref = theta_ref - self.last_theta_ref
        if rank > 0:
            delta_gravity = U_obs @ (U_obs.T @ delta_ref)
        else:
            delta_gravity = np.zeros_like(delta_ref)

        self.theta_ref_for_feedforward = self.theta_ref_for_feedforward + delta_gravity
        self.last_theta_ref = theta_ref.copy()
        debug["feedforward_observable_rank"] = rank
        debug["feedforward_observable_eigvals"] = eigvals
        return self.theta_ref_for_feedforward.copy()

    def step(
        self,
        theta_ref: np.ndarray,
        imu_observations: Optional[Sequence[FrameImuObservation]],
        dt: float,
        stamp: Optional[float] = None,
    ) -> CompensationStepResult:
        theta_ref = np.asarray(theta_ref, dtype=float)
        debug: Dict[str, Any] = {}

        if self.delay_estimator is not None:
            latest_obs_stamp = None
            if imu_observations:
                obs_stamps = [obs.stamp for obs in imu_observations if obs.stamp is not None]
                latest_obs_stamp = max(obs_stamps) if obs_stamps else None
            debug["delay"] = self.delay_estimator.update(self.last_stamp, latest_obs_stamp)

        update_stiffness = _as_bool(self.config.get("update_stiffness", True))
        theta_eq_obs: Optional[np.ndarray] = None
        if update_stiffness and self.last_theta_cmd is not None and imu_observations:
            a_map = self.observation_builder.build_A_map(imu_observations)
            if a_map:
                theta_init = (
                    self.stiffness_estimator.last_theta_eq
                    if self.stiffness_estimator.last_theta_eq is not None
                    else theta_ref
                )
                kp_lim = self.config.get("kp_lim")
                theta_eq_obs = self.stiffness_estimator.update_with_multi(
                    theta_cmd=self.last_theta_cmd,
                    A_map=a_map,
                    theta_init_eq_pred=theta_init,
                    kp_lim=kp_lim,
                )
                debug["observation_count"] = len(a_map)
        debug["update_stiffness"] = update_stiffness

        kp_hat = self.stiffness_estimator.kp_hat
        theta_gravity = self._theta_ref_for_gravity(theta_ref, imu_observations, debug)
        tau_gravity = self.robot.tau_gravity(theta_gravity)
        theta_cmd_raw = self.spring_model.theta_cmd_from_theta_ref(
            tau_gravity=tau_gravity,
            theta_ref=theta_ref,
            kp_vec=kp_hat,
        )

        if self.last_theta_cmd is not None:
            tau = float(self.config.get("theta_cmd_tau", 0.2))
            dt_cmd = float(dt)
            if stamp is not None and self.last_stamp is not None:
                dt_cmd = max(0.0, float(stamp) - float(self.last_stamp))
            theta_cmd = lowpass_theta_cmd(
                theta_raw=theta_cmd_raw,
                theta_prev=self.last_theta_cmd,
                dt=dt_cmd,
                tau=tau,
            )
        else:
            theta_cmd = theta_cmd_raw

        theta_eq_init = (
            theta_eq_obs
            if theta_eq_obs is not None
            else (self.last_theta_eq if self.last_theta_eq is not None else theta_ref)
        )
        theta_eq_hat = self.equilibrium_solver.solve(
            theta_cmd=theta_cmd,
            kp_vec=kp_hat,
            theta_init=theta_eq_init,
        )
        tau_hat = self.robot.tau_gravity(theta_eq_hat)

        self.last_theta_cmd = theta_cmd.copy()
        self.last_theta_eq = theta_eq_hat.copy()
        self.last_stamp = stamp

        debug.update(
            {
                "kp_cov_diag": np.diag(self.stiffness_estimator.P).copy(),
                "stamp": stamp,
            }
        )
        return CompensationStepResult(
            theta_cmd=theta_cmd,
            theta_cmd_raw=theta_cmd_raw,
            theta_eq_hat=theta_eq_hat,
            kp_hat=kp_hat,
            tau_hat=tau_hat,
            debug=debug,
        )
