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
    kp_est: np.ndarray
    kp_exec: np.ndarray
    kp_exec_target: np.ndarray
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
        self.last_processed_observation_stamp: Optional[float] = None
        self.x_exec = self.stiffness_estimator.x_est.copy()
        self.x_exec_target = self.stiffness_estimator.x_est.copy()

    @property
    def kp_est(self) -> np.ndarray:
        return self.stiffness_estimator.kp_est

    @property
    def kp_exec(self) -> np.ndarray:
        return np.exp(self.x_exec)

    @property
    def kp_exec_target(self) -> np.ndarray:
        return np.exp(self.x_exec_target)

    def _log_kp_limits(self):
        kp_lim = self.config.get("kp_lim")
        if kp_lim is None:
            return None
        return np.log(float(kp_lim[0])), np.log(float(kp_lim[1]))

    def _clip_log_kp(self, x: np.ndarray) -> np.ndarray:
        limits = self._log_kp_limits()
        if limits is None:
            return np.asarray(x, dtype=float).copy()
        return np.clip(np.asarray(x, dtype=float), limits[0], limits[1])

    def _update_exec_stiffness_target(self, x_est: np.ndarray) -> None:
        self.x_exec_target = self._clip_log_kp(x_est)

    def _smooth_exec_stiffness(self, dt: float) -> np.ndarray:
        dt = max(0.0, float(dt))
        tau = float(self.config.get("kp_exec_tau", 1.0))
        if tau <= 0.0:
            delta = self.x_exec_target - self.x_exec
        else:
            alpha = 1.0 - float(np.exp(-dt / max(tau, 1e-9)))
            delta = alpha * (self.x_exec_target - self.x_exec)

        max_step = float(self.config.get("max_log_kp_exec_step", 0.0))
        if max_step > 0.0:
            delta = np.clip(delta, -max_step, max_step)

        self.x_exec = self._clip_log_kp(self.x_exec + delta)
        return delta

    def _observation_stamp(self, imu_observations: Optional[Sequence[FrameImuObservation]]) -> Optional[float]:
        if not imu_observations:
            return None
        stamps = [obs.stamp for obs in imu_observations if obs.stamp is not None]
        return max(stamps) if stamps else None

    def _observation_is_unprocessed(self, observation_stamp: Optional[float]) -> bool:
        if observation_stamp is None:
            return True
        if self.last_processed_observation_stamp is None:
            return True
        return float(observation_stamp) > float(self.last_processed_observation_stamp) + 1e-12

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
        dt_exec = float(dt)
        if stamp is not None and self.last_stamp is not None:
            dt_exec = max(0.0, float(stamp) - float(self.last_stamp))

        if self.delay_estimator is not None:
            latest_obs_stamp = None
            if imu_observations:
                obs_stamps = [obs.stamp for obs in imu_observations if obs.stamp is not None]
                latest_obs_stamp = max(obs_stamps) if obs_stamps else None
            debug["delay"] = self.delay_estimator.update(self.last_stamp, latest_obs_stamp)

        update_stiffness = _as_bool(self.config.get("update_stiffness", True))
        observation_stamp = self._observation_stamp(imu_observations)
        debug["used_theta_cmd_sent_for_update"] = False
        if update_stiffness and self.last_theta_cmd is not None and imu_observations:
            if not self._observation_is_unprocessed(observation_stamp):
                debug["est_update_skipped_reason"] = "duplicate_observation"
            else:
                a_map = self.observation_builder.build_A_map(imu_observations)
                debug["observation_count"] = len(a_map)
                if not a_map:
                    debug["est_update_skipped_reason"] = "empty_observation_map"
                else:
                    theta_cmd_sent = self.last_theta_cmd.copy()
                    debug["used_theta_cmd_sent_for_update"] = True
                    if observation_stamp is not None:
                        self.last_processed_observation_stamp = float(observation_stamp)
                    else:
                        self.last_processed_observation_stamp = None
                    theta_init = (
                        self.stiffness_estimator.last_theta_eq
                        if self.stiffness_estimator.last_theta_eq is not None
                        else theta_ref
                    )
                    kp_lim = self.config.get("kp_lim")
                    update_result = self.stiffness_estimator.update_with_multi(
                        theta_cmd_sent=theta_cmd_sent,
                        A_map=a_map,
                        theta_init_eq_pred=theta_init,
                        kp_lim=kp_lim,
                    )
                    self._update_exec_stiffness_target(update_result.x_est)
                    debug.update(update_result.debug)
        debug["update_stiffness"] = update_stiffness

        log_kp_exec_delta = self._smooth_exec_stiffness(dt_exec)
        kp_est = self.kp_est
        kp_exec = self.kp_exec
        kp_exec_target = self.kp_exec_target
        theta_gravity = self._theta_ref_for_gravity(theta_ref, imu_observations, debug)
        tau_gravity = self.robot.tau_gravity(theta_gravity)
        theta_cmd_raw = self.spring_model.theta_cmd_from_theta_ref(
            tau_gravity=tau_gravity,
            theta_ref=theta_ref,
            kp_vec=kp_exec,
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
            self.last_theta_eq if self.last_theta_eq is not None else theta_ref
        )
        theta_eq_hat = self.equilibrium_solver.solve(
            theta_cmd=theta_cmd,
            kp_vec=kp_exec,
            theta_init=theta_eq_init,
        )
        tau_hat = self.robot.tau_gravity(theta_eq_hat)

        self.last_theta_cmd = theta_cmd.copy()
        self.last_theta_eq = theta_eq_hat.copy()
        self.last_stamp = stamp

        debug.update(
            {
                "kp_est": kp_est.copy(),
                "kp_exec": kp_exec.copy(),
                "kp_exec_target": kp_exec_target.copy(),
                "log_kp_est": self.stiffness_estimator.x_est.copy(),
                "log_kp_exec": self.x_exec.copy(),
                "log_kp_exec_target": self.x_exec_target.copy(),
                "log_kp_exec_delta": log_kp_exec_delta.copy(),
                "kp_cov_diag": np.diag(self.stiffness_estimator.P_est).copy(),
                "stamp": stamp,
            }
        )
        return CompensationStepResult(
            theta_cmd=theta_cmd,
            theta_cmd_raw=theta_cmd_raw,
            theta_eq_hat=theta_eq_hat,
            kp_hat=kp_est,
            kp_est=kp_est,
            kp_exec=kp_exec,
            kp_exec_target=kp_exec_target,
            tau_hat=tau_hat,
            debug=debug,
        )
