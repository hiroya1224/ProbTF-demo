#!/usr/bin/env python3
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String

from deflecomp_core.control.feedforward import CommandGenerator
from deflecomp_core.estimator.initialization import initial_log_kp_state, initial_log_kp_std
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import spring_model_from_name
from deflecomp_core.observation.imu_buffer import (
    ImuBuffer,
    TimedVectorHistory,
    imu_sample_is_quasi_static,
)
from deflecomp_core.observation.imu_frame_config import ImuFrameConfig, resolve_imu_frame_configs
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.pipeline.compensator import DeflectionCompensator
from deflecomp_core.robot.pinocchio_robot import RobotArm


def parse_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def map_jointstate_to_model(msg: JointState, model_names: Sequence[str]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
    q = np.zeros(len(model_names), dtype=float)
    positions = list(msg.position) if msg.position else []
    for idx, name in enumerate(model_names):
        msg_idx = name_to_idx.get(name)
        if msg_idx is not None and msg_idx < len(positions):
            q[idx] = float(positions[msg_idx])
    return q


def jointstate_position_map(msg: JointState) -> Dict[str, float]:
    positions = list(msg.position) if msg.position else []
    return {
        name: float(positions[idx])
        for idx, name in enumerate(msg.name)
        if idx < len(positions)
    }


def expand_joint_positions(
    output_names: Sequence[str],
    active_names: Sequence[str],
    active_positions: np.ndarray,
    fallback_positions: Dict[str, float],
) -> List[float]:
    values = dict(fallback_positions)
    for name, position in zip(active_names, np.asarray(active_positions, dtype=float)):
        values[name] = float(position)
    return [float(values.get(name, 0.0)) for name in output_names]


def resolve_default_urdf() -> str:
    return os.path.join(rospkg.RosPack().get_path("deflecomp_description"), "urdf", "simple6r.urdf")


class DeflecompNode:
    def __init__(
        self,
        urdf_path: str,
        imu_frames: Any,
        topic_ref: str,
        topic_imu: str,
        topic_cmd_out: str,
        dt: float,
        A_param: float,
        kp_lim: Tuple[float, float],
        log_kp_process_noise_var: float,
        spring_model_name: str,
        theta_cmd_tau: float,
        theta_cmd_l1_regularization: bool,
        theta_cmd_l1_regularization_weight: float,
        theta_cmd_equilibrium_refine: bool,
        theta_cmd_equilibrium_refine_maxiter: int,
        theta_cmd_equilibrium_refine_tol: float,
        theta_cmd_equilibrium_refine_max_delta: float,
        equilibrium_refine: bool,
        equilibrium_refine_maxiter: int,
        equilibrium_refine_tol: float,
        update_stiffness: bool,
        observability_rcond: float,
        observability_abs: float,
        project_unobservable_feedforward: bool,
        kp_exec_tau: float,
        max_log_kp_exec_step: float,
        publish_kp_exec: bool,
        particle_scan_enabled: bool,
        particle_scan_window_size: int,
        particle_scan_grid_size: int,
        particle_scan_reset_std: float,
        particle_scan_backend: str,
        particle_pursuit_mixture_weight: float,
        imu_gravity_norm: float,
        imu_acceleration_tolerance: float,
        imu_max_angular_speed: float,
        imu_settle_time: float,
        imu_max_age: float,
        estimation_settle_time: float = 0.5,
        estimation_command_tolerance: float = 1.0e-3,
        estimation_reference_tolerance: float = 1.0e-4,
        command_apply_delay: float = 0.0,
        laplace_outer_iterations: int = 5,
        max_log_kp_update_step: float = 3.0,
        max_equilibrium_pose_jump: float = 0.30,
        joint_limit_reaction_torque_tol: float = 1.0e-3,
        max_log_kp_covariance_var: float = 0.0,
    ) -> None:
        self.robot = RobotArm(urdf_path)
        self.spring_model = spring_model_from_name(spring_model_name, self.robot.model_joint_types)
        self.solver = EquilibriumSolver(
            robot=self.robot,
            spring_model=self.spring_model,
            cfg=EquilibriumConfig(
                maxiter=80,
                refine=bool(equilibrium_refine),
                refine_maxiter=int(equilibrium_refine_maxiter),
                refine_tol=float(equilibrium_refine_tol),
            ),
        )
        self.sensitivity = SensitivityCalculator(robot=self.robot, spring_model=self.spring_model)
        self.n = self.robot.nv
        self.model_joint_names = self.robot.model_joint_names
        self.output_joint_names = self.robot.urdf_info.movable_joint_names or self.model_joint_names

        self.imu_frame_configs: List[ImuFrameConfig] = resolve_imu_frame_configs(
            robot=self.robot,
            value=imu_frames,
            count=max(1, min(3, len(self.model_joint_names))),
        )
        if not self.imu_frame_configs:
            raise ValueError(f"No valid IMU frames found in URDF: {urdf_path}")
        self.imu_config_by_frame_id: Dict[str, ImuFrameConfig] = {
            cfg.frame_id: cfg for cfg in self.imu_frame_configs
        }
        x0 = initial_log_kp_state(self.n, kp_lim)
        initial_log_kp_std_vec = np.ones(self.n, dtype=float) * initial_log_kp_std(kp_lim)
        P0 = np.diag(initial_log_kp_std_vec ** 2)
        Q = np.eye(self.n) * float(log_kp_process_noise_var)
        resolved_covariance_max_var = float(max_log_kp_covariance_var)
        if resolved_covariance_max_var <= 0.0:
            resolved_covariance_max_var = float(np.max(initial_log_kp_std_vec ** 2))
        rospy.loginfo(
            "deflecomp_node: initial K=%s log(K) std=%s log(K) process noise var=%.6g covariance max var=%.6g",
            ", ".join(f"{v:.6g}" for v in np.exp(x0)),
            ", ".join(f"{v:.6g}" for v in initial_log_kp_std_vec),
            float(log_kp_process_noise_var),
            resolved_covariance_max_var,
        )
        estimator = MultiFrameStiffnessWEKF(
            x0=x0,
            P0=P0,
            Q=Q,
            solver=self.solver,
            sensitivity=self.sensitivity,
            eps_def=1e-6,
            observability_rcond=float(observability_rcond),
            observability_abs=float(observability_abs),
            laplace_outer_iterations=int(laplace_outer_iterations),
            max_log_kp_update_step=float(max_log_kp_update_step),
            max_equilibrium_pose_jump=float(max_equilibrium_pose_jump),
            joint_limit_reaction_torque_tol=float(joint_limit_reaction_torque_tol),
            max_log_kp_covariance_var=resolved_covariance_max_var,
        )
        observation_builder = ImuObservationBuilder(
            robot=self.robot,
            g_world=np.array([0.0, 0.0, -9.81], dtype=float),
            parameter_A=float(A_param),
        )
        self.compensator = DeflectionCompensator(
            robot=self.robot,
            spring_model=self.spring_model,
            stiffness_estimator=estimator,
            command_generator=CommandGenerator(robot=self.robot, spring_model=self.spring_model),
            equilibrium_solver=self.solver,
            observation_builder=observation_builder,
            config={
                "theta_cmd_tau": float(theta_cmd_tau),
                "theta_cmd_l1_regularization": bool(theta_cmd_l1_regularization),
                "theta_cmd_l1_regularization_weight": float(theta_cmd_l1_regularization_weight),
                "theta_cmd_equilibrium_refine": bool(theta_cmd_equilibrium_refine),
                "theta_cmd_equilibrium_refine_maxiter": int(theta_cmd_equilibrium_refine_maxiter),
                "theta_cmd_equilibrium_refine_tol": float(theta_cmd_equilibrium_refine_tol),
                "theta_cmd_equilibrium_refine_max_delta": float(theta_cmd_equilibrium_refine_max_delta),
                "kp_lim": tuple(float(v) for v in kp_lim),
                "update_stiffness": bool(update_stiffness),
                "project_unobservable_feedforward": bool(project_unobservable_feedforward),
                "feedforward_observability_rcond": float(observability_rcond),
                "feedforward_observability_abs": float(observability_abs),
                "kp_exec_tau": float(kp_exec_tau),
                "max_log_kp_exec_step": float(max_log_kp_exec_step),
                "particle_scan_enabled": bool(particle_scan_enabled),
                "particle_scan_window_size": int(particle_scan_window_size),
                "particle_scan_grid_size": int(particle_scan_grid_size),
                "particle_scan_reset_std": float(particle_scan_reset_std),
                "particle_scan_backend": str(particle_scan_backend),
                "particle_pursuit_mixture_weight": float(particle_pursuit_mixture_weight),
            },
        )
        rospy.on_shutdown(self.compensator.shutdown)
        particle_scan_actual_backend = getattr(self.compensator, "_particle_scan_backend", "none")
        rospy.loginfo(
            "deflecomp_node: particle_scan enabled=%s backend=%s window_size=%d grid_size=%d reset_std=%.6g pursuit_mixture_weight=%.6g",
            bool(particle_scan_enabled),
            str(particle_scan_actual_backend),
            int(particle_scan_window_size),
            int(particle_scan_grid_size),
            float(particle_scan_reset_std),
            float(particle_pursuit_mixture_weight),
        )
        self.kp_lim = kp_lim
        self.dt = float(dt)
        self.publish_kp_exec = bool(publish_kp_exec)
        self.imu_gravity_norm = float(imu_gravity_norm)
        self.imu_acceleration_tolerance = float(imu_acceleration_tolerance)
        self.imu_max_angular_speed = float(imu_max_angular_speed)
        self.imu_settle_time = max(0.0, float(imu_settle_time))
        self.imu_max_age = max(0.0, float(imu_max_age))
        self.estimation_settle_time = max(0.0, float(estimation_settle_time))
        self.estimation_command_tolerance = max(
            0.0, float(estimation_command_tolerance)
        )
        self.estimation_reference_tolerance = max(
            0.0, float(estimation_reference_tolerance)
        )
        self.command_apply_delay = max(0.0, float(command_apply_delay))

        self.q_ref = np.zeros(self.n, dtype=float)
        self.ref_joint_positions: Dict[str, float] = {name: 0.0 for name in self.output_joint_names}
        self.have_ref = False
        self.imu_bufs: Dict[str, ImuBuffer] = {cfg.frame_id: ImuBuffer(maxlen=2000) for cfg in self.imu_frame_configs}
        self.imu_reject_until: Dict[str, float] = {cfg.frame_id: -np.inf for cfg in self.imu_frame_configs}
        self.command_history = TimedVectorHistory(maxlen=4000)
        self.reference_history = TimedVectorHistory(maxlen=4000)

        self.sub_ref = rospy.Subscriber(topic_ref, JointState, self.cb_ref, queue_size=50)
        self.sub_imu = rospy.Subscriber(topic_imu, Imu, self.cb_imu, queue_size=400)
        self.pub_cmd = rospy.Publisher(topic_cmd_out, JointState, queue_size=10)
        self.pub_kp = rospy.Publisher("/deflecomp/kp_hat", Float64MultiArray, queue_size=10)
        self.pub_kp_est = rospy.Publisher("/deflecomp/kp_est", Float64MultiArray, queue_size=10)
        self.pub_kp_exec = rospy.Publisher("/deflecomp/kp_exec", Float64MultiArray, queue_size=10)
        self.pub_kp_exec_target = rospy.Publisher("/deflecomp/kp_exec_target", Float64MultiArray, queue_size=10)
        self.pub_cov = rospy.Publisher("/deflecomp/kp_cov_diag", Float64MultiArray, queue_size=10)
        self.pub_theta_eq = rospy.Publisher("/deflecomp/theta_eq_hat", Float64MultiArray, queue_size=10)
        self.pub_tau = rospy.Publisher("/deflecomp/tau_hat", Float64MultiArray, queue_size=10)
        self.pub_debug = rospy.Publisher("/deflecomp/debug", Float64MultiArray, queue_size=10)
        self.pub_estimation_gate_status = rospy.Publisher(
            "/deflecomp/estimation_gate_status", String, queue_size=10
        )
        self.pub_particle_scan_status = rospy.Publisher("/deflecomp/particle_scan_status", String, queue_size=10)
        self.pub_particle_scan_debug = rospy.Publisher("/deflecomp/particle_scan_debug", Float64MultiArray, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo(
            "deflecomp_node: base=%s tip=%s joints=%s locked_joints=%s imu_frames=%s spring=%s estimation_settle=%.3fs command_apply_delay=%.3fs",
            self.robot.base_link_name,
            self.robot.tip_link_name,
            ", ".join(self.model_joint_names),
            ", ".join(self.robot.locked_joint_names) if self.robot.locked_joint_names else "(none)",
            ", ".join(f"{cfg.frame_id}->{cfg.model_frame}" for cfg in self.imu_frame_configs),
            type(self.spring_model).__name__,
            self.estimation_settle_time,
            self.command_apply_delay,
        )

    def cb_ref(self, msg: JointState) -> None:
        q_ref = map_jointstate_to_model(msg, self.model_joint_names)
        self.q_ref = q_ref
        self.ref_joint_positions.update(jointstate_position_map(msg))
        self.reference_history.push(rospy.Time.now().to_sec(), q_ref)
        self.have_ref = True

    def cb_imu(self, msg: Imu) -> None:
        frame_name = (msg.header.frame_id or "").strip()
        cfg = self.imu_config_by_frame_id.get(frame_name)
        if cfg is None:
            return
        accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=float,
        )
        angular_velocity = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=float,
        )
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        if stamp <= 0.0:
            stamp = rospy.get_time()
        if not imu_sample_is_quasi_static(
            linear_acceleration=accel,
            angular_velocity=angular_velocity,
            gravity_norm=self.imu_gravity_norm,
            acceleration_tolerance=self.imu_acceleration_tolerance,
            max_angular_speed=self.imu_max_angular_speed,
        ):
            self.imu_reject_until[frame_name] = max(
                self.imu_reject_until[frame_name], stamp + self.imu_settle_time
            )
            self.imu_bufs[frame_name].clear()
            return
        if stamp < self.imu_reject_until[frame_name]:
            return
        g_sensor = -accel / (np.linalg.norm(accel) + 1e-12)
        g_dir = cfg.R_model_imu @ g_sensor
        g_dir = g_dir / (np.linalg.norm(g_dir) + 1e-12)
        self.imu_bufs[frame_name].push(stamp, g_dir)

    def _build_observations_at(self, t_align: Optional[float]) -> List[FrameImuObservation]:
        if t_align is None:
            return []
        observations: List[FrameImuObservation] = []
        for cfg in self.imu_frame_configs:
            sample = self.imu_bufs[cfg.frame_id].interpolate_with_support_stamp(
                t_align,
                max_age=self.imu_max_age,
            )
            if sample is None:
                continue
            g_dir, support_stamp = sample
            observations.append(
                FrameImuObservation(
                    frame_name=cfg.model_frame,
                    gravity_dir=g_dir,
                    stamp=t_align,
                    source_stamp=support_stamp,
                )
            )
        return observations

    def _latest_common_imu_stamp(self) -> Optional[float]:
        latest_stamps = [
            self.imu_bufs[cfg.frame_id].latest_timestamp()
            for cfg in self.imu_frame_configs
        ]
        if not latest_stamps or any(stamp is None for stamp in latest_stamps):
            return None
        return float(min(stamp for stamp in latest_stamps if stamp is not None))

    def _settled_observations_and_command(
        self,
        now: float,
    ) -> Tuple[List[FrameImuObservation], Optional[np.ndarray], Optional[float], str]:
        """Build one causal, quasi-static estimator input batch.

        IMU data are aligned at the newest timestamp shared by every configured
        frame.  The command is then looked up in publication history with the
        configured application delay.  Both command and reference must have
        remained stable throughout the dwell window; otherwise the static
        equilibrium likelihood is not applicable.
        """
        t_align = self._latest_common_imu_stamp()
        if t_align is None:
            return [], None, None, "waiting_for_all_imu_frames"
        if t_align > float(now) + 1.0e-6:
            return [], None, t_align, "imu_stamp_in_future"
        if float(now) - t_align > self.imu_max_age:
            return [], None, t_align, "latest_imu_is_stale"

        reference_match = self.reference_history.settled_value_at(
            observation_stamp=t_align,
            dwell_time=self.estimation_settle_time,
            tolerance=self.estimation_reference_tolerance,
        )
        if reference_match is None:
            return [], None, t_align, "reference_not_settled"

        command_match = self.command_history.settled_value_at(
            observation_stamp=t_align,
            dwell_time=self.estimation_settle_time,
            tolerance=self.estimation_command_tolerance,
            apply_delay=self.command_apply_delay,
        )
        if command_match is None:
            return [], None, t_align, "command_not_settled"

        observations = self._build_observations_at(t_align)
        if len(observations) != len(self.imu_frame_configs):
            return [], None, t_align, "imu_frames_not_time_aligned"
        theta_cmd_sent, _ = command_match
        return observations, theta_cmd_sent, t_align, "ready"

    def on_timer(self, event) -> None:
        del event
        if not self.have_ref:
            return

        now = rospy.Time.now().to_sec()
        observations, theta_cmd_sent, t_align, gate_reason = (
            self._settled_observations_and_command(now)
        )
        result = self.compensator.step(
            theta_ref=self.q_ref,
            imu_observations=observations,
            dt=self.dt,
            stamp=now,
            theta_cmd_sent_for_update=theta_cmd_sent,
        )
        result.debug["estimation_gate_reason"] = gate_reason
        result.debug["estimation_alignment_stamp"] = t_align
        align_text = "none" if t_align is None else f"{t_align:.9f}"
        est_update_applied = bool(result.debug.get("est_update_applied", False))
        est_update_skipped_reason = result.debug.get("est_update_skipped_reason")
        if est_update_skipped_reason is None:
            est_update_skipped_reason = "none" if est_update_applied else "not_attempted"
        laplace_step_scale = float(result.debug.get("laplace_step_scale", 0.0))
        laplace_dx_max_abs = float(result.debug.get("laplace_dx_max_abs", 0.0))
        laplace_outer_requested = int(
            result.debug.get("laplace_outer_iterations_requested", 0)
        )
        laplace_outer_completed = int(
            result.debug.get("laplace_outer_iterations_completed", 0)
        )
        laplace_outer_accepted = int(
            result.debug.get("laplace_outer_iterations_accepted", 0)
        )
        laplace_outer_stop_reason = result.debug.get("laplace_outer_stop_reason", "none")
        laplace_prior_covariance_capped = bool(
            result.debug.get("laplace_prior_covariance_capped", False)
        )
        self.pub_estimation_gate_status.publish(
            String(
                data=(
                    f"reason={gate_reason} alignment_stamp={align_text} "
                    f"observation_count={len(observations)} "
                    f"time_aligned_command={int(theta_cmd_sent is not None)} "
                    f"est_update_applied={int(est_update_applied)} "
                    f"est_update_skipped_reason={est_update_skipped_reason} "
                    f"laplace_step_scale={laplace_step_scale:.9g} "
                    f"laplace_dx_max_abs={laplace_dx_max_abs:.9g} "
                    f"laplace_outer_requested={laplace_outer_requested} "
                    f"laplace_outer_completed={laplace_outer_completed} "
                    f"laplace_outer_accepted={laplace_outer_accepted} "
                    f"laplace_outer_stop_reason={laplace_outer_stop_reason} "
                    f"laplace_prior_covariance_capped={int(laplace_prior_covariance_capped)}"
                )
            )
        )

        kp_hat = result.kp_hat

        # Record the actual publication time, not the timer callback start.
        # Estimator work can take milliseconds, and stamping at callback entry
        # formerly made the command appear to precede the IMU sample that was
        # already available at entry.
        command_publish_time = rospy.Time.now()
        command_publish_stamp = command_publish_time.to_sec()
        cmd_msg = JointState()
        cmd_msg.header.stamp = command_publish_time
        cmd_msg.name = self.output_joint_names
        cmd_msg.position = expand_joint_positions(
            output_names=self.output_joint_names,
            active_names=self.model_joint_names,
            active_positions=result.theta_cmd,
            fallback_positions=self.ref_joint_positions,
        )
        self.pub_cmd.publish(cmd_msg)
        self.command_history.push(command_publish_stamp, result.theta_cmd)

        cov_diag = np.clip(np.diag(self.compensator.stiffness_estimator.P_est), 0.0, np.inf)
        self.pub_kp.publish(Float64MultiArray(data=kp_hat.tolist()))
        self.pub_kp_est.publish(Float64MultiArray(data=result.kp_est.tolist()))
        if self.publish_kp_exec:
            self.pub_kp_exec.publish(Float64MultiArray(data=result.kp_exec.tolist()))
            self.pub_kp_exec_target.publish(Float64MultiArray(data=result.kp_exec_target.tolist()))
        self.pub_cov.publish(Float64MultiArray(data=cov_diag.tolist()))
        self.pub_theta_eq.publish(Float64MultiArray(data=result.theta_eq_hat.tolist()))
        self.pub_tau.publish(Float64MultiArray(data=result.tau_hat.tolist()))

        debug_vector = np.concatenate(
            [
                np.asarray(result.theta_cmd_raw, dtype=float),
                np.asarray(result.theta_eq_hat, dtype=float),
                np.asarray(result.tau_hat, dtype=float),
                np.asarray(cov_diag, dtype=float),
                np.asarray(result.kp_est, dtype=float),
                np.asarray(result.kp_exec, dtype=float),
                np.asarray(result.kp_exec_target, dtype=float),
            ]
        )
        self.pub_debug.publish(Float64MultiArray(data=debug_vector.tolist()))
        self._publish_particle_scan_debug(result.debug)

    def _publish_particle_scan_debug(self, debug: Dict[str, Any]) -> None:
        scan = debug.get("particle_scan")
        if not isinstance(scan, dict):
            status = "attempted=0 accepted=0 reason=not_run"
            vector = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, 0.0, 0.0], dtype=float)
        else:
            attempted = bool(scan.get("attempted", False))
            accepted = bool(scan.get("accepted", False))
            reason = str(scan.get("reason", "unknown"))
            gain_per_obs = float(scan.get("gain_per_obs", 0.0))
            score_current = float(scan.get("score_current", -np.inf))
            score_best = float(scan.get("score_best", -np.inf))
            candidate_count = int(scan.get("candidate_count", 0))
            max_jump = float(scan.get("max_jump", 0.0))
            active_indices = np.asarray(scan.get("active_indices", []), dtype=int)
            window_size = int(scan.get("window_size", 0))
            status = (
                f"attempted={int(attempted)} accepted={int(accepted)} reason={reason} "
                f"gain_per_obs={gain_per_obs:.6g} candidates={candidate_count} "
                f"score_current={score_current:.6g} score_best={score_best:.6g} "
                f"max_jump={max_jump:.6g} active_indices={active_indices.tolist()}"
            )
            vector = np.array(
                [
                    float(attempted),
                    float(accepted),
                    gain_per_obs,
                    float(candidate_count),
                    score_current,
                    score_best,
                    max_jump,
                    float(window_size),
                ],
                dtype=float,
            )
            if accepted:
                rospy.loginfo_throttle(
                    1.0,
                    "deflecomp_node: particle scan accepted gain_per_obs=%.6g score_current=%.6g score_best=%.6g max_jump=%.6g active_indices=%s",
                    gain_per_obs,
                    score_current,
                    score_best,
                    max_jump,
                    active_indices.tolist(),
                )
        self.pub_particle_scan_status.publish(String(data=status))
        self.pub_particle_scan_debug.publish(Float64MultiArray(data=vector.tolist()))


def main() -> None:
    rospy.init_node("deflecomp_node", anonymous=False)

    urdf_path = rospy.get_param("~urdf_path", resolve_default_urdf())
    imu_frames = rospy.get_param("~imu_frames", rospy.get_param("~frames", []))
    topic_ref = rospy.get_param("~topic_ref", "/ref/joint_states")
    topic_imu = rospy.get_param("~topic_imu", "/imu")
    topic_cmd_out = rospy.get_param("~topic_cmd_out", "/deflecomp/theta_cmd")
    dt = float(rospy.get_param("~dt", 0.02))
    A_param = float(rospy.get_param("~A_param", 100.0))
    kp_min = float(rospy.get_param("~kp_min", 1.0))
    kp_max = float(rospy.get_param("~kp_max", 500.0))
    log_kp_process_noise_var = float(rospy.get_param("~log_kp_process_noise_var", 0.30))
    max_log_kp_covariance_var = float(
        rospy.get_param("~max_log_kp_covariance_var", 0.0)
    )
    spring_model_name = rospy.get_param("~spring_model", "auto")
    theta_cmd_tau = float(rospy.get_param("~theta_cmd_tau", 0.2))
    theta_cmd_l1_regularization = parse_bool(rospy.get_param("~theta_cmd_l1_regularization", True))
    theta_cmd_l1_regularization_weight = float(rospy.get_param("~theta_cmd_l1_regularization_weight", 1e-4))
    theta_cmd_equilibrium_refine = parse_bool(
        rospy.get_param("~theta_cmd_equilibrium_refine", False)
    )
    theta_cmd_equilibrium_refine_maxiter = int(
        rospy.get_param("~theta_cmd_equilibrium_refine_maxiter", 2)
    )
    theta_cmd_equilibrium_refine_tol = float(
        rospy.get_param("~theta_cmd_equilibrium_refine_tol", 1e-5)
    )
    theta_cmd_equilibrium_refine_max_delta = float(
        rospy.get_param("~theta_cmd_equilibrium_refine_max_delta", 0.25)
    )
    equilibrium_refine = parse_bool(rospy.get_param("~equilibrium_refine", True))
    equilibrium_refine_maxiter = int(rospy.get_param("~equilibrium_refine_maxiter", 40))
    equilibrium_refine_tol = float(rospy.get_param("~equilibrium_refine_tol", 1e-12))
    update_stiffness = parse_bool(rospy.get_param("~update_stiffness", True))
    observability_rcond = float(rospy.get_param("~observability_rcond", 1e-4))
    observability_abs = float(rospy.get_param("~observability_abs", 1e-10))
    laplace_outer_iterations = int(rospy.get_param("~laplace_outer_iterations", 5))
    max_log_kp_update_step = float(rospy.get_param("~max_log_kp_update_step", 3.0))
    max_equilibrium_pose_jump = float(rospy.get_param("~max_equilibrium_pose_jump", 0.30))
    joint_limit_reaction_torque_tol = float(
        rospy.get_param("~joint_limit_reaction_torque_tol", 1.0e-3)
    )
    project_unobservable_feedforward = parse_bool(rospy.get_param("~project_unobservable_feedforward", False))
    kp_exec_tau = float(rospy.get_param("~kp_exec_tau", 1.0))
    max_log_kp_exec_step = float(rospy.get_param("~max_log_kp_exec_step", 0.0))
    publish_kp_exec = parse_bool(rospy.get_param("~publish_kp_exec", True))
    particle_scan_enabled = parse_bool(rospy.get_param("~particle_scan_enabled", False))
    particle_scan_window_size = int(rospy.get_param("~particle_scan_window_size", 20))
    particle_scan_grid_size = int(rospy.get_param("~particle_scan_grid_size", 21))
    particle_scan_reset_std = float(rospy.get_param("~particle_scan_reset_std", 0.10))
    particle_scan_backend = str(rospy.get_param("~particle_scan_backend", "process"))
    particle_pursuit_mixture_weight = float(rospy.get_param("~particle_pursuit_mixture_weight", 0.35))
    imu_gravity_norm = float(rospy.get_param("~imu_gravity_norm", 9.81))
    imu_acceleration_tolerance = float(rospy.get_param("~imu_acceleration_tolerance", 0.75))
    imu_max_angular_speed = float(rospy.get_param("~imu_max_angular_speed", 0.20))
    imu_settle_time = float(rospy.get_param("~imu_settle_time", 0.25))
    imu_max_age = float(rospy.get_param("~imu_max_age", 0.10))
    estimation_settle_time = float(rospy.get_param("~estimation_settle_time", 0.50))
    estimation_command_tolerance = float(
        rospy.get_param("~estimation_command_tolerance", 1.0e-3)
    )
    estimation_reference_tolerance = float(
        rospy.get_param("~estimation_reference_tolerance", 1.0e-4)
    )
    command_apply_delay = float(rospy.get_param("~command_apply_delay", 0.0))

    DeflecompNode(
        urdf_path=urdf_path,
        imu_frames=imu_frames,
        topic_ref=topic_ref,
        topic_imu=topic_imu,
        topic_cmd_out=topic_cmd_out,
        dt=dt,
        A_param=A_param,
        kp_lim=(kp_min, kp_max),
        log_kp_process_noise_var=log_kp_process_noise_var,
        spring_model_name=spring_model_name,
        theta_cmd_tau=theta_cmd_tau,
        theta_cmd_l1_regularization=theta_cmd_l1_regularization,
        theta_cmd_l1_regularization_weight=theta_cmd_l1_regularization_weight,
        theta_cmd_equilibrium_refine=theta_cmd_equilibrium_refine,
        theta_cmd_equilibrium_refine_maxiter=theta_cmd_equilibrium_refine_maxiter,
        theta_cmd_equilibrium_refine_tol=theta_cmd_equilibrium_refine_tol,
        theta_cmd_equilibrium_refine_max_delta=theta_cmd_equilibrium_refine_max_delta,
        equilibrium_refine=equilibrium_refine,
        equilibrium_refine_maxiter=equilibrium_refine_maxiter,
        equilibrium_refine_tol=equilibrium_refine_tol,
        update_stiffness=update_stiffness,
        observability_rcond=observability_rcond,
        observability_abs=observability_abs,
        laplace_outer_iterations=laplace_outer_iterations,
        max_log_kp_update_step=max_log_kp_update_step,
        max_equilibrium_pose_jump=max_equilibrium_pose_jump,
        joint_limit_reaction_torque_tol=joint_limit_reaction_torque_tol,
        max_log_kp_covariance_var=max_log_kp_covariance_var,
        project_unobservable_feedforward=project_unobservable_feedforward,
        kp_exec_tau=kp_exec_tau,
        max_log_kp_exec_step=max_log_kp_exec_step,
        publish_kp_exec=publish_kp_exec,
        particle_scan_enabled=particle_scan_enabled,
        particle_scan_window_size=particle_scan_window_size,
        particle_scan_grid_size=particle_scan_grid_size,
        particle_scan_reset_std=particle_scan_reset_std,
        particle_scan_backend=particle_scan_backend,
        particle_pursuit_mixture_weight=particle_pursuit_mixture_weight,
        imu_gravity_norm=imu_gravity_norm,
        imu_acceleration_tolerance=imu_acceleration_tolerance,
        imu_max_angular_speed=imu_max_angular_speed,
        imu_settle_time=imu_settle_time,
        imu_max_age=imu_max_age,
        estimation_settle_time=estimation_settle_time,
        estimation_command_tolerance=estimation_command_tolerance,
        estimation_reference_tolerance=estimation_reference_tolerance,
        command_apply_delay=command_apply_delay,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
