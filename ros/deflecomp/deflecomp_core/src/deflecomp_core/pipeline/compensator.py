from dataclasses import asdict, dataclass
from copy import deepcopy
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from deflecomp_core.control.feedforward import CommandGenerator, lowpass_theta_cmd
from deflecomp_core.estimator.delay_rls import CommandDelayRLS
from deflecomp_core.estimator.stiffness_particle_supervisor import (
    StiffnessParticleScanConfig,
    StiffnessParticleScanSupervisor,
)
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import JointTypeAwareSpringModel, LinearSpringModel, PeriodicSpringModel
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


def _spring_model_spec(spring_model) -> Optional[Dict[str, Any]]:
    if isinstance(spring_model, JointTypeAwareSpringModel):
        return {
            "type": "joint_type_aware",
            "periodic_mask": np.asarray(spring_model.periodic_mask, dtype=bool).copy(),
        }
    if isinstance(spring_model, LinearSpringModel):
        return {"type": "linear"}
    if isinstance(spring_model, PeriodicSpringModel):
        return {"type": "periodic"}
    return None


def _spring_model_from_spec(spec: Dict[str, Any]):
    spring_type = str(spec.get("type", "")).strip().lower()
    if spring_type == "joint_type_aware":
        return JointTypeAwareSpringModel(np.asarray(spec["periodic_mask"], dtype=bool))
    if spring_type == "linear":
        return LinearSpringModel()
    if spring_type == "periodic":
        return PeriodicSpringModel()
    raise ValueError(f"Unsupported spring model for particle scan process: {spring_type}")


def _particle_worker_config(
    robot: RobotArm,
    spring_model,
    estimator: MultiFrameStiffnessWEKF,
) -> Optional[Dict[str, Any]]:
    spring_spec = _spring_model_spec(spring_model)
    if spring_spec is None:
        return None
    if not hasattr(robot, "urdf_path"):
        return None
    solver_cfg = getattr(getattr(estimator, "solver", None), "cfg", EquilibriumConfig())
    return {
        "urdf_path": robot.urdf_path,
        "tip_link": getattr(robot, "tip_link_name", None),
        "base_link": getattr(robot, "base_link_name", None),
        "spring_model": spring_spec,
        "equilibrium_config": asdict(solver_cfg),
        "Q": np.asarray(estimator.Q, dtype=float).copy(),
        "eps_def": float(estimator.eps_def),
        "observability_rcond": float(estimator.observability_rcond),
        "observability_abs": float(estimator.observability_abs),
        "laplace_negative_info_tol": float(estimator.laplace_negative_info_tol),
        "laplace_jitter": float(estimator.laplace_jitter),
        "nice": 10,
    }


def _build_particle_worker_estimator(config: Dict[str, Any]) -> MultiFrameStiffnessWEKF:
    robot = RobotArm(
        config["urdf_path"],
        tip_link=config.get("tip_link"),
        base_link=config.get("base_link"),
    )
    spring_model = _spring_model_from_spec(config["spring_model"])
    solver = EquilibriumSolver(
        robot=robot,
        spring_model=spring_model,
        cfg=EquilibriumConfig(**dict(config["equilibrium_config"])),
    )
    sensitivity = SensitivityCalculator(robot=robot, spring_model=spring_model)
    q = np.asarray(config["Q"], dtype=float)
    n = q.shape[0]
    return MultiFrameStiffnessWEKF(
        x0=np.zeros(n, dtype=float),
        P0=np.eye(n, dtype=float),
        Q=q,
        solver=solver,
        sensitivity=sensitivity,
        eps_def=float(config["eps_def"]),
        observability_rcond=float(config["observability_rcond"]),
        observability_abs=float(config["observability_abs"]),
        laplace_negative_info_tol=float(config["laplace_negative_info_tol"]),
        laplace_jitter=float(config["laplace_jitter"]),
    )


def _particle_scan_process_main(worker_config: Dict[str, Any], task_queue, result_queue) -> None:
    try:
        os.nice(int(worker_config.get("nice", 0)))
    except Exception:
        pass

    evaluator = None
    while True:
        task = task_queue.get()
        if task is None:
            return

        supervisor = task["supervisor"]
        try:
            if evaluator is None:
                evaluator = _build_particle_worker_estimator(worker_config)
            result = _run_particle_scan_task(
                supervisor=supervisor,
                evaluator=evaluator,
                kp_lim=task["kp_lim"],
                x_est=task["x_est"],
                P_est=task["P_est"],
                last_theta_eq=task["last_theta_eq"],
            )
        except Exception as exc:
            result = supervisor.status_result(
                reason="exception",
                attempted=True,
                x_current=task.get("x_est"),
                debug_extra={"exception": str(exc), "backend": "process"},
            )
        result_queue.put(result)


def _run_particle_scan_task(supervisor, evaluator, kp_lim, x_est, P_est, last_theta_eq):
    evaluator.x_est = np.asarray(x_est, dtype=float).copy()
    evaluator.P_est = np.asarray(P_est, dtype=float).copy()
    if hasattr(evaluator, "last_theta_eq"):
        evaluator.last_theta_eq = None if last_theta_eq is None else np.asarray(last_theta_eq, dtype=float).copy()
    return supervisor.maybe_scan(estimator=evaluator, kp_lim=kp_lim)


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
        self.stiffness_particle_supervisor = None
        if _as_bool(self.config.get("particle_scan_enabled", False)):
            particle_config = StiffnessParticleScanConfig(
                enabled=True,
                window_size=int(self.config.get("particle_scan_window_size", 20)),
                grid_size=int(self.config.get("particle_scan_grid_size", 21)),
                reset_std=float(self.config.get("particle_scan_reset_std", 0.10)),
            )
            self.stiffness_particle_supervisor = StiffnessParticleScanSupervisor(particle_config)

        self._particle_scan_lock = threading.RLock()
        self._particle_scan_thread: Optional[threading.Thread] = None
        self._particle_scan_running = False
        self._particle_scan_running_kind: Optional[str] = None
        self._particle_scan_pending_result = None
        self._particle_scan_last_result = (
            None if self.stiffness_particle_supervisor is None else self.stiffness_particle_supervisor.last_result
        )
        self._particle_scan_backend = "none"
        self._particle_scan_evaluator = None
        self._particle_scan_process = None
        self._particle_scan_task_queue = None
        self._particle_scan_result_queue = None
        self._particle_scan_process_context = None
        if self.stiffness_particle_supervisor is not None:
            requested_backend = str(self.config.get("particle_scan_backend", "process")).strip().lower()
            if requested_backend == "process" and self._start_particle_scan_process():
                self._particle_scan_backend = "process"
            else:
                self._particle_scan_backend = "thread"
                self._particle_scan_evaluator = self._make_particle_scan_evaluator()

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

    def _make_particle_scan_evaluator(self):
        clone_fn = getattr(self.stiffness_estimator, "clone_for_evaluation", None)
        if callable(clone_fn):
            return clone_fn()
        return deepcopy(self.stiffness_estimator)

    def _start_particle_scan_process(self) -> bool:
        try:
            worker_config = _particle_worker_config(
                robot=self.robot,
                spring_model=self.spring_model,
                estimator=self.stiffness_estimator,
            )
        except Exception:
            return False
        if worker_config is None:
            return False
        try:
            context = mp.get_context("spawn")
            task_queue = context.Queue(maxsize=1)
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=_particle_scan_process_main,
                args=(worker_config, task_queue, result_queue),
                name="deflecomp_particle_scan",
                daemon=True,
            )
            process.start()
        except Exception:
            return False

        self._particle_scan_process_context = context
        self._particle_scan_task_queue = task_queue
        self._particle_scan_result_queue = result_queue
        self._particle_scan_process = process
        return True

    def shutdown(self) -> None:
        process = self._particle_scan_process
        task_queue = self._particle_scan_task_queue
        if process is None:
            return
        if process.is_alive() and task_queue is not None:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
            process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)

    def _collect_particle_scan_result(self):
        if self._particle_scan_backend == "process":
            return self._collect_particle_scan_process_result()
        with self._particle_scan_lock:
            result = self._particle_scan_pending_result
            self._particle_scan_pending_result = None
            if result is not None:
                self._particle_scan_last_result = result
            return result

    def _collect_particle_scan_process_result(self):
        if self._particle_scan_result_queue is None:
            return None

        result = None
        while True:
            try:
                result = self._particle_scan_result_queue.get_nowait()
            except queue.Empty:
                break

        with self._particle_scan_lock:
            if result is not None:
                self._particle_scan_running = False
                self._particle_scan_running_kind = None
                self._particle_scan_last_result = result
                return result

            process = self._particle_scan_process
            if self._particle_scan_running and process is not None and not process.is_alive():
                self._particle_scan_running = False
                self._particle_scan_running_kind = None
                result = self.stiffness_particle_supervisor.status_result(
                    reason="process_stopped",
                    attempted=True,
                    x_current=self.stiffness_estimator.x_est,
                    debug_extra={"backend": "process"},
                )
                self._particle_scan_last_result = result
                return result
        return None

    def _start_particle_scan_if_idle(self, kp_lim) -> bool:
        if self.stiffness_particle_supervisor is None:
            return False
        if not self.stiffness_particle_supervisor.records:
            return False
        if self._particle_scan_backend == "process":
            return self._start_particle_scan_process_if_idle(kp_lim)
        with self._particle_scan_lock:
            if self._particle_scan_running or self._particle_scan_pending_result is not None:
                return False
            scan_supervisor = self.stiffness_particle_supervisor.snapshot()
            evaluator = self._particle_scan_evaluator
            if evaluator is None:
                return False
            x_est = self.stiffness_estimator.x_est.copy()
            P_est = self.stiffness_estimator.P_est.copy()
            last_theta_eq = (
                None
                if self.stiffness_estimator.last_theta_eq is None
                else self.stiffness_estimator.last_theta_eq.copy()
            )
            self._particle_scan_running = True
            self._particle_scan_running_kind = "scan"

        def worker() -> None:
            try:
                result = _run_particle_scan_task(
                    supervisor=scan_supervisor,
                    evaluator=evaluator,
                    kp_lim=kp_lim,
                    x_est=x_est,
                    P_est=P_est,
                    last_theta_eq=last_theta_eq,
                )
            except Exception as exc:
                result = scan_supervisor.status_result(
                    reason="exception",
                    attempted=True,
                    x_current=getattr(evaluator, "x_est", None),
                    debug_extra={"exception": str(exc), "backend": "thread"},
                )
            with self._particle_scan_lock:
                self._particle_scan_pending_result = result
                self._particle_scan_running = False
                self._particle_scan_running_kind = None

        thread = threading.Thread(target=worker, name="deflecomp_particle_scan", daemon=True)
        with self._particle_scan_lock:
            self._particle_scan_thread = thread
        thread.start()
        return True

    def _start_particle_scan_process_if_idle(self, kp_lim) -> bool:
        if self._particle_scan_task_queue is None:
            return False
        with self._particle_scan_lock:
            if self._particle_scan_running:
                return False
            process = self._particle_scan_process
            if process is None or not process.is_alive():
                self._particle_scan_last_result = self.stiffness_particle_supervisor.status_result(
                    reason="process_stopped",
                    attempted=True,
                    x_current=self.stiffness_estimator.x_est,
                    debug_extra={"backend": "process"},
                )
                return False
            scan_supervisor = self.stiffness_particle_supervisor.snapshot()
            last_theta_eq = (
                None
                if self.stiffness_estimator.last_theta_eq is None
                else self.stiffness_estimator.last_theta_eq.copy()
            )
            task = {
                "supervisor": scan_supervisor,
                "kp_lim": None if kp_lim is None else tuple(float(v) for v in kp_lim),
                "x_est": self.stiffness_estimator.x_est.copy(),
                "P_est": self.stiffness_estimator.P_est.copy(),
                "last_theta_eq": last_theta_eq,
            }
            self._particle_scan_running = True
            self._particle_scan_running_kind = "scan"
        try:
            self._particle_scan_task_queue.put_nowait(task)
        except queue.Full:
            with self._particle_scan_lock:
                self._particle_scan_running = False
                self._particle_scan_running_kind = None
            return False
        except Exception:
            with self._particle_scan_lock:
                self._particle_scan_running = False
                self._particle_scan_running_kind = None
                self._particle_scan_last_result = self.stiffness_particle_supervisor.status_result(
                    reason="enqueue_failed",
                    attempted=True,
                    x_current=self.stiffness_estimator.x_est,
                    debug_extra={"backend": "process"},
                )
            return False
        return True

    def wait_for_particle_scan(self, timeout: Optional[float] = None) -> bool:
        if self._particle_scan_backend == "process":
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            if timeout is None:
                while True:
                    self._collect_particle_scan_process_result()
                    with self._particle_scan_lock:
                        if not self._particle_scan_running:
                            return True
                    time.sleep(0.01)
            while time.monotonic() < deadline:
                self._collect_particle_scan_process_result()
                with self._particle_scan_lock:
                    if not self._particle_scan_running:
                        return True
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            self._collect_particle_scan_process_result()
            with self._particle_scan_lock:
                return not self._particle_scan_running
        with self._particle_scan_lock:
            thread = self._particle_scan_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        with self._particle_scan_lock:
            return not self._particle_scan_running

    def _particle_scan_status_result(self):
        if self.stiffness_particle_supervisor is None:
            return None
        with self._particle_scan_lock:
            running = bool(self._particle_scan_running)
            running_kind = self._particle_scan_running_kind
            pending_result = self._particle_scan_pending_result
            last_result = self._particle_scan_last_result or self.stiffness_particle_supervisor.last_result
        if pending_result is not None:
            return self.stiffness_particle_supervisor.status_result(
                reason="pending",
                attempted=True,
                x_current=self.stiffness_estimator.x_est,
                debug_extra={
                    "async_pending": True,
                    "pending_reason": pending_result.reason,
                },
            )
        if running:
            return self.stiffness_particle_supervisor.status_result(
                reason="running",
                attempted=True,
                x_current=self.stiffness_estimator.x_est,
                debug_extra={
                    "async_running": True,
                    "async_task": running_kind or "scan",
                    "backend": self._particle_scan_backend,
                    "last_completed_reason": last_result.reason,
                },
            )
        return last_result

    def _publish_particle_scan_debug_fields(self, debug: Dict[str, Any], scan_result) -> None:
        if scan_result is None:
            return
        scan_debug = dict(scan_result.debug)
        scan_debug.setdefault("backend", self._particle_scan_backend)
        debug["particle_scan"] = scan_debug
        debug["particle_scan_attempted"] = scan_result.attempted
        debug["particle_scan_accepted"] = scan_result.accepted
        debug["particle_scan_reason"] = scan_result.reason
        debug["particle_scan_gain_per_obs"] = scan_result.gain_per_obs
        debug["particle_scan_candidate_count"] = scan_result.candidate_count
        debug["particle_scan_active_indices"] = scan_result.active_indices.copy()
        debug["particle_scan_score_current"] = scan_result.score_current
        debug["particle_scan_score_best"] = scan_result.score_best
        debug["particle_scan_max_jump"] = float(scan_result.debug.get("max_jump", 0.0))

    def _apply_particle_scan_result(self, scan_result, debug: Dict[str, Any]) -> bool:
        self._publish_particle_scan_debug_fields(debug, scan_result)
        if scan_result is None or not scan_result.accepted:
            return False
        self.stiffness_estimator.apply_particle_correction(
            x_new=scan_result.x_best,
            active_indices=scan_result.active_indices,
            reset_std=self.stiffness_particle_supervisor.config.reset_std,
            pursuit_mixture_weight=float(self.config.get("particle_pursuit_mixture_weight", 0.35)),
            theta_eq=scan_result.theta_eq_best,
            kp_lim=self.config.get("kp_lim"),
        )
        self._update_exec_stiffness_target(self.stiffness_estimator.x_est)
        return True

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

    def _theta_cmd_equilibrium_delta(
        self,
        theta_ref: np.ndarray,
        theta_cmd: np.ndarray,
        theta_eq: np.ndarray,
        kp_vec: np.ndarray,
        debug: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        err = np.asarray(theta_ref, dtype=float) - np.asarray(theta_eq, dtype=float)
        n = err.size
        try:
            d_tau_g = np.asarray(self.robot.d_tau_gravity(theta_eq), dtype=float)
            spring_k = np.asarray(
                self.spring_model.stiffness_diag(theta_eq, theta_cmd, kp_vec),
                dtype=float,
            )
            if d_tau_g.shape != (n, n) or spring_k.shape != (n,):
                return err
            j_q = d_tau_g + np.diag(spring_k)
            j_cmd = np.diag(spring_k)
            try:
                eq_cmd_jac = np.linalg.solve(j_q, j_cmd)
            except np.linalg.LinAlgError:
                eq_cmd_jac = np.linalg.pinv(j_q, rcond=1e-12) @ j_cmd
            delta_l2 = np.linalg.pinv(eq_cmd_jac, rcond=1e-12) @ err
            if not self._theta_cmd_l1_regularization_enabled():
                return delta_l2
            return self._theta_cmd_l1_regularized_delta(
                eq_cmd_jac=eq_cmd_jac,
                err=err,
                theta_cmd=theta_cmd,
                theta_ref=theta_ref,
                delta_seed=delta_l2,
                debug=debug,
            )
        except Exception:
            return err

    def _theta_cmd_l1_regularization_enabled(self) -> bool:
        return _as_bool(self.config.get("theta_cmd_l1_regularization", True))

    def _theta_cmd_l1_regularization_weight(self) -> float:
        return max(0.0, float(self.config.get("theta_cmd_l1_regularization_weight", 1e-4)))

    def _theta_cmd_l1_regularization_epsilon(self) -> float:
        return max(1e-12, float(self.config.get("theta_cmd_l1_regularization_epsilon", 1e-6)))

    def _theta_cmd_l1_regularization_maxiter(self) -> int:
        return max(1, int(self.config.get("theta_cmd_l1_regularization_maxiter", 50)))

    def _theta_cmd_l1_regularized_delta(
        self,
        eq_cmd_jac: np.ndarray,
        err: np.ndarray,
        theta_cmd: np.ndarray,
        theta_ref: np.ndarray,
        delta_seed: np.ndarray,
        debug: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        weight = self._theta_cmd_l1_regularization_weight()
        if weight <= 0.0:
            return np.asarray(delta_seed, dtype=float)

        eq_cmd_jac = np.asarray(eq_cmd_jac, dtype=float)
        err = np.asarray(err, dtype=float)
        theta_cmd = np.asarray(theta_cmd, dtype=float)
        theta_ref = np.asarray(theta_ref, dtype=float)
        delta_seed = np.asarray(delta_seed, dtype=float)
        if (
            eq_cmd_jac.shape != (err.size, err.size)
            or theta_cmd.shape != err.shape
            or theta_ref.shape != err.shape
            or delta_seed.shape != err.shape
            or not np.all(np.isfinite(eq_cmd_jac))
        ):
            return delta_seed

        eps = self._theta_cmd_l1_regularization_epsilon()

        def objective(delta: np.ndarray):
            delta = np.asarray(delta, dtype=float)
            residual = eq_cmd_jac @ delta - err
            correction = theta_cmd + delta - theta_ref
            smooth_abs = np.sqrt(correction * correction + eps * eps)
            value = 0.5 * float(np.dot(residual, residual))
            value += weight * float(np.sum(smooth_abs - eps))
            grad = eq_cmd_jac.T @ residual + weight * correction / smooth_abs
            return value, grad

        seed_value, _ = objective(delta_seed)
        zero_delta = np.zeros_like(delta_seed)
        zero_value, _ = objective(zero_delta)
        x0 = zero_delta if zero_value < seed_value else delta_seed
        baseline_delta = x0
        baseline_value = min(seed_value, zero_value)

        try:
            res = minimize(
                fun=objective,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self._theta_cmd_l1_regularization_maxiter(),
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                },
            )
        except Exception as exc:
            if debug is not None:
                debug["theta_cmd_l1_regularization_success"] = False
                debug["theta_cmd_l1_regularization_reason"] = str(exc)
            return baseline_delta

        delta_opt = np.asarray(res.x, dtype=float)
        opt_value, _ = objective(delta_opt)
        if np.all(np.isfinite(delta_opt)) and opt_value <= baseline_value + 1e-12:
            if debug is not None:
                debug["theta_cmd_l1_regularization_success"] = bool(res.success)
                debug["theta_cmd_l1_regularization_objective"] = float(opt_value)
            return delta_opt

        if debug is not None:
            debug["theta_cmd_l1_regularization_success"] = False
            debug["theta_cmd_l1_regularization_reason"] = "no_objective_improvement"
        return baseline_delta

    def _theta_cmd_from_theta_ref(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
        debug: Dict[str, Any],
    ) -> np.ndarray:
        theta_cmd_direct = self.spring_model.theta_cmd_from_theta_ref(
            tau_gravity=tau_gravity,
            theta_ref=theta_ref,
            kp_vec=kp_vec,
        )
        enabled = self._theta_cmd_l1_regularization_enabled()
        weight = self._theta_cmd_l1_regularization_weight()
        debug["theta_cmd_l1_regularization_enabled"] = bool(enabled and weight > 0.0)
        debug["theta_cmd_l1_regularization_weight"] = float(weight)
        if not enabled or weight <= 0.0:
            return theta_cmd_direct

        return self._theta_cmd_l1_regularized_feedforward(
            tau_gravity=tau_gravity,
            theta_ref=theta_ref,
            kp_vec=kp_vec,
            theta_cmd_seed=theta_cmd_direct,
            debug=debug,
        )

    def _theta_cmd_l1_regularized_feedforward(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
        theta_cmd_seed: np.ndarray,
        debug: Dict[str, Any],
    ) -> np.ndarray:
        tau_gravity = np.asarray(tau_gravity, dtype=float)
        theta_ref = np.asarray(theta_ref, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        theta_cmd_seed = np.asarray(theta_cmd_seed, dtype=float)
        if tau_gravity.shape != theta_ref.shape or kp_vec.shape != theta_ref.shape:
            return theta_cmd_seed

        scale = np.maximum(np.abs(kp_vec), 1e-12)
        weight = self._theta_cmd_l1_regularization_weight()
        eps = self._theta_cmd_l1_regularization_epsilon()

        def objective(theta_cmd: np.ndarray):
            theta_cmd = np.asarray(theta_cmd, dtype=float)
            torque = np.asarray(
                self.spring_model.torque(theta_ref, theta_cmd, kp_vec),
                dtype=float,
            )
            residual = (tau_gravity + torque) / scale
            correction = theta_cmd - theta_ref
            smooth_abs = np.sqrt(correction * correction + eps * eps)
            value = 0.5 * float(np.dot(residual, residual))
            value += weight * float(np.sum(smooth_abs - eps))

            spring_k = np.asarray(
                self.spring_model.stiffness_diag(theta_ref, theta_cmd, kp_vec),
                dtype=float,
            )
            if spring_k.shape != theta_ref.shape:
                raise ValueError("spring stiffness shape does not match theta_ref")
            grad_residual = -(spring_k / scale) * residual
            grad_l1 = weight * correction / smooth_abs
            return value, grad_residual + grad_l1

        baseline = theta_cmd_seed.copy()
        try:
            seed_value, _ = objective(theta_cmd_seed)
            ref_value, _ = objective(theta_ref)
            x0 = theta_ref.copy() if ref_value < seed_value else theta_cmd_seed.copy()
            baseline = x0.copy()
            baseline_value = min(seed_value, ref_value)
            res = minimize(
                fun=objective,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self._theta_cmd_l1_regularization_maxiter(),
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                },
            )
        except Exception as exc:
            debug["theta_cmd_l1_feedforward_success"] = False
            debug["theta_cmd_l1_feedforward_reason"] = str(exc)
            return baseline

        theta_cmd_opt = np.asarray(res.x, dtype=float)
        opt_value, _ = objective(theta_cmd_opt)
        if np.all(np.isfinite(theta_cmd_opt)) and opt_value <= baseline_value + 1e-12:
            debug["theta_cmd_l1_feedforward_success"] = bool(res.success)
            debug["theta_cmd_l1_feedforward_objective"] = float(opt_value)
            return theta_cmd_opt

        debug["theta_cmd_l1_feedforward_success"] = False
        debug["theta_cmd_l1_feedforward_reason"] = "no_objective_improvement"
        return baseline

    def _refine_theta_cmd_for_equilibrium_ref(
        self,
        theta_cmd_seed: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
        theta_eq_init: np.ndarray,
        debug: Dict[str, Any],
    ):
        theta_cmd = np.asarray(theta_cmd_seed, dtype=float).copy()
        theta_eq = np.asarray(theta_eq_init, dtype=float).copy()
        maxiter = 2
        tol = 1e-5
        max_delta = 0.0
        final_err = np.inf
        applied_iters = 0

        for idx in range(maxiter + 1):
            theta_eq = self.equilibrium_solver.solve(
                theta_cmd=theta_cmd,
                kp_vec=kp_vec,
                theta_init=theta_eq,
            )
            err = np.asarray(theta_ref, dtype=float) - np.asarray(theta_eq, dtype=float)
            final_err = float(np.linalg.norm(err))
            if final_err <= tol or idx >= maxiter:
                break
            delta_cmd = self._theta_cmd_equilibrium_delta(
                theta_ref=theta_ref,
                theta_cmd=theta_cmd,
                theta_eq=theta_eq,
                kp_vec=kp_vec,
                debug=debug,
            )
            if not np.all(np.isfinite(delta_cmd)):
                break
            if float(np.linalg.norm(delta_cmd)) <= 1e-12:
                break
            if max_delta > 0.0:
                delta_cmd = np.clip(delta_cmd, -max_delta, max_delta)
            theta_cmd = theta_cmd + delta_cmd
            applied_iters += 1

        debug["theta_cmd_equilibrium_refine_iters"] = int(applied_iters)
        debug["theta_cmd_equilibrium_refine_error_norm"] = float(final_err)
        debug["theta_cmd_equilibrium_refine_enabled"] = True
        return theta_cmd, theta_eq

    def step(
        self,
        theta_ref: np.ndarray,
        imu_observations: Optional[Sequence[FrameImuObservation]],
        dt: float,
        stamp: Optional[float] = None,
    ) -> CompensationStepResult:
        theta_ref = np.asarray(theta_ref, dtype=float)
        debug: Dict[str, Any] = {}
        completed_particle_scan = self._collect_particle_scan_result()
        if completed_particle_scan is not None:
            self._apply_particle_scan_result(completed_particle_scan, debug)
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
                    debug.update(update_result.debug)
                    if self.stiffness_particle_supervisor is not None:
                        self.stiffness_particle_supervisor.add_record(
                            theta_cmd_sent=theta_cmd_sent,
                            A_map=a_map,
                            theta_init_eq_pred=theta_init,
                            stamp=observation_stamp,
                        )
                        self._start_particle_scan_if_idle(kp_lim)
                        if "particle_scan" not in debug:
                            self._publish_particle_scan_debug_fields(
                                debug,
                                self._particle_scan_status_result(),
                    )
                    self._update_exec_stiffness_target(self.stiffness_estimator.x_est)
        if self.stiffness_particle_supervisor is not None:
            self._start_particle_scan_if_idle(self.config.get("kp_lim"))
        if self.stiffness_particle_supervisor is not None and "particle_scan" not in debug:
            self._publish_particle_scan_debug_fields(debug, self._particle_scan_status_result())
        debug["update_stiffness"] = update_stiffness

        log_kp_exec_delta = self._smooth_exec_stiffness(dt_exec)
        kp_est = self.kp_est
        kp_exec = self.kp_exec
        kp_exec_target = self.kp_exec_target
        theta_gravity = self._theta_ref_for_gravity(theta_ref, imu_observations, debug)
        tau_gravity = self.robot.tau_gravity(theta_gravity)
        theta_cmd_raw = self._theta_cmd_from_theta_ref(
            tau_gravity=tau_gravity,
            theta_ref=theta_ref,
            kp_vec=kp_exec,
            debug=debug,
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
        theta_cmd, theta_eq_hat = self._refine_theta_cmd_for_equilibrium_ref(
            theta_cmd_seed=theta_cmd,
            theta_ref=theta_ref,
            kp_vec=kp_exec,
            theta_eq_init=theta_eq_init,
            debug=debug,
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
