#!/usr/bin/env python3
import os
import threading
from bisect import bisect_left
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from deflecomp_core.control.feedforward import CommandGenerator
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import JointTypeAwareSpringModel, LinearSpringModel, PeriodicSpringModel
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.pipeline.compensator import DeflectionCompensator
from deflecomp_core.robot.pinocchio_robot import RobotArm


class ImuBuffer:
    def __init__(self, maxlen: int = 1000) -> None:
        self.t_list: List[float] = []
        self.g_list: List[np.ndarray] = []
        self.maxlen = int(maxlen)
        self.lock = threading.RLock()

    def push(self, t: float, g_dir: np.ndarray) -> None:
        g = np.asarray(g_dir, dtype=float)
        g = g / (np.linalg.norm(g) + 1e-12)
        with self.lock:
            idx = bisect_left(self.t_list, t)
            if idx < len(self.t_list) and abs(self.t_list[idx] - t) < 1e-12:
                self.t_list[idx] = t
                self.g_list[idx] = g
            else:
                self.t_list.insert(idx, t)
                self.g_list.insert(idx, g)
            while len(self.t_list) > self.maxlen:
                self.t_list.pop(0)
                self.g_list.pop(0)

    def interpolate(self, t: float) -> Optional[np.ndarray]:
        with self.lock:
            if not self.t_list:
                return None
            if t <= self.t_list[0]:
                return self.g_list[0].copy()
            if t >= self.t_list[-1]:
                return self.g_list[-1].copy()

            idx = bisect_left(self.t_list, t)
            t0 = self.t_list[idx - 1]
            t1 = self.t_list[idx]
            g0 = self.g_list[idx - 1]
            g1 = self.g_list[idx]
            if t1 - t0 <= 1e-12:
                return g1.copy()
            alpha = (t - t0) / (t1 - t0)
            g = (1.0 - alpha) * g0 + alpha * g1
            return g / (np.linalg.norm(g) + 1e-12)


def parse_string_list(value) -> List[str]:
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def parse_float_list(value) -> List[float]:
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    return [float(item) for item in items if str(item).strip()]


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


def resolve_spring_model(name: str, robot: RobotArm):
    spring_name = str(name).strip().lower()
    if spring_name == "auto":
        return JointTypeAwareSpringModel.from_joint_types(robot.model_joint_types)
    if spring_name == "linear":
        return LinearSpringModel()
    return PeriodicSpringModel()


class DeflecompNode:
    def __init__(
        self,
        urdf_path: str,
        frames: Sequence[str],
        topic_ref: str,
        topic_imu: str,
        topic_cmd_out: str,
        dt: float,
        A_param: float,
        kp0: Sequence[float],
        kp_lim: Tuple[float, float],
        q_proc: float,
        spring_model_name: str,
        theta_cmd_tau: float,
        update_stiffness: bool,
        observability_rcond: float,
        observability_abs: float,
        measurement_info_eig_cap: float,
        stiffness_update_gain: float,
        max_log_kp_step: float,
        min_log_kp_step: float,
        project_unobservable_feedforward: bool,
    ) -> None:
        self.robot = RobotArm(urdf_path)
        self.spring_model = resolve_spring_model(spring_model_name, self.robot)
        self.solver = EquilibriumSolver(
            robot=self.robot,
            spring_model=self.spring_model,
            cfg=EquilibriumConfig(maxiter=80),
        )
        self.sensitivity = SensitivityCalculator(robot=self.robot, spring_model=self.spring_model)
        self.n = self.robot.nv
        self.model_joint_names = self.robot.model_joint_names
        self.output_joint_names = self.robot.urdf_info.movable_joint_names or self.model_joint_names

        self.frames = self.robot.suggest_imu_frames(
            preferred=frames,
            count=max(1, min(3, len(self.model_joint_names))),
        )
        if not self.frames:
            raise ValueError(f"No valid IMU frames found in URDF: {urdf_path}")
        x0 = np.log(np.resize(np.asarray(kp0, dtype=float), self.n))
        P0 = np.eye(self.n) * 1.0
        Q = np.eye(self.n) * float(q_proc)
        estimator = MultiFrameStiffnessWEKF(
            x0=x0,
            P0=P0,
            Q=Q,
            solver=self.solver,
            sensitivity=self.sensitivity,
            eps_def=1e-6,
            observability_rcond=float(observability_rcond),
            observability_abs=float(observability_abs),
            measurement_info_eig_cap=float(measurement_info_eig_cap),
            update_gain=float(stiffness_update_gain),
            max_log_kp_step=float(max_log_kp_step),
            min_log_kp_step=float(min_log_kp_step),
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
                "kp_lim": tuple(float(v) for v in kp_lim),
                "update_stiffness": bool(update_stiffness),
                "project_unobservable_feedforward": bool(project_unobservable_feedforward),
                "feedforward_observability_rcond": float(observability_rcond),
                "feedforward_observability_abs": float(observability_abs),
            },
        )
        self.kp_lim = kp_lim
        self.dt = float(dt)

        self.q_ref = np.zeros(self.n, dtype=float)
        self.ref_joint_positions: Dict[str, float] = {name: 0.0 for name in self.output_joint_names}
        self.have_ref = False
        self.imu_bufs: Dict[str, ImuBuffer] = {name: ImuBuffer(maxlen=2000) for name in self.frames}

        self.sub_ref = rospy.Subscriber(topic_ref, JointState, self.cb_ref, queue_size=50)
        self.sub_imu = rospy.Subscriber(topic_imu, Imu, self.cb_imu, queue_size=400)
        self.pub_cmd = rospy.Publisher(topic_cmd_out, JointState, queue_size=10)
        self.pub_kp = rospy.Publisher("/deflecomp/kp_hat", Float64MultiArray, queue_size=10)
        self.pub_cov = rospy.Publisher("/deflecomp/kp_cov_diag", Float64MultiArray, queue_size=10)
        self.pub_theta_eq = rospy.Publisher("/deflecomp/theta_eq_hat", Float64MultiArray, queue_size=10)
        self.pub_tau = rospy.Publisher("/deflecomp/tau_hat", Float64MultiArray, queue_size=10)
        self.pub_debug = rospy.Publisher("/deflecomp/debug", Float64MultiArray, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo(
            "deflecomp_node: base=%s tip=%s joints=%s locked_joints=%s frames=%s spring=%s",
            self.robot.base_link_name,
            self.robot.tip_link_name,
            ", ".join(self.model_joint_names),
            ", ".join(self.robot.locked_joint_names) if self.robot.locked_joint_names else "(none)",
            ", ".join(self.frames),
            type(self.spring_model).__name__,
        )

    def cb_ref(self, msg: JointState) -> None:
        self.q_ref = map_jointstate_to_model(msg, self.model_joint_names)
        self.ref_joint_positions.update(jointstate_position_map(msg))
        self.have_ref = True

    def cb_imu(self, msg: Imu) -> None:
        frame_name = (msg.header.frame_id or "").strip()
        if frame_name not in self.imu_bufs:
            return
        accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=float,
        )
        if np.linalg.norm(accel) < 1e-9:
            return
        g_dir = -accel / (np.linalg.norm(accel) + 1e-12)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        self.imu_bufs[frame_name].push(stamp, g_dir)

    def _build_observations_at(self, t_align: Optional[float]) -> List[FrameImuObservation]:
        if t_align is None:
            return []
        observations: List[FrameImuObservation] = []
        for frame_name in self.frames:
            g_dir = self.imu_bufs[frame_name].interpolate(t_align)
            if g_dir is None:
                continue
            observations.append(FrameImuObservation(frame_name=frame_name, gravity_dir=g_dir, stamp=t_align))
        return observations

    def on_timer(self, event) -> None:
        del event
        if not self.have_ref:
            return

        now = rospy.Time.now().to_sec()
        observations = self._build_observations_at(self.compensator.last_stamp)
        result = self.compensator.step(
            theta_ref=self.q_ref,
            imu_observations=observations,
            dt=self.dt,
            stamp=now,
        )

        kp_hat = result.kp_hat

        cmd_msg = JointState()
        cmd_msg.header.stamp = rospy.Time.from_sec(now)
        cmd_msg.name = self.output_joint_names
        cmd_msg.position = expand_joint_positions(
            output_names=self.output_joint_names,
            active_names=self.model_joint_names,
            active_positions=result.theta_cmd,
            fallback_positions=self.ref_joint_positions,
        )
        self.pub_cmd.publish(cmd_msg)

        cov_diag = np.clip(np.diag(self.compensator.stiffness_estimator.P), 0.0, np.inf)
        self.pub_kp.publish(Float64MultiArray(data=kp_hat.tolist()))
        self.pub_cov.publish(Float64MultiArray(data=cov_diag.tolist()))
        self.pub_theta_eq.publish(Float64MultiArray(data=result.theta_eq_hat.tolist()))
        self.pub_tau.publish(Float64MultiArray(data=result.tau_hat.tolist()))

        debug_vector = np.concatenate(
            [
                np.asarray(result.theta_cmd_raw, dtype=float),
                np.asarray(result.theta_eq_hat, dtype=float),
                np.asarray(result.tau_hat, dtype=float),
                np.asarray(cov_diag, dtype=float),
            ]
        )
        self.pub_debug.publish(Float64MultiArray(data=debug_vector.tolist()))


def main() -> None:
    rospy.init_node("deflecomp_node", anonymous=False)

    urdf_path = rospy.get_param("~urdf_path", resolve_default_urdf())
    frames = parse_string_list(rospy.get_param("~frames", []))
    topic_ref = rospy.get_param("~topic_ref", "/ref/joint_states")
    topic_imu = rospy.get_param("~topic_imu", "/imu")
    topic_cmd_out = rospy.get_param("~topic_cmd_out", "/deflecomp/theta_cmd")
    dt = float(rospy.get_param("~dt", 0.02))
    A_param = float(rospy.get_param("~A_param", 100.0))
    kp0 = parse_float_list(rospy.get_param("~kp0", [50, 50, 50, 50, 50, 50]))
    kp_min = float(rospy.get_param("~kp_min", 1.0))
    kp_max = float(rospy.get_param("~kp_max", 500.0))
    q_proc = float(rospy.get_param("~q_proc", 1e-8))
    spring_model_name = rospy.get_param("~spring_model", "auto")
    theta_cmd_tau = float(rospy.get_param("~theta_cmd_tau", 0.2))
    update_stiffness = parse_bool(rospy.get_param("~update_stiffness", True))
    observability_rcond = float(rospy.get_param("~observability_rcond", 1e-4))
    observability_abs = float(rospy.get_param("~observability_abs", 1e-10))
    measurement_info_eig_cap = float(rospy.get_param("~measurement_info_eig_cap", 1.0))
    stiffness_update_gain = float(rospy.get_param("~stiffness_update_gain", 0.2))
    max_log_kp_step = float(rospy.get_param("~max_log_kp_step", 0.002))
    min_log_kp_step = float(rospy.get_param("~min_log_kp_step", 0.0))
    project_unobservable_feedforward = parse_bool(rospy.get_param("~project_unobservable_feedforward", True))

    DeflecompNode(
        urdf_path=urdf_path,
        frames=frames,
        topic_ref=topic_ref,
        topic_imu=topic_imu,
        topic_cmd_out=topic_cmd_out,
        dt=dt,
        A_param=A_param,
        kp0=kp0,
        kp_lim=(kp_min, kp_max),
        q_proc=q_proc,
        spring_model_name=spring_model_name,
        theta_cmd_tau=theta_cmd_tau,
        update_stiffness=update_stiffness,
        observability_rcond=observability_rcond,
        observability_abs=observability_abs,
        measurement_info_eig_cap=measurement_info_eig_cap,
        stiffness_update_gain=stiffness_update_gain,
        max_log_kp_step=max_log_kp_step,
        min_log_kp_step=min_log_kp_step,
        project_unobservable_feedforward=project_unobservable_feedforward,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
