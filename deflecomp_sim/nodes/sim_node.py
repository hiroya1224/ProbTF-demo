#!/usr/bin/env python3
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import rospkg
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu, JointState

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.observation.imu_frame_config import ImuFrameConfig, resolve_imu_frame_configs
from deflecomp_core.model.spring import JointTypeAwareSpringModel, LinearSpringModel, PeriodicSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.dynamic_simulator import DynamicParams, FlexibleJointSimulator


def resolve_default_urdf() -> str:
    return os.path.join(rospkg.RosPack().get_path("deflecomp_description"), "urdf", "simple6r.urdf")


def parse_float_list(value) -> List[float]:
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    return [float(item) for item in items if str(item).strip()]


def parse_index_list(value) -> List[int]:
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    return [int(item) for item in items if str(item).strip()]


def parse_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def quat_xyzw_from_R(Rw: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(Rw))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (Rw[2, 1] - Rw[1, 2]) / s
        qy = (Rw[0, 2] - Rw[2, 0]) / s
        qz = (Rw[1, 0] - Rw[0, 1]) / s
    else:
        idx = int(np.argmax([Rw[0, 0], Rw[1, 1], Rw[2, 2]]))
        if idx == 0:
            s = np.sqrt(1.0 + Rw[0, 0] - Rw[1, 1] - Rw[2, 2]) * 2.0
            qx = 0.25 * s
            qy = (Rw[0, 1] + Rw[1, 0]) / s
            qz = (Rw[0, 2] + Rw[2, 0]) / s
            qw = (Rw[2, 1] - Rw[1, 2]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + Rw[1, 1] - Rw[0, 0] - Rw[2, 2]) * 2.0
            qx = (Rw[0, 1] + Rw[1, 0]) / s
            qy = 0.25 * s
            qz = (Rw[1, 2] + Rw[2, 1]) / s
            qw = (Rw[0, 2] - Rw[2, 0]) / s
        else:
            s = np.sqrt(1.0 + Rw[2, 2] - Rw[0, 0] - Rw[1, 1]) * 2.0
            qx = (Rw[0, 2] + Rw[2, 0]) / s
            qy = (Rw[1, 2] + Rw[2, 1]) / s
            qz = 0.25 * s
            qw = (Rw[1, 0] - Rw[0, 1]) / s
    return float(qx), float(qy), float(qz), float(qw)


def map_jointstate_to_model(msg: JointState, model_names: List[str]) -> np.ndarray:
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
    output_names: List[str],
    active_names: List[str],
    active_positions: np.ndarray,
    fallback_positions: Dict[str, float],
) -> List[float]:
    values = dict(fallback_positions)
    for name, position in zip(active_names, np.asarray(active_positions, dtype=float)):
        values[name] = float(position)
    return [float(values.get(name, 0.0)) for name in output_names]


def resolve_spring_model(name: str, robot: RobotArm):
    spring_name = str(name).strip().lower()
    if spring_name == "auto":
        return JointTypeAwareSpringModel.from_joint_types(robot.model_joint_types)
    if spring_name == "periodic":
        return PeriodicSpringModel()
    return LinearSpringModel()


class SimNode:
    def __init__(
        self,
        urdf_path: str,
        dt: float,
        kp_true: List[float],
        zeta: float,
        vel_lim: float,
        topic_cmd: str,
        topic_equil: str,
        imu_frames,
        imu_topic: str,
        integrator: str,
        ref_tau: float,
        ref_max_vel: float,
        eq_mode: str,
        tau_eq: float,
        qs_noise_std_deg: float,
        qs_vib_amp_deg: float,
        qs_vib_freq_hz: float,
        qs_vib_axes: List[int],
        qs_seed: Optional[int],
        spring_model_name: str,
        equilibrium_refine: bool,
        equilibrium_refine_maxiter: int,
        equilibrium_refine_tol: float,
        external_wrench_topic: str,
        external_wrench_timeout: float,
    ) -> None:
        self.dt = float(dt)
        self.topic_equil = topic_equil
        self.imu_topic = imu_topic
        self.external_wrench_timeout = float(external_wrench_timeout)

        self.robot = RobotArm(urdf_path)
        self.n = self.robot.nv
        self.joint_names = self.robot.model_joint_names
        self.output_joint_names = self.robot.urdf_info.movable_joint_names or self.joint_names

        Ktrue = np.resize(np.asarray(kp_true, dtype=float), self.n)
        params = DynamicParams(
            K=Ktrue,
            D=None,
            zeta=float(zeta),
            q0_for_damp=np.zeros(self.n, dtype=float),
            use_pinv=True,
            limit_velocity=np.ones(self.n, dtype=float) * float(vel_lim),
            limit_position_low=self.robot.model.lowerPositionLimit,
            limit_position_high=self.robot.model.upperPositionLimit,
            integrator=integrator,
            ref_tau=float(ref_tau),
            ref_max_vel=float(ref_max_vel),
            eq_mode=eq_mode,
            tau_eq=float(tau_eq),
            qs_noise_std_deg=float(qs_noise_std_deg),
            qs_vib_amp_deg=float(qs_vib_amp_deg),
            qs_vib_freq_hz=float(qs_vib_freq_hz),
            qs_vib_axes=(np.asarray(qs_vib_axes, dtype=int) if qs_vib_axes else None),
            qs_seed=qs_seed,
        )
        self.sim = FlexibleJointSimulator(
            robot=self.robot,
            params=params,
            spring_model=resolve_spring_model(spring_model_name, self.robot),
        )
        self.sim.set_eq_solver(
            EquilibriumSolver(
                robot=self.robot,
                spring_model=self.sim.spring_model,
                cfg=EquilibriumConfig(
                    maxiter=80,
                    refine=bool(equilibrium_refine),
                    refine_maxiter=int(equilibrium_refine_maxiter),
                    refine_tol=float(equilibrium_refine_tol),
                ),
            )
        )
        self.sim.reset(q=np.zeros(self.n, dtype=float), qd=np.zeros(self.n, dtype=float))

        self.have_cmd = False
        self.theta_cmd = np.zeros(self.n, dtype=float)
        self.cmd_joint_positions: Dict[str, float] = {name: 0.0 for name in self.output_joint_names}
        self.external_wrench_frame: Optional[str] = None
        self.external_wrench_reference_frame = "world"
        self.external_wrench_force = np.zeros(3, dtype=float)
        self.external_wrench_torque = np.zeros(3, dtype=float)
        self.external_wrench_stamp: Optional[float] = None
        self.external_wrench_warned_frames = set()

        self.pub_equil = rospy.Publisher(self.topic_equil, JointState, queue_size=10)
        self.pub_imu = rospy.Publisher(self.imu_topic, Imu, queue_size=20)
        self.sub_cmd = rospy.Subscriber(topic_cmd, JointState, self.cb_cmd, queue_size=50)
        self.sub_external_wrench = rospy.Subscriber(
            external_wrench_topic,
            WrenchStamped,
            self.cb_external_wrench,
            queue_size=10,
        )

        self.imu_frame_configs: List[ImuFrameConfig] = resolve_imu_frame_configs(
            robot=self.robot,
            value=imu_frames,
            count=max(1, min(3, len(self.joint_names))),
        )
        self.imu_fids: Dict[str, int] = {}
        for cfg in self.imu_frame_configs:
            self.imu_fids[cfg.frame_id] = self.robot.get_frame_id(cfg.model_frame)

        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo(
            "deflecomp_sim: base=%s tip=%s joints=%s locked_joints=%s imu_frames=%s spring=%s external_wrench_topic=%s",
            self.robot.base_link_name,
            self.robot.tip_link_name,
            ", ".join(self.joint_names),
            ", ".join(self.robot.locked_joint_names) if self.robot.locked_joint_names else "(none)",
            ", ".join(f"{cfg.frame_id}->{cfg.model_frame}" for cfg in self.imu_frame_configs),
            type(self.sim.spring_model).__name__,
            external_wrench_topic,
        )

    def cb_cmd(self, msg: JointState) -> None:
        self.theta_cmd = map_jointstate_to_model(msg, self.joint_names)
        self.cmd_joint_positions.update(jointstate_position_map(msg))
        for name, position in zip(self.joint_names, self.theta_cmd):
            self.cmd_joint_positions[name] = float(position)
        self.have_cmd = True

    def cb_external_wrench(self, msg: WrenchStamped) -> None:
        frame_name, reference_frame = self._parse_external_wrench_frame_id(msg.header.frame_id)
        if not frame_name:
            self.external_wrench_frame = None
            self.external_wrench_reference_frame = "world"
            self.external_wrench_force[:] = 0.0
            self.external_wrench_torque[:] = 0.0
            self.external_wrench_stamp = rospy.get_time()
            return

        self.external_wrench_frame = frame_name
        self.external_wrench_reference_frame = reference_frame
        self.external_wrench_force = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
            dtype=float,
        )
        self.external_wrench_torque = np.array(
            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            dtype=float,
        )
        self.external_wrench_stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()

    def _model_frame_name_from_tf_frame(self, frame_name: str) -> str:
        candidate = (frame_name or "").strip().lstrip("/")
        if self.robot.has_frame(candidate):
            return candidate
        while "/" in candidate:
            candidate = candidate.split("/", 1)[1]
            if self.robot.has_frame(candidate):
                return candidate
        return (frame_name or "").strip()

    def _parse_external_wrench_frame_id(self, value: str) -> Tuple[str, str]:
        raw = (value or "").strip()
        if not raw:
            return "", "world"
        if "@" not in raw:
            return self._model_frame_name_from_tf_frame(raw), "world"
        frame_name, reference_frame = [part.strip() for part in raw.split("@", 1)]
        model_frame_name = self._model_frame_name_from_tf_frame(frame_name)
        reference = reference_frame.lower()
        if reference in ("", "world", "base", "map"):
            return model_frame_name, "world"
        if reference in ("local", "frame", "target"):
            return model_frame_name, "local"
        rospy.logwarn(
            "deflecomp_sim: unsupported external wrench reference frame '%s'; using world",
            reference_frame,
        )
        return model_frame_name, "world"

    def _external_wrench_active(self) -> bool:
        if self.external_wrench_frame is None:
            return False
        if self.external_wrench_timeout > 0.0 and self.external_wrench_stamp is not None:
            if rospy.get_time() - float(self.external_wrench_stamp) > self.external_wrench_timeout:
                return False
        return bool(
            np.linalg.norm(self.external_wrench_force) > 0.0
            or np.linalg.norm(self.external_wrench_torque) > 0.0
        )

    def _external_wrench_tau(self, q: np.ndarray) -> np.ndarray:
        if not self._external_wrench_active():
            return np.zeros(self.n, dtype=float)

        frame_name = str(self.external_wrench_frame)
        if not self.robot.has_frame(frame_name):
            if frame_name not in self.external_wrench_warned_frames:
                rospy.logwarn("deflecomp_sim: external wrench target frame not found: %s", frame_name)
                self.external_wrench_warned_frames.add(frame_name)
            return np.zeros(self.n, dtype=float)

        fid = self.robot.get_frame_id(frame_name)
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        J6 = pin.computeFrameJacobian(
            self.robot.model,
            self.robot.data,
            q,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        force = self.external_wrench_force.reshape(3)
        torque = self.external_wrench_torque.reshape(3)
        if self.external_wrench_reference_frame == "local":
            R_wf = self.robot.data.oMf[fid].rotation
            force = R_wf @ force
            torque = R_wf @ torque
        return J6[0:3, :].T @ force + J6[3:6, :].T @ torque

    def publish_imus(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, now: rospy.Time) -> None:
        pin.forwardKinematics(self.robot.model, self.robot.data, q, qd, qdd)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        g_world = self.robot.model.gravity.linear
        for cfg in self.imu_frame_configs:
            fid = self.imu_fids[cfg.frame_id]
            v6_l = pin.getFrameVelocity(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            a6_l = pin.getFrameAcceleration(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            w_l = np.array(v6_l.angular).reshape(3)
            alpha_l = np.array(a6_l.angular).reshape(3)
            a_o_l = np.array(a6_l.linear).reshape(3)
            r_li = cfg.xyz.reshape(3)
            a_p_l = a_o_l + np.cross(alpha_l, r_li) + np.cross(w_l, np.cross(w_l, r_li))
            R_wl = self.robot.data.oMf[fid].rotation
            R_wi = R_wl @ cfg.R_model_imu
            a_p_i = cfg.R_model_imu.T @ a_p_l
            w_i = cfg.R_model_imu.T @ w_l
            g_i = R_wi.T @ g_world
            a_meas = a_p_i - g_i
            qx, qy, qz, qw = quat_xyzw_from_R(R_wi)

            imu_msg = Imu()
            imu_msg.header.stamp = now
            imu_msg.header.frame_id = cfg.frame_id
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            imu_msg.angular_velocity.x = float(w_i[0])
            imu_msg.angular_velocity.y = float(w_i[1])
            imu_msg.angular_velocity.z = float(w_i[2])
            imu_msg.linear_acceleration.x = float(a_meas[0])
            imu_msg.linear_acceleration.y = float(a_meas[1])
            imu_msg.linear_acceleration.z = float(a_meas[2])
            imu_msg.orientation_covariance[0] = -1.0
            imu_msg.angular_velocity_covariance[0] = -1.0
            imu_msg.linear_acceleration_covariance[0] = -1.0
            self.pub_imu.publish(imu_msg)

    def on_timer(self, event) -> None:
        del event
        if not self.have_cmd:
            return
        q_prev, qd_prev = self.sim.state()
        tau_ext_fn = self._external_wrench_tau if self._external_wrench_active() else None
        q_next, qd_next = self.sim.step(dt=self.dt, q_ref=self.theta_cmd, tau_ext_fn=tau_ext_fn)
        qdd_est = (qd_next - qd_prev) / max(self.dt, 1e-9)
        now = rospy.Time.now()

        js = JointState()
        js.header.stamp = now
        js.name = self.output_joint_names
        js.position = expand_joint_positions(
            output_names=self.output_joint_names,
            active_names=self.joint_names,
            active_positions=q_next,
            fallback_positions=self.cmd_joint_positions,
        )
        self.pub_equil.publish(js)
        self.publish_imus(q_next, qd_next, qdd_est, now)


def main() -> None:
    rospy.init_node("deflecomp_sim", anonymous=False)

    urdf_path = rospy.get_param("~urdf_path", resolve_default_urdf())
    dt = float(rospy.get_param("~dt", 0.004))
    kp_true = parse_float_list(rospy.get_param("~kp_true", [60, 60, 40, 40, 20, 20]))
    zeta = float(rospy.get_param("~zeta", 0.9))
    vel_lim = float(rospy.get_param("~vel_limit", 10.0))
    topic_cmd = rospy.get_param("~topic_cmd", "/deflecomp/theta_cmd")
    topic_equil = rospy.get_param("~topic_equil", "/joint_states")
    imu_frames = rospy.get_param("~imu_frames", rospy.get_param("~frames", ""))
    imu_topic = rospy.get_param("~topic_imu", "/imu")
    integrator = rospy.get_param("~integrator", "rk4")
    ref_tau = float(rospy.get_param("~ref_tau", 1e-9))
    ref_max_vel = float(rospy.get_param("~ref_max_vel", 1000.0))
    eq_mode = rospy.get_param("~eq_mode", "dynamic")
    tau_eq = float(rospy.get_param("~tau_eq", 0.05))
    qs_noise_std_deg = float(rospy.get_param("~qs_noise_std_deg", 0.0))
    qs_vib_amp_deg = float(rospy.get_param("~qs_vib_amp_deg", 0.0))
    qs_vib_freq_hz = float(rospy.get_param("~qs_vib_freq_hz", 50.0))
    qs_vib_axes = parse_index_list(rospy.get_param("~qs_vib_axes", []))
    qs_seed = rospy.get_param("~qs_seed", None)
    spring_model_name = rospy.get_param("~spring_model", "auto")
    equilibrium_refine = parse_bool(rospy.get_param("~equilibrium_refine", True))
    equilibrium_refine_maxiter = int(rospy.get_param("~equilibrium_refine_maxiter", 40))
    equilibrium_refine_tol = float(rospy.get_param("~equilibrium_refine_tol", 1e-12))
    external_wrench_topic = rospy.get_param("~external_wrench_topic", "/deflecomp_sim/external_wrench")
    external_wrench_timeout = float(rospy.get_param("~external_wrench_timeout", 0.0))

    SimNode(
        urdf_path=urdf_path,
        dt=dt,
        kp_true=kp_true,
        zeta=zeta,
        vel_lim=vel_lim,
        topic_cmd=topic_cmd,
        topic_equil=topic_equil,
        imu_frames=imu_frames,
        imu_topic=imu_topic,
        integrator=integrator,
        ref_tau=ref_tau,
        ref_max_vel=ref_max_vel,
        eq_mode=eq_mode,
        tau_eq=tau_eq,
        qs_noise_std_deg=qs_noise_std_deg,
        qs_vib_amp_deg=qs_vib_amp_deg,
        qs_vib_freq_hz=qs_vib_freq_hz,
        qs_vib_axes=qs_vib_axes,
        qs_seed=(int(qs_seed) if qs_seed is not None else None),
        spring_model_name=spring_model_name,
        equilibrium_refine=equilibrium_refine,
        equilibrium_refine_maxiter=equilibrium_refine_maxiter,
        equilibrium_refine_tol=equilibrium_refine_tol,
        external_wrench_topic=external_wrench_topic,
        external_wrench_timeout=external_wrench_timeout,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
