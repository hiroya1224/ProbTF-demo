#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import rospkg
import rospy
from sensor_msgs.msg import Imu, JointState

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import JointTypeAwareSpringModel, LinearSpringModel, PeriodicSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.dynamic_simulator import DynamicParams, FlexibleJointSimulator


@dataclass
class ImuOffset:
    R_li: np.ndarray
    r_li: np.ndarray
    name: str


def resolve_default_urdf() -> str:
    return os.path.join(rospkg.RosPack().get_path("deflecomp_description"), "urdf", "simple6r.urdf")


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


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


def parse_imu_frames(spec: str) -> List[Tuple[str, ImuOffset]]:
    items = [item.strip() for item in spec.split(";") if item.strip()]
    parsed: List[Tuple[str, ImuOffset]] = []
    for item in items:
        name = item
        r_li = np.zeros(3, dtype=float)
        R_li = np.eye(3, dtype=float)
        if "@" in item:
            parts = item.split("@")
            name = parts[0].strip()
            if len(parts) >= 2 and parts[1]:
                r_li = np.array([float(x) for x in parts[1].split(",")], dtype=float)
            if len(parts) >= 3 and parts[2]:
                roll, pitch, yaw = [float(x) for x in parts[2].split(",")]
                R_li = rpy_to_R(roll, pitch, yaw)
        parsed.append((name, ImuOffset(R_li=R_li, r_li=r_li, name=name)))
    return parsed


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


def resolve_imu_offsets(robot: RobotArm, spec: str, count: int) -> Dict[str, ImuOffset]:
    parsed = parse_imu_frames(spec)
    resolved: Dict[str, ImuOffset] = {}
    for frame_name, offset in parsed:
        if robot.has_frame(frame_name):
            resolved[frame_name] = offset

    suggested_frames = robot.suggest_imu_frames(
        preferred=list(resolved.keys()),
        count=max(1, count),
    )
    for frame_name in suggested_frames:
        resolved.setdefault(
            frame_name,
            ImuOffset(R_li=np.eye(3, dtype=float), r_li=np.zeros(3, dtype=float), name=frame_name),
        )
    return resolved


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
        imu_frames: str,
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
    ) -> None:
        self.dt = float(dt)
        self.topic_equil = topic_equil
        self.imu_topic = imu_topic

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

        self.pub_equil = rospy.Publisher(self.topic_equil, JointState, queue_size=10)
        self.pub_imu = rospy.Publisher(self.imu_topic, Imu, queue_size=20)
        self.sub_cmd = rospy.Subscriber(topic_cmd, JointState, self.cb_cmd, queue_size=50)

        self.imu_offsets: Dict[str, ImuOffset] = {}
        self.imu_fids: Dict[str, int] = {}
        for frame_name, offset in resolve_imu_offsets(
            robot=self.robot,
            spec=imu_frames,
            count=max(1, min(3, len(self.joint_names))),
        ).items():
            self.imu_offsets[frame_name] = offset
            self.imu_fids[frame_name] = self.robot.get_frame_id(frame_name)

        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo(
            "deflecomp_sim: base=%s tip=%s joints=%s locked_joints=%s imu_frames=%s spring=%s",
            self.robot.base_link_name,
            self.robot.tip_link_name,
            ", ".join(self.joint_names),
            ", ".join(self.robot.locked_joint_names) if self.robot.locked_joint_names else "(none)",
            ", ".join(self.imu_fids.keys()),
            type(self.sim.spring_model).__name__,
        )

    def cb_cmd(self, msg: JointState) -> None:
        self.theta_cmd = map_jointstate_to_model(msg, self.joint_names)
        self.cmd_joint_positions.update(jointstate_position_map(msg))
        for name, position in zip(self.joint_names, self.theta_cmd):
            self.cmd_joint_positions[name] = float(position)
        self.have_cmd = True

    def publish_imus(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, now: rospy.Time) -> None:
        pin.forwardKinematics(self.robot.model, self.robot.data, q, qd, qdd)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        g_world = self.robot.model.gravity.linear
        for frame_name, fid in self.imu_fids.items():
            offset = self.imu_offsets[frame_name]
            v6_l = pin.getFrameVelocity(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            a6_l = pin.getFrameAcceleration(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            w_l = np.array(v6_l.angular).reshape(3)
            alpha_l = np.array(a6_l.angular).reshape(3)
            a_o_l = np.array(a6_l.linear).reshape(3)
            r_li = offset.r_li.reshape(3)
            a_p_l = a_o_l + np.cross(alpha_l, r_li) + np.cross(w_l, np.cross(w_l, r_li))
            R_wl = self.robot.data.oMf[fid].rotation
            R_wi = R_wl @ offset.R_li
            a_p_i = offset.R_li.T @ a_p_l
            w_i = offset.R_li.T @ w_l
            g_i = R_wi.T @ g_world
            a_meas = a_p_i - g_i
            qx, qy, qz, qw = quat_xyzw_from_R(R_wi)

            imu_msg = Imu()
            imu_msg.header.stamp = now
            imu_msg.header.frame_id = frame_name
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
        q_next, qd_next = self.sim.step(dt=self.dt, q_ref=self.theta_cmd, tau_ext=None)
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
    imu_frames = rospy.get_param("~imu_frames", "")
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
    )
    rospy.spin()


if __name__ == "__main__":
    main()
