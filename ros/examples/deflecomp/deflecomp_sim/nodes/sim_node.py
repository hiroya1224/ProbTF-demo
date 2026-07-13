#!/usr/bin/env python3
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import Imu, JointState
from visualization_msgs.msg import Marker

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.observation.imu_frame_config import ImuFrameConfig, resolve_imu_frame_configs
from deflecomp_core.model.spring import spring_model_from_name
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_sim.dynamic_simulator import DynamicParams, FlexibleJointSimulator
from deflecomp_sim.external_wrench import (
    external_force_arrow_points,
    frame_wrench_in_world,
    generalized_external_wrench,
)
from deflecomp_sim.sensor_simulator import build_imu_kinematic_samples


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
        external_wrench_marker_topic: str,
        external_wrench_force_marker_scale: float,
        external_wrench_marker_shaft_diameter: float,
        external_wrench_marker_head_diameter: float,
        external_wrench_marker_head_length: float,
        publish_rate_hz: float,
    ) -> None:
        self.dt = float(dt)
        self.topic_equil = topic_equil
        self.imu_topic = imu_topic
        self.external_wrench_timeout = float(external_wrench_timeout)
        self.external_wrench_marker_topic = str(external_wrench_marker_topic)
        self.external_wrench_force_marker_scale = float(
            external_wrench_force_marker_scale
        )
        self.external_wrench_marker_shaft_diameter = float(
            external_wrench_marker_shaft_diameter
        )
        self.external_wrench_marker_head_diameter = float(
            external_wrench_marker_head_diameter
        )
        self.external_wrench_marker_head_length = float(
            external_wrench_marker_head_length
        )
        marker_dimensions = (
            self.external_wrench_force_marker_scale,
            self.external_wrench_marker_shaft_diameter,
            self.external_wrench_marker_head_diameter,
            self.external_wrench_marker_head_length,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in marker_dimensions):
            raise ValueError("external wrench marker scale and dimensions must be positive")
        self.publish_rate_hz = float(publish_rate_hz)
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.publish_every_steps = max(
            1,
            int(round(1.0 / (self.dt * self.publish_rate_hz))),
        )
        self.steps_since_publish = 0

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
            spring_model=spring_model_from_name(spring_model_name, self.robot.model_joint_types),
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
        self.external_wrench_marker_visible = False

        self.pub_equil = rospy.Publisher(self.topic_equil, JointState, queue_size=10)
        self.pub_imu = rospy.Publisher(self.imu_topic, Imu, queue_size=20)
        self.pub_external_wrench_marker = rospy.Publisher(
            self.external_wrench_marker_topic,
            Marker,
            queue_size=1,
            latch=True,
        )
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
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo(
            "deflecomp_sim: base=%s tip=%s joints=%s locked_joints=%s imu_frames=%s spring=%s external_wrench_topic=%s external_wrench_marker_topic=%s publish_rate_hz=%.3f",
            self.robot.base_link_name,
            self.robot.tip_link_name,
            ", ".join(self.joint_names),
            ", ".join(self.robot.locked_joint_names) if self.robot.locked_joint_names else "(none)",
            ", ".join(f"{cfg.frame_id}->{cfg.model_frame}" for cfg in self.imu_frame_configs),
            type(self.sim.spring_model).__name__,
            external_wrench_topic,
            self.external_wrench_marker_topic,
            1.0 / (self.publish_every_steps * self.dt),
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

        return generalized_external_wrench(
            robot=self.robot,
            q=q,
            frame_name=frame_name,
            force=self.external_wrench_force,
            torque=self.external_wrench_torque,
            reference_frame=self.external_wrench_reference_frame,
        )

    def _external_wrench_marker(self, q: np.ndarray, now: rospy.Time) -> Optional[Marker]:
        if not self._external_wrench_active():
            return None

        frame_name = str(self.external_wrench_frame)
        if not self.robot.has_frame(frame_name):
            return None

        application_point_world, force_world, _ = frame_wrench_in_world(
            robot=self.robot,
            q=q,
            frame_name=frame_name,
            force=self.external_wrench_force,
            torque=self.external_wrench_torque,
            reference_frame=self.external_wrench_reference_frame,
        )
        if float(np.linalg.norm(force_world)) <= 1.0e-12:
            return None
        start_xyz, end_xyz = external_force_arrow_points(
            application_point_world=application_point_world,
            force_world=force_world,
            scale=self.external_wrench_force_marker_scale,
        )

        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self.robot.base_link_name
        marker.ns = "applied_external_force"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.points = [
            Point(x=float(start_xyz[0]), y=float(start_xyz[1]), z=float(start_xyz[2])),
            Point(x=float(end_xyz[0]), y=float(end_xyz[1]), z=float(end_xyz[2])),
        ]
        marker.scale.x = self.external_wrench_marker_shaft_diameter
        marker.scale.y = self.external_wrench_marker_head_diameter
        marker.scale.z = self.external_wrench_marker_head_length
        marker.color.r = 0.8
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration.from_sec(
            max(0.1, 2.0 / self.publish_rate_hz)
        )
        return marker

    def publish_external_wrench_marker(self, q: np.ndarray, now: rospy.Time) -> None:
        marker = self._external_wrench_marker(q, now)
        if marker is not None:
            self.pub_external_wrench_marker.publish(marker)
            self.external_wrench_marker_visible = True
            return
        if not self.external_wrench_marker_visible:
            return

        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = self.robot.base_link_name
        clear.ns = "applied_external_force"
        clear.id = 0
        clear.action = Marker.DELETE
        self.pub_external_wrench_marker.publish(clear)
        self.external_wrench_marker_visible = False

    def publish_imus(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, now: rospy.Time) -> None:
        samples = build_imu_kinematic_samples(self.robot, self.imu_frame_configs, q, qd, qdd)
        for sample in samples:
            qx, qy, qz, qw = sample.orientation_xyzw
            imu_msg = Imu()
            imu_msg.header.stamp = now
            imu_msg.header.frame_id = sample.frame_id
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            imu_msg.angular_velocity.x = float(sample.angular_velocity[0])
            imu_msg.angular_velocity.y = float(sample.angular_velocity[1])
            imu_msg.angular_velocity.z = float(sample.angular_velocity[2])
            imu_msg.linear_acceleration.x = float(sample.linear_acceleration[0])
            imu_msg.linear_acceleration.y = float(sample.linear_acceleration[1])
            imu_msg.linear_acceleration.z = float(sample.linear_acceleration[2])
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
        self.steps_since_publish += 1
        if self.steps_since_publish < self.publish_every_steps:
            return
        self.steps_since_publish = 0
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
        self.publish_external_wrench_marker(q_next, now)


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
    integrator = rospy.get_param("~integrator", "semi_implicit")
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
    external_wrench_marker_topic = rospy.get_param(
        "~external_wrench_marker_topic",
        "/deflecomp_sim/external_wrench_marker",
    )
    external_wrench_force_marker_scale = float(
        rospy.get_param("~external_wrench_force_marker_scale", 0.05)
    )
    external_wrench_marker_shaft_diameter = float(
        rospy.get_param("~external_wrench_marker_shaft_diameter", 0.015)
    )
    external_wrench_marker_head_diameter = float(
        rospy.get_param("~external_wrench_marker_head_diameter", 0.03)
    )
    external_wrench_marker_head_length = float(
        rospy.get_param("~external_wrench_marker_head_length", 0.04)
    )
    publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 100.0))

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
        external_wrench_marker_topic=external_wrench_marker_topic,
        external_wrench_force_marker_scale=external_wrench_force_marker_scale,
        external_wrench_marker_shaft_diameter=external_wrench_marker_shaft_diameter,
        external_wrench_marker_head_diameter=external_wrench_marker_head_diameter,
        external_wrench_marker_head_length=external_wrench_marker_head_length,
        publish_rate_hz=publish_rate_hz,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
