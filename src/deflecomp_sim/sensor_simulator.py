from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel, SpringModel
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.observation.imu_frame_config import ImuFrameConfig, quat_xyzw_from_matrix
from deflecomp_core.robot.pinocchio_robot import RobotArm


@dataclass(frozen=True)
class ImuKinematicSample:
    frame_id: str
    orientation_xyzw: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


def build_imu_kinematic_samples(
    robot: RobotArm,
    frame_configs: Sequence[ImuFrameConfig],
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> List[ImuKinematicSample]:
    configuration = np.asarray(q, dtype=float).reshape(robot.nv)
    velocity = np.asarray(qd, dtype=float).reshape(robot.nv)
    acceleration = np.asarray(qdd, dtype=float).reshape(robot.nv)
    pin.forwardKinematics(robot.model, robot.data, configuration, velocity, acceleration)
    pin.updateFramePlacements(robot.model, robot.data)

    samples = []
    gravity_world = robot.model.gravity.linear
    for config in frame_configs:
        frame_id = robot.get_frame_id(config.model_frame)
        velocity_local = pin.getFrameVelocity(robot.model, robot.data, frame_id, pin.ReferenceFrame.LOCAL)
        acceleration_local = pin.getFrameAcceleration(
            robot.model,
            robot.data,
            frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        angular_velocity_local = np.asarray(velocity_local.angular, dtype=float).reshape(3)
        angular_acceleration_local = np.asarray(acceleration_local.angular, dtype=float).reshape(3)
        origin_acceleration_local = np.asarray(acceleration_local.linear, dtype=float).reshape(3)
        offset_local = config.xyz.reshape(3)
        point_acceleration_local = (
            origin_acceleration_local
            + np.cross(angular_acceleration_local, offset_local)
            + np.cross(
                angular_velocity_local,
                np.cross(angular_velocity_local, offset_local),
            )
        )
        rotation_world_model = robot.data.oMf[frame_id].rotation
        rotation_world_imu = rotation_world_model @ config.R_model_imu
        point_acceleration_imu = config.R_model_imu.T @ point_acceleration_local
        angular_velocity_imu = config.R_model_imu.T @ angular_velocity_local
        gravity_imu = rotation_world_imu.T @ gravity_world

        samples.append(
            ImuKinematicSample(
                frame_id=config.frame_id,
                orientation_xyzw=quat_xyzw_from_matrix(rotation_world_imu),
                angular_velocity=angular_velocity_imu,
                linear_acceleration=point_acceleration_imu - gravity_imu,
            )
        )
    return samples


class SyntheticObservationBuilder:
    def __init__(
        self,
        robot_sim: RobotArm,
        spring_model: Optional[SpringModel] = None,
        g_world: Optional[np.ndarray] = None,
        parameter_A: float = 100.0,
    ) -> None:
        self.robot_sim = robot_sim
        self.g_world = np.array([0.0, 0.0, -9.81], dtype=float) if g_world is None else np.asarray(g_world, dtype=float)
        self.spring_model = spring_model or LinearSpringModel()
        self.observation_builder = ImuObservationBuilder(robot=robot_sim, g_world=self.g_world, parameter_A=parameter_A)

    def build_frame_observations(
        self,
        theta_cmd: np.ndarray,
        kp_true: np.ndarray,
        frame_names: List[str],
        newton_iter_true: int = 60,
        theta_ws_true: Optional[np.ndarray] = None,
    ) -> Tuple[List[FrameImuObservation], np.ndarray]:
        solver = EquilibriumSolver(
            robot=self.robot_sim,
            spring_model=self.spring_model,
            cfg=EquilibriumConfig(maxiter=newton_iter_true),
        )
        theta_equil_true = solver.solve(
            theta_cmd=np.asarray(theta_cmd, dtype=float),
            kp_vec=np.asarray(kp_true, dtype=float),
            theta_init=theta_ws_true,
        )
        observations: List[FrameImuObservation] = []
        for name in frame_names:
            fid = self.robot_sim.get_frame_id(name)
            g_f = self.robot_sim.gravity_dir_in_frame(theta_equil_true, self.g_world, fid)
            observations.append(FrameImuObservation(frame_name=name, gravity_dir=g_f))
        return observations, theta_equil_true

    def build_A_multi(
        self,
        theta_cmd: np.ndarray,
        kp_true: np.ndarray,
        frame_names: List[str],
        newton_iter_true: int = 60,
        theta_ws_true: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        observations, theta_equil_true = self.build_frame_observations(
            theta_cmd=theta_cmd,
            kp_true=kp_true,
            frame_names=frame_names,
            newton_iter_true=newton_iter_true,
            theta_ws_true=theta_ws_true,
        )
        return self.observation_builder.build_A_map(observations), theta_equil_true
