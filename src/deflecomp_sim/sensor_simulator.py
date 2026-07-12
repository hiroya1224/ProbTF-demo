from typing import Dict, List, Optional, Tuple

import numpy as np

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel, SpringModel
from deflecomp_core.observation.imu_observation import FrameImuObservation, ImuObservationBuilder
from deflecomp_core.robot.pinocchio_robot import RobotArm


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
