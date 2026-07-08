from typing import TYPE_CHECKING, Tuple

import numpy as np

from deflecomp_core.model.spring import SpringModel

if TYPE_CHECKING:
    from deflecomp_core.robot.pinocchio_robot import RobotArm


class SensitivityCalculator:
    def __init__(self, robot: "RobotArm", spring_model: SpringModel) -> None:
        self.robot = robot
        self.spring_model = spring_model

    def equilibrium_jacobians(
        self,
        theta_eq: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_tau_g = self.robot.d_tau_gravity(theta_eq)
        j_q = d_tau_g + np.diag(self.spring_model.stiffness_diag(theta_eq, theta_cmd, kp_vec))
        j_x = np.diag(self.spring_model.log_stiffness_jacobian_diag(theta_eq, theta_cmd, kp_vec))
        return j_q, j_x
