from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deflecomp_core.robot.pinocchio_robot import RobotArm


def gravity_torque(robot: "RobotArm", theta: np.ndarray) -> np.ndarray:
    return robot.tau_gravity(theta)


def gravity_torque_derivative(robot: "RobotArm", theta: np.ndarray) -> np.ndarray:
    return robot.d_tau_gravity(theta)
