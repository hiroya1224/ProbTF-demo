import numpy as np
from deflecomp_core.model.spring import LinearSpringModel, SpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm


class CommandGenerator:
    def __init__(self, robot: RobotArm, spring_model: SpringModel) -> None:
        self.robot = robot
        self.spring_model = spring_model

    def theta_cmd_from_theta_ref(self, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        theta_ref = np.asarray(theta_ref, dtype=float)
        tau_g = self.robot.tau_gravity(theta_ref)
        return self.spring_model.theta_cmd_from_theta_ref(
            tau_gravity=tau_g,
            theta_ref=theta_ref,
            kp_vec=np.asarray(kp_vec, dtype=float),
        )


def theta_cmd_from_theta_ref(
    robot: RobotArm,
    theta_ref: np.ndarray,
    kp_vec: np.ndarray,
    spring_model: SpringModel = None,
) -> np.ndarray:
    model = spring_model or LinearSpringModel()
    return CommandGenerator(robot=robot, spring_model=model).theta_cmd_from_theta_ref(
        theta_ref=np.asarray(theta_ref, dtype=float),
        kp_vec=np.asarray(kp_vec, dtype=float),
    )

def lowpass_theta_cmd(theta_raw: np.ndarray,
                      theta_prev: np.ndarray,
                      dt: float,
                      tau: float = 0.2,
                      alpha: float = None) -> np.ndarray:
    """First-order low-pass (lag) filter for theta_cmd.
    y_{t} = y_{t-1} + a * (x_t - y_{t-1}), with
      a = 1 - exp(-dt / tau)  if alpha is None,
      else a = clip(alpha, 0, 1).

    Args:
        theta_raw: newly computed command (unfiltered)
        theta_prev: previous filtered command
        dt: time interval [s]
        tau: time constant [s] (ignored if alpha is given)
        alpha: optional direct smoothing factor in [0, 1]

    Returns:
        Filtered theta command.
    """
    # safety
    if theta_prev is None or dt is None:
        return np.asarray(theta_raw, dtype=float)

    if alpha is None:
        tau_safe = float(max(1e-6, tau))
        a = 1.0 - float(np.exp(-float(max(0.0, dt)) / tau_safe))
    else:
        a = float(max(0.0, min(1.0, alpha)))

    theta_raw = np.asarray(theta_raw, dtype=float)
    theta_prev = np.asarray(theta_prev, dtype=float)
    return theta_prev + a * (theta_raw - theta_prev)
