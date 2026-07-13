from typing import TYPE_CHECKING, Tuple

import numpy as np

from deflecomp_core.model.spring import SpringModel

if TYPE_CHECKING:
    from deflecomp_core.robot.pinocchio_robot import RobotArm


class SensitivityCalculator:
    def __init__(
        self,
        robot: "RobotArm",
        spring_model: SpringModel,
        active_set_tol: float = 1.0e-8,
        kkt_tol: float = 1.0e-8,
    ) -> None:
        self.robot = robot
        self.spring_model = spring_model
        self.active_set_tol = float(active_set_tol)
        self.kkt_tol = float(kkt_tol)
        self.last_active_set = np.zeros(self.robot.nv, dtype=bool)

    def equilibrium_active_set(
        self,
        theta_eq: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        """Return joints whose box constraint is active at equilibrium.

        For a lower-bound joint the potential gradient must be non-negative;
        for an upper-bound joint it must be non-positive.  Testing both bound
        proximity and this KKT sign avoids freezing a joint merely because an
        inaccurate iterate happens to be close to a limit.
        """
        theta = np.asarray(theta_eq, dtype=float)
        lower = np.asarray(self.robot.model.lowerPositionLimit, dtype=float)
        upper = np.asarray(self.robot.model.upperPositionLimit, dtype=float)
        if lower.shape != theta.shape or upper.shape != theta.shape:
            return np.zeros(theta.shape, dtype=bool)

        gradient = self.robot.tau_gravity(theta) + self.spring_model.torque(
            theta,
            theta_cmd,
            kp_vec,
        )
        at_lower = np.isfinite(lower) & (theta <= lower + self.active_set_tol)
        at_upper = np.isfinite(upper) & (theta >= upper - self.active_set_tol)
        lower_active = at_lower & (gradient >= -self.kkt_tol)
        upper_active = at_upper & (gradient <= self.kkt_tol)
        return lower_active | upper_active

    def equilibrium_jacobians(
        self,
        theta_eq: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_tau_g = self.robot.d_tau_gravity(theta_eq)
        j_q = d_tau_g + np.diag(self.spring_model.stiffness_diag(theta_eq, theta_cmd, kp_vec))
        j_x = np.diag(self.spring_model.log_stiffness_jacobian_diag(theta_eq, theta_cmd, kp_vec))
        active = self.equilibrium_active_set(theta_eq, theta_cmd, kp_vec)
        self.last_active_set = active.copy()

        # Differentiate the box-constrained KKT system on its current active
        # set.  An active joint has d(theta_i)/dx = 0; free rows retain the
        # ordinary torque-equilibrium derivative.  Both the active rows and
        # columns must be removed before inserting the identity constraint.
        # Clearing only the rows works for a nonsingular free block, but a
        # Moore--Penrose solve of a singular block can otherwise use an active
        # joint through the remaining free-row coupling and return a nonzero
        # d(theta_active)/dx.
        active_indices = np.flatnonzero(active)
        if active_indices.size:
            j_q[:, active_indices] = 0.0
            j_q[active_indices, :] = 0.0
            j_q[active_indices, active_indices] = 1.0
            j_x[active_indices, :] = 0.0
        return j_q, j_x
