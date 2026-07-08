from typing import Dict, Optional, Tuple
import numpy as np
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.observation.bingham import BinghamUtils
from deflecomp_core.robot.pinocchio_robot import RobotArm


class MultiFrameStiffnessWEKF:
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        solver: EquilibriumSolver,
        sensitivity: SensitivityCalculator,
        eps_def: float = 1e-6,
    ) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.solver = solver
        self.sensitivity = sensitivity
        self.robot = sensitivity.robot
        self.eps_def = float(eps_def)
        self.last_theta_eq: Optional[np.ndarray] = None

    def predict(self) -> None:
        self.P = self.P + self.Q

    @property
    def kp_hat(self) -> np.ndarray:
        return np.exp(self.x)

    def _accumulate_frame_terms(
        self,
        theta_eq: np.ndarray,
        fid: int,
        A_f: np.ndarray,
        J_q: np.ndarray,
        J_x: np.ndarray,
    ):
        z_f = self.robot.frame_quaternion_wxyz_base(theta_eq, fid)
        Qz_f = BinghamUtils.qmat_from_quat_wxyz(z_f)
        J_w_f = self.robot.frame_angular_jacobian_world(theta_eq, fid)

        v_f = Qz_f.T @ (A_f @ z_f)
        u_f = J_w_f.T @ v_f

        X = np.linalg.pinv(J_q, rcond=1e-12) @ J_x
        M_f = Qz_f @ (J_w_f @ X)
        H0_f = 0.5 * (M_f.T @ (A_f @ M_f))
        MtM_f = M_f.T @ M_f
        return u_f, H0_f, MtM_f

    def _stabilize_hessian(self, H0_total: np.ndarray, MtM_total: np.ndarray) -> np.ndarray:
        H0s = 0.5 * (H0_total + H0_total.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            return H0s, 0.

        Bs = 0.5 * (MtM_total + MtM_total.T)
        wB = np.linalg.eigvalsh(Bs)
        lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0

        if lam_max_B <= 1e-12:
            return H0s - (lam_max_H + self.eps_def) * np.eye(H0s.shape[0]), 0.

        c = -2.0 * (lam_max_H + self.eps_def) / lam_max_B
        return H0s + 0.5 * c * MtM_total, c

    def _grad_hess_multi(
        self,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init: Optional[np.ndarray],
    ):
        n = self.x.size
        k_diag = self.kp_hat

        theta_eq = self.solver.solve(theta_cmd=theta_cmd, kp_vec=k_diag, theta_init=theta_init)
        J_q, J_x = self.sensitivity.equilibrium_jacobians(theta_eq, theta_cmd, k_diag)

        u_total = np.zeros(n, dtype=float)
        H0_total = np.zeros((n, n), dtype=float)
        MtM_total = np.zeros((n, n), dtype=float)

        for fid, A_f in A_map.items():
            u_f, H0_f, MtM_f = self._accumulate_frame_terms(theta_eq, fid, A_f, J_q, J_x)
            u_total += u_f
            H0_total += H0_f
            MtM_total += MtM_f

        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total
        g = -(J_x.T @ y)
        H, c_bingham = self._stabilize_hessian(H0_total, MtM_total)
        return g, H, theta_eq, c_bingham

    def update_with_multi(
        self,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float]] = None,
    ) -> np.ndarray:
        if kp_lim is None:
            kp_lim = (1e-10, 2000.0)
        
        self.predict()
        g, H, theta_eq, c_bingham = self._grad_hess_multi(
            theta_cmd=theta_cmd,
            A_map=A_map,
            theta_init=theta_init_eq_pred,
        )
        Sinv = -H
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])

        Pinv = np.linalg.pinv(self.P, rcond=1e-12)

        lam = 1e-6
        P_post = np.linalg.pinv(Pinv + Sinv + lam * np.eye(Sinv.shape[0]), rcond=1e-12)
        x_post = self.x + P_post @ g

        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))

        self.P = 0.5 * (P_post + P_post.T)
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq


MultiFrameWeirdEKF = MultiFrameStiffnessWEKF
