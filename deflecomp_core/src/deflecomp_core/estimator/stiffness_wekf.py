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
        observability_rcond: float = 1e-4,
        observability_abs: float = 1e-10,
        measurement_info_eig_cap: float = 1.0,
        update_gain: float = 1.0,
        max_log_kp_step: float = 0.0,
        min_log_kp_step: float = 0.0,
    ) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.solver = solver
        self.sensitivity = sensitivity
        self.robot = sensitivity.robot
        self.eps_def = float(eps_def)
        self.observability_rcond = float(observability_rcond)
        self.observability_abs = float(observability_abs)
        self.measurement_info_eig_cap = float(measurement_info_eig_cap)
        self.update_gain = float(update_gain)
        self.max_log_kp_step = float(max_log_kp_step)
        self.min_log_kp_step = float(min_log_kp_step)
        self.last_theta_eq: Optional[np.ndarray] = None
        self.last_observable_rank = 0
        self.last_observable_eigvals = np.zeros_like(self.x)
        self.last_information_scale = 1.0
        self.last_update_step = np.zeros_like(self.x)
        self.last_update_norm = 0.0
        self.last_update_applied = False

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
        return u_f, H0_f

    def _observable_subspace(self, information: np.ndarray):
        info = 0.5 * (information + information.T)
        eigvals, eigvecs = np.linalg.eigh(info)
        lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
        threshold = max(self.observability_abs, self.observability_rcond * max(lam_max, 0.0))
        keep = eigvals > threshold
        self.last_observable_rank = int(np.count_nonzero(keep))
        self.last_observable_eigvals = eigvals.copy()
        return eigvals, eigvecs, keep

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

        for fid, A_f in A_map.items():
            u_f, H0_f = self._accumulate_frame_terms(theta_eq, fid, A_f, J_q, J_x)
            u_total += u_f
            H0_total += H0_f

        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total
        g = -(J_x.T @ y)
        information = -0.5 * (H0_total + H0_total.T)
        return g, information, theta_eq

    def update_with_multi(
        self,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        if kp_lim is None:
            kp_lim = (1e-10, 2000.0)
        
        self.predict()
        g, information, theta_eq = self._grad_hess_multi(
            theta_cmd=theta_cmd,
            A_map=A_map,
            theta_init=theta_init_eq_pred,
        )
        eigvals, eigvecs, keep = self._observable_subspace(information)
        if not np.any(keep):
            self.last_theta_eq = theta_eq.copy()
            self.last_information_scale = 0.0
            self.last_update_step = np.zeros_like(self.x)
            self.last_update_norm = 0.0
            self.last_update_applied = False
            return theta_eq

        U_obs = eigvecs[:, keep]
        U_unobs = eigvecs[:, ~keep]
        info_obs = np.maximum(eigvals[keep], 0.0)
        P_obs = U_obs.T @ self.P @ U_obs
        g_obs = U_obs.T @ g

        info_scale = max(0.0, self.update_gain)
        info_cap = self.measurement_info_eig_cap
        if info_cap > 0.0 and info_obs.size:
            info_max = float(np.max(info_obs))
            if info_max > info_cap:
                info_scale *= info_cap / max(info_max, 1e-12)
        if info_scale <= 0.0:
            self.last_theta_eq = theta_eq.copy()
            self.last_information_scale = 0.0
            self.last_update_step = np.zeros_like(self.x)
            self.last_update_norm = 0.0
            self.last_update_applied = False
            return theta_eq

        Sinv_obs = np.diag(info_scale * info_obs)
        g_obs = info_scale * g_obs

        Pinv_obs = np.linalg.pinv(P_obs, rcond=1e-12)
        lam = 1e-6
        P_post_obs = np.linalg.pinv(Pinv_obs + Sinv_obs + lam * np.eye(Sinv_obs.shape[0]), rcond=1e-12)
        dx = U_obs @ (P_post_obs @ g_obs)
        step_scale = 1.0
        max_step = self.max_log_kp_step
        if max_step > 0.0:
            step_norm_inf = float(np.max(np.abs(dx))) if dx.size else 0.0
            if step_norm_inf > max_step:
                step_scale = max_step / max(step_norm_inf, 1e-12)
                dx = step_scale * dx

        step_norm = float(np.linalg.norm(dx))
        min_step = self.min_log_kp_step
        if min_step > 0.0 and step_norm < min_step:
            dx = np.zeros_like(dx)
            step_norm = 0.0
            step_scale = 0.0

        x_post = self.x + dx

        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))
        dx_applied = x_post - self.x

        if U_unobs.size:
            P_unobs = U_unobs @ (U_unobs.T @ self.P @ U_unobs) @ U_unobs.T
        else:
            P_unobs = np.zeros_like(self.P)
        if step_scale < 1.0:
            P_post_obs = P_obs + step_scale * (P_post_obs - P_obs)
        P_post = U_obs @ P_post_obs @ U_obs.T + P_unobs
        self.P = 0.5 * (P_post + P_post.T)
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        self.last_information_scale = info_scale
        self.last_update_step = dx_applied
        self.last_update_norm = float(np.linalg.norm(dx_applied))
        self.last_update_applied = self.last_update_norm > 0.0
        return theta_eq


MultiFrameWeirdEKF = MultiFrameStiffnessWEKF
