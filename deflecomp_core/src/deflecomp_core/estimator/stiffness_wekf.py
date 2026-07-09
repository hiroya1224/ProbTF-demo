from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.observation.bingham import BinghamUtils


@dataclass
class StiffnessUpdateResult:
    theta_eq: np.ndarray
    x_est: np.ndarray
    kp_est: np.ndarray
    P_est: np.ndarray
    gradient: np.ndarray
    information: np.ndarray
    obs_rank: int
    update_applied: bool
    update_skipped_reason: Optional[str]
    debug: Dict[str, Any]


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
        laplace_negative_info_tol: float = 1e-9,
        laplace_jitter: float = 1e-6,
    ) -> None:
        self.x_est = x0.copy()
        self.P_est = P0.copy()
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
        self.laplace_negative_info_tol = float(laplace_negative_info_tol)
        self.laplace_jitter = float(laplace_jitter)
        self.last_theta_eq: Optional[np.ndarray] = None
        self.last_observable_rank = 0
        self.last_observable_eigvals = np.zeros_like(self.x_est)
        self.last_information_scale = 1.0
        self.last_update_step = np.zeros_like(self.x_est)
        self.last_update_norm = 0.0
        self.last_update_applied = False
        self.last_debug: Dict[str, Any] = {}

    @property
    def x(self) -> np.ndarray:
        return self.x_est

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self.x_est = np.asarray(value, dtype=float).copy()

    @property
    def P(self) -> np.ndarray:
        return self.P_est

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self.P_est = np.asarray(value, dtype=float).copy()

    def predict(self) -> None:
        self.P_est = self.P_est + self.Q

    @property
    def kp_hat(self) -> np.ndarray:
        return self.kp_est

    @property
    def kp_est(self) -> np.ndarray:
        return np.exp(self.x_est)

    def _compute_frame_laplace_term(
        self,
        theta_eq: np.ndarray,
        fid: int,
        A_f: np.ndarray,
        theta_x: np.ndarray,
    ) -> Dict[str, Any]:
        A_sym = 0.5 * (A_f + A_f.T)
        z_f = self.robot.frame_quaternion_wxyz_base(theta_eq, fid)
        Qz_f = BinghamUtils.qmat_from_quat_wxyz(z_f)
        J_w_f = self.robot.frame_angular_jacobian_world(theta_eq, fid)

        M_f = Qz_f @ (J_w_f @ theta_x)
        grad_f = -(M_f.T @ (A_sym @ z_f))
        # With dz/dx ~= -0.5 * M_f, Hessian(ell) ~= 0.5 * M_f.T A M_f.
        hess_f = 0.5 * (M_f.T @ (A_sym @ M_f))
        info_f = -0.5 * (hess_f + hess_f.T)
        ell_f = float(z_f.T @ (A_sym @ z_f))
        return {
            "frame_id": fid,
            "log_likelihood": ell_f,
            "gradient": grad_f,
            "information": info_f,
            "z_pred": z_f,
        }

    def _compute_local_laplace_terms(
        self,
        theta_eq: np.ndarray,
        J_q: np.ndarray,
        J_x: np.ndarray,
        A_map: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Approximate the IMU Bingham log-likelihood around current log stiffness.

        The scalar likelihood is ell(x) = sum_f z_f(theta_eq(x)).T A_f z_f(theta_eq(x)).
        Equilibrium sensitivity gives d theta_eq / d x = -pinv(J_q) J_x.  The returned
        information matrix is the local positive-semidefinite approximation
        information = -Hessian(ell).
        """
        n = self.x_est.size
        theta_x = np.linalg.pinv(J_q, rcond=1e-12) @ J_x
        grad = np.zeros(n, dtype=float)
        info = np.zeros((n, n), dtype=float)
        ell = 0.0
        frame_terms: List[Dict[str, Any]] = []

        for fid, A_f in A_map.items():
            term = self._compute_frame_laplace_term(theta_eq, fid, A_f, theta_x)
            grad += term["gradient"]
            info += term["information"]
            ell += float(term["log_likelihood"])
            frame_terms.append(term)

        info = 0.5 * (info + info.T)
        return {
            "log_likelihood": ell,
            "gradient": grad,
            "information": info,
            "frame_terms": frame_terms,
        }

    def _observable_subspace(self, information: np.ndarray):
        info = 0.5 * (information + information.T)
        raw_eigvals, eigvecs = np.linalg.eigh(info)
        eigvals = np.maximum(raw_eigvals, 0.0)
        lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
        threshold = max(self.observability_abs, self.observability_rcond * max(lam_max, 0.0))
        keep = eigvals > threshold
        self.last_observable_rank = int(np.count_nonzero(keep))
        self.last_observable_eigvals = eigvals.copy()
        min_raw_eig = float(np.min(raw_eigvals)) if raw_eigvals.size else 0.0
        return eigvals, eigvecs, keep, raw_eigvals, min_raw_eig

    def _grad_hess_multi(
        self,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init: Optional[np.ndarray],
    ):
        k_diag = self.kp_est

        theta_eq = self.solver.solve(theta_cmd=theta_cmd, kp_vec=k_diag, theta_init=theta_init)
        J_q, J_x = self.sensitivity.equilibrium_jacobians(theta_eq, theta_cmd, k_diag)
        terms = self._compute_local_laplace_terms(theta_eq, J_q, J_x, A_map)
        return terms["gradient"], terms["information"], theta_eq

    def _information_scale(self, info_obs: np.ndarray) -> float:
        del info_obs
        return 1.0

    def _finish_skipped_update(
        self,
        theta_eq: np.ndarray,
        P_pred: np.ndarray,
        debug: Dict[str, Any],
        reason: str,
        gradient: np.ndarray,
        information: np.ndarray,
    ) -> StiffnessUpdateResult:
        debug["laplace_update_skipped_reason"] = reason
        debug["est_update_skipped_reason"] = reason
        debug.setdefault("laplace_information_scale", 0.0)
        debug["laplace_dx_norm"] = 0.0
        debug["laplace_dx_max_abs"] = 0.0
        debug["est_update_applied"] = False
        self.P_est = 0.5 * (P_pred + P_pred.T)
        self.last_theta_eq = theta_eq.copy()
        self.last_information_scale = 0.0
        self.last_update_step = np.zeros_like(self.x_est)
        self.last_update_norm = 0.0
        self.last_update_applied = False
        self.last_debug = debug
        return StiffnessUpdateResult(
            theta_eq=theta_eq.copy(),
            x_est=self.x_est.copy(),
            kp_est=self.kp_est.copy(),
            P_est=self.P_est.copy(),
            gradient=gradient.copy(),
            information=information.copy(),
            obs_rank=self.last_observable_rank,
            update_applied=False,
            update_skipped_reason=reason,
            debug=debug,
        )

    def update_with_multi(
        self,
        theta_cmd_sent: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float, float]] = None,
    ) -> StiffnessUpdateResult:
        if kp_lim is None:
            kp_lim = (1e-10, 2000.0)

        x_pred = self.x_est.copy()
        P_pred = self.P_est + self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        K_pred = np.exp(x_pred)

        theta_eq = self.solver.solve(
            theta_cmd=theta_cmd_sent,
            kp_vec=K_pred,
            theta_init=theta_init_eq_pred,
        )
        J_q, J_x = self.sensitivity.equilibrium_jacobians(theta_eq, theta_cmd_sent, K_pred)
        terms = self._compute_local_laplace_terms(theta_eq, J_q, J_x, A_map)
        g = terms["gradient"]
        information = terms["information"]

        eigvals, eigvecs, keep, raw_eigvals, min_raw_eig = self._observable_subspace(information)
        debug: Dict[str, Any] = {
            "laplace_log_likelihood": float(terms["log_likelihood"]),
            "laplace_grad_norm": float(np.linalg.norm(g)),
            "laplace_info_eigs": eigvals.copy(),
            "laplace_obs_rank": int(np.count_nonzero(keep)),
            "laplace_info_negative_min_eig": min_raw_eig,
            "laplace_info_has_large_negative_eig": bool(min_raw_eig < -self.laplace_negative_info_tol),
            "laplace_raw_info_eigs": raw_eigvals.copy(),
            "est_obs_rank": int(np.count_nonzero(keep)),
            "est_information_eigs": eigvals.copy(),
            "est_gradient_norm": float(np.linalg.norm(g)),
        }
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(information)):
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "nonfinite_laplace_terms",
                g,
                information,
            )
        if not np.any(keep):
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "no_observable_information",
                g,
                information,
            )

        U_obs = eigvecs[:, keep]
        U_unobs = eigvecs[:, ~keep]
        info_obs = eigvals[keep]
        P_obs = U_obs.T @ P_pred @ U_obs
        g_obs = U_obs.T @ g

        info_scale = self._information_scale(info_obs)
        debug["laplace_information_scale"] = float(info_scale)
        if info_scale <= 0.0:
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "nonpositive_information_scale",
                g,
                information,
            )

        Sinv_obs = np.diag(info_scale * info_obs)
        g_obs = info_scale * g_obs

        Pinv_obs = np.linalg.pinv(P_obs, rcond=1e-12)
        jitter = max(0.0, self.laplace_jitter)
        P_post_obs = np.linalg.pinv(
            Pinv_obs + Sinv_obs + jitter * np.eye(Sinv_obs.shape[0]),
            rcond=1e-12,
        )
        dx = U_obs @ (P_post_obs @ g_obs)
        if not np.all(np.isfinite(dx)):
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "nonfinite_posterior_step",
                g,
                information,
            )

        x_post = x_pred + dx

        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))
        dx_applied = x_post - x_pred

        if U_unobs.size:
            P_unobs = U_unobs @ (U_unobs.T @ P_pred @ U_unobs) @ U_unobs.T
        else:
            P_unobs = np.zeros_like(self.P_est)
        P_post = U_obs @ P_post_obs @ U_obs.T + P_unobs
        self.P_est = 0.5 * (P_post + P_post.T)
        self.x_est = x_post
        self.last_theta_eq = theta_eq.copy()
        self.last_information_scale = info_scale
        self.last_update_step = dx_applied
        self.last_update_norm = float(np.linalg.norm(dx_applied))
        self.last_update_applied = self.last_update_norm > 0.0
        debug["laplace_dx_norm"] = self.last_update_norm
        debug["laplace_dx_max_abs"] = float(np.max(np.abs(dx_applied))) if dx_applied.size else 0.0
        debug["laplace_step_scale"] = 1.0
        debug["est_update_applied"] = True
        debug["est_update_skipped_reason"] = None
        self.last_debug = debug
        return StiffnessUpdateResult(
            theta_eq=theta_eq.copy(),
            x_est=self.x_est.copy(),
            kp_est=self.kp_est.copy(),
            P_est=self.P_est.copy(),
            gradient=g.copy(),
            information=information.copy(),
            obs_rank=self.last_observable_rank,
            update_applied=True,
            update_skipped_reason=None,
            debug=debug,
        )


MultiFrameWeirdEKF = MultiFrameStiffnessWEKF
