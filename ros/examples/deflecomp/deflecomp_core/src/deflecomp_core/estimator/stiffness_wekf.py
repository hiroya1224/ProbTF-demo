from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.observation.bingham import BinghamUtils
from deflecomp_core.robot.pinocchio_robot import RobotArm


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


@dataclass
class StiffnessLikelihoodEvaluation:
    log_likelihood: float
    theta_eq: np.ndarray
    kp_vec: np.ndarray
    x_eval: np.ndarray
    valid: bool
    error: Optional[str]


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
        laplace_negative_info_tol: float = 1e-9,
        laplace_jitter: float = 1e-6,
        laplace_backtracking_max_steps: int = 16,
        laplace_backtracking_factor: float = 0.5,
        laplace_acceptance_tol: float = 1e-10,
        max_log_kp_update_step: float = 0.25,
        max_equilibrium_pose_jump: float = 0.10,
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
        self.laplace_negative_info_tol = float(laplace_negative_info_tol)
        self.laplace_jitter = float(laplace_jitter)
        self.laplace_backtracking_max_steps = max(1, int(laplace_backtracking_max_steps))
        self.laplace_backtracking_factor = float(laplace_backtracking_factor)
        if not 0.0 < self.laplace_backtracking_factor < 1.0:
            raise ValueError("laplace_backtracking_factor must be strictly between 0 and 1")
        self.laplace_acceptance_tol = max(0.0, float(laplace_acceptance_tol))
        self.max_log_kp_update_step = float(max_log_kp_update_step)
        if self.max_log_kp_update_step <= 0.0 or np.isnan(self.max_log_kp_update_step):
            raise ValueError("max_log_kp_update_step must be positive")
        self.max_equilibrium_pose_jump = float(max_equilibrium_pose_jump)
        if self.max_equilibrium_pose_jump <= 0.0 or np.isnan(self.max_equilibrium_pose_jump):
            raise ValueError("max_equilibrium_pose_jump must be positive")
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

    def clone_for_evaluation(self) -> "MultiFrameStiffnessWEKF":
        robot = RobotArm(
            self.robot.urdf_path,
            tip_link=self.robot.tip_link_name,
            base_link=self.robot.base_link_name,
        )
        spring_model = deepcopy(self.solver.spring_model)
        solver = EquilibriumSolver(
            robot=robot,
            spring_model=spring_model,
            cfg=deepcopy(self.solver.cfg),
        )
        sensitivity = SensitivityCalculator(
            robot=robot,
            spring_model=spring_model,
            active_set_tol=float(getattr(self.sensitivity, "active_set_tol", 1.0e-8)),
            kkt_tol=float(getattr(self.sensitivity, "kkt_tol", 1.0e-8)),
        )
        clone = MultiFrameStiffnessWEKF(
            x0=self.x_est.copy(),
            P0=self.P_est.copy(),
            Q=self.Q.copy(),
            solver=solver,
            sensitivity=sensitivity,
            eps_def=self.eps_def,
            observability_rcond=self.observability_rcond,
            observability_abs=self.observability_abs,
            laplace_negative_info_tol=self.laplace_negative_info_tol,
            laplace_jitter=self.laplace_jitter,
            laplace_backtracking_max_steps=self.laplace_backtracking_max_steps,
            laplace_backtracking_factor=self.laplace_backtracking_factor,
            laplace_acceptance_tol=self.laplace_acceptance_tol,
            max_log_kp_update_step=self.max_log_kp_update_step,
            max_equilibrium_pose_jump=self.max_equilibrium_pose_jump,
        )
        clone.last_theta_eq = None if self.last_theta_eq is None else self.last_theta_eq.copy()
        clone.last_observable_rank = int(self.last_observable_rank)
        clone.last_observable_eigvals = self.last_observable_eigvals.copy()
        clone.last_information_scale = float(self.last_information_scale)
        clone.last_update_step = self.last_update_step.copy()
        clone.last_update_norm = float(self.last_update_norm)
        clone.last_update_applied = bool(self.last_update_applied)
        clone.last_debug = deepcopy(self.last_debug)
        return clone

    def evaluate_log_likelihood_at_x(
        self,
        x_eval: np.ndarray,
        theta_cmd_sent: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float, float]] = None,
    ) -> StiffnessLikelihoodEvaluation:
        try:
            x = np.asarray(x_eval, dtype=float).copy()
            if x.shape != self.x_est.shape:
                return StiffnessLikelihoodEvaluation(
                    log_likelihood=-np.inf,
                    theta_eq=np.array([], dtype=float),
                    kp_vec=np.array([], dtype=float),
                    x_eval=x,
                    valid=False,
                    error=f"x_eval shape {x.shape} does not match estimator shape {self.x_est.shape}",
                )
            if not np.all(np.isfinite(x)):
                return StiffnessLikelihoodEvaluation(
                    log_likelihood=-np.inf,
                    theta_eq=np.array([], dtype=float),
                    kp_vec=np.array([], dtype=float),
                    x_eval=x,
                    valid=False,
                    error="nonfinite_x_eval",
                )

            if kp_lim is not None:
                kp_min, kp_max = (float(v) for v in kp_lim)
                if kp_min <= 0.0 or kp_max < kp_min:
                    return StiffnessLikelihoodEvaluation(
                        log_likelihood=-np.inf,
                        theta_eq=np.array([], dtype=float),
                        kp_vec=np.array([], dtype=float),
                        x_eval=x,
                        valid=False,
                        error=f"invalid_kp_lim: {kp_lim}",
                    )
                x = np.clip(x, np.log(kp_min), np.log(kp_max))

            kp_vec = np.exp(x)
            if not np.all(np.isfinite(kp_vec)):
                return StiffnessLikelihoodEvaluation(
                    log_likelihood=-np.inf,
                    theta_eq=np.array([], dtype=float),
                    kp_vec=kp_vec,
                    x_eval=x,
                    valid=False,
                    error="nonfinite_kp_vec",
                )

            theta_eq = self.solver.solve(
                theta_cmd=theta_cmd_sent,
                kp_vec=kp_vec,
                theta_init=theta_init_eq_pred,
            )
            theta_eq = np.asarray(theta_eq, dtype=float)
            if not np.all(np.isfinite(theta_eq)):
                return StiffnessLikelihoodEvaluation(
                    log_likelihood=-np.inf,
                    theta_eq=theta_eq.copy(),
                    kp_vec=kp_vec,
                    x_eval=x,
                    valid=False,
                    error="nonfinite_theta_eq",
                )

            log_likelihood = 0.0
            for fid, A_f in A_map.items():
                A_arr = np.asarray(A_f, dtype=float)
                A_sym = 0.5 * (A_arr + A_arr.T)
                z_f = np.asarray(self.robot.frame_quaternion_wxyz_base(theta_eq, fid), dtype=float)
                ell_f = float(z_f.T @ (A_sym @ z_f))
                if not np.isfinite(ell_f):
                    return StiffnessLikelihoodEvaluation(
                        log_likelihood=-np.inf,
                        theta_eq=theta_eq.copy(),
                        kp_vec=kp_vec,
                        x_eval=x,
                        valid=False,
                        error=f"nonfinite_frame_likelihood:{fid}",
                    )
                log_likelihood += ell_f

            if not np.isfinite(log_likelihood):
                return StiffnessLikelihoodEvaluation(
                    log_likelihood=-np.inf,
                    theta_eq=theta_eq.copy(),
                    kp_vec=kp_vec,
                    x_eval=x,
                    valid=False,
                    error="nonfinite_log_likelihood",
                )

            return StiffnessLikelihoodEvaluation(
                log_likelihood=float(log_likelihood),
                theta_eq=theta_eq.copy(),
                kp_vec=kp_vec.copy(),
                x_eval=x.copy(),
                valid=True,
                error=None,
            )
        except Exception as exc:
            return StiffnessLikelihoodEvaluation(
                log_likelihood=-np.inf,
                theta_eq=np.array([], dtype=float),
                kp_vec=np.array([], dtype=float),
                x_eval=np.asarray(x_eval, dtype=float).copy(),
                valid=False,
                error=str(exc),
            )

    def apply_particle_correction(
        self,
        x_new: np.ndarray,
        active_indices: np.ndarray,
        reset_std: float,
        pursuit_mixture_weight: float = 1.0,
        theta_eq: Optional[np.ndarray] = None,
        kp_lim: Optional[Tuple[float, float]] = None,
    ) -> None:
        x_old = self.x_est.copy()
        x = np.asarray(x_new, dtype=float).copy()
        if x.shape != self.x_est.shape:
            raise ValueError(f"x_new shape {x.shape} does not match estimator shape {self.x_est.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x_new contains nonfinite values")

        if kp_lim is not None:
            kp_min, kp_max = (float(v) for v in kp_lim)
            if kp_min <= 0.0 or kp_max < kp_min:
                raise ValueError(f"invalid kp_lim: {kp_lim}")
            x = np.clip(x, np.log(kp_min), np.log(kp_max))

        active = np.unique(np.asarray(active_indices, dtype=int))
        active = active[(0 <= active) & (active < self.x_est.size)]
        x_pursuit = x_old.copy()
        x_pursuit[active] = x[active]
        weight = float(np.clip(float(pursuit_mixture_weight), 0.0, 1.0))

        P_zealot = 0.5 * (self.P_est + self.P_est.T)
        P_pursuit = P_zealot.copy()
        reset_var = float(reset_std) ** 2
        for j in active:
            P_pursuit[int(j), int(j)] = max(float(P_pursuit[int(j), int(j)]), reset_var)

        x_mix_raw = (1.0 - weight) * x_old + weight * x_pursuit
        dz = x_old - x_mix_raw
        dp = x_pursuit - x_mix_raw
        P_mix = (
            (1.0 - weight) * (P_zealot + np.outer(dz, dz))
            + weight * (P_pursuit + np.outer(dp, dp))
        )
        P_mix = 0.5 * (P_mix + P_mix.T)
        if not np.all(np.isfinite(P_mix)):
            P_mix = P_zealot

        self.x_est = x_mix_raw
        if kp_lim is not None:
            self.x_est = np.clip(self.x_est, np.log(kp_min), np.log(kp_max))
        self.P_est = P_mix

        if theta_eq is not None and weight >= 1.0:
            self.last_theta_eq = np.asarray(theta_eq, dtype=float).copy()

        dx = self.x_est - x_old
        self.last_update_step = dx.copy()
        self.last_update_norm = float(np.linalg.norm(dx))
        self.last_update_applied = True
        debug = dict(self.last_debug)
        debug["particle_correction_applied"] = True
        debug["particle_correction_active_indices"] = active.copy()
        debug["particle_correction_reset_std"] = float(reset_std)
        debug["particle_correction_pursuit_mixture_weight"] = float(weight)
        debug["particle_correction_x_zealot"] = x_old.copy()
        debug["particle_correction_x_pursuit"] = x_pursuit.copy()
        debug["particle_correction_dx_norm"] = self.last_update_norm
        debug["particle_correction_dx_max_abs"] = float(np.max(np.abs(dx))) if dx.size else 0.0
        debug["est_update_applied"] = True
        debug["est_update_skipped_reason"] = None
        self.last_debug = debug

    def _compute_frame_laplace_term(
        self,
        theta_eq: np.ndarray,
        fid: int,
        A_f: np.ndarray,
        theta_x: np.ndarray,
    ) -> Dict[str, Any]:
        A_sym = 0.5 * (A_f + A_f.T)
        z_f = self.robot.frame_quaternion_wxyz_base(theta_eq, fid)
        # z_f represents R_base,frame.  Its spatial tangent must be paired
        # with the relative angular Jacobian expressed in base coordinates.
        # The former implementation paired a body/local tangent (left
        # quaternion multiplication) with a WORLD Jacobian; that coordinate
        # mismatch corrupted both the stiffness gradient and information.
        Qz_f = BinghamUtils.spatial_qmat_from_quat_wxyz(z_f)
        J_b_f = self.robot.frame_angular_jacobian_base(theta_eq, fid)

        M_f = Qz_f @ (J_b_f @ theta_x)
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

    @staticmethod
    def _symmetric_psd_sqrt(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the symmetric square root of a covariance and its clipped spectrum."""
        array = np.asarray(matrix, dtype=float)
        symmetric = 0.5 * (array + array.T)
        eigvals, eigvecs = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigvals, 0.0)
        sqrt_matrix = (eigvecs * np.sqrt(clipped)) @ eigvecs.T
        return 0.5 * (sqrt_matrix + sqrt_matrix.T), clipped

    def _prior_whitened_observable_subspace(
        self,
        information: np.ndarray,
        covariance: np.ndarray,
    ):
        """Find observable directions in dimensionless prior-whitened coordinates.

        Applying the relative cutoff directly to ``information`` makes the largest
        mode of every individual sample suppress all weaker modes forever.  With
        ``delta_x = sqrt(P) delta_y``, the relevant dimensionless information is
        ``sqrt(P) information sqrt(P)``.  Modes already learned by earlier samples
        then lose weight while independent weak modes retain their prior variance
        and become observable on subsequent samples.
        """
        covariance_sqrt, covariance_eigvals = self._symmetric_psd_sqrt(covariance)
        info = 0.5 * (information + information.T)
        whitened = covariance_sqrt @ info @ covariance_sqrt
        whitened = 0.5 * (whitened + whitened.T)
        raw_eigvals, eigvecs = np.linalg.eigh(whitened)
        eigvals = np.maximum(raw_eigvals, 0.0)
        lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
        threshold = max(self.observability_abs, self.observability_rcond * max(lam_max, 0.0))
        keep = eigvals > threshold
        self.last_observable_rank = int(np.count_nonzero(keep))
        self.last_observable_eigvals = eigvals.copy()
        min_raw_eig = float(np.min(raw_eigvals)) if raw_eigvals.size else 0.0
        return (
            eigvals,
            eigvecs,
            keep,
            raw_eigvals,
            min_raw_eig,
            covariance_sqrt,
            covariance_eigvals,
            whitened,
            threshold,
        )

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
        debug["laplace_information_scale"] = 0.0
        debug["laplace_step_scale"] = 0.0
        debug["laplace_step_acceptance_reason"] = reason
        debug.setdefault(
            "laplace_log_likelihood_before",
            debug.get("laplace_log_likelihood", -np.inf),
        )
        debug.setdefault("laplace_log_likelihood_after", debug["laplace_log_likelihood_before"])
        debug.setdefault("laplace_posterior_objective_before", debug["laplace_log_likelihood_before"])
        debug.setdefault("laplace_posterior_objective_after", debug["laplace_posterior_objective_before"])
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
        log_likelihood_before = float(terms["log_likelihood"])
        debug: Dict[str, Any] = {
            "laplace_log_likelihood": log_likelihood_before,
            "laplace_log_likelihood_before": log_likelihood_before,
            "laplace_log_likelihood_after": log_likelihood_before,
            "laplace_posterior_objective_before": log_likelihood_before,
            "laplace_posterior_objective_after": log_likelihood_before,
            "laplace_grad_norm": float(np.linalg.norm(g)),
            "est_gradient_norm": float(np.linalg.norm(g)),
            "laplace_step_scale": 0.0,
            "laplace_step_acceptance_reason": "not_evaluated",
            "laplace_backtracking_trials": [],
            "laplace_max_log_kp_update_step": self.max_log_kp_update_step,
            "laplace_max_equilibrium_pose_jump": self.max_equilibrium_pose_jump,
        }
        if (
            not np.all(np.isfinite(g))
            or not np.all(np.isfinite(information))
            or not np.isfinite(log_likelihood_before)
        ):
            self.last_observable_rank = 0
            self.last_observable_eigvals = np.zeros_like(self.x_est)
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "nonfinite_laplace_terms",
                g,
                information,
            )

        raw_information_eigvals = np.linalg.eigvalsh(0.5 * (information + information.T))
        raw_information_eigvals_clipped = np.maximum(raw_information_eigvals, 0.0)
        min_raw_information_eig = (
            float(np.min(raw_information_eigvals)) if raw_information_eigvals.size else 0.0
        )
        (
            eigvals,
            eigvecs,
            keep,
            whitened_raw_eigvals,
            min_whitened_raw_eig,
            covariance_sqrt,
            covariance_eigvals,
            _whitened_information,
            observability_threshold,
        ) = self._prior_whitened_observable_subspace(information, P_pred)
        debug.update(
            {
                # Retain the historical raw-information fields for consumers
                # which plot them, and expose the dimensionless spectrum used
                # for the actual observability decision separately.
                "laplace_info_eigs": raw_information_eigvals_clipped.copy(),
                "laplace_raw_info_eigs": raw_information_eigvals.copy(),
                "laplace_info_negative_min_eig": min_raw_information_eig,
                "laplace_info_has_large_negative_eig": bool(
                    min_raw_information_eig < -self.laplace_negative_info_tol
                ),
                "laplace_prior_whitened_info_eigs": eigvals.copy(),
                "laplace_prior_whitened_raw_info_eigs": whitened_raw_eigvals.copy(),
                "laplace_prior_whitened_info_negative_min_eig": min_whitened_raw_eig,
                "laplace_prior_covariance_eigs": covariance_eigvals.copy(),
                "laplace_prior_whitened_observability_threshold": float(observability_threshold),
                "laplace_obs_rank": int(np.count_nonzero(keep)),
                "est_obs_rank": int(np.count_nonzero(keep)),
                "est_information_eigs": eigvals.copy(),
            }
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
        info_obs = eigvals[keep]
        g_whitened = covariance_sqrt @ g
        g_obs = U_obs.T @ g_whitened

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

        # Backtrack a tempered local-Laplace update.  The local Hessian is only
        # a tangent approximation because every candidate stiffness requires a
        # new nonlinear equilibrium solve.  A candidate is therefore committed
        # only after the exact Bingham likelihood and the Gaussian-prior
        # posterior objective have both been checked.
        jitter = max(0.0, self.laplace_jitter)
        prior_precision = np.linalg.pinv(P_pred, rcond=1e-12)
        accepted_scale: Optional[float] = None
        accepted_eval: Optional[StiffnessLikelihoodEvaluation] = None
        accepted_objective = log_likelihood_before
        accepted_covariance_whitened: Optional[np.ndarray] = None
        accepted_pose_jump_norm = 0.0
        accepted_pose_jump_max_abs = 0.0
        trial_debug: List[Dict[str, Any]] = []
        lower_log_kp = float(np.log(kp_lim[0]))
        upper_log_kp = float(np.log(kp_lim[1]))

        for trial_index in range(self.laplace_backtracking_max_steps):
            step_scale = self.laplace_backtracking_factor ** trial_index
            tempered_precision = step_scale * (info_scale * info_obs + jitter)
            posterior_variance_obs = 1.0 / (1.0 + tempered_precision)
            delta_y_obs = posterior_variance_obs * (step_scale * info_scale * g_obs)
            delta_x = covariance_sqrt @ (U_obs @ delta_y_obs)
            if not np.all(np.isfinite(delta_x)):
                trial_debug.append(
                    {
                        "scale": float(step_scale),
                        "valid": False,
                        "accepted": False,
                        "reason": "nonfinite_posterior_step",
                    }
                )
                continue

            x_candidate = np.clip(x_pred + delta_x, lower_log_kp, upper_log_kp)
            delta_x_applied = x_candidate - x_pred
            delta_x_norm = float(np.linalg.norm(delta_x_applied))
            delta_x_max_abs = (
                float(np.max(np.abs(delta_x_applied))) if delta_x_applied.size else 0.0
            )
            if delta_x_max_abs > self.max_log_kp_update_step + 1.0e-12:
                # Do not project a large local step onto the trust-region
                # boundary.  Continuing the same tempering sequence preserves
                # its direction and covariance interpretation while finding a
                # genuinely local candidate.
                trial_debug.append(
                    {
                        "scale": float(step_scale),
                        "valid": True,
                        "exact_evaluated": False,
                        "accepted": False,
                        "dx_norm": delta_x_norm,
                        "dx_max_abs": delta_x_max_abs,
                        "reason": "log_kp_trust_region_exceeded",
                    }
                )
                continue

            candidate_eval = self.evaluate_log_likelihood_at_x(
                x_eval=x_candidate,
                theta_cmd_sent=theta_cmd_sent,
                A_map=A_map,
                # Keeping the current solution as the initial point makes the
                # acceptance test follow the same equilibrium branch.
                theta_init_eq_pred=theta_eq,
                kp_lim=kp_lim,
            )
            if not candidate_eval.valid:
                trial_debug.append(
                    {
                        "scale": float(step_scale),
                        "valid": False,
                        "exact_evaluated": True,
                        "accepted": False,
                        "dx_norm": delta_x_norm,
                        "dx_max_abs": delta_x_max_abs,
                        "reason": candidate_eval.error or "invalid_exact_likelihood",
                    }
                )
                continue

            pose_delta = np.asarray(candidate_eval.theta_eq, dtype=float) - np.asarray(
                theta_eq,
                dtype=float,
            )
            pose_jump_norm = float(np.linalg.norm(pose_delta))
            pose_jump_max_abs = float(np.max(np.abs(pose_delta))) if pose_delta.size else 0.0
            if (
                not np.all(np.isfinite(pose_delta))
                or pose_jump_norm > self.max_equilibrium_pose_jump + 1.0e-12
            ):
                trial_debug.append(
                    {
                        "scale": float(step_scale),
                        "valid": bool(np.all(np.isfinite(pose_delta))),
                        "exact_evaluated": True,
                        "accepted": False,
                        "log_likelihood": float(candidate_eval.log_likelihood),
                        "dx_norm": delta_x_norm,
                        "dx_max_abs": delta_x_max_abs,
                        "equilibrium_pose_jump_norm": pose_jump_norm,
                        "equilibrium_pose_jump_max_abs": pose_jump_max_abs,
                        "reason": (
                            "equilibrium_pose_jump_exceeded"
                            if np.all(np.isfinite(pose_delta))
                            else "nonfinite_equilibrium_pose_jump"
                        ),
                    }
                )
                continue

            prior_cost = 0.5 * float(delta_x_applied.T @ (prior_precision @ delta_x_applied))
            candidate_objective = float(candidate_eval.log_likelihood - prior_cost)
            comparison_tol = self.laplace_acceptance_tol * max(
                1.0,
                abs(log_likelihood_before),
                abs(float(candidate_eval.log_likelihood)),
            )
            likelihood_nonworsening = bool(
                candidate_eval.log_likelihood + comparison_tol >= log_likelihood_before
            )
            posterior_nonworsening = bool(
                candidate_objective + comparison_tol >= log_likelihood_before
            )
            accepted = likelihood_nonworsening and posterior_nonworsening
            trial_debug.append(
                {
                    "scale": float(step_scale),
                    "valid": True,
                    "exact_evaluated": True,
                    "accepted": bool(accepted),
                    "log_likelihood": float(candidate_eval.log_likelihood),
                    "posterior_objective": candidate_objective,
                    "prior_cost": prior_cost,
                    "dx_norm": delta_x_norm,
                    "dx_max_abs": delta_x_max_abs,
                    "equilibrium_pose_jump_norm": pose_jump_norm,
                    "equilibrium_pose_jump_max_abs": pose_jump_max_abs,
                    "reason": (
                        "accepted"
                        if accepted
                        else (
                            "likelihood_worsened"
                            if not likelihood_nonworsening
                            else "posterior_objective_worsened"
                        )
                    ),
                }
            )
            if accepted:
                accepted_scale = float(step_scale)
                accepted_eval = candidate_eval
                accepted_objective = candidate_objective
                accepted_pose_jump_norm = pose_jump_norm
                accepted_pose_jump_max_abs = pose_jump_max_abs
                accepted_covariance_whitened = np.eye(self.x_est.size, dtype=float)
                accepted_covariance_whitened += U_obs @ (
                    np.diag(posterior_variance_obs - 1.0)
                ) @ U_obs.T
                break

        debug["laplace_backtracking_trials"] = trial_debug
        if accepted_eval is None or accepted_scale is None or accepted_covariance_whitened is None:
            debug["laplace_step_acceptance_reason"] = "exact_posterior_not_improved"
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "exact_posterior_not_improved",
                g,
                information,
            )

        x_post = accepted_eval.x_eval.copy()
        dx_applied = x_post - x_pred
        if not np.all(np.isfinite(dx_applied)):
            return self._finish_skipped_update(
                theta_eq,
                P_pred,
                debug,
                "nonfinite_posterior_step",
                g,
                information,
            )

        P_post = covariance_sqrt @ accepted_covariance_whitened @ covariance_sqrt
        self.P_est = 0.5 * (P_post + P_post.T)
        self.x_est = x_post
        theta_eq_post = accepted_eval.theta_eq.copy()
        self.last_theta_eq = theta_eq_post.copy()
        self.last_information_scale = info_scale * accepted_scale
        self.last_update_step = dx_applied
        self.last_update_norm = float(np.linalg.norm(dx_applied))
        self.last_update_applied = self.last_update_norm > 0.0
        debug["laplace_dx_norm"] = self.last_update_norm
        debug["laplace_dx_max_abs"] = float(np.max(np.abs(dx_applied))) if dx_applied.size else 0.0
        debug["laplace_step_scale"] = accepted_scale
        debug["laplace_step_acceptance_reason"] = (
            "full_step_exact_posterior_improved"
            if accepted_scale == 1.0
            else "backtracked_exact_posterior_improved"
        )
        debug["laplace_log_likelihood_after"] = float(accepted_eval.log_likelihood)
        debug["laplace_posterior_objective_after"] = float(accepted_objective)
        debug["laplace_equilibrium_pose_jump_norm"] = accepted_pose_jump_norm
        debug["laplace_equilibrium_pose_jump_max_abs"] = accepted_pose_jump_max_abs
        debug["est_update_applied"] = True
        debug["est_update_skipped_reason"] = None
        self.last_debug = debug
        return StiffnessUpdateResult(
            theta_eq=theta_eq_post,
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
