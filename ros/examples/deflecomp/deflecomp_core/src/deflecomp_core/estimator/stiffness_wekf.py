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
        laplace_outer_iterations: int = 1,
        max_log_kp_update_step: float = 0.25,
        max_equilibrium_pose_jump: float = 0.10,
        joint_limit_reaction_torque_tol: float = 1.0e-3,
        max_log_kp_covariance_var: float = np.inf,
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
        self.laplace_outer_iterations = max(1, int(laplace_outer_iterations))
        self.max_log_kp_update_step = float(max_log_kp_update_step)
        if self.max_log_kp_update_step <= 0.0 or np.isnan(self.max_log_kp_update_step):
            raise ValueError("max_log_kp_update_step must be positive")
        self.max_equilibrium_pose_jump = float(max_equilibrium_pose_jump)
        if self.max_equilibrium_pose_jump <= 0.0 or np.isnan(self.max_equilibrium_pose_jump):
            raise ValueError("max_equilibrium_pose_jump must be positive")
        self.joint_limit_reaction_torque_tol = max(
            0.0,
            float(joint_limit_reaction_torque_tol),
        )
        self.max_log_kp_covariance_var = float(max_log_kp_covariance_var)
        if (
            self.max_log_kp_covariance_var <= 0.0
            or np.isnan(self.max_log_kp_covariance_var)
        ):
            raise ValueError("max_log_kp_covariance_var must be positive")
        self.P_est = self._bounded_covariance(self.P_est)[0]
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
        self.P_est = self._bounded_covariance(self.P_est + self.Q)[0]

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
            laplace_outer_iterations=self.laplace_outer_iterations,
            max_log_kp_update_step=self.max_log_kp_update_step,
            max_equilibrium_pose_jump=self.max_equilibrium_pose_jump,
            joint_limit_reaction_torque_tol=self.joint_limit_reaction_torque_tol,
            max_log_kp_covariance_var=self.max_log_kp_covariance_var,
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
        self.P_est = self._bounded_covariance(P_mix)[0]

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

    def _bounded_covariance(
        self,
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a log-K covariance to PSD with a bounded eigen-spectrum.

        Large process variance is intentional for payload change-points, but
        adding it on every static batch would otherwise make unobservable modes
        grow without bound.  This ceiling limits uncertainty, not the K_est
        mean update.
        """
        array = np.asarray(matrix, dtype=float)
        symmetric = 0.5 * (array + array.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        bounded = np.maximum(eigenvalues, 0.0)
        if np.isfinite(self.max_log_kp_covariance_var):
            bounded = np.minimum(bounded, self.max_log_kp_covariance_var)
        covariance = (eigenvectors * bounded) @ eigenvectors.T
        return 0.5 * (covariance + covariance.T), eigenvalues, bounded

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

    def _equilibrium_active_set_or_empty(
        self,
        theta_eq: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        """Return the constrained equilibrium active set when it is available."""
        active_set_fn = getattr(self.sensitivity, "equilibrium_active_set", None)
        if active_set_fn is None:
            return np.zeros(np.asarray(theta_eq).shape, dtype=bool)
        try:
            active = np.asarray(active_set_fn(theta_eq, theta_cmd, kp_vec), dtype=bool)
        except Exception:
            return np.zeros(np.asarray(theta_eq).shape, dtype=bool)
        if active.shape != np.asarray(theta_eq).shape:
            return np.zeros(np.asarray(theta_eq).shape, dtype=bool)
        return active

    def _joint_limit_reaction_magnitudes_or_zero(
        self,
        theta_eq: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        """Return absolute KKT reaction torques for a constrained solution."""
        theta = np.asarray(theta_eq, dtype=float)
        spring_model = getattr(self.sensitivity, "spring_model", None)
        if spring_model is None:
            return np.zeros(theta.shape, dtype=float)
        try:
            gradient = self.robot.tau_gravity(theta) + spring_model.torque(
                theta,
                theta_cmd,
                kp_vec,
            )
        except Exception:
            return np.zeros(theta.shape, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != theta.shape or not np.all(np.isfinite(gradient)):
            return np.zeros(theta.shape, dtype=float)
        return np.abs(gradient)

    def update_with_multi(
        self,
        theta_cmd_sent: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float, float]] = None,
    ) -> StiffnessUpdateResult:
        """Assimilate one synchronized IMU batch with an iterated Laplace/MAP step.

        The prior center and covariance are frozen for the entire batch.  Each
        outer iteration re-solves equilibrium and relinearizes the *same*
        observation; it does not add process noise or count that observation a
        second time.  Covariance is formed once, from the final linearization.
        """
        if kp_lim is None:
            kp_lim = (1e-10, 2000.0)
        kp_min, kp_max = (float(value) for value in kp_lim)
        if kp_min <= 0.0 or kp_max < kp_min:
            raise ValueError(f"invalid kp_lim: {kp_lim}")

        x_prior = self.x_est.copy()
        P_prior_uncapped = 0.5 * (
            self.P_est + self.Q + (self.P_est + self.Q).T
        )
        P_prior, prior_covariance_eigvals_raw, prior_covariance_eigvals_bounded = (
            self._bounded_covariance(P_prior_uncapped)
        )
        prior_precision = np.linalg.pinv(P_prior, rcond=1e-12)
        covariance_sqrt, covariance_eigvals = self._symmetric_psd_sqrt(P_prior)
        covariance_sqrt_pinv = np.linalg.pinv(covariance_sqrt, rcond=1e-12)
        lower_log_kp = float(np.log(kp_min))
        upper_log_kp = float(np.log(kp_max))
        jitter = max(0.0, self.laplace_jitter)

        x_iter = x_prior.copy()
        theta_iter = np.asarray(
            self.solver.solve(
                theta_cmd=theta_cmd_sent,
                kp_vec=np.exp(x_iter),
                theta_init=theta_init_eq_pred,
            ),
            dtype=float,
        )
        theta_prior = theta_iter.copy()
        batch_initial_active = self._equilibrium_active_set_or_empty(
            theta_prior,
            theta_cmd_sent,
            np.exp(x_prior),
        )

        def linearize(x_value: np.ndarray, theta_value: np.ndarray) -> Dict[str, Any]:
            kp_value = np.exp(x_value)
            J_q, J_x = self.sensitivity.equilibrium_jacobians(
                theta_value,
                theta_cmd_sent,
                kp_value,
            )
            local_terms = self._compute_local_laplace_terms(
                theta_value,
                J_q,
                J_x,
                A_map,
            )
            information_value = np.asarray(local_terms["information"], dtype=float)
            if np.all(np.isfinite(information_value)):
                spectrum = self._prior_whitened_observable_subspace(
                    information_value,
                    P_prior,
                )
            else:
                dimension = self.x_est.size
                spectrum = (
                    np.zeros(dimension, dtype=float),
                    np.eye(dimension, dtype=float),
                    np.zeros(dimension, dtype=bool),
                    np.zeros(dimension, dtype=float),
                    0.0,
                    covariance_sqrt,
                    covariance_eigvals,
                    np.zeros((dimension, dimension), dtype=float),
                    self.observability_abs,
                )
            return {
                "terms": local_terms,
                "gradient": np.asarray(local_terms["gradient"], dtype=float),
                "information": information_value,
                "eigvals": spectrum[0],
                "eigvecs": spectrum[1],
                "keep": spectrum[2],
                "whitened_raw_eigvals": spectrum[3],
                "min_whitened_raw_eig": spectrum[4],
                "covariance_eigvals": spectrum[6],
                "observability_threshold": spectrum[8],
            }

        local = linearize(x_iter, theta_iter)
        initial_gradient = local["gradient"].copy()
        initial_information = local["information"].copy()
        initial_likelihood = float(local["terms"]["log_likelihood"])
        current_likelihood = initial_likelihood
        current_objective = initial_likelihood
        debug: Dict[str, Any] = {
            "laplace_log_likelihood": initial_likelihood,
            "laplace_log_likelihood_before": initial_likelihood,
            "laplace_log_likelihood_after": initial_likelihood,
            "laplace_posterior_objective_before": initial_likelihood,
            "laplace_posterior_objective_after": initial_likelihood,
            "laplace_grad_norm": float(np.linalg.norm(initial_gradient)),
            "est_gradient_norm": float(np.linalg.norm(initial_gradient)),
            "laplace_step_scale": 0.0,
            "laplace_step_acceptance_reason": "not_evaluated",
            "laplace_backtracking_trials": [],
            "laplace_outer_iteration_trials": [],
            "laplace_outer_iterations_requested": self.laplace_outer_iterations,
            "laplace_outer_iterations_completed": 0,
            "laplace_outer_iterations_accepted": 0,
            "laplace_max_log_kp_update_step": self.max_log_kp_update_step,
            "laplace_max_equilibrium_pose_jump": self.max_equilibrium_pose_jump,
            "laplace_joint_limit_reaction_torque_tol": (
                self.joint_limit_reaction_torque_tol
            ),
            "laplace_max_log_kp_covariance_var": self.max_log_kp_covariance_var,
            "laplace_prior_covariance_eigs_before_cap": (
                prior_covariance_eigvals_raw.copy()
            ),
            "laplace_prior_covariance_eigs_after_cap": (
                prior_covariance_eigvals_bounded.copy()
            ),
            "laplace_prior_covariance_capped": bool(
                np.any(
                    np.abs(
                        prior_covariance_eigvals_raw
                        - prior_covariance_eigvals_bounded
                    )
                    > 1.0e-12
                )
            ),
            "laplace_prior_center": x_prior.copy(),
            "laplace_prior_covariance": P_prior.copy(),
        }

        def terms_are_finite(local_value: Dict[str, Any]) -> bool:
            return bool(
                np.all(np.isfinite(local_value["gradient"]))
                and np.all(np.isfinite(local_value["information"]))
                and np.isfinite(float(local_value["terms"]["log_likelihood"]))
            )

        if not np.all(np.isfinite(theta_iter)) or not terms_are_finite(local):
            self.last_observable_rank = 0
            self.last_observable_eigvals = np.zeros_like(self.x_est)
            return self._finish_skipped_update(
                theta_iter,
                P_prior,
                debug,
                "nonfinite_laplace_terms",
                initial_gradient,
                initial_information,
            )

        accepted_any = False
        stationary = False
        accepted_scales: List[float] = []
        accepted_pose_jumps: List[float] = []
        accepted_pose_jumps_max_abs: List[float] = []
        flat_trial_debug: List[Dict[str, Any]] = []
        # Preserve the historical one-linearization filter exactly when the
        # optional outer loop is disabled.  In particular, a backtracked
        # single-pass update tempers its covariance by the accepted scale.
        single_pass_covariance_whitened: Optional[np.ndarray] = None

        for outer_index in range(self.laplace_outer_iterations):
            debug["laplace_outer_iterations_completed"] = outer_index + 1
            g = local["gradient"]
            information = local["information"]
            eigvals = local["eigvals"]
            eigvecs = local["eigvecs"]
            keep = local["keep"]
            if not terms_are_finite(local):
                if not accepted_any:
                    return self._finish_skipped_update(
                        theta_iter,
                        P_prior,
                        debug,
                        "nonfinite_laplace_terms",
                        g,
                        information,
                    )
                break
            if not np.any(keep):
                if not accepted_any:
                    return self._finish_skipped_update(
                        theta_iter,
                        P_prior,
                        debug,
                        "no_observable_information",
                        g,
                        information,
                    )
                break

            U_obs = eigvecs[:, keep]
            info_obs = eigvals[keep]
            info_scale = self._information_scale(info_obs)
            if info_scale <= 0.0:
                if not accepted_any:
                    return self._finish_skipped_update(
                        theta_iter,
                        P_prior,
                        debug,
                        "nonpositive_information_scale",
                        g,
                        information,
                    )
                break

            # In fixed prior-whitened coordinates y, the local MAP gradient is
            # S*g - y and the positive posterior curvature is I + S*I*S.
            # Crucially, y is measured from the one fixed batch prior, not from
            # the previous outer iterate.
            y_iter = covariance_sqrt_pinv @ (x_iter - x_prior)
            posterior_gradient_obs = U_obs.T @ (covariance_sqrt @ g - y_iter)
            delta_y_obs = posterior_gradient_obs / (1.0 + info_scale * info_obs + jitter)
            delta_x_full = covariance_sqrt @ (U_obs @ delta_y_obs)
            if not np.all(np.isfinite(delta_x_full)):
                if not accepted_any:
                    return self._finish_skipped_update(
                        theta_iter,
                        P_prior,
                        debug,
                        "nonfinite_posterior_step",
                        g,
                        information,
                    )
                break

            if float(np.linalg.norm(delta_x_full, ord=np.inf)) <= 1.0e-10:
                stationary = True
                debug["laplace_outer_stop_reason"] = "local_map_stationary"
                if self.laplace_outer_iterations == 1:
                    posterior_variance_obs = 1.0 / (
                        1.0 + info_scale * info_obs + jitter
                    )
                    single_pass_covariance_whitened = np.eye(
                        self.x_est.size,
                        dtype=float,
                    )
                    single_pass_covariance_whitened += U_obs @ (
                        np.diag(posterior_variance_obs - 1.0)
                    ) @ U_obs.T
                break

            outer_trials: List[Dict[str, Any]] = []
            accepted_eval: Optional[StiffnessLikelihoodEvaluation] = None
            accepted_scale: Optional[float] = None
            accepted_objective = current_objective
            accepted_pose_jump_norm = 0.0
            accepted_pose_jump_max_abs = 0.0

            for trial_index in range(self.laplace_backtracking_max_steps):
                step_scale = self.laplace_backtracking_factor ** trial_index
                if self.laplace_outer_iterations == 1:
                    # Backward-compatible tempered Laplace proposal.  The
                    # iterated path below instead line-searches a fixed-prior
                    # Newton/MAP direction.
                    tempered_precision = step_scale * (
                        info_scale * info_obs + jitter
                    )
                    posterior_variance_trial = 1.0 / (1.0 + tempered_precision)
                    delta_y_trial = posterior_variance_trial * (
                        step_scale * posterior_gradient_obs
                    )
                    trial_delta_x = covariance_sqrt @ (U_obs @ delta_y_trial)
                else:
                    posterior_variance_trial = None
                    trial_delta_x = step_scale * delta_x_full
                x_candidate = np.clip(
                    x_iter + trial_delta_x,
                    lower_log_kp,
                    upper_log_kp,
                )
                local_delta = x_candidate - x_iter
                total_delta = x_candidate - x_prior
                local_delta_norm = float(np.linalg.norm(local_delta))
                local_delta_max_abs = (
                    float(np.max(np.abs(local_delta))) if local_delta.size else 0.0
                )
                total_delta_max_abs = (
                    float(np.max(np.abs(total_delta))) if total_delta.size else 0.0
                )
                trial: Dict[str, Any] = {
                    "outer_iteration": outer_index,
                    "scale": float(step_scale),
                    "valid": True,
                    "accepted": False,
                    "dx_norm": local_delta_norm,
                    "dx_max_abs": local_delta_max_abs,
                    "batch_dx_max_abs": total_delta_max_abs,
                }
                if total_delta_max_abs > self.max_log_kp_update_step + 1.0e-12:
                    trial.update(
                        {
                            "exact_evaluated": False,
                            "reason": "log_kp_trust_region_exceeded",
                        }
                    )
                    outer_trials.append(trial)
                    flat_trial_debug.append(trial)
                    continue
                if local_delta_max_abs <= 1.0e-14:
                    stationary = True
                    trial.update(
                        {
                            "exact_evaluated": False,
                            "accepted": True,
                            "reason": "bounded_local_map_stationary",
                        }
                    )
                    outer_trials.append(trial)
                    flat_trial_debug.append(trial)
                    break

                candidate_eval = self.evaluate_log_likelihood_at_x(
                    x_eval=x_candidate,
                    theta_cmd_sent=theta_cmd_sent,
                    A_map=A_map,
                    theta_init_eq_pred=theta_iter,
                    kp_lim=kp_lim,
                )
                trial["exact_evaluated"] = True
                if not candidate_eval.valid:
                    trial.update(
                        {
                            "valid": False,
                            "reason": candidate_eval.error or "invalid_exact_likelihood",
                        }
                    )
                    outer_trials.append(trial)
                    flat_trial_debug.append(trial)
                    continue

                pose_delta = np.asarray(candidate_eval.theta_eq) - theta_iter
                pose_jump_norm = float(np.linalg.norm(pose_delta))
                pose_jump_max_abs = (
                    float(np.max(np.abs(pose_delta))) if pose_delta.size else 0.0
                )
                trial["equilibrium_pose_jump_norm"] = pose_jump_norm
                trial["equilibrium_pose_jump_max_abs"] = pose_jump_max_abs
                if (
                    not np.all(np.isfinite(pose_delta))
                    or pose_jump_norm > self.max_equilibrium_pose_jump + 1.0e-12
                ):
                    trial.update(
                        {
                            "valid": bool(np.all(np.isfinite(pose_delta))),
                            "log_likelihood": float(candidate_eval.log_likelihood),
                            "reason": (
                                "equilibrium_pose_jump_exceeded"
                                if np.all(np.isfinite(pose_delta))
                                else "nonfinite_equilibrium_pose_jump"
                            ),
                        }
                    )
                    outer_trials.append(trial)
                    flat_trial_debug.append(trial)
                    continue

                candidate_active = self._equilibrium_active_set_or_empty(
                    candidate_eval.theta_eq,
                    theta_cmd_sent,
                    candidate_eval.kp_vec,
                )
                # Compare with the active set at the beginning of this entire
                # batch, not merely the preceding outer iterate.  Otherwise a
                # grazing contact could be admitted in one iteration and then
                # grow into a strong reaction in the next one without ever
                # appearing "new" to the local check.
                new_active = candidate_active & ~batch_initial_active
                reaction_magnitudes = self._joint_limit_reaction_magnitudes_or_zero(
                    candidate_eval.theta_eq,
                    theta_cmd_sent,
                    candidate_eval.kp_vec,
                )
                new_strong_active = new_active & (
                    reaction_magnitudes > self.joint_limit_reaction_torque_tol
                )
                if np.any(new_strong_active):
                    trial.update(
                        {
                            "log_likelihood": float(candidate_eval.log_likelihood),
                            "new_joint_limit_active_indices": np.flatnonzero(
                                new_active
                            ),
                            "new_strong_joint_limit_active_indices": np.flatnonzero(
                                new_strong_active
                            ),
                            "new_joint_limit_reaction_magnitudes": reaction_magnitudes[
                                new_active
                            ].copy(),
                            "reason": "new_strong_joint_limit_active_set",
                        }
                    )
                    outer_trials.append(trial)
                    flat_trial_debug.append(trial)
                    continue

                prior_cost = 0.5 * float(total_delta.T @ (prior_precision @ total_delta))
                candidate_objective = float(candidate_eval.log_likelihood - prior_cost)
                comparison_tol = self.laplace_acceptance_tol * max(
                    1.0,
                    abs(current_likelihood),
                    abs(float(candidate_eval.log_likelihood)),
                    abs(current_objective),
                    abs(candidate_objective),
                )
                likelihood_nonworsening = bool(
                    candidate_eval.log_likelihood + comparison_tol >= current_likelihood
                )
                posterior_nonworsening = bool(
                    candidate_objective + comparison_tol >= current_objective
                )
                accepted = likelihood_nonworsening and posterior_nonworsening
                trial.update(
                    {
                        "accepted": accepted,
                        "log_likelihood": float(candidate_eval.log_likelihood),
                        "posterior_objective": candidate_objective,
                        "prior_cost": prior_cost,
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
                outer_trials.append(trial)
                flat_trial_debug.append(trial)
                if accepted:
                    accepted_eval = candidate_eval
                    accepted_scale = float(step_scale)
                    accepted_objective = candidate_objective
                    accepted_pose_jump_norm = pose_jump_norm
                    accepted_pose_jump_max_abs = pose_jump_max_abs
                    if (
                        self.laplace_outer_iterations == 1
                        and posterior_variance_trial is not None
                    ):
                        single_pass_covariance_whitened = np.eye(
                            self.x_est.size,
                            dtype=float,
                        )
                        single_pass_covariance_whitened += U_obs @ (
                            np.diag(posterior_variance_trial - 1.0)
                        ) @ U_obs.T
                    break

            debug["laplace_outer_iteration_trials"].append(outer_trials)
            if stationary:
                break
            if accepted_eval is None or accepted_scale is None:
                debug["laplace_outer_stop_reason"] = "exact_posterior_not_improved"
                if not accepted_any:
                    debug["laplace_backtracking_trials"] = flat_trial_debug
                    return self._finish_skipped_update(
                        theta_iter,
                        P_prior,
                        debug,
                        "exact_posterior_not_improved",
                        g,
                        information,
                    )
                break

            x_iter = accepted_eval.x_eval.copy()
            theta_iter = accepted_eval.theta_eq.copy()
            current_likelihood = float(accepted_eval.log_likelihood)
            current_objective = float(accepted_objective)
            accepted_any = True
            accepted_scales.append(accepted_scale)
            accepted_pose_jumps.append(accepted_pose_jump_norm)
            accepted_pose_jumps_max_abs.append(accepted_pose_jump_max_abs)
            debug["laplace_outer_iterations_accepted"] = len(accepted_scales)
            local = linearize(x_iter, theta_iter)
        else:
            # A fully consumed outer budget is a normal termination, but make
            # it distinguishable from a missing diagnostic in the ROS status.
            debug["laplace_outer_stop_reason"] = "max_outer_iterations"

        # One final linearization supplies one posterior covariance for this
        # observation.  Outer iterations above altered only the MAP mean; they
        # never recursively fed P_post or Q back into the same batch.
        final_local = linearize(x_iter, theta_iter)
        final_g = final_local["gradient"]
        final_information = final_local["information"]
        if (
            self.laplace_outer_iterations == 1
            and single_pass_covariance_whitened is not None
        ):
            final_info_scale = accepted_scales[-1] if accepted_scales else 1.0
            P_post = (
                covariance_sqrt
                @ single_pass_covariance_whitened
                @ covariance_sqrt
            )
        elif not terms_are_finite(final_local):
            if not accepted_any:
                return self._finish_skipped_update(
                    theta_iter,
                    P_prior,
                    debug,
                    "nonfinite_final_laplace_terms",
                    final_g,
                    final_information,
                )
            P_post = P_prior.copy()
            final_info_scale = 0.0
        else:
            final_keep = final_local["keep"]
            final_eigvals = final_local["eigvals"]
            final_eigvecs = final_local["eigvecs"]
            if np.any(final_keep):
                final_U_obs = final_eigvecs[:, final_keep]
                final_info_obs = final_eigvals[final_keep]
                final_info_scale = self._information_scale(final_info_obs)
                posterior_variance_obs = 1.0 / (
                    1.0 + final_info_scale * final_info_obs + jitter
                )
                covariance_whitened = np.eye(self.x_est.size, dtype=float)
                covariance_whitened += final_U_obs @ (
                    np.diag(posterior_variance_obs - 1.0)
                ) @ final_U_obs.T
                P_post = covariance_sqrt @ covariance_whitened @ covariance_sqrt
            else:
                final_info_scale = 0.0
                P_post = P_prior.copy()

        if np.all(np.isfinite(final_information)):
            raw_information_eigvals = np.linalg.eigvalsh(
                0.5 * (final_information + final_information.T)
            )
        else:
            raw_information_eigvals = np.zeros(self.x_est.size, dtype=float)
        min_raw_information_eig = (
            float(np.min(raw_information_eigvals)) if raw_information_eigvals.size else 0.0
        )
        final_dx = x_iter - x_prior
        total_pose_delta = theta_iter - theta_prior
        self.x_est = x_iter.copy()
        self.P_est = self._bounded_covariance(P_post)[0]
        self.last_theta_eq = theta_iter.copy()
        self.last_information_scale = float(final_info_scale)
        self.last_update_step = final_dx.copy()
        self.last_update_norm = float(np.linalg.norm(final_dx))
        # A stationary MAP still assimilates the observation through P_post.
        # Keep the public result, debug flag, and cached flag semantically
        # consistent: "applied" means the batch was committed, not that its
        # mean happened to move by a non-zero floating-point amount.
        self.last_update_applied = True

        debug.update(
            {
                "laplace_backtracking_trials": flat_trial_debug,
                "laplace_information_scale": float(final_info_scale),
                "laplace_info_eigs": np.maximum(raw_information_eigvals, 0.0),
                "laplace_raw_info_eigs": raw_information_eigvals.copy(),
                "laplace_info_negative_min_eig": min_raw_information_eig,
                "laplace_info_has_large_negative_eig": bool(
                    min_raw_information_eig < -self.laplace_negative_info_tol
                ),
                "laplace_prior_whitened_info_eigs": final_local["eigvals"].copy(),
                "laplace_prior_whitened_raw_info_eigs": final_local[
                    "whitened_raw_eigvals"
                ].copy(),
                "laplace_prior_whitened_info_negative_min_eig": float(
                    final_local["min_whitened_raw_eig"]
                ),
                "laplace_prior_covariance_eigs": covariance_eigvals.copy(),
                "laplace_prior_whitened_observability_threshold": float(
                    final_local["observability_threshold"]
                ),
                "laplace_obs_rank": int(np.count_nonzero(final_local["keep"])),
                "est_obs_rank": int(np.count_nonzero(final_local["keep"])),
                "est_information_eigs": final_local["eigvals"].copy(),
                "laplace_grad_norm_after": float(np.linalg.norm(final_g)),
                "laplace_dx_norm": self.last_update_norm,
                "laplace_dx_max_abs": (
                    float(np.max(np.abs(final_dx))) if final_dx.size else 0.0
                ),
                "laplace_step_scale": accepted_scales[-1] if accepted_scales else 0.0,
                "laplace_step_scales": accepted_scales.copy(),
                "laplace_step_acceptance_reason": (
                    "iterated_exact_posterior_improved"
                    if accepted_any
                    else "stationary_exact_posterior"
                ),
                "laplace_log_likelihood_after": current_likelihood,
                "laplace_posterior_objective_after": current_objective,
                "laplace_equilibrium_pose_jump_norm": (
                    accepted_pose_jumps[-1] if accepted_pose_jumps else 0.0
                ),
                "laplace_equilibrium_pose_jump_max_abs": (
                    accepted_pose_jumps_max_abs[-1]
                    if accepted_pose_jumps_max_abs
                    else 0.0
                ),
                "laplace_equilibrium_total_pose_jump_norm": float(
                    np.linalg.norm(total_pose_delta)
                ),
                "laplace_covariance_commit_count": 1,
                "est_update_applied": True,
                "est_update_skipped_reason": None,
            }
        )
        self.last_debug = debug
        return StiffnessUpdateResult(
            theta_eq=theta_iter.copy(),
            x_est=self.x_est.copy(),
            kp_est=self.kp_est.copy(),
            P_est=self.P_est.copy(),
            gradient=final_g.copy(),
            information=final_information.copy(),
            obs_rank=self.last_observable_rank,
            update_applied=True,
            update_skipped_reason=None,
            debug=debug,
        )


MultiFrameWeirdEKF = MultiFrameStiffnessWEKF
