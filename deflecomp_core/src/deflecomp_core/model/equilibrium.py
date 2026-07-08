from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares, minimize

from deflecomp_core.model.spring import SpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm

@dataclass
class EquilibriumConfig:
    maxiter: int = 200
    k_stiffness: float = 100.0
    n_lambda: int = 10
    ftol: float = 1e-12
    refine: bool = True
    refine_maxiter: int = 40
    refine_tol: float = 1e-12
    verbose: bool = False

class EquilibriumSolver:
    def __init__(
        self,
        robot: RobotArm,
        spring_model: SpringModel,
        cfg: Optional[EquilibriumConfig] = None,
    ) -> None:
        self.robot = robot
        self.spring_model = spring_model
        self.cfg = cfg or EquilibriumConfig()
        self.eq_path_last: List[np.ndarray] = []

    @staticmethod
    def _V_total(
        robot: RobotArm,
        spring_model: SpringModel,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        k_eff_diag: np.ndarray,
    ) -> float:
        U = robot.potential_gravity(theta)
        return float(U + spring_model.potential(theta, theta_cmd, k_eff_diag))

    @staticmethod
    def _grad_theta(
        robot: RobotArm,
        spring_model: SpringModel,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        k_eff_diag: np.ndarray,
    ) -> np.ndarray:
        tau_g = robot.tau_gravity(theta)
        return tau_g + spring_model.torque(theta, theta_cmd, k_eff_diag)

    def _position_bounds(self, size: int) -> List[Tuple[Optional[float], Optional[float]]]:
        lo = np.asarray(self.robot.model.lowerPositionLimit, dtype=float)
        hi = np.asarray(self.robot.model.upperPositionLimit, dtype=float)
        if lo.shape != (size,) or hi.shape != (size,):
            return [(None, None)] * size
        return [
            (
                float(lo_i) if np.isfinite(lo_i) else None,
                float(hi_i) if np.isfinite(hi_i) else None,
            )
            for lo_i, hi_i in zip(lo, hi)
        ]

    @staticmethod
    def _clip_to_bounds(theta: np.ndarray, bounds: List[Tuple[Optional[float], Optional[float]]]) -> np.ndarray:
        clipped = np.asarray(theta, dtype=float).copy()
        for idx, (lo, hi) in enumerate(bounds):
            if lo is not None:
                clipped[idx] = max(clipped[idx], lo)
            if hi is not None:
                clipped[idx] = min(clipped[idx], hi)
        return clipped

    def _stage_minimize(self, theta_cmd: np.ndarray, k_eff_diag: np.ndarray, theta0: np.ndarray) -> np.ndarray:
        bounds = self._position_bounds(theta0.size)
        theta_start = self._clip_to_bounds(theta0, bounds)

        def f_obj(theta: np.ndarray) -> float:
            return self._V_total(self.robot, self.spring_model, theta, theta_cmd, k_eff_diag)

        def f_jac(theta: np.ndarray) -> np.ndarray:
            return self._grad_theta(self.robot, self.spring_model, theta, theta_cmd, k_eff_diag)

        res = minimize(
            fun=f_obj,
            x0=theta_start,
            jac=f_jac,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.maxiter),
                "ftol": float(self.cfg.ftol),
                "gtol": min(float(self.cfg.ftol), 1e-10),
                "maxls": 50,
                "disp": bool(self.cfg.verbose),
            },
        )
        return np.asarray(res.x, dtype=float)

    def _bounds_arrays(self, bounds: List[Tuple[Optional[float], Optional[float]]]) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array(
            [(-np.inf if bound[0] is None else float(bound[0])) for bound in bounds],
            dtype=float,
        )
        hi = np.array(
            [(np.inf if bound[1] is None else float(bound[1])) for bound in bounds],
            dtype=float,
        )
        return lo, hi

    def _refine_stationary(
        self,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
        theta0: np.ndarray,
    ) -> np.ndarray:
        bounds = self._position_bounds(theta0.size)
        theta_start = self._clip_to_bounds(theta0, bounds)
        lo, hi = self._bounds_arrays(bounds)

        def residual(theta: np.ndarray) -> np.ndarray:
            return self._grad_theta(self.robot, self.spring_model, theta, theta_cmd, kp_vec)

        def jacobian(theta: np.ndarray) -> np.ndarray:
            d_tau_g = self.robot.d_tau_gravity(theta)
            spring_k = self.spring_model.stiffness_diag(theta, theta_cmd, kp_vec)
            return d_tau_g + np.diag(spring_k)

        residual_start = residual(theta_start)
        start_norm = float(np.linalg.norm(residual_start))
        if start_norm <= float(self.cfg.refine_tol):
            return theta_start

        res = least_squares(
            fun=residual,
            jac=jacobian,
            x0=theta_start,
            bounds=(lo, hi),
            method="trf",
            ftol=float(self.cfg.refine_tol),
            xtol=float(self.cfg.refine_tol),
            gtol=float(self.cfg.refine_tol),
            max_nfev=int(self.cfg.refine_maxiter),
            x_scale="jac",
            verbose=2 if self.cfg.verbose else 0,
        )
        theta_refined = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(theta_refined)):
            return theta_start

        refined_norm = float(np.linalg.norm(residual(theta_refined)))
        if refined_norm <= start_norm:
            return theta_refined
        return theta_start

    def solve(
        self,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
        theta_init: Optional[np.ndarray] = None,
        lambdas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, self.cfg.n_lambda)

        theta0 = theta_init.copy() if theta_init is not None else theta_cmd.copy()

        self.eq_path_last = []
        for lam in lambdas:
            k_eff_diag = kp_vec + float(lam) * float(self.cfg.k_stiffness)
            theta_opt = self._stage_minimize(theta_cmd, k_eff_diag, theta0)
            self.eq_path_last.append(theta_opt.copy())
            theta0 = theta_opt
        if self.cfg.refine:
            theta_opt = self._refine_stationary(theta_cmd, kp_vec, theta0)
            self.eq_path_last.append(theta_opt.copy())
        return theta_opt
