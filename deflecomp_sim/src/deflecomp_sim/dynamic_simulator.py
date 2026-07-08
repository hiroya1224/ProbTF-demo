from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pinocchio as pin

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel, SpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm


@dataclass
class DynamicParams:
    K: np.ndarray
    D: Optional[np.ndarray] = None
    zeta: float = 0.05
    q0_for_damp: Optional[np.ndarray] = None
    use_pinv: bool = True
    limit_velocity: Optional[np.ndarray] = None
    limit_position_low: Optional[np.ndarray] = None
    limit_position_high: Optional[np.ndarray] = None
    integrator: str = "rk4"
    ref_tau: Optional[float] = 1e-4
    ref_max_vel: Optional[float] = 10.0
    eq_mode: Literal["dynamic", "relax_to_eq", "quasistatic"] = "dynamic"
    tau_eq: Optional[float] = 0.05
    qs_noise_std_deg: float = 0.0
    qs_vib_amp_deg: float = 0.0
    qs_vib_freq_hz: float = 50.0
    qs_vib_axes: Optional[np.ndarray] = None
    qs_seed: Optional[int] = None


class FlexibleJointSimulator:
    def __init__(
        self,
        robot: RobotArm,
        params: DynamicParams,
        spring_model: Optional[SpringModel] = None,
    ) -> None:
        self.robot = robot
        self.params = params
        self.spring_model = spring_model or LinearSpringModel()

        n = self.robot.nv
        K = np.asarray(params.K, dtype=float)
        if K.shape != (n,):
            raise ValueError(f"K must have shape ({n},), got {K.shape}")
        self.K = K

        if params.D is not None:
            D = np.asarray(params.D, dtype=float)
            if D.shape != (n,):
                raise ValueError(f"D must have shape ({n},), got {D.shape}")
            self.D = D.copy()
        else:
            q0 = params.q0_for_damp
            if q0 is None:
                lo = self.robot.model.lowerPositionLimit
                hi = self.robot.model.upperPositionLimit
                q0 = 0.5 * (lo + hi) if (lo.shape[0] == n and hi.shape[0] == n) else np.zeros(n)
            M0 = pin.crba(self.robot.model, self.robot.data, q0)
            M0 = 0.5 * (M0 + M0.T)
            Mdiag = np.clip(np.diag(M0), 1e-6, np.inf)
            self.D = 2.0 * float(params.zeta) * np.sqrt(self.K * Mdiag)

        self.q = np.zeros(n, dtype=float)
        self.qd = np.zeros(n, dtype=float)
        self.vel_lim = None if params.limit_velocity is None else np.asarray(params.limit_velocity, dtype=float)
        lo = self.robot.model.lowerPositionLimit
        hi = self.robot.model.upperPositionLimit
        self.pos_lo = (
            np.asarray(params.limit_position_low, dtype=float)
            if params.limit_position_low is not None
            else (lo if lo.shape[0] == n else None)
        )
        self.pos_hi = (
            np.asarray(params.limit_position_high, dtype=float)
            if params.limit_position_high is not None
            else (hi if hi.shape[0] == n else None)
        )

        self.q_ref_filt = self.q.copy()
        self.q_ref_prev = self.q.copy()
        self.eq_solver = EquilibriumSolver(
            robot=self.robot,
            spring_model=self.spring_model,
            cfg=EquilibriumConfig(maxiter=80),
        )
        self._rng = np.random.default_rng(params.qs_seed)
        self._t = 0.0

    def set_eq_solver(self, solver: EquilibriumSolver) -> None:
        self.eq_solver = solver

    def reset(self, q: Optional[np.ndarray] = None, qd: Optional[np.ndarray] = None) -> None:
        n = self.robot.nv
        self._t = 0.0
        self.q = np.zeros(n, dtype=float) if q is None else np.asarray(q, dtype=float).copy()
        self.qd = np.zeros(n, dtype=float) if qd is None else np.asarray(qd, dtype=float).copy()
        self.q_ref_filt = self.q.copy()
        self.q_ref_prev = self.q.copy()

    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.qd.copy()

    def _shape_reference(self, dt: float, q_ref_in: np.ndarray) -> np.ndarray:
        q_ref = np.asarray(q_ref_in, dtype=float)
        if self.params.ref_max_vel is not None and self.params.ref_max_vel > 0.0:
            delta = np.clip(
                q_ref - self.q_ref_prev,
                -abs(self.params.ref_max_vel) * dt,
                abs(self.params.ref_max_vel) * dt,
            )
            q_ref_slew = self.q_ref_prev + delta
        else:
            q_ref_slew = q_ref
        self.q_ref_prev = q_ref_slew

        if self.params.ref_tau is not None and self.params.ref_tau > 0.0:
            alpha = 1.0 - np.exp(-dt / float(self.params.ref_tau))
            self.q_ref_filt = self.q_ref_filt + alpha * (q_ref_slew - self.q_ref_filt)
        else:
            self.q_ref_filt = q_ref_slew
        return self.q_ref_filt

    def _solve_equilibrium(self, theta_cmd: np.ndarray, kp_vec: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        return self.eq_solver.solve(
            theta_cmd=np.asarray(theta_cmd, dtype=float),
            kp_vec=np.asarray(kp_vec, dtype=float),
            theta_init=np.asarray(q_init, dtype=float),
        )

    def _dyn_rhs(self, q: np.ndarray, qd: np.ndarray, q_ref_eff: np.ndarray, tau_ext: Optional[np.ndarray]) -> np.ndarray:
        n = self.robot.nv
        if tau_ext is None:
            tau_ext = np.zeros(n, dtype=float)
        else:
            tau_ext = np.asarray(tau_ext, dtype=float)

        M = pin.crba(self.robot.model, self.robot.data, q)
        M = 0.5 * (M + M.T)
        b = pin.rnea(self.robot.model, self.robot.data, q, qd, np.zeros(n, dtype=float))
        tau_spring = -self.spring_model.torque(q, q_ref_eff, self.K) - self.D * qd
        rhs = tau_ext + tau_spring - b
        if self.params.use_pinv:
            return np.linalg.pinv(M, rcond=1e-12) @ rhs
        return np.linalg.solve(M, rhs)

    def _apply_quasistatic_perturbation(self, q_eq: np.ndarray, dt: float) -> np.ndarray:
        self._t += float(dt)
        n = q_eq.shape[0]
        noise = np.zeros(n, dtype=float)
        if self.params.qs_noise_std_deg > 0.0:
            noise = self._rng.standard_normal(n) * np.deg2rad(self.params.qs_noise_std_deg)

        vib = np.zeros(n, dtype=float)
        if self.params.qs_vib_amp_deg > 0.0:
            scalar = np.sin(2.0 * np.pi * float(self.params.qs_vib_freq_hz) * self._t)
            vib = np.ones(n, dtype=float) * np.deg2rad(self.params.qs_vib_amp_deg) * scalar
            if self.params.qs_vib_axes is not None:
                mask = np.zeros(n, dtype=float)
                for idx in np.asarray(self.params.qs_vib_axes, dtype=int).ravel():
                    if 0 <= int(idx) < n:
                        mask[int(idx)] = 1.0
                vib = vib * mask
        return q_eq + noise + vib

    def step(
        self,
        dt: float,
        q_ref: np.ndarray,
        tau_ext: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_ref = np.asarray(q_ref, dtype=float)
        n = self.robot.nv
        if q_ref.shape != (n,):
            raise ValueError(f"q_ref must have shape ({n},), got {q_ref.shape}")

        mode = self.params.eq_mode
        if mode in ("quasistatic", "relax_to_eq"):
            q_eq = self._solve_equilibrium(theta_cmd=q_ref, kp_vec=self.K, q_init=self.q)
            if mode == "quasistatic":
                q_next = self._apply_quasistatic_perturbation(q_eq=q_eq, dt=dt)
                qd_next = np.zeros_like(q_eq)
            else:
                tau = float(self.params.tau_eq if (self.params.tau_eq is not None and self.params.tau_eq > 0.0) else 0.05)
                alpha = 1.0 - np.exp(-dt / max(tau, 1e-6))
                q_next = self.q + alpha * (q_eq - self.q)
                qd_next = (q_next - self.q) / max(dt, 1e-9)
        else:
            q_ref_eff = self._shape_reference(dt=dt, q_ref_in=q_ref)
            if self.params.integrator.lower() == "rk4":
                q0 = self.q.copy()
                v0 = self.qd.copy()

                def rhs(q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                    return v, self._dyn_rhs(q, v, q_ref_eff, tau_ext)

                k1_v, k1_a = rhs(q0, v0)
                k2_v, k2_a = rhs(q0 + 0.5 * dt * k1_v, v0 + 0.5 * dt * k1_a)
                k3_v, k3_a = rhs(q0 + 0.5 * dt * k2_v, v0 + 0.5 * dt * k2_a)
                k4_v, k4_a = rhs(q0 + dt * k3_v, v0 + dt * k3_a)

                q_next = q0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
                qd_next = v0 + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
            else:
                qdd = self._dyn_rhs(self.q, self.qd, q_ref_eff, tau_ext)
                qd_next = self.qd + dt * qdd
                q_next = self.q + dt * qd_next

        if self.vel_lim is not None:
            qd_next = np.clip(qd_next, -self.vel_lim, self.vel_lim)
        if self.pos_lo is not None and self.pos_hi is not None:
            q_next = np.minimum(np.maximum(q_next, self.pos_lo), self.pos_hi)

        self.q = q_next
        self.qd = qd_next
        return q_next.copy(), qd_next.copy()

    def simulate(
        self,
        dt: float,
        q_ref_seq: np.ndarray,
        tau_ext_seq: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_ref_seq = np.asarray(q_ref_seq, dtype=float)
        T = q_ref_seq.shape[0]
        Q = np.zeros_like(q_ref_seq)
        Qd = np.zeros_like(q_ref_seq)
        for k in range(T):
            tau_ext = None if tau_ext_seq is None else tau_ext_seq[k]
            Q[k], Qd[k] = self.step(dt=dt, q_ref=q_ref_seq[k], tau_ext=tau_ext)
        return Q, Qd


DynamicSimulator = FlexibleJointSimulator
