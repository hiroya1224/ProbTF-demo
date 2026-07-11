from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF


@dataclass
class StiffnessParticleRecord:
    theta_cmd_sent: np.ndarray
    A_map: Dict[int, np.ndarray]
    theta_init_eq_pred: Optional[np.ndarray]
    stamp: Optional[float]


@dataclass
class StiffnessParticleScanConfig:
    enabled: bool = False
    plain: bool = True
    window_size: int = 20
    period: int = 5
    grid_size: int = 21
    max_active_dims: int = 2
    std_trigger: float = 0.15
    info_abs: float = 1.0e-8
    min_gain_per_obs: float = 1.0
    min_log_jump: float = 0.05
    reset_std: float = 0.10
    cooldown: int = 20
    mode: str = "axis"


@dataclass
class StiffnessParticleScanResult:
    attempted: bool
    accepted: bool
    reason: str
    x_current: Optional[np.ndarray]
    x_best: Optional[np.ndarray]
    score_current: float
    score_best: float
    gain_per_obs: float
    active_indices: np.ndarray
    candidate_count: int
    theta_eq_best: Optional[np.ndarray]
    debug: Dict[str, Any]


class StiffnessParticleScanSupervisor:
    def __init__(self, config: StiffnessParticleScanConfig) -> None:
        self.config = config
        self.records: Deque[StiffnessParticleRecord] = deque(maxlen=self._window_size())
        self.step_count = 0
        self.cooldown_count = 0
        self.last_result = self._result(
            attempted=False,
            accepted=False,
            reason="not_run",
            x_current=None,
            x_best=None,
            score_current=-np.inf,
            score_best=-np.inf,
            gain_per_obs=0.0,
            active_indices=np.array([], dtype=int),
            candidate_count=0,
            theta_eq_best=None,
            debug_extra={},
        )

    def _window_size(self) -> int:
        return max(1, int(self.config.window_size))

    def add_record(
        self,
        theta_cmd_sent: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        stamp: Optional[float],
    ) -> None:
        copied_map = {fid: np.asarray(A_f, dtype=float).copy() for fid, A_f in A_map.items()}
        record = StiffnessParticleRecord(
            theta_cmd_sent=np.asarray(theta_cmd_sent, dtype=float).copy(),
            A_map=copied_map,
            theta_init_eq_pred=None
            if theta_init_eq_pred is None
            else np.asarray(theta_init_eq_pred, dtype=float).copy(),
            stamp=stamp,
        )
        self.records.append(record)

    def _result(
        self,
        attempted: bool,
        accepted: bool,
        reason: str,
        x_current: Optional[np.ndarray],
        x_best: Optional[np.ndarray],
        score_current: float,
        score_best: float,
        gain_per_obs: float,
        active_indices: np.ndarray,
        candidate_count: int,
        theta_eq_best: Optional[np.ndarray],
        debug_extra: Dict[str, Any],
    ) -> StiffnessParticleScanResult:
        active = np.asarray(active_indices, dtype=int).copy()
        debug: Dict[str, Any] = {
            "window_size": len(self.records),
            "window_capacity": int(self.records.maxlen or 0),
            "step_count": int(self.step_count),
            "cooldown_count": int(self.cooldown_count),
            "plain": bool(self.config.plain),
            "active_indices": active.copy(),
            "candidate_count": int(candidate_count),
            "score_current": float(score_current),
            "score_best": float(score_best),
            "gain_per_obs": float(gain_per_obs),
            "accepted": bool(accepted),
            "attempted": bool(attempted),
            "reason": reason,
        }
        if x_current is not None:
            x_cur = np.asarray(x_current, dtype=float).copy()
            debug["x_current"] = x_cur
            debug["kp_current"] = np.exp(x_cur)
        if x_best is not None:
            x_b = np.asarray(x_best, dtype=float).copy()
            debug["x_best"] = x_b
            debug["kp_best"] = np.exp(x_b)
            if x_current is not None:
                debug["max_jump"] = float(np.max(np.abs(x_b - np.asarray(x_current, dtype=float))))
        debug.update(debug_extra)
        return StiffnessParticleScanResult(
            attempted=attempted,
            accepted=accepted,
            reason=reason,
            x_current=None if x_current is None else np.asarray(x_current, dtype=float).copy(),
            x_best=None if x_best is None else np.asarray(x_best, dtype=float).copy(),
            score_current=float(score_current),
            score_best=float(score_best),
            gain_per_obs=float(gain_per_obs),
            active_indices=active,
            candidate_count=int(candidate_count),
            theta_eq_best=None if theta_eq_best is None else np.asarray(theta_eq_best, dtype=float).copy(),
            debug=debug,
        )

    def _active_indices(
        self,
        P_est: np.ndarray,
        information: Optional[np.ndarray],
    ) -> np.ndarray:
        max_active_dims = int(self.config.max_active_dims)
        if max_active_dims <= 0:
            return np.array([], dtype=int)

        P = np.asarray(P_est, dtype=float)
        diag = np.diag(P)
        std = np.sqrt(np.maximum(diag, 0.0))
        mask = np.isfinite(std) & (std <= float(self.config.std_trigger))

        info_diag = None
        if information is not None:
            info = np.asarray(information, dtype=float)
            if info.ndim == 2 and info.shape[0] == info.shape[1] and info.shape[0] == diag.size:
                info_diag = np.diag(info)
                mask = mask & np.isfinite(info_diag) & (np.abs(info_diag) >= float(self.config.info_abs))
            else:
                mask = np.zeros_like(mask, dtype=bool)

        indices = np.flatnonzero(mask).astype(int)
        if indices.size <= max_active_dims:
            return indices

        if info_diag is not None:
            order = np.argsort(-np.abs(info_diag[indices]))
        else:
            order = np.argsort(diag[indices])
        return indices[order[:max_active_dims]]

    def _make_axis_candidates(
        self,
        x_current: np.ndarray,
        active_indices: np.ndarray,
        kp_lim: Tuple[float, float],
    ) -> List[np.ndarray]:
        kp_min, kp_max = (float(v) for v in kp_lim)
        if kp_min <= 0.0 or kp_max < kp_min:
            return []
        grid_size = int(self.config.grid_size)
        if grid_size <= 0:
            return []

        log_min = float(np.log(kp_min))
        log_max = float(np.log(kp_max))
        x_base = np.clip(np.asarray(x_current, dtype=float), log_min, log_max)
        grid = np.linspace(log_min, log_max, grid_size)

        candidates: List[np.ndarray] = []
        seen = set()

        def add_candidate(x: np.ndarray) -> None:
            x_clip = np.clip(np.asarray(x, dtype=float), log_min, log_max)
            key = tuple(np.round(x_clip, decimals=12))
            if key not in seen:
                seen.add(key)
                candidates.append(x_clip.copy())

        add_candidate(x_base)
        for j in np.asarray(active_indices, dtype=int):
            if j < 0 or j >= x_base.size:
                continue
            for value in grid:
                x = x_base.copy()
                x[int(j)] = float(value)
                add_candidate(x)
        return candidates

    def _score_candidate(
        self,
        estimator: MultiFrameStiffnessWEKF,
        x_candidate: np.ndarray,
        kp_lim: Tuple[float, float],
    ) -> Tuple[float, Optional[np.ndarray], Dict[str, Any]]:
        score = 0.0
        theta_eq_last: Optional[np.ndarray] = None
        errors: List[str] = []

        for idx, record in enumerate(self.records):
            evaluation = estimator.evaluate_log_likelihood_at_x(
                x_eval=x_candidate,
                theta_cmd_sent=record.theta_cmd_sent,
                A_map=record.A_map,
                theta_init_eq_pred=record.theta_init_eq_pred,
                kp_lim=kp_lim,
            )
            if not evaluation.valid or not np.isfinite(evaluation.log_likelihood):
                errors.append(f"record_{idx}:{evaluation.error}")
                return -np.inf, None, {"errors": errors}
            score += float(evaluation.log_likelihood)
            theta_eq_last = evaluation.theta_eq.copy()

        return float(score), theta_eq_last, {"errors": errors}

    def maybe_scan(
        self,
        estimator: MultiFrameStiffnessWEKF,
        latest_information: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float, float]],
    ) -> StiffnessParticleScanResult:
        self.step_count += 1
        empty_active = np.array([], dtype=int)

        if not bool(self.config.enabled):
            self.last_result = self._result(
                attempted=False,
                accepted=False,
                reason="disabled",
                x_current=None,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        min_records = 1 if bool(self.config.plain) else self._window_size()
        if len(self.records) < min_records:
            self.last_result = self._result(
                attempted=False,
                accepted=False,
                reason="no_records" if bool(self.config.plain) else "window_not_full",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        if not bool(self.config.plain) and self.cooldown_count > 0:
            self.cooldown_count -= 1
            self.last_result = self._result(
                attempted=False,
                accepted=False,
                reason="cooldown",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        period = max(1, int(self.config.period))
        if not bool(self.config.plain) and self.step_count % period != 0:
            self.last_result = self._result(
                attempted=False,
                accepted=False,
                reason="period_skip",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        if kp_lim is None:
            self.last_result = self._result(
                attempted=False,
                accepted=False,
                reason="missing_kp_lim",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        if bool(self.config.plain):
            active_indices = np.arange(np.asarray(estimator.x_est, dtype=float).size, dtype=int)
        else:
            active_indices = self._active_indices(estimator.P_est, latest_information)
        if active_indices.size == 0:
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="no_active_dimensions",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        if str(self.config.mode).strip().lower() != "axis":
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason=f"unsupported_mode:{self.config.mode}",
                x_current=estimator.x_est,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        x_current = estimator.x_est.copy()
        candidates = self._make_axis_candidates(x_current, active_indices, kp_lim)
        if not candidates:
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="no_candidates",
                x_current=x_current,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra={},
            )
            return self.last_result

        score_current, theta_eq_current, debug_current = self._score_candidate(estimator, x_current, kp_lim)
        if not np.isfinite(score_current):
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="nonfinite_current_score",
                x_current=x_current,
                x_best=None,
                score_current=score_current,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=len(candidates),
                theta_eq_best=None,
                debug_extra={"current_score_debug": debug_current},
            )
            return self.last_result

        x_best = x_current.copy()
        score_best = score_current
        theta_eq_best = theta_eq_current
        candidate_errors: List[Dict[str, Any]] = []
        for candidate in candidates:
            score, theta_eq, score_debug = self._score_candidate(estimator, candidate, kp_lim)
            if not np.isfinite(score):
                if score_debug.get("errors"):
                    candidate_errors.append(score_debug)
                continue
            if score > score_best:
                score_best = score
                x_best = candidate.copy()
                theta_eq_best = None if theta_eq is None else theta_eq.copy()

        if not np.isfinite(score_best):
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="nonfinite_best_score",
                x_current=x_current,
                x_best=None,
                score_current=score_current,
                score_best=score_best,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=len(candidates),
                theta_eq_best=None,
                debug_extra={"candidate_errors": candidate_errors},
            )
            return self.last_result

        gain_per_obs = (float(score_best) - float(score_current)) / max(1, len(self.records))
        max_jump = float(np.max(np.abs(x_best - x_current))) if x_best.size else 0.0
        if bool(self.config.plain):
            accepted = float(score_best) > float(score_current)
            reason = "accepted_plain" if accepted else "no_score_improvement"
        else:
            accepted = (
                gain_per_obs >= float(self.config.min_gain_per_obs)
                and max_jump >= float(self.config.min_log_jump)
            )
            reason = "accepted" if accepted else "insufficient_gain_or_jump"
        if accepted and not bool(self.config.plain):
            self.cooldown_count = max(0, int(self.config.cooldown))

        self.last_result = self._result(
            attempted=True,
            accepted=accepted,
            reason=reason,
            x_current=x_current,
            x_best=x_best,
            score_current=score_current,
            score_best=score_best,
            gain_per_obs=gain_per_obs,
            active_indices=active_indices,
            candidate_count=len(candidates),
            theta_eq_best=theta_eq_best,
            debug_extra={
                "max_jump": max_jump,
                "candidate_errors": candidate_errors,
            },
        )
        return self.last_result
