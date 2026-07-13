from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF


@dataclass
class StiffnessParticleRecord:
    theta_cmd_sent: np.ndarray
    A_map: Dict[int, np.ndarray]
    theta_init_eq_pred: Optional[np.ndarray]
    stamp: Optional[float]
    information: Optional[np.ndarray] = None


@dataclass
class StiffnessParticleScanConfig:
    enabled: bool = False
    window_size: int = 20
    period: int = 5
    grid_size: int = 21
    max_active_dims: int = 2
    info_abs: float = 1.0e-8
    info_rcond: float = 1.0e-4
    require_information: bool = True
    validation_fraction: float = 0.25
    min_validation_records: int = 4
    min_gain_per_obs: float = 1.0
    min_validation_gain_per_obs: Optional[float] = None
    min_log_jump: float = 0.02
    max_log_jump: float = float(np.log(2.0))
    max_equilibrium_jump: float = 0.35
    reset_std: float = 0.10
    cooldown: int = 20
    max_result_age_records: int = 5
    max_estimator_drift: float = 0.15


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
    """Conservative, deterministic bounded maximum-likelihood supervisor.

    The WEKF remains the primary estimator.  This supervisor only proposes a
    bounded correction after a complete observation window has accumulated.
    Candidate discovery and validation use disjoint records, and only locally
    observable information eigen-directions are searched.  Candidate scores
    contain no parameter prior, so this procedure is deliberately not called
    MAP.
    """

    def __init__(self, config: StiffnessParticleScanConfig) -> None:
        self.config = config
        self.records: Deque[StiffnessParticleRecord] = deque(maxlen=self._window_size())
        self.step_count = 0
        self.record_count = 0
        self.last_scan_record_count = 0
        self.last_accept_record_count = 0
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

    def snapshot(self) -> "StiffnessParticleScanSupervisor":
        snapshot = StiffnessParticleScanSupervisor(deepcopy(self.config))
        snapshot.records = deque(
            (
                StiffnessParticleRecord(
                    theta_cmd_sent=record.theta_cmd_sent.copy(),
                    A_map={fid: A_f.copy() for fid, A_f in record.A_map.items()},
                    theta_init_eq_pred=None
                    if record.theta_init_eq_pred is None
                    else record.theta_init_eq_pred.copy(),
                    stamp=record.stamp,
                    information=None
                    if record.information is None
                    else record.information.copy(),
                )
                for record in self.records
            ),
            maxlen=self.records.maxlen,
        )
        snapshot.step_count = int(self.step_count)
        snapshot.record_count = int(self.record_count)
        snapshot.last_scan_record_count = int(self.last_scan_record_count)
        snapshot.last_accept_record_count = int(self.last_accept_record_count)
        snapshot.last_result = self.last_result
        return snapshot

    def readiness_reason(self) -> str:
        if not bool(self.config.enabled):
            return "disabled"
        if len(self.records) < self._window_size():
            return "window_not_full"
        if self.last_accept_record_count > 0:
            records_since_accept = self.record_count - self.last_accept_record_count
            if records_since_accept < max(0, int(self.config.cooldown)):
                return "cooldown"
        if self.last_scan_record_count > 0:
            records_since_scan = self.record_count - self.last_scan_record_count
            if records_since_scan < max(1, int(self.config.period)):
                return "period_skip"
        return "ready"

    def register_result(self, result: StiffnessParticleScanResult, applied: bool = False) -> None:
        """Persist bookkeeping from an asynchronous snapshot scan."""
        if result is None:
            return
        self.step_count = max(self.step_count, int(result.debug.get("step_count", 0)))
        result_record_count = int(result.debug.get("record_count", self.record_count))
        if result.attempted:
            self.last_scan_record_count = max(self.last_scan_record_count, result_record_count)
        if result.accepted and applied:
            self.last_accept_record_count = max(self.last_accept_record_count, result_record_count)
        self.last_result = result

    def result_freshness(
        self,
        result: StiffnessParticleScanResult,
        x_current: np.ndarray,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        result_record_count = int(result.debug.get("record_count", self.record_count))
        age_records = max(0, int(self.record_count) - result_record_count)
        max_age = int(self.config.max_result_age_records)
        x_now = np.asarray(x_current, dtype=float)
        if result.x_current is None or np.asarray(result.x_current).shape != x_now.shape:
            drift = np.inf
        else:
            drift = float(np.max(np.abs(x_now - np.asarray(result.x_current, dtype=float))))
        debug = {
            "result_age_records": age_records,
            "max_result_age_records": max_age,
            "estimator_drift": drift,
            "max_estimator_drift": float(self.config.max_estimator_drift),
        }
        if max_age >= 0 and age_records > max_age:
            return False, "stale_result_records", debug
        max_drift = float(self.config.max_estimator_drift)
        if max_drift >= 0.0 and (not np.isfinite(drift) or drift > max_drift):
            return False, "stale_result_estimator_drift", debug
        return True, "fresh", debug

    def status_result(
        self,
        reason: str,
        attempted: bool = False,
        x_current: Optional[np.ndarray] = None,
        debug_extra: Optional[Dict[str, Any]] = None,
    ) -> StiffnessParticleScanResult:
        return self._result(
            attempted=bool(attempted),
            accepted=False,
            reason=reason,
            x_current=x_current,
            x_best=None,
            score_current=-np.inf,
            score_best=-np.inf,
            gain_per_obs=0.0,
            active_indices=np.array([], dtype=int),
            candidate_count=0,
            theta_eq_best=None,
            debug_extra={} if debug_extra is None else dict(debug_extra),
        )

    def add_record(
        self,
        theta_cmd_sent: np.ndarray,
        A_map: Dict[int, np.ndarray],
        theta_init_eq_pred: Optional[np.ndarray],
        stamp: Optional[float],
        information: Optional[np.ndarray] = None,
    ) -> None:
        copied_map = {fid: np.asarray(A_f, dtype=float).copy() for fid, A_f in A_map.items()}
        copied_information = None
        if information is not None:
            copied_information = np.asarray(information, dtype=float).copy()
        record = StiffnessParticleRecord(
            theta_cmd_sent=np.asarray(theta_cmd_sent, dtype=float).copy(),
            A_map=copied_map,
            theta_init_eq_pred=None
            if theta_init_eq_pred is None
            else np.asarray(theta_init_eq_pred, dtype=float).copy(),
            stamp=stamp,
            information=copied_information,
        )
        self.records.append(record)
        self.record_count += 1

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
            "record_count": int(self.record_count),
            "last_scan_record_count": int(self.last_scan_record_count),
            "last_accept_record_count": int(self.last_accept_record_count),
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

    def _validation_split(
        self,
    ) -> Tuple[List[StiffnessParticleRecord], List[StiffnessParticleRecord]]:
        records = list(self.records)
        if len(records) < 2:
            return [], []
        fraction = float(np.clip(float(self.config.validation_fraction), 0.0, 0.9))
        validation_count = max(
            1,
            int(self.config.min_validation_records),
            int(np.ceil(fraction * len(records))),
        )
        validation_count = min(len(records) - 1, validation_count)
        split = len(records) - validation_count
        return records[:split], records[split:]

    def _active_directions(
        self,
        records: Sequence[StiffnessParticleRecord],
        dimension: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        information = np.zeros((dimension, dimension), dtype=float)
        valid_count = 0
        for record in records:
            if record.information is None:
                continue
            info = np.asarray(record.information, dtype=float)
            if info.shape != information.shape or not np.all(np.isfinite(info)):
                continue
            information += 0.5 * (info + info.T)
            valid_count += 1

        debug: Dict[str, Any] = {
            "information_record_count": int(valid_count),
            "information_required_record_count": int(len(records)),
            "information_sum": information.copy(),
        }
        if bool(self.config.require_information) and valid_count < len(records):
            debug["information_reason"] = "missing_or_invalid_information"
            return np.zeros((dimension, 0), dtype=float), debug
        if valid_count == 0:
            debug["information_reason"] = "no_information"
            return np.zeros((dimension, 0), dtype=float), debug

        raw_eigvals, eigvecs = np.linalg.eigh(0.5 * (information + information.T))
        eigvals = np.maximum(raw_eigvals, 0.0)
        lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
        threshold = max(
            float(self.config.info_abs),
            float(self.config.info_rcond) * max(lam_max, 0.0),
        )
        keep_indices = np.flatnonzero(eigvals > threshold)
        keep_indices = keep_indices[np.argsort(-eigvals[keep_indices])]
        keep_indices = keep_indices[: max(0, int(self.config.max_active_dims))]
        directions = eigvecs[:, keep_indices].copy()
        # Eigenvector signs are arbitrary. Canonical signs keep candidates and
        # diagnostics deterministic across LAPACK implementations.
        for column in range(directions.shape[1]):
            direction = directions[:, column]
            pivot = int(np.argmax(np.abs(direction)))
            if direction[pivot] < 0.0:
                directions[:, column] *= -1.0

        debug.update(
            {
                "information_raw_eigvals": raw_eigvals.copy(),
                "information_eigvals": eigvals.copy(),
                "information_threshold": float(threshold),
                "active_direction_eigvals": eigvals[keep_indices].copy(),
                "active_directions": directions.copy(),
                "active_direction_count": int(directions.shape[1]),
            }
        )
        return directions, debug

    def _make_direction_candidates(
        self,
        x_current: np.ndarray,
        active_directions: np.ndarray,
        kp_lim: Tuple[float, float],
    ) -> List[np.ndarray]:
        kp_min, kp_max = (float(v) for v in kp_lim)
        if kp_min <= 0.0 or kp_max < kp_min:
            return []
        grid_size = int(self.config.grid_size)
        max_log_jump = float(self.config.max_log_jump)
        if grid_size <= 0 or not np.isfinite(max_log_jump) or max_log_jump <= 0.0:
            return []

        log_min = float(np.log(kp_min))
        log_max = float(np.log(kp_max))
        x_base = np.clip(np.asarray(x_current, dtype=float), log_min, log_max)
        directions = np.asarray(active_directions, dtype=float)
        if directions.ndim != 2 or directions.shape[0] != x_base.size:
            return []

        candidates: List[np.ndarray] = []
        seen = set()

        def add_candidate(x: np.ndarray) -> None:
            x_clip = np.clip(np.asarray(x, dtype=float), log_min, log_max)
            if np.max(np.abs(x_clip - x_base)) > max_log_jump + 1.0e-12:
                return
            key = tuple(np.round(x_clip, decimals=12))
            if key not in seen:
                seen.add(key)
                candidates.append(x_clip.copy())

        add_candidate(x_base)
        for column in range(directions.shape[1]):
            direction = directions[:, column]
            if not np.all(np.isfinite(direction)):
                continue
            max_component = float(np.max(np.abs(direction))) if direction.size else 0.0
            if max_component <= 0.0:
                continue
            distance_limit = max_log_jump / max_component
            for distance in np.linspace(-distance_limit, distance_limit, grid_size):
                add_candidate(x_base + float(distance) * direction)
        return candidates

    def _make_axis_candidates(
        self,
        x_current: np.ndarray,
        active_indices: np.ndarray,
        kp_lim: Tuple[float, float],
    ) -> List[np.ndarray]:
        """Backward-compatible helper; the scan itself uses information directions."""
        x = np.asarray(x_current, dtype=float)
        directions = np.zeros((x.size, len(np.asarray(active_indices).reshape(-1))), dtype=float)
        column = 0
        for index in np.asarray(active_indices, dtype=int).reshape(-1):
            if 0 <= index < x.size:
                directions[int(index), column] = 1.0
            column += 1
        return self._make_direction_candidates(x, directions, kp_lim)

    def _score_candidate(
        self,
        estimator: MultiFrameStiffnessWEKF,
        x_candidate: np.ndarray,
        kp_lim: Tuple[float, float],
        records: Sequence[StiffnessParticleRecord],
        continuity_reference: Optional[Sequence[np.ndarray]] = None,
    ) -> Tuple[float, List[np.ndarray], Dict[str, Any]]:
        score = 0.0
        theta_equilibria: List[np.ndarray] = []
        errors: List[str] = []
        max_equilibrium_delta = 0.0

        for idx, record in enumerate(records):
            evaluation = estimator.evaluate_log_likelihood_at_x(
                x_eval=x_candidate,
                theta_cmd_sent=record.theta_cmd_sent,
                A_map=record.A_map,
                theta_init_eq_pred=record.theta_init_eq_pred,
                kp_lim=kp_lim,
            )
            if not evaluation.valid or not np.isfinite(evaluation.log_likelihood):
                errors.append(f"record_{idx}:{evaluation.error}")
                return -np.inf, [], {"errors": errors, "reason": "invalid_likelihood"}
            theta_eq = np.asarray(evaluation.theta_eq, dtype=float).copy()
            if continuity_reference is not None:
                reference = np.asarray(continuity_reference[idx], dtype=float)
                if theta_eq.shape != reference.shape or not np.all(np.isfinite(theta_eq)):
                    errors.append(f"record_{idx}:invalid_equilibrium_shape")
                    return -np.inf, [], {"errors": errors, "reason": "invalid_equilibrium"}
                equilibrium_delta = float(np.max(np.abs(theta_eq - reference))) if theta_eq.size else 0.0
                max_equilibrium_delta = max(max_equilibrium_delta, equilibrium_delta)
                jump_limit = float(self.config.max_equilibrium_jump)
                if jump_limit >= 0.0 and equilibrium_delta > jump_limit:
                    errors.append(
                        f"record_{idx}:equilibrium_jump:{equilibrium_delta:.9g}>{jump_limit:.9g}"
                    )
                    return -np.inf, [], {
                        "errors": errors,
                        "reason": "equilibrium_branch_discontinuity",
                        "max_equilibrium_delta": max_equilibrium_delta,
                    }
            score += float(evaluation.log_likelihood)
            theta_equilibria.append(theta_eq)

        return float(score), theta_equilibria, {
            "errors": errors,
            "max_equilibrium_delta": max_equilibrium_delta,
        }

    def maybe_scan(
        self,
        estimator: MultiFrameStiffnessWEKF,
        kp_lim: Optional[Tuple[float, float]],
    ) -> StiffnessParticleScanResult:
        self.step_count += 1
        empty_active = np.array([], dtype=int)

        readiness = self.readiness_reason()
        if readiness != "ready":
            self.last_result = self.status_result(
                reason=readiness,
                attempted=False,
                x_current=None if readiness == "disabled" else estimator.x_est,
            )
            return self.last_result

        if kp_lim is None:
            self.last_result = self.status_result(
                reason="missing_kp_lim",
                attempted=False,
                x_current=estimator.x_est,
            )
            return self.last_result

        train_records, validation_records = self._validation_split()
        if not train_records or not validation_records:
            self.last_result = self.status_result(
                reason="insufficient_records_for_holdout",
                attempted=False,
                x_current=estimator.x_est,
            )
            return self.last_result

        self.last_scan_record_count = int(self.record_count)
        x_current = np.asarray(estimator.x_est, dtype=float).copy()
        active_directions, information_debug = self._active_directions(
            train_records,
            x_current.size,
        )
        if active_directions.shape[1] == 0:
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="no_active_directions",
                x_current=x_current,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=empty_active,
                candidate_count=0,
                theta_eq_best=None,
                debug_extra=information_debug,
            )
            return self.last_result

        support = np.any(np.abs(active_directions) > 1.0e-10, axis=1)
        active_indices = np.flatnonzero(support).astype(int)
        candidates = self._make_direction_candidates(x_current, active_directions, kp_lim)
        if len(candidates) <= 1:
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
                candidate_count=len(candidates),
                theta_eq_best=None,
                debug_extra=information_debug,
            )
            return self.last_result

        train_current, train_current_eq, train_current_debug = self._score_candidate(
            estimator, x_current, kp_lim, train_records
        )
        validation_current, validation_current_eq, validation_current_debug = self._score_candidate(
            estimator, x_current, kp_lim, validation_records
        )
        if not np.isfinite(train_current) or not np.isfinite(validation_current):
            self.last_result = self._result(
                attempted=True,
                accepted=False,
                reason="nonfinite_current_score",
                x_current=x_current,
                x_best=None,
                score_current=-np.inf,
                score_best=-np.inf,
                gain_per_obs=0.0,
                active_indices=active_indices,
                candidate_count=len(candidates),
                theta_eq_best=None,
                debug_extra={
                    **information_debug,
                    "train_current_debug": train_current_debug,
                    "validation_current_debug": validation_current_debug,
                },
            )
            return self.last_result

        x_best = x_current.copy()
        train_best = train_current
        train_best_eq = train_current_eq
        candidate_errors: List[Dict[str, Any]] = []
        branch_rejections = 0
        for candidate in candidates:
            score, theta_eq, score_debug = self._score_candidate(
                estimator,
                candidate,
                kp_lim,
                train_records,
                continuity_reference=train_current_eq,
            )
            if not np.isfinite(score):
                if score_debug.get("reason") == "equilibrium_branch_discontinuity":
                    branch_rejections += 1
                if score_debug.get("errors"):
                    candidate_errors.append(score_debug)
                continue
            if score > train_best:
                train_best = score
                x_best = candidate.copy()
                train_best_eq = theta_eq

        train_gain_per_obs = (float(train_best) - float(train_current)) / len(train_records)
        max_jump = float(np.max(np.abs(x_best - x_current))) if x_best.size else 0.0
        min_gain = float(self.config.min_gain_per_obs)
        min_jump = max(0.0, float(self.config.min_log_jump))
        max_jump_limit = float(self.config.max_log_jump)

        prevalidation_reason: Optional[str] = None
        if train_best <= train_current:
            prevalidation_reason = (
                "equilibrium_branch_discontinuity" if branch_rejections else "no_training_improvement"
            )
        elif train_gain_per_obs < min_gain:
            prevalidation_reason = "training_gain_too_small"
        elif max_jump < min_jump:
            prevalidation_reason = "jump_too_small"
        elif max_jump > max_jump_limit + 1.0e-12:
            prevalidation_reason = "jump_too_large"

        validation_best = -np.inf
        validation_best_eq: List[np.ndarray] = []
        validation_best_debug: Dict[str, Any] = {}
        validation_gain_per_obs = -np.inf
        if prevalidation_reason is None:
            validation_best, validation_best_eq, validation_best_debug = self._score_candidate(
                estimator,
                x_best,
                kp_lim,
                validation_records,
                continuity_reference=validation_current_eq,
            )
            if not np.isfinite(validation_best):
                prevalidation_reason = validation_best_debug.get(
                    "reason", "nonfinite_validation_score"
                )
            else:
                validation_gain_per_obs = (
                    float(validation_best) - float(validation_current)
                ) / len(validation_records)
                validation_min_gain = self.config.min_validation_gain_per_obs
                if validation_min_gain is None:
                    validation_min_gain = min_gain
                if validation_gain_per_obs < float(validation_min_gain):
                    prevalidation_reason = "validation_gain_too_small"

        accepted = prevalidation_reason is None
        reason = "accepted" if accepted else str(prevalidation_reason)
        if accepted:
            score_current = float(train_current + validation_current)
            score_best = float(train_best + validation_best)
            theta_eq_best = validation_best_eq[-1].copy() if validation_best_eq else None
        else:
            score_current = float(train_current + validation_current)
            score_best = (
                float(train_best + validation_best)
                if np.isfinite(validation_best)
                else float(train_best)
            )
            theta_eq_best = None
        gain_per_obs = (score_best - score_current) / max(1, len(self.records))

        debug_extra = {
            **information_debug,
            "training_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "training_score_current": float(train_current),
            "training_score_best": float(train_best),
            "training_gain_per_obs": float(train_gain_per_obs),
            "validation_score_current": float(validation_current),
            "validation_score_best": float(validation_best),
            "validation_gain_per_obs": float(validation_gain_per_obs),
            "min_gain_per_obs": min_gain,
            "max_jump": max_jump,
            "min_log_jump": min_jump,
            "max_log_jump": max_jump_limit,
            "branch_rejection_count": int(branch_rejections),
            "candidate_errors": candidate_errors,
            "training_best_equilibrium_count": len(train_best_eq),
            "validation_best_debug": validation_best_debug,
        }
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
            debug_extra=debug_extra,
        )
        return self.last_result
