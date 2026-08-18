#!/usr/bin/env python3
"""Shared first-order-thrust-lag estimator and JSON contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from grape_param_estim.system import ActuatorParameters  # noqa: E402
from single_bag_savgol_core import (  # noqa: E402
    ACTUATOR_ACTIVE_SET_NAMES,
    ActuatorHistory,
    COMMON_SCALE_DIRECTION,
    PHYSICAL_CHART_LABELS,
    PHYSICAL_DIMENSION,
    SiParameterChart,
    SingleBagDynamicsProblem,
    common_scale_quotient_basis,
)


ESTIMATE_SCHEMA = "grape-param-estim/first-order-lag-estimate/v1"
GAIN_REGION_SCHEMA = "grape-param-estim/first-order-lag-pid-gain-region/v1"
COVARIANCE_NAMES = (
    "naive",
    "overlap_corrected",
    "conservative_fusion",
)
PID_GROUPS = ("xy", "z", "roll_pitch", "yaw")
PID_GAIN_NAMES = ("p_gain", "i_gain", "d_gain")
CASE_BAG_JSONS = {
    "failure1": _MINIMAL / "bag_jsons" / "single_rosbag_1.json",
    "failure2": _MINIMAL / "bag_jsons" / "single_rosbag_2.json",
    "success": _MINIMAL / "bag_jsons" / "single_rosbag_succeeded.json",
}
CASE_OUTCOMES = {
    "failure1": "crashed",
    "failure2": "crashed",
    "success": "successful",
}


@dataclass(frozen=True)
class FirstOrderThrustHistory:
    time: np.ndarray
    thrust: np.ndarray
    log_tau_jacobian: np.ndarray
    active_set_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        thrust = np.asarray(self.thrust, dtype=float)
        jacobian = np.asarray(self.log_tau_jacobian, dtype=float)
        if (
            time.ndim != 1
            or time.size == 0
            or thrust.shape != (time.size, 4)
            or jacobian.shape != thrust.shape
            or np.any(~np.isfinite(time))
            or np.any(~np.isfinite(thrust))
            or np.any(~np.isfinite(jacobian))
            or (time.size > 1 and np.any(np.diff(time) <= 0.0))
        ):
            raise ValueError("first-order thrust history is invalid")
        time_copy = time.copy()
        thrust_copy = thrust.copy()
        jacobian_copy = jacobian.copy()
        for value in (time_copy, thrust_copy, jacobian_copy):
            value.setflags(write=False)
        object.__setattr__(self, "time", time_copy)
        object.__setattr__(self, "thrust", thrust_copy)
        object.__setattr__(self, "log_tau_jacobian", jacobian_copy)
        object.__setattr__(
            self,
            "active_set_counts",
            {str(key): int(value) for key, value in self.active_set_counts.items()},
        )


def _first_order_step(
    state: np.ndarray,
    state_log_tau_jacobian: np.ndarray,
    target: np.ndarray,
    duration: float,
    time_constant: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(duration)
    tau = float(time_constant)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("first-order propagation duration is invalid")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("first-order time constant must be finite and positive")
    if dt == 0.0:
        return state.copy(), state_log_tau_jacobian.copy()
    decay = float(np.exp(-dt / tau))
    decay_log_tau_derivative = decay * dt / tau
    difference = state - target
    next_state = target + decay * difference
    next_jacobian = (
        decay * state_log_tau_jacobian
        + decay_log_tau_derivative * difference
    )
    return next_state, next_jacobian


def first_order_zoh_thrust_history(
    *,
    command_times: Sequence[float],
    command_values: np.ndarray,
    evaluation_times: Sequence[float],
    time_constant: float,
    minimum_thrust: float,
    maximum_thrust: float,
) -> FirstOrderThrustHistory:
    """Exact ZOH-input first-order response and derivative w.r.t. log(tau).

    The oldest retained command is used as the actuator state at the beginning
    of retained prehistory.  The existing bag adapter supplies causal prehistory
    before the estimator interval, so the selected interval is reached only
    after propagating through the retained command history.
    """

    command_time = np.asarray(command_times, dtype=float)
    command = np.asarray(command_values, dtype=float)
    query = np.asarray(evaluation_times, dtype=float)
    lower = float(minimum_thrust)
    upper = float(maximum_thrust)
    tau = float(time_constant)
    if (
        command_time.ndim != 1
        or command_time.size == 0
        or command.shape != (command_time.size, 4)
        or query.ndim != 1
        or query.size == 0
        or np.any(~np.isfinite(command_time))
        or np.any(~np.isfinite(command))
        or np.any(~np.isfinite(query))
        or (command_time.size > 1 and np.any(np.diff(command_time) <= 0.0))
        or (query.size > 1 and np.any(np.diff(query) <= 0.0))
        or query[0] < command_time[0]
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
        or not np.isfinite(tau)
        or tau <= 0.0
    ):
        raise ValueError("first-order ZOH history inputs are invalid")

    clipped = np.clip(command, lower, upper)
    lower_count = int(np.count_nonzero(command <= lower))
    upper_count = int(np.count_nonzero(command >= upper))
    output = np.empty((query.size, 4), dtype=float)
    derivative = np.empty_like(output)

    current_time = float(command_time[0])
    current_target = clipped[0].copy()
    current_state = current_target.copy()
    current_derivative = np.zeros(4, dtype=float)
    command_index = 0

    for row, evaluation_time in enumerate(query):
        evaluation = float(evaluation_time)
        while (
            command_index + 1 < command_time.size
            and command_time[command_index + 1] <= evaluation
        ):
            switch = float(command_time[command_index + 1])
            current_state, current_derivative = _first_order_step(
                current_state,
                current_derivative,
                current_target,
                switch - current_time,
                tau,
            )
            command_index += 1
            current_time = switch
            current_target = clipped[command_index].copy()
        current_state, current_derivative = _first_order_step(
            current_state,
            current_derivative,
            current_target,
            evaluation - current_time,
            tau,
        )
        current_time = evaluation
        output[row] = current_state
        derivative[row] = current_derivative

    counts = {
        name: 0 for name in ACTUATOR_ACTIVE_SET_NAMES
    }
    counts["thrust_command_lower"] = lower_count
    counts["thrust_command_upper"] = upper_count
    return FirstOrderThrustHistory(query, output, derivative, counts)


class FirstOrderLagDynamicsProblem(SingleBagDynamicsProblem):
    """Existing SG/Newton-Euler problem with pure delay replaced by one tau."""

    def actuator_history(
        self,
        rotor_lag_seconds: float,
        *,
        command_mode: str,
        epsilon: Optional[float] = None,
    ) -> ActuatorHistory:
        # The inherited objective calls its fifteenth coordinate
        # ``rotor_lag_seconds``.  In this isolated model that coordinate is
        # eta=log(tau), and the derivative column is therefore d/d eta.
        del epsilon
        if command_mode not in ("strict", "smooth"):
            raise ValueError("unknown command mode")
        log_tau = float(rotor_lag_seconds)
        if not np.isfinite(log_tau):
            raise ValueError("log time constant must be finite")
        tau = float(math.exp(log_tau))
        response = first_order_zoh_thrust_history(
            command_times=self.dataset.rotor_history.times,
            command_values=self.dataset.rotor_history.values,
            evaluation_times=self.dataset.time,
            time_constant=tau,
            minimum_thrust=self.actuator_parameters.minimum_thrust,
            maximum_thrust=self.actuator_parameters.maximum_thrust,
        )
        if self.gimbal_source == "measured_sg":
            gimbal = self.dataset.gimbal_sg_angle
        elif self.gimbal_source == "measured_linear":
            gimbal = self.dataset.gimbal_linear_angle
        else:
            gimbal = self.gimbal_replay.angle
        counts = dict(response.active_set_counts)
        for name in ACTUATOR_ACTIVE_SET_NAMES:
            if name.startswith("gimbal_"):
                counts[name] = int(
                    self.gimbal_replay.active_set_counts.get(name, 0)
                )
        return ActuatorHistory(
            time=self.dataset.time,
            actual_thrust=response.thrust,
            actual_gimbal=gimbal,
            actual_thrust_lag_jacobian=response.log_tau_jacobian,
            active_set_counts=counts,
            gimbal_source=self.gimbal_source,
            command_mode="strict",
            strict_final=True,
        )

    def evaluate_first_order(
        self,
        physical_coordinate: Sequence[float],
        log_time_constant: float,
        *,
        reference: bool = False,
    ) -> Any:
        return self.evaluate_physical(
            physical_coordinate,
            float(log_time_constant),
            command_mode="strict",
            reference=reference,
        )


def quotient_distribution_payload(
    physical_coordinate: Sequence[float],
    covariance_result: Any,
) -> Mapping[str, Any]:
    coordinate = np.asarray(physical_coordinate, dtype=float)
    if coordinate.shape != (PHYSICAL_DIMENSION,):
        raise ValueError("physical coordinate must be 14-D")
    basis = common_scale_quotient_basis()
    result: dict[str, Any] = {
        "physical_chart_labels": list(PHYSICAL_CHART_LABELS),
        "physical_chart_coordinate": coordinate.tolist(),
        "common_scale_direction": COMMON_SCALE_DIRECTION.tolist(),
        "quotient_basis": basis.tolist(),
        "quotient_coordinate": (basis.T @ coordinate).tolist(),
        "covariances": {},
    }
    for name in COVARIANCE_NAMES:
        covariance = np.asarray(getattr(covariance_result, name), dtype=float)
        reduced = basis.T @ covariance @ basis
        reduced = 0.5 * (reduced + reduced.T)
        result["covariances"][name] = reduced.tolist()
    return result


def physical_point_payload(parameters: Any) -> Mapping[str, Any]:
    inertia = np.asarray(parameters.inertia, dtype=float)
    force = np.asarray(parameters.force_effectiveness, dtype=float)
    mass = float(parameters.mass)
    return {
        "mass_kg": mass,
        "inertia_kg_m2": inertia.tolist(),
        "cog_position_body_m": np.asarray(parameters.cog_offset).tolist(),
        "force_effectiveness": force.tolist(),
        "scale_free": {
            "inertia_over_mass_m2": (inertia / mass).tolist(),
            "cog_position_body_m": np.asarray(parameters.cog_offset).tolist(),
            "force_effectiveness_over_mass": (force / mass).tolist(),
        },
    }


def controller_snapshot_payload(flight: Any) -> Mapping[str, Any]:
    snapshot = flight.controller_snapshot
    groups = tuple(snapshot.groups)
    gains = np.asarray(snapshot.gains, dtype=float)
    if groups != PID_GROUPS or gains.shape != (4, 3):
        raise ValueError("recorded controller snapshot has an unexpected layout")
    nested = {
        group: {
            name: float(gains[row, column])
            for column, name in enumerate(PID_GAIN_NAMES)
        }
        for row, group in enumerate(groups)
    }
    return {
        "groups": list(groups),
        "gains": nested,
        "record_times": np.asarray(snapshot.record_times).tolist(),
        "pid_control_flags": np.asarray(snapshot.pid_control_flags).astype(bool).tolist(),
        "source_kinds": list(snapshot.source_kinds),
    }


def controller_period_payload(flight: Any) -> Mapping[str, Any]:
    stream = flight.rotor_command
    if stream is None:
        raise ValueError("flight has no rotor command stream")
    times = np.asarray(stream.record_times, dtype=float)
    gaps = np.diff(times)
    if gaps.size == 0 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
        raise ValueError("rotor command record times are not strictly increasing")
    return {
        "sample_count": int(times.size),
        "median_seconds": float(np.median(gaps)),
        "mean_seconds": float(np.mean(gaps)),
        "standard_deviation_seconds": float(np.std(gaps, ddof=0)),
        "min_seconds": float(np.min(gaps)),
        "max_seconds": float(np.max(gaps)),
    }


def load_estimate_json(path: Path) -> Mapping[str, Any]:
    import json

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("first-order estimate JSON root must be an object")
    if payload.get("schema") != ESTIMATE_SCHEMA:
        raise ValueError("unsupported first-order estimate schema")
    if payload.get("status") != "completed":
        raise ValueError("first-order estimate must be completed")
    return payload


def draw_quotient_coordinates(
    estimate: Mapping[str, Any],
    covariance_name: str,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    distribution = estimate["plant_distribution"]
    center = np.asarray(distribution["quotient_coordinate"], dtype=float)
    covariance = np.asarray(
        distribution["covariances"][covariance_name], dtype=float
    )
    if center.shape != (13,) or covariance.shape != (13, 13):
        raise ValueError("first-order quotient distribution has invalid shape")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 13.0 * np.finfo(float).eps * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError("first-order quotient covariance is not PSD")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(int(seed))
    standard = rng.standard_normal((int(sample_count), 13))
    return center[None, :] + standard @ factor.T


def quotient_to_scale_free_plants(
    estimate: Mapping[str, Any],
    quotient_coordinates: np.ndarray,
    vehicle_parameters: Any,
    scale_free_type: Any,
) -> tuple[Any, ...]:
    distribution = estimate["plant_distribution"]
    basis = np.asarray(distribution["quotient_basis"], dtype=float)
    selected = np.asarray(quotient_coordinates, dtype=float)
    if basis.shape != (14, 13) or selected.ndim != 2 or selected.shape[1] != 13:
        raise ValueError("quotient decode input has invalid shape")
    chart = SiParameterChart(vehicle_parameters)
    result = []
    for row in selected:
        parameters = chart.decode(basis @ row)
        result.append(
            scale_free_type(
                inertia_over_mass=np.asarray(parameters.inertia) / parameters.mass,
                cog_position_body=np.asarray(parameters.cog_offset),
                force_effectiveness_over_mass=(
                    np.asarray(parameters.force_effectiveness) / parameters.mass
                ),
                rotor_lag_seconds=0.0,
            )
        )
    return tuple(result)


def actuator_parameters_from_estimate(
    estimate: Mapping[str, Any],
) -> ActuatorParameters:
    actuator = estimate["actuator_model"]
    tau = float(actuator["thrust_time_constant_seconds"])
    return ActuatorParameters(
        thrust_time_constant=tau,
        gimbal_time_constant=float(actuator["gimbal_time_constant_seconds"]),
        delay=0.0,
        minimum_thrust=float(actuator["minimum_thrust"]),
        maximum_thrust=float(actuator["maximum_thrust"]),
        maximum_gimbal_angle=float(actuator["maximum_gimbal_angle"]),
        maximum_gimbal_rate=float(actuator["maximum_gimbal_rate"]),
    )
