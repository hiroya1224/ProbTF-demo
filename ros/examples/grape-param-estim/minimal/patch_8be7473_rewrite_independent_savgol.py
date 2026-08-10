#!/usr/bin/env python3
"""Rebuild the SG dynamics estimator from commit 8be7473 as a standalone module.

The generated deterministic_savgol_dynamics_estimator.py contains the downstream
rigid-body/optimizer/report implementation directly and does not import or
monkey-patch deterministic_spline_dynamics_estimator.py at runtime.

The command-lag implementation uses separate rotor/gimbal lags, measured publish
period defaults, overlapping 4/2/1/0.5-period smooth continuation, and adaptive
strict-ZOH 2-D search. No .bak files are created.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import py_compile
import re
import subprocess
import tempfile
import textwrap

EXPECTED = {
    "deterministic_savgol_dynamics_estimator.py": "46cdf94af9fbfeb35b1e21ce149b3178bad1f491",
    "smooth_command.py": "41b54d65e7e91f7960dbfe350e2d8328e73d3160",
    "savgol_dynamics_confidence.py": "7e9e2200ec7372a09fa75dbfd701bf40e053cb9f",
    "savgol_window_ablation.py": "35d2a87b7beec3ca18cfcfa01e714876c8ab17a6",
    "deterministic_savgol_dynamics_data_dictionary.md": "1eedcc6e6064de6ba192a5d8691c5e42a9ef113d",
    "deterministic_spline_dynamics_estimator.py": "948eb27b22e6c14087f1b2fff41f9ebe5586cdf0",
    "spline_dynamics_confidence.py": "c3b08eb0249854108571caef3954449621a81cd9",
    "../src/grape_param_estim/dynamics.py": "a7f8cf3c22435399a9974dd45bb74ecc57e12af8",
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def replace_first(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"{label}: expected at least one occurrence, found 0")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return out


def patch_smooth_command(text: str) -> str:
    prefix, marker, _ = text.partition("class QuinticSmoothZoh:")
    if not marker:
        raise RuntimeError("QuinticSmoothZoh class not found")
    cls = '''class QuinticSmoothZoh:
    """Differentiable surrogate of a causal ZOH command.

    The ZOH is written as its initial value plus command jumps.  Each Heaviside
    jump is replaced by a quintic smoothstep.  Transition supports are allowed
    to overlap deliberately, so broad continuation stages have a broad lag
    derivative.

    ``width_fraction`` is the transition half-width in units of the median
    recorded publish period.  It is not clipped to avoid overlap.
    """

    def __init__(self, times: Sequence[float], values: np.ndarray) -> None:
        sample_times = np.asarray(times, dtype=float)
        sample_values = np.asarray(values, dtype=float)
        if (
            sample_times.ndim != 1
            or sample_times.size < 1
            or sample_values.ndim != 2
            or sample_values.shape[0] != sample_times.size
            or sample_values.shape[1] < 1
            or np.any(~np.isfinite(sample_times))
            or np.any(~np.isfinite(sample_values))
            or np.any(np.diff(sample_times) < 0.0)
        ):
            raise ValueError("command history must be finite and time ordered")
        sample_times, sample_values = _deduplicate_last(sample_times, sample_values)
        if sample_times.size > 1 and np.any(np.diff(sample_times) <= 0.0):
            raise RuntimeError("deduplicated command times are not strict")
        self.times = sample_times
        self.values = sample_values
        self.dimension = sample_values.shape[1]
        self.median_period = (
            float(np.median(np.diff(sample_times))) if sample_times.size > 1 else 0.0
        )
        self.times.setflags(write=False)
        self.values.setflags(write=False)

    def transition_half_widths(self, width_fraction: float) -> np.ndarray:
        fraction = float(width_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("width fraction must be finite and positive")
        if self.times.size < 2:
            return np.empty(0, dtype=float)
        half_width = fraction * self.median_period
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError("transition half-width is invalid")
        return np.full(self.times.size - 1, half_width, dtype=float)

    def exact_zoh(self, time: float, delay: float) -> np.ndarray:
        query = float(time) - float(delay)
        if not np.isfinite(query):
            raise ValueError("command query must be finite")
        index = int(np.searchsorted(self.times, query, side="right") - 1)
        index = min(max(index, 0), self.times.size - 1)
        return self.values[index].copy()

    def evaluate(self, time: float, delay: float, width_fraction: float) -> CommandEvaluation:
        evaluation_time = float(time)
        lag = float(delay)
        if not np.isfinite(evaluation_time) or not np.isfinite(lag):
            raise ValueError("evaluation time and delay must be finite")
        if self.times.size < 2:
            return CommandEvaluation(
                value=self.values[0],
                delay_derivative=np.zeros(self.dimension, dtype=float),
            )
        fraction = float(width_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("width fraction must be finite and positive")
        half_width = fraction * self.median_period
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError("transition half-width is invalid")

        query = evaluation_time - lag
        transitions = self.times[1:]
        completed = int(np.searchsorted(transitions, query - half_width, side="right"))
        active_end = int(np.searchsorted(transitions, query + half_width, side="left"))
        value = self.values[completed].copy()
        delay_derivative = np.zeros(self.dimension, dtype=float)
        for local_index in range(completed, active_end):
            transition_index = local_index + 1
            transition_time = float(self.times[transition_index])
            q = (query - transition_time + half_width) / (2.0 * half_width)
            smooth = q**3 * (q * (6.0 * q - 15.0) + 10.0)
            smooth_derivative = 30.0 * q**2 * (1.0 - q) ** 2
            delta = self.values[transition_index] - self.values[transition_index - 1]
            value += delta * smooth
            delay_derivative += -delta * smooth_derivative / (2.0 * half_width)
        return CommandEvaluation(value=value, delay_derivative=delay_derivative)
'''
    return prefix + cls


def patch_dynamics(text: str) -> str:
    text = replace_once(
        text,
        "        self._inverse_inertia = np.linalg.pinv(parameters.inertia, hermitian=True)\n",
        "",
        "remove plant pinv",
    )
    return replace_once(
        text,
        "        angular_acceleration = self._inverse_inertia @ (\n"
        "            wrench[3:]\n"
        "            - np.cross(\n"
        "                state.angular_velocity,\n"
        "                self.parameters.inertia @ state.angular_velocity,\n"
        "            )\n"
        "        )\n",
        "        angular_acceleration = np.linalg.solve(\n"
        "            self.parameters.inertia,\n"
        "            wrench[3:]\n"
        "            - np.cross(\n"
        "                state.angular_velocity,\n"
        "                self.parameters.inertia @ state.angular_velocity,\n"
        "            ),\n"
        "        )\n",
        "plant solve inertia",
    )


def patch_base(text: str) -> str:
    text = replace_once(
        text,
        "@dataclass(frozen=True)\n"
        "class DynamicsSolution:\n"
        "    physical_coordinate: np.ndarray\n"
        "    delay_seconds: float\n"
        "    evaluation: JointDynamicsEvaluation\n"
        "    optimizer: Mapping[str, Any]\n",
        "@dataclass(frozen=True)\n"
        "class DynamicsSolution:\n"
        "    physical_coordinate: np.ndarray\n"
        "    delay_seconds: float\n"
        "    evaluation: JointDynamicsEvaluation\n"
        "    optimizer: Mapping[str, Any]\n"
        "    gimbal_delay_seconds: Optional[float] = None\n",
        "DynamicsSolution gimbal lag",
    )

    # forward_rollout supports an explicit gimbal lag.
    s = text.index("\ndef forward_rollout(")
    e = text.index("\n\n@dataclass(frozen=True)\nclass WrenchReplayEvaluation", s)
    b = text[s:e]
    b = replace_once(
        b,
        "    external_body_wrench: Optional[BodyWrenchHistory] = None,\n"
        ") -> baseline.Simulation:\n",
        "    external_body_wrench: Optional[BodyWrenchHistory] = None,\n"
        "    gimbal_delay: Optional[float] = None,\n"
        ") -> baseline.Simulation:\n",
        "forward signature",
    )
    b = replace_once(
        b,
        '        raise TypeError("external_body_wrench must be BodyWrenchHistory")\n\n',
        '        raise TypeError("external_body_wrench must be BodyWrenchHistory")\n'
        '    rotor_delay = float(delay)\n'
        '    gimbal_delay_value = rotor_delay if gimbal_delay is None else float(gimbal_delay)\n\n',
        "forward lag variables",
    )
    b = b.replace("        physical_coordinate,\n        delay,\n", "        physical_coordinate,\n        rotor_delay,\n", 1)
    b = replace_once(
        b,
        "    actuators = ActuatorState(\n"
        "        thrust=bag.rotor_history.exact_zoh(\n"
        "            float(direct.internal_time[0]), delay\n"
        "        ),\n"
        "        gimbal_angle=bag.initial_gimbal,\n"
        "    )\n",
        "    actuators = ActuatorState(\n"
        "        thrust=bag.rotor_history.exact_zoh(\n"
        "            float(direct.internal_time[0]), rotor_delay\n"
        "        ),\n"
        "        gimbal_angle=bag.gimbal_history.exact_zoh(\n"
        "            float(direct.internal_time[0]), gimbal_delay_value\n"
        "        ),\n"
        "    )\n",
        "forward initial actuators",
    )
    b = replace_once(
        b,
        "        command = smooth._command(\n"
        "            bag.rotor_history.exact_zoh(midpoint, delay),\n"
        "            bag.gimbal_history.exact_zoh(midpoint, delay),\n"
        "        )\n",
        "        command = smooth._command(\n"
        "            bag.rotor_history.exact_zoh(midpoint, rotor_delay),\n"
        "            bag.gimbal_history.exact_zoh(midpoint, gimbal_delay_value),\n"
        "        )\n",
        "forward commands",
    )
    text = text[:s] + b + text[e:]

    # Wrench replay supports lag pair and no ordinary matrix inverse.
    s = text.index("\nclass WrenchReplayProblem:")
    e = text.index("\n\ndef _solve_wrench_replay(", s)
    b = text[s:e]
    b = replace_once(
        b,
        "        reference_parameters: VehicleParameters,\n"
        "    ) -> None:\n",
        "        reference_parameters: VehicleParameters,\n"
        "        gimbal_delay: Optional[float] = None,\n"
        "    ) -> None:\n",
        "replay signature",
    )
    b = replace_once(
        b,
        "        self.delay = float(delay)\n"
        "        self.dynamics_evaluation = dynamics_evaluation\n",
        "        self.delay = float(delay)\n"
        "        self.gimbal_delay = self.delay if gimbal_delay is None else float(gimbal_delay)\n"
        "        self.dynamics_evaluation = dynamics_evaluation\n",
        "replay lag pair",
    )
    b = replace_once(
        b,
        "            or self.delay < 0.0\n"
        "        ):\n",
        "            or self.delay < 0.0\n"
        "            or not np.isfinite(self.gimbal_delay)\n"
        "            or self.gimbal_delay < 0.0\n"
        "        ):\n",
        "replay lag validation",
    )
    b = replace_once(
        b,
        "        self.inverse_inertia = np.linalg.inv(\n"
        "            self.parameters.inertia\n"
        "        )\n",
        "        self.inverse_inertia = np.linalg.solve(\n"
        "            self.parameters.inertia, np.eye(3)\n"
        "        )\n",
        "replay solve inertia",
    )
    b = replace_once(
        b,
        "                self.bag.gimbal_history.exact_zoh(\n"
        "                    midpoint, self.delay\n"
        "                ),\n",
        "                self.bag.gimbal_history.exact_zoh(\n"
        "                    midpoint, self.gimbal_delay\n"
        "                ),\n",
        "replay gimbal lag",
    )
    text = text[:s] + b + text[e:]

    s = text.index("\ndef _solve_wrench_replay(")
    e = text.index("\ndef _orientation_errors(", s)
    b = text[s:e]
    b = replace_once(
        b,
        "    reference_parameters: VehicleParameters,\n"
        ") -> tuple[\n",
        "    reference_parameters: VehicleParameters,\n"
        "    gimbal_delay: Optional[float] = None,\n"
        ") -> tuple[\n",
        "solve replay signature",
    )
    b = replace_once(
        b,
        "        dynamics_evaluation,\n"
        "        reference_parameters,\n"
        "    )\n",
        "        dynamics_evaluation,\n"
        "        reference_parameters,\n"
        "        gimbal_delay=gimbal_delay,\n"
        "    )\n",
        "solve replay constructor",
    )
    text = text[:s] + b + text[e:]

    # Hook an SG-specific split-lag solver into base.run, leaving the scalar
    # spline path intact.
    physical_marker = (
        "    physical_lower, physical_upper = _physical_bounds(\n"
        "        initial_physical_coordinate\n"
        "    )\n"
    )
    p = text.index(physical_marker)
    scalar_start = p + len(physical_marker)
    scalar_end = text.index("\n\n    output_directory = (", scalar_start)
    scalar = text[scalar_start:scalar_end]
    split = physical_marker + (
        "    split_lag_solver = getattr(arguments, \"split_command_lag_solver\", None)\n"
        "    command_lag_search = None\n"
        "    if split_lag_solver is not None:\n"
        "        split_lag_result = split_lag_solver(\n"
        "            problem, initial_physical_coordinate, physical_lower, physical_upper, arguments\n"
        "        )\n"
        "        selected = split_lag_result[\"selected_solution\"]\n"
        "        command_lag_search = {\n"
        "            key: value for key, value in split_lag_result.items()\n"
        "            if key != \"selected_solution\"\n"
        "        }\n"
        "        smooth_stage_payloads = command_lag_search[\"smooth_stages\"]\n"
        "        strict_payloads = command_lag_search[\"strict_candidates\"]\n"
        "        smooth_delay = command_lag_search[\"smooth_result\"][\"rotor_delay_seconds\"]\n"
        "        candidate_delays = np.empty(0, dtype=float)\n"
        "        screening_costs = np.empty(0, dtype=float)\n"
        "        strict_solutions = [selected]\n"
        "        print(\n"
        "            \"selected strict lags rotor={:.6f}s gimbal={:.6f}s; producing reconstruction and reports\".format(\n"
        "                selected.delay_seconds, selected.gimbal_delay_seconds\n"
        "            ), flush=True\n"
        "        )\n"
        "    else:\n"
        + textwrap.indent(scalar, "    ")
    )
    text = text[:p] + split + text[scalar_end:]

    text = replace_once(
        text,
        "    selected_parameters = selected.evaluation.decoded.parameters\n"
        "    for index, bag in enumerate(bags):\n",
        "    selected_parameters = selected.evaluation.decoded.parameters\n"
        "    selected_gimbal_delay = (\n"
        "        selected.delay_seconds if selected.gimbal_delay_seconds is None else float(selected.gimbal_delay_seconds)\n"
        "    )\n"
        "    initial_gimbal_delay = float(getattr(arguments, \"initial_gimbal_delay\", initial_delay))\n"
        "    for index, bag in enumerate(bags):\n",
        "run lag pair variables",
    )
    text = replace_once(
        text,
        "            arguments,\n"
        "            reference_parameters,\n"
        "        )\n"
        "        external_wrench_time = (\n",
        "            arguments,\n"
        "            reference_parameters,\n"
        "            gimbal_delay=selected_gimbal_delay,\n"
        "        )\n"
        "        external_wrench_time = (\n",
        "run replay gimbal lag",
    )
    text = replace_once(
        text,
        "            selected.delay_seconds,\n"
        "            reference_parameters,\n"
        "        )\n"
        "        reference_rollout = forward_rollout(\n",
        "            selected.delay_seconds,\n"
        "            reference_parameters,\n"
        "            gimbal_delay=selected_gimbal_delay,\n"
        "        )\n"
        "        reference_rollout = forward_rollout(\n",
        "run estimated rollout pair",
    )
    text = replace_once(
        text,
        "            initial_delay,\n"
        "            reference_parameters,\n"
        "        )\n"
        "        observations = bag.direct_problem.observations\n",
        "            initial_delay,\n"
        "            reference_parameters,\n"
        "            gimbal_delay=initial_gimbal_delay,\n"
        "        )\n"
        "        observations = bag.direct_problem.observations\n",
        "run reference rollout pair",
    )
    text = replace_once(
        text,
        '            "shared_delay_seconds": selected.delay_seconds,\n',
        '            "shared_delay_seconds": selected.delay_seconds,\n'
        '            "shared_rotor_delay_seconds": selected.delay_seconds,\n'
        '            "shared_gimbal_delay_seconds": selected_gimbal_delay,\n',
        "bag lag pair fields",
    )

    old_report = (
        "    _write_delay_profile_pdf(\n"
        "        output_directory / \"delay_profile.pdf\",\n"
        "        smooth_delay,\n"
        "        candidate_delays,\n"
        "        screening_costs,\n"
        "        strict_solutions,\n"
        "        selected,\n"
        "    )\n"
    )
    text = replace_once(
        text,
        old_report,
        "    split_writer = getattr(arguments, \"split_command_lag_report_writer\", None)\n"
        "    if split_writer is not None:\n"
        "        split_writer(output_directory, command_lag_search)\n"
        "    else:\n" + textwrap.indent(old_report, "    "),
        "custom lag report",
    )

    text = replace_once(
        text,
        '            "delay_seconds": initial_delay,\n',
        '            "delay_seconds": initial_delay,\n'
        '            "rotor_delay_seconds": initial_delay,\n'
        '            "gimbal_delay_seconds": initial_gimbal_delay,\n',
        "root initial pair",
    )
    text = replace_once(
        text,
        '        "strict_zoh_polish": strict_payloads,\n',
        '        "strict_zoh_polish": strict_payloads,\n'
        '        "command_lag_search": command_lag_search,\n',
        "root lag search",
    )
    text = replace_once(
        text,
        '            "delay_seconds": selected.delay_seconds,\n'
        '            "parameters": baseline._physical_parameters(selected_parameters),\n',
        '            "delay_seconds": selected.delay_seconds,\n'
        '            "rotor_delay_seconds": selected.delay_seconds,\n'
        '            "gimbal_delay_seconds": selected_gimbal_delay,\n'
        '            "parameters": baseline._physical_parameters(selected_parameters),\n',
        "selection pair",
    )
    return text


def patch_legacy_confidence(text: str) -> str:
    text = replace_once(
        text,
        "    reference_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        np.zeros(\n"
        "            deterministic.PHYSICAL_DIMENSION,\n"
        "            dtype=float,\n"
        "        ),\n"
        "        initial_delay,\n"
        "        reference_parameters,\n"
        "    )\n",
        "    selected_gimbal_delay = (\n"
        "        selected.delay_seconds if getattr(selected, \"gimbal_delay_seconds\", None) is None\n"
        "        else float(selected.gimbal_delay_seconds)\n"
        "    )\n"
        "    initial_gimbal_delay = float(getattr(arguments, \"initial_gimbal_delay\", initial_delay))\n"
        "    reference_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        np.zeros(deterministic.PHYSICAL_DIMENSION, dtype=float),\n"
        "        initial_delay,\n"
        "        reference_parameters,\n"
        "        gimbal_delay=initial_gimbal_delay,\n"
        "    )\n",
        "confidence reference pair",
    )
    text = replace_once(
        text,
        "    parameter_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        selected.physical_coordinate,\n"
        "        selected.delay_seconds,\n"
        "        reference_parameters,\n"
        "    )\n",
        "    parameter_rollout = deterministic.forward_rollout(\n"
        "        bag, selected.physical_coordinate, selected.delay_seconds, reference_parameters,\n"
        "        gimbal_delay=selected_gimbal_delay,\n"
        "    )\n",
        "confidence parameter pair",
    )
    text = replace_once(
        text,
        "        arguments,\n"
        "        reference_parameters,\n"
        "    )\n"
        "    replay_rollout = wrench_evaluation.simulation\n",
        "        arguments,\n"
        "        reference_parameters,\n"
        "        gimbal_delay=selected_gimbal_delay,\n"
        "    )\n"
        "    replay_rollout = wrench_evaluation.simulation\n",
        "confidence replay pair",
    )
    return text


def patch_sg(text: str) -> str:
    text = replace_once(
        text,
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v2"',
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v3"',
        "SG schema",
    )
    text = replace_once(
        text,
        "PHYSICAL_DIMENSION = base.PHYSICAL_DIMENSION\n"
        "GLOBAL_DIMENSION = base.GLOBAL_DIMENSION\n"
        "DELAY_INDEX = base.DELAY_INDEX\n",
        "PHYSICAL_DIMENSION = base.PHYSICAL_DIMENSION\n"
        "ROTOR_DELAY_INDEX = PHYSICAL_DIMENSION\n"
        "GIMBAL_DELAY_INDEX = PHYSICAL_DIMENSION + 1\n"
        "GLOBAL_DIMENSION = PHYSICAL_DIMENSION + 2\n"
        "DELAY_INDEX = ROTOR_DELAY_INDEX\n",
        "SG lag dimensions",
    )
    text = replace_once(
        text,
        "_ACTIVE_WINDOW_SECONDS: Optional[float] = None\n"
        "_ACTIVE_BAGS: dict[str, base.BagSplineData] = {}\n",
        "_ACTIVE_WINDOW_SECONDS: Optional[float] = None\n"
        "_ACTIVE_ROTOR_PERIOD_SECONDS: Optional[float] = None\n"
        "_ACTIVE_GIMBAL_PERIOD_SECONDS: Optional[float] = None\n"
        "_ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS: Optional[float] = None\n"
        "_ACTIVE_BAGS: dict[str, base.BagSplineData] = {}\n",
        "SG lag globals",
    )

    # Replace residual-wrench objective subclass with acceleration objective plus split lag.
    start = text.index("\ndef _reference_wrench_scaling(")
    end = text.index("\ndef _jacobian_spectrum(", start)
    split_class = '''
class SplineDynamicsProblem(base.SplineDynamicsProblem):
    """Acceleration-residual SG problem with separate rotor/gimbal lags."""

    def _actuator_series(self, bag, decoded, parameter_jacobian, dimension, smooth_mode, width_fraction):
        time_axis = bag.collocation_time
        rotor_delay = float(self._active_rotor_delay)
        gimbal_delay = float(self._active_gimbal_delay)
        if smooth_mode:
            rotor0 = bag.rotor_history.evaluate(float(time_axis[0]), rotor_delay, width_fraction)
            gimbal0 = bag.gimbal_history.evaluate(float(time_axis[0]), gimbal_delay, width_fraction)
            initial_command = smooth._command(rotor0.value, gimbal0.value)
        else:
            initial_command = smooth._command(
                bag.rotor_history.exact_zoh(float(time_axis[0]), rotor_delay),
                bag.gimbal_history.exact_zoh(float(time_axis[0]), gimbal_delay),
            )
        state = base.ActuatorState(
            thrust=initial_command.thrust,
            gimbal_angle=initial_command.gimbal_angle,
        )
        sensitivity = np.zeros((8, dimension), dtype=float)
        if smooth_mode:
            sensitivity[:4, ROTOR_DELAY_INDEX] = rotor0.delay_derivative
            sensitivity[4:, GIMBAL_DELAY_INDEX] = gimbal0.delay_derivative
        thrust = np.empty((time_axis.size, 4), dtype=float)
        gimbal = np.empty((time_axis.size, 4), dtype=float)
        state_jacobian = np.empty((time_axis.size, 8, dimension), dtype=float)
        for index, sample_time in enumerate(time_axis):
            thrust[index] = state.thrust
            gimbal[index] = state.gimbal_angle
            state_jacobian[index] = sensitivity
            if index == time_axis.size - 1:
                break
            dt = float(time_axis[index + 1] - sample_time)
            midpoint = float(sample_time + 0.5 * dt)
            command_sensitivity = np.zeros((8, dimension), dtype=float)
            if smooth_mode:
                rotor = bag.rotor_history.evaluate(midpoint, rotor_delay, width_fraction)
                gimbal_eval = bag.gimbal_history.evaluate(midpoint, gimbal_delay, width_fraction)
                command = smooth._command(rotor.value, gimbal_eval.value)
                command_sensitivity[:4, ROTOR_DELAY_INDEX] = rotor.delay_derivative
                command_sensitivity[4:, GIMBAL_DELAY_INDEX] = gimbal_eval.delay_derivative
            else:
                command = smooth._command(
                    bag.rotor_history.exact_zoh(midpoint, rotor_delay),
                    bag.gimbal_history.exact_zoh(midpoint, gimbal_delay),
                )
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state, sensitivity, command, decoded, parameter_jacobian,
                0.5 * dt, command_sensitivity,
            )
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state, sensitivity, command, decoded, parameter_jacobian,
                0.5 * dt, command_sensitivity,
            )
        return thrust, gimbal, state_jacobian

    def evaluate_smooth(self, coordinate: Sequence[float], width_fraction: float):
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (GLOBAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("smooth SG dynamics coordinate must be 16-D")
        self._active_rotor_delay = float(value[ROTOR_DELAY_INDEX])
        self._active_gimbal_delay = float(value[GIMBAL_DELAY_INDEX])
        if self._active_rotor_delay < 0.0 or self._active_gimbal_delay < 0.0:
            raise ValueError("command lags must be non-negative")
        return self._evaluate_joint(
            physical_coordinate=value[:PHYSICAL_DIMENSION],
            delay=self._active_rotor_delay,
            dimension=GLOBAL_DIMENSION,
            smooth_mode=True,
            width_fraction=float(width_fraction),
        )

    def evaluate_strict(self, physical_coordinate, rotor_delay: float, gimbal_delay: Optional[float] = None):
        physical = np.asarray(physical_coordinate, dtype=float)
        rotor = float(rotor_delay)
        gimbal_value = rotor if gimbal_delay is None else float(gimbal_delay)
        if (
            physical.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(physical))
            or not np.isfinite(rotor)
            or not np.isfinite(gimbal_value)
            or rotor < 0.0
            or gimbal_value < 0.0
        ):
            raise ValueError("strict SG dynamics coordinate is invalid")
        self._active_rotor_delay = rotor
        self._active_gimbal_delay = gimbal_value
        return self._evaluate_joint(
            physical_coordinate=physical,
            delay=rotor,
            dimension=PHYSICAL_DIMENSION,
            smooth_mode=False,
            width_fraction=1.0,
        )

'''
    text = text[:start] + "\n" + split_class + text[end:]

    # Explicit lag gradient in optimizer diagnostics.
    text = replace_once(
        text,
        '        "jacobian": _jacobian_spectrum(jacobian),\n'
        '    }\n'
        '    return result\n',
        '        "jacobian": _jacobian_spectrum(jacobian),\n'
        '    }\n'
        '    if gradient.size == GLOBAL_DIMENSION:\n'
        '        result["command_lag_gradient"] = {\n'
        '            "rotor": float(gradient[ROTOR_DELAY_INDEX]),\n'
        '            "gimbal": float(gradient[GIMBAL_DELAY_INDEX]),\n'
        '        }\n'
        '    return result\n',
        "lag gradient diagnostic",
    )

    # Replace directional checker completely: negative gradient + each lag axis, no catch.
    s = text.index("\ndef _directional_jacobian_checks(")
    e = text.index("\ndef _optimizer_diagnostics(", s)
    checker = '''
def _directional_jacobian_checks(evaluator, coordinate, evaluation, lower, upper):
    x = np.asarray(coordinate, dtype=float)
    residual = np.asarray(evaluation.residual, dtype=float)
    jacobian = np.asarray(evaluation.jacobian, dtype=float)
    gradient = jacobian.T @ residual
    directions = []
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 0.0:
        directions.append(("negative_gradient", -gradient / gradient_norm))
    if x.size == GLOBAL_DIMENSION:
        r = np.zeros(x.size); r[ROTOR_DELAY_INDEX] = 1.0
        g = np.zeros(x.size); g[GIMBAL_DELAY_INDEX] = 1.0
        directions.extend((("rotor_delay_seconds", r), ("gimbal_delay_seconds", g)))
    if not directions:
        d = np.zeros(x.size); d[0] = 1.0
        directions.append(("coordinate:" + PHYSICAL_PARAMETER_NAMES[0], d))
    checks = []
    base_step = np.cbrt(np.finfo(float).eps) * max(1.0, float(np.linalg.norm(x)))
    for label, direction in directions:
        step = float(base_step)
        plus_limit = math.inf
        minus_limit = math.inf
        positive = direction > 0.0
        negative = direction < 0.0
        if np.any(positive):
            plus_limit = min(plus_limit, float(np.min((upper[positive] - x[positive]) / direction[positive])))
            minus_limit = min(minus_limit, float(np.min((x[positive] - lower[positive]) / direction[positive])))
        if np.any(negative):
            plus_limit = min(plus_limit, float(np.min((x[negative] - lower[negative]) / (-direction[negative]))))
            minus_limit = min(minus_limit, float(np.min((upper[negative] - x[negative]) / (-direction[negative]))))
        analytic = jacobian @ direction
        if plus_limit > step and minus_limit > step:
            plus = evaluator(x + step * direction).residual
            minus = evaluator(x - step * direction).residual
            finite = (plus - minus) / (2.0 * step)
            scheme = "central"
        elif plus_limit > 0.0:
            step = min(step, 0.5 * plus_limit)
            finite = (evaluator(x + step * direction).residual - residual) / step
            scheme = "forward"
        elif minus_limit > 0.0:
            step = min(step, 0.5 * minus_limit)
            finite = (residual - evaluator(x - step * direction).residual) / step
            scheme = "backward"
        else:
            raise RuntimeError("finite-difference diagnostic has no feasible step")
        difference = np.asarray(finite) - analytic
        denominator = max(float(np.linalg.norm(finite)), float(np.linalg.norm(analytic)), math.sqrt(np.finfo(float).eps))
        checks.append({
            "direction": label,
            "scheme": scheme,
            "step": step,
            "analytic_l2_norm": float(np.linalg.norm(analytic)),
            "finite_difference_l2_norm": float(np.linalg.norm(finite)),
            "difference_l2_norm": float(np.linalg.norm(difference)),
            "relative_error": float(np.linalg.norm(difference) / denominator),
        })
    return checks

'''
    text = text[:s] + checker + text[e:]

    # Remove artificial invalid-trial safety wrapper.
    text = regex_once(
        text,
        r"\n\nclass _SafeCachedObjective:.*?\n\ndef _solve_smooth\(",
        "\n\ndef _solve_smooth(",
        "remove SafeCachedObjective",
        flags=re.DOTALL,
    )
    text = replace_first(text, "    objective = _SafeCachedObjective(evaluator, np.asarray(initial, dtype=float))\n", "    objective = base._CachedObjective(evaluator)\n", "smooth objective")
    text = replace_first(text, "    payload[\"diagnostics\"][\"invalid_trial_evaluations\"] = objective.diagnostics()\n", "", "smooth invalid diagnostic")
    text = replace_first(text, "    objective = _SafeCachedObjective(evaluator, np.asarray(initial, dtype=float))\n", "    objective = base._CachedObjective(evaluator)\n", "strict objective")
    text = replace_first(text, "    payload[\"diagnostics\"][\"invalid_trial_evaluations\"] = objective.diagnostics()\n", "", "strict invalid diagnostic")

    # Insert split-lag search machinery before parameter reporting.
    insert_at = text.index("\ndef _parameter_lines(")
    machinery = '''
def _recorded_command_periods(config_path: Path):
    config = base.multi.load_multi_bag_config(config_path.expanduser().resolve())
    rotor_periods = []
    gimbal_periods = []
    for specification in config.bags:
        flight = load_flight_data(
            str(specification.path), start_local=specification.start, end_local=specification.end,
            include_fc_specific_force=True, compute_sha256=False,
        )
        for target, axis in ((rotor_periods, flight.rotor_command.all_times), (gimbal_periods, flight.gimbal_command.all_times)):
            dt = np.diff(np.asarray(axis, dtype=float))
            positive = dt[np.isfinite(dt) & (dt > 0.0)]
            if positive.size == 0:
                raise ValueError("recorded command timestamps have no positive interval")
            target.append(float(np.median(positive)))
    return float(np.median(rotor_periods)), float(np.median(gimbal_periods))


def _resolve_lag_defaults(arguments):
    global _ACTIVE_ROTOR_PERIOD_SECONDS, _ACTIVE_GIMBAL_PERIOD_SECONDS, _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS
    rotor_period, gimbal_period = _recorded_command_periods(arguments.config)
    _ACTIVE_ROTOR_PERIOD_SECONDS = rotor_period
    _ACTIVE_GIMBAL_PERIOD_SECONDS = gimbal_period
    rotor_initial = rotor_period if arguments.initial_rotor_delay is None else float(arguments.initial_rotor_delay)
    gimbal_initial = gimbal_period if arguments.initial_gimbal_delay is None else float(arguments.initial_gimbal_delay)
    _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS = gimbal_initial
    arguments.initial_delay = rotor_initial
    arguments.initial_gimbal_delay = gimbal_initial
    arguments.split_command_lag_solver = _split_command_lag_search
    arguments.split_command_lag_report_writer = _write_split_delay_report


def _solve_smooth_pair(problem, initial, width, lower, upper, arguments):
    evaluator = lambda value: problem.evaluate_smooth(value, width)
    objective = base._CachedObjective(evaluator)
    result = base.least_squares(
        objective.residual, initial, jac=objective.jacobian,
        bounds=(lower, upper), method="trf", x_scale="jac", loss="linear",
        ftol=arguments.ftol, xtol=arguments.xtol, gtol=arguments.gtol,
        max_nfev=arguments.smooth_max_nfev, verbose=0,
    )
    evaluation = evaluator(result.x)
    payload = base._optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator, np.asarray(initial), result, evaluation,
        np.asarray(lower), np.asarray(upper), arguments,
    )
    return result.x, evaluation, payload


def _solve_strict_pair(problem, initial, rotor_delay, gimbal_delay, lower, upper, arguments):
    evaluator = lambda value: problem.evaluate_strict(value, rotor_delay, gimbal_delay)
    objective = base._CachedObjective(evaluator)
    result = base.least_squares(
        objective.residual, initial, jac=objective.jacobian,
        bounds=(lower, upper), method="trf", x_scale="jac", loss="linear",
        ftol=arguments.ftol, xtol=arguments.xtol, gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev, verbose=0,
    )
    evaluation = evaluator(result.x)
    payload = base._optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator, np.asarray(initial), result, evaluation,
        np.asarray(lower), np.asarray(upper), arguments,
    )
    return base.DynamicsSolution(
        physical_coordinate=np.asarray(result.x).copy(), delay_seconds=float(rotor_delay),
        gimbal_delay_seconds=float(gimbal_delay), evaluation=evaluation, optimizer=payload,
    )


def _strict_pair_screen(problem, physical, center_rotor, center_gimbal, rotor_period, gimbal_period, delay_bounds):
    lower, upper = map(float, delay_bounds)
    rotor_values = {round(float(np.clip(center_rotor + d, lower, upper)), 12) for d in (-rotor_period, 0.0, rotor_period)}
    gimbal_values = {round(float(np.clip(center_gimbal + d, lower, upper)), 12) for d in (-gimbal_period, 0.0, gimbal_period)}
    costs = {}
    expansions = []
    def evaluate_new():
        for r in sorted(rotor_values):
            for g in sorted(gimbal_values):
                if (r, g) in costs:
                    continue
                ev = problem.evaluate_strict(physical, r, g)
                costs[(r, g)] = 0.5 * float(ev.residual @ ev.residual)
    while True:
        evaluate_new()
        best = min(costs, key=lambda key: (costs[key], key[0], key[1]))
        rs = sorted(rotor_values); gs = sorted(gimbal_values)
        additions = []
        if best[0] == rs[0] and best[0] > lower:
            value = round(max(lower, best[0] - rotor_period), 12)
            if value not in rotor_values:
                rotor_values.add(value); additions.append({"axis": "rotor", "delay_seconds": value})
        elif best[0] == rs[-1] and best[0] < upper:
            value = round(min(upper, best[0] + rotor_period), 12)
            if value not in rotor_values:
                rotor_values.add(value); additions.append({"axis": "rotor", "delay_seconds": value})
        if best[1] == gs[0] and best[1] > lower:
            value = round(max(lower, best[1] - gimbal_period), 12)
            if value not in gimbal_values:
                gimbal_values.add(value); additions.append({"axis": "gimbal", "delay_seconds": value})
        elif best[1] == gs[-1] and best[1] < upper:
            value = round(min(upper, best[1] + gimbal_period), 12)
            if value not in gimbal_values:
                gimbal_values.add(value); additions.append({"axis": "gimbal", "delay_seconds": value})
        if not additions:
            break
        expansions.append({"best_before_expansion": {"rotor_delay_seconds": best[0], "gimbal_delay_seconds": best[1], "cost": costs[best]}, "added": additions})
    evaluate_new()
    best = min(costs, key=lambda key: (costs[key], key[0], key[1]))
    return {
        "best_pair": best,
        "best_cost": costs[best],
        "rotor_range_seconds": (min(rotor_values), max(rotor_values)),
        "gimbal_range_seconds": (min(gimbal_values), max(gimbal_values)),
        "expansions": expansions,
        "candidates": [
            {"rotor_delay_seconds": r, "gimbal_delay_seconds": g, "cost": costs[(r,g)], "selected": (r,g) == best}
            for r in sorted(rotor_values) for g in sorted(gimbal_values)
        ],
    }


def _split_command_lag_search(problem, initial_physical, physical_lower, physical_upper, arguments):
    rotor_period = float(_ACTIVE_ROTOR_PERIOD_SECONDS)
    gimbal_period = float(_ACTIVE_GIMBAL_PERIOD_SECONDS)
    lower_lag, upper_lag = map(float, arguments.delay_bounds)
    coordinate = np.concatenate((initial_physical, np.asarray((arguments.initial_delay, arguments.initial_gimbal_delay))))
    lower = np.concatenate((physical_lower, np.asarray((lower_lag, lower_lag))))
    upper = np.concatenate((physical_upper, np.asarray((upper_lag, upper_lag))))
    smooth_stages = []
    for index, width in enumerate(arguments.smoothstep_width_fractions):
        print(f"smooth lag {index+1}/{len(arguments.smoothstep_width_fractions)}: half-width={width} publish periods", flush=True)
        coordinate, evaluation, optimizer = _solve_smooth_pair(problem, coordinate, float(width), lower, upper, arguments)
        smooth_stages.append({
            "half_width_period_multiplier": float(width),
            "rotor_transition_half_width_seconds": float(width) * rotor_period,
            "gimbal_transition_half_width_seconds": float(width) * gimbal_period,
            "objective_cost": 0.5 * float(evaluation.residual @ evaluation.residual),
            "data_loss": evaluation.data_loss,
            "prior_cost": evaluation.prior_cost,
            "physical_coordinate": coordinate[:PHYSICAL_DIMENSION],
            "rotor_delay_seconds": float(coordinate[ROTOR_DELAY_INDEX]),
            "gimbal_delay_seconds": float(coordinate[GIMBAL_DELAY_INDEX]),
            "optimizer": optimizer,
        })
        print(f"  rotor={coordinate[ROTOR_DELAY_INDEX]:.6f}s gimbal={coordinate[GIMBAL_DELAY_INDEX]:.6f}s cost={smooth_stages[-1]['objective_cost']:.9g}", flush=True)
    physical = coordinate[:PHYSICAL_DIMENSION].copy()
    rotor = float(coordinate[ROTOR_DELAY_INDEX]); gimbal = float(coordinate[GIMBAL_DELAY_INDEX])
    profile = []
    while True:
        screen = _strict_pair_screen(problem, physical, rotor, gimbal, rotor_period, gimbal_period, arguments.delay_bounds)
        best_rotor, best_gimbal = screen["best_pair"]
        solution = _solve_strict_pair(problem, physical, best_rotor, best_gimbal, physical_lower, physical_upper, arguments)
        verify = _strict_pair_screen(problem, solution.physical_coordinate, best_rotor, best_gimbal, rotor_period, gimbal_period, arguments.delay_bounds)
        profile.append({
            "iteration": len(profile) + 1,
            "screening": screen,
            "refined_solution": {
                "rotor_delay_seconds": best_rotor,
                "gimbal_delay_seconds": best_gimbal,
                "objective_cost": base._solution_cost(solution),
                "physical_coordinate": solution.physical_coordinate,
                "optimizer": solution.optimizer,
            },
            "post_refinement_screening": verify,
        })
        next_rotor, next_gimbal = verify["best_pair"]
        if np.isclose(next_rotor, best_rotor, atol=5e-13, rtol=0.0) and np.isclose(next_gimbal, best_gimbal, atol=5e-13, rtol=0.0):
            selected = solution
            break
        physical = solution.physical_coordinate.copy(); rotor = float(next_rotor); gimbal = float(next_gimbal)
    final_candidates = []
    for item in profile[-1]["post_refinement_screening"]["candidates"]:
        value = dict(item)
        value["selected"] = bool(
            np.isclose(value["rotor_delay_seconds"], selected.delay_seconds, atol=5e-13, rtol=0.0)
            and np.isclose(value["gimbal_delay_seconds"], selected.gimbal_delay_seconds, atol=5e-13, rtol=0.0)
        )
        final_candidates.append(value)
    return {
        "schema": SCHEMA + "/command-lag-search/v1",
        "recorded_command_period_seconds": {"rotor": rotor_period, "gimbal": gimbal_period},
        "initial_delay_seconds": {"rotor": float(arguments.initial_delay), "gimbal": float(arguments.initial_gimbal_delay)},
        "delay_bounds_seconds": (lower_lag, upper_lag),
        "smooth_surrogate": {
            "representation": "sum of overlapping quintic-smoothed ZOH jumps",
            "half_width_period_multipliers": list(arguments.smoothstep_width_fractions),
        },
        "smooth_stages": smooth_stages,
        "smooth_result": {"rotor_delay_seconds": float(coordinate[ROTOR_DELAY_INDEX]), "gimbal_delay_seconds": float(coordinate[GIMBAL_DELAY_INDEX])},
        "strict_zoh": {
            "grid_step_seconds": {"rotor": rotor_period, "gimbal": gimbal_period},
            "profile_iterations": profile,
            "termination": "same lag pair remains selected after physical-parameter refinement",
        },
        "strict_candidates": final_candidates,
        "selected_solution": selected,
        "selected": {"rotor_delay_seconds": float(selected.delay_seconds), "gimbal_delay_seconds": float(selected.gimbal_delay_seconds), "objective_cost": base._solution_cost(selected)},
    }


def _write_split_delay_report(output_directory, search):
    payload = {key: value for key, value in search.items() if key != "selected_solution"}
    (output_directory / "delay_profile.json").write_text(
        json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    periods = search["recorded_command_period_seconds"]
    selected = search["selected"]
    lines = [
        "Savitzky-Golay split command-lag search", "",
        f"rotor median publish period [s]: {periods['rotor']:.12g}",
        f"gimbal median publish period [s]: {periods['gimbal']:.12g}",
        f"initial lags [s]: {search['initial_delay_seconds']}",
        f"smooth half-width multipliers: {search['smooth_surrogate']['half_width_period_multipliers']}", "",
    ]
    for stage in search["smooth_stages"]:
        lines.append(
            "smooth width={:.6g}: rotor={:.9g}s gimbal={:.9g}s cost={:.12g} lag_gradient={} FD={}".format(
                stage["half_width_period_multiplier"], stage["rotor_delay_seconds"], stage["gimbal_delay_seconds"], stage["objective_cost"],
                stage["optimizer"]["diagnostics"]["final"].get("command_lag_gradient"),
                stage["optimizer"]["diagnostics"]["finite_difference_directional_checks"],
            )
        )
    lines.extend(["", f"strict profile iterations: {len(search['strict_zoh']['profile_iterations'])}"])
    for item in search["strict_zoh"]["profile_iterations"]:
        screen = item["screening"]
        lines.append(f"iteration {item['iteration']}: best={screen['best_pair']} cost={screen['best_cost']:.12g} rotor_range={screen['rotor_range_seconds']} gimbal_range={screen['gimbal_range_seconds']} expansions={screen['expansions']}")
    lines.extend([
        "",
        "selected rotor lag {:.12g} s ({:.6g} publish periods)".format(selected["rotor_delay_seconds"], selected["rotor_delay_seconds"] / periods["rotor"]),
        "selected gimbal lag {:.12g} s ({:.6g} publish periods)".format(selected["gimbal_delay_seconds"], selected["gimbal_delay_seconds"] / periods["gimbal"]),
        f"selected objective cost {selected['objective_cost']:.12g}",
    ])
    base.strict._write_text(output_directory / "delay_profile.txt", lines)
    _write_parameters_pdf(output_directory / "delay_profile.pdf", lines)

'''
    text = text[:insert_at] + machinery + text[insert_at:]

    # Restore acceleration labels and report both lags.
    old = '''    lines = [
        line.replace(
            "Deterministic pose-spline dynamics estimator",
            "Deterministic geometric Savitzky-Golay dynamics estimator",
        )
        .replace(
            "joint dynamics loss:",
            "joint residual-wrench accumulated squared loss:",
        )
        .replace(
            "bag dynamics loss:",
            "legacy acceleration-residual loss diagnostic:",
        )
        for line in lines
    ]
'''
    new = '''    lines = [
        line.replace(
            "Deterministic pose-spline dynamics estimator",
            "Deterministic geometric Savitzky-Golay dynamics estimator",
        ).replace("command lag", "rotor command lag").replace(
            "selected strict-ZOH lag", "selected strict-ZOH rotor lag"
        )
        for line in lines
    ]
    initial_gimbal = initial_delay if _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS is None else float(_ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS)
    selected_gimbal = selected.delay_seconds if selected.gimbal_delay_seconds is None else float(selected.gimbal_delay_seconds)
    lines.append("  gimbal command lag [s] {: .10g} -> {: .10g}".format(initial_gimbal, selected_gimbal))
'''
    text = replace_once(text, old, new, "restore report objective")

    # Remove wrench-objective prose block.
    s = text.find("    scaling = _reference_wrench_scaling(reference_parameters)\n")
    if s != -1:
        e = text.index("    optimizer = selected.optimizer\n", s)
        text = text[:s] + text[e:]
    text = text.replace(
        '                    "invalid_trial_evaluations={}".format(\n'
        '                        diagnostics.get("invalid_trial_evaluations")\n'
        '                    ),\n',
        "",
    )

    # Restore per-bag acceleration loss.
    text = text.replace(
        '    legacy_acceleration_loss = diagnostics.pop("dynamics_loss", None)\n'
        '    if legacy_acceleration_loss is not None:\n'
        '        diagnostics["legacy_acceleration_residual_loss_diagnostic"] = (\n'
        '            legacy_acceleration_loss\n'
        '        )\n',
        "",
    )
    text = replace_once(text, '        "selected_lag_seconds": None,\n', '        "selected_rotor_lag_seconds": None,\n        "selected_gimbal_lag_seconds": None,\n', "timing lag pair keys")

    # JSON method semantics and settings.
    text = replace_once(
        text,
        '        "parameter_data_objective": (\n'
        '            "weighted empirical uncentered second moment of reference-scaled "\n'
        '            "residual body wrench; no residual mean subtraction"\n'
        '        ),\n',
        '        "parameter_data_objective": (\n'
        '            "mean squared translational/angular acceleration residual; "\n'
        '            "angular residual uses the reference inertia/mass metric"\n'
        '        ),\n',
        "method objective",
    )
    text = replace_once(
        text,
        '        "command_mode_during_search": "quintic smoothstep ZOH",\n'
        '        "command_mode_final": "strict ZOH",\n',
        '        "command_mode_during_search": "overlapping quintic-smoothed ZOH jumps with separate rotor/gimbal lags",\n'
        '        "command_mode_final": "strict ZOH with separate rotor/gimbal lags",\n',
        "method lag semantics",
    )
    text = replace_once(
        text,
        '        "smoothstep_width_fractions": old_settings.get(\n'
        '            "smoothstep_width_fractions"\n'
        '        ),\n',
        '        "smoothstep_half_width_period_multipliers": list(arguments.smoothstep_width_fractions),\n'
        '        "smoothstep_transitions_may_overlap": True,\n'
        '        "strict_zoh_grid_step_seconds": {\n'
        '            "rotor": _ACTIVE_ROTOR_PERIOD_SECONDS,\n'
        '            "gimbal": _ACTIVE_GIMBAL_PERIOD_SECONDS,\n'
        '        },\n',
        "lag settings",
    )
    text = replace_once(
        text,
        '    initial["delay_default"] = (\n'
        '        "zero unless --initial-delay is explicitly supplied"\n'
        '    )\n',
        '    initial["delay_default"] = "one measured median publish period per command channel unless explicitly supplied"\n'
        '    initial["rotor_delay_seconds"] = float(initial_delay)\n'
        '    initial["gimbal_delay_seconds"] = float(arguments.initial_gimbal_delay)\n'
        '    initial["recorded_command_period_seconds"] = {\n'
        '        "rotor": _ACTIVE_ROTOR_PERIOD_SECONDS, "gimbal": _ACTIVE_GIMBAL_PERIOD_SECONDS\n'
        '    }\n',
        "initial lag metadata",
    )

    # Replace scalar selected-lag timing updates by explicit channel lag.
    text = replace_once(
        text,
        '    selected_lag = float(root.get("selection", {}).get("delay_seconds", math.nan))\n',
        '    selection_root = root.get("selection", {})\n'
        '    selected_rotor_lag = float(selection_root.get("rotor_delay_seconds", selection_root.get("delay_seconds", math.nan)))\n'
        '    selected_gimbal_lag = float(selection_root.get("gimbal_delay_seconds", selected_rotor_lag))\n',
        "selected pair root",
    )
    old_loop = '''                timing["selected_lag_seconds"] = selected_lag
                for channel in ("rotor", "gimbal"):
                    block = timing.get(channel)
                    if isinstance(block, dict):
                        median = block.get("median_seconds")
                        block["selected_lag_over_median_interval"] = (
                            None
                            if median is None or not np.isfinite(selected_lag)
                            else float(selected_lag / float(median))
                        )
'''
    new_loop = '''                timing["selected_rotor_lag_seconds"] = selected_rotor_lag
                timing["selected_gimbal_lag_seconds"] = selected_gimbal_lag
                for channel, lag in (("rotor", selected_rotor_lag), ("gimbal", selected_gimbal_lag)):
                    block = timing.get(channel)
                    if isinstance(block, dict):
                        median = block.get("median_seconds")
                        block["selected_lag_over_median_interval"] = (
                            None if median is None or not np.isfinite(lag)
                            else float(lag / float(median))
                        )
'''
    text = replace_once(text, old_loop, new_loop, "root timing pair")

    # per-bag timing block
    old_bag = '''                    selected_lag = float(payload.get("shared_delay_seconds", math.nan))
                    timing["selected_lag_seconds"] = selected_lag
                    for channel in ("rotor", "gimbal"):
                        block = timing.get(channel)
                        if isinstance(block, dict):
                            median = block.get("median_seconds")
                            block["selected_lag_over_median_interval"] = (
                                None
                                if median is None or not np.isfinite(selected_lag)
                                else float(selected_lag / float(median))
                            )
'''
    new_bag = '''                    selected_rotor_lag = float(payload.get("shared_rotor_delay_seconds", payload.get("shared_delay_seconds", math.nan)))
                    selected_gimbal_lag = float(payload.get("shared_gimbal_delay_seconds", selected_rotor_lag))
                    timing["selected_rotor_lag_seconds"] = selected_rotor_lag
                    timing["selected_gimbal_lag_seconds"] = selected_gimbal_lag
                    for channel, lag in (("rotor", selected_rotor_lag), ("gimbal", selected_gimbal_lag)):
                        block = timing.get(channel)
                        if isinstance(block, dict):
                            median = block.get("median_seconds")
                            block["selected_lag_over_median_interval"] = (
                                None if median is None or not np.isfinite(lag)
                                else float(lag / float(median))
                            )
'''
    text = replace_once(text, old_bag, new_bag, "bag timing pair")

    # Remove wrench-objective aliases.
    text = regex_once(
        text,
        r'    selection = root\.get\("selection"\)\n    if isinstance\(selection, dict\):.*?\n\n    optimizer_diagnostics = \{',
        '    selection = root.get("selection")\n\n    optimizer_diagnostics = {',
        "remove wrench selection aliases",
        flags=re.DOTALL,
    )
    text = replace_once(
        text,
        '        "selected_optimizer": (\n'
        '            selection.get("optimizer") if isinstance(selection, dict) else None\n'
        '        ),\n',
        '        "selected_optimizer": (\n'
        '            selection.get("optimizer") if isinstance(selection, dict) else None\n'
        '        ),\n'
        '        "command_lag_search": root.get("command_lag_search"),\n',
        "optimizer lag payload",
    )
    text = replace_once(text, '    outputs = root.setdefault("outputs", {})\n', '    outputs = root.setdefault("outputs", {})\n    outputs["delay_profile_json"] = "delay_profile.json"\n    outputs["delay_profile_txt"] = "delay_profile.txt"\n', "lag output links")

    # Parser: two lags, broad continuation, no fixed polish knobs.
    old_parser = '''    parser.add_argument(
        "--initial-delay",
        type=float,
        default=None,
        help="Initial command lag. Default is exactly 0 s.",
    )
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(0.50, 0.20, 0.05),
    )
    parser.add_argument(
        "--zoh-polish-radius",
        type=float,
        default=0.004,
    )
    parser.add_argument(
        "--zoh-polish-step",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--zoh-polish-top-k",
        type=int,
        default=3,
    )
'''
    new_parser = '''    parser.add_argument(
        "--initial-rotor-delay",
        type=float,
        default=None,
        help="Initial rotor-command lag; default is one measured rotor publish period.",
    )
    parser.add_argument(
        "--initial-gimbal-delay",
        type=float,
        default=None,
        help="Initial gimbal-command lag; default is one measured gimbal publish period.",
    )
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(4.0, 2.0, 1.0, 0.5),
        help="Transition half-widths in measured publish periods; supports may overlap.",
    )
'''
    text = replace_once(text, old_parser, new_parser, "SG parser")
    text = text.replace(
        "        arguments.gtol,\n        arguments.zoh_polish_radius,\n        arguments.zoh_polish_step,\n        arguments.zoh_polish_top_k,\n",
        "        arguments.gtol,\n",
        1,
    )
    text = replace_once(
        text,
        "        or not bounds[0] <= initial_delay <= bounds[1]\n        or widths.ndim != 1\n",
        "        or not bounds[0] <= initial_delay <= bounds[1]\n"
        "        or not bounds[0] <= float(arguments.initial_gimbal_delay) <= bounds[1]\n"
        "        or widths.ndim != 1\n",
        "validate gimbal lag",
    )

    text = replace_once(
        text,
        "    # Command-lag initialization is deliberately zero by default.\n"
        "    if arguments.initial_delay is None:\n"
        "        arguments.initial_delay = 0.0\n\n",
        "    _resolve_lag_defaults(arguments)\n\n",
        "resolve lag defaults",
    )
    text = replace_once(
        text,
        "    base.SplineDynamicsProblem = SplineDynamicsProblem\n"
        "    base._solve_smooth = _solve_smooth\n"
        "    base._solve_strict = _solve_strict\n",
        "    base.SplineDynamicsProblem = SplineDynamicsProblem\n",
        "remove scalar solve hooks",
    )
    old_run = '''    original_stdout = sys.stdout
    sys.stdout = _RewritingStdout(original_stdout)
    try:
        try:
            result = base.run(arguments)
        except ValueError as error:
            # Window feasibility and local-polynomial conditioning failures
            # should stop before optimization with a concise diagnostic rather
            # than a long traceback from the compatibility backend.
            raise SystemExit(str(error)) from error
    finally:
        sys.stdout = original_stdout
'''
    new_run = '''    original_stdout = sys.stdout
    sys.stdout = _RewritingStdout(original_stdout)
    try:
        result = base.run(arguments)
    finally:
        sys.stdout = original_stdout
'''
    text = replace_once(text, old_run, new_run, "propagate SG numerical errors")
    return text


def build_standalone_sg(patched_base: str, patched_sg: str) -> str:
    """Inline the 8be7473 downstream dynamics into the SG estimator.

    ``patched_base`` is source material only. The generated estimator does not
    import, mutate, or dispatch through deterministic_spline_dynamics_estimator.
    """
    base_source = patched_base
    base_source = replace_once(
        base_source,
        '"""Multi-bag gradient matching from pose-only continuous-time splines."""',
        '"""Independent geometric Savitzky--Golay rigid-body dynamics estimator.\n\n'
        'This module owns its rigid-body dynamics, optimization, rollout, command-lag\n'
        'search, and report generation. The spline estimator is a separate\n'
        'ablation/reference implementation and is not imported.\n'
        '"""',
        "standalone module docstring",
    )
    base_source = regex_once(
        base_source,
        r'\n\nif __name__ == "__main__":\n    sys\.exit\(main\(\)\)\s*$',
        "\n",
        "strip copied backend main guard",
        flags=re.DOTALL,
    )
    base_source = replace_once(
        base_source,
        "import math\n",
        "import math\nimport re\n",
        "standalone re import",
    )
    base_source = replace_once(
        base_source,
        "import numpy as np\n",
        "import numpy as np\n\nimport savgol_trajectory as sg\n",
        "standalone SG import",
    )

    # The copied backend's public CLI and scalar-lag fallback are not part of
    # the SG estimator. Keep the reusable numerical implementation, but make
    # the copied run path call the split-lag solver directly.
    parser_start = base_source.index("\ndef create_argument_parser()")
    parser_end = base_source.index("\ndef _solution_cost(", parser_start)
    base_source = base_source[:parser_start] + "\n" + base_source[parser_end:]

    main_start = base_source.rfind("\ndef main(")
    if main_start < 0:
        raise RuntimeError("copied backend main function not found")
    base_source = base_source[:main_start] + "\n"

    physical_marker = (
        "    physical_lower, physical_upper = _physical_bounds(\n"
        "        initial_physical_coordinate\n"
        "    )\n"
    )
    physical_pos = base_source.index(physical_marker)
    solver_start = physical_pos + len(physical_marker)
    solver_end = base_source.index("\n\n    output_directory = (", solver_start)
    direct_solver = (
        "    split_lag_result = _split_command_lag_search(\n"
        "        problem,\n"
        "        initial_physical_coordinate,\n"
        "        physical_lower,\n"
        "        physical_upper,\n"
        "        arguments,\n"
        "    )\n"
        "    selected = split_lag_result[\"selected_solution\"]\n"
        "    command_lag_search = {\n"
        "        key: value for key, value in split_lag_result.items()\n"
        "        if key != \"selected_solution\"\n"
        "    }\n"
        "    smooth_stage_payloads = command_lag_search[\"smooth_stages\"]\n"
        "    strict_payloads = command_lag_search[\"strict_candidates\"]\n"
        "    smooth_delay = command_lag_search[\"smooth_result\"][\"rotor_delay_seconds\"]\n"
        "    candidate_delays = np.empty(0, dtype=float)\n"
        "    screening_costs = np.empty(0, dtype=float)\n"
        "    strict_solutions = [selected]\n"
        "    print(\n"
        "        \"selected strict lags rotor={:.6f}s gimbal={:.6f}s; \"\n"
        "        \"producing reconstruction and reports\".format(\n"
        "            selected.delay_seconds, selected.gimbal_delay_seconds\n"
        "        ),\n"
        "        flush=True,\n"
        "    )\n"
    )
    base_source = (
        base_source[:solver_start]
        + direct_solver
        + base_source[solver_end:]
    )

    report_start = base_source.index(
        '    split_writer = getattr(arguments, "split_command_lag_report_writer", None)\n'
    )
    report_end = base_source.index(
        "    elapsed = time.perf_counter() - started\n",
        report_start,
    )
    base_source = (
        base_source[:report_start]
        + "    _write_split_delay_report(output_directory, command_lag_search)\n"
        + base_source[report_end:]
    )

    bridge = (
        "\n\n# Reusable downstream implementation copied into this module from the\n"
        "# 8be7473 spline estimator. These aliases are local implementation\n"
        "# snapshots; there is no runtime spline-estimator dependency.\n"
        "_SharedSplineDynamicsProblem = SplineDynamicsProblem\n"
        "_shared_parameter_lines = _parameter_lines\n"
        "_shared_run = run\n\n"
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v3"\n'
        'OUTPUT_SUBDIRECTORY = "deterministic_savgol_dynamics"\n'
        'DATA_DICTIONARY_SOURCE = Path(__file__).resolve().with_name(\n'
        '    "deterministic_savgol_dynamics_data_dictionary.md"\n'
        ')\n'
        "ROTOR_DELAY_INDEX = PHYSICAL_DIMENSION\n"
        "GIMBAL_DELAY_INDEX = PHYSICAL_DIMENSION + 1\n"
        "GLOBAL_DIMENSION = PHYSICAL_DIMENSION + 2\n"
        "DELAY_INDEX = ROTOR_DELAY_INDEX\n"
        "candidate_payload = sg.candidate_payload\n"
    )

    body_start = patched_sg.index(
        "_ACTIVE_WINDOW_SECONDS: Optional[float] = None\n"
    )
    body = patched_sg[body_start:]
    body = replace_once(
        body,
        "_ORIGINAL_PARAMETER_LINES = base._parameter_lines\n",
        "_ORIGINAL_PARAMETER_LINES = _shared_parameter_lines\n",
        "standalone parameter-line alias",
    )
    body = replace_once(
        body,
        "class SplineDynamicsProblem(base.SplineDynamicsProblem):",
        "class SplineDynamicsProblem(_SharedSplineDynamicsProblem):",
        "standalone SG problem base",
    )

    # The standalone copied run path calls the split solver/report directly;
    # callback attributes used by the old monkey-patch architecture are gone.
    body = body.replace(
        "    arguments.split_command_lag_solver = _split_command_lag_search\n",
        "",
    )
    body = body.replace(
        "    arguments.split_command_lag_report_writer = _write_split_delay_report\n",
        "",
    )

    monkey_start = body.index(
        "    # Patch only this process.  The existing spline file on disk is untouched.\n"
    )
    monkey_end_marker = "    base._parameter_lines = _parameter_lines\n"
    monkey_end = body.index(monkey_end_marker, monkey_start) + len(monkey_end_marker)
    body = body[:monkey_start] + body[monkey_end:]
    body = replace_once(
        body,
        "result = base.run(arguments)",
        "result = _shared_run(arguments)",
        "standalone run dispatch",
    )
    body = body.replace("base.", "")
    body = body.replace(
        "Required by run's old report serializer but intentionally unused",
        "Required by the shared report serializer but intentionally unused",
    )

    source = base_source.rstrip() + bridge + "\n" + body.lstrip()
    forbidden = (
        "import deterministic_spline_dynamics_estimator",
        "from deterministic_spline_dynamics_estimator",
        "base.",
        "split_command_lag_solver",
        "split_command_lag_report_writer",
        "_SafeCachedObjective",
        "penalty_residual",
        "invalid_trial_evaluations",
        "_reference_wrench_scaling",
        "zoh_polish_radius",
        "zoh_polish_step",
        "zoh_polish_top_k",
    )
    remaining = [token for token in forbidden if token in source]
    if remaining:
        raise RuntimeError(
            "standalone SG source still contains forbidden implementation: {}".format(
                ", ".join(remaining)
            )
        )
    required = (
        "_SharedSplineDynamicsProblem = SplineDynamicsProblem",
        "_shared_run = run",
        "class SplineDynamicsProblem(_SharedSplineDynamicsProblem):",
        "ROTOR_DELAY_INDEX = PHYSICAL_DIMENSION",
        "GIMBAL_DELAY_INDEX = PHYSICAL_DIMENSION + 1",
        "def _split_command_lag_search(",
        "def _write_split_delay_report(",
        "def _write_parameters_pdf(",
        '_write_parameters_pdf(output_directory / "delay_profile.pdf", lines)',
        "default=(4.0, 2.0, 1.0, 0.5)",
        "split_lag_result = _split_command_lag_search(",
        "_write_split_delay_report(output_directory, command_lag_search)",
        "result = _shared_run(arguments)",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(
            "standalone SG source is missing required implementation: {}".format(
                ", ".join(missing)
            )
        )
    compile(source, "deterministic_savgol_dynamics_estimator.py", "exec")
    return source


def patch_confidence(text: str) -> str:
    text = replace_once(text, 'SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v3"', 'SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v4"', "confidence schema")
    s = text.index("\ndef _residual_parameter_diagnostics(")
    e = text.index("\ndef create_argument_parser()", s)
    text = text[:s] + "\n" + text[e:]
    old = '''    # SG estimator semantics: zero command-lag initialization unless explicitly
    # overridden.  Do this before config handling so the config's historical
    # initial_delay_seconds cannot silently re-enter.
    if arguments.initial_delay is None:
        arguments.initial_delay = 0.0
    deterministic._ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)
'''
    text = replace_once(text, old, '    deterministic._ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)\n    deterministic._resolve_lag_defaults(arguments)\n', "confidence lag defaults")
    text = replace_once(text, "    initial_delay = float(arguments.initial_delay)\n", "    initial_delay = float(arguments.initial_delay)\n    initial_gimbal_delay = float(arguments.initial_gimbal_delay)\n", "confidence gimbal init")
    text = replace_once(
        text,
        "    if arguments.deterministic_result is None:\n"
        "        selected, optimizer_history = legacy._estimate_solution(\n"
        "            bag,\n"
        "            arguments,\n"
        "            initial_delay,\n"
        "            vehicle_model.parameters,\n"
        "            parameter_prior,\n"
        "        )\n",
        "    if arguments.deterministic_result is None:\n"
        "        initial_physical = np.zeros(deterministic.PHYSICAL_DIMENSION, dtype=float)\n"
        "        physical_lower, physical_upper = deterministic._physical_bounds(initial_physical)\n"
        "        lag_search = deterministic._split_command_lag_search(\n"
        "            deterministic.SplineDynamicsProblem((bag,), vehicle_model.parameters, parameter_prior),\n"
        "            initial_physical, physical_lower, physical_upper, arguments,\n"
        "        )\n"
        "        selected = lag_search[\"selected_solution\"]\n"
        "        optimizer_history = {\"source\": \"split_command_lag_search\", \"command_lag_search\": {key: value for key, value in lag_search.items() if key != \"selected_solution\"}}\n",
        "confidence standalone split solver",
    )
    text = replace_once(text, '            delay_seconds = float(selection["delay_seconds"])\n', '            rotor_delay_seconds = float(selection.get("rotor_delay_seconds", selection["delay_seconds"]))\n            gimbal_delay_seconds = float(selection.get("gimbal_delay_seconds", rotor_delay_seconds))\n', "confidence result pair")
    text = replace_once(text, "        selected_evaluation = problem.evaluate_strict(\n            physical_coordinate,\n            delay_seconds,\n        )\n", "        selected_evaluation = problem.evaluate_strict(\n            physical_coordinate, rotor_delay_seconds, gimbal_delay_seconds\n        )\n", "confidence strict pair")
    text = replace_once(text, "            delay_seconds=delay_seconds,\n            evaluation=selected_evaluation,\n", "            delay_seconds=rotor_delay_seconds,\n            gimbal_delay_seconds=gimbal_delay_seconds,\n            evaluation=selected_evaluation,\n", "confidence solution pair")
    text = replace_once(text, '        "selected delay {:.6f}s; mass {:.6g} kg".format(\n            selected.delay_seconds,\n            selected.evaluation.decoded.parameters.mass,\n        ),\n', '        "selected lags rotor={:.6f}s gimbal={:.6f}s; mass {:.6g} kg".format(\n            selected.delay_seconds, selected.gimbal_delay_seconds,\n            selected.evaluation.decoded.parameters.mass,\n        ),\n', "confidence pair print")
    # remove residual diagnostic computation
    text = regex_once(text, r'    residual_parameter_diagnostics = _residual_parameter_diagnostics\(.*?\n    \)\n\n', '', "remove residual parameter computation", flags=re.DOTALL)
    text = replace_once(text, '            "second_moment_dimensionless": (\n                residual_parameter_diagnostics["residual_wrench_second_moment"][\n                    "dimensionless"\n                ]\n            ),\n', '            "second_moment_dimensionless": (wrench_dimensionless.T @ wrench_dimensionless) / wrench_dimensionless.shape[0],\n', "direct wrench second moment")
    # delete absorbability block and posterior residual error
    text = regex_once(text, r'        "residual_parameter_absorbability": \{.*?        \},\n', '', "remove likelihood absorbability", flags=re.DOTALL)
    text = regex_once(text, r'        "residual_implied_parameter_error": \(.*?        \),\n', '', "remove posterior implied error", flags=re.DOTALL)
    text = replace_once(text, '            "delay_seconds": float(selected.delay_seconds),\n', '            "delay_seconds": float(selected.delay_seconds),\n            "rotor_delay_seconds": float(selected.delay_seconds),\n            "gimbal_delay_seconds": float(selected.gimbal_delay_seconds),\n', "likelihood pair")
    text = replace_once(text, '            "delay_seconds": selected.delay_seconds,\n            "objective_cost": float(deterministic._solution_cost(selected)),\n', '            "delay_seconds": selected.delay_seconds,\n            "rotor_delay_seconds": selected.delay_seconds,\n            "gimbal_delay_seconds": selected.gimbal_delay_seconds,\n            "objective_cost": float(deterministic._solution_cost(selected)),\n', "confidence deterministic pair")
    text = text.replace('        "residual_parameter_diagnostics": residual_parameter_diagnostics,\n', '')
    # remove files generated solely by removed diagnostic
    s = text.find('    diagnostic_lines = _residual_parameter_diagnostic_lines(')
    if s != -1:
        e = text.index('    _write_json(output_directory / "confidence.json", payload)', s)
        text = text[:s] + text[e:]
    text = regex_once(text, r'\n    print\(\n        "residual absorbability:.*?\n    \)\n', '\n', "remove absorbability print", flags=re.DOTALL)
    return text


def patch_ablation(text: str) -> str:
    text = replace_once(text, 'SCHEMA = "grape-param-estim/savgol-window-ablation/v2"', 'SCHEMA = "grape-param-estim/savgol-window-ablation/v3"', "ablation schema")
    text = replace_once(text, '        "selected_delay_seconds": float(selection["delay_seconds"]),\n', '        "selected_delay_seconds": float(selection["delay_seconds"]),\n        "selected_rotor_delay_seconds": float(selection.get("rotor_delay_seconds", selection["delay_seconds"])),\n        "selected_gimbal_delay_seconds": float(selection.get("gimbal_delay_seconds", selection["delay_seconds"])),\n', "ablation pair")
    text = regex_once(text, r'        "joint_residual_wrench_accumulated_squared_loss_dimensionless": float\(.*?        \),\n', '', "remove ablation wrench objective", flags=re.DOTALL)
    text = text.replace('        axes[0, 0].set_ylabel("residual-wrench accumulated squared loss")\n', '        axes[0, 0].set_ylabel("acceleration-residual dynamics loss")\n', 1)
    text = replace_once(text, '        axes[1, 1].plot(windows, 1000.0 * vector("selected_delay_seconds"), marker="o", label="selected lag")\n', '        axes[1, 1].plot(windows, 1000.0 * vector("selected_rotor_delay_seconds"), marker="o", label="rotor lag")\n        axes[1, 1].plot(windows, 1000.0 * vector("selected_gimbal_delay_seconds"), marker="o", label="gimbal lag")\n', "ablation lag plot")
    text = replace_once(text, '                "  residual-wrench accumulated squared loss={}".format(\n                    item.get("joint_residual_wrench_accumulated_squared_loss_dimensionless")\n                ),\n                "  selected delay={} s".format(item.get("selected_delay_seconds")),\n', '                "  acceleration-residual dynamics loss={}".format(item.get("joint_dynamics_loss")),\n                "  selected rotor delay={} s".format(item.get("selected_rotor_delay_seconds")),\n                "  selected gimbal delay={} s".format(item.get("selected_gimbal_delay_seconds")),\n', "ablation text")
    text = regex_once(text, r'                "  residual absorbable fraction=.*?                "  confidence residual sample count=', '                "  confidence residual sample count=', "remove ablation absorb text", flags=re.DOTALL)
    text = regex_once(
        text,
        r'                            "residual_absorbable_fraction": float\(.*?                            "residual_implied_parameter_std_raw_coordinate": np\.asarray\(.*?                            \),\n',
        '',
        "remove ablation absorb/implied-error fields",
        flags=re.DOTALL,
    )
    text = replace_once(text, '    try:\n        return run(arguments, passthrough)\n    except ValueError as error:\n        raise SystemExit(str(error)) from error\n', '    return run(arguments, passthrough)\n', "ablation propagate errors")
    return text


def patch_dictionary(text: str) -> str:
    start = text.index("## 7. Command timestamp diagnostics")
    replacement = '''## 7. Command lags

The recorded control input contains two separately timestamped command channels:

```text
rotor_command   : four rotor thrust commands
gimbal_command  : four gimbal angle commands
```

The estimator therefore uses two lag coordinates, `rotor_delay_seconds` and
`gimbal_delay_seconds`.  A single common lag is not imposed.

For each channel, the median positive recorded timestamp interval is the
channel's data-derived publish period.  Unless explicitly overridden, the
initial lag is one measured publish period for that channel.

### Smooth continuation

The smooth command is the ZOH initial value plus a sum of command jumps, with
each Heaviside jump replaced by a quintic smoothstep.  Transition supports are
allowed to overlap.  The default transition half-widths are `4, 2, 1, 0.5`
times each channel's measured publish period.

Both lag columns are included in the analytic Jacobian.  Optimizer diagnostics
record rotor/gimbal lag gradients and finite-difference checks along both lag
axes.

### Strict-ZOH refinement

The previous fixed `±4 ms`, `1 ms`, `top-k=3` polish is removed.  Strict ZOH is
screened on a 2-D lag grid whose axis steps are the measured rotor and gimbal
publish periods.  The initial grid spans one period around the smooth result.
If the best point lies on an edge, that axis is extended by one publish period
in the improving direction.  Physical parameters are optimized at the selected
lag pair, the lag grid is screened again, and the alternation stops when the
same pair remains selected.

Detailed history is written to `delay_profile.json`, `delay_profile.txt`, the
text-only `delay_profile.pdf`, and `optimizer_diagnostics.json`.

## 8. Deterministic parameter objective

The deterministic SG objective is again the original acceleration-domain
gradient-matching objective.  Translation uses body-frame acceleration error;
rotation uses angular-acceleration error with the reference inertia/mass metric.
The bag data term is the mean squared residual over valid centered SG times.
The Gaussian physical prior is a separate residual block.

Residual body-wrench mean, covariance, second moment, standard deviation and RMS
remain diagnostics; they are not the deterministic parameter objective.

## 9. Residual wrench and confidence

A raw residual-wrench sample is retained at every valid centered SG evaluation
time.  No confidence-specific temporal thinning is applied.

The temporary residual-parameter absorbability and residual-implied parameter
bias/covariance/second-moment diagnostics are removed.  The data-only SVD,
information matrix, residual-wrench Gaussian model, and Gaussian-prior fusion
remain.

Moore--Penrose pseudoinverse is used only where rank-deficient information or
precision matrices are intentionally part of the model.

## 10. Numerical failure policy

Invalid optimizer trials are not replaced by an artificial large residual or a
zero Jacobian.  Numerical exceptions propagate at the point where the actual
calculation becomes invalid and stop the run.

Physical inertia dynamics use solve-based linear algebra and therefore fail on
a genuinely singular physical inertia.  Pseudoinverse is reserved for intended
rank-deficient information/precision calculations.
'''
    return text[:start] + replacement


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    needed = (
        "deterministic_savgol_dynamics_estimator.py",
        "deterministic_spline_dynamics_estimator.py",
        "smooth_command.py",
        "../src/grape_param_estim/dynamics.py",
    )
    paths = {name: (root / name).resolve() for name in needed}
    originals = {}
    for name in needed:
        path = paths[name]
        if not path.is_file():
            raise SystemExit(f"missing target: {path}")
        data = path.read_bytes()
        actual = blob_sha(data)
        expected = EXPECTED[name]
        if actual != expected:
            raise SystemExit(
                f"refusing to patch {name}: expected 8be7473 blob {expected}, got {actual}"
            )
        originals[name] = data

    patched_base = patch_base(
        originals["deterministic_spline_dynamics_estimator.py"].decode("utf-8")
    )
    patched_sg_wrapper = patch_sg(
        originals["deterministic_savgol_dynamics_estimator.py"].decode("utf-8")
    )
    standalone_sg = build_standalone_sg(patched_base, patched_sg_wrapper)
    replacements = {
        "smooth_command.py": patch_smooth_command(
            originals["smooth_command.py"].decode("utf-8")
        ),
        "../src/grape_param_estim/dynamics.py": patch_dynamics(
            originals["../src/grape_param_estim/dynamics.py"].decode("utf-8")
        ),
        "deterministic_savgol_dynamics_estimator.py": standalone_sg,
    }
    replacement_bytes = {
        name: value.encode("utf-8") for name, value in replacements.items()
    }

    with tempfile.TemporaryDirectory(prefix="grape-independent-sg-") as td_value:
        td = Path(td_value)
        for name, data in replacement_bytes.items():
            target = td / Path(name).name
            target.write_bytes(data)
            py_compile.compile(str(target), doraise=True)
        namespace = {}
        smooth_path = td / "smooth_command.py"
        exec(compile(smooth_path.read_text(), str(smooth_path), "exec"), namespace)
        import numpy as np
        history_type = namespace["QuinticSmoothZoh"]
        history = history_type(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([[0.0], [1.0], [-0.5], [0.75]]),
        )
        widths = history.transition_half_widths(2.0)
        if not np.allclose(widths, 2.0):
            raise RuntimeError("smooth-command half-width was clipped")
        sample_time = 1.55
        lag = 0.23
        step = 1.0e-7
        analytic = history.evaluate(sample_time, lag, 2.0).delay_derivative
        numeric = (
            history.evaluate(sample_time, lag + step, 2.0).value
            - history.evaluate(sample_time, lag - step, 2.0).value
        ) / (2.0 * step)
        if not np.allclose(analytic, numeric, rtol=2.0e-6, atol=2.0e-8):
            raise RuntimeError("smooth-command delay derivative self-test failed")

    written = []
    try:
        for name, data in replacement_bytes.items():
            path = paths[name]
            temporary = path.with_name(path.name + ".patch-tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
            written.append(name)
        subprocess.run(
            ["git", "diff", "--check", "--", *replacements.keys()],
            cwd=root,
            check=True,
        )
        sg_text = paths["deterministic_savgol_dynamics_estimator.py"].read_text(
            encoding="utf-8"
        )
        for token in (
            "deterministic_spline_dynamics_estimator",
            "base.",
            "_SafeCachedObjective",
            "penalty_residual",
            "invalid_trial_evaluations",
            "_reference_wrench_scaling",
        ):
            if token in sg_text:
                raise RuntimeError(
                    f"independent SG postcondition failed; remaining token: {token}"
                )
        if "def _write_parameters_pdf(" not in sg_text:
            raise RuntimeError("local PDF writer is missing from independent SG estimator")
        if '_write_parameters_pdf(output_directory / "delay_profile.pdf", lines)' not in sg_text:
            raise RuntimeError("split-lag report does not call its local PDF writer")
        spline_actual = blob_sha(
            paths["deterministic_spline_dynamics_estimator.py"].read_bytes()
        )
        if spline_actual != EXPECTED["deterministic_spline_dynamics_estimator.py"]:
            raise RuntimeError("spline estimator changed although it must remain untouched")
    except Exception:
        for name in written:
            paths[name].write_bytes(originals[name])
        raise

    print("patched 8be7473 -> independent SG estimator")
    print("  deterministic_spline_dynamics_estimator.py: unchanged")
    print("  deterministic_savgol_dynamics_estimator.py: self-contained")
    print("  objective: acceleration/angular-acceleration residual")
    print("  lags: separate rotor and gimbal")
    print("  initial lag: one measured publish period per channel")
    print("  smooth half-widths: 4, 2, 1, 0.5 publish periods; overlap allowed")
    print("  strict lag search: adaptive 2-D publish-period grid")
    print("  fixed +/-4 ms / 1 ms / top-k: removed")
    print("  artificial invalid-trial penalty handling: removed")
    print("  delay_profile PDF writer: local to SG estimator")
    print("  backups: none")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
