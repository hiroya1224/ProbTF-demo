#!/usr/bin/env python3
"""Differentiable lag search followed by strict-ZOH multiple-shooting polish."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import deterministic_estimator as baseline
from . import deterministic_multiple_shooting_estimator as strict
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.system import ActuatorCommand, ActuatorState
from smooth_command import QuinticSmoothZoh


SCHEMA = "grape-param-estim/minimal-deterministic-smooth-lag-multiple-shooting/v1"
OUTPUT_SUBDIRECTORY = "deterministic_smooth_lag_multiple_shooting"
GLOBAL_DIMENSION = 14
DELAY_INDEX = 13


def _command(
    thrust: np.ndarray,
    gimbal: np.ndarray,
) -> ActuatorCommand:
    return ActuatorCommand(
        thrust=thrust,
        gimbal_angle=gimbal,
        virtual_force=np.zeros(8, dtype=float),
        desired_acceleration=np.zeros(6, dtype=float),
    )


class SmoothLagMultipleShootingProblem(strict.MultipleShootingProblem):
    """Fourteen-global-coordinate multiple shooting with smooth command lag."""

    def __init__(
        self,
        *,
        flight: Any,
        sample_step: float,
        integration_step: float,
        initial_delay: float,
        width_fraction: float,
        segment_duration: float,
        body_displacement_scale: float,
        prior_weight: float,
        node_position_bound: float,
        node_orientation_bound: float,
        node_velocity_bound: float,
        node_angular_velocity_bound: float,
    ) -> None:
        self.initial_delay = float(initial_delay)
        self.width_fraction = float(width_fraction)
        self.body_displacement_scale = float(body_displacement_scale)
        if self.body_displacement_scale <= 0.0:
            raise ValueError("body displacement scale must be positive")
        direct_problem = baseline.DirectShootingProblem(
            flight=flight,
            sample_step=sample_step,
            integration_step=integration_step,
            command_delay=self.initial_delay,
            prior_weight=prior_weight,
        )
        super().__init__(
            direct_problem=direct_problem,
            delay=self.initial_delay,
            segment_duration=segment_duration,
            prior_weight=prior_weight,
            node_position_bound=node_position_bound,
            node_orientation_bound=node_orientation_bound,
            node_velocity_bound=node_velocity_bound,
            node_angular_velocity_bound=node_angular_velocity_bound,
            global_dimension=GLOBAL_DIMENSION,
        )
        self.pose_residual_factor = (
            self.pose_residual_factor / self.body_displacement_scale
        )
        self.rotor_command_history = QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        )
        self.gimbal_command_history = QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        )

    def set_command_width_fraction(self, width_fraction: float) -> None:
        value = float(width_fraction)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("smoothstep width fraction must be positive")
        self.width_fraction = value

    def initial_coordinate(self) -> np.ndarray:
        result = super().initial_coordinate()
        result[DELAY_INDEX] = self.initial_delay
        return result

    def coordinate_delay(self, coordinate: Sequence[float]) -> float:
        global_coordinate, _nodes = self.split_coordinate(coordinate)
        return float(global_coordinate[DELAY_INDEX])

    def _decode_global_coordinate(
        self,
        global_coordinate: Sequence[float],
    ) -> tuple[Any, Any]:
        value = np.asarray(global_coordinate, dtype=float)
        if value.shape != (GLOBAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("smooth-lag global coordinate must be finite and 14-D")
        decoded, physical_jacobian = strict._physical_parameter_jacobian(
            self.parameterization,
            value[: strict.PHYSICAL_DIMENSION],
            float(value[DELAY_INDEX]),
        )
        return decoded, strict._extend_parameter_jacobian(
            physical_jacobian,
            GLOBAL_DIMENSION,
        )

    def _command_with_sensitivity(
        self,
        step_index: int,
        local_dimension: int,
    ) -> tuple[ActuatorCommand, np.ndarray]:
        delay = float(self._active_delay)
        midpoint_time = (
            float(self.direct_problem.internal_time[step_index])
            + 0.5 * self.direct_problem.integration_step
        )
        rotor = self.rotor_command_history.evaluate(
            midpoint_time,
            delay,
            self.width_fraction,
        )
        gimbal = self.gimbal_command_history.evaluate(
            midpoint_time,
            delay,
            self.width_fraction,
        )
        sensitivity = np.zeros((8, local_dimension), dtype=float)
        sensitivity[:4, DELAY_INDEX] = rotor.delay_derivative
        sensitivity[4:, DELAY_INDEX] = gimbal.delay_derivative
        return _command(rotor.value, gimbal.value), sensitivity

    def _initial_state_with_sensitivity(self, *args: Any, **kwargs: Any) -> Any:
        result = list(super()._initial_state_with_sensitivity(*args, **kwargs))
        decoded = args[0] if args else kwargs["decoded"]
        segment_index = args[2] if len(args) >= 3 else kwargs["segment_index"]
        if segment_index == 0:
            rotor = self.rotor_command_history.evaluate(
                float(self.direct_problem.internal_time[0]),
                float(decoded.delay),
                self.width_fraction,
            )
            actuator = result[1]
            result[1] = ActuatorState(
                thrust=rotor.value,
                gimbal_angle=actuator.gimbal_angle,
            )
            result[3][:4, DELAY_INDEX] = rotor.delay_derivative
        return tuple(result)

    def evaluate(self, coordinate: Sequence[float]) -> strict.ProblemEvaluation:
        self._active_delay = self.coordinate_delay(coordinate)
        return super().evaluate(coordinate)

    def full_rollout(
        self,
        global_coordinate: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = np.asarray(global_coordinate, dtype=float)
        self._active_delay = float(value[DELAY_INDEX])
        return super().full_rollout(value)


def _global_bounds(
    delay_bounds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(GLOBAL_DIMENSION, -np.inf, dtype=float)
    upper = np.full(GLOBAL_DIMENSION, np.inf, dtype=float)
    lower[DELAY_INDEX], upper[DELAY_INDEX] = (
        float(delay_bounds[0]),
        float(delay_bounds[1]),
    )
    return lower, upper


def _strict_problem(
    flight: Any,
    delay: float,
    arguments: argparse.Namespace,
) -> strict.MultipleShootingProblem:
    direct_problem = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=float(delay),
        prior_weight=arguments.prior_weight,
    )
    return strict.MultipleShootingProblem(
        direct_problem=direct_problem,
        delay=float(delay),
        segment_duration=arguments.segment_duration,
        prior_weight=arguments.prior_weight,
        node_position_bound=arguments.node_position_bound,
        node_orientation_bound=arguments.node_orientation_bound,
        node_velocity_bound=arguments.node_velocity_bound,
        node_angular_velocity_bound=arguments.node_angular_velocity_bound,
    )


def _strict_initial_from_smooth(
    target: strict.MultipleShootingProblem,
    source: SmoothLagMultipleShootingProblem,
    coordinate: Sequence[float],
) -> np.ndarray:
    global_coordinate, source_nodes = source.split_coordinate(coordinate)
    result = target.initial_coordinate()
    result[: strict.PHYSICAL_DIMENSION] = global_coordinate[
        : strict.PHYSICAL_DIMENSION
    ]
    if source.node_count != target.node_count:
        raise ValueError("smooth and strict shooting schedules differ")
    for index in range(source.node_count):
        rigid, actuator, _rigid_jacobian, _actuator_jacobian = (
            strict._decode_node(source.node_references[index], source_nodes[index])
        )
        start = strict.PHYSICAL_DIMENSION + index * strict.NODE_DIMENSION
        result[start : start + strict.NODE_DIMENSION] = strict._encode_node(
            target.node_references[index],
            rigid,
            actuator,
        )
    return result


def zoh_polish_delays(
    center: float,
    radius: float,
    step: float,
    delay_bounds: Sequence[float],
) -> np.ndarray:
    count = int(math.floor(radius / step + 1.0e-12))
    offsets = np.arange(-count, count + 1, dtype=float) * step
    values = np.clip(float(center) + offsets, delay_bounds[0], delay_bounds[1])
    return np.unique(np.round(values, 12))


def _continuity_max(solution: strict.FixedDelaySolution) -> float:
    residual = solution.evaluation.continuity_residual
    return 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))


def _full_loss(solution: strict.FixedDelaySolution) -> float:
    return 0.5 * float(
        solution.full_rollout_residual @ solution.full_rollout_residual
    )


def _smooth_stage_payload(
    problem: SmoothLagMultipleShootingProblem,
    solution: strict.FixedDelaySolution,
    width_fraction: float,
) -> dict[str, Any]:
    global_coordinate, _nodes = problem.split_coordinate(solution.coordinate)
    evaluation = solution.evaluation
    pose = evaluation.data_residual[: problem.pose_residual_dimension]
    delay_gradient = float(
        evaluation.data_jacobian[:, DELAY_INDEX] @ evaluation.data_residual
    )
    return {
        "width_fraction": width_fraction,
        "delay_seconds": float(global_coordinate[DELAY_INDEX]),
        "stitched_inertia_radius_loss_m2": 0.5 * float(pose @ pose),
        "full_rollout_inertia_radius_loss_m2": _full_loss(solution),
        "soft_prior_cost": 0.5
        * problem.prior_weight
        * float(
            (global_coordinate[: strict.PHYSICAL_DIMENSION] / problem.prior_scales)
            @ (
                global_coordinate[: strict.PHYSICAL_DIMENSION]
                / problem.prior_scales
            )
        ),
        "continuity_max_normalized": _continuity_max(solution),
        "lag_data_gradient": delay_gradient,
        "parameters": strict._physical_payload(evaluation.decoded),
        "optimizer_history": list(solution.optimizer_history),
        "elapsed_seconds": solution.elapsed_seconds,
    }


def _write_delay_profile_pdf(
    path: Path,
    smooth_delay: float,
    candidate_delays: np.ndarray,
    unrefined_losses: np.ndarray,
    refined: Sequence[tuple[strict.MultipleShootingProblem, strict.FixedDelaySolution]],
    selected_delay: float,
) -> None:
    figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
    axis.plot(
        candidate_delays * 1000.0,
        unrefined_losses,
        marker="o",
        label="strict ZOH, smooth physical parameters",
    )
    if refined:
        axis.scatter(
            [solution.delay * 1000.0 for _problem, solution in refined],
            [_full_loss(solution) for _problem, solution in refined],
            marker="s",
            s=70,
            label="strict ZOH, refined",
        )
    axis.axvline(
        smooth_delay * 1000.0,
        color="#9467bd",
        linestyle="--",
        label="smoothstep estimate",
    )
    selected = min(refined, key=lambda item: abs(item[1].delay - selected_delay))[1]
    axis.scatter(
        [selected_delay * 1000.0],
        [_full_loss(selected)],
        marker="*",
        s=220,
        color="#1e965f",
        label="selected strict ZOH",
        zorder=5,
    )
    axis.set_xlabel("recorded-command delay [ms]")
    axis.set_ylabel("full-rollout inertia-radius loss [m²]")
    axis.set_title("Smooth lag search and strict-ZOH local polish")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    with PdfPages(path) as pdf:
        pdf.savefig(figure)
    plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search command lag with quintic smooth ZOH continuation, then "
            "polish only the best local strict-ZOH candidates."
        )
    )
    parser.add_argument("--bag", type=Path, default=baseline.DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--smooth-max-nfev", type=int, default=60)
    parser.add_argument("--augmented-lagrangian-iterations", type=int, default=10)
    parser.add_argument("--continuity-penalty-initial", type=float, default=1.0)
    parser.add_argument("--continuity-penalty-growth", type=float, default=10.0)
    parser.add_argument("--continuity-penalty-max", type=float, default=1.0e6)
    parser.add_argument("--penalty-reduction-target", type=float, default=0.50)
    parser.add_argument("--continuity-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument("--delay-bounds", type=float, nargs=2, default=(0.0, 0.20))
    parser.add_argument("--initial-delay", type=float, default=0.01)
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(0.50, 0.20, 0.05),
    )
    parser.add_argument("--zoh-polish-radius", type=float, default=0.004)
    parser.add_argument("--zoh-polish-step", type=float, default=0.001)
    parser.add_argument("--zoh-polish-top-k", type=int, default=3)
    parser.add_argument("--body-displacement-scale", type=float, default=1.0)
    parser.add_argument("--node-position-bound", type=float, default=2.0)
    parser.add_argument("--node-orientation-bound", type=float, default=1.5)
    parser.add_argument("--node-velocity-bound", type=float, default=5.0)
    parser.add_argument("--node-angular-velocity-bound", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.segment_duration,
        arguments.max_nfev,
        arguments.smooth_max_nfev,
        arguments.augmented_lagrangian_iterations,
        arguments.continuity_penalty_initial,
        arguments.continuity_penalty_growth,
        arguments.continuity_penalty_max,
        arguments.continuity_tolerance,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.zoh_polish_top_k,
        arguments.body_displacement_scale,
        arguments.node_position_bound,
        arguments.node_orientation_bound,
        arguments.node_velocity_bound,
        arguments.node_angular_velocity_bound,
    )
    bounds = np.asarray(arguments.delay_bounds, dtype=float)
    widths = np.asarray(arguments.smoothstep_width_fractions, dtype=float)
    if (
        not np.isfinite(arguments.start)
        or not np.isfinite(arguments.end)
        or arguments.start >= arguments.end
        or any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or not np.isfinite(arguments.prior_weight)
        or arguments.prior_weight < 0.0
        or arguments.continuity_penalty_growth <= 1.0
        or not 0.0 < arguments.penalty_reduction_target < 1.0
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or not bounds[0] <= arguments.initial_delay <= bounds[1]
        or widths.ndim != 1
        or widths.size < 1
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit("smooth-lag multiple-shooting settings are invalid")


def run(arguments: argparse.Namespace) -> int:
    _validate_arguments(arguments)
    bag = arguments.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit("bag does not exist: {}".format(bag))
    started = time.perf_counter()
    print(
        "loading {} [{:.3f}, {:.3f}] s".format(
            bag,
            arguments.start,
            arguments.end,
        ),
        flush=True,
    )
    flight = load_flight_data(
        str(bag),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    widths = tuple(float(value) for value in arguments.smoothstep_width_fractions)
    smooth_problem = SmoothLagMultipleShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        initial_delay=arguments.initial_delay,
        width_fraction=widths[0],
        segment_duration=arguments.segment_duration,
        body_displacement_scale=arguments.body_displacement_scale,
        prior_weight=arguments.prior_weight,
        node_position_bound=arguments.node_position_bound,
        node_orientation_bound=arguments.node_orientation_bound,
        node_velocity_bound=arguments.node_velocity_bound,
        node_angular_velocity_bound=arguments.node_angular_velocity_bound,
    )
    global_lower, global_upper = _global_bounds(arguments.delay_bounds)
    bounds = smooth_problem.bounds(global_lower, global_upper)
    coordinate = smooth_problem.initial_coordinate()
    smooth_arguments = argparse.Namespace(**vars(arguments))
    smooth_arguments.max_nfev = arguments.smooth_max_nfev
    stage_solutions: list[strict.FixedDelaySolution] = []
    stage_payloads: list[dict[str, Any]] = []
    for stage_index, width in enumerate(widths):
        smooth_problem.set_command_width_fraction(width)
        print(
            "smoothstep stage {}/{}: width_fraction={:.6g}".format(
                stage_index + 1,
                len(widths),
                width,
            ),
            flush=True,
        )
        solution = strict._solve_fixed_delay(
            smooth_problem,
            coordinate,
            bounds,
            smooth_arguments,
        )
        coordinate = solution.coordinate.copy()
        stage_solutions.append(solution)
        stage_payloads.append(_smooth_stage_payload(smooth_problem, solution, width))

    final_smooth = stage_solutions[-1]
    smooth_global, _smooth_nodes = smooth_problem.split_coordinate(
        final_smooth.coordinate
    )
    smooth_delay = float(smooth_global[DELAY_INDEX])
    candidate_delays = zoh_polish_delays(
        smooth_delay,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.delay_bounds,
    )
    strict_problems: dict[float, strict.MultipleShootingProblem] = {}
    unrefined_losses = np.empty(candidate_delays.size, dtype=float)
    for index, delay in enumerate(candidate_delays):
        problem = _strict_problem(flight, float(delay), arguments)
        strict_problems[round(float(delay), 12)] = problem
        _position, _orientation, residual = problem.full_rollout(
            smooth_global[: strict.PHYSICAL_DIMENSION]
        )
        unrefined_losses[index] = 0.5 * float(residual @ residual)
    top_count = min(arguments.zoh_polish_top_k, candidate_delays.size)
    top_indices = np.argsort(unrefined_losses, kind="stable")[:top_count]
    refined: list[
        tuple[strict.MultipleShootingProblem, strict.FixedDelaySolution]
    ] = []
    for rank, candidate_index in enumerate(top_indices):
        delay = float(candidate_delays[candidate_index])
        problem = strict_problems[round(delay, 12)]
        initial = _strict_initial_from_smooth(
            problem,
            smooth_problem,
            final_smooth.coordinate,
        )
        print(
            "strict ZOH polish {}/{}: delay={:.6f}s, screening_loss={:.9g}".format(
                rank + 1,
                top_count,
                delay,
                unrefined_losses[candidate_index],
            ),
            flush=True,
        )
        solution = strict._solve_fixed_delay(
            problem,
            initial,
            problem.bounds(
                np.full(strict.PHYSICAL_DIMENSION, -np.inf),
                np.full(strict.PHYSICAL_DIMENSION, np.inf),
            ),
            arguments,
        )
        refined.append((problem, solution))
    converged = [
        item
        for item in refined
        if _continuity_max(item[1]) <= arguments.continuity_tolerance
    ]
    selected_problem, selected_solution = min(
        converged if converged else refined,
        key=lambda item: _full_loss(item[1]),
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "result.json"
    trajectory_path = output_directory / "trajectory.pdf"
    delay_profile_path = output_directory / "delay_profile.pdf"
    parameters_path = output_directory / "parameters.txt"
    selected_payload = strict._solution_payload(
        selected_problem,
        selected_solution,
        arguments.continuity_tolerance,
    )
    stitched_metrics = selected_payload["stitched_recorded_control_metrics"]
    parameter_lines = strict._parameter_summary_lines(
        selected_problem,
        selected_solution,
        stitched_metrics,
        arguments.continuity_tolerance,
    )
    refined_payloads = []
    for problem, solution in refined:
        payload = strict._solution_payload(
            problem,
            solution,
            arguments.continuity_tolerance,
        )
        payload["screening_loss"] = float(
            unrefined_losses[
                int(np.argmin(np.abs(candidate_delays - solution.delay)))
            ]
        )
        refined_payloads.append(payload)
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "path": str(bag),
            "sha256": baseline._sha256(bag),
            "requested_interval_seconds": [arguments.start, arguments.end],
        },
        "method": {
            "name": "smooth_lag_then_strict_zoh_multiple_shooting",
            "smooth_search_global_dimension": GLOBAL_DIMENSION,
            "physical_dimension": strict.PHYSICAL_DIMENSION,
            "delay_index": DELAY_INDEX,
            "pose_metric": "||rho||^2 + phi^T (J0 / m0) phi",
            "body_displacement_scale_m": arguments.body_displacement_scale,
            "final_model": "strict causal ZOH",
            "physical_jacobian": "analytic forward sensitivity",
            "lag_jacobian": "analytic smooth-command forward sensitivity",
            "smooth_max_nfev_per_augmented_iteration": (
                arguments.smooth_max_nfev
            ),
            "strict_zoh_max_nfev_per_augmented_iteration": arguments.max_nfev,
        },
        "smoothstep_search": {
            "initial_delay_seconds": arguments.initial_delay,
            "delay_bounds_seconds": arguments.delay_bounds,
            "width_fractions": widths,
            "stage_results": stage_payloads,
            "estimated_delay_seconds": smooth_delay,
        },
        "exact_zoh_polish": {
            "candidate_delays_seconds": candidate_delays,
            "unrefined_full_rollout_losses_m2": unrefined_losses,
            "refined_candidates": refined_payloads,
            "selected_delay_seconds": selected_solution.delay,
            "smooth_to_selected_delay_difference_seconds": (
                selected_solution.delay - smooth_delay
            ),
        },
        "selection": selected_payload,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "result_json": "result.json",
            "trajectory_pdf": "trajectory.pdf",
            "delay_profile_pdf": "delay_profile.pdf",
            "parameters_text": "parameters.txt",
        },
    }
    baseline._write_json(result_path, result)
    strict._write_text(parameters_path, parameter_lines)
    strict._write_pdf(
        trajectory_path,
        selected_problem,
        selected_solution,
        stitched_metrics,
        parameter_lines,
        arguments.continuity_tolerance,
    )
    _write_delay_profile_pdf(
        delay_profile_path,
        smooth_delay,
        candidate_delays,
        unrefined_losses,
        refined,
        selected_solution.delay,
    )
    print(
        "smooth delay {:.6f}s -> selected strict ZOH delay {:.6f}s, loss {:.9g}".format(
            smooth_delay,
            selected_solution.delay,
            _full_loss(selected_solution),
        ),
        flush=True,
    )
    for path in (result_path, trajectory_path, delay_profile_path, parameters_path):
        print("wrote {}".format(path), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
