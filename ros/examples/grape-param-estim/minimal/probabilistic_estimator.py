#!/usr/bin/env python3
"""Minimal bridge to the GUI backend's sparse trajectory Laplace-EM core.

This module deliberately bypasses the GUI, worker process, project state,
lag profiling, MCMC, and production artifact bundle.  It retains the core
scientific path that is under investigation: a latent full-trajectory MAP
with analytic factor Jacobians, sparse LM or IEKS linearized solves, a
Laplace covariance pass, and a six-component diagonal body-wrench Q update.
The command delay remains fixed so Q/state/static-parameter behaviour can be
examined before adding the GUI backend's outer lag profile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import deterministic_estimator as baseline
from grape_param_estim.batch.em_loop import (
    EStepPhase,
    LaplaceEStepFailure,
    LaplaceEmResult,
    LaplaceEmSettings,
    run_fixed_q,
    run_laplace_em,
)
from grape_param_estim.batch.graph_builder import build_initial_batch_state
from grape_param_estim.batch_artifact_export import _factor_payload
from grape_param_estim.batch.laplace_em import (
    BODY_WRENCH_COMPONENT_NAMES,
    BODY_WRENCH_COMPONENT_UNITS,
    BODY_WRENCH_QUANTITY,
    QIntervalModel,
)
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.batch.preparation import (
    PreparationSelection,
    prepare_fixed_batch_graph_data,
)
from grape_param_estim.batch.dynamics_moments import (
    evaluate_prepared_dynamics_intervals,
)
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_request import (
    ACCELEROMETER_BIAS_PRIOR_COVARIANCE_BLOCKS,
    BATCH_ESTIMATION_REQUEST_SCHEMA,
    FIXED_FACTOR_COVARIANCE_BLOCKS,
    INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS,
    OBSERVATION_COVARIANCE_BLOCKS,
    OBSERVATION_FACTOR_NAMES,
    validate_batch_estimation_request,
)
from grape_param_estim.estimation import (
    FixedGraphSolveFailure,
    FixedGraphLaplaceSolution,
    solve_fixed_graph_laplace,
)
from grape_param_estim.real_estimation import (
    prepare_real_estimation_inputs,
)


PROBABILISTIC_SCHEMA = "grape-param-estim/minimal-probabilistic-bridge/v1"


def _diagonal_covariance(
    contract: tuple[tuple[str, ...], tuple[str, ...]],
    *,
    source: str,
) -> dict[str, Any]:
    coordinates, units = contract
    return {
        "source": source,
        "representation": "diagonal",
        "coordinates": list(coordinates),
        "units": list(units),
        "values": [1.0] * len(coordinates),
    }


def _bag_factor_configuration() -> dict[str, Any]:
    factors = {
        name: {
            "enabled": True,
            "disabled_reason": None,
            "covariances": {
                block_name: _diagonal_covariance(
                    contract,
                    source="project_configuration",
                )
                for block_name, contract in (
                    OBSERVATION_COVARIANCE_BLOCKS[name].items()
                )
            },
        }
        for name in OBSERVATION_FACTOR_NAMES
    }
    initial_contracts = dict(INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS)
    initial_contracts.update(ACCELEROMETER_BIAS_PRIOR_COVARIANCE_BLOCKS)
    return {
        "observation_factors": factors,
        "fixed_factor_covariances": {
            name: _diagonal_covariance(
                contract,
                source="numerical_tolerance",
            )
            for name, contract in FIXED_FACTOR_COVARIANCE_BLOCKS.items()
        },
        "initial_state_prior_covariances": {
            name: _diagonal_covariance(
                contract,
                source="project_configuration",
            )
            for name, contract in initial_contracts.items()
        },
    }


def _request_payload(
    arguments: argparse.Namespace,
    baseline_coordinates: np.ndarray,
    output_directory: Path,
) -> dict[str, Any]:
    bag = arguments.bag.expanduser().resolve()
    bag_id = "minimal-flight"
    q_policy = str(arguments.q_policy)
    payload = {
        "schema": BATCH_ESTIMATION_REQUEST_SCHEMA,
        "run_id": "minimal-probabilistic",
        "run_mode": "estimate_only",
        "resume": False,
        "output_directory": str(output_directory.resolve()),
        "bags": [
            {
                "bag_id": bag_id,
                "path": str(bag),
                "sha256": "sha256:" + baseline._sha256(bag),
                "interval_seconds": [arguments.start, arguments.end],
                **_bag_factor_configuration(),
            }
        ],
        "q": {
            "update_policy": q_policy,
            "residual_quantity": BODY_WRENCH_QUANTITY,
            "interval_model": QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY.value,
            "component_names": list(BODY_WRENCH_COMPONENT_NAMES),
            "component_units": list(BODY_WRENCH_COMPONENT_UNITS),
            "initial_diagonal": list(arguments.initial_q),
            "floor_diagonal": list(arguments.q_floor),
        },
        "parameter_prior": {
            "kind": "gaussian",
            "mean_coordinate": baseline_coordinates.tolist(),
            "covariance": np.eye(18).tolist(),
        },
        "delay": {
            "prior_kind": "uniform",
            "bounds_seconds": [0.0, 0.08],
            "initial_seconds": arguments.command_delay,
            "coarse_grid_points": 3,
            "refinement_tolerance_seconds": 1.0e-4,
            "maximum_refinement_evaluations": 2,
        },
        "actuator_model": {
            "source": "minimal bridge matching GUI project defaults",
            "thrust_time_constant_seconds": 0.01,
            "gimbal_time_constant_seconds": 0.02,
            "minimum_thrust_newtons": 1.5,
            "maximum_thrust_newtons": 27.6145,
            "maximum_gimbal_angle_radians": 3.14,
            "maximum_gimbal_rate_radians_per_second": 6.0,
        },
        "knot_policy": {
            "period_seconds": arguments.sample_step,
            "origin": "interval_start",
            "maximum_measurement_gap_seconds": 0.06,
        },
        "interpolation_policy": {
            "euclidean": "linear",
            "orientation": "so3_geodesic",
            "command": "zoh_record_issue_time",
            "allow_extrapolation": False,
        },
        "controller_snapshot_policy": {
            "source": "bag_startup_parameter_updates",
            "require_constant_within_interval": True,
        },
        "mode_hypotheses": [
            {
                "mode_id": "recorded-mode",
                "bag_schedules": {
                    bag_id: {
                        "flight_state_source": "recorded_causal_schedule",
                        "integration_gate_source": "deterministic_replay",
                    }
                },
            }
        ],
        "solver_settings": {
            "method": arguments.probabilistic_solver,
            "maximum_iterations": arguments.probabilistic_max_iterations,
            "maximum_factorization_retries": 4,
            "maximum_model_evaluation_retries": 4,
            "acceptance_ratio": 1.0e-4,
            "gradient_tolerance": 1.0e-6,
            "scaled_step_tolerance": 1.0e-7,
            "relative_objective_tolerance": 1.0e-8,
            "initial_damping": 1.0e-3,
            "minimum_damping": 1.0e-12,
            "maximum_damping": 1.0e12,
        },
        "em_settings": {
            "maximum_iterations": arguments.q_em_iterations,
            "minimum_iterations": 1,
            "maximum_repeated_q_rejections": 3,
            "maximum_repeated_lag_profile_failures": 3,
            "log_q_tolerance": 1.0e-3,
            "lag_tolerance": 1.0e-5,
            "map_objective_tolerance": 1.0e-5,
            "marginal_objective_tolerance": 1.0e-5,
            "q_acceptance_objective_tolerance": 0.0,
            "q_minimum_alpha": 1.0 / 64.0,
        },
        "mcmc_settings": {"enabled": False},
    }
    return payload


def _static_coordinate(state: BatchState) -> np.ndarray:
    key = next(
        key
        for key in state.layout.variable_keys
        if key.kind is VariableKind.STATIC_PARAMETERS
    )
    return np.asarray(state.value(key), dtype=float)


@dataclass
class _FixedLagLaplaceSolver:
    graph_factory: Any
    initial_static_coordinate: np.ndarray
    fixed_delay: float
    lm_settings: LMSettings

    def __post_init__(self) -> None:
        self._result_solutions: dict[int, tuple[Any, FixedGraphLaplaceSolution]] = {}
        self._state_solutions: dict[int, FixedGraphLaplaceSolution] = {}

    def _remember(self, solution: FixedGraphLaplaceSolution, reason: str):
        result = solution.as_e_step_result(
            termination_reason=reason,
        )
        self._result_solutions[id(result)] = (result, solution)
        self._state_solutions[id(solution.lm.state)] = solution
        return result

    def __call__(
        self,
        q: np.ndarray,
        phase: EStepPhase,
        lag: float,
        warm_start: Optional[BatchState],
    ):
        if phase is EStepPhase.LOCAL_LAG_PROFILE and warm_start is not None:
            cached = self._state_solutions.get(id(warm_start))
            if (
                cached is not None
                and np.array_equal(cached.prepared.dynamics.q, q)
            ):
                return self._remember(cached, "fixed_lag_reuse")
        static = (
            self.initial_static_coordinate
            if warm_start is None
            else _static_coordinate(warm_start)
        )

        def progress(record: Any) -> None:
            print(
                "probabilistic MAP iteration {:2d}: objective={:.9g} "
                "accepted={} damping={:.3g}".format(
                    record.iteration + 1,
                    record.objective_before,
                    record.accepted,
                    record.damping_after,
                ),
                flush=True,
            )

        try:
            solution = solve_fixed_graph_laplace(
                self.graph_factory,
                np.asarray(q, dtype=float),
                self.fixed_delay,
                static,
                self.lm_settings,
                warm_start=warm_start,
                lm_progress=progress,
            )
        except FixedGraphSolveFailure as error:
            raise LaplaceEStepFailure(
                error.reason,
                error.inner_iterations,
                detail="fixed-delay minimal bridge",
            ) from error
        return self._remember(
            solution,
            "fixed_lag_{}".format(phase.value),
        )

    def take_solution(self, result: Any) -> FixedGraphLaplaceSolution:
        selected = self._result_solutions.get(id(result))
        if selected is None or selected[0] is not result:
            raise ValueError("final E-step has no retained Laplace solution")
        return selected[1]


def _baseline_matches(
    payload: Mapping[str, Any], arguments: argparse.Namespace
) -> bool:
    try:
        return (
            payload["schema"]
            == "grape-param-estim/minimal-direct-shooting/v2"
            and Path(payload["bag"]["path"]).resolve()
            == arguments.bag.expanduser().resolve()
            and payload["bag"]["requested_interval_seconds"]
            == [arguments.start, arguments.end]
            and payload["model"]["sample_step_seconds"]
            == arguments.sample_step
            and payload["model"]["integration_step_seconds"]
            == arguments.integration_step
            and payload["model"]["fixed_parameters"][
                "command_delay_seconds"
            ]
            == arguments.command_delay
        )
    except (KeyError, TypeError, ValueError):
        return False


def _load_or_run_baseline(arguments: argparse.Namespace) -> Mapping[str, Any]:
    output = arguments.output_dir.expanduser().resolve()
    path = output / "result.json"
    payload = None
    if path.is_file() and not arguments.recompute_baseline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
    if payload is None or not _baseline_matches(payload, arguments):
        print("running deterministic baseline before probabilistic bridge")
        baseline.run(arguments)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        print("reusing matching deterministic baseline {}".format(path))
    return payload


def _em_history(em: LaplaceEmResult) -> list[dict[str, Any]]:
    result = []
    for record in em.iterations:
        result.append(
            {
                "iteration": record.iteration,
                "input_q": record.input_step.q,
                "q_target": record.q_target.target,
                "q_target_map_contribution": (
                    record.q_target.map_second_moment
                ),
                "q_target_covariance_contribution": (
                    record.q_target.covariance_correction
                ),
                "q_accepted": record.q_update.accepted,
                "accepted_alpha": record.q_update.accepted_alpha,
                "output_q": record.output_step.q,
                "map_objective": record.output_step.map_objective,
                "approximate_marginal_objective": (
                    record.output_step.approximate_marginal_objective
                ),
                "inner_iterations": record.output_step.inner_iterations,
                "termination_reason": record.output_step.termination_reason,
            }
        )
    return result


def _plot_method_comparison(
    path: Path,
    problem: baseline.DirectShootingProblem,
    nominal: baseline.Simulation,
    deterministic: baseline.Simulation,
    probabilistic: baseline.Simulation,
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    observed = problem.observations
    styles = (
        ("observed (rosbag)", "#1e5abe", "-", 2.2),
        ("nominal rollout", "#d2691e", "--", 1.3),
        ("deterministic baseline", "#1e965f", ":", 1.7),
        ("probabilistic bridge", "#8b4bb7", "-.", 1.7),
    )
    simulations = (None, nominal, deterministic, probabilistic)
    with baseline.PdfPages(path) as pdf:
        figure = baseline.plt.figure(
            figsize=(11.7, 8.3), constrained_layout=True
        )
        figure.suptitle("Minimal estimator method comparison")
        grid = figure.add_gridspec(2, 2)
        axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
        positions = (
            observed.sensor_position,
            nominal.sensor_position,
            deterministic.sensor_position,
            probabilistic.sensor_position,
        )
        for position, style in zip(positions, styles):
            label, color, line_style, width = style
            axis_3d.plot(
                position[:, 0],
                position[:, 1],
                position[:, 2],
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=width,
            )
        axis_3d.set_xlabel("x [m]")
        axis_3d.set_ylabel("y [m]")
        axis_3d.set_zlabel("z [m]")
        axis_3d.legend(loc="best", fontsize=7)
        metric_axis = figure.add_subplot(grid[0, 1])
        metric_axis.axis("off")
        lines = [
            "metric                   nominal deterministic probabilistic"
        ]
        for key, label in (
            ("position_rmse_m", "position [m]"),
            ("orientation_angle_rmse_deg", "orientation [deg]"),
            ("velocity_rmse_m_per_s", "velocity [m/s]"),
        ):
            lines.append(
                "{:<22s} {:>8.4g} {:>13.4g} {:>13.4g}".format(
                    label,
                    metrics["nominal"][key],
                    metrics["deterministic"][key],
                    metrics["probabilistic"][key],
                )
            )
        metric_axis.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=8,
        )
        error_axis = figure.add_subplot(grid[1, 1])
        relative_time = observed.time - observed.time[0]
        for simulation, style in zip(simulations[1:], styles[1:]):
            label, color, line_style, width = style
            error_axis.plot(
                relative_time,
                np.linalg.norm(
                    simulation.sensor_position
                    - observed.sensor_position,
                    axis=1,
                ),
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=width,
            )
        error_axis.set_xlabel("time [s]")
        error_axis.set_ylabel("position error norm [m]")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend(loc="best", fontsize=7)
        pdf.savefig(figure)
        baseline.plt.close(figure)

        observed_values = (
            observed.sensor_position,
            baseline._rpy_series(observed.sensor_orientation_xyzw),
            observed.sensor_velocity_world,
            observed.angular_velocity_sensor,
            observed.specific_force_sensor,
        )
        simulation_values = tuple(
            (
                simulation.sensor_position,
                baseline._rpy_series(
                    simulation.sensor_orientation_xyzw
                ),
                simulation.sensor_velocity_world,
                simulation.angular_velocity_sensor,
                simulation.specific_force_sensor,
            )
            for simulation in simulations[1:]
        )
        for index, (title, labels, unit) in enumerate(
            (
                ("Sensor position", ("x", "y", "z"), "[m]"),
                ("Sensor orientation RPY", ("roll", "pitch", "yaw"), "[rad]"),
                ("Sensor velocity", ("vx", "vy", "vz"), "[m/s]"),
                ("IMU angular velocity", ("wx", "wy", "wz"), "[rad/s]"),
                ("IMU specific force", ("fx", "fy", "fz"), "[m/s²]"),
            )
        ):
            figure, axes = baseline.plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            figure.suptitle(title)
            all_values = (observed_values[index],) + tuple(
                values[index] for values in simulation_values
            )
            for component, axis in enumerate(axes):
                for values, style in zip(all_values, styles):
                    label, color, line_style, width = style
                    axis.plot(
                        relative_time,
                        values[:, component],
                        label=label,
                        color=color,
                        linestyle=line_style,
                        linewidth=width,
                    )
                axis.set_ylabel("{} {}".format(labels[component], unit))
                axis.grid(True, alpha=0.25)
            axes[0].legend(loc="best", fontsize=7)
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            baseline.plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = baseline.create_argument_parser()
    parser.description = (
        "Run a fixed-lag minimal bridge to the GUI sparse full-trajectory "
        "MAP and diagonal-Q Laplace-EM backend."
    )
    parser.add_argument(
        "--q-policy",
        choices=("fixed", "laplace_em"),
        default="fixed",
    )
    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=6,
        default=(25.0, 25.0, 25.0, 1.0, 1.0, 1.0),
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
    )
    parser.add_argument(
        "--q-floor",
        type=float,
        nargs=6,
        default=(1.0e-8,) * 6,
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
    )
    parser.add_argument("--q-em-iterations", type=int, default=2)
    parser.add_argument(
        "--probabilistic-solver",
        choices=("sparse_lm", "ieks"),
        default="sparse_lm",
    )
    parser.add_argument("--probabilistic-max-iterations", type=int, default=30)
    parser.add_argument("--recompute-baseline", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> int:
    started = time.perf_counter()
    if (
        arguments.q_em_iterations < 1
        or arguments.probabilistic_max_iterations < 1
        or any(value <= 0.0 for value in arguments.initial_q)
        or any(value <= 0.0 for value in arguments.q_floor)
    ):
        raise SystemExit("probabilistic iteration and Q settings are invalid")
    baseline_payload = _load_or_run_baseline(arguments)
    baseline_coordinates = np.asarray(
        baseline_payload["coordinates"]["estimated_full"], dtype=float
    )
    output_root = arguments.output_dir.expanduser().resolve()
    policy_directory = (
        "fixed_q" if arguments.q_policy == "fixed" else "laplace_em"
    )
    probabilistic_output = output_root / "probabilistic" / policy_directory
    probabilistic_output.mkdir(parents=True, exist_ok=True)
    request_payload = _request_payload(
        arguments,
        baseline_coordinates,
        probabilistic_output / "backend-unused",
    )
    request = validate_batch_estimation_request(request_payload)
    print("preparing GUI-backend full-trajectory graph", flush=True)
    inputs = prepare_real_estimation_inputs(
        request,
        progress=lambda stage, done, total, message: print(
            "{} {}/{}: {}".format(stage, done, total, message),
            flush=True,
        ),
    )
    mode_id = "recorded-mode"

    def graph_factory(q: np.ndarray, delay: float, static: np.ndarray):
        return prepare_fixed_batch_graph_data(
            request=request,
            flight_data=inputs.flight_data,
            initializations=inputs.initializations,
            parameter_chart=inputs.parameter_chart,
            geometry=inputs.geometry,
            actuator_parameters=inputs.actuator_parameters,
            scaling=inputs.scaling,
            selection=PreparationSelection(
                mode_id=mode_id,
                fixed_delay_seconds=delay,
                q_diagonal=q,
                initial_parameter_coordinates=static,
            ),
        )

    initial_q = np.asarray(arguments.initial_q, dtype=float)
    q_floor = np.asarray(arguments.q_floor, dtype=float)
    prepared_initial = graph_factory(
        initial_q,
        arguments.command_delay,
        baseline_coordinates,
    )
    initial_state = build_initial_batch_state(prepared_initial)
    initial_dynamics = evaluate_prepared_dynamics_intervals(
        prepared_initial,
        initial_state,
    )
    solver = _FixedLagLaplaceSolver(
        graph_factory=graph_factory,
        initial_static_coordinate=baseline_coordinates,
        fixed_delay=arguments.command_delay,
        lm_settings=LMSettings(**dict(request.payload["solver_settings"])),
    )
    common = dict(
        definition=prepared_initial.dynamics.q_definition,
        q_floor=q_floor,
        interval_time_steps=initial_dynamics.time_step,
        initial_lag=arguments.command_delay,
        solver=solver,
        initial_warm_start=initial_state,
        progress=lambda record: print(
            "Q iteration {}: accepted={} Q={}".format(
                record.iteration + 1,
                record.q_update.accepted,
                np.array2string(record.output_step.q, precision=6),
            ),
            flush=True,
        ),
    )
    if arguments.q_policy == "fixed":
        em = run_fixed_q(fixed_q=initial_q, **common)
    else:
        em = run_laplace_em(
            initial_q=initial_q,
            settings=LaplaceEmSettings(
                **dict(request.payload["em_settings"])
            ),
            **common,
        )
    final_solution = solver.take_solution(em.final_step)
    final_coordinates = _static_coordinate(final_solution.lm.state)
    _audits, objective_components, _per_bag_factors = _factor_payload(
        final_solution
    )
    geometry = final_solution.static_geometry()
    direct_problem = baseline.DirectShootingProblem(
        flight=inputs.flight_data[0],
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=arguments.command_delay,
        prior_weight=arguments.prior_weight,
    )
    nominal_simulation = direct_problem.simulate_full_coordinates(
        np.zeros(18)
    )
    deterministic_simulation = direct_problem.simulate_full_coordinates(
        baseline_coordinates
    )
    probabilistic_simulation = direct_problem.simulate_full_coordinates(
        final_coordinates
    )
    metrics = {
        "nominal": baseline._metrics(direct_problem, nominal_simulation),
        "deterministic": baseline._metrics(
            direct_problem, deterministic_simulation
        ),
        "probabilistic": baseline._metrics(
            direct_problem, probabilistic_simulation
        ),
    }
    result_payload = {
        "schema": PROBABILISTIC_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "trajectory": "GUI-backend latent full-trajectory MAP",
            "linearized_solver": arguments.probabilistic_solver,
            "q": (
                "fixed"
                if arguments.q_policy == "fixed"
                else "GUI-backend diagonal body-wrench Laplace-EM"
            ),
            "fixed_delay_seconds": arguments.command_delay,
            "lag_profile": False,
            "mcmc": False,
            "analytic_factor_jacobians": True,
        },
        "bag": {
            "path": str(arguments.bag.expanduser().resolve()),
            "interval_seconds": [arguments.start, arguments.end],
        },
        "baseline": {
            "schema": baseline_payload["schema"],
            "result": "../../result.json",
            "estimated_full_coordinates": baseline_coordinates,
        },
        "graph": {
            "total_dimension": final_solution.problem.layout.total_dimension,
            "knot_count": len(final_solution.prepared.bags[0].knots),
            "factor_count": len(final_solution.final_linearization.factors),
        },
        "q": {
            "initial_diagonal": initial_q,
            "floor_diagonal": q_floor,
            "final_diagonal": em.final_step.q,
            "history": _em_history(em),
            "termination_reason": em.reason.value,
        },
        "optimizer": {
            "converged": final_solution.lm.converged,
            "termination_reason": final_solution.lm.reason.value,
            "objective": final_solution.lm.objective,
            "iterations": len(final_solution.lm.iterations),
            "gradient_inf_norm": final_solution.lm.final_gradient_inf_norm,
            "approximate_marginal_objective": (
                final_solution.marginal_objective.value
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "objective_components": objective_components,
        "local_parameter_geometry": {
            "parameter_names": geometry.information.parameter_names,
            "likelihood_eigenvalues": (
                geometry.information.likelihood.eigenvalues
            ),
            "posterior_eigenvalues": (
                geometry.information.posterior.eigenvalues
            ),
            "likelihood_effective_rank": (
                geometry.information.likelihood.effective_rank
            ),
            "posterior_effective_rank": (
                geometry.information.posterior.effective_rank
            ),
            "exact_ridge_alignment": geometry.ridge_alignment,
        },
        "coordinates": {
            "deterministic_baseline_full": baseline_coordinates,
            "probabilistic_map_full": final_coordinates,
        },
        "parameters": {
            "deterministic_baseline": baseline._physical_parameters(
                direct_problem.chart.decode(baseline_coordinates)
            ),
            "probabilistic_map": baseline._physical_parameters(
                direct_problem.chart.decode(final_coordinates)
            ),
        },
        "recorded_control_open_loop_metrics": metrics,
        "limitations": [
            "constant command delay is fixed rather than profiled",
            "MCMC and posterior parameter sampling are not included",
            "GUI project, worker, checkpoint, and artifact orchestration are bypassed",
            "factor covariance values currently match the GUI project's provisional unit defaults",
        ],
        "outputs": {
            "trajectory_pdf": "trajectory.pdf",
            "method_comparison_pdf": "method_comparison.pdf",
        },
    }
    baseline._write_json(
        probabilistic_output / "result.json",
        result_payload,
    )
    baseline._write_pdf(
        probabilistic_output / "trajectory.pdf",
        direct_problem,
        nominal_simulation,
        probabilistic_simulation,
        metrics["nominal"],
        metrics["probabilistic"],
    )
    _plot_method_comparison(
        probabilistic_output / "method_comparison.pdf",
        direct_problem,
        nominal_simulation,
        deterministic_simulation,
        probabilistic_simulation,
        metrics,
    )
    print(
        "probabilistic metrics: {}".format(
            json.dumps(metrics["probabilistic"], sort_keys=True)
        )
    )
    print("wrote {}".format(probabilistic_output / "result.json"))
    print(
        "wrote {}".format(probabilistic_output / "method_comparison.pdf")
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
