#!/usr/bin/env python3
"""Deterministic single-shooting fit with an inverse-dynamics Q estimate.

The open-loop trajectory objective and thirteen-dimensional parameter chart
remain identical to ``deterministic_estimator``.  A second, observation-side
inverse-dynamics residual computes the body wrench required by the recorded
IMU trajectory minus the wrench predicted from the recorded actuator input.
For fixed diagonal Q, that residual is appended to the least-squares problem.
For fixed parameters, Q has the closed-form Gaussian maximum-likelihood
update.  No latent trajectory, residual-wrench state, sparse smoother, or GUI
backend is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

import deterministic_estimator as baseline
from grape_param_estim.system import ActuatorState, RigidBodyState


SCHEMA = "grape-param-estim/minimal-deterministic-diagonal-q/v1"
Q_COMPONENT_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
WRENCH_COMPONENT_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")
Q_DIAGONAL_UNITS = (
    "N^2*s",
    "N^2*s",
    "N^2*s",
    "N^2*m^2*s",
    "N^2*m^2*s",
    "N^2*m^2*s",
)


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
    output_root = arguments.output_dir.expanduser().resolve()
    path = output_root / "result.json"
    payload = None
    if path.is_file() and not arguments.recompute_baseline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
    if payload is None or not _baseline_matches(payload, arguments):
        print("running deterministic baseline before deterministic-Q fit")
        baseline.run(arguments)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        print("reusing matching deterministic baseline {}".format(path))
    return payload


def _active_coordinate(payload: Mapping[str, Any]) -> np.ndarray:
    values = payload["coordinates"]["estimated_active"]
    result = np.asarray(
        [values[name] for name in baseline.ACTIVE_PARAMETER_NAMES],
        dtype=float,
    )
    if result.shape != (baseline.ACTIVE_PARAMETER_DIMENSION,):
        raise ValueError("baseline active coordinate has the wrong shape")
    return result


def observed_wrench_residual(
    problem: baseline.DirectShootingProblem,
    active: Sequence[float],
    actuator_reference: baseline.Simulation,
) -> np.ndarray:
    """Return required-minus-modeled body wrench on observed left knots."""

    coordinate = np.asarray(active, dtype=float)
    parameters = problem.chart.decode(problem.full_coordinates(coordinate))
    plant = baseline.FullSixDofPlant(parameters, problem.geometry)
    count = problem.output_time.size - 1
    residual = np.empty((count, 6), dtype=float)
    for index in range(count):
        rotation = problem.observed_body_rotation[index]
        omega = problem.smoothed_omega_body[index]
        alpha = problem.observed_angular_acceleration_body[index]
        pose_lever = problem.pose_sensor_position - parameters.cog_offset
        velocity_lever = (
            problem.velocity_sensor_position - parameters.cog_offset
        )
        rigid = RigidBodyState(
            position=(
                problem.observations.sensor_position[index]
                - rotation @ pose_lever
            ),
            orientation_xyzw=baseline.matrix_to_quaternion(rotation),
            linear_velocity=(
                problem.observations.sensor_velocity_world[index]
                - rotation @ np.cross(omega, velocity_lever)
            ),
            angular_velocity=omega,
        )
        actuators = ActuatorState(
            thrust=actuator_reference.actuator_thrust[index],
            gimbal_angle=actuator_reference.actuator_gimbal[index],
        )
        modeled = plant.total_body_wrench(
            float(problem.output_time[index]), rigid, actuators
        )
        imu_lever = problem.imu_sensor_position - parameters.cog_offset
        measured_specific_force_body = (
            problem.body_to_imu_rotation.T
            @ (
                problem.observations.specific_force_sensor[index]
                - problem.accelerometer_bias
            )
        )
        required_force = parameters.mass * (
            measured_specific_force_body
            - np.cross(alpha, imu_lever)
            - np.cross(omega, np.cross(omega, imu_lever))
        )
        required_torque = (
            parameters.inertia @ alpha
            + np.cross(omega, parameters.inertia @ omega)
        )
        residual[index, :3] = required_force - modeled[:3]
        residual[index, 3:] = required_torque - modeled[3:]
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("observed wrench residual is non-finite")
    return residual


def q_target(
    wrench_residual: np.ndarray,
    time_step: np.ndarray,
    floor: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(wrench_residual, dtype=float)
    dt = np.asarray(time_step, dtype=float)
    selected_floor = np.asarray(floor, dtype=float)
    if (
        residual.ndim != 2
        or residual.shape[1] != 6
        or dt.shape != (residual.shape[0],)
        or selected_floor.shape != (6,)
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(dt))
        or np.any(dt <= 0.0)
        or np.any(selected_floor <= 0.0)
    ):
        raise ValueError("Q target inputs are invalid")
    raw = np.mean(dt[:, None] * residual * residual, axis=0)
    return np.maximum(raw, selected_floor)


def q_negative_log_likelihood(
    wrench_residual: np.ndarray,
    time_step: np.ndarray,
    q: np.ndarray,
) -> float:
    """Return mean Gaussian NLL for Sigma_k = Q / dt_k."""

    residual = np.asarray(wrench_residual, dtype=float)
    dt = np.asarray(time_step, dtype=float)
    diagonal = np.asarray(q, dtype=float)
    if (
        residual.ndim != 2
        or residual.shape[1] != 6
        or dt.shape != (residual.shape[0],)
        or diagonal.shape != (6,)
        or np.any(dt <= 0.0)
        or np.any(diagonal <= 0.0)
    ):
        raise ValueError("Q likelihood inputs are invalid")
    mahalanobis = np.sum(
        dt[:, None] * residual * residual / diagonal[None, :], axis=1
    )
    log_determinant = float(np.sum(np.log(diagonal))) - 6.0 * np.log(dt)
    normalizer = 6.0 * math.log(2.0 * math.pi)
    return 0.5 * float(
        np.mean(mahalanobis + log_determinant + normalizer)
    )


def _augmented_residual(
    problem: baseline.DirectShootingProblem,
    active: np.ndarray,
    actuator_reference: baseline.Simulation,
    q: np.ndarray,
    time_step: np.ndarray,
) -> np.ndarray:
    trajectory = problem.residual(active)
    wrench = observed_wrench_residual(problem, active, actuator_reference)
    whitening = (
        np.sqrt(time_step[:, None] / wrench.shape[0])
        / np.sqrt(q)[None, :]
    )
    return np.concatenate((trajectory, (whitening * wrench).ravel()))


def _trajectory_objective(
    problem: baseline.DirectShootingProblem, active: np.ndarray
) -> float:
    residual = problem.residual(active)
    return 0.5 * float(residual @ residual)


def _objective_record(
    problem: baseline.DirectShootingProblem,
    active: np.ndarray,
    actuator_reference: baseline.Simulation,
    q: np.ndarray,
    time_step: np.ndarray,
) -> dict[str, float]:
    trajectory = _trajectory_objective(problem, active)
    q_nll = q_negative_log_likelihood(
        observed_wrench_residual(problem, active, actuator_reference),
        time_step,
        q,
    )
    return {
        "trajectory_and_parameter_prior": trajectory,
        "mean_wrench_gaussian_nll": q_nll,
        "total_composite_objective": trajectory + q_nll,
    }


def _write_comparison_pdf(
    path: Path,
    problem: baseline.DirectShootingProblem,
    nominal: baseline.Simulation,
    deterministic: baseline.Simulation,
    estimated: baseline.Simulation,
) -> None:
    observed = problem.observations
    styles = (
        ("observed", observed.sensor_position, "#1e5abe", "-", 2.2),
        ("nominal", nominal.sensor_position, "#d2691e", "--", 1.4),
        (
            "deterministic",
            deterministic.sensor_position,
            "#1e965f",
            ":",
            1.7,
        ),
        (
            "deterministic + Q",
            estimated.sensor_position,
            "#8b4bb7",
            "-.",
            1.7,
        ),
    )
    relative_time = observed.time - observed.time[0]
    with baseline.PdfPages(path) as pdf:
        figure = baseline.plt.figure(
            figsize=(11.7, 8.3), constrained_layout=True
        )
        axis = figure.add_subplot(111, projection="3d")
        for label, values, color, line_style, width in styles:
            axis.plot(
                values[:, 0],
                values[:, 1],
                values[:, 2],
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=width,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="best")
        pdf.savefig(figure)
        baseline.plt.close(figure)

        figure, axes = baseline.plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            for label, values, color, line_style, width in styles:
                axis.plot(
                    relative_time,
                    values[:, component],
                    label=label,
                    color=color,
                    linestyle=line_style,
                    linewidth=width,
                )
            axis.set_ylabel(("x [m]", "y [m]", "z [m]")[component])
            axis.grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("time [s]")
        pdf.savefig(figure)
        baseline.plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = baseline.create_argument_parser()
    parser.description = (
        "Fit the deterministic open-loop trajectory while alternating a "
        "six-axis observed-wrench diagonal-Q likelihood."
    )
    parser.add_argument("--q-iterations", type=int, default=2)
    parser.add_argument("--q-max-nfev", type=int, default=8)
    parser.add_argument("--q-log-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=6,
        default=None,
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
    )
    parser.add_argument(
        "--q-floor",
        type=float,
        nargs=6,
        default=(1.0e-8,) * 6,
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
    )
    parser.add_argument("--recompute-baseline", action="store_true")
    return parser


def _solve_theta(
    problem: baseline.DirectShootingProblem,
    initial: np.ndarray,
    actuator_reference: baseline.Simulation,
    q: np.ndarray,
    time_step: np.ndarray,
    maximum_evaluations: int,
):
    lower, upper = baseline.parameter_bounds()
    return least_squares(
        lambda active: _augmented_residual(
            problem,
            active,
            actuator_reference,
            q,
            time_step,
        ),
        initial,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="linear",
        ftol=1.0e-6,
        xtol=1.0e-6,
        gtol=1.0e-6,
        max_nfev=maximum_evaluations,
        verbose=1,
    )


def run(arguments: argparse.Namespace) -> int:
    if (
        arguments.q_iterations < 1
        or arguments.q_max_nfev < 1
        or arguments.q_log_tolerance < 0.0
        or any(value <= 0.0 for value in arguments.q_floor)
        or (
            arguments.initial_q is not None
            and any(value <= 0.0 for value in arguments.initial_q)
        )
    ):
        raise SystemExit("deterministic-Q settings are invalid")
    started = time.perf_counter()
    baseline_payload = _load_or_run_baseline(arguments)
    active = _active_coordinate(baseline_payload)
    flight = baseline.load_flight_data(
        str(arguments.bag.expanduser().resolve()),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    problem = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=arguments.command_delay,
        prior_weight=arguments.prior_weight,
    )
    nominal_simulation = problem.simulate(np.zeros(active.size))
    deterministic_simulation = problem.simulate(active)
    actuator_reference = nominal_simulation
    time_step = np.diff(problem.output_time)
    floor = np.asarray(arguments.q_floor, dtype=float)
    initial_residual = observed_wrench_residual(
        problem, active, actuator_reference
    )
    q = (
        q_target(initial_residual, time_step, floor)
        if arguments.initial_q is None
        else np.maximum(np.asarray(arguments.initial_q, dtype=float), floor)
    )
    initial_q = q.copy()
    history = []
    for iteration in range(arguments.q_iterations):
        input_q = q.copy()
        before = _objective_record(
            problem, active, actuator_reference, input_q, time_step
        )
        result = _solve_theta(
            problem,
            active,
            actuator_reference,
            input_q,
            time_step,
            arguments.q_max_nfev,
        )
        active = result.x
        residual = observed_wrench_residual(
            problem, active, actuator_reference
        )
        q = q_target(residual, time_step, floor)
        after = _objective_record(
            problem, active, actuator_reference, q, time_step
        )
        history.append(
            {
                "iteration": iteration,
                "input_q": input_q,
                "target_q": q,
                "maximum_log_q_change": float(
                    np.max(np.abs(np.log(q) - np.log(input_q)))
                ),
                "theta_nfev": result.nfev,
                "theta_success": result.success,
                "theta_message": result.message,
                "objective_before": before,
                "objective_after": after,
            }
        )
        print(
            "deterministic-Q iteration {}: Q={}".format(
                iteration + 1, np.array2string(q, precision=6)
            ),
            flush=True,
        )
        if history[-1]["maximum_log_q_change"] <= arguments.q_log_tolerance:
            break
    final_result = _solve_theta(
        problem,
        active,
        actuator_reference,
        q,
        time_step,
        arguments.q_max_nfev,
    )
    active = final_result.x
    final_wrench_residual = observed_wrench_residual(
        problem, active, actuator_reference
    )
    final_q_target = q_target(final_wrench_residual, time_step, floor)
    final_q_fixed_point_error = float(
        np.max(np.abs(np.log(final_q_target) - np.log(q)))
    )
    final_simulation = problem.simulate(active)
    full_coordinate = problem.full_coordinates(active)
    metrics = {
        "nominal": baseline._metrics(problem, nominal_simulation),
        "deterministic": baseline._metrics(
            problem, deterministic_simulation
        ),
        "deterministic_q": baseline._metrics(problem, final_simulation),
    }
    output = arguments.output_dir.expanduser().resolve() / "deterministic_q"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "trajectory": "deterministic single-shooting open-loop",
            "wrench_residual": (
                "observed inverse dynamics required minus modeled body wrench"
            ),
            "q_update": "closed-form diagonal Gaussian maximum likelihood",
            "latent_trajectory": False,
            "residual_wrench_state": False,
            "fixed_delay_seconds": arguments.command_delay,
        },
        "bag": {
            "path": str(arguments.bag.expanduser().resolve()),
            "interval_seconds": [arguments.start, arguments.end],
        },
        "q": {
            "quantity": "body_wrench_continuous_spectral_density",
            "component_names": Q_COMPONENT_NAMES,
            "residual_component_units": WRENCH_COMPONENT_UNITS,
            "diagonal_units": Q_DIAGONAL_UNITS,
            "interval_covariance": "Q / dt",
            "initial_diagonal": initial_q,
            "floor_diagonal": floor,
            "final_diagonal": q,
            "next_closed_form_target": final_q_target,
            "maximum_log_fixed_point_error": final_q_fixed_point_error,
            "fixed_point_converged": (
                final_q_fixed_point_error <= arguments.q_log_tolerance
            ),
            "history": history,
            "final_mean_gaussian_nll": q_negative_log_likelihood(
                final_wrench_residual, time_step, q
            ),
            "final_map_second_moment": np.mean(
                time_step[:, None] * final_wrench_residual**2, axis=0
            ),
        },
        "optimizer": {
            "name": "scipy.optimize.least_squares alternating with analytic Q MLE",
            "q_iterations": arguments.q_iterations,
            "completed_q_iterations": len(history),
            "q_log_tolerance": arguments.q_log_tolerance,
            "theta_max_nfev_per_solve": arguments.q_max_nfev,
            "final_theta_success": final_result.success,
            "final_theta_message": final_result.message,
            "final_theta_nfev": final_result.nfev,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "coordinates": {
            "deterministic_baseline_full": np.asarray(
                baseline_payload["coordinates"]["estimated_full"], dtype=float
            ),
            "deterministic_q_full": full_coordinate,
            "deterministic_q_active": {
                name: value
                for name, value in zip(
                    baseline.ACTIVE_PARAMETER_NAMES, active
                )
            },
        },
        "parameters": {
            "deterministic_q": baseline._physical_parameters(
                problem.chart.decode(full_coordinate)
            )
        },
        "recorded_control_open_loop_metrics": metrics,
        "final_objective": _objective_record(
            problem, active, actuator_reference, q, time_step
        ),
        "limitations": [
            "the open-loop observation loss and inverse-dynamics residual reuse the same recorded flight as a composite likelihood",
            "Q has no parameter-uncertainty or sensor-noise covariance correction",
            "angular acceleration uses the deterministic baseline Savitzky-Golay gyro derivative",
            "wrench residual is evaluated at left knots with fixed actuator and delay models",
            "only the deterministic baseline thirteen-dimensional parameter subspace is optimized",
        ],
        "outputs": {
            "result_json": "result.json",
            "trajectory_pdf": "trajectory.pdf",
            "method_comparison_pdf": "method_comparison.pdf",
        },
    }
    baseline._write_json(output / "result.json", payload)
    baseline._write_pdf(
        output / "trajectory.pdf",
        problem,
        nominal_simulation,
        final_simulation,
        metrics["nominal"],
        metrics["deterministic_q"],
    )
    _write_comparison_pdf(
        output / "method_comparison.pdf",
        problem,
        nominal_simulation,
        deterministic_simulation,
        final_simulation,
    )
    print(
        "deterministic-Q metrics: {}".format(
            json.dumps(metrics["deterministic_q"], sort_keys=True)
        )
    )
    print("wrote {}".format(output / "result.json"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
