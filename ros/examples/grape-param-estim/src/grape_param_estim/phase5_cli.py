"""Command-line entry point for one Phase-5 real rosbag assimilation."""

import argparse
import json

from grape_param_estim.real_assimilation import (
    run_real_rosbag_assimilation,
    save_real_assimilation,
)
from grape_param_estim.real_rosbag import DEFAULT_GRAPE_BAG


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assimilate one continuous airborne Grape rosbag episode with "
            "sparse weak-constraint IEnKS-Q."
        )
    )
    parser.add_argument("--bag", default=DEFAULT_GRAPE_BAG)
    parser.add_argument("--output", default="grape_phase5_real.npz")
    parser.add_argument("--sample-period", type=float, default=0.04)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-local", type=float)
    parser.add_argument("--end-local", type=float)
    parser.add_argument(
        "--window-state",
        type=int,
        default=5,
        help=(
            "flight_state interval to use (default 5: continuous airborne "
            "hover); use --full-flight only for a model with ground contact"
        ),
    )
    parser.add_argument(
        "--full-flight",
        action="store_true",
        help="explicitly select TAKEOFF-to-STOP instead of one state interval",
    )
    parser.add_argument(
        "--maximum-knots",
        type=int,
        default=12,
        help="0 uses every OU bridge-resolution knot",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=0,
        help="0 selects augmented dimension + 2",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--skip-sha256", action="store_true")
    arguments = parser.parse_args()
    if (arguments.start_local is None) != (arguments.end_local is None):
        parser.error("--start-local and --end-local must be given together")
    if arguments.maximum_knots == 1 or arguments.maximum_knots < 0:
        parser.error("--maximum-knots must be 0 or at least 2")
    result = run_real_rosbag_assimilation(
        bag_path=arguments.bag,
        sample_period=arguments.sample_period,
        episode_index=arguments.episode_index,
        start_local=arguments.start_local,
        end_local=arguments.end_local,
        window_state=(None if arguments.full_flight else arguments.window_state),
        maximum_knots=(
            None if arguments.maximum_knots == 0 else arguments.maximum_knots
        ),
        ensemble_size=(
            None if arguments.ensemble_size == 0 else arguments.ensemble_size
        ),
        maximum_iterations=arguments.iterations,
        seed=arguments.seed,
        compute_sha256=not arguments.skip_sha256,
    )
    destination = save_real_assimilation(arguments.output, result)
    print(
        json.dumps(
            {
                "schema": "grape-weak-constraint/phase5-summary",
                "output": str(destination),
                "bag": result.episode.provenance.bag_path,
                "window_local_seconds": [
                    result.episode.window_start_local_time,
                    result.episode.window_end_local_time,
                ],
                "samples": int(result.episode.observations.times.size),
                "members": int(result.posterior.control_ensemble.shape[0]),
                "augmented_control_dimension": int(
                    result.posterior.control_ensemble.shape[1]
                ),
                "selected_knots": int(result.wrench_process.knot_indices.size),
                "required_knots": int(
                    result.knot_resolution.required_knot_count
                ),
                "q_resolution_sufficient": bool(
                    result.knot_resolution.resolution_sufficient
                ),
                "q_calibration_method": result.calibration.method,
                "q_correlation_time_seconds": (
                    result.calibration.correlation_time
                ),
                "nominal_position_rmse_m": (
                    result.metrics.nominal_position_rmse
                ),
                "posterior_position_rmse_m": (
                    result.metrics.posterior_center_position_rmse
                ),
                "nominal_rotation_rmse_rad": (
                    result.metrics.nominal_rotation_rmse
                ),
                "posterior_rotation_rmse_rad": (
                    result.metrics.posterior_center_rotation_rmse
                ),
                "pose_component_coverage": (
                    result.metrics.observed_pose_component_coverage
                ),
                "ridge_variance_ratio": (
                    result.metrics.posterior_expected_ridge_variance_ratio
                ),
                "selected_mode": (
                    result.mode_diagnostic.selected_mode_id
                ),
                "converged": result.posterior.converged,
                "termination_reason": result.posterior.termination_reason,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
