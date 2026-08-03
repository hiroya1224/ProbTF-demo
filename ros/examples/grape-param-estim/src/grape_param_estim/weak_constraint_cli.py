"""Command-line runner for weak-constraint IEnKS-Q Experiment C."""

import argparse
import json

from grape_param_estim.weak_constraint_experiments import (
    run_weak_constraint_experiment,
    save_weak_constraint_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strong constraint and full-block IEnKS-Q on "
            "model-error Experiment C."
        )
    )
    parser.add_argument(
        "--output", default="grape_weak_constraint_experiment.npz"
    )
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=0,
        help="0 selects augmented control dimension + 2",
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    arguments = parser.parse_args()
    result = run_weak_constraint_experiment(
        duration=arguments.duration,
        time_step=arguments.time_step,
        ensemble_size=(
            None if arguments.ensemble_size == 0 else arguments.ensemble_size
        ),
        maximum_iterations=arguments.iterations,
        seed=arguments.seed,
    )
    destination = save_weak_constraint_experiment(arguments.output, result)
    metrics = result.metrics
    print(
        json.dumps(
            {
                "schema": (
                    "grape-param-estim/weak-constraint-experiment-summary/v1"
                ),
                "output": str(destination),
                "members": int(
                    result.weak_posterior.control_ensemble.shape[0]
                ),
                "augmented_control_dimension": int(
                    result.weak_posterior.control_ensemble.shape[1]
                ),
                "matched_strong_static_bias": (
                    metrics.matched_strong_static_bias
                ),
                "strong_static_bias": metrics.strong_static_bias,
                "weak_static_bias": metrics.weak_static_bias,
                "strong_path_coverage": metrics.strong_path_coverage,
                "weak_path_coverage": metrics.weak_path_coverage,
                "residual_acceleration_r_squared": (
                    metrics.residual_acceleration_r_squared
                ),
                "residual_excited_channel_correlation": (
                    metrics.residual_excited_channel_correlation
                ),
                "weak_converged": result.weak_posterior.converged,
                "weak_termination_reason": (
                    result.weak_posterior.termination_reason
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
