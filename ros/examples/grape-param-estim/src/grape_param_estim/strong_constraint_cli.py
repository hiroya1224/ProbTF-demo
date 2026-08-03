"""Command-line runner for strong-constraint IEnKS experiments A and B."""

import argparse
import json

from grape_param_estim.strong_constraint_experiments import (
    run_strong_constraint_experiment,
    save_strong_constraint_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-window strong-constraint Grape IEnKS Experiment A or B."
        )
    )
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument(
        "--output", default="grape_strong_constraint_experiment.npz"
    )
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument("--ensemble-size", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    arguments = parser.parse_args()
    result = run_strong_constraint_experiment(
        label=arguments.experiment,
        duration=arguments.duration,
        time_step=arguments.time_step,
        ensemble_size=arguments.ensemble_size,
        maximum_iterations=arguments.iterations,
        seed=arguments.seed,
    )
    destination = save_strong_constraint_experiment(
        arguments.output, result
    )
    metrics = result.metrics
    print(
        json.dumps(
            {
                "schema": (
                    "grape-param-estim/strong-constraint-experiment-summary/v1"
                ),
                "experiment": result.label,
                "output": str(destination),
                "members": int(result.posterior.control_ensemble.shape[0]),
                "iterations": len(result.posterior.iterations),
                "converged": result.posterior.converged,
                "termination_reason": result.posterior.termination_reason,
                "prior_pose_rmse_m": metrics.prior_pose_rmse,
                "posterior_pose_rmse_m": metrics.posterior_pose_rmse,
                "prior_velocity_rmse_mps": metrics.prior_velocity_rmse,
                "posterior_velocity_rmse_mps": (
                    metrics.posterior_velocity_rmse
                ),
                "ridge_variance_ratio": metrics.ridge_variance_ratio,
                "truth_equivalence_mahalanobis": (
                    metrics.truth_equivalence_mahalanobis
                ),
                "truth_pose_component_coverage": (
                    metrics.truth_pose_component_coverage
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
