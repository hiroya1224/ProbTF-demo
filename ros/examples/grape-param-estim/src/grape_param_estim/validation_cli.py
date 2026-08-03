"""Command-line entry points for ridge, size and mode validation."""

import argparse
import json

import numpy as np

from grape_param_estim.ensemble_convergence import (
    run_ensemble_size_convergence,
    save_ensemble_convergence,
)
from grape_param_estim.mode_validation import (
    ActuatorWiringMeasurement,
    NOMINAL_MODE_ID,
    SWAPPED_MODE_ID,
    condition_on_actuator_wiring,
    plant_wiring_mode,
    run_mode_validation_experiment,
    save_mode_validation,
)
from grape_param_estim.ridge_validation import (
    save_ridge_validation,
    validate_strong_constraint_ridge,
    validate_weak_zero_realization_ridge,
)
from grape_param_estim.strong_constraint_experiments import (
    run_strong_constraint_experiment,
)


def _ridge(arguments):
    duration = 0.4 if arguments.duration is None else arguments.duration
    size = 38 if arguments.ensemble_size is None else arguments.ensemble_size
    seed = 11 if arguments.seed is None else arguments.seed
    reports = []
    experiment_a = None
    for label in ("A", "B"):
        result = run_strong_constraint_experiment(
            label=label,
            duration=duration,
            time_step=arguments.time_step,
            ensemble_size=size,
            maximum_iterations=arguments.iterations,
            seed=seed,
        )
        if label == "A":
            experiment_a = result
        reports.append(validate_strong_constraint_ridge(result))
    if experiment_a is None:
        raise AssertionError("Experiment A was not evaluated")
    weak_report = validate_weak_zero_realization_ridge(
        experiment_a,
        maximum_iterations=arguments.iterations,
        seed=seed,
    )
    destination = save_ridge_validation(
        arguments.output, reports, weak_report
    )
    return {
        "schema": "grape-param-estim/ridge-validation-summary/v1",
        "output": str(destination),
        "experiments": {
            value.experiment_label: {
                "lambda_variance_ratio": (
                    value.posterior_lambda_variance_ratio
                ),
                "quotient_truth_mahalanobis": (
                    value.quotient_truth_mahalanobis
                ),
                "information_leak": (
                    value.prior_whitened_information_leak
                ),
                "path_component_coverage": value.path_component_coverage,
            }
            for value in reports
        },
        "weak_zero_realization": {
            "lambda_variance_ratio": (
                weak_report.posterior_lambda_variance_ratio
            ),
            "information_leak": (
                weak_report.prior_whitened_information_leak
            ),
            "augmented_pose_residual_max_error": float(
                np.max(weak_report.augmented_pose_residual_max_error)
            ),
            "particle_correction_required": (
                weak_report.particle_correction_required
            ),
        },
    }


def _convergence(arguments):
    duration = 0.28 if arguments.duration is None else arguments.duration
    seed = 43 if arguments.seed is None else arguments.seed
    report = run_ensemble_size_convergence(
        duration=duration,
        time_step=arguments.time_step,
        maximum_iterations=arguments.iterations,
        seed=seed,
    )
    destination = save_ensemble_convergence(arguments.output, report)
    strong = report.strong_endpoint_comparison
    weak = report.weak_endpoint_comparison
    return {
        "schema": (
            "grape-param-estim/ensemble-convergence-summary/v1"
        ),
        "output": str(destination),
        "strong_sizes": [value.ensemble_size for value in report.strong_laws],
        "weak_sizes": [value.ensemble_size for value in report.weak_laws],
        "strong_identifiable_sliced_w1": strong.identifiable_sliced_w1,
        "strong_path_sliced_w1": strong.path_sliced_w1,
        "weak_identifiable_sliced_w1": weak.identifiable_sliced_w1,
        "weak_path_sliced_w1": weak.path_sliced_w1,
        "weak_strong_conclusion_stable": (
            report.weak_strong_conclusion_stable
        ),
    }


def _mode(arguments):
    duration = 0.3 if arguments.duration is None else arguments.duration
    size = 38 if arguments.ensemble_size is None else arguments.ensemble_size
    seed = 19 if arguments.seed is None else arguments.seed
    result = run_mode_validation_experiment(
        truth_mode_id=arguments.truth_mode,
        duration=duration,
        time_step=arguments.time_step,
        ensemble_size=size,
        maximum_iterations=arguments.iterations,
        seed=seed,
    )
    truth_wiring = plant_wiring_mode(
        arguments.truth_mode
    ).channel_to_rotor
    conditioning = condition_on_actuator_wiring(
        result,
        ActuatorWiringMeasurement(
            channel_to_rotor=np.asarray(truth_wiring, dtype=np.int64),
            correctness_probability=arguments.measurement_correctness,
        ),
    )
    destination = save_mode_validation(
        arguments.output, result, conditioning
    )
    return {
        "schema": "grape-param-estim/mode-validation-summary/v1",
        "output": str(destination),
        "mode_ids": list(result.mode_ids),
        "pose_mode_probabilities": result.pose_mode_probabilities.tolist(),
        "conditioned_mode_probabilities": (
            conditioning.conditioned_mode_probabilities.tolist()
        ),
        "selected_mode_id": conditioning.selected_mode_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one validation without mixing posterior members."
        )
    )
    parser.add_argument(
        "--section",
        choices=("ridge", "convergence", "mode"),
        default="ridge",
    )
    parser.add_argument(
        "--output", default="grape_assimilation_validation.npz"
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--truth-mode",
        choices=(NOMINAL_MODE_ID, SWAPPED_MODE_ID),
        default=SWAPPED_MODE_ID,
    )
    parser.add_argument(
        "--measurement-correctness", type=float, default=0.995
    )
    arguments = parser.parse_args()
    if (
        arguments.section == "convergence"
        and arguments.ensemble_size is not None
    ):
        parser.error(
            "--ensemble-size is not valid for convergence; the section "
            "runs its declared two-size sweeps"
        )
    runners = {
        "ridge": _ridge,
        "convergence": _convergence,
        "mode": _mode,
    }
    print(
        json.dumps(
            runners[arguments.section](arguments),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
