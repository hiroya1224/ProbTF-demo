"""CLI for the strict sparse-batch synthetic truth artifact."""

import argparse
import json
from typing import Optional, Sequence

import numpy as np

from grape_param_estim.synthetic import (
    SYNTHETIC_BATCH_TRUTH_SCHEMA,
    SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA,
    generate_perfect_model_batch_trajectory,
    load_synthetic_batch_truth_artifact,
    save_synthetic_batch_truth_artifact,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a variable-step, perfect-model sparse-batch trajectory "
            "from the production analytic dynamics factor and save its strict "
            "pickle-free solver-truth artifact."
        )
    )
    parser.add_argument(
        "--output",
        default="grape_synthetic_batch_truth.npz",
        help="destination NPZ (default: %(default)s)",
    )
    parser.add_argument(
        "--interval-count",
        type=int,
        default=36,
        help="number of variable-step dynamics intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=917,
        help="deterministic excitation seed (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)

    trajectory = generate_perfect_model_batch_trajectory(
        interval_count=arguments.interval_count,
        seed=arguments.seed,
    )
    destination = save_synthetic_batch_truth_artifact(
        arguments.output,
        trajectory,
        generator_seed=arguments.seed,
    )
    artifact = load_synthetic_batch_truth_artifact(str(destination))
    time_step = artifact.trajectory.time_step
    print(
        json.dumps(
            {
                "schema": SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA,
                "artifact_schema": SYNTHETIC_BATCH_TRUTH_SCHEMA,
                "output": str(destination),
                "samples": int(artifact.trajectory.times.size),
                "intervals": int(artifact.trajectory.interval_count),
                "duration_s": float(
                    artifact.trajectory.times[-1]
                    - artifact.trajectory.times[0]
                ),
                "minimum_time_step_s": float(np.min(time_step)),
                "maximum_time_step_s": float(np.max(time_step)),
                "truth_parameter_dimension": int(
                    artifact.trajectory.truth_parameter_coordinates.size
                ),
                "direct_truth_channels": [
                    "position",
                    "rotation_so3",
                    "linear_velocity",
                    "angular_velocity",
                    "actuator_thrust",
                    "gimbal_angle",
                ],
                "payload_sha256": artifact.payload_sha256,
                "perfect_model": True,
                "production_factor_analytic_jacobian": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
