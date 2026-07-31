"""Command-line entry points for headless synthetic experiments."""

import argparse
import json

import numpy as np

from grape_param_estim.synthetic import (
    run_perfect_model_experiment,
    run_synthetic_experiment,
    save_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one continuous full-6-DoF Grape closed-loop synthetic "
            "episode and save pose-only observations plus latent truth."
        )
    )
    parser.add_argument(
        "--output",
        default="grape_phase1_synthetic.npz",
        help="destination NPZ (default: %(default)s)",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--perfect-model",
        action="store_true",
        help="disable parameter/model/actuator mismatch and pose noise",
    )
    arguments = parser.parse_args()
    if arguments.perfect_model:
        experiment = run_perfect_model_experiment(
            arguments.duration, arguments.time_step
        )
    else:
        experiment = run_synthetic_experiment(
            duration=arguments.duration,
            time_step=arguments.time_step,
            seed=arguments.seed,
        )
    destination = save_experiment(arguments.output, experiment)
    translation = np.linalg.norm(
        experiment.correction_translation, axis=1
    )
    rotation = np.linalg.norm(
        experiment.correction_rotation_vector, axis=1
    )
    print(
        json.dumps(
            {
                "schema": "grape-weak-constraint/phase1-summary",
                "output": str(destination),
                "samples": int(experiment.nominal.times.size),
                "duration_s": float(
                    experiment.nominal.times[-1]
                    - experiment.nominal.times[0]
                ),
                "maximum_correction_translation_m": float(
                    np.max(translation)
                ),
                "maximum_correction_rotation_deg": float(
                    np.rad2deg(np.max(rotation))
                ),
                "perfect_model": bool(arguments.perfect_model),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
