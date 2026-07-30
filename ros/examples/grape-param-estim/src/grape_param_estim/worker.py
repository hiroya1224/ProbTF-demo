"""One-shot Phase-2 worker process and run-directory persistence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Any, Mapping

import numpy as np

from grape_param_estim.data import load_yaml, read_bag
from grape_param_estim.estimator import (
    LikelihoodWeights,
    estimate_parameters,
    parameter_summary,
    save_result,
)
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    RigidBodyParameters,
)


class EstimationCancelled(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, content: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _nominal_parameters(configuration: Mapping[str, Any]):
    nominal = configuration["model"]["nominal"]
    return RigidBodyParameters.from_diagonal(
        nominal["mass"], nominal["inertia_diagonal"]
    )


def _summary(result, elapsed_seconds: float) -> dict:
    bags = []
    for bag in result.bags:
        bags.append(
            {
                "bag_path": bag.bag_path,
                "start_time": float(bag.analysis.start_time),
                "end_time": float(bag.analysis.end_time),
                "samples": int(bag.analysis.times.size),
                "segments": int(bag.analysis.segment_count),
                "nominal_translation_rms": float(
                    np.sqrt(
                        np.mean(
                            bag.nominal.translation_residual_norm**2
                        )
                    )
                ),
                "nominal_rotation_rms_deg": float(
                    np.rad2deg(
                        np.sqrt(
                            np.mean(
                                bag.nominal.rotation_residual_norm**2
                            )
                        )
                    )
                ),
            }
        )
    return {
        "schema": "grape_param_estim/phase2-summary",
        "completed_at": _utc_now(),
        "elapsed_seconds": float(elapsed_seconds),
        "particle_count": int(result.posterior.particles.shape[0]),
        "effective_sample_size": result.posterior.effective_sample_size,
        "resampled": bool(result.posterior.resampled),
        "seed": int(result.seed),
        "parameters": parameter_summary(result.posterior),
        "bags": bags,
    }


def run_worker(run_directory: str) -> None:
    run_path = Path(run_directory).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(str(run_path))
    config_path = run_path / "config.yaml"
    status_path = run_path / "status.json"
    result_path = run_path / "result.npz"
    summary_path = run_path / "summary.json"
    error_path = run_path / "error.txt"
    configuration = load_yaml(str(config_path))
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    cancelled = {"requested": False}

    def handle_signal(_signal_number, _frame) -> None:
        cancelled["requested"] = True
        raise EstimationCancelled("stop requested")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    error_path.write_text("", encoding="utf-8")
    base_status = {
        "schema": "grape_param_estim/phase2-status",
        "pid": os.getpid(),
        "started_at": started_at,
        "run_directory": str(run_path),
    }
    last_status_write = {"time": 0.0, "progress": 0.0}

    def write_status(
        state: str,
        progress: float,
        stage: str,
        message: str,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and now - last_status_write["time"] < 0.20
            and progress < 1.0
        ):
            return
        elapsed = now - started_monotonic
        bounded = float(np.clip(progress, 0.0, 1.0))
        eta = (
            elapsed * (1.0 - bounded) / bounded
            if 0.0 < bounded < 1.0
            else 0.0
        )
        status = dict(base_status)
        status.update(
            {
                "state": state,
                "progress": bounded,
                "stage": stage,
                "message": message,
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta),
                "updated_at": _utc_now(),
            }
        )
        _atomic_json(status_path, status)
        last_status_write["time"] = now
        last_status_write["progress"] = bounded

    try:
        write_status(
            "running", 0.0, "loading data", "reading configuration", True
        )
        topics = configuration["data"]["topics"]
        datasets = configuration["analysis"]["datasets"]
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("analysis.datasets must contain at least one bag")
        analyses = []
        for index, dataset in enumerate(datasets):
            if cancelled["requested"]:
                raise EstimationCancelled("stop requested")
            write_status(
                "running",
                0.08 * index / len(datasets),
                "loading data",
                "reading {}".format(Path(dataset["bag_path"]).name),
                True,
            )
            recording = read_bag(dataset["bag_path"], topics)
            analyses.append(
                recording.select_interval(
                    dataset["start_time"],
                    dataset["end_time"],
                    dataset["segment_duration"],
                )
            )

        estimator_config = configuration["estimator"]
        likelihood_config = estimator_config["likelihood"]
        nominal_parameters = _nominal_parameters(configuration)

        def estimator_progress(
            fraction: float, stage: str, message: str
        ) -> None:
            if cancelled["requested"]:
                raise EstimationCancelled("stop requested")
            write_status(
                "running",
                0.08 + 0.86 * fraction,
                stage,
                message,
            )

        result = estimate_parameters(
            analyses=tuple(analyses),
            nominal_parameters=nominal_parameters,
            prior_bounds={
                name: (
                    estimator_config["priors"][name]["min"],
                    estimator_config["priors"][name]["max"],
                )
                for name in (
                    "mass_scale",
                    "force_scale",
                    "inertia_scale",
                    "torque_scale",
                )
            },
            particle_count=estimator_config["particle_count"],
            likelihood_weights=LikelihoodWeights(
                translation=likelihood_config["translation_weight"],
                rotation=likelihood_config["rotation_weight"],
            ),
            seed=estimator_config.get("seed", 42),
            resample_ess_fraction=estimator_config.get(
                "resample_ess_fraction", 0.10
            ),
            jitter_fraction=estimator_config.get(
                "jitter_fraction", 0.03
            ),
            model=GrapeRigidBodyModel(
                maximum_time_step=estimator_config.get(
                    "maximum_time_step", 0.005
                )
            ),
            progress_callback=estimator_progress,
        )
        write_status(
            "running", 0.95, "saving", "writing result.npz", True
        )
        save_result(str(result_path), result)
        summary = _summary(result, time.monotonic() - started_monotonic)
        _atomic_json(summary_path, summary)
        write_status(
            "completed",
            1.0,
            "complete",
            "posterior and uncertain transforms saved",
            True,
        )
    except EstimationCancelled as error:
        write_status(
            "stopped",
            last_status_write["progress"],
            "stopped",
            str(error),
            True,
        )
    except Exception as error:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_status(
            "failed",
            last_status_write["progress"],
            "failed",
            "{}: {}".format(type(error).__name__, error),
            True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    arguments = parser.parse_args()
    run_worker(arguments.run_dir)


__all__ = ["EstimationCancelled", "main", "run_worker"]


if __name__ == "__main__":
    main()
