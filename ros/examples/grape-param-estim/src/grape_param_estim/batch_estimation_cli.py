"""One-command worker for a strict sparse batch estimation request."""

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from grape_param_estim.artifact_io import request_fingerprint
from grape_param_estim.batch_artifact import (
    BatchEstimationRun,
    write_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    DelayLocalGeometry,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_performance import measure_run_performance
from grape_param_estim.batch_request import (
    BatchEstimationRequest,
    load_batch_estimation_request,
)
from grape_param_estim.progress import CancellationToken
from grape_param_estim.real_estimation import (
    prepare_real_estimation_inputs,
    run_real_estimation,
)


StageProgress = Callable[[str, int, int, str], None]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "cannot fingerprint value of type {}".format(type(value).__name__)
    )


def configuration_fingerprint(request: BatchEstimationRequest) -> str:
    """Fingerprint scientific settings separately from data file identity."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    payload = request.payload
    configuration = {
        key: _jsonable(payload[key])
        for key in (
            "run_mode",
            "q",
            "parameter_prior",
            "delay",
            "knot_policy",
            "interpolation_policy",
            "controller_snapshot_policy",
            "mode_hypotheses",
            "solver_settings",
            "em_settings",
            "mcmc_settings",
        )
    }
    configuration["bags"] = [
        {
            "bag_id": value["bag_id"],
            "interval_seconds": _jsonable(value["interval_seconds"]),
            "observation_factors": _jsonable(value["observation_factors"]),
            "fixed_factor_covariances": _jsonable(
                value["fixed_factor_covariances"]
            ),
            "initial_state_prior_covariances": _jsonable(
                value["initial_state_prior_covariances"]
            ),
        }
        for value in payload["bags"]
    ]
    return request_fingerprint(configuration)


def controller_snapshot_fingerprint(inputs: object) -> str:
    """Fingerprint the decoded per-bag controller configurations in order."""

    flights = getattr(inputs, "flight_data", None)
    if type(flights) is not tuple or not flights:
        raise TypeError("inputs must expose non-empty flight_data")
    return request_fingerprint(
        {
            "controller_configuration_by_bag": [
                {
                    "bag_id": value.bag_id,
                    "configuration": _jsonable(
                        value.controller_configuration
                    ),
                }
                for value in flights
            ]
        }
    )


def discover_estimator_revision() -> str:
    """Return an explicit environment revision or the enclosing Git commit."""

    supplied = os.environ.get("GRAPE_PARAM_ESTIM_REVISION")
    if supplied is not None:
        selected = supplied.strip()
        if not selected:
            raise ValueError("GRAPE_PARAM_ESTIM_REVISION cannot be blank")
        return selected
    source = Path(__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise RuntimeError(
            "estimator revision is unavailable; set GRAPE_PARAM_ESTIM_REVISION"
        )
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        revision = completed.stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain", "--", "ros/examples/grape-param-estim"),
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot determine estimator Git revision") from error
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("Git returned a non-canonical estimator revision")
    return revision + ("-dirty" if dirty else "")


def _warnings(result: object, estimator_revision: str) -> tuple:
    warnings = []
    if estimator_revision.endswith("-dirty"):
        warnings.append("estimator source tree had uncommitted package changes")
    for mode in result.modes:
        if not mode.em.converged:
            warnings.append(
                "mode {} Laplace-EM terminated with {}".format(
                    mode.mode_id, mode.em.reason.value
                )
            )
    selected = result.selected_mode
    if selected.delay_uncertainty.curvature is None:
        warnings.append(selected.delay_uncertainty.source)
    if result.mcmc is not None and not result.mcmc.diagnostics.converged:
        warnings.append("MCMC completed without satisfying convergence thresholds")
    return tuple(warnings)


def execute_batch_estimation(
    request: BatchEstimationRequest,
    *,
    estimator_revision: str,
    cancellation_token: Optional[CancellationToken] = None,
    progress: Optional[StageProgress] = None,
) -> BatchEstimationRun:
    """Execute, export, and atomically publish one validated request."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    if not isinstance(estimator_revision, str) or not estimator_revision.strip():
        raise ValueError("estimator_revision must be non-empty")
    if request.payload["resume"]:
        raise ValueError(
            "resume requires an existing sparse-run checkpoint and is not a fresh run"
        )
    cancellation = (
        CancellationToken() if cancellation_token is None else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    cancellation.raise_if_cancelled()
    inputs = prepare_real_estimation_inputs(
        request,
        cancellation_requested=lambda: cancellation.cancelled,
        progress=progress,
    )
    cancellation.raise_if_cancelled()
    result = run_real_estimation(
        inputs,
        cancellation_requested=lambda: cancellation.cancelled,
        progress=progress,
    )
    cancellation.raise_if_cancelled()
    if progress is not None:
        progress(
            "computing_local_posterior_geometry",
            0,
            1,
            "benchmarking final undamped sparse geometry",
        )
    performance = measure_run_performance(result)
    selected = result.selected_mode
    identity = ArtifactRunIdentity(
        estimator_revision=estimator_revision.strip(),
        configuration_fingerprint=configuration_fingerprint(request),
        controller_snapshot_fingerprint=controller_snapshot_fingerprint(inputs),
        warnings=_warnings(result, estimator_revision),
    )
    payload = export_batch_estimation_artifact_payload(
        request=request,
        flight_data=inputs.flight_data,
        initializations=inputs.initializations,
        final_solution=selected.final_solution,
        em_result=selected.em,
        static_geometry=selected.static_geometry,
        final_q_lag_profile=(
            selected.final_q_lag_profile_history[-1]
            if selected.final_q_lag_profile_history
            else None
        ),
        delay_geometry=DelayLocalGeometry(
            selected.delay_uncertainty.standard_deviation_seconds,
            selected.delay_uncertainty.source,
            selected.delay_uncertainty.curvature,
        ),
        identity=identity,
        performance=performance,
        mcmc_chains=(
            () if result.mcmc is None else result.mcmc.chains
        ),
        mcmc_diagnostics=(
            None if result.mcmc is None else result.mcmc.diagnostics
        ),
    )
    if progress is not None:
        progress(
            "computing_local_posterior_geometry",
            1,
            1,
            "strict solver payload validated",
        )
        progress("writing_artifacts", 0, 1, "publishing strict run")
    cancellation.raise_if_cancelled()
    written = write_batch_estimation_run(
        request.output_directory, **payload.writer_arguments
    )
    if progress is not None:
        progress("writing_artifacts", 1, 1, "run complete")
    return written


def run_request(
    request_path: str,
    *,
    estimator_revision: Optional[str] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress: Optional[StageProgress] = None,
) -> BatchEstimationRun:
    request = load_batch_estimation_request(request_path)
    return execute_batch_estimation(
        request,
        estimator_revision=(
            discover_estimator_revision()
            if estimator_revision is None
            else estimator_revision
        ),
        cancellation_token=cancellation_token,
        progress=progress,
    )


def _signal_reason(signum: int) -> str:
    try:
        return "signal_{}".format(signal.Signals(signum).name)
    except ValueError:
        return "signal_{}".format(signum)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Grape full trajectories with sparse MAP, Laplace-EM, "
            "ridge analysis, and optional MCMC."
        )
    )
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args(argv)
    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel(_signal_reason(signum))

    previous = {
        signum: signal.signal(signum, request_cancel)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        output = run_request(
            arguments.request, cancellation_token=cancellation
        )
        print("batch estimation complete: {}".format(output.root), file=sys.stderr)
        return 0
    except Exception as error:  # pylint: disable=broad-except
        print("batch estimation failed: {}".format(error), file=sys.stderr)
        return 2 if cancellation.cancelled else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StageProgress",
    "configuration_fingerprint",
    "controller_snapshot_fingerprint",
    "discover_estimator_revision",
    "execute_batch_estimation",
    "main",
    "run_request",
]
