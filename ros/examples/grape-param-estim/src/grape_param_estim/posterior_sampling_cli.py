"""Independent resumable MCMC substage for a completed estimate-only run."""

import argparse
from dataclasses import replace
from enum import Enum
from pathlib import Path
import signal
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from grape_param_estim.artifact_io import ArtifactValidationError
from grape_param_estim.batch_artifact import (
    BatchEstimationRun,
    load_batch_estimation_run,
    replace_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    BatchArtifactPayload,
    append_posterior_sampling_artifact_payload,
)
from grape_param_estim.batch_checkpoint import (
    begin_posterior_sampling_checkpoint,
    load_batch_estimation_checkpoint,
    mark_batch_checkpoint_cancelled,
    mark_batch_checkpoint_published,
    save_batch_chain_checkpoint,
)
from grape_param_estim.batch_estimation_cli import (
    BatchProgressReporter,
    StageProgress,
    configuration_fingerprint,
    controller_snapshot_fingerprint,
    discover_estimator_revision,
)
from grape_param_estim.batch_request import (
    BatchEstimationRequest,
    load_batch_estimation_request,
    validate_batch_estimation_request,
)
from grape_param_estim.posterior_sampling_request import (
    PosteriorSamplingRequest,
    load_posterior_sampling_request,
)
from grape_param_estim.progress import CancellationToken, JsonlProgressWriter
from grape_param_estim.real_estimation import (
    prepare_real_estimation_inputs,
    restore_laplace_checkpoint,
    sample_laplace_solution,
)
from grape_param_estim.trajectory_sampling import (
    sample_selected_conditional_trajectories,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def posterior_batch_request(
    sampling: PosteriorSamplingRequest,
    estimation: BatchEstimationRequest,
) -> BatchEstimationRequest:
    """Build the explicit in-memory MCMC request without changing upstream."""

    if not isinstance(sampling, PosteriorSamplingRequest):
        raise TypeError("sampling must be PosteriorSamplingRequest")
    if not isinstance(estimation, BatchEstimationRequest):
        raise TypeError("estimation must be BatchEstimationRequest")
    payload = _plain(estimation.payload)
    payload["run_mode"] = "estimate_and_sample"
    payload["resume"] = False
    payload["mcmc_settings"] = {
        "enabled": True,
        **_plain(sampling.payload["mcmc_settings"]),
    }
    return validate_batch_estimation_request(
        payload, source_path=sampling.source_path
    )


def _request_upstream_identity(
    estimation: BatchEstimationRequest,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    bags = tuple(estimation.payload["bags"])
    return {
        "run_id": str(estimation.payload["run_id"]),
        "request_fingerprint": estimation.fingerprint,
        "configuration_fingerprint": configuration_fingerprint(estimation),
        "controller_snapshot_fingerprint": manifest[
            "controller_snapshot_fingerprint"
        ],
        "estimator_revision": manifest["estimator_revision"],
        "selected_bag_ids": [str(value["bag_id"]) for value in bags],
        "selected_intervals": {
            str(value["bag_id"]): list(value["interval_seconds"])
            for value in bags
        },
        "selected_bag_sha256": {
            str(value["bag_id"]): str(value["sha256"]) for value in bags
        },
    }


def _core_from_run(run: BatchEstimationRun) -> BatchArtifactPayload:
    metadata = {
        key: value
        for key, value in run.manifest.items()
        if key not in {"schema", "status", "artifacts"}
    }
    return BatchArtifactPayload(
        manifest_metadata=metadata,
        map_static=run.map_static,
        q_em=run.q_em,
        laplace=run.laplace,
        diagnostics=run.diagnostics,
        bags=run.bags,
        mcmc_samples=run.mcmc_samples,
        trajectories=run.trajectories,
    )


def _verify_checkpoint_matches_run(checkpoint, run: BatchEstimationRun) -> None:
    def exact_arrays(first, second, location):
        if set(first) != set(second) or any(
            not np.array_equal(first[key], second[key]) for key in first
        ):
            raise ArtifactValidationError(
                "checkpoint {} differs from completed estimation run".format(
                    location
                )
            )

    for name in ("map_static", "q_em", "laplace", "diagnostics"):
        exact_arrays(getattr(checkpoint.core, name), getattr(run, name), name)
    for bag_id in run.manifest["selected_bag_ids"]:
        exact_arrays(
            checkpoint.core.bags[bag_id], run.bags[bag_id], "bag " + bag_id
        )


def execute_posterior_sampling(
    request: PosteriorSamplingRequest,
    *,
    sampler_revision: str,
    cancellation_token: Optional[CancellationToken] = None,
    progress: Optional[StageProgress] = None,
) -> BatchEstimationRun:
    """Append MCMC atomically while leaving the estimate artifact untouched."""

    if not isinstance(request, PosteriorSamplingRequest):
        raise TypeError("request must be PosteriorSamplingRequest")
    if not isinstance(sampler_revision, str) or not sampler_revision.strip():
        raise ValueError("sampler_revision must be non-empty")
    cancellation = (
        CancellationToken() if cancellation_token is None else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    estimation = load_batch_estimation_request(request.estimation_request_path)
    if estimation.output_directory != request.estimation_run_directory:
        raise ArtifactValidationError(
            "sampling must target the exact upstream output directory"
        )
    if (
        estimation.payload["run_mode"] != "estimate_only"
        or bool(estimation.payload["mcmc_settings"]["enabled"])
    ):
        raise ArtifactValidationError(
            "upstream request must be a strict estimate_only request"
        )
    run = load_batch_estimation_run(request.estimation_run_directory)
    expected_upstream = _request_upstream_identity(estimation, run.manifest)
    if _plain(request.payload["upstream"]) != expected_upstream:
        raise ArtifactValidationError("sampling upstream identity mismatch")
    for key, value in expected_upstream.items():
        if key in run.manifest and run.manifest[key] != value:
            raise ArtifactValidationError(
                "completed estimation {} mismatch".format(key)
            )

    if run.mcmc_samples is not None:
        stored = run.manifest["mcmc_settings"].get(
            "sampling_request_fingerprint"
        )
        if stored == request.fingerprint:
            if progress is not None:
                progress(
                    "writing_artifacts",
                    0,
                    1,
                    "posterior samples already complete",
                )
                progress("writing_artifacts", 1, 1, "run complete")
            return run
        raise ArtifactValidationError(
            "estimation run already contains samples from another request"
        )
    if run.manifest["mcmc_settings"] != {"enabled": False}:
        raise ArtifactValidationError("completed run is not estimate-only")

    cancellation.raise_if_cancelled()
    inputs = prepare_real_estimation_inputs(
        estimation,
        cancellation_requested=lambda: cancellation.cancelled,
        progress=progress,
    )
    actual_controller = controller_snapshot_fingerprint(inputs)
    if actual_controller != expected_upstream["controller_snapshot_fingerprint"]:
        raise ArtifactValidationError(
            "decoded controller snapshot changed since estimation"
        )
    checkpoint = load_batch_estimation_checkpoint(
        estimation.output_directory,
        request=estimation,
        estimator_revision=expected_upstream["estimator_revision"],
        configuration_fingerprint=expected_upstream["configuration_fingerprint"],
        controller_snapshot_fingerprint=expected_upstream[
            "controller_snapshot_fingerprint"
        ],
        allow_published=True,
    )
    _verify_checkpoint_matches_run(checkpoint, run)
    context = checkpoint.manifest["sampling_context"]
    if bool(request.payload["resume"]) != (context is not None):
        raise ArtifactValidationError(
            "sampling resume flag disagrees with checkpoint state"
        )
    if context is None and checkpoint.chain_checkpoints:
        raise ArtifactValidationError(
            "fresh sampling checkpoint unexpectedly contains chain state"
        )
    clean_mcmc_settings = {
        "enabled": True,
        **_plain(request.payload["mcmc_settings"]),
    }
    begin_posterior_sampling_checkpoint(
        checkpoint.root,
        sampling_request_fingerprint=request.fingerprint,
        mcmc_settings=clean_mcmc_settings,
        sampler_revision=sampler_revision.strip(),
    )
    derived = posterior_batch_request(request, estimation)
    sampling_inputs = replace(inputs, request=derived)
    final_solution, geometry, delay_static_geometry = restore_laplace_checkpoint(
        sampling_inputs,
        str(checkpoint.manifest["selected_mode_id"]),
        checkpoint.state_values,
        checkpoint.core.map_static,
        checkpoint.core.q_em,
        checkpoint.core.laplace,
    )
    target_timings = []
    try:
        cancellation.raise_if_cancelled()
        mcmc = sample_laplace_solution(
            sampling_inputs,
            str(checkpoint.manifest["selected_mode_id"]),
            final_solution,
            geometry,
            delay_static_geometry,
            cancellation_requested=lambda: cancellation.cancelled,
            progress=progress,
            target_timing_callback=target_timings.append,
            chain_checkpoints=checkpoint.chain_checkpoints,
            checkpoint_chain_proposal=lambda _chain_id, value: (
                save_batch_chain_checkpoint(checkpoint.root, value)
            ),
        )
        cancellation.raise_if_cancelled()
        conditional = sample_selected_conditional_trajectories(
            sampling_inputs,
            str(checkpoint.manifest["selected_mode_id"]),
            final_solution,
            mcmc.chains,
            cancellation_requested=lambda: cancellation.cancelled,
            progress=(
                None
                if progress is None
                else lambda completed, total, message: progress(
                    "writing_artifacts", completed, total + 1, message
                )
            ),
        )
        cancellation.raise_if_cancelled()
    except Exception:
        if cancellation.cancelled:
            mark_batch_checkpoint_cancelled(
                checkpoint.root, cancellation.reason
            )
        raise

    audited_settings = {
        **clean_mcmc_settings,
        "sampling_request_fingerprint": request.fingerprint,
        "upstream_estimation_request_fingerprint": estimation.fingerprint,
        "sampler_revision": sampler_revision.strip(),
    }
    completed = append_posterior_sampling_artifact_payload(
        _core_from_run(run),
        final_solution,
        mcmc.chains,
        mcmc.diagnostics,
        target_timings,
        audited_settings,
        inputs.flight_data,
        inputs.initializations,
        conditional.trajectories,
        conditional.selection.manifest_payload,
    )
    if progress is not None:
        progress(
            "writing_artifacts",
            len(conditional.selection.selected_sample_ids),
            len(conditional.selection.selected_sample_ids) + 1,
            "publishing posterior samples",
        )
    if cancellation.cancelled:
        mark_batch_checkpoint_cancelled(
            checkpoint.root, cancellation.reason
        )
    cancellation.raise_if_cancelled()
    upgraded = replace_batch_estimation_run(
        request.estimation_run_directory,
        expected_request_fingerprint=estimation.fingerprint,
        **completed.writer_arguments
    )
    mark_batch_checkpoint_published(checkpoint.root)
    if progress is not None:
        progress(
            "writing_artifacts",
            len(conditional.selection.selected_sample_ids) + 1,
            len(conditional.selection.selected_sample_ids) + 1,
            "run complete",
        )
    return upgraded


def _signal_reason(signum: int) -> str:
    try:
        return "signal_{}".format(signal.Signals(signum).name)
    except ValueError:
        return "signal_{}".format(signum)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append resumable ridge-aware MCMC to one estimate-only run."
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
        request = load_posterior_sampling_request(arguments.request)
        estimation = load_batch_estimation_request(
            request.estimation_request_path
        )
        progress = BatchProgressReporter(
            posterior_batch_request(request, estimation),
            JsonlProgressWriter(sys.stdout),
        )
        output = execute_posterior_sampling(
            request,
            sampler_revision=discover_estimator_revision(),
            cancellation_token=cancellation,
            progress=progress,
        )
        print(
            "posterior sampling complete: {}".format(output.root),
            file=sys.stderr,
        )
        return 0
    except Exception as error:  # pylint: disable=broad-except
        print("posterior sampling failed: {}".format(error), file=sys.stderr)
        return 2 if cancellation.cancelled else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


__all__ = [
    "execute_posterior_sampling",
    "main",
    "posterior_batch_request",
]
