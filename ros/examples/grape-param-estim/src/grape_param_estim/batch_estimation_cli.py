"""One-command worker for a strict sparse batch estimation request."""

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from grape_param_estim.artifact_io import request_fingerprint
from grape_param_estim.batch_artifact import (
    BatchEstimationRun,
    load_batch_estimation_run,
    write_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    DelayLocalGeometry,
    complete_pending_mcmc_artifact_payload,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_checkpoint import (
    batch_checkpoint_path,
    load_batch_estimation_checkpoint,
    mark_batch_checkpoint_cancelled,
    mark_batch_checkpoint_published,
    save_batch_chain_checkpoint,
    write_batch_estimation_checkpoint,
)
from grape_param_estim.batch_performance import (
    measure_estimation_modes_performance,
    measure_run_performance,
)
from grape_param_estim.batch_request import (
    BatchEstimationRequest,
    load_batch_estimation_request,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCallback,
    ProgressEvent,
    ProgressValidationError,
    STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY,
    STAGE_LABELS,
    STAGE_OPTIMIZING_FULL_TRAJECTORY,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_REFINING_CONSTANT_DELAY,
    STAGE_SAMPLING_PARAMETER_POSTERIOR,
    STAGE_UPDATING_MODEL_ERROR_COVARIANCE,
    STAGE_WRITING_ARTIFACTS,
    stage_label,
)
from grape_param_estim.real_estimation import (
    estimate_real_modes,
    prepare_real_estimation_inputs,
    restore_laplace_checkpoint,
    run_real_estimation,
    sample_laplace_solution,
)


StageProgress = Callable[[str, int, int, str], None]


def planned_progress_units(request: BatchEstimationRequest) -> int:
    """Return the fixed-point resolution of the overall progress wire."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    return 10_000


class BatchProgressReporter:
    """Adapt estimator-local counters to strict monotonic progress events."""

    def __init__(
        self,
        request: BatchEstimationRequest,
        callback: ProgressCallback,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(request, BatchEstimationRequest):
            raise TypeError("request must be BatchEstimationRequest")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._run_id = str(request.payload["run_id"])
        self._total = planned_progress_units(request)
        self._callback = callback
        self._clock = clock
        self._started = float(clock())
        self._last_time = self._started
        self._stage_started = self._started
        self._stage_id: Optional[str] = None
        self._stage_last_completed = 0
        self._last_fraction = 0.0
        self._terminal = False
        payload = request.payload
        self._q_update_total = max(
            1,
            len(payload["mode_hypotheses"])
            * int(payload["em_settings"]["maximum_iterations"]),
        )
        delay = payload["delay"]
        profile_evaluations = int(delay["coarse_grid_points"]) + int(
            delay["maximum_refinement_evaluations"]
        )
        typical_lm_iterations = min(
            int(payload["solver_settings"]["maximum_iterations"]), 5
        )
        # One profile for the current Q and normally one for its proposed Q.
        self._optimization_callbacks_per_q = max(
            1, 2 * profile_evaluations * typical_lm_iterations
        )
        self._q_updates_seen = 0
        self._optimization_callbacks_since_q = 0
        if bool(payload["mcmc_settings"]["enabled"]):
            self._weights = {
                "preparation": 0.05,
                "inference": 0.70,
                "geometry": 0.05,
                "sampling": 0.15,
                "writing": 0.05,
            }
        else:
            self._weights = {
                "preparation": 0.05,
                "inference": 0.85,
                "geometry": 0.05,
                "sampling": 0.0,
                "writing": 0.05,
            }

    @property
    def total_units(self) -> int:
        return self._total

    def __call__(
        self,
        stage_id: str,
        completed_units: int,
        total_units: int,
        message: str,
    ) -> None:
        if self._terminal:
            raise ProgressValidationError(
                "progress was emitted after the terminal artifact event"
            )
        if stage_id not in STAGE_LABELS:
            raise ProgressValidationError(
                "unsupported progress stage_id {!r}".format(stage_id)
            )
        if (
            isinstance(completed_units, bool)
            or isinstance(total_units, bool)
            or not isinstance(completed_units, (int, np.integer))
            or not isinstance(total_units, (int, np.integer))
        ):
            raise ProgressValidationError(
                "stage progress units must be integers"
            )
        local_completed = int(completed_units)
        local_total = int(total_units)
        if local_total < 1 or not 0 <= local_completed <= local_total:
            raise ProgressValidationError(
                "stage progress units are outside their valid range"
            )
        if not isinstance(message, str):
            raise ProgressValidationError("progress message must be text")

        now = float(self._clock())
        if not np.isfinite(now) or now < self._last_time:
            raise ProgressValidationError("monotonic progress clock regressed")
        stage_restarted = (
            self._stage_id != stage_id
            or (
                stage_id == STAGE_OPTIMIZING_FULL_TRAJECTORY
                and local_completed <= self._stage_last_completed
            )
        )
        if stage_restarted:
            self._stage_id = stage_id
            self._stage_started = now
        self._stage_last_completed = local_completed
        self._last_time = now
        elapsed = now - self._started
        stage_elapsed = now - self._stage_started

        terminal = (
            stage_id == STAGE_WRITING_ARTIFACTS
            and local_completed == local_total
            and message == "run complete"
        )
        if terminal:
            overall_completed = self._total
            self._terminal = True
        else:
            candidate = self._phase_fraction(
                stage_id, local_completed, local_total
            )
            self._last_fraction = max(self._last_fraction, candidate)
            overall_completed = min(
                self._total - 1,
                int(np.floor(self._last_fraction * self._total)),
            )

        stage_eta = None
        if local_completed == local_total:
            stage_eta = 0.0
        elif local_completed > 0 and stage_elapsed > 0.0:
            stage_eta = (
                stage_elapsed
                * float(local_total - local_completed)
                / float(local_completed)
            )
        overall_eta = None
        if overall_completed == self._total:
            overall_eta = 0.0
        elif overall_completed >= 2 and elapsed > 0.0:
            overall_eta = (
                elapsed
                * float(self._total - overall_completed)
                / float(overall_completed)
            )
        event = ProgressEvent(
            run_id=self._run_id,
            stage_id=stage_id,
            stage_label=stage_label(stage_id),
            stage_completed_units=local_completed,
            stage_total_units=local_total,
            stage_fraction=float(local_completed) / float(local_total),
            completed_units=overall_completed,
            total_units=self._total,
            fraction=float(overall_completed) / float(self._total),
            stage_elapsed_seconds=stage_elapsed,
            stage_eta_seconds=stage_eta,
            elapsed_seconds=elapsed,
            eta_seconds=overall_eta,
            message=message,
        )
        self._callback(event)

    def _phase_fraction(
        self, stage_id: str, completed: int, total: int
    ) -> float:
        local = float(completed) / float(total)
        preparation = self._weights["preparation"]
        inference = self._weights["inference"]
        geometry = self._weights["geometry"]
        sampling = self._weights["sampling"]
        if stage_id == STAGE_PREPARING_TRAJECTORY:
            return preparation * local
        if stage_id in (
            STAGE_OPTIMIZING_FULL_TRAJECTORY,
            STAGE_REFINING_CONSTANT_DELAY,
        ):
            self._optimization_callbacks_since_q += 1
            provisional = min(
                0.95,
                float(self._optimization_callbacks_since_q)
                / float(self._optimization_callbacks_per_q),
            )
            inference_fraction = min(
                1.0,
                (
                    float(self._q_updates_seen) + provisional
                )
                / float(self._q_update_total),
            )
            return preparation + inference * inference_fraction
        if stage_id == STAGE_UPDATING_MODEL_ERROR_COVARIANCE:
            self._q_updates_seen = min(
                self._q_update_total, self._q_updates_seen + 1
            )
            self._optimization_callbacks_since_q = 0
            return preparation + inference * (
                float(self._q_updates_seen) / float(self._q_update_total)
            )
        inference_end = preparation + inference
        if stage_id == STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY:
            return inference_end + geometry * local
        geometry_end = inference_end + geometry
        if stage_id == STAGE_SAMPLING_PARAMETER_POSTERIOR:
            return geometry_end + sampling * local
        if stage_id == STAGE_WRITING_ARTIFACTS:
            return geometry_end + sampling + self._weights["writing"] * local
        raise ProgressValidationError(
            "unsupported progress stage_id {!r}".format(stage_id)
        )


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
            "actuator_model",
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
    cancellation = (
        CancellationToken() if cancellation_token is None else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    resume = bool(request.payload["resume"])
    if not resume and request.output_directory.exists():
        raise ValueError("fresh run output_directory already exists")

    cancellation.raise_if_cancelled()
    inputs = prepare_real_estimation_inputs(
        request,
        cancellation_requested=lambda: cancellation.cancelled,
        progress=progress,
    )
    cancellation.raise_if_cancelled()
    selected_configuration_fingerprint = configuration_fingerprint(request)
    selected_controller_fingerprint = controller_snapshot_fingerprint(inputs)
    if resume and request.output_directory.exists():
        completed = load_batch_estimation_run(request.output_directory)
        expected = {
            "run_id": str(request.payload["run_id"]),
            "request_fingerprint": request.fingerprint,
            "configuration_fingerprint": selected_configuration_fingerprint,
            "controller_snapshot_fingerprint": selected_controller_fingerprint,
            "estimator_revision": estimator_revision.strip(),
        }
        for key, value in expected.items():
            if completed.manifest[key] != value:
                raise ValueError("completed resume {} mismatch".format(key))
        return completed

    if bool(request.payload["mcmc_settings"]["enabled"]):
        return _execute_resumable_mcmc_run(
            request=request,
            inputs=inputs,
            estimator_revision=estimator_revision.strip(),
            configuration_digest=selected_configuration_fingerprint,
            controller_digest=selected_controller_fingerprint,
            cancellation=cancellation,
            progress=progress,
        )
    if resume:
        checkpoint = load_batch_estimation_checkpoint(
            request.output_directory,
            request=request,
            estimator_revision=estimator_revision.strip(),
            configuration_fingerprint=selected_configuration_fingerprint,
            controller_snapshot_fingerprint=selected_controller_fingerprint,
        )
        if bool(checkpoint.core.manifest_metadata["mcmc_settings"]["enabled"]):
            raise ValueError("estimate-only resume checkpoint enables MCMC")
        if progress is not None:
            progress("writing_artifacts", 0, 1, "publishing strict run")
        written = write_batch_estimation_run(
            request.output_directory, **checkpoint.core.writer_arguments
        )
        mark_batch_checkpoint_published(checkpoint.root)
        if progress is not None:
            progress("writing_artifacts", 1, 1, "run complete")
        return written
    result = run_real_estimation(
        inputs,
        cancellation_requested=lambda: cancellation.cancelled,
        progress=progress,
    )
    cancellation.raise_if_cancelled()
    performance = measure_run_performance(result)
    selected = result.selected_mode
    identity = ArtifactRunIdentity(
        estimator_revision=estimator_revision.strip(),
        configuration_fingerprint=selected_configuration_fingerprint,
        controller_snapshot_fingerprint=selected_controller_fingerprint,
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
        progress("writing_artifacts", 0, 1, "publishing strict run")
    cancellation.raise_if_cancelled()
    checkpoint = write_batch_estimation_checkpoint(
        request.output_directory,
        request=request,
        estimator_revision=estimator_revision.strip(),
        configuration_fingerprint=selected_configuration_fingerprint,
        controller_snapshot_fingerprint=selected_controller_fingerprint,
        selected_mode_id=selected.mode_id,
        core=payload,
        state=selected.final_solution.lm.state,
    )
    written = write_batch_estimation_run(
        request.output_directory, **payload.writer_arguments
    )
    mark_batch_checkpoint_published(checkpoint.root)
    if progress is not None:
        progress("writing_artifacts", 1, 1, "run complete")
    return written


def _execute_resumable_mcmc_run(
    *,
    request: BatchEstimationRequest,
    inputs: object,
    estimator_revision: str,
    configuration_digest: str,
    controller_digest: str,
    cancellation: CancellationToken,
    progress: Optional[StageProgress],
) -> BatchEstimationRun:
    """Reuse a completed inference core and proposal-boundary chain states."""

    checkpoint_root = batch_checkpoint_path(request.output_directory)
    if bool(request.payload["resume"]):
        checkpoint = load_batch_estimation_checkpoint(
            request.output_directory,
            request=request,
            estimator_revision=estimator_revision,
            configuration_fingerprint=configuration_digest,
            controller_snapshot_fingerprint=controller_digest,
        )
        checkpoint_root = checkpoint.root
        core = checkpoint.core
        selected_mode_id = str(checkpoint.manifest["selected_mode_id"])
        final_solution, static_geometry, delay_uncertainty = (
            restore_laplace_checkpoint(
                inputs,
                selected_mode_id,
                checkpoint.state_values,
                core.map_static,
                core.q_em,
                core.laplace,
            )
        )
        chain_checkpoints = checkpoint.chain_checkpoints
    else:
        modes, selected_mode_id = estimate_real_modes(
            inputs,
            cancellation_requested=lambda: cancellation.cancelled,
            progress=progress,
        )
        selected = next(
            value for value in modes if value.mode_id == selected_mode_id
        )
        final_solution = selected.final_solution
        static_geometry = selected.static_geometry
        delay_uncertainty = selected.delay_uncertainty
        performance = measure_estimation_modes_performance(
            modes, selected_mode_id
        )
        provisional_result = SimpleNamespace(
            modes=modes,
            selected_mode=selected,
            mcmc=None,
        )
        identity = ArtifactRunIdentity(
            estimator_revision=estimator_revision,
            configuration_fingerprint=configuration_digest,
            controller_snapshot_fingerprint=controller_digest,
            warnings=_warnings(provisional_result, estimator_revision),
        )
        core = export_batch_estimation_artifact_payload(
            request=request,
            flight_data=inputs.flight_data,
            initializations=inputs.initializations,
            final_solution=final_solution,
            em_result=selected.em,
            static_geometry=static_geometry,
            final_q_lag_profile=(
                selected.final_q_lag_profile_history[-1]
                if selected.final_q_lag_profile_history
                else None
            ),
            delay_geometry=DelayLocalGeometry(
                delay_uncertainty.standard_deviation_seconds,
                delay_uncertainty.source,
                delay_uncertainty.curvature,
            ),
            identity=identity,
            performance=performance,
            pending_mcmc_checkpoint=True,
        )
        checkpoint = write_batch_estimation_checkpoint(
            request.output_directory,
            request=request,
            estimator_revision=estimator_revision,
            configuration_fingerprint=configuration_digest,
            controller_snapshot_fingerprint=controller_digest,
            selected_mode_id=selected_mode_id,
            core=core,
            state=final_solution.lm.state,
        )
        checkpoint_root = checkpoint.root
        chain_checkpoints = {}

    target_timings = []
    try:
        cancellation.raise_if_cancelled()
        mcmc = sample_laplace_solution(
            inputs,
            selected_mode_id,
            final_solution,
            static_geometry,
            delay_uncertainty,
            cancellation_requested=lambda: cancellation.cancelled,
            progress=progress,
            target_timing_callback=target_timings.append,
            chain_checkpoints=chain_checkpoints,
            checkpoint_chain_proposal=lambda _chain_id, value: (
                save_batch_chain_checkpoint(checkpoint_root, value)
            ),
        )
        cancellation.raise_if_cancelled()
    except Exception:
        if cancellation.cancelled:
            mark_batch_checkpoint_cancelled(
                checkpoint_root, cancellation.reason
            )
        raise

    completed_payload = complete_pending_mcmc_artifact_payload(
        core,
        request,
        final_solution,
        mcmc.chains,
        mcmc.diagnostics,
        target_timings,
    )
    if progress is not None:
        progress("writing_artifacts", 0, 1, "publishing strict run")
    cancellation.raise_if_cancelled()
    written = write_batch_estimation_run(
        request.output_directory, **completed_payload.writer_arguments
    )
    mark_batch_checkpoint_published(checkpoint_root)
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
        request = load_batch_estimation_request(arguments.request)
        progress = BatchProgressReporter(
            request, JsonlProgressWriter(sys.stdout)
        )
        output = execute_batch_estimation(
            request,
            estimator_revision=discover_estimator_revision(),
            cancellation_token=cancellation,
            progress=progress,
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
    "BatchProgressReporter",
    "StageProgress",
    "configuration_fingerprint",
    "controller_snapshot_fingerprint",
    "discover_estimator_revision",
    "execute_batch_estimation",
    "main",
    "planned_progress_units",
    "run_request",
]
