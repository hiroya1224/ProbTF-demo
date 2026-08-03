"""Request-file worker for production posterior-predictive PID evaluation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
from typing import Optional, Sequence

from grape_param_estim.artifact_io import (
    ASSIMILATION_RUN_SCHEMA,
    COMPLETE_STATUS,
    PID_PROPOSAL_EVALUATION_SCHEMA,
    WRITING_STATUS,
    begin_bundle,
    mark_bundle_cancelled,
    read_manifest,
)
from grape_param_estim.pid_evaluation_input import (
    candidates_from_request,
    input_from_assimilation_run,
    load_pid_evaluation_request,
)
from grape_param_estim.posterior_predictive import (
    evaluate_pid_proposals,
    save_pid_proposal_evaluation,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCallback,
    ProgressCancelled,
    ProgressEvent,
    ProgressTracker,
)


def _evaluation_manifest(request, source_manifest, created_at):
    bag_ids = tuple(source_manifest["selected_bag_ids"])
    return {
        "schema": PID_PROPOSAL_EVALUATION_SCHEMA,
        "evaluation_id": request.evaluation_id,
        "source_run_id": str(source_manifest["run_id"]),
        "created_at": str(created_at),
        "selected_bag_ids": list(bag_ids),
        "artifacts": {
            "proposal_ensemble": "proposal_ensemble.npz",
            "summary": "summary.npz",
            "proposed_yaml": "proposed_GimbalrotorControl.yaml",
            "proposed_diff_yaml": (
                "proposed_GimbalrotorControl.diff.yaml"
            ),
            "bags": {
                bag_id: "bags/{}.npz".format(bag_id)
                for bag_id in bag_ids
            },
        },
    }


def _cancel_writing_bundle(output: Path, reason: str) -> None:
    manifest = read_manifest(output)
    if manifest["status"] == WRITING_STATUS:
        mark_bundle_cancelled(output, reason)


def run_request(
    request_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> Path:
    """Evaluate one strict request against one complete assimilation run."""

    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    cancellation.raise_if_cancelled()
    request = load_pid_evaluation_request(request_path)
    cancellation.raise_if_cancelled()
    source_manifest = read_manifest(request.assimilation_run)
    if source_manifest.get("schema") != ASSIMILATION_RUN_SCHEMA:
        raise ValueError("assimilation_run is not an assimilation bundle")
    if source_manifest.get("status") != COMPLETE_STATUS:
        raise ValueError("assimilation_run must be complete")
    bag_ids = tuple(source_manifest["selected_bag_ids"])
    request.residual_policies(bag_ids)
    member_count = int(source_manifest["shared_member_count"])
    forecast_units = len(request.candidates) * len(bag_ids) * member_count
    total_units = forecast_units + 4
    created_at = datetime.now(timezone.utc).isoformat()
    output = Path(output_path).expanduser().resolve()
    begin_bundle(
        output,
        _evaluation_manifest(request, source_manifest, created_at),
    )
    tracker = ProgressTracker(
        run_id=request.evaluation_id,
        total_units=total_units,
        callback=progress_callback,
        cancellation_token=cancellation,
        eta_calibration_units=min(16, max(2, forecast_units // 8)),
    )
    try:
        tracker.emit(
            1,
            "request_validation",
            "PID evaluation request validated",
        )
        predictive_input = input_from_assimilation_run(
            request.assimilation_run,
            request.baseline_bag_id,
            residual_policy=request.residual_policies(bag_ids),
            cancellation_token=cancellation,
        )
        candidates = candidates_from_request(request, predictive_input)
        tracker.emit(
            2,
            "evaluation_input_restoration",
            "Completed assimilation run restored",
            message="{} bags, {} raw members".format(
                len(bag_ids), member_count
            ),
        )

        def forecast_progress(event: ProgressEvent) -> None:
            tracker.emit(
                2 + event.completed_units,
                event.stage_id,
                event.stage_label,
                iteration=event.iteration,
                maximum_iterations=event.maximum_iterations,
                bag_id=event.bag_id,
                member_id=event.member_id,
                message=event.message,
            )

        decision = evaluate_pid_proposals(
            predictive_input,
            candidates=candidates,
            cvar_level=request.cvar_level,
            thresholds=request.thresholds,
            selected_candidate_id=request.selected_candidate_id,
            progress_callback=forecast_progress,
            progress_run_id=request.evaluation_id,
            cancellation_token=cancellation,
        )
        completed_forecasts = 2 + forecast_units
        tracker.emit(
            completed_forecasts,
            "artifact_writing",
            "Writing PID proposal evaluation",
        )
        save_pid_proposal_evaluation(
            str(output),
            decision,
            evaluation_id=request.evaluation_id,
            source_run_id=str(source_manifest["run_id"]),
            created_at=created_at,
            yaml_candidate_id=request.selected_candidate_id,
            bundle_started=True,
            cancellation_token=cancellation,
        )
        try:
            tracker.emit(
                completed_forecasts + 1,
                "artifact_writing",
                "PID proposal artifacts written",
            )
            tracker.emit(
                total_units,
                "complete",
                "PID proposal evaluation complete",
            )
        except ProgressCancelled:
            if read_manifest(output)["status"] != COMPLETE_STATUS:
                raise
        return output
    except ProgressCancelled as error:
        _cancel_writing_bundle(output, error.reason)
        raise


def _signal_reason(signum: int) -> str:
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    return "signal_{}".format(name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact PID candidates against a completed assimilation run."
        )
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel(_signal_reason(signum))

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)
    try:
        output = run_request(
            arguments.request,
            arguments.output,
            progress_callback=JsonlProgressWriter(sys.stdout),
            cancellation_token=cancellation,
        )
        print(
            "PID proposal evaluation complete: {}".format(output),
            file=sys.stderr,
        )
        return 0
    except ProgressCancelled as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:  # pylint: disable=broad-except
        print("PID evaluation failed: {}".format(error), file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_request"]
