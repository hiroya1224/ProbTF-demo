"""One-command worker for posterior PID candidate cross-evaluation."""

import argparse
import signal
import sys
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

from grape_param_estim.batch_artifact import (
    BatchEstimationRun,
    file_sha256,
    load_batch_estimation_run,
)
from grape_param_estim.batch_estimation_cli import controller_snapshot_fingerprint
from grape_param_estim.controller_config import (
    PidGainConfiguration,
    configuration_from_controller_snapshot,
    select_baseline_pid_configuration,
)
from grape_param_estim.pid.artifact import (
    PidCandidatePopulationAudit,
    PidEvaluationArtifactIdentity,
    PidEvaluationRuntimeDiagnostics,
    PidProposalEvaluationArtifact,
    write_pid_proposal_evaluation,
)
from grape_param_estim.pid.checkpoint import (
    PidForecastCheckpointIdentity,
    PidForecastCheckpointStore,
    checkpoint_root_for_output,
)
from grape_param_estim.pid.input import (
    PidBagForecastModel,
    forecast_scenarios_from_batch_run,
    physical_posterior_from_batch_run,
)
from grape_param_estim.pid.particle_search import (
    MODEL_DISCREPANCY_INTERVAL_MODELS,
    MODEL_DISCREPANCY_QUANTITIES,
    ModelDiscrepancyConfiguration,
    build_initial_candidate_population,
    evaluate_pid_candidates,
    resolve_forecast_worker_count,
)
from grape_param_estim.pid.predictive import ClosedLoopPidForecastEvaluator
from grape_param_estim.pid.proposal import (
    derive_pid_proposals,
    user_pid_candidate,
)
from grape_param_estim.pid.request import (
    PidEvaluationRequest,
    load_pid_evaluation_request,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCallback,
    ProgressTracker,
    STAGE_OPTIMIZING_FULL_TRAJECTORY,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_WRITING_ARTIFACTS,
)
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    VehicleParameters,
)


def _actuator_parameters(manifest: Mapping[str, object]) -> ActuatorParameters:
    model = manifest.get("actuator_model")
    if not isinstance(model, Mapping):
        raise ValueError("estimation manifest has no explicit actuator_model")
    expected = {
        "source",
        "thrust_time_constant_seconds",
        "gimbal_time_constant_seconds",
        "minimum_thrust_newtons",
        "maximum_thrust_newtons",
        "maximum_gimbal_angle_radians",
        "maximum_gimbal_rate_radians_per_second",
    }
    if set(model) != expected:
        raise ValueError("estimation manifest actuator_model fields disagree")
    if not isinstance(model["source"], str) or not model["source"]:
        raise ValueError("estimation manifest actuator_model source is invalid")
    return ActuatorParameters(
        thrust_time_constant=float(model["thrust_time_constant_seconds"]),
        gimbal_time_constant=float(model["gimbal_time_constant_seconds"]),
        delay=0.0,
        minimum_thrust=float(model["minimum_thrust_newtons"]),
        maximum_thrust=float(model["maximum_thrust_newtons"]),
        maximum_gimbal_angle=float(model["maximum_gimbal_angle_radians"]),
        maximum_gimbal_rate=float(
            model["maximum_gimbal_rate_radians_per_second"]
        ),
    )


def _q_contract(run: BatchEstimationRun):
    definition = str(run.manifest["q_definition"]["definition"])
    parts = definition.split("/", 1)
    if len(parts) != 2:
        raise ValueError("estimation Q definition must include its interval model")
    quantity, interval_model = parts
    if quantity not in MODEL_DISCREPANCY_QUANTITIES:
        raise ValueError("estimation Q residual quantity is unsupported")
    if interval_model not in MODEL_DISCREPANCY_INTERVAL_MODELS:
        raise ValueError("estimation Q interval model is unsupported")
    return quantity, interval_model


def _requested_candidates(request, proposals):
    users = tuple(
        user_pid_candidate(
            item.candidate_id, PidGainConfiguration(item.gain_values)
        )
        for item in request.candidates
        if item.source == "user"
    )
    return build_initial_candidate_population(
        proposals,
        maximum_derived_candidates=request.maximum_derived_candidates,
        required_source_sample_ids=request.required_derived_sample_ids,
        user_candidates=users,
    )


def execute_pid_evaluation(
    request: PidEvaluationRequest,
    *,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> PidProposalEvaluationArtifact:
    """Load one completed batch run, forecast all candidates, and publish."""

    if not isinstance(request, PidEvaluationRequest):
        raise TypeError("request must be PidEvaluationRequest")
    cancellation = (
        CancellationToken() if cancellation_token is None else cancellation_token
    )
    cancellation.raise_if_cancelled()
    run = load_batch_estimation_run(request.estimation_run)
    expected_bags = tuple(run.manifest["selected_bag_ids"])
    if tuple(value.bag_id for value in request.bags) != expected_bags:
        raise ValueError("PID request bags must preserve estimation bag order")
    posterior = physical_posterior_from_batch_run(
        run,
        fixed_linear_drag=request.fixed_linear_drag,
        fixed_angular_drag=request.fixed_angular_drag,
        selected_mode_id=request.selected_mode_id,
    )
    selected_sample_count = (
        len(posterior.samples)
        if request.plant_sample_ids is None
        else len(request.plant_sample_ids)
    )
    derived_candidate_count = min(
        len(posterior.samples),
        (
            len(posterior.samples)
            if request.maximum_derived_candidates is None
            else request.maximum_derived_candidates
        ),
    )
    candidate_count = 1 + derived_candidate_count + sum(
        value.source == "user" for value in request.candidates
    )
    forecast_count = (
        candidate_count
        * selected_sample_count
        * len(request.bags)
        * request.discrepancy_replicates
    )
    tracker = ProgressTracker(
        request.evaluation_id,
        overall_total_units=len(request.bags) + forecast_count + 1,
        callback=progress_callback,
        cancellation_token=cancellation,
    )
    preparation = tracker.begin_stage(
        STAGE_PREPARING_TRAJECTORY, len(request.bags)
    )
    flight_data = {}
    for completed, bag in enumerate(request.bags, start=1):
        tracker.checkpoint()
        if file_sha256(bag.path) != bag.sha256:
            raise ValueError("PID request rosbag SHA-256 disagrees with estimation")
        if run.manifest["selected_bag_sha256"][bag.bag_id] != bag.sha256:
            raise ValueError("PID request rosbag SHA-256 disagrees with manifest")
        interval = run.manifest["selected_intervals"][bag.bag_id]
        flight_data[bag.bag_id] = load_flight_data(
            path=str(bag.path),
            start_local=float(interval[0]),
            end_local=float(interval[1]),
            compute_sha256=False,
            checkpoint=tracker.checkpoint,
            bag_id=bag.bag_id,
        )
        preparation.emit(
            completed,
            bag_id=bag.bag_id,
            message="Loaded recorded controller snapshot and bag contract",
        )
    controller_inputs = SimpleNamespace(
        flight_data=tuple(flight_data[bag_id] for bag_id in expected_bags)
    )
    if controller_snapshot_fingerprint(controller_inputs) != run.manifest[
        "controller_snapshot_fingerprint"
    ]:
        raise ValueError("recorded controller snapshots disagree with estimation")
    current = select_baseline_pid_configuration(
        {
            bag_id: configuration_from_controller_snapshot(
                flight.controller_snapshot, bag_id
            )
            for bag_id, flight in flight_data.items()
        },
        request.baseline_bag_id,
    )
    nominal = VehicleParameters.nominal()
    geometry = GrapeGeometry.grape()
    actuator_parameters = _actuator_parameters(run.manifest)
    bag_request = {value.bag_id: value for value in request.bags}
    models = tuple(
        PidBagForecastModel(
            bag_id=bag_id,
            controller_configuration=flight_data[bag_id].controller_configuration,
            controller_nominal_parameters=nominal,
            controller_geometry=geometry,
            plant_geometry=geometry,
            actuator_parameters=actuator_parameters,
            roll_pitch_integration_active=(
                bag_request[bag_id].roll_pitch_integration_active
            ),
            maximum_reference_age_seconds=request.maximum_reference_age_seconds,
        )
        for bag_id in expected_bags
    )
    scenarios = forecast_scenarios_from_batch_run(run, posterior, models)
    proposals = derive_pid_proposals(posterior, nominal, geometry, current)
    candidates = _requested_candidates(request, proposals)
    if len(candidates) != candidate_count:
        raise RuntimeError("derived PID candidate count changed during evaluation")
    q_quantity, q_interval_model = _q_contract(run)
    discrepancy = ModelDiscrepancyConfiguration(
        policy=request.discrepancy_policy,
        diagonal_q=run.map_static["q_diagonal"],
        base_seed=request.discrepancy_base_seed,
        residual_quantity=q_quantity,
        interval_model=q_interval_model,
        replicates=request.discrepancy_replicates,
    )
    evaluation_stage = tracker.begin_stage(
        STAGE_OPTIMIZING_FULL_TRAJECTORY, forecast_count
    )

    checkpoint = PidForecastCheckpointStore.open(
        checkpoint_root_for_output(request.output_directory),
        PidForecastCheckpointIdentity(
            evaluation_id=request.evaluation_id,
            estimation_run_id=str(run.manifest["run_id"]),
            request_fingerprint=request.fingerprint,
            estimation_request_fingerprint=str(
                run.manifest["request_fingerprint"]
            ),
        ),
        resume=request.resume,
        flush_size=resolve_forecast_worker_count(
            request.forecast_workers, forecast_count
        ),
    )
    resumed_forecasts = checkpoint.resumed_record_count
    if resumed_forecasts:
        evaluation_stage.emit(
            resumed_forecasts,
            message="Resumed {} completed PID forecasts".format(
                resumed_forecasts
            ),
        )

    def report(completed, total, record):
        if total != forecast_count:
            raise RuntimeError("PID forecast work count changed during evaluation")
        evaluation_stage.emit(
            completed,
            bag_id=record.bag_id,
            sample_id=record.sample_id,
            message="candidate={} replicate={}".format(
                record.candidate_id, record.replicate_index
            ),
        )

    try:
        evaluation = evaluate_pid_candidates(
            candidates,
            posterior,
            expected_bags,
            ClosedLoopPidForecastEvaluator(scenarios),
            current,
            discrepancy,
            sample_ids=request.plant_sample_ids,
            plant_sample_subset_method=request.plant_sample_subset_method,
            quantile_level=request.quantile_level,
            cvar_level=request.cvar_level,
            cancellation_requested=lambda: cancellation.cancelled,
            progress=report,
            worker_count=request.forecast_workers,
            initial_records=checkpoint.records,
            forecast_completed=checkpoint.record_completed,
        )
    except BaseException:
        checkpoint.flush()
        raise
    checkpoint.flush()
    writing = tracker.begin_stage(STAGE_WRITING_ARTIFACTS, 1)
    artifact = write_pid_proposal_evaluation(
        request.output_directory,
        identity=PidEvaluationArtifactIdentity(
            evaluation_id=request.evaluation_id,
            estimation_run_id=str(run.manifest["run_id"]),
            estimation_request_fingerprint=str(run.manifest["request_fingerprint"]),
            request_fingerprint=request.fingerprint,
        ),
        posterior=posterior,
        proposals=proposals,
        evaluation=evaluation,
        candidate_population=PidCandidatePopulationAudit(
            method=request.derived_candidate_method,
            maximum_candidates=request.maximum_derived_candidates,
            required_source_sample_ids=request.required_derived_sample_ids,
            raw_derived_candidate_count=len(posterior.samples),
        ),
        selected_candidate_id=request.selected_candidate_id,
        runtime_diagnostics=PidEvaluationRuntimeDiagnostics(
            requested_forecast_workers=request.forecast_workers,
            used_forecast_workers=resolve_forecast_worker_count(
                request.forecast_workers, forecast_count
            ),
            forecast_count=forecast_count,
            resumed_forecast_count=resumed_forecasts,
        ),
    )
    checkpoint.discard()
    writing.complete(message="PID evaluation artifact complete")
    return artifact


def _signal_reason(signum: int) -> str:
    try:
        return "signal_{}".format(signal.Signals(signum).name)
    except ValueError:
        return "signal_{}".format(signum)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate PID gains over retained plant posterior samples."
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
        request = load_pid_evaluation_request(arguments.request)
        artifact = execute_pid_evaluation(
            request,
            cancellation_token=cancellation,
            progress_callback=JsonlProgressWriter(sys.stdout),
        )
        print("PID evaluation complete: {}".format(artifact.root), file=sys.stderr)
        return 0
    except Exception as error:  # pylint: disable=broad-except
        print("PID evaluation failed: {}".format(error), file=sys.stderr)
        return 2 if cancellation.cancelled else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute_pid_evaluation", "main"]
