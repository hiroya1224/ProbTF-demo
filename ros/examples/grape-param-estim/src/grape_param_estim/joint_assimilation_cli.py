"""Request-file worker for one or more real-flight assimilation windows."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import sys
from typing import Mapping, Sequence

import numpy as np

from grape_param_estim.artifact_io import (
    begin_bundle,
    mark_bundle_cancelled,
    mark_bundle_complete,
    read_json,
    request_fingerprint,
)
from grape_param_estim.ensemble_solver import EstimationCancelled
from grape_param_estim.joint_assimilation import (
    FORECAST_START_METHOD,
    JointIEnKSConfig,
    assimilate_joint_flights,
    assimilation_run_manifest,
    initial_prior_forecast_manifest,
    prepare_joint_flight,
    write_joint_assimilation_payloads,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressTracker,
)
from grape_param_estim.real_rosbag import (
    build_real_flight_episode,
    read_grape_rosbag_arrays,
)


ASSIMILATION_REQUEST_SCHEMA = (
    "grape-param-estim/assimilation-request/v1"
)
_ITERATION_PROGRESS_LABEL = re.compile(
    r"\biteration\s+(\d+)/(\d+)\b", re.IGNORECASE
)


def _iteration_progress_metadata(label: str):
    match = _ITERATION_PROGRESS_LABEL.search(str(label))
    if match is None:
        return None, None
    iteration = int(match.group(1))
    maximum = int(match.group(2))
    if iteration < 1 or maximum < 1 or iteration > maximum:
        return None, None
    return iteration, maximum


def _finite(value, name: str, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError("{} is invalid".format(name))
    return result


def _validate_request(value: Mapping[str, object]) -> Mapping[str, object]:
    if value.get("schema") != ASSIMILATION_REQUEST_SCHEMA:
        raise ValueError("unsupported assimilation request schema")
    for key in (
        "run_id",
        "project_id",
        "project_request_fingerprint",
        "bags",
        "settings",
        "baseline_bag_id",
    ):
        if key not in value:
            raise ValueError("assimilation request is missing {}".format(key))
    if not isinstance(value["run_id"], str) or not value["run_id"]:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(value["project_id"], str) or not value["project_id"]:
        raise ValueError("project_id must be a non-empty string")
    project_fingerprint = value["project_request_fingerprint"]
    if (
        not isinstance(project_fingerprint, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", project_fingerprint)
    ):
        raise ValueError("project_request_fingerprint is invalid")
    bags = value["bags"]
    if not isinstance(bags, list) or not bags:
        raise ValueError("request must select at least one bag")
    identifiers = []
    for index, bag in enumerate(bags):
        if not isinstance(bag, dict):
            raise ValueError("bag request {} must be an object".format(index))
        required = {
            "bag_id",
            "path",
            "sha256",
            "episode_index",
            "selected_interval",
            "configuration_fingerprint",
        }
        if not required.issubset(bag):
            raise ValueError("bag request {} is incomplete".format(index))
        identifier = bag["bag_id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("bag_id must be a non-empty string")
        identifiers.append(identifier)
        path = Path(str(bag["path"])).expanduser().resolve()
        if not path.is_file():
            raise ValueError("bag file does not exist: {}".format(path))
        digest = str(bag["sha256"])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise ValueError("bag SHA256 is invalid")
        interval = bag["selected_interval"]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or _finite(interval[1], "interval end")
            <= _finite(interval[0], "interval start")
        ):
            raise ValueError("selected interval must be increasing")
        if int(bag["episode_index"]) < 0:
            raise ValueError("episode_index cannot be negative")
        if not isinstance(bag["configuration_fingerprint"], str) or not bag[
            "configuration_fingerprint"
        ]:
            raise ValueError("configuration fingerprint cannot be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("selected bag IDs must be unique")
    if value["baseline_bag_id"] not in identifiers:
        raise ValueError("baseline_bag_id must identify one selected bag")
    settings = value["settings"]
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    for key in (
        "sample_period",
        "maximum_knots",
        "ensemble_size",
        "maximum_iterations",
        "seed",
        "delay_prior_mean",
        "delay_prior_standard_deviation",
        "allow_configuration_mismatch",
    ):
        if key not in settings:
            raise ValueError("settings is missing {}".format(key))
    _finite(settings["sample_period"], "sample_period", positive=True)
    maximum_knots = int(settings["maximum_knots"])
    if maximum_knots == 1 or maximum_knots < 0:
        raise ValueError("maximum_knots must be zero or at least two")
    if int(settings["ensemble_size"]) < 3:
        raise ValueError("ensemble_size must be at least three")
    if int(settings["maximum_iterations"]) < 1:
        raise ValueError("maximum_iterations must be positive")
    raw_maximum_backoff = settings.get(
        "maximum_initial_prior_backoff_trials", 8
    )
    maximum_backoff = int(raw_maximum_backoff)
    if (
        isinstance(raw_maximum_backoff, (bool, np.bool_))
        or maximum_backoff != raw_maximum_backoff
        or not 0 <= maximum_backoff <= 30
    ):
        raise ValueError(
            "maximum_initial_prior_backoff_trials must be in [0, 30]"
        )
    delay_prior_mean = _finite(
        settings["delay_prior_mean"], "delay prior mean"
    )
    if delay_prior_mean < 0.0:
        raise ValueError("delay prior mean cannot be negative")
    _finite(
        settings["delay_prior_standard_deviation"],
        "delay prior standard deviation",
        positive=True,
    )
    if not isinstance(settings["allow_configuration_mismatch"], bool):
        raise ValueError("allow_configuration_mismatch must be boolean")
    fingerprints = {
        str(bag["configuration_fingerprint"]) for bag in bags
    }
    if len(fingerprints) > 1 and not settings[
        "allow_configuration_mismatch"
    ]:
        raise ValueError(
            "selected bags have different configuration fingerprints"
        )
    return value


def _configuration_fingerprint(request: Mapping[str, object]) -> str:
    values = sorted(
        {
            str(bag["configuration_fingerprint"])
            for bag in request["bags"]
        }
    )
    if len(values) == 1:
        return values[0]
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    return "explicit-override:sha256:{}".format(digest)


def _maximum_line_search_trials(minimum_fraction: float) -> int:
    return int(math.floor(math.log(1.0 / minimum_fraction, 2))) + 1


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolved_forecast_workers(
    request: Mapping[str, object], ensemble_size: int
) -> int:
    execution = request.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    requested = execution.get("forecast_workers")
    if requested is None:
        requested = os.environ.get(
            "GRAPE_PARAM_ESTIM_FORECAST_WORKERS", "auto"
        )
    if isinstance(requested, str):
        normalised = requested.strip().lower()
        if normalised in {"", "0", "auto"}:
            cpu_count = _available_cpu_count()
            return min(int(ensemble_size), 16, max(1, cpu_count // 2))
        try:
            requested = int(normalised)
        except ValueError as error:
            raise ValueError(
                "forecast_workers must be auto or a positive integer"
            ) from error
    try:
        valid = (
            not isinstance(requested, (bool, np.bool_))
            and int(requested) == requested
            and 1 <= int(requested) <= 256
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(
            "forecast_workers must be auto or an integer in [1, 256]"
        )
    return min(int(requested), int(ensemble_size))


def run_request(request_path: str, output_path: str) -> Path:
    request_source = Path(request_path).expanduser().resolve()
    request = _validate_request(read_json(request_source))
    output = Path(output_path).expanduser().resolve()
    settings = request["settings"]
    forecast_workers = _resolved_forecast_workers(
        request, int(settings["ensemble_size"])
    )
    configuration = JointIEnKSConfig(
        ensemble_size=int(settings["ensemble_size"]),
        maximum_iterations=int(settings["maximum_iterations"]),
        convergence_tolerance=float(
            settings.get("convergence_tolerance", 1.0e-3)
        ),
        minimum_line_search_step=float(
            settings.get("minimum_line_search_step", 1.0 / 64.0)
        ),
        seed=int(settings["seed"]),
        maximum_initial_prior_backoff_trials=int(
            settings.get("maximum_initial_prior_backoff_trials", 8)
        ),
        forecast_workers=forecast_workers,
    )
    intervals = {
        str(bag["bag_id"]): tuple(bag["selected_interval"])
        for bag in request["bags"]
    }
    fingerprint = request_fingerprint(request)
    manifest = assimilation_run_manifest(
        run_id=request["run_id"],
        request_path=str(request_source),
        request_fingerprint=fingerprint,
        project_request_fingerprint=request[
            "project_request_fingerprint"
        ],
        selected_intervals=intervals,
        configuration_fingerprint=_configuration_fingerprint(request),
        member_count=configuration.ensemble_size,
        estimator_revision=os.environ.get(
            "GRAPE_PARAM_ESTIM_REVISION", "workspace"
        ),
    )
    manifest["execution"] = {
        "forecast_workers": forecast_workers,
        "multiprocessing_start_method": (
            FORECAST_START_METHOD if forecast_workers > 1 else None
        ),
    }
    begin_bundle(output, manifest)
    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel("signal_{}".format(signum))

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)
    bag_count = len(request["bags"])
    trials = _maximum_line_search_trials(
        configuration.minimum_line_search_step
    )
    forecast_units = bag_count * (
        1
        + configuration.maximum_iterations
        * (configuration.ensemble_size + trials)
        # Every failed initial-prior attempt is retried as one complete,
        # globally contracted ensemble.  Reserve the configured worst case
        # so completed units never exceed the immutable progress total.
        + configuration.maximum_initial_prior_backoff_trials
        * configuration.ensemble_size
        # Posterior linearization + posterior residual ensemble, followed by
        # the raw posterior + prior trajectory replays saved in the bundle.
        + 4 * configuration.ensemble_size
        + 1
    )
    preparation_units = 2 + 6 * bag_count
    total_units = preparation_units + forecast_units + 2
    tracker = ProgressTracker(
        run_id=request["run_id"],
        total_units=total_units,
        callback=JsonlProgressWriter(sys.stdout),
        cancellation_token=cancellation,
        eta_calibration_units=min(16, max(2, total_units // 20)),
    )
    completed = [0]
    active_stage = [
        "ensemble_forecast",
        "Joint ensemble forecast",
        None,
        None,
    ]
    active_batch_units = [0, 0]
    prior_unit_recorded = [False]

    def advance(
        stage_id,
        stage_label,
        bag_id=None,
        member_id=None,
        message="",
        iteration=None,
        maximum_iterations=None,
    ):
        completed[0] += 1
        tracker.emit(
            completed[0],
            stage_id,
            stage_label,
            iteration=iteration,
            maximum_iterations=maximum_iterations,
            bag_id=bag_id,
            member_id=member_id,
            message=message,
        )

    def advance_units(
        units,
        stage_id,
        stage_label,
        message="",
        iteration=None,
        maximum_iterations=None,
    ):
        count = int(units)
        if count <= 0:
            return
        completed[0] += count
        tracker.emit(
            completed[0],
            stage_id,
            stage_label,
            iteration=iteration,
            maximum_iterations=maximum_iterations,
            message=message,
        )

    def forecast_batch_units(stage_id, label):
        if stage_id in {
            "ensemble_forecast",
            "initial_prior_backoff_forecast",
        }:
            return configuration.ensemble_size * bag_count
        if stage_id == "posterior_ensemble_forecast":
            if label in {"posterior linearization", "posterior ensemble"}:
                return configuration.ensemble_size * bag_count
            if label == "posterior center":
                return bag_count
        if stage_id in {"initial_forecast", "line_search_trial"}:
            return bag_count
        return 0

    def solver_stage(stage_id, stage_done, _total, message):
        label = message or stage_id.replace("_", " ")
        iteration, maximum = _iteration_progress_metadata(label)
        if stage_id == "initial_prior_forecast_failed":
            missing = active_batch_units[0] - active_batch_units[1]
            if missing < 0:
                raise RuntimeError(
                    "initial-prior forecast progress exceeded its batch size"
                )
            advance_units(
                missing,
                "initial_prior_backoff_skipped",
                label,
                message=(
                    "{} member-bag forecasts were not executed after the "
                    "numerical failure"
                ).format(missing),
                iteration=iteration,
                maximum_iterations=maximum,
            )
        active_stage[:] = [stage_id, label, iteration, maximum]
        active_batch_units[:] = [forecast_batch_units(stage_id, label), 0]
        if (
            stage_id == "prior_ensemble_generation"
            and stage_done == 1
            and not prior_unit_recorded[0]
        ):
            prior_unit_recorded[0] = True
            advance(
                stage_id,
                active_stage[1],
                iteration=iteration,
                maximum_iterations=maximum,
            )
            return
        tracker.emit(
            completed[0],
            stage_id,
            message or "ensemble forecast",
            iteration=iteration,
            maximum_iterations=maximum,
        )

    def member_bag(member, bag_id, _done, _total):
        active_batch_units[1] += 1
        advance(
            active_stage[0],
            active_stage[1],
            bag_id=bag_id,
            member_id=int(member),
            message="forecasting member {} for {}".format(member, bag_id),
            iteration=active_stage[2],
            maximum_iterations=active_stage[3],
        )

    try:
        advance("request_validation", "Request validated")
        prepared = []
        for bag_request in sorted(
            request["bags"], key=lambda value: value["bag_id"]
        ):
            tracker.checkpoint()
            bag_id = str(bag_request["bag_id"])
            arrays = read_grape_rosbag_arrays(
                str(Path(bag_request["path"]).expanduser().resolve()),
                compute_sha256=True,
            )
            advance("bag_reading", "Bag records loaded", bag_id)
            if arrays.bag_sha256.lower() != str(
                bag_request["sha256"]
            ).lower():
                raise ValueError("bag SHA256 changed: {}".format(bag_id))
            advance("sha256_validation", "Bag SHA256 verified", bag_id)
            advance(
                "topic_contract_validation",
                "Required topic and message contracts validated",
                bag_id,
            )
            interval = bag_request["selected_interval"]
            episode = build_real_flight_episode(
                arrays,
                sample_period=float(settings["sample_period"]),
                episode_index=int(bag_request["episode_index"]),
                start_local=float(interval[0]),
                end_local=float(interval[1]),
                window_state=None,
            )
            advance(
                "interval_building", "Flight interval built", bag_id
            )
            advance(
                "pose_covariance_calibration",
                "Pose covariance calibrated",
                bag_id,
            )
            prepared.append(
                prepare_joint_flight(
                    bag_id,
                    episode,
                    str(bag_request["configuration_fingerprint"]),
                    maximum_knots=(
                        None
                        if int(settings["maximum_knots"]) == 0
                        else int(settings["maximum_knots"])
                    ),
                    initial_delay=float(settings["delay_prior_mean"]),
                )
            )
            advance(
                "model_error_calibration",
                "Model error calibrated",
                bag_id,
            )
        result = assimilate_joint_flights(
            prepared,
            configuration=configuration,
            delay_mean=float(settings["delay_prior_mean"]),
            delay_standard_deviation=float(
                settings["delay_prior_standard_deviation"]
            ),
            allow_configuration_mismatch=bool(
                settings["allow_configuration_mismatch"]
            ),
            progress_callback=solver_stage,
            cancel_requested=lambda: cancellation.cancelled,
            member_bag_callback=member_bag,
        )
        unused_backoff_units = (
            configuration.maximum_initial_prior_backoff_trials
            - result.posterior.initial_prior_forecast.backoff_trials
        ) * configuration.ensemble_size * bag_count
        advance_units(
            unused_backoff_units,
            "initial_prior_backoff_unused_reserve",
            "Unused initial-prior backoff reserve",
            message=(
                "{} configured radial-backoff trials were not required"
            ).format(
                configuration.maximum_initial_prior_backoff_trials
                - result.posterior.initial_prior_forecast.backoff_trials
            ),
        )
        if completed[0] < total_units - 1:
            tracker.emit(
                total_units - 1,
                "artifact_writing",
                "Writing assimilation run",
            )
            completed[0] = total_units - 1
        write_joint_assimilation_payloads(str(output), result)
        mark_bundle_complete(
            output,
            {
                "termination_reason": result.posterior.termination_reason,
                "converged": bool(result.posterior.converged),
                "initial_prior_forecast": initial_prior_forecast_manifest(
                    result.posterior
                ),
            },
        )
        tracker.emit(total_units, "complete", "Assimilation run complete")
        return output
    except (ProgressCancelled, EstimationCancelled) as error:
        reason = getattr(error, "reason", str(error))
        mark_bundle_cancelled(output, str(reason))
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run joint full closed-loop weak-constraint assimilation."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run_request(arguments.request, arguments.output)
    except (ProgressCancelled, EstimationCancelled) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
