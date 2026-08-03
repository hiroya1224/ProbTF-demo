"""Strict request worker for fixed-Q augmented-parameter assimilation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    CANCELLED_STATUS,
    COMPLETE_STATUS,
    WRITING_STATUS,
    canonical_json_bytes,
    read_json,
    request_fingerprint,
)
from grape_param_estim.augmented_parameter_artifact import (
    AugmentedParameterArtifactBagInput,
    diagonal_q_artifact_fingerprint,
    mark_augmented_parameter_artifact_cancelled,
    read_augmented_parameter_manifest,
    write_augmented_parameter_artifact,
)
from grape_param_estim.augmented_parameter_state import (
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    AugmentedParameterPrior,
)
from grape_param_estim.diagonal_q_artifact import load_diagonal_q_artifact
from grape_param_estim.diagonal_q_stage_cli import (
    _artifact_bag_input as _stage1_artifact_bag_input,
)
from grape_param_estim.multi_bag_augmented_parameter import (
    PreparedAugmentedParameterBag,
    run_multi_bag_augmented_parameter_filter,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressEvent,
    ProgressTracker,
)
from grape_param_estim.real_diagonal_q_estimation import (
    prepare_real_diagonal_q_bag,
)
from grape_param_estim.real_rosbag import (
    build_real_flight_episode,
    read_grape_rosbag_arrays,
)
from grape_param_estim.timing import BoundedDelayChart


AUGMENTED_PARAMETER_STAGE_REQUEST_SCHEMA = (
    "grape-param-estim/augmented-parameter-stage-request/v1"
)
AUGMENTED_PARAMETER_STAGE_ID = "static_parameters"
AUGMENTED_PARAMETER_ALGORITHM_VERSION = "augmented-static-enkf-v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIGURATION_FINGERPRINT = re.compile(
    r"^(?:(?:complete|incomplete):[0-9a-f]{64}"
    r"|manual-group:sha256:[0-9a-f]{64})$"
)
_TOP_LEVEL_KEYS = {
    "schema",
    "run_id",
    "project_fingerprint",
    "stage_id",
    "stage_input_fingerprint",
    "upstream_diagonal_q",
    "bags",
    "settings",
}
_UPSTREAM_KEYS = {"path", "artifact_fingerprint"}
_BAG_KEYS = {
    "bag_id",
    "path",
    "sha256",
    "episode_index",
    "selected_interval_local_seconds",
    "configuration_fingerprint",
}
_SETTINGS_KEYS = {
    "sample_period",
    "ensemble_size",
    "delay_prior_mean_seconds",
    "delay_prior_standard_deviation_seconds",
    "maximum_delay_seconds",
    "covariance_rcond",
    "seed",
    "forecast_workers",
}


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(label))
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "{} keys differ from schema; missing={}, extra={}".format(
                label, missing, extra
            )
        )
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError("{} must be a safe identifier".format(label))
    return value


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA256 fingerprint".format(label))
    return value


def _finite(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a finite number".format(label))
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number".format(label)) from error
    if not np.isfinite(selected):
        raise ValueError("{} must be finite".format(label))
    if positive and selected <= 0.0:
        raise ValueError("{} must be positive".format(label))
    if nonnegative and selected < 0.0:
        raise ValueError("{} must be non-negative".format(label))
    return selected


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("{} must be an integer".format(label))
    selected = int(value)
    if selected < minimum or (maximum is not None and selected > maximum):
        raise ValueError("{} is outside its supported range".format(label))
    return selected


def validate_augmented_parameter_stage_request(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact GUI-to-stage-2 request contract."""

    request = _exact_keys(
        value, _TOP_LEVEL_KEYS, "augmented-parameter request"
    )
    if request["schema"] != AUGMENTED_PARAMETER_STAGE_REQUEST_SCHEMA:
        raise ValueError("unsupported augmented-parameter request schema")
    _safe_id(request["run_id"], "run_id")
    _fingerprint(request["project_fingerprint"], "project_fingerprint")
    if request["stage_id"] != AUGMENTED_PARAMETER_STAGE_ID:
        raise ValueError(
            "stage_id must be {!r}".format(AUGMENTED_PARAMETER_STAGE_ID)
        )
    _fingerprint(
        request["stage_input_fingerprint"], "stage_input_fingerprint"
    )
    upstream = _exact_keys(
        request["upstream_diagonal_q"],
        _UPSTREAM_KEYS,
        "upstream_diagonal_q",
    )
    path_text = upstream["path"]
    if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
        raise ValueError("upstream_diagonal_q.path must be non-empty")
    try:
        upstream_path = Path(path_text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "upstream_diagonal_q.path cannot be resolved: {}".format(error)
        ) from error
    if not upstream_path.is_dir():
        raise ValueError("upstream_diagonal_q.path must name a directory")
    _fingerprint(
        upstream["artifact_fingerprint"],
        "upstream_diagonal_q.artifact_fingerprint",
    )

    bags = request["bags"]
    if not isinstance(bags, list) or not bags:
        raise ValueError("bags must be a non-empty list")
    identifiers = []
    configurations = set()
    for index, raw_bag in enumerate(bags):
        label = "bags[{}]".format(index)
        bag = _exact_keys(raw_bag, _BAG_KEYS, label)
        identifier = _safe_id(bag["bag_id"], label + ".bag_id")
        identifiers.append(identifier)
        bag_path_text = bag["path"]
        if (
            not isinstance(bag_path_text, str)
            or not bag_path_text
            or "\x00" in bag_path_text
        ):
            raise ValueError("{}.path must be non-empty".format(label))
        try:
            bag_path = Path(bag_path_text).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                "{}.path cannot be resolved: {}".format(label, error)
            ) from error
        if not bag_path.is_file():
            raise ValueError("{}.path must name a regular file".format(label))
        digest = bag["sha256"]
        if not isinstance(digest, str) or _RAW_SHA256.fullmatch(digest) is None:
            raise ValueError("{}.sha256 must be lowercase hexadecimal".format(label))
        _integer(bag["episode_index"], label + ".episode_index")
        interval = bag["selected_interval_local_seconds"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("{}.selected interval must have two bounds".format(label))
        start = _finite(interval[0], label + ".selected interval start")
        end = _finite(interval[1], label + ".selected interval end")
        if end <= start or not np.isfinite(end - start):
            raise ValueError("{}.selected interval must be increasing".format(label))
        configuration = bag["configuration_fingerprint"]
        if (
            not isinstance(configuration, str)
            or _CONFIGURATION_FINGERPRINT.fullmatch(configuration) is None
        ):
            raise ValueError(
                "{}.configuration_fingerprint is invalid".format(label)
            )
        configurations.add(configuration)
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("bags must have sorted, unique bag IDs")
    if len(configurations) != 1:
        raise ValueError("selected bags must share one configuration fingerprint")

    settings = _exact_keys(request["settings"], _SETTINGS_KEYS, "settings")
    _finite(settings["sample_period"], "sample_period", positive=True)
    _integer(
        settings["ensemble_size"],
        "ensemble_size",
        minimum=MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    )
    delay_mean = _finite(
        settings["delay_prior_mean_seconds"],
        "delay_prior_mean_seconds",
        positive=True,
    )
    _finite(
        settings["delay_prior_standard_deviation_seconds"],
        "delay_prior_standard_deviation_seconds",
        positive=True,
    )
    maximum_delay = _finite(
        settings["maximum_delay_seconds"],
        "maximum_delay_seconds",
        positive=True,
    )
    if delay_mean >= maximum_delay:
        raise ValueError("delay prior mean must be below maximum_delay_seconds")
    _finite(settings["covariance_rcond"], "covariance_rcond", positive=True)
    _integer(settings["seed"], "seed", maximum=2**32 - 1)
    workers = settings["forecast_workers"]
    if workers != "auto":
        _integer(workers, "forecast_workers", minimum=1, maximum=256)
    return request


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolved_forecast_workers(value: Any, ensemble_size: int) -> int:
    members = _integer(
        ensemble_size,
        "ensemble_size",
        minimum=MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    )
    if value == "auto":
        return min(members, 32, max(1, _available_cpu_count() // 2))
    return min(
        members,
        _integer(value, "forecast_workers", minimum=1, maximum=256),
    )


def _mapping_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _validate_request_against_upstream(
    request: Mapping[str, Any], upstream_bundle: Any
) -> None:
    if (
        upstream_bundle.manifest["project_fingerprint"]
        != request["project_fingerprint"]
    ):
        raise ValueError("upstream diagonal-Q project fingerprint differs")
    if upstream_bundle.manifest["stage_id"] != "diagonal_q":
        raise ValueError("upstream artifact is not the diagonal_q stage")
    requested_ids = tuple(value["bag_id"] for value in request["bags"])
    if requested_ids != upstream_bundle.bag_ids:
        raise ValueError("stage-2 bags differ from the upstream diagonal-Q bags")
    upstream_by_id = {
        value.bag_id: value for value in upstream_bundle.bag_inputs
    }
    for bag in request["bags"]:
        upstream = upstream_by_id[bag["bag_id"]]
        if bag["sha256"] != upstream.source_sha256:
            raise ValueError(
                "bag SHA256 differs from upstream: {}".format(bag["bag_id"])
            )
        if int(bag["episode_index"]) != upstream.episode_index:
            raise ValueError(
                "episode index differs from upstream: {}".format(
                    bag["bag_id"]
                )
            )
        if tuple(bag["selected_interval_local_seconds"]) != tuple(
            upstream.selected_interval_local_seconds
        ):
            raise ValueError(
                "selected interval differs from upstream: {}".format(
                    bag["bag_id"]
                )
            )
        if (
            bag["configuration_fingerprint"]
            != upstream.configuration_fingerprint
        ):
            raise ValueError(
                "configuration fingerprint differs from upstream: {}".format(
                    bag["bag_id"]
                )
            )


def _validate_rebuilt_bag_against_upstream(
    bag_id: str,
    rebuilt: Any,
    prepared: Any,
    upstream_input: Any,
    upstream_expectation: Any,
    upstream_pilot: Any,
) -> None:
    # ``source_path`` is provenance, not identity.  A saved project may be
    # restored below a different project root; SHA256 and size remain the
    # authoritative bag identity in that case.
    scalar_fields = (
        "source_sha256",
        "source_size_bytes",
        "selected_interval_local_seconds",
        "effective_interval_local_seconds",
        "episode_index",
        "configuration_fingerprint",
        "constant_delay_seconds",
        "fixed_model_fingerprint",
        "observation_covariance_fingerprint",
    )
    for name in scalar_fields:
        if getattr(rebuilt, name) != getattr(upstream_input, name):
            raise ValueError(
                "rebuilt {} differs from upstream for {}".format(name, bag_id)
            )
    for name in ("translation_covariance", "rotation_covariance"):
        if not np.array_equal(
            getattr(rebuilt, name), getattr(upstream_input, name)
        ):
            raise ValueError(
                "rebuilt {} differs from upstream for {}".format(name, bag_id)
            )
    if not _mapping_equal(
        rebuilt.fixed_model_provenance,
        upstream_input.fixed_model_provenance,
    ):
        raise ValueError("rebuilt fixed model differs from upstream for {}".format(bag_id))
    if not _mapping_equal(
        rebuilt.fixed_r_provenance,
        upstream_input.fixed_r_provenance,
    ):
        raise ValueError("rebuilt R provenance differs from upstream for {}".format(bag_id))
    if not np.array_equal(
        prepared.problem.observations.times, upstream_expectation.times
    ):
        raise ValueError("rebuilt time grid differs from upstream for {}".format(bag_id))
    if prepared.calibration.correlation_time != upstream_expectation.correlation_time:
        raise ValueError("rebuilt OU correlation time differs for {}".format(bag_id))
    if not np.array_equal(
        prepared.calibration.stationary_standard_deviation,
        upstream_pilot.stationary_standard_deviation,
    ):
        raise ValueError("rebuilt Q pilot differs from upstream for {}".format(bag_id))


def _stage2_model_provenance(
    upstream_input: Any,
    upstream_bundle: Any,
    settings: Mapping[str, Any],
    effective_workers: int,
) -> Mapping[str, Any]:
    return {
        "schema": "grape-param-estim/augmented-parameter-model/v1",
        "algorithm_version": AUGMENTED_PARAMETER_ALGORITHM_VERSION,
        "upstream_diagonal_q_artifact_fingerprint": request_fingerprint(
            upstream_bundle.manifest
        ),
        "upstream_fixed_model": upstream_input.fixed_model_provenance,
        "upstream_fixed_model_fingerprint": (
            upstream_input.fixed_model_fingerprint
        ),
        "upstream_fixed_r_provenance": upstream_input.fixed_r_provenance,
        "upstream_fixed_r_fingerprint": (
            upstream_input.observation_covariance_fingerprint
        ),
        "upstream_fixed_initial_delay_seconds": (
            upstream_input.constant_delay_seconds
        ),
        "fixed_q_stationary_variance": (
            upstream_bundle.covariance.stationary_variance.tolist()
        ),
        "stage_settings": {
            key: value for key, value in settings.items()
            if key != "forecast_workers"
        },
        "execution": {"forecast_workers": int(effective_workers)},
    }


def _implementation_provenance(effective_workers: int) -> Mapping[str, Any]:
    revision = os.environ.get("GRAPE_PARAM_ESTIM_REVISION", "workspace")
    if not revision or "\x00" in revision:
        raise ValueError("GRAPE_PARAM_ESTIM_REVISION is invalid")
    dirty = os.environ.get("GRAPE_PARAM_ESTIM_SOURCE_DIRTY")
    if dirty is None:
        source_dirty = True
    elif dirty.strip().lower() in {"1", "true", "yes", "dirty"}:
        source_dirty = True
    elif dirty.strip().lower() in {"0", "false", "no", "clean"}:
        source_dirty = False
    else:
        raise ValueError("GRAPE_PARAM_ESTIM_SOURCE_DIRTY is invalid")
    return {
        "algorithm_version": AUGMENTED_PARAMETER_ALGORITHM_VERSION,
        "source_revision": revision,
        "source_dirty": source_dirty,
        "forecast_workers": int(effective_workers),
        "multiprocessing_start_method": (
            "spawn" if effective_workers > 1 else None
        ),
    }


class _ProgressBridge:
    def __init__(
        self,
        tracker: ProgressTracker,
        preparation_units: int,
        compute_units: int,
    ) -> None:
        self.tracker = tracker
        self.preparation_units = int(preparation_units)
        self.compute_units = int(compute_units)
        self.completed = self.preparation_units
        self.last_emitted = self.preparation_units
        self.last_stage = ""
        self.stride = max(1, tracker.total_units // 1000)

    def __call__(self, event: ProgressEvent) -> None:
        if event.total_units != self.compute_units:
            raise RuntimeError("augmented-parameter progress total changed")
        completed = self.preparation_units + event.completed_units
        if completed < self.completed:
            raise RuntimeError("augmented-parameter progress moved backwards")
        self.completed = completed
        stage_changed = event.stage_id != self.last_stage
        if not stage_changed and completed - self.last_emitted < self.stride:
            return
        self.tracker.emit(
            completed,
            event.stage_id,
            event.stage_label,
            bag_id=event.bag_id,
            member_id=event.member_id,
            message=event.message,
        )
        self.last_emitted = completed
        self.last_stage = event.stage_id


def _filter_work_units(member_count: int, boundary_count: int) -> int:
    return boundary_count + (member_count + 1) * (boundary_count - 1)


def _cancel_writing_artifact_if_present(output: Path, reason: str) -> bool:
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return False
    value = read_augmented_parameter_manifest(output)
    if value["status"] == COMPLETE_STATUS:
        return False
    if value["status"] == WRITING_STATUS:
        mark_augmented_parameter_artifact_cancelled(output, reason)
    return value["status"] in {WRITING_STATUS, CANCELLED_STATUS}


def run_request(request_path: str, output_path: str) -> Path:
    request_source = Path(request_path).expanduser().resolve()
    request = validate_augmented_parameter_stage_request(
        read_json(request_source)
    )
    output = Path(output_path).expanduser().resolve()
    settings = request["settings"]
    ensemble_size = int(settings["ensemble_size"])
    workers = resolved_forecast_workers(
        settings["forecast_workers"], ensemble_size
    )
    upstream_root = Path(
        request["upstream_diagonal_q"]["path"]
    ).expanduser().resolve()
    expected_upstream_fingerprint = request["upstream_diagonal_q"][
        "artifact_fingerprint"
    ]
    actual_upstream_fingerprint = diagonal_q_artifact_fingerprint(
        upstream_root
    )
    if actual_upstream_fingerprint != expected_upstream_fingerprint:
        raise ValueError("upstream diagonal-Q artifact fingerprint changed")
    upstream = load_diagonal_q_artifact(upstream_root)
    if request_fingerprint(upstream.manifest) != actual_upstream_fingerprint:
        raise ValueError("upstream diagonal-Q artifact changed while loading")
    _validate_request_against_upstream(request, upstream)

    cancellation = CancellationToken()

    def request_cancel(signum: int, _frame: Any) -> None:
        cancellation.cancel("signal_{}".format(signum))

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)

    try:
        upstream_input_by_id = {
            value.bag_id: value for value in upstream.bag_inputs
        }
        upstream_expectation_by_id = {
            value.bag_id: value for value in upstream.expectations
        }
        upstream_pilot_by_id = {
            value.bag_id: value for value in upstream.pilots
        }
        prepared_bags = []
        artifact_inputs = []
        for bag_request in request["bags"]:
            cancellation.raise_if_cancelled()
            bag_id = str(bag_request["bag_id"])
            q_input = upstream_input_by_id[bag_id]
            path = Path(str(bag_request["path"])).expanduser().resolve()
            arrays = read_grape_rosbag_arrays(
                str(path),
                compute_sha256=True,
                checkpoint=cancellation.raise_if_cancelled,
            )
            if arrays.bag_sha256 != bag_request["sha256"]:
                raise ValueError("bag SHA256 changed: {}".format(bag_id))
            if int(arrays.bag_size_bytes) != int(path.stat().st_size):
                raise ValueError("bag size changed: {}".format(bag_id))
            interval = bag_request["selected_interval_local_seconds"]
            episode = build_real_flight_episode(
                arrays,
                sample_period=float(settings["sample_period"]),
                episode_index=int(bag_request["episode_index"]),
                start_local=float(interval[0]),
                end_local=float(interval[1]),
                window_state=None,
                controller_source_revision=os.environ.get(
                    "GRAPE_PARAM_ESTIM_REVISION", "workspace"
                ),
            )
            diagonal_q_bag = prepare_real_diagonal_q_bag(
                bag_id,
                episode,
                str(bag_request["configuration_fingerprint"]),
                initial_delay=q_input.constant_delay_seconds,
            )
            rebuilt = _stage1_artifact_bag_input(
                bag_request,
                arrays,
                episode,
                diagonal_q_bag,
                q_input.constant_delay_seconds,
            )
            _validate_rebuilt_bag_against_upstream(
                bag_id,
                rebuilt,
                diagonal_q_bag,
                q_input,
                upstream_expectation_by_id[bag_id],
                upstream_pilot_by_id[bag_id],
            )
            prepared_bags.append(
                PreparedAugmentedParameterBag.from_diagonal_q_bag(
                    diagonal_q_bag
                )
            )
            artifact_inputs.append(
                AugmentedParameterArtifactBagInput(
                    bag_id=bag_id,
                    episode_index=int(bag_request["episode_index"]),
                    episode=episode,
                    problem=diagonal_q_bag.problem,
                    nominal_trajectory=(
                        diagonal_q_bag.problem.nominal_trajectory
                    ),
                    configuration_fingerprint=str(
                        bag_request["configuration_fingerprint"]
                    ),
                    model_provenance=_stage2_model_provenance(
                        q_input, upstream, settings, workers
                    ),
                )
            )

        delay_chart = BoundedDelayChart(
            float(settings["maximum_delay_seconds"])
        )
        prior = AugmentedParameterPrior.grape(
            delay_mean=float(settings["delay_prior_mean_seconds"]),
            delay_standard_deviation=float(
                settings["delay_prior_standard_deviation_seconds"]
            ),
            maximum_delay=delay_chart.maximum_delay,
        )
        preparation_units = 2 + 6 * len(prepared_bags)
        compute_units = sum(
            _filter_work_units(
                ensemble_size, int(value.problem.observations.times.size)
            )
            for value in prepared_bags
        )
        total_units = preparation_units + compute_units + 2
        tracker = ProgressTracker(
            run_id=str(request["run_id"]),
            total_units=total_units,
            callback=JsonlProgressWriter(sys.stdout),
            eta_calibration_units=min(16, max(2, total_units // 50)),
        )
        tracker.emit(
            preparation_units,
            "augmented_parameter_preparation",
            "Augmented-parameter inputs prepared",
            message="verified upstream Q and prepared {} bags".format(
                len(prepared_bags)
            ),
        )
        bridge = _ProgressBridge(
            tracker, preparation_units, compute_units
        )
        result = run_multi_bag_augmented_parameter_filter(
            prepared_bags,
            upstream.covariance,
            ensemble_size=ensemble_size,
            seed=int(settings["seed"]),
            prior=prior,
            delay_chart=delay_chart,
            covariance_rcond=float(settings["covariance_rcond"]),
            forecast_workers=workers,
            progress_callback=bridge,
            cancellation_token=cancellation,
            run_id=str(request["run_id"]),
        )
        cancellation.raise_if_cancelled()
        tracker.emit(
            total_units - 1,
            "augmented_parameter_artifact_writing",
            "Writing augmented-parameter artifact",
            message="publishing audited fixed-Q stage result",
        )
        write_augmented_parameter_artifact(
            output,
            run_id=str(request["run_id"]),
            stage_id=str(request["stage_id"]),
            request_fingerprint=request_fingerprint(request),
            project_fingerprint=str(request["project_fingerprint"]),
            stage_input_fingerprint=str(request["stage_input_fingerprint"]),
            implementation_provenance=_implementation_provenance(workers),
            upstream_diagonal_q_path=upstream_root,
            upstream_diagonal_q_fingerprint=actual_upstream_fingerprint,
            bag_inputs=artifact_inputs,
            result=result,
        )
        tracker.emit(
            total_units,
            "complete",
            "Augmented-parameter stage complete",
            message="augmented-parameter artifact is complete",
        )
        return output
    except ProgressCancelled as error:
        try:
            _cancel_writing_artifact_if_present(output, error.reason)
        except (ArtifactValidationError, OSError) as marker_error:
            print(
                "could not mark augmented-parameter artifact cancelled: {}".format(
                    marker_error
                ),
                file=sys.stderr,
            )
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate shared Grape parameters with fixed diagonal Q."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    selected = parser.parse_args(arguments)
    try:
        run_request(selected.request, selected.output)
    except ProgressCancelled as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "AUGMENTED_PARAMETER_ALGORITHM_VERSION",
    "AUGMENTED_PARAMETER_STAGE_ID",
    "AUGMENTED_PARAMETER_STAGE_REQUEST_SCHEMA",
    "resolved_forecast_workers",
    "run_request",
    "validate_augmented_parameter_stage_request",
]


if __name__ == "__main__":
    main()
