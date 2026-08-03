"""Strict request-file worker for the real-flight diagonal-Q stage.

The worker keeps stdout reserved for monotonic progress JSON Lines.  All
scientific inputs are authenticated before an episode is built, and the only
published result is the audited diagonal-Q stage-boundary artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import re
import signal
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    COMPLETE_STATUS,
    WRITING_STATUS,
    read_json,
    request_fingerprint,
)
from grape_param_estim.diagonal_q import BODY_WRENCH_FRAME
from grape_param_estim.diagonal_q_artifact import (
    DiagonalQArtifactBagInput,
    mark_diagonal_q_artifact_cancelled,
    read_diagonal_q_manifest,
    write_diagonal_q_artifact,
)
from grape_param_estim.diagonal_q_em import DiagonalQEmConfig
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressEvent,
    ProgressTracker,
)
from grape_param_estim.real_diagonal_q_estimation import (
    Q_ONLY_MINIMUM_MEMBER_COUNT,
    PreparedDiagonalQBag,
    prepare_real_diagonal_q_bag,
    run_real_diagonal_q_em,
)
from grape_param_estim.real_rosbag import (
    RealFlightEpisode,
    build_real_flight_episode,
    read_grape_rosbag_arrays,
)


DIAGONAL_Q_STAGE_REQUEST_SCHEMA = (
    "grape-param-estim/diagonal-q-stage-request/v1"
)
DIAGONAL_Q_STAGE_ID = "diagonal_q"
DIAGONAL_Q_ALGORITHM_VERSION = "diagonal-q-em-v1"

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
    "bags",
    "settings",
}
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
    "maximum_em_iterations",
    "log_q_tolerance",
    "component_floor",
    "fixed_initial_delay_seconds",
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


def _finite_float(
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
    if (
        not np.isfinite(selected)
        or (positive and selected <= 0.0)
        or (nonnegative and selected < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError("{} must be finite and {}".format(label, qualifier))
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
        if maximum is None:
            bounds = "at least {}".format(minimum)
        else:
            bounds = "in [{}, {}]".format(minimum, maximum)
        raise ValueError("{} must be {}".format(label, bounds))
    return selected


def validate_diagonal_q_stage_request(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the complete, versioned GUI-to-worker request contract."""

    request = _exact_keys(value, _TOP_LEVEL_KEYS, "diagonal-Q request")
    if request["schema"] != DIAGONAL_Q_STAGE_REQUEST_SCHEMA:
        raise ValueError("unsupported diagonal-Q stage request schema")
    _safe_id(request["run_id"], "run_id")
    _fingerprint(request["project_fingerprint"], "project_fingerprint")
    if request["stage_id"] != DIAGONAL_Q_STAGE_ID:
        raise ValueError("stage_id must be {!r}".format(DIAGONAL_Q_STAGE_ID))
    _fingerprint(
        request["stage_input_fingerprint"], "stage_input_fingerprint"
    )

    bags = request["bags"]
    if not isinstance(bags, list) or not bags:
        raise ValueError("bags must be a non-empty list")
    identifiers = []
    configurations = set()
    for index, raw_bag in enumerate(bags):
        label = "bags[{}]".format(index)
        bag = _exact_keys(raw_bag, _BAG_KEYS, label)
        identifier = _safe_id(bag["bag_id"], "{}.bag_id".format(label))
        identifiers.append(identifier)
        path_text = bag["path"]
        if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
            raise ValueError("{}.path must be a non-empty path".format(label))
        try:
            path = Path(path_text).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                "{}.path cannot be resolved: {}".format(label, error)
            ) from error
        if not path.is_file():
            raise ValueError("{}.path must name a regular file".format(label))
        digest = bag["sha256"]
        if not isinstance(digest, str) or _RAW_SHA256.fullmatch(digest) is None:
            raise ValueError("{}.sha256 must be lowercase hexadecimal".format(label))
        _integer(bag["episode_index"], "{}.episode_index".format(label))
        interval = bag["selected_interval_local_seconds"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(
                "{}.selected_interval_local_seconds must contain two bounds".format(
                    label
                )
            )
        start = _finite_float(
            interval[0], "{}.selected interval start".format(label)
        )
        end = _finite_float(
            interval[1], "{}.selected interval end".format(label)
        )
        if end <= start or not np.isfinite(end - start):
            raise ValueError("{}.selected interval must be increasing".format(label))
        configuration = bag["configuration_fingerprint"]
        if (
            not isinstance(configuration, str)
            or _CONFIGURATION_FINGERPRINT.fullmatch(configuration) is None
        ):
            raise ValueError(
                "{}.configuration_fingerprint must be a complete, incomplete, "
                "or manual-group SHA256".format(label)
            )
        configurations.add(configuration)
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("bags must have sorted, unique bag IDs")
    if len(configurations) != 1:
        raise ValueError("selected bags must share one configuration fingerprint")

    settings = _exact_keys(request["settings"], _SETTINGS_KEYS, "settings")
    _finite_float(settings["sample_period"], "sample_period", positive=True)
    _integer(
        settings["ensemble_size"],
        "ensemble_size",
        minimum=Q_ONLY_MINIMUM_MEMBER_COUNT,
    )
    _integer(
        settings["maximum_em_iterations"],
        "maximum_em_iterations",
        minimum=1,
    )
    _finite_float(
        settings["log_q_tolerance"], "log_q_tolerance", positive=True
    )
    component_floor = settings["component_floor"]
    if not isinstance(component_floor, list) or len(component_floor) != 6:
        raise ValueError("component_floor must contain exactly six values")
    for index, component in enumerate(component_floor):
        _finite_float(
            component,
            "component_floor[{}]".format(index),
            positive=True,
        )
    _finite_float(
        settings["fixed_initial_delay_seconds"],
        "fixed_initial_delay_seconds",
        nonnegative=True,
    )
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
    """Resolve ``auto`` without exceeding useful ensemble parallelism."""

    members = _integer(
        ensemble_size,
        "ensemble_size",
        minimum=Q_ONLY_MINIMUM_MEMBER_COUNT,
    )
    if value == "auto":
        return min(members, 32, max(1, _available_cpu_count() // 2))
    requested = _integer(value, "forecast_workers", minimum=1, maximum=256)
    return min(requested, members)


def _json_value(value: Any) -> Any:
    """Convert immutable model dataclasses to finite JSON provenance."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError("model provenance contains a non-finite number")
        return selected
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        if np.any(~np.isfinite(value)):
            raise ValueError("model provenance contains a non-finite array")
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("model provenance contains a non-string key")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ValueError(
        "model provenance cannot encode {}".format(type(value).__name__)
    )


def _fixed_model_provenance(
    prepared: PreparedDiagonalQBag,
    episode: RealFlightEpisode,
) -> Mapping[str, Any]:
    problem = prepared.problem
    calibration = prepared.calibration
    nominal_parameters = problem.parameter_chart.decode(
        np.zeros(PARAMETER_DIMENSION, dtype=float)
    )
    return {
        "schema": "grape-param-estim/fixed-diagonal-q-model/v1",
        "plant": {
            "implementation": "FullSixDofPlant",
            "parameter_chart_coordinates": [0.0] * PARAMETER_DIMENSION,
            "vehicle_parameters": _json_value(nominal_parameters),
        },
        "controller": {
            "implementation": "GrapeController",
            "configuration": _json_value(problem.controller_configuration),
            "allocation_vehicle_parameters": _json_value(
                problem.controller_parameters
            ),
        },
        "geometry": _json_value(problem.geometry),
        "actuator": _json_value(problem.actuator_parameters),
        "articulated_model": "GrapeArticulatedModel/default/v1",
        "residual_wrench": {
            "frame": BODY_WRENCH_FRAME,
            "process": "stationary-diagonal-ornstein-uhlenbeck/v1",
            "correlation_time_seconds": float(calibration.correlation_time),
            "pilot_method": str(calibration.method),
            "pilot_derivative_window_samples": int(
                calibration.derivative_window_samples
            ),
            "pilot_valid_sample_count": int(
                np.count_nonzero(calibration.valid_mask)
            ),
            "pilot_location": calibration.pilot_location.tolist(),
        },
        "episode_anchor": {
            "controller_state": _json_value(episode.initial_controller_state),
            "actuator_state": _json_value(episode.initial_actuator_state),
        },
    }


def _fixed_r_provenance(
    episode: RealFlightEpisode,
    arrays: Any,
) -> Mapping[str, Any]:
    provenance = episode.provenance
    local_origin = float(arrays.bag_record_start)
    return {
        "schema": "grape-param-estim/fixed-pose-r/v1",
        "method": "robust_preflight_static_pose_covariance/v1",
        "bag_record_interval_local_seconds": [
            0.0,
            float(arrays.bag_record_end) - local_origin,
        ],
        "source_available_interval_local_seconds": [
            float(provenance.source_available_start) - local_origin,
            float(provenance.source_available_end) - local_origin,
        ],
        "requested_interval_local_seconds": [
            float(provenance.requested_window_start) - local_origin,
            float(provenance.requested_window_end) - local_origin,
        ],
        "effective_interval_local_seconds": [
            float(episode.window_start_local_time),
            float(episode.window_end_local_time),
        ],
        "static_window_record_seconds": [
            float(provenance.static_window_start),
            float(provenance.static_window_end),
        ],
        "translation_samples": int(provenance.static_position_samples),
        "translation_inliers": int(provenance.static_position_inliers),
        "rotation_samples": int(provenance.static_orientation_samples),
        "rotation_inliers": int(provenance.static_orientation_inliers),
        "outlier_threshold": float(provenance.covariance_outlier_threshold),
        "eigenvalue_floor": float(provenance.covariance_eigenvalue_floor),
    }


def _artifact_bag_input(
    bag_request: Mapping[str, Any],
    arrays: Any,
    episode: RealFlightEpisode,
    prepared: PreparedDiagonalQBag,
    fixed_delay: float,
) -> DiagonalQArtifactBagInput:
    observations = prepared.problem.observations
    return DiagonalQArtifactBagInput(
        bag_id=str(bag_request["bag_id"]),
        source_path=str(arrays.bag_path),
        source_sha256=str(arrays.bag_sha256),
        source_size_bytes=int(arrays.bag_size_bytes),
        selected_interval_local_seconds=tuple(
            float(value)
            for value in bag_request["selected_interval_local_seconds"]
        ),
        effective_interval_local_seconds=(
            float(episode.window_start_local_time),
            float(episode.window_end_local_time),
        ),
        episode_index=int(bag_request["episode_index"]),
        configuration_fingerprint=str(
            bag_request["configuration_fingerprint"]
        ),
        fixed_model_provenance=_fixed_model_provenance(prepared, episode),
        constant_delay_seconds=float(fixed_delay),
        translation_covariance=observations.translation_covariance,
        rotation_covariance=observations.rotation_covariance,
        fixed_r_provenance=_fixed_r_provenance(episode, arrays),
    )


def _implementation_provenance() -> Mapping[str, Any]:
    revision = os.environ.get("GRAPE_PARAM_ESTIM_REVISION", "workspace")
    if not revision or "\x00" in revision:
        raise ValueError("GRAPE_PARAM_ESTIM_REVISION is invalid")
    dirty_text = os.environ.get("GRAPE_PARAM_ESTIM_SOURCE_DIRTY")
    if dirty_text is None:
        source_dirty = True
    elif dirty_text.strip().lower() in {"1", "true", "yes", "dirty"}:
        source_dirty = True
    elif dirty_text.strip().lower() in {"0", "false", "no", "clean"}:
        source_dirty = False
    else:
        raise ValueError("GRAPE_PARAM_ESTIM_SOURCE_DIRTY is invalid")
    return {
        "algorithm_version": DIAGONAL_Q_ALGORITHM_VERSION,
        "source_revision": revision,
        "source_dirty": source_dirty,
    }


class _MonotonicProgressBridge:
    """Map nested EM and per-bag trackers onto one immutable total."""

    def __init__(
        self,
        *,
        tracker: ProgressTracker,
        preparation_units: int,
        maximum_iterations: int,
        bag_units: Mapping[str, int],
    ) -> None:
        self.tracker = tracker
        self.preparation_units = int(preparation_units)
        self.maximum_iterations = int(maximum_iterations)
        self.bag_ids = tuple(sorted(bag_units))
        self.bag_units = {key: int(bag_units[key]) for key in self.bag_ids}
        self.cycle_units = sum(self.bag_units.values())
        self.prefix = {}
        running = 0
        for bag_id in self.bag_ids:
            self.prefix[bag_id] = running
            running += self.bag_units[bag_id]
        self.completed = self.preparation_units
        self._last_emitted = self.preparation_units
        self._last_stage = ""
        self._stride = max(1, tracker.total_units // 1000)

    def _emit(
        self,
        completed: int,
        stage_id: str,
        stage_label: str,
        *,
        force: bool = False,
        iteration: int | None = None,
        bag_id: str | None = None,
        member_id: int | None = None,
        message: str = "",
    ) -> None:
        selected = int(completed)
        if selected < self.completed:
            raise RuntimeError("nested diagonal-Q progress moved backwards")
        self.completed = selected
        stage_changed = stage_id != self._last_stage
        if not force and not stage_changed and (
            selected - self._last_emitted < self._stride
        ):
            return
        maximum = self.maximum_iterations if iteration is not None else None
        self.tracker.emit(
            selected,
            stage_id,
            stage_label,
            iteration=iteration,
            maximum_iterations=maximum,
            bag_id=bag_id,
            member_id=member_id,
            message=message,
        )
        self._last_emitted = selected
        self._last_stage = stage_id

    def bag_event(self, iteration: int, bag_id: str, event: ProgressEvent) -> None:
        if bag_id not in self.bag_units:
            raise RuntimeError("unknown bag in diagonal-Q progress")
        if not 1 <= int(iteration) <= self.maximum_iterations + 1:
            raise RuntimeError("diagonal-Q E-step index exceeds its reserve")
        expected = self.bag_units[bag_id]
        if event.total_units != expected or event.completed_units > expected:
            raise RuntimeError("diagonal-Q bag progress total changed")
        completed = (
            self.preparation_units
            + (int(iteration) - 1) * self.cycle_units
            + self.prefix[bag_id]
            + event.completed_units
        )
        displayed_iteration = (
            int(iteration)
            if int(iteration) <= self.maximum_iterations
            else None
        )
        self._emit(
            completed,
            event.stage_id,
            event.stage_label,
            iteration=displayed_iteration,
            bag_id=bag_id,
            member_id=event.member_id,
            message=event.message,
        )

    def em_event(self, event: ProgressEvent) -> None:
        self._emit(
            self.completed,
            event.stage_id,
            event.stage_label,
            force=True,
            iteration=event.iteration,
            message=event.message,
        )


def _q_e_step_units(boundary_count: int, ensemble_size: int) -> int:
    boundaries = _integer(boundary_count, "boundary_count", minimum=2)
    members = _integer(
        ensemble_size,
        "ensemble_size",
        minimum=Q_ONLY_MINIMUM_MEMBER_COUNT,
    )
    return boundaries + (members + 1) * (boundaries - 1)


def _cancel_writing_artifact_if_present(
    output: Path, reason: str
) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_diagonal_q_manifest(output)
    if manifest["status"] == COMPLETE_STATUS:
        return False
    if manifest["status"] == WRITING_STATUS:
        mark_diagonal_q_artifact_cancelled(output, reason)
    return True


def run_request(request_path: str, output_path: str) -> Path:
    """Run one authenticated request and publish one complete Q artifact."""

    request_source = Path(request_path).expanduser().resolve()
    request = validate_diagonal_q_stage_request(read_json(request_source))
    output = Path(output_path).expanduser().resolve()
    settings = request["settings"]
    ensemble_size = int(settings["ensemble_size"])
    maximum_iterations = int(settings["maximum_em_iterations"])
    fixed_delay = float(settings["fixed_initial_delay_seconds"])
    workers = resolved_forecast_workers(
        settings["forecast_workers"], ensemble_size
    )
    configuration = DiagonalQEmConfig(
        maximum_iterations=maximum_iterations,
        log_q_tolerance=float(settings["log_q_tolerance"]),
        component_floor=np.asarray(settings["component_floor"], dtype=float),
    )
    cancellation = CancellationToken()

    def request_cancel(signum: int, _frame: Any) -> None:
        cancellation.cancel("signal_{}".format(signum))

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)

    try:
        prepared_bags = []
        artifact_inputs = []
        for bag_request in request["bags"]:
            cancellation.raise_if_cancelled()
            path = Path(str(bag_request["path"])).expanduser().resolve()
            arrays = read_grape_rosbag_arrays(
                str(path),
                compute_sha256=True,
                checkpoint=cancellation.raise_if_cancelled,
            )
            cancellation.raise_if_cancelled()
            if arrays.bag_sha256 != bag_request["sha256"]:
                raise ValueError(
                    "bag SHA256 changed: {}".format(bag_request["bag_id"])
                )
            if int(arrays.bag_size_bytes) != int(path.stat().st_size):
                raise ValueError(
                    "bag size changed: {}".format(bag_request["bag_id"])
                )
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
            cancellation.raise_if_cancelled()
            prepared = prepare_real_diagonal_q_bag(
                str(bag_request["bag_id"]),
                episode,
                str(bag_request["configuration_fingerprint"]),
                initial_delay=fixed_delay,
            )
            prepared_bags.append(prepared)
            artifact_inputs.append(
                _artifact_bag_input(
                    bag_request, arrays, episode, prepared, fixed_delay
                )
            )

        preparation_units = 1 + 4 * len(prepared_bags)
        bag_units = {
            bag.bag_id: _q_e_step_units(
                bag.problem.observations.times.size, ensemble_size
            )
            for bag in prepared_bags
        }
        reserved_compute_units = (maximum_iterations + 1) * sum(
            bag_units.values()
        )
        total_units = preparation_units + reserved_compute_units + 2
        tracker = ProgressTracker(
            run_id=str(request["run_id"]),
            total_units=total_units,
            callback=JsonlProgressWriter(sys.stdout),
            eta_calibration_units=min(16, max(2, total_units // 50)),
        )
        tracker.emit(
            preparation_units,
            "diagonal_q_preparation",
            "Diagonal Q inputs prepared",
            message="verified and prepared {} selected bags".format(
                len(prepared_bags)
            ),
        )
        bridge = _MonotonicProgressBridge(
            tracker=tracker,
            preparation_units=preparation_units,
            maximum_iterations=maximum_iterations,
            bag_units=bag_units,
        )
        result = run_real_diagonal_q_em(
            prepared_bags,
            configuration,
            ensemble_size=ensemble_size,
            forecast_workers=workers,
            seed=int(settings["seed"]),
            progress_callback=bridge.em_event,
            bag_progress_callback=bridge.bag_event,
            cancellation_token=cancellation,
            run_id=str(request["run_id"]),
        )
        cancellation.raise_if_cancelled()
        tracker.emit(
            total_units - 1,
            "diagonal_q_artifact_writing",
            "Writing diagonal Q artifact",
            message="publishing audited stage-boundary result",
        )
        write_diagonal_q_artifact(
            output,
            run_id=str(request["run_id"]),
            stage_id=str(request["stage_id"]),
            request_fingerprint=request_fingerprint(request),
            project_fingerprint=str(request["project_fingerprint"]),
            stage_input_fingerprint=str(request["stage_input_fingerprint"]),
            implementation_provenance=_implementation_provenance(),
            bag_inputs=artifact_inputs,
            result=result.em_result,
            expectations=result.em_result.final_expectations,
        )
        # Artifact publication is the commit point.  A signal delivered during
        # its bounded atomic write does not retroactively cancel a complete
        # scientific result.
        tracker.emit(
            total_units,
            "complete",
            "Diagonal Q stage complete",
            message="diagonal Q artifact is complete",
        )
        return output
    except ProgressCancelled as error:
        try:
            _cancel_writing_artifact_if_present(output, error.reason)
        except (ArtifactValidationError, OSError) as marker_error:
            print(
                "could not mark diagonal-Q artifact cancelled: {}".format(
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
        description="Estimate a shared diagonal residual-wrench Q from flights."
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
    "DIAGONAL_Q_ALGORITHM_VERSION",
    "DIAGONAL_Q_STAGE_ID",
    "DIAGONAL_Q_STAGE_REQUEST_SCHEMA",
    "resolved_forecast_workers",
    "run_request",
    "validate_diagonal_q_stage_request",
]


if __name__ == "__main__":
    main()
