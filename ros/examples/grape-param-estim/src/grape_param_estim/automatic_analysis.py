"""Automatic multi-bag analysis with explicit fit and failure masks."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from grape_param_estim.controller_advice import (
    build_controller_advice,
)
from grape_param_estim.effective_estimator import (
    EstimatorSettings,
    canonical_sha256,
    effective_parameter_trace,
    estimate_effective_parameters,
    evaluate_effective_parameters,
    prepared_timestamps,
)
from grape_param_estim.episode_detection import (
    DetectedEpisode,
    EpisodeDetectionSettings,
    command_valid_mask,
    detect_control_episodes,
    flight_state_at,
    mask_to_intervals,
)
from grape_param_estim.failure_bag import (
    FailureBagRecording,
    read_failure_recording,
)


CONFIG_SCHEMA = "grape_failure_automatic_analysis/v2"
RESULT_SCHEMA = "grape_failure_automatic_result/v2"


@dataclass(frozen=True)
class AutomaticAnalysisConfig:
    raw: Mapping[str, Any]
    sha256: str
    topics: Mapping[str, str]
    detection: EpisodeDetectionSettings
    estimation: Mapping[str, Any]
    parameter_trace_step_s: float
    controller_topics: Mapping[str, str] = field(default_factory=dict)


def _progress_slice(callback, start: float, end: float):
    if callback is None:
        return None

    def report(fraction, phase):
        value = float(np.clip(fraction, 0.0, 1.0))
        callback(start + (end - start) * value, phase)

    return report


def load_automatic_config(path) -> AutomaticAnalysisConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or values.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported automatic analysis config schema")
    topics = values.get("topics")
    required_topics = {
        "command",
        "gimbal",
        "imu",
        "odometry",
        "flight_state",
    }
    if not isinstance(topics, dict) or set(topics) != required_topics:
        raise ValueError(
            "automatic config must define exactly five input topics"
        )
    normalized_topics = {
        str(name): str(topic) for name, topic in topics.items()
    }
    controller_topics = values.get("controller_topics")
    required_controller_topics = {
        "pid_debug",
        "xy",
        "z",
        "roll_pitch",
        "yaw",
    }
    if (
        not isinstance(controller_topics, dict)
        or set(controller_topics) != required_controller_topics
    ):
        raise ValueError(
            "automatic config must define PID debug and four gain topics"
        )
    normalized_controller_topics = {
        str(name): str(topic)
        for name, topic in controller_topics.items()
    }
    estimation = values.get("estimation")
    if not isinstance(estimation, dict):
        raise ValueError("estimation config must be a mapping")
    required_estimation = {
        "sample_rate_hz",
        "smoothing_window_s",
        "maximum_delay_s",
        "delay_step_s",
        "bootstrap_samples",
        "bootstrap_block_s",
        "huber_delta",
        "ridge",
        "seed",
        "minimum_input_std",
        "minimum_r2",
        "maximum_relative_interval_width",
    }
    if set(estimation) != required_estimation:
        raise ValueError("automatic estimation settings are incomplete")
    trace_step = float(values["presentation"]["parameter_trace_step_s"])
    if not np.isfinite(trace_step) or trace_step <= 0.0:
        raise ValueError("parameter_trace_step_s must be positive")
    detection = EpisodeDetectionSettings.from_mapping(
        values["episode_detection"]
    )
    # Exercise the estimator's validation without imposing a real interval.
    EstimatorSettings.from_mapping(
        {
            **estimation,
            "start_offset_s": 0.0,
            "end_offset_s": max(
                1.0, detection.minimum_airborne_duration_s
            ),
        }
    )
    return AutomaticAnalysisConfig(
        raw=values,
        sha256=canonical_sha256(values),
        topics=normalized_topics,
        detection=detection,
        estimation=dict(estimation),
        parameter_trace_step_s=trace_step,
        controller_topics=normalized_controller_topics,
    )


def _estimator_settings(
    config: AutomaticAnalysisConfig,
    start_s: float,
    end_s: float,
) -> EstimatorSettings:
    return EstimatorSettings.from_mapping(
        {
            **config.estimation,
            "start_offset_s": float(start_s),
            "end_offset_s": float(end_s),
        }
    )


def _persistent_mask(
    timestamps: np.ndarray,
    raw_mask: np.ndarray,
    minimum_duration_s: float,
) -> np.ndarray:
    result = np.zeros(raw_mask.shape, dtype=bool)
    for start_s, end_s in mask_to_intervals(timestamps, raw_mask):
        if end_s - start_s >= minimum_duration_s:
            result |= (timestamps >= start_s) & (timestamps < end_s)
    return result


def _residual_score(
    residual: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_mask, dtype=bool)
    if reference.shape != residual.shape[:1] or not np.any(reference):
        raise ValueError("residual reference mask is empty or invalid")
    reference_residual = residual[reference]
    center = np.median(reference_residual, axis=0)
    absolute = np.abs(reference_residual - center)
    mad_scale = 1.4826 * np.median(absolute, axis=0)
    standard_scale = 0.1 * np.std(reference_residual, axis=0)
    scale = np.maximum(
        np.maximum(mad_scale, standard_scale),
        np.finfo(float).eps,
    )
    normalized = (residual - center) / scale
    return np.sqrt(np.mean(normalized * normalized, axis=1))


def _fit_and_diagnose(
    recording: FailureBagRecording,
    episode: DetectedEpisode,
    config: AutomaticAnalysisConfig,
    progress_callback=None,
):
    if recording.command_times.size < 3:
        raise ValueError("recording has no usable command stream")
    model_start = max(
        episode.liftoff_s,
        float(recording.command_times[0])
        + float(config.estimation["maximum_delay_s"]),
    )
    model_end = min(
        episode.end_s,
        float(recording.command_times[-1]),
    )
    if (
        not np.isfinite(model_start)
        or model_end - model_start
        < config.detection.minimum_airborne_duration_s
    ):
        raise ValueError("command-covered airborne interval is too short")
    settings = _estimator_settings(config, model_start, model_end)
    data = recording.estimator_data(model_start, model_end)
    timestamps = prepared_timestamps(data, settings)

    command_valid = command_valid_mask(
        recording, timestamps, delay_s=0.0
    )
    command_valid &= command_valid_mask(
        recording,
        timestamps,
        delay_s=float(config.estimation["maximum_delay_s"]),
    )
    sampled_flight_state = flight_state_at(recording, timestamps)
    diagnostic_state = np.isin(
        sampled_flight_state,
        config.detection.diagnostic_flight_states,
    )
    minimum_samples = max(
        30,
        int(
            np.ceil(
                config.detection.minimum_airborne_duration_s
                * float(config.estimation["sample_rate_hz"])
            )
        ),
    )
    sampled_height = np.interp(
        timestamps,
        recording.state_times,
        recording.position[:, 2],
    )
    support_clearance = max(
        config.detection.minimum_liftoff_height_m,
        config.detection.standardized_threshold
        * max(episode.support_height_sigma_m, 0.0),
    )
    support_contact = _persistent_mask(
        timestamps,
        sampled_height
        <= episode.support_height_m + support_clearance,
        config.detection.persistence_s,
    )
    preliminary_selected = (
        command_valid & ~diagnostic_state & ~support_contact
    )
    if int(np.sum(preliminary_selected)) < minimum_samples:
        raise ValueError("too few command-valid airborne samples")

    preliminary = estimate_effective_parameters(
        data,
        settings,
        fit_mask=preliminary_selected,
        bootstrap=False,
        progress_callback=_progress_slice(
            progress_callback, 0.0, 0.08
        ),
    )
    preliminary_evaluation = evaluate_effective_parameters(
        data, settings, preliminary
    )
    residual_score = _residual_score(
        preliminary_evaluation["residual"],
        preliminary_selected,
    )
    mismatch = _persistent_mask(
        timestamps,
        residual_score > config.detection.standardized_threshold,
        config.detection.persistence_s,
    )
    selected = preliminary_selected & ~mismatch
    if int(np.sum(selected)) < minimum_samples:
        raise ValueError(
            "model mismatch leaves too few fit samples"
        )

    intermediate = estimate_effective_parameters(
        data,
        settings,
        fit_mask=selected,
        bootstrap=False,
        progress_callback=_progress_slice(
            progress_callback, 0.08, 0.16
        ),
    )
    evaluation = evaluate_effective_parameters(
        data, settings, intermediate
    )
    final_score = _residual_score(
        evaluation["residual"], selected
    )
    final_mismatch = _persistent_mask(
        timestamps,
        final_score > config.detection.standardized_threshold,
        config.detection.persistence_s,
    )
    final_selected = preliminary_selected & ~final_mismatch
    if (
        int(np.sum(final_selected)) >= minimum_samples
        and not np.array_equal(final_selected, selected)
    ):
        selected = final_selected
        mismatch = final_mismatch
    elif np.array_equal(final_selected, selected):
        mismatch = final_mismatch

    estimate = estimate_effective_parameters(
        data,
        settings,
        fit_mask=selected,
        progress_callback=_progress_slice(
            progress_callback, 0.16, 0.70
        ),
    )
    evaluation = evaluate_effective_parameters(
        data, settings, estimate
    )
    final_score = _residual_score(
        evaluation["residual"], selected
    )

    trace_span = float(timestamps[-1] - timestamps[0])
    trace_step = max(
        config.parameter_trace_step_s,
        trace_span / 120.0,
    )
    trace = effective_parameter_trace(
        data,
        settings,
        estimate,
        fit_mask=selected,
        minimum_duration_s=(
            config.detection.minimum_airborne_duration_s
        ),
        step_s=trace_step,
        progress_callback=_progress_slice(
            progress_callback, 0.70, 1.0
        ),
    )
    return {
        "settings": settings,
        "timestamps": timestamps,
        "command_valid": command_valid,
        "flight_state": sampled_flight_state,
        "diagnostic_state_mask": diagnostic_state,
        "support_contact_mask": support_contact,
        "fit_mask": selected,
        "mismatch_mask": mismatch,
        "residual_score": final_score,
        "estimate": estimate,
        "evaluation": evaluation,
        "parameter_trace": trace,
        "parameter_trace_step_s": trace_step,
    }


def _finite_or_none(value: float):
    number = float(value)
    return number if np.isfinite(number) else None


def _interval_rows(
    timestamps: np.ndarray,
    mask: np.ndarray,
    reason: str,
    merge_gap_s: float = 0.0,
) -> list:
    intervals = mask_to_intervals(timestamps, mask)
    merged = []
    for start_s, end_s in intervals:
        if (
            merged
            and start_s - merged[-1][1] <= merge_gap_s
        ):
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    return [
        {"start_s": start_s, "end_s": end_s, "reason": reason}
        for start_s, end_s in merged
    ]


def _downsample_indices(size: int, maximum: int = 2000) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=int)
    return np.unique(
        np.linspace(0, size - 1, maximum).round().astype(int)
    )


def _episode_result(
    recording: FailureBagRecording,
    episode: DetectedEpisode,
    config: AutomaticAnalysisConfig,
    sequence_offset_s: float,
    progress_callback=None,
) -> Mapping[str, Any]:
    base = {
        "episode_index": episode.index,
        "status": episode.status,
        "reason": episode.reason,
        "start_s": episode.start_s,
        "end_s": episode.end_s,
        "sequence_start_s": sequence_offset_s + episode.start_s,
        "sequence_end_s": sequence_offset_s + episode.end_s,
        "flight_states": list(episode.flight_states),
        "support": {
            "height_m": _finite_or_none(episode.support_height_m),
            "height_sigma_m": _finite_or_none(
                episode.support_height_sigma_m
            ),
            "vertical_velocity_sigma_m_s": _finite_or_none(
                episode.support_vertical_velocity_sigma_m_s
            ),
            "sample_count": episode.support_sample_count,
            "source": episode.support_source,
        },
        "liftoff_s": _finite_or_none(episode.liftoff_s),
        "selection": {
            "fit_intervals": [],
            "failure_diagnostic_intervals": [
                {
                    "start_s": episode.start_s,
                    "end_s": (
                        episode.liftoff_s
                        if np.isfinite(episode.liftoff_s)
                        else episode.end_s
                    ),
                    "reason": (
                        "controlled_supported"
                        if np.isfinite(episode.liftoff_s)
                        else episode.reason
                    ),
                }
            ],
        },
        "estimate": None,
        "parameter_trace": [],
        "parameter_trace_step_s": None,
        "model_diagnostics": None,
        "controller_advice": {
            "status": "not_available",
            "reason": "episode was not estimated",
            "groups": [],
        },
    }
    if not episode.identifiable:
        if progress_callback is not None:
            progress_callback(1.0, episode.reason)
        return base

    try:
        fit = _fit_and_diagnose(
            recording,
            episode,
            config,
            progress_callback=_progress_slice(
                progress_callback, 0.0, 0.82
            ),
        )
    except ValueError as error:
        base["status"] = "not_identifiable"
        base["reason"] = str(error)
        base["selection"]["failure_diagnostic_intervals"].append(
            {
                "start_s": episode.liftoff_s,
                "end_s": episode.end_s,
                "reason": "estimation_not_identifiable",
            }
        )
        if progress_callback is not None:
            progress_callback(1.0, "episode not identifiable")
        return base

    timestamps = fit["timestamps"]
    base["status"] = "estimated"
    base["reason"] = "automatic_fit_complete"
    base["selection"]["fit_intervals"] = [
        {"start_s": start_s, "end_s": end_s}
        for start_s, end_s in mask_to_intervals(
            timestamps, fit["fit_mask"]
        )
    ]
    base["selection"]["failure_diagnostic_intervals"].extend(
        _interval_rows(
            timestamps,
            fit["support_contact_mask"],
            "support_contact",
        )
    )
    base["selection"]["failure_diagnostic_intervals"].extend(
        _interval_rows(
            timestamps,
            ~fit["command_valid"],
            "missing_command",
            merge_gap_s=2.0 * config.detection.persistence_s,
        )
    )
    for state in config.detection.diagnostic_flight_states:
        base["selection"]["failure_diagnostic_intervals"].extend(
            _interval_rows(
                timestamps,
                fit["flight_state"] == state,
                "diagnostic_flight_state_{}".format(state),
            )
        )
    base["selection"]["failure_diagnostic_intervals"].extend(
        _interval_rows(
            timestamps,
            fit["mismatch_mask"] & ~fit["diagnostic_state_mask"],
            "persistent_model_mismatch",
        )
    )
    if timestamps[-1] < episode.end_s:
        base["selection"]["failure_diagnostic_intervals"].append(
            {
                "start_s": float(timestamps[-1]),
                "end_s": episode.end_s,
                "reason": "outside_command_covered_model_interval",
            }
        )
    base["estimate"] = fit["estimate"]
    base["controller_advice"] = build_controller_advice(
        recording,
        fit,
        episode.liftoff_s,
        progress_callback=_progress_slice(
            progress_callback, 0.82, 1.0
        ),
    )
    if progress_callback is not None:
        progress_callback(1.0, "episode complete")
    base["parameter_trace_step_s"] = fit[
        "parameter_trace_step_s"
    ]
    base["parameter_trace"] = [
        {
            **row,
            "sequence_time_s": (
                sequence_offset_s + float(row["time_s"])
            ),
        }
        for row in fit["parameter_trace"]
    ]
    selected = _downsample_indices(timestamps.size, maximum=1200)
    base["model_diagnostics"] = {
        "timestamps_s": [
            float(value) for value in timestamps[selected]
        ],
        "residual_score": [
            float(value)
            for value in fit["residual_score"][selected]
        ],
        "fit_mask": [
            bool(value) for value in fit["fit_mask"][selected]
        ],
        "command_valid": [
            bool(value)
            for value in fit["command_valid"][selected]
        ],
        "flight_state": [
            int(value) for value in fit["flight_state"][selected]
        ],
        "standardized_threshold": (
            config.detection.standardized_threshold
        ),
    }
    return base


def _recording_plot_data(
    recording: FailureBagRecording,
) -> Mapping[str, Any]:
    selected = _downsample_indices(recording.state_times.size)
    timestamps = recording.state_times[selected]
    command_indices = np.searchsorted(
        recording.command_times, timestamps, side="right"
    ) - 1
    command_valid = (command_indices >= 0) & command_valid_mask(
        recording, timestamps
    )
    vertical_command = np.full(timestamps.shape, np.nan)
    vertical_command[command_valid] = recording.command_wrench[
        command_indices[command_valid], 2
    ]
    flight_indices = np.searchsorted(
        recording.flight_state_times,
        timestamps,
        side="right",
    ) - 1
    flight_state = np.full(timestamps.shape, -1, dtype=int)
    valid_flight = flight_indices >= 0
    flight_state[valid_flight] = recording.flight_state[
        flight_indices[valid_flight]
    ]
    specific_force = np.column_stack(
        [
            np.interp(
                timestamps,
                recording.imu_times,
                recording.specific_force[:, column],
            )
            for column in range(3)
        ]
    )
    angular_velocity = np.column_stack(
        [
            np.interp(
                timestamps,
                recording.imu_times,
                recording.angular_velocity[:, column],
            )
            for column in range(3)
        ]
    )
    return {
        "time_s": [float(value) for value in timestamps],
        "z_m": [
            float(value)
            for value in recording.position[selected, 2]
        ],
        "speed_m_s": [
            float(value)
            for value in np.linalg.norm(
                recording.linear_velocity[selected], axis=1
            )
        ],
        "specific_force_norm_m_s2": [
            float(value)
            for value in np.linalg.norm(specific_force, axis=1)
        ],
        "angular_velocity_norm_rad_s": [
            float(value)
            for value in np.linalg.norm(angular_velocity, axis=1)
        ],
        "vertical_command": [
            None if not np.isfinite(value) else float(value)
            for value in vertical_command
        ],
        "flight_state": [int(value) for value in flight_state],
    }


def analyze_recordings(
    recordings: Sequence[FailureBagRecording],
    config: AutomaticAnalysisConfig,
    progress_callback=None,
) -> Mapping[str, Any]:
    """Analyze already loaded recordings in the supplied sequence order."""

    if not recordings:
        raise ValueError("at least one recording is required")
    bag_rows = []
    sequence_offset = 0.0
    recording_count = len(recordings)
    for bag_index, recording in enumerate(recordings):
        detected = detect_control_episodes(
            recording, config.detection
        )
        episode_count = max(1, len(detected))
        episodes = []
        for episode_index, episode in enumerate(detected):
            local_callback = _progress_slice(
                progress_callback,
                (
                    bag_index + episode_index / episode_count
                )
                / recording_count,
                (
                    bag_index
                    + (episode_index + 1) / episode_count
                )
                / recording_count,
            )
            episodes.append(
                _episode_result(
                    recording,
                    episode,
                    config,
                    sequence_offset,
                    progress_callback=local_callback,
                )
            )
        if not detected and progress_callback is not None:
            progress_callback(
                (bag_index + 1) / recording_count,
                "no controller-active episode",
            )
        bag_rows.append(
            {
                "bag_index": bag_index,
                "path": recording.bag_path,
                "sha256": recording.bag_sha256,
                "bag_start_time": recording.bag_start_time,
                "duration_s": recording.bag_duration_s,
                "sequence_offset_s": sequence_offset,
                "episode_count": len(episodes),
                "episodes": episodes,
                "plot": _recording_plot_data(recording),
            }
        )
        sequence_offset += recording.bag_duration_s

    result = {
        "schema": RESULT_SCHEMA,
        "config_sha256": config.sha256,
        "interpretation": (
            "Each episode is fitted independently. Sequence time only "
            "orders trials and does not silently pool their parameters."
        ),
        "bag_count": len(bag_rows),
        "sequence_duration_s": sequence_offset,
        "bags": bag_rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def analyze_bags(
    bag_paths: Sequence,
    config: AutomaticAnalysisConfig,
    progress_callback=None,
) -> Mapping[str, Any]:
    paths = list(bag_paths)
    if not paths:
        raise ValueError("at least one bag is required")
    recordings = []
    for index, path in enumerate(paths):
        read_progress = _progress_slice(
            progress_callback,
            0.15 * index / len(paths),
            0.15 * (index + 1) / len(paths),
        )
        recordings.append(
            read_failure_recording(
                path,
                config.topics,
                config.controller_topics,
                progress_callback=read_progress,
            )
        )
    return analyze_recordings(
        recordings,
        config,
        progress_callback=_progress_slice(
            progress_callback, 0.15, 1.0
        ),
    )


def merge_analysis_results(
    existing: Mapping[str, Any],
    addition: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Append independently analyzed bags without reprocessing old bags."""

    for name, result in (
        ("existing", existing),
        ("addition", addition),
    ):
        if result.get("schema") != RESULT_SCHEMA:
            raise ValueError(
                "{} result has an unsupported schema".format(name)
            )
        if not isinstance(result.get("bags"), list):
            raise ValueError(
                "{} result has no bag list".format(name)
            )
    if existing.get("config_sha256") != addition.get("config_sha256"):
        raise ValueError(
            "cannot merge results made with different configurations"
        )
    if existing.get("interpretation") != addition.get("interpretation"):
        raise ValueError(
            "cannot merge results with different interpretations"
        )

    merged = deepcopy(existing)
    sequence_shift = float(existing["sequence_duration_s"])
    bag_index_shift = len(merged["bags"])
    for source_bag in addition["bags"]:
        bag = deepcopy(source_bag)
        bag["bag_index"] = bag_index_shift + int(bag["bag_index"])
        bag["sequence_offset_s"] = (
            sequence_shift + float(bag["sequence_offset_s"])
        )
        for episode in bag["episodes"]:
            episode["sequence_start_s"] = (
                sequence_shift + float(episode["sequence_start_s"])
            )
            episode["sequence_end_s"] = (
                sequence_shift + float(episode["sequence_end_s"])
            )
            for row in episode["parameter_trace"]:
                row["sequence_time_s"] = (
                    sequence_shift + float(row["sequence_time_s"])
                )
        merged["bags"].append(bag)

    merged["bag_count"] = len(merged["bags"])
    merged["sequence_duration_s"] = (
        sequence_shift + float(addition["sequence_duration_s"])
    )
    merged.pop("result_sha256", None)
    merged["result_sha256"] = canonical_sha256(merged)
    return merged


__all__ = [
    "AutomaticAnalysisConfig",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "analyze_bags",
    "analyze_recordings",
    "load_automatic_config",
    "merge_analysis_results",
]
