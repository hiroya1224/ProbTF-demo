"""Robust effective-parameter estimation for one failed-flight interval."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Tuple

import numpy as np
import yaml

from grape_param_estim.failure_bag import (
    FailureBagData,
    read_failure_bag,
)


CONFIG_SCHEMA = "grape_failure_effective_estimator/v1"
RESULT_SCHEMA = "grape_failure_effective_parameters/v1"
MODEL_ID = "diagonal_effective_wrench_response/v1"
GEOMETRY_ID = "grape_source_geometry_sample/v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class EstimatorSettings:
    """Small set of numerical controls for the effective regression."""

    start_offset_s: float
    end_offset_s: float
    sample_rate_hz: float
    smoothing_window_s: float
    maximum_delay_s: float
    delay_step_s: float
    bootstrap_samples: int
    bootstrap_block_s: float
    huber_delta: float
    ridge: float
    seed: int
    minimum_input_std: float
    minimum_r2: float
    maximum_relative_interval_width: float

    def __post_init__(self) -> None:
        finite_positive = (
            self.sample_rate_hz,
            self.smoothing_window_s,
            self.delay_step_s,
            self.bootstrap_block_s,
            self.huber_delta,
            self.ridge,
            self.minimum_input_std,
            self.maximum_relative_interval_width,
        )
        if (
            not 0.0 <= self.start_offset_s < self.end_offset_s
            or not all(
                np.isfinite(value) and value > 0.0
                for value in finite_positive
            )
            or not np.isfinite(self.maximum_delay_s)
            or self.maximum_delay_s < 0.0
            or int(self.bootstrap_samples) < 20
            or not 0.0 <= self.minimum_r2 < 1.0
        ):
            raise ValueError("failure estimator settings are invalid")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        return cls(
            start_offset_s=float(values["start_offset_s"]),
            end_offset_s=float(values["end_offset_s"]),
            sample_rate_hz=float(values["sample_rate_hz"]),
            smoothing_window_s=float(values["smoothing_window_s"]),
            maximum_delay_s=float(values["maximum_delay_s"]),
            delay_step_s=float(values["delay_step_s"]),
            bootstrap_samples=int(values["bootstrap_samples"]),
            bootstrap_block_s=float(values["bootstrap_block_s"]),
            huber_delta=float(values["huber_delta"]),
            ridge=float(values["ridge"]),
            seed=int(values["seed"]),
            minimum_input_std=float(values["minimum_input_std"]),
            minimum_r2=float(values["minimum_r2"]),
            maximum_relative_interval_width=float(
                values["maximum_relative_interval_width"]
            ),
        )


@dataclass(frozen=True)
class LoadedConfig:
    raw: Mapping[str, Any]
    sha256: str
    expected_bag_sha256: str
    topics: Mapping[str, str]
    settings: EstimatorSettings


@dataclass(frozen=True)
class _PreparedSignals:
    timestamps: np.ndarray
    response: np.ndarray
    state: np.ndarray


def load_config(path) -> LoadedConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or values.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported failure estimator config schema")
    expected = str(values.get("expected_bag_sha256", "")).lower()
    if len(expected) != 64 or any(
        item not in "0123456789abcdef" for item in expected
    ):
        raise ValueError("expected_bag_sha256 must be a lowercase SHA-256")
    topics = values.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("config topics must be a mapping")
    normalized_topics = {
        str(name): str(topic) for name, topic in topics.items()
    }
    if set(normalized_topics) != {
        "command",
        "gimbal",
        "imu",
        "odometry",
    }:
        raise ValueError("config must define exactly four input topics")
    settings = EstimatorSettings.from_mapping(values["estimation"])
    return LoadedConfig(
        raw=values,
        sha256=canonical_sha256(values),
        expected_bag_sha256=expected,
        topics=normalized_topics,
        settings=settings,
    )


def _interpolate(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    if query_times[0] < source_times[0] or query_times[-1] > source_times[-1]:
        raise ValueError("input stream does not cover the analysis interval")
    return np.column_stack(
        [
            np.interp(query_times, source_times, source_values[:, column])
            for column in range(source_values.shape[1])
        ]
    )


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.full(width, 1.0 / width, dtype=float)
    return np.column_stack(
        [
            np.convolve(values[:, column], kernel, mode="same")
            for column in range(values.shape[1])
        ]
    )


def _prepare_signals(
    data: FailureBagData,
    settings: EstimatorSettings,
) -> _PreparedSignals:
    step = 1.0 / settings.sample_rate_hz
    timestamps = np.arange(
        settings.start_offset_s,
        settings.end_offset_s + 0.25 * step,
        step,
        dtype=float,
    )
    if timestamps.size < 30:
        raise ValueError("analysis interval is too short")
    force = _interpolate(
        data.imu_times, data.specific_force, timestamps
    )
    gyro = _interpolate(
        data.imu_times, data.angular_velocity, timestamps
    )
    velocity = _interpolate(
        data.state_times, data.linear_velocity, timestamps
    )
    width = max(
        3,
        int(round(settings.smoothing_window_s * settings.sample_rate_hz)),
    )
    if width % 2 == 0:
        width += 1
    if width >= timestamps.size // 3:
        raise ValueError("smoothing window is too wide")
    half = width // 2
    force = _moving_average(force, width)
    gyro = _moving_average(gyro, width)
    velocity = _moving_average(velocity, width)
    angular_acceleration = np.gradient(gyro, step, axis=0)
    selected = slice(half + 1, -half - 1)
    response = np.column_stack(
        (force[selected], angular_acceleration[selected])
    )
    state = np.column_stack(
        (velocity[selected], gyro[selected])
    )
    return _PreparedSignals(
        timestamps=timestamps[selected],
        response=response,
        state=state,
    )


def _zero_order_hold(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(source_times, query_times, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= source_times.size):
        raise ValueError("recorded command does not cover candidate delay")
    return source_values[indices]


def _robust_fit(
    input_value: np.ndarray,
    state_value: np.ndarray,
    response: np.ndarray,
    settings: EstimatorSettings,
) -> Tuple[np.ndarray, np.ndarray]:
    predictors = np.column_stack((input_value, state_value))
    means = np.mean(predictors, axis=0)
    scales = np.std(predictors, axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    design = np.column_stack(
        (
            np.ones(response.size),
            (predictors - means) / scales,
        )
    )
    weights = np.ones(response.size, dtype=float)
    coefficients = np.zeros(3, dtype=float)
    penalty = np.diag((0.0, settings.ridge, settings.ridge))
    for _ in range(20):
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        weighted_response = response * root_weight
        normal = weighted_design.T @ weighted_design + penalty
        target = weighted_design.T @ weighted_response
        updated = np.linalg.lstsq(normal, target, rcond=None)[0]
        residual = response - design @ updated
        scale = (
            1.4826
            * np.median(np.abs(residual - np.median(residual)))
            + 1.0e-12
        )
        threshold = settings.huber_delta * scale
        absolute = np.abs(residual)
        new_weights = np.ones_like(weights)
        large = absolute > threshold
        new_weights[large] = threshold / absolute[large]
        if np.max(np.abs(updated - coefficients)) < 1.0e-10:
            coefficients = updated
            break
        coefficients = updated
        weights = new_weights

    converted = np.asarray(
        (
            coefficients[0]
            - coefficients[1] * means[0] / scales[0]
            - coefficients[2] * means[1] / scales[1],
            coefficients[1] / scales[0],
            coefficients[2] / scales[1],
        ),
        dtype=float,
    )
    prediction = (
        converted[0]
        + converted[1] * input_value
        + converted[2] * state_value
    )
    return converted, prediction


def _delay_score(
    wrench: np.ndarray,
    prepared: _PreparedSignals,
    settings: EstimatorSettings,
    fit_mask: np.ndarray,
) -> float:
    scores = []
    for channel in range(6):
        response = prepared.response[fit_mask, channel]
        _, prediction = _robust_fit(
            wrench[fit_mask, channel],
            prepared.state[fit_mask, channel],
            response,
            settings,
        )
        scale = max(float(np.std(response)), 1.0e-9)
        scores.append(
            float(np.sqrt(np.mean((response - prediction) ** 2)) / scale)
        )
    return float(np.mean(scores))


def _block_indices(
    sample_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pieces = []
    maximum_start = max(1, sample_count - block_length + 1)
    while sum(piece.size for piece in pieces) < sample_count:
        start = int(rng.integers(0, maximum_start))
        pieces.append(
            np.arange(start, min(start + block_length, sample_count))
        )
    return np.concatenate(pieces)[:sample_count]


def _channel_grade(
    estimate: float,
    interval: Tuple[float, float],
    input_std: float,
    r_squared: float,
    settings: EstimatorSettings,
) -> str:
    if input_std < settings.minimum_input_std:
        return "not_excited"
    lower, upper = interval
    excludes_zero = lower > 0.0 or upper < 0.0
    relative_width = (upper - lower) / max(abs(estimate), 1.0e-12)
    if (
        excludes_zero
        and r_squared >= settings.minimum_r2
        and relative_width <= settings.maximum_relative_interval_width
    ):
        return "informative"
    return "weak"


def _coefficient_entry(
    estimate: float,
    samples: np.ndarray,
    unit: str,
) -> Mapping[str, Any]:
    interval = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate": float(estimate),
        "ci95": [float(interval[0]), float(interval[1])],
        "unit": unit,
    }


def estimate_effective_parameters(
    data: FailureBagData,
    settings: EstimatorSettings,
    fit_mask: np.ndarray = None,
    bootstrap: bool = True,
) -> Mapping[str, Any]:
    """Estimate diagonal effective response and autocorrelation-aware CIs."""

    prepared = _prepare_signals(data, settings)
    if fit_mask is None:
        selected_for_fit = np.ones(
            prepared.timestamps.shape, dtype=bool
        )
    else:
        selected_for_fit = np.asarray(fit_mask, dtype=bool)
        if selected_for_fit.shape != prepared.timestamps.shape:
            raise ValueError(
                "fit_mask must match the prepared estimator timestamps"
            )
    if int(np.sum(selected_for_fit)) < 30:
        raise ValueError("fewer than 30 selected fit samples remain")
    delays = np.arange(
        0.0,
        settings.maximum_delay_s + 0.5 * settings.delay_step_s,
        settings.delay_step_s,
        dtype=float,
    )
    delay_rows = []
    candidate_wrench = []
    for delay in delays:
        wrench = _zero_order_hold(
            data.command_times,
            data.command_wrench,
            prepared.timestamps - delay,
        )
        score = _delay_score(
            wrench, prepared, settings, selected_for_fit
        )
        delay_rows.append(
            {"delay_s": float(delay), "normalized_rmse": score}
        )
        candidate_wrench.append(wrench)
    best_index = min(
        range(len(delay_rows)),
        key=lambda index: (
            delay_rows[index]["normalized_rmse"],
            delay_rows[index]["delay_s"],
        ),
    )
    selected_delay = float(delays[best_index])
    wrench = candidate_wrench[best_index]

    axis_names = ("x", "y", "z", "roll", "pitch", "yaw")
    response_names = (
        "specific_force",
        "specific_force",
        "specific_force",
        "angular_acceleration",
        "angular_acceleration",
        "angular_acceleration",
    )
    bias_units = (
        "m/s^2",
        "m/s^2",
        "m/s^2",
        "rad/s^2",
        "rad/s^2",
        "rad/s^2",
    )
    gain_units = (
        "(m/s^2)/command_force_unit",
        "(m/s^2)/command_force_unit",
        "(m/s^2)/command_force_unit",
        "(rad/s^2)/command_torque_unit",
        "(rad/s^2)/command_torque_unit",
        "(rad/s^2)/command_torque_unit",
    )
    block_length = max(
        2,
        int(round(settings.bootstrap_block_s * settings.sample_rate_hz)),
    )
    block_length = min(block_length, int(np.sum(selected_for_fit)))
    rng = np.random.default_rng(settings.seed)
    parameters = {}
    channels = {}

    for channel, axis in enumerate(axis_names):
        response = prepared.response[:, channel]
        state = prepared.state[:, channel]
        input_value = wrench[:, channel]
        fit_response = response[selected_for_fit]
        fit_state = state[selected_for_fit]
        fit_input = input_value[selected_for_fit]
        coefficients, fit_prediction = _robust_fit(
            fit_input, fit_state, fit_response, settings
        )
        if bootstrap:
            samples = np.empty(
                (settings.bootstrap_samples, 3), dtype=float
            )
            for sample in range(settings.bootstrap_samples):
                indices = _block_indices(
                    fit_response.size, block_length, rng
                )
                samples[sample], _ = _robust_fit(
                    fit_input[indices],
                    fit_state[indices],
                    fit_response[indices],
                    settings,
                )
        else:
            samples = np.repeat(coefficients[None, :], 2, axis=0)
        gain_interval_values = np.quantile(
            samples[:, 1], (0.025, 0.975)
        )
        residual = fit_response - fit_prediction
        denominator = float(
            np.sum((fit_response - np.mean(fit_response)) ** 2)
        )
        r_squared = (
            0.0
            if denominator <= 1.0e-16
            else 1.0 - float(np.sum(residual ** 2)) / denominator
        )
        grade = _channel_grade(
            float(coefficients[1]),
            (
                float(gain_interval_values[0]),
                float(gain_interval_values[1]),
            ),
            float(np.std(fit_input)),
            r_squared,
            settings,
        )
        prefix = "{}_{}".format(response_names[channel], axis)
        bias_name = "{}_bias".format(prefix)
        gain_name = "{}_gain".format(prefix)
        feedback_name = "{}_velocity_feedback".format(prefix)
        parameters[bias_name] = _coefficient_entry(
            coefficients[0], samples[:, 0], bias_units[channel]
        )
        parameters[gain_name] = _coefficient_entry(
            coefficients[1], samples[:, 1], gain_units[channel]
        )
        parameters[feedback_name] = _coefficient_entry(
            coefficients[2], samples[:, 2], "1/s"
        )
        channels[axis] = {
            "information_grade": grade,
            "gain_parameter": gain_name,
            "input_standard_deviation": float(np.std(fit_input)),
            "response_standard_deviation": float(
                np.std(fit_response)
            ),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r_squared": r_squared,
        }

    return {
        "model": {
            "model_id": MODEL_ID,
            "geometry_id": GEOMETRY_ID,
            "geometry_evidence": "source_assumption_not_bag_verified",
            "interpretation": (
                "diagnostic closed-loop association; not a causal actuator "
                "gain, calibrated physical mass, or physical inertia"
            ),
            "limitations": [
                "recorded commands are endogenous closed-loop signals",
                "command force and torque units are not calibrated",
                "the diagonal model omits cross-axis coupling",
                "alignment lag is not identified as pure transport delay",
            ],
        },
        "selected_alignment_lag_s": selected_delay,
        "sample_rate_hz": settings.sample_rate_hz,
        "sample_count": int(prepared.timestamps.size),
        "fit_sample_count": int(np.sum(selected_for_fit)),
        "bootstrap": {
            "method": (
                "moving_block_bootstrap"
                if bootstrap
                else "disabled_point_fit"
            ),
            "samples": settings.bootstrap_samples if bootstrap else 0,
            "block_s": settings.bootstrap_block_s,
            "seed": settings.seed,
        },
        "parameters": parameters,
        "channels": channels,
        "delay_search": delay_rows,
    }


def evaluate_effective_parameters(
    data: FailureBagData,
    settings: EstimatorSettings,
    estimate: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    """Evaluate one fitted model at every prepared sample."""

    prepared = _prepare_signals(data, settings)
    delay = float(estimate["selected_alignment_lag_s"])
    wrench = _zero_order_hold(
        data.command_times,
        data.command_wrench,
        prepared.timestamps - delay,
    )
    axis_names = ("x", "y", "z", "roll", "pitch", "yaw")
    response_names = (
        "specific_force",
        "specific_force",
        "specific_force",
        "angular_acceleration",
        "angular_acceleration",
        "angular_acceleration",
    )
    prediction = np.empty(prepared.response.shape, dtype=float)
    for channel, axis in enumerate(axis_names):
        prefix = "{}_{}".format(response_names[channel], axis)
        bias = float(
            estimate["parameters"]["{}_bias".format(prefix)]["estimate"]
        )
        gain = float(
            estimate["parameters"]["{}_gain".format(prefix)]["estimate"]
        )
        feedback = float(
            estimate["parameters"][
                "{}_velocity_feedback".format(prefix)
            ]["estimate"]
        )
        prediction[:, channel] = (
            bias
            + gain * wrench[:, channel]
            + feedback * prepared.state[:, channel]
        )
    return {
        "timestamps": prepared.timestamps,
        "wrench": wrench,
        "response": prepared.response,
        "state": prepared.state,
        "prediction": prediction,
        "residual": prepared.response - prediction,
    }


def prepared_timestamps(
    data: FailureBagData,
    settings: EstimatorSettings,
) -> np.ndarray:
    """Return the exact timestamps used by the regression preprocessor."""

    return _prepare_signals(data, settings).timestamps.copy()


def effective_parameter_trace(
    data: FailureBagData,
    settings: EstimatorSettings,
    estimate: Mapping[str, Any],
    fit_mask: np.ndarray = None,
    minimum_duration_s: float = 0.5,
    step_s: float = 0.5,
) -> list:
    """Return cumulative robust-fit coefficients without repeated bootstrap."""

    evaluation = evaluate_effective_parameters(
        data, settings, estimate
    )
    timestamps = evaluation["timestamps"]
    if fit_mask is None:
        selected = np.ones(timestamps.shape, dtype=bool)
    else:
        selected = np.asarray(fit_mask, dtype=bool)
        if selected.shape != timestamps.shape:
            raise ValueError(
                "fit_mask must match the prepared estimator timestamps"
            )
    minimum = float(minimum_duration_s)
    step = float(step_s)
    if minimum <= 0.0 or step <= 0.0:
        raise ValueError("trace duration and step must be positive")
    first = float(timestamps[0] + minimum)
    final = float(timestamps[-1])
    if first > final:
        return []
    cutoffs = list(np.arange(first, final + 0.5 * step, step))
    if not cutoffs or final - cutoffs[-1] > 1.0e-9:
        cutoffs.append(final)

    axis_names = ("x", "y", "z", "roll", "pitch", "yaw")
    response_names = (
        "specific_force",
        "specific_force",
        "specific_force",
        "angular_acceleration",
        "angular_acceleration",
        "angular_acceleration",
    )
    rows = []
    for cutoff in cutoffs:
        prefix_mask = selected & (timestamps <= cutoff)
        if int(np.sum(prefix_mask)) < 30:
            continue
        parameters = {}
        for channel, axis in enumerate(axis_names):
            coefficients, _ = _robust_fit(
                evaluation["wrench"][prefix_mask, channel],
                evaluation["state"][prefix_mask, channel],
                evaluation["response"][prefix_mask, channel],
                settings,
            )
            prefix = "{}_{}".format(response_names[channel], axis)
            parameters["{}_bias".format(prefix)] = float(
                coefficients[0]
            )
            parameters["{}_gain".format(prefix)] = float(
                coefficients[1]
            )
            parameters[
                "{}_velocity_feedback".format(prefix)
            ] = float(coefficients[2])
        rows.append(
            {
                "time_s": float(min(cutoff, final)),
                "fit_sample_count": int(np.sum(prefix_mask)),
                "parameters": parameters,
            }
        )
    return rows


def run_from_bag(
    bag_path,
    config: LoadedConfig,
) -> Mapping[str, Any]:
    margin = (
        config.settings.maximum_delay_s
        + config.settings.smoothing_window_s
        + 0.05
    )
    data = read_failure_bag(
        bag_path,
        config.topics,
        config.settings.start_offset_s,
        config.settings.end_offset_s,
        margin,
    )
    if data.bag_sha256 != config.expected_bag_sha256:
        raise ValueError(
            "bag SHA-256 mismatch: expected {}, got {}".format(
                config.expected_bag_sha256, data.bag_sha256
            )
        )
    estimate = estimate_effective_parameters(
        data, config.settings
    )
    result = {
        "schema": RESULT_SCHEMA,
        "source": {
            "bag": data.bag_path,
            "bag_sha256": data.bag_sha256,
            "bag_start_time": data.bag_start_time,
            "start_offset_s": config.settings.start_offset_s,
            "end_offset_s": config.settings.end_offset_s,
            "topics": dict(sorted(config.topics.items())),
            "config_sha256": config.sha256,
        },
        **estimate,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def write_result(path, result: Mapping[str, Any], overwrite=False) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                result,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


__all__ = [
    "CONFIG_SCHEMA",
    "EstimatorSettings",
    "LoadedConfig",
    "canonical_sha256",
    "effective_parameter_trace",
    "estimate_effective_parameters",
    "evaluate_effective_parameters",
    "load_config",
    "prepared_timestamps",
    "run_from_bag",
    "write_result",
]
