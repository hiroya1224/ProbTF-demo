"""Probabilistic plant ridges and conservative PID/model suggestions."""

from typing import Any, Mapping

import numpy as np

from grape_param_estim.effective_estimator import (
    _block_indices,
    _robust_fit,
)
from grape_param_estim.failure_bag import (
    FailureBagRecording,
    PID_AXES,
)


GROUP_AXES = {
    "xy": ("x", "y"),
    "z": ("z",),
    "roll_pitch": ("roll", "pitch"),
    "yaw": ("yaw",),
}
MODEL_NAMES = {
    "xy": "controller_mass_equivalent_kg",
    "z": "controller_mass_equivalent_kg",
    "roll_pitch": "controller_inertia_equivalent_kg_m2",
    "yaw": "controller_inertia_equivalent_kg_m2",
}


def _quaternion_at(
    recording: FailureBagRecording,
    timestamps: np.ndarray,
) -> np.ndarray:
    if recording.orientation_xyzw is None:
        result = np.zeros((timestamps.size, 4), dtype=float)
        result[:, 3] = 1.0
        return result
    source = np.asarray(recording.orientation_xyzw, dtype=float).copy()
    for index in range(1, source.shape[0]):
        if float(np.dot(source[index - 1], source[index])) < 0.0:
            source[index] *= -1.0
    result = np.column_stack(
        [
            np.interp(
                timestamps,
                recording.state_times,
                source[:, column],
            )
            for column in range(4)
        ]
    )
    norm = np.linalg.norm(result, axis=1)
    return result / norm[:, None]


def _world_to_body(
    vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    x_value = quaternions_xyzw[:, 0]
    y_value = quaternions_xyzw[:, 1]
    z_value = quaternions_xyzw[:, 2]
    w_value = quaternions_xyzw[:, 3]
    rotation = np.empty((vectors.shape[0], 3, 3), dtype=float)
    rotation[:, 0, 0] = 1.0 - 2.0 * (
        y_value * y_value + z_value * z_value
    )
    rotation[:, 0, 1] = 2.0 * (
        x_value * y_value - z_value * w_value
    )
    rotation[:, 0, 2] = 2.0 * (
        x_value * z_value + y_value * w_value
    )
    rotation[:, 1, 0] = 2.0 * (
        x_value * y_value + z_value * w_value
    )
    rotation[:, 1, 1] = 1.0 - 2.0 * (
        x_value * x_value + z_value * z_value
    )
    rotation[:, 1, 2] = 2.0 * (
        y_value * z_value - x_value * w_value
    )
    rotation[:, 2, 0] = 2.0 * (
        x_value * z_value - y_value * w_value
    )
    rotation[:, 2, 1] = 2.0 * (
        y_value * z_value + x_value * w_value
    )
    rotation[:, 2, 2] = 1.0 - 2.0 * (
        x_value * x_value + y_value * y_value
    )
    return np.einsum("nji,nj->ni", rotation, vectors)


def _interpolated_pid(
    recording: FailureBagRecording,
    timestamps: np.ndarray,
):
    controller = recording.controller
    if controller is None:
        return None
    valid = (
        (timestamps >= controller.times[0])
        & (timestamps <= controller.times[-1])
    )
    terms = {}
    for name in (
        "total",
        "proportional",
        "integral",
        "derivative",
    ):
        source = np.asarray(getattr(controller, name), dtype=float)
        values = np.column_stack(
            [
                np.interp(
                    timestamps,
                    controller.times,
                    source[:, column],
                )
                for column in range(len(PID_AXES))
            ]
        )
        quaternions = _quaternion_at(recording, timestamps)
        values[:, :3] = _world_to_body(
            values[:, :3], quaternions
        )
        terms[name] = values
    return valid, terms


def _command_at(
    recording: FailureBagRecording,
    timestamps: np.ndarray,
):
    indices = np.searchsorted(
        recording.command_times, timestamps, side="right"
    ) - 1
    valid = (
        (indices >= 0)
        & (timestamps <= recording.command_times[-1])
    )
    values = np.zeros((timestamps.size, 6), dtype=float)
    values[valid] = recording.command_wrench[indices[valid]]
    return valid, values


def _r_squared(response, prediction) -> float:
    denominator = float(
        np.sum((response - np.mean(response)) ** 2)
    )
    if denominator <= 1.0e-16:
        return 0.0
    return 1.0 - float(
        np.sum((response - prediction) ** 2)
    ) / denominator


def _axis_posterior(
    pid_input: np.ndarray,
    state: np.ndarray,
    response: np.ndarray,
    command: np.ndarray,
    settings,
    rng,
    progress_callback=None,
) -> Mapping[str, Any]:
    coefficients, prediction = _robust_fit(
        pid_input, state, response, settings
    )
    model_coefficients, _ = _robust_fit(
        pid_input,
        np.zeros(pid_input.shape, dtype=float),
        command,
        settings,
    )
    sample_count = pid_input.size
    block_length = max(
        2,
        int(round(
            settings.bootstrap_block_s * settings.sample_rate_hz
        )),
    )
    block_length = min(block_length, sample_count)
    response_samples = np.empty(
        settings.bootstrap_samples, dtype=float
    )
    model_samples = np.empty(
        settings.bootstrap_samples, dtype=float
    )
    for sample in range(settings.bootstrap_samples):
        indices = _block_indices(
            sample_count, block_length, rng
        )
        response_fit, _ = _robust_fit(
            pid_input[indices],
            state[indices],
            response[indices],
            settings,
        )
        model_fit, _ = _robust_fit(
            pid_input[indices],
            np.zeros(indices.size, dtype=float),
            command[indices],
            settings,
        )
        response_samples[sample] = response_fit[1]
        model_samples[sample] = model_fit[1]
        if progress_callback is not None:
            progress_callback(sample + 1)
    response_interval = np.quantile(
        response_samples, (0.025, 0.975)
    )
    model_interval = np.quantile(
        model_samples, (0.025, 0.975)
    )
    estimate = float(coefficients[1])
    relative_width = (
        float(response_interval[1] - response_interval[0])
        / max(abs(estimate), 1.0e-12)
    )
    informative = (
        float(np.std(pid_input)) >= settings.minimum_input_std
        and response_interval[0] > 0.0
        and _r_squared(response, prediction) >= settings.minimum_r2
        and relative_width
        <= settings.maximum_relative_interval_width
    )
    return {
        "estimate": estimate,
        "ci95": [
            float(response_interval[0]),
            float(response_interval[1]),
        ],
        "model_estimate": float(model_coefficients[1]),
        "model_ci95": [
            float(model_interval[0]),
            float(model_interval[1]),
        ],
        "input_standard_deviation": float(np.std(pid_input)),
        "r_squared": _r_squared(response, prediction),
        "sample_count": int(sample_count),
        "information_grade": (
            "informative" if informative else "weak"
        ),
        "_response_samples": response_samples,
        "_model_samples": model_samples,
    }


def _positive_geometric_samples(rows, key: str):
    if not rows:
        return np.empty(0, dtype=float)
    samples = np.column_stack([row[key] for row in rows])
    valid = np.all(np.isfinite(samples) & (samples > 0.0), axis=1)
    if not np.any(valid):
        return np.empty(0, dtype=float)
    return np.exp(np.mean(np.log(samples[valid]), axis=1))


def _bounded_scale(value: float) -> float:
    return float(np.clip(value, 0.8, 1.2))


def _group_advice(
    group: str,
    rows,
    gains,
) -> Mapping[str, Any]:
    response_samples = _positive_geometric_samples(
        rows, "_response_samples"
    )
    model_samples = _positive_geometric_samples(
        rows, "_model_samples"
    )
    if response_samples.size < 10:
        return {
            "group": group,
            "status": "not_identifiable",
            "reason": "positive response-scale posterior is unavailable",
            "current_pid": gains,
        }
    response = float(np.median(response_samples))
    response_interval = np.quantile(
        response_samples, (0.025, 0.975)
    )
    model = (
        float(np.median(model_samples))
        if model_samples.size >= 10
        else None
    )
    model_interval = (
        np.quantile(model_samples, (0.025, 0.975))
        if model_samples.size >= 10
        else None
    )
    informative = (
        response_interval[0] > 0.0
        and all(
            row["information_grade"] == "informative"
            for row in rows
        )
        and max(
            row["feedforward_relative_rms"] for row in rows
        )
        <= 0.5
    )

    raw_pid_scale = 1.0 / np.sqrt(response)
    raw_model_scale = 1.0 / np.sqrt(response)
    nominal_in_interval = bool(
        response_interval[0] <= 1.0 <= response_interval[1]
    )
    apply_revision = informative and not nominal_in_interval
    if model is None:
        raw_pid_scale = 1.0 / response
        raw_model_scale = 1.0
    pid_scale = (
        _bounded_scale(raw_pid_scale)
        if apply_revision
        else 1.0
    )
    model_scale = (
        _bounded_scale(raw_model_scale)
        if apply_revision and model is not None
        else 1.0
    )
    proposed_pid = (
        {
            name: float(value * pid_scale)
            for name, value in gains.items()
        }
        if gains is not None
        else None
    )
    ridge_points = []
    for physical_ratio in (0.7, 0.85, 1.0, 1.15, 1.3):
        ridge_points.append(
            {
                "physical_parameter_ratio": physical_ratio,
                "physical_parameter": (
                    None
                    if model is None
                    else float(model * physical_ratio)
                ),
                "actuator_scale": float(
                    response * physical_ratio
                ),
                "actuator_scale_ci95": [
                    float(response_interval[0] * physical_ratio),
                    float(response_interval[1] * physical_ratio),
                ],
            }
        )
    return {
        "group": group,
        "status": (
            "weak_evidence"
            if not informative
            else (
                "nominal_within_uncertainty"
                if nominal_in_interval
                else "proposal_available"
            )
        ),
        "axes": list(GROUP_AXES[group]),
        "response_scale": {
            "estimate": response,
            "ci95": [
                float(response_interval[0]),
                float(response_interval[1]),
            ],
            "interpretation": (
                "actual acceleration divided by recorded PID "
                "desired acceleration"
            ),
        },
        "current_pid": gains,
        "pid_scaling_assumption": {
            "maximum_feedforward_relative_rms": float(
                max(
                    row["feedforward_relative_rms"]
                    for row in rows
                )
            ),
            "interpretation": (
                "P/I/D are scaled together; recorded feedforward is "
                "not changed"
            ),
        },
        "controller_model": {
            "parameter": MODEL_NAMES[group],
            "estimate": model,
            "ci95": (
                None
                if model_interval is None
                else [
                    float(model_interval[0]),
                    float(model_interval[1]),
                ]
            ),
            "evidence": (
                "recorded command wrench divided by PID desired "
                "acceleration; source-geometry equivalent"
            ),
        },
        "non_identifiability_ridge": {
            "equation": (
                "response_scale = actuator_scale / "
                "physical_parameter_ratio"
            ),
            "points": ridge_points,
        },
        "minimum_log_change": {
            "decision": (
                "apply_bounded_first_step"
                if apply_revision
                else "hold_current_values"
            ),
            "objective": (
                "equal log-distance for grouped PID scale and "
                "controller-model scale"
            ),
            "unbounded_pid_scale": float(raw_pid_scale),
            "unbounded_model_scale": float(raw_model_scale),
            "recommended_first_step_pid_scale": pid_scale,
            "recommended_first_step_model_scale": model_scale,
            "proposed_pid": proposed_pid,
            "proposed_controller_model_parameter": (
                None if model is None else float(model * model_scale)
            ),
            "predicted_response_scale_after_first_step": float(
                response * pid_scale * model_scale
            ),
            "per_step_bound": "each scale is limited to [0.8, 1.2]",
        },
    }


def build_controller_advice(
    recording: FailureBagRecording,
    fit: Mapping[str, Any],
    episode_time_s: float,
    progress_callback=None,
) -> Mapping[str, Any]:
    """Build advisory-only PID/model revisions from airborne fit samples."""

    controller = recording.controller
    if controller is None:
        return {
            "status": "not_available",
            "reason": "PID debug topic is absent",
            "groups": [],
        }
    timestamps = np.asarray(fit["timestamps"], dtype=float)
    alignment_lag_s = float(
        fit["estimate"]["selected_alignment_lag_s"]
    )
    controller_timestamps = timestamps - alignment_lag_s
    sampled = _interpolated_pid(
        recording, controller_timestamps
    )
    if sampled is None:
        return {
            "status": "not_available",
            "reason": "PID debug samples are unavailable",
            "groups": [],
        }
    pid_valid, terms = sampled
    command_valid, command = _command_at(
        recording, controller_timestamps
    )
    selected = (
        np.asarray(fit["fit_mask"], dtype=bool)
        & pid_valid
        & command_valid
    )
    if int(np.sum(selected)) < 30:
        return {
            "status": "not_available",
            "reason": "too few airborne PID-aligned samples",
            "groups": [],
        }

    response = np.asarray(
        fit["evaluation"]["response"], dtype=float
    )
    state = np.asarray(fit["evaluation"]["state"], dtype=float)
    settings = fit["settings"]
    rng = np.random.default_rng(settings.seed + 104729)
    axis_rows = {}
    serializable_axes = {}
    bootstrap_count = settings.bootstrap_samples
    for channel, axis in enumerate(PID_AXES):
        def report_axis(sample, channel_index=channel):
            if progress_callback is not None:
                progress_callback(
                    (
                        channel_index * bootstrap_count + sample
                    )
                    / (len(PID_AXES) * bootstrap_count),
                    "controller ridge posterior",
                )

        row = _axis_posterior(
            terms["total"][selected, channel],
            state[selected, channel],
            response[selected, channel],
            command[selected, channel],
            settings,
            rng,
            report_axis,
        )
        feedback = (
            terms["proportional"][selected, channel]
            + terms["integral"][selected, channel]
            + terms["derivative"][selected, channel]
        )
        feedforward = (
            terms["total"][selected, channel] - feedback
        )
        total_rms = float(
            np.sqrt(
                np.mean(
                    terms["total"][selected, channel] ** 2
                )
            )
        )
        row["feedforward_relative_rms"] = float(
            np.sqrt(np.mean(feedforward * feedforward))
            / max(total_rms, np.finfo(float).eps)
        )
        axis_rows[axis] = row
        serializable_axes[axis] = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        serializable_axes[axis]["pid_term_rms"] = {
            name: float(
                np.sqrt(
                    np.mean(
                        terms[term][selected, channel] ** 2
                    )
                )
            )
            for name, term in (
                ("p", "proportional"),
                ("i", "integral"),
                ("d", "derivative"),
            )
        }
        serializable_axes[axis][
            "feedforward_relative_rms"
        ] = row["feedforward_relative_rms"]

    groups = []
    for group, axes in GROUP_AXES.items():
        gains = controller.gains_at(group, episode_time_s)
        rows = [axis_rows[axis] for axis in axes]
        groups.append(_group_advice(group, rows, gains))
    return {
        "status": "available",
        "interpretation": (
            "Advisory first-step values only. They preserve the "
            "actuator/physical-parameter ridge and are not automatically "
            "written to the controller."
        ),
        "airborne_only": True,
        "alignment_lag_s": alignment_lag_s,
        "axes": serializable_axes,
        "groups": groups,
    }


__all__ = ["build_controller_advice"]
