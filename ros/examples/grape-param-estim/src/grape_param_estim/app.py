"""Streamlit application for Grape replay and particle estimation."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from grape_param_estim.data import (
    TOPIC_KEYS,
    load_yaml,
    read_bag,
    save_yaml,
    scan_bag_paths,
    suggest_analysis_interval,
)
from grape_param_estim.estimator import (
    PARAMETER_NAMES,
    load_result,
    relative_transform_from_poses,
    residual_rms,
    residual_se3_from_poses,
    weighted_quantile,
)
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    RigidBodyParameters,
    replay_segments,
)
from grape_param_estim.plots import (
    make_bag_overview_figure,
    make_body_frame_particle_figure,
    make_command_figure,
    make_correction_figure,
    make_estimated_pose_figure,
    make_parameter_ridge_figure,
    make_posterior_trajectory_figure,
    make_replay_pose_figure,
    make_replay_trajectory_figure,
    make_segment_residual_figure,
    make_state_figure,
    make_trajectory_figure,
    make_transform_particle_figure,
    make_uncertain_transform_time_figure,
)
from grape_param_estim.runs import (
    active_run_directory,
    latest_completed_run_directory,
    latest_run_directory,
    read_json,
    request_stop,
    start_run,
)


PARAMETER_LABELS = {
    "mass_scale": "Mass scale",
    "force_scale": "Force scale",
    "inertia_scale": "Inertia scale",
    "torque_scale": "Torque scale",
}


def _package_path() -> Path:
    try:
        import rospkg

        return Path(rospkg.RosPack().get_path("grape_param_estim"))
    except Exception:
        return Path(__file__).resolve().parents[2]


def _initial_config_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    arguments, _ = parser.parse_known_args()
    if arguments.config:
        return str(Path(arguments.config).expanduser())
    return str(_package_path() / "config" / "default.yaml")


def _configured_value(
    mapping: Mapping[str, Any], key: str, fallback: Any
) -> Any:
    value = mapping.get(key, fallback)
    return fallback if value is None else value


def _resolved_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _widget_suffix(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _bag_label(path: str, search_root: str) -> str:
    try:
        return str(
            Path(path).resolve().relative_to(Path(search_root).resolve())
        )
    except (OSError, ValueError):
        return str(Path(path))


def _configured_datasets(
    data_config: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    raw_datasets = analysis_config.get("datasets", ())
    if isinstance(raw_datasets, list):
        for item in raw_datasets:
            if not isinstance(item, Mapping) or not item.get("bag_path"):
                continue
            try:
                bag_path = _resolved_path(str(item["bag_path"]))
                result[bag_path] = {
                    "bag_path": bag_path,
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "segment_duration": float(item["segment_duration"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    if result:
        return result

    legacy_bag = str(data_config.get("bag_path", ""))
    if legacy_bag:
        try:
            bag_path = _resolved_path(legacy_bag)
            result[bag_path] = {
                "bag_path": bag_path,
                "start_time": float(analysis_config["start_time"]),
                "end_time": float(analysis_config["end_time"]),
                "segment_duration": float(
                    analysis_config.get("segment_duration", 0.75)
                ),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return result


def _saved_configuration(
    bag_directory: str,
    preview_bag_path: str,
    selected_bag_paths: Sequence[str],
    topics: Mapping[str, str],
    datasets: Sequence[Mapping[str, Any]],
    nominal_mass: float,
    nominal_inertia: np.ndarray,
    estimator: Mapping[str, Any],
    save_path: str,
    run_root: str,
) -> Dict[str, Any]:
    return {
        "schema": "grape_param_estim/phase2",
        "data": {
            "bag_directory": bag_directory,
            "bag_path": preview_bag_path,
            "preview_bag_path": preview_bag_path,
            "bag_paths": list(selected_bag_paths),
            "topics": {key: topics[key] for key in TOPIC_KEYS},
        },
        "analysis": {
            "datasets": [
                {
                    "bag_path": str(dataset["bag_path"]),
                    "start_time": float(dataset["start_time"]),
                    "end_time": float(dataset["end_time"]),
                    "segment_duration": float(
                        dataset["segment_duration"]
                    ),
                }
                for dataset in datasets
            ]
        },
        "model": {
            "nominal": {
                "mass": float(nominal_mass),
                "inertia_diagonal": [
                    float(value) for value in nominal_inertia
                ],
                "force_scale": 1.0,
                "torque_scale": 1.0,
            }
        },
        "estimator": {
            "particle_count": int(estimator["particle_count"]),
            "seed": int(estimator["seed"]),
            "priors": {
                name: {
                    "min": float(estimator["priors"][name]["min"]),
                    "max": float(estimator["priors"][name]["max"]),
                }
                for name in PARAMETER_NAMES
            },
            "likelihood": {
                "translation_weight": float(
                    estimator["likelihood"]["translation_weight"]
                ),
                "rotation_weight": float(
                    estimator["likelihood"]["rotation_weight"]
                ),
            },
            "resample_ess_fraction": float(
                estimator["resample_ess_fraction"]
            ),
            "jitter_fraction": float(estimator["jitter_fraction"]),
            "maximum_time_step": float(estimator["maximum_time_step"]),
        },
        "output": {
            "config_path": save_path,
            "run_root": run_root,
        },
    }


def _parameter_rows(particles: np.ndarray, weights: np.ndarray):
    quantiles = weighted_quantile(
        particles, weights, (0.025, 0.25, 0.5, 0.75, 0.975)
    )
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        rows.append(
            {
                "quantity": name,
                "2.5%": quantiles[0, index],
                "25%": quantiles[1, index],
                "median": quantiles[2, index],
                "75%": quantiles[3, index],
                "97.5%": quantiles[4, index],
            }
        )
    for name, values in (
        ("force / mass", particles[:, 1] / particles[:, 0]),
        ("torque / inertia", particles[:, 3] / particles[:, 2]),
    ):
        ratio_quantiles = weighted_quantile(
            values, weights, (0.025, 0.25, 0.5, 0.75, 0.975)
        )
        rows.append(
            {
                "quantity": name,
                "2.5%": ratio_quantiles[0],
                "25%": ratio_quantiles[1],
                "median": ratio_quantiles[2],
                "75%": ratio_quantiles[3],
                "97.5%": ratio_quantiles[4],
            }
        )
    return rows


def _estimated_parameter_rows(
    particles: np.ndarray, estimated_particle_index: int
):
    estimated = np.asarray(particles)[estimated_particle_index]
    rows = [
        {
            "quantity": name,
            "estimated nominal (MAP)": float(estimated[index]),
        }
        for index, name in enumerate(PARAMETER_NAMES)
    ]
    rows.extend(
        (
            {
                "quantity": "force / mass",
                "estimated nominal (MAP)": float(
                    estimated[1] / estimated[0]
                ),
            },
            {
                "quantity": "torque / inertia",
                "estimated nominal (MAP)": float(
                    estimated[3] / estimated[2]
                ),
            },
        )
    )
    return rows


def _result_bag_view(payload: Mapping[str, np.ndarray], bag_index: int):
    """Resolve schema-1/2 result arrays to unambiguous display semantics."""

    prefix = "bag_{}_".format(int(bag_index))
    weights = np.asarray(payload["weights"], dtype=float)
    particles = np.asarray(payload["particles"], dtype=float)
    map_index = int(np.argmax(weights))
    if "map_particle_index" in payload:
        candidate = int(np.asarray(payload["map_particle_index"]).flat[0])
        if 0 <= candidate < particles.shape[0]:
            map_index = candidate

    times = np.asarray(payload[prefix + "times"], dtype=float)
    segment_id = np.asarray(payload[prefix + "segment_id"])
    observed_position = np.asarray(
        payload[prefix + "observed_position"], dtype=float
    )
    observed_orientation = np.asarray(
        payload[prefix + "observed_orientation_xyzw"], dtype=float
    )
    baseline_position = np.asarray(
        payload.get(
            prefix + "baseline_position",
            payload[prefix + "nominal_position"],
        ),
        dtype=float,
    )
    baseline_orientation = np.asarray(
        payload.get(
            prefix + "baseline_orientation_xyzw",
            payload[prefix + "nominal_orientation_xyzw"],
        ),
        dtype=float,
    )
    posterior_position = np.asarray(
        payload[prefix + "posterior_position"], dtype=float
    )
    posterior_orientation = np.asarray(
        payload[prefix + "posterior_orientation_xyzw"], dtype=float
    )
    estimated_position = posterior_position[map_index]
    estimated_orientation = posterior_orientation[map_index]

    posterior_residual_key = prefix + "posterior_residual_se3"
    if posterior_residual_key in payload:
        posterior_residual = np.asarray(
            payload[posterior_residual_key], dtype=float
        )
    else:
        posterior_residual = residual_se3_from_poses(
            observed_position,
            observed_orientation,
            posterior_position,
            posterior_orientation,
            segment_id,
        )
    baseline_residual = residual_se3_from_poses(
        observed_position,
        observed_orientation,
        baseline_position,
        baseline_orientation,
        segment_id,
    )

    baseline_translation_key = prefix + "baseline_delta_translation"
    baseline_rotation_key = prefix + "baseline_delta_rotation_vector"
    if (
        baseline_translation_key in payload
        and baseline_rotation_key in payload
    ):
        baseline_delta_translation = np.asarray(
            payload[baseline_translation_key], dtype=float
        )
        baseline_delta_rotation = np.asarray(
            payload[baseline_rotation_key], dtype=float
        )
    else:
        baseline_delta_translation, baseline_delta_rotation = (
            relative_transform_from_poses(
                baseline_position,
                baseline_orientation,
                posterior_position,
                posterior_orientation,
            )
        )

    schema_version = int(
        np.asarray(payload.get("schema_version", (1,))).flat[0]
    )
    if (
        schema_version >= 2
        and prefix + "delta_translation" in payload
        and prefix + "delta_rotation_vector" in payload
    ):
        delta_translation = np.asarray(
            payload[prefix + "delta_translation"], dtype=float
        )
        delta_rotation = np.asarray(
            payload[prefix + "delta_rotation_vector"], dtype=float
        )
    else:
        delta_translation, delta_rotation = relative_transform_from_poses(
            estimated_position,
            estimated_orientation,
            posterior_position,
            posterior_orientation,
        )
    return {
        "times": times,
        "segment_id": segment_id,
        "observed_position": observed_position,
        "observed_orientation": observed_orientation,
        "baseline_position": baseline_position,
        "baseline_orientation": baseline_orientation,
        "estimated_position": estimated_position,
        "estimated_orientation": estimated_orientation,
        "posterior_position": posterior_position,
        "posterior_orientation": posterior_orientation,
        "baseline_residual_se3": baseline_residual,
        "posterior_residual_se3": posterior_residual,
        "delta_translation": delta_translation,
        "delta_rotation_vector": delta_rotation,
        "baseline_delta_translation": baseline_delta_translation,
        "baseline_delta_rotation_vector": baseline_delta_rotation,
        "estimated_particle_index": map_index,
    }


def _fit_rows(
    bag_view: Mapping[str, np.ndarray],
    weights: np.ndarray,
    translation_weight: float,
    rotation_weight: float,
):
    baseline_residual = bag_view["baseline_residual_se3"]
    posterior_residual = bag_view["posterior_residual_se3"]
    map_index = int(bag_view["estimated_particle_index"])
    baseline_translation, baseline_rotation = residual_rms(
        baseline_residual
    )
    posterior_translation, posterior_rotation = residual_rms(
        posterior_residual
    )

    def log_likelihood(residual):
        return -0.5 * (
            float(translation_weight) * np.sum(residual[..., :3] ** 2)
            + float(rotation_weight) * np.sum(residual[..., 3:] ** 2)
        )

    return [
        {
            "trajectory": "pre-fit baseline",
            "translation RMS [m]": float(baseline_translation),
            "rotation RMS [deg]": float(
                np.rad2deg(baseline_rotation)
            ),
            "log likelihood": float(log_likelihood(baseline_residual)),
        },
        {
            "trajectory": "estimated nominal (maximum-weight particle)",
            "translation RMS [m]": float(
                posterior_translation[map_index]
            ),
            "rotation RMS [deg]": float(
                np.rad2deg(posterior_rotation[map_index])
            ),
            "log likelihood": float(
                log_likelihood(posterior_residual[map_index])
            ),
        },
        {
            "trajectory": (
                "posterior weighted expectation "
                "(diagnostic, not one trajectory)"
            ),
            "translation RMS [m]": float(
                np.sum(weights * posterior_translation)
            ),
            "rotation RMS [deg]": float(
                np.rad2deg(np.sum(weights * posterior_rotation))
            ),
            "log likelihood": float(
                np.sum(
                    weights
                    * np.asarray(
                        [
                            log_likelihood(residual)
                            for residual in posterior_residual
                        ]
                    )
                )
            ),
        },
    ]


def _transform_rows(
    delta_translation: np.ndarray,
    delta_rotation_vector: np.ndarray,
    weights: np.ndarray,
):
    rotation_degrees = np.rad2deg(delta_rotation_vector)
    quantiles = (0.025, 0.25, 0.5, 0.75, 0.975)
    translation_summary = weighted_quantile(
        delta_translation, weights, quantiles
    )
    rotation_summary = weighted_quantile(
        rotation_degrees, weights, quantiles
    )
    rows = []
    for prefix, unit, values in (
        ("translation", "m", translation_summary),
        ("rotation vector", "deg", rotation_summary),
    ):
        for axis, label in enumerate(("x", "y", "z")):
            rows.append(
                {
                    "coordinate": "{} {}".format(prefix, label),
                    "unit": unit,
                    "2.5%": values[0, axis],
                    "25%": values[1, axis],
                    "median": values[2, axis],
                    "75%": values[3, axis],
                    "97.5%": values[4, axis],
                }
            )
    return rows


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(seconds):
        return "—"
    if seconds < 60.0:
        return "{:.1f} s".format(seconds)
    return "{}m {:.0f}s".format(int(seconds // 60), seconds % 60.0)


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Grape trajectory estimation",
        page_icon="🍇",
        layout="wide",
    )
    st.title("Grape trajectory estimation")
    st.caption(
        "Phase 0–2: inspect recorded motion, replay a nominal rigid body, "
        "infer a shared four-scale particle posterior, and push it forward "
        "to uncertain relative transforms."
    )

    config_source = _initial_config_path()
    try:
        configuration = load_yaml(config_source)
    except Exception as exc:
        st.error("Cannot load configuration {}: {}".format(config_source, exc))
        st.stop()

    data_config = dict(configuration.get("data", {}))
    analysis_config = dict(configuration.get("analysis", {}))
    model_config = dict(configuration.get("model", {}))
    nominal_config = dict(model_config.get("nominal", {}))
    estimator_config = dict(configuration.get("estimator", {}))
    output_config = dict(configuration.get("output", {}))
    configured_topics = dict(data_config.get("topics", {}))
    initial_datasets = _configured_datasets(data_config, analysis_config)
    if "phase2_dataset_by_path" not in st.session_state:
        st.session_state["phase2_dataset_by_path"] = dict(initial_datasets)

    st.sidebar.header("ROS bag")
    st.sidebar.caption("Loaded defaults from `{}`".format(config_source))
    bag_directory = st.sidebar.text_input(
        "Bag directory or single `.bag`",
        value=str(
            _configured_value(
                data_config,
                "bag_directory",
                "/home/leus/catkin_ws/bags/grape-drone",
            )
        ),
        help=(
            "A directory is scanned recursively. A direct .bag path produces "
            "a one-item file list."
        ),
    )
    bag_paths = scan_bag_paths(bag_directory)
    if not bag_paths:
        st.warning("No `.bag` file was found below `{}`.".format(bag_directory))
        st.stop()

    configured_bag = str(
        data_config.get(
            "preview_bag_path", data_config.get("bag_path", "")
        )
    )
    if configured_bag:
        configured_bag = _resolved_path(configured_bag)
    selected_index = (
        bag_paths.index(configured_bag)
        if configured_bag in bag_paths
        else 0
    )
    bag_path = st.sidebar.selectbox(
        "Preview / edit bag",
        bag_paths,
        index=selected_index,
        format_func=lambda path: _bag_label(path, bag_directory),
        help=(
            "This chooses the bag shown in the overview. It does not by "
            "itself change which bags are combined in the posterior."
        ),
    )

    configured_selected = data_config.get("bag_paths")
    if not isinstance(configured_selected, list):
        configured_selected = list(initial_datasets)
    configured_selected = [
        _resolved_path(str(path))
        for path in configured_selected
        if _resolved_path(str(path)) in bag_paths
    ]
    if not configured_selected and configured_bag in bag_paths:
        configured_selected = [configured_bag]
    estimation_bag_paths = st.sidebar.multiselect(
        "Bags used in one posterior",
        bag_paths,
        default=configured_selected,
        format_func=lambda path: _bag_label(path, bag_directory),
        help=(
            "Likelihoods from all selected bag intervals are added for the "
            "same static particles. Preview each bag once to inspect and "
            "adjust its own interval."
        ),
    )
    st.sidebar.caption(
        "{} bag(s) found; {} selected for estimation.".format(
            len(bag_paths), len(estimation_bag_paths)
        )
    )

    default_topics = {
        "thrust_command": "/gimbalrotor/four_axes/command",
        "gimbal_command": "/gimbalrotor/gimbals_ctrl",
        "gimbal_state": "/gimbalrotor/joint_states",
        "imu": "/gimbalrotor/sensor_plugin/imu1/ros_converted",
        "cog_odometry": "/gimbalrotor/uav/cog/odom",
        "body_odometry": "/gimbalrotor/uav/baselink/odom",
        "flight_state": "/gimbalrotor/flight_state",
    }
    topics = {}
    with st.sidebar.expander("Advanced: ROS topics"):
        for key in TOPIC_KEYS:
            topics[key] = st.text_input(
                key.replace("_", " ").title(),
                value=str(
                    _configured_value(
                        configured_topics, key, default_topics[key]
                    )
                ),
                key="topic_{}".format(key),
            )

    @st.cache_data(show_spinner=False)
    def cached_read(
        path: str,
        topic_items,
        file_size: int,
        modified_ns: int,
    ):
        del file_size, modified_ns
        return read_bag(path, dict(topic_items))

    @st.cache_data(show_spinner=False)
    def cached_result(
        path: str, file_size: int, modified_ns: int
    ):
        del file_size, modified_ns
        return load_result(path)

    @st.cache_data(show_spinner=False)
    def cached_bag_view(
        path: str,
        file_size: int,
        modified_ns: int,
        bag_index: int,
    ):
        del file_size, modified_ns
        return _result_bag_view(load_result(path), bag_index)

    bag_stat = Path(bag_path).stat()
    try:
        with st.spinner("Reading selected ROS bag…"):
            recording = cached_read(
                bag_path,
                tuple((key, topics[key]) for key in TOPIC_KEYS),
                int(bag_stat.st_size),
                int(bag_stat.st_mtime_ns),
            )
    except Exception as exc:
        st.error("Cannot read `{}`: {}".format(bag_path, exc))
        st.stop()

    available_start, available_end = recording.analysis_bounds
    minimum_window = min(0.25, 0.25 * (available_end - available_start))
    dataset_by_path = dict(st.session_state["phase2_dataset_by_path"])
    configured_dataset = dataset_by_path.get(bag_path)
    if configured_dataset is not None:
        requested_start = float(configured_dataset["start_time"])
        requested_end = float(configured_dataset["end_time"])
        requested_duration = max(
            minimum_window, requested_end - requested_start
        )
    else:
        requested_duration = min(5.28, available_end - available_start)
        requested_start, requested_end = suggest_analysis_interval(
            recording, requested_duration
        )
    configured_start = float(
        np.clip(
            requested_start,
            available_start,
            available_end - minimum_window,
        )
    )
    configured_end = float(
        np.clip(
            requested_end,
            configured_start + minimum_window,
            available_end,
        )
    )
    suggested_interval = suggest_analysis_interval(
        recording,
        max(minimum_window, configured_end - configured_start),
    )
    initial_interval = (configured_start, configured_end)
    suffix = _widget_suffix(bag_path)
    interval_key = "analysis_interval_{}".format(suffix)
    current_interval = st.session_state.get(interval_key, initial_interval)
    current_start = float(
        np.clip(
            current_interval[0],
            available_start,
            available_end - minimum_window,
        )
    )
    current_end = float(
        np.clip(
            current_interval[1],
            current_start + minimum_window,
            available_end,
        )
    )
    st.session_state[interval_key] = (current_start, current_end)

    st.subheader("1. Choose the motion to analyse")
    st.caption(
        "`{}` · common physical-stream range {:.3f}–{:.3f} s. "
        "The orange band is this bag's saved interval.".format(
            bag_path, available_start, available_end
        )
    )
    overview_event = st.plotly_chart(
        make_bag_overview_figure(
            recording, st.session_state[interval_key]
        ),
        use_container_width=True,
        key="bag_overview_{}".format(suffix),
        on_select="rerun",
        selection_mode=("box",),
        config={"displaylogo": False},
    )
    try:
        selected_points = overview_event["selection"]["points"]
    except (KeyError, TypeError):
        selected_points = ()
    selected_x = sorted(
        float(point["x"])
        for point in selected_points
        if point.get("x") is not None
    )
    selection_seen_key = "overview_selection_seen_{}".format(suffix)
    if len(selected_x) >= 2:
        plot_selection = (
            float(np.clip(selected_x[0], available_start, available_end)),
            float(np.clip(selected_x[-1], available_start, available_end)),
        )
        signature = tuple(round(value, 6) for value in plot_selection)
        if (
            plot_selection[1] - plot_selection[0] >= minimum_window
            and st.session_state.get(selection_seen_key) != signature
        ):
            st.session_state[selection_seen_key] = signature
            st.session_state[interval_key] = plot_selection
            st.rerun()

    interval_column, suggestion_column = st.columns((5, 1))
    with suggestion_column:
        if st.button(
            "Use suggested window",
            key="suggest_interval_{}".format(suffix),
            use_container_width=True,
        ):
            st.session_state[interval_key] = suggested_interval
            st.rerun()
    with interval_column:
        selected_interval = st.slider(
            "Analysis interval [bag-local s]",
            min_value=float(available_start),
            max_value=float(available_end),
            step=0.01,
            format="%.2f s",
            key=interval_key,
            help=(
                "Drag either handle for exact adjustment. You can also box-"
                "select a horizontal range in the overview above."
            ),
        )
    start_time, end_time = (
        float(selected_interval[0]),
        float(selected_interval[1]),
    )

    st.sidebar.header("Short replay")
    st.sidebar.caption(
        "Preview interval {:.3f}–{:.3f} s ({:.3f} s).".format(
            start_time, end_time, end_time - start_time
        )
    )
    segment_key = "segment_duration_{}".format(suffix)
    configured_segment = (
        float(configured_dataset["segment_duration"])
        if configured_dataset is not None
        else 0.75
    )
    segment_default = float(
        np.clip(
            configured_segment,
            0.05,
            max(0.05, float(end_time - start_time)),
        )
    )
    st.session_state[segment_key] = float(
        np.clip(
            st.session_state.get(segment_key, segment_default),
            0.05,
            max(0.05, float(end_time - start_time)),
        )
    )
    segment_duration = st.sidebar.number_input(
        "Segment length [s]",
        min_value=0.05,
        max_value=max(0.05, float(end_time - start_time)),
        step=0.05,
        format="%.3f",
        key=segment_key,
    )
    dataset_by_path[bag_path] = {
        "bag_path": bag_path,
        "start_time": start_time,
        "end_time": end_time,
        "segment_duration": float(segment_duration),
    }
    st.session_state["phase2_dataset_by_path"] = dataset_by_path

    default_inertia = np.asarray(
        (0.0649940671, 0.0649466618, 0.1289801290), dtype=float
    )
    configured_inertia = np.asarray(
        nominal_config.get("inertia_diagonal", default_inertia),
        dtype=float,
    )
    if (
        configured_inertia.shape != (3,)
        or np.any(~np.isfinite(configured_inertia))
        or np.any(configured_inertia <= 0.0)
    ):
        configured_inertia = default_inertia
    configured_mass = float(
        nominal_config.get("mass", 2.351557590812377)
    )
    if not np.isfinite(configured_mass) or configured_mass <= 0.0:
        configured_mass = 2.351557590812377

    st.sidebar.header("Nominal rigid body")
    st.sidebar.caption(
        "Defaults are the current Grape zero-joint URDF aggregate."
    )
    nominal_mass = st.sidebar.number_input(
        "Mass [kg]",
        min_value=0.01,
        max_value=100.0,
        value=configured_mass,
        step=0.01,
        format="%.6f",
    )
    nominal_inertia = np.asarray(
        [
            st.sidebar.number_input(
                "{} [kg m²]".format(label),
                min_value=0.000001,
                max_value=100.0,
                value=float(configured_inertia[index]),
                step=0.001,
                format="%.6f",
            )
            for index, label in enumerate(("Ixx", "Iyy", "Izz"))
        ]
    )

    try:
        analysis = recording.select_interval(
            start_time, end_time, segment_duration
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    parameters = RigidBodyParameters.from_diagonal(
        nominal_mass, nominal_inertia
    )
    try:
        with st.spinner("Replaying nominal short segments…"):
            replay = replay_segments(
                analysis, GrapeRigidBodyModel(), parameters
            )
    except Exception as exc:
        st.error("Nominal replay failed: {}".format(exc))
        st.stop()

    default_save_path = str(
        _configured_value(
            output_config,
            "config_path",
            "~/.ros/grape_param_estim/analysis.yaml",
        )
    )
    save_path = st.sidebar.text_input(
        "Save YAML path", value=default_save_path
    )
    run_root = st.sidebar.text_input(
        "Estimation run directory",
        value=str(
            _configured_value(
                output_config,
                "run_root",
                "~/.ros/grape_param_estim/runs",
            )
        ),
        help=(
            "Each run gets config.yaml, status.json, result.npz, "
            "summary.json, and error.txt below this directory."
        ),
    )

    selected_datasets = [
        dataset_by_path[path]
        for path in estimation_bag_paths
        if path in dataset_by_path
    ]
    missing_dataset_paths = [
        path
        for path in estimation_bag_paths
        if path not in dataset_by_path
    ]

    st.subheader("2. Inspect and estimate")
    (
        data_tab,
        replay_tab,
        estimate_tab,
        transform_tab,
    ) = st.tabs(
        (
            "Selected data",
            "Nominal replay",
            "Parameter posterior",
            "Uncertain transform",
        )
    )
    chart_config = {"displaylogo": False}
    trajectory_config = {"displaylogo": False, "scrollZoom": True}

    with data_tab:
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Selected duration",
            "{:.3f} s".format(end_time - start_time),
        )
        metric_columns[1].metric(
            "State samples", str(analysis.times.size)
        )
        metric_columns[2].metric(
            "Segments", str(analysis.segment_count)
        )
        metric_columns[3].metric(
            "Bag duration", "{:.3f} s".format(recording.bag_duration)
        )
        st.info(
            "The replay frame is centred at measured CoG translation and "
            "aligned with the physical baselink orientation. Solid gimbal "
            "traces are measured joints; dotted traces are targets. Gray "
            "vertical lines are segment boundaries."
        )
        st.plotly_chart(
            make_trajectory_figure(analysis),
            use_container_width=True,
            config=trajectory_config,
        )
        st.plotly_chart(
            make_state_figure(analysis),
            use_container_width=True,
            config=chart_config,
        )
        st.plotly_chart(
            make_command_figure(analysis),
            use_container_width=True,
            config=chart_config,
        )

    with replay_tab:
        translation_rms = float(
            np.sqrt(np.mean(replay.translation_residual_norm**2))
        )
        translation_max = float(
            np.max(replay.translation_residual_norm)
        )
        rotation_rms = float(
            np.rad2deg(
                np.sqrt(np.mean(replay.rotation_residual_norm**2))
            )
        )
        rotation_max = float(
            np.rad2deg(np.max(replay.rotation_residual_norm))
        )
        replay_metrics = st.columns(4)
        replay_metrics[0].metric(
            "Translation RMS", "{:.4f} m".format(translation_rms)
        )
        replay_metrics[1].metric(
            "Translation max", "{:.4f} m".format(translation_max)
        )
        replay_metrics[2].metric(
            "Rotation RMS", "{:.2f}°".format(rotation_rms)
        )
        replay_metrics[3].metric(
            "Rotation max", "{:.2f}°".format(rotation_max)
        )
        st.info(
            "Every segment starts from its observed `(p, R, v, ω)`. "
            "Black is recorded motion; dashed orange is nominal and resets "
            "at every gray segment boundary. The 3-D frame is centred/scaled "
            "from the black trajectory only; use the mouse wheel to zoom. "
            "The SE(3) residual is "
            "`Log(T_obs,rel⁻¹ T_nom,rel)`. The correction plot shows "
            "`T_nom⁻¹ T_obs`."
        )
        st.plotly_chart(
            make_replay_trajectory_figure(analysis, replay),
            use_container_width=True,
            config=trajectory_config,
        )
        st.plotly_chart(
            make_replay_pose_figure(analysis, replay),
            use_container_width=True,
            config=chart_config,
        )
        st.plotly_chart(
            make_correction_figure(analysis, replay),
            use_container_width=True,
            config=chart_config,
        )
        st.plotly_chart(
            make_segment_residual_figure(analysis, replay),
            use_container_width=True,
            config=chart_config,
        )

    estimator_settings = {}
    completed_run = None
    result_payload = None
    result_configuration = {}
    result_path = None
    result_stat = None
    with estimate_tab:
        st.markdown("#### Data entering this posterior")
        if selected_datasets:
            st.dataframe(
                [
                    {
                        "bag": _bag_label(
                            str(dataset["bag_path"]), bag_directory
                        ),
                        "start [s]": dataset["start_time"],
                        "end [s]": dataset["end_time"],
                        "segment [s]": dataset["segment_duration"],
                    }
                    for dataset in selected_datasets
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Select at least one bag in the sidebar.")
        if missing_dataset_paths:
            st.warning(
                "Preview these selected bags once before starting, so their "
                "bag-local interval can be inspected and stored: {}".format(
                    ", ".join(
                        _bag_label(path, bag_directory)
                        for path in missing_dataset_paths
                    )
                )
            )

        st.markdown("#### Static-particle settings")
        setting_columns = st.columns(4)
        particle_count = setting_columns[0].number_input(
            "Particles",
            min_value=8,
            max_value=4096,
            value=int(estimator_config.get("particle_count", 64)),
            step=8,
            help=(
                "More particles resolve ridges better but replay cost grows "
                "linearly."
            ),
        )
        seed = setting_columns[1].number_input(
            "Random seed",
            min_value=0,
            max_value=2147483647,
            value=int(estimator_config.get("seed", 42)),
            step=1,
        )
        likelihood_config = dict(
            estimator_config.get("likelihood", {})
        )
        translation_weight = setting_columns[2].number_input(
            "Translation weight [1/m²]",
            min_value=0.000001,
            max_value=1000000.0,
            value=float(
                likelihood_config.get("translation_weight", 10.0)
            ),
            format="%.6g",
        )
        rotation_weight = setting_columns[3].number_input(
            "Rotation weight [1/rad²]",
            min_value=0.000001,
            max_value=1000000.0,
            value=float(
                likelihood_config.get("rotation_weight", 1.0)
            ),
            format="%.6g",
        )

        prior_defaults = {
            "mass_scale": (0.60, 1.60),
            "force_scale": (0.40, 1.60),
            "inertia_scale": (0.50, 2.00),
            "torque_scale": (0.05, 1.50),
        }
        configured_priors = dict(estimator_config.get("priors", {}))
        prior_columns = st.columns(4)
        priors = {}
        for index, name in enumerate(PARAMETER_NAMES):
            configured_prior = dict(configured_priors.get(name, {}))
            with prior_columns[index]:
                st.caption(PARAMETER_LABELS[name])
                lower = st.number_input(
                    "Minimum",
                    min_value=0.000001,
                    max_value=1000.0,
                    value=float(
                        configured_prior.get(
                            "min", prior_defaults[name][0]
                        )
                    ),
                    format="%.5g",
                    key="prior_min_{}".format(name),
                )
                upper = st.number_input(
                    "Maximum",
                    min_value=0.000001,
                    max_value=1000.0,
                    value=float(
                        configured_prior.get(
                            "max", prior_defaults[name][1]
                        )
                    ),
                    format="%.5g",
                    key="prior_max_{}".format(name),
                )
                priors[name] = {"min": lower, "max": upper}

        with st.expander("Advanced particle settings"):
            advanced_columns = st.columns(3)
            resample_fraction = advanced_columns[0].number_input(
                "Resample below ESS / N",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    estimator_config.get("resample_ess_fraction", 0.10)
                ),
                step=0.01,
                format="%.3f",
            )
            jitter_fraction = advanced_columns[1].number_input(
                "Jitter / prior width",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    estimator_config.get("jitter_fraction", 0.03)
                ),
                step=0.01,
                format="%.3f",
            )
            maximum_time_step = advanced_columns[2].number_input(
                "Integrator max step [s]",
                min_value=0.0001,
                max_value=0.1,
                value=float(
                    estimator_config.get("maximum_time_step", 0.005)
                ),
                step=0.001,
                format="%.4f",
            )

        estimator_settings = {
            "particle_count": int(particle_count),
            "seed": int(seed),
            "priors": priors,
            "likelihood": {
                "translation_weight": float(translation_weight),
                "rotation_weight": float(rotation_weight),
            },
            "resample_ess_fraction": float(resample_fraction),
            "jitter_fraction": float(jitter_fraction),
            "maximum_time_step": float(maximum_time_step),
        }
        current_configuration = _saved_configuration(
            bag_directory=bag_directory,
            preview_bag_path=bag_path,
            selected_bag_paths=estimation_bag_paths,
            topics=topics,
            datasets=selected_datasets,
            nominal_mass=nominal_mass,
            nominal_inertia=nominal_inertia,
            estimator=estimator_settings,
            save_path=save_path,
            run_root=run_root,
        )
        invalid_priors = [
            name
            for name in PARAMETER_NAMES
            if priors[name]["max"] <= priors[name]["min"]
        ]
        if invalid_priors:
            st.error(
                "Prior maximum must exceed minimum: {}".format(
                    ", ".join(invalid_priors)
                )
            )

        latest_run = latest_run_directory(run_root)
        active_run = active_run_directory(run_root)
        action_columns = st.columns(4)
        if action_columns[0].button(
            "Start estimation",
            type="primary",
            use_container_width=True,
            disabled=bool(
                not selected_datasets
                or missing_dataset_paths
                or invalid_priors
                or active_run is not None
            ),
        ):
            try:
                start_run(run_root, current_configuration)
            except Exception as exc:
                st.error("Could not start estimator: {}".format(exc))
            else:
                st.rerun()
        if action_columns[1].button(
            "Stop worker",
            use_container_width=True,
            disabled=active_run is None,
        ):
            try:
                request_stop(run_root)
            except Exception as exc:
                st.error("Could not stop estimator: {}".format(exc))
            else:
                st.rerun()
        action_columns[2].button(
            "Refresh status", use_container_width=True
        )
        if action_columns[3].button(
            "Save analysis YAML", use_container_width=True
        ):
            try:
                written_path = save_yaml(save_path, current_configuration)
            except Exception as exc:
                st.error("Could not save YAML: {}".format(exc))
            else:
                st.success("Saved `{}`".format(written_path))

        latest_run = latest_run_directory(run_root)
        if latest_run is not None:
            status = read_json(latest_run / "status.json")
            state = str(status.get("state", "unknown"))
            progress = float(
                np.clip(status.get("progress", 0.0), 0.0, 1.0)
            )
            st.markdown("#### Latest run")
            status_columns = st.columns(4)
            status_columns[0].metric("State", state)
            status_columns[1].metric(
                "Progress", "{:.1f}%".format(100.0 * progress)
            )
            status_columns[2].metric(
                "Elapsed",
                _format_seconds(status.get("elapsed_seconds")),
            )
            status_columns[3].metric(
                "ETA", _format_seconds(status.get("eta_seconds"))
            )
            st.progress(progress)
            st.caption(
                "`{}` · {} · {}".format(
                    latest_run,
                    status.get("stage", ""),
                    status.get("message", ""),
                )
            )
            if state == "failed":
                try:
                    error_text = (latest_run / "error.txt").read_text(
                        encoding="utf-8"
                    )
                except OSError:
                    error_text = ""
                if error_text:
                    st.code(error_text[-6000:])

        completed_run = latest_completed_run_directory(run_root)
        if completed_run is not None:
            result_path = completed_run / "result.npz"
            result_stat = result_path.stat()
            try:
                result_payload = cached_result(
                    str(result_path),
                    int(result_stat.st_size),
                    int(result_stat.st_mtime_ns),
                )
            except Exception as exc:
                st.error("Cannot load result.npz: {}".format(exc))
                result_payload = None
            try:
                result_configuration = load_yaml(
                    str(completed_run / "config.yaml")
                )
            except Exception:
                result_configuration = {}
        if result_payload is not None:
            particles = result_payload["particles"]
            weights = result_payload["weights"]
            ess = float(1.0 / np.sum(weights**2))
            map_index = (
                int(result_payload["map_particle_index"][0])
                if "map_particle_index" in result_payload
                else int(np.argmax(weights))
            )
            result_bag_paths = [
                str(path) for path in result_payload["bag_paths"]
            ]
            result_estimator = dict(
                result_configuration.get("estimator", {})
            )
            result_likelihood = dict(
                result_estimator.get("likelihood", {})
            )
            result_translation_weight = float(
                result_likelihood.get("translation_weight", 10.0)
            )
            result_rotation_weight = float(
                result_likelihood.get("rotation_weight", 1.0)
            )
            posterior_metrics = st.columns(4)
            posterior_metrics[0].metric(
                "Particles", str(particles.shape[0])
            )
            posterior_metrics[1].metric("ESS", "{:.2f}".format(ess))
            posterior_metrics[2].metric(
                "ESS / N", "{:.3f}".format(ess / particles.shape[0])
            )
            posterior_metrics[3].metric(
                "Resampled",
                "yes" if bool(result_payload["resampled"][0]) else "no",
            )
            st.caption(
                "Showing the latest completed run `{}`.".format(completed_run)
            )
            st.write(
                "**Result data:** {}".format(
                    ", ".join(
                        "`{}`".format(
                            _bag_label(path, bag_directory)
                        )
                        for path in result_bag_paths
                    )
                )
            )
            if bag_path not in result_bag_paths:
                st.warning(
                    "The overview currently previews `{}`, but this completed "
                    "posterior was estimated from {}. The result below does "
                    "not belong to the previewed bag.".format(
                        _bag_label(bag_path, bag_directory),
                        ", ".join(
                            _bag_label(path, bag_directory)
                            for path in result_bag_paths
                        ),
                    )
                )
            if ess < max(10.0, 0.20 * particles.shape[0]):
                st.warning(
                    "Posterior resolution is low: ESS {:.2f} / {}. "
                    "The estimated trajectory is the best realised particle, "
                    "but uncertainty intervals are supported by only a few "
                    "effective particles. Increase particle count or revise "
                    "the prior before interpreting interval width.".format(
                        ess, particles.shape[0]
                    )
                )

            st.markdown("#### Estimated nominal and fit improvement")
            st.caption(
                "The estimated nominal is one physically realised trajectory: "
                "the maximum-weight particle (index {}). It is not a "
                "coordinate-wise median curve.".format(map_index)
            )
            estimated_columns = st.columns((1, 2))
            with estimated_columns[0]:
                st.dataframe(
                    _estimated_parameter_rows(particles, map_index),
                    use_container_width=True,
                    hide_index=True,
                )
            fit_table = []
            for result_bag_index, result_bag_path in enumerate(
                result_bag_paths
            ):
                bag_view = cached_bag_view(
                    str(result_path),
                    int(result_stat.st_size),
                    int(result_stat.st_mtime_ns),
                    result_bag_index,
                )
                for row in _fit_rows(
                    bag_view,
                    weights,
                    result_translation_weight,
                    result_rotation_weight,
                ):
                    fit_table.append(
                        {
                            "bag": _bag_label(
                                result_bag_path, bag_directory
                            ),
                            **row,
                        }
                    )
            with estimated_columns[1]:
                st.dataframe(
                    fit_table,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("#### Posterior ridges and intervals")
            st.plotly_chart(
                make_parameter_ridge_figure(particles, weights),
                use_container_width=True,
                config=chart_config,
            )
            st.dataframe(
                _parameter_rows(particles, weights),
                use_container_width=True,
                hide_index=True,
            )

    with transform_tab:
        if result_payload is None:
            completed_run = latest_completed_run_directory(run_root)
            if completed_run is not None:
                result_path = completed_run / "result.npz"
                result_stat = result_path.stat()
                try:
                    result_payload = cached_result(
                        str(result_path),
                        int(result_stat.st_size),
                        int(result_stat.st_mtime_ns),
                    )
                except Exception as exc:
                    st.error("Cannot load result.npz: {}".format(exc))
                try:
                    result_configuration = load_yaml(
                        str(completed_run / "config.yaml")
                    )
                except Exception:
                    result_configuration = {}
        if result_payload is None:
            st.info(
                "Complete an estimation run first. This tab always reloads "
                "the latest completed run after a browser refresh."
            )
        else:
            weights = result_payload["weights"]
            result_bag_paths = [
                str(path) for path in result_payload["bag_paths"]
            ]
            current_result_index = (
                result_bag_paths.index(bag_path)
                if bag_path in result_bag_paths
                else 0
            )
            result_bag_path = st.selectbox(
                "Result bag",
                result_bag_paths,
                index=current_result_index,
                format_func=lambda path: _bag_label(path, bag_directory),
                key="uncertain_transform_bag",
            )
            bag_index = result_bag_paths.index(result_bag_path)
            bag_view = cached_bag_view(
                str(result_path),
                int(result_stat.st_size),
                int(result_stat.st_mtime_ns),
                bag_index,
            )
            times = bag_view["times"]
            segment_id = bag_view["segment_id"]
            observed_position = bag_view["observed_position"]
            observed_orientation = bag_view["observed_orientation"]
            baseline_position = bag_view["baseline_position"]
            baseline_orientation = bag_view["baseline_orientation"]
            estimated_position = bag_view["estimated_position"]
            estimated_orientation = bag_view["estimated_orientation"]
            posterior_position = bag_view["posterior_position"]
            posterior_orientation = bag_view["posterior_orientation"]
            map_index = int(bag_view["estimated_particle_index"])
            ess = float(1.0 / np.sum(np.asarray(weights) ** 2))

            if bag_path != result_bag_path:
                st.warning(
                    "The overview previews `{}`, while this plot shows the "
                    "stored result for `{}`. Time and trajectory selections "
                    "come from the result bag.".format(
                        _bag_label(bag_path, bag_directory),
                        _bag_label(result_bag_path, bag_directory),
                    )
                )
            if ess < max(10.0, 0.20 * len(weights)):
                st.warning(
                    "This run has low posterior resolution (ESS {:.2f} / {}). "
                    "The MAP fit can be inspected, but the 50% / 95% "
                    "transform intervals are supported by only a few "
                    "effective particles.".format(ess, len(weights))
                )
            result_estimator = dict(
                result_configuration.get("estimator", {})
            )
            result_likelihood = dict(
                result_estimator.get("likelihood", {})
            )
            fit_rows = _fit_rows(
                bag_view,
                weights,
                float(
                    result_likelihood.get("translation_weight", 10.0)
                ),
                float(result_likelihood.get("rotation_weight", 1.0)),
            )
            st.markdown("#### 1. Does the estimated model explain the data?")
            st.dataframe(
                [
                    {
                        "bag": _bag_label(
                            result_bag_path, bag_directory
                        ),
                        **row,
                    }
                    for row in fit_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            baseline_fit = fit_rows[0]
            estimated_fit = fit_rows[1]
            if (
                estimated_fit["log likelihood"]
                > baseline_fit["log likelihood"]
            ):
                st.success(
                    "For this bag, the estimated nominal improves log "
                    "likelihood from {:.3f} to {:.3f}; translation RMS "
                    "{:.4f} → {:.4f} m and rotation RMS {:.2f} → {:.2f}°."
                    .format(
                        baseline_fit["log likelihood"],
                        estimated_fit["log likelihood"],
                        baseline_fit["translation RMS [m]"],
                        estimated_fit["translation RMS [m]"],
                        baseline_fit["rotation RMS [deg]"],
                        estimated_fit["rotation RMS [deg]"],
                    )
                )
            else:
                st.warning(
                    "The estimated nominal does not improve this bag's "
                    "likelihood over the pre-fit baseline. With multiple "
                    "bags the shared posterior may trade one bag against "
                    "another; inspect all per-bag rows before accepting it."
                )
            st.info(
                "Black is recorded motion. Dashed orange is the pre-fit "
                "scale=1 baseline. Blue is the estimated nominal generated "
                "by maximum-weight particle {}. Faint blue trajectories are "
                "the remaining weighted posterior—not a coordinate-wise "
                "median curve.".format(map_index)
            )
            st.plotly_chart(
                make_posterior_trajectory_figure(
                    times,
                    segment_id,
                    observed_position,
                    baseline_position,
                    posterior_position,
                    weights,
                    map_index,
                ),
                use_container_width=True,
                config=trajectory_config,
            )
            st.plotly_chart(
                make_estimated_pose_figure(
                    times,
                    segment_id,
                    observed_position,
                    observed_orientation,
                    baseline_position,
                    baseline_orientation,
                    estimated_position,
                    estimated_orientation,
                ),
                use_container_width=True,
                config=chart_config,
            )

            st.markdown(
                "#### 2. Push the parameter posterior into transforms"
            )
            reference_choice = st.radio(
                "ΔT reference trajectory",
                (
                    "Estimated nominal (posterior uncertainty)",
                    "Pre-fit baseline (estimated model correction)",
                ),
                horizontal=True,
                help=(
                    "The first view answers how uncertain the trajectory is "
                    "around the fitted model. The second shows how far the "
                    "posterior-corrected model moved from scale=1."
                ),
            )
            if reference_choice.startswith("Estimated"):
                delta_translation = bag_view["delta_translation"]
                delta_rotation = bag_view["delta_rotation_vector"]
                reference_label = "estimated nominal (MAP)"
                st.success(
                    "Showing `ΔTᵢ(t) = T_estimated(t)⁻¹ T_particle,i(t)`. "
                    "The MAP particle is exactly identity; the other "
                    "particles show posterior uncertainty around the fitted "
                    "trajectory."
                )
            else:
                delta_translation = bag_view[
                    "baseline_delta_translation"
                ]
                delta_rotation = bag_view[
                    "baseline_delta_rotation_vector"
                ]
                reference_label = "pre-fit baseline"
                st.info(
                    "Showing `ΔTᵢ(t) = T_baseline(t)⁻¹ T_particle,i(t)`. "
                    "This is the inferred correction from the original "
                    "scale=1 model, not uncertainty centred on the fit."
                )
            st.plotly_chart(
                make_uncertain_transform_time_figure(
                    times,
                    segment_id,
                    delta_translation,
                    delta_rotation,
                    weights,
                    reference_label,
                ),
                use_container_width=True,
                config=chart_config,
            )
            positive_steps = np.diff(times)
            positive_steps = positive_steps[positive_steps > 0.0]
            time_step = (
                float(np.median(positive_steps))
                if positive_steps.size
                else 0.01
            )
            selected_time = st.slider(
                "Inspect transform at bag-local time [s]",
                min_value=float(times[0]),
                max_value=float(times[-1]),
                value=float(times[len(times) // 2]),
                step=max(0.001, time_step),
                format="%.3f s",
                key="transform_time_{}_{}".format(
                    _widget_suffix(str(completed_run)),
                    bag_index,
                ),
            )
            time_index = int(np.argmin(np.abs(times - selected_time)))
            st.caption(
                "Nearest stored sample: {:.6f} s · segment {} · index {}."
                .format(
                    times[time_index],
                    int(segment_id[time_index]),
                    time_index,
                )
            )
            st.plotly_chart(
                make_transform_particle_figure(
                    delta_translation[:, time_index],
                    delta_rotation[:, time_index],
                    weights,
                    reference_label,
                ),
                use_container_width=True,
                config=trajectory_config,
            )
            st.dataframe(
                _transform_rows(
                    delta_translation[:, time_index],
                    delta_rotation[:, time_index],
                    weights,
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.plotly_chart(
                make_body_frame_particle_figure(
                    observed_position[time_index],
                    observed_orientation[time_index],
                    baseline_position[time_index],
                    baseline_orientation[time_index],
                    estimated_position[time_index],
                    estimated_orientation[time_index],
                    posterior_position[:, time_index],
                    posterior_orientation[:, time_index],
                    weights,
                    observed_position,
                ),
                use_container_width=True,
                config=trajectory_config,
            )


if __name__ == "__main__":
    main()
