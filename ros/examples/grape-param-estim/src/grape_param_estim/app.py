"""Single-page Streamlit application for selecting Grape replay data."""

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from grape_param_estim.data import (
    TOPIC_KEYS,
    load_yaml,
    read_bag,
    save_yaml,
    scan_bag_paths,
)
from grape_param_estim.plots import (
    make_correction_figure,
    make_command_figure,
    make_replay_pose_figure,
    make_replay_trajectory_figure,
    make_segment_residual_figure,
    make_state_figure,
    make_trajectory_figure,
)
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    RigidBodyParameters,
    replay_segments,
)


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


def _saved_configuration(
    bag_directory: str,
    bag_path: str,
    topics: Mapping[str, str],
    start_time: float,
    end_time: float,
    segment_duration: float,
    nominal_mass: float,
    nominal_inertia: np.ndarray,
    save_path: str,
) -> Dict[str, Any]:
    return {
        "schema": "grape_param_estim/phase1",
        "data": {
            "bag_directory": bag_directory,
            "bag_path": bag_path,
            "topics": {key: topics[key] for key in TOPIC_KEYS},
        },
        "analysis": {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "segment_duration": float(segment_duration),
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
        "output": {"config_path": save_path},
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Grape trajectory replay",
        page_icon="🍇",
        layout="wide",
    )
    st.title("Grape trajectory replay")
    st.caption(
        "Phase 0–1: inspect the selected bag data, then replay each short "
        "segment with the recorded command and a nominal rigid-body model."
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
    output_config = dict(configuration.get("output", {}))
    configured_topics = dict(data_config.get("topics", {}))

    st.sidebar.header("Data")
    st.sidebar.caption("Loaded defaults from `{}`".format(config_source))
    bag_directory = st.sidebar.text_input(
        "Bag directory",
        value=str(
            _configured_value(
                data_config,
                "bag_directory",
                "/home/leus/catkin_ws/bags/grape-drone",
            )
        ),
    )
    bag_paths = scan_bag_paths(bag_directory)
    configured_bag = str(data_config.get("bag_path", ""))
    if configured_bag and Path(configured_bag).is_file():
        configured_bag = str(Path(configured_bag).expanduser().resolve())
        if configured_bag not in bag_paths:
            bag_paths = (configured_bag,) + bag_paths
    if not bag_paths:
        st.warning("No `.bag` file was found below `{}`.".format(bag_directory))
        st.stop()
    selected_index = (
        bag_paths.index(configured_bag)
        if configured_bag in bag_paths
        else 0
    )
    bag_path = st.sidebar.selectbox(
        "Bag",
        bag_paths,
        index=selected_index,
        format_func=lambda path: "{}/{}".format(
            Path(path).parent.name, Path(path).name
        ),
    )

    default_topics = {
        "command": "/gimbalrotor/four_axes/command",
        "gimbal": "/gimbalrotor/gimbals_ctrl",
        "imu": "/gimbalrotor/sensor_plugin/imu1/ros_converted",
        "odometry": "/gimbalrotor/uav/cog/odom",
        "flight_state": "/gimbalrotor/flight_state",
    }
    topics = {}
    with st.sidebar.expander("Topics"):
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
    minimum_window = min(0.05, 0.25 * (available_end - available_start))
    configured_start = float(
        _configured_value(analysis_config, "start_time", available_start)
    )
    configured_end = float(
        _configured_value(analysis_config, "end_time", available_end)
    )
    configured_start = float(
        np.clip(
            configured_start,
            available_start,
            available_end - minimum_window,
        )
    )
    configured_end = float(
        np.clip(
            configured_end,
            configured_start + minimum_window,
            available_end,
        )
    )

    st.sidebar.header("Analysis interval")
    start_time = st.sidebar.number_input(
        "Start [bag-local s]",
        min_value=float(available_start),
        max_value=float(available_end - minimum_window),
        value=configured_start,
        step=0.05,
        format="%.3f",
    )
    end_time = st.sidebar.number_input(
        "End [bag-local s]",
        min_value=float(start_time + minimum_window),
        max_value=float(available_end),
        value=max(configured_end, float(start_time + minimum_window)),
        step=0.05,
        format="%.3f",
    )
    segment_duration = st.sidebar.number_input(
        "Segment length [s]",
        min_value=0.05,
        max_value=max(0.05, float(end_time - start_time)),
        value=float(
            np.clip(
                float(
                    _configured_value(
                        analysis_config, "segment_duration", 0.75
                    )
                ),
                0.05,
                max(0.05, float(end_time - start_time)),
            )
        ),
        step=0.05,
        format="%.3f",
    )

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
    current_configuration = _saved_configuration(
        bag_directory,
        bag_path,
        topics,
        start_time,
        end_time,
        segment_duration,
        nominal_mass,
        nominal_inertia,
        save_path,
    )
    if st.sidebar.button("Save analysis YAML", type="primary"):
        try:
            written_path = save_yaml(save_path, current_configuration)
        except Exception as exc:
            st.sidebar.error("Could not save YAML: {}".format(exc))
        else:
            st.sidebar.success("Saved `{}`".format(written_path))

    data_tab, replay_tab = st.tabs(("Data", "Nominal replay"))
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
            "Dotted vertical lines are short-replay segment boundaries. "
            "Large motion and post-contact samples are intentionally not "
            "hidden."
        )
        st.plotly_chart(
            make_trajectory_figure(analysis), use_container_width=True
        )
        st.plotly_chart(
            make_state_figure(analysis), use_container_width=True
        )
        st.plotly_chart(
            make_command_figure(analysis), use_container_width=True
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
            "The SE(3) residual is "
            "`Log(T_obs,rel⁻¹ T_nom,rel)`. The correction plot shows "
            "`T_nom⁻¹ T_obs`."
        )
        st.plotly_chart(
            make_replay_trajectory_figure(analysis, replay),
            use_container_width=True,
        )
        st.plotly_chart(
            make_replay_pose_figure(analysis, replay),
            use_container_width=True,
        )
        st.plotly_chart(
            make_correction_figure(analysis, replay),
            use_container_width=True,
        )
        st.plotly_chart(
            make_segment_residual_figure(analysis, replay),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
