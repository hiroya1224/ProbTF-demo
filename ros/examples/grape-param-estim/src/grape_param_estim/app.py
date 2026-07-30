"""Single-page Streamlit application for selecting Grape replay data."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from grape_param_estim.data import (
    TOPIC_KEYS,
    load_yaml,
    read_bag,
    save_yaml,
    scan_bag_paths,
    suggest_analysis_interval,
)
from grape_param_estim.plots import (
    make_bag_overview_figure,
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


def _widget_suffix(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _bag_label(path: str, search_root: str) -> str:
    try:
        return str(
            Path(path).resolve().relative_to(Path(search_root).resolve())
        )
    except (OSError, ValueError):
        return str(Path(path))


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
        "Phase 0–1: choose motion from a whole-bag overview, then replay each "
        "short segment with the recorded command and a nominal rigid body."
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
    configured_bag = str(data_config.get("bag_path", ""))
    if configured_bag and Path(configured_bag).is_file():
        configured_bag = str(Path(configured_bag).expanduser().resolve())
    if not bag_paths:
        st.warning("No `.bag` file was found below `{}`.".format(bag_directory))
        st.stop()
    selected_index = (
        bag_paths.index(configured_bag)
        if configured_bag in bag_paths
        else 0
    )
    bag_path = st.sidebar.selectbox(
        "Bag file",
        bag_paths,
        index=selected_index,
        format_func=lambda path: _bag_label(path, bag_directory),
    )
    st.sidebar.caption(
        "{} bag(s) found recursively. Times below are local to the "
        "selected bag.".format(len(bag_paths))
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

    configured_duration = max(
        minimum_window, configured_end - configured_start
    )
    suggested_interval = suggest_analysis_interval(
        recording, configured_duration
    )
    initial_interval = (
        (configured_start, configured_end)
        if bag_path == configured_bag
        else suggested_interval
    )
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
        "The orange band is the current interval.".format(
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
        "Selected {:.3f}–{:.3f} s ({:.3f} s).".format(
            start_time, end_time, end_time - start_time
        )
    )
    segment_key = "segment_duration_{}".format(suffix)
    segment_default = float(
        np.clip(
            float(
                _configured_value(
                    analysis_config, "segment_duration", 0.75
                )
            ),
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

    st.subheader("2. Inspect the selected interval")
    data_tab, replay_tab = st.tabs(("Selected data", "Nominal replay"))
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


if __name__ == "__main__":
    main()
