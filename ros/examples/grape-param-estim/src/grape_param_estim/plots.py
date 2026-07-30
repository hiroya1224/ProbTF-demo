"""Plotly figures for the selected Grape analysis interval."""

from typing import Iterable

import numpy as np

from grape_param_estim.data import AnalysisData, BagRecording
from grape_param_estim.model import ReplayResult


AXES = ("x", "y", "z")
ROTOR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def _quaternion_to_euler_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized ``[x, y, z, w]`` quaternions to XYZ Euler angles."""

    q = np.asarray(quaternion, dtype=float)
    x_value = q[:, 0]
    y_value = q[:, 1]
    z_value = q[:, 2]
    w_value = q[:, 3]
    roll = np.arctan2(
        2.0 * (w_value * x_value + y_value * z_value),
        1.0 - 2.0 * (x_value * x_value + y_value * y_value),
    )
    pitch_argument = 2.0 * (
        w_value * y_value - z_value * x_value
    )
    pitch = np.arcsin(np.clip(pitch_argument, -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (w_value * z_value + x_value * y_value),
        1.0 - 2.0 * (y_value * y_value + z_value * z_value),
    )
    return np.column_stack((roll, pitch, yaw))


def _segment_boundaries(data: AnalysisData) -> Iterable[float]:
    for _, segment in data.segments():
        if segment.start:
            yield float(data.times[segment.start])


def _add_segment_lines(figure, data: AnalysisData, rows: int) -> None:
    for boundary in _segment_boundaries(data):
        for row in range(1, rows + 1):
            figure.add_vline(
                x=boundary,
                line_width=1,
                line_dash="dot",
                line_color="rgba(100, 116, 139, 0.55)",
                row=row,
                col=1,
            )


def _decimated_indices(size: int, maximum: int = 5000) -> np.ndarray:
    stride = max(1, int(np.ceil(float(size) / float(maximum))))
    indices = np.arange(0, size, stride, dtype=int)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


def _observed_scene(position: np.ndarray) -> dict:
    """Return a cubic scene centred and scaled only from observed motion."""

    values = np.asarray(position, dtype=float)
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    centre = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 0.20)
    half = 0.60 * span
    return {
        "xaxis": {
            "title": "world x [m]",
            "range": [centre[0] - half, centre[0] + half],
        },
        "yaxis": {
            "title": "world y [m]",
            "range": [centre[1] - half, centre[1] + half],
        },
        "zaxis": {
            "title": "world z [m]",
            "range": [centre[2] - half, centre[2] + half],
        },
        "aspectmode": "cube",
    }


def make_bag_overview_figure(
    recording: BagRecording,
    selected_interval,
):
    """Show the whole bag and expose its x values for box selection."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=(
            "CoG position in world",
            "Physical baselink orientation",
            "CoG linear / body angular velocity",
            "Flight state",
        ),
    )
    state_indices = _decimated_indices(recording.state_times.size)
    body_indices = _decimated_indices(recording.body_times.size)
    flight_indices = _decimated_indices(recording.flight_state_times.size)
    body_euler = np.rad2deg(
        _quaternion_to_euler_xyzw(
            recording.body_orientation_xyzw[body_indices]
        )
    )
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for column, axis in enumerate(AXES):
        figure.add_trace(
            go.Scattergl(
                x=recording.state_times[state_indices],
                y=recording.position[state_indices, column],
                mode="lines",
                line={"color": colors[column]},
                name="p{}".format(axis),
                legendgroup="p{}".format(axis),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=recording.body_times[body_indices],
                y=body_euler[:, column],
                mode="lines",
                line={"color": colors[column]},
                name=("roll", "pitch", "yaw")[column],
                legendgroup=("roll", "pitch", "yaw")[column],
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=recording.state_times[state_indices],
                y=recording.linear_velocity[state_indices, column],
                mode="lines",
                line={"color": colors[column]},
                name="v{}".format(axis),
                legendgroup="v{}".format(axis),
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=recording.body_times[body_indices],
                y=recording.body_angular_velocity[body_indices, column],
                mode="lines",
                line={"color": colors[column], "dash": "dot"},
                name="ω{}".format(axis),
                legendgroup="omega{}".format(axis),
            ),
            row=3,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=recording.flight_state_times[flight_indices],
            y=recording.flight_state[flight_indices],
            mode="lines",
            line={"shape": "hv", "color": "#475569"},
            name="flight state",
        ),
        row=4,
        col=1,
    )
    start, end = (float(value) for value in selected_interval)
    for row in range(1, 5):
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(249, 115, 22, 0.16)",
            line_width=1,
            line_color="rgba(249, 115, 22, 0.8)",
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text="position [m]", row=1, col=1)
    figure.update_yaxes(title_text="angle [deg]", row=2, col=1)
    figure.update_yaxes(title_text="m/s, rad/s", row=3, col=1)
    figure.update_yaxes(title_text="state", row=4, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=4, col=1)
    figure.update_layout(
        title=(
            "Whole-bag overview — drag horizontally on any panel "
            "to select an interval"
        ),
        height=830,
        margin={"l": 45, "r": 20, "t": 80, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
    )
    return figure


def make_trajectory_figure(data: AnalysisData):
    """Return a 3-D observed trajectory split into replay segments."""

    import plotly.graph_objects as go

    figure = go.Figure()
    for segment_id, segment in data.segments():
        position = data.position[segment]
        figure.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines+markers",
                marker={"size": 2},
                line={"width": 4},
                name="segment {}".format(segment_id),
                customdata=data.times[segment],
                hovertemplate=(
                    "t=%{customdata:.3f} s"
                    "<br>x=%{x:.4f} m"
                    "<br>y=%{y:.4f} m"
                    "<br>z=%{z:.4f} m<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Observed body pose in world",
        height=560,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene=_observed_scene(data.position),
        legend={"orientation": "h"},
        uirevision=data.bag_path,
    )
    return figure


def make_state_figure(data: AnalysisData):
    """Return position, orientation, velocity, and flight-state traces."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Position in world",
            "Orientation (XYZ Euler, display only)",
            "Linear / angular velocity",
            "Flight state",
        ),
    )
    euler = np.rad2deg(
        _quaternion_to_euler_xyzw(data.orientation_xyzw)
    )
    for column, axis in enumerate(AXES):
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=data.position[:, column],
                mode="lines",
                name="p{}".format(axis),
                legendgroup="position",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=euler[:, column],
                mode="lines",
                name="{} [deg]".format(("roll", "pitch", "yaw")[column]),
                legendgroup="orientation",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=data.linear_velocity[:, column],
                mode="lines",
                name="v{}".format(axis),
                legendgroup="velocity",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=data.angular_velocity[:, column],
                mode="lines",
                line={"dash": "dot"},
                name="ω{}".format(axis),
                legendgroup="angular_velocity",
            ),
            row=3,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=data.times,
            y=data.flight_state,
            mode="lines",
            line={"shape": "hv", "color": "#475569"},
            name="flight state",
        ),
        row=4,
        col=1,
    )
    _add_segment_lines(figure, data, rows=4)
    figure.update_yaxes(title_text="position [m]", row=1, col=1)
    figure.update_yaxes(title_text="angle [deg]", row=2, col=1)
    figure.update_yaxes(title_text="m/s, rad/s", row=3, col=1)
    figure.update_yaxes(title_text="state", row=4, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=4, col=1)
    figure.update_layout(
        height=900,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def make_command_figure(data: AnalysisData):
    """Return the recorded rotor thrust and gimbal-angle command."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Recorded base thrust command",
            "Gimbal target and measured physical angle",
        ),
    )
    for rotor in range(4):
        color = ROTOR_COLORS[rotor]
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=data.base_thrust[:, rotor],
                mode="lines",
                line={"color": color},
                name="rotor {}".format(rotor + 1),
                legendgroup="rotor{}".format(rotor),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=np.rad2deg(data.gimbal_target_angle[:, rotor]),
                mode="lines",
                line={"color": color, "dash": "dot"},
                name="gimbal {} target".format(rotor + 1),
                legendgroup="rotor{}".format(rotor),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=np.rad2deg(data.gimbal_measured_angle[:, rotor]),
                mode="lines",
                line={"color": color},
                name="gimbal {} measured".format(rotor + 1),
                legendgroup="rotor{}".format(rotor),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    _add_segment_lines(figure, data, rows=2)
    figure.update_yaxes(title_text="command force", row=1, col=1)
    figure.update_yaxes(title_text="angle [deg]", row=2, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=2, col=1)
    figure.update_layout(
        height=620,
        margin={"l": 40, "r": 20, "t": 65, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def make_replay_trajectory_figure(
    data: AnalysisData, replay: ReplayResult
):
    """Overlay observed and nominal world trajectories without reset jumps."""

    import plotly.graph_objects as go

    figure = go.Figure()
    for segment_id, segment in data.segments():
        observed = data.position[segment]
        nominal = replay.position[segment]
        for label, position, color, dash, width in (
            ("observed", observed, "#0f172a", "solid", 7),
            ("nominal", nominal, "#f97316", "dash", 4),
        ):
            figure.add_trace(
                go.Scatter3d(
                    x=position[:, 0],
                    y=position[:, 1],
                    z=position[:, 2],
                    mode="lines",
                    line={"width": width, "color": color, "dash": dash},
                    name=label,
                    legendgroup=label,
                    showlegend=segment_id == 0,
                    customdata=data.times[segment],
                    hovertemplate=(
                        label
                        + "<br>t=%{customdata:.3f} s"
                        + "<br>x=%{x:.4f} m"
                        + "<br>y=%{y:.4f} m"
                        + "<br>z=%{z:.4f} m<extra></extra>"
                    ),
                )
            )
    figure.update_layout(
        title="Observed and nominal body trajectory in world",
        height=620,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene=_observed_scene(data.position),
        legend={"orientation": "h"},
        uirevision=data.bag_path,
    )
    return figure


def _nominal_with_segment_breaks(
    data: AnalysisData, values: np.ndarray
) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for segment_id, segment in data.segments():
        if segment_id:
            result[segment.start] = np.nan
    return result


def make_replay_pose_figure(
    data: AnalysisData, replay: ReplayResult
):
    """Compare observed and nominal position and world orientation."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Position in world",
            "Orientation in world (XYZ Euler, display only)",
        ),
    )
    observed_euler = np.rad2deg(
        _quaternion_to_euler_xyzw(data.orientation_xyzw)
    )
    nominal_euler = np.rad2deg(
        _quaternion_to_euler_xyzw(replay.orientation_xyzw)
    )
    nominal_position = _nominal_with_segment_breaks(
        data, replay.position
    )
    nominal_euler = _nominal_with_segment_breaks(data, nominal_euler)
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for column, axis in enumerate(AXES):
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=data.position[:, column],
                mode="lines",
                line={"color": colors[column]},
                name="observed p{}".format(axis),
                legendgroup="p{}".format(axis),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=nominal_position[:, column],
                mode="lines",
                line={"color": colors[column], "dash": "dot"},
                name="nominal p{}".format(axis),
                legendgroup="p{}".format(axis),
            ),
            row=1,
            col=1,
        )
        angle_name = ("roll", "pitch", "yaw")[column]
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=observed_euler[:, column],
                mode="lines",
                line={"color": colors[column]},
                name="observed {}".format(angle_name),
                legendgroup=angle_name,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=nominal_euler[:, column],
                mode="lines",
                line={"color": colors[column], "dash": "dot"},
                name="nominal {}".format(angle_name),
                legendgroup=angle_name,
            ),
            row=2,
            col=1,
        )
    _add_segment_lines(figure, data, rows=2)
    figure.update_yaxes(title_text="position [m]", row=1, col=1)
    figure.update_yaxes(title_text="angle [deg]", row=2, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=2, col=1)
    figure.update_layout(
        height=720,
        margin={"l": 40, "r": 20, "t": 65, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def make_correction_figure(
    data: AnalysisData, replay: ReplayResult
):
    """Plot ``T_nominal^-1 T_observed`` translation and rotation vector."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Correction translation in nominal body",
            "Correction rotation vector",
        ),
    )
    rotation_degrees = np.rad2deg(replay.correction_rotation_vector)
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for column, axis in enumerate(AXES):
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=replay.correction_translation[:, column],
                mode="lines",
                line={"color": colors[column]},
                name="Δt{}".format(axis),
                legendgroup=axis,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.times,
                y=rotation_degrees[:, column],
                mode="lines",
                line={"color": colors[column]},
                name="Δr{}".format(axis),
                legendgroup=axis,
            ),
            row=2,
            col=1,
        )
    _add_segment_lines(figure, data, rows=2)
    figure.update_yaxes(title_text="translation [m]", row=1, col=1)
    figure.update_yaxes(title_text="rotation vector [deg]", row=2, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=2, col=1)
    figure.update_layout(
        height=680,
        margin={"l": 40, "r": 20, "t": 65, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def make_segment_residual_figure(
    data: AnalysisData, replay: ReplayResult
):
    """Compare SE(3) residual growth for every short segment."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "SE(3) translation-coordinate norm",
            "SO(3) rotation-vector norm",
        ),
    )
    for segment_id, segment in data.segments():
        local_time = data.times[segment] - data.times[segment.start]
        color = ROTOR_COLORS[segment_id % len(ROTOR_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=local_time,
                y=replay.translation_residual_norm[segment],
                mode="lines",
                line={"color": color},
                name="segment {}".format(segment_id),
                legendgroup="segment{}".format(segment_id),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=local_time,
                y=np.rad2deg(replay.rotation_residual_norm[segment]),
                mode="lines",
                line={"color": color},
                name="segment {}".format(segment_id),
                legendgroup="segment{}".format(segment_id),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="translation [m]", row=1, col=1)
    figure.update_yaxes(title_text="rotation [deg]", row=2, col=1)
    figure.update_xaxes(title_text="segment-local time [s]", row=2, col=1)
    figure.update_layout(
        height=680,
        margin={"l": 40, "r": 20, "t": 65, "b": 40},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


__all__ = [
    "make_bag_overview_figure",
    "make_correction_figure",
    "make_command_figure",
    "make_replay_pose_figure",
    "make_replay_trajectory_figure",
    "make_segment_residual_figure",
    "make_state_figure",
    "make_trajectory_figure",
]
