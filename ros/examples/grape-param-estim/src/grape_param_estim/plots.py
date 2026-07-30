"""Plotly figures for the selected Grape analysis interval."""

from typing import Iterable

import numpy as np

from grape_param_estim.data import AnalysisData, BagRecording
from grape_param_estim.estimator import weighted_quantile
from grape_param_estim.model import (
    ReplayResult,
    quaternion_to_matrix,
)


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


def _break_segments(values: np.ndarray, segment_id: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    boundaries = np.flatnonzero(np.diff(segment_id)) + 1
    result[boundaries] = np.nan
    return result


def _representative_particle_indices(
    weights: np.ndarray, maximum: int = 20
) -> np.ndarray:
    count = min(int(maximum), weights.size)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    targets = (np.arange(count) + 0.5) / count
    return np.unique(np.searchsorted(cumulative, targets, side="left"))


def _particle_marker(weights: np.ndarray) -> dict:
    values = np.asarray(weights, dtype=float)
    relative = values / max(float(np.max(values)), np.finfo(float).eps)
    return {
        "size": 4.0 + 10.0 * np.sqrt(relative),
        "color": np.log10(np.maximum(values, np.finfo(float).tiny)),
        "colorscale": "Viridis",
        "colorbar": {"title": "log10 weight"},
        "opacity": 0.72,
    }


def make_parameter_ridge_figure(
    particles: np.ndarray, weights: np.ndarray
):
    """Show the two structurally identifiable ratio ridges."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    values = np.asarray(particles, dtype=float)
    probability = np.asarray(weights, dtype=float)
    force_mass = values[:, 1] / values[:, 0]
    torque_inertia = values[:, 3] / values[:, 2]
    ratio_median = weighted_quantile(
        np.column_stack((force_mass, torque_inertia)),
        probability,
        (0.5,),
    )[0]
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Translation ridge: force / mass",
            "Rotation ridge: torque / inertia",
        ),
    )
    marker = _particle_marker(probability)
    figure.add_trace(
        go.Scatter(
            x=values[:, 0],
            y=values[:, 1],
            mode="markers",
            marker=marker,
            customdata=np.column_stack((probability, force_mass)),
            hovertemplate=(
                "mass=%{x:.4f}<br>force=%{y:.4f}"
                "<br>force/mass=%{customdata[1]:.4f}"
                "<br>weight=%{customdata[0]:.3g}<extra></extra>"
            ),
            name="particles",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    second_marker = dict(marker)
    second_marker.pop("colorbar")
    figure.add_trace(
        go.Scatter(
            x=values[:, 2],
            y=values[:, 3],
            mode="markers",
            marker=second_marker,
            customdata=np.column_stack((probability, torque_inertia)),
            hovertemplate=(
                "inertia=%{x:.4f}<br>torque=%{y:.4f}"
                "<br>torque/inertia=%{customdata[1]:.4f}"
                "<br>weight=%{customdata[0]:.3g}<extra></extra>"
            ),
            name="particles",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for column, x_values, ratio in (
        (1, values[:, 0], ratio_median[0]),
        (2, values[:, 2], ratio_median[1]),
    ):
        x_line = np.asarray((np.min(x_values), np.max(x_values)))
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=ratio * x_line,
                mode="lines",
                line={"color": "#ef4444", "dash": "dash"},
                name="weighted median ratio",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="mass scale", row=1, col=1)
    figure.update_yaxes(title_text="force scale", row=1, col=1)
    figure.update_xaxes(title_text="inertia scale", row=1, col=2)
    figure.update_yaxes(title_text="torque scale", row=1, col=2)
    figure.update_layout(
        height=560,
        margin={"l": 45, "r": 20, "t": 65, "b": 45},
        title=(
            "Weighted parameter posterior — median ratios "
            "F/m={:.4f}, τ/J={:.4f}".format(
                ratio_median[0], ratio_median[1]
            )
        ),
    )
    return figure


def make_posterior_trajectory_figure(
    times: np.ndarray,
    segment_id: np.ndarray,
    observed_position: np.ndarray,
    nominal_position: np.ndarray,
    posterior_position: np.ndarray,
    weights: np.ndarray,
):
    """Overlay observed, nominal, weighted samples, and posterior median."""

    import plotly.graph_objects as go

    figure = go.Figure()
    particle_indices = _representative_particle_indices(weights)
    for display_index, particle_index in enumerate(particle_indices):
        position = _break_segments(
            posterior_position[particle_index], segment_id
        )
        figure.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines",
                line={"color": "rgba(37, 99, 235, 0.18)", "width": 2},
                name="posterior particles",
                legendgroup="posterior particles",
                showlegend=display_index == 0,
                hoverinfo="skip",
            )
        )
    posterior_median = weighted_quantile(
        posterior_position, weights, (0.5,)
    )[0]
    for label, position, color, width, dash in (
        (
            "observed",
            observed_position,
            "#0f172a",
            8,
            "solid",
        ),
        (
            "nominal",
            _break_segments(nominal_position, segment_id),
            "#f97316",
            5,
            "dash",
        ),
        (
            "posterior median",
            _break_segments(posterior_median, segment_id),
            "#2563eb",
            6,
            "solid",
        ),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines",
                line={"color": color, "width": width, "dash": dash},
                name=label,
                customdata=times,
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
        title="Observed, nominal, and posterior trajectories",
        height=650,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene=_observed_scene(observed_position),
        legend={"orientation": "h"},
    )
    return figure


def make_uncertain_transform_time_figure(
    times: np.ndarray,
    segment_id: np.ndarray,
    delta_translation: np.ndarray,
    delta_rotation_vector: np.ndarray,
    weights: np.ndarray,
):
    """Plot median, 50%, and 95% intervals of transform magnitudes."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    translation_norm = np.linalg.norm(delta_translation, axis=2)
    rotation_norm = np.rad2deg(
        np.linalg.norm(delta_rotation_vector, axis=2)
    )
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "ΔT translation magnitude",
            "ΔT rotation-vector magnitude",
        ),
    )
    for row, values, unit in (
        (1, translation_norm, "m"),
        (2, rotation_norm, "deg"),
    ):
        quantile = weighted_quantile(
            values, weights, (0.025, 0.25, 0.5, 0.75, 0.975)
        )
        quantile = np.asarray(
            [_break_segments(value, segment_id) for value in quantile]
        )
        for lower, upper, color, label in (
            (quantile[0], quantile[4], "rgba(37,99,235,0.14)", "95%"),
            (quantile[1], quantile[3], "rgba(37,99,235,0.30)", "50%"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=color,
                    name="{} interval".format(label),
                    legendgroup="{} interval".format(label),
                    showlegend=row == 1,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=quantile[2],
                mode="lines",
                line={"color": "#2563eb", "width": 3},
                name="weighted median",
                legendgroup="median",
                showlegend=row == 1,
                hovertemplate="%{y:.4f} " + unit + "<extra></extra>",
            ),
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text="translation [m]", row=1, col=1)
    figure.update_yaxes(title_text="rotation [deg]", row=2, col=1)
    figure.update_xaxes(title_text="bag-local time [s]", row=2, col=1)
    figure.update_layout(
        height=680,
        margin={"l": 45, "r": 20, "t": 65, "b": 45},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def make_transform_particle_figure(
    delta_translation: np.ndarray,
    delta_rotation_vector: np.ndarray,
    weights: np.ndarray,
):
    """Scatter raw uncertain-transform particles at one selected time."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    translation = np.asarray(delta_translation, dtype=float)
    rotation = np.rad2deg(np.asarray(delta_rotation_vector, dtype=float))
    medians = (
        weighted_quantile(translation, weights, (0.5,))[0],
        weighted_quantile(rotation, weights, (0.5,))[0],
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.04,
        subplot_titles=(
            "ΔT translation particles [m]",
            "ΔT rotation-vector particles [deg]",
        ),
    )
    for column, values, median in (
        (1, translation, medians[0]),
        (2, rotation, medians[1]),
    ):
        marker = _particle_marker(weights)
        if column == 2:
            marker = dict(marker)
            marker.pop("colorbar")
        figure.add_trace(
            go.Scatter3d(
                x=values[:, 0],
                y=values[:, 1],
                z=values[:, 2],
                mode="markers",
                marker=marker,
                customdata=weights,
                name="raw particles",
                showlegend=column == 1,
                hovertemplate=(
                    "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}"
                    "<br>weight=%{customdata:.3g}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter3d(
                x=(median[0],),
                y=(median[1],),
                z=(median[2],),
                mode="markers",
                marker={"size": 8, "color": "#ef4444", "symbol": "diamond"},
                name="weighted median",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        height=600,
        margin={"l": 0, "r": 0, "t": 65, "b": 0},
        legend={"orientation": "h"},
    )
    return figure


def _add_frame_axes(
    figure,
    position: np.ndarray,
    quaternion: np.ndarray,
    length: float,
    width: float,
    opacity: float,
    legend_group: str,
    name: str,
    show_legend: bool,
) -> None:
    import plotly.graph_objects as go

    rotation = quaternion_to_matrix(quaternion)
    colors = ("#dc2626", "#16a34a", "#2563eb")
    for axis in range(3):
        end = position + length * rotation[:, axis]
        figure.add_trace(
            go.Scatter3d(
                x=(position[0], end[0]),
                y=(position[1], end[1]),
                z=(position[2], end[2]),
                mode="lines",
                line={
                    "color": colors[axis],
                    "width": width,
                },
                opacity=opacity,
                name=name,
                legendgroup=legend_group,
                showlegend=show_legend and axis == 0,
                hoverinfo="skip",
            )
        )


def make_body_frame_particle_figure(
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
    nominal_position: np.ndarray,
    nominal_orientation_xyzw: np.ndarray,
    posterior_position: np.ndarray,
    posterior_orientation_xyzw: np.ndarray,
    weights: np.ndarray,
    scene_reference_position: np.ndarray,
):
    """Show observed, nominal, and representative posterior body frames."""

    import plotly.graph_objects as go

    reference = np.asarray(scene_reference_position, dtype=float)
    span = max(float(np.max(np.ptp(reference, axis=0))), 0.20)
    axis_length = max(0.025, 0.12 * span)
    figure = go.Figure()
    representative = _representative_particle_indices(weights, maximum=12)
    figure.add_trace(
        go.Scatter3d(
            x=posterior_position[:, 0],
            y=posterior_position[:, 1],
            z=posterior_position[:, 2],
            mode="markers",
            marker=_particle_marker(weights),
            customdata=weights,
            name="posterior origins",
            hovertemplate=(
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}"
                "<br>weight=%{customdata:.3g}<extra></extra>"
            ),
        )
    )
    for display_index, particle_index in enumerate(representative):
        _add_frame_axes(
            figure,
            posterior_position[particle_index],
            posterior_orientation_xyzw[particle_index],
            axis_length,
            2,
            0.20,
            "posterior frames",
            "posterior frames",
            display_index == 0,
        )
    for name, position, quaternion, width in (
        (
            "observed frame",
            observed_position,
            observed_orientation_xyzw,
            8,
        ),
        (
            "nominal frame",
            nominal_position,
            nominal_orientation_xyzw,
            6,
        ),
    ):
        _add_frame_axes(
            figure,
            np.asarray(position),
            np.asarray(quaternion),
            1.35 * axis_length,
            width,
            1.0,
            name,
            name,
            True,
        )
    figure.update_layout(
        title="Body frames at selected time (x red, y green, z blue)",
        height=650,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene=_observed_scene(reference),
        legend={"orientation": "h"},
    )
    return figure


__all__ = [
    "make_bag_overview_figure",
    "make_body_frame_particle_figure",
    "make_correction_figure",
    "make_command_figure",
    "make_parameter_ridge_figure",
    "make_posterior_trajectory_figure",
    "make_replay_pose_figure",
    "make_replay_trajectory_figure",
    "make_segment_residual_figure",
    "make_state_figure",
    "make_transform_particle_figure",
    "make_trajectory_figure",
    "make_uncertain_transform_time_figure",
]
