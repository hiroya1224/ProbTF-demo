"""Plotly figures for the selected Grape analysis interval."""

from typing import Iterable

import numpy as np

from grape_param_estim.data import AnalysisData


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
        scene={
            "xaxis_title": "world x [m]",
            "yaxis_title": "world y [m]",
            "zaxis_title": "world z [m]",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
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
        subplot_titles=("Recorded base thrust", "Recorded gimbal angle"),
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
                y=np.rad2deg(data.gimbal_angle[:, rotor]),
                mode="lines",
                line={"color": color},
                name="gimbal {}".format(rotor + 1),
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


__all__ = [
    "make_command_figure",
    "make_state_figure",
    "make_trajectory_figure",
]
