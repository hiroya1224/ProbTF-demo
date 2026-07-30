"""Matplotlib figures used by the interactive failure-analysis GUI."""

from pathlib import Path
from typing import Mapping

import numpy as np


AXIS_COLORS = {
    "x": "#1f77b4",
    "y": "#ff7f0e",
    "z": "#2ca02c",
    "roll": "#d62728",
    "pitch": "#9467bd",
    "yaw": "#8c564b",
}


def draw_placeholder(figure, text: str) -> None:
    figure.clear()
    axis = figure.subplots(1, 1)
    axis.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_axis_off()
    figure.tight_layout()


def _shade_selection(axes, episodes) -> None:
    for episode in episodes:
        for interval in episode["selection"]["fit_intervals"]:
            for axis in axes:
                axis.axvspan(
                    interval["start_s"],
                    interval["end_s"],
                    color="#4c78a8",
                    alpha=0.12,
                )
        diagnostics = episode["selection"][
            "failure_diagnostic_intervals"
        ]
        for interval in diagnostics:
            for axis in axes:
                axis.axvspan(
                    interval["start_s"],
                    interval["end_s"],
                    color="#e45756",
                    alpha=0.10,
                )
        support = episode["support"]["height_m"]
        if support is not None:
            axes[0].hlines(
                support,
                episode["start_s"],
                episode["end_s"],
                color="#666666",
                linestyle="--",
                linewidth=0.8,
            )
        liftoff = episode["liftoff_s"]
        if liftoff is not None:
            for axis in axes:
                axis.axvline(
                    liftoff,
                    color="#2a9d8f",
                    linestyle=":",
                    linewidth=1.0,
                )


def draw_timeline(figure, bag: Mapping) -> None:
    """Draw one bag timeline into an existing interactive Figure."""

    figure.clear()
    plot = bag["plot"]
    time = np.asarray(plot["time_s"], dtype=float)
    command = np.asarray(
        [
            np.nan if value is None else float(value)
            for value in plot["vertical_command"]
        ],
        dtype=float,
    )
    axes = figure.subplots(6, 1, sharex=True)
    axes[0].plot(time, plot["z_m"], linewidth=0.9)
    axes[0].set_ylabel("z [m]")
    axes[1].plot(time, plot["speed_m_s"], linewidth=0.9)
    axes[1].set_ylabel("speed [m/s]")
    axes[2].plot(
        time, plot["specific_force_norm_m_s2"], linewidth=0.9
    )
    axes[2].set_ylabel("|specific force|")
    axes[3].plot(
        time,
        plot["angular_velocity_norm_rad_s"],
        linewidth=0.9,
    )
    axes[3].set_ylabel("|angular rate|")
    axes[4].plot(time, command, linewidth=0.9)
    axes[4].set_ylabel("command Fz")
    axes[5].step(
        time, plot["flight_state"], where="post", linewidth=0.9
    )
    axes[5].set_ylabel("flight state")
    axes[5].set_xlabel("bag-local time [s]")
    _shade_selection(axes, bag["episodes"])
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(Path(bag["path"]).name)
    figure.tight_layout()


def draw_parameter_trace(figure, result: Mapping) -> None:
    """Draw cumulative parameter traces from all completed bags."""

    figure.clear()
    groups = (
        ("gain", "effective gain"),
        ("velocity_feedback", "velocity feedback [1/s]"),
        ("bias", "bias"),
    )
    axes = figure.subplots(len(groups), 2, sharex=True)
    plotted = False
    labels_seen = set()
    for bag in result["bags"]:
        for episode in bag["episodes"]:
            rows = episode["parameter_trace"]
            if not rows:
                continue
            time = np.asarray(
                [row["sequence_time_s"] for row in rows],
                dtype=float,
            )
            names = sorted(rows[0]["parameters"])
            for row_index, (suffix, _) in enumerate(groups):
                for name in names:
                    if not name.endswith("_{}".format(suffix)):
                        continue
                    stem = name[: -(len(suffix) + 1)]
                    physical_axis = stem.rsplit("_", 1)[-1]
                    column = (
                        0
                        if physical_axis in ("x", "y", "z")
                        else 1
                    )
                    label_key = (
                        row_index,
                        column,
                        physical_axis,
                    )
                    axes[row_index, column].plot(
                        time,
                        [row["parameters"][name] for row in rows],
                        color=AXIS_COLORS.get(
                            physical_axis, "#333333"
                        ),
                        linewidth=1.0,
                        label=(
                            physical_axis
                            if label_key not in labels_seen
                            else None
                        ),
                    )
                    labels_seen.add(label_key)
                    plotted = True
    for row_index, (_, ylabel) in enumerate(groups):
        for column, prefix in enumerate(
            ("translational", "rotational")
        ):
            axis = axes[row_index, column]
            axis.set_ylabel("{} {}".format(prefix, ylabel))
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend(ncol=3, fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("concatenated trial time [s]")
    axes[-1, 1].set_xlabel("concatenated trial time [s]")
    if not plotted:
        axes[0, 0].text(
            0.5,
            0.5,
            "No identifiable parameter trace",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
    figure.suptitle(
        "Cumulative estimates within each independent episode"
    )
    figure.tight_layout()


def parameter_rows(episode: Mapping) -> tuple:
    """Return table-ready final parameter rows for one episode."""

    estimate = episode["estimate"]
    if estimate is None:
        return ()
    rows = []
    for axis, diagnostics in estimate["channels"].items():
        gain_name = diagnostics["gain_parameter"]
        prefix = gain_name[: -len("_gain")]
        names = (
            "{}_bias".format(prefix),
            gain_name,
            "{}_velocity_feedback".format(prefix),
        )
        for name in names:
            parameter = estimate["parameters"][name]
            interval = parameter["ci95"]
            rows.append(
                (
                    axis,
                    name,
                    "{:.6g}".format(parameter["estimate"]),
                    "[{:.6g}, {:.6g}]".format(
                        interval[0], interval[1]
                    ),
                    (
                        diagnostics["information_grade"]
                        if name == gain_name
                        else ""
                    ),
                )
            )
    return tuple(rows)


__all__ = [
    "draw_parameter_trace",
    "draw_placeholder",
    "draw_timeline",
    "parameter_rows",
]
