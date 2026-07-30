"""Self-contained browser report for automatic failed-bag analysis."""

import html
import io
import os
from pathlib import Path
import tempfile
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np


AXIS_COLORS = {
    "x": "#1f77b4",
    "y": "#ff7f0e",
    "z": "#2ca02c",
    "roll": "#d62728",
    "pitch": "#9467bd",
    "yaw": "#8c564b",
}


def _svg(figure) -> str:
    stream = io.StringIO()
    figure.savefig(stream, format="svg", bbox_inches="tight")
    plt.close(figure)
    value = stream.getvalue()
    return value[value.find("<svg") :]


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
        for interval in episode["selection"][
            "failure_diagnostic_intervals"
        ]:
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
        if episode["liftoff_s"] is not None:
            for axis in axes:
                axis.axvline(
                    episode["liftoff_s"],
                    color="#2a9d8f",
                    linestyle=":",
                    linewidth=1.0,
                )


def _timeline_svg(bag: Mapping) -> str:
    plot = bag["plot"]
    time = np.asarray(plot["time_s"], dtype=float)
    command = np.asarray(
        [
            np.nan if value is None else float(value)
            for value in plot["vertical_command"]
        ]
    )
    figure, axes = plt.subplots(
        6, 1, figsize=(11.0, 9.5), sharex=True
    )
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
    return _svg(figure)


def _parameter_trace_svg(result: Mapping) -> str:
    groups = (
        ("gain", "effective gain"),
        ("velocity_feedback", "velocity feedback [1/s]"),
        ("bias", "bias"),
    )
    figure, axes = plt.subplots(
        len(groups), 2, figsize=(11.0, 9.5), sharex=True
    )
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
            for axis_index, (suffix, _) in enumerate(groups):
                for name in names:
                    if not name.endswith("_{}".format(suffix)):
                        continue
                    stem = name[
                        : -(len(suffix) + 1)
                    ]
                    physical_axis = stem.rsplit("_", 1)[-1]
                    column = (
                        0
                        if physical_axis in ("x", "y", "z")
                        else 1
                    )
                    values = [
                        row["parameters"][name] for row in rows
                    ]
                    label = physical_axis
                    axes[axis_index, column].plot(
                        time,
                        values,
                        color=AXIS_COLORS.get(
                            physical_axis, "#333333"
                        ),
                        linewidth=1.0,
                        label=(
                            label
                            if (axis_index, column, label)
                            not in labels_seen
                            else None
                        ),
                    )
                    labels_seen.add(
                        (axis_index, column, label)
                    )
                    plotted = True
    for row, (_, ylabel) in enumerate(groups):
        for column, prefix in enumerate(
            ("translational", "rotational")
        ):
            axis = axes[row, column]
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
        "Cumulative parameter estimates within each independent episode"
    )
    figure.tight_layout()
    return _svg(figure)


def _residual_svg(episode: Mapping) -> str:
    diagnostics = episode["model_diagnostics"]
    if diagnostics is None:
        return ""
    figure, axis = plt.subplots(1, 1, figsize=(10.5, 2.7))
    time = diagnostics["timestamps_s"]
    score = diagnostics["residual_score"]
    axis.plot(time, score, linewidth=0.9, color="#e45756")
    axis.axhline(
        diagnostics["standardized_threshold"],
        linestyle="--",
        color="#333333",
        linewidth=0.9,
        label="persistent mismatch threshold",
    )
    axis.set_xlabel("bag-local time [s]")
    axis.set_ylabel("standardized residual")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _svg(figure)


def _parameter_table(episode: Mapping) -> str:
    estimate = episode["estimate"]
    if estimate is None:
        return "<p>推定値なし</p>"
    rows = []
    for axis, diagnostics in estimate["channels"].items():
        name = diagnostics["gain_parameter"]
        parameter = estimate["parameters"][name]
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{:.6g}</td>"
            "<td>[{:.6g}, {:.6g}]</td><td>{}</td></tr>".format(
                html.escape(axis),
                html.escape(name),
                parameter["estimate"],
                parameter["ci95"][0],
                parameter["ci95"][1],
                html.escape(diagnostics["information_grade"]),
            )
        )
    return (
        "<table><thead><tr><th>axis</th><th>parameter</th>"
        "<th>estimate</th><th>block-bootstrap 95%</th>"
        "<th>grade</th></tr></thead><tbody>{}</tbody></table>".format(
            "".join(rows)
        )
    )


def _episode_html(episode: Mapping) -> str:
    support = episode["support"]["height_m"]
    support_text = "unavailable" if support is None else "{:.4f} m".format(
        support
    )
    liftoff = episode["liftoff_s"]
    liftoff_text = "not detected" if liftoff is None else "{:.3f} s".format(
        liftoff
    )
    diagnostics = "".join(
        "<li>{:.3f}–{:.3f} s: {}</li>".format(
            row["start_s"],
            row["end_s"],
            html.escape(row["reason"]),
        )
        for row in episode["selection"][
            "failure_diagnostic_intervals"
        ]
    )
    estimate = episode["estimate"]
    lag = (
        ""
        if estimate is None
        else "<p>Selected alignment lag: {:.3f} s; fit samples: {}</p>".format(
            estimate["selected_alignment_lag_s"],
            estimate["fit_sample_count"],
        )
    )
    return """
      <details class="episode">
        <summary>Episode {index}: {status} ({start:.3f}–{end:.3f} s)</summary>
        <p>Reason: <code>{reason}</code>; support: {support};
           liftoff: {liftoff}; states: {states}</p>
        {lag}
        <h4>Effective parameters</h4>
        {table}
        <h4>Failure diagnostic intervals</h4>
        <ul>{diagnostics}</ul>
        {residual}
      </details>
    """.format(
        index=episode["episode_index"],
        status=html.escape(episode["status"]),
        start=episode["start_s"],
        end=episode["end_s"],
        reason=html.escape(episode["reason"]),
        support=support_text,
        liftoff=liftoff_text,
        states=", ".join(str(value) for value in episode["flight_states"]),
        lag=lag,
        table=_parameter_table(episode),
        diagnostics=diagnostics or "<li>none</li>",
        residual=_residual_svg(episode),
    )


def render_browser_report(
    result: Mapping,
    destination,
    overwrite: bool = False,
) -> Path:
    """Write one offline HTML report with inline SVG plots."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    bags = []
    for bag in result["bags"]:
        episodes = "".join(
            _episode_html(episode) for episode in bag["episodes"]
        )
        bags.append(
            """
            <details class="bag" open>
              <summary>Bag {index}: {name} — {count} episode(s)</summary>
              <p><code>{path}</code><br>SHA-256: <code>{sha}</code></p>
              {timeline}
              {episodes}
            </details>
            """.format(
                index=bag["bag_index"],
                name=html.escape(Path(bag["path"]).name),
                count=bag["episode_count"],
                path=html.escape(bag["path"]),
                sha=html.escape(bag["sha256"]),
                timeline=_timeline_svg(bag),
                episodes=episodes or "<p>No active episode detected.</p>",
            )
        )
    document = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grape failure-bag automatic analysis</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 2rem auto;
            padding: 0 1rem; color: #202124; }}
    summary {{ cursor: pointer; font-weight: 650; padding: .5rem 0; }}
    .bag {{ border-top: 1px solid #ddd; margin-top: 1rem; }}
    .episode {{ margin: .5rem 1rem 1rem; padding-left: .75rem;
                border-left: 3px solid #ddd; }}
    svg {{ width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: .35rem .5rem;
              text-align: left; }}
    th {{ background: #f4f5f7; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Grape failure-bag automatic analysis</h1>
  <p>{bags} bag(s), concatenated trial time {duration:.3f} s.</p>
  <p>{interpretation}</p>
  <h2>Parameter evolution</h2>
  {parameter_trace}
  <h2>Bag timelines</h2>
  <p>Blue shading: fit interval; red shading: failure diagnostic interval;
     green dotted line: detected liftoff.</p>
  {bag_sections}
</body>
</html>
""".format(
        bags=result["bag_count"],
        duration=result["sequence_duration_s"],
        interpretation=html.escape(result["interpretation"]),
        parameter_trace=_parameter_trace_svg(result),
        bag_sections="".join(bags),
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


__all__ = ["render_browser_report"]
