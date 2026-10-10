#!/usr/bin/env python3
"""One success/failure PID comparison, shown as an exact slice and in 3D."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pid-focus-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from data_io import ROOT, CASES, Grid, LABELS, load_grid, slice_at, exposed_faces

def merge_uniform_cells(data, x_edges, y_edges):
    """Merge only exactly equal adjacent cells into axis-aligned rectangles.

    This preserves the piecewise-constant field and its nonuniform geometry,
    while avoiding thousands of identical vector patches in PDF/SVG files.
    """
    active, rectangles = {}, []
    for row, values in enumerate(data):
        ends = np.r_[0, np.flatnonzero(values[:-1] != values[1:]) + 1, len(values)]
        current = {}
        for start, stop in zip(ends[:-1], ends[1:]):
            key = (int(start), int(stop), float(values[start]))
            initial = active.pop(key)[0] if key in active else row
            current[key] = (initial, row + 1)
        for (start, stop, value), (first, last) in active.items():
            rectangles.append((first, last, start, stop, value))
        active = current
    for (start, stop, value), (first, last) in active.items():
        rectangles.append((first, last, start, stop, value))
    vertices = []
    for first, last, start, stop, value in rectangles:
        x0, x1, y0, y1 = x_edges[start], x_edges[stop], y_edges[first], y_edges[last]
        vertices.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return np.asarray(vertices), np.asarray([r[4] for r in rectangles]), rectangles



def boundary_segments(classes, x_edges, y_edges):
    segments = []
    rows, cols = np.nonzero(classes[:, :-1] != classes[:, 1:])
    for row, col in zip(rows, cols):
        x = x_edges[col + 1]
        segments.append(((x, y_edges[row]), (x, y_edges[row + 1])))
    rows, cols = np.nonzero(classes[:-1, :] != classes[1:, :])
    for row, col in zip(rows, cols):
        y = y_edges[row + 1]
        segments.append(((x_edges[col], y), (x_edges[col + 1], y)))
    return segments


def label_gain(gain):
    return "(" + ", ".join(format(float(x), ".5g") for x in gain) + ")"


def draw_point(ax, grid, name):
    gain = grid.gain(name)
    if name == "success":
        ax.scatter(gain[0], gain[2], marker="D", s=95, color="#171717", edgecolor="white", linewidth=1.3, zorder=6)
        pos = (gain[0] - 2.1, gain[2] + 4.0)
        label = "Success"
    else:
        ax.scatter(gain[0], gain[2], marker="o", s=22, color="#171717", edgecolor="white", linewidth=.65, zorder=6)
        pos = (gain[0] - 1.1, gain[2] - 3.0)
        label = "Failure"
    ax.annotate(label + "\n" + label_gain(gain), xy=(gain[0], gain[2]), xytext=pos,
                ha="center", va="center", fontsize=10.5, color="#111111", zorder=7,
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#111111", "shrinkA": 4, "shrinkB": 8},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 3})


def comparison_slice(grids, settings):
    fig, axes = plt.subplots(1, len(grids), figsize=(5.1 * len(grids), 5.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, top=.84, bottom=.22, wspace=.23)
    for ax, grid in zip(axes.flat, grids):
        index = grid.index(grid.gain("success"))[1]
        if grid.gain("recorded")[1] != grid.axes[1][index]:
            raise ValueError("Success and failure gains must lie on the same exact I plane")
        _, survival, edges = slice_at(grid, 1, index)
        green = survival >= settings["alpha"]
        vertices, values, _ = merge_uniform_cells(green.astype(float), *edges)
        mesh = PolyCollection(vertices, array=values, cmap=colors.ListedColormap([settings["red"], settings["green"]]),
                              norm=colors.Normalize(0, 1), edgecolors="none", linewidth=0, antialiased=False)
        ax.add_collection(mesh)
        ax.add_collection(LineCollection(boundary_segments(green, *edges), colors="#343434", linewidths=.85))
        for edge, setter in zip(edges, (ax.set_xlim, ax.set_ylim)):
            span = edge[-1] - edge[0]
            setter(edge[0] - .05 * span, edge[-1] + .05 * span)
        ax.set_xticks(np.linspace(edges[0][0], edges[0][-1], 5))
        ax.set_yticks([5, 10, 15, 20, 25] if grid.group == "roll_pitch" else np.linspace(edges[1][0], edges[1][-1], 5))
        ax.set_xlabel("P gain", fontsize=12)
        ax.set_ylabel("D gain", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(grid.case.replace("failure", "Failure ") + " posterior  ·  I = " + format(grid.axes[1][index], ".5g"), fontsize=12, pad=11)
        draw_point(ax, grid, "success")
        draw_point(ax, grid, "recorded")
    return fig


def comparison_volume(grids, settings):
    fig = plt.figure(figsize=(5.2 * len(grids), 5.5))
    for column, grid in enumerate(grids):
        ax = fig.add_axes([.035 + column / len(grids), .18, .88 / len(grids), .65], projection="3d", computed_zorder=False)
        fixed = grid.index(grid.gain("success"))[1]
        lower_i = grid.axes[1][fixed]
        cut = Grid(grid.case, grid.group, grid.size,
                   {"p": grid.axes[0], "i": grid.axes[1][fixed:], "d": grid.axes[2],
                    "survival_fraction": grid.survival[:, fixed:, :]}, grid.references, grid.metadata)
        faces, classes = exposed_faces(cut, settings["alpha"])
        rgba = np.empty((len(classes), 4))
        for code, name in ((2, "red"), (3, "green")):
            rgba[classes == code] = colors.to_rgba(settings[name], settings[name + "_opacity"])
        rgba[np.all(faces[:, :, 1] == lower_i, axis=1), 3] = .90
        ax.add_collection3d(Poly3DCollection(faces, facecolors=rgba, edgecolors="none", linewidth=0,
                                           zsort="average", rasterized=True, zorder=1))
        for axis, setter in zip(cut.axes, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            span = axis[-1] - axis[0]
            setter(axis[0] - .1 * span, axis[-1] + .1 * span)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.set_pane_color((1, 1, 1, 1))
            axis._axinfo["grid"].update(color=(.5, .5, .5, .25), linewidth=.5)
        ax.set_xlabel("P gain", labelpad=4)
        ax.set_ylabel("I gain", labelpad=4)
        ax.set_zlabel("D gain", labelpad=4)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        camera = settings["camera_oblique"]
        ax.view_init(elev=camera["elevation"], azim=camera["azimuth"])
        ax.set_title(grid.case.replace("failure", "Failure ") + " posterior · cut at I=1", pad=9, fontsize=12)
        for name in ("success", "recorded"):
            gain = grid.gain(name)
            ax.scatter(*gain, marker="D" if name == "success" else "o", s=55 if name == "success" else 23,
                       color="#151515", edgecolor="white", depthshade=False, linewidth=.6, zorder=6)
            x, y, _ = proj3d.proj_transform(*gain, ax.get_proj())
            offset = (-35, 33) if name == "success" else ((39, -17) if grid.case == "failure1" else (-35, -24))
            ax.annotate(("Success" if name == "success" else "Failure") + "\n" + label_gain(gain),
                        xy=(x, y), xytext=offset, textcoords="offset points", ha="center", va="center",
                        fontsize=9.5, zorder=8, annotation_clip=False,
                        arrowprops={"arrowstyle": "->", "lw": 1, "shrinkA": 4, "shrinkB": 7},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .96, "pad": 2})
    return fig


def save_comparison(fig, grids, settings, output, view):
    group = grids[0].group
    fig.suptitle(LABELS[group] + " PID gains: success and failure", y=.99, fontsize=15)
    fig.legend(handles=[Patch(facecolor=settings["green"], label="Success region (survival ≥ %.0f%%)" % (100 * settings["alpha"])),
                        Patch(facecolor=settings["red"], label="Failure region (survival < %.0f%%)" % (100 * settings["alpha"]))],
               loc="lower center", bbox_to_anchor=(.5, .08), ncol=2, frameon=False, fontsize=10.5)
    fig.text(.5, .025, "Each case uses its own posterior; other controller gains are fixed at the successful-flight values.",
             ha="center", fontsize=8.5, color="#4b535a")
    name = group + "_success_failure_" + view
    for ext in settings["formats"]:
        path = output / (name + "." + ext)
        fig.savefig(path, dpi=settings["dpi"] if ext == "png" else settings["surface_raster_dpi"], bbox_inches="tight", pad_inches=.12)
        if path.stat().st_size >= 50_000_000:
            raise ValueError("file exceeds 50 MB")
    plt.close(fig)
    return {"figure": name, "group": group, "cases": [g.case for g in grids], "view": view,
            "caption": "Recorded failure gains lie in the red region and successful-flight gains in the green region in both Roll/Pitch posterior analyses. "
                       + ("The P–D slice is exactly at the shared I=1 reference plane." if view == "PD" else "The volume is cut at I=1 to expose both reference nodes on the same classified face."),
            "references": {g.case: {name: g.mark(name) for name in ("success", "recorded")} for g in grids}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--settings", type=Path, default=ROOT / "figure_settings.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid", type=int, choices=(33, 65), default=65)
    parser.add_argument("--view", choices=("PD", "3D", "both"), default="both")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    plt.rcParams.update({"font.family": settings["font_family"], "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    output = args.output or args.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    grids = [load_grid(case, "roll_pitch", args.grid, args.root) for case in CASES]
    records = []
    for view, draw in (("PD", comparison_slice), ("3D", comparison_volume)):
        if args.view in (view, "both"):
            records.append(save_comparison(draw(grids, settings), grids, settings, output, view))
    (output / "figure_index.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "figure_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "group", "view", "cases", "caption"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in records)
    print(json.dumps({"figures": [row["figure"] for row in records], "references": records[0]["references"]}, indent=2))


if __name__ == "__main__":
    main()
