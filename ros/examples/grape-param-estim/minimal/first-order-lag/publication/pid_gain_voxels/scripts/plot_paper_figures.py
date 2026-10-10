#!/usr/bin/env python3
"""Publication figures from portable node arrays, with editable settings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pid-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from data_io import (ROOT, CASES, GROUPS, LABELS, GAIN_NAMES, load_grid, exposed_faces,
                     slice_at, marks_in_slice)


def fmt(value):
    return format(float(value), ".5g")


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


class PaperFigures:
    def __init__(self, root, output, settings, size=65):
        self.root, self.output, self.settings, self.size = root, output, settings, size
        self.records = []
        self.geometry = {}
        plt.rcParams.update({"font.family": settings["font_family"], "font.size": settings["font_size"],
                             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.titleweight": "normal", "savefig.facecolor": "white"})

    def legend(self, fig, continuous=False, y=.07):
        cfg = self.settings
        handles = [] if continuous else [Patch(facecolor=cfg["green"], label="Survival >= %s%%" % fmt(100 * cfg["alpha"])),
                                         Patch(facecolor=cfg["red"], label="Survival < %s%%" % fmt(100 * cfg["alpha"]))]
        handles += [Line2D([], [], marker="D", markersize=6, markerfacecolor=cfg["success_color"],
                           markeredgecolor="white", linestyle="none", label="Successful-flight gain (S)"),
                    Line2D([], [], marker="x", markersize=7, markeredgewidth=1.7, color=cfg["recorded_color"],
                           linestyle="none", label="Recorded group gain (R)")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y),
                   ncol=len(handles), frameon=False, fontsize=8)

    def footnote(self, fig, grid, y=.025):
        fig.text(.5, y, "Conditional on %d/%d success-stable plants; other nine gains fixed at successful-flight values."
                 % (grid.baseline_count, int(grid.arrays["sample_count"])), ha="center", fontsize=7.5, color="#4d5359")

    def save(self, fig, relative, caption, metadata):
        base = self.output / relative
        base.parent.mkdir(parents=True, exist_ok=True)
        for extension in self.settings["formats"]:
            dpi = self.settings["dpi"] if extension == "png" else self.settings["surface_raster_dpi"]
            path = base.with_suffix("." + extension)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=.12)
            if path.stat().st_size >= 50_000_000:
                raise ValueError("figure exceeds the 50 MB file cap: %s" % path)
        self.records.append({"figure": str(relative), "caption": caption, "grid": self.size, **metadata})
        plt.close(fig)
        print("figure " + str(relative), flush=True)

    def draw_3d(self, ax, grid, camera):
        cfg = self.settings
        key = (grid.case, grid.group, self.size)
        if key not in self.geometry:
            self.geometry[key] = exposed_faces(grid, cfg["alpha"])
        faces, codes = self.geometry[key]
        rgba = np.empty((len(codes), 4))
        for code, name in ((2, "red"), (3, "green")):
            rgba[codes == code] = colors.to_rgba(cfg[name], cfg[name + "_opacity"])
        collection = Poly3DCollection(faces, facecolors=rgba, edgecolors="none", linewidths=0,
                                      zsort="average", rasterized=True)
        collection.set_zorder(1)
        ax.add_collection3d(collection)
        for k, (axis, setter) in enumerate(zip(grid.axes, (ax.set_xlim, ax.set_ylim, ax.set_zlim))):
            span = axis[-1] - axis[0]
            pad = cfg["frame_padding"] * span
            setter(axis[0] - pad, axis[-1] + pad)
            getattr(ax, ("xaxis", "yaxis", "zaxis")[k]).set_major_locator(MaxNLocator(4))
            getattr(ax, ("xaxis", "yaxis", "zaxis")[k]).set_major_formatter(FuncFormatter(lambda v, p: fmt(v)))
        ax.set_xlabel("P gain", labelpad=5)
        ax.set_ylabel("I gain", labelpad=5)
        ax.set_zlabel("D gain", labelpad=5)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        view = cfg["camera_" + camera]
        ax.view_init(elev=view["elevation"], azim=view["azimuth"])
        if camera == "top":
            # The D axis is viewed end-on: suppress collapsed, overlapping ticks.
            ax.set_zticks([])
            ax.set_zlabel(" ")
            ax.zaxis.label.set_alpha(0)
            ax.zaxis.line.set_alpha(0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((.98, .98, .98, 1))
            axis._axinfo["grid"].update(color=(.55, .55, .55, .3), linewidth=.5)
        ax.tick_params(pad=1, labelsize=8)
        coincident = np.array_equal(grid.gain("success"), grid.gain("recorded"))
        for name, marker, color, order in (("success", "D", cfg["success_color"], 10),
                                           ("recorded", "x", cfg["recorded_color"], 11)):
            gain = grid.gain(name)
            kw = {"facecolor": color, "edgecolor": "white", "linewidth": .8} if name == "success" else {"color": color, "linewidth": 1.7}
            ax.scatter(*gain, marker=marker, s=52, depthshade=False, zorder=order, **kw)
            if not coincident or name == "success":
                shift = np.array([.025 * (axis[-1] - axis[0]) for axis in grid.axes])
                label = "S=R" if coincident else ("S" if name == "success" else "R")
                if camera == "top":
                    label += " (D=%s)" % fmt(gain[2])
                ax.text(*(gain + shift), label, zorder=12, color="#263238", fontsize=9,
                        bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": .5})
        ax.set_title(LABELS[grid.group] + (" — looking down D" if camera == "top" else ""), pad=7, fontsize=11)

    def overview(self, grid, camera):
        fig = plt.figure(figsize=(6.3, 5.8))
        ax = fig.add_axes([.07, .18, .84, .70], projection="3d", computed_zorder=False)
        self.draw_3d(ax, grid, camera)
        fig.suptitle("%s — %s PID volume" % (grid.case.replace("failure", "Failure "), LABELS[grid.group]), y=.99, fontsize=12)
        success, recorded = grid.mark("success"), grid.mark("recorded")
        fig.text(.5, .145, "S = (%s)  |  R = (%s); survival at R: %.2f%%" %
                 (", ".join(map(fmt, success["gain"])), ", ".join(map(fmt, recorded["gain"])), recorded["survival"] * 100),
                 ha="center", fontsize=8)
        self.legend(fig)
        self.footnote(fig, grid)
        caption = ("%s, %s: node-centred transparent PID voxels, %s view. Green: conditional survival >= %.2f%%; red: lower. "
                   "The surface is a discrete cell display without interpolation or a fitted plane. S and R are exact interior grid nodes."
                   % (grid.case, LABELS[grid.group], camera, 100 * self.settings["alpha"]))
        self.save(fig, Path(grid.case) / grid.group / ("overview_3d_" + camera), caption,
                  {"case": grid.case, "group": grid.group, "kind": "overview_3d", "camera": camera})

    def overview_all(self, grids, camera):
        fig = plt.figure(figsize=(11.2, 10))
        for k, grid in enumerate(grids):
            row, col = divmod(k, 2)
            ax = fig.add_axes([.06 + .49 * col, .54 - .405 * row, .40, .36],
                              projection="3d", computed_zorder=False)
            self.draw_3d(ax, grid, camera)
        fig.suptitle("%s — PID volumes by control group (%s)" %
                     (grids[0].case.replace("failure", "Failure "), camera), y=.99, fontsize=14)
        self.legend(fig, y=.047)
        self.footnote(fig, grids[0], y=.024)
        self.save(fig, Path(grids[0].case) / ("all_groups_3d_" + camera),
                  "%s: separate PID volumes for Roll/Pitch, Yaw, XY and Z; same color criterion and camera." % grids[0].case,
                  {"case": grids[0].case, "group": "all", "kind": "overview_all", "camera": camera})

    @staticmethod
    def boundary_segments(classes, x_edges, y_edges):
        # No interpolated contour: threshold transitions follow exact cell edges.
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

    def draw_slice(self, ax, grid, fixed_axis, index, continuous=False, compact=False):
        free, survival, edges = slice_at(grid, fixed_axis, index)
        green = survival >= self.settings["alpha"]
        cmap = plt.get_cmap("RdYlGn") if continuous else colors.ListedColormap([self.settings["red"], self.settings["green"]])
        data = survival if continuous else green.astype(float)
        vertices, values, _ = merge_uniform_cells(data, *edges)
        mesh = PolyCollection(vertices, array=values, cmap=cmap, norm=colors.Normalize(0, 1),
                              edgecolors="none", linewidths=0, antialiased=False, rasterized=False)
        ax.add_collection(mesh)
        segments = self.boundary_segments(green, edges[0], edges[1])
        if segments:
            ax.add_collection(LineCollection(segments, colors="#252525", linewidths=.55 if compact else .8))
        for name in marks_in_slice(grid, fixed_axis, index):
            gain = grid.gain(name)
            if name == "success":
                ax.scatter(gain[free[0]], gain[free[1]], marker="D", s=48 if compact else 68,
                           facecolor=self.settings["success_color"], edgecolor="white", linewidth=.8, zorder=6)
            else:
                ax.scatter(gain[free[0]], gain[free[1]], marker="x", s=60 if compact else 82,
                           color=self.settings["recorded_color"], linewidth=1.8, zorder=7)
        ax.set_xlim(edges[0][0], edges[0][-1])
        ax.set_ylim(edges[1][0], edges[1][-1])
        ax.set_xlabel(GAIN_NAMES[free[0]] + " gain")
        ax.set_ylabel(GAIN_NAMES[free[1]] + " gain")
        ax.xaxis.set_major_locator(MaxNLocator(4 if compact else 5))
        ax.yaxis.set_major_locator(MaxNLocator(4 if compact else 5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt(v)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt(v)))
        ax.tick_params(labelsize=8)
        fixed = "%s = %s" % (GAIN_NAMES[fixed_axis], fmt(grid.axes[fixed_axis][index]))
        cutoff = fmt(self.settings["alpha"] * 100) + "%"
        ax.set_title(fixed if compact else (fixed + " — " + ("survival" if continuous else cutoff + " classification")), fontsize=9)
        return mesh

    def reference_slice(self, grid, fixed_axis, index, tag):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))
        fig.subplots_adjust(left=.085, right=.94, top=.86, bottom=.32, wspace=.38)
        self.draw_slice(axes[0], grid, fixed_axis, index)
        mesh = self.draw_slice(axes[1], grid, fixed_axis, index, continuous=True)
        cbar = fig.colorbar(mesh, ax=axes[1], fraction=.06, pad=.045, ticks=[0, .25, .5, .75, .95, 1])
        cbar.ax.set_yticklabels(["0", "25", "50", "75", "95", "100"])
        cbar.set_label("Conditional survival (%)", fontsize=8)
        free = [GAIN_NAMES[k] for k in range(3) if k != fixed_axis]
        fig.suptitle("%s — %s: %s–%s slice" %
                     (grid.case.replace("failure", "Failure "), LABELS[grid.group], *free), y=.99, fontsize=12)
        visible = marks_in_slice(grid, fixed_axis, index)
        fig.text(.5, .185, "Reference nodes on this plane: " + ", ".join("S" if name == "success" else "R" for name in visible)
                 + ". Black lines follow the " + fmt(self.settings["alpha"] * 100) + "% cell boundary.", ha="center", fontsize=8)
        self.legend(fig, y=.075)
        self.footnote(fig, grid)
        value = fmt(grid.axes[fixed_axis][index])
        name = "slice_%s_%s_%s_%s" % ("".join(free), GAIN_NAMES[fixed_axis], value.replace(".", "p"), tag)
        caption = ("%s %s: exact %s=%s slice, categorical (left) and continuous (right). Only reference nodes lying on this plane are drawn; no off-plane projections."
                   % (grid.case, LABELS[grid.group], GAIN_NAMES[fixed_axis], value))
        self.save(fig, Path(grid.case) / grid.group / name, caption,
                  {"case": grid.case, "group": grid.group, "kind": "reference_slice", "fixed_axis": GAIN_NAMES[fixed_axis],
                   "fixed_value": float(grid.axes[fixed_axis][index]), "references_on_plane": ",".join(visible)})

    def slice_sheet(self, grid, fixed_axis):
        axis = grid.axes[fixed_axis]
        indices = {int(np.argmin(np.abs(axis - (axis[0] + fraction * (axis[-1] - axis[0])))))
                   for fraction in self.settings["slice_fractions"]}
        indices.update(grid.index(grid.gain(name))[fixed_axis] for name in ("success", "recorded"))
        indices = sorted(indices)
        cols = 3
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10.6, 2.7 * rows + 1.35), squeeze=False)
        fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.24, wspace=.28, hspace=.45)
        for ax, index in zip(axes.flat, indices):
            self.draw_slice(ax, grid, fixed_axis, index, continuous=True, compact=True)
        for ax in list(axes.flat)[len(indices):]:
            ax.set_visible(False)
        fig.suptitle("%s — %s: slices across %s" %
                     (grid.case.replace("failure", "Failure "), LABELS[grid.group], GAIN_NAMES[fixed_axis]), y=.985, fontsize=12)
        cax = fig.add_axes([.35, .135, .30, .016])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="RdYlGn"), cax=cax, orientation="horizontal")
        cbar.set_ticks([0, .25, .5, .75, 1])
        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        cbar.ax.tick_params(labelsize=8)
        self.legend(fig, continuous=True, y=.045)
        fractions = "/".join(fmt(v * 100) for v in self.settings["slice_fractions"])
        footer = (f"Nearest nodes to {fractions}% of the axis + reference planes; "
                  f"conditioned on {grid.baseline_count}/{int(grid.arrays['sample_count'])} plants; "
                  f"black: {fmt(self.settings['alpha'] * 100)}% boundary.")
        fig.text(.5, .008, footer,
                 ha="center", fontsize=7.5)
        self.save(fig, Path(grid.case) / grid.group / ("slices_across_" + GAIN_NAMES[fixed_axis]),
                  "%s %s: discrete conditional-survival slices at several exact evaluated %s values, including both reference planes. No interpolation." %
                  (grid.case, LABELS[grid.group], GAIN_NAMES[fixed_axis]),
                  {"case": grid.case, "group": grid.group, "kind": "slice_sheet", "fixed_axis": GAIN_NAMES[fixed_axis],
                   "fixed_values": ",".join(fmt(axis[index]) for index in indices)})

    def slices_all(self, grids):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))
        fig.subplots_adjust(left=.085, right=.94, top=.89, bottom=.15, wspace=.3, hspace=.38)
        for ax, grid in zip(axes.flat, grids):
            index = grid.index(grid.gain("success"))[1]
            self.draw_slice(ax, grid, 1, index)
            ax.set_title(LABELS[grid.group] + " — I = " + fmt(grid.axes[1][index]))
        fig.suptitle("%s — P–D slices at successful-flight I" % grids[0].case.replace("failure", "Failure "), y=.985, fontsize=13)
        self.legend(fig, y=.045)
        self.footnote(fig, grids[0], y=.02)
        self.save(fig, Path(grids[0].case) / "all_groups_PD_at_success_I",
                  "%s: P–D slices through successful-flight integral gains. Recorded group nodes appear only when their I gain equals the plane." % grids[0].case,
                  {"case": grids[0].case, "group": "all", "kind": "slices_all"})

    def finish(self):
        self.output.mkdir(parents=True, exist_ok=True)
        index = self.output / "figure_index.json"
        old = json.loads(index.read_text()) if index.exists() else []
        merged = {row["figure"]: row for row in old}
        merged.update({row["figure"]: row for row in self.records})
        self.records = sorted(merged.values(), key=lambda row: row["figure"])
        (self.output / "figure_index.json").write_text(json.dumps(self.records, indent=2) + "\n")
        keys = ["figure", "case", "group", "kind", "grid", "camera", "fixed_axis", "fixed_value", "fixed_values", "references_on_plane", "caption"]
        with (self.output / "figure_index.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "figure_settings.json")
    parser.add_argument("--grid", type=int, choices=(33, 65), default=65)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--group", choices=GROUPS, action="append")
    parser.add_argument("--only", choices=("all", "overview", "slices"), default="all")
    parser.add_argument("--fixed-axis", choices=GAIN_NAMES)
    parser.add_argument("--fixed-value", type=float)
    args = parser.parse_args()
    if (args.fixed_axis is None) != (args.fixed_value is None):
        parser.error("--fixed-axis and --fixed-value are required together")
    settings = json.loads(args.settings.read_text())
    artist = PaperFigures(args.root, args.output or args.root / "figures", settings, args.grid)
    for case in args.case or CASES:
        grids = []
        for group in args.group or GROUPS:
            grid = load_grid(case, group, args.grid, args.root)
            grids.append(grid)
            if args.fixed_axis is not None:
                axis = GAIN_NAMES.index(args.fixed_axis)
                index = int(np.argmin(np.abs(grid.axes[axis] - args.fixed_value)))
                print("requested %s=%s; evaluated plane %s=%s" %
                      (args.fixed_axis, args.fixed_value, args.fixed_axis, grid.axes[axis][index]), flush=True)
                artist.reference_slice(grid, axis, index, "custom_nearest_node")
                continue
            if args.only in ("all", "overview"):
                for camera in ("oblique", "top"):
                    artist.overview(grid, camera)
            if args.only in ("all", "slices"):
                for fixed_axis in range(3):
                    plane_names = {}
                    for name in ("success", "recorded"):
                        index = grid.index(grid.gain(name))[fixed_axis]
                        plane_names.setdefault(index, []).append(name)
                    for index, names in sorted(plane_names.items()):
                        artist.reference_slice(grid, fixed_axis, index, "_and_".join(names))
                    artist.slice_sheet(grid, fixed_axis)
        if args.fixed_axis is None and args.group is None:
            if args.only in ("all", "overview"):
                for camera in ("oblique", "top"):
                    artist.overview_all(grids, camera)
            if args.only in ("all", "slices"):
                artist.slices_all(grids)
    artist.finish()


if __name__ == "__main__":
    main()
