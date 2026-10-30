#!/usr/bin/env python3
"""Render failure-only PID volumes in the historical eight-panel paper layout.

This renderer does not run an estimator or change any stability statistic. The
historical success-conditioned numerical code and its figures are never used
as input. Historical visual settings and layout are reused. An optional,
separate reference provides successful gain coordinates without stability scores.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pid-failure-publication-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

HERE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = HERE / "outputs/failure_only_33_65_20261028"
DEFAULT_SETTINGS = HERE / "publication/pid_gain_voxels/figure_settings.json"
CASES = ("failure1", "failure2")
GROUPS = ("roll_pitch", "yaw", "xy", "z")
SOURCE_GROUPS = ("xy", "z", "roll_pitch", "yaw")
LABELS = {"roll_pitch": "Roll / Pitch", "yaw": "Yaw", "xy": "XY", "z": "Z"}
SCHEMA = "grape-param-estim/failure-only-paper-figures/v1"
SUCCESS_REFERENCE_SCHEMA = "grape-param-estim/successful-controller-position-reference/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midpoint_edges(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError("expected a finite, strictly increasing gain axis")
    return np.r_[values[0], (values[:-1] + values[1:]) / 2, values[-1]]


def rate_colormap(settings: Mapping[str, Any]):
    positions = np.linspace(0, 1, 256)
    rgb = matplotlib.colormaps[settings["rate_colormap"]](positions)[:, :3]
    white = float(settings["white_mix"])
    if not 0 <= white < 1:
        raise ValueError("white_mix must lie in [0, 1)")
    rgb = white + (1 - white) * rgb
    # Preserve the historical palette's 8-bit hexadecimal anchors before
    # interpolation, including its rounding. This changes no stability values.
    anchors = [(float(q), colors.to_hex(color)) for q, color in zip(positions, rgb)]
    return colors.LinearSegmentedColormap.from_list("continuous_rate", anchors, N=4096)


def load_success_reference(path: Path) -> dict[str, Any]:
    """Read post-calculation coordinates only; never consult success scores."""
    path = Path(path)
    payload = json.loads(path.read_text())
    gains = np.asarray(payload.get("gains"), dtype=float)
    if (payload.get("schema") != SUCCESS_REFERENCE_SCHEMA
            or payload.get("gain_order") != list(SOURCE_GROUPS)
            or gains.shape != (4, 3) or not np.all(np.isfinite(gains))):
        raise ValueError("expected a finite 4x3 successful gain position reference in SOURCE_GROUPS order")
    return {"schema": SUCCESS_REFERENCE_SCHEMA, "role": "coordinates_only",
            "gain_order": list(SOURCE_GROUPS), "gains": gains.tolist(),
            "source": payload.get("source"), "path": str(path), "sha256": sha256(path)}


def reference_gain(reference, group):
    return np.asarray(reference["gains"][SOURCE_GROUPS.index(group)], dtype=float)


def padded_limits(axis, padding, reference=None):
    """Expand the display frame for a marker, never the evaluated data domain."""
    lower, upper = float(axis[0]), float(axis[-1])
    if reference is not None:
        lower, upper = min(lower, float(reference)), max(upper, float(reference))
    span = upper - lower
    return lower - padding * span, upper + padding * span


def success_panel_metadata(grid, reference, settings, view):
    gain = reference_gain(reference, grid.group)
    if view == "PD" and gain[1] != grid.gain[1]:
        raise ValueError("successful reference I does not lie on this exact P-D slice")
    fixed = grid.index()[1]
    drawn = [grid.axes[0], grid.axes[1][fixed:], grid.axes[2]]
    display = [list(padded_limits(axis, float(settings["frame_padding"]), value))
               for axis, value in zip(drawn, gain)]
    if view == "PD":
        display[1] = [float(grid.gain[1]), float(grid.gain[1])]
    return {"role": "coordinates_only", "gain": gain.tolist(),
            "full_successful_controller_gains": reference["gains"],
            "other_nine_gains_context": "this failure record, unchanged",
            "stability_value_assigned": False,
            "computed_axis_limits": [[float(axis[0]), float(axis[-1])] for axis in grid.axes],
            "display_axis_limits": display,
            "inside_computed_box": bool(all(axis[0] <= value <= axis[-1]
                                             for axis, value in zip(grid.axes, gain))),
            "coincides_with_failure": bool(np.array_equal(gain, grid.gain)),
            "on_recorded_I_plane": bool(gain[1] == grid.gain[1])}


@dataclass
class Grid:
    case: str
    group: str
    size: int
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    stable_fraction: np.ndarray
    stable_count: np.ndarray
    valid_count: np.ndarray
    unresolved_count: np.ndarray
    sample_count: int
    recorded_gains: np.ndarray
    provenance: dict[str, Any]

    @property
    def edges(self):
        return tuple(midpoint_edges(axis) for axis in self.axes)

    @property
    def gain(self):
        return self.recorded_gains[SOURCE_GROUPS.index(self.group)]

    def index(self, gain=None):
        gain = self.gain if gain is None else np.asarray(gain, dtype=float)
        indices = []
        for value, axis in zip(gain, self.axes):
            matches = np.flatnonzero(axis == value)
            if len(matches) != 1:
                raise ValueError("failure reference must be an exact computed node")
            indices.append(int(matches[0]))
        return tuple(indices)

    def mark(self):
        index = self.index()
        return {"gain": self.gain.tolist(), "stable_fraction": float(self.stable_fraction[index]),
                "stable_count": int(self.stable_count[index]), "sample_count": self.sample_count,
                "valid_count": int(self.valid_count[index]), "unresolved_count": int(self.unresolved_count[index]),
                "full_fixed_controller_gains": self.recorded_gains.tolist()}


def _recount_poles(directory: Path, shape: tuple[int, ...], sample_count: int, axes=None):
    """Prefer per-sample radii, then exact stable/valid bits, then saved counts.

    The latter are explicitly identified in the figure provenance, never
    described as an independent per-sample recount by this renderer.
    """
    if (directory / "spectral_radius.npy").exists():
        radii = np.load(directory / "spectral_radius.npy", mmap_mode="r", allow_pickle=False)
        done = np.load(directory / "evaluated.npy", mmap_mode="r", allow_pickle=False)
        kind = "per-sample memmaps"
    elif (directory / "volume.npz").exists():
        with np.load(directory / "volume.npz", allow_pickle=False) as saved:
            radii, done = saved["spectral_radius"], saved["evaluated"]
        kind = "complete per-sample volume archive"
    elif (directory / "sample_bits.npz").exists():
        with np.load(directory / "sample_bits.npz", allow_pickle=False) as saved:
            packed_stable = saved["stable_bits"]
            packed_valid = saved["valid_bits"]
            done = saved["evaluated"]
            expected_shape = shape + ((sample_count + 7) // 8,)
            if (int(saved["sample_count"]) != sample_count
                    or str(saved["bitorder"]) != "little"
                    or packed_stable.dtype != np.uint8 or packed_valid.dtype != np.uint8
                    or packed_stable.shape != expected_shape or packed_valid.shape != expected_shape
                    or done.shape != shape or not np.all(done)):
                raise ValueError("per-sample bit archive is incomplete or has a different population")
            if axes is not None and any(not np.array_equal(saved[key], axis)
                                        for key, axis in zip(("p", "i", "d"), axes)):
                raise ValueError("per-sample bit archive axes differ from the evaluated axes")
            stable = np.unpackbits(packed_stable, axis=-1, count=sample_count,
                                   bitorder="little").astype(bool)
            valid = np.unpackbits(packed_valid, axis=-1, count=sample_count,
                                  bitorder="little").astype(bool)
            if np.any(stable & ~valid):
                raise ValueError("a stable sample bit must also have a valid sample bit")
            counts = (np.count_nonzero(stable, axis=-1).astype(np.int32),
                      np.count_nonzero(valid, axis=-1).astype(np.int32))
            for name, values in zip(("stable_count", "valid_count"), counts):
                if name in saved.files and not np.array_equal(saved[name], values):
                    raise ValueError("saved bit-archive counts differ from the original sample bits")
        return counts, "complete per-sample stable/valid bit archive"
    else:
        return None, "saved all-sample counts only"
    if radii.shape != shape + (sample_count,) or done.shape != shape or not np.all(done):
        raise ValueError("per-sample source is incomplete or has a different population")
    stable = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=np.int32)
    for p in range(shape[0]):
        finite = np.isfinite(radii[p])
        valid[p] = np.count_nonzero(finite, axis=-1)
        stable[p] = np.count_nonzero(finite & (radii[p] < 1.0), axis=-1)
    return (stable, valid), kind


def load_grid(root: Path, case: str, group: str, size: int) -> Grid:
    folder = Path(root) / case
    directory = folder / group / f"grid_{size}"
    status = json.loads((directory / "status.json").read_text())
    if not status.get("complete") or status.get("pending", 0):
        raise ValueError(f"grid is not complete: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text())
    if "unconditional-failure-context" not in metadata.get("schema", ""):
        raise ValueError("success-conditioned or unspecified grid schema is not a valid input")
    references = json.loads((folder / "references.json").read_text())
    recorded = np.asarray(references["recorded_case"]["gains"], dtype=float)
    if recorded.shape != (4, 3) or not np.array_equal(recorded, metadata["full_fixed_controller_gains"]):
        raise ValueError("the other nine gains are not this failure recording's gains")
    source = directory / "stability_fraction.npz"
    with np.load(source, allow_pickle=False) as saved:
        axes = tuple(saved[key].copy() for key in ("p", "i", "d"))
        fraction, count, valid, unresolved = (saved[key].copy() for key in (
            "stable_fraction", "stable_count", "valid_count", "unresolved_count"))
        n = int(saved["sample_count"])
    shape = tuple(len(axis) for axis in axes)
    if n <= 0 or n != int(metadata["plant_samples"]) or list(shape) != metadata["shape"]:
        raise ValueError("sample population or actual grid shape differs from metadata")
    if any(not np.array_equal(axis, declared) for axis, declared in zip(axes, metadata["axes"])):
        raise ValueError("saved plotting axes differ from the evaluated axes")
    for axis in axes:
        midpoint_edges(axis)
    if any(values.shape != shape for values in (fraction, count, valid, unresolved)):
        raise ValueError("stability arrays do not match the complete grid")
    if np.any(~np.isfinite(fraction)) or not np.array_equal(fraction, count / n):
        raise ValueError("rate must be stable count divided by ALL original samples")
    if np.any(count < 0) or np.any(count > valid) or np.any(valid > n) or not np.array_equal(unresolved + valid, np.full(shape, n)):
        raise ValueError("validity and all-sample counts are inconsistent")
    recount, source_kind = _recount_poles(directory, shape, n, axes)
    if recount is not None and (not np.array_equal(recount[0], count) or not np.array_equal(recount[1], valid)):
        raise ValueError("saved rate/counts differ from the per-sample poles/stability bits")
    provenance = {"source": str(source), "source_sha256": sha256(source),
                  "metadata_sha256": sha256(directory / "metadata.json"),
                  "sample_count": n, "actual_shape": list(shape), "per_sample_check": source_kind,
                  "conditioned_on_success": False, "fixed_context": "this failure record",
                  "full_fixed_controller_gains": recorded.tolist()}
    grid = Grid(case, group, size, axes, fraction, count, valid, unresolved, n, recorded, provenance)
    grid.index()
    return grid


def merge_uniform_cells(data, x_edges, y_edges):
    """Merge only exactly equal neighbors, preserving nonuniform cell areas."""
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


def exposed_faces(values, axes):
    """Exact visualization quads; equal-rate internal faces alone are omitted."""
    edges = [midpoint_edges(axis) for axis in axes]
    faces_all, rates_all = [], []
    for axis in range(3):
        free = [k for k in range(3) if k != axis]
        for side in (0, 1):
            visible = np.ones(values.shape, dtype=bool)
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[axis] = slice(1, None) if side == 0 else slice(None, -1)
            src[axis] = slice(None, -1) if side == 0 else slice(1, None)
            visible[tuple(dst)] = values[tuple(dst)] != values[tuple(src)]
            locations = np.argwhere(visible)
            faces = np.empty((len(locations), 4, 3), dtype=float)
            faces[:, :, axis] = edges[axis][locations[:, axis] + side, None]
            for corner, (a, b) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                faces[:, corner, free[0]] = edges[free[0]][locations[:, free[0]] + a]
                faces[:, corner, free[1]] = edges[free[1]][locations[:, free[1]] + b]
            faces_all.append(faces)
            rates_all.append(values[tuple(locations.T)])
    return np.concatenate(faces_all), np.concatenate(rates_all)


def point_text(grid):
    mark = grid.mark()
    gains = "(" + ", ".join(format(float(v), ".5g") for v in mark["gain"]) + ")"
    return "Failure\n" + gains + "\nrate : " + format(100 * mark["stable_fraction"], ".1f") + "%"


def point_style(settings, coincident=False):
    if coincident:
        return dict(marker="o", s=190, facecolors="none", edgecolors=settings["recorded_color"],
                    linewidth=2.2, zorder=7)
    return dict(marker="X", s=95, color=settings["recorded_color"], edgecolor="white", linewidth=1.15, zorder=7)


def success_point_style(settings):
    return dict(marker="D", s=68, color=settings["success_color"], edgecolor="white", linewidth=1.1, zorder=8)


def success_point_text(gain):
    gains = "(" + ", ".join(format(float(v), ".5g") for v in gain) + ")"
    return "Success ref.\n" + gains + "\nposition only"


def annotation_style(settings, success=False):
    color = settings["success_color" if success else "recorded_color"]
    return dict(ha="center", va="center", fontsize=9.5, color=color, zorder=10,
                annotation_clip=False,
                arrowprops={"arrowstyle": "->", "lw": 1.55, "color": color, "shrinkA": 4, "shrinkB": 7},
                bbox={"facecolor": "white", "edgecolor": color, "linewidth": .6, "alpha": .97, "pad": 3})


def comparison_slice(grids, settings, success_reference=None):
    cmap = rate_colormap(settings)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    fig.subplots_adjust(left=.055, right=.982, top=.87, bottom=.15, wspace=.34, hspace=.48)
    for row, case in enumerate(CASES):
        fig.text(.012, .695 - row * .423, case.replace("failure", "Failure "), rotation=90, ha="center", va="center", fontsize=14)
        for column, group in enumerate(GROUPS):
            grid, ax = grids[case, group], axes[row, column]
            success = reference_gain(success_reference, group) if success_reference is not None else None
            if success is not None:
                success_panel_metadata(grid, success_reference, settings, "PD")
            index = grid.index()[1]
            values = grid.stable_fraction[:, index, :].T
            edges = (grid.edges[0], grid.edges[2])
            vertices, rates, _ = merge_uniform_cells(values, *edges)
            ax.add_collection(PolyCollection(vertices, array=rates, cmap=cmap, norm=colors.Normalize(0, 1),
                                             edgecolors="none", linewidth=0, antialiased=False))
            for dimension, edge, setter in zip((0, 2), edges, (ax.set_xlim, ax.set_ylim)):
                setter(*padded_limits(edge, float(settings["frame_padding"]),
                                      None if success is None else success[dimension]))
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("P gain", fontsize=10); ax.set_ylabel("D gain", fontsize=10)
            ax.tick_params(labelsize=9); ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(LABELS[group], fontsize=11, color="#263238", pad=3)
            ax.text(.98, .98, "I = " + format(grid.gain[1], ".5g"), transform=ax.transAxes, ha="right", va="top", fontsize=9)
            pspan, dspan = [edge[-1] - edge[0] for edge in edges]
            gain = grid.gain
            coincident = success is not None and np.array_equal(gain, success)
            failure_point = ax.scatter(gain[0], gain[2], **point_style(settings, coincident))
            failure_point.set_gid("failure_reference")
            if group == "roll_pitch":
                offset = ((-.35 if case == "failure1" else .40) * pspan, -.09 * dspan)
            else:
                offset = ((.10 if group == "z" else -.12) * pspan, -.18 * dspan)
            ax.annotate(point_text(grid), xy=(gain[0], gain[2]),
                        xytext=(gain[0] + offset[0], gain[2] + offset[1]), **annotation_style(settings))
            if success is not None:
                point = ax.scatter(success[0], success[2], **success_point_style(settings))
                point.set_gid("success_reference")
                offset = ((-.12 * pspan, -.28 * dspan) if group == "roll_pitch" else
                          (.06 * pspan, -.18 * dspan) if group == "xy" else
                          (-.06 * pspan, .18 * dspan) if group == "z" else (.06 * pspan, .18 * dspan))
                text = ax.annotate(success_point_text(success), xy=(success[0], success[2]),
                                   xytext=(success[0] + offset[0], success[2] + offset[1]),
                                   **annotation_style(settings, success=True))
                text.set_gid("success_reference_label")
    return fig


def comparison_volume(grids, settings, success_reference=None):
    cmap = rate_colormap(settings)
    fig = plt.figure(figsize=(16, 8.6))
    for row, case in enumerate(CASES):
        fig.text(.013, .69 - row * .425, case.replace("failure", "Failure "), rotation=90, ha="center", va="center", fontsize=14)
        for column, group in enumerate(GROUPS):
            grid = grids[case, group]
            success = reference_gain(success_reference, group) if success_reference is not None else None
            ax = fig.add_axes([.043 + .238 * column, .545 - .425 * row, .224, .35], projection="3d", computed_zorder=False)
            fixed = grid.index()[1]
            axes = (grid.axes[0], grid.axes[1][fixed:], grid.axes[2])
            values = grid.stable_fraction[:, fixed:, :]
            faces, rates = exposed_faces(values, axes)
            rgba = cmap(rates); rgba[:, 3] = float(settings["voxel_opacity"])
            ax.add_collection3d(Poly3DCollection(faces, facecolors=rgba, edgecolors="none", linewidth=0,
                                                zsort="average", rasterized=True, zorder=1))
            for dimension, axis, setter in zip(range(3), axes, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(*padded_limits(axis, float(settings["frame_padding"]),
                                      None if success is None else success[dimension]))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3)); axis.set_pane_color((1, 1, 1, 1))
                axis._axinfo["grid"].update(color=(.6, .6, .6, .18), linewidth=.45)
            ax.tick_params(labelsize=8, pad=0)
            ax.set_xlabel("P gain", labelpad=0, fontsize=10)
            ax.set_ylabel("I gain", labelpad=0, fontsize=10)
            ax.set_zlabel("D gain", labelpad=0, fontsize=10)
            ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho")
            camera = settings["camera_oblique"]
            ax.view_init(elev=camera["elevation"], azim=camera["azimuth"])
            ax.set_title(LABELS[group], fontsize=11, color="#263238", pad=3)
            gain = grid.gain
            coincident = success is not None and np.array_equal(gain, success)
            failure_point = ax.scatter(*gain, depthshade=False, **point_style(settings, coincident))
            failure_point.set_gid("failure_reference")
            x, y, _ = proj3d.proj_transform(*gain, ax.get_proj())
            offset = ((22, -28) if case == "failure1" else (-12, -28)) if group == "roll_pitch" else {
                "yaw": (-28, -30), "xy": (-8, -30), "z": (28, -32)}[group]
            ax.annotate(point_text(grid), xy=(x, y), xytext=offset, textcoords="offset points", **annotation_style(settings))
            if success is not None:
                point = ax.scatter(*success, depthshade=False, **success_point_style(settings))
                point.set_gid("success_reference")
                x, y, _ = proj3d.proj_transform(*success, ax.get_proj())
                offset = (17, 32) if group in ("yaw", "xy") else (-20, 34)
                text = ax.annotate(success_point_text(success), xy=(x, y), xytext=offset,
                                   textcoords="offset points", **annotation_style(settings, success=True))
                text.set_gid("success_reference_label")
    return fig


def save_comparison(fig, grids, settings, output: Path, view: str, grid_size: int, success_reference=None):
    bar_axes = fig.add_axes([.375, .071, .25, .012])
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap=rate_colormap(settings)),
                      cax=bar_axes, orientation="horizontal", ticks=[0, .25, .5, .75, 1])
    bar.ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    bar.ax.tick_params(labelsize=10, length=2, pad=2); bar.outline.set_linewidth(.5)
    fig.text(.36, .077, "rate (%)", ha="right", va="center", fontsize=11)
    note = "3D volumes are cut at each recorded I." if view == "3D" else "Each exact P-D slice passes through the recorded failure point."
    markers = (" Purple: failure gains; blue: successful gain coordinates only."
               if success_reference is not None else " Purple marks identify the failure-flight gains.")
    fig.text(.5, .03, note + markers, ha="center", fontsize=10, color="#52606B")
    fig.text(.5, .007, "Each case uses all original plant samples; other nine gains are fixed to that failure record.", ha="center", fontsize=9, color="#52606B")
    name = f"all_pid_failure_{view}_grid_{grid_size}"
    files = {}
    try:
        for ext in settings["formats"]:
            path = output / (name + "." + ext)
            fig.savefig(path, dpi=settings["dpi"] if ext == "png" else settings["surface_raster_dpi"], bbox_inches="tight", pad_inches=.12)
            files[ext] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    finally:
        plt.close(fig)
    record = {"schema": SCHEMA, "figure": name, "groups": list(GROUPS), "cases": list(CASES),
            "view": view, "panel_count": 8, "grid_size": grid_size,
            "caption": "Unconditional local stability fraction on all original plant samples. Other nine gains are fixed independently to the corresponding failure record. "
                       + ("Blue diamonds locate successful gain coordinates only, with no assigned stability value. " if success_reference is not None else "")
                       + "No success-100% condition is used.",
            "color_encoding": {"quantity": "stable_fraction", "range": [0, 1], "colormap": settings["rate_colormap"], "white_mix": settings["white_mix"], "continuous": True, "surface_opacity": settings["voxel_opacity"]},
            "references": {case: {group: grids[case, group].mark() for group in GROUPS} for case in CASES},
            "source_grids": {case: {group: grids[case, group].provenance for group in GROUPS} for case in CASES},
            "files": files}
    if success_reference is not None:
        record["success_reference"] = {**success_reference,
            "panels": {case: {group: success_panel_metadata(grids[case, group], success_reference, settings, view)
                               for group in GROUPS} for case in CASES}}
    return record


def render(input_root: Path, output: Path, grid_size: int, settings_path: Path = DEFAULT_SETTINGS,
           view="both", success_reference_path=None):
    input_root, output, settings_path = Path(input_root), Path(output), Path(settings_path)
    settings = json.loads(settings_path.read_text())
    plt.rcParams.update({"font.family": settings["font_family"], "font.size": settings["font_size"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    grids = {(case, group): load_grid(input_root, case, group, grid_size) for case in CASES for group in GROUPS}
    # This input is read only after every completed map has been loaded and
    # checked. It cannot affect the map population, axes, counts or colors.
    success_reference = load_success_reference(success_reference_path) if success_reference_path is not None else None
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, draw in (("3D", comparison_volume), ("PD", comparison_slice)):
        if view in (kind, "both"):
            record = save_comparison(draw(grids, settings, success_reference), grids, settings,
                                     output, kind, grid_size, success_reference)
            record["style"] = {"source": str(settings_path), "source_sha256": sha256(settings_path), "settings": settings}
            records.append(record)
    (output / "figure_index.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "figure_index.csv").open("w", newline="") as stream:
        fields = ("figure", "groups", "view", "cases", "panel_count", "caption")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in records)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--grid", type=int, choices=(33, 65), default=65)
    parser.add_argument("--view", choices=("PD", "3D", "both"), default="both")
    parser.add_argument("--success-gains-json", type=Path,
                        help="Optional post-map successful gain coordinates only; does not assign a stability rate")
    args = parser.parse_args(argv)
    records = render(args.input_root, args.output, args.grid, args.settings, args.view, args.success_gains_json)
    print(json.dumps({"grid": args.grid, "figures": [row["figure"] for row in records], "panel_count": 8}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
