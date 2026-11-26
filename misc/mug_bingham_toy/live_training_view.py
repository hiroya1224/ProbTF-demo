#!/usr/bin/env python3
"""Read-only desktop view of the no-blur run's saved epoch measurements.

The training process evaluates the five frozen preview images once per epoch.
This viewer reads those measurements; it never imports Torch, loads weights,
runs inference, or changes a distribution. The filled spheres use the saved
camera-frame A. Their RGB channels are relative axis densities for display.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/vgg16_10000_cylinder_center_no_blur_20261127"
DEFAULT_INPUTS = HERE / "examples/cylinder_pointlight_preview/inputs.npz"
ANGLES = (0, 60, 90, 120, 180)
CROP = (157, 74, 483, 330)
COLORS = ("#c33130", "#247b44", "#2b64b2")


@dataclass(frozen=True)
class Preview:
    images: np.ndarray
    angles: np.ndarray
    sha256: str


def load_preview(path: Path) -> Preview:
    """Freeze the exact RGB packet in memory, hashing the same bytes read."""
    raw = Path(path).read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as packet:
        images = packet["rgb"].copy()
        angles = packet["angles_deg"].copy()
    if images.ndim != 4 or images.shape[0] != 5 or images.shape[-1] != 3:
        raise ValueError("Preview packet must contain five RGB images")
    if images.dtype != np.uint8 or tuple(angles.tolist()) != ANGLES:
        raise ValueError("Preview RGB type or angle order differs from the five training inputs")
    images.setflags(write=False)
    angles.setflags(write=False)
    return Preview(images, angles, hashlib.sha256(raw).hexdigest())


def read_metric_records(path: Path) -> tuple[list[dict], list[str]]:
    """Read complete JSONL records; an in-progress final write stays pending."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return [], []
    rows, errors = [], []
    lines = raw.splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if not line.endswith(b"\n"):
            break
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            rows.append(value)
        except (ValueError, UnicodeDecodeError) as error:
            errors.append(f"metrics line {number}: {error}")
    return rows, errors


def measurement(row: dict) -> dict | None:
    """Validate the shared training/viewer schema without modifying its values."""
    value = row.get("tangent_xz")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("epoch") != row.get("epoch"):
        raise ValueError("Tangent measurement and loss record must have the same epoch")
    if tuple(value.get("angles_deg", ())) != ANGLES:
        raise ValueError("Measured angle order differs from the frozen preview")
    matrix = np.asarray(value.get("A_camera"), dtype=float)
    if matrix.shape != (5, 4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected five finite camera-frame Bingham matrices")
    if not np.allclose(matrix, matrix.transpose(0, 2, 1), rtol=0, atol=1e-10):
        raise ValueError("Saved Bingham matrices must be symmetric")
    if not isinstance(value.get("inputs_sha256"), str):
        raise ValueError("The measured preview hash is missing")
    for axis in ("X", "Z"):
        if not isinstance(value.get(axis), list) or len(value[axis]) != 5:
            raise ValueError(f"Expected five {axis} tangent measurements")
    return value


def principal_values(component: dict | None, key: str) -> np.ndarray:
    """Return measured principal values; undefined values remain NaN gaps."""
    if (component is None or component.get("covariance_defined") is False
            or component.get(key) is None):
        return np.full(2, np.nan)
    values = np.asarray(component[key], dtype=float)
    if values.shape != (2,):
        raise ValueError(f"{key} must contain two principal values")
    if np.any(np.isfinite(values) & (values < 0)) or np.isinf(values).any():
        raise ValueError(f"{key} contains an invalid principal value")
    return values.copy()


def variance_history(records: list[dict], axis: str, angle: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the actual variances, including arbitrarily small positive values."""
    epochs, values = [], []
    for row in records:
        current = measurement(row)
        if current is None:
            continue
        index = current["angles_deg"].index(angle)
        epochs.append(float(row["epoch"]))
        values.append(principal_values(current[axis][index], "principal_variance_rad2"))
    return np.asarray(epochs), np.asarray(values).reshape((-1, 2))


def sigma_label(component: dict | None) -> str:
    values = principal_values(component, "principal_sigma_deg")
    return " / ".join("—" if np.isnan(value) else f"{value:.2f}" for value in values)


class TrainingViewer:
    def __init__(self, run: Path, inputs: Path, *, subdivisions: int = 4):
        import matplotlib.pyplot as plt

        self.run, self.inputs = Path(run), Path(inputs)
        self.subdivisions = subdivisions
        self.preview = None
        self._signature = None
        self._last_plot_key = None
        self._timer = None
        plt.rcParams.update({"font.family": ["DejaVu Sans", "IPAexGothic"],
                             "font.size": 10, "axes.linewidth": .7})
        self.figure = plt.figure(figsize=(14.4, 9.0), dpi=100, facecolor="white")
        manager = self.figure.canvas.manager
        if manager is not None:
            manager.set_window_title("Mug Bingham training | no blur | X/Z tangent variances")
        self.figure.text(.5, .975, "ぼかしなし · 全 VGG16 の学習と X/Z 軸の接分散",
                         ha="center", va="top", fontsize=16)
        self.status = self.figure.text(.5, .938, "測定待ち", ha="center", va="top", fontsize=10)
        outer = self.figure.add_gridspec(3, 1, left=.062, right=.985, bottom=.075,
                                        top=.89, height_ratios=(1., 1.15, 1.), hspace=.38)
        photo_grid = outer[0].subgridspec(1, 5, wspace=.10)
        sphere_grid = outer[1].subgridspec(1, 5, wspace=.10)
        history_grid = outer[2].subgridspec(1, 2, wspace=.24)
        self.photo_axes = [self.figure.add_subplot(photo_grid[0, j]) for j in range(5)]
        self.sphere_axes = [self.figure.add_subplot(sphere_grid[0, j]) for j in range(5)]
        self.sigma_text = []
        for angle, photo, sphere in zip(ANGLES, self.photo_axes, self.sphere_axes):
            photo.set_title(f"{angle} [deg]", fontsize=12)
            photo.axis("off")
            sphere.set(aspect="equal", xlim=(-1.08, 1.08), ylim=(1.08, -1.08))
            sphere.axis("off")
            sphere.text(.5, .5, "初回の実測値を待っています", transform=sphere.transAxes,
                        ha="center", va="center", fontsize=8, color=".4")
            self.sigma_text.append(sphere.text(.5, -.08, "", transform=sphere.transAxes,
                                               ha="center", va="top", fontsize=8.5))
        top = self.sphere_axes[0].get_position().y1
        self.figure.text(.5, top+.018,
                         "同じカメラから見た球面密度 · 赤 X / 緑 Y / 青 Z · 各軸のメッシュ最大値に対する相対色 · 両半球を表示",
                         ha="center", va="bottom", fontsize=9)
        self.history_axes, self.history_lines = {}, {}
        for column, axis in enumerate(("X", "Z")):
            ax = self.figure.add_subplot(history_grid[0, column])
            color = COLORS[0 if axis == "X" else 2]
            ax.set(title=f"取っ手が後ろ (90°): 物体 {axis} 軸", xlabel="epoch",
                   ylabel="接共分散の主値 [rad²]", xlim=(0, 1))
            ax.grid(True, alpha=.2, which="both")
            lines = []
            for angle, linestyle, alpha in ((90, "-", 1.), (0, "--", .5)):
                for eigen_index, width in ((0, 1.1), (1, 2.0)):
                    line, = ax.plot([], [], color=color, ls=linestyle, alpha=alpha, lw=width,
                                    marker="o" if angle == 90 else ".", markersize=3,
                                    label=f"{angle}° / 主値 {eigen_index+1}")
                    lines.append(line)
            ax.legend(loc="best", fontsize=7.5, ncol=2)
            self.history_axes[axis], self.history_lines[axis] = ax, lines
        self.figure.text(.5, .018,
                         "第4章の有限主要指数代理による接共分散 (+I/3) · 定義できない値は欠測として表示 · 分散の倍率変更なし",
                         ha="center", va="bottom", fontsize=9)
        self.poll(force=True)

    def _freeze_preview(self):
        if self.preview is not None:
            return
        self.preview = load_preview(self.inputs)
        x0, y0, x1, y1 = CROP
        for ax, image in zip(self.photo_axes, self.preview.images):
            ax.imshow(image[y0:y1, x0:x1], interpolation="none")

    def _draw_spheres(self, current):
        from matplotlib.collections import PolyCollection
        from figure_geometry import (camera_sphere_density_mesh, mode_quaternion,
                                     quaternion_matrix, triangulated_unit_sphere)

        graticule = triangulated_unit_sphere(subdivisions=2)
        for index, (ax, A) in enumerate(zip(self.sphere_axes, current["A_camera"])):
            ax.clear()
            ax.set(aspect="equal", xlim=(-1.08, 1.08), ylim=(1.08, -1.08))
            ax.axis("off")
            mesh = camera_sphere_density_mesh(np.asarray(A), subdivisions=self.subdivisions,
                                               log_normalizer=None)
            order = np.argsort(mesh["face_depth"])[::-1]
            polygons = mesh["camera_xy"][mesh["faces"]][order]
            ax.add_collection(PolyCollection(polygons, facecolors=(.89, .89, .89, .14),
                                             edgecolors="none", antialiaseds=False, zorder=1))
            rgb = mesh["face_rgb"][order]
            rgba = np.column_stack((rgb, .9*rgb.max(axis=1)))
            ax.add_collection(PolyCollection(polygons, facecolors=rgba, edgecolors="none",
                                             antialiaseds=False, zorder=2))
            ax.add_collection(PolyCollection(graticule["camera_xy"][graticule["faces"]],
                                             facecolors="none", edgecolors=(.58, .58, .58, .25),
                                             linewidths=.22, antialiaseds=False, zorder=3))
            eigenvalues = np.linalg.eigvalsh(A)
            if eigenvalues[-1] > eigenvalues[-2]:
                peak = quaternion_matrix(mode_quaternion(np.asarray(A)))[:2]
                for k, color in enumerate(COLORS):
                    ax.plot([0, peak[0, k]], [0, peak[1, k]], lw=1.1, color=color, zorder=4)
            label = f"X σ [deg] {sigma_label(current['X'][index])}\nZ σ [deg] {sigma_label(current['Z'][index])}"
            self.sigma_text[index] = ax.text(.5, -.08, label, transform=ax.transAxes,
                                              ha="center", va="top", fontsize=8.5)

    def _draw_history(self, records):
        for axis, ax in self.history_axes.items():
            all_finite = []
            for start, angle in ((0, 90), (2, 0)):
                epochs, values = variance_history(records, axis, angle)
                for eigen_index in range(2):
                    self.history_lines[axis][start+eigen_index].set_data(epochs, values[:, eigen_index])
                all_finite.extend(values[np.isfinite(values)].tolist())
            # A zero value is left unchanged and uses a linear axis. Positive
            # variances use a log axis without an epsilon, lower bound or floor.
            ax.set_yscale("log" if all_finite and min(all_finite) > 0 else "linear")
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            maximum = max((float(row["epoch"]) for row in records), default=1)
            ax.set_xlim(0, max(1, maximum)*1.04)

    def poll(self, *, force=False):
        try:
            self._freeze_preview()
            metrics_path = self.run / "metrics.jsonl"
            try:
                stat = metrics_path.stat()
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                signature = None
            complete = (self.run / "test_metrics.json").exists()
            signature = (signature, complete)
            if not force and signature == self._signature:
                return True
            self._signature = signature
            rows, errors = read_metric_records(metrics_path)
            compatible = []
            for row in rows:
                try:
                    current = measurement(row)
                    if current is None:
                        continue
                    if current["inputs_sha256"] != self.preview.sha256:
                        errors.append(f"epoch {row['epoch']}: 入力画像の hash が表示中の画像と一致しません")
                        continue
                    # Validate the plotted values before drawing any of this epoch.
                    for axis in ("X", "Z"):
                        for component in current[axis]:
                            principal_values(component, "principal_variance_rad2")
                            principal_values(component, "principal_sigma_deg")
                    compatible.append(row)
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"epoch {row.get('epoch', '?')}: {error}")
            if not compatible:
                self.status.set_text("初回 epoch の測定待ち · 5画像の初回評価を待っています" if not errors
                                     else "表示保留: " + errors[-1])
            else:
                row = compatible[-1]
                current = measurement(row)
                key = (row["epoch"], hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest())
                if key != self._last_plot_key:
                    self._draw_spheres(current)
                    self._draw_history(compatible)
                    self._last_plot_key = key
                train_nll = row.get("train_nll_mean")
                val_nll = row.get("validation", {}).get("nll_mean")
                train_text = "—" if train_nll is None else f"{train_nll:.5f}"
                val_text = "—" if val_nll is None else f"{val_nll:.5f}"
                state = "完了" if complete else "学習中"
                status = (f"{state} · epoch {row['epoch']} · train NLL {train_text} · validation NLL {val_text}"
                          " · 同じ epoch の重みと入力画像による測定")
                if errors:
                    status += "\n表示保留: " + errors[-1]
                self.status.set_text(status)
            self.figure.canvas.draw_idle()
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.status.set_text("表示待ち: " + str(error))
            self.figure.canvas.draw_idle()
        return True

    def start(self, interval_seconds: float):
        import matplotlib.pyplot as plt

        self._timer = self.figure.canvas.new_timer(interval=max(100, round(interval_seconds*1000)))
        self._timer.add_callback(self.poll)
        self._timer.start()
        self.figure.canvas.mpl_connect("close_event", lambda event: self._timer.stop())
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("--subdivisions", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--snapshot", type=Path, help="Write one diagnostic PNG under /tmp without opening a window")
    args = parser.parse_args()
    if args.interval <= 0 or not np.isfinite(args.interval):
        parser.error("--interval must be finite and positive")
    if args.snapshot is not None:
        try:
            args.snapshot.resolve().relative_to("/tmp")
        except ValueError:
            parser.error("Diagnostic snapshots must be under /tmp")
    import matplotlib
    if args.snapshot is not None:
        matplotlib.use("Agg")
    else:
        os.environ.setdefault("QT_API", "pyqt5")
        matplotlib.use("Qt5Agg")
    viewer = TrainingViewer(args.run.resolve(), args.inputs.resolve(), subdivisions=args.subdivisions)
    if args.snapshot is not None:
        viewer.figure.savefig(args.snapshot, dpi=100)
        print(str(args.snapshot.resolve()), flush=True)
        return
    viewer.start(args.interval)


if __name__ == "__main__":
    main()
