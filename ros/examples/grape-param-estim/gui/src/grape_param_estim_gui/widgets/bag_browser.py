from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import (
    BagEstimationResult,
    BatchEstimationRun,
    ConditionalTrajectory,
    FlightResult,
    SelectedTrajectorySet,
)
from ..state import BagRecord, ProjectStore, TimeState

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover - depends on optional visual stack
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]


def _record_time(record: BagRecord | None) -> np.ndarray | None:
    if record is None or record.data is None:
        return None
    if isinstance(record.data, BagEstimationResult):
        return record.data.knot_time
    return record.data.time


def _interval_time(time: np.ndarray) -> np.ndarray:
    value = np.asarray(time, dtype=float)
    return 0.5 * (value[:-1] + value[1:])


class BatchTrajectoryScene(QWidget):
    """PyVista overview driven directly by MAP and selected trajectories."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview: FlightResult | None = None
        self.result: BagEstimationResult | None = None
        self.subset: SelectedTrajectorySet | None = None
        self.selected: ConditionalTrajectory | None = None
        self.current_time = 0.0
        self.view_mode = "world"
        self.layers = {
            "reference": True,
            "observed": True,
            "nominal": True,
            "map": True,
            "posterior": True,
            "selected": True,
        }
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        disabled = os.environ.get("GRAPE_PARAM_ESTIM_DISABLE_3D") == "1"
        if QtInteractor is None or pv is None or disabled:
            self.plotter = None
            self.status_label = QLabel(
                "3D viewer disabled. Trajectory arrays remain visible in the "
                "linked component plots below."
            )
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("padding: 24px; font-size: 14px;")
            layout.addWidget(self.status_label)
            return
        self.status_label = None
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#20242b")
        self.plotter.add_axes()

    def set_context(
        self,
        preview: FlightResult | None,
        result: BagEstimationResult | None,
        subset: SelectedTrajectorySet | None,
        selected: ConditionalTrajectory | None,
    ) -> None:
        self.preview = preview
        self.result = result
        self.subset = subset
        self.selected = selected
        time = (
            result.knot_time
            if result is not None
            else None if preview is None else preview.time
        )
        if time is not None and time.size:
            self.current_time = float(time[0])
        self.rebuild_scene()

    def set_current_time(self, value: float) -> None:
        self.current_time = float(value)
        if self.plotter is not None:
            self.plotter.add_text(
                "t = {:.3f} s".format(self.current_time),
                position="lower_left",
                font_size=10,
                name="time-label",
            )
            self.plotter.render()

    def set_view_mode(self, mode: str) -> None:
        if mode not in {"world", "correction"}:
            raise ValueError("unknown trajectory scene mode {!r}".format(mode))
        if mode != self.view_mode:
            self.view_mode = mode
            self.rebuild_scene()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        if layer in self.layers:
            self.layers[layer] = bool(visible)
            self.rebuild_scene()

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self.plotter.render()

    def _world_paths(self) -> list[tuple[str, np.ndarray, str, float]]:
        paths: list[tuple[str, np.ndarray, str, float]] = []
        if self.result is not None:
            result = self.result
            if self.layers["reference"]:
                paths.append(
                    ("Reference", result.reference.position, "#aaaaaa", 2.0)
                )
            if self.layers["observed"]:
                paths.append(
                    (
                        "Observed pose",
                        result.pose.position[result.pose.valid],
                        "#3d8bf2",
                        4.0,
                    )
                )
            if self.layers["nominal"]:
                paths.append(
                    ("Nominal", result.nominal.position, "#ec8a2f", 2.5)
                )
            if self.layers["map"]:
                paths.append(
                    ("MAP trajectory", result.map_trajectory.position, "#34b77b", 3.4)
                )
        elif self.preview is not None:
            if self.layers["reference"]:
                paths.append(
                    ("Reference", self.preview.reference_position, "#aaaaaa", 2.0)
                )
            if self.layers["observed"]:
                paths.append(
                    ("Observed pose", self.preview.observed_position, "#3d8bf2", 4.0)
                )
        if self.layers["posterior"] and self.subset is not None:
            for index, points in enumerate(self.subset.conditional_position):
                paths.append(
                    (
                        "stored sample {}".format(index),
                        points,
                        "#6bc493",
                        0.8,
                    )
                )
        if self.layers["selected"] and self.selected is not None:
            paths.append(
                (
                    "Selected conditional sample",
                    self.selected.state.position,
                    "#d45bd4",
                    4.0,
                )
            )
        return paths

    def _correction_paths(self) -> list[tuple[str, np.ndarray, str, float]]:
        paths: list[tuple[str, np.ndarray, str, float]] = []
        if self.result is not None:
            zero = np.zeros_like(self.result.correction_translation)
            if self.layers["nominal"]:
                paths.append(("Nominal identity", zero, "#ec8a2f", 2.0))
            if self.layers["map"]:
                paths.append(
                    (
                        "MAP correction",
                        self.result.correction_translation,
                        "#34b77b",
                        3.4,
                    )
                )
        if self.layers["posterior"] and self.subset is not None:
            for index, points in enumerate(self.subset.correction_translation):
                paths.append(
                    (
                        "stored correction {}".format(index),
                        points,
                        "#6bc493",
                        0.8,
                    )
                )
        if self.layers["selected"] and self.selected is not None:
            paths.append(
                (
                    "Selected correction",
                    self.selected.correction_translation,
                    "#d45bd4",
                    4.0,
                )
            )
        return paths

    def rebuild_scene(self) -> None:
        paths = (
            self._world_paths()
            if self.view_mode == "world"
            else self._correction_paths()
        )
        if self.plotter is None:
            if self.status_label is not None:
                self.status_label.setText(
                    "3D viewer disabled. {} path(s) are loaded and remain "
                    "visible in the linked plots below.".format(len(paths))
                )
            return
        self.plotter.clear()
        self.plotter.add_axes()
        legend: list[tuple[str, str]] = []
        for name, points, color, width in paths:
            value = np.asarray(points, dtype=float)
            finite = np.all(np.isfinite(value), axis=1)
            value = value[finite]
            if value.shape[0] < 2:
                continue
            polyline = pv.lines_from_points(value)
            self.plotter.add_mesh(
                polyline,
                color=color,
                line_width=width,
                opacity=0.18 if name.startswith("stored") else 1.0,
                render_lines_as_tubes=width >= 2.0,
                name=name,
            )
            if not name.startswith("stored"):
                legend.append((name, color))
        self.plotter.add_text(
            "World trajectory"
            if self.view_mode == "world"
            else "Nominal-to-estimated correction",
            position="upper_left",
            font_size=11,
        )
        self.plotter.add_text(
            "t = {:.3f} s".format(self.current_time),
            position="lower_left",
            font_size=10,
            name="time-label",
        )
        if legend:
            self.plotter.add_legend(legend, bcolor="#20242b")
        if paths:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
        self.plotter.render()

    def close_scene(self) -> None:
        if self.plotter is not None:
            self.plotter.close()


class BatchSignalPanel(QWidget):
    """Linked component plots that retain each stream's true time axis."""

    currentTimeRequested = Signal(float)
    viewRangeRequested = Signal(float, float)
    estimationRangeEdited = Signal(float, float)

    _COLORS = {
        "reference": (80, 80, 80),
        "observed": (30, 90, 190),
        "nominal": (210, 105, 30),
        "map": (30, 150, 95),
        "selected": (150, 45, 170),
        "posterior": (75, 175, 120),
        "q_band": (190, 105, 35),
        "normalized": (55, 55, 55),
    }

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if kind not in {"trajectory", "correction", "dynamics"}:
            raise ValueError("unsupported panel kind {!r}".format(kind))
        self.kind = kind
        self.preview: FlightResult | None = None
        self.result: BagEstimationResult | None = None
        self.subset: SelectedTrajectorySet | None = None
        self.selected: ConditionalTrajectory | None = None
        self.run: BatchEstimationRun | None = None
        self.current_time = 0.0
        self.series_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._view_update = False
        self._auto_range = (0.0, 1.0)
        self._estimation_range = (0.0, 1.0)
        self._estimation_movable = True
        self._updating_regions = False
        self.q_reference_label = "Q reference unavailable"
        self.current_lines: list[pg.InfiniteLine] = []
        self.auto_regions: list[pg.LinearRegionItem] = []
        self.estimation_regions: list[pg.LinearRegionItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.dynamics_display_combo: QComboBox | None = None
        if self.kind == "dynamics":
            self.dynamics_display_combo = QComboBox()
            self.dynamics_display_combo.addItem(
                "Physical residual and Q reference band", "physical"
            )
            self.dynamics_display_combo.addItem(
                "Normalized residual", "normalized"
            )
            self.dynamics_display_combo.currentIndexChanged.connect(
                lambda _index: self._render()
            )
            layout.addWidget(self.dynamics_display_combo)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.plots: list[pg.PlotWidget] = []
        for label in self._tab_labels():
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setLabel("bottom", "time", units="s")
            plot.getPlotItem().setClipToView(True)
            plot.getPlotItem().setDownsampling(auto=True, mode="peak")
            plot.scene().sigMouseClicked.connect(
                lambda event, target=plot: self._mouse_clicked(event, target)
            )
            self.tabs.addTab(plot, label)
            self.plots.append(plot)
            if kind == "trajectory":
                auto_region = pg.LinearRegionItem(
                    values=self._auto_range,
                    movable=False,
                    brush=pg.mkBrush(60, 120, 220, 20),
                    pen=pg.mkPen((60, 120, 220), style=Qt.DashLine),
                )
                selected_region = pg.LinearRegionItem(
                    values=self._estimation_range,
                    movable=True,
                    brush=pg.mkBrush(35, 160, 95, 32),
                    pen=pg.mkPen((35, 160, 95), width=1.8),
                )
                selected_region.sigRegionChanged.connect(
                    lambda *_args, source=selected_region: self._sync_regions(source)
                )
                selected_region.sigRegionChangeFinished.connect(
                    lambda *_args, source=selected_region: self._finish_region(source)
                )
                self.auto_regions.append(auto_region)
                self.estimation_regions.append(selected_region)
        for plot in self.plots[1:]:
            plot.setXLink(self.plots[0])
        self.plots[0].getViewBox().sigXRangeChanged.connect(
            self._x_range_changed
        )

    def _tab_labels(self) -> tuple[str, ...]:
        if self.kind == "trajectory":
            return ("position x", "position y", "position z", "roll", "pitch", "yaw")
        if self.kind == "correction":
            return ("translation x", "translation y", "translation z", "rotvec x", "rotvec y", "rotvec z")
        return ("force x", "force y", "force z", "torque x", "torque y", "torque z")

    def set_context(
        self,
        preview: FlightResult | None,
        result: BagEstimationResult | None,
        subset: SelectedTrajectorySet | None,
        selected: ConditionalTrajectory | None,
        run: BatchEstimationRun | None,
    ) -> None:
        self.preview = preview
        self.result = result
        self.subset = subset
        self.selected = selected
        self.run = run
        self._render()

    def set_estimation_ranges(
        self,
        auto_range: tuple[float, float],
        selected_range: tuple[float, float],
    ) -> None:
        self._auto_range = tuple(sorted(float(value) for value in auto_range))
        self._estimation_range = tuple(
            sorted(float(value) for value in selected_range)
        )
        self._update_regions()

    def set_estimation_range(self, start: float, end: float) -> None:
        self._estimation_range = tuple(sorted((float(start), float(end))))
        self._update_regions()

    def set_estimation_movable(self, movable: bool) -> None:
        self._estimation_movable = bool(movable)
        for region in self.estimation_regions:
            region.setMovable(self._estimation_movable)

    def set_current_time(self, value: float) -> None:
        self.current_time = float(value)
        for line in self.current_lines:
            line.setValue(self.current_time)

    def set_view_range(self, start: float, end: float) -> None:
        self._view_update = True
        try:
            self.plots[0].setXRange(float(start), float(end), padding=0.0)
        finally:
            self._view_update = False

    def _trajectory_series(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.result is not None:
            result = self.result
            if result.reference.time.size:
                series["reference"] = (
                    result.reference.time,
                    np.concatenate((result.reference.position, result.reference.rpy), axis=1),
                )
            if result.pose.time.size:
                valid = result.pose.valid
                series["observed"] = (
                    result.pose.time[valid],
                    np.concatenate((result.pose.position[valid], result.pose.rpy[valid]), axis=1),
                )
            series["nominal"] = (
                result.knot_time,
                np.concatenate((result.nominal.position, result.nominal.rpy), axis=1),
            )
            series["map"] = (
                result.knot_time,
                np.concatenate((result.map_trajectory.position, result.map_trajectory.rpy), axis=1),
            )
        elif self.preview is not None:
            series["reference"] = (
                self.preview.time,
                np.concatenate((self.preview.reference_position, self.preview.reference_rpy), axis=1),
            )
            series["observed"] = (
                self.preview.time,
                np.concatenate((self.preview.observed_position, self.preview.observed_rpy), axis=1),
            )
        if self.subset is not None and self.subset.sample_id.size:
            # Position quantiles are Euclidean.  Euler-angle quantiles would
            # be wrong across the ±pi branch, so the orientation band remains
            # absent until an SO(3)-tangent credible band is available.
            lower_position = np.quantile(
                self.subset.conditional_position, 0.05, axis=0
            )
            upper_position = np.quantile(
                self.subset.conditional_position, 0.95, axis=0
            )
            absent_orientation = np.full(
                (self.subset.knot_time.size, 3), np.nan
            )
            series["posterior_lower"] = (
                self.subset.knot_time,
                np.concatenate((lower_position, absent_orientation), axis=1),
            )
            series["posterior_upper"] = (
                self.subset.knot_time,
                np.concatenate((upper_position, absent_orientation), axis=1),
            )
        if self.selected is not None:
            series["selected"] = (
                self.selected.knot_time,
                np.concatenate(
                    (self.selected.state.position, self.selected.state.rpy), axis=1
                ),
            )
        return series

    def _correction_series(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.result is not None:
            values = np.concatenate(
                (
                    self.result.correction_translation,
                    self.result.correction_rotation_vector,
                ),
                axis=1,
            )
            series["nominal"] = (
                self.result.knot_time,
                np.zeros_like(values),
            )
            series["map"] = (self.result.knot_time, values)
        if self.subset is not None and self.subset.sample_id.size:
            values = np.concatenate(
                (
                    self.subset.correction_translation,
                    self.subset.correction_rotation_vector,
                ),
                axis=2,
            )
            series["posterior_lower"] = (
                self.subset.knot_time,
                np.quantile(values, 0.05, axis=0),
            )
            series["posterior_upper"] = (
                self.subset.knot_time,
                np.quantile(values, 0.95, axis=0),
            )
        if self.selected is not None:
            series["selected"] = (
                self.selected.knot_time,
                np.concatenate(
                    (
                        self.selected.correction_translation,
                        self.selected.correction_rotation_vector,
                    ),
                    axis=1,
                ),
            )
        return series

    def _dynamics_series(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.result is None or self.run is None:
            return {}
        result = self.result
        time = _interval_time(result.knot_time)
        valid = result.map_dynamics_residual_valid
        map_residual = result.map_dynamics_residual
        dt = np.diff(result.knot_time)
        definition = str(self.run.manifest["q_definition"]["definition"])
        if definition.endswith("/continuous_spectral_density"):
            reference_scale = np.sqrt(
                self.run.static_map.q_diagonal[None, :] / dt[:, None]
            )
            self.q_reference_label = "sqrt(Q/Δt)"
        elif definition.endswith("/fixed_interval_covariance"):
            reference_scale = np.sqrt(
                np.broadcast_to(
                    self.run.static_map.q_diagonal[None, :],
                    (dt.size, self.run.static_map.q_diagonal.size),
                )
            )
            self.q_reference_label = "sqrt(Q)"
        else:
            raise ValueError(
                "unsupported Q interval definition {!r}".format(definition)
            )
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "map": (time[valid], map_residual[valid]),
            "q_upper": (time, reference_scale),
            "q_lower": (time, -reference_scale),
            "map_normalized": (time[valid], (map_residual / reference_scale)[valid]),
        }
        if self.subset is not None and self.subset.sample_id.size:
            residual = np.where(
                self.subset.dynamics_residual_valid[:, :, None],
                self.subset.dynamics_residual,
                np.nan,
            )
            series["posterior_lower"] = (
                time,
                np.nanquantile(residual, 0.05, axis=0),
            )
            series["posterior_upper"] = (
                time,
                np.nanquantile(residual, 0.95, axis=0),
            )
        if self.selected is not None:
            selected_valid = self.selected.dynamics_residual_valid
            selected_residual = self.selected.dynamics_residual
            series["selected"] = (
                time[selected_valid],
                selected_residual[selected_valid],
            )
            series["selected_normalized"] = (
                time[selected_valid],
                (selected_residual / reference_scale)[selected_valid],
            )
        return series

    def _render(self) -> None:
        for plot in self.plots:
            plot.clear()
        self.current_lines.clear()
        if self.kind == "trajectory":
            self.series_data = self._trajectory_series()
            self.status_label.setText(
                "Reference, asynchronous observed pose, nominal, full-trajectory MAP, "
                "and the selected stored conditional sample."
            )
        elif self.kind == "correction":
            self.series_data = self._correction_series()
            self.status_label.setText(
                "Correction is nominal-to-MAP or nominal-to-selected conditional trajectory."
            )
        else:
            self.series_data = self._dynamics_series()
            normalized = (
                self.dynamics_display_combo is not None
                and self.dynamics_display_combo.currentData() == "normalized"
            )
            self.status_label.setText(
                "Dynamics residual is a factor residual, not a latent wrench. "
                + (
                    "Dashed orange: ±1 after normalization by {}.".format(
                        self.q_reference_label
                    )
                    if normalized
                    else "Dashed orange: ±{}.".format(self.q_reference_label)
                )
            )

        lower = self.series_data.get("posterior_lower")
        upper = self.series_data.get("posterior_upper")
        normalized_dynamics = (
            self.kind == "dynamics"
            and self.dynamics_display_combo is not None
            and self.dynamics_display_combo.currentData() == "normalized"
        )
        for component, plot in enumerate(self.plots):
            if self.kind == "trajectory":
                plot.setLabel(
                    "left",
                    (("position [m]",) * 3 + ("angle [rad]",) * 3)[component],
                )
            elif self.kind == "correction":
                plot.setLabel(
                    "left",
                    (
                        ("translation [m]",) * 3
                        + ("rotation vector [rad]",) * 3
                    )[component],
                )
            elif normalized_dynamics:
                plot.setLabel("left", "normalized residual")
            else:
                unit = (
                    "residual"
                    if self.run is None
                    else str(self.run.manifest["q_definition"]["units"][component])
                )
                plot.setLabel("left", "dynamics residual [{}]".format(unit))
            visible_keys = (
                ("map_normalized", "selected_normalized")
                if normalized_dynamics
                else ("reference", "observed", "nominal", "map", "selected")
            )
            for key in visible_keys:
                item = self.series_data.get(key)
                if item is None:
                    continue
                time, values = item
                style = Qt.DashLine if key in {"reference", "nominal"} else Qt.SolidLine
                color_key = (
                    "selected"
                    if key == "selected_normalized"
                    else "normalized" if key == "map_normalized" else key
                )
                plot.plot(
                    time,
                    values[:, component],
                    pen=pg.mkPen(
                        self._COLORS[color_key],
                        width=2.4 if color_key == "selected" else 1.8,
                        style=style,
                    ),
                    name=key,
                )
            if lower is not None and upper is not None and not normalized_dynamics:
                lower_curve = plot.plot(
                    lower[0], lower[1][:, component], pen=pg.mkPen(None)
                )
                upper_curve = plot.plot(
                    upper[0], upper[1][:, component], pen=pg.mkPen(None)
                )
                plot.addItem(
                    pg.FillBetweenItem(
                        lower_curve,
                        upper_curve,
                        brush=pg.mkBrush(75, 175, 120, 42),
                    )
                )
            if (
                self.kind == "dynamics"
                and not normalized_dynamics
                and "q_upper" in self.series_data
                and "q_lower" in self.series_data
            ):
                for key in ("q_upper", "q_lower"):
                    time, values = self.series_data[key]
                    plot.plot(
                        time,
                        values[:, component],
                        pen=pg.mkPen(self._COLORS["q_band"], width=1.4, style=Qt.DashLine),
                    )
            elif normalized_dynamics and "q_upper" in self.series_data:
                reference_time = self.series_data["q_upper"][0]
                for value in (-1.0, 1.0):
                    plot.plot(
                        reference_time,
                        np.full(reference_time.shape, value),
                        pen=pg.mkPen(
                            self._COLORS["q_band"],
                            width=1.4,
                            style=Qt.DashLine,
                        ),
                    )
            line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen((35, 35, 35), width=1.2),
            )
            line.setValue(self.current_time)
            plot.addItem(line)
            self.current_lines.append(line)
        self._attach_regions()

    def _attach_regions(self) -> None:
        if not self.estimation_regions:
            return
        time = (
            self.result.knot_time
            if self.result is not None
            else None if self.preview is None else self.preview.time
        )
        if time is None or not time.size:
            return
        bounds = (float(time[0]), float(time[-1]))
        for plot, auto_region, selected_region in zip(
            self.plots,
            self.auto_regions,
            self.estimation_regions,
            strict=True,
        ):
            auto_region.setBounds(bounds)
            auto_region.setRegion(self._auto_range)
            selected_region.setBounds(bounds)
            selected_region.setRegion(self._estimation_range)
            selected_region.setMovable(self._estimation_movable)
            plot.addItem(auto_region)
            plot.addItem(selected_region)

    def _update_regions(self) -> None:
        self._updating_regions = True
        try:
            for auto_region, selected_region in zip(
                self.auto_regions,
                self.estimation_regions,
                strict=True,
            ):
                auto_region.setRegion(self._auto_range)
                selected_region.setRegion(self._estimation_range)
        finally:
            self._updating_regions = False

    def _sync_regions(self, source: pg.LinearRegionItem) -> None:
        if self._updating_regions:
            return
        self._estimation_range = tuple(
            sorted(float(value) for value in source.getRegion())
        )
        self._updating_regions = True
        try:
            for region in self.estimation_regions:
                if region is not source:
                    region.setRegion(self._estimation_range)
        finally:
            self._updating_regions = False

    def _finish_region(self, source: pg.LinearRegionItem) -> None:
        if not self._updating_regions:
            self._sync_regions(source)
            self.estimationRangeEdited.emit(*self._estimation_range)

    def _x_range_changed(self, _view_box: object, value: object) -> None:
        if not self._view_update:
            start, end = value
            self.viewRangeRequested.emit(float(start), float(end))

    def _mouse_clicked(self, event: Any, plot: pg.PlotWidget) -> None:
        if event.button() != Qt.LeftButton or not plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = plot.getViewBox().mapSceneToView(event.scenePos())
        self.currentTimeRequested.emit(float(point.x()))


class BagBrowserView(QWidget):
    statusMessage = Signal(str)
    filesSelected = Signal(object)
    reinspectionRequested = Signal(object)
    configurationRequested = Signal(str)
    configurationGroupRequested = Signal(str)

    def __init__(self, store: ProjectStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.time_state = TimeState(self)
        self.current_record: BagRecord | None = None
        self._refreshing_table = False
        self._loading_record = False

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(33)
        self.playback_timer.timeout.connect(self._advance_playback)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(main_splitter)
        main_splitter.addWidget(self._build_left_panel())

        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([380, 1100])
        scene_splitter = QSplitter(Qt.Horizontal)
        self.scene = BatchTrajectoryScene()
        scene_splitter.addWidget(self.scene)
        self.scene_controls_scroll = QScrollArea()
        self.scene_controls_scroll.setWidgetResizable(True)
        self.scene_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene_controls_scroll.setFrameShape(QScrollArea.NoFrame)
        self.scene_controls_scroll.setMinimumWidth(220)
        self.scene_controls_scroll.setWidget(self._build_scene_controls())
        scene_splitter.addWidget(self.scene_controls_scroll)
        scene_splitter.setSizes([900, 240])
        right_splitter.addWidget(scene_splitter)

        self.signal_tabs = QTabWidget()
        self.trajectory_panel = BatchSignalPanel("trajectory")
        self.correction_panel = BatchSignalPanel("correction")
        self.dynamics_panel = BatchSignalPanel("dynamics")
        self.signal_tabs.addTab(self.trajectory_panel, "Trajectory")
        self.signal_tabs.addTab(self.correction_panel, "Correction transform")
        self.signal_tabs.addTab(self.dynamics_panel, "Dynamics residual")
        self.flight_state_plot = pg.PlotWidget(title="Recorded flight state")
        self.flight_state_plot.showGrid(x=True, y=True, alpha=0.18)
        self.signal_tabs.addTab(self.flight_state_plot, "Flight state")
        right_splitter.addWidget(self.signal_tabs)
        right_splitter.setSizes([470, 690])

        self._connect_state()
        self.refresh_table()
        if self.store.current_record() is not None:
            self._load_record(self.store.current_record())

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        source_group = QGroupBox("Bag files")
        source_layout = QVBoxLayout(source_group)
        button_row = QHBoxLayout()
        self.add_files_button = QPushButton("Add files…")
        self.remove_button = QPushButton("Remove")
        button_row.addWidget(self.add_files_button)
        button_row.addWidget(self.remove_button)
        source_layout.addLayout(button_row)
        self.bag_table = QTableWidget(0, 7)
        self.bag_table.setHorizontalHeaderLabels(
            ["Use", "Bag", "Group", "Auto", "Selected", "State", "Status"]
        )
        self.bag_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bag_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.bag_table.setAlternatingRowColors(True)
        self.bag_table.verticalHeader().setVisible(False)
        self.bag_table.horizontalHeader().setStretchLastSection(True)
        self.bag_table.itemChanged.connect(self._on_table_item_changed)
        self.bag_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        source_layout.addWidget(self.bag_table, 1)
        layout.addWidget(source_group, 3)

        interval_group = QGroupBox("Estimation window")
        interval_layout = QVBoxLayout(interval_group)
        buttons = QHBoxLayout()
        self.auto_detect_button = QPushButton("Re-inspect")
        self.restore_auto_button = QPushButton("Restore auto")
        self.lock_button = QPushButton("Lock")
        self.lock_button.setCheckable(True)
        for button in (self.auto_detect_button, self.restore_auto_button, self.lock_button):
            buttons.addWidget(button)
        interval_layout.addLayout(buttons)
        form = QFormLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" s")
        form.addRow("start", self.start_spin)
        form.addRow("end", self.end_spin)
        interval_layout.addLayout(form)
        self.interval_state_label = QLabel("No bag selected")
        interval_layout.addWidget(self.interval_state_label)
        layout.addWidget(interval_group)

        details_group = QGroupBox("Current bag")
        details = QFormLayout(details_group)
        self.path_label = QLabel("—")
        self.path_label.setWordWrap(True)
        self.include_checkbox = QCheckBox("Use this bag in batch estimation")
        self.confirm_group_button = QPushButton("Confirm configuration group…")
        self.configure_button = QPushButton("Set configuration provenance…")
        self.group_label = QLabel("—")
        self.samples_label = QLabel("—")
        self.sha_label = QLabel("—")
        self.inspection_details = QLabel("Inspection pending")
        self.inspection_details.setWordWrap(True)
        self.inspection_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details.addRow("path", self.path_label)
        details.addRow("group", self.group_label)
        details.addRow("samples", self.samples_label)
        details.addRow("SHA256", self.sha_label)
        details.addRow(self.include_checkbox)
        details.addRow(self.confirm_group_button)
        details.addRow(self.configure_button)
        details.addRow("sensor status", self.inspection_details)
        layout.addWidget(details_group)

        self.add_files_button.clicked.connect(self.open_files)
        self.remove_button.clicked.connect(self.remove_current)
        self.auto_detect_button.clicked.connect(self.auto_detect_current_interval)
        self.restore_auto_button.clicked.connect(self.restore_auto_interval)
        self.lock_button.toggled.connect(self._set_interval_locked)
        self.start_spin.valueChanged.connect(self._on_numeric_interval_changed)
        self.end_spin.valueChanged.connect(self._on_numeric_interval_changed)
        self.include_checkbox.toggled.connect(self._on_include_checkbox_toggled)
        self.confirm_group_button.clicked.connect(
            lambda: self.configurationGroupRequested.emit(self.current_record.bag_id)
            if self.current_record is not None
            else None
        )
        self.configure_button.clicked.connect(
            lambda: self.configurationRequested.emit(self.current_record.bag_id)
            if self.current_record is not None
            else None
        )
        return panel

    def _build_scene_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        view_group = QGroupBox("3D view")
        view_layout = QVBoxLayout(view_group)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("World trajectory", "world")
        self.view_mode_combo.addItem("Correction transform", "correction")
        self.reset_camera_button = QPushButton("Reset camera")
        view_layout.addWidget(self.view_mode_combo)
        view_layout.addWidget(self.reset_camera_button)
        layout.addWidget(view_group)
        layer_group = QGroupBox("Layers")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_checkboxes: dict[str, QCheckBox] = {}
        for layer, text in (
            ("reference", "Reference"),
            ("observed", "Observed pose"),
            ("nominal", "Nominal"),
            ("map", "MAP"),
            ("posterior", "Stored posterior subset"),
            ("selected", "Selected conditional sample"),
        ):
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, name=layer: self.scene.set_layer_visible(name, checked)
            )
            layer_layout.addWidget(checkbox)
            self.layer_checkboxes[layer] = checkbox
        layout.addWidget(layer_group)
        selection_group = QGroupBox("Shared selection")
        selection_layout = QFormLayout(selection_group)
        self.current_time_spin = QDoubleSpinBox()
        self.current_time_spin.setDecimals(3)
        self.current_time_spin.setSuffix(" s")
        self.sample_label = QLabel(self.store.selected_sample_id or "MAP only")
        self.sample_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        selection_layout.addRow("current time", self.current_time_spin)
        selection_layout.addRow("sample ID", self.sample_label)
        layout.addWidget(selection_group)
        layout.addStretch(1)
        self.view_mode_combo.currentIndexChanged.connect(
            lambda _index: self.scene.set_view_mode(str(self.view_mode_combo.currentData()))
        )
        self.reset_camera_button.clicked.connect(self.scene.reset_camera)
        self.current_time_spin.valueChanged.connect(self.time_state.set_current_time)
        return panel

    def _connect_state(self) -> None:
        self.store.bagsChanged.connect(self.refresh_table)
        self.store.recordChanged.connect(self._on_record_changed)
        self.store.currentBagChanged.connect(self._load_record)
        self.store.selectedSampleChanged.connect(self._on_selected_sample_changed)
        self.store.posteriorChanged.connect(lambda _run: self._reload_context())
        self.time_state.currentTimeChanged.connect(self.scene.set_current_time)
        for panel in (self.trajectory_panel, self.correction_panel, self.dynamics_panel):
            self.time_state.currentTimeChanged.connect(panel.set_current_time)
            self.time_state.viewRangeChanged.connect(panel.set_view_range)
            panel.currentTimeRequested.connect(self.time_state.set_current_time)
            panel.viewRangeRequested.connect(self.time_state.set_view_range)
        self.time_state.currentTimeChanged.connect(self._store_current_time)
        self.time_state.currentTimeChanged.connect(self._update_current_time_spin)
        self.time_state.estimationRangeChanged.connect(self.trajectory_panel.set_estimation_range)
        self.time_state.viewRangeChanged.connect(self._store_view_range)
        self.trajectory_panel.estimationRangeEdited.connect(self._on_estimation_region_edited)

    def refresh_table(self) -> None:
        selected_bag_id = self.store.current_bag_id
        self._refreshing_table = True
        try:
            records = self.store.records()
            self.bag_table.setRowCount(len(records))
            selected_row = -1
            for row, record in enumerate(records):
                use_item = QTableWidgetItem()
                flags = use_item.flags() | Qt.ItemIsUserCheckable
                if record.status not in {"ready", "complete"}:
                    flags &= ~Qt.ItemIsEnabled
                use_item.setFlags(flags)
                use_item.setCheckState(Qt.Checked if record.included else Qt.Unchecked)
                use_item.setData(Qt.UserRole, record.bag_id)
                self.bag_table.setItem(row, 0, use_item)
                values = (
                    record.display_name,
                    record.configuration_group,
                    self._format_range(record.auto_range),
                    self._format_range(record.selected_range),
                    record.interval_state,
                    record.status,
                )
                for column, value in enumerate(values, start=1):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, record.bag_id)
                    self.bag_table.setItem(row, column, item)
                if record.bag_id == selected_bag_id:
                    selected_row = row
            self.bag_table.resizeColumnsToContents()
            if selected_row >= 0:
                self.bag_table.selectRow(selected_row)
        finally:
            self._refreshing_table = False

    def open_files(self) -> None:
        files, _filter = QFileDialog.getOpenFileNames(
            self, "Add rosbag files", str(Path.home()), "ROS bags (*.bag *.db3);;All files (*)"
        )
        if files:
            self.filesSelected.emit(tuple(Path(value) for value in files))

    def remove_current(self) -> None:
        if self.current_record is not None:
            self.store.remove(self.current_record.bag_id)

    def auto_detect_current_interval(self) -> None:
        if self.current_record is not None:
            self.reinspectionRequested.emit((self.current_record.bag_id,))

    def restore_auto_interval(self) -> None:
        if self.current_record is not None:
            self.store.restore_auto_interval(self.current_record.bag_id)
            self.time_state.set_estimation_range(*self.current_record.auto_range)
            self._update_interval_controls(self.current_record)

    def toggle_playback(self) -> None:
        if self.current_record is None:
            return
        self.time_state.set_playing(not self.time_state.playing)
        if self.time_state.playing:
            self.playback_timer.start()
        else:
            self.playback_timer.stop()

    def stop_playback(self) -> None:
        self.playback_timer.stop()
        self.time_state.set_playing(False)

    def set_playback_speed(self, speed: float) -> None:
        self.time_state.set_playback_speed(speed)

    def step_samples(self, sample_delta: int) -> None:
        time = _record_time(self.current_record)
        if time is None or not time.size:
            return
        index = int(np.argmin(np.abs(time - self.time_state.current_time)))
        index = int(np.clip(index + int(sample_delta), 0, time.size - 1))
        self.time_state.set_current_time(float(time[index]))

    def fit_view(self) -> None:
        time = _record_time(self.current_record)
        if time is not None and time.size:
            self.time_state.set_view_range(float(time[0]), float(time[-1]))

    def clear_inspection_selection(self) -> None:
        if self.current_record is not None:
            self.time_state.set_current_time(self.current_record.selected_range[0])

    def close_scene(self) -> None:
        self.scene.close_scene()

    def _selected_context(
        self,
    ) -> tuple[
        FlightResult | None,
        BagEstimationResult | None,
        SelectedTrajectorySet | None,
        ConditionalTrajectory | None,
    ]:
        record = self.current_record
        preview = None if record is None else record.preview
        result = None if record is None else record.result
        run = self.store.estimation_run
        subset = (
            None
            if record is None or run is None
            else run.selected_trajectories.get(record.bag_id)
        )
        selected = (
            None
            if record is None or run is None or self.store.selected_sample_id is None
            else run.selected_trajectory(record.bag_id, self.store.selected_sample_id)
        )
        return preview, result, subset, selected

    def _reload_context(self) -> None:
        preview, result, subset, selected = self._selected_context()
        self.scene.set_context(preview, result, subset, selected)
        for panel in (self.trajectory_panel, self.correction_panel, self.dynamics_panel):
            panel.set_context(preview, result, subset, selected, self.store.estimation_run)

    def _load_record(self, record: BagRecord | None) -> None:
        self._loading_record = True
        try:
            self.current_record = record
            self._reload_context()
            self._render_flight_state(record)
            if record is None:
                return
            time = _record_time(record)
            if time is None or not time.size:
                self._update_details(record)
                self._select_table_row(record.bag_id)
                return
            minimum, maximum = float(time[0]), float(time[-1])
            for spin in (self.start_spin, self.end_spin, self.current_time_spin):
                spin.setRange(minimum, maximum)
            self.time_state.current_time = float(np.clip(record.current_time, minimum, maximum))
            self.time_state.estimation_start, self.time_state.estimation_end = record.selected_range
            view_start = max(record.view_range[0], minimum)
            view_end = min(record.view_range[1], maximum)
            if view_end <= view_start:
                view_start, view_end = minimum, maximum
            self.time_state.view_start, self.time_state.view_end = view_start, view_end
            self.time_state.currentTimeChanged.emit(self.time_state.current_time)
            self.time_state.estimationRangeChanged.emit(*record.selected_range)
            self.time_state.viewRangeChanged.emit(view_start, view_end)
            self._update_interval_controls(record)
            self._update_details(record)
            self._select_table_row(record.bag_id)
        finally:
            self._loading_record = False

    def _update_interval_controls(self, record: BagRecord) -> None:
        blockers = [QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin), QSignalBlocker(self.lock_button)]
        self.start_spin.setValue(record.selected_range[0])
        self.end_spin.setValue(record.selected_range[1])
        self.lock_button.setChecked(record.interval_state == "LOCKED")
        del blockers
        self.trajectory_panel.set_estimation_ranges(record.auto_range, record.selected_range)
        self.trajectory_panel.set_estimation_movable(record.interval_state != "LOCKED")
        self.interval_state_label.setText(
            "{} | auto {} | selected {}".format(
                record.interval_state,
                self._format_range(record.auto_range),
                self._format_range(record.selected_range),
            )
        )

    def _update_details(self, record: BagRecord) -> None:
        self.path_label.setText(str(record.path))
        self.group_label.setText(record.configuration_group)
        time = _record_time(record)
        self.samples_label.setText(
            "—"
            if time is None
            else "{} samples / {:.1f} s".format(time.size, float(time[-1] - time[0]))
        )
        self.sha_label.setText(record.sha256[:16] + "…")
        if record.result is not None:
            factor_lines = []
            for name, factor in record.result.observation_factors.items():
                factor_lines.append(
                    "{}: {}{}".format(
                        name,
                        "used" if factor["enabled"] else "disabled",
                        "" if factor["enabled"] else " ({})".format(factor["disabled_reason"]),
                    )
                )
            self.inspection_details.setText(
                "status: {}\n{}\nsensor contract: {}".format(
                    record.status,
                    "\n".join(factor_lines),
                    json.dumps(record.result.sensor_contract, ensure_ascii=False, sort_keys=True),
                )
            )
        elif record.inspection is None:
            self.inspection_details.setText("Inspection pending")
        else:
            self.inspection_details.setText(
                "status: {}\ntopic contract: {}\nwarnings: {}".format(
                    record.status,
                    json.dumps(record.inspection.get("topic_contract", {}), ensure_ascii=False, sort_keys=True),
                    "; ".join(str(value) for value in record.inspection.get("warnings", [])) or "none",
                )
            )
        blocker = QSignalBlocker(self.include_checkbox)
        self.include_checkbox.setChecked(record.included)
        self.include_checkbox.setEnabled(record.status in {"ready", "complete"})
        del blocker
        fingerprint = None if record.inspection is None else record.inspection.get("configuration_fingerprint")
        incomplete = isinstance(fingerprint, dict) and not bool(fingerprint.get("complete", False))
        needs_confirmation = record.status == "needs_configuration_confirmation"
        self.confirm_group_button.setVisible(needs_confirmation)
        self.confirm_group_button.setEnabled(needs_confirmation)
        self.configure_button.setVisible(incomplete)
        self.configure_button.setEnabled(incomplete)

    def _render_flight_state(self, record: BagRecord | None) -> None:
        self.flight_state_plot.clear()
        if record is None or record.preview is None or record.preview.flight_state is None:
            self.flight_state_plot.setTitle("Recorded flight state (not stored in batch run)")
            return
        self.flight_state_plot.setTitle("Recorded flight state")
        self.flight_state_plot.plot(
            record.preview.time,
            record.preview.flight_state,
            pen=pg.mkPen((45, 120, 185), width=1.8),
        )

    def _on_selected_sample_changed(self, sample_id: str | None) -> None:
        self.sample_label.setText(sample_id or "MAP only")
        self._reload_context()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._refreshing_table and item.column() == 0:
            self.store.set_included(str(item.data(Qt.UserRole)), item.checkState() == Qt.Checked)

    def _on_table_selection_changed(self) -> None:
        if self._refreshing_table:
            return
        rows = self.bag_table.selectionModel().selectedRows()
        if rows:
            item = self.bag_table.item(rows[0].row(), 1)
            if item is not None:
                self.store.set_current(str(item.data(Qt.UserRole)))

    def _select_table_row(self, bag_id: str) -> None:
        for row in range(self.bag_table.rowCount()):
            item = self.bag_table.item(row, 1)
            if item is not None and str(item.data(Qt.UserRole)) == bag_id:
                self.bag_table.selectRow(row)
                return

    def _on_include_checkbox_toggled(self, checked: bool) -> None:
        if self.current_record is not None:
            self.store.set_included(self.current_record.bag_id, checked)

    def _set_interval_locked(self, locked: bool) -> None:
        if self.current_record is None or self._loading_record:
            return
        self.store.update_interval(
            self.current_record.bag_id,
            (self.start_spin.value(), self.end_spin.value()),
            state="LOCKED" if locked else "MODIFIED",
        )
        self._update_interval_controls(self.current_record)

    def _on_numeric_interval_changed(self, _value: float) -> None:
        if self.current_record is None or self._loading_record or self.lock_button.isChecked():
            return
        if self.start_spin.value() < self.end_spin.value():
            self.store.update_interval(
                self.current_record.bag_id,
                (self.start_spin.value(), self.end_spin.value()),
                state="MODIFIED",
            )
            self.time_state.set_estimation_range(self.start_spin.value(), self.end_spin.value())

    def _on_estimation_region_edited(self, start: float, end: float) -> None:
        if self.current_record is None or self.current_record.interval_state == "LOCKED":
            return
        self.store.update_interval(self.current_record.bag_id, (start, end), state="MODIFIED")
        self.time_state.set_estimation_range(start, end)
        self._update_interval_controls(self.current_record)

    def _advance_playback(self) -> None:
        time = _record_time(self.current_record)
        if time is None or not time.size:
            self.stop_playback()
            return
        target = self.time_state.current_time + 0.033 * self.time_state.playback_speed
        if target >= float(time[-1]):
            target = float(time[-1])
            self.stop_playback()
        self.time_state.set_current_time(target)

    def _store_current_time(self, value: float) -> None:
        if self.current_record is not None:
            self.current_record.current_time = float(value)

    def _store_view_range(self, start: float, end: float) -> None:
        if self.current_record is not None:
            self.current_record.view_range = (float(start), float(end))

    def _update_current_time_spin(self, value: float) -> None:
        blocker = QSignalBlocker(self.current_time_spin)
        self.current_time_spin.setValue(float(value))
        del blocker

    def _on_record_changed(self, bag_id: str) -> None:
        self.refresh_table()
        if self.current_record is not None and self.current_record.bag_id == bag_id:
            self._load_record(self.current_record)

    @staticmethod
    def _format_range(value: tuple[float, float]) -> str:
        return "{:.2f}–{:.2f}".format(*value) if value[1] > value[0] else "—"


__all__ = [
    "BagBrowserView",
    "BatchSignalPanel",
    "BatchTrajectoryScene",
]
