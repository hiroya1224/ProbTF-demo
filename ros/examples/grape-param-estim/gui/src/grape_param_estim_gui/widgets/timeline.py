from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import FlightResult, SharedPosterior


class SignalPanel(QWidget):
    """Tabbed, linked plots used for trajectory, correction, or residual views."""

    _SERIES_STYLES = {
        "reference": ("Reference", (80, 80, 80), "┄┄"),
        "observed": ("Observed", (30, 90, 190), "━━"),
        "nominal": ("Nominal", (210, 105, 30), "┄┄"),
        "interval": ("90% interval", (30, 150, 95), "■"),
        "center": ("Posterior mean", (30, 150, 95), "━━"),
        "selected": ("Selected member", (150, 45, 170), "━━"),
        "current": ("Current time", (35, 35, 35), "┃"),
    }

    currentTimeRequested = Signal(float)
    viewRangeRequested = Signal(float, float)
    estimationRangeEdited = Signal(float, float)

    def __init__(
        self,
        kind: str,
        parameter_ensemble: SharedPosterior | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if kind not in {"trajectory", "correction", "residual"}:
            raise ValueError(f"Unsupported panel kind: {kind}")

        self.kind = kind
        self.parameter_ensemble = parameter_ensemble
        self.session: FlightResult | None = None
        self.selected_member_id = (
            None if parameter_ensemble is None or parameter_ensemble.size == 0
            else int(parameter_ensemble.member_id[0])
        )
        self.current_time = 0.0
        self._updating_view = False

        self.plots: list[pg.PlotWidget] = []
        self.current_lines: list[pg.InfiniteLine] = []
        self.auto_regions: list[pg.LinearRegionItem] = []
        self.estimation_regions: list[pg.LinearRegionItem] = []
        self._auto_estimation_range = (0.0, 1.0)
        self._estimation_range = (0.0, 1.0)
        self._estimation_movable = True
        self._updating_estimation_regions = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        legend_group = QGroupBox("Visible series")
        legend_layout = QGridLayout(legend_group)
        legend_layout.setContentsMargins(7, 5, 7, 5)
        legend_layout.setHorizontalSpacing(10)
        legend_layout.setVerticalSpacing(2)
        self.series_checkboxes: dict[str, QCheckBox] = {}
        for index, key in enumerate(self._series_keys()):
            text, color, swatch_text = self._SERIES_STYLES[key]
            text = self._series_text(key, text)
            item = QWidget()
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(3)

            swatch = QLabel(swatch_text)
            swatch.setFixedWidth(24)
            swatch.setAlignment(Qt.AlignCenter)
            swatch.setStyleSheet(
                f"color: rgb({color[0]}, {color[1]}, {color[2]}); font-weight: 700;"
            )
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_series_visibility_changed)
            self.series_checkboxes[key] = checkbox

            item_layout.addWidget(swatch)
            item_layout.addWidget(checkbox)
            item_layout.addStretch(1)
            legend_layout.addWidget(item, index // 4, index % 4)
        layout.addWidget(legend_group)

        if self.kind == "trajectory":
            range_help = QLabel(
                "Analysis range: drag the green band or either edge in any component. "
                "The blue dashed band is the auto-detected range."
            )
            range_help.setWordWrap(True)
            range_help.setStyleSheet(
                "background: #eef7f1; color: #245b39; padding: 4px 7px; "
                "border-radius: 3px;"
            )
            layout.addWidget(range_help)

        self.component_tabs = QTabWidget()
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.setToolTip(
            "Reset the shared time range and this component's vertical range"
        )
        self.reset_view_button.clicked.connect(self.reset_current_view)
        self.component_tabs.setCornerWidget(
            self.reset_view_button,
            Qt.TopRightCorner,
        )
        layout.addWidget(self.component_tabs, 1)

        labels = self._labels()
        tab_labels = self._tab_labels()
        for index, (label, tab_label) in enumerate(zip(labels, tab_labels, strict=True)):
            plot = pg.PlotWidget()
            plot.setMinimumHeight(160)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setLabel("left", label)
            plot.setLabel("bottom", "time", units="s")
            plot.getPlotItem().setClipToView(True)
            plot.getPlotItem().setDownsampling(auto=True, mode="peak")
            view_box = plot.getViewBox()
            view_box.setDefaultPadding(0.08)
            view_box.setAutoVisible(y=True)
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.component_tabs.addTab(plot, tab_label)
            self.plots.append(plot)

            if self.kind == "trajectory":
                auto_region = pg.LinearRegionItem(
                    values=self._auto_estimation_range,
                    movable=False,
                    brush=pg.mkBrush(60, 120, 220, 20),
                    pen=pg.mkPen((60, 120, 220), width=1.2, style=Qt.DashLine),
                )
                auto_region.setZValue(-20)
                estimation_region = pg.LinearRegionItem(
                    values=self._estimation_range,
                    movable=True,
                    brush=pg.mkBrush(35, 160, 95, 32),
                    pen=pg.mkPen((35, 160, 95), width=1.8),
                )
                estimation_region.setZValue(-10)
                estimation_region.sigRegionChanged.connect(
                    lambda *_args, source=estimation_region: self._sync_estimation_regions(source)
                )
                estimation_region.sigRegionChangeFinished.connect(
                    lambda *_args, source=estimation_region: self._finish_estimation_region_edit(
                        source
                    )
                )
                self.auto_regions.append(auto_region)
                self.estimation_regions.append(estimation_region)

        for plot in self.plots[1:]:
            plot.setXLink(self.plots[0])

        self.plots[0].getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)
        for plot in self.plots:
            plot.scene().sigMouseClicked.connect(
                lambda event, target_plot=plot: self._on_mouse_clicked(event, target_plot)
            )

    def _series_keys(self) -> tuple[str, ...]:
        if self.kind == "trajectory":
            return (
                "reference",
                "observed",
                "nominal",
                "interval",
                "center",
                "selected",
                "current",
            )
        if self.kind == "correction":
            return ("observed", "nominal", "interval", "center", "selected", "current")
        return ("interval", "center", "selected", "current")

    def _series_text(self, key: str, default: str) -> str:
        if self.kind == "correction":
            return {
                "observed": "Observed correction (nominal⁻¹ · observed)",
                "nominal": "Zero / identity correction (controller nominal)",
                "interval": "90% posterior correction interval",
                "center": "Posterior correction mean",
                "selected": "Selected member correction",
            }.get(key, default)
        if self.kind == "residual":
            return {
                "interval": "90% residual-wrench interval",
                "center": "Residual-wrench mean",
                "selected": "Selected member residual wrench",
            }.get(key, default)
        return default

    def _tab_labels(self) -> Sequence[str]:
        if self.kind == "trajectory":
            return ("position x", "position y", "position z", "roll", "pitch", "yaw")
        if self.kind == "correction":
            return (
                "translation x",
                "translation y",
                "translation z",
                "rotation-vector x",
                "rotation-vector y",
                "rotation-vector z",
            )
        return ("force x", "force y", "force z", "torque x", "torque y", "torque z")

    def _labels(self) -> Sequence[str]:
        if self.kind == "trajectory":
            return (
                "position x [m]",
                "position y [m]",
                "position z [m]",
                "roll [rad]",
                "pitch [rad]",
                "yaw [rad]",
            )
        if self.kind == "correction":
            return (
                "correction tx [m]",
                "correction ty [m]",
                "correction tz [m]",
                "correction rx [rad]",
                "correction ry [rad]",
                "correction rz [rad]",
            )
        return (
            "residual body force x [N]",
            "residual body force y [N]",
            "residual body force z [N]",
            "residual body torque x [N m]",
            "residual body torque y [N m]",
            "residual body torque z [N m]",
        )

    def set_session(self, session: FlightResult | None) -> None:
        self.session = session
        self._render()

    def set_estimation_ranges(
        self,
        auto_range: tuple[float, float],
        selected_range: tuple[float, float],
    ) -> None:
        auto_start, auto_end = sorted(float(value) for value in auto_range)
        selected_start, selected_end = sorted(float(value) for value in selected_range)
        self._auto_estimation_range = (auto_start, auto_end)
        self._estimation_range = (selected_start, selected_end)
        self._update_estimation_region_values()

    def set_estimation_range(self, start: float, end: float) -> None:
        selected_start, selected_end = sorted((float(start), float(end)))
        self._estimation_range = (selected_start, selected_end)
        self._update_estimation_region_values()

    def set_estimation_movable(self, movable: bool) -> None:
        self._estimation_movable = bool(movable)
        for region in self.estimation_regions:
            region.setMovable(self._estimation_movable)

    def set_parameter_ensemble(self, ensemble: SharedPosterior | None) -> None:
        self.parameter_ensemble = ensemble
        if ensemble is None or ensemble.size == 0:
            self.selected_member_id = None
        elif self.selected_member_id not in set(ensemble.member_id.tolist()):
            self.selected_member_id = int(ensemble.member_id[0])
        self._render()

    def set_selected_member(self, member_id: int | None) -> None:
        self.selected_member_id = None if member_id is None else int(member_id)
        if self.session is not None:
            self._render()

    def set_current_time(self, value: float) -> None:
        self.current_time = float(value)
        for line in self.current_lines:
            line.setValue(self.current_time)

    def set_view_range(self, start: float, end: float) -> None:
        if self.session is None:
            return
        self._updating_view = True
        try:
            self.plots[0].setXRange(float(start), float(end), padding=0.0)
        finally:
            self._updating_view = False

    def fit_all(self) -> None:
        if self.session is None:
            return
        self.set_view_range(float(self.session.time[0]), float(self.session.time[-1]))

    def reset_current_view(self) -> None:
        if self.session is None:
            return

        start = float(self.session.time[0])
        end = float(self.session.time[-1])
        self.set_view_range(start, end)
        self.viewRangeRequested.emit(start, end)

        index = max(0, self.component_tabs.currentIndex())
        view_box = self.plots[index].getViewBox()
        view_box.setAutoVisible(y=True)
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        view_box.updateAutoRange()

    def _update_estimation_region_values(self) -> None:
        if not self.estimation_regions:
            return
        self._updating_estimation_regions = True
        try:
            for auto_region, estimation_region in zip(
                self.auto_regions,
                self.estimation_regions,
                strict=True,
            ):
                auto_region.setRegion(self._auto_estimation_range)
                estimation_region.setRegion(self._estimation_range)
        finally:
            self._updating_estimation_regions = False

    def _sync_estimation_regions(self, source: pg.LinearRegionItem) -> None:
        if self._updating_estimation_regions:
            return
        start, end = sorted(float(value) for value in source.getRegion())
        self._estimation_range = (start, end)
        self._updating_estimation_regions = True
        try:
            for region in self.estimation_regions:
                if region is not source:
                    region.setRegion(self._estimation_range)
        finally:
            self._updating_estimation_regions = False

    def _finish_estimation_region_edit(self, source: pg.LinearRegionItem) -> None:
        if self._updating_estimation_regions:
            return
        self._sync_estimation_regions(source)
        self.estimationRangeEdited.emit(*self._estimation_range)

    def _on_series_visibility_changed(self, _checked: bool) -> None:
        if self.session is None:
            return
        x_range = tuple(float(value) for value in self.plots[0].viewRange()[0])
        self._render()
        self.set_view_range(*x_range)

    def _render(self) -> None:
        self.current_lines.clear()
        for plot in self.plots:
            plot.clear()

        if self.session is None:
            return

        time = (
            self.session.time[:-1]
            if self.kind == "residual" and self.session.residual_wrench is not None
            else self.session.time
        )
        observed, nominal, center, lower, upper, selected, reference = self._extract_series()

        pens = {
            "reference": pg.mkPen((80, 80, 80), width=1.2, style=Qt.DashLine),
            "observed": pg.mkPen((30, 90, 190), width=2.0),
            "nominal": pg.mkPen((210, 105, 30), width=1.8, style=Qt.DashLine),
            "center": pg.mkPen((30, 150, 95), width=2.0),
            "selected": pg.mkPen((150, 45, 170), width=2.4),
        }
        band_brush = pg.mkBrush(30, 150, 95, 45)

        for component, plot in enumerate(self.plots):
            if reference is not None and self._series_is_visible("reference"):
                plot.plot(time, reference[:, component], pen=pens["reference"], name="reference")
            if observed is not None and self._series_is_visible("observed"):
                plot.plot(time, observed[:, component], pen=pens["observed"], name="observed")
            if nominal is not None and self._series_is_visible("nominal"):
                plot.plot(time, nominal[:, component], pen=pens["nominal"], name="nominal")

            if lower is not None and upper is not None and self._series_is_visible("interval"):
                lower_curve = plot.plot(time, lower[:, component], pen=pg.mkPen(None))
                upper_curve = plot.plot(time, upper[:, component], pen=pg.mkPen(None))
                plot.addItem(pg.FillBetweenItem(lower_curve, upper_curve, brush=band_brush))
            if center is not None and self._series_is_visible("center"):
                plot.plot(time, center[:, component], pen=pens["center"], name="posterior center")
            if selected is not None and self._series_is_visible("selected"):
                plot.plot(
                    time,
                    selected[:, component],
                    pen=pens["selected"],
                    name="selected member",
                )

            if self._series_is_visible("current"):
                current_line = pg.InfiniteLine(
                    angle=90,
                    movable=False,
                    pen=pg.mkPen((35, 35, 35), width=1.2),
                )
                current_line.setValue(self.current_time)
                plot.addItem(current_line)
                self.current_lines.append(current_line)

            view_box = plot.getViewBox()
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            view_box.setAutoVisible(y=True)

        self._attach_estimation_regions()
        self.fit_all()

    def _attach_estimation_regions(self) -> None:
        if self.session is None or not self.estimation_regions:
            return

        bounds = (float(self.session.time[0]), float(self.session.time[-1]))
        self._updating_estimation_regions = True
        try:
            for plot, auto_region, estimation_region in zip(
                self.plots,
                self.auto_regions,
                self.estimation_regions,
                strict=True,
            ):
                auto_region.setBounds(bounds)
                auto_region.setRegion(self._auto_estimation_range)
                estimation_region.setBounds(bounds)
                estimation_region.setRegion(self._estimation_range)
                estimation_region.setMovable(self._estimation_movable)
                plot.addItem(auto_region)
                plot.addItem(estimation_region)
        finally:
            self._updating_estimation_regions = False

    def _series_is_visible(self, key: str) -> bool:
        checkbox = self.series_checkboxes.get(key)
        return checkbox is not None and checkbox.isChecked()

    def _extract_series(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        assert self.session is not None
        session = self.session
        if self.kind == "trajectory":
            observed = np.concatenate([session.observed_position, session.observed_rpy], axis=1)
            reference = np.concatenate([session.reference_position, session.reference_rpy], axis=1)
            nominal = (
                None if session.nominal_position is None or session.nominal_rpy is None
                else np.concatenate([session.nominal_position, session.nominal_rpy], axis=1)
            )
            members = (
                None if session.member_position is None or session.member_rpy is None
                else np.concatenate([session.member_position, session.member_rpy], axis=2)
            )
        elif self.kind == "correction":
            observed = (
                None
                if session.observed_correction_translation is None
                or session.observed_correction_rotation_vector is None
                else np.concatenate(
                    [
                        session.observed_correction_translation,
                        session.observed_correction_rotation_vector,
                    ],
                    axis=1,
                )
            )
            nominal = None if observed is None else np.zeros_like(observed)
            reference = None
            members = (
                None
                if session.correction_translation is None
                or session.correction_rotation_vector is None
                else np.concatenate(
                    [
                        session.correction_translation,
                        session.correction_rotation_vector,
                    ],
                    axis=2,
                )
            )
        else:
            observed = None
            nominal = None
            reference = None
            members = session.residual_wrench
        if members is None or members.shape[0] == 0:
            return observed, nominal, None, None, None, None, reference
        center = np.mean(members, axis=0)
        lower = np.quantile(members, 0.05, axis=0)
        upper = np.quantile(members, 0.95, axis=0)
        selected = None
        if self.parameter_ensemble is not None and self.selected_member_id is not None:
            matches = np.flatnonzero(
                self.parameter_ensemble.member_id == self.selected_member_id
            )
            if matches.size:
                selected = members[int(matches[0])]
        return observed, nominal, center, lower, upper, selected, reference

    def _on_mouse_clicked(self, event: object, plot: pg.PlotWidget) -> None:
        if not hasattr(event, "button") or event.button() != Qt.LeftButton:
            return
        scene_position = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_position):
            return
        mapped = plot.getViewBox().mapSceneToView(scene_position)
        self.currentTimeRequested.emit(float(mapped.x()))

    def _on_x_range_changed(self, _view_box: object, x_range: tuple[float, float]) -> None:
        if self._updating_view:
            return
        self.viewRangeRequested.emit(float(x_range[0]), float(x_range[1]))
