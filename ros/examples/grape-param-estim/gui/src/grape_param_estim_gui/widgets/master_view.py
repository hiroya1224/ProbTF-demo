from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import AssimilationRun, SharedPosterior
from ..presentation import member_parameter_text
from ..state import ProjectStore


def _metric(value: object, default: float = float("nan")) -> float:
    if isinstance(value, Mapping):
        for key in ("value", "fraction", "mean"):
            if key in value:
                return _metric(value[key], default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _text_mode(value: object) -> str:
    if value is None or np.asarray(value).size == 0:
        return "—"
    return str(np.asarray(value).reshape(-1)[0])


class RidgePlotWidget(QWidget):
    memberSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameter_ensemble: SharedPosterior | None = None
        self.selected_member_id: int | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.trans_plot = pg.PlotWidget(title="mass × mean force effectiveness")
        self.trans_plot.setLabel("bottom", "mass", units="kg")
        self.trans_plot.setLabel("left", "mean force effectiveness")
        self.rot_plot = pg.PlotWidget(title="inertia trace × mean torque effectiveness")
        self.rot_plot.setLabel("bottom", "inertia trace", units="kg m²")
        self.rot_plot.setLabel("left", "mean torque effectiveness")
        for plot in (self.trans_plot, self.rot_plot):
            plot.showGrid(x=True, y=True, alpha=0.18)
            layout.addWidget(plot, 1)
        self._selected_items: list[pg.ScatterPlotItem] = []

    def set_ensemble(self, ensemble: SharedPosterior | None) -> None:
        self.parameter_ensemble = ensemble
        self.selected_member_id = (
            None if ensemble is None or ensemble.size == 0 else int(ensemble.member_id[0])
        )
        self._render()

    def set_selected_member(self, member_id: int | None) -> None:
        self.selected_member_id = None if member_id is None else int(member_id)
        self._render_selected()

    def _coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.parameter_ensemble is not None
        ensemble = self.parameter_ensemble
        return (
            ensemble.mass,
            np.mean(ensemble.force_effectiveness, axis=1),
            np.trace(ensemble.inertia, axis1=1, axis2=2),
            np.mean(ensemble.torque_effectiveness, axis=1),
        )

    def _render(self) -> None:
        for plot in (self.trans_plot, self.rot_plot):
            plot.clear()
        ensemble = self.parameter_ensemble
        if ensemble is None or ensemble.size == 0:
            return
        mass, force, inertia, torque = self._coordinates()
        for plot, x_values, y_values in (
            (self.trans_plot, mass, force),
            (self.rot_plot, inertia, torque),
        ):
            spots = [
                {
                    "pos": (float(x), float(y)),
                    "data": int(member_id),
                    "size": 8.0,
                    "brush": pg.mkBrush(55, 145, 105, 125),
                    "pen": pg.mkPen(35, 105, 75, 130),
                }
                for x, y, member_id in zip(x_values, y_values, ensemble.member_id, strict=True)
            ]
            scatter = pg.ScatterPlotItem(spots=spots, hoverable=True)
            scatter.sigClicked.connect(self._on_points_clicked)
            plot.addItem(scatter)
            if not self._add_expected_ridge_direction(plot, x_values, y_values):
                self._add_principal_direction(plot, x_values, y_values)
        self._render_selected()

    def _render_selected(self) -> None:
        for item in self._selected_items:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        self._selected_items.clear()
        ensemble = self.parameter_ensemble
        if ensemble is None or self.selected_member_id is None:
            return
        matches = np.flatnonzero(ensemble.member_id == self.selected_member_id)
        if not matches.size:
            return
        index = int(matches[0])
        mass, force, inertia, torque = self._coordinates()
        for plot, x_values, y_values in (
            (self.trans_plot, mass, force),
            (self.rot_plot, inertia, torque),
        ):
            item = pg.ScatterPlotItem(
                [float(x_values[index])], [float(y_values[index])], size=16,
                brush=pg.mkBrush(210, 70, 205, 210),
                pen=pg.mkPen(95, 25, 95, width=2.2),
            )
            plot.addItem(item)
            self._selected_items.append(item)

    @staticmethod
    def _add_principal_direction(
        plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        if x_values.size < 2:
            return
        points = np.column_stack((x_values, y_values))
        center = np.mean(points, axis=0)
        covariance = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        maximum = max(float(np.max(eigenvalues)), 0.0)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        endpoints = np.vstack(
            (center - 1.7 * np.sqrt(maximum) * direction, center + 1.7 * np.sqrt(maximum) * direction)
        )
        plot.plot(
            endpoints[:, 0], endpoints[:, 1],
            pen=pg.mkPen((70, 70, 70), width=1.4, style=Qt.DashLine),
        )

    def _add_expected_ridge_direction(
        self, plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray
    ) -> bool:
        ensemble = self.parameter_ensemble
        if ensemble is None or "expected_direction" not in ensemble.ridge:
            return False
        direction = np.asarray(ensemble.ridge["expected_direction"], dtype=float)
        coordinates = np.asarray(ensemble.parameter_coordinate, dtype=float)
        if direction.shape != (coordinates.shape[1],) or coordinates.shape[0] < 2:
            return False
        ridge_coordinate = (coordinates - np.mean(coordinates, axis=0)) @ direction
        variance = float(np.dot(ridge_coordinate, ridge_coordinate))
        if variance <= 0.0:
            return False
        centered_x = x_values - np.mean(x_values)
        centered_y = y_values - np.mean(y_values)
        slope = np.array(
            [np.dot(ridge_coordinate, centered_x), np.dot(ridge_coordinate, centered_y)]
        ) / variance
        bounds = np.quantile(ridge_coordinate, (0.05, 0.95))
        center = np.array([np.mean(x_values), np.mean(y_values)])
        endpoints = center[None, :] + bounds[:, None] * slope[None, :]
        plot.plot(
            endpoints[:, 0], endpoints[:, 1],
            pen=pg.mkPen((70, 70, 70), width=1.6, style=Qt.DashLine),
        )
        return True

    def _on_points_clicked(
        self, _item: pg.ScatterPlotItem, points: list[pg.SpotItem], _event: object
    ) -> None:
        if points:
            self.memberSelected.emit(int(points[0].data()))


class MasterView(QWidget):
    bagActivated = Signal(str)

    def __init__(self, store: ProjectStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self._refreshing = False
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.configuration_warning = QLabel()
        self.configuration_warning.setWordWrap(True)
        root.addWidget(self.configuration_warning)

        top_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(top_splitter, 3)
        bag_group = QGroupBox("Bags participating in joint smoothing")
        bag_layout = QVBoxLayout(bag_group)
        self.bag_table = QTableWidget(0, 10)
        self.bag_table.setHorizontalHeaderLabels(
            [
                "Use", "Bag", "Config", "Auto interval", "Selected interval",
                "State", "Samples", "Status", "Objective", "Coverage",
            ]
        )
        self.bag_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bag_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.bag_table.setAlternatingRowColors(True)
        self.bag_table.verticalHeader().setVisible(False)
        self.bag_table.horizontalHeader().setStretchLastSection(True)
        self.bag_table.itemChanged.connect(self._on_table_item_changed)
        self.bag_table.cellDoubleClicked.connect(self._on_bag_double_clicked)
        bag_layout.addWidget(self.bag_table)
        top_splitter.addWidget(bag_group)

        ridge_group = QGroupBox("Shared static parameter ensemble (equal-weight raw members)")
        ridge_layout = QVBoxLayout(ridge_group)
        self.ridge_widget = RidgePlotWidget()
        self.ridge_widget.set_ensemble(store.parameter_ensemble)
        ridge_layout.addWidget(self.ridge_widget)
        self.member_detail = QLabel("No completed assimilation run")
        self.member_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.member_detail.setWordWrap(True)
        ridge_layout.addWidget(self.member_detail)
        top_splitter.addWidget(ridge_group)
        top_splitter.setSizes([620, 760])

        bottom_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(bottom_splitter, 2)
        contribution_group = QGroupBox("Bag contribution and consistency")
        contribution_layout = QVBoxLayout(contribution_group)
        self.objective_plot = pg.PlotWidget(title="per-bag objective contribution")
        self.coverage_plot = pg.PlotWidget(title="correction-path coverage")
        for plot in (self.objective_plot, self.coverage_plot):
            plot.showGrid(x=True, y=True, alpha=0.18)
            contribution_layout.addWidget(plot)
        self.objective_plot.setLabel("left", "objective")
        self.coverage_plot.setLabel("left", "coverage")
        self.coverage_plot.setYRange(0.0, 1.05)
        bottom_splitter.addWidget(contribution_group)

        diagnostic_group = QGroupBox("IEnKS-Q diagnostics")
        diagnostic_layout = QVBoxLayout(diagnostic_group)
        self.iteration_plot = pg.PlotWidget(title="objective by iteration")
        self.iteration_plot.showGrid(x=True, y=True, alpha=0.18)
        self.iteration_plot.setLabel("bottom", "iteration")
        self.iteration_plot.setLabel("left", "objective")
        diagnostic_layout.addWidget(self.iteration_plot)
        self.diagnostic_label = QLabel(
            "termination: not run\nraw members: —\n"
            "residual-wrench Q time resolution: pending"
        )
        self.diagnostic_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.setToolTip(
            "Residual-wrench time resolution is sufficient when the calibrated "
            "OU process is represented by enough knots for the selected interval."
        )
        diagnostic_layout.addWidget(self.diagnostic_label)
        bottom_splitter.addWidget(diagnostic_group)
        bottom_splitter.setSizes([660, 620])

        self.ridge_widget.memberSelected.connect(self.store.set_selected_member)
        self.store.bagsChanged.connect(self.refresh)
        self.store.recordChanged.connect(lambda _bag_id: self.refresh())
        self.store.selectedMemberChanged.connect(self._on_selected_member_changed)
        self.store.posteriorChanged.connect(self.set_run)
        self.refresh()
        self._on_selected_member_changed(self.store.selected_member_id)

    @staticmethod
    def _range_text(value: tuple[float, float]) -> str:
        return "{:.2f}–{:.2f} s".format(*value) if value[1] > value[0] else "—"

    def refresh(self) -> None:
        self._refreshing = True
        try:
            records = self.store.records()
            self.bag_table.setRowCount(len(records))
            for row, record in enumerate(records):
                use_item = QTableWidgetItem()
                use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
                use_item.setCheckState(Qt.Checked if record.included else Qt.Unchecked)
                use_item.setData(Qt.UserRole, record.bag_id)
                self.bag_table.setItem(row, 0, use_item)
                data = record.data
                objective = None if record.result is None else record.result.objective_contribution
                coverage = None if record.result is None else _metric(record.result.coverage)
                values = (
                    record.display_name,
                    record.configuration_group,
                    self._range_text(record.auto_range),
                    self._range_text(record.selected_range),
                    record.interval_state,
                    "—" if data is None else str(data.sample_count),
                    record.status,
                    "—" if objective is None else "{:.4g}".format(float(objective)),
                    "—" if coverage is None or not np.isfinite(coverage) else "{:.3f}".format(coverage),
                )
                for column, text in enumerate(values, start=1):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, record.bag_id)
                    self.bag_table.setItem(row, column, item)
            self.bag_table.resizeColumnsToContents()
            self._refresh_contribution_plots()
            self._refresh_configuration_warning()
        finally:
            self._refreshing = False

    def set_run(self, run: AssimilationRun | None) -> None:
        if run is None:
            self.ridge_widget.set_ensemble(None)
            self.iteration_plot.clear()
            self.diagnostic_label.setText(
                "termination: not run\nraw members: —\n"
                "residual-wrench Q time resolution: pending"
            )
            self.member_detail.setText("No completed assimilation run")
            self.refresh()
            return
        self.ridge_widget.set_ensemble(run.shared_posterior)
        diagnostics = run.diagnostics
        history = next(
            (
                np.asarray(diagnostics[key], dtype=float).reshape(-1)
                for key in ("objective_history", "objective", "iteration_objective")
                if key in diagnostics
            ),
            np.empty(0),
        )
        self.iteration_plot.clear()
        if history.size:
            self.iteration_plot.plot(
                np.arange(history.size), history,
                pen=pg.mkPen((45, 120, 185), width=2.0), symbol="o", symbolSize=6,
            )
        termination = str(run.manifest.get("termination_reason", "unknown"))
        convergence = bool(run.manifest.get("converged", False))
        q_warning_count = sum(
            record.result is not None
            and record.result.q_resolution_sufficient is False
            for record in self.store.records()
        )
        run_warning_text = (
            "none"
            if not run.warnings
            else "\n  - " + "\n  - ".join(str(value) for value in run.warnings)
        )
        accepted = diagnostics.get("accepted_fraction")
        gradient = diagnostics.get("gradient_norm")
        step = diagnostics.get("step_norm")
        mode_ids = run.shared_posterior.mode.get("mode_id")
        mode_weights = run.shared_posterior.mode.get("mode_weight")
        selected_mode = run.shared_posterior.mode.get("selected_mode_id")
        mode_text = "—" if mode_ids is None else "{} weights {} (selected {})".format(
            np.asarray(mode_ids).astype(str).tolist(),
            np.asarray(mode_weights).tolist() if mode_weights is not None else "—",
            _text_mode(selected_mode),
        )
        self.diagnostic_label.setText(
            "termination: {}{}\nraw members: {} (equal weight)\n"
            "accepted fraction: {}\ngradient norm: {}\nstep norm: {}\n"
            "residual-wrench Q time-resolution insufficient bags: {}\n"
            "run warnings: {}\nmode law: {}".format(
                termination,
                " (converged)" if convergence else " (not labelled converged)",
                run.shared_posterior.size,
                "—" if accepted is None else np.array2string(np.asarray(accepted), precision=4),
                "—" if gradient is None else np.array2string(np.asarray(gradient), precision=4),
                "—" if step is None else np.array2string(np.asarray(step), precision=4),
                q_warning_count,
                run_warning_text,
                mode_text,
            )
        )
        self._on_selected_member_changed(self.store.selected_member_id)
        self.refresh()

    def _refresh_contribution_plots(self) -> None:
        records = self.store.records()
        self.objective_plot.clear()
        self.coverage_plot.clear()
        if not records:
            return
        x_values = np.arange(len(records), dtype=float)
        objectives = np.array(
            [
                np.nan if record.result is None or record.result.objective_contribution is None
                else record.result.objective_contribution
                for record in records
            ], dtype=float,
        )
        coverage = np.array(
            [np.nan if record.result is None else _metric(record.result.coverage) for record in records],
            dtype=float,
        )
        for plot, values, color in (
            (self.objective_plot, objectives, (75, 130, 190, 150)),
            (self.coverage_plot, coverage, (50, 155, 105, 150)),
        ):
            finite = np.isfinite(values)
            if np.any(finite):
                plot.addItem(pg.BarGraphItem(x=x_values[finite], height=values[finite], width=0.62, brush=pg.mkBrush(*color)))
            plot.getAxis("bottom").setTicks([[(float(i), str(i + 1)) for i in range(len(records))]])

    def _refresh_configuration_warning(self) -> None:
        groups = {record.configuration_group for record in self.store.included_records()}
        if len(groups) <= 1:
            self.configuration_warning.setText(
                "Shared-parameter configuration fingerprint: {}".format(next(iter(groups), "none"))
            )
            self.configuration_warning.setStyleSheet("")
        else:
            self.configuration_warning.setText(
                "Selected bags have different configuration fingerprints; "
                "joint smoothing requires explicit mismatch confirmation."
            )
            self.configuration_warning.setStyleSheet(
                "background: #fff1c7; color: #6c4a00; padding: 6px; border-radius: 4px;"
            )

    def _on_selected_member_changed(self, member_id: int | None) -> None:
        self.ridge_widget.set_selected_member(member_id)
        ensemble = self.store.parameter_ensemble
        if ensemble is None or member_id is None:
            self.member_detail.setText("No completed assimilation run")
            return
        matches = np.flatnonzero(ensemble.member_id == member_id)
        if not matches.size:
            return
        self.member_detail.setText(member_parameter_text(ensemble, member_id))

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._refreshing and item.column() == 0:
            self.store.set_included(
                str(item.data(Qt.UserRole)), item.checkState() == Qt.Checked
            )

    def _on_bag_double_clicked(self, row: int, _column: int) -> None:
        item = self.bag_table.item(row, 1)
        if item is not None:
            self.bagActivated.emit(str(item.data(Qt.UserRole)))


__all__ = ["MasterView", "RidgePlotWidget"]
