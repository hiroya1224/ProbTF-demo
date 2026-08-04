from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import BatchEstimationRun, McmcPosterior
from ..presentation import map_parameter_text, sample_parameter_text
from ..state import ProjectStore


Q_COMPONENT_LABELS = ("Fx", "Fy", "Fz", "τx", "τy", "τz")


class PosteriorPlotWidget(QWidget):
    """Compact MAP/MCMC projection and Laplace ridge diagnostics."""

    sampleSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.run: BatchEstimationRun | None = None
        self.selected_sample_id: str | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.trans_plot = pg.PlotWidget(
            title="mass × mean force effectiveness"
        )
        self.trans_plot.setLabel("bottom", "mass", units="kg")
        self.trans_plot.setLabel("left", "mean force effectiveness")
        self.rot_plot = pg.PlotWidget(
            title="inertia trace × mean torque effectiveness"
        )
        self.rot_plot.setLabel("bottom", "inertia trace", units="kg m²")
        self.rot_plot.setLabel("left", "mean torque effectiveness")
        self.geometry_plot = pg.PlotWidget(
            title="Laplace ridge directions in static coordinates"
        )
        self.geometry_plot.setLabel("bottom", "static coordinate index")
        self.geometry_plot.setLabel("left", "unit direction component")
        self.geometry_plot.addLegend()
        for plot in (self.trans_plot, self.rot_plot, self.geometry_plot):
            plot.showGrid(x=True, y=True, alpha=0.18)
            layout.addWidget(plot, 1)
        self._selected_items: list[pg.ScatterPlotItem] = []

    def set_run(self, run: BatchEstimationRun | None) -> None:
        self.run = run
        if run is None or run.mcmc is None:
            self.selected_sample_id = None
        elif self.selected_sample_id not in set(run.mcmc.sample_id.tolist()):
            self.selected_sample_id = str(run.mcmc.sample_id[0])
        self._render()

    def set_selected_sample(self, sample_id: str | None) -> None:
        self.selected_sample_id = (
            None if sample_id is None else str(sample_id)
        )
        self._render_selected()

    @staticmethod
    def _sample_coordinates(
        posterior: McmcPosterior,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            posterior.mass,
            np.mean(posterior.force_effectiveness, axis=1),
            np.trace(posterior.inertia, axis1=1, axis2=2),
            np.mean(posterior.torque_effectiveness, axis=1),
        )

    def _render(self) -> None:
        self._selected_items.clear()
        for plot in (self.trans_plot, self.rot_plot, self.geometry_plot):
            plot.clear()
        if self.run is None:
            return

        static_map = self.run.static_map
        map_coordinates = (
            static_map.mass,
            float(np.mean(static_map.force_effectiveness)),
            float(np.trace(static_map.inertia)),
            float(np.mean(static_map.torque_effectiveness)),
        )
        for plot, x_value, y_value in (
            (self.trans_plot, map_coordinates[0], map_coordinates[1]),
            (self.rot_plot, map_coordinates[2], map_coordinates[3]),
        ):
            plot.addItem(
                pg.ScatterPlotItem(
                    [x_value],
                    [y_value],
                    size=16,
                    symbol="star",
                    brush=pg.mkBrush(220, 115, 30, 230),
                    pen=pg.mkPen(120, 55, 10, width=1.5),
                )
            )

        posterior = self.run.mcmc
        if posterior is not None and posterior.size:
            mass, force, inertia, torque = self._sample_coordinates(posterior)
            for plot, x_values, y_values in (
                (self.trans_plot, mass, force),
                (self.rot_plot, inertia, torque),
            ):
                spots = [
                    {
                        "pos": (float(x), float(y)),
                        "data": str(sample_id),
                        "size": 8.0,
                        "brush": pg.mkBrush(55, 145, 105, 125),
                        "pen": pg.mkPen(35, 105, 75, 130),
                    }
                    for x, y, sample_id in zip(
                        x_values, y_values, posterior.sample_id, strict=True
                    )
                ]
                scatter = pg.ScatterPlotItem(spots=spots, hoverable=True)
                scatter.sigClicked.connect(self._on_points_clicked)
                plot.addItem(scatter)

        laplace = self.run.laplace
        coordinate = np.arange(laplace.exact_ridge_direction.size)
        numerical_index = int(np.argmin(np.abs(laplace.eigenvalues)))
        numerical_direction = laplace.eigenvectors[:, numerical_index]
        self.geometry_plot.plot(
            coordinate,
            laplace.exact_ridge_direction,
            pen=pg.mkPen((45, 105, 190), width=2.0),
            symbol="o",
            symbolSize=4,
            name="exact ridge",
        )
        self.geometry_plot.plot(
            coordinate,
            numerical_direction,
            pen=pg.mkPen((155, 55, 165), width=1.8, style=Qt.DashLine),
            symbol="t",
            symbolSize=4,
            name="numerical near-ridge",
        )
        self.geometry_plot.plot(
            coordinate,
            np.sqrt(np.maximum(np.diag(laplace.covariance), 0.0)),
            pen=pg.mkPen((225, 125, 35), width=1.8),
            symbol="s",
            symbolSize=4,
            name="Laplace marginal 1σ",
        )
        self._render_selected()

    def _render_selected(self) -> None:
        for plot, item in zip(
            (self.trans_plot, self.rot_plot),
            self._selected_items,
            strict=False,
        ):
            plot.removeItem(item)
        self._selected_items.clear()
        if (
            self.run is None
            or self.run.mcmc is None
            or self.selected_sample_id is None
        ):
            return
        try:
            index = self.run.mcmc.index_of(self.selected_sample_id)
        except KeyError:
            return
        mass, force, inertia, torque = self._sample_coordinates(self.run.mcmc)
        for plot, x_values, y_values in (
            (self.trans_plot, mass, force),
            (self.rot_plot, inertia, torque),
        ):
            item = pg.ScatterPlotItem(
                [float(x_values[index])],
                [float(y_values[index])],
                size=16,
                brush=pg.mkBrush(210, 70, 205, 210),
                pen=pg.mkPen(95, 25, 95, width=2.2),
            )
            plot.addItem(item)
            self._selected_items.append(item)

    def _on_points_clicked(
        self,
        _item: pg.ScatterPlotItem,
        points: list[pg.SpotItem],
        _event: object,
    ) -> None:
        if points:
            self.sampleSelected.emit(str(points[0].data()))


class McmcTraceWidget(QWidget):
    """Compact chain-selectable diagnostics without a large pair plot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.run: BatchEstimationRun | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("MCMC trace chain:"))
        self.chain_combo = QComboBox()
        self.chain_combo.currentIndexChanged.connect(self._render)
        selector.addWidget(self.chain_combo)
        selector.addStretch(1)
        root.addLayout(selector)
        self.ridge_plot = pg.PlotWidget(title="ridge coordinate trace")
        self.delay_plot = pg.PlotWidget(title="delay trace")
        self.log_posterior_plot = pg.PlotWidget(title="log posterior trace")
        self.delay_plot.setLabel("left", "delay", units="s")
        self.log_posterior_plot.setLabel("bottom", "retained draw")
        for plot in (
            self.ridge_plot,
            self.delay_plot,
            self.log_posterior_plot,
        ):
            plot.showGrid(x=True, y=True, alpha=0.18)
            root.addWidget(plot, 1)
        self.kernel_label = QLabel("MCMC traces unavailable.")
        self.kernel_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.kernel_label.setWordWrap(True)
        root.addWidget(self.kernel_label)

    def set_run(self, run: BatchEstimationRun | None) -> None:
        self.run = run
        self.chain_combo.blockSignals(True)
        self.chain_combo.clear()
        diagnostic = None if run is None else run.diagnostics.mcmc
        if diagnostic is not None:
            self.chain_combo.addItem("all chains", None)
            for index, chain_id in enumerate(diagnostic.chain_id):
                self.chain_combo.addItem(str(chain_id), index)
        self.chain_combo.blockSignals(False)
        self._render()

    def _selected_chain_indices(self) -> tuple[int, ...]:
        if self.run is None or self.run.diagnostics.mcmc is None:
            return ()
        selected = self.chain_combo.currentData()
        if selected is None:
            return tuple(range(self.run.diagnostics.mcmc.chain_id.size))
        return (int(selected),)

    def _render(self, _index: int = -1) -> None:
        for plot in (
            self.ridge_plot,
            self.delay_plot,
            self.log_posterior_plot,
        ):
            plot.clear()
        if self.run is None or self.run.diagnostics.mcmc is None:
            self.kernel_label.setText("MCMC traces unavailable.")
            return
        diagnostic = self.run.diagnostics.mcmc
        for color_index, chain_index in enumerate(
            self._selected_chain_indices()
        ):
            chain_id = str(diagnostic.chain_id[chain_index])
            pen = pg.mkPen(pg.intColor(color_index, hues=max(1, diagnostic.chain_id.size)), width=1.8)
            draw = np.arange(diagnostic.draws_per_chain, dtype=float)
            for plot, trace in (
                (self.ridge_plot, diagnostic.ridge_coordinate_trace),
                (self.delay_plot, diagnostic.delay_trace),
                (self.log_posterior_plot, diagnostic.log_posterior_trace),
            ):
                plot.plot(
                    draw,
                    trace[chain_index],
                    pen=pen,
                    name=chain_id,
                )
        kernel_parts = []
        for index, name in enumerate(diagnostic.kernel_names):
            kernel_parts.append(
                "{}: stage1 {}/{}, stage2 {}/{}, cache {}, inner failures {}, inner iterations {}".format(
                    name,
                    int(diagnostic.kernel_stage_one_accepted[index]),
                    int(diagnostic.kernel_attempts[index]),
                    int(diagnostic.kernel_stage_two_accepted[index]),
                    int(diagnostic.kernel_stage_two_attempted[index]),
                    int(diagnostic.kernel_full_target_cache_hits[index]),
                    int(diagnostic.kernel_inner_solve_failures[index]),
                    int(diagnostic.kernel_inner_iterations[index]),
                )
            )
        self.kernel_label.setText("Kernel diagnostics — " + "; ".join(kernel_parts))


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
        self.bag_group = QGroupBox("Bags in sparse batch estimation")
        bag_layout = QVBoxLayout(self.bag_group)
        self.bag_table = QTableWidget(0, 10)
        self.bag_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Bag",
                "Config",
                "Auto interval",
                "Selected interval",
                "State",
                "Knots",
                "Status",
                "Objective",
                "Sensor factors",
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
        top_splitter.addWidget(self.bag_group)

        posterior_group = QGroupBox(
            "Static parameter MAP, Laplace geometry, and MCMC posterior"
        )
        posterior_layout = QVBoxLayout(posterior_group)
        self.posterior_widget = PosteriorPlotWidget()
        self.posterior_widget.set_run(store.estimation_run)
        posterior_layout.addWidget(self.posterior_widget)
        self.sample_detail = QLabel("No completed sparse batch run")
        self.sample_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sample_detail.setWordWrap(True)
        posterior_layout.addWidget(self.sample_detail)
        top_splitter.addWidget(posterior_group)
        top_splitter.setSizes([620, 900])

        bottom_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(bottom_splitter, 2)
        contribution_group = QGroupBox("Bag objective and Laplace-EM history")
        contribution_layout = QVBoxLayout(contribution_group)
        self.objective_plot = pg.PlotWidget(title="per-bag MAP objective")
        self.em_plot = pg.PlotWidget(title="Laplace-EM accepted Q diagonal")
        self.em_objective_plot = pg.PlotWidget(
            title="Laplace-EM MAP and approximate marginal objectives"
        )
        for plot in (
            self.objective_plot,
            self.em_plot,
            self.em_objective_plot,
        ):
            plot.showGrid(x=True, y=True, alpha=0.18)
            contribution_layout.addWidget(plot)
        self.objective_plot.setLabel("left", "objective")
        self.em_plot.setLabel("bottom", "EM iteration")
        self.em_objective_plot.setLabel("bottom", "EM iteration")
        self.em_objective_plot.setLabel("left", "objective")
        self.em_plot.addLegend()
        self.em_objective_plot.addLegend()
        bottom_splitter.addWidget(contribution_group)

        self.diagnostic_group = QGroupBox("Batch posterior diagnostics")
        diagnostic_layout = QVBoxLayout(self.diagnostic_group)
        self.diagnostic_label = QLabel("run status: not run")
        self.diagnostic_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnostic_label.setWordWrap(True)
        diagnostic_layout.addWidget(self.diagnostic_label)
        self.mcmc_trace_widget = McmcTraceWidget()
        diagnostic_layout.addWidget(self.mcmc_trace_widget, 1)
        bottom_splitter.addWidget(self.diagnostic_group)
        bottom_splitter.setSizes([720, 560])

        self.posterior_widget.sampleSelected.connect(
            self.store.set_selected_sample
        )
        self.store.bagsChanged.connect(self.refresh)
        self.store.recordChanged.connect(lambda _bag_id: self.refresh())
        self.store.selectedSampleChanged.connect(
            self._on_selected_sample_changed
        )
        self.store.posteriorChanged.connect(self.set_run)
        self.set_run(self.store.estimation_run)

    @staticmethod
    def _range_text(value: tuple[float, float]) -> str:
        return "{:.2f}–{:.2f} s".format(*value) if value[1] > value[0] else "—"

    @staticmethod
    def _factor_text(run: BatchEstimationRun | None, bag_id: str) -> str:
        if run is None or bag_id not in run.bags:
            return "—"
        factors = run.bags[bag_id].observation_factors
        used = sorted(
            name for name, value in factors.items() if bool(value["enabled"])
        )
        disabled = sorted(
            name for name, value in factors.items() if not bool(value["enabled"])
        )
        return "used {}{}".format(
            ", ".join(used) if used else "none",
            " | disabled " + ", ".join(disabled) if disabled else "",
        )

    def refresh(self) -> None:
        self._refreshing = True
        try:
            run = self.store.estimation_run
            records = self.store.records()
            self.bag_table.setRowCount(len(records))
            for row, record in enumerate(records):
                use_item = QTableWidgetItem()
                use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
                use_item.setCheckState(
                    Qt.Checked if record.included else Qt.Unchecked
                )
                use_item.setData(Qt.UserRole, record.bag_id)
                self.bag_table.setItem(row, 0, use_item)
                result = record.result
                objective = (
                    None
                    if run is None
                    else run.static_map.bag_objective.get(record.bag_id)
                )
                values = (
                    record.display_name,
                    record.configuration_group,
                    self._range_text(record.auto_range),
                    self._range_text(record.selected_range),
                    record.interval_state,
                    "—" if result is None else str(result.sample_count),
                    record.status,
                    "—" if objective is None else "{:.5g}".format(objective),
                    self._factor_text(run, record.bag_id),
                )
                for column, text in enumerate(values, start=1):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, record.bag_id)
                    self.bag_table.setItem(row, column, item)
            self.bag_table.resizeColumnsToContents()
            self._refresh_contribution_plot()
            self._refresh_configuration_warning()
        finally:
            self._refreshing = False

    def set_run(self, run: BatchEstimationRun | None) -> None:
        self.posterior_widget.set_run(run)
        self.mcmc_trace_widget.set_run(run)
        self.objective_plot.clear()
        self.em_plot.clear()
        self.em_objective_plot.clear()
        if run is None:
            self.diagnostic_label.setText("run status: not run")
            self.sample_detail.setText("No completed sparse batch run")
            self.refresh()
            return

        q_history = run.q_em
        for component, label in enumerate(Q_COMPONENT_LABELS):
            self.em_plot.plot(
                q_history.iteration,
                q_history.accepted_q[:, component],
                pen=pg.intColor(component, hues=6),
                symbol="o",
                symbolSize=4,
                name="Q {}".format(label),
            )
        self.em_objective_plot.plot(
            q_history.iteration,
            q_history.map_objective,
            pen=pg.mkPen((45, 120, 185), width=2.0),
            symbol="o",
            symbolSize=4,
            name="MAP objective",
        )
        self.em_objective_plot.plot(
            q_history.iteration,
            q_history.approximate_marginal_objective,
            pen=pg.mkPen((30, 30, 30), width=2.0, style=Qt.DashLine),
            name="approx. marginal objective",
        )

        substages = run.manifest["substage_status"]
        stage_text = ", ".join(
            "{}={} ({})".format(
                name,
                "converged" if value["converged"] else "not converged",
                value["termination_reason"],
            )
            for name, value in substages.items()
        )
        laplace = run.laplace
        mcmc_text = "disabled; MAP/Laplace result only"
        if run.mcmc is not None:
            diagnostic = run.diagnostics.mcmc
            assert diagnostic is not None
            finite_rhat = diagnostic.split_rhat[
                np.isfinite(diagnostic.split_rhat)
            ]
            mcmc_text = (
                "{} retained equal-weight samples; completed={}; "
                "converged={}; max R-hat={}; min ESS={:.4g}; "
                "inner solve failures={}"
            ).format(
                run.mcmc.size,
                diagnostic.completed,
                diagnostic.converged,
                "—" if not finite_rhat.size else "{:.4g}".format(
                    float(np.max(finite_rhat))
                ),
                float(np.min(diagnostic.effective_sample_size)),
                int(np.sum(diagnostic.kernel_inner_solve_failures)),
            )
        warning_text = "none" if not run.warnings else "; ".join(run.warnings)
        self.diagnostic_label.setText(
            "run ID: {}\nsubstage status: {}\n"
            "Q definition: {} [{}]\n"
            "Q diagonal: {}\nEM iterations: {}\n"
            "Laplace rank: {}/18; condition number: {:.4g}; "
            "ridge alignment: {:.4g}; delay σ: {:.4g} s\n"
            "MCMC: {}\nwarnings: {}".format(
                run.run_id,
                stage_text,
                run.manifest["q_definition"]["definition"],
                ", ".join(run.manifest["q_definition"]["units"]),
                np.array2string(run.static_map.q_diagonal, precision=5),
                q_history.iteration.size,
                laplace.effective_rank,
                laplace.condition_number,
                laplace.ridge_alignment,
                laplace.delay_local_uncertainty,
                mcmc_text,
                warning_text,
            )
        )
        self._on_selected_sample_changed(self.store.selected_sample_id)
        self.refresh()

    def _refresh_contribution_plot(self) -> None:
        self.objective_plot.clear()
        run = self.store.estimation_run
        records = self.store.records()
        if run is None or not records:
            return
        values = np.asarray(
            [
                run.static_map.bag_objective.get(record.bag_id, np.nan)
                for record in records
            ],
            dtype=float,
        )
        finite = np.isfinite(values)
        x_values = np.arange(len(records), dtype=float)
        if np.any(finite):
            self.objective_plot.addItem(
                pg.BarGraphItem(
                    x=x_values[finite],
                    height=values[finite],
                    width=0.62,
                    brush=pg.mkBrush(75, 130, 190, 150),
                )
            )
        self.objective_plot.getAxis("bottom").setTicks(
            [[(float(index), record.display_name) for index, record in enumerate(records)]]
        )

    def _refresh_configuration_warning(self) -> None:
        groups = {
            record.configuration_group for record in self.store.included_records()
        }
        if len(groups) <= 1:
            self.configuration_warning.setText(
                "Shared-parameter configuration fingerprint: {}".format(
                    next(iter(groups), "none")
                )
            )
            self.configuration_warning.setStyleSheet("")
            return
        self.configuration_warning.setText(
            "Selected bags have different configuration fingerprints; "
            "one sparse batch run requires a confirmed shared configuration."
        )
        self.configuration_warning.setStyleSheet(
            "background: #fff1c7; color: #6c4a00; padding: 6px; "
            "border-radius: 4px;"
        )

    def _on_selected_sample_changed(self, sample_id: str | None) -> None:
        self.posterior_widget.set_selected_sample(sample_id)
        run = self.store.estimation_run
        if run is None:
            self.sample_detail.setText("No completed sparse batch run")
        elif sample_id is None or run.mcmc is None:
            self.sample_detail.setText(map_parameter_text(run.static_map))
        else:
            self.sample_detail.setText(
                sample_parameter_text(run.mcmc, sample_id)
            )

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._refreshing and item.column() == 0:
            self.store.set_included(
                str(item.data(Qt.UserRole)),
                item.checkState() == Qt.Checked,
            )

    def _on_bag_double_clicked(self, row: int, _column: int) -> None:
        item = self.bag_table.item(row, 1)
        if item is not None:
            self.bagActivated.emit(str(item.data(Qt.UserRole)))


__all__ = ["MasterView", "McmcTraceWidget", "PosteriorPlotWidget"]
