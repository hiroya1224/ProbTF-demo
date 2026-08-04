"""Linked trajectory plots for strict batch-estimation display models."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ..artifact_loader import (
    BagEstimationResult,
    ConditionalTrajectory,
    FlightResult,
)


def dynamics_residual_time_axis(
    knot_time: np.ndarray, dynamics_residual: np.ndarray
) -> np.ndarray:
    """Align a dynamics factor value with its interval midpoint."""

    time = np.asarray(knot_time, dtype=float)
    residual = np.asarray(dynamics_residual, dtype=float)
    if time.ndim != 1 or residual.ndim != 2 or residual.shape[1] != 6:
        raise ValueError("dynamics residual must have interval and six component axes")
    if residual.shape[0] != max(time.size - 1, 0):
        raise ValueError("dynamics residual count must equal knot count minus one")
    return 0.5 * (time[:-1] + time[1:])


class SignalPanel(QWidget):
    """Compact plot panel driven by MAP and one selected conditional sample."""

    currentTimeRequested = Signal(float)
    viewRangeRequested = Signal(float, float)
    estimationRangeEdited = Signal(float, float)

    def __init__(self, kind: str, _posterior: object = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if kind == "residual":
            kind = "dynamics"
        if kind not in {"trajectory", "correction", "dynamics"}:
            raise ValueError("unsupported panel kind {!r}".format(kind))
        self.kind = kind
        self.session: FlightResult | BagEstimationResult | None = None
        self.selected_trajectory: ConditionalTrajectory | None = None
        self.selected_sample_id: str | None = None
        self.current_time = 0.0
        self._view_update = False
        self._estimation_range = (0.0, 1.0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        tabs_header = QHBoxLayout()
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self.reset_current_view)
        tabs_header.addStretch(1)
        tabs_header.addWidget(self.reset_view_button)
        layout.addLayout(tabs_header)
        self.component_tabs = QTabWidget()
        layout.addWidget(self.component_tabs, 1)
        self.plots: list[pg.PlotWidget] = []
        labels = (
            ("x", "y", "z", "roll", "pitch", "yaw")
            if kind != "dynamics"
            else ("force x", "force y", "force z", "torque x", "torque y", "torque z")
        )
        for label in labels:
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setLabel("bottom", "time", units="s")
            plot.setLabel("left", label)
            self.component_tabs.addTab(plot, label)
            self.plots.append(plot)
        for plot in self.plots[1:]:
            plot.setXLink(self.plots[0])
        self.plots[0].getViewBox().sigXRangeChanged.connect(self._x_range_changed)

    def set_session(self, session: FlightResult | BagEstimationResult | None) -> None:
        self.session = session
        self.render()

    def set_selected_trajectory(
        self, trajectory: ConditionalTrajectory | None
    ) -> None:
        self.selected_trajectory = trajectory
        self.selected_sample_id = None if trajectory is None else trajectory.sample_id
        self.render()

    def set_selected_sample(self, sample_id: str | None) -> None:
        self.selected_sample_id = None if sample_id is None else str(sample_id)
        if self.selected_trajectory is not None and self.selected_trajectory.sample_id != self.selected_sample_id:
            self.selected_trajectory = None
        self.render()

    def set_current_time(self, value: float) -> None:
        self.current_time = float(value)

    def set_view_range(self, start: float, end: float) -> None:
        self._view_update = True
        for plot in self.plots:
            plot.setXRange(float(start), float(end), padding=0.0)
        self._view_update = False

    def set_estimation_range(self, start: float, end: float, *, movable: bool = True) -> None:
        del movable
        self._estimation_range = tuple(sorted((float(start), float(end))))

    def set_auto_estimation_range(self, start: float, end: float) -> None:
        del start, end

    def reset_current_view(self) -> None:
        for plot in self.plots:
            plot.enableAutoRange()

    def _x_range_changed(self, _view: object, value: object) -> None:
        if self._view_update:
            return
        start, end = np.asarray(value, dtype=float)
        self.viewRangeRequested.emit(float(start), float(end))

    def _plot_vector(self, time: np.ndarray, values: np.ndarray, color: str, width: float, style: Qt.PenStyle = Qt.SolidLine) -> None:
        selected = np.asarray(values, dtype=float)
        if selected.ndim != 2 or selected.shape[1] != len(self.plots):
            return
        pen = pg.mkPen(color, width=width, style=style)
        for component, plot in enumerate(self.plots):
            plot.plot(time, selected[:, component], pen=pen)

    def render(self) -> None:
        for plot in self.plots:
            plot.clear()
        session = self.session
        if session is None:
            self.status_label.setText("No trajectory loaded.")
            return
        if isinstance(session, FlightResult):
            if self.kind == "trajectory":
                observed = np.column_stack((session.observed_position, session.observed_rpy))
                reference = np.column_stack((session.reference_position, session.reference_rpy))
                self._plot_vector(session.time, reference, "#666666", 1.5, Qt.DashLine)
                self._plot_vector(session.time, observed, "#1e5abe", 2.5)
                self.status_label.setText("Inspection preview: reference and observed pose.")
            else:
                self.status_label.setText("Run estimation to display {}.".format(self.kind))
            return
        if self.kind == "trajectory":
            self._plot_vector(session.reference.time, np.column_stack((session.reference.position, session.reference.rpy)), "#666666", 1.5, Qt.DashLine)
            self._plot_vector(session.pose.time[session.pose.valid], np.column_stack((session.pose.position[session.pose.valid], session.pose.rpy[session.pose.valid])), "#1e5abe", 2.5)
            self._plot_vector(session.knot_time, np.column_stack((session.map_trajectory.position, session.map_trajectory.rpy)), "#1e965f", 2.5)
            if self.selected_trajectory is not None:
                state = self.selected_trajectory.state
                self._plot_vector(self.selected_trajectory.knot_time, np.column_stack((state.position, state.rpy)), "#962daa", 2.5)
        elif self.kind == "correction":
            self._plot_vector(session.knot_time, np.column_stack((session.correction_translation, session.correction_rotation_vector)), "#1e965f", 2.5)
            if self.selected_trajectory is not None:
                self._plot_vector(self.selected_trajectory.knot_time, np.column_stack((self.selected_trajectory.correction_translation, self.selected_trajectory.correction_rotation_vector)), "#962daa", 2.5)
        else:
            time = dynamics_residual_time_axis(session.knot_time, session.map_dynamics_residual)
            self._plot_vector(time, session.map_dynamics_residual, "#1e965f", 2.5)
            if self.selected_trajectory is not None:
                selected_time = dynamics_residual_time_axis(self.selected_trajectory.knot_time, self.selected_trajectory.dynamics_residual)
                self._plot_vector(selected_time, self.selected_trajectory.dynamics_residual, "#962daa", 2.5)
        self.status_label.setText("MAP{} {}.".format(" and selected sample" if self.selected_trajectory is not None else "", self.kind))


__all__ = ["SignalPanel", "dynamics_residual_time_axis"]
