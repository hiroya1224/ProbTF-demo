from __future__ import annotations

import json
from pathlib import Path

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

from ..state import BagRecord, ProjectStore, TimeState
from .scene_3d import Scene3DWidget
from .timeline import SignalPanel


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
        self._loaded_session: object | None = None
        self._refreshing_table = False
        self._loading_record = False

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(33)
        self.playback_timer.timeout.connect(self._advance_playback)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(main_splitter)

        left_panel = self._build_left_panel()
        main_splitter.addWidget(left_panel)

        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([380, 1100])

        scene_splitter = QSplitter(Qt.Horizontal)
        self.scene = Scene3DWidget(store.parameter_ensemble)
        scene_splitter.addWidget(self.scene)
        self.scene_controls_scroll = QScrollArea()
        self.scene_controls_scroll.setWidgetResizable(True)
        self.scene_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene_controls_scroll.setFrameShape(QScrollArea.NoFrame)
        self.scene_controls_scroll.setMinimumWidth(220)
        self.scene_controls_scroll.setMinimumHeight(120)
        self.scene_controls_scroll.setWidget(self._build_scene_controls())
        scene_splitter.addWidget(self.scene_controls_scroll)
        scene_splitter.setSizes([900, 240])
        right_splitter.addWidget(scene_splitter)

        self.signal_tabs = QTabWidget()
        self.trajectory_panel = SignalPanel("trajectory", store.parameter_ensemble)
        self.correction_panel = SignalPanel("correction", store.parameter_ensemble)
        self.residual_panel = SignalPanel("residual", store.parameter_ensemble)
        self.signal_tabs.addTab(self.trajectory_panel, "Trajectory")
        self.signal_tabs.addTab(self.correction_panel, "Correction transform")
        self.signal_tabs.addTab(self.residual_panel, "Residual wrench")
        self.flight_state_plot = pg.PlotWidget(title="Recorded flight state")
        self.flight_state_plot.showGrid(x=True, y=True, alpha=0.18)
        self.flight_state_plot.setLabel("bottom", "time", units="s")
        self.flight_state_plot.setLabel("left", "flight state")
        self.flight_state_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((35, 35, 35), width=1.2)
        )
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
        interval_buttons = QHBoxLayout()
        self.auto_detect_button = QPushButton("Re-inspect")
        self.restore_auto_button = QPushButton("Restore auto")
        self.lock_button = QPushButton("Lock")
        self.lock_button.setCheckable(True)
        interval_buttons.addWidget(self.auto_detect_button)
        interval_buttons.addWidget(self.restore_auto_button)
        interval_buttons.addWidget(self.lock_button)
        interval_layout.addLayout(interval_buttons)

        interval_form = QFormLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" s")
        interval_form.addRow("start", self.start_spin)
        interval_form.addRow("end", self.end_spin)
        interval_layout.addLayout(interval_form)
        self.interval_state_label = QLabel("No bag selected")
        interval_layout.addWidget(self.interval_state_label)
        layout.addWidget(interval_group)

        details_group = QGroupBox("Current bag")
        details_layout = QFormLayout(details_group)
        self.path_label = QLabel("—")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.include_checkbox = QCheckBox("Use this bag in joint smoothing")
        self.confirm_group_button = QPushButton("Confirm configuration group…")
        self.configure_button = QPushButton("Set configuration provenance…")
        self.group_label = QLabel("—")
        self.samples_label = QLabel("—")
        self.sha_label = QLabel("—")
        self.sha_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.inspection_details = QLabel("Inspection pending")
        self.inspection_details.setWordWrap(True)
        self.inspection_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.inspection_details.setToolTip(
            "Residual-wrench time resolution reports whether the calibrated OU "
            "process has enough knots over this bag's selected interval."
        )
        details_layout.addRow("path", self.path_label)
        details_layout.addRow("group", self.group_label)
        details_layout.addRow("samples", self.samples_label)
        details_layout.addRow("SHA256", self.sha_label)
        details_layout.addRow(self.include_checkbox)
        details_layout.addRow(self.confirm_group_button)
        details_layout.addRow(self.configure_button)
        details_layout.addRow("inspection", self.inspection_details)
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
            lambda: (
                self.configurationGroupRequested.emit(self.current_record.bag_id)
                if self.current_record is not None else None
            )
        )
        self.configure_button.clicked.connect(
            lambda: (
                self.configurationRequested.emit(self.current_record.bag_id)
                if self.current_record is not None else None
            )
        )

        return panel

    def _build_scene_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(200)
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
            ("observed", "Observed"),
            ("nominal", "Controller nominal"),
            ("posterior", "Posterior members"),
            ("selected", "Selected member"),
        ):
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, layer_name=layer: self.scene.set_layer_visible(layer_name, checked)
            )
            layer_layout.addWidget(checkbox)
            self.layer_checkboxes[layer] = checkbox
        layout.addWidget(layer_group)

        selection_group = QGroupBox("Shared selection")
        selection_layout = QFormLayout(selection_group)
        self.current_time_spin = QDoubleSpinBox()
        self.current_time_spin.setDecimals(3)
        self.current_time_spin.setSuffix(" s")
        self.member_label = QLabel(str(self.store.selected_member_id))
        self.member_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        selection_layout.addRow("current time", self.current_time_spin)
        selection_layout.addRow("member ID", self.member_label)
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
        self.store.selectedMemberChanged.connect(self._on_selected_member_changed)
        self.store.posteriorChanged.connect(self._on_posterior_changed)

        self.time_state.currentTimeChanged.connect(self.scene.set_current_time)
        self.time_state.currentTimeChanged.connect(self.trajectory_panel.set_current_time)
        self.time_state.currentTimeChanged.connect(self.correction_panel.set_current_time)
        self.time_state.currentTimeChanged.connect(self.residual_panel.set_current_time)
        self.time_state.currentTimeChanged.connect(self._store_current_time)
        self.time_state.currentTimeChanged.connect(self._update_current_time_spin)
        self.time_state.currentTimeChanged.connect(self.flight_state_line.setValue)

        self.time_state.viewRangeChanged.connect(self.trajectory_panel.set_view_range)
        self.time_state.viewRangeChanged.connect(self.correction_panel.set_view_range)
        self.time_state.viewRangeChanged.connect(self.residual_panel.set_view_range)
        self.time_state.viewRangeChanged.connect(self._store_view_range)

        self.time_state.estimationRangeChanged.connect(
            self.trajectory_panel.set_estimation_range
        )

        for panel in (self.trajectory_panel, self.correction_panel, self.residual_panel):
            panel.currentTimeRequested.connect(self.time_state.set_current_time)
            panel.viewRangeRequested.connect(self.time_state.set_view_range)

        self.trajectory_panel.estimationRangeEdited.connect(
            self._on_estimation_region_edited
        )

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
        file_names, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Add rosbag files",
            str(Path.home()),
            "ROS bags (*.bag *.db3);;All files (*)",
        )
        if not file_names:
            return
        self.filesSelected.emit(tuple(Path(file_name) for file_name in file_names))

    def remove_current(self) -> None:
        if self.current_record is None:
            return
        self.store.remove(self.current_record.bag_id)

    def auto_detect_current_interval(self) -> None:
        if self.current_record is None:
            return
        self.reinspectionRequested.emit((self.current_record.bag_id,))

    def restore_auto_interval(self) -> None:
        if self.current_record is None:
            return
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
        if self.current_record is None:
            return
        if self.current_record.data is None:
            return
        time = self.current_record.data.time
        index = int(np.argmin(np.abs(time - self.time_state.current_time)))
        index = int(np.clip(index + int(sample_delta), 0, time.size - 1))
        self.time_state.set_current_time(float(time[index]))

    def fit_view(self) -> None:
        if self.current_record is None:
            return
        if self.current_record.data is None:
            return
        time = self.current_record.data.time
        self.time_state.set_view_range(float(time[0]), float(time[-1]))

    def clear_inspection_selection(self) -> None:
        if self.current_record is None:
            return
        self.time_state.set_current_time(float(self.current_record.selected_range[0]))

    def close_scene(self) -> None:
        self.scene.close_scene()

    def _load_record(self, record: BagRecord | None) -> None:
        self._loading_record = True
        try:
            self.current_record = record
            session = record.data if record is not None else None
            self._loaded_session = session
            self.scene.set_session(session)
            self.trajectory_panel.set_session(session)
            self.correction_panel.set_session(session)
            self.residual_panel.set_session(session)
            self._render_flight_state(record)

            if record is None:
                return

            if record.data is None:
                self._update_details(record)
                self._select_table_row(record.bag_id)
                return
            time = record.data.time
            minimum = float(time[0])
            maximum = float(time[-1])
            for spin in (self.start_spin, self.end_spin, self.current_time_spin):
                spin.setRange(minimum, maximum)

            self.time_state.current_time = record.current_time
            self.time_state.estimation_start, self.time_state.estimation_end = record.selected_range
            self.time_state.view_start, self.time_state.view_end = record.view_range
            self.time_state.currentTimeChanged.emit(record.current_time)
            self.time_state.estimationRangeChanged.emit(*record.selected_range)
            self.time_state.viewRangeChanged.emit(*record.view_range)
            self._update_interval_controls(record)
            self._update_details(record)
            self._select_table_row(record.bag_id)
        finally:
            self._loading_record = False

    def _update_interval_controls(self, record: BagRecord) -> None:
        blockers = [
            QSignalBlocker(self.start_spin),
            QSignalBlocker(self.end_spin),
            QSignalBlocker(self.lock_button),
        ]
        self.start_spin.setValue(record.selected_range[0])
        self.end_spin.setValue(record.selected_range[1])
        self.lock_button.setChecked(record.interval_state == "LOCKED")
        del blockers
        self.trajectory_panel.set_estimation_ranges(
            record.auto_range,
            record.selected_range,
        )
        self.trajectory_panel.set_estimation_movable(record.interval_state != "LOCKED")
        self.interval_state_label.setText(
            f"{record.interval_state} | auto {self._format_range(record.auto_range)} | "
            f"selected {self._format_range(record.selected_range)}"
        )

    def _update_details(self, record: BagRecord) -> None:
        self.path_label.setText(str(record.path))
        self.group_label.setText(record.configuration_group)
        self.samples_label.setText(
            "—" if record.data is None else
            f"{record.data.sample_count} samples / {record.data.duration:.1f} s"
        )
        self.sha_label.setText(record.sha256[:16] + "…")
        if record.inspection is None:
            self.inspection_details.setText("Inspection pending")
        else:
            warnings = record.inspection.get("warnings", [])
            topic_contract = record.inspection.get("topic_contract", {})
            q_text = (
                "not evaluated" if record.result is None
                else ("sufficient" if record.result.q_resolution_sufficient else "insufficient")
            )
            self.inspection_details.setText(
                "status: {}\nresidual-wrench Q time resolution: {}\n"
                "controller snapshot: {}\n"
                "topic contract: {}\nwarnings: {}".format(
                    record.status,
                    q_text,
                    json.dumps(record.controller_snapshot, ensure_ascii=False, sort_keys=True),
                    json.dumps(topic_contract, ensure_ascii=False, sort_keys=True),
                    "none" if not warnings else "; ".join(str(value) for value in warnings),
                )
            )
        blocker = QSignalBlocker(self.include_checkbox)
        self.include_checkbox.setChecked(record.included)
        self.include_checkbox.setEnabled(record.status in {"ready", "complete"})
        del blocker
        fingerprint = (
            None
            if record.inspection is None
            else record.inspection.get("configuration_fingerprint")
        )
        incomplete = (
            isinstance(fingerprint, dict)
            and not bool(fingerprint.get("complete", False))
        )
        self.confirm_group_button.setEnabled(
            incomplete
            and (
                record.status == "needs_configuration_confirmation"
                or bool(record.configuration_confirmation)
            )
        )
        self.confirm_group_button.setToolTip(
            "Assign an explicit manual group without claiming that the missing "
            "hardware provenance was recorded."
        )

    def _on_record_changed(self, bag_id: str) -> None:
        self.refresh_table()
        record = self.store.get(str(bag_id))
        if record is None or record.bag_id != self.store.current_bag_id:
            return
        if self._loaded_session is not record.data:
            self._load_record(record)
            return
        self._update_details(record)
        if record.data is not None:
            self._update_interval_controls(record)

    def _render_flight_state(self, record: BagRecord | None) -> None:
        self.flight_state_plot.clear()
        if (
            record is None
            or record.preview is None
            or record.preview.flight_state is None
        ):
            return
        self.flight_state_plot.plot(
            record.preview.time,
            record.preview.flight_state,
            pen=pg.mkPen((50, 95, 175), width=1.8),
        )
        self.flight_state_plot.addItem(self.flight_state_line)

    def _on_estimation_region_edited(self, start: float, end: float) -> None:
        if self.current_record is None or self._loading_record:
            return
        if self.current_record.interval_state == "LOCKED":
            self.trajectory_panel.set_estimation_range(
                *self.current_record.selected_range
            )
            return
        self.store.update_interval(
            self.current_record.bag_id,
            (start, end),
            state="MODIFIED",
        )
        self.time_state.set_estimation_range(start, end)
        self._update_interval_controls(self.current_record)

    def _on_numeric_interval_changed(self, _value: float) -> None:
        if self.current_record is None or self._loading_record:
            return
        if self.current_record.interval_state == "LOCKED":
            return
        start, end = sorted((self.start_spin.value(), self.end_spin.value()))
        self.store.update_interval(
            self.current_record.bag_id,
            (start, end),
            state="MODIFIED",
        )
        self.time_state.set_estimation_range(start, end)
        self._update_interval_controls(self.current_record)

    def _set_interval_locked(self, locked: bool) -> None:
        if self.current_record is None or self._loading_record:
            return
        state = "LOCKED" if locked else "MODIFIED"
        self.store.update_interval(
            self.current_record.bag_id,
            self.current_record.selected_range,
            state=state,
        )
        self.trajectory_panel.set_estimation_movable(not locked)
        self._update_interval_controls(self.current_record)

    def _on_include_checkbox_toggled(self, checked: bool) -> None:
        if self.current_record is None or self._loading_record:
            return
        self.store.set_included(self.current_record.bag_id, checked)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing_table or item.column() != 0:
            return
        bag_id = str(item.data(Qt.UserRole))
        self.store.set_included(bag_id, item.checkState() == Qt.Checked)

    def _on_table_selection_changed(self) -> None:
        if self._refreshing_table:
            return
        row = self.bag_table.currentRow()
        if row < 0:
            return
        item = self.bag_table.item(row, 1)
        if item is None:
            return
        self.store.set_current(str(item.data(Qt.UserRole)))

    def _on_selected_member_changed(self, member_id: int | None) -> None:
        self.member_label.setText("—" if member_id is None else str(member_id))
        self.scene.set_selected_member(member_id)
        self.trajectory_panel.set_selected_member(member_id)
        self.correction_panel.set_selected_member(member_id)
        self.residual_panel.set_selected_member(member_id)

    def _on_posterior_changed(self, _run: object) -> None:
        ensemble = self.store.parameter_ensemble
        self.scene.set_parameter_ensemble(ensemble)
        self.trajectory_panel.set_parameter_ensemble(ensemble)
        self.correction_panel.set_parameter_ensemble(ensemble)
        self.residual_panel.set_parameter_ensemble(ensemble)
        record = self.store.current_record()
        if record is not None:
            self._load_record(record)

    def _store_current_time(self, value: float) -> None:
        if self.current_record is not None and not self._loading_record:
            self.current_record.current_time = float(value)

    def _store_view_range(self, start: float, end: float) -> None:
        if self.current_record is not None and not self._loading_record:
            self.current_record.view_range = (float(start), float(end))

    def _update_current_time_spin(self, value: float) -> None:
        with QSignalBlocker(self.current_time_spin):
            self.current_time_spin.setValue(float(value))

    def _advance_playback(self) -> None:
        if self.current_record is None:
            self.stop_playback()
            return
        if self.current_record.data is None:
            self.stop_playback()
            return
        time = self.current_record.data.time
        current = self.time_state.current_time
        delta = 0.033 * self.time_state.playback_speed
        target = current + delta
        if target >= float(time[-1]):
            target = float(time[-1])
            self.stop_playback()
        self.time_state.set_current_time(target)

    def _select_table_row(self, bag_id: str) -> None:
        for row in range(self.bag_table.rowCount()):
            item = self.bag_table.item(row, 1)
            if item is not None and str(item.data(Qt.UserRole)) == bag_id:
                self.bag_table.selectRow(row)
                return

    @staticmethod
    def _format_range(value: tuple[float, float]) -> str:
        return f"{value[0]:.2f}–{value[1]:.2f} s"
