"""Posterior-driven PID proposal controls and strict artifact presentation."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import PidProposalEvaluation
from ..pid_request import (
    MODEL_DISCREPANCY_POLICIES,
    PidEvaluationLaunchOptions,
)
from ..state import ProjectStore
from .scene_3d import PidComparisonScene3DWidget


_GROUPS = ("xy", "z", "roll / pitch", "yaw")
_GAINS = ("P", "I", "D")
_METRIC_LABELS = {
    "position_rmse": "Position RMSE [m]",
    "orientation_rmse": "Orientation RMSE [rad]",
    "maximum_position_error": "Maximum position error [m]",
    "maximum_orientation_error": "Maximum orientation error [rad]",
    "numerical_failure_count": "Numerical failures",
    "actuator_saturation_duration": "Saturation duration [s]",
    "actuator_saturation_rate": "Saturation rate",
}


def _number(value: object) -> str:
    selected = float(value)
    return "—" if not np.isfinite(selected) else "{:.8g}".format(selected)


def _text_item(value: object) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _snapshot_vector(snapshot: object, name: str) -> np.ndarray | None:
    if not isinstance(snapshot, dict):
        return None
    candidates = (snapshot.get(name), snapshot.get("physical_parameters", {}).get(name) if isinstance(snapshot.get("physical_parameters"), dict) else None)
    for candidate in candidates:
        value = np.asarray(candidate if candidate is not None else (), dtype=float)
        if value.shape == (3,) and np.all(np.isfinite(value)) and np.all(value >= 0.0):
            return value
    return None


class NextExperimentView(QWidget):
    """Configure and inspect candidate × posterior-sample cross evaluation."""

    evaluationRequested = Signal(object)

    def __init__(self, store: ProjectStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.evaluation: PidProposalEvaluation | None = store.pid_evaluation
        self._evaluation_running = False
        self._candidate_index = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        intro = QLabel(
            "Derive exact PID candidates from retained MCMC samples, then cross-evaluate "
            "every candidate against the selected posterior plant population and bags."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        self.scenario_label = QLabel("No PID evaluation loaded.")
        self.scenario_label.setWordWrap(True)
        self.scenario_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.scenario_label.setStyleSheet("background:#eef4ff;color:#264a73;padding:7px;border-radius:4px;")
        root.addWidget(self.scenario_label)

        launch = QGroupBox("Posterior cross-evaluation")
        launch_layout = QVBoxLayout(launch)
        controls = QHBoxLayout()
        self.source_sample_label = QLabel("Selected MCMC sample: —")
        self.source_sample_label.setMinimumWidth(205)
        controls.addWidget(self.source_sample_label)
        controls.addWidget(QLabel("Current PID snapshot:"))
        self.baseline_combo = QComboBox()
        controls.addWidget(self.baseline_combo)
        controls.addWidget(QLabel("Model discrepancy:"))
        self.discrepancy_combo = QComboBox()
        for policy in MODEL_DISCREPANCY_POLICIES:
            label = "zero" if policy == "zero_model_discrepancy" else "sampled Q"
            self.discrepancy_combo.addItem(label, policy)
        controls.addWidget(self.discrepancy_combo)
        self.evaluate_button = QPushButton("Evaluate candidate population")
        self.evaluate_button.clicked.connect(self._request_evaluation)
        controls.addWidget(self.evaluate_button)
        launch_layout.addLayout(controls)

        advanced = QHBoxLayout()
        self.cvar_spin = QDoubleSpinBox()
        self.cvar_spin.setRange(0.0, 0.999)
        self.cvar_spin.setDecimals(3)
        self.cvar_spin.setValue(0.9)
        advanced.addWidget(QLabel("Upper CVaR level:"))
        advanced.addWidget(self.cvar_spin)
        self.quantile_spin = QDoubleSpinBox()
        self.quantile_spin.setRange(0.001, 0.999)
        self.quantile_spin.setDecimals(3)
        self.quantile_spin.setValue(0.95)
        advanced.addWidget(QLabel("Quantile level:"))
        advanced.addWidget(self.quantile_spin)
        self.replicates_spin = QSpinBox()
        self.replicates_spin.setRange(1, 10000)
        self.replicates_spin.setValue(1)
        advanced.addWidget(QLabel("Discrepancy replicates:"))
        advanced.addWidget(self.replicates_spin)
        self.derived_candidate_spin = QSpinBox()
        self.derived_candidate_spin.setRange(0, 10000)
        self.derived_candidate_spin.setValue(12)
        self.derived_candidate_spin.setSpecialValueText("all")
        advanced.addWidget(QLabel("Derived candidates:"))
        advanced.addWidget(self.derived_candidate_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        advanced.addWidget(QLabel("Base seed:"))
        advanced.addWidget(self.seed_spin)
        self.maximum_reference_age_spin = QDoubleSpinBox()
        self.maximum_reference_age_spin.setDecimals(4)
        self.maximum_reference_age_spin.setRange(0.0001, 60.0)
        self.maximum_reference_age_spin.setValue(0.5)
        advanced.addWidget(QLabel("Maximum reference age [s]:"))
        advanced.addWidget(self.maximum_reference_age_spin)
        self.selection_target_combo = QComboBox()
        self.selection_target_combo.addItem("No automatic selection", None)
        self.selection_target_combo.addItem("Selected sample candidate", "sample-derived")
        self.selection_target_combo.addItem("Exact user candidate", "user")
        advanced.addWidget(QLabel("YAML target:"))
        advanced.addWidget(self.selection_target_combo)
        advanced.addStretch(1)
        launch_layout.addLayout(advanced)
        root.addWidget(launch)

        explicit = QSplitter(Qt.Horizontal)
        drag_group = QGroupBox("Fixed plant assumptions (explicit)")
        drag_form = QFormLayout(drag_group)
        self.linear_drag_inputs = self._vector_editors(drag_form, "Linear drag x/y/z")
        self.angular_drag_inputs = self._vector_editors(drag_form, "Angular drag x/y/z")
        self.roll_pitch_integration_checkbox = QCheckBox(
            "Roll/pitch integration active at selected interval start (all bags)"
        )
        drag_form.addRow(self.roll_pitch_integration_checkbox)
        self.drag_status_label = QLabel("Verify these fixed values before evaluation.")
        self.drag_status_label.setWordWrap(True)
        drag_form.addRow(self.drag_status_label)
        explicit.addWidget(drag_group)
        self.user_candidate_group = QGroupBox("Include exact user PID candidate")
        self.user_candidate_group.setCheckable(True)
        self.user_candidate_group.setChecked(False)
        user_layout = QVBoxLayout(self.user_candidate_group)
        self.user_gain_table = QTableWidget(4, 3)
        self.user_gain_table.setHorizontalHeaderLabels(_GAINS)
        self.user_gain_table.setVerticalHeaderLabels(_GROUPS)
        self.user_gain_inputs: list[list[QDoubleSpinBox]] = []
        for group in range(4):
            row: list[QDoubleSpinBox] = []
            for gain in range(3):
                editor = QDoubleSpinBox()
                editor.setDecimals(10)
                editor.setRange(0.0, 1.0e6)
                self.user_gain_table.setCellWidget(group, gain, editor)
                row.append(editor)
            self.user_gain_inputs.append(row)
        self.user_gain_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        user_layout.addWidget(self.user_gain_table)
        explicit.addWidget(self.user_candidate_group)
        explicit.setSizes((380, 600))
        root.addWidget(explicit)

        self.detail_tabs = QTabWidget()
        root.addWidget(self.detail_tabs, 1)
        self.candidate_table = QTableWidget(0, 0)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        self.detail_tabs.addTab(self.candidate_table, "Candidate metrics")
        self.source_table = QTableWidget(0, 8)
        self.source_table.setHorizontalHeaderLabels(("Sample ID", "Mode", "Delay", "Mass", "CoG x", "CoG y", "CoG z", "Used as plant"))
        self.source_table.verticalHeader().setVisible(False)
        self.detail_tabs.addTab(self.source_table, "Plant samples")
        self.comparison_scene = PidComparisonScene3DWidget()
        self.detail_tabs.addTab(self.comparison_scene, "Selected sample trajectory")
        yaml_widget = QWidget()
        yaml_layout = QVBoxLayout(yaml_widget)
        self.yaml_safety_label = QLabel("This view never applies gains automatically; it only displays explicitly selected, recommended YAML.")
        self.yaml_safety_label.setWordWrap(True)
        yaml_layout.addWidget(self.yaml_safety_label)
        self.yaml_text = QPlainTextEdit()
        self.yaml_text.setReadOnly(True)
        yaml_layout.addWidget(self.yaml_text)
        self.diff_text = QPlainTextEdit()
        self.diff_text.setReadOnly(True)
        yaml_layout.addWidget(self.diff_text)
        self.detail_tabs.addTab(yaml_widget, "Proposed YAML / difference")
        self.recommendation_label = QLabel("Recommendation: —")
        self.recommendation_label.setWordWrap(True)
        root.addWidget(self.recommendation_label)

        store.pidEvaluationChanged.connect(self.set_evaluation)
        store.posteriorChanged.connect(self._source_run_changed)
        store.selectedSampleChanged.connect(self._selected_sample_changed)
        store.selectedModeChanged.connect(lambda _mode: self._update_evaluate_enabled())
        store.currentBagChanged.connect(lambda _record: self._refresh_scene())
        store.selectedPidProposalChanged.connect(self._select_candidate_id)
        self.baseline_combo.currentTextChanged.connect(self._baseline_snapshot_changed)
        self.user_candidate_group.toggled.connect(self._user_candidate_toggled)
        self.selection_target_combo.currentIndexChanged.connect(self._selection_target_changed)
        self.discrepancy_combo.currentIndexChanged.connect(lambda _index: self._refresh_scenario())
        self._source_run_changed(store.estimation_run)
        if self.evaluation is not None:
            self.set_evaluation(self.evaluation)

    @staticmethod
    def _vector_editors(form: QFormLayout, label: str) -> list[QDoubleSpinBox]:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        editors = []
        for _axis in range(3):
            editor = QDoubleSpinBox()
            editor.setDecimals(10)
            editor.setRange(0.0, 1.0e6)
            layout.addWidget(editor)
            editors.append(editor)
        form.addRow(label, widget)
        return editors

    def _source_run_changed(self, run: object) -> None:
        current = self.baseline_combo.currentText()
        self.baseline_combo.clear()
        if run is not None:
            for bag_id in run.manifest.get("selected_bag_ids", ()):  # type: ignore[union-attr]
                self.baseline_combo.addItem(str(bag_id))
        preferred = current
        record = self.store.current_record()
        if not preferred and record is not None:
            preferred = record.bag_id
        index = self.baseline_combo.findText(preferred)
        if index >= 0:
            self.baseline_combo.setCurrentIndex(index)
        self._baseline_snapshot_changed(self.baseline_combo.currentText())
        self._selected_sample_changed(self.store.selected_sample_id)

    def _baseline_snapshot_changed(self, bag_id: str) -> None:
        record = self.store.get(str(bag_id))
        if record is None:
            return
        found = []
        for name, editors in (("linear_drag", self.linear_drag_inputs), ("angular_drag", self.angular_drag_inputs)):
            values = _snapshot_vector(dict(record.controller_snapshot), name)
            found.append(values is not None)
            if values is not None:
                for editor, value in zip(editors, values):
                    editor.setValue(float(value))
        gains = np.asarray(record.controller_snapshot.get("gains", ()), dtype=float)
        if gains.shape == (4, 3) and np.all(np.isfinite(gains)) and np.all(gains >= 0.0):
            for row, values in zip(self.user_gain_inputs, gains):
                for editor, value in zip(row, values):
                    editor.setValue(float(value))
        self.drag_status_label.setText(
            "Loaded from controller snapshot." if all(found) else "Snapshot lacks one or more drag vectors; displayed values are explicit request inputs and must be verified."
        )

    def _selected_sample_changed(self, sample_id: object) -> None:
        self.source_sample_label.setText("Selected MCMC sample: —" if sample_id is None else "Selected MCMC sample: {}".format(sample_id))
        self._update_evaluate_enabled()
        self._refresh_scene()

    def _refresh_scene(self) -> None:
        record = self.store.current_record()
        self.comparison_scene.set_context(
            self.store.estimation_run,
            None if record is None else record.bag_id,
            self.store.selected_sample_id,
        )

    def _refresh_scenario(self) -> None:
        if self.evaluation is not None:
            return
        self.scenario_label.setText(
            "Future discrepancy: {} using the estimation run's declared Q quantity and common random numbers. Past dynamics-residual paths are never replayed.".format(
                self.discrepancy_combo.currentData()
            )
        )

    def _update_evaluate_enabled(self) -> None:
        self.evaluate_button.setEnabled(
            not self._evaluation_running
            and self.store.estimation_run is not None
            and self.store.posterior_samples is not None
            and self.store.selected_sample_id is not None
            and self.store.selected_mode_id is not None
            and self.baseline_combo.count() > 0
        )

    def set_evaluation_running(self, running: bool) -> None:
        self._evaluation_running = bool(running)
        self._update_evaluate_enabled()

    def _user_candidate_toggled(self, included: bool) -> None:
        if not included and self.selection_target_combo.currentData() == "user":
            self.selection_target_combo.setCurrentIndex(0)

    def _selection_target_changed(self, _index: int) -> None:
        if self.selection_target_combo.currentData() == "user" and not self.user_candidate_group.isChecked():
            self.user_candidate_group.setChecked(True)

    def _request_evaluation(self) -> None:
        sample_id = self.store.selected_sample_id
        mode_id = self.store.selected_mode_id
        baseline = self.baseline_combo.currentText()
        if sample_id is None or mode_id is None or not baseline:
            return
        run = self.store.estimation_run
        if run is None:
            return
        bag_inputs = []
        for bag_id in run.manifest.get("selected_bag_ids", ()):
            record = self.store.get(str(bag_id))
            if record is None:
                return
            bag_inputs.append(
                (
                    record.bag_id,
                    str(record.path),
                    record.sha256,
                    self.roll_pitch_integration_checkbox.isChecked(),
                )
            )
        user = None
        if self.user_candidate_group.isChecked():
            user = tuple(tuple(editor.value() for editor in row) for row in self.user_gain_inputs)
        self.evaluationRequested.emit(
            PidEvaluationLaunchOptions(
                source_sample_id=sample_id,
                baseline_bag_id=baseline,
                selected_mode_id=mode_id,
                bags=tuple(bag_inputs),
                fixed_linear_drag=tuple(editor.value() for editor in self.linear_drag_inputs),
                fixed_angular_drag=tuple(editor.value() for editor in self.angular_drag_inputs),
                model_discrepancy_policy=str(self.discrepancy_combo.currentData()),
                maximum_derived_candidates=(
                    None
                    if self.derived_candidate_spin.value() == 0
                    else self.derived_candidate_spin.value()
                ),
                quantile_level=self.quantile_spin.value(),
                cvar_level=self.cvar_spin.value(),
                base_seed=self.seed_spin.value(),
                replicates=self.replicates_spin.value(),
                maximum_reference_age_seconds=self.maximum_reference_age_spin.value(),
                user_candidate_values=user,
                selected_candidate_source=self.selection_target_combo.currentData(),
            )
        )

    def set_evaluation(self, evaluation: PidProposalEvaluation | None) -> None:
        self.evaluation = evaluation
        self.candidate_table.clear()
        self.candidate_table.setRowCount(0)
        self.source_table.setRowCount(0)
        self.yaml_text.clear()
        self.diff_text.clear()
        self.recommendation_label.setText("Recommendation: —")
        if evaluation is None:
            self._refresh_scenario()
            return
        manifest = evaluation.manifest
        self.scenario_label.setText(
            "{}; {} {}; {} replicate(s); {} plant sample(s) × {} bag(s).".format(
                manifest["model_discrepancy_policy"],
                manifest["model_discrepancy_residual_quantity"],
                "Q",
                manifest["model_discrepancy_replicates"],
                len(manifest["plant_sample_ids"]),
                len(manifest["bag_ids"]),
            )
        )
        self.yaml_text.setPlainText(evaluation.proposed_yaml or "No candidate was explicitly selected for YAML output.")
        self.diff_text.setPlainText(evaluation.proposed_diff_yaml or "No YAML difference is available.")
        self._populate_candidates()
        self._populate_sources()
        recommended = tuple(str(value) for value in manifest["recommended_candidate_ids"])
        self.recommendation_label.setText(
            "Recommendation: {}".format(", ".join(recommended))
            if recommended
            else "Recommendation unavailable: {}".format(manifest["rejection_reason"])
        )
        self._select_candidate_id(self.store.selected_pid_proposal_id)

    def _populate_candidates(self) -> None:
        assert self.evaluation is not None
        particles = self.evaluation.candidate_particles
        summary = self.evaluation.summary
        metrics = tuple(str(value) for value in np.asarray(summary["metric_names"]).tolist())
        headers = ["Candidate", "Source", "Source sample", "Generation", "XY P/I/D", "Z P/I/D", "Roll/pitch P/I/D", "Yaw P/I/D", "Completion mean", "Completion lower CVaR"]
        for metric in metrics:
            headers.extend(("{} mean".format(_METRIC_LABELS.get(metric, metric)), "{} upper CVaR".format(_METRIC_LABELS.get(metric, metric))))
        headers.extend(("Gain change", "Pareto", "Recommended"))
        self.candidate_table.setColumnCount(len(headers))
        self.candidate_table.setHorizontalHeaderLabels(headers)
        candidate_ids = np.asarray(summary["candidate_id"]).astype(str)
        sources = np.asarray(particles["source"]).astype(str)
        source_ids = np.asarray(particles["source_sample_id"]).astype(str)
        generation = np.asarray(particles["generation"])
        gains = np.asarray(particles["gain_values"], dtype=float)
        nondominated = set(str(value) for value in np.asarray(summary["nondominated_candidate_id"]).tolist())
        recommended = set(str(value) for value in np.asarray(summary["recommended_candidate_id"]).tolist())
        self.candidate_table.setRowCount(candidate_ids.size)
        for row, candidate_id in enumerate(candidate_ids):
            values: list[str] = [candidate_id, sources[row], source_ids[row] or "—", str(int(generation[row]))]
            values.extend(" / ".join(_number(value) for value in gains[row, group]) for group in range(4))
            values.extend((_number(summary["forecast_completion_mean"][row]), _number(summary["forecast_completion_lower_cvar"][row])))
            for column in range(len(metrics)):
                values.extend((_number(summary["mean"][row, column]), _number(summary["upper_cvar"][row, column])))
            values.extend((_number(summary["gain_change_magnitude"][row]), "yes" if candidate_id in nondominated else "no", "yes" if candidate_id in recommended else "no"))
            for column, value in enumerate(values):
                self.candidate_table.setItem(row, column, _text_item(value))
        self.candidate_table.resizeColumnsToContents()

    def _populate_sources(self) -> None:
        assert self.evaluation is not None
        arrays = self.evaluation.source_samples
        used = set(str(value) for value in self.evaluation.manifest["plant_sample_ids"])
        ids = np.asarray(arrays["sample_id"]).astype(str)
        self.source_table.setRowCount(ids.size)
        for row, sample_id in enumerate(ids):
            cog = arrays["cog"][row]
            values = (sample_id, arrays["source_mode_id"][row], _number(arrays["delay"][row]), _number(arrays["mass"][row]), _number(cog[0]), _number(cog[1]), _number(cog[2]), "yes" if sample_id in used else "no")
            for column, value in enumerate(values):
                self.source_table.setItem(row, column, _text_item(value))
        self.source_table.resizeColumnsToContents()

    def _candidate_selected(self) -> None:
        rows = sorted({index.row() for index in self.candidate_table.selectedIndexes()})
        if not rows or self.evaluation is None:
            return
        self._candidate_index = rows[0]
        candidate_id = str(np.asarray(self.evaluation.summary["candidate_id"])[self._candidate_index])
        self.store.set_selected_pid_proposal(candidate_id)

    def _select_candidate_id(self, candidate_id: object) -> None:
        if self.evaluation is None or candidate_id is None:
            return
        ids = np.asarray(self.evaluation.summary["candidate_id"]).astype(str)
        matches = np.flatnonzero(ids == str(candidate_id))
        if matches.size == 1:
            self._candidate_index = int(matches[0])
            self.candidate_table.selectRow(self._candidate_index)


__all__ = ["NextExperimentView"]
