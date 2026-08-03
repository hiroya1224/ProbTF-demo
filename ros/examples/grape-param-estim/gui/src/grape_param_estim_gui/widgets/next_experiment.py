from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..artifact_loader import PidProposalEvaluation
from ..pid_request import PidEvaluationLaunchOptions, RESIDUAL_POLICIES
from ..presentation import scenario_assumption_text
from ..state import ProjectStore
from .scene_3d import PidComparisonScene3DWidget


_GROUPS = ("xy", "z", "roll_pitch", "yaw")
_GAINS = ("P", "I", "D")
_METRICS = (
    ("position_rmse", "Position RMSE", "m"),
    ("orientation_rmse", "Orientation RMSE", "rad"),
    ("maximum_position_error", "Maximum position error", "m"),
    ("maximum_orientation_error", "Maximum orientation error", "rad"),
)


def _text_scalar(array: object, fallback: str = "—") -> str:
    if array is None:
        return fallback
    value = np.asarray(array).reshape(-1)
    return fallback if value.size == 0 else str(value[0])


def _number(value: float) -> str:
    return "—" if not np.isfinite(value) else "{:.12g}".format(float(value))


class NextExperimentView(QWidget):
    """Exact PID proposals and full closed-loop evaluation artifact viewer."""

    evaluationRequested = Signal(object)

    def __init__(self, store: ProjectStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.evaluation: PidProposalEvaluation | None = store.pid_evaluation
        self._candidate_index = 0
        self._evaluation_running = False
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        purpose_label = QLabel(
            "Updated PID gains are proposed to align the estimated physical-parameter "
            "posterior with the fixed controller nominal model."
        )
        purpose_label.setWordWrap(True)
        root.addWidget(purpose_label)

        self.scenario_label = QLabel()
        self.scenario_label.setWordWrap(True)
        self.scenario_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.scenario_label.setStyleSheet(
            "background: #eef4ff; color: #264a73; padding: 7px; border-radius: 4px;"
        )
        root.addWidget(self.scenario_label)

        launch_group = QGroupBox("Posterior-predictive evaluation")
        launch_layout = QVBoxLayout(launch_group)
        launch_controls = QHBoxLayout()
        self.source_member_label = QLabel("Selected member: —")
        self.source_member_label.setMinimumWidth(145)
        launch_controls.addWidget(self.source_member_label)
        launch_controls.addWidget(QLabel("Current PID snapshot:"))
        self.baseline_combo = QComboBox()
        self.baseline_combo.setMinimumWidth(125)
        launch_controls.addWidget(self.baseline_combo)
        launch_controls.addWidget(QLabel("Residual:"))
        self.residual_combo = QComboBox()
        for policy in RESIDUAL_POLICIES:
            self.residual_combo.addItem(policy, policy)
        launch_controls.addWidget(self.residual_combo)
        launch_controls.addWidget(QLabel("CVaR:"))
        self.cvar_spin = QDoubleSpinBox()
        self.cvar_spin.setDecimals(3)
        self.cvar_spin.setRange(0.0, 0.999)
        self.cvar_spin.setSingleStep(0.05)
        self.cvar_spin.setValue(0.9)
        self.cvar_spin.setToolTip(
            "Upper CVaR is the mean of the raw-member errors in the worst tail; "
            "this value sets where that tail begins."
        )
        launch_controls.addWidget(self.cvar_spin)
        launch_controls.addWidget(QLabel("Recommendation target:"))
        self.selection_target_combo = QComboBox()
        self.selection_target_combo.addItem("None", None)
        self.selection_target_combo.addItem(
            "Selected member candidate", "member-derived"
        )
        self.selection_target_combo.addItem("Exact user candidate", "user")
        self.selection_target_combo.setToolTip(
            "A target is evaluated for recommendation only when selected explicitly; "
            "no representative candidate is selected automatically."
        )
        launch_controls.addWidget(self.selection_target_combo)
        self.evaluate_button = QPushButton("Evaluate selected member")
        self.evaluate_button.clicked.connect(self._request_evaluation)
        launch_controls.addWidget(self.evaluate_button)
        launch_controls.addStretch(1)
        launch_layout.addLayout(launch_controls)
        self.threshold_label = QLabel("Thresholds: Not configured")
        self.threshold_label.setStyleSheet("color: #666;")
        self.threshold_label.setWordWrap(True)
        launch_layout.addWidget(self.threshold_label)
        root.addWidget(launch_group)

        self.user_candidate_group = QGroupBox("Include exact user PID candidate")
        self.user_candidate_group.setCheckable(True)
        self.user_candidate_group.setChecked(False)
        self.user_candidate_group.setToolTip(
            "Enter the exact xy, z, roll/pitch, and yaw P/I/D values."
        )
        user_layout = QVBoxLayout(self.user_candidate_group)
        self.user_gain_table = QTableWidget(4, 3)
        self.user_gain_table.setHorizontalHeaderLabels(("P", "I", "D"))
        self.user_gain_table.setVerticalHeaderLabels(
            ("xy", "z", "roll / pitch", "yaw")
        )
        self.user_gain_inputs: list[list[QDoubleSpinBox]] = []
        for group_index in range(4):
            row_inputs = []
            for gain_index in range(3):
                editor = QDoubleSpinBox()
                editor.setDecimals(12)
                editor.setRange(0.0, 1.0e6)
                editor.setSingleStep(0.1)
                editor.setKeyboardTracking(False)
                self.user_gain_table.setCellWidget(
                    group_index, gain_index, editor
                )
                row_inputs.append(editor)
            self.user_gain_inputs.append(row_inputs)
        self.user_gain_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.user_gain_table.setMaximumHeight(180)
        user_layout.addWidget(self.user_gain_table)
        root.addWidget(self.user_candidate_group)

        top_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(top_splitter, 2)
        candidate_group = QGroupBox("Evaluated candidates")
        candidate_layout = QVBoxLayout(candidate_group)
        self.candidate_table = QTableWidget(0, 20)
        self.candidate_table.setHorizontalHeaderLabels(
            [
                "Candidate", "Source", "XY P / I / D", "Z P / I / D",
                "Roll/pitch P / I / D", "Yaw P / I / D",
                "Forecast completion", "Numerical failures",
                "Position RMSE mean [m]", "Position RMSE CVaR [m]",
                "Orientation RMSE mean [rad]", "Orientation RMSE CVaR [rad]",
                "Maximum position error mean [m]", "Maximum position error CVaR [m]",
                "Maximum orientation error mean [rad]", "Maximum orientation error CVaR [rad]",
                "Log gain change", "Pareto", "Eligible", "Rejection reason",
            ]
        )
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        candidate_layout.addWidget(self.candidate_table)
        self.recommendation_label = QLabel("Recommendation: —")
        self.recommendation_label.setWordWrap(True)
        candidate_layout.addWidget(self.recommendation_label)
        top_splitter.addWidget(candidate_group)

        gain_group = QGroupBox("Exact PID configuration")
        gain_layout = QVBoxLayout(gain_group)
        self.gain_table = QTableWidget(12, 9)
        self.gain_table.setHorizontalHeaderLabels(
            [
                "Group", "Gain", "Current", "Proposed", "Difference", "Ratio",
                "50% range", "95% range", "Full-forecast validation",
            ]
        )
        self.gain_table.verticalHeader().setVisible(False)
        self.gain_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.gain_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        gain_layout.addWidget(self.gain_table)
        top_splitter.addWidget(gain_group)
        top_splitter.setSizes([650, 760])

        self.detail_tabs = QTabWidget()
        root.addWidget(self.detail_tabs, 3)
        self.metric_table = QTableWidget(0, 6)
        self.metric_table.setHorizontalHeaderLabels(
            ["Metric", "Mean", "Upper CVaR", "Threshold exceedance", "Unit", "CVaR level"]
        )
        self.metric_table.verticalHeader().setVisible(False)
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        self.detail_tabs.addTab(self.metric_table, "Aggregate metrics")

        self.bag_table = QTableWidget(0, 14)
        self.bag_table.setHorizontalHeaderLabels(
            [
                "Bag", "Forecast completion", "Numerical failures", "Position RMSE mean", "Position RMSE CVaR",
                "Orientation RMSE mean", "Orientation RMSE CVaR", "Max position mean",
                "Max position CVaR", "Max orientation mean", "Max orientation CVaR",
                "Position threshold exceedance rate",
                "Orientation threshold exceedance rate",
                "Correction zero coverage: translation / rotation / joint 6D",
            ]
        )
        self.bag_table.horizontalHeaderItem(13).setToolTip(
            "Fractions of raw-member correction paths whose zero correction is "
            "covered: translation components / rotation-vector components / all "
            "six components jointly."
        )
        self.bag_table.verticalHeader().setVisible(False)
        self.bag_table.horizontalHeader().setStretchLastSection(True)
        self.detail_tabs.addTab(self.bag_table, "Per-bag metrics")

        correction_widget = QWidget()
        correction_layout = QVBoxLayout(correction_widget)
        correction_layout.setContentsMargins(0, 0, 0, 0)
        self.correction_key = QLabel(
            "Components: x = red, y = green, z = blue. Dashed lines are the "
            "current PID selected-member path; solid lines are the selected "
            "candidate selected-member path. Shading is the 5–95% raw-member "
            "interval (lighter for current, darker for the selected candidate); "
            "the grey line is zero desired correction."
        )
        self.correction_key.setWordWrap(True)
        self.correction_key.setStyleSheet("color: #555; padding: 3px;")
        correction_layout.addWidget(self.correction_key)
        correction_plots = QHBoxLayout()
        self.translation_plot = pg.PlotWidget(title="Correction translation: current and selected candidate")
        self.rotation_plot = pg.PlotWidget(title="Correction rotation vector: current and selected candidate")
        for plot, unit in ((self.translation_plot, "m"), (self.rotation_plot, "rad")):
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setLabel("bottom", "time", units="s")
            plot.setLabel("left", "component", units=unit)
            correction_plots.addWidget(plot)
        correction_layout.addLayout(correction_plots, 1)
        self.detail_tabs.addTab(correction_widget, "Correction paths")

        self.comparison_scene = PidComparisonScene3DWidget()
        self.detail_tabs.addTab(self.comparison_scene, "3D comparison")

        self.member_detail = QPlainTextEdit()
        self.member_detail.setReadOnly(True)
        self.detail_tabs.addTab(self.member_detail, "Selected member")
        yaml_widget = QWidget()
        yaml_layout = QVBoxLayout(yaml_widget)
        yaml_layout.setContentsMargins(0, 0, 0, 0)
        self.yaml_safety_label = QLabel(
            "This view does not apply gains automatically; dynamic reconfigure "
            "only applies gains chosen elsewhere."
        )
        self.yaml_safety_label.setWordWrap(True)
        self.yaml_safety_label.setStyleSheet("color: #7a4d00; padding: 4px;")
        yaml_layout.addWidget(self.yaml_safety_label)
        self.yaml_text = QPlainTextEdit()
        self.yaml_text.setReadOnly(True)
        yaml_layout.addWidget(self.yaml_text, 1)
        self.detail_tabs.addTab(yaml_widget, "Proposed YAML")
        self.diff_text = QPlainTextEdit()
        self.diff_text.setReadOnly(True)
        self.detail_tabs.addTab(self.diff_text, "YAML difference")

        store.pidEvaluationChanged.connect(self.set_evaluation)
        store.posteriorChanged.connect(self._source_run_changed)
        store.selectedMemberChanged.connect(self._selected_member_changed)
        store.currentBagChanged.connect(lambda _record: self._refresh_member_and_paths())
        store.selectedPidProposalChanged.connect(self._select_candidate_id)
        self.baseline_combo.currentTextChanged.connect(
            self._baseline_snapshot_changed
        )
        self.residual_combo.currentTextChanged.connect(
            lambda _value: self._refresh_pending_scenario_assumption()
        )
        self.user_candidate_group.toggled.connect(
            self._user_candidate_toggled
        )
        self.selection_target_combo.currentIndexChanged.connect(
            self._selection_target_changed
        )
        self._source_run_changed(store.assimilation_run)
        self._refresh_pending_scenario_assumption()
        if self.evaluation is not None:
            self.set_evaluation(self.evaluation)

    def _source_run_changed(self, run: object) -> None:
        current_baseline = self.baseline_combo.currentText()
        self.baseline_combo.clear()
        if run is not None:
            for bag_id in run.manifest.get("selected_bag_ids", ()):  # type: ignore[union-attr]
                self.baseline_combo.addItem(str(bag_id))
        preferred = current_baseline
        current_record = self.store.current_record()
        if not preferred and current_record is not None:
            preferred = current_record.bag_id
        index = self.baseline_combo.findText(preferred)
        if index >= 0:
            self.baseline_combo.setCurrentIndex(index)
        self._baseline_snapshot_changed(self.baseline_combo.currentText())
        self._selected_member_changed(self.store.selected_member_id)

    def _baseline_snapshot_changed(self, bag_id: str) -> None:
        record = self.store.get(str(bag_id))
        if record is None:
            return
        gains = np.asarray(record.controller_snapshot.get("gains", ()), dtype=float)
        if (
            gains.shape != (4, 3)
            or np.any(~np.isfinite(gains))
            or np.any(gains < 0.0)
        ):
            return
        for group_index, row in enumerate(self.user_gain_inputs):
            for gain_index, editor in enumerate(row):
                editor.setValue(float(gains[group_index, gain_index]))

    def _user_candidate_toggled(self, included: bool) -> None:
        if not included and self.selection_target_combo.currentData() == "user":
            self.selection_target_combo.setCurrentIndex(0)

    def _selection_target_changed(self, _index: int) -> None:
        if (
            self.selection_target_combo.currentData() == "user"
            and not self.user_candidate_group.isChecked()
        ):
            self.user_candidate_group.setChecked(True)

    def _exact_user_candidate(self) -> tuple[tuple[float, ...], ...] | None:
        if not self.user_candidate_group.isChecked():
            return None
        return tuple(
            tuple(editor.value() for editor in row)
            for row in self.user_gain_inputs
        )

    def _refresh_pending_scenario_assumption(self) -> None:
        if self.evaluation is not None:
            return
        policy = str(self.residual_combo.currentData() or "unspecified")
        self.scenario_label.setText(
            "Counterfactual assumption: same recorded reference; same posterior "
            "member initial state; same estimated static plant member and constant "
            "delay; residual policy = {}; this is not a forecast of a new "
            "disturbance realization".format(policy)
        )

    def _selected_member_changed(self, member_id: object) -> None:
        self.source_member_label.setText(
            "Selected member: —"
            if member_id is None
            else "Selected member: {}".format(int(member_id))
        )
        self._update_evaluate_enabled()
        self._refresh_member_and_paths()

    def _update_evaluate_enabled(self) -> None:
        self.evaluate_button.setEnabled(
            not self._evaluation_running
            and self.store.assimilation_run is not None
            and self.store.selected_member_id is not None
            and self.baseline_combo.count() > 0
        )

    def set_evaluation_running(self, running: bool) -> None:
        self._evaluation_running = bool(running)
        self._update_evaluate_enabled()

    def _request_evaluation(self) -> None:
        member_id = self.store.selected_member_id
        baseline = self.baseline_combo.currentText()
        if member_id is None or not baseline:
            return
        options = PidEvaluationLaunchOptions(
            source_member_id=member_id,
            baseline_bag_id=baseline,
            residual_policy=str(self.residual_combo.currentData()),
            cvar_level=self.cvar_spin.value(),
            user_candidate_values=self._exact_user_candidate(),
            selected_candidate_source=self.selection_target_combo.currentData(),
        )
        self.evaluationRequested.emit(options)

    def set_evaluation(self, evaluation: PidProposalEvaluation | None) -> None:
        if evaluation is None:
            self.evaluation = None
            self._candidate_index = 0
            self.comparison_scene.set_evaluation(None)
            self._refresh_pending_scenario_assumption()
            self.candidate_table.setRowCount(0)
            self.gain_table.clearContents()
            self.metric_table.setRowCount(0)
            self.bag_table.setRowCount(0)
            self.recommendation_label.setText("Recommendation: —")
            self.threshold_label.setText("Thresholds: Not configured")
            self.yaml_text.clear()
            self.diff_text.clear()
            self.member_detail.clear()
            self.translation_plot.clear()
            self.rotation_plot.clear()
            return
        self.evaluation = evaluation
        self.comparison_scene.set_evaluation(evaluation)
        summary = evaluation.summary
        assumption = _text_scalar(summary.get("scenario_assumption"), "unspecified")
        self.scenario_label.setText(scenario_assumption_text(assumption))
        self._refresh_threshold_label()
        self.yaml_text.setPlainText(evaluation.proposed_yaml)
        self.diff_text.setPlainText(evaluation.proposed_diff_yaml)
        self._populate_candidates()
        self._select_candidate_id(self.store.selected_pid_proposal_id)

    def _populate_candidates(self) -> None:
        assert self.evaluation is not None
        summary = self.evaluation.summary
        candidate_ids = np.asarray(summary["candidate_id"]).astype(str)
        sources = np.asarray(summary["candidate_source"]).astype(str)
        completion = np.asarray(summary.get("forecast_completion", np.full(candidate_ids.size, np.nan)))
        failures = np.asarray(summary.get("numerical_failure_count", np.full(candidate_ids.size, -1)))
        exact_pid = np.asarray(summary["proposed_pid"], dtype=float)
        change = np.asarray(summary.get("log_gain_change", np.full(candidate_ids.size, np.nan)))
        dominated = np.asarray(summary.get("pareto_dominated", np.zeros(candidate_ids.size, dtype=bool)))
        improves = np.asarray(summary.get("improves_current", np.zeros(candidate_ids.size, dtype=bool)))
        eligible = np.asarray(summary.get("candidate_eligible", improves))
        rejection = summary.get("candidate_rejection_reason")
        self.candidate_table.setRowCount(candidate_ids.size)
        for row, candidate_id in enumerate(candidate_ids):
            reason = ""
            if rejection is not None:
                reason = str(np.asarray(rejection).reshape(-1)[row])
            metrics = []
            for key, _label, _unit in _METRICS:
                mean = np.asarray(
                    summary.get(
                        "aggregate_{}_mean".format(key),
                        np.full(candidate_ids.size, np.nan),
                    )
                )[row]
                cvar = np.asarray(
                    summary.get(
                        "aggregate_{}_upper_cvar".format(key),
                        np.full(candidate_ids.size, np.nan),
                    )
                )[row]
                metrics.extend((_number(float(mean)), _number(float(cvar))))
            values = (
                candidate_id,
                sources[row],
                *(
                    " / ".join(_number(value) for value in exact_pid[row, group])
                    for group in range(4)
                ),
                _number(float(completion[row])),
                str(int(failures[row])) if int(failures[row]) >= 0 else "—",
                *metrics, _number(float(change[row])),
                "dominated" if bool(dominated[row]) else "non-dominated",
                "yes" if bool(eligible[row]) else "no", reason,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, row)
                self.candidate_table.setItem(row, column, item)
        self.candidate_table.resizeColumnsToContents()
        available = bool(np.asarray(summary["recommendation_available"]).reshape(-1)[0])
        recommended = _text_scalar(summary.get("recommended_candidate_id"), "")
        rejection_reason = _text_scalar(summary.get("rejection_reason"), "")
        self.recommendation_label.setText(
            "Recommendation: {}".format(recommended)
            if available and recommended
            else "Recommendation: none{}".format(
                " — " + rejection_reason if rejection_reason else ""
            )
        )
        if candidate_ids.size:
            self.candidate_table.selectRow(0)

    def _refresh_threshold_label(self) -> None:
        assert self.evaluation is not None
        summary = self.evaluation.summary
        labels = {
            "position_rmse": "position RMSE",
            "maximum_position_error": "maximum position error",
            "orientation_rmse": "orientation RMSE",
            "maximum_orientation_error": "maximum orientation error",
        }
        parts = []
        for prefix, unit in (("position", "m"), ("orientation", "rad")):
            threshold = float(
                np.asarray(summary.get(prefix + "_threshold", [np.nan]))
                .reshape(-1)[0]
            )
            configured = bool(
                np.asarray(
                    summary.get(
                        prefix + "_threshold_configured",
                        [np.isfinite(threshold)],
                    )
                ).reshape(-1)[0]
            )
            if not configured:
                parts.append("{} Not configured".format(prefix))
                continue
            metric = _text_scalar(
                summary.get(prefix + "_threshold_metric"),
                prefix + "_rmse",
            )
            parts.append(
                "{} limit = {} {}".format(
                    labels.get(metric, metric), _number(threshold), unit
                )
            )
        self.threshold_label.setText("Thresholds: " + "; ".join(parts))

    def _candidate_selected(self) -> None:
        if self.evaluation is None:
            return
        row = self.candidate_table.currentRow()
        if row < 0:
            return
        self._candidate_index = row
        candidate_id = str(np.asarray(self.evaluation.summary["candidate_id"])[row])
        if candidate_id != self.store.selected_pid_proposal_id:
            self.store.set_selected_pid_proposal(candidate_id)
        self._refresh_gain_table()
        self._refresh_metric_table()
        self._refresh_bag_table()
        self._refresh_member_and_paths()

    def _select_candidate_id(self, candidate_id: str | None) -> None:
        if self.evaluation is None or candidate_id is None:
            return
        ids = np.asarray(self.evaluation.summary["candidate_id"]).astype(str)
        matches = np.flatnonzero(ids == candidate_id)
        if matches.size:
            self.candidate_table.selectRow(int(matches[0]))

    def _proposal_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.evaluation is not None
        proposals = np.asarray(self.evaluation.proposal_ensemble["proposed_pid"], dtype=float)
        return (
            np.quantile(proposals, (0.25, 0.75), axis=0),
            np.quantile(proposals, (0.025, 0.975), axis=0),
        )

    def _refresh_gain_table(self) -> None:
        assert self.evaluation is not None
        summary = self.evaluation.summary
        current = np.asarray(summary["current_pid"], dtype=float)
        proposed = np.asarray(summary["proposed_pid"], dtype=float)[self._candidate_index]
        raw_difference = summary.get("difference")
        difference = (
            proposed - current
            if raw_difference is None
            else np.asarray(raw_difference, dtype=float)[self._candidate_index]
        )
        raw_ratio = summary.get("ratio")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (
                proposed / current
                if raw_ratio is None
                else np.asarray(raw_ratio, dtype=float)[self._candidate_index]
            )
        range_50, range_95 = self._proposal_ranges()
        completion = float(
            np.asarray(summary.get("forecast_completion", [np.nan]))[
                self._candidate_index
            ]
        )
        failures = int(
            np.asarray(summary.get("numerical_failure_count", [-1]))[
                self._candidate_index
            ]
        )
        dominated = bool(
            np.asarray(summary.get("pareto_dominated", [False]))[
                self._candidate_index
            ]
        )
        eligible = bool(
            np.asarray(
                summary.get(
                    "candidate_eligible",
                    summary.get("improves_current", [False]),
                )
            )[
                self._candidate_index
            ]
        )
        validation = (
            "completion {}; numerical failures {}; {}; eligible {}"
        ).format(
            _number(completion),
            "—" if failures < 0 else failures,
            "dominated" if dominated else "non-dominated",
            "yes" if eligible else "no",
        )
        row = 0
        for group_index, group in enumerate(_GROUPS):
            for gain_index, gain in enumerate(_GAINS):
                values = (
                    group, gain, _number(current[group_index, gain_index]),
                    _number(proposed[group_index, gain_index]),
                    _number(difference[group_index, gain_index]),
                    _number(ratio[group_index, gain_index]),
                    "{} … {}".format(
                        _number(range_50[0, group_index, gain_index]),
                        _number(range_50[1, group_index, gain_index]),
                    ),
                    "{} … {}".format(
                        _number(range_95[0, group_index, gain_index]),
                        _number(range_95[1, group_index, gain_index]),
                    ),
                    validation,
                )
                for column, value in enumerate(values):
                    self.gain_table.setItem(row, column, QTableWidgetItem(str(value)))
                row += 1

    def _refresh_metric_table(self) -> None:
        assert self.evaluation is not None
        summary = self.evaluation.summary
        row_count = len(_METRICS) + 2
        self.metric_table.setRowCount(row_count)
        level = float(np.asarray(summary.get("cvar_level", [np.nan])).reshape(-1)[0])
        position_threshold = float(np.asarray(summary.get("position_threshold", [np.nan])).reshape(-1)[0])
        orientation_threshold = float(np.asarray(summary.get("orientation_threshold", [np.nan])).reshape(-1)[0])
        position_metric = _text_scalar(
            summary.get("position_threshold_metric"), "position_rmse"
        )
        orientation_metric = _text_scalar(
            summary.get("orientation_threshold_metric"), "orientation_rmse"
        )
        position_configured = bool(
            np.asarray(
                summary.get("position_threshold_configured", [np.isfinite(position_threshold)])
            ).reshape(-1)[0]
        )
        orientation_configured = bool(
            np.asarray(
                summary.get("orientation_threshold_configured", [np.isfinite(orientation_threshold)])
            ).reshape(-1)[0]
        )
        for row, (key, label, unit) in enumerate(_METRICS):
            mean = np.asarray(summary.get("aggregate_{}_mean".format(key), [np.nan]))[self._candidate_index]
            cvar = np.asarray(summary.get("aggregate_{}_upper_cvar".format(key), [np.nan]))[self._candidate_index]
            is_position = key in {"position_rmse", "maximum_position_error"}
            configured = position_configured if is_position else orientation_configured
            configured_metric = position_metric if is_position else orientation_metric
            threshold = position_threshold if is_position else orientation_threshold
            exceedance_key = (
                "aggregate_position_threshold_exceedance"
                if is_position
                else "aggregate_orientation_threshold_exceedance"
            )
            if not configured:
                exceedance = "Not configured"
            elif key != configured_metric:
                exceedance = "—"
            else:
                raw_exceedance = np.asarray(
                    summary.get(exceedance_key, np.full(1, np.nan))
                )[self._candidate_index]
                exceedance = "{} (limit {} {})".format(
                    _number(float(raw_exceedance)), _number(threshold), unit
                )
            values = (label, _number(float(mean)), _number(float(cvar)), exceedance, unit, _number(level))
            for column, value in enumerate(values):
                self.metric_table.setItem(row, column, QTableWidgetItem(str(value)))
        completion = np.asarray(summary.get("forecast_completion", [np.nan]))[self._candidate_index]
        failures = np.asarray(summary.get("numerical_failure_count", [-1]))[self._candidate_index]
        for row, values in (
            (len(_METRICS), ("Forecast completion", _number(float(completion)), "—", "—", "fraction", "—")),
            (len(_METRICS) + 1, ("Numerical failure count", str(int(failures)), "—", "—", "count", "—")),
        ):
            for column, value in enumerate(values):
                self.metric_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.metric_table.resizeColumnsToContents()

    def _refresh_bag_table(self) -> None:
        assert self.evaluation is not None
        summary = self.evaluation.summary
        bag_ids = np.asarray(summary.get("bag_id", self.evaluation.manifest["selected_bag_ids"])).astype(str)
        completion = np.asarray(summary.get("member_bag_forecast_completion", []))
        position_configured = bool(
            np.asarray(summary.get("position_threshold_configured", [False])).reshape(-1)[0]
        )
        orientation_configured = bool(
            np.asarray(summary.get("orientation_threshold_configured", [False])).reshape(-1)[0]
        )
        self.bag_table.setRowCount(bag_ids.size)
        for bag_index, bag_id in enumerate(bag_ids):
            completed = completion[self._candidate_index, bag_index] if completion.size else np.empty(0, dtype=bool)
            values: list[str] = [
                bag_id,
                _number(float(np.mean(completed))) if completed.size else "—",
                str(int(completed.size - np.count_nonzero(completed))) if completed.size else "—",
            ]
            for key, _label, _unit in _METRICS:
                candidate_count = np.asarray(summary["candidate_id"]).size
                mean = np.asarray(summary.get("per_bag_{}_mean".format(key), np.full((candidate_count, bag_ids.size), np.nan)))[self._candidate_index, bag_index]
                cvar = np.asarray(summary.get("per_bag_{}_upper_cvar".format(key), np.full((candidate_count, bag_ids.size), np.nan)))[self._candidate_index, bag_index]
                values.extend((_number(float(mean)), _number(float(cvar))))
            for configured, key in (
                (position_configured, "per_bag_position_threshold_exceedance"),
                (orientation_configured, "per_bag_orientation_threshold_exceedance"),
            ):
                if not configured:
                    values.append("Not configured")
                else:
                    threshold_value = np.asarray(
                        summary.get(
                            key,
                            np.full(
                                (np.asarray(summary["candidate_id"]).size, bag_ids.size),
                                np.nan,
                            ),
                        )
                    )[self._candidate_index, bag_index]
                    values.append(_number(float(threshold_value)))
            coverage = []
            for key in (
                "per_bag_correction_translation_zero_coverage",
                "per_bag_correction_rotation_zero_coverage",
                "per_bag_correction_transform_zero_coverage",
            ):
                raw = np.asarray(
                    summary.get(
                        key,
                        np.full(
                            (np.asarray(summary["candidate_id"]).size, bag_ids.size),
                            np.nan,
                        ),
                    )
                )[self._candidate_index, bag_index]
                coverage.append(_number(float(raw)))
            values.append(" / ".join(coverage))
            for column, value in enumerate(values):
                self.bag_table.setItem(bag_index, column, QTableWidgetItem(value))
        self.bag_table.resizeColumnsToContents()

    def _selected_indices(self) -> tuple[str, int] | None:
        assert self.evaluation is not None
        record = self.store.current_record()
        if record is None or record.bag_id not in self.evaluation.bags:
            return None
        bag_id = record.bag_id
        member_ids = np.asarray(self.evaluation.bags[bag_id]["member_id"])
        requested = self.store.selected_member_id
        matches = np.flatnonzero(member_ids == requested) if requested is not None else np.empty(0, dtype=int)
        return None if not matches.size else (bag_id, int(matches[0]))

    def _refresh_member_and_paths(self) -> None:
        if self.evaluation is None or not self.evaluation.bags:
            self.comparison_scene.set_selection(None, None, None)
            return
        selection = self._selected_indices()
        if selection is None:
            self.member_detail.setPlainText(
                "The current GUI bag/member selection is not present in this evaluation."
            )
            self.translation_plot.clear()
            self.rotation_plot.clear()
            self.comparison_scene.set_selection(None, None, None)
            return
        bag_id, member_index = selection
        arrays = self.evaluation.bags[bag_id]
        member_id = int(np.asarray(arrays["member_id"])[member_index])
        candidates = np.asarray(arrays["candidate_id"]).astype(str)
        current_matches = np.flatnonzero(candidates == "current")
        current_index = int(current_matches[0]) if current_matches.size else 0
        selected_candidate = str(candidates[self._candidate_index])
        self.comparison_scene.set_selection(
            bag_id, member_id, selected_candidate
        )
        self._render_correction_plot(arrays, current_index, self._candidate_index, member_index)
        summary = self.evaluation.summary
        posterior = self.store.parameter_ensemble
        physical = "source physical parameters unavailable"
        if posterior is not None:
            matches = np.flatnonzero(posterior.member_id == member_id)
            if matches.size:
                index = int(matches[0])
                physical = (
                    "mass={} kg\ninertia={} kg m²\nCoG={} m\nforce effectiveness={}\n"
                    "torque effectiveness={}\nconstant delay τ={} s"
                ).format(
                    _number(posterior.mass[index]),
                    np.array2string(posterior.inertia[index], precision=6),
                    np.array2string(posterior.cog[index], precision=6),
                    np.array2string(posterior.force_effectiveness[index], precision=6),
                    np.array2string(posterior.torque_effectiveness[index], precision=6),
                    _number(posterior.constant_delay[index]),
                )
        metrics = []
        summary_bag_ids = np.asarray(
            summary.get("bag_id", list(self.evaluation.bags))
        ).astype(str)
        bag_index = int(np.flatnonzero(summary_bag_ids == bag_id)[0])
        for key, label, unit in _METRICS:
            raw = summary.get("member_bag_{}".format(key))
            if raw is not None:
                value = np.asarray(raw)[self._candidate_index, bag_index, member_index]
                metrics.append("{}: {} {}".format(label, _number(value), unit))
        for configured_key, exceeded_key, label in (
            (
                "position_threshold_configured",
                "member_bag_position_threshold_exceeded",
                "Position threshold exceeded",
            ),
            (
                "orientation_threshold_configured",
                "member_bag_orientation_threshold_exceeded",
                "Orientation threshold exceeded",
            ),
        ):
            configured = bool(
                np.asarray(summary.get(configured_key, [False])).reshape(-1)[0]
            )
            if configured and exceeded_key in summary:
                exceeded = bool(
                    np.asarray(summary[exceeded_key])[
                        self._candidate_index, bag_index, member_index
                    ]
                )
                metrics.append("{}: {}".format(label, "yes" if exceeded else "no"))
        proposal = self.evaluation.proposal_ensemble
        source_ids = np.asarray(proposal["source_member_id"])
        source_matches = np.flatnonzero(source_ids == member_id)
        member_proposal = "member-derived PID proposal unavailable"
        if source_matches.size:
            proposal_index = int(source_matches[0])
            member_proposal = "member-derived PID proposal:\n{}".format(
                np.array2string(
                    np.asarray(proposal["proposed_pid"])[proposal_index], precision=7
                )
            )
        success = bool(np.asarray(arrays["forecast_success"])[self._candidate_index, member_index])
        reason = str(np.asarray(arrays["forecast_failure_reason"])[self._candidate_index, member_index])
        residual_policy = str(
            np.asarray(
                arrays.get(
                    "residual_policy",
                    np.full(candidates.size, "unspecified"),
                )
            )[self._candidate_index]
        )
        self.member_detail.setPlainText(
            "Member {}\nBag {}\n\n{}\n\n{}\n\nCandidate {}\n{}\nForecast completed: {}\nNumerical failure reason: {}\nResidual policy: {}".format(
                member_id, bag_id, physical, member_proposal, candidates[self._candidate_index],
                "\n".join(metrics) if metrics else "Member metrics unavailable",
                "yes" if success else "no", reason or "none",
                residual_policy,
            )
        )

    def _render_correction_plot(
        self,
        arrays: dict[str, np.ndarray] | object,
        current_index: int,
        selected_index: int,
        member_index: int,
    ) -> None:
        times = np.asarray(arrays["times"])
        translation = np.asarray(arrays["correction_translation"])
        rotation = np.asarray(arrays["correction_rotation_vector"])
        success = np.asarray(arrays["forecast_success"])
        colors = ((210, 70, 70), (50, 145, 90), (65, 105, 205))
        for plot, paths in ((self.translation_plot, translation), (self.rotation_plot, rotation)):
            plot.clear()
            plot.addLine(y=0.0, pen=pg.mkPen((80, 80, 80), style=Qt.DashLine))
            for component, color in enumerate(colors):
                for candidate, style, alpha in (
                    (current_index, Qt.DashLine, 150),
                    (selected_index, Qt.SolidLine, 235),
                ):
                    valid = success[candidate]
                    if np.any(valid):
                        member_paths = paths[candidate, valid, :, component]
                        lower, upper = np.quantile(member_paths, (0.05, 0.95), axis=0)
                        low_curve = plot.plot(times, lower, pen=pg.mkPen(None))
                        high_curve = plot.plot(times, upper, pen=pg.mkPen(None))
                        band_alpha = 10 if candidate == current_index else 32
                        plot.addItem(
                            pg.FillBetweenItem(
                                low_curve,
                                high_curve,
                                brush=pg.mkBrush(*color, band_alpha),
                            )
                        )
                    if success[candidate, member_index]:
                        plot.plot(
                            times, paths[candidate, member_index, :, component],
                            pen=pg.mkPen((*color, alpha), width=1.8, style=style),
                        )


__all__ = ["NextExperimentView"]
