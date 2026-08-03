"""Stage-oriented launch dialog for the resumable estimation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..workflow import StageStatus, WorkflowMode


_STAGE_ORDER = ("diagonal_q", "static_parameters")
_STAGE_COPY = {
    "diagonal_q": (
        "Estimate diagonal Q",
        "Estimate the six diagonal entries of disturbance covariance Q.",
    ),
    "static_parameters": (
        "Estimate static parameters",
        "Reuse the completed Q artifact to estimate 18 vehicle parameters and one delay.",
    ),
}
_STARTABLE = {StageStatus.READY, StageStatus.RETRY, StageStatus.STALE}
_BADGE_STYLE = {
    StageStatus.READY: "background: #e7f1ff; color: #174a7e;",
    StageStatus.BLOCKED: "background: #eceff1; color: #54606a;",
    StageStatus.RUNNING: "background: #dceeff; color: #075985;",
    StageStatus.COMPLETE: "background: #dcf3e4; color: #17603a;",
    StageStatus.RETRY: "background: #fff0cf; color: #76510b;",
    StageStatus.STALE: "background: #ffe6d5; color: #8a3f0a;",
}


@dataclass(frozen=True)
class WorkflowLaunchSelection:
    """Immutable value emitted when the user starts a workflow."""

    mode: WorkflowMode


class WorkflowLaunchDialog(QDialog):
    """Choose staged or continuous execution and inspect stage readiness."""

    launchRequested = Signal(object)

    def __init__(
        self,
        statuses: Mapping[str, StageStatus | str],
        *,
        reusable_artifacts: Mapping[str, bool] | None = None,
        artifact_details: Mapping[str, str] | None = None,
        running: bool = False,
        selected_mode: WorkflowMode | str = WorkflowMode.STEP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workflowLaunchDialog")
        self.setWindowTitle("Run parameter estimation")
        self.setModal(True)
        self.setMinimumWidth(560)

        if set(statuses) != set(_STAGE_ORDER):
            raise ValueError(
                "statuses must contain exactly diagonal_q and static_parameters"
            )
        artifact_flags = dict(reusable_artifacts or {})
        unexpected_artifacts = set(artifact_flags) - set(_STAGE_ORDER)
        if unexpected_artifacts:
            raise ValueError(
                "unexpected reusable-artifact stages: {}".format(
                    ", ".join(sorted(unexpected_artifacts))
                )
            )
        details = dict(artifact_details or {})
        unexpected_details = set(details) - set(_STAGE_ORDER)
        if unexpected_details:
            raise ValueError(
                "unexpected artifact-detail stages: {}".format(
                    ", ".join(sorted(unexpected_details))
                )
            )
        if any(not isinstance(value, str) for value in details.values()):
            raise ValueError("artifact details must be text")

        self._statuses = {
            stage_id: StageStatus(statuses[stage_id])
            for stage_id in _STAGE_ORDER
        }
        self._reusable_artifacts = {
            stage_id: bool(artifact_flags.get(stage_id, False))
            for stage_id in _STAGE_ORDER
        }
        self._artifact_details = {
            stage_id: details.get(stage_id, "").strip()
            for stage_id in _STAGE_ORDER
        }
        self._explicit_running = bool(running)
        self._selection: WorkflowLaunchSelection | None = None
        self.stage_badges: dict[str, QLabel] = {}
        self.artifact_labels: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        introduction = QLabel(
            "Choose whether to stop after each stage or continue through all ready stages."
        )
        introduction.setWordWrap(True)
        root.addWidget(introduction)

        mode_group = QGroupBox("Execution")
        mode_layout = QVBoxLayout(mode_group)
        self.staged_mode_radio = QRadioButton(
            "Run one stage at a time (recommended)"
        )
        self.staged_mode_radio.setObjectName("stagedModeRadio")
        mode_layout.addWidget(self.staged_mode_radio)
        staged_help = QLabel("Stop after each completed stage.")
        staged_help.setStyleSheet("color: #5f6368; margin-left: 22px;")
        mode_layout.addWidget(staged_help)
        self.all_mode_radio = QRadioButton("Run all stages")
        self.all_mode_radio.setObjectName("allModeRadio")
        mode_layout.addWidget(self.all_mode_radio)
        self.all_mode_help = QLabel(
            "Continue automatically after a stage completes; scientific "
            "non-convergence pauses for review."
        )
        self.all_mode_help.setObjectName("allModeHelp")
        self.all_mode_help.setWordWrap(True)
        self.all_mode_help.setStyleSheet(
            "color: #5f6368; margin-left: 22px;"
        )
        mode_layout.addWidget(self.all_mode_help)
        root.addWidget(mode_group)

        stages_group = QGroupBox("Stages")
        stages_layout = QVBoxLayout(stages_group)
        stages_layout.setSpacing(8)
        for index, stage_id in enumerate(_STAGE_ORDER, start=1):
            stages_layout.addWidget(self._make_stage_row(index, stage_id))
        root.addWidget(stages_group)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.start_button = self.button_box.button(QDialogButtonBox.Ok)
        self.start_button.setText("Start")
        self.start_button.setObjectName("startWorkflowButton")
        self.cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        self.cancel_button.setObjectName("cancelWorkflowButton")
        self.button_box.accepted.connect(self._start)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        initial_mode = WorkflowMode(selected_mode)
        self.staged_mode_radio.setChecked(initial_mode is WorkflowMode.STEP)
        self.all_mode_radio.setChecked(initial_mode is WorkflowMode.ALL)
        self._refresh_stage_rows()
        self._refresh_enabled_state()

    def _make_stage_row(self, index: int, stage_id: str) -> QFrame:
        title, detail = _STAGE_COPY[stage_id]
        row = QFrame()
        row.setObjectName("{}StageRow".format(stage_id))
        row.setFrameShape(QFrame.StyledPanel)
        layout = QGridLayout(row)
        layout.setColumnStretch(1, 1)

        number = QLabel(str(index))
        number.setAlignment(Qt.AlignCenter)
        number.setFixedSize(24, 24)
        number.setStyleSheet(
            "background: #eef1f5; color: #39434d; border-radius: 12px;"
        )
        layout.addWidget(number, 0, 0, 2, 1, Qt.AlignTop)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(title_label, 0, 1)
        detail_label = QLabel(detail)
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("color: #5f6368;")
        layout.addWidget(detail_label, 1, 1)

        badge = QLabel()
        badge.setObjectName("{}StatusBadge".format(stage_id))
        badge.setAlignment(Qt.AlignCenter)
        badge.setMinimumWidth(84)
        layout.addWidget(badge, 0, 2, 2, 1, Qt.AlignTop)
        self.stage_badges[stage_id] = badge

        artifact = QLabel()
        artifact.setObjectName("{}ArtifactReuse".format(stage_id))
        artifact.setWordWrap(True)
        artifact.setStyleSheet("color: #5f6368; font-size: 11px;")
        layout.addWidget(artifact, 2, 1, 1, 2)
        self.artifact_labels[stage_id] = artifact
        return row

    @property
    def selected_mode(self) -> WorkflowMode:
        return (
            WorkflowMode.ALL
            if self.all_mode_radio.isChecked()
            else WorkflowMode.STEP
        )

    @property
    def launch_selection(self) -> WorkflowLaunchSelection | None:
        return self._selection

    @property
    def running(self) -> bool:
        return self._explicit_running or any(
            status is StageStatus.RUNNING for status in self._statuses.values()
        )

    def set_selected_mode(self, mode: WorkflowMode | str) -> None:
        """Change the proposed mode unless an attempt is running."""

        if self.running:
            return
        resolved = WorkflowMode(mode)
        self.staged_mode_radio.setChecked(resolved is WorkflowMode.STEP)
        self.all_mode_radio.setChecked(resolved is WorkflowMode.ALL)

    def set_running(self, running: bool) -> None:
        """Lock or unlock launch controls for an external running attempt."""

        self._explicit_running = bool(running)
        self._refresh_enabled_state()

    def set_stage_status(
        self,
        stage_id: str,
        status: StageStatus | str,
        *,
        reusable_artifact: bool = False,
        artifact_detail: str = "",
    ) -> None:
        """Update one already-derived stage status and its reuse indicator."""

        if stage_id not in _STAGE_ORDER:
            raise ValueError("unknown workflow stage: {}".format(stage_id))
        self._statuses[stage_id] = StageStatus(status)
        self._reusable_artifacts[stage_id] = bool(reusable_artifact)
        if not isinstance(artifact_detail, str):
            raise ValueError("artifact detail must be text")
        self._artifact_details[stage_id] = artifact_detail.strip()
        self._refresh_stage_row(stage_id)
        self._refresh_enabled_state()

    def status_text(self, stage_id: str) -> str:
        return self.stage_badges[stage_id].text()

    def artifact_text(self, stage_id: str) -> str:
        return self.artifact_labels[stage_id].text()

    def _refresh_stage_rows(self) -> None:
        for stage_id in _STAGE_ORDER:
            self._refresh_stage_row(stage_id)

    def _refresh_stage_row(self, stage_id: str) -> None:
        status = self._statuses[stage_id]
        badge = self.stage_badges[stage_id]
        badge.setText(status.value)
        badge.setStyleSheet(
            "{} padding: 4px 8px; border-radius: 4px; font-weight: 600;".format(
                _BADGE_STYLE[status]
            )
        )
        if self._reusable_artifacts[stage_id]:
            artifact_text = "Completed artifact will be reused."
        elif status is StageStatus.STALE:
            artifact_text = "Completed artifact is stale and will not be reused."
        elif status is StageStatus.COMPLETE:
            artifact_text = "Completed artifact is not selected for reuse."
        else:
            artifact_text = "No reusable completed artifact."
        detail = self._artifact_details[stage_id]
        if detail:
            artifact_text = "{} {}".format(artifact_text, detail)
        self.artifact_labels[stage_id].setText(artifact_text)

    def _refresh_enabled_state(self) -> None:
        locked = self.running
        self.staged_mode_radio.setEnabled(not locked)
        self.all_mode_radio.setEnabled(not locked)
        has_startable_stage = any(
            status in _STARTABLE for status in self._statuses.values()
        )
        self.start_button.setEnabled(not locked and has_startable_stage)
        self.cancel_button.setEnabled(True)

    def _start(self) -> None:
        if not self.start_button.isEnabled():
            return
        self._selection = WorkflowLaunchSelection(mode=self.selected_mode)
        self.launchRequested.emit(self._selection)
        super().accept()

    def reject(self) -> None:
        self._selection = None
        super().reject()
