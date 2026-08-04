"""Choose the stopping boundary of one sparse batch estimation run."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..workflow import WorkflowMode


@dataclass(frozen=True)
class WorkflowLaunchSelection:
    mode: WorkflowMode


class WorkflowLaunchDialog(QDialog):
    """Map STEP/ALL directly to estimate-only/estimate-and-sample."""

    launchRequested = Signal(object)

    def __init__(
        self,
        *,
        running: bool = False,
        selected_mode: WorkflowMode | str = WorkflowMode.STEP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workflowLaunchDialog")
        self.setWindowTitle("Run sparse batch estimation")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._selection: WorkflowLaunchSelection | None = None
        root = QVBoxLayout(self)
        introduction = QLabel(
            "Choose whether this run stops after MAP, Laplace-EM, and local "
            "posterior geometry, or continues into resumable MCMC sampling."
        )
        introduction.setWordWrap(True)
        root.addWidget(introduction)
        group = QGroupBox("Stopping boundary")
        group_layout = QVBoxLayout(group)
        self.staged_mode_radio = QRadioButton(
            "Estimate only (recommended for an initial review)"
        )
        self.staged_mode_radio.setObjectName("estimateOnlyRadio")
        group_layout.addWidget(self.staged_mode_radio)
        estimate_help = QLabel(
            "Run full-trajectory MAP, delay refinement, diagonal-Q Laplace-EM, "
            "ridge diagnostics, and local posterior geometry, then stop."
        )
        estimate_help.setWordWrap(True)
        group_layout.addWidget(estimate_help)
        self.all_mode_radio = QRadioButton("Estimate and sample posterior")
        self.all_mode_radio.setObjectName("estimateAndSampleRadio")
        group_layout.addWidget(self.all_mode_radio)
        sample_help = QLabel(
            "Continue the same run with multi-chain ridge-aware MCMC. "
            "Cancellation is checked at proposal boundaries and the chain "
            "checkpoint can be resumed."
        )
        sample_help.setWordWrap(True)
        group_layout.addWidget(sample_help)
        root.addWidget(group)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.start_button = self.button_box.button(QDialogButtonBox.Ok)
        self.start_button.setText("Start")
        self.button_box.accepted.connect(self._start)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)
        self.set_selected_mode(selected_mode)
        self.set_running(running)

    @property
    def selected_mode(self) -> WorkflowMode:
        return WorkflowMode.ALL if self.all_mode_radio.isChecked() else WorkflowMode.STEP

    @property
    def launch_selection(self) -> WorkflowLaunchSelection | None:
        return self._selection

    def set_selected_mode(self, mode: WorkflowMode | str) -> None:
        selected = WorkflowMode(mode)
        self.staged_mode_radio.setChecked(selected is WorkflowMode.STEP)
        self.all_mode_radio.setChecked(selected is WorkflowMode.ALL)

    def set_running(self, running: bool) -> None:
        locked = bool(running)
        self.staged_mode_radio.setEnabled(not locked)
        self.all_mode_radio.setEnabled(not locked)
        self.start_button.setEnabled(not locked)

    def _start(self) -> None:
        if not self.start_button.isEnabled():
            return
        self._selection = WorkflowLaunchSelection(self.selected_mode)
        self.launchRequested.emit(self._selection)
        super().accept()

    def reject(self) -> None:
        self._selection = None
        super().reject()


__all__ = ["WorkflowLaunchDialog", "WorkflowLaunchSelection"]
