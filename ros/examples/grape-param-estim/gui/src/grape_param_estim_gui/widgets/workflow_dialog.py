"""Choose the stopping boundary of one sparse batch estimation run."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..workflow import WorkflowMode


@dataclass(frozen=True)
class WorkflowLaunchSelection:
    mode: WorkflowMode
    q_update_policy: str
    solver_method: str
    maximum_iterations: int

    def __post_init__(self) -> None:
        if self.q_update_policy not in {"fixed", "laplace_em"}:
            raise ValueError("q_update_policy must be fixed or laplace_em")
        if self.solver_method not in {"sparse_lm", "ieks"}:
            raise ValueError("solver_method must be sparse_lm or ieks")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, Integral)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be a positive integer")


class WorkflowLaunchDialog(QDialog):
    """Map STEP/ALL directly to estimate-only/estimate-and-sample."""

    launchRequested = Signal(object)

    def __init__(
        self,
        *,
        running: bool = False,
        selected_mode: WorkflowMode | str = WorkflowMode.STEP,
        selected_q_update_policy: str = "fixed",
        selected_solver_method: str = "sparse_lm",
        selected_maximum_iterations: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workflowLaunchDialog")
        self.setWindowTitle("Run sparse batch estimation")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(620)
        self._selection: WorkflowLaunchSelection | None = None
        root = QVBoxLayout(self)
        introduction = QLabel(
            "Choose the solver, iteration limit, Q handling, and stopping "
            "boundary for this run."
        )
        introduction.setWordWrap(True)
        root.addWidget(introduction)
        solver_group = QGroupBox("Nonlinear trajectory solver")
        solver_layout = QVBoxLayout(solver_group)
        self.sparse_lm_radio = QRadioButton(
            "Sparse batch Levenberg-Marquardt (existing method)"
        )
        self.sparse_lm_radio.setObjectName("sparseLmRadio")
        solver_layout.addWidget(self.sparse_lm_radio)
        self.ieks_radio = QRadioButton(
            "Iterative Extended Kalman Smoother (IEKS)"
        )
        self.ieks_radio.setObjectName("ieksRadio")
        solver_layout.addWidget(self.ieks_radio)
        ieks_help = QLabel(
            "IEKS repeatedly relinearizes, filters forward, and smooths "
            "backward. Lag stays outside its state and is optimized by the "
            "separate delay profile."
        )
        ieks_help.setWordWrap(True)
        solver_layout.addWidget(ieks_help)
        iteration_label = QLabel("Maximum nonlinear iterations")
        solver_layout.addWidget(iteration_label)
        self.maximum_iterations_spin = QSpinBox()
        self.maximum_iterations_spin.setObjectName("maximumIterationsSpin")
        self.maximum_iterations_spin.setRange(1, 10000)
        self.maximum_iterations_spin.setSuffix(" iterations")
        self.maximum_iterations_spin.setToolTip(
            "Per lag candidate, IEKS relinearizes up to this many times and "
            "may stop earlier when a convergence tolerance is met."
        )
        solver_layout.addWidget(self.maximum_iterations_spin)
        iteration_help = QLabel(
            "This is the nonlinear convergence limit for each delay-profile "
            "candidate, not a count of complete estimation repetitions. "
            "Values below 10 can stop before a usable Laplace point; 30 is "
            "the default."
        )
        iteration_help.setWordWrap(True)
        solver_layout.addWidget(iteration_help)
        root.addWidget(solver_group)
        q_group = QGroupBox("Model-error covariance Q")
        q_layout = QVBoxLayout(q_group)
        self.fixed_q_radio = QRadioButton(
            "Keep the configured Q fixed (recommended for diagnosis; faster)"
        )
        self.fixed_q_radio.setObjectName("fixedQRadio")
        q_layout.addWidget(self.fixed_q_radio)
        fixed_help = QLabel(
            "Run one delay-profiled trajectory MAP/Laplace solve. The displayed "
            "Q target is diagnostic only and is not applied."
        )
        fixed_help.setWordWrap(True)
        q_layout.addWidget(fixed_help)
        self.estimate_q_radio = QRadioButton(
            "Estimate Q with diagonal-Q Laplace-EM (slow)"
        )
        self.estimate_q_radio.setObjectName("estimateQRadio")
        q_layout.addWidget(self.estimate_q_radio)
        em_help = QLabel(
            "Re-solve the trajectory MAP for Q candidates and refine delay "
            "across EM iterations. This can take several times longer."
        )
        em_help.setWordWrap(True)
        q_layout.addWidget(em_help)
        root.addWidget(q_group)
        group = QGroupBox("Stopping boundary")
        group_layout = QVBoxLayout(group)
        self.staged_mode_radio = QRadioButton(
            "Estimate only (recommended for an initial review)"
        )
        self.staged_mode_radio.setObjectName("estimateOnlyRadio")
        group_layout.addWidget(self.staged_mode_radio)
        estimate_help = QLabel(
            "Run full-trajectory MAP, delay refinement, the selected Q policy, "
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
        self.set_selected_q_update_policy(selected_q_update_policy)
        self.set_selected_solver_method(selected_solver_method)
        self.set_selected_maximum_iterations(selected_maximum_iterations)
        self.set_running(running)

    @property
    def selected_mode(self) -> WorkflowMode:
        return WorkflowMode.ALL if self.all_mode_radio.isChecked() else WorkflowMode.STEP

    @property
    def launch_selection(self) -> WorkflowLaunchSelection | None:
        return self._selection

    @property
    def selected_q_update_policy(self) -> str:
        return "fixed" if self.fixed_q_radio.isChecked() else "laplace_em"

    @property
    def selected_solver_method(self) -> str:
        return "ieks" if self.ieks_radio.isChecked() else "sparse_lm"

    @property
    def selected_maximum_iterations(self) -> int:
        return int(self.maximum_iterations_spin.value())

    def set_selected_mode(self, mode: WorkflowMode | str) -> None:
        selected = WorkflowMode(mode)
        self.staged_mode_radio.setChecked(selected is WorkflowMode.STEP)
        self.all_mode_radio.setChecked(selected is WorkflowMode.ALL)

    def set_selected_q_update_policy(self, policy: str) -> None:
        if policy not in {"fixed", "laplace_em"}:
            raise ValueError("q_update_policy must be fixed or laplace_em")
        self.fixed_q_radio.setChecked(policy == "fixed")
        self.estimate_q_radio.setChecked(policy == "laplace_em")

    def set_selected_solver_method(self, method: str) -> None:
        if method not in {"sparse_lm", "ieks"}:
            raise ValueError("solver_method must be sparse_lm or ieks")
        self.sparse_lm_radio.setChecked(method == "sparse_lm")
        self.ieks_radio.setChecked(method == "ieks")

    def set_selected_maximum_iterations(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("maximum_iterations must be a positive integer")
        if not (
            self.maximum_iterations_spin.minimum()
            <= value
            <= self.maximum_iterations_spin.maximum()
        ):
            raise ValueError("maximum_iterations is outside the GUI range")
        self.maximum_iterations_spin.setValue(int(value))

    def set_running(self, running: bool) -> None:
        locked = bool(running)
        self.staged_mode_radio.setEnabled(not locked)
        self.all_mode_radio.setEnabled(not locked)
        self.fixed_q_radio.setEnabled(not locked)
        self.estimate_q_radio.setEnabled(not locked)
        self.sparse_lm_radio.setEnabled(not locked)
        self.ieks_radio.setEnabled(not locked)
        self.maximum_iterations_spin.setEnabled(not locked)
        self.start_button.setEnabled(not locked)

    def _start(self) -> None:
        if not self.start_button.isEnabled():
            return
        self._selection = WorkflowLaunchSelection(
            self.selected_mode,
            self.selected_q_update_policy,
            self.selected_solver_method,
            self.selected_maximum_iterations,
        )
        self.launchRequested.emit(self._selection)
        super().accept()

    def reject(self) -> None:
        self._selection = None
        super().reject()


__all__ = ["WorkflowLaunchDialog", "WorkflowLaunchSelection"]
