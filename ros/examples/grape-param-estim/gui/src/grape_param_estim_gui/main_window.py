from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

from PySide6.QtCore import QPoint, QRect, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QToolBar,
)

from .artifact_loader import (
    BatchEstimationRun,
    GuiArtifactError,
    load_batch_estimation_run,
    load_inspection,
    load_pid_evaluation,
)
from .process_runner import EstimatorProcessRunner
from .pid_request import (
    PidEvaluationLaunchOptions,
    build_pid_evaluation_request,
)
from .project_io import (
    GUI_STATE_NAME,
    ProjectIoError,
    copy_bag_into_project,
    load_project_archive,
    read_project_manifest,
    save_project_archive,
    unique_project_id,
    utc_now,
    write_project_manifest,
)
from .stage_requests import (
    DIAGONAL_Q_STAGE_ID,
    STATIC_PARAMETERS_STAGE_ID,
    augmented_parameter_stage_settings,
    build_augmented_parameter_stage_request,
    build_diagonal_q_stage_request,
    diagonal_q_stage_settings,
    stage_bag_requests,
)
from .state import BagRecord, ProjectStore
from .workflow import (
    StageAttempt,
    StageStatus,
    UpstreamRef,
    WorkflowError,
    WorkflowMode,
    canonical_fingerprint,
    stage_input_fingerprint,
)
from .workflow_artifacts import artifact_ref_from_validated_bundle
from .workflow_io import (
    WorkflowIoError,
    load_workflow,
    recover_interrupted_attempt,
    save_workflow,
)
from .widgets.bag_browser import BagBrowserView
from .widgets.master_view import MasterView
from .widgets.next_experiment import NextExperimentView
from .widgets.workflow_dialog import WorkflowLaunchDialog


class MainWindow(QMainWindow):
    _PREFERRED_SIZE = QSize(1680, 1040)
    _MINIMUM_SIZE = QSize(960, 640)
    _SCREEN_MARGIN = QSize(48, 64)

    def __init__(self, store: ProjectStore, package_root: str | Path) -> None:
        super().__init__()
        self.store = store
        self.package_root = Path(package_root).resolve()
        self.worker_python = os.environ.get(
            "GRAPE_PARAM_ESTIM_WORKER_PYTHON", "/usr/bin/python3"
        )
        self.runner = EstimatorProcessRunner(self)
        self._operation: str | None = None
        self._operation_context: dict[str, object] = {}
        self._log_path: Path | None = None
        self._workflow_state = None
        self._workflow_error: str | None = None
        self._close_after_worker = False
        self._pending_configuration_prompts: tuple[
            tuple[str, str, str], ...
        ] = ()
        self._configuration_prompt_timer = QTimer(self)
        self._configuration_prompt_timer.setSingleShot(True)
        self._configuration_prompt_timer.timeout.connect(
            self._show_pending_configuration_prompts
        )
        self.setWindowTitle("Grape sparse batch parameter estimation")
        self._reload_workflow_state()

        self.tabs = QTabWidget()
        self.master_view = MasterView(store)
        self.bag_browser = BagBrowserView(store)
        self.next_experiment = NextExperimentView(store)
        self.tabs.addTab(self.master_view, "Master")
        self.tabs.addTab(self.bag_browser, "Bag browser")
        self.tabs.addTab(self.next_experiment, "Next experiment")
        self.setCentralWidget(self.tabs)
        self._build_toolbar()
        self._build_menu()
        self._build_shortcuts()
        self._connect_state()
        self._update_freshness(self.store.results_stale)
        self._update_run_action()
        self.statusBar().showMessage("Add rosbag files to start inspection.")
        self._fit_initial_geometry_to_screen()

    def _connect_state(self) -> None:
        self.master_view.bagActivated.connect(self._open_bag_from_master)
        self.bag_browser.statusMessage.connect(self.statusBar().showMessage)
        self.bag_browser.filesSelected.connect(self.add_bag_files)
        self.bag_browser.reinspectionRequested.connect(self.inspect_bags)
        self.bag_browser.configurationRequested.connect(self.set_configuration_provenance)
        self.bag_browser.configurationGroupRequested.connect(
            self.confirm_configuration_group
        )
        self.next_experiment.evaluationRequested.connect(self.start_pid_evaluation)
        self.bag_browser.time_state.playingChanged.connect(self._update_play_action)
        self.store.freshnessChanged.connect(self._update_freshness)
        self.store.projectChanged.connect(self._update_project_title)
        self.store.projectChanged.connect(self._cancel_configuration_prompts)
        self.store.projectChanged.connect(self._reload_workflow_state)
        self.store.bagsChanged.connect(self._update_run_action)
        self.store.recordChanged.connect(lambda _bag_id: self._update_run_action())
        self.runner.progress.connect(self._update_progress)
        self.runner.stderrLog.connect(self._worker_log)
        self.runner.finished.connect(self._worker_finished)
        self.runner.cancelled.connect(self._worker_cancelled)
        self.runner.failed.connect(self._worker_failed)
        self.runner.runningChanged.connect(self._worker_running_changed)

    def _fit_initial_geometry_to_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            self.resize(self._PREFERRED_SIZE)
            return
        available = screen.availableGeometry()
        width = min(self._PREFERRED_SIZE.width(), max(1, available.width() - self._SCREEN_MARGIN.width()))
        height = min(self._PREFERRED_SIZE.height(), max(1, available.height() - self._SCREEN_MARGIN.height()))
        self.setMinimumSize(min(self._MINIMUM_SIZE.width(), width), min(self._MINIMUM_SIZE.height(), height))
        geometry = QRect(QPoint(0, 0), QSize(width, height))
        geometry.moveCenter(available.center())
        self.setGeometry(geometry)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.add_bags_action = QAction("Add bags…", self)
        self.add_bags_action.triggered.connect(self.bag_browser.open_files)
        toolbar.addAction(self.add_bags_action)
        self.save_project_action = QAction("Save Project…", self)
        self.save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(self.save_project_action)
        self.load_project_action = QAction("Load Project…", self)
        self.load_project_action.triggered.connect(self.load_project)
        toolbar.addAction(self.load_project_action)
        toolbar.addSeparator()
        self.run_action = QAction("Run estimation…", self)
        self.run_action.triggered.connect(self.start_assimilation)
        toolbar.addAction(self.run_action)
        self.stop_action = QAction("Stop", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(
            lambda _checked=False: self.runner.request_cancel("user_requested")
        )
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        self.previous_action = QAction("◀", self)
        self.previous_action.triggered.connect(lambda: self.bag_browser.step_samples(-1))
        toolbar.addAction(self.previous_action)
        self.play_action = QAction("▶", self)
        self.play_action.triggered.connect(self.bag_browser.toggle_playback)
        toolbar.addAction(self.play_action)
        self.next_action = QAction("▶|", self)
        self.next_action.triggered.connect(lambda: self.bag_browser.step_samples(1))
        toolbar.addAction(self.next_action)
        self.speed_combo = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem("×{:g}".format(speed), speed)
        self.speed_combo.setCurrentText("×1")
        self.speed_combo.currentIndexChanged.connect(
            lambda _index: self.bag_browser.set_playback_speed(float(self.speed_combo.currentData()))
        )
        toolbar.addWidget(self.speed_combo)
        toolbar.addSeparator()
        self.stage_label = QLabel("idle")
        self.stage_label.setMinimumWidth(360)
        toolbar.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setMinimumWidth(220)
        toolbar.addWidget(self.progress_bar)
        self.eta_label = QLabel("ETA —")
        self.eta_label.setMinimumWidth(100)
        toolbar.addWidget(self.eta_label)
        self.freshness_label = QLabel()
        self.freshness_label.setMinimumWidth(105)
        toolbar.addWidget(self.freshness_label)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.add_bags_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.load_project_action)
        self.load_pid_action = QAction("Import PID evaluation…", self)
        self.load_pid_action.triggered.connect(self.import_pid_evaluation)
        file_menu.addAction(self.load_pid_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = self.menuBar().addMenu("View")
        for text, widget in (
            ("Master", self.master_view),
            ("Bag browser", self.bag_browser),
            ("Next experiment", self.next_experiment),
        ):
            action = QAction(text, self)
            action.triggered.connect(lambda _checked=False, target=widget: self.tabs.setCurrentWidget(target))
            view_menu.addAction(action)
        fit_action = QAction("Fit full time range", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.bag_browser.fit_view)
        view_menu.addAction(fit_action)

    def _build_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        for key, callback in (
            ("Space", self.bag_browser.toggle_playback),
            ("Left", lambda: self.bag_browser.step_samples(-1)),
            ("Right", lambda: self.bag_browser.step_samples(1)),
            ("Shift+Left", lambda: self.bag_browser.step_samples(-20)),
            ("Shift+Right", lambda: self.bag_browser.step_samples(20)),
            ("Home", self._go_to_start),
            ("End", self._go_to_end),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def add_bag_files(self, paths: object) -> None:
        if self.runner.running:
            return
        added: list[str] = []
        try:
            existing = {item["bag_id"]: item for item in self.store.manifest["bags"]}
            for raw_path in paths:  # type: ignore[union-attr]
                entry = copy_bag_into_project(self.store.project_path, Path(raw_path))
                if entry["bag_id"] in existing:
                    continue
                self.store.manifest["bags"].append(entry)
                self.store.add(
                    BagRecord(
                        bag_id=entry["bag_id"],
                        path=self.store.project_path / entry["relative_path"],
                        source_path=Path(entry["source_path"]),
                        sha256=entry["sha256"],
                    )
                )
                existing[entry["bag_id"]] = entry
                added.append(entry["bag_id"])
            write_project_manifest(self.store.project_path, self.store.manifest)
        except (OSError, ProjectIoError, ValueError) as error:
            self._show_error("Cannot add rosbag", error)
            return
        if added:
            self.tabs.setCurrentWidget(self.bag_browser)
            self.inspect_bags(tuple(added))
        else:
            self.statusBar().showMessage("The selected rosbag files are already in this project.")

    def inspect_bags(self, bag_ids: object = None) -> None:
        if self.runner.running or not self.store.records():
            return
        requested_ids = (
            tuple(record.bag_id for record in self.store.records())
            if bag_ids is None
            else tuple(str(value) for value in bag_ids)
        )
        include_when_ready = tuple(
            bag_id
            for bag_id in requested_ids
            if self.store.get(bag_id) is not None
            and self.store.get(bag_id).status
            in {"awaiting inspection", "needs_configuration_confirmation"}
        )
        request_id = "inspection-{}".format(uuid.uuid4().hex[:12])
        settings = self.store.manifest.get("estimator_settings", {})
        bags = []
        for record in self.store.records():
            bags.append(
                {
                    "bag_id": record.bag_id,
                    "path": str(record.path),
                    "episode_index": 0,
                    "configuration_provenance": dict(record.configuration_provenance),
                }
            )
            record.status = "inspection queued"
        request = {
            "schema": "grape-param-estim/inspection-request/v1",
            "request_id": request_id,
            "preview_max_samples": 1200,
            "bags": bags,
            "estimator_settings": {
                key: settings[key]
                for key in ("sample_period", "ensemble_size", "maximum_iterations", "maximum_knots")
                if key in settings
            },
        }
        request_path = self.store.project_path / "logs" / (request_id + ".request.json")
        output = self.store.project_path / (".inspection-" + request_id)
        self._write_request(request_path, request)
        started = self._start_worker(
            "inspection", request_id, request_path, output,
            self.package_root / "scripts" / "grape_inspect_flights.py",
        )
        if not started:
            self._restore_transient_record_statuses()
            return
        if self._operation == "inspection":
            self._operation_context["include_when_ready"] = include_when_ready

    def set_configuration_provenance(self, bag_id: str) -> None:
        record = self.store.get(bag_id)
        if record is None or self.runner.running:
            return
        fields = (
            "payload",
            "rotor_propeller",
            "geometry",
            "robot_model_revision",
            "actuator_wiring",
            "hardware_revision",
        )
        initial = {field: record.configuration_provenance.get(field, "") for field in fields}
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Configuration provenance",
            "Enter a non-empty identifier for every field as JSON:",
            json.dumps(initial, ensure_ascii=False, indent=2),
        )
        if not accepted:
            return
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError("configuration JSON must contain exactly the displayed fields")
            provenance = {field: str(value[field]).strip() for field in fields}
            if any(not value for value in provenance.values()):
                raise ValueError("every configuration provenance field must be non-empty")
        except (TypeError, ValueError) as error:
            self._show_error("Invalid configuration provenance", error)
            return
        record.configuration_provenance = provenance
        self.inspect_bags((bag_id,))

    def confirm_configuration_group(self, bag_id: str) -> None:
        record = self.store.get(bag_id)
        if record is None or self.runner.running or record.inspection is None:
            return
        fingerprint = record.inspection.get("configuration_fingerprint", {})
        missing = (
            tuple(str(value) for value in fingerprint.get("missing_components", ()))
            if isinstance(fingerprint, dict)
            else ()
        )
        suggested = (
            "single-bag-{}".format(record.sha256[:12])
            if len(self.store.records()) == 1
            else "isolated-bag-{}".format(record.sha256[:12])
        )
        group_id, accepted = QInputDialog.getText(
            self,
            "Confirm configuration group",
            "The bag did not record: {}.\n\n"
            "To enable sparse batch estimation, explicitly assign a "
            "configuration group. "
            "Use the same ID for multiple bags only when you know they used "
            "the same payload, rotors, geometry, model, wiring, and hardware. "
            "The missing provenance remains recorded as a warning.".format(
                ", ".join(missing) if missing else "hardware provenance"
            ),
            QLineEdit.Normal,
            suggested,
        )
        if not accepted:
            self.statusBar().showMessage(
                "Configuration group is still unconfirmed; parameter "
                "estimation remains disabled."
            )
            return
        try:
            self.store.confirm_configuration_group(bag_id, group_id)
            write_project_manifest(self.store.project_path, self.store.manifest)
        except (OSError, ProjectIoError, ValueError) as error:
            self._show_error("Cannot confirm configuration group", error)
            return
        self.statusBar().showMessage(
            "Confirmed manual configuration group; parameter estimation is ready."
        )

    def _schedule_configuration_prompts(
        self, bag_ids: tuple[str, ...]
    ) -> None:
        if self._close_after_worker:
            return
        prompts: list[tuple[str, str, str]] = []
        for bag_id in bag_ids:
            record = self.store.get(bag_id)
            if record is None or record.inspection is None:
                continue
            fingerprint = record.inspection.get("configuration_fingerprint")
            source_fingerprint = (
                str(fingerprint.get("value", ""))
                if isinstance(fingerprint, dict)
                else ""
            )
            if source_fingerprint:
                prompts.append(
                    (self.store.project_id, bag_id, source_fingerprint)
                )
        self._pending_configuration_prompts = tuple(prompts)
        if prompts:
            self._configuration_prompt_timer.start(0)

    def _show_pending_configuration_prompts(self) -> None:
        prompts = self._pending_configuration_prompts
        self._pending_configuration_prompts = ()
        if self._close_after_worker or not self.isVisible():
            return
        for project_id, bag_id, expected_fingerprint in prompts:
            if (
                self._close_after_worker
                or not self.isVisible()
                or self.store.project_id != project_id
            ):
                return
            record = self.store.get(bag_id)
            if (
                record is None
                or record.status != "needs_configuration_confirmation"
                or record.inspection is None
            ):
                continue
            fingerprint = record.inspection.get("configuration_fingerprint")
            current_fingerprint = (
                str(fingerprint.get("value", ""))
                if isinstance(fingerprint, dict)
                else ""
            )
            if current_fingerprint == expected_fingerprint:
                self.confirm_configuration_group(bag_id)

    def _cancel_configuration_prompts(self) -> None:
        self._configuration_prompt_timer.stop()
        self._pending_configuration_prompts = ()

    def _reload_workflow_state(self) -> None:
        """Load project-local workflow state and recover an interrupted worker."""

        self._workflow_state = None
        self._workflow_error = None
        try:
            loaded = load_workflow(
                self.store.project_path, self.store.project_id
            )
            recovered = recover_interrupted_attempt(loaded)
            if recovered != loaded:
                save_workflow(
                    self.store.project_path,
                    self.store.project_id,
                    recovered,
                )
            self._workflow_state = recovered
        except (OSError, ValueError, WorkflowError, WorkflowIoError) as error:
            self._workflow_error = str(error)

    def start_assimilation(self) -> None:
        """Open the stage chooser and launch the next required stage."""

        if self.runner.running:
            return
        try:
            selected = self._validated_workflow_records()
            inputs = self._derive_workflow_inputs(selected)
        except (OSError, ValueError, WorkflowError, WorkflowIoError, GuiArtifactError) as error:
            self._show_error("Cannot run estimation", error)
            return
        mode = self._choose_workflow_mode(inputs)
        if mode is None:
            return
        try:
            self._workflow_state = self._workflow_state.with_mode(mode)
            self._save_workflow_state()
            self._launch_next_workflow_stage()
        except (OSError, ValueError, WorkflowError, WorkflowIoError, GuiArtifactError) as error:
            self._show_error("Cannot run estimation", error)

    def _validated_workflow_records(self) -> tuple[BagRecord, ...]:
        selected = self.store.included_records()
        if not selected:
            raise ValueError("Select at least one inspected bag.")
        if any(
            record.inspection is None or record.selected_interval is None
            for record in selected
        ):
            raise ValueError(
                "Every selected bag must have a completed inspection and interval."
            )
        if any(
            record.status not in {"ready", "complete"} for record in selected
        ):
            raise ValueError(
                "Every selected bag must have ready inspection status and "
                "confirmed configuration provenance."
            )
        fingerprints = {
            record.configuration_fingerprint for record in selected
        }
        if "" in fingerprints or len(fingerprints) != 1:
            raise ValueError(
                "Selected bags must share one confirmed configuration fingerprint."
            )
        if self._workflow_error is not None or self._workflow_state is None:
            raise WorkflowIoError(
                self._workflow_error or "workflow state is unavailable"
            )
        return selected

    def _derive_workflow_inputs(
        self, selected: tuple[BagRecord, ...]
    ) -> dict[str, object]:
        state = self._workflow_state
        if state is None:
            raise WorkflowIoError("workflow state is unavailable")
        root_fingerprint = self.store.request_fingerprint()
        bags = stage_bag_requests(selected)
        estimator_settings = self.store.manifest["estimator_settings"]
        q_settings = diagonal_q_stage_settings(estimator_settings)
        q_stage = state.stage(DIAGONAL_Q_STAGE_ID)
        q_input = stage_input_fingerprint(
            definition_fingerprint=state.definition_fingerprint,
            stage_id=q_stage.stage_id,
            algorithm_version=q_stage.algorithm_version,
            root_input_fingerprint=root_fingerprint,
            stage_settings=q_settings,
        )
        q_status = state.stage_status(DIAGONAL_Q_STAGE_ID, q_input)
        q_upstream = state.completion_ref(DIAGONAL_Q_STAGE_ID, q_input)
        q_path = None
        q_fingerprint = None
        q_detail = ""
        if q_upstream is not None:
            raise WorkflowError(
                "legacy staged Q artifacts are unsupported; create a new "
                "sparse batch estimation run"
            )

        parameter_settings = augmented_parameter_stage_settings(
            estimator_settings
        )
        parameter_stage = state.stage(STATIC_PARAMETERS_STAGE_ID)
        upstream = () if q_upstream is None else (q_upstream,)
        parameter_input = stage_input_fingerprint(
            definition_fingerprint=state.definition_fingerprint,
            stage_id=parameter_stage.stage_id,
            algorithm_version=parameter_stage.algorithm_version,
            root_input_fingerprint=root_fingerprint,
            stage_settings=parameter_settings,
            upstream=upstream,
        )
        parameter_status = state.stage_status(
            STATIC_PARAMETERS_STAGE_ID, parameter_input, upstream
        )
        parameter_upstream = state.completion_ref(
            STATIC_PARAMETERS_STAGE_ID, parameter_input, upstream
        )
        parameter_detail = ""
        if parameter_upstream is not None:
            raise WorkflowError(
                "legacy staged parameter artifacts are unsupported; create "
                "a new sparse batch estimation run"
            )
        return {
            "selected": selected,
            "root_fingerprint": root_fingerprint,
            "bags": bags,
            "q_settings": q_settings,
            "q_input": q_input,
            "q_status": q_status,
            "q_upstream": q_upstream,
            "q_path": q_path,
            "q_fingerprint": q_fingerprint,
            "q_detail": q_detail,
            "parameter_settings": parameter_settings,
            "parameter_input": parameter_input,
            "parameter_status": parameter_status,
            "parameter_upstream": parameter_upstream,
            "parameter_detail": parameter_detail,
        }

    def _verify_workflow_artifact(
        self, attempt: StageAttempt, root: Path, manifest: object
    ) -> None:
        reference = artifact_ref_from_validated_bundle(
            project_root=self.store.project_path,
            artifact_root=root,
            manifest=manifest,
            expected_stage_id=self._workflow_stage_for_attempt(attempt),
            expected_stage_input=attempt.stage_input_fingerprint,
            expected_request_fingerprint=attempt.request_fingerprint,
        )
        if reference != attempt.artifact:
            raise WorkflowError(
                "completed artifact content differs from workflow.json"
            )

    def _workflow_stage_for_attempt(self, attempt: StageAttempt) -> str:
        state = self._workflow_state
        if state is None:
            raise WorkflowIoError("workflow state is unavailable")
        for stage in state.stages:
            if attempt in stage.attempts:
                return stage.stage_id
        raise WorkflowError("workflow attempt has no owning stage")

    def _choose_workflow_mode(
        self, inputs: dict[str, object]
    ) -> WorkflowMode | None:
        state = self._workflow_state
        if state is None:
            raise WorkflowIoError("workflow state is unavailable")
        dialog = WorkflowLaunchDialog(
            {
                DIAGONAL_Q_STAGE_ID: inputs["q_status"],
                STATIC_PARAMETERS_STAGE_ID: inputs["parameter_status"],
            },
            reusable_artifacts={
                DIAGONAL_Q_STAGE_ID: inputs["q_upstream"] is not None,
                STATIC_PARAMETERS_STAGE_ID: (
                    inputs["parameter_upstream"] is not None
                ),
            },
            artifact_details={
                DIAGONAL_Q_STAGE_ID: str(inputs["q_detail"]),
                STATIC_PARAMETERS_STAGE_ID: str(
                    inputs["parameter_detail"]
                ),
            },
            selected_mode=state.mode,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return None
        selection = dialog.launch_selection
        return None if selection is None else selection.mode

    def _launch_next_workflow_stage(self) -> None:
        selected = self._validated_workflow_records()
        inputs = self._derive_workflow_inputs(selected)
        q_status = inputs["q_status"]
        parameter_status = inputs["parameter_status"]
        startable = {StageStatus.READY, StageStatus.RETRY, StageStatus.STALE}
        if q_status in startable:
            self._launch_diagonal_q_stage(inputs)
            return
        if q_status is StageStatus.COMPLETE and parameter_status in startable:
            self._launch_static_parameter_stage(inputs)
            return
        if (
            q_status is StageStatus.COMPLETE
            and parameter_status is StageStatus.COMPLETE
        ):
            self.statusBar().showMessage(
                "Both estimation stages are already complete."
            )
            return
        raise WorkflowError(
            "No estimation stage can start from the current project inputs."
        )

    def _launch_diagonal_q_stage(self, inputs: dict[str, object]) -> None:
        attempt_id = "q-{}".format(uuid.uuid4().hex[:12])
        parent = self.store.project_path / "runs" / attempt_id
        output = parent / "diagonal_q"
        request_path = parent / "request.json"
        parent.mkdir(parents=True, exist_ok=False)
        request = build_diagonal_q_stage_request(
            run_id=attempt_id,
            project_fingerprint=str(inputs["root_fingerprint"]),
            stage_input_fingerprint=str(inputs["q_input"]),
            bags=inputs["bags"],
            settings=inputs["q_settings"],
        )
        self._launch_workflow_worker(
            operation=DIAGONAL_Q_STAGE_ID,
            attempt_id=attempt_id,
            request=request,
            request_path=request_path,
            output=output,
            stage_input=str(inputs["q_input"]),
            root_input=str(inputs["root_fingerprint"]),
            upstream=(),
            script=self.package_root / "scripts" / "grape_estimate_diagonal_q.py",
        )

    def _launch_static_parameter_stage(
        self, inputs: dict[str, object]
    ) -> None:
        if (
            inputs["q_upstream"] is None
            or inputs["q_path"] is None
            or inputs["q_fingerprint"] is None
        ):
            raise WorkflowError("parameter stage requires a reusable Q artifact")
        attempt_id = "parameters-{}".format(uuid.uuid4().hex[:12])
        parent = self.store.project_path / "runs" / attempt_id
        output = parent / "assimilation_run"
        request_path = parent / "request.json"
        parent.mkdir(parents=True, exist_ok=False)
        request = build_augmented_parameter_stage_request(
            run_id=attempt_id,
            project_fingerprint=str(inputs["root_fingerprint"]),
            stage_input_fingerprint=str(inputs["parameter_input"]),
            upstream_diagonal_q_path=inputs["q_path"],
            upstream_diagonal_q_fingerprint=str(inputs["q_fingerprint"]),
            bags=inputs["bags"],
            settings=inputs["parameter_settings"],
        )
        self._launch_workflow_worker(
            operation=STATIC_PARAMETERS_STAGE_ID,
            attempt_id=attempt_id,
            request=request,
            request_path=request_path,
            output=output,
            stage_input=str(inputs["parameter_input"]),
            root_input=str(inputs["root_fingerprint"]),
            upstream=(inputs["q_upstream"],),
            script=(
                self.package_root
                / "scripts"
                / "grape_estimate_augmented_parameters.py"
            ),
        )

    def _launch_workflow_worker(
        self,
        *,
        operation: str,
        attempt_id: str,
        request: dict[str, object],
        request_path: Path,
        output: Path,
        stage_input: str,
        root_input: str,
        upstream: tuple[UpstreamRef, ...],
        script: Path,
    ) -> None:
        state = self._workflow_state
        if state is None:
            raise WorkflowIoError("workflow state is unavailable")
        self._write_request(request_path, request)
        request_hash = canonical_fingerprint(request)
        project = self.store.project_path
        created = utc_now()
        state = state.begin_attempt(
            stage_id=operation,
            attempt_id=attempt_id,
            request_path=request_path.relative_to(project).as_posix(),
            output_path=output.relative_to(project).as_posix(),
            root_input_fingerprint=root_input,
            stage_input=stage_input,
            request_fingerprint=request_hash,
            upstream=upstream,
            created_at=created,
        )
        state = state.mark_running(attempt_id, started_at=utc_now())
        self._workflow_state = state
        self._save_workflow_state()
        for record in self.store.included_records():
            record.status = "queued"
        self.store.bagsChanged.emit()
        try:
            started = self._start_worker(
                operation,
                attempt_id,
                request_path,
                output,
                script,
            )
        except BaseException as error:
            self._mark_workflow_attempt_unsuccessful(
                "failed",
                "worker_start_error: {}".format(error),
                attempt_id=attempt_id,
            )
            self._restore_transient_record_statuses()
            self._operation = None
            self._operation_context = {}
            raise
        if not started:
            self._mark_workflow_attempt_unsuccessful(
                "failed", "worker_unavailable", attempt_id=attempt_id
            )
            self._restore_transient_record_statuses()
            self._operation = None
            self._operation_context = {}
            return
        self._operation_context.update(
            {
                "attempt_id": attempt_id,
                "root_input_fingerprint": root_input,
                "stage_input_fingerprint": stage_input,
                "request_fingerprint": request_hash,
            }
        )
        self.freshness_label.setText("RUNNING")
        self.freshness_label.setStyleSheet(
            "padding-left: 8px; color: #2e6b99; font-weight: 600;"
        )

    def _save_workflow_state(self) -> None:
        if self._workflow_state is None:
            raise WorkflowIoError("workflow state is unavailable")
        save_workflow(
            self.store.project_path,
            self.store.project_id,
            self._workflow_state,
        )

    def _complete_workflow_attempt(
        self, artifact_root: Path, manifest: object
    ) -> None:
        """Bind a validated complete bundle to the active workflow attempt."""

        state = self._workflow_state
        attempt_id = self._operation_context.get("attempt_id")
        if state is None or not isinstance(attempt_id, str):
            raise WorkflowIoError("active workflow attempt is unavailable")
        attempt = state.attempt(attempt_id)
        if self._operation is None:
            raise WorkflowError("active workflow stage is unavailable")
        reference = artifact_ref_from_validated_bundle(
            project_root=self.store.project_path,
            artifact_root=artifact_root,
            manifest=manifest,
            expected_stage_id=self._operation,
            expected_stage_input=attempt.stage_input_fingerprint,
            expected_request_fingerprint=attempt.request_fingerprint,
        )
        self._workflow_state = state.mark_complete(
            attempt_id, reference, finished_at=utc_now()
        )
        self._save_workflow_state()

    def _mark_workflow_attempt_unsuccessful(
        self,
        outcome: str,
        reason: str,
        *,
        attempt_id: str | None = None,
    ) -> None:
        """Persist failure/cancellation when the current operation is staged."""

        if self._operation not in {
            DIAGONAL_Q_STAGE_ID,
            STATIC_PARAMETERS_STAGE_ID,
        } and attempt_id is None:
            return
        state = self._workflow_state
        selected_id = (
            attempt_id
            if attempt_id is not None
            else self._operation_context.get("attempt_id")
        )
        if state is None or not isinstance(selected_id, str):
            return
        attempt = state.attempt(selected_id)
        if not attempt.status.active:
            return
        if outcome == "failed":
            state = state.mark_failed(
                selected_id, str(reason), finished_at=utc_now()
            )
        elif outcome == "cancelled":
            state = state.mark_cancelled(
                selected_id, str(reason), finished_at=utc_now()
            )
        else:
            raise ValueError("workflow outcome must be failed or cancelled")
        self._workflow_state = state
        self._save_workflow_state()


    def _choose_baseline_bag(self, records: tuple[BagRecord, ...]) -> str | None:
        snapshots = {
            json.dumps(record.controller_snapshot, allow_nan=False, sort_keys=True, separators=(",", ":"))
            for record in records
        }
        if len(snapshots) == 1:
            return records[0].bag_id
        labels = ["{} — {}".format(record.bag_id, record.display_name) for record in records]
        selected, accepted = QInputDialog.getItem(
            self, "Select baseline controller snapshot",
            "Selected bags have different PID snapshots. Choose the exact current baseline:",
            labels, 0, False,
        )
        if not accepted:
            return None
        return records[labels.index(selected)].bag_id

    def start_pid_evaluation(self, options: object) -> None:
        """Launch exact current/member posterior-predictive evaluation."""

        if self.runner.running:
            return
        if self.store.estimation_run is None:
            self._show_error(
                "Cannot evaluate PID proposal",
                "Complete fixed-Q parameter estimation before evaluating a PID proposal.",
            )
            return
        if not isinstance(options, PidEvaluationLaunchOptions):
            self._show_error(
                "Cannot evaluate PID proposal", "invalid GUI evaluation options"
            )
            return
        evaluation_id = "pid-eval-{}".format(uuid.uuid4().hex[:12])
        output = (
            self.store.project_path
            / "pid_proposals"
            / evaluation_id
            / "pid_proposal_evaluation"
        )
        request_path = output.parent / "request.json"
        try:
            request = build_pid_evaluation_request(
                self.store.estimation_run, evaluation_id, options
            )
            output.parent.mkdir(parents=True, exist_ok=False)
            self._write_request(request_path, request)
        except (OSError, TypeError, ValueError) as error:
            self._show_error("Cannot evaluate PID proposal", error)
            return
        started = self._start_worker(
            "pid_evaluation",
            evaluation_id,
            request_path,
            output,
            self.package_root / "scripts" / "grape_evaluate_pid_proposals.py",
        )
        if started and self._operation == "pid_evaluation":
            self._operation_context["source_run_id"] = str(
                self.store.estimation_run.manifest["run_id"]
            )
            self.tabs.setCurrentWidget(self.next_experiment)
            self.statusBar().showMessage(
                "Evaluating current PID, exact member {} proposal{}.".format(
                    options.source_member_id,
                    " and exact user candidate"
                    if options.user_candidate_values is not None
                    else "",
                )
            )

    def _start_worker(
        self,
        operation: str,
        run_id: str,
        request_path: Path,
        output: Path,
        script: Path,
    ) -> bool:
        worker = script
        if not worker.is_file():
            install_space_worker = (
                self.package_root.parent.parent
                / "lib"
                / "grape_param_estim"
                / worker.name
            )
            installed = shutil.which(worker.name)
            if install_space_worker.is_file():
                worker = install_space_worker
            elif installed is not None:
                worker = Path(installed)
            else:
                self._show_error("Worker is unavailable", worker)
                return False
        self._operation = operation
        self._operation_context = {"request": request_path, "output": output}
        self._log_path = self.store.project_path / "logs" / (run_id + ".stderr.log")
        self.runner.start(
            self.worker_python,
            (worker, "--request", request_path, "--output", output),
            output_directory=output,
            run_id=run_id,
            working_directory=self.package_root,
        )
        return True

    def _worker_finished(self, output_text: str) -> None:
        output = Path(output_text)
        continue_all = False
        try:
            if self._operation == "inspection":
                artifact = load_inspection(output)
                canonical = self.store.project_path / "inspection"
                backup = self.store.project_path / (
                    ".inspection-previous-" + uuid.uuid4().hex[:10]
                )
                if canonical.exists():
                    os.replace(canonical, backup)
                try:
                    os.replace(output, canonical)
                except BaseException:
                    if backup.exists() and not canonical.exists():
                        os.replace(backup, canonical)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
                artifact = replace(artifact, root=canonical)
                self.store.apply_inspection(artifact)
                pending_confirmation: list[str] = []
                for bag_id in self._operation_context.get("include_when_ready", ()):
                    record = self.store.get(str(bag_id))
                    if (
                        record is not None
                        and record.status == "ready"
                        and record.selected_interval is not None
                    ):
                        self.store.set_included(record.bag_id, True)
                    elif (
                        record is not None
                        and record.status == "needs_configuration_confirmation"
                        and record.selected_interval is not None
                    ):
                        pending_confirmation.append(record.bag_id)
                self.tabs.setCurrentWidget(self.bag_browser)
                if pending_confirmation:
                    self.statusBar().showMessage(
                        "Inspection completed; confirm a configuration group "
                        "to enable parameter estimation."
                    )
                    self._schedule_configuration_prompts(
                        tuple(pending_confirmation)
                    )
                else:
                    self.statusBar().showMessage("Rosbag inspection completed.")
            elif self._operation == "batch_estimation":
                run = load_batch_estimation_run(output)
                self.store.manifest["run_request_fingerprint"] = self._operation_context[
                    "project_request_fingerprint"
                ]
                self.store.apply_estimation(run)
                self._update_freshness(self.store.results_stale)
                self.tabs.setCurrentWidget(self.master_view)
                self.statusBar().showMessage("Sparse batch estimation completed.")
            elif self._operation == "pid_evaluation":
                evaluation = load_pid_evaluation(output)
                if self.store.estimation_run is None:
                    raise ProjectIoError(
                        "the source batch estimation run is no longer loaded"
                    )
                source_run_id = str(evaluation.manifest["source_run_id"])
                if source_run_id != self._operation_context.get("source_run_id"):
                    raise ProjectIoError(
                        "PID evaluation source_run_id does not match its request"
                    )
                if source_run_id != str(
                    self.store.estimation_run.manifest["run_id"]
                ):
                    raise ProjectIoError(
                        "PID evaluation source_run_id does not match the current run"
                    )
                self.store.apply_pid_evaluation(evaluation)
                self._restore_transient_record_statuses()
                self.tabs.setCurrentWidget(self.next_experiment)
                self.statusBar().showMessage(
                    "Posterior-predictive PID evaluation completed."
                )
            write_project_manifest(self.store.project_path, self.store.manifest)
        except (
            OSError,
            ValueError,
            GuiArtifactError,
            ProjectIoError,
            WorkflowError,
            WorkflowIoError,
        ) as error:
            continue_all = False
            try:
                self._mark_workflow_attempt_unsuccessful(
                    "failed", "invalid_worker_output: {}".format(error)
                )
            except (
                OSError,
                ValueError,
                WorkflowError,
                WorkflowIoError,
            ) as state_error:
                error = RuntimeError(
                    "{}; additionally could not persist workflow failure: {}".format(
                        error, state_error
                    )
                )
            self._restore_transient_record_statuses()
            self._show_error("Worker output is invalid", error)
        finally:
            self._operation = None
            self._operation_context = {}
            self.stage_label.setText("idle")
            self._finish_pending_close()
        if continue_all:
            QTimer.singleShot(0, self._continue_all_workflow)

    def _continue_all_workflow(self) -> None:
        """Launch the next stage after a successful ALL-mode boundary."""

        if (
            self.runner.running
            or self._close_after_worker
            or self._workflow_state is None
            or self._workflow_state.mode is not WorkflowMode.ALL
        ):
            return
        try:
            self._launch_next_workflow_stage()
        except (
            OSError,
            ValueError,
            GuiArtifactError,
            WorkflowError,
            WorkflowIoError,
        ) as error:
            self._show_error("Cannot continue estimation", error)

    def _worker_failed(self, message: str) -> None:
        workflow_error = None
        try:
            self._mark_workflow_attempt_unsuccessful(
                "failed", "worker_failed: {}".format(message)
            )
        except (
            OSError,
            ValueError,
            WorkflowError,
            WorkflowIoError,
        ) as error:
            workflow_error = error
        self._restore_transient_record_statuses()
        self._update_freshness(self.store.results_stale)
        if workflow_error is not None:
            message = (
                "{}; additionally could not persist workflow failure: {}"
            ).format(message, workflow_error)
        self._show_error("Estimator worker failed", message)
        self._operation = None
        self._operation_context = {}
        self._finish_pending_close()

    def _worker_cancelled(self) -> None:
        reason = (
            "application_closing"
            if self._close_after_worker
            else "user_requested"
        )
        workflow_error = None
        try:
            self._mark_workflow_attempt_unsuccessful("cancelled", reason)
        except (
            OSError,
            ValueError,
            WorkflowError,
            WorkflowIoError,
        ) as error:
            workflow_error = error
        self._restore_transient_record_statuses()
        message = "Worker cancelled; incomplete artifacts were not loaded."
        if workflow_error is not None:
            message = "{} Workflow state error: {}".format(
                message, workflow_error
            )
        self.statusBar().showMessage(message)
        self.stage_label.setText("cancelled")
        self.eta_label.setText("ETA —")
        self._update_freshness(self.store.results_stale)
        self._operation = None
        self._operation_context = {}
        self._finish_pending_close()

    def _restore_transient_record_statuses(self) -> None:
        transient = {"queued", "running", "writing", "inspection queued"}
        for record in self.store.records():
            if record.status not in transient:
                continue
            if record.inspection is None:
                restored_status = "awaiting inspection"
            else:
                restored_status = str(
                    record.inspection.get("status", "awaiting inspection")
                )
                fingerprint = record.inspection.get(
                    "configuration_fingerprint", {}
                )
                source_fingerprint = (
                    str(fingerprint.get("value", ""))
                    if isinstance(fingerprint, dict)
                    else ""
                )
                if (
                    restored_status == "needs_configuration_confirmation"
                    and record.configuration_confirmation.get(
                        "source_fingerprint"
                    )
                    == source_fingerprint
                    and record.configuration_confirmation.get(
                        "confirmed_fingerprint"
                    )
                ):
                    restored_status = "ready"
                if restored_status == "ready" and record.result is not None:
                    restored_status = "complete"
            record.status = restored_status
            if restored_status not in {"ready", "complete"}:
                record.included = False
            self.store.recordChanged.emit(record.bag_id)
        self.store._sync_manifest_inputs()
        self.store.bagsChanged.emit()

    def _finish_pending_close(self) -> None:
        if self._close_after_worker:
            self._close_after_worker = False
            QTimer.singleShot(0, self.close)

    def _worker_running_changed(self, running: bool) -> None:
        self.add_bags_action.setEnabled(not running)
        self.save_project_action.setEnabled(not running)
        self.load_project_action.setEnabled(not running)
        self.load_pid_action.setEnabled(not running)
        self.next_experiment.set_evaluation_running(running)
        self.stop_action.setEnabled(running)
        self._update_run_action()

    def _update_run_action(self) -> None:
        if self.runner.running:
            self.run_action.setEnabled(False)
            self.run_action.setToolTip("An estimator worker is running.")
            return
        if self._workflow_error is not None or self._workflow_state is None:
            self.run_action.setEnabled(False)
            self.run_action.setToolTip(
                "The staged workflow state cannot be loaded: {}".format(
                    self._workflow_error or "unknown error"
                )
            )
            return
        selected = self.store.included_records()
        if not selected:
            self.run_action.setEnabled(False)
            self.run_action.setToolTip(
                "Inspect a bag and confirm its configuration group first."
            )
            return
        if any(
            record.inspection is None
            or record.selected_interval is None
            or record.status not in {"ready", "complete"}
            or not record.configuration_fingerprint
            for record in selected
        ):
            self.run_action.setEnabled(False)
            self.run_action.setToolTip(
                "Every selected bag needs an interval and confirmed configuration group."
            )
            return
        fingerprints = {
            record.configuration_fingerprint for record in selected
        }
        if len(fingerprints) != 1:
            self.run_action.setEnabled(False)
            self.run_action.setToolTip(
                "Selected bags must share one confirmed configuration group."
            )
            return
        self.run_action.setEnabled(True)
        self.run_action.setToolTip(
            "Run sparse full-trajectory MAP, Laplace-EM, and optional MCMC."
        )

    def _update_progress(self, event: object) -> None:
        self.progress_bar.setValue(int(round(1000.0 * float(event.fraction))))
        detail = event.message or event.stage_label
        self.stage_label.setText("{} — {}".format(event.stage_label, detail))
        self.eta_label.setText(
            "ETA —" if event.eta_seconds is None
            else "ETA {}".format(self._format_duration(event.eta_seconds))
        )
        if event.bag_id:
            record = self.store.get(event.bag_id)
            if record is not None:
                record.status = (
                    "writing"
                    if event.stage_id == "writing_artifacts"
                    else "running"
                )
                self.store.recordChanged.emit(record.bag_id)

    def _worker_log(self, line: str) -> None:
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        if line:
            self.statusBar().showMessage(line, 5000)

    def save_project(self) -> None:
        if self.runner.running:
            return
        default = Path.home() / (self.store.project_id + ".zip")
        file_name, _filter = QFileDialog.getSaveFileName(
            self, "Save Project", str(default), "Grape project ZIP (*.zip)"
        )
        if not file_name:
            return
        try:
            self._write_gui_state()
            write_project_manifest(self.store.project_path, self.store.manifest)
            save_project_archive(self.store.project_path, file_name)
        except (OSError, ValueError, ProjectIoError) as error:
            self._show_error("Cannot save project", error)
            return
        self.statusBar().showMessage("Saved self-contained project to {}".format(file_name))

    def load_project(self) -> None:
        if self.runner.running:
            return
        file_name, _filter = QFileDialog.getOpenFileName(
            self, "Load Project", str(Path.home()), "Grape project ZIP (*.zip)"
        )
        if not file_name:
            return
        try:
            projects_root = self.package_root / "projects"
            project_path = load_project_archive(file_name, projects_root)
            manifest = read_project_manifest(project_path, verify_bags=True)
            records = self._records_from_manifest(project_path, manifest)
            self.store.replace_project(project_path, manifest, records)
            inspection_path = project_path / "inspection"
            if (inspection_path / "manifest.json").is_file():
                self.store.apply_inspection(load_inspection(inspection_path))
            run_id = manifest.get("current_estimation_run_id")
            if run_id:
                self.store.apply_estimation(
                    self._load_project_estimation(
                        project_path
                        / "runs"
                        / str(run_id)
                        / "estimation_run"
                    )
                )
            evaluation_id = manifest.get("current_pid_proposal_evaluation_id")
            if evaluation_id:
                self.store.apply_pid_evaluation(
                    load_pid_evaluation(project_path / "pid_proposals" / str(evaluation_id) / "pid_proposal_evaluation")
                )
            self._restore_gui_state(project_path / GUI_STATE_NAME)
            self.statusBar().showMessage("Loaded project {}".format(self.store.project_id))
        except (OSError, ValueError, GuiArtifactError, ProjectIoError) as error:
            self._show_error("Cannot load project", error)

    @staticmethod
    def _load_project_estimation(path: Path) -> BatchEstimationRun:
        """Load only the strict sparse-batch estimation schema."""

        return load_batch_estimation_run(path)

    def import_pid_evaluation(self) -> None:
        source_name = QFileDialog.getExistingDirectory(
            self, "Import PID proposal evaluation", str(Path.home())
        )
        if not source_name:
            return
        try:
            source = Path(source_name).resolve()
            evaluation = load_pid_evaluation(source)
            if self.store.estimation_run is None:
                raise ProjectIoError(
                    "load the source batch run before importing its PID evaluation"
                )
            if evaluation.manifest["source_run_id"] != self.store.estimation_run.manifest["run_id"]:
                raise ProjectIoError("PID evaluation source_run_id does not match the current run")
            evaluation_id = str(evaluation.manifest["evaluation_id"])
            canonical = self.store.project_path / "pid_proposals" / evaluation_id / "pid_proposal_evaluation"
            if source != canonical:
                if canonical.exists():
                    raise ProjectIoError("PID evaluation ID already exists in this project")
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, canonical)
                evaluation = load_pid_evaluation(canonical)
            self.store.apply_pid_evaluation(evaluation)
            write_project_manifest(self.store.project_path, self.store.manifest)
            self.tabs.setCurrentWidget(self.next_experiment)
        except (OSError, ValueError, GuiArtifactError, ProjectIoError) as error:
            self._show_error("Cannot import PID evaluation", error)

    @staticmethod
    def _records_from_manifest(project_path: Path, manifest: dict[str, object]) -> list[BagRecord]:
        intervals = manifest["intervals"]
        selected = set(manifest["selected_bag_ids"])
        fingerprints = manifest["configuration_fingerprints"]
        confirmations = manifest.get("configuration_confirmations", {})
        snapshots = manifest["controller_snapshots"]
        records = []
        for item in manifest["bags"]:
            interval = intervals.get(item["bag_id"])
            records.append(
                BagRecord(
                    bag_id=item["bag_id"],
                    path=project_path / item["relative_path"],
                    source_path=Path(item["source_path"]),
                    sha256=item["sha256"],
                    included=item["bag_id"] in selected,
                    auto_interval=None if interval is None else tuple(interval["auto"]),
                    selected_interval=None if interval is None else tuple(interval["selected"]),
                    interval_state="AUTO" if interval is None else interval["state"],
                    configuration_fingerprint=fingerprints.get(item["bag_id"], ""),
                    configuration_confirmation=confirmations.get(
                        item["bag_id"], {}
                    ),
                    controller_snapshot=snapshots.get(item["bag_id"]) or {},
                )
            )
        return records

    def _write_gui_state(self) -> None:
        payload = {
            "schema": "grape-param-estim/gui-state/v2",
            "current_bag_id": self.store.current_bag_id,
            "selected_sample_id": self.store.selected_sample_id,
            "selected_mode_id": self.store.selected_mode_id,
            "selected_pid_proposal_id": self.store.selected_pid_proposal_id,
            "bags": {
                record.bag_id: {
                    "current_time": record.current_time,
                    "view_range": list(record.view_range),
                }
                for record in self.store.records()
            },
        }
        self._write_request(self.store.project_path / GUI_STATE_NAME, payload)

    def _restore_gui_state(self, path: Path) -> None:
        if not path.is_file():
            return
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "grape-param-estim/gui-state/v2":
            raise ProjectIoError("unsupported GUI state schema")
        for bag_id, state in value.get("bags", {}).items():
            record = self.store.get(bag_id)
            if record is not None:
                record.current_time = float(state["current_time"])
                record.view_range = tuple(float(item) for item in state["view_range"])
        current = value.get("current_bag_id")
        if current:
            self.store.set_current(str(current))
        sample = value.get("selected_sample_id")
        if sample is not None:
            self.store.set_selected_sample(str(sample))
        self.store.set_selected_mode(value.get("selected_mode_id"))
        self.store.set_selected_pid_proposal(value.get("selected_pid_proposal_id"))

    @staticmethod
    def _write_request(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _update_freshness(self, stale: bool) -> None:
        if self.store.manifest.get("run_request_fingerprint") is None:
            self.freshness_label.setText("NOT ESTIMATED")
            color = "#666"
        elif stale:
            self.freshness_label.setText("STALE")
            color = "#9a5c00"
        else:
            self.freshness_label.setText("UP TO DATE")
            color = "#217543"
        self.freshness_label.setStyleSheet("padding-left: 8px; color: {}; font-weight: 600;".format(color))

    def _update_project_title(self) -> None:
        self.setWindowTitle(
            "Grape sparse batch parameter estimation — {}".format(
                self.store.project_id
            )
        )

    def _open_bag_from_master(self, bag_id: str) -> None:
        self.store.set_current(bag_id)
        self.tabs.setCurrentWidget(self.bag_browser)

    def _update_play_action(self, playing: bool) -> None:
        self.play_action.setText("❚❚" if playing else "▶")

    def _go_to_start(self) -> None:
        record = self.store.current_record()
        if record is not None and record.data is not None:
            self.bag_browser.time_state.set_current_time(float(record.data.time[0]))

    def _go_to_end(self) -> None:
        record = self.store.current_record()
        if record is not None and record.data is not None:
            self.bag_browser.time_state.set_current_time(float(record.data.time[-1]))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        if seconds < 60.0:
            return "{:.0f} s".format(seconds)
        minutes = int(seconds // 60.0)
        return "{}m {:02d}s".format(minutes, int(round(seconds - 60.0 * minutes)))

    def _show_error(self, title: str, error: object) -> None:
        QMessageBox.critical(self, title, str(error))
        self.statusBar().showMessage("{}: {}".format(title, error), 10000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._cancel_configuration_prompts()
        if self.runner.running:
            self._close_after_worker = True
            self.runner.request_cancel("application_closing")
            event.ignore()
            return
        self.bag_browser.close_scene()
        self.next_experiment.comparison_scene.close_scene()
        super().closeEvent(event)


__all__ = ["MainWindow"]
