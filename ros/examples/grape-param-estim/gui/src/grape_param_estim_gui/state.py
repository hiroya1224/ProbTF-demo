"""Qt application state backed only by inspection and estimator artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping
try:
    from PySide6.QtCore import QObject, Signal
except ImportError:  # Qt-free state/loader tests; the application still requires PySide6.
    class _BoundSignal:
        def __init__(self) -> None:
            self._callbacks: list[Any] = []

        def connect(self, callback: Any) -> None:
            self._callbacks.append(callback)

        def emit(self, *arguments: Any) -> None:
            for callback in tuple(self._callbacks):
                callback(*arguments)

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *_types: Any) -> None:
            self._name = ""

        def __set_name__(self, _owner: type, name: str) -> None:
            self._name = "_headless_signal_" + name

        def __get__(self, instance: Any, _owner: type) -> Any:
            if instance is None:
                return self
            signal = instance.__dict__.get(self._name)
            if signal is None:
                signal = _BoundSignal()
                instance.__dict__[self._name] = signal
            return signal

    class QObject:  # type: ignore[no-redef]
        def __init__(self, _parent: Any = None) -> None:
            pass

from .artifact_loader import (
    BagEstimationResult,
    BatchEstimationRun,
    FlightResult,
    InspectionArtifact,
    McmcPosterior,
    PidProposalEvaluation,
)
from .project_io import freshness_fingerprint, result_is_fresh, utc_now


MANUAL_CONFIGURATION_GROUP_SCHEMA = (
    "grape-param-estim/manual-configuration-group/v1"
)


@dataclass
class BagRecord:
    bag_id: str
    path: Path
    source_path: Path
    sha256: str
    inspection: Mapping[str, Any] | None = None
    preview: FlightResult | None = None
    result: BagEstimationResult | None = None
    included: bool = False
    auto_interval: tuple[float, float] | None = None
    selected_interval: tuple[float, float] | None = None
    interval_state: str = "AUTO"
    status: str = "awaiting inspection"
    configuration_fingerprint: str = ""
    configuration_provenance: Mapping[str, str] = field(default_factory=dict)
    configuration_confirmation: Mapping[str, str] = field(default_factory=dict)
    controller_snapshot: Mapping[str, Any] = field(default_factory=dict)
    current_time: float = 0.0
    view_range: tuple[float, float] = (0.0, 1.0)

    @property
    def display_name(self) -> str:
        return self.path.name

    @property
    def data(self) -> FlightResult | BagEstimationResult | None:
        """The most informative real array set currently available."""

        return self.result if self.result is not None else self.preview

    @property
    def auto_range(self) -> tuple[float, float]:
        return self.auto_interval or (0.0, 0.0)

    @auto_range.setter
    def auto_range(self, value: tuple[float, float]) -> None:
        self.auto_interval = value

    @property
    def selected_range(self) -> tuple[float, float]:
        return self.selected_interval or self.auto_range

    @selected_range.setter
    def selected_range(self, value: tuple[float, float]) -> None:
        self.selected_interval = value

    @property
    def configuration_group(self) -> str:
        group_id = self.configuration_confirmation.get("group_id")
        if group_id:
            return "manual: {}".format(group_id)
        return self.configuration_fingerprint or "unconfirmed"


@dataclass(frozen=True)
class ProjectState:
    project_id: str
    project_path: Path
    project_schema: str
    project_loader_id: str
    bag_records: tuple[BagRecord, ...]
    selected_bag_ids: tuple[str, ...]
    current_bag_id: str | None
    selected_sample_id: str | None
    selected_mode_id: str | None
    selected_pid_proposal_id: str | None
    estimation_run_path: Path | None
    pid_proposal_evaluation_path: Path | None
    results_stale: bool


class TimeState(QObject):
    currentTimeChanged = Signal(float)
    estimationRangeChanged = Signal(float, float)
    viewRangeChanged = Signal(float, float)
    playingChanged = Signal(bool)
    playbackSpeedChanged = Signal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.current_time = 0.0
        self.estimation_start = 0.0
        self.estimation_end = 1.0
        self.view_start = 0.0
        self.view_end = 1.0
        self.playing = False
        self.playback_speed = 1.0

    def set_current_time(self, value: float) -> None:
        value = float(value)
        if abs(value - self.current_time) < 1.0e-12:
            return
        self.current_time = value
        self.currentTimeChanged.emit(value)

    def set_estimation_range(self, start: float, end: float) -> None:
        start, end = sorted((float(start), float(end)))
        if (start, end) == (self.estimation_start, self.estimation_end):
            return
        self.estimation_start, self.estimation_end = start, end
        self.estimationRangeChanged.emit(start, end)

    def set_view_range(self, start: float, end: float) -> None:
        start, end = sorted((float(start), float(end)))
        if (start, end) == (self.view_start, self.view_end):
            return
        self.view_start, self.view_end = start, end
        self.viewRangeChanged.emit(start, end)

    def set_playing(self, value: bool) -> None:
        value = bool(value)
        if value != self.playing:
            self.playing = value
            self.playingChanged.emit(value)

    def set_playback_speed(self, value: float) -> None:
        value = max(float(value), 0.01)
        if abs(value - self.playback_speed) >= 1.0e-12:
            self.playback_speed = value
            self.playbackSpeedChanged.emit(value)


class ProjectStore(QObject):
    bagsChanged = Signal()
    currentBagChanged = Signal(object)
    currentBagIdChanged = Signal(str)
    selectedSampleChanged = Signal(object)
    selectedModeChanged = Signal(object)
    selectedPidProposalChanged = Signal(object)
    recordChanged = Signal(str)
    posteriorChanged = Signal(object)
    pidEvaluationChanged = Signal(object)
    freshnessChanged = Signal(bool)
    projectChanged = Signal()

    def __init__(
        self,
        project_path: str | Path,
        manifest: dict[str, Any],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_path = Path(project_path).resolve()
        self.manifest = manifest
        self._records: list[BagRecord] = []
        self._current_bag_id: str | None = None
        self._selected_sample_id: str | None = None
        self._selected_mode_id: str | None = None
        self._selected_pid_proposal_id: str | None = None
        self.estimation_run: BatchEstimationRun | None = None
        self.pid_evaluation: PidProposalEvaluation | None = None
        self._results_stale = bool(manifest.get("run_request_fingerprint")) and not result_is_fresh(manifest)

    @property
    def project_id(self) -> str:
        return str(self.manifest["project_id"])

    @property
    def selected_sample_id(self) -> str | None:
        return self._selected_sample_id

    @property
    def selected_mode_id(self) -> str | None:
        return self._selected_mode_id

    @property
    def selected_pid_proposal_id(self) -> str | None:
        return self._selected_pid_proposal_id

    @property
    def current_bag_id(self) -> str | None:
        return self._current_bag_id

    @property
    def results_stale(self) -> bool:
        return self._results_stale

    @property
    def posterior_samples(self) -> McmcPosterior | None:
        return None if self.estimation_run is None else self.estimation_run.mcmc

    def records(self) -> tuple[BagRecord, ...]:
        return tuple(self._records)

    def replace_project(
        self,
        project_path: str | Path,
        manifest: dict[str, Any],
        records: Iterable[BagRecord],
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.manifest = manifest
        self._records = list(records)
        self._current_bag_id = None
        self._selected_sample_id = None
        self._selected_mode_id = None
        self._selected_pid_proposal_id = None
        self.estimation_run = None
        self.pid_evaluation = None
        self._results_stale = bool(manifest.get("run_request_fingerprint")) and not result_is_fresh(manifest)
        self.projectChanged.emit()
        self.posteriorChanged.emit(None)
        self.pidEvaluationChanged.emit(None)
        self.bagsChanged.emit()
        if self._records:
            self.set_current(self._records[0].bag_id)
        else:
            self.currentBagIdChanged.emit("")
            self.currentBagChanged.emit(None)
        self.selectedSampleChanged.emit(None)
        self.freshnessChanged.emit(self._results_stale)

    def included_records(self) -> tuple[BagRecord, ...]:
        return tuple(record for record in self._records if record.included)

    def current_record(self) -> BagRecord | None:
        return None if self._current_bag_id is None else self.get(self._current_bag_id)

    def get(self, bag_id: str) -> BagRecord | None:
        return next((record for record in self._records if record.bag_id == bag_id), None)

    def add(self, record: BagRecord) -> None:
        if self.get(record.bag_id) is not None:
            raise ValueError("duplicate bag ID {}".format(record.bag_id))
        self._records.append(record)
        self._sync_manifest_inputs()
        self.bagsChanged.emit()
        if self._current_bag_id is None:
            self.set_current(record.bag_id)

    def extend(self, records: Iterable[BagRecord]) -> None:
        for record in records:
            self.add(record)

    def remove(self, bag_id: str) -> None:
        self._records = [record for record in self._records if record.bag_id != bag_id]
        self.manifest["bags"] = [
            item for item in self.manifest.get("bags", [])
            if item.get("bag_id") != bag_id
        ]
        if self._current_bag_id == bag_id:
            self._current_bag_id = None
            if self._records:
                self.set_current(self._records[0].bag_id)
            else:
                self.currentBagIdChanged.emit("")
                self.currentBagChanged.emit(None)
        self._sync_manifest_inputs()
        self.bagsChanged.emit()

    def set_current(self, bag_id: str) -> None:
        record = self.get(bag_id)
        if record is None or bag_id == self._current_bag_id:
            return
        self._current_bag_id = bag_id
        self.currentBagIdChanged.emit(bag_id)
        self.currentBagChanged.emit(record)

    def set_included(self, bag_id: str, included: bool) -> None:
        record = self.get(bag_id)
        if record is None or record.included == bool(included):
            return
        record.included = bool(included)
        self._sync_manifest_inputs()
        self.recordChanged.emit(bag_id)
        self.bagsChanged.emit()

    def confirm_configuration_group(self, bag_id: str, group_id: str) -> None:
        """Record an explicit human grouping without inventing provenance."""

        record = self.get(bag_id)
        if record is None:
            raise ValueError("unknown bag ID {}".format(bag_id))
        if record.inspection is None or record.selected_interval is None:
            raise ValueError("inspect the bag before confirming its group")
        if str(record.inspection.get("status", "")) != (
            "needs_configuration_confirmation"
        ):
            raise ValueError(
                "manual grouping cannot override a blocked inspection"
            )
        fingerprint = record.inspection.get("configuration_fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise ValueError("inspection configuration fingerprint is invalid")
        if bool(fingerprint.get("complete", False)):
            raise ValueError("configuration provenance is already complete")
        source_fingerprint = str(fingerprint.get("value", ""))
        if not source_fingerprint:
            raise ValueError("inspection configuration fingerprint is empty")
        identifier = str(group_id).strip()
        if not identifier:
            raise ValueError("configuration group ID cannot be empty")
        if len(identifier) > 160 or any(ord(character) < 32 for character in identifier):
            raise ValueError(
                "configuration group ID must be at most 160 printable characters"
            )
        digest = hashlib.sha256(
            (MANUAL_CONFIGURATION_GROUP_SCHEMA + "\0" + identifier).encode(
                "utf-8"
            )
        ).hexdigest()
        confirmed_fingerprint = "manual-group:sha256:" + digest
        record.configuration_confirmation = {
            "schema": MANUAL_CONFIGURATION_GROUP_SCHEMA,
            "group_id": identifier,
            "source_fingerprint": source_fingerprint,
            "confirmed_fingerprint": confirmed_fingerprint,
            "confirmed_at": utc_now(),
        }
        record.configuration_fingerprint = confirmed_fingerprint
        record.status = "ready"
        record.included = True
        self._sync_manifest_inputs()
        self.recordChanged.emit(bag_id)
        self.bagsChanged.emit()

    def update_interval(
        self,
        bag_id: str,
        selected_range: tuple[float, float],
        state: str = "MODIFIED",
    ) -> None:
        record = self.get(bag_id)
        if record is None:
            return
        start, end = sorted((float(selected_range[0]), float(selected_range[1])))
        if start >= end:
            raise ValueError("selected interval must have positive duration")
        if state not in {"AUTO", "MODIFIED", "LOCKED"}:
            raise ValueError("unknown interval state")
        record.selected_interval = (start, end)
        record.interval_state = state
        self._sync_manifest_inputs()
        self.recordChanged.emit(bag_id)
        self.bagsChanged.emit()

    def restore_auto_interval(self, bag_id: str) -> None:
        record = self.get(bag_id)
        if record is None or record.auto_interval is None:
            return
        self.update_interval(bag_id, record.auto_interval, state="AUTO")

    def apply_inspection(self, artifact: InspectionArtifact) -> None:
        for bag_id, inspection in artifact.inspections.items():
            record = self.get(bag_id)
            if record is None:
                raise ValueError("inspection contains unregistered bag {}".format(bag_id))
            if str(inspection["bag_sha256"]) != record.sha256:
                raise ValueError("inspection SHA256 differs for {}".format(bag_id))
            recommendation = inspection.get("recommended_interval")
            if recommendation is None:
                recommendation = inspection.get("recommendation")
            first_inspection = record.inspection is None
            previous_status = record.status
            record.inspection = inspection
            record.preview = artifact.previews[bag_id]
            if recommendation is not None:
                if not isinstance(recommendation, Mapping):
                    raise ValueError("inspection recommended interval is invalid")
                interval = recommendation.get("interval", recommendation)
                if not isinstance(interval, Mapping):
                    raise ValueError("inspection recommended interval is invalid")
                start = float(
                    interval.get(
                        "start_local_time",
                        interval.get("start", interval.get("record_start")),
                    )
                )
                end = float(
                    interval.get(
                        "end_local_time",
                        interval.get("end", interval.get("record_end")),
                    )
                )
                record.auto_interval = (start, end)
                if record.selected_interval is None or record.interval_state == "AUTO":
                    record.selected_interval = record.auto_interval
                    record.interval_state = "AUTO"
            fingerprint = inspection.get("configuration_fingerprint", {})
            inspection_fingerprint = (
                str(fingerprint.get("value", ""))
                if isinstance(fingerprint, Mapping)
                else str(fingerprint)
            )
            record.configuration_fingerprint = inspection_fingerprint
            if isinstance(fingerprint, Mapping):
                record.configuration_provenance = {
                    str(key): str(value)
                    for key, value in fingerprint.get("components", {}).items()
                }
            record.controller_snapshot = inspection.get("controller_snapshot") or {}
            record.status = str(inspection.get("status", "inspected"))
            confirmation = dict(record.configuration_confirmation)
            if isinstance(fingerprint, Mapping) and bool(
                fingerprint.get("complete", False)
            ):
                record.configuration_confirmation = {}
            elif (
                record.status == "needs_configuration_confirmation"
                and confirmation.get("schema")
                == MANUAL_CONFIGURATION_GROUP_SCHEMA
                and confirmation.get("source_fingerprint")
                == inspection_fingerprint
                and confirmation.get("confirmed_fingerprint")
            ):
                record.configuration_fingerprint = str(
                    confirmation["confirmed_fingerprint"]
                )
                record.status = "ready"
            else:
                record.configuration_confirmation = {}
            if record.status != "ready":
                record.included = False
            if (
                (first_inspection or previous_status == "needs_configuration_confirmation")
                and record.status == "ready"
                and record.selected_interval is not None
            ):
                record.included = True
            record.current_time = float(record.preview.time[0])
            record.view_range = (float(record.preview.time[0]), float(record.preview.time[-1]))
            self.recordChanged.emit(bag_id)
        self._sync_manifest_inputs()
        self.bagsChanged.emit()

    def apply_estimation(self, run: BatchEstimationRun) -> None:
        """Attach a validated batch run and synchronize its sample identity."""

        project_fingerprint = run.request_fingerprint
        current_fingerprint = self.request_fingerprint()
        if project_fingerprint != current_fingerprint:
            raise ValueError(
                "batch run request_fingerprint does not match "
                "the current project inputs"
            )
        unknown_bag_ids = sorted(
            bag_id for bag_id in run.bags if self.get(bag_id) is None
        )
        if unknown_bag_ids:
            raise ValueError(
                "run contains unregistered bags: {}".format(
                    ", ".join(unknown_bag_ids)
                )
            )
        for record in self._records:
            record.result = None
        for bag_id, result in run.bags.items():
            record = self.get(bag_id)
            assert record is not None
            record.result = result
            record.status = "complete"
            self.recordChanged.emit(bag_id)
        self.estimation_run = run
        run_id = run.run_id
        if (
            self.pid_evaluation is not None
            and str(self.pid_evaluation.manifest.get("estimation_run_id", ""))
            != run_id
        ):
            self.pid_evaluation = None
            self._selected_pid_proposal_id = None
            self.manifest["current_pid_proposal_evaluation_id"] = None
            self.pidEvaluationChanged.emit(None)
            self.selectedPidProposalChanged.emit(None)
        self._selected_sample_id = (
            None if not run.sample_ids else run.sample_ids[0]
        )
        if run.mcmc is None or run.mcmc.source_mode_id.size == 0:
            self._selected_mode_id = None
        else:
            self._selected_mode_id = str(run.mcmc.source_mode_id[0])
        self.manifest["current_estimation_run_id"] = run.run_id
        self._refresh_stale()
        self.posteriorChanged.emit(run)
        self.selectedSampleChanged.emit(self._selected_sample_id)
        self.selectedModeChanged.emit(self._selected_mode_id)
        self.bagsChanged.emit()

    def apply_pid_evaluation(self, evaluation: PidProposalEvaluation) -> None:
        if self.estimation_run is None:
            raise ValueError(
                "a PID evaluation requires its source batch estimation run"
            )
        source_run_id = str(evaluation.manifest.get("estimation_run_id", ""))
        current_run_id = self.estimation_run.run_id
        if not source_run_id or source_run_id != current_run_id:
            raise ValueError(
                "PID evaluation estimation_run_id does not match the current "
                "batch estimation run"
            )
        self.pid_evaluation = evaluation
        self.manifest["current_pid_proposal_evaluation_id"] = evaluation.manifest.get("evaluation_id")
        candidates = evaluation.summary.get("candidate_id")
        if candidates is not None and len(candidates):
            self._selected_pid_proposal_id = str(candidates[0])
        self.pidEvaluationChanged.emit(evaluation)
        self.selectedPidProposalChanged.emit(self._selected_pid_proposal_id)

    def set_selected_sample(self, sample_id: str | None) -> None:
        if sample_id is None:
            selected = None
        else:
            posterior = self.posterior_samples
            selected = str(sample_id)
            if (
                posterior is None
                or selected not in set(posterior.sample_id.tolist())
            ):
                return
        if selected != self._selected_sample_id:
            self._selected_sample_id = selected
            self.selectedSampleChanged.emit(selected)

    def set_selected_mode(self, mode_id: str | None) -> None:
        if mode_id != self._selected_mode_id:
            self._selected_mode_id = mode_id
            self.selectedModeChanged.emit(mode_id)

    def set_selected_pid_proposal(self, candidate_id: str | None) -> None:
        if candidate_id != self._selected_pid_proposal_id:
            self._selected_pid_proposal_id = candidate_id
            self.selectedPidProposalChanged.emit(candidate_id)

    def set_estimator_settings(self, settings: Mapping[str, Any]) -> None:
        self.manifest["estimator_settings"] = dict(settings)
        self._sync_manifest_inputs()

    def request_fingerprint(self) -> str:
        self._sync_manifest_inputs()
        return freshness_fingerprint(self.manifest)

    def snapshot(self) -> ProjectState:
        return ProjectState(
            project_id=self.project_id,
            project_path=self.project_path,
            project_schema=str(self.manifest["schema"]),
            project_loader_id=str(self.manifest["loader"]["id"]),
            bag_records=self.records(),
            selected_bag_ids=tuple(record.bag_id for record in self.included_records()),
            current_bag_id=self.current_bag_id,
            selected_sample_id=self.selected_sample_id,
            selected_mode_id=self.selected_mode_id,
            selected_pid_proposal_id=self.selected_pid_proposal_id,
            estimation_run_path=(
                None if self.estimation_run is None else self.estimation_run.root
            ),
            pid_proposal_evaluation_path=None if self.pid_evaluation is None else self.pid_evaluation.root,
            results_stale=self.results_stale,
        )

    def _sync_manifest_inputs(self) -> None:
        self.manifest["selected_bag_ids"] = [
            record.bag_id for record in self._records if record.included
        ]
        self.manifest["intervals"] = {
            record.bag_id: {
                "auto": list(record.auto_range),
                "selected": list(record.selected_range),
                "state": record.interval_state,
            }
            for record in self._records
            if record.auto_interval is not None and record.selected_interval is not None
        }
        self.manifest["configuration_fingerprints"] = {
            record.bag_id: record.configuration_fingerprint for record in self._records
        }
        self.manifest["configuration_confirmations"] = {
            record.bag_id: dict(record.configuration_confirmation)
            for record in self._records
            if record.configuration_confirmation
        }
        self.manifest["controller_snapshots"] = {
            record.bag_id: dict(record.controller_snapshot) for record in self._records
        }
        self._refresh_stale()

    def _refresh_stale(self) -> None:
        stale = bool(self.manifest.get("run_request_fingerprint")) and not result_is_fresh(self.manifest)
        if stale != self._results_stale:
            self._results_stale = stale
            self.manifest["result_freshness"] = "STALE" if stale else "UP_TO_DATE"
            self.freshnessChanged.emit(stale)


__all__ = [
    "BagRecord",
    "MANUAL_CONFIGURATION_GROUP_SCHEMA",
    "ProjectState",
    "ProjectStore",
    "TimeState",
]
