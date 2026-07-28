"""Capability and exactness gates for controller replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .contracts import (
    ControllerBackend,
    ControllerBackendIdentity,
    ControllerCoreInput,
    ControllerCoreOutput,
    ControllerCoreState,
    ControllerFidelity,
    ControllerTask,
    FIDELITY_PC_EXACT,
    FIDELITY_PC_MCU_EXACT,
    SerializableContract,
    expand_capabilities,
    normalize_fidelity,
    required_capabilities_for_fidelity,
    required_capabilities_for_task,
)
from .snapshot import ControllerSnapshot


@dataclass(frozen=True)
class CapabilityCheck(SerializableContract):
    passed: bool
    required: Tuple[str, ...]
    available: Tuple[str, ...]
    missing: Tuple[str, ...]
    fidelity: Optional[str] = None
    task: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("CapabilityCheck.passed must be a built-in bool")
        object.__setattr__(
            self, "required", tuple(str(item) for item in self.required)
        )
        object.__setattr__(
            self, "available", tuple(str(item) for item in self.available)
        )
        object.__setattr__(
            self, "missing", tuple(str(item) for item in self.missing)
        )
        if self.fidelity is not None:
            object.__setattr__(
                self, "fidelity", normalize_fidelity(self.fidelity)
            )
        if self.task is not None:
            task = (
                self.task.value
                if isinstance(self.task, ControllerTask)
                else str(self.task)
            )
            object.__setattr__(self, "task", task)


def check_capabilities(
    identity_or_capabilities: Any,
    *,
    fidelity: Optional[Any] = None,
    task: Optional[Any] = None,
) -> CapabilityCheck:
    """Check a backend declaration against one task or fidelity boundary."""

    if fidelity is None and task is None:
        raise ValueError("a fidelity or task capability check is required")
    identity = (
        identity_or_capabilities
        if isinstance(identity_or_capabilities, ControllerBackendIdentity)
        else None
    )
    raw = (
        identity.capabilities
        if identity is not None
        else tuple(identity_or_capabilities)
    )
    available = expand_capabilities(raw)
    if task is not None:
        required = required_capabilities_for_task(task, fidelity=fidelity)
    else:
        required = required_capabilities_for_fidelity(fidelity)
    missing = tuple(
        item for item in required if item not in set(available)
    )
    return CapabilityCheck(
        passed=not missing,
        required=tuple(required),
        available=available,
        missing=missing,
        fidelity=None if fidelity is None else normalize_fidelity(fidelity),
        task=None
        if task is None
        else (
            task.value if isinstance(task, ControllerTask) else str(task)
        ),
    )


def require_capabilities(
    identity_or_capabilities: Any,
    *,
    fidelity: Optional[Any] = None,
    task: Optional[Any] = None,
) -> CapabilityCheck:
    report = check_capabilities(
        identity_or_capabilities, fidelity=fidelity, task=task
    )
    if not report.passed:
        raise ValueError(
            "controller backend lacks required capabilities: {}".format(
                ", ".join(report.missing)
            )
        )
    return report


@dataclass(frozen=True)
class FactualControllerReplay(SerializableContract):
    snapshot_sha256: str
    backend_identity: ControllerBackendIdentity
    initial_state: ControllerCoreState
    inputs: Tuple[ControllerCoreInput, ...]
    outputs: Tuple[ControllerCoreOutput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_identity, ControllerBackendIdentity):
            raise TypeError("backend_identity must be ControllerBackendIdentity")
        if not isinstance(self.initial_state, ControllerCoreState):
            raise TypeError("initial_state must be ControllerCoreState")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not all(isinstance(item, ControllerCoreInput) for item in inputs):
            raise TypeError("factual replay inputs must be ControllerCoreInput")
        if not all(isinstance(item, ControllerCoreOutput) for item in outputs):
            raise TypeError("factual replay outputs must be ControllerCoreOutput")
        if len(inputs) != len(outputs):
            raise ValueError("factual replay inputs and outputs are not aligned")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


def run_factual_controller_replay(
    backend: ControllerBackend,
    snapshot: ControllerSnapshot,
    initial_state: ControllerCoreState,
    inputs: Sequence[ControllerCoreInput],
) -> FactualControllerReplay:
    """Replay teacher-forced controller inputs without a plant argument.

    The deliberately narrow signature is an architecture constraint: plant or
    actuator hypotheses cannot enter factual controller replay.
    """

    identity = getattr(backend, "identity", None)
    if not isinstance(identity, ControllerBackendIdentity):
        raise TypeError("controller backend lacks a v2 identity")
    if not isinstance(snapshot, ControllerSnapshot):
        raise TypeError("snapshot must be ControllerSnapshot")
    if not isinstance(initial_state, ControllerCoreState):
        raise TypeError("initial_state must be ControllerCoreState")
    items = tuple(inputs)
    if not all(isinstance(item, ControllerCoreInput) for item in items):
        raise TypeError("inputs must contain ControllerCoreInput values")
    backend.reset(snapshot, initial_state)
    outputs = tuple(backend.step(item) for item in items)
    if not all(isinstance(item, ControllerCoreOutput) for item in outputs):
        raise TypeError("controller backend returned a non-contract output")
    return FactualControllerReplay(
        snapshot_sha256=snapshot.content_sha256,
        backend_identity=identity,
        initial_state=initial_state,
        inputs=items,
        outputs=outputs,
    )


@dataclass(frozen=True)
class ExactClosedLoopGateReport(SerializableContract):
    passed: bool
    status: str
    reasons: Tuple[str, ...]
    identity: Optional[ControllerBackendIdentity]
    capability_check: Optional[CapabilityCheck]
    factual_replay_passed: bool
    required_fidelity: str
    conformance_report: Optional[Any] = None
    factual_evidence_sha256: Optional[str] = None
    schema: str = "grape_exact_closed_loop_gate/v2"

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("gate passed must be a built-in bool")
        if type(self.factual_replay_passed) is not bool:
            raise TypeError(
                "factual_replay_passed must be a built-in bool"
            )
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )
        object.__setattr__(
            self,
            "required_fidelity",
            normalize_fidelity(self.required_fidelity),
        )
        if self.schema != "grape_exact_closed_loop_gate/v2":
            raise ValueError("unsupported exact closed-loop gate schema")
        from grape_param_estim.alternative_backends import (
            ExactOracleConformanceReport,
        )

        report = self.conformance_report
        if report is not None and not isinstance(
            report, ExactOracleConformanceReport
        ):
            raise TypeError(
                "conformance_report must be ExactOracleConformanceReport"
            )
        evidence_hash = (
            None if report is None else report.evidence_sha256
        )
        if (
            self.factual_evidence_sha256 is not None
            and str(self.factual_evidence_sha256) != evidence_hash
        ):
            raise ValueError("factual conformance evidence hash mismatch")
        report_identity = (
            None if report is None else _coerce_identity(report.identity)
        )
        if self.factual_replay_passed and (
            report is None
            or not report.content_is_valid()
            or not report.passed
            or report.status != "PASS"
            or report_identity is None
            or self.identity != report_identity
        ):
            raise ValueError(
                "passing factual replay state lacks bound conformance evidence"
            )
        if self.passed and (
            self.status != "PASS"
            or not self.factual_replay_passed
            or self.identity is None
            or self.identity.is_exact is not True
            or self.capability_check is None
            or not self.capability_check.passed
            or report is None
            or not _fidelity_satisfies(
                report.fidelity, self.required_fidelity
            )
        ):
            raise ValueError(
                "passing exact gate lacks complete bound exact evidence"
            )
        if not self.passed and self.status == "PASS":
            raise ValueError("a rejected exact gate cannot say PASS")
        object.__setattr__(
            self, "factual_evidence_sha256", evidence_hash
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "status": self.status,
            "reasons": list(self.reasons),
            "identity": (
                None
                if self.identity is None
                else self.identity.to_mapping()
            ),
            "capability_check": (
                None
                if self.capability_check is None
                else self.capability_check.to_mapping()
            ),
            "factual_replay_passed": self.factual_replay_passed,
            "required_fidelity": self.required_fidelity,
            "factual_evidence_sha256": (
                self.factual_evidence_sha256
            ),
            "conformance_report": (
                None
                if self.conformance_report is None
                else self.conformance_report.to_mapping()
            ),
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactClosedLoopGateReport":
        from grape_param_estim.alternative_backends import (
            ExactOracleConformanceReport,
        )

        if not isinstance(values, Mapping):
            raise TypeError("exact closed-loop gate must be a mapping")
        try:
            raw_identity = values.get("identity")
            raw_capability = values.get("capability_check")
            raw_report = values.get("conformance_report")
            return cls(
                passed=values["passed"],
                status=values["status"],
                reasons=tuple(values["reasons"]),
                identity=(
                    None
                    if raw_identity is None
                    else ControllerBackendIdentity.from_mapping(
                        raw_identity
                    )
                ),
                capability_check=(
                    None
                    if raw_capability is None
                    else CapabilityCheck(
                        passed=raw_capability["passed"],
                        required=tuple(raw_capability["required"]),
                        available=tuple(raw_capability["available"]),
                        missing=tuple(raw_capability["missing"]),
                        fidelity=raw_capability.get("fidelity"),
                        task=raw_capability.get("task"),
                    )
                ),
                factual_replay_passed=values[
                    "factual_replay_passed"
                ],
                required_fidelity=values["required_fidelity"],
                conformance_report=(
                    None
                    if raw_report is None
                    else ExactOracleConformanceReport.from_mapping(
                        raw_report
                    )
                ),
                factual_evidence_sha256=values.get(
                    "factual_evidence_sha256"
                ),
                schema=values["schema"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "exact closed-loop gate report is incomplete"
            ) from exc


def _fidelity_satisfies(actual: str, required: str) -> bool:
    if actual == required:
        return True
    return (
        actual == FIDELITY_PC_MCU_EXACT
        and required == FIDELITY_PC_EXACT
    )


def _coerce_identity(value: Any) -> Optional[ControllerBackendIdentity]:
    candidate = (
        value
        if isinstance(value, ControllerBackendIdentity)
        else getattr(value, "identity", value)
    )
    if isinstance(candidate, ControllerBackendIdentity):
        return candidate
    from grape_param_estim.alternative_backends import ExactOracleIdentity

    if not isinstance(candidate, ExactOracleIdentity):
        return None
    return ControllerBackendIdentity(
        backend_id=candidate.backend_id,
        fidelity=candidate.fidelity,
        is_exact=True,
        capabilities=tuple(candidate.capabilities),
        implementation_language=candidate.implementation_language,
        source_commit=candidate.source_commit,
        artifact_sha256=candidate.artifact_sha256,
        protocol=candidate.protocol,
    )


def evaluate_exact_closed_loop_gate(
    backend_or_identity: Any,
    conformance_report: Any,
    *,
    required_fidelity: Any = FIDELITY_PC_EXACT,
) -> ExactClosedLoopGateReport:
    """Fail closed before an exact-controller closed-loop rollout."""

    identity = _coerce_identity(backend_or_identity)
    required = normalize_fidelity(required_fidelity)
    from grape_param_estim.alternative_backends import (
        ExactOracleConformanceReport,
    )

    typed_report = (
        conformance_report
        if isinstance(
            conformance_report, ExactOracleConformanceReport
        )
        else None
    )
    report_identity = (
        None
        if typed_report is None
        else _coerce_identity(typed_report.identity)
    )
    factual_pass = bool(
        typed_report is not None
        and typed_report.content_is_valid()
        and typed_report.passed
        and typed_report.status == "PASS"
        and report_identity is not None
    )
    reasons = []
    capability_report = None
    if not isinstance(identity, ControllerBackendIdentity):
        reasons.append("backend lacks a verified controller identity")
    else:
        if identity.is_exact is not True:
            reasons.append("Python or other surrogate backends are not exact")
        if (
            not identity.source_commit.strip()
            or identity.source_commit.strip().lower() == "unknown"
        ):
            reasons.append("exact backend source commit is unavailable")
        if not _fidelity_satisfies(identity.fidelity, required):
            reasons.append(
                "backend fidelity {} does not satisfy {}".format(
                    identity.fidelity, required
                )
            )
        capability_report = check_capabilities(
            identity, fidelity=required
        )
        if not capability_report.passed:
            reasons.append(
                "backend lacks capabilities: {}".format(
                    ", ".join(capability_report.missing)
                )
            )
    if typed_report is None:
        reasons.append(
            "factual evidence must be ExactOracleConformanceReport"
        )
    elif not typed_report.content_is_valid():
        reasons.append("factual conformance evidence hash is invalid")
    elif (
        report_identity is None
        or identity is None
        or report_identity != identity
    ):
        reasons.append(
            "factual conformance identity/artifact/source does not match backend"
        )
        factual_pass = False
    elif not _fidelity_satisfies(typed_report.fidelity, required):
        reasons.append(
            "factual conformance fidelity {} does not match {}".format(
                typed_report.fidelity, required
            )
        )
        factual_pass = False
    if not factual_pass:
        reasons.append("factual controller replay gate has not passed")
    passed = not reasons
    return ExactClosedLoopGateReport(
        passed=passed,
        status="PASS" if passed else "REJECTED",
        reasons=tuple(reasons),
        identity=identity
        if isinstance(identity, ControllerBackendIdentity)
        else None,
        capability_check=capability_report,
        factual_replay_passed=bool(factual_pass),
        required_fidelity=required,
        conformance_report=typed_report,
        factual_evidence_sha256=(
            None
            if typed_report is None
            else typed_report.evidence_sha256
        ),
    )


exact_closed_loop_gate = evaluate_exact_closed_loop_gate


__all__ = [
    "CapabilityCheck",
    "ExactClosedLoopGateReport",
    "FactualControllerReplay",
    "check_capabilities",
    "evaluate_exact_closed_loop_gate",
    "exact_closed_loop_gate",
    "require_capabilities",
    "run_factual_controller_replay",
]
