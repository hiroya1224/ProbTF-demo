"""Strict, Qt-free adapters from estimator bundles to GUI data models.

The estimator owns the wire format.  This module deliberately delegates all
schema, dtype, member-alignment, completeness, and path validation to
``grape_param_estim.artifact_io`` before exposing convenient immutable GUI
views.  No pickle or object arrays are accepted by that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .project_io import (
    PROJECT_ARTIFACT_LOADER_ID,
    PROJECT_ARTIFACT_LOADER_VERSION,
)

try:
    from grape_param_estim import artifact_io
except ImportError as error:  # pragma: no cover - exercised by GUI startup
    artifact_io = None  # type: ignore[assignment]
    _BACKEND_IMPORT_ERROR = error
else:
    _BACKEND_IMPORT_ERROR = None


GUI_ARTIFACT_LOADER_ID = PROJECT_ARTIFACT_LOADER_ID
GUI_ARTIFACT_LOADER_VERSION = PROJECT_ARTIFACT_LOADER_VERSION


class GuiArtifactError(ValueError):
    """An estimator artifact cannot be represented by this GUI."""


def _backend() -> Any:
    if artifact_io is None:
        raise GuiArtifactError(
            "grape_param_estim.artifact_io is unavailable; start the GUI "
            "from the package launcher or add the estimator src directory "
            "to PYTHONPATH"
        ) from _BACKEND_IMPORT_ERROR
    return artifact_io


def _array(value: Any, *, copy: bool = False) -> np.ndarray:
    result = np.asarray(value)
    return result.copy() if copy else result


def _quaternion_xyzw_to_rpy(value: np.ndarray) -> np.ndarray:
    """Convert display orientation only; correction paths remain rotvec data."""

    quaternion = np.asarray(value, dtype=float)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.stack((roll, pitch, yaw), axis=-1)


@dataclass(frozen=True)
class SharedPosterior:
    member_id: np.ndarray
    parameter_coordinate: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    cog: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    constant_delay: np.ndarray
    ridge: Mapping[str, np.ndarray]
    mode: Mapping[str, np.ndarray]
    iteration_diagnostics: Mapping[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(self.member_id.size)

    @property
    def equal_weights(self) -> np.ndarray:
        if self.size == 0:
            return np.empty(0, dtype=float)
        return np.full(self.size, 1.0 / float(self.size), dtype=float)


@dataclass(frozen=True)
class FlightResult:
    bag_id: str
    time: np.ndarray
    record_time: np.ndarray
    reference_position: np.ndarray
    reference_rpy: np.ndarray
    observed_position: np.ndarray
    observed_orientation_xyzw: np.ndarray
    nominal_position: np.ndarray | None
    nominal_orientation_xyzw: np.ndarray | None
    member_position: np.ndarray | None
    member_orientation_xyzw: np.ndarray | None
    correction_translation: np.ndarray | None
    correction_rotation_vector: np.ndarray | None
    observed_correction_translation: np.ndarray | None
    observed_correction_rotation_vector: np.ndarray | None
    residual_wrench: np.ndarray | None
    flight_state: np.ndarray | None
    q_resolution_sufficient: bool | None
    provenance: Mapping[str, Any]
    calibration: Mapping[str, Any]
    coverage: Mapping[str, Any]
    objective_contribution: float | None

    @property
    def sample_count(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def has_posterior(self) -> bool:
        return self.member_position is not None

    @property
    def observed_rpy(self) -> np.ndarray:
        return _quaternion_xyzw_to_rpy(self.observed_orientation_xyzw)

    @property
    def nominal_rpy(self) -> np.ndarray | None:
        if self.nominal_orientation_xyzw is None:
            return None
        return _quaternion_xyzw_to_rpy(self.nominal_orientation_xyzw)

    @property
    def member_rpy(self) -> np.ndarray | None:
        if self.member_orientation_xyzw is None:
            return None
        return _quaternion_xyzw_to_rpy(self.member_orientation_xyzw)


@dataclass(frozen=True)
class InspectionArtifact:
    root: Path
    manifest: Mapping[str, Any]
    inspections: Mapping[str, Mapping[str, Any]]
    previews: Mapping[str, FlightResult]


@dataclass(frozen=True)
class AssimilationRun:
    root: Path
    manifest: Mapping[str, Any]
    shared_posterior: SharedPosterior
    bag_results: Mapping[str, FlightResult]
    diagnostics: Mapping[str, np.ndarray]
    warnings: tuple[str, ...]

    @property
    def request_fingerprint(self) -> str:
        return str(self.manifest.get("request_fingerprint", ""))


@dataclass(frozen=True)
class PidProposalEvaluation:
    root: Path
    manifest: Mapping[str, Any]
    proposal_ensemble: Mapping[str, np.ndarray]
    summary: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    proposed_yaml: str
    proposed_diff_yaml: str


def _diagnostic_groups(
    diagnostics: Mapping[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    return {
        key[len(prefix) :]: value
        for key, value in diagnostics.items()
        if key.startswith(prefix)
    }


def _preview_result(
    bag_id: str,
    arrays: Mapping[str, np.ndarray],
    inspection: Mapping[str, Any],
) -> FlightResult:
    time_key = "times" if "times" in arrays else "time"
    record_key = "record_times" if "record_times" in arrays else time_key
    return FlightResult(
        bag_id=bag_id,
        time=_array(arrays[time_key]),
        record_time=_array(arrays[record_key]),
        reference_position=_array(arrays["reference_position"]),
        reference_rpy=_array(arrays["reference_rpy"]),
        observed_position=_array(arrays["position"]),
        observed_orientation_xyzw=_array(arrays["orientation_xyzw"]),
        nominal_position=None,
        nominal_orientation_xyzw=None,
        member_position=None,
        member_orientation_xyzw=None,
        correction_translation=None,
        correction_rotation_vector=None,
        observed_correction_translation=None,
        observed_correction_rotation_vector=None,
        residual_wrench=None,
        flight_state=_array(arrays["flight_state"]),
        q_resolution_sufficient=None,
        provenance=inspection,
        calibration={},
        coverage={},
        objective_contribution=None,
    )


def load_inspection(path: str | Path) -> InspectionArtifact:
    """Load a complete inspection bundle through the backend validator."""

    bundle = _backend().load_inspection_bundle(path)
    previews = {
        bag_id: _preview_result(
            bag_id, bundle.previews[bag_id], bundle.inspections[bag_id]
        )
        for bag_id in bundle.inspections
    }
    return InspectionArtifact(
        root=bundle.root,
        manifest=bundle.manifest,
        inspections=bundle.inspections,
        previews=previews,
    )


def load_assimilation(path: str | Path) -> AssimilationRun:
    """Load a complete run, preserving member identity and raw paths."""

    bundle = _backend().load_assimilation_run(path)
    shared = bundle.shared_posterior
    diagnostics = bundle.diagnostics
    posterior = SharedPosterior(
        member_id=_array(shared["member_id"]),
        parameter_coordinate=_array(shared["parameter_coordinates"]),
        mass=_array(shared["mass"]),
        inertia=_array(shared["inertia"]),
        cog=_array(shared["cog"]),
        force_effectiveness=_array(shared["force_effectiveness"]),
        torque_effectiveness=_array(shared["torque_effectiveness"]),
        constant_delay=_array(shared["constant_delay"]),
        ridge=_diagnostic_groups(shared, "ridge_"),
        mode={
            key: shared[key]
            for key in ("mode_id", "mode_weight", "selected_mode_id")
            if key in shared
        },
        iteration_diagnostics=diagnostics,
    )
    bag_results: dict[str, FlightResult] = {}
    for bag_id, arrays in bundle.bags.items():
        provenance = {
            key[len("provenance_") :]: np.asarray(value).tolist()
            for key, value in arrays.items()
            if key.startswith("provenance_")
        }
        bag_results[bag_id] = FlightResult(
            bag_id=bag_id,
            time=_array(arrays["times"]),
            record_time=_array(arrays["record_times"]),
            reference_position=_array(arrays["reference_position"]),
            reference_rpy=_array(arrays["reference_rpy"]),
            observed_position=_array(arrays["observed_position"]),
            observed_orientation_xyzw=_array(arrays["observed_orientation_xyzw"]),
            nominal_position=_array(arrays["nominal_position"]),
            nominal_orientation_xyzw=_array(arrays["nominal_orientation_xyzw"]),
            member_position=_array(arrays["posterior_position"]),
            member_orientation_xyzw=_array(arrays["posterior_orientation_xyzw"]),
            correction_translation=_array(arrays["correction_translation"]),
            correction_rotation_vector=_array(arrays["correction_rotation_vector"]),
            observed_correction_translation=_array(
                arrays["observed_correction_translation"]
            ),
            observed_correction_rotation_vector=_array(
                arrays["observed_correction_rotation_vector"]
            ),
            residual_wrench=_array(arrays["residual_wrench_interval"]),
            flight_state=None,
            q_resolution_sufficient=bool(
                np.asarray(arrays["q_resolution_sufficient"]).reshape(-1)[0]
            ),
            provenance=provenance,
            calibration={
                key: arrays[key]
                for key in (
                    "observation_translation_covariance",
                    "observation_rotation_covariance",
                    "q_stationary_standard_deviation",
                    "q_correlation_time",
                    "q_knot_indices",
                    "q_knot_times",
                )
                if key in arrays
            },
            coverage={"value": float(np.asarray(arrays["pose_component_coverage"]).reshape(-1)[0])},
            objective_contribution=float(np.mean(arrays["objective_contribution"])),
        )
    return AssimilationRun(
        root=bundle.root,
        manifest=bundle.manifest,
        shared_posterior=posterior,
        bag_results=bag_results,
        diagnostics=diagnostics,
        warnings=tuple(bundle.warnings),
    )


def load_pid_evaluation(path: str | Path) -> PidProposalEvaluation:
    """Load exact PID candidates, forecasts, metrics, and YAML text."""

    bundle = _backend().load_pid_proposal_evaluation(path)
    return PidProposalEvaluation(
        root=bundle.root,
        manifest=bundle.manifest,
        proposal_ensemble=bundle.proposal_ensemble,
        summary=bundle.summary,
        bags=bundle.bags,
        proposed_yaml=bundle.proposed_yaml_path.read_text(encoding="utf-8"),
        proposed_diff_yaml=bundle.proposed_diff_yaml_path.read_text(
            encoding="utf-8"
        ),
    )


__all__ = [
    "AssimilationRun",
    "FlightResult",
    "GUI_ARTIFACT_LOADER_ID",
    "GUI_ARTIFACT_LOADER_VERSION",
    "GuiArtifactError",
    "InspectionArtifact",
    "PidProposalEvaluation",
    "SharedPosterior",
    "load_assimilation",
    "load_inspection",
    "load_pid_evaluation",
]
