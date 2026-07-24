"""Closed-loop posterior counterfactual evaluation and support labelling."""

from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.spatial.transform import Rotation

from .controller_replay import (
    AXIS_COUNT,
    ControllerParameters,
    ControllerReplay,
    ReplayRequest,
)
from .effective_response import (
    EffectiveResponseParameters,
    EffectiveResponsePosterior,
    LowDimensionalEffectiveResponse,
    ResponseState,
)
from .episode import stable_hash


SUPPORTED = "SUPPORTED"
EXTRAPOLATIVE = "EXTRAPOLATIVE"
UNSUPPORTED = "UNSUPPORTED"
_SUPPORT_LABELS = (SUPPORTED, EXTRAPOLATIVE, UNSUPPORTED)
DEPENDENCE_JOINT_SAMPLES = "JOINT_POSTERIOR_SAMPLES"
DEPENDENCE_APPROXIMATED = "DEPENDENCE_APPROXIMATED_INDEPENDENT_PRODUCT"
EXPERIMENTAL = "EXPERIMENTAL"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
_EXACT_REPLAY_RMSE_MAX = 0.01
_EXACT_REPLAY_MAXIMUM_ERROR = 0.03
_EXACT_REPLAY_EVENT_AGREEMENT_MIN = 1.0


class PythonControllerReplayBackend:
    """Explicit approximation backend for candidate screening."""

    backend_id = "python_vector_pid_surrogate/v1"
    is_exact = False
    supports_closed_loop_plant_callback = True
    applies_candidate_parameters = True
    applies_delay_compensation = True
    identity = {
        "backend_id": backend_id,
        "implementation_language": "Python",
        "is_exact": False,
    }

    def run(self, *args, **kwargs):
        return ControllerReplay().run(*args, **kwargs)


def _backend_identity(backend: Any) -> Mapping[str, Any]:
    identity = getattr(backend, "identity", None)
    if is_dataclass(identity):
        identity = asdict(identity)
    elif isinstance(identity, Mapping):
        identity = dict(identity)
    if not isinstance(identity, Mapping):
        raise ValueError("controller backend must expose a stable identity mapping")
    backend_id = str(getattr(backend, "backend_id", identity.get("backend_id", "")))
    if not backend_id:
        raise ValueError("controller backend_id must not be empty")
    is_exact = getattr(backend, "is_exact", False)
    if type(is_exact) is not bool:
        raise TypeError("controller backend is_exact must be a built-in bool")
    result = dict(identity)
    result["backend_id"] = backend_id
    result["is_exact"] = is_exact
    for name in (
        "supports_closed_loop_plant_callback",
        "applies_candidate_parameters",
        "applies_delay_compensation",
    ):
        value = getattr(backend, name, False)
        if type(value) is not bool:
            raise TypeError(
                "controller backend {} must be a built-in bool".format(name)
            )
        result[name] = value
    return result


def _vector(values, name, positive=False):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(AXIS_COUNT, float(array))
    if array.shape != (AXIS_COUNT,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite scalar or six-vector".format(name))
    if positive and np.any(array <= 0.0):
        raise ValueError("{} must be strictly positive".format(name))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _matrix(values, rows, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, AXIS_COUNT) or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape ({}, 6)".format(name, rows))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _weighted_mean(values, weights):
    return np.tensordot(weights, values, axes=(0, 0))


def _deep_freeze(value):
    """Return a defensive, recursively immutable copy of artifact metadata."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _deep_freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    return value


def _passes_frozen_exact_replay_metric(metric):
    rmse = np.asarray(getattr(metric, "normalized_rmse", ()), dtype=float)
    maximum = np.asarray(
        getattr(metric, "normalized_maximum_error", ()), dtype=float
    )
    try:
        event_agreement = float(metric.event_agreement)
        rmse_threshold = float(metric.rmse_threshold)
        maximum_threshold = float(metric.maximum_error_threshold)
        event_threshold = float(metric.event_agreement_threshold)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        type(getattr(metric, "passed", None)) is bool
        and metric.passed
        and rmse.size > 0
        and maximum.size > 0
        and np.all(np.isfinite(rmse))
        and np.all(np.isfinite(maximum))
        and np.isfinite(event_agreement)
        and rmse_threshold == _EXACT_REPLAY_RMSE_MAX
        and maximum_threshold == _EXACT_REPLAY_MAXIMUM_ERROR
        and event_threshold == _EXACT_REPLAY_EVENT_AGREEMENT_MIN
        and np.all(rmse <= _EXACT_REPLAY_RMSE_MAX)
        and np.all(maximum <= _EXACT_REPLAY_MAXIMUM_ERROR)
        and event_agreement >= _EXACT_REPLAY_EVENT_AGREEMENT_MIN
    )


def _sha256(value, name):
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


@dataclass(frozen=True)
class TargetTrajectory:
    timestamps: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("target timestamps must be finite and increasing")
        count = times.size
        times_copy = np.array(times, copy=True)
        times_copy.setflags(write=False)
        object.__setattr__(self, "timestamps", times_copy)
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(
                self, name, _matrix(getattr(self, name), count, name)
            )


@dataclass(frozen=True)
class TargetTube:
    position_tolerance: np.ndarray
    velocity_tolerance: np.ndarray
    evaluation_start_offset_s: float = 0.0
    maximum_continuous_saturation_s: float = float("inf")
    minimum_height_m: Optional[float] = None
    maximum_tilt_rad: Optional[float] = None
    absolute_velocity_limit: Optional[np.ndarray] = None
    allowed_outside_duration_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "position_tolerance",
            _vector(self.position_tolerance, "position_tolerance", positive=True),
        )
        object.__setattr__(
            self,
            "velocity_tolerance",
            _vector(self.velocity_tolerance, "velocity_tolerance", positive=True),
        )
        start = float(self.evaluation_start_offset_s)
        saturation = float(self.maximum_continuous_saturation_s)
        allowed = float(self.allowed_outside_duration_s)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("evaluation_start_offset_s must be finite and non-negative")
        if saturation <= 0.0 or np.isnan(saturation):
            raise ValueError("maximum_continuous_saturation_s must be positive")
        if not np.isfinite(allowed) or allowed < 0.0:
            raise ValueError("allowed_outside_duration_s must be finite and non-negative")
        if self.minimum_height_m is not None and not np.isfinite(
            float(self.minimum_height_m)
        ):
            raise ValueError("minimum_height_m must be finite")
        if self.maximum_tilt_rad is not None and (
            not np.isfinite(float(self.maximum_tilt_rad))
            or float(self.maximum_tilt_rad) <= 0.0
        ):
            raise ValueError("maximum_tilt_rad must be finite and positive")
        if self.absolute_velocity_limit is not None:
            object.__setattr__(
                self,
                "absolute_velocity_limit",
                _vector(
                    self.absolute_velocity_limit,
                    "absolute_velocity_limit",
                    positive=True,
                ),
            )
        object.__setattr__(self, "evaluation_start_offset_s", start)
        object.__setattr__(self, "maximum_continuous_saturation_s", saturation)
        object.__setattr__(self, "allowed_outside_duration_s", allowed)


@dataclass(frozen=True)
class TubeEvaluation:
    success: bool
    violations: Tuple[str, ...]
    diagnostic_exceedances: Tuple[str, ...]
    outside_duration_s: float
    maximum_continuous_saturation_s: float
    maximum_position_ratio: float
    maximum_velocity_ratio: float


def _maximum_true_duration(timestamps, mask):
    times = np.asarray(timestamps, dtype=float)
    active = np.asarray(mask, dtype=bool)
    if times.size != active.size:
        raise ValueError("duration timestamps and mask must align")
    maximum = current = 0.0
    for index in range(times.size - 1):
        if active[index]:
            current += float(times[index + 1] - times[index])
            maximum = max(maximum, current)
        else:
            current = 0.0
    return maximum


def evaluate_target_tube(
    target: TargetTrajectory,
    tube: TargetTube,
    position: np.ndarray,
    velocity: np.ndarray,
    saturation: np.ndarray,
) -> TubeEvaluation:
    count = target.timestamps.size
    positions = _matrix(position, count, "position")
    velocities = _matrix(velocity, count, "velocity")
    saturated = np.asarray(saturation, dtype=bool)
    if saturated.shape != (count, AXIS_COUNT):
        raise ValueError("saturation must have shape (N, 6)")
    evaluation = (
        target.timestamps
        >= target.timestamps[0] + tube.evaluation_start_offset_s
    )
    if not np.any(evaluation):
        raise ValueError("target tube starts after the trajectory horizon")
    position_error = positions - target.position
    # Orientation coordinates are rotation vectors, not independent Euler
    # angles.  Use the SO(3) relative log so coupled/large rotations are not
    # misclassified by component-wise wrapping.
    desired_orientation = Rotation.from_rotvec(
        np.array(target.position[:, 3:], copy=True)
    )
    actual_orientation = Rotation.from_rotvec(
        np.array(positions[:, 3:], copy=True)
    )
    position_error[:, 3:] = (
        desired_orientation.inv() * actual_orientation
    ).as_rotvec()
    velocity_error = velocities - target.velocity
    position_ratio = np.max(
        np.abs(position_error[evaluation]) / tube.position_tolerance
    )
    velocity_ratio = np.max(
        np.abs(velocity_error[evaluation]) / tube.velocity_tolerance
    )
    outside = np.zeros(count, dtype=bool)
    outside[evaluation] = (
        np.any(
            np.abs(position_error[evaluation])
            > tube.position_tolerance,
            axis=1,
        )
        | np.any(
            np.abs(velocity_error[evaluation])
            > tube.velocity_tolerance,
            axis=1,
        )
    )
    outside_duration = float(
        np.sum(
            np.diff(target.timestamps)
            * outside[:-1]
        )
    )
    saturation_any = np.any(saturated, axis=1) & evaluation
    maximum_saturation = _maximum_true_duration(
        target.timestamps, saturation_any
    )
    diagnostics = []
    if position_ratio > 1.0:
        diagnostics.append("position_tube")
    if velocity_ratio > 1.0:
        diagnostics.append("velocity_tube")
    violations = []
    # With zero allowance the pointwise tube is strict.  With a non-zero
    # allowance, pointwise excursions remain diagnostics and only their total
    # duration determines success.
    if tube.allowed_outside_duration_s == 0.0:
        violations.extend(diagnostics)
    if outside_duration > tube.allowed_outside_duration_s:
        violations.append("outside_duration")
    if maximum_saturation > tube.maximum_continuous_saturation_s:
        violations.append("saturation_duration")
    if tube.minimum_height_m is not None and np.any(
        positions[evaluation, 2] < float(tube.minimum_height_m)
    ):
        violations.append("ground_or_contact")
    if tube.maximum_tilt_rad is not None:
        body_up_world = actual_orientation.apply(
            np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
        )
        physical_tilt = np.arccos(
            np.clip(body_up_world[:, 2], -1.0, 1.0)
        )
        if np.any(physical_tilt[evaluation] > float(tube.maximum_tilt_rad)):
            violations.append("tilt_safety")
    if tube.absolute_velocity_limit is not None and np.any(
        np.abs(velocities[evaluation]) > tube.absolute_velocity_limit
    ):
        violations.append("velocity_safety")
    violations = tuple(dict.fromkeys(violations))
    return TubeEvaluation(
        success=not violations,
        violations=violations,
        diagnostic_exceedances=tuple(diagnostics),
        outside_duration_s=outside_duration,
        maximum_continuous_saturation_s=maximum_saturation,
        maximum_position_ratio=float(position_ratio),
        maximum_velocity_ratio=float(velocity_ratio),
    )


@dataclass(frozen=True)
class CounterfactualCandidate:
    candidate_id: str
    controller: ControllerParameters
    metadata: Mapping[str, str] = None

    def __post_init__(self):
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not isinstance(self.controller, ControllerParameters):
            raise TypeError("controller must be ControllerParameters")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def vector(self):
        controller = self.controller
        result = np.concatenate(
            (
                controller.p_gain,
                controller.i_gain,
                controller.d_gain,
                np.array([controller.controller_mass]),
                controller.controller_inertia_diagonal,
                controller.allocation_scale,
                np.array(
                    [
                        controller.thrust_scale,
                        controller.delay_compensation_s,
                    ]
                ),
            )
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class InitialStateSample:
    sample_id: int
    state: ResponseState
    weight: float
    stamp: Optional[float] = None
    controller_integral_state: Optional[np.ndarray] = None
    integrator_state_source: str = "UNKNOWN"

    def __post_init__(self):
        if not isinstance(self.state, ResponseState):
            raise TypeError("state must be ResponseState")
        if int(self.sample_id) < 0:
            raise ValueError("sample_id must be non-negative")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("initial-state weight must be finite and positive")
        if self.stamp is not None and not np.isfinite(float(self.stamp)):
            raise ValueError("initial-state stamp must be finite when provided")
        integral = self.controller_integral_state
        if integral is not None:
            integral = _vector(integral, "controller_integral_state")
            if self.integrator_state_source not in (
                "restored_from_controller_state",
                "latent_posterior_sample",
                "explicit_test_assumption",
            ):
                raise ValueError(
                    "an integral state requires explicit restored/latent/assumed provenance"
                )
            object.__setattr__(self, "controller_integral_state", integral)
        elif self.integrator_state_source != "UNKNOWN":
            raise ValueError(
                "integrator_state_source must be UNKNOWN when state is absent"
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "sample_id", int(self.sample_id))
        if self.stamp is not None:
            object.__setattr__(self, "stamp", float(self.stamp))


@dataclass(frozen=True)
class JointPosteriorSample:
    """One coupled initial-state/response-parameter posterior atom."""

    joint_sample_id: int
    initial_sample_id: int
    response_sample_index: int
    weight: float

    def __post_init__(self):
        weight = float(self.weight)
        if (
            int(self.joint_sample_id) < 0
            or int(self.response_sample_index) < 0
            or not np.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("joint posterior sample fields are invalid")
        object.__setattr__(self, "joint_sample_id", int(self.joint_sample_id))
        object.__setattr__(self, "initial_sample_id", int(self.initial_sample_id))
        object.__setattr__(
            self, "response_sample_index", int(self.response_sample_index)
        )
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class SupportReference:
    observed_candidate_vectors: np.ndarray
    observed_state_action_points: np.ndarray
    candidate_scale: np.ndarray
    state_action_scale: np.ndarray
    supported_distance: float = 2.0
    unsupported_distance: float = 5.0
    minimum_importance_ess: float = 2.0
    maximum_predictive_std: float = float("inf")

    def __post_init__(self):
        candidates = np.asarray(self.observed_candidate_vectors, dtype=float)
        points = np.asarray(self.observed_state_action_points, dtype=float)
        if (
            candidates.ndim != 2
            or candidates.shape[0] == 0
            or points.ndim != 2
            or points.shape[0] == 0
            or not np.all(np.isfinite(candidates))
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("support reference needs finite non-empty matrices")
        candidate_scale = np.asarray(self.candidate_scale, dtype=float)
        point_scale = np.asarray(self.state_action_scale, dtype=float)
        if candidate_scale.shape != (candidates.shape[1],) or np.any(
            candidate_scale <= 0.0
        ):
            raise ValueError("candidate_scale does not match support vectors")
        if point_scale.shape != (points.shape[1],) or np.any(point_scale <= 0.0):
            raise ValueError("state_action_scale does not match support points")
        supported = float(self.supported_distance)
        unsupported = float(self.unsupported_distance)
        minimum_ess = float(self.minimum_importance_ess)
        maximum_std = float(self.maximum_predictive_std)
        if (
            not 0.0 < supported < unsupported
            or minimum_ess <= 0.0
            or maximum_std <= 0.0
            or np.isnan(maximum_std)
        ):
            raise ValueError("support thresholds are invalid")
        object.__setattr__(self, "observed_candidate_vectors", candidates)
        object.__setattr__(self, "observed_state_action_points", points)
        object.__setattr__(self, "candidate_scale", candidate_scale)
        object.__setattr__(self, "state_action_scale", point_scale)
        object.__setattr__(self, "supported_distance", supported)
        object.__setattr__(self, "unsupported_distance", unsupported)
        object.__setattr__(self, "minimum_importance_ess", minimum_ess)
        object.__setattr__(self, "maximum_predictive_std", maximum_std)


@dataclass(frozen=True)
class SupportDiagnostics:
    label: str
    candidate_distance: float
    state_action_distance_p95: float
    importance_weight_ess: float
    maximum_predictive_std: float
    reasons: Tuple[str, ...]

    def __post_init__(self):
        if self.label not in _SUPPORT_LABELS:
            raise ValueError("invalid support label")


@dataclass(frozen=True)
class ProbabilityCalibrationReport:
    """Verified held-out evidence for interpreting q as a probability."""

    source_bag_hashes: Tuple[str, ...]
    normalized_dataset_hashes: Tuple[str, ...]
    protocol_sha256: str
    manifest_sha256: str
    selection_result_sha256: str
    model_version: str
    controller_backend_id: str
    controller_backend_identity_sha256: str
    exact_conformance_report_sha256: str
    source_commit: str
    outer_fold_count: int
    selection_candidate_id: str
    required_hard_gates: Tuple[str, ...]
    primary_metric: str
    primary_metric_value: float
    primary_metric_ci_lower: float
    primary_metric_ci_upper: float
    primary_metric_standard_error: float
    selection_protocol: Mapping[str, Any]
    selection_result: Mapping[str, Any]
    content_sha256: str
    schema: str = "grape_probability_calibration/v1"

    @staticmethod
    def _payload(
        source_bag_hashes,
        normalized_dataset_hashes,
        protocol_sha256,
        manifest_sha256,
        selection_result_sha256,
        model_version,
        controller_backend_id,
        controller_backend_identity_sha256,
        exact_conformance_report_sha256,
        source_commit,
        outer_fold_count,
        selection_candidate_id,
        required_hard_gates,
        primary_metric,
        primary_metric_value,
        primary_metric_ci_lower,
        primary_metric_ci_upper,
        primary_metric_standard_error,
        selection_protocol,
        selection_result,
        schema,
    ):
        return {
            "schema": schema,
            "source_bag_hashes": tuple(source_bag_hashes),
            "normalized_dataset_hashes": tuple(normalized_dataset_hashes),
            "protocol_sha256": str(protocol_sha256),
            "manifest_sha256": str(manifest_sha256),
            "selection_result_sha256": str(selection_result_sha256),
            "model_version": str(model_version),
            "controller_backend_id": str(controller_backend_id),
            "controller_backend_identity_sha256": str(
                controller_backend_identity_sha256
            ),
            "exact_conformance_report_sha256": str(
                exact_conformance_report_sha256
            ),
            "source_commit": str(source_commit),
            "outer_fold_count": int(outer_fold_count),
            "selection_candidate_id": str(selection_candidate_id),
            "required_hard_gates": tuple(required_hard_gates),
            "primary_metric": str(primary_metric),
            "primary_metric_value": float(primary_metric_value),
            "primary_metric_ci_lower": float(primary_metric_ci_lower),
            "primary_metric_ci_upper": float(primary_metric_ci_upper),
            "primary_metric_standard_error": float(
                primary_metric_standard_error
            ),
            "selection_protocol": dict(selection_protocol),
            "selection_result": dict(selection_result),
        }

    @classmethod
    def from_selection_result(
        cls,
        *,
        selection_protocol,
        selection_result,
        model_version,
        controller_backend_identity,
    ):
        """Verify a frozen selection result and create a promotion report.

        This is intentionally not a ``passed=True`` constructor: the PASS is
        derived from the frozen LOBO protocol and its content-addressed result.
        """

        protocol = dict(selection_protocol)
        result = dict(selection_result)
        backend_identity = dict(controller_backend_identity)
        episodes = protocol.get("episodes", {})
        source_hashes = tuple(
            episodes[key]["bag_sha256"] for key in sorted(episodes)
        )
        candidate_id = "bayesian_closed_loop"
        candidate = result.get("candidates", {}).get(candidate_id, {})
        statistics = candidate.get("statistics") or {}
        normalized_dataset_hashes = tuple(
            candidate.get("trajectory_sample_bundle_hashes", ())
        )
        exact_hashes = tuple(
            candidate.get("exact_conformance_report_hashes", ())
        )
        required = tuple(
            next(
                item
                for item in protocol["candidate_groups"][
                    "counterfactual_usefulness"
                ]["candidates"]
                if item["candidate_id"] == candidate_id
            ).get("required_hard_gates", ())
        )
        schema = "grape_probability_calibration/v1"
        payload = cls._payload(
            source_hashes,
            normalized_dataset_hashes,
            stable_hash(protocol),
            protocol["manifest_hash"],
            result["result_hash"],
            model_version,
            backend_identity["backend_id"],
            stable_hash(backend_identity),
            exact_hashes[0] if len(exact_hashes) == 1 else "",
            result["source_commit"],
            result["outer_fold_count"],
            candidate_id,
            required,
            candidate["primary_metric"],
            statistics["mean"],
            statistics["bootstrap_lower"],
            statistics["bootstrap_upper"],
            statistics["standard_error"],
            protocol,
            result,
            schema,
        )
        return cls(content_sha256=stable_hash(payload), **payload)

    def __post_init__(self):
        from .selection import (
            DEFAULT_CANDIDATE,
            SELECTION_SCHEMA,
            validate_selection_protocol,
        )

        if self.schema != "grape_probability_calibration/v1":
            raise ValueError("unsupported probability calibration schema")
        protocol = dict(self.selection_protocol)
        result = dict(self.selection_result)
        validate_selection_protocol(protocol)
        if result.get("schema") != SELECTION_SCHEMA:
            raise ValueError("calibration selection result schema mismatch")
        result_without_hash = dict(result)
        recorded_result_hash = result_without_hash.pop("result_hash", None)
        if (
            recorded_result_hash is None
            or stable_hash(result_without_hash) != recorded_result_hash
        ):
            raise ValueError("calibration selection result hash mismatch")
        if stable_hash(protocol) != result.get("protocol_hash"):
            raise ValueError("calibration protocol/result hash mismatch")
        if protocol["manifest_hash"] != result.get("manifest_hash"):
            raise ValueError("calibration manifest binding mismatch")
        folds = tuple(protocol["outer_folds"])
        held_out = tuple(item["held_out_episode"] for item in folds)
        held_out_hashes = tuple(item["held_out_bag_sha256"] for item in folds)
        episode_hashes = {
            key: value["bag_sha256"]
            for key, value in protocol["episodes"].items()
        }
        if (
            len(folds) != 12
            or len(set(held_out)) != 12
            or set(held_out) != set(episode_hashes)
            or any(
                episode_hashes[episode] != digest
                for episode, digest in zip(held_out, held_out_hashes)
            )
        ):
            raise ValueError(
                "probability calibration requires 12 distinct held-out bags"
            )
        groups = result.get("groups", {})
        if (
            result.get("selection_complete") is not True
            or set(groups) != set(protocol["candidate_groups"])
            or any(
                item.get("selected_default") is None
                for item in groups.values()
            )
        ):
            raise ValueError("selection is incomplete")
        group = groups.get("counterfactual_usefulness", {})
        candidate_id = "bayesian_closed_loop"
        candidate = result.get("candidates", {}).get(candidate_id, {})
        protocol_candidate = next(
            (
                item
                for item in protocol["candidate_groups"][
                    "counterfactual_usefulness"
                ]["candidates"]
                if item["candidate_id"] == candidate_id
            ),
            None,
        )
        required_gates = tuple(
            () if protocol_candidate is None
            else protocol_candidate.get("required_hard_gates", ())
        )
        run_hashes = tuple(candidate.get("run_hashes", ()))
        statistics = candidate.get("statistics")
        if (
            group.get("selected_default") != candidate_id
            or candidate.get("status") != DEFAULT_CANDIDATE
            or candidate.get("primary_metric") != "held_out_brier_score"
            or int(candidate.get("observation_count", -1)) != 12
            or tuple(candidate.get("missing_folds", ()))
            or tuple(candidate.get("missing_metric_folds", ()))
            or tuple(candidate.get("failed_hard_gates", ()))
            or len(run_hashes) != 12
            or len(set(run_hashes)) != 12
            or any(_sha256(item, "selection run hash") != item for item in run_hashes)
            or not required_gates
            or statistics is None
            or int(statistics.get("episode_count", -1)) != 12
        ):
            raise ValueError(
                "bayesian closed-loop calibration did not pass all 12 folds "
                "and required hard gates"
            )
        protocol_candidate_ids = {
            item["candidate_id"]
            for group_value in protocol["candidate_groups"].values()
            for item in group_value["candidates"]
        }
        result_candidates = result.get("candidates", {})
        if (
            set(result_candidates) != protocol_candidate_ids
            or any(
                int(item.get("observation_count", -1)) != 12
                or tuple(item.get("missing_folds", ()))
                or tuple(item.get("missing_metric_folds", ()))
                for item in result_candidates.values()
            )
        ):
            raise ValueError(
                "selection must evaluate every protocol candidate on all folds"
            )
        statistic_values = tuple(
            float(statistics[key])
            for key in (
                "mean",
                "bootstrap_lower",
                "bootstrap_upper",
                "standard_error",
            )
        )
        if (
            not np.all(np.isfinite(statistic_values))
            or statistic_values[1] > statistic_values[0]
            or statistic_values[0] > statistic_values[2]
            or statistic_values[3] < 0.0
        ):
            raise ValueError("calibration primary metric/CI is invalid")
        source_hashes = tuple(
            _sha256(item, "source_bag_hash")
            for item in self.source_bag_hashes
        )
        dataset_hashes = tuple(
            _sha256(item, "normalized_dataset_hash")
            for item in self.normalized_dataset_hashes
        )
        protocol_hash = _sha256(self.protocol_sha256, "protocol_sha256")
        manifest_hash = _sha256(self.manifest_sha256, "manifest_sha256")
        selection_hash = _sha256(
            self.selection_result_sha256, "selection_result_sha256"
        )
        backend_hash = _sha256(
            self.controller_backend_identity_sha256,
            "controller_backend_identity_sha256",
        )
        exact_conformance_hash = _sha256(
            self.exact_conformance_report_sha256,
            "exact_conformance_report_sha256",
        )
        if (
            len(source_hashes) != 12
            or len(dataset_hashes) != 12
            or len(set(source_hashes)) != len(source_hashes)
            or len(set(dataset_hashes)) != len(dataset_hashes)
            or set(source_hashes) != set(episode_hashes.values())
        ):
            raise ValueError(
                "calibration requires 12 unique bag and dataset hashes"
            )
        if (
            not self.model_version
            or not self.controller_backend_id
            or not self.source_commit
            or self.source_commit == "UNKNOWN"
        ):
            raise ValueError(
                "calibration model/backend/source commit binding is required"
            )
        model_versions = tuple(candidate.get("model_versions", ()))
        backend_hashes = tuple(
            candidate.get("controller_backend_identity_hashes", ())
        )
        exact_hashes = tuple(
            candidate.get("exact_conformance_report_hashes", ())
        )
        trajectory_hashes = tuple(
            candidate.get("trajectory_sample_bundle_hashes", ())
        )
        if (
            model_versions != (self.model_version,)
            or backend_hashes != (backend_hash,)
            or exact_hashes != (exact_conformance_hash,)
            or len(trajectory_hashes) != 12
            or len(set(trajectory_hashes)) != 12
            or tuple(sorted(dataset_hashes))
            != tuple(sorted(trajectory_hashes))
        ):
            raise ValueError(
                "calibration folds are not bound to one model, controller "
                "backend, exact-conformance report, and dataset bundle each"
            )
        if (
            int(self.outer_fold_count) != 12
            or int(result.get("outer_fold_count", -1)) != 12
            or self.source_commit != result.get("source_commit")
            or protocol_hash != result.get("protocol_hash")
            or manifest_hash != result.get("manifest_hash")
            or selection_hash != recorded_result_hash
            or self.selection_candidate_id != candidate_id
            or tuple(self.required_hard_gates) != required_gates
            or self.primary_metric != candidate["primary_metric"]
            or not np.allclose(
                (
                    self.primary_metric_value,
                    self.primary_metric_ci_lower,
                    self.primary_metric_ci_upper,
                    self.primary_metric_standard_error,
                ),
                statistic_values,
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError("calibration report is not bound to selection evidence")
        payload = self._payload(
            source_hashes,
            dataset_hashes,
            protocol_hash,
            manifest_hash,
            selection_hash,
            self.model_version,
            self.controller_backend_id,
            backend_hash,
            exact_conformance_hash,
            self.source_commit,
            12,
            candidate_id,
            required_gates,
            self.primary_metric,
            self.primary_metric_value,
            self.primary_metric_ci_lower,
            self.primary_metric_ci_upper,
            self.primary_metric_standard_error,
            protocol,
            result,
            self.schema,
        )
        content_hash = _sha256(self.content_sha256, "content_sha256")
        if stable_hash(payload) != content_hash:
            raise ValueError("calibration report content hash mismatch")
        object.__setattr__(self, "source_bag_hashes", source_hashes)
        object.__setattr__(
            self, "normalized_dataset_hashes", dataset_hashes
        )
        object.__setattr__(self, "protocol_sha256", protocol_hash)
        object.__setattr__(self, "manifest_sha256", manifest_hash)
        object.__setattr__(self, "selection_result_sha256", selection_hash)
        object.__setattr__(
            self, "controller_backend_identity_sha256", backend_hash
        )
        object.__setattr__(
            self,
            "exact_conformance_report_sha256",
            exact_conformance_hash,
        )
        object.__setattr__(self, "outer_fold_count", 12)
        object.__setattr__(self, "required_hard_gates", required_gates)
        object.__setattr__(self, "selection_protocol", protocol)
        object.__setattr__(self, "selection_result", result)
        object.__setattr__(self, "content_sha256", content_hash)

    @property
    def passed(self):
        return True

    @property
    def status(self):
        return "PASS"


def classify_support(
    candidate_vector: np.ndarray,
    rollout_state_action_points: np.ndarray,
    predictive_std: float,
    reference: SupportReference,
) -> SupportDiagnostics:
    candidate = np.asarray(candidate_vector, dtype=float).reshape(-1)
    rollout_points = np.asarray(rollout_state_action_points, dtype=float)
    if candidate.shape != (reference.observed_candidate_vectors.shape[1],):
        raise ValueError("candidate vector dimension does not match support reference")
    if (
        rollout_points.ndim != 2
        or rollout_points.shape[1]
        != reference.observed_state_action_points.shape[1]
        or rollout_points.shape[0] == 0
    ):
        raise ValueError("rollout state/action points do not match support reference")
    candidate_distances = np.linalg.norm(
        (reference.observed_candidate_vectors - candidate)
        / reference.candidate_scale,
        axis=1,
    )
    candidate_distance = float(np.min(candidate_distances))
    log_importance = -0.5 * candidate_distances * candidate_distances
    log_importance -= float(np.max(log_importance))
    importance = np.exp(log_importance)
    importance /= np.sum(importance)
    importance_ess = float(1.0 / np.dot(importance, importance))
    # Chunking avoids a T x N x D allocation for long offline rollouts.
    nearest = []
    for start in range(0, rollout_points.shape[0], 256):
        chunk = rollout_points[start : start + 256]
        distance = np.linalg.norm(
            (
                chunk[:, None, :]
                - reference.observed_state_action_points[None, :, :]
            )
            / reference.state_action_scale,
            axis=2,
        )
        nearest.extend(np.min(distance, axis=1))
    state_distance = float(np.quantile(nearest, 0.95))
    uncertainty = float(predictive_std)
    reasons = []
    if candidate_distance > reference.unsupported_distance:
        reasons.append("candidate_parameter_distance")
    if state_distance > reference.unsupported_distance:
        reasons.append("state_action_distance")
    if importance_ess < reference.minimum_importance_ess:
        reasons.append("importance_weight_ess")
    if not np.isfinite(uncertainty) or uncertainty > reference.maximum_predictive_std:
        reasons.append("posterior_predictive_uncertainty")
    if reasons:
        label = UNSUPPORTED
    elif (
        candidate_distance > reference.supported_distance
        or state_distance > reference.supported_distance
    ):
        label = EXTRAPOLATIVE
        reasons.append("near_support_boundary")
    else:
        label = SUPPORTED
    return SupportDiagnostics(
        label=label,
        candidate_distance=candidate_distance,
        state_action_distance_p95=state_distance,
        importance_weight_ess=importance_ess,
        maximum_predictive_std=uncertainty,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CounterfactualConfig:
    process_noise_sigma: np.ndarray
    process_noise_replicates: int = 1
    credible_probability: float = 0.95
    seed: int = 7
    analysis_mode: str = "retrospective"
    prefix_cutoff: Optional[float] = None
    inference_data_end_time: Optional[float] = None
    source_bag_hashes: Tuple[str, ...] = ()
    normalized_dataset_hashes: Tuple[str, ...] = ()
    source_commit: str = "UNKNOWN"
    recommendation_threshold: float = 0.80

    def __post_init__(self):
        sigma = _vector(self.process_noise_sigma, "process_noise_sigma")
        if np.any(sigma < 0.0):
            raise ValueError("process_noise_sigma must be non-negative")
        replicates = int(self.process_noise_replicates)
        probability = float(self.credible_probability)
        if replicates < 1 or not 0.0 < probability < 1.0:
            raise ValueError("replicate count or credible probability is invalid")
        recommendation_threshold = float(self.recommendation_threshold)
        if not 0.0 < recommendation_threshold < 1.0:
            raise ValueError("recommendation_threshold must lie in (0, 1)")
        if self.analysis_mode not in ("retrospective", "online_prefix"):
            raise ValueError("analysis_mode must be retrospective or online_prefix")
        if self.analysis_mode == "online_prefix":
            if self.prefix_cutoff is None or not np.isfinite(float(self.prefix_cutoff)):
                raise ValueError("online_prefix analysis requires a finite prefix_cutoff")
            if self.inference_data_end_time is None or not np.isfinite(
                float(self.inference_data_end_time)
            ):
                raise ValueError(
                    "online_prefix analysis requires a finite inference_data_end_time"
                )
            if float(self.inference_data_end_time) > float(self.prefix_cutoff):
                raise ValueError(
                    "inference_data_end_time cannot exceed prefix_cutoff"
                )
        object.__setattr__(self, "process_noise_sigma", sigma)
        object.__setattr__(self, "process_noise_replicates", replicates)
        object.__setattr__(self, "credible_probability", probability)
        object.__setattr__(
            self, "recommendation_threshold", recommendation_threshold
        )
        object.__setattr__(self, "source_bag_hashes", tuple(self.source_bag_hashes))
        object.__setattr__(
            self, "normalized_dataset_hashes", tuple(self.normalized_dataset_hashes)
        )


@dataclass(frozen=True)
class TrajectoryRollout:
    rollout_id: int
    initial_sample_id: int
    response_sample_id: int
    noise_sample_id: int
    weight: float
    position: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    saturation: np.ndarray
    tube: TubeEvaluation
    joint_sample_id: int = -1

    def __post_init__(self):
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("rollout weight must be finite and positive")
        position = np.asarray(self.position, dtype=float)
        count = position.shape[0] if position.ndim == 2 else -1
        arrays = (
            ("position", position, float),
            ("velocity", self.velocity, float),
            ("command", self.command, float),
            ("saturation", self.saturation, bool),
        )
        for name, values, dtype in arrays:
            array = np.asarray(values, dtype=dtype)
            if array.shape != (count, AXIS_COUNT):
                raise ValueError(
                    "{} must have aligned shape (N, 6)".format(name)
                )
            if dtype is float and not np.all(np.isfinite(array)):
                raise ValueError("{} must be finite".format(name))
            copy = np.array(array, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        if not isinstance(self.tube, TubeEvaluation):
            raise TypeError("tube must be TubeEvaluation")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "joint_sample_id", int(self.joint_sample_id))


@dataclass(frozen=True)
class CounterfactualResult:
    candidate: CounterfactualCandidate
    success_probability: float
    credible_lower: float
    credible_upper: float
    lower_credible_bound: float
    support: SupportDiagnostics
    violation_probability: Mapping[str, float]
    effective_rollout_sample_size: float
    rollouts: Tuple[TrajectoryRollout, ...]
    run_id: str
    provenance: Mapping[str, object]
    recommendation_threshold: float = 0.80
    exact_controller_gate_passed: bool = False
    probability_calibration_gate_passed: bool = False
    integrator_state_gate_passed: bool = False
    dependence_handling: str = DEPENDENCE_APPROXIMATED
    workflow_status: str = EXPERIMENTAL

    def __post_init__(self):
        if not isinstance(self.candidate, CounterfactualCandidate):
            raise TypeError("candidate must be CounterfactualCandidate")
        if not isinstance(self.support, SupportDiagnostics):
            raise TypeError("support must be SupportDiagnostics")
        rollouts = tuple(self.rollouts)
        if not rollouts or any(
            not isinstance(item, TrajectoryRollout) for item in rollouts
        ):
            raise TypeError(
                "rollouts must contain at least one TrajectoryRollout"
            )
        for name in (
            "exact_controller_gate_passed",
            "probability_calibration_gate_passed",
            "integrator_state_gate_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("{} must be a built-in bool".format(name))
        if self.dependence_handling not in (
            DEPENDENCE_JOINT_SAMPLES,
            DEPENDENCE_APPROXIMATED,
        ):
            raise ValueError("dependence_handling is invalid")

        probability_values = tuple(
            float(value)
            for value in (
                self.success_probability,
                self.credible_lower,
                self.credible_upper,
                self.lower_credible_bound,
                self.recommendation_threshold,
            )
        )
        (
            success_probability,
            credible_lower,
            credible_upper,
            lower_credible_bound,
            recommendation_threshold,
        ) = probability_values
        if (
            not np.all(np.isfinite(probability_values))
            or not 0.0 <= success_probability <= 1.0
            or not 0.0 <= credible_lower <= credible_upper <= 1.0
            or not 0.0 <= lower_credible_bound <= 1.0
            or not 0.0 < recommendation_threshold < 1.0
        ):
            raise ValueError(
                "counterfactual probability, interval, or threshold is invalid"
            )
        weights = np.asarray([item.weight for item in rollouts], dtype=float)
        if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("counterfactual rollout weights must sum to one")
        if any(type(item.tube.success) is not bool for item in rollouts):
            raise TypeError("trajectory tube success must be a built-in bool")
        measured_success = float(
            np.dot(
                weights,
                np.asarray([item.tube.success for item in rollouts], dtype=float),
            )
        )
        effective_count = float(1.0 / np.dot(weights, weights))
        if (
            not np.isclose(
                success_probability,
                measured_success,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                float(self.effective_rollout_sample_size),
                effective_count,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "counterfactual summary does not match its trajectory rollouts"
            )
        violations = {
            str(name): float(value)
            for name, value in self.violation_probability.items()
        }
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in violations.values()
        ):
            raise ValueError("violation probabilities must lie in [0, 1]")

        provenance = dict(self.provenance)
        gate_fields = (
            "exact_controller_gate_passed",
            "probability_calibration_gate_passed",
            "integrator_state_gate_passed",
        )
        if any(
            name not in provenance
            or type(provenance[name]) is not bool
            or provenance[name] is not getattr(self, name)
            for name in gate_fields
        ):
            raise ValueError(
                "counterfactual gate fields do not match provenance"
            )
        if (
            provenance.get("dependence_handling") != self.dependence_handling
            or provenance.get("workflow_status") != self.workflow_status
            or float(provenance.get("recommendation_threshold", np.nan))
            != recommendation_threshold
        ):
            raise ValueError(
                "counterfactual status fields do not match provenance"
            )
        content_hash = _sha256(
            provenance.get("counterfactual_content_hash", ""),
            "counterfactual_content_hash",
        )
        run_id = str(self.run_id)
        if not run_id or run_id != content_hash[:20]:
            raise ValueError("counterfactual run_id does not match content hash")

        statistically_eligible = (
            self.exact_controller_gate_passed
            and self.probability_calibration_gate_passed
            and self.integrator_state_gate_passed
            and self.dependence_handling == DEPENDENCE_JOINT_SAMPLES
            and self.support.label == SUPPORTED
            and lower_credible_bound >= recommendation_threshold
        )
        expected_status = (
            MANUAL_REVIEW_REQUIRED if statistically_eligible else EXPERIMENTAL
        )
        if self.workflow_status != expected_status:
            raise ValueError(
                "workflow_status does not match the counterfactual gates"
            )

        object.__setattr__(self, "success_probability", success_probability)
        object.__setattr__(self, "credible_lower", credible_lower)
        object.__setattr__(self, "credible_upper", credible_upper)
        object.__setattr__(self, "lower_credible_bound", lower_credible_bound)
        object.__setattr__(
            self, "recommendation_threshold", recommendation_threshold
        )
        object.__setattr__(
            self, "effective_rollout_sample_size", effective_count
        )
        object.__setattr__(self, "rollouts", rollouts)
        object.__setattr__(
            self, "violation_probability", _deep_freeze(violations)
        )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "provenance", _deep_freeze(provenance))

    @property
    def recommendable(self):
        return bool(
            self.exact_controller_gate_passed
            and self.probability_calibration_gate_passed
            and self.integrator_state_gate_passed
            and self.dependence_handling == DEPENDENCE_JOINT_SAMPLES
            and self.support.label == SUPPORTED
            and self.lower_credible_bound >= self.recommendation_threshold
        )


class ClosedLoopCounterfactualEvaluator:
    def __init__(
        self,
        support_reference: SupportReference,
        response_model: Optional[LowDimensionalEffectiveResponse] = None,
        controller_backend_factory: Optional[Callable[[], Any]] = None,
        exact_oracle_conformance_report: Optional[Any] = None,
        probability_calibration_report: Optional[
            ProbabilityCalibrationReport
        ] = None,
    ):
        if not isinstance(support_reference, SupportReference):
            raise TypeError("support_reference must be SupportReference")
        self.support_reference = support_reference
        self.response_model = response_model or LowDimensionalEffectiveResponse()
        self.controller_backend_factory = (
            controller_backend_factory
            if controller_backend_factory is not None
            else PythonControllerReplayBackend
        )
        if not callable(self.controller_backend_factory):
            raise TypeError("controller_backend_factory must be callable")
        if exact_oracle_conformance_report is not None:
            from .alternative_backends import ExactOracleConformanceReport

            if not isinstance(
                exact_oracle_conformance_report, ExactOracleConformanceReport
            ):
                raise TypeError(
                    "exact_oracle_conformance_report must be a verified "
                    "ExactOracleConformanceReport"
                )
        self.exact_oracle_conformance_report = (
            exact_oracle_conformance_report
        )
        if (
            probability_calibration_report is not None
            and not isinstance(
                probability_calibration_report,
                ProbabilityCalibrationReport,
            )
        ):
            raise TypeError(
                "probability_calibration_report must be a content-addressed "
                "ProbabilityCalibrationReport"
            )
        self.probability_calibration_report = probability_calibration_report

    def evaluate(
        self,
        candidate: CounterfactualCandidate,
        target: TargetTrajectory,
        tube: TargetTube,
        response_posterior: EffectiveResponsePosterior,
        initial_state_samples: Sequence[InitialStateSample],
        config: CounterfactualConfig,
        joint_posterior_samples: Optional[
            Sequence[JointPosteriorSample]
        ] = None,
    ) -> CounterfactualResult:
        if not isinstance(candidate, CounterfactualCandidate):
            raise TypeError("candidate must be CounterfactualCandidate")
        initials = tuple(initial_state_samples)
        if not initials:
            raise ValueError("at least one initial-state sample is required")
        if any(item.controller_integral_state is None for item in initials):
            raise ValueError(
                "every initial-state sample requires an explicit restored or "
                "latent controller_integral_state"
            )
        backend = self.controller_backend_factory()
        backend_identity = _backend_identity(backend)
        if (
            not callable(getattr(backend, "run", None))
            or not backend_identity["supports_closed_loop_plant_callback"]
            or not backend_identity["applies_candidate_parameters"]
            or not backend_identity["applies_delay_compensation"]
        ):
            raise TypeError(
                "controller backend must support closed-loop plant callbacks "
                "and apply candidate parameters/delay compensation"
            )
        backend_is_exact = backend_identity["is_exact"]
        if config.analysis_mode == "online_prefix":
            cutoff = float(config.prefix_cutoff)
            tolerance = 1.0e-9
            if abs(float(target.timestamps[0]) - cutoff) > tolerance:
                raise ValueError(
                    "online-prefix counterfactual target must start at prefix_cutoff"
                )
            if any(
                item.stamp is None or abs(float(item.stamp) - cutoff) > tolerance
                for item in initials
            ):
                raise ValueError(
                    "online-prefix initial states must be stamped at prefix_cutoff"
                )
        initial_ids = [item.sample_id for item in initials]
        if len(set(initial_ids)) != len(initials):
            raise ValueError("initial-state sample IDs must be unique")
        initial_weights = np.asarray([item.weight for item in initials], dtype=float)
        initial_weights /= np.sum(initial_weights)
        initial_by_id = {
            item.sample_id: index for index, item in enumerate(initials)
        }
        joint = None
        if joint_posterior_samples is None:
            coupling = DEPENDENCE_APPROXIMATED
            pairs = [
                (
                    -1,
                    initial_index,
                    response_index,
                    initial_weights[initial_index]
                    * float(response_posterior.weights[response_index]),
                )
                for initial_index in range(len(initials))
                for response_index in range(len(response_posterior.samples))
                if response_posterior.weights[response_index] > 0.0
            ]
        else:
            joint = tuple(joint_posterior_samples)
            if not joint or any(
                not isinstance(item, JointPosteriorSample) for item in joint
            ):
                raise ValueError(
                    "joint_posterior_samples must contain coupled posterior atoms"
                )
            joint_ids = [item.joint_sample_id for item in joint]
            if len(set(joint_ids)) != len(joint_ids):
                raise ValueError("joint posterior sample IDs must be unique")
            total_joint_weight = float(np.sum([item.weight for item in joint]))
            pairs = []
            for item in joint:
                if item.initial_sample_id not in initial_by_id:
                    raise ValueError(
                        "joint sample references an unknown initial sample"
                    )
                if (
                    item.response_sample_index
                    >= len(response_posterior.samples)
                    or response_posterior.weights[item.response_sample_index]
                    <= 0.0
                ):
                    raise ValueError(
                        "joint sample references an unsupported response sample"
                    )
                pairs.append(
                    (
                        item.joint_sample_id,
                        initial_by_id[item.initial_sample_id],
                        item.response_sample_index,
                        item.weight / total_joint_weight,
                    )
                )
            coupling = DEPENDENCE_JOINT_SAMPLES
        request = ReplayRequest(
            timestamps=target.timestamps,
            reference_position=target.position,
            reference_velocity=target.velocity,
            reference_acceleration=target.acceleration,
        )
        rollouts = []
        rollout_id = 0
        for joint_sample_id, initial_index, response_index, pair_weight in pairs:
            initial = initials[initial_index]
            response_parameters = response_posterior.samples[response_index]
            for noise_index in range(config.process_noise_replicates):
                    seed_sequence = np.random.SeedSequence(
                        [
                            int(config.seed),
                            int(initial.sample_id),
                            int(response_index),
                            int(noise_index),
                            int(joint_sample_id + 1),
                        ]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    process_noise = rng.normal(
                        0.0,
                        config.process_noise_sigma,
                        size=(target.timestamps.size - 1, AXIS_COUNT),
                    )
                    actuator_state = initial.state.actuator_state.copy()
                    command_times = []
                    commands = []

                    def plant_step(position, velocity, command, delta, index):
                        nonlocal actuator_state
                        command_times.append(float(target.timestamps[index]))
                        commands.append(np.array(command, copy=True))
                        current = ResponseState(
                            position, velocity, actuator_state
                        )
                        transition = self.response_model.transition(
                            current,
                            np.asarray(command_times),
                            np.asarray(commands),
                            target.timestamps[index],
                            delta,
                            response_parameters,
                            process_noise=process_noise[index],
                        )
                        actuator_state = transition.state.actuator_state.copy()
                        return (
                            transition.state.generalized_position,
                            transition.state.generalized_velocity,
                        )

                    replay = backend.run(
                        request,
                        candidate.controller,
                        replay_mode="free_run",
                        initial_position=initial.state.generalized_position,
                        initial_velocity=initial.state.generalized_velocity,
                        initial_integral_state=initial.controller_integral_state,
                        plant_step=plant_step,
                        plant_input="generalized_wrench_command",
                        apply_delay_compensation=True,
                    )
                    replay_backend_id = getattr(replay, "backend_id", None)
                    replay_is_exact = getattr(replay, "is_exact", None)
                    if (
                        replay_backend_id != backend_identity["backend_id"]
                        or type(replay_is_exact) is not bool
                        or replay_is_exact != backend_is_exact
                    ):
                        raise ValueError(
                            "controller replay result does not match the "
                            "bound backend identity/exactness"
                        )
                    replay_identity = getattr(replay, "identity", None)
                    if (
                        replay_identity is not None
                        and replay_identity != getattr(backend, "identity", None)
                    ):
                        raise ValueError(
                            "controller replay result identity does not match "
                            "the bound backend"
                        )
                    replay_capabilities = getattr(
                        replay, "capabilities", None
                    )
                    if (
                        replay_capabilities is not None
                        and tuple(replay_capabilities)
                        != tuple(backend_identity.get("capabilities", ()))
                    ):
                        raise ValueError(
                            "controller replay result capabilities do not "
                            "match the bound backend"
                        )
                    saturation = replay.term_saturated | replay.output_saturated
                    tube_result = evaluate_target_tube(
                        target,
                        tube,
                        replay.feedback_position,
                        replay.feedback_velocity,
                        saturation,
                    )
                    weight = pair_weight / config.process_noise_replicates
                    rollouts.append(
                        TrajectoryRollout(
                            rollout_id=rollout_id,
                            initial_sample_id=int(initial.sample_id),
                            response_sample_id=int(response_index),
                            noise_sample_id=int(noise_index),
                            weight=float(weight),
                            position=replay.feedback_position,
                            velocity=replay.feedback_velocity,
                            command=replay.generalized_wrench_command,
                            saturation=saturation,
                            tube=tube_result,
                            joint_sample_id=joint_sample_id,
                        )
                    )
                    rollout_id += 1
        if not rollouts:
            raise ValueError(
                "response posterior and initial samples produced no rollout mass"
            )
        weights = np.asarray([item.weight for item in rollouts], dtype=float)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("counterfactual rollout weights are invalid")
        weights /= np.sum(weights)
        # Normalize the immutable rollout weights at the result boundary.
        normalized_rollouts = []
        for item, weight in zip(rollouts, weights):
            normalized_rollouts.append(
                TrajectoryRollout(
                    rollout_id=item.rollout_id,
                    initial_sample_id=item.initial_sample_id,
                    response_sample_id=item.response_sample_id,
                    noise_sample_id=item.noise_sample_id,
                    weight=float(weight),
                    position=item.position,
                    velocity=item.velocity,
                    command=item.command,
                    saturation=item.saturation,
                    tube=item.tube,
                    joint_sample_id=item.joint_sample_id,
                )
            )
        rollouts = tuple(normalized_rollouts)
        successes = np.asarray([item.tube.success for item in rollouts], dtype=float)
        success_probability = float(
            np.clip(np.dot(weights, successes), 0.0, 1.0)
        )
        effective_count = float(1.0 / np.dot(weights, weights))
        successful_equivalent = success_probability * effective_count
        alpha = 0.5 + successful_equivalent
        beta = 0.5 + effective_count - successful_equivalent
        tail = 0.5 * (1.0 - config.credible_probability)
        lower = float(beta_distribution.ppf(tail, alpha, beta))
        upper = float(beta_distribution.ppf(1.0 - tail, alpha, beta))
        all_violations = sorted(
            set(
                violation
                for item in rollouts
                for violation in item.tube.violations
            )
        )
        violation_probability = {
            violation: float(
                np.dot(
                    weights,
                    np.asarray(
                        [
                            violation in item.tube.violations
                            for item in rollouts
                        ],
                        dtype=float,
                    ),
                )
            )
            for violation in all_violations
        }
        positions = np.asarray([item.position for item in rollouts])
        mean_position = _weighted_mean(positions, weights)
        variance = _weighted_mean(
            (positions - mean_position) ** 2, weights
        )
        predictive_std = float(np.max(np.sqrt(np.maximum(variance, 0.0))))
        state_action_points = np.concatenate(
            [
                np.column_stack(
                    (item.position, item.velocity, item.command)
                )
                for item in rollouts
            ],
            axis=0,
        )
        support = classify_support(
            candidate.vector(),
            state_action_points,
            predictive_std,
            self.support_reference,
        )
        from .alternative_backends import REQUIRED_CONFORMANCE_CHANNELS

        conformance = self.exact_oracle_conformance_report
        conformance_identity = None
        identity_bound = False
        if conformance is not None and conformance.identity is not None:
            conformance_identity = asdict(conformance.identity)
            conformance_identity["is_exact"] = True
            conformance_identity[
                "supports_closed_loop_plant_callback"
            ] = True
            conformance_identity["applies_candidate_parameters"] = True
            conformance_identity["applies_delay_compensation"] = True
            identity_bound = conformance_identity == backend_identity
        conformance_metrics_passed = bool(
            conformance is not None
            and set(conformance.channel_metrics)
            == set(REQUIRED_CONFORMANCE_CHANNELS)
            and all(
                _passes_frozen_exact_replay_metric(metric)
                for metric in conformance.channel_metrics.values()
            )
        )
        fixture_provenance = (
            None if conformance is None else conformance.fixture_provenance
        )
        conformance_fixture_bound = bool(
            fixture_provenance is not None
            and fixture_provenance.content_is_valid()
            and conformance.fixture_content_sha256
            == fixture_provenance.content_sha256
            and conformance.request_payload_sha256
            == fixture_provenance.fixture_input_payload_sha256
            and fixture_provenance.source_bag_sha256
            in config.source_bag_hashes
        )
        exact_gate_passed = bool(
            backend_is_exact
            and conformance is not None
            and type(conformance.passed) is bool
            and conformance.passed
            and conformance.status == "PASS"
            and conformance_metrics_passed
            and conformance_fixture_bound
            and identity_bound
        )
        calibration = self.probability_calibration_report
        calibration_binding = {
            "report_present": calibration is not None,
            "model_version_matches": bool(
                calibration is not None
                and calibration.model_version == self.response_model.model_id
            ),
            "backend_id_matches": bool(
                calibration is not None
                and calibration.controller_backend_id
                == backend_identity["backend_id"]
            ),
            "backend_identity_matches": bool(
                calibration is not None
                and calibration.controller_backend_identity_sha256
                == stable_hash(backend_identity)
            ),
            "exact_conformance_report_matches": bool(
                calibration is not None
                and conformance is not None
                and calibration.exact_conformance_report_sha256
                == stable_hash(asdict(conformance))
            ),
            "exact_fixture_source_bag_calibrated": bool(
                calibration is not None
                and fixture_provenance is not None
                and fixture_provenance.source_bag_sha256
                in calibration.source_bag_hashes
            ),
            "source_commit_matches": bool(
                calibration is not None
                and config.source_commit != "UNKNOWN"
                and calibration.source_commit == config.source_commit
            ),
            "source_bags_covered": bool(
                calibration is not None
                and config.source_bag_hashes
                and set(config.source_bag_hashes).issubset(
                    calibration.source_bag_hashes
                )
            ),
            "normalized_datasets_covered": bool(
                calibration is not None
                and config.normalized_dataset_hashes
                and set(config.normalized_dataset_hashes).issubset(
                    calibration.normalized_dataset_hashes
                )
            ),
        }
        calibration_gate_passed = bool(
            calibration is not None
            and calibration.passed
            and all(calibration_binding.values())
        )
        integrator_state_gate_passed = all(
            item.integrator_state_source
            in (
                "restored_from_controller_state",
                "latent_posterior_sample",
            )
            for item in initials
        )
        content_payload = {
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "controller_vector": candidate.vector(),
                "controller_limits": {
                    "output": candidate.controller.limits.output,
                    "p_term": candidate.controller.limits.p_term,
                    "i_term": candidate.controller.limits.i_term,
                    "d_term": candidate.controller.limits.d_term,
                    "p_error": candidate.controller.limits.p_error,
                    "i_state": candidate.controller.limits.i_state,
                    "d_error": candidate.controller.limits.d_error,
                },
                "metadata": candidate.metadata,
            },
            "target": {
                "timestamps": target.timestamps,
                "position": target.position,
                "velocity": target.velocity,
                "acceleration": target.acceleration,
            },
            "target_tube": {
                "position_tolerance": tube.position_tolerance,
                "velocity_tolerance": tube.velocity_tolerance,
                "evaluation_start_offset_s": tube.evaluation_start_offset_s,
                "maximum_continuous_saturation_s": (
                    tube.maximum_continuous_saturation_s
                    if np.isfinite(tube.maximum_continuous_saturation_s)
                    else "UNBOUNDED"
                ),
                "minimum_height_m": tube.minimum_height_m,
                "maximum_tilt_rad": tube.maximum_tilt_rad,
                "absolute_velocity_limit": tube.absolute_velocity_limit,
                "allowed_outside_duration_s": tube.allowed_outside_duration_s,
            },
            "response_posterior": {
                "parameter_vectors": [
                    item.as_vector() for item in response_posterior.samples
                ],
                "weights": response_posterior.weights,
                "grid_delay_s": response_posterior.grid_delay_s,
                "grid_time_constant_s": response_posterior.grid_time_constant_s,
                "grid_weights": response_posterior.grid_weights,
                "log_evidence": response_posterior.log_evidence,
                "approximation": response_posterior.approximation,
                "source_sample_ids": response_posterior.source_sample_ids,
            },
            "initial_samples": [
                {
                    "sample_id": item.sample_id,
                    "weight": item.weight,
                    "stamp": item.stamp,
                    "state_position": item.state.generalized_position,
                    "state_velocity": item.state.generalized_velocity,
                    "actuator_state": item.state.actuator_state,
                    "controller_integral_state": (
                        item.controller_integral_state
                    ),
                    "integrator_state_source": item.integrator_state_source,
                }
                for item in initials
            ],
            "joint_posterior_samples": (
                None
                if joint is None
                else [
                    {
                        "joint_sample_id": item.joint_sample_id,
                        "initial_sample_id": item.initial_sample_id,
                        "response_sample_index": item.response_sample_index,
                        "weight": item.weight,
                    }
                    for item in joint
                ]
            ),
            "dependence_handling": coupling,
            "process_noise_sigma": config.process_noise_sigma,
            "process_noise_replicates": config.process_noise_replicates,
            "credible_probability": config.credible_probability,
            "seed": config.seed,
            "analysis_mode": config.analysis_mode,
            "prefix_cutoff": config.prefix_cutoff,
            "inference_data_end_time": config.inference_data_end_time,
            "source_bag_hashes": config.source_bag_hashes,
            "normalized_dataset_hashes": config.normalized_dataset_hashes,
            "source_commit": config.source_commit,
            "recommendation_threshold": config.recommendation_threshold,
            "exact_controller_gate_passed": exact_gate_passed,
            "probability_calibration_gate_passed": calibration_gate_passed,
            "integrator_state_gate_passed": integrator_state_gate_passed,
            "support_reference": {
                "observed_candidate_vectors": (
                    self.support_reference.observed_candidate_vectors
                ),
                "observed_state_action_points": (
                    self.support_reference.observed_state_action_points
                ),
                "candidate_scale": self.support_reference.candidate_scale,
                "state_action_scale": self.support_reference.state_action_scale,
                "supported_distance": self.support_reference.supported_distance,
                "unsupported_distance": (
                    self.support_reference.unsupported_distance
                ),
                "minimum_importance_ess": (
                    self.support_reference.minimum_importance_ess
                ),
                "maximum_predictive_std": (
                    self.support_reference.maximum_predictive_std
                    if np.isfinite(
                        self.support_reference.maximum_predictive_std
                    )
                    else "UNBOUNDED"
                ),
            },
            "controller_backend_identity": backend_identity,
            "exact_oracle_conformance": {
                "present": conformance is not None,
                "passed": bool(
                    conformance is not None and conformance.passed
                ),
                "status": (
                    None if conformance is None else conformance.status
                ),
                "identity_bound_to_backend": identity_bound,
                "all_required_metrics_passed": conformance_metrics_passed,
                "fixture_provenance_bound_to_source_bag": (
                    conformance_fixture_bound
                ),
                "identity": conformance_identity,
                "report": (
                    None if conformance is None else asdict(conformance)
                ),
            },
            "probability_calibration": {
                "binding": calibration_binding,
                "report": (
                    None if calibration is None else asdict(calibration)
                ),
            },
            "response_model_id": self.response_model.model_id,
        }
        content_hash = stable_hash(content_payload)
        statistically_eligible = bool(
            exact_gate_passed
            and calibration_gate_passed
            and integrator_state_gate_passed
            and coupling == DEPENDENCE_JOINT_SAMPLES
            and support.label == SUPPORTED
            and lower >= config.recommendation_threshold
        )
        workflow_status = (
            MANUAL_REVIEW_REQUIRED if statistically_eligible else EXPERIMENTAL
        )
        provenance = {
            "source_bag_hashes": config.source_bag_hashes,
            "normalized_dataset_hashes": config.normalized_dataset_hashes,
            "source_commit": config.source_commit,
            "model_version": self.response_model.model_id,
            "controller_backend": backend_identity["backend_id"],
            "controller_backend_identity": backend_identity,
            "controller_backend_is_exact": backend_is_exact,
            "exact_oracle_conformance_status": (
                None if conformance is None else conformance.status
            ),
            "exact_oracle_identity_bound_to_backend": identity_bound,
            "exact_oracle_fixture_bound_to_source_bag": (
                conformance_fixture_bound
            ),
            "exact_controller_gate_passed": exact_gate_passed,
            "probability_calibration_gate_passed": calibration_gate_passed,
            "integrator_state_gate_passed": integrator_state_gate_passed,
            "probability_calibration_binding": calibration_binding,
            "probability_calibration_report_sha256": (
                None
                if calibration is None
                else calibration.content_sha256
            ),
            "recommendation_threshold": config.recommendation_threshold,
            "dependence_handling": coupling,
            "dependence_diagnostic": (
                "NONE"
                if coupling == DEPENDENCE_JOINT_SAMPLES
                else DEPENDENCE_APPROXIMATED
            ),
            "controller_integrator_state_sources": tuple(
                item.integrator_state_source for item in initials
            ),
            "seed": int(config.seed),
            "analysis_mode": config.analysis_mode,
            "prefix_cutoff": config.prefix_cutoff,
            "inference_data_end_time": config.inference_data_end_time,
            "trajectory_sample_ids": tuple(
                int(item.sample_id) for item in initials
            ),
            "response_posterior_approximation": response_posterior.approximation,
            "counterfactual_content_hash": content_hash,
            "workflow_status": workflow_status,
        }
        run_id = content_hash[:20]
        return CounterfactualResult(
            candidate=candidate,
            success_probability=success_probability,
            credible_lower=lower,
            credible_upper=upper,
            lower_credible_bound=lower,
            support=support,
            violation_probability=violation_probability,
            effective_rollout_sample_size=effective_count,
            rollouts=rollouts,
            run_id=run_id,
            provenance=provenance,
            recommendation_threshold=config.recommendation_threshold,
            exact_controller_gate_passed=exact_gate_passed,
            probability_calibration_gate_passed=calibration_gate_passed,
            integrator_state_gate_passed=integrator_state_gate_passed,
            dependence_handling=coupling,
            workflow_status=workflow_status,
        )


def connected_candidate_regions(
    results: Sequence[CounterfactualResult],
    gamma: float,
    neighbor_radius: float = 1.05,
) -> Tuple[Tuple[str, ...], ...]:
    """Return connected, supported regions with lower bound at least gamma."""

    threshold = float(gamma)
    radius = float(neighbor_radius)
    if not 0.0 <= threshold <= 1.0 or radius <= 0.0:
        raise ValueError("gamma or neighbor_radius is invalid")
    eligible = [
        item
        for item in results
        if item.recommendable and item.lower_credible_bound >= threshold
    ]
    if not eligible:
        return ()
    vectors = np.asarray([item.candidate.vector() for item in eligible])
    scale = np.ptp(vectors, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    distances = np.linalg.norm(
        (vectors[:, None, :] - vectors[None, :, :]) / scale,
        axis=2,
    )
    unseen = set(range(len(eligible)))
    components = []
    while unseen:
        seed = unseen.pop()
        frontier = [seed]
        component = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                index
                for index in tuple(unseen)
                if distances[current, index] <= radius
            ]
            for index in neighbours:
                unseen.remove(index)
                frontier.append(index)
                component.append(index)
        components.append(
            tuple(sorted(eligible[index].candidate.candidate_id for index in component))
        )
    return tuple(sorted(components))


__all__ = [
    "DEPENDENCE_APPROXIMATED",
    "DEPENDENCE_JOINT_SAMPLES",
    "EXPERIMENTAL",
    "EXTRAPOLATIVE",
    "MANUAL_REVIEW_REQUIRED",
    "SUPPORTED",
    "UNSUPPORTED",
    "ClosedLoopCounterfactualEvaluator",
    "CounterfactualCandidate",
    "CounterfactualConfig",
    "CounterfactualResult",
    "InitialStateSample",
    "JointPosteriorSample",
    "PythonControllerReplayBackend",
    "SupportDiagnostics",
    "SupportReference",
    "TargetTrajectory",
    "TargetTube",
    "TrajectoryRollout",
    "TubeEvaluation",
    "classify_support",
    "connected_candidate_regions",
    "evaluate_target_tube",
]
