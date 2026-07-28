"""Production open-loop and exact-gated closed-loop plant assimilation.

Recorded-command inference remains the available path for current bags.
Closed-loop inference requires injected, hash-bound controller fixtures,
snapshots, a passing exact factual gate, and a state-continuous controller
backend; none of those inputs is synthesized from incomplete bag evidence.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation, Slerp
import yaml

from grape_param_estim.controller import (
    ControllerBackend,
    ControllerBackendIdentity,
    ControllerCoreState,
    ControllerSnapshot,
    ExactClosedLoopGateReport,
    evaluate_exact_closed_loop_gate,
)
from grape_param_estim.controller.exact_inputs import (
    ExactEpisodeConformanceBundle,
)
from grape_param_estim.controller.external_oracle import (
    batched_exact_controller_backend_factory,
)
from grape_param_estim.data import (
    ControllerReplayFixture,
    EpisodeTimeGrids,
    EventGrid,
    audit_controller_replay_inventory,
    build_replay_audit_bundle,
    initial_state_posterior,
    read_bag_topic_inventory,
    read_forward_episode,
    with_disturbance_samples,
)
from grape_param_estim.episode import stable_hash
from grape_param_estim.forward import (
    ClosedLoopForwardModel,
    ClosedLoopGateError,
    OpenLoopForwardModel,
)
from grape_param_estim.grape_geometry import (
    validate_fixed_geometry_declaration,
)
from grape_param_estim.inference import (
    BatchPlantInference,
    BoundedLogitTransform,
    ControllerEventObservations,
    EpisodeLikelihood,
    IndependentBoundedPrior,
    LikelihoodConfig,
    MultipleEpisodeLikelihood,
    ObservationDataset,
    PriorDimension,
    RolloutCache,
    RolloutCacheKey,
    TemperedSmcConfig,
    episode_excitation_report,
    local_identifiability,
)
from grape_param_estim.output import (
    PlantAssimilationArtifactWriter,
    PlantRunProvenance,
    plain_data,
)
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
    PlantHypothesis,
    effective_identifiable_quantities,
)
from grape_param_estim.plant.actuator import FirstOrderActuatorBackend
from grape_param_estim.state_smoother import (
    SmootherConfig,
    TrajectoryObservations,
    smooth_trajectory,
)
from grape_param_estim.validation import (
    FailureEvent,
    RolloutSafetyFailureDetector,
    SuccessGateConfig,
    censor_after_failure,
    evaluate_success_episode,
    evaluate_success_gate,
    validate_posterior_predictive,
)


SCHEMA = "grape_plant_assimilation/v2"
_ALLOWED_ROLES = (
    "inference_failure",
    "validation_failure",
    "validation_success",
)
OPEN_LOOP_POSTERIOR_MODEL_ID = (
    "open_loop_plant_identification/{}".format(
        EFFECTIVE_CLOSED_LOOP_MODEL_ID
    )
)
CLOSED_LOOP_POSTERIOR_MODEL_ID = (
    "closed_loop_plant_identification/{}".format(
        EFFECTIVE_CLOSED_LOOP_MODEL_ID
    )
)
OPEN_LOOP_PLANT_BACKEND_ID = "open_loop_effective_forward_v1"
CLOSED_LOOP_PLANT_BACKEND_ID = "closed_loop_exact_effective_forward_v1"
EFFECTIVE_DISTURBANCE_MODEL_ID = (
    "effective_constant_acceleration_disturbance_v1"
)
EFFECTIVE_DISTURBANCE_PARAMETER_NAMES = (
    "wind_acceleration_world_x",
    "wind_acceleration_world_y",
    "wind_acceleration_world_z",
    "angular_acceleration_bias_body_x",
    "angular_acceleration_bias_body_y",
    "angular_acceleration_bias_body_z",
)


@dataclass(frozen=True)
class ExactClosedLoopDependencies:
    """Injected, hash-bound dependencies required for exact closed-loop SMC.

    The factory must return a fresh stateful controller for every rollout.
    In particular, the batch subprocess oracle is not sufficient unless an
    adapter restores its returned controller state before the next tick.
    """

    controller_backend_factory: Callable[[], ControllerBackend]
    fixtures: Mapping[str, ControllerReplayFixture]
    snapshots: Mapping[str, ControllerSnapshot]
    gate_report: ExactClosedLoopGateReport
    conformance_bundle: ExactEpisodeConformanceBundle

    def __post_init__(self) -> None:
        from grape_param_estim.alternative_backends import (
            ExactOracleConformanceReport,
        )

        if not callable(self.controller_backend_factory):
            raise TypeError("controller_backend_factory must be callable")
        if not isinstance(self.gate_report, ExactClosedLoopGateReport):
            raise TypeError(
                "gate_report must be ExactClosedLoopGateReport"
            )
        if not isinstance(
            self.conformance_bundle, ExactEpisodeConformanceBundle
        ):
            raise TypeError(
                "conformance_bundle must be ExactEpisodeConformanceBundle"
            )
        report = self.gate_report
        conformance = report.conformance_report
        if (
            not report.passed
            or not report.factual_replay_passed
            or report.identity is None
            or report.identity.is_exact is not True
            or report.capability_check is None
            or not report.capability_check.passed
            or not isinstance(
                conformance, ExactOracleConformanceReport
            )
            or not conformance.content_is_valid()
            or conformance.evidence_sha256
            != report.factual_evidence_sha256
        ):
            raise ClosedLoopGateError(
                "exact closed-loop dependencies require a passing factual gate"
            )
        fixtures = {
            str(key): value for key, value in self.fixtures.items()
        }
        snapshots = {
            str(key): value for key, value in self.snapshots.items()
        }
        if (
            not fixtures
            or set(fixtures) != set(snapshots)
            or any(
                not isinstance(value, ControllerReplayFixture)
                for value in fixtures.values()
            )
            or any(
                not isinstance(value, ControllerSnapshot)
                for value in snapshots.values()
            )
        ):
            raise TypeError(
                "closed-loop fixtures and snapshots must be aligned typed mappings"
            )
        identity = report.identity
        try:
            self.conformance_bundle.require_bound(
                fixtures, snapshots
            )
        except (TypeError, ValueError) as exc:
            raise ClosedLoopGateError(
                "per-episode factual conformance evidence is not bound "
                "to the injected fixtures/snapshots"
            ) from exc
        representative = (
            self.conformance_bundle.representative_report
        )
        if (
            representative.evidence_sha256
            != report.factual_evidence_sha256
        ):
            raise ClosedLoopGateError(
                "aggregate exact gate must use the bundle's canonical "
                "representative episode report"
            )
        for episode_id, evidence in (
            self.conformance_bundle.episodes.items()
        ):
            episode_gate = evaluate_exact_closed_loop_gate(
                identity,
                evidence.conformance_report,
                required_fidelity=report.required_fidelity,
            )
            if (
                not episode_gate.passed
                or episode_gate.identity != identity
            ):
                raise ClosedLoopGateError(
                    "episode {} did not pass the identity-consistent "
                    "exact factual gate".format(episode_id)
                )
        for episode_id, snapshot in snapshots.items():
            if (
                snapshot.backend_id != identity.backend_id
                or snapshot.artifact_sha256
                != identity.artifact_sha256
                or snapshot.source_commit != identity.source_commit
            ):
                raise ClosedLoopGateError(
                    "episode {} snapshot does not match the gated backend".format(
                        episode_id
                    )
                )
            options = snapshot.static_options
            geometry = snapshot.nominal_geometry
            if (
                bool(options["gimbal_calc_in_fc"])
                or int(options["gimbal_dof"]) != 1
                or len(
                    geometry.get("rotor_origins_from_cog", ())
                )
                != 4
                or len(geometry.get("rotor_directions", ())) != 4
                or len(
                    geometry.get("thrust_coordinate_rotations", ())
                )
                != 4
            ):
                raise ClosedLoopGateError(
                    "episode {} exact controller boundary emits actuator "
                    "dimensions unsupported by the four-rotor, one-DOF "
                    "FirstOrderActuatorBackend".format(episode_id)
                )
        object.__setattr__(
            self,
            "fixtures",
            MappingProxyType(dict(sorted(fixtures.items()))),
        )
        object.__setattr__(
            self,
            "snapshots",
            MappingProxyType(dict(sorted(snapshots.items()))),
        )

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "gate": self.gate_report.to_mapping(),
                "fixtures": {
                    key: value.fixture_sha256
                    for key, value in self.fixtures.items()
                },
                "snapshots": {
                    key: value.snapshot_id
                    for key, value in self.snapshots.items()
                },
                "episode_conformance_bundle_sha256": (
                    self.conformance_bundle.content_sha256
                ),
            }
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_disturbance_prior(
    config: Mapping[str, Any],
) -> Tuple[str, np.ndarray, np.ndarray, int]:
    nuisance = config.get("episode_nuisance")
    if not isinstance(nuisance, Mapping):
        raise ValueError(
            "episode_nuisance must explicitly configure disturbance"
        )
    values = nuisance.get("disturbance")
    if not isinstance(values, Mapping):
        raise ValueError(
            "episode_nuisance.disturbance must be a mapping"
        )
    required = {
        "model_id",
        "parameter_names",
        "prior",
        "sample_count",
    }
    if set(values) != required:
        raise ValueError(
            "disturbance config must contain exactly {}".format(
                ", ".join(sorted(required))
            )
        )
    model_id = str(values["model_id"])
    if model_id != EFFECTIVE_DISTURBANCE_MODEL_ID:
        raise ValueError(
            "effective plant requires {}".format(
                EFFECTIVE_DISTURBANCE_MODEL_ID
            )
        )
    if tuple(values["parameter_names"]) != (
        EFFECTIVE_DISTURBANCE_PARAMETER_NAMES
    ):
        raise ValueError(
            "episode disturbance parameter order differs from its schema"
        )
    prior = values["prior"]
    if (
        not isinstance(prior, Mapping)
        or set(prior) != {"type", "lower", "upper"}
        or prior.get("type") != "bounded_uniform"
    ):
        raise ValueError(
            "episode disturbance requires an explicit bounded-uniform prior"
        )
    lower = np.asarray(prior["lower"], dtype=float).reshape(-1)
    upper = np.asarray(prior["upper"], dtype=float).reshape(-1)
    count = int(values["sample_count"])
    if (
        lower.shape != (6,)
        or upper.shape != (6,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > 0.0)
        or np.any(upper < 0.0)
        or np.any(upper <= lower)
        or count < 2
    ):
        raise ValueError(
            "disturbance prior needs six finite bounds containing zero "
            "and at least two marginalization samples"
        )
    return model_id, lower, upper, count


def _parallelism_config(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    values = config.get("inference")
    if not isinstance(values, Mapping):
        raise TypeError("inference configuration must be a mapping")
    output = {}
    for name in (
        "worker_count",
        "chain_worker_count",
        "exact_controller_batch_size",
    ):
        value = values.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
        ):
            raise TypeError(
                "inference.{} must be an integer".format(name)
            )
        if int(value) < 1:
            raise ValueError(
                "inference.{} must be positive".format(name)
            )
        output[name] = int(value)
    wait = values.get("exact_controller_batch_wait_s")
    if (
        isinstance(wait, bool)
        or not isinstance(wait, Real)
    ):
        raise TypeError(
            "inference.exact_controller_batch_wait_s "
            "must be a real number"
        )
    wait = float(wait)
    if not np.isfinite(wait) or wait < 0.0:
        raise ValueError(
            "inference.exact_controller_batch_wait_s "
            "must be finite and non-negative"
        )
    output["exact_controller_batch_wait_s"] = wait
    return MappingProxyType(output)


def _controller_event_likelihood_policy(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    values = config.get("observation")
    if not isinstance(values, Mapping):
        raise TypeError("observation configuration must be a mapping")
    required = values.get("require_controller_event_evidence")
    if type(required) is not bool:
        raise TypeError(
            "observation.require_controller_event_evidence "
            "must be a built-in bool"
        )
    closed_loop = (
        config.get("mode") == "closed_loop_plant_identification"
    )
    if closed_loop and required is not True:
        raise ValueError(
            "exact closed-loop inference must require hash-bound "
            "controller event evidence"
        )
    if not closed_loop and required is not False:
        raise ValueError(
            "current recorded-command open-loop evidence has no factual "
            "controller event frames and cannot require event scoring"
        )
    output = {
        "require_controller_event_evidence": required,
    }
    for name in (
        "saturation_event_error_probability",
        "mode_event_error_probability",
        "other_event_error_probability",
    ):
        value = values.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                "observation.{} must be a real number".format(name)
            )
        probability = float(value)
        if (
            not np.isfinite(probability)
            or not 0.0 < probability < 0.5
        ):
            raise ValueError(
                "observation.{} must lie strictly between 0 and 0.5".format(
                    name
                )
            )
        output[name] = probability
    return MappingProxyType(output)


def load_assimilation_config(path: Any) -> Tuple[Mapping[str, Any], str]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping) or loaded.get("schema") != SCHEMA:
        raise ValueError("unsupported plant-assimilation config schema")
    config = dict(loaded)
    if config.get("mode") not in (
        "open_loop_plant_identification",
        "closed_loop_plant_identification",
    ):
        raise ValueError("unsupported plant-assimilation mode")
    if config["plant"].get("model") != EFFECTIVE_CLOSED_LOOP_MODEL_ID:
        raise ValueError(
            "current production config supports effective_closed_loop_v1; "
            "physical models require an actuator-calibration artifact"
        )
    validate_fixed_geometry_declaration(
        config["plant"].get("geometry_profile")
    )
    plant_names = tuple(
        item["name"] for item in config["plant"]["parameters"]
    )
    actuator_names = tuple(
        item["name"] for item in config["plant"]["actuator_parameters"]
    )
    if plant_names != EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES:
        raise ValueError("effective plant parameter order differs from its schema")
    if actuator_names != ACTUATOR_PARAMETER_NAMES[:5]:
        raise ValueError("actuator inference parameter order differs from its schema")
    _parallelism_config(config)
    _episode_disturbance_prior(config)
    _controller_event_likelihood_policy(config)
    episodes = tuple(config.get("episodes", ()))
    if (
        not episodes
        or len({item["episode_id"] for item in episodes}) != len(episodes)
        or any(item.get("role") not in _ALLOWED_ROLES for item in episodes)
    ):
        raise ValueError("episodes require unique IDs and declared roles")
    validation = config["validation"]
    for name in (
        "failure_bags_for_inference",
        "success_bags_for_inference",
        "success_bags_for_validation",
    ):
        if type(validation.get(name)) is not bool:
            raise TypeError(
                "validation.{} must be a built-in bool".format(name)
            )
    if (
        validation.get("success_bags_for_inference") is not False
        or validation.get("success_bags_for_validation") is not True
    ):
        raise ValueError("success bags must be held out from inference")
    if not any(item["role"] == "inference_failure" for item in episodes):
        raise ValueError("at least one failure episode must update the posterior")
    credible = float(validation["credible_probability"])
    coverage = float(validation["minimum_success_coverage"])
    failure_coverage = float(
        validation.get(
            "minimum_failure_trajectory_coverage", coverage
        )
    )
    false_failure = float(
        validation["maximum_success_failure_probability"]
    )
    if (
        not np.isfinite(credible)
        or not 0.0 < credible < 1.0
        or not np.isfinite(coverage)
        or not 0.0 <= coverage <= 1.0
        or not np.isfinite(failure_coverage)
        or not 0.0 <= failure_coverage <= 1.0
        or not np.isfinite(false_failure)
        or not 0.0 <= false_failure <= 1.0
    ):
        raise ValueError("validation probabilities are invalid")
    if type(config["observation"].get("censor_after_failure")) is not bool:
        raise TypeError(
            "observation.censor_after_failure must be a built-in bool"
        )
    return config, hashlib.sha256(text.encode("utf-8")).hexdigest()


def repository_source_identity(repository_root: Any) -> Tuple[str, bool]:
    root = Path(repository_root).expanduser().resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    return commit, not bool(status.strip())


def _uniform_grid(start: float, end: float, rate: float) -> np.ndarray:
    first = float(start)
    last = float(end)
    frequency = float(rate)
    if (
        not np.isfinite(first)
        or not np.isfinite(last)
        or not np.isfinite(frequency)
        or frequency <= 0.0
        or last <= first
    ):
        raise ValueError("uniform grid requires finite start < end and rate > 0")
    delta = 1.0 / frequency
    count = int(np.floor((last - first) / delta))
    values = first + np.arange(count + 1, dtype=float) * delta
    # Use the source support endpoint verbatim. Decimal rounding can move a
    # query a few ulps beyond Slerp's closed interpolation interval.
    values = values[values < last - 1.0e-12]
    return np.append(values, last)


def _interp_matrix(
    source_times: np.ndarray, source: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(query, source_times, source[:, index])
            for index in range(source.shape[1])
        ]
    )


@dataclass(frozen=True)
class PreparedEpisode:
    config: Mapping[str, Any]
    data: Any
    grids: EpisodeTimeGrids
    observations: ObservationDataset
    nuisance_samples: Tuple[Any, ...]
    commands: Any
    trajectory_posterior: Any


def prepare_episode(
    bag_root: Any,
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
) -> PreparedEpisode:
    bag_path = Path(bag_root).expanduser().resolve() / str(episode["bag"])
    data = read_forward_episode(
        bag_path,
        episode_id=episode["episode_id"],
        replay_start_offset_s=episode["replay_start_offset_s"],
        score_start_offset_s=episode["score_start_offset_s"],
        score_end_offset_s=episode["score_end_offset_s"],
        topics=config["topics"],
        source_bag_sha256=episode["source_bag_sha256"],
    )
    smoother_values = config["smoother"]
    trajectory = smooth_trajectory(
        TrajectoryObservations(
            mocap_times=data.mocap_times,
            mocap_positions_world=data.position_world,
            mocap_quaternions_xyzw=data.orientation_xyzw,
            imu_times=data.imu_times,
            accelerometer_body=data.specific_force_body,
            gyro_body=data.angular_velocity_body,
        ),
        SmootherConfig(
            trajectory_sample_count=int(
                smoother_values["trajectory_sample_count"]
            ),
            seed=int(config["seed"]),
            mocap_position_sigma=float(
                smoother_values["mocap_position_sigma"]
            ),
            mocap_orientation_sigma=float(
                smoother_values["mocap_orientation_sigma"]
            ),
            accelerometer_noise_sigma=float(
                smoother_values["accelerometer_noise_sigma"]
            ),
            gyro_noise_sigma=float(smoother_values["gyro_noise_sigma"]),
        ),
    )
    closed_loop = (
        config["mode"] == "closed_loop_plant_identification"
    )
    support_starts = [
        float(episode["replay_start_offset_s"]),
        float(trajectory.timestamps[0]),
    ]
    support_ends = [
        float(episode["score_end_offset_s"]),
        float(trajectory.timestamps[-1]),
    ]
    if not closed_loop:
        support_starts.append(float(data.command_times[0]))
        support_ends.append(float(data.command_times[-1]))
    start = max(support_starts)
    end = min(support_ends)
    score_start = max(float(episode["score_start_offset_s"]), start)
    if end <= score_start:
        raise ValueError(
            "episode has no common scored observation support"
        )
    uniform_integration = _uniform_grid(
        start, end, float(config["plant"]["integration_rate_hz"])
    )
    likelihood_times = _uniform_grid(
        score_start,
        end,
        float(config["observation"]["likelihood_rate_hz"]),
    )
    report_times = _uniform_grid(
        score_start,
        end,
        float(config["observation"]["report_rate_hz"]),
    )
    controller_times = (
        np.asarray((start, end), dtype=float)
        if closed_loop
        else data.command_times[
            (data.command_times >= start)
            & (data.command_times <= end)
        ]
    )
    # Controller ticks are mandatory plant-integration events.  A nominal
    # 100 Hz grid alone cannot represent irregular hardware timestamps
    # exactly, and snapping/dropping them would change PID/feedback ordering.
    integration = (
        uniform_integration
        if closed_loop
        else np.unique(
            np.concatenate(
                (uniform_integration, controller_times)
            )
        )
    )
    observation_times = trajectory.timestamps[
        (trajectory.timestamps >= start) & (trajectory.timestamps <= end)
    ]
    grids = EpisodeTimeGrids(
        controller_tick_grid=EventGrid(
            "controller_tick", tuple(controller_times)
        ),
        plant_integration_grid=EventGrid(
            "plant_integration", tuple(integration)
        ),
        observation_grid=EventGrid(
            "observation", tuple(observation_times)
        ),
        likelihood_grid=EventGrid(
            "likelihood", tuple(likelihood_times)
        ),
        report_grid=EventGrid("report", tuple(report_times)),
    )
    position = _interp_matrix(
        trajectory.timestamps, trajectory.position_world, likelihood_times
    )
    velocity = _interp_matrix(
        trajectory.timestamps, trajectory.velocity_world, likelihood_times
    )
    orientation = Slerp(
        trajectory.timestamps,
        Rotation.from_quat(trajectory.quaternion_xyzw),
    )(likelihood_times).as_quat()
    specific = _interp_matrix(
        data.imu_times, data.specific_force_body, likelihood_times
    )
    angular = _interp_matrix(
        data.imu_times, data.angular_velocity_body, likelihood_times
    )
    failure_time = episode.get("failure_time_offset_s")
    # If the observed event occurs just after the requested score endpoint,
    # it still supplies the event likelihood while post-event state remains
    # censored.
    observations = ObservationDataset(
        episode_id=episode["episode_id"],
        role=episode["role"],
        timestamps=likelihood_times,
        position_world=position,
        orientation_xyzw=orientation,
        velocity_world=velocity,
        specific_force_body=specific,
        angular_velocity_body=angular,
        failure_time=failure_time,
        failure_type=episode.get("failure_type"),
        source_bag_sha256=data.source_bag_sha256,
        normalized_episode_sha256=data.normalized_episode_sha256,
    )
    initial = initial_state_posterior(
        episode["episode_id"],
        trajectory,
        start,
        maximum_samples=int(
            smoother_values["initial_state_sample_count"]
        ),
    )
    (
        disturbance_model_id,
        disturbance_lower,
        disturbance_upper,
        disturbance_sample_count,
    ) = _episode_disturbance_prior(config)
    nuisance_seed = int(
        stable_hash(
            {
                "schema": "grape_episode_disturbance_sampling_seed/v1",
                "global_seed": int(config["seed"]),
                "episode_id": str(episode["episode_id"]),
                "model_id": disturbance_model_id,
            }
        )[:16],
        16,
    )
    initial = with_disturbance_samples(
        initial,
        model_id=disturbance_model_id,
        lower=disturbance_lower,
        upper=disturbance_upper,
        sample_count=disturbance_sample_count,
        seed=nuisance_seed,
    )
    return PreparedEpisode(
        config=dict(episode),
        data=data,
        grids=grids,
        observations=observations,
        nuisance_samples=initial.samples,
        commands=data.recorded_commands(),
        trajectory_posterior=trajectory,
    )


def prepare_episodes(
    bag_root: Any, config: Mapping[str, Any]
) -> Mapping[str, PreparedEpisode]:
    return {
        item["episode_id"]: prepare_episode(bag_root, item, config)
        for item in config["episodes"]
    }


def _prior(config: Mapping[str, Any]) -> IndependentBoundedPrior:
    rows = tuple(config["plant"]["parameters"]) + tuple(
        config["plant"]["actuator_parameters"]
    )
    return IndependentBoundedPrior(
        tuple(
            PriorDimension(
                name=row["name"],
                kind=row["prior"]["type"],
                lower=row["prior"]["lower"],
                upper=row["prior"]["upper"],
            )
            for row in rows
        )
    )


def _hypothesis_builder(config: Mapping[str, Any]):
    plant_count = len(EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES)
    fixed = config["plant"]["fixed_actuator_parameters"]

    def build(vector: np.ndarray) -> PlantHypothesis:
        values = np.asarray(vector, dtype=float).reshape(-1)
        plant = values[:plant_count]
        inferred_actuator = values[plant_count:]
        actuator = np.concatenate(
            (
                inferred_actuator,
                [
                    float(fixed["minimum_thrust"]),
                    float(fixed["maximum_thrust"]),
                ],
            )
        )
        return PlantHypothesis(
            model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            plant_parameters=plant,
            actuator_parameters=actuator,
            disturbance_parameters=np.zeros(0),
            plant_parameter_names=EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
            actuator_parameter_names=ACTUATOR_PARAMETER_NAMES,
            derived_quantities=effective_identifiable_quantities(
                plant, actuator
            ),
        )

    return build


def _controller_snapshot_placeholder(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = {
        "schema": "grape.controller-snapshot-unavailable/v1",
        "status": "UNAVAILABLE_FROM_CURRENT_BAG",
        "backend_id": config["controller"]["backend"],
        "requested_fidelity": config["controller"]["fidelity"],
        "snapshot_policy": config["controller"]["snapshot_policy"],
        "nominal_model_policy": config["controller"]["nominal_model_policy"],
        "used_by_open_loop_rollout": False,
        "plant_geometry_profile": dict(
            config["plant"]["geometry_profile"]
        ),
        "reason": (
            "complete source/model/parameter/controller-state identity is "
            "not present in current bags"
        ),
    }
    return dict(base, snapshot_id=stable_hash(base))


def _controller_snapshot_bundle(
    dependencies: ExactClosedLoopDependencies,
) -> Mapping[str, Any]:
    snapshots = {
        episode_id: dict(
            snapshot.to_mapping(),
            snapshot_id=snapshot.snapshot_id,
        )
        for episode_id, snapshot in dependencies.snapshots.items()
    }
    base = {
        "schema": "grape.controller-snapshot-bundle/v1",
        "status": "HASH_BOUND_EXACT",
        "gate_report_sha256": dependencies.gate_report.content_sha256,
        "episode_conformance_bundle_sha256": (
            dependencies.conformance_bundle.content_sha256
        ),
        "snapshots": snapshots,
    }
    return dict(base, snapshot_id=stable_hash(base))


def _same_grid(left: EventGrid, right: EventGrid) -> bool:
    return (
        isinstance(left, EventGrid)
        and isinstance(right, EventGrid)
        and left.name == right.name
        and left.timestamps == right.timestamps
    )


def _verified_controller(
    factory: Callable[[], ControllerBackend],
    expected_identity: ControllerBackendIdentity,
) -> ControllerBackend:
    backend = factory()
    if (
        not isinstance(backend, ControllerBackend)
        or not isinstance(
            getattr(backend, "identity", None),
            ControllerBackendIdentity,
        )
        or backend.identity != expected_identity
    ):
        raise ClosedLoopGateError(
            "controller backend factory must return a stateful backend "
            "matching the exact factual gate identity"
        )
    return backend


def _validate_closed_loop_dependencies(
    config: Mapping[str, Any],
    prepared: Mapping[str, PreparedEpisode],
    dependencies: Optional[ExactClosedLoopDependencies],
) -> ExactClosedLoopDependencies:
    if not isinstance(dependencies, ExactClosedLoopDependencies):
        raise ClosedLoopGateError(
            "closed-loop mode requires injected exact fixtures, snapshots, "
            "a stateful controller backend factory, and a passing factual gate"
        )
    controller_config = config["controller"]
    report = dependencies.gate_report
    conformance = report.conformance_report
    identity = report.identity
    if (
        conformance is None
        or not conformance.content_is_valid()
        or conformance.evidence_sha256
        != report.factual_evidence_sha256
        or controller_config.get("require_factual_replay_pass") is not True
        or controller_config.get("snapshot_policy")
        == "unavailable_from_current_bag"
        or controller_config.get("nominal_model_policy") != "frozen"
        or controller_config.get("backend") != identity.backend_id
        or controller_config.get("fidelity") != report.required_fidelity
    ):
        raise ClosedLoopGateError(
            "closed-loop controller config is not bound to the injected exact gate"
        )
    episode_ids = set(prepared)
    if (
        not episode_ids
        or set(dependencies.fixtures) != episode_ids
        or set(dependencies.snapshots) != episode_ids
    ):
        raise ClosedLoopGateError(
            "every prepared episode requires one exact fixture and snapshot"
        )
    grid_names = (
        "controller_tick_grid",
        "plant_integration_grid",
        "observation_grid",
        "likelihood_grid",
        "report_grid",
    )
    for episode_id, episode in prepared.items():
        fixture = dependencies.fixtures[episode_id]
        if (
            fixture.episode_id != episode_id
            or fixture.source_bag_sha256
            != episode.observations.source_bag_sha256
            or not fixture.controller_inputs
            or any(
                not _same_grid(
                    getattr(fixture.grids, name),
                    getattr(episode.grids, name),
                )
                for name in grid_names
            )
        ):
            raise ClosedLoopGateError(
                "episode {} exact fixture is not aligned with prepared evidence".format(
                    episode_id
                )
            )
        if (
            not episode.nuisance_samples
            or any(
                not isinstance(item.controller_state, ControllerCoreState)
                for item in episode.nuisance_samples
            )
        ):
            raise ClosedLoopGateError(
                "episode {} lacks reconstructable controller state".format(
                    episode_id
                )
            )
    first_probe = _verified_controller(
        dependencies.controller_backend_factory, identity
    )
    second_probe = _verified_controller(
        dependencies.controller_backend_factory, identity
    )
    if first_probe is second_probe:
        raise ClosedLoopGateError(
            "controller backend factory must return fresh per-rollout state"
        )
    first_transport = getattr(first_probe, "transport", None)
    second_transport = getattr(second_probe, "transport", None)
    if (
        (
            getattr(
                first_transport,
                "transport_is_persistent",
                False,
            )
            or getattr(
                second_transport,
                "transport_is_persistent",
                False,
            )
        )
        and first_transport is not second_transport
    ):
        raise ClosedLoopGateError(
            "persistent controller backends must share one oracle process"
        )
    for probe in (first_probe, second_probe):
        close = getattr(probe, "close", None)
        if callable(close):
            close()
    return dependencies


def _same_controller_event_observations(
    left: ControllerEventObservations,
    right: ControllerEventObservations,
) -> bool:
    return bool(
        isinstance(left, ControllerEventObservations)
        and isinstance(right, ControllerEventObservations)
        and left.schema == right.schema
        and left.mode_event_mask == right.mode_event_mask
        and left.saturation_event_mask
        == right.saturation_event_mask
        and np.array_equal(left.timestamps, right.timestamps)
        and np.array_equal(
            left.event_bitmasks, right.event_bitmasks
        )
        and np.array_equal(left.saturated, right.saturated)
    )


def _bind_exact_controller_event_observations(
    prepared: Mapping[str, PreparedEpisode],
    dependencies: ExactClosedLoopDependencies,
) -> Mapping[str, PreparedEpisode]:
    """Bind scored ReplayFrame events from the passing exact evidence.

    The conformance fixture is the recorded controller-output source.  The
    ordinary command topics used by open-loop inference do not carry these
    event masks and are never promoted into negative event observations.
    Pre-roll ticks reconstruct controller state but are excluded because the
    first observation timestamp is the configured likelihood score start.
    """

    output = {}
    for episode_id, episode in sorted(prepared.items()):
        fixture = dependencies.fixtures[episode_id]
        snapshot = dependencies.snapshots[episode_id]
        evidence = dependencies.conformance_bundle.episodes[
            episode_id
        ]
        if (
            not evidence.binds(fixture, snapshot)
            or evidence.source_bag_sha256
            != episode.observations.source_bag_sha256
        ):
            raise ClosedLoopGateError(
                "episode {} event evidence is not bound to the runtime "
                "fixture, snapshot, and source bag".format(episode_id)
            )
        conformance_fixture = evidence.conformance_fixture
        recorded_times = np.asarray(
            conformance_fixture.continuous.get(
                "command_timestamp"
            ),
            dtype=float,
        )
        factual_input_times = np.asarray(
            [
                float(getattr(item, "stamp"))
                for item in fixture.controller_inputs
            ],
            dtype=float,
        )
        recorded_events = np.asarray(conformance_fixture.events)
        if (
            recorded_times.shape != (factual_input_times.size, 1)
            or recorded_events.shape
            != (factual_input_times.size,)
            or not np.array_equal(
                recorded_times[:, 0], factual_input_times
            )
        ):
            raise ClosedLoopGateError(
                "episode {} factual controller events are not exactly "
                "aligned with the hash-bound replay inputs".format(
                    episode_id
                )
            )
        controller_times = np.asarray(
            fixture.grids.controller_tick_grid.timestamps,
            dtype=float,
        )
        likelihood_horizon = np.asarray(
            fixture.grids.likelihood_grid.timestamps,
            dtype=float,
        )
        score_times = controller_times[
            (controller_times >= likelihood_horizon[0])
            & (controller_times <= likelihood_horizon[-1])
        ]
        if score_times.size == 0:
            raise ClosedLoopGateError(
                "episode {} has no controller event frames in the "
                "likelihood score interval".format(episode_id)
            )
        indexes = np.searchsorted(
            factual_input_times, score_times, side="left"
        )
        if (
            np.any(indexes >= factual_input_times.size)
            or not np.array_equal(
                factual_input_times[indexes], score_times
            )
        ):
            raise ClosedLoopGateError(
                "episode {} scored controller ticks are absent from the "
                "factual conformance fixture".format(episode_id)
            )
        expected = ControllerEventObservations(
            timestamps=score_times,
            event_bitmasks=recorded_events[indexes],
        )
        existing = episode.observations.event_observations
        if (
            existing is not None
            and not _same_controller_event_observations(
                existing, expected
            )
        ):
            raise ClosedLoopGateError(
                "episode {} carries controller event observations that "
                "differ from the exact conformance evidence".format(
                    episode_id
                )
            )
        observations = replace(
            episode.observations,
            event_observations=expected,
        )
        output[episode_id] = replace(
            episode, observations=observations
        )
    return MappingProxyType(output)


def _rollout_system(
    config: Mapping[str, Any],
    prepared: Mapping[str, PreparedEpisode],
    source_commit: str,
):
    hypothesis_builder = _hypothesis_builder(config)
    geometry_sha256 = config["plant"]["geometry_profile"][
        "profile_sha256"
    ]
    forward_model = OpenLoopForwardModel(
        actuator_factory=lambda: FirstOrderActuatorBackend(
            geometry_sha256
        )
    )
    snapshot = _controller_snapshot_placeholder(config)
    snapshot_hash = snapshot["snapshot_id"]
    controller_artifact_hash = stable_hash(
        {"status": "not_invoked_in_open_loop"}
    )
    cache = RolloutCache(
        int(config["inference"].get("rollout_cache_entries", 8192))
    )

    def rollout(
        particle: np.ndarray,
        observations: ObservationDataset,
        nuisance: Any,
    ):
        episode = prepared[observations.episode_id]
        key = RolloutCacheKey(
            source_bag_sha256=observations.source_bag_sha256,
            normalized_episode_sha256=observations.normalized_episode_sha256,
            controller_snapshot_sha256=snapshot_hash,
            controller_artifact_sha256=controller_artifact_hash,
            plant_backend_model_id=OPEN_LOOP_PLANT_BACKEND_ID,
            parameter_vector_sha256=stable_hash(particle),
            initial_state_sample_id=nuisance.state_sample_id,
            process_noise_seed=int(config["seed"]),
            source_commit=source_commit,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = forward_model.run(
            episode.commands,
            hypothesis_builder(particle),
            nuisance,
            episode.grids,
        )
        cache.put(key, result)
        return result

    return hypothesis_builder, rollout, snapshot, controller_artifact_hash, cache


def _closed_loop_rollout_system(
    config: Mapping[str, Any],
    prepared: Mapping[str, PreparedEpisode],
    source_commit: str,
    dependencies: ExactClosedLoopDependencies,
):
    identity = dependencies.gate_report.identity
    parallelism = _parallelism_config(config)
    batched_backend_factory = (
        batched_exact_controller_backend_factory(
            dependencies.controller_backend_factory,
            max_batch_size=parallelism[
                "exact_controller_batch_size"
            ],
            batch_wait_s=parallelism[
                "exact_controller_batch_wait_s"
            ],
        )
    )

    def controller_factory():
        return _verified_controller(
            batched_backend_factory, identity
        )

    hypothesis_builder = _hypothesis_builder(config)
    forward_model = ClosedLoopForwardModel(
        controller_factory,
        dependencies.gate_report,
        actuator_factory=lambda: FirstOrderActuatorBackend(
            config["plant"]["geometry_profile"]["profile_sha256"]
        ),
    )
    snapshot = _controller_snapshot_bundle(dependencies)
    controller_artifact_hash = identity.artifact_sha256
    cache = RolloutCache(
        int(config["inference"].get("rollout_cache_entries", 8192))
    )

    def rollout(
        particle: np.ndarray,
        observations: ObservationDataset,
        nuisance: Any,
    ):
        episode_id = observations.episode_id
        episode = prepared[episode_id]
        fixture = dependencies.fixtures[episode_id]
        episode_snapshot = dependencies.snapshots[episode_id]
        evidence_hash = stable_hash(
            {
                "normalized_episode_sha256": (
                    observations.normalized_episode_sha256
                ),
                "controller_fixture_sha256": fixture.fixture_sha256,
                "exact_gate_sha256": (
                    dependencies.gate_report.content_sha256
                ),
                "episode_conformance_evidence_sha256": (
                    dependencies.conformance_bundle.episodes[
                        episode_id
                    ].content_sha256
                ),
            }
        )
        key = RolloutCacheKey(
            source_bag_sha256=observations.source_bag_sha256,
            normalized_episode_sha256=evidence_hash,
            controller_snapshot_sha256=episode_snapshot.snapshot_id,
            controller_artifact_sha256=controller_artifact_hash,
            plant_backend_model_id=CLOSED_LOOP_PLANT_BACKEND_ID,
            parameter_vector_sha256=stable_hash(particle),
            initial_state_sample_id=nuisance.state_sample_id,
            process_noise_seed=int(config["seed"]),
            source_commit=source_commit,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = forward_model.run(
            fixture=fixture,
            snapshot=episode_snapshot,
            hypothesis=hypothesis_builder(particle),
            nuisance=nuisance,
            grids=fixture.grids,
        )
        cache.put(key, result)
        return result

    return hypothesis_builder, rollout, snapshot, controller_artifact_hash, cache


def _likelihood_config(config: Mapping[str, Any]) -> LikelihoodConfig:
    values = config["observation"]
    event_policy = _controller_event_likelihood_policy(config)
    return LikelihoodConfig(
        position_sigma=float(values["position_sigma"]),
        orientation_sigma_rad=float(values["orientation_sigma_rad"]),
        velocity_sigma=float(values["velocity_sigma"]),
        imu_sigma=float(values["imu_sigma"]),
        angular_velocity_sigma=float(values["angular_velocity_sigma"]),
        student_t_degrees_of_freedom=float(
            values["student_t_degrees_of_freedom"]
        ),
        saturation_event_error_probability=event_policy[
            "saturation_event_error_probability"
        ],
        mode_event_error_probability=event_policy[
            "mode_event_error_probability"
        ],
        other_event_error_probability=event_policy[
            "other_event_error_probability"
        ],
        require_controller_event_evidence=event_policy[
            "require_controller_event_evidence"
        ],
        censor_after_failure=bool(values["censor_after_failure"]),
    )


def infer_plant_posterior(
    config: Mapping[str, Any],
    prepared: Mapping[str, PreparedEpisode],
    source_commit: str,
    closed_loop_dependencies: Optional[
        ExactClosedLoopDependencies
    ] = None,
):
    parallelism = _parallelism_config(config)
    closed_loop = (
        config["mode"] == "closed_loop_plant_identification"
    )
    if not closed_loop and closed_loop_dependencies is not None:
        raise ValueError(
            "exact closed-loop dependencies cannot be ignored in open-loop mode"
        )
    if not closed_loop and any(
        getattr(item.observations, "event_observations", None) is not None
        for item in prepared.values()
    ):
        raise ValueError(
            "recorded-command open-loop inference cannot accept "
            "controller replay event observations"
        )
    if closed_loop:
        closed_loop_dependencies = (
            _validate_closed_loop_dependencies(
                config, prepared, closed_loop_dependencies
            )
        )
        prepared = _bind_exact_controller_event_observations(
            prepared, closed_loop_dependencies
        )
    inference_episodes = tuple(
        item.observations
        for item in prepared.values()
        if item.observations.role == "inference_failure"
    )
    nuisance = {
        item.observations.episode_id: item.nuisance_samples
        for item in prepared.values()
        if item.observations.role == "inference_failure"
    }
    (
        hypothesis_builder,
        rollout,
        snapshot,
        controller_artifact_hash,
        cache,
    ) = (
        _rollout_system(config, prepared, source_commit)
        if not closed_loop
        else _closed_loop_rollout_system(
            config,
            prepared,
            source_commit,
            closed_loop_dependencies,
        )
    )
    component_likelihood = EpisodeLikelihood(
        _likelihood_config(config),
        failure_detector=RolloutSafetyFailureDetector(),
    )
    multiple = MultipleEpisodeLikelihood(
        inference_episodes,
        nuisance,
        rollout,
        component_likelihood,
        worker_count=parallelism["worker_count"],
    )
    prior = _prior(config)
    inference_values = config["inference"]
    smc_config = TemperedSmcConfig(
        particle_count=int(inference_values["particle_count"]),
        target_ess_fraction=float(
            inference_values["target_ess_fraction"]
        ),
        resample_ess_fraction=float(
            inference_values["resample_ess_fraction"]
        ),
        mcmc_steps=int(inference_values["mcmc_steps"]),
        proposal_scale=float(inference_values["proposal_scale"]),
        seed=int(config["seed"]),
    )
    prior_id = "grape_effective_bounded_prior/{}".format(
        stable_hash(
            {
                "names": prior.names,
                "lower": prior.lower,
                "upper": prior.upper,
                "spec": [
                    plain_data(item) for item in prior.dimensions_spec
                ],
            }
        )
    )
    posterior = BatchPlantInference(
        prior=prior,
        transform=BoundedLogitTransform(prior.lower, prior.upper),
        hypothesis_builder=hypothesis_builder,
        log_likelihood=multiple,
        config=smc_config,
        chain_count=int(inference_values["chain_count"]),
        chain_worker_count=parallelism[
            "chain_worker_count"
        ],
    ).run(
        model_id=(
            OPEN_LOOP_POSTERIOR_MODEL_ID
            if config["mode"] == "open_loop_plant_identification"
            else CLOSED_LOOP_POSTERIOR_MODEL_ID
        ),
        prior_id=prior_id,
        likelihood_id=component_likelihood.config.likelihood_id,
        controller_snapshot_id=snapshot["snapshot_id"],
        credible_probability=float(
            config["validation"]["credible_probability"]
        ),
        provenance={
            "seed": int(config["seed"]),
            "source_commit": source_commit,
            "success_bags_used_for_inference": False,
            "rollout_mode": config["mode"],
            "plant_geometry_profile_id": config["plant"][
                "geometry_profile"
            ]["profile_id"],
            "plant_geometry_sha256": config["plant"][
                "geometry_profile"
            ]["profile_sha256"],
            "plant_geometry_evidence_status": config["plant"][
                "geometry_profile"
            ]["evidence_status"],
            "runtime_override": dict(config.get("runtime_override", {})),
            "parallelism": dict(parallelism),
        },
    )
    return (
        posterior,
        hypothesis_builder,
        rollout,
        component_likelihood,
        snapshot,
        controller_artifact_hash,
        cache,
        prior,
    )


def _prediction_at(rollout: Any, query: np.ndarray) -> np.ndarray:
    return _interp_matrix(
        rollout.integration_timestamps, rollout.positions, query
    )


def validate_posterior(
    posterior: Any,
    prepared: Mapping[str, PreparedEpisode],
    rollout: Any,
    component_likelihood: EpisodeLikelihood,
    validation_config: Optional[Mapping[str, Any]] = None,
    closed_loop_dependencies: Optional[
        ExactClosedLoopDependencies
    ] = None,
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, np.ndarray],
    Sequence[Mapping[str, Any]],
]:
    requires_controller_events = bool(
        getattr(
            getattr(component_likelihood, "config", None),
            "require_controller_event_evidence",
            False,
        )
    )
    if requires_controller_events:
        if not isinstance(
            closed_loop_dependencies, ExactClosedLoopDependencies
        ):
            raise ClosedLoopGateError(
                "exact validation requires the same hash-bound "
                "controller event evidence as inference"
            )
        prepared = _bind_exact_controller_event_observations(
            prepared, closed_loop_dependencies
        )
    elif closed_loop_dependencies is not None:
        raise ValueError(
            "exact closed-loop dependencies cannot be supplied to an "
            "event-unscored validation likelihood"
        )
    elif any(
        getattr(item.observations, "event_observations", None) is not None
        for item in prepared.values()
    ):
        raise ValueError(
            "event-unscored validation cannot accept controller replay "
            "event observations"
        )
    detector = RolloutSafetyFailureDetector()
    success_config = (
        SuccessGateConfig(
            credible_probability=posterior.credible_probability
        )
        if validation_config is None
        else SuccessGateConfig(
            credible_probability=posterior.credible_probability,
            minimum_trajectory_coverage=float(
                validation_config["minimum_success_coverage"]
            ),
            maximum_false_failure_probability=float(
                validation_config[
                    "maximum_success_failure_probability"
                ]
            ),
        )
    )
    failure_coverage = (
        success_config.minimum_trajectory_coverage
        if validation_config is None
        else float(
            validation_config.get(
                "minimum_failure_trajectory_coverage",
                validation_config["minimum_success_coverage"],
            )
        )
    )
    predictive: Dict[str, np.ndarray] = {}
    likelihood_rows = []
    failure_reports = []
    success_reports = []
    for episode_id, episode in prepared.items():
        trajectories = []
        failures = []
        joint_weights = []
        particle_indexes = []
        nuisance_indexes = []
        nuisance_ids = []
        disturbance_parameters = []
        disturbance_model_ids = []
        nuisance_weights = np.asarray(
            [item.weight for item in episode.nuisance_samples],
            dtype=float,
        )
        if (
            nuisance_weights.size == 0
            or not np.all(np.isfinite(nuisance_weights))
            or np.any(nuisance_weights <= 0.0)
        ):
            raise ValueError(
                "posterior prediction requires positive nuisance samples"
            )
        nuisance_weights /= np.sum(nuisance_weights)
        for particle_index, hypothesis in enumerate(posterior.particles):
            # The raw SMC vector excludes the two fixed actuator bounds.
            raw = np.concatenate(
                (
                    hypothesis.plant_parameters,
                    hypothesis.actuator_parameters[:5],
                )
            )
            particle_rows = []
            particle_component_totals = []
            particle_start = len(trajectories)
            for nuisance_index, nuisance in enumerate(
                episode.nuisance_samples
            ):
                result = rollout(raw, episode.observations, nuisance)
                trajectories.append(
                    _prediction_at(
                        result, episode.observations.timestamps
                    )
                )
                failures.append(detector.detect(result))
                particle_indexes.append(particle_index)
                nuisance_indexes.append(nuisance_index)
                nuisance_ids.append(nuisance.state_sample_id)
                disturbance_parameters.append(
                    nuisance.disturbance_parameters
                )
                disturbance_model_ids.append(
                    nuisance.disturbance_model_id
                )
                if episode.observations.role == "inference_failure":
                    components = component_likelihood.evaluate(
                        result, episode.observations
                    )
                    particle_component_totals.append(
                        components.total
                    )
                    row = plain_data(components)
                    row["particle_index"] = particle_index
                    row["nuisance_sample_index"] = nuisance_index
                    row["initial_state_sample_id"] = (
                        nuisance.state_sample_id
                    )
                    row["disturbance_model_id"] = (
                        nuisance.disturbance_model_id
                    )
                    row["disturbance_parameters"] = list(
                        nuisance.disturbance_parameters
                    )
                    particle_rows.append(row)
            if episode.observations.role == "inference_failure":
                log_joint = np.log(nuisance_weights) + np.asarray(
                    particle_component_totals, dtype=float
                )
                conditional_weights = np.exp(
                    log_joint - logsumexp(log_joint)
                )
            else:
                conditional_weights = nuisance_weights
            for local_index, conditional_weight in enumerate(
                conditional_weights
            ):
                weight = float(
                    posterior.weights[particle_index]
                    * conditional_weight
                )
                joint_weights.append(weight)
                if particle_rows:
                    row = particle_rows[local_index]
                    row["conditional_nuisance_weight"] = float(
                        conditional_weight
                    )
                    row["normalized_weight"] = weight
                    likelihood_rows.append(row)
            if len(trajectories) - particle_start != len(
                conditional_weights
            ):
                raise RuntimeError(
                    "nuisance posterior does not align with rollouts"
                )
        values = np.stack(trajectories)
        weights = np.asarray(joint_weights, dtype=float)
        weights /= np.sum(weights)
        key = episode_id.replace("-", "_")
        predictive["{}_timestamps".format(key)] = (
            episode.observations.timestamps
        )
        predictive["{}_position_particles".format(key)] = values
        predictive["{}_weights".format(key)] = weights
        predictive["{}_particle_index".format(key)] = np.asarray(
            particle_indexes, dtype=np.int64
        )
        predictive["{}_nuisance_sample_index".format(key)] = np.asarray(
            nuisance_indexes, dtype=np.int64
        )
        predictive["{}_initial_state_sample_id".format(key)] = np.asarray(
            nuisance_ids, dtype=np.str_
        )
        predictive["{}_disturbance_parameters".format(key)] = np.asarray(
            disturbance_parameters, dtype=float
        )
        predictive["{}_disturbance_model_id".format(key)] = np.asarray(
            disturbance_model_ids, dtype=np.str_
        )
        failure_indicator = np.asarray(
            [item is not None for item in failures], dtype=np.bool_
        )
        predictive["{}_failure_indicator".format(key)] = (
            failure_indicator
        )
        predictive["{}_failure_time".format(key)] = np.asarray(
            [
                np.nan if item is None else item.failure_time
                for item in failures
            ],
            dtype=float,
        )
        predictive["{}_failure_type".format(key)] = np.asarray(
            [
                "" if item is None else item.failure_type
                for item in failures
            ],
            dtype=np.str_,
        )
        predictive["{}_failure_probability".format(key)] = np.asarray(
            float(np.sum(weights[failure_indicator]))
        )
        if episode.observations.role == "validation_failure":
            if (
                episode.observations.failure_type is None
                or episode.observations.failure_time is None
            ):
                raise ValueError(
                    "validation_failure episode requires failure type/time"
                )
            observed = FailureEvent(
                failure_type=str(episode.observations.failure_type),
                stamp=float(episode.observations.failure_time),
                detector_id="bag_metadata/v1",
            )
            censoring = censor_after_failure(
                episode.observations.timestamps,
                observed,
                include_failure_sample=True,
            )
            combined = validate_posterior_predictive(
                timestamps=episode.observations.timestamps,
                observed_trajectory=(
                    episode.observations.position_world
                ),
                predictive_trajectories=values,
                observed_failure=observed,
                predicted_failures=failures,
                weights=weights,
                credible_probability=(
                    posterior.credible_probability
                ),
                minimum_coverage_fraction=failure_coverage,
                score_mask=censoring.score_mask,
                dataset_provenance_sha256=getattr(
                    episode.observations,
                    "normalized_episode_sha256",
                    None,
                ),
            )
            reasons = []
            if not combined.trajectory.passed:
                reasons.append("trajectory:coverage")
            reasons.extend(
                "event:{}".format(item)
                for item in combined.failure.reasons
            )
            failure_reports.append(
                {
                    "episode_id": episode_id,
                    "trajectory": plain_data(combined.trajectory),
                    "event": plain_data(combined.failure),
                    "censoring": {
                        "policy": (
                            "at_or_before_observed_failure"
                        ),
                        "include_failure_sample": True,
                        "score_mask": censoring.score_mask,
                        "evaluated_sample_count": int(
                            np.count_nonzero(censoring.score_mask)
                        ),
                        "censored_sample_count": (
                            censoring.censored_count
                        ),
                    },
                    "passed": combined.passed,
                    "reasons": tuple(reasons),
                }
            )
        elif episode.observations.role == "validation_success":
            success_reports.append(
                evaluate_success_episode(
                    episode_id=episode_id,
                    role="validation_success",
                    timestamps=episode.observations.timestamps,
                    observed_trajectory=episode.observations.position_world,
                    predictive_trajectories=values,
                    predicted_failures=failures,
                    weights=weights,
                    config=success_config,
                    dataset_provenance_sha256=getattr(
                        episode.observations,
                        "normalized_episode_sha256",
                        None,
                    ),
                )
            )
    success_gate = evaluate_success_gate(success_reports)
    failure_payload = {
        "schema": "grape_failure_validation/v1",
        "held_out": [plain_data(item) for item in failure_reports],
        "config": {
            "credible_probability": posterior.credible_probability,
            "minimum_trajectory_coverage": failure_coverage,
            "trajectory_horizon": (
                "at_or_before_observed_failure"
            ),
        },
        "passed": bool(
            failure_reports
            and all(item["passed"] for item in failure_reports)
        ),
        "note": (
            "Each held-out failure must pass both the censored trajectory "
            "coverage component and the separate occurrence/type/time "
            "event component."
        ),
    }
    success_payload = {
        "schema": "grape_success_validation/v1",
        "config": plain_data(success_config),
        "episodes": [plain_data(item) for item in success_reports],
        "passed": success_gate.passed,
        "reasons": success_gate.reasons,
    }
    return failure_payload, success_payload, predictive, likelihood_rows


def _position_jacobian(
    center: np.ndarray,
    prior: IndependentBoundedPrior,
    episode: PreparedEpisode,
    nuisance: Any,
    rollout: Any,
) -> np.ndarray:
    base = rollout(center, episode.observations, nuisance)
    base_position = _prediction_at(
        base, episode.observations.timestamps
    ).reshape(-1)
    jacobian = np.empty((base_position.size, center.size))
    for parameter in range(center.size):
        step = max(
            1.0e-6,
            1.0e-4 * (prior.upper[parameter] - prior.lower[parameter]),
        )
        changed = np.array(center, copy=True)
        changed[parameter] = np.clip(
            changed[parameter] + step,
            prior.lower[parameter] + 1.0e-9,
            prior.upper[parameter] - 1.0e-9,
        )
        actual_step = changed[parameter] - center[parameter]
        if abs(actual_step) <= 1.0e-12:
            changed[parameter] = np.clip(
                center[parameter] - step,
                prior.lower[parameter] + 1.0e-9,
                prior.upper[parameter] - 1.0e-9,
            )
            actual_step = changed[parameter] - center[parameter]
        if abs(actual_step) <= 1.0e-12:
            raise ValueError(
                "cannot perturb parameter within prior bounds: {}".format(
                    prior.names[parameter]
                )
            )
        prediction = rollout(
            changed, episode.observations, nuisance
        )
        jacobian[:, parameter] = (
            _prediction_at(
                prediction, episode.observations.timestamps
            ).reshape(-1)
            - base_position
        ) / actual_step
    return jacobian


def _identifiability(
    posterior: Any,
    prior: IndependentBoundedPrior,
    prepared: Mapping[str, PreparedEpisode],
    rollout: Any,
):
    episodes = tuple(
        sorted(
            (
                item
                for item in prepared.values()
                if item.observations.role == "inference_failure"
            ),
            key=lambda item: item.observations.episode_id,
        )
    )
    if not episodes:
        raise ValueError(
            "identifiability requires at least one inference failure episode"
        )
    index = int(np.argmax(posterior.weights))
    hypothesis = posterior.particles[index]
    center = np.concatenate(
        (
            hypothesis.plant_parameters,
            hypothesis.actuator_parameters[:5],
        )
    )
    if center.shape != prior.lower.shape:
        raise ValueError(
            "identifiability center and prior dimensions are incompatible"
        )

    episode_reports = []
    episode_jacobians = []
    for episode in episodes:
        nuisance_samples = tuple(episode.nuisance_samples)
        if not nuisance_samples:
            raise ValueError(
                "identifiability episode has no nuisance samples: {}".format(
                    episode.observations.episode_id
                )
            )
        nuisance_weights = np.asarray(
            [float(item.weight) for item in nuisance_samples],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(nuisance_weights))
            or np.any(nuisance_weights <= 0.0)
        ):
            raise ValueError(
                "identifiability nuisance weights must be finite and positive"
            )
        nuisance_weights /= float(np.sum(nuisance_weights))
        weighted_jacobians = tuple(
            np.sqrt(nuisance_weights[sample_index])
            * _position_jacobian(
                center,
                prior,
                episode,
                nuisance,
                rollout,
            )
            for sample_index, nuisance in enumerate(nuisance_samples)
        )
        episode_jacobian = np.vstack(weighted_jacobians)
        episode_jacobians.append(episode_jacobian)
        episode_reports.append(
            episode_excitation_report(
                episode_jacobian,
                prior.names,
                episode.observations.episode_id,
                tuple(
                    item.state_sample_id for item in nuisance_samples
                ),
                nuisance_weights,
                structural_gauge_dimension=1,
            )
        )

    jacobian = np.vstack(episode_jacobians)
    particle_matrix = np.column_stack(
        (
            posterior.raw_parameters[
                :, : len(EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES)
            ],
            posterior.raw_parameters[
                :,
                len(EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES) : len(
                    EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES
                )
                + 5,
            ],
        )
    )
    return local_identifiability(
        jacobian,
        prior.names,
        EFFECTIVE_CLOSED_LOOP_MODEL_ID,
        structural_gauge_dimension=1,
        posterior_particles=particle_matrix,
        prior_lower=prior.lower,
        prior_upper=prior.upper,
        episode_excitation=tuple(episode_reports),
    )


def build_replay_audits(
    prepared: Mapping[str, PreparedEpisode],
) -> Mapping[str, Any]:
    audits = []
    for episode in prepared.values():
        inventory = read_bag_topic_inventory(
            episode.data.bag_path,
            source_bag_sha256=episode.data.source_bag_sha256,
        )
        audits.append(
            audit_controller_replay_inventory(
                inventory, episode_id=episode.data.episode_id
            )
        )
    return build_replay_audit_bundle(audits)


def _injected_replay_audit(
    dependencies: ExactClosedLoopDependencies,
) -> Mapping[str, Any]:
    return {
        "schema": "grape_injected_controller_replay_evidence/v1",
        "overall_exact_replay_ready": True,
        "source": "hash_bound_injected_fixtures",
        "gate_report_sha256": dependencies.gate_report.content_sha256,
        "dependency_bundle_sha256": dependencies.content_sha256,
        "episode_conformance_bundle_sha256": (
            dependencies.conformance_bundle.content_sha256
        ),
        "episodes": {
            episode_id: {
                "fixture_sha256": fixture.fixture_sha256,
                "source_bag_sha256": fixture.source_bag_sha256,
                "controller_snapshot_sha256": (
                    dependencies.snapshots[episode_id].snapshot_id
                ),
                "conformance_evidence": (
                    dependencies.conformance_bundle.episodes[
                        episode_id
                    ].to_mapping()
                ),
            }
            for episode_id, fixture in dependencies.fixtures.items()
        },
    }


def _exact_gate_payload(
    dependencies: ExactClosedLoopDependencies,
) -> Mapping[str, Any]:
    return dict(
        {
            "schema": "grape_exact_closed_loop_gate/v2",
            "closed_loop_exact_allowed": True,
            "gate_report_sha256": (
                dependencies.gate_report.content_sha256
            ),
            "episode_conformance_bundle": (
                dependencies.conformance_bundle.to_mapping()
            ),
        },
        **dependencies.gate_report.to_mapping()
    )


def write_assimilation_run(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    prepared: Mapping[str, PreparedEpisode],
    output_root: Any,
    run_id: str,
    source_commit: str,
    closed_loop_dependencies: Optional[
        ExactClosedLoopDependencies
    ] = None,
):
    closed_loop = config["mode"] == "closed_loop_plant_identification"
    if (
        not closed_loop
        and closed_loop_dependencies is not None
    ):
        raise ValueError(
            "exact closed-loop dependencies cannot be ignored in open-loop mode"
        )
    (
        posterior,
        _,
        rollout,
        component_likelihood,
        snapshot,
        controller_artifact_hash,
        _,
        prior,
    ) = infer_plant_posterior(
        config,
        prepared,
        source_commit,
        closed_loop_dependencies=closed_loop_dependencies,
    )
    (
        failure_validation,
        success_validation,
        predictive,
        likelihood_rows,
    ) = validate_posterior(
        posterior,
        prepared,
        rollout,
        component_likelihood,
        config["validation"],
        closed_loop_dependencies=closed_loop_dependencies,
    )
    audit = (
        _injected_replay_audit(closed_loop_dependencies)
        if closed_loop
        else build_replay_audits(prepared)
    )
    identifiability = _identifiability(
        posterior, prior, prepared, rollout
    )
    package_root = Path(__file__).resolve().parent
    backend_dependencies = (
        "dynamics.py",
        "grape_geometry.py",
        "plant/parameters.py",
        "plant/actuator.py",
        "plant/rigid_body.py",
        "plant/disturbance.py",
        "plant/sensor.py",
        "forward/rollout.py",
        "forward/open_loop.py",
        "data/initial_state.py",
    ) + (
        (
            "forward/closed_loop.py",
            "controller/contracts.py",
            "controller/exact_inputs.py",
            "controller/external_oracle.py",
            "controller/replay_gate.py",
            "controller/snapshot.py",
            "data/controller_fixture.py",
            "data/event_scheduler.py",
            "alternative_backends.py",
        )
        if closed_loop
        else ()
    )
    plant_backend_hash = stable_hash(
        {
            name: _sha256_file(package_root / name)
            for name in backend_dependencies
        }
    )
    nuisance_evidence = {
        key: [
            {
                "state_sample_id": nuisance.state_sample_id,
                "weight": nuisance.weight,
                "initial_plant_state": (
                    nuisance.initial_plant_state
                ),
                "initial_actuator_state": (
                    nuisance.initial_actuator_state
                ),
                "disturbance_model_id": (
                    nuisance.disturbance_model_id
                ),
                "disturbance_parameters": (
                    nuisance.disturbance_parameters
                ),
                "sensor_bias": nuisance.sensor_bias,
            }
            for nuisance in item.nuisance_samples
        ]
        for key, item in prepared.items()
    }
    fixture_hash = (
        stable_hash(
            {
                "exact_controller_dependencies_sha256": (
                    closed_loop_dependencies.content_sha256
                ),
                "episode_nuisance_evidence": nuisance_evidence,
                "episode_controller_state_evidence": {
                    key: [
                        {
                            "state_sample_id": nuisance.state_sample_id,
                            "controller_state_sha256": (
                                nuisance.controller_state.content_sha256
                            ),
                        }
                        for nuisance in item.nuisance_samples
                    ]
                    for key, item in prepared.items()
                },
            }
        )
        if closed_loop
        else stable_hash(
            {
                key: {
                    "episode": item.data.normalized_episode_sha256,
                    "commands": item.commands.content_sha256,
                    "grids": item.grids.to_dict(),
                    "plant_geometry_sha256": (
                        config["plant"]["geometry_profile"][
                            "profile_sha256"
                        ]
                    ),
                    "nuisance_samples": nuisance_evidence[key],
                }
                for key, item in prepared.items()
            }
        )
    )
    provenance = PlantRunProvenance(
        source_commit=source_commit,
        source_bag_sha256=tuple(
            item.data.source_bag_sha256 for item in prepared.values()
        ),
        normalized_episode_sha256=tuple(
            item.data.normalized_episode_sha256
            for item in prepared.values()
        ),
        controller_snapshot_sha256=snapshot["snapshot_id"],
        controller_artifact_sha256=controller_artifact_hash,
        plant_backend_id=(
            CLOSED_LOOP_PLANT_BACKEND_ID
            if closed_loop
            else OPEN_LOOP_PLANT_BACKEND_ID
        ),
        plant_backend_sha256=plant_backend_hash,
        plant_geometry_profile_id=config["plant"][
            "geometry_profile"
        ]["profile_id"],
        plant_geometry_sha256=config["plant"]["geometry_profile"][
            "profile_sha256"
        ],
        prior_id=posterior.prior_id,
        likelihood_id=posterior.likelihood_id,
        seed=int(config["seed"]),
        config_sha256=config_sha256,
        fixture_sha256=fixture_hash,
        actuator_calibration_sha256=None,
    )
    return PlantAssimilationArtifactWriter(output_root).write(
        run_id=run_id,
        posterior=posterior,
        provenance=provenance,
        controller_snapshot=snapshot,
        controller_replay_audit=audit,
        factual_replay_report=(
            _exact_gate_payload(closed_loop_dependencies)
            if closed_loop
            else {
                "schema": "grape_factual_replay_report/v2",
                "passed": False,
                "status": "NOT_REQUIRED_FOR_OPEN_LOOP",
                "requested_fidelity": config["controller"]["fidelity"],
                "exact_claim_available": False,
                "closed_loop_exact_allowed": False,
            }
        ),
        identifiability_report=identifiability,
        likelihood_components=likelihood_rows,
        posterior_predictive=predictive,
        failure_validation=failure_validation,
        success_validation=success_validation,
        interpretation="effective_plant_posterior",
    )


__all__ = [
    "CLOSED_LOOP_PLANT_BACKEND_ID",
    "CLOSED_LOOP_POSTERIOR_MODEL_ID",
    "ExactClosedLoopDependencies",
    "OPEN_LOOP_PLANT_BACKEND_ID",
    "OPEN_LOOP_POSTERIOR_MODEL_ID",
    "PreparedEpisode",
    "SCHEMA",
    "build_replay_audits",
    "infer_plant_posterior",
    "load_assimilation_config",
    "prepare_episode",
    "prepare_episodes",
    "repository_source_identity",
    "validate_posterior",
    "write_assimilation_run",
]
