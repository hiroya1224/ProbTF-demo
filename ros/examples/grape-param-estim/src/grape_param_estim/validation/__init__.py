"""Validation contracts for plant posterior and controller design.

This package intentionally has no import-time dependency on the evolving
``plant`` or ``forward`` packages.  Controller design uses a structural
posterior protocol, and established target-tube/support symbols are loaded
only if a caller requests those compatibility names.
"""

from typing import Any

from .controller_design import (
    CONTROLLER_EVALUATOR_IDENTITY_SCHEMA,
    CONTROLLER_CANDIDATE_SCHEMA,
    CONTROLLER_PARTICLE_OUTCOME_SCHEMA,
    CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA,
    CONTROLLER_RECOMMENDATION_BINDING_SCHEMA,
    CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA,
    BoundParticleEvaluator,
    ControllerCandidate,
    ControllerDesignEvaluation,
    ControllerEvaluatorIdentity,
    ControllerParticleOutcome,
    ControllerParticleOutputEvidence,
    ControllerRecommendationBinding,
    ControllerRecommendationEvidence,
    ControllerRecommendationGates,
    PlantPosteriorLike,
    VerifiedPlantArtifactIdentity,
    WeightedControllerParticleOutcome,
    evaluate_candidate_over_posterior,
    evaluate_controller_candidate,
    measure_particle_evaluator_sha256,
)
from .failure_event import (
    CompositeFailureDetector,
    FailureCensoring,
    FailureDetector,
    FailureEvent,
    FirstMaskFailureDetector,
    RolloutSafetyFailureDetector,
    ThresholdFailureDetector,
    censor_after_failure,
)
from .posterior_predictive import (
    HeldOutFailureValidation,
    PosteriorPredictiveValidation,
    TrajectoryEnvelope,
    TrajectoryEnvelopeValidation,
    ValidationDatasetIdentity,
    posterior_failure_probability,
    trajectory_envelope,
    validate_held_out_failure,
    validate_posterior_predictive,
    validate_trajectory_envelope,
)
from .success_gate import (
    SUCCESS_VALIDATION_ROLE,
    SuccessEpisodeValidation,
    SuccessGateConfig,
    SuccessGateReport,
    assert_success_episodes_validation_only,
    evaluate_success_episode,
    evaluate_success_gate,
)


_COMPATIBILITY_NAMES = frozenset(
    (
        "EXTRAPOLATIVE",
        "SUPPORTED",
        "UNSUPPORTED",
        "SupportDiagnostics",
        "SupportReference",
        "TargetTrajectory",
        "TargetTube",
        "TubeEvaluation",
        "classify_support",
        "evaluate_target_tube",
    )
)


def __getattr__(name: str) -> Any:
    if name not in _COMPATIBILITY_NAMES:
        raise AttributeError(name)
    if name in (
        "TargetTrajectory",
        "TargetTube",
        "TubeEvaluation",
        "evaluate_target_tube",
    ):
        from . import trajectory_tube

        return getattr(trajectory_tube, name)
    from . import support

    return getattr(support, name)


__all__ = [
    "BoundParticleEvaluator",
    "CONTROLLER_CANDIDATE_SCHEMA",
    "CONTROLLER_EVALUATOR_IDENTITY_SCHEMA",
    "CONTROLLER_PARTICLE_OUTCOME_SCHEMA",
    "CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA",
    "CONTROLLER_RECOMMENDATION_BINDING_SCHEMA",
    "CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA",
    "CompositeFailureDetector",
    "ControllerCandidate",
    "ControllerDesignEvaluation",
    "ControllerEvaluatorIdentity",
    "ControllerParticleOutcome",
    "ControllerParticleOutputEvidence",
    "ControllerRecommendationBinding",
    "ControllerRecommendationEvidence",
    "ControllerRecommendationGates",
    "FailureCensoring",
    "FailureDetector",
    "FailureEvent",
    "FirstMaskFailureDetector",
    "RolloutSafetyFailureDetector",
    "HeldOutFailureValidation",
    "PlantPosteriorLike",
    "VerifiedPlantArtifactIdentity",
    "PosteriorPredictiveValidation",
    "SUCCESS_VALIDATION_ROLE",
    "SuccessEpisodeValidation",
    "SuccessGateConfig",
    "SuccessGateReport",
    "ThresholdFailureDetector",
    "TrajectoryEnvelope",
    "TrajectoryEnvelopeValidation",
    "ValidationDatasetIdentity",
    "WeightedControllerParticleOutcome",
    "assert_success_episodes_validation_only",
    "censor_after_failure",
    "evaluate_candidate_over_posterior",
    "evaluate_controller_candidate",
    "evaluate_success_episode",
    "evaluate_success_gate",
    "posterior_failure_probability",
    "measure_particle_evaluator_sha256",
    "trajectory_envelope",
    "validate_held_out_failure",
    "validate_posterior_predictive",
    "validate_trajectory_envelope",
] + sorted(_COMPATIBILITY_NAMES)
