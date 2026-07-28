"""Static-parameter inference for forward plant models.

Every legacy public symbol remains available here so existing estimators keep
working while redesigned callers use component likelihoods, rollout caching,
identifiability reports, and full weighted ``PlantPosterior`` objects.
"""

from grape_param_estim._legacy_inference import (
    BoundedLogitTransform,
    BoxUniformPrior,
    ChainDiagnostics,
    IdentityTransform,
    SmcPosterior,
    SmcStage,
    TemperedResampleMoveSmc,
    TemperedSmcConfig,
    chain_diagnostics,
    effective_sample_size,
    marginalize_trajectory_log_likelihood,
    predictive_interval_coverage,
    systematic_resample,
)
from grape_param_estim.inference.cache import RolloutCache, RolloutCacheKey
from grape_param_estim.inference.identifiability import (
    EpisodeExcitationReport,
    IdentifiabilityReport,
    episode_excitation_report,
    local_identifiability,
)
from grape_param_estim.inference.likelihood import (
    CONTROLLER_EVENT_OBSERVATIONS_SCHEMA,
    CONTROLLER_MODE_EVENT_MASK,
    CONTROLLER_SATURATION_EVENT_MASK,
    ControllerEventObservations,
    EpisodeLikelihood,
    LikelihoodComponents,
    LikelihoodConfig,
    MultipleEpisodeLikelihood,
    ObservationDataset,
)
from grape_param_estim.inference.posterior import PlantPosterior
from grape_param_estim.inference.prior import (
    BoundedLogUniformPrior,
    IndependentBoundedPrior,
    PriorDimension,
)
from grape_param_estim.inference.tempered_smc import BatchPlantInference

__all__ = [
    "BatchPlantInference",
    "BoundedLogUniformPrior",
    "BoundedLogitTransform",
    "BoxUniformPrior",
    "ChainDiagnostics",
    "CONTROLLER_EVENT_OBSERVATIONS_SCHEMA",
    "CONTROLLER_MODE_EVENT_MASK",
    "CONTROLLER_SATURATION_EVENT_MASK",
    "ControllerEventObservations",
    "EpisodeExcitationReport",
    "EpisodeLikelihood",
    "IdentifiabilityReport",
    "IdentityTransform",
    "IndependentBoundedPrior",
    "LikelihoodComponents",
    "LikelihoodConfig",
    "MultipleEpisodeLikelihood",
    "ObservationDataset",
    "PlantPosterior",
    "PriorDimension",
    "RolloutCache",
    "RolloutCacheKey",
    "SmcPosterior",
    "SmcStage",
    "TemperedResampleMoveSmc",
    "TemperedSmcConfig",
    "chain_diagnostics",
    "effective_sample_size",
    "episode_excitation_report",
    "local_identifiability",
    "marginalize_trajectory_log_likelihood",
    "predictive_interval_coverage",
    "systematic_resample",
]
