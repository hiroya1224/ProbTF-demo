"""Compatibility boundary for optional particle-marginal MH inference."""

from grape_param_estim.alternative_backends import (
    LinearGaussianRandomWalkModel,
    ParticleLikelihoodDegeneracy,
    ParticleMarginalMetropolisHastings,
    ParticleMarginalMhConfig,
    ParticleMarginalMhPosterior,
    ParticleStateSpaceModel,
    bootstrap_particle_log_likelihood,
)

__all__ = [
    "LinearGaussianRandomWalkModel",
    "ParticleLikelihoodDegeneracy",
    "ParticleMarginalMetropolisHastings",
    "ParticleMarginalMhConfig",
    "ParticleMarginalMhPosterior",
    "ParticleStateSpaceModel",
    "bootstrap_particle_log_likelihood",
]
