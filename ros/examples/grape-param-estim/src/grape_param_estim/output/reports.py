"""Small report helpers for weighted posterior summaries."""

from typing import Any, Mapping

from grape_param_estim.inference.posterior import PlantPosterior


def posterior_report_mapping(posterior: PlantPosterior) -> Mapping[str, Any]:
    if not isinstance(posterior, PlantPosterior):
        raise TypeError("posterior must be PlantPosterior")
    return {
        "model_id": posterior.model_id,
        "particle_count": len(posterior.particles),
        "effective_sample_size": posterior.effective_sample_size,
        "hpd_particle_count": int(posterior.hpd_indices.size),
        "hpd_weight": posterior.hpd_weight,
        "mean": posterior.mean,
        "covariance": posterior.covariance,
        "correlation": posterior.correlation,
        "multimodality": posterior.multimodality_diagnostic(),
    }


__all__ = ["posterior_report_mapping"]
