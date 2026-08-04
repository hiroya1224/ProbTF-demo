"""Qt-free text formatting for batch-estimation result widgets."""

from __future__ import annotations

import numpy as np

from .artifact_loader import McmcPosterior, StaticParameterMap


def map_parameter_text(static_map: StaticParameterMap) -> str:
    return (
        "MAP | mass {:.6g} kg | full inertia {} kg m² | CoG {} m | "
        "force effectiveness {} | torque effectiveness {} | "
        "constant delay τ {:.6g} s"
    ).format(
        static_map.mass,
        np.array2string(static_map.inertia, precision=5),
        np.array2string(static_map.cog, precision=5),
        np.array2string(static_map.force_effectiveness, precision=5),
        np.array2string(static_map.torque_effectiveness, precision=5),
        static_map.delay,
    )


def sample_parameter_text(
    posterior: McmcPosterior, sample_id: str
) -> str:
    sample = posterior.sample(str(sample_id))
    return (
        "MCMC sample {} | chain {} draw {} | mass {:.6g} kg | "
        "full inertia {} kg m² | CoG {} m | force effectiveness {} | "
        "torque effectiveness {} | constant delay τ {:.6g} s"
    ).format(
        sample.sample_id,
        sample.chain_id,
        sample.draw_index,
        sample.mass,
        np.array2string(sample.inertia, precision=5),
        np.array2string(sample.cog, precision=5),
        np.array2string(sample.force_effectiveness, precision=5),
        np.array2string(sample.torque_effectiveness, precision=5),
        sample.delay,
    )


def scenario_assumption_text(assumption: object) -> str:
    """Prefix the backend's already-complete assumption exactly once."""

    value = str(assumption).strip()
    if not value:
        value = "unspecified"
    return "Counterfactual assumption: {}".format(value)


__all__ = [
    "map_parameter_text",
    "sample_parameter_text",
    "scenario_assumption_text",
]
