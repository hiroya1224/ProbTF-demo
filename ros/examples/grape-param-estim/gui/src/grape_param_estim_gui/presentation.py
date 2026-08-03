"""Qt-free formatting shared by production widgets and headless tests."""

from __future__ import annotations

import numpy as np

from .artifact_loader import SharedPosterior


def member_parameter_text(posterior: SharedPosterior, member_id: int) -> str:
    matches = np.flatnonzero(posterior.member_id == int(member_id))
    if not matches.size:
        raise KeyError("unknown posterior member {}".format(member_id))
    index = int(matches[0])
    return (
        "Member {} | mass {:.6g} kg | full inertia {} kg m² | CoG {} m | "
        "force effectiveness {} | torque effectiveness {} | constant delay τ {:.6g} s"
    ).format(
        member_id,
        posterior.mass[index],
        np.array2string(posterior.inertia[index], precision=5),
        np.array2string(posterior.cog[index], precision=5),
        np.array2string(posterior.force_effectiveness[index], precision=5),
        np.array2string(posterior.torque_effectiveness[index], precision=5),
        posterior.constant_delay[index],
    )


def scenario_assumption_text(assumption: object) -> str:
    """Prefix the backend's already-complete assumption exactly once."""

    value = str(assumption).strip()
    if not value:
        value = "unspecified"
    return "Counterfactual assumption: {}".format(value)


__all__ = ["member_parameter_text", "scenario_assumption_text"]
