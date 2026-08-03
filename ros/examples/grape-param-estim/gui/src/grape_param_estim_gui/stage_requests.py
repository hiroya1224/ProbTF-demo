"""Deterministic GUI request builders for the staged estimator workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .workflow import canonical_fingerprint


DIAGONAL_Q_STAGE_REQUEST_SCHEMA = (
    "grape-param-estim/diagonal-q-stage-request/v1"
)
DIAGONAL_Q_STAGE_ID = "diagonal_q"
MINIMUM_STAGED_ENSEMBLE_SIZE = 58


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a finite number".format(label))
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number".format(label)) from error
    if not np.isfinite(selected) or (positive and selected <= 0.0):
        raise ValueError("{} must be finite and positive".format(label))
    return selected


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be an integer".format(label))
    try:
        selected = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be an integer".format(label)) from error
    if selected != value or selected < minimum:
        raise ValueError(
            "{} must be an integer of at least {}".format(label, minimum)
        )
    return selected


def diagonal_q_stage_settings(
    estimator_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve explicit six-component Q settings from a project manifest."""

    if not isinstance(estimator_settings, Mapping):
        raise ValueError("estimator_settings must be an object")
    sample_period = _finite(
        estimator_settings.get("sample_period"),
        "sample_period",
        positive=True,
    )
    ensemble_size = _integer(
        estimator_settings.get("ensemble_size"),
        "ensemble_size",
        minimum=MINIMUM_STAGED_ENSEMBLE_SIZE,
    )
    iterations = _integer(
        estimator_settings.get(
            "q_maximum_em_iterations",
            estimator_settings.get("maximum_iterations"),
        ),
        "q_maximum_em_iterations",
        minimum=1,
    )
    tolerance = _finite(
        estimator_settings.get(
            "q_log_q_tolerance",
            estimator_settings.get("convergence_tolerance", 1.0e-3),
        ),
        "q_log_q_tolerance",
        positive=True,
    )
    raw_floor = estimator_settings.get(
        "q_component_floor", [1.0e-9] * 6
    )
    if not isinstance(raw_floor, (list, tuple)) or len(raw_floor) != 6:
        raise ValueError("q_component_floor must contain six values")
    floor = [
        _finite(value, "q_component_floor[{}]".format(index), positive=True)
        for index, value in enumerate(raw_floor)
    ]
    fixed_delay = _finite(
        estimator_settings.get(
            "q_fixed_initial_delay_seconds",
            estimator_settings.get("delay_prior_mean"),
        ),
        "q_fixed_initial_delay_seconds",
    )
    if fixed_delay < 0.0:
        raise ValueError("q_fixed_initial_delay_seconds cannot be negative")
    seed = _integer(estimator_settings.get("seed"), "seed", minimum=0)
    if seed >= 2**32:
        raise ValueError("seed must be below 2**32")
    workers = estimator_settings.get("forecast_workers", "auto")
    if workers != "auto":
        workers = _integer(workers, "forecast_workers", minimum=1)
        if workers > 256:
            raise ValueError("forecast_workers cannot exceed 256")
    return {
        "sample_period": sample_period,
        "ensemble_size": ensemble_size,
        "maximum_em_iterations": iterations,
        "log_q_tolerance": tolerance,
        "component_floor": floor,
        "fixed_initial_delay_seconds": fixed_delay,
        "seed": seed,
        "forecast_workers": workers,
    }


def stage_bag_requests(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Detach the exact selected bag inputs required by both stages."""

    bags = []
    for record in sorted(tuple(records), key=lambda value: value.bag_id):
        if record.inspection is None or record.selected_interval is None:
            raise ValueError(
                "selected bag {} has no inspection interval".format(
                    record.bag_id
                )
            )
        recommendation = record.inspection.get("recommended_interval")
        if not isinstance(recommendation, Mapping):
            raise ValueError(
                "selected bag {} has no episode index".format(record.bag_id)
            )
        bags.append(
            {
                "bag_id": str(record.bag_id),
                "path": str(Path(record.path).resolve()),
                "sha256": str(record.sha256),
                "episode_index": int(recommendation["episode_index"]),
                "selected_interval_local_seconds": [
                    float(value) for value in record.selected_range
                ],
                "configuration_fingerprint": str(
                    record.configuration_fingerprint
                ),
            }
        )
    if not bags:
        raise ValueError("at least one selected bag is required")
    return bags


def build_diagonal_q_stage_request(
    *,
    run_id: str,
    project_fingerprint: str,
    stage_input_fingerprint: str,
    bags: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact request accepted by ``grape_estimate_diagonal_q``."""

    request = {
        "schema": DIAGONAL_Q_STAGE_REQUEST_SCHEMA,
        "run_id": str(run_id),
        "project_fingerprint": str(project_fingerprint),
        "stage_id": DIAGONAL_Q_STAGE_ID,
        "stage_input_fingerprint": str(stage_input_fingerprint),
        "bags": [dict(value) for value in bags],
        "settings": dict(settings),
    }
    # Fail here rather than enqueue a request containing non-finite JSON.
    canonical_fingerprint(request)
    return request


__all__ = [
    "DIAGONAL_Q_STAGE_ID",
    "DIAGONAL_Q_STAGE_REQUEST_SCHEMA",
    "MINIMUM_STAGED_ENSEMBLE_SIZE",
    "build_diagonal_q_stage_request",
    "diagonal_q_stage_settings",
    "stage_bag_requests",
]
