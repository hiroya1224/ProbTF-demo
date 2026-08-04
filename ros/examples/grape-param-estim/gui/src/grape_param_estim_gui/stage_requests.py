"""Deterministic GUI request builder for the sparse batch worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .workflow import WorkflowMode, canonical_fingerprint


BATCH_ESTIMATION_REQUEST_SCHEMA = (
    "grape-param-estim/batch-estimation-request/v1"
)
BATCH_ESTIMATION_STAGE_ID = "batch_estimation"
RUN_MODES = ("estimate_only", "estimate_and_sample")

_SETTING_KEYS = {
    "q",
    "parameter_prior",
    "delay",
    "actuator_model",
    "knot_policy",
    "interpolation_policy",
    "controller_snapshot_policy",
    "mode_hypotheses",
    "solver_settings",
    "em_settings",
    "mcmc_settings",
}
_BAG_SETTING_KEYS = {
    "observation_factors",
    "fixed_factor_covariances",
    "initial_state_prior_covariances",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    supplied = set(value)
    if supplied == expected:
        return
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    details = []
    if missing:
        details.append("missing {}".format(", ".join(missing)))
    if unknown:
        details.append("unexpected {}".format(", ".join(unknown)))
    raise ValueError("{} has {}".format(label, "; ".join(details)))


def _finite_json_copy(value: Any, label: str) -> Any:
    try:
        canonical_fingerprint(value)
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be finite JSON".format(label)) from error


def workflow_mode_run_mode(mode: WorkflowMode | str) -> str:
    """Map staged user interaction to the worker's explicit run mode."""

    try:
        selected = WorkflowMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError("workflow mode must be STEP or ALL") from error
    return (
        "estimate_only"
        if selected == WorkflowMode.STEP
        else "estimate_and_sample"
    )


def batch_estimation_settings(
    estimator_settings: Mapping[str, Any], *, run_mode: str
) -> dict[str, Any]:
    """Detach the exact scientific settings accepted by the worker.

    No covariance, prior, solver, EM, or MCMC default is supplied by the GUI.
    The project must persist every scientific choice explicitly.
    """

    if not isinstance(estimator_settings, Mapping):
        raise ValueError("estimator_settings must be an object")
    _exact_keys(estimator_settings, _SETTING_KEYS, "estimator_settings")
    if run_mode not in RUN_MODES:
        raise ValueError("run_mode must be estimate_only or estimate_and_sample")
    settings = _finite_json_copy(estimator_settings, "estimator_settings")
    mcmc = settings["mcmc_settings"]
    if not isinstance(mcmc, dict) or not isinstance(mcmc.get("enabled"), bool):
        raise ValueError("mcmc_settings.enabled must be explicit boolean")
    expected_enabled = run_mode == "estimate_and_sample"
    if mcmc["enabled"] != expected_enabled:
        raise ValueError(
            "mcmc_settings.enabled must match run_mode"
        )
    return settings


def stage_bag_requests(
    records: Sequence[Any],
    bag_settings: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind selected bags and intervals to explicit factor configuration."""

    if not isinstance(bag_settings, Mapping):
        raise ValueError("bag_settings must be an object")
    selected_records = sorted(tuple(records), key=lambda value: value.bag_id)
    selected_ids = {str(record.bag_id) for record in selected_records}
    if not selected_records:
        raise ValueError("at least one selected bag is required")
    if set(bag_settings) != selected_ids:
        raise ValueError("bag_settings must exactly match selected bag IDs")

    bags = []
    for record in selected_records:
        if record.inspection is None or record.selected_interval is None:
            raise ValueError(
                "selected bag {} has no inspection interval".format(
                    record.bag_id
                )
            )
        configuration = bag_settings[str(record.bag_id)]
        if not isinstance(configuration, Mapping):
            raise ValueError("bag settings must be objects")
        _exact_keys(
            configuration,
            _BAG_SETTING_KEYS,
            "bag_settings.{}".format(record.bag_id),
        )
        digest = str(record.sha256)
        if not digest.startswith("sha256:"):
            digest = "sha256:" + digest
        bags.append(
            {
                "bag_id": str(record.bag_id),
                "path": str(Path(record.path).resolve()),
                "sha256": digest,
                "interval_seconds": [
                    float(value) for value in record.selected_range
                ],
                **_finite_json_copy(
                    configuration,
                    "bag_settings.{}".format(record.bag_id),
                ),
            }
        )
    canonical_fingerprint(bags)
    return bags


def build_batch_estimation_request(
    *,
    run_id: str,
    run_mode: str,
    resume: bool,
    output_directory: str | Path,
    bags: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact one-command request accepted by the batch worker."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if run_mode not in RUN_MODES:
        raise ValueError("run_mode must be estimate_only or estimate_and_sample")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    output = Path(output_directory)
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError("output_directory must be an absolute path without '..'")
    resolved_settings = batch_estimation_settings(settings, run_mode=run_mode)
    request = {
        "schema": BATCH_ESTIMATION_REQUEST_SCHEMA,
        "run_id": run_id,
        "run_mode": run_mode,
        "resume": resume,
        "output_directory": str(output.resolve()),
        "bags": [_finite_json_copy(dict(value), "bag request") for value in bags],
        **resolved_settings,
    }
    canonical_fingerprint(request)
    return request


__all__ = [
    "BATCH_ESTIMATION_REQUEST_SCHEMA",
    "BATCH_ESTIMATION_STAGE_ID",
    "RUN_MODES",
    "batch_estimation_settings",
    "build_batch_estimation_request",
    "stage_bag_requests",
    "workflow_mode_run_mode",
]
