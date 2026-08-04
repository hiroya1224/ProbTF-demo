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
POSTERIOR_SAMPLING_REQUEST_SCHEMA = (
    "grape-param-estim/posterior-sampling-request/v1"
)
POSTERIOR_MCMC_SETTING_KEYS = (
    "chain_count",
    "warmup_steps",
    "retained_draws",
    "thinning",
    "random_seed",
    "local_scale",
    "exact_ridge_scale",
    "near_ridge_scale",
    "identified_scale",
    "delay_scale_seconds",
    "near_relative_threshold",
    "rhat_threshold",
    "minimum_effective_sample_size",
)

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


def build_posterior_sampling_request(
    *,
    sampling_id: str,
    resume: bool,
    estimation_run_directory: str | Path,
    estimation_request_path: str | Path,
    estimation_manifest: Mapping[str, Any],
    mcmc_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the strict request that appends MCMC to estimate-only output.

    The upstream identity comes only from the already validated estimation
    manifest.  The GUI neither recomputes nor weakens those bindings.
    """

    if not isinstance(sampling_id, str) or not sampling_id.strip():
        raise ValueError("sampling_id must be canonical non-empty text")
    if sampling_id.strip() != sampling_id:
        raise ValueError("sampling_id must be canonical non-empty text")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    run_directory = Path(estimation_run_directory)
    request_path = Path(estimation_request_path)
    for value, label in (
        (run_directory, "estimation_run_directory"),
        (request_path, "estimation_request_path"),
    ):
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError(
                "{} must be an absolute path without '..'".format(label)
            )
    if not request_path.is_file():
        raise ValueError("estimation_request_path must name an existing file")
    if not isinstance(estimation_manifest, Mapping):
        raise ValueError("estimation_manifest must be an object")
    if estimation_manifest.get("status") != "complete":
        raise ValueError("estimation manifest must be complete")
    if estimation_manifest.get("mcmc_settings") != {"enabled": False}:
        raise ValueError("estimation manifest must be estimate-only")

    if not isinstance(mcmc_settings, Mapping):
        raise ValueError("mcmc_settings must be an object")
    _exact_keys(
        mcmc_settings,
        {"enabled", *POSTERIOR_MCMC_SETTING_KEYS},
        "mcmc_settings",
    )
    if mcmc_settings.get("enabled") is not True:
        raise ValueError("mcmc_settings.enabled must be true")
    sampling_settings = {
        key: mcmc_settings[key] for key in POSTERIOR_MCMC_SETTING_KEYS
    }

    upstream_keys = (
        "run_id",
        "request_fingerprint",
        "configuration_fingerprint",
        "controller_snapshot_fingerprint",
        "estimator_revision",
        "selected_bag_ids",
        "selected_intervals",
        "selected_bag_sha256",
    )
    missing = [key for key in upstream_keys if key not in estimation_manifest]
    if missing:
        raise ValueError(
            "estimation manifest lacks {}".format(", ".join(missing))
        )
    upstream = {
        key: estimation_manifest[key] for key in upstream_keys
    }
    request = {
        "schema": POSTERIOR_SAMPLING_REQUEST_SCHEMA,
        "sampling_id": sampling_id,
        "resume": resume,
        "estimation_run_directory": str(run_directory.resolve()),
        "estimation_request_path": str(request_path.resolve()),
        "upstream": _finite_json_copy(upstream, "estimation upstream"),
        "mcmc_settings": _finite_json_copy(
            sampling_settings, "mcmc_settings"
        ),
    }
    canonical_fingerprint(request)
    return request


def posterior_sampling_request_fingerprint(
    request: Mapping[str, Any],
) -> str:
    """Return the backend sampling identity, excluding only ``resume``."""

    if not isinstance(request, Mapping):
        raise ValueError("posterior sampling request must be an object")
    payload = _finite_json_copy(request, "posterior sampling request")
    if payload.get("schema") != POSTERIOR_SAMPLING_REQUEST_SCHEMA:
        raise ValueError("posterior sampling request schema is unsupported")
    if not isinstance(payload.get("resume"), bool):
        raise ValueError("posterior sampling request resume must be boolean")
    payload["resume"] = False
    return canonical_fingerprint(payload)


__all__ = [
    "BATCH_ESTIMATION_REQUEST_SCHEMA",
    "BATCH_ESTIMATION_STAGE_ID",
    "POSTERIOR_MCMC_SETTING_KEYS",
    "POSTERIOR_SAMPLING_REQUEST_SCHEMA",
    "RUN_MODES",
    "batch_estimation_settings",
    "build_batch_estimation_request",
    "build_posterior_sampling_request",
    "posterior_sampling_request_fingerprint",
    "stage_bag_requests",
    "workflow_mode_run_mode",
]
