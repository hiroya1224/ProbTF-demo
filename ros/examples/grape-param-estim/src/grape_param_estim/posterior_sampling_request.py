"""Strict request contract for appending posterior samples to one run."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    read_json,
    request_fingerprint,
)


POSTERIOR_SAMPLING_REQUEST_SCHEMA = (
    "grape-param-estim/posterior-sampling-request/v1"
)
MCMC_SETTING_KEYS = (
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


def _error(location: str, message: str):
    raise ArtifactValidationError("{} {}".format(location, message))


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(location, "must be an object")
    return value


def _keys(value: Mapping[str, Any], expected, location: str) -> None:
    if set(value) != set(expected):
        _error(location, "keys must be exactly {}".format(sorted(expected)))


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _error(location, "must be canonical non-empty text")
    return value


def _sha256(value: Any, location: str) -> str:
    selected = _text(value, location)
    if (
        len(selected) != 71
        or not selected.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in selected[7:])
    ):
        _error(location, "must have form sha256:<64 lowercase hex>")
    return selected


def _absolute_path(value: Any, location: str, must_exist: bool) -> Path:
    selected = Path(_text(value, location)).expanduser()
    if not selected.is_absolute():
        _error(location, "must be absolute")
    result = selected.resolve()
    if must_exist and not result.is_file():
        _error(location, "must name an existing file")
    return result


def _integer(value: Any, minimum: int, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _error(location, "must be an integer")
    selected = int(value)
    if selected < minimum:
        _error(location, "must be >= {}".format(minimum))
    return selected


def _positive(value: Any, location: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        _error(location, "must be numeric")
    selected = float(value)
    if not np.isfinite(selected) or selected <= 0.0:
        _error(location, "must be finite and positive")
    return selected


def _interval(value: Any, location: str) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        _error(location, "must contain [start, end]")
    start = float(value[0])
    end = float(value[1])
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        _error(location, "must contain finite increasing bounds")
    return start, end


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class PosteriorSamplingRequest:
    source_path: Path
    payload: Mapping[str, Any]
    fingerprint: str
    estimation_run_directory: Path
    estimation_request_path: Path


def validate_posterior_sampling_request(
    payload: Mapping[str, Any], source_path: Union[str, Path] = "<memory>"
) -> PosteriorSamplingRequest:
    request = _mapping(payload, "request")
    expected = (
        "schema",
        "sampling_id",
        "resume",
        "estimation_run_directory",
        "estimation_request_path",
        "upstream",
        "mcmc_settings",
    )
    _keys(request, expected, "request")
    if request["schema"] != POSTERIOR_SAMPLING_REQUEST_SCHEMA:
        _error("request.schema", "is unsupported")
    _text(request["sampling_id"], "request.sampling_id")
    if not isinstance(request["resume"], bool):
        _error("request.resume", "must be boolean")
    run_directory = _absolute_path(
        request["estimation_run_directory"],
        "request.estimation_run_directory",
        False,
    )
    estimation_request_path = _absolute_path(
        request["estimation_request_path"],
        "request.estimation_request_path",
        True,
    )
    upstream = _mapping(request["upstream"], "request.upstream")
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
    _keys(upstream, upstream_keys, "request.upstream")
    _text(upstream["run_id"], "request.upstream.run_id")
    for name in (
        "request_fingerprint",
        "configuration_fingerprint",
        "controller_snapshot_fingerprint",
    ):
        _sha256(upstream[name], "request.upstream." + name)
    _text(upstream["estimator_revision"], "request.upstream.estimator_revision")
    bag_ids = upstream["selected_bag_ids"]
    if (
        not isinstance(bag_ids, list)
        or not bag_ids
        or any(not isinstance(value, str) or not value for value in bag_ids)
        or len(set(bag_ids)) != len(bag_ids)
    ):
        _error("request.upstream.selected_bag_ids", "must be unique text")
    intervals = _mapping(
        upstream["selected_intervals"], "request.upstream.selected_intervals"
    )
    hashes = _mapping(
        upstream["selected_bag_sha256"], "request.upstream.selected_bag_sha256"
    )
    if set(intervals) != set(bag_ids) or set(hashes) != set(bag_ids):
        _error("request.upstream", "bag maps must exactly match selected_bag_ids")
    for bag_id in bag_ids:
        _interval(intervals[bag_id], "request.upstream.selected_intervals." + bag_id)
        _sha256(hashes[bag_id], "request.upstream.selected_bag_sha256." + bag_id)

    settings = _mapping(request["mcmc_settings"], "request.mcmc_settings")
    _keys(settings, MCMC_SETTING_KEYS, "request.mcmc_settings")
    for name, minimum in (
        ("chain_count", 2),
        ("warmup_steps", 0),
        ("retained_draws", 4),
        ("thinning", 1),
        ("random_seed", 0),
    ):
        _integer(settings[name], minimum, "request.mcmc_settings." + name)
    for name in MCMC_SETTING_KEYS[5:]:
        _positive(settings[name], "request.mcmc_settings." + name)
    if float(settings["rhat_threshold"]) <= 1.0:
        _error("request.mcmc_settings.rhat_threshold", "must exceed one")

    fingerprint_payload = dict(request)
    fingerprint_payload["resume"] = False
    return PosteriorSamplingRequest(
        source_path=Path(source_path),
        payload=_freeze(request),
        fingerprint=request_fingerprint(fingerprint_payload),
        estimation_run_directory=run_directory,
        estimation_request_path=estimation_request_path,
    )


def load_posterior_sampling_request(
    path: Union[str, Path]
) -> PosteriorSamplingRequest:
    source = Path(path).expanduser().resolve()
    return validate_posterior_sampling_request(read_json(source), source)


__all__ = [
    "MCMC_SETTING_KEYS",
    "POSTERIOR_SAMPLING_REQUEST_SCHEMA",
    "PosteriorSamplingRequest",
    "load_posterior_sampling_request",
    "validate_posterior_sampling_request",
]
