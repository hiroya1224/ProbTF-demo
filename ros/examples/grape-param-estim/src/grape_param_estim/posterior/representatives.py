"""Deterministic role-union representatives of retained posterior draws.

The selector in this module is deliberately independent of trajectory solves
and artifact I/O.  Exporters and readers can therefore recompute the same
selection from retained MCMC arrays and reject provenance that does not match
the scientific policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


STATIC_PARAMETER_DIMENSION = 18
POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT = 8
PID_STRATIFIED_MAXIMUM_REPRESENTATIVE_COUNT = 4

CONDITIONAL_TRAJECTORY_SELECTION_POLICY = (
    "deterministic_posterior_role_union_v2"
)
CONDITIONAL_TRAJECTORY_SAMPLE_ORDER = (
    "role_priority_then_canonical_chain_draw_first_occurrence"
)
CONDITIONAL_TRAJECTORY_EVALUATION_METHOD = "fresh_conditional_sparse_map"
CONDITIONAL_TRAJECTORY_WARM_START_POLICY = "selected_mode_map_local_state"

ROLE_PRIORITY = (
    "highest_log_posterior",
    "mode_representative",
    "exact_ridge_quantile",
    "pid_stratified_medoid",
)
ROLE_RECORD_KEYS = (
    "role_class",
    "role_id",
    "sample_id",
    "mode_id",
    "quantile_numerator",
    "quantile_denominator",
    "ordinal",
)
RIDGE_QUANTILES = ((1, 10), (1, 2), (9, 10))

FEATURE_POLICY = MappingProxyType(
    {
        "static_feature": "parameter_prior_lower_cholesky_whitened_18d",
        "delay_feature": (
            "bounds_midpoint_centered_divided_by_request_delay_scale"
        ),
        "feature_dimension": 19,
        "distance": "squared_euclidean",
        "weighting": "unweighted_retained_draws",
        "medoid_algorithm": "deterministic_pam_build_swap_v1",
        "maximum_pid_representative_count": (
            PID_STRATIFIED_MAXIMUM_REPRESENTATIVE_COUNT
        ),
    }
)
RIDGE_COORDINATE_POLICY = MappingProxyType(
    {
        "coordinate": (
            "prior_centered_dot_canonical_exact_ridge_direction"
        ),
        "sign": "first_nonzero_component_positive",
        "empirical_quantile": "nearest_order_statistic_half_up",
    }
)

SELECTION_MANIFEST_KEYS = (
    "policy",
    "sample_order",
    "role_priority",
    "feature_policy",
    "ridge_coordinate_policy",
    "available_sample_count",
    "maximum_sample_count",
    "selected_sample_ids",
    "selected_bag_ids",
    "role_records",
    "conditional_evaluation_method",
    "warm_start_policy",
)


def _canonical_text(value: object, name: str) -> str:
    selected = str(value)
    if not selected or selected.strip() != selected:
        raise ValueError("{} must be canonical non-empty text".format(name))
    return selected


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _finite_vector(
    value: object, shape: Tuple[int, ...], name: str
) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != shape or not np.all(np.isfinite(selected)):
        raise ValueError("{} must be a finite array of shape {}".format(name, shape))
    return selected


@dataclass(frozen=True)
class RepresentativeRole:
    """One role attached to one retained posterior sample."""

    role_class: str
    role_id: str
    sample_id: str
    mode_id: object = None
    quantile_numerator: object = None
    quantile_denominator: object = None
    ordinal: object = None

    def __post_init__(self) -> None:
        role_class = _canonical_text(self.role_class, "role_class")
        if role_class not in ROLE_PRIORITY:
            raise ValueError("unknown posterior representative role class")
        role_id = _canonical_text(self.role_id, "role_id")
        sample_id = _canonical_text(self.sample_id, "sample_id")
        mode_id = self.mode_id
        numerator = self.quantile_numerator
        denominator = self.quantile_denominator
        ordinal = self.ordinal
        if role_class == "mode_representative":
            mode_id = _canonical_text(mode_id, "mode_id")
        elif mode_id is not None:
            raise ValueError("mode_id is reserved for mode representatives")
        if role_class == "exact_ridge_quantile":
            numerator = _positive_integer(numerator, "quantile_numerator")
            denominator = _positive_integer(
                denominator, "quantile_denominator"
            )
            if (numerator, denominator) not in RIDGE_QUANTILES:
                raise ValueError("unknown exact-ridge quantile role")
        elif numerator is not None or denominator is not None:
            raise ValueError("quantile fields are reserved for ridge roles")
        if role_class == "pid_stratified_medoid":
            if (
                isinstance(ordinal, (bool, np.bool_))
                or not isinstance(ordinal, Integral)
                or int(ordinal) < 0
                or int(ordinal)
                >= PID_STRATIFIED_MAXIMUM_REPRESENTATIVE_COUNT
            ):
                raise ValueError("PID medoid ordinal is outside the policy")
            ordinal = int(ordinal)
        elif ordinal is not None:
            raise ValueError("ordinal is reserved for PID medoid roles")
        object.__setattr__(self, "role_class", role_class)
        object.__setattr__(self, "role_id", role_id)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "mode_id", mode_id)
        object.__setattr__(self, "quantile_numerator", numerator)
        object.__setattr__(self, "quantile_denominator", denominator)
        object.__setattr__(self, "ordinal", ordinal)

    @property
    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "role_class": self.role_class,
            "role_id": self.role_id,
            "sample_id": self.sample_id,
            "mode_id": self.mode_id,
            "quantile_numerator": self.quantile_numerator,
            "quantile_denominator": self.quantile_denominator,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class PosteriorRepresentativeSelection:
    """A bounded role union, before its Cartesian product with bags."""

    available_sample_count: int
    maximum_sample_count: int
    selected_sample_ids: Tuple[str, ...]
    role_records: Tuple[RepresentativeRole, ...]

    def __post_init__(self) -> None:
        available = _positive_integer(
            self.available_sample_count, "available_sample_count"
        )
        maximum = _positive_integer(
            self.maximum_sample_count, "maximum_sample_count"
        )
        if maximum != POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT:
            raise ValueError("posterior representative union uses fixed bound eight")
        if (
            type(self.selected_sample_ids) is not tuple
            or not self.selected_sample_ids
            or any(
                type(value) is not str
                or not value
                or value.strip() != value
                for value in self.selected_sample_ids
            )
            or len(set(self.selected_sample_ids)) != len(self.selected_sample_ids)
        ):
            raise ValueError("selected_sample_ids must be unique canonical text")
        if len(self.selected_sample_ids) > min(available, maximum):
            raise ValueError("posterior representative union exceeds its bound")
        if (
            type(self.role_records) is not tuple
            or not self.role_records
            or any(
                not isinstance(value, RepresentativeRole)
                for value in self.role_records
            )
        ):
            raise ValueError("role_records must contain representative roles")
        role_ids = tuple(value.role_id for value in self.role_records)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("posterior representative role IDs must be unique")
        if any(
            value.sample_id not in self.selected_sample_ids
            for value in self.role_records
        ):
            raise ValueError("every role must reference a selected sample")
        if self.role_records[0].role_class != "highest_log_posterior":
            raise ValueError("highest log-posterior must be the first role")
        if self.selected_sample_ids[0] != self.role_records[0].sample_id:
            raise ValueError("primary posterior sample must be first")
        object.__setattr__(self, "available_sample_count", available)
        object.__setattr__(self, "maximum_sample_count", maximum)

    def manifest_payload(self, selected_bag_ids: Sequence[object]):
        bag_ids = tuple(
            _canonical_text(value, "selected bag ID")
            for value in selected_bag_ids
        )
        if not bag_ids or len(set(bag_ids)) != len(bag_ids):
            raise ValueError("selected_bag_ids must be non-empty and unique")
        return {
            "policy": CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
            "sample_order": CONDITIONAL_TRAJECTORY_SAMPLE_ORDER,
            "role_priority": list(ROLE_PRIORITY),
            "feature_policy": dict(FEATURE_POLICY),
            "ridge_coordinate_policy": dict(RIDGE_COORDINATE_POLICY),
            "available_sample_count": self.available_sample_count,
            "maximum_sample_count": self.maximum_sample_count,
            "selected_sample_ids": list(self.selected_sample_ids),
            "selected_bag_ids": list(bag_ids),
            "role_records": [
                dict(value.manifest_payload) for value in self.role_records
            ],
            "conditional_evaluation_method": (
                CONDITIONAL_TRAJECTORY_EVALUATION_METHOD
            ),
            "warm_start_policy": CONDITIONAL_TRAJECTORY_WARM_START_POLICY,
        }


@dataclass(frozen=True)
class _CanonicalDraws:
    sample_id: Tuple[str, ...]
    chain_id: Tuple[str, ...]
    draw_index: np.ndarray
    static_coordinate: np.ndarray
    delay: np.ndarray
    log_posterior: np.ndarray
    source_mode_id: Tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.sample_id)

    def canonical_key(self, index: int):
        return (
            self.chain_id[index],
            int(self.draw_index[index]),
            self.sample_id[index],
        )


def _canonical_draws(
    *,
    sample_id: Sequence[object],
    chain_id: Sequence[object],
    draw_index: object,
    static_coordinate: object,
    delay: object,
    log_posterior: object,
    source_mode_id: Sequence[object],
) -> _CanonicalDraws:
    samples = tuple(_canonical_text(value, "sample_id") for value in sample_id)
    count = len(samples)
    if count == 0 or len(set(samples)) != count:
        raise ValueError("sample IDs must be non-empty and globally unique")
    chains = tuple(_canonical_text(value, "chain_id") for value in chain_id)
    modes = tuple(
        _canonical_text(value, "source_mode_id") for value in source_mode_id
    )
    if len(chains) != count or len(modes) != count:
        raise ValueError("posterior sample labels have inconsistent lengths")
    draws = np.asarray(draw_index)
    if (
        draws.shape != (count,)
        or draws.dtype.kind not in "iu"
        or np.any(draws < 0)
    ):
        raise ValueError("draw_index must be a non-negative integer vector")
    coordinates = _finite_vector(
        static_coordinate,
        (count, STATIC_PARAMETER_DIMENSION),
        "static_coordinate",
    )
    delays = _finite_vector(delay, (count,), "delay")
    log_density = _finite_vector(log_posterior, (count,), "log_posterior")
    keys = tuple(
        (chains[index], int(draws[index]), samples[index])
        for index in range(count)
    )
    if len(set((value[0], value[1]) for value in keys)) != count:
        raise ValueError("(chain_id, draw_index) pairs must be unique")
    order = tuple(sorted(range(count), key=keys.__getitem__))
    return _CanonicalDraws(
        sample_id=tuple(samples[index] for index in order),
        chain_id=tuple(chains[index] for index in order),
        draw_index=np.asarray(tuple(draws[index] for index in order), dtype=np.int64),
        static_coordinate=np.asarray(coordinates[list(order)], dtype=float),
        delay=np.asarray(delays[list(order)], dtype=float),
        log_posterior=np.asarray(log_density[list(order)], dtype=float),
        source_mode_id=tuple(modes[index] for index in order),
    )


def _canonical_ridge_direction(value: object) -> np.ndarray:
    direction = _finite_vector(
        value, (STATIC_PARAMETER_DIMENSION,), "exact_ridge_direction"
    )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("exact_ridge_direction must be nonzero")
    direction = direction / norm
    nonzero = np.flatnonzero(direction != 0.0)
    if nonzero.size == 0:  # pragma: no cover - norm excludes it
        raise ValueError("exact_ridge_direction must be nonzero")
    if direction[int(nonzero[0])] < 0.0:
        direction = -direction
    return direction


def _nearest_empirical_index(count: int, numerator: int, denominator: int) -> int:
    # floor((count - 1) * q + 1/2), expressed entirely with integers.
    return (
        2 * (count - 1) * numerator + denominator
    ) // (2 * denominator)


def _posterior_features(
    draws: _CanonicalDraws,
    prior_mean_coordinate: object,
    prior_covariance: object,
    delay_bounds_seconds: Sequence[object],
    delay_scale_seconds: object,
) -> np.ndarray:
    mean = _finite_vector(
        prior_mean_coordinate,
        (STATIC_PARAMETER_DIMENSION,),
        "prior_mean_coordinate",
    )
    covariance = _finite_vector(
        prior_covariance,
        (STATIC_PARAMETER_DIMENSION, STATIC_PARAMETER_DIMENSION),
        "prior_covariance",
    )
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("prior_covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("prior_covariance must be positive definite") from error
    if not isinstance(delay_bounds_seconds, (tuple, list)) or len(
        delay_bounds_seconds
    ) != 2:
        raise ValueError("delay_bounds_seconds must contain two values")
    bounds = np.asarray(delay_bounds_seconds, dtype=float)
    if (
        not np.all(np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
    ):
        raise ValueError("delay_bounds_seconds must be a finite interval")
    if (
        isinstance(delay_scale_seconds, (bool, np.bool_))
        or not isinstance(delay_scale_seconds, Real)
    ):
        raise ValueError("delay_scale_seconds must be a positive real scalar")
    scale = float(delay_scale_seconds)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("delay_scale_seconds must be positive and finite")
    if np.any(draws.delay < bounds[0]) or np.any(draws.delay > bounds[1]):
        raise ValueError("retained delays must lie within request bounds")
    static = np.linalg.solve(
        cholesky, (draws.static_coordinate - mean[None, :]).T
    ).T
    midpoint = 0.5 * float(bounds[0] + bounds[1])
    normalized_delay = (draws.delay - midpoint) / scale
    result = np.column_stack((static, normalized_delay))
    if result.shape != (draws.count, STATIC_PARAMETER_DIMENSION + 1):
        raise RuntimeError("posterior feature dimension changed")
    return result


def _pairwise_squared_distance(features: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(features * features, axis=1)
    distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * features.dot(features.T)
    )
    np.maximum(distance, 0.0, out=distance)
    np.fill_diagonal(distance, 0.0)
    return distance


def _pam_medoids(features: np.ndarray, count: int) -> Tuple[int, ...]:
    population = int(features.shape[0])
    selected_count = min(_positive_integer(count, "medoid count"), population)
    if selected_count == population:
        return tuple(range(population))
    distance = _pairwise_squared_distance(features)
    medoids = []
    first_cost = np.sum(distance, axis=0)
    medoids.append(min(range(population), key=lambda index: (first_cost[index], index)))
    nearest = distance[:, medoids[0]].copy()
    while len(medoids) < selected_count:
        candidates = tuple(index for index in range(population) if index not in medoids)
        selected = min(
            candidates,
            key=lambda index: (
                float(np.sum(np.minimum(nearest, distance[:, index]))),
                index,
            ),
        )
        medoids.append(selected)
        nearest = np.minimum(nearest, distance[:, selected])

    medoids = sorted(medoids)
    while True:
        current_cost = float(np.sum(np.min(distance[:, medoids], axis=1)))
        best_cost = current_cost
        best = tuple(medoids)
        non_medoids = tuple(
            index for index in range(population) if index not in medoids
        )
        for removed in medoids:
            for added in non_medoids:
                candidate = tuple(sorted(
                    added if value == removed else value for value in medoids
                ))
                cost = float(np.sum(np.min(distance[:, candidate], axis=1)))
                if cost < best_cost or (cost == best_cost and candidate < best):
                    best_cost = cost
                    best = candidate
        if best_cost >= current_cost:
            return tuple(medoids)
        medoids = list(best)


def select_posterior_representatives(
    *,
    sample_id: Sequence[object],
    chain_id: Sequence[object],
    draw_index: object,
    static_coordinate: object,
    delay: object,
    log_posterior: object,
    source_mode_id: Sequence[object],
    prior_mean_coordinate: object,
    prior_covariance: object,
    delay_bounds_seconds: Sequence[object],
    delay_scale_seconds: object,
    exact_ridge_direction: object,
) -> PosteriorRepresentativeSelection:
    """Select the strict role union from equal-weight retained MCMC draws."""

    maximum = POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT
    draws = _canonical_draws(
        sample_id=sample_id,
        chain_id=chain_id,
        draw_index=draw_index,
        static_coordinate=static_coordinate,
        delay=delay,
        log_posterior=log_posterior,
        source_mode_id=source_mode_id,
    )
    mean = _finite_vector(
        prior_mean_coordinate,
        (STATIC_PARAMETER_DIMENSION,),
        "prior_mean_coordinate",
    )
    ridge = _canonical_ridge_direction(exact_ridge_direction)
    ridge_coordinate = (draws.static_coordinate - mean[None, :]).dot(ridge)

    primary = min(
        range(draws.count),
        key=lambda index: (-draws.log_posterior[index], draws.canonical_key(index)),
    )
    mandatory_roles = [
        RepresentativeRole(
            "highest_log_posterior",
            "primary",
            draws.sample_id[primary],
        )
    ]
    for mode_id in sorted(set(draws.source_mode_id)):
        members = tuple(
            index
            for index, value in enumerate(draws.source_mode_id)
            if value == mode_id
        )
        selected = min(
            members,
            key=lambda index: (
                -draws.log_posterior[index],
                draws.canonical_key(index),
            ),
        )
        mandatory_roles.append(
            RepresentativeRole(
                "mode_representative",
                "mode:{}".format(mode_id),
                draws.sample_id[selected],
                mode_id=mode_id,
            )
        )
    ridge_order = tuple(
        sorted(
            range(draws.count),
            key=lambda index: (
                ridge_coordinate[index],
                draws.canonical_key(index),
            ),
        )
    )
    for numerator, denominator in RIDGE_QUANTILES:
        selected = ridge_order[
            _nearest_empirical_index(draws.count, numerator, denominator)
        ]
        mandatory_roles.append(
            RepresentativeRole(
                "exact_ridge_quantile",
                "ridge_quantile:{}/{}".format(numerator, denominator),
                draws.sample_id[selected],
                quantile_numerator=numerator,
                quantile_denominator=denominator,
            )
        )

    selected_ids = []
    roles = []
    for role in mandatory_roles:
        if role.sample_id not in selected_ids:
            if len(selected_ids) >= maximum:
                raise ValueError(
                    "maximum_sample_count cannot contain all mandatory roles"
                )
            selected_ids.append(role.sample_id)
        roles.append(role)

    features = _posterior_features(
        draws,
        prior_mean_coordinate,
        prior_covariance,
        delay_bounds_seconds,
        delay_scale_seconds,
    )
    medoids = _pam_medoids(
        features,
        min(PID_STRATIFIED_MAXIMUM_REPRESENTATIVE_COUNT, draws.count),
    )
    for ordinal, selected in enumerate(medoids):
        sample = draws.sample_id[selected]
        if sample not in selected_ids:
            if len(selected_ids) >= maximum:
                continue
            selected_ids.append(sample)
        roles.append(
            RepresentativeRole(
                "pid_stratified_medoid",
                "pid_stratified_medoid:{:02d}".format(ordinal),
                sample,
                ordinal=ordinal,
            )
        )
    return PosteriorRepresentativeSelection(
        available_sample_count=draws.count,
        maximum_sample_count=maximum,
        selected_sample_ids=tuple(selected_ids),
        role_records=tuple(roles),
    )


def select_posterior_representatives_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    prior_mean_coordinate: object,
    prior_covariance: object,
    delay_bounds_seconds: Sequence[object],
    delay_scale_seconds: object,
    exact_ridge_direction: object,
) -> PosteriorRepresentativeSelection:
    """Recompute the policy directly from strict MCMC artifact arrays."""

    required = (
        "sample_id",
        "chain_id",
        "draw_index",
        "parameter_coordinate",
        "delay",
        "log_posterior",
        "source_mode_id",
    )
    missing = tuple(name for name in required if name not in arrays)
    if missing:
        raise ValueError(
            "MCMC arrays are missing representative inputs: {}".format(
                ", ".join(missing)
            )
        )
    return select_posterior_representatives(
        sample_id=arrays["sample_id"].tolist(),
        chain_id=arrays["chain_id"].tolist(),
        draw_index=arrays["draw_index"],
        static_coordinate=arrays["parameter_coordinate"],
        delay=arrays["delay"],
        log_posterior=arrays["log_posterior"],
        source_mode_id=arrays["source_mode_id"].tolist(),
        prior_mean_coordinate=prior_mean_coordinate,
        prior_covariance=prior_covariance,
        delay_bounds_seconds=delay_bounds_seconds,
        delay_scale_seconds=delay_scale_seconds,
        exact_ridge_direction=exact_ridge_direction,
    )


__all__ = [
    "CONDITIONAL_TRAJECTORY_EVALUATION_METHOD",
    "CONDITIONAL_TRAJECTORY_SAMPLE_ORDER",
    "CONDITIONAL_TRAJECTORY_SELECTION_POLICY",
    "CONDITIONAL_TRAJECTORY_WARM_START_POLICY",
    "FEATURE_POLICY",
    "PID_STRATIFIED_MAXIMUM_REPRESENTATIVE_COUNT",
    "POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT",
    "PosteriorRepresentativeSelection",
    "RIDGE_COORDINATE_POLICY",
    "RIDGE_QUANTILES",
    "ROLE_PRIORITY",
    "ROLE_RECORD_KEYS",
    "RepresentativeRole",
    "SELECTION_MANIFEST_KEYS",
    "select_posterior_representatives",
    "select_posterior_representatives_from_arrays",
]
