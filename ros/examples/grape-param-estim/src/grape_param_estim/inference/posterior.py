"""Full weighted empirical plant posterior and joint HPD subset."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.plant.parameters import PlantHypothesis


def _readonly(values: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite {}D array".format(name, ndim))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class PlantPosterior:
    particles: Tuple[PlantHypothesis, ...]
    weights: np.ndarray
    log_likelihood: np.ndarray
    raw_parameters: np.ndarray
    derived_parameters: np.ndarray
    raw_parameter_names: Tuple[str, ...]
    derived_parameter_names: Tuple[str, ...]
    model_id: str
    prior_id: str
    likelihood_id: str
    controller_snapshot_id: str
    credible_probability: float = 0.95
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        particles = tuple(self.particles)
        if not particles or any(
            not isinstance(item, PlantHypothesis) for item in particles
        ):
            raise TypeError("particles must contain PlantHypothesis objects")
        weights = _readonly(self.weights, "weights", 1)
        likelihood = _readonly(
            self.log_likelihood, "log_likelihood", 1
        )
        raw = _readonly(self.raw_parameters, "raw_parameters", 2)
        derived = _readonly(
            self.derived_parameters, "derived_parameters", 2
        )
        count = len(particles)
        if (
            weights.shape != (count,)
            or likelihood.shape != (count,)
            or raw.shape[0] != count
            or derived.shape[0] != count
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0)
        ):
            raise ValueError("posterior arrays/weights are inconsistent")
        raw_names = tuple(str(item) for item in self.raw_parameter_names)
        derived_names = tuple(
            str(item) for item in self.derived_parameter_names
        )
        if (
            raw.shape[1] != len(raw_names)
            or derived.shape[1] != len(derived_names)
            or len(set(raw_names)) != len(raw_names)
            or len(set(derived_names)) != len(derived_names)
        ):
            raise ValueError("posterior parameter names are inconsistent")
        credible = float(self.credible_probability)
        if not 0.0 < credible < 1.0:
            raise ValueError("credible_probability must lie in (0, 1)")
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "log_likelihood", likelihood)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "derived_parameters", derived)
        object.__setattr__(self, "raw_parameter_names", raw_names)
        object.__setattr__(self, "derived_parameter_names", derived_names)
        object.__setattr__(self, "credible_probability", credible)
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )

    @property
    def mean(self) -> np.ndarray:
        result = np.average(
            self.raw_parameters, axis=0, weights=self.weights
        )
        result.setflags(write=False)
        return result

    @property
    def covariance(self) -> np.ndarray:
        centered = self.raw_parameters - self.mean
        result = (centered * self.weights[:, None]).T @ centered
        result.setflags(write=False)
        return result

    @property
    def correlation(self) -> np.ndarray:
        covariance = self.covariance
        scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(scale, scale)
        result = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
        np.fill_diagonal(result, np.where(scale > 0.0, 1.0, 0.0))
        result.setflags(write=False)
        return result

    @property
    def hpd_indices(self) -> np.ndarray:
        order = np.argsort(-self.weights, kind="stable")
        cumulative = np.cumsum(self.weights[order])
        count = int(
            np.searchsorted(
                cumulative, self.credible_probability, side="left"
            )
            + 1
        )
        result = np.array(order[:count], copy=True)
        result.setflags(write=False)
        return result

    @property
    def hpd_weight(self) -> float:
        return float(np.sum(self.weights[self.hpd_indices]))

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.dot(self.weights, self.weights))

    def multimodality_diagnostic(self, bins: int = 24) -> Mapping[str, Any]:
        modes = {}
        for index, name in enumerate(self.raw_parameter_names):
            histogram, _ = np.histogram(
                self.raw_parameters[:, index],
                bins=int(bins),
                weights=self.weights,
            )
            local = 0
            for item in range(1, histogram.size - 1):
                if (
                    histogram[item] > histogram[item - 1]
                    and histogram[item] >= histogram[item + 1]
                    and histogram[item] > 0.02
                ):
                    local += 1
            modes[name] = max(1, local)
        return MappingProxyType(
            {
                "per_parameter_mode_count": MappingProxyType(modes),
                "any_multimodal": bool(any(value > 1 for value in modes.values())),
                "method": "weighted_histogram_local_maxima_v1",
            }
        )

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "weights": self.weights,
                "log_likelihood": self.log_likelihood,
                "raw_parameters": self.raw_parameters,
                "derived_parameters": self.derived_parameters,
                "raw_parameter_names": self.raw_parameter_names,
                "derived_parameter_names": self.derived_parameter_names,
                "model_id": self.model_id,
                "prior_id": self.prior_id,
                "likelihood_id": self.likelihood_id,
                "controller_snapshot_id": self.controller_snapshot_id,
                "credible_probability": self.credible_probability,
                "provenance": self.provenance,
            }
        )

    @classmethod
    def from_arrays(
        cls,
        particles: Sequence[PlantHypothesis],
        weights: Any,
        log_likelihood: Any,
        model_id: str,
        prior_id: str,
        likelihood_id: str,
        controller_snapshot_id: str,
        credible_probability: float = 0.95,
        provenance: Mapping[str, Any] = None,
    ) -> "PlantPosterior":
        hypotheses = tuple(particles)
        raw = np.stack([item.vector for item in hypotheses])
        names = tuple(
            list(hypotheses[0].plant_parameter_names)
            + list(hypotheses[0].actuator_parameter_names)
            + [
                "disturbance_{}".format(index)
                for index in range(
                    hypotheses[0].disturbance_parameters.size
                )
            ]
        )
        derived_names = tuple(
            sorted(
                {
                    key
                    for item in hypotheses
                    for key in item.derived_quantities.keys()
                }
            )
        )
        derived = np.asarray(
            [
                [item.derived_quantities.get(name, np.nan) for name in derived_names]
                for item in hypotheses
            ],
            dtype=float,
        )
        if derived_names and not np.all(np.isfinite(derived)):
            raise ValueError(
                "every posterior particle must provide the same derived quantities"
            )
        if not derived_names:
            derived = np.empty((len(hypotheses), 0))
        return cls(
            particles=hypotheses,
            weights=np.asarray(weights, dtype=float),
            log_likelihood=np.asarray(log_likelihood, dtype=float),
            raw_parameters=raw,
            derived_parameters=derived,
            raw_parameter_names=names,
            derived_parameter_names=derived_names,
            model_id=model_id,
            prior_id=prior_id,
            likelihood_id=likelihood_id,
            controller_snapshot_id=controller_snapshot_id,
            credible_probability=credible_probability,
            provenance={} if provenance is None else provenance,
        )


__all__ = ["PlantPosterior"]
