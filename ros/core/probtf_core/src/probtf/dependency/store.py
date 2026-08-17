"""Thread-safe versioned storage for local Gaussian dependencies."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from threading import RLock

import numpy as np

from probtf.dependency.binding import (
    EdgeLatentBinding,
    POSE_PERTURBATION_CONVENTION,
)
from probtf.dependency.gaussian import (
    GaussianLatentFactor,
    GaussianObservationFactor,
    GaussianUpdateResult,
    immutable_matrix,
    immutable_psd_matrix,
)
from probtf.provenance import Provenance


def _cross_key(left, right):
    return (left, right) if left < right else (right, left)


def _unique(values):
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return tuple(output)


@dataclass(frozen=True)
class GaussianLatentSnapshot:
    """One immutable, atomic view of factors, bindings, and cross blocks."""

    revision: int
    factors: object
    bindings_by_edge: object
    cross_covariances: object

    def __post_init__(self):
        revision = int(self.revision)
        if revision < 0 or revision != self.revision:
            raise ValueError("revision must be a non-negative integer.")
        factors = dict(self.factors)
        if any(
            key != factor.factor_id or not isinstance(factor, GaussianLatentFactor)
            for key, factor in factors.items()
        ):
            raise ValueError("factors must map factor IDs to matching factors.")
        bindings = {}
        for edge_id, values in dict(self.bindings_by_edge).items():
            entries = tuple(values)
            if any(
                not isinstance(binding, EdgeLatentBinding)
                or binding.edge_id != edge_id
                for binding in entries
            ):
                raise ValueError("bindings_by_edge contains an invalid binding.")
            if len({binding.factor_id for binding in entries}) != len(entries):
                raise ValueError("An edge may have at most one binding per factor.")
            for binding in entries:
                factor = factors.get(binding.factor_id)
                if factor is None:
                    raise ValueError("A binding references an unknown factor.")
                if binding.latent_dimension != factor.dimension:
                    raise ValueError("Binding and factor dimensions do not match.")
                if binding.factor_version != factor.version:
                    raise ValueError("Binding and factor versions do not match.")
                if binding.perturbation_convention != POSE_PERTURBATION_CONVENTION:
                    raise ValueError("Binding perturbation convention is unsupported.")
            bindings[str(edge_id)] = entries
        cross = {}
        for key, value in dict(self.cross_covariances).items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key[0] >= key[1]
                or key[0] not in factors
                or key[1] not in factors
            ):
                raise ValueError("cross_covariances contains an invalid factor pair.")
            shape = (factors[key[0]].dimension, factors[key[1]].dimension)
            cross[key] = immutable_matrix(value, shape, "cross_covariance")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "factors", MappingProxyType(factors))
        object.__setattr__(self, "bindings_by_edge", MappingProxyType(bindings))
        object.__setattr__(self, "cross_covariances", MappingProxyType(cross))

    def factor(self, factor_id):
        return self.factors[str(factor_id)]

    def bindings_for_edge(self, edge_id):
        return self.bindings_by_edge.get(str(edge_id), ())

    def bindings_for_path(self, edge_ids):
        return tuple(
            binding
            for edge_id in edge_ids
            for binding in self.bindings_for_edge(edge_id)
        )

    def factor_versions(self, factor_ids):
        return tuple(
            (factor_id, self.factor(factor_id).version)
            for factor_id in factor_ids
        )

    def joint_mean_covariance(self, factor_ids):
        order = tuple(str(value) for value in factor_ids)
        if len(set(order)) != len(order):
            raise ValueError("factor_ids must be unique.")
        factors = tuple(self.factor(factor_id) for factor_id in order)
        offsets = {}
        size = 0
        for factor in factors:
            offsets[factor.factor_id] = slice(size, size + factor.dimension)
            size += factor.dimension
        mean = np.empty(size, dtype=float)
        covariance = np.zeros((size, size), dtype=float)
        for factor in factors:
            selected = offsets[factor.factor_id]
            mean[selected] = factor.mean
            covariance[selected, selected] = factor.covariance
        for left_index, left in enumerate(order):
            for right in order[left_index + 1 :]:
                key = _cross_key(left, right)
                block = self.cross_covariances.get(key)
                if block is None:
                    continue
                selected = block if key == (left, right) else block.T
                covariance[offsets[left], offsets[right]] = selected
                covariance[offsets[right], offsets[left]] = selected.T
        covariance = 0.5 * (covariance + covariance.T)
        mean.setflags(write=False)
        covariance.setflags(write=False)
        return mean, covariance, MappingProxyType(offsets)


class GaussianLatentStore:
    """Own the mutable posterior while exposing only immutable snapshots."""

    def __init__(self):
        self._factors = {}
        self._bindings_by_edge = {}
        self._cross_covariances = {}
        self._revision = 0
        self._lock = RLock()

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def snapshot(self):
        with self._lock:
            return GaussianLatentSnapshot(
                self._revision,
                self._factors,
                self._bindings_by_edge,
                self._cross_covariances,
            )

    def _refresh_binding_version_locked(self, factor_id, version):
        for edge_id, bindings in tuple(self._bindings_by_edge.items()):
            self._bindings_by_edge[edge_id] = tuple(
                replace(binding, factor_version=version)
                if binding.factor_id == factor_id
                else binding
                for binding in bindings
            )

    def _drop_cross_covariances_locked(self, factor_id):
        self._cross_covariances = {
            key: value
            for key, value in self._cross_covariances.items()
            if factor_id not in key
        }

    def put_factor(
        self,
        factor_id,
        mean,
        covariance,
        stamp,
        provenance=None,
    ):
        """Insert or replace a factor and monotonically advance its version."""

        with self._lock:
            old = self._factors.get(str(factor_id).strip())
            version = 1 if old is None else old.version + 1
            selected_provenance = Provenance() if provenance is None else provenance
            factor = GaussianLatentFactor(
                factor_id,
                mean,
                covariance,
                stamp,
                version,
                selected_provenance,
            )
            for bindings in self._bindings_by_edge.values():
                for binding in bindings:
                    if (
                        binding.factor_id == factor.factor_id
                        and binding.latent_dimension != factor.dimension
                    ):
                        raise ValueError(
                            "Cannot change the dimension of a bound factor."
                        )
            self._factors[factor.factor_id] = factor
            self._drop_cross_covariances_locked(factor.factor_id)
            self._refresh_binding_version_locked(factor.factor_id, factor.version)
            self._revision += 1
            return factor

    def replace_factor(self, factor, *, expected_version=None):
        if not isinstance(factor, GaussianLatentFactor):
            raise TypeError("factor must be GaussianLatentFactor.")
        with self._lock:
            old = self._factors.get(factor.factor_id)
            if old is not None:
                if expected_version is not None and old.version != int(expected_version):
                    raise RuntimeError("Factor version changed before replacement.")
                if factor.version <= old.version:
                    raise ValueError("Replacement factor version must increase.")
                if factor.dimension != old.dimension and any(
                    binding.factor_id == factor.factor_id
                    for bindings in self._bindings_by_edge.values()
                    for binding in bindings
                ):
                    raise ValueError("Cannot change the dimension of a bound factor.")
            elif factor.version != 1:
                raise ValueError("A newly inserted factor must start at version 1.")
            self._factors[factor.factor_id] = factor
            self._drop_cross_covariances_locked(factor.factor_id)
            self._refresh_binding_version_locked(factor.factor_id, factor.version)
            self._revision += 1
            return factor

    def bind_edge(self, binding):
        if not isinstance(binding, EdgeLatentBinding):
            raise TypeError("binding must be EdgeLatentBinding.")
        with self._lock:
            factor = self._factors.get(binding.factor_id)
            if factor is None:
                raise KeyError(binding.factor_id)
            if binding.factor_version != factor.version:
                raise ValueError("Binding factor_version is stale.")
            if binding.latent_dimension != factor.dimension:
                raise ValueError("Binding and factor dimensions do not match.")
            existing = tuple(
                item
                for item in self._bindings_by_edge.get(binding.edge_id, ())
                if item.factor_id != binding.factor_id
            )
            self._bindings_by_edge[binding.edge_id] = existing + (binding,)
            self._revision += 1
            return binding

    def unbind_edge(self, edge_id, factor_id=None):
        edge_id = str(edge_id).strip()
        with self._lock:
            existing = self._bindings_by_edge.get(edge_id, ())
            if factor_id is None:
                removed = existing
                self._bindings_by_edge.pop(edge_id, None)
            else:
                factor_id = str(factor_id).strip()
                removed = tuple(
                    binding for binding in existing if binding.factor_id == factor_id
                )
                retained = tuple(
                    binding for binding in existing if binding.factor_id != factor_id
                )
                if retained:
                    self._bindings_by_edge[edge_id] = retained
                else:
                    self._bindings_by_edge.pop(edge_id, None)
            if removed:
                self._revision += 1
            return removed

    def _correlated_closure_locked(self, factor_ids):
        selected = set(factor_ids)
        changed = True
        while changed:
            changed = False
            for left, right in self._cross_covariances:
                if left in selected or right in selected:
                    before = len(selected)
                    selected.update((left, right))
                    changed = changed or len(selected) != before
        return selected

    def _snapshot_locked(self):
        return GaussianLatentSnapshot(
            self._revision,
            self._factors,
            self._bindings_by_edge,
            self._cross_covariances,
        )

    def apply_observation(self, observation, *, expected_versions=None):
        """Condition the correlated factor closure in one atomic transaction."""

        if not isinstance(observation, GaussianObservationFactor):
            raise TypeError("observation must be GaussianObservationFactor.")
        with self._lock:
            missing = [
                factor_id
                for factor_id in observation.latent_factor_ids
                if factor_id not in self._factors
            ]
            if missing:
                raise KeyError(tuple(missing))
            if expected_versions is not None:
                for factor_id, version in dict(expected_versions).items():
                    if self._factors[factor_id].version != int(version):
                        raise RuntimeError(
                            "Factor version changed before observation update."
                        )
            for factor_id, block in zip(
                observation.latent_factor_ids,
                observation.jacobian_blocks,
            ):
                if block.shape[1] != self._factors[factor_id].dimension:
                    raise ValueError(
                        "Observation Jacobian dimension does not match factor {!r}.".format(
                            factor_id
                        )
                    )

            closure = self._correlated_closure_locked(
                observation.latent_factor_ids
            )
            order = tuple(observation.latent_factor_ids) + tuple(
                sorted(closure.difference(observation.latent_factor_ids))
            )
            prior_snapshot = self._snapshot_locked()
            mean, covariance, slices = prior_snapshot.joint_mean_covariance(order)
            mean = np.array(mean, copy=True)
            covariance = np.array(covariance, copy=True)
            rows = observation.residual.size
            jacobian = np.zeros((rows, mean.size), dtype=float)
            for factor_id, block in zip(
                observation.latent_factor_ids,
                observation.jacobian_blocks,
            ):
                jacobian[:, slices[factor_id]] = block

            innovation = (
                jacobian @ covariance @ jacobian.T
                + observation.noise_covariance
            )
            innovation = 0.5 * (innovation + innovation.T)
            gain = np.linalg.solve(
                innovation,
                jacobian @ covariance,
            ).T
            posterior_mean = mean - gain @ observation.residual
            identity = np.eye(mean.size, dtype=float)
            joseph = identity - gain @ jacobian
            posterior_covariance = (
                joseph @ covariance @ joseph.T
                + gain @ observation.noise_covariance @ gain.T
            )
            posterior_covariance = immutable_psd_matrix(
                posterior_covariance,
                mean.size,
                "posterior_covariance",
            )

            prior_versions = tuple(
                (factor_id, self._factors[factor_id].version)
                for factor_id in order
            )
            posterior_factors = {}
            for factor_id in order:
                old = self._factors[factor_id]
                combined_provenance = Provenance(
                    source_ids=_unique(
                        old.provenance.source_ids
                        + observation.provenance.source_ids
                    ),
                    derived_from_edge_ids=_unique(
                        old.provenance.derived_from_edge_ids
                        + observation.provenance.derived_from_edge_ids
                    ),
                    method="gaussian_observation_update",
                    detail=observation.observation_id,
                )
                selected = slices[factor_id]
                posterior_factors[factor_id] = GaussianLatentFactor(
                    factor_id,
                    posterior_mean[selected],
                    posterior_covariance[selected, selected],
                    observation.stamp,
                    old.version + 1,
                    combined_provenance,
                )

            for factor_id, factor in posterior_factors.items():
                self._factors[factor_id] = factor
            for key in tuple(self._cross_covariances):
                if key[0] in closure and key[1] in closure:
                    del self._cross_covariances[key]
            for left_index, left in enumerate(order):
                for right in order[left_index + 1 :]:
                    block = posterior_covariance[slices[left], slices[right]]
                    key = _cross_key(left, right)
                    selected = block if key == (left, right) else block.T
                    self._cross_covariances[key] = immutable_matrix(
                        selected,
                        (
                            posterior_factors[key[0]].dimension,
                            posterior_factors[key[1]].dimension,
                        ),
                        "cross_covariance",
                    )
            for factor_id, factor in posterior_factors.items():
                self._refresh_binding_version_locked(factor_id, factor.version)
            self._revision += 1
            posterior_versions = tuple(
                (factor_id, posterior_factors[factor_id].version)
                for factor_id in order
            )
            return GaussianUpdateResult(
                observation.observation_id,
                prior_versions,
                posterior_versions,
                innovation,
                gain,
                self._revision,
            )


__all__ = ["GaussianLatentSnapshot", "GaussianLatentStore"]
