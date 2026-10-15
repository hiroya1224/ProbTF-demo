"""Shared initial-transform laws under a declared deterministic control map.

Native source laws remain authoritative. Exact finite sums and joint Monte
Carlo realizations have distinct query types and approximation metadata.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from probtf.kernels.base import TransformKernelExpression
from probtf.provenance import ApproximationInfo, ApproximationKind


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _weights(value, count):
    weights = _readonly(value)
    if (weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.)
            or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-10)):
        raise ValueError("query weights must be normalized source probabilities")
    return weights


@dataclass(frozen=True)
class JointTransformLaw:
    """A source law and its complete shared local-transform control map."""
    source_law: object
    generator: object
    frame_names: tuple
    model_id: str
    fixed_local_indices: tuple = ()
    relative_generator: object = None
    relative_generator_id: str = ""
    _cache: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self):
        frames = tuple(self.frame_names)
        if (not frames or frames != tuple(self.source_law.frame_names)
                or not self.model_id or not callable(self.generator)):
            raise ValueError("joint law requires matching source frames, model ID and control map")
        if self.relative_generator is not None:
            if (not callable(self.relative_generator) or not isinstance(self.relative_generator_id, str)
                    or not self.relative_generator_id.strip()):
                raise ValueError("relative control map requires a callable and explicit model ID")
        elif self.relative_generator_id:
            raise ValueError("relative model ID requires a relative control map")
        indices = tuple(sorted({index % len(frames) for index in self.fixed_local_indices}))
        object.__setattr__(self, "frame_names", frames)
        object.__setattr__(self, "fixed_local_indices", indices)

    def at(self, command):
        if np.ndim(command) != 1 or not np.all(np.isfinite(command)):
            raise ValueError("command must be a finite vector")
        return JointTransformState(self, tuple(float(value) for value in command))

    def descriptor(self):
        return dict(source_law=self.source_law.descriptor(), model_id=self.model_id,
                    frame_names=self.frame_names, fixed_local_indices=self.fixed_local_indices,
                    relative_generator_id=self.relative_generator_id or None,
                    result_kind="joint_transform_law",
                    generator="shared initial transform law -> declared deterministic control map")


@dataclass(frozen=True)
class JointTransformState:
    law: JointTransformLaw
    command: tuple

    @property
    def query_id(self):
        payload = (self.law.descriptor(), self.command)
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    def evaluate(self):
        key = self.query_id
        if key not in self.law._cache:
            source = self.law.source_law
            initial, weights = source.quadrature()
            local = self.law.generator(initial, np.array(self.command))
            if np.shape(local) != (len(weights), len(self.law.frame_names), 4, 4):
                raise ValueError("control map must return the complete local transform tuple")
            for index in self.law.fixed_local_indices:
                positive = local[np.asarray(weights) > 0., index]
                if not np.allclose(positive, positive[0], rtol=0., atol=1e-12):
                    raise ValueError("control map violates its declared deterministic attachment")
            metadata = dict(source.query_metadata(), query_id=key, model_id=self.law.model_id)
            result = JointTransformEvaluation(source.node_ids, weights, local, self, metadata)
            if len(self.law._cache) >= 16:
                self.law._cache.pop(next(iter(self.law._cache)))
            self.law._cache[key] = result
        return self.law._cache[key]

    def root(self, index):
        return JointTransformExpression(self, index)

    def relative(self, before, index, reference_index=-2):
        return JointTransformExpression(self, index, before, reference_index)


@dataclass(frozen=True)
class JointTransformEvaluation:
    body_ids: tuple
    weights: np.ndarray
    local_transforms: np.ndarray
    expression: object = None
    metadata: dict = field(default_factory=dict)
    root_transforms: np.ndarray = field(init=False)

    def __post_init__(self):
        ids = tuple(self.body_ids)
        weights, local = _weights(self.weights, len(ids)), _readonly(self.local_transforms)
        if (not ids or len(set(ids)) != len(ids) or local.ndim != 4 or local.shape[0] != len(ids)
                or not local.shape[1] or local.shape[-2:] != (4, 4) or not np.all(np.isfinite(local))):
            raise ValueError("invalid complete joint transform evaluation")
        root = np.empty_like(local)
        current = np.broadcast_to(np.eye(4), (len(ids), 4, 4))
        for index in range(local.shape[1]):
            current = current @ local[:, index]
            root[:, index] = current
        object.__setattr__(self, "body_ids", ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "local_transforms", local)
        object.__setattr__(self, "root_transforms", _readonly(root))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def candidate_ids(self):
        return self.body_ids

    def relative_transforms(self, before, reference_index=-2):
        if (self.body_ids != before.body_ids or not np.array_equal(self.weights, before.weights)
                or (self.expression is not None and before.expression is not None
                    and self.expression.law is not before.expression.law)):
            raise ValueError("before/after must share the same source snapshot and joint realizations")
        reference_index %= self.local_transforms.shape[1]
        law = None if self.expression is None else self.expression.law
        if law is not None and before.expression is not None and law.relative_generator is not None:
            initial, _weights = law.source_law.quadrature()
            result = np.asarray(law.relative_generator(initial, np.array(before.expression.command),
                np.array(self.expression.command), reference_index), dtype=float)
            if result.shape != self.local_transforms.shape or not np.all(np.isfinite(result)):
                raise ValueError("relative control map must return the complete finite transform tuple")
            result = result.copy()
        else:
            result = np.linalg.inv(before.root_transforms[:, reference_index])[:, None] @ self.root_transforms
        if self is before:
            result[:, reference_index] = np.eye(4)
            if reference_index+1 < self.local_transforms.shape[1]:
                result[:, reference_index+1] = self.local_transforms[:, reference_index+1]
        return _readonly(result)


@dataclass(frozen=True)
class TransformEvaluation:
    transforms: np.ndarray
    weights: np.ndarray
    metadata: dict
    expression: object = None

    def __post_init__(self):
        transforms = _readonly(self.transforms)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not np.all(np.isfinite(transforms)):
            raise ValueError("transform evaluation requires finite (N,4,4) matrices")
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "weights", _weights(self.weights, len(transforms)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def apply(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point must have three finite coordinates")
        points = np.einsum("sij,j->si", self.transforms[:, :3, :3], point) + self.transforms[:, :3, 3]
        method = self.metadata["integration_method"]
        if method == "finite_sum":
            return PointQuadrature(points, self.weights, self.metadata, self.expression)
        if method == "monte_carlo":
            return PointSamples(points, self.weights, self.metadata, self.expression)
        raise ValueError("unresolved joint evaluation method")


@dataclass(frozen=True)
class _PointEvaluation:
    points: np.ndarray
    weights: np.ndarray
    metadata: dict
    expression: object = None

    def __post_init__(self):
        points = _readonly(self.points)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("point evaluation requires finite (N,3) points")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", _weights(self.weights, len(points)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def source_law(self):
        return None if self.expression is None else self.expression.state.law.source_law

    @property
    def approximation(self):
        sampled = isinstance(self, PointSamples)
        return ApproximationInfo(kind=ApproximationKind.MONTE_CARLO if sampled else ApproximationKind.EXACT,
            lossy=sampled, source=self.metadata.get("source_law_id", ""),
            detail=json.dumps(dict(self.metadata), sort_keys=True))

    def expected_log_density(self, density):
        values, gradients = density.log_density_and_gradient(self.points)
        with np.errstate(divide="ignore"):
            terms = values + np.log(self.weights)
        score = float(logsumexp(terms))
        if not np.isfinite(score):
            raise FloatingPointError("expected density must have finite positive mass")
        return score, np.exp(terms-score)[:, None]*gradients


@dataclass(frozen=True)
class PointQuadrature(_PointEvaluation):
    """Exact finite-source weighted point-action sum."""

    def __post_init__(self):
        super().__post_init__()
        if self.metadata.get("integration_method", "finite_sum") != "finite_sum":
            raise ValueError("PointQuadrature requires an exact finite-source sum")


@dataclass(frozen=True)
class PointSamples(_PointEvaluation):
    """Derived joint Monte Carlo realizations of an unreplaced native law."""

    def __post_init__(self):
        super().__post_init__()
        if self.metadata.get("integration_method") != "monte_carlo":
            raise ValueError("PointSamples requires explicit Monte Carlo query metadata")


@dataclass(frozen=True)
class JointTransformExpression(TransformKernelExpression):
    state: JointTransformState
    index: int
    before: object = None
    reference_index: int = -2

    def __post_init__(self):
        if not isinstance(self.state, JointTransformState):
            raise TypeError("state must reference a joint transform law snapshot")
        if self.before is not None and (not isinstance(self.before, JointTransformState)
                                       or self.before.law is not self.state.law):
            raise ValueError("relative queries must resolve the same shared source law snapshot")

    def latent_dependency_ids(self):
        return frozenset((self.state.law.source_law.law_id,))

    def evaluate(self):
        after = self.state.evaluate()
        index = self.index % len(self.state.law.frame_names)
        reference = self.reference_index % len(self.state.law.frame_names)
        transforms = after.root_transforms[:, index]
        if self.before is not None:
            before = self.before.evaluate()
            if self.state.command == self.before.command and index == reference:
                transforms = np.broadcast_to(np.eye(4), transforms.shape)
            elif self.state.command == self.before.command and index == reference+1:
                transforms = after.local_transforms[:, index]
            elif self.state.law.relative_generator is not None:
                law = self.state.law
                initial, _weights = law.source_law.quadrature()
                transforms = law.relative_generator(initial, np.array(self.before.command),
                    np.array(self.state.command), reference, index)
            else:
                transforms = np.linalg.inv(before.root_transforms[:, reference]) @ transforms
        metadata = dict(after.metadata, operation="root" if self.before is None else "paired_relative",
            link_index=index, reference_query_id=None if self.before is None else self.before.query_id)
        return TransformEvaluation(transforms, after.weights, metadata, self)

    def quadrature(self):
        result = self.evaluate()
        if result.metadata["integration_method"] != "finite_sum":
            raise ValueError("continuous native source uses joint samples, not a finite quadrature law")
        return result

    def samples(self):
        result = self.evaluate()
        if result.metadata["integration_method"] != "monte_carlo":
            raise ValueError("finite source is evaluated by an exact sum, not Monte Carlo")
        return result
