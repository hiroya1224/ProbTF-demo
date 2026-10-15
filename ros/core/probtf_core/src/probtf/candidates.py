"""Configured initial local-transform laws, independent of a plant model.

Finite joint tuples are exact source laws. Continuous native edge laws stay
authoritative; a cached table of joint realizations is only a sampling query.
"""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, DistributionStatus,
    OrientationKind, TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_to_rotmat
from probtf.probability.sampling import sample_transform_distribution


def _readonly(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _identity(law):
    if not isinstance(law.law_id, str) or not law.law_id.strip():
        raise ValueError("source law_id must be a nonempty string")
    if isinstance(law.revision, bool) or not isinstance(law.revision, (int, np.integer)) or law.revision < 1:
        raise ValueError("source revision must be a positive integer")
    reference = tuple(float(value) for value in law.reference_command)
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_command must be finite")
    object.__setattr__(law, "reference_command", reference)


def _frames(values):
    names = tuple(values)
    if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("frame_names must be nonempty unique frame IDs")
    return names


def _weights(values, count):
    weights = np.asarray(values, dtype=float)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.) or not np.any(weights > 0.):
        raise ValueError("joint weights must be finite, nonnegative and have positive mass")
    scaled = weights / weights.max()
    return _readonly(scaled/scaled.sum())


def _rows(values, edge_count):
    rows = _readonly(values)
    if (rows.ndim != 4 or not len(rows) or rows.shape[1:] != (edge_count, 4, 4)
            or not np.all(np.isfinite(rows))):
        raise ValueError("local transform rows must have shape (N, edges, 4, 4)")
    rotation = rows[..., :3, :3]
    if (not np.allclose(rows[..., 3, :], [0., 0., 0., 1.], rtol=0., atol=1e-10)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), rtol=0., atol=1e-8)
            or not np.allclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-8)):
        raise ValueError("each local transform must belong to SE(3)")
    return rows


def _matrix(component):
    result = np.eye(4)
    result[:3, :3] = quat_to_rotmat(component.orientation.reference_quaternion_wxyz)
    result[:3, 3] = component.translation.mean_at_reference
    return result


def _is_dirac(component):
    # With a Dirac rotation, the conditional coupling is zero ON its support,
    # even when the original native 3x9 coupling matrix is nonzero.
    return (component.orientation.kind is OrientationKind.DIRAC
            and np.array_equal(component.translation.residual_covariance, np.zeros((3, 3))))


def _record_descriptor(record):
    components = []
    for component in record.distribution.components:
        orientation = component.orientation
        components.append(dict(id=component.component_id, raw_weight=component.raw_weight,
            orientation=dict(kind=orientation.kind.value,
                parameter_matrix=(orientation.backend_parameter_matrix().tolist()
                    if orientation.kind is OrientationKind.FINITE_BINGHAM else None),
                shape_matrix=orientation.shape_matrix.tolist(),
                reference_quaternion_wxyz=orientation.reference_quaternion_wxyz.tolist()),
            translation=dict(mean_at_reference=component.translation.mean_at_reference.tolist(),
                residual_covariance=component.translation.residual_covariance.tolist(),
                rotation_coupling=component.translation.rotation_coupling.tolist())))
    return dict(parent=record.parent_frame_id, child=record.child_frame_id, edge_id=record.edge_id,
                stamp=record.stamp, authority=record.authority, components=components)


@dataclass(frozen=True)
class JointFiniteTransformLaw:
    initial_local_transforms: np.ndarray
    weights: np.ndarray
    node_ids: tuple
    frame_names: tuple
    law_id: str = "initial_transforms"
    revision: int = 1
    reference_command: tuple = ()

    def __post_init__(self):
        _identity(self)
        frames = _frames(self.frame_names)
        rows = _rows(self.initial_local_transforms, len(frames))
        ids = tuple(self.node_ids)
        if (len(ids) != len(rows) or any(not isinstance(name, str) or not name for name in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("joint candidate IDs must be nonempty and unique")
        object.__setattr__(self, "frame_names", frames)
        object.__setattr__(self, "initial_local_transforms", rows)
        object.__setattr__(self, "weights", _weights(self.weights, len(rows)))
        object.__setattr__(self, "node_ids", ids)

    @property
    def is_finite(self):
        return True

    @property
    def deterministic_local_indices(self):
        positive = self.initial_local_transforms[self.weights > 0.]
        return tuple(index for index in range(len(self.frame_names))
                     if np.array_equal(positive[:, index], np.broadcast_to(positive[0, index], positive[:, index].shape)))

    def quadrature(self, order=None):
        return self.initial_local_transforms, self.weights

    def query_metadata(self):
        return dict(result_kind="joint_finite_pushforward", integration_method="finite_sum",
                    source_law_id=self.law_id, source_revision=self.revision,
                    node_count=len(self.weights), approximation="exact_finite_input_law", error_bound=None)

    def descriptor(self):
        return dict(law_id=self.law_id, revision=self.revision, kind="joint_finite",
                    frame_names=self.frame_names, reference_command=self.reference_command,
                    dependence="explicit complete transform tuples and joint weights",
                    node_ids=self.node_ids, weights=self.weights.tolist(),
                    initial_local_transforms=self.initial_local_transforms.tolist())


@dataclass(frozen=True)
class IndependentTransformLaw:
    edge_records: tuple
    independence: bool
    law_id: str = "initial_transforms"
    revision: int = 1
    sample_count: int = 64
    seed: int = 21
    reference_command: tuple = ()
    _rows: np.ndarray = field(init=False, repr=False, compare=False)
    _weights: np.ndarray = field(init=False, repr=False, compare=False)
    _node_ids: tuple = field(init=False, repr=False)
    _finite: bool = field(init=False, repr=False)
    _deterministic_indices: tuple = field(init=False, repr=False)

    def __post_init__(self):
        _identity(self)
        if self.independence is not True:
            raise ValueError("per-edge laws require explicit independence=True; unresolved dependence is unsupported")
        records = tuple(self.edge_records)
        if not records or any(not isinstance(record, TransformDistributionStamped) for record in records):
            raise TypeError("edge_records must contain native stamped transform laws")
        _frames(record.child_frame_id for record in records)
        if (len({record.edge_id for record in records}) != len(records)
                or any(first.child_frame_id != second.parent_frame_id for first, second in zip(records, records[1:]))):
            raise ValueError("local edges must form an ordered chain with unique physical edge IDs")
        normalized = []
        deterministic = []
        for index, record in enumerate(records):
            raw = np.array([component.raw_weight for component in record.distribution.components])
            distribution = record.distribution.normalize_weights()
            if np.any(raw < 0.) or distribution.status is not DistributionStatus.OK:
                raise ValueError("native input laws require valid nonnegative mixture weights")
            normalized.append(distribution.components)
            components = [item.component for item in distribution.components]
            if all(_is_dirac(component) for component in components):
                matrices = np.array([_matrix(component) for component in components])
                if np.array_equal(matrices, np.broadcast_to(matrices[0], matrices.shape)):
                    deterministic.append(index)
        finite = all(_is_dirac(item.component) for edge in normalized for item in edge)
        if finite:
            combinations = list(product(*normalized))
            rows = np.array([[_matrix(item.component) for item in joint] for joint in combinations])
            weights = _weights([np.prod([item.weight for item in joint]) for joint in combinations], len(rows))
            ids = tuple("joint_{:06d}".format(index) for index in range(len(rows)))
        else:
            if (isinstance(self.sample_count, bool) or not isinstance(self.sample_count, (int, np.integer))
                    or self.sample_count < 1 or isinstance(self.seed, bool)
                    or not isinstance(self.seed, (int, np.integer)) or self.seed < 0):
                raise ValueError("sampling policy requires a positive count and nonnegative integer seed")
            generator = np.random.default_rng(self.seed)
            rows = np.broadcast_to(np.eye(4), (self.sample_count, len(records), 4, 4)).copy()
            for index, record in enumerate(records):
                batch = sample_transform_distribution(record.distribution, self.sample_count, generator)
                rows[:, index, :3, :3] = [quat_to_rotmat(q) for q in batch.rotations_wxyz]
                rows[:, index, :3, 3] = batch.translations
            weights = _readonly(np.full(self.sample_count, 1./self.sample_count))
            ids = tuple("sample_{:06d}".format(index) for index in range(self.sample_count))
        object.__setattr__(self, "edge_records", records)
        object.__setattr__(self, "_finite", finite)
        object.__setattr__(self, "_rows", _rows(rows, len(records)))
        object.__setattr__(self, "_weights", weights)
        object.__setattr__(self, "_node_ids", ids)
        object.__setattr__(self, "_deterministic_indices", tuple(deterministic))

    @property
    def frame_names(self):
        return tuple(record.child_frame_id for record in self.edge_records)

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def is_finite(self):
        return self._finite

    @property
    def deterministic_local_indices(self):
        return self._deterministic_indices

    def quadrature(self, order=None):
        """Return the fixed query policy; changing order does NOT refine MC."""
        return self._rows, self._weights

    def query_metadata(self):
        result = dict(result_kind="joint_finite_pushforward" if self.is_finite else "joint_transform_samples",
                      integration_method="finite_sum" if self.is_finite else "monte_carlo",
                      source_law_id=self.law_id, source_revision=self.revision,
                      node_count=len(self.node_ids), error_bound=None)
        if self.is_finite:
            result["approximation"] = "exact_finite_input_law"
        else:
            result.update(approximation="derived_joint_samples", sample_count=self.sample_count, seed=self.seed)
        return result

    def descriptor(self):
        return dict(law_id=self.law_id, revision=self.revision, kind="independent_native_transforms",
                    frame_names=self.frame_names, reference_command=self.reference_command, independence=True,
                    edges=[_record_descriptor(record) for record in self.edge_records],
                    query_policy=(dict(method="finite_sum") if self.is_finite else
                                  dict(method="monte_carlo", sample_count=self.sample_count, seed=self.seed)))


def _orientation(value):
    kind = value["kind"]
    if kind == "dirac":
        return BinghamOrientation.dirac(value["quaternion_wxyz"])
    if kind == "uniform":
        return BinghamOrientation.uniform(value.get("reference_quaternion_wxyz", [1., 0., 0., 0.]))
    if kind in ("bingham", "finite_bingham"):
        return BinghamOrientation.from_parameter_matrix(value["parameter_matrix"], value.get("reference_quaternion_wxyz"))
    raise ValueError("unsupported native orientation kind: " + str(kind))


def _finite_matrix(value):
    if not isinstance(value, dict):
        return np.asarray(value, dtype=float)
    result = np.eye(4)
    result[:3, :3] = quat_to_rotmat(value["quaternion_wxyz"])
    result[:3, 3] = value["translation"]
    return result


def load_initial_transform_law(path):
    """Load explicit native source laws, never a stiffness/noise initializer."""
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = dict(law_id=config["law_id"], revision=config.get("revision", 1),
                  reference_command=tuple(config.get("reference_command", ())))
    if config["kind"] == "joint_finite":
        scenarios = config["scenarios"]
        return JointFiniteTransformLaw(
            np.array([[_finite_matrix(value) for value in row["local_transforms"]] for row in scenarios]),
            np.array([row["weight"] for row in scenarios]), tuple(row["id"] for row in scenarios),
            tuple(config["frame_names"]), **common)
    if config["kind"] != "independent":
        raise ValueError("only explicit joint_finite or independent native transform inputs are supported")
    records = []
    for edge in config["edges"]:
        components = []
        for item in edge["components"]:
            translation = item["translation"]
            components.append(TransformComponent(item["id"], item["weight"], _orientation(item["orientation"]),
                ConditionalGaussianTranslation(translation["mean"], translation.get("covariance", np.zeros((3, 3))),
                                               translation.get("coupling", np.zeros((3, 9))))))
        records.append(TransformDistributionStamped(edge["parent"], edge["child"], float(edge.get("stamp", 0.)),
            edge.get("edge_id", edge["parent"] + "__" + edge["child"]), "configured_initial_law",
            TransformDistribution(tuple(components))))
    sampling = config.get("sampling", {})
    return IndependentTransformLaw(tuple(records), config.get("independence"),
        sample_count=sampling.get("count", 64), seed=sampling.get("seed", 21), **common)
