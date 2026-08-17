"""Dependency-aware local Gaussian transform-moment evaluation."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from probtf.dependency.binding import POSE_PERTURBATION_CONVENTION
from probtf.dependency.gaussian import immutable_psd_matrix
from probtf.dependency.store import GaussianLatentSnapshot, GaussianLatentStore
from probtf.geometry import (
    DeterministicTransform,
    compose_transforms,
    quat_mul,
    quat_to_rotmat,
    relative_transform,
    rotation_vector_to_quaternion,
    se3_log,
    skew,
)
from probtf.kernels.composed import ComposedTransformKernel, IdentityTransformKernel
from probtf.kernels.forward import ForwardEdgeKernel
from probtf.kernels.inverse import InverseEdgeKernel
from probtf.kernels.mixture import MixtureTransformKernel
from probtf.probability import PointMomentSummary
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)
from probtf.temporal.backends import (
    component_pose_covariance,
    component_representative,
    component_from_pose_covariance,
)


class DependencyMomentError(RuntimeError):
    def __init__(self, code, message, repeated_dependency_ids=()):
        super().__init__(str(message))
        self.code = str(code)
        self.repeated_dependency_ids = tuple(repeated_dependency_ids)


@dataclass(frozen=True)
class TransformMomentSummary:
    mean: DeterministicTransform
    covariance: np.ndarray
    edge_ids: Tuple[str, ...] = ()
    factor_versions: Tuple[Tuple[str, int], ...] = ()
    perturbation_convention: str = POSE_PERTURBATION_CONVENTION
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)
    provenance: TransformProvenance = field(default_factory=TransformProvenance)
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.mean, DeterministicTransform):
            raise TypeError("mean must be DeterministicTransform.")
        covariance = immutable_psd_matrix(
            self.covariance,
            6,
            "covariance",
        )
        if self.perturbation_convention != POSE_PERTURBATION_CONVENTION:
            raise ValueError("Unsupported transform perturbation convention.")
        if not isinstance(self.approximation, ApproximationInfo):
            raise TypeError("approximation must be ApproximationInfo.")
        if not isinstance(self.provenance, TransformProvenance):
            raise TypeError("provenance must be TransformProvenance.")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "edge_ids", tuple(self.edge_ids))
        object.__setattr__(self, "factor_versions", tuple(self.factor_versions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_component(
        self,
        component_id,
        *,
        raw_weight=1.0,
        provenance=None,
    ):
        selected_provenance = (
            ComponentProvenance(
                source_ids=self.provenance.source_ids,
                derived_from_edge_ids=self.provenance.derived_from_edge_ids,
                method=self.provenance.method,
                detail=self.provenance.detail,
            )
            if provenance is None
            else provenance
        )
        return component_from_pose_covariance(
            component_id=component_id,
            raw_weight=raw_weight,
            transform=self.mean,
            covariance=self.covariance,
            provenance=selected_provenance,
            approximation=self.approximation,
        )

    def apply_to_point(self, mean, covariance=None):
        point = np.asarray(mean, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point mean must be a finite 3-vector.")
        point_covariance = (
            np.zeros((3, 3), dtype=float)
            if covariance is None
            else np.asarray(covariance, dtype=float)
        )
        if (
            point_covariance.shape != (3, 3)
            or not np.all(np.isfinite(point_covariance))
        ):
            raise ValueError("point covariance must be a finite 3x3 matrix.")
        rotation = quat_to_rotmat(self.mean.rotation_wxyz)
        jacobian = np.hstack(
            (
                np.eye(3, dtype=float),
                -rotation @ skew(point),
            )
        )
        output_covariance = (
            jacobian @ self.covariance @ jacobian.T
            + rotation @ point_covariance @ rotation.T
        )
        return PointMomentSummary(
            rotation @ point + self.mean.translation,
            0.5 * (output_covariance + output_covariance.T),
        )


def _unique(values):
    output = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return tuple(output)


def _edge_kernel(expression):
    return (
        expression.edge_kernel
        if isinstance(expression, MixtureTransformKernel)
        else expression
    )


def _kernel_sequence(kernel):
    if isinstance(kernel, ComposedTransformKernel):
        return kernel.kernels
    if isinstance(kernel, IdentityTransformKernel):
        return ()
    if isinstance(kernel, (ForwardEdgeKernel, InverseEdgeKernel, MixtureTransformKernel)):
        return (kernel,)
    raise TypeError("Unsupported transform kernel expression.")


def _record_local_moments(record):
    normalized = record.distribution.normalize_weights()
    if len(normalized.components) != 1:
        raise DependencyMomentError(
            "LOCAL_GAUSSIAN_MIXTURE_UNAVAILABLE",
            "Transform moments require one concentrated local component per edge.",
        )
    component = normalized.components[0].component
    try:
        covariance = component_pose_covariance(component)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise DependencyMomentError(
            "LOCAL_GAUSSIAN_UNAVAILABLE",
            "Edge {!r} has no finite local Gaussian chart: {}".format(
                record.edge_id,
                error,
            ),
        ) from error
    return component_representative(component), covariance


def apply_mixed_pose_perturbation(transform, perturbation):
    value = np.asarray(perturbation, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("perturbation must be a finite 6-vector.")
    return DeterministicTransform(
        transform.translation + value[:3],
        quat_mul(
            transform.rotation_wxyz,
            rotation_vector_to_quaternion(value[3:]),
        ),
    )


def inverse_mixed_pose_jacobian(transform):
    """Map a physical forward-edge perturbation into its inverse view."""

    if not isinstance(transform, DeterministicTransform):
        raise TypeError("transform must be DeterministicTransform.")
    rotation = quat_to_rotmat(transform.rotation_wxyz)
    body_translation = rotation.T @ transform.translation
    jacobian = np.zeros((6, 6), dtype=float)
    jacobian[:3, :3] = -rotation.T
    jacobian[:3, 3:] = -skew(body_translation)
    jacobian[3:, 3:] = -rotation
    return jacobian


def _pose_close(left, right, tolerance=1.0e-8):
    if np.linalg.norm(left.translation - right.translation, ord=np.inf) > tolerance:
        return False
    rotation_difference = se3_log(relative_transform(left, right))[3:]
    return np.linalg.norm(rotation_difference, ord=np.inf) <= tolerance


def _path_composition_jacobians(directed_transforms):
    count = len(directed_transforms)
    prefixes = [DeterministicTransform.identity()]
    for transform in directed_transforms:
        prefixes.append(compose_transforms(transform, prefixes[-1]))
    suffixes = [None] * (count + 1)
    suffixes[count] = DeterministicTransform.identity()
    for index in range(count - 1, -1, -1):
        suffixes[index] = compose_transforms(
            suffixes[index + 1],
            directed_transforms[index],
        )

    jacobians = []
    for index, transform in enumerate(directed_transforms):
        before = prefixes[index]
        after = suffixes[index + 1]
        rotation_before = quat_to_rotmat(before.rotation_wxyz)
        rotation_after = quat_to_rotmat(after.rotation_wxyz)
        rotation_edge = quat_to_rotmat(transform.rotation_wxyz)
        jacobian = np.zeros((6, 6), dtype=float)
        jacobian[:3, :3] = rotation_after
        jacobian[:3, 3:] = (
            -rotation_after @ rotation_edge @ skew(before.translation)
        )
        jacobian[3:, 3:] = rotation_before.T
        jacobians.append(jacobian)
    return prefixes[-1], tuple(jacobians)


class DependencyAwareMomentEvaluator:
    """Matrix-only path-local aggregation of residual and latent covariance."""

    def __init__(self):
        self._cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _snapshot(value):
        if value is None:
            return GaussianLatentSnapshot(0, {}, {}, {})
        if isinstance(value, GaussianLatentStore):
            return value.snapshot()
        if isinstance(value, GaussianLatentSnapshot):
            return value
        raise TypeError("latent_state must be GaussianLatentStore or snapshot.")

    @staticmethod
    def _cache_key(kernel, snapshot):
        signature = []
        for expression in _kernel_sequence(kernel):
            base = _edge_kernel(expression)
            signature.append(
                (
                    id(base.edge_record),
                    base.edge_record.edge_id,
                    base.edge_record.stamp,
                    isinstance(base, InverseEdgeKernel),
                )
            )
        return tuple(signature), snapshot.revision

    def evaluate(self, kernel, latent_state=None):
        snapshot = self._snapshot(latent_state)
        key = self._cache_key(kernel, snapshot)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        result = self._evaluate_uncached(kernel, snapshot)
        self._cache[key] = result
        return result

    def _evaluate_uncached(self, kernel, snapshot):
        sequence = _kernel_sequence(kernel)
        if not sequence:
            return TransformMomentSummary(
                DeterministicTransform.identity(),
                np.zeros((6, 6), dtype=float),
            )

        repeated = (
            kernel.repeated_dependency_ids()
            if isinstance(kernel, ComposedTransformKernel)
            else ()
        )
        edge_data = []
        used_factor_ids = set()
        for expression in sequence:
            base = _edge_kernel(expression)
            if not isinstance(base, (ForwardEdgeKernel, InverseEdgeKernel)):
                raise DependencyMomentError(
                    "UNSUPPORTED_KERNEL_EXPRESSION",
                    "Dependency-aware moments support forward/inverse edge kernels.",
                )
            record = base.edge_record
            physical_mean, residual_covariance = _record_local_moments(record)
            bindings = snapshot.bindings_for_edge(record.edge_id)
            if bindings:
                reference = bindings[0].linearization_pose
                if not _pose_close(reference, physical_mean):
                    raise DependencyMomentError(
                        "LINEARIZATION_MISMATCH",
                        "Binding linearization pose does not match edge {!r}.".format(
                            record.edge_id
                        ),
                        repeated,
                    )
                if any(
                    binding.linearization_stamp != record.stamp
                    or not _pose_close(binding.linearization_pose, reference)
                    for binding in bindings
                ):
                    raise DependencyMomentError(
                        "LINEARIZATION_MISMATCH",
                        "Bindings for edge {!r} do not share its linearization metadata.".format(
                            record.edge_id
                        ),
                        repeated,
                    )
                perturbation = np.zeros(6, dtype=float)
                for binding in bindings:
                    factor = snapshot.factors.get(binding.factor_id)
                    if factor is None or factor.version != binding.factor_version:
                        raise DependencyMomentError(
                            "DEPENDENCY_UNRESOLVED",
                            "Binding for edge {!r} has no matching factor version.".format(
                                record.edge_id
                            ),
                            repeated,
                        )
                    perturbation += binding.sensitivity @ factor.mean
                    used_factor_ids.add(binding.factor_id)
                physical_mean = apply_mixed_pose_perturbation(
                    reference,
                    perturbation,
                )

            directed_map = np.eye(6, dtype=float)
            directed_mean = physical_mean
            if isinstance(base, InverseEdgeKernel):
                directed_map = inverse_mixed_pose_jacobian(physical_mean)
                directed_mean = physical_mean.inverse()
            edge_data.append(
                (
                    base,
                    directed_mean,
                    directed_map,
                    residual_covariance,
                    bindings,
                )
            )

        for dependency_id in repeated:
            if dependency_id not in snapshot.factors:
                raise DependencyMomentError(
                    "DEPENDENCY_UNRESOLVED",
                    "Repeated dependency {!r} has no Gaussian factor.".format(
                        dependency_id
                    ),
                    repeated,
                )
            for base, _, _, _, bindings in edge_data:
                if dependency_id not in base.latent_dependency_ids():
                    continue
                if not any(
                    binding.factor_id == dependency_id for binding in bindings
                ):
                    raise DependencyMomentError(
                        "DEPENDENCY_UNRESOLVED",
                        "Edge {!r} lacks a binding for repeated dependency {!r}.".format(
                            base.edge_record.edge_id,
                            dependency_id,
                        ),
                        repeated,
                    )

        directed_transforms = tuple(item[1] for item in edge_data)
        output_mean, path_jacobians = _path_composition_jacobians(
            directed_transforms
        )

        residual_aggregates = {}
        latent_aggregates = {}
        for (
            base,
            _,
            directed_map,
            residual_covariance,
            bindings,
        ), path_jacobian in zip(edge_data, path_jacobians):
            edge_id = base.edge_record.edge_id
            mapped = path_jacobian @ directed_map
            if edge_id in residual_aggregates:
                previous_covariance, aggregate = residual_aggregates[edge_id]
                if not np.allclose(
                    previous_covariance,
                    residual_covariance,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise DependencyMomentError(
                        "DEPENDENCY_UNRESOLVED",
                        "Repeated edge {!r} changed residual covariance.".format(edge_id),
                        repeated,
                    )
                residual_aggregates[edge_id] = (
                    previous_covariance,
                    aggregate + mapped,
                )
            else:
                residual_aggregates[edge_id] = (
                    residual_covariance,
                    mapped,
                )
            for binding in bindings:
                contribution = mapped @ binding.sensitivity
                latent_aggregates[binding.factor_id] = (
                    latent_aggregates.get(
                        binding.factor_id,
                        np.zeros_like(contribution),
                    )
                    + contribution
                )

        output_covariance = np.zeros((6, 6), dtype=float)
        for residual_covariance, aggregate in residual_aggregates.values():
            output_covariance += (
                aggregate @ residual_covariance @ aggregate.T
            )

        factor_order = tuple(sorted(latent_aggregates))
        if factor_order:
            _, joint_covariance, slices = snapshot.joint_mean_covariance(
                factor_order
            )
            eigenvalues = np.linalg.eigvalsh(joint_covariance)
            scale = max(1.0, float(np.linalg.norm(joint_covariance, ord=np.inf)))
            if float(eigenvalues[0]) < -1.0e-10 * scale:
                raise DependencyMomentError(
                    "INVALID_LATENT_COVARIANCE",
                    "Path-local latent covariance is not positive semidefinite.",
                    repeated,
                )
            aggregate = np.zeros((6, joint_covariance.shape[0]), dtype=float)
            for factor_id in factor_order:
                aggregate[:, slices[factor_id]] = latent_aggregates[factor_id]
            output_covariance += aggregate @ joint_covariance @ aggregate.T

        output_covariance = 0.5 * (
            output_covariance + output_covariance.T
        )
        stochastic = np.max(np.abs(output_covariance)) > 1.0e-15
        approximation = (
            ApproximationInfo(
                kind=ApproximationKind.MOMENT_SUMMARY,
                lossy=True,
                detail=(
                    "Local Gaussian transform moments in the existing mixed "
                    "translation/right-rotation chart."
                ),
                source="probtf.dependency.DependencyAwareMomentEvaluator",
            )
            if stochastic
            else ApproximationInfo()
        )
        records = tuple(_edge_kernel(item).edge_record for item in sequence)
        provenance = TransformProvenance(
            source_ids=_unique(
                value
                for record in records
                for value in record.provenance.source_ids
            ),
            derived_from_edge_ids=_unique(record.edge_id for record in records),
            method="dependency_aware_local_gaussian_moments",
            detail=POSE_PERTURBATION_CONVENTION,
        )
        diagnostics = []
        if repeated:
            diagnostics.append(
                "resolved repeated dependencies: {}".format(
                    ", ".join(repeated)
                )
            )
        return TransformMomentSummary(
            output_mean,
            output_covariance,
            tuple(record.edge_id for record in records),
            snapshot.factor_versions(factor_order),
            POSE_PERTURBATION_CONVENTION,
            approximation,
            provenance,
            tuple(diagnostics),
        )


__all__ = [
    "DependencyAwareMomentEvaluator",
    "DependencyMomentError",
    "TransformMomentSummary",
    "apply_mixed_pose_perturbation",
    "inverse_mixed_pose_jacobian",
]
