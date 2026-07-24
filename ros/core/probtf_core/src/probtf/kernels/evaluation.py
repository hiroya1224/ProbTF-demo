"""Explicit evaluators for lazy transform-kernel expressions."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from probtf.distributions import DistributionStatus
from probtf.kernels.base import (
    AppliedKernelExpression,
    DiracPointLaw,
    GaussianPointLaw,
    KernelDiagnosticCode,
    KernelDiagnostics,
    KernelEvaluationOptions,
    KernelRepresentation,
    KernelResult,
    PointLaw,
    TransformKernelExpression,
)
from probtf.kernels.composed import ComposedTransformKernel, IdentityTransformKernel
from probtf.kernels.forward import ForwardEdgeKernel
from probtf.kernels.inverse import InverseEdgeKernel
from probtf.kernels.mixture import MixtureTransformKernel
from probtf.probability import (
    PointMomentSummary,
    apply_transform_samples,
    forward_component_point_moments,
    mixture_point_moments,
    sample_transform_distribution,
    sample_transform_distribution_components,
)
from probtf.provenance import ApproximationInfo, ApproximationKind
from probtf.temporal.provenance import parse_temporal_detail
from probtf.spherical_law import (
    InducedVectorLaw,
    IslBackendUnavailableError,
    IslEvaluationOptions,
    UnavailableExactIslBackend,
)


@dataclass(frozen=True)
class UnavailableKernelValue:
    code: str
    reason: str


@dataclass(frozen=True)
class UncoupledPointActionLaw:
    """Convolution of an induced rotation law and independent Gaussian translation."""

    induced_vector_law: InducedVectorLaw
    translation_mean: np.ndarray
    translation_covariance: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.translation_mean, dtype=float)
        covariance = np.asarray(self.translation_covariance, dtype=float)
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise ValueError("translation_mean must be a finite 3-vector.")
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("translation_covariance must be a finite 3x3 matrix.")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
            raise ValueError("translation_covariance must be symmetric.")
        mean = mean.copy()
        covariance = covariance.copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "translation_mean", mean)
        object.__setattr__(self, "translation_covariance", covariance)

    @property
    def approximation(self):
        return self.induced_vector_law.approximation


@dataclass(frozen=True)
class WeightedPointActionLaw:
    weight: float
    law: object


@dataclass(frozen=True)
class MixturePointActionLaw:
    components: Tuple[WeightedPointActionLaw, ...]

    @property
    def approximation(self):
        approximations = tuple(
            getattr(component.law, "approximation", ApproximationInfo())
            for component in self.components
        )
        lossy = tuple(item for item in approximations if item.lossy)
        if not lossy:
            return ApproximationInfo()
        return ApproximationInfo(
            kind=lossy[0].kind,
            lossy=True,
            detail="At least one mixture component uses an approximate induced law.",
        )


def _edge_kernel(kernel):
    return kernel.edge_kernel if isinstance(kernel, MixtureTransformKernel) else kernel


def _kernel_sequence(kernel):
    if isinstance(kernel, ComposedTransformKernel):
        return kernel.kernels
    if isinstance(kernel, IdentityTransformKernel):
        return ()
    if isinstance(kernel, TransformKernelExpression):
        return (kernel,)
    raise TypeError("kernel must be a TransformKernelExpression.")


def _record_for_kernel(kernel):
    base = _edge_kernel(kernel)
    return base.edge_record if isinstance(base, (ForwardEdgeKernel, InverseEdgeKernel)) else None


def _repeated_dependencies(kernel):
    if isinstance(kernel, ComposedTransformKernel):
        return kernel.repeated_dependency_ids()
    return ()


def _temporal_sample_key(record):
    payload = parse_temporal_detail(record.provenance.detail)
    if payload is None or payload.get("backend") != "sample":
        return None
    component_ids = tuple(
        component.component_id for component in record.distribution.components
    )
    expected = tuple(
        "sample:{:06d}".format(index) for index in range(len(component_ids))
    )
    if component_ids != expected:
        return None
    return (
        payload.get("random_seed"),
        payload.get("random_stream", ""),
        component_ids,
    )


def _has_dependency_aware_temporal_samples(kernel):
    stochastic_records = []
    for expression in _kernel_sequence(kernel):
        record = _record_for_kernel(expression)
        if record is None or record.distribution.deterministic_transform() is not None:
            continue
        stochastic_records.append(record)
    return bool(stochastic_records) and all(
        _temporal_sample_key(record) is not None for record in stochastic_records
    )


def _input_moments(input_law):
    if isinstance(input_law, DiracPointLaw):
        return PointMomentSummary(input_law.point, np.zeros((3, 3)))
    if isinstance(input_law, GaussianPointLaw):
        return PointMomentSummary(input_law.mean, input_law.covariance)
    raise TypeError("Unsupported PointLaw type.")


def _transform_moments(summary, transform):
    from probtf.geometry import quat_to_rotmat

    rotation = quat_to_rotmat(transform.rotation_wxyz)
    return PointMomentSummary(
        rotation @ summary.mean + transform.translation,
        rotation @ summary.covariance @ rotation.T,
    )


class KernelEvaluator:
    def __init__(self, isl_backend=None, bingham_integration_steps=120):
        self.isl_backend = UnavailableExactIslBackend() if isl_backend is None else isl_backend
        steps = int(bingham_integration_steps)
        if steps < 1 or steps != bingham_integration_steps:
            raise ValueError("bingham_integration_steps must be a positive integer.")
        self.bingham_integration_steps = steps

    def apply_to_point(
        self,
        kernel,
        point,
        representation=KernelRepresentation.EXPRESSION,
        options=None,
    ):
        if options is None:
            options = KernelEvaluationOptions(representation=representation)
        elif options.representation is not representation:
            raise ValueError("options.representation must match representation.")
        return self.apply(kernel, DiracPointLaw(np.asarray(point, dtype=float)), options)

    def _distribution_status(self, kernel):
        diagnostics = []
        status = DistributionStatus.OK
        for expression in _kernel_sequence(kernel):
            record = _record_for_kernel(expression)
            if record is None:
                continue
            normalized = record.distribution.normalize_weights()
            if normalized.status is DistributionStatus.INVALID:
                return DistributionStatus.INVALID, KernelDiagnostics(
                    (KernelDiagnosticCode.INVALID_DISTRIBUTION,),
                    ("Edge '{}' has invalid component weights.".format(record.edge_id),),
                )
            if normalized.status is DistributionStatus.ZERO_MASS:
                return DistributionStatus.ZERO_MASS, KernelDiagnostics(
                    (KernelDiagnosticCode.ZERO_MASS,),
                    ("Edge '{}' has zero usable mixture mass.".format(record.edge_id),),
                )
            diagnostics.extend(
                "{}:{}".format(item.code, item.component_id) for item in normalized.diagnostics
            )
        return status, KernelDiagnostics(messages=tuple(diagnostics))

    def _deterministic_transforms(self, kernel):
        transforms = []
        for expression in _kernel_sequence(kernel):
            base = _edge_kernel(expression)
            if not isinstance(base, (ForwardEdgeKernel, InverseEdgeKernel)):
                return None
            transform = base.edge_record.distribution.deterministic_transform()
            if transform is None:
                return None
            transforms.append(transform if isinstance(base, ForwardEdgeKernel) else transform.inverse())
        return tuple(transforms)

    def _deterministic_law(self, kernel, input_law):
        transforms = self._deterministic_transforms(kernel)
        if transforms is None:
            return None
        summary = _input_moments(input_law)
        for transform in transforms:
            summary = _transform_moments(summary, transform)
        if isinstance(input_law, DiracPointLaw):
            return DiracPointLaw(summary.mean)
        return GaussianPointLaw(summary.mean, summary.covariance)

    def _moment_result(self, kernel, input_law, diagnostics):
        deterministic = self._deterministic_law(kernel, input_law)
        if deterministic is not None:
            value = _input_moments(deterministic)
            return KernelResult(
                DistributionStatus.OK,
                KernelRepresentation.MOMENTS,
                value,
                ApproximationInfo(),
                diagnostics,
            )

        summary = _input_moments(input_law)
        for expression in _kernel_sequence(kernel):
            base = _edge_kernel(expression)
            if isinstance(base, InverseEdgeKernel):
                return self._unavailable(
                    KernelRepresentation.MOMENTS,
                    "UNAVAILABLE_INVERSE_STOCHASTIC_MOMENTS",
                    "Stochastic inverse covariance requires higher-order joint rotation/coupling moments.",
                )
            if not isinstance(base, ForwardEdgeKernel):
                return self._unavailable(
                    KernelRepresentation.MOMENTS,
                    "UNSUPPORTED_KERNEL_EXPRESSION",
                    "Unsupported kernel expression for moment evaluation.",
                )
            normalized = base.edge_record.distribution.normalize_weights()
            weighted = tuple(
                (
                    item.weight,
                    forward_component_point_moments(
                        item.component,
                        summary,
                        self.bingham_integration_steps,
                    ),
                )
                for item in normalized.components
            )
            summary = mixture_point_moments(weighted)
        return KernelResult(
            DistributionStatus.OK,
            KernelRepresentation.MOMENTS,
            summary,
            ApproximationInfo(
                kind=ApproximationKind.MOMENT_SUMMARY,
                lossy=True,
                detail="First two point moments are evaluated without identifying them with the original law.",
            ),
            diagnostics,
        )

    def _numerical_law_result(self, kernel, input_law, options, diagnostics):
        deterministic = self._deterministic_law(kernel, input_law)
        if deterministic is not None:
            return KernelResult(
                DistributionStatus.OK,
                KernelRepresentation.NUMERICAL_LAW,
                deterministic,
                ApproximationInfo(),
                diagnostics,
            )
        sequence = _kernel_sequence(kernel)
        if len(sequence) != 1 or not isinstance(input_law, DiracPointLaw):
            return self._unavailable(
                KernelRepresentation.NUMERICAL_LAW,
                "UNAVAILABLE_COMPOSED_ISL_BACKEND",
                "Numerical stochastic composition currently supports one forward edge and a Dirac point.",
            )
        base = _edge_kernel(sequence[0])
        if not isinstance(base, ForwardEdgeKernel):
            return self._unavailable(
                KernelRepresentation.NUMERICAL_LAW,
                "UNAVAILABLE_INVERSE_ISL_BACKEND",
                "Numerical stochastic inverse action is not implemented.",
            )
        normalized = base.edge_record.distribution.normalize_weights()
        laws = []
        isl_options = options.isl_options or IslEvaluationOptions()
        try:
            for item in normalized.components:
                component = item.component
                if not np.allclose(component.translation.rotation_coupling, 0.0, rtol=0.0, atol=0.0):
                    return self._unavailable(
                        KernelRepresentation.NUMERICAL_LAW,
                        "UNAVAILABLE_COUPLED_ISL_INTEGRATOR",
                        "Coupled vec(R) translation requires a joint numerical point-action integrator.",
                    )
                induced = self.isl_backend.rotate_vector(
                    component.orientation,
                    input_law.point,
                    isl_options,
                )
                laws.append(
                    WeightedPointActionLaw(
                        item.weight,
                        UncoupledPointActionLaw(
                            induced,
                            component.translation.mean_at_reference,
                            component.translation.residual_covariance,
                        ),
                    )
                )
        except IslBackendUnavailableError as exc:
            return self._unavailable(
                KernelRepresentation.NUMERICAL_LAW,
                exc.code,
                str(exc),
            )
        value = laws[0].law if len(laws) == 1 else MixturePointActionLaw(tuple(laws))
        approximation = getattr(value, "approximation", ApproximationInfo())
        return KernelResult(
            DistributionStatus.OK,
            KernelRepresentation.NUMERICAL_LAW,
            value,
            approximation,
            diagnostics,
        )

    def _sample_result(self, kernel, input_law, options, diagnostics):
        count = options.sample_count
        generator = np.random.default_rng(options.rng)
        if isinstance(input_law, DiracPointLaw):
            points = np.repeat(input_law.point[None, :], count, axis=0)
        else:
            points = generator.multivariate_normal(
                input_law.mean,
                input_law.covariance,
                size=count,
            ).reshape(count, 3)

        aligned_component_choices = {}
        for expression in _kernel_sequence(kernel):
            base = _edge_kernel(expression)
            if not isinstance(base, (ForwardEdgeKernel, InverseEdgeKernel)):
                return self._unavailable(
                    KernelRepresentation.SAMPLES,
                    "UNSUPPORTED_KERNEL_EXPRESSION",
                    "Sampling supports only forward, inverse, and composed edge kernels.",
                )
            sample_key = _temporal_sample_key(base.edge_record)
            if sample_key is None:
                transform_samples = sample_transform_distribution(
                    base.edge_record.distribution,
                    count,
                    generator,
                )
            else:
                choices = aligned_component_choices.get(sample_key)
                if choices is None:
                    normalized = base.edge_record.distribution.normalize_weights()
                    weights = np.array(
                        [item.weight for item in normalized.components],
                        dtype=float,
                    )
                    choices = generator.choice(
                        len(normalized.components),
                        size=count,
                        p=weights,
                    )
                    aligned_component_choices[sample_key] = choices
                transform_samples = sample_transform_distribution_components(
                    base.edge_record.distribution,
                    choices,
                    generator,
                )
            points = apply_transform_samples(
                transform_samples,
                points,
                inverse=isinstance(base, InverseEdgeKernel),
            )

        points.setflags(write=False)
        deterministic = self._deterministic_law(kernel, input_law)
        approximation = (
            ApproximationInfo()
            if isinstance(deterministic, DiracPointLaw)
            else ApproximationInfo(
                kind=ApproximationKind.MONTE_CARLO,
                lossy=True,
                detail="Finite samples from the native transform-kernel law.",
                source="probtf.kernels.KernelEvaluator",
            )
        )
        return KernelResult(
            DistributionStatus.OK,
            KernelRepresentation.SAMPLES,
            points,
            approximation,
            diagnostics,
        )

    @staticmethod
    def _unavailable(representation, code, reason):
        return KernelResult(
            DistributionStatus.INVALID,
            representation,
            UnavailableKernelValue(code, reason),
            ApproximationInfo(
                kind=ApproximationKind.UNAVAILABLE,
                lossy=False,
                detail=reason,
            ),
            KernelDiagnostics((KernelDiagnosticCode.UNAVAILABLE_BACKEND,), (reason,)),
        )

    def apply(self, kernel, input_law, options):
        if not isinstance(kernel, TransformKernelExpression):
            raise TypeError("kernel must be a TransformKernelExpression.")
        if not isinstance(input_law, PointLaw):
            raise TypeError("input_law must be a PointLaw.")
        if not isinstance(options, KernelEvaluationOptions):
            raise TypeError("options must be KernelEvaluationOptions.")

        status, diagnostics = self._distribution_status(kernel)
        if status is not DistributionStatus.OK:
            return KernelResult(
                status,
                options.representation,
                None,
                ApproximationInfo(),
                diagnostics,
            )
        repeated = _repeated_dependencies(kernel)
        if (
            repeated
            and self._deterministic_transforms(kernel) is None
            and not (
                options.representation is KernelRepresentation.SAMPLES
                and _has_dependency_aware_temporal_samples(kernel)
            )
        ):
            return KernelResult(
                DistributionStatus.INVALID,
                options.representation,
                UnavailableKernelValue(
                    "DEPENDENCY_UNRESOLVED",
                    "Repeated latent edge realizations require a dependency-aware evaluator.",
                ),
                ApproximationInfo(
                    kind=ApproximationKind.UNAVAILABLE,
                    detail="Repeated latent dependency is unresolved.",
                ),
                KernelDiagnostics(
                    (KernelDiagnosticCode.DEPENDENCY_UNRESOLVED,),
                    ("Repeated latent dependency is unresolved.",),
                    repeated,
                ),
            )
        if options.representation is KernelRepresentation.EXPRESSION:
            return KernelResult(
                DistributionStatus.OK,
                options.representation,
                AppliedKernelExpression(kernel, input_law),
                ApproximationInfo(),
                diagnostics,
            )
        if options.representation is KernelRepresentation.MOMENTS:
            return self._moment_result(kernel, input_law, diagnostics)
        if options.representation is KernelRepresentation.NUMERICAL_LAW:
            return self._numerical_law_result(kernel, input_law, options, diagnostics)
        if options.representation is KernelRepresentation.SAMPLES:
            return self._sample_result(kernel, input_law, options, diagnostics)
        return self._unavailable(
            options.representation,
            "UNAVAILABLE_CLOSED_MIXTURE_BACKEND",
            "Closed-mixture projection requires an explicit reduction policy.",
        )
