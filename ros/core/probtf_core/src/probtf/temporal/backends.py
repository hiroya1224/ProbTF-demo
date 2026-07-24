"""Moment and common-random-number sample backends for temporal models."""

from dataclasses import replace
import hashlib

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    OrientationKind,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    body_twist_between,
    compose_transforms,
    interpolate_transform,
    quat_left_matrix,
    quat_to_rotmat,
    right_perturbation_vec_rotation_jacobian,
    se3_exp,
    skew,
)
from probtf.probability import sample_transform_distribution
from probtf.provenance import ApproximationInfo, ApproximationKind
from probtf.temporal.provenance import (
    make_component_provenance,
    make_transform_provenance,
    source_record_dependency_id,
)


def _nearest_psd(matrix, tolerance=1.0e-12):
    value = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if float(eigenvalues[0]) < -1.0e-8:
        raise ValueError("Covariance is not positive semidefinite.")
    clipped = np.maximum(eigenvalues, tolerance)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def record_representative(record):
    """Return a deterministic central pose without claiming it is a global MAP."""

    deterministic = record.distribution.deterministic_transform()
    if deterministic is not None:
        return deterministic
    if record.representative is not None:
        return record.representative
    normalized = record.distribution.normalize_weights()
    if not normalized.components:
        raise ValueError("Temporal evaluation requires a usable transform distribution.")
    component = max(normalized.components, key=lambda item: item.weight).component
    quaternion = component.orientation.mode_wxyz
    return DeterministicTransform(
        component.conditional_translation_mean(quaternion),
        quaternion,
    )


def component_representative(component):
    quaternion = component.orientation.mode_wxyz
    return DeterministicTransform(
        component.conditional_translation_mean(quaternion),
        quaternion,
    )


def orientation_tangent_covariance(orientation):
    """Laplace covariance in right-local rotation-vector coordinates."""

    if orientation.kind is OrientationKind.DIRAC:
        return np.zeros((3, 3), dtype=float)
    if orientation.kind is OrientationKind.UNIFORM:
        raise ValueError("A uniform orientation has no finite tangent covariance.")
    parameter = orientation.backend_parameter_matrix()
    mode = orientation.mode_wxyz
    basis = quat_left_matrix(mode)[:, 1:]
    precision = -0.5 * (basis.T @ parameter @ basis)
    precision = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("Bingham orientation has no positive local precision.")
    return eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T


def component_pose_covariance(component):
    """Return mixed ``[translation_parent, rotation_body]`` covariance."""

    rotation_covariance = orientation_tangent_covariance(component.orientation)
    jacobian = (
        component.translation.rotation_coupling
        @ right_perturbation_vec_rotation_jacobian(
            component.orientation.reference_quaternion_wxyz
        )
    )
    cross_covariance = jacobian @ rotation_covariance
    translation_covariance = (
        component.translation.residual_covariance
        + jacobian @ rotation_covariance @ jacobian.T
    )
    covariance = np.block(
        [
            [translation_covariance, cross_covariance],
            [cross_covariance.T, rotation_covariance],
        ]
    )
    return 0.5 * (covariance + covariance.T)


def _orientation_from_tangent_covariance(quaternion, covariance):
    value = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    if np.max(np.abs(value)) <= 1.0e-14:
        return BinghamOrientation.dirac(quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if float(eigenvalues[0]) < -1.0e-9:
        raise ValueError("orientation covariance must be positive semidefinite.")
    inverse = eigenvectors @ np.diag(1.0 / np.maximum(eigenvalues, 1.0e-10)) @ eigenvectors.T
    tangent_basis = quat_left_matrix(quaternion)[:, 1:]
    parameter = tangent_basis @ (-2.0 * inverse) @ tangent_basis.T
    return BinghamOrientation.from_parameter_matrix(
        parameter,
        reference_quaternion_wxyz=quaternion,
    )


def component_from_pose_covariance(
    *,
    component_id,
    raw_weight,
    transform,
    covariance,
    provenance,
    approximation,
):
    """Encode a local pose moment in the native conditional v2 component."""

    value = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    if value.shape != (6, 6) or not np.all(np.isfinite(value)):
        raise ValueError("covariance must be a finite 6x6 matrix.")
    rotation_covariance = value[3:, 3:]
    if np.max(np.abs(rotation_covariance)) <= 1.0e-14:
        orientation = BinghamOrientation.dirac(transform.rotation_wxyz)
        coupling = np.zeros((3, 9), dtype=float)
        residual = value[:3, :3]
    else:
        orientation = _orientation_from_tangent_covariance(
            transform.rotation_wxyz,
            rotation_covariance,
        )
        linear = value[:3, 3:] @ np.linalg.pinv(rotation_covariance, rcond=1.0e-10)
        rotation_jacobian = right_perturbation_vec_rotation_jacobian(
            transform.rotation_wxyz
        )
        coupling = 0.5 * linear @ rotation_jacobian.T
        residual = value[:3, :3] - linear @ rotation_covariance @ linear.T
    residual = _nearest_psd(residual, tolerance=0.0)
    return TransformComponent(
        component_id=component_id,
        raw_weight=raw_weight,
        orientation=orientation,
        translation=ConditionalGaussianTranslation(
            transform.translation,
            residual,
            coupling,
        ),
        provenance=provenance,
        approximation=approximation,
    )


def _output_record(
    anchor,
    stamp,
    components,
    representative,
    source_records,
    detail,
    approximation,
):
    return TransformDistributionStamped(
        parent_frame_id=anchor.parent_frame_id,
        child_frame_id=anchor.child_frame_id,
        stamp=stamp,
        edge_id=anchor.edge_id,
        authority=anchor.authority,
        distribution=TransformDistribution(tuple(components)),
        representative=representative,
        representative_kind=RepresentativeKind.MOMENT_REPRESENTATIVE,
        provenance=make_transform_provenance(source_records, detail),
        is_static=False,
        approximation=approximation,
    )


def copy_record_at_stamp(record, stamp):
    """Copy a time-invariant/sample result without altering its distribution."""

    return replace(record, stamp=float(stamp))


def record_uncertainty_trace(record):
    normalized = record.distribution.normalize_weights()
    if not normalized.components:
        return float("inf")
    center = record_representative(record)
    total = 0.0
    for weighted in normalized.components:
        component = weighted.component
        covariance = component_pose_covariance(component)
        pose = component_representative(component)
        translation_offset = pose.translation - center.translation
        # This diagnostic is only a scalar safety summary; the actual
        # distribution remains in its native representation.
        from probtf.geometry import relative_transform, se3_log

        rotation_offset = se3_log(relative_transform(center, pose))[3:]
        offset = np.concatenate((translation_offset, rotation_offset))
        total += weighted.weight * (float(np.trace(covariance)) + float(offset @ offset))
    return max(0.0, total)


def moment_interpolate(
    left,
    right,
    requested_stamp,
    detail,
):
    duration = right.stamp - left.stamp
    alpha = (requested_stamp - left.stamp) / duration
    approximation = ApproximationInfo(
        kind=ApproximationKind.TANGENT_SURROGATE,
        lossy=True,
        detail=(
            "Endpoint distributions are interpolated component-wise in SE(3); "
            "unavailable cross-time covariance is diagnosed by the caller."
        ),
        source="probtf.temporal.moment",
    )
    components = []
    component_provenance = make_component_provenance((left, right), detail)
    for left_component in left.distribution.components:
        for right_component in right.distribution.components:
            transform = interpolate_transform(
                component_representative(left_component),
                component_representative(right_component),
                alpha,
            )
            covariance = (
                (1.0 - alpha) ** 2 * component_pose_covariance(left_component)
                + alpha ** 2 * component_pose_covariance(right_component)
            )
            components.append(
                component_from_pose_covariance(
                    component_id="{}~{}".format(
                        left_component.component_id,
                        right_component.component_id,
                    ),
                    raw_weight=left_component.raw_weight * right_component.raw_weight,
                    transform=transform,
                    covariance=covariance,
                    provenance=component_provenance,
                    approximation=approximation,
                )
            )
    representative = interpolate_transform(
        record_representative(left),
        record_representative(right),
        alpha,
    )
    return _output_record(
        left,
        requested_stamp,
        components,
        representative,
        (left, right),
        detail,
        approximation,
    )


def moment_predict(
    history,
    requested_stamp,
    mean_increment,
    process_covariance,
    detail,
):
    anchor = history[-1]
    increment = se3_exp(mean_increment)
    approximation = ApproximationInfo(
        kind=ApproximationKind.TANGENT_SURROGATE,
        lossy=True,
        detail=(
            "SE(3) body increment with first-order local-moment propagation "
            "and continuous-time process noise."
        ),
        source="probtf.temporal.moment",
    )
    components = []
    component_provenance = make_component_provenance(history, detail)
    increment_rotation = quat_to_rotmat(increment.rotation_wxyz)
    for component in anchor.distribution.components:
        old_transform = component_representative(component)
        new_transform = compose_transforms(old_transform, increment)
        old_rotation = quat_to_rotmat(old_transform.rotation_wxyz)
        jacobian = np.zeros((6, 6), dtype=float)
        jacobian[:3, :3] = np.eye(3)
        jacobian[:3, 3:] = -old_rotation @ skew(increment.translation)
        jacobian[3:, 3:] = increment_rotation.T
        propagated = jacobian @ component_pose_covariance(component) @ jacobian.T
        process_map = np.zeros((6, 6), dtype=float)
        process_map[:3, :3] = quat_to_rotmat(new_transform.rotation_wxyz)
        process_map[3:, 3:] = np.eye(3)
        propagated += process_map @ process_covariance @ process_map.T
        components.append(
            component_from_pose_covariance(
                component_id="{}:predicted".format(component.component_id),
                raw_weight=component.raw_weight,
                transform=new_transform,
                covariance=propagated,
                provenance=component_provenance,
                approximation=approximation,
            )
        )
    representative = compose_transforms(record_representative(anchor), increment)
    return _output_record(
        anchor,
        requested_stamp,
        components,
        representative,
        history,
        detail,
        approximation,
    )


def _derived_seed(seed, stream, dependency_id):
    digest = hashlib.sha256()
    digest.update(str(0 if seed is None else int(seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(stream).encode("utf-8"))
    digest.update(b"\0")
    digest.update(dependency_id.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _record_samples(record, count, seed, stream):
    dependency = source_record_dependency_id(record)
    generator = np.random.default_rng(_derived_seed(seed, stream, dependency))
    return sample_transform_distribution(record.distribution, count, generator)


def _sample_components(transforms, source_records, detail):
    approximation = ApproximationInfo(
        kind=ApproximationKind.MONTE_CARLO,
        lossy=True,
        detail="Common-random-number SE(3) temporal samples.",
        source="probtf.temporal.sample",
    )
    provenance = make_component_provenance(source_records, detail)
    return tuple(
        TransformComponent(
            component_id="sample:{:06d}".format(index),
            raw_weight=1.0,
            orientation=BinghamOrientation.dirac(transform.rotation_wxyz),
            translation=ConditionalGaussianTranslation(
                transform.translation,
                np.zeros((3, 3)),
                np.zeros((3, 9)),
            ),
            provenance=provenance,
            approximation=approximation,
        )
        for index, transform in enumerate(transforms)
    ), approximation


def _batch_transforms(batch):
    return tuple(
        DeterministicTransform(translation, quaternion)
        for translation, quaternion in zip(
            batch.translations,
            batch.rotations_wxyz,
        )
    )


def sample_interpolate(
    left,
    right,
    requested_stamp,
    sample_count,
    seed,
    stream,
    detail,
):
    left_samples = _batch_transforms(_record_samples(left, sample_count, seed, stream))
    right_samples = _batch_transforms(_record_samples(right, sample_count, seed, stream))
    alpha = (requested_stamp - left.stamp) / (right.stamp - left.stamp)
    transforms = tuple(
        interpolate_transform(left_transform, right_transform, alpha)
        for left_transform, right_transform in zip(left_samples, right_samples)
    )
    components, approximation = _sample_components(
        transforms,
        (left, right),
        detail,
    )
    representative = interpolate_transform(
        record_representative(left),
        record_representative(right),
        alpha,
    )
    return _output_record(
        left,
        requested_stamp,
        components,
        representative,
        (left, right),
        detail,
        approximation,
    )


def sample_predict(
    history,
    requested_stamp,
    acceleration,
    process_noise_spectral_density,
    sample_count,
    seed,
    stream,
    detail,
):
    previous, anchor = history[-2:]
    previous_samples = _batch_transforms(
        _record_samples(previous, sample_count, seed, stream)
    )
    anchor_samples = _batch_transforms(
        _record_samples(anchor, sample_count, seed, stream)
    )
    source_duration = anchor.stamp - previous.stamp
    horizon = requested_stamp - anchor.stamp
    process_generator = np.random.default_rng(
        _derived_seed(seed, stream, "process:" + source_record_dependency_id(anchor))
    )
    process_covariance = process_noise_spectral_density * horizon
    if np.max(np.abs(process_covariance)) <= 1.0e-16:
        process_samples = np.zeros((sample_count, 6), dtype=float)
    else:
        process_samples = process_generator.multivariate_normal(
            np.zeros(6),
            process_covariance,
            size=sample_count,
            check_valid="raise",
        )
    acceleration = np.asarray(acceleration, dtype=float).reshape(6)
    transforms = []
    for previous_transform, anchor_transform, noise in zip(
        previous_samples,
        anchor_samples,
        process_samples,
    ):
        twist = body_twist_between(
            previous_transform,
            anchor_transform,
            source_duration,
        )
        endpoint_twist = twist + 0.5 * acceleration * source_duration
        deterministic_increment = (
            endpoint_twist * horizon
            + 0.5 * acceleration * horizon ** 2
        )
        prediction = compose_transforms(
            anchor_transform,
            se3_exp(deterministic_increment),
        )
        transforms.append(compose_transforms(prediction, se3_exp(noise)))
    components, approximation = _sample_components(
        tuple(transforms),
        history,
        detail,
    )
    central_twist = body_twist_between(
        record_representative(previous),
        record_representative(anchor),
        source_duration,
    )
    central_endpoint_twist = (
        central_twist + 0.5 * acceleration * source_duration
    )
    representative = compose_transforms(
        record_representative(anchor),
        se3_exp(
            central_endpoint_twist * horizon
            + 0.5 * acceleration * horizon ** 2
        ),
    )
    return _output_record(
        anchor,
        requested_stamp,
        components,
        representative,
        history,
        detail,
        approximation,
    )


__all__ = [
    "component_pose_covariance",
    "copy_record_at_stamp",
    "moment_interpolate",
    "moment_predict",
    "record_representative",
    "record_uncertainty_trace",
    "sample_interpolate",
    "sample_predict",
]
