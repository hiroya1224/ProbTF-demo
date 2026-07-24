"""Moment and common-random-number sample backends for temporal models."""

from dataclasses import replace
import hashlib

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    OrientationKind,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    compose_transforms,
    interpolate_transform,
    quat_left_matrix,
    quat_to_rotmat,
    relative_transform,
    right_perturbation_vec_rotation_jacobian,
    se3_exp,
    se3_log,
)
from probtf.probability import (
    TransformSampleBatch,
    sample_bingham_orientation,
)
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


def _combined_component_id(prefix, *component_ids):
    digest = hashlib.sha256()
    for component_id in component_ids:
        encoded = str(component_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return "{}:{}".format(prefix, digest.hexdigest()[:24])


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
            component.orientation.mode_wxyz
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
        try:
            covariance = component_pose_covariance(component)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
        pose = component_representative(component)
        translation_offset = pose.translation - center.translation
        # This diagnostic is only a scalar safety summary; the actual
        # distribution remains in its native representation.
        from probtf.geometry import relative_transform, se3_log

        rotation_offset = se3_log(relative_transform(center, pose))[3:]
        offset = np.concatenate((translation_offset, rotation_offset))
        total += weighted.weight * (float(np.trace(covariance)) + float(offset @ offset))
    return max(0.0, total)


def _mixed_pose_perturb(transform, delta):
    value = np.asarray(delta, dtype=float).reshape(6)
    rotation_delta = se3_exp(np.concatenate((np.zeros(3), value[3:])))
    rotated = compose_transforms(transform, rotation_delta)
    return DeterministicTransform(
        transform.translation + value[:3],
        rotated.rotation_wxyz,
    )


def _mixed_pose_residual(reference, value):
    return np.concatenate(
        (
            value.translation - reference.translation,
            se3_log(relative_transform(reference, value))[3:],
        )
    )


def _finite_difference_pose_jacobians(function, inputs, epsilon=1.0e-6):
    """Jacobians for mixed parent-translation/right-rotation pose moments."""

    nominal = function(*inputs)
    jacobians = []
    for input_index, transform in enumerate(inputs):
        jacobian = np.zeros((6, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            positive = list(inputs)
            negative = list(inputs)
            positive[input_index] = _mixed_pose_perturb(transform, delta)
            negative[input_index] = _mixed_pose_perturb(transform, -delta)
            plus = _mixed_pose_residual(nominal, function(*positive))
            minus = _mixed_pose_residual(nominal, function(*negative))
            jacobian[:, column] = (plus - minus) / (2.0 * epsilon)
        jacobians.append(jacobian)
    return nominal, tuple(jacobians)


def _body_process_map(transform):
    process_map = np.zeros((6, 6), dtype=float)
    process_map[:3, :3] = quat_to_rotmat(transform.rotation_wxyz)
    process_map[3:, 3:] = np.eye(3)
    return process_map


def moment_interpolate(
    left,
    right,
    requested_stamp,
    process_noise_spectral_density,
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
    left_normalized = left.distribution.normalize_weights()
    right_normalized = right.distribution.normalize_weights()
    if not left_normalized.components or not right_normalized.components:
        raise ValueError("Moment interpolation requires positive finite mixture mass.")
    bridge_covariance = (
        np.asarray(process_noise_spectral_density, dtype=float)
        * duration
        * alpha
        * (1.0 - alpha)
    )
    for left_weighted in left_normalized.components:
        for right_weighted in right_normalized.components:
            left_component = left_weighted.component
            right_component = right_weighted.component

            def interpolation_function(left_pose, right_pose):
                return interpolate_transform(left_pose, right_pose, alpha)

            left_pose = component_representative(left_component)
            right_pose = component_representative(right_component)
            left_covariance = component_pose_covariance(left_component)
            right_covariance = component_pose_covariance(right_component)
            if (
                not np.any(left_covariance)
                and not np.any(right_covariance)
            ):
                transform = interpolation_function(left_pose, right_pose)
                covariance = np.zeros((6, 6), dtype=float)
            else:
                transform, (left_jacobian, right_jacobian) = (
                    _finite_difference_pose_jacobians(
                        interpolation_function,
                        (left_pose, right_pose),
                    )
                )
                covariance = (
                    left_jacobian @ left_covariance @ left_jacobian.T
                    + right_jacobian @ right_covariance @ right_jacobian.T
                )
            process_map = _body_process_map(transform)
            covariance += process_map @ bridge_covariance @ process_map.T
            components.append(
                component_from_pose_covariance(
                    component_id=_combined_component_id(
                        "interpolated",
                        left_component.component_id,
                        right_component.component_id,
                    ),
                    raw_weight=left_weighted.weight * right_weighted.weight,
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
    acceleration,
    endpoint_twist_function,
    increment_function,
    process_covariance,
    detail,
):
    previous, anchor = history[-2:]
    source_duration = anchor.stamp - previous.stamp
    horizon = requested_stamp - anchor.stamp
    acceleration = np.asarray(acceleration, dtype=float).reshape(6)
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
    previous_normalized = previous.distribution.normalize_weights()
    anchor_normalized = anchor.distribution.normalize_weights()
    if not previous_normalized.components or not anchor_normalized.components:
        raise ValueError("Moment prediction requires positive finite mixture mass.")

    def prediction_function(previous_pose, anchor_pose):
        endpoint_twist = endpoint_twist_function(
            previous_pose,
            anchor_pose,
            source_duration,
            acceleration,
        )
        return compose_transforms(
            anchor_pose,
            increment_function(endpoint_twist, acceleration, horizon),
        )

    for previous_weighted in previous_normalized.components:
        for anchor_weighted in anchor_normalized.components:
            previous_component = previous_weighted.component
            anchor_component = anchor_weighted.component
            previous_pose = component_representative(previous_component)
            anchor_pose = component_representative(anchor_component)
            previous_covariance = component_pose_covariance(previous_component)
            anchor_covariance = component_pose_covariance(anchor_component)
            if (
                not np.any(previous_covariance)
                and not np.any(anchor_covariance)
            ):
                new_transform = prediction_function(previous_pose, anchor_pose)
                propagated = np.zeros((6, 6), dtype=float)
            else:
                new_transform, (previous_jacobian, anchor_jacobian) = (
                    _finite_difference_pose_jacobians(
                        prediction_function,
                        (previous_pose, anchor_pose),
                    )
                )
                propagated = (
                    previous_jacobian
                    @ previous_covariance
                    @ previous_jacobian.T
                    + anchor_jacobian
                    @ anchor_covariance
                    @ anchor_jacobian.T
                )
            process_map = _body_process_map(new_transform)
            propagated += process_map @ process_covariance @ process_map.T
            components.append(
                component_from_pose_covariance(
                    component_id=_combined_component_id(
                        "predicted",
                        previous_component.component_id,
                        anchor_component.component_id,
                    ),
                    raw_weight=previous_weighted.weight * anchor_weighted.weight,
                    transform=new_transform,
                    covariance=propagated,
                    provenance=component_provenance,
                    approximation=approximation,
                )
            )
    representative = prediction_function(
        record_representative(previous),
        record_representative(anchor),
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


def _derived_seed(seed, stream, dependency_id):
    digest = hashlib.sha256()
    digest.update(str(0 if seed is None else int(seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(stream).encode("utf-8"))
    digest.update(b"\0")
    digest.update(dependency_id.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _sampling_dependency_id(record):
    """Sampling identity for a complete record law.

    ``source_ids`` can name only one factor in a record whose residual noise
    is independent.  Collapsing the whole record to those IDs would therefore
    create false perfect cross-time correlation.  Until factor-level joint
    samples are part of the public contract, raw endpoint records use their
    complete immutable content identity and the approximation is diagnosed.
    """

    return source_record_dependency_id(record)


def _record_samples(record, count, seed, stream):
    dependency = _sampling_dependency_id(record)
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError(
            "Cannot sample a {} transform distribution.".format(
                normalized.status.value
            )
        )
    choice_generator = np.random.default_rng(
        _derived_seed(seed, stream, dependency + ":mixture")
    )
    weights = np.array([item.weight for item in normalized.components], dtype=float)
    choices = choice_generator.choice(
        len(normalized.components),
        size=count,
        p=weights,
    )
    translations = np.empty((count, 3), dtype=float)
    rotations = np.empty((count, 4), dtype=float)
    for component_index, weighted in enumerate(normalized.components):
        selected = np.flatnonzero(choices == component_index)
        if not len(selected):
            continue
        component = weighted.component
        component_key = _combined_component_id(
            dependency,
            component.component_id,
        )
        orientation_generator = np.random.default_rng(
            _derived_seed(seed, stream, component_key + ":orientation")
        )
        selected_rotations = sample_bingham_orientation(
            component.orientation,
            len(selected),
            orientation_generator,
        )
        selected_translations = np.asarray(
            [
                component.conditional_translation_mean(quaternion)
                for quaternion in selected_rotations
            ],
            dtype=float,
        ).reshape(len(selected), 3)
        covariance = component.translation.residual_covariance
        if not np.allclose(covariance, 0.0, rtol=0.0, atol=0.0):
            residual_generator = np.random.default_rng(
                _derived_seed(seed, stream, component_key + ":residual")
            )
            selected_translations += residual_generator.multivariate_normal(
                np.zeros(3),
                covariance,
                size=len(selected),
            )
        translations[selected] = selected_translations
        rotations[selected] = selected_rotations
    return TransformSampleBatch(translations, rotations)


def _spectral_square_root(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    )
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))


def _dyadic_process_samples(
    process_noise_spectral_density,
    time,
    domain,
    count,
    seed,
    stream,
    dependency_id,
    *,
    bridge=False,
    depth=48,
):
    """Evaluate a stateless Brownian path on a deterministic dyadic tree.

    Every query reuses the same endpoint and Brownian-bridge node innovations
    addressed by ``(seed, stream, dependency, level, interval)``.  Thus
    distinct horizons share their common path ancestry, while short horizons
    retain ``Qc * h`` variance instead of suffering a fixed-basis cutoff.
    """

    time = float(time)
    domain = float(domain)
    if time == 0.0 or np.max(np.abs(process_noise_spectral_density)) <= 1.0e-16:
        return np.zeros((count, 6), dtype=float)
    if domain <= 0.0 or time < 0.0 or time > domain + 1.0e-12:
        raise ValueError("Gaussian path time must lie within a positive domain.")
    depth = int(depth)
    if depth < 1:
        raise ValueError("Brownian path depth must be positive.")
    prefix = ("bridge:" if bridge else "brownian:") + dependency_id

    left_time = 0.0
    right_time = domain
    left_value = np.zeros((count, 6), dtype=float)
    if bridge:
        right_value = np.zeros((count, 6), dtype=float)
    else:
        endpoint_generator = np.random.default_rng(
            _derived_seed(seed, stream, prefix + ":endpoint")
        )
        right_value = (
            np.sqrt(domain)
            * endpoint_generator.standard_normal((count, 6))
        )
    if np.isclose(time, domain, rtol=0.0, atol=0.0):
        scalar_path = right_value
    else:
        interval_index = 0
        scalar_path = None
        for level in range(depth):
            midpoint = 0.5 * (left_time + right_time)
            node_generator = np.random.default_rng(
                _derived_seed(
                    seed,
                    stream,
                    "{}:level:{}:interval:{}".format(
                        prefix,
                        level,
                        interval_index,
                    ),
                )
            )
            midpoint_value = (
                0.5 * (left_value + right_value)
                + np.sqrt(0.25 * (right_time - left_time))
                * node_generator.standard_normal((count, 6))
            )
            if np.isclose(time, midpoint, rtol=0.0, atol=0.0):
                scalar_path = midpoint_value
                break
            if time < midpoint:
                right_time = midpoint
                right_value = midpoint_value
                interval_index *= 2
            else:
                left_time = midpoint
                left_value = midpoint_value
                interval_index = 2 * interval_index + 1
        if scalar_path is None:
            fraction = (time - left_time) / (right_time - left_time)
            scalar_path = (
                (1.0 - fraction) * left_value + fraction * right_value
            )
    return scalar_path @ _spectral_square_root(process_noise_spectral_density).T


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
    process_noise_spectral_density,
    sample_count,
    seed,
    stream,
    process_depth,
    detail,
):
    left_samples = _batch_transforms(_record_samples(left, sample_count, seed, stream))
    right_samples = _batch_transforms(_record_samples(right, sample_count, seed, stream))
    alpha = (requested_stamp - left.stamp) / (right.stamp - left.stamp)
    central_transforms = tuple(
        interpolate_transform(left_transform, right_transform, alpha)
        for left_transform, right_transform in zip(left_samples, right_samples)
    )
    duration = right.stamp - left.stamp
    bridge_samples = _dyadic_process_samples(
        process_noise_spectral_density,
        requested_stamp - left.stamp,
        duration,
        sample_count,
        seed,
        stream,
        "{}|{}".format(
            _sampling_dependency_id(left),
            _sampling_dependency_id(right),
        ),
        bridge=True,
        depth=process_depth,
    )
    transforms = tuple(
        compose_transforms(transform, se3_exp(noise))
        for transform, noise in zip(central_transforms, bridge_samples)
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
    endpoint_twist_function,
    increment_function,
    process_noise_spectral_density,
    maximum_horizon,
    sample_count,
    seed,
    stream,
    process_depth,
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
    process_samples = _dyadic_process_samples(
        process_noise_spectral_density,
        horizon,
        maximum_horizon,
        sample_count,
        seed,
        stream,
        _sampling_dependency_id(anchor),
        depth=process_depth,
    )
    acceleration = np.asarray(acceleration, dtype=float).reshape(6)
    transforms = []
    for previous_transform, anchor_transform, noise in zip(
        previous_samples,
        anchor_samples,
        process_samples,
    ):
        endpoint_twist = endpoint_twist_function(
            previous_transform,
            anchor_transform,
            source_duration,
            acceleration,
        )
        prediction = compose_transforms(
            anchor_transform,
            increment_function(endpoint_twist, acceleration, horizon),
        )
        transforms.append(compose_transforms(prediction, se3_exp(noise)))
    components, approximation = _sample_components(
        tuple(transforms),
        history,
        detail,
    )
    central_endpoint_twist = endpoint_twist_function(
        record_representative(previous),
        record_representative(anchor),
        source_duration,
        acceleration,
    )
    representative = compose_transforms(
        record_representative(anchor),
        increment_function(central_endpoint_twist, acceleration, horizon),
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
