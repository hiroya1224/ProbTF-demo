import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
    compose_with_deterministic_right,
)
from probtf.geometry import (
    DeterministicTransform,
    axis_angle_to_quat,
    quat_mul,
    quat_to_rotmat,
)
from probtf.provenance import ComponentProvenance, TransformProvenance


def _record(orientation, coupling=None, representative=None):
    component = TransformComponent(
        "object",
        2.5,
        orientation,
        ConditionalGaussianTranslation(
            np.array([0.2, -0.1, 0.4]),
            np.diag([0.01, 0.02, 0.03]),
            np.zeros((3, 9)) if coupling is None else coupling,
        ),
        ComponentProvenance(source_ids=("camera",)),
    )
    return TransformDistributionStamped(
        "world",
        "object",
        3.0,
        "world_object",
        "camera",
        TransformDistribution((component,)),
        representative=representative,
        representative_kind=(
            RepresentativeKind.NONE
            if representative is None
            else RepresentativeKind.PRODUCER_SUPPLIED
        ),
        provenance=TransformProvenance(source_ids=("camera",)),
    )


def test_deterministic_right_composition_preserves_conditional_transform_samples():
    orientation = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -3.0, -5.0, -8.0])
    )
    rng = np.random.default_rng(17)
    coupling = rng.normal(scale=0.05, size=(3, 9))
    record = _record(orientation, coupling)
    fixed = DeterministicTransform(
        np.array([0.15, -0.04, 0.08]),
        axis_angle_to_quat([0.0, 0.0, 1.0], 0.35),
    )

    composed = compose_with_deterministic_right(
        record,
        fixed,
        child_frame_id="grasp",
        edge_id="world_grasp",
        authority="grasp_builder",
    )
    old_component = record.distribution.components[0]
    new_component = composed.distribution.components[0]

    for _ in range(20):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        new_quaternion = quat_mul(quaternion, fixed.rotation_wxyz)
        old_translation = old_component.conditional_translation_mean(quaternion)
        expected_translation = old_translation + quat_to_rotmat(quaternion) @ fixed.translation
        np.testing.assert_allclose(
            new_component.conditional_translation_mean(new_quaternion),
            expected_translation,
            atol=1e-12,
        )

    assert new_component.raw_weight == 2.5
    np.testing.assert_allclose(
        new_component.translation.residual_covariance,
        old_component.translation.residual_covariance,
    )
    assert new_component.provenance.derived_from_edge_ids == ("world_object",)
    assert composed.provenance.derived_from_edge_ids == ("world_object",)


def test_deterministic_right_composition_preserves_mixture_and_representative():
    representative = DeterministicTransform(
        np.array([1.0, 2.0, 3.0]),
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.2),
    )
    record = _record(
        BinghamOrientation.dirac(representative.rotation_wxyz),
        representative=representative,
    )
    fixed = DeterministicTransform(
        np.array([0.4, 0.0, 0.0]),
        axis_angle_to_quat([0.0, 1.0, 0.0], -0.3),
    )

    composed = compose_with_deterministic_right(
        record,
        fixed,
        "grasp",
        "world_grasp",
    )

    expected = fixed.then(representative)
    np.testing.assert_allclose(composed.representative.translation, expected.translation)
    np.testing.assert_allclose(composed.representative.rotation_wxyz, expected.rotation_wxyz)
    assert composed.representative_kind is RepresentativeKind.PRODUCER_SUPPLIED
    assert composed.distribution.components[0].component_id == "world_grasp:object"
