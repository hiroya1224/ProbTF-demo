import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    DistributionValidationError,
    OrientationKind,
    RepresentativeKind,
    RepresentativePolicy,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
    bingham_shape_magnitude,
    dirac_shape_from_mode,
)
from probtf.geometry import (
    pack_symmetric_upper,
    quat_to_rotmat,
    right_perturbation_vec_rotation_jacobian,
    rotation_vector_from_quaternion,
    unpack_symmetric_upper,
)
from probtf.provenance import ApproximationKind, ComponentProvenance
from probtf_estimators import coupling_from_hessian
from probtf.compatibility import (
    LegacyProjectionPolicy,
    distribution_to_legacy_transform,
    legacy_transform_to_stamped,
)
from probtf.models import ProbabilisticTransform


def _translation(mean=(0.0, 0.0, 0.0), covariance=None, coupling=None):
    return ConditionalGaussianTranslation(
        mean_at_reference=np.asarray(mean, dtype=float),
        residual_covariance=np.zeros((3, 3)) if covariance is None else covariance,
        rotation_coupling=np.zeros((3, 9)) if coupling is None else coupling,
    )


def _component(component_id="component", weight=1.0, orientation=None, translation=None):
    return TransformComponent(
        component_id=component_id,
        raw_weight=weight,
        orientation=BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation,
        translation=_translation() if translation is None else translation,
        provenance=ComponentProvenance(source_ids=("test",)),
    )


def test_finite_bingham_uses_jmaa_shape_magnitude_and_reconstructs_parameter():
    parameter = np.diag([3.0, 1.0, -1.0, -3.0])
    orientation = BinghamOrientation.from_parameter_matrix(parameter)

    assert orientation.kind is OrientationKind.FINITE_BINGHAM
    assert orientation.inverse_concentration == pytest.approx(0.25)
    assert np.trace(orientation.shape_matrix) == pytest.approx(0.0)
    assert bingham_shape_magnitude(orientation.shape_matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(orientation.parameter_matrix(), parameter)
    assert not np.isclose(np.linalg.norm(orientation.shape_matrix), 1.0)


def test_small_nonzero_bingham_parameter_is_not_silently_made_uniform():
    parameter = 1e-12 * np.diag([3.0, 1.0, -1.0, -3.0])
    orientation = BinghamOrientation.from_parameter_matrix(parameter)
    assert orientation.kind is OrientationKind.FINITE_BINGHAM
    np.testing.assert_allclose(orientation.parameter_matrix(), parameter, rtol=1e-12, atol=0.0)


def test_bingham_validation_rejects_non_jmaa_shape_and_nonfinite_parameter():
    with pytest.raises(DistributionValidationError, match="JMAA normalization"):
        BinghamOrientation(
            OrientationKind.FINITE_BINGHAM,
            1.0,
            np.diag([0.75, -0.25, -0.25, -0.25]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    with pytest.raises(DistributionValidationError, match="finite"):
        BinghamOrientation.from_parameter_matrix(np.diag([np.nan, 0.0, 0.0, 0.0]))


def test_dirac_and_uniform_have_distinct_scale_contracts():
    mode = np.array([0.5, 0.5, 0.5, 0.5])
    dirac = BinghamOrientation.dirac(mode)
    uniform = BinghamOrientation.uniform(mode)

    assert dirac.inverse_concentration == 0.0
    np.testing.assert_allclose(dirac.shape_matrix, dirac_shape_from_mode(mode))
    with pytest.raises(DistributionValidationError, match="no finite"):
        dirac.parameter_matrix()
    assert uniform.inverse_concentration == np.inf
    np.testing.assert_allclose(uniform.parameter_matrix(), np.zeros((4, 4)))

    with pytest.raises(DistributionValidationError, match="zero inverse"):
        BinghamOrientation(OrientationKind.DIRAC, 1.0, dirac.shape_matrix, mode)
    with pytest.raises(DistributionValidationError, match="infinite"):
        BinghamOrientation(OrientationKind.UNIFORM, 0.0, np.zeros((4, 4)), mode)


def test_distribution_arrays_are_copied_read_only_and_covariance_is_psd():
    shape = dirac_shape_from_mode([1.0, 0.0, 0.0, 0.0])
    orientation = BinghamOrientation(
        OrientationKind.DIRAC,
        0.0,
        shape,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    shape[0, 0] = 123.0
    assert orientation.shape_matrix[0, 0] != 123.0
    with pytest.raises(ValueError):
        orientation.shape_matrix[0, 0] = 2.0

    with pytest.raises(DistributionValidationError, match="positive semidefinite"):
        _translation(covariance=np.diag([1.0, 1.0, -1e-3]))


def test_coupling_is_column_major_and_quaternion_sign_invariant():
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    query = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    coupling = np.arange(27, dtype=float).reshape(3, 9) / 10.0
    translation = _translation(mean=[1.0, 2.0, 3.0], coupling=coupling)
    component = _component(
        orientation=BinghamOrientation.dirac(reference),
        translation=translation,
    )

    expected = translation.mean_at_reference + coupling @ (
        quat_to_rotmat(query).reshape(9, order="F")
        - quat_to_rotmat(reference).reshape(9, order="F")
    )
    np.testing.assert_allclose(component.conditional_translation_mean(query), expected)
    np.testing.assert_allclose(
        component.conditional_translation_mean(query),
        component.conditional_translation_mean(-query),
    )
    np.testing.assert_allclose(
        rotation_vector_from_quaternion(query),
        quat_to_rotmat(query).reshape(9, order="F"),
    )


def test_right_perturbation_jacobian_has_documented_minimum_norm_inverse():
    jacobian = right_perturbation_vec_rotation_jacobian([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(0.5 * jacobian.T @ jacobian, np.eye(3), atol=1e-12)
    local_map = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0], [2.0, 0.0, 1.0]])
    coupling = local_map @ (0.5 * jacobian.T)
    np.testing.assert_allclose(coupling @ jacobian, local_map, atol=1e-12)


def test_hessian_coupling_uses_right_perturbation_minimum_norm_solution():
    hessian_xx = -np.diag([2.0, 4.0, 8.0])
    hessian_xu = np.array(
        [[1.0, 2.0, 3.0], [0.0, 2.0, 4.0], [8.0, 0.0, -8.0]],
        dtype=float,
    )
    result = coupling_from_hessian(
        hessian_xx,
        hessian_xu,
        [1.0, 0.0, 0.0, 0.0],
    )
    expected_local = -np.linalg.solve(hessian_xx, hessian_xu)
    np.testing.assert_allclose(result.local_translation_map, expected_local)
    np.testing.assert_allclose(
        result.rotation_coupling @ result.rotation_jacobian,
        expected_local,
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="nonsingular"):
        coupling_from_hessian(np.zeros((3, 3)), np.eye(3), [1.0, 0.0, 0.0, 0.0])


def test_weight_normalization_clamps_negative_and_preserves_order():
    distribution = TransformDistribution(
        (
            _component("first", -2.0),
            _component("second", 2.0),
            _component("third", 6.0),
        )
    )
    normalized = distribution.normalize_weights()

    assert normalized.status is DistributionStatus.OK
    assert [item.component.component_id for item in normalized.components] == ["second", "third"]
    assert [item.weight for item in normalized.components] == pytest.approx([0.25, 0.75])
    assert normalized.diagnostics[0].code == "NEGATIVE_WEIGHT_CLAMPED"


def test_weight_normalization_is_stable_near_float_max():
    distribution = TransformDistribution(
        (_component("first", 1e308), _component("second", 1e308))
    )
    normalized = distribution.normalize_weights()
    assert normalized.status is DistributionStatus.OK
    assert [item.weight for item in normalized.components] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("weights", [(0.0, 0.0), (-1.0, 0.0), (-1.0, -2.0)])
def test_all_nonpositive_weight_is_zero_mass(weights):
    distribution = TransformDistribution(
        tuple(_component("c{}".format(index), weight) for index, weight in enumerate(weights))
    )
    assert distribution.status() is DistributionStatus.ZERO_MASS
    assert distribution.normalize_weights().components == ()


def test_nonfinite_weight_is_invalid_not_identity_or_zero_mass():
    distribution = TransformDistribution((_component(weight=np.nan),))
    assert distribution.status() is DistributionStatus.INVALID
    assert distribution.normalize_weights().status is DistributionStatus.INVALID


def test_deterministic_reduction_and_explicit_lossy_representative():
    deterministic = TransformDistribution(
        (_component(translation=_translation([1.0, 2.0, 3.0])),)
    )
    exact = deterministic.representative(RepresentativePolicy.EXACT_ONLY)
    assert exact.kind is RepresentativeKind.EXACT_MAP
    np.testing.assert_allclose(exact.transform.apply([1.0, 0.0, 0.0]), [2.0, 2.0, 3.0])

    finite = BinghamOrientation.from_parameter_matrix(np.diag([3.0, 1.0, -1.0, -3.0]))
    stochastic = TransformDistribution((_component(orientation=finite),))
    with pytest.raises(ValueError, match="no exact"):
        stochastic.representative(RepresentativePolicy.EXACT_ONLY)
    projected = stochastic.representative(RepresentativePolicy.HIGHEST_WEIGHT_COMPONENT_MODE)
    assert projected.kind is RepresentativeKind.COMPONENT_MODE_APPROXIMATION
    assert projected.approximation.lossy


def test_stamped_edge_direction_and_frame_normalization():
    distribution = TransformDistribution((_component(),))
    record = TransformDistributionStamped(
        parent_frame_id="/world",
        child_frame_id="tool",
        stamp=1.5,
        edge_id="world_to_tool",
        authority="test",
        distribution=distribution,
        is_static=True,
    )
    assert record.parent_frame_id == "world"
    assert record.child_frame_id == "tool"
    assert record.is_static


@pytest.mark.parametrize("size", [3, 4])
def test_symmetric_upper_round_trip_uses_fixed_order(size):
    values = np.arange(size * size, dtype=float).reshape(size, size)
    matrix = values + values.T
    packed = pack_symmetric_upper(matrix)
    np.testing.assert_allclose(unpack_symmetric_upper(packed, size), matrix)


def test_legacy_adapter_preserves_single_uncoupled_finite_component():
    legacy = ProbabilisticTransform.from_arrays(
        parent_frame_id="world",
        child_frame_id="tool",
        position_mean=[1.0, 2.0, 3.0],
        position_covariance=np.diag([0.1, 0.2, 0.3]),
        orientation_bingham=np.diag([0.0, -1.0, -2.0, -3.0]),
        orientation_mode_wxyz=[1.0, 0.0, 0.0, 0.0],
        stamp=2.0,
        edge_id="edge",
        source_id="producer",
    )
    converted = legacy_transform_to_stamped(legacy)
    assert converted.diagnostics == ("LEGACY_INDEPENDENCE_ASSUMPTION",)
    component = converted.value.distribution.components[0]
    np.testing.assert_allclose(component.translation.rotation_coupling, np.zeros((3, 9)))

    round_trip = distribution_to_legacy_transform(converted.value)
    np.testing.assert_allclose(round_trip.value.position_mean, legacy.position_mean)
    np.testing.assert_allclose(
        round_trip.value.orientation_bingham,
        legacy.orientation_bingham,
    )


def test_legacy_projection_rejects_silent_mixture_coupling_and_dirac_loss():
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -1.0, -2.0, -3.0]))
    coupled = _component(
        "coupled",
        orientation=finite,
        translation=_translation(coupling=np.ones((3, 9))),
    )
    other = _component("other", orientation=finite)
    record = TransformDistributionStamped(
        "world",
        "tool",
        0.0,
        "edge",
        "test",
        TransformDistribution((coupled, other)),
    )
    with pytest.raises(ValueError, match="one component"):
        distribution_to_legacy_transform(record)
    projected = distribution_to_legacy_transform(
        record,
        LegacyProjectionPolicy.HIGHEST_WEIGHT_COMPONENT_MODE,
    )
    assert "MIXTURE_COMPONENT_MODE_PROJECTION" in projected.diagnostics
    assert "ROTATION_COUPLING_EVALUATED_AT_MODE" in projected.diagnostics

    dirac_record = TransformDistributionStamped(
        "world",
        "camera",
        0.0,
        "dirac",
        "test",
        TransformDistribution((_component(),)),
    )
    with pytest.raises(ValueError, match="Dirac"):
        distribution_to_legacy_transform(dirac_record)


def test_legacy_lossy_approximation_metadata_survives_round_trip():
    legacy = ProbabilisticTransform.from_arrays(
        parent_frame_id="world",
        child_frame_id="tool",
        position_mean=np.zeros(3),
        position_covariance=np.eye(3),
        orientation_bingham=np.diag([0.0, -1.0, -2.0, -3.0]),
        approximation_type="moment_closure",
        closure_approximation=True,
    )
    converted = legacy_transform_to_stamped(legacy)
    assert converted.value.approximation.kind is ApproximationKind.MOMENT_SUMMARY
    assert converted.value.approximation.lossy
    assert "LEGACY_LOSSY_APPROXIMATION" in converted.diagnostics

    restored = distribution_to_legacy_transform(converted.value)
    assert restored.value.approximation_type == "moment_closure"
    assert restored.value.closure_approximation
    assert "SOURCE_APPROXIMATION_PRESERVED" in restored.diagnostics
