import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    body_twist_between,
    compose_transforms,
    interpolate_transform,
    relative_transform,
    rotation_vector_to_quaternion,
    se3_exp,
    se3_log,
)
from probtf.temporal import (
    ConstantBodyAccelerationModel,
    ConstantBodyTwistModel,
    EndpointConditionedSampleInterpolationModel,
    TemporalDiagnosticCode,
    TemporalEvaluationKind,
    TemporalEvaluationRequest,
    TemporalPolicy,
    TemporalQueryMode,
    TemporalUncertaintyBackend,
    discrete_process_noise_to_spectral_density,
    parse_temporal_detail,
    source_record_dependency_id,
)


def _record(
    stamp,
    transform,
    *,
    covariance=None,
    orientation=None,
    component_id="component",
):
    covariance = (
        np.zeros((3, 3), dtype=float)
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    orientation = (
        BinghamOrientation.dirac(transform.rotation_wxyz)
        if orientation is None
        else orientation
    )
    return TransformDistributionStamped(
        "world",
        "tool",
        float(stamp),
        "world_tool",
        "test_authority",
        TransformDistribution(
            (
                TransformComponent(
                    component_id,
                    1.0,
                    orientation,
                    ConditionalGaussianTranslation(
                        transform.translation,
                        covariance,
                        np.zeros((3, 9)),
                    ),
                ),
            )
        ),
    )


def _request(
    stamp,
    policy,
    anchors,
    mode=TemporalQueryMode.ONLINE,
    seed=7,
    model_selector="se3_constant_body_twist",
):
    return TemporalEvaluationRequest(
        requested_stamp=stamp,
        policy=policy,
        anchors=tuple(anchors),
        model_selector=model_selector,
        max_prediction_horizon=1.0,
        max_age=2.0,
        random_seed=seed,
        random_stream="unit-test",
        query_mode=mode,
    )


def test_se3_exp_log_roundtrip_zero_short_and_near_pi():
    cases = (
        np.zeros(6),
        np.array([1.0e-12, -2.0e-12, 3.0e-12, 2.0e-12, 0.0, -1.0e-12]),
        np.array([0.4, -0.2, 0.8, 0.1, -0.3, 0.2]),
        np.array([0.2, 0.3, -0.1, np.pi - 1.0e-9, 0.0, 0.0]),
    )
    for value in cases:
        np.testing.assert_allclose(se3_log(se3_exp(value)), value, atol=2.0e-8)


def test_se3_interpolation_endpoints_sign_and_body_twist_convention():
    left = DeterministicTransform.identity()
    right = se3_exp(np.array([1.0, 0.5, -0.2, 0.0, 0.0, np.pi]))
    np.testing.assert_allclose(interpolate_transform(left, right, 0.0).translation, left.translation)
    np.testing.assert_allclose(interpolate_transform(left, right, 1.0).translation, right.translation)
    half = interpolate_transform(left, right, 0.5)
    reconstructed = compose_transforms(left, se3_exp(body_twist_between(left, right, 2.0)))
    np.testing.assert_allclose(reconstructed.translation, half.translation, atol=1.0e-10)
    np.testing.assert_allclose(
        relative_transform(left, reconstructed).rotation_wxyz,
        half.rotation_wxyz,
        atol=1.0e-10,
    )
    sign_flipped = DeterministicTransform(right.translation, -right.rotation_wxyz)
    np.testing.assert_allclose(
        interpolate_transform(left, sign_flipped, 0.5).rotation_wxyz,
        half.rotation_wxyz,
        atol=1.0e-10,
    )


def test_temporal_request_and_continuous_process_noise_contract():
    anchors = (_record(0.0, DeterministicTransform.identity()),)
    with pytest.raises(ValueError, match="model-based"):
        _request(0.0, TemporalPolicy.EXACT, anchors)
    qd = np.diag([0.2, 0.1, 0.3, 0.02, 0.01, 0.03])
    np.testing.assert_allclose(
        discrete_process_noise_to_spectral_density(qd, 0.2),
        qd / 0.2,
    )
    with pytest.raises(ValueError, match="positive"):
        discrete_process_noise_to_spectral_density(qd, 0.0)


def test_constant_body_twist_prediction_uses_coupled_se3_increment_and_provenance():
    start = _record(1.0, DeterministicTransform.identity())
    one_step = se3_exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0]))
    anchor = _record(2.0, one_step)
    model = ConstantBodyTwistModel(np.zeros((6, 6)), 2.0)
    result = model.predict(
        (start, anchor),
        _request(3.0, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor)),
    )
    expected = compose_transforms(
        one_step,
        se3_exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0])),
    )
    actual = result.record.distribution.deterministic_transform()
    np.testing.assert_allclose(actual.translation, expected.translation, atol=1.0e-10)
    np.testing.assert_allclose(actual.rotation_wxyz, expected.rotation_wxyz, atol=1.0e-10)
    assert result.evaluation_kind is TemporalEvaluationKind.MODEL_PREDICTION
    assert result.horizon == 1.0
    assert result.source_stamps == (1.0, 2.0)
    assert len(result.dependency_ids) == 2
    payload = parse_temporal_detail(result.record.provenance.detail)
    assert payload["model_id"] == model.model_id
    assert payload["backend"] == "moment"
    assert payload["requested_stamp"] == pytest.approx(3.0)
    assert payload["source_stamps"] == [1.0, 2.0]


def test_positive_qc_increases_prediction_uncertainty_and_zero_qc_is_deterministic():
    start = _record(0.0, DeterministicTransform.identity())
    anchor = _record(1.0, DeterministicTransform.identity())
    zero = ConstantBodyTwistModel(np.zeros((6, 6)), 2.0).predict(
        (start, anchor),
        _request(1.5, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor)),
    )
    assert zero.record.distribution.deterministic_transform() is not None
    assert zero.result_uncertainty_trace == pytest.approx(0.0, abs=1.0e-12)

    noisy_model = ConstantBodyTwistModel(np.diag([0.2] * 3 + [0.1] * 3), 2.0)
    short = noisy_model.predict(
        (start, anchor),
        _request(1.2, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor)),
    )
    long = noisy_model.predict(
        (start, anchor),
        _request(1.8, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor)),
    )
    assert short.result_uncertainty_trace > 0.0
    assert long.result_uncertainty_trace > short.result_uncertainty_trace


def test_constant_acceleration_requires_metadata_and_never_falls_back():
    kwargs = dict(
        body_acceleration=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        process_noise_spectral_density=np.zeros((6, 6)),
        maximum_horizon=2.0,
    )
    with pytest.raises(ValueError, match="acceleration_source"):
        ConstantBodyAccelerationModel(
            acceleration_source="",
            acceleration_frame="tool",
            **kwargs,
        )
    model = ConstantBodyAccelerationModel(
        acceleration_source="imu:linear_acceleration",
        acceleration_frame="tool",
        **kwargs,
    )
    start = _record(0.0, DeterministicTransform.identity())
    anchor = _record(1.0, DeterministicTransform.identity())
    result = model.predict(
        (start, anchor),
        _request(
            2.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (start, anchor),
            model_selector=model.model_id,
        ),
    )
    np.testing.assert_allclose(
        result.record.distribution.deterministic_transform().translation,
        [2.0, 0.0, 0.0],
        atol=1.0e-12,
    )


def test_stochastic_moment_interpolation_is_endpoint_conditioned_and_diagnosed():
    finite = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -20.0, -15.0, -10.0])
    )
    left = _record(
        0.0,
        DeterministicTransform.identity(),
        covariance=np.diag([0.1, 0.2, 0.3]),
        orientation=finite,
        component_id="left",
    )
    right = _record(
        2.0,
        DeterministicTransform(
            [2.0, 0.0, 0.0],
            rotation_vector_to_quaternion([0.0, 0.0, 0.4]),
        ),
        covariance=np.diag([0.2, 0.1, 0.2]),
        orientation=finite,
        component_id="right",
    )
    model = ConstantBodyTwistModel(np.zeros((6, 6)), 2.0)
    result = model.interpolate(
        left,
        right,
        _request(
            1.0,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            (left, right),
            TemporalQueryMode.OFFLINE_SMOOTHING,
        ),
    )
    np.testing.assert_allclose(result.record.representative.translation, [1.0, 0.0, 0.0])
    assert TemporalDiagnosticCode.MODEL_INTERPOLATION in result.diagnostics
    assert TemporalDiagnosticCode.ENDPOINT_CONDITIONED in result.diagnostics
    assert TemporalDiagnosticCode.DEPENDENCE_APPROXIMATED in result.diagnostics
    assert result.warnings


def test_sample_backend_reuses_sample_ids_and_is_seed_reproducible():
    finite = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -8.0, -5.0, -3.0])
    )
    start = _record(0.0, DeterministicTransform.identity(), orientation=finite)
    anchor = _record(
        1.0,
        DeterministicTransform([0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        orientation=finite,
    )
    model = ConstantBodyTwistModel(
        np.diag([0.01] * 6),
        1.0,
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=24,
    )
    first = model.predict(
        (start, anchor),
        _request(1.2, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor), seed=42),
    )
    second = model.predict(
        (start, anchor),
        _request(1.2, TemporalPolicy.PREDICT_WITH_MODEL, (start, anchor), seed=42),
    )
    first_components = first.record.distribution.components
    second_components = second.record.distribution.components
    assert [item.component_id for item in first_components] == [
        "sample:{:06d}".format(index) for index in range(24)
    ]
    for left, right in zip(first_components, second_components):
        np.testing.assert_array_equal(
            left.translation.mean_at_reference,
            right.translation.mean_at_reference,
        )
        np.testing.assert_array_equal(
            left.orientation.reference_quaternion_wxyz,
            right.orientation.reference_quaternion_wxyz,
        )


def test_endpoint_conditioned_sample_model_is_interpolation_only():
    model = EndpointConditionedSampleInterpolationModel(sample_count=8)
    assert model.supports_interpolation
    assert not model.supports_prediction
    with pytest.raises(NotImplementedError):
        model.predict((), None)


def test_source_dependency_identity_is_stable_and_content_sensitive():
    first = _record(1.0, DeterministicTransform.identity())
    identical = _record(1.0, DeterministicTransform.identity())
    changed = _record(
        1.0,
        DeterministicTransform([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    )
    assert source_record_dependency_id(first) == source_record_dependency_id(identical)
    assert source_record_dependency_id(first) != source_record_dependency_id(changed)
