from dataclasses import replace

import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    compose_transforms,
    infer_endpoint_body_twist,
    integrate_linear_body_twist,
    right_perturbation_vec_rotation_jacobian,
    se3_exp,
    se3_log,
    relative_transform,
)
from probtf.provenance import TransformProvenance
from probtf.temporal import (
    ConstantBodyAccelerationModel,
    ConstantBodyTwistModel,
    TemporalDiagnosticCode,
    TemporalEvaluationRequest,
    TemporalPolicy,
    TemporalQueryMode,
    TemporalUncertaintyBackend,
    adapt_discrete_process_noise,
    parse_temporal_detail,
    source_record_dependency_id,
)
from probtf.temporal.backends import (
    _dyadic_process_samples,
    component_pose_covariance,
)


def _component(
    component_id="component",
    *,
    transform=None,
    orientation=None,
    covariance=None,
    coupling=None,
    weight=1.0,
):
    transform = DeterministicTransform.identity() if transform is None else transform
    orientation = (
        BinghamOrientation.dirac(transform.rotation_wxyz)
        if orientation is None
        else orientation
    )
    return TransformComponent(
        component_id,
        weight,
        orientation,
        ConditionalGaussianTranslation(
            transform.translation,
            np.zeros((3, 3)) if covariance is None else covariance,
            np.zeros((3, 9)) if coupling is None else coupling,
        ),
    )


def _record(
    stamp,
    *,
    transform=None,
    components=None,
    representative=None,
    representative_kind=RepresentativeKind.NONE,
    source_ids=(),
):
    if components is None:
        components = (_component(transform=transform),)
    return TransformDistributionStamped(
        "world",
        "tool",
        float(stamp),
        "world_tool",
        "authority",
        TransformDistribution(tuple(components)),
        representative=representative,
        representative_kind=representative_kind,
        provenance=TransformProvenance(source_ids=tuple(source_ids)),
    )


def _request(
    model,
    stamp,
    policy,
    anchors,
    *,
    mode=TemporalQueryMode.ONLINE,
    seed=7,
    max_horizon=2.0,
    max_age=2.0,
):
    return TemporalEvaluationRequest(
        requested_stamp=stamp,
        policy=policy,
        anchors=tuple(anchors),
        model_selector=model.model_id,
        max_prediction_horizon=max_horizon,
        max_age=max_age,
        random_seed=seed,
        random_stream="audit",
        query_mode=mode,
    )


def test_qc_adapter_is_symmetric_traceable_and_model_configuration_is_frozen():
    nonsymmetric = np.zeros((6, 6))
    nonsymmetric[0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetric"):
        ConstantBodyTwistModel(nonsymmetric, 1.0)

    adaptation = adapt_discrete_process_noise(np.eye(6) * 0.2, 0.1)
    np.testing.assert_allclose(adaptation.spectral_density, np.eye(6) * 2.0)
    assert adaptation.diagnostic is TemporalDiagnosticCode.DISCRETE_PROCESS_NOISE_ADAPTED
    assert "Qd" in adaptation.detail and "Qc" in adaptation.detail

    model = ConstantBodyTwistModel(np.zeros((6, 6)), 1.0)
    fingerprint = model.config_fingerprint
    for name, value in (
        ("sample_count", 99),
        ("maximum_horizon", 999.0),
        ("process_noise_spectral_density", np.eye(6)),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(model, name, value)
    assert model.config_fingerprint == fingerprint


def test_distribution_support_rejects_empty_uniform_and_axial_moment_laws():
    moment = ConstantBodyTwistModel(np.zeros((6, 6)), 1.0)
    sample = ConstantBodyTwistModel(
        np.zeros((6, 6)),
        1.0,
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=8,
    )
    empty = _record(0.0, components=())
    uniform = _record(
        0.0,
        components=(_component(orientation=BinghamOrientation.uniform()),),
    )
    axial = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, 0.0, -3.0, -5.0])
    )
    axial_record = _record(
        0.0,
        components=(_component(orientation=axial),),
    )
    assert not moment.validate_distribution(empty)
    assert not sample.validate_distribution(empty)
    assert not moment.validate_distribution(uniform)
    assert not moment.validate_distribution(axial_record)
    assert sample.validate_distribution(uniform)
    assert sample.validate_distribution(axial_record)


def test_translation_rotation_coupling_is_linearized_at_component_mode():
    orientation = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -6.0, -9.0, -12.0]),
        reference_quaternion_wxyz=[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
    )
    coupling = np.arange(27, dtype=float).reshape(3, 9) * 1.0e-3
    component = _component(
        orientation=orientation,
        coupling=coupling,
        covariance=np.eye(3) * 0.01,
    )
    covariance = component_pose_covariance(component)
    rotation_covariance = covariance[3:, 3:]
    expected_cross = (
        coupling
        @ right_perturbation_vec_rotation_jacobian(orientation.mode_wxyz)
        @ rotation_covariance
    )
    reference_cross = (
        coupling
        @ right_perturbation_vec_rotation_jacobian(
            orientation.reference_quaternion_wxyz
        )
        @ rotation_covariance
    )
    np.testing.assert_allclose(covariance[:3, 3:], expected_cross, atol=1.0e-12)
    assert np.linalg.norm(expected_cross - reference_cross) > 1.0e-3


def test_moment_prediction_propagates_both_anchor_covariances():
    covariance = np.diag([0.04, 0.0, 0.0])
    previous = _record(
        0.0,
        components=(_component(covariance=covariance),),
    )
    anchor = _record(
        1.0,
        components=(_component(covariance=covariance),),
    )
    model = ConstantBodyTwistModel(np.zeros((6, 6)), 2.0)
    result = model.predict(
        (previous, anchor),
        _request(
            model,
            2.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (previous, anchor),
        ),
    )
    output_covariance = component_pose_covariance(
        result.record.distribution.components[0]
    )
    # Scalar constant-velocity extrapolation is -x0 + 2*x1.
    assert output_covariance[0, 0] == pytest.approx(5.0 * 0.04, rel=2.0e-4)
    assert TemporalDiagnosticCode.DEPENDENCE_APPROXIMATED in result.diagnostics


def test_interpolation_transports_endpoint_moments_adds_bridge_qc_and_safe_ids():
    finite = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -80.0, -70.0, -60.0])
    )
    left_components = (
        _component(
            "a~b",
            orientation=finite,
            covariance=np.eye(3) * 0.01,
            weight=1.0,
        ),
        _component(
            "a",
            transform=se3_exp([0.1, 0.0, 0.0, 0.0, 0.0, 0.1]),
            orientation=finite,
            covariance=np.eye(3) * 0.01,
            weight=1.0,
        ),
        _component("negative", weight=-100.0),
    )
    right_components = (
        _component(
            "c",
            transform=se3_exp([1.0, 0.2, 0.0, 1.2, 0.7, -0.4]),
            orientation=finite,
            covariance=np.eye(3) * 0.02,
        ),
        _component(
            "b~c",
            transform=se3_exp([1.1, -0.1, 0.2, 1.1, 0.6, -0.3]),
            orientation=finite,
            covariance=np.eye(3) * 0.02,
        ),
    )
    left = _record(0.0, components=left_components)
    right = _record(2.0, components=right_components)
    model = ConstantBodyTwistModel(np.eye(6) * 0.05, 2.0)
    result = model.interpolate(
        left,
        right,
        _request(
            model,
            0.8,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            (left, right),
            mode=TemporalQueryMode.OFFLINE_SMOOTHING,
        ),
    )
    components = result.record.distribution.components
    assert len(components) == 4
    assert len({component.component_id for component in components}) == 4
    covariance = component_pose_covariance(components[0])
    assert np.linalg.norm(covariance - covariance.T) < 1.0e-10
    assert np.linalg.eigvalsh(covariance)[0] >= -1.0e-9
    assert np.linalg.norm(covariance[:3, 3:]) > 1.0e-5

    deterministic_left = _record(0.0)
    deterministic_right = _record(
        2.0,
        transform=se3_exp([1.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
    )
    bridge = model.interpolate(
        deterministic_left,
        deterministic_right,
        _request(
            model,
            1.0,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            (deterministic_left, deterministic_right),
            mode=TemporalQueryMode.OFFLINE_SMOOTHING,
        ),
    )
    assert bridge.result_uncertainty_trace > 0.0


def test_noncommuting_constant_acceleration_uses_shooting_and_time_ordering():
    initial_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    acceleration = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    source_duration = 1.0
    previous_pose = DeterministicTransform.identity()
    source_increment = integrate_linear_body_twist(
        initial_twist,
        acceleration,
        source_duration,
        substeps=512,
    )
    anchor_pose = compose_transforms(previous_pose, source_increment)
    inferred = infer_endpoint_body_twist(
        previous_pose,
        anchor_pose,
        source_duration,
        acceleration,
        substeps=128,
    )
    np.testing.assert_allclose(
        inferred,
        initial_twist + acceleration * source_duration,
        atol=2.0e-5,
    )

    model = ConstantBodyAccelerationModel(
        acceleration,
        "synthetic",
        "tool",
        np.zeros((6, 6)),
        1.0,
        integration_substeps=128,
    )
    previous = _record(0.0, transform=previous_pose)
    anchor = _record(1.0, transform=anchor_pose)
    horizon = 0.5
    result = model.predict(
        (previous, anchor),
        _request(
            model,
            1.0 + horizon,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (previous, anchor),
        ),
    )
    truth = compose_transforms(
        anchor_pose,
        integrate_linear_body_twist(
            initial_twist + acceleration * source_duration,
            acceleration,
            horizon,
            substeps=512,
        ),
    )
    actual = result.record.distribution.deterministic_transform()
    np.testing.assert_allclose(actual.translation, truth.translation, atol=3.0e-5)
    np.testing.assert_allclose(
        se3_log(relative_transform(actual, truth)),
        np.zeros(6),
        atol=3.0e-5,
    )
    collapsed = compose_transforms(
        anchor_pose,
        se3_exp(
            (initial_twist + acceleration * source_duration) * horizon
            + 0.5 * acceleration * horizon ** 2
        ),
    )
    assert np.linalg.norm(
        se3_log(relative_transform(collapsed, truth))
    ) > 1.0e-3


def test_dyadic_brownian_path_has_short_horizon_variance_and_joint_covariance():
    count = 12000
    qc = np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    short = _dyadic_process_samples(
        qc,
        1.0e-4,
        10.0,
        count,
        123,
        "path",
        "anchor",
        depth=48,
    )[:, 0]
    assert np.var(short, ddof=1) / 1.0e-4 == pytest.approx(1.0, abs=0.04)

    quarter = _dyadic_process_samples(
        qc,
        0.25,
        1.0,
        count,
        321,
        "path",
        "anchor",
        depth=48,
    )[:, 0]
    whole = _dyadic_process_samples(
        qc,
        1.0,
        1.0,
        count,
        321,
        "path",
        "anchor",
        depth=48,
    )[:, 0]
    covariance = np.cov(quarter, whole, ddof=1)[0, 1]
    correlation = np.corrcoef(quarter, whole)[0, 1]
    assert covariance == pytest.approx(0.25, abs=0.015)
    assert correlation == pytest.approx(0.5, abs=0.03)


def test_sample_seed_none_is_recorded_as_effective_zero_and_uniform_is_unbounded():
    uniform = BinghamOrientation.uniform()
    previous = _record(
        0.0,
        components=(_component(orientation=uniform),),
    )
    anchor = _record(
        1.0,
        components=(_component(orientation=uniform),),
    )
    model = ConstantBodyTwistModel(
        np.zeros((6, 6)),
        1.0,
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=24,
    )
    request = _request(
        model,
        1.2,
        TemporalPolicy.PREDICT_WITH_MODEL,
        (previous, anchor),
        seed=None,
    )
    first = model.predict((previous, anchor), request)
    second = model.predict((previous, anchor), request)
    assert first.random_seed == 0
    assert parse_temporal_detail(first.record.provenance.detail)["random_seed"] == 0
    assert np.isinf(first.initial_uncertainty_trace)
    assert np.isinf(first.result_uncertainty_trace)
    for left, right in zip(
        first.record.distribution.components,
        second.record.distribution.components,
    ):
        np.testing.assert_array_equal(
            left.translation.mean_at_reference,
            right.translation.mean_at_reference,
        )


def test_shared_source_lineage_does_not_create_false_perfect_record_correlation():
    covariance = np.diag([0.04, 0.0, 0.0])
    previous = _record(
        0.0,
        components=(_component(covariance=covariance),),
        source_ids=("shared-calibration",),
    )
    anchor = _record(
        1.0,
        components=(_component(covariance=covariance),),
        source_ids=("shared-calibration",),
    )
    model = ConstantBodyTwistModel(
        np.zeros((6, 6)),
        2.0,
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=4000,
    )
    result = model.predict(
        (previous, anchor),
        _request(
            model,
            2.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (previous, anchor),
            seed=99,
        ),
    )
    values = np.array(
        [
            component.translation.mean_at_reference[0]
            for component in result.record.distribution.components
        ]
    )
    # The current marginal-only contract cannot split shared calibration from
    # per-stamp measurement residual.  It deliberately avoids the worse
    # failure of correlating the complete records by source_id and reports the
    # resulting independence approximation.
    assert np.var(values, ddof=1) == pytest.approx(0.20, abs=0.02)
    assert TemporalDiagnosticCode.DEPENDENCE_APPROXIMATED in result.diagnostics
    assert result.warnings


def test_direct_model_api_rejects_selector_causal_and_anchor_contract_violations():
    previous = _record(0.0)
    anchor = _record(1.0)
    model = ConstantBodyTwistModel(np.zeros((6, 6)), 1.0)
    online_interpolation = _request(
        model,
        0.5,
        TemporalPolicy.INTERPOLATE_WITH_MODEL,
        (previous, anchor),
    )
    with pytest.raises(ValueError, match="NON_CAUSAL_INPUT_REJECTED"):
        model.interpolate(previous, anchor, online_interpolation)

    too_far = _request(
        model,
        1.8,
        TemporalPolicy.PREDICT_WITH_MODEL,
        (previous, anchor),
        max_horizon=0.2,
    )
    with pytest.raises(ValueError, match="PREDICTION_HORIZON_EXCEEDED"):
        model.predict((previous, anchor), too_far)

    mismatched = _request(
        model,
        1.2,
        TemporalPolicy.PREDICT_WITH_MODEL,
        (previous, anchor),
    )
    with pytest.raises(ValueError, match="ANCHOR_MISMATCH"):
        model.predict((_record(0.1), anchor), mismatched)


def test_dependency_hash_covers_representative_and_chained_lineage_is_flattened():
    base = _record(0.0)
    representative = replace(
        base,
        representative=DeterministicTransform(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ),
        representative_kind=RepresentativeKind.PRODUCER_SUPPLIED,
    )
    assert source_record_dependency_id(base) != source_record_dependency_id(
        representative
    )

    anchor = _record(1.0)
    model = ConstantBodyTwistModel(np.zeros((6, 6)), 2.0)
    first = model.predict(
        (base, anchor),
        _request(
            model,
            2.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (base, anchor),
        ),
    )
    second = model.predict(
        (anchor, first.record),
        _request(
            model,
            3.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (anchor, first.record),
        ),
    )
    assert set(second.dependency_ids) == {
        source_record_dependency_id(base),
        source_record_dependency_id(anchor),
    }
