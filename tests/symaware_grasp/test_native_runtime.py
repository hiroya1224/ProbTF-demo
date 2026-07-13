import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import rotation_action_matrix
from probtf.provenance import ApproximationInfo, ApproximationKind
from probtf_ros import ProbTfListener
from probtf_ros.v2_conversions import V2MessageTypes
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.beliefs import distribution_point_moments, make_transform_record
from symaware_grasp.distribution_metrics import fit_bingham_from_quaternion_samples
from symaware_grasp.ee_belief import EndEffectorBeliefModel
from symaware_grasp.models import GraspCandidate
from symaware_grasp.runtime import lookup_direct_record
from symaware_grasp.symmetry_aware_ik import SymmetryAwareIKSolver, _component_cost_models
from symaware_grasp_ros.messages import (
    SymawareMessageTypes,
    grasp_targets_to_msg,
    hand_belief_to_msg,
    object_belief_to_msg,
    record_from_app_message,
    selected_target_to_msg,
)


def _record(orientation=None):
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -3.0, -4.0, -5.0]))
    orientation = finite if orientation is None else orientation
    coupling = rotation_action_matrix([0.1, 0.0, 0.0])
    components = (
        TransformComponent(
            "first",
            1.0,
            orientation,
            ConditionalGaussianTranslation(
                np.array([0.2, 0.0, 0.0]),
                1e-4 * np.eye(3),
                coupling,
            ),
        ),
        TransformComponent(
            "second",
            3.0,
            finite,
            ConditionalGaussianTranslation(
                np.array([0.4, 0.0, 0.0]),
                2e-4 * np.eye(3),
                np.zeros((3, 9)),
            ),
        ),
    )
    return TransformDistributionStamped(
        "base_link",
        "target",
        2.0,
        "base_target",
        "test",
        TransformDistribution(components),
    )


def test_listener_lookup_preserves_direct_v2_mixture_and_coupling():
    record = _record()
    listener = ProbTfListener()
    listener.receive_record(record)

    resolved = lookup_direct_record(listener, "base_link", "target", stamp=2.0)
    assert resolved is record
    assert len(resolved.distribution.components) == 2
    np.testing.assert_array_equal(
        resolved.distribution.components[0].translation.rotation_coupling,
        record.distribution.components[0].translation.rotation_coupling,
    )

    listener_moments = listener.lookup_point_moments(
        "base_link",
        "target",
        [0.0, 0.0, 0.0],
        stamp=2.0,
    )
    direct_moments = distribution_point_moments(record)
    np.testing.assert_allclose(listener_moments.value.mean, direct_moments.mean)
    np.testing.assert_allclose(listener_moments.value.covariance, direct_moments.covariance)


class _Wrapper:
    def __init__(self):
        self.header = None
        self.object_id = ""
        self.hand_id = ""
        self.grasp_id = ""
        self.transform = None


class _TargetWrapper(_Wrapper):
    def __init__(self):
        super().__init__()
        self.weight = 0.0
        self.approach_axis = _Vector3()
        self.finger_axis = _Vector3()


class _TargetArrayWrapper:
    def __init__(self):
        self.header = None
        self.object_id = ""
        self.targets = []


class _Stamp(float):
    def to_sec(self):
        return float(self)


class _Header:
    def __init__(self):
        self.frame_id = ""
        self.stamp = _Stamp(0.0)


class _Vector3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quaternion:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class _Transform:
    def __init__(self):
        self.translation = _Vector3()
        self.rotation = _Quaternion()


class _Approximation:
    def __init__(self):
        self.kind = 0
        self.lossy = False
        self.detail = ""
        self.source = ""
        self.has_error_bound = False
        self.error_bound = 0.0


class _Provenance:
    def __init__(self):
        self.source_ids = []
        self.derived_from_edge_ids = []
        self.method = ""
        self.detail = ""


class _Orientation:
    def __init__(self):
        self.kind = 0
        self.inverse_concentration = 1.0
        self.shape_upper_wxyz = [0.0] * 10
        self.reference_quaternion = _Quaternion()


class _Translation:
    def __init__(self):
        self.mean_at_reference = _Vector3()
        self.residual_covariance_upper = [0.0] * 6
        self.rotation_coupling = [0.0] * 27


class _Component:
    def __init__(self):
        self.component_id = ""
        self.weight = 0.0
        self.orientation = _Orientation()
        self.translation = _Translation()
        self.approximation = _Approximation()
        self.provenance = _Provenance()


class _Stamped:
    def __init__(self):
        self.header = _Header()
        self.child_frame_id = ""
        self.edge_id = ""
        self.authority = ""
        self.is_static = False
        self.representative_kind = 0
        self.representative = _Transform()
        self.components = []
        self.approximation = _Approximation()
        self.provenance = _Provenance()


class _Array:
    def __init__(self):
        self.header = _Header()
        self.transforms = []


_V2_TYPES = V2MessageTypes(
    _Orientation,
    _Translation,
    _Component,
    _Stamped,
    _Array,
    _Approximation,
    _Provenance,
)


def test_object_wrapper_roundtrip_keeps_all_v2_components():
    record = _record()
    message_types = SymawareMessageTypes(
        _Wrapper,
        _Wrapper,
        _Wrapper,
        _Wrapper,
        _Wrapper,
        _V2_TYPES,
    )
    message = object_belief_to_msg(
        record,
        "object",
        message_types=message_types,
        time_factory=_Stamp,
    )
    restored = record_from_app_message(message)

    assert message.object_id == "object"
    assert [component.component_id for component in restored.distribution.components] == [
        "first",
        "second",
    ]
    assert [component.raw_weight for component in restored.distribution.components] == [1.0, 3.0]
    np.testing.assert_allclose(
        restored.distribution.components[0].translation.rotation_coupling,
        record.distribution.components[0].translation.rotation_coupling,
    )


def test_all_application_wrappers_keep_nested_v2_record():
    record = _record()
    candidate = GraspCandidate(
        "target",
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    message_types = SymawareMessageTypes(
        _Wrapper,
        _Wrapper,
        _TargetWrapper,
        _TargetArrayWrapper,
        _Wrapper,
        _V2_TYPES,
    )

    hand = hand_belief_to_msg(record, "hand", message_types, _Stamp)
    targets = grasp_targets_to_msg([record], [candidate], "object", message_types, _Stamp)
    selected = selected_target_to_msg(record, "object", "target", message_types, _Stamp)

    assert hand.hand_id == "hand"
    assert targets.object_id == "object"
    assert targets.targets[0].weight == 1.0
    assert selected.grasp_id == "target"
    for message in (hand, targets.targets[0], selected):
        restored = record_from_app_message(message)
        assert len(restored.distribution.components) == 2
        np.testing.assert_allclose(
            restored.distribution.components[0].translation.rotation_coupling,
            record.distribution.components[0].translation.rotation_coupling,
        )


def test_ik_component_models_keep_mixture_members_and_coupled_point_moments():
    record = _record()
    models = _component_cost_models(record, integration_steps=30)

    assert len(models) == 2
    assert [model.weight for model in models] == pytest.approx([0.25, 0.75])
    assert np.linalg.norm(models[0].mean - np.array([0.2, 0.0, 0.0])) > 1e-6


def test_pointwise_ik_cost_accepts_native_mixture_components():
    models = _component_cost_models(_record(), integration_steps=20)
    solver = SymmetryAwareIKSolver(bingham_integration_steps=20)
    result = solver.evaluate_cost(
        np.zeros(6),
        np.zeros(6),
        models,
        SymmetryAwareIKSolver.METHOD_POINTWISE,
    )
    assert np.isfinite(result["total_cost"])
    assert np.isfinite(result["position_cost"])
    assert np.isfinite(result["orientation_cost"])


def test_ik_rejects_zero_mass_component_set():
    record = _record()
    components = tuple(
        TransformComponent(
            component.component_id,
            0.0,
            component.orientation,
            component.translation,
            component.provenance,
            component.approximation,
        )
        for component in record.distribution.components
    )
    zero_mass = TransformDistributionStamped(
        record.parent_frame_id,
        record.child_frame_id,
        record.stamp,
        record.edge_id,
        record.authority,
        TransformDistribution(components),
    )
    with pytest.raises(ValueError, match="zero_mass"):
        _component_cost_models(zero_mass, integration_steps=20)


def test_bhattacharyya_ik_rejects_non_finite_orientation_components():
    target = _record(BinghamOrientation.uniform())
    solver = SymmetryAwareIKSolver(
        hand_belief_model=object(),
        bingham_integration_steps=20,
        max_iterations=1,
        restarts=0,
    )
    with pytest.raises(ValueError, match="finite Bingham"):
        solver.solve_single_target(
            target,
            np.zeros(6),
            SymmetryAwareIKSolver.METHOD_BHATTACHARYYA,
        )


def test_native_bingham_fit_does_not_mutate_samples_or_weights():
    samples = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.1, 0.0, 0.0],
            [0.99, -0.1, 0.0, 0.0],
            [0.98, 0.0, 0.2, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 2.0, 1.0], dtype=float)
    expected_samples = samples.copy()
    expected_weights = weights.copy()

    fit_bingham_from_quaternion_samples(
        samples,
        weights=weights,
        integration_steps=20,
        max_iterations=2,
    )

    np.testing.assert_array_equal(samples, expected_samples)
    np.testing.assert_array_equal(weights, expected_weights)


def test_hand_belief_marks_sample_fit_and_independence_assumption_lossy():
    robot = ToyArm6DOF()
    model = EndEffectorBeliefModel(
        robot,
        np.full(robot.dof, 0.03),
        sample_count=8,
        bingham_integration_steps=20,
        bingham_fit_max_iterations=2,
    )
    record = model.estimate_record(np.zeros(robot.dof), stamp=1.0)
    component = record.distribution.components[0]

    assert record.approximation.kind is ApproximationKind.MOMENT_SUMMARY
    assert record.approximation.lossy
    assert component.approximation == record.approximation
    assert "cross-coupling" in record.approximation.detail


def test_producer_supplied_approximation_survives_app_wire_roundtrip():
    approximation = ApproximationInfo(
        ApproximationKind.PRODUCER_SUPPLIED,
        False,
        "configured object law",
        "object_pose_node",
    )
    record = make_transform_record(
        "base_link",
        "object",
        1.0,
        "object_edge",
        "test",
        np.zeros(3),
        np.eye(3) * 1e-4,
        np.diag([0.0, -1.0, -2.0, -3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        "configured_object",
        approximation=approximation,
    )
    types = SymawareMessageTypes(
        _Wrapper,
        _Wrapper,
        _TargetWrapper,
        _TargetArrayWrapper,
        _Wrapper,
        _V2_TYPES,
    )
    restored = record_from_app_message(
        object_belief_to_msg(record, "object", types, _Stamp)
    )

    assert restored.approximation == approximation
    assert restored.distribution.components[0].approximation == approximation
