from dataclasses import replace
from threading import Thread

import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.graph import ProbTfGraph
from probtf.geometry import DeterministicTransform
from probtf.kernels import KernelRepresentation
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)
from probtf.temporal import TemporalPolicy

from probtf_ros.bridge import (
    ProbTfBroadcaster,
    ProbTfListener,
    RosProbTfListener,
    child_frame_matches_prefixes,
    parse_frame_prefixes,
)
from probtf_ros.tf_bridge import (
    ProbTfTfBridge,
    TfExportPolicy,
    deterministic_tf_to_record,
    record_to_deterministic_tf,
)
from probtf_ros.v2_conversions import (
    V2MessageTypes,
    transform_array_from_msg,
    transform_array_to_msg,
    transform_distribution_from_msg,
    transform_distribution_to_msg,
)


class Stamp(float):
    def to_sec(self):
        return float(self)


class Header:
    def __init__(self):
        self.frame_id = ""
        self.stamp = Stamp(0.0)


class Vector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Quaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class Transform:
    def __init__(self):
        self.translation = Vector3()
        self.rotation = Quaternion()


class ApproximationMessage:
    def __init__(self):
        self.kind = 0
        self.lossy = False
        self.detail = ""
        self.source = ""
        self.has_error_bound = False
        self.error_bound = 0.0


class ProvenanceMessage:
    def __init__(self):
        self.source_ids = []
        self.derived_from_edge_ids = []
        self.method = ""
        self.detail = ""


class OrientationMessage:
    def __init__(self):
        self.kind = 0
        self.inverse_concentration = 1.0
        self.shape_upper_wxyz = [0.0] * 10
        self.reference_quaternion = Quaternion()


class TranslationMessage:
    def __init__(self):
        self.mean_at_reference = Vector3()
        self.residual_covariance_upper = [0.0] * 6
        self.rotation_coupling = [0.0] * 27


class ComponentMessage:
    def __init__(self):
        self.component_id = ""
        self.weight = 0.0
        self.orientation = OrientationMessage()
        self.translation = TranslationMessage()
        self.approximation = ApproximationMessage()
        self.provenance = ProvenanceMessage()


class StampedMessage:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.edge_id = ""
        self.authority = ""
        self.is_static = False
        self.representative_kind = 0
        self.representative = Transform()
        self.components = []
        self.approximation = ApproximationMessage()
        self.provenance = ProvenanceMessage()


class ArrayMessage:
    def __init__(self):
        self.header = Header()
        self.transforms = []


class TransformStampedMessage:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.transform = Transform()


MESSAGE_TYPES = V2MessageTypes(
    OrientationMessage,
    TranslationMessage,
    ComponentMessage,
    StampedMessage,
    ArrayMessage,
    ApproximationMessage,
    ProvenanceMessage,
)


def _component(component_id, weight, orientation, coupling_scale=0.0):
    return TransformComponent(
        component_id,
        weight,
        orientation,
        ConditionalGaussianTranslation(
            np.array([1.0, 2.0, 3.0]),
            np.array([[0.1, 0.01, 0.02], [0.01, 0.2, 0.03], [0.02, 0.03, 0.3]]),
            coupling_scale * np.arange(27, dtype=float).reshape(3, 9),
        ),
        ComponentProvenance(
            source_ids=("camera",),
            derived_from_edge_ids=("calibration",),
            method="producer",
        ),
        ApproximationInfo(
            ApproximationKind.PRODUCER_SUPPLIED,
            False,
            "producer model",
            "test",
            0.05,
        ),
    )


def _mixture_record(is_static=True):
    finite = BinghamOrientation.from_parameter_matrix(
        np.diag([3.0, 1.0, -1.0, -3.0]),
        [1.0, 0.0, 0.0, 0.0],
    )
    components = (
        _component("finite", 2.0, finite, 0.01),
        _component("dirac", -1.0, BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0])),
        _component("uniform", 4.0, BinghamOrientation.uniform()),
    )
    representative = DeterministicTransform(
        components[1].translation.mean_at_reference,
        components[1].orientation.mode_wxyz,
    )
    return TransformDistributionStamped(
        "world",
        "tool",
        12.5,
        "world_tool",
        "producer",
        TransformDistribution(components),
        representative=representative,
        representative_kind=RepresentativeKind.PRODUCER_SUPPLIED,
        provenance=TransformProvenance(
            source_ids=("producer",),
            derived_from_edge_ids=("raw_edge",),
            method="mixture_producer",
        ),
        is_static=is_static,
        approximation=ApproximationInfo(ApproximationKind.PRODUCER_SUPPLIED),
    )


def _assert_record_equal(actual, expected):
    assert actual.parent_frame_id == expected.parent_frame_id
    assert actual.child_frame_id == expected.child_frame_id
    assert actual.edge_id == expected.edge_id
    assert actual.authority == expected.authority
    assert actual.stamp == expected.stamp
    assert actual.is_static == expected.is_static
    assert actual.representative_kind is expected.representative_kind
    assert actual.approximation == expected.approximation
    assert actual.provenance == expected.provenance
    assert len(actual.distribution.components) == len(expected.distribution.components)
    for left, right in zip(actual.distribution.components, expected.distribution.components):
        assert left.component_id == right.component_id
        assert left.raw_weight == right.raw_weight
        assert left.orientation.kind is right.orientation.kind
        assert left.orientation.inverse_concentration == right.orientation.inverse_concentration
        np.testing.assert_allclose(left.orientation.shape_matrix, right.orientation.shape_matrix)
        np.testing.assert_allclose(
            left.orientation.reference_quaternion_wxyz,
            right.orientation.reference_quaternion_wxyz,
        )
        np.testing.assert_allclose(left.translation.mean_at_reference, right.translation.mean_at_reference)
        np.testing.assert_allclose(
            left.translation.residual_covariance,
            right.translation.residual_covariance,
        )
        np.testing.assert_allclose(left.translation.rotation_coupling, right.translation.rotation_coupling)
        assert left.approximation == right.approximation
        assert left.provenance == right.provenance


def test_v2_round_trip_preserves_joint_components_packing_and_metadata():
    record = _mixture_record()
    message = transform_distribution_to_msg(record, MESSAGE_TYPES, Stamp)

    assert message.header.frame_id == "world"
    assert not hasattr(message, "parent_frame_id")
    assert message.components[0].weight == 2.0
    assert message.components[1].weight == -1.0
    assert message.components[0].orientation.shape_upper_wxyz == pytest.approx(
        [0.75, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, -0.75]
    )
    assert message.components[0].translation.residual_covariance_upper == pytest.approx(
        [0.1, 0.01, 0.02, 0.2, 0.03, 0.3]
    )
    assert message.components[0].translation.rotation_coupling == pytest.approx(
        (0.01 * np.arange(27)).tolist()
    )
    assert message.components[0].orientation.reference_quaternion.w == 1.0

    restored = transform_distribution_from_msg(message)
    _assert_record_equal(restored, record)
    np.testing.assert_allclose(restored.representative.translation, record.representative.translation)


def test_v2_array_and_broadcaster_listener_route_static_and_dynamic_records():
    static_record = _mixture_record(is_static=True)
    dynamic_record = _mixture_record(is_static=False)
    array_message = transform_array_to_msg(
        (static_record, dynamic_record),
        MESSAGE_TYPES,
        Stamp,
    )
    restored = transform_array_from_msg(array_message)
    assert [record.is_static for record in restored] == [True, False]

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    dynamic_publisher = Publisher()
    static_publisher = Publisher()
    broadcaster = ProbTfBroadcaster(
        dynamic_publisher,
        static_publisher,
        MESSAGE_TYPES,
        Stamp,
    )
    broadcaster.send_transform(static_record)
    broadcaster.send_transform(
        replace(static_record, child_frame_id="camera", edge_id="world_camera")
    )
    broadcaster.send_transform(dynamic_record)
    assert len(static_publisher.messages) == 2
    assert [item.edge_id for item in static_publisher.messages[-1].transforms] == [
        "world_camera",
        "world_tool",
    ]
    assert len(dynamic_publisher.messages) == 1

    listener = ProbTfListener(ProbTfGraph())
    inserted = listener.receive_transform(dynamic_publisher.messages[0])
    assert listener.graph.edge_buffer(inserted.edge_id).latest_stamp == 12.5
    static_listener = ProbTfListener(ProbTfGraph())
    assert len(static_listener.receive_array(static_publisher.messages[-1])) == 2


def test_broadcaster_bulk_send_publishes_one_complete_static_set():
    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    dynamic_publisher = Publisher()
    static_publisher = Publisher()
    broadcaster = ProbTfBroadcaster(
        dynamic_publisher,
        static_publisher,
        MESSAGE_TYPES,
        Stamp,
    )
    world_tool = _mixture_record(is_static=True)
    world_camera = replace(
        world_tool,
        child_frame_id="camera",
        edge_id="world_camera",
    )

    messages = broadcaster.send_transforms((world_tool, world_camera))

    assert len(messages) == 1
    assert len(static_publisher.messages) == 1
    assert dynamic_publisher.messages == []
    assert [item.edge_id for item in messages[0].transforms] == [
        "world_camera",
        "world_tool",
    ]
    assert broadcaster.send_transforms(()) == ()


def test_listener_delegates_lookup_point_moments_and_condition_waits():
    listener = ProbTfListener(ProbTfGraph())
    wait_results = []

    assert not listener.can_lookup("world", "tool", 12.5)
    waiter = Thread(
        target=lambda: wait_results.append(
            listener.wait_for_lookup("world", "tool", 12.5, timeout=1.0)
        )
    )
    waiter.start()
    listener.receive_transform(
        transform_distribution_to_msg(
            _mixture_record(is_static=False),
            MESSAGE_TYPES,
            Stamp,
        )
    )
    waiter.join(1.0)

    assert wait_results == [True]
    assert listener.can_lookup("world", "tool", 12.5)
    assert listener.lookup_path("world", "tool", 12.5).target_frame == "world"
    assert listener.lookup_kernel("world", "tool", 12.5).path.source_frame == "tool"
    moments = listener.lookup_point_moments("world", "tool", [0.1, 0.0, 0.0], 12.5)
    assert moments.status is DistributionStatus.OK
    assert moments.representation is KernelRepresentation.MOMENTS
    assert moments.value.mean.shape == (3,)
    assert listener.frames == frozenset(("world", "tool"))
    assert [edge.edge_id for edge in listener.edges] == ["world_tool"]
    assert not listener.wait_for_lookup(
        "world",
        "missing",
        policy=TemporalPolicy.LATEST,
        timeout=0.01,
    )
    with pytest.raises(ValueError, match="timeout"):
        listener.wait_for_lookup("world", "tool", 12.5, timeout=float("nan"))


def test_ros_listener_owns_subscribers_validates_channels_and_bounds_history():
    class Subscriber:
        instances = []

        def __init__(self, topic, message_type, callback, queue_size):
            self.topic = topic
            self.message_type = message_type
            self.callback = callback
            self.queue_size = queue_size
            self.unregister_count = 0
            self.instances.append(self)

        def unregister(self):
            self.unregister_count += 1

    Subscriber.instances = []
    listener = RosProbTfListener(
        dynamic_topic="/robot/probtf",
        static_topic="/robot/probtf_static",
        max_records_per_edge=2,
        dynamic_queue_size=7,
        static_queue_size=3,
        subscriber_factory=Subscriber,
        dynamic_message_type=StampedMessage,
        static_message_type=ArrayMessage,
    )
    dynamic_subscriber, static_subscriber = Subscriber.instances

    assert dynamic_subscriber.topic == "/robot/probtf"
    assert dynamic_subscriber.queue_size == 7
    assert static_subscriber.topic == "/robot/probtf_static"
    assert static_subscriber.queue_size == 3
    assert listener.graph.max_records_per_edge == 2

    static_record = replace(
        _mixture_record(is_static=True),
        child_frame_id="camera",
        edge_id="world_camera",
    )
    dynamic_record = _mixture_record(is_static=False)
    with pytest.raises(ValueError, match="dynamic Prob-TF topic"):
        dynamic_subscriber.callback(
            transform_distribution_to_msg(static_record, MESSAGE_TYPES, Stamp)
        )
    with pytest.raises(ValueError, match="static Prob-TF topic"):
        static_subscriber.callback(
            transform_array_to_msg((dynamic_record,), MESSAGE_TYPES, Stamp)
        )
    assert listener.frames == frozenset()

    for stamp in (12.5, 13.5, 14.5):
        dynamic_subscriber.callback(
            transform_distribution_to_msg(
                replace(dynamic_record, stamp=stamp),
                MESSAGE_TYPES,
                Stamp,
            )
        )
    static_subscriber.callback(
        transform_array_to_msg((static_record,), MESSAGE_TYPES, Stamp)
    )

    assert [
        record.stamp for record in listener.graph.edge_buffer("world_tool").records
    ] == [13.5, 14.5]
    assert listener.frames == frozenset(("world", "tool", "camera"))

    listener.unregister()
    listener.unregister()
    assert dynamic_subscriber.unregister_count == 1
    assert static_subscriber.unregister_count == 1


def test_tf_import_child_prefix_filter_normalizes_names_and_matches_subtrees():
    prefixes = parse_frame_prefixes(" /ref,cmd/, ,equil ")

    assert prefixes == ("ref", "cmd", "equil")
    assert child_frame_matches_prefixes("/ref", prefixes)
    assert child_frame_matches_prefixes("ref/tool0", prefixes)
    assert child_frame_matches_prefixes("/cmd/link3", prefixes)
    assert not child_frame_matches_prefixes("reference/link3", prefixes)
    assert child_frame_matches_prefixes("anything", ())


def test_tf_import_export_is_exact_dirac_and_bridge_prevents_loops():
    tf_message = TransformStampedMessage()
    tf_message.header.frame_id = "/world"
    tf_message.header.stamp = Stamp(3.0)
    tf_message.child_frame_id = "/camera"
    tf_message.transform.translation.x = 1.0
    tf_message.transform.translation.y = 2.0
    tf_message.transform.translation.z = 3.0
    tf_message.transform.rotation.z = np.sqrt(0.5)
    tf_message.transform.rotation.w = np.sqrt(0.5)

    record = deterministic_tf_to_record(tf_message, "/tf_source", is_static=True)
    assert record.distribution.deterministic_transform() is not None
    assert record.representative_kind is RepresentativeKind.EXACT_MAP
    exported = record_to_deterministic_tf(
        record,
        TransformStampedMessage,
        Stamp,
    )
    assert not exported.approximation.lossy
    assert exported.message.header.frame_id == "world"
    assert exported.message.child_frame_id == "camera"

    listener = ProbTfListener(ProbTfGraph())
    bridge = ProbTfTfBridge(listener, own_authority="bridge")
    bridge_export = bridge.export_transform(
        record,
        message_type=TransformStampedMessage,
        time_factory=Stamp,
    )
    assert bridge.import_transform(bridge_export.message, "rosbag_replay", True) is None
    assert bridge.import_transform(tf_message, "bridge", True) is None
    tf_message.header.stamp = Stamp(4.0)
    imported = bridge.import_transform(tf_message, "external", True)
    assert imported is not None


def test_tf_import_can_skip_duplicate_bridge_history_for_forward_only_use():
    tf_message = TransformStampedMessage()
    tf_message.header.frame_id = "world"
    tf_message.header.stamp = Stamp(3.0)
    tf_message.child_frame_id = "camera"

    listener = ProbTfListener(ProbTfGraph())
    bridge = ProbTfTfBridge(listener, store_imports=False)

    imported = bridge.import_transform(tf_message, "external")

    assert imported is not None
    assert listener.frames == frozenset()


def test_tf_export_requires_explicit_stochastic_projection_policy():
    record = _mixture_record(is_static=False)
    with pytest.raises(ValueError, match="explicit representative"):
        record_to_deterministic_tf(
            record,
            TransformStampedMessage,
            Stamp,
            TfExportPolicy.EXACT_ONLY,
        )
    projected = record_to_deterministic_tf(
        record,
        TransformStampedMessage,
        Stamp,
        TfExportPolicy.HIGHEST_WEIGHT_COMPONENT_MODE,
    )
    assert projected.approximation.kind is ApproximationKind.REPRESENTATIVE_PROJECTION
    assert projected.approximation.lossy
