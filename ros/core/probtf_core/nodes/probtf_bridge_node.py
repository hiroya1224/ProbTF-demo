#!/usr/bin/env python3

"""Runtime wiring for Prob-TF v2 topics and deterministic TF bridges."""

import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from tf2_msgs.msg import TFMessage

from probtf.graph import ProbTfGraph
from probtf_ros.bridge import (
    DEFAULT_MAX_RECORDS_PER_EDGE,
    LatestTfImportBuffer,
    PROBTF_STATIC_TOPIC,
    PROBTF_TOPIC,
    ProbTfBroadcaster,
    ProbTfListener,
    child_frame_matches_prefixes,
    parse_frame_prefixes,
)
from probtf_ros.tf_bridge import ProbTfTfBridge, TfExportPolicy
from probtf_ros.v2_conversions import V2MessageTypes


def _caller_id(message, fallback):
    header = getattr(message, "_connection_header", None) or {}
    return str(header.get("callerid", fallback))


class ProbTfBridgeNode:
    def __init__(self):
        max_records_per_edge = int(
            rospy.get_param("~max_records_per_edge", DEFAULT_MAX_RECORDS_PER_EDGE)
        )
        if max_records_per_edge < 1:
            raise ValueError("~max_records_per_edge must be positive.")
        self.graph = ProbTfGraph(max_records_per_edge=max_records_per_edge)
        self.listener = ProbTfListener(self.graph)
        self.message_types = V2MessageTypes.defaults()
        self.node_authority = rospy.get_name()

        probtf_topic = rospy.get_param("~probtf_topic", PROBTF_TOPIC)
        probtf_static_topic = rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC)
        self.dynamic_publisher = rospy.Publisher(
            probtf_topic,
            ProbabilisticTransformStamped,
            queue_size=100,
        )
        self.static_publisher = rospy.Publisher(
            probtf_static_topic,
            ProbabilisticTransformArray,
            queue_size=1,
            latch=True,
        )
        self.probtf_broadcaster = ProbTfBroadcaster(
            self.dynamic_publisher,
            self.static_publisher,
            self.message_types,
            rospy.Time.from_sec,
        )
        self.tf_bridge = ProbTfTfBridge(self.listener, self.node_authority)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        policy_name = str(rospy.get_param("~tf_export_policy", "exact_only"))
        try:
            self.tf_export_policy = TfExportPolicy(policy_name)
        except ValueError as exc:
            raise ValueError("Unknown ~tf_export_policy '{}'.".format(policy_name)) from exc

        self.import_tf = bool(rospy.get_param("~import_tf", True))
        self.export_tf = bool(rospy.get_param("~export_tf", True))
        self.tf_import_max_rate_hz = float(
            rospy.get_param("~tf_import_max_rate_hz", 0.0)
        )
        if (
            not math.isfinite(self.tf_import_max_rate_hz)
            or self.tf_import_max_rate_hz < 0.0
        ):
            raise ValueError("~tf_import_max_rate_hz must be finite and non-negative.")
        self.tf_import_child_prefixes = parse_frame_prefixes(
            rospy.get_param("~tf_import_child_prefixes", ())
        )
        self.tf_import_buffer = LatestTfImportBuffer()
        self.tf_import_timer = None

        # An import-only bridge publishes records for another process and does
        # not need to retain a duplicate local history.
        self.tf_bridge.store_imports = self.export_tf

        self.dynamic_subscriber = None
        self.static_subscriber = None
        if self.export_tf:
            self.dynamic_subscriber = rospy.Subscriber(
                probtf_topic,
                ProbabilisticTransformStamped,
                self._on_probtf,
                queue_size=100,
            )
            self.static_subscriber = rospy.Subscriber(
                probtf_static_topic,
                ProbabilisticTransformArray,
                self._on_probtf_static,
                queue_size=1,
            )
        self.tf_subscriber = None
        self.tf_static_subscriber = None
        if self.import_tf:
            # Preserve the lossless legacy behaviour when coalescing is
            # disabled.  queue_size=1 is appropriate only for latest-only
            # import, where retaining old socket work would reintroduce lag.
            tf_queue_size = 1 if self.tf_import_max_rate_hz > 0.0 else 100
            self.tf_subscriber = rospy.Subscriber(
                "/tf",
                TFMessage,
                self._on_tf,
                callback_args=False,
                queue_size=tf_queue_size,
            )
            self.tf_static_subscriber = rospy.Subscriber(
                "/tf_static",
                TFMessage,
                self._on_tf,
                callback_args=True,
                queue_size=10,
            )
            if self.tf_import_max_rate_hz > 0.0:
                self.tf_import_timer = rospy.Timer(
                    rospy.Duration.from_sec(1.0 / self.tf_import_max_rate_hz),
                    self._flush_tf_imports,
                )

    def _is_own_message(self, message):
        return _caller_id(message, "") == self.node_authority

    def _export_record(self, record):
        if not self.export_tf:
            return
        try:
            result = self.tf_bridge.export_transform(
                record,
                message_type=TransformStamped,
                time_factory=rospy.Time.from_sec,
                policy=self.tf_export_policy,
            )
        except ValueError as exc:
            rospy.logwarn_throttle(
                5.0,
                "Prob-TF edge '%s' was not exported to TF: %s",
                record.edge_id,
                exc,
            )
            return
        broadcaster = self.tf_static_broadcaster if record.is_static else self.tf_broadcaster
        broadcaster.sendTransform(result.message)

    def _on_probtf(self, message):
        if self._is_own_message(message):
            return
        record = self.listener.receive_transform(message)
        self._export_record(record)

    def _on_probtf_static(self, message):
        if self._is_own_message(message):
            return
        for record in self.listener.receive_array(message):
            self._export_record(record)

    def _on_tf(self, message, is_static):
        authority = _caller_id(message, "unknown_tf_authority")
        if not is_static and self.tf_import_max_rate_hz > 0.0:
            for transform in message.transforms:
                if child_frame_matches_prefixes(
                    transform.child_frame_id,
                    self.tf_import_child_prefixes,
                ):
                    self.tf_import_buffer.put(transform, authority)
            return

        records = []
        for transform in message.transforms:
            if not child_frame_matches_prefixes(
                transform.child_frame_id,
                self.tf_import_child_prefixes,
            ):
                continue
            record = self.tf_bridge.import_transform(transform, authority, is_static)
            if record is not None:
                records.append(record)
        self.probtf_broadcaster.send_transforms(records)

    def _flush_tf_imports(self, event):
        del event
        records = []
        for transform, authority in self.tf_import_buffer.drain():
            record = self.tf_bridge.import_transform(transform, authority, False)
            if record is not None:
                records.append(record)
        self.probtf_broadcaster.send_transforms(records)


def main():
    rospy.init_node("probtf_bridge")
    ProbTfBridgeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
