"""Publisher/listener facades for ``/probtf`` and ``/probtf_static``."""

import math
import time
from threading import Condition, RLock

from probtf.graph import ProbTfGraph
from probtf.graph.status import ProbTfGraphError
from probtf.temporal import TemporalPolicy
from probtf_ros.v2_conversions import (
    V2MessageTypes,
    transform_array_from_msg,
    transform_array_to_msg,
    transform_distribution_from_msg,
    transform_distribution_to_msg,
)


PROBTF_TOPIC = "/probtf"
PROBTF_STATIC_TOPIC = "/probtf_static"
DEFAULT_MAX_RECORDS_PER_EDGE = 1000


def parse_frame_prefixes(value):
    if value is None:
        return ()
    entries = value.split(",") if isinstance(value, str) else tuple(value)
    return tuple(
        prefix
        for prefix in (str(entry).strip().strip("/") for entry in entries)
        if prefix
    )


def child_frame_matches_prefixes(child_frame_id, prefixes):
    prefixes = tuple(prefixes)
    if not prefixes:
        return True
    child = str(child_frame_id).strip().strip("/")
    return any(child == prefix or child.startswith(prefix + "/") for prefix in prefixes)


class ProbTfBroadcaster:
    def __init__(
        self,
        dynamic_publisher,
        static_publisher,
        message_types=None,
        time_factory=None,
    ):
        self.dynamic_publisher = dynamic_publisher
        self.static_publisher = static_publisher
        self.message_types = V2MessageTypes.defaults() if message_types is None else message_types
        self.time_factory = time_factory
        self._static_records = {}
        self._lock = RLock()

    def send_transform(self, record):
        return self.send_transforms((record,))[0]

    def send_transforms(self, records):
        """Publish many records, emitting at most one full static-set message."""

        records = tuple(records)
        if not records:
            return ()

        with self._lock:
            static_records = tuple(record for record in records if record.is_static)
            dynamic_records = tuple(record for record in records if not record.is_static)
            messages = []

            if static_records:
                for record in static_records:
                    self._static_records[record.edge_id] = record
                message = transform_array_to_msg(
                    tuple(self._static_records[key] for key in sorted(self._static_records)),
                    self.message_types,
                    self.time_factory,
                )
                self.static_publisher.publish(message)
                messages.append(message)

            for record in dynamic_records:
                message = transform_distribution_to_msg(
                    record,
                    self.message_types,
                    self.time_factory,
                )
                self.dynamic_publisher.publish(message)
                messages.append(message)

            return tuple(messages)


class ProbTfListener:
    def __init__(self, graph=None):
        self.graph = ProbTfGraph() if graph is None else graph
        self._condition = Condition()

    @property
    def frames(self):
        return self.graph.frames

    @property
    def edges(self):
        return self.graph.edges

    def _notify_update(self):
        with self._condition:
            self._condition.notify_all()

    def receive_record(self, record):
        self.graph.insert(record)
        self._notify_update()
        return record

    def receive_records(self, records):
        records = tuple(records)
        inserted = False
        try:
            for record in records:
                self.graph.insert(record)
                inserted = True
        finally:
            if inserted:
                self._notify_update()
        return records

    def receive_transform(self, message):
        record = transform_distribution_from_msg(message)
        return self.receive_record(record)

    def receive_array(self, message):
        records = transform_array_from_msg(message)
        return self.receive_records(records)

    def lookup_path(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
    ):
        return self.graph.lookup_path(
            target_frame,
            source_frame,
            stamp,
            policy,
            tolerance,
            max_age,
        )

    def lookup_kernel(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
    ):
        return self.graph.lookup_kernel(
            target_frame,
            source_frame,
            stamp,
            policy,
            tolerance,
            max_age,
        )

    def lookup_point_moments(
        self,
        target_frame,
        source_frame,
        point,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
        evaluator=None,
    ):
        from probtf.kernels import KernelEvaluator, KernelRepresentation

        kernel = self.lookup_kernel(
            target_frame,
            source_frame,
            stamp,
            policy,
            tolerance,
            max_age,
        )
        evaluator = KernelEvaluator() if evaluator is None else evaluator
        return evaluator.apply_to_point(
            kernel,
            point,
            KernelRepresentation.MOMENTS,
        )

    def can_lookup(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
    ):
        try:
            self.lookup_path(
                target_frame,
                source_frame,
                stamp,
                policy,
                tolerance,
                max_age,
            )
        except ProbTfGraphError:
            return False
        return True

    def wait_for_lookup(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
        timeout=None,
    ):
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0.0:
                raise ValueError("timeout must be finite and non-negative or None.")
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self.can_lookup(
                target_frame,
                source_frame,
                stamp,
                policy,
                tolerance,
                max_age,
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
        return True


class RosProbTfListener(ProbTfListener):
    """A ROS-owned v2 topic listener backed by a bounded local graph."""

    def __init__(
        self,
        graph=None,
        dynamic_topic=PROBTF_TOPIC,
        static_topic=PROBTF_STATIC_TOPIC,
        max_records_per_edge=DEFAULT_MAX_RECORDS_PER_EDGE,
        dynamic_queue_size=100,
        static_queue_size=1,
        subscriber_factory=None,
        dynamic_message_type=None,
        static_message_type=None,
    ):
        if graph is None:
            graph = ProbTfGraph(max_records_per_edge=max_records_per_edge)
        super().__init__(graph)

        dynamic_queue_size = _positive_integer(dynamic_queue_size, "dynamic_queue_size")
        static_queue_size = _positive_integer(static_queue_size, "static_queue_size")
        dynamic_topic = _topic_name(dynamic_topic, "dynamic_topic")
        static_topic = _topic_name(static_topic, "static_topic")

        if subscriber_factory is None:
            import rospy

            subscriber_factory = rospy.Subscriber
        if dynamic_message_type is None or static_message_type is None:
            types = V2MessageTypes.defaults()
            dynamic_message_type = types.stamped if dynamic_message_type is None else dynamic_message_type
            static_message_type = types.array if static_message_type is None else static_message_type

        self.dynamic_subscriber = subscriber_factory(
            dynamic_topic,
            dynamic_message_type,
            self._receive_dynamic,
            queue_size=dynamic_queue_size,
        )
        try:
            self.static_subscriber = subscriber_factory(
                static_topic,
                static_message_type,
                self._receive_static,
                queue_size=static_queue_size,
            )
        except Exception:
            self.dynamic_subscriber.unregister()
            self.dynamic_subscriber = None
            raise

    def _receive_dynamic(self, message):
        record = transform_distribution_from_msg(message)
        if record.is_static:
            raise ValueError("The dynamic Prob-TF topic cannot contain a static record.")
        return self.receive_record(record)

    def _receive_static(self, message):
        records = transform_array_from_msg(message)
        if any(not record.is_static for record in records):
            raise ValueError("The static Prob-TF topic can contain only static records.")
        return self.receive_records(records)

    def unregister(self):
        for name in ("dynamic_subscriber", "static_subscriber"):
            subscriber = getattr(self, name, None)
            if subscriber is not None:
                subscriber.unregister()
                setattr(self, name, None)


def _positive_integer(value, name):
    result = int(value)
    if result < 1 or result != value:
        raise ValueError("{} must be a positive integer.".format(name))
    return result


def _topic_name(value, name):
    result = str(value).strip()
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result
