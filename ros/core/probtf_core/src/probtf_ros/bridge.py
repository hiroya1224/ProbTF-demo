"""Publisher/listener facades for ``/probtf`` and ``/probtf_static``."""

from probtf.graph import ProbTfGraph
from probtf_ros.v2_conversions import (
    V2MessageTypes,
    transform_array_from_msg,
    transform_array_to_msg,
    transform_distribution_from_msg,
    transform_distribution_to_msg,
)


PROBTF_TOPIC = "/probtf"
PROBTF_STATIC_TOPIC = "/probtf_static"


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

    def send_transform(self, record):
        if record.is_static:
            self._static_records[record.edge_id] = record
            message = transform_array_to_msg(
                tuple(self._static_records[key] for key in sorted(self._static_records)),
                self.message_types,
                self.time_factory,
            )
            publisher = self.static_publisher
        else:
            message = transform_distribution_to_msg(
                record,
                self.message_types,
                self.time_factory,
            )
            publisher = self.dynamic_publisher
        publisher.publish(message)
        return message


class ProbTfListener:
    def __init__(self, graph=None):
        self.graph = ProbTfGraph() if graph is None else graph

    def receive_transform(self, message):
        record = transform_distribution_from_msg(message)
        self.graph.insert(record)
        return record

    def receive_array(self, message):
        records = transform_array_from_msg(message)
        for record in records:
            self.graph.insert(record)
        return records
