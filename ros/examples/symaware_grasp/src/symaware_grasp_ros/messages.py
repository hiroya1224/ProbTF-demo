"""Application message wrappers that retain the complete ProbTF v2 payload."""

from dataclasses import dataclass

from probtf_ros.v2_conversions import (
    V2MessageTypes,
    transform_distribution_from_msg,
    transform_distribution_to_msg,
)


@dataclass(frozen=True)
class SymawareMessageTypes:
    object_belief: object
    grasp_target: object
    grasp_target_array: object
    selected_target: object
    v2: object

    @classmethod
    def defaults(cls):
        from symaware_grasp.msg import (
            GraspTarget,
            GraspTargetArray,
            ObjectBelief,
            SelectedGraspTarget,
        )

        return cls(
            ObjectBelief,
            GraspTarget,
            GraspTargetArray,
            SelectedGraspTarget,
            V2MessageTypes.defaults(),
        )


def _types(message_types):
    return SymawareMessageTypes.defaults() if message_types is None else message_types


def _assign_vector3(message, values):
    message.x, message.y, message.z = (float(value) for value in values)


def object_belief_to_msg(record, object_id, message_types=None, time_factory=None):
    types = _types(message_types)
    message = types.object_belief()
    message.object_id = str(object_id)
    message.transform = transform_distribution_to_msg(record, types.v2, time_factory)
    message.header = message.transform.header
    return message


def grasp_targets_to_msg(
    records,
    candidates,
    object_id,
    message_types=None,
    time_factory=None,
):
    records = tuple(records)
    candidates = tuple(candidates)
    if len(records) != len(candidates):
        raise ValueError("records and candidates must have the same length.")
    types = _types(message_types)
    output = types.grasp_target_array()
    output.object_id = str(object_id)
    messages = []
    for record, candidate in zip(records, candidates):
        message = types.grasp_target()
        message.object_id = str(object_id)
        message.grasp_id = candidate.grasp_id
        message.weight = float(candidate.weight)
        _assign_vector3(message.approach_axis, candidate.approach_axis)
        _assign_vector3(message.finger_axis, candidate.finger_axis)
        message.transform = transform_distribution_to_msg(record, types.v2, time_factory)
        messages.append(message)
    output.targets = messages
    if messages:
        output.header = messages[0].transform.header
    return output


def selected_target_to_msg(
    record,
    object_id,
    grasp_id,
    message_types=None,
    time_factory=None,
):
    types = _types(message_types)
    message = types.selected_target()
    message.object_id = str(object_id)
    message.grasp_id = str(grasp_id)
    message.transform = transform_distribution_to_msg(record, types.v2, time_factory)
    message.header = message.transform.header
    return message


def record_from_app_message(message):
    """Deserialize a wrapper without changing any v2 component fields."""

    return transform_distribution_from_msg(message.transform)
