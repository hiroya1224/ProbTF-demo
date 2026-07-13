from probtf.distributions import (
    TransformDistributionStamped,
    compose_with_deterministic_right,
)
from probtf.geometry import DeterministicTransform


def compose_grasp_targets(
    object_transform,
    candidates,
    authority="symaware_grasp_targets",
):
    """Compose native v2 object pose components with fixed grasp offsets."""

    if not isinstance(object_transform, TransformDistributionStamped):
        raise TypeError("object_transform must be a TransformDistributionStamped.")

    targets = []
    for candidate in candidates:
        targets.append(
            compose_with_deterministic_right(
                object_transform,
                DeterministicTransform(
                    candidate.object_to_grasp_position,
                    candidate.object_to_grasp_orientation_wxyz,
                ),
                child_frame_id=candidate.grasp_id,
                edge_id="{}__to__{}".format(
                    object_transform.parent_frame_id,
                    candidate.grasp_id,
                ),
                authority=authority,
            )
        )
    return targets
