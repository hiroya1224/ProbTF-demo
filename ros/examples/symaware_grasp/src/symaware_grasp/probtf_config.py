"""Load the demo arm's uncertain joints as native ProbTF v2 records."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    OrientationKind,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    axis_angle_to_quat,
    complete_orthonormal_basis,
)
from probtf.graph import ProbTfGraph
from probtf.provenance import ComponentProvenance, TransformProvenance


@dataclass(frozen=True)
class ProbTfConfig:
    root_frame: str
    frames: tuple
    records: tuple

    def build_graph(self):
        graph = ProbTfGraph()
        for record in self.records:
            graph.insert(record)
        return graph


def bingham_parameter_from_mode(mode_wxyz, eigenvalues):
    basis = complete_orthonormal_basis(mode_wxyz)
    values = np.asarray(eigenvalues, dtype=float).reshape(4)
    parameter = basis @ np.diag(values) @ basis.T
    return 0.5 * (parameter + parameter.T)


def _orientation(edge):
    joint_type = str(edge["type"]).strip().lower()
    if joint_type == "fixed":
        quaternion = edge.get("mode_quaternion", [1.0, 0.0, 0.0, 0.0])
        return BinghamOrientation.dirac(quaternion)
    if joint_type != "revolute":
        raise ValueError("Unsupported ProbTF joint type {!r}.".format(edge["type"]))
    nominal = axis_angle_to_quat(
        edge["axis"],
        float(edge.get("nominal_angle", 0.0)),
    )
    mode = edge.get("mode_quaternion", nominal)
    return BinghamOrientation.from_parameter_matrix(
        bingham_parameter_from_mode(mode, edge["bingham_eigenvalues"]),
        reference_quaternion_wxyz=mode,
    )


def _record(edge, stamp, authority, source_id):
    edge_id = str(edge["joint"]).strip()
    orientation = _orientation(edge)
    translation = np.asarray(edge["translation"], dtype=float).reshape(3)
    representative = DeterministicTransform(
        translation,
        orientation.reference_quaternion_wxyz,
    )
    representative_kind = (
        RepresentativeKind.EXACT_MAP
        if orientation.kind is OrientationKind.DIRAC
        else RepresentativeKind.COMPONENT_MODE_APPROXIMATION
    )
    component = TransformComponent(
        component_id="{}:configured".format(edge_id),
        raw_weight=1.0,
        orientation=orientation,
        translation=ConditionalGaussianTranslation(
            translation,
            np.zeros((3, 3)),
            np.zeros((3, 9)),
        ),
        provenance=ComponentProvenance(
            source_ids=(source_id,),
            method="configured_joint_distribution",
        ),
    )
    return TransformDistributionStamped(
        parent_frame_id=edge["parent"],
        child_frame_id=edge["child"],
        stamp=stamp,
        edge_id=edge_id,
        authority=authority,
        distribution=TransformDistribution((component,)),
        representative=representative,
        representative_kind=representative_kind,
        provenance=TransformProvenance(
            source_ids=(source_id,),
            method="configured_joint_distribution",
        ),
        is_static=True,
    )


def config_from_mapping(mapping, stamp=0.0, authority="symaware_grasp_config", source_id="config"):
    root = str(mapping["root"]).strip()
    frames = tuple(str(frame).strip() for frame in mapping.get("frames", (root,)))
    if not root or any(not frame for frame in frames):
        raise ValueError("root and frame identifiers must not be empty.")
    if root not in frames:
        raise ValueError("Configured frames must contain the root frame.")
    records = tuple(
        _record(edge, float(stamp), str(authority), str(source_id))
        for edge in mapping.get("edges", ())
    )
    graph_frames = set((root,))
    for record in records:
        graph_frames.add(record.parent_frame_id)
        graph_frames.add(record.child_frame_id)
    if set(frames) != graph_frames:
        raise ValueError("Configured frame list does not match the v2 edge topology.")
    config = ProbTfConfig(root, frames, records)
    config.build_graph()
    return config


def load_prob_tf_config(path, stamp=0.0, authority="symaware_grasp_config"):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    return config_from_mapping(
        mapping,
        stamp=stamp,
        authority=authority,
        source_id="config:{}".format(config_path.name),
    )
