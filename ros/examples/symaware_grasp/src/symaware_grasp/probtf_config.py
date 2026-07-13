"""Load the demo arm's uncertain joints as native ProbTF v2 records."""

from collections.abc import Mapping
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
    quat_mul,
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


def revolute_joint_names(mapping):
    return tuple(
        str(edge["joint"]).strip()
        for edge in mapping.get("edges", ())
        if str(edge["type"]).strip().lower() == "revolute"
    )


def _normalize_joint_positions(mapping, joint_positions):
    joint_names = revolute_joint_names(mapping)
    if joint_positions is None:
        return {joint_name: 0.0 for joint_name in joint_names}
    if isinstance(joint_positions, Mapping):
        unknown = set(joint_positions) - set(joint_names)
        if unknown:
            raise ValueError("Unknown configured joint positions: {}.".format(sorted(unknown)))
        values = {
            joint_name: float(joint_positions.get(joint_name, 0.0))
            for joint_name in joint_names
        }
    else:
        positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        if positions.shape != (len(joint_names),):
            raise ValueError(
                "joint_positions must contain {} revolute joint values.".format(len(joint_names))
            )
        values = dict(zip(joint_names, positions.tolist()))
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("joint_positions must be finite.")
    return values


def _orientation(edge, joint_position=0.0):
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
    configured_mode = edge.get("mode_quaternion", nominal)
    mode = quat_mul(configured_mode, axis_angle_to_quat(edge["axis"], joint_position))
    return BinghamOrientation.from_parameter_matrix(
        bingham_parameter_from_mode(mode, edge["bingham_eigenvalues"]),
        reference_quaternion_wxyz=mode,
    )


def _record(edge, stamp, authority, source_id, joint_position=0.0, is_static=True):
    edge_id = str(edge["joint"]).strip()
    orientation = _orientation(edge, joint_position)
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
        is_static=bool(is_static),
    )


def config_from_mapping(
    mapping,
    stamp=0.0,
    authority="symaware_grasp_config",
    source_id="config",
    joint_positions=None,
    dynamic_joints=False,
):
    root = str(mapping["root"]).strip()
    frames = tuple(str(frame).strip() for frame in mapping.get("frames", (root,)))
    if not root or any(not frame for frame in frames):
        raise ValueError("root and frame identifiers must not be empty.")
    if root not in frames:
        raise ValueError("Configured frames must contain the root frame.")
    positions = _normalize_joint_positions(mapping, joint_positions)
    records = tuple(
        _record(
            edge,
            float(stamp),
            str(authority),
            str(source_id),
            joint_position=positions.get(str(edge["joint"]).strip(), 0.0),
            is_static=not (
                bool(dynamic_joints) and str(edge["type"]).strip().lower() == "revolute"
            ),
        )
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


def load_prob_tf_mapping(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, Mapping):
        raise ValueError("ProbTF config must contain a mapping at its root.")
    return mapping


def load_prob_tf_config(
    path,
    stamp=0.0,
    authority="symaware_grasp_config",
    joint_positions=None,
    dynamic_joints=False,
):
    config_path = Path(path)
    mapping = load_prob_tf_mapping(config_path)
    return config_from_mapping(
        mapping,
        stamp=stamp,
        authority=authority,
        source_id="config:{}".format(config_path.name),
        joint_positions=joint_positions,
        dynamic_joints=dynamic_joints,
    )
