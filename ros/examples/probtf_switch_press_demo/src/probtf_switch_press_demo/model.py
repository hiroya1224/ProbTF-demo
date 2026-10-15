"""Direct local transform laws and their deterministic, full-path moments.

The primitive laws are configuration inputs, not equilibrium hypotheses.  Each
joint's configured right action is part of that input family.  Independent
primitives are composed before making the requested final Gaussian × Bingham
comparison approximation.  No point samples enter this module.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from probtf.bingham import (
    bingham_second_moment, canonical_bingham_parameter, match_bingham_to_second_moment,
    quaternion_product_second_moment,
)
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, OrientationKind,
    RepresentativeKind, TransformComponent, TransformDistribution,
    TransformDistributionStamped,
)
from probtf.distributions.validation import immutable_array, immutable_symmetric_matrix
from probtf.geometry import (
    DeterministicTransform, axis_angle_to_quat, quat_mul, quat_right_matrix,
    quat_to_rotmat, rotation_action_matrix, rotation_vector_from_quaternion,
    rotmat_to_quat,
)
from probtf.probability.moments import (
    PointMomentSummary, RotationVectorMoments, expected_rotated_covariance,
    mixture_point_moments, rotation_vector_moments,
)
from probtf.provenance import (
    ApproximationInfo, ApproximationKind, ComponentProvenance, TransformProvenance,
)
from symaware_grasp.grasp_collision import BottleCollisionModel
from symaware_grasp.probtf_config import bingham_parameter_from_mode


BODY_FRAMES = tuple("link_{}".format(i) for i in range(1, 7)) + ("tool0", "rod_tip")
COMPARISON_METHOD = "indet_ik_product_gaussian_bingham"


def _frozen_metadata(metadata):
    return MappingProxyType(dict(metadata))


def _metadata_detail(metadata):
    return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))


def _comparison_parameter(orientation, parameter):
    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("comparison orientation must be a native BinghamOrientation")
    if orientation.kind is OrientationKind.DIRAC:
        if parameter is not None:
            raise ValueError("a Dirac comparison orientation has no finite parameter")
        return None
    if parameter is None:
        raise ValueError("a finite comparison orientation needs its parameter matrix")
    native = orientation.backend_parameter_matrix()
    if not np.allclose(canonical_bingham_parameter(parameter), native, rtol=1e-12, atol=1e-9):
        raise ValueError("objective parameter differs from native display orientation")
    # Both consumers use this identical native gauge.
    return immutable_symmetric_matrix(native, 4, "orientation_parameter")


@dataclass(frozen=True)
class PoseSummary:
    mean: np.ndarray
    covariance: np.ndarray
    orientation_parameter: object
    orientation_moment: np.ndarray
    orientation: BinghamOrientation
    metadata: object

    def __post_init__(self):
        object.__setattr__(self, "mean", immutable_array(self.mean, (3,), "mean"))
        object.__setattr__(self, "covariance", immutable_symmetric_matrix(
            self.covariance, 3, "covariance", positive_semidefinite=True))
        object.__setattr__(self, "orientation_moment", immutable_symmetric_matrix(
            self.orientation_moment, 4, "orientation_moment", positive_semidefinite=True))
        object.__setattr__(self, "orientation_parameter", _comparison_parameter(
            self.orientation, self.orientation_parameter))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def to_record(self, child_frame_id, stamp=None):
        """Publish these exact comparison parameters, including source context."""
        return _summary_record(self, child_frame_id, stamp)


@dataclass(frozen=True)
class GoalSummary:
    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    orientation_parameters: tuple
    orientations: tuple
    metadata: object
    branch_summaries: tuple

    def __post_init__(self):
        object.__setattr__(self, "mean", immutable_array(self.mean, (3,), "mean"))
        object.__setattr__(self, "covariance", immutable_symmetric_matrix(
            self.covariance, 3, "covariance", positive_semidefinite=True))
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1 or not len(weights) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("goal weights must be positive and sum to one")
        if not (len(weights) == len(self.orientations) == len(self.orientation_parameters)
                == len(self.branch_summaries)):
            raise ValueError("goal branch arrays have different lengths")
        object.__setattr__(self, "weights", immutable_array(weights, weights.shape, "weights"))
        object.__setattr__(self, "orientations", tuple(self.orientations))
        parameters = tuple(_comparison_parameter(orientation, value)
            for orientation, value in zip(self.orientations, self.orientation_parameters))
        object.__setattr__(self, "orientation_parameters", parameters)
        object.__setattr__(self, "branch_summaries", tuple(self.branch_summaries))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def to_record(self, child_frame_id="switch_goal", stamp=None):
        return _summary_record(self, child_frame_id, stamp)


def _summary_record(summary, child, stamp):
    context = dict(summary.metadata)
    if stamp is not None:
        # A transport heartbeat must not rewrite the source observation time.
        context["publication_stamp"] = float(stamp)
    source_stamp = float(context.get("stamp", 0.0))
    source_id = str(context.get("source_id", "configured_transform_laws"))
    source_ids = tuple(dict.fromkeys((source_id,) + tuple(context.get("observation_source_ids", ()))))
    orientations = summary.orientations if isinstance(summary, GoalSummary) else (summary.orientation,)
    weights = summary.weights if isinstance(summary, GoalSummary) else (1.0,)
    method = ("indet_ik_common_gaussian_bingham_mixture"
              if isinstance(summary, GoalSummary) else COMPARISON_METHOD)
    components = tuple(TransformComponent(
        "{}:comparison:{}".format(child, index), float(weight), orientation,
        ConditionalGaussianTranslation(summary.mean, summary.covariance, np.zeros((3, 9))),
        provenance=ComponentProvenance(source_ids=source_ids, method=method,
                                      detail=_metadata_detail(context)),
        approximation=ApproximationInfo(ApproximationKind.BINGHAM_CLOSURE, True,
            "Full-path moments; final independent position Gaussian and orientation Bingham"))
        for index, (weight, orientation) in enumerate(zip(weights, orientations)))
    # The mode axis is an explicit representative of this comparison law.
    largest = int(np.argmax(weights))
    return TransformDistributionStamped(
        str(context.get("frame_id", "base_link")), child, source_stamp,
        "comparison_" + child, "indet_ik", TransformDistribution(components),
        representative=DeterministicTransform(summary.mean, orientations[largest].mode_wxyz),
        representative_kind=(RepresentativeKind.COMPONENT_MODE_APPROXIMATION
                             if isinstance(summary, GoalSummary)
                             else RepresentativeKind.MOMENT_REPRESENTATIVE),
        provenance=TransformProvenance(source_ids=source_ids, method=method,
                                      detail=_metadata_detail(context)),
        approximation=ApproximationInfo(ApproximationKind.BINGHAM_CLOSURE, True,
            "Only the final comparison product discards position/orientation dependence"))


@dataclass(frozen=True)
class _Prepared:
    component: TransformComponent
    rotation: RotationVectorMoments
    quaternion_moment: np.ndarray
    isotropic_factor: object = None

    def __post_init__(self):
        object.__setattr__(self, "quaternion_moment", immutable_symmetric_matrix(
            self.quaternion_moment, 4, "quaternion_moment", positive_semidefinite=True))

    @classmethod
    def from_component(cls, component, integration_steps, isotropic=False):
        orientation = component.orientation
        if orientation.kind is OrientationKind.DIRAC:
            quaternion_moment = np.outer(orientation.mode_wxyz, orientation.mode_wxyz)
        elif orientation.kind is OrientationKind.UNIFORM:
            quaternion_moment = np.eye(4) / 4
        else:
            quaternion_moment = bingham_second_moment(
                orientation.backend_parameter_matrix(), integration_steps)
        factor = None
        if orientation.kind is OrientationKind.DIRAC:
            factor = 1.
        elif orientation.kind is OrientationKind.UNIFORM:
            factor = 0.
        elif isotropic:
            mode = orientation.mode_wxyz
            factor = float((4. * mode @ quaternion_moment @ mode - 1.) / 3.)
        return cls(component, rotation_vector_moments(orientation, integration_steps),
                   immutable_symmetric_matrix(quaternion_moment, 4, "quaternion_moment"), factor)

    def forward(self, point):
        """Existing forward_component_point_moments identity with cached moments.

        Only the numerical Bingham moment query is cached.  In particular the
        C+action term retains both translation/rotation cross terms.
        """
        translation = self.component.translation
        coupling = translation.rotation_coupling
        linear = rotation_action_matrix(point.mean) + coupling
        offset = translation.mean_at_reference - coupling @ rotation_vector_from_quaternion(
            self.component.orientation.reference_quaternion_wxyz)
        centered = self.rotation.second_vector - np.outer(
            self.rotation.mean_vector, self.rotation.mean_vector)
        return PointMomentSummary(
            offset + linear @ self.rotation.mean_vector,
            linear @ centered @ linear.T
            + expected_rotated_covariance(self.rotation, point.covariance)
            + translation.residual_covariance)

    def right_rotation(self, quaternion):
        orientation = self.component.orientation
        right = quat_right_matrix(quaternion)
        moved_orientation = BinghamOrientation(
            orientation.kind, orientation.inverse_concentration,
            right @ orientation.shape_matrix @ right.T,
            quat_mul(orientation.reference_quaternion_wxyz, quaternion))
        rotation = quat_to_rotmat(quaternion)
        new_from_old = np.kron(rotation.T, np.eye(3))
        moved_rotation = RotationVectorMoments(
            self.rotation.mean_rotation @ rotation,
            new_from_old @ self.rotation.mean_vector,
            new_from_old @ self.rotation.second_vector @ new_from_old.T)
        old_translation = self.component.translation
        moved = replace(self.component, orientation=moved_orientation,
            translation=ConditionalGaussianTranslation(old_translation.mean_at_reference,
                old_translation.residual_covariance,
                old_translation.rotation_coupling @ np.kron(rotation, np.eye(3))))
        return _Prepared(moved, moved_rotation,
                         right @ self.quaternion_moment @ right.T, self.isotropic_factor)


@dataclass(frozen=True)
class TransformPath:
    """Independent native primitives, with cached exact first/fourth moments."""
    prepared: tuple
    metadata: object
    integration_steps: int = 480

    def __post_init__(self):
        object.__setattr__(self, "prepared", tuple(self.prepared))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def __iter__(self):
        return (item.component for item in self.prepared)

    def __len__(self):
        return len(self.prepared)


@lru_cache(maxsize=128)
def _isotropic_fit(factor, integration_steps):
    moment = np.diag([.25 + .75 * factor] + [.25 - .25 * factor] * 3)
    _orientation, parameter, _residual = _matched_orientation(moment, integration_steps)
    # The source certificate implies exactly three equal minor moments. Its
    # moment-matched Bingham has the same symmetry; remove only optimizer
    # symmetry-breaking roundoff before transporting the unique axial law.
    concentration = -float(np.trace(parameter)) / 3.
    parameter = np.diag([0., -concentration, -concentration, -concentration])
    fitted_moment = bingham_second_moment(parameter, integration_steps)
    parameter.setflags(write=False)
    fitted_moment.setflags(write=False)
    return parameter, fitted_moment


def _matched_orientation(moment, integration_steps, exact_orientation=None, isotropic_moment=None):
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if eigenvalues[0] < -1e-9 or abs(float(np.trace(moment)) - 1) > 1e-8:
        raise ValueError("invalid propagated quaternion second moment")
    if exact_orientation is not None:
        parameter = (None if exact_orientation.kind is OrientationKind.DIRAC
                     else exact_orientation.backend_parameter_matrix())
        return exact_orientation, parameter, 0.0
    if eigenvalues[0] <= 0.:
        raise RuntimeError("finite orientation moments are not numerically resolved")
    # High-concentration moment asymptotics provide only the optimizer's seed;
    # the returned law is fit by the existing deterministic normalizer moments.
    if isotropic_moment is None:
        initial = np.r_[-.5 / np.maximum(eigenvalues[:3], 1e-12), 0.0]
        parameter = match_bingham_to_second_moment(moment, integration_steps,
                                                   initial_eigenvalues=initial)
        fitted_moment = bingham_second_moment(parameter, integration_steps)
    else:
        factor, mode = isotropic_moment
        canonical, canonical_moment = _isotropic_fit(float(factor), integration_steps)
        right = quat_right_matrix(mode)
        parameter = right @ canonical @ right.T
        fitted_moment = right @ canonical_moment @ right.T
    residual = float(np.max(np.abs(fitted_moment - moment)))
    relative_eigen_residual = float(np.max(np.abs(np.linalg.eigvalsh(fitted_moment) - eigenvalues)
                                         / eigenvalues))
    if residual > 2e-5 or relative_eigen_residual > .01:
        raise RuntimeError("Bingham moment match failed: max residual {}".format(residual))
    orientation = BinghamOrientation.from_parameter_matrix(parameter)
    return orientation, orientation.backend_parameter_matrix(), residual


def summarize(path, finalpoint=(0., 0., 0.)):
    """Compose the whole independent path before its final product projection."""
    if not isinstance(path, TransformPath):
        raise TypeError("summarize requires a TransformPath with explicit independence")
    point = PointMomentSummary(finalpoint, np.zeros((3, 3)))
    for item in reversed(path.prepared):
        point = item.forward(point)
    quaternion_moment = np.diag([1., 0., 0., 0.])
    for item in path.prepared:
        quaternion_moment = quaternion_product_second_moment(
            quaternion_moment, item.quaternion_moment)
    # Special law types follow the primitive source types, never a concentration
    # threshold that would silently replace a finite source by a Dirac law.
    kinds = tuple(item.component.orientation.kind for item in path.prepared)
    exact_orientation = None
    isotropic_moment = None
    if OrientationKind.UNIFORM in kinds:
        exact_orientation = BinghamOrientation.uniform()
    elif all(kind is OrientationKind.DIRAC for kind in kinds):
        quaternion = np.array([1., 0., 0., 0.])
        for item in path.prepared:
            quaternion = quat_mul(quaternion, item.component.orientation.mode_wxyz)
        exact_orientation = BinghamOrientation.dirac(quaternion)
    elif all(item.isotropic_factor is not None for item in path.prepared):
        quaternion = np.array([1., 0., 0., 0.])
        factor = 1.
        for item in path.prepared:
            quaternion = quat_mul(quaternion, item.component.orientation.mode_wxyz)
            factor *= item.isotropic_factor
        isotropic_moment = (factor, quaternion)
    orientation, parameter, residual = _matched_orientation(
        quaternion_moment, path.integration_steps, exact_orientation, isotropic_moment)
    metadata = dict(path.metadata)
    metadata.update(moment_match_max_residual=residual,
                    finalpoint=tuple(float(value) for value in finalpoint),
                    comparison_method=COMPARISON_METHOD,
                    point_moments="full_independent_primitive_path",
                    final_projection="Gaussian_position_times_single_Bingham_orientation")
    return PoseSummary(point.mean, point.covariance, parameter, quaternion_moment,
                       orientation, metadata)


def _fixed(translation, quaternion, name, integration_steps):
    component = TransformComponent(name, 1., BinghamOrientation.dirac(quaternion),
        ConditionalGaussianTranslation(translation, np.zeros((3, 3)), np.zeros((3, 9))))
    return _Prepared.from_component(component, integration_steps)


class Model:
    """A frozen, directly configured family q -> independent local SE(3) laws."""
    def __init__(self, urdf, mapping, rod_length, hand_camera, integration_steps=480):
        if mapping.get("independent") is not True:
            raise ValueError("independent primitive laws must be explicitly declared")
        self.root_frame = str(mapping.get("root_frame", "base_link"))
        self.source_id = str(mapping["law_id"])
        self.revision = str(mapping["revision"])
        self.reference_command = immutable_array(mapping["reference_command"], (6,), "reference_command")
        self.integration_steps = int(integration_steps)
        if self.integration_steps != integration_steps or self.integration_steps < 1:
            raise ValueError("integration_steps must be a positive integer")
        descriptor = json.loads(json.dumps(mapping))
        edges, prepared, actions = [], [], []
        parent = self.root_frame
        for edge in mapping["edges"]:
            if edge["parent"] != parent:
                raise ValueError("configured local edges must form the root-to-tool chain")
            orientation_spec = edge["orientation"]
            kind = orientation_spec["kind"]
            if kind == "dirac":
                orientation = BinghamOrientation.dirac(orientation_spec["mode_wxyz"])
            elif kind == "bingham":
                mode = orientation_spec.get("mode_wxyz", [1., 0., 0., 0.])
                parameter = orientation_spec.get("parameter_matrix")
                if parameter is None:
                    parameter = bingham_parameter_from_mode(mode, orientation_spec["eigenvalues"])
                orientation = BinghamOrientation.from_parameter_matrix(parameter, mode)
            elif kind == "uniform":
                orientation = BinghamOrientation.uniform(
                    orientation_spec.get("mode_wxyz", [1., 0., 0., 0.]))
            else:
                raise ValueError("unsupported native orientation kind: " + str(kind))
            translation = edge["translation"]
            component = TransformComponent(edge["child"] + ":source", 1., orientation,
                ConditionalGaussianTranslation(translation["mean"], translation["covariance"],
                    translation.get("rotation_coupling", np.zeros((3, 9)))),
                provenance=ComponentProvenance(source_ids=(self.source_id,),
                    method="configured_native_local_transform_family"))
            action = edge.get("right_joint_action")
            if action is None:
                actions.append(None)
            else:
                joint_index = int(action["joint_index"])
                axis = immutable_array(action["axis"], (3,), "joint_axis")
                if joint_index < 0 or joint_index >= 6 or np.linalg.norm(axis) <= 0:
                    raise ValueError("invalid configured joint action")
                actions.append((joint_index, immutable_array(axis / np.linalg.norm(axis),
                                                             (3,), "joint_axis")))
            values = orientation_spec.get("eigenvalues")
            isotropic = (kind == "bingham" and "parameter_matrix" not in orientation_spec and values is not None
                         and values[0] > values[1] and values[1] == values[2] == values[3])
            prepared.append(_Prepared.from_component(component, self.integration_steps, isotropic=isotropic))
            edges.append((edge["parent"], edge["child"]))
            parent = edge["child"]
        if tuple(child for _parent, child in edges) != BODY_FRAMES[:-1]:
            raise ValueError("configure all six local link laws followed by tool0")
        self._base = tuple(prepared)
        self._actions = tuple(actions)
        self._nominal_local = tuple(self._reference_pose(item.component) for item in self._base)
        self.edges = tuple(edges) + (("tool0", "rod_tip"),)
        self._rod = _fixed([float(rod_length), 0., 0.], [1., 0., 0., 0.],
                           "known_rod_attachment", self.integration_steps)
        hand_camera = np.asarray(hand_camera, dtype=float)
        if hand_camera.shape != (4, 4):
            raise ValueError("hand_camera must be a 4x4 known rigid attachment")
        self._camera = _fixed(hand_camera[:3, 3], rotmat_to_quat(hand_camera[:3, :3]),
                              "known_camera_attachment", self.integration_steps)
        descriptor["known_attachments"] = dict(rod_length=float(rod_length),
                                               hand_camera=hand_camera.tolist())
        self._descriptor_json = json.dumps(descriptor, sort_keys=True)
        self.source_fingerprint = sha256(self._descriptor_json.encode()).hexdigest()
        self.knowngeometry = BottleCollisionModel(Path(urdf))

    @property
    def descriptor(self):
        # An editable export cannot mutate this frozen source-family snapshot.
        return json.loads(self._descriptor_json)

    @staticmethod
    def _reference_pose(component):
        pose = np.eye(4)
        pose[:3, :3] = quat_to_rotmat(component.orientation.reference_quaternion_wxyz)
        pose[:3, 3] = component.translation.mean_at_reference
        pose.setflags(write=False)
        return pose

    @classmethod
    def from_config(cls, urdf, configpath=None, rod_length=.325, hand_camera=None,
                    integration_steps=480):
        if configpath is None:
            configpath = Path(__file__).resolve().parents[2] / "config/independent_transform_laws.yaml"
        if isinstance(configpath, dict):
            mapping = configpath
        else:
            mapping = yaml.safe_load(Path(configpath).read_text())
        if hand_camera is None:
            hand_camera = np.eye(4)
        return cls(urdf, mapping, rod_length, hand_camera, integration_steps)

    @lru_cache(maxsize=128)
    def _command_primitives(self, command):
        return tuple(item if action is None else item.right_rotation(axis_angle_to_quat(
            action[1], command[action[0]] - self.reference_command[action[0]]))
            for item, action in zip(self._base, self._actions))

    def chain(self, q, endpoint="tip", stamp=0.):
        q = immutable_array(q, (6,), "command")
        # The family is frozen. Exact command keys share only its transported
        # native primitives; observation/query metadata is constructed below.
        items = self._command_primitives(tuple(float(value) for value in q))
        if endpoint in ("tip", "rod_tip"):
            items += (self._rod,)
        elif endpoint in ("camera", "camera_optical_frame"):
            items += (self._camera,)
        elif endpoint in ("flange", "tool0"):
            pass
        elif endpoint in BODY_FRAMES:
            items = items[:BODY_FRAMES.index(endpoint) + 1]
        else:
            raise ValueError("unknown chain endpoint: " + str(endpoint))
        metadata = dict(frame_id=self.root_frame, source_id=self.source_id,
            revision=self.revision, source_fingerprint=self.source_fingerprint,
            command=tuple(float(x) for x in q), stamp=float(stamp), endpoint=endpoint,
            primitive_dependence="independent", integration_method="deterministic_moments")
        metadata["moment_integration_steps"] = self.integration_steps
        return TransformPath(items, metadata, self.integration_steps)

    def hand(self, q, stamp=0.):
        return summarize(self.chain(q, "tip", stamp))

    def position_moments(self, q, endpoint="tip"):
        """Full-path position moments without the final orientation fit."""
        path = self.chain(q, endpoint)
        point = PointMomentSummary(np.zeros(3), np.zeros((3, 3)))
        for item in reversed(path.prepared):
            point = item.forward(point)
        return point.mean, point.covariance

    def camera(self, q, stamp=0.):
        return summarize(self.chain(q, "camera", stamp))

    def nominal_roots(self, q):
        q = immutable_array(q, (6,), "command")
        roots, pose = {self.root_frame: np.eye(4)}, np.eye(4)
        for frame, reference, action in zip(BODY_FRAMES, self._nominal_local, self._actions):
            local = reference
            if action is not None:
                local = reference.copy()
                local[:3, :3] = reference[:3, :3] @ quat_to_rotmat(axis_angle_to_quat(
                    action[1], q[action[0]] - self.reference_command[action[0]]))
            pose = pose @ local
            roots[frame] = pose.copy()
        roots["rod_tip"] = roots["tool0"] @ self._reference_pose(self._rod.component)
        roots["press_rod"] = roots["tool0"]
        roots["camera_optical_frame"] = roots["tool0"] @ self._reference_pose(self._camera.component)
        return roots

    def local_records(self, q, stamp=0.):
        path = self.chain(q, "tip", stamp)
        return tuple(TransformDistributionStamped(parent, child, float(stamp), "local_" + child,
            "indet_ik", TransformDistribution((item.component,)),
            provenance=TransformProvenance(source_ids=(self.source_id,),
                method="configured_native_local_transform_family", detail=_metadata_detail(path.metadata)))
            for (parent, child), item in zip(self.edges, path.prepared))

    def display_records(self, q, stamp=0., hand_summary=None):
        """Publish the primitive path and only the specified Hand projection.

        Intermediate transforms retain their native local laws; RViz queries
        the whole prefix directly. No intermediate X/R product is invented.
        The tip is the SAME final comparison law passed to the objective.
        """
        full = self.chain(q, "tip", stamp)
        summary = self.hand(q, stamp) if hand_summary is None else hand_summary
        if float(summary.metadata["stamp"]) != float(stamp):
            raise ValueError("hand display source stamp differs from objective source")
        if tuple(summary.metadata["command"]) != tuple(full.metadata["command"]):
            raise ValueError("hand display command differs from objective command")
        if (summary.metadata["source_fingerprint"] != self.source_fingerprint
                or summary.metadata["revision"] != self.revision
                or summary.metadata["frame_id"] != self.root_frame):
            raise ValueError("hand display source differs from objective source")
        if (summary.metadata["endpoint"] not in ("tip", "rod_tip")
                or summary.metadata["finalpoint"] != (0., 0., 0.)):
            raise ValueError("hand display endpoint differs from objective endpoint")
        return self.local_records(q, stamp)[:-1] + (summary.to_record("rod_tip", stamp),)

    def _observed(self, q, marker_record, attachment, endpoint):
        if not isinstance(marker_record, TransformDistributionStamped):
            raise TypeError("marker_record must retain the native observed transform law")
        if marker_record.parent_frame_id != "camera_optical_frame":
            raise ValueError("marker law must be expressed in the configured optical camera frame")
        camera_path = self.chain(q, "camera", marker_record.stamp)
        normalized = marker_record.distribution.normalized_components()
        if not normalized:
            raise ValueError("observed marker has no positive candidate mass")
        summaries, weights = [], []
        observation_sources = tuple(dict.fromkeys(tuple(marker_record.provenance.source_ids) + tuple(
            source for branch in normalized for source in branch.component.provenance.source_ids)))
        for branch in normalized:
            marker = _Prepared.from_component(branch.component, self.integration_steps)
            items = camera_path.prepared + (marker,)
            if attachment is not None:
                items += (attachment,)
            metadata = dict(camera_path.metadata, endpoint=endpoint,
                observation_edge_id=marker_record.edge_id,
                observation_stamp=float(marker_record.stamp),
                observation_source_ids=observation_sources,
                observation_command=tuple(float(x) for x in q),
                observation_component_id=branch.component.component_id)
            summaries.append(summarize(TransformPath(items, metadata, self.integration_steps)))
            weights.append(branch.weight)
        position = mixture_point_moments(tuple((weight, PointMomentSummary(summary.mean, summary.covariance))
            for weight, summary in zip(weights, summaries)))
        metadata = dict(camera_path.metadata, endpoint=endpoint,
            observation_edge_id=marker_record.edge_id,
            observation_stamp=float(marker_record.stamp),
            observation_source_ids=observation_sources,
            observation_command=tuple(float(x) for x in q),
            orientation_component_ids=tuple(branch.component.component_id for branch in normalized),
            comparison_method=COMPARISON_METHOD,
            final_projection="common_Gaussian_position_times_Bingham_mixture_orientation")
        return GoalSummary(position.mean, position.covariance, np.asarray(weights),
            tuple(summary.orientation_parameter for summary in summaries),
            tuple(summary.orientation for summary in summaries), metadata, tuple(summaries))

    def target(self, q_observed, marker_record, goal_offset,
               goal_orientation_wxyz=(1., 0., 0., 0.)):
        attachment = _fixed(goal_offset, goal_orientation_wxyz,
                            "known_marker_goal_attachment", self.integration_steps)
        return self._observed(q_observed, marker_record, attachment, "switch_goal")

    def marker(self, q_observed, marker_record):
        return self._observed(q_observed, marker_record, None, "observed_ar_marker")
