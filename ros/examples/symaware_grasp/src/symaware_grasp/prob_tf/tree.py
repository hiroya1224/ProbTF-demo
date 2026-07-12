from dataclasses import dataclass

import numpy as np

from symaware_grasp.prob_tf.bingham_match import match_bingham_to_second_moment, quaternion_product_second_moment
from symaware_grasp.prob_tf.geometry import axis_angle_to_quat
from symaware_grasp.prob_tf.path_expression import EdgeView, PathExpression
from symaware_grasp.prob_tf.rotation_moments import (
    identity_rotation_moment,
    rotation_moment_from_bingham,
)
from symaware_grasp.prob_tf.tangent_surrogate import induced_vector_moments_tangent


def _symmetrize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _identity_quaternion_second_moment():
    moment = np.zeros((4, 4), dtype=float)
    moment[0, 0] = 1.0
    return moment


@dataclass
class ProbTfEdge:
    edge_id: str
    parent: str
    child: str
    translation: np.ndarray
    joint_type: str
    axis: np.ndarray = None
    nominal_angle: float = 0.0
    nominal_quaternion: np.ndarray = None
    bingham_param: np.ndarray = None
    rotation_moment: object = None
    quaternion_second_moment: np.ndarray = None

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        if self.axis is not None:
            self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        if self.nominal_quaternion is None:
            if self.joint_type == "fixed":
                self.nominal_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                self.nominal_quaternion = axis_angle_to_quat(self.axis, self.nominal_angle)
        else:
            self.nominal_quaternion = np.asarray(self.nominal_quaternion, dtype=float).reshape(4)

        if self.joint_type == "fixed":
            self.bingham_param = None if self.bingham_param is None else np.asarray(self.bingham_param, dtype=float)
            self.rotation_moment = identity_rotation_moment() if self.rotation_moment is None else self.rotation_moment
            self.quaternion_second_moment = (
                _identity_quaternion_second_moment()
                if self.quaternion_second_moment is None
                else _symmetrize(self.quaternion_second_moment)
            )
        else:
            self.bingham_param = np.asarray(self.bingham_param, dtype=float).reshape(4, 4)
            if self.rotation_moment is None:
                self.rotation_moment = rotation_moment_from_bingham(self.bingham_param)
            if self.quaternion_second_moment is None:
                from symaware_grasp.prob_tf.bingham_moments import bingham_second_moment

                self.quaternion_second_moment = bingham_second_moment(self.bingham_param)
            self.quaternion_second_moment = _symmetrize(self.quaternion_second_moment)


@dataclass
class ProbTfResult:
    source: str
    target: str
    mean_translation: np.ndarray
    cov_translation: np.ndarray
    mean_rotation: np.ndarray = None
    bingham_rotation: np.ndarray = None
    path: object = None
    method: str = None
    closure_approximation: bool = False


class ProbTfTree:
    def __init__(self, root):
        self.root = root
        self.edges = {}
        self.parent_of = {}
        self.children_of = {}
        self.frames = {root}
        self.frame_order = [root]

    def add_edge(self, edge, closure_approximation=False):
        if isinstance(edge, ProbTfResult) and not closure_approximation:
            raise ValueError(
                "Summarized ProbTfResult objects must not be re-registered as independent edges "
                "unless closure_approximation=True is explicitly acknowledged."
            )
        if not isinstance(edge, ProbTfEdge):
            raise TypeError("ProbTfTree.add_edge expects a ProbTfEdge.")
        if edge.edge_id in self.edges:
            raise ValueError("Duplicate edge_id '%s'." % edge.edge_id)
        if edge.child in self.parent_of:
            raise ValueError("Frame '%s' already has a parent." % edge.child)

        self.edges[edge.edge_id] = edge
        self.parent_of[edge.child] = edge.parent
        self.children_of.setdefault(edge.parent, []).append(edge.child)
        self.frames.add(edge.parent)
        self.frames.add(edge.child)
        if edge.parent not in self.frame_order:
            self.frame_order.append(edge.parent)
        if edge.child not in self.frame_order:
            self.frame_order.append(edge.child)

    def to_core_graph(self, stamp=0.0, authority="symaware_grasp_legacy_adapter"):
        """Embed legacy demo edges in a timestamped :class:`ProbTfGraph`.

        Legacy edges have deterministic translation and either a deterministic
        or Bingham rotation. They map to one uncoupled component per physical
        edge. No legacy moment summary is re-registered as an edge.
        """

        from probtf.distributions import (
            BinghamOrientation,
            ConditionalGaussianTranslation,
            TransformComponent,
            TransformDistribution,
            TransformDistributionStamped,
        )
        from probtf.graph import ProbTfGraph
        from probtf.provenance import (
            ApproximationInfo,
            ApproximationKind,
            ComponentProvenance,
            TransformProvenance,
        )

        graph = ProbTfGraph()
        for edge in self.edges.values():
            orientation = (
                BinghamOrientation.dirac(edge.nominal_quaternion)
                if edge.joint_type == "fixed"
                else BinghamOrientation.from_parameter_matrix(
                    edge.bingham_param,
                    edge.nominal_quaternion,
                )
            )
            component = TransformComponent(
                component_id="{}:legacy".format(edge.edge_id),
                raw_weight=1.0,
                orientation=orientation,
                translation=ConditionalGaussianTranslation(
                    edge.translation,
                    np.zeros((3, 3)),
                    np.zeros((3, 9)),
                ),
                provenance=ComponentProvenance(
                    source_ids=(authority,),
                    method="symaware_grasp_prob_tf_edge_adapter",
                ),
                approximation=ApproximationInfo(
                    ApproximationKind.LEGACY_ADAPTER,
                    False,
                    "Legacy fixed-translation edge embedded with zero rotation coupling.",
                ),
            )
            graph.insert(
                TransformDistributionStamped(
                    parent_frame_id=edge.parent,
                    child_frame_id=edge.child,
                    stamp=float(stamp),
                    edge_id=edge.edge_id,
                    authority=authority,
                    distribution=TransformDistribution((component,)),
                    provenance=TransformProvenance(
                        source_ids=(authority,),
                        method="symaware_grasp_tree_adapter",
                    ),
                    is_static=True,
                )
            )
        return graph

    def lookup_core_kernel(self, source, target, stamp=0.0):
        """Return a lazy core kernel while preserving legacy lookup arguments.

        Legacy ``lookup_path(source, target)`` asks for the pose of ``target``
        relative to ``source``. The core uses tf2/action order
        ``lookup_kernel(target_frame, source_frame, stamp)``, so the two frame
        arguments are intentionally swapped here.
        """

        from probtf.temporal import TemporalPolicy

        graph = self.to_core_graph(stamp=stamp)
        return graph.lookup_kernel(
            target_frame=source,
            source_frame=target,
            stamp=stamp,
            policy=TemporalPolicy.EXACT,
        )

    def _edge_to_parent(self, frame):
        if frame == self.root:
            return None
        parent = self.parent_of[frame]
        for edge in self.edges.values():
            if edge.parent == parent and edge.child == frame:
                return edge
        raise KeyError("No incoming edge found for frame '%s'." % frame)

    def _path_from_root(self, frame):
        edges = []
        current = frame
        while current != self.root:
            edge = self._edge_to_parent(current)
            edges.append(edge)
            current = edge.parent
        edges.reverse()
        return edges

    def _ancestor_chain(self, frame):
        chain = [frame]
        current = frame
        while current != self.root:
            current = self.parent_of[current]
            chain.append(current)
        return chain

    def lookup_path(self, source, target):
        if source not in self.frames or target not in self.frames:
            raise KeyError("Unknown frame in lookup_path('%s', '%s')." % (source, target))

        source_chain = self._ancestor_chain(source)
        target_chain = self._ancestor_chain(target)
        target_set = set(target_chain)

        lca = None
        for frame in source_chain:
            if frame in target_set:
                lca = frame
                break
        if lca is None:
            raise RuntimeError("No common ancestor found.")

        views = []
        current = source
        while current != lca:
            edge = self._edge_to_parent(current)
            views.append(EdgeView(edge.edge_id, -1))
            current = edge.parent

        down_frames = []
        current = target
        while current != lca:
            edge = self._edge_to_parent(current)
            down_frames.append(EdgeView(edge.edge_id, +1))
            current = edge.parent
        views.extend(reversed(down_frames))

        path = PathExpression(views).reduce_adjacent_inverses()
        path.assert_no_repeated_edge_ids()
        return path

    def lookup(self, source, target, return_bingham=False, summarize=True):
        path = self.lookup_path(source, target)
        if not summarize:
            return path
        if source != self.root:
            raise NotImplementedError(
                "Full moment summary is only implemented for root-to-target queries in the initial prototype."
            )
        return self.compute_link_origin_moments(target, return_bingham=return_bingham, path=path)

    def lookup_point(self, source, target, local_point, return_bingham=False, summarize=True):
        path = self.lookup_path(source, target)
        if not summarize:
            return path
        if source != self.root:
            raise NotImplementedError(
                "Full point-moment summary is only implemented for root-to-target queries in the initial prototype."
            )
        return self.compute_attached_point_moments(
            target,
            local_point=local_point,
            return_bingham=return_bingham,
            path=path,
        )

    def lookup_point_tangent_surrogate(
        self,
        source,
        target,
        local_point,
        return_bingham=False,
        summarize=True,
        use_jacobian_correction=True,
    ):
        path = self.lookup_path(source, target)
        if not summarize:
            return path
        if source != self.root:
            raise NotImplementedError(
                "Full point-moment tangent surrogate summary is only implemented for root-to-target queries "
                "in the initial prototype."
            )
        return self.compute_attached_point_tangent_surrogate(
            target,
            local_point=local_point,
            return_bingham=return_bingham,
            path=path,
            use_jacobian_correction=use_jacobian_correction,
        )

    def _forward_edges_from_path(self, path):
        edges = []
        for view in path:
            if view.direction != +1:
                raise NotImplementedError(
                    "Root-to-target moment summary only supports forward edges in the initial prototype."
                )
            edges.append(self.edges[view.edge_id])
        return edges

    def _build_prefix_rotation_summaries(self, edges):
        prefix_mean_rotations = [np.eye(3, dtype=float)]
        prefix_rotation_moments = [identity_rotation_moment()]
        prefix_quaternion_seconds = [_identity_quaternion_second_moment()]

        for edge in edges:
            prefix_mean_rotations.append(prefix_mean_rotations[-1] @ edge.rotation_moment.mean_rot)
            prefix_rotation_moments.append(prefix_rotation_moments[-1].compose(edge.rotation_moment))
            prefix_quaternion_seconds.append(
                quaternion_product_second_moment(prefix_quaternion_seconds[-1], edge.quaternion_second_moment)
            )

        return prefix_mean_rotations, prefix_rotation_moments, prefix_quaternion_seconds

    def _position_moments_from_terms(self, edges, terms, prefix_mean_rotations, prefix_rotation_moments):
        mean_terms = [prefix_mean_rotations[prefix_length] @ vector for prefix_length, vector in terms]
        mean_translation = np.sum(mean_terms, axis=0) if mean_terms else np.zeros(3, dtype=float)

        second_moment = np.zeros((3, 3), dtype=float)
        for term_index_i, (prefix_i, vector_i) in enumerate(terms):
            for term_index_j, (prefix_j, vector_j) in enumerate(terms):
                if term_index_i <= term_index_j:
                    segment_rotation = np.eye(3, dtype=float)
                    for edge_index in range(prefix_i, prefix_j):
                        segment_rotation = segment_rotation @ edges[edge_index].rotation_moment.mean_rot
                    inner = np.outer(vector_i, segment_rotation @ vector_j)
                    term = prefix_rotation_moments[prefix_i].apply_second(inner)
                    second_moment += term if term_index_i == term_index_j else term + term.T

        cov_translation = _symmetrize(second_moment - np.outer(mean_translation, mean_translation))
        return mean_translation, cov_translation

    def _cross_second_moment_forward(
        self,
        edges,
        prefix_rotation_moments,
        left_prefix_length,
        left_vector,
        right_prefix_length,
        right_vector,
    ):
        if left_prefix_length > right_prefix_length:
            return self._cross_second_moment_forward(
                edges,
                prefix_rotation_moments,
                right_prefix_length,
                right_vector,
                left_prefix_length,
                left_vector,
            ).T

        segment_rotation = np.eye(3, dtype=float)
        for edge_index in range(left_prefix_length, right_prefix_length):
            segment_rotation = segment_rotation @ edges[edge_index].rotation_moment.mean_rot
        inner = np.outer(left_vector, segment_rotation @ right_vector)
        return prefix_rotation_moments[left_prefix_length].apply_second(inner)

    def compute_link_origin_moments(self, target, return_bingham=False, path=None):
        if target == self.root:
            return ProbTfResult(
                source=self.root,
                target=self.root,
                mean_translation=np.zeros(3, dtype=float),
                cov_translation=np.zeros((3, 3), dtype=float),
                mean_rotation=np.eye(3, dtype=float),
                bingham_rotation=None,
                path=PathExpression([]),
                method="root_to_link_moment_propagation",
            )

        if path is None:
            path = self.lookup_path(self.root, target)
        edges = self._forward_edges_from_path(path)
        prefix_mean_rotations, prefix_rotation_moments, prefix_quaternion_seconds = self._build_prefix_rotation_summaries(
            edges
        )
        terms = [(index, edge.translation) for index, edge in enumerate(edges)]
        mean_translation, cov_translation = self._position_moments_from_terms(
            edges,
            terms,
            prefix_mean_rotations,
            prefix_rotation_moments,
        )

        bingham_rotation = None
        if return_bingham:
            bingham_rotation = match_bingham_to_second_moment(prefix_quaternion_seconds[-1])

        return ProbTfResult(
            source=self.root,
            target=target,
            mean_translation=mean_translation,
            cov_translation=cov_translation,
            mean_rotation=prefix_mean_rotations[-1],
            bingham_rotation=bingham_rotation,
            path=path,
            method="root_to_link_moment_propagation",
            closure_approximation=bool(return_bingham),
        )

    def compute_attached_point_moments(self, target, local_point, return_bingham=False, path=None):
        point = np.asarray(local_point, dtype=float).reshape(3)
        if target == self.root:
            mean_translation = point.copy()
            cov_translation = np.zeros((3, 3), dtype=float)
            return ProbTfResult(
                source=self.root,
                target=target,
                mean_translation=mean_translation,
                cov_translation=cov_translation,
                mean_rotation=np.eye(3, dtype=float),
                bingham_rotation=None,
                path=PathExpression([]),
                method="root_to_attached_point_moment_propagation",
            )

        if path is None:
            path = self.lookup_path(self.root, target)
        edges = self._forward_edges_from_path(path)
        prefix_mean_rotations, prefix_rotation_moments, prefix_quaternion_seconds = self._build_prefix_rotation_summaries(
            edges
        )
        terms = [(index, edge.translation) for index, edge in enumerate(edges)]
        terms.append((len(edges), point))
        mean_translation, cov_translation = self._position_moments_from_terms(
            edges,
            terms,
            prefix_mean_rotations,
            prefix_rotation_moments,
        )

        bingham_rotation = None
        if return_bingham:
            bingham_rotation = match_bingham_to_second_moment(prefix_quaternion_seconds[-1])

        return ProbTfResult(
            source=self.root,
            target=target,
            mean_translation=mean_translation,
            cov_translation=cov_translation,
            mean_rotation=prefix_mean_rotations[-1],
            bingham_rotation=bingham_rotation,
            path=path,
            method="root_to_attached_point_moment_propagation",
            closure_approximation=bool(return_bingham),
        )

    def compute_attached_point_tangent_surrogate(
        self,
        target,
        local_point,
        return_bingham=False,
        path=None,
        use_jacobian_correction=True,
    ):
        point = np.asarray(local_point, dtype=float).reshape(3)
        if target == self.root:
            mean_translation = point.copy()
            cov_translation = np.zeros((3, 3), dtype=float)
            return ProbTfResult(
                source=self.root,
                target=target,
                mean_translation=mean_translation,
                cov_translation=cov_translation,
                mean_rotation=np.eye(3, dtype=float),
                bingham_rotation=None,
                path=PathExpression([]),
                method="root_to_attached_point_tangent_surrogate",
            )

        if path is None:
            path = self.lookup_path(self.root, target)
        edges = self._forward_edges_from_path(path)
        prefix_mean_rotations, prefix_rotation_moments, prefix_quaternion_seconds = self._build_prefix_rotation_summaries(
            edges
        )
        origin_terms = [(index, edge.translation) for index, edge in enumerate(edges)]
        origin_mean, origin_cov = self._position_moments_from_terms(
            edges,
            origin_terms,
            prefix_mean_rotations,
            prefix_rotation_moments,
        )

        cumulative_bingham = match_bingham_to_second_moment(prefix_quaternion_seconds[-1])
        vector_surrogate = induced_vector_moments_tangent(
            point,
            cumulative_bingham,
            use_jacobian_correction=use_jacobian_correction,
        )

        cross_second = np.zeros((3, 3), dtype=float)
        for prefix_length, translation in origin_terms:
            cross_second += self._cross_second_moment_forward(
                edges,
                prefix_rotation_moments,
                prefix_length,
                translation,
                len(edges),
                point,
            )
        cross_cov = cross_second - np.outer(origin_mean, vector_surrogate.mean)

        mean_translation = origin_mean + vector_surrogate.mean
        cov_translation = _symmetrize(origin_cov + vector_surrogate.cov + cross_cov + cross_cov.T)

        return ProbTfResult(
            source=self.root,
            target=target,
            mean_translation=mean_translation,
            cov_translation=cov_translation,
            mean_rotation=prefix_mean_rotations[-1],
            bingham_rotation=cumulative_bingham if return_bingham else None,
            path=path,
            method="root_to_attached_point_tangent_surrogate",
            closure_approximation=True,
        )
