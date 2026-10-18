"""Endpoint-only auxiliary stiffness factors from the retained body laws.

This module receives estimated laws and geometry only.  It has no simulator or
observation source.  Every law is propagated by the same demo-native/core pose
query used elsewhere; no individual intermediate-link discrepancy is scored.
"""

import hashlib
import math

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from probtf.geometry import quat_to_rotmat, skew
from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import require_in_support
from .body_native import materialize_state_in_chart, pose_query
from .joint_batch_fusion import add_log_evidence_parts, log_evidence_parts_difference


def _weight(component):
    supplied = getattr(component, 'log_weight', None)
    return (float(supplied) if supplied is not None else
            -np.inf if component.weight == 0. else math.log(component.weight))


def _right_log_derivative(vector):
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    coefficient = (1./12.+angle*angle/720. if angle < 1.e-6 else
                   (1.-.5*angle/math.tan(.5*angle))/(angle*angle))
    return np.eye(3)+.5*cross+coefficient*(cross @ cross)


def _rotation_volume(vector):
    """One rotation-vector chart's volume relative to the same Haar measure."""
    angle = float(np.linalg.norm(vector))
    if angle < 1.e-5:
        return (-angle**2/12.-angle**4/1440.,
                -(1./6.+angle**2/360.)*vector)
    return (2.*math.log(math.sin(.5*angle)/(.5*angle)),
            (1./math.tan(.5*angle)-2./angle)*vector/angle)


def _density(distribution, value):
    support = distribution.support
    error = value-distribution.mean
    require_in_support(error, support, 'auxiliary endpoint prediction in hand support')
    score = -support.inverse @ error
    parts = add_log_evidence_parts((0., 0.),
        (-.5*(support.rank*math.log(2.*math.pi)+support.log_pdet),
         .5*float(error @ score)))
    return parts, score


def _mixture(parts, gradients):
    reference = max(parts, key=lambda item: (item[0], item[1]))
    shifted = np.asarray([log_evidence_parts_difference(item, reference) for item in parts])
    normalizer = float(logsumexp(shifted))
    return (add_log_evidence_parts(reference, (normalizer,)),
            np.exp(shifted-normalizer) @ np.asarray(gradients))


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


class HandPoseDensityRatio:
    """Ratio of pose push-forwards, with each law's own modal rotation chart.

    Native transition-prior branches remain the denominator.  When supplied,
    the numerator is the declared canonical single body law shared by control
    and visualization.  The endpoint Gaussian is the existing deterministic
    moment approximation, not an exact nonlinear Bayes likelihood.
    """

    def __init__(self, branches, body_edge_ids, hand_path, posterior_branch=None):
        self.branches = tuple(branches)
        self.body_edge_ids = tuple(body_edge_ids)
        self.hand_path = tuple(hand_path)
        self.posterior_branch = posterior_branch
        if not self.branches:
            raise ValueError('Hand stiffness factors require actual frozen body branches.')
        posterior_index = None
        if posterior_branch is not None:
            matches = [i for i, branch in enumerate(self.branches)
                if branch.reference_state.state_id == posterior_branch.reference_state.state_id]
            if len(matches) != 1:
                raise ValueError('The canonical hand numerator needs its actual frozen body chart.')
            posterior_index = matches[0]
            reference = self.branches[posterior_index].reference_state
            if (not np.array_equal(posterior_branch.prior.mean, reference.mean)
                    or not np.array_equal(posterior_branch.prior.covariance, reference.covariance)):
                raise ValueError('The canonical hand numerator changed its reference body law.')
            if len(posterior_branch.posterior_components) != 1:
                raise ValueError('The canonical hand numerator must be the shared single internal law.')
        self.marginals, self.prior_queries, self.child_queries = [], [], []
        self.identity = True
        for i, branch in enumerate(self.branches):
            reference = branch.reference_state
            prior_query = self._query(reference, branch.prior, 'prior')
            prior = Gaussian(prior_query.mean, prior_query.covariance)
            sources = (branch.posterior_components if posterior_branch is None else
                posterior_branch.posterior_components if i == posterior_index else ())
            children, queries = [], []
            for child in sources:
                query = self._query(reference, child.distribution, 'posterior:'+child.label)
                children.append((child, Gaussian(query.mean, query.covariance)))
                queries.append(query)
            mass = float(logsumexp([_weight(child) for child in sources])) if sources else -np.inf
            self.identity &= (all(np.array_equal(child.distribution.mean, branch.prior.mean)
                and np.array_equal(child.distribution.covariance, branch.prior.covariance)
                for child in sources) and mass == _weight(branch))
            self.marginals.append((prior, tuple(children), prior.support))
            self.prior_queries.append(prior_query)
            self.child_queries.append(tuple(queries))
        for weights in ([_weight(branch) for branch in self.branches],
                        [_weight(child) for _, children, _ in self.marginals for child, _ in children]):
            if not np.isclose(logsumexp(weights), 0., atol=1.e-10, rtol=0.):
                raise ValueError('Hand prior/posterior global mixture masses must be normalized.')
        if not self.identity and any(distribution.support.rank != 6
                for prior, children, _ in self.marginals
                for distribution in (prior,)+(tuple(item[1] for item in children))):
            raise ValueError('A singular endpoint law needs an explicit common physical support measure; no jitter is added.')
        self.prior_queries, self.child_queries = tuple(self.prior_queries), tuple(self.child_queries)
        self.pose_charts = tuple(quat_to_rotmat(query.rotation_wxyz) for query in self.prior_queries)

    def _query(self, reference, distribution, label):
        # Reuse the actual state exactly for an unchanged law.  Other laws use
        # the existing G P G.T/Bingham reexpression with no prior conditioning.
        state = reference if (np.array_equal(distribution.mean, reference.mean)
            and np.array_equal(distribution.covariance, reference.covariance)) else materialize_state_in_chart(
                reference, distribution.mean, distribution.covariance,
                version=reference.version, component_id=reference.component_id+':hand_K:'+label)
        return pose_query(state, self.hand_path, query_id='stiffness:hand:'+label,
                          frame='base_link')

    def evaluate_pose(self, position, rotation, position_jacobian, body_angular_jacobian):
        position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        position_jacobian = np.asarray(position_jacobian, dtype=float)
        body_angular_jacobian = np.asarray(body_angular_jacobian, dtype=float)
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or position_jacobian.ndim != 2 or position_jacobian.shape[0] != 3
                or body_angular_jacobian.shape != position_jacobian.shape
                or not all(np.all(np.isfinite(item)) for item in
                    (position, rotation, position_jacobian, body_angular_jacobian))):
            raise ValueError('The endpoint stiffness prediction and its derivatives must be finite and dimensionally consistent.')
        minus_parts, plus_parts, minus_gradients, plus_gradients = [], [], [], []
        def evaluate(query, distribution):
            delta = Rotation.from_matrix(quat_to_rotmat(query.rotation_wxyz).T @ rotation).as_rotvec()
            value = np.r_[position, delta]
            parts, score = _density(distribution, value)
            volume, volume_gradient = _rotation_volume(delta)
            score = score-np.r_[np.zeros(3), volume_gradient]
            jacobian = np.vstack((position_jacobian,
                _right_log_derivative(delta) @ body_angular_jacobian))
            return add_log_evidence_parts(parts, (-volume,)), jacobian.T @ score
        for branch, (prior, children, _), query, queries in zip(
                self.branches, self.marginals, self.prior_queries, self.child_queries):
            parts, gradient = evaluate(query, prior)
            if self.identity:
                continue
            minus_parts.append(add_log_evidence_parts(parts, (_weight(branch),)))
            minus_gradients.append(gradient)
            for (child, distribution), child_query in zip(children, queries):
                parts, gradient = evaluate(child_query, distribution)
                plus_parts.append(add_log_evidence_parts(parts, (_weight(child),)))
                plus_gradients.append(gradient)
        if self.identity:
            return 0., np.zeros(position_jacobian.shape[1])
        minus, minus_gradient = _mixture(minus_parts, minus_gradients)
        plus, plus_gradient = _mixture(plus_parts, plus_gradients)
        return log_evidence_parts_difference(plus, minus), plus_gradient-minus_gradient

    def properness_bounds(self, position_radius):
        position_radius = float(position_radius)
        if not np.isfinite(position_radius) or position_radius < 0.:
            raise ValueError('The declared compact hand-position radius must be finite and nonnegative.')
        if self.identity:
            return dict(verified=True, log_factor_lower_bound=0., log_factor_upper_bound=0.,
                        argument='declared exact identity endpoint density ratio')
        lower_minus, upper_minus, lower_plus, upper_plus = [], [], [], []
        def bounds(distribution):
            support = distribution.support
            if support.rank != 6:
                raise ValueError('Hand properness needs an explicit common physical measure for singular endpoint laws.')
            radius_squared = ((position_radius+np.linalg.norm(distribution.mean[:3]))**2
                +(math.pi+np.linalg.norm(distribution.mean[3:]))**2)
            maximum = -.5*(6*math.log(2.*math.pi)+support.log_pdet)
            minimum = maximum-.5*np.linalg.eigvalsh(support.inverse)[-1]*radius_squared
            return float(minimum), float(maximum+2.*math.log(math.pi/2.))
        for branch, (prior, children, _) in zip(self.branches, self.marginals):
            low, high = bounds(prior)
            lower_minus.append(_weight(branch)+low); upper_minus.append(_weight(branch)+high)
            for child, distribution in children:
                low, high = bounds(distribution)
                lower_plus.append(_weight(child)+low); upper_plus.append(_weight(child)+high)
        lower = float(logsumexp(lower_plus)-logsumexp(upper_minus))
        upper = float(logsumexp(upper_plus)-logsumexp(lower_minus))
        if not np.isfinite(lower+upper):
            raise ValueError('The endpoint ratio compact-domain bounds are nonfinite.')
        return dict(verified=True, log_factor_lower_bound=lower, log_factor_upper_bound=upper,
            argument='positive full-support six-dimensional pose Gaussians on bounded URDF hand positions and principal rotation charts',
            hand_position_radius_m=position_radius, intermediate_link_density_scored=False,
            body_density_domain='existing Bingham-derived propagated Gaussian pose approximation; no exact nonlinear density claim')

    def record(self):
        def query_record(query, distribution):
            return dict(source_state_id=query.state_id, source_version=query.version,
                source_stamp=query.stamp, source_command=list(query.command),
                query_id=query.query_id, modal_rotation_wxyz=query.rotation_wxyz.tolist(),
                modal_translation=query.mode[:3].tolist(), mean=distribution.mean.tolist(),
                covariance=distribution.covariance.tolist(),
                mean_sha256=_hash(distribution.mean), covariance_sha256=_hash(distribution.covariance),
                full_position_orientation_cross_retained=True)
        return dict(factor_domain='root_to_hand_pose', hand_dimension=6,
            hand_path=[list(item) if not isinstance(item, str) else [item, False] for item in self.hand_path],
            body_edge_ids=list(self.body_edge_ids), intermediate_link_density_scored=False,
            chart_measure='translation Lebesgue times common Haar rotation volume; every prior and posterior has its own modal right chart',
            denominator_semantics='push-forwards of all actual weighted transition-prior body laws',
            numerator_semantics=('push-forward of the shared canonical single internal body law' if self.posterior_branch is not None
                                 else 'push-forwards of all frozen globally weighted posterior body laws'),
            approximation='same existing Bingham-derived deterministic pose moments; endpoint marginal ratio is not the original full-body likelihood ratio',
            physical_truth_received=False, monte_carlo_used=False,
            branches=[dict(label=branch.label, weight=branch.weight, log_weight=_weight(branch),
                source_state_id=branch.reference_state.state_id,
                prior=query_record(query, prior),
                components=[dict(label=child.label, weight=child.weight, log_weight=_weight(child),
                    **query_record(child_query, distribution))
                    for (child, distribution), child_query in zip(children, queries)])
                for branch, (prior, children, _), query, queries in zip(
                    self.branches, self.marginals, self.prior_queries, self.child_queries)])
