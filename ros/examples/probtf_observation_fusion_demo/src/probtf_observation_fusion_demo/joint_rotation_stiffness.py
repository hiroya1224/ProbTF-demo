"""K factors on the full joint marginal of local link rotations.

The retained body law and its charts are unchanged. Translation coordinates
are integrated out by Gaussian marginal extraction; the complete rotation
cross-covariance remains. No independent product of Bingham marginals is used.
"""

import math

import numpy as np
from scipy.special import logsumexp

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.support import require_in_support, require_same_support
from .joint_batch_fusion import add_log_evidence_parts, log_evidence_parts_difference
from .stiffness_identification import (
    ChartedBodyDensityRatio, _density_parts, _log_mixture_parts,
    _log_weight, _rotation_chart_volume,
)


class JointRotationDensityRatio(ChartedBodyDensityRatio):
    """Joint 3L Gaussian rotation ratio in each actual frozen source chart.

    ``evaluate`` accepts the existing packed 6L auxiliary-body prediction and
    its Jacobian, or an explicitly rotation-only 3L prediction. Both forms
    evaluate the same Gaussian marginal, never a translation-conditioned law.
    """

    def __init__(self, branches, body_edge_ids, posterior_branch=None):
        super().__init__(branches, body_edge_ids, posterior_branch=posterior_branch,
                         coordinate_offsets=(3, 4, 5), require_full_rank_mixture=False)
        self.rotation_dimension = 3*len(self.body_edge_ids)
        self.prediction_indices = np.asarray([6*i+coordinate
            for i in range(len(self.body_edge_ids)) for coordinate in (3, 4, 5)], dtype=int)
        self.source_rotation_indices = tuple(tuple(6*branch.reference_state.edge_ids.index(name)+coordinate
            for name in self.body_edge_ids for coordinate in (3, 4, 5)) for branch in self.branches)
        def charts(branch):
            by_id = {edge.edge_id: edge for edge in branch.reference_state.transforms}
            return tuple(quat_to_rotmat(by_id[name].rotation_wxyz) for name in self.body_edge_ids)
        reference_charts = charts(self.branches[0])
        self.common_rotation_chart = all(all(np.array_equal(left, right)
            for left, right in zip(reference_charts, charts(branch)))
            for branch in self.branches[1:])
        singular = any(support.rank != prior.mean.size for prior, _, support in self.marginals)
        if singular and len(self.branches) > 1:
            if not self.common_rotation_chart:
                raise ValueError('Different singular rotation charts need an explicit common physical support measure.')
            reference_prior, _, reference_support = self.marginals[0]
            for prior, _, support in self.marginals[1:]:
                require_same_support(reference_support, support,
                                     'common-chart joint-rotation mixture')
                require_in_support(prior.mean-reference_prior.mean, reference_support,
                                   'common-chart joint-rotation affine support')

    def _prediction(self, prediction):
        value, jacobian = map(lambda item: np.asarray(item, dtype=float), prediction[:2])
        if (value.ndim != 1 or jacobian.ndim != 2 or jacobian.shape[0] != value.size
                or not np.all(np.isfinite(value)) or not np.all(np.isfinite(jacobian))):
            raise ValueError('A joint-rotation prediction and its Jacobian must have finite matching dimensions.')
        if value.size == 6*len(self.body_edge_ids):
            return value[self.prediction_indices], jacobian[self.prediction_indices]
        if value.size == self.rotation_dimension:
            return value, jacobian
        raise ValueError('A joint-rotation factor requires the original packed 6L or rotational 3L prediction.')

    def _volume(self, value):
        # Reuse exactly the existing product-SO(3) chart volume convention,
        # with zero unused translation slots. No new orientation law is added.
        full = np.zeros(6*len(self.body_edge_ids))
        full[self.prediction_indices] = value
        volume, gradient = _rotation_chart_volume(full)
        return volume, gradient[self.prediction_indices]

    def evaluate(self, predictions):
        predictions = tuple(predictions)
        if len(predictions) != len(self.branches):
            raise ValueError('Every frozen rotation chart needs the same physical local-rotation prediction.')
        minus_terms, plus_terms, minus_gradients, plus_gradients = [], [], [], []
        parameter_dimension = None
        for branch, marginal, prediction, child_supports in zip(
                self.branches, self.marginals, predictions, self.child_supports):
            prior, children, support = marginal
            value, jacobian = self._prediction(prediction)
            if parameter_dimension is None:
                parameter_dimension = jacobian.shape[1]
            elif parameter_dimension != jacobian.shape[1]:
                raise ValueError('All frozen-chart stiffness derivatives need the same parameter dimension.')
            if self.identity:
                require_in_support(value-prior.mean, support, 'identity joint-rotation ratio support')
                continue
            # One common chart has an identical volume factor in every term,
            # including mixtures supported on one common affine subspace.
            # Different full-rank charts need their own factor inside LSE.
            volume, volume_gradient = ((0., np.zeros_like(value)) if self.common_rotation_chart
                                        else self._volume(value))
            log_minus, score_minus = _density_parts(prior, value, support)
            minus_terms.append(add_log_evidence_parts(log_minus, (_log_weight(branch), -volume)))
            minus_gradients.append(jacobian.T @ (score_minus-volume_gradient))
            for (child, distribution), child_support in zip(children, child_supports):
                log_plus, score_plus = _density_parts(distribution, value, child_support)
                plus_terms.append(add_log_evidence_parts(log_plus, (_log_weight(child), -volume)))
                plus_gradients.append(jacobian.T @ (score_plus-volume_gradient))
        if self.identity:
            return 0., np.zeros(parameter_dimension)
        minus, minus_responsibility = _log_mixture_parts(minus_terms)
        plus, plus_responsibility = _log_mixture_parts(plus_terms)
        return (log_evidence_parts_difference(plus, minus),
            plus_responsibility @ np.asarray(plus_gradients)
            -minus_responsibility @ np.asarray(minus_gradients))

    def properness_bounds(self, *, model_support_verified=False):
        """Bounds on supported principal rotations; expose global proof needs.

        Singular Gaussian ratios are evaluated on their actual shared affine
        support. A compact norm bound alone cannot establish that every model
        prediction belongs to that support. Keep the valid conditional bounds
        and report this missing model-specific proof without fabricating an
        inverse, broadening support, or rejecting a finite supported query.
        """
        if self.identity:
            return dict(verified=True, log_factor_lower_bound=0., log_factor_upper_bound=0.,
                        argument='declared exact identity joint-rotation density ratio')
        lower_minus, upper_minus, lower_plus, upper_plus = [], [], [], []
        singular = any(distribution.support.rank != distribution.mean.size
            for prior, children, _ in self.marginals
            for distribution in (prior,)+tuple(child for _, child in children))
        volume_bound = (len(self.body_edge_ids)*2.*math.log(math.pi/2.)
                        if not self.common_rotation_chart else 0.)
        def bounds(distribution):
            support = distribution.support
            radius_squared = sum((math.pi+np.linalg.norm(block))**2
                                 for block in distribution.mean.reshape(-1, 3))
            maximum = -.5*(support.rank*math.log(2.*math.pi)+support.log_pdet)
            inverse_norm = float(np.max(np.linalg.eigvalsh(support.inverse), initial=0.))
            minimum = maximum-.5*inverse_norm*radius_squared
            return float(minimum), float(maximum+volume_bound)
        for branch, (prior, children, _) in zip(self.branches, self.marginals):
            low, high = bounds(prior)
            lower_minus.append(_log_weight(branch)+low); upper_minus.append(_log_weight(branch)+high)
            for child, distribution in children:
                low, high = bounds(distribution)
                lower_plus.append(_log_weight(child)+low); upper_plus.append(_log_weight(child)+high)
        lower = float(logsumexp(lower_plus)-logsumexp(upper_minus))
        upper = float(logsumexp(upper_plus)-logsumexp(lower_minus))
        if not np.isfinite(lower+upper):
            raise ValueError('The joint-rotation factor compact-domain bounds are nonfinite.')
        return dict(verified=not singular or bool(model_support_verified),
            log_factor_lower_bound=lower, log_factor_upper_bound=upper,
            argument='joint rotation Gaussian density bounds on the actual supported subset of the compact product of principal local-rotation charts',
            conditional_on_model_support=singular,
            global_model_support_proof_required=singular and not bool(model_support_verified),
            model_support_verified=bool(model_support_verified),
            translation_density_scored=False, inter_link_rotation_cross_retained=True,
            rotation_dimension=self.rotation_dimension,
            density_domain='existing Bingham-derived joint Gaussian proxy; not an exact joint Bingham distribution')

    def record(self):
        record = super().record()
        record.update(factor_domain='joint_local_rotations',
            rotation_dimension=self.rotation_dimension, body_edge_ids=list(self.body_edge_ids),
            coordinate_offsets=[3, 4, 5],
            common_rotation_chart=self.common_rotation_chart,
            common_affine_rotation_support_verified=True,
            translation_density_scored=False, translation_conditioning_used=False,
            inter_link_rotation_cross_retained=True,
            nuisance_policy='all local translations, their rotation cross blocks, unknown marker and other variables integrated out by Gaussian marginal extraction; no Schur complement',
            chart_measure=('one common rotation chart and common affine-support volume; its common Jacobian cancels from the ratio'
                if self.common_rotation_chart else
                'common product Haar rotation volume; each full-rank native chart correction remains inside log-sum-exp'),
            approximation='full joint rotational marginal of the existing Bingham-derived Gaussian proxy; not an independent product of Bingham marginals',
            prediction_coordinate_indices=self.prediction_indices.tolist(),
            independent_link_product_used=False, physical_truth_received=False, monte_carlo_used=False)
        for branch_record, indices in zip(record['branches'], self.source_rotation_indices):
            branch_record['source_rotation_coordinate_indices'] = list(indices)
        return record
