"""Density queries for the original Bingham/conditional-Gaussian point law."""
import numpy as np
from functools import lru_cache
from scipy.special import logsumexp

from probtf.bingham import _rotation_quadratic_forms, bingham_log_normalizer, bingham_second_moment
from probtf.distributions import BinghamOrientation, DistributionStatus, OrientationKind, TransformDistributionStamped
from probtf.geometry import quat_left_matrix, quat_to_rotmat, rotation_action_matrix
from probtf.probability.quadrature import (
    METHOD, _equivariant_eigenframe, _order, _orientation_rule, orientation_quadrature, orientation_quadrature_diagnostics,
    quaternion_rotation_matrices,
)
from probtf.provenance import ApproximationInfo, ApproximationKind


_ROTATION_FORMS = np.asarray(_rotation_quadratic_forms()).reshape(3, 3, 4, 4).transpose(1, 0, 2, 3).reshape(9, 4, 4)
ADAPTIVE_METHOD = "point_conditioned_full_rp3_importance_cubature_v1"
NORMALIZER_STEPS = 480


class DensityIntegrationError(RuntimeError):
    """A requested numerical density evaluation could not be established."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


@lru_cache(maxsize=12)
def _hermite_tensor(order):
    values, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3) * np.sqrt(2.)
    log_weights = np.log(np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1).reshape(-1)) - 1.5*np.log(np.pi)
    return nodes, log_weights


def _proposal_rule(parameter, mode, order):
    eigenvalues, frame = _equivariant_eigenframe(parameter, mode)
    eigenvalues -= eigenvalues[-1]
    if -2*eigenvalues[-2] > 50.:
        # Gaussian integration coordinates over the entire R3 chart. The
        # original quaternion exponent and exact chart Jacobian remain in
        # the integrand: this does not replace the law by a local Gaussian.
        scale = 1. / np.sqrt(-2*eigenvalues[:3])
        nodes, log_weights = _hermite_tensor(order)
        chart = nodes * scale
        denominator = 1. + np.sum(chart**2, axis=1)
        local = np.column_stack((chart, np.ones(len(chart)))) / np.sqrt(denominator[:, None])
        quaternions = local @ frame.T
        log_surface = (np.log(2.) + np.log(scale).sum() - 2*np.log(denominator)
                       + 1.5*np.log(2*np.pi) + .5*np.sum(nodes**2, axis=1))
        exponent = np.einsum("ni,ij,nj->n", quaternions, parameter, quaternions)
        return (quaternions, log_weights + log_surface + exponent, 0., parameter,
                "full_rp3_gauss_hermite_change_of_variables", 0.)
    proposal = BinghamOrientation.from_parameter_matrix(parameter, mode)
    parameter = proposal.backend_parameter_matrix()
    nodes, log_weights, _mass = _orientation_rule(proposal, order, logarithmic=True)
    normalizer = bingham_log_normalizer(parameter, NORMALIZER_STEPS)
    return (nodes, log_weights, normalizer, parameter,
            "full_rp3_four_chart_bingham_importance_gauss_legendre", 0.)


def _stationary_mode(parameter, whitened_forms, target, initial):
    """Newton optimization of the exact quartic exponent on the unit sphere."""
    q = np.array(initial, copy=True)
    def values(quaternion):
        products = whitened_forms @ quaternion
        residual = target - products @ quaternion
        value = quaternion @ parameter @ quaternion - .5 * np.dot(residual, residual)
        gradient = 2 * (parameter @ quaternion + residual @ products)
        hessian = 2 * parameter + 2 * np.einsum("i,ijk->jk", residual, whitened_forms) - 4 * products.T @ products
        return value, gradient, hessian
    for _iteration in range(40):
        value, gradient, hessian = values(q)
        tangent = quat_left_matrix(q)[:, 1:]
        projected = tangent.T @ gradient
        curvature = tangent.T @ (np.dot(q, gradient) * np.eye(4) - hessian) @ tangent
        if np.linalg.norm(projected) <= 1e-7 * max(1., np.sqrt(np.linalg.norm(curvature))):
            eigenvalues, eigenvectors = np.linalg.eigh(curvature)
            if eigenvalues[0] >= -1e-8 * max(1., np.linalg.norm(curvature)):
                break
            # A stationary saddle is not an integration mode.
            q += .2 * (tangent @ eigenvectors[:, 0])
            q /= np.linalg.norm(q)
            continue
        smallest = np.linalg.eigvalsh(curvature)[0]
        direction = np.linalg.solve(curvature + max(0., 1. - smallest) * np.eye(3), projected)
        direction *= min(1., .5 / max(np.linalg.norm(direction), 1e-15))
        for fraction in .5 ** np.arange(20):
            trial = q + fraction * (tangent @ direction)
            trial /= np.linalg.norm(trial)
            if values(trial)[0] >= value + 1e-4 * fraction * np.dot(projected, direction) - 1e-9:
                q = trial
                break
        else:
            break
    value, gradient, hessian = values(q)
    tangent = quat_left_matrix(q)[:, 1:]
    curvature = tangent.T @ (np.dot(q, gradient) * np.eye(4) - hessian) @ tangent
    eigenvalues, vectors = np.linalg.eigh(curvature)
    # A Bingham proposal supplies only an integration coordinate system.
    # Its exponent's tangent Hessian is -2 * these nonnegative eigenvalues.
    precision = np.maximum(eigenvalues, 0.) / 2
    axes = tangent @ vectors
    proposal_parameter = -(axes * precision) @ axes.T
    proposal_parameter = .5 * (proposal_parameter + proposal_parameter.T)
    return value, q, proposal_parameter, float(np.linalg.norm(tangent.T @ gradient))


class ConditionalPointDensity:
    """Numerically evaluate the coupled point-action density, retaining its law.

    Given a source pose and fixed child point z, integrate the Gaussian with
    mean m-C*rho_ref+(B_z+C)*rho(q) over the source Bingham law. Quadrature
    nodes remain internal evaluation data and never replace ``source_record``.
    Positive-definite residual covariance is required for a Lebesgue density.
    """
    def __init__(self, record, point=(0., 0., 0.), order=9):
        if not isinstance(record, TransformDistributionStamped):
            raise TypeError("record must be TransformDistributionStamped")
        self.source_record = record
        self.point = np.asarray(point, dtype=float).copy()
        if self.point.shape != (3,) or not np.all(np.isfinite(self.point)):
            raise ValueError("point must contain three finite coordinates")
        self.point.setflags(write=False)
        self.order = _order(order)
        normalized = record.distribution.normalize_weights()
        if normalized.status is not DistributionStatus.OK:
            raise ValueError("point density requires a valid positive-mass source law")
        self.terms = []
        self.component_diagnostics = []
        self._components = []
        approximate = False
        for item in normalized.components:
            component = item.component
            covariance = component.translation.residual_covariance
            try:
                cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError("a point density requires positive-definite conditional residual covariance") from error
            quaternions, weights = orientation_quadrature(component.orientation, self.order)
            # Discard only arithmetic underflow zeros, never positive mass.
            positive = weights > 0
            quaternions, weights = quaternions[positive], weights[positive]
            rotations = quaternion_rotation_matrices(quaternions)
            rho = rotations.transpose(0, 2, 1).reshape(-1, 9)
            reference = quat_to_rotmat(component.orientation.reference_quaternion_wxyz).reshape(9, order="F")
            means = (np.einsum("nij,j->ni", rotations, self.point)
                     + component.translation.mean_at_reference
                     + (rho - reference) @ component.translation.rotation_coupling.T)
            log_normalizer = -1.5 * np.log(2*np.pi) - np.log(np.diag(cholesky)).sum()
            for array in (means, cholesky):
                array.setflags(write=False)
            self.terms.append((item.weight, means, weights, cholesky, float(log_normalizer)))
            inverse_cholesky = np.linalg.inv(cholesky)
            forms = np.einsum("ij,jkl->ikl", rotation_action_matrix(self.point)
                              + component.translation.rotation_coupling, _ROTATION_FORMS)
            offset = component.translation.mean_at_reference - component.translation.rotation_coupling @ reference
            parameter = component.orientation.backend_parameter_matrix() if component.orientation.kind is not OrientationKind.DIRAC else None
            seeds = orientation_quadrature(component.orientation, 5)[0]
            self._components.append(dict(weight=item.weight, orientation=component.orientation,
                parameter=parameter, offset=offset, forms=forms,
                whitened_forms=np.einsum("ij,jkl->ikl", inverse_cholesky, forms),
                inverse_cholesky=inverse_cholesky, log_normalizer=float(log_normalizer),
                log_orientation_normalizer=(0. if parameter is None else bingham_log_normalizer(parameter, NORMALIZER_STEPS)),
                seeds=seeds, warm=None))
            self.component_diagnostics.append(dict(component_id=component.component_id,
                active_node_count=len(weights),
                **orientation_quadrature_diagnostics(component.orientation, self.order)))
            approximate |= component.orientation.kind is not OrientationKind.DIRAC
        self.terms = tuple(self.terms)
        self.component_diagnostics = tuple(self.component_diagnostics)
        self.approximation = (ApproximationInfo(kind=ApproximationKind.NUMERICAL_INTEGRATION, lossy=True,
            detail="{}; order={}; original source law retained; no certified density error bound".format(ADAPTIVE_METHOD, self.order),
            source="probtf.probability.ConditionalPointDensity") if approximate else ApproximationInfo())
        self.metadata = dict(method=ADAPTIVE_METHOD if approximate else "exact_conditional_gaussian",
            order=self.order, source_edge_id=record.edge_id, source_stamp=record.stamp,
            prior_quadrature_reference=self.component_diagnostics, error_bound=None)
        self.last_diagnostics = ()
        self.metadata.update(normalization="original Bingham Z retained; exact Jacobians for Gaussian chart coordinates; independent proposal Z for bounded Bingham charts",
            normalizer_integration_steps=NORMALIZER_STEPS,
            proposal="stationary modes and Hessians of full Bingham times conditional Gaussian integrand; multiple charts use partition of unity",
            partition="positive unnormalized kernels exp(q^T B_i q) / sum_j exp(q^T B_j q); no proposal normalizer needed for partition",
            gradient_integration="differentiate original conditional Gaussian under the integral; no law closure",
            mode_search="at most five separated full-sphere seeds, including the preceding mode; global completeness is not certified")

    def _conditional_integral(self, component, point):
        inverse_cholesky = component["inverse_cholesky"]
        target = inverse_cholesky @ (point - component["offset"])
        orientation = component["orientation"]
        if orientation.kind is OrientationKind.DIRAC:
            q = orientation.reference_quaternion_wxyz
            residual = target - np.einsum("i,kij,j->k", q, component["whitened_forms"], q)
            return (component["log_normalizer"] - .5 * np.dot(residual, residual),
                    -residual @ inverse_cholesky, dict(method="exact_conditional_gaussian", modes=1))
        parameter, forms = component["parameter"], component["whitened_forms"]
        seeds = component["seeds"]
        residuals = target - np.einsum("ni,kij,nj->nk", seeds, forms, seeds)
        seed_values = np.einsum("ni,ij,nj->n", seeds, parameter, seeds) - .5 * np.sum(residuals**2, axis=1)
        starts = [] if component["warm"] is None else [component["warm"]]
        for index in np.argsort(seed_values)[::-1]:
            candidate = seeds[index]
            if not starts or max(abs(np.dot(candidate, start)) for start in starts) < .98:
                starts.append(candidate)
            if len(starts) >= 5:
                break
        modes = []
        for start in starts:
            mode = _stationary_mode(parameter, forms, target, start)
            if not modes or max(abs(np.dot(mode[1], other[1])) for other in modes) < 1 - 1e-7:
                modes.append(mode)
        modes.sort(key=lambda item: item[0], reverse=True)
        component["warm"] = modes[0][1].copy()
        proposals = []
        for _value, mode, proposal_parameter, _gradient in modes:
            proposals.append(_proposal_rule(proposal_parameter, mode, self.order))
        all_scores, all_gradients, corrections = [], [], []
        for proposal_index, (nodes, log_weights, log_z, proposal_parameter, integration_kind, partition_z) in enumerate(proposals):
            values = np.einsum("ni,ij,nj->n", nodes, parameter, nodes)
            residual = target - np.einsum("ni,kij,nj->nk", nodes, forms, nodes)
            exponent = values - .5 * np.sum(residual ** 2, axis=1)
            proposal_exponents = np.array([np.einsum("ni,ij,nj->n", nodes, other[3], nodes) for other in proposals])
            # A partition of unity needs positive kernels, not normalized
            # proposal laws. The original source normalizer stays unchanged.
            proposal_log_kernels = proposal_exponents - np.array([other[5] for other in proposals])[:, None]
            partition = proposal_log_kernels[proposal_index] - logsumexp(proposal_log_kernels, axis=0)
            all_scores.append(log_weights + log_z + exponent - proposal_exponents[proposal_index] + partition
                              - component["log_orientation_normalizer"] + component["log_normalizer"])
            gradients = -residual @ inverse_cholesky
            all_gradients.append(gradients)
            # The spatial Gaussian score is quadratic in q. Integrate that
            # part exactly under the importance proposal and use quadrature
            # only for its nonconstant likelihood-ratio correction.
            if integration_kind == "full_rp3_four_chart_bingham_importance_gauss_legendre":
                moment = bingham_second_moment(proposal_parameter, NORMALIZER_STEPS)
                exact_score_mean = -(target - np.einsum("kij,ij->k", forms, moment)) @ inverse_cholesky
                numerical_score_mean = np.exp(log_weights) @ gradients
                mode = modes[proposal_index][1]
                mode_proposals = np.array([mode @ other[3] @ mode - other[5] for other in proposals])
                log_constant = (log_z + modes[proposal_index][0] - partition_z - logsumexp(mode_proposals)
                                - component["log_orientation_normalizer"] + component["log_normalizer"])
                corrections.append((log_constant, exact_score_mean - numerical_score_mean))
        scores, gradients = np.concatenate(all_scores), np.concatenate(all_gradients)
        log_density = float(logsumexp(scores))
        if not np.isfinite(log_density):
            raise DensityIntegrationError("point-density integral did not yield a finite log density",
                                          dict(point=point.tolist(), order=self.order))
        gradient = np.exp(scores - log_density) @ gradients
        for log_constant, correction in corrections:
            gradient += np.exp(log_constant - log_density) * correction
        diagnostic = dict(method=ADAPTIVE_METHOD, order=self.order, modes=len(modes),
            maximum_stationary_gradient=max(mode[3] for mode in modes),
            mode_quaternions=[mode[1].tolist() for mode in modes],
            proposal_parameters=[proposal[3].tolist() for proposal in proposals],
            integration_rules=[proposal[4] for proposal in proposals],
            integration_node_counts=[len(proposal[0]) for proposal in proposals], error_bound=None)
        return log_density, gradient, diagnostic

    def log_density_and_gradient(self, points):
        """Return log density and spatial gradient of log density at each point."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must have a final dimension of three finite coordinates")
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        unique, reverse = np.unique(flat, axis=0, return_inverse=True)
        values, gradients, diagnostics = [], [], []
        for point in unique:
            component_results = [self._conditional_integral(component, point) for component in self._components]
            scores = np.array([np.log(component["weight"]) + result[0]
                               for component, result in zip(self._components, component_results)])
            value = float(logsumexp(scores))
            values.append(value)
            gradients.append(np.exp(scores - value) @ np.array([result[1] for result in component_results]))
            diagnostics.append(dict(point=point.tolist(), components=[result[2] for result in component_results]))
        self.last_diagnostics = tuple(diagnostics)
        return np.asarray(values)[reverse].reshape(shape), np.asarray(gradients)[reverse].reshape(shape + (3,))

    def log_density(self, points):
        return self.log_density_and_gradient(points)[0]

    def density(self, points):
        return np.exp(self.log_density(points))

    def refinement_diagnostics(self, points, refined_order=None, *, log_tolerance=None, gradient_tolerance=None):
        """Compare two actual evaluations; report differences, not a bound."""
        refined_order = self.order + 2 if refined_order is None else _order(refined_order)
        if refined_order <= self.order:
            raise ValueError("refined_order must exceed the initial order")
        refined = ConditionalPointDensity(self.source_record, self.point, refined_order)
        before, before_gradient = self.log_density_and_gradient(points)
        after, after_gradient = refined.log_density_and_gradient(points)
        diagnostics = dict(method=ADAPTIVE_METHOD, order=self.order, refined_order=refined_order,
                    maximum_log_density_change=float(np.max(np.abs(after - before))),
                    maximum_log_gradient_change=float(np.max(np.abs(after_gradient - before_gradient))),
                    error_bound=None, query_diagnostics=refined.last_diagnostics)
        if ((log_tolerance is not None and diagnostics["maximum_log_density_change"] > log_tolerance)
                or (gradient_tolerance is not None and diagnostics["maximum_log_gradient_change"] > gradient_tolerance)):
            raise DensityIntegrationError("point-density quadrature did not meet the requested refinement tolerance", diagnostics)
        return diagnostics
