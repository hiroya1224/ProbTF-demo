"""Auxiliary log-stiffness inference from the frozen body density ratio.

The primary body law is never replaced by an equilibrium prediction.  All
body mixture components and cross covariances enter the density ratio. The
continuous auxiliary log-K density retains every new factor without sampling
or a Gaussian projection. Finite numerical trials serve initial proposals only.
"""

from pathlib import Path
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np
import rospkg
import yaml
from scipy.special import logsumexp

from deflecomp_core.estimator.initialization import initial_log_kp_state, initial_log_kp_std
from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import LinearSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf.geometry import quat_to_rotmat, skew
from .body_native import reexpress_state, materialize_state_in_chart
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf.spherical_law.support import (
    require_in_support, require_same_support,
)
from scipy.spatial.transform import Rotation

from .stiffness_bayes_factor import (
    ContinuousStiffnessBFS, LocalLaplaceEvidence, ResetMeasure,
    StiffnessEvidenceUnavailable, update_hazard_log_probabilities,
)
from .joint_batch_fusion import add_log_evidence_parts, log_evidence_parts_difference


def _runtime_stage(owner, name, **context):
    monitor = getattr(owner, 'runtime_monitor', None)
    return nullcontext() if monitor is None else monitor.stage(name, **context)


def _stiffness_surprise_settings(settings):
    supplied = settings.get('bayes_factor_surprise', {})
    if not isinstance(supplied, dict):
        raise ValueError('stiffness bayes_factor_surprise must be a mapping.')
    enabled = supplied.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('stiffness bayes_factor_surprise.enabled must be a boolean.')
    probability = float(supplied.get('change_probability', .10))
    ContinuousStiffnessBFS._validate_rho(probability)
    if enabled and probability > 0.:
        required = dict(reset_measure='initial_log_stiffness',
            event_probability_unit='per_effective_sensor_event', evidence_method='laplace')
        missing = [name for name in required if supplied.get(name) is None]
        if missing:
            raise ValueError('Positive stiffness Bayes surprise requires explicit configuration: '+
                ', '.join('bayes_factor_surprise.'+name+'='+required[name] for name in missing)+'.')
        for name, expected in required.items():
            if supplied[name] != expected:
                raise ValueError('Unsupported stiffness bayes_factor_surprise.'+name+
                                 '; the implemented explicit option is '+expected+'.')
    return dict(supplied, enabled=enabled, change_probability=probability)


def _density_parts(distribution, value, support=None):
    support = distribution.support if support is None else support
    error = np.asarray(value)-distribution.mean
    require_in_support(error, support, 'auxiliary prediction in frozen body support')
    score = -support.inverse @ error
    parts = add_log_evidence_parts((0., 0.),
        (-.5*(support.rank*math.log(2.*math.pi)+support.log_pdet),
         .5*float(error @ score)))
    return parts, score


def _density(distribution, value, support=None):
    parts, score = _density_parts(distribution, value, support)
    return parts[0]+parts[1], score


def _log_mixture_parts(parts):
    reference = max(parts, key=lambda item:(item[0], item[1]))
    shifted = np.asarray([log_evidence_parts_difference(item, reference) for item in parts])
    normalizer = float(logsumexp(shifted))
    return add_log_evidence_parts(reference, (normalizer,)), np.exp(shifted-normalizer)


def _mixture_density(components, value, supports=None, *, return_parts=False):
    evaluations = [_density_parts(component.distribution, value,
        None if supports is None else supports[index]) for index, component in enumerate(components)]
    terms = [add_log_evidence_parts(item[0], (_log_weight(component),))
             for component, item in zip(components, evaluations)]
    total_parts, responsibility = _log_mixture_parts(terms)
    total = total_parts[0]+total_parts[1]
    score = sum(weight*item[1] for weight, item in zip(responsibility, evaluations))
    component_logs = ([item[0] for item in evaluations] if return_parts else
                      [item[0][0]+item[0][1] for item in evaluations])
    return total_parts if return_parts else total, score, component_logs, responsibility


def _log_weight(component):
    """Do not drop an existing branch whose ordinary weight underflows."""
    supplied = getattr(component, 'log_weight', None)
    return (float(supplied) if supplied is not None else
            -np.inf if component.weight == 0 else math.log(component.weight))


def _continuous_initial_density(components, x):
    value, score, _, _ = _mixture_density(components, np.asarray(x, dtype=float))
    return value, score


def _initial_density_id(components):
    identity = hashlib.sha256()
    for item in components:
        identity.update(item.distribution.mean.tobytes()+item.distribution.covariance.tobytes())
        identity.update(repr(_log_weight(item)).encode())
    return identity.hexdigest()


def _same_mixture(left, right):
    # Labels are provenance, not additional random physical variables.
    def masses(items):
        groups = {}
        for item in items:
            key = (item.distribution.mean.tobytes(), item.distribution.covariance.tobytes())
            groups.setdefault(key, []).append(_log_weight(item))
        return {key: float(logsumexp(values)) for key, values in groups.items()}
    return masses(left) == masses(right)


class BodyDensityRatio:
    """A marginal of the actual raw Gaussians, in their common frozen chart."""

    def __init__(self, prior, posterior_components, indices):
        indices = np.asarray(indices, dtype=int)
        if isinstance(prior, Gaussian):
            prior = (PosteriorComponent('frozen_body_prior', prior),)
        def marginal(item):
            kwargs = ({} if getattr(item, 'log_weight', None) is None else
                      {'log_weight': item.log_weight})
            return PosteriorComponent(item.label, Gaussian(item.distribution.mean[indices],
                item.distribution.covariance[np.ix_(indices, indices)]),
                item.weight, item.lineage, **kwargs)
        self.prior_components = tuple(marginal(item) for item in prior)
        self.prior = self.prior_components[0].distribution
        self.components = tuple(marginal(item) for item in posterior_components)
        if not self.prior_components or not np.isclose(
                logsumexp([_log_weight(item) for item in self.prior_components]), 0.):
            raise ValueError('The supplied prior body weights must be normalized.')
        if not self.components or not np.isclose(sum(item.weight for item in self.components), 1.):
            raise ValueError('The supplied fused body weights must be normalized.')
        for item in self.prior_components+self.components:
            require_same_support(self.prior.support, item.distribution.support,
                                 'frozen body prior/posterior')
            require_in_support(item.distribution.mean-self.prior.mean, self.prior.support,
                               'frozen body posterior affine support')
        self.prior_support = self.prior.support
        self.prior_supports = tuple(item.distribution.support for item in self.prior_components)
        self.component_supports = tuple(item.distribution.support for item in self.components)
        self.identity = _same_mixture(self.prior_components, self.components)

    def evaluate(self, value, branch=None):
        if self.identity:
            require_in_support(np.asarray(value)-self.prior.mean, self.prior_support,
                               'identity body density ratio support')
            return 0., np.zeros_like(self.prior.mean), [0.]*len(self.components)
        minus, minus_score, _, _ = _mixture_density(self.prior_components, value,
            self.prior_supports, return_parts=True)
        if branch is None:
            plus, plus_score, component_logs, _ = _mixture_density(self.components, value,
                self.component_supports, return_parts=True)
        else:
            plus, plus_score = _density_parts(self.components[branch].distribution, value, self.component_supports[branch])
            component_logs = [plus]
        return (log_evidence_parts_difference(plus, minus), plus_score-minus_score,
                [log_evidence_parts_difference(item, minus) for item in component_logs])


def _rotation_chart_volume(value):
    """Log volume of the rotation-vector chart relative to one Haar measure.

    The constant quaternion half-angle volume cancels between all components.
    This is a density coordinate change, not an assumed Lie Gaussian law.
    The underlying Gaussian proxies are still the given Bingham-derived laws.
    """
    value = np.asarray(value, dtype=float)
    gradient = np.zeros_like(value)
    log_volume = 0.
    for offset in range(3, value.size, 6):
        vector = value[offset:offset+3]
        angle = float(np.linalg.norm(vector))
        if angle > math.pi+1e-10:
            raise ValueError('A body density query left the principal rotation chart.')
        if angle < 1e-5:
            log_volume -= angle**2/12.+angle**4/1440.
            gradient[offset:offset+3] = -(1./6.+angle**2/360.)*vector
        else:
            log_volume += 2.*math.log(math.sin(.5*angle)/(.5*angle))
            gradient[offset:offset+3] = (1./math.tan(.5*angle)-2./angle)*vector/angle
    return float(log_volume), gradient


class ChartedBodyDensityRatio:
    """Body densities, retaining each denominator's actual frozen chart.

    Every numerator and denominator is converted to the same physical volume
    measure before log-sum-exp. An explicitly supplied posterior branch is the
    already declared single-law internal approximation used by IK/display;
    it does not replace the native transition-prior mixture in the denominator.
    General singular mixtures require an explicit common manifold measure;
    this implementation refuses to invent one or add jitter.
    """

    def __init__(self, branches, body_edge_ids, posterior_branch=None):
        self.branches = tuple(branches)
        self.body_edge_ids = tuple(body_edge_ids)
        self.posterior_branch = posterior_branch
        posterior_index = None
        if posterior_branch is not None:
            matches = [index for index, branch in enumerate(self.branches)
                       if branch.reference_state.state_id == posterior_branch.reference_state.state_id]
            if len(matches) != 1:
                raise ValueError('The projected body numerator must use one actual transition-prior chart.')
            posterior_index = matches[0]
            reference = self.branches[posterior_index].reference_state
            if (not np.array_equal(posterior_branch.prior.mean, reference.mean)
                    or not np.array_equal(posterior_branch.prior.covariance, reference.covariance)):
                raise ValueError('The projected numerator reference does not match its native prior chart.')
            if len(posterior_branch.posterior_components) != 1:
                raise ValueError('The projected body numerator must be the declared single internal law.')
        self.marginals = []
        self.child_supports = []
        self.identity = True
        for index, branch in enumerate(self.branches):
            state = branch.reference_state
            indices = [6*state.edge_ids.index(name)+coordinate
                       for name in body_edge_ids for coordinate in range(6)]
            def marginal(distribution):
                return Gaussian(distribution.mean[indices],
                    distribution.covariance[np.ix_(indices, indices)])
            prior = marginal(branch.prior)
            sources = (branch.posterior_components if posterior_branch is None else
                       posterior_branch.posterior_components if index == posterior_index else ())
            children = tuple((child, marginal(child.distribution)) for child in sources)
            support = prior.support
            for _, child in children:
                require_same_support(support, child.support, 'same-branch body mixture')
                require_in_support(child.mean-prior.mean, support, 'same-branch affine support')
            if len(self.branches) > 1 and support.rank != prior.mean.size:
                raise ValueError('Different singular body charts need a supplied common physical support measure.')
            equal = all(np.array_equal(child.mean, prior.mean)
                        and np.array_equal(child.covariance, prior.covariance)
                        for _, child in children)
            mass = (float(logsumexp([_log_weight(child) for child, _ in children]))
                    if children else -np.inf)
            # L=1 is an exact algebraic shortcut. A small finite change in
            # mixture mass can change the K likelihood and must not be
            # discarded by an approximate-equality threshold.
            self.identity &= equal and mass == _log_weight(branch)
            self.marginals.append((prior, children, support))
            self.child_supports.append(tuple(child.support for _, child in children))
        if not self.branches:
            raise ValueError('The actual frozen body branch correspondence is required.')
        for weights in ([_log_weight(branch) for branch in self.branches],
                        [_log_weight(child) for _, children, _ in self.marginals
                         for child, _ in children]):
            if not np.isclose(logsumexp(weights), 0., atol=1e-10, rtol=0.):
                raise ValueError('Body prior/posterior global mixture weights must be normalized.')

    def evaluate(self, predictions):
        if len(predictions) != len(self.branches):
            raise ValueError('Every transition-prior chart needs the same physical prediction.')
        minus_terms, plus_terms, minus_gradients, plus_gradients = [], [], [], []
        for branch, marginal, prediction, child_supports in zip(
                self.branches, self.marginals, predictions, self.child_supports):
            prior, children, support = marginal
            value, jacobian = prediction[:2]
            if self.identity:
                require_in_support(value-prior.mean, support, 'identity body mixture support')
                continue
            volume, volume_gradient = ((0., np.zeros_like(value)) if len(self.branches) == 1
                                        else _rotation_chart_volume(value))
            log_minus, score_minus = _density_parts(prior, value, support)
            minus_terms.append(add_log_evidence_parts(log_minus, (_log_weight(branch), -volume)))
            minus_gradients.append(jacobian.T @ (score_minus-volume_gradient))
            for (child, distribution), child_support in zip(children, child_supports):
                log_plus, score_plus = _density_parts(distribution, value, child_support)
                plus_terms.append(add_log_evidence_parts(log_plus, (_log_weight(child), -volume)))
                plus_gradients.append(jacobian.T @ (score_plus-volume_gradient))
        if self.identity:
            return 0., np.zeros(predictions[0][1].shape[1])
        minus, minus_responsibility = _log_mixture_parts(minus_terms)
        plus, plus_responsibility = _log_mixture_parts(plus_terms)
        minus_gradient = minus_responsibility @ np.asarray(minus_gradients)
        plus_gradient = plus_responsibility @ np.asarray(plus_gradients)
        return log_evidence_parts_difference(plus, minus), plus_gradient-minus_gradient

    def record(self):
        return dict(chart_measure='common Haar rotation volume and parent-translation Lebesgue volume; single-chart Jacobian cancels',
            nuisance_policy='unknown marker and other variables integrated out by Gaussian marginal extraction',
            denominator_semantics=('native weighted transition-prior mixture in its actual charts'
                if self.posterior_branch is not None else 'actual frozen body prior mixture'),
            numerator_semantics=('declared projected single internal body law shared by estimation, IK and display; not conditional keep-branch children'
                if self.posterior_branch is not None else 'actual frozen globally weighted posterior children'),
            projected_numerator_source_state_id=(None if self.posterior_branch is None else
                self.posterior_branch.reference_state.state_id),
            branches=[dict(label=branch.label, weight=branch.weight, log_weight=_log_weight(branch),
                source_state_id=branch.reference_state.state_id,
                source_version=branch.reference_state.version,
                source_stamp=branch.reference_state.stamp,
                source_command=list(branch.reference_state.command),
                reference_transforms=[dict(edge_id=edge.edge_id, translation=edge.translation.tolist(),
                    rotation_wxyz=edge.rotation_wxyz.tolist(),
                    bingham=None if edge.bingham is None else edge.bingham.tolist())
                    for edge in branch.reference_state.transforms if edge.edge_id in self.body_edge_ids],
                prior_mean=prior.mean.tolist(), prior_covariance=prior.covariance.tolist(),
                components_role=('projected numerator in this reference chart'
                    if self.posterior_branch is not None else 'native globally weighted posterior children'),
                components=[dict(label=child.label, weight=child.weight, log_weight=_log_weight(child),
                    mean=distribution.mean.tolist(), covariance=distribution.covariance.tolist())
                    for child, distribution in children])
                for branch, (prior, children, _) in zip(self.branches, self.marginals)])


@dataclass(frozen=True)
class FrozenStiffnessFactor:
    factor_id: str
    observation_ids: tuple
    ratio: object
    states: tuple
    branch_anchor: object
    properness: dict


@dataclass(frozen=True)
class FrozenTimeIndexedStiffnessFactor:
    """A batch product whose atomic factors keep their acquisition commands.

    The product is one BFS event.  Its individual body-density ratios are
    evaluated at their own ``h(K, q_i)``, not at a transported endpoint pose.
    """
    factor_id: str
    observation_ids: tuple
    parts: tuple
    properness: dict

    @property
    def ratio(self):
        return SimpleNamespace(identity=all(part.ratio.identity for part in self.parts))


def _atomic_stiffness_factors(factors):
    for factor in factors:
        if isinstance(factor, FrozenTimeIndexedStiffnessFactor):
            yield from _atomic_stiffness_factors(factor.parts)
        else:
            yield factor


def _right_log_derivative(vector):
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    coefficient = (1./12.+angle**2/720. if angle < 1e-6 else
                   (1.-.5*angle/math.tan(.5*angle))/angle**2)
    return np.eye(3)+.5*cross+coefficient*(cross @ cross)



class StiffnessIdentifier:
    """One sensor-independent, downstream auxiliary parameter updater.

    URDF inertial data and gravity are explicitly known auxiliary inputs.
    No simulator stiffness, realized joints or marker placement is accepted.
    The initializer's bounds construct its existing Gaussian, without
    truncating its support. No process noise, covariance cap or jitter is used.
    """

    def __init__(self, urdf_path, body_edge_ids, settings):
        self.settings = dict(settings)
        self.stiffness_surprise_settings = _stiffness_surprise_settings(settings)
        self.body_edge_ids = tuple(body_edge_ids)
        self.robot = RobotArm(str(urdf_path), tip_link='tool0', base_link='base_link')
        self.spring = LinearSpringModel()
        self.solver = EquilibriumSolver(self.robot, self.spring,
            EquilibriumConfig(maxiter=100, ftol=1e-14, refine_tol=1e-14))
        # A converged interior root remains free even a few nanoradians from
        # a stop. The Newton candidates are actually projected onto the box;
        # only that real boundary with the outward KKT sign is constrained.
        self.sensitivity = SensitivityCalculator(self.robot, self.spring,
            active_set_tol=0., kkt_tol=0.)
        config_path = settings.get('prior_config_path')
        if config_path is None:
            config_path = Path(rospkg.RosPack().get_path(settings.get(
                'prior_config_package', 'deflecomp_ros')))/settings.get('prior_config_file', 'config/estimator.yaml')
        config_path = Path(config_path)
        existing = yaml.safe_load(config_path.read_text())
        limits = (float(existing['kp_min']), float(existing['kp_max']))
        mean = initial_log_kp_state(self.robot.nv, limits)
        covariance = np.eye(self.robot.nv)*initial_log_kp_std(limits)**2
        self.components = (PosteriorComponent('existing_deflecomp_logK_prior', Gaussian(mean, covariance)),)
        self.initial_components = self.components
        self.factors = ()
        self.factor_cache = {}
        # These are numerical memoization tables, not candidate measures.
        # Physical predictions at one identical log-K query can be shared by
        # factors with the same command and frozen continuation anchor.
        self._physical_prediction_cache = {}
        self._physical_prediction_query = None
        self._evaluation_cache_statistics = dict(factor_cache_hits=0,
            physical_cache_hits=0, native_batch_calls=0, native_batch_rows=0,
            native_continuation_rows=0, native_convex_certified_rows=0,
            native_direct_rows=0, native_direct_fallback_rows=0)
        self.observation_ids = frozenset()
        self.stiffness_bfs = None
        self.pending_stiffness_events = ()
        self._bfs_evidence_modes = ()
        self._joint_batch_signatures = {}
        if (self.stiffness_surprise_settings['enabled']
                and self.stiffness_surprise_settings['change_probability'] > 0.):
            initial_density = partial(_continuous_initial_density, self.initial_components)
            initial_source_id = _initial_density_id(self.initial_components)
            reset = ResetMeasure(initial_source_id, initial_density,
                description='explicit initial_log_stiffness continuous Gaussian-mixture reset; no point reset')
            laplace_settings = self.stiffness_surprise_settings.get('laplace', {})
            if not isinstance(laplace_settings, dict):
                raise ValueError('stiffness bayes_factor_surprise.laplace must be a mapping.')
            integrator = LocalLaplaceEvidence(self._evidence_starts, **laplace_settings)
            self.stiffness_bfs = ContinuousStiffnessBFS(initial_density,
                initial_source_id=initial_source_id,
                rho0=self.stiffness_surprise_settings['change_probability'],
                reset=reset, integrator=integrator)
        self.prior_source = dict(file=str(config_path), sha256=hashlib.sha256(
            config_path.read_bytes()).hexdigest(), kp_min=limits[0], kp_max=limits[1],
            initialization='deflecomp_core.estimator.initialization initial_log_kp_state/std',
            bounds_role='initial Gaussian construction only; log K support is unbounded')
        self.model_source = dict(urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
            total_mass_kg=self.robot.total_mass, gravity=self.robot.model.gravity.linear.tolist(),
            declaration=settings.get('known_model_source', 'URDF geometry, masses and inertias declared known for the auxiliary model'),
            spring_law='existing deflecomp LinearSpringModel',
            active_joint_limit_policy='actual box boundary and outward KKT sign; zero proximity/sign tolerance; converged interior roots stay free',
            equilibrium='existing EquilibriumSolver stiffness-homotopy continuation; initialized at sent command, not physical truth',
            numerical_continuation_stages=self.solver.cfg.n_lambda,
            numerical_continuation_stiffness=self.solver.cfg.k_stiffness)
        tree = ET.parse(urdf_path).getroot()
        joints = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
        self.edges = []
        for name in self.body_edge_ids:
            joint = joints[name]
            if joint.get('type') != 'revolute':
                raise ValueError('Auxiliary body edges must be the declared revolute URDF joints.')
            joint_id = self.robot.model.getJointId(joint.get('name'))
            model_joint = self.robot.model.joints[joint_id]
            if model_joint.nq != 1 or model_joint.nv != 1:
                raise ValueError('This auxiliary model requires scalar revolute joint coordinates.')
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            axis /= np.linalg.norm(axis)
            self.edges.append((name, self.robot.get_frame_id(joint.find('parent').get('link')),
                               self.robot.get_frame_id(name), model_joint.idx_q, axis))
        self.last_stamp = None
        self.last_key = self.last_record = None
        self._branch_anchor = None
        self.last_equilibrium_audit = None
        self.last_continuation_audit = None
        self.equilibrium_audit_summary = dict(evaluations=0,
            residual_warning_count=0, nonpositive_curvature_count=0,
            maximum_KKT_residual_nm=0.)
        backend = settings.get('auxiliary_equilibrium_backend', 'native')
        if backend not in ('native', 'python_reference'):
            raise ValueError('auxiliary_equilibrium_backend must be native or python_reference.')
        self._auxiliary_native = None
        if backend == 'native':
            from .auxiliary_equilibrium_native import AuxiliaryEquilibriumNative
            self._auxiliary_native = AuxiliaryEquilibriumNative(str(urdf_path), self.robot.nv,
                certified_fastpath=settings.get('auxiliary_certified_direct', False))
            self.model_source['global_convexity_certificate'] = dict(
                enabled=self._auxiliary_native.certified_fastpath,
                gravity_hessian_absolute_bound=self._auxiliary_native.gravity_hessian_absolute_bound.tolist(),
                proof='URDF mass-times-gravity times rotated-segment lengths bound every gravity Hessian entry; exact gravity-parallel root rotation symmetry gives zero rows; positive weighted row witness certifies diag(K)-B positive definite',
                uncertified_policy='retain canonical continuation; no candidate or update dropped',
                finite_direct_root_policy='use only after the existing 1e-11 Nm exact-box KKT residual target; otherwise canonical continuation')
        if self.stiffness_bfs is not None:
            self.stiffness_bfs.batch_factor_evaluator = self._batch_factor_evaluation
        self.model_source['anchored_equilibrium_backend'] = backend
        self.model_source['direct_unanchored_equilibrium_backend'] = 'existing Python stiffness homotopy'

    def numerical_diagnostics(self):
        """Report numerical quality without making finite audits fatal.

        An inaccurate finite auxiliary root is an explicitly approximate
        command/parameter model. No covariance, density or Hessian is altered
        to turn an audit into a passed check. Actual nonfinite arithmetic and
        failed linear solves still raise at their point of use.
        """
        return deepcopy(dict(
            policy='finite residual, curvature and iteration audits are diagnostic only; actual nonfinite arithmetic or unsolved linear systems remain errors',
            last_equilibrium=getattr(self, 'last_equilibrium_audit', None),
            last_continuation=getattr(self, 'last_continuation_audit', None),
            equilibrium_summary=getattr(self, 'equilibrium_audit_summary', None),
            evaluation_cache=getattr(self, '_evaluation_cache_statistics', None)))

    def _evidence_starts(self):
        """Numerical evidence-mode starts; never a finite stiffness prior."""
        candidates = []
        for item in self.initial_components:
            centre = item.distribution.mean
            scale = np.linalg.cholesky(item.distribution.covariance)
            candidates.append(centre.copy())
            for direction in scale.T:
                candidates.extend((centre+direction, centre-direction))
            coupled = np.sum(scale, axis=1)
            candidates.extend((centre+coupled, centre-coupled))
        candidates.extend(self._bfs_evidence_modes)
        previous = getattr(self, '_proposal_representative', None)
        if previous is not None:
            candidates.append(previous['logK'])
        return tuple(candidates)

    def _apply_pending_stiffness_events(self):
        """Apply queued factors in order, committing each complete BFS atomically.

        A failed numerical integral leaves the frozen factor in this queue.
        Later events cannot bypass it. Observation data are consequently not
        discarded merely because a local evidence approximation failed.
        """
        recovered = []
        if self.stiffness_bfs is not None:
            self.stiffness_bfs.runtime_monitor = getattr(self, 'runtime_monitor', None)
        while self.pending_stiffness_events:
            factor, event_id, metadata = self.pending_stiffness_events[0]
            with _runtime_stage(self, 'stiffness.bfs_update', event_id=event_id,
                    factor_id=factor.factor_id, observation_count=len(factor.observation_ids),
                    retained_factor_count=len(self.factors)):
                record = self.stiffness_bfs.advance(factor.factor_id, factor.observation_ids,
                    partial(self._factor_evaluation, factor), event_id=event_id,
                    identity_factor=bool(factor.ratio.identity), metadata=metadata,
                    rho0=metadata.get('joint_batch_rho0'),
                    log_keep_probability=metadata.get('joint_batch_log_keep_probability'),
                    log_reset_probability=metadata.get('joint_batch_log_reset_probability'))
            # Only after both branch integrals succeeded does the accepted
            # expression, factor list and consumed-ID set advance together.
            self.factors += (factor,)
            self.observation_ids = self.observation_ids.union(factor.observation_ids)
            self.pending_stiffness_events = self.pending_stiffness_events[1:]
            modes = []
            for name in ('keep_evidence', 'reset_evidence'):
                evidence = record.get(name)
                mode = None if evidence is None else evidence['diagnostics'].get('mode')
                if mode is not None:
                    value = np.asarray(mode, dtype=float)
                    if not any(np.array_equal(value, old) for old in modes):
                        modes.append(value)
            if modes:
                self._bfs_evidence_modes = tuple(modes)
            recovered.append(record)
        return recovered

    def _refine_equilibrium(self, theta, command, stiffness):
        theta = np.asarray(theta).copy()
        # L-BFGS can stop on floating-point potential changes before its torque
        # residual is accurate enough to differentiate. Refine the very same
        # existing potential/KKT equations, without adding an error model.
        iterations, termination = 0, 'refinement_iteration_budget'
        for _ in range(24):
            hessian, _ = self.sensitivity.equilibrium_jacobians(theta, command, stiffness)
            residual = self.robot.tau_gravity(theta)+self.spring.torque(theta, command, stiffness)
            residual[self.sensitivity.last_active_set] = 0.
            if (not np.all(np.isfinite(theta)) or not np.all(np.isfinite(hessian))
                    or not np.all(np.isfinite(residual))):
                raise ValueError('Auxiliary equilibrium refinement has nonfinite arithmetic.')
            norm = float(np.max(np.abs(residual)))
            if norm <= 1e-11:
                termination = 'refinement_residual_target'
                break
            direction = np.linalg.solve(hessian, residual)
            if not np.all(np.isfinite(direction)):
                raise ValueError('Auxiliary equilibrium Newton direction is nonfinite.')
            iterations += 1
            potential = self.solver._V_total(self.robot, self.spring, theta, command, stiffness)
            accepted = False
            for backtrack in range(16):
                candidate = np.clip(theta-(.5**backtrack)*direction,
                    self.robot.model.lowerPositionLimit, self.robot.model.upperPositionLimit)
                candidate_residual = self.robot.tau_gravity(candidate)+self.spring.torque(candidate, command, stiffness)
                active = self.sensitivity.equilibrium_active_set(candidate, command, stiffness)
                candidate_residual[active] = 0.
                candidate_potential = self.solver._V_total(self.robot, self.spring, candidate, command, stiffness)
                if (np.max(np.abs(candidate_residual)) < norm
                        and candidate_potential <= potential+1e-12*max(1., abs(potential))):
                    theta, accepted = candidate, True
                    break
            if not accepted:
                termination = 'no_further_residual_and_potential_reduction'
                break
        hessian, parameter_jacobian = self.sensitivity.equilibrium_jacobians(theta, command, stiffness)
        residual = self.robot.tau_gravity(theta)+self.spring.torque(theta, command, stiffness)
        residual[self.sensitivity.last_active_set] = 0.
        if (not np.all(np.isfinite(theta)) or not np.all(np.isfinite(hessian))
                or not np.all(np.isfinite(parameter_jacobian))
                or not np.all(np.isfinite(residual))):
            raise ValueError('Auxiliary equilibrium result has nonfinite arithmetic.')
        eigenvalues, curvature_error = None, None
        try:
            candidate_eigenvalues = np.linalg.eigvalsh(.5*hessian+.5*hessian.T)
            if np.all(np.isfinite(candidate_eigenvalues)):
                eigenvalues = candidate_eigenvalues
            else:
                curvature_error = 'Diagnostic Hessian eigendecomposition returned nonfinite eigenvalues.'
        except np.linalg.LinAlgError as exc:
            curvature_error = '%s: %s' % (type(exc).__name__, exc)
        residual_norm = float(np.max(np.abs(residual)))
        self.last_equilibrium_audit = dict(theta=theta.tolist(),
            command=np.asarray(command).tolist(), selected_K=np.asarray(stiffness).tolist(),
            KKT_residual_nm=residual.tolist(), KKT_residual_inf_nm=residual_norm,
            KKT_audit_tolerance_nm=1e-9, KKT_tolerance_satisfied=residual_norm <= 1e-9,
            Hessian_eigenvalues=None if eigenvalues is None else eigenvalues.tolist(),
            positive_curvature=None if eigenvalues is None else bool(eigenvalues[0] > 0.),
            curvature_audit_error=curvature_error,
            refinement_iterations=iterations, refinement_termination=termination,
            refinement_target_nm=1e-11, refinement_converged=residual_norm <= 1e-11,
            active_joint_limits=self.sensitivity.last_active_set.tolist(),
            used_finite_auxiliary_result=True, audit_is_fatal=False,
            Hessian_modified=False, physical_stability_guaranteed=False)
        summary = getattr(self, 'equilibrium_audit_summary', dict(evaluations=0,
            residual_warning_count=0, nonpositive_curvature_count=0,
            maximum_KKT_residual_nm=0.))
        summary['evaluations'] += 1
        summary['residual_warning_count'] += int(residual_norm > 1e-9)
        summary['nonpositive_curvature_count'] += int(eigenvalues is not None and eigenvalues[0] <= 0.)
        summary['curvature_audit_failure_count'] = summary.get('curvature_audit_failure_count', 0)+int(curvature_error is not None)
        summary['maximum_KKT_residual_nm'] = max(summary['maximum_KKT_residual_nm'], residual_norm)
        self.equilibrium_audit_summary = summary
        return theta, hessian, parameter_jacobian, self.sensitivity.last_active_set.copy()

    def _freeze_branch(self, state, record):
        # The branch anchor is a numerical convention of the auxiliary model.
        # It is frozen for this retained factor. The initial Gaussian mode is
        # not a projection or replacement of the cumulative parameter law.
        anchor = max(self.components, key=lambda item: item.weight).distribution.mean.copy()
        command = np.asarray(state.command)
        stiffness = np.exp(anchor)
        theta = self.solver.solve(command, stiffness, theta_init=command)
        theta, _, _, active = self._refine_equilibrium(theta, command, stiffness)
        self._branch_anchor = (state.state_id, anchor, theta)
        record['auxiliary_branch_selection'] = dict(
            policy='fixed initial-Gaussian-mode numerical anchor; canonical straight log-K continuation; no evaluation-order warm starts',
            anchor_logK=anchor.tolist(), anchor_equilibrium=theta.tolist(),
            anchor_active_joint_limits=active.tolist(), source_state_id=state.state_id,
            maximum_logK_step=.1, maximum_segment_bisections=10,
            maximum_logK_step_role='nominal numerical step; long finite paths use the recorded coarser bounded-work partition',
            anchor_numerical_audit=deepcopy(self.last_equilibrium_audit),
            role='auxiliary numerical equilibrium branch only; no body or parameter-prior replacement and no physical stability guarantee')

    def _branch_equilibrium(self, x, state):
        command = np.asarray(state.command)
        anchor = getattr(self, '_branch_anchor', None)
        if anchor is None or anchor[0] != state.state_id:
            self.last_continuation_audit = dict(method='direct auxiliary solve',
                requested_stages=None, used_stages=None, work_budget_saturated=False,
                audit_is_fatal=False)
            stiffness = np.exp(x)
            return self._refine_equilibrium(
                self.solver.solve(command, stiffness, theta_init=command), command, stiffness)
        _, start, theta = anchor
        native = getattr(self, '_auxiliary_native', None)
        if native is not None:
            result = native.continue_from(start, theta, x, command)
            self._accept_native_equilibrium(result)
            return (result['theta'], result['hessian'],
                    result['parameter_jacobian'], result['active'])
        # Each query starts from the same anchor and takes the same path,
        # independently of every earlier optimizer or Hessian evaluation.
        def segment(left, right, initial, depth):
            try:
                return self._refine_equilibrium(initial, command, np.exp(right))
            except (ValueError, np.linalg.LinAlgError):
                if depth == 10:
                    raise ValueError('Canonical auxiliary continuation could not retain a stable equilibrium branch.')
                middle = .5*(left+right)
                midway = segment(left, middle, initial, depth+1)
                return segment(middle, right, midway[0], depth+1)
        requested_count = max(1, int(np.ceil(np.max(np.abs(x-start))/.1)))
        count = min(requested_count, 1000)
        self.last_continuation_audit = dict(method='same anchor and straight log-K path',
            requested_stages=requested_count, used_stages=count,
            nominal_maximum_logK_step=.1,
            used_maximum_logK_step=float(np.max(np.abs(x-start))/count),
            work_budget_saturated=bool(requested_count > count), numerical_work_budget=1000,
            stiffness_candidate_clipped=False, anchor_changed=False, audit_is_fatal=False)
        left = start
        result = None
        for fraction in range(1, count+1):
            right = start+(fraction/count)*(x-start)
            result = segment(left, right, theta, 0)
            theta, left = result[0], right
        return result

    def _accept_native_equilibrium(self, result):
        """Expose actual native diagnostics; a failed row is never a value."""
        if result.get('status') not in ('ok', 'success', 'available'):
            raise ValueError('Native auxiliary equilibrium failed: '+str(result.get('error')))
        for name in ('theta', 'hessian', 'parameter_jacobian', 'active'):
            if not np.all(np.isfinite(result[name])):
                raise ValueError('Native auxiliary equilibrium returned nonfinite '+name+'.')
        self.last_equilibrium_audit = deepcopy(result.get('audit',
            result.get('equilibrium_audit')))
        self.last_continuation_audit = deepcopy(result.get('continuation',
            result.get('continuation_audit')))
        certificate = (self.last_continuation_audit or {}).get('convexity_certificate', {})
        statistics = self._evaluation_cache_statistics
        statistics['native_continuation_rows'] += 1
        statistics['native_convex_certified_rows'] += int(certificate.get('certified', False))
        statistics['native_direct_rows'] += int(certificate.get('direct_used', False))
        statistics['native_direct_fallback_rows'] += int(
            certificate.get('direct_attempted', False) and not certificate.get('direct_used', False))
        current = result.get('equilibrium_summary', result.get('equilibrium_audit_summary'))
        if current is not None:
            summary = self.equilibrium_audit_summary
            for key in ('evaluations', 'residual_warning_count',
                        'nonpositive_curvature_count', 'curvature_audit_failure_count'):
                summary[key] = summary.get(key, 0)+current.get(key, 0)
            summary['maximum_KKT_residual_nm'] = max(summary['maximum_KKT_residual_nm'],
                current.get('maximum_KKT_residual_nm', 0.))

    def _physical_prediction_key(self, x, state, anchor=None):
        anchor = getattr(self, '_branch_anchor', None) if anchor is None else anchor
        if anchor is None or anchor[0] != state.state_id:
            return None
        return (np.asarray(state.command, dtype=float).tobytes(),
                np.asarray(anchor[1], dtype=float).tobytes(),
                np.asarray(anchor[2], dtype=float).tobytes())

    def _physical_query_cache(self, x):
        query = np.asarray(x, dtype=float).tobytes()
        if getattr(self, '_physical_prediction_query', None) != query:
            # Clear cached arithmetic only. All factors and source candidates
            # remain in their immutable retained histories.
            self._physical_prediction_cache = {}
            self._physical_prediction_query = query
        return self._physical_prediction_cache

    def _physical_prediction(self, x, state):
        stiffness = np.exp(x)
        if not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0.):
            raise ValueError('Nonfinite auxiliary stiffness candidate.')
        key = self._physical_prediction_key(x, state)
        cache = self._physical_query_cache(x)
        if key is not None and key in cache:
            self._evaluation_cache_statistics['physical_cache_hits'] += 1
            value = cache[key]
            if 'error' in value:
                raise ValueError(value['error'])
            self.last_equilibrium_audit = deepcopy(value['audit'])
            self.last_continuation_audit = deepcopy(value['continuation'])
            return value
        theta, hessian, parameter_jacobian, active = self._branch_equilibrium(x, state)
        derivative = -np.linalg.solve(hessian, parameter_jacobian)
        if not np.all(np.isfinite(derivative)):
            raise ValueError('Auxiliary equilibrium parameter sensitivity is nonfinite.')
        prediction = self._store_physical_prediction(theta, derivative, active,
            self.last_equilibrium_audit, self.last_continuation_audit)
        if key is not None:
            cache[key] = prediction
        return prediction

    def _store_physical_prediction(self, theta, derivative, active, audit, continuation):
        self.robot._fk_update(theta)
        local_poses = []
        for _, parent, child, _, _ in self.edges:
            local = self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]
            local_poses.append((np.array(local.translation, copy=True),
                                np.array(local.rotation, copy=True)))
        return dict(theta=np.array(theta, copy=True), derivative=np.array(derivative, copy=True),
            active=np.array(active, copy=True), local_poses=tuple(local_poses),
            audit=deepcopy(audit), continuation=deepcopy(continuation))

    def _prediction(self, x, state):
        physical = self._physical_prediction(x, state)
        value, jacobian = self._body_chart(physical['theta'], physical['derivative'], state,
                                         local_poses=physical['local_poses'])
        return value, jacobian, physical['theta'], physical['active']

    def _body_chart(self, theta, derivative, state, *, local_poses=None):
        if local_poses is None:
            self.robot._fk_update(theta)
        by_id = {item.edge_id: item for item in state.transforms}
        value = np.zeros(6*len(self.edges)); jacobian = np.zeros((value.size, derivative.shape[1]))
        for index, (name, parent, child, joint_index, axis) in enumerate(self.edges):
            if local_poses is None:
                local = self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]
                translation, rotation = local.translation, local.rotation
            else:
                translation, rotation = local_poses[index]
            edge = by_id[name]
            rotation_vector = Rotation.from_matrix(
                quat_to_rotmat(edge.rotation_wxyz).T @ rotation).as_rotvec()
            value[6*index:6*index+3] = translation-edge.translation
            value[6*index+3:6*index+6] = rotation_vector
            jacobian[6*index+3:6*index+6] = np.outer(
                _right_log_derivative(rotation_vector) @ axis, derivative[joint_index])
        return value, jacobian

    def update_batch(self, result, *, events, command_ledger, batch_id=None,
                     event_body_ratios=None):
        """Use time-matched body ratios and one declared batch hazard.

        Per-observation intermediate evidence/representative computations are
        absent.  With ``event_body_ratios`` every effective acquisition has its
        own unprojected before/after Gaussian ratio and its own command.  The
        endpoint body is only the final shared IK/display law.  ``None`` keeps
        the explicitly named legacy endpoint approximation for old synthetic
        callers, never for a valid production joint-batch observation.
        """
        if (event_body_ratios is None and 'joint_batch' in result.records
                and result.records.get('observation_ids')):
            raise ValueError('A production joint batch must supply every acquisition-specific stiffness ratio; endpoint-only fallback is forbidden.')
        from .joint_batch_stiffness import make_joint_batch_stiffness_context
        context = make_joint_batch_stiffness_context(result, events=events,
            command_ledger=command_ledger, batch_id=batch_id,
            probability=self.stiffness_surprise_settings['change_probability'],
            enabled=self.stiffness_bfs is not None)
        return self.update(result, batch_context=context,
                           event_body_ratios=event_body_ratios)

    def _time_indexed_ratios(self, specifications, batch_record):
        """Validate original acquisition identity and freeze each actual chart.

        A prefix prior Gaussian can have a nonzero mean and can differ from
        the initial reference covariance.  Keeping that bias in the original
        chart is intentional: intermediate reexpression/projection is absent.
        """
        if batch_record is None:
            raise ValueError('Time-indexed stiffness factors require a declared acquisition batch.')
        specifications = tuple(specifications)
        acquired = {item['event_id']: item for item in batch_record['acquired_events']}
        expected = tuple(batch_record['effective_observation_ids'])
        if tuple(item.get('observation_id') for item in specifications) != expected:
            raise ValueError('Time-indexed stiffness ratios must retain every effective acquisition in order.')
        frozen, digest = [], hashlib.sha256(b'ProbTF time-indexed stiffness batch v1')
        for spec in specifications:
            identity = spec['observation_id']
            event = acquired[identity]
            stamp, command = float(spec['source_stamp']), tuple(spec['source_command'])
            if stamp != event['stamp'] or command != tuple(event['command']):
                raise ValueError('A stiffness ratio must use its original acquisition command and timestamp.')
            branches = tuple(spec['branches'])
            projected = spec.get('posterior_branch')
            if not branches:
                raise ValueError('Every acquisition stiffness ratio needs its full frozen body branches.')
            for branch in branches+(() if projected is None else (projected,)):
                reference = branch.reference_state
                if (reference.stamp != stamp or tuple(reference.command) != command
                        or branch.prior.mean.shape != reference.mean.shape
                        or branch.prior.covariance.shape != reference.covariance.shape):
                    raise ValueError('An acquisition stiffness ratio changed its frozen source chart or clock.')
                if any(child.distribution.mean.shape != reference.mean.shape
                       or child.distribution.covariance.shape != reference.covariance.shape
                       for child in branch.posterior_components):
                    raise ValueError('An acquisition stiffness ratio lost full joint Gaussian dimensions.')
            ratio = ChartedBodyDensityRatio(branches, self.body_edge_ids,
                                           posterior_branch=projected)
            fingerprint = hashlib.sha256(repr((identity, stamp, command)).encode())
            for branch in branches+(() if projected is None else (projected,)):
                fingerprint.update(branch.reference_state.state_id.encode())
                fingerprint.update(repr((branch.label, _log_weight(branch))).encode())
                fingerprint.update(branch.prior.mean.tobytes()+branch.prior.covariance.tobytes())
                for child in branch.posterior_components:
                    fingerprint.update(repr((child.label, _log_weight(child))).encode())
                    fingerprint.update(child.distribution.mean.tobytes()
                                       +child.distribution.covariance.tobytes())
            factor_id = fingerprint.hexdigest()
            digest.update(factor_id.encode())
            frozen.append((factor_id, identity, ratio, branches,
                           deepcopy(spec.get('provenance', spec.get('record')))))
        return tuple(frozen), digest.hexdigest()

    def _freeze_time_indexed_factor(self, definitions, composite_id, record):
        """Capture all command-specific anchors without solving evidence yet."""
        parts, diagnostics, saved = [], [], self._branch_anchor
        try:
            for factor_id, identity, ratio, branches, provenance in definitions:
                source = branches[0].reference_state
                properness = (dict(verified=True, log_factor_lower_bound=0.,
                    log_factor_upper_bound=0., argument='declared exact identity body density ratio')
                    if ratio.identity else self._properness_bounds(ratio))
                parameter_records = []
                for branch, (prior, children, _) in zip(branches, ratio.marginals):
                    parameter_records.append(dict(label=branch.label,
                        log_weight=_log_weight(branch),
                        reference_state_id=branch.reference_state.state_id,
                        reference_version=branch.reference_state.version,
                        mean_dimension=prior.mean.size, support_rank=prior.support.rank,
                        prior_parameters_sha256=hashlib.sha256(
                            prior.mean.tobytes()+prior.covariance.tobytes()).hexdigest(),
                        children=[dict(label=child.label, log_weight=_log_weight(child),
                            parameters_sha256=hashlib.sha256(distribution.mean.tobytes()
                                +distribution.covariance.tobytes()).hexdigest())
                            for child, distribution in children]))
                anchor, item = None, dict(observation_id=identity,
                    source_stamp=source.stamp, source_command=list(source.command),
                    factor_id=factor_id,
                    body_marginal=dict(edge_ids=list(self.body_edge_ids),
                        frozen_factor_sha256=factor_id, branches=parameter_records,
                        full_joint_parameters_retained_in_immutable_factor=True,
                        repeated_full_matrix_JSON_expansion=False,
                        digest_array_convention='float64 native-order C-contiguous mean bytes followed by covariance bytes'),
                    prefix_source=provenance,
                    trial_stiffness_mechanics_evaluated_at_original_acquisition=True,
                    intermediate_body_projection_used=False)
                if not ratio.identity:
                    self._freeze_branch(source, item)
                    anchor = self._branch_anchor
                parts.append(FrozenStiffnessFactor(factor_id, (identity,), ratio,
                    tuple(branch.reference_state for branch in branches), anchor, properness))
                diagnostics.append(item)
        finally:
            self._branch_anchor = saved
        properness = dict(verified=all(part.properness['verified'] for part in parts),
            log_factor_lower_bound=sum(part.properness['log_factor_lower_bound'] for part in parts),
            log_factor_upper_bound=sum(part.properness['log_factor_upper_bound'] for part in parts),
            argument='sum of the retained bounded acquisition-specific body log-density ratios')
        record['time_indexed_body_factors'] = diagnostics
        record['stiffness_batch_factorization'] = dict(
            representation='product of all acquisition-specific full-body prefix density ratios',
            log_factor='sum_i log[p_body_i_plus(h(K,q_i)) / p_body_i_minus(h(K,q_i))]',
            atomic_factor_count=len(parts), atomic_factor_ids=[part.factor_id for part in parts],
            all_acquisition_commands_requeried_for_each_trial_K=True,
            endpoint_body_ratio_used_as_K_factor=False,
            intermediate_Laplace_updates=False, intermediate_command_proposals=False,
            batch_evidence_updates=1 if parts else 0,
            parameter_density_projection_applied=False, monte_carlo_used=False)
        return FrozenTimeIndexedStiffnessFactor(composite_id,
            tuple(part.observation_ids[0] for part in parts), tuple(parts), properness)

    def update(self, result, *, batch_context=None, event_body_ratios=None):
        """Retain new likelihood factors, without projecting away information.

        Numerical proposal fits are deliberately separate from this density.
        In particular, inability to find a smooth Laplace mode cannot erase
        an observation or replace its posterior by the preceding prior.
        """
        state = result.prior
        if (event_body_ratios is None and 'joint_batch' in result.records
                and result.records.get('observation_ids')):
            raise ValueError('A production joint batch must supply every acquisition-specific stiffness ratio; endpoint-only fallback is forbidden.')
        batch_record = None
        if batch_context is not None:
            from .joint_batch_stiffness import JointBatchStiffnessContext
            if not isinstance(batch_context, JointBatchStiffnessContext):
                raise ValueError('A batch update needs its validated immutable stiffness batch context.')
            batch_context.validate_result(result)
            batch_record = batch_context.record()
            previous = self._joint_batch_signatures.get(batch_context.batch_id)
            if previous is not None and previous != batch_context.source_digest:
                raise ValueError('An already acquired stiffness batch identity changed its contents.')
        time_indexed = None
        if event_body_ratios is not None:
            with _runtime_stage(self, 'stiffness.time_indexed_ratio_validation',
                    batch_id=None if batch_record is None else batch_record['batch_id'],
                    acquisition_count=(len(event_body_ratios)
                        if hasattr(event_body_ratios, '__len__') else None)):
                time_indexed, time_indexed_id = self._time_indexed_ratios(event_body_ratios, batch_record)
            batch_record.update(factor_semantics='product of acquisition-specific full-body prefix posterior/prior ratios; each trial K is evaluated by auxiliary equilibrium/FK at every original acquisition command',
                time_indexed_atomic_factor_count=len(time_indexed),
                time_indexed_atomic_factor_ids=[item[0] for item in time_indexed],
                time_indexed_factorization_id=time_indexed_id,
                endpoint_only_stiffness_factor=False)
        if result.frozen_body_prior is None or result.frozen_body_posterior is None:
            raise ValueError('Stiffness requires actual frozen pre-reexpression body Gaussians.')
        raw_prior, raw_posterior = result.frozen_body_prior, result.frozen_body_posterior
        branches = tuple(getattr(result, 'frozen_body_branches', ()))
        if not branches:
            if not isinstance(raw_prior, Gaussian):
                raise ValueError('Every body mixture parent needs its frozen chart correspondence.')
            branches = (SimpleNamespace(label=state.component_id, weight=1., reference_state=state,
                prior=raw_prior, posterior_components=tuple(raw_posterior.components)),)
        projected_numerator = getattr(result, 'projected_body_numerator', None)
        numerator_branches = branches if projected_numerator is None else (projected_numerator,)
        observation_ids = tuple(result.records.get('observation_ids', raw_posterior.observation_ids))
        digest = hashlib.sha256()
        for branch in branches:
            digest.update(branch.reference_state.state_id.encode())
            digest.update(repr((branch.label, _log_weight(branch))).encode())
            digest.update(branch.prior.mean.tobytes()+branch.prior.covariance.tobytes())
            for item in branch.posterior_components:
                digest.update(repr((item.label, _log_weight(item))).encode())
                digest.update(item.distribution.mean.tobytes()+item.distribution.covariance.tobytes())
        if isinstance(raw_prior, Gaussian):
            digest.update(raw_prior.mean.tobytes()+raw_prior.covariance.tobytes())
        for item in raw_posterior.components:
            digest.update(item.distribution.mean.tobytes()+item.distribution.covariance.tobytes())
            digest.update(repr((item.label, _log_weight(item))).encode())
        if projected_numerator is not None:
            digest.update(b'declared-projected-single-body-numerator')
            digest.update(projected_numerator.reference_state.state_id.encode())
            digest.update(projected_numerator.prior.mean.tobytes()
                          +projected_numerator.prior.covariance.tobytes())
            for item in projected_numerator.posterior_components:
                digest.update(repr((item.label, _log_weight(item))).encode())
                digest.update(item.distribution.mean.tobytes()+item.distribution.covariance.tobytes())
        if time_indexed is not None:
            digest.update(b'acquisition-specific-stiffness-body-ratios')
            digest.update(time_indexed_id.encode())
        key = (digest.hexdigest(), observation_ids,
               tuple((item.label, item.weight, _log_weight(item), item.state.state_id)
                     for item in result.stages[-1].components))
        if batch_context is not None:
            key += (batch_context.source_digest,)
        if key == self.last_key:
            return self.last_record
        if time_indexed is not None and self.last_stamp is not None and any(
                branches[0].reference_state.stamp < self.last_stamp
                for _, _, _, branches, _ in time_indexed):
            raise ValueError('A stiffness acquisition predates the retained batch frontier; explicit checkpoint replay is required.')
        bfs = getattr(self, 'stiffness_bfs', None)
        if self.last_stamp is not None and (state.stamp < self.last_stamp
                or state.stamp == self.last_stamp and bfs is None
                    and result.records.get('sensor_event') is None and batch_context is None):
            raise ValueError('Stiffness acquisition times must increase; distinct configured sensor events may share a time.')
        if isinstance(raw_prior, Gaussian) and (
                not np.array_equal(raw_prior.mean, state.mean)
                or not np.array_equal(raw_prior.covariance, state.covariance)):
            raise ValueError('Frozen body prior does not match its source state.')
        if raw_posterior.prior_version is not None and (
                raw_posterior.prior_version != state.version or raw_posterior.stamp != state.stamp
                or tuple(raw_posterior.command) != state.command):
            raise ValueError('Raw posterior and prior source metadata differ.')
        for branch in branches+(() if projected_numerator is None else (projected_numerator,)):
            reference = branch.reference_state
            if (not np.array_equal(branch.prior.mean, reference.mean)
                    or not np.array_equal(branch.prior.covariance, reference.covariance)
                    or reference.stamp != state.stamp or reference.command != state.command):
                raise ValueError('Frozen body prior does not match its source state.')
        displayed = tuple(result.stages[-1].components)
        if len(displayed) != len(raw_posterior.components):
            raise ValueError('Frozen and displayed body mixture branches differ.')
        source_by_label = {child.label: (branch, child) for branch in numerator_branches
                           for child in branch.posterior_components}
        if len(source_by_label) != len(displayed):
            raise ValueError('Frozen body branch correspondence lost or duplicated a displayed component.')
        for source, item in zip(raw_posterior.components, displayed):
            if item.label not in source_by_label:
                raise ValueError('Frozen body posterior and displayed provenance differ.')
            branch, child = source_by_label[item.label]
            reference = branch.reference_state
            if (not np.array_equal(branch.prior.mean, reference.mean)
                    or not np.array_equal(branch.prior.covariance, reference.covariance)):
                raise ValueError('Frozen body prior does not match its source state.')
            if (source.label != item.label or source.weight != item.weight
                    or _log_weight(source) != _log_weight(item)
                    or _log_weight(source) != _log_weight(child)
                    or not np.array_equal(source.distribution.mean, child.distribution.mean)
                    or not np.array_equal(source.distribution.covariance, child.distribution.covariance)
                    or item.state.stamp != state.stamp or item.state.command != state.command):
                raise ValueError('Frozen body posterior and displayed provenance differ.')
            if (np.array_equal(child.distribution.mean, reference.mean)
                    and np.array_equal(child.distribution.covariance, reference.covariance)
                    and item.state.state_id == reference.state_id):
                expected_id = reference.state_id
            else:
                materialize = (materialize_state_in_chart
                    if projected_numerator is not None and result.records.get(
                        'internal_projection', {}).get('common_chart_materialization', False)
                    else reexpress_state)
                expected_id = materialize(reference, child.distribution.mean,
                    child.distribution.covariance, version=item.state.version,
                    component_id=item.state.component_id).state_id
            if expected_id != item.state.state_id:
                raise ValueError('Frozen posterior parameters do not reexpress to the displayed state.')
        metadata = dict(source_stamp=state.stamp, source_command=list(state.command),
            source_prior_state_id=state.state_id, source_prior_version=state.version,
            source_prior_state_ids=[branch.reference_state.state_id for branch in branches],
            source_prior_weights=[branch.weight for branch in branches],
            body_posterior_state_ids=[item.state.state_id for item in displayed],
            body_posterior_versions=[item.state.version for item in displayed],
            body_posterior_weights=[item.weight for item in displayed],
            body_posterior_log_weights=[_log_weight(item) for item in displayed],
            body_posterior_labels=[item.label for item in displayed],
            observation_ids=list(observation_ids))
        prior_record = self._density_record()
        prior_bfs_event_count = 0 if bfs is None else len(bfs.updates)
        record = dict(metadata, prior=prior_record, posterior=prior_record,
            status='not_updated', update_applied=False, information_rank=None,
            information_eigenvalues=None, prior_source=self.prior_source,
            auxiliary_model=self.model_source, feedback_to_body=False,
            feedback_to_body_or_control=False, feedback_to_control=None,
            proposal_approximation=None, proposal_status='not_computed', command_proposal=None,
            approximation='specified Bingham-derived local Gaussian body proxies; no Gaussian projection of cumulative stiffness density',
            support_regularization_added=False)
        if batch_record is not None:
            record['joint_batch'] = batch_record
        projection = result.records.get('internal_projection')
        if projection is not None:
            record['body_internal_approximation'] = dict(
                internal_state_id=projection['internal_state_id'],
                source_prior_state_id=projection['source_prior_state_id'],
                approximation=projection['approximation'],
                ratio_semantics=('native transition-prior mixture versus the declared full joint moment approximation after sensor fusion; not a conditional keep-branch likelihood'
                    if projected_numerator is not None else
                    'prior versus the declared full joint moment approximation after sensor fusion; not the unprojected exact observation likelihood'))
            if time_indexed is not None:
                record['body_internal_approximation'].update(
                    role_for_stiffness='endpoint shared IK/display law diagnostic only; not the time-indexed K numerator',
                    ratio_semantics='K uses unprojected acquisition-specific prefix body ratios, not this terminal projected body law')
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError('New stiffness evidence contains a duplicate observation ID.')
        repeated = self.observation_ids.intersection(observation_ids)
        if bfs is not None:
            pending_ids = {identifier for factor, _, _ in self.pending_stiffness_events
                           for identifier in factor.observation_ids}
            repeated = repeated.union(pending_ids.intersection(observation_ids))
        if repeated:
            raise ValueError('A stiffness density factor cannot count an old observation again: '+repr(sorted(repeated)))
        sensor_event = result.records.get('sensor_event')
        if bfs is not None and observation_ids and batch_context is None:
            if len(observation_ids) != 1:
                raise ValueError('per_effective_sensor_event stiffness BFS requires one separately fused sensor event, not a multi-sensor packet.')
            if sensor_event is not None and (
                    sensor_event.get('kind') not in ('imu', 'image')
                    or not sensor_event.get('enabled', False)
                    or sensor_event.get('event_id') != observation_ids[0]):
                raise ValueError('Stiffness BFS observation identity does not match its effective sensor event.')
        try:
            with _runtime_stage(self, 'stiffness.endpoint_body_ratio_diagnostic',
                    source_stamp=state.stamp, observation_count=len(observation_ids)):
                ratio = ChartedBodyDensityRatio(branches, self.body_edge_ids,
                                               posterior_branch=projected_numerator)
                record['body_marginal'] = dict(edge_ids=list(self.body_edge_ids), **ratio.record())
            identity = (ratio.identity if time_indexed is None else
                        all(item[2].identity for item in time_indexed))
            if time_indexed is not None:
                record['body_marginal']['role_for_stiffness'] = 'terminal body diagnostic only; not used as a stiffness factor'
            if identity and (bfs is None or not observation_ids):
                self.observation_ids = self.observation_ids.union(observation_ids)
                record.update(status='identity_density_ratio', log_density_ratio=0.,
                              information_rank=0, identity_preserved_bitwise=True)
            else:
                if not observation_ids:
                    raise ValueError('A changed posterior without new observation IDs is not new stiffness evidence.')
                if time_indexed is not None:
                    with _runtime_stage(self, 'stiffness.factor_freeze',
                            source_stamp=state.stamp, acquisition_count=len(time_indexed)):
                        factor = self._freeze_time_indexed_factor(time_indexed, digest.hexdigest(), record)
                    properness = factor.properness
                else:
                    properness = (dict(verified=True, log_factor_lower_bound=0.,
                        log_factor_upper_bound=0., argument='declared exact identity body density ratio')
                        if ratio.identity else self._properness_bounds(ratio))
                    # An equilibrium query failure does not become a zero
                    # likelihood. The frozen density factor remains in memory.
                    anchor = None
                    if not ratio.identity:
                        try:
                            with np.errstate(over='raise', invalid='raise'):
                                self._freeze_branch(branches[0].reference_state, record)
                            anchor = self._branch_anchor
                        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                            record.update(proposal_status='unavailable', proposal_failure_reason=str(error))
                    factor = FrozenStiffnessFactor(digest.hexdigest(), observation_ids, ratio,
                        tuple(branch.reference_state for branch in branches), anchor, properness)
                if bfs is None:
                    self.factors += (factor,)
                    self.observation_ids = self.observation_ids.union(observation_ids)
                else:
                    event_id = (batch_context.batch_id if batch_context is not None else
                        sensor_event['event_id'] if sensor_event is not None else
                        hashlib.sha256(repr(observation_ids).encode()).hexdigest())
                    event_metadata = dict(source_stamp=state.stamp,
                        source_command=list(state.command), source_prior_state_id=state.state_id,
                        body_posterior_state_ids=metadata['body_posterior_state_ids'],
                        probability_unit=self.stiffness_surprise_settings['event_probability_unit'],
                        sensor_event=None if sensor_event is None else dict(sensor_event))
                    if batch_record is not None:
                        event_metadata.update(probability_unit='declared joint batch of effective sensor events',
                            joint_batch=batch_record, joint_batch_rho0=batch_context.rho_batch,
                            joint_batch_log_keep_probability=batch_context.hazard_logs[0],
                            joint_batch_log_reset_probability=batch_context.hazard_logs[1])
                    self.pending_stiffness_events += ((factor, event_id, event_metadata),)
                    applied_events = self._apply_pending_stiffness_events()
                    record['bayes_factor_surprise'] = applied_events[-1]
                    record['recovered_pending_events'] = applied_events[:-1]
                record.update(status='updated_density_factor', update_applied=True,
                    posterior=self._density_record(), new_factor_id=factor.factor_id,
                    new_factor_properness=properness,
                    new_factor_observation_ids=list(observation_ids))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            retained = self._density_record() if bfs is not None else prior_record
            record.update(status='failed_density_factor', failure_reason=str(error),
                          update_applied=False, posterior=None,
                          retained_prior=retained, prior_retained=True,
                          information_rank=None, information_eigenvalues=None)
            if bfs is not None:
                record.update(stiffness_posterior_defined=False,
                    pending_factor_ids=[factor.factor_id for factor, _, _ in self.pending_stiffness_events],
                    pending_observation_ids=[list(factor.observation_ids)
                                             for factor, _, _ in self.pending_stiffness_events],
                    evidence_failure_diagnostics=(dict(error.diagnostics)
                        if isinstance(error, StiffnessEvidenceUnavailable) else None),
                    recovered_pending_events=[item.record for item in bfs.updates[prior_bfs_event_count:]])
        if bfs is not None and self.pending_stiffness_events and record['posterior'] is not None:
            record.update(status='pending_stiffness_evidence', posterior=None,
                retained_prior=self._density_record(), prior_retained=True,
                stiffness_posterior_defined=False,
                failure_reason='A previously acquired stiffness factor still awaits a valid evidence calculation.',
                pending_factor_ids=[factor.factor_id for factor, _, _ in self.pending_stiffness_events],
                pending_observation_ids=[list(factor.observation_ids)
                                         for factor, _, _ in self.pending_stiffness_events])
        if bfs is not None and record['posterior'] is not None:
            record['stiffness_posterior_defined'] = True
        self.last_stamp = state.stamp
        if batch_context is not None:
            self._joint_batch_signatures[batch_context.batch_id] = batch_context.source_digest
        record['auxiliary_numerical_diagnostics'] = self.numerical_diagnostics()
        self.last_key, self.last_record = key, record
        return record

    def _properness_bounds(self, ratio):
        """Finite ratio bounds on the known compact joint-configuration set.

        The principal relative rotation vector has length at most pi. Local
        translations of these URDF revolute joints are fixed. This proves
        integrability without pretending that global K moments were computed.
        """
        lower_minus, upper_minus, lower_plus, upper_plus = [], [], [], []
        self.robot._fk_update(np.asarray(ratio.branches[0].reference_state.command))
        translations = [(self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]).translation.copy()
                        for _, parent, child, _, _ in self.edges]
        volume_bound = len(self.edges)*2.*math.log(math.pi/2.) if len(ratio.branches) > 1 else 0.
        def bounds(distribution, reference):
            support = distribution.support
            if support.rank != distribution.mean.size:
                raise ValueError('Global stiffness properness needs an explicit all-prediction affine-support proof for singular body priors.')
            radius_squared = 0.
            by_id = dict(zip(reference.edge_ids, reference.transforms))
            for index, (name, *_) in enumerate(self.edges):
                radius_squared += np.linalg.norm(translations[index]-by_id[name].translation
                    -distribution.mean[6*index:6*index+3])**2
                radius_squared += (math.pi+np.linalg.norm(distribution.mean[6*index+3:6*index+6]))**2
            maximum = -.5*(support.rank*math.log(2.*math.pi)+support.log_pdet)
            minimum = maximum-.5*np.linalg.eigvalsh(support.inverse)[-1]*radius_squared
            return float(minimum), float(maximum+volume_bound)
        for branch, (prior, children, _) in zip(ratio.branches, ratio.marginals):
            low, high = bounds(prior, branch.reference_state)
            lower_minus.append(_log_weight(branch)+low); upper_minus.append(_log_weight(branch)+high)
            for child, distribution in children:
                low, high = bounds(distribution, branch.reference_state)
                lower_plus.append(_log_weight(child)+low); upper_plus.append(_log_weight(child)+high)
        lower = float(logsumexp(lower_plus)-logsumexp(upper_minus))
        upper = float(logsumexp(upper_plus)-logsumexp(lower_minus))
        if not np.isfinite(lower+upper):
            raise ValueError('No finite global bounds for the retained stiffness factor were established.')
        return dict(verified=True, log_factor_lower_bound=lower, log_factor_upper_bound=upper,
            argument='positive full-support Gaussian mixtures on compact URDF joint transforms; principal rotation chart volume bounded away from zero',
            auxiliary_selector='fixed measurable stable-equilibrium selector; numerical failures mean unavailable evaluation, never zero likelihood',
            body_density_domain='the specified high-concentration local Gaussian proxy; no global wrapped-rotation accuracy claim')

    def _density_record(self):
        initial = getattr(self, 'initial_components', self.components)
        factors = getattr(self, 'factors', ())
        identity = hashlib.sha256()
        initial_records = []
        for item in initial:
            identity.update(item.distribution.mean.tobytes()+item.distribution.covariance.tobytes())
            identity.update(repr(_log_weight(item)).encode())
            initial_records.append(dict(label=item.label, weight=item.weight, log_weight=_log_weight(item),
                x_est=item.distribution.mean.tolist(), P_est=item.distribution.covariance.tolist()))
        for factor in factors:
            identity.update(factor.factor_id.encode())
        record = dict(density_id=identity.hexdigest(), factor_ids=[item.factor_id for item in factors],
            initial_components=initial_records,
            representation='initial log-K Gaussian mixture times frozen body density-ratio factors',
            normalization_constant=None,
            normalization_definition='Z = integral initial_logK_density(x) * exp(sum_t ell_t(exp(x))) dx',
            global_moments=None,
            properness=dict(verified=all(item.properness['verified'] for item in factors),
                log_normalization_lower_bound=sum(item.properness['log_factor_lower_bound'] for item in factors),
                log_normalization_upper_bound=sum(item.properness['log_factor_upper_bound'] for item in factors)),
            parameterization='x = log(K / (1 N m/rad))', K_units='N m/rad')
        bfs = getattr(self, 'stiffness_bfs', None)
        if bfs is not None:
            # Finite upper/lower bounds prove integrability of the retained
            # expression even when the branch masses used for comparison
            # are local approximations. Do not reuse product-only Z bounds.
            lower = upper = bfs.initial_log_mass
            for update, factor in zip(bfs.updates, factors):
                log_keep, log_reset = update_hazard_log_probabilities(update)
                if log_reset != -np.inf:
                    reset_term = log_reset+update.keep_log_mass
                    if log_keep == -np.inf:
                        lower = upper = update.keep_log_mass
                    else:
                        keep_term = log_keep
                        lower = float(logsumexp([keep_term+lower, reset_term]))
                        upper = float(logsumexp([keep_term+upper, reset_term]))
                lower += factor.properness['log_factor_lower_bound']
                upper += factor.properness['log_factor_upper_bound']
            record.update(density_id=bfs.density_id,
                representation='continuous initial log-K law with every body-ratio factor and Keep/Reset logsumexp recurrence',
                normalization_definition='mass of the retained continuous recurrence; local evidence estimates set branch-relative scales only',
                normalization_log_mass_estimate=bfs.log_mass,
                normalization_estimate_is_exact=False,
                bayes_factor_surprise=bfs.record(),
                bayes_factor_configuration=dict(self.stiffness_surprise_settings),
                properness=dict(verified=all(item.properness['verified'] for item in factors),
                    log_normalization_lower_bound=lower, log_normalization_upper_bound=upper,
                    proof='bounded positive full-joint body ratio factors and finite branch-relative scales; no claim of exact Laplace normalization'))
        return record

    def _prepare_physical_batch(self, factors, x):
        """Evaluate independent frozen continuation problems in one native call.

        Only equal commands and equal physical anchors share a solution.
        Factor distributions, chart correspondences and source IDs stay
        separate and are evaluated below in their original order.
        """
        native = getattr(self, '_auxiliary_native', None)
        if native is None:
            return
        cache = self._physical_query_cache(x)
        groups = {}
        for factor in _atomic_stiffness_factors(factors):
            if factor.ratio.identity or factor.branch_anchor is None:
                continue
            key = self._physical_prediction_key(x, factor.states[0], factor.branch_anchor)
            if key is not None and key not in cache:
                groups.setdefault(key, factor)
        if not groups:
            return
        keys, selected = tuple(groups), tuple(groups.values())
        result = native.evaluate_batch(
            np.repeat(np.asarray(x, dtype=float)[None, :], len(selected), axis=0),
            np.asarray([factor.states[0].command for factor in selected], dtype=float),
            np.asarray([factor.branch_anchor[1] for factor in selected], dtype=float),
            np.asarray([factor.branch_anchor[2] for factor in selected], dtype=float))
        self._evaluation_cache_statistics['native_batch_calls'] += 1
        self._evaluation_cache_statistics['native_batch_rows'] += len(selected)
        for index, key in enumerate(keys):
            diagnostic = result['diagnostics'][index]
            row = {name: result[name][index] for name in
                   ('theta', 'hessian', 'parameter_jacobian', 'active')}
            row.update(diagnostic)
            try:
                self._accept_native_equilibrium(row)
                derivative = (result['derivative'][index] if 'derivative' in result else
                              -np.linalg.solve(row['hessian'], row['parameter_jacobian']))
                if not np.all(np.isfinite(derivative)):
                    raise ValueError('Native auxiliary parameter sensitivity is nonfinite.')
                cache[key] = self._store_physical_prediction(row['theta'], derivative,
                    row['active'], self.last_equilibrium_audit, self.last_continuation_audit)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                # A failure is not a density value and is not silently hidden.
                # Its own factor query reports it; other rows remain distinct.
                cache[key] = dict(error=str(error))

    def _batch_factor_evaluation(self, x, updates):
        by_id = {factor.factor_id: factor for factor in self.factors}
        factors = tuple(by_id[update.factor_id] for update in updates)
        x = np.asarray(x, dtype=float)
        uncached = tuple(factor for factor in factors
            if (factor.factor_id, x.tobytes()) not in self.factor_cache)
        # New batch products need all their own q_i rows as well as the old
        # retained history.  Preparing them together shares only identical
        # command/anchor problems, never a different acquisition command.
        pending = tuple(factor for factor, _, _ in self.pending_stiffness_events)
        self._prepare_physical_batch(uncached+pending, x)
        return tuple(self._factor_evaluation(factor, x) for factor in factors)

    def _factor_evaluation(self, factor, x):
        if factor.ratio.identity:
            return 0., np.zeros_like(np.asarray(x, dtype=float))
        key = (factor.factor_id, np.asarray(x, dtype=float).tobytes())
        if key not in self.factor_cache:
            if isinstance(factor, FrozenTimeIndexedStiffnessFactor):
                x = np.asarray(x, dtype=float)
                self._prepare_physical_batch((factor,), x)
                value, gradient = 0., np.zeros_like(x)
                for part in factor.parts:
                    contribution, derivative = self._factor_evaluation(part, x)
                    value += contribution
                    gradient += derivative
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError('A time-indexed stiffness product has a nonfinite value or score.')
                self.factor_cache[key] = float(value), gradient
                return self.factor_cache[key]
            saved = self._branch_anchor
            try:
                self._branch_anchor = factor.branch_anchor
                first = self._prediction(np.asarray(x), factor.states[0])
                predictions = [first]
                if len(factor.states) > 1:
                    physical = self._physical_query_cache(x).get(
                        self._physical_prediction_key(x, factor.states[0]))
                    if physical is not None and 'error' not in physical:
                        derivative, local_poses = physical['derivative'], physical['local_poses']
                    else:
                        hessian, parameter_jacobian = self.sensitivity.equilibrium_jacobians(
                            first[2], np.asarray(factor.states[0].command), np.exp(x))
                        derivative = -np.linalg.solve(hessian, parameter_jacobian)
                        local_poses = None
                    predictions += [self._body_chart(first[2], derivative, state,
                                                     local_poses=local_poses)
                                    for state in factor.states[1:]]
                self.factor_cache[key] = factor.ratio.evaluate(predictions)
            finally:
                self._branch_anchor = saved
        else:
            statistics = getattr(self, '_evaluation_cache_statistics', None)
            if statistics is not None:
                statistics['factor_cache_hits'] += 1
        return self.factor_cache[key]

    def log_density(self, x):
        """Unnormalized continuous log-K density and its exact query gradient.

        All historical factors remain explicit. Unknown normalization and
        global moments are not replaced by point candidates or a Gaussian.
        """
        x = np.asarray(x, dtype=float)
        bfs = getattr(self, 'stiffness_bfs', None)
        if bfs is not None:
            return bfs.evaluate(x)
        total, gradient, _, _ = _mixture_density(self.initial_components, x)
        for factor in self.factors:
            contribution, derivative = self._factor_evaluation(factor, x)
            total += contribution; gradient = gradient+derivative
        return float(total), gradient

    def command_proposal(self, result):
        """Select and evaluate a local representative of the continuous K law.

        Deterministic prior-derived starts only initialize numerical density
        maximization. They never represent the parameter law or a density
        integral. Every accepted ascent step evaluates the complete retained
        density and its score. The IK still judges the resulting command by
        the current transform distribution's complete reaching criterion.
        Every changed density receives a bounded score update from its best
        stable start, including a previous representative that still beats
        the fixed starts. Only an unchanged density retains that point
        without another refinement.
        All candidate densities are evaluated before current-command stable
        equilibria are resolved in density order. Lower ranked candidates
        cannot change the selected best stable start and need no such solve.
        """
        state = result.prior
        if self.last_record is None or (state.stamp != self.last_record['source_stamp']
                or list(state.command) != self.last_record['source_command']
                or state.state_id != self.last_record['source_prior_state_id']
                or [item.state.state_id for item in result.stages[-1].components]
                    != self.last_record['body_posterior_state_ids']
                or [_log_weight(item) for item in result.stages[-1].components]
                    != self.last_record['body_posterior_log_weights']):
            raise ValueError('A stiffness command proposal must use the current recorded body update.')
        bfs = getattr(self, 'stiffness_bfs', None)
        if bfs is not None and self.last_record.get('posterior') is None:
            self.last_record.update(proposal_status='undefined_stiffness_posterior', command_proposal=None,
                proposal_failure_reason='Pending stiffness evidence has no valid posterior; retained prior is not relabeled as an update.')
            return None
        if not self.factors:
            self.last_record.update(proposal_status='no_informative_density_factor', command_proposal=None)
            return None
        if bfs is not None and not any(not item.ratio.identity for item in self.factors):
            self.last_record.update(proposal_status='no_informative_density_factor', command_proposal=None)
            return None
        if self.last_record.get('command_proposal') is not None:
            return self.last_record['command_proposal']
        if not hasattr(self, '_proposal_candidates'):
            candidates = []
            for item in self.initial_components:
                centre = item.distribution.mean
                scale = np.linalg.cholesky(item.distribution.covariance)
                candidates.append(centre.copy())
                for direction in scale.T:
                    candidates.extend((centre+direction, centre-direction))
                # A single-coordinate start can leave every other weak
                # spring at a joint stop, where its K derivative is zero.
                # These coupled starts use the same prior, without adding
                # a stiffness assumption or changing its continuous law.
                coupled = np.sum(scale, axis=1)
                candidates.extend((centre+coupled, centre-coupled))
            self._proposal_candidates = tuple(candidates)

        maximum_iterations = min(int(self.settings.get('optimizer_max_iterations', 80)),
            int(self.settings.get('representative_max_iterations', 1)))
        gradient_tolerance = float(self.settings.get('optimizer_gradient_tolerance', 1e-6))
        maximum_backtracks = int(self.settings.get('representative_max_backtracks', 8))
        if maximum_iterations < 0 or gradient_tolerance <= 0. or maximum_backtracks < 0:
            raise ValueError('Stiffness representative optimization requires valid iteration limits and tolerance.')
        factor_ids = tuple(factor.factor_id for factor in self.factors)
        previous = getattr(self, '_proposal_representative', None)
        density_id = (None if bfs is None else self.last_record['posterior']['density_id'])
        unchanged_density = (previous is not None and (
            previous['factor_ids'] == factor_ids if bfs is None else
            previous.get('density_id') == density_id))
        evaluations, failures, cache, density_cache = [], [], {}, {}
        density_evaluations, equilibrium_evaluations = 0, 0

        def evaluate_density(x, *, retained=None):
            nonlocal density_evaluations
            x = np.asarray(x, dtype=float)
            key = x.tobytes()
            if key in density_cache:
                return density_cache[key]
            if retained is None:
                density_evaluations += 1
                with np.errstate(over='raise', invalid='raise'):
                    density, score = self.log_density(x)
                score = np.asarray(score, dtype=float)
                if not np.isfinite(density) or not np.all(np.isfinite(score)):
                    raise ValueError('Nonfinite cumulative stiffness density at a representative trial.')
                evaluations.append(dict(logK=x.tolist(), log_unnormalized_density=float(density)))
            else:
                # The factor IDs and initial law are unchanged, so this is
                # exactly the already evaluated density, not a new update.
                density, score = retained['density'], retained['score']
            item = dict(logK=x.copy(), density=float(density), score=np.array(score, copy=True))
            density_cache[key] = item
            return item

        def evaluate(x, *, retained=None):
            nonlocal equilibrium_evaluations
            x = np.asarray(x, dtype=float)
            key = x.tobytes()
            if key in cache:
                return cache[key]
            item = evaluate_density(x, retained=retained)
            saved = self._branch_anchor
            try:
                self._branch_anchor = None
                equilibrium_evaluations += 1
                _, _, equilibrium, active = self._prediction(x, state)
            finally:
                self._branch_anchor = saved
            item = dict(item, equilibrium=np.array(equilibrium, copy=True),
                        active=np.array(active, copy=True),
                        equilibrium_numerical_audit=deepcopy(self.last_equilibrium_audit))
            cache[key] = item
            return item

        selection = dict(
            role='numerical representative for the frozen K command action and initial command proposal; not a discrete K law or density integral',
            candidate_construction='initial-prior modes, plus/minus Cholesky columns and coupled column sums; previous evaluated representative when available',
            method='rank deterministic starts, then bounded full-density score ascent with Armijo backtracking',
            global_mode_claimed=False, maximum_iterations=maximum_iterations,
            gradient_tolerance=gradient_tolerance, maximum_backtracks=maximum_backtracks,
            retained_unchanged_density=unchanged_density,
            factor_ids=list(factor_ids), line_search=[], iterations=0,
            convergence_claimed=False, selected_density_evaluated=True,
            density_evaluation_count=0, unavailable_candidates=failures)
        if unchanged_density:
            try:
                selected = evaluate(previous['logK'], retained=previous)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                self.last_record.update(proposal_status='unavailable', proposal_selection=selection,
                    proposal_failure_reason=str(error), command_proposal=None)
                return None
            base, index = selected, previous['candidate_index']
            selection.update(candidate_count=0, evaluated_count=0,
                retained_representative_bitwise=True, termination_reason='unchanged continuous density',
                local_refinement_trigger='unchanged continuous density',
                local_refinement_skipped=True,
                refinement_skip_reason='no new density factor; retain the evaluated representative bitwise')
        else:
            candidates = list(self._proposal_candidates)
            if previous is not None:
                candidates.append(previous['logK'])
            ranked = []
            for candidate_index, x in enumerate(candidates):
                try:
                    ranked.append((evaluate_density(x), candidate_index))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    failures.append(dict(candidate_index=candidate_index, stage='density', reason=str(error)))
            # The original dense procedure picks the first stable maximum in
            # declared order, then gives a stable previous representative
            # precedence on an exact density tie. Resolve that same ordering
            # without computing equilibria of candidates that cannot win.
            previous_index = len(candidates)-1 if previous is not None else None
            ranked.sort(key=lambda item:(item[0]['density'], item[1] == previous_index), reverse=True)
            best = None
            checked = 0
            for item, candidate_index in ranked:
                checked += 1
                try:
                    best = (evaluate(item['logK']), candidate_index)
                    break
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    failures.append(dict(candidate_index=candidate_index,
                                         stage='current_equilibrium', reason=str(error)))
            selection.update(candidate_count=len(candidates), evaluated_count=int(best is not None),
                density_ranked_count=len(ranked),
                current_equilibrium_candidate_checks=checked,
                lower_ranked_equilibrium_checks_skipped=len(ranked)-checked,
                candidate_evaluation_policy='all exact densities and scores; current stable equilibria only in density rank order until the identical best stable start is known',
                evaluated_count_semantics='stable initial candidates actually resolved; lower ranked candidate stability remains unevaluated')
            if best is None:
                selection['density_evaluation_count'] = density_evaluations
                selection['current_equilibrium_evaluation_count'] = equilibrium_evaluations
                self.last_record.update(proposal_status='unavailable', proposal_selection=selection,
                    proposal_failure_reason='No deterministic stiffness representative start has an evaluable density and stable current equilibrium.',
                    command_proposal=None)
                return None
            base, index = best
            # Winning against finitely many starts does not imply stationarity
            # of the updated continuous density. New evidence must reach the
            # K representative used by IK even when the old point still wins.
            local_refinement = maximum_iterations > 0
            trigger = ('initial representative' if previous is None else
                       'new density factor; refine the best evaluated stable start')
            selection.update(local_refinement_trigger=trigger,
                local_refinement_skipped=not local_refinement,
                refinement_skip_reason=(None if local_refinement else
                    'configured zero representative-refinement iteration budget'))
            selected = base
            centre = sum(item.weight*item.distribution.mean for item in self.initial_components)
            metric = sum(item.weight*(item.distribution.covariance+np.outer(
                item.distribution.mean-centre, item.distribution.mean-centre)) for item in self.initial_components)
            initial_scale = (min(1., 2.*previous['step_scale'])
                if previous is not None and np.array_equal(base['logK'], previous['logK'])
                    and previous['step_scale'] > 0. else 1.)
            selection['termination_reason'] = ('bounded local iteration budget' if local_refinement else
                'configured zero representative-refinement iteration budget')
            for iteration in range(maximum_iterations if local_refinement else 0):
                score = selected['score']
                if np.max(np.abs(score)) <= gradient_tolerance:
                    selection.update(convergence_claimed=True,
                        termination_reason='full-density score satisfies configured tolerance')
                    break
                magnitude = max(1., float(np.max(np.abs(score))))
                scaled_score = score/magnitude
                scaled_step = metric @ scaled_score
                length = math.sqrt(max(0., float(scaled_score @ scaled_step)))
                direction = scaled_step/max(1./magnitude, length)
                slope = float(score @ direction)
                accepted = False
                for backtrack in range(maximum_backtracks+1):
                    scale = initial_scale*(.5**backtrack)
                    entry = dict(iteration=iteration, scale=scale,
                        armijo_bound=selected['density']+1e-4*scale*slope)
                    try:
                        trial = evaluate(selected['logK']+scale*direction)
                        entry['log_unnormalized_density'] = trial['density']
                        accepted = trial['density'] >= entry['armijo_bound']
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                        entry['unavailable_reason'] = str(error)
                    entry['accepted'] = accepted
                    selection['line_search'].append(entry)
                    if accepted:
                        selected, initial_scale = trial, min(1., 2.*scale)
                        break
                selection['iterations'] = iteration+1
                if not accepted:
                    selection['termination_reason'] = 'Armijo line search exhausted; evaluated best start retained'
                    break
            self._proposal_representative = dict(factor_ids=factor_ids,
                logK=selected['logK'].copy(), density=selected['density'],
                score=selected['score'].copy(), candidate_index=index,
                step_scale=(selection['line_search'][-1]['scale']
                    if selection['line_search'] and selection['line_search'][-1]['accepted'] else 0.))
            if bfs is not None:
                self._proposal_representative['density_id'] = density_id

        x, equilibrium, active = selected['logK'], selected['equilibrium'], selected['active']
        selection.update(base_candidate_index=index, base_logK=base['logK'].tolist(),
            base_log_unnormalized_density=base['density'], base_density_score=base['score'].tolist(),
            selected_log_unnormalized_density=selected['density'],
            selected_density_score=selected['score'].tolist(),
            score_step=(x-base['logK']).tolist(),
            score_step_metric='existing initial-prior full covariance; bounded metric ascent with evaluated Armijo acceptance',
            density_improvement_claimed=bool(selected['density'] > base['density']),
            density_evaluation_count=density_evaluations,
            successful_density_evaluation_count=len(evaluations),
            current_equilibrium_evaluation_count=equilibrium_evaluations,
            selected_equilibrium_numerical_audit=selected['equilibrium_numerical_audit'])
        command = np.asarray(state.command)
        proposed = command+(command-equilibrium)
        proposal = dict(command=proposed.tolist(), source_command=list(state.command),
            source_stamp=state.stamp, source_prior_state_id=state.state_id,
            body_posterior_state_ids=self.last_record['body_posterior_state_ids'],
            density_id=(self.last_record['posterior'] or self.last_record['retained_prior'])['density_id'],
            selected_candidate_index=index, selected_logK=x.tolist(),
            selected_K=np.exp(x).tolist(),
            selected_log_unnormalized_density=selected['density'],
            base_candidate_log_unnormalized_density=base['density'],
            density_improvement_claimed=selection['density_improvement_claimed'],
            auxiliary_equilibrium=equilibrium.tolist(), auxiliary_active_joint_limits=active.tolist(),
            auxiliary_equilibrium_numerical_audit=selected['equilibrium_numerical_audit'],
            method='current command plus its equilibrium deflection under an evaluated continuous-density local K representative',
            acceptance='caller must project command limits and compare the complete current body-distribution objective',
            global_mode_claimed=False)
        self.last_record.update(proposal_status='deterministic_density_representative',
            proposal_selection=selection, command_proposal=proposal,
            auxiliary_numerical_diagnostics=self.numerical_diagnostics(),
            feedback_to_body=False, feedback_to_body_or_control=True,
            feedback_to_control='initial command proposal only')
        return proposal
