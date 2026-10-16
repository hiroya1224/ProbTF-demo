"""Auxiliary log-stiffness inference from the frozen body density ratio.

The primary body law is never replaced by an equilibrium prediction.  All
body mixture components and cross covariances enter the density ratio. The
continuous auxiliary log-K density retains every new factor without sampling
or a Gaussian projection. Finite numerical trials serve initial proposals only.
"""

from pathlib import Path
from dataclasses import dataclass
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
from probtf.spherical_law.first_order import reexpress_state
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf.spherical_law.support import (
    require_in_support, require_same_support,
)
from scipy.spatial.transform import Rotation


def _density(distribution, value, support=None):
    support = distribution.support if support is None else support
    error = np.asarray(value)-distribution.mean
    require_in_support(error, support, 'auxiliary prediction in frozen body support')
    score = -support.inverse @ error
    log_value = -.5*(support.rank*math.log(2.*math.pi)+support.log_pdet-error @ score)
    return float(log_value), score


def _mixture_density(components, value, supports=None):
    evaluations = [_density(component.distribution, value,
        None if supports is None else supports[index]) for index, component in enumerate(components)]
    terms = [_log_weight(component)+item[0]
             for component, item in zip(components, evaluations)]
    total = float(logsumexp(terms))
    responsibility = np.exp(np.asarray(terms)-total)
    score = sum(weight*item[1] for weight, item in zip(responsibility, evaluations))
    return total, score, [item[0] for item in evaluations], responsibility


def _log_weight(component):
    """Do not drop an existing branch whose ordinary weight underflows."""
    supplied = getattr(component, 'log_weight', None)
    return (float(supplied) if supplied is not None else
            -np.inf if component.weight == 0 else math.log(component.weight))


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
        minus, minus_score, _, _ = _mixture_density(self.prior_components, value, self.prior_supports)
        if branch is None:
            plus, plus_score, component_logs, _ = _mixture_density(self.components, value, self.component_supports)
        else:
            plus, plus_score = _density(self.components[branch].distribution, value, self.component_supports[branch])
            component_logs = [plus]
        return plus-minus, plus_score-minus_score, [item-minus for item in component_logs]


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
            self.identity &= equal and np.isclose(mass, _log_weight(branch), atol=1e-12, rtol=0.)
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
            log_minus, score_minus = _density(prior, value, support)
            minus_terms.append(_log_weight(branch)+log_minus-volume)
            minus_gradients.append(jacobian.T @ (score_minus-volume_gradient))
            for (child, distribution), child_support in zip(children, child_supports):
                log_plus, score_plus = _density(distribution, value, child_support)
                plus_terms.append(_log_weight(child)+log_plus-volume)
                plus_gradients.append(jacobian.T @ (score_plus-volume_gradient))
        if self.identity:
            return 0., np.zeros(predictions[0][1].shape[1])
        minus, plus = float(logsumexp(minus_terms)), float(logsumexp(plus_terms))
        minus_gradient = np.exp(np.asarray(minus_terms)-minus) @ np.asarray(minus_gradients)
        plus_gradient = np.exp(np.asarray(plus_terms)-plus) @ np.asarray(plus_gradients)
        return plus-minus, plus_gradient-minus_gradient

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
        self.body_edge_ids = tuple(body_edge_ids)
        self.robot = RobotArm(str(urdf_path), tip_link='tool0', base_link='base_link')
        self.spring = LinearSpringModel()
        self.solver = EquilibriumSolver(self.robot, self.spring,
            EquilibriumConfig(maxiter=100, ftol=1e-14, refine_tol=1e-14))
        self.sensitivity = SensitivityCalculator(self.robot, self.spring)
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
        self.observation_ids = frozenset()
        self.prior_source = dict(file=str(config_path), sha256=hashlib.sha256(
            config_path.read_bytes()).hexdigest(), kp_min=limits[0], kp_max=limits[1],
            initialization='deflecomp_core.estimator.initialization initial_log_kp_state/std',
            bounds_role='initial Gaussian construction only; log K support is unbounded')
        self.model_source = dict(urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
            total_mass_kg=self.robot.total_mass, gravity=self.robot.model.gravity.linear.tolist(),
            declaration=settings.get('known_model_source', 'URDF geometry, masses and inertias declared known for the auxiliary model'),
            spring_law='existing deflecomp LinearSpringModel',
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

    def _refine_equilibrium(self, theta, command, stiffness):
        theta = np.asarray(theta).copy()
        # L-BFGS can stop on floating-point potential changes before its torque
        # residual is accurate enough to differentiate. Refine the very same
        # existing potential/KKT equations, without adding an error model.
        for _ in range(24):
            hessian, _ = self.sensitivity.equilibrium_jacobians(theta, command, stiffness)
            if np.linalg.eigvalsh(.5*(hessian+hessian.T))[0] <= 0.:
                raise ValueError('Auxiliary continuation left its stable equilibrium branch.')
            residual = self.robot.tau_gravity(theta)+self.spring.torque(theta, command, stiffness)
            residual[self.sensitivity.last_active_set] = 0.
            norm = float(np.max(np.abs(residual)))
            if norm <= 1e-11:
                break
            direction = np.linalg.solve(hessian, residual)
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
                break
        hessian, parameter_jacobian = self.sensitivity.equilibrium_jacobians(theta, command, stiffness)
        residual = self.robot.tau_gravity(theta)+self.spring.torque(theta, command, stiffness)
        residual[self.sensitivity.last_active_set] = 0.
        if not np.all(np.isfinite(theta)) or np.max(np.abs(residual)) > 1e-9:
            raise ValueError('Auxiliary model did not satisfy the bounded equilibrium KKT residual.')
        if np.linalg.eigvalsh(.5*(hessian+hessian.T))[0] <= 0.:
            raise ValueError('Auxiliary equilibrium is not a stable branch.')
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
            role='auxiliary equilibrium branch only; no body or parameter-prior replacement')

    def _branch_equilibrium(self, x, state):
        command = np.asarray(state.command)
        anchor = getattr(self, '_branch_anchor', None)
        if anchor is None or anchor[0] != state.state_id:
            stiffness = np.exp(x)
            return self._refine_equilibrium(
                self.solver.solve(command, stiffness, theta_init=command), command, stiffness)
        _, start, theta = anchor
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
        count = max(1, int(np.ceil(np.max(np.abs(x-start))/.1)))
        if count > 1000:
            raise ValueError('Auxiliary continuation exceeds its numerical path-length budget.')
        left = start
        result = None
        for fraction in range(1, count+1):
            right = start+(fraction/count)*(x-start)
            result = segment(left, right, theta, 0)
            theta, left = result[0], right
        return result

    def _prediction(self, x, state):
        stiffness = np.exp(x)
        if not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0.):
            raise ValueError('Nonfinite auxiliary stiffness candidate.')
        theta, hessian, parameter_jacobian, active = self._branch_equilibrium(x, state)
        derivative = -np.linalg.solve(hessian, parameter_jacobian)
        value, jacobian = self._body_chart(theta, derivative, state)
        return value, jacobian, theta, active

    def _body_chart(self, theta, derivative, state):
        self.robot._fk_update(theta)
        by_id = {item.edge_id: item for item in state.transforms}
        value = np.zeros(6*len(self.edges)); jacobian = np.zeros((value.size, derivative.shape[1]))
        for index, (name, parent, child, joint_index, axis) in enumerate(self.edges):
            local = self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]
            edge = by_id[name]
            rotation_vector = Rotation.from_matrix(
                quat_to_rotmat(edge.rotation_wxyz).T @ local.rotation).as_rotvec()
            value[6*index:6*index+3] = local.translation-edge.translation
            value[6*index+3:6*index+6] = rotation_vector
            jacobian[6*index+3:6*index+6] = np.outer(
                _right_log_derivative(rotation_vector) @ axis, derivative[joint_index])
        return value, jacobian

    def update(self, result):
        """Retain new likelihood factors, without projecting away information.

        Numerical proposal fits are deliberately separate from this density.
        In particular, inability to find a smooth Laplace mode cannot erase
        an observation or replace its posterior by the preceding prior.
        """
        state = result.prior
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
        key = (digest.hexdigest(), observation_ids,
               tuple((item.label, item.weight, _log_weight(item), item.state.state_id)
                     for item in result.stages[-1].components))
        if key == self.last_key:
            return self.last_record
        if self.last_stamp is not None and state.stamp <= self.last_stamp:
            raise ValueError('An observation time may update stiffness only once and must increase.')
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
                expected_id = reexpress_state(reference, child.distribution.mean,
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
        record = dict(metadata, prior=prior_record, posterior=prior_record,
            status='not_updated', update_applied=False, information_rank=None,
            information_eigenvalues=None, prior_source=self.prior_source,
            auxiliary_model=self.model_source, feedback_to_body=False,
            feedback_to_body_or_control=False, feedback_to_control=None,
            proposal_approximation=None, proposal_status='not_computed', command_proposal=None,
            approximation='specified Bingham-derived local Gaussian body proxies; no Gaussian projection of cumulative stiffness density',
            support_regularization_added=False)
        projection = result.records.get('internal_projection')
        if projection is not None:
            record['body_internal_approximation'] = dict(
                internal_state_id=projection['internal_state_id'],
                source_prior_state_id=projection['source_prior_state_id'],
                approximation=projection['approximation'],
                ratio_semantics=('native transition-prior mixture versus the declared full joint moment approximation after sensor fusion; not a conditional keep-branch likelihood'
                    if projected_numerator is not None else
                    'prior versus the declared full joint moment approximation after sensor fusion; not the unprojected exact observation likelihood'))
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError('New stiffness evidence contains a duplicate observation ID.')
        repeated = self.observation_ids.intersection(observation_ids)
        if repeated:
            raise ValueError('A stiffness density factor cannot count an old observation again: '+repr(sorted(repeated)))
        try:
            ratio = ChartedBodyDensityRatio(branches, self.body_edge_ids,
                                           posterior_branch=projected_numerator)
            record['body_marginal'] = dict(edge_ids=list(self.body_edge_ids), **ratio.record())
            if ratio.identity:
                self.observation_ids = self.observation_ids.union(observation_ids)
                record.update(status='identity_density_ratio', log_density_ratio=0.,
                              information_rank=0, identity_preserved_bitwise=True)
            else:
                if not observation_ids:
                    raise ValueError('A changed posterior without new observation IDs is not new stiffness evidence.')
                properness = self._properness_bounds(ratio)
                # An equilibrium query failure does not become a zero
                # likelihood. The frozen density factor remains in memory.
                try:
                    with np.errstate(over='raise', invalid='raise'):
                        self._freeze_branch(branches[0].reference_state, record)
                    anchor = self._branch_anchor
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    anchor = None
                    record.update(proposal_status='unavailable', proposal_failure_reason=str(error))
                factor = FrozenStiffnessFactor(digest.hexdigest(), observation_ids, ratio,
                    tuple(branch.reference_state for branch in branches), anchor, properness)
                self.factors += (factor,)
                self.observation_ids = self.observation_ids.union(observation_ids)
                record.update(status='updated_density_factor', update_applied=True,
                    posterior=self._density_record(), new_factor_id=factor.factor_id,
                    new_factor_properness=properness,
                    new_factor_observation_ids=list(observation_ids))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            record.update(status='failed_density_factor', failure_reason=str(error),
                          update_applied=False, posterior=None,
                          retained_prior=prior_record, prior_retained=True,
                          information_rank=None, information_eigenvalues=None)
        self.last_stamp = state.stamp
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
        return dict(density_id=identity.hexdigest(), factor_ids=[item.factor_id for item in factors],
            initial_components=initial_records,
            representation='initial log-K Gaussian mixture times frozen body density-ratio factors',
            normalization_constant=None,
            normalization_definition='Z = integral initial_logK_density(x) * exp(sum_t ell_t(exp(x))) dx',
            global_moments=None,
            properness=dict(verified=all(item.properness['verified'] for item in factors),
                log_normalization_lower_bound=sum(item.properness['log_factor_lower_bound'] for item in factors),
                log_normalization_upper_bound=sum(item.properness['log_factor_upper_bound'] for item in factors)),
            parameterization='x = log(K / (1 N m/rad))', K_units='N m/rad')

    def _factor_evaluation(self, factor, x):
        key = (factor.factor_id, np.asarray(x, dtype=float).tobytes())
        if key not in self.factor_cache:
            saved = self._branch_anchor
            try:
                self._branch_anchor = factor.branch_anchor
                first = self._prediction(np.asarray(x), factor.states[0])
                predictions = [first]
                if len(factor.states) > 1:
                    hessian, parameter_jacobian = self.sensitivity.equilibrium_jacobians(
                        first[2], np.asarray(factor.states[0].command), np.exp(x))
                    derivative = -np.linalg.solve(hessian, parameter_jacobian)
                    predictions += [self._body_chart(first[2], derivative, state)
                                    for state in factor.states[1:]]
                self.factor_cache[key] = factor.ratio.evaluate(predictions)
            finally:
                self._branch_anchor = saved
        return self.factor_cache[key]

    def log_density(self, x):
        """Unnormalized continuous log-K density and its exact query gradient.

        All historical factors remain explicit. Unknown normalization and
        global moments are not replaced by point candidates or a Gaussian.
        """
        x = np.asarray(x, dtype=float)
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
        if not self.factors:
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
        unchanged_density = previous is not None and previous['factor_ids'] == factor_ids
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
                        active=np.array(active, copy=True))
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
            current_equilibrium_evaluation_count=equilibrium_evaluations)
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
            method='current command plus its equilibrium deflection under an evaluated continuous-density local K representative',
            acceptance='caller must project command limits and compare the complete current body-distribution objective',
            global_mode_claimed=False)
        self.last_record.update(proposal_status='deterministic_density_representative',
            proposal_selection=selection, command_proposal=proposal,
            feedback_to_body=False, feedback_to_body_or_control=True,
            feedback_to_control='initial command proposal only')
        return proposal
