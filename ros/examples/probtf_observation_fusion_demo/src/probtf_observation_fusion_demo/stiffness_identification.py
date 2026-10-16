"""Auxiliary log-stiffness inference from the frozen body density ratio.

The primary body law is never replaced by an equilibrium prediction.  All
body mixture components and cross covariances enter the density ratio; only
the auxiliary log-K posterior receives a deterministic local Laplace fit.
"""

from pathlib import Path
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np
import rospkg
import yaml
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, ndtr

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
    terms = [(-np.inf if component.weight == 0 else math.log(component.weight))+item[0]
             for component, item in zip(components, evaluations)]
    total = float(logsumexp(terms))
    responsibility = np.exp(np.asarray(terms)-total)
    score = sum(weight*item[1] for weight, item in zip(responsibility, evaluations))
    return total, score, [item[0] for item in evaluations], responsibility


class BodyDensityRatio:
    """A marginal of the actual raw Gaussians, in their common frozen chart."""

    def __init__(self, prior, posterior_components, indices):
        indices = np.asarray(indices, dtype=int)
        self.prior = Gaussian(prior.mean[indices], prior.covariance[np.ix_(indices, indices)])
        self.components = tuple(PosteriorComponent(
            item.label, Gaussian(item.distribution.mean[indices],
                                 item.distribution.covariance[np.ix_(indices, indices)]),
            item.weight, item.lineage) for item in posterior_components)
        if not self.components or not np.isclose(sum(item.weight for item in self.components), 1.):
            raise ValueError('The supplied fused body weights must be normalized.')
        for item in self.components:
            require_same_support(self.prior.support, item.distribution.support,
                                 'frozen body prior/posterior')
            require_in_support(item.distribution.mean-self.prior.mean, self.prior.support,
                               'frozen body posterior affine support')
        self.prior_support = self.prior.support
        self.component_supports = tuple(item.distribution.support for item in self.components)
        self.identity = all(np.array_equal(item.distribution.mean, self.prior.mean)
                            and np.array_equal(item.distribution.covariance, self.prior.covariance)
                            for item in self.components)

    def evaluate(self, value, branch=None):
        if self.identity:
            require_in_support(np.asarray(value)-self.prior.mean, self.prior_support,
                               'identity body density ratio support')
            return 0., np.zeros_like(self.prior.mean), [0.]*len(self.components)
        minus, minus_score = _density(self.prior, value, self.prior_support)
        if branch is None:
            plus, plus_score, component_logs, _ = _mixture_density(self.components, value, self.component_supports)
        else:
            plus, plus_score = _density(self.components[branch].distribution, value, self.component_supports[branch])
            component_logs = [plus]
        return plus-minus, plus_score-minus_score, [item-minus for item in component_logs]


def _right_log_derivative(vector):
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    coefficient = (1./12.+angle**2/720. if angle < 1e-6 else
                   (1.-.5*angle/math.tan(.5*angle))/angle**2)
    return np.eye(3)+.5*cross+coefficient*(cross @ cross)


def _parameter_record(components):
    if len(components) == 1:
        mean, covariance = components[0].distribution.mean, components[0].distribution.covariance
    else:
        mean = sum(item.weight*item.distribution.mean for item in components)
        covariance = sum(item.weight*(item.distribution.covariance+np.outer(
            item.distribution.mean-mean, item.distribution.mean-mean)) for item in components)
    if len(components) == 1:
        median = np.exp(mean)
    else:
        median = []
        for index in range(mean.size):
            centres = [item.distribution.mean[index] for item in components]
            spreads = [math.sqrt(item.distribution.covariance[index, index]) for item in components]
            def cdf(value):
                return sum(item.weight*ndtr((value-centre)/spread)
                           for item, centre, spread in zip(components, centres, spreads))-.5
            width = max(spreads)
            median.append(math.exp(brentq(cdf, min(centres)-12*width, max(centres)+12*width)))
        median = np.asarray(median)
    return dict(x_est=mean.tolist(), P_est=covariance.tolist(), K_median=median.tolist(),
        components=[dict(label=item.label, weight=item.weight,
            x_est=item.distribution.mean.tolist(), P_est=item.distribution.covariance.tolist())
            for item in components],
        representation='Gaussian mixture in log K; x_est/P_est are mixture moments',
        parameterization='x = log(K / (1 N m/rad)); P_est is the full dimensionless log-K covariance',
        K_units='N m/rad')


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
        # It is frozen for the whole density-ratio update. All prior mixture
        # components still enter the parameter density without selection.
        anchor = max(self.components, key=lambda item: item.weight).distribution.mean.copy()
        command = np.asarray(state.command)
        stiffness = np.exp(anchor)
        theta = self.solver.solve(command, stiffness, theta_init=command)
        theta, _, _, active = self._refine_equilibrium(theta, command, stiffness)
        self._branch_anchor = (state.state_id, anchor, theta)
        record['auxiliary_branch_selection'] = dict(
            policy='fixed prior-mode anchor; canonical straight log-K continuation; no evaluation-order warm starts',
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
        self.robot._fk_update(theta)
        by_id = {item.edge_id: item for item in state.transforms}
        value = np.zeros(6*len(self.edges)); jacobian = np.zeros((value.size, x.size))
        for index, (name, parent, child, joint_index, axis) in enumerate(self.edges):
            local = self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]
            edge = by_id[name]
            rotation_vector = Rotation.from_matrix(
                quat_to_rotmat(edge.rotation_wxyz).T @ local.rotation).as_rotvec()
            value[6*index:6*index+3] = local.translation-edge.translation
            value[6*index+3:6*index+6] = rotation_vector
            jacobian[6*index+3:6*index+6] = np.outer(
                _right_log_derivative(rotation_vector) @ axis, derivative[joint_index])
        return value, jacobian, theta, active

    def update(self, result):
        state = result.prior
        if result.frozen_body_prior is None or result.frozen_body_posterior is None:
            raise ValueError('Stiffness requires actual frozen pre-reexpression body Gaussians.')
        raw_prior, raw_posterior = result.frozen_body_prior, result.frozen_body_posterior
        key = (state.state_id, hashlib.sha256(raw_prior.mean.tobytes()
            + raw_prior.covariance.tobytes()).hexdigest(),
            tuple((item.label, item.weight, item.state.state_id) for item in result.stages[-1].components),
            raw_posterior.prior_version, raw_posterior.stamp, tuple(raw_posterior.command),
            tuple(raw_posterior.observation_ids), tuple(
            (item.label, item.weight, hashlib.sha256(item.distribution.mean.tobytes()
             + item.distribution.covariance.tobytes()).hexdigest()) for item in raw_posterior.components))
        if key == self.last_key:
            return self.last_record
        if self.last_stamp is not None and state.stamp <= self.last_stamp:
            raise ValueError('An observation time may update stiffness only once and must increase.')
        if (not np.array_equal(raw_prior.mean, state.mean)
                or not np.array_equal(raw_prior.covariance, state.covariance)):
            raise ValueError('Frozen body prior does not match its source state.')
        if raw_posterior.prior_version is not None and (
                raw_posterior.prior_version != state.version or raw_posterior.stamp != state.stamp
                or tuple(raw_posterior.command) != state.command):
            raise ValueError('Raw posterior and prior source metadata differ.')
        displayed = tuple(result.stages[-1].components)
        if len(displayed) != len(raw_posterior.components):
            raise ValueError('Frozen and displayed body mixture branches differ.')
        for source, item in zip(raw_posterior.components, displayed):
            if (source.label != item.label or source.weight != item.weight
                    or item.state.stamp != state.stamp or item.state.command != state.command):
                raise ValueError('Frozen body posterior and displayed provenance differ.')
            if (np.array_equal(source.distribution.mean, state.mean)
                    and np.array_equal(source.distribution.covariance, state.covariance)
                    and item.state.state_id == state.state_id):
                expected_id = state.state_id
            else:
                expected_id = reexpress_state(state, source.distribution.mean,
                    source.distribution.covariance, version=item.state.version,
                    component_id=item.state.component_id).state_id
            if expected_id != item.state.state_id:
                raise ValueError('Frozen posterior parameters do not reexpress to the displayed state.')
        metadata = dict(source_stamp=state.stamp, source_command=list(state.command),
            source_prior_state_id=state.state_id, source_prior_version=state.version,
            body_posterior_state_ids=[item.state.state_id for item in displayed],
            body_posterior_versions=[item.state.version for item in displayed],
            body_posterior_weights=[item.weight for item in displayed],
            body_posterior_labels=[item.label for item in displayed],
            observation_ids=list(raw_posterior.observation_ids))
        prior_record = _parameter_record(self.components)
        record = dict(metadata, prior=prior_record, posterior=prior_record,
            status='not_updated', update_applied=False, information_rank=0,
            information_eigenvalues=[0.]*self.robot.nv, prior_source=self.prior_source,
            auxiliary_model=self.model_source, feedback_to_body_or_control=False,
            approximation='local deterministic Laplace in log K; no global unimodality guarantee',
            support_regularization_added=False)
        indices = [6*state.edge_ids.index(name)+coordinate for name in self.body_edge_ids
                   for coordinate in range(6)]
        try:
            ratio = BodyDensityRatio(raw_prior, raw_posterior.components, indices)
            record['body_marginal'] = dict(edge_ids=list(self.body_edge_ids), indices=indices,
                prior_mean=ratio.prior.mean.tolist(), prior_covariance=ratio.prior.covariance.tolist(),
                components=[dict(label=item.label, weight=item.weight,
                    mean=item.distribution.mean.tolist(), covariance=item.distribution.covariance.tolist())
                    for item in ratio.components],
                nuisance_policy='unknown marker and other variables integrated out by Gaussian marginal extraction')
            if ratio.identity:
                record.update(status='identity_density_ratio', log_density_ratio=0.,
                              information_rank=0, identity_preserved_bitwise=True)
            else:
                with np.errstate(over='raise', invalid='raise'):
                    self._fit(ratio, state, record)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            record.update(status='failed_support_or_laplace', failure_reason=str(error),
                          update_applied=False, posterior=None,
                          retained_prior=prior_record, prior_retained=True,
                          information_rank=None, information_eigenvalues=None)
        self.last_stamp = state.stamp
        self.last_key, self.last_record = key, record
        return record

    def _fit(self, ratio, state, record):
        prior_components = self.components
        cache = {}
        rejected = []
        rejection_record = dict(count=0, reasons=[])
        record['rejected_auxiliary_trials'] = rejection_record
        def reject(error):
            rejected.append(str(error))
            rejection_record.update(count=len(rejected), reasons=sorted(set(rejected)))
        if state is not None:
            try:
                self._freeze_branch(state, record)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                reject(error)
                raise
        def prediction(x):
            key = tuple(float(value) for value in x)
            if key not in cache:
                cache[key] = self._prediction(np.asarray(x), state)
            return cache[key]
        def objective(x, branch=None):
            try:
                value, jacobian, _, _ = prediction(x)
                log_prior, score_prior, _, _ = _mixture_density(prior_components, x)
                log_ratio, score_ratio, _ = ratio.evaluate(value, branch)
                return -log_prior-log_ratio, -score_prior-jacobian.T @ score_ratio
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                reject(error)
                # Reject this trial in the ordinary deterministic line search.
                # An invalid prediction never receives a replacement density.
                return np.inf, np.zeros_like(x)
        options = dict(maxiter=int(self.settings.get('optimizer_max_iterations', 80)),
                       gtol=float(self.settings.get('optimizer_gradient_tolerance', 1e-6)))
        centre = sum(item.weight*item.distribution.mean for item in prior_components)
        prior_covariance = sum(item.weight*(item.distribution.covariance+np.outer(
            item.distribution.mean-centre, item.distribution.mean-centre)) for item in prior_components)
        scale = np.linalg.cholesky(prior_covariance)
        def optimize(seed, branch=None):
            def whitened(y):
                cost, gradient = objective(centre+scale @ y, branch)
                return cost, scale.T @ gradient
            result = minimize(whitened, np.linalg.solve(scale, seed-centre), jac=True,
                              method='BFGS', options=options)
            result.x = centre+scale @ result.x
            return result
        attempts = []
        record['optimizer_mode_attempts'] = attempts
        record['optimizer_coordinates'] = 'affine whitening by the previous log-K mixture covariance; density objective unchanged'
        seeds = [item.distribution.mean for item in prior_components]
        if not np.isfinite(objective(seeds[0])[0]):
            raise ValueError('The auxiliary prediction at the parameter prior is invalid: '+rejected[-1])
        if len(ratio.components) > 1:
            for branch in range(len(ratio.components)):
                fit = optimize(seeds[0], branch)
                if np.all(np.isfinite(fit.x)):
                    seeds.append(fit.x)
        modes, rejected_modes = [], []
        expanded_search = False
        def extend_search():
            nonlocal expanded_search
            if not expanded_search:
                # Additional deterministic starts come from the existing prior
                # covariance. They do not change its density or add a prior.
                for column in scale.T:
                    seeds.extend((centre+column, centre-column))
                expanded_search = True
        step = float(self.settings.get('laplace_fd_step', 1e-4))
        for seed in seeds:
            fit = optimize(seed)
            value, gradient = objective(fit.x)
            scaled_gradient = scale.T @ gradient
            attempts.append(dict(x=fit.x.tolist(), objective=float(value),
                gradient=gradient.tolist(), whitened_gradient=scaled_gradient.tolist(),
                iterations=int(fit.nit), optimizer_message=str(fit.message)))
            if not np.isfinite(value) or np.linalg.norm(scaled_gradient, ord=np.inf) > max(1e-4, 10*options['gtol']):
                rejected_modes.append('MAP did not reach a stationary point: '+str(fit.message))
                extend_search()
                continue
            if any(np.linalg.norm(fit.x-item[0]) < 1e-3 for item in modes):
                continue
            differences = [(objective(fit.x+step*np.eye(fit.x.size)[j]),
                            objective(fit.x-step*np.eye(fit.x.size)[j]))
                           for j in range(fit.x.size)]
            if any(not np.isfinite(item[0]) for pair in differences for item in pair):
                rejected_modes.append('Laplace neighbourhood leaves the valid prediction support.')
                extend_search()
                continue
            active = prediction(fit.x)[3]
            if any(not np.array_equal(prediction(fit.x+sign*step*np.eye(fit.x.size)[j])[3], active)
                   for j in range(fit.x.size) for sign in (-1., 1.)):
                rejected_modes.append('Laplace neighbourhood crosses an auxiliary joint-limit active-set boundary.')
                extend_search()
                continue
            hessian = np.column_stack([(plus[1]-minus[1])/(2*step) for plus, minus in differences])
            hessian = .5*(hessian+hessian.T)
            if np.linalg.eigvalsh(hessian)[0] <= 0.:
                rejected_modes.append('Laplace Hessian is not positive definite; no jitter is allowed.')
                extend_search()
                continue
            covariance = np.linalg.inv(hessian)
            evidence = -value+.5*fit.x.size*math.log(2*math.pi)-.5*np.linalg.slogdet(hessian)[1]
            modes.append((fit.x.copy(), covariance, evidence, hessian))
        if not modes:
            raise ValueError('No justified local Laplace mode was found: '+'; '.join(rejected_modes))
        weights = np.exp(np.array([item[2] for item in modes])-logsumexp([item[2] for item in modes]))
        components = tuple(PosteriorComponent('logK_laplace_mode_{}'.format(index),
            Gaussian(item[0], item[1]), float(weight)) for index, (item, weight) in enumerate(zip(modes, weights)))
        chosen = int(np.argmax(weights))
        x, covariance, _, hessian = modes[chosen]
        # This is the likelihood's local observed information, without clipping
        # negative curvature or claiming every parameter is identifiable.
        prior_hessian = np.column_stack([(-_mixture_density(prior_components, x+step*np.eye(x.size)[j])[1]
            + _mixture_density(prior_components, x-step*np.eye(x.size)[j])[1])/(2*step)
            for j in range(x.size)])
        information = .5*((hessian-prior_hessian)+(hessian-prior_hessian).T)
        values, vectors = np.linalg.eigh(information)
        tolerance = 1e-7*max(1., float(np.max(np.abs(values))))
        value, _, theta, active = prediction(x)
        log_ratio, _, branch_ratios = ratio.evaluate(value)
        record.update(status='updated_local_laplace' if len(modes)==1 else 'updated_multimodal_laplace',
            update_applied=True, posterior=_parameter_record(components),
            information_rank=int(np.sum(np.abs(values)>tolerance)),
            information_eigenvalues=values.tolist(), information_matrix=information.tolist(),
            unidentifiable_directions=vectors[:, np.abs(values)<=tolerance].tolist(),
            log_density_ratio=float(log_ratio), body_component_log_density_ratios=branch_ratios,
            auxiliary_equilibrium=theta.tolist(), auxiliary_active_joint_limits=active.tolist(),
            auxiliary_prediction_chart=value.tolist(), model_evaluation_count=len(cache),
            retained_logK_modes=len(modes), mode_search='previous parameter modes, every body branch seed, and prior-covariance deterministic starts when needed; no global completeness claim')
        record['rejected_mode_candidates'] = rejected_modes
        self.components = components
