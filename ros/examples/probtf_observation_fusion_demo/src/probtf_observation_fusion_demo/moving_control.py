"""Complete relative objective and legacy short-command selection.

The only extrapolation is the input family's declared right joint action.
Every trial retains the complete joint covariance, including body/marker
blocks.  The marker variable is unchanged while a future body command is
queried. The running-gradient API evaluates the objective independently of
endpoint selection. The legacy comparison path retains one projected update
and backtracking line search; it does not imply convergence.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from probtf.bingham import bingham_log_normalizer
from probtf.geometry import (
    axis_angle_to_quat, quat_mul, quat_right_matrix, quat_to_rotmat,
    rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    point_query, pose_query,
)
from probtf.spherical_law.support import covariance_support, require_in_support
from .joint_batch_fusion import add_log_evidence_parts, log_evidence_parts_difference


@lru_cache(maxsize=4096)
def _log_normalizer(eigenvalues, integration_steps):
    return float(bingham_log_normalizer(np.diag(eigenvalues),
                                       integration_steps=integration_steps))


def command_state(state, command, body_ids, actions, *, joint_increment=None):
    """Push the same posterior through its declared deterministic right action.

    R exp(zeta) D = R D exp(D.T zeta); translations use parent coordinates.
    Thus all covariance blocks are transported by diag(I, D.T), including
    cross blocks. ``joint_increment`` can supply the explicitly frozen
    estimated-K equilibrium increment while metadata keeps the sent command.
    Reconstructing independent marginals here would be wrong.
    """
    command = np.asarray(command, dtype=float)
    current = np.asarray(state.command, dtype=float)
    if command.shape != current.shape or not np.all(np.isfinite(command)):
        raise ValueError("A trial command must match the finite source command.")
    increment = command-current if joint_increment is None else np.asarray(joint_increment, dtype=float)
    if increment.shape != command.shape or not np.all(np.isfinite(increment)):
        raise ValueError("A declared joint action must contain one finite rotation per command.")
    if np.array_equal(command, current):
        if np.any(increment != 0.):
            raise ValueError("An unchanged command must preserve its anchored candidate distribution.")
        return state
    action_by_id = dict(zip(body_ids, actions))
    chart = np.eye(state.mean.size)
    transforms = []
    for index, edge in enumerate(state.transforms):
        action = action_by_id.get(edge.edge_id)
        if action is None:
            transforms.append(edge)
            continue
        joint = int(action["joint_index"])
        if increment[joint] == 0.:
            transforms.append(edge)
            continue
        delta = axis_angle_to_quat(action["axis"], increment[joint])
        right = quat_right_matrix(delta)
        rotation = quat_to_rotmat(delta)
        chart[6*index+3:6*index+6, 6*index+3:6*index+6] = rotation.T
        parameter = right @ edge.bingham @ right.T
        parameter = .5*(parameter+parameter.T)
        transforms.append(BinghamTransform(
            edge.edge_id, edge.translation, parameter,
            edge.translation_covariance))
    covariance = chart @ state.covariance @ chart.T
    return JointTransformState(tuple(transforms), covariance, version=state.version,
                               stamp=state.stamp, command=tuple(command),
                               component_id=state.component_id)


def _component_log_weight(component):
    value = getattr(component, 'log_weight', None)
    return float(value if value is not None else (
        np.log(component.weight) if component.weight > 0. else -np.inf))


def _relative_mixture_log_parts(branches):
    """Subtract the common density scale before adding native log weights.

    A float alone cannot retain a small log mixture correction beside a huge
    common log PDF. The pair represents the same scalar criterion and allows
    its differences to retain that correction without modifying any mass.
    """
    active=[branch for branch in branches if branch['log_weight'] != -np.inf]
    if not active:
        raise ValueError('The complete relative mixture has no positive finite mass.')
    reference=max((tuple(branch['log_density_parts']) for branch in active),
                  key=lambda parts:(parts[0],parts[1]))
    if not np.all(np.isfinite(reference)):
        raise ValueError('The complete relative mixture density is not finite.')
    terms=[log_evidence_parts_difference(branch['log_density_parts'],reference)
           +branch['log_weight'] for branch in branches]
    normalizer=float(logsumexp(terms))
    if not np.isfinite(normalizer):
        raise ValueError('The complete relative mixture density is not finite.')
    return add_log_evidence_parts(reference,(normalizer,))


def _bounded_gradient_direction(metric, gradient, baseline, lower, upper):
    """Precondition the complete-criterion gradient within command bounds.

    The quadratic is a numerical step model, not an additional task
    criterion. The caller checks the actual complete criterion on each
    line-search trial.
    """
    metric, gradient, baseline, lower, upper = map(np.asarray,
        (metric, gradient, baseline, lower, upper))
    scale = np.maximum(np.maximum(upper-baseline, baseline-lower), 1e-12)
    quadratic = scale[:, None]*metric*scale[None, :]
    linear = scale*gradient
    normalization = max(1., float(np.linalg.norm(quadratic, ord=np.inf)),
                        float(np.linalg.norm(linear, ord=np.inf)))
    quadratic, linear = quadratic/normalization, linear/normalization
    solved = minimize(lambda value: .5*value @ quadratic @ value+linear @ value,
        np.zeros_like(baseline), jac=lambda value: quadratic @ value+linear,
        bounds=list(zip((lower-baseline)/scale, (upper-baseline)/scale)),
        method='SLSQP',
        options=dict(maxiter=80, ftol=1e-12))
    direction = scale*solved.x
    feasible = (np.all(np.isfinite(direction))
        and np.all(baseline+direction >= lower-1e-12)
        and np.all(baseline+direction <= upper+1e-12))
    accepted = bool(feasible and gradient @ direction < 0.)
    if not accepted:
        direction = np.zeros_like(baseline)
    return direction, dict(method='coupled box-constrained Fisher quadratic',
        solver='deterministic SLSQP with analytic quadratic gradient',
        solver_success=bool(solved.success), solver_status=int(solved.status),
        optimizer_success_is_acceptance_gate=False,
        solver_message=str(solved.message), iterations=int(solved.nit),
        constraints_satisfied=bool(feasible),
        accepted=accepted,
        objective_slope=float(gradient @ direction),
        fallback='zero direction at the feasible initial point' if not accepted else None)


def switch_goal_queries(state, marker_path, goal_offset, goal_orientation,
                        axis_length, *, frame, query_prefix='switch_goal'):
    """Query the known marker-to-switch action on this exact goal source.

    These root-frame points describe the target origin and its three axes.
    They are drawing/recording queries, never a second goal distribution.
    The controller evaluates the same attachment in marker-relative space.
    """
    rotation = quat_to_rotmat(goal_orientation)
    endpoints = (('origin', np.zeros(3)),
                 ('x', np.array([axis_length, 0., 0.])),
                 ('y', np.array([0., axis_length, 0.])),
                 ('z', np.array([0., 0., axis_length])))
    return tuple((axis, point_query(state, marker_path,
        np.asarray(goal_offset)+rotation @ endpoint,
        query_id=query_prefix+'/'+axis, frame=frame))
        for axis, endpoint in endpoints)


def control_components(result):
    """Return the internal law and the explicitly retained goal query laws.

    A goal branch may change the environment conditional and its cross blocks,
    but its entire body marginal must be the very same internal approximation.
    This check also prevents a display-only goal law from entering control.
    Legacy stages without a separate goal family keep their original path.
    """
    stage = result.stages[-1]
    internal = tuple(stage.components)
    goals = tuple(getattr(stage, 'goal_components', ()))
    if not goals:
        return internal, internal
    if (len(internal) != 1 or internal[0].weight != 1.
            or _component_log_weight(internal[0]) != 0.):
        raise ValueError('Retained goal candidates require one normalized internal body law.')
    if (any(not np.isfinite(item.weight) or item.weight < 0. for item in goals)
            or not np.isclose(sum(item.weight for item in goals), 1.)
            or not np.isclose(logsumexp([_component_log_weight(item) for item in goals]), 0., atol=1e-12)):
        raise ValueError('Retained goal candidate weights and log weights must be normalized.')
    source = internal[0].state
    marker_ids = {item if isinstance(item, str) else item[0] for item in result.marker_path}
    if not marker_ids or not marker_ids.issubset(source.edge_ids):
        raise ValueError('Retained goals require the declared environment marker path.')
    body_indices = [index for index, name in enumerate(source.edge_ids) if name not in marker_ids]
    coordinates = np.array([6*index+offset for index in body_indices for offset in range(6)], dtype=int)
    body_covariance = source.covariance[np.ix_(coordinates, coordinates)]
    for candidate in goals:
        state = candidate.state
        if (state.edge_ids != source.edge_ids or state.command != source.command
                or state.stamp != source.stamp or state.version != source.version
                or not np.array_equal(state.covariance[np.ix_(coordinates, coordinates)], body_covariance)):
            raise ValueError('Every retained goal must share the same internal body marginal and metadata.')
        for index in body_indices:
            actual, expected = state.transforms[index], source.transforms[index]
            if (not np.array_equal(actual.translation, expected.translation)
                    or not np.array_equal(actual.translation_covariance, expected.translation_covariance)
                    or not np.array_equal(actual.rotation_wxyz, expected.rotation_wxyz)
                    or (actual.bingham is None) != (expected.bingham is None)
                    or (actual.bingham is not None
                        and not np.array_equal(actual.bingham, expected.bingham))):
                raise ValueError('Every retained goal must share the same internal body marginal and metadata.')
    return internal, goals


def _relative_density(query, goal_offset, goal_orientation, integration_steps):
    """Full positional normalizer and normalized Bingham orientation density.

    This is an explicit final comparison proxy of the propagated relative
    pose.  Its Bingham rotation block is derived from the same core query.
    The O(P) orientation mean correction shifts the modal quaternion.  Its
    effect on covariance transport is O(P**2), outside the specified order.
    No independent body/goal marginal convolution is performed.
    """
    covariance = query.covariance[:3, :3]
    support = covariance_support(covariance, "relative hand position")
    difference = np.asarray(goal_offset)-query.mean[:3]
    require_in_support(difference, support, "relative target position")
    position_parts=add_log_evidence_parts((0.,0.),(
        -.5*support.rank*np.log(2.0*np.pi),-.5*support.log_pdet,
        -.5*float(difference @ support.inverse @ difference)))
    log_position=float(position_parts[0]+position_parts[1])
    mean_orientation = quat_mul(query.rotation_wxyz,
        rotation_vector_to_quaternion(query.correction[3:]))
    parameter = bingham_from_rotation_coordinates(
        mean_orientation, query.covariance[3:, 3:])
    eigenvalues = np.linalg.eigvalsh(parameter)
    shift = float(eigenvalues[-1])
    log_z_parts=add_log_evidence_parts((_log_normalizer(tuple(eigenvalues-shift), int(integration_steps)),0.),(shift,))
    log_z=float(log_z_parts[0]+log_z_parts[1])
    orientation_parts=add_log_evidence_parts((float(goal_orientation @ parameter @ goal_orientation),0.),
        (-log_z_parts[0],-log_z_parts[1]))
    log_orientation=float(orientation_parts[0]+orientation_parts[1])
    density_parts=add_log_evidence_parts(position_parts,orientation_parts)
    return float(density_parts[0]+density_parts[1]), {
        "log_position": float(log_position), "log_position_pdet": support.log_pdet,
        "log_orientation": log_orientation, "log_bingham_normalizer": float(log_z),
        "log_position_parts": list(position_parts),
        "log_orientation_parts": list(orientation_parts),
        "log_bingham_normalizer_parts": list(log_z_parts),
        "log_density_parts": list(density_parts),
        "relative_position_mean": query.mean[:3].tolist(),
        "relative_position_covariance": covariance.tolist(),
        "relative_orientation_mean_wxyz": mean_orientation.tolist(),
        "relative_orientation_bingham": parameter.tolist(),
        "query_mean": query.mean.tolist(),
        "query_covariance": query.covariance.tolist(),
        "query_within": query.within.tolist(),
        "query_between": query.between.tolist(),
        "query_id": query.query_id, "query_frame": query.frame,
        "query_stamp": float(query.stamp), "query_version": int(query.version),
        "query_command": list(query.command),
        "state_id": query.state_id,
    }


@dataclass(frozen=True)
class GradientEvaluation:
    """Immutable complete-objective result, including explicit failed results.

    The serialised record owns its data. ``record`` returns a fresh copy, so a
    consumer cannot change the numerical result or its captured provenance.
    A failed evaluation has ``gradient=None``, never an artificial zero input.
    """
    record_json: str
    packet_id: str

    @property
    def record(self):
        return json.loads(self.record_json)

    @property
    def gradient(self):
        value = self.record['gradient']
        return None if value is None else tuple(value)

    @property
    def status(self):
        return self.record['status']


class MovingController:
    """One sensor-independent selection rule; no physical state enters it."""

    def __init__(self, inference, robot_description, goal_offset,
                 goal_orientation_wxyz, motion, integration_steps=480):
        self.body_ids = tuple(inference.body_ids)
        self.actions = tuple(dict(action) for action in inference.actions)
        self.hand_path = tuple(inference.paths["hand"])
        marker_path = tuple(inference.paths["marker"])
        self.marker_path = marker_path
        self.relative_path = tuple((name, not inverse)
                                   for name, inverse in reversed(marker_path)) + self.hand_path
        self.root_frame = str(inference.settings["root_frame"])
        self.goal_axis_length = float(inference.settings.get('display', {}).get(
            'axis_length', 0.07500000111758709))
        self.goal_offset = np.asarray(goal_offset, dtype=float).copy()
        self.goal_orientation = np.asarray(goal_orientation_wxyz, dtype=float).copy()
        if (self.goal_offset.shape != (3,) or self.goal_orientation.shape != (4,)
                or not np.all(np.isfinite(self.goal_offset))
                or not np.all(np.isfinite(self.goal_orientation))
                or not np.isclose(np.linalg.norm(self.goal_orientation), 1.0)):
            raise ValueError("Known marker/goal attachment requires a point and unit quaternion.")
        joints = [joint for joint in ET.fromstring(robot_description).findall("joint")
                  if joint.get("type") == "revolute"]
        self.lower = np.array([float(joint.find("limit").get("lower")) for joint in joints])
        self.upper = np.array([float(joint.find("limit").get("upper")) for joint in joints])
        self.speed = float(motion["max_joint_velocity"])
        self.acceleration = float(motion["max_joint_acceleration"])
        self.gradient_step = float(motion.get("gradient_step", 1e-5))
        configured_backtracks = int(motion.get("gradient_max_backtracks", 8))
        self.max_backtracks = max(8, configured_backtracks)
        self.damping = float(motion.get("damping", 0.025))
        self.integration_steps = int(integration_steps)
        if (min(self.speed, self.acceleration, self.gradient_step, self.damping) <= 0.0
                or self.integration_steps <= 0 or configured_backtracks < 0):
            raise ValueError("Positive command limits and numerical solver settings are required.")
        self.velocity = np.zeros(len(joints))
        self.visual_approach_settings = dict(motion.get('visual_approach', {}))
        self.visual_approach_policy = None

    def reaching_step(self, result, dt, *, command_map=None):
        """Use the complete IK throughout; centering only proposes a start."""
        if not result.records.get('fixed_goal_memory', {}).get('available', False):
            command, diagnostics = self.step(result, dt, command_map=command_map)
            diagnostics.update(reaching_phase='indet_ik', visual_approach=dict(
                phase='unavailable', selection_reason='no observed marker goal is available',
                physical_truth_received=False, monte_carlo_used=False))
            return command, diagnostics
        if self.visual_approach_policy is None:
            from .visual_approach import VisualApproach
            self.visual_approach_policy = VisualApproach(self, self.visual_approach_settings)
        target, record = self.visual_approach_policy.propose(result, dt, command_map=command_map)
        command, diagnostics = self.step(result, dt, command_map=command_map,
                                         initial_command=target)
        record.update(proposal_criterion=record['selection_criterion'],
            selection_criterion='complete normalized Gaussian/Bingham criterion at every reaching update',
            full_gaussian_bingham_criterion_used_for_selection=True,
            full_gaussian_bingham_criterion_role='selection in every reaching update; visual loss proposes an initial point only',
            visual_velocity_executed_directly=False,
            initial_proposal=diagnostics['approach_initial_proposal'])
        diagnostics.update(reaching_phase='indet_ik', visual_approach=record)
        return command, diagnostics

    def _query(self, state):
        return pose_query(state, self.relative_path,
                          query_id="control:marker_relative_hand", frame="marker")

    def _evaluate_command_objective(self, components, command, command_map, *, trial_states=None):
        """Evaluate the unchanged relative density, independently of an IK step.

        All branches use the same joint law and captured command action. The
        log-sum-exp includes every mixture log weight, position normalizer,
        propagated covariance, and Bingham normalizer. The pose query retains
        the complete joint covariance before the final comparison proxy.
        """
        branches, queries = [], []
        for component in components:
            trial_state = (command_state(component.state, command,
                self.body_ids, self.actions) if command_map is None else
                command_map.push(component.state, command, self.body_ids, self.actions))
            if trial_states is not None:
                trial_states.append(trial_state)
            query = self._query(trial_state)
            queries.append(query)
            log_density, branch = _relative_density(query, self.goal_offset,
                self.goal_orientation, self.integration_steps)
            log_weight = _component_log_weight(component)
            branch.update(label=component.label, weight=float(component.weight),
                log_weight=log_weight, state_id=trial_state.state_id,
                log_density_parts=list(branch.get('log_density_parts',(log_density,0.))),
                parent_state_id=component.state.state_id,
                parent_command=list(component.state.command),
                parent_stamp=float(component.state.stamp))
            branches.append(branch)
        parts=_relative_mixture_log_parts(branches)
        return float(parts[0]+parts[1]), branches, tuple(queries)

    def _objective_source(self, result, command_map):
        body, goals = control_components(result)
        command = np.asarray(result.prior.command, dtype=float)
        if (command.shape != self.lower.shape or not np.all(np.isfinite(command))
                or np.any(command < self.lower) or np.any(command > self.upper)):
            raise ValueError('The source command must be finite and within physical joint positions.')
        if (not body or not goals
                or any(not np.isfinite(item.weight) or item.weight < 0. for item in goals)
                or not np.isclose(sum(item.weight for item in goals), 1.)
                or not np.isclose(logsumexp([_component_log_weight(item) for item in goals]), 0., atol=1e-12)):
            raise ValueError('A complete-objective source requires normalized candidate shape and log weights.')
        for component in body+goals:
            if (component.state.stamp != result.prior.stamp
                    or component.state.command != result.prior.command):
                raise ValueError('Objective candidates must share the frozen source command and stamp.')
        if command_map is not None:
            command_map.validate_result(result)
        return body, goals, command

    def objective_at_command(self, result, command, *, command_map=None):
        """Return full J at one candidate, with no endpoint selection or clipping.

        The caller supplies the already history-transported snapshot. Only
        physical joint positions bound this query; speed, acceleration, the
        old endpoint duration, a quadratic step and line search play no role.
        The scalar is a rounded summary; each branch retains compensated log
        parts, which the gradient API subtracts before discarding any scale.
        """
        _, components, _ = self._objective_source(result, command_map)
        command = np.asarray(command, dtype=float)
        if (command.shape != self.lower.shape or not np.all(np.isfinite(command))
                or np.any(command < self.lower) or np.any(command > self.upper)):
            raise ValueError('An objective query must be finite and within physical joint positions.')
        log_density, branches, _ = self._evaluate_command_objective(components, command, command_map)
        objective = -log_density
        if not np.isfinite(objective):
            raise ValueError('The complete relative objective is not finite at the candidate command.')
        return objective, branches

    def gradient_at_command(self, result, source_command=None, *, command_map=None,
                            evaluation_observer=None):
        """Differentiate full J at the snapshot's sent command, not an IK endpoint.

        Each side is restricted only by physical joint positions and the
        captured action's actual evaluation domain. Unequal two-sided steps
        use the derivative of their quadratic interpolant; one successful
        side gives an explicitly reported first-order difference. If a joint
        has no evaluable side the whole result fails, without zero filling.
        The same immutable source and stable-branch action serve all probes.
        """
        started = perf_counter()
        record = dict(gradient=None, source_command=None,
            source_stamp=float(result.prior.stamp), body_state_ids=[], goal_state_ids=[],
            parameter_density_id=None, selected_K=None, objective_at_source=None,
            computation_seconds=0., finite_difference_diagnostics=dict(
                method='physical-position-aware two-sided/one-sided full-objective finite differences',
                log_difference_arithmetic='compensated common-scale subtraction; native log weights added after density-scale subtraction',
                configured_step=self.gradient_step, evaluations=[], coordinates=[]),
            status='failed', reason=None, physical_truth_received=False,
            monte_carlo_used=False, endpoint_selection_performed=False,
            fixed_goal_memory_available=bool(result.records.get('fixed_goal_memory', {}).get('available', False)),
            source_observation_ids=list(result.records.get('observation_ids', ())))
        diagnostics = record['finite_difference_diagnostics']
        try:
            body, components, command = self._objective_source(result, command_map)
            record.update(source_command=command.tolist(),
                body_state_ids=[item.state.state_id for item in body],
                goal_state_ids=[item.state.state_id for item in components],
                body_versions=[int(item.state.version) for item in body],
                goal_versions=[int(item.state.version) for item in components],
                body_weights=[float(item.weight) for item in body],
                goal_weights=[float(item.weight) for item in components],
                body_log_weights=[_component_log_weight(item) for item in body],
                goal_log_weights=[_component_log_weight(item) for item in components])
            requested = command if source_command is None else np.asarray(source_command, dtype=float)
            if requested.shape != command.shape or not np.array_equal(requested, command):
                raise ValueError('The gradient evaluation command must equal the snapshot sent command; transport history first.')
            action_record = None if command_map is None else command_map.record()
            if action_record is not None:
                record.update(parameter_density_id=action_record['density_id'],
                    selected_K=list(action_record['selected_K']))
            record.update(command_action=action_record,
                command_propagation=('declared rigid joint action; no stiffness representative selected'
                    if command_map is None else 'captured estimated-K anchored equilibrium joint action'))
            cache = {};objective_parts_by_command={}

            def evaluate(candidate):
                key = tuple(float(value) for value in candidate)
                if key not in cache:
                    attempt = dict(command=list(key), status='failed', reason=None)
                    diagnostics['evaluations'].append(attempt)
                    try:
                        if evaluation_observer is None:
                            log_density, branches, _ = self._evaluate_command_objective(
                                components, np.asarray(candidate), command_map)
                        else:
                            states=[]
                            log_density, branches, _ = self._evaluate_command_objective(
                                components, np.asarray(candidate), command_map,trial_states=states)
                            # Selection-layer observers receive the exact
                            # already-transported laws, never an independent
                            # copy or an alternative task objective.
                            evaluation_observer(np.asarray(candidate),components,tuple(states))
                        objective = -log_density
                        if not np.isfinite(objective):
                            raise ValueError('The complete relative objective is not finite.')
                        log_parts=_relative_mixture_log_parts(branches)
                        parts=(-log_parts[0],-log_parts[1])
                        objective_parts_by_command[key]=parts
                        cache[key] = (objective, branches)
                        attempt.update(status='ok', objective=objective,objective_parts=list(parts))
                    except Exception as error:
                        attempt['reason'] = type(error).__name__+': '+str(error)
                        raise
                return cache[key]

            objective, branches = evaluate(command)
            source_parts=objective_parts_by_command[tuple(command)]
            record.update(objective_at_source=objective,objective_at_source_parts=list(source_parts),source_branches=branches)
            gradient = []
            for index in range(command.size):
                coordinate = dict(joint_index=index, source_position=float(command[index]),
                    position_lower=float(self.lower[index]), position_upper=float(self.upper[index]),
                    forward=None, backward=None, method=None, derivative=None)
                diagnostics['coordinates'].append(coordinate)
                successful = {}
                for side, boundary, sign in (('forward', self.upper[index], 1.),
                                              ('backward', self.lower[index], -1.)):
                    value = (min(boundary, command[index]+self.gradient_step) if sign > 0.
                             else max(boundary, command[index]-self.gradient_step))
                    step = float(value-command[index])
                    probe_record = dict(step=step, status='outside_position_domain', reason=None)
                    coordinate[side] = probe_record
                    if step == 0.:
                        continue
                    candidate = command.copy()
                    candidate[index] = value
                    try:
                        probed, _ = evaluate(candidate)
                        parts=objective_parts_by_command[tuple(candidate)]
                        difference=log_evidence_parts_difference(parts,source_parts)
                        successful[side] = (abs(step), difference)
                        probe_record.update(status='ok', objective=probed,objective_parts=list(parts),
                            objective_difference_from_source=difference)
                    except Exception as error:
                        probe_record.update(status='evaluation_failed',
                            reason=type(error).__name__+': '+str(error))
                if len(successful) == 2:
                    forward, plus_difference = successful['forward']
                    backward, minus_difference = successful['backward']
                    derivative = (backward*plus_difference/forward
                        - forward*minus_difference/backward)/(forward+backward)
                    coordinate['method'] = 'two-sided unequal-step quadratic derivative'
                elif 'forward' in successful:
                    forward, plus_difference = successful['forward']
                    derivative = plus_difference/forward
                    coordinate['method'] = 'one-sided forward derivative'
                elif 'backward' in successful:
                    backward, minus_difference = successful['backward']
                    derivative = -minus_difference/backward
                    coordinate['method'] = 'one-sided backward derivative'
                else:
                    raise ValueError('No finite objective difference is available for joint '+str(index)+'.')
                if not np.isfinite(derivative):
                    raise ValueError('The complete-objective derivative is nonfinite for joint '+str(index)+'.')
                coordinate['derivative'] = float(derivative)
                gradient.append(float(derivative))
            # The map rejects epoch changes in push(); this final validation
            # also detects source/proposal edits before publishing a result.
            if command_map is not None:
                command_map.validate_result(result)
                if command_map.record() != action_record:
                    raise ValueError('The captured stiffness command action changed during differentiation.')
            record.update(gradient=gradient, status='ok')
        except Exception as error:
            record.update(gradient=None, status='failed',
                reason=type(error).__name__+': '+str(error))
        record['computation_seconds'] = perf_counter()-started
        diagnostics['objective_evaluation_count'] = len(diagnostics['evaluations'])
        encoded = json.dumps(record, sort_keys=True, separators=(',', ':'))
        return GradientEvaluation(encoded, hashlib.sha256(encoded.encode()).hexdigest())

    def _step_metric(self, components, queries, command_jacobian=None):
        """A Fisher-like scaling metric, never a replacement objective gradient.

        Modal action sensitivities are sufficient for this preconditioner.
        All mean/covariance/normalizer derivatives instead enter the separately
        evaluated full-J command gradient. Damping modifies only this numerical
        step metric; it never changes any candidate covariance.
        """
        metric = self.damping**2*np.eye(len(self.velocity))
        command_jacobian = (np.eye(len(self.velocity)) if command_jacobian is None
                            else np.asarray(command_jacobian))
        for component, query in zip(components, queries):
            sensitivity = np.zeros((component.state.mean.size, len(self.velocity)))
            for edge_id, action in zip(self.body_ids, self.actions):
                index = component.state.edge_ids.index(edge_id)
                axis = np.asarray(action["axis"], dtype=float)
                sensitivity[6*index+3:6*index+6] = np.outer(
                    axis/np.linalg.norm(axis), command_jacobian[int(action["joint_index"])])
            jacobian = query.jacobian @ sensitivity
            precision = np.zeros((6, 6))
            precision[:3, :3] = covariance_support(query.covariance[:3, :3]).inverse
            precision[3:, 3:] = covariance_support(query.covariance[3:, 3:]).inverse
            metric += component.weight*(jacobian.T @ precision @ jacobian)
        return metric

    def _command_bounds(self, command, dt, *, position_lower=None,
                        position_upper=None, max_speed=None):
        """Shared feasible increments, including room to brake at each limit."""
        command = np.asarray(command, dtype=float)
        dt = float(dt)
        if (command.shape != self.velocity.shape or not np.all(np.isfinite(command))
                or not np.isfinite(dt) or dt <= 0.):
            raise ValueError('Command bounds require one finite command per joint and a positive interval.')
        lower_position = self.lower
        upper_position = self.upper
        if position_lower is not None:
            value = np.asarray(position_lower, dtype=float)
            if value.shape != command.shape or not np.all(np.isfinite(value)):
                raise ValueError('Local position bounds require one finite position per joint.')
            lower_position = np.maximum(lower_position, value)
        if position_upper is not None:
            value = np.asarray(position_upper, dtype=float)
            if value.shape != command.shape or not np.all(np.isfinite(value)):
                raise ValueError('Local position bounds require one finite position per joint.')
            upper_position = np.minimum(upper_position, value)
        if np.any(lower_position > upper_position):
            raise ValueError('Local position bounds have no overlap with the joint limits.')
        speed = self.speed
        if max_speed is not None:
            value = np.asarray(max_speed, dtype=float)
            if (value.shape not in ((), command.shape) or not np.all(np.isfinite(value))
                    or np.any(value < 0.)):
                raise ValueError('A local speed cap must be finite and nonnegative.')
            speed = np.minimum(speed, value)
        # For an outward next velocity w and remaining distance D, require
        # dt*w+w*w/(2*a) <= D. Maximum discrete braking then consumes no more
        # room. This rationalized positive root avoids cancellation at a limit.
        braking_change = self.acceleration*dt
        lower_distance = np.maximum(command-lower_position, 0.)
        upper_distance = np.maximum(upper_position-command, 0.)
        lower_stopping_speed = (2.*self.acceleration*lower_distance
            /(np.sqrt(braking_change**2+2.*self.acceleration*lower_distance)+braking_change))
        upper_stopping_speed = (2.*self.acceleration*upper_distance
            /(np.sqrt(braking_change**2+2.*self.acceleration*upper_distance)+braking_change))
        lower = np.maximum(lower_position-command,
            dt*np.maximum(np.maximum(-speed, self.velocity-braking_change), -lower_stopping_speed))
        upper = np.minimum(upper_position-command,
            dt*np.minimum(np.minimum(speed, self.velocity+braking_change), upper_stopping_speed))
        if np.any(lower > upper+1e-14):
            raise ValueError('The joint, speed and acceleration limits have no common short command.')
        return lower, upper, lower_stopping_speed, upper_stopping_speed

    def _selection_sources(self, result, body_components, components,
                           next_command, command_map, selected_branches):
        """Use the same full candidate provenance for reaching and searching."""
        if getattr(result.stages[-1], 'goal_components', ()):
            selected_body_ids = [(command_state(item.state, next_command,
                self.body_ids, self.actions) if command_map is None else command_map.push(
                    item.state, next_command, self.body_ids, self.actions)).state_id
                for item in body_components]
        else:
            selected_body_ids = [branch['state_id'] for branch in selected_branches]
        return {
            "command_propagation": ("declared rigid joint action" if command_map is None
                                    else "estimated-K anchored equilibrium joint action"),
            "stiffness_command_correction": (None if command_map is None else command_map.record()),
            "selected_command_action": (None if command_map is None
                                        else command_map.selection_record(next_command)),
            "global_optimum_claimed": False, "observation_stamp": float(result.prior.stamp),
            "source_stamp": float(result.prior.stamp),
            "source_version": int(components[0].state.version),
            "source_versions": [int(component.state.version) for component in components],
            "source_observation_ids": list(result.records["observation_ids"]),
            "source_command": list(result.prior.command), "next_command": next_command.tolist(),
            "source_state_ids": [component.state.state_id for component in components],
            "source_weights": [float(component.weight) for component in components],
            "source_log_weights": [_component_log_weight(component) for component in components],
            "body_source_state_ids": [component.state.state_id for component in body_components],
            "body_source_versions": [int(component.state.version) for component in body_components],
            "body_source_weights": [float(component.weight) for component in body_components],
            "body_source_log_weights": [_component_log_weight(component) for component in body_components],
            "selected_body_state_ids": selected_body_ids,
            "goal_source_state_ids": [component.state.state_id for component in components],
            "goal_source_versions": [int(component.state.version) for component in components],
            "goal_source_weights": [float(component.weight) for component in components],
            "goal_source_log_weights": [_component_log_weight(component) for component in components],
            "goal_parent_internal_state_id": (body_components[0].state.state_id
                if getattr(result.stages[-1], 'goal_components', ()) else None),
            "fixed_goal_memory_available": result.records.get('fixed_goal_memory', {}).get('available', False),
            "goal_memory_observation_stamp": result.records.get('fixed_goal_memory', {}).get('observation_stamp'),
            "root_goal_candidates": [dict(
                label=component.label, weight=float(component.weight),
                log_weight=_component_log_weight(component),
                source_state_id=component.state.state_id,
                source_stamp=component.state.stamp, source_version=component.state.version,
                source_command=list(component.state.command),
                points=[dict(axis=axis, mean=query.mean.tolist(),
                    covariance=query.covariance.tolist(),
                    within=query.within.tolist(), between=query.between.tolist(),
                    query_id=query.query_id)
                    for axis, query in switch_goal_queries(component.state,
                        self.marker_path, self.goal_offset, self.goal_orientation, self.goal_axis_length,
                        frame=self.root_frame)]) for component in components],
            "relative_path": [list(item) for item in self.relative_path],
            "goal_offset": self.goal_offset.tolist(),
            "goal_orientation_wxyz": self.goal_orientation.tolist(),
            "physical_truth_received": False,
            "orientation_reexpression": "modal quaternion times first-order mean correction; covariance unchanged to first order",
        }

    def step(self, result, dt, *, stiffness_proposal=None, command_map=None,
             initial_command=None):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("The next command interval must be positive and finite.")
        command = np.asarray(result.prior.command, dtype=float)
        if command.shape != self.velocity.shape:
            raise ValueError("Command dimension differs from the configured joint limits.")
        body_components, components = control_components(result)
        if command_map is not None:
            command_map.validate_result(result)
        if not components or not np.isclose(sum(item.weight for item in components), 1.0):
            raise ValueError("The fused candidate weights must already be normalized.")
        for component in components:
            state = component.state
            if state.stamp != result.prior.stamp or state.command != result.prior.command:
                raise ValueError("Control candidates must share the current command and observation time.")
        lower, upper, lower_stopping_speed, upper_stopping_speed = self._command_bounds(command, dt)
        cache = {}

        def evaluate(increment):
            increment = np.asarray(increment, dtype=float)
            if np.any(increment < lower-1e-10) or np.any(increment > upper+1e-10):
                raise ValueError("The optimizer evaluated an infeasible short command.")
            increment = np.clip(increment, lower, upper)
            trial_command = command + increment
            key = tuple(float(value) for value in trial_command)
            if key in cache:
                return cache[key]
            log_density, branches, queries = self._evaluate_command_objective(
                components, trial_command, command_map)
            cache[key] = (log_density, increment.copy(), branches, queries,
                          _relative_mixture_log_parts(branches))
            return cache[key]

        # Start at the feasible increment closest to stopping. Retaining the
        # previous velocity would keep moving when a flat objective or a
        # rejected line search supplies no corrective step. If acceleration
        # limits prevent stopping now, this baseline brakes as far as allowed.
        baseline = np.clip(np.zeros_like(self.velocity), lower, upper)
        initial_item = evaluate(baseline)
        braking_item = initial_item
        proposal_diagnostics = {'available': stiffness_proposal is not None,
                                'accepted_as_initial_point': False}
        if stiffness_proposal is not None:
            stiffness_record = result.records.get('stiffness')
            stiffness_density = None if stiffness_record is None else (
                stiffness_record['posterior'] if stiffness_record['posterior'] is not None
                else stiffness_record.get('retained_prior'))
            if (tuple(stiffness_proposal['source_command']) != tuple(command)
                    or float(stiffness_proposal['source_stamp']) != result.prior.stamp
                    or stiffness_proposal.get('source_prior_state_id') != result.prior.state_id
                    or stiffness_proposal.get('body_posterior_state_ids') != [item.state.state_id for item in body_components]
                    or stiffness_density is None
                    or stiffness_proposal.get('density_id') != stiffness_density['density_id']):
                raise ValueError('A stiffness command proposal belongs to a different body update.')
            if stiffness_record.get('command_proposal') is not stiffness_proposal:
                raise ValueError('Control requires the same recorded stiffness command proposal.')
            proposed_command = np.asarray(stiffness_proposal['command'], dtype=float)
            if proposed_command.shape != command.shape or not np.all(np.isfinite(proposed_command)):
                raise ValueError('A stiffness command proposal must be a finite joint command.')
            candidate = evaluate(np.clip(proposed_command-command, lower, upper))
            proposal_diagnostics.update(
                source=stiffness_proposal,
                projected_command=(command+candidate[1]).tolist(),
                log_body_criterion=candidate[0],
                log_braking_criterion=braking_item[0],
                role='initial command proposal only; accepted by the same complete body criterion',
                physical_truth_received=False)
            if log_evidence_parts_difference(candidate[4],initial_item[4]) > 0.:
                initial_item = candidate
                baseline = candidate[1].copy()
                proposal_diagnostics['accepted_as_initial_point'] = True
        approach_proposal = dict(available=initial_command is not None,
            accepted_as_initial_point=False,
            role='initial point only; acceptance and subsequent gradient use the same complete criterion')
        if initial_command is not None:
            proposed = np.asarray(initial_command, dtype=float)
            if proposed.shape != command.shape or not np.all(np.isfinite(proposed)):
                raise ValueError('An approach proposal requires one finite target per joint.')
            candidate = evaluate(np.clip(proposed-command, lower, upper))
            approach_proposal.update(requested_command=proposed.tolist(),
                projected_command=(command+candidate[1]).tolist(),
                proposed_objective=-candidate[0], comparison_objective=-initial_item[0])
            if log_evidence_parts_difference(candidate[4],initial_item[4]) > 0.:
                initial_item = candidate
                baseline = candidate[1].copy()
                approach_proposal['accepted_as_initial_point'] = True
        initial_cost = -initial_item[0]
        raw_gradient = np.zeros_like(baseline)
        difference_steps = np.zeros_like(baseline)
        for index in range(baseline.size):
            forward = min(self.gradient_step, upper[index]-baseline[index])
            backward = min(self.gradient_step, baseline[index]-lower[index])
            difference = forward if forward >= .5*self.gradient_step else -backward
            if abs(difference) <= 1e-15:
                continue
            probe = baseline.copy()
            probe[index] += difference
            # Differentiate J with respect to q only: every moment, covariance,
            # logdet, Bingham parameter/normalizer and weight enters each value.
            probed = evaluate(probe)
            raw_gradient[index] = -log_evidence_parts_difference(probed[4],initial_item[4])/difference
            difference_steps[index] = difference
        command_jacobian = (None if command_map is None
                            else command_map.jacobian(command+baseline))
        metric = self._step_metric(components, initial_item[3], command_jacobian)
        direction, quadratic_record = _bounded_gradient_direction(
            metric, raw_gradient, baseline, lower, upper)
        solver_failed = not quadratic_record['constraints_satisfied']
        if solver_failed:
            quadratic_record['fallback'] = 'maximal braking baseline'
        elif not quadratic_record['accepted']:
            quadratic_record['fallback'] = 'validated feasible initial candidate retained'
        metric_kind = "coupled box-constrained Fisher quadratic; full-J finite-difference gradient"
        slope = float(raw_gradient @ direction)
        selected = braking_item if solver_failed else initial_item
        line_search = []
        accepted_step = False
        step_scale = 0.0
        accepted_backtracks = 0
        if slope < 0.:
            for backtracks in range(self.max_backtracks+1):
                scale = 0.5**backtracks
                trial = evaluate(baseline+scale*direction)
                trial_cost = -trial[0]
                sufficient_decrease = initial_cost+1e-4*scale*slope
                objective_change=-log_evidence_parts_difference(trial[4],initial_item[4])
                change_bound=1e-4*scale*slope
                line_search.append({"scale": scale, "objective": trial_cost,
                    "armijo_bound": sufficient_decrease,
                    "objective_change":objective_change,"armijo_change_bound":change_bound,
                    "direction_kind": "coupled quadratic", "backtracks": backtracks})
                if objective_change <= change_bound:
                    selected, accepted_step, step_scale = trial, True, scale
                    accepted_backtracks = backtracks
                    break
        increment = selected[1]
        projected_gradient = raw_gradient.copy()
        at_lower = baseline <= lower+1e-10
        at_upper = baseline >= upper-1e-10
        projected_gradient[(at_lower & (raw_gradient > 0.0))
                           | (at_upper & (raw_gradient < 0.0))] = 0.0
        self.velocity = increment/dt
        next_command = command + increment
        diagnostics = {
            **self._selection_sources(result, body_components, components,
                                      next_command, command_map, selected[2]),
            "method": "one projected complete-objective gradient update with Armijo backtracking",
            "duration": dt, "increment": increment.tolist(), "velocity": self.velocity.tolist(),
            "log_expected_density": selected[0],
            "log_expected_density_parts":list(selected[4]),
            "objective_evaluation_count": len(cache),
            "gradient_evaluation_count": 1,
            "objective_used_for_command_selection": True,
            "gradient_method": "bound-aware forward/backward finite differences of the complete objective with respect to command",
            "gradient_step": self.gradient_step,
            "gradient_difference_steps": difference_steps.tolist(),
            "gradient_at_command": (command+baseline).tolist(),
            "gradient_at_state_ids": [query.state_id for query in initial_item[3]],
            "objective_gradient": raw_gradient.tolist(),
            "projected_gradient_at_baseline": projected_gradient.tolist(),
            "projected_gradient_norm_at_baseline": float(np.linalg.norm(projected_gradient, ord=np.inf)),
            "preconditioner": metric_kind,
            "preconditioner_damping": self.damping,
            "preconditioner_matrix": metric.tolist(),
            "quadratic_step": quadratic_record,
            "feasible_baseline_increment": baseline.tolist(),
            "feasible_baseline_objective": initial_cost,
            "braking_baseline_increment": braking_item[1].tolist(),
            "braking_baseline_objective": -braking_item[0],
            "stiffness_initial_proposal": proposal_diagnostics,
            "approach_initial_proposal": approach_proposal,
            "line_search_accepted": accepted_step,
            "step_scale": step_scale,
            "step_backtracks": (accepted_backtracks if accepted_step else
                                 max((item['backtracks'] for item in line_search), default=0)),
            "maximum_line_search_backtracks": self.max_backtracks,
            "selected_gradient_direction": direction.tolist(),
            "line_search": line_search,
            "improved_from_feasible_baseline": bool(
                log_evidence_parts_difference(selected[4],initial_item[4]) > 0.),
            "zero_increment_feasible": bool(np.all(lower <= 0.0) and np.all(upper >= 0.0)),
            "baseline_motion_required_by_acceleration_limits": bool(
                np.any(lower > 0.0) or np.any(upper < 0.0)),
            "selection_reason": ("full-objective Armijo gradient step" if accepted_step
                else ("validated complete-criterion initial proposal retained; no gradient trial accepted"
                    if selected is initial_item and initial_item is not braking_item
                    else "feasible braking baseline retained; no gradient trial accepted")),
            "convergence_claimed": False,
            "increment_bounds": np.column_stack((lower, upper)).tolist(),
            "joint_limit_stopping_speed_bounds": np.column_stack(
                (-lower_stopping_speed, upper_stopping_speed)).tolist(),
            "joint_limit_braking_condition": "dt*abs(outward next velocity) + velocity**2/(2*max_acceleration) <= distance to joint limit",
            "selected_branches": selected[2],
        }
        return next_command, diagnostics

    def local_step(self, result, dt, *, mode, target_command=None,
                   max_speed=None, position_lower=None, position_upper=None,
                   command_map=None, requested_velocity=None):
        """Execute a local search or visual velocity, retaining the same laws.

        The search policy supplies command-space waypoints, not physical poses
        or replacement candidate distributions. A target uses the same
        stopping-distance bound as the joint limits. The full current goal
        mixture is queried once at the selected command for diagnostics; that
        criterion is not used to select or accept a search move.
        """
        if mode not in {'braking', 'settling', 'searching', 'search_exhausted', 'visual_approach'}:
            raise ValueError('A local marker action requires an explicit braking or search mode.')
        if requested_velocity is not None and (mode != 'visual_approach' or target_command is not None):
            raise ValueError('A visual velocity must be separate from search waypoints.')
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.:
            raise ValueError('The next command interval must be positive and finite.')
        command = np.asarray(result.prior.command, dtype=float)
        if command.shape != self.velocity.shape:
            raise ValueError('Command dimension differs from the configured joint limits.')
        body_components, components = control_components(result)
        if command_map is not None:
            command_map.validate_result(result)
        if not components or not np.isclose(sum(item.weight for item in components), 1.):
            raise ValueError('The fused candidate weights must already be normalized.')
        for component in components:
            state = component.state
            if state.stamp != result.prior.stamp or state.command != result.prior.command:
                raise ValueError('Control candidates must share the current command and observation time.')
        lower, upper, lower_stopping_speed, upper_stopping_speed = self._command_bounds(
            command, dt, position_lower=position_lower,
            position_upper=position_upper, max_speed=max_speed)
        previous_velocity = self.velocity.copy()
        baseline = np.clip(np.zeros_like(self.velocity), lower, upper)
        target = None
        target_stopping_speed = None
        if requested_velocity is not None:
            desired_velocity = np.asarray(requested_velocity, dtype=float)
            if desired_velocity.shape != command.shape or not np.all(np.isfinite(desired_velocity)):
                raise ValueError('A visual velocity requires one finite value per joint.')
            increment = np.clip(dt*desired_velocity, lower, upper)
        elif target_command is None:
            increment = baseline.copy()
        else:
            target = np.asarray(target_command, dtype=float)
            if target.shape != command.shape or not np.all(np.isfinite(target)):
                raise ValueError('A local search target must contain one finite position per joint.')
            difference = target-command
            distance = np.abs(difference)
            braking_change = self.acceleration*dt
            target_stopping_speed = (2.*self.acceleration*distance
                /(np.sqrt(braking_change**2+2.*self.acceleration*distance)+braking_change))
            speed = (self.speed if max_speed is None else
                     np.minimum(self.speed, np.asarray(max_speed, dtype=float)))
            desired_velocity = np.sign(difference)*np.minimum(target_stopping_speed, speed)
            increment = np.clip(dt*desired_velocity, lower, upper)
        next_command = command+increment
        branches = []
        for component in components:
            trial_state = (command_state(component.state, next_command,
                                         self.body_ids, self.actions) if command_map is None
                           else command_map.push(component.state, next_command,
                                                 self.body_ids, self.actions))
            query = self._query(trial_state)
            log_density, branch = _relative_density(
                query, self.goal_offset, self.goal_orientation, self.integration_steps)
            log_weight = _component_log_weight(component)
            branch.update(label=component.label, weight=float(component.weight),
                          log_weight=log_weight, state_id=trial_state.state_id,
                          log_density_parts=list(branch.get('log_density_parts',(log_density,0.))),
                          parent_state_id=component.state.state_id,
                          parent_command=list(component.state.command),
                          parent_stamp=float(component.state.stamp))
            branches.append(branch)
        log_parts=_relative_mixture_log_parts(branches)
        log_density = float(log_parts[0]+log_parts[1])
        sources = self._selection_sources(result, body_components, components,
                                          next_command, command_map, branches)
        self.velocity = increment/dt
        selected_is_baseline = bool(np.array_equal(increment, baseline))
        diagnostics = {
            **sources,
            "method": 'deterministic local marker '+mode+' command; complete objective is diagnostic only',
            "action_mode": mode,
            "duration": dt, "increment": increment.tolist(), "velocity": self.velocity.tolist(),
            "previous_velocity": previous_velocity.tolist(),
            "target_command": None if target is None else target.tolist(),
            "requested_visual_velocity": (None if requested_velocity is None
                                          else np.asarray(requested_velocity).tolist()),
            "target_stopping_speed": (None if target_stopping_speed is None
                                      else target_stopping_speed.tolist()),
            "local_max_speed": None if max_speed is None else np.asarray(max_speed).tolist(),
            "local_position_lower": None if position_lower is None else np.asarray(position_lower).tolist(),
            "local_position_upper": None if position_upper is None else np.asarray(position_upper).tolist(),
            "log_expected_density": log_density,
            "log_expected_density_parts":list(log_parts),
            "objective_used_for_command_selection": False,
            "objective_evaluation_count": 1,
            "gradient_evaluation_count": 0,
            "gradient_method": 'not evaluated during a deterministic local marker action',
            "gradient_step": self.gradient_step,
            "gradient_difference_steps": None,
            "gradient_at_command": None,
            "gradient_at_state_ids": [],
            "objective_gradient": None,
            "projected_gradient_at_baseline": None,
            "projected_gradient_norm_at_baseline": None,
            "preconditioner": None,
            "preconditioner_damping": None,
            "preconditioner_matrix": None,
            "feasible_baseline_increment": baseline.tolist(),
            "feasible_baseline_objective": -log_density if selected_is_baseline else None,
            "braking_baseline_increment": baseline.tolist(),
            "braking_baseline_objective": -log_density if selected_is_baseline else None,
            "stiffness_initial_proposal": {
                'available': result.records.get('stiffness', {}).get('command_proposal') is not None,
                'accepted_as_initial_point': False,
                'role': 'not used for local waypoints; the recorded estimated-K command action still propagates the same body law'},
            "line_search_accepted": False,
            "step_scale": None,
            "step_backtracks": 0,
            "line_search": [],
            "improved_from_feasible_baseline": None,
            "zero_increment_feasible": bool(np.all(lower <= 0.) and np.all(upper >= 0.)),
            "baseline_motion_required_by_acceleration_limits": bool(np.any(lower > 0.) or np.any(upper < 0.)),
            "selection_reason": ('same-law visual servo velocity through shared command limits' if requested_velocity is not None else
                                 'maximum feasible braking toward zero velocity' if target is None
                                 else 'bounded local search waypoint with stopping-distance speed'),
            "convergence_claimed": False,
            "increment_bounds": np.column_stack((lower, upper)).tolist(),
            "joint_limit_stopping_speed_bounds": np.column_stack(
                (-lower_stopping_speed, upper_stopping_speed)).tolist(),
            "joint_limit_braking_condition": 'dt*abs(outward next velocity) + velocity**2/(2*max_acceleration) <= distance to active position limit',
            "selected_branches": branches,
        }
        return next_command, diagnostics
