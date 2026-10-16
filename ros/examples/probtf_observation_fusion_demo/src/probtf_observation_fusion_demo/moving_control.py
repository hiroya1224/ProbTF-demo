"""Short commands selected from the fused transform candidates.

The only extrapolation is the input family's declared right joint action.
Every trial retains the complete joint covariance, including body/marker
blocks.  The marker variable is unchanged while a future body command is
queried. Each cycle makes one projected update from the complete relative
Gaussian/Bingham objective gradient, with a short backtracking line search.
One such update does not imply an IK optimum or convergence.
"""

from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from scipy.special import logsumexp

from probtf.bingham import bingham_log_normalizer
from probtf.geometry import (
    axis_angle_to_quat, quat_mul, quat_right_matrix, quat_to_rotmat,
    rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    pose_query,
)
from probtf.spherical_law.support import covariance_support, require_in_support


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
    log_position = -0.5 * (
        support.rank*np.log(2.0*np.pi) + support.log_pdet
        + difference @ support.inverse @ difference)
    mean_orientation = quat_mul(query.rotation_wxyz,
        rotation_vector_to_quaternion(query.correction[3:]))
    parameter = bingham_from_rotation_coordinates(
        mean_orientation, query.covariance[3:, 3:])
    eigenvalues = np.linalg.eigvalsh(parameter)
    shift = float(eigenvalues[-1])
    log_z = _log_normalizer(tuple(eigenvalues-shift), int(integration_steps))+shift
    log_orientation = float(goal_orientation @ parameter @ goal_orientation - log_z)
    return float(log_position + log_orientation), {
        "log_position": float(log_position), "log_position_pdet": support.log_pdet,
        "log_orientation": log_orientation, "log_bingham_normalizer": float(log_z),
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


class MovingController:
    """One sensor-independent selection rule; no physical state enters it."""

    def __init__(self, inference, robot_description, goal_offset,
                 goal_orientation_wxyz, motion, integration_steps=480):
        self.body_ids = tuple(inference.body_ids)
        self.actions = tuple(dict(action) for action in inference.actions)
        self.hand_path = tuple(inference.paths["hand"])
        marker_path = tuple(inference.paths["marker"])
        self.relative_path = tuple((name, not inverse)
                                   for name, inverse in reversed(marker_path)) + self.hand_path
        self.root_frame = str(inference.settings["root_frame"])
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
        self.max_backtracks = int(motion.get("gradient_max_backtracks", 3))
        self.damping = float(motion.get("damping", 0.025))
        self.integration_steps = int(integration_steps)
        if (min(self.speed, self.acceleration, self.gradient_step, self.damping) <= 0.0
                or self.integration_steps <= 0 or self.max_backtracks < 0):
            raise ValueError("Positive command limits and numerical solver settings are required.")
        self.velocity = np.zeros(len(joints))

    def _query(self, state):
        return pose_query(state, self.relative_path,
                          query_id="control:marker_relative_hand", frame="marker")

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

    def step(self, result, dt, *, stiffness_proposal=None, command_map=None):
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
        # Leave room to brake after this short position command.  For an
        # outward next velocity w and remaining joint distance D, require
        # dt*w+w*w/(2*a) <= D.  The next state then retains its continuous
        # stopping distance; maximum discrete braking consumes no more room.
        # The rationalized positive root avoids cancellation close to a limit.
        braking_change = self.acceleration*dt
        lower_distance = np.maximum(command-self.lower, 0.)
        upper_distance = np.maximum(self.upper-command, 0.)
        lower_stopping_speed = (2.*self.acceleration*lower_distance
            /(np.sqrt(braking_change**2+2.*self.acceleration*lower_distance)+braking_change))
        upper_stopping_speed = (2.*self.acceleration*upper_distance
            /(np.sqrt(braking_change**2+2.*self.acceleration*upper_distance)+braking_change))
        lower = np.maximum(self.lower-command,
            dt*np.maximum(np.maximum(-self.speed, self.velocity-braking_change), -lower_stopping_speed))
        upper = np.minimum(self.upper-command,
            dt*np.minimum(np.minimum(self.speed, self.velocity+braking_change), upper_stopping_speed))
        if np.any(lower > upper+1e-14):
            raise ValueError("The joint, speed and acceleration limits have no common short command.")
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
            terms, branches, queries = [], [], []
            for component in components:
                trial_state = (command_state(component.state, trial_command,
                                             self.body_ids, self.actions) if command_map is None
                               else command_map.push(component.state, trial_command,
                                                     self.body_ids, self.actions))
                query = self._query(trial_state)
                queries.append(query)
                log_density, branch = _relative_density(
                    query, self.goal_offset, self.goal_orientation, self.integration_steps)
                log_weight = getattr(component, 'log_weight', None)
                if log_weight is None:
                    log_weight = np.log(component.weight) if component.weight > 0.0 else -np.inf
                branch.update(label=component.label, weight=float(component.weight),
                              log_weight=float(log_weight),
                              state_id=trial_state.state_id,
                              parent_state_id=component.state.state_id,
                              parent_command=list(component.state.command),
                              parent_stamp=float(component.state.stamp))
                branches.append(branch)
                terms.append(log_weight+log_density)
            cache[key] = (float(logsumexp(terms)), increment.copy(), branches, tuple(queries))
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
            if candidate[0] > initial_item[0]:
                initial_item = candidate
                baseline = candidate[1].copy()
                proposal_diagnostics['accepted_as_initial_point'] = True
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
            raw_gradient[index] = (-evaluate(probe)[0]-initial_cost)/difference
            difference_steps[index] = difference
        command_jacobian = (None if command_map is None
                            else command_map.jacobian(command+baseline))
        metric = self._step_metric(components, initial_item[3], command_jacobian)
        unbounded_direction = -np.linalg.solve(metric, raw_gradient)
        direction = np.clip(baseline+unbounded_direction, lower, upper)-baseline
        metric_kind = "full Fisher-like modal sensitivity preconditioner; full-J finite-difference gradient"
        if raw_gradient @ direction >= 0.0 and np.linalg.norm(direction) > 0.0:
            # Box projection can turn a cross-coupled preconditioned direction
            # uphill. Diagonal scaling then guarantees a projected descent
            # direction from the same feasible baseline.
            direction = np.clip(baseline-raw_gradient/np.diag(metric), lower, upper)-baseline
            metric_kind = "diagonal Fisher-like preconditioner after non-descent box projection"
        slope = float(raw_gradient @ direction)
        selected = initial_item
        line_search = []
        accepted_step = False
        step_scale = 0.0
        if slope < 0.0:
            for backtracks in range(self.max_backtracks+1):
                scale = 0.5**backtracks
                trial = evaluate(baseline+scale*direction)
                trial_cost = -trial[0]
                sufficient_decrease = initial_cost+1e-4*scale*slope
                line_search.append({"scale": scale, "objective": trial_cost,
                                    "armijo_bound": sufficient_decrease})
                if trial_cost <= sufficient_decrease:
                    selected, accepted_step, step_scale = trial, True, scale
                    break
        increment = selected[1]
        projected_gradient = raw_gradient.copy()
        at_lower = baseline <= lower+1e-10
        at_upper = baseline >= upper-1e-10
        projected_gradient[(at_lower & (raw_gradient > 0.0))
                           | (at_upper & (raw_gradient < 0.0))] = 0.0
        self.velocity = increment/dt
        next_command = command + increment
        if getattr(result.stages[-1], 'goal_components', ()):
            selected_body_ids = [(command_state(item.state, next_command,
                self.body_ids, self.actions) if command_map is None else command_map.push(
                    item.state, next_command, self.body_ids, self.actions)).state_id
                for item in body_components]
        else:
            selected_body_ids = [branch['state_id'] for branch in selected[2]]
        diagnostics = {
            "command_propagation": ("declared rigid joint action" if command_map is None
                                    else "estimated-K anchored equilibrium joint action"),
            "stiffness_command_correction": (None if command_map is None else command_map.record()),
            "selected_command_action": (None if command_map is None
                                        else command_map.selection_record(next_command)),
            "method": "one projected complete-objective gradient update with Armijo backtracking",
            "global_optimum_claimed": False, "observation_stamp": float(result.prior.stamp),
            "source_stamp": float(result.prior.stamp),
            "source_version": int(components[0].state.version),
            "source_versions": [int(component.state.version) for component in components],
            "source_observation_ids": list(result.records["observation_ids"]),
            "source_command": command.tolist(), "next_command": next_command.tolist(),
            "duration": dt, "increment": increment.tolist(), "velocity": self.velocity.tolist(),
            "source_state_ids": [component.state.state_id for component in components],
            "source_weights": [float(component.weight) for component in components],
            "source_log_weights": [float(getattr(component, 'log_weight', None)
                if getattr(component, 'log_weight', None) is not None else (
                    np.log(component.weight) if component.weight > 0.0 else -np.inf))
                for component in components],
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
            "relative_path": [list(item) for item in self.relative_path],
            "goal_offset": self.goal_offset.tolist(),
            "goal_orientation_wxyz": self.goal_orientation.tolist(),
            "log_expected_density": selected[0],
            "objective_evaluation_count": len(cache),
            "gradient_evaluation_count": 1,
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
            "feasible_baseline_increment": baseline.tolist(),
            "feasible_baseline_objective": initial_cost,
            "braking_baseline_increment": braking_item[1].tolist(),
            "braking_baseline_objective": -braking_item[0],
            "stiffness_initial_proposal": proposal_diagnostics,
            "line_search_accepted": accepted_step,
            "step_scale": step_scale,
            "step_backtracks": max(0, len(line_search)-1),
            "line_search": line_search,
            "improved_from_feasible_baseline": bool(selected[0] > initial_item[0]),
            "zero_increment_feasible": bool(np.all(lower <= 0.0) and np.all(upper >= 0.0)),
            "baseline_motion_required_by_acceleration_limits": bool(
                np.any(lower > 0.0) or np.any(upper < 0.0)),
            "selection_reason": ("full-objective Armijo gradient step" if accepted_step
                else "feasible braking baseline retained; no gradient trial accepted"),
            "convergence_claimed": False,
            "increment_bounds": np.column_stack((lower, upper)).tolist(),
            "joint_limit_stopping_speed_bounds": np.column_stack(
                (-lower_stopping_speed, upper_stopping_speed)).tolist(),
            "joint_limit_braking_condition": "dt*abs(outward next velocity) + velocity**2/(2*max_acceleration) <= distance to joint limit",
            "selected_branches": selected[2],
            "physical_truth_received": False,
            "orientation_reexpression": "modal quaternion times first-order mean correction; covariance unchanged to first order",
        }
        return next_command, diagnostics
