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


def command_state(state, command, body_ids, actions):
    """Push the same posterior through its declared deterministic right action.

    R exp(zeta) D = R D exp(D.T zeta); translations use parent coordinates.
    Thus all covariance blocks are transported by diag(I, D.T), including
    cross blocks.  Reconstructing independent marginals here would be wrong.
    """
    command = np.asarray(command, dtype=float)
    current = np.asarray(state.command, dtype=float)
    if command.shape != current.shape or not np.all(np.isfinite(command)):
        raise ValueError("A trial command must match the finite source command.")
    if np.array_equal(command, current):
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
        delta = axis_angle_to_quat(action["axis"], command[joint]-current[joint])
        right = quat_right_matrix(delta)
        rotation = quat_to_rotmat(delta)
        chart[6*index+3:6*index+6, 6*index+3:6*index+6] = rotation.T
        transforms.append(BinghamTransform(
            edge.edge_id, edge.translation, right @ edge.bingham @ right.T,
            edge.translation_covariance))
    covariance = chart @ state.covariance @ chart.T
    return JointTransformState(tuple(transforms), covariance, version=state.version,
                               stamp=state.stamp, command=tuple(command),
                               component_id=state.component_id)


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

    def _step_metric(self, components, queries):
        """A Fisher-like scaling metric, never a replacement objective gradient.

        Modal action sensitivities are sufficient for this preconditioner.
        All mean/covariance/normalizer derivatives instead enter the separately
        evaluated full-J command gradient. Damping modifies only this numerical
        step metric; it never changes any candidate covariance.
        """
        metric = self.damping**2*np.eye(len(self.velocity))
        for component, query in zip(components, queries):
            sensitivity = np.zeros((component.state.mean.size, len(self.velocity)))
            for edge_id, action in zip(self.body_ids, self.actions):
                index = component.state.edge_ids.index(edge_id)
                axis = np.asarray(action["axis"], dtype=float)
                sensitivity[6*index+3:6*index+6, int(action["joint_index"])] = axis/np.linalg.norm(axis)
            jacobian = query.jacobian @ sensitivity
            precision = np.zeros((6, 6))
            precision[:3, :3] = covariance_support(query.covariance[:3, :3]).inverse
            precision[3:, 3:] = covariance_support(query.covariance[3:, 3:]).inverse
            metric += component.weight*(jacobian.T @ precision @ jacobian)
        return metric

    def step(self, result, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("The next command interval must be positive and finite.")
        command = np.asarray(result.prior.command, dtype=float)
        if command.shape != self.velocity.shape:
            raise ValueError("Command dimension differs from the configured joint limits.")
        components = tuple(result.stages[-1].components)
        if not components or not np.isclose(sum(item.weight for item in components), 1.0):
            raise ValueError("The fused candidate weights must already be normalized.")
        for component in components:
            state = component.state
            if state.stamp != result.prior.stamp or state.command != result.prior.command:
                raise ValueError("Control candidates must share the current command and observation time.")
        lower = np.maximum(self.lower-command,
                           dt*np.maximum(-self.speed, self.velocity-self.acceleration*dt))
        upper = np.minimum(self.upper-command,
                           dt*np.minimum(self.speed, self.velocity+self.acceleration*dt))
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
                trial_state = command_state(component.state, trial_command,
                                            self.body_ids, self.actions)
                query = self._query(trial_state)
                queries.append(query)
                log_density, branch = _relative_density(
                    query, self.goal_offset, self.goal_orientation, self.integration_steps)
                branch.update(label=component.label, weight=float(component.weight),
                              parent_state_id=component.state.state_id,
                              parent_command=list(component.state.command),
                              parent_stamp=float(component.state.stamp))
                branches.append(branch)
                terms.append(np.log(component.weight)+log_density
                             if component.weight > 0.0 else -np.inf)
            cache[key] = (float(logsumexp(terms)), increment.copy(), branches, tuple(queries))
            return cache[key]

        baseline = np.clip(self.velocity*dt, lower, upper)
        initial_item = evaluate(baseline)
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
        metric = self._step_metric(components, initial_item[3])
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
        diagnostics = {
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
            "line_search_accepted": accepted_step,
            "step_scale": step_scale,
            "step_backtracks": max(0, len(line_search)-1),
            "line_search": line_search,
            "improved_from_feasible_baseline": bool(selected[0] > initial_item[0]),
            "zero_increment_feasible": bool(np.all(lower <= 0.0) and np.all(upper >= 0.0)),
            "baseline_motion_required_by_acceleration_limits": bool(
                np.any(lower > 0.0) or np.any(upper < 0.0)),
            "selection_reason": ("full-objective Armijo gradient step" if accepted_step
                else "feasible coasting baseline retained; no gradient trial accepted"),
            "convergence_claimed": False,
            "increment_bounds": np.column_stack((lower, upper)).tolist(),
            "selected_branches": selected[2],
            "physical_truth_received": False,
            "orientation_reexpression": "modal quaternion times first-order mean correction; covariance unchanged to first order",
        }
        return next_command, diagnostics
