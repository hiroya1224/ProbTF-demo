"""One finite command step for the shared body/environment reaching score."""

import xml.etree.ElementTree as ET

import numpy as np

from .body_native import point_query
from .moving_control import command_state


def joint_reach_score(state, hand_path, marker_path, goal_offset, root_frame='base_link'):
    """E[||p_hand-p_goal||²] in one root frame and one correlated law.

    The two core queries retain their O(P) mean corrections.  Their cross
    covariance comes from the same original full P, as in the existing joint
    sensor-query assembly.  No independent goal law or relative-frame closure
    is substituted for this root-frame squared-distance expectation.
    """
    hand = point_query(state, hand_path, query_id='simple:hand', frame=root_frame)
    goal = point_query(state, marker_path, point=goal_offset,
        query_id='simple:goal', frame=root_frame)
    cross = hand.jacobian @ state.covariance @ goal.jacobian.T
    difference = hand.mean-goal.mean
    covariance = hand.covariance+goal.covariance-cross-cross.T
    score = float(difference @ difference+np.trace(covariance))
    if not np.isfinite(score) or not np.all(np.isfinite(covariance)):
        raise ValueError('The shared-law reaching expectation is nonfinite.')
    return score, dict(score=score, mean_difference=difference.tolist(),
        difference_covariance=covariance.tolist(), hand_mean=hand.mean.tolist(),
        goal_mean=goal.mean.tolist(), hand_goal_cross_covariance=cross.tolist(),
        hand_state_id=hand.state_id, goal_state_id=goal.state_id,
        joint_source_state_id=state.state_id, full_body_environment_cross_retained=True,
        root_frame=root_frame, monte_carlo_used=False)


class SimpleSelector:
    """Select one bounded Euclidean descent step of the sole reaching score.

    The frozen command map supplies only an equilibrium *increment* acting on
    the current estimated joint law.  This selector has no velocity state,
    curvature model, view factor, independent goal, or density-overlap cost.
    """

    def __init__(self, inference, robot_description, settings):
        self.inference = inference
        self.settings = dict(settings)
        control = inference.settings.get('control', {})
        self.goal_offset = np.asarray(self.settings.get('goal_offset',
            control.get('goal_offset', (0., 0., 0.))), dtype=float)
        self.max_step = float(self.settings.get('max_step_rad', .02))
        self.difference_step = float(self.settings.get('difference_step_rad', 1e-4))
        self.duration = float(self.settings.get('duration_s', .5))
        self.backtracks = int(self.settings.get('backtracking_steps', 8))
        if (self.goal_offset.shape != (3,) or not np.all(np.isfinite(self.goal_offset))
                or not np.all(np.isfinite((self.max_step, self.difference_step, self.duration)))
                or min(self.max_step, self.difference_step, self.duration) <= 0.
                or self.backtracks < 0):
            raise ValueError('Simple reaching needs finite geometry, positive steps/duration and a nonnegative backtracking budget.')
        joints = {joint.find('child').get('link'): joint
            for joint in ET.fromstring(robot_description).findall('joint')}
        size = len(inference.reference_command)
        self.lower, self.upper = np.empty(size), np.empty(size)
        covered = set()
        for name, action in zip(inference.body_ids, inference.actions):
            index = int(action['joint_index'])
            limits = joints[name].find('limit')
            self.lower[index] = float(limits.get('lower'))
            self.upper[index] = float(limits.get('upper'))
            covered.add(index)
        if (covered != set(range(size)) or not np.all(np.isfinite(self.lower))
                or not np.all(np.isfinite(self.upper)) or np.any(self.lower > self.upper)):
            raise ValueError('Every simple-selector command coordinate needs its declared finite URDF bounds.')

    def score(self, state):
        return joint_reach_score(state, self.inference.paths['hand'],
            self.inference.paths['marker'], self.goal_offset,
            self.inference.settings.get('root_frame', 'base_link'))

    def select(self, result, command_map=None, goal_observed=None):
        stage = result.stages[-1]
        if len(stage.components) != 1 or stage.goal_components:
            raise ValueError('Simple reaching requires one shared canonical body/environment law and no independent goal components.')
        component = stage.components[0]
        state = component.state
        command = np.asarray(result.prior.command, dtype=float)
        if (state.command != result.prior.command or state.stamp != result.prior.stamp
                or command.shape != self.lower.shape or not np.all(np.isfinite(command))
                or np.any(command < self.lower) or np.any(command > self.upper)):
            raise ValueError('The simple command source must retain its actual finite acquisition command/time and joint bounds.')
        if goal_observed is None:
            goal_observed = result.records.get('simple_inference', {}).get(
                'goal_observed', getattr(self.inference, 'goal_observed', False))
        if command_map is not None:
            command_map.validate_result(result)
        capture = None if command_map is None else command_map.record()
        diagnostics = dict(method='single_finite_step_shared_joint_expected_position_error',
            objective='E_mu_c[||p_hand(B)-p_goal(E)||^2]',
            source_stamp=state.stamp, source_command=command.tolist(),
            source_prior_state_id=result.prior.state_id,
            body_source_state_ids=[state.state_id], goal_source_state_ids=[state.state_id],
            source_version=state.version, goal_observed=bool(goal_observed),
            duration_s=self.duration, max_step_rad=self.max_step,
            difference_step_rad=self.difference_step, command_map=capture,
            command_action=('estimated_K_equilibrium_increment_on_current_joint_law'
                if command_map is not None else 'declared_right_joint_action_no_selected_K'),
            body_reconstructed_from_stiffness=False, independent_goal_used=False,
            independent_velocity_state=False, physical_truth_received=False, monte_carlo_used=False)
        if not goal_observed:
            diagnostics.update(status='holding_before_first_valid_marker',
                q_next=command.tolist(), gradient=None, S_before=None, S_after=None,
                step_norm_rad=0., backtracking_trials=[])
            return command.copy(), diagnostics

        cache = {}
        def evaluate(trial):
            key = tuple(float(value) for value in trial)
            if key not in cache:
                candidate = (command_state(state, trial, self.inference.body_ids,
                    self.inference.actions) if command_map is None else command_map.push(
                        state, trial, self.inference.body_ids, self.inference.actions))
                cache[key] = self.score(candidate)
            return cache[key]

        before, before_query = evaluate(command)
        gradient, spans = np.zeros(command.size), np.zeros(command.size)
        for index in range(command.size):
            plus, minus = command.copy(), command.copy()
            plus[index] = min(self.upper[index], command[index]+self.difference_step)
            minus[index] = max(self.lower[index], command[index]-self.difference_step)
            spans[index] = plus[index]-minus[index]
            if spans[index] != 0.:
                gradient[index] = (evaluate(plus)[0]-evaluate(minus)[0])/spans[index]
        if not np.all(np.isfinite(gradient)):
            raise ValueError('The simple reaching score derivative is nonfinite.')
        norm = float(np.linalg.norm(gradient))
        selected, after, after_query = command.copy(), before, before_query
        trials = []
        status = 'holding_zero_gradient' if norm == 0. else 'holding_no_decreasing_finite_step'
        if norm != 0.:
            direction = -self.max_step*gradient/norm
            for count in range(self.backtracks+1):
                trial = np.clip(command+(.5**count)*direction, self.lower, self.upper)
                value, query = evaluate(trial)
                accepted = bool(value < before)
                trials.append(dict(q=trial.tolist(), score=value, accepted=accepted))
                if accepted:
                    selected, after, after_query = trial, value, query
                    status = 'selected_decreasing_finite_step'
                    break
        diagnostics.update(status=status, q_next=selected.tolist(), gradient=gradient.tolist(),
            gradient_difference_spans_rad=spans.tolist(), S_before=before, S_after=after,
            source_score=before_query, selected_score=after_query,
            step_norm_rad=float(np.linalg.norm(selected-command)), backtracking_trials=trials,
            score_evaluations=len(cache), joint_lower=self.lower.tolist(), joint_upper=self.upper.tolist())
        if command_map is not None:
            diagnostics['selected_command_action'] = command_map.selection_record(selected)
        return selected, diagnostics
