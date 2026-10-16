"""Recorded command actions and prediction-only views of inferred joint laws.

The action ledger is a record of inputs that were actually executed. It is
neither a new equilibrium-generated body nor a new sensor update. Distinct K
versions may give a nonzero net body action even on a closed command path.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from probtf.geometry import axis_angle_to_quat, quat_right_matrix, quat_to_rotmat
from probtf.spherical_law.first_order import BinghamTransform, JointTransformState

from .model import Stage, WeightedState, component_log_weight
from .moving_control import control_components


def _vector(value, name, dimension=None):
    array = np.asarray(value, dtype=float)
    if (array.ndim != 1 or not array.size or not np.all(np.isfinite(array))
            or (dimension is not None and array.size != dimension)):
        raise ValueError(name+' must contain one finite value per joint.')
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class _FrozenMapping(Mapping):
    """An immutable provenance mapping that can cross process queues."""

    entries: tuple

    def __getitem__(self, key):
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self.entries)

    def __len__(self):
        return len(self.entries)


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Recorded provenance keys must be strings.')
        return _FrozenMapping(tuple((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _freeze(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError('Recorded provenance must have immutable finite JSON values.')


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ActionStep:
    """One completed input interval, with the K version actually used for it."""

    index: int
    start_stamp: float
    end_stamp: float
    start_command: tuple
    end_command: tuple
    joint_increment: tuple
    density_id: object = None
    estimate_version: object = None
    selected_K: object = None
    anchor_command: object = None
    anchor_equilibrium: object = None
    source_state_ids: tuple = ()
    source_stamp: object = None
    source_version: object = None
    provenance: object = None

    def __post_init__(self):
        if (isinstance(self.index, bool) or not isinstance(self.index, (int, np.integer))
                or self.index < 0):
            raise ValueError('A recorded action requires a nonnegative integer interval index.')
        object.__setattr__(self, 'index', int(self.index))
        for name in ('start_stamp', 'end_stamp'):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError('A recorded action requires finite interval timestamps.')
            object.__setattr__(self, name, value)
        if self.end_stamp <= self.start_stamp:
            raise ValueError('A recorded action interval must have positive duration.')
        command = _vector(self.start_command, 'Start command')
        object.__setattr__(self, 'start_command', command)
        for name in ('end_command', 'joint_increment'):
            object.__setattr__(self, name, _vector(getattr(self, name), name, len(command)))
        if self.start_command == self.end_command and any(self.joint_increment):
            raise ValueError('An unchanged command interval cannot have a nonzero equilibrium increment.')
        for name in ('selected_K', 'anchor_command', 'anchor_equilibrium'):
            value = getattr(self, name)
            if value is not None:
                value = _vector(value, name, len(command))
                if name == 'selected_K' and any(item <= 0. for item in value):
                    raise ValueError('Recorded selected K must be positive.')
                object.__setattr__(self, name, value)
        if self.density_id is not None and (not isinstance(self.density_id, str) or not self.density_id):
            raise ValueError('A recorded K density ID must be a nonempty string.')
        if self.selected_K is not None and self.density_id is None:
            raise ValueError('A selected K requires its recorded parameter density ID.')
        for name in ('estimate_version', 'source_version'):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError('Recorded estimate versions must be nonnegative integers.')
                object.__setattr__(self, name, int(value))
        if self.source_stamp is not None:
            source_stamp = float(self.source_stamp)
            if not np.isfinite(source_stamp) or source_stamp > self.start_stamp:
                raise ValueError('An interval cannot use a parameter estimate from its future.')
            object.__setattr__(self, 'source_stamp', source_stamp)
        ids = tuple(self.source_state_ids)
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError('Recorded source IDs must be nonempty strings.')
        object.__setattr__(self, 'source_state_ids', ids)
        object.__setattr__(self, 'provenance', _freeze({} if self.provenance is None else self.provenance))

    def record(self):
        return {name: _thaw(getattr(self, name)) for name in self.__dataclass_fields__}


def _action_signature(body_ids, actions, dimension):
    names = tuple(body_ids)
    if not names or len(names) != len(set(names)) or len(actions) != len(names):
        raise ValueError('Recorded command actions require distinct body edges and matching axes.')
    signature = []
    for name, action in zip(names, actions):
        index = action['joint_index']
        if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < dimension:
            raise ValueError('A recorded command axis has an invalid joint index.')
        axis = _vector(action['axis'], 'Joint axis', 3)
        if np.linalg.norm(axis) == 0.:
            raise ValueError('A recorded command axis must be nonzero.')
        signature.append((name, int(index), axis))
    return tuple(signature)


def _recorded_action(state, command, increment, signature):
    """The declared right action, including a history whose endpoint returns.

    This is the same Q A Q.T and G P G.T action as ``command_state``. Its
    different endpoint rule is explicit: a closed *history* may accumulate a
    nonzero action across K changes; a frozen one-command family may not.
    """
    if tuple(command) == state.command and not np.any(increment):
        return state
    actions = {name: (joint, axis) for name, joint, axis in signature}
    chart = np.eye(state.mean.size)
    transforms = []
    for index, edge in enumerate(state.transforms):
        action = actions.get(edge.edge_id)
        if action is None or increment[action[0]] == 0.:
            transforms.append(edge)
            continue
        if edge.bingham is None:
            raise ValueError('A recorded uncertain joint action requires the existing Bingham law.')
        quaternion = axis_angle_to_quat(action[1], increment[action[0]])
        right = quat_right_matrix(quaternion)
        chart[6*index+3:6*index+6, 6*index+3:6*index+6] = quat_to_rotmat(quaternion).T
        parameter = right @ edge.bingham @ right.T
        transforms.append(BinghamTransform(edge.edge_id, edge.translation,
            .5*(parameter+parameter.T), edge.translation_covariance))
    return JointTransformState(tuple(transforms), chart @ state.covariance @ chart.T,
        version=state.version, stamp=state.stamp, command=tuple(command),
        component_id=state.component_id)


class RecordedCommandPath:
    """Apply only the actual, contiguous input history to one frozen source.

    This map has no future-trial family or derivative. Future IK must bind
    its latest accepted K to the current predicted body using a separate map.
    """

    def __init__(self, result, steps, *, body_ids, actions):
        body, goals = control_components(result)
        self._body, self._goals = tuple(body), tuple(goals)
        self._source_prior_id = result.prior.state_id
        self.source_stamp = float(result.prior.stamp)
        self.source_command = tuple(result.prior.command)
        dimension = len(self.source_command)
        self._signature = _action_signature(body_ids, actions, dimension)
        if any(name not in result.prior.edge_ids for name in body_ids):
            raise ValueError('Recorded body actions refer to an absent source edge.')
        all_sources = self._body+self._goals
        if (not self._body or any(item.state.stamp != self.source_stamp
                or item.state.command != self.source_command for item in all_sources)):
            raise ValueError('Recorded path sources must have one current command and timestamp.')
        self.source_state_ids = frozenset(item.state.state_id for item in all_sources)
        self._source_versions = {item.state.state_id: item.state.version for item in all_sources}
        self.steps = tuple(steps)
        stamp, command, previous_index = self.source_stamp, self.source_command, None
        total = np.zeros(dimension)
        for step in self.steps:
            if not isinstance(step, ActionStep):
                raise ValueError('Command histories require immutable ActionStep intervals.')
            if (step.start_stamp != stamp or step.start_command != command
                    or (previous_index is not None and step.index != previous_index+1)
                    or len(step.joint_increment) != dimension):
                raise ValueError('Recorded command history has a gap, overlap, duplicate or mismatched command.')
            stamp, command, previous_index = step.end_stamp, step.end_command, step.index
            total += np.asarray(step.joint_increment)
        if not np.all(np.isfinite(total)):
            raise ValueError('Recorded command increments do not have a finite cumulative action.')
        self.target_stamp, self.target_command = stamp, command
        self.joint_increment = tuple(float(value) for value in total)

    def validate_result(self, result):
        body, goals = control_components(result)
        def sources(components):
            return tuple((item.label, item.weight, component_log_weight(item),
                          item.state.state_id, item.state.version) for item in components)
        if (result.prior.state_id != self._source_prior_id
                or result.prior.stamp != self.source_stamp
                or result.prior.command != self.source_command
                or sources(body) != sources(self._body) or sources(goals) != sources(self._goals)):
            raise ValueError('A recorded command path belongs to a different frozen source result.')

    def push(self, state, command, body_ids, actions):
        if (state.state_id not in self.source_state_ids
                or state.command != self.source_command or state.stamp != self.source_stamp
                or state.version != self._source_versions.get(state.state_id)):
            raise ValueError('A recorded command action requires its exact frozen source law.')
        if (tuple(command) != self.target_command or
                _action_signature(body_ids, actions, len(self.source_command)) != self._signature):
            raise ValueError('A recorded path cannot be used for another endpoint or joint action.')
        return _recorded_action(state, self.target_command, np.asarray(self.joint_increment), self._signature)

    def record(self):
        return dict(kind='recorded_executed_command_history',
            source_stamp=self.source_stamp, source_command=list(self.source_command),
            source_prior_state_id=self._source_prior_id,
            body_posterior_state_ids=[item.state.state_id for item in self._body],
            goal_source_state_ids=[item.state.state_id for item in self._goals],
            body_posterior_versions=[item.state.version for item in self._body],
            goal_source_versions=[item.state.version for item in self._goals],
            body_posterior_weights=[item.weight for item in self._body],
            body_posterior_log_weights=[component_log_weight(item) for item in self._body],
            goal_source_weights=[item.weight for item in self._goals],
            goal_source_log_weights=[component_log_weight(item) for item in self._goals],
            target_stamp=self.target_stamp, target_command=list(self.target_command),
            joint_increment=list(self.joint_increment), steps=[step.record() for step in self.steps],
            axes=[dict(edge_id=name, joint_index=joint, axis=list(axis))
                  for name, joint, axis in self._signature],
            covariance_action='G P G.T on the full joint state, G=diag(I,D_j.T); environmental block is identity',
            composition='ordered recorded increments about each declared fixed joint axis, summed per joint',
            future_trial_family=False, body_reconstructed_from_stiffness=False,
            observations_assimilated=0, bayes_factor_updates=0, stiffness_factor_updates=0,
            physical_truth_received=False, monte_carlo_used=False)

    def selection_record(self, command):
        if tuple(command) != self.target_command:
            raise ValueError('A recorded path has only its actual executed endpoint.')
        return dict(self.record(), command=list(self.target_command), command_jacobian=None)

    def jacobian(self, _command):
        raise ValueError('An executed command history is not a future IK prediction family.')


def predict_result(result, steps, target_stamp, frame_index, body_ids, actions):
    """Expose one transported law to control/display, without another update.

    The observation result and all its frozen Bayes/stiffness charts remain
    untouched. This prediction-only view has no new sensor posteriors or
    density-ratio charts, so it cannot become another observation update.
    """
    if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
        raise ValueError('A prediction requires a nonnegative integer frame index.')
    path = RecordedCommandPath(result, steps, body_ids=body_ids, actions=actions)
    target_stamp = float(target_stamp)
    if not np.isfinite(target_stamp) or target_stamp != path.target_stamp:
        raise ValueError('The present prediction timestamp must equal the recorded history endpoint.')
    body, goals = control_components(result)
    version = max(max(item.state.version for item in body)+1, 3*int(frame_index)+2)
    def predict_components(components, prefix):
        output = []
        for index, source in enumerate(components):
            label = '{}:{}:{}'.format(prefix, int(frame_index), index)
            state = replace(path.push(source.state, path.target_command, body_ids, actions),
                version=version, stamp=target_stamp, component_id=label)
            output.append(WeightedState(label, source.weight, state, component_log_weight(source)))
        return tuple(output)
    predicted_body = predict_components(body, 'predicted')
    explicit_goals = tuple(getattr(result.stages[-1], 'goal_components', ()))
    predicted_goals = predict_components(goals, 'predicted_goal') if explicit_goals else ()
    stage = Stage('fused', predicted_body, (), predicted_goals)
    previous = result.records.get('async_prediction', {})
    observation_source = deepcopy(previous.get('observation_source'))
    if observation_source is None:
        observation_source = dict(stamp=result.prior.stamp,
            command=list(result.prior.command), prior_state_id=result.prior.state_id,
            body_state_ids=[item.state.state_id for item in body],
            body_versions=[item.state.version for item in body],
            observation_ids=list(result.records.get('observation_ids', ())),
            stiffness=deepcopy(result.records.get('stiffness')),
            bayes_factor_surprise=deepcopy(result.records.get('bayes_factor_surprise')))
    records = dict(observation_ids=[], prediction_only=True,
        async_prediction=dict(observation_source=observation_source,
            present_stamp=target_stamp, present_command=list(path.target_command),
            present_version=version, frame_index=int(frame_index),
            observation_age_s=target_stamp-float(observation_source['stamp']),
            parent_state_ids=[item.state.state_id for item in body],
            parent_goal_state_ids=[item.state.state_id for item in goals],
            body_state_ids=[item.state.state_id for item in predicted_body],
            goal_state_ids=[item.state.state_id for item in predicted_goals],
            command_path=path.record(), observations_assimilated=0,
            bayes_factor_updates=0, stiffness_factor_updates=0),
        fixed_goal_memory=deepcopy(result.records.get('fixed_goal_memory')),
        fixed_goal_memory_event='prediction_hold',
        bayes_factor_surprise=dict(
            enabled=result.records.get('bayes_factor_surprise', {}).get('enabled', False),
            applied=False, reason='prediction only; no observation or evidence reevaluation',
            observation_source_stamp=observation_source['stamp']))
    if result.records.get('stiffness') is not None:
        records['stiffness'] = deepcopy(result.records['stiffness'])
    if 'behavior' in result.records:
        records['behavior'] = deepcopy(result.records['behavior'])
    predicted = replace(result, prior=predicted_body[0].state, stages=(stage,),
        sensor_posteriors=(), local_estimates=(), records=records,
        frozen_body_prior=None, frozen_body_posterior=None,
        frozen_body_branches=(), projected_body_numerator=None)
    control_components(predicted)
    return predicted
