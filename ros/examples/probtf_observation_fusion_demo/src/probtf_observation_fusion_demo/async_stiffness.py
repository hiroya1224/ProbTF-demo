"""Retain an estimated K while predicting the same observed body in real time.

Only immutable captured numbers cross worker boundaries. Each consumer owns
its own auxiliary mechanics engine; no estimator identifier, robot workspace,
factor cache or mutable StiffnessCommandMap is shared with another worker.
"""

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

from probtf_observation_fusion_demo.model import component_log_weight
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap, equilibrium_anchor_diagnostics
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenKPacket:
    """A captured actual updater record, with no mutable cross-worker alias.

    JSON is the immutable storage, rather than a frozen dataclass containing
    mutable dictionaries/arrays. The record property returns an independent
    copy and keeps all original density IDs, evidence and source metadata.
    """
    record_json: str
    packet_id: str
    _proposal_json: object = field(default=None, init=False, repr=False, compare=False)

    @property
    def record(self):
        return json.loads(self.record_json)

    @property
    def proposal(self):
        # Keep the complete immutable history and its hash verbatim. Reading
        # the small control representative must not decode that history at
        # every physical/manifest/gradient access. Cache immutable JSON only;
        # each caller still receives an independent mutable copy as before.
        encoded = getattr(self, '_proposal_json', None)
        if encoded is None:
            encoded = json.dumps(self.record['command_proposal'],
                                 separators=(',', ':'))
            object.__setattr__(self, '_proposal_json', encoded)
        return json.loads(encoded)


@dataclass(frozen=True)
class FrozenActionKPacket:
    """The unchanged estimated K and a current-command stable branch anchor."""
    estimated_packet: FrozenKPacket
    action_json: str
    packet_id: str

    @property
    def record(self):
        return self.estimated_packet.record

    @property
    def proposal(self):
        return self.estimated_packet.proposal

    @property
    def action(self):
        return json.loads(self.action_json)


def _action_packet_id(estimated_packet, action_json):
    return hashlib.sha256(('direct-action\n'+estimated_packet.packet_id+'\n'+action_json).encode()).hexdigest()


def capture_direct_stiffness_action(engine, result, estimated_packet, previous=None):
    """Capture K_est verbatim and reanchor its action at the current command.

    The previous adopted action supplies only the stable-branch initial
    equilibrium. The final equilibrium is solved with the exact new K_est;
    no parameter density is evaluated or changed here.
    """
    if not isinstance(estimated_packet, FrozenKPacket):
        raise TypeError('A direct action needs an original estimated K packet.')
    if previous is not None and not isinstance(previous, FrozenActionKPacket):
        raise TypeError('A direct action continues a captured direct-action packet.')
    estimate = estimated_packet.record
    if float(estimate['source_stamp']) > result.prior.stamp:
        raise ValueError('A direct action cannot use an estimate from the future.')
    if previous is not None and previous.action['anchor_stamp'] > result.prior.stamp:
        raise ValueError('A direct action cannot continue a branch from the future.')
    body, goals = control_components(result)
    command = np.asarray(result.prior.command, dtype=float)
    if (not body or any(item.state.command != tuple(command)
                       or item.state.stamp != result.prior.stamp for item in body)):
        raise ValueError('A direct action needs a coherent current body prediction.')
    # Validate the new estimate before using the previous adopted branch.
    engine.set(estimated_packet)
    if previous is not None:
        engine.set(previous)
    initial = engine._evaluate(command)[0]
    initial_K = engine.stiffness.tolist()
    stiffness = np.asarray(estimated_packet.proposal['selected_K'], dtype=float)
    theta, _, _, active = engine.identifier._refine_equilibrium(initial, command, stiffness)
    action = dict(
        estimated_packet_id=estimated_packet.packet_id,
        estimated_density_id=estimated_packet.proposal['density_id'],
        estimated_source_stamp=estimate['source_stamp'],
        selected_K=estimated_packet.proposal['selected_K'],
        anchor_command=command.tolist(), anchor_stamp=result.prior.stamp,
        auxiliary_equilibrium=theta.tolist(),
        auxiliary_active_joint_limits=np.asarray(active, dtype=bool).tolist(),
        anchor_prior_state_id=result.prior.state_id,
        anchor_body_state_ids=[item.state.state_id for item in body],
        anchor_body_versions=[item.state.version for item in body],
        anchor_body_weights=[item.weight for item in body],
        anchor_body_log_weights=[component_log_weight(item) for item in body],
        anchor_goal_state_ids=[item.state.state_id for item in goals],
        anchor_goal_versions=[item.state.version for item in goals],
        anchor_goal_weights=[item.weight for item in goals],
        anchor_goal_log_weights=[component_log_weight(item) for item in goals],
        previous_action_packet_id=None if previous is None else previous.packet_id,
        branch_initial_packet_id=engine.packet.packet_id,
        branch_initial_K=initial_K, branch_initial_equilibrium=initial.tolist(),
        branch_policy='continue the captured stable branch to the current command, then refine with the exact estimated K; every trial starts at this new captured anchor',
        role='direct estimated K representative; original parameter density and proposal unchanged',
        parameter_density_updated=False, body_reconstructed_from_stiffness=False,
        physical_truth_received=False, monte_carlo_used=False)
    encoded = json.dumps(action, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return FrozenActionKPacket(estimated_packet, encoded,
                              _action_packet_id(estimated_packet, encoded))


@dataclass(frozen=True)
class FrozenExecutionKPacket:
    """A control value and branch, separate from its unchanged K estimate."""
    estimated_packet: FrozenKPacket
    execution_json: str
    packet_id: str

    @property
    def record(self):
        return self.estimated_packet.record

    @property
    def proposal(self):
        return self.estimated_packet.proposal

    @property
    def execution(self):
        return json.loads(self.execution_json)


def _execution_packet_id(estimated_packet, execution_json):
    return hashlib.sha256((estimated_packet.packet_id+'\n'+execution_json).encode()).hexdigest()


def execution_sample_record(sample, estimated_packet):
    """Bind the interpolation sample to its unchanged estimation source."""
    record = json.loads(json.dumps(sample, allow_nan=False))
    previous_id = record.pop('sample_id', None)
    record.update(estimated_packet_id=estimated_packet.packet_id,
        estimated_density_id=estimated_packet.proposal['density_id'],
        estimated_source_stamp=estimated_packet.record['source_stamp'])
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    sample_id = hashlib.sha256(encoded.encode()).hexdigest()
    if previous_id is not None and previous_id != sample_id:
        raise ValueError('An execution-filter sample hash does not match its target or values.')
    record['sample_id'] = sample_id
    return record


def execution_stiffness_packet(engine, result, estimated_packet, sample, previous=None):
    """Bind a captured continuous execution-filter value to one IK request.

    The coordinator advances the filter independently of estimation and IK.
    This function only freezes its value and a stable branch for this one
    calculation. The auxiliary equilibrium does not generate a body law.
    """
    if not isinstance(estimated_packet, FrozenKPacket):
        raise TypeError('Execution damping needs an unchanged estimated K packet.')
    if previous is not None and not isinstance(previous, FrozenExecutionKPacket):
        raise TypeError('Execution damping must resume from an adopted execution packet.')
    engine.set(estimated_packet)
    if previous is not None:
        engine.set(previous)
    estimate = estimated_packet.record
    if float(estimate['source_stamp']) > result.prior.stamp:
        raise ValueError('Execution damping cannot use an estimate from the future.')
    execution = execution_sample_record(sample, estimated_packet)
    if (execution['stamp'] != result.prior.stamp
            or not np.array_equal(execution['estimated_K'], estimated_packet.proposal['selected_K'])):
        raise ValueError('An IK execution sample must retain its actual filter time and target estimate.')
    command = np.asarray(result.prior.command, dtype=float)
    initial = engine._evaluate(command)[0]
    theta, _, _, active = engine.identifier._refine_equilibrium(
        initial, command, np.asarray(execution['execution_K']))
    body, _ = control_components(result)
    execution.update(filter_sample=dict(execution),
        estimated_packet_id=estimated_packet.packet_id,
        estimated_density_id=estimated_packet.proposal['density_id'],
        estimated_source_stamp=estimate['source_stamp'],
        estimated_source_command=estimate['source_command'],
        estimated_body_state_ids=estimate['body_posterior_state_ids'],
        previous_execution_packet_id=None if previous is None else previous.packet_id,
        anchor_command=command.tolist(), auxiliary_equilibrium=theta.tolist(),
        auxiliary_active_joint_limits=np.asarray(active, dtype=bool).tolist(),
        anchor_prior_state_id=result.prior.state_id,
        anchor_body_state_ids=[item.state.state_id for item in body],
        anchor_body_versions=[item.state.version for item in body],
        role='damped execution parameter; not a parameter posterior or a newly selected density mode',
        timebase='physical simulated time; filter advances at every coordinator motion tick',
        update_cadence='filter advances without waiting for inference/IK; capture once per IK calculation and atomically adopt with its snapshot',
        body_reconstructed_from_stiffness=False, parameter_density_updated=False,
        physical_truth_received=False, monte_carlo_used=False)
    encoded = json.dumps(execution, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return FrozenExecutionKPacket(estimated_packet, encoded,
                                  _execution_packet_id(estimated_packet, encoded))


def capture_stiffness_packet(result):
    """Capture a completed observed result's actual K representative."""
    record = result.records.get('stiffness')
    if record is not None and record.get('execution_stiffness') is not None:
        raise ValueError('An execution parameter view cannot become a new stiffness estimate.')
    if record is not None and record.get('action_stiffness') is not None:
        raise ValueError('An action parameter view cannot become a new stiffness estimate.')
    proposal = None if record is None else record.get('command_proposal')
    density = None if record is None else record.get('posterior') or record.get('retained_prior')
    body, _ = control_components(result)
    if record is None or proposal is None or density is None:
        raise ValueError('A retained K packet requires a completed actual stiffness representative.')
    ids = [item.state.state_id for item in body]
    command = tuple(result.prior.command)
    if (record.get('source_prior_state_id') != result.prior.state_id
            or proposal.get('source_prior_state_id') != result.prior.state_id
            or record.get('source_stamp') != result.prior.stamp
            or proposal.get('source_stamp') != result.prior.stamp
            or tuple(record.get('source_command', ())) != command
            or tuple(proposal.get('source_command', ())) != command
            or record.get('body_posterior_state_ids') != ids
            or proposal.get('body_posterior_state_ids') != ids
            or record.get('body_posterior_versions') != [item.state.version for item in body]
            or record.get('body_posterior_weights') != [item.weight for item in body]
            or record.get('body_posterior_log_weights') != [component_log_weight(item) for item in body]
            or density.get('density_id') != proposal.get('density_id')):
        raise ValueError('A retained K packet must preserve its actual observation and body provenance.')
    if any(item.state.command != command or item.state.stamp != result.prior.stamp for item in body):
        raise ValueError('Captured body candidates must share their actual observation command/time.')
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return FrozenKPacket(encoded, hashlib.sha256(encoded.encode()).hexdigest())


class PhysicalActionEngine:
    """A worker-local known auxiliary model for one immutable retained K.

    The name denotes a model-based command action, never a source of physical
    truth. Queries start at the same captured equilibrium, independently of
    evaluation order. Failure is explicit; no K substitution or rigid fallback
    is introduced. Each worker must construct a separate instance.
    """
    def __init__(self, urdf_path, body_edge_ids, settings):
        self.identifier = StiffnessIdentifier(urdf_path, body_edge_ids, settings)
        self.packet = None
        self._cache = {}
        self.equilibrium_evaluation_count = 0

    def set(self, packet):
        if not isinstance(packet, (FrozenKPacket, FrozenActionKPacket, FrozenExecutionKPacket)):
            raise TypeError('A command action requires an immutable captured K packet.')
        estimated = (packet.estimated_packet
                     if isinstance(packet, (FrozenActionKPacket, FrozenExecutionKPacket)) else packet)
        if not isinstance(estimated, FrozenKPacket):
            raise TypeError('An execution packet must reference an original estimated K packet.')
        if hashlib.sha256(estimated.record_json.encode()).hexdigest() != estimated.packet_id:
            raise ValueError('A retained K packet hash does not match its captured record.')
        execution = packet.execution if isinstance(packet, FrozenExecutionKPacket) else None
        if execution is not None and packet.packet_id != _execution_packet_id(estimated, packet.execution_json):
            raise ValueError('An execution K packet hash does not match its captured control value.')
        action = packet.action if isinstance(packet, FrozenActionKPacket) else None
        if action is not None and packet.packet_id != _action_packet_id(estimated, packet.action_json):
            raise ValueError('An action K packet hash does not match its captured anchor.')
        if self.packet is not None and packet.packet_id == self.packet.packet_id:
            return
        record = packet.record
        proposal = record.get('command_proposal')
        density = record.get('posterior') or record.get('retained_prior')
        if (proposal is None or density is None or proposal.get('density_id') != density.get('density_id')
                or record.get('auxiliary_model', {}).get('urdf_sha256')
                    != self.identifier.model_source['urdf_sha256']
                or record.get('auxiliary_model', {}).get('gravity')
                    != self.identifier.model_source['gravity']
                or record.get('body_marginal', {}).get('edge_ids') != list(self.identifier.body_edge_ids)):
            raise ValueError('The retained K packet and this worker must use the same declared auxiliary model.')
        if execution is not None and (
                execution['estimated_packet_id'] != estimated.packet_id
                or execution['estimated_density_id'] != proposal['density_id']
                or not np.array_equal(execution['estimated_K'], proposal['selected_K'])
                or not np.allclose(execution['execution_logK'], np.log(execution['execution_K']),
                                   rtol=0., atol=1e-14)):
            raise ValueError('An execution K packet must retain the exact source estimate.')
        if execution is not None and (
                execution_sample_record(execution['filter_sample'], estimated) != execution['filter_sample']
                or any(execution.get(key) != value for key, value in execution['filter_sample'].items())):
            raise ValueError('An execution packet must freeze its exact interpolated filter sample.')
        if action is not None and (
                action['estimated_packet_id'] != estimated.packet_id
                or action['estimated_density_id'] != proposal['density_id']
                or action['estimated_source_stamp'] != record['source_stamp']
                or not np.isfinite(action['anchor_stamp'])
                or action['anchor_stamp'] < record['source_stamp']
                or not np.array_equal(action['selected_K'], proposal['selected_K'])):
            raise ValueError('A direct action packet must retain the exact source estimate.')
        stiffness = _readonly(proposal['selected_K'] if execution is None else execution['execution_K'])
        anchor = action if action is not None else execution
        captured_command = _readonly(proposal['source_command'] if anchor is None else anchor['anchor_command'])
        captured_equilibrium = _readonly(proposal['auxiliary_equilibrium'] if anchor is None else anchor['auxiliary_equilibrium'])
        if (stiffness.shape != (self.identifier.robot.nv,)
                or captured_command.shape != stiffness.shape
                or captured_equilibrium.shape != stiffness.shape
                or not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0.)
                or not np.all(np.isfinite(captured_command))
                or not np.all(np.isfinite(captured_equilibrium))):
            raise ValueError('The retained K and its captured equilibrium must be finite and dimensionally valid.')
        # Validate the captured equilibrium itself before accepting the epoch.
        hessian, _ = self.identifier.sensitivity.equilibrium_jacobians(
            captured_equilibrium, captured_command, stiffness)
        active = self.identifier.sensitivity.last_active_set.copy()
        residual = self.identifier.robot.tau_gravity(captured_equilibrium)+self.identifier.spring.torque(
            captured_equilibrium, captured_command, stiffness)
        residual[active] = 0.
        anchor_diagnostics=equilibrium_anchor_diagnostics(hessian,residual)
        if action is not None and not np.array_equal(action['auxiliary_active_joint_limits'], active):
            raise ValueError('A direct action packet must retain its actual equilibrium active set.')
        self.packet = packet
        self.stiffness = stiffness
        self.captured_command = captured_command
        self.captured_equilibrium = captured_equilibrium
        self._captured_record = record
        self._execution_record = execution
        self._action_record = action
        self._anchor_numerical_diagnostics=anchor_diagnostics
        self._cache = {}
        self.equilibrium_evaluation_count = 0
        self._store(captured_command, captured_equilibrium, hessian, active)

    def _store(self, command, equilibrium, hessian, active):
        rhs = np.diag(self.stiffness)
        rhs[active] = 0.
        derivative = np.linalg.solve(hessian, rhs)
        if not np.all(np.isfinite(derivative)):
            raise ValueError('A retained-K command sensitivity is nonfinite.')
        item = (_readonly(equilibrium), _readonly(derivative), _readonly(active, bool))
        self._cache[tuple(float(value) for value in command)] = item
        return item

    def _evaluate(self, command):
        if self.packet is None:
            raise ValueError('A command action requires a previously captured K packet.')
        command = np.asarray(command, dtype=float)
        if command.shape != self.captured_command.shape or not np.all(np.isfinite(command)):
            raise ValueError('A retained-K action requires one finite command per joint.')
        key = tuple(float(value) for value in command)
        if key not in self._cache:
            self.equilibrium_evaluation_count += 1
            theta, hessian, _, active = self.identifier._refine_equilibrium(
                self.captured_equilibrium, command, self.stiffness)
            self._store(command, theta, hessian, active)
        return self._cache[key]

    def provenance(self):
        if self.packet is None:
            raise ValueError('No retained K packet has been accepted.')
        record = self._captured_record
        proposal = record['command_proposal']
        result = dict(packet_id=self.packet.packet_id, density_id=proposal['density_id'],
            selected_K=self.stiffness.tolist(), captured_source_stamp=record['source_stamp'],
            captured_source_command=list(record['source_command']),
            captured_source_prior_state_id=record['source_prior_state_id'],
            captured_body_posterior_state_ids=list(record['body_posterior_state_ids']),
            captured_body_posterior_versions=list(record['body_posterior_versions']),
            captured_observation_ids=list(record['observation_ids']),
            numerical_branch_anchor=self.captured_equilibrium.tolist(),
            numerical_branch_anchor_command=self.captured_command.tolist(),
            branch_policy='all retained-K queries start at the same captured finite auxiliary equilibrium iterate; no evaluation-order warm starts',
            numerical_anchor_diagnostics=dict(self._anchor_numerical_diagnostics),
            physical_truth_received=False, body_reconstructed_from_stiffness=False,
            parameter_density_updated=False)
        if self._execution_record is not None:
            result.update(estimated_K=list(proposal['selected_K']),
                estimated_packet_id=self.packet.estimated_packet.packet_id,
                execution_packet_id=self.packet.packet_id,
                execution_stiffness=self.packet.execution,
                stiffness_role='damped execution K; captured density and its representative are unchanged')
        elif self._action_record is not None:
            result.update(estimated_K=list(proposal['selected_K']),
                estimated_packet_id=self.packet.estimated_packet.packet_id,
                action_packet_id=self.packet.packet_id,
                action_stiffness=self.packet.action,
                stiffness_role='direct estimated K representative; original parameter density and proposal unchanged')
        return result

    def anchor(self, command):
        before = self.equilibrium_evaluation_count
        theta, derivative, active = self._evaluate(command)
        return dict(command=np.asarray(command).tolist(), equilibrium=theta.tolist(),
            command_jacobian=derivative.tolist(), active_joint_limits=active.tolist(),
            current_equilibrium_evaluation_count=self.equilibrium_evaluation_count-before,
            **self.provenance())

    def increment(self, command_from, command_to):
        count_before = self.equilibrium_evaluation_count
        before, _, _ = self._evaluate(command_from)
        after, derivative, active = self._evaluate(command_to)
        return dict(source_command=np.asarray(command_from).tolist(),
            query_command=np.asarray(command_to).tolist(), joint_increment=(after-before).tolist(),
            command_jacobian=derivative.tolist(), source_equilibrium=before.tolist(),
            query_equilibrium=after.tolist(), active_joint_limits=active.tolist(),
            current_equilibrium_evaluation_count=self.equilibrium_evaluation_count-count_before,
            **self.provenance())


class RetainedStiffnessCommandMap(StiffnessCommandMap):
    """The strict existing source contract, with one shared frozen action.

    The IK and command-history predictor query the very same captured-anchor
    numerical map. Merely resolving both from the same K with different
    initial equilibria would not establish this on multiple stable branches.
    """
    def __init__(self, engine, result, proposal):
        self._action_engine = engine
        self._packet_id = engine.packet.packet_id
        super().__init__(engine.identifier, result, proposal)
        self._record['branch_policy'] = engine.provenance()['branch_policy']
        self._record['captured_parameter_packet'] = engine.provenance()
        self._record['command_action_evaluator'] = 'same worker-local captured-anchor PhysicalActionEngine for prediction and every IK trial'
        if isinstance(engine.packet, FrozenExecutionKPacket):
            self._record['stiffness_role'] = 'damped execution control parameter; original estimated representative and density unchanged'
            self._record['execution_stiffness'] = engine.packet.execution
        elif isinstance(engine.packet, FrozenActionKPacket):
            self._record['stiffness_role'] = engine.provenance()['stiffness_role']
            self._record['action_stiffness'] = engine.packet.action

    def _evaluate(self, command):
        if self._action_engine.packet.packet_id != self._packet_id:
            raise ValueError('The retained-K command map cannot cross a parameter-packet epoch; reanchor a new map.')
        return self._action_engine._evaluate(command)


def reanchor_retained_stiffness(engine, result, packet=None):
    """Build an honestly labelled current prediction view with the same K.

    The current body is supplied by the caller's actual command-history
    prediction. This function never reconstructs it from K and never updates
    any observation or stiffness density. The captured K evidence remains
    separate from the command/time metadata of the predicted body.
    """
    if packet is not None:
        engine.set(packet)
    if engine.packet is None:
        raise ValueError('A retained-K view requires a completed captured K packet.')
    body, _ = control_components(result)
    command = tuple(result.prior.command)
    if not body or any(item.state.command != command or item.state.stamp != result.prior.stamp
                       for item in body):
        raise ValueError('A retained-K view requires one coherent current body prediction.')
    if isinstance(engine.packet, FrozenActionKPacket):
        action = engine.packet.action
        if action['anchor_stamp'] > result.prior.stamp:
            raise ValueError('A direct action view cannot use an anchor from the future.')
        if action['anchor_stamp'] == result.prior.stamp and (
                action['anchor_command'] != list(command)
                or action['anchor_prior_state_id'] != result.prior.state_id
                or action['anchor_body_state_ids'] != [item.state.state_id for item in body]
                or action['anchor_body_versions'] != [item.state.version for item in body]
                or action['anchor_body_weights'] != [item.weight for item in body]
                or action['anchor_body_log_weights'] != [component_log_weight(item) for item in body]):
            raise ValueError('A direct action anchor belongs to a different current body source.')
    captured = engine.packet.record
    density = captured.get('posterior') or captured.get('retained_prior')
    anchor = engine.anchor(command)
    damped = isinstance(engine.packet, FrozenExecutionKPacket)
    direct = isinstance(engine.packet, FrozenActionKPacket)
    # The original estimated command proposal is kept verbatim below. The
    # execution proposal is a separate current-body action, never a new mode
    # or density/score evaluation at the filtered value.
    proposal = (dict(density_id=density['density_id']) if damped or direct else
                dict(captured['command_proposal']))
    proposal.update(command=(2.*np.asarray(command)-np.asarray(anchor['equilibrium'])).tolist(),
        source_command=list(command), source_stamp=result.prior.stamp,
        source_prior_state_id=result.prior.state_id,
        body_posterior_state_ids=[item.state.state_id for item in body],
        selected_K=engine.stiffness.tolist(),
        auxiliary_equilibrium=anchor['equilibrium'],
        auxiliary_active_joint_limits=anchor['active_joint_limits'],
        density_improvement_claimed=False,
        method=('damped execution K; current-command equilibrium reanchoring only' if damped else
                'same captured K representative; current-command equilibrium reanchoring only'),
        captured_parameter_packet=engine.provenance())
    if damped:
        proposal['selected_logK'] = engine.packet.execution['execution_logK']
    elif direct:
        proposal['selected_logK'] = captured['command_proposal']['selected_logK']
    view = dict(source_stamp=result.prior.stamp, source_command=list(command),
        source_prior_state_id=result.prior.state_id, source_prior_version=result.prior.version,
        source_prior_state_ids=[result.prior.state_id], source_prior_weights=[1.],
        body_posterior_state_ids=[item.state.state_id for item in body],
        body_posterior_versions=[item.state.version for item in body],
        body_posterior_weights=[item.weight for item in body],
        body_posterior_log_weights=[component_log_weight(item) for item in body],
        body_posterior_labels=[item.label for item in body],
        observation_ids=list(result.records.get('observation_ids', ())),
        prior=density, posterior=None, retained_prior=density, prior_retained=True,
        status='prediction_with_retained_parameters', update_applied=False,
        information_rank=None, information_eigenvalues=None,
        prior_source=captured.get('prior_source'), auxiliary_model=captured['auxiliary_model'],
        approximation='same observed body law advanced by executed command actions; captured stiffness density unchanged',
        support_regularization_added=False, proposal_approximation=None,
        proposal_status='retained_captured_representative', command_proposal=proposal,
        proposal_selection=dict(method='no parameter selection or density reevaluation; retain the completed representative',
            density_evaluation_count=0,
            current_equilibrium_evaluation_count=anchor['current_equilibrium_evaluation_count'],
            captured_parameter_packet=engine.provenance()),
        captured_source_stamp=captured['source_stamp'],
        captured_source_command=list(captured['source_command']),
        captured_source_prior_state_id=captured['source_prior_state_id'],
        captured_body_posterior_state_ids=list(captured['body_posterior_state_ids']),
        captured_body_posterior_versions=list(captured['body_posterior_versions']),
        captured_observation_ids=list(captured['observation_ids']),
        captured_status=captured['status'], captured_update_applied=captured['update_applied'],
        captured_density_id=density['density_id'], captured_packet_id=engine.packet.packet_id,
        body_reconstructed_from_stiffness=False, feedback_to_observation_update=False,
        feedback_to_body=True, feedback_to_body_prediction=True,
        feedback_to_body_or_control=True,
        feedback_to_control='retained K corrects IK and the same next command action; no new parameter observation')
    if damped:
        view.update(command_proposal=captured['command_proposal'],
            execution_command_proposal=proposal, execution_stiffness=engine.packet.execution,
            captured_estimate_packet_id=engine.packet.estimated_packet.packet_id,
            proposal_status='damped_execution_K',
            approximation='same observed body advanced by recorded execution-K actions; estimated stiffness density unchanged')
        view['proposal_selection']['method'] = 'no density reevaluation; damped execution control value, separate from the estimated representative'
    elif direct:
        view.update(command_proposal=captured['command_proposal'],
            action_command_proposal=proposal, action_stiffness=engine.packet.action,
            captured_estimate_packet_id=engine.packet.estimated_packet.packet_id,
            proposal_status='direct_estimated_K_action')
    view_result = replace(result, records=dict(result.records, stiffness=view))
    # This identifier is private to this consumer. The original inference
    # identifier and its last_record/factors are never accessed or modified.
    engine.identifier.last_record = view
    mapping = RetainedStiffnessCommandMap(engine, view_result, proposal)
    view['command_correction'] = mapping.record()
    return view_result, mapping
