"""Retain an estimated K while predicting the same observed body in real time.

Only immutable captured numbers cross worker boundaries. Each consumer owns
its own auxiliary mechanics engine; no estimator identifier, robot workspace,
factor cache or mutable StiffnessCommandMap is shared with another worker.
"""

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from probtf_observation_fusion_demo.model import component_log_weight
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
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

    @property
    def record(self):
        return json.loads(self.record_json)

    @property
    def proposal(self):
        return self.record['command_proposal']


def capture_stiffness_packet(result):
    """Capture a completed observed result's actual K representative."""
    record = result.records.get('stiffness')
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
        if not isinstance(packet, FrozenKPacket):
            raise TypeError('A command action requires an immutable captured K packet.')
        if hashlib.sha256(packet.record_json.encode()).hexdigest() != packet.packet_id:
            raise ValueError('A retained K packet hash does not match its captured record.')
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
        stiffness = _readonly(proposal['selected_K'])
        captured_command = _readonly(proposal['source_command'])
        captured_equilibrium = _readonly(proposal['auxiliary_equilibrium'])
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
        if (np.max(np.abs(residual)) > 1e-9
                or not np.all(np.isfinite(hessian))
                or np.linalg.eigvalsh(.5*(hessian+hessian.T))[0] <= 0.):
            raise ValueError('The captured K anchor is not a stable bounded equilibrium.')
        self.packet = packet
        self.stiffness = stiffness
        self.captured_command = captured_command
        self.captured_equilibrium = captured_equilibrium
        self._captured_record = record
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
        return dict(packet_id=self.packet.packet_id, density_id=proposal['density_id'],
            selected_K=self.stiffness.tolist(), captured_source_stamp=record['source_stamp'],
            captured_source_command=list(record['source_command']),
            captured_source_prior_state_id=record['source_prior_state_id'],
            captured_body_posterior_state_ids=list(record['body_posterior_state_ids']),
            captured_body_posterior_versions=list(record['body_posterior_versions']),
            captured_observation_ids=list(record['observation_ids']),
            numerical_branch_anchor=self.captured_equilibrium.tolist(),
            branch_policy='all retained-K queries start at the same captured stable equilibrium; no evaluation-order warm starts',
            physical_truth_received=False, body_reconstructed_from_stiffness=False,
            parameter_density_updated=False)

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
    captured = engine.packet.record
    density = captured.get('posterior') or captured.get('retained_prior')
    anchor = engine.anchor(command)
    proposal = dict(captured['command_proposal'])
    proposal.update(command=(2.*np.asarray(command)-np.asarray(anchor['equilibrium'])).tolist(),
        source_command=list(command), source_stamp=result.prior.stamp,
        source_prior_state_id=result.prior.state_id,
        body_posterior_state_ids=[item.state.state_id for item in body],
        selected_K=engine.stiffness.tolist(), auxiliary_equilibrium=anchor['equilibrium'],
        auxiliary_active_joint_limits=anchor['active_joint_limits'],
        density_improvement_claimed=False,
        method='same captured K representative; current-command equilibrium reanchoring only',
        captured_parameter_packet=engine.provenance())
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
    view_result = replace(result, records=dict(result.records, stiffness=view))
    # This identifier is private to this consumer. The original inference
    # identifier and its last_record/factors are never accessed or modified.
    engine.identifier.last_record = view
    mapping = RetainedStiffnessCommandMap(engine, view_result, proposal)
    view['command_correction'] = mapping.record()
    return view_result, mapping
