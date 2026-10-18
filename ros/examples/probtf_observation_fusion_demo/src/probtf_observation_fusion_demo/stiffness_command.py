"""Frozen stiffness correction of the observed body's command action.

The auxiliary model supplies only a change of equilibrium joint angles.  That
change acts on the complete, already inferred transform distribution; it never
constructs a replacement body from the stiffness estimate.
"""

from copy import deepcopy

import numpy as np

from .moving_control import command_state, control_components
from .model import component_log_weight


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def equilibrium_anchor_diagnostics(hessian, residual):
    """Record numerical quality without rejecting a usable command action."""
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(residual)):
        raise ValueError('The command anchor equations contain nonfinite values.')
    record=dict(residual_inf=float(np.max(np.abs(residual))),
        residual_within_legacy_tolerance=bool(np.max(np.abs(residual))<=1e-9),
        numerical_quality_is_acceptance_gate=False)
    try:
        eigenvalues=np.linalg.eigvalsh(.5*hessian+.5*hessian.T)
        available=bool(np.all(np.isfinite(eigenvalues)))
        record.update(hessian_eigenvalues=eigenvalues.tolist() if available else None,
            positive_curvature=bool(eigenvalues[0]>0.) if available else None)
    except np.linalg.LinAlgError as error:
        record.update(hessian_eigenvalues=None,positive_curvature=None,
            eigendecomposition_diagnostic_error=str(error))
    return record


class StiffnessCommandMap:
    """One observation's fixed K and equilibrium branch for all trial commands."""

    def __init__(self, identifier, result, proposal):
        components, goal_components = control_components(result)
        record = result.records.get('stiffness')
        density = None if record is None else (
            record.get('posterior') or record.get('retained_prior'))
        command = tuple(result.prior.command)
        state_ids = [item.state.state_id for item in components]
        if (not components or proposal is None or record is None
                or identifier.last_record is not record
                or record.get('action_command_proposal', record.get('execution_command_proposal', record.get('command_proposal'))) is not proposal):
            raise ValueError('A stiffness command map requires the same recorded current proposal.')
        if (tuple(proposal['source_command']) != command
                or float(proposal['source_stamp']) != result.prior.stamp
                or proposal['source_prior_state_id'] != result.prior.state_id
                or proposal['body_posterior_state_ids'] != state_ids
                or density is None or proposal['density_id'] != density['density_id']
                or record['source_command'] != list(command)
                or record['source_stamp'] != result.prior.stamp
                or record['source_prior_state_id'] != result.prior.state_id
                or record['body_posterior_state_ids'] != state_ids
                or record['body_posterior_versions'] != [item.state.version for item in components]
                or record['body_posterior_weights'] != [item.weight for item in components]
                or record['body_posterior_log_weights'] != [component_log_weight(item) for item in components]):
            raise ValueError('A stiffness command map belongs to a different body update.')
        if any(item.state.command != command or item.state.stamp != result.prior.stamp
               for item in components):
            raise ValueError('All source body candidates must share the current command and stamp.')
        self.identifier = identifier
        self._source_record = record
        self._source_proposal = proposal
        self._source_prior_id = result.prior.state_id
        self._source_ordered_ids = tuple(state_ids)
        self._source_versions = tuple(item.state.version for item in components)
        self._source_weights = tuple(item.weight for item in components)
        self._source_log_weights = tuple(component_log_weight(item) for item in components)
        self._goal_ordered_ids = tuple(item.state.state_id for item in goal_components)
        self._goal_versions = tuple(item.state.version for item in goal_components)
        self._goal_weights = tuple(item.weight for item in goal_components)
        self._goal_log_weights = tuple(component_log_weight(item) for item in goal_components)
        self._goal_parent_internal_id = (components[0].state.state_id
            if getattr(result.stages[-1], 'goal_components', ()) else None)
        self._density_id = density['density_id']
        self.source_command = _readonly(command)
        self.source_stamp = result.prior.stamp
        self.source_state_ids = frozenset(state_ids + list(self._goal_ordered_ids))
        self.stiffness = _readonly(proposal['selected_K'])
        self.anchor = _readonly(proposal['auxiliary_equilibrium'])
        if (self.stiffness.shape != self.source_command.shape
                or self.anchor.shape != self.source_command.shape
                or not np.all(np.isfinite(self.stiffness))
                or np.any(self.stiffness <= 0.)
                or not np.all(np.isfinite(self.anchor))):
            raise ValueError('A stiffness command map needs finite positive K and a finite anchor.')
        self._cache = {}
        hessian, _ = identifier.sensitivity.equilibrium_jacobians(
            self.anchor, self.source_command, self.stiffness)
        active = identifier.sensitivity.last_active_set.copy()
        residual = identifier.robot.tau_gravity(self.anchor) + identifier.spring.torque(
            self.anchor, self.source_command, self.stiffness)
        residual[active] = 0.
        anchor_diagnostics=equilibrium_anchor_diagnostics(hessian,residual)
        anchor_jacobian = self._store(self.source_command, self.anchor, hessian, active)[1]
        self._record = dict(
            source_command=list(command), source_stamp=self.source_stamp,
            source_prior_state_id=result.prior.state_id,
            body_posterior_state_ids=state_ids,
            body_posterior_versions=[item.state.version for item in components],
            body_posterior_weights=[float(item.weight) for item in components],
            body_posterior_log_weights=list(self._source_log_weights),
            goal_source_state_ids=list(self._goal_ordered_ids),
            goal_source_versions=list(self._goal_versions),
            goal_source_weights=list(self._goal_weights),
            goal_source_log_weights=list(self._goal_log_weights),
            goal_parent_internal_state_id=self._goal_parent_internal_id,
            density_id=proposal['density_id'], selected_K=self.stiffness.tolist(),
            K_units='N m/rad', anchor_equilibrium=self.anchor.tolist(),
            anchor_command_jacobian=anchor_jacobian.tolist(),
            anchor_active_joint_limits=active.tolist(),
            numerical_anchor_diagnostics=anchor_diagnostics,
            model_source=deepcopy(identifier.model_source),
            prediction_method='same observed joint law acted on by theta_eq(q,K)-theta_eq(c,K)',
            branch_policy='every command query starts at the same recorded current equilibrium; no evaluation-order warm starts',
            command_sensitivity='solve(H, diag(K)) on the free KKT coordinates; active RHS rows zero',
            stiffness_role='fixed numerical representative of the retained parameter density; no parameter integration or global mode claimed',
            body_reconstructed_from_stiffness=False, physical_truth_received=False,
        )

    def validate_result(self, result):
        """Reject using this fixed prediction family for a different update."""
        try:
            components, goal_components = control_components(result)
        except ValueError as error:
            raise ValueError('A stiffness command map belongs to a different or modified body update: '
                             + str(error)) from error
        record = result.records.get('stiffness')
        density = None if record is None else (
            record.get('posterior') or record.get('retained_prior'))
        if (result.prior.state_id != self._source_prior_id
                or result.prior.stamp != self.source_stamp
                or result.prior.command != tuple(self.source_command)
                or tuple(item.state.state_id for item in components) != self._source_ordered_ids
                or tuple(item.state.version for item in components) != self._source_versions
                or tuple(item.weight for item in components) != self._source_weights
                or tuple(component_log_weight(item) for item in components) != self._source_log_weights
                or tuple(item.state.state_id for item in goal_components) != self._goal_ordered_ids
                or tuple(item.state.version for item in goal_components) != self._goal_versions
                or tuple(item.weight for item in goal_components) != self._goal_weights
                or tuple(component_log_weight(item) for item in goal_components) != self._goal_log_weights
                or (components[0].state.state_id if getattr(result.stages[-1], 'goal_components', ())
                    else None) != self._goal_parent_internal_id
                or record is not self._source_record
                or record.get('action_command_proposal', record.get('execution_command_proposal', record.get('command_proposal'))) is not self._source_proposal
                or density is None or density['density_id'] != self._density_id
                or record['source_prior_state_id'] != self._source_prior_id
                or record['source_stamp'] != self.source_stamp
                or tuple(record['source_command']) != tuple(self.source_command)
                or tuple(record['body_posterior_state_ids']) != self._source_ordered_ids
                or tuple(record['body_posterior_versions']) != self._source_versions
                or tuple(record['body_posterior_weights']) != self._source_weights
                or tuple(record['body_posterior_log_weights']) != self._source_log_weights
                or self._source_proposal['density_id'] != self._density_id
                or self._source_proposal['source_prior_state_id'] != self._source_prior_id
                or tuple(self._source_proposal['body_posterior_state_ids']) != self._source_ordered_ids
                or self._source_proposal['source_stamp'] != self.source_stamp
                or tuple(self._source_proposal['source_command']) != tuple(self.source_command)
                or not np.array_equal(self._source_proposal['selected_K'], self.stiffness)
                or not np.array_equal(self._source_proposal['auxiliary_equilibrium'], self.anchor)):
            raise ValueError('A stiffness command map belongs to a different or modified body update.')

    def _store(self, command, theta, hessian, active):
        right_hand_side = np.diag(self.stiffness)
        right_hand_side[active, :] = 0.
        derivative = np.linalg.solve(hessian, right_hand_side)
        if not np.all(np.isfinite(derivative)):
            raise ValueError('Nonfinite stiffness-corrected command sensitivity.')
        item = (_readonly(theta), _readonly(derivative), _readonly(active, bool))
        self._cache[tuple(float(value) for value in command)] = item
        return item

    def _evaluate(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != self.source_command.shape or not np.all(np.isfinite(command)):
            raise ValueError('A stiffness-corrected trial needs one finite command per joint.')
        key = tuple(float(value) for value in command)
        if key not in self._cache:
            theta, hessian, _, active = self.identifier._refine_equilibrium(
                self.anchor, command, self.stiffness)
            self._store(command, theta, hessian, active)
        return self._cache[key]

    def push(self, state, command, body_ids, actions):
        """Right-act on the full source law, retaining the actual sent command."""
        if (state.state_id not in self.source_state_ids
                or state.stamp != self.source_stamp
                or state.command != tuple(self.source_command)):
            raise ValueError('A stiffness command action requires its frozen source body candidate.')
        if tuple(body_ids) != self.identifier.body_edge_ids or len(actions) != len(body_ids):
            raise ValueError('The stiffness command action must use the same declared body path.')
        for action, edge in zip(actions, self.identifier.edges):
            axis = np.asarray(action['axis'], dtype=float)
            if (int(action['joint_index']) != edge[3]
                    or np.linalg.norm(axis) == 0.
                    or not np.allclose(axis/np.linalg.norm(axis), edge[4], atol=1e-12, rtol=0.)):
                raise ValueError('The stiffness command action differs from the auxiliary joint axes.')
        theta, _, _ = self._evaluate(command)
        if np.array_equal(command, self.source_command):
            return state
        return command_state(state, command, body_ids, actions,
                             joint_increment=theta-self.anchor)

    def jacobian(self, command):
        """Derivative of equilibrium angles with respect to the sent command."""
        return self._evaluate(command)[1]

    def selection_record(self, command):
        """The exact equilibrium action for one selected actual command."""
        theta, derivative, active = self._evaluate(command)
        return dict(command=np.asarray(command, dtype=float).tolist(),
            auxiliary_equilibrium=theta.tolist(),
            joint_increment=(theta-self.anchor).tolist(),
            command_jacobian=derivative.tolist(), active_joint_limits=active.tolist(),
            source_command=self.source_command.tolist(), source_stamp=self.source_stamp,
            source_prior_state_id=self._source_prior_id,
            body_posterior_state_ids=list(self._source_ordered_ids),
            goal_source_state_ids=list(self._goal_ordered_ids),
            goal_source_weights=list(self._goal_weights),
            goal_source_log_weights=list(self._goal_log_weights),
            goal_parent_internal_state_id=self._goal_parent_internal_id,
            density_id=self._density_id)

    def record(self):
        """Independent serializable provenance; later cache writes cannot mutate it."""
        return deepcopy(self._record)
