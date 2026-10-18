"""Non-gating agreement of captured auxiliary FK with the retained body law.

This does not estimate K or reconstruct the body.  The equilibrium is the one
already captured for this exact trajectory request.  One worker-local URDF FK
evaluates it; every statistical query still reads the original canonical law.
"""

import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf.geometry import quat_to_rotmat, rotmat_to_quat
from probtf.spherical_law.support import covariance_support

from .body_native import pose_query
from .model import component_log_weight
from .moving_control import control_components


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _identity_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _support_diagnostic(residual, covariance):
    """Use the full supplied covariance, never independent per-edge scores."""
    support = covariance_support(covariance, 'K/body diagnostic covariance')
    outside = residual-support.basis @ (support.basis.T @ residual)
    outside_norm = float(np.linalg.norm(outside))
    # This is the existing Gaussian affine-support convention, not a control
    # threshold, concentration criterion, covariance floor or acceptance gate.
    support_tolerance = 1.e-10*max(1., float(np.linalg.norm(residual)))
    inside = outside_norm <= support_tolerance
    projected = float(residual @ support.inverse @ residual)
    return dict(status='available' if inside else 'unavailable',
        reason=None if inside else 'auxiliary FK lies outside the body marginal affine support',
        dimension=int(residual.size), rank=int(support.rank),
        mahalanobis_squared=projected if inside else None,
        projected_mahalanobis_squared=projected,
        affine_support_residual_norm=outside_norm,
        affine_support_roundoff_tolerance=support_tolerance,
        covariance_rank_roundoff_tolerance=float(support.tolerance),
        covariance_sha256=_array_hash(covariance),
        covariance_diagonal=np.diag(covariance).tolist(),
        residual=residual.tolist(), full_cross_covariance_used=True,
        covariance_modified=False)


class KBodyConsistencyEvaluator:
    """Build once in the artifact worker; no solver or simulator is accepted.

    ``evaluate(result, candidate_identity, request=None)`` returns only a JSON
    diagnostic.  A missing packet, inconsistent provenance, unsupported query,
    or failed diagnostic arithmetic becomes ``unavailable`` rather than an
    exception that can reject a trajectory or change a candidate distribution.
    """

    def __init__(self, urdf_path):
        self.robot = None
        self.urdf_sha256 = None
        self.initialization_error = None
        try:
            self.urdf_path = Path(urdf_path)
            self.urdf_sha256 = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
            self.robot = RobotArm(str(self.urdf_path), tip_link='tool0', base_link='base_link')
            self.joints = {joint.find('child').get('link'): joint
                           for joint in ET.parse(str(self.urdf_path)).getroot().findall('joint')}
        except Exception as error:
            # Even optional instrumentation initialization must not kill the
            # artifact worker that publishes an already accepted trajectory.
            self.initialization_error = dict(exception_type=type(error).__name__, reason=str(error))

    def evaluate(self, result, candidate_identity, *, request=None):
        started = time.monotonic()
        request = {} if request is None else request
        record = dict(schema='captured_K_body_consistency_v1',
            candidate_id=None, request_id=request.get('request_id'),
            index=request.get('index'), capture_index=request.get('capture_index'),
            diagnostic_only=True, control_gating=False, source_laws_modified=False,
            physical_truth_received=False, equilibrium_solve_performed=False,
            interpretation='absolute auxiliary-model agreement; control transports the retained body by increments without reconstructing it from FK',
            urdf_sha256=self.urdf_sha256, status='unavailable')
        try:
            record['candidate_id'] = _identity_hash(candidate_identity)
            parameter = candidate_identity.get('parameter') or {}
            record.update(source_stamp=candidate_identity.get('stamp'),
                source_command=candidate_identity.get('command'),
                source_prior_state_id=candidate_identity.get('prior_state_id'),
                source_version=candidate_identity.get('version'),
                body_source_state_ids=[item['state_id'] for item in candidate_identity.get('body', ())],
                goal_source_state_ids=[item['state_id'] for item in candidate_identity.get('goals', ())],
                density_id=parameter.get('density_id'),
                parameter_packet_id=parameter.get('estimated_packet_id'),
                selected_K=parameter.get('selected_K'))
            if self.initialization_error is not None:
                record['initialization_error'] = dict(self.initialization_error)
                raise RuntimeError('Optional K/body diagnostic initialization failed: '+
                                   self.initialization_error['reason'])
            record.update(self._evaluate(result, candidate_identity))
        except Exception as error:
            # This whole operation is optional instrumentation, including its
            # source audit.  Never turn its exception into a control failure.
            record.update(status='unavailable', reason=str(error),
                          exception_type=type(error).__name__)
        record['computation_seconds'] = time.monotonic()-started
        return record

    def _evaluate(self, result, identity):
        body, goals = control_components(result)
        parameter = identity.get('parameter') or {}
        reference = parameter.get('action_packet_ref')
        if reference is None:
            return dict(status='unavailable', reason='no captured direct-action K packet',
                        source_stamp=result.prior.stamp, source_command=list(result.prior.command))
        action = json.loads(reference['action_json'])
        packet_hash = hashlib.sha256(('direct-action\n'+reference['estimated_packet_id']+'\n'+
                                     reference['action_json']).encode()).hexdigest()
        if packet_hash != reference['packet_id']:
            raise ValueError('Captured direct-action K packet checksum differs.')
        source_command = list(result.prior.command)
        source_stamp = result.prior.stamp
        body_ids = [item.state.state_id for item in body]
        goal_ids = [item.state.state_id for item in goals]
        if (action['anchor_command'] != source_command or action['anchor_stamp'] != source_stamp
                or action['anchor_prior_state_id'] != result.prior.state_id
                or action['anchor_body_state_ids'] != body_ids
                or identity['stamp'] != source_stamp or identity['command'] != source_command
                or [item['state_id'] for item in identity['body']] != body_ids
                or [item['state_id'] for item in identity['goals']] != goal_ids):
            raise ValueError('Captured K anchor and canonical candidate source differ.')
        if (parameter.get('selected_K') != action['selected_K']
                or parameter.get('density_id') != action['estimated_density_id']
                or parameter.get('estimated_packet_id') != action['estimated_packet_id']
                or reference.get('estimated_packet_id') != action['estimated_packet_id']):
            raise ValueError('Captured K values or density/packet provenance differ.')
        theta = np.asarray(action['auxiliary_equilibrium'], dtype=float)
        stiffness = np.asarray(action['selected_K'], dtype=float)
        if (theta.shape != (self.robot.model.nq,) or not np.all(np.isfinite(theta))
                or stiffness.shape != theta.shape or not np.all(np.isfinite(stiffness))):
            raise ValueError('Captured auxiliary equilibrium/K is not a finite model input.')
        if any(item.state.stamp != source_stamp or list(item.state.command) != source_command
               for item in body):
            raise ValueError('Canonical body components have different commands/timestamps.')

        paths = dict(result.link_paths)
        paths['hand'] = result.hand_path
        paths['camera'] = result.camera_path
        edge_ids = tuple(dict.fromkeys(edge_id for path in paths.values() for edge_id, _ in path))
        # Do not reuse a mutable Pinocchio placement left by a different FD
        # trial. Reevaluate FK once at the immutable already-solved anchor.
        self.robot._fk_update(theta)
        local_poses = {}
        for edge_id in edge_ids:
            joint = self.joints[edge_id]
            parent = self.robot.get_frame_id(joint.find('parent').get('link'))
            child = self.robot.get_frame_id(edge_id)
            pose = self.robot.data.oMf[parent].inverse()*self.robot.data.oMf[child]
            local_poses[edge_id] = (np.array(pose.translation, copy=True),
                                    np.array(pose.rotation, copy=True))

        components = [self._component(item, paths, edge_ids, local_poses, result.root_frame)
                      for item in body]
        global_queries = [query for component in components
                          for query in component['global_links'].values()]
        return dict(status='available' if all(item['status'] == 'available' for item in components) else 'partial',
            reason=None, source_stamp=source_stamp, source_command=source_command,
            source_prior_state_id=result.prior.state_id, source_version=result.prior.version,
            body_source_state_ids=body_ids, goal_source_state_ids=goal_ids,
            observation_ids_sha256=identity.get('observation_ids_sha256'),
            observation_id_count=identity.get('observation_id_count'),
            original_sensor_event=identity.get('original_sensor_event'),
            action_ledger_sha256=identity.get('action_ledger_sha256'),
            parameter_packet_id=parameter['estimated_packet_id'],
            action_packet_id=reference.get('packet_id'), density_id=parameter['density_id'],
            selected_K=stiffness.tolist(), auxiliary_equilibrium=theta.tolist(),
            K_estimated_source_stamp=action['estimated_source_stamp'],
            K_source_age_s=source_stamp-action['estimated_source_stamp'],
            FK_evaluations=1, components=components,
            summary=dict(maximum_global_modal_position_residual_m=max(
                query['modal_position_residual_m'] for query in global_queries),
                maximum_global_corrected_mean_position_residual_m=max(
                query['corrected_mean_position_residual_m'] for query in global_queries),
                maximum_global_modal_angular_residual_rad=max(
                query['modal_angular_residual_rad'] for query in global_queries),
                hand_modal_position_residual_m=[component['global_links']['hand']['modal_position_residual_m']
                    for component in components],
                hand_corrected_mean_position_residual_m=[component['global_links']['hand']['corrected_mean_position_residual_m']
                    for component in components]),
            corrected_orientation_description='native core intrinsic local orientation-mean coordinate; no global mean rotation is asserted')

    @staticmethod
    def _compose(path, local_poses):
        translation, rotation = np.zeros(3), np.eye(3)
        for edge_id, inverse in path:
            local_t, local_r = local_poses[edge_id]
            if inverse:
                local_t, local_r = -local_r.T @ local_t, local_r.T
            translation = translation+rotation @ local_t
            rotation = rotation @ local_r
        return translation, rotation

    def _component(self, item, paths, edge_ids, local_poses, frame):
        state = item.state
        result = dict(label=item.label, weight=item.weight,
            log_weight=component_log_weight(item), state_id=state.state_id,
            component_id=state.component_id, version=state.version,
            stamp=state.stamp, command=list(state.command),
            original_joint_covariance_sha256=_array_hash(state.covariance),
            local_links={}, global_links={}, status='available')
        by_id = {edge.edge_id: (i, edge) for i, edge in enumerate(state.transforms)}
        indices, residual = [], []
        for edge_id in edge_ids:
            index, edge = by_id[edge_id]
            predicted_t, predicted_r = local_poses[edge_id]
            coordinates = np.r_[predicted_t-edge.translation,
                Rotation.from_matrix(quat_to_rotmat(edge.rotation_wxyz).T @ predicted_r).as_rotvec()]
            indices.extend(range(6*index, 6*index+6))
            residual.extend(coordinates-state.mean[6*index:6*index+6])
            result['local_links'][edge_id] = self._query_record(
                state, ((edge_id, False),), predicted_t, predicted_r, edge_id)
        indices = np.asarray(indices, dtype=int)
        covariance = state.covariance[np.ix_(indices, indices)]
        result['body_joint'] = _support_diagnostic(np.asarray(residual), covariance)
        result['body_joint']['edge_ids'] = list(edge_ids)
        result['body_joint']['source_coordinate_indices'] = indices.tolist()
        # This is a marginal over all body/attachment variables. Environment
        # rows are integrated out, with all body-to-body cross blocks retained.
        result['body_joint']['environment_marginalized'] = True
        for name, path in paths.items():
            predicted_t, predicted_r = self._compose(path, local_poses)
            result['global_links'][name] = self._query_record(
                state, path, predicted_t, predicted_r, frame)
        if result['body_joint']['status'] != 'available':
            result['status'] = 'partial'
        return result

    @staticmethod
    def _query_record(state, path, predicted_t, predicted_r, frame):
        query = pose_query(state, path, frame=frame, query_id='K_body_consistency')
        angular = Rotation.from_matrix(quat_to_rotmat(query.rotation_wxyz).T @ predicted_r).as_rotvec()
        residual = np.r_[predicted_t-query.mean[:3], angular-query.mean[3:]]
        return dict(source_state_id=query.state_id, path=[list(item) for item in path],
            predicted_translation=predicted_t.tolist(),
            predicted_rotation_wxyz=rotmat_to_quat(predicted_r).tolist(),
            modal_translation=query.mode[:3].tolist(),
            modal_rotation_wxyz=query.rotation_wxyz.tolist(),
            corrected_mean_translation=query.mean[:3].tolist(),
            corrected_orientation_local_mean=query.mean[3:].tolist(),
            modal_position_residual_m=float(np.linalg.norm(predicted_t-query.mode[:3])),
            corrected_mean_position_residual_m=float(np.linalg.norm(residual[:3])),
            modal_angular_residual_rad=float(np.linalg.norm(angular)),
            corrected_orientation_chart_residual_rad=float(np.linalg.norm(residual[3:])),
            covariance=query.covariance.tolist(),
            corrected_chart_residual=residual.tolist(),
            propagated_support=_support_diagnostic(residual, query.covariance))
