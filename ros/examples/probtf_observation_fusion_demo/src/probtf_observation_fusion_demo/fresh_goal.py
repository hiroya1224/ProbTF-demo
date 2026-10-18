"""Capture task goals from the new image-relative law and the current body.

The recursive body/map posterior is an input, never replaced here.  For the
task query only, the new image-relative law is coupled independently to
that posterior and composed through its camera path. It is either the native
single-marker mixture or the explicitly projected board-center observation.
This is an explicit
cut approximation: the same image can already have informed the body, so
the constructed task law is not an exact joint Bayesian posterior.
"""

import numpy as np
from scipy.special import logsumexp

from probtf.geometry import (
    quat_mul, quat_to_rotmat, quaternion_to_rotation_vector, rotmat_to_quat,
    rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, _right_jacobian, bingham_from_rotation_coordinates,
)
from .body_native import pose_query
from probtf.spherical_law.support import covariance_support

from .model import WeightedState
from .sensors import _marker_mode_chart, _rotation_log_right_jacobian_inverse


def _native_modes(detected):
    components = tuple(detected.record.distribution.components)
    logs = np.asarray(detected.log_masses, dtype=float)
    weights = np.asarray(detected.weights, dtype=float)
    if (not components or logs.shape != (len(components),)
            or weights.shape != logs.shape or not np.all(np.isfinite(logs))
            or not np.all(np.isfinite(weights)) or np.any(weights < 0.)):
        raise ValueError('Fresh goals require every finite native image mode and mass.')
    if len({item.component_id for item in components}) != len(components):
        raise ValueError('Fresh image component IDs must be unique.')
    log_weights = logs-float(logsumexp(logs))
    if (not np.isclose(np.sum(weights), 1., rtol=0., atol=1e-12)
            or not np.allclose(weights, np.exp(log_weights), rtol=1e-12, atol=1e-15)):
        raise ValueError('Native image weights disagree with their normalized log masses.')
    seeds = tuple(detected.seed_indices)
    if seeds and len(seeds) != len(components):
        raise ValueError('Native image seed IDs and modes differ in length.')
    output = []
    for index, component in enumerate(components):
        quaternion = component.orientation.mode_wxyz
        law, log_volume = _marker_mode_chart(component, quat_to_rotmat(quaternion))
        # This chart is at the component's own mode. Its rotation origin is
        # exactly zero; a computed roundoff residual is not another shift.
        chart_residual = np.asarray(law.mean[3:])
        if not np.all(np.isfinite(chart_residual)):
            raise ValueError('A native image mode has nonfinite rotation-chart coordinates.')
        output.append(dict(native_component_id=component.component_id,
            native_seed_index=None if not seeds else int(seeds[index]),
            native_log_mass=float(logs[index]), image_log_weight=float(log_weights[index]),
            weight=float(weights[index]), log_weight=float(log_weights[index]),
            translation=law.mean[:3].tolist(), rotation_wxyz=quaternion.tolist(),
            bingham=component.orientation.parameter_matrix().tolist(),
            native_translation_mean=component.translation.mean_at_reference.tolist(),
            native_translation_residual_covariance=component.translation.residual_covariance.tolist(),
            native_translation_rotation_coupling=component.translation.rotation_coupling.tolist(),
            image_only_mean=np.concatenate((law.mean[:3], np.zeros(3))).tolist(),
            image_only_covariance=law.covariance.tolist(),
            native_chart_log_volume=float(log_volume),
            numerical_chart_diagnostics=dict(
                rotation_origin_residual=chart_residual.tolist(),
                rotation_origin_residual_inf=float(np.max(np.abs(chart_residual))),
                legacy_rotation_origin_tolerance_met=bool(np.allclose(chart_residual, 0., rtol=0., atol=1e-12)),
                numerical_quality_is_acceptance_gate=False)))
    return output


def _query_record(query):
    return dict(query_id=query.query_id, state_id=query.state_id,
        component_id=query.component_id, frame=query.frame, stamp=query.stamp,
        command=list(query.command), version=query.version,
        rotation_wxyz=query.rotation_wxyz.tolist(),
        **{name: getattr(query, name).tolist() for name in (
            'mean', 'mode', 'correction', 'covariance', 'within', 'between',
            'jacobian', 'cross_covariance')})


def fresh_marker_goals(result, canonical_state, *, marker_id='marker'):
    """Return all fresh task modes sharing the bit-identical current body.

    The temporary marker slot denotes camera-relative image coordinates.
    The returned marker slot denotes their root-frame push-forward.  The
    retained internal marker remains unchanged in ``canonical_state``.
    """
    estimates = tuple(item for item in result.local_estimates
        if item.posterior is not None and item.detector_result is not None)
    if not estimates:
        return (), dict(applied=False, reason='no valid new marker observation',
            canonical_internal_state_id=canonical_state.state_id,
            monte_carlo_used=False, internal_state_replaced=False)
    if (marker_id not in canonical_state.edge_ids
            or tuple(result.marker_path) != ((marker_id, False),)
            or any(name == marker_id for name, _inverse in result.camera_path)):
        raise ValueError('Fresh goals require a root marker slot outside the camera path.')
    identities = {(item.posterior.observation_id, item.posterior.sensor_id)
                  for item in estimates}
    if len(identities) != 1:
        raise ValueError('Fresh goals require one shared new marker observation.')
    observation_id, sensor_id = next(iter(identities))
    detected = estimates[0].detector_result
    board_fusion = estimates[0].board_fusion
    is_board = board_fusion is not None
    native = _native_modes(detected)
    for item in estimates:
        query = item.posterior.query
        if (query.stamp != canonical_state.stamp
                or tuple(query.command) != canonical_state.command
                or item.detector_result.record.stamp != canonical_state.stamp):
            raise ValueError('Fresh image and canonical body have different capture metadata.')
        if (item.detector_result is not detected
                and _native_modes(item.detector_result) != native):
            raise ValueError('Keep/reset evaluations do not share the same image-only modes.')
        if item.board_fusion is not board_fusion and item.board_fusion != board_fusion:
            raise ValueError('Keep/reset evaluations do not share the board-center observation.')
    if is_board:
        central = board_fusion['central_distribution']
        central_components = central.get('components', (central,))
        if ([item['component_id'] for item in central_components]
                != [item['native_component_id'] for item in native]
                or central['stamp'] != detected.record.stamp):
            raise ValueError('Fresh board goals do not reference the fused central law.')
    raw_observations = [] if not is_board else board_fusion['raw_observations']

    marker_index = canonical_state.edge_ids.index(marker_id)
    environment = np.arange(6*marker_index, 6*marker_index+6)
    body = np.array([coordinate for index in range(len(canonical_state.transforms))
        if index != marker_index for coordinate in range(6*index, 6*index+6)], dtype=int)
    body_covariance = canonical_state.covariance[np.ix_(body, body)]
    camera_path = tuple(result.camera_path)
    composed_path = camera_path+((marker_id, False),)
    goals, records = [], []
    for mode in native:
        label = 'fresh_goal:{}:{}'.format(observation_id, mode['native_component_id'])
        covariance = np.zeros_like(canonical_state.covariance)
        covariance[np.ix_(body, body)] = body_covariance
        covariance[np.ix_(environment, environment)] = mode['image_only_covariance']
        transforms = list(canonical_state.transforms)
        transforms[marker_index] = BinghamTransform(marker_id, mode['translation'],
            mode['bingham'], np.asarray(mode['image_only_covariance'])[:3, :3])
        augmented = JointTransformState(tuple(transforms), covariance,
            version=canonical_state.version, stamp=canonical_state.stamp,
            command=canonical_state.command, component_id=label+':camera_relative')
        query = pose_query(augmented, composed_path,
            query_id=label+':root_marker', frame=result.root_frame)

        # The output pose has its modal right-rotation chart. Move only the
        # marker's coordinates to its corrected mean, preserving all shared
        # body cross blocks and the body's original Bingham parameters.
        marker_chart = np.eye(6)
        marker_chart[3:, 3:] = _right_jacobian(query.mean[3:])
        marker_covariance = marker_chart @ query.covariance @ marker_chart.T
        marker_covariance = .5*(marker_covariance+marker_covariance.T)
        cross = query.cross_covariance[body] @ marker_chart.T
        quaternion = quat_mul(query.rotation_wxyz,
                              rotation_vector_to_quaternion(query.mean[3:]))
        marker = BinghamTransform(marker_id, query.mean[:3],
            bingham_from_rotation_coordinates(quaternion, marker_covariance[3:, 3:]),
            marker_covariance[:3, :3])
        goal_covariance = canonical_state.covariance.copy()
        goal_covariance[np.ix_(body, environment)] = cross
        goal_covariance[np.ix_(environment, body)] = cross.T
        goal_covariance[np.ix_(environment, environment)] = marker_covariance
        covariance_support(goal_covariance, 'fresh root goal complete joint covariance')
        goal_transforms = list(canonical_state.transforms)
        goal_transforms[marker_index] = marker
        state = JointTransformState(tuple(goal_transforms), goal_covariance,
            version=canonical_state.version, stamp=canonical_state.stamp,
            command=canonical_state.command, component_id=label)
        relative_path = tuple((name, not inverse)
            for name, inverse in reversed(camera_path))+((marker_id, False),)
        closure = pose_query(state, relative_path,
            query_id=label+':camera_relative_closure', frame=detected.record.parent_frame_id)
        # Compare the re-queried pose in the native image mode's chart,
        # rather than comparing two different modal rotation origins.
        closure_rotation = (quat_to_rotmat(mode['rotation_wxyz']).T
            @ quat_to_rotmat(closure.rotation_wxyz)
            @ quat_to_rotmat(rotation_vector_to_quaternion(closure.mean[3:])))
        closure_mean = closure.mean.copy()
        closure_mean[3:] = quaternion_to_rotation_vector(rotmat_to_quat(closure_rotation))
        closure_chart = np.eye(6)
        closure_chart[3:, 3:] = (_rotation_log_right_jacobian_inverse(closure_mean[3:])
            @ _right_jacobian(closure.mean[3:]))
        closure_covariance = closure_chart @ closure.covariance @ closure_chart.T
        closure_covariance = .5*(closure_covariance+closure_covariance.T)
        closure_mean_error = closure_mean-np.asarray(mode['image_only_mean'])
        closure_covariance_error = closure_covariance-np.asarray(mode['image_only_covariance'])
        goals.append(WeightedState(label, mode['weight'], state, mode['log_weight']))
        records.append(dict(mode,
            goal_label=label, goal_source_state_id=state.state_id,
            augmented_joint_state=dict(state_id=augmented.state_id,
                component_id=augmented.component_id, version=augmented.version,
                stamp=augmented.stamp, command=list(augmented.command),
                edge_ids=list(augmented.edge_ids), marker_coordinate_indices=environment.tolist(),
                body_coordinate_indices=body.tolist(),
                covariance_rule='canonical P_BB, fresh image full P_CM, zero B/CM cross'),
            root_pose_query=_query_record(query),
            root_marker_chart_jacobian=marker_chart.tolist(),
            root_marker_covariance=marker_covariance.tolist(),
            body_to_root_marker_cross_covariance=cross.tolist(),
            root_marker_translation=marker.translation.tolist(),
            root_marker_rotation_wxyz=marker.rotation_wxyz.tolist(),
            root_marker_bingham=marker.bingham.tolist(),
            camera_relative_closure=dict(
                query=_query_record(closure),
                native_reference_rotation_wxyz=mode['rotation_wxyz'],
                native_chart_jacobian=closure_chart.tolist(),
                mean=closure_mean.tolist(), covariance=closure_covariance.tolist(),
                mean_error=closure_mean_error.tolist(),
                covariance_error=closure_covariance_error.tolist(),
                translation_mean_error_m=float(np.linalg.norm(closure_mean_error[:3])),
                rotation_mean_error_rad=float(np.linalg.norm(closure_mean_error[3:])),
                maximum_covariance_error=float(np.max(np.abs(closure_covariance_error))),
                comparison_order='measured residual: the direction-dependent S2 mean approximation is not closed under root-pose collapse and inverse composition; position mean error can be first order in covariance',
                exact_closure_claimed=False,
                root_forward_pose_mean_preserved=True,
                mean_recentered_to_force_inverse_closure=False)))
    return tuple(goals), dict(applied=True,
        construction=('fresh derived board-center camera-relative law composed through the captured canonical camera/body law'
            if is_board else 'fresh image-only camera-relative mixture composed through the captured canonical camera/body law'),
        observation_kind='fused_board_center' if is_board else 'individual_marker',
        marker_board_fusion=board_fusion,
        approximation='task-only independent fresh measurement cut, followed by existing first-order Bingham-derived pose push-forward and marker-only reexpression',
        exact_joint_bayesian_posterior_claimed=False,
        camera_relative_closure_assumed_exact=False,
        camera_relative_position_mean_closure_error_may_be_first_order=True,
        forward_pushforward_mean_altered_to_force_closure=False,
        fresh_measurement_independent_of_retained_internal_state_for_task_construction=True,
        image_already_used_in_body_update=True, observation_fused_again=False,
        canonical_internal_state_id=canonical_state.state_id,
        capture_stamp=canonical_state.stamp, capture_command=list(canonical_state.command),
        capture_version=canonical_state.version, root_frame=result.root_frame,
        observation_id=observation_id, sensor_id=sensor_id,
        image_stamp=detected.record.stamp,
        image_parent_frame=detected.record.parent_frame_id,
        image_child_frame=detected.record.child_frame_id,
        image_edge_id=detected.record.edge_id,
        sensor_parent_evaluation_count=len(estimates), native_component_count=len(native),
        input_component_count=len(native),
        central_component_count=len(native) if is_board else None,
        raw_marker_count=len(raw_observations) if is_board else 1,
        raw_native_component_count=(sum(item['native_component_count'] for item in raw_observations)
                                    if is_board else len(native)),
        component_record_semantics=('the legacy native_* keys below describe the derived central input, not a raw detector Hessian; raw native modes are retained in marker_board_fusion.raw_observations'
            if is_board else 'native_* keys describe the raw individual marker image modes'),
        goal_component_count=len(goals),
        mixture_semantics=('derived board-center components after image-only fusion; concentrated central law is moment-projected once, unresolved broad central laws retain every narrow branch; raw per-marker mixtures remain in the board record; keep/reset already projected into the common body'
            if is_board else 'one component per native image mode; keep/reset parent hypotheses have already been projected into the shared internal body, not multiplied into this fresh task mixture'),
        goal_weights_source=('derived board-center normalized weights and log masses after image-only fusion, with central projection only within the existing concentration domain; no retained-map prior overlap reweighting'
            if is_board else 'native detector float weights and normalized image log masses; no retained-map prior overlap reweighting'),
        old_internal_marker_used_for_goal_construction=False,
        internal_unknown_marker_and_all_cross_covariances_retained=True,
        internal_state_replaced=False, stiffness_memory_changed=False,
        camera_path=[list(item) for item in camera_path],
        composition_path=[list(item) for item in composed_path],
        source_sensor_prior_state_ids=[item.posterior.query.state_id for item in estimates],
        goal_source_state_ids=[item.state.state_id for item in goals],
        goal_source_labels=[item.label for item in goals],
        goal_source_weights=[item.weight for item in goals],
        goal_source_log_weights=[item.log_weight for item in goals],
        components=records, monte_carlo_used=False, physical_truth_received=False,
        component_pruning_used=False, covariance_regularization_added=False)
