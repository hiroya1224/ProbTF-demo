"""Far reaching by image centering on the existing joint candidate laws.

This is an explicitly different far-phase selection rule, not the gradient
of the complete Gaussian/Bingham reaching criterion.  Every goal mode is
queried separately.  Its local pinhole and hand-position models retain the
core mean and covariance; the command differential uses the core's analytic
modal action Jacobian, with the moment correction and covariance frozen for
this short local model.  No law is replaced by the control loss or its fit.
"""

from copy import deepcopy

import numpy as np

from probtf.spherical_law.first_order import pose_query

from .moving_control import _component_log_weight, control_components


def _query_record(query):
    return dict(query_id=query.query_id, source_state_id=query.state_id,
        source_component_id=query.component_id, source_frame=query.frame,
        source_stamp=query.stamp, source_command=list(query.command),
        source_version=query.version, rotation_wxyz=query.rotation_wxyz.tolist(),
        **{name: getattr(query, name).tolist() for name in (
            'mode', 'mean', 'correction', 'covariance', 'within', 'between',
            'jacobian', 'cross_covariance')})


class VisualApproach:
    """Center the marker before approaching, then latch ordinary near IK.

    ``propose`` does not execute, clip individual joints, or mutate a source
    distribution. Its target corresponds to one common scaled velocity.
    The caller must pass that velocity through the controller's existing
    velocity, acceleration, joint-limit and stopping-distance constraints.
    """

    def __init__(self, controller, settings=None):
        self.controller = controller
        settings = {} if settings is None else dict(settings)
        self.enabled = bool(settings.get('enabled', True))
        self.near_distance = float(settings.get('near_distance_m', .08))
        self.center_gain = float(settings.get('center_gain', 2.))
        self.approach_gain = float(settings.get('approach_gain', .8))
        self.center_tolerance = float(settings.get('center_tolerance', .10))
        self.center_stop = float(settings.get('center_stop', .15))
        self.center_weight = float(settings.get('center_weight', 25.))
        self.reach_scale = float(settings.get('reach_scale_m', .4))
        self.minimum_depth = float(settings.get('minimum_depth_m', .02))
        self.damping = float(settings.get('damping', controller.damping))
        values = (self.near_distance, self.center_gain, self.approach_gain,
                  self.center_tolerance, self.center_stop, self.center_weight,
                  self.reach_scale, self.minimum_depth, self.damping)
        if (not all(np.isfinite(value) and value > 0. for value in values)
                or self.center_stop <= self.center_tolerance):
            raise ValueError('Visual approach needs finite positive scales and an ordered centering gate.')
        self.near_latched = False

    def _sensitivity(self, state, command_jacobian):
        sensitivity = np.zeros((state.mean.size, command_jacobian.shape[1]))
        for edge_id, action in zip(self.controller.body_ids, self.controller.actions):
            index = state.edge_ids.index(edge_id)
            axis = np.asarray(action['axis'], dtype=float)
            norm = np.linalg.norm(axis)
            if not np.isfinite(norm) or norm <= 0.:
                raise ValueError('Visual approach requires a finite nonzero declared joint axis.')
            sensitivity[6*index+3:6*index+6] = np.outer(
                axis/norm, command_jacobian[int(action['joint_index'])])
        return sensitivity

    def propose(self, result, dt, *, command_map=None):
        """Return ``(target_command or None, serializable provenance)``.

        ``None`` delegates to ordinary near-phase IK. An invalid/behind-camera
        mode instead returns the current command: it must not silently select
        another image mode or start approaching without a usable image law.
        The marker-loss policy remains the caller's higher-priority decision.
        """
        dt = float(dt)
        command = np.asarray(result.prior.command, dtype=float)
        if not np.isfinite(dt) or dt <= 0.:
            raise ValueError('Visual approach requires a positive finite command interval.')
        if command.shape != self.controller.velocity.shape or not np.all(np.isfinite(command)):
            raise ValueError('Visual approach command differs from the configured joints.')
        record = dict(enabled=self.enabled, phase='near' if self.near_latched else 'far',
            near_latched=self.near_latched, near_distance_m=self.near_distance,
            selection_criterion='all-mode expected local image-centering and hand-position squared loss',
            full_gaussian_bingham_criterion_used_for_selection=False,
            full_gaussian_bingham_criterion_role='diagnostic in far phase; selection in near phase',
            linearization='core mean and full covariance; analytic modal joint-action Jacobian; moment correction and covariance frozen during this short local model',
            camera_model='normalized calibrated pinhole bearing (x/z, y/z)',
            field_of_view_guaranteed=False,
            field_of_view_limit='marker-origin centering, not a bound on every marker corner or every candidate realization',
            source_stamp=result.prior.stamp, source_command=command.tolist(),
            source_version=result.stages[-1].components[0].state.version,
            prior_source_version=result.prior.version, duration_s=dt,
            monte_carlo_used=False, physical_truth_received=False,
            source_laws_modified=False, mixture_collapsed=False,
            limits_applied_by='existing MovingController common command bounds',
            branches=[])
        memory = result.records.get('fixed_goal_memory', {})
        image_source = memory.get('image_relative_source', {})
        record['goal_acquisition'] = deepcopy({key: memory[key] for key in (
            'available', 'root_frame', 'observation_stamp', 'observation_command',
            'observation_version', 'observation_ids', 'observation_sensor_ids',
            'observation_source_state_ids', 'marker_to_switch') if key in memory})
        record['native_image_source'] = deepcopy({key: image_source[key] for key in (
            'observation_id', 'sensor_id', 'image_stamp', 'capture_stamp', 'capture_command',
            'capture_version', 'image_parent_frame', 'image_child_frame', 'image_edge_id',
            'canonical_internal_state_id') if key in image_source})
        record['native_image_components'] = [deepcopy({key: item[key] for key in (
            'native_component_id', 'native_seed_index', 'native_log_mass',
            'image_log_weight', 'weight', 'log_weight', 'image_only_mean',
            'image_only_covariance', 'bingham', 'goal_source_state_id') if key in item})
            for item in image_source.get('components', ())]
        if not self.enabled or self.near_latched:
            record['selection_reason'] = 'visual phase disabled' if not self.enabled else 'near phase already latched'
            return None, record

        internal, components = control_components(result)
        if not components:
            raise ValueError('Visual approach requires the existing complete goal mixture.')
        if command_map is not None:
            command_map.validate_result(result)
        derivative = (np.eye(command.size) if command_map is None
                      else np.asarray(command_map.jacobian(command), dtype=float))
        if derivative.shape != (command.size, command.size) or not np.all(np.isfinite(derivative)):
            raise ValueError('Visual approach needs the same finite command-action Jacobian.')
        source_version = internal[0].state.version
        record.update(source_version=source_version,
            body_source_state_ids=[item.state.state_id for item in internal],
            goal_source_state_ids=[item.state.state_id for item in components],
            goal_source_weights=[item.weight for item in components],
            goal_source_log_weights=[_component_log_weight(item) for item in components],
            command_jacobian=derivative.tolist(),
            command_action='declared right joint action' if command_map is None else 'same frozen estimated-K command action',
            stiffness_command_correction=None if command_map is None else command_map.record())
        camera_path = tuple((name, not inverse)
            for name, inverse in reversed(result.camera_path))+tuple(result.marker_path)
        reach_squared, bearing_squared, centering_loss = 0., 0., 0.
        center_metric = np.zeros((command.size, command.size))
        center_rhs = np.zeros(command.size)
        reach_metric = np.zeros_like(center_metric)
        reach_rhs = np.zeros_like(center_rhs)
        all_positive_depth = True
        for component in components:
            state = component.state
            if (state.stamp != result.prior.stamp or state.command != result.prior.command
                    or state.version != source_version):
                raise ValueError('Visual goal candidates have inconsistent current source metadata.')
            camera = pose_query(state, camera_path,
                query_id='visual:camera_relative_marker', frame='camera')
            hand = pose_query(state, self.controller.relative_path,
                query_id='visual:marker_relative_hand', frame='marker')
            sensitivity = self._sensitivity(state, derivative)
            camera_jacobian = camera.jacobian[:3] @ sensitivity
            hand_jacobian = hand.jacobian[:3] @ sensitivity
            error = hand.mean[:3]-self.controller.goal_offset
            squared = float(error @ error+np.trace(hand.covariance[:3, :3]))
            if squared < -1e-12:
                raise ValueError('A visual hand-goal squared distance became negative.')
            reach_squared += component.weight*max(0., squared)
            reach_metric += component.weight*(hand_jacobian.T @ hand_jacobian)/self.reach_scale**2
            reach_rhs -= component.weight*self.approach_gain*(hand_jacobian.T @ error)/self.reach_scale**2
            branch = dict(label=component.label, weight=float(component.weight),
                log_weight=_component_log_weight(component), source_state_id=state.state_id,
                camera_relative_marker=_query_record(camera),
                marker_relative_hand=_query_record(hand),
                camera_position_command_jacobian=camera_jacobian.tolist(),
                hand_position_command_jacobian=hand_jacobian.tolist(),
                hand_goal_squared_distance_expectation=squared,
                hand_goal_error_mean=error.tolist())
            x, y, z = camera.mean[:3]
            positive_depth = bool(z > self.minimum_depth)
            branch['positive_camera_depth'] = positive_depth
            all_positive_depth = all_positive_depth and positive_depth
            if positive_depth:
                bearing = np.array([x/z, y/z])
                projection = np.array([[1./z, 0., -x/z**2], [0., 1./z, -y/z**2]])
                covariance = projection @ camera.covariance[:3, :3] @ projection.T
                jacobian = projection @ camera_jacobian
                bearing_squared += component.weight*float(bearing @ bearing)
                branch_loss = float(bearing @ bearing+np.trace(covariance))
                centering_loss += component.weight*branch_loss
                center_metric += component.weight*(jacobian.T @ jacobian)
                center_rhs -= component.weight*self.center_gain*(jacobian.T @ bearing)
                branch.update(normalized_bearing_mean=bearing.tolist(),
                    normalized_bearing_covariance=covariance.tolist(),
                    normalized_bearing_command_jacobian=jacobian.tolist(),
                    normalized_bearing_squared_loss_expectation=branch_loss,
                    pinhole_linearization=projection.tolist())
            record['branches'].append(branch)

        distance = float(np.sqrt(reach_squared))
        bearing_rms = float(np.sqrt(bearing_squared))
        gate = float(np.clip((self.center_stop-bearing_rms)/(
            self.center_stop-self.center_tolerance), 0., 1.))
        record.update(hand_goal_rms_distance_m=distance,
            hand_goal_squared_distance_expectation=reach_squared,
            normalized_bearing_mean_rms=bearing_rms,
            centering_squared_loss_expectation=centering_loss,
            center_gain=self.center_gain, approach_gain=self.approach_gain,
            center_tolerance=self.center_tolerance, center_stop=self.center_stop,
            center_weight=self.center_weight, reach_scale_m=self.reach_scale,
            progression_gate=gate, all_modes_positive_depth=all_positive_depth)
        if distance <= self.near_distance:
            self.near_latched = True
            record.update(phase='near', near_latched=True,
                selection_reason='all-mode expected hand-goal RMS entered the near threshold')
            return None, record
        if not all_positive_depth:
            record.update(selection_reason='no progression: at least one goal mode has unusable camera depth',
                requested_velocity=np.zeros_like(command).tolist(), target_command=command.tolist())
            return command.copy(), record

        metric = (self.center_weight*center_metric+gate*reach_metric
                  +self.damping**2*np.eye(command.size))
        rhs = self.center_weight*center_rhs+gate*reach_rhs
        unconstrained = np.linalg.solve(metric, rhs)
        if not np.all(np.isfinite(unconstrained)):
            raise ValueError('Visual approach produced a nonfinite local velocity.')
        # Preserve the coupled image-centering direction when reducing speed.
        # Individual joint clipping and all remaining limits belong to the
        # existing common controller, which also knows previous velocity.
        peak = float(np.max(np.abs(unconstrained)))
        scale = min(1., self.controller.speed/peak) if peak else 1.
        velocity = scale*unconstrained
        target = command+dt*velocity
        record.update(selection_reason='center marker; permit positional approach only inside the centering gate',
            local_metric=metric.tolist(), local_rhs=rhs.tolist(),
            unscaled_velocity=unconstrained.tolist(), common_velocity_scale=scale,
            requested_velocity=velocity.tolist(), target_command=target.tolist())
        return target, record
