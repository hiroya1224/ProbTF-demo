"""Physical robots/scene are references; candidate clouds use only core queries."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from .display import goal_pose_records
from .physical_occlusion import marker_occluder_settings

from probtf_switch_press_demo.ros_display import Display as SwitchDisplay
from probtf_switch_press_demo.scene import apply


def transform_message(parent, child, pose, stamp):
    message = TransformStamped()
    message.header.frame_id, message.header.stamp = parent, stamp
    message.child_frame_id = child
    message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = pose[:3, 3]
    quaternion = Rotation.from_matrix(pose[:3, :3].copy()).as_quat()
    (message.transform.rotation.x, message.transform.rotation.y,
     message.transform.rotation.z, message.transform.rotation.w) = quaternion
    return message


class MotionDisplay:
    # Reuse the original switch's geometry drawing and AprilTag bitmap.
    marker = SwitchDisplay.marker
    points = staticmethod(SwitchDisplay.points)

    def _board_markers(self):
        """Draw the same native bitmaps and known tile offsets as the camera."""
        camera = self.scene.camera
        board = self.scene.marker.copy()
        board[:3, 3] = apply(self.scene.marker, [0., 0., .0005])
        output = [self.marker('marker_board', Marker.CUBE, (1., 1., 1., 1.),
            (*camera.board_size, .001), pose=board)]
        for tile in camera.layout:
            texture = camera.marker_textures[tile.marker_id]
            triangles = []
            module = texture.shape[0]/10
            for row in range(8):
                for column in range(8):
                    if texture[int((row+1.5)*module), int((column+1.5)*module)] > 128:
                        continue
                    x0, x1 = (column/8-.5)*tile.size_m, ((column+1)/8-.5)*tile.size_m
                    y0, y1 = (.5-(row+1)/8)*tile.size_m, (.5-row/8)*tile.size_m
                    triangles.extend(((x0,y0,.0015),(x1,y0,.0015),(x1,y1,.0015),
                                      (x0,y0,.0015),(x1,y1,.0015),(x0,y1,.0015)))
            local = np.asarray(triangles)+np.asarray(tile.center_to_marker_translation)
            marker = self.marker('marker_board/tag_'+str(tile.marker_id),
                Marker.TRIANGLE_LIST, (.01,.01,.01,1.), (1.,1.,1.))
            output.append(self.points(marker, apply(self.scene.marker, local)))
        return output

    @staticmethod
    def _reaching_phase_label(trajectory_row, execution=None):
        """Label the active control plan, separately from search overrides.

        The asynchronous physical row and published execution record carry
        the same adopted plan provenance. Do not infer its phase from the
        current image, a future control result, or the physical tip distance.
        Legacy recordings without phase metadata retain their old display.
        """
        keys = ('reaching_phase', 'reaching_phase_source')
        recorded = any(key in trajectory_row for key in keys)
        reported = execution is not None and any(key in execution for key in keys)
        if not recorded and not reported:
            return None
        if not all(key in trajectory_row for key in keys):
            raise ValueError('Physical reaching phase requires its adopted control source.')
        phase = trajectory_row['reaching_phase']
        labels = {
            'visual_approach': 'APPROACH | CENTER MARKER',
            'indet_ik': 'IK',
            'waiting_for_control': 'WAITING FOR CONTROL',
            'waiting_for_goal': 'WAITING FOR OBSERVED GOAL',
            'running_gradient': 'RUNNING GRADIENT',
            'simple_reference': 'FINITE POSITION STEP',
        }
        if phase not in labels:
            raise ValueError('Unknown physical reaching phase: '+str(phase))
        source = trajectory_row['reaching_phase_source']
        if phase in ('waiting_for_control', 'waiting_for_goal'):
            if source is not None:
                raise ValueError('Waiting for control cannot claim an adopted control source.')
        elif (not isinstance(source, dict)
              or not all(key in source for key in ('source_stamp', 'source_command',
                  'source_version', 'body_source_state_ids', 'goal_source_state_ids'))):
            raise ValueError('A reaching phase requires the full adopted control source.')
        if execution is not None and (reported or execution.get('execution_mode') in ('async', 'simple')):
            if (not all(key in execution for key in keys)
                    or any(execution[key] != trajectory_row[key] for key in keys)):
                raise AssertionError('Physical and execution reaching phase/control source differ.')
        return labels[phase]

    @staticmethod
    def _color_action_path(marker, trajectory):
        """Color each executed interval, without recoloring its history.

        The arriving trajectory row records the command interval that produced
        it.  This differs from its next action after a new image is processed:
        a final search move may already have reacquired the marker.  Separate
        line segments keep a search/reaching boundary from blending colors.
        These points remain the physical reference path, not a candidate law.
        """
        if len(marker.points) != len(trajectory):
            raise AssertionError('Physical path points and recorded command intervals differ.')
        points, colors = [], []
        modes = {'initial_hold', 'waiting_for_goal', 'reaching', 'braking',
                 'settling', 'searching', 'search_exhausted'}
        for start, end, row in zip(marker.points, marker.points[1:], trajectory[1:]):
            mode = row.get('command_interval_mode', 'reaching')
            if mode not in modes:
                raise ValueError('Unknown physical path command interval mode: '+str(mode))
            rgba = (1., .45, 0., 1.) if mode == 'searching' else (.02, .65, .75, 1.)
            points.extend((start, end))
            colors.extend((ColorRGBA(*rgba), ColorRGBA(*rgba)))
        marker.type, marker.points, marker.colors = Marker.LINE_LIST, points, colors

    def __init__(self, scene, case, sensor_names):
        self.scene, self.case, self.sensor_names = scene, case, tuple(sensor_names)
        self.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
        self.marker_id = 0
        rospy.set_param('/observation_fusion/robot_description', scene.robot_description)
        self.transforms = TransformBroadcaster()
        self.static = StaticTransformBroadcaster()
        stamp = rospy.Time.now()
        # The original setting explicitly gives the base placement in world.
        self.static.sendTransform([
            transform_message('world', 'base_link', scene.world_base, stamp),
            transform_message('base_link', 'rigid/base_link', np.eye(4), stamp),
        ])
        self.publisher = rospy.Publisher('/observation_fusion/scene', MarkerArray,
                                         queue_size=1, latch=True)
        self.goal_tf_sources = rospy.Publisher('/observation_fusion/goal_tf_sources',
                                               String, queue_size=1, latch=True)

    @staticmethod
    def stiffness_marker(snapshot, record):
        """Display the updater's numbers only for this exact body snapshot.

        The auxiliary stiffness model never supplies a new body distribution
        to this adapter. All posterior component IDs, weights and versions are
        checked against the actual core queries used for the candidate clouds.
        """
        if (float(record['source_stamp']) != snapshot.stamp
                or tuple(record['source_command']) != tuple(snapshot.command)
                or sorted(record['observation_ids']) != sorted(snapshot.observation_ids)):
            raise AssertionError('Stiffness update and displayed body have different observation/command times.')
        components = {}
        for point in snapshot.points:
            query = point.query
            if (query.stamp != snapshot.stamp or tuple(query.command) != tuple(snapshot.command)
                    or query.frame != snapshot.frame_id):
                raise AssertionError('A body query belongs to another stiffness source snapshot.')
            # Goal-mixture queries share the internal body's marginal, but
            # carry their own observed marker conditionals and source IDs.
            if point.frame_label in ('marker', 'switch_goal'):
                continue
            source = (query.state_id, query.version, point.weight)
            if point.component_id in components and components[point.component_id] != source:
                raise AssertionError('A displayed component has inconsistent stiffness provenance.')
            components[point.component_id] = source
        expected = tuple((label, values[0], values[1], values[2])
                         for label, values in components.items())
        labels = record['body_posterior_labels']
        ids = record['body_posterior_state_ids']
        versions = record['body_posterior_versions']
        weights = record['body_posterior_weights']
        if not len(labels) == len(ids) == len(versions) == len(weights) == len(expected):
            raise AssertionError('Stiffness source lost a displayed body mixture component.')
        actual = tuple(zip(labels, ids, versions, weights))
        if actual != expected or versions[0] != snapshot.version:
            raise AssertionError('Stiffness source IDs, versions or weights differ from the displayed body.')
        if 'body_posterior_log_weights' in record:
            logs = {}
            for point in snapshot.points:
                if point.frame_label in ('marker', 'switch_goal'):
                    continue
                if point.component_id in logs and logs[point.component_id] != point.log_weight:
                    raise AssertionError('A displayed component has inconsistent log mass.')
                logs[point.component_id] = point.log_weight
            if tuple(logs.values()) != tuple(record['body_posterior_log_weights']):
                raise AssertionError('Stiffness source log masses differ from the displayed body.')
        if 'factor_ids' in record['prior']:
            return MotionDisplay._density_stiffness_marker(snapshot, record, versions)
        failed = record['status'].startswith('failed')
        if failed:
            if (record['posterior'] is not None or record.get('prior_retained') is not True
                    or record.get('retained_prior') != record['prior']
                    or record.get('information_rank') is not None
                    or record.get('information_eigenvalues') is not None
                    or record.get('update_applied') is True):
                raise ValueError('A failed stiffness update must retain its exact prior without a posterior or information rank.')
            displayed = record['retained_prior']
        else:
            if record['posterior'] is None:
                raise ValueError('A successful stiffness display requires its actual posterior.')
            displayed = record['posterior']
        prior = np.asarray(record['prior']['K_median'], dtype=float)
        posterior = np.asarray(displayed['K_median'], dtype=float)
        covariance = np.asarray(displayed['P_est'], dtype=float)
        if (posterior.ndim != 1 or prior.shape != posterior.shape
                or covariance.shape != (posterior.size, posterior.size)
                or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(posterior))
                or not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0.0)):
            raise ValueError('Stiffness display requires finite updater values and nonnegative variances.')
        standard_deviation = np.sqrt(np.diag(covariance))
        values = lambda array: ', '.join('{:.4g}'.format(value) for value in array)
        marker = Marker()
        marker.header.frame_id = snapshot.frame_id
        marker.header.stamp = rospy.Time.from_sec(snapshot.stamp)
        marker.ns, marker.id = 'estimated_stiffness', 0
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x, marker.pose.position.z = .72, .70
        marker.scale.z = .023
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .10, .12, .19, 1.0
        if failed:
            reason = ' '.join(str(record.get('failure_reason', 'not supplied')).split())
            if len(reason) > 100:
                reason = reason[:97]+'...'
            marker.text = (
                'K UPDATE UNAVAILABLE; prior retained\n'
                'Retained prior K median [Nm/rad]: {}\n'
                'Retained sigma(log K): {}\n'
                'Information rank: unavailable\n'
                'Reason: {}\n'
                't={:.3f} | body version {}'
            ).format(values(prior), values(standard_deviation), reason,
                     snapshot.stamp, ','.join(str(version) for version in versions))
        else:
            marker.text = (
                'Effective stiffness | auxiliary model\n'
                'Prior K median [Nm/rad]: {}\n'
                'Posterior K median: {}\n'
                'sigma(log K): {}\n'
                'Information rank {}/{} | {}\n'
                't={:.3f} | body version {}'
            ).format(values(prior), values(posterior), values(standard_deviation),
                     record['information_rank'], posterior.size, record['status'],
                     snapshot.stamp, ','.join(str(version) for version in versions))
        return marker

    @staticmethod
    def _density_stiffness_marker(snapshot, record, versions):
        """Distinguish the full retained density from a local proposal fit."""
        posterior = record['posterior']
        actual_density = posterior if posterior is not None else record.get('retained_prior')
        if actual_density is not None and actual_density.get('representation') == 'single normalized full-covariance Gaussian in log K':
            from .simple_display import simple_stiffness_marker
            return simple_stiffness_marker(snapshot, record, versions)
        if posterior is None:
            if (record.get('prior_retained') is not True
                    or record.get('retained_prior') != record['prior']
                    or record.get('update_applied') is True):
                raise ValueError('An undefined stiffness density must retain its exact prior.')
            density = record['retained_prior']
            title = ('K density retained; command prediction'
                     if record['status'] == 'prediction_with_retained_parameters'
                     else 'K density update unavailable; prior retained')
        else:
            density = posterior
            title = 'K density factors | normalization unevaluated'
        if (density.get('global_moments') is not None
                or density.get('normalization_constant') is not None):
            raise ValueError('The factor density HUD requires its actual normalization/moment convention.')
        prior_ids = record['prior']['factor_ids']
        ids = density['factor_ids']
        if ids[:len(prior_ids)] != prior_ids:
            raise AssertionError('Stiffness factor memory lost its prior factors.')
        proposal = record.get('proposal_approximation')
        if proposal is None:
            estimated_proposal = record.get('command_proposal')
            command_proposal = record.get('action_command_proposal',
                record.get('execution_command_proposal', estimated_proposal))
            if command_proposal is not None:
                values = np.asarray(command_proposal['selected_K'], dtype=float)
                if values.ndim != 1 or not np.all(np.isfinite(values)):
                    raise ValueError('The auxiliary command proposal requires its actual finite K values.')
                proposal_line = 'Numerical proposal K [Nm/rad]: '+', '.join('{:.4g}'.format(value) for value in values)
                detail = 'Proposal only; global moments not evaluated'
                correction = record.get('command_correction')
                if correction is not None:
                    if (correction['density_id'] != density['density_id']
                            or correction['source_stamp'] != snapshot.stamp
                            or tuple(correction['source_command']) != tuple(snapshot.command)
                            or correction['body_posterior_state_ids'] != record['body_posterior_state_ids']
                            or not np.array_equal(correction['selected_K'], values)):
                        raise ValueError('The displayed IK correction belongs to a different stiffness/body update.')
                    proposal_line = 'IK correction K [Nm/rad]: '+', '.join('{:.4g}'.format(value) for value in values)
                    detail = 'Equilibrium command action; global moments not evaluated'
                execution = record.get('execution_stiffness')
                action = record.get('action_stiffness')
                if action is not None:
                    if (execution is not None or estimated_proposal is None
                            or correction is None
                            or action['estimated_density_id'] != density['density_id']
                            or action['estimated_source_stamp'] != estimated_proposal['source_stamp']
                            or not np.array_equal(action['selected_K'], estimated_proposal['selected_K'])
                            or not np.array_equal(action['selected_K'], values)
                            or correction.get('action_stiffness') != action
                            or correction['captured_parameter_packet'].get('action_stiffness') != action
                            or command_proposal['source_stamp'] != snapshot.stamp
                            or tuple(command_proposal['source_command']) != tuple(snapshot.command)
                            or command_proposal['body_posterior_state_ids'] != record['body_posterior_state_ids']):
                        raise ValueError('Displayed estimated K and direct IK K_est must retain the same source and action.')
                    estimate_text = ', '.join('{:.4g}'.format(value) for value in estimated_proposal['selected_K'])
                    proposal_line = 'Estimated K [Nm/rad]: '+estimate_text+'\nIK K_est [Nm/rad]: '+estimate_text
                    detail = 'Direct estimated representative; captured anchor action; density unchanged'
                if execution is not None:
                    if (estimated_proposal is None
                            or not np.array_equal(execution['estimated_K'], estimated_proposal['selected_K'])
                            or not np.array_equal(execution['execution_K'], values)
                            or correction is None
                            or correction.get('execution_stiffness') != execution
                            or correction['captured_parameter_packet'].get('execution_stiffness') != execution):
                        raise ValueError('Displayed estimated/execution K must match the same IK action and source.')
                    estimate_text = ', '.join('{:.4g}'.format(value) for value in execution['estimated_K'])
                    execution_text = ', '.join('{:.4g}'.format(value) for value in values)
                    proposal_line = 'Estimated K [Nm/rad]: '+estimate_text+'\nExecution IK K [Nm/rad]: '+execution_text
                    detail = 'log K damping: tau={:.3g}s, cap={:.3g}/motion tick; density unchanged'.format(
                        execution['kp_exec_tau'], execution['max_log_kp_exec_step'])
            else:
                proposal_line = 'No numerical proposal; global moments not evaluated'
                reason = record.get('proposal_failure_reason', record.get('failure_reason', record.get('proposal_status', 'not computed')))
                detail = ' '.join(str(reason).split())
                if len(detail) > 100:
                    detail = detail[:97]+'...'
        else:
            median = np.asarray(proposal['K_median'], dtype=float)
            covariance = np.asarray(proposal['P_est'], dtype=float)
            if (median.ndim != 1 or covariance.shape != (median.size, median.size)
                    or not np.all(np.isfinite(median)) or not np.all(np.isfinite(covariance))
                    or np.any(np.diag(covariance) < 0)):
                raise ValueError('A local stiffness proposal requires its actual finite log-K parameters.')
            values = lambda array: ', '.join('{:.4g}'.format(value) for value in array)
            proposal_line = 'Local proposal K median [Nm/rad]: '+values(median)
            detail = 'Proposal sigma(log K): '+values(np.sqrt(np.diag(covariance)))
        marker = Marker()
        marker.header.frame_id = snapshot.frame_id
        marker.header.stamp = rospy.Time.from_sec(snapshot.stamp)
        marker.ns, marker.id = 'estimated_stiffness', 0
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.orientation.w = 1.
        marker.pose.position.x, marker.pose.position.z = .72, .70
        marker.scale.z = .023
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .10, .12, .19, 1.
        marker.text = ('{}\nFactors: {} -> {} | {}\n{}\n{}\n'
                       'Density {}\nt={:.3f} | body version {}').format(
            title, len(prior_ids), len(ids), record['status'], proposal_line, detail,
            density['density_id'][:16], snapshot.stamp, ','.join(str(version) for version in versions))
        return marker

    def publish(self, frame, frame_index, snapshot, inference_index, *, stiffness_record=None,
                execution=None, publish_scene=True):
        # Reject a stale or unrelated auxiliary update before publishing any
        # frame. The supplied record is the updater's own object, not a second
        # estimate reconstructed from the physical robot or its stiffness.
        if execution is not None and execution.get('execution_mode') in ('async', 'simple'):
            if (execution['physical_stamp'] != frame.stamp
                    or tuple(execution['command']) != tuple(frame.command)
                    or execution['displayed_prediction_stamp'] != snapshot.stamp
                    or tuple(execution['displayed_prediction_command']) != tuple(snapshot.command)
                    or execution['displayed_prediction_age_s'] != frame.stamp-snapshot.stamp
                    or snapshot.stamp > frame.stamp):
                raise AssertionError('Async physical and predicted source clocks are inconsistent.')
        elif stiffness_record is not None and (frame.stamp != snapshot.stamp
                or tuple(frame.command) != tuple(snapshot.command)):
            raise AssertionError('Physical frame and stiffness/body snapshot have different command times.')
        phase_label = (self._reaching_phase_label(self.scene.trajectory[frame_index], execution)
                       if publish_scene else None)
        stiffness = (None if stiffness_record is None else
                     self.stiffness_marker(snapshot, stiffness_record))
        stamp = rospy.Time.now()
        messages = []
        for prefix, poses in (('', frame.link_poses), ('rigid/', frame.rigid_link_poses)):
            for name, pose in poses.items():
                if name != 'base_link':
                    messages.append(transform_message('base_link', prefix + name, pose, stamp))
        # A TF is one component's explicitly queried modal pose, not the
        # complete mixture or an extra goal estimate. Convert the simulation
        # source clock to the physical reference's ROS display clock, keeping
        # the prediction age. Raw simulation stamps restart on replay; using
        # them directly in /tf would leave old data in the RViz TF buffer.
        # Publish both clocks and their mapping, once per immutable snapshot.
        goal_poses = goal_pose_records(snapshot) if hasattr(snapshot, 'points') else ()
        goal_sources = tuple((item['source_state_id'], item['source_stamp'],
                              item['child_frame_id']) for item in goal_poses)
        if goal_poses and goal_sources != getattr(self, '_last_goal_tf_sources', None):
            source_age = frame.stamp-snapshot.stamp
            tf_stamp = stamp-rospy.Duration.from_sec(source_age)
            for item in goal_poses:
                transform = TransformStamped()
                transform.header.frame_id = item['source_frame']
                transform.header.stamp = tf_stamp
                transform.child_frame_id = item['child_frame_id']
                (transform.transform.translation.x, transform.transform.translation.y,
                 transform.transform.translation.z) = item['translation']
                w, x, y, z = item['rotation_wxyz']
                (transform.transform.rotation.w, transform.transform.rotation.x,
                 transform.transform.rotation.y, transform.transform.rotation.z) = (w, x, y, z)
                messages.append(transform)
            self.goal_tf_sources.publish(String(data=json.dumps(dict(
                source_stamp=snapshot.stamp,
                physical_source_stamp=frame.stamp,
                render_stamp_ros=stamp.to_sec(),
                tf_stamp_ros=tf_stamp.to_sec(),
                tf_stamp_ros_parts=dict(secs=tf_stamp.secs, nsecs=tf_stamp.nsecs),
                source_age_s=source_age,
                clock_mapping='render_ros_stamp - (physical_sim_stamp - candidate_sim_stamp)',
                goal_poses=goal_poses), allow_nan=False)))
            self._last_goal_tf_sources = goal_sources
        self.transforms.sendTransform(messages)
        if not publish_scene:
            return
        self.trial = SimpleNamespace(
            config=self.scene.config, marker=self.scene.marker, goal=self.scene.goal,
            status='reached' if frame.reached else 'running',
            monitor=SimpleNamespace(reached=frame.reached, collided=frame.collided),
            plant=self.scene.plant, pose=frame.joints, tip_offset=self.scene.tip_offset,
            trajectory=self.scene.trajectory[:frame_index+1], phase='closed_loop', cycle=0,
            ik_iteration=0,
        )
        output = SwitchDisplay.scene_markers(self)
        if self.scene.config.get('marker_board', {}).get('enabled', False):
            # Only the physical tag artwork changes. Candidate distributions,
            # robot/obstacle geometry, trajectories and all display sizes keep
            # their existing sources and configured values.
            output.markers = [marker for marker in output.markers if marker.ns != 'tag']
            output.markers.extend(self._board_markers())
        occluder = marker_occluder_settings(self.scene.config.get('camera', {}))
        if occluder['enabled']:
            pose = self.scene.marker.copy()
            pose[:3, 3] = apply(self.scene.marker, occluder['center'])
            gray = occluder['intensity']/255.
            output.markers.append(self.marker('marker_occluder', Marker.CUBE,
                (gray, gray, gray, 1.), (*occluder['size'], occluder['thickness']), pose=pose))
        for marker in output.markers:
            if marker.ns == 'actual_path':
                self._color_action_path(marker, self.trial.trajectory)
            if marker.ns == 'status':
                if execution is not None and execution['execution_mode'] == 'async':
                    rate = lambda key: ('measuring' if execution[key] is None else
                                        '{:.1f} Hz'.format(execution[key]))
                    worker_label=('Gradient' if execution.get('control_mode')=='running_gradient' else 'IK')
                    clock = ('Async motion {} | camera {}\n'+worker_label+' {} | inference {}\n'
                             'Observation age {:.2f}s | displayed prediction age {:.2f}s').format(
                        rate('actual_motion_hz'), rate('actual_camera_hz'),
                        rate('actual_control_hz'), rate('actual_estimate_hz'),
                        execution['observation_age_s'], execution['displayed_prediction_age_s'])
                    if execution['finished']:
                        clock = ('FINISHED: wall-clock motion | ' if execution['estimation_drained']
                                 else 'FINISHED: motion | draining queued observations\n')+clock
                    elif execution.get('startup_phase') == 'initial_prior':
                        clock = 'INITIAL PRIOR | BEFORE OBSERVATIONS\n'+clock
                elif execution is not None and execution['execution_mode'] == 'simple':
                    clock = 'Simple reference | motion {:.1f} Hz | camera {:.1f} Hz'.format(
                        execution['actual_motion_hz'], execution['actual_camera_hz'])
                    if execution['finished']:
                        clock = ('FINISHED | ' if execution['estimation_drained'] else 'FINISHED | draining observations | ')+clock
                    elif execution.get('startup_phase') == 'initial_prior':
                        clock = 'INITIAL PRIOR | BEFORE OBSERVATIONS\n'+clock
                elif execution is not None and execution['execution_mode'] == 'live':
                    hz = execution['actual_update_hz']
                    rate = 'measuring' if hz is None else '{:.2f} Hz'.format(hz)
                    clock = 'Live updates {} | simulation dt {:.3f} s'.format(rate, execution['simulated_step_s'])
                    if execution.get('experiment_finished'):
                        clock = 'FINISHED: final experiment frame | last update rate '+rate
                    if execution.get('paused_at_update_limit'):
                        clock = 'PAUSED: requested {} updates; experiment incomplete'.format(
                            execution['requested_update_limit'])
                    if execution.get('failure_reason'):
                        clock = 'LIVE STOPPED: '+execution['failure_reason'][:100]
                else:
                    clock = 'Prepared playback | 20 fps (not computation rate)'
                marker.text = (
                    '{} | {:.1f} / {:.1f} s\n'
                    '{}\n'
                    'Opaque: gravity-flexed | Translucent: rigid FK\n'
                    'RGB: estimated XYZ endpoint distributions\n'
                    'Actual tip {:.1f} mm | Rigid tip {:.1f} mm\n'
                    'Observations {} | estimate t={:.1f} s | frame {}'
                ).format(self.case, frame.time, self.scene.duration, clock,
                         1000*np.linalg.norm(frame.tip-self.scene.goal),
                         1000*np.linalg.norm(frame.rigid_tip-self.scene.goal),
                         len(snapshot.observation_ids), snapshot.stamp-1792080000., inference_index)
                if not frame.physics_diagnostics.get('collision_checking_enabled', True):
                    marker.text += '\nCollision checks: OFF | lamp: distance only'
                if phase_label is not None:
                    marker.text += '\nPhase: '+phase_label
                action = self.trial.trajectory[-1].get('action_mode')
                if action is not None:
                    label = {
                        'reaching': 'REACHING',
                        'braking': 'MARKER LOST | BRAKING',
                        'settling': 'MARKER SEARCH | SETTLING',
                        'searching': 'MARKER SEARCH',
                        'search_exhausted': 'MARKER SEARCH EXHAUSTED | HOLD',
                        'waiting_for_goal': 'WAITING FOR OBSERVED GOAL',
                    }.get(action)
                    if label is None:
                        raise ValueError('Unknown physical frame action mode: '+str(action))
                    prefix = ('Last action: ' if execution is not None
                              and (execution.get('experiment_finished') or execution.get('finished')) else 'Action: ')
                    marker.text += '\n'+prefix+label
        imu = self.marker('imu_mount', Marker.CUBE,
                          (.55,.12,.75,1.) if 'imu' in self.sensor_names else (.4,.4,.4,.5),
                          (.025,.025,.025), pose=frame.link_poses[self.scene.imu_target_link].copy())
        output.markers.append(imu)
        if stiffness is not None:
            output.markers.append(stiffness)
        self.publisher.publish(output)
