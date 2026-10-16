"""Physical robots/scene are references; candidate clouds use only core queries."""
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from probtf_switch_press_demo.ros_display import Display as SwitchDisplay


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
        if posterior is None:
            if (record.get('prior_retained') is not True
                    or record.get('retained_prior') != record['prior']
                    or record.get('update_applied') is True):
                raise ValueError('An undefined stiffness density must retain its exact prior.')
            density = record['retained_prior']
            title = 'K density update unavailable; prior retained'
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
            command_proposal = record.get('command_proposal')
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

    def publish(self, frame, frame_index, snapshot, inference_index, *, stiffness_record=None, execution=None):
        # Reject a stale or unrelated auxiliary update before publishing any
        # frame. The supplied record is the updater's own object, not a second
        # estimate reconstructed from the physical robot or its stiffness.
        if stiffness_record is not None and (frame.stamp != snapshot.stamp
                or tuple(frame.command) != tuple(snapshot.command)):
            raise AssertionError('Physical frame and stiffness/body snapshot have different command times.')
        stiffness = (None if stiffness_record is None else
                     self.stiffness_marker(snapshot, stiffness_record))
        stamp = rospy.Time.now()
        messages = []
        for prefix, poses in (('', frame.link_poses), ('rigid/', frame.rigid_link_poses)):
            for name, pose in poses.items():
                if name != 'base_link':
                    messages.append(transform_message('base_link', prefix + name, pose, stamp))
        self.transforms.sendTransform(messages)
        self.trial = SimpleNamespace(
            config=self.scene.config, marker=self.scene.marker, goal=self.scene.goal,
            status='reached' if frame.reached else 'running',
            monitor=SimpleNamespace(reached=frame.reached, collided=frame.collided),
            plant=self.scene.plant, pose=frame.joints, tip_offset=self.scene.tip_offset,
            trajectory=self.scene.trajectory[:frame_index+1], phase='closed_loop', cycle=0,
            ik_iteration=0,
        )
        output = SwitchDisplay.scene_markers(self)
        for marker in output.markers:
            if marker.ns == 'status':
                if execution is not None and execution['execution_mode'] == 'live':
                    hz = execution['actual_update_hz']
                    rate = 'measuring' if hz is None else '{:.2f} Hz'.format(hz)
                    clock = 'Live updates {} | simulation dt {:.3f} s'.format(rate, execution['simulated_step_s'])
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
        imu = self.marker('imu_mount', Marker.CUBE,
                          (.55,.12,.75,1.) if 'imu' in self.sensor_names else (.4,.4,.4,.5),
                          (.025,.025,.025), pose=frame.link_poses[self.scene.imu_target_link].copy())
        output.markers.append(imu)
        if stiffness is not None:
            output.markers.append(stiffness)
        self.publisher.publish(output)
