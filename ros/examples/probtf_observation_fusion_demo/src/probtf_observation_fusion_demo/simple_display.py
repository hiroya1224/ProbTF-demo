"""HUD for the actual normalized single-Gaussian stiffness reference."""

import numpy as np
import rospy
from visualization_msgs.msg import Marker


def simple_stiffness_marker(snapshot, record, versions):
    """Render the retained updater parameters with the unchanged HUD style.

    MotionDisplay.stiffness_marker owns the complete snapshot/body-query
    validation before this dispatch. This helper checks the Gaussian and the
    numerical K/action provenance; it never estimates a replacement law.
    """
    density = record.get('posterior')
    if density is None:
        if (record.get('prior_retained') is not True
                or record.get('retained_prior') != record.get('prior')
                or record.get('update_applied') is True):
            raise ValueError('A retained simple K display must preserve its actual prior.')
        density = record['retained_prior']
    if (density.get('representation') != 'single normalized full-covariance Gaussian in log K'
            or density.get('normalization_constant') != 1.
            or density.get('bayes_factor_surprise_enabled') is not False
            or density.get('evidence_computed') is not False):
        raise ValueError('The simple K HUD needs its actual normalized Gaussian convention without BFS/evidence.')
    mean = np.asarray(density['mean'], dtype=float)
    covariance = np.asarray(density['covariance'], dtype=float)
    if (mean.shape != (6,) or covariance.shape != (6, 6)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))):
        raise ValueError('The simple stiffness display needs the retained finite full six-dimensional Gaussian.')
    # The covariance is the recorded updater value, without floor or jitter.
    np.linalg.cholesky(covariance)
    components = density.get('components', ())
    if (len(components) != 1 or components[0]['weight'] != 1.
            or components[0]['log_weight'] != 0.
            or not np.array_equal(components[0]['x_est'], mean)
            or not np.array_equal(components[0]['P_est'], covariance)):
        raise ValueError('The simple stiffness HUD parameters differ from the retained Gaussian component.')
    representative = np.exp(mean)
    if not np.all(np.isfinite(representative)) or np.any(representative <= 0.):
        raise ValueError('The displayed stiffness representative is not representable as finite positive K.')
    estimated = record.get('command_proposal')
    proposal = record.get('action_command_proposal',
        record.get('execution_command_proposal', estimated))
    if record.get('execution_stiffness') is not None:
        raise ValueError('The simple stiffness reference has no separately damped execution K.')
    if proposal is not None:
        if (proposal['density_id'] != density['density_id']
                or proposal['source_stamp'] != snapshot.stamp
                or tuple(proposal['source_command']) != tuple(snapshot.command)
                or proposal['source_prior_state_id'] != record['source_prior_state_id']
                or proposal['body_posterior_state_ids'] != record['body_posterior_state_ids']
                or not np.array_equal(proposal['selected_logK'], mean)
                or not np.array_equal(proposal['selected_K'], representative)):
            raise ValueError('The displayed simple K representative belongs to a different Gaussian/body source.')
    correction = record.get('command_correction')
    if correction is not None:
        if (proposal is None or correction['density_id'] != density['density_id']
                or correction['source_stamp'] != snapshot.stamp
                or tuple(correction['source_command']) != tuple(snapshot.command)
                or correction['body_posterior_state_ids'] != record['body_posterior_state_ids']
                or not np.array_equal(correction['selected_K'], representative)):
            raise ValueError('The displayed simple K command correction has a different source.')
    action = record.get('action_stiffness')
    if action is not None:
        if (estimated is None or correction is None
                or action['estimated_density_id'] != density['density_id']
                or estimated['density_id'] != density['density_id']
                or action['estimated_source_stamp'] != estimated['source_stamp']
                or not np.array_equal(estimated['selected_logK'], mean)
                or not np.array_equal(estimated['selected_K'], representative)
                or not np.array_equal(action['selected_K'], representative)
                or correction.get('action_stiffness') != action
                or correction.get('captured_parameter_packet', {}).get('action_stiffness') != action):
            raise ValueError('The direct simple K action lost its original estimate or captured correction.')
    status = str(record.get('captured_status', record.get('status', 'not supplied')))
    unavailable = ('unavailable' in status or bool(record.get('simple_projection_failures')))
    from_prior = bool((estimated or {}).get('representative_from_prior',
                                          record.get('representative_from_prior', False)))
    if unavailable:
        title = 'K Gaussian | update unavailable; prior retained'
    elif from_prior:
        title = 'K Gaussian | initial prior representative'
    elif record.get('status') == 'prediction_with_retained_parameters':
        title = 'K Gaussian | retained between observations'
    else:
        title = 'K Gaussian | sequential posterior approximation'
    values = lambda array: ', '.join('{:.4g}'.format(value) for value in array)
    marker = Marker()
    marker.header.frame_id = snapshot.frame_id
    marker.header.stamp = rospy.Time.from_sec(snapshot.stamp)
    marker.ns, marker.id = 'estimated_stiffness', 0
    marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
    marker.pose.orientation.w = 1.
    marker.pose.position.x, marker.pose.position.z = .72, .70
    marker.scale.z = .023
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = .10, .12, .19, 1.
    marker.text = ('{}\nRepresentative exp(mean log K) [Nm/rad]: {}\n'
                   'sigma(log K): {}\nNormalized full-covariance log-K Gaussian\n'
                   'BFS off | evidence not computed | {}\n'
                   'Density {}\nt={:.3f} | body version {}').format(
        title, values(representative), values(np.sqrt(np.diag(covariance))), status,
        density['density_id'][:16], snapshot.stamp, ','.join(str(version) for version in versions))
    if proposal is None:
        marker.text += '\nRepresentative command action unavailable'
    return marker
