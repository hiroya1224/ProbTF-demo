"""Generic Prob-TF domain/ROS conversions.

Estimator-specific IMU and evidence conversions live in
``probtf_estimators.ros_conversions`` so this bridge depends only on the
foundation package.
"""

from probtf.models import ProbabilisticTransform


def _assign_vector3(vector, values):
    vector.x, vector.y, vector.z = (float(value) for value in values)


def _time_from_seconds(seconds, time_factory=None):
    if time_factory is None:
        import rospy

        time_factory = rospy.Time.from_sec
    return time_factory(float(seconds))


def probabilistic_transform_to_msg(transform, message_type=None, time_factory=None):
    if not isinstance(transform, ProbabilisticTransform):
        raise TypeError("transform must be a ProbabilisticTransform.")
    if message_type is None:
        from probtf_msgs.msg import ProbabilisticTF

        message_type = ProbabilisticTF
    message = message_type()
    message.header.frame_id = transform.parent_frame_id
    if transform.stamp is not None:
        message.header.stamp = _time_from_seconds(transform.stamp, time_factory)
    message.parent_frame_id = transform.parent_frame_id
    message.child_frame_id = transform.child_frame_id
    message.edge_id = transform.edge_id
    message.source_id = transform.source_id
    message.evidence_source_ids = list(transform.evidence_source_ids)
    message.has_position = True
    _assign_vector3(message.position_mean, transform.position_mean)
    message.position_covariance = transform.position_covariance.reshape(-1).tolist()
    message.orientation_bingham.matrix = transform.orientation_bingham.reshape(-1).tolist()
    message.has_orientation = True
    mode = transform.orientation_mode_wxyz
    message.orientation_mode.w = float(mode[0])
    message.orientation_mode.x = float(mode[1])
    message.orientation_mode.y = float(mode[2])
    message.orientation_mode.z = float(mode[3])
    message.approximation_type = transform.approximation_type
    message.closure_approximation = transform.closure_approximation
    return message
