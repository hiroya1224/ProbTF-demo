# Locally fitted IMU kinematics for relative-pose production.
# header.frame_id identifies the sensor frame.
std_msgs/Header header
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 angular_acceleration
float64[9] angular_acceleration_covariance
geometry_msgs/Vector3 specific_force
float64[9] specific_force_covariance
