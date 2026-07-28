# One lossless, controller-tick replay frame. Header stamp is the controller tick.
std_msgs/Header header
string schema
string controller_snapshot_sha256
float64 dt

# Exact input seen by the controller.
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 angular_velocity
geometry_msgs/Point target_position
geometry_msgs/Vector3 target_velocity
geometry_msgs/Vector3 target_acceleration
geometry_msgs/Quaternion target_orientation
geometry_msgs/Vector3 target_rpy
geometry_msgs/Vector3 target_angular_velocity
geometry_msgs/Vector3 target_angular_acceleration
# Signed values preserve the core's -1 "no previous mode" sentinel.
int32[6] control_mode
# Per-axis integral enable semantics for this tick. Roll/pitch reflect the
# pre-step integration gate, including the trigger tick's zero integral dt.
bool[6] integration_enabled
int32 flight_state
bool force_landing
bool reset
float64 initial_height
string[] joint_names
float64[] joint_positions
# Resolved per-tick allocation geometry; names/offsets follow ReplayMetadata.
string[] allocation_geometry_names
uint32[] allocation_geometry_offsets
float64[] allocation_geometry_values

# Six axes, axis-major. Each axis stores error_p, error_i, previous_error_i,
# error_d, result, p_term, i_term, d_term.
float64[48] pid_state_before
float64[48] pid_state_after
bool start_roll_pitch_integration_before
bool start_roll_pitch_integration_after
float64 previous_stamp_before
float64 previous_stamp_after
int32 previous_flight_state_before
int32 previous_flight_state_after
int32[6] previous_control_mode_before
int32[6] previous_control_mode_after
bool previous_force_landing_before
bool previous_force_landing_after
bool has_previous_force_landing_before
bool has_previous_force_landing_after
uint32 pending_events_before
uint32 pending_events_after
# Allocation state retained by the upstream 2-DOF zero-component branch.
float64[] target_gimbal_angles_before
float64[] target_gimbal_angles_after
float64 target_roll_before
float64 target_roll_after
float64 target_pitch_before
float64 target_pitch_after

# RPY actually used by the live wrapper after its tf conversion. The matrix
# orientation above remains authoritative; this field preserves branch-level
# conformance around Euler singularities.
geometry_msgs/Vector3 current_rpy

# PC-side controller output and conformance channels.
float64[6] pid_result
float64[6] pid_p_term
float64[6] pid_i_term
float64[6] pid_d_term
float64[] target_vectoring_force
# Exact values written to FourAxisCommand.angles after the controller's
# float32 boundary; these are not interchangeable with candidate_yaw_term.
float64[3] four_axis_angles
float64[] base_thrust
float64[] gimbal_angle
uint32 torque_allocation_rows
uint32 torque_allocation_columns
# Row-major matrix.
float64[] torque_allocation_matrix_inverse
float64 target_roll
float64 target_pitch
float64 candidate_yaw_term
geometry_msgs/Vector3 effective_target_acceleration
bool saturated
# Each entry is one set bit from the core's uint32 event mask.
uint32[] events
