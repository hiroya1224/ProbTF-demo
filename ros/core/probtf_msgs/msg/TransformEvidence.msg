# One source's likelihood contribution for a directed transform.
std_msgs/Header header
string parent_frame_id
string child_frame_id
string source_id
bool has_sequence
uint64 sequence
bool has_orientation
BinghamDistribution orientation_bingham
bool has_position
float64[9] position_information
geometry_msgs/Vector3 position_information_vector
