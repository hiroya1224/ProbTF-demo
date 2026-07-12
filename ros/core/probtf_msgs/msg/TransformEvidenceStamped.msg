# One source's natural-parameter contribution for a directed transform.
# header.frame_id is the parent frame; child_frame_id is the child frame.
std_msgs/Header header
string child_frame_id
string source_id
string evidence_kind

bool has_sequence
uint64 sequence

bool has_orientation
# Trace-zero symmetric Bingham natural parameter in the [w,x,y,z] basis.
# Upper-triangle order: ww, wx, wy, wz, xx, xy, xz, yy, yz, zz.
float64[10] orientation_natural_parameter_upper_wxyz

bool has_translation
# Singular positive-semidefinite information matrices are allowed.
# Upper-triangle order: xx, xy, xz, yy, yz, zz.
float64[6] translation_information_upper
geometry_msgs/Vector3 translation_information_vector

ApproximationInfo approximation
Provenance provenance
