uint8 REPRESENTATIVE_NONE=0
uint8 REPRESENTATIVE_EXACT_MAP=1
uint8 REPRESENTATIVE_COMPONENT_MODE_APPROXIMATION=2
uint8 REPRESENTATIVE_PRODUCER_SUPPLIED=3
uint8 REPRESENTATIVE_MOMENT=4

# header.frame_id is the parent/target frame. A component maps coordinates
# from child_frame_id into header.frame_id: z_parent = R(Q) z_child + X.
std_msgs/Header header
string child_frame_id
string edge_id
string authority
bool is_static

uint8 representative_kind
geometry_msgs/Transform representative

ProbabilisticTransformComponent[] components
ApproximationInfo approximation
Provenance provenance
