# A possibly decimated particle snapshot for offline inspection.
# The i-th retained particle starts at values[i * stride].
# normalized_weight contains one entry per retained particle and sums to one.
std_msgs/Header header
string[] parameter_names
uint32 particle_count
uint32 stride
float64[] values
float64[] normalized_weight
