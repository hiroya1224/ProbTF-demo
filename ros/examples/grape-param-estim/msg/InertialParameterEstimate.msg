# Posterior summary for one ordered inertial-parameter vector.
# Every per-parameter array has the same length as parameter_names.
# covariance is a row-major square matrix with that dimension.
std_msgs/Header header
string source_bag
string model
string[] parameter_names
float64[] mean
float64[] map
float64[] std
float64[] lower_95
float64[] upper_95
float64[] covariance

uint32 particle_count
float64 effective_sample_size
uint64 update_index
uint64 observation_count
bool resampled
float64 log_likelihood
uint64 seed

probtf_msgs/Provenance provenance
probtf_msgs/ApproximationInfo approximation
