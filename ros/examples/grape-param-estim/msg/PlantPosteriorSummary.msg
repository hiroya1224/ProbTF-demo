# Visualization/transport summary only. posterior_particles.npz is authoritative.
std_msgs/Header header
string schema
string run_id
string model_id
string interpretation
string prior_id
string likelihood_id
string controller_snapshot_sha256
string posterior_particles_sha256
string run_manifest_sha256
string source_commit
string[] source_bag_sha256
uint64 seed

string[] parameter_names
float64[] mean
float64[] map
float64[] lower_95
float64[] upper_95
# Row-major square matrices in parameter_names order.
float64[] covariance
float64[] correlation

string[] derived_quantity_names
float64[] derived_mean
float64[] derived_lower_95
float64[] derived_upper_95

uint32 particle_count
uint32 hpd95_particle_count
float64 hpd95_weight
float64 effective_sample_size
float64 credible_probability
bool calibrated_physical_parameters
bool exact_controller_used
bool failure_validation_passed
bool success_validation_passed
bool recommendation_available
