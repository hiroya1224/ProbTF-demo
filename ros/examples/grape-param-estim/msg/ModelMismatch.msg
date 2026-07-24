# Probabilistic desired/nominal/actual gap at one event time.
std_msgs/Header header
string run_id
string sample_set_id

geometry_msgs/Transform desired
geometry_msgs/Transform nominal
geometry_msgs/Transform actual_posterior_mean
geometry_msgs/Transform nominal_to_actual_mean

# SE(3) log coordinates: translation xyz, rotation xyz.
float64[6] tracking_residual_mean
float64[6] tracking_residual_lower
float64[6] tracking_residual_upper
float64[6] model_residual_mean
float64[6] model_residual_lower
float64[6] model_residual_upper
float64[36] model_residual_covariance

uint32 trajectory_sample_count
string[] diagnostics
string[] source_bag_sha256
string[] normalized_dataset_sha256
string[] source_topics
time source_interval_start
time source_interval_end
string config_sha256
string source_commit
string model_version
uint64 seed
bool causal
time prefix_cutoff
probtf_msgs/Provenance provenance
