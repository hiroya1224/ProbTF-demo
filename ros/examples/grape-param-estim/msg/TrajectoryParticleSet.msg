# Flattened trajectory particles. transforms/twists are sample-major:
# index = sample_index * trajectory_length + time_index.
std_msgs/Header header
string run_id
string trajectory_id
uint32 trajectory_length
uint64[] sample_ids
float64[] sample_weights
time[] stamps
geometry_msgs/Transform[] transforms
geometry_msgs/Twist[] twists

string candidate_id
string[] candidate_parameter_names
float64[] candidate_parameters

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
string approximation
probtf_msgs/Provenance provenance
