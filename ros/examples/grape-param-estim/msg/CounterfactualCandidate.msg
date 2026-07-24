# Human-review-only closed-loop candidate evaluation.
std_msgs/Header header
string run_id
string candidate_id
string[] parameter_names
float64[] parameter_values

float64 success_probability
float64 credible_lower
float64 credible_upper
float64 lower_credible_bound
float64 recommendation_threshold
string support_label
float64 candidate_support_distance
float64 state_action_support_distance_p95
float64 importance_weight_ess
float64 maximum_predictive_std

string[] constraint_names
float64[] constraint_violation_probability
bool recommendable
string proposal_status

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
