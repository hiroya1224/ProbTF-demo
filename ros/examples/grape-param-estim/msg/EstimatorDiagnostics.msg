# Per-update diagnostics. Record time and event time are separate because some
# Grape command messages do not carry a Header.
std_msgs/Header header
uint64 update_index
uint64 observation_count
time source_record_time
time event_time

float64 ess_before
float64 ess_after
bool resampled
float64 normalized_innovation_squared
float64 force_residual_norm
float64 torque_residual_norm
float64 log_likelihood_increment

uint32 mcmc_accepted
uint32 mcmc_proposed
uint32 excitation_rank
float64 excitation_condition
string gate_reason
