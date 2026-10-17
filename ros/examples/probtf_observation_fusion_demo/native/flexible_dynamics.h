#ifndef PROBTF_DEMO_FLEXIBLE_DYNAMICS_H
#define PROBTF_DEMO_FLEXIBLE_DYNAMICS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed-base arms with 1..6 bounded, scalar-revolute joints only. All arrays are float64;
 * matrix outputs are row-major. All int-returning operations return 0 on
 * success and -1 on failure. No C++ exception crosses this boundary. */
typedef struct ProbtfDynamicsOptions {
  double max_step_s;
  double min_step_s;
  double absolute_residual_tolerance;
  double relative_residual_tolerance;
  int max_newton_iterations;
  int max_line_search_steps;
  int max_step_halvings;
} ProbtfDynamicsOptions;

enum ProbtfDynamicsDoubleDiagnostic {
  PROBTF_DYN_TIME_S = 0,
  PROBTF_DYN_TOTAL_ENERGY_J,
  PROBTF_DYN_KINETIC_ENERGY_J,
  PROBTF_DYN_GRAVITY_ENERGY_J,
  PROBTF_DYN_SPRING_ENERGY_J,
  PROBTF_DYN_DAMPING_LOSS_J,
  PROBTF_DYN_COMMAND_WORK_J,
  PROBTF_DYN_IMPACT_LOSS_J,
  PROBTF_DYN_ENERGY_BALANCE_RESIDUAL_J,
  PROBTF_DYN_MAX_TORQUE_RESIDUAL_NM,
  PROBTF_DYN_MIN_ACCEPTED_STEP_S,
  PROBTF_DYN_MAX_ACCEPTED_STEP_S,
  PROBTF_DYN_LAST_ADVANCE_DT_S,
  PROBTF_DYN_DOUBLE_DIAGNOSTIC_COUNT
};

enum ProbtfDynamicsCountDiagnostic {
  PROBTF_DYN_ACCEPTED_SUBSTEPS = 0,
  PROBTF_DYN_NEWTON_ITERATIONS,
  PROBTF_DYN_LINE_SEARCH_EVALUATIONS,
  PROBTF_DYN_STEP_HALVINGS,
  PROBTF_DYN_JOINT_IMPACTS,
  PROBTF_DYN_ACTIVE_STOP_SUBSTEPS,
  PROBTF_DYN_COUNT_DIAGNOSTIC_COUNT
};

/* Diagnostic energy increments/counts describe the last advance only;
 * TIME is since reset. Energy balance is E_end-E_start-work+damping+impact.
 * A nonzero balance residual is measured, never removed by scaling state. */
int probtf_dynamics_abi_version(void);
void probtf_dynamics_default_options(ProbtfDynamicsOptions *options);
void *probtf_dynamics_create(const char *urdf_path, int dimension,
    const double *stiffness, const double *damping_or_null,
    double damping_ratio, const ProbtfDynamicsOptions *options_or_null,
    char *error, size_t error_capacity);
void probtf_dynamics_destroy(void *handle);
const char *probtf_dynamics_last_error(void *handle);

/* The FIRST successful reset fixes reference mass and D for this handle.
 * Later resets preserve both. If no D was supplied, initially compute
 * D_i=2*zeta*sqrt(K_i*M(q_initial)_ii). No reference filter is added. */
int probtf_dynamics_reset(void *handle, const double *q,
    const double *velocity, const double *command);
/* Linearly interpolate the actual command between the supplied endpoints.
 * u_start must equal the preceding accepted command. Failure rolls back
 * the entire advance, retaining an error plus numerical diagnostics.
 * dt==0 with unchanged command returns a singleton trace and preserves state
 * and diagnostics; changing the command at zero duration is rejected. */
int probtf_dynamics_advance(void *handle, double dt,
    const double *u_start, const double *u_end);

/* Trace includes t=0 and every accepted substep endpoint, in relative
 * seconds. At a joint impact the endpoint velocity is the post-impact
 * velocity; no equilibrium samples or synthetic oscillations are inserted. */
size_t probtf_dynamics_trace_size(void *handle);
int probtf_dynamics_trace_copy(void *handle, size_t row_capacity,
    double *times, double *positions, double *velocities);
/* Acceleration is continuous endpoint acceleration after any inelastic
 * impulse, not a finite difference through that instantaneous impulse. */
int probtf_dynamics_get_state(void *handle, double *q,
    double *velocity, double *acceleration);
int probtf_dynamics_get_parameters(void *handle, double *stiffness,
    double *fixed_damping, double *initial_mass_row_major);
int probtf_dynamics_get_diagnostics(void *handle, double *values,
    size_t value_capacity, uint64_t *counts, size_t count_capacity);

#ifdef __cplusplus
}
#endif
#endif
