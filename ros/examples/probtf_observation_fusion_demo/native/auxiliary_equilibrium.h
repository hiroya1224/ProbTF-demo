#ifndef PROBTF_AUXILIARY_EQUILIBRIUM_H
#define PROBTF_AUXILIARY_EQUILIBRIUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This ABI belongs only to the inference demo, never the physical plant. */
typedef struct {
  int status;
  int refinement_termination;
  int refinement_iterations;
  int eigenvalues_available;
  int convexity_certified;
  int direct_attempted;
  int direct_used;
  int direct_failed;
  uint64_t requested_stages;
  uint64_t used_stages;
  uint64_t refinement_calls;
  uint64_t completed_refinements;
  uint64_t newton_iterations;
  uint64_t line_search_evaluations;
  uint64_t bisections;
  uint64_t residual_warning_count;
  uint64_t nonpositive_curvature_count;
  uint64_t curvature_audit_failure_count;
  double last_residual;
  double maximum_residual;
  double certificate_ratio_upper;
  double certificate_rounding_margin;
  double direct_KKT_residual;
  double eigenvalues[6];
  char direct_error[256];
  char error[512];
} ProbtfAuxiliaryDiagnostic;

int probtf_auxiliary_abi_version(void);
void *probtf_auxiliary_create(const char *urdf, int dimension,
                             const int *spring_types, char *error, size_t error_size);
void probtf_auxiliary_destroy(void *handle);
int probtf_auxiliary_gravity_hessian_bound(void *handle, double *bound);
/* Rows are independent, outputs use contiguous row-major matrices.
 * mode=0: target=logK, same anchored canonical straight-logK continuation.
 * mode=1: target=K, one refinement from anchor_theta, no log/exp roundtrip.
 * mode=2: logK; certified globally strongly convex direct solve, otherwise mode=0.
 * Failed rows retain their diagnostic and have NaN numeric outputs. */
int probtf_auxiliary_evaluate_batch(void *handle, size_t rows, int mode,
    int workers, const double *target, const double *command,
    const double *anchor_logK, const double *anchor_theta,
    double *theta, double *hessian, double *parameter_jacobian,
    double *derivative, unsigned char *active, double *residual,
    ProbtfAuxiliaryDiagnostic *diagnostics);

#ifdef __cplusplus
}
#endif
#endif
