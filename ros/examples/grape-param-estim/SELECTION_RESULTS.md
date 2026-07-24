# Grape backend selection results

Selection status: `EXPERIMENTAL`.

This file records frozen held-out decisions. A missing comparison is reported as `EXPERIMENTAL`; it is not treated as evidence for a default.

- Source commit: `b479764`
- Bag manifest hash: `ed85aa1e1d72aacdab55888f2208f31cd06d9fb2ca494327f56bd5a25081b5ac`
- Selection protocol hash: `23ff95871e5199f2fec121b9fee5b6e6d542a141b94e68cc4f3a3b82accd92b9`
- Result hash: `083bca4d626c47361cdd928d48099f382897d85499099a3214293aebd7fe972f`
- Outer held-out folds: `12`
- Submitted observations: `0`
- Resampling unit: whole episode/bag

## Decisions

| component | candidate | status | held-out bags | metric mean (95% bootstrap CI) | hard-gate failures | reason |
|---|---|---|---:|---|---|---|
| controller_replay | exact_cpp_pc_mcu_replay | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| controller_replay | python_vector_pid_surrogate | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| counterfactual_usefulness | bayesian_closed_loop | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| counterfactual_usefulness | climatology | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| counterfactual_usefulness | nominal_deterministic | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| effective_response | effective_sparse_gp_residual | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| effective_response | low_dim_effective_response | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| effective_response | structured_6dof_mechanics | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| inference | joint_pmcmc | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| inference | likelihood_free_bayessim | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| inference | modular_tempered_smc | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| trajectory_smoother | error_state_ekf_rts | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |
| trajectory_smoother | factor_graph_imu_preintegration | `EXPERIMENTAL` | 0 | not measured | not evaluated | incomplete_outer_fold_evaluation |

## Current limitation

No candidate may be promoted from this file until all frozen outer folds and hard gates are present. In particular, an unavailable or unverified exact PC/MCU controller oracle blocks counterfactual recommendation; Python replay remains an approximation.
