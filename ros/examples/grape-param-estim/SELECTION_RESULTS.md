# Grape backend selection results

Selection status: `EXPERIMENTAL`.

This file records frozen held-out decisions. A missing comparison is reported as `EXPERIMENTAL`; it is not treated as evidence for a default.

- Source commit: `13aeb69792c90660568243705af0362df4013f0a`
- Bag manifest hash: `ed85aa1e1d72aacdab55888f2208f31cd06d9fb2ca494327f56bd5a25081b5ac`
- Selection protocol hash: `23ff95871e5199f2fec121b9fee5b6e6d542a141b94e68cc4f3a3b82accd92b9`
- Result hash: `9b2725232b94cbd289243e31fca560e0e2b334698112f8610a45fed6596db059`
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

## Post-hardening regeneration

The baseline machine-readable result is
[`config/selection_results.json`](config/selection_results.json). After the
strict exact-backend capability and counterfactual-result gates were added,
the same frozen protocol was regenerated without observations and preserved
separately as
[`config/selection_results_2026-07-24_post_hardening.json`](config/selection_results_2026-07-24_post_hardening.json).

- Grape implementation commit: `9b83e4de3eca38e59a65957c2387a5d2c6750bdc`
- Protocol hash: `23ff95871e5199f2fec121b9fee5b6e6d542a141b94e68cc4f3a3b82accd92b9`
- Post-hardening result hash: `ef2d1837aa33b1d87c88dea5607f0c739f9b1a73fd81a97754f84c57c66bc52c`
- Observations/folds: `0/12`
- `selection_complete`: `false`

Apart from the source commit and resulting content hash, the decision payload
is identical to the baseline: every candidate remains `EXPERIMENTAL`, and no
default or recommendation is selected.

## Real-bag diagnostic evidence

The bag 4/7/8 vertical slices recorded on 2026-07-24 are indexed in
[`results/real_bag_vertical_slice/2026-07-24/INDEX.md`](results/real_bag_vertical_slice/2026-07-24/INDEX.md).
They bind source commit
`9b83e4de3eca38e59a65957c2387a5d2c6750bdc`, source-bag, normalized-input,
configuration, coherent-trajectory, and per-file hashes.

These runs exercise the common error-state EKF/RTS and low-dimensional
effective-response pipeline on real evidence, but all recommendation gates
remain false. They are diagnostics, not held-out selection observations, so
the submitted-observation count above remains zero.
