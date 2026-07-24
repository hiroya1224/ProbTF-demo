# Grape real-bag vertical slice: 20260612-08

Workflow status: `EXPERIMENTAL`. Recommendation available: `false`.

- Run ID: `bcb0786294cad8f7f565`
- Source bag SHA-256: `dc141d4f6d9d3289d279771eb8d85b0765d1e76cd960f5c79211bf57869863c0`
- Interval: `294.003`–`301.997` s from bag start
- Source commit: `9b83e4de3eca38e59a65957c2387a5d2c6750bdc`
- Effective response: `low_dimensional_effective_response/v1`

## Trajectory diagnostics

- desired→actual position RMS: `0.0346057 m`
- desired→actual attitude RMS: `0.0545102 rad`
- nominal approximation→actual position RMS: `0.495223 m`
- nominal approximation→actual attitude RMS: `1.06209 rad`

The nominal curve is an equilibrium-centered local integration of recorded `PoseControlPid.total`; it is not the deployed PC/MCU replay oracle. Numerical acceleration is reported only as a diagnostic and never enters the likelihood.

## Effective-response posterior

- Identifiable at the conditional design level: `True` (rank `48/48`, condition `2493.365411029549`)
- `roll_effectiveness`: mean `0.560003`, interval `[0.2339, 0.776992]`
- `pitch_effectiveness`: mean `0.351051`, interval `[0.287391, 0.427468]`
- `roll_from_pitch_cross_coupling`: mean `-0.0167354`, interval `[-0.0532489, 0.0198436]`
- `pitch_from_roll_cross_coupling`: mean `-0.18701`, interval `[-0.347442, -0.0757397]`
- `roll_delay_s`: mean `0.0625`, interval `[0, 0.12]`
- `pitch_delay_s`: mean `0.0625`, interval `[0, 0.12]`

## Recommendation gates

- `exact_controller_replay`: `False` (ORACLE_UNAVAILABLE)
- `bag_derived_exact_fixture`: `False` (BAG_DERIVED_FIXTURE_UNAVAILABLE)
- `probability_calibration`: `False` (NO_COMPLETE_12_FOLD_SELECTION_RESULT)
- `joint_state_parameter_dependence`: `False` (MODULAR_TRAJECTORY_MIXTURE_ONLY)
- `controller_integrator_state`: `False` (NOT_RECORDED_OR_LATENTLY_INFERRED)
- `recommendation`: `False` (EXPERIMENTAL)

The candidate CSV is therefore an unevaluated common grid. It contains no success-probability or support claim.
