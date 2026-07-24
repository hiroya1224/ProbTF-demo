# Grape real-bag vertical slice: 20260612-07

Workflow status: `EXPERIMENTAL`. Recommendation available: `false`.

- Run ID: `c63e4ebe570b6943f7dd`
- Source bag SHA-256: `75292b0c79dd1a3be2869eb3a0c3766df9561336efe2948386ebcb86e67297b8`
- Interval: `45.003`–`52.998` s from bag start
- Source commit: `9b83e4de3eca38e59a65957c2387a5d2c6750bdc`
- Effective response: `low_dimensional_effective_response/v1`

## Trajectory diagnostics

- desired→actual position RMS: `0.0370493 m`
- desired→actual attitude RMS: `0.068621 rad`
- nominal approximation→actual position RMS: `0.210854 m`
- nominal approximation→actual attitude RMS: `1.08808 rad`

The nominal curve is an equilibrium-centered local integration of recorded `PoseControlPid.total`; it is not the deployed PC/MCU replay oracle. Numerical acceleration is reported only as a diagnostic and never enters the likelihood.

## Effective-response posterior

- Identifiable at the conditional design level: `True` (rank `48/48`, condition `2647.0391850074616`)
- `roll_effectiveness`: mean `0.332203`, interval `[0.253557, 0.409944]`
- `pitch_effectiveness`: mean `0.354095`, interval `[0.225123, 0.570384]`
- `roll_from_pitch_cross_coupling`: mean `0.00795866`, interval `[-0.0152268, 0.0403554]`
- `pitch_from_roll_cross_coupling`: mean `-0.241361`, interval `[-0.476003, -0.0781567]`
- `roll_delay_s`: mean `0.08875`, interval `[0, 0.12]`
- `pitch_delay_s`: mean `0.08875`, interval `[0, 0.12]`

## Recommendation gates

- `exact_controller_replay`: `False` (ORACLE_UNAVAILABLE)
- `bag_derived_exact_fixture`: `False` (BAG_DERIVED_FIXTURE_UNAVAILABLE)
- `probability_calibration`: `False` (NO_COMPLETE_12_FOLD_SELECTION_RESULT)
- `joint_state_parameter_dependence`: `False` (MODULAR_TRAJECTORY_MIXTURE_ONLY)
- `controller_integrator_state`: `False` (NOT_RECORDED_OR_LATENTLY_INFERRED)
- `recommendation`: `False` (EXPERIMENTAL)

The candidate CSV is therefore an unevaluated common grid. It contains no success-probability or support claim.
