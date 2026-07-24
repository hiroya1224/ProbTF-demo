# Grape real-bag vertical slice: 20260612-04

Workflow status: `EXPERIMENTAL`. Recommendation available: `false`.

- Run ID: `95902dcc5f562795e72d`
- Source bag SHA-256: `bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8`
- Interval: `18.006`–`25.547` s from bag start
- Source commit: `b48bc63d1d961fae7d7bdc73b9d08aff84835910`
- Effective response: `low_dimensional_effective_response/v1`

## Trajectory diagnostics

- desired→actual position RMS: `0.332293 m`
- desired→actual attitude RMS: `0.244735 rad`
- nominal approximation→actual position RMS: `2.81515 m`
- nominal approximation→actual attitude RMS: `0.857021 rad`

The nominal curve is an equilibrium-centered local integration of recorded `PoseControlPid.total`; it is not the deployed PC/MCU replay oracle. Numerical acceleration is reported only as a diagnostic and never enters the likelihood.

## Effective-response posterior

- Identifiable at the conditional design level: `True` (rank `48/48`, condition `457.9691567066706`)
- `roll_effectiveness`: mean `1.14387`, interval `[0.9342, 1.35396]`
- `pitch_effectiveness`: mean `0.675231`, interval `[0.591963, 0.74152]`
- `roll_from_pitch_cross_coupling`: mean `0.0437039`, interval `[0.0245693, 0.0641312]`
- `pitch_from_roll_cross_coupling`: mean `-0.427874`, interval `[-0.672339, -0.159955]`
- `roll_delay_s`: mean `0.0025`, interval `[0, 0.04]`
- `pitch_delay_s`: mean `0.0025`, interval `[0, 0.04]`

## Recommendation gates

- `exact_controller_replay`: `False` (ORACLE_UNAVAILABLE)
- `bag_derived_exact_fixture`: `False` (BAG_DERIVED_FIXTURE_UNAVAILABLE)
- `probability_calibration`: `False` (NO_COMPLETE_12_FOLD_SELECTION_RESULT)
- `joint_state_parameter_dependence`: `False` (MODULAR_TRAJECTORY_MIXTURE_ONLY)
- `controller_integrator_state`: `False` (NOT_RECORDED_OR_LATENTLY_INFERRED)
- `recommendation`: `False` (EXPERIMENTAL)

The candidate CSV is therefore an unevaluated common grid. It contains no success-probability or support claim.
