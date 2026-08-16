# Optional parameter-prior JSON

The estimator is prior-free by default. A v1 prior is enabled only with
`--prior-json` and may constrain the following gauge-invariant SI quantities:

- `cog_position_body_m`: `x`, `y`, `z` in metres;
- `inertia_over_mass_m2`: `xx`, `yy`, `zz`, `xy`, `xz`, `yz` in m²;
- `force_effectiveness_over_mass`: `rotor_1` through `rotor_4` in kg⁻¹.

The schema marker is `grape-param-estim/parameter-prior/v1`. Each Gaussian
factor specifies an ordered component list, either a
`vehicle_model_nominal` target or an explicit SI `value`, and exactly one of a
positive diagonal `std` vector or a symmetric positive-definite full
`covariance`. Nominal targets are resolved from the vehicle model loaded for
that run. Whitening uses a Cholesky solve.

Only common-scale-gauge-invariant quantities are accepted. Absolute mass,
absolute inertia, and absolute force effectiveness are deliberately outside
v1 because they would change the estimator's exact scale gauge.

The files under `pseudo_conditioning/` are artificial ablation factors, not
calibrated measurements. Their deliberately tight standard deviations are
10⁻⁵ m for CoG, 10⁻⁶ m² for inertia/mass, and 10⁻⁵ kg⁻¹ for force/mass. A real
external prior can use the same schema but must document a defensible target
and covariance and should not claim these artificial values as uncertainty.

Example:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --prior-json minimal/config/priors/pseudo_conditioning/group/cog_all_nominal.json
```
