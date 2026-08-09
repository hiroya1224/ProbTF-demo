# Deterministic geometric Savitzky--Golay dynamics estimator: data dictionary

This document describes the files emitted by
`deterministic_savgol_dynamics_estimator.py`.

The estimator is the Savitzky--Golay/local-polynomial counterpart of
`deterministic_spline_dynamics_estimator.py`.  The downstream rigid-body
parameterization, analytic parameter Jacobian, actuator/wrench model,
external-wrench replay, and rollout diagnostics are retained.  The pose
kinematics front end is replaced completely.

## 1. Pose-to-kinematics front end

For every mocap pose observation at time `t_i`, the raw timestamp is retained.
There is no uniform pose resampling before the local-polynomial fit used by the
parameter loss.

The polynomial degree is fixed to 5.  The user supplies one physical window
width `W` in seconds.  All raw pose samples contained in the window are used.
For interior parameter-estimation times the window is centered:

```text
[t_i - W/2, t_i + W/2].
```

The parameter loss is evaluated only at raw pose timestamps for which this
complete centered window exists and for which the delayed command history is
available.  At the beginning/end of the recorded pose interval a full-width
window may be shifted inward, but these shifted edge estimates are used only
for fit/rollout diagnostics, not for the Newton--Euler parameter loss.

A degree-5 polynomial requires at least 6 pose samples in every window.  A
window that cannot satisfy this condition is rejected before parameter
optimization.  With exactly 6 samples the polynomial derivatives are defined,
but there are no residual degrees of freedom for estimating an empirical
least-squares covariance; the corresponding covariance is stored as NaN.

### Translation

For local time offset `tau = t - t_i`, the estimator fits

```math
p(t_i+\tau) \approx \sum_{r=0}^{5} \frac{a_r}{r!}\tau^r.
```

The center values are

```math
p_i=a_0,\qquad v_i=a_1,\qquad a_i=a_2.
```

The implementation solves the least-squares system directly with the actual
(possibly nonuniform) timestamps.  Time offsets are internally scaled for
conditioning; reported derivatives use SI seconds.

When the local fit has positive residual degrees of freedom, the full 3x3
translation residual covariance is propagated through the local linear
least-squares map to obtain covariance matrices for position, velocity, and
acceleration.  These covariance matrices are diagnostic outputs in this
version; the deterministic parameter objective does not yet use them as
weights.

### Rotation

Rotation uses the geometric Savitzky--Golay construction on `SO(3)`.  Around a
local reference rotation, logarithmic relative rotations are fit by a degree-5
polynomial in the Lie algebra.  The exponential differential gives spatial
angular velocity and acceleration, which are then transformed to the body
frame used by the rigid-body dynamics.

Reported fields are therefore

```text
body_rotation                  R(t_i)
body_angular_velocity          omega_B(t_i)
body_angular_acceleration      alpha_B(t_i)
```

rather than derivatives of independently filtered Euler angles or quaternion
components.

## 2. Parameter-estimation inputs

`vehicle_model.json` is deterministic model/chart/geometry input only.
`parameter_prior.json` is the independent finite Gaussian physical-parameter
prior.

The estimated physical chart is the same 14-dimensional chart as the current
spline estimator:

1. mass log scale;
2. six physically admissible inertia coordinates;
3. body-frame CoG position x/y/z;
4. four independent rotor force-effectiveness log scales.

Command lag is searched separately.  Its default initial value in this SG
estimator is exactly `0 s`; a nonzero value is used only when explicitly passed
with `--initial-delay`.

## 3. Per-bag output files

For bag id `<id>`, files are written under

```text
output/deterministic_savgol_dynamics/bags/<id>/
```

### `result.json`

Contains source bag metadata, shared parameter result, selected command lag,
all rollout/reconstruction metrics, inferred external-wrench statistics, and
Savitzky--Golay diagnostics.

The `diagnostics.savgol` object contains at least:

- `degree`: fixed at 5;
- `window_seconds`: physical window width W;
- `minimum_required_points`: 6;
- `minimum_window_sample_count` / `maximum_window_sample_count`;
- raw fit and parameter-estimation intervals;
- fit residual metrics;
- local-polynomial condition diagnostics;
- an explicit statement that raw pose is not resampled for parameter loss.

### `savgol_fit.pdf`

Window/front-end diagnostics:

- raw mocap position and local-polynomial estimate;
- raw orientation and geometric-SG orientation;
- pose residuals;
- translational velocity/acceleration;
- body angular velocity/angular acceleration;
- local translational acceleration standard deviations where estimable;
- number of raw samples in each window and local least-squares condition
  numbers.

### `trajectory.pdf`

Preserves the existing trajectory/reconstruction presentation.  It includes the
observed trajectory and the estimated trajectory reconstructed with the
inferred external wrench.

### `sensor_consistency.pdf`

Preserves the existing gyro/specific-force consistency diagnostics.

### `diagnostic.pdf` and `diagnostic.json`

Preserve the existing numerical/visual consistency diagnostics, with SG-derived
pose kinematics replacing spline-derived kinematics.

### `external_wrench.pdf`

Preserves the existing six-axis inferred external body-wrench plots and
statistics.

### `savgol_dynamics.npz`

Contains the numerical time series used by the reports.  Important groups are:

#### Raw observations

```text
raw_pose_time
raw_pose_sensor_position
raw_pose_sensor_orientation_xyzw
observed_sensor_position
observed_sensor_orientation_xyzw
observed_sensor_velocity_world
observed_angular_velocity_sensor
observed_specific_force_sensor
```

`raw_pose_*` are the original pose-message samples used by SG.  The
`observed_*` arrays retained from the downstream diagnostic grid are not the
source of the SG fit.

#### SG kinematics on the parameter-evaluation times

```text
collocation_time
savgol_sensor_position
savgol_sensor_velocity_world
savgol_sensor_acceleration_world
savgol_body_rotation
savgol_body_angular_velocity
savgol_body_angular_acceleration
```

The historical NPZ name `collocation_time` is retained for downstream
compatibility; in the SG estimator these are raw mocap timestamps satisfying the
centered-window/support conditions, not a separately generated uniform
collocation grid.

#### SG window/covariance diagnostics

```text
savgol_window_seconds
savgol_polynomial_degree
savgol_window_sample_count
savgol_position_fit_condition_number
savgol_rotation_fit_condition_number
savgol_sensor_position_covariance
savgol_sensor_velocity_world_covariance
savgol_sensor_acceleration_world_covariance
```

Each covariance has shape `(N,3,3)`.  NaN means that the local fit had no
residual degrees of freedom from which to estimate empirical observation
variance; it does not mean zero uncertainty.

#### Dynamics and inferred wrench

```text
required_body_wrench
modeled_body_wrench
residual_body_wrench
raw_inferred_external_body_wrench_time
raw_inferred_external_body_wrench
inferred_external_body_wrench_time
inferred_external_body_wrench
inferred_external_body_wrench_initial
inferred_external_body_wrench_correction
```

The raw residual wrench remains

```math
W_{\mathrm{res}}(t)=W_{\mathrm{required}}(t)-W_{\mathrm{modeled}}(t).
```

The refined inferred external wrench is the piecewise-linear wrench trajectory
used by the existing replay/reconstruction check.

#### Forward/replay trajectories

The existing `estimated_forward_*`, `external_wrench_forward_*`, and
`reference_forward_*` arrays are retained with the same meaning as in the
spline estimator.

## 4. Root output files

Under `output/deterministic_savgol_dynamics/`:

- `result.json`: shared estimator result and links to per-bag outputs;
- `parameters.txt` / `parameters.pdf`: physical parameter comparison and
  optimization/reconstruction summary;
- `delay_profile.pdf`: command-lag search diagnostic;
- `DATA_DICTIONARY.md`: copy of this document.

The legacy forward-rollout diagnostic still has a numerical output/integration
step.  That grid does **not** determine the SG derivative estimates or the
Newton--Euler parameter-loss timestamps.

## 5. Window ablation

`savgol_window_ablation.py` computes the smallest feasible degree-5 window from
the actual raw pose timestamps for every bag and uses the largest per-bag value
as the global minimum.  By default it studies

```text
W_min, 0.5 s, 1.0 s, 1.5 s, 2.0 s
```

(after duplicate removal).  Any requested W below `W_min` is rejected before
optimization and recorded as `skipped_below_minimum`.

Each valid W is a completely separate estimator run under

```text
output/savgol_window_ablation/W_<...>s/deterministic_savgol_dynamics/
```

The ablation root contains:

- `ablation.json`: exact W support, per-case parameter/result summaries and
  paths to the full per-W `result.json` files;
- `ablation.pdf`: W versus objective/lag, physical parameters, six-axis
  external-wrench RMS, and rollout/replay errors.

No single W is silently selected as the canonical answer by the ablation
runner.

## 6. SG confidence/ridge output

`savgol_dynamics_confidence.py` preserves the current confidence-layer files
for a single W:

```text
output/savgol_dynamics_confidence/<bag-id>/confidence.pdf
output/savgol_dynamics_confidence/<bag-id>/confidence.json
output/savgol_dynamics_confidence/<bag-id>/parameter_likelihood.json
output/savgol_dynamics_confidence/<bag-id>/parameter_posterior.json
```

The deterministic parameter point and the confidence likelihood both use
**all** valid centered raw-pose SG evaluations.  Therefore one residual body
wrench sample and its parameter Jacobian are retained for every valid SG center;
there is no confidence-specific temporal subsampling.

The retained first-layer residual-wrench model is Gaussian with an empirical
nonzero mean and 6x6 covariance estimated from all of those residual-wrench
samples.  The raw data-only SVD/ridge information, local Gaussian likelihood
factor, Gaussian-prior fusion, trajectory reconstruction check, and confidence
PDF pages retain their previous meanings.  The translation local-LS derivative
covariance remains available as a separate diagnostic and is not used directly
in this residual-wrench likelihood.

By default the W-ablation runner also executes `savgol_dynamics_confidence.py`
for each valid W, reusing that W's already optimized deterministic
`result.json`.  Thus the physical parameter optimizer is not run a second time.
The per-W directory additionally contains the confidence/ridge PDF and the
likelihood/posterior JSON files.  Use `--skip-confidence` only when a faster
pure deterministic ablation is desired.

## 7. Command timestamp diagnostics

The SG estimator does not assume a command publish period.  For each bag it
measures the positive intervals between recorded rotor-command timestamps and
between recorded gimbal-command timestamps and reports minimum/median/mean/
maximum interval and the reciprocal of the median interval.  The selected lag
is also reported as a ratio to each channel's measured median interval.  These
quantities are diagnostics only; they do not set the lag initial value.  The
default lag initial value remains exactly zero.

## 8. Raw inverse-dynamics residual wrench

In addition to the retained `external_wrench.pdf` (the trajectory-fitted replay
wrench), each SG bag now contains

```text
raw_residual_wrench.pdf
```

This plots `required_body_wrench - modeled_body_wrench` directly at the raw
centered SG dynamics-evaluation timestamps.  It does not pass through the
legacy uniform forward-rollout grid and it is not altered by the subsequent
external-wrench replay optimization.  Its six-axis mean/std/RMS are stored in
`diagnostics.raw_inverse_dynamics_residual_wrench_statistics` and are included
explicitly in the W-ablation comparison.  This is the preferred diagnostic for
checking whether a periodic wrench pattern moves or disappears as W changes.
