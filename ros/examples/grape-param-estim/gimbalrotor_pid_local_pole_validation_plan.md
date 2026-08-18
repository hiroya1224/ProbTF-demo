# Gimbalrotor PID Local Closed-Loop Pole Validation — Implementation Plan

## 0. Purpose and status of this document

This document specifies a **new, independent diagnostic implementation** for the current Gimbalrotor parameter-estimation / PID postprocessing workflow in `ProbTF-demo`.

The implementation target is the repository state at:

```text
repository: hiroya1224/ProbTF-demo
base commit: aba27b2e51efab80271aa6cd94cd8e521a3a2efd
commit message: debugged core / update probtf_global_fusion
```

The `grape-param-estim` subtree is unchanged between the production Monte Carlo PID result commit

```text
02c53cf15e00640220b8a08a9aaaae6a1e3e41a9
```

and the current base commit above. Therefore all paths and contracts described below refer to the current HEAD while remaining compatible with the already-generated Gimbalrotor estimator and PID-postprocess artifacts.

This document is the implementation contract for the new local-pole validation. It does **not** modify the existing estimator, static PID postprocessor, Monte Carlo static PID postprocessor, or their scientific interpretation.

The primary scientific question is:

> Given the plant distribution inferred from each recorded flight and the exact PID gains that were actually used in that same flight, what distribution of local sampled-data closed-loop poles is obtained around a hover equilibrium, and does that local stability information distinguish the two crashed flights from the successful flight?

The current three flight labels are:

```text
failure1 : crashed
failure2 : crashed
success  : successful flight
```

The implementation must never force its numerical result to agree with these labels. The labels are metadata used only for scientific comparison after the pole calculation.

---

# 1. Motivation and relationship to the current static PID postprocessor

The current static PID postprocessor computes four groupwise static correction factors

```text
xy
z
roll_pitch
yaw
```

and realizes each factor by multiplying `P`, `I`, and `D` in that group by the same scalar.

For a highly simplified scalar plant

```text
x_ddot = b * u
u = -Kp * x - Kd * x_dot - Ki * integral(x)
```

the closed-loop characteristic polynomial is

```text
s^3 + b Kd s^2 + b Kp s + b Ki.
```

If only the static effectiveness changes from `b0` to `b`, preserving the nominal closed-loop poles gives exactly

```text
Kp = (b0 / b) * Kp0
Ki = (b0 / b) * Ki0
Kd = (b0 / b) * Kd0.
```

Thus the existing equal scaling of `P/I/D` is not arbitrary: it coincides with pole preservation for a static-effectiveness double-integrator model.

The actual Gimbalrotor closed loop is substantially richer:

- nonlinear six-DoF rigid-body dynamics;
- full inertia tensor;
- CoG offset;
- four independent rotor force-effectiveness values;
- state-dependent gimbaled thrust directions and moment arms;
- reaction torque;
- six-axis PID;
- nominal-model acceleration allocation;
- integral controller states;
- rotor thrust delay;
- thrust and gimbal actuator limits;
- gimbal rate limit;
- optional actuator time constants;
- sampled controller operation.

The new implementation asks whether the **actual local closed-loop dynamics implied by the current source-compatible model** explain the observed success/failure pattern.

This is a diagnostic stage. It does not yet search for a new gain and does not emit a deployable PID proposal.

---

# 2. New files and placement

Add the following files without changing the existing estimator or static/Monte-Carlo PID postprocessors:

```text
ros/examples/grape-param-estim/
├── gimbalrotor_pid_local_pole_validation_plan.md
└── minimal/
    ├── gimbalrotor_pid_local_pole_validation.py
    ├── three_bag_gimbalrotor_pid_local_pole_validation.py
    └── tests/
        └── test_gimbalrotor_pid_local_pole_validation.py
```

The file in this document should be placed at:

```text
ros/examples/grape-param-estim/gimbalrotor_pid_local_pole_validation_plan.md
```

No code under

```text
minimal/gimbalrotor_pid_monte_carlo_postprocess.py
minimal/gimbalrotor_pid_postprocess_sensitivity.py
src/grape_param_estim/controller.py
src/grape_param_estim/dynamics.py
src/grape_param_estim/closed_loop_stepper.py
```

should be modified merely to make this experiment work.

If a genuinely reusable helper is eventually worth factoring out, do that only after the independent script works and its parity tests pass.

---

# 3. Existing source of truth to reuse

## 3.1 Controller

Use the existing Python port:

```text
src/grape_param_estim/controller.py
```

The relevant public behavior is `GrapeController.step()`.

The local analysis must use the same controller logic as the flight-compatible model:

```text
state/reference
    -> six PID axes
    -> desired six-axis acceleration
    -> nominal acceleration allocation
    -> source-compatible pseudoinverse
    -> four thrust commands + four gimbal commands
```

The controller nominal model must remain the **nominal controller model**, not the sampled real plant.

Use:

```text
ControllerConfig.grape()
```

only as the source-compatible controller structure, limits, control modes, integration gating, feed-forward behavior, and other fixed controller settings.

The actual flight gains must replace the embedded template gains using:

```text
grape_param_estim.controller_config.apply_pid_gain_configuration()
```

The recorded flight gains are the source of truth. Do not use the gains embedded in `ControllerConfig.grape()` as the flight baseline.

## 3.2 Recorded PID gains

Reuse the already-audited static postprocess result for each bag:

```text
pid_gain_postprocess.json
```

Read:

```text
controller_gain_snapshot.gains
controller_gain_snapshot.source
controller_gain_snapshot.record_times
controller_gain_snapshot.source_kinds
controller_mode
input.controller_yaml
input.controller_yaml_sha256
```

The expected gain source is:

```text
rosbag_recorded_dynamic_reconfigure
```

Do not average gains across the three bags.

The known roll/pitch recorded values are:

```text
failure1 : P=20, I=1, D=8
failure2 : P=10, I=1, D=8
success  : P=13, I=1, D=20
```

The script must parse all four groups from the artifact rather than hard-coding these numbers.

## 3.3 Real plant dynamics

Use:

```text
src/grape_param_estim/dynamics.py
```

and in particular:

```text
FullSixDofPlant
advance_actuators
```

The real plant contains:

```text
mass
full inertia tensor
CoG offset
force effectiveness[4]
torque effectiveness[4]
linear drag[3]
angular drag[3]
gimbal-dependent rotor force directions
gimbal-dependent moment arms
reaction torque
gravity
```

The forward rigid-body propagation remains `FullSixDofPlant.step()`.

## 3.4 Vehicle model and geometry

Use:

```text
minimal/grape_vehicle_model.json
```

as the nominal model.

Current production values include zero nominal linear/angular drag.

For the **controller geometry**, reuse:

```text
build_controller_snapshot_geometry()
```

from:

```text
src/grape_param_estim/gimbalrotor_pid_postprocess.py
```

This conversion is mandatory because the JSON stores BODY-frame rotor origins while the controller allocation requires rotor origins expressed relative to the nominal aggregate CoG.

Do not pass `load_vehicle_model(...).body_geometry` directly to `GrapeController`.

For the **real plant**, use the BODY-frame geometry because `FullSixDofPlant` subtracts the sampled CoG when constructing rotor moment arms.

The distinction is:

```text
controller:
    nominal parameters
    CoG-relative controller geometry

real plant:
    sampled parameters
    BODY-frame rotor geometry
```

Do not merge these two geometry conventions.

---

# 4. Important correction: what `rotor_lag_seconds` currently means

This section is mandatory because previous informal discussion sometimes used the word "lag" as if it were a first-order actuator time constant.

In the current estimator, `rotor_lag_seconds` is a **pure time delay of the recorded thrust-command stream under strict zero-order hold**.

The estimator implements exact delay cells through:

```text
minimal/rotor_lag.py
StrictZohCellGrid
```

and the lag derivative acts on:

```text
actual_thrust_lag_jacobian
```

not on the gimbal trajectory.

The current production estimator uses:

```text
gimbal_source = measured_sg
thrust_time_constant = 0.0
gimbal_time_constant = 0.0
```

for the current production runs.

Therefore:

```text
rotor_lag_seconds != thrust_time_constant
rotor_lag_seconds != gimbal_time_constant
rotor_lag_seconds != common delay of the entire ActuatorCommand
```

The new local-pole model must interpret fitted `rotor_lag_seconds` as:

```text
pure delay applied to issued rotor thrust commands only
```

The gimbal command is not delayed by this fitted quantity.

This means the existing generic `ClosedLoopStepper`, whose `ActuatorParameters.delay` delays the complete `ActuatorCommand`, must **not** be reused blindly for the primary local-pole calculation.

The new script should implement an explicit thrust-only ZOH delay queue.

---

# 5. What uncertainty is available in the current estimator

The physical optimizer has:

```text
14 physical chart coordinates
+ 1 rotor_lag_seconds variable
```

but the current post-fit parameter covariance is constructed from the physical 14-D Jacobian and then quotiented by the exact common-scale gauge.

The currently saved Gaussian used by the Monte Carlo PID postprocessor is therefore:

```text
13-D common-scale quotient Gaussian
```

for the physical parameters only.

The current artifact does **not** provide a calibrated or even local joint covariance containing `rotor_lag_seconds`.

Consequently v1 must use:

```text
physical 13-D quotient coordinates : Monte Carlo sampled
rotor_lag_seconds                  : fixed at the fitted point estimate for that bag
```

Do not invent:

```text
rotor-lag variance
rotor-lag Gaussian
physical/lag cross covariance
```

Primary analysis:

```text
delay_mode = fitted_thrust_delay
```

Required ablation:

```text
delay_mode = zero_thrust_delay
```

This comparison explicitly tests how much the fitted delay changes the local stability conclusion while keeping the currently justified uncertainty model intact.

---

# 6. Scientific experiment

For each bag `j`, define:

```text
p_j(theta)
```

as the estimator's selected 13-D quotient Gaussian approximation for that bag, and:

```text
K_j_recorded
```

as the exact recorded PID gains used in that flight.

For plant samples:

```text
theta_j^(n) ~ p_j(theta)
```

construct the local discrete sampled-data closed-loop map:

```text
delta_x_(k+1) = F_cl(theta_j^(n), K_j_recorded) delta_x_k.
```

Compute all eigenvalues:

```text
z_i = eig(F_cl).
```

The primary local stability criterion is:

```text
max_i |z_i| < 1.
```

Do not use a left-half-plane criterion as the primary result because the actual implementation is sampled-data with ZOH and a pure command delay.

Define:

```text
spectral_radius = max_i |z_i|
spectral_margin = 1 - spectral_radius
unstable_pole_count = count(|z_i| > 1)
marginal_pole_count = count(|z_i| == 1)
```

Use the strict mathematical unit-circle comparison.

Do not introduce an arbitrary stability epsilon that silently changes the classification.

A separate numerical diagnostic may report distance from the unit circle.

For each bag and covariance mode report:

```text
stable_sample_fraction_among_valid
valid_sample_fraction
spectral-radius empirical quantiles
spectral-margin empirical quantiles
unstable-pole-count histogram
```

The phrase `stable_sample_fraction` is preferred over `probability of stability` because the input covariance is currently an estimator Gaussian approximation, not a calibrated posterior.

---

# 7. Outcome labels are comparison metadata, not constraints

Use:

```text
failure1 -> crashed
failure2 -> crashed
success  -> successful
```

only in the final summary.

The program must not contain logic such as:

```text
if case_name.startswith("failure"):
    expect_unstable = True
```

or any equivalent behavior.

Possible scientific outcomes include:

### Outcome A

```text
success: high stable fraction
failure1/failure2: low or mixed stable fraction
```

Interpretation:

```text
hover-local closed-loop stability contains a signal consistent with the observed
success/failure outcome.
```

### Outcome B

```text
all three: high stable fraction
```

Interpretation:

```text
the crashes are not explained by hover-local linear instability under this model.
```

This is not an implementation failure. It points to nonlinear excursion, saturation, state-dependent allocation, transient integral state, or another effect.

### Outcome C

```text
failure2: broad/mixed pole distribution
success: narrow stable pole distribution
```

This would be especially consistent with the current excitation/identifiability hypothesis: failure2 contains only a short lift-off-to-crash interval, whereas failure1 contains repeated roll excitation and longer flight.

Do not write acceptance criteria that require any of these scientific outcomes.

---

# 8. Covariance modes and sampling

The required covariance modes are:

```text
conservative_fusion
overlap_corrected
```

Use exactly the same native quotient sampling convention as the existing production Monte Carlo PID postprocessor.

Reuse:

```text
load_sensitivity_artifacts()
prepare_sampling_coordinates(..., coordinate_mode="estimator_quotient")
_psd_eigendecomposition()
```

from the current sensitivity implementation, or duplicate only the minimal validated logic if import boundaries require it.

Do not transform the Gaussian into `centered_scale_free_spd`.

Do not re-fit a Gaussian after a nonlinear coordinate transform.

Default:

```text
sample_count = 512
seed = 0
```

The calculation is substantially more expensive than the static PID push-forward, so 10,000 samples are not required for the first production run.

From the same ordered Monte Carlo sample set, also report prefix summaries for:

```text
N = 128
N = 256
N = 512
```

when the requested sample count is at least 512.

This is a convergence diagnostic only. Do not fail the run because the prefix summaries differ.

---

# 9. Constructing a full real plant from a scale-free sample

Each decoded estimator sample gives:

```text
J_over_m
cog_position_body
force_effectiveness_over_mass
```

The common-scale gauge makes absolute mass unidentifiable.

For the current model, choose the nominal mass gauge:

```text
m = nominal_model.mass
J = m * J_over_m
force_effectiveness = m * force_effectiveness_over_mass
```

and retain:

```text
cog_offset = sampled CoG
torque_effectiveness = nominal model value
linear_drag = nominal model value
angular_drag = nominal model value
```

Current drag is zero, so the common mass/inertia/force scaling leaves the current local rigid-body acceleration model invariant.

Record explicitly in output:

```text
physical_gauge = nominal_mass_gauge
mass_kg = nominal mass
```

If a future nominal model uses nonzero drag, continue to report the chosen gauge explicitly; do not pretend the drag contribution is gauge invariant.

---

# 10. Controller model for one sampled real plant

For every sample:

1. Load nominal `VehicleParameters` and controller geometry.
2. Construct the controller from `ControllerConfig.grape()`.
3. Replace its four PID groups with the exact recorded gains for that bag.
4. Keep all existing PID limits and controller flags unchanged.
5. Construct `FullSixDofPlant` from the sampled real plant.
6. Keep controller allocation based on the nominal controller model.
7. Use the sampled CoG/effectiveness/inertia only in the real plant.

This separation is essential:

```text
recorded PID + nominal controller allocation -> issued command
sampled real plant                         -> actual motion
```

Do not rebuild the controller allocation from the sampled real plant.

---

# 11. Controller YAML and provenance

The current audited controller YAML path recorded by the static postprocess is:

```text
/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml
```

The current recorded SHA-256 in the production static postprocess is:

```text
fd5649d86e67483a6d5e8fe1cb6e4c508a0e03311b74cf0c83a7469df9d6cc8f
```

The new script should accept:

```text
--controller-yaml
```

and verify provenance against the static postprocess artifact.

The YAML numeric gain values are not the flight gain source.

Use the YAML/controller port for fixed controller structure and use:

```text
controller_gain_snapshot.gains
```

from the static postprocess for the actual flight gains.

A provenance mismatch is an input-contract error, not a Monte Carlo sample failure.

---

# 12. Representative controller sample period

The local autonomous pole model requires one representative controller sample period `dt`.

Do not hard-code a guessed controller frequency.

Use the selected ROS bag through the existing audited adapter:

```text
grape_param_estim.real_rosbag.load_flight_data()
```

The issued rotor-thrust command stream is available as:

```text
flight.commanded_thrust.record_times
```

For the selected interval compute:

```text
dt_sequence = diff(record_times)
dt_median
dt_mean
dt_std
dt_min
dt_max
```

Use:

```text
controller_dt = median(dt_sequence)
```

by default.

Also support:

```text
--controller-dt
```

as an explicit override.

Record all timing statistics regardless of override.

Do not reject a bag merely because timing jitter is nonzero. The constant-`dt` pole model is then an approximation, and the jitter statistics must make that approximation visible.

The fitted pure delay is converted into an exact finite-dimensional delay queue relative to this selected `controller_dt`.

---

# 13. Exact thrust-only ZOH delay state

Let:

```text
tau = fitted rotor_lag_seconds
dt  = controller_dt
```

For `tau > 0`, write:

```text
tau = m * dt + r
0 <= r < dt.
```

Use a numerically stable decomposition that canonicalizes values within machine precision of an exact multiple of `dt`.

Define:

```text
delay_depth = ceil(tau / dt)
```

for positive delay and:

```text
delay_depth = 0
```

for zero delay.

Each delayed command state contains only the four issued thrust commands:

```text
q_k in R^4.
```

Therefore the queue contributes:

```text
4 * delay_depth
```

states, not `8 * delay_depth`.

The fitted estimator delay must not be applied to gimbal commands.

## 13.1 Exact-multiple delay

If:

```text
r == 0
tau = m * dt
```

then the entire interval `[k dt, (k+1) dt)` uses:

```text
thrust_target = c_thrust[k-m].
```

## 13.2 Fractional delay

If:

```text
0 < r < dt
```

the current controller interval is split:

```text
segment 1: duration r
    thrust_target = c_thrust[k-m-1]

segment 2: duration dt-r
    thrust_target = c_thrust[k-m]
```

When `m == 0`, segment 2 uses the newly issued current thrust command:

```text
c_thrust[k].
```

The gimbal target for both segments is the current, non-delayed controller gimbal command.

## 13.3 Queue update

After issuing the current command:

```text
new_queue = old_queue shifted left + current issued thrust command
```

The queue stores **issued command values**, not clipped actual actuator thrust.

At hover trim the queue is filled with identical steady issued thrust commands, so pure delay does not change the equilibrium itself.

---

# 14. Actuator propagation inside one controller interval

The estimator production arguments for the current runs record:

```text
thrust_time_constant = 0.0
gimbal_time_constant = 0.0
minimum_thrust = 1.5
maximum_thrust = 27.6145
maximum_gimbal_angle = 3.14
maximum_gimbal_rate = 6.0
```

Read these values from the estimator case's sibling:

```text
arguments.json
```

rather than silently assuming the current defaults.

Set generic `ActuatorParameters.delay = 0.0` in the local implementation because the fitted delay is handled explicitly as a thrust-only queue.

For each constant-command delay segment, mirror the current source-compatible midpoint actuator integration:

```text
actuator_mid = advance_actuators(
    actuator_start,
    segment_command,
    actuator_parameters,
    0.5 * segment_dt,
)

rigid_end = plant.step(
    segment_start_time,
    rigid_start,
    actuator_mid,
    segment_dt,
)

actuator_end = advance_actuators(
    actuator_mid,
    segment_command,
    actuator_parameters,
    0.5 * segment_dt,
)
```

`segment_command` uses:

```text
thrust       = delayed thrust target for that segment
gimbal_angle = current controller gimbal command
```

The `virtual_force` and `desired_acceleration` fields are not used by the actuator transition and may be copied from the current controller command for provenance.

Do not replace the pure delay by:

```text
first-order lag
Padé approximation
low-pass filter
```

in v1.

---

# 15. Hover reference and local equilibrium

The pole calculation must linearize around an actual fixed point of the sampled closed loop, not around a nominal state that is not an equilibrium for the sampled plant.

Use the hover reference:

```text
position             = [0, 0, 0]
linear_velocity      = [0, 0, 0]
linear_acceleration  = [0, 0, 0]
rpy                  = [0, 0, 0]
angular_velocity     = [0, 0, 0]
angular_acceleration = [0, 0, 0]
```

Use the rigid-body hover state:

```text
position             = [0, 0, 0]
orientation          = identity
linear_velocity      = [0, 0, 0]
angular_velocity     = [0, 0, 0]
```

Set:

```text
roll_pitch_integration_active = True
```

because this is the post-takeoff hover branch, not the controller reset branch.

Pure thrust delay does not alter a constant equilibrium because the entire delay queue contains the same steady thrust command.

---

# 16. Solve the sampled-plant trim; do not assume only the z integrator is nonzero

A sampled real plant can have:

```text
CoG offset
asymmetric force effectiveness
full non-diagonal inertia
```

so its equilibrium controller integral need not be:

```text
[0, 0, g / Ki_z, 0, 0, 0].
```

The real equilibrium may require static x/y/roll/pitch/yaw PID outputs to cancel the sampled wrench mismatch.

The controller also recomputes its allocation from the current gimbal angles, so the equilibrium gimbal angles must be self-consistent.

## 16.1 Trim unknown

Use a 10-D trim variable:

```text
y_trim =
[
    integral_error[6],
    steady_gimbal_angle[4],
]
```

All variable names in code must use ASCII.

## 16.2 Trim evaluation

For a candidate `y_trim`:

1. Construct the hover rigid-body state.
2. Construct `ControllerState(integral_error, True)`.
3. Pass the candidate steady gimbal angle to `GrapeController.step()`.
4. Obtain issued thrust and gimbal commands.
5. Form the steady actuator state:
   - thrust = actuator-limit target corresponding to issued thrust;
   - gimbal = candidate steady gimbal.
6. Evaluate the real plant acceleration at this state.
7. Evaluate the fixed-point discrepancy between candidate gimbal and the actuator-limit target of the issued gimbal command.

Define the 10-D root residual as:

```text
[
    real linear acceleration[3],
    real angular acceleration[3],
    target_gimbal - candidate_gimbal[4],
]
```

Use the actual controller and actuator limit conventions.

The initial guess is:

```text
integral from initial_controller_state(..., trim_hover=True)
gimbal = [0, 0, 0, 0]
```

Use a standard nonlinear root/least-squares solver only to find the equilibrium.

This is not gain optimization.

Do not introduce hand-designed bounds, clipping of the trim unknown, regularization toward the nominal trim, or a "plausible trim" prior.

## 16.3 Fixed-point verification

After the trim solve:

- construct the complete augmented steady state;
- fill the thrust delay queue with the steady issued thrust command;
- run one exact local forward controller interval;
- measure the complete one-step fixed-point defect.

Record:

```text
trim_root_status
trim_root_message
trim_residual_vector
trim_residual_norm
full_one_step_trim_defect
controller_integral_defect
gimbal_fixed_point_defect
```

Do not trust the solver's Boolean success flag by itself.

Do not discard a finite solution merely because the solver status is pessimistic.

Conversely, a local "stability" label is meaningful only as an equilibrium statement. If the best finite trim has a material one-step defect, retain the sample and diagnostics but set:

```text
equilibrium_valid = false
```

rather than silently classifying it as stable or unstable.

The default numerical equilibrium tolerance should be derived from machine precision and the scale of the evaluated fixed-point map. It must not be an arbitrary engineering plausibility threshold.

Expose the chosen numerical tolerance in the JSON output.

---

# 17. Local augmented state

At one controller issue time, use the local augmented state:

```text
rigid body local state:
    position                     3
    orientation right tangent    3
    world linear velocity        3
    body angular velocity        3

controller:
    integral error               6

actual actuators:
    thrust                       4
    gimbal angle                 4

thrust delay queue:
    4 * delay_depth
```

Therefore:

```text
local_dimension = 26 + 4 * delay_depth.
```

The discrete Boolean:

```text
roll_pitch_integration_active
```

is fixed to `True` for the hover branch and is not a differentiable state.

---

# 18. Rigid-body local coordinate convention

Use the same orientation convention as the existing controller Jacobian: a right-tangent perturbation.

For a trim rotation `R0` and local perturbation `dtheta`:

```text
R = R0 * Exp(dtheta).
```

For encoding the propagated state:

```text
dtheta_next = Log(R0.T * R_next).
```

Other local coordinates are additive around the trim:

```text
position
velocity
omega
integral
actual thrust
actual gimbal
delay queue thrust commands
```

Implement explicit helpers:

```text
decode_local_state(delta, trim)
encode_local_state(state, trim)
```

Do not finite-difference quaternion components directly.

---

# 19. One-step autonomous map

Implement one deterministic function:

```text
local_closed_loop_step(delta, context) -> delta_next
```

where `context` contains:

```text
sampled real plant
nominal controller
recorded PID
hover reference
trim state
controller_dt
delay decomposition
actuator parameters
```

The forward order is:

```text
1. Decode local augmented state.
2. Run GrapeController.step().
3. Append the newly issued thrust to the conceptual delay history.
4. Select exact delayed thrust target(s) for the current interval.
5. Apply current gimbal command without fitted thrust delay.
6. Propagate actuator + rigid-body state over one or two exact ZOH segments.
7. Update controller integral.
8. Shift the thrust-delay queue.
9. Encode the next augmented state relative to the trim.
```

At:

```text
delta = 0
```

the result must be the stored trim defect.

---

# 20. Linearization strategy for v1

For the first implementation, use a **full central finite-difference Jacobian of the complete augmented one-step map**.

This is preferred for v1 because it:

- differentiates the exact controller code used in the forward map;
- captures PID integral states;
- captures gimbal-dependent nominal allocation;
- captures actuator clips/rate limits on the active branch;
- captures the thrust-only exact delay queue;
- captures the nonlinear rigid-body RK4 propagation;
- avoids introducing a separate analytic model that can silently disagree with the source-compatible forward implementation;
- avoids relying on `step_with_jacobian()`'s separate allocation condition-threshold failure path.

For each state column `i`:

```text
F[:, i] =
    (
        f(+h_i e_i)
        - f(-h_i e_i)
    ) / (2 h_i)
```

around the local origin.

Use finite-difference steps based on machine precision:

```text
base_step = eps ** (1 / 3)
```

for central first derivatives, scaled by the corresponding absolute trim/state scale.

Do not choose a huge perturbation merely to make the matrix look better conditioned.

## 20.1 Step-size diagnostic

At least for:

```text
center plant
a configurable subset of Monte Carlo samples
```

recompute the Jacobian with:

```text
h
h / 2
```

and report:

```text
relative Frobenius difference
maximum absolute entry difference
spectral-radius difference
```

Do not reject a sample solely because the two finite-difference matrices differ.

This is a diagnostic of local linearization quality.

## 20.2 Future analytic acceleration

An analytic/sparse block Jacobian may be introduced later by composing:

```text
GrapeController.step_with_jacobian()
advance_actuators_with_jacobian()
rigid-body local Jacobians
exact delay queue shifts
```

but this is explicitly not required for v1.

Correct forward parity is more important than premature optimization.

---

# 21. Piecewise-smooth boundaries and saturation

The current controller and actuator model contain clips and rate limits.

Do not pre-reject a sample because:

```text
condition number is large
allocation rank under a heuristic threshold is reduced
gain is large
a trim is unusual
a finite pole is far outside the unit circle
```

If the hover trim lies on or extremely near a PID/actuator kink, record:

```text
piecewise_linearization_near_kink = true
```

and retain the calculation if the forward map remains finite.

A central difference across a kink can represent a symmetric secant rather than a one-sided branch derivative; report this condition.

Do not hide it by clipping the perturbation to remain on a preferred branch.

Unexpected programming `ValueError` exceptions must not be converted into "invalid Monte Carlo sample" records.

Catch only narrowly identified numerical failures.

Do not blanket-catch:

```text
ValueError
Exception
```

inside the per-sample Monte Carlo loop.

---

# 22. Eigenvalue calculation and storage

For every equilibrium-valid finite Jacobian:

```text
eigenvalues = np.linalg.eigvals(F)
```

Store the complete complex spectrum in the raw NPZ artifact.

Per sample save:

```text
eigenvalue_real
eigenvalue_imag
eigenvalue_magnitude
spectral_radius
spectral_margin
stable
unstable_pole_count
marginal_pole_count
equilibrium_valid
numerical_valid
```

Do not attempt to assign a persistent identity to every complex pole across Monte Carlo samples in v1.

Mode tracking is a separate problem.

The summary should therefore focus on permutation-invariant statistics:

```text
spectral radius
stable/unstable sample fraction
number of unstable poles
largest few pole magnitudes
```

rather than "pole 7 mean" across samples.

---

# 23. Optional continuous-equivalent pole diagnostic

For interpretation only, nonzero discrete poles may be mapped by:

```text
s = log(z) / dt
```

using the principal complex logarithm.

This is not the stability criterion and should not be required for v1 output.

Zero discrete poles make this transform undefined and must remain explicitly undefined rather than being clipped.

---

# 24. Input contract for one case

The single-case CLI should accept:

```text
--result
--arrays
--static-postprocess
--arguments-json
--bag-json
--controller-yaml
--vehicle-model
--covariance-mode
--samples
--seed
--delay-mode
--controller-dt              optional override
--fd-check-samples
--output-dir
```

Recommended choices:

```text
covariance-mode:
    conservative_fusion
    overlap_corrected

delay-mode:
    fitted_thrust_delay
    zero_thrust_delay
```

The result and `arrays.npz` must come from the same completed prior-free estimator case.

The sibling `arguments.json` supplies the actuator settings used in the physical estimator.

`bag-json` supplies the exact bag path and selected time interval and is used to recover recorded command timing through the audited ROS adapter.

The static postprocess supplies the exact recorded gains and controller provenance.

---

# 25. Current three production cases

## failure1

Estimator:

```text
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_1_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/
```

Use:

```text
result.json
arrays.npz
arguments.json
```

Static PID snapshot:

```text
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/
gimbalrotor_pid_postprocess/
single_rosbag_1_prior_free_static_pid_production_20260817/
pid_gain_postprocess.json
```

Bag JSON:

```text
minimal/bag_jsons/single_rosbag_1.json
```

Outcome:

```text
crashed
```

## failure2

Estimator:

```text
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_2_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/
```

Static PID snapshot:

```text
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/
gimbalrotor_pid_postprocess/
single_rosbag_2_prior_free_static_pid_production_20260817/
pid_gain_postprocess.json
```

Bag JSON:

```text
minimal/bag_jsons/single_rosbag_2.json
```

Outcome:

```text
crashed
```

The current selected interval is:

```text
start_seconds = 25.5
end_seconds   = 31.0
```

## success

Estimator:

```text
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/
```

Static PID snapshot:

```text
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/
gimbalrotor_pid_postprocess/
single_rosbag_succeeded_prior_free_static_pid_production_20260817/
pid_gain_postprocess.json
```

Bag JSON:

```text
minimal/bag_jsons/single_rosbag_succeeded.json
```

Outcome:

```text
successful
```

---

# 26. Single-case CLI example

Run from:

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
```

Example for failure2:

```bash
python3 minimal/gimbalrotor_pid_local_pole_validation.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --arrays minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz \
  --arguments-json minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arguments.json \
  --static-postprocess minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_2_prior_free_static_pid_production_20260817/pid_gain_postprocess.json \
  --bag-json minimal/bag_jsons/single_rosbag_2.json \
  --controller-yaml /home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml \
  --vehicle-model minimal/grape_vehicle_model.json \
  --covariance-mode conservative_fusion \
  --samples 512 \
  --seed 0 \
  --delay-mode fitted_thrust_delay \
  --output-dir /tmp/grape-local-poles-failure2
```

The script must also work with:

```bash
--covariance-mode overlap_corrected
--delay-mode zero_thrust_delay
```

---

# 27. Three-bag wrapper

Implement:

```text
minimal/three_bag_gimbalrotor_pid_local_pole_validation.py
```

as a thin orchestration layer.

Its current production default should run:

```text
3 bags
x 2 covariance modes
x 2 delay modes
```

giving 12 independent case analyses:

```text
failure1 / conservative / fitted
failure1 / conservative / zero
failure1 / overlap      / fitted
failure1 / overlap      / zero

failure2 / ...
success  / ...
```

The wrapper must not average the three bag distributions.

The wrapper may contain the current production relative paths because it is a current-experiment convenience script; the single-case implementation remains generic.

---

# 28. Output namespace and provenance

After implementation is committed, define:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
OUT=minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_local_pole_validation
```

Production results belong under the **implementation source commit**, not under the plan base commit.

The top-level result must record:

```text
source_commit
plan_base_commit = aba27b2e51efab80271aa6cd94cd8e521a3a2efd
estimator_source_commit
static_postprocess_source_commit
controller_yaml_sha256
vehicle_model path
bag path
bag interval
flight_outcome
covariance_mode
delay_mode
fitted_rotor_lag_seconds
controller_dt
controller timing statistics
sample_count
seed
recorded PID gains
```

---

# 29. Per-case output files

Write:

```text
local_pole_validation.json
local_pole_validation.md
local_pole_samples.npz
status.json
```

No YAML gain proposal is written by this diagnostic.

## 29.1 `local_pole_validation.json`

Required sections:

```text
schema
method
source_commit
input
flight_outcome
controller
plant_distribution
delay_model
controller_timing
center_result
sampling
stability_distribution
finite_difference_diagnostics
warnings
```

## 29.2 `local_pole_samples.npz`

Store at least:

```text
quotient_delta_samples
scale_free_samples

trim_integral
trim_gimbal
trim_issued_thrust
trim_issued_gimbal
trim_actual_thrust
trim_actual_gimbal
trim_residual_norm
trim_one_step_defect
equilibrium_valid

eigenvalue_real
eigenvalue_imag
eigenvalue_magnitude
spectral_radius
spectral_margin
stable
unstable_pole_count
marginal_pole_count

numerical_valid
```

If the augmented dimension differs across cases due to different fitted delay, that is fine because each case has its own NPZ.

## 29.3 `status.json`

Report:

```text
requested_samples
numerical_valid_samples
equilibrium_valid_samples
pole_valid_samples
numerical_failure_count
trim_unresolved_count
warnings
status
```

Do not hide invalid samples by reporting only the post-filtered sample count.

---

# 30. Per-case summary metrics

For `spectral_radius` and `spectral_margin`, report empirical:

```text
mean
standard deviation
min
max
q025
q16
q50
q84
q975
```

Report:

```text
stable_fraction_among_pole_valid
unstable_fraction_among_pole_valid
pole_valid_fraction_of_requested
```

and the histogram:

```text
unstable_pole_count -> number of samples
```

Also report center-plant quantities separately:

```text
center_spectral_radius
center_spectral_margin
center_stable
center_eigenvalues
```

The center is the estimator point estimate, not a Monte Carlo median.

---

# 31. Three-bag summary

Write:

```text
three_bag_local_pole_summary.json
three_bag_local_pole_summary.md
```

The primary table should contain:

```text
case
actual outcome
covariance mode
delay mode
recorded roll/pitch P/I/D
fitted thrust delay
controller dt
requested samples
pole-valid samples
stable fraction
center spectral radius
median spectral radius
16-84% spectral-radius range
2.5-97.5% spectral-radius range
median unstable-pole count
```

Also make a focused comparison:

```text
fitted-delay result
versus
zero-delay result
```

for each bag.

This directly tests whether the fitted thrust delay materially changes the stability diagnosis.

---

# 32. Interpretation of the delay ablation

The delay comparison should be read as follows.

If:

```text
fitted delay -> substantially larger spectral radius / lower stable fraction
zero delay   -> substantially more stable
```

then the inferred thrust delay contributes meaningfully to the local closed-loop instability.

If both delay modes are similar, then the local result is primarily controlled by:

```text
mass/inertia/CoG/effectiveness
MIMO allocation
PID gains
```

rather than the pure thrust delay.

Do not attribute causality from the delay ablation alone; report the numerical contrast.

---

# 33. Expected scientific interpretation, without forcing an expected result

The implementation is successful if it computes the requested model faithfully.

It is scientifically interesting if, for example:

```text
success:
    narrow distribution with spectral radius below 1

failure1:
    larger radius / mixed stability

failure2:
    broad distribution or substantial mass above 1
```

but these are hypotheses, not acceptance criteria.

In particular, failure2's short, weakly-excited crash segment may yield a broad plant distribution and therefore a broad pole distribution.

Failure1's longer roll oscillation may have constrained rotational plant directions more strongly.

The program must preserve whichever result the model actually gives.

---

# 34. If all crashed-flight samples remain locally stable

If failure1 and failure2 are locally stable under nearly all samples, the next scientific step is not to change the stability criterion.

The correct conclusion is:

```text
hover-local linear stability does not explain the observed crashes.
```

Possible next analyses, outside v1, are:

```text
local linearization along the recorded operating trajectory
nonlinear closed-loop forecast
actuator saturation
gimbal-rate saturation
large-angle attitude dynamics
state-dependent allocation
integral-state transient
time-varying sampled-data analysis
```

Do not implement these automatically in the v1 script.

---

# 35. Future gain-design stage — explicitly out of scope

Only after the recorded-gain validation is understood should a future tool consider:

```text
K -> stable fraction across plant samples
```

or search for a region such as:

```text
{K : empirical stable fraction >= target}
```

That future stage may produce a PID candidate distribution or robust-stability region.

The current script must **not**:

```text
sample new PID gains
optimize PID gains
rank candidate gains
write controller YAML
recommend deployment
```

Its sole controller input is the exact recorded gain used in the corresponding flight.

---

# 36. Required tests

Implement at least the following tests.

## 36.1 Input/provenance tests

1. `result.json` and `arrays.npz` must agree on the estimator case.
2. The static PID artifact must refer to the same estimator source/case.
3. The recorded controller gain source must be retained.
4. The current controller YAML SHA must be checked against the audited artifact.
5. The recorded gains must replace `ControllerConfig.grape()` template gains exactly.

## 36.2 Scale-free plant tests

6. The quotient center decodes to the estimator's saved scale-free point.
7. Nominal-mass-gauge construction reproduces the saved scale-free ratios.
8. With zero drag, multiplying mass/inertia/force effectiveness by a common positive scale leaves the local rigid-body acceleration and local pole result unchanged to numerical precision.

## 36.3 Trim tests

9. Nominal symmetric plant + nominal controller admits a hover trim near the known gravity-support solution.
10. The solved trim has zero/machine-scale real rigid-body acceleration.
11. The solved steady gimbal is a fixed point of the issued gimbal command.
12. The controller integral state is unchanged over one zero-error controller tick.
13. Filling the thrust delay queue with the steady issued thrust leaves the trim unchanged for nonzero pure delay.
14. Delay changes the local Jacobian but not the constant equilibrium.

## 36.4 Delay tests

15. `tau = 0` uses the current issued thrust for the full interval.
16. `tau = m * dt` uses exactly `c[k-m]` for the full interval.
17. `tau = m * dt + r` uses `c[k-m-1]` for `r` and `c[k-m]` for `dt-r`.
18. When `m = 0` and `r > 0`, the second segment uses the current issued thrust.
19. The delay queue stores thrust only; changing a queued value cannot directly change the gimbal target.
20. Queue shift behavior matches a hand-constructed ZOH command sequence.

## 36.5 Forward-map tests

21. Local orientation encode/decode round-trips with right-tangent perturbations.
22. `local_closed_loop_step(0)` reproduces the stored trim defect.
23. A complete augmented state forward step remains finite for the nominal trim.
24. Zero-delay and fitted-delay maps use identical controller commands at the same instantaneous state; only delayed thrust application differs.

## 36.6 Linearization tests

25. Central finite difference with `h` and `h/2` agrees on a smooth nominal test case.
26. The discrete Jacobian from the augmented map matches a brute-force independently coded finite difference on a small synthetic no-delay case.
27. No sample is rejected solely because a matrix condition number is large.
28. No stability epsilon is inserted: a synthetic eigenvalue at magnitude `<1`, `==1`, and `>1` is classified exactly as stable, marginal, unstable.
29. Unexpected `ValueError` is not swallowed as an ordinary invalid sample.

## 36.7 Simplified-model regression

30. Construct a scalar/double-integrator synthetic case with no delay, no coupling, and no saturation.
31. Change only static effectiveness by factor `b/b0`.
32. Verify that scaling `P/I/D` by `b0/b` preserves the synthetic closed-loop poles.

This test provides the exact mathematical bridge between the existing static equal-scaling method and the new pole calculation.

## 36.8 Monte Carlo tests

33. Zero covariance gives a point-mass pole distribution.
34. Fixed seed gives bitwise/reproducible quotient draws.
35. `conservative_fusion` and `overlap_corrected` are both accepted.
36. Fitted delay is held fixed across physical Monte Carlo samples in v1.
37. No lag variance is fabricated.
38. Raw invalid samples remain represented in output masks/counts.
39. Prefix summaries at 128/256/512 use the same ordered sample realization.

## 36.9 Scientific-label tests

40. Changing the metadata label from `failure` to `success` must not change any pole.
41. The program never changes stability classification based on case name.
42. The three-bag wrapper keeps all three distributions separate.

---

# 37. Numerical policy

This project has an explicit numerical policy:

> Do not stop a calculation merely because an intermediate value looks unusual, ill-conditioned, or physically surprising. Continue until the requested mathematical operation actually becomes undefined or non-finite.

Apply that policy here.

Do not reject because:

```text
spectral radius is huge
condition number is huge
a pole is far outside the unit circle
a sampled inertia is extreme but representable
a trim integral is large but finite
a gimbal trim is unusual but representable by the actual forward model
```

Do not add:

```text
large residual replacement
gain clipping
condition-number cutoff as sample rejection
pole-radius clipping
silent covariance shrinkage
```

If a requested operation genuinely becomes non-finite, retain:

```text
sample index
stage
exception type
message
```

and continue the Monte Carlo only for narrowly classified numerical failures.

Programming/configuration errors must fail loudly.

---

# 38. Production run sequence

The intended repository workflow is:

```text
1. apply implementation
2. run unit tests
3. commit implementation
4. SOURCE_COMMIT=$(git rev-parse HEAD)
5. run 3-bag production analysis
6. write results under minimal/outputs/${SOURCE_COMMIT}/...
7. inspect results
8. commit production results
9. git push origin HEAD
```

Do not put production results under:

```text
aba27b2e...
02c53cf...
```

unless either is actually the implementation source commit.

The result namespace must identify the code that generated it.

---

# 39. Recommended initial production command

The three-bag wrapper should provide a current default equivalent to:

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim

SOURCE_COMMIT=$(git rev-parse HEAD)
OUT=minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_local_pole_validation

python3 minimal/three_bag_gimbalrotor_pid_local_pole_validation.py \
  --controller-yaml /home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml \
  --vehicle-model minimal/grape_vehicle_model.json \
  --samples 512 \
  --seed 0 \
  --output-dir "${OUT}"
```

By default the wrapper should run:

```text
conservative_fusion
overlap_corrected
```

and:

```text
fitted_thrust_delay
zero_thrust_delay
```

for all three bags.

Provide CLI switches to restrict either dimension for debugging.

---

# 40. First result inspection checklist

After the first production run, inspect in this order.

### 40.1 Trim

For each bag/covariance/delay case:

```text
center trim residual
center one-step defect
trim integral
trim gimbal
trim thrust
actuator saturation/kink diagnostics
```

Verify that delay mode does not materially change the equilibrium itself.

### 40.2 Recorded-gain center poles

Before interpreting Monte Carlo:

```text
failure1 center spectral radius
failure2 center spectral radius
success center spectral radius
```

Compare fitted-delay and zero-delay variants.

### 40.3 Monte Carlo validity

Inspect:

```text
requested sample count
numerical valid count
equilibrium valid count
pole valid count
```

Do not inspect only the post-filtered distribution.

### 40.4 Stability distribution

Compare:

```text
stable fraction
spectral-radius median
16-84%
2.5-97.5%
unstable-pole-count histogram
```

### 40.5 Failure2 uncertainty

Check whether failure2 is substantially broader than success and whether the fitted delay shifts its pole distribution.

### 40.6 Covariance construction

Compare `conservative_fusion` and `overlap_corrected`.

A conclusion that appears only under one covariance construction should be described as covariance-model dependent.

### 40.7 Delay ablation

Compare fitted vs zero delay within each exact same set of physical Monte Carlo draws.

Use the same seed so the physical samples are paired.

---

# 41. Acceptance criteria for the implementation

The implementation is complete when all of the following hold:

- [ ] Base assumptions are relative to repository commit `aba27b2e51efab80271aa6cd94cd8e521a3a2efd`.
- [ ] Existing estimator code is not rewritten.
- [ ] Existing static PID postprocessor is not rewritten.
- [ ] Existing Monte Carlo static PID postprocessor remains intact.
- [ ] New pole analysis is in a new script.
- [ ] Exact recorded gains are used per bag.
- [ ] The controller still uses the nominal controller model.
- [ ] Real plant parameters are sampled in the estimator's native 13-D quotient coordinate.
- [ ] `rotor_lag_seconds` is treated as thrust-only pure ZOH delay.
- [ ] Fitted delay is not reinterpreted as an actuator time constant.
- [ ] Gimbal command is not given the fitted thrust delay.
- [ ] Lag covariance is not fabricated.
- [ ] Fitted-delay and zero-delay analyses are both available.
- [ ] An actual sampled-plant hover equilibrium is solved before pole classification.
- [ ] Local state includes rigid body, PID integral, actuator state, and exact thrust-delay queue.
- [ ] Orientation uses a right-tangent local chart.
- [ ] The primary Jacobian is the finite-difference derivative of the actual augmented one-step forward map.
- [ ] Stability is classified in discrete time by the unit circle.
- [ ] No outcome label is used by the numerical calculation.
- [ ] No arbitrary rank/condition/gain/pole guard drops finite samples.
- [ ] All complex poles are retained in raw output.
- [ ] Three bag results remain separate.
- [ ] Both covariance modes are run.
- [ ] Full provenance is written.
- [ ] Tests include the simplified equal-PID-scaling pole-preservation regression.
- [ ] A production three-bag summary is generated.
- [ ] Scientific success/failure separation is **not** an implementation acceptance condition.

---

# 42. What this experiment can establish

If the recorded-gain pole distributions track the observed flight outcome, this gives a principled bridge:

```text
estimated physical plant distribution
    -> sampled closed-loop dynamics
    -> local pole distribution
    -> stability evidence
```

This is a stronger controller-level use of the inferred physical distribution than the current static effectiveness scaling.

It also gives `I` gain an appropriate role: `I` is not inferred from rigid-body physics alone. It participates as one state-feedback design parameter in the augmented closed-loop dynamics, and its effect is judged by the resulting pole structure.

If the local poles do not distinguish the crashes, that result is equally informative: it isolates the missing explanation to behavior beyond local hover stability and motivates the next nonlinear/trajectory-level stage without altering the physical estimator.

---

# 43. Explicitly deferred follow-up

After this validation is understood, a separate future plan may define a gain-space map:

```text
K
    -> empirical stable fraction over sampled plants
    -> spectral-radius / damping distribution
```

and then construct a set or distribution of candidate PID gains.

That future work must be a separate implementation and must use the present validation as its model check.

Do not add it to this patch.
