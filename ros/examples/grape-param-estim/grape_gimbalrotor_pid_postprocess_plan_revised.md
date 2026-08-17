# Grape Gimbalrotor PID Gain Postprocess — Revised Implementation Plan

**Target repository:** `hiroya1224/ProbTF-demo`
**Implementation baseline / current pushed HEAD:** `057421223b6ad23f126be2364e7265f808eb33ef`
**Canonical estimator source revision used by the current committed production results:** `916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1`
**Reference Gimbalrotor controller source:** `sugikazu75/jsk_aerial_robot@2786cc3e0c054d1beb8df33205213e4b2c648537`
**Plan date:** 2026-08-17

---

# 0. Revision policy

The mathematical core of that draft is retained:

```text
nominal controller model
        +
identified scale-free real plant
        |
        v
nominal 6x8 controller allocation A_cmd
estimated 6x8 real-plant effectiveness A_real
        |
        v
H = A_real A_cmd^+
        |
        v
dimensionless local effectiveness mismatch
        |
        v
4 PID gain-group scale factors
xy / z / roll_pitch / yaw
```

The revisions in this document are primarily implementation/provenance corrections:

1. use the **actual current repository APIs** instead of recreating already
   implemented Grape controller/allocation logic;
2. explicitly document the **three production rosbag paths and selected
   intervals**;
3. distinguish the current estimator's BODY-coordinate rotor geometry from the
   controller allocation's CoG-relative geometry;
4. use the current `result.json` scale-free contract exactly as committed;
5. use each rosbag's recorded dynamic-reconfigure snapshot as the authoritative
   baseline for gains actually used in flight; use the supplied controller YAML
   only as the proposal-file template and controller-mode contract;
6. write production outputs under a commit-namespaced output directory;
7. require source commit, production-results commit, and `git push origin HEAD`;
8. retain the static method as the interpretable v1 gain proposal, but describe
   the already-existing closed-loop simulator correctly for the later dynamic
   refinement instead of treating it as infrastructure that still needs to be
   written.

Do not change the physical estimator in this task.

---

# 1. Scientific purpose

The physical parameter estimator is now the source of the plant model.

The controller remains the Gimbalrotor controller designed around the nominal
Grape vehicle model.

The intended control-design structure is

```text
reference r
    |
    v
Gimbalrotor controller
  - nominal mass/inertia/geometry
  - existing PID structure
  - tunable PID gains
    |
    v
rotor/gimbal command
    |
    v
identified real plant
  - estimated J/m
  - estimated CoG
  - estimated f_i/m
  - estimated rotor lag
    |
    v
motion
```

The controller's internal nominal model is **not replaced by the estimated
plant** in this task.

The estimated plant is the plant that the nominal-model controller must
control.

This is the original intended use of the physical identification:

```text
failed flight data
    -> plant identification
    -> controller gain correction
    -> next flight experiment
```

---

# 2. Scope of the first implementation

The first implementation is an interpretable **static local Gimbalrotor
effectiveness correction around hover / zero gimbal**.

It computes four gain-group scales:

```text
xy
z
roll_pitch
yaw
```

and multiplies P, I, and D together inside each group.

This v1 calculation is intentionally local and static.

It does **not** claim to solve the full nonlinear closed-loop tuning problem.

The current repository already contains the infrastructure for the later
closed-loop refinement. Section 28 defines how that extension must reuse the
current implementation.

---

# 3. Exact repository baseline

Start from a clean checkout whose HEAD is:

```text
057421223b6ad23f126be2364e7265f808eb33ef
```

The estimator source revision associated with the current production outputs is:

```text
916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1
```

Do not implement against an older estimator contract.

Relevant current files are:

```text
ros/examples/grape-param-estim/minimal/
  single_bag_savgol_estimator.py
  single_bag_savgol_core.py
  single_bag_savgol_reports.py
  grape_vehicle_model.json
  bag_jsons/
    single_rosbag_1.json
    single_rosbag_2.json
    single_rosbag_succeeded.json

ros/examples/grape-param-estim/src/grape_param_estim/
  controller.py
  dynamics.py
  real_rosbag.py
  sensor_models.py
  system.py
```

The new postprocessor is a downstream consumer.

Do not modify estimator loss, covariance, parameterization, gauge treatment,
prior support, SG processing, or current production results.

---

# 4. Exact production bags

The physical estimator's three current production datasets are fixed.

## 4.1 Failure bag 1

Committed bag JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_1.json
```

Absolute bag path:

```text
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/
20260612_grape_hovering_4_2026-06-12-17-33-59.bag
```

Selected interval:

```text
19.0 s <= t <= 25.0 s
```

## 4.2 Failure bag 2

Committed bag JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_2.json
```

Absolute bag path:

```text
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/
20260612_grape_hovering_6_2026-06-12-17-40-34.bag
```

Selected interval:

```text
25.5 s <= t <= 31.0 s
```

## 4.3 Successful bag

Committed bag JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_succeeded.json
```

Absolute bag path:

```text
/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/
20260613_grape_hovering_1_2026-06-13-13-44-01.bag
```

Selected interval:

```text
65.0 s <= t <= 75.0 s
```

The static matrix calculation does not depend on flight signals, but the v1
postprocessor opens each `.bag` through the existing rosbag adapter to recover
the P/I/D gains actually used in that selected interval.

Nevertheless these exact paths and intervals must be written in this plan and
in production provenance so the PID proposal is traceable to the flight from
which the plant was identified.

---

# 5. Exact current prior-free result inputs

Use the current committed **prior-free** point estimates as the default plant
inputs.

Do not use pseudo-conditioning ablation cases by default.

All paths below are relative to:

```text
ros/examples/grape-param-estim/
```

## Failure bag 1

```text
minimal/outputs/
916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_1_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/result.json
```

## Failure bag 2

```text
minimal/outputs/
916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_2_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/result.json
```

## Successful bag

```text
minimal/outputs/
916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/
prior_ablation/
single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/
cases/prior_free/result.json
```

These production files were generated from source commit:

```text
916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1
```

and committed in repository commit:

```text
057421223b6ad23f126be2364e7265f808eb33ef
```

---

# 6. Which estimate is the plant source of truth?

The ordinary default is:

```text
case_name == "prior_free"
```

This is deliberate.

The current estimator works without a physical prior, so the controller
postprocessor must not make pseudo-conditioning an implicit prerequisite.

A non-prior-free result may be accepted only when the user explicitly requests
it, for example:

```text
--allow-non-prior-free-result
```

When such a result is used, record:

```text
case_name
prior.active
prior.name
prior.role
prior source
```

in the output report.

Never silently substitute a pseudo-conditioned result for the prior-free plant.

---

# 7. Estimator result contract

The postprocessor reads the ordinary estimator `result.json`.

Required status fields:

```python
payload["overall_case_status"]
payload["optimization_status"]
payload["success"]
payload["case_name"]
payload["source_commit"]
```

Accept the point estimate by default only if:

```python
payload["overall_case_status"] == "completed"
payload["optimization_status"] == "completed"
payload["success"] is True
```

A result with:

```text
overall_case_status == "point_estimate_completed"
```

may be accepted only with an explicit override, because it indicates that the
optimizer completed but a post-fit uncertainty product failed.

If accepted, emit:

```text
postfit_uncertainty_unavailable
```

as a warning.

The physical interface is:

```python
payload["parameters"]["scale_free"]["inertia_over_mass_m2"]
payload["parameters"]["scale_free"]["cog_position_body_m"]
payload["parameters"]["scale_free"]["force_effectiveness_over_mass"]
payload["parameters"]["rotor_lag_seconds"]
```

Use no other estimated physical quantities in the v1 plant effectiveness
calculation.

---

# 8. Exact scale-free rule

The estimator has the exact common scale gauge:

```text
(m, J, f_1, ..., f_4)
    ->
(lambda m, lambda J, lambda f_1, ..., lambda f_4).
```

The controller postprocessor therefore uses only:

```math
J_m = J/m,
```

```math
c = c_{\rm CoG},
```

```math
f_m = f/m.
```

Never use:

```python
payload["parameters"]["estimated"]["mass_kg"]
payload["parameters"]["estimated"]["inertia_kg_m2"]
payload["parameters"]["estimated"]["force_effectiveness"]
```

to construct `A_real`.

Two result files with identical scale-free quantities but different absolute
gauge representatives must generate the same static PID proposal.

This requires a dedicated test.

---

# 9. Current controller implementation that must be reused

The current repository already contains a Grape controller port:

```text
src/grape_param_estim/controller.py
```

It contains:

```text
PIDConfig
ControllerConfig
GrapeController
acceleration_allocation_matrix(...)
source-compatible controller pseudoinverse semantics
PID clamps
yaw D-control behavior
nonnegative z integral
roll/pitch integration activation
current-gimbal allocation support
```

Do not create another independent controller model for v1.

The existing allocation implementation is the source of truth for the nominal
controller allocation.

The current source-compatible controller pseudoinverse uses the aerial-robot
absolute SVD threshold:

```text
singular value > 1e-4
```

This must be matched exactly in the v1 postprocessor.

Because the current helper in `controller.py` is private (`_source_pseudoinverse`),
the production postprocessor may implement a small local equivalent, but it
must have a parity unit test against the current controller implementation.

Do not use the default cutoff of `np.linalg.pinv()`.

---

# 10. Current controller configuration sources

There are three distinct sources and they have different roles.

## 10.1 Actual YAML proposal template

Reference repository file:

```text
sugikazu75/jsk_aerial_robot
robots/gimbalrotor/config/grape/GimbalrotorControl.yaml
```

Reference source revision:

```text
2786cc3e0c054d1beb8df33205213e4b2c648537
```

Expected local catkin checkout for production:

```text
/home/leus/catkin_ws/src/jsk_aerial_robot/
robots/gimbalrotor/config/grape/GimbalrotorControl.yaml
```

Do not hard-code this absolute path into the reusable math module.

The CLI receives:

```text
--controller-yaml PATH
```

and production commands must pass the actual local path explicitly.

If the expected checkout does not exist, fail clearly.

Do not silently fall back to a checked-in fixture.

The supplied YAML is authoritative for the output file structure and
controller-mode settings. It is not authoritative for the P/I/D values used in
the recorded flight. The 12 proposal leaves are replaced with scaled values
whose baseline comes from the bag snapshot.

Reference values are:

```yaml
controller:
  gimbal_calc_in_fc: false

  xy:
    p_gain: 4.0
    i_gain: 0.1
    d_gain: 2.0

  z:
    p_gain: 5.0
    i_gain: 1.0
    d_gain: 2.5

  roll_pitch:
    p_gain: 13.0
    i_gain: 1.0
    d_gain: 20.0

  yaw:
    p_gain: 6.0
    i_gain: 1.0
    d_gain: 2.0
    need_d_control: true
```

## 10.2 `ControllerConfig.grape()`

The current Python port contains a hard-coded snapshot matching the reference
Grape gains and limits.

Use it as a parity/reference implementation.

Do **not** make it the production source of the flight-time gains.

If the supplied YAML gains differ from the bag-recorded gains, apply the scale
to the bag-recorded values, write those proposals into the output YAML copy,
and report the difference explicitly.

## 10.3 Controller snapshot recorded in each bag

The current rosbag adapter already parses:

```text
/gimbalrotor/controller/xy/parameter_updates
/gimbalrotor/controller/z/parameter_updates
/gimbalrotor/controller/roll_pitch/parameter_updates
/gimbalrotor/controller/yaw/parameter_updates
```

and exposes:

```text
FlightData.controller_snapshot
FlightData.controller_configuration
```

These are required by v1 and are the sole source of its current P/I/D baseline.
Call the existing rosbag adapter and use `FlightData.controller_snapshot`; do
not invent another gain-history parser. Record gains, event record times,
`pid_control_flags`, and `source_kinds` in the result.

The current production snapshots are:

```text
failure1: xy=[3,0.1,1], z=[5,1,2.5], roll_pitch=[20,1,8], yaw=[4,1,2]
failure2: xy=[3,0.1,1], z=[5,1,2.5], roll_pitch=[10,1,8], yaw=[4,1,2]
success:  xy=[4,0.1,2], z=[5,1,2.5], roll_pitch=[13,1,20], yaw=[6,1,2]
```

---

# 11. Controller-mode validation: correct source semantics

The Gimbalrotor C++ source resolves:

```text
gimbal_dof          default 1
gimbal_calc_in_fc   default true
hovering_approximate default false
underactuate        default false
```

The Grape YAML explicitly sets:

```text
gimbal_calc_in_fc = false
```

but the reference YAML does not explicitly contain `gimbal_dof` or
`underactuate`.

Therefore the postprocessor must not require those keys to exist in the YAML.

Resolve them using source defaults when absent:

```text
gimbal_dof = 1
underactuate = false
```

If the supplied YAML explicitly overrides them, respect the explicit value and
reject unsupported paths.

v1 requires:

```text
gimbal_dof == 1
underactuate == false
gimbal_calc_in_fc == false
yaw.need_d_control == true
```

`hovering_approximate` is irrelevant when `underactuate == false`, but report
its resolved value if present.

---

# 12. Critical geometry distinction in the current code

This is the most important implementation detail.

The current estimator vehicle-model JSON stores:

```text
geometry.rotor_origins_body_m
cog_position_body_m
```

as separate BODY-frame quantities.

For the nominal model:

```text
rotor origin in BODY
    !=
rotor origin from CoG
```

The current plant dynamics follows this convention.

`actuator_wrench(...)` computes:

```python
origin_from_cog =
    geometry.thrust_origins(gimbal_angle)[rotor]
    - parameters.cog_offset
```

so the `geometry` passed through `load_vehicle_model()` is correctly BODY
referenced for plant dynamics.

By contrast, the controller allocation API:

```python
acceleration_allocation_matrix(
    parameters,
    geometry,
    gimbal_angles,
)
```

expects `geometry.rotor_origins` to already be expressed about the controller
snapshot's CoG.

`GrapeGeometry.grape()` follows that controller convention.

Therefore:

> **Never pass `load_vehicle_model(...).geometry` directly to
> `acceleration_allocation_matrix(...)`.**

Construct the controller-snapshot geometry explicitly:

```python
controller_rotor_origins_from_cog = (
    rotor_origins_body_m - nominal_cog_body_m[None, :]
)
```

and create a `GrapeGeometry` using those CoG-relative origins.

This must have a regression test.

---

# 13. Nominal vehicle model

Production nominal model:

```text
minimal/grape_vehicle_model.json
```

Current nominal values:

```text
mass =
2.3515975908123767 kg
```

```text
J =
[[ 0.065000061483315,  -7.27899253e-7,   1.9015080033e-5 ],
 [ -7.27899253e-7,     0.064952656340165, 5.9167305e-8   ],
 [ 1.9015080033e-5,    5.9167305e-8,     0.128992110664428 ]]
kg m^2
```

```text
CoG_body =
[-0.002024708562282,
 -0.000030526578941,
  0.009509749599446]
m
```

```text
force_effectiveness =
[1, 1, 1, 1]
```

```text
torque_effectiveness =
[1, 1, 1, 1]
```

BODY-coordinate thrust-link origins at zero gimbal:

```text
[[-0.22309, -0.22309, 0.056],
 [ 0.22309, -0.22309, 0.056],
 [ 0.22309,  0.22309, 0.056],
 [-0.22309,  0.22309, 0.056]]
m
```

Arm yaws:

```text
[-2.3562, -0.7854, 0.7854, 2.3562]
rad
```

Rotor directions:

```text
[-1, +1, -1, +1]
```

Moment-force rate:

```text
-0.0181 m
```

Thrust offset:

```text
0.056 m
```

Use `load_vehicle_model()` or the same current JSON contract to read these.

Do not duplicate production numeric constants in the new code.

---

# 14. Static linearization point

v1 is evaluated at:

```text
gimbal angles = [0, 0, 0, 0] rad
body angular velocity = 0
unsaturated PID region
```

The zero-angular-velocity assumption is important because the Gimbalrotor
source contains the source-compatible gyro term.

At hover:

```math
\omega = 0
```

so that term vanishes.

No ad-hoc static correction for rotor lag is introduced.

---

# 15. Nominal controller acceleration allocation `A_cmd`

Define:

```math
A_{\rm cmd}\in\mathbb R^{6\times8}.
```

It maps the controller's 8-D virtual vectoring force to desired generalized
acceleration under the nominal controller model.

For production, construct:

1. nominal mass and inertia from `grape_vehicle_model.json`;
2. controller CoG-relative geometry from
   `rotor_origins_body_m - nominal_cog`;
3. zero gimbal angles;
4. call the **existing**:

```python
grape_param_estim.controller.acceleration_allocation_matrix(...)
```

Do not independently rewrite the nominal allocation as the production source.

A direct formula may exist in tests as an independent parity check.

The resulting 6x8 matrix must have row rank 6 under the source-compatible
absolute `1e-4` SVD rule.

---

# 16. Source-compatible pseudoinverse

For v1 define:

```math
A_{\rm cmd}^{+}
```

using the same semantics as `aerial_robot_model::pseudoinverse` represented by
the current Python controller port.

Equivalent implementation:

```python
def source_compatible_pseudoinverse(matrix: np.ndarray) -> np.ndarray:
    u, singular_values, vt = np.linalg.svd(
        np.asarray(matrix, dtype=float),
        full_matrices=False,
    )
    inverse = np.asarray([
        1.0 / value if value > 1.0e-4 else 0.0
        for value in singular_values
    ])
    return vt.T @ np.diag(inverse) @ u.T
```

Unit-test it against the existing controller port.

Do not use a relative NumPy default threshold.

---

# 17. Estimated real-plant scale-free allocation `A_real`

Define:

```math
A_{\rm real}\in\mathbb R^{6\times8}.
```

It describes the local plant response to the same virtual vectoring-force
coordinates used by the controller.

Read:

```python
Jm = result["parameters"]["scale_free"]["inertia_over_mass_m2"]
cog = result["parameters"]["scale_free"]["cog_position_body_m"]
fm = result["parameters"]["scale_free"]["force_effectiveness_over_mass"]
```

Use fixed vehicle-model:

```text
torque_effectiveness
rotor_origins_body_m
arm_yaws_rad
rotor_directions
moment_force_rate_m
```

For each rotor, the local virtual-force basis at zero gimbal is:

```python
local_basis = (
    np.asarray((0.0, 1.0, 0.0)),  # lateral
    np.asarray((0.0, 0.0, 1.0)),  # axial
)
```

Rotate the basis about the arm yaw:

```python
d = rotate_z(local_force, arm_yaw)
```

Use estimated CoG:

```python
r = rotor_origin_body - cog
```

For rotor `i` and force-direction column `d`:

```python
tau_per_unit_force = (
    np.cross(r, d)
    + torque_effectiveness[i]
      * rotor_directions[i]
      * moment_force_rate
      * d
)
```

Then:

```python
A_real[:3, col] = fm[i] * d
```

and:

```python
A_real[3:, col] = np.linalg.solve(
    Jm,
    fm[i] * tau_per_unit_force,
)
```

This is exactly common-scale invariant.

No estimated absolute mass is required.

No estimated absolute inertia is required.

No estimated absolute force effectiveness is required.

---

# 18. Why `f_i/m` multiplies both virtual-force columns

The C++ controller's one-DoF vectoring representation uses, per rotor:

```text
[lateral virtual force, axial virtual force]
```

and converts that 2-vector to:

```text
thrust magnitude
gimbal angle
```

The fitted physical `force_effectiveness_i` multiplies the resulting rotor force
magnitude.

At the local force-map level, it therefore scales both virtual-force basis
components for that rotor.

Use one `fm[i]` for both columns of rotor `i`.

---

# 19. Torque-effectiveness assumption

The current physical estimator does not estimate torque effectiveness.

Use:

```python
vehicle_model.parameters.torque_effectiveness
```

for `A_real`.

Record this explicitly:

```text
torque_effectiveness_source =
    fixed_nominal_vehicle_model
```

in the report.

Do not silently treat it as estimated.

---

# 20. Effective controller-to-plant mismatch

The controller asks for generalized acceleration:

```math
\nu_{\rm cmd}\in\mathbb R^6.
```

The nominal allocation produces:

```math
q=A_{\rm cmd}^{+}\nu_{\rm cmd}.
```

The estimated real plant produces:

```math
\nu_{\rm real}=A_{\rm real}q.
```

Therefore:

```math
\boxed{
H=A_{\rm real}A_{\rm cmd}^{+}
}
```

and:

```math
\nu_{\rm real}=H\nu_{\rm cmd}.
```

Identity test:

```math
A_{\rm real}=A_{\rm cmd}
\quad\Rightarrow\quad
H=I_6
```

up to the source-compatible numerical pseudoinverse convention.

This is the primary implementation invariant.

---

# 21. Mixed-unit normalization

The generalized acceleration contains:

```text
linear acceleration   [m/s^2]
angular acceleration  [1/s^2]
```

Use nominal radius-of-gyration characteristic length:

```math
\ell
=
\sqrt{
\frac{1}{3}\operatorname{tr}(J_0/m_0)
}.
```

For the current nominal model:

```text
ell ~= 0.1915849943 m
```

Define:

```math
S=
\operatorname{diag}(1,1,1,\ell,\ell,\ell)
```

and:

```math
\boxed{
\bar H=S H S^{-1}.
}
```

All scalar fitting and cross-block coupling norms use `H_bar`.

Keep raw `H` in the report.

Allow:

```text
--characteristic-length METERS
```

as an explicit diagnostic override.

Default to the nominal radius-of-gyration expression.

---

# 22. Four PID gain-group scales

The controller feedback gains are grouped exactly as in the actual Gimbalrotor
configuration:

```python
GAIN_GROUPS = {
    "xy": (0, 1),
    "z": (2,),
    "roll_pitch": (3, 4),
    "yaw": (5,),
}
```

Represent gain correction by:

```math
D(s)=
\operatorname{diag}
(s_{xy},s_{xy},s_z,s_{rp},s_{rp},s_{yaw}).
```

Fit:

```math
\min_s \|\bar H D(s)-I\|_F^2.
```

For one group `G`:

```math
\boxed{
s_G
=
\frac{
\sum_{j\in G}\bar H_{jj}
}{
\sum_{j\in G}\|\bar H_{:j}\|_2^2
}.
}
```

Reject:

```text
nonfinite scale
scale <= 0
numerically zero denominator
```

Do not silently clamp a mathematically valid scale.

---

# 23. PID transformation

For each group:

```text
P_new = scale * P_old
I_new = scale * I_old
D_new = scale * D_old
```

This keeps the existing P:I:D ratio within the group.

The rationale is that the v1 correction represents a static multiplicative
mismatch on the complete feedback acceleration channel.

Do not:

```text
change P:I:D ratio
change feed-forward acceleration
change any controller limit
change error limits
change need_d_control
change gimbal_calc_in_fc
change publication settings
```

Dynamic P/I/D ratio retuning is not part of the static v1 derivation.

---

# 24. YAML handling

Use semantic YAML parsing.

Required gain leaves:

```text
controller.xy.p_gain
controller.xy.i_gain
controller.xy.d_gain

controller.z.p_gain
controller.z.i_gain
controller.z.d_gain

controller.roll_pitch.p_gain
controller.roll_pitch.i_gain
controller.roll_pitch.d_gain

controller.yaw.p_gain
controller.yaw.i_gain
controller.yaw.d_gain
```

Everything else must remain semantically identical.

Write:

```text
pid_gain_postprocess.json
pid_gain_overlay.yaml
GimbalrotorControl.pid-proposal.yaml
```

Do not overwrite the source YAML.

The full proposed YAML may lose original formatting/comments if PyYAML is used,
but a test must prove that the parsed semantic tree differs only at the 12 gain
leaves.

---

# 25. Proposed implementation structure

Use a reusable core module plus a thin CLI.

Recommended:

```text
ros/examples/grape-param-estim/src/grape_param_estim/
  gimbalrotor_pid_postprocess.py

ros/examples/grape-param-estim/minimal/
  gimbalrotor_pid_postprocess.py
  three_bag_gimbalrotor_pid_postprocess_summary.py

ros/examples/grape-param-estim/minimal/tests/
  test_gimbalrotor_pid_postprocess.py
```

The core module should own:

```text
data classes
result parsing
vehicle-model adaptation
controller-YAML parsing
nominal allocation construction
real scale-free allocation construction
H/H_bar
gain-scale calculation
YAML transformation
report object
```

The CLI should own:

```text
argument parsing
filesystem I/O
terminal summary
exit codes
```

Do not put this logic into `single_bag_savgol_core.py`.

---

# 26. Recommended Python contracts

```python
@dataclass(frozen=True)
class ScaleFreePlant:
    inertia_over_mass: np.ndarray       # 3x3
    cog_position_body: np.ndarray       # 3
    force_effectiveness_over_mass: np.ndarray  # 4
    rotor_lag_seconds: float


@dataclass(frozen=True)
class ControllerGainGroup:
    p_gain: float
    i_gain: float
    d_gain: float


@dataclass(frozen=True)
class AllocationDiagnostics:
    matrix: np.ndarray
    singular_values: np.ndarray
    source_threshold_rank: int
    condition_number: float


@dataclass(frozen=True)
class GainCorrection:
    group: str
    axes: tuple[int, ...]
    scale: float
    old: ControllerGainGroup
    proposed: ControllerGainGroup
    error_before: float
    error_after: float
```

Recommended reusable functions:

```python
load_estimator_result(...)
load_scale_free_plant(...)
load_controller_yaml(...)
resolve_controller_mode(...)
build_controller_snapshot_geometry(...)
build_nominal_controller_allocation(...)
build_real_scale_free_allocation(...)
source_compatible_pseudoinverse(...)
characteristic_length(...)
dimensionless_effectiveness(...)
group_scale(...)
calculate_gain_corrections(...)
apply_gain_corrections_to_yaml(...)
build_report(...)
```

Use ASCII identifiers.

---

# 27. Diagnostics and warnings

Report at least:

## Provenance

```text
postprocessor source commit
estimator result path
estimator source commit
estimator case
bag JSON path
absolute bag path
selected bag interval
vehicle-model path
controller-YAML path
controller-YAML SHA256
controller gain source = rosbag recorded dynamic reconfigure
recorded P/I/D snapshot
snapshot event record times
snapshot pid_control_flags
snapshot source_kinds
```

## Estimated plant

```text
J/m
CoG
f/m
rotor lag
```

## Nominal controller model

```text
mass
J
J/m
CoG
torque effectiveness
controller CoG-relative rotor origins
arm yaws
rotor directions
moment-force rate
```

## Allocation

```text
A_cmd
A_real
singular values
rank
condition number
H
H_bar
diag(H_bar)
```

## Overall metrics

```python
error_before = np.linalg.norm(H_bar - np.eye(6), ord="fro")
error_after = np.linalg.norm(H_bar @ D - np.eye(6), ord="fro")
```

```python
improvement_fraction = (
    (error_before - error_after) / error_before
    if error_before > 0.0
    else 0.0
)
```

Dimensionless coupling ratio:

```python
coupling_ratio = (
    np.linalg.norm(
        H_bar - np.diag(np.diag(H_bar)),
        ord="fro",
    )
    / np.linalg.norm(H_bar, ord="fro")
)
```

Suggested warnings:

```text
non_prior_free_estimate
postfit_uncertainty_unavailable
actuator_rate_limit_active
actuator_saturation_active
large_static_gain_change
strong_axis_coupling
estimated_allocation_ill_conditioned
static_correction_does_not_cover_rotor_lag
feedforward_not_corrected
controller_reference_values_differ
controller_mode_resolved_from_source_default
```

Suggested review thresholds:

```text
scale < 0.5 or scale > 2.0
    -> large_static_gain_change

coupling_ratio > 0.20
    -> strong_axis_coupling
```

These are review thresholds only.

Do not clamp the numerical result.

---

# 28. Rotor lag

Read:

```python
payload["parameters"]["rotor_lag_seconds"]
```

and report it.

The static matrix `H` does not represent delay.

Do not invent:

```text
D gain correction = function(rotor_lag)
```

or another unsupported algebraic lag rule.

The lag becomes important in the later closed-loop simulation.

---

# 29. Current repository already supports later closed-loop refinement

The old draft described closed-loop simulation as a future infrastructure task.

That is no longer accurate.

The current repository already contains:

```python
grape_param_estim.controller.GrapeController
grape_param_estim.dynamics.FullSixDofPlant
grape_param_estim.dynamics.simulate_closed_loop
```

and the current rosbag adapter already exposes:

```text
FlightData.reference
FlightData.controller_snapshot
FlightData.controller_configuration
FlightData.pose
FlightData.velocity
FlightData.flight_mode
```

Therefore a later dynamic refinement should **reuse these APIs**.

Do not create a second independent closed-loop simulator.

Do not reconstruct controller reference trajectories from PID output manually
when `FlightData.reference` already exists.

Do not infer flight-time gains from the YAML when the bag contains
`controller_snapshot`.

---

# 30. How to instantiate the identified plant for later closed-loop simulation

The estimator identifies:

```text
J/m
f/m
CoG
```

but not the common absolute scale.

For the existing rigid-body model, choose the nominal-mass gauge:

```math
m_{\rm sim}=m_{\rm nominal},
```

```math
J_{\rm sim}=m_{\rm nominal}(J/m)_{\rm estimated},
```

```math
f_{\rm sim}=m_{\rm nominal}(f/m)_{\rm estimated}.
```

Use:

```text
estimated CoG
fixed nominal torque effectiveness
fixed nominal drag convention
```

This is a gauge choice only.

The resulting acceleration response is the same scale-free identified plant.

Set the controller's nominal parameters independently to the nominal vehicle
model.

Do not feed `J_sim`, `f_sim`, or estimated CoG back into the controller's
nominal allocation.

For delay-aware closed-loop simulation, the identified rotor lag should enter
the existing actuator/command-delay representation, not a PID algebraic
formula.

---

# 31. Dynamic refinement progression after v1

After the static v1 calculation is verified:

## Phase 2 — closed-loop validation of current vs static-proposed gains

For each bag:

1. use the actual bag JSON;
2. load `FlightData` with the current rosbag adapter;
3. use `FlightData.reference`;
4. use the recorded controller gain snapshot as the baseline consistency check;
5. initialize state from the selected flight interval;
6. use the nominal controller model;
7. use the identified real plant;
8. run `simulate_closed_loop()` without observation-state resets;
9. compare:
   - recorded/current gains,
   - static v1 proposed gains.

This phase is validation, not yet optimization.

## Phase 3 — four-scale dynamic refinement

Parameterize:

```math
s_G = \exp(\alpha_G)
```

for:

```text
xy
z
roll_pitch
yaw
```

and optimize four `alpha` values using the existing closed-loop simulator.

Use the static v1 scales as the initial point.

Keep P:I:D ratios fixed in this first dynamic refinement.

Only after this works should separate P/D or P/I/D ratio optimization be
considered.

---

# 32. v1 opens the bag only for the controller gain snapshot

The static v1 calculation consumes:

```text
result.json
vehicle model JSON
controller YAML
bag JSON
the selected bag's recorded controller gain snapshot
```

Use the existing `load_flight_data()` adapter and
`FlightData.controller_snapshot`. Do not derive the gains from YAML and do not
create a second dynamic-reconfigure parser. Other flight signals do not enter
the static matrix calculation.

Recommended optional CLI field:

```text
--bag-json PATH
```

For the three production runs, this argument is mandatory.

The CLI loads the bag JSON to obtain:

```text
bag_path
start_seconds
end_seconds
```

and opens that exact bag and interval to reconstruct the effective P/I/D
snapshot. A ROS environment providing `rosbag` is therefore required.

but it must not open the actual rosbag in v1.

This keeps the static postprocessor fast and deterministic while preserving
complete provenance.

---

# 33. CLI

Recommended:

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim

python3 minimal/gimbalrotor_pid_postprocess.py \
  --result <result.json> \
  --bag-json <bag-json> \
  --vehicle-model minimal/grape_vehicle_model.json \
  --controller-yaml /home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml \
  --output-dir <output-dir>
```

Optional:

```text
--allow-non-prior-free-result
--allow-point-estimate-only
--characteristic-length
--large-scale-min
--large-scale-max
--strong-coupling-threshold
```

No deployment option.

No automatic write into `jsk_aerial_robot`.

---

# 34. Golden failure-1 smoke-test values

Using the current failure-1 prior-free result:

```text
J/m =
[[ 0.101580540184, -0.000122880371, -0.016491563434],
 [-0.000122880371,  0.077355423609,  0.000009522580],
 [-0.016491563434,  0.000009522580,  0.171547961559]]
```

```text
CoG =
[ 0.017628084552,
 -0.004117347605,
 -0.044375619045]
m
```

```text
f/m =
[0.334909059692,
 0.431263094023,
 0.376598159747,
 0.312238471199]
kg^-1
```

the current static derivation gives approximately:

```text
characteristic length =
0.1915849943 m
```

```text
scale_xy =
1.15204476
```

```text
scale_z =
1.16976482
```

```text
scale_roll_pitch =
3.52877431
```

```text
scale_yaw =
3.37886790
```

```text
dimensionless error before =
1.30016466
```

```text
dimensionless error after =
0.44348763
```

```text
off-diagonal coupling ratio =
0.11628596
```

For the failure-1 bag-recorded controller gains this implies approximately:

```text
xy:
  P 3.0 -> 3.45613
  I 0.1 -> 0.115204
  D 1.0 -> 1.15204

z:
  P 5.0 -> 5.84882
  I 1.0 -> 1.16976
  D 2.5 -> 2.92441

roll_pitch:
  P 20.0 -> 70.5755
  I 1.0  -> 3.52877
  D 8.0  -> 28.2302

yaw:
  P 4.0 -> 13.5155
  I 1.0 -> 3.37887
  D 2.0 -> 6.75774
```

These are golden implementation checks.

They are **not approved flight gains**.

The large rotational corrections must produce:

```text
proposal_status = review_required
```

with:

```text
large_static_gain_change
```

---

# 35. Mandatory unit tests

Create:

```text
minimal/tests/test_gimbalrotor_pid_postprocess.py
```

At minimum:

## Test 1 — current result contract

Parse a compact fixture matching current `result.json`.

Verify scale-free plant and rotor lag.

## Test 2 — current committed smoke input

Read the current committed failure-1 prior-free result and verify the golden
numbers in Section 34 within numerical tolerance.

## Test 3 — failed optimizer rejection

Reject:

```text
optimization_status != completed
success != true
```

## Test 4 — case policy

Reject non-prior-free result by default.

Accept only under explicit override.

## Test 5 — point-estimate-only policy

Reject `point_estimate_completed` by default.

Accept only under explicit override and require warning.

## Test 6 — scale-free plant validation

Reject:

```text
nonfinite values
non-symmetric J/m
non-SPD J/m
f_i/m <= 0
```

## Test 7 — BODY-vs-CoG geometry conversion

Given the nominal model, verify:

```text
controller rotor origin =
body rotor origin - nominal CoG
```

and ensure the raw BODY origins are not accidentally passed as controller
origins.

## Test 8 — nominal allocation parity

Construct controller-snapshot geometry from the current vehicle model and
compare:

```python
acceleration_allocation_matrix(...)
```

against an independent direct formula at zero gimbal.

## Test 9 — current hard-coded snapshot parity

For the current canonical model only, verify that the adapted nominal allocation
agrees with the current `GrapeGeometry.grape()` / reference-controller snapshot
to tight tolerance.

Do not use `GrapeGeometry.grape()` as production input.

## Test 10 — source pseudoinverse parity

Verify exact absolute `1e-4` cutoff semantics.

## Test 11 — identity invariant

Set:

```text
J/m = J_nominal / m_nominal
CoG = nominal CoG
f_i/m = 1 / m_nominal
```

Require:

```text
H ~= I
H_bar ~= I
all scales ~= 1
```

## Test 12 — common-scale gauge invariance

Different absolute gauge representatives with identical scale-free plant must
produce identical results.

## Test 13 — uniform force/m scaling

Scale every `f_i/m` by `c` around otherwise nominal plant.

Require:

```text
H ~= c I
all gain scales ~= 1/c
```

## Test 14 — CoG perturbation

Verify induced torque/cross-axis coupling.

## Test 15 — inertia perturbation

Verify translational `A_real` block is unchanged while rotational response
changes.

## Test 16 — rotor-specific effectiveness

Verify off-diagonal coupling and that the fitted grouped scale cannot increase
the grouped least-squares objective when its scale is valid.

## Test 17 — mixed-unit metric

Verify fitting uses `H_bar`.

## Test 18 — YAML semantic preservation

After removing the 12 allowed gain leaves, parsed input and output YAML trees
must be equal.

## Test 19 — source-default controller modes

With reference YAML:

```text
gimbal_dof absent -> resolve 1
underactuate absent -> resolve false
gimbal_calc_in_fc -> false from YAML
```

Do not reject absent default-valued keys.

## Test 20 — unsupported controller branch

Reject explicit:

```text
gimbal_dof != 1
underactuate == true
gimbal_calc_in_fc == true
yaw.need_d_control == false
```

## Test 21 — ROS bag gains are authoritative

Supply YAML gains that intentionally differ from a recorded controller
snapshot. Verify that every `old` gain and every proposed gain uses the
recorded snapshot, while the YAML is used only as the semantic output template.
Verify that the existing rosbag adapter is called with the exact bag path and
selected interval.

---

# 36. Existing tests

Run the full relevant suite:

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
python3 -m pytest minimal/tests
```

Do not weaken or delete existing estimator tests.

The PID postprocessor must not alter prior-free estimator behavior.

---

# 37. Production output namespace

After the implementation is committed:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
```

Write production outputs under:

```text
minimal/outputs/<SOURCE_COMMIT>/gimbalrotor_pid_postprocess/
```

Recommended run directories:

```text
single_rosbag_1_prior_free_static_pid_production_20260817
single_rosbag_2_prior_free_static_pid_production_20260817
single_rosbag_succeeded_prior_free_static_pid_production_20260817
```

Each directory contains:

```text
pid_gain_postprocess.json
pid_gain_overlay.yaml
GimbalrotorControl.pid-proposal.yaml
status.json
```

Optionally:

```text
terminal_summary.txt
```

---

# 38. Production runs

Run the same v1 postprocessor on all three current prior-free estimates.

Use exactly one:

```text
minimal/grape_vehicle_model.json
```

and exactly one actual supplied YAML proposal template:

```text
GimbalrotorControl.yaml
```

for all three. Each run must nevertheless use its own bag-recorded P/I/D
baseline because the three flights did not use identical gains.

Do not use pseudo-conditioned estimates in the primary production set.

Do not automatically average the three gain proposals.

The three results are measurements of how the same controller correction depends
on the bag-specific identified plant.

That disagreement is itself important output.

---

# 39. Three-bag summary

Add:

```text
minimal/three_bag_gimbalrotor_pid_postprocess_summary.py
```

Input:

```text
three completed per-bag pid_gain_postprocess.json files
```

Output:

```text
gimbalrotor_pid_postprocess_three_bag.json
gimbalrotor_pid_postprocess_three_bag.md
```

Report:

```text
scale_xy per bag
scale_z per bag
scale_roll_pitch per bag
scale_yaw per bag

error_before per bag
error_after per bag
coupling_ratio per bag

proposed P/I/D per bag
```

Also report spread:

```text
min
max
mean
standard deviation
```

for each of the four scales.

These summaries are diagnostic only.

Do not automatically create a "mean deployment YAML".

A single deployment candidate requires an explicit later design choice.

---

# 40. Result JSON schema

Recommended:

```text
grape-param-estim/gimbalrotor-pid-postprocess/v1
```

Include:

```json
{
  "schema": "grape-param-estim/gimbalrotor-pid-postprocess/v1",
  "method": "scale_free_static_effectiveness_inverse",
  "source_commit": "...",
  "input": {
    "estimator_result_json": "...",
    "estimator_source_commit": "...",
    "estimator_case_name": "...",
    "bag_json": "...",
    "bag_path": "...",
    "bag_interval_seconds": [0.0, 0.0],
    "vehicle_model_json": "...",
    "controller_yaml": "...",
    "controller_yaml_sha256": "...",
    "controller_gain_source": "rosbag_recorded_dynamic_reconfigure"
  },
  "controller_gain_snapshot": {
    "gains": {},
    "record_times": [],
    "pid_control_flags": [],
    "source_kinds": [],
    "controller_yaml_template_gains": {},
    "recorded_gains_differ_from_yaml": true
  },
  "controller_mode": {
    "gimbal_dof": 1,
    "gimbal_dof_source": "cpp_default",
    "underactuate": false,
    "underactuate_source": "cpp_default",
    "gimbal_calc_in_fc": false,
    "gimbal_calc_in_fc_source": "yaml",
    "yaw_need_d_control": true
  },
  "scale_free_plant": {},
  "nominal_controller_model": {},
  "allocation": {
    "A_cmd": [],
    "A_real": [],
    "A_cmd_singular_values": [],
    "A_real_singular_values": [],
    "H": [],
    "H_dimensionless": []
  },
  "gain_groups": {},
  "overall": {
    "error_before_frobenius": 0.0,
    "error_after_frobenius": 0.0,
    "improvement_fraction": 0.0,
    "off_diagonal_coupling_ratio": 0.0,
    "proposal_status": "valid",
    "warnings": []
  }
}
```

Do not write JSON NaN.

---

# 41. Terminal summary

Print:

```text
Gimbalrotor PID static-effectiveness postprocess

plant result:
bag:
bag interval:
estimator source commit:
controller YAML:
rotor lag:
characteristic length:

group         scale      P old -> new     I old -> new     D old -> new
xy            ...
z             ...
roll_pitch    ...
yaw           ...

Hbar error: before ... -> after ...
coupling ratio: ...
proposal status: ...
warnings:
  - ...
```

No deployment command.

---

# 42. Error handling

Recommended exit codes:

```text
0  completed, including review_required
2  invalid input / unsupported controller mode
3  numerical/allocation failure
4  output write failure
```

On failure, if the output directory exists, write:

```text
status.json
```

with:

```text
failure_stage
exception_type
message
input paths
source commit
```

---

# 43. Source implementation commit procedure

Start from:

```text
057421223b6ad23f126be2364e7265f808eb33ef
```

Implement source and tests.

Do not add production output files yet.

Run:

```bash
git status
git diff --check

cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
python3 -m pytest minimal/tests
```

Then commit.

Recommended commit message:

```text
Add Gimbalrotor PID gain postprocessor
```

Record:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
```

Production outputs must be generated from this exact committed source revision.

---

# 44. Production execution commands

From:

```bash
source /home/leus/catkin_ws/devel/setup.bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
```

set:

```bash
CONTROLLER_YAML=/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml
```

Failure 1:

```bash
python3 minimal/gimbalrotor_pid_postprocess.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --controller-yaml "${CONTROLLER_YAML}" \
  --output-dir minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_postprocess/single_rosbag_1_prior_free_static_pid_production_20260817
```

Failure 2:

```bash
python3 minimal/gimbalrotor_pid_postprocess.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --bag-json minimal/bag_jsons/single_rosbag_2.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --controller-yaml "${CONTROLLER_YAML}" \
  --output-dir minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_postprocess/single_rosbag_2_prior_free_static_pid_production_20260817
```

Successful bag:

```bash
python3 minimal/gimbalrotor_pid_postprocess.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --bag-json minimal/bag_jsons/single_rosbag_succeeded.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --controller-yaml "${CONTROLLER_YAML}" \
  --output-dir minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_postprocess/single_rosbag_succeeded_prior_free_static_pid_production_20260817
```

Then run the three-bag summary.

Do not use `/tmp` for the canonical production outputs.

---

# 45. Production review before commit

Inspect:

```text
three gain proposals
three bag-recorded baseline gain snapshots and event provenance
four scale factors per bag
warning sets
A_cmd identity/parity
H/H_bar
error before/after
YAML semantic diff
```

Confirm that:

```text
input controller YAML was not modified
```

and:

```text
only 12 gain leaves differ in each full proposal YAML.
```

Because failure-1 currently produces very large rotational scales, the expected
status is at least:

```text
review_required
```

unless the implementation reveals a genuine discrepancy with the golden
derivation.

Do not suppress the warning merely to obtain a "valid" status.

---

# 46. Production-results commit

After production and review:

```bash
git status
git diff --check
git add ros/examples/grape-param-estim/minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_postprocess/
git commit -m "Add Gimbalrotor PID postprocess results"
```

Do not amend the source implementation commit after production has been
generated.

If source code changes are needed, create a new source commit and regenerate
production outputs under the new source-commit namespace.

---

# 47. Push

The task is not complete until the branch is pushed.

Use:

```bash
git push origin HEAD
```

Do not force-push.

If push fails, report the failure and do not claim completion.

---

# 48. Final implementer report

After push, report:

```text
baseline commit:
057421223b6ad23f126be2364e7265f808eb33ef

source implementation commit:
<sha>

production results commit:
<sha>

pushed branch/ref:
<branch>

test command:
<command>

test result:
<pass count>
```

Report the three production directories and the three-bag summary path.

Provide a compact table:

```text
bag       xy scale   z scale   roll_pitch scale   yaw scale   status
failure1
failure2
success
```

Also provide current -> proposed P/I/D values for every group.

---

# 49. Definition of done

The task is complete only when all of the following are true:

- [ ] baseline is current HEAD `057421223...`;
- [ ] the current prior-free estimator result is the default plant input;
- [ ] the exact three production bag paths and intervals are documented;
- [ ] `result.json` scale-free fields are the only estimated plant quantities
      used by v1;
- [ ] estimated absolute mass/inertia/force effectiveness cannot affect the
      proposal;
- [ ] nominal controller allocation reuses the current
      `acceleration_allocation_matrix()`;
- [ ] BODY rotor origins are explicitly converted to controller CoG-relative
      origins before nominal allocation;
- [ ] real `A_real` uses BODY origins minus estimated CoG;
- [ ] torque effectiveness remains a fixed nominal assumption and is reported;
- [ ] controller-compatible absolute `1e-4` SVD pseudoinverse semantics are
      used;
- [ ] `H = A_real A_cmd^+` satisfies the nominal identity invariant;
- [ ] mixed translational/rotational units are normalized by the characteristic
      length;
- [ ] exactly four gain-group scales are computed;
- [ ] P/I/D are scaled together inside each group;
- [ ] each P/I/D baseline comes from that bag's recorded controller snapshot,
      never from YAML;
- [ ] the existing rosbag adapter is reused without another gain-history parser;
- [ ] controller limits and non-gain fields are unchanged;
- [ ] source-default controller options are resolved correctly when absent from
      YAML;
- [ ] rotor lag is reported but not converted into an unsupported static gain
      formula;
- [ ] the input YAML is never overwritten;
- [ ] the failure-1 golden result matches the current expected numbers;
- [ ] all existing and new tests pass;
- [ ] static proposals are generated for failure1, failure2, and success;
- [ ] the three proposals are compared but not silently averaged;
- [ ] source implementation is committed;
- [ ] production outputs are committed;
- [ ] `git push origin HEAD` succeeds.

---

# 50. Final implementation principle

The first-stage PID postprocessor has one narrow scientific role:

```text
identified scale-free real plant
        +
actual nominal Gimbalrotor controller model
        |
        v
local controller-to-plant effectiveness mismatch
        |
        v
interpretable four-group PID gain correction
```

The physical estimator remains independent.

The controller's nominal model remains nominal.

The proposed PID gains are downstream design parameters.

The v1 static result is an interpretable initialization/proposal, not an
automatic flight-ready gain set.

The current repository's existing rosbag reference streams, Grape controller
port, actuator dynamics, rotor delay representation, and closed-loop simulator
provide the correct next step for dynamic validation and refinement after this
static calculation is verified.
