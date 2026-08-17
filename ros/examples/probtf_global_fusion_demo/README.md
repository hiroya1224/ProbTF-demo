# probtf_global_fusion_demo

A synthetic ROS 1 / RViz demo for a narrow question:

> Can a robot keep a global orientation belief, notice which rotational degree
> of freedom is still unresolved, and move an eye-in-hand camera continuously
> toward an observation that sharpens that belief?

The demo treats the relative orientation of two robot bases as a probabilistic
transform. Their relative translation is deliberately fixed so that the visual
result isolates the rotational issue.

## Scene

Two stylized robots are shown:

- **Tou** on the left, lifted above the main `xy` plane.
- **Kasuga** on the right, on the lower plane.

Each robot has a stylized **6-DoF arm**. Three revolute joints position the
wrist and a three-DoF spherical wrist orients an eye-in-hand camera. The camera
body, optical axis, wrist axes, and a small field-of-view frustum are rendered
explicitly. The yellow `moon_*` spheres are shared landmarks. Thin colored rays
show what each camera currently sees.

This is a lightweight kinematic visualization, not a rigid-body physics
simulation or an exact myCobot URDF. The separation between the three
positioning joints and the spherical wrist is intentional: the wrist can be
servoed locally while the ProbTF belief remains global on `SO(3)`.

## Default active-view sequence

`active_view:=true` is the default.

1. Both cameras initially face the `moon_0` / `moon_1` pair. Their default field
   of view contains those two landmarks but excludes `moon_2`.
2. The first shared direction produces a Bingham posterior with the expected
   global `S1` ridge.
3. The demo evaluates the unused landmark evidence directly in Bingham natural
   parameter space. The score rewards increased concentration and, especially,
   removal of the top-eigenvalue degeneracy, with a small angular-motion cost.
4. The most informative remaining landmark is selected. No `look_left()` /
   `look_right()` scan trajectory is encoded.
5. Both spherical wrists turn toward that target with a bounded angular
   velocity. The cameras do **not** stop between observations.
6. When the new shared landmark enters both fields of view, the second
   non-collinear Bingham evidence is fused. The posterior becomes localized and
   the information-seeking gaze motion stops.

The control used here is deliberately a first information-servo surrogate, not
a full POMDP or Monte-Carlo expected-information-gain optimizer. It already has
the architectural property needed for the next demo: a compact global Bingham
belief drives a local, continuous, safe camera motion without constructing a
tangent-space Gaussian for the belief itself.

## What the orange support means

The deterministic articulated arm and camera geometry are attached to the
uncertain Kasuga base. The propagated point is the actual eye-in-hand camera
position expressed in `kasuga_base`, not an abstract straight-stick endpoint.
Its support changes as

```text
SO(3) uncertainty  ->  spherical support
one direction      ->  circular support
full orientation   ->  localized support
```

The orange points in RViz are direct samples of that uncertain Kasuga camera
position. The native `probtf_rviz/ProbabilisticTF` display simultaneously
samples the actual ProbTF edge and the composed `world -> kasuga_base ->
kasuga_tool` path.

## Passive fallback

The old timed observability demonstration is still available:

```bash
roslaunch probtf_global_fusion_demo global_fusion_demo.launch active_view:=false
```

Then the state advances as

```text
uniform SO(3) --phase_duration--> one direction / S1
                --phase_duration--> two directions / local
```

## Run

```bash
cd /home/leus/catkin_ws
catkin build probtf_global_fusion_demo
source devel/setup.bash
roslaunch probtf_global_fusion_demo global_fusion_demo.launch
```

Useful overrides:

```bash
roslaunch probtf_global_fusion_demo global_fusion_demo.launch \
  concentration:=120.0 \
  gaze_speed_deg:=10.0 \
  camera_hfov_deg:=58.0 \
  camera_vfov_deg:=65.0
```

`gaze_speed_deg` is the maximum angular speed of the continuous wrist gaze
servo. Smaller values make the information-seeking motion easier to see.

The positioning-joint defaults can be changed with ROS parameters
`tou_arm_joints_deg` and `kasuga_arm_joints_deg`.

## Topics

| Topic | Type | Meaning |
|---|---|---|
| `/global_fusion_demo/probtf` | `probtf_msgs/ProbabilisticTransformStamped` | `world -> kasuga_base` posterior and deterministic articulated `kasuga_base -> kasuga_tool` eye-in-hand pose |
| `/global_fusion_demo/markers` | `visualization_msgs/MarkerArray` | articulated Tou/Kasuga arms, camera frusta, currently visible observation rays, landmark state, phase text, and orange Kasuga camera-position support |

## Model

The true relative rotation is a yaw rotation only so the intermediate global
ambiguity is visually obvious. The prior is uniform on `SO(3)`. Landmark
pair differences produce the vector-alignment Bingham evidence implemented by
`vector_alignment_bingham_evidence()`.

The first direction deliberately leaves all rotations about the aligned
direction with equal likelihood. The second non-collinear direction removes
that degeneracy. In active mode, the transition is caused by camera motion and
field-of-view visibility rather than by elapsed time.

## Next step

The current wrist controller selects among discrete informative landmarks and
then follows the selected gaze with a bounded continuous angular velocity. The
next natural extension is a genuine local velocity-space optimizer:

```text
global Bingham belief
        +
small safe qdot candidates
        -> expected information gain - motion cost
        -> smooth eye-in-hand motion
```

That version can use the same compact ProbTF belief at TF-rate while reserving
sampling, if needed, for the much slower action-selection loop.

## Look-away + uncertain-joint launch

A second launch is provided for the next-stage demonstration:

```bash
roslaunch probtf_global_fusion_demo furisake_joint_uncertainty.launch
```

Its default scene deliberately starts Kasuga about 125 degrees away from the
shared moon field. The eye-in-hand gaze servo turns continuously until common
landmarks enter the FOV, produces the global Bingham `SO(3) -> S1 -> local`
sequence, and then keeps the camera transform stochastic instead of replacing
the arm by a deterministic endpoint.

Both robots carry a demo-local six-joint zero-offset belief. The twelve offsets
are stored as one Gaussian state

```text
b = [b_tou_1 ... b_tou_6 b_kasuga_1 ... b_kasuga_6]
```

with a full `12 x 12` covariance. Each camera-pose marginal is obtained from the
current kinematic Jacobian by

```text
Sigma_camera = J_camera Sigma_b J_camera^T
```

and encoded back into the existing native ProbTF
Bingham/conditional-Gaussian component. Therefore both
`tou_base -> tou_tool` and `kasuga_base -> kasuga_tool` are stochastic edges in
this launch. The translucent ellipsoids around the cameras show the local
position part of these joint-derived marginals, while the orange support still
shows the global Kasuga-base Bingham uncertainty.

After the relative orientation has reached the two-vector state, common visual
landmark bearings condition the joint-offset Gaussian with an EKF/Joseph matrix
update. The status text reports the RMS joint posterior sigma for each arm, so
the local camera ellipsoids visibly contract as observations accumulate.
Inference in this joint-update path is matrix-only; the Bingham point cloud is
used only as the existing RViz visualization of the global orientation law.

### Important boundary of this patch

The correlated joint Gaussian currently lives inside the demo node. The
published tool edge is its **marginal**, not a new core-level latent factor.
Consequently this patch does not claim to solve arbitrary cross-edge smoothing
inside `probtf_core`. The companion `ros/core/probtf_core/docs/PROBTF_DEPENDENCY_SMOOTHER_PLAN.md`
specifies the core work needed to make per-joint edges share a latent Gaussian,
condition that joint posterior from loop-closure-like observations, and answer
dependency-aware spatial queries without sampling.

Useful overrides include:

```bash
roslaunch probtf_global_fusion_demo furisake_joint_uncertainty.launch \
  kasuga_initial_gaze_yaw_offset_deg:=155 \
  gaze_speed_deg:=12 \
  joint_prior_sigma_deg:=2.0 \
  joint_update_rate:=2.0
```
