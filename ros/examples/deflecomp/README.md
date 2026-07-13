This repository separates the deflection compensation logic from the simulator.
The core compensation, estimation, and observation models are implemented in
`deflecomp_core` without ROS dependencies. ROS nodes and simulation components
are provided as wrappers around the core package.

## Packages

- `deflecomp_core`: ROS-free compensation, equilibrium, observation, and estimation logic.
- `deflecomp_ros`: ROS1 node wrappers and launch/config files for real or replayed topics.
- `deflecomp_sim`: flexible-joint simulator and simulated IMU publishers.
- `deflecomp_description`: URDF and RViz assets.
- `deflecomp_examples`: offline demos and example scripts.

## Build

```bash
cd /home/leus/catkin_ws
catkin_make
```

## Quick Start

Launch the full RViz viewer stack with the default model:

```bash
roslaunch deflecomp_ros deflecomp_frames.launch
```

Launch the same stack with a specific URDF:

```bash
roslaunch deflecomp_ros deflecomp_frames.launch model:=/abs/path/to/robot.urdf
```

## Parameter Files

`deflecomp_frames.launch` loads these parameter files:

- `deflecomp_ros/config/simple6r.yaml`: URDF-independent ROS topic/frame wiring.
- `deflecomp_ros/config/controller.yaml`: command generation, command lag filter, and spring model for the compensator.
- `deflecomp_ros/config/estimator.yaml`: stiffness estimator parameters.
- `deflecomp_ros/config/imu_frames.yaml`: default IMU frame config. Override it with `imu_config:=...` for each robot model.
- `deflecomp_sim/config/sim_params.yaml`: simulator stiffness, dynamics, noise, lag, topics, and simulator spring model.

The checked-in default is an idealized dynamic simulation with no synthetic sensor noise or transport delay. It is not a no-delay/quasi-static configuration: the simulator uses `eq_mode: dynamic`, `ref_tau: 0.1 s`, and a 1 ms integration step, while the compensator uses `theta_cmd_tau: 0.2 s`. The simulator stiffness is `[5, 5, 5, 10, 20, 20]`; the estimator instead starts from `sqrt(kp_min * kp_max) = sqrt(500)` on every active axis and adapts from IMU observations. After transients and estimator/execution settling, the intended compensation check is:

```text
||equil - ref|| < ||cmd - ref||
```

### Common Settings

| What to change | File | Parameter |
| --- | --- | --- |
| Command low-pass delay from `ref` to `cmd` | `deflecomp_ros/config/controller.yaml` | `theta_cmd_tau` |
| Minimum-correction L1 command regularization | `deflecomp_ros/config/controller.yaml` | `theta_cmd_l1_regularization`, `theta_cmd_l1_regularization_weight` |
| Compensator spring model | `deflecomp_ros/config/controller.yaml` | `spring_model` |
| Compensator equilibrium refinement | `deflecomp_ros/config/controller.yaml` | `equilibrium_refine`, `equilibrium_refine_maxiter`, `equilibrium_refine_tol` |
| Simulator spring model | `deflecomp_sim/config/sim_params.yaml` | `spring_model` |
| Simulator equilibrium refinement | `deflecomp_sim/config/sim_params.yaml` | `equilibrium_refine`, `equilibrium_refine_maxiter`, `equilibrium_refine_tol` |
| True simulator stiffness | `deflecomp_sim/config/sim_params.yaml` | `kp_true` |
| Stiffness estimator update and uncertainty ceiling | `deflecomp_ros/config/estimator.yaml` | `update_stiffness`, `log_kp_process_noise_var`, `max_log_kp_covariance_var` |
| Estimator iterated MAP and nonlinear trust region | `deflecomp_ros/config/estimator.yaml` | `laplace_outer_iterations`, `max_log_kp_update_step`, `max_equilibrium_pose_jump`, `joint_limit_reaction_torque_tol` |
| Static estimator timing gate | `deflecomp_ros/config/estimator.yaml` | `estimation_settle_time`, `estimation_command_tolerance`, `estimation_reference_tolerance`, `command_apply_delay` |
| Execution stiffness smoothing | `deflecomp_ros/config/estimator.yaml` | `kp_exec_tau`, `max_log_kp_exec_step`, `publish_kp_exec` |
| Stiffness estimate limits and initialization range | `deflecomp_ros/config/estimator.yaml` | `kp_min`, `kp_max` |
| Observability gating for stiffness/feedforward | `deflecomp_ros/config/estimator.yaml` | `observability_rcond`, `observability_abs`, `project_unobservable_feedforward` |
| Deterministic particle scan | `deflecomp_ros/config/estimator.yaml` | `particle_scan_enabled`, `particle_scan_window_size`, `particle_scan_grid_size`, `particle_scan_reset_std` |
| Simulator mode | `deflecomp_sim/config/sim_params.yaml` | `eq_mode` |
| Simulator command/equilibrium lag | `deflecomp_sim/config/sim_params.yaml` | `ref_tau`, `ref_max_vel`, `vel_limit`, `tau_eq` |
| Quasi-static link noise | `deflecomp_sim/config/sim_params.yaml` | `qs_noise_std_deg` |
| Quasi-static vibration | `deflecomp_sim/config/sim_params.yaml` | `qs_vib_amp_deg`, `qs_vib_freq_hz`, `qs_vib_axes` |
| Reference topic | `deflecomp_ros/config/simple6r.yaml` | `topic_ref` |
| Command topic | `deflecomp_ros/config/simple6r.yaml` | `topic_cmd_out` |
| Simulator command input topic | `deflecomp_sim/config/sim_params.yaml` | `topic_cmd` |
| Simulator equilibrium output topic | `deflecomp_sim/config/sim_params.yaml` | `topic_equil` |

Use `spring_model: linear` to match `online-deflecomp` commit `ad5163a`. Use `spring_model: periodic` in both `controller.yaml` and `sim_params.yaml` for the circular spring model.

Set `viewer:=true` on `deflecomp_frames.launch` to start the `deflecomp_debug` stiffness plotter. By default it shows `/deflecomp/kp_est` and `/deflecomp/kp_exec` side by side, with `/deflecomp/kp_est` shaded by the `/deflecomp/kp_cov_diag` +/-2 sigma range.

Robot-specific IMU frames are supplied through `imu_config`. The same YAML is loaded into the estimator, simulator, and optional static TF publisher:

```bash
roslaunch deflecomp_ros deflecomp_frames.launch \
  model:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/urdf/yamaguchi_6axis_arm_nejineji.urdf \
  imu_config:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/config/deflecomp_imu_frames.yaml
```

The YAML schema separates the incoming IMU frame from the URDF frame used by Pinocchio:

```yaml
imu_frames:
  - frame_id: module4_imu
    model_frame: module4_link2
    parent_frame: module4_link2
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]
    publish_static_tf: true
static_transforms: []
```

If the IMU frame already exists in the URDF, set `frame_id` and `model_frame` to that frame and leave `publish_static_tf` unset. If the IMU frame is only a fixed child of an existing URDF link, set `frame_id` to the IMU `sensor_msgs/Imu.header.frame_id`, `model_frame`/`parent_frame` to the URDF link, and enable `publish_static_tf:=true` on the launch. In `deflecomp_frames.launch`, static IMU TF parents are prefixed with `equil` by default so they attach to the equilibrium display tree.

For URDFs that include passive, mimic, or zero-velocity joints, `deflecomp_core.robot.RobotArm` builds a Pinocchio reduced model and locks those non-controllable joints at zero. For example, `yamaguchi_6axis_arm_nejineji.urdf` contains gripper mimic/prismatic joints in addition to the six arm joints; the reduced model keeps only the six controllable arm joints and treats the gripper/camera links as fixed payloads. The ROS node logs the active `joints` and `locked_joints` at startup. `/cmd/joint_states` and `/equil/joint_states` are expanded back to the full movable URDF joint list for RViz/TF; locked joints are filled from the latest reference-derived joint values.

### Staged Estimation

The current staged-estimation setting treats `K_est` as an effective, disturbance-absorbing parameter and therefore permits a settled observation batch to move it substantially. The initial stiffness is fixed to the log-space midpoint of `kp_min` and `kp_max` for every active joint:

```yaml
# deflecomp_ros/config/estimator.yaml
A_param: 1000.0
update_stiffness: true
kp_min: 1.0
kp_max: 500.0
log_kp_process_noise_var: 0.30
max_log_kp_covariance_var: 0.0
observability_rcond: 0.0001
observability_abs: 1.0e-10
laplace_outer_iterations: 5
max_log_kp_update_step: 3.0
max_equilibrium_pose_jump: 0.30
joint_limit_reaction_torque_tol: 0.001
project_unobservable_feedforward: false
kp_exec_tau: 0.5
max_log_kp_exec_step: 0.05
publish_kp_exec: true
estimation_settle_time: 0.50
estimation_command_tolerance: 1.0e-3
estimation_reference_tolerance: 1.0e-4
command_apply_delay: 0.0
```

```yaml
# deflecomp_ros/config/controller.yaml
spring_model: periodic
equilibrium_refine: true
```

`equilibrium_refine` keeps the staged L-BFGS-B equilibrium solve, then refines the same total potential with box-constrained L-BFGS-B.  This preserves the correct KKT condition at a joint limit, where a reaction torque means the raw quasi-static residual need not vanish. Leave it enabled when checking whether `equil` matches `ref`. Disable it only when measuring raw solver speed or isolating optimizer behavior.

The observability gate is based on prior-whitened local IMU gravity-direction information, not on joint names or a hard-coded yaw/gravity assumption. Stiffness updates are applied only in the supported stiffness subspace. One synchronized, settled IMU batch defines one fixed Gaussian prior, `x0, spectral_cap(P0 + Q)`. The estimator then re-solves equilibrium and relinearizes that same likelihood up to five times; these are optimization iterations, not five independent observations, so neither `Q` nor measurement evidence is added again. Every local candidate uses at most 16 backtracking scales, must keep `||delta q_eq||_2 <= 0.30 rad`, and must not acquire, relative to the batch's initial equilibrium, a new joint-limit contact whose KKT reaction exceeds `joint_limit_reaction_torque_tol: 1e-3 N m`. Comparing with the batch-initial active set prevents a grazing contact in one outer iteration from becoming a strong contact in the next. Candidates must also make both the exact likelihood and the same fixed-prior posterior objective nonworse. `max_log_kp_update_step: 3.0` caps the total `max|x - x0|` over the whole batch, not each outer iteration. Covariance is committed once from the final linearization.

The estimator keeps `K_est`, while command generation uses the separately smoothed `K_exec`; `/deflecomp/kp_hat` and `/deflecomp/kp_est` publish the estimate, and `/deflecomp/kp_exec` publishes the execution stiffness. `K_est` may therefore change abruptly without being injected directly into the command. `K_exec` approaches `K_est` with `kp_exec_tau: 0.5 s` and an independent `max_log_kp_exec_step: 0.05` per control cycle. Initial `K_est` is `sqrt(kp_min * kp_max)` for every active joint, and initial `log K` standard deviation is `(log(kp_max) - log(kp_min)) / 4`, so the configured stiffness range is about +/-2 sigma. The present `log_kp_process_noise_var: 0.30` is intentional: `1e-8` made the posterior covariance collapse after the no-load hold, so a subsequent load step was strongly opposed by the stale Gaussian prior.

Because `Q = 0.30 I` is added for every fresh settled batch, an exactly unobservable covariance eigenmode would otherwise grow by `0.30` each time even though its mean does not move; a 1000-batch probe reached approximately `300`. Before whitening each batch, the estimator therefore clips the eigenvalues of `P_est + Q` to `max_log_kp_covariance_var`. In the ROS/YAML interface, `0.0` is an automatic-value sentinel, not “disabled”: the node resolves it to the initial log-K prior variance. With bounds `[1, 500]`, this is `((log(500) - log(1)) / 4)^2 = 2.4138346`. The cap bounds uncertainty only; it is not a limit on the `K_est` mean, `delta log K_est`, or update rate. Regression tests keep the `K_est` mean exactly unchanged in the zero-information probe and keep one-batch payload adaptation unchanged, while unobservable covariance eigenvalues remain at or below `2.4138346`.

The simulator's `apply_frame_load.py` publishes a wrench only to the plant. For example, `_mass_kg:=0.5` applies a world-downward `4.905 N` force at the selected frame origin; the estimator is not told the mass, force, or simulator state. Because its model still contains only gravity and diagonal joint stiffness, the resulting `K_est` is a pose- and load-dependent effective stiffness that absorbs the unmodelled static wrench. It is not an identification of the material joint stiffness, and a positive diagonal `K` cannot in general represent arbitrary loads across multiple poses.

Gravity feedforward is evaluated at the known desired pose. `project_unobservable_feedforward` remains only as an opt-in comparison mode and must normally remain `false`: its former pose-dependent incremental projection is path-dependent and can make the gravity-evaluation pose drift away from the desired pose. Estimator records are accepted only after command and reference have remained stable for `estimation_settle_time`; the command is selected from actual publication history at the newest timestamp common to every IMU frame. All outer iterations finish inside that one accepted batch, before `K_exec_target` is changed, so they do not wait for the intentionally slow `K_exec` ramp to settle between relinearizations. `/deflecomp/estimation_gate_status` reports the timing-gate reason plus aggregate `est_update_applied`, `est_update_skipped_reason`, final `laplace_step_scale`, total `laplace_dx_max_abs`, `laplace_outer_requested`, `laplace_outer_completed`, `laplace_outer_accepted`, `laplace_outer_stop_reason`, and `laplace_prior_covariance_capped`.

Gravity-only IMUs directly constrain observed gravity directions, not rotation about gravity or a unique full joint pose. Multiple link IMUs can make the matched-model pose prediction improve, but that is not a general full-pose observability guarantee. Add encoders, heading, or vision observations when full joint/frame pose is required.

### First-Stage Debug: No Noise, No Delay

Use this state when checking the basic relationship between `ref`, `cmd`, and `equil` with stiffness estimation frozen:

```yaml
# deflecomp_ros/config/controller.yaml
theta_cmd_tau: 0.0
spring_model: periodic
equilibrium_refine: true
```

```yaml
# deflecomp_ros/config/estimator.yaml
update_stiffness: false
kp_min: 1.0
kp_max: 500.0
```

```yaml
# deflecomp_sim/config/sim_params.yaml
kp_true: [5.0, 5.0, 5.0, 10.0, 20.0, 20.0]
ref_tau: 0.0
ref_max_vel: 0.0
eq_mode: quasistatic
qs_noise_std_deg: 0.0
qs_vib_amp_deg: 0.0
spring_model: periodic
```

With `update_stiffness: false`, the estimator uses the fixed initial stiffness `sqrt(kp_min * kp_max)` for every active joint. Keep the simulator `kp_true` near that value while isolating the feedforward/equilibrium logic, or leave stiffness estimation enabled so `K_est` can adapt.

In quasi-static mode, `qs_noise_std_deg` and `qs_vib_amp_deg` perturb the simulated link trajectory. The simulator finite-differences only that perturbation, so synthetic IMU `angular_velocity` and `linear_acceleration` reflect vibration/noise while zero noise/vibration restores gravity-only quasi-static IMU observations.

### Returning To Dynamic Simulation

After `equil` is closer to `ref` than `cmd` is in the no-delay case, return to dynamic simulation deliberately:

```yaml
# deflecomp_sim/config/sim_params.yaml
eq_mode: dynamic
ref_tau: 0.04
ref_max_vel: 4.0
vel_limit: 4.0
```

## Simulation

```bash
roslaunch deflecomp_sim sim.launch
```

The simulator publishes:

- `/equil/joint_states`
- `/imu`

It subscribes to `/cmd/joint_states` with the default `sim_params.yaml`.

## Compensation Node

```bash
roslaunch deflecomp_ros deflecomp.launch
```

## RViz Viewer

```bash
roslaunch deflecomp_ros deflecomp_frames.launch model:=/abs/path/to/robot.urdf
```

This launch starts the angle controller GUI, `deflecomp_ros`, `deflecomp_sim`,
the three robot-state publishers for `ref`, `cmd`, and `equil`, and RViz.

Key outputs:

- `/cmd/joint_states`
- `/deflecomp/kp_hat`
- `/deflecomp/debug`

## Offline Demo

```bash
rosrun deflecomp_examples offline_demo.py
```

Set `~urdf_path` in launch or pass `--urdf` to scripts if you want to override
the default `deflecomp_description/urdf/simple6r.urdf`.
