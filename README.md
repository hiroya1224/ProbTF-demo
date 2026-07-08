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
- `deflecomp_ros/config/imu_frames.yaml`: IMU frame list.
- `deflecomp_sim/config/sim_params.yaml`: simulator stiffness, dynamics, noise, lag, topics, and simulator spring model.

The current default is the no-noise/no-delay simulation baseline. In this mode the simulator uses quasi-static equilibrium, the command low-pass filter is disabled, and the estimator initial stiffness is set equal to the simulator stiffness. `equil` should be much closer to `ref` than `cmd` is:

```text
||equil - ref|| < ||cmd - ref||
```

### Common Settings

| What to change | File | Parameter |
| --- | --- | --- |
| Command low-pass delay from `ref` to `cmd` | `deflecomp_ros/config/controller.yaml` | `theta_cmd_tau` |
| Compensator spring model | `deflecomp_ros/config/controller.yaml` | `spring_model` |
| Simulator spring model | `deflecomp_sim/config/sim_params.yaml` | `spring_model` |
| True simulator stiffness | `deflecomp_sim/config/sim_params.yaml` | `kp_true` |
| Initial estimated stiffness | `deflecomp_ros/config/estimator.yaml` | `kp0` |
| Stiffness estimator update | `deflecomp_ros/config/estimator.yaml` | `update_stiffness`, `q_proc`, `stiffness_update_gain`, `max_log_kp_step` |
| Stiffness estimate limits | `deflecomp_ros/config/estimator.yaml` | `kp_min`, `kp_max` |
| Observability gating for stiffness/feedforward | `deflecomp_ros/config/estimator.yaml` | `observability_rcond`, `observability_abs`, `project_unobservable_feedforward` |
| Per-update IMU information cap | `deflecomp_ros/config/estimator.yaml` | `measurement_info_eig_cap` |
| Simulator mode | `deflecomp_sim/config/sim_params.yaml` | `eq_mode` |
| Simulator command/equilibrium lag | `deflecomp_sim/config/sim_params.yaml` | `ref_tau`, `ref_max_vel`, `vel_limit`, `tau_eq` |
| Quasi-static joint noise | `deflecomp_sim/config/sim_params.yaml` | `qs_noise_std_deg` |
| Quasi-static vibration | `deflecomp_sim/config/sim_params.yaml` | `qs_vib_amp_deg`, `qs_vib_freq_hz`, `qs_vib_axes` |
| Reference topic | `deflecomp_ros/config/simple6r.yaml` | `topic_ref` |
| Command topic | `deflecomp_ros/config/simple6r.yaml` | `topic_cmd_out` |
| Simulator command input topic | `deflecomp_sim/config/sim_params.yaml` | `topic_cmd` |
| Simulator equilibrium output topic | `deflecomp_sim/config/sim_params.yaml` | `topic_equil` |

Use `spring_model: linear` to match `online-deflecomp` commit `ad5163a`. Use `spring_model: periodic` in both `controller.yaml` and `sim_params.yaml` for the circular spring model.

For URDFs that include passive, mimic, or zero-velocity joints, `deflecomp_core.robot.RobotArm` builds a Pinocchio reduced model and locks those non-controllable joints at zero. For example, `yamaguchi_6axis_arm_nejineji.urdf` contains gripper mimic/prismatic joints in addition to the six arm joints; the reduced model keeps only the six controllable arm joints and treats the gripper/camera links as fixed payloads. The ROS node logs the active `joints` and `locked_joints` at startup. `/cmd/joint_states` and `/equil/joint_states` are expanded back to the full movable URDF joint list for RViz/TF; locked joints are filled from the latest reference-derived joint values.

### Staged Estimation

The current staged-estimation setting re-enables the Bingham/WEKF stiffness update with a gentle process noise and an initial stiffness matched to the current loose simulator stiffness:

```yaml
# deflecomp_ros/config/estimator.yaml
update_stiffness: true
kp0: [5.0, 5.0, 5.0, 10.0, 20.0, 20.0]
kp_min: 1.0
kp_max: 500.0
q_proc: 1.0e-8
observability_rcond: 0.0001
observability_abs: 1.0e-10
measurement_info_eig_cap: 1.0
stiffness_update_gain: 0.2
max_log_kp_step: 0.002
min_log_kp_step: 0.0
project_unobservable_feedforward: true
```

```yaml
# deflecomp_ros/config/controller.yaml
spring_model: periodic
```

The observability gate is based on the local IMU gravity-direction Jacobian, not on joint names or a hard-coded yaw/gravity assumption. Stiffness updates are applied only in the stiffness subspace supported by the current IMU observations. `measurement_info_eig_cap` limits the information strength of one IMU update, `stiffness_update_gain` tempers the update, and `max_log_kp_step` bounds the per-cycle change in log stiffness. Keep `q_proc` small for static stiffness; increasing it makes the estimator keep adapting during a hold and can make `theta_cmd` drift toward `theta_ref` when the IMU residual contains bias or noise. When `project_unobservable_feedforward` is true, the gravity feedforward term is re-evaluated only along locally observable reference-motion components; the target `theta_ref` itself is not projected. If no IMU observation is available for a cycle, the gravity feedforward evaluation pose is held instead of being reset to `theta_ref`.

### First-Stage Debug: No Noise, No Delay

Use this state when checking the basic relationship between `ref`, `cmd`, and `equil` with stiffness estimation frozen:

```yaml
# deflecomp_ros/config/controller.yaml
theta_cmd_tau: 0.0
spring_model: periodic
```

```yaml
# deflecomp_ros/config/estimator.yaml
update_stiffness: false
kp0: [5.0, 5.0, 5.0, 10.0, 20.0, 20.0]
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

Keep `deflecomp_ros/config/estimator.yaml::kp0` equal to `deflecomp_sim/config/sim_params.yaml::kp_true` while isolating the feedforward/equilibrium logic. Keep `update_stiffness: false` while checking whether the Bingham/WEKF stiffness update is causing oscillation. If `kp0` and `kp_true` differ, the static inverse-statics command can be wrong even when noise and lag are disabled.

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
