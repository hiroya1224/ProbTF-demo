# ProbTF integrated packages

This repository combines native probabilistic transforms, ROS 1 transport,
symmetry-aware grasping, and deflection compensation. Python source is owned by
the catkin package that installs it; there is no shared top-level `src/` relay.

## Package layout

- `ros/core/probtf_msgs`: message-only package for the native v2 wire contract
- `ros/core/probtf_core`: owns `probtf`, `probtf_estimators`, `probtf_ros`, and
  the ProbTF/TF bridge node
- `ros/examples/probtf_imu_demo`: two-IMU transform producer and symbolic URDF
  materialization
- `ros/examples/probtf_orientation_demo`: gyro, gravity, and magnetic
  orientation estimation
- `ros/examples/symaware_grasp`: grasp beliefs, v2 ProbTF publication, lookup,
  visualization, and symmetry-aware IK
- `ros/examples/deflecomp`: ROS-free compensation/simulation libraries and ROS
  runtime packages

Each Python namespace lives below its owning package's `src/` directory. The
root `setup.py` only aggregates these first-party package roots for non-catkin
development.

## Installation

For ROS use, place the repository in a catkin workspace and build the packages
directly. A separate root `pip install` is not required.

```bash
cd /path/to/catkin_ws
catkin build probtf_msgs probtf_core probtf_imu_demo probtf_orientation_demo \
  symaware_grasp deflecomp_core deflecomp_sim deflecomp_ros
source devel/setup.bash
```

For Python-only development, install the first-party aggregate:

```bash
python3 -m pip install -e .
```

The Bingham normalizer used by `probtf` is vendored under `probtf._vendor`.
Root installation does not package or require the external BinghamNLL
submodule. Optional dependencies are available with
`python3 -m pip install -e '.[visualization,examples,test]'`.

## Native v2 messages

The transform law is a weighted mixture of
`ProbabilisticTransformComponent`. Each component contains a
`BinghamOrientation` and a `ConditionalGaussianTranslation`, including the
3-by-9 rotation/translation coupling matrix. Metadata uses `Provenance` and
`ApproximationInfo`.

Runtime transport uses:

- `ProbabilisticTransformStamped` on `/probtf` for dynamic physical edges
- `ProbabilisticTransformArray` on `/probtf_static` for the complete static set
- `TransformEvidenceStamped` for likelihood/natural-parameter evidence
- `OrientationDistributionStamped` for orientation-only posteriors
- `ImuKinematics` at the two-IMU producer boundary

Quaternion arrays use `[w, x, y, z]`; ROS quaternion fields are converted at
the adapter boundary. The removed v1 partial transform messages are not part of
the runtime or generated message set.

## Listener and lookup

`ProbTfBroadcaster` publishes native records. `ProbTfListener` provides an
in-process timestamped graph, while `RosProbTfListener` subscribes to
`/probtf` and `/probtf_static` with a bounded history per edge. Both listeners
provide:

- `lookup_path(target_frame, source_frame, ...)`
- `lookup_kernel(target_frame, source_frame, ...)`
- `lookup_point_moments(target_frame, source_frame, point, ...)`
- `can_lookup(...)` and `wait_for_lookup(...)`

Temporal selection is explicit through `TemporalPolicy` (`EXACT`, `LATEST`,
`NEAREST_WITHIN_TOLERANCE`, or `LATEST_COMMON`). Forward/inverse traversal uses
the same latent physical edge rather than constructing an independent inverse
distribution.

`probtf_bridge_node.py` connects native topics to `/tf` and `/tf_static`. Its
default export policy is `exact_only`; exporting a stochastic record requires
an explicit representative policy.

## Demo launches

```bash
# Native ProbTF <-> TF bridge
roslaunch probtf_core probtf_bridge.launch

# Two-IMU full transform and orientation-only estimation
roslaunch probtf_imu_demo two_imu_relative_pose.launch
roslaunch probtf_orientation_demo orientation_filter.launch

# Symmetry-aware grasp workflow and static-chain point cloud
roslaunch symaware_grasp probabilistic_tf_demo.launch
roslaunch symaware_grasp prob_tf_link_cloud.launch

# Deflection-compensation simulation and multi-frame viewer/runtime
roslaunch deflecomp_sim sim_with_deflecomp.launch
roslaunch deflecomp_ros deflecomp_frames.launch viewer:=true
```

See `docs/lectures/probtf_jmaa_kernel_architecture.md` for the distribution, graph,
kernel, temporal, and approximation contracts.
