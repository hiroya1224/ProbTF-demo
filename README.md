# ProbTF integrated packages

This repository collects probabilistic-transform producers, fusion, query
experiments, symmetry-aware grasping, and deflection compensation in one
ProbTF context. Reusable numerical code lives under `src/`; ROS 1 is kept at
the transport and runtime boundary under `ros/`.

## Python installation

Clone with submodules, or initialize them in an existing checkout, then install
the root project:

```bash
git submodule update --init --recursive
python3 -m pip install .
```

The installation provides these main Python namespaces:

- `probtf`: distributions, Bingham moments, evidence fusion, IMU relative-pose
  production, quaternion prediction, sensor configuration, and symbolic URDF
  materialization
- `symaware_grasp`: probabilistic transforms and symmetry-aware IK
- `deflecomp_core`: ROS-free deflection compensation and estimation
- `deflecomp_sim`: ROS-free flexible-joint simulation
- `deflecomp_examples`: offline example helpers
- `bingham`: BinghamNLL from the pinned `develop` submodule

Optional plotting and example dependencies are available with
`python3 -m pip install '.[visualization,examples]'`.

## ROS workspace

The ROS tree is organized by role:

- `ros/core/probtf_msgs`: reusable distribution, kinematics, and evidence messages
- `ros/core/probtf_core`: thin ROS nodes and Python package relays
- `ros/examples/probtf_imu_demo`: two-IMU relative-pose and symbolic URDF demo
- `ros/examples/probtf_orientation_demo`: separated gyro/gravity/magnetic demo
- `ros/examples/deflecomp`, `ros/examples/symaware_grasp`: existing applications

Link or clone this repository into a catkin workspace, then build the core and
examples. `probtf_core` exposes the root `probtf` package in both devel and
install spaces; installing the root project separately is only needed for
standalone non-ROS use or for the older examples' Python namespaces.

```bash
cd /path/to/catkin_ws
catkin build probtf_msgs probtf_core probtf_imu_demo probtf_orientation_demo
source devel/setup.bash
```

Run the two migrated producer examples with:

```bash
roslaunch probtf_imu_demo two_imu_relative_pose.launch
roslaunch probtf_orientation_demo orientation_filter.launch
```

Quaternion arrays and Bingham matrices use `[w, x, y, z]`; ROS
`geometry_msgs/Quaternion` is converted at the adapter boundary. A
`ProbabilisticTF` explicitly states whether position and orientation are
present. Source likelihoods and gyro predictions travel as `TransformEvidence`
and carry source/provenance identifiers so independent evidence is not counted
twice accidentally.

See `docs/phase1-migration.md` for the migration map, current approximations,
and issues intentionally deferred to the next design phase.
