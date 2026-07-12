# ProbTF integrated packages

This repository combines symmetry-aware probabilistic grasping and deflection
compensation. Reusable Python code lives under `src/`; ROS 1 packages under
`ros/` contain messages, nodes, launch files, configuration, and robot assets.

## Python installation

Clone with submodules, or initialize them in an existing checkout, then install
the root project:

```bash
git submodule update --init --recursive
python3 -m pip install .
```

The installation provides these Python namespaces:

- `probtf`: shared probabilistic-transform numerical primitives
- `symaware_grasp`: probabilistic transforms and symmetry-aware IK
- `deflecomp_core`: ROS-free deflection compensation and estimation
- `deflecomp_sim`: ROS-free flexible-joint simulation
- `deflecomp_examples`: offline example helpers
- `bingham`: BinghamNLL from the pinned `develop` submodule

Optional plotting and example dependencies are available with
`python3 -m pip install '.[visualization,examples]'`.

## ROS workspace

Install the Python project first, then link or clone this repository into a
catkin workspace and build the packages under `ros/`:

```bash
cd /path/to/ProbTF-demo
python3 -m pip install -e .
cd /path/to/catkin_ws
catkin build
```

`probtf_msgs` owns the reusable message contract. `symaware_grasp`,
`deflecomp_ros`, `deflecomp_sim`, and the remaining ROS packages are adapters
and runtime assets around the root Python implementation.
