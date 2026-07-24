cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_generation
    probtf_msgs
    std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
    EstimatorDiagnostics.msg
    InertialParameterEstimate.msg
    ParameterParticleSet.msg
    TrajectoryParticleSet.msg
    ModelMismatch.msg
    CounterfactualCandidate.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    probtf_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    probtf_msgs
    std_msgs
)

catkin_install_python(
  PROGRAMS
    scripts/estimate_grape_bag.py
    scripts/evaluate_sanity.py
    scripts/evaluate_selection.py
    scripts/generate_sanity_bag.py
    scripts/build_bag_manifest.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    launch
    lectures
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES
    README.md
    SELECTION_RESULTS.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
