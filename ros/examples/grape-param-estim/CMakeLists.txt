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
    GimbalrotorControllerReplayMetadata.msg
    GimbalrotorControllerReplayFrame.msg
    PlantPosteriorSummary.msg
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
    scripts/analyze_grape_counterfactual.py
    scripts/audit_grape_controller_replay.py
    scripts/estimate_grape_bag.py
    scripts/estimate_grape_plant.py
    scripts/evaluate_sanity.py
    scripts/evaluate_selection.py
    scripts/generate_sanity_bag.py
    scripts/build_bag_manifest.py
    scripts/materialize_grape_controller_replay.py
    scripts/verify_inverse_dynamics_baseline.py
    scripts/verify_synthetic_sanity_baseline.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    launch
    lectures
    plans
    results
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES
    README.md
    SELECTION_RESULTS.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
