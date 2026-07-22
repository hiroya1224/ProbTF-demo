cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(
  catkin REQUIRED COMPONENTS
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
)

generate_messages(
  DEPENDENCIES
    probtf_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    probtf_msgs
    std_msgs
)

catkin_install_python(
  PROGRAMS
    scripts/estimate_grape_bag.py
    scripts/evaluate_sanity.py
    scripts/generate_sanity_bag.py
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
  FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
