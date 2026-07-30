cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/analyze_failure_bags.py
    scripts/estimate_failure_parameters.py
    scripts/failure_analysis_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    lectures
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
