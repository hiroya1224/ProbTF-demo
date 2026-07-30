cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/grape_param_estim_app.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    lectures
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES
    README.md
    requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
