cmake_minimum_required(VERSION 3.0.2)
project(probtf_orientation_demo)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(
  PROGRAMS
    nodes/orientation_filter_node.py
    nodes/probtf_fusion_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
