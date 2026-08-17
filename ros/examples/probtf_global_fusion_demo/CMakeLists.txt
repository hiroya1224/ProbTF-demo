cmake_minimum_required(VERSION 3.0.2)
project(probtf_global_fusion_demo)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  probtf_core
  probtf_msgs
  probtf_rviz
  rospy
  std_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    probtf_core
    probtf_msgs
    probtf_rviz
    rospy
    std_msgs
    visualization_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/global_fusion_demo_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
