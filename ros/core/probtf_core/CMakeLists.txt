cmake_minimum_required(VERSION 3.0.2)
project(probtf_core)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    probtf_msgs
    rospy
    std_msgs
    tf2_ros
    tf2_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    probtf_msgs
    rospy
    std_msgs
    tf2_ros
    tf2_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/probtf_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
