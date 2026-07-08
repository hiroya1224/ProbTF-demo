cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_sim)

find_package(
  catkin REQUIRED COMPONENTS
    deflecomp_core
    rospy
    sensor_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    deflecomp_core
    rospy
    sensor_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
