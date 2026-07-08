cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_ros)

find_package(
  catkin REQUIRED COMPONENTS
    deflecomp_core
    rospy
    sensor_msgs
    std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    deflecomp_core
    rospy
    sensor_msgs
    std_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/deflecomp_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
