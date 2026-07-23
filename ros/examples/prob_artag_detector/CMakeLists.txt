cmake_minimum_required(VERSION 3.0.2)
project(prob_artag_detector)

find_package(
  catkin REQUIRED COMPONENTS
    cv_bridge
    geometry_msgs
    probtf_core
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    probtf_core
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/prob_artag_camera_demo_node.py
    nodes/prob_artag_detector_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY config launch rviz samples
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
