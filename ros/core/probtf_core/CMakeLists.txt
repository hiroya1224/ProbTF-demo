cmake_minimum_required(VERSION 3.0.2)
project(probtf_core)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_filters
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_filters
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/imu_kinematics_node.py
    nodes/imu_relative_pose_node.py
    nodes/symbolic_urdf_materializer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
