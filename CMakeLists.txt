cmake_minimum_required(VERSION 3.0.2)
project(probik_demo)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_generation
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
    BinghamDistribution.msg
    GaussianPosition.msg
    ProbabilisticTF.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/ptf_demo_publisher.py
    scripts/ptf_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    launch
    msg
    rviz
    urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
