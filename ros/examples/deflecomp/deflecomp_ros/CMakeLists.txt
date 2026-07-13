cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_ros)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(
  catkin REQUIRED COMPONENTS
    deflecomp_core
    geometry_msgs
    probtf_core
    probtf_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf2_ros
    urdf
    visualization_msgs
)

find_package(Eigen3 REQUIRED)
get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    deflecomp_core
    geometry_msgs
    probtf_core
    probtf_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf2_ros
    urdf
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(probtf_point_moments_node
  nodes/probtf_point_moments_node.cpp
)
add_dependencies(probtf_point_moments_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(
  probtf_point_moments_node
  Eigen3::Eigen
  ${catkin_LIBRARIES}
)

catkin_install_python(
  PROGRAMS
    nodes/deflecomp_node.py
    nodes/probtf_point_moments_node.py
    nodes/static_imu_tf_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  TARGETS probtf_point_moments_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
