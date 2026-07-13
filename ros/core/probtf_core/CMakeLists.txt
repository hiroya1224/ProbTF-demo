cmake_minimum_required(VERSION 3.0.2)
project(probtf_core)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    probtf_msgs
    roscpp
    rospy
    std_msgs
    tf2_ros
    tf2_msgs
)

find_package(Eigen3 REQUIRED)
get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES probtf_runtime_cpp
  CATKIN_DEPENDS
    geometry_msgs
    probtf_msgs
    roscpp
    rospy
    std_msgs
    tf2_ros
    tf2_msgs
  DEPENDS Eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(probtf_runtime_cpp
  src/latest_snapshot.cpp
)
add_dependencies(probtf_runtime_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(probtf_runtime_cpp Eigen3::Eigen ${catkin_LIBRARIES})

add_executable(probtf_bridge_node
  nodes/probtf_bridge_node.cpp
)
add_dependencies(probtf_bridge_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(
  probtf_bridge_node
  probtf_runtime_cpp
  ${catkin_LIBRARIES}
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

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  TARGETS probtf_runtime_cpp probtf_bridge_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_latest_snapshot
    test/test_latest_snapshot.cpp
  )
  if(TARGET test_latest_snapshot)
    add_dependencies(test_latest_snapshot ${catkin_EXPORTED_TARGETS})
    target_link_libraries(
      test_latest_snapshot
      probtf_runtime_cpp
      ${catkin_LIBRARIES}
    )
  endif()
endif()
