cmake_minimum_required(VERSION 3.0.2)
project(probtf_rviz)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(
  catkin REQUIRED COMPONENTS
    pluginlib
    probtf_core
    probtf_msgs
    roscpp
    rviz
)
find_package(Eigen3 REQUIRED)
get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
find_package(Qt5 ${rviz_QT_VERSION} EXACT REQUIRED Core Widgets)

set(CMAKE_AUTOMOC ON)
add_definitions(-DQT_NO_KEYWORDS)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES probtf_rviz
  CATKIN_DEPENDS
    pluginlib
    probtf_core
    probtf_msgs
    roscpp
    rviz
  DEPENDS Eigen3
)

include_directories(
  include
)
include_directories(
  SYSTEM
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(
  probtf_rviz
  include/probtf_rviz/probabilistic_pose_display.hpp
  include/probtf_rviz/probabilistic_tf_display.hpp
  src/probabilistic_pose_display.cpp
  src/probabilistic_tf_display.cpp
  src/transform_visual.cpp
  src/plugin_exports.cpp
)
add_dependencies(probtf_rviz ${catkin_EXPORTED_TARGETS})
target_link_libraries(
  probtf_rviz
  Eigen3::Eigen
  Qt5::Widgets
  ${catkin_LIBRARIES}
)

install(
  TARGETS probtf_rviz
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(
  FILES plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
  DIRECTORY rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_plugin_loading
    test/test_plugin_loading.cpp
  )
  if(TARGET test_plugin_loading)
    add_dependencies(test_plugin_loading probtf_rviz ${catkin_EXPORTED_TARGETS})
    target_link_libraries(
      test_plugin_loading
      probtf_rviz
      ${catkin_LIBRARIES}
    )
  endif()
endif()
