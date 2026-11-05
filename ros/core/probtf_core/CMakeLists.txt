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
  LIBRARIES probtf_runtime_cpp probtf_bingham_sampler
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

add_library(probtf_bingham_sampler SHARED src/bingham_sampler.cpp)
target_link_libraries(probtf_bingham_sampler Eigen3::Eigen)

# Use the interpreter selected by catkin, including its exact extension ABI.
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE PROBTF_PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE PROBTF_PYTHON_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
add_library(probtf_python_sampling MODULE src/python_sampling.cpp)
target_include_directories(probtf_python_sampling PRIVATE ${PROBTF_PYTHON_INCLUDE})
target_link_libraries(probtf_python_sampling probtf_bingham_sampler)
set_target_properties(probtf_python_sampling PROPERTIES
  PREFIX "" OUTPUT_NAME "_sampling" SUFFIX "${PROBTF_PYTHON_SUFFIX}"
  LIBRARY_OUTPUT_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/probtf"
  INSTALL_RPATH "$ORIGIN/../../..")

add_library(probtf_runtime_cpp
  src/gaussian_latent_store.cpp
  src/latest_snapshot.cpp
  src/sampling.cpp
)
add_dependencies(probtf_runtime_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(probtf_runtime_cpp probtf_bingham_sampler Eigen3::Eigen ${catkin_LIBRARIES})

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
  DIRECTORY launch docs lectures
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES
    README.md
    SELECTION_RESULTS.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  PROGRAMS
    test/run_temporal_selection.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/test
)

install(
  FILES
    test/temporal_selection.yaml
    test/temporal_selection_results_2026-07-24.json
    test/temporal_selection_results_2026-07-24_post_hardening.json
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/test
)

install(
  DIRECTORY test/fixtures/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/test/fixtures
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  TARGETS probtf_bingham_sampler probtf_runtime_cpp probtf_bridge_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS probtf_python_sampling
  LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}/probtf)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_bingham_sampler test/test_bingham_sampler.cpp)
  if(TARGET test_bingham_sampler)
    target_link_libraries(test_bingham_sampler probtf_bingham_sampler)
  endif()
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

  catkin_add_gtest(test_sampling
    test/test_sampling.cpp
  )
  if(TARGET test_sampling)
    add_dependencies(test_sampling ${catkin_EXPORTED_TARGETS})
    target_link_libraries(
      test_sampling
      probtf_runtime_cpp
      ${catkin_LIBRARIES}
    )
  endif()

  catkin_add_gtest(test_dependency_smoother
    test/test_dependency_smoother.cpp
  )
  if(TARGET test_dependency_smoother)
    add_dependencies(test_dependency_smoother ${catkin_EXPORTED_TARGETS})
    target_link_libraries(
      test_dependency_smoother
      probtf_runtime_cpp
      ${catkin_LIBRARIES}
    )
  endif()

  add_executable(dependency_smoother_benchmark
    test/dependency_smoother_benchmark.cpp
  )
  add_dependencies(dependency_smoother_benchmark ${catkin_EXPORTED_TARGETS})
  target_link_libraries(
    dependency_smoother_benchmark
    probtf_runtime_cpp
    ${catkin_LIBRARIES}
  )
endif()
