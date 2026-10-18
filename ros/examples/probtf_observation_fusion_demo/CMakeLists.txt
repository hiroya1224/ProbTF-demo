cmake_minimum_required(VERSION 3.0.2)
project(probtf_observation_fusion_demo)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs std_srvs visualization_msgs geometry_msgs probtf_core prob_artag_detector)
catkin_python_setup()

# Use the Pinocchio installation belonging to catkin's selected Python. This
# is configure-time dependency discovery, never runtime compilation.
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "import importlib.util; from pathlib import Path; s=importlib.util.find_spec('pinocchio'); print(next((str(p) for p in Path(s.origin).parents if (p/'lib/cmake/pinocchio/pinocchioConfig.cmake').is_file()), '') if s else '')"
  OUTPUT_VARIABLE PROBTF_PINOCCHIO_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PROBTF_PINOCCHIO_DISCOVERY_RESULT)
if(NOT PROBTF_PINOCCHIO_DISCOVERY_RESULT EQUAL 0 OR NOT PROBTF_PINOCCHIO_PREFIX)
  message(FATAL_ERROR "The selected Python must provide Pinocchio and its CMake development files.")
endif()
list(INSERT CMAKE_PREFIX_PATH 0 "${PROBTF_PINOCCHIO_PREFIX}")
set(BOOST_ROOT "${PROBTF_PINOCCHIO_PREFIX}")
set(Boost_ROOT "${PROBTF_PINOCCHIO_PREFIX}")
set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
# Ubuntu 20.04's header-only package predates its imported CMake target.
# Newer urdfdom exports reference that target, so expose its existing headers.
find_package(urdfdom_headers REQUIRED)
if(NOT TARGET urdfdom_headers::urdfdom_headers)
  add_library(urdfdom_headers::urdfdom_headers INTERFACE IMPORTED)
  set_target_properties(urdfdom_headers::urdfdom_headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${urdfdom_headers_INCLUDE_DIRS}")
endif()
find_package(pinocchio 3.3 CONFIG REQUIRED
  PATHS "${PROBTF_PINOCCHIO_PREFIX}/lib/cmake/pinocchio" NO_DEFAULT_PATH)
find_package(LAPACK REQUIRED)
find_package(OpenMP QUIET)
catkin_package(LIBRARIES probtf_flexible_dynamics probtf_auxiliary_equilibrium)
add_library(probtf_flexible_dynamics SHARED native/flexible_dynamics.cpp)
target_compile_features(probtf_flexible_dynamics PRIVATE cxx_std_14)
target_link_libraries(probtf_flexible_dynamics PRIVATE
  pinocchio::pinocchio_default pinocchio::pinocchio_parsers)
set_target_properties(probtf_flexible_dynamics PROPERTIES
  BUILD_RPATH "${PROBTF_PINOCCHIO_PREFIX}/lib"
  INSTALL_RPATH "${PROBTF_PINOCCHIO_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS probtf_flexible_dynamics
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
add_library(probtf_auxiliary_equilibrium SHARED native/auxiliary_equilibrium.cpp)
target_compile_features(probtf_auxiliary_equilibrium PRIVATE cxx_std_14)
target_link_libraries(probtf_auxiliary_equilibrium PRIVATE
  pinocchio::pinocchio_default pinocchio::pinocchio_parsers ${LAPACK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_compile_options(probtf_auxiliary_equilibrium PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(probtf_auxiliary_equilibrium PRIVATE ${OpenMP_CXX_FLAGS})
endif()
set_target_properties(probtf_auxiliary_equilibrium PROPERTIES
  BUILD_RPATH "${PROBTF_PINOCCHIO_PREFIX}/lib"
  INSTALL_RPATH "${PROBTF_PINOCCHIO_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS probtf_auxiliary_equilibrium
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
catkin_install_python(PROGRAMS scripts/observation_fusion_demo.py scripts/observation_fusion_motion.py scripts/print_motion_robot.py scripts/record_four_cases.py scripts/benchmark_realtime.py scripts/benchmark_physics.py scripts/evaluate_running_gradient.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES README.md REQUIREMENTS.md REQUIREMENTS_STIFFNESS_CORRECTION.md REQUIREMENTS_SEQUENTIAL_UPDATE.md MOTION_REQUIREMENTS.md DYNAMIC_PHYSICS.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Demo-local implementations of the unchanged first-order body formulas.
add_library(probtf_body_law SHARED native/body_law.cpp)
target_compile_features(probtf_body_law PRIVATE cxx_std_14)
target_link_libraries(probtf_body_law PRIVATE pinocchio::pinocchio_default)
install(TARGETS probtf_body_law LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

# Image-only central-board Bingham moments: unchanged deterministic integrals.
add_library(probtf_marker_board_math SHARED native/marker_board_math.cpp)
target_compile_features(probtf_marker_board_math PRIVATE cxx_std_14)
install(TARGETS probtf_marker_board_math LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
