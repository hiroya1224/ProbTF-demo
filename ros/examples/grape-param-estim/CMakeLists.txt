cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/grape_strong_constraint_ienks.py
    scripts/grape_weak_constraint_synthetic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  FILES
    README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/grape_param_estim
  )
endif()
