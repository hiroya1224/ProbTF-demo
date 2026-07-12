cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_examples)

find_package(
  catkin REQUIRED COMPONENTS
    deflecomp_core
    deflecomp_sim
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    deflecomp_core
    deflecomp_sim
)

catkin_install_python(
  PROGRAMS
    scripts/offline_demo.py
    scripts/simulator_flexible.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
