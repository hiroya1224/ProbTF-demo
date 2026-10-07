cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_debug)

find_package(
  catkin REQUIRED COMPONENTS
    rospy
    std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/kp_plotter.py
    nodes/run_convergence_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    launch
    config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
