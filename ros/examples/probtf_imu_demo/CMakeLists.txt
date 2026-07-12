cmake_minimum_required(VERSION 3.0.2)
project(probtf_imu_demo)

find_package(catkin REQUIRED)

catkin_package()

install(
  DIRECTORY config launch symbolic_models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
