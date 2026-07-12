cmake_minimum_required(VERSION 3.0.2)
project(probtf_imu_demo)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(
  PROGRAMS
    nodes/imu_kinematics_node.py
    nodes/imu_relative_pose_node.py
    nodes/sensor_mount_tf_node.py
    nodes/symbolic_urdf_materializer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY config launch symbolic_models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
