cmake_minimum_required(VERSION 3.0.2)
project(relative_coords_from_paired_imu)

find_package(catkin REQUIRED COMPONENTS geometry_msgs message_generation std_msgs)

catkin_python_setup()

add_message_files(FILES ImuKinematics.msg)
generate_messages(DEPENDENCIES geometry_msgs std_msgs)

catkin_package(CATKIN_DEPENDS geometry_msgs message_runtime std_msgs)

catkin_install_python(
  PROGRAMS
    nodes/imu_kinematics_node.py
    nodes/imu_relative_pose_node.py
    nodes/raw_imu_relative_pose_node.py
    nodes/sensor_mount_tf_node.py
    nodes/symbolic_urdf_materializer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES README.md requirements.txt DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(
  DIRECTORY config launch symbolic_models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
