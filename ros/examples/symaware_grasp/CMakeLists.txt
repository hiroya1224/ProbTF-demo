cmake_minimum_required(VERSION 3.0.2)
project(symaware_grasp)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_generation
    probtf_core
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
    GraspTarget.msg
    GraspTargetArray.msg
    IKResult.msg
    ObjectBelief.msg
    SelectedGraspTarget.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    probtf_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    probtf_core
    probtf_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/arm_joint_state_controller.py
    scripts/object_pose_node.py
    scripts/probtf_static_broadcaster.py
    scripts/prob_tf_link_cloud_node.py
    scripts/prob_tf_grasp_target_node.py
    scripts/ptf_visualizer.py
    scripts/symmetry_aware_ik_node.py
    scripts/write_simple_six_dof_prob_tf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    configs
    launch
    lectures
    rviz
    urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
