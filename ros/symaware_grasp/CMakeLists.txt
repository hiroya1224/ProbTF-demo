cmake_minimum_required(VERSION 3.0.2)
project(symaware_grasp)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_generation
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
    BinghamDistribution.msg
    GraspCandidate.msg
    GaussianPosition.msg
    IKResult.msg
    ProbabilisticTF.msg
    ProbabilisticTFArray.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
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
    scripts/hand_prob_tf_publisher.py
    scripts/object_pose_node.py
    scripts/ptf_demo_publisher.py
    scripts/prob_tf_link_cloud_node.py
    scripts/prob_tf_grasp_target_node.py
    scripts/ptf_visualizer.py
    scripts/sample_check_simple_six_dof.py
    scripts/show_link_prob_tf.py
    scripts/symmetry_aware_ik_node.py
    scripts/write_simple_six_dof_prob_tf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    config
    configs
    launch
    msg
    rviz
    urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
