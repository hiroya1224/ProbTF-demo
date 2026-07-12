cmake_minimum_required(VERSION 3.0.2)
project(probtf_msgs)

find_package(
  catkin REQUIRED COMPONENTS
    geometry_msgs
    message_generation
    std_msgs
)

add_message_files(
  FILES
    ApproximationInfo.msg
    BinghamOrientation.msg
    BinghamDistribution.msg
    ConditionalGaussianTranslation.msg
    GaussianPosition.msg
    GraspCandidate.msg
    IKResult.msg
    ImuKinematics.msg
    OrientationDistributionStamped.msg
    ProbabilisticTF.msg
    ProbabilisticTFArray.msg
    ProbabilisticTransformArray.msg
    ProbabilisticTransformComponent.msg
    ProbabilisticTransformStamped.msg
    Provenance.msg
    TransformEvidenceStamped.msg
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
    std_msgs
)
