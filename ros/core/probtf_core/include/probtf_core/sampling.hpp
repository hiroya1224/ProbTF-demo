#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <probtf_msgs/ProbabilisticTransformStamped.h>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace probtf_core {

// One joint sample from a Prob-TF transform distribution.  The transform maps
// child-frame coordinates into the record's parent frame.
struct TransformSample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
};

using TransformSampleVector =
    std::vector<TransformSample, Eigen::aligned_allocator<TransformSample>>;

// Draw exact Monte Carlo samples from the native v2 transform law.
//
// Positive finite component weights are normalized scale-safely; negative and
// zero weights carry no mass.  Numerically invalid component laws and unusable
// mixture weights are rejected.  This function never substitutes a
// deterministic representative for an invalid or zero-mass law; callers that
// explicitly want a display fallback can call representativeTransform().
bool sampleTransformDistribution(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    std::size_t count,
    std::mt19937* generator,
    TransformSampleVector* output,
    std::string* error = nullptr);

// Return the record's stored representative when present.  If none is stored,
// derive the mode of the highest positive-weight component and evaluate its
// conditional translation at that mode.
bool representativeTransform(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    Eigen::Isometry3d* output,
    std::string* error = nullptr);

}  // namespace probtf_core
