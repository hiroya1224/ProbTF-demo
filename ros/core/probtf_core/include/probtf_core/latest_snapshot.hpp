#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/TransformStamped.h>
#include <probtf_core/gaussian_latent_store.hpp>
#include <probtf_msgs/ApproximationInfo.h>
#include <probtf_msgs/ProbabilisticTransformArray.h>
#include <probtf_msgs/ProbabilisticTransformStamped.h>
#include <probtf_msgs/Provenance.h>

#include <ros/time.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace probtf_core {

using Matrix9d = Eigen::Matrix<double, 9, 9>;
using Vector9d = Eigen::Matrix<double, 9, 1>;

struct PointMoments {
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
};

struct PointMomentObservation {
  std::string target_frame;
  std::string source_frame;
  ros::Time resolved_stamp;
  std::vector<std::string> edge_ids;
  PointMoments moments;
};

struct TransformMoments {
  Eigen::Isometry3d mean = Eigen::Isometry3d::Identity();
  Matrix6d covariance = Matrix6d::Zero();
  std::vector<std::pair<std::string, std::uint64_t>> factor_versions;
  std::string perturbation_convention = kPosePerturbationConvention;
  probtf_msgs::ApproximationInfo approximation;
  probtf_msgs::Provenance provenance;
  std::vector<std::string> diagnostics;
};

struct TransformMomentObservation {
  std::string target_frame;
  std::string source_frame;
  ros::Time resolved_stamp;
  std::vector<std::string> edge_ids;
  TransformMoments moments;
};

struct TransformPathObservation {
  std::string target_frame;
  std::string source_frame;
  ros::Time resolved_stamp;
  std::vector<std::string> edge_ids;
};

// A read-only, latest-only view over one complete dynamic batch and the
// latched static set.  Message ownership remains with the caller for the
// lifetime of this object.
class LatestSnapshot {
 public:
  LatestSnapshot(const probtf_msgs::ProbabilisticTransformArray& dynamic_records,
                 const probtf_msgs::ProbabilisticTransformArray& static_records,
                 std::shared_ptr<const GaussianLatentStore> latent_store =
                     std::shared_ptr<const GaussianLatentStore>());

  bool valid(std::string* error = nullptr) const;

  // Resolve topology, dependency validity, path edge IDs, and the oldest
  // dynamic source stamp without evaluating distribution moments.
  bool lookupPathMetadata(const std::string& target_frame,
                          const std::string& source_frame,
                          TransformPathObservation* observation,
                          std::string* error = nullptr) const;

  bool lookupPointMoments(const std::string& target_frame,
                          const std::string& source_frame,
                          const Eigen::Vector3d& source_point,
                          PointMomentObservation* observation,
                          std::string* error = nullptr,
                          int bingham_integration_steps = 120) const;

  bool lookupTransformMoments(
      const std::string& target_frame,
      const std::string& source_frame,
      TransformMomentObservation* observation,
      std::string* error = nullptr) const;

 private:
  struct TransformMomentCache;

  struct PathStep {
    const probtf_msgs::ProbabilisticTransformStamped* record = nullptr;
    bool inverse = false;
  };

  bool buildPath(const std::string& target_frame,
                 const std::string& source_frame,
                 std::vector<PathStep>* path,
                 std::string* error) const;

  bool analyzePath(const std::string& target_frame,
                   const std::string& source_frame,
                   const std::vector<PathStep>& path,
                   TransformPathObservation* observation,
                   std::string* error,
                   bool allow_dependency_resolution = false) const;

  void addRecords(const probtf_msgs::ProbabilisticTransformArray& records,
                  bool expected_static);

  std::unordered_map<std::string,
                     const probtf_msgs::ProbabilisticTransformStamped*>
      edge_by_child_;
  std::unordered_set<std::string> frames_;
  bool valid_ = true;
  std::string validation_error_;
  std::shared_ptr<const GaussianLatentStore> latent_store_;
  std::shared_ptr<TransformMomentCache> transform_moment_cache_;
};

// Convert deterministic TF into the exact one-component Prob-TF wire format.
bool deterministicTfToProbTf(const geometry_msgs::TransformStamped& transform,
                             const std::string& authority,
                             bool is_static,
                             probtf_msgs::ProbabilisticTransformStamped* output,
                             std::string* error = nullptr);

// Extract an exact deterministic map.  This intentionally rejects stochastic
// records rather than silently projecting them to a representative.
bool exactProbTfToTf(const probtf_msgs::ProbabilisticTransformStamped& record,
                     geometry_msgs::TransformStamped* output,
                     std::string* error = nullptr);

}  // namespace probtf_core
