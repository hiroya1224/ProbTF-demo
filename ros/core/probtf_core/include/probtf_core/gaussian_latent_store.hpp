#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/time.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace probtf_core {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

constexpr const char* kPosePerturbationConvention =
    "translation_parent_rotation_right_local";

struct LatentProvenance {
  std::vector<std::string> source_ids;
  std::vector<std::string> derived_from_edge_ids;
  std::string method;
  std::string detail;
};

struct GaussianLatentFactor {
  std::string factor_id;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  ros::Time stamp;
  std::uint64_t version = 0;
  LatentProvenance provenance;
};

struct EdgeLatentBinding {
  std::string edge_id;
  std::string factor_id;
  Eigen::MatrixXd sensitivity;
  std::uint64_t factor_version = 0;
  ros::Time linearization_stamp;
  Eigen::Isometry3d linearization_pose = Eigen::Isometry3d::Identity();
  std::string perturbation_convention = kPosePerturbationConvention;
};

struct GaussianObservationFactor {
  std::string observation_id;
  std::vector<std::string> latent_factor_ids;
  Eigen::VectorXd residual;
  std::vector<Eigen::MatrixXd> jacobian_blocks;
  Eigen::MatrixXd noise_covariance;
  ros::Time stamp;
  LatentProvenance provenance;
};

struct GaussianUpdateResult {
  std::string observation_id;
  std::vector<std::pair<std::string, std::uint64_t>> prior_versions;
  std::vector<std::pair<std::string, std::uint64_t>> posterior_versions;
  Eigen::MatrixXd innovation_covariance;
  Eigen::MatrixXd kalman_gain;
  std::uint64_t store_revision = 0;
};

struct GaussianLatentSnapshot {
  std::uint64_t revision = 0;
  std::map<std::string, GaussianLatentFactor> factors;
  std::map<std::string, std::vector<EdgeLatentBinding>> bindings_by_edge;
  std::map<std::pair<std::string, std::string>, Eigen::MatrixXd>
      cross_covariances;

  const GaussianLatentFactor* factor(const std::string& factor_id) const;
  const std::vector<EdgeLatentBinding>* bindingsForEdge(
      const std::string& edge_id) const;

  bool jointMeanCovariance(
      const std::vector<std::string>& factor_ids,
      Eigen::VectorXd* mean,
      Eigen::MatrixXd* covariance,
      std::map<std::string, std::pair<int, int>>* offsets,
      std::string* error = nullptr) const;
};

class GaussianLatentStore {
 public:
  GaussianLatentStore() = default;

  GaussianLatentStore(const GaussianLatentStore&) = delete;
  GaussianLatentStore& operator=(const GaussianLatentStore&) = delete;

  std::uint64_t revision() const;
  GaussianLatentSnapshot snapshot() const;

  bool putFactor(const std::string& factor_id,
                 const Eigen::VectorXd& mean,
                 const Eigen::MatrixXd& covariance,
                 const ros::Time& stamp,
                 GaussianLatentFactor* output = nullptr,
                 std::string* error = nullptr,
                 const LatentProvenance& provenance = LatentProvenance());

  bool bindEdge(const EdgeLatentBinding& binding,
                std::string* error = nullptr);

  bool applyObservation(
      const GaussianObservationFactor& observation,
      GaussianUpdateResult* output = nullptr,
      std::string* error = nullptr,
      const std::map<std::string, std::uint64_t>& expected_versions = {});

 private:
  void refreshBindingVersionLocked(const std::string& factor_id,
                                   std::uint64_t version);
  void dropCrossCovariancesLocked(const std::string& factor_id);

  mutable std::mutex mutex_;
  std::uint64_t revision_ = 0;
  std::map<std::string, GaussianLatentFactor> factors_;
  std::map<std::string, std::vector<EdgeLatentBinding>> bindings_by_edge_;
  std::map<std::pair<std::string, std::string>, Eigen::MatrixXd>
      cross_covariances_;
};

}  // namespace probtf_core
