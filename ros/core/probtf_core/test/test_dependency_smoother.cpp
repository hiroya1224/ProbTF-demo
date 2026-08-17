#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <probtf_core/gaussian_latent_store.hpp>
#include <probtf_core/latest_snapshot.hpp>
#include <probtf_core/transform_moments.hpp>

#include <memory>
#include <string>

namespace {

geometry_msgs::TransformStamped transformMessage(
    const std::string& parent,
    const std::string& child,
    double stamp,
    const Eigen::Isometry3d& transform) {
  geometry_msgs::TransformStamped output;
  output.header.frame_id = parent;
  output.header.stamp = ros::Time(stamp);
  output.child_frame_id = child;
  output.transform.translation.x = transform.translation().x();
  output.transform.translation.y = transform.translation().y();
  output.transform.translation.z = transform.translation().z();
  const Eigen::Quaterniond rotation(transform.rotation());
  output.transform.rotation.w = rotation.w();
  output.transform.rotation.x = rotation.x();
  output.transform.rotation.y = rotation.y();
  output.transform.rotation.z = rotation.z();
  return output;
}

probtf_msgs::ProbabilisticTransformStamped exactRecord(
    const std::string& parent,
    const std::string& child,
    double stamp,
    const Eigen::Isometry3d& transform) {
  probtf_msgs::ProbabilisticTransformStamped output;
  std::string error;
  EXPECT_TRUE(probtf_core::deterministicTfToProbTf(
      transformMessage(parent, child, stamp, transform), "test", false,
      &output, &error))
      << error;
  return output;
}

Eigen::Isometry3d pose(const Eigen::Vector3d& translation,
                       const Eigen::AngleAxisd& rotation =
                           Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX())) {
  Eigen::Isometry3d output = Eigen::Isometry3d::Identity();
  output.translation() = translation;
  output.linear() = rotation.toRotationMatrix();
  return output;
}

probtf_core::EdgeLatentBinding binding(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    const probtf_core::GaussianLatentFactor& factor,
    const Eigen::MatrixXd& sensitivity,
    const Eigen::Isometry3d& linearization_pose) {
  probtf_core::EdgeLatentBinding output;
  output.edge_id = record.edge_id;
  output.factor_id = factor.factor_id;
  output.sensitivity = sensitivity;
  output.factor_version = factor.version;
  output.linearization_stamp = record.header.stamp;
  output.linearization_pose = linearization_pose;
  return output;
}

void expectMatrixNear(const Eigen::MatrixXd& actual,
                      const Eigen::MatrixXd& expected,
                      double tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (int row = 0; row < actual.rows(); ++row) {
    for (int column = 0; column < actual.cols(); ++column) {
      EXPECT_NEAR(actual(row, column), expected(row, column), tolerance);
    }
  }
}

TEST(GaussianLatentStore, ObservationPreservesNegativeCrossCovariance) {
  probtf_core::GaussianLatentStore store;
  probtf_core::GaussianLatentFactor factor;
  std::string error;
  ASSERT_TRUE(store.putFactor(
      "joint", Eigen::Vector2d::Zero(),
      0.4 * Eigen::Matrix2d::Identity(), ros::Time(0.0), &factor, &error))
      << error;
  probtf_core::GaussianObservationFactor observation;
  observation.observation_id = "sum";
  observation.latent_factor_ids.push_back("joint");
  observation.residual = Eigen::VectorXd::Zero(1);
  observation.jacobian_blocks.emplace_back(1, 2);
  observation.jacobian_blocks.back() << 1.0, 1.0;
  observation.noise_covariance =
      1.0e-12 * Eigen::MatrixXd::Identity(1, 1);
  observation.stamp = ros::Time(1.0);
  probtf_core::GaussianUpdateResult update;
  ASSERT_TRUE(store.applyObservation(observation, &update, &error)) << error;
  const auto snapshot = store.snapshot();
  const auto* posterior = snapshot.factor("joint");
  ASSERT_NE(posterior, nullptr);
  Eigen::Matrix2d expected;
  expected << 0.2, -0.2, -0.2, 0.2;
  expectMatrixNear(posterior->covariance, expected, 2.0e-12);
  EXPECT_EQ(posterior->version, 2U);
  ASSERT_EQ(update.posterior_versions.size(), 1U);
}

TEST(TransformMoments, InverseJacobianMatchesFiniteDifference) {
  const Eigen::Isometry3d transform = pose(
      Eigen::Vector3d(0.4, -0.3, 0.2),
      Eigen::AngleAxisd(0.45, Eigen::Vector3d(1.0, 2.0, -1.0).normalized()));
  const Eigen::Isometry3d inverse = transform.inverse();
  const probtf_core::Matrix6d analytic =
      probtf_core::inverseMixedPoseJacobian(transform);
  probtf_core::Matrix6d numerical = probtf_core::Matrix6d::Zero();
  const double step = 1.0e-7;
  for (int column = 0; column < 6; ++column) {
    Eigen::Matrix<double, 6, 1> perturbation =
        Eigen::Matrix<double, 6, 1>::Zero();
    perturbation(column) = step;
    const Eigen::Isometry3d changed =
        probtf_core::applyMixedPosePerturbation(transform, perturbation)
            .inverse();
    numerical.block<3, 1>(0, column) =
        (changed.translation() - inverse.translation()) / step;
    const Eigen::AngleAxisd rotation_difference(
        inverse.rotation().transpose() * changed.rotation());
    numerical.block<3, 1>(3, column) =
        rotation_difference.axis() * rotation_difference.angle() / step;
  }
  expectMatrixNear(analytic, numerical, 8.0e-8);
}

TEST(TransformMoments, ResolvesRepeatedFactorAndUsesUpdatedPosterior) {
  const Eigen::Isometry3d first_pose =
      pose(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Eigen::Isometry3d second_pose =
      pose(Eigen::Vector3d(0.0, 1.0, 0.0));
  auto first = exactRecord("world", "a", 1.0, first_pose);
  auto second = exactRecord("a", "tool", 1.0, second_pose);
  first.provenance.derived_from_edge_ids.push_back("shared");
  second.provenance.derived_from_edge_ids.push_back("shared");
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  dynamic_records.transforms.push_back(first);
  dynamic_records.transforms.push_back(second);
  probtf_msgs::ProbabilisticTransformArray static_records;

  auto store = std::make_shared<probtf_core::GaussianLatentStore>();
  probtf_core::GaussianLatentFactor factor;
  std::string error;
  ASSERT_TRUE(store->putFactor(
      "shared", Eigen::VectorXd::Zero(1),
      0.04 * Eigen::MatrixXd::Identity(1, 1), ros::Time(1.0), &factor,
      &error))
      << error;
  Eigen::MatrixXd sensitivity = Eigen::MatrixXd::Zero(6, 1);
  sensitivity(0, 0) = 1.0;
  ASSERT_TRUE(
      store->bindEdge(binding(first, factor, sensitivity, first_pose), &error))
      << error;

  probtf_core::LatestSnapshot incomplete(dynamic_records, static_records,
                                         store);
  probtf_core::TransformMomentObservation output;
  EXPECT_FALSE(
      incomplete.lookupTransformMoments("world", "tool", &output, &error));
  EXPECT_NE(error.find("lacks"), std::string::npos);

  ASSERT_TRUE(store->bindEdge(
      binding(second, factor, sensitivity, second_pose), &error))
      << error;
  probtf_core::LatestSnapshot runtime(dynamic_records, static_records, store);
  ASSERT_TRUE(
      runtime.lookupTransformMoments("world", "tool", &output, &error))
      << error;
  EXPECT_NEAR(output.moments.mean.translation().x(), 1.0, 1.0e-12);
  EXPECT_NEAR(output.moments.mean.translation().y(), 1.0, 1.0e-12);
  EXPECT_NEAR(output.moments.covariance(0, 0), 0.16, 1.0e-12);
  ASSERT_EQ(output.moments.factor_versions.size(), 1U);
  EXPECT_EQ(output.moments.factor_versions.front().second, 1U);
  EXPECT_EQ(output.moments.approximation.kind,
            probtf_msgs::ApproximationInfo::MOMENT_SUMMARY);
  EXPECT_TRUE(output.moments.approximation.lossy);
  EXPECT_EQ(output.moments.approximation.source,
            "probtf.dependency.DependencyAwareMomentEvaluator");
  EXPECT_EQ(output.moments.provenance.method,
            "dependency_aware_local_gaussian_moments");
  EXPECT_EQ(output.moments.provenance.detail,
            probtf_core::kPosePerturbationConvention);
  EXPECT_EQ(output.moments.provenance.derived_from_edge_ids,
            output.edge_ids);
  ASSERT_EQ(output.moments.diagnostics.size(), 1U);
  EXPECT_EQ(output.moments.diagnostics.front(),
            "resolved repeated dependencies: shared");

  probtf_core::GaussianObservationFactor observation;
  observation.observation_id = "camera";
  observation.latent_factor_ids.push_back("shared");
  observation.residual = Eigen::VectorXd::Zero(1);
  observation.jacobian_blocks.push_back(Eigen::MatrixXd::Identity(1, 1));
  observation.noise_covariance =
      0.01 * Eigen::MatrixXd::Identity(1, 1);
  observation.stamp = ros::Time(2.0);
  ASSERT_TRUE(store->applyObservation(observation, nullptr, &error)) << error;
  ASSERT_TRUE(
      runtime.lookupTransformMoments("world", "tool", &output, &error))
      << error;
  EXPECT_LT(output.moments.covariance(0, 0), 0.04);
  EXPECT_EQ(output.moments.factor_versions.front().second, 2U);
}

TEST(TransformMoments, SupportsLocalGaussianStochasticInverse) {
  const Eigen::Isometry3d forward_pose = pose(
      Eigen::Vector3d(0.4, -0.2, 0.1),
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
  auto record = exactRecord("base", "tool", 3.0, forward_pose);
  record.components.front().translation.residual_covariance_upper[0] = 0.04;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  dynamic_records.transforms.push_back(record);
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_core::LatestSnapshot runtime(dynamic_records, static_records);
  probtf_core::TransformMomentObservation output;
  std::string error;
  ASSERT_TRUE(
      runtime.lookupTransformMoments("tool", "base", &output, &error))
      << error;
  probtf_core::Matrix6d physical = probtf_core::Matrix6d::Zero();
  physical(0, 0) = 0.04;
  const probtf_core::Matrix6d inverse =
      probtf_core::inverseMixedPoseJacobian(forward_pose);
  expectMatrixNear(output.moments.covariance,
                   inverse * physical * inverse.transpose(), 1.0e-12);
  expectMatrixNear(output.moments.mean.matrix(),
                   forward_pose.inverse().matrix(), 1.0e-12);
}

TEST(TransformMoments, ConcentratedFiniteBinghamUsesZeroMaximumGauge) {
  const Eigen::Isometry3d forward_pose =
      pose(Eigen::Vector3d(0.1, 0.2, -0.1));
  auto record = exactRecord("base", "tool", 4.0, forward_pose);
  auto& orientation = record.components.front().orientation;
  orientation.kind =
      probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
  orientation.inverse_concentration = 0.002;
  orientation.shape_upper_wxyz =
      {{0.75, 0.0, 0.0, 0.0, 0.25,
        0.0, 0.0, -0.25, 0.0, -0.75}};
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  dynamic_records.transforms.push_back(record);
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_core::LatestSnapshot runtime(dynamic_records, static_records);
  probtf_core::TransformMomentObservation output;
  std::string error;
  ASSERT_TRUE(
      runtime.lookupTransformMoments("base", "tool", &output, &error))
      << error;
  EXPECT_TRUE(output.moments.covariance.allFinite());
  EXPECT_GT((output.moments.covariance.block<3, 3>(3, 3).trace()), 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
