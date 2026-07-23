#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <probtf_core/latest_snapshot.hpp>
#include <probtf_core/sampling.hpp>

#include <probtf_msgs/BinghamOrientation.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>

namespace {

geometry_msgs::TransformStamped transform(const std::string& parent,
                                          const std::string& child,
                                          double stamp,
                                          const Eigen::Vector3d& translation,
                                          const Eigen::Quaterniond& rotation) {
  geometry_msgs::TransformStamped output;
  output.header.frame_id = parent;
  output.header.stamp = ros::Time(stamp);
  output.child_frame_id = child;
  output.transform.translation.x = translation.x();
  output.transform.translation.y = translation.y();
  output.transform.translation.z = translation.z();
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
    const Eigen::Vector3d& translation,
    const Eigen::Quaterniond& rotation,
    bool is_static = false) {
  probtf_msgs::ProbabilisticTransformStamped output;
  std::string error;
  EXPECT_TRUE(probtf_core::deterministicTfToProbTf(
      transform(parent, child, stamp, translation, rotation), "test", is_static,
      &output, &error))
      << error;
  return output;
}

probtf_msgs::ProbabilisticTransformComponent component(
    const Eigen::Vector3d& translation,
    double weight) {
  const auto record = exactRecord("base", "temporary", 1.0, translation,
                                  Eigen::Quaterniond::Identity());
  auto output = record.components.front();
  output.weight = weight;
  return output;
}

void expectVectorNear(const Eigen::Vector3d& actual,
                      const Eigen::Vector3d& expected,
                      double tolerance) {
  for (int index = 0; index < 3; ++index) {
    EXPECT_NEAR(actual(index), expected(index), tolerance);
  }
}

void expectMatrixNear(const Eigen::Matrix3d& actual,
                      const Eigen::Matrix3d& expected,
                      double tolerance) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      EXPECT_NEAR(actual(row, column), expected(row, column), tolerance);
    }
  }
}

TEST(LatestSnapshot, ComposesExactForwardPath) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  static_records.transforms.push_back(exactRecord(
      "base", "ref/base", 0.0, Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Quaterniond::Identity(), true));
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  dynamic_records.transforms.push_back(exactRecord(
      "ref/base", "ref/tool", 2.0, Eigen::Vector3d(0.0, 2.0, 0.0),
      Eigen::Quaterniond(Eigen::AngleAxisd(
          0.5 * 3.14159265358979323846, Eigen::Vector3d::UnitZ()))));

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::PointMomentObservation observation;
  std::string error;
  ASSERT_TRUE(snapshot.lookupPointMoments(
      "base", "ref/tool", Eigen::Vector3d(1.0, 0.0, 0.0), &observation,
      &error))
      << error;
  expectVectorNear(observation.moments.mean, Eigen::Vector3d(1.0, 3.0, 0.0),
                   1.0e-12);
  expectMatrixNear(observation.moments.covariance, Eigen::Matrix3d::Zero(),
                   1.0e-12);
  EXPECT_DOUBLE_EQ(observation.resolved_stamp.toSec(), 2.0);

  probtf_core::TransformPathObservation path_observation;
  ASSERT_TRUE(snapshot.lookupPathMetadata(
      "base", "ref/tool", &path_observation, &error))
      << error;
  EXPECT_EQ(path_observation.target_frame, "base");
  EXPECT_EQ(path_observation.source_frame, "ref/tool");
  EXPECT_DOUBLE_EQ(path_observation.resolved_stamp.toSec(), 2.0);
  ASSERT_EQ(path_observation.edge_ids.size(), 2U);
}

TEST(LatestSnapshot, PathMetadataDoesNotRequireMomentEvaluation) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  auto record = exactRecord("base", "tool", 7.0, Eigen::Vector3d::Zero(),
                            Eigen::Quaterniond::Identity());
  auto& orientation = record.components.front().orientation;
  orientation.kind = probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
  orientation.inverse_concentration = 1.0e-100;
  orientation.shape_upper_wxyz =
      {{-2.0 / 3.0, 0.0, 0.0, 0.0, -1.0 / 3.0,
        0.0, 0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0}};
  dynamic_records.transforms.push_back(record);

  std::mt19937 generator(43);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      dynamic_records.transforms.front(), 4, &generator, &samples, &error))
      << error;

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::TransformPathObservation observation;
  ASSERT_TRUE(
      snapshot.lookupPathMetadata("base", "tool", &observation, &error))
      << error;
  EXPECT_DOUBLE_EQ(observation.resolved_stamp.toSec(), 7.0);
  ASSERT_EQ(observation.edge_ids.size(), 1U);
  EXPECT_EQ(observation.edge_ids.front(), record.edge_id);
}

TEST(LatestSnapshot, EvaluatesUniformOrientationMoments) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  auto record = exactRecord("base", "tool", 3.0, Eigen::Vector3d::Zero(),
                            Eigen::Quaterniond::Identity());
  auto& orientation = record.components.front().orientation;
  orientation.kind = probtf_msgs::BinghamOrientation::UNIFORM;
  orientation.inverse_concentration = std::numeric_limits<double>::infinity();
  std::fill(orientation.shape_upper_wxyz.begin(),
            orientation.shape_upper_wxyz.end(), 0.0);
  dynamic_records.transforms.push_back(record);

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::PointMomentObservation observation;
  std::string error;
  ASSERT_TRUE(snapshot.lookupPointMoments(
      "base", "tool", Eigen::Vector3d::UnitX(), &observation, &error))
      << error;
  expectVectorNear(observation.moments.mean, Eigen::Vector3d::Zero(), 1.0e-12);
  expectMatrixNear(observation.moments.covariance,
                   Eigen::Matrix3d::Identity() / 3.0, 1.0e-12);
}

TEST(LatestSnapshot, PreservesMixtureBetweenComponentCovariance) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  probtf_msgs::ProbabilisticTransformStamped record;
  record.header.frame_id = "base";
  record.header.stamp = ros::Time(4.0);
  record.child_frame_id = "tool";
  record.edge_id = "base__to__tool";
  record.is_static = false;
  record.components.push_back(component(Eigen::Vector3d::UnitX(), 1.0));
  record.components.push_back(component(-Eigen::Vector3d::UnitX(), 1.0));
  record.components[0].component_id = "positive";
  record.components[1].component_id = "negative";
  dynamic_records.transforms.push_back(record);

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::PointMomentObservation observation;
  std::string error;
  ASSERT_TRUE(snapshot.lookupPointMoments(
      "base", "tool", Eigen::Vector3d::Zero(), &observation, &error))
      << error;
  expectVectorNear(observation.moments.mean, Eigen::Vector3d::Zero(), 1.0e-12);
  Eigen::Matrix3d expected = Eigen::Matrix3d::Zero();
  expected(0, 0) = 1.0;
  expectMatrixNear(observation.moments.covariance, expected, 1.0e-12);
}

TEST(LatestSnapshot, MatchesPythonFiniteBinghamPointMoments) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  auto record = exactRecord("base", "tool", 5.0,
                            Eigen::Vector3d(0.1, -0.3, 0.2),
                            Eigen::Quaterniond::Identity());
  auto& component = record.components.front();
  component.orientation.kind =
      probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
  component.orientation.inverse_concentration = 0.5;
  const double shape[] = {-0.75, 0.0, 0.0, 0.0, -0.25,
                          0.0,   0.0, 0.25, 0.0, 0.75};
  std::copy(std::begin(shape), std::end(shape),
            component.orientation.shape_upper_wxyz.begin());
  component.orientation.reference_quaternion.w = 0.0;
  component.orientation.reference_quaternion.x = 0.0;
  component.orientation.reference_quaternion.y = 0.0;
  component.orientation.reference_quaternion.z = 1.0;
  component.translation.residual_covariance_upper =
      {{0.01, 0.0, 0.0, 0.02, 0.0, 0.03}};
  dynamic_records.transforms.push_back(record);

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::PointMomentObservation observation;
  std::string error;
  ASSERT_TRUE(snapshot.lookupPointMoments(
      "base", "tool", Eigen::Vector3d(0.4, -0.2, 0.7), &observation,
      &error, 120))
      << error;
  const Eigen::Vector3d expected_mean(-0.026118965764804214,
                                      -0.2662420753554137,
                                      0.2538317478947404);
  Eigen::Matrix3d expected_covariance;
  expected_covariance << 0.22119775040878326, 0.00048204371678367834,
      0.002593515928884053, 0.00048204371678367834,
      0.24771551822247423, -0.0007410164792143221,
      0.002593515928884053, -0.0007410164792143221,
      0.2611432832854462;
  expectVectorNear(observation.moments.mean, expected_mean, 2.0e-10);
  expectMatrixNear(observation.moments.covariance, expected_covariance,
                   2.0e-10);
}

TEST(LatestSnapshot, RejectsStochasticInverseMoments) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  auto record = exactRecord("base", "tool", 3.0, Eigen::Vector3d::Zero(),
                            Eigen::Quaterniond::Identity());
  record.components.front().translation.residual_covariance_upper[0] = 1.0;
  dynamic_records.transforms.push_back(record);

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::PointMomentObservation observation;
  std::string error;
  EXPECT_FALSE(snapshot.lookupPointMoments(
      "tool", "base", Eigen::Vector3d::Zero(), &observation, &error));
  EXPECT_NE(error.find("Stochastic inverse"), std::string::npos);
}

TEST(LatestSnapshot, RejectsRepeatedStochasticLatentDependencies) {
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  auto first = exactRecord("base", "middle", 6.0,
                           Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity());
  first.components.front().translation.residual_covariance_upper[0] = 0.1;
  auto second = exactRecord("middle", "tool", 6.0,
                            Eigen::Vector3d::Zero(),
                            Eigen::Quaterniond::Identity());
  second.provenance.derived_from_edge_ids.push_back(first.edge_id);
  dynamic_records.transforms.push_back(first);
  dynamic_records.transforms.push_back(second);

  probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
  probtf_core::TransformPathObservation path_observation;
  probtf_core::PointMomentObservation observation;
  std::string error;
  EXPECT_FALSE(snapshot.lookupPathMetadata(
      "base", "tool", &path_observation, &error));
  EXPECT_NE(error.find("Repeated latent"), std::string::npos);
  error.clear();
  EXPECT_FALSE(snapshot.lookupPointMoments(
      "base", "tool", Eigen::Vector3d::Zero(), &observation, &error));
  EXPECT_NE(error.find("Repeated latent"), std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
