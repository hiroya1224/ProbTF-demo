#include "../src/quadrature_display.hpp"

#include <gtest/gtest.h>

#include <QJsonObject>

#include <cmath>

namespace {

using Record = probtf_msgs::ProbabilisticTransformStamped;

Record query() {
  Record record;
  record.header.frame_id = "base";
  record.header.stamp = ros::Time(12, 500000000U);
  record.child_frame_id = "tip";
  record.edge_id = "base_to_tip_query";
  record.provenance.method = "joint_pushforward_quadrature";
  record.provenance.derived_from_edge_ids = {"body_law"};
  QJsonObject object;
  object["result_kind"] = "joint_pushforward_quadrature";
  object["source_law_id"] = "body_law";
  object["source_revision"] = 2;
  object["query_id"] = "command_4";
  object["order"] = 3;
  object["source_observation_stamp"] = 12.5;
  object["display_result_kind"] = "frozen_law_query";
  record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
  for (int index = 0; index < 2; ++index) {
    probtf_msgs::ProbabilisticTransformComponent component;
    component.component_id = "node_" + std::to_string(index);
    component.weight = index == 0 ? .25 : .75;
    component.orientation.kind = component.orientation.DIRAC;
    component.orientation.reference_quaternion.w = index == 0 ? 1. : std::sqrt(.5);
    component.orientation.reference_quaternion.z = index == 0 ? 0. : std::sqrt(.5);
    component.translation.mean_at_reference.x = index == 0 ? 1. : -1.;
    component.translation.mean_at_reference.y = index == 0 ? 2. : 0.;
    component.translation.mean_at_reference.z = index == 0 ? 3. : 1.;
    record.components.push_back(component);
  }
  return record;
}

TEST(QuadratureDisplay, EveryWeightedNodeProducesExactlyThreeEndpoints) {
  const Record record = query();
  probtf_rviz::QuadratureEndpoints points;
  std::string error;
  ASSERT_TRUE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &points, &error)) << error;
  ASSERT_EQ(points.size(), 6U);
  EXPECT_NEAR((points[0].position - Eigen::Vector3d(1.1, 2., 3.)).norm(), 0., 1e-14);
  EXPECT_NEAR((points[1].position - Eigen::Vector3d(1., 2.1, 3.)).norm(), 0., 1e-14);
  EXPECT_NEAR((points[3].position - Eigen::Vector3d(-1., .1, 1.)).norm(), 0., 1e-14);
  EXPECT_NEAR((points[4].position - Eigen::Vector3d(-1.1, 0., 1.)).norm(), 0., 1e-14);
  EXPECT_DOUBLE_EQ(points[0].normalized_weight, .25);
  EXPECT_DOUBLE_EQ(points[3].normalized_weight, .75);
  EXPECT_FLOAT_EQ(points[0].relative_alpha, 1.F / 3.F);
  EXPECT_FLOAT_EQ(points[3].relative_alpha, 1.F);
  probtf_rviz::QuadratureEndpoints again;
  ASSERT_TRUE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &again, &error));
  for (std::size_t index = 0; index < points.size(); ++index) {
    EXPECT_TRUE(points[index].position == again[index].position);
    EXPECT_EQ(points[index].axis, static_cast<int>(index % 3U));
  }
}

TEST(QuadratureDisplay, WeightScaleAndZeroMassDoNotAddOrMoveNodes) {
  Record record = query();
  for (auto& component : record.components) {
    component.weight *= 1.e200;
  }
  record.components.push_back(record.components.front());
  record.components.back().component_id = "zero";
  record.components.back().weight = 0.;
  probtf_rviz::QuadratureEndpoints points;
  ASSERT_TRUE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &points));
  ASSERT_EQ(points.size(), 6U);
  EXPECT_DOUBLE_EQ(points[0].normalized_weight, .25);
  EXPECT_FLOAT_EQ(points[0].relative_alpha, 1.F / 3.F);
}

TEST(QuadratureDisplay, InvalidOrNonDeterministicNodesAreUnavailable) {
  for (int failure = 0; failure < 4; ++failure) {
    Record record = query();
    if (failure == 0) record.components[0].orientation.kind = record.components[0].orientation.FINITE_BINGHAM;
    if (failure == 1) record.components[0].translation.residual_covariance_upper[0] = .001;
    if (failure == 2) record.components[0].weight = -.1;
    if (failure == 3) record.components[0].orientation.reference_quaternion.w = 2.;
    probtf_rviz::QuadratureEndpoints points;
    std::string error;
    EXPECT_FALSE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &points, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(points.empty());
  }
}

TEST(QuadratureDisplay, PointLimitFailsWithoutRandomSubsampling) {
  probtf_rviz::QuadratureEndpoints points;
  std::string error;
  EXPECT_FALSE(probtf_rviz::quadratureEndpoints(query(), .1, 5U, &points, &error));
  EXPECT_NE(error.find("no nodes were resampled"), std::string::npos);
}

TEST(QuadratureDisplay, QueryMetadataPreservesSourceTimeAndResolvableLawIdentity) {
  Record record = query();
  probtf_rviz::RecordDisplayMetadata metadata;
  ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  EXPECT_TRUE(metadata.frozen_query);
  EXPECT_TRUE(metadata.deterministic_quadrature);
  EXPECT_EQ(metadata.source_law_id, "body_law");
  EXPECT_EQ(metadata.query_id, "command_4");
  EXPECT_EQ(metadata.source_revision, 2.);
  EXPECT_EQ(metadata.order, 3);
  record.header.stamp = ros::Time(13, 0U);
  EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  record = query();
  record.provenance.derived_from_edge_ids.clear();
  EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  record = query();
  record.provenance.detail = "not structured metadata";
  EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
}

TEST(QuadratureDisplay, FrozenViewDoesNotAuthorizeIndependentPathComposition) {
  const Record first = query(), second = query();
  probtf_rviz::RecordDisplayMetadata metadata;
  std::string error;
  EXPECT_FALSE(probtf_rviz::pathDisplayMetadata({&first, &second}, &metadata, &error));
  EXPECT_NE(error.find("resolved joint pushforward"), std::string::npos);
  ASSERT_TRUE(probtf_rviz::pathDisplayMetadata({&first}, &metadata));
  EXPECT_TRUE(metadata.deterministic_quadrature);
}

TEST(QuadratureDisplay, ExactFiniteSumHasNoIntegrationOrder) {
  Record record = query();
  QJsonObject object = QJsonDocument::fromJson(
      QByteArray::fromStdString(record.provenance.detail)).object();
  object["integration_method"] = "finite_sum";
  object["order"] = QJsonValue(QJsonValue::Null);
  record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
  probtf_rviz::RecordDisplayMetadata metadata;
  ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  EXPECT_EQ(metadata.order, 0);
  probtf_rviz::QuadratureEndpoints points;
  ASSERT_TRUE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &points));
  EXPECT_EQ(points.size(), 6U);
  object.remove("integration_method");
  record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
  EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  object["integration_method"] = "tensor_gauss_legendre_log_coordinates";
  record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
  EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
}

TEST(QuadratureDisplay, OrdinaryNativeLawRemainsSeparateFromExplicitFrozenView) {
  Record record;
  record.header.stamp = ros::Time(12, 500000000U);
  record.provenance.method = "image_estimator";
  probtf_rviz::RecordDisplayMetadata metadata;
  ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  EXPECT_FALSE(metadata.frozen_query);
  EXPECT_FALSE(metadata.deterministic_quadrature);
  record.provenance.detail = "{\"display_result_kind\":\"frozen_law_query\",\"source_observation_stamp\":12.5}";
  ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
  EXPECT_TRUE(metadata.frozen_query);
  EXPECT_FALSE(metadata.deterministic_quadrature);
}

TEST(QuadratureDisplay, ExactFiniteAtomsAndJointSamplesAreRenderedOnceEach) {
  for (const std::string kind : {"joint_finite_pushforward", "joint_transform_samples"}) {
    Record record = query();
    QJsonObject object = QJsonDocument::fromJson(
        QByteArray::fromStdString(record.provenance.detail)).object();
    object["result_kind"] = QString::fromStdString(kind);
    object.remove("order");
    object["integration_method"] = kind == "joint_finite_pushforward" ? "finite_sum" : "monte_carlo";
    if (kind == "joint_transform_samples") {
      object["sample_count"] = 2;
      object["seed"] = 21;
    }
    record.provenance.method = kind;
    record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
    probtf_rviz::RecordDisplayMetadata metadata;
    ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
    EXPECT_TRUE(metadata.direct_support);
    EXPECT_FALSE(metadata.deterministic_quadrature);
    probtf_rviz::QuadratureEndpoints points;
    ASSERT_TRUE(probtf_rviz::quadratureEndpoints(record, .1, 100U, &points));
    EXPECT_EQ(points.size(), 6U);
    EXPECT_FLOAT_EQ(points[0].relative_alpha, 1.F / 3.F);
    EXPECT_FALSE(probtf_rviz::pathDisplayMetadata({&record, &record}, &metadata));
    ASSERT_TRUE(probtf_rviz::recordDisplayMetadata(record, &metadata));
    const std::string description = probtf_rviz::supportDisplayDescription(metadata, 2).toStdString();
    if (kind == "joint_finite_pushforward") {
      EXPECT_NE(description.find("exact finite atoms"), std::string::npos);
    } else {
      EXPECT_NE(description.find("2 joint Monte Carlo samples"), std::string::npos);
      EXPECT_NE(description.find("method=monte_carlo, seed=21"), std::string::npos);
      EXPECT_NE(description.find("source=body_law v2"), std::string::npos);
      object["sample_count"] = 3;
      record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
      EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
      object["sample_count"] = 2;
      object.remove("seed");
      record.provenance.detail = QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
      EXPECT_FALSE(probtf_rviz::recordDisplayMetadata(record, &metadata));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
