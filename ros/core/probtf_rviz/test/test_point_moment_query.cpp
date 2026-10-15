#include <gtest/gtest.h>

#include <probtf_rviz/point_moment_query.hpp>

#include <probtf_msgs/BinghamOrientation.h>

#include <cmath>
#include <limits>

namespace {

using Record = probtf_msgs::ProbabilisticTransformStamped;

Record nativeRecord(const std::string& parent, const std::string& child,
                    const Eigen::Vector3d& translation) {
  Record output;
  output.header.frame_id = parent;
  output.child_frame_id = child;
  output.header.stamp = ros::Time(123, 456);
  output.edge_id = parent + "_to_" + child;
  output.authority = "indet_ik";
  output.provenance.method = "configured_native_local_transform_family";
  output.provenance.source_ids.push_back("same_native_law_revision");
  output.provenance.detail = "{\"command\":[0.1,0.2],\"revision\":7,\"moment_integration_steps\":480}";
  output.components.resize(1);
  auto& component = output.components.front();
  component.component_id = child;
  component.weight = 1.0;
  component.orientation.kind = probtf_msgs::BinghamOrientation::DIRAC;
  component.orientation.reference_quaternion.w = 1.0;
  component.orientation.shape_upper_wxyz = {{1.5, 0., 0., 0., -0.5, 0., 0., -0.5, 0., -0.5}};
  component.translation.mean_at_reference.x = translation.x();
  component.translation.mean_at_reference.y = translation.y();
  component.translation.mean_at_reference.z = translation.z();
  return output;
}

void setFiniteOrientation(double angle, const Eigen::Vector3d& axis,
                          const Eigen::Vector4d& eigenvalues, double concentration,
                          probtf_msgs::BinghamOrientation* output) {
  const Eigen::Quaterniond mode(Eigen::AngleAxisd(angle, axis.normalized()));
  Eigen::Matrix4d left;
  left << mode.w(), -mode.x(), -mode.y(), -mode.z(),
          mode.x(), mode.w(), -mode.z(), mode.y(),
          mode.y(), mode.z(), mode.w(), -mode.x(),
          mode.z(), -mode.y(), mode.x(), mode.w();
  const double shape_magnitude = eigenvalues(0) + eigenvalues(1);
  const Eigen::Matrix4d shape = left * eigenvalues.asDiagonal() * left.transpose() / shape_magnitude;
  output->kind = probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
  output->inverse_concentration = 1.0 / (concentration * shape_magnitude);
  output->reference_quaternion.w = mode.w();
  output->reference_quaternion.x = mode.x();
  output->reference_quaternion.y = mode.y();
  output->reference_quaternion.z = mode.z();
  std::size_t index = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = row; column < 4; ++column) {
      output->shape_upper_wxyz[index++] = shape(row, column);
    }
  }
}

probtf_msgs::ProbabilisticTransformArray nativePath(bool coupled) {
  probtf_msgs::ProbabilisticTransformArray records;
  records.transforms.push_back(nativeRecord("middle", "tip", Eigen::Vector3d(0.23, -0.11, 0.17)));
  records.transforms.push_back(nativeRecord("base", "middle", Eigen::Vector3d(-0.31, 0.24, 0.08)));
  setFiniteOrientation(0.61, Eigen::Vector3d(1., 2., -0.5),
      Eigen::Vector4d(1., 0., -0.4, -0.6), 400., &records.transforms[0].components[0].orientation);
  setFiniteOrientation(-0.48, Eigen::Vector3d(-0.2, 0.7, 1.),
      Eigen::Vector4d(0.8, 0.1, -0.2, -0.7), 320., &records.transforms[1].components[0].orientation);
  records.transforms[0].components[0].translation.residual_covariance_upper =
      {{4e-6, 1e-6, -0.5e-6, 9e-6, 0.6e-6, 16e-6}};
  records.transforms[1].components[0].translation.residual_covariance_upper =
      {{8e-6, -0.8e-6, 0.7e-6, 3e-6, 0.4e-6, 12e-6}};
  if (coupled) {
    for (std::size_t index = 0; index < 2; ++index) {
      auto& coupling = records.transforms[index].components[0].translation.rotation_coupling;
      coupling[0] = 0.031 * (index + 1);
      coupling[13] = -0.024 * (index + 1);
      coupling[26] = 0.017 * (index + 1);
      coupling[5] = -0.013;
    }
  }
  return records;
}

void expectMomentsNear(const probtf_rviz::AxisPointMoments& actual,
                       const std::array<Eigen::Vector3d, 3>& means,
                       const std::array<Eigen::Matrix3d, 3>& covariance) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_LT((actual[axis].mean - means[axis]).cwiseAbs().maxCoeff(), 2e-10);
    EXPECT_LT((actual[axis].covariance - covariance[axis]).cwiseAbs().maxCoeff(), 2e-10);
  }
}

TEST(PointMomentDisplayQuery, NativeFullPathMatchesIndependentPythonM2M4Moments) {
  // Gold from Python forward_component_point_moments(..., 480), starting at
  // each .12*e_i and applying the child then root component, without any
  // intermediate Gaussian-x-Bingham projection or sampling.
  const auto records = nativePath(false);
  probtf_msgs::ProbabilisticTransformArray no_static;
  probtf_core::LatestSnapshot snapshot(records, no_static);
  probtf_rviz::AxisPointMoments actual;
  std::string error;
  ASSERT_TRUE(probtf_rviz::queryAxisPointMoments(snapshot, "base", "tip", 0.12, 480, &actual, &error)) << error;
  const std::array<Eigen::Vector3d, 3> means = {{
      {-0.087682618192880535, 0.0037102221872792329, 0.2499274552849603},
      {-0.13794615635870058, 0.14293637089066047, 0.32932107644849151},
      {-0.18091642674010755, -0.0079795060847390464, 0.3889564430382999}}};
  std::array<Eigen::Matrix3d, 3> covariance;
  covariance[0] << 0.00045554738632574228, 0.00022460996136033969, -0.00025721481319763132,
                   0.00022460996136033969, 0.00042859890172687609, 0.00023976268302565177,
                  -0.00025721481319763132, 0.00023976268302565177, 0.00076726195954278018;
  covariance[1] << 0.00050006051220693776, 8.0326244163123384e-5, -0.00027768834728647703,
                   8.0326244163123384e-5, 0.00058339939412843059, 0.00012392137312496334,
                  -0.00027768834728647703, 0.00012392137312496334, 0.00033310129427330926;
  covariance[2] << 0.00095612546120835834, 0.00014810371766178253, -0.00026095574127826349,
                   0.00014810371766178253, 0.00084053156238110105, 0.00058834942874415178,
                  -0.00026095574127826349, 0.00058834942874415178, 0.00060683320006635864;
  expectMomentsNear(actual, means, covariance);

  std::mt19937 generator(114);
  std::vector<probtf_rviz::ColoredPoint> points;
  ASSERT_TRUE(probtf_rviz::sampleAxisPointGaussians(actual, 40000, &generator, &points, &error)) << error;
  ASSERT_EQ(points.size(), 120000U);
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (std::size_t index = axis * 40000U; index < (axis + 1) * 40000U; ++index) {
      const auto centered = (points[index].position - actual[axis].mean).eval();
      mean += centered;
      covariance += centered * centered.transpose();
      EXPECT_EQ(points[index].color, probtf_rviz::axisColor(axis));
    }
    mean /= 40000.;
    covariance /= 40000.;
    EXPECT_LT(mean.cwiseAbs().maxCoeff(), 5e-4);
    EXPECT_LT((covariance - actual[axis].covariance).cwiseAbs().maxCoeff(), 2.5e-5);
  }
}

TEST(PointMomentDisplayQuery, NativeConditionalTranslationCouplingMatchesPython) {
  const auto records = nativePath(true);
  probtf_msgs::ProbabilisticTransformArray no_static;
  probtf_core::LatestSnapshot snapshot(records, no_static);
  probtf_rviz::AxisPointMoments actual;
  std::string error;
  ASSERT_TRUE(probtf_rviz::queryAxisPointMoments(snapshot, "base", "tip", 0.12, 480, &actual, &error)) << error;
  const std::array<Eigen::Vector3d, 3> means = {{
      {-0.087977822436967779, 0.0041174671754177217, 0.24961419071901339},
      {-0.13824136060278788, 0.1433436158787989, 0.32900781188254458},
      {-0.18121163098419482, -0.0075722610966005299, 0.38864317847235308}}};
  std::array<Eigen::Matrix3d, 3> covariance;
  covariance[0] << 0.00055255143778555171, 0.00023669385585470027, -0.0002500558302400874,
                   0.00023669385585470027, 0.00039138957657663406, 0.0002290032716360011,
                  -0.0002500558302400874, 0.0002290032716360011, 0.00075668085890746384;
  covariance[1] << 0.00059357972434256477, 0.00011026776600810306, -0.00027844466668821785,
                   0.00011026776600810306, 0.00053992362999081081, 0.00013005699243510884,
                  -0.00027844466668821785, 0.00013005699243510884, 0.0003107556020410422;
  covariance[2] << 0.0010880613058736155, 0.00017029087933139345, -0.00023572228810942049,
                   0.00017029087933139345, 0.00079462978220346263, 0.00058504986142188015,
                  -0.00023572228810942049, 0.00058504986142188015, 0.00060532137073038153;
  expectMomentsNear(actual, means, covariance);
}

TEST(PointMomentDisplayQuery, SingularAndDiracQueriesAddNoArtificialVariance) {
  probtf_rviz::AxisPointMoments moments;
  moments[0].mean = Eigen::Vector3d(0.2, -0.3, 0.4);
  moments[1].covariance(1, 1) = 0.001;
  std::mt19937 generator(1);
  std::string error;
  std::vector<probtf_rviz::ColoredPoint> points;
  ASSERT_TRUE(probtf_rviz::sampleAxisPointGaussians(moments, 100, &generator, &points, &error)) << error;
  for (std::size_t index = 0; index < 100; ++index) {
    EXPECT_TRUE(points[index].position.isApprox(moments[0].mean, 0.));
    EXPECT_DOUBLE_EQ(points[index + 100].position.x(), 0.);
    EXPECT_DOUBLE_EQ(points[index + 100].position.z(), 0.);
    EXPECT_TRUE(points[index + 200].position.isZero(0.));
  }
  moments[1].covariance(0, 0) = -0.001;
  EXPECT_FALSE(probtf_rviz::sampleAxisPointGaussians(moments, 100, &generator, &points, &error));
  EXPECT_EQ(points.size(), 300U);
}

TEST(PointMomentDisplayQuery, MixedSourceContextAndInvalidQueriesAreRejected) {
  auto records = nativePath(false);
  auto& child = records.transforms[0];
  auto& root = records.transforms[1];
  const std::vector<const Record*> path{&child, &root};
  std::string error;
  ASSERT_TRUE(probtf_rviz::sameNativeLocalLawContext(path, &error)) << error;
  const auto original = root;
  root.header.stamp.nsec++;
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  root = original;
  root.provenance.detail = "{\"command\":[1,2]}";
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  root = original;
  root.provenance.source_ids[0] = "different_law";
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  root = original;
  root.authority = "different_estimator";
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  root = original;
  root.child_frame_id = "different_parent";
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  root = original;
  root.provenance.method = "indet_ik_product_gaussian_bingham";
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext(path, &error));
  EXPECT_FALSE(probtf_rviz::sameNativeLocalLawContext({nullptr}, &error));

  int steps;
  EXPECT_TRUE(probtf_rviz::pointMomentIntegrationSteps("", &steps, &error));
  EXPECT_EQ(steps, 480);
  EXPECT_TRUE(probtf_rviz::pointMomentIntegrationSteps("{\"moment_integration_steps\":240}", &steps, &error));
  EXPECT_EQ(steps, 240);
  EXPECT_FALSE(probtf_rviz::pointMomentIntegrationSteps("{\"moment_integration_steps\":0}", &steps, &error));
  EXPECT_FALSE(probtf_rviz::pointMomentIntegrationSteps("{\"moment_integration_steps\":3.5}", &steps, &error));
  EXPECT_FALSE(probtf_rviz::pointMomentIntegrationSteps("invalid json", &steps, &error));
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
