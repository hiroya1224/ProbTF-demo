#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <probtf_core/sampling.hpp>

#include <probtf_msgs/BinghamOrientation.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using Matrix39d = Eigen::Matrix<double, 3, 9>;
using Matrix4d = Eigen::Matrix4d;
using Vector4d = Eigen::Vector4d;
using Vector9d = Eigen::Matrix<double, 9, 1>;

void assignQuaternion(const Eigen::Quaterniond& quaternion,
                      geometry_msgs::Quaternion* output) {
  output->w = quaternion.w();
  output->x = quaternion.x();
  output->y = quaternion.y();
  output->z = quaternion.z();
}

void assignVector(const Eigen::Vector3d& vector,
                  geometry_msgs::Vector3* output) {
  output->x = vector.x();
  output->y = vector.y();
  output->z = vector.z();
}

void packSymmetric4(const Matrix4d& matrix,
                    boost::array<double, 10>* output) {
  std::size_t index = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = row; column < 4; ++column) {
      (*output)[index++] = matrix(row, column);
    }
  }
}

void packSymmetric3(const Eigen::Matrix3d& matrix,
                    boost::array<double, 6>* output) {
  std::size_t index = 0;
  for (int row = 0; row < 3; ++row) {
    for (int column = row; column < 3; ++column) {
      (*output)[index++] = matrix(row, column);
    }
  }
}

Vector9d rotationVector(const Eigen::Matrix3d& rotation) {
  Vector9d output;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      output(row + 3 * column) = rotation(row, column);
    }
  }
  return output;
}

void setDiracOrientation(const Eigen::Quaterniond& input,
                         probtf_msgs::BinghamOrientation* output) {
  const Eigen::Quaterniond quaternion = input.normalized();
  output->kind = probtf_msgs::BinghamOrientation::DIRAC;
  output->inverse_concentration = 0.0;
  assignQuaternion(quaternion, &output->reference_quaternion);
  const Vector4d vector(quaternion.w(), quaternion.x(), quaternion.y(),
                       quaternion.z());
  packSymmetric4(
      2.0 * vector * vector.transpose() - 0.5 * Matrix4d::Identity(),
      &output->shape_upper_wxyz);
}

void setUniformOrientation(const Eigen::Quaterniond& reference,
                           probtf_msgs::BinghamOrientation* output) {
  output->kind = probtf_msgs::BinghamOrientation::UNIFORM;
  output->inverse_concentration = std::numeric_limits<double>::infinity();
  assignQuaternion(reference.normalized(), &output->reference_quaternion);
  std::fill(output->shape_upper_wxyz.begin(),
            output->shape_upper_wxyz.end(), 0.0);
}

void setFiniteOrientation(const Matrix4d& shape,
                          double inverse_concentration,
                          const Eigen::Quaterniond& reference,
                          probtf_msgs::BinghamOrientation* output) {
  output->kind = probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
  output->inverse_concentration = inverse_concentration;
  assignQuaternion(reference.normalized(), &output->reference_quaternion);
  packSymmetric4(shape, &output->shape_upper_wxyz);
}

void setTranslation(const Eigen::Vector3d& mean,
                    const Eigen::Matrix3d& covariance,
                    const Matrix39d& coupling,
                    probtf_msgs::ConditionalGaussianTranslation* output) {
  assignVector(mean, &output->mean_at_reference);
  packSymmetric3(covariance, &output->residual_covariance_upper);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 9; ++column) {
      output->rotation_coupling[static_cast<std::size_t>(row * 9 + column)] =
          coupling(row, column);
    }
  }
}

probtf_msgs::ProbabilisticTransformComponent component(
    const std::string& component_id,
    double weight,
    const Eigen::Vector3d& translation = Eigen::Vector3d::Zero()) {
  probtf_msgs::ProbabilisticTransformComponent output;
  output.component_id = component_id;
  output.weight = weight;
  setDiracOrientation(Eigen::Quaterniond::Identity(), &output.orientation);
  setTranslation(translation, Eigen::Matrix3d::Zero(), Matrix39d::Zero(),
                 &output.translation);
  return output;
}

probtf_msgs::ProbabilisticTransformStamped record() {
  probtf_msgs::ProbabilisticTransformStamped output;
  output.header.frame_id = "base";
  output.child_frame_id = "tool";
  output.edge_id = "base__to__tool";
  output.authority = "sampling_test";
  output.representative_kind =
      probtf_msgs::ProbabilisticTransformStamped::REPRESENTATIVE_NONE;
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

TEST(TransformDistributionSampling, DiracSamplesAreExact) {
  auto input = record();
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(0.5 * 3.14159265358979323846,
                        Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d translation(1.0, -2.0, 0.5);
  auto exact = component("exact", 1.0, translation);
  setDiracOrientation(rotation, &exact.orientation);
  input.components.push_back(exact);

  std::mt19937 generator(3);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 8, &generator, &samples, &error))
      << error;
  ASSERT_EQ(samples.size(), 8U);
  for (const auto& sample : samples) {
    expectVectorNear(sample.translation, translation, 0.0);
    expectMatrixNear(sample.rotation.toRotationMatrix(),
                     rotation.toRotationMatrix(), 1.0e-15);
  }
}

TEST(TransformDistributionSampling, UniformQuaternionsAreIsotropic) {
  auto input = record();
  auto uniform = component("uniform", 1.0);
  setUniformOrientation(Eigen::Quaterniond::Identity(),
                        &uniform.orientation);
  input.components.push_back(uniform);

  std::mt19937 generator(19);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 20000, &generator, &samples, &error))
      << error;
  Matrix4d second_moment = Matrix4d::Zero();
  for (const auto& sample : samples) {
    const Vector4d quaternion(sample.rotation.w(), sample.rotation.x(),
                              sample.rotation.y(), sample.rotation.z());
    EXPECT_NEAR(quaternion.norm(), 1.0, 1.0e-12);
    second_moment += quaternion * quaternion.transpose();
  }
  second_moment /= static_cast<double>(samples.size());
  EXPECT_LT((second_moment - 0.25 * Matrix4d::Identity())
                .cwiseAbs()
                .maxCoeff(),
            1.5e-2);
}

TEST(TransformDistributionSampling, FiniteBinghamMatchesPythonSecondMoment) {
  auto input = record();
  auto finite = component("finite", 1.0);
  Matrix4d shape = Matrix4d::Zero();
  shape.diagonal() << -0.75, -0.25, 0.25, 0.75;
  setFiniteOrientation(shape, 0.5, Eigen::Quaterniond::Identity(),
                       &finite.orientation);
  input.components.push_back(finite);

  std::mt19937 generator(23);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 30000, &generator, &samples, &error))
      << error;
  Matrix4d second_moment = Matrix4d::Zero();
  for (const auto& sample : samples) {
    const Vector4d quaternion(sample.rotation.w(), sample.rotation.x(),
                              sample.rotation.y(), sample.rotation.z());
    second_moment += quaternion * quaternion.transpose();
  }
  second_moment /= static_cast<double>(samples.size());
  Matrix4d expected = Matrix4d::Zero();
  expected.diagonal() << 0.14820386, 0.19414743, 0.26740132,
      0.39024738;
  EXPECT_LT((second_moment - expected).cwiseAbs().maxCoeff(), 1.2e-2);
}

TEST(TransformDistributionSampling,
     RotatedFiniteBinghamPreservesItsEigenbasis) {
  auto input = record();
  auto finite = component("finite", 1.0);
  Matrix4d basis;
  basis << 1.0, 1.0, 1.0, 1.0,
      1.0, -1.0, 1.0, -1.0,
      1.0, 1.0, -1.0, -1.0,
      1.0, -1.0, -1.0, 1.0;
  basis *= 0.5;
  Matrix4d shape_eigenvalues = Matrix4d::Zero();
  shape_eigenvalues.diagonal() << -0.75, -0.25, 0.25, 0.75;
  setFiniteOrientation(basis * shape_eigenvalues * basis.transpose(), 0.5,
                       Eigen::Quaterniond::Identity(),
                       &finite.orientation);
  input.components.push_back(finite);

  std::mt19937 generator(25);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 30000, &generator, &samples, &error))
      << error;
  Matrix4d second_moment = Matrix4d::Zero();
  for (const auto& sample : samples) {
    const Vector4d quaternion(sample.rotation.w(), sample.rotation.x(),
                              sample.rotation.y(), sample.rotation.z());
    second_moment += quaternion * quaternion.transpose();
  }
  second_moment /= static_cast<double>(samples.size());
  Matrix4d expected_eigenvalues = Matrix4d::Zero();
  expected_eigenvalues.diagonal() << 0.14820386, 0.19414743, 0.26740132,
      0.39024738;
  const Matrix4d expected =
      basis * expected_eigenvalues * basis.transpose();
  EXPECT_LT((second_moment - expected).cwiseAbs().maxCoeff(), 1.2e-2);
}

TEST(TransformDistributionSampling,
     ExtremelyConcentratedRotatedBinghamRemainsNumericallyBounded) {
  auto input = record();
  auto finite = component("finite", 1.0);
  Matrix4d basis;
  basis << 1.0, 1.0, 1.0, 1.0,
      1.0, -1.0, 1.0, -1.0,
      1.0, 1.0, -1.0, -1.0,
      1.0, -1.0, -1.0, 1.0;
  basis *= 0.5;
  Matrix4d eigenvalues = Matrix4d::Zero();
  eigenvalues.diagonal() << -2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0,
      2.0 / 3.0;
  setFiniteOrientation(basis * eigenvalues * basis.transpose(), 1.0e-100,
                       Eigen::Quaterniond::Identity(),
                       &finite.orientation);
  input.components.push_back(finite);

  std::mt19937 generator(27);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 16, &generator, &samples, &error))
      << error;
  ASSERT_EQ(samples.size(), 16U);
  const Vector4d expected_mode = basis.col(3);
  for (const auto& sample : samples) {
    EXPECT_TRUE(sample.rotation.coeffs().allFinite());
    EXPECT_NEAR(sample.rotation.norm(), 1.0, 1.0e-12);
    const Vector4d quaternion(sample.rotation.w(), sample.rotation.x(),
                              sample.rotation.y(), sample.rotation.z());
    EXPECT_GT(std::abs(quaternion.dot(expected_mode)), 1.0 - 1.0e-12);
  }
}

TEST(TransformDistributionSampling, MixtureWeightsAreScaleSafe) {
  auto input = record();
  const double largest = std::numeric_limits<double>::max();
  input.components.push_back(
      component("left", largest / 3.0, Eigen::Vector3d::Zero()));
  input.components.push_back(
      component("right", largest, Eigen::Vector3d(4.0, 0.0, 0.0)));
  input.components.push_back(
      component("ignored", -largest, Eigen::Vector3d(100.0, 0.0, 0.0)));

  std::mt19937 generator(29);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 20000, &generator, &samples, &error))
      << error;
  std::size_t right_count = 0;
  for (const auto& sample : samples) {
    if (sample.translation.x() == 4.0) {
      ++right_count;
    }
  }
  const double right_fraction =
      static_cast<double>(right_count) / static_cast<double>(samples.size());
  EXPECT_NEAR(right_fraction, 0.75, 1.5e-2);
}

TEST(TransformDistributionSampling,
     ConditionalCouplingUsesColumnMajorRotationVector) {
  auto input = record();
  auto coupled = component("coupled", 1.0);
  setUniformOrientation(Eigen::Quaterniond::Identity(),
                        &coupled.orientation);
  Matrix39d coupling;
  coupling << 0.2, 0.0, 0.1, -0.3, 0.2, 0.0, 0.0, 0.1, -0.2,
      0.0, -0.2, 0.1, 0.2, 0.1, 0.0, -0.1, 0.3, 0.0,
      0.1, 0.0, -0.1, 0.0, 0.2, -0.2, 0.3, 0.0, 0.1;
  const Eigen::Vector3d mean(0.2, -0.1, 0.3);
  const Eigen::Matrix3d covariance =
      (Eigen::Vector3d(0.03, 0.02, 0.01)).asDiagonal();
  setTranslation(mean, covariance, coupling, &coupled.translation);
  input.components.push_back(coupled);

  std::mt19937 generator(31);
  probtf_core::TransformSampleVector samples;
  std::string error;
  ASSERT_TRUE(probtf_core::sampleTransformDistribution(
      input, 30000, &generator, &samples, &error))
      << error;
  const Vector9d reference =
      rotationVector(Eigen::Matrix3d::Identity());
  Eigen::Vector3d residual_mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d residual_second = Eigen::Matrix3d::Zero();
  for (const auto& sample : samples) {
    const Eigen::Vector3d conditional =
        mean + coupling *
                   (rotationVector(sample.rotation.toRotationMatrix()) -
                    reference);
    const Eigen::Vector3d residual = sample.translation - conditional;
    residual_mean += residual;
    residual_second += residual * residual.transpose();
  }
  residual_mean /= static_cast<double>(samples.size());
  const Eigen::Matrix3d residual_covariance =
      residual_second / static_cast<double>(samples.size()) -
      residual_mean * residual_mean.transpose();
  expectVectorNear(residual_mean, Eigen::Vector3d::Zero(), 4.0e-3);
  expectMatrixNear(residual_covariance, covariance, 2.0e-3);
}

TEST(TransformDistributionSampling,
     InvalidWeightDoesNotMutateExistingOutput) {
  auto input = record();
  input.components.push_back(component("finite", 1.0));
  input.components.push_back(
      component("invalid", std::numeric_limits<double>::infinity()));
  std::mt19937 generator(37);
  probtf_core::TransformSampleVector samples(1);
  std::string error;
  EXPECT_FALSE(probtf_core::sampleTransformDistribution(
      input, 4, &generator, &samples, &error));
  EXPECT_NE(error.find("non-finite"), std::string::npos);
  EXPECT_EQ(samples.size(), 1U);
}

TEST(RepresentativeTransform, UsesStoredRepresentativeWhenAvailable) {
  auto input = record();
  input.representative_kind =
      probtf_msgs::ProbabilisticTransformStamped::
          REPRESENTATIVE_PRODUCER_SUPPLIED;
  const Eigen::Vector3d translation(0.3, -0.4, 0.8);
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()));
  assignVector(translation, &input.representative.translation);
  assignQuaternion(rotation, &input.representative.rotation);

  Eigen::Isometry3d representative;
  std::string error;
  ASSERT_TRUE(probtf_core::representativeTransform(
      input, &representative, &error))
      << error;
  expectVectorNear(representative.translation(), translation, 0.0);
  expectMatrixNear(representative.linear(), rotation.toRotationMatrix(),
                   1.0e-15);
}

TEST(RepresentativeTransform,
     FallsBackToHighestWeightComponentModeWithCoupling) {
  auto input = record();
  input.components.push_back(component("lower", 1.0));
  auto selected = component("selected", 3.0);
  Matrix4d shape = Matrix4d::Zero();
  shape.diagonal() << -0.75, -0.25, 0.25, 0.75;
  setFiniteOrientation(shape, 0.5, Eigen::Quaterniond::Identity(),
                       &selected.orientation);
  Matrix39d coupling = Matrix39d::Zero();
  coupling(0, 0) = 0.25;
  coupling(1, 4) = 0.5;
  setTranslation(Eigen::Vector3d(1.0, 2.0, 3.0),
                 Eigen::Matrix3d::Zero(), coupling,
                 &selected.translation);
  input.components.push_back(selected);

  Eigen::Isometry3d representative;
  std::string error;
  ASSERT_TRUE(probtf_core::representativeTransform(
      input, &representative, &error))
      << error;
  Eigen::Matrix3d expected_rotation = Eigen::Matrix3d::Identity();
  expected_rotation(0, 0) = -1.0;
  expected_rotation(1, 1) = -1.0;
  expectMatrixNear(representative.linear(), expected_rotation, 1.0e-12);
  expectVectorNear(representative.translation(),
                   Eigen::Vector3d(0.5, 1.0, 3.0), 1.0e-12);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
