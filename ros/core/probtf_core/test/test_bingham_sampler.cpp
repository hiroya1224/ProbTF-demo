#include <gtest/gtest.h>

#include <probtf_core/bingham_sampler.hpp>

#include <cmath>
#include <limits>

namespace {

using probtf_core::BinghamSampler;
using probtf_core::BinghamSamplingStatistics;
using probtf_core::StandardSamplingRandomSource;

TEST(Bacg, OptimalScaleSolvesPaperEquation36) {
  for (double concentration : {0.0, 1.0, 10.0, 1.0e6, 1.0e300}) {
    Eigen::Vector4d precision(0.0, concentration, concentration, concentration);
    BinghamSampler sampler(precision, Eigen::Matrix4d::Identity());
    const double b = sampler.proposalScale();
    EXPECT_GE(b, 1.0);
    EXPECT_LE(b, 4.0);
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
      sum += 0.5 / (precision(i) + 0.5 * b);
    }
    EXPECT_NEAR(sum, 1.0, 2.0e-15);
  }
}

TEST(Bacg, ReproducesPaperTable2S2Efficiency) {
  // Rounded percentages from Kent et al., Table 2, q=3.
  for (const auto& test : {Eigen::Vector4d(0, 0, 10, .84),
                           Eigen::Vector4d(0, 10, 10, .58),
                           Eigen::Vector4d(0, 0, 100, .80),
                           Eigen::Vector4d(0, 100, 100, .53)}) {
    BinghamSampler sampler(test.head<3>(), Eigen::Matrix3d::Identity());
    std::mt19937_64 engine(20261106);
    StandardSamplingRandomSource<std::mt19937_64> random(engine);
    BinghamSamplingStatistics statistics;
    for (int i = 0; i < 40000; ++i) {
      EXPECT_NEAR(sampler.draw(random, &statistics).norm(), 1.0, 2.0e-14);
    }
    const double efficiency = static_cast<double>(statistics.accepted) /
                              static_cast<double>(statistics.proposals);
    EXPECT_NEAR(efficiency, test(3), .015);
  }
}

TEST(Bacg, UniformAndExtremeConcentrationHaveBoundedCost) {
  for (double concentration : {0.0, 1.0e6, 1.0e12, 1.0e300}) {
    Eigen::Vector4d precision(0.0, concentration, concentration, concentration);
    BinghamSampler sampler(precision, Eigen::Matrix4d::Identity());
    std::mt19937_64 engine(271828);
    StandardSamplingRandomSource<std::mt19937_64> random(engine);
    BinghamSamplingStatistics statistics;
    Eigen::Matrix4d moment = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 20000; ++i) {
      const Eigen::VectorXd sample = sampler.draw(random, &statistics);
      ASSERT_TRUE(sample.allFinite());
      ASSERT_NEAR(sample.norm(), 1.0, 2.0e-14);
      moment.noalias() += sample * sample.transpose();
    }
    moment /= 20000.0;
    if (concentration == 0.0) {
      EXPECT_EQ(statistics.proposals, statistics.accepted);
      EXPECT_LT((moment - .25 * Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff(), .01);
    } else {
      const double efficiency = static_cast<double>(statistics.accepted) /
                                static_cast<double>(statistics.proposals);
      EXPECT_GT(efficiency, .43);
      EXPECT_LT(efficiency, .47);
      EXPECT_GT(moment(0, 0), .999);
      // The minor coordinates still scale with the requested concentration.
      EXPECT_NEAR(concentration * moment(1, 1), .5, .025);
    }
  }
}

TEST(Bacg, SpectralSignAndDegenerateBasisDoNotChangeSeededSamples) {
  const Eigen::Vector4d precision(0, 0, 3, 3);
  Eigen::Matrix4d alternate = Eigen::Matrix4d::Identity();
  const double c = std::cos(.37), s = std::sin(.37);
  alternate(0, 0) = alternate(1, 1) = c;
  alternate(0, 1) = -s;
  alternate(1, 0) = s;
  alternate(2, 2) = -1;
  BinghamSampler first(precision, Eigen::Matrix4d::Identity());
  BinghamSampler second(precision, alternate);
  std::mt19937_64 first_engine(42), second_engine(42);
  StandardSamplingRandomSource<std::mt19937_64> first_random(first_engine);
  StandardSamplingRandomSource<std::mt19937_64> second_random(second_engine);
  for (int i = 0; i < 200; ++i) {
    EXPECT_LT((first.draw(first_random) - second.draw(second_random)).norm(), 1.0e-14);
  }
}

TEST(Bacg, RejectsInvalidParametersRatherThanChangingTheLaw) {
  EXPECT_THROW(BinghamSampler(Eigen::Vector4d(-1, 0, 1, 2), Eigen::Matrix4d::Identity()),
               std::invalid_argument);
  EXPECT_THROW(BinghamSampler(Eigen::Vector4d(0, 1, 2, 3), Eigen::Matrix4d::Zero()),
               std::invalid_argument);
  Eigen::Matrix4d invalid = Eigen::Matrix4d::Zero();
  invalid(0, 1) = 1.0;
  EXPECT_THROW(BinghamSampler::fromParameterMatrix(invalid), std::invalid_argument);
  invalid.setConstant(std::numeric_limits<double>::infinity());
  EXPECT_THROW(BinghamSampler::fromParameterMatrix(invalid), std::invalid_argument);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
