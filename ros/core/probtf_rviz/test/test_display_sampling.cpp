#include <gtest/gtest.h>
#include <probtf_rviz/display_sampling.hpp>
#include <probtf_core/sampling.hpp>
#include <limits>

namespace {
probtf_msgs::ProbabilisticTransformStamped law(double concentration) {
  probtf_msgs::ProbabilisticTransformStamped record;
  record.header.frame_id = "parent";
  record.child_frame_id = "child";
  record.edge_id = "imu_pair";
  record.components.resize(1);
  auto& component = record.components.front();
  component.component_id = "posterior";
  component.weight = 1.;
  auto& orientation = component.orientation;
  orientation.reference_quaternion.w = 1.;
  if (concentration == 0.) {
    orientation.kind = probtf_msgs::BinghamOrientation::UNIFORM;
    orientation.inverse_concentration = std::numeric_limits<double>::infinity();
  } else {
    orientation.kind = probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
    orientation.inverse_concentration = 1./concentration;
    orientation.shape_upper_wxyz = {{1.5,0.,0.,0.,-.5,0.,0.,-.5,0.,-.5}};
  }
  component.translation.residual_covariance_upper = {{1e-8,0.,0.,1e-8,0.,1e-8}};
  return record;
}

probtf_core::TransformSampleVector draw(probtf_rviz::DisplaySamplingStreams& streams,
    const probtf_msgs::ProbabilisticTransformStamped& record, uint32_t seed=29) {
  probtf_core::TransformSampleVector samples;
  std::string error;
  EXPECT_TRUE(probtf_core::sampleTransformDistribution(record, 120,
      &streams.generator(seed,record.edge_id), &samples, &error)) << error;
  return samples;
}
}  // namespace

TEST(DisplaySampling, EachUpdateGetsNewBacgAndTranslationSamplesAtEveryConcentration) {
  for (double concentration : {0., 100., 1e9}) {
    probtf_rviz::DisplaySamplingStreams streams, replay;
    const auto record = law(concentration);
    const auto first = draw(streams,record), second = draw(streams,record);
    const auto replay_first = draw(replay,record), replay_second = draw(replay,record);
    ASSERT_EQ(120U,first.size());ASSERT_EQ(first.size(),second.size());
    for (std::size_t i=0; i<first.size(); ++i) {
      EXPECT_FALSE(first[i].rotation.coeffs().isApprox(second[i].rotation.coeffs(),0.));
      EXPECT_FALSE(first[i].translation.isApprox(second[i].translation,0.));
      EXPECT_TRUE(first[i].rotation.coeffs().isApprox(replay_first[i].rotation.coeffs(),0.));
      EXPECT_TRUE(second[i].translation.isApprox(replay_second[i].translation,0.));
    }
  }
}

TEST(DisplaySampling, OtherEdgesDoNotRestartOrAdvanceThisEdgesStream) {
  probtf_rviz::DisplaySamplingStreams streams, replay;
  const auto record=law(100.);
  draw(streams,record);draw(replay,record);
  auto other=record;other.edge_id="other_pair";draw(streams,other);
  const auto actual=draw(streams,record), expected=draw(replay,record);
  EXPECT_TRUE(actual[0].translation.isApprox(expected[0].translation,0.));
  streams.reset();
  const auto restarted=draw(streams,record);
  probtf_rviz::DisplaySamplingStreams fresh;
  EXPECT_TRUE(restarted[0].translation.isApprox(draw(fresh,record)[0].translation,0.));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}
