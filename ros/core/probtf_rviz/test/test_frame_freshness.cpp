#include <probtf_rviz/frame_freshness.hpp>

#include <gtest/gtest.h>

namespace {

ros::Time seconds(uint32_t value) {
  return ros::Time(value, 0U);
}

TEST(StampFreshness, FadesDuringFinalTimeoutThird) {
  probtf_rviz::StampFreshness freshness;
  freshness.observe(seconds(7), seconds(100));

  auto state = freshness.state(seconds(105), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 1.0F);

  state = freshness.state(ros::Time(107, 500000000U), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 1.0F);

  state = freshness.state(seconds(110), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 1.0F);

  state = freshness.state(ros::Time(112, 500000000U), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 0.5F);

  state = freshness.state(seconds(115), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 0.0F);

  state = freshness.state(ros::Time(115, 1U), 15.0);
  EXPECT_FALSE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 0.0F);
}

TEST(StampFreshness, RepeatedSnapshotStampDoesNotExtendLifetime) {
  probtf_rviz::StampFreshness freshness;
  freshness.observe(seconds(20), seconds(100));
  freshness.observe(seconds(20), seconds(110));

  EXPECT_EQ(freshness.lastProgress(), seconds(100));
  EXPECT_FALSE(freshness.state(seconds(116), 15.0).visible);
}

TEST(StampFreshness, NewSourceStampRevivesFrame) {
  probtf_rviz::StampFreshness freshness;
  freshness.observe(seconds(20), seconds(100));
  EXPECT_FALSE(freshness.state(seconds(116), 15.0).visible);

  freshness.observe(seconds(21), seconds(116));
  const auto state = freshness.state(seconds(116), 15.0);
  EXPECT_TRUE(state.visible);
  EXPECT_FLOAT_EQ(state.alpha, 1.0F);
  EXPECT_EQ(freshness.lastProgress(), seconds(116));
}

TEST(StampFreshness, RepeatedZeroStampStillExpires) {
  probtf_rviz::StampFreshness freshness;
  freshness.observe(ros::Time(), seconds(100));
  freshness.observe(ros::Time(), seconds(110));

  EXPECT_TRUE(freshness.initialized());
  EXPECT_EQ(freshness.lastProgress(), seconds(100));
  EXPECT_FALSE(freshness.state(seconds(116), 15.0).visible);
}

TEST(StampFreshness, ExplicitProgressRefreshesReceiptBasedLifetime) {
  probtf_rviz::StampFreshness freshness;
  freshness.markProgress(seconds(100));
  freshness.markProgress(seconds(110));

  EXPECT_EQ(freshness.lastProgress(), seconds(110));
  EXPECT_TRUE(freshness.state(seconds(125), 15.0).visible);
  EXPECT_FALSE(freshness.state(ros::Time(125, 1U), 15.0).visible);
}

TEST(StampFreshness, RosClockRewindRebasesLifetime) {
  probtf_rviz::StampFreshness freshness;
  freshness.observe(seconds(20), seconds(100));
  freshness.state(seconds(110), 15.0);

  EXPECT_TRUE(freshness.state(seconds(50), 15.0).visible);
  EXPECT_EQ(freshness.lastProgress(), seconds(50));
  EXPECT_TRUE(freshness.state(seconds(65), 15.0).visible);
  EXPECT_FALSE(freshness.state(ros::Time(65, 1U), 15.0).visible);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
