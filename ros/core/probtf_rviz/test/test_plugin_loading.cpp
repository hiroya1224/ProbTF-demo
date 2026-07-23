#include <gtest/gtest.h>

#include <pluginlib/class_loader.h>
#include <rviz/display.h>

#include <algorithm>
#include <string>
#include <vector>

TEST(ProbTfRvizPlugin, DeclaresBothDisplays) {
  pluginlib::ClassLoader<rviz::Display> loader("rviz", "rviz::Display");
  const std::vector<std::string> classes = loader.getDeclaredClasses();
  EXPECT_NE(std::find(classes.begin(), classes.end(),
                      "probtf_rviz/ProbabilisticPose"),
            classes.end());
  EXPECT_NE(std::find(classes.begin(), classes.end(),
                      "probtf_rviz/ProbabilisticTF"),
            classes.end());
  EXPECT_EQ(loader.getClassType("probtf_rviz/ProbabilisticPose"),
            "probtf_rviz::ProbabilisticPoseDisplay");
  EXPECT_EQ(loader.getClassType("probtf_rviz/ProbabilisticTF"),
            "probtf_rviz::ProbabilisticTfDisplay");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
