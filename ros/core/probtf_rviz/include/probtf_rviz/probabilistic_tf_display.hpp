#pragma once

#include <rviz/display.h>

#include <memory>

namespace rviz {
class BoolProperty;
class EnumProperty;
class FloatProperty;
class IntProperty;
class RosTopicProperty;
class StringProperty;
}

namespace probtf_rviz {

class ProbabilisticTfDisplay : public rviz::Display {
  Q_OBJECT

 public:
  ProbabilisticTfDisplay();
  ~ProbabilisticTfDisplay() override;

  void onInitialize() override;
  void reset() override;
  void update(float wall_dt, float ros_dt) override;

 protected:
  void onEnable() override;
  void onDisable() override;
  void fixedFrameChanged() override;

 private Q_SLOTS:
  void updateTopics();
  void updateAppearance();
  void updateGeometry();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  rviz::RosTopicProperty* dynamic_topic_property_;
  rviz::RosTopicProperty* dynamic_batch_topic_property_;
  rviz::RosTopicProperty* static_topic_property_;
  rviz::IntProperty* queue_size_property_;
  rviz::FloatProperty* frame_timeout_property_;
  rviz::StringProperty* root_frame_property_;
  rviz::IntProperty* sample_count_property_;
  rviz::FloatProperty* axis_length_property_;
  rviz::FloatProperty* point_size_property_;
  rviz::EnumProperty* point_style_property_;
  rviz::FloatProperty* alpha_property_;
  rviz::BoolProperty* show_representative_property_;
  rviz::FloatProperty* representative_radius_property_;
  rviz::IntProperty* seed_property_;
};

}  // namespace probtf_rviz
