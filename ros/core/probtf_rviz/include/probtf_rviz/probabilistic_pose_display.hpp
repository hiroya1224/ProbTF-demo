#pragma once

#include <probtf_rviz/frame_freshness.hpp>

#ifndef Q_MOC_RUN
#include <probtf_msgs/ProbabilisticTransformStamped.h>
#endif

#include <rviz/message_filter_display.h>

#include <memory>

namespace rviz {
class BoolProperty;
class EnumProperty;
class FloatProperty;
class IntProperty;
}

namespace probtf_rviz {

class TransformVisual;

class ProbabilisticPoseDisplay
    : public rviz::MessageFilterDisplay<
          probtf_msgs::ProbabilisticTransformStamped> {
  Q_OBJECT

 public:
  ProbabilisticPoseDisplay();
  ~ProbabilisticPoseDisplay() override;

  void onInitialize() override;
  void reset() override;
  void update(float wall_dt, float ros_dt) override;

 private Q_SLOTS:
  void updateAppearance();
  void updateGeometry();

 private:
  using Message = probtf_msgs::ProbabilisticTransformStamped;

  void processMessage(const Message::ConstPtr& message) override;
  void renderMessage(const Message::ConstPtr& message);
  void refreshFreshness();

  std::unique_ptr<TransformVisual> visual_;
  Message::ConstPtr latest_message_;
  StampFreshness freshness_;
  bool message_renderable_ = false;

  rviz::FloatProperty* frame_timeout_property_;
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
