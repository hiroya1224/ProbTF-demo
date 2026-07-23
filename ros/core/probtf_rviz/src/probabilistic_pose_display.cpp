#include <probtf_rviz/probabilistic_pose_display.hpp>

#include <probtf_rviz/transform_visual.hpp>

#include <probtf_core/sampling.hpp>

#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include <rviz/display_context.h>
#include <rviz/frame_manager.h>
#include <rviz/properties/bool_property.h>
#include <rviz/properties/enum_property.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/properties/status_property.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace probtf_rviz {
namespace {

rviz::PointCloud::RenderMode renderMode(rviz::EnumProperty* property) {
  switch (property->getOptionInt()) {
    case 0:
      return rviz::PointCloud::RM_POINTS;
    case 1:
      return rviz::PointCloud::RM_SQUARES;
    default:
      return rviz::PointCloud::RM_SPHERES;
  }
}

}  // namespace

ProbabilisticPoseDisplay::ProbabilisticPoseDisplay()
    : sample_count_property_(new rviz::IntProperty(
          "Sample Count", 300,
          "Number of correlated transforms sampled from each message.", this,
          SLOT(updateGeometry()))),
      axis_length_property_(new rviz::FloatProperty(
          "Axis Length", 0.18,
          "Distance from the sampled origin to each colored axis endpoint.",
          this, SLOT(updateGeometry()))),
      point_size_property_(new rviz::FloatProperty(
          "Point Size", 0.01, "Rendered diameter of each endpoint.", this,
          SLOT(updateAppearance()))),
      point_style_property_(new rviz::EnumProperty(
          "Point Style", "Spheres", "Rendering primitive for sampled points.",
          this, SLOT(updateAppearance()))),
      alpha_property_(new rviz::FloatProperty(
          "Alpha", 0.75, "Opacity of the cloud and representative axes.", this,
          SLOT(updateAppearance()))),
      show_representative_property_(new rviz::BoolProperty(
          "Show Representative", true,
          "Draw the supplied representative, or the highest-weight component "
          "mode when no representative was supplied.",
          this, SLOT(updateAppearance()))),
      representative_radius_property_(new rviz::FloatProperty(
          "Representative Radius", 0.01,
          "Radius of the representative coordinate axes.", this,
          SLOT(updateAppearance()))),
      seed_property_(new rviz::IntProperty(
          "Random Seed", 7,
          "Stable seed used to resample this distribution.", this,
          SLOT(updateGeometry()))) {
  sample_count_property_->setMin(1);
  sample_count_property_->setMax(100000);
  axis_length_property_->setMin(0.0001);
  point_size_property_->setMin(0.0001);
  alpha_property_->setMin(0.0);
  alpha_property_->setMax(1.0);
  representative_radius_property_->setMin(0.0001);
  seed_property_->setMin(0);

  point_style_property_->addOption("Points", 0);
  point_style_property_->addOption("Squares", 1);
  point_style_property_->addOption("Spheres", 2);
}

ProbabilisticPoseDisplay::~ProbabilisticPoseDisplay() = default;

void ProbabilisticPoseDisplay::onInitialize() {
  MFDClass::onInitialize();
  visual_.reset(
      new TransformVisual(context_->getSceneManager(), scene_node_));
  updateAppearance();
}

void ProbabilisticPoseDisplay::reset() {
  MFDClass::reset();
  latest_message_.reset();
  if (visual_) {
    visual_->setPoints({});
    visual_->setVisible(false);
  }
}

void ProbabilisticPoseDisplay::updateAppearance() {
  if (!visual_) {
    return;
  }
  VisualStyle style;
  style.point_size = point_size_property_->getFloat();
  style.alpha = alpha_property_->getFloat();
  style.axis_length = axis_length_property_->getFloat();
  style.axis_radius = representative_radius_property_->getFloat();
  style.show_representative = show_representative_property_->getBool();
  style.render_mode = renderMode(point_style_property_);
  visual_->setStyle(style);
  context_->queueRender();
}

void ProbabilisticPoseDisplay::updateGeometry() {
  updateAppearance();
  if (latest_message_) {
    renderMessage(latest_message_);
  }
}

void ProbabilisticPoseDisplay::processMessage(
    const Message::ConstPtr& message) {
  latest_message_ = message;
  renderMessage(message);
}

void ProbabilisticPoseDisplay::renderMessage(
    const Message::ConstPtr& message) {
  if (!visual_) {
    return;
  }
  if (message->header.frame_id.empty()) {
    visual_->setVisible(false);
    setStatus(rviz::StatusProperty::Error, "Message",
              "header.frame_id must not be empty.");
    return;
  }

  Ogre::Vector3 frame_position;
  Ogre::Quaternion frame_orientation;
  if (!context_->getFrameManager()->getTransform(
          message->header, frame_position, frame_orientation)) {
    visual_->setVisible(false);
    setStatus(rviz::StatusProperty::Error, "Transform",
              QString("Could not transform '%1' into the RViz fixed frame.")
                  .arg(QString::fromStdString(message->header.frame_id)));
    return;
  }

  std::mt19937 generator(
      static_cast<std::mt19937::result_type>(seed_property_->getInt()));
  probtf_core::TransformSampleVector samples;
  std::string error;
  if (!probtf_core::sampleTransformDistribution(
          *message, static_cast<std::size_t>(sample_count_property_->getInt()),
          &generator, &samples, &error)) {
    visual_->setVisible(false);
    setStatus(rviz::StatusProperty::Error, "Distribution",
              QString::fromStdString(error));
    return;
  }

  const double axis_length = axis_length_property_->getFloat();
  std::vector<ColoredPoint> points;
  points.reserve(3U * samples.size());
  for (const auto& sample : samples) {
    for (int axis = 0; axis < 3; ++axis) {
      ColoredPoint point;
      point.position =
          sample.translation +
          sample.rotation * (axis_length * Eigen::Vector3d::Unit(axis));
      point.color = axisColor(axis);
      points.push_back(point);
    }
  }

  Eigen::Isometry3d representative = Eigen::Isometry3d::Identity();
  if (!probtf_core::representativeTransform(*message, &representative,
                                            &error)) {
    visual_->setVisible(false);
    setStatus(rviz::StatusProperty::Error, "Representative",
              QString::fromStdString(error));
    return;
  }

  visual_->setFramePose(frame_position, frame_orientation);
  visual_->setPoints(points);
  visual_->setRepresentative(representative);
  visual_->setVisible(true);
  setStatus(rviz::StatusProperty::Ok, "Distribution",
            QString("%1 correlated samples").arg(samples.size()));
  deleteStatus("Message");
  deleteStatus("Transform");
  deleteStatus("Representative");
  context_->queueRender();
}

}  // namespace probtf_rviz
