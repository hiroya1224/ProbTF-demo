#include <probtf_rviz/transform_visual.hpp>

#include <OgreSceneManager.h>
#include <OgreSceneNode.h>

#include <rviz/ogre_helpers/axes.h>

#include <algorithm>
#include <cmath>

namespace probtf_rviz {

Ogre::ColourValue axisColor(int axis_index) {
  switch (axis_index) {
    case 0:
      return Ogre::ColourValue(1.0F, 0.18F, 0.18F, 1.0F);
    case 1:
      return Ogre::ColourValue(0.18F, 1.0F, 0.18F, 1.0F);
    default:
      return Ogre::ColourValue(0.18F, 0.32F, 1.0F, 1.0F);
  }
}

TransformVisual::TransformVisual(Ogre::SceneManager* scene_manager,
                                 Ogre::SceneNode* parent_node)
    : scene_manager_(scene_manager),
      frame_node_(parent_node->createChildSceneNode()),
      cloud_(new rviz::PointCloud()),
      axes_(new rviz::Axes(scene_manager_, frame_node_)) {
  frame_node_->attachObject(cloud_.get());
  setStyle(style_);
}

TransformVisual::~TransformVisual() {
  axes_.reset();
  frame_node_->detachObject(cloud_.get());
  cloud_.reset();
  scene_manager_->destroySceneNode(frame_node_);
}

void TransformVisual::setFramePose(const Ogre::Vector3& position,
                                   const Ogre::Quaternion& orientation) {
  frame_node_->setPosition(position);
  frame_node_->setOrientation(orientation);
}

void TransformVisual::setRepresentative(const Eigen::Isometry3d& transform) {
  const Eigen::Vector3d& translation = transform.translation();
  const Eigen::Quaterniond rotation(transform.rotation());
  axes_->setPosition(Ogre::Vector3(
      static_cast<float>(translation.x()), static_cast<float>(translation.y()),
      static_cast<float>(translation.z())));
  axes_->setOrientation(Ogre::Quaternion(
      static_cast<float>(rotation.w()), static_cast<float>(rotation.x()),
      static_cast<float>(rotation.y()), static_cast<float>(rotation.z())));
}

void TransformVisual::setPoints(const std::vector<ColoredPoint>& points) {
  std::vector<rviz::PointCloud::Point> output(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3d& position = points[index].position;
    output[index].position =
        Ogre::Vector3(static_cast<float>(position.x()),
                      static_cast<float>(position.y()),
                      static_cast<float>(position.z()));
    output[index].color = points[index].color;
  }
  cloud_->clear();
  if (!output.empty()) {
    cloud_->addPoints(output.data(), static_cast<uint32_t>(output.size()));
  }
}

void TransformVisual::setStyle(const VisualStyle& style) {
  style_ = style;
  cloud_->setRenderMode(style_.render_mode);
  cloud_->setDimensions(style_.point_size, style_.point_size,
                        style_.point_size);
  const float alpha = style_.alpha * fade_;
  cloud_->setAlpha(alpha, true);
  axes_->set(style_.axis_length, style_.axis_radius, alpha);
  axes_->setToDefaultColors();
  cloud_->setVisible(visible_);
  axes_->getSceneNode()->setVisible(
      visible_ && style_.show_representative);
}

void TransformVisual::setFade(float fade) {
  const float clamped = std::max(0.0F, std::min(1.0F, fade));
  if (std::abs(clamped - fade_) <= 1.0e-6F) {
    return;
  }
  fade_ = clamped;
  const float alpha = style_.alpha * fade_;
  cloud_->setAlpha(alpha, true);
  axes_->updateAlpha(alpha);
  axes_->setToDefaultColors();
}

void TransformVisual::setVisible(bool visible) {
  visible_ = visible;
  cloud_->setVisible(visible_);
  axes_->getSceneNode()->setVisible(
      visible_ && style_.show_representative);
}

}  // namespace probtf_rviz
