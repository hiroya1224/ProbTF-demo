#pragma once

#include <Eigen/Geometry>

#include <OgreColourValue.h>
#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include <rviz/ogre_helpers/point_cloud.h>

#include <memory>
#include <vector>

namespace Ogre {
class SceneManager;
class SceneNode;
}

namespace rviz {
class Axes;
}

namespace probtf_rviz {

struct ColoredPoint {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Ogre::ColourValue color;
};

struct VisualStyle {
  float point_size = 0.01F;
  float alpha = 0.75F;
  float axis_length = 0.18F;
  float axis_radius = 0.01F;
  bool show_representative = true;
  rviz::PointCloud::RenderMode render_mode = rviz::PointCloud::RM_SPHERES;
};

Ogre::ColourValue axisColor(int axis_index);

class TransformVisual {
 public:
  TransformVisual(Ogre::SceneManager* scene_manager,
                  Ogre::SceneNode* parent_node);
  ~TransformVisual();

  TransformVisual(const TransformVisual&) = delete;
  TransformVisual& operator=(const TransformVisual&) = delete;

  void setFramePose(const Ogre::Vector3& position,
                    const Ogre::Quaternion& orientation);
  void setRepresentative(const Eigen::Isometry3d& transform);
  void setPoints(const std::vector<ColoredPoint>& points);
  void setStyle(const VisualStyle& style);
  void setFade(float fade);
  void setVisible(bool visible);

 private:
  Ogre::SceneManager* scene_manager_;
  Ogre::SceneNode* frame_node_;
  std::unique_ptr<rviz::PointCloud> cloud_;
  std::unique_ptr<rviz::Axes> axes_;
  VisualStyle style_;
  float fade_ = 1.0F;
  bool visible_ = false;
};

}  // namespace probtf_rviz
