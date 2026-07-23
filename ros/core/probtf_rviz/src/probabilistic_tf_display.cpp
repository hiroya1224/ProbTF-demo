#include <probtf_rviz/probabilistic_tf_display.hpp>

#include <probtf_rviz/frame_freshness.hpp>
#include <probtf_rviz/transform_visual.hpp>

#include <probtf_core/latest_snapshot.hpp>
#include <probtf_core/sampling.hpp>
#include <probtf_msgs/ProbabilisticTransformArray.h>
#include <probtf_msgs/ProbabilisticTransformStamped.h>

#include <Eigen/Eigenvalues>

#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include <rviz/display_context.h>
#include <rviz/frame_manager.h>
#include <rviz/properties/bool_property.h>
#include <rviz/properties/enum_property.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/properties/ros_topic_property.h>
#include <rviz/properties/status_property.h>
#include <rviz/properties/string_property.h>

#include <boost/thread/lock_guard.hpp>
#include <boost/thread/mutex.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace probtf_rviz {
namespace {

using Array = probtf_msgs::ProbabilisticTransformArray;
using Record = probtf_msgs::ProbabilisticTransformStamped;

constexpr std::size_t kMaxTreePointCount = 1500000U;

std::string cleanFrame(const std::string& value) {
  const std::size_t first = value.find_first_not_of("/ \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const std::size_t last = value.find_last_not_of("/ \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::string recordKey(const Record& record) {
  if (!record.edge_id.empty()) {
    return record.edge_id;
  }
  return cleanFrame(record.header.frame_id) + "->" +
         cleanFrame(record.child_frame_id);
}

QString frameStatusKey(const std::string& child) {
  return "Frame/" + QString::fromStdString(child);
}

uint64_t stableHash(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char character : value) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  return hash;
}

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

bool sampleGaussian(const Eigen::Vector3d& mean,
                    const Eigen::Matrix3d& covariance, std::size_t count,
                    std::mt19937* generator,
                    std::vector<Eigen::Vector3d>* output,
                    std::string* error) {
  if (!mean.allFinite() || !covariance.allFinite()) {
    if (error != nullptr) {
      *error = "Point moments contain non-finite values.";
    }
    return false;
  }
  const Eigen::Matrix3d symmetric =
      0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    if (error != nullptr) {
      *error = "Point covariance eigendecomposition failed.";
    }
    return false;
  }
  const double scale =
      std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
  if (solver.eigenvalues().minCoeff() < -1.0e-9 * scale) {
    if (error != nullptr) {
      *error = "Point covariance is not positive semidefinite.";
    }
    return false;
  }
  const Eigen::Vector3d standard_deviation =
      solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const Eigen::Matrix3d square_root =
      solver.eigenvectors() * standard_deviation.asDiagonal();

  std::normal_distribution<double> normal(0.0, 1.0);
  output->clear();
  output->reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const Eigen::Vector3d standard(normal(*generator), normal(*generator),
                                   normal(*generator));
    output->push_back(mean + square_root * standard);
  }
  return true;
}

}  // namespace

class ProbabilisticTfDisplay::Impl {
 public:
  explicit Impl(ProbabilisticTfDisplay* display) : display_(display) {
  }

  void subscribe(ros::NodeHandle& node_handle, const std::string& dynamic_topic,
                 const std::string& dynamic_batch_topic,
                 const std::string& static_topic, uint32_t queue_size) {
    unsubscribe();
    if (!dynamic_topic.empty()) {
      dynamic_subscriber_ = node_handle.subscribe<Record>(
          dynamic_topic, queue_size, &Impl::handleDynamic, this);
    }
    if (!dynamic_batch_topic.empty()) {
      dynamic_batch_subscriber_ = node_handle.subscribe<Array>(
          dynamic_batch_topic, queue_size, &Impl::handleDynamicBatch, this);
    }
    if (!static_topic.empty()) {
      static_subscriber_ = node_handle.subscribe<Array>(
          static_topic, 1U, &Impl::handleStatic, this);
    }
  }

  void unsubscribe() {
    dynamic_subscriber_.shutdown();
    dynamic_batch_subscriber_.shutdown();
    static_subscriber_.shutdown();
  }

  void clear() {
    clearFrameStatuses();
    display_->deleteStatus("Sampling");
    boost::lock_guard<boost::mutex> lock(mutex_);
    pending_dynamic_.clear();
    pending_dynamic_batch_.reset();
    pending_static_.reset();
    dynamic_records_.clear();
    static_records_.clear();
    visuals_.clear();
    frame_bindings_.clear();
    geometry_errors_.clear();
    frame_refresh_enabled_ = false;
    callback_sequence_ = 0;
    batch_sequence_ = 0;
    static_sequence_ = 0;
    dirty_ = true;
  }

  void markDirty() {
    boost::lock_guard<boost::mutex> lock(mutex_);
    dirty_ = true;
  }

  void updateStyles(const VisualStyle& style) {
    for (auto& keyed : visuals_) {
      keyed.second->setStyle(style);
    }
  }

  void update() {
    consumePending();
    bool should_render = false;
    {
      boost::lock_guard<boost::mutex> lock(mutex_);
      should_render = dirty_;
      dirty_ = false;
    }
    if (should_render) {
      render();
    }
    refreshFramePoses();
  }

 private:
  struct PendingRecord {
    uint64_t sequence = 0;
    Record::ConstPtr message;
  };

  struct PendingArray {
    uint64_t sequence = 0;
    Array::ConstPtr message;
  };

  struct FrameBinding {
    std::string root;
    std::string path_signature;
    ros::Time stamp;
    bool has_dynamic = false;
    StampFreshness freshness;
  };

  void handleDynamic(const Record::ConstPtr& message) {
    if (!message) {
      return;
    }
    boost::lock_guard<boost::mutex> lock(mutex_);
    PendingRecord pending;
    pending.sequence = ++callback_sequence_;
    pending.message = message;
    pending_dynamic_[recordKey(*message)] = pending;
  }

  void handleDynamicBatch(const Array::ConstPtr& message) {
    if (!message) {
      return;
    }
    boost::lock_guard<boost::mutex> lock(mutex_);
    pending_dynamic_batch_.reset(new PendingArray());
    pending_dynamic_batch_->sequence = ++callback_sequence_;
    pending_dynamic_batch_->message = message;
  }

  void handleStatic(const Array::ConstPtr& message) {
    if (!message) {
      return;
    }
    boost::lock_guard<boost::mutex> lock(mutex_);
    pending_static_.reset(new PendingArray());
    pending_static_->sequence = ++callback_sequence_;
    pending_static_->message = message;
  }

  void consumePending() {
    std::map<std::string, PendingRecord> dynamic;
    std::unique_ptr<PendingArray> dynamic_batch;
    std::unique_ptr<PendingArray> static_batch;
    {
      boost::lock_guard<boost::mutex> lock(mutex_);
      dynamic.swap(pending_dynamic_);
      dynamic_batch.swap(pending_dynamic_batch_);
      static_batch.swap(pending_static_);
    }

    bool changed = false;
    if (dynamic_batch) {
      dynamic_records_.clear();
      for (const Record& record : dynamic_batch->message->transforms) {
        dynamic_records_[recordKey(record)] = record;
      }
      batch_sequence_ = dynamic_batch->sequence;
      changed = true;
    }
    for (const auto& keyed : dynamic) {
      if (keyed.second.sequence >= batch_sequence_) {
        dynamic_records_[keyed.first] = *keyed.second.message;
        changed = true;
      }
    }
    if (static_batch && static_batch->sequence >= static_sequence_) {
      static_records_.clear();
      for (const Record& record : static_batch->message->transforms) {
        static_records_[recordKey(record)] = record;
      }
      static_sequence_ = static_batch->sequence;
      changed = true;
    }
    if (changed) {
      boost::lock_guard<boost::mutex> lock(mutex_);
      dirty_ = true;
    }
  }

  bool buildRootPath(
      const std::string& child,
      const std::unordered_map<std::string, const Record*>& by_child,
      const std::string& requested_root, std::string* root,
      std::vector<const Record*>* child_to_root, std::string* error) const {
    child_to_root->clear();
    std::unordered_set<std::string> visited;
    std::string current = child;
    while (true) {
      if (!visited.insert(current).second) {
        if (error != nullptr) {
          *error = "ProbTF topology contains a cycle at frame '" + current +
                   "'.";
        }
        return false;
      }
      if (!requested_root.empty() && current == requested_root) {
        *root = current;
        return true;
      }
      const auto edge = by_child.find(current);
      if (edge == by_child.end()) {
        if (!requested_root.empty()) {
          if (error != nullptr) {
            *error = "Requested root '" + requested_root +
                     "' is not an ancestor of frame '" + child + "'.";
          }
          return false;
        }
        *root = current;
        return true;
      }
      child_to_root->push_back(edge->second);
      current = cleanFrame(edge->second->header.frame_id);
    }
  }

  bool composedRepresentative(
      const std::vector<const Record*>& child_to_root,
      Eigen::Isometry3d* output, std::string* error) const {
    *output = Eigen::Isometry3d::Identity();
    for (auto iterator = child_to_root.rbegin();
         iterator != child_to_root.rend(); ++iterator) {
      Eigen::Isometry3d edge = Eigen::Isometry3d::Identity();
      if (!probtf_core::representativeTransform(**iterator, &edge, error)) {
        return false;
      }
      *output = *output * edge;
    }
    return true;
  }

  VisualStyle style() const {
    VisualStyle output;
    output.point_size = display_->point_size_property_->getFloat();
    output.alpha = display_->alpha_property_->getFloat();
    output.axis_length = display_->axis_length_property_->getFloat();
    output.axis_radius =
        display_->representative_radius_property_->getFloat();
    output.show_representative =
        display_->show_representative_property_->getBool();
    output.render_mode = renderMode(display_->point_style_property_);
    return output;
  }

  void clearFrameStatuses() {
    for (const std::string& child : status_children_) {
      display_->deleteStatus(frameStatusKey(child));
    }
    status_children_.clear();
  }

  void refreshFramePoses() {
    if (!frame_refresh_enabled_) {
      return;
    }

    std::set<std::string> failed_status_children;
    std::size_t rendered_count = 0U;
    std::size_t expired_count = 0U;
    std::size_t failed_count = geometry_errors_.size();
    const ros::Time now = ros::Time::now();
    const double timeout = display_->frame_timeout_property_->getFloat();
    for (const auto& keyed : geometry_errors_) {
      display_->setStatus(rviz::StatusProperty::Error,
                          frameStatusKey(keyed.first),
                          QString::fromStdString(keyed.second));
      failed_status_children.insert(keyed.first);
    }

    for (auto& keyed : visuals_) {
      const auto binding = frame_bindings_.find(keyed.first);
      if (binding == frame_bindings_.end()) {
        keyed.second->setVisible(false);
        display_->setStatus(rviz::StatusProperty::Error,
                            frameStatusKey(keyed.first),
                            "Internal frame binding is missing.");
        failed_status_children.insert(keyed.first);
        ++failed_count;
        continue;
      }

      FreshnessVisualState freshness;
      if (binding->second.has_dynamic) {
        freshness = binding->second.freshness.state(now, timeout);
      }
      keyed.second->setFade(freshness.alpha);
      if (!freshness.visible) {
        keyed.second->setVisible(false);
        display_->deleteStatus(frameStatusKey(keyed.first));
        ++expired_count;
        continue;
      }

      Ogre::Vector3 frame_position;
      Ogre::Quaternion frame_orientation;
      if (!display_->context_->getFrameManager()->getTransform(
              binding->second.root, binding->second.stamp, frame_position,
              frame_orientation)) {
        keyed.second->setVisible(false);
        display_->setStatus(
            rviz::StatusProperty::Error, frameStatusKey(keyed.first),
            QString("Could not transform root '%1' into the RViz fixed frame.")
                .arg(QString::fromStdString(binding->second.root)));
        failed_status_children.insert(keyed.first);
        ++failed_count;
        continue;
      }

      keyed.second->setFramePose(frame_position, frame_orientation);
      keyed.second->setVisible(true);
      display_->deleteStatus(frameStatusKey(keyed.first));
      ++rendered_count;
    }

    for (const std::string& child : status_children_) {
      if (failed_status_children.count(child) == 0U) {
        display_->deleteStatus(frameStatusKey(child));
      }
    }
    status_children_ = std::move(failed_status_children);

    if (rendered_count == 0U && failed_count > 0U) {
      display_->setStatus(rviz::StatusProperty::Error, "ProbTF",
                          QString("No fresh frame rendered (%1 expired, %2 "
                                  "failed).")
                              .arg(expired_count)
                              .arg(failed_count));
    } else if (rendered_count == 0U && expired_count > 0U) {
      display_->setStatus(
          rviz::StatusProperty::Warn, "ProbTF",
          QString("No fresh frame rendered (%1 expired).").arg(expired_count));
    } else {
      display_->setStatus(
          failed_count == 0U && expired_count == 0U
              ? rviz::StatusProperty::Ok
              : rviz::StatusProperty::Warn,
          "ProbTF",
          QString("%1 frames rendered, %2 expired, %3 failed.")
              .arg(rendered_count)
              .arg(expired_count)
              .arg(failed_count));
    }
    display_->context_->queueRender();
  }

  void render() {
    Array dynamic;
    Array static_set;
    for (const auto& keyed : dynamic_records_) {
      dynamic.transforms.push_back(keyed.second);
    }
    for (const auto& keyed : static_records_) {
      static_set.transforms.push_back(keyed.second);
    }
    if (dynamic.transforms.empty() && static_set.transforms.empty()) {
      visuals_.clear();
      frame_bindings_.clear();
      geometry_errors_.clear();
      frame_refresh_enabled_ = false;
      clearFrameStatuses();
      display_->deleteStatus("Sampling");
      display_->setStatus(rviz::StatusProperty::Warn, "ProbTF",
                          "Waiting for ProbTF records.");
      return;
    }

    probtf_core::LatestSnapshot snapshot(dynamic, static_set);
    std::string error;
    if (!snapshot.valid(&error)) {
      visuals_.clear();
      frame_bindings_.clear();
      geometry_errors_.clear();
      frame_refresh_enabled_ = false;
      clearFrameStatuses();
      display_->deleteStatus("Sampling");
      display_->setStatus(rviz::StatusProperty::Error, "ProbTF",
                          QString::fromStdString(error));
      return;
    }

    std::unordered_map<std::string, const Record*> by_child;
    std::vector<const Record*> records;
    records.reserve(dynamic.transforms.size() + static_set.transforms.size());
    for (const Record& record : static_set.transforms) {
      const std::string child = cleanFrame(record.child_frame_id);
      by_child[child] = &record;
      records.push_back(&record);
    }
    for (const Record& record : dynamic.transforms) {
      const std::string child = cleanFrame(record.child_frame_id);
      by_child[child] = &record;
      records.push_back(&record);
    }

    frame_refresh_enabled_ = true;
    geometry_errors_.clear();
    const std::string requested_root =
        cleanFrame(display_->root_frame_property_->getStdString());
    const double axis_length = display_->axis_length_property_->getFloat();
    const std::size_t requested_sample_count =
        static_cast<std::size_t>(display_->sample_count_property_->getInt());
    const std::size_t frame_count = std::max<std::size_t>(1U, by_child.size());
    const std::size_t sample_limit =
        std::max<std::size_t>(1U, kMaxTreePointCount / (3U * frame_count));
    const std::size_t sample_count =
        std::min(requested_sample_count, sample_limit);
    if (sample_count < requested_sample_count) {
      display_->setStatus(
          rviz::StatusProperty::Warn, "Sampling",
          QString("Sample Count limited to %1 per frame (%2 total-point cap).")
              .arg(sample_count)
              .arg(kMaxTreePointCount));
    } else {
      display_->deleteStatus("Sampling");
    }
    const uint32_t base_seed =
        static_cast<uint32_t>(display_->seed_property_->getInt());
    std::set<std::string> seen_children;
    std::set<std::string> successful_children;

    for (const Record* record : records) {
      const std::string child = cleanFrame(record->child_frame_id);
      if (!seen_children.insert(child).second) {
        continue;
      }

      std::string root;
      std::vector<const Record*> path;
      if (!buildRootPath(child, by_child, requested_root, &root, &path,
                         &error)) {
        geometry_errors_[child] = error;
        continue;
      }

      std::vector<ColoredPoint> points;
      points.reserve(3U * sample_count);
      ros::Time resolved_stamp;
      bool endpoint_failed = false;
      for (int axis = 0; axis < 3; ++axis) {
        probtf_core::PointMomentObservation observation;
        if (!snapshot.lookupPointMoments(
                root, child, axis_length * Eigen::Vector3d::Unit(axis),
                &observation, &error)) {
          endpoint_failed = true;
          break;
        }
        if (axis == 0) {
          resolved_stamp = observation.resolved_stamp;
        }
        const uint64_t hash = stableHash(child);
        std::seed_seq seeds{
            base_seed, static_cast<uint32_t>(hash),
            static_cast<uint32_t>(hash >> 32U), static_cast<uint32_t>(axis)};
        std::mt19937 generator(seeds);
        std::vector<Eigen::Vector3d> endpoint_samples;
        if (!sampleGaussian(observation.moments.mean,
                            observation.moments.covariance, sample_count,
                            &generator, &endpoint_samples, &error)) {
          endpoint_failed = true;
          break;
        }
        for (const Eigen::Vector3d& endpoint : endpoint_samples) {
          ColoredPoint point;
          point.position = endpoint;
          point.color = axisColor(axis);
          points.push_back(point);
        }
      }
      if (endpoint_failed) {
        geometry_errors_[child] = error;
        continue;
      }

      Eigen::Isometry3d representative = Eigen::Isometry3d::Identity();
      if (!composedRepresentative(path, &representative, &error)) {
        geometry_errors_[child] = error;
        continue;
      }

      auto visual = visuals_.find(child);
      if (visual == visuals_.end()) {
        visual =
            visuals_
                .emplace(
                    child,
                    std::unique_ptr<TransformVisual>(new TransformVisual(
                        display_->context_->getSceneManager(),
                        display_->scene_node_)))
                .first;
      }
      visual->second->setStyle(style());
      visual->second->setPoints(points);
      visual->second->setRepresentative(representative);
      FrameBinding binding;
      binding.root = root;
      binding.stamp = resolved_stamp;
      std::ostringstream path_signature;
      for (const Record* edge : path) {
        binding.has_dynamic = binding.has_dynamic || !edge->is_static;
        path_signature << recordKey(*edge) << '\n';
      }
      binding.path_signature = path_signature.str();
      const auto previous_binding = frame_bindings_.find(child);
      if (binding.has_dynamic && previous_binding != frame_bindings_.end() &&
          previous_binding->second.has_dynamic &&
          previous_binding->second.root == binding.root &&
          previous_binding->second.path_signature == binding.path_signature) {
        binding.freshness = previous_binding->second.freshness;
      }
      if (binding.has_dynamic) {
        binding.freshness.observe(resolved_stamp, ros::Time::now());
      }
      frame_bindings_[child] = binding;
      successful_children.insert(child);
    }

    for (auto iterator = visuals_.begin(); iterator != visuals_.end();) {
      if (successful_children.count(iterator->first) == 0U) {
        iterator = visuals_.erase(iterator);
      } else {
        ++iterator;
      }
    }
    for (auto iterator = frame_bindings_.begin();
         iterator != frame_bindings_.end();) {
      if (successful_children.count(iterator->first) == 0U) {
        iterator = frame_bindings_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  ProbabilisticTfDisplay* display_;
  ros::Subscriber dynamic_subscriber_;
  ros::Subscriber dynamic_batch_subscriber_;
  ros::Subscriber static_subscriber_;

  boost::mutex mutex_;
  uint64_t callback_sequence_ = 0;
  uint64_t batch_sequence_ = 0;
  uint64_t static_sequence_ = 0;
  std::map<std::string, PendingRecord> pending_dynamic_;
  std::unique_ptr<PendingArray> pending_dynamic_batch_;
  std::unique_ptr<PendingArray> pending_static_;
  bool dirty_ = true;

  std::map<std::string, Record> dynamic_records_;
  std::map<std::string, Record> static_records_;
  std::map<std::string, std::unique_ptr<TransformVisual>> visuals_;
  std::map<std::string, FrameBinding> frame_bindings_;
  std::map<std::string, std::string> geometry_errors_;
  std::set<std::string> status_children_;
  bool frame_refresh_enabled_ = false;
};

ProbabilisticTfDisplay::ProbabilisticTfDisplay()
    : impl_(new Impl(this)),
      dynamic_topic_property_(new rviz::RosTopicProperty(
          "Dynamic Topic", "/probtf",
          QString::fromStdString(ros::message_traits::datatype<Record>()),
          "Incremental ProbabilisticTransformStamped edge stream.", this,
          SLOT(updateTopics()))),
      dynamic_batch_topic_property_(new rviz::RosTopicProperty(
          "Dynamic Snapshot Topic", "/probtf_batch",
          QString::fromStdString(ros::message_traits::datatype<Array>()),
          "Optional complete latest-only dynamic snapshot.", this,
          SLOT(updateTopics()))),
      static_topic_property_(new rviz::RosTopicProperty(
          "Static Topic", "/probtf_static",
          QString::fromStdString(ros::message_traits::datatype<Array>()),
          "Complete latched static ProbTF set.", this, SLOT(updateTopics()))),
      queue_size_property_(new rviz::IntProperty(
          "Queue Size", 10, "ROS subscription queue size.", this,
          SLOT(updateTopics()))),
      frame_timeout_property_(new rviz::FloatProperty(
          "Frame Timeout", 15.0,
          "Seconds without a new dynamic source stamp before clouds and "
          "representatives disappear. Static records do not expire.",
          this, SLOT(updateAppearance()))),
      root_frame_property_(new rviz::StringProperty(
          "Root Frame", "",
          "Optional ProbTF root. Empty selects the root of each connected "
          "tree automatically.",
          this, SLOT(updateGeometry()))),
      sample_count_property_(new rviz::IntProperty(
          "Sample Count", 80,
          "Number of terminal Gaussian samples per frame and axis.", this,
          SLOT(updateGeometry()))),
      axis_length_property_(new rviz::FloatProperty(
          "Axis Length", 0.08,
          "Distance from each sampled frame origin to its colored endpoints.",
          this, SLOT(updateGeometry()))),
      point_size_property_(new rviz::FloatProperty(
          "Point Size", 0.006, "Rendered diameter of each endpoint.", this,
          SLOT(updateAppearance()))),
      point_style_property_(new rviz::EnumProperty(
          "Point Style", "Spheres", "Rendering primitive for sampled points.",
          this, SLOT(updateAppearance()))),
      alpha_property_(new rviz::FloatProperty(
          "Alpha", 0.75, "Opacity of clouds and representative axes.", this,
          SLOT(updateAppearance()))),
      show_representative_property_(new rviz::BoolProperty(
          "Show Representatives", true,
          "Draw a composed representative coordinate frame for every child.",
          this, SLOT(updateAppearance()))),
      representative_radius_property_(new rviz::FloatProperty(
          "Representative Radius", 0.004,
          "Radius of the representative coordinate axes.", this,
          SLOT(updateAppearance()))),
      seed_property_(new rviz::IntProperty(
          "Random Seed", 29, "Stable seed for terminal point sampling.", this,
          SLOT(updateGeometry()))) {
  queue_size_property_->setMin(1);
  queue_size_property_->setMax(10000);
  frame_timeout_property_->setMin(1.0);
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

ProbabilisticTfDisplay::~ProbabilisticTfDisplay() = default;

void ProbabilisticTfDisplay::onInitialize() {
  Display::onInitialize();
  updateAppearance();
}

void ProbabilisticTfDisplay::reset() {
  Display::reset();
  impl_->clear();
}

void ProbabilisticTfDisplay::update(float /*wall_dt*/, float /*ros_dt*/) {
  impl_->update();
}

void ProbabilisticTfDisplay::onEnable() {
  updateTopics();
}

void ProbabilisticTfDisplay::onDisable() {
  impl_->unsubscribe();
  reset();
}

void ProbabilisticTfDisplay::fixedFrameChanged() {
  context_->queueRender();
}

void ProbabilisticTfDisplay::updateTopics() {
  impl_->unsubscribe();
  impl_->clear();
  if (!isEnabled()) {
    return;
  }
  try {
    impl_->subscribe(
        update_nh_, dynamic_topic_property_->getTopicStd(),
        dynamic_batch_topic_property_->getTopicStd(),
        static_topic_property_->getTopicStd(),
        static_cast<uint32_t>(queue_size_property_->getInt()));
    setStatus(rviz::StatusProperty::Ok, "Topics", "Subscribed.");
  } catch (const ros::Exception& exception) {
    setStatus(rviz::StatusProperty::Error, "Topics",
              QString::fromStdString(exception.what()));
  }
}

void ProbabilisticTfDisplay::updateAppearance() {
  VisualStyle style;
  style.point_size = point_size_property_->getFloat();
  style.alpha = alpha_property_->getFloat();
  style.axis_length = axis_length_property_->getFloat();
  style.axis_radius = representative_radius_property_->getFloat();
  style.show_representative = show_representative_property_->getBool();
  style.render_mode = renderMode(point_style_property_);
  impl_->updateStyles(style);
  context_->queueRender();
}

void ProbabilisticTfDisplay::updateGeometry() {
  updateAppearance();
  impl_->markDirty();
}

}  // namespace probtf_rviz
