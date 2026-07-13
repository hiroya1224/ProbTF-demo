#include <probtf_core/latest_snapshot.hpp>

#include <Eigen/Eigenvalues>

#include <probtf_msgs/ProbabilisticTransformArray.h>
#include <probtf_msgs/ProbabilisticTransformStamped.h>
#include <ros/ros.h>
#include <urdf/model.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::string cleanFrame(const std::string& value) {
  const std::size_t first = value.find_first_not_of("/ \t\r\n");
  if (first == std::string::npos) {
    return std::string();
  }
  const std::size_t last = value.find_last_not_of("/ \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> stringListParam(ros::NodeHandle& node,
                                         const std::string& name) {
  XmlRpc::XmlRpcValue value;
  if (!node.getParam(name, value)) {
    return {};
  }
  std::vector<std::string> output;
  if (value.getType() == XmlRpc::XmlRpcValue::TypeString) {
    std::string text = static_cast<std::string>(value);
    std::size_t begin = 0;
    while (begin <= text.size()) {
      const std::size_t end = text.find(',', begin);
      const std::string entry = cleanFrame(
          text.substr(begin, end == std::string::npos ? std::string::npos
                                                     : end - begin));
      if (!entry.empty()) {
        output.push_back(entry);
      }
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1;
    }
  } else if (value.getType() == XmlRpc::XmlRpcValue::TypeArray) {
    for (int index = 0; index < value.size(); ++index) {
      if (value[index].getType() != XmlRpc::XmlRpcValue::TypeString) {
        throw std::runtime_error("~" + name + " must contain only strings.");
      }
      const std::string entry = cleanFrame(static_cast<std::string>(value[index]));
      if (!entry.empty()) {
        output.push_back(entry);
      }
    }
  } else {
    throw std::runtime_error("~" + name + " must be a string or string array.");
  }
  return output;
}

Eigen::Vector3d vectorParam(ros::NodeHandle& node,
                            const std::string& name,
                            const Eigen::Vector3d& fallback) {
  XmlRpc::XmlRpcValue value;
  if (!node.getParam(name, value)) {
    return fallback;
  }
  if (value.getType() != XmlRpc::XmlRpcValue::TypeArray || value.size() != 3) {
    throw std::runtime_error("~" + name + " must contain three numbers.");
  }
  Eigen::Vector3d output;
  for (int index = 0; index < 3; ++index) {
    if (value[index].getType() == XmlRpc::XmlRpcValue::TypeInt) {
      output(index) = static_cast<int>(value[index]);
    } else if (value[index].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
      output(index) = static_cast<double>(value[index]);
    } else {
      throw std::runtime_error("~" + name + " must contain three numbers.");
    }
  }
  if (!output.allFinite()) {
    throw std::runtime_error("~" + name + " must be finite.");
  }
  return output;
}

int linkDepth(urdf::LinkConstSharedPtr link) {
  int depth = 0;
  while (link && link->getParent()) {
    ++depth;
    link = link->getParent();
  }
  return depth;
}

std::string inferTip(const urdf::Model& model) {
  std::string selected;
  int selected_depth = -1;
  for (const auto& keyed : model.joints_) {
    const urdf::JointConstSharedPtr& joint = keyed.second;
    if (!joint || joint->type == urdf::Joint::FIXED || joint->mimic ||
        (joint->limits && joint->limits->velocity <= 0.0)) {
      continue;
    }
    const urdf::LinkConstSharedPtr child = model.getLink(joint->child_link_name);
    const int depth = linkDepth(child);
    if (depth > selected_depth ||
        (depth == selected_depth && joint->child_link_name > selected)) {
      selected = joint->child_link_name;
      selected_depth = depth;
    }
  }
  if (!selected.empty()) {
    return selected;
  }
  for (const auto& keyed : model.links_) {
    const int depth = linkDepth(keyed.second);
    if (depth > selected_depth ||
        (depth == selected_depth && keyed.first > selected)) {
      selected = keyed.first;
      selected_depth = depth;
    }
  }
  return selected;
}

geometry_msgs::Point pointMessage(const Eigen::Vector3d& value) {
  geometry_msgs::Point output;
  output.x = value.x();
  output.y = value.y();
  output.z = value.z();
  return output;
}

std::string markerNamespace(std::string value) {
  std::replace(value.begin(), value.end(), '/', '_');
  return value;
}

class DeflecompProbTfPointMomentsNode {
 public:
  DeflecompProbTfPointMomentsNode() : private_node_("~") {
    private_node_.param<std::string>("dynamic_topic", dynamic_topic_,
                                     "/deflecomp/probtf");
    private_node_.param<std::string>("dynamic_batch_topic", batch_topic_,
                                     "/deflecomp/probtf_batch");
    private_node_.param<std::string>("static_topic", static_topic_,
                                     "/deflecomp/probtf_static");
    private_node_.param<std::string>("marker_topic", marker_topic_,
                                     "/deflecomp/probtf_point_moments");
    private_node_.param("subscribe_individual_dynamic",
                        subscribe_individual_dynamic_, false);
    private_node_.param("lookup_rate_hz", lookup_rate_hz_, 50.0);
    private_node_.param("marker_max_age", marker_max_age_, 0.5);
    private_node_.param("sigma_scale", sigma_scale_, 2.0);
    private_node_.param("point_scale", point_scale_, 0.025);
    private_node_.param("axis_width", axis_width_, 0.006);
    private_node_.param("bingham_integration_steps", integration_steps_, 120);
    if (!std::isfinite(lookup_rate_hz_) || lookup_rate_hz_ <= 0.0 ||
        !std::isfinite(marker_max_age_) || marker_max_age_ < 0.0 ||
        !std::isfinite(sigma_scale_) || sigma_scale_ < 0.0 ||
        !std::isfinite(point_scale_) || point_scale_ <= 0.0 ||
        !std::isfinite(axis_width_) || axis_width_ <= 0.0 ||
        integration_steps_ < 1) {
      throw std::runtime_error("Invalid Prob-TF marker numeric parameter.");
    }
    source_point_ =
        vectorParam(private_node_, "source_point", Eigen::Vector3d::Zero());

    std::string urdf_path;
    private_node_.param<std::string>("urdf_path", urdf_path, "");
    if (urdf_path.empty()) {
      throw std::runtime_error("~urdf_path is required.");
    }
    urdf::Model model;
    if (!model.initFile(urdf_path)) {
      throw std::runtime_error("Unable to parse URDF file '" + urdf_path + "'.");
    }
    private_node_.param<std::string>("target_frame", target_frame_, "");
    target_frame_ = cleanFrame(target_frame_);
    if (target_frame_.empty()) {
      target_frame_ = model.getLink("base_link")
                          ? "base_link"
                          : (model.getRoot() ? model.getRoot()->name : "");
    }
    std::string tip_frame;
    private_node_.param<std::string>("tip_frame", tip_frame, "");
    tip_frame = cleanFrame(tip_frame);
    if (tip_frame.empty()) {
      tip_frame = inferTip(model);
    }
    if (target_frame_.empty() || tip_frame.empty()) {
      throw std::runtime_error("Unable to infer target or tip frame from URDF.");
    }

    source_frames_ = stringListParam(private_node_, "source_frames");
    if (source_frames_.empty()) {
      std::vector<std::string> prefixes =
          stringListParam(private_node_, "frame_prefixes");
      if (prefixes.empty()) {
        prefixes = {"ref", "cmd", "equil"};
      }
      for (const std::string& prefix : prefixes) {
        source_frames_.push_back(prefix + "/" + tip_frame);
      }
    }
    if (source_frames_.empty()) {
      throw std::runtime_error("At least one Prob-TF source frame is required.");
    }

    marker_publisher_ =
        node_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1, false);
    batch_subscriber_ = node_.subscribe(
        batch_topic_, 1, &DeflecompProbTfPointMomentsNode::onBatch, this,
        ros::TransportHints().tcpNoDelay());
    static_subscriber_ = node_.subscribe(
        static_topic_, 1, &DeflecompProbTfPointMomentsNode::onStatic, this,
        ros::TransportHints().tcpNoDelay());
    if (subscribe_individual_dynamic_) {
      dynamic_subscriber_ = node_.subscribe(
          dynamic_topic_, 1,
          &DeflecompProbTfPointMomentsNode::onIndividualDynamic, this,
          ros::TransportHints().tcpNoDelay());
    }
    worker_ = std::thread(&DeflecompProbTfPointMomentsNode::workerLoop, this);

    ROS_INFO_STREAM("Deflecomp C++ Prob-TF consumer: " << target_frame_ << " <- "
                                                        << joinSources()
                                                        << " via " << batch_topic_);
  }

  ~DeflecompProbTfPointMomentsNode() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  using Array = probtf_msgs::ProbabilisticTransformArray;
  using ArrayConstPtr = probtf_msgs::ProbabilisticTransformArray::ConstPtr;

  std::string joinSources() const {
    std::string output;
    for (const std::string& source : source_frames_) {
      if (!output.empty()) {
        output += ", ";
      }
      output += source;
    }
    return output;
  }

  void onBatch(const ArrayConstPtr& message) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_batch_ = message;
      ++generation_;
      work_pending_ = true;
    }
    condition_.notify_one();
  }

  void onStatic(const ArrayConstPtr& message) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_static_ = message;
      ++generation_;
      work_pending_ = true;
    }
    condition_.notify_one();
  }

  void onIndividualDynamic(
      const probtf_msgs::ProbabilisticTransformStamped::ConstPtr& message) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto existing = legacy_records_.find(message->edge_id);
      if (existing != legacy_records_.end() &&
          message->header.stamp < existing->second.header.stamp) {
        return;
      }
      legacy_records_[message->edge_id] = *message;
      ++generation_;
      work_pending_ = true;
    }
    condition_.notify_one();
  }

  ArrayConstPtr legacyBatchLocked() const {
    if (legacy_records_.empty()) {
      return ArrayConstPtr();
    }
    boost::shared_ptr<Array> batch(new Array());
    batch->header.stamp = ros::Time::now();
    batch->transforms.reserve(legacy_records_.size());
    for (const auto& keyed : legacy_records_) {
      batch->transforms.push_back(keyed.second);
    }
    return batch;
  }

  visualization_msgs::MarkerArray computeMarkers(const Array& dynamic_records,
                                                   const Array& static_records) {
    probtf_core::LatestSnapshot snapshot(dynamic_records, static_records);
    std::string validation_error;
    if (!snapshot.valid(&validation_error)) {
      throw std::runtime_error(validation_error);
    }

    visualization_msgs::MarkerArray output;
    visualization_msgs::Marker clear;
    clear.action = visualization_msgs::Marker::DELETEALL;
    output.markers.push_back(clear);
    const ros::Duration lifetime(2.5 / lookup_rate_hz_);
    const ros::Time now = ros::Time::now();

    static const std::array<std::array<float, 3>, 4> colors = {{
        {{0.18F, 0.55F, 0.95F}},
        {{0.95F, 0.38F, 0.22F}},
        {{0.20F, 0.75F, 0.42F}},
        {{0.76F, 0.45F, 0.90F}},
    }};

    for (std::size_t source_index = 0; source_index < source_frames_.size();
         ++source_index) {
      probtf_core::PointMomentObservation observation;
      std::string error;
      if (!snapshot.lookupPointMoments(
              target_frame_, source_frames_[source_index], source_point_,
              &observation, &error, integration_steps_)) {
        ROS_WARN_STREAM_THROTTLE(
            3.0, "Deflecomp Prob-TF lookup " << target_frame_ << " <- "
                                              << source_frames_[source_index]
                                              << " is unavailable: " << error);
        continue;
      }
      if (!observation.resolved_stamp.isZero()) {
        const double age = std::max(
            0.0, (now - observation.resolved_stamp).toSec());
        if (age > marker_max_age_) {
          ROS_WARN_STREAM_THROTTLE(
              3.0, "Deflecomp Prob-TF lookup " << target_frame_ << " <- "
                                                << source_frames_[source_index]
                                                << " is stale by " << age
                                                << " s; marker suppressed");
          continue;
        }
      }

      const auto& color = colors[source_index % colors.size()];
      const std::string marker_namespace =
          markerNamespace(observation.source_frame);
      const ros::Time stamp = observation.resolved_stamp.isZero()
                                  ? dynamic_records.header.stamp
                                  : observation.resolved_stamp;

      visualization_msgs::Marker mean;
      mean.header.frame_id = observation.target_frame;
      mean.header.stamp = stamp;
      mean.ns = marker_namespace;
      mean.id = static_cast<int>(2 * source_index);
      mean.type = visualization_msgs::Marker::SPHERE;
      mean.action = visualization_msgs::Marker::ADD;
      mean.pose.position = pointMessage(observation.moments.mean);
      mean.pose.orientation.w = 1.0;
      mean.scale.x = point_scale_;
      mean.scale.y = point_scale_;
      mean.scale.z = point_scale_;
      mean.color.r = color[0];
      mean.color.g = color[1];
      mean.color.b = color[2];
      mean.color.a = 0.95F;
      mean.lifetime = lifetime;
      output.markers.push_back(mean);

      visualization_msgs::Marker axes;
      axes.header = mean.header;
      axes.ns = marker_namespace;
      axes.id = static_cast<int>(2 * source_index + 1);
      axes.type = visualization_msgs::Marker::LINE_LIST;
      axes.action = visualization_msgs::Marker::ADD;
      axes.pose.orientation.w = 1.0;
      axes.scale.x = axis_width_;
      axes.color.r = color[0];
      axes.color.g = color[1];
      axes.color.b = color[2];
      axes.color.a = 0.8F;
      axes.lifetime = lifetime;
      const Eigen::Matrix3d covariance =
          0.5 * (observation.moments.covariance +
                 observation.moments.covariance.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
      if (solver.info() != Eigen::Success) {
        ROS_WARN_STREAM_THROTTLE(3.0,
                                 "Prob-TF covariance eigendecomposition failed");
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          const double radius =
              sigma_scale_ * std::sqrt(std::max(0.0, solver.eigenvalues()(axis)));
          const Eigen::Vector3d offset = radius * solver.eigenvectors().col(axis);
          axes.points.push_back(pointMessage(observation.moments.mean - offset));
          axes.points.push_back(pointMessage(observation.moments.mean + offset));
        }
      }
      output.markers.push_back(axes);
    }
    return output;
  }

  void workerLoop() {
    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / lookup_rate_hz_));
    Clock::time_point next_publish = Clock::now();

    while (ros::ok()) {
      ArrayConstPtr dynamic_records;
      ArrayConstPtr static_records;
      std::uint64_t generation = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stop_ || work_pending_; });
        if (stop_) {
          return;
        }
        if (Clock::now() < next_publish) {
          condition_.wait_until(lock, next_publish,
                                [this] { return stop_; });
          if (stop_) {
            return;
          }
        }
        dynamic_records = latest_batch_ ? latest_batch_ : legacyBatchLocked();
        static_records = latest_static_;
        generation = generation_;
        work_pending_ = false;
      }
      if (!dynamic_records || !static_records) {
        continue;
      }

      const Clock::time_point compute_start = Clock::now();
      visualization_msgs::MarkerArray markers;
      try {
        markers = computeMarkers(*dynamic_records, *static_records);
      } catch (const std::exception& error) {
        ROS_WARN_STREAM_THROTTLE(3.0,
                                 "Prob-TF marker computation failed: "
                                     << error.what());
        continue;
      }

      bool superseded = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        superseded = generation_ != generation;
        if (superseded) {
          work_pending_ = true;
        }
      }
      if (superseded) {
        ++discarded_results_;
        condition_.notify_one();
        continue;
      }

      marker_publisher_.publish(markers);
      const Clock::time_point publish_end = Clock::now();
      const Clock::time_point scheduled = next_publish + period;
      next_publish = scheduled > publish_end ? scheduled : publish_end + period;
      const double compute_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - compute_start)
              .count();
      ROS_DEBUG_STREAM_THROTTLE(
          5.0, "Prob-TF marker compute " << compute_ms << " ms; discarded "
                                          << discarded_results_
                                          << " superseded results");
    }
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  std::string dynamic_topic_;
  std::string batch_topic_;
  std::string static_topic_;
  std::string marker_topic_;
  std::string target_frame_;
  std::vector<std::string> source_frames_;
  Eigen::Vector3d source_point_ = Eigen::Vector3d::Zero();
  bool subscribe_individual_dynamic_ = false;
  double lookup_rate_hz_ = 50.0;
  double marker_max_age_ = 0.5;
  double sigma_scale_ = 2.0;
  double point_scale_ = 0.025;
  double axis_width_ = 0.006;
  int integration_steps_ = 120;

  ros::Publisher marker_publisher_;
  ros::Subscriber batch_subscriber_;
  ros::Subscriber static_subscriber_;
  ros::Subscriber dynamic_subscriber_;

  std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
  bool work_pending_ = false;
  ArrayConstPtr latest_batch_;
  ArrayConstPtr latest_static_;
  std::map<std::string, probtf_msgs::ProbabilisticTransformStamped>
      legacy_records_;
  std::uint64_t generation_ = 0;
  std::uint64_t discarded_results_ = 0;
  std::thread worker_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "deflecomp_probtf_point_moments");
  try {
    DeflecompProbTfPointMomentsNode node;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("Deflecomp C++ Prob-TF consumer failed: " << error.what());
    return 1;
  }
  return 0;
}
