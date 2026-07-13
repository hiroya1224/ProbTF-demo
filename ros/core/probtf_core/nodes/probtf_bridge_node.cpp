#include <probtf_core/latest_snapshot.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <probtf_msgs/ProbabilisticTransformArray.h>
#include <probtf_msgs/ProbabilisticTransformStamped.h>
#include <ros/message_event.h>
#include <ros/ros.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <map>
#include <mutex>
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

Eigen::Matrix<double, 3, 9> couplingMatrix(
    const boost::array<double, 27>& packed) {
  Eigen::Matrix<double, 3, 9> output;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 9; ++column) {
      output(row, column) = packed[static_cast<std::size_t>(row * 9 + column)];
    }
  }
  return output;
}

Eigen::Matrix4d unpackShape(const boost::array<double, 10>& packed) {
  Eigen::Matrix4d output = Eigen::Matrix4d::Zero();
  std::size_t index = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = row; column < 4; ++column) {
      output(row, column) = packed[index++];
      output(column, row) = output(row, column);
    }
  }
  return output;
}

Eigen::Matrix<double, 9, 1> rotationVector(
    const Eigen::Matrix3d& rotation) {
  Eigen::Matrix<double, 9, 1> output;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      output(row + 3 * column) = rotation(row, column);
    }
  }
  return output;
}

bool componentMode(
    const probtf_msgs::ProbabilisticTransformComponent& component,
    Eigen::Quaterniond* mode,
    std::string* error) {
  if (component.orientation.kind == probtf_msgs::BinghamOrientation::DIRAC ||
      component.orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    *mode = Eigen::Quaterniond(component.orientation.reference_quaternion.w,
                               component.orientation.reference_quaternion.x,
                               component.orientation.reference_quaternion.y,
                               component.orientation.reference_quaternion.z);
  } else if (component.orientation.kind ==
             probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(
        unpackShape(component.orientation.shape_upper_wxyz));
    if (solver.info() != Eigen::Success) {
      *error = "Bingham mode eigendecomposition failed.";
      return false;
    }
    Eigen::Vector4d vector = solver.eigenvectors().col(3);
    Eigen::Index pivot = 0;
    vector.cwiseAbs().maxCoeff(&pivot);
    if (vector(pivot) < 0.0) {
      vector = -vector;
    }
    *mode = Eigen::Quaterniond(vector(0), vector(1), vector(2), vector(3));
  } else {
    *error = "Unknown orientation kind.";
    return false;
  }
  if (!mode->coeffs().allFinite() || mode->norm() <= 1.0e-12) {
    *error = "Representative quaternion is invalid.";
    return false;
  }
  mode->normalize();
  return true;
}

class ProbTfBridgeNode {
 public:
  ProbTfBridgeNode() : private_node_("~") {
    private_node_.param("import_tf", import_tf_, true);
    private_node_.param("export_tf", export_tf_, true);
    private_node_.param("publish_individual_dynamic",
                        publish_individual_dynamic_, true);
    private_node_.param("tf_import_max_rate_hz", import_rate_hz_, 0.0);
    if (!std::isfinite(import_rate_hz_) || import_rate_hz_ < 0.0) {
      throw std::runtime_error(
          "~tf_import_max_rate_hz must be finite and non-negative.");
    }
    private_node_.param<std::string>("tf_export_policy", export_policy_,
                                     "exact_only");
    if (export_policy_ != "exact_only" &&
        export_policy_ != "stored_representative" &&
        export_policy_ != "highest_weight_component_mode") {
      throw std::runtime_error("Unknown ~tf_export_policy '" + export_policy_ +
                               "'.");
    }
    private_node_.param<std::string>("probtf_topic", dynamic_topic_,
                                     "/probtf");
    private_node_.param<std::string>("probtf_batch_topic", batch_topic_,
                                     "/probtf_batch");
    private_node_.param<std::string>("probtf_static_topic", static_topic_,
                                     "/probtf_static");
    prefixes_ = stringListParam(private_node_, "tf_import_child_prefixes");

    if (import_tf_) {
      batch_publisher_ =
          node_.advertise<probtf_msgs::ProbabilisticTransformArray>(
              batch_topic_, 1, false);
      static_publisher_ =
          node_.advertise<probtf_msgs::ProbabilisticTransformArray>(
              static_topic_, 1, true);
      if (publish_individual_dynamic_) {
        dynamic_publisher_ =
            node_.advertise<probtf_msgs::ProbabilisticTransformStamped>(
                dynamic_topic_, 100, false);
      }
      tf_subscriber_ = node_.subscribe(
          "/tf", 1, &ProbTfBridgeNode::onTf, this,
          ros::TransportHints().tcpNoDelay());
      tf_static_subscriber_ = node_.subscribe(
          // /tf_static has one latched connection per publisher.  A queue of
          // one can drop a different publisher's startup message, even though
          // updates are latest-only per edge.
          "/tf_static", 100, &ProbTfBridgeNode::onTfStatic, this,
          ros::TransportHints().tcpNoDelay());
      worker_ = std::thread(&ProbTfBridgeNode::workerLoop, this);
    }

    if (export_tf_) {
      probtf_subscriber_ = node_.subscribe(
          dynamic_topic_, 100, &ProbTfBridgeNode::onProbTf, this,
          ros::TransportHints().tcpNoDelay());
      probtf_static_subscriber_ = node_.subscribe(
          static_topic_, 1, &ProbTfBridgeNode::onProbTfStatic, this,
          ros::TransportHints().tcpNoDelay());
    }

    ROS_INFO_STREAM("Prob-TF C++ bridge: latest-only TF -> " << batch_topic_
                    << " (max " << import_rate_hz_
                    << " Hz, individual compatibility="
                    << (publish_individual_dynamic_ ? "on" : "off") << ")");
  }

  ~ProbTfBridgeNode() {
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
  struct TfSample {
    geometry_msgs::TransformStamped transform;
    std::string authority;
  };

  using TfSampleMap = std::map<std::string, TfSample>;
  using RecordMap =
      std::map<std::string, probtf_msgs::ProbabilisticTransformStamped>;

  bool matchesPrefix(const std::string& child_value) const {
    if (prefixes_.empty()) {
      return true;
    }
    const std::string child = cleanFrame(child_value);
    for (const std::string& prefix : prefixes_) {
      if (child == prefix || child.compare(0, prefix.size() + 1,
                                           prefix + "/") == 0) {
        return true;
      }
    }
    return false;
  }

  static std::string key(const geometry_msgs::TransformStamped& transform) {
    return cleanFrame(transform.header.frame_id) + "\n" +
           cleanFrame(transform.child_frame_id);
  }

  void stageTf(const tf2_msgs::TFMessage::ConstPtr& message,
               const std::string& authority,
               bool is_static) {
    if (authority == ros::this_node::getName()) {
      return;
    }
    bool changed = false;
    std::lock_guard<std::mutex> lock(mutex_);
    TfSampleMap& pending = is_static ? pending_static_ : pending_dynamic_;
    std::map<std::string, ros::Time>& newest =
        is_static ? newest_static_stamp_ : newest_dynamic_stamp_;
    for (const auto& transform : message->transforms) {
      if (!matchesPrefix(transform.child_frame_id)) {
        continue;
      }
      const std::string edge_key = key(transform);
      const auto existing = newest.find(edge_key);
      if (existing != newest.end() && transform.header.stamp < existing->second) {
        continue;
      }
      newest[edge_key] = transform.header.stamp;
      pending[edge_key] = TfSample{transform, authority};
      changed = true;
    }
    if (changed) {
      if (is_static) {
        ++static_generation_;
      } else {
        ++dynamic_generation_;
      }
      condition_.notify_one();
    }
  }

  void onTf(const ros::MessageEvent<tf2_msgs::TFMessage const>& event) {
    stageTf(event.getMessage(), event.getPublisherName(), false);
  }

  void onTfStatic(const ros::MessageEvent<tf2_msgs::TFMessage const>& event) {
    stageTf(event.getMessage(), event.getPublisherName(), true);
  }

  std::vector<probtf_msgs::ProbabilisticTransformStamped> convert(
      const TfSampleMap& samples,
      bool is_static) {
    std::vector<probtf_msgs::ProbabilisticTransformStamped> output;
    output.reserve(samples.size());
    for (const auto& keyed : samples) {
      probtf_msgs::ProbabilisticTransformStamped record;
      std::string error;
      if (!probtf_core::deterministicTfToProbTf(
              keyed.second.transform, keyed.second.authority, is_static,
              &record, &error)) {
        ROS_WARN_STREAM_THROTTLE(3.0, "Ignoring invalid TF edge: " << error);
        continue;
      }
      output.push_back(std::move(record));
    }
    return output;
  }

  void workerLoop() {
    using Clock = std::chrono::steady_clock;
    const auto period = import_rate_hz_ > 0.0
                            ? std::chrono::duration_cast<Clock::duration>(
                                  std::chrono::duration<double>(1.0 /
                                                                import_rate_hz_))
                            : Clock::duration::zero();
    Clock::time_point next_dynamic_publish = Clock::now();

    while (ros::ok()) {
      TfSampleMap dynamic_samples;
      TfSampleMap static_samples;
      std::uint64_t dynamic_generation = 0;
      std::uint64_t static_generation = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return stop_ || !pending_dynamic_.empty() || !pending_static_.empty();
        });
        if (stop_) {
          return;
        }
        if (!pending_dynamic_.empty() && pending_static_.empty() &&
            Clock::now() < next_dynamic_publish) {
          condition_.wait_until(lock, next_dynamic_publish, [this] {
            return stop_ || !pending_static_.empty();
          });
          if (stop_) {
            return;
          }
        }
        if (!pending_static_.empty()) {
          static_samples.swap(pending_static_);
          static_generation = static_generation_;
        }
        if (!pending_dynamic_.empty() && Clock::now() >= next_dynamic_publish) {
          dynamic_samples.swap(pending_dynamic_);
          dynamic_generation = dynamic_generation_;
        }
      }

      if (!static_samples.empty()) {
        const auto records = convert(static_samples, true);
        for (const auto& record : records) {
          static_records_[record.edge_id] = record;
        }
        bool superseded = false;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          superseded = static_generation_ != static_generation;
        }
        if (!superseded) {
          probtf_msgs::ProbabilisticTransformArray message;
          message.header.seq = static_sequence_++;
          message.header.stamp = ros::Time::now();
          for (const auto& keyed : static_records_) {
            message.transforms.push_back(keyed.second);
          }
          static_publisher_.publish(message);
        }
      }

      if (dynamic_samples.empty()) {
        continue;
      }
      auto records = convert(dynamic_samples, false);
      for (const auto& record : records) {
        dynamic_records_[record.edge_id] = record;
      }
      bool superseded = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        superseded = dynamic_generation_ != dynamic_generation;
      }
      if (superseded) {
        ++discarded_batches_;
        continue;
      }

      probtf_msgs::ProbabilisticTransformArray batch;
      batch.header.seq = dynamic_sequence_++;
      batch.header.stamp = ros::Time::now();
      batch.transforms.reserve(dynamic_records_.size());
      for (const auto& keyed : dynamic_records_) {
        batch.transforms.push_back(keyed.second);
      }
      batch_publisher_.publish(batch);
      if (publish_individual_dynamic_) {
        for (const auto& record : records) {
          dynamic_publisher_.publish(record);
        }
      }
      const Clock::time_point now = Clock::now();
      const Clock::time_point scheduled = next_dynamic_publish + period;
      next_dynamic_publish =
          scheduled > now ? scheduled : now + period;
      ROS_DEBUG_STREAM_THROTTLE(
          5.0, "Prob-TF bridge published " << batch.transforms.size()
                                            << " latest edges; discarded "
                                            << discarded_batches_
                                            << " superseded batches");
    }
  }

  bool representativeTransform(
      const probtf_msgs::ProbabilisticTransformStamped& record,
      geometry_msgs::TransformStamped* output,
      std::string* error) const {
    if (probtf_core::exactProbTfToTf(record, output, error)) {
      return true;
    }
    if (export_policy_ == "exact_only") {
      return false;
    }
    *output = geometry_msgs::TransformStamped();
    output->header = record.header;
    output->child_frame_id = record.child_frame_id;
    if (export_policy_ == "stored_representative") {
      if (record.representative_kind ==
          probtf_msgs::ProbabilisticTransformStamped::REPRESENTATIVE_NONE) {
        *error = "No stored representative is available.";
        return false;
      }
      output->transform = record.representative;
      return true;
    }

    const probtf_msgs::ProbabilisticTransformComponent* selected = nullptr;
    double selected_weight = -1.0;
    for (const auto& component : record.components) {
      if (std::isfinite(component.weight) && component.weight > selected_weight) {
        selected = &component;
        selected_weight = component.weight;
      }
    }
    if (selected == nullptr || selected_weight <= 0.0) {
      *error = "Distribution has no positive finite component weight.";
      return false;
    }
    Eigen::Quaterniond mode;
    if (!componentMode(*selected, &mode, error)) {
      return false;
    }
    Eigen::Quaterniond reference(
        selected->orientation.reference_quaternion.w,
        selected->orientation.reference_quaternion.x,
        selected->orientation.reference_quaternion.y,
        selected->orientation.reference_quaternion.z);
    if (!reference.coeffs().allFinite() || reference.norm() <= 1.0e-12) {
      *error = "Component reference quaternion is invalid.";
      return false;
    }
    reference.normalize();
    const Eigen::Vector3d mean(
        selected->translation.mean_at_reference.x,
        selected->translation.mean_at_reference.y,
        selected->translation.mean_at_reference.z);
    const Eigen::Vector3d translation =
        mean + couplingMatrix(selected->translation.rotation_coupling) *
                   (rotationVector(mode.toRotationMatrix()) -
                    rotationVector(reference.toRotationMatrix()));
    output->transform.translation.x = translation.x();
    output->transform.translation.y = translation.y();
    output->transform.translation.z = translation.z();
    output->transform.rotation.w = mode.w();
    output->transform.rotation.x = mode.x();
    output->transform.rotation.y = mode.y();
    output->transform.rotation.z = mode.z();
    return true;
  }

  void exportRecord(
      const probtf_msgs::ProbabilisticTransformStamped& record) {
    geometry_msgs::TransformStamped transform;
    std::string error;
    if (!representativeTransform(record, &transform, &error)) {
      ROS_WARN_STREAM_THROTTLE(
          5.0, "Prob-TF edge '" << record.edge_id
                                 << "' was not exported to TF: " << error);
      return;
    }
    if (record.is_static) {
      static_broadcaster_.sendTransform(transform);
    } else {
      broadcaster_.sendTransform(transform);
    }
  }

  void onProbTf(
      const ros::MessageEvent<
          probtf_msgs::ProbabilisticTransformStamped const>& event) {
    if (event.getPublisherName() == ros::this_node::getName()) {
      return;
    }
    exportRecord(*event.getMessage());
  }

  void onProbTfStatic(
      const ros::MessageEvent<probtf_msgs::ProbabilisticTransformArray const>&
          event) {
    if (event.getPublisherName() == ros::this_node::getName()) {
      return;
    }
    for (const auto& record : event.getMessage()->transforms) {
      exportRecord(record);
    }
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  bool import_tf_ = true;
  bool export_tf_ = true;
  bool publish_individual_dynamic_ = true;
  double import_rate_hz_ = 0.0;
  std::string export_policy_;
  std::string dynamic_topic_;
  std::string batch_topic_;
  std::string static_topic_;
  std::vector<std::string> prefixes_;

  ros::Publisher dynamic_publisher_;
  ros::Publisher batch_publisher_;
  ros::Publisher static_publisher_;
  ros::Subscriber tf_subscriber_;
  ros::Subscriber tf_static_subscriber_;
  ros::Subscriber probtf_subscriber_;
  ros::Subscriber probtf_static_subscriber_;
  tf2_ros::TransformBroadcaster broadcaster_;
  tf2_ros::StaticTransformBroadcaster static_broadcaster_;

  std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
  TfSampleMap pending_dynamic_;
  TfSampleMap pending_static_;
  std::map<std::string, ros::Time> newest_dynamic_stamp_;
  std::map<std::string, ros::Time> newest_static_stamp_;
  std::uint64_t dynamic_generation_ = 0;
  std::uint64_t static_generation_ = 0;
  RecordMap dynamic_records_;
  RecordMap static_records_;
  std::thread worker_;
  std::uint32_t dynamic_sequence_ = 0;
  std::uint32_t static_sequence_ = 0;
  std::uint64_t discarded_batches_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "probtf_bridge");
  try {
    ProbTfBridgeNode node;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("Prob-TF C++ bridge failed: " << error.what());
    return 1;
  }
  return 0;
}
