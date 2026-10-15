#pragma once

// A resolved query already contains weighted atoms or joint samples.
// Rendering them must not draw another random empirical approximation.
#include <probtf_msgs/ProbabilisticTransformStamped.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace probtf_rviz {

struct RecordDisplayMetadata {
  bool direct_support = false;
  bool deterministic_quadrature = false;
  bool frozen_query = false;
  std::string result_kind;
  std::string integration_method;
  std::string source_law_id;
  std::string query_id;
  double source_revision = 0.0;
  double source_observation_stamp = 0.0;
  int order = 0;
  int sample_count = 0;
  double seed = 0.0;
};

inline bool displayFailure(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

inline bool finiteJsonNumber(const QJsonValue& value, double* number) {
  if (!value.isDouble() || !std::isfinite(value.toDouble())) {
    return false;
  }
  *number = value.toDouble();
  return true;
}

inline bool recordDisplayMetadata(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    RecordDisplayMetadata* metadata, std::string* error = nullptr) {
  if (metadata == nullptr) {
    return displayFailure(error, "Display metadata output is null.");
  }
  *metadata = RecordDisplayMetadata();
  const std::string quadrature_kind = "joint_pushforward_quadrature";
  const std::string finite_kind = "joint_finite_pushforward";
  const std::string sample_kind = "joint_transform_samples";
  const bool declared_support =
      record.provenance.method.find(quadrature_kind) != std::string::npos ||
      record.provenance.method.find(finite_kind) != std::string::npos ||
      record.provenance.method.find(sample_kind) != std::string::npos;
  QJsonParseError parse_error;
  const QJsonDocument document = QJsonDocument::fromJson(
      QByteArray::fromStdString(record.provenance.detail), &parse_error);
  if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
    return !declared_support || displayFailure(
        error, "Joint support query requires structured source-law metadata.");
  }
  const QJsonObject object = document.object();
  metadata->result_kind = object.value("result_kind").toString().toStdString();
  metadata->deterministic_quadrature = metadata->result_kind == quadrature_kind;
  metadata->direct_support = metadata->deterministic_quadrature ||
      metadata->result_kind == finite_kind || metadata->result_kind == sample_kind;
  metadata->frozen_query =
      object.value("display_result_kind").toString() == "frozen_law_query";
  if (declared_support && !metadata->direct_support) {
    return displayFailure(error, "Joint support method and result_kind disagree.");
  }
  if (metadata->direct_support || metadata->frozen_query) {
    if (!finiteJsonNumber(object.value("source_observation_stamp"),
                          &metadata->source_observation_stamp) ||
        metadata->source_observation_stamp < 0.0 ||
        std::abs(metadata->source_observation_stamp - record.header.stamp.toSec()) >
            1.0e-7) {
      return displayFailure(error,
          "Display query must preserve its source observation stamp.");
    }
  }
  if (!metadata->direct_support) {
    return true;
  }
  metadata->source_law_id = object.value("source_law_id").toString().toStdString();
  metadata->query_id = object.value("query_id").toString().toStdString();
  metadata->integration_method = object.value("integration_method").toString().toStdString();
  const QJsonValue order_value = object.value("order");
  const bool finite_sum = object.value("integration_method").toString() == "finite_sum";
  const bool order_not_applicable = finite_sum && order_value.isNull();
  double order = 0.0;
  if (metadata->source_law_id.empty() || metadata->query_id.empty() ||
      !finiteJsonNumber(object.value("source_revision"), &metadata->source_revision) ||
      metadata->source_revision < 0.0 ||
      std::floor(metadata->source_revision) != metadata->source_revision ||
      (metadata->deterministic_quadrature && !order_not_applicable &&
       (!finiteJsonNumber(order_value, &order) || order < 1.0 ||
        order > std::numeric_limits<int>::max() || std::floor(order) != order))) {
    return displayFailure(error, "Joint support query identity/revision/order is invalid.");
  }
  metadata->order = static_cast<int>(order);
  if (metadata->result_kind == finite_kind && metadata->integration_method != "finite_sum") {
    return displayFailure(error, "Finite input-law query must declare finite_sum.");
  }
  if (metadata->result_kind == sample_kind) {
    double count = 0.0;
    if (metadata->integration_method != "monte_carlo" ||
        !finiteJsonNumber(object.value("sample_count"), &count) || count < 1.0 ||
        count > std::numeric_limits<int>::max() || std::floor(count) != count ||
        count != record.components.size() ||
        !finiteJsonNumber(object.value("seed"), &metadata->seed) ||
        metadata->seed < 0.0 || std::floor(metadata->seed) != metadata->seed) {
      return displayFailure(error, "Joint samples require method/count/seed metadata matching their support.");
    }
    metadata->sample_count = static_cast<int>(count);
  }
  const auto& dependencies = record.provenance.derived_from_edge_ids;
  if (std::find(dependencies.begin(), dependencies.end(), metadata->source_law_id) ==
      dependencies.end()) {
    return displayFailure(error, "Joint support source law is not a declared dependency.");
  }
  return true;
}

inline bool pathDisplayMetadata(
    const std::vector<const probtf_msgs::ProbabilisticTransformStamped*>& path,
    RecordDisplayMetadata* metadata, std::string* error = nullptr) {
  *metadata = RecordDisplayMetadata();
  bool has_dynamic = false;
  bool all_dynamic_frozen = true;
  for (const auto* record : path) {
    RecordDisplayMetadata current;
    if (record == nullptr || !recordDisplayMetadata(*record, &current, error)) {
      return displayFailure(error, "Path contains invalid display-query metadata.");
    }
    if (current.direct_support && path.size() != 1U) {
      return displayFailure(error,
          "UNAVAILABLE: support path requires a resolved joint pushforward; independent support composition is forbidden.");
    }
    if (path.size() == 1U) {
      *metadata = current;
    }
    if (!record->is_static) {
      has_dynamic = true;
      all_dynamic_frozen = all_dynamic_frozen && current.frozen_query;
    }
  }
  metadata->frozen_query = has_dynamic && all_dynamic_frozen;
  return true;
}

inline QString supportDisplayDescription(const RecordDisplayMetadata& metadata,
                                         std::size_t node_count) {
  if (metadata.result_kind == "joint_transform_samples") {
    return QString("%1 joint Monte Carlo samples, once each; method=%2, seed=%3, source=%4 v%5; alpha is relative weight.")
        .arg(node_count).arg(QString::fromStdString(metadata.integration_method))
        .arg(metadata.seed, 0, 'f', 0).arg(QString::fromStdString(metadata.source_law_id))
        .arg(metadata.source_revision, 0, 'f', 0);
  }
  if (metadata.result_kind == "joint_finite_pushforward") {
    return QString("%1 exact finite atoms, once each; source=%2 v%3; alpha is relative weight.")
        .arg(node_count).arg(QString::fromStdString(metadata.source_law_id))
        .arg(metadata.source_revision, 0, 'f', 0);
  }
  return QString("%1 quadrature nodes, once each; alpha is relative weight.").arg(node_count);
}

struct QuadratureEndpoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  int axis = 0;
  double normalized_weight = 0.0;
  float relative_alpha = 0.0F;
};

using QuadratureEndpoints =
    std::vector<QuadratureEndpoint, Eigen::aligned_allocator<QuadratureEndpoint>>;

inline bool quadratureEndpoints(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    double axis_length, std::size_t maximum_points, QuadratureEndpoints* output,
    std::string* error = nullptr) {
  RecordDisplayMetadata metadata;
  if (!recordDisplayMetadata(record, &metadata, error) ||
      !metadata.direct_support) {
    return displayFailure(error, "Record is not a valid resolved joint support query.");
  }
  if (output == nullptr || !std::isfinite(axis_length) || axis_length <= 0.0) {
    return displayFailure(error, "Quadrature display arguments are invalid.");
  }
  double maximum_weight = 0.0;
  std::size_t positive_count = 0;
  for (const auto& component : record.components) {
    if (!std::isfinite(component.weight) || component.weight < 0.0) {
      return displayFailure(error, "Quadrature weights must be finite and nonnegative.");
    }
    if (component.weight > 0.0) {
      maximum_weight = std::max(maximum_weight, component.weight);
      ++positive_count;
    }
  }
  if (positive_count == 0U || positive_count > maximum_points / 3U) {
    return displayFailure(error,
        "Quadrature support is empty or exceeds the display point limit; no nodes were resampled.");
  }
  double scaled_mass = 0.0;
  for (const auto& component : record.components) {
    scaled_mass += component.weight / maximum_weight;
  }
  QuadratureEndpoints points;
  points.reserve(3U * positive_count);
  for (const auto& component : record.components) {
    if (component.weight == 0.0) {
      continue;
    }
    if (component.orientation.kind != component.orientation.DIRAC) {
      return displayFailure(error, "Quadrature display nodes must have Dirac rotations.");
    }
    for (const double covariance : component.translation.residual_covariance_upper) {
      if (covariance != 0.0) {
        return displayFailure(error, "Quadrature display nodes must have zero residual covariance.");
      }
    }
    for (const double coupling : component.translation.rotation_coupling) {
      if (!std::isfinite(coupling)) {
        return displayFailure(error, "Quadrature node coupling is non-finite.");
      }
    }
    const auto& reference = component.orientation.reference_quaternion;
    Eigen::Quaterniond rotation(reference.w, reference.x, reference.y, reference.z);
    const auto& mean = component.translation.mean_at_reference;
    const Eigen::Vector3d translation(mean.x, mean.y, mean.z);
    if (!rotation.coeffs().allFinite() ||
        std::abs(rotation.norm() - 1.0) > 1.0e-8 || !translation.allFinite()) {
      return displayFailure(error, "Quadrature node pose is non-finite or non-unit.");
    }
    rotation.normalize();
    const double relative_weight = component.weight / maximum_weight;
    for (int axis = 0; axis < 3; ++axis) {
      QuadratureEndpoint point;
      point.position = translation + rotation * (axis_length * Eigen::Vector3d::Unit(axis));
      point.axis = axis;
      point.normalized_weight = relative_weight / scaled_mass;
      point.relative_alpha = static_cast<float>(relative_weight);
      points.push_back(point);
    }
  }
  *output = std::move(points);
  return true;
}

}  // namespace probtf_rviz
