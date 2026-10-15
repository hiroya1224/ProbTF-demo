#include <probtf_rviz/point_moment_query.hpp>

#include <Eigen/Eigenvalues>

#include <QJsonDocument>
#include <QJsonObject>

#include <cmath>
#include <limits>
#include <utility>

namespace probtf_rviz {
namespace {

bool fail(const char* message, std::string* error) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

std::string cleanFrame(const std::string& value) {
  const auto first = value.find_first_not_of("/ \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  return value.substr(first, value.find_last_not_of("/ \t\r\n") - first + 1U);
}

}  // namespace

bool pointMomentIntegrationSteps(const std::string& detail, int* steps,
                                std::string* error) {
  if (steps == nullptr) {
    return fail("Point-moment integration steps require an output.", error);
  }
  *steps = 480;
  if (detail.empty()) {
    return true;
  }
  const auto document = QJsonDocument::fromJson(QByteArray::fromStdString(detail));
  if (!document.isObject()) {
    return fail("Point-moment source context must be a JSON object.", error);
  }
  const auto value = document.object().value("moment_integration_steps");
  if (value.isUndefined()) {
    return true;
  }
  const double count = value.toDouble(-1.0);
  if (!value.isDouble() || !std::isfinite(count) || count < 1.0 ||
      count > std::numeric_limits<int>::max() || count != std::floor(count)) {
    return fail("Point-moment integration steps must be a positive integer.", error);
  }
  *steps = static_cast<int>(count);
  return true;
}

bool sameNativeLocalLawContext(
    const std::vector<const probtf_msgs::ProbabilisticTransformStamped*>& path,
    std::string* error) {
  if (path.empty() || path.front() == nullptr) {
    return fail("Native local-law point query requires a nonempty path.", error);
  }
  const auto* context = path.front();
  for (std::size_t index = 0; index < path.size(); ++index) {
    const auto* edge = path[index];
    if (edge == nullptr || edge->provenance.method != "configured_native_local_transform_family" ||
        edge->provenance.detail.empty() || edge->provenance.source_ids.empty() ||
        edge->authority.empty() || edge->provenance.detail != context->provenance.detail ||
        edge->provenance.source_ids != context->provenance.source_ids ||
        edge->authority != context->authority || edge->header.stamp != context->header.stamp) {
      return fail("Native local-law point query requires matching family, source context, authority and stamp.", error);
    }
    if (index > 0U && cleanFrame(path[index - 1U]->header.frame_id) !=
                          cleanFrame(edge->child_frame_id)) {
      return fail("Native local-law point query requires adjacent child-to-root frames.", error);
    }
  }
  return true;
}

bool queryAxisPointMoments(const probtf_core::LatestSnapshot& snapshot,
                          const std::string& root, const std::string& child,
                          double axis_length, int integration_steps,
                          AxisPointMoments* output, std::string* error) {
  if (output == nullptr || !std::isfinite(axis_length) || axis_length < 0.0 ||
      integration_steps < 1) {
    return fail("Axis point query requires output, finite nonnegative length and positive integration steps.", error);
  }
  AxisPointMoments result;
  for (int axis = 0; axis < 3; ++axis) {
    probtf_core::PointMomentObservation observation;
    if (!snapshot.lookupPointMoments(root, child,
            axis_length * Eigen::Vector3d::Unit(axis), &observation, error, integration_steps)) {
      return false;
    }
    result[axis] = observation.moments;
  }
  *output = std::move(result);
  return true;
}

bool sampleAxisPointGaussians(const AxisPointMoments& moments, std::size_t count,
                             std::mt19937* generator,
                             std::vector<ColoredPoint>* output,
                             std::string* error) {
  if (generator == nullptr || output == nullptr ||
      count > output->max_size() / 3U) {
    return fail("Axis point Gaussian sampler requires generator, output and valid count.", error);
  }
  std::vector<ColoredPoint> result;
  result.reserve(3U * count);
  for (int axis = 0; axis < 3; ++axis) {
    const auto& point_moments = moments[axis];
    const Eigen::Matrix3d covariance = 0.5 * (point_moments.covariance + point_moments.covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
        covariance.cwiseAbs().maxCoeff();
    if (!point_moments.mean.allFinite() || !covariance.allFinite() ||
        solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -tolerance) {
      return fail("Point-moment Gaussian query has invalid mean or covariance.", error);
    }
    const Eigen::Matrix3d square_root = solver.eigenvectors() *
        solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
    std::normal_distribution<double> normal(0.0, 1.0);
    for (std::size_t index = 0; index < count; ++index) {
      Eigen::Vector3d gaussian;
      for (int dimension = 0; dimension < 3; ++dimension) {
        gaussian(dimension) = normal(*generator);
      }
      ColoredPoint point;
      point.position = point_moments.mean + square_root * gaussian;
      point.color = axisColor(axis);
      result.push_back(point);
    }
  }
  *output = std::move(result);
  return true;
}

}  // namespace probtf_rviz
