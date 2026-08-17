#include <Eigen/Geometry>

#include <probtf_core/gaussian_latent_store.hpp>
#include <probtf_core/latest_snapshot.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

Eigen::Isometry3d edgePose(int index) {
  Eigen::Isometry3d output = Eigen::Isometry3d::Identity();
  output.translation() =
      Eigen::Vector3d(0.08, 0.002 * static_cast<double>(index), 0.0);
  return output;
}

probtf_msgs::ProbabilisticTransformStamped record(
    const std::string& parent,
    const std::string& child,
    int index,
    const std::vector<std::string>& factor_ids) {
  const Eigen::Isometry3d pose = edgePose(index);
  geometry_msgs::TransformStamped transform;
  transform.header.frame_id = parent;
  transform.header.stamp = ros::Time(1.0);
  transform.child_frame_id = child;
  transform.transform.translation.x = pose.translation().x();
  transform.transform.translation.y = pose.translation().y();
  transform.transform.translation.z = pose.translation().z();
  transform.transform.rotation.w = 1.0;
  probtf_msgs::ProbabilisticTransformStamped output;
  std::string error;
  if (!probtf_core::deterministicTfToProbTf(
          transform, "benchmark", false, &output, &error)) {
    throw std::runtime_error(error);
  }
  output.provenance.derived_from_edge_ids = factor_ids;
  if (index == 0) {
    auto& orientation = output.components.front().orientation;
    orientation.kind =
        probtf_msgs::BinghamOrientation::FINITE_BINGHAM;
    orientation.inverse_concentration = 0.002;
    orientation.shape_upper_wxyz =
        {{0.75, 0.0, 0.0, 0.0, 0.25,
          0.0, 0.0, -0.25, 0.0, -0.75}};
  }
  return output;
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const double selected = fraction * static_cast<double>(sorted.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(selected);
  const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double alpha = selected - static_cast<double>(lower);
  return (1.0 - alpha) * sorted[lower] + alpha * sorted[upper];
}

void runCase(int path_length, int factor_count, int factor_dimension) {
  std::vector<std::string> factor_ids;
  for (int index = 0; index < factor_count; ++index) {
    factor_ids.push_back("factor_" + std::to_string(index));
  }
  auto store = std::make_shared<probtf_core::GaussianLatentStore>();
  std::vector<probtf_core::GaussianLatentFactor> factors;
  std::string error;
  for (const std::string& factor_id : factor_ids) {
    probtf_core::GaussianLatentFactor factor;
    if (!store->putFactor(
            factor_id, Eigen::VectorXd::Zero(factor_dimension),
            1.0e-6 * Eigen::MatrixXd::Identity(
                           factor_dimension, factor_dimension),
            ros::Time(1.0), &factor, &error)) {
      throw std::runtime_error(error);
    }
    factors.push_back(factor);
  }

  probtf_msgs::ProbabilisticTransformArray dynamic_records;
  std::string parent = "world";
  for (int edge_index = 0; edge_index < path_length; ++edge_index) {
    const std::string child = "frame_" + std::to_string(edge_index);
    auto edge = record(parent, child, edge_index, factor_ids);
    dynamic_records.transforms.push_back(edge);
    for (int factor_index = 0; factor_index < factor_count; ++factor_index) {
      Eigen::MatrixXd sensitivity =
          Eigen::MatrixXd::Zero(6, factor_dimension);
      for (int column = 0; column < factor_dimension; ++column) {
        sensitivity((edge_index + column) % 6, column) =
            0.01 * static_cast<double>(1 + ((edge_index + column) % 5));
      }
      probtf_core::EdgeLatentBinding binding;
      binding.edge_id = edge.edge_id;
      binding.factor_id = factors[factor_index].factor_id;
      binding.sensitivity = sensitivity;
      binding.factor_version = factors[factor_index].version;
      binding.linearization_stamp = edge.header.stamp;
      binding.linearization_pose = edgePose(edge_index);
      if (!store->bindEdge(binding, &error)) {
        throw std::runtime_error(error);
      }
    }
    parent = child;
  }
  probtf_msgs::ProbabilisticTransformArray static_records;
  probtf_core::TransformMomentObservation output;
  for (int index = 0; index < 10; ++index) {
    probtf_core::LatestSnapshot uncached_runtime(
        dynamic_records, static_records, store);
    if (!uncached_runtime.lookupTransformMoments(
            "world", parent, &output, &error)) {
      throw std::runtime_error(error);
    }
  }
  std::vector<double> uncached_timings;
  uncached_timings.reserve(100);
  for (int index = 0; index < 100; ++index) {
    probtf_core::LatestSnapshot uncached_runtime(
        dynamic_records, static_records, store);
    const auto start = std::chrono::steady_clock::now();
    if (!uncached_runtime.lookupTransformMoments(
            "world", parent, &output, &error)) {
      throw std::runtime_error(error);
    }
    const auto end = std::chrono::steady_clock::now();
    uncached_timings.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  std::sort(uncached_timings.begin(), uncached_timings.end());

  probtf_core::LatestSnapshot runtime(dynamic_records, static_records, store);
  for (int index = 0; index < 10; ++index) {
    if (!runtime.lookupTransformMoments("world", parent, &output, &error)) {
      throw std::runtime_error(error);
    }
  }
  std::vector<double> cached_timings;
  cached_timings.reserve(100);
  for (int index = 0; index < 100; ++index) {
    const auto start = std::chrono::steady_clock::now();
    if (!runtime.lookupTransformMoments("world", parent, &output, &error)) {
      throw std::runtime_error(error);
    }
    const auto end = std::chrono::steady_clock::now();
    cached_timings.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  std::sort(cached_timings.begin(), cached_timings.end());
  std::cout << "{\"path_length\":" << path_length
            << ",\"factor_count\":" << factor_count
            << ",\"factor_dimension\":" << factor_dimension
            << ",\"global_edge\":\"concentrated_finite_bingham\""
            << ",\"queries\":100"
            << ",\"uncached_p50_us\":"
            << percentile(uncached_timings, 0.50)
            << ",\"uncached_p95_us\":"
            << percentile(uncached_timings, 0.95)
            << ",\"cached_p50_us\":"
            << percentile(cached_timings, 0.50)
            << ",\"cached_p95_us\":"
            << percentile(cached_timings, 0.95)
            << "}" << std::endl;
}

}  // namespace

int main() {
  try {
    for (const int path_length : {4, 8, 16, 32}) {
      runCase(path_length, 0, 0);
      for (const int factor_count : {1, 4}) {
        for (const int dimension : {6, 12, 24, 48}) {
          runCase(path_length, factor_count, dimension);
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  return 0;
}
