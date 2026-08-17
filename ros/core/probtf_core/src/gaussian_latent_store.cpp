#include <probtf_core/gaussian_latent_store.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace probtf_core {
namespace {

void setError(std::string* output, const std::string& message) {
  if (output != nullptr) {
    *output = message;
  }
}

std::string cleanIdentifier(const std::string& value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return std::string();
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::pair<std::string, std::string> crossKey(const std::string& left,
                                              const std::string& right) {
  return left < right ? std::make_pair(left, right)
                      : std::make_pair(right, left);
}

bool normalizedPsd(Eigen::MatrixXd* value,
                   bool positive_definite,
                   const std::string& name,
                   std::string* error) {
  if (value->rows() < 1 || value->rows() != value->cols() ||
      !value->allFinite()) {
    setError(error, name + " must be finite, square, and non-empty.");
    return false;
  }
  *value = 0.5 * (*value + value->transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(*value);
  if (solver.info() != Eigen::Success) {
    setError(error, name + " eigendecomposition failed.");
    return false;
  }
  const double scale =
      std::max(1.0, value->cwiseAbs().rowwise().sum().maxCoeff());
  const double minimum = solver.eigenvalues()(0);
  if (positive_definite) {
    if (minimum <= std::numeric_limits<double>::epsilon() * scale) {
      setError(error, name + " must be positive definite.");
      return false;
    }
  } else if (minimum < -1.0e-10 * scale) {
    setError(error, name + " must be positive semidefinite.");
    return false;
  }
  *value = solver.eigenvectors() *
           solver.eigenvalues().cwiseMax(0.0).asDiagonal() *
           solver.eigenvectors().transpose();
  *value = 0.5 * (*value + value->transpose());
  return true;
}

bool validPose(const Eigen::Isometry3d& pose) {
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = pose.rotation();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .cwiseAbs()
                 .maxCoeff() <= 1.0e-8 &&
         std::abs(rotation.determinant() - 1.0) <= 1.0e-8;
}

template <class Value>
void appendUnique(std::vector<Value>* output, const std::vector<Value>& input) {
  for (const Value& value : input) {
    if (std::find(output->begin(), output->end(), value) == output->end()) {
      output->push_back(value);
    }
  }
}

}  // namespace

const GaussianLatentFactor* GaussianLatentSnapshot::factor(
    const std::string& factor_id) const {
  const auto iterator = factors.find(factor_id);
  return iterator == factors.end() ? nullptr : &iterator->second;
}

const std::vector<EdgeLatentBinding>*
GaussianLatentSnapshot::bindingsForEdge(const std::string& edge_id) const {
  const auto iterator = bindings_by_edge.find(edge_id);
  return iterator == bindings_by_edge.end() ? nullptr : &iterator->second;
}

bool GaussianLatentSnapshot::jointMeanCovariance(
    const std::vector<std::string>& factor_ids,
    Eigen::VectorXd* mean,
    Eigen::MatrixXd* covariance,
    std::map<std::string, std::pair<int, int>>* offsets,
    std::string* error) const {
  if (mean == nullptr || covariance == nullptr || offsets == nullptr) {
    setError(error, "Joint-distribution outputs must not be null.");
    return false;
  }
  std::set<std::string> unique;
  int dimension = 0;
  offsets->clear();
  for (const std::string& factor_id : factor_ids) {
    if (!unique.insert(factor_id).second) {
      setError(error, "Joint factor IDs must be unique.");
      return false;
    }
    const GaussianLatentFactor* selected = factor(factor_id);
    if (selected == nullptr) {
      setError(error, "Joint distribution references an unknown factor.");
      return false;
    }
    (*offsets)[factor_id] =
        std::make_pair(dimension, static_cast<int>(selected->mean.size()));
    dimension += static_cast<int>(selected->mean.size());
  }
  mean->setZero(dimension);
  covariance->setZero(dimension, dimension);
  for (const std::string& factor_id : factor_ids) {
    const GaussianLatentFactor& selected = factors.at(factor_id);
    const auto offset = offsets->at(factor_id);
    mean->segment(offset.first, offset.second) = selected.mean;
    covariance->block(offset.first, offset.first, offset.second, offset.second) =
        selected.covariance;
  }
  for (std::size_t left_index = 0; left_index < factor_ids.size();
       ++left_index) {
    const std::string& left = factor_ids[left_index];
    const auto left_offset = offsets->at(left);
    for (std::size_t right_index = left_index + 1;
         right_index < factor_ids.size(); ++right_index) {
      const std::string& right = factor_ids[right_index];
      const auto key = crossKey(left, right);
      const auto iterator = cross_covariances.find(key);
      if (iterator == cross_covariances.end()) {
        continue;
      }
      const Eigen::MatrixXd block =
          key.first == left ? iterator->second : iterator->second.transpose();
      const auto right_offset = offsets->at(right);
      covariance->block(left_offset.first, right_offset.first,
                        left_offset.second, right_offset.second) = block;
      covariance->block(right_offset.first, left_offset.first,
                        right_offset.second, left_offset.second) =
          block.transpose();
    }
  }
  *covariance = 0.5 * (*covariance + covariance->transpose());
  return mean->allFinite() && covariance->allFinite();
}

std::uint64_t GaussianLatentStore::revision() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return revision_;
}

GaussianLatentSnapshot GaussianLatentStore::snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  GaussianLatentSnapshot output;
  output.revision = revision_;
  output.factors = factors_;
  output.bindings_by_edge = bindings_by_edge_;
  output.cross_covariances = cross_covariances_;
  return output;
}

void GaussianLatentStore::refreshBindingVersionLocked(
    const std::string& factor_id,
    std::uint64_t version) {
  for (auto& edge_bindings : bindings_by_edge_) {
    for (EdgeLatentBinding& binding : edge_bindings.second) {
      if (binding.factor_id == factor_id) {
        binding.factor_version = version;
      }
    }
  }
}

void GaussianLatentStore::dropCrossCovariancesLocked(
    const std::string& factor_id) {
  for (auto iterator = cross_covariances_.begin();
       iterator != cross_covariances_.end();) {
    if (iterator->first.first == factor_id ||
        iterator->first.second == factor_id) {
      iterator = cross_covariances_.erase(iterator);
    } else {
      ++iterator;
    }
  }
}

bool GaussianLatentStore::putFactor(
    const std::string& input_factor_id,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& input_covariance,
    const ros::Time& stamp,
    GaussianLatentFactor* output,
    std::string* error,
    const LatentProvenance& provenance) {
  const std::string factor_id = cleanIdentifier(input_factor_id);
  if (factor_id.empty() || mean.size() < 1 || !mean.allFinite() ||
      input_covariance.rows() != mean.size() ||
      input_covariance.cols() != mean.size()) {
    setError(error, "A factor requires an ID, finite mean, and matching covariance.");
    return false;
  }
  Eigen::MatrixXd covariance = input_covariance;
  if (!normalizedPsd(&covariance, false, "Factor covariance", error)) {
    return false;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  for (const auto& edge_bindings : bindings_by_edge_) {
    for (const EdgeLatentBinding& binding : edge_bindings.second) {
      if (binding.factor_id == factor_id &&
          binding.sensitivity.cols() != mean.size()) {
        setError(error, "Cannot change the dimension of a bound factor.");
        return false;
      }
    }
  }
  const auto old = factors_.find(factor_id);
  GaussianLatentFactor factor;
  factor.factor_id = factor_id;
  factor.mean = mean;
  factor.covariance = covariance;
  factor.stamp = stamp;
  factor.version = old == factors_.end() ? 1 : old->second.version + 1;
  factor.provenance = provenance;
  factors_[factor_id] = factor;
  dropCrossCovariancesLocked(factor_id);
  refreshBindingVersionLocked(factor_id, factor.version);
  ++revision_;
  if (output != nullptr) {
    *output = factor;
  }
  return true;
}

bool GaussianLatentStore::bindEdge(const EdgeLatentBinding& input,
                                   std::string* error) {
  EdgeLatentBinding binding = input;
  binding.edge_id = cleanIdentifier(binding.edge_id);
  binding.factor_id = cleanIdentifier(binding.factor_id);
  if (binding.edge_id.empty() || binding.factor_id.empty() ||
      binding.sensitivity.rows() != 6 ||
      binding.sensitivity.cols() < 1 || !binding.sensitivity.allFinite() ||
      binding.factor_version < 1 ||
      binding.perturbation_convention != kPosePerturbationConvention ||
      !validPose(binding.linearization_pose)) {
    setError(error, "Edge latent binding is invalid.");
    return false;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  const auto selected = factors_.find(binding.factor_id);
  if (selected == factors_.end()) {
    setError(error, "Binding references an unknown factor.");
    return false;
  }
  if (binding.factor_version != selected->second.version) {
    setError(error, "Binding factor version is stale.");
    return false;
  }
  if (binding.sensitivity.cols() != selected->second.mean.size()) {
    setError(error, "Binding and factor dimensions do not match.");
    return false;
  }
  std::vector<EdgeLatentBinding>& existing =
      bindings_by_edge_[binding.edge_id];
  existing.erase(
      std::remove_if(existing.begin(), existing.end(),
                     [&binding](const EdgeLatentBinding& value) {
                       return value.factor_id == binding.factor_id;
                     }),
      existing.end());
  existing.push_back(binding);
  ++revision_;
  return true;
}

bool GaussianLatentStore::applyObservation(
    const GaussianObservationFactor& observation,
    GaussianUpdateResult* output,
    std::string* error,
    const std::map<std::string, std::uint64_t>& expected_versions) {
  if (cleanIdentifier(observation.observation_id).empty() ||
      observation.latent_factor_ids.empty() ||
      observation.residual.size() < 1 ||
      !observation.residual.allFinite() ||
      observation.jacobian_blocks.size() !=
          observation.latent_factor_ids.size() ||
      observation.noise_covariance.rows() != observation.residual.size() ||
      observation.noise_covariance.cols() != observation.residual.size()) {
    setError(error, "Gaussian observation factor is invalid.");
    return false;
  }
  Eigen::MatrixXd noise = observation.noise_covariance;
  if (!normalizedPsd(&noise, true, "Observation noise covariance", error)) {
    return false;
  }

  std::lock_guard<std::mutex> guard(mutex_);
  std::set<std::string> requested;
  for (std::size_t index = 0; index < observation.latent_factor_ids.size();
       ++index) {
    const std::string factor_id =
        cleanIdentifier(observation.latent_factor_ids[index]);
    const auto factor_iterator = factors_.find(factor_id);
    if (factor_id.empty() || factor_iterator == factors_.end() ||
        !requested.insert(factor_id).second) {
      setError(error, "Observation factor IDs must be unique and registered.");
      return false;
    }
    const Eigen::MatrixXd& block = observation.jacobian_blocks[index];
    if (block.rows() != observation.residual.size() ||
        block.cols() != factor_iterator->second.mean.size() ||
        !block.allFinite()) {
      setError(error, "Observation Jacobian block has incompatible dimensions.");
      return false;
    }
  }
  for (const auto& expected : expected_versions) {
    const auto selected = factors_.find(expected.first);
    if (selected == factors_.end() ||
        selected->second.version != expected.second) {
      setError(error, "Factor version changed before observation update.");
      return false;
    }
  }

  std::set<std::string> closure = requested;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& cross : cross_covariances_) {
      if (closure.count(cross.first.first) != 0U ||
          closure.count(cross.first.second) != 0U) {
        const std::size_t before = closure.size();
        closure.insert(cross.first.first);
        closure.insert(cross.first.second);
        changed = changed || closure.size() != before;
      }
    }
  }
  std::vector<std::string> order;
  for (const std::string& factor_id : observation.latent_factor_ids) {
    order.push_back(cleanIdentifier(factor_id));
  }
  for (const std::string& factor_id : closure) {
    if (requested.count(factor_id) == 0U) {
      order.push_back(factor_id);
    }
  }

  GaussianLatentSnapshot prior;
  prior.revision = revision_;
  prior.factors = factors_;
  prior.bindings_by_edge = bindings_by_edge_;
  prior.cross_covariances = cross_covariances_;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  std::map<std::string, std::pair<int, int>> offsets;
  if (!prior.jointMeanCovariance(order, &mean, &covariance, &offsets, error)) {
    return false;
  }
  Eigen::MatrixXd jacobian =
      Eigen::MatrixXd::Zero(observation.residual.size(), mean.size());
  for (std::size_t index = 0; index < observation.latent_factor_ids.size();
       ++index) {
    const std::string factor_id =
        cleanIdentifier(observation.latent_factor_ids[index]);
    const auto offset = offsets.at(factor_id);
    jacobian.block(0, offset.first, jacobian.rows(), offset.second) =
        observation.jacobian_blocks[index];
  }
  Eigen::MatrixXd innovation =
      jacobian * covariance * jacobian.transpose() + noise;
  innovation = 0.5 * (innovation + innovation.transpose());
  Eigen::LDLT<Eigen::MatrixXd> decomposition(innovation);
  if (decomposition.info() != Eigen::Success ||
      (decomposition.vectorD().array() <= 0.0).any()) {
    setError(error, "Observation innovation covariance is not positive definite.");
    return false;
  }
  const Eigen::MatrixXd gain =
      decomposition.solve(jacobian * covariance).transpose();
  const Eigen::VectorXd posterior_mean =
      mean - gain * observation.residual;
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(mean.size(), mean.size());
  const Eigen::MatrixXd joseph = identity - gain * jacobian;
  Eigen::MatrixXd posterior_covariance =
      joseph * covariance * joseph.transpose() +
      gain * noise * gain.transpose();
  if (!normalizedPsd(&posterior_covariance, false, "Posterior covariance",
                     error)) {
    return false;
  }

  GaussianUpdateResult result;
  result.observation_id = cleanIdentifier(observation.observation_id);
  result.innovation_covariance = innovation;
  result.kalman_gain = gain;
  for (const std::string& factor_id : order) {
    GaussianLatentFactor factor = factors_.at(factor_id);
    result.prior_versions.emplace_back(factor_id, factor.version);
    const auto offset = offsets.at(factor_id);
    factor.mean = posterior_mean.segment(offset.first, offset.second);
    factor.covariance = posterior_covariance.block(
        offset.first, offset.first, offset.second, offset.second);
    factor.stamp = observation.stamp;
    ++factor.version;
    appendUnique(&factor.provenance.source_ids,
                 observation.provenance.source_ids);
    appendUnique(&factor.provenance.derived_from_edge_ids,
                 observation.provenance.derived_from_edge_ids);
    factor.provenance.method = "gaussian_observation_update";
    factor.provenance.detail = result.observation_id;
    factors_[factor_id] = factor;
    result.posterior_versions.emplace_back(factor_id, factor.version);
  }
  for (auto iterator = cross_covariances_.begin();
       iterator != cross_covariances_.end();) {
    if (closure.count(iterator->first.first) != 0U &&
        closure.count(iterator->first.second) != 0U) {
      iterator = cross_covariances_.erase(iterator);
    } else {
      ++iterator;
    }
  }
  for (std::size_t left_index = 0; left_index < order.size(); ++left_index) {
    const std::string& left = order[left_index];
    const auto left_offset = offsets.at(left);
    for (std::size_t right_index = left_index + 1;
         right_index < order.size(); ++right_index) {
      const std::string& right = order[right_index];
      const auto right_offset = offsets.at(right);
      Eigen::MatrixXd block = posterior_covariance.block(
          left_offset.first, right_offset.first,
          left_offset.second, right_offset.second);
      const auto key = crossKey(left, right);
      cross_covariances_[key] = key.first == left ? block : block.transpose();
    }
  }
  for (const std::string& factor_id : order) {
    refreshBindingVersionLocked(factor_id, factors_.at(factor_id).version);
  }
  ++revision_;
  result.store_revision = revision_;
  if (output != nullptr) {
    *output = result;
  }
  return true;
}

}  // namespace probtf_core
