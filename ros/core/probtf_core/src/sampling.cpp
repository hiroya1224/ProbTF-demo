#include <probtf_core/sampling.hpp>

#include <Eigen/Eigenvalues>

#include <probtf_msgs/BinghamOrientation.h>
#include <probtf_msgs/ProbabilisticTransformComponent.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace probtf_core {
namespace {

using Matrix39d = Eigen::Matrix<double, 3, 9>;
using Matrix4d = Eigen::Matrix4d;
using Vector4d = Eigen::Vector4d;
using Vector9d = Eigen::Matrix<double, 9, 1>;

constexpr double kPsdTolerance = 1.0e-10;
constexpr double kQuaternionTolerance = 1.0e-8;
constexpr double kShapeTolerance = 1.0e-8;
constexpr std::size_t kMaxOrientationProposalAttempts = 4096U;

void setError(std::string* output, const std::string& value) {
  if (output != nullptr) {
    *output = value;
  }
}

Eigen::Vector3d vector3(const geometry_msgs::Vector3& value) {
  return Eigen::Vector3d(value.x, value.y, value.z);
}

bool normalizedQuaternion(const geometry_msgs::Quaternion& value,
                          Eigen::Quaterniond* output,
                          std::string* error) {
  Eigen::Quaterniond quaternion(value.w, value.x, value.y, value.z);
  const double norm = quaternion.norm();
  if (!quaternion.coeffs().allFinite() || !std::isfinite(norm) ||
      std::abs(norm - 1.0) > kQuaternionTolerance) {
    setError(error, "Quaternion must be finite and have unit norm.");
    return false;
  }
  quaternion.normalize();
  *output = quaternion;
  return true;
}

Matrix4d unpackSymmetric4(const boost::array<double, 10>& packed) {
  Matrix4d output = Matrix4d::Zero();
  std::size_t index = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = row; column < 4; ++column) {
      output(row, column) = packed[index++];
      output(column, row) = output(row, column);
    }
  }
  return output;
}

Eigen::Matrix3d unpackSymmetric3(const boost::array<double, 6>& packed) {
  Eigen::Matrix3d output = Eigen::Matrix3d::Zero();
  std::size_t index = 0;
  for (int row = 0; row < 3; ++row) {
    for (int column = row; column < 3; ++column) {
      output(row, column) = packed[index++];
      output(column, row) = output(row, column);
    }
  }
  return output;
}

Matrix39d couplingMatrix(const boost::array<double, 27>& packed) {
  Matrix39d output;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 9; ++column) {
      output(row, column) =
          packed[static_cast<std::size_t>(row * 9 + column)];
    }
  }
  return output;
}

Vector9d rotationVector(const Eigen::Matrix3d& rotation) {
  Vector9d output;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      output(row + 3 * column) = rotation(row, column);
    }
  }
  return output;
}

struct OrientationData {
  uint8_t kind = probtf_msgs::BinghamOrientation::DIRAC;
  Eigen::Quaterniond reference = Eigen::Quaterniond::Identity();
  Matrix4d shape = Matrix4d::Zero();
  Vector4d shape_eigenvalues = Vector4d::Zero();
  Matrix4d shape_eigenvectors = Matrix4d::Identity();
};

bool validatedOrientation(const probtf_msgs::BinghamOrientation& orientation,
                          OrientationData* output,
                          std::string* error) {
  OrientationData value;
  value.kind = orientation.kind;
  if (!normalizedQuaternion(orientation.reference_quaternion, &value.reference,
                            error)) {
    return false;
  }
  value.shape = unpackSymmetric4(orientation.shape_upper_wxyz);
  if (!value.shape.allFinite() ||
      std::abs(value.shape.trace()) > kShapeTolerance) {
    setError(error, "Bingham shape must be finite and trace-zero.");
    return false;
  }

  if (orientation.kind == probtf_msgs::BinghamOrientation::DIRAC) {
    if (orientation.inverse_concentration != 0.0) {
      setError(error, "Dirac orientation requires zero inverse concentration.");
      return false;
    }
    const Vector4d quaternion(value.reference.w(), value.reference.x(),
                              value.reference.y(), value.reference.z());
    const Matrix4d expected =
        2.0 * quaternion * quaternion.transpose() -
        0.5 * Matrix4d::Identity();
    if ((value.shape - expected).cwiseAbs().maxCoeff() > kShapeTolerance) {
      setError(error, "Dirac Bingham shape does not match its quaternion.");
      return false;
    }
    *output = value;
    return true;
  }

  if (orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    if (!std::isinf(orientation.inverse_concentration) ||
        orientation.inverse_concentration < 0.0 ||
        value.shape.cwiseAbs().maxCoeff() > kShapeTolerance) {
      setError(
          error,
          "Uniform orientation requires infinite inverse concentration and a "
          "zero shape.");
      return false;
    }
    *output = value;
    return true;
  }

  if (orientation.kind !=
      probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    setError(error, "Unknown Bingham orientation kind.");
    return false;
  }
  if (!std::isfinite(orientation.inverse_concentration) ||
      orientation.inverse_concentration <= 0.0) {
    setError(error,
             "Finite Bingham inverse concentration must be positive.");
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Matrix4d> solver(value.shape);
  if (solver.info() != Eigen::Success) {
    setError(error, "Finite Bingham eigendecomposition failed.");
    return false;
  }
  value.shape_eigenvalues = solver.eigenvalues();
  value.shape_eigenvectors = solver.eigenvectors();
  if (std::abs(value.shape_eigenvalues(3) +
                   value.shape_eigenvalues(2) -
               1.0) > kShapeTolerance) {
    setError(error, "Finite Bingham shape is not JMAA-normalized.");
    return false;
  }
  *output = value;
  return true;
}

struct TranslationData {
  Eigen::Vector3d mean_at_reference = Eigen::Vector3d::Zero();
  Matrix39d coupling = Matrix39d::Zero();
  Eigen::Matrix3d residual_sqrt = Eigen::Matrix3d::Zero();
  bool has_residual = false;
};

bool validatedTranslation(
    const probtf_msgs::ConditionalGaussianTranslation& translation,
    TranslationData* output,
    std::string* error) {
  TranslationData value;
  value.mean_at_reference = vector3(translation.mean_at_reference);
  value.coupling = couplingMatrix(translation.rotation_coupling);
  Eigen::Matrix3d covariance =
      unpackSymmetric3(translation.residual_covariance_upper);
  if (!value.mean_at_reference.allFinite() || !value.coupling.allFinite() ||
      !covariance.allFinite()) {
    setError(error,
             "Conditional translation must contain only finite values.");
    return false;
  }
  covariance = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success ||
      solver.eigenvalues().minCoeff() < -kPsdTolerance) {
    setError(error,
             "Translation residual covariance must be positive semidefinite.");
    return false;
  }
  const Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
  value.residual_sqrt =
      solver.eigenvectors() * eigenvalues.cwiseSqrt().asDiagonal() *
      solver.eigenvectors().transpose();
  value.has_residual = value.residual_sqrt.cwiseAbs().maxCoeff() > 0.0;
  *output = value;
  return true;
}

double binghamScaleEquation(const Vector4d& precision_eigenvalues,
                            double scale) {
  double value = -1.0;
  for (int index = 0; index < 4; ++index) {
    value += 1.0 / (scale + 2.0 * precision_eigenvalues(index));
  }
  return value;
}

bool binghamProposalScale(const Vector4d& precision_eigenvalues,
                          double* output,
                          std::string* error) {
  double lower = 1.0e-12;
  double upper = 4.0 + 1.0e-12;
  double lower_value = binghamScaleEquation(precision_eigenvalues, lower);
  double upper_value = binghamScaleEquation(precision_eigenvalues, upper);
  if (!std::isfinite(lower_value) || !std::isfinite(upper_value) ||
      lower_value <= 0.0 || upper_value >= 0.0) {
    setError(error, "Finite Bingham proposal scale could not be bracketed.");
    return false;
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    const double value =
        binghamScaleEquation(precision_eigenvalues, middle);
    if (!std::isfinite(value)) {
      setError(error, "Finite Bingham proposal scale became non-finite.");
      return false;
    }
    if (value > 0.0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  *output = 0.5 * (lower + upper);
  return true;
}

struct PreparedComponent {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OrientationData orientation;
  TranslationData translation;
  Vector4d precision_eigenvalues = Vector4d::Zero();
  Vector4d proposal_standard_deviations = Vector4d::Ones();
  double proposal_scale = 4.0;
  double log_envelope = 0.0;
  Vector9d reference_rotation_vector = Vector9d::Zero();
};

bool prepareComponent(
    const probtf_msgs::ProbabilisticTransformComponent& component,
    bool prepare_sampler,
    PreparedComponent* output,
    std::string* error) {
  PreparedComponent value;
  if (!validatedOrientation(component.orientation, &value.orientation,
                            error) ||
      !validatedTranslation(component.translation, &value.translation,
                            error)) {
    return false;
  }
  value.reference_rotation_vector =
      rotationVector(value.orientation.reference.toRotationMatrix());

  if (prepare_sampler &&
      value.orientation.kind ==
          probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    const double inverse = component.orientation.inverse_concentration;
    for (int index = 0; index < 4; ++index) {
      value.precision_eigenvalues(index) =
          (value.orientation.shape_eigenvalues(3) -
           value.orientation.shape_eigenvalues(index)) /
          inverse;
    }
    if (!value.precision_eigenvalues.allFinite() ||
        value.precision_eigenvalues.minCoeff() < -kPsdTolerance) {
      setError(error, "Finite Bingham precision is not finite positive "
                      "semidefinite.");
      return false;
    }
    value.precision_eigenvalues =
        value.precision_eigenvalues.cwiseMax(0.0);
    if (!binghamProposalScale(value.precision_eigenvalues,
                              &value.proposal_scale, error)) {
      return false;
    }
    for (int index = 0; index < 4; ++index) {
      const double denominator =
          1.0 + 2.0 * value.precision_eigenvalues(index) /
                    value.proposal_scale;
      if (!std::isfinite(denominator) || denominator <= 0.0) {
        setError(error, "Finite Bingham proposal covariance is invalid.");
        return false;
      }
      value.proposal_standard_deviations(index) =
          1.0 / std::sqrt(denominator);
    }
    value.log_envelope =
        -(4.0 - value.proposal_scale) / 2.0 +
        2.0 * std::log(4.0 / value.proposal_scale);
    if (!value.proposal_standard_deviations.allFinite() ||
        !std::isfinite(value.log_envelope)) {
      setError(error, "Finite Bingham proposal is non-finite.");
      return false;
    }
  }

  *output = value;
  return true;
}

bool positiveMixtureWeights(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    std::vector<std::size_t>* component_indices,
    std::vector<double>* scaled_weights,
    std::string* error) {
  component_indices->clear();
  scaled_weights->clear();
  double scale = 0.0;
  for (const auto& component : record.components) {
    if (!std::isfinite(component.weight)) {
      setError(error, "Prob-TF component weight is non-finite.");
      return false;
    }
    scale = std::max(scale, std::max(0.0, component.weight));
  }
  if (scale <= 0.0) {
    setError(error, "Prob-TF distribution has zero usable mixture mass.");
    return false;
  }
  for (std::size_t index = 0; index < record.components.size(); ++index) {
    const double weight = record.components[index].weight;
    if (weight > 0.0) {
      component_indices->push_back(index);
      scaled_weights->push_back(weight / scale);
    }
  }
  return true;
}

bool sampleUnitGaussianQuaternion(std::mt19937* generator,
                                  std::normal_distribution<double>* normal,
                                  Eigen::Quaterniond* output,
                                  std::string* error) {
  for (std::size_t attempt = 0; attempt < kMaxOrientationProposalAttempts;
       ++attempt) {
    Vector4d vector;
    for (int index = 0; index < 4; ++index) {
      vector(index) = (*normal)(*generator);
    }
    const double norm = vector.norm();
    if (std::isfinite(norm) && norm > 0.0) {
      vector /= norm;
      *output =
          Eigen::Quaterniond(vector(0), vector(1), vector(2), vector(3));
      return true;
    }
  }
  setError(error, "Uniform quaternion sampler exhausted its proposal budget.");
  return false;
}

bool sampleOrientation(const PreparedComponent& component,
                       std::mt19937* generator,
                       std::normal_distribution<double>* normal,
                       std::uniform_real_distribution<double>* uniform,
                       Eigen::Quaterniond* output,
                       std::string* error) {
  if (component.orientation.kind == probtf_msgs::BinghamOrientation::DIRAC) {
    *output = component.orientation.reference;
    return true;
  }
  if (component.orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    return sampleUnitGaussianQuaternion(generator, normal, output, error);
  }

  for (std::size_t attempt = 0; attempt < kMaxOrientationProposalAttempts;
       ++attempt) {
    Vector4d gaussian;
    for (int index = 0; index < 4; ++index) {
      gaussian(index) = (*normal)(*generator);
    }
    Vector4d coordinates =
        component.proposal_standard_deviations.array() * gaussian.array();
    const double norm = coordinates.norm();
    if (!std::isfinite(norm) || norm <= 0.0) {
      continue;
    }
    coordinates /= norm;
    const double quadratic =
        (component.precision_eigenvalues.array() *
         coordinates.array().square())
            .sum();
    const double proposal_form =
        1.0 + 2.0 * quadratic / component.proposal_scale;
    const double log_probability =
        -quadratic + 2.0 * std::log(proposal_form) -
        component.log_envelope;
    if (!std::isfinite(log_probability)) {
      setError(error, "Finite Bingham acceptance probability is non-finite.");
      return false;
    }
    const double draw = (*uniform)(*generator);
    if (log_probability >= 0.0 || draw == 0.0 ||
        std::log(draw) < log_probability) {
      const Vector4d candidate =
          component.orientation.shape_eigenvectors * coordinates;
      *output = Eigen::Quaterniond(candidate(0), candidate(1),
                                   candidate(2), candidate(3));
      output->normalize();
      return true;
    }
  }
  setError(error,
           "Finite Bingham sampler exhausted its proposal budget.");
  return false;
}

bool componentMode(const PreparedComponent& component,
                   Eigen::Quaterniond* output,
                   std::string* error) {
  if (component.orientation.kind == probtf_msgs::BinghamOrientation::DIRAC ||
      component.orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    *output = component.orientation.reference;
    return true;
  }
  if (component.orientation.kind !=
      probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    setError(error, "Unknown Bingham orientation kind.");
    return false;
  }
  Vector4d mode = component.orientation.shape_eigenvectors.col(3);
  Eigen::Index pivot = 0;
  mode.cwiseAbs().maxCoeff(&pivot);
  if (mode(pivot) < 0.0) {
    mode = -mode;
  }
  *output = Eigen::Quaterniond(mode(0), mode(1), mode(2), mode(3));
  output->normalize();
  return true;
}

Eigen::Vector3d conditionalTranslation(const PreparedComponent& component,
                                       const Eigen::Quaterniond& rotation) {
  return component.translation.mean_at_reference +
         component.translation.coupling *
             (rotationVector(rotation.toRotationMatrix()) -
              component.reference_rotation_vector);
}

bool storedRepresentative(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    Eigen::Isometry3d* output,
    std::string* error) {
  Eigen::Quaterniond rotation;
  if (!normalizedQuaternion(record.representative.rotation, &rotation, error)) {
    setError(error, "Stored representative quaternion is invalid.");
    return false;
  }
  const Eigen::Vector3d translation(
      record.representative.translation.x,
      record.representative.translation.y,
      record.representative.translation.z);
  if (!translation.allFinite()) {
    setError(error, "Stored representative translation is non-finite.");
    return false;
  }
  *output = Eigen::Isometry3d::Identity();
  output->linear() = rotation.toRotationMatrix();
  output->translation() = translation;
  return true;
}

}  // namespace

bool sampleTransformDistribution(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    std::size_t count,
    std::mt19937* generator,
    TransformSampleVector* output,
    std::string* error) {
  if (generator == nullptr) {
    setError(error, "Random generator must not be null.");
    return false;
  }
  if (output == nullptr) {
    setError(error, "Transform sample output must not be null.");
    return false;
  }

  std::vector<std::size_t> component_indices;
  std::vector<double> weights;
  if (!positiveMixtureWeights(record, &component_indices, &weights, error)) {
    return false;
  }

  std::vector<PreparedComponent, Eigen::aligned_allocator<PreparedComponent>>
      prepared;
  prepared.reserve(component_indices.size());
  for (std::size_t index = 0; index < record.components.size(); ++index) {
    PreparedComponent component;
    const bool usable = record.components[index].weight > 0.0;
    if (!prepareComponent(record.components[index], usable, &component,
                          error)) {
      return false;
    }
    if (usable) {
      prepared.push_back(std::move(component));
    }
  }

  std::discrete_distribution<std::size_t> mixture(weights.begin(),
                                                  weights.end());
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  TransformSampleVector samples;
  samples.reserve(count);
  for (std::size_t sample_index = 0; sample_index < count; ++sample_index) {
    const PreparedComponent& component = prepared[mixture(*generator)];
    TransformSample sample;
    if (!sampleOrientation(component, generator, &normal, &uniform,
                           &sample.rotation, error)) {
      return false;
    }
    sample.translation = conditionalTranslation(component, sample.rotation);
    if (component.translation.has_residual) {
      Eigen::Vector3d gaussian;
      for (int index = 0; index < 3; ++index) {
        gaussian(index) = normal(*generator);
      }
      sample.translation += component.translation.residual_sqrt * gaussian;
    }
    if (!sample.translation.allFinite()) {
      setError(error, "Sampled transform translation became non-finite.");
      return false;
    }
    samples.push_back(std::move(sample));
  }
  *output = std::move(samples);
  return true;
}

bool representativeTransform(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    Eigen::Isometry3d* output,
    std::string* error) {
  if (output == nullptr) {
    setError(error, "Representative transform output must not be null.");
    return false;
  }
  if (record.representative_kind >
      probtf_msgs::ProbabilisticTransformStamped::REPRESENTATIVE_MOMENT) {
    setError(error, "Unknown representative kind.");
    return false;
  }
  if (record.representative_kind !=
      probtf_msgs::ProbabilisticTransformStamped::REPRESENTATIVE_NONE) {
    return storedRepresentative(record, output, error);
  }

  std::vector<std::size_t> component_indices;
  std::vector<double> weights;
  if (!positiveMixtureWeights(record, &component_indices, &weights, error)) {
    return false;
  }
  std::size_t selected = component_indices.front();
  double selected_weight = record.components[selected].weight;
  for (const std::size_t index : component_indices) {
    if (record.components[index].weight > selected_weight) {
      selected = index;
      selected_weight = record.components[index].weight;
    }
  }

  PreparedComponent component;
  if (!prepareComponent(record.components[selected], false, &component,
                        error)) {
    return false;
  }
  Eigen::Quaterniond rotation;
  if (!componentMode(component, &rotation, error)) {
    return false;
  }
  const Eigen::Vector3d translation =
      conditionalTranslation(component, rotation);
  if (!translation.allFinite()) {
    setError(error, "Representative translation became non-finite.");
    return false;
  }
  *output = Eigen::Isometry3d::Identity();
  output->linear() = rotation.toRotationMatrix();
  output->translation() = translation;
  return true;
}

}  // namespace probtf_core
