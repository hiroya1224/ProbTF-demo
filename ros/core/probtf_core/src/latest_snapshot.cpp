#include <probtf_core/latest_snapshot.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <probtf_msgs/ApproximationInfo.h>
#include <probtf_msgs/BinghamOrientation.h>
#include <probtf_msgs/ProbabilisticTransformComponent.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace probtf_core {
namespace {

using Matrix4d = Eigen::Matrix4d;

void setError(std::string* output, const std::string& value) {
  if (output != nullptr) {
    *output = value;
  }
}

std::string cleanFrame(const std::string& value) {
  const std::size_t first = value.find_first_not_of("/ \t\r\n");
  if (first == std::string::npos) {
    return std::string();
  }
  const std::size_t last = value.find_last_not_of("/ \t\r\n");
  return value.substr(first, last - first + 1);
}

bool finiteVector(const Eigen::Vector3d& value) { return value.allFinite(); }

Eigen::Vector3d vector3(const geometry_msgs::Vector3& value) {
  return Eigen::Vector3d(value.x, value.y, value.z);
}

Eigen::Quaterniond quaternion(const geometry_msgs::Quaternion& value) {
  return Eigen::Quaterniond(value.w, value.x, value.y, value.z);
}

bool normalizedQuaternion(const geometry_msgs::Quaternion& value,
                          Eigen::Quaterniond* output,
                          std::string* error) {
  Eigen::Quaterniond candidate = quaternion(value);
  const double norm = candidate.norm();
  if (!candidate.coeffs().allFinite() || !std::isfinite(norm) ||
      std::abs(norm - 1.0) > 1.0e-8) {
    setError(error, "Quaternion must be finite and have unit norm.");
    return false;
  }
  candidate.normalize();
  *output = candidate;
  return true;
}

void assignVector(const Eigen::Vector3d& value, geometry_msgs::Vector3* output) {
  output->x = value.x();
  output->y = value.y();
  output->z = value.z();
}

void assignQuaternion(const Eigen::Quaterniond& value,
                      geometry_msgs::Quaternion* output) {
  output->w = value.w();
  output->x = value.x();
  output->y = value.y();
  output->z = value.z();
}

Eigen::Matrix3d unpackSymmetric3(
    const boost::array<double, 6>& packed) {
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

bool validatedOrientation(
    const probtf_msgs::BinghamOrientation& orientation,
    Eigen::Quaterniond* reference,
    std::string* error) {
  if (!normalizedQuaternion(orientation.reference_quaternion, reference,
                            error)) {
    return false;
  }
  const Matrix4d shape = unpackSymmetric4(orientation.shape_upper_wxyz);
  if (!shape.allFinite() || std::abs(shape.trace()) > 1.0e-8) {
    setError(error, "Bingham shape must be finite and trace-zero.");
    return false;
  }
  if (orientation.kind == probtf_msgs::BinghamOrientation::DIRAC) {
    if (orientation.inverse_concentration != 0.0) {
      setError(error, "Dirac orientation requires zero inverse concentration.");
      return false;
    }
    const Eigen::Vector4d q(reference->w(), reference->x(), reference->y(),
                            reference->z());
    const Matrix4d expected =
        2.0 * q * q.transpose() - 0.5 * Matrix4d::Identity();
    if ((shape - expected).cwiseAbs().maxCoeff() > 1.0e-8) {
      setError(error, "Dirac Bingham shape does not match its quaternion.");
      return false;
    }
    return true;
  }
  if (orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    if (!std::isinf(orientation.inverse_concentration) ||
        orientation.inverse_concentration < 0.0 ||
        shape.cwiseAbs().maxCoeff() > 1.0e-8) {
      setError(error,
               "Uniform orientation requires infinite inverse concentration "
               "and a zero shape.");
      return false;
    }
    return true;
  }
  if (orientation.kind ==
      probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    if (!std::isfinite(orientation.inverse_concentration) ||
        orientation.inverse_concentration <= 0.0) {
      setError(error,
               "Finite Bingham inverse concentration must be positive.");
      return false;
    }
    Eigen::SelfAdjointEigenSolver<Matrix4d> solver(shape);
    if (solver.info() != Eigen::Success ||
        std::abs(solver.eigenvalues()(3) + solver.eigenvalues()(2) - 1.0) >
            1.0e-8) {
      setError(error, "Finite Bingham shape is not JMAA-normalized.");
      return false;
    }
    return true;
  }
  setError(error, "Unknown Bingham orientation kind.");
  return false;
}

bool validatedCovariance(Eigen::Matrix3d* covariance, std::string* error) {
  *covariance = 0.5 * (*covariance + covariance->transpose());
  if (!covariance->allFinite()) {
    setError(error, "Point covariance contains non-finite values.");
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(*covariance);
  if (solver.info() != Eigen::Success) {
    setError(error, "Point covariance eigendecomposition failed.");
    return false;
  }
  const double scale =
      std::max(1.0, covariance->cwiseAbs().rowwise().sum().maxCoeff());
  if (solver.eigenvalues()(0) < -1.0e-10 * scale) {
    setError(error, "Point covariance is not positive semidefinite.");
    return false;
  }
  const Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
  *covariance = solver.eigenvectors() * eigenvalues.asDiagonal() *
                solver.eigenvectors().transpose();
  *covariance = 0.5 * (*covariance + covariance->transpose());
  return true;
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

Vector9d rotationVector(const Eigen::Matrix3d& rotation) {
  Vector9d output;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      output(row + 3 * column) = rotation(row, column);
    }
  }
  return output;
}

Eigen::Matrix<double, 3, 9> rotationAction(const Eigen::Vector3d& point) {
  Eigen::Matrix<double, 3, 9> output =
      Eigen::Matrix<double, 3, 9>::Zero();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      output(row, row + 3 * column) = point(column);
    }
  }
  return output;
}

std::array<Matrix4d, 9> makeRotationForms() {
  std::array<Matrix4d, 9> forms;
  for (Matrix4d& form : forms) {
    form.setZero();
  }

  forms[0].diagonal() << 1.0, 1.0, -1.0, -1.0;
  forms[1](0, 3) = forms[1](3, 0) = -1.0;
  forms[1](1, 2) = forms[1](2, 1) = 1.0;
  forms[2](0, 2) = forms[2](2, 0) = 1.0;
  forms[2](1, 3) = forms[2](3, 1) = 1.0;

  forms[3](0, 3) = forms[3](3, 0) = 1.0;
  forms[3](1, 2) = forms[3](2, 1) = 1.0;
  forms[4].diagonal() << 1.0, -1.0, 1.0, -1.0;
  forms[5](0, 1) = forms[5](1, 0) = -1.0;
  forms[5](2, 3) = forms[5](3, 2) = 1.0;

  forms[6](0, 2) = forms[6](2, 0) = -1.0;
  forms[6](1, 3) = forms[6](3, 1) = 1.0;
  forms[7](0, 1) = forms[7](1, 0) = 1.0;
  forms[7](2, 3) = forms[7](3, 2) = 1.0;
  forms[8].diagonal() << 1.0, -1.0, -1.0, 1.0;
  return forms;
}

const std::array<Matrix4d, 9>& rotationForms() {
  static const std::array<Matrix4d, 9> forms = makeRotationForms();
  return forms;
}

std::size_t fourthIndex(int a, int b, int c, int d) {
  return static_cast<std::size_t>(((a * 4 + b) * 4 + c) * 4 + d);
}

struct NormalizerDerivatives {
  double constant = 0.0;
  Eigen::Vector4d first = Eigen::Vector4d::Zero();
  Matrix4d second = Matrix4d::Zero();
};

bool binghamNormalizerDerivatives(const Eigen::Vector4d& input_eigenvalues,
                                  int steps,
                                  NormalizerDerivatives* output,
                                  std::string* error) {
  if (steps < 1) {
    setError(error, "Bingham integration steps must be positive.");
    return false;
  }

  Eigen::Vector4d eigenvalues =
      input_eigenvalues.array() - input_eigenvalues.maxCoeff();
  constexpr double kPi = 3.141592653589793238462643383279502884;
  constexpr double kWd = 0.5;
  constexpr double kRatio = 2.5;
  constexpr double kMinimumSteps = 15.0;
  const double c = kMinimumSteps * kPi /
                   (kRatio * kRatio * (1.0 + kRatio) * kWd);
  const double d = 0.5 * c;
  const double h =
      std::sqrt(2.0 * kPi * d * (1.0 + kRatio) / (kWd * steps));
  const double p1 = std::sqrt(steps * h / kWd);
  const double p2 = std::sqrt(kWd * steps * h / 4.0);

  std::complex<double> constant_sum(0.0, 0.0);
  std::array<std::complex<double>, 4> first_sum;
  std::array<std::array<std::complex<double>, 4>, 4> second_sum;
  for (auto& value : first_sum) {
    value = std::complex<double>(0.0, 0.0);
  }
  for (auto& row : second_sum) {
    for (auto& value : row) {
      value = std::complex<double>(0.0, 0.0);
    }
  }

  const std::complex<double> imaginary(0.0, 1.0);
  for (int n = -steps - 1; n <= steps; ++n) {
    const double t = static_cast<double>(n) * h;
    const double weight = 0.5 * std::erfc(std::abs(t) / p1 - p2);
    std::array<std::complex<double>, 4> inverse;
    std::complex<double> function(1.0, 0.0);
    for (int index = 0; index < 4; ++index) {
      const std::complex<double> denominator(
          -eigenvalues(index) + c, t);
      inverse[index] = 1.0 / denominator;
      function /= std::sqrt(denominator);
    }
    const std::complex<double> common =
        weight * function * std::exp(imaginary * t);
    constant_sum += common;
    for (int row = 0; row < 4; ++row) {
      first_sum[row] += 0.5 * inverse[row] * common;
      for (int column = row; column < 4; ++column) {
        const double coefficient = 0.25 + (row == column ? 0.5 : 0.0);
        second_sum[row][column] +=
            coefficient * inverse[row] * inverse[column] * common;
      }
    }
  }

  const double factor = kPi * std::exp(c) * h;
  output->constant = factor * constant_sum.real();
  for (int row = 0; row < 4; ++row) {
    output->first(row) = factor * first_sum[row].real();
    for (int column = row; column < 4; ++column) {
      output->second(row, column) =
          factor * second_sum[row][column].real();
      output->second(column, row) = output->second(row, column);
    }
  }

  if (!std::isfinite(output->constant) || output->constant <= 0.0 ||
      !output->first.allFinite() || !output->second.allFinite()) {
    setError(error, "Bingham normalizer evaluation produced invalid values.");
    return false;
  }
  return true;
}

struct RotationMoments {
  Eigen::Matrix3d mean_rotation = Eigen::Matrix3d::Zero();
  Vector9d mean_vector = Vector9d::Zero();
  Matrix9d second_vector = Matrix9d::Zero();
};

bool rotationMoments(const probtf_msgs::BinghamOrientation& orientation,
                     int integration_steps,
                     RotationMoments* output,
                     std::string* error) {
  Eigen::Quaterniond reference;
  if (!validatedOrientation(orientation, &reference, error)) {
    return false;
  }
  if (orientation.kind == probtf_msgs::BinghamOrientation::DIRAC) {
    const Eigen::Quaterniond& q = reference;
    output->mean_rotation = q.toRotationMatrix();
    output->mean_vector = rotationVector(output->mean_rotation);
    output->second_vector =
        output->mean_vector * output->mean_vector.transpose();
    return true;
  }

  Matrix4d quaternion_second = Matrix4d::Zero();
  std::array<double, 256> quaternion_fourth;
  quaternion_fourth.fill(0.0);

  if (orientation.kind == probtf_msgs::BinghamOrientation::UNIFORM) {
    quaternion_second = 0.25 * Matrix4d::Identity();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
          for (int ell = 0; ell < 4; ++ell) {
            const double value =
                ((i == j && k == ell) ? 1.0 : 0.0) +
                ((i == k && j == ell) ? 1.0 : 0.0) +
                ((i == ell && j == k) ? 1.0 : 0.0);
            quaternion_fourth[fourthIndex(i, j, k, ell)] = value / 24.0;
          }
        }
      }
    }
  } else if (orientation.kind ==
             probtf_msgs::BinghamOrientation::FINITE_BINGHAM) {
    const Matrix4d shape = unpackSymmetric4(orientation.shape_upper_wxyz);
    const Matrix4d parameter = shape / orientation.inverse_concentration;
    Eigen::SelfAdjointEigenSolver<Matrix4d> solver(parameter);
    if (solver.info() != Eigen::Success) {
      setError(error, "Finite Bingham eigendecomposition failed.");
      return false;
    }

    NormalizerDerivatives derivatives;
    if (!binghamNormalizerDerivatives(solver.eigenvalues(), integration_steps,
                                      &derivatives, error)) {
      return false;
    }
    Eigen::Vector4d diagonal_second =
        derivatives.first / derivatives.constant;
    const double second_total = diagonal_second.sum();
    if (!std::isfinite(second_total) || second_total <= 0.0) {
      setError(error, "Finite Bingham second-moment normalization failed.");
      return false;
    }
    diagonal_second /= second_total;
    quaternion_second = solver.eigenvectors() * diagonal_second.asDiagonal() *
                        solver.eigenvectors().transpose();

    struct FourthTerm {
      int i;
      int j;
      int k;
      int ell;
      double value;
    };
    std::vector<FourthTerm> terms;
    for (int index = 0; index < 4; ++index) {
      terms.push_back({index, index, index, index,
                       derivatives.second(index, index) /
                           derivatives.constant});
    }
    for (int row = 0; row < 4; ++row) {
      for (int column = row + 1; column < 4; ++column) {
        const double value = derivatives.second(row, column) /
                             derivatives.constant;
        terms.push_back({row, row, column, column, value});
        terms.push_back({row, column, row, column, value});
        terms.push_back({row, column, column, row, value});
        terms.push_back({column, row, row, column, value});
        terms.push_back({column, row, column, row, value});
        terms.push_back({column, column, row, row, value});
      }
    }

    const Matrix4d& basis = solver.eigenvectors();
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            double value = 0.0;
            for (const FourthTerm& term : terms) {
              value += basis(a, term.i) * basis(b, term.j) *
                       basis(c, term.k) * basis(d, term.ell) * term.value;
            }
            quaternion_fourth[fourthIndex(a, b, c, d)] = value;
          }
        }
      }
    }
    double unit_norm = 0.0;
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 4; ++k) {
        unit_norm += quaternion_fourth[fourthIndex(i, i, k, k)];
      }
    }
    if (!std::isfinite(unit_norm) || unit_norm <= 0.0) {
      setError(error, "Finite Bingham fourth-moment normalization failed.");
      return false;
    }
    for (double& value : quaternion_fourth) {
      value /= unit_norm;
    }
  } else {
    setError(error, "Unknown Bingham orientation kind.");
    return false;
  }

  const auto& forms = rotationForms();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      output->mean_rotation(row, column) =
          forms[static_cast<std::size_t>(row * 3 + column)]
              .cwiseProduct(quaternion_second)
              .sum();
    }
  }
  output->mean_vector = rotationVector(output->mean_rotation);

  for (int vector_row = 0; vector_row < 9; ++vector_row) {
    const int row_a = vector_row % 3;
    const int column_a = vector_row / 3;
    const Matrix4d& form_a =
        forms[static_cast<std::size_t>(row_a * 3 + column_a)];
    for (int vector_column = 0; vector_column < 9; ++vector_column) {
      const int row_b = vector_column % 3;
      const int column_b = vector_column / 3;
      const Matrix4d& form_b =
          forms[static_cast<std::size_t>(row_b * 3 + column_b)];
      double value = 0.0;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 4; ++k) {
            for (int ell = 0; ell < 4; ++ell) {
              value += form_a(i, j) * form_b(k, ell) *
                       quaternion_fourth[fourthIndex(i, j, k, ell)];
            }
          }
        }
      }
      output->second_vector(vector_row, vector_column) = value;
    }
  }
  output->second_vector =
      0.5 * (output->second_vector + output->second_vector.transpose());
  return output->mean_rotation.allFinite() &&
         output->second_vector.allFinite();
}

bool normalizedComponents(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    std::vector<std::pair<const probtf_msgs::ProbabilisticTransformComponent*,
                          double>>* output,
    std::string* error) {
  output->clear();
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
  double total = 0.0;
  for (const auto& component : record.components) {
    if (component.weight > 0.0) {
      total += component.weight / scale;
    }
  }
  for (const auto& component : record.components) {
    if (component.weight > 0.0) {
      output->emplace_back(&component, component.weight / scale / total);
    }
  }
  return true;
}

bool componentIsDeterministic(
    const probtf_msgs::ProbabilisticTransformComponent& component) {
  if (component.orientation.kind !=
      probtf_msgs::BinghamOrientation::DIRAC) {
    return false;
  }
  return std::all_of(
             component.translation.residual_covariance_upper.begin(),
             component.translation.residual_covariance_upper.end(),
             [](double value) { return value == 0.0; }) &&
         std::all_of(component.translation.rotation_coupling.begin(),
                     component.translation.rotation_coupling.end(),
                     [](double value) { return value == 0.0; });
}

bool exactTransform(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    Eigen::Quaterniond* rotation,
    Eigen::Vector3d* translation,
    std::string* error) {
  std::vector<std::pair<const probtf_msgs::ProbabilisticTransformComponent*,
                        double>>
      components;
  if (!normalizedComponents(record, &components, error)) {
    return false;
  }
  if (components.size() != 1 ||
      !componentIsDeterministic(*components.front().first)) {
    setError(error, "Prob-TF record is stochastic.");
    return false;
  }
  const auto& component = *components.front().first;
  if (!validatedOrientation(component.orientation, rotation, error)) {
    return false;
  }
  *translation = vector3(component.translation.mean_at_reference);
  if (!translation->allFinite()) {
    setError(error, "Prob-TF translation contains non-finite values.");
    return false;
  }
  return true;
}

bool applyComponent(
    const probtf_msgs::ProbabilisticTransformComponent& component,
    const PointMoments& input,
    int integration_steps,
    PointMoments* output,
    std::string* error) {
  RotationMoments rotation;
  if (!rotationMoments(component.orientation, integration_steps, &rotation,
                       error)) {
    return false;
  }
  const Eigen::Matrix<double, 3, 9> coupling =
      couplingMatrix(component.translation.rotation_coupling);
  const Eigen::Matrix<double, 3, 9> linear =
      rotationAction(input.mean) + coupling;
  Eigen::Quaterniond reference;
  if (!validatedOrientation(component.orientation, &reference, error)) {
    return false;
  }
  const Eigen::Vector3d mean_at_reference =
      vector3(component.translation.mean_at_reference);
  Eigen::Matrix3d residual =
      unpackSymmetric3(component.translation.residual_covariance_upper);
  if (!mean_at_reference.allFinite() || !coupling.allFinite() ||
      !residual.allFinite()) {
    setError(error, "Prob-TF component contains non-finite translation data.");
    return false;
  }
  if (!validatedCovariance(&residual, error)) {
    setError(error, "Translation residual covariance is not positive semidefinite.");
    return false;
  }
  const Eigen::Vector3d offset =
      mean_at_reference - coupling * rotationVector(reference.toRotationMatrix());
  output->mean = offset + linear * rotation.mean_vector;
  const Matrix9d centered =
      rotation.second_vector -
      rotation.mean_vector * rotation.mean_vector.transpose();
  output->covariance = linear * centered * linear.transpose() + residual;

  Eigen::Matrix3d rotated_covariance = Eigen::Matrix3d::Zero();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int input_row = 0; input_row < 3; ++input_row) {
        for (int input_column = 0; input_column < 3; ++input_column) {
          rotated_covariance(row, column) +=
              input.covariance(input_row, input_column) *
              rotation.second_vector(row + 3 * input_row,
                                     column + 3 * input_column);
        }
      }
    }
  }
  output->covariance += rotated_covariance;
  output->covariance =
      0.5 * (output->covariance + output->covariance.transpose());
  if (!output->mean.allFinite() || !output->covariance.allFinite()) {
    setError(error, "Prob-TF point-moment evaluation became non-finite.");
    return false;
  }
  return validatedCovariance(&output->covariance, error);
}

bool applyForward(const probtf_msgs::ProbabilisticTransformStamped& record,
                  const PointMoments& input,
                  int integration_steps,
                  PointMoments* output,
                  std::string* error) {
  Eigen::Quaterniond exact_rotation;
  Eigen::Vector3d exact_translation;
  std::string exact_error;
  if (exactTransform(record, &exact_rotation, &exact_translation,
                     &exact_error)) {
    const Eigen::Matrix3d rotation = exact_rotation.toRotationMatrix();
    output->mean = rotation * input.mean + exact_translation;
    output->covariance = rotation * input.covariance * rotation.transpose();
    return true;
  }

  std::vector<std::pair<const probtf_msgs::ProbabilisticTransformComponent*,
                        double>>
      components;
  if (!normalizedComponents(record, &components, error)) {
    return false;
  }
  std::vector<std::pair<double, PointMoments>> summaries;
  summaries.reserve(components.size());
  for (const auto& weighted : components) {
    PointMoments summary;
    if (!applyComponent(*weighted.first, input, integration_steps, &summary,
                        error)) {
      return false;
    }
    summaries.emplace_back(weighted.second, std::move(summary));
  }
  output->mean.setZero();
  for (const auto& weighted : summaries) {
    output->mean += weighted.first * weighted.second.mean;
  }
  output->covariance.setZero();
  for (const auto& weighted : summaries) {
    const Eigen::Vector3d difference = weighted.second.mean - output->mean;
    output->covariance +=
        weighted.first *
        (weighted.second.covariance + difference * difference.transpose());
  }
  output->covariance =
      0.5 * (output->covariance + output->covariance.transpose());
  return validatedCovariance(&output->covariance, error);
}

bool applyInverse(const probtf_msgs::ProbabilisticTransformStamped& record,
                  const PointMoments& input,
                  PointMoments* output,
                  std::string* error) {
  Eigen::Quaterniond rotation_quaternion;
  Eigen::Vector3d translation;
  if (!exactTransform(record, &rotation_quaternion, &translation, error)) {
    setError(error,
             "Stochastic inverse point moments are unavailable, matching the "
             "Prob-TF Python evaluator contract.");
    return false;
  }
  const Eigen::Matrix3d inverse_rotation =
      rotation_quaternion.toRotationMatrix().transpose();
  output->mean = inverse_rotation * (input.mean - translation);
  output->covariance =
      inverse_rotation * input.covariance * inverse_rotation.transpose();
  return true;
}

void setExactApproximation(probtf_msgs::ApproximationInfo* approximation) {
  approximation->kind = probtf_msgs::ApproximationInfo::EXACT;
  approximation->lossy = false;
  approximation->detail.clear();
  approximation->source.clear();
  approximation->has_error_bound = false;
  approximation->error_bound = 0.0;
}

}  // namespace

LatestSnapshot::LatestSnapshot(
    const probtf_msgs::ProbabilisticTransformArray& dynamic_records,
    const probtf_msgs::ProbabilisticTransformArray& static_records) {
  addRecords(static_records, true);
  if (valid_) {
    addRecords(dynamic_records, false);
  }
}

void LatestSnapshot::addRecords(
    const probtf_msgs::ProbabilisticTransformArray& records,
    bool expected_static) {
  for (const auto& record : records.transforms) {
    const std::string parent = cleanFrame(record.header.frame_id);
    const std::string child = cleanFrame(record.child_frame_id);
    if (parent.empty() || child.empty() || record.edge_id.empty()) {
      valid_ = false;
      validation_error_ = "Prob-TF records require parent, child, and edge IDs.";
      return;
    }
    if (record.is_static != expected_static) {
      valid_ = false;
      validation_error_ = expected_static
                              ? "Dynamic record appeared on the static topic."
                              : "Static record appeared in the dynamic batch.";
      return;
    }
    if (edge_by_child_.count(child) != 0U) {
      valid_ = false;
      validation_error_ = "Frame '" + child +
                          "' has more than one physical parent edge.";
      return;
    }
    edge_by_child_[child] = &record;
    frames_.insert(parent);
    frames_.insert(child);
  }
}

bool LatestSnapshot::valid(std::string* error) const {
  if (!valid_) {
    setError(error, validation_error_);
  }
  return valid_;
}

bool LatestSnapshot::buildPath(const std::string& target_frame_value,
                               const std::string& source_frame_value,
                               std::vector<PathStep>* path,
                               std::string* error) const {
  path->clear();
  if (!valid(error)) {
    return false;
  }
  const std::string target_frame = cleanFrame(target_frame_value);
  const std::string source_frame = cleanFrame(source_frame_value);
  if (frames_.count(target_frame) == 0U ||
      frames_.count(source_frame) == 0U) {
    setError(error, "Prob-TF lookup references an unknown frame.");
    return false;
  }
  if (target_frame == source_frame) {
    return true;
  }

  std::unordered_map<std::string, std::size_t> source_ancestors;
  std::unordered_set<std::string> visited;
  std::string current = source_frame;
  std::size_t depth = 0;
  while (true) {
    if (!visited.insert(current).second) {
      setError(error, "Prob-TF topology contains a cycle.");
      return false;
    }
    source_ancestors[current] = depth++;
    const auto edge = edge_by_child_.find(current);
    if (edge == edge_by_child_.end()) {
      break;
    }
    current = cleanFrame(edge->second->header.frame_id);
  }

  std::vector<const probtf_msgs::ProbabilisticTransformStamped*> target_up;
  visited.clear();
  current = target_frame;
  while (source_ancestors.count(current) == 0U) {
    if (!visited.insert(current).second) {
      setError(error, "Prob-TF topology contains a cycle.");
      return false;
    }
    const auto edge = edge_by_child_.find(current);
    if (edge == edge_by_child_.end()) {
      setError(error, "Prob-TF source and target frames are disconnected.");
      return false;
    }
    target_up.push_back(edge->second);
    current = cleanFrame(edge->second->header.frame_id);
  }
  const std::string common_ancestor = current;

  current = source_frame;
  while (current != common_ancestor) {
    const auto edge = edge_by_child_.find(current);
    if (edge == edge_by_child_.end()) {
      setError(error, "Prob-TF source path ended unexpectedly.");
      return false;
    }
    path->push_back({edge->second, false});
    current = cleanFrame(edge->second->header.frame_id);
  }
  for (auto iterator = target_up.rbegin(); iterator != target_up.rend();
       ++iterator) {
    path->push_back({*iterator, true});
  }
  return true;
}

bool LatestSnapshot::lookupPointMoments(
    const std::string& target_frame,
    const std::string& source_frame,
    const Eigen::Vector3d& source_point,
    PointMomentObservation* observation,
    std::string* error,
    int bingham_integration_steps) const {
  if (observation == nullptr) {
    setError(error, "Point-moment output must not be null.");
    return false;
  }
  if (!finiteVector(source_point)) {
    setError(error, "Source point must be finite.");
    return false;
  }
  std::vector<PathStep> path;
  if (!buildPath(target_frame, source_frame, &path, error)) {
    return false;
  }

  bool all_deterministic = true;
  bool repeated_dependency = false;
  std::unordered_set<std::string> path_dependencies;
  for (const PathStep& step : path) {
    Eigen::Quaterniond deterministic_rotation;
    Eigen::Vector3d deterministic_translation;
    std::string deterministic_error;
    all_deterministic =
        exactTransform(*step.record, &deterministic_rotation,
                       &deterministic_translation, &deterministic_error) &&
        all_deterministic;

    std::unordered_set<std::string> edge_dependencies;
    edge_dependencies.insert(step.record->edge_id);
    edge_dependencies.insert(
        step.record->provenance.derived_from_edge_ids.begin(),
        step.record->provenance.derived_from_edge_ids.end());
    for (const auto& component : step.record->components) {
      edge_dependencies.insert(
          component.provenance.derived_from_edge_ids.begin(),
          component.provenance.derived_from_edge_ids.end());
    }
    for (const std::string& dependency : edge_dependencies) {
      if (!dependency.empty() && !path_dependencies.insert(dependency).second) {
        repeated_dependency = true;
      }
    }
  }
  if (repeated_dependency && !all_deterministic) {
    setError(error,
             "Repeated latent edge dependencies require a dependency-aware "
             "stochastic evaluator.");
    return false;
  }

  PointMoments current;
  current.mean = source_point;
  ros::Time resolved_stamp;
  bool has_dynamic_stamp = false;
  std::vector<std::string> edge_ids;
  edge_ids.reserve(path.size());
  for (const PathStep& step : path) {
    PointMoments next;
    const bool success =
        step.inverse
            ? applyInverse(*step.record, current, &next, error)
            : applyForward(*step.record, current, bingham_integration_steps,
                           &next, error);
    if (!success) {
      return false;
    }
    current = std::move(next);
    edge_ids.push_back(step.record->edge_id);
    if (!step.record->is_static &&
        (!has_dynamic_stamp || step.record->header.stamp < resolved_stamp)) {
      resolved_stamp = step.record->header.stamp;
      has_dynamic_stamp = true;
    }
  }

  observation->target_frame = cleanFrame(target_frame);
  observation->source_frame = cleanFrame(source_frame);
  observation->resolved_stamp = has_dynamic_stamp ? resolved_stamp : ros::Time(0);
  observation->edge_ids = std::move(edge_ids);
  observation->moments = std::move(current);
  return true;
}

bool deterministicTfToProbTf(
    const geometry_msgs::TransformStamped& transform,
    const std::string& authority,
    bool is_static,
    probtf_msgs::ProbabilisticTransformStamped* output,
    std::string* error) {
  if (output == nullptr) {
    setError(error, "Prob-TF output must not be null.");
    return false;
  }
  const std::string parent = cleanFrame(transform.header.frame_id);
  const std::string child = cleanFrame(transform.child_frame_id);
  if (parent.empty() || child.empty()) {
    setError(error, "TF parent and child frames must be non-empty.");
    return false;
  }
  Eigen::Quaterniond rotation;
  if (!normalizedQuaternion(transform.transform.rotation, &rotation, error)) {
    return false;
  }
  const Eigen::Vector3d translation = vector3(transform.transform.translation);
  if (!translation.allFinite()) {
    setError(error, "TF translation must be finite.");
    return false;
  }

  *output = probtf_msgs::ProbabilisticTransformStamped();
  output->header = transform.header;
  output->header.frame_id = parent;
  output->child_frame_id = child;
  output->edge_id = parent + "__to__" + child;
  output->authority = authority;
  output->is_static = is_static;
  output->representative_kind =
      probtf_msgs::ProbabilisticTransformStamped::REPRESENTATIVE_EXACT_MAP;
  assignVector(translation, &output->representative.translation);
  assignQuaternion(rotation, &output->representative.rotation);
  setExactApproximation(&output->approximation);
  output->provenance.source_ids.push_back(authority);
  output->provenance.method = "tf_import";

  output->components.resize(1);
  auto& component = output->components.front();
  component.component_id = output->edge_id + ":tf";
  component.weight = 1.0;
  component.orientation.kind = probtf_msgs::BinghamOrientation::DIRAC;
  component.orientation.inverse_concentration = 0.0;
  assignQuaternion(rotation, &component.orientation.reference_quaternion);
  const Eigen::Vector4d quaternion_wxyz(
      rotation.w(), rotation.x(), rotation.y(), rotation.z());
  const Matrix4d shape =
      2.0 * quaternion_wxyz * quaternion_wxyz.transpose() -
      0.5 * Matrix4d::Identity();
  std::size_t shape_index = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = row; column < 4; ++column) {
      component.orientation.shape_upper_wxyz[shape_index++] =
          shape(row, column);
    }
  }
  assignVector(translation, &component.translation.mean_at_reference);
  std::fill(component.translation.residual_covariance_upper.begin(),
            component.translation.residual_covariance_upper.end(), 0.0);
  std::fill(component.translation.rotation_coupling.begin(),
            component.translation.rotation_coupling.end(), 0.0);
  setExactApproximation(&component.approximation);
  component.provenance.source_ids.push_back(authority);
  component.provenance.method = "tf_import";
  return true;
}

bool exactProbTfToTf(
    const probtf_msgs::ProbabilisticTransformStamped& record,
    geometry_msgs::TransformStamped* output,
    std::string* error) {
  if (output == nullptr) {
    setError(error, "TF output must not be null.");
    return false;
  }
  Eigen::Quaterniond rotation;
  Eigen::Vector3d translation;
  if (!exactTransform(record, &rotation, &translation, error)) {
    return false;
  }
  *output = geometry_msgs::TransformStamped();
  output->header = record.header;
  output->header.frame_id = cleanFrame(record.header.frame_id);
  output->child_frame_id = cleanFrame(record.child_frame_id);
  assignVector(translation, &output->transform.translation);
  assignQuaternion(rotation, &output->transform.rotation);
  return true;
}

}  // namespace probtf_core
