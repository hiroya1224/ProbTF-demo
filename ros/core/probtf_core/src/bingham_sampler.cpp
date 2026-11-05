#include <probtf_core/bingham_sampler.hpp>

#include <Eigen/Eigenvalues>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace probtf_core {
namespace {

double scaleEquation(const Eigen::VectorXd& precision, double b) {
  double sum = 0.0;
  for (Eigen::Index i = 0; i < precision.size(); ++i) {
    // Algebraically 1/(b + 2*lambda), without overflowing 2*lambda.
    sum += 0.5 / (precision(i) + 0.5 * b);
  }
  return sum - 1.0;
}

double optimalScale(const Eigen::VectorXd& precision) {
  // Equation (3.6). At least one eigenvalue is zero, hence b is in [1,q].
  double lower = 1.0;
  double upper = static_cast<double>(precision.size());
  if (precision.maxCoeff() == 0.0) {
    return upper;
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double middle = lower + 0.5 * (upper - lower);
    if (middle == lower || middle == upper) {
      break;
    }
    if (scaleEquation(precision, middle) > 0.0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower + 0.5 * (upper - lower);
}

double logProposalForm(double quadratic, double half_b) {
  if (quadratic <= half_b) {
    return std::log1p(quadratic / half_b);
  }
  // log(1 + 2*u/b), also when the ratio itself would overflow.
  return std::log(quadratic) - std::log(half_b) +
         std::log1p(half_b / quadratic);
}

}  // namespace

BinghamSampler BinghamSampler::fromParameterMatrix(
    const Eigen::MatrixXd& parameter) {
  if (parameter.rows() < 2 || parameter.rows() != parameter.cols() ||
      !parameter.allFinite() ||
      !parameter.isApprox(parameter.transpose(), 1.0e-12)) {
    throw std::invalid_argument("Bingham parameter must be finite and symmetric, dimension >= 2.");
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(parameter);
  if (solver.info() != Eigen::Success) {
    throw std::invalid_argument("Bingham parameter eigendecomposition failed.");
  }
  const Eigen::VectorXd precision =
      Eigen::VectorXd::Constant(parameter.rows(), solver.eigenvalues().maxCoeff()) -
      solver.eigenvalues();
  return BinghamSampler(precision, solver.eigenvectors());
}

BinghamSampler::BinghamSampler(const Eigen::VectorXd& precision,
                               const Eigen::MatrixXd& basis)
    : precision_(precision), basis_(basis) {
  const Eigen::Index q = precision.size();
  if (q < 2 || basis.rows() != q || basis.cols() != q ||
      !precision.allFinite() || precision.minCoeff() != 0.0 ||
      !basis.allFinite() ||
      !(basis.transpose() * basis).isApprox(Eigen::MatrixXd::Identity(q, q), 1.0e-10)) {
    throw std::invalid_argument("BACG requires nonnegative precision with minimum zero and an orthonormal basis.");
  }
  scale_ = optimalScale(precision_);
  const double half_b = 0.5 * scale_;
  proposal_standard_deviations_.resize(q);
  for (Eigen::Index i = 0; i < q; ++i) {
    // sqrt(1/(1 + 2*lambda/b)); no dense inverse or concentration cap.
    proposal_standard_deviations_(i) =
        std::sqrt(half_b) / std::sqrt(precision_(i) + half_b);
  }
  square_root_precision_ = precision_.cwiseSqrt();
  log_envelope_ = -0.5 * (static_cast<double>(q) - scale_) +
                  0.5 * static_cast<double>(q) *
                      std::log(static_cast<double>(q) / scale_);
}

Eigen::VectorXd BinghamSampler::draw(
    SamplingRandomSource& random, BinghamSamplingStatistics* statistics) const {
  const Eigen::Index q = dimension();
  Eigen::VectorXd normal(q);
  Eigen::VectorXd coordinates(q);
  for (;;) {
    for (Eigen::Index i = 0; i < q; ++i) {
      normal(i) = random.normal();
    }
    if (!normal.allFinite()) {
      throw std::runtime_error("Random source returned a non-finite normal variate.");
    }
    // Apply the symmetric square root in world coordinates. Eigenvector signs
    // and bases within repeated eigenspaces do not change the seeded proposal.
    coordinates.noalias() = basis_.transpose() * normal;
    coordinates.array() *= proposal_standard_deviations_.array();
    const double norm = coordinates.stableNorm();
    if (norm == 0.0) {
      continue;
    }
    coordinates /= norm;
    if (statistics != nullptr) {
      ++statistics->proposals;
    }
    const double quadratic =
        (square_root_precision_.array() * coordinates.array()).matrix().squaredNorm();
    const double log_probability = std::isfinite(quadratic)
        ? -quadratic + 0.5 * static_cast<double>(q) *
                           logProposalForm(quadratic, 0.5 * scale_) - log_envelope_
        : -std::numeric_limits<double>::infinity();
    const double uniform = random.uniform();
    if (!std::isfinite(uniform) || uniform < 0.0 || uniform >= 1.0) {
      throw std::runtime_error("Random source returned a variate outside [0,1).");
    }
    // Equations (3.4)-(3.5). Never replace a rejection by a mode, uniform draw,
    // tangent Gaussian, or a sample from a clipped concentration.
    if (log_probability >= 0.0 ||
        (std::isfinite(log_probability) &&
         (uniform == 0.0 || std::log(uniform) < log_probability))) {
      if (statistics != nullptr) {
        ++statistics->accepted;
      }
      return basis_ * coordinates;
    }
  }
}

void BinghamSampler::drawBatch(std::size_t count, SamplingRandomSource& random,
                               double* row_major_output,
                               BinghamSamplingStatistics* statistics) const {
  if (count != 0 && row_major_output == nullptr) {
    throw std::invalid_argument("Bingham sample output must not be null.");
  }
  for (std::size_t i = 0; i < count; ++i) {
    Eigen::Map<Eigen::VectorXd>(row_major_output + i * dimension(), dimension()) =
        draw(random, statistics);
  }
}

}  // namespace probtf_core
