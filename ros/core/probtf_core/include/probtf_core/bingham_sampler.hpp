#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <random>

namespace probtf_core {

// Kent, Ganeiber and Mardia, arXiv:1310.8110, Section 3 (BACG).
// This header and bingham_sampler.cpp have no ROS or Python dependencies.
class SamplingRandomSource {
 public:
  virtual ~SamplingRandomSource() = default;
  virtual double normal() = 0;
  virtual double uniform() = 0;
};

template <class Engine>
class StandardSamplingRandomSource final : public SamplingRandomSource {
 public:
  explicit StandardSamplingRandomSource(Engine& engine) : engine_(engine) {}
  double normal() override { return normal_(engine_); }
  double uniform() override { return uniform_(engine_); }

 private:
  Engine& engine_;
  std::normal_distribution<double> normal_{0.0, 1.0};
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

struct BinghamSamplingStatistics {
  std::uint64_t proposals = 0;
  std::uint64_t accepted = 0;
};

class BinghamSampler {
 public:
  // The repository uses exp(x' parameter x). The paper uses exp(-x' P x).
  static BinghamSampler fromParameterMatrix(const Eigen::MatrixXd& parameter);

  // P = basis * diag(precision) * basis', precision >= 0, min(precision) = 0.
  // Allows native wire adapters to reuse an already validated decomposition.
  BinghamSampler(const Eigen::VectorXd& precision, const Eigen::MatrixXd& basis);

  Eigen::VectorXd draw(SamplingRandomSource& random,
                       BinghamSamplingStatistics* statistics = nullptr) const;
  void drawBatch(std::size_t count, SamplingRandomSource& random,
                 double* row_major_output,
                 BinghamSamplingStatistics* statistics = nullptr) const;

  Eigen::Index dimension() const { return precision_.size(); }
  double proposalScale() const { return scale_; }
  double logEnvelope() const { return log_envelope_; }

 private:
  Eigen::VectorXd precision_;
  Eigen::VectorXd square_root_precision_;
  Eigen::MatrixXd basis_;
  Eigen::VectorXd proposal_standard_deviations_;
  double scale_;
  double log_envelope_;
};

}  // namespace probtf_core
