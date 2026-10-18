// Demo-local analytic reverse differentiation of the unchanged high-
// concentration moment formulas in spherical_law/first_order.py.
//
// A trial N(m,P) is expressed in the ORIGINAL joint chart.  Reexpression
// uses t'=t+m_t, R'=R Exp(m_r), Q=G P G^T, G_r=J_r(m_r), exactly as the
// existing reexpress_state.  The first-order O(P) correction and ALL joint
// covariance entries are then propagated.  No point posterior, sampling,
// numerical parameter differences, Bingham eigenderivatives, or core edits.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace moment_reverse {
struct Node {
  double value, adjoint, da, db;
  int a, b;
};
class Tape {
public:
  std::vector<Node> nodes;
  int input(double value) {
    nodes.push_back({value, 0., 0., 0., -1, -1});
    return static_cast<int>(nodes.size()) - 1;
  }
  int operation(double value, int a, double da, int b = -1, double db = 0.) {
    nodes.push_back({value, 0., da, db, a, b});
    return static_cast<int>(nodes.size()) - 1;
  }
  void backward(int output) {
    if (output < 0)
      return;
    nodes[output].adjoint = 1.;
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
      const auto node = nodes[i];
      if (node.a >= 0)
        nodes[node.a].adjoint += node.adjoint * node.da;
      if (node.b >= 0)
        nodes[node.b].adjoint += node.adjoint * node.db;
    }
  }
};
static thread_local Tape *active_tape = nullptr;
struct Scalar {
  double value;
  int index;
  Scalar(double v = 0.) : value(v), index(-1) {}
  static Scalar input(double v) {
    Scalar out(v);
    out.index = active_tape->input(v);
    return out;
  }
  static Scalar unary(double v, const Scalar &a, double da) {
    Scalar out(v);
    if (a.index >= 0)
      out.index = active_tape->operation(v, a.index, da);
    return out;
  }
  static Scalar binary(double v, const Scalar &a, double da, const Scalar &b,
                       double db) {
    Scalar out(v);
    if (a.index >= 0 || b.index >= 0)
      out.index = active_tape->operation(v, a.index, da, b.index, db);
    return out;
  }
  Scalar &operator+=(const Scalar &b);
  Scalar &operator-=(const Scalar &b);
  Scalar &operator*=(const Scalar &b);
};
static Scalar operator+(const Scalar &a, const Scalar &b) {
  if (a.index < 0 && a.value == 0.)
    return b;
  if (b.index < 0 && b.value == 0.)
    return a;
  return Scalar::binary(a.value + b.value, a, 1., b, 1.);
}
static Scalar operator-(const Scalar &a, const Scalar &b) {
  if (b.index < 0 && b.value == 0.)
    return a;
  return Scalar::binary(a.value - b.value, a, 1., b, -1.);
}
static Scalar operator-(const Scalar &a) {
  return Scalar::unary(-a.value, a, -1.);
}
static Scalar operator*(const Scalar &a, const Scalar &b) {
  if ((a.index < 0 && a.value == 0.) || (b.index < 0 && b.value == 0.))
    return Scalar(0.);
  if (a.index < 0 && a.value == 1.)
    return b;
  if (b.index < 0 && b.value == 1.)
    return a;
  return Scalar::binary(a.value * b.value, a, b.value, b, a.value);
}
static Scalar operator/(const Scalar &a, const Scalar &b) {
  if (b.value == 0.)
    throw std::invalid_argument("A moment derivative divides by zero.");
  return Scalar::binary(a.value / b.value, a, 1. / b.value, b,
                        -a.value / (b.value * b.value));
}
static bool operator==(const Scalar &a, const Scalar &b) {
  return a.value == b.value;
}
static bool operator!=(const Scalar &a, const Scalar &b) { return !(a == b); }
Scalar &Scalar::operator+=(const Scalar &b) { return *this = *this + b; }
Scalar &Scalar::operator-=(const Scalar &b) { return *this = *this - b; }
Scalar &Scalar::operator*=(const Scalar &b) { return *this = *this * b; }
static Scalar sqrt(const Scalar &a) {
  if (a.value <= 0.)
    throw std::invalid_argument(
        "A differentiated square root is not positive.");
  const double value = std::sqrt(a.value);
  return Scalar::unary(value, a, .5 / value);
}
static Scalar sin(const Scalar &a) {
  return Scalar::unary(std::sin(a.value), a, std::cos(a.value));
}
static Scalar cos(const Scalar &a) {
  return Scalar::unary(std::cos(a.value), a, -std::sin(a.value));
}
static Scalar atan2(const Scalar &a, const Scalar &b) {
  const double denominator = a.value * a.value + b.value * b.value;
  if (denominator == 0.)
    throw std::invalid_argument("Rotation chart angle is undefined.");
  return Scalar::binary(std::atan2(a.value, b.value), a, b.value / denominator,
                        b, -a.value / denominator);
}
// Required by Eigen scalar products; all values are real scalars.
static Scalar conj(const Scalar &a) { return a; }
static Scalar real(const Scalar &a) { return a; }
static Scalar imag(const Scalar &) { return Scalar(0.); }
static Scalar abs2(const Scalar &a) { return a * a; }
} // namespace moment_reverse

namespace Eigen {
template <>
struct NumTraits<moment_reverse::Scalar>
    : GenericNumTraits<moment_reverse::Scalar> {
  using Real = moment_reverse::Scalar;
  using NonInteger = moment_reverse::Scalar;
  using Nested = moment_reverse::Scalar;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 8,
    MulCost = 8
  };
};
} // namespace Eigen

namespace moment_reverse {
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
using Row =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleMatrix = Eigen::MatrixXd;
using DoubleVector = Eigen::VectorXd;

static Matrix3 skew(const Vector3 &v) {
  Matrix3 out;
  out << Scalar(0.), -v[2], v[1], v[2], Scalar(0.), -v[0], -v[1], v[0],
      Scalar(0.);
  return out;
}
static Vector3 vee(const Matrix3 &m) {
  Vector3 out;
  out << (m(2, 1) - m(1, 2)) * Scalar(.5), (m(0, 2) - m(2, 0)) * Scalar(.5),
      (m(1, 0) - m(0, 1)) * Scalar(.5);
  return out;
}
static Vector4 normalize(const Vector4 &q) {
  Scalar squared(0.);
  for (int i = 0; i < 4; ++i)
    squared += q[i] * q[i];
  return q / sqrt(squared);
}
static Vector4 multiply(const Vector4 &a, const Vector4 &b) {
  Vector4 out;
  out[0] = a[0] * b[0] - a.tail<3>().dot(b.tail<3>());
  out.tail<3>() =
      a[0] * b.tail<3>() + b[0] * a.tail<3>() + a.tail<3>().cross(b.tail<3>());
  return normalize(out);
}
static Vector4 rotation_quaternion(const Vector3 &v) {
  const Scalar squared = v.dot(v);
  Vector4 q;
  if (squared.value < 1e-12) {
    const Scalar scale =
        Scalar(.5) - squared / Scalar(48.) + squared * squared / Scalar(3840.);
    q[0] = Scalar(1.) - squared / Scalar(8.);
    q.tail<3>() = scale * v;
  } else {
    const Scalar angle = sqrt(squared);
    q[0] = cos(Scalar(.5) * angle);
    q.tail<3>() = sin(Scalar(.5) * angle) / angle * v;
  }
  return normalize(q);
}
static Matrix3 rotation(const Vector4 &input) {
  const Vector4 q = normalize(input);
  const Scalar w = q[0], x = q[1], y = q[2], z = q[3];
  Matrix3 R;
  R << w * w + x * x - y * y - z * z, Scalar(2.) * (x * y - w * z),
      Scalar(2.) * (x * z + w * y), Scalar(2.) * (x * y + w * z),
      w * w - x * x + y * y - z * z, Scalar(2.) * (y * z - w * x),
      Scalar(2.) * (x * z - w * y), Scalar(2.) * (y * z + w * x),
      w * w - x * x - y * y + z * z;
  return R;
}
static Matrix3 right_jacobian(const Vector3 &v) {
  const Scalar squared = v.dot(v);
  const Matrix3 K = skew(v);
  if (squared.value < 1e-12)
    return Matrix3::Identity() - Scalar(.5) * K + Scalar(1. / 6.) * K * K;
  const Scalar angle = sqrt(squared);
  return Matrix3::Identity() - (Scalar(1.) - cos(angle)) / squared * K +
         (angle - sin(angle)) / (squared * angle) * K * K;
}
static Matrix3 right_jacobian_inverse(const Vector3 &v) {
  const Scalar squared = v.dot(v);
  const Matrix3 K = skew(v);
  Scalar coefficient;
  if (squared.value < 1e-12)
    coefficient = Scalar(1. / 12.) + squared / Scalar(720.);
  else {
    const Scalar angle = sqrt(squared), half = Scalar(.5) * angle;
    coefficient = (Scalar(1.) - half * cos(half) / sin(half)) / squared;
  }
  return Matrix3::Identity() + Scalar(.5) * K + coefficient * K * K;
}
static Vector3 rotation_vector(Vector4 q) {
  q = normalize(q);
  // Same shortest quaternion chart as geometry.quaternion_to_rotation_vector.
  if (q[0].value < 0.)
    q = -q;
  else if (std::abs(q[0].value) < 1e-15) {
    int pivot = 1;
    for (int i = 2; i < 4; ++i)
      if (std::abs(q[i].value) > std::abs(q[pivot].value))
        pivot = i;
    if (q[pivot].value < 0.)
      q = -q;
  }
  const Scalar squared = q.tail<3>().dot(q.tail<3>());
  if (squared.value < 1e-12)
    return Scalar(2.) * q.tail<3>();
  const Scalar length = sqrt(squared);
  const Scalar positive_real = q[0].value < 0. ? Scalar(0.) : q[0];
  return Scalar(2.) * atan2(length, positive_real) / length * q.tail<3>();
}
static Matrix constant_matrix(const Eigen::Ref<const DoubleMatrix> &input) {
  Matrix out(input.rows(), input.cols());
  for (int i = 0; i < input.rows(); ++i)
    for (int j = 0; j < input.cols(); ++j)
      out(i, j) = Scalar(input(i, j));
  return out;
}
static DoubleMatrix values(const Matrix &input) {
  DoubleMatrix out(input.rows(), input.cols());
  for (int i = 0; i < input.rows(); ++i)
    for (int j = 0; j < input.cols(); ++j)
      out(i, j) = input(i, j).value;
  return out;
}

struct Result {
  Vector mean;
  Matrix covariance;
};
static Result query(const std::vector<Vector3> &translations,
                    const std::vector<Vector4> &quaternions,
                    const unsigned char *random, const Matrix &P, int pathn,
                    const int *indices, const unsigned char *inverse,
                    const Eigen::Vector3d &point, int kind,
                    const Eigen::Vector4d &reference,
                    const DoubleMatrix &projection) {
  const int n = P.rows(), dimension = kind == 2 ? 6 : 3;
  Vector3 value = point.cast<Scalar>(), correction = Vector3::Zero();
  Matrix J = Matrix::Zero(3, n);
  Matrix3 generators[3] = {skew(Vector3::UnitX()), skew(Vector3::UnitY()),
                           skew(Vector3::UnitZ())};
  // first_order.py::_point_jet, including radial AND shared mixed O(P).
  for (int k = pathn - 1; k >= 0; --k) {
    const int edge = indices[k], offset = 6 * edge;
    Matrix3 R = rotation(quaternions[edge]), D[3];
    for (int a = 0; a < 3; ++a)
      D[a] = R * generators[a];
    if (inverse[k]) {
      R.transposeInPlace();
      for (auto &d : D)
        d.transposeInPlace();
    }
    Vector3 input = value;
    Matrix inputJ = J;
    if (inverse[k] && kind != 1) {
      input -= translations[edge];
      inputJ.block<3, 3>(0, offset) -= Matrix3::Identity();
    }
    const Vector3 mode = R * input;
    Matrix own = Matrix::Zero(3, n);
    if (random[edge])
      for (int a = 0; a < 3; ++a)
        own.col(offset + 3 + a) = D[a] * input;
    const Matrix3 own_covariance = own * P * own.transpose();
    const Scalar squared = input.dot(input);
    Vector3 radial = Vector3::Zero();
    if (squared.value != 0.)
      radial = Scalar(-.5) * own_covariance.trace() / squared * mode;
    Vector3 mixed = Vector3::Zero();
    if (random[edge])
      for (int a = 0; a < 3; ++a)
        mixed += D[a] * (inputJ * P.row(offset + 3 + a).transpose());
    correction = R * correction + radial + mixed;
    J = R * inputJ + own;
    value = mode;
    if (!inverse[k] && kind != 1) {
      value += translations[edge];
      J.block<3, 3>(0, offset) += Matrix3::Identity();
    }
  }
  Vector mean(dimension);
  mean.head<3>() = value + correction;
  Matrix jacobian(dimension, n);
  jacobian.topRows(3) = J;
  if (kind == 2) {
    Matrix3 R = Matrix3::Identity(), second = Matrix3::Zero();
    Vector4 modal;
    modal << Scalar(1.), Scalar(0.), Scalar(0.), Scalar(0.);
    std::vector<Matrix3> first(n, Matrix3::Zero());
    // first_order.py::pose_query: Q=sum_ab P_ab d_ab R and all source
    // first derivatives share their original coordinate columns.
    for (int k = 0; k < pathn; ++k) {
      const int edge = indices[k], offset = 6 * edge;
      Vector4 edgeq = quaternions[edge];
      Matrix3 r = rotation(edgeq), d[3];
      for (int a = 0; a < 3; ++a)
        d[a] = r * generators[a];
      if (inverse[k]) {
        r.transposeInPlace();
        edgeq.tail<3>() = -edgeq.tail<3>();
        for (auto &v : d)
          v.transposeInPlace();
      }
      Matrix3 next_second = second * r;
      if (random[edge]) {
        for (int b = 0; b < 3; ++b) {
          Matrix3 weighted = Matrix3::Zero();
          for (int a = 0; a < n; ++a)
            weighted +=
                (P(a, offset + 3 + b) + P(offset + 3 + b, a)) * first[a];
          next_second += weighted * d[b];
        }
        Matrix3 local = Matrix3::Zero();
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            local +=
                P(offset + 3 + a, offset + 3 + b) * Scalar(.5) *
                (generators[a] * generators[b] + generators[b] * generators[a]);
        local =
            inverse[k] ? Matrix3(local.transpose() * r) : Matrix3(r * local);
        next_second += R * local;
      }
      std::vector<Matrix3> next_first(n);
      for (int a = 0; a < n; ++a)
        next_first[a] = first[a] * r;
      if (random[edge])
        for (int a = 0; a < 3; ++a)
          next_first[offset + 3 + a] += R * d[a];
      R = R * r;
      first.swap(next_first);
      second = next_second;
      modal = multiply(modal, edgeq);
    }
    for (int a = 0; a < n; ++a)
      jacobian.block<3, 1>(3, a) = vee(R.transpose() * first[a]);
    const Vector3 orientation_correction =
        Scalar(.5) * vee(R.transpose() * second);
    Vector4 inverse_reference = reference.cast<Scalar>();
    inverse_reference.tail<3>() = -inverse_reference.tail<3>();
    const Vector3 phi = rotation_vector(multiply(inverse_reference, modal));
    const Matrix3 D = right_jacobian_inverse(phi);
    mean.tail<3>() = phi + D * orientation_correction;
    jacobian.bottomRows(3) = (D * jacobian.bottomRows(3)).eval();
  }
  Result out{mean, jacobian * P * jacobian.transpose()};
  // Core within+between equals J P J^T on its declared support.  Compute
  // that identical expression directly to avoid differentiating arbitrary
  // support eigensystem bases, retaining every off-diagonal P entry.
  if (projection.rows()) {
    const Matrix A = constant_matrix(projection);
    out.mean = A * out.mean;
    out.covariance = A * out.covariance * A.transpose();
  }
  return out;
}
} // namespace moment_reverse

extern "C" int probtf_moment_reverse_abi_version() { return 1; }
extern "C" int probtf_moment_reverse_evaluate(
    int edges, const double *tin, const double *qin,
    const unsigned char *random, const double *meanin, const double *Pin,
    int pathn, const int *indices, const unsigned char *inverse,
    const double *point, int kind, const double *reference, int projection_rows,
    const double *projection, int cost_kind, const double *target,
    const double *precision, const double *adjoint_mean,
    const double *adjoint_covariance, double *meanout, double *covarianceout,
    double *costout, double *gradient_mean, double *gradient_covariance,
    long long *tape_nodes, char *error, size_t error_size) {
  using namespace moment_reverse;
  try {
    if (edges < 0 || pathn < 0 || kind < 0 || kind > 2 || cost_kind < 0 ||
        cost_kind > 2 || projection_rows < 0)
      throw std::invalid_argument(
          "Invalid propagated-moment derivative dimensions.");
    const int n = 6 * edges, rawdim = kind == 2 ? 6 : 3;
    Eigen::Map<const Row> t(tin, edges, 3), q(qin, edges, 4), rawP(Pin, n, n);
    Eigen::Map<const DoubleVector> rawmean(meanin, n);
    if (!t.allFinite() || !q.allFinite() || !rawP.allFinite() ||
        !rawmean.allFinite())
      throw std::invalid_argument(
          "Propagated moment parameters must be finite.");
    if (n && (rawP - rawP.transpose()).cwiseAbs().maxCoeff() >
                 1e-14 + 1e-12 * rawP.cwiseAbs().maxCoeff())
      throw std::invalid_argument(
          "Propagated moment covariance must be symmetric.");
    for (int k = 0; k < pathn; ++k)
      if (indices[k] < 0 || indices[k] >= edges)
        throw std::invalid_argument(
            "Moment path refers to an absent transform.");
    Tape tape;
    tape.nodes.reserve(200000);
    active_tape = &tape;
    Vector mean(n);
    Matrix P(n, n);
    for (int i = 0; i < n; ++i)
      mean[i] = Scalar::input(rawmean[i]);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        P(i, j) = Scalar::input(rawP(i, j));
    std::vector<Vector3> translations(edges);
    std::vector<Vector4> quaternions(edges);
    std::vector<Matrix3> blocks(2 * edges, Matrix3::Identity());
    for (int e = 0; e < edges; ++e) {
      translations[e] =
          t.row(e).transpose().cast<Scalar>() + mean.segment<3>(6 * e);
      quaternions[e] =
          multiply(q.row(e).transpose().cast<Scalar>(),
                   rotation_quaternion(mean.segment<3>(6 * e + 3)));
      if (random[e])
        blocks[2 * e + 1] = right_jacobian(mean.segment<3>(6 * e + 3));
    }
    // Block form of G P G^T: never discard cross-edge or t/r entries.
    Matrix transported(n, n);
    for (int i = 0; i < 2 * edges; ++i)
      for (int j = 0; j < 2 * edges; ++j)
        transported.block<3, 3>(3 * i, 3 * j) =
            blocks[i] * P.block<3, 3>(3 * i, 3 * j) * blocks[j].transpose();
    Eigen::Map<const Eigen::Vector3d> point_value(point);
    Eigen::Map<const Eigen::Vector4d> reference_value(reference);
    Eigen::Map<const Row> projection_value(projection, projection_rows, rawdim);
    Result result =
        query(translations, quaternions, random, transported, pathn, indices,
              inverse, point_value, kind, reference_value, projection_value);
    const int dimension = result.mean.size();
    Scalar cost(0.);
    if (cost_kind == 1) {
      Eigen::Map<const DoubleVector> y(target, dimension);
      Eigen::Map<const Row> Rinv(precision, dimension, dimension);
      if (!y.allFinite() || !Rinv.allFinite())
        throw std::invalid_argument("Likelihood parameters must be finite.");
      const Vector residual = result.mean - y.cast<Scalar>();
      for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j)
          cost += Scalar(.5 * Rinv(i, j)) *
                  (residual[i] * residual[j] + result.covariance(j, i));
    } else if (cost_kind == 2) {
      Eigen::Map<const DoubleVector> a(adjoint_mean, dimension);
      Eigen::Map<const Row> B(adjoint_covariance, dimension, dimension);
      if (!a.allFinite() || !B.allFinite())
        throw std::invalid_argument("Moment adjoints must be finite.");
      for (int i = 0; i < dimension; ++i) {
        cost += Scalar(a[i]) * result.mean[i];
        for (int j = 0; j < dimension; ++j)
          cost += Scalar(B(i, j)) * result.covariance(i, j);
      }
    }
    const DoubleMatrix output_mean = values(result.mean),
                       output_covariance = values(result.covariance);
    if (!output_mean.allFinite() || !output_covariance.allFinite() ||
        !std::isfinite(cost.value))
      throw std::runtime_error(
          "Propagated moments or likelihood became nonfinite.");
    Eigen::Map<DoubleVector>(meanout, dimension) = output_mean;
    Eigen::Map<Row>(covarianceout, dimension, dimension) = output_covariance;
    *costout = cost.value;
    tape.backward(cost.index);
    Eigen::Map<DoubleVector> gm(gradient_mean, n);
    Eigen::Map<Row> gP(gradient_covariance, n, n);
    for (int i = 0; i < n; ++i)
      gm[i] = tape.nodes[mean[i].index].adjoint;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        gP(i, j) = .5 * (tape.nodes[P(i, j).index].adjoint +
                         tape.nodes[P(j, i).index].adjoint);
    if (!gm.allFinite() || !gP.allFinite())
      throw std::runtime_error("Moment reverse derivative became nonfinite.");
    *tape_nodes = static_cast<long long>(tape.nodes.size());
    active_tape = nullptr;
    return 0;
  } catch (const std::exception &exception) {
    moment_reverse::active_tape = nullptr;
    if (error_size) {
      std::strncpy(error, exception.what(), error_size - 1);
      error[error_size - 1] = 0;
    }
    return 1;
  }
}
