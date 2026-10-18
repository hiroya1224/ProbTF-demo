// Demo-local C++ port of the existing first-order Bingham/body formulas.
// No candidate integration, stochastic sampling, or covariance regularization.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Row =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;
using V4 = Eigen::Vector4d;
using M4 = Eigen::Matrix4d;
static double maximum(const Mat &a) {
  return a.size() ? a.cwiseAbs().maxCoeff() : 0.;
}
static void finite(const Mat &a) {
  if (!a.allFinite())
    throw std::invalid_argument("Distribution parameters must be finite.");
}
static bool close(const Mat &a, const Mat &b, double rt = 1e-8,
                  double at = 1e-12) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         ((a - b).cwiseAbs().array() <= (at + rt * b.cwiseAbs().array())).all();
}
struct Support {
  Mat inverse, basis;
  Vec values;
};
// spherical_law/support.py::covariance_support and support_inverse:
// retain the affine support, using its reciprocal spectrum without jitter.
static Support support(const Mat &input) {
  finite(input);
  if (input.rows() != input.cols())
    throw std::invalid_argument("Covariance must be square.");
  if (maximum(input - input.transpose()) > 1e-14 + 1e-12 * maximum(input))
    throw std::invalid_argument("Covariance must be symmetric.");
  int n = input.rows();
  std::vector<int> active;
  for (int i = 0; i < n; ++i)
    if ((input.row(i).array() != 0.).any() ||
        (input.col(i).array() != 0.).any())
      active.push_back(i);
  Mat reduced(active.size(), active.size());
  for (size_t i = 0; i < active.size(); ++i)
    for (size_t j = 0; j < active.size(); ++j)
      reduced(i, j) =
          .5 * (input(active[i], active[j]) + input(active[j], active[i]));
  Support out;
  out.inverse = Mat::Zero(n, n);
  if (active.empty()) {
    out.basis = Mat(n, 0);
    out.values = Vec(0);
    return out;
  }
  Eigen::SelfAdjointEigenSolver<Mat> eig(reduced);
  if (eig.info() != Eigen::Success)
    throw std::runtime_error("Covariance eigensystem unavailable.");
  double tol = 1e-14 + 1e-12 * eig.eigenvalues().cwiseAbs().maxCoeff();
  int rank = 0;
  for (int i = 0; i < eig.eigenvalues().size(); ++i) {
    if (eig.eigenvalues()[i] < -tol)
      throw std::invalid_argument(
          "Covariance has a significant negative eigenvalue.");
    if (eig.eigenvalues()[i] > tol)
      ++rank;
  }
  out.basis = Mat::Zero(n, rank);
  out.values = Vec(rank);
  int k = 0;
  for (int j = 0; j < eig.eigenvalues().size(); ++j)
    if (eig.eigenvalues()[j] > tol) {
      out.values[k] = eig.eigenvalues()[j];
      for (size_t i = 0; i < active.size(); ++i)
        out.basis(active[i], k) = eig.eigenvectors()(i, j);
      ++k;
    }
  if (rank)
    out.inverse = out.basis * out.values.cwiseInverse().asDiagonal() *
                  out.basis.transpose();
  return out;
}
static V4 normq(const V4 &q) {
  if (!q.allFinite() || q.norm() < 1e-12)
    throw std::invalid_argument("Quaternion must be finite and nonzero.");
  return q / q.norm();
}
static M4 left(const V4 &input) {
  auto q = normq(input);
  double w = q[0], x = q[1], y = q[2], z = q[3];
  M4 a;
  a << w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w;
  return a;
}
static V4 mul(const V4 &a, const V4 &b) { return normq(left(a) * normq(b)); }
static M3 skew(const V3 &v) {
  M3 s;
  s << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return s;
}
static V4 rvq(const V3 &v) {
  double angle = v.norm();
  V4 q;
  if (angle < 1e-6) {
    double scale = .5 - angle * angle / 48. + std::pow(angle, 4) / 3840.;
    q << 1. - angle * angle / 8., scale * v[0], scale * v[1], scale * v[2];
    return normq(q);
  }
  q << std::cos(.5 * angle), std::sin(.5 * angle) * v[0] / angle,
      std::sin(.5 * angle) * v[1] / angle, std::sin(.5 * angle) * v[2] / angle;
  return normq(q);
}
static M3 rotation(const V4 &input) {
  auto q = normq(input);
  double w = q[0], x = q[1], y = q[2], z = q[3];
  M3 R;
  R << w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z;
  return R;
}
static V4 rotq(const M3 &R) {
  Eigen::Quaterniond q(R);
  V4 v;
  v << q.w(), q.x(), q.y(), q.z();
  return normq(v);
}
static M3 rightj(const V3 &v) {
  double a = v.norm();
  M3 S = skew(v);
  if (a < 1e-6)
    return M3::Identity() - .5 * S + (1. / 6.) * S * S;
  return M3::Identity() - (1 - std::cos(a)) / (a * a) * S +
         (a - std::sin(a)) / (a * a * a) * S * S;
}
// A continuous rotation covariance/modal gap needs an actual PD inverse.
// Do not use the Gaussian affine-support rank cutoff to discard a positive
// but small axis of this required 3-by-3 inverse. No jitter or floor is used.
static M3 positive_inverse3(const M3 &input, const char *name) {
  finite(input);
  if (!close(input, input.transpose(), 1e-12, 1e-12))
    throw std::invalid_argument(std::string(name) + " must be symmetric.");
  const M3 symmetric = .5 * input + .5 * input.transpose();
  Eigen::LLT<M3> factor(symmetric);
  if (factor.info() != Eigen::Success)
    throw std::invalid_argument(std::string(name) +
                                " requires a positive definite inverse.");
  const M3 inverse = factor.solve(M3::Identity());
  finite(inverse);
  return .5 * inverse + .5 * inverse.transpose();
}
// first_order.py::bingham_rotation_coordinates:
// C_A = 2 (lambda_max I - N(q)^T A N(q))^{-1}, N(q)=L(q)[:,1:4].
static std::pair<V4, M3> coordinates(const M4 &input) {
  finite(input);
  if (!close(input, input.transpose(), 1e-12, 1e-12))
    throw std::invalid_argument("Bingham parameter must be symmetric.");
  M4 A = input - .25 * input.trace() * M4::Identity();
  Eigen::SelfAdjointEigenSolver<M4> e(A);
  if (e.info() != Eigen::Success)
    throw std::runtime_error("Bingham eigensystem unavailable.");
  if (!e.eigenvalues().allFinite())
    throw std::invalid_argument("Bingham eigenvalues must be finite.");
  if (!(e.eigenvalues()[3] > e.eigenvalues()[2]))
    throw std::invalid_argument(
        "A single-pose proxy needs a simple maximum Bingham eigenvalue.");
  V4 q = normq(e.eigenvectors().col(3));
  Eigen::Index pivot;
  q.cwiseAbs().maxCoeff(&pivot);
  if (q[pivot] < 0.)
    q = -q;
  auto N = left(q).rightCols<3>().eval();
  // A is validated symmetric above, so N^T A N is symmetric analytically.
  // Assemble that structural symmetry before the required PD inverse, as
  // the core does. Product-roundoff skew is not an invalid input law.
  const M3 raw_gap =
      e.eigenvalues()[3] * M3::Identity() - N.transpose() * A * N;
  const M3 gap = .5 * raw_gap + .5 * raw_gap.transpose();
  M3 C = 2 * positive_inverse3(gap, "Bingham modal gap");
  // Concentration is an approximation-quality diagnostic, not an algebraic
  // prerequisite. The user explicitly removed the former 0.1 rad^2 gate;
  // retain the same finite coordinates and covariance without clipping.
  return {q, C};
}
static void error(char *out, size_t n, const std::exception &e) {
  if (n) {
    std::strncpy(out, e.what(), n - 1);
    out[n - 1] = 0;
  }
}
extern "C" int probtf_body_abi_version() { return 1; }
// first_order.py::reexpress_state: move all means simultaneously, transport
// the complete covariance as G P G^T, then use A=-2 N Omega^{-1} N^T.
static int reexpress_kernel(int edges, const double *tin, const double *qin,
                            const unsigned char *random, const double *sourceP,
                            const double *mean, const double *posteriorP,
                            double *tout, double *qout, double *Aout,
                            double *Pout, char *err, size_t errn,
                            bool check_source_support) {
  try {
    if (edges < 0)
      throw std::invalid_argument("A transform count cannot be negative.");
    int n = 6 * edges;
    Eigen::Map<const Row> t(tin, edges, 3), qs(qin, edges, 4), P(sourceP, n, n),
        C(posteriorP, n, n);
    Eigen::Map<const Vec> m(mean, n);
    finite(t);
    finite(qs);
    finite(m);
    auto post = support(C);
    (void)post;
    if (check_source_support) {
      auto src = support(P);
      Vec residual = m - src.basis * (src.basis.transpose() * m);
      if (residual.norm() > 1e-10 * std::max(1., m.norm()))
        throw std::invalid_argument("posterior body mean shift lies outside the "
                                    "reference affine support.");
    }
    Mat G = Mat::Identity(n, n);
    for (int i = 0; i < edges; ++i)
      if (random[i])
        G.block<3, 3>(6 * i + 3, 6 * i + 3) = rightj(m.segment<3>(6 * i + 3));
    Mat transported = G * C * G.transpose();
    Eigen::Map<Row> to(tout, edges, 3), qo(qout, edges, 4), Po(Pout, n, n);
    Po = transported;
    for (int i = 0; i < edges; ++i) {
      to.row(i) = t.row(i) + m.segment<3>(6 * i).transpose();
      V4 q = mul(qs.row(i).transpose(), rvq(m.segment<3>(6 * i + 3)));
      const M3 raw_Rc = transported.block<3, 3>(6 * i + 3, 6 * i + 3);
      // The supplied C was validated before computing G C G^T. Preserve
      // that complete transported matrix verbatim; only its internally
      // assembled rotation quadratic is symmetrized for the PD inverse.
      const M3 Rc = .5 * raw_Rc + .5 * raw_Rc.transpose();
      Eigen::Map<Row> A(Aout + 16 * i, 4, 4);
      A.setZero();
      if (random[i]) {
        const M3 inverse = positive_inverse3(Rc, "rotation covariance");
        auto N = left(q).rightCols<3>().eval();
        M4 parameter = -2 * N * inverse * N.transpose();
        parameter = .5 * (parameter + parameter.transpose()).eval();
        A = parameter;
        auto b = coordinates(parameter);
        q = b.first;
        if (!close(Rc, b.second))
          throw std::invalid_argument(
              "The Bingham and joint rotation marginal disagree.");
      } else if ((raw_Rc.array() != 0.).any())
        throw std::invalid_argument(
            "A known fixed rotation cannot carry a random rotation block.");
      qo.row(i) = q.transpose();
      support(transported.block<3, 3>(6 * i, 6 * i));
    }
    support(transported);
    return 0;
  } catch (const std::exception &e) {
    error(err, errn, e);
    return 1;
  }
}
// Existing ABI and strict Bayesian-posterior support contract are unchanged.
extern "C" int probtf_body_reexpress(
    int edges, const double *tin, const double *qin, const unsigned char *random,
    const double *sourceP, const double *mean, const double *posteriorP,
    double *tout, double *qout, double *Aout, double *Pout, char *err,
    size_t errn) {
  return reexpress_kernel(edges, tin, qin, random, sourceP, mean, posteriorP,
                          tout, qout, Aout, Pout, err, errn, true);
}
// A common coordinate chart is geometry, not the support of the OTHER law
// being expressed there. This separate extension omits only that unrelated
// reference-prior absolute-continuity test. All supplied-covariance, finite,
// fixed-rotation and required-PD-inverse checks remain in the shared kernel.
extern "C" int probtf_body_materialize_in_chart_abi_version() { return 1; }
extern "C" int probtf_body_materialize_in_chart(
    int edges, const double *tin, const double *qin, const unsigned char *random,
    const double *sourceP, const double *mean, const double *posteriorP,
    double *tout, double *qout, double *Aout, double *Pout, char *err,
    size_t errn) {
  return reexpress_kernel(edges, tin, qin, random, sourceP, mean, posteriorP,
                          tout, qout, Aout, Pout, err, errn, false);
}
// first_order.py::conditional_total_covariance: use the support-aware Schur
// within/between sum as the returned covariance; J P J^T is only its check.
static void covariance(const Mat &P, const Mat &J, Mat &out, Mat &within,
                       Mat &between) {
  int n = P.rows(), split = std::min(6, n);
  Mat p11 = P.topLeftCorner(split, split),
      p21 = P.bottomLeftCorner(n - split, split);
  auto s = support(p11);
  Mat reg = p21 * s.inverse;
  Mat cond =
      P.bottomRightCorner(n - split, n - split) - reg * p11 * reg.transpose();
  cond = .5 * (cond + cond.transpose()).eval();
  support(cond);
  Mat j1 = J.leftCols(split), j2 = J.rightCols(n - split), eff = j1 + j2 * reg;
  within = j2 * cond * j2.transpose();
  between = eff * p11 * eff.transpose();
  out = within + between;
  if (!close(out, J * P * J.transpose()))
    throw std::invalid_argument(
        "Conditional total covariance disagrees with J P J.T.");
}
static V3 vee(const M3 &M) {
  V3 v;
  v << M(2, 1) - M(1, 2), M(0, 2) - M(2, 0), M(1, 0) - M(0, 1);
  return .5 * v;
}
extern "C" int
probtf_body_query(int edges, const double *tin, const double *qin,
                  const unsigned char *random, const double *Pin, int pathn,
                  const int *indices, const unsigned char *inverse,
                  const double *point, int kind, double *meanout,
                  double *modeout, double *correctionout, double *Jout,
                  double *Cout, double *crossout, double *withinout,
                  double *betweenout, double *qout, char *err, size_t errn) {
  try {
    if (edges < 0 || pathn < 0 || kind < 0 || kind > 2)
      throw std::invalid_argument("Invalid body query arguments.");
    int n = 6 * edges, dim = kind == 2 ? 6 : 3;
    Eigen::Map<const Row> ts(tin, edges, 3), qs(qin, edges, 4), P(Pin, n, n);
    finite(ts);
    finite(qs);
    finite(P);
    V3 value = Eigen::Map<const V3>(point), correction = V3::Zero();
    Mat J = Mat::Zero(3, n);
    M3 S[3] = {skew(V3::UnitX()), skew(V3::UnitY()), skew(V3::UnitZ())};
    // first_order.py::_point_jet: propagate mode, first derivative, and the
    // separate O(P) radial/mixed contraction on the original shared P.
    // Corrections are never multiplied together into higher-order P terms.
    for (int k = pathn - 1; k >= 0; --k) {
      int e = indices[k];
      if (e < 0 || e >= edges)
        throw std::invalid_argument("Path refers to an absent transform.");
      M3 R = rotation(qs.row(e).transpose());
      M3 D[3];
      for (int a = 0; a < 3; ++a)
        D[a] = R * S[a];
      if (inverse[k]) {
        R.transposeInPlace();
        for (int a = 0; a < 3; ++a)
          D[a].transposeInPlace();
      }
      V3 input = value;
      Mat IJ = J;
      if (inverse[k] && kind != 1) {
        input -= ts.row(e).transpose();
        IJ.block<3, 3>(0, 6 * e) -= M3::Identity();
      }
      V3 mode = R * input;
      Mat own = Mat::Zero(3, n);
      if (random[e])
        for (int a = 0; a < 3; ++a)
          own.col(6 * e + 3 + a) = D[a] * input;
      Mat owncov = own * P * own.transpose();
      V3 radial = input.squaredNorm() == 0.
                      ? V3::Zero()
                      : V3(-.5 * owncov.trace() / input.squaredNorm() * mode);
      V3 mixed = V3::Zero();
      if (random[e])
        for (int a = 0; a < 3; ++a)
          mixed += D[a] * (IJ * P.row(6 * e + 3 + a).transpose());
      correction = R * correction + radial + mixed;
      J = R * IJ + own;
      value = mode;
      if (!inverse[k] && kind != 1) {
        value += ts.row(e).transpose();
        J.block<3, 3>(0, 6 * e) += M3::Identity();
      }
    }
    Vec mode(dim), cor(dim);
    Mat jac(dim, n);
    mode.head<3>() = value;
    cor.head<3>() = correction;
    jac.topRows(3) = J;
    V4 q = V4(1, 0, 0, 0);
    if (kind == 2) {
      // first_order.py::pose_query: contract Q=sum_ab P_ab d_ab R during
      // composition; keep every shared-variable first derivative explicit.
      M3 R = M3::Identity(), second = M3::Zero();
      std::vector<M3> first(n, M3::Zero());
      for (int k = 0; k < pathn; ++k) {
        int e = indices[k];
        M3 r = rotation(qs.row(e).transpose()), d[3];
        for (int a = 0; a < 3; ++a)
          d[a] = r * S[a];
        if (inverse[k]) {
          r.transposeInPlace();
          for (int a = 0; a < 3; ++a)
            d[a].transposeInPlace();
        }
        M3 nextsecond = second * r;
        if (random[e]) {
          for (int b = 0; b < 3; ++b) {
            M3 weighted = M3::Zero();
            for (int a = 0; a < n; ++a)
              weighted +=
                  (P(a, 6 * e + 3 + b) + P(6 * e + 3 + b, a)) * first[a];
            nextsecond += weighted * d[b];
          }
          M3 local = M3::Zero();
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              local += P(6 * e + 3 + a, 6 * e + 3 + b) * .5 *
                       (S[a] * S[b] + S[b] * S[a]);
          local = inverse[k] ? M3(local.transpose() * r) : M3(r * local);
          nextsecond += R * local;
        }
        std::vector<M3> nextfirst(n);
        for (int a = 0; a < n; ++a)
          nextfirst[a] = first[a] * r;
        if (random[e])
          for (int a = 0; a < 3; ++a)
            nextfirst[6 * e + 3 + a] += R * d[a];
        R = R * r;
        first.swap(nextfirst);
        second = nextsecond;
      }
      for (int a = 0; a < n; ++a)
        jac.block<3, 1>(3, a) = vee(R.transpose() * first[a]);
      mode.tail<3>().setZero();
      // The modal right-chart correction is 1/2 vee(R^T Q).
      cor.tail<3>() = .5 * vee(R.transpose() * second);
      q = rotq(R);
    }
    Mat out, within, between;
    covariance(P, jac, out, within, between);
    Eigen::Map<Vec>(meanout, dim) = mode + cor;
    Eigen::Map<Vec>(modeout, dim) = mode;
    Eigen::Map<Vec>(correctionout, dim) = cor;
    Eigen::Map<Row>(Jout, dim, n) = jac;
    Eigen::Map<Row>(Cout, dim, dim) = out;
    Eigen::Map<Row>(withinout, dim, dim) = within;
    Eigen::Map<Row>(betweenout, dim, dim) = between;
    Eigen::Map<Row>(crossout, n, dim) = P * jac.transpose();
    Eigen::Map<V4> qo(qout);
    qo = q;
    return 0;
  } catch (const std::exception &e) {
    error(err, errn, e);
    return 1;
  }
}

// Independent extension API: the existing first-order distribution functions
// above keep their ABI and equations. This evaluates finite modal geometry,
// not a push-forward mean, covariance, or candidate-law reexpression.
extern "C" int probtf_body_finite_modal_abi_version() { return 1; }

extern "C" int probtf_body_finite_modal_path(
    int edges, const double *tin, const double *qin, const double *delta_in,
    int pathn, const int *indices, const unsigned char *inverse, double *tout,
    double *qout, double *Jtout, double *Jwout, char *err, size_t errn) {
  try {
    if (edges < 0 || pathn < 0)
      throw std::invalid_argument("Invalid finite modal path dimensions.");
    const int n = 6 * edges;
    Eigen::Map<const Row> translations(tin, edges, 3),
        quaternions(qin, edges, 4);
    Eigen::Map<const Vec> delta(delta_in, n);
    finite(translations);
    finite(quaternions);
    finite(delta);

    V3 translation = V3::Zero();
    M3 rotation_total = M3::Identity();
    Mat Jt = Mat::Zero(3, n), Jw = Mat::Zero(3, n);
    for (int k = 0; k < pathn; ++k) {
      const int edge = indices[k];
      if (edge < 0 || edge >= edges)
        throw std::invalid_argument(
            "Finite modal path refers to an absent transform.");
      const int offset = 6 * edge;
      const V3 shift = delta.segment<3>(offset + 3);
      const M3 Jr = rightj(shift);
      const M3 forward_rotation =
          rotation(mul(quaternions.row(edge).transpose(), rvq(shift)));
      const V3 forward_translation =
          translations.row(edge).transpose() + delta.segment<3>(offset);

      M3 Q = forward_rotation;
      V3 a = forward_translation;
      M3 Ja_translation = M3::Identity(), Ja_rotation = M3::Zero();
      M3 JQ_rotation = Jr;
      if (inverse[k]) {
        Q = forward_rotation.transpose();
        a = -Q * forward_translation;
        // d(-R^T t) = -R^T dt + [a]_x Jr(delta_r) d(delta_r).
        // The inverse rotation's right perturbation is -R Jr(delta_r)dδr.
        Ja_translation = -Q;
        Ja_rotation = skew(a) * Jr;
        JQ_rotation = -forward_rotation * Jr;
      }

      // Exact product rule on T_total * (a,Q), using 3-by-n derivatives:
      // Jt' = Jt - R_total[a]_x Jw + R_total Ja,
      // Jw' = Q^T Jw + JQ, with dR = R[Jw dδ]_x.
      // Repeated appearances share the same offset; no variable is copied.
      Mat next_Jt = Jt - rotation_total * skew(a) * Jw;
      next_Jt.block<3, 3>(0, offset) += rotation_total * Ja_translation;
      next_Jt.block<3, 3>(0, offset + 3) += rotation_total * Ja_rotation;
      Mat next_Jw = Q.transpose() * Jw;
      next_Jw.block<3, 3>(0, offset + 3) += JQ_rotation;
      translation += rotation_total * a;
      rotation_total = rotation_total * Q;
      Jt.swap(next_Jt);
      Jw.swap(next_Jw);
    }
    finite(translation);
    finite(rotation_total);
    finite(Jt);
    finite(Jw);
    const V4 quaternion = rotq(rotation_total);
    Eigen::Map<V3> translated(tout);
    Eigen::Map<V4> rotated(qout);
    translated = translation;
    rotated = quaternion;
    Eigen::Map<Row>(Jtout, 3, n) = Jt;
    Eigen::Map<Row>(Jwout, 3, n) = Jw;
    return 0;
  } catch (const std::exception &e) {
    error(err, errn, e);
    return 1;
  }
}
