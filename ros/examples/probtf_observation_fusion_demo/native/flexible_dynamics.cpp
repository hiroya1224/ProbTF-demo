#include "flexible_dynamics.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
constexpr double position_tolerance = 2e-11;
constexpr double contact_velocity_tolerance = 2e-11;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
struct TraceRow { double t; Vector q, v; };
struct Midpoint { Vector q, v, qm, vm, um; double residual; bool contact; };

void require(bool condition, const std::string &message) {
  if (!condition) throw Failure(message);
}
Vector read_vector(const double *value, int n, const char *name) {
  require(value != nullptr, std::string(name) + " pointer is null");
  Vector result = Eigen::Map<const Vector>(value, n);
  require(result.allFinite(), std::string(name) + " contains non-finite values");
  return result;
}
double maximum_absolute(const Vector &x) { return x.size() ? x.cwiseAbs().maxCoeff() : 0.; }
ProbtfDynamicsOptions defaults() {
  ProbtfDynamicsOptions result;
  result.max_step_s = .001;
  result.min_step_s = 1e-9;
  result.absolute_residual_tolerance = 1e-10;
  result.relative_residual_tolerance = 1e-9;
  result.max_newton_iterations = 20;
  result.max_line_search_steps = 16;
  result.max_step_halvings = 20;
  return result;
}

struct Dynamics {
  pinocchio::Model model;
  std::unique_ptr<pinocchio::Data> data;
  int n;
  Vector stiffness, damping, lower, upper, q, v, a, command;
  Matrix reference_mass;
  double zeta, time = 0.;
  bool explicit_damping, initialized = false, parameters_fixed = false;
  ProbtfDynamicsOptions options;
  std::array<double, PROBTF_DYN_DOUBLE_DIAGNOSTIC_COUNT> diagnostic{};
  std::array<uint64_t, PROBTF_DYN_COUNT_DIAGNOSTIC_COUNT> counts{};
  std::vector<TraceRow> trace;
  std::string error;

  Dynamics(const char *urdf, int dimension, const double *k, const double *d,
           double damping_ratio, const ProbtfDynamicsOptions *settings)
      : n(dimension), zeta(damping_ratio), explicit_damping(d != nullptr),
        options(settings ? *settings : defaults()) {
    require(urdf && *urdf, "URDF path is empty");
    require(n >= 1 && n <= 6, "native dynamics supports 1..6 scalar-revolute joints");
    require(std::isfinite(zeta) && zeta >= 0., "damping ratio must be finite and nonnegative");
    require(std::isfinite(options.max_step_s) && options.max_step_s > 0. &&
            std::isfinite(options.min_step_s) && options.min_step_s > 0. &&
            options.min_step_s <= options.max_step_s, "invalid integrator step limits");
    require(std::isfinite(options.absolute_residual_tolerance) && options.absolute_residual_tolerance > 0. &&
            std::isfinite(options.relative_residual_tolerance) && options.relative_residual_tolerance >= 0.,
            "invalid residual tolerances");
    require(options.max_newton_iterations > 0 && options.max_line_search_steps > 0 &&
            options.max_step_halvings >= 0, "invalid integrator iteration budgets");
    stiffness = read_vector(k, n, "stiffness");
    require((stiffness.array() > 0.).all(), "stiffness must be strictly positive");
    damping = d ? read_vector(d, n, "damping") : Vector::Zero(n);
    require((damping.array() >= 0.).all(), "damping must be nonnegative");
    pinocchio::urdf::buildModel(std::string(urdf), model);
    require(model.nq == n && model.nv == n, "URDF nq/nv differ from supplied dimension");
    for (pinocchio::JointIndex j = 1; j < model.joints.size(); ++j) {
      require(model.joints[j].nq() == 1 && model.joints[j].nv() == 1 &&
              model.joints[j].shortname().find("JointModelR") == 0,
              "URDF must contain only bounded scalar-revolute moving joints");
    }
    model.gravity.linear() << 0., 0., -9.81;
    lower = model.lowerPositionLimit;
    upper = model.upperPositionLimit;
    require(lower.allFinite() && upper.allFinite() && (lower.array() < upper.array()).all(),
            "URDF requires finite, ordered joint position limits");
    data.reset(new pinocchio::Data(model));
    q = v = a = command = Vector::Zero(n);
    reference_mass = Matrix::Zero(n, n);
  }

  void bounded(const Vector &x, const char *name) const {
    require((x.array() >= lower.array()).all() && (x.array() <= upper.array()).all(),
            std::string(name) + " violates URDF joint position limits");
  }
  Matrix mass(const Vector &position) {
    Matrix result = pinocchio::crba(model, *data, position);
    result.triangularView<Eigen::StrictlyLower>() = result.transpose().triangularView<Eigen::StrictlyLower>();
    require(result.allFinite(), "non-finite mass matrix");
    Eigen::LLT<Matrix> factor(result);
    require(factor.info() == Eigen::Success, "mass matrix is not positive definite");
    return result;
  }
  Vector bias(const Vector &position, const Vector &velocity) {
    Vector result = pinocchio::rnea(model, *data, position, velocity, Vector::Zero(n));
    require(result.allFinite(), "non-finite RNEA bias torque");
    return result;
  }
  std::vector<std::pair<int, double>> contacts(const Vector &position, const Vector &velocity,
                                               bool require_rest) const {
    std::vector<std::pair<int, double>> result;
    for (int j = 0; j < n; ++j) {
      if (require_rest && std::abs(velocity[j]) > contact_velocity_tolerance) continue;
      if (position[j] <= lower[j] + position_tolerance) result.emplace_back(j, -1.);
      else if (position[j] >= upper[j] - position_tolerance) result.emplace_back(j, 1.);
    }
    return result;
  }

  // Exact convex projection in the M metric. At most six contact inequalities:
  // enumerate their KKT active sets; no componentwise velocity clipping.
  Vector cone_projection(const Matrix &m, const Vector &input,
                         const std::vector<std::pair<int, double>> &contact) {
    if (contact.empty()) return input;
    Matrix normals = Matrix::Zero(contact.size(), n);
    for (size_t j = 0; j < contact.size(); ++j)
      normals(j, contact[j].first) = contact[j].second;
    Eigen::LLT<Matrix> m_factor(m);
    require(m_factor.info() == Eigen::Success, "contact mass factorization failed");
    const double tolerance = 2e-11 * (1. + input.norm());
    double best_cost = std::numeric_limits<double>::infinity();
    Vector best;
    for (unsigned mask = 0; mask < (1u << contact.size()); ++mask) {
      int rows = 0;
      for (size_t j = 0; j < contact.size(); ++j) if (mask & (1u << j)) ++rows;
      Vector candidate = input;
      if (rows) {
        Matrix active(rows, n);
        int row = 0;
        for (size_t j = 0; j < contact.size(); ++j)
          if (mask & (1u << j)) active.row(row++) = normals.row(j);
        Matrix inverse_normals = m_factor.solve(active.transpose());
        Matrix gram = active * inverse_normals;
        Eigen::LLT<Matrix> gram_factor(gram);
        if (gram_factor.info() != Eigen::Success) continue;
        Vector multiplier = gram_factor.solve(active * input);
        if ((multiplier.array() < -tolerance).any()) continue;
        candidate -= inverse_normals * multiplier;
      }
      if ((normals * candidate).maxCoeff() > tolerance) continue;
      const Vector difference = candidate - input;
      const double cost = difference.dot(m * difference);
      if (cost < best_cost) { best_cost = cost; best = candidate; }
    }
    require(best.size() == n && best.allFinite(), "no feasible mass-metric contact projection");
    return best;
  }

  Vector endpoint_acceleration() {
    const Matrix m = mass(q);
    const Vector force = (stiffness.array() * (command - q).array() - damping.array() * v.array()).matrix() - bias(q, v);
    Vector unconstrained = m.llt().solve(force);
    require(unconstrained.allFinite(), "non-finite forward acceleration");
    return cone_projection(m, unconstrained, contacts(q, v, true));
  }
  std::array<double, 3> energy() {
    const double kinetic = pinocchio::computeKineticEnergy(model, *data, q, v);
    const double gravity = pinocchio::computePotentialEnergy(model, *data, q);
    const double spring = .5 * ((q - command).array().square() * stiffness.array()).sum();
    require(std::isfinite(kinetic) && std::isfinite(gravity) && std::isfinite(spring), "non-finite physical energy");
    return {{kinetic, gravity, spring}};
  }
  void energy_diagnostic() {
    const auto e = energy();
    diagnostic[PROBTF_DYN_TIME_S] = time;
    diagnostic[PROBTF_DYN_KINETIC_ENERGY_J] = e[0];
    diagnostic[PROBTF_DYN_GRAVITY_ENERGY_J] = e[1];
    diagnostic[PROBTF_DYN_SPRING_ENERGY_J] = e[2];
    diagnostic[PROBTF_DYN_TOTAL_ENERGY_J] = e[0] + e[1] + e[2];
  }
  void reset(const double *position, const double *velocity, const double *u) {
    const Vector next_q = read_vector(position, n, "position");
    const Vector next_v = read_vector(velocity, n, "velocity");
    const Vector next_u = read_vector(u, n, "command");
    bounded(next_q, "position"); bounded(next_u, "command");
    // Compute all potentially failing initial model quantities before committing.
    const Matrix next_mass = mass(next_q);
    Vector next_damping = damping;
    if (!parameters_fixed && !explicit_damping)
      next_damping = (2. * zeta * (stiffness.array() * next_mass.diagonal().array()).sqrt()).matrix();
    require(next_damping.allFinite(), "invalid initial fixed damping");
    // An outward velocity at a hard stop is an inconsistent reset, not an
    // undocumented impulse. advance() handles impacts reached by integration.
    for (const auto &contact : contacts(next_q, next_v, false))
      require(contact.second * next_v[contact.first] <= contact_velocity_tolerance,
              "reset velocity points outward through a joint stop");
    const Vector saved_q=q, saved_v=v, saved_a=a, saved_u=command, saved_d=damping;
    const Matrix saved_mass=reference_mass;
    const bool saved_initialized=initialized, saved_fixed=parameters_fixed;
    const double saved_time=time;
    const auto saved_diagnostic=diagnostic; const auto saved_counts=counts;
    const auto saved_trace=trace;
    try {
      q = next_q; v = next_v; command = next_u; damping = next_damping;
      if (!parameters_fixed) reference_mass = next_mass;
      a = endpoint_acceleration();
      parameters_fixed = initialized = true;
      time = 0.; diagnostic.fill(0.); counts.fill(0);
      trace.clear(); trace.push_back({0., q, v});
      energy_diagnostic();
    } catch (...) {
      q=saved_q; v=saved_v; a=saved_a; command=saved_u; damping=saved_d;
      reference_mass=saved_mass; initialized=saved_initialized; parameters_fixed=saved_fixed;
      time=saved_time; diagnostic=saved_diagnostic; counts=saved_counts; trace=saved_trace;
      throw;
    }
  }

  Vector residual(const Vector &position, const Vector &velocity,
                  const Vector &next_v, double h, const Vector &mid_command,
                  Vector *qm_out = nullptr, Vector *vm_out = nullptr) {
    Vector qm = position + .25 * h * (velocity + next_v);
    Vector vm = .5 * (velocity + next_v);
    Vector acceleration = (next_v - velocity) / h;
    Vector value = pinocchio::rnea(model, *data, qm, vm, acceleration);
    value.array() += damping.array() * vm.array() + stiffness.array() * (qm - mid_command).array();
    require(value.allFinite(), "non-finite implicit midpoint residual");
    if (qm_out) *qm_out = qm;
    if (vm_out) *vm_out = vm;
    return value;
  }
  double free_norm(const Vector &value, const std::vector<int> &free) const {
    double result = 0.; for (int j : free) result = std::max(result, std::abs(value[j])); return result;
  }
  Midpoint solve_midpoint(const Vector &position, const Vector &velocity, double h,
                           const Vector &mid_command) {
    auto contact = contacts(position, velocity, true);
    std::string last_failure = "no feasible resting-contact active set";
    for (unsigned mask = 0; mask < (1u << contact.size()); ++mask) {
      std::vector<int> free;
      std::vector<bool> active(n, false);
      for (size_t j = 0; j < contact.size(); ++j)
        if (mask & (1u << j)) active[contact[j].first] = true;
      for (int j = 0; j < n; ++j) if (!active[j]) free.push_back(j);
      Vector next_v = velocity;
      for (int j = 0; j < n; ++j) if (active[j]) next_v[j] = -velocity[j];
      try {
        bool converged = false;
        Vector value;
        double tolerance = 0.;
        for (int iteration = 0; iteration <= options.max_newton_iterations; ++iteration) {
          value = residual(position, velocity, next_v, h, mid_command);
          const double force_scale = std::max(1., std::max(maximum_absolute((stiffness.array() * (position - mid_command).array()).matrix()),
                                                        maximum_absolute((damping.array() * velocity.array()).matrix())));
          tolerance = options.absolute_residual_tolerance + options.relative_residual_tolerance * force_scale;
          if (free_norm(value, free) <= tolerance) { converged = true; break; }
          require(iteration < options.max_newton_iterations, "implicit midpoint Newton budget exhausted");
          const Vector qm = position + .25 * h * (velocity + next_v);
          const Vector vm = .5 * (velocity + next_v);
          const Vector acceleration = (next_v - velocity) / h;
          Matrix dq = Matrix::Zero(n,n), dv = Matrix::Zero(n,n), da = Matrix::Zero(n,n);
          pinocchio::computeRNEADerivatives(model, *data, qm, vm, acceleration, dq, dv, da);
          da.triangularView<Eigen::StrictlyLower>() = da.transpose().triangularView<Eigen::StrictlyLower>();
          Matrix jacobian = da / h + .5 * dv + .25 * h * dq;
          jacobian.diagonal().array() += .5 * damping.array() + .25 * h * stiffness.array();
          Matrix reduced(free.size(), free.size()); Vector rhs(free.size());
          for (size_t row = 0; row < free.size(); ++row) {
            rhs[row] = -value[free[row]];
            for (size_t col = 0; col < free.size(); ++col) reduced(row,col) = jacobian(free[row],free[col]);
          }
          require(reduced.allFinite(), "non-finite midpoint Newton Jacobian");
          Eigen::FullPivLU<Matrix> factor(reduced);
          require(factor.isInvertible(), "singular midpoint Newton Jacobian");
          const Vector update = factor.solve(rhs);
          require(update.allFinite(), "non-finite midpoint Newton direction");
          ++counts[PROBTF_DYN_NEWTON_ITERATIONS];
          const double old_norm = free_norm(value, free);
          bool accepted = false; double alpha = 1.;
          for (int trial = 0; trial < options.max_line_search_steps; ++trial, alpha *= .5) {
            Vector candidate = next_v;
            for (size_t row = 0; row < free.size(); ++row) candidate[free[row]] += alpha * update[row];
            const Vector candidate_value = residual(position, velocity, candidate, h, mid_command);
            ++counts[PROBTF_DYN_LINE_SEARCH_EVALUATIONS];
            const double candidate_norm = free_norm(candidate_value, free);
            if (candidate_norm <= tolerance || candidate_norm <= (1. - 1e-4 * alpha) * old_norm) {
              next_v = candidate; accepted = true; break;
            }
          }
          require(accepted, "midpoint residual line search failed");
        }
        require(converged, "implicit midpoint did not converge");
        bool feasible = true;
        const double contact_tolerance = std::max(tolerance, 2e-10);
        for (const auto &c : contact) {
          if (active[c.first]) { if (c.second * value[c.first] > contact_tolerance) feasible = false; }
          else if (c.second * next_v[c.first] > contact_velocity_tolerance) feasible = false;
        }
        if (!feasible) continue;
        Midpoint result;
        result.q = position + .5 * h * (velocity + next_v);
        result.v = next_v;
        result.qm = position + .25 * h * (velocity + next_v);
        result.vm = .5 * (velocity + next_v);
        result.um = mid_command;
        result.residual = free_norm(value, free);
        result.contact = mask != 0;
        require(result.q.allFinite() && result.v.allFinite(), "non-finite midpoint state");
        return result;
      } catch (const Failure &exception) { last_failure = exception.what(); }
    }
    throw Failure(last_failure);
  }
  bool inside(const Vector &position) const {
    return (position.array() >= lower.array()).all() && (position.array() <= upper.array()).all();
  }

  void advance(double dt, const double *start, const double *end) {
    require(initialized, "reset must precede advance");
    require(std::isfinite(dt) && dt >= 0., "advance duration must be finite and nonnegative");
    const Vector u0 = read_vector(start, n, "u_start");
    const Vector u1 = read_vector(end, n, "u_end");
    bounded(u0, "u_start"); bounded(u1, "u_end");
    require((u0.array() == command.array()).all(), "u_start differs from preceding accepted command");
    if (dt == 0.) {
      require((u0.array() == u1.array()).all(), "zero-duration advance cannot change the command");
      trace.clear(); trace.push_back({0.,q,v});
      return; // Preserve state, fixed parameters, time, and diagnostics.
    }
    const Vector command_delta=u1-u0;
    require(command_delta.allFinite(), "command endpoint difference is not representable");
    const double nominal_count_value = std::ceil(dt/options.max_step_s);
    require(std::isfinite(nominal_count_value) && nominal_count_value <= 10000000.,
            "nominal substep budget exceeded");
    const uint64_t nominal_count = static_cast<uint64_t>(std::max(1.,nominal_count_value));
    const Vector saved_q = q, saved_v = v, saved_a = a, saved_u = command;
    const double saved_time = time;
    const auto saved_trace = trace;
    diagnostic.fill(0.); counts.fill(0);
    diagnostic[PROBTF_DYN_LAST_ADVANCE_DT_S] = dt;
    diagnostic[PROBTF_DYN_MIN_ACCEPTED_STEP_S] = std::numeric_limits<double>::infinity();
    energy_diagnostic();
    const double initial_energy = diagnostic[PROBTF_DYN_TOTAL_ENERGY_J];
    trace.clear(); trace.push_back({0., q, v});
    auto command_at = [&](double t) -> Vector { return t >= dt ? u1 : (u0 + (t / dt) * command_delta).eval(); };
    // Integer nominal boundaries avoid accumulating thousands of rounded
    // double 1ms additions and inventing a tiny final Newton problem. Impacts
    // and Newton halving subdivide the current boundary interval; they never
    // move the next intended boundary or discard its remaining physical time.
    uint64_t boundary_index=1;
    long double elapsed_precise = 0.;
    try {
      while (boundary_index <= nominal_count) {
        const long double boundary = static_cast<long double>(dt)*boundary_index/nominal_count;
        const double elapsed = static_cast<double>(elapsed_precise);
        const long double remaining = boundary-elapsed_precise;
        require(counts[PROBTF_DYN_ACCEPTED_SUBSTEPS] < 10000000, "substep safety budget exhausted");
        double h = static_cast<double>(remaining);
        const double requested_h = h;
        Midpoint next;
        int halvings = 0;
        for (;;) {
          try { next = solve_midpoint(q, v, h, command_at(elapsed + .5*h)); break; }
          catch (const Failure &failure) {
            if (halvings >= options.max_step_halvings || .5*h < options.min_step_s) {
              std::ostringstream message;
              message << "implicit midpoint failed at step " << h
                      << " s (minimum/budget): " << failure.what();
              throw Failure(message.str());
            }
            h *= .5; ++halvings; ++counts[PROBTF_DYN_STEP_HALVINGS];
          }
        }
        bool impact = !inside(next.q);
        if (impact) {
          // Locate the first hard-stop event along the implicit solution from
          // this state. No finite penetration is projected away afterwards.
          double lo = 0., hi = h;
          Midpoint safe;
          for (int iteration = 0; iteration < 55; ++iteration) {
            const double mid = .5 * (lo + hi);
            if (mid <= 0.) break;
            Midpoint candidate = solve_midpoint(q, v, mid, command_at(elapsed + .5*mid));
            if (inside(candidate.q)) { lo = mid; safe = candidate; }
            else hi = mid;
            if (hi - lo <= 2e-14 * std::max(1., h)) break;
          }
          require(lo > 1e-14 && safe.q.size() == n, "could not resolve joint-stop impact time");
          h = lo; next = safe;
          std::vector<std::pair<int, double>> hit;
          for (int j = 0; j < n; ++j) {
            if (next.q[j] <= lower[j] + position_tolerance && next.v[j] < 0.) {
              next.q[j] = lower[j]; hit.emplace_back(j,-1.);
            } else if (next.q[j] >= upper[j] - position_tolerance && next.v[j] > 0.) {
              next.q[j] = upper[j]; hit.emplace_back(j,1.);
            }
          }
          require(!hit.empty(), "joint-stop event did not identify a boundary");
          const Matrix m = mass(next.q);
          const Vector incoming = next.v;
          // Include all simultaneous boundaries, allowing coupled impulses to
          // change other joint velocities according to the physical mass.
          next.v = cone_projection(m, incoming, contacts(next.q, incoming, false));
          double loss = .5*(incoming.dot(m*incoming)-next.v.dot(m*next.v));
          require(loss >= -1e-10*(1.+incoming.dot(m*incoming)), "joint impact increased kinetic energy");
          diagnostic[PROBTF_DYN_IMPACT_LOSS_J] += std::max(0.,loss);
          ++counts[PROBTF_DYN_JOINT_IMPACTS];
        }
        diagnostic[PROBTF_DYN_DAMPING_LOSS_J] += h*(damping.array()*next.vm.array().square()).sum();
        diagnostic[PROBTF_DYN_MAX_TORQUE_RESIDUAL_NM] = std::max(diagnostic[PROBTF_DYN_MAX_TORQUE_RESIDUAL_NM], next.residual);
        diagnostic[PROBTF_DYN_MIN_ACCEPTED_STEP_S] = std::min(diagnostic[PROBTF_DYN_MIN_ACCEPTED_STEP_S],h);
        diagnostic[PROBTF_DYN_MAX_ACCEPTED_STEP_S] = std::max(diagnostic[PROBTF_DYN_MAX_ACCEPTED_STEP_S],h);
        ++counts[PROBTF_DYN_ACCEPTED_SUBSTEPS];
        if (next.contact) ++counts[PROBTF_DYN_ACTIVE_STOP_SUBSTEPS];
        q = next.q; v = next.v;
        const long double previous = elapsed_precise;
        if (h == requested_h) { elapsed_precise=boundary; ++boundary_index; }
        else elapsed_precise += static_cast<long double>(h);
        require(elapsed_precise > previous, "substep time failed to advance");
        const double accepted_time = static_cast<double>(elapsed_precise);
        const Vector accepted_command=command_at(accepted_time);
        // Integrate -K(q_mid-u_mid) du directly. Forming du/dt can overflow
        // for finite command endpoints and a representable tiny duration.
        diagnostic[PROBTF_DYN_COMMAND_WORK_J] -=
            ((stiffness.array()*(next.qm-next.um).array()).matrix()).dot(accepted_command-command);
        command=accepted_command;
        trace.push_back({accepted_time,q,v});
      }
      command = u1; time = saved_time + dt;
      a = endpoint_acceleration();
      energy_diagnostic();
      diagnostic[PROBTF_DYN_ENERGY_BALANCE_RESIDUAL_J] = diagnostic[PROBTF_DYN_TOTAL_ENERGY_J]-initial_energy-
          diagnostic[PROBTF_DYN_COMMAND_WORK_J]+diagnostic[PROBTF_DYN_DAMPING_LOSS_J]+diagnostic[PROBTF_DYN_IMPACT_LOSS_J];
      for (double value : diagnostic) require(std::isfinite(value), "non-finite physical energy/integrator accounting");
    } catch (...) {
      q=saved_q; v=saved_v; a=saved_a; command=saved_u; time=saved_time; trace=saved_trace;
      diagnostic[PROBTF_DYN_TIME_S] = time;
      throw;
    }
  }
};
template<class Function> int protect(void *handle, Function function) {
  if (!handle) return -1;
  Dynamics &d = *static_cast<Dynamics *>(handle);
  try { function(d); d.error.clear(); return 0; }
  catch (const std::exception &exception) { d.error=exception.what(); return -1; }
  catch (...) { d.error="unknown native dynamics exception"; return -1; }
}
} // namespace

extern "C" {
int probtf_dynamics_abi_version(void) { return 1; }
void probtf_dynamics_default_options(ProbtfDynamicsOptions *value) { if(value) *value=defaults(); }
void *probtf_dynamics_create(const char *urdf, int n, const double *k, const double *d,
                           double zeta, const ProbtfDynamicsOptions *options, char *error, size_t capacity) {
  try { auto *result=new Dynamics(urdf,n,k,d,zeta,options); if(error&&capacity) error[0]='\0'; return result; }
  catch(const std::exception &exception) { if(error&&capacity) { std::strncpy(error,exception.what(),capacity-1); error[capacity-1]='\0'; } return nullptr; }
  catch(...) { if(error&&capacity) { std::strncpy(error,"unknown create failure",capacity-1); error[capacity-1]='\0'; } return nullptr; }
}
void probtf_dynamics_destroy(void *handle) { delete static_cast<Dynamics *>(handle); }
const char *probtf_dynamics_last_error(void *handle) { return handle ? static_cast<Dynamics *>(handle)->error.c_str() : "native dynamics handle is null"; }
int probtf_dynamics_reset(void *handle,const double *q,const double *v,const double *u) {
  return protect(handle,[&](Dynamics &d){d.reset(q,v,u);});
}
int probtf_dynamics_advance(void *handle,double dt,const double *u0,const double *u1) {
  return protect(handle,[&](Dynamics &d){d.advance(dt,u0,u1);});
}
size_t probtf_dynamics_trace_size(void *handle) { return handle ? static_cast<Dynamics *>(handle)->trace.size() : 0; }
int probtf_dynamics_trace_copy(void *handle,size_t rows,double *times,double *positions,double *velocities) {
  return protect(handle,[&](Dynamics &d){
    require(rows>=d.trace.size(),"trace output capacity is too small");
    require(times&&positions&&velocities,"trace output pointer is null");
    for(size_t i=0;i<d.trace.size();++i){times[i]=d.trace[i].t; Eigen::Map<Vector>(positions+i*d.n,d.n)=d.trace[i].q; Eigen::Map<Vector>(velocities+i*d.n,d.n)=d.trace[i].v;}
  });
}
int probtf_dynamics_get_state(void *handle,double *q,double *v,double *a) {
  return protect(handle,[&](Dynamics &d){require(d.initialized,"reset must precede state query"); require(q&&v&&a,"state output pointer is null"); Eigen::Map<Vector>(q,d.n)=d.q; Eigen::Map<Vector>(v,d.n)=d.v; Eigen::Map<Vector>(a,d.n)=d.a;});
}
int probtf_dynamics_get_parameters(void *handle,double *k,double *damping,double *mass) {
  return protect(handle,[&](Dynamics &d){require(d.parameters_fixed,"reset must precede parameter query"); require(k&&damping&&mass,"parameter output pointer is null"); Eigen::Map<Vector>(k,d.n)=d.stiffness; Eigen::Map<Vector>(damping,d.n)=d.damping; Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(mass,d.n,d.n)=d.reference_mass;});
}
int probtf_dynamics_get_diagnostics(void *handle,double *values,size_t nv,uint64_t *counts,size_t nc) {
  return protect(handle,[&](Dynamics &d){require(values&&counts,"diagnostic output pointer is null"); require(nv>=d.diagnostic.size()&&nc>=d.counts.size(),"diagnostic output capacity is too small"); std::copy(d.diagnostic.begin(),d.diagnostic.end(),values); std::copy(d.counts.begin(),d.counts.end(),counts);});
}
}
