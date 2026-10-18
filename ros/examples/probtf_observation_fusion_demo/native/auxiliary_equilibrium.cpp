#include "auxiliary_equilibrium.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* LAPACK uses zero pivot detection, without a relative condition audit gate.
 * This is the same dense solve family used by numpy.linalg.solve. */
extern "C" void dgesv_(const int *, const int *, double *, const int *,
                         int *, double *, const int *, int *);

namespace {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
void require(bool condition, const char *message) { if (!condition) throw Failure(message); }
void copy_error(char *destination, size_t size, const std::string &error) {
  if (destination && size) { std::strncpy(destination, error.c_str(), size-1); destination[size-1]='\0'; }
}
Vector read(const double *values, int n) {
  require(values != nullptr, "Auxiliary vector pointer is null.");
  Vector result = Eigen::Map<const Vector>(values, n);
  require(result.allFinite(), "Auxiliary input contains NaN or infinity.");
  return result;
}
Matrix solve(Matrix matrix, Matrix rhs) {
  require(matrix.allFinite() && rhs.allFinite(), "Auxiliary linear solve has nonfinite inputs.");
  const int n = matrix.rows(), columns = rhs.cols();
  std::vector<int> pivots(n); int info=0;
  dgesv_(&n, &columns, matrix.data(), &n, pivots.data(), rhs.data(), &n, &info);
  require(info == 0, "Auxiliary linear system could not be solved (LAPACK dgesv).");
  require(rhs.allFinite(), "Auxiliary linear solve returned NaN or infinity.");
  return rhs;
}
long double upward(long double value) {
  return std::nextafter(value, std::numeric_limits<long double>::infinity());
}
long double norm_upper(const Eigen::Vector3d &value) {
  long double square=0.;
  for(int i=0;i<3;++i) {
    const long double component=value[i];
    square=upward(square+upward(component*component));
  }
  return upward(std::sqrt(square));
}
struct Result {
  Vector theta, residual;
  Matrix hessian, parameter_jacobian, derivative;
  std::vector<unsigned char> active;
};
struct Model {
  pinocchio::Model model;
  int n;
  std::vector<int> spring_types;
  Vector lower, upper;
  Matrix gravity_hessian_bound;
  Model(const char *urdf, int dimension, const int *types): n(dimension) {
    require(urdf && *urdf, "Auxiliary URDF path is empty.");
    require(n >= 1 && n <= 6, "Auxiliary model supports 1..6 scalar revolute joints.");
    pinocchio::urdf::buildModel(std::string(urdf), model);
    require(model.nq == n && model.nv == n, "Auxiliary URDF nq/nv differ from supplied dimension.");
    for (pinocchio::JointIndex j=1; j<model.joints.size(); ++j)
      require(model.joints[j].nq()==1 && model.joints[j].nv()==1 &&
              model.joints[j].shortname().find("JointModelR")==0,
              "Auxiliary model requires bounded scalar revolute joints.");
    model.gravity.linear() << 0., 0., -9.81;
    lower=model.lowerPositionLimit; upper=model.upperPositionLimit;
    require(lower.allFinite() && upper.allFinite() && (lower.array()<upper.array()).all(),
            "Auxiliary URDF requires finite ordered joint limits.");
    for(int j=0; j<n; ++j) {
      int kind=types ? types[j] : 0;
      require(kind==0 || kind==1, "Auxiliary spring type must be linear(0) or periodic(1).");
      spring_types.push_back(kind);
    }
    // Each COM position is a sum of rotated joint-origin translations and
    // a rotated COM offset. A revolute derivative inserts a unit-axis skew
    // matrix, whose operator norm is at most one. Each second derivative is
    // bounded by that segment's length whenever both joints act on it.
    gravity_hessian_bound=Matrix::Zero(n,n);
    const long double gravity=norm_upper(model.gravity.linear());
    for(pinocchio::JointIndex body=1;body<model.joints.size();++body) {
      const long double mass=model.inertias[body].mass();
      require(std::isfinite(mass) && mass>=0., "Auxiliary inertia requires nonnegative finite mass.");
      std::vector<pinocchio::JointIndex> chain;
      for(pinocchio::JointIndex joint=body;joint;joint=model.parents[joint]) chain.push_back(joint);
      std::reverse(chain.begin(),chain.end());
      std::vector<int> affected;
      auto add_segment=[&](const Eigen::Vector3d &segment) {
        const long double contribution=upward(upward(mass*gravity)*norm_upper(segment));
        for(int i:affected) for(int j:affected) {
          const long double sum=upward(static_cast<long double>(gravity_hessian_bound(i,j))+contribution);
          gravity_hessian_bound(i,j)=std::nextafter(static_cast<double>(sum),
              std::numeric_limits<double>::infinity());
        }
      };
      for(auto joint:chain) {
        // The joint-origin translation precedes this joint's rotation.
        add_segment(model.jointPlacements[joint].translation());
        affected.push_back(model.joints[joint].idx_v());
      }
      add_segment(model.inertias[body].lever());
    }
    // A root revolute axis exactly parallel to the single nonzero gravity
    // coordinate rotates its whole subtree without changing any COM height.
    // Use structural zeros/ones only; approximate alignment is not a proof.
    int gravity_axis=-1;
    for(int axis=0;axis<3;++axis) if(model.gravity.linear()[axis]!=0.) {
      if(gravity_axis!=-1) { gravity_axis=-2; break; }
      gravity_axis=axis;
    }
    if(gravity_axis>=0) for(pinocchio::JointIndex joint=1;joint<model.joints.size();++joint) {
      if(model.parents[joint]!=0) continue;
      const std::string name=model.joints[joint].shortname();
      const int axis=name=="JointModelRX" ? 0 : name=="JointModelRY" ? 1 : name=="JointModelRZ" ? 2 : -1;
      if(axis<0) continue;
      const Eigen::Vector3d direction=model.jointPlacements[joint].rotation().col(axis);
      bool aligned=std::abs(direction[gravity_axis])==1.;
      for(int other=0;other<3;++other) if(other!=gravity_axis) aligned&=direction[other]==0.;
      if(aligned) {
        const int coordinate=model.joints[joint].idx_v();
        gravity_hessian_bound.row(coordinate).setZero();
        gravity_hessian_bound.col(coordinate).setZero();
      }
    }
  }
  bool certify_convexity(const Vector &k, ProbtfAuxiliaryDiagnostic &diagnostic) const {
    diagnostic.certificate_ratio_upper=std::numeric_limits<double>::infinity();
    diagnostic.certificate_rounding_margin=64.*std::numeric_limits<double>::epsilon();
    if(std::any_of(spring_types.begin(),spring_types.end(),[](int kind){return kind!=0;}) ||
       !gravity_hessian_bound.allFinite()) return false;
    // A positive witness v and Bv < diag(K)v prove rho(K^-1 B)<1.
    // Symmetry then implies diag(K)-B positive definite, and hence global
    // strict convexity of the unchanged potential on the joint box.
    const Vector roots=k.array().sqrt().matrix();
    Matrix normalized=gravity_hessian_bound;
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) normalized(i,j)/=roots[i]*roots[j];
    Eigen::SelfAdjointEigenSolver<Matrix> eig(normalized);
    Vector witness=Vector::Ones(n);
    if(eig.info()==Eigen::Success && eig.eigenvectors().allFinite()) {
      witness=eig.eigenvectors().col(n-1).cwiseAbs().cwiseQuotient(roots);
      const double maximum=witness.maxCoeff();
      if(maximum>0. && std::isfinite(maximum)) witness=(witness/maximum).cwiseMax(1e-12);
      else witness=Vector::Ones(n);
    }
    long double ratio=0.;
    for(int i=0;i<n;++i) {
      long double numerator=0.;
      for(int j=0;j<n;++j) numerator=upward(numerator+upward(
          static_cast<long double>(gravity_hessian_bound(i,j))*witness[j]));
      // Directed rounding of the denominator downward preserves an upper
      // bound. The extra explicit margin is conservative, never a law gate.
      const long double denominator=std::nextafter(static_cast<long double>(k[i])*witness[i],0.L);
      if(!(denominator>0.) || !std::isfinite(numerator)) return false;
      ratio=std::max(ratio,upward(numerator/denominator));
    }
    diagnostic.certificate_ratio_upper=std::nextafter(static_cast<double>(ratio),
        std::numeric_limits<double>::infinity());
    return diagnostic.certificate_ratio_upper < 1.-diagnostic.certificate_rounding_margin;
  }
  double exact_box_KKT_residual(pinocchio::Data &data, const Vector &theta,
      const Vector &command, const Vector &k) const {
    Vector gradient=pinocchio::computeGeneralizedGravity(model,data,theta)+torque(theta,command,k);
    require(gradient.allFinite(), "Direct auxiliary KKT check has nonfinite arithmetic.");
    for(int j=0;j<n;++j) {
      if(theta[j]==lower[j]) gradient[j]=std::min(gradient[j],0.);
      else if(theta[j]==upper[j]) gradient[j]=std::max(gradient[j],0.);
    }
    return gradient.cwiseAbs().maxCoeff();
  }
  Vector torque(const Vector &theta, const Vector &command, const Vector &k) const {
    Vector result(n);
    for(int j=0;j<n;++j) result[j]=spring_types[j] ?
      2.*k[j]*std::sin(.5*(theta[j]-command[j])) : k[j]*(theta[j]-command[j]);
    return result;
  }
  double potential(pinocchio::Data &data, const Vector &theta,
                   const Vector &command, const Vector &k) const {
    double spring=0.;
    for(int j=0;j<n;++j) {
      double delta=theta[j]-command[j];
      spring+=spring_types[j] ? 4.*k[j]*(1.-std::cos(.5*delta)) : .5*delta*k[j]*delta;
    }
    const double result=pinocchio::computePotentialEnergy(model,data,theta)+spring;
    require(std::isfinite(result), "Auxiliary potential is nonfinite.");
    return result;
  }
  std::vector<unsigned char> active_set(const Vector &theta, const Vector &gradient) const {
    std::vector<unsigned char> active(n);
    for(int j=0;j<n;++j) active[j]=
      (theta[j]<=lower[j] && gradient[j]>=0.) ||
      (theta[j]>=upper[j] && gradient[j]<=0.);
    return active;
  }
  void equations(pinocchio::Data &data, const Vector &theta,
      const Vector &command, const Vector &k, Result &result) const {
    result.hessian=Matrix::Zero(n,n);
    pinocchio::computeGeneralizedGravityDerivatives(model,data,theta,result.hessian);
    result.parameter_jacobian=Matrix::Zero(n,n);
    Vector spring=torque(theta,command,k);
    for(int j=0;j<n;++j) {
      result.hessian(j,j)+=spring_types[j] ? k[j]*std::cos(.5*(theta[j]-command[j])) : k[j];
      result.parameter_jacobian(j,j)=spring[j];
    }
    result.residual=pinocchio::computeGeneralizedGravity(model,data,theta)+spring;
    require(theta.allFinite() && result.hessian.allFinite() &&
      result.parameter_jacobian.allFinite() && result.residual.allFinite(),
      "Auxiliary equations contain NaN or infinity.");
    result.active=active_set(theta,result.residual);
    for(int j=0;j<n;++j) if(result.active[j]) {
      result.hessian.col(j).setZero(); result.hessian.row(j).setZero();
      result.hessian(j,j)=1.; result.parameter_jacobian.row(j).setZero();
      result.residual[j]=0.;
    }
  }
  Result refine(pinocchio::Data &data, Vector theta, const Vector &command,
                const Vector &k, ProbtfAuxiliaryDiagnostic &diagnostic) const {
    require(theta.allFinite() && command.allFinite() && k.allFinite() && (k.array()>0.).all(),
            "Auxiliary refinement needs finite theta/command and positive finite K.");
    ++diagnostic.refinement_calls;
    Result result;
    diagnostic.refinement_iterations=0; diagnostic.refinement_termination=0;
    for(int iteration=0;iteration<24;++iteration) {
      equations(data,theta,command,k,result);
      double norm=result.residual.cwiseAbs().maxCoeff();
      if(norm<=1e-11) { diagnostic.refinement_termination=1; break; }
      const Vector direction=solve(result.hessian,result.residual).col(0);
      ++diagnostic.refinement_iterations; ++diagnostic.newton_iterations;
      double old_potential=potential(data,theta,command,k); bool accepted=false;
      for(int backtrack=0;backtrack<16;++backtrack) {
        Vector candidate=(theta-std::ldexp(1.,-backtrack)*direction).cwiseMax(lower).cwiseMin(upper);
        Vector residual=pinocchio::computeGeneralizedGravity(model,data,candidate)+torque(candidate,command,k);
        require(candidate.allFinite() && residual.allFinite(), "Auxiliary candidate residual is nonfinite.");
        const auto active=active_set(candidate,residual);
        for(int j=0;j<n;++j) if(active[j]) residual[j]=0.;
        const double candidate_potential=potential(data,candidate,command,k);
        ++diagnostic.line_search_evaluations;
        if(residual.cwiseAbs().maxCoeff()<norm &&
           candidate_potential<=old_potential+1e-12*std::max(1.,std::abs(old_potential))) {
          theta=candidate; accepted=true; break;
        }
      }
      if(!accepted) { diagnostic.refinement_termination=2; break; }
    }
    equations(data,theta,command,k,result); result.theta=theta;
    ++diagnostic.completed_refinements;
    diagnostic.last_residual=result.residual.cwiseAbs().maxCoeff();
    diagnostic.maximum_residual=std::max(diagnostic.maximum_residual,diagnostic.last_residual);
    diagnostic.residual_warning_count+=diagnostic.last_residual>1e-9;
    Eigen::SelfAdjointEigenSolver<Matrix> eig(.5*result.hessian+.5*result.hessian.transpose());
    diagnostic.eigenvalues_available=eig.info()==Eigen::Success && eig.eigenvalues().allFinite();
    if(diagnostic.eigenvalues_available) {
      for(int j=0;j<n;++j) diagnostic.eigenvalues[j]=eig.eigenvalues()[j];
      diagnostic.nonpositive_curvature_count+=eig.eigenvalues()[0]<=0.;
    } else ++diagnostic.curvature_audit_failure_count;
    return result;
  }
  Result segment(pinocchio::Data &data, const Vector &left, const Vector &right,
      const Vector &initial, const Vector &command, int depth,
      ProbtfAuxiliaryDiagnostic &diagnostic) const {
    try { return refine(data,initial,command,right.array().exp().matrix(),diagnostic); }
    catch(const std::exception &) {
      if(depth==10) throw;
      ++diagnostic.bisections;
      const Vector middle=.5*(left+right);
      Result halfway=segment(data,left,middle,initial,command,depth+1,diagnostic);
      return segment(data,middle,right,halfway.theta,command,depth+1,diagnostic);
    }
  }
  Result evaluate(const Vector &target, const Vector &command, const Vector &anchor_logK,
                  const Vector &anchor_theta, int mode, ProbtfAuxiliaryDiagnostic &diagnostic) const {
    pinocchio::Data data(model); // Every batch row owns all mutable mechanics.
    Result result;
    if(mode==1) result=refine(data,anchor_theta,command,target,diagnostic);
    else {
      const Vector k=target.array().exp().matrix();
      require(k.allFinite() && (k.array()>0.).all(), "Auxiliary logK has nonfinite or zero exponential.");
      const double distance=(target-anchor_logK).cwiseAbs().maxCoeff();
      const double requested=std::max(1.,std::ceil(distance/.1));
      require(std::isfinite(requested), "Auxiliary continuation count is nonfinite.");
      diagnostic.requested_stages=static_cast<uint64_t>(requested);
      diagnostic.used_stages=std::min<uint64_t>(diagnostic.requested_stages,1000);
      diagnostic.convexity_certified=mode==2 && certify_convexity(k,diagnostic);
      if(diagnostic.convexity_certified) {
        diagnostic.direct_attempted=1;
        try {
          result=refine(data,anchor_theta,command,k,diagnostic);
          diagnostic.direct_KKT_residual=exact_box_KKT_residual(data,result.theta,command,k);
          if(diagnostic.direct_KKT_residual<=1e-11) {
            diagnostic.direct_used=1; diagnostic.used_stages=1;
            result.derivative=-solve(result.hessian,result.parameter_jacobian);
            return result;
          }
          copy_error(diagnostic.direct_error,sizeof(diagnostic.direct_error),
              "Direct finite iterate did not resolve the unique KKT root; canonical continuation retained.");
        } catch(const std::exception &failure) {
          diagnostic.direct_failed=1;
          copy_error(diagnostic.direct_error,sizeof(diagnostic.direct_error),failure.what());
        }
      }
      Vector theta=anchor_theta, left=anchor_logK;
      for(uint64_t stage=1;stage<=diagnostic.used_stages;++stage) {
        const double fraction=static_cast<double>(stage)/diagnostic.used_stages;
        const Vector right=anchor_logK+fraction*(target-anchor_logK);
        result=segment(data,left,right,theta,command,0,diagnostic);
        theta=result.theta; left=right;
      }
    }
    result.derivative=-solve(result.hessian,result.parameter_jacobian);
    return result;
  }
};
}

extern "C" {
int probtf_auxiliary_abi_version() { return 2; }
void *probtf_auxiliary_create(const char *urdf,int dimension,const int *types,
                            char *error,size_t error_size) {
  try { return new Model(urdf,dimension,types); }
  catch(const std::exception &failure) { copy_error(error,error_size,failure.what()); return nullptr; }
}
void probtf_auxiliary_destroy(void *handle) { delete static_cast<Model *>(handle); }
int probtf_auxiliary_gravity_hessian_bound(void *handle,double *bound) {
  if(!handle || !bound) return -1;
  const auto &model=*static_cast<Model *>(handle);
  Eigen::Map<RowMatrix>(bound,model.n,model.n)=model.gravity_hessian_bound;
  return 0;
}
int probtf_auxiliary_evaluate_batch(void *handle,size_t rows,int mode,int workers,
    const double *target,const double *command,const double *anchor_logK,const double *anchor_theta,
    double *theta,double *hessian,double *parameter_jacobian,double *derivative,
    unsigned char *active,double *residual,ProbtfAuxiliaryDiagnostic *diagnostics) {
  if(!handle || !target || !command || !anchor_theta || !theta || !hessian ||
     !parameter_jacobian || !derivative || !active || !residual || !diagnostics ||
     (mode!=0 && mode!=1 && mode!=2) || (mode!=1 && !anchor_logK) || workers<1) return -1;
  const auto &model=*static_cast<Model *>(handle); const int n=model.n;
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(workers)
  #endif
  for(size_t row=0;row<rows;++row) {
    auto &diagnostic=diagnostics[row]; diagnostic=ProbtfAuxiliaryDiagnostic{};
    diagnostic.direct_KKT_residual=std::numeric_limits<double>::quiet_NaN();
    std::fill(theta+row*n,theta+(row+1)*n,std::numeric_limits<double>::quiet_NaN());
    std::fill(residual+row*n,residual+(row+1)*n,std::numeric_limits<double>::quiet_NaN());
    for(double *output : {hessian,parameter_jacobian,derivative})
      std::fill(output+row*n*n,output+(row+1)*n*n,std::numeric_limits<double>::quiet_NaN());
    std::fill(active+row*n,active+(row+1)*n,0);
    try {
      Result result=model.evaluate(read(target+row*n,n),read(command+row*n,n),
          mode!=1 ? read(anchor_logK+row*n,n) : Vector::Zero(n),read(anchor_theta+row*n,n),mode,diagnostic);
      Eigen::Map<Vector>(theta+row*n,n)=result.theta;
      Eigen::Map<Vector>(residual+row*n,n)=result.residual;
      Eigen::Map<RowMatrix>(hessian+row*n*n,n,n)=result.hessian;
      Eigen::Map<RowMatrix>(parameter_jacobian+row*n*n,n,n)=result.parameter_jacobian;
      Eigen::Map<RowMatrix>(derivative+row*n*n,n,n)=result.derivative;
      std::copy(result.active.begin(),result.active.end(),active+row*n);
    } catch(const std::exception &failure) {
      diagnostic.status=1; copy_error(diagnostic.error,sizeof(diagnostic.error),failure.what());
    }
  }
  return 0;
}
}
