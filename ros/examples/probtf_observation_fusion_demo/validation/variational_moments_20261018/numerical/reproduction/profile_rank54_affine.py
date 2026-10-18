"""Deterministic affine Gaussian fixture; no physical or live-rate claim."""
from pathlib import Path
import json,math,time
import numpy as np
from probtf.spherical_law.fusion import Gaussian,condition_gaussian
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.variational_conditioning import condition_variational,ExpectedLikelihoodEvaluation,_WhitenedObjective

n=66;r=54;i=np.arange(r)
variance=.008+.022*(i%7)/6.
P=np.zeros((n,n));P[:r,:r]=np.sqrt(variance[:,None]*variance[None,:])*.27**np.abs(i[:,None]-i[None,:])
m=.02*np.sin(np.arange(n)*.21);prior=Gaussian(m,P)
rows=np.arange(1,7)[:,None];cols=np.arange(1,r+1)[None,:]
H=np.zeros((6,n));H[:,:r]=.2*np.cos(rows*cols*.17)+.1*np.sin((rows+1)*cols*.11)
y=H@m+np.array([.03,-.02,.01,.04,-.03,.02]);R=np.diag([.005**2]*3+[.02**2]*3);Q=np.linalg.inv(R)
normalizer=.5*(np.linalg.slogdet(R)[1]+6*np.log(2*np.pi))
def evaluate(mean,covariance):
 residual=H@mean-y;quadratic=.5*float(residual@Q@residual);trace=.5*float(np.trace(Q@H@covariance@H.T))
 return ExpectedLikelihoodEvaluation(quadratic+trace+normalizer,H.T@Q@residual,.5*H.T@Q@H,
  expected_negative_log_likelihood_terms=(quadratic,trace,normalizer))
expected,evidence=condition_gaussian(prior,y,R,H)
records=[]
for budget in (0,24,48,96,192,384):
 started=time.perf_counter();result=condition_variational(prior,evaluate,max_iterations=budget,gradient_tolerance=1e-6)
 elapsed=time.perf_counter()-started;d=result.diagnostics
 record={'max_iterations':budget,'wall_seconds':elapsed,'actual_iterations':d['completed_iterations'],'evaluator_calls':d['valid_evaluator_calls'],'gradient_inf':d['final_gradient_inf'],'stationarity_verified':d['stationarity_verified'],'mean_max_abs_error':float(np.max(np.abs(result.distribution.mean-expected.mean))),'covariance_max_abs_error':float(np.max(np.abs(result.distribution.covariance-expected.covariance))),'affine_log_evidence_lower_bound_gap':float(evidence-result.log_evidence),'optimizer_message':d['optimizer_message']}
 records.append(record);print(json.dumps(record),flush=True)
family=_WhitenedObjective(prior,evaluate);support=covariance_support(P);inverse_B=(support.basis/np.sqrt(support.eigenvalues)).T
mean_white=inverse_B@(expected.mean-prior.mean);V=inverse_B@expected.covariance@inverse_B.T;L=np.linalg.cholesky(.5*V+.5*V.T)
packed=L[family.lower].copy();packed[family.diagonal]=np.log(np.diag(L));x=np.r_[mean_white,packed]
started=time.perf_counter();seed=family.evaluate(x);seed_seconds=time.perf_counter()-started
out={'fixture':'synthetic deterministic affine n66/rank54/observation6; no Monte Carlo, no physical input','full_cholesky_plus_mean_parameter_count':family.parameter_count,'observation_prior_covariance_eigenvalues':np.linalg.eigvalsh(H@P@H.T).tolist(),'noise_variances':np.diag(R).tolist(),'records':records,'analytic_affine_seed_diagnostic_only':{'objective_evaluation_seconds':seed_seconds,'gradient_inf':seed['gradient_inf'],'evidence_error':float(-seed['objective']-evidence),'method':'core condition_gaussian posterior converted to same original whitened full-Cholesky chart; not a production option','nonlinear_performance_or_improvement_claimed':False},'native_actual_backend_cost_excluded':True,'production_changed':False}
path=Path('/tmp/probtf_variational_rank54_affine_benchmark_20261018.json');path.write_text(json.dumps(out,indent=2)+'\n');print(path)
