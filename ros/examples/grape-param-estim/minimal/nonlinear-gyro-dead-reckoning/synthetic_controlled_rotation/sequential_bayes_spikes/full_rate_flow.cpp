// Same physical flow as the existing implementation; only avoid retaining
// past state tensors when evaluating a complete likelihood prefix.
#include "native_flow.cpp"

extern "C" int probtf_full_rate_prefix(int nodes,const double* omega0,const double* q0,
    int count,const double* times,const double* observations,int prefix,
    int torque_count,const double* torque_times,const double* torques,
    const double* tensor,const double* inverse,const double* precision,
    double likelihood_constant,double student_df,double max_step,int nonlinear,
    double* past_log_likelihood,double* block_output,
    double* max_norm_correction,long long* step_count) {
    if(nodes<1 || count<1 || prefix<0 || prefix>=count || torque_count<2 || !(max_step>0.)
        || !std::isfinite(student_df) || (student_df!=0. && !(student_df>2.))) return 1;
    std::vector<double> state(size_t(nodes)*7);
    for(int n=0;n<nodes;++n) {
        for(int d=0;d<3;++d) state[size_t(n)*7+d]=omega0[size_t(n)*3+d];
        for(int d=0;d<4;++d) state[size_t(n)*7+3+d]=q0[d];
        past_log_likelihood[n]=0.;
    }
    if(prefix==0) std::copy(state.begin(),state.end(),block_output);
    double correction=0.;long long steps_total=0;
    for(int row=1;row<count;++row) {
        const double lower=times[row-1],upper=times[row];
        if(!(upper>lower)) return 1;
        const int steps=std::max(1,int(std::ceil((upper-lower)/max_step)));
        const double dt=(upper-lower)/steps;
        for(int step=0;step<steps;++step) {
            const double stamp=lower+step*dt;
            double t1[3],t2[3],t4[3];
            interpolate(stamp,torque_count,torque_times,torques,t1);
            interpolate(stamp+.5*dt,torque_count,torque_times,torques,t2);
            interpolate(stamp+dt,torque_count,torque_times,torques,t4);
            for(int n=0;n<nodes;++n) {
                double* s=state.data()+size_t(n)*7;
                double k1[7],k2[7],k3[7],k4[7],temp[7];
                derivative(s,t1,tensor,inverse,nonlinear,k1);
                for(int d=0;d<7;++d) temp[d]=s[d]+.5*dt*k1[d];
                derivative(temp,t2,tensor,inverse,nonlinear,k2);
                for(int d=0;d<7;++d) temp[d]=s[d]+.5*dt*k2[d];
                derivative(temp,t2,tensor,inverse,nonlinear,k3);
                for(int d=0;d<7;++d) temp[d]=s[d]+dt*k3[d];
                derivative(temp,t4,tensor,inverse,nonlinear,k4);
                for(int d=0;d<7;++d) {
                    s[d]+=dt/6.*(k1[d]+2.*k2[d]+2.*k3[d]+k4[d]);
                    if(!std::isfinite(s[d])) return 2;
                }
                const double norm=std::sqrt(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]+s[6]*s[6]);
                if(!(norm>0.) || !std::isfinite(norm)) return 2;
                correction=std::max(correction,std::abs(norm-1.));
                for(int d=3;d<7;++d) s[d]/=norm;
            }
            ++steps_total;
        }
        if(row<prefix) {
            for(int n=0;n<nodes;++n) {
                const double* s=state.data()+size_t(n)*7;
                double difference[3],whitened[3];
                for(int d=0;d<3;++d) difference[d]=s[d]-observations[size_t(row)*3+d];
                multiply3(precision,difference,whitened);
                if(student_df>0.) {
                    const double mahalanobis=difference[0]*whitened[0]
                        +difference[1]*whitened[1]+difference[2]*whitened[2];
                    past_log_likelihood[n]+=likelihood_constant
                        -.5*(student_df+3.)*std::log1p(mahalanobis/student_df);
                } else {
                    past_log_likelihood[n]+=likelihood_constant-.5*(difference[0]*whitened[0]
                        +difference[1]*whitened[1]+difference[2]*whitened[2]);
                }
            }
        }
        if(row>=prefix) std::copy(state.begin(),state.end(),block_output+size_t(row-prefix)*nodes*7);
    }
    *max_norm_correction=correction;*step_count=steps_total;
    return 0;
}
