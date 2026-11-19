// Same controlled Euler/quaternion RK4 as the Python reference, in a batch.
// No inference, sampling, covariance manipulation or observation logic here.
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
void multiply3(const double* a, const double* x, double* y) {
    for (int r=0;r<3;++r) y[r]=a[3*r]*x[0]+a[3*r+1]*x[1]+a[3*r+2]*x[2];
}
void interpolate(double t, int count, const double* time, const double* value, double* out) {
    const double* found=std::upper_bound(time,time+count,t);
    int lower=std::max(0,std::min(count-2,int(found-time)-1));
    double mix=(t-time[lower])/(time[lower+1]-time[lower]);
    // Permit the final recorded endpoint; no state/output saturation occurs.
    if(t==time[count-1]) mix=1.;
    for(int d=0;d<3;++d) out[d]=value[3*lower+d]*(1.-mix)+value[3*(lower+1)+d]*mix;
}
void derivative(const double* s,const double* tau,const double* tensor,const double* inverse,
                int nonlinear,double* ds) {
    double momentum[3], force[3];
    multiply3(tensor,s,momentum);
    const double cross[3]={s[1]*momentum[2]-s[2]*momentum[1],
                           s[2]*momentum[0]-s[0]*momentum[2],
                           s[0]*momentum[1]-s[1]*momentum[0]};
    for(int d=0;d<3;++d) force[d]=tau[d]-(nonlinear?cross[d]:0.);
    multiply3(inverse,force,ds);
    const double w=s[3],x=s[4],y=s[5],z=s[6],a=s[0],b=s[1],c=s[2];
    ds[3]=-.5*(x*a+y*b+z*c);
    ds[4]=.5*(w*a+y*c-z*b);
    ds[5]=.5*(w*b+z*a-x*c);
    ds[6]=.5*(w*c+x*b-y*a);
}
}

extern "C" int probtf_controlled_flow(int nodes,const double* omega0,const double* q0,
    int outputs,const double* times,int torque_count,const double* torque_times,
    const double* torques,const double* tensor,const double* inverse,double max_step,
    int nonlinear,double* output,double* max_norm_correction,long long* step_count) {
    if(nodes<1 || outputs<1 || torque_count<2 || !(max_step>0.)) return 1;
    std::vector<double> state(size_t(nodes)*7);
    for(int n=0;n<nodes;++n) {
        for(int d=0;d<3;++d) state[size_t(n)*7+d]=omega0[size_t(n)*3+d];
        for(int d=0;d<4;++d) state[size_t(n)*7+3+d]=q0[d];
    }
    std::copy(state.begin(),state.end(),output);
    double correction=0.; long long steps_total=0;
    for(int row=1;row<outputs;++row) {
        double lower=times[row-1],upper=times[row];
        if(!(upper>lower)) return 1;
        int count=std::max(1,int(std::ceil((upper-lower)/max_step)));
        double dt=(upper-lower)/count;
        for(int step=0;step<count;++step) {
            double stamp=lower+step*dt, t1[3],t2[3],t4[3];
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
                double norm=std::sqrt(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]+s[6]*s[6]);
                if(!(norm>0.) || !std::isfinite(norm)) return 2;
                correction=std::max(correction,std::abs(norm-1.));
                for(int d=3;d<7;++d) s[d]/=norm;
            }
            ++steps_total;
        }
        std::copy(state.begin(),state.end(),output+size_t(row)*nodes*7);
    }
    *max_norm_correction=correction; *step_count=steps_total;
    return 0;
}
