"""Differentiate mocap, then integrate its body rate: a kinematic consistency test.

This deliberately uses mocap-derived angular velocity throughout. It is not
an independent IMU reconstruction, a causal estimator, or a force/torque model
replay. Initial attitude is aligned once; there are no later attitude resets.
Quaternions map body to world; Hamilton multiplication acts on the right.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quaternion_multiply(left, right):
    """Hamilton product in wxyz convention with NumPy broadcasting."""
    left, right = np.broadcast_arrays(np.asarray(left,float),np.asarray(right,float))
    scalar = left[..., :1]*right[..., :1] - np.sum(left[...,1:]*right[...,1:],axis=-1,keepdims=True)
    vector = left[..., :1]*right[...,1:] + right[..., :1]*left[...,1:] + np.cross(left[...,1:],right[...,1:])
    return np.concatenate((scalar,vector),axis=-1)


def _increment(omega,dt):
    vector=np.asarray(omega,float)*float(dt)
    angle=np.linalg.norm(vector)
    return np.r_[np.cos(angle/2),.5*np.sinc(angle/(2*np.pi))*vector]


def integrate_body_rate(times,q0_wxyz,rate_times,omega_body,*,interpolation='linear',max_step=.002):
    """Integrate qdot=q*(0,omega_body)/2 on the original supplied time axis.

    Linear interpolation uses RK4 and splits at all supplied rate knots, with
    finite maximum step ``max_step``. ZOH uses exact constant-rate increments
    and also splits at each acquisition; its final observation is held to the
    requested endpoint. Neither path obtains any later attitude observations.
    """
    query,clock,rate=np.asarray(times,float),np.asarray(rate_times,float),np.asarray(omega_body,float)
    initial=np.asarray(q0_wxyz,float)
    if (query.ndim!=1 or len(query)<1 or not np.isfinite(query).all() or np.any(np.diff(query)<=0)
            or clock.ndim!=1 or len(clock)<1 or not np.isfinite(clock).all() or np.any(np.diff(clock)<=0)
            or rate.shape!=(len(clock),3) or not np.isfinite(rate).all()
            or initial.shape!=(4,) or not np.isfinite(initial).all() or np.linalg.norm(initial)==0):
        raise ValueError('Finite increasing clocks, Nx3 body rates and one nonzero quaternion are required')
    if interpolation not in ('linear','zoh'):
        raise ValueError('interpolation must be linear or zoh')
    if query[0]<clock[0] or (interpolation=='linear' and query[-1]>clock[-1]):
        raise ValueError('Requested integration exceeds the rate support')
    step=float(max_step)
    if not np.isfinite(step) or step<=0:
        raise ValueError('A positive finite maximum RK4 step is required')
    q=initial/np.linalg.norm(initial)
    output=np.empty((len(query),4));output[0]=q

    def derivative(time,point):
        value=np.array([np.interp(time,clock,rate[:,axis]) for axis in range(3)])
        return .5*quaternion_multiply(point,np.r_[0.,value])

    for row in range(1,len(query)):
        begin,end=float(query[row-1]),float(query[row])
        inner=clock[(clock>begin)&(clock<end)]
        boundaries=np.r_[begin,inner,end]
        for lower,upper in zip(boundaries[:-1],boundaries[1:]):
            duration=float(upper-lower)
            if interpolation=='zoh':
                index=int(np.searchsorted(clock,lower,side='right')-1)
                q=quaternion_multiply(q,_increment(rate[index],duration))
            else:
                count=max(1,int(np.ceil(duration/step)))
                dt=duration/count
                for substep in range(count):
                    time=lower+substep*dt
                    first=derivative(time,q)
                    second=derivative(time+dt/2,q+dt*first/2)
                    third=derivative(time+dt/2,q+dt*second/2)
                    fourth=derivative(time+dt,q+dt*third)
                    q=q+dt*(first+2*second+2*third+fourth)/6
                    if not np.isfinite(q).all() or np.linalg.norm(q)==0:
                        raise FloatingPointError('Kinematic integration produced a nonfinite or zero quaternion')
                    q=q/np.linalg.norm(q)
        output[row]=q/np.linalg.norm(q)
    return output


def _rpy(quaternion_xyzw):
    return np.unwrap(Rotation.from_quat(quaternion_xyzw).as_euler('xyz'),axis=0)


def _errors(predicted_wxyz,reference_xyzw):
    predicted=Rotation.from_quat(predicted_wxyz[:,[1,2,3,0]])
    reference=Rotation.from_quat(reference_xyzw)
    geodesic=(reference.inv()*predicted).magnitude()
    rpy_difference=(predicted.as_euler('xyz')-reference.as_euler('xyz')+np.pi)%(2*np.pi)-np.pi
    return geodesic,rpy_difference,dict(
        geodesic_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(geodesic**2)))),
        geodesic_max_deg=float(np.rad2deg(geodesic.max())),
        geodesic_final_deg=float(np.rad2deg(geodesic[-1])),
        rpy_rmse_deg=np.rad2deg(np.sqrt(np.mean(rpy_difference**2,axis=0))).tolist(),
        rpy_final_error_deg=np.rad2deg(rpy_difference[-1]).tolist())


def roundtrip_case(data,sg_result,sg_time,window_seconds,*,max_step=.002):
    """Reintegrate an already evaluated SG body-rate curve over the fixed 5s.

    ``data`` needs eval_interval, mocap_time and mocap_quaternion_xyzw.
    ``sg_result`` has omega_body and quaternion_xyzw from the SG derivative.
    Later mocap values are used only to create the diagnostic rate input and
    to compare the result; they do not reset the integrated quaternion.
    """
    start,end=np.asarray(data['eval_interval'],float)
    source_time=np.asarray(data['mocap_time'],float)
    source_q=np.asarray(data['mocap_quaternion_xyzw'],float)
    rate_time=np.asarray(sg_time,float)
    rate=np.asarray(sg_result['omega_body'],float)
    fitted=np.asarray(sg_result['quaternion_xyzw'],float)
    if rate_time[0]>start or rate_time[-1]<end:
        raise ValueError('SG rate must cover the complete fixed evaluation interval')
    desired_sparse=start+np.arange(0.,end-start,2.)
    # Evaluate the ZOH comparison at exact requested 0.5 Hz times using only
    # this already noncausal SG-derived diagnostic curve, not native gyro.
    sparse_rate=np.column_stack([np.interp(desired_sparse,rate_time,rate[:,axis]) for axis in range(3)])
    native=source_time[(source_time>=start)&(source_time<=end)]
    query=np.unique(np.r_[start,native,desired_sparse,end])
    reference=Slerp(source_time,Rotation.from_quat(source_q))(query).as_quat()
    initial=reference[0,[3,0,1,2]]
    full=integrate_body_rate(query,initial,rate_time,rate,max_step=max_step)
    sparse=integrate_body_rate(query,initial,desired_sparse,sparse_rate,interpolation='zoh',max_step=max_step)
    full_error,full_rpy_error,full_score=_errors(full,reference)
    sparse_error,sparse_rpy_error,sparse_score=_errors(sparse,reference)
    fit_reference=Slerp(rate_time,Rotation.from_quat(fitted))(query).as_quat()
    _,_,full_vs_fit=_errors(full,fit_reference)
    arrays=dict(time=query,rate_time=rate_time,sg_omega_body=rate,
        sg_quaternion_xyzw=fitted,reference_quaternion_xyzw=reference,
        reference_rpy_rad=_rpy(reference),full_q_wxyz=full,sparse_q_wxyz=sparse,
        full_rpy_rad=_rpy(full[:,[1,2,3,0]]),sparse_rpy_rad=_rpy(sparse[:,[1,2,3,0]]),
        full_error_geodesic_rad=full_error,sparse_error_geodesic_rad=sparse_error,
        full_error_rpy_rad=full_rpy_error,sparse_error_rpy_rad=sparse_rpy_error,
        sparse_rate_time=desired_sparse,sparse_rate_body=sparse_rate,
        fitted_reference_quaternion_xyzw=fit_reference)
    metrics=dict(window_seconds=float(window_seconds),max_integrator_step_seconds=float(max_step),
        full_rate=full_score,sparse_0p5hz=sparse_score,full_rate_vs_fitted_mocap=full_vs_fit,
        comparison_count=len(query),comparison_clock='native mocap header times plus fixed endpoints and sparse-rate knots',
        units='angular velocity rad/s, quaternion wxyz body-to-world, diagnostic errors degrees',
        initial_attitude='raw mocap SLERP once at evaluation start; no later resets',
        angular_input='mocap-derived centered-SG body angular velocity; not an independent sensor observation',
        full_rate_input_interpolation='piecewise linear over supplied SG evaluation grid',
        sparse_input='SG-derived rate at exactly start+0/2/4 seconds, then zero-order hold',
        uncertainty_propagated=False,force_torque_model_used=False,
        experiment_role='noncausal offline differentiation/integration consistency diagnostic, separate from IMU prediction')
    return dict(arrays=arrays,metrics=metrics)


def run_roundtrip(data,window_seconds=.16,degree=3,*,max_step=.002,rate_grid_dt=.0025):
    """Evaluate SG on a fine finite grid, then perform both reintegrations."""
    from sg_compare import differentiate_quaternions
    start,end=np.asarray(data['eval_interval'],float)
    dt=float(rate_grid_dt)
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('SG rate grid step must be positive and finite')
    rate_time=np.linspace(start,end,int(np.ceil((end-start)/dt))+1)
    sg=differentiate_quaternions(data['mocap_time'],data['mocap_quaternion_xyzw'],rate_time,
        window_seconds=window_seconds,degree=degree)
    result=roundtrip_case(data,sg,rate_time,window_seconds,max_step=max_step)
    result['metrics']['sg_rate_grid_max_step_seconds']=float(np.max(np.diff(rate_time)))
    result['metrics']['sg_degree']=int(degree)
    result['metrics']['sg_queries']=len(rate_time)
    result['metrics']['sg_grid_role']='numerical polynomial evaluation grid, not additional independently acquired mocap observations'
    return result
