"""Fixed-size moments of the SAME factored six-point rotation law.

There is no Bingham projection, particle draw, component pruning, or repeated
substep noise here.  The independent six-point increment factors define the
candidate law.  Its quaternion second moment and its rotation-entry mean and
full covariance are propagated exactly for that finite law, up to floating
arithmetic and the pre-existing finite-step nonlinear flow integration.

Euler coordinates are nonlinear functions of rotation entries.  The plotted
Euler mean and covariance are explicitly delta-method proxies from these
same moments, not exact Euler moments or guaranteed 95% regions.  This is
the only additional display approximation needed to avoid enumerating 6^N
products when N=25 (5 Hz for five seconds).
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
for directory in (PARENT, PARENT.parents[4]/'ros/core/probtf_core/src'):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from propagation import integrate_six_point_increment
from probtf.bingham import quaternion_product_second_moment, rotation_first_moment


def rotation_moment_update(parent_mean,parent_covariance,increments_wxyz,weights):
    """Exact law-of-total-covariance update for R_parent R_increment.

    vec uses column-major order.  For each fixed right increment,
    A=R_increment.T kron I_3 maps vec(R_parent). The between term is a sum
    of weighted centered outer products; no subtraction of nearly equal
    raw moments, jitter, variance floor, or covariance clipping is used.
    """
    mean=np.asarray(parent_mean,float)
    covariance=np.asarray(parent_covariance,float)
    delta=np.asarray(increments_wxyz,float)
    mass=np.asarray(weights,float)
    if (mean.shape!=(3,3) or covariance.shape!=(9,9) or delta.ndim!=2
            or delta.shape[1]!=4 or mass.shape!=(len(delta),)
            or not all(np.isfinite(v).all() for v in (mean,covariance,delta,mass))
            or np.any(mass<0.) or not mass.sum()>0.):
        raise ValueError('Finite rotation moments and positive increment masses are required')
    rotations=Rotation.from_quat(delta[:,[1,2,3,0]]).as_matrix()
    actions=np.stack([np.kron(rotation.T,np.eye(3)) for rotation in rotations])
    vector=mean.reshape(9,order='F')
    conditional_mean=np.einsum('sij,j->si',actions,vector)
    updated_mean=np.einsum('s,si->i',mass,conditional_mean)
    within=np.einsum('s,sai,ij,sbj->ab',mass,actions,covariance,actions,optimize=True)
    centered=conditional_mean-updated_mean
    between=np.einsum('s,si,sj->ij',mass,centered,centered)
    updated_covariance=within+between
    return updated_mean.reshape(3,3,order='F'),.5*(updated_covariance+updated_covariance.T)


def euler_proxy(mean_rotation,covariance_rotation):
    """RPY proxy f(E[R]), Df(E[R]) Cov(vec R) Df(E[R]).T.

    f extends the usual xyz Euler extraction to a nonsingular ambient matrix:
    roll=atan2(R32,R33), pitch=atan2(-R31,hypot(R11,R21)),
    yaw=atan2(R21,R11). It does not project E[R] to SO(3) and does not change
    the candidate law or its moments.  At a degenerate Euler chart this proxy
    is undefined; it is not repaired by floors or altered covariance.
    """
    mean=np.asarray(mean_rotation,float)
    covariance=np.asarray(covariance_rotation,float)
    if mean.shape!=(3,3) or covariance.shape!=(9,9):
        raise ValueError('Expected a 3x3 rotation mean and column-major 9x9 covariance')
    a,b,c,d,e=mean[0,0],mean[1,0],mean[2,0],mean[2,1],mean[2,2]
    h=np.hypot(a,b)
    roll_den=d*d+e*e
    pitch_den=h*h+c*c
    if h==0. or roll_den==0. or pitch_den==0.:
        raise ValueError('Euler moment proxy is undefined in this degenerate chart')
    value=np.array([np.arctan2(d,e),np.arctan2(-c,h),np.arctan2(b,a)])
    derivative=np.zeros((3,9))
    derivative[0,5]=e/roll_den
    derivative[0,8]=-d/roll_den
    derivative[1,0]=c*a/(h*pitch_den)
    derivative[1,1]=c*b/(h*pitch_den)
    derivative[1,2]=-h/pitch_den
    derivative[2,0]=-b/(h*h)
    derivative[2,1]=a/(h*h)
    angular_covariance=derivative@covariance@derivative.T
    return value,.5*(angular_covariance+angular_covariance.T),derivative


def predict_factorized(data,torque_profile,sigma_deg_s,mode,output_dt=.01,max_step=.002):
    """Propagate any number of independent six-point angular-rate acquisitions.

    The same existing torque profile and six nonlinear trajectories are used
    as in the original 0.5 Hz experiment.  Only the representation of the
    product law changes: exact moment recursion replaces exponential product
    enumeration.  Each interval freezes its start-law; intermediate total
    increments are never applied repeatedly to an already advanced law.
    """
    start,end=np.asarray(data['time'],float)
    observations=np.asarray(data['sparse_gyro_time'],float)
    measured=np.asarray(data['sparse_gyro'],float)
    sigma=float(sigma_deg_s)
    if (observations.ndim!=1 or not len(observations) or measured.shape!=(len(observations),3)
            or not np.isfinite(observations).all() or not np.isfinite(measured).all()
            or np.any(np.diff(observations)<=0.) or observations[0]!=start
            or observations[-1]>=end or end<=start or not np.isfinite(sigma) or sigma<0.):
        raise ValueError('Original increasing observation clocks must start at the fixed initial time')
    q0=np.asarray(data['initial_quaternion_wxyz'],float)
    if q0.shape!=(4,) or not np.isfinite(q0).all() or np.linalg.norm(q0)==0.:
        raise ValueError('One finite initial orientation is required')
    q0=q0/np.linalg.norm(q0)
    parent_moment=np.outer(q0,q0)
    parent_mean=rotation_first_moment(parent_moment)
    parent_covariance=np.zeros((9,9))
    previous_rpy=euler_proxy(parent_mean,parent_covariance)[0]
    gyro_covariance=np.eye(3)*np.deg2rad(sigma)**2
    grid=np.unique(np.r_[np.linspace(start,end,int(round((end-start)/output_dt))+1),observations])
    fields=('time','quaternion_second_moment','rotation_mean','rotation_covariance_colmajor',
            'rpy_mean_rad','rpy_covariance_rad2','representative_quaternion_wxyz',
            'omega_mean_rad_s','omega_covariance_rad2_s2','block_index')
    outputs={name:[] for name in fields}
    details={}
    residuals=[]
    for block,begin in enumerate(observations):
        finish=observations[block+1] if block+1<len(observations) else end
        times=grid[(grid>=begin)&(grid<=finish)]
        prediction=integrate_six_point_increment(measured[block],gyro_covariance,times,
            torque_profile['torque'],torque_profile['inertia'],mode=mode,max_step=max_step)
        count=len(times)
        moment=np.empty((count,4,4));mean=np.empty((count,3,3));covariance=np.empty((count,9,9))
        rpy=np.empty((count,3));rpy_cov=np.empty((count,3,3));representative=np.empty((count,4))
        errors=[]
        for row in range(count):
            moment[row]=quaternion_product_second_moment(parent_moment,prediction['M2delta'][row])
            mean[row],covariance[row]=rotation_moment_update(parent_mean,parent_covariance,
                prediction['delta'][row],prediction['weights'])
            errors.append(float(np.max(np.abs(mean[row]-rotation_first_moment(moment[row])))))
            rpy[row],rpy_cov[row],_derivative=euler_proxy(mean[row],covariance[row])
            representative[row]=np.linalg.eigh(moment[row])[1][:,-1]
        rpy=np.unwrap(np.vstack((previous_rpy,rpy)),axis=0)[1:]
        keep=slice(None) if block==len(observations)-1 else slice(None,-1)
        values=dict(time=times,quaternion_second_moment=moment,rotation_mean=mean,
            rotation_covariance_colmajor=covariance,rpy_mean_rad=rpy,rpy_covariance_rad2=rpy_cov,
            representative_quaternion_wxyz=representative,
            omega_mean_rad_s=prediction['mean_omega'],omega_covariance_rad2_s2=prediction['covariance_omega'],
            block_index=np.full(count,block,dtype=int))
        for name,value in values.items():outputs[name].append(value[keep])
        for name,value in dict(time=times,delta_wxyz=prediction['delta'],omega=prediction['omega'],
                weights=prediction['weights'],parent_M2=parent_moment,parent_rotation_mean=parent_mean,
                parent_rotation_covariance=parent_covariance).items():
            details[f'block_{block}_{name}']=np.asarray(value).copy()
        residuals.append(max(errors))
        parent_moment=moment[-1].copy();parent_mean=mean[-1].copy();parent_covariance=covariance[-1].copy()
        previous_rpy=rpy[-1].copy()
    result={name:np.concatenate(values) for name,values in outputs.items()}
    result.update(details)
    result['core_vs_rotation_mean_max_abs']=np.asarray(residuals)
    return result
