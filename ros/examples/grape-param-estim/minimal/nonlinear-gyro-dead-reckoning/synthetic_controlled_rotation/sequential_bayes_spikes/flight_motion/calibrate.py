#!/usr/bin/env python3
"""Freeze synthetic motion from the fixed 40--60 s bag-8 flight reference.

Reference data are read only. Frequency selection and amplitude calibration
never inspect any reconstruction/filter result. No random draw is used here.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, periodogram
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SYNTHETIC = HERE.parents[1]
SOURCE = SYNTHETIC.parent/'gyro_mocap_sg_check/publication/bag8_50_55s'
WINDOW = (40.,60.)
MOTION_SCALE = 1.5
OLD = dict(amplitude_deg=np.array([[23.,15.,9.],[17.,10.,7.],[32.,21.,12.]]),
    frequency_hz=np.array([[.29,.73,1.17],[.37,.83,1.31],[.23,.61,1.03]]),
    phase_rad=np.array([[.2,1.1,-.7],[-.4,.8,1.7],[.5,-1.2,.4]]),offset_deg=np.zeros(3))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate(time,parameters):
    """Analytic body omega for the same RzRyRx trajectory convention."""
    time=np.asarray(time,float)
    amp=np.deg2rad(parameters['amplitude_deg']); rate=2*np.pi*np.asarray(parameters['frequency_hz'])
    phase=time[:,None,None]*rate+np.asarray(parameters['phase_rad'])
    rpy=np.deg2rad(parameters.get('offset_deg',np.zeros(3)))+np.sum(amp*np.sin(phase),axis=2)
    derivative=np.sum(amp*rate*np.cos(phase),axis=2)
    roll,pitch,_=rpy.T; rd,pd,yd=derivative.T
    omega=np.column_stack([rd-yd*np.sin(pitch),
        pd*np.cos(roll)+yd*np.sin(roll)*np.cos(pitch),
        -pd*np.sin(roll)+yd*np.cos(roll)*np.cos(pitch)])
    return rpy,omega

def metrics(x):
    x=np.rad2deg(x)
    return dict(mean_deg_s=x.mean(axis=0).tolist(),std_deg_s=x.std(axis=0).tolist(),
        rms_deg_s=np.sqrt(np.mean(x*x,axis=0)).tolist(),
        absolute_p95_deg_s=np.percentile(abs(x),95,axis=0).tolist(),
        absolute_p99_deg_s=np.percentile(abs(x),99,axis=0).tolist(),
        absolute_max_deg_s=abs(x).max(axis=0).tolist())

def spectral(time,omega,fs=120.):
    """Spectral diagnostic only; its interpolation never updates an estimate."""
    t=np.arange(float(time[0]),float(time[-1]),1/fs)
    x=np.column_stack([np.interp(t,time,omega[:,i]) for i in range(3)])
    f,p=periodogram(np.rad2deg(x),fs=fs,window='hann',detrend='linear',axis=0,scaling='density')
    return f,p

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value))

def main():
    source=np.load(SOURCE/'input.npz');sg=np.load(SOURCE/'window_320ms_sg.npz')
    lower,upper=WINDOW
    keep=(sg['time']>=lower)&(sg['time']<upper)
    time=sg['time'][keep]; local=time-lower
    q=sg['quaternion_xyzw'][keep];omega=sg['omega_body'][keep]
    rpy=np.unwrap(Rotation.from_quat(q).as_euler('xyz'),axis=0)
    # Body omega is not the derivative of roll/pitch/yaw. Frequencies come
    # from its spectrum; values fitted below are the measured RPY themselves.
    frequency,psd=spectral(time,omega)
    picked=[]; detail=[]
    for axis in range(3):
        peaks=find_peaks(psd[:,axis])[0]
        peaks=peaks[(frequency[peaks]>=.05)&(frequency[peaks]<=3.)]
        chosen=[]
        for index in peaks[np.argsort(psd[peaks,axis])[::-1]]:
            if all(abs(frequency[index]-frequency[other])>=.1-1e-9 for other in chosen):
                chosen.append(int(index))
            if len(chosen)==3:break
        if len(chosen)!=3:raise ValueError('Three resolved body-motion peaks required')
        # Keep strongest-first ordering; distinguish the spectral observation
        # from a claim that these are unique physical oscillation frequencies.
        picked.append(frequency[chosen]);
        peak_energy=np.array([np.trapz(psd[(frequency>=frequency[k]-.05-1e-9)&
            (frequency<=frequency[k]+.05+1e-9),axis],frequency[(frequency>=frequency[k]-.05-1e-9)&
            (frequency<=frequency[k]+.05+1e-9)]) for k in chosen])
        total=np.trapz(psd[:,axis],frequency)
        detail.append(dict(selected_bins=chosen,frequency_hz=frequency[chosen],
            spectral_density_deg2_s2_per_hz=psd[chosen,axis],
            plus_minus_one_bin_power_fraction=peak_energy/total,
            total_psd_integral_deg2_s2=total,
            low_1hz_power_fraction=np.trapz(psd[frequency<=1,axis],frequency[frequency<=1])/total,
            low_3hz_power_fraction=np.trapz(psd[frequency<=3,axis],frequency[frequency<=3])/total))
    picked=np.asarray(picked)
    amplitudes=[]; phases=[]; offsets=[]; fit_rpy=np.empty_like(rpy)
    for axis in range(3):
        theta=2*np.pi*local[:,None]*picked[axis]
        design=np.column_stack([np.ones(len(local)),np.sin(theta),np.cos(theta)])
        coeff=np.linalg.lstsq(design,rpy[:,axis],rcond=None)[0]
        offsets.append(coeff[0]);amplitudes.append(np.hypot(coeff[1:4],coeff[4:7]));phases.append(np.arctan2(coeff[4:7],coeff[1:4]))
        fit_rpy[:,axis]=design@coeff
    base=dict(amplitude_deg=np.rad2deg(amplitudes),frequency_hz=picked,
        phase_rad=np.asarray(phases),offset_deg=np.rad2deg(offsets))
    target=MOTION_SCALE*omega.std(axis=0)
    # The per-axis RPY amplitude multipliers are coupled by the exact body-rate
    # mapping. Solve only these three deterministic calibration parameters.
    def candidate(log_scale):
        return dict(base,amplitude_deg=base['amplitude_deg']*np.exp(log_scale)[:,None])
    def residual(log_scale):
        _,trial=evaluate(local,candidate(log_scale))
        return np.log(trial.std(axis=0)/target)
    initial=np.log(target/evaluate(local,base)[1].std(axis=0))
    fit=least_squares(residual,initial,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=100)
    profile=candidate(fit.x)
    profile.update(schema='grape/flight-calibrated-three-frequency-motion/v1',
        source_window_seconds=WINDOW,physical_motion_std_multiplier=MOTION_SCALE,
        calibration_source='bag8 mocap quaternion SG body omega; reference only, never filter input',
        frequency_rule='three strongest distinct body-omega Hann periodogram local maxima in 0.05..3 Hz, >=0.10 Hz apart',
        selection_uses_reconstruction_result=False)
    (HERE/'flight_motion_profile.json').write_text(json.dumps(profile,indent=2,default=serial,allow_nan=False)+'\n')
    print('PROFILE READY: '+str(HERE/'flight_motion_profile.json'),flush=True)
    rpy_new,omega_new=evaluate(local,profile);rpy_old,omega_old=evaluate(local,OLD)
    raw_keep=(source['gyro_time']>=lower)&(source['gyro_time']<upper)
    pose_keep=(source['mocap_time']>=lower)&(source['mocap_time']<upper)
    raw_time=source['gyro_time'][raw_keep];raw=source['gyro_rad_s'][raw_keep]
    pose=source['mocap_position'][pose_keep]
    fs,ps=spectral(time,omega_new);fo,po=spectral(time,omega_old);fr,pr=spectral(raw_time,raw,200.)
    metadata=json.loads((SOURCE/'input.json').read_text())
    calibration=dict(schema='grape/flight-motion-calibration/v1',
        source_bag_path=metadata['bag_path'],source_bag_copied=False,
        source_time_origin_ns=metadata['bag_origin_record_ns'],source_time_origin='first original bag record; gyro/mocap use original header stamps',
        requested_window_seconds=WINDOW,reason_for_fixed_window='Existing independent SG diagnostic fixed 40--60 s before current reconstruction results; no best segment selection',
        source_files={str((SOURCE/name).relative_to(SYNTHETIC.parent)):sha(SOURCE/name) for name in ['input.npz','input.json','window_320ms_sg.npz']},
        source_gyro_topic='/gimbalrotor/imu',source_mocap_topic='/gimbalrotor/mocap/pose',
        source_gyro_field='spinal/Imu.gyro, rad/s in body/FC frame; main_body-to-FC rotation verified identity in source extraction',
        source_quaternion='xyzw Hamilton body-to-world quaternion; world frame mocap position',
        native_gyro_count=len(raw_time),native_gyro_mean_rate_hz=(len(raw_time)-1)/(raw_time[-1]-raw_time[0]),
        mocap_sg_count=len(time),mocap_mean_rate_hz=(len(time)-1)/(time[-1]-time[0]),
        mocap_z_world_m=dict(mean=pose[:,2].mean(),minimum=pose[:,2].min(),maximum=pose[:,2].max(),
            initial_0to8p5s_mean_from_prior_audit=.24537,mean_above_initial=pose[:,2].mean()-.24537),
        flight_evidence='mocap Z mean ~0.847 m vs initial 0.245 m, ~0.602 m above initial; this confirms sustained height, not full asymptotic flight stability',
        smoothing='Existing offline centered 0.32 s cubic SG quaternion fit on original irregular mocap stamps; differentiated normalization, omega=2 vec(q^-1 qdot); no gyro smoothing used as raw measurement',
        spectrum=dict(rate_hz=120.,frequency_resolution_hz=float(frequency[1]-frequency[0]),
            method='Full 20 s Hann periodogram with linear detrend; interpolation to uniform grid for spectral diagnostic only',
            choice_band_hz=[.05,3.],minimum_peak_separation_hz=.1,per_axis=detail,
            scope='Finite-window spectral maxima, not uniquely identified physical mode frequencies or sensor noise'),
        harmonic_fit='Ordinary least squares measured SG RPY = constant + three sine/cosine pairs at fixed body-omega peaks; no trend or extra component; convert each pair to amplitude/phase',
        fit_rpy_rmse_deg=np.rad2deg(np.sqrt(np.mean((fit_rpy-rpy)**2,axis=0))),
        fit_body_omega_rmse_deg_s=np.rad2deg(np.sqrt(np.mean((evaluate(local,base)[1]-omega)**2,axis=0))),
        fit_body_omega_variance_explained=1-np.mean((evaluate(local,base)[1]-omega)**2,axis=0)/omega.var(axis=0),
        unrepresented_motion='Three global harmonics do not reproduce nonstationary real flight or all spectral power; fitted profile is a smooth statistical-scale test, not real-flight replay',
        amplitude_calibration=dict(target='1.5 times real SG body-omega per-axis standard deviation',
            multiplier=MOTION_SCALE,target_std_deg_s=np.rad2deg(target),rpy_axis_amplitude_multipliers=np.exp(fit.x),
            residual_log_std_ratio=residual(fit.x),least_squares_nfev=fit.nfev,success=bool(fit.success),
            parameter_selection_completed_before_filter_run=True,filters_or_estimated_covariances_used=False),
        raw_gyro=metrics(raw),mocap_sg_body_omega=metrics(omega),old_synthetic=metrics(omega_old),new_synthetic=metrics(omega_new),
        new_rpy_range_deg=np.rad2deg(np.ptp(rpy_new,axis=0)),
        fitted_profile_sha256=sha(HERE/'flight_motion_profile.json'),calibrator_sha256=sha(Path(__file__)),
        inference_changed=False,random_numbers_used=False,core_changed=False,
        sensor_white_noise_or_spike_parameters_changed=False)
    np.savez_compressed(HERE/'comparison.npz',
        real_sg_time=time,real_sg_quaternion_xyzw=q,real_sg_body_omega_rad_s=omega,
        real_sg_rpy_rad=rpy,synthetic_local_time=local,new_synthetic_rpy_rad=rpy_new,
        old_synthetic_rpy_rad=rpy_old,new_synthetic_body_omega_rad_s=omega_new,
        old_synthetic_body_omega_rad_s=omega_old,real_raw_gyro_time=raw_time,real_raw_gyro_rad_s=raw,
        real_raw_gyro_stamp_ns=source['gyro_stamp_ns'][raw_keep],
        real_raw_gyro_record_ns=source['gyro_record_ns'][raw_keep],real_raw_gyro_source_index=source['gyro_source_index'][raw_keep],
        real_mocap_time=source['mocap_time'][pose_keep],real_mocap_stamp_ns=source['mocap_stamp_ns'][pose_keep],
        real_mocap_record_ns=source['mocap_record_ns'][pose_keep],real_mocap_source_index=source['mocap_source_index'][pose_keep],
        real_mocap_position=pose,real_mocap_quaternion_xyzw=source['mocap_quaternion_xyzw'][pose_keep],
        frequency_hz=frequency,real_sg_psd_deg2_s2_per_hz=psd,new_synthetic_psd_deg2_s2_per_hz=ps,
        old_synthetic_psd_deg2_s2_per_hz=po,raw_spectrum_frequency_hz=fr,raw_gyro_psd_deg2_s2_per_hz=pr)
    (HERE/'calibration.json').write_text(json.dumps(calibration,indent=2,default=serial,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    labels=['x','y','z']
    for axis in range(3):
        ax=axes[0,axis];ax.plot(raw_time-lower,np.rad2deg(raw[:,axis]),color='0.65',lw=.65,label='Flight raw gyro')
        ax.plot(local,np.rad2deg(omega[:,axis]),color='black',lw=1.0,label='Flight mocap-SG omega')
        ax.plot(local,np.rad2deg(omega_new[:,axis]),color='#0072B2',lw=1.1,label='New synthetic (1.5 x SD)')
        ax.set(title='Body omega_'+labels[axis],ylabel='deg/s',xlabel='s from start of 20 s window');ax.grid(alpha=.25)
        if axis==0:ax.legend(fontsize=8)
        ax=axes[1,axis];ax.plot(local,np.rad2deg(omega_old[:,axis]),color='#D55E00',lw=.8,label='Old synthetic')
        ax.plot(local,np.rad2deg(omega_new[:,axis]),color='#0072B2',lw=1.1,label='New synthetic')
        ax.set(ylabel='deg/s',xlabel='s',title='Old vs new, same vertical units');ax.grid(alpha=.25)
        if axis==0:ax.legend(fontsize=8)
        ax=axes[2,axis]
        ax.semilogy(fr,pr[:,axis],color='0.65',lw=.7,label='Flight raw')
        ax.semilogy(frequency,psd[:,axis],color='black',lw=1.1,label='Flight mocap-SG')
        ax.semilogy(fs,ps[:,axis],color='#0072B2',lw=1.1,label='New synthetic')
        ax.semilogy(fo,po[:,axis],color='#D55E00',lw=.7,alpha=.7,label='Old synthetic')
        ax.scatter(picked[axis],psd[[int(np.argmin(abs(frequency-p))) for p in picked[axis]],axis],marker='x',color='black',s=50,zorder=10)
        ax.set(xlim=(0,3),ylim=(1e-6,1e6),xlabel='Hz',ylabel='(deg/s)^2 / Hz',title='Hann spectrum; selected real peaks marked x');ax.grid(alpha=.25)
        if axis==0:ax.legend(fontsize=8)
    fig.suptitle('Motion calibrated to bag 8, 40-60 s: frequencies from real body motion; 1.5 x per-axis SD\nSensor white noise and spikes remain separate and unchanged',fontsize=13)
    fig.savefig(HERE/'comparison.png',dpi=150);fig.savefig(HERE/'comparison.pdf');plt.close(fig)
    print(json.dumps(dict(frequency_hz=picked,new_synthetic=metrics(omega_new),calibration_body_rpy_multiplier=np.exp(fit.x)),default=serial,indent=2))

if __name__=='__main__':main()
