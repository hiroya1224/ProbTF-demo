"""Figures drawn directly from the saved finite-law moments and raw references."""
from pathlib import Path
import csv
import os
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLORS={'nonlinear':'#1769aa','linear_forced':'#009688','constant':'#de7c22'}
LABELS={'nonlinear':'Controlled nonlinear Euler','linear_forced':'Controlled, no gyroscopic term',
        'constant':'Held angular velocity (original)'}
METHODS=('nonlinear','linear_forced','constant')
AXES=('Roll','Pitch','Yaw')


def decorate(axis,data,start):
    for time in data['sparse_gyro_time']:
        axis.axvline(time-start,color='.5',lw=.8,ls=':',alpha=.6)
    axis.grid(alpha=.25)
    axis.set_xlim(0.,float(data['time'][1]-start))


def rpy_figure(data,cases,sigma):
    start=float(data['time'][0])
    figure,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for dimension,axis in enumerate(axes):
        base=cases['nonlinear'];time=base['time']-start
        axis.plot(time,np.rad2deg(base['reference_rpy_rad'][:,dimension]),'k',lw=1.6,label='Mocap reference')
        for method in METHODS:
            value=cases[method];mean=np.rad2deg(value['rpy_mean_rad'][:,dimension])
            std=np.rad2deg(np.sqrt(value['rpy_covariance_rad2'][:,dimension,dimension]))
            if method!='linear_forced':
                axis.fill_between(time,mean-2*std,mean+2*std,color=COLORS[method],
                                  alpha=.18 if method=='nonlinear' else .1,lw=0)
            axis.plot(time,mean,color=COLORS[method],lw=1.8 if method=='nonlinear' else 1.2,
                      ls='-' if method=='nonlinear' else '--' if method=='constant' else ':',label=LABELS[method])
        axis.set_ylabel(AXES[dimension]+' [deg]')
        decorate(axis,data,start)
    axes[0].legend(loc='best',fontsize=9)
    axes[-1].set_xlabel('Time from first sparse gyro [s]')
    figure.suptitle(f'Bag 8: 0.5 Hz gyro, assumed sigma = {sigma:g} deg/s per axis\n'
        'Mocap initializes attitude once; bands are mean +/- 2 weighted standard deviations',fontsize=12)
    figure.tight_layout(rect=(0,.015,1,.93))
    return figure


def omega_figure(data,cases,sigma):
    start=float(data['time'][0])
    figure,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for dimension,axis in enumerate(axes):
        axis.plot(data['dense_imu_time']-start,np.rad2deg(data['dense_gyro'][:,dimension]),
            color='.55',lw=.9,label='Native high-rate gyro (reference only)')
        for method in METHODS:
            value=cases[method];mean=np.rad2deg(value['omega_mean_rad_s'][:,dimension])
            std=np.rad2deg(np.sqrt(value['omega_covariance_rad2_s2'][:,dimension,dimension]))
            if method=='nonlinear':
                axis.fill_between(value['time']-start,mean-2*std,mean+2*std,color=COLORS[method],alpha=.18,lw=0)
            axis.plot(value['time']-start,mean,color=COLORS[method],lw=1.5,
                ls='-' if method=='nonlinear' else '--' if method=='constant' else ':',label=LABELS[method])
        axis.errorbar(data['sparse_gyro_time']-start,np.rad2deg(data['sparse_gyro'][:,dimension]),
            yerr=2*sigma,fmt='o',color='k',ms=4,capsize=3,label='Actual 0.5 Hz readings +/- 2 sigma')
        axis.set_ylabel(('wx','wy','wz')[dimension]+' [deg/s]')
        decorate(axis,data,start)
    axes[0].legend(loc='best',fontsize=8)
    axes[-1].set_xlabel('Time from first sparse gyro [s]')
    figure.suptitle(f'Angular-velocity prediction between acquisitions / sigma = {sigma:g} deg/s\n'
                   'Vertical changes at 2/4 s replace the rate input, never the attitude',fontsize=12)
    figure.tight_layout(rect=(0,.015,1,.93))
    return figure


def uncertainty_figure(data,cases,sigma):
    start=float(data['time'][0])
    figure,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for dimension,axis in enumerate(axes[:3]):
        for method in METHODS:
            value=cases[method]
            width=2*np.rad2deg(np.sqrt(value['rpy_covariance_rad2'][:,dimension,dimension]))
            axis.plot(value['time']-start,width,color=COLORS[method],lw=1.5,label=LABELS[method])
        axis.set_ylabel(AXES[dimension]+' 2sigma [deg]')
        decorate(axis,data,start)
    for method in METHODS:
        value=cases[method]
        axes[3].plot(value['time']-start,np.rad2deg(value['error_geodesic_rad']),
                     color=COLORS[method],lw=1.5,label=LABELS[method])
    axes[3].set_ylabel('SO(3) error [deg]')
    decorate(axes[3],data,start)
    axes[0].legend(loc='best',fontsize=9)
    axes[-1].set_xlabel('Time from first sparse gyro [s]')
    figure.suptitle(f'Propagated spread and mocap comparison / sigma = {sigma:g} deg/s\n'
                   'Conditional on fixed nominal inertia, actuator inputs and past residual-torque mean',fontsize=12)
    figure.tight_layout(rect=(0,.015,1,.93))
    return figure


def render_case(destination,data,torque,cases,record):
    destination=Path(destination)
    with PdfPages(destination/'comparison.pdf') as pdf:
        for name,make in (('rpy',rpy_figure),('omega',omega_figure),('uncertainty',uncertainty_figure)):
            figure=make(data,cases,record['sigma_deg_s'])
            figure.savefig(destination/(name+'.png'),dpi=170)
            pdf.savefig(figure)
            plt.close(figure)


def render_overview(destination,data,summaries):
    destination=Path(destination)
    fields=['sigma_deg_s','method','roll_rmse_deg','pitch_rmse_deg','yaw_rmse_deg',
            'geodesic_rmse_deg','geodesic_final_deg','roll_final_2sigma_deg',
            'pitch_final_2sigma_deg','yaw_final_2sigma_deg','max_core_product_enumeration_residual']
    with (destination/'comparison_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for summary in summaries:
            for method,metric in summary['methods'].items():
                row=dict(sigma_deg_s=summary['sigma_deg_s'],method=method,
                    geodesic_rmse_deg=metric['geodesic_rmse_deg'],
                    geodesic_final_deg=metric['geodesic_final_deg'],
                    max_core_product_enumeration_residual=metric['max_core_product_enumeration_residual'])
                for dimension,axis in enumerate(('roll','pitch','yaw')):
                    row[axis+'_rmse_deg']=metric['rpy_rmse_deg'][dimension]
                    row[axis+'_final_2sigma_deg']=metric['rpy_final_two_sigma_deg'][dimension]
                writer.writerow(row)
    figure,axes=plt.subplots(len(summaries),3,figsize=(14,3*len(summaries)),sharex=True,squeeze=False)
    start=float(data['time'][0])
    for row,summary in enumerate(summaries):
        sigma=summary['sigma_deg_s'];label=format(sigma,'g').replace('.','p')
        value=dict(np.load(destination/f'sigma_{label}_deg_s/nonlinear.npz'))
        for dimension,axis in enumerate(axes[row]):
            time=value['time']-start
            mean=np.rad2deg(value['rpy_mean_rad'][:,dimension])
            std=np.rad2deg(np.sqrt(value['rpy_covariance_rad2'][:,dimension,dimension]))
            axis.fill_between(time,mean-2*std,mean+2*std,color=COLORS['nonlinear'],alpha=.2)
            axis.plot(time,mean,color=COLORS['nonlinear'],lw=1.3,label='Nonlinear mean +/- 2sigma')
            axis.plot(time,np.rad2deg(value['reference_rpy_rad'][:,dimension]),color='k',lw=1.1,label='Mocap')
            axis.set_title(f'{AXES[dimension]} / sigma={sigma:g} deg/s')
            decorate(axis,data,start)
            axis.set_ylabel('Angle [deg]')
            if row==len(summaries)-1: axis.set_xlabel('Time [s]')
    axes[0,0].legend(fontsize=8)
    figure.suptitle('Controlled nonlinear six-point prediction: separate assumed gyro noise levels',fontsize=14)
    figure.tight_layout(rect=(0,0,1,.97))
    figure.savefig(destination/'noise_overview.png',dpi=180)
    figure.savefig(destination/'noise_overview.pdf')
    plt.close(figure)
