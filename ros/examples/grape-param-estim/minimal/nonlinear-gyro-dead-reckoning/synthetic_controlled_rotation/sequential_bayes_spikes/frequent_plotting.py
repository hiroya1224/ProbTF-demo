"""Show native noisy measurements and the SAME saved Bayesian moments.

No estimation, covariance fitting, smoothing or sample selection is done here.
All native sensor samples are drawn. The fixed 2--4 s zoom is a display view,
not a sensor or inference window. Reference subtraction is diagnostic only.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation


plt.rcParams['path.simplify'] = False
COLORS = {'nonlinear': '#1769aa', 'linear_forced': '#ba4141'}
LABELS = {'nonlinear': 'Nonlinear Euler', 'linear_forced': 'No gyroscopic term'}
STYLES = {'nonlinear': '-', 'linear_forced': '--'}
RAW = '#738b88'
SPIKE = '#bc2026'
CASES = ('nonlinear', 'linear_forced')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _halfwidth(covariance):
    return 2.*np.rad2deg(np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)))


def _find(metadata, key):
    if not isinstance(metadata, dict):
        return None
    if key in metadata:
        return metadata[key]
    for value in metadata.values():
        if isinstance(value, dict):
            result = _find(value, key)
            if result is not None:
                return result
    return None


def _settings(source, acquisitions, sensor, cases, metadata):
    sensor_time = np.asarray(sensor['time'])
    intervals = np.diff(np.asarray(acquisitions['time']))
    rate = float(1./np.median(intervals)) if len(intervals) else None
    native_rate = float(1./np.median(np.diff(sensor_time)))
    covariance = np.asarray(cases['nonlinear']['measurement_covariance_rad2_s2'])
    assumed = np.rad2deg(np.sqrt(np.diag(covariance)))
    white_residual = np.asarray(source['gyro_realized_noise_rad_s'])
    empirical = np.rad2deg(white_residual.std(axis=0, ddof=1))
    generation_covariance = source.get('generator_gyro_covariance_rad2_s2')
    generated = (np.rad2deg(np.sqrt(np.diag(generation_covariance))).tolist()
                 if generation_covariance is not None else None)
    active_until = _find(metadata, 'spike_active_until_seconds')
    spike_rate = _find(metadata, 'spike_rate_hz')
    if spike_rate is None:
        spike_rate = _find(metadata, 'spike_expected_arrivals_per_second')
    return dict(rate_hz=rate, native_rate_hz=native_rate,
        assumed_sigma_xyz_deg_s=assumed.tolist(),
        generated_white_sigma_xyz_deg_s=generated,
        empirical_white_sigma_xyz_deg_s=empirical.tolist(),
        native_sample_count=len(sensor_time),
        acquired_sample_count=len(acquisitions['time']),
        contaminated_acquisition_count=int(np.count_nonzero(acquisitions['spike_mask'])),
        native_contaminated_sample_count=int(np.count_nonzero(sensor['spike_mask'])),
        spike_rate_hz=spike_rate,
        spike_active_until_seconds=(float(active_until) if active_until is not None else None),
        fixed_raw_zoom_seconds=[2., 4.])


def _native_truth(source, sensor_time):
    if ('gyro_true_body_rad_s' in source
            and len(source['gyro_true_body_rad_s']) == len(sensor_time)):
        return np.asarray(source['gyro_true_body_rad_s'])
    return np.column_stack([np.interp(sensor_time, source['reference_time'],
        source['reference_omega_body_rad_s'][:, axis]) for axis in range(3)])


def _spike_caption(settings):
    rate=settings['spike_rate_hz']
    active=settings['spike_active_until_seconds']
    text=(f'Poisson spike starts: {float(rate):g}/s' if rate is not None else 'Random native spike starts')
    if active is not None:text+=f' until {active:g} s, then a deliberate clean tail'
    return text+f'; acquired contaminated readings: {settings["contaminated_acquisition_count"]}.'


def _reference(source, times, key):
    return np.column_stack([np.interp(times, source['reference_time'],
        source[key][:, axis]) for axis in range(3)])


def _error(source, case):
    truth = Rotation.from_euler('xyz', _reference(source, case['time'], 'reference_rpy_rad'))
    estimate = Rotation.from_euler('xyz', case['rpy_proxy_rad'])
    return np.rad2deg((truth.inv()*estimate).magnitude())


def _recovery(axis, settings):
    stamp = settings['spike_active_until_seconds']
    low, high = axis.get_xlim()
    if stamp is not None and low < stamp < high:
        axis.axvspan(stamp, high, facecolor='.85', alpha=.30, zorder=0)
        axis.axvline(stamp, color='.45', ls=':', lw=1.)
        axis.text((stamp+high)/2., .985, 'clean tail',
                  transform=axis.get_xaxis_transform(), ha='center', va='top',
                  fontsize=8, color='.35')


def _event_lines(axis, acquisitions):
    for stamp, bad in zip(acquisitions['time'], acquisitions['spike_mask']):
        axis.axvline(stamp, color=SPIKE if bad else '.6', alpha=.28,
                     lw=.9 if bad else .5, ls=':' if bad else '-', zorder=0)


def _save(figure, output, stem):
    figure.savefig(output/(stem+'.pdf'))
    figure.savefig(output/(stem+'.png'), dpi=160)
    plt.close(figure)
    return [stem+'.pdf', stem+'.png']


def _mean_legend(names=CASES, raw=True):
    values = [Line2D([], [], color='black', lw=1.8, label='Synthetic truth')]
    if raw:
        values.append(Line2D([], [], color=RAW, lw=.8, label='Raw gyro: every native sample'))
    values.extend(Line2D([], [], color=COLORS[name], ls=STYLES[name], lw=1.7,
                         label=LABELS[name]) for name in names)
    values.extend((Line2D([], [], color='.25', marker='o', mfc='white', ls='',
                          label='Acquired gyro: clean'),
                   Line2D([], [], color=SPIKE, marker='x', ls='',
                          label='Acquired gyro: spike present')))
    return values


def _comparison(output, source, acquisitions, sensor, cases, settings, names, stem):
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9), sharex=True)
    reference_time = np.asarray(source['reference_time'])
    lower, upper = float(reference_time[0]), float(reference_time[-1])
    contaminated = np.asarray(acquisitions['spike_mask'], bool)
    for dimension in range(3):
        top, bottom = axes[:, dimension]
        top.plot(reference_time, np.rad2deg(source['reference_rpy_rad'][:, dimension]),
                 color='black', lw=1.9, zorder=5)
        bottom.plot(sensor['time'], np.rad2deg(sensor['gyro_spike'][:, dimension]),
                    color=RAW, lw=.55, alpha=.85, zorder=2)
        bottom.plot(reference_time, np.rad2deg(source['reference_omega_body_rad_s'][:, dimension]),
                    color='black', lw=1.5, zorder=4)
        for name in names:
            case = cases[name]
            for axis, mean_key, cov_key in (
                    (top, 'rpy_proxy_rad', 'rpy_covariance_delta_rad2'),
                    (bottom, 'omega_mean_rad_s', 'omega_covariance_rad2_s2')):
                mean = np.rad2deg(case[mean_key][:, dimension])
                half = _halfwidth(case[cov_key])[:, dimension]
                axis.fill_between(case['time'], mean-half, mean+half,
                                  color=COLORS[name], alpha=.16, lw=0, zorder=1)
                axis.plot(case['time'], mean, color=COLORS[name], ls=STYLES[name], lw=1.5, zorder=6)
                if mean_key=='omega_mean_rad_s':
                    axis.plot(case['time'],mean-half,color=COLORS[name],ls=STYLES[name],
                              lw=.65,alpha=.85,zorder=3)
                    axis.plot(case['time'],mean+half,color=COLORS[name],ls=STYLES[name],
                              lw=.65,alpha=.85,zorder=3)
        good = ~contaminated
        observed = np.rad2deg(acquisitions['gyro_spike'][:, dimension])
        bottom.scatter(acquisitions['time'][good], observed[good],
                       facecolors='white', edgecolors='.2', marker='o', s=24, lw=.8, zorder=8)
        bottom.scatter(acquisitions['time'][contaminated], observed[contaminated],
                       color=SPIKE, marker='x', s=46, lw=1.4, zorder=9)
        for row, axis in enumerate((top, bottom)):
            axis.set_xlim(lower, upper)
            axis.set_title((('Roll','Pitch','Yaw') if row == 0 else
                            ('omega x','omega y','omega z'))[dimension], fontsize=13)
            axis.set_ylabel('deg' if row == 0 else 'deg/s', fontsize=11)
            axis.tick_params(labelsize=10)
            axis.grid(alpha=.22)
            _event_lines(axis, acquisitions)
            _recovery(axis, settings)
        bottom.set_xlabel('Data time [s]', fontsize=11)
    title = ' / '.join(LABELS[name] for name in names)
    figure.suptitle(f'{title} — same spiky gyro; {settings["rate_hz"]:g} Hz updates', fontsize=15, y=.993)
    legend=_mean_legend(names)
    figure.legend(handles=legend, loc='upper center',
                  bbox_to_anchor=(.5,.955), ncol=3, frameon=False, fontsize=10)
    figure.text(.035,.015,
        'Raw curve: every native measurement, including white noise and the realized short spikes. '
        'Only the marked acquired samples update the estimator.\n'
        'Both models use the same initial law, J, torque, readings and Bayes updates. Only omega cross (J omega) is omitted in the red model.\n'
        'Bands: saved +/-2sigma; RPY is a local display approximation, not a guaranteed 95% region. '+_spike_caption(settings), fontsize=9.3)
    figure.tight_layout(rect=(0,.085,1,.88))
    return _save(figure, output, stem)


def _raw_sensor(output, source, acquisitions, sensor, settings):
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 9.5))
    time = np.asarray(sensor['time'])
    truth = np.rad2deg(_native_truth(source,time))
    spike = np.rad2deg(sensor['gyro_spike'])
    windows = [(float(time[0]),float(time[-1])),(2.,4.)]
    for dimension in range(3):
        for column, window in enumerate(windows):
            axis = axes[dimension,column]
            selected = (time>=window[0]) & (time<=window[1])
            axis.plot(time[selected], spike[selected,dimension], color=RAW,lw=.7,
                      label='Shared raw gyro with spikes')
            axis.plot(time[selected],truth[selected,dimension],color='black',lw=1.2,label='Synthetic truth')
            observed = (acquisitions['time']>=window[0])&(acquisitions['time']<=window[1])
            bad = observed & acquisitions['spike_mask']
            good = observed & ~acquisitions['spike_mask']
            axis.scatter(acquisitions['time'][good],np.rad2deg(acquisitions['gyro_spike'][good,dimension]),
                         marker='o',facecolors='white',edgecolors='.2',s=25,lw=.8,zorder=6)
            axis.scatter(acquisitions['time'][bad],np.rad2deg(acquisitions['gyro_spike'][bad,dimension]),
                         marker='x',color=SPIKE,s=45,lw=1.4,zorder=7)
            axis.set_xlim(*window)
            axis.set_ylabel('omega '+'xyz'[dimension]+' [deg/s]')
            axis.grid(alpha=.22)
            _recovery(axis,settings)
            if dimension == 2:axis.set_xlabel('Data time [s]')
        axes[dimension,0].set_title('All native samples: full interval',fontsize=12)
        axes[dimension,1].set_title('Fixed zoom: 2–4 s (not selected by results)',fontsize=12)
    handles,labels=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.953),ncol=3,frameon=False)
    figure.suptitle('The same spiky measurements used by both dynamics models',fontsize=16,y=.994)
    empirical=', '.join(f'{value:.3f}' for value in settings['empirical_white_sigma_xyz_deg_s'])
    figure.text(.04,.015,
        f'Native rate {settings["native_rate_hz"]:g} Hz; every raw point is drawn without averaging or downsampling. '
        f'Empirical white sigma xyz = ({empirical}) deg/s.\n'
        'Open circles: acquired clean readings. Red crosses: acquired readings containing a spike. '
        'Unmarked native spikes are not secretly fused.\n'+_spike_caption(settings),fontsize=9.5)
    figure.tight_layout(rect=(0,.09,1,.91))
    return _save(figure,output,'native_gyro_full_and_zoom')


def _innovations(acquisitions,case):
    stamps=[];values=[]
    for stamp,observed in zip(acquisitions['time'][1:],acquisitions['gyro_spike'][1:]):
        row=np.flatnonzero((case['time']==stamp)&(case['event_side']==-1))
        if not len(row):raise ValueError('Saved pre-observation state is required')
        values.append(np.rad2deg(observed-case['omega_mean_rad_s'][row[0]]))
        stamps.append(stamp)
    return np.asarray(stamps),np.asarray(values)


def _diagnostics(output,source,acquisitions,cases,settings):
    figure,axes=plt.subplots(2,3,figsize=(15.5,9))
    for dimension,axis in enumerate(axes[0]):
        for name in CASES:
            case=cases[name]
            width=_halfwidth(case['omega_covariance_rad2_s2'])[:,dimension]
            axis.plot(case['time'],width,color=COLORS[name],ls=STYLES[name],lw=1.5)
            for side,filled in ((-1,False),(1,True)):
                selected=case['event_side']==side
                axis.scatter(case['time'][selected],width[selected],s=20,
                    facecolors=COLORS[name] if filled else 'white',edgecolors=COLORS[name],zorder=5)
        axis.set_title(r'Rate half-width $2\sigma_{\omega_'+ 'xyz'[dimension] + r'}$',fontsize=12)
        axis.set_ylabel('deg/s')
    for name in CASES:
        case=cases[name]
        axes[1,0].plot(case['time'],_error(source,case),color=COLORS[name],ls=STYLES[name],lw=1.5)
        stamp,innovation=_innovations(acquisitions,case)
        axes[1,1].plot(stamp,np.linalg.norm(innovation,axis=1),marker='o',ls='',ms=4.5,color=COLORS[name])
        axes[1,2].plot(case['time'],np.linalg.norm(case['rotation_omega_covariance_colmajor'],axis=(1,2)),
                       color=COLORS[name],ls=STYLES[name],lw=1.5)
    axes[1,0].set_title('Attitude-center SO(3) error');axes[1,0].set_ylabel('deg')
    axes[1,1].set_title('Acquired gyro innovation norm');axes[1,1].set_ylabel('deg/s')
    axes[1,2].set_title('Rotation/rate cross-covariance norm');axes[1,2].set_ylabel('rad/s')
    for axis in axes.flat:
        axis.set_xlim(float(source['reference_time'][0]),float(source['reference_time'][-1]))
        axis.set_xlabel('Data time [s]');axis.grid(alpha=.22)
        _event_lines(axis,acquisitions);_recovery(axis,settings)
    figure.suptitle('Same spiky observations: nonlinear Euler versus no gyroscopic term',fontsize=15,y=.994)
    figure.legend(handles=_mean_legend(raw=False)[1:3],loc='upper center',bbox_to_anchor=(.5,.953),
                  ncol=2,frameon=False)
    figure.text(.04,.015,
        'Open circles: saved prior. Filled circles: saved posterior. Innovation markers exist only at actual acquisitions.\n'
        'All bands and cross terms are those used by estimation; a narrow posterior is not a guarantee that a spiked reading is valid.',
        fontsize=9.5)
    figure.tight_layout(rect=(0,.07,1,.9))
    return _save(figure,output,'bayes_diagnostics')


def _gyro_uncertainty(output,source,acquisitions,cases,settings):
    """Expose actual saved angular-rate uncertainty without visual rescaling."""
    figure,axes=plt.subplots(2,3,figsize=(15.5,9),sharex=True)
    for dimension in range(3):
        top,bottom=axes[:,dimension]
        top.axhline(0.,color='black',lw=.8,alpha=.6)
        for name in CASES:
            case=cases[name]
            truth=_reference(source,case['time'],'reference_omega_body_rad_s')[:,dimension]
            error=np.rad2deg(case['omega_mean_rad_s'][:,dimension]-truth)
            half=_halfwidth(case['omega_covariance_rad2_s2'])[:,dimension]
            color,style=COLORS[name],STYLES[name]
            top.fill_between(case['time'],error-half,error+half,color=color,alpha=.16,lw=0)
            top.plot(case['time'],error,color=color,ls=style,lw=1.4)
            top.plot(case['time'],error-half,color=color,ls=style,lw=.65,alpha=.85)
            top.plot(case['time'],error+half,color=color,ls=style,lw=.65,alpha=.85)
            bottom.plot(case['time'],half,color=color,ls=style,lw=1.5)
            for side,filled in ((-1,False),(1,True)):
                selected=case['event_side']==side
                bottom.scatter(case['time'][selected],half[selected],s=20,
                    facecolors=color if filled else 'white',edgecolors=color,zorder=5)
        top.set_title('omega '+'xyz'[dimension]+' error +/-2sigma',fontsize=12)
        bottom.set_title('2sigma half-width: omega '+'xyz'[dimension],fontsize=12)
        top.set_ylabel('mean - truth [deg/s]')
        bottom.set_ylabel('2sigma [deg/s]')
        bottom.set_xlabel('Data time [s]')
        for axis in (top,bottom):
            axis.set_xlim(float(source['reference_time'][0]),float(source['reference_time'][-1]))
            axis.grid(alpha=.22)
            _event_lines(axis,acquisitions);_recovery(axis,settings)
    figure.suptitle('Angular velocity is estimated: saved full-covariance marginals',fontsize=15,y=.994)
    figure.legend(handles=_mean_legend(raw=False)[1:3],loc='upper center',
                  bbox_to_anchor=(.5,.953),ncol=2,frameon=False)
    figure.text(.04,.015,
        'Top: estimated rate minus read-only synthetic truth, with the SAME saved +/-2sigma; reference values never return to the filter.\n'
        'Bottom: 2*sqrt(diag(saved angular-rate covariance)); open markers are priors, filled markers are posteriors. No scaling or clipping.',
        fontsize=9.5)
    figure.tight_layout(rect=(0,.07,1,.9))
    return _save(figure,output,'gyro_uncertainty')


def _visible(case,stamp,phase):
    at=case['time']==stamp
    if phase=='prior':at &= case['event_side']!=1
    return (case['time']<stamp)|at


def _band_vertices(time,mean,width):
    return np.column_stack((np.r_[time,time[::-1]],np.r_[mean-width,(mean+width)[::-1]]))


def _schedule(acquisitions,lower,upper,fps):
    result=[];begin=lower
    events=list(np.asarray(acquisitions['time'])[1:])
    for end in events+[upper]:
        count=max(1,round(30.*(end-begin)/(upper-lower)*fps))
        result.extend((float(t),'prediction') for t in np.linspace(begin,end,count,endpoint=False))
        if end in events:
            for phase in ('prior','posterior'):
                result.extend([(float(end),phase)]*max(1,round(.25*fps)))
        begin=end
    result.extend([(upper,'prediction')]*fps)
    return result


def _movie(output,source,acquisitions,sensor,cases,settings,fps=12):
    width,height=960,540
    figure,axes=plt.subplots(2,3,figsize=(width/100,height/100),dpi=100)
    figure.subplots_adjust(left=.066,right=.985,bottom=.18,top=.81,wspace=.28,hspace=.44)
    figure.suptitle('Same spiky gyro: nonlinear Euler vs no gyroscopic term',fontsize=14,y=.986)
    handles=_mean_legend()
    figure.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),
                  ncol=3,frameon=False,fontsize=7.8)
    figure.text(.045,.07,'Same spiky measurements and Bayes updates; only the red model omits omega cross (J omega). Bands: saved +/-2sigma.',
                fontsize=8)
    empirical=float(np.mean(settings['empirical_white_sigma_xyz_deg_s']))
    assumed=float(np.mean(settings['assumed_sigma_xyz_deg_s']))
    rate=settings['spike_rate_hz']
    rate_note=f'; spike starts {float(rate):g}/s' if rate is not None else ''
    figure.text(.045,.045,
        f'White sigma observed {empirical:.2f} deg/s; assumed {assumed:g} deg/s{rate_note}. Gray tail deliberately clean.',
        fontsize=8)
    clock=figure.text(.045,.013,'',fontsize=9,color='#334b65',animated=True)
    phase_text=figure.text(.98,.013,'',fontsize=9,ha='right',animated=True)
    width_texts={
        'nonlinear':figure.text(.045,.096,'',fontsize=8,color=COLORS['nonlinear'],animated=True),
        'linear_forced':figure.text(.98,.096,'',fontsize=8,ha='right',
                                   color=COLORS['linear_forced'],animated=True)}
    dynamic=[clock,phase_text,*width_texts.values()];layers=[];cursors=[]
    reference_time=np.asarray(source['reference_time'])
    lower,upper=float(reference_time[0]),float(reference_time[-1])
    for dimension in range(3):
        for row,(mean_key,cov_key,truth_key) in enumerate((
                ('rpy_proxy_rad','rpy_covariance_delta_rad2','reference_rpy_rad'),
                ('omega_mean_rad_s','omega_covariance_rad2_s2','reference_omega_body_rad_s'))):
            axis=axes[row,dimension]
            truth=np.rad2deg(source[truth_key][:,dimension])
            truth_line,=axis.plot([],[],color='black',lw=1.3,animated=True,zorder=5)
            dynamic.append(truth_line);layers.append(('line',truth_line,reference_time,truth))
            limits=[truth]
            if row:
                raw=np.rad2deg(sensor['gyro_spike'][:,dimension])
                raw_line,=axis.plot([],[],color=RAW,lw=.6,animated=True,zorder=2)
                dynamic.append(raw_line);layers.append(('line',raw_line,np.asarray(sensor['time']),raw))
                limits.append(raw)
                acquired=np.rad2deg(acquisitions['gyro_spike'][:,dimension])
                for bad,color,marker in ((False,'.25','o'),(True,SPIKE,'x')):
                    mask=np.asarray(acquisitions['spike_mask'])==bad
                    line,=axis.plot([],[],color=color,marker=marker,ls='',ms=4,
                        mfc='white' if not bad else color,mew=1.,animated=True,zorder=9)
                    dynamic.append(line)
                    layers.append(('observation',line,np.asarray(acquisitions['time'])[mask],acquired[mask]))
            for name in CASES:
                case=cases[name];mean=np.rad2deg(case[mean_key][:,dimension])
                half=_halfwidth(case[cov_key])[:,dimension]
                limits.extend((mean-half,mean+half))
                band=PolyCollection([],facecolors=COLORS[name],edgecolors='none',alpha=.17,
                                    animated=True,zorder=1)
                axis.add_collection(band)
                line,=axis.plot([],[],color=COLORS[name],ls=STYLES[name],lw=1.15,animated=True,zorder=6)
                endpoint,=axis.plot([],[],color=COLORS[name],marker='o',ls='',ms=3,animated=True,zorder=8)
                width_line,=axis.plot([],[],color=COLORS[name],lw=1.,animated=True,zorder=7)
                lower_edge,=axis.plot([],[],color=COLORS[name],ls=STYLES[name],lw=.55,
                                     alpha=.85,animated=True,zorder=3)
                upper_edge,=axis.plot([],[],color=COLORS[name],ls=STYLES[name],lw=.55,
                                     alpha=.85,animated=True,zorder=3)
                dynamic.extend((band,line,endpoint,width_line,lower_edge,upper_edge))
                layers.append(('case',(band,line,endpoint,width_line,lower_edge,upper_edge),case,mean,half))
            low=min(float(np.min(x)) for x in limits);high=max(float(np.max(x)) for x in limits)
            margin=.07*(high-low) if high!=low else 1.
            axis.set_ylim(low-margin,high+margin);axis.set_xlim(lower,upper)
            axis.set_title((('roll','pitch','yaw') if row==0 else
                           ('omega x','omega y','omega z'))[dimension],fontsize=10)
            axis.set_ylabel('deg' if row==0 else 'deg/s',fontsize=9)
            if row:axis.set_xlabel('Data time [s]',fontsize=9)
            axis.tick_params(labelsize=8);axis.grid(alpha=.18)
            _recovery(axis,settings)
            cursor=axis.axvline(lower,color='.35',lw=.8,alpha=.55,animated=True)
            dynamic.append(cursor);cursors.append(cursor)
    figure.canvas.draw();background=figure.canvas.copy_from_bbox(figure.bbox)
    schedule=_schedule(acquisitions,lower,upper,fps)
    movie=output/'frequent_spikes_comparison.mp4'
    process=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','veryfast','-crf','22','-maxrate','1500k','-bufsize','3000k',
        '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(movie)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for stamp,phase in schedule:
            figure.canvas.restore_region(background)
            for layer in layers:
                kind=layer[0]
                if kind in ('line','observation'):
                    _,line,time,value=layer
                    selected=time<=stamp
                    if kind=='observation' and phase=='prior':selected &= time<stamp
                    line.set_data(time[selected],value[selected])
                else:
                    _,(band,line,endpoint,width_line,lower_edge,upper_edge),case,mean,half=layer
                    selected=_visible(case,stamp,phase)
                    time=case['time'][selected];value=mean[selected];uncertainty=half[selected]
                    line.set_data(time,value);endpoint.set_data(time[-1:],value[-1:])
                    width_line.set_data([time[-1],time[-1]],
                                       [value[-1]-uncertainty[-1],value[-1]+uncertainty[-1]])
                    lower_edge.set_data(time,value-uncertainty)
                    upper_edge.set_data(time,value+uncertainty)
                    band.set_verts([_band_vertices(time,value,uncertainty)])
            for name in CASES:
                visible=_visible(cases[name],stamp,phase)
                half=_halfwidth(cases[name]['omega_covariance_rad2_s2'][visible][-1:])[0]
                short='NL' if name=='nonlinear' else 'no gyro term'
                width_texts[name].set_text('2sigma omega '+short+' xyz = ('+
                    ', '.join(f'{v:.2f}' for v in half)+') deg/s')
            clock.set_text(f'Data t={stamp:.3f} s  |  {settings["native_rate_hz"]:g} Hz raw / {settings["rate_hz"]:g} Hz update')
            phase_text.set_text({'prior':'Prior: before update','posterior':'After Bayes update',
                                 'prediction':'Initial posterior' if stamp==lower else 'Prediction'}[phase])
            for cursor in cursors:cursor.set_xdata([stamp,stamp])
            for artist in dynamic:figure.draw_artist(artist)
            process.stdin.write(np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[...,:3]).tobytes())
        process.stdin.close();error=process.stderr.read().decode();status=process.wait()
        if status:raise RuntimeError('FFmpeg failed: '+error)
    finally:
        if process.poll() is None:process.kill()
        plt.close(figure)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format',
                                            '-of','json',str(movie)],text=True))
    video=next(value for value in info['streams'] if value['codec_type']=='video')
    if movie.stat().st_size>=50_000_000:
        raise RuntimeError('Movie exceeds the requested 50 MB per-file limit')
    return dict(file=movie.name,bytes=movie.stat().st_size,sha256=_sha(movie),
        width=int(video['width']),height=int(video['height']),fps=fps,frames=len(schedule),
        duration_seconds=float(info['format']['duration']),
        event_hold_seconds_per_side=round(.25*fps)/fps,
        animation='Reveal every native measurement and saved posterior; hold both sides of each acquired observation at its original data timestamp',
        narration=False)


def render(output,source,acquisitions,sensor,cases,metadata=None):
    """Write static figures and a small movie from saved sensor/state arrays."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if set(cases)!=set(CASES):
        raise ValueError('Expected only nonlinear and linear_forced models using the same spiky readings')
    settings=_settings(source,acquisitions,sensor,cases,metadata or {})
    files=[]
    files+=_comparison(output,source,acquisitions,sensor,cases,settings,CASES,'attitude_and_gyro')
    files+=_raw_sensor(output,source,acquisitions,sensor,settings)
    files+=_diagnostics(output,source,acquisitions,cases,settings)
    files+=_gyro_uncertainty(output,source,acquisitions,cases,settings)
    movie=_movie(output,source,acquisitions,sensor,cases,settings)
    files.append(movie['file'])
    record=dict(schema='grape/frequent-spikes-model-comparison-plots/v2',settings=settings,movie=movie,files=files,
        models=list(CASES),model_labels=LABELS,
        observation_series_shared='sensor.gyro_spike and acquisitions.gyro_spike for both models',
        model_difference='linear_forced omits only omega cross (J omega); quaternion kinematics and Bayesian observation model are shared',
        renderer_sha256=_sha(Path(__file__)),
        all_native_samples_drawn=True,matplotlib_path_simplification=False,
        averaging=False,downsampling=False,smoothing=False,covariance_adjustment=False,
        reference='Read-only synthetic truth; not returned to either estimator',
        raw_zoom='Fixed a priori 2–4 s, display only',
        bands='Saved +/-2sigma from same state moments; local RPY display approximation; not a guaranteed 95% region',
        before_after_timestamps_preserved=True)
    (output/'frequent_plotting.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    return record
