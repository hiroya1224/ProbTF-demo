#!/usr/bin/env python3
"""Read-only, data-backed 960x540 animation scenes for the synthetic explainer.

All physical curves, six cubature trajectories, quaternions and covariances
come from archived experiment arrays. Animation changes only which saved
times are visible. It does not fit distributions or modify an inference law.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent/'publication/three_frequency_0p5hz_sigma5'
DEFAULT_DERIVED = Path('/tmp/synthetic-explainer-20261113/derived_data.npz')
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(FONT)).get_name(), 'DejaVu Sans']
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                     'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})

BLUE, RED, ORANGE = '#1769aa', '#ba4141', '#d68113'
NODE_COLORS = ('#1769aa', '#14926e', '#a35ea4', '#d78016', '#cf4760', '#4c7e8a')
PROJECTION = np.array([[.8660254, -.5, 0.], [.25, .4330127, .8660254]])
ALIASES = {'body_rate': 'gyro', 'sampling': 'acquisition', 'covariance': 'rotation_cov',
           'comparison': 'results', 'overview': 'scope'}
KINDS = ('title','scope','motion','frames','gyro','acquisition','noise','torque','euler',
         'linear_compare','cubature','node_flow','rk4','quaternion','moment',
         'rotation_cov','euler_proxy','update','widths','results','limits')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_data(directory):
    directory = Path(directory)
    names = ('input','acquisitions','nonlinear','linear_forced',
             'nonlinear_rpy_plot','linear_forced_rpy_plot',
             'nonlinear_gyro_plot','linear_forced_gyro_plot')
    result = {name: dict(np.load(directory/(name+'.npz'), allow_pickle=False)) for name in names}
    result['summary'] = json.loads((directory/'summary.json').read_text())
    result['gyro_events'] = json.loads((directory/'gyro_events.json').read_text())
    result['audit'] = json.loads((directory/'audit.json').read_text())
    return result


def _view(scene, key, default=None):
    return scene.get('data_view', {}).get(key, scene.get(key, default))


def _axis_index(value):
    if isinstance(value, str):
        return {'x':0,'y':1,'z':2,'roll':0,'pitch':1,'yaw':2}.get(value.lower(), 0)
    return int(value or 0)


def _row_at(times, value):
    return max(0, min(len(times)-1, int(np.searchsorted(times, value, side='right')-1)))


class Scene:
    def __init__(self, scene, data_dir=DEFAULT_DATA):
        self.scene = scene
        self.data_dir = Path(data_dir).resolve()
        self.data = load_data(str(self.data_dir))
        self.input = self.data['input']
        self.axis = _axis_index(_view(scene,'axis',0))
        self.block = int(_view(scene,'block',0))
        self.method = _view(scene,'method','nonlinear')
        self.kind = ALIASES.get(scene.get('kind','scope'),scene.get('kind','scope'))
        self.used = {'input': {'time'}}
        self.derived_path = Path(_view(scene, 'derived_data', os.environ.get('SYNTHETIC_EXPLAINER_DERIVED', DEFAULT_DERIVED)))
        self.derived = dict(np.load(self.derived_path,allow_pickle=False)) if self.derived_path.exists() else {}
        self.figure = plt.figure(figsize=(8,4.5), dpi=120, facecolor='white')
        self.artists = []
        self.updates = []
        self.lower, self.upper = self.input['time']
        self.animation_progress=0.
        self.clock_mapper=None
        self.clock_override=None
        self.clock_prefix='保存データ時刻'
        self.formulas = scene.get('formulas', [])
        if not self.formulas and scene.get('formula'): self.formulas = [scene['formula']]
        if isinstance(self.formulas, str): self.formulas = [self.formulas]
        self.caption_plot_base = .335 + .065*min(3,len(self.formulas))
        self.plot_bottom = self.caption_plot_base + (.03 if self.formulas else 0.)
        if self.kind == 'results' and scene.get('id') in ('p01s01','p06s01'): self.plot_bottom=.335
        if scene.get('id')=='p06s02':self.plot_bottom+=.02
        self.plot_top = .845
        title = scene.get('title','Synthetic rotation')
        self.figure.text(.04,.966,title,fontsize=16,weight='bold',va='top',color='#1a2636')
        self.clock = self.figure.text(.96,.907,'',fontsize=10.5,ha='right',color='#40556b')
        self._dynamic(self.clock)
        self._captions()
        getattr(self, '_build_'+self.kind, self._build_scope)()
        self.figure.text(.04,.012,'合成データ・角速度は deg/s表示（内部はrad/s）／ 0.5 Hz・仮定σ=5 deg/s',
                         fontsize=8.6,color='#52606d')
        for artist in self.artists: artist.set_visible(False)
        self.figure.canvas.draw()
        # Fit formula text to the actual 960-pixel canvas rather than silently
        # clipping equations. Source formulas are short, complete expressions.
        for artist in self.formula_artists:
            box=artist.get_window_extent(self.figure.canvas.get_renderer())
            if box.width>900:
                size=max(11.5,artist.get_fontsize()*900/box.width)
                artist.set_fontsize(size)
        self.figure.canvas.draw()
        for artist in self.formula_artists:
            box=artist.get_window_extent(self.figure.canvas.get_renderer())
            if box.x0<15 or box.x1>945:
                raise ValueError('Complete formula exceeds canvas: '+scene.get('id',''))
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        for artist in self.artists: artist.set_visible(True)

    def _source(self,name,*keys):
        self.used.setdefault(name,set()).update(keys)
        return self.data[name]

    def _captions(self):
        self.formula_artists=[]
        formulas = self.formulas[:3]
        for index, expression in enumerate(formulas):
            expression = str(expression).strip().strip('$').replace('\\tfrac','\\frac')
            y=self.caption_plot_base-.13-.065*index
            size=16
            if self.scene.get('id')=='p06s02':
                y=(.35,.25)[index];size=15 if index==0 else 14
            self.formula_artists.append(self.figure.text(.5,y,'$'+expression+'$',
                             fontsize=size,ha='center',va='center',color='#162d4c'))
        bullets = self.scene.get('bullets',[])
        if isinstance(bullets,str): bullets=[bullets]
        for index, line in enumerate(bullets[:4]):
            top=.18 if self.scene.get('id')=='p06s02' else .187
            self.figure.text(.055,top-.041*index,'・'+str(line),fontsize=11.6,va='center',color='#253341')

    def _dynamic(self,artist):
        self.artists.append(artist)
        return artist

    def _axes(self, rect=None, **kwargs):
        if rect is None: rect=(.09,self.plot_bottom,.85,self.plot_top-self.plot_bottom)
        axis=self.figure.add_axes(rect,**kwargs)
        axis.grid(alpha=.20)
        return axis

    def _curve(self,axis,time,values,color=BLUE,label=None,ls='-',width=1.8,reveal=True):
        time,values=np.asarray(time),np.asarray(values)
        line,=axis.plot(time,values,color=color,lw=width,ls=ls,label=label)
        if reveal:
            self._dynamic(line)
            self.updates.append(lambda stamp,line=line,t=time,y=values:line.set_data(t[t<=stamp],y[t<=stamp]))
        return line

    def _band(self,axis,time,center,halfwidth,color=BLUE,label=None):
        time,center,halfwidth=map(np.asarray,(time,center,halfwidth))
        low,high=center-halfwidth,center+halfwidth
        polygon=PolyCollection([],facecolor=color,alpha=.18,edgecolor='none')
        axis.add_collection(polygon)
        self._dynamic(polygon)
        def update(stamp):
            keep=time<=stamp
            x,a,b=time[keep],low[keep],high[keep]
            if len(x): polygon.set_verts([np.column_stack((np.r_[x,x[::-1]],np.r_[a,b[::-1]]))])
            else: polygon.set_verts([])
        self.updates.append(update)
        self._curve(axis,time,center,color,label)
        axis.update_datalim(np.column_stack((np.r_[time,time],np.r_[low,high])))
        axis.autoscale_view()

    def _cursor(self,axis):
        line=axis.axvline(self.lower,color='#6f7a83',lw=.9,alpha=.7)
        self._dynamic(line)
        self.updates.append(lambda stamp:line.set_xdata([stamp,stamp]))

    def _observations(self,axis,values=None):
        obs=self._source('acquisitions','time','gyro')
        for stamp in obs['time']:axis.axvline(stamp,color='.7',lw=.6,alpha=.6)
        if values is not None:
            dots=axis.scatter(obs['time'],values,c=ORANGE,marker='x',s=38,linewidths=1.4,label='取得gyro',zorder=8)
            self._dynamic(dots)
            self.updates.append(lambda stamp:dots.set_offsets(np.column_stack((obs['time'][obs['time']<=stamp],values[obs['time']<=stamp]))))

    def _finalize_axis(self,axis,ylabel,legend=True):
        axis.set_xlabel('データ時刻 [s]')
        axis.set_ylabel(ylabel)
        axis.set_xlim(self.lower,self.upper)
        if legend:axis.legend(loc='best',fontsize=9,ncol=2)
        self._cursor(axis)

    def _build_title(self):
        self._build_motion()

    def _build_scope(self):
        self._build_euler_proxy()

    def _build_limits(self):
        self._build_widths()

    def _triad(self,axis,quaternions,time,increment=False):
        axis.set_aspect('equal')
        axis.set_xlim(-1.3,1.3);axis.set_ylim(-1.2,1.3)
        axis.set_xticks([]);axis.set_yticks([]);axis.grid(False)
        for a,color in enumerate(('#d44141','#2d9c66','#337cbd')):
            base=PROJECTION[:,a]
            axis.plot([0,1.15*base[0]],[0,1.15*base[1]],color='.78',lw=1)
            line,=axis.plot([0,base[0]],[0,base[1]],color=color,lw=3)
            tag=axis.text(base[0]*1.08,base[1]*1.08,'xyz'[a],color=color,fontsize=13)
            self._dynamic(line);self._dynamic(tag)
            def update(stamp,a=a,line=line,tag=tag):
                row=_row_at(time,stamp)
                vector=PROJECTION@Rotation.from_quat(quaternions[row]).as_matrix()[:,a]
                line.set_data([0,vector[0]],[0,vector[1]])
                tag.set_position(vector*1.12)
            self.updates.append(update)
        axis.set_title('保存した区間内の回転増分' if increment else '実際の姿勢：body軸→世界',fontsize=11)

    def _build_motion(self):
        if _view(self.scene,'components',False):
            return self._build_components()
        d=self._source('input','reference_time','reference_quaternion_xyzw','reference_rpy_rad')
        left=self._axes((.04,self.plot_bottom,.39,self.plot_top-self.plot_bottom))
        self._triad(left,d['reference_quaternion_xyzw'],d['reference_time'])
        right=self._axes((.53,self.plot_bottom,.41,self.plot_top-self.plot_bottom))
        for a,color in enumerate(('#d44141','#2d9c66','#337cbd')):
            self._curve(right,d['reference_time'],np.rad2deg(d['reference_rpy_rad'][:,a]),color,('roll','pitch','yaw')[a])
        self._finalize_axis(right,'姿勢 [deg]')

    def _build_components(self):
        if 'rpy_components_rad' not in self.derived:
            raise FileNotFoundError('Component scene requires derived_data.npz')
        time=self.derived['rpy_component_time']
        components=np.rad2deg(self.derived['rpy_components_rad'][:,self.axis])
        self.used['derived_data.npz']={'rpy_component_time','rpy_components_rad'}
        axis=self._axes()
        for index,color in enumerate((BLUE,'#2d9c66','#a35ea4')):
            self._curve(axis,time,components[:,index],color,f'成分{index+1}',width=1.2)
        self._curve(axis,time,components.sum(axis=1),'black','3成分の和',width=2.)
        self._finalize_axis(axis,('roll','pitch','yaw')[self.axis]+' [deg]')

    def _build_frames(self):
        self._build_motion()

    def _build_gyro(self):
        if self.scene.get('kind')=='body_rate':
            return self._build_linear_compare()
        if _view(self.scene,'quantity')=='alpha':
            d=self._source('input','reference_time','reference_alpha_body_rad_s2')
            axis=self._axes()
            for index,color in enumerate(('#d44141','#2d9c66','#337cbd')):
                self._curve(axis,d['reference_time'],np.rad2deg(d['reference_alpha_body_rad_s2'][:,index]),
                            color,'α'+'xyz'[index])
            return self._finalize_axis(axis,'解析的な角加速度 [deg/s²]')
        d=self._source('input','imu_time','gyro','gyro_true_body_rad_s')
        axis=self._axes()
        self._curve(axis,d['imu_time'],np.rad2deg(d['gyro'][:,self.axis]),'#8a8a8a','生成したgyro',width=.9)
        self._curve(axis,d['imu_time'],np.rad2deg(d['gyro_true_body_rad_s'][:,self.axis]),'black','人工真値',width=1.7)
        self._observations(axis,np.rad2deg(self.data['acquisitions']['gyro'][:,self.axis]))
        self._finalize_axis(axis,'ω'+'xyz'[self.axis]+' [deg/s]')

    def _build_acquisition(self):
        if _view(self.scene,'before_after',False):
            return self._build_update()
        self._build_gyro()

    def _build_noise(self):
        d=self._source('input','imu_time','gyro_realized_noise_rad_s','gyro_bias_rad_s','generator_gyro_covariance_rad2_s2')
        axis=self._axes((.09,self.plot_bottom,.58,self.plot_top-self.plot_bottom))
        self._curve(axis,d['imu_time'],np.rad2deg(d['gyro_realized_noise_rad_s'][:,self.axis]),'#6c7d91','実現した白色ノイズ',width=.8)
        axis.axhline(np.rad2deg(d['gyro_bias_rad_s'][self.axis]),color=ORANGE,lw=1.6,label='固定bias')
        self._finalize_axis(axis,'生成誤差 [deg/s]')
        self.figure.text(.73,.71,'生成ノイズ σ',fontsize=12)
        self.figure.text(.75,.65,'0.2 deg/s',fontsize=19,color='#446b91')
        self.figure.text(.73,.52,'候補の仮定 σ',fontsize=12)
        self.figure.text(.75,.46,'5 deg/s',fontsize=19,color=BLUE)

    def _build_torque(self):
        d=self._source('input','torque_time','torque_body_nm','reference_time','reference_alpha_body_rad_s2','reference_gyroscopic_torque_body_nm','inertia_body_kg_m2')
        axis=self._axes()
        inertial=d['reference_alpha_body_rad_s2']@d['inertia_body_kg_m2'].T
        self._curve(axis,d['torque_time'],d['torque_body_nm'][:,self.axis],'black','既知入力 τ')
        self._curve(axis,d['reference_time'],inertial[:,self.axis],BLUE,'J α')
        self._curve(axis,d['reference_time'],d['reference_gyroscopic_torque_body_nm'][:,self.axis],'#984ea3','ω × Jω')
        self._finalize_axis(axis,'xyz'[self.axis]+'軸トルク [Nm]')

    def _build_euler(self):
        self._build_torque()

    def _gyro_comparison(self,axis,dimension,widths=False,errors=False):
        base=self.data['nonlinear_gyro_plot']
        if not widths:
            truth=base['reference_omega_rad_s'][:,dimension]
            self._curve(axis,base['time'],np.zeros(len(truth)) if errors else np.rad2deg(truth),'black','人工真値',reveal=False)
        for method,color,name in (('linear_forced',RED,'ジャイロ項なし'),('nonlinear',BLUE,'非線形Euler')):
            d=self._source(method+'_gyro_plot','time','mean_rad_s','covariance_rad2_s2','reference_omega_rad_s')
            center=np.rad2deg(d['mean_rad_s'][:,dimension])
            half=2*np.rad2deg(np.sqrt(d['covariance_rad2_s2'][:,dimension,dimension]))
            if errors:center-=np.rad2deg(d['reference_omega_rad_s'][:,dimension])
            if widths:self._curve(axis,d['time'],half,color,name,ls='--' if method=='linear_forced' else '-')
            else:self._band(axis,d['time'],center,half,color,name)
        self._observations(axis)

    def _build_linear_compare(self):
        axis=self._axes()
        self._gyro_comparison(axis,self.axis,errors=bool(_view(self.scene,'error',False)))
        self._finalize_axis(axis,'ω'+'xyz'[self.axis]+' [deg/s]')

    def _build_update(self):
        focus=float(_view(self.scene,'focus_time',2.))
        self.lower,self.upper=max(0.,focus-.35),min(5.,focus+.35)
        if self.scene.get('id')=='p05s04' or _view(self.scene,'quantity')=='rpy':
            self._build_euler_proxy()
        else:self._build_linear_compare()

    def _build_widths(self):
        axis=self._axes()
        self._gyro_comparison(axis,self.axis,widths=True)
        self._finalize_axis(axis,'±2σの半幅 [deg/s]')
        events=self.data['gyro_events']['nonlinear']['events']
        self.used['gyro_events.json']={'nonlinear.events'}
        before=[e for e in events if e['before_two_sigma_deg_s'] is not None]
        axis.scatter([e['time_seconds'] for e in before],[e['before_two_sigma_deg_s'][self.axis] for e in before],
                     facecolor='white',edgecolor=BLUE,s=33,zorder=6)
        axis.scatter([e['time_seconds'] for e in events],[e['after_two_sigma_deg_s'][self.axis] for e in events],
                     c=BLUE,s=23,zorder=7)

    def _build_cubature(self):
        d=self._source(self.method,f'block_{self.block}_time',f'block_{self.block}_omega',f'block_{self.block}_weights')
        nodes=np.rad2deg(d[f'block_{self.block}_omega'][0])
        weights=d[f'block_{self.block}_weights']
        self.lower=self.upper=float(d[f'block_{self.block}_time'][0])
        self.clock_override=f'固定の観測時刻 t={self.lower:.3f} s：6点の構成'
        center=weights@nodes
        projected=(nodes-center)@PROJECTION.T
        axis=self._axes((.12,self.plot_bottom,.56,self.plot_top-self.plot_bottom))
        axis.set_aspect('equal');axis.set_xlim(-13,13);axis.set_ylim(-13,13)
        axis.axhline(0,color='.8',lw=.8);axis.axvline(0,color='.8',lw=.8)
        axis.set_xlabel('body角速度の固定2D投影 [deg/s]')
        axis.set_ylabel('観測平均からの差')
        for index,point in enumerate(projected):
            dot=axis.scatter(*point,s=75,color=NODE_COLORS[index],zorder=4)
            tag=axis.text(point[0]+.55,point[1]+.55,('+x','+y','+z','−x','−y','−z')[index],fontsize=11)
            self._dynamic(dot);self._dynamic(tag)
            def update(stamp,index=index,dot=dot,tag=tag):
                shown=self.animation_progress>index/7
                dot.set_visible(shown);tag.set_visible(shown)
            self.updates.append(update)
        self.figure.text(.74,.70,'固定6点',fontsize=19,color=BLUE)
        self.figure.text(.73,.59,'各点の重み',fontsize=12)
        self.figure.text(.76,.51,'1 / 6',fontsize=23)
        self.figure.text(.70,.40,'ランダム抽出なし',fontsize=11.5)

    def _build_node_flow(self):
        d=self._source(self.method,f'block_{self.block}_time',f'block_{self.block}_omega',f'block_{self.block}_delta_wxyz',f'block_{self.block}_weights')
        time=d[f'block_{self.block}_time'];omega=d[f'block_{self.block}_omega']
        self.lower,self.upper=float(time[0]),float(time[-1])
        left=self._axes((.09,self.plot_bottom,.53,self.plot_top-self.plot_bottom))
        for node,color in enumerate(NODE_COLORS):
            self._curve(left,time,np.rad2deg(omega[:,node,self.axis]),color,f'点{node+1}',width=1.2)
        self._finalize_axis(left,'ω'+'xyz'[self.axis]+' [deg/s]')
        right=self._axes((.72,self.plot_bottom,.23,self.plot_top-self.plot_bottom))
        right.set_aspect('equal');right.set_xlim(-1.25,1.25);right.set_ylim(-1.25,1.25)
        right.set_xticks([]);right.set_yticks([])
        delta=d[f'block_{self.block}_delta_wxyz']
        for node,color in enumerate(NODE_COLORS):
            line,=right.plot([0,1],[0,0],color=color,lw=1.6)
            self._dynamic(line)
            def update(stamp,node=node,line=line):
                row=_row_at(time,stamp)
                q=delta[row,node,[1,2,3,0]]
                value=PROJECTION@Rotation.from_quat(q).as_matrix()[:,self.axis]
                line.set_data([0,value[0]],[0,value[1]])
            self.updates.append(update)
        right.set_title('局所増分の'+'xyz'[self.axis]+'軸（6点）',fontsize=9.5)

    def _build_quaternion(self):
        d=self._source(self.method,f'block_{self.block}_time',f'block_{self.block}_delta_wxyz')
        time=d[f'block_{self.block}_time'];delta=d[f'block_{self.block}_delta_wxyz']
        self.lower,self.upper=float(time[0]),float(time[-1])
        node=int(_view(self.scene,'node',0))
        left=self._axes((.05,self.plot_bottom,.34,self.plot_top-self.plot_bottom))
        self._triad(left,delta[:,node,[1,2,3,0]],time,increment=True)
        right=self._axes((.51,self.plot_bottom,.43,self.plot_top-self.plot_bottom))
        for a,color in enumerate(('#18293c','#d44141','#2d9c66','#337cbd')):
            self._curve(right,time,delta[:,node,a],color,('w','x','y','z')[a])
        self._finalize_axis(right,'増分quaternion成分')

    def _build_rk4(self):
        required=('rk4_k_omega','rk4_stage_omega','rk4_stage_time')
        if not all(key in self.derived for key in required):
            raise FileNotFoundError('Data-backed RK4 scene requires derived_data.npz: '+str(self.derived_path))
        values=np.rad2deg(self.derived['rk4_k_omega'][:,int(_view(self.scene,'node',0)),self.axis])
        stage_time=np.asarray(self.derived['rk4_stage_time'],float)
        self.lower,self.upper=float(stage_time[0]),float(stage_time[-1])
        self.clock_prefix='第1ステップの実際の評価時刻'
        self.clock_mapper=lambda progress:float(stage_time[min(3,int(progress*4))])
        axis=self._axes((.12,self.plot_bottom,.54,self.plot_top-self.plot_bottom))
        x=np.arange(4)
        bars=axis.bar(x,values,color=[BLUE,'#319c80','#8664ad',ORANGE],width=.55)
        axis.set_xticks(x);axis.set_xticklabels(('k1','k2','k3','k4'))
        axis.set_ylabel('実際の角加速度 [deg/s²]')
        axis.set_title(f'候補{int(_view(self.scene,"node",0))+1}・ω'+ 'xyz'[self.axis]+' の傾き（4評価）',fontsize=11)
        for index,bar in enumerate(bars):
            self._dynamic(bar)
            self.updates.append(lambda stamp,index=index,bar=bar:bar.set_visible(index<=min(3,int(self.animation_progress*4))))
        dt=float(self.derived.get('rk4_dt',.002))
        self.figure.text(.73,.70,f'h = {dt:g} s',fontsize=17,color=BLUE)
        self.figure.text(.73,.59,'同じ6候補を',fontsize=12)
        self.figure.text(.73,.53,'同時に進める',fontsize=12)
        self.figure.text(.70,.41,'元の入力・元の計算式',fontsize=11)
        self.used['derived_data.npz']=set(required+('rk4_dt',))

    def _build_moment(self):
        space=_view(self.scene,'moment_space','global')
        if space in ('increment','local'):
            d=self._source(self.method,f'block_{self.block}_time',f'block_{self.block}_delta_wxyz',f'block_{self.block}_weights')
            time=d[f'block_{self.block}_time'];delta=d[f'block_{self.block}_delta_wxyz'];weights=d[f'block_{self.block}_weights']
            matrix=np.einsum('s,tsi,tsj->tij',weights,delta,delta)
            self.lower,self.upper=float(time[0]),float(time[-1])
            title='区間内増分の二次モーメント'
        else:
            d=self._source(self.method,'time','quaternion_second_moment')
            time=d['time'];matrix=d['quaternion_second_moment'];title='伝播した姿勢の二次モーメント'
        self._heatmap(time,matrix,title,('w','x','y','z'),trace_bounds=(0.,1.05))

    def _heatmap(self,time,matrix,title,labels,within_between=False,trace_bounds=None):
        left=self._axes((.09,self.plot_bottom,.40,self.plot_top-self.plot_bottom))
        scale=float(np.max(np.abs(matrix))) or 1.
        image=left.imshow(matrix[0],cmap='coolwarm',vmin=-scale,vmax=scale,interpolation='nearest')
        left.grid(False);left.set_title(title,fontsize=10.5)
        left.set_xticks(np.arange(len(labels)));left.set_xticklabels(labels,fontsize=8)
        left.set_yticks(np.arange(len(labels)));left.set_yticklabels(labels,fontsize=8)
        self.figure.colorbar(image,ax=left,fraction=.035,pad=.035)
        self._dynamic(image)
        self.updates.append(lambda stamp:image.set_data(matrix[_row_at(time,stamp)]))
        right=self._axes((.64,self.plot_bottom,.30,self.plot_top-self.plot_bottom))
        if within_between:
            keys=('rotation_cov_time','rotation_cov_within_trace','rotation_cov_between_trace','rotation_cov_total_trace')
            if not all(key in self.derived for key in keys):
                raise FileNotFoundError('Within/between scene requires exact derived covariance traces')
            trace_time=self.derived['rotation_cov_time']
            for key,color,name in ((keys[1],BLUE,'within'),(keys[2],ORANGE,'between'),(keys[3],'black','合計')):
                self._curve(right,trace_time,self.derived[key],color,name,width=1.6)
            self.used.setdefault('derived_data.npz',set()).update(keys)
            right.set_title('区間内の分解（合計は連続）',fontsize=9.5)
        else:self._curve(right,time,np.trace(matrix,axis1=1,axis2=2),BLUE,'trace')
        self._finalize_axis(right,'行列のtrace')
        if trace_bounds is not None:right.set_ylim(*trace_bounds)

    def _build_rotation_cov(self):
        if self.scene.get('id')=='p04s06' or _view(self.scene,'quantity')=='mean_rotation':
            d=self._source(self.method,'time','rotation_mean')
            return self._heatmap(d['time'],d['rotation_mean'],'回転行列の平均 E[R]',('x','y','z'))
        d=self._source(self.method,'time','rotation_covariance_colmajor')
        self._heatmap(d['time'],d['rotation_covariance_colmajor'],'回転行列の全9×9共分散',
                      ('11','21','31','12','22','32','13','23','33'),
                      within_between=bool(_view(self.scene,'within_between',self.scene.get('id')=='p04s08')))

    def _rpy_comparison(self,axis,dimension):
        base=self.data['nonlinear_rpy_plot']
        self._curve(axis,base['time'],np.rad2deg(base['reference_rpy_rad'][:,dimension]),'black','人工真値',reveal=False)
        for method,color,name in (('linear_forced',RED,'ジャイロ項なし'),('nonlinear',BLUE,'非線形Euler')):
            d=self._source(method+'_rpy_plot','time','rpy_proxy_rad','rpy_covariance_delta_rad2','reference_rpy_rad')
            self._band(axis,d['time'],np.rad2deg(d['rpy_proxy_rad'][:,dimension]),
                       2*np.rad2deg(np.sqrt(d['rpy_covariance_delta_rad2'][:,dimension,dimension])),color,name)
        self._observations(axis)

    def _build_euler_proxy(self):
        axis=self._axes();self._rpy_comparison(axis,self.axis)
        self._finalize_axis(axis,('roll','pitch','yaw')[self.axis]+' [deg]')

    def _build_results(self):
        if self.scene.get('id')=='p06s05':return self._build_ideal_results()
        if self.scene.get('kind')=='comparison':return self._build_error_results()
        bottom=self.plot_bottom
        height=(self.plot_top-bottom-.065)/2
        for dimension in range(3):
            x=.07+dimension*.315
            top=self._axes((x,bottom+height+.065,.265,height))
            low=self._axes((x,bottom,.265,height))
            self._rpy_comparison(top,dimension);self._gyro_comparison(low,dimension)
            self._observations(low,np.rad2deg(self.data['acquisitions']['gyro'][:,dimension]))
            top.set_title(('roll','pitch','yaw')[dimension],fontsize=11)
            low.set_title('ω'+'xyz'[dimension],fontsize=11)
            top.set_ylabel('deg',fontsize=9);low.set_ylabel('deg/s',fontsize=9)
            top.set_xlim(0,5);low.set_xlim(0,5)
            top.tick_params(labelsize=8);low.tick_params(labelsize=8)
            low.set_xlabel('時刻 [s]',fontsize=9)
            self._cursor(top);self._cursor(low)
        # Colors are constant across all explanation scenes.
        self.figure.text(.07,.903,'黒：真値　青：非線形Euler　赤：ジャイロ項なし',fontsize=9.5)

    def _build_error_results(self):
        if self.scene.get('id')=='p06s04':
            focus=float(_view(self.scene,'focus_time',4.))
            self.lower,self.upper=max(0.,focus-.6),min(5.,focus+.6)
            return self._build_euler_proxy()
        axis=self._axes()
        for method,color,name in (('linear_forced',RED,'ジャイロ項なし'),('nonlinear',BLUE,'非線形Euler')):
            d=self._source(method+'_rpy_plot','time','error_geodesic_rad')
            self._curve(axis,d['time'],np.rad2deg(d['error_geodesic_rad']),color,name)
        self._finalize_axis(axis,'姿勢中心のSO(3)誤差 [deg]')

    def _build_ideal_results(self):
        self.clock_override='保存した別検証の集計：全5秒間'
        self.used['audit.json']={'separate_zero_noise_no_bias_unit_experiment'}
        audit=self.data['audit']['separate_zero_noise_no_bias_unit_experiment']
        rates=audit['rates'];selected=rates.get('0.5',next(iter(rates.values())))['methods']
        values=[selected[method]['geodesic_max_deg'] for method in ('nonlinear','linear_forced')]
        axis=self._axes((.16,self.plot_bottom,.68,self.plot_top-self.plot_bottom))
        axis.bar(('非線形Euler','ジャイロ項なし'),values,color=(BLUE,RED),width=.5)
        axis.set_yscale('log');axis.set_ylabel('理想条件の最大姿勢誤差 [deg]（対数軸）')
        for i,value in enumerate(values):axis.text(i,value*1.25,f'{value:.6g}°',ha='center',fontsize=12)
        axis.set_ylim(min(values)*.25,max(values)*5)
        axis.set_title('保存した別検証：noise=0・bias=0',fontsize=11)

    def frame(self,progress):
        progress=min(1.,max(0.,float(progress)))
        self.animation_progress=progress
        stamp=(self.clock_mapper(progress) if self.clock_mapper is not None else
               self.lower+(self.upper-self.lower)*min(1.,progress*1.08))
        self.figure.canvas.restore_region(self.background)
        self.clock.set_text(self.clock_override or f'{self.clock_prefix} t={stamp:.3f} s')
        for update in self.updates:update(stamp)
        for artist in self.artists:
            if artist.get_visible():self.figure.draw_artist(artist)
        return np.asarray(self.figure.canvas.buffer_rgba())[...,:3].copy()

    def provenance(self):
        sources=[]
        for name,keys in sorted(self.used.items()):
            path=self.derived_path if name=='derived_data.npz' else self.data_dir/(name if name.endswith('.json') else name+'.npz')
            sources.append(dict(path=str(path),sha256=_sha(path),keys=sorted(keys)))
        return dict(schema='grape/synthetic-explainer-render-scene/v1',scene_id=self.scene.get('id'),
                    kind=self.kind,source_data=str(self.data_dir),sources=sources,
                    data_time_interval=[float(self.lower),float(self.upper)],
                    animation='reveal saved samples in time; heatmaps/axes use latest saved sample at or before displayed time; cubature construction keeps its actual observation clock fixed; duplicated acquisition limits retained',
                    physical_units='deg and deg/s for display; archived radian arrays unchanged',
                    modifies_inference=False,monte_carlo=False,
                    dimension=[960,540],renderer_sha256=_sha(__file__))

    def close(self):
        plt.close(self.figure)


def preview_frame(scene,progress,destination,data_dir=DEFAULT_DATA):
    renderer=Scene(scene,data_dir)
    try:
        from PIL import Image
        Image.fromarray(renderer.frame(progress)).save(destination)
    finally:renderer.close()


def render_scene(scene,duration,destination,data_dir=DEFAULT_DATA,fps=15):
    """Write one silent H.264 scene and its actual-data provenance sidecar."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    duration=float(duration);fps=float(fps)
    if duration<=0 or fps<=0:raise ValueError('Positive finite scene duration/fps required')
    renderer=Scene(scene,data_dir)
    requested_frames=duration*fps
    nearest=round(requested_frames)
    frames=max(1,int(nearest if abs(requested_frames-nearest)<1e-7 else math.ceil(requested_frames)))
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
             '-pix_fmt','rgb24','-s','960x540','-r',str(fps),'-i','-','-an',
             '-c:v','libx264','-preset','veryfast','-crf','24','-maxrate','1400k','-bufsize','2800k',
             '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(destination)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for index in range(frames):
            process.stdin.write(renderer.frame(index/max(1,frames-1)).tobytes())
        process.stdin.close()
        error=process.stderr.read().decode()
        status=process.wait()
        if status:raise RuntimeError('ffmpeg rendering failed: '+error)
        provenance=renderer.provenance()
        provenance.update(requested_duration_seconds=duration,encoded_duration_seconds=frames/fps,
                          frames=frames,fps=fps,output_bytes=destination.stat().st_size,
                          output_sha256=_sha(destination))
        destination.with_suffix('.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
        return provenance
    finally:
        renderer.close()
        if process.poll() is None:process.kill();process.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--story',type=Path,required=True)
    parser.add_argument('--scene-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=12.)
    parser.add_argument('--fps',type=float,default=15.)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    story=json.loads(args.story.read_text())
    scenes=[scene for part in story['parts'] for scene in part['scenes']]
    scene=next(scene for scene in scenes if scene['id']==args.scene_id)
    if args.preview:preview_frame(scene,.65,args.output,args.data_dir)
    else:render_scene(scene,args.duration,args.output,args.data_dir,args.fps)
