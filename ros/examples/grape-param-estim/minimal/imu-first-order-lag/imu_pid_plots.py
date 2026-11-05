"""PID node-rate figures for the independent IMU model experiment.

Only supplied gain axes and statistics are drawn.  No estimator, controller
search, sampling, interpolation, success input or covariance change occurs.
Midpoint tiles/boxes are visualization glyphs for evaluated node values; their
interiors are not certified stable.  Pending nodes stay blank.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/imu-pid-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np


HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE.parent/'first-order-lag/publication/pid_gain_voxels/figure_settings.json'
GROUPS = ('xy', 'z', 'roll_pitch', 'yaw')
LABELS = {'xy':'XY', 'z':'Z', 'roll_pitch':'Roll / Pitch', 'yaw':'Yaw'}


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def historical_settings():
    """Read visual settings only; never the retired publication's numbers."""
    return json.loads(SETTINGS_PATH.read_text())


def rate_colormap(settings=None):
    settings = historical_settings() if settings is None else settings
    positions = np.linspace(0., 1., 256)
    rgb = plt.get_cmap(settings['rate_colormap'])(positions)[:, :3]
    rgb = float(settings['white_mix']) + (1.-float(settings['white_mix']))*rgb
    anchors = [(float(q), colors.to_hex(value)) for q, value in zip(positions, rgb)]
    return colors.LinearSegmentedColormap.from_list('imu_pid_rate', anchors, N=4096)


def midpoint_edges(axis):
    values = np.asarray(axis, float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0.)):
        raise ValueError('A displayed gain axis must be finite and strictly increasing')
    return np.r_[values[0], .5*(values[:-1]+values[1:]), values[-1]]


def _faces(values, evaluated, axes):
    """Historical exposed midpoint-box faces, omitting pending glyphs only."""
    edges = [midpoint_edges(a) for a in axes]
    faces, rates = [], []
    for dimension in range(3):
        free = [j for j in range(3) if j != dimension]
        for side in (0, 1):
            visible = evaluated.copy()
            target, source = [slice(None)]*3, [slice(None)]*3
            target[dimension] = slice(1, None) if side == 0 else slice(None, -1)
            source[dimension] = slice(None, -1) if side == 0 else slice(1, None)
            # Omit an internal face only when both node glyphs have the
            # exactly same supplied rate.  No statistical class is merged.
            visible[tuple(target)] &= (~evaluated[tuple(source)]
                | (values[tuple(target)] != values[tuple(source)]))
            locations = np.argwhere(visible)
            if not len(locations):
                continue
            vertices = np.empty((len(locations), 4, 3))
            vertices[:, :, dimension] = edges[dimension][locations[:, dimension]+side, None]
            for corner, (a, b) in enumerate(((0,0),(1,0),(1,1),(0,1))):
                vertices[:, corner, free[0]] = edges[free[0]][locations[:, free[0]]+a]
                vertices[:, corner, free[1]] = edges[free[1]][locations[:, free[1]]+b]
            faces.append(vertices)
            rates.append(values[tuple(locations.T)])
    return ((np.concatenate(faces), np.concatenate(rates)) if faces else
            (np.empty((0,4,3)), np.empty(0)))


def _plane_tiles(values, evaluated, p_axis, d_axis):
    p_edges, d_edges = midpoint_edges(p_axis), midpoint_edges(d_axis)
    locations = np.argwhere(evaluated)
    vertices = np.empty((len(locations), 4, 2))
    for corner, (a,b) in enumerate(((0,0),(1,0),(1,1),(0,1))):
        vertices[:, corner, 0] = p_edges[locations[:,0]+a]
        vertices[:, corner, 1] = d_edges[locations[:,1]+b]
    return vertices, values[tuple(locations.T)]


def _limits(axis, padding, references=()):
    lower,upper = float(axis[0]),float(axis[-1])
    if len(references):
        lower,upper = min(lower,min(references)),max(upper,max(references))
    span = upper-lower
    return float(lower-padding*span),float(upper+padding*span)


def _reference_points(reference, info, mode):
    """Caller-supplied post-map references; never an input to map values."""
    points = [dict(kind='failure',label='Failure',gain=np.asarray(reference,float).tolist())]
    for entry in info.get('reference_points', ()):
        point = dict(entry)
        gain = np.asarray(point.get('gain'),float)
        if gain.shape != (3,) or not np.isfinite(gain).all():
            raise ValueError('A reference point needs its actual finite P/I/D triple')
        point['gain'] = gain.tolist()
        point.setdefault('kind','success')
        point.setdefault('label','Success ref.' if point['kind']=='success' else 'Failure')
        if point['kind']=='failure' and np.array_equal(gain,reference):
            points[0] = point
        else:
            points.append(point)
    for point in points:
        point['shown'] = bool(mode=='3d' or point['gain'][1]==1.)
        point['projected'] = False
        point['omitted_reason'] = None if point['shown'] else 'actual I is not 1'
    return points


def _reference_text(point):
    text = str(point['label'])+'\n('+', '.join(f'{v:.5g}' for v in point['gain'])+')'
    rest = point.get('rest_pid')
    if rest is not None:
        text += '\nrest PID'
        for context in ('success','failure'):
            row = rest[context]
            value = row.get('stable_fraction',row.get('rate')) if isinstance(row,dict) else row
            rate = float(value)
            if not np.isfinite(rate) or not 0. <= rate <= 1.:
                raise ValueError('Reference rates must be supplied finite fractions')
            text += f'\n{context}: rate = {100.*rate:.1f}%'
    return text


def build_figure(case, group, p_axis, i_axis, d_axis, rate, evaluated,
                 base_gains, mode='3d', metadata=None):
    """Build a figure without changing any supplied numerical value."""
    if group not in GROUPS or mode not in ('3d','plane'):
        raise ValueError('Expected a known PID group and mode 3d or plane')
    axes = tuple(np.asarray(a, float) for a in (p_axis,i_axis,d_axis))
    for axis in axes:
        if (axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0.)):
            raise ValueError('Gain axes must be finite increasing vectors')
    values, done = np.asarray(rate, float), np.asarray(evaluated, bool)
    shape = tuple(len(a) for a in axes)
    gains = np.asarray(base_gains, float)
    if values.shape != shape or done.shape != shape or gains.shape != (4,3):
        raise ValueError('P/I/D array and four-group gain shapes disagree')
    if not np.isfinite(gains).all():
        raise ValueError('Recorded controller gains must be finite')
    if np.any(~np.isfinite(values[done])) or np.any((values[done] < 0.) | (values[done] > 1.)):
        raise ValueError('Evaluated rates must be finite ALL-N fractions in [0,1]')
    if mode == 'plane' and not np.array_equal(axes[1], [1.]):
        raise ValueError('The requested plane requires the exact I axis [1.0]')
    if mode == '3d' and any(len(a) < 2 for a in axes):
        raise ValueError('3D plotting requires at least two nodes on every axis')
    settings = historical_settings()
    cmap = rate_colormap(settings)
    norm = colors.Normalize(0.,1.)
    reference = gains[GROUPS.index(group)]
    info = dict(metadata or {})
    method = info.get('assessment_method','supplied_numerical_rates')
    analytic = bool(info.get('analytic_certificate',False) or 'analytic' in str(method).lower())
    criterion = str(info.get('criterion_label',
        'Q = Pr(full-state spectral radius < 1)' if analytic else
        'Stable fraction (rest PID: recorded failure)'))
    points = _reference_points(reference,info,mode)
    shown = [p for p in points if p['shown']]
    cut = bool(info.get('cut_at_recorded_i', True)) and mode == '3d'
    start = int(np.searchsorted(axes[1], reference[1])) if cut else 0
    if cut and (start >= len(axes[1]) or len(axes[1])-start < 2):
        raise ValueError('The recorded-I cut leaves fewer than two displayed I nodes')
    displayed_axes = (axes[0], axes[1][start:], axes[2])
    selected, mask = values[:,start:,:], done[:,start:,:]
    fig = plt.figure(figsize=(8.6,7.2))
    axis = (fig.add_subplot(111, projection='3d', computed_zorder=False)
            if mode == '3d' else fig.add_subplot(111))
    if mode == '3d':
        vertices, colors_at_nodes = _faces(selected, mask, displayed_axes)
        rgba = cmap(colors_at_nodes)
        rgba[:,3] = float(settings['voxel_opacity'])
        collection = Poly3DCollection(vertices, facecolors=rgba, edgecolors='none',
            linewidth=0., zsort='average', rasterized=True, zorder=1)
        collection.set_gid('evaluated_node_glyphs')
        axis.add_collection3d(collection)
        for dimension,(a,setter) in enumerate(zip(displayed_axes, (axis.set_xlim,axis.set_ylim,axis.set_zlim))):
            setter(*_limits(a,float(settings['frame_padding']),[p['gain'][dimension] for p in shown]))
        axis.set(xlabel='P gain',ylabel='I gain',zlabel='D gain')
        axis.set_box_aspect((1.,1.,1.))
        axis.set_proj_type('ortho')
        camera = settings['camera_oblique']
        axis.view_init(elev=camera['elevation'],azim=camera['azimuth'])
    else:
        vertices, colors_at_nodes = _plane_tiles(values[:,0,:],done[:,0,:],axes[0],axes[2])
        collection = PolyCollection(vertices,array=colors_at_nodes,cmap=cmap,norm=norm,
                                    edgecolors='none',linewidth=0.,antialiased=False)
        collection.set_gid('evaluated_node_glyphs')
        axis.add_collection(collection)
        axis.set(xlabel='P gain',ylabel='D gain',
            xlim=_limits(axes[0],.1,[p['gain'][0] for p in shown]),
            ylim=_limits(axes[2],.1,[p['gain'][2] for p in shown]))
        axis.text(.98,.98,'I = 1',transform=axis.transAxes,ha='right',va='top')
    on_plane = bool(reference[1] == 1.)
    reference_shown = mode == '3d' or on_plane
    for point in shown:
        gain = point['gain']
        success = point['kind']=='success'
        coincident = not success and any(p['kind']=='success' and p['gain']==gain for p in shown)
        style = (dict(marker='D',s=68,color=settings['success_color'],edgecolor='white') if success else
                 dict(marker='o',s=170,facecolors='none',edgecolors=settings['recorded_color']) if coincident else
                 dict(marker='X',s=95,color=settings['recorded_color'],edgecolor='white'))
        color = settings['success_color' if success else 'recorded_color']
        arguments = tuple(gain) if mode=='3d' else (gain[0],gain[2])
        marker = axis.scatter(*arguments,linewidth=1.1,label=point['label'],zorder=8 if success else 7,
            **style,**({'depthshade':False} if mode=='3d' else {}))
        marker.set_gid('success_reference' if success else 'recorded_failure_reference')
        if info.get('reference_points'):
            xy = proj3d.proj_transform(*gain,axis.get_proj())[:2] if mode=='3d' else arguments
            text = axis.annotate(_reference_text(point),xy=xy,
                xytext=(.68,.78) if success else (.25,.2),textcoords='axes fraction',
                ha='center',va='center',multialignment='center',fontsize=9,color=color,
                arrowprops=dict(arrowstyle='->',color=color,lw=1.2),
                bbox=dict(facecolor='white',edgecolor=color,alpha=.97,pad=3),zorder=10)
            text.set_gid('success_reference_label' if success else 'failure_reference_label')
    if shown:
        axis.legend(loc='best',fontsize=9)
    bar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axis,
                       orientation='horizontal',fraction=.045,pad=.13)
    bar.set_ticks([0.,.25,.5,.75,1.])
    bar.set_ticklabels(['0%','25%','50%','75%','100%'])
    bar.set_label(criterion)
    title = f'{case} — {LABELS[group]} PID' + (' / I = 1' if mode == 'plane' else ' / P, I, D')
    assessed = 'Certified' if analytic else 'Evaluated'
    progress = f'{assessed} nodes {int(np.count_nonzero(mask))}/{mask.size}'
    if cut:
        progress = (f'{assessed} displayed nodes {int(np.count_nonzero(mask))}/{mask.size}; '
                    f'full grid {int(np.count_nonzero(done))}/{done.size}')
    if analytic:
        progress = 'Analytic certificate — '+progress
    axis.set_title(title + '\n' + progress,fontsize=12)
    note = 'Node values shown as midpoint glyphs; pending nodes blank; no cell-interior guarantee.'
    if analytic:
        note = ('Analytical full-state stability assessment; no numerical poles were computed.\n'
                'This is not a flight-failure probability. Pending nodes remain blank.')
    if cut:
        note += '\n3D display cut at the recorded I; all computed data retained.'
    elif mode == 'plane' and not on_plane:
        note += f'\nRecorded I={reference[1]:.5g} is off this plane; its marker is not projected.'
    fig.text(.055,.018,note,fontsize=9)
    fig.tight_layout(rect=(.01,.08,.99,.98))
    plotting = dict(schema='grape/imu-pid-node-rate-figures/v2',case=case,group=group,mode=mode,
        settings_source=str(SETTINGS_PATH),settings_sha256=hashlib.sha256(SETTINGS_PATH.read_bytes()).hexdigest(),
        color_quantity=criterion if analytic else 'finite valid rho<1 count / ALL fixed Gaussian plant nodes',color_range=[0.,1.],
        criterion_label=criterion,assessment_method=method,analytic_certificate=analytic,
        numerical_poles_computed=False if analytic else None,
        colormap=settings['rate_colormap'],white_mix=settings['white_mix'],
        voxel_opacity=settings['voxel_opacity'],threshold_colors=False,
        full_axis_order=['P','I','D'],full_axes=[a.tolist() for a in axes],
        displayed_axes=[a.tolist() for a in displayed_axes],
        exact_fixed_i=1. if mode == 'plane' else None,
        full_evaluated_count=int(np.count_nonzero(done)),full_grid_node_count=int(done.size),
        displayed_evaluated_count=int(np.count_nonzero(mask)),displayed_grid_node_count=int(mask.size),
        full_certified_count=int(np.count_nonzero(done)) if analytic else None,
        displayed_certified_count=int(np.count_nonzero(mask)) if analytic else None,
        evaluated_count_definition=('analytically certified grid nodes, not numerical eigensystems' if analytic
                                    else 'grid nodes with supplied numerical rate values'),
        pending_nodes_shown_as_zero=False,cut_at_recorded_i=cut,
        recorded_reference=reference.tolist(),recorded_reference_shown=reference_shown,
        reference_points=points,
        view_limits={'P':list(axis.get_xlim()),'D':list(axis.get_zlim() if mode=='3d' else axis.get_ylim()),
                     'I':list(axis.get_ylim()) if mode=='3d' else [1.,1.]},
        reference_projected=False,full_fixed_controller_gains=gains.tolist(),
        source_array_sha256={'rate':_hash(values),'evaluated':_hash(done),
                            **{name:_hash(a) for name,a in zip(('p','i','d'),axes)}},
        supplied_metadata=info,law_or_statistics_changed=False,
        interpretation=('analytic full-state strict-stability probability, not flight-failure probability' if analytic
                        else 'local model-node stability fraction, not flight-success probability'))
    return fig, axis, plotting


def render_grid(destination, case, group, p_axis, i_axis, d_axis, rate, evaluated,
                base_gains, mode='3d', metadata=None):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    fig,axis,info = build_figure(case,group,p_axis,i_axis,d_axis,rate,evaluated,
                               base_gains,mode,metadata)
    stem = 'pid_3d' if mode == '3d' else 'pid_i1'
    try:
        for extension in ('png','pdf'):
            path = destination/(stem+'.'+extension)
            fig.savefig(path,dpi=400 if extension=='png' else 220,bbox_inches='tight')
    finally:
        plt.close(fig)
    info['files'] = [stem+'.png',stem+'.pdf']
    (destination/'plot_metadata.json').write_text(json.dumps(info,default=_json_default,
        indent=2,allow_nan=False)+'\n')
    return info
