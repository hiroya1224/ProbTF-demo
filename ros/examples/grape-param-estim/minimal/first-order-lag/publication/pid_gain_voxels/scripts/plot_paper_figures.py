#!/usr/bin/env python3
"""All-axis success/failure PID comparison, with exact references and pastel voxel colors."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pid-focus-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from data_io import ROOT, CASES, GROUPS, Grid, LABELS, load_grid, slice_at, exposed_faces

def merge_uniform_cells(data, x_edges, y_edges):
    """Merge only exactly equal adjacent cells into axis-aligned rectangles.

    This preserves the piecewise-constant field and its nonuniform geometry,
    while avoiding thousands of identical vector patches in PDF/SVG files.
    """
    active, rectangles = {}, []
    for row, values in enumerate(data):
        ends = np.r_[0, np.flatnonzero(values[:-1] != values[1:]) + 1, len(values)]
        current = {}
        for start, stop in zip(ends[:-1], ends[1:]):
            key = (int(start), int(stop), float(values[start]))
            initial = active.pop(key)[0] if key in active else row
            current[key] = (initial, row + 1)
        for (start, stop, value), (first, last) in active.items():
            rectangles.append((first, last, start, stop, value))
        active = current
    for (start, stop, value), (first, last) in active.items():
        rectangles.append((first, last, start, stop, value))
    vertices = []
    for first, last, start, stop, value in rectangles:
        x0, x1, y0, y1 = x_edges[start], x_edges[stop], y_edges[first], y_edges[last]
        vertices.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return np.asarray(vertices), np.asarray([r[4] for r in rectangles]), rectangles



def boundary_segments(classes, x_edges, y_edges):
    segments = []
    rows, cols = np.nonzero(classes[:, :-1] != classes[:, 1:])
    for row, col in zip(rows, cols):
        x = x_edges[col + 1]
        segments.append(((x, y_edges[row]), (x, y_edges[row + 1])))
    rows, cols = np.nonzero(classes[:-1, :] != classes[1:, :])
    for row, col in zip(rows, cols):
        y = y_edges[row + 1]
        segments.append(((x_edges[col], y), (x_edges[col + 1], y)))
    return segments


def label_gain(gain):
    return "(" + ", ".join(format(float(x), ".5g") for x in gain) + ")"


def reference_mark(grid, name, alpha):
    mark = grid.mark(name)
    mark['classification'] = 3 if mark['survival'] >= alpha else 2
    return mark


def point_style(grid, name, settings):
    coincident = np.array_equal(grid.gain('success'), grid.gain('recorded'))
    if name == 'success':
        return dict(marker='D', s=68, color=settings['success_color'], edgecolor='white', linewidth=1.1, zorder=8)
    if coincident:
        return dict(marker='o', s=190, facecolors='none', edgecolors=settings['recorded_color'], linewidth=2.2, zorder=7)
    return dict(marker='X', s=95, color=settings['recorded_color'], edgecolor='white', linewidth=1.15, zorder=7)


def point_text(grid, name, settings):
    mark = reference_mark(grid, name, settings['alpha'])
    label = 'Success' if name == 'success' else 'Failure'
    return label + '\n' + label_gain(mark['gain']) + '\nrate : ' + format(100 * mark['survival'], '.1f') + '%'


def annotation_style(name, settings):
    color = settings['success_color' if name == 'success' else 'recorded_color']
    return dict(ha='center', va='center', fontsize=9.5, color=color, zorder=10,
                annotation_clip=False,
                arrowprops={'arrowstyle':'->', 'lw':1.55, 'color':color, 'shrinkA':4, 'shrinkB':7},
                bbox={'facecolor':'white', 'edgecolor':color, 'linewidth':.6, 'alpha':.97, 'pad':3})


def title_panel(ax, grid):
    ax.set_title(LABELS[grid.group], fontsize=11, color='#263238', pad=3)


def comparison_slice(grids, settings):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    fig.subplots_adjust(left=.055, right=.982, top=.87, bottom=.15, wspace=.34, hspace=.48)
    for row, case in enumerate(CASES):
        fig.text(.012, .695 - row * .423, case.replace('failure', 'Failure '), rotation=90, ha='center', va='center', fontsize=14)
        for column, group in enumerate(GROUPS):
            grid, ax = grids[case, group], axes[row, column]
            index = grid.index(grid.gain('success'))[1]
            if grid.gain('recorded')[1] != grid.axes[1][index]:
                raise ValueError('Both references must lie on the exact I plane')
            _, survival, edges = slice_at(grid, 1, index)
            green = survival >= settings['alpha']
            vertices, values, _ = merge_uniform_cells(green.astype(float), *edges)
            ax.add_collection(PolyCollection(vertices, array=values,
                cmap=colors.ListedColormap([settings['red'], settings['green']]), norm=colors.Normalize(0, 1),
                edgecolors='none', linewidth=0, antialiased=False))
            ax.add_collection(LineCollection(boundary_segments(green, *edges), colors='#62736B', linewidths=.65))
            for edge, setter in zip(edges, (ax.set_xlim, ax.set_ylim)):
                span = edge[-1] - edge[0]
                setter(edge[0] - .10 * span, edge[-1] + .10 * span)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel('P gain', fontsize=10); ax.set_ylabel('D gain', fontsize=10)
            ax.tick_params(labelsize=9); ax.spines[['top', 'right']].set_visible(False)
            title_panel(ax, grid)
            ax.text(.98, .98, 'I = ' + format(grid.gain('success')[1], '.5g'), transform=ax.transAxes, ha='right', va='top', fontsize=9)
            pspan, dspan = [edge[-1] - edge[0] for edge in edges]
            for name in ('recorded', 'success'):
                gain = grid.gain(name)
                ax.scatter(gain[0], gain[2], **point_style(grid, name, settings))
                if name == 'success': offset = (.06 * pspan, .18 * dspan)
                else: offset = ((.10 if group in ('roll_pitch', 'z') else -.12) * pspan, (-.07 if group == 'roll_pitch' else -.18) * dspan)
                ax.annotate(point_text(grid, name, settings), xy=(gain[0],gain[2]),
                            xytext=(gain[0]+offset[0],gain[2]+offset[1]), **annotation_style(name, settings))
    return fig


def comparison_volume(grids, settings):
    fig = plt.figure(figsize=(16, 8.6))
    for row, case in enumerate(CASES):
        fig.text(.013, .69 - row * .425, case.replace('failure', 'Failure '), rotation=90, ha='center', va='center', fontsize=14)
        for column, group in enumerate(GROUPS):
            grid = grids[case, group]
            ax = fig.add_axes([.043 + .238 * column, .545 - .425 * row, .224, .35],
                              projection='3d', computed_zorder=False)
            fixed = grid.index(grid.gain('success'))[1]
            lower_i = grid.axes[1][fixed]
            if grid.gain('recorded')[1] != lower_i:
                raise ValueError('Both references must lie on the exact cut plane')
            cut = Grid(grid.case, grid.group, grid.size,
                {'p':grid.axes[0], 'i':grid.axes[1][fixed:], 'd':grid.axes[2],
                 'survival_fraction':grid.survival[:,fixed:,:]}, grid.references, grid.metadata)
            faces, classes = exposed_faces(cut, settings['alpha'])
            rgba = np.empty((len(classes),4))
            for code,name in ((2,'red'),(3,'green')):
                rgba[classes==code] = colors.to_rgba(settings[name],settings[name+'_opacity'])
            ax.add_collection3d(Poly3DCollection(faces, facecolors=rgba, edgecolors='none', linewidth=0,
                                 zsort='average', rasterized=True, zorder=1))
            for axis,setter in zip(cut.axes,(ax.set_xlim,ax.set_ylim,ax.set_zlim)):
                span = axis[-1]-axis[0]; setter(axis[0]-.10*span,axis[-1]+.10*span)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.set_major_locator(MaxNLocator(3)); axis.set_pane_color((1,1,1,1))
                axis._axinfo['grid'].update(color=(.6,.6,.6,.18),linewidth=.45)
            ax.tick_params(labelsize=8,pad=0)
            ax.set_xlabel('P gain',labelpad=0,fontsize=10)
            ax.set_ylabel('I gain',labelpad=0,fontsize=10)
            ax.set_zlabel('D gain',labelpad=0,fontsize=10)
            ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho')
            camera=settings['camera_oblique']; ax.view_init(elev=camera['elevation'],azim=camera['azimuth'])
            title_panel(ax,grid)
            for name in ('recorded','success'):
                gain=grid.gain(name)
                ax.scatter(*gain,depthshade=False,**point_style(grid,name,settings))
                x,y,_=proj3d.proj_transform(*gain,ax.get_proj())
                if name=='success':offset=(17,32) if group in ('yaw','xy') else (-20,34)
                elif group=='roll_pitch':offset=(22,-28) if case=='failure1' else (-12,-28)
                elif group=='yaw':offset=(-28,-30)
                elif group=='xy':offset=(-8,-30)
                else:offset=(28,-32)
                ax.annotate(point_text(grid,name,settings),xy=(x,y),xytext=offset,textcoords='offset points',
                            **annotation_style(name,settings))
    return fig


def save_comparison(fig, grids, settings, output, view):
    all_success_green=all(reference_mark(grid,'success',settings['alpha'])['classification']==3 for grid in grids.values())
    title='All PID gains: Success is in the green region on every axis' if all_success_green else 'All PID gains: success and failure reference points'
    fig.suptitle(title,y=.99,fontsize=17)
    fig.legend(handles=[Patch(facecolor=settings['green'],label='Green region: survival ≥ %.0f%%'%(100*settings['alpha'])),
                        Patch(facecolor=settings['red'],label='Red region: survival < %.0f%%'%(100*settings['alpha']))],
               loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=11)
    plane_note='3D volumes are cut at each shared reference I to show both points.' if view=='3D' else 'Each exact P–D slice passes through both reference points.'
    fig.text(.5,.03,plane_note+' Purple marks identify the failure-flight gains; Z gains coincide.',ha='center',fontsize=10,color='#52606B')
    fig.text(.5,.007,'Each case uses its own posterior. Other controller gains are fixed at the successful-flight values.',ha='center',fontsize=9,color='#52606B')
    name='all_pid_success_failure_'+view
    for ext in settings['formats']:
        path=output/(name+'.'+ext)
        fig.savefig(path,dpi=settings['dpi'] if ext=='png' else settings['surface_raster_dpi'],bbox_inches='tight',pad_inches=.12)
        if path.stat().st_size>=50_000_000:raise ValueError('File exceeds 50 MB')
    plt.close(fig)
    return {'figure':name,'groups':list(GROUPS),'cases':list(CASES),'view':view,'panel_count':8,
            'caption':'The success reference is green in all eight case/axis volumes. Recorded failure-flight group gains are explicitly located; only Roll/Pitch is red under the single-group substitutions.',
            'references':{case:{group:{name:reference_mark(grids[case,group],name,settings['alpha']) for name in ('success','recorded')} for group in GROUPS} for case in CASES}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--settings',type=Path,default=ROOT/'figure_settings.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--grid',type=int,choices=(33,65),default=65)
    parser.add_argument('--view',choices=('PD','3D','both'),default='both')
    args=parser.parse_args();settings=json.loads(args.settings.read_text())
    plt.rcParams.update({'font.family':settings['font_family'],'font.size':settings['font_size'],
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    output=args.output or args.root/'figures';output.mkdir(parents=True,exist_ok=True)
    grids={(case,group):load_grid(case,group,args.grid,args.root) for case in CASES for group in GROUPS}
    records=[]
    for view,draw in (('3D',comparison_volume),('PD',comparison_slice)):
        if args.view in (view,'both'):records.append(save_comparison(draw(grids,settings),grids,settings,output,view))
    (output/'figure_index.json').write_text(json.dumps(records,indent=2)+'\n')
    with (output/'figure_index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['figure','groups','view','cases','panel_count','caption'],lineterminator='\n')
        writer.writeheader();writer.writerows({key:row[key] for key in writer.fieldnames} for row in records)
    print(json.dumps({'figures':[row['figure'] for row in records],'panels_per_figure':8,'all_success_green':all_success(grids,settings)},indent=2))


def all_success(grids,settings):
    return all(reference_mark(grid,'success',settings['alpha'])['classification']==3 for grid in grids.values())


if __name__=='__main__':
    main()
