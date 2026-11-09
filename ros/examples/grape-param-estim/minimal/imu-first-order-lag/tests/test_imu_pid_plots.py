"""PID figure semantics: exact plane, historical colors and immutable data."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import imu_pid_plots as plots


def load_tests(_loader, suite, _pattern):
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            suite.addTest(unittest.FunctionTestCase(function, description=name))
    return suite


def gains():
    return np.array([[3.,.1,1.],[5.,1.,2.5],[20.,1.,8.],[4.,1.,2.]])


def test_exact_plane_pending_mask_and_no_projected_xy_reference():
    p,d=np.array([1.,3.,6.]),np.array([.25,1.,2.])
    values=np.arange(9.).reshape(3,1,3)/8.
    done=np.ones_like(values,dtype=bool);done[1,0,1]=False
    original=values.copy()
    fig,axis,meta=plots.build_figure('failure1','xy',p,[1.],d,values,done,gains(),mode='plane')
    try:
        surface=next(c for c in axis.collections if c.get_gid()=='evaluated_node_glyphs')
        assert len(surface.get_paths())==8
        np.testing.assert_array_equal(surface.get_array(),values[done])
        assert not meta['recorded_reference_shown']
        assert not meta['reference_projected']
        assert not meta['pending_nodes_shown_as_zero']
        assert meta['exact_fixed_i']==1.
        assert all(c.get_gid()!='recorded_failure_reference' for c in axis.collections)
        np.testing.assert_array_equal(values,original)
    finally:
        plots.plt.close(fig)
    try:
        plots.build_figure('failure1','xy',p,[.999999],d,values,done,gains(),mode='plane')
    except ValueError as error:
        assert 'exact I axis' in str(error)
    else:
        raise AssertionError('A nearby I plane was accepted')


def test_historical_color_rule_and_recorded_i_display_cut():
    p,i,d=np.array([1.,3.,6.]),np.array([.25,1.,2.]),np.array([.25,1.,2.])
    values=np.zeros((3,3,3));values[:,1,:]=.5;values[:,2,:]=1.
    done=np.ones_like(values,dtype=bool);done[1,2,1]=False
    before=values.copy()
    fig,axis,meta=plots.build_figure('failure1','yaw',p,i,d,values,done,gains(),mode='3d')
    try:
        assert meta['cut_at_recorded_i']
        assert meta['displayed_axes'][1]==[1.,2.]
        assert meta['full_axes'][1]==[.25,1.,2.]
        assert meta['white_mix']==.75 and meta['voxel_opacity']==.3
        assert not meta['threshold_colors']
        assert meta['color_range']==[0.,1.]
        assert meta['recorded_reference_shown']
        assert meta['displayed_evaluated_count']==17
        np.testing.assert_array_equal(values,before)
        settings=plots.historical_settings()
        positions=np.linspace(0.,1.,256)
        from matplotlib import colors
        expected=np.array([colors.to_rgba(colors.to_hex(v)) for v in
            .75+.25*plots.plt.get_cmap('RdYlGn')(positions)[:,:3]])
        np.testing.assert_allclose(plots.rate_colormap(settings)(positions),expected,atol=1./255.)
    finally:
        plots.plt.close(fig)


def test_render_grid_writes_only_requested_artifacts_and_retains_data_hash():
    values=np.array([0.,.25,.5,1.]).reshape(2,1,2)
    done=np.ones_like(values,dtype=bool)
    with TemporaryDirectory(prefix='imu-pid-plot-test-') as folder:
        with patch.object(plots.plt.Figure,'savefig') as save:
            info=plots.render_grid(folder,'failure1','roll_pitch',[1.,2.],[1.],[2.,4.],
                                   values,done,gains(),mode='plane',metadata={'N':128})
        assert save.call_count==2
        assert info['files']==['pid_i1.png','pid_i1.pdf']
        assert info['source_array_sha256']['rate']==plots._hash(values)
        assert not info['law_or_statistics_changed']
        assert (Path(folder)/'plot_metadata.json').exists()
        assert info['supplied_metadata']['N']==128


def test_analytic_certificate_is_not_reported_as_numerically_computed_poles():
    values=np.zeros((3,1,3));done=np.ones_like(values,dtype=bool)
    values[0,0,0]=np.nan;done[0,0,0]=False
    before=values.copy()
    criterion='Q = Pr(full-state spectral radius < 1) = 0'
    fig,axis,meta=plots.build_figure('failure1','xy',[1.,3.,6.],[1.],[.25,1.,2.],
        values,done,gains(),mode='plane',metadata={
            'assessment_method':'analytic_translation_symmetry_unit_modes',
            'criterion_label':criterion})
    try:
        assert 'Certified nodes' in axis.get_title()
        assert 'Evaluated nodes' not in axis.get_title()
        assert any('no numerical poles were computed' in text.get_text() for text in fig.texts)
        assert any('not a flight-failure probability' in text.get_text() for text in fig.texts)
        assert meta['analytic_certificate']
        assert meta['numerical_poles_computed'] is False
        assert meta['color_quantity']==criterion
        assert not any(key in meta for key in ('spectral_radius','pole_valid_count','computed_poles'))
        assert meta['source_array_sha256']['rate']==plots._hash(before)
        np.testing.assert_array_equal(values,before)
        surface=next(c for c in axis.collections if c.get_gid()=='evaluated_node_glyphs')
        assert len(surface.get_paths())==8
        np.testing.assert_array_equal(surface.get_array(),np.zeros(8))
    finally:
        plots.plt.close(fig)


def test_reference_view_extension_and_exact_plane_without_projection():
    p,i,d=np.array([5.,20.,40.]),np.array([.25,1.,2.]),np.array([2.,8.,16.])
    values=np.zeros((3,3,3));done=np.ones_like(values,dtype=bool)
    refs=[dict(kind='failure',label='Failure',gain=[20.,1.,8.],
               rest_pid={'success':0.,'failure':0.}),
          dict(kind='success',label='Success ref.',gain=[13.,1.,20.],
               rest_pid={'success':0.,'failure':0.})]
    fig,axis,meta=plots.build_figure('failure1','roll_pitch',p,i,d,values,done,gains(),
        mode='3d',metadata={'assessment_method':'analytic_translation_symmetry',
                            'reference_points':refs})
    try:
        assert meta['full_axes'][2]==[2.,8.,16.]
        assert meta['view_limits']['D'][1]>20.
        assert all(not point['projected'] for point in meta['reference_points'])
        assert any(c.get_gid()=='success_reference' for c in axis.collections)
        annotations=[t for t in axis.texts if t.get_gid() in (
            'success_reference_label','failure_reference_label')]
        assert len(annotations)==2
        for annotation in annotations:
            assert '\nrest PID\nsuccess: rate = 0.0%\nfailure: rate = 0.0%' in annotation.get_text()
            assert annotation.get_horizontalalignment()=='center'
        np.testing.assert_array_equal(values,np.zeros((3,3,3)))
    finally:
        plots.plt.close(fig)
    offplane=[dict(kind='success',label='Success ref.',gain=[4.,.1,2.],
                   rest_pid={'success':0.,'failure':0.})]
    fig,axis,meta=plots.build_figure('failure1','xy',[1.,3.,6.],[1.],[.25,1.,2.],
        np.zeros((3,1,3)),np.ones((3,1,3),bool),gains(),mode='plane',metadata={
            'assessment_method':'analytic_translation_symmetry','reference_points':offplane})
    try:
        assert all(not point['shown'] and not point['projected'] for point in meta['reference_points'])
        assert all(c.get_gid() not in ('success_reference','recorded_failure_reference') for c in axis.collections)
        assert all(point['omitted_reason']=='actual I is not 1' for point in meta['reference_points'])
    finally:
        plots.plt.close(fig)
