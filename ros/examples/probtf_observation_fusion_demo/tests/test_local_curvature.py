"""The new selection model reuses one frozen law, not an altered objective."""
from copy import deepcopy
import math

import numpy as np
import pytest

from probtf_observation_fusion_demo.local_curvature import build_local_curvature
from probtf_observation_fusion_demo.selection_gradient import selection_gradient_at_command
from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor, direction_penalty
from test_visual_approach_integration import problems, controller, forbid_sampling


def _toy(*, second_component=False, covariance_changes=False, quaternion_changes=False):
    step=1e-5
    source=(0.,0.)
    commands=[source,(step,0.),(-step,0.),(0.,step),(0.,-step)]
    coordinates=[dict(joint_index=i,method='two-sided unequal-step quadratic derivative',
        forward=dict(status='ok',step=step),backward=dict(status='ok',step=-step)) for i in range(2)]
    task=dict(source_command=list(source),source_stamp=10.,body_state_ids=['body'],
        goal_state_ids=['goal1','goal2'] if second_component else ['goal1'],
        selected_K=[100.,120.],parameter_density_id='continuous_K',
        selected_gradient=[.3,.4],finite_difference_diagnostics={'coordinates':coordinates})
    probes={}
    for command in commands:
        branches=[]
        for index in range(2 if second_component else 1):
            scale=1.+index
            mean=np.array([scale*command[0],2.*scale*command[1],0.])
            covariance=np.diag([4.*math.exp(command[0]) if covariance_changes else 4.,9.,16.])
            quaternion=([math.sqrt(1.-(.5*command[0])**2),.5*command[0],0.,0.]
                        if quaternion_changes else [1.,0.,0.,0.])
            full=np.zeros((6,6));full[:3,:3]=covariance;full[3:,3:]=.02*np.eye(3)
            weight=(.25 if index==0 else .75) if second_component else 1.
            branches.append(dict(relative_position_mean=mean.tolist(),
                relative_position_covariance=covariance.tolist(),
                relative_orientation_mean_wxyz=quaternion,query_covariance=full.tolist(),
                state_id='goal'+str(index+1),label='native_'+str(index),
                weight=weight,log_weight=math.log(weight),log_density_parts=[-1e20,0.]))
        probes[command]={'branches':tuple(branches)}
    return task,probes


def _disabled():
    return VisibilityFactor({'enabled':False,'weight':2e6})


def test_linear_gaussian_Fisher_and_native_mixture_weights_are_analytic():
    task,probes=_toy(second_component=True)
    H,record=build_local_curvature(task,probes,{},_disabled(),
        {'regularization_absolute':1e-6,'regularization_relative':1e-6})
    # Both source densities have the huge common scale -1e20. The .25/.75
    # native masses must survive cancellation before the logarithmic sum.
    factor=.25+4.*.75
    expected=np.diag([factor/4.,4.*factor/9.])
    np.testing.assert_allclose(record['task_curvature'],expected,atol=1e-13,rtol=0.)
    np.testing.assert_allclose([r['responsibility'] for r in record['task_components']],
                              [.25,.75],atol=1e-15,rtol=0.)
    epsilon=1e-6+1e-6*np.trace(expected)/2.
    np.testing.assert_allclose(H,expected+epsilon*np.eye(2),atol=1e-13,rtol=0.)
    assert record['component_count']==2
    assert record['additional_command_pushes']==record['additional_core_queries']==0
    assert record['exact_Hessian'] is False


def test_covariance_derivative_and_Bingham_tangent_mean_both_enter_Fisher():
    task,probes=_toy(covariance_changes=True,quaternion_changes=True)
    _,record=build_local_curvature(task,probes,{},_disabled(),{})
    # Position mean 1/4 + covariance .5 + Bingham-derived tangent mean 1/.02.
    expected=np.diag([.25+.5+50.,4./9.])
    np.testing.assert_allclose(record['task_curvature'],expected,atol=1e-8,rtol=0.)
    # Antipodal quaternion sign choices cannot create a different model.
    changed=deepcopy(probes)
    for probe in changed.values():
        for branch in probe['branches']:
            branch['relative_orientation_mean_wxyz']=(-np.array(branch['relative_orientation_mean_wxyz'])).tolist()
    _,other=build_local_curvature(task,changed,{},_disabled(),{})
    np.testing.assert_array_equal(record['task_curvature'],other['task_curvature'])


def test_covariance_and_branch_records_are_not_regularized_or_mutated():
    task,probes=_toy()
    saved=deepcopy(probes)
    H,record=build_local_curvature(task,probes,{},_disabled(),
        {'regularization_absolute':.1,'regularization_relative':.2})
    assert probes==saved
    np.linalg.cholesky(H)
    assert record['regularization_role'].startswith('declared local model only')
    assert record['candidate_law_modified'] is False


def test_same_one_sided_task_stencil_builds_the_same_linear_Fisher():
    task,probes=_toy()
    task['finite_difference_diagnostics']['coordinates'][0]['forward']['status']='outside_position_domain'
    _,record=build_local_curvature(task,probes,{},_disabled(),{})
    np.testing.assert_allclose(record['task_curvature'],np.diag([.25,4./9.]),atol=1e-14,rtol=0.)


@pytest.mark.parametrize('offset',[.05,.5])
def test_view_GN_reuses_each_probe_and_is_zero_only_inside_cone(offset):
    from types import SimpleNamespace
    task,probes=_toy()
    factor=VisibilityFactor({'enabled':True,'weight':2e6})
    views={}
    for command in probes:
        query=SimpleNamespace(mean=np.array([offset+command[0],command[1],1.,0.,0.,0.]),
            covariance=np.eye(6),stamp=10.,command=command,version=1,state_id='goal1')
        component=SimpleNamespace(weight=1.,log_weight=0.,label='native')
        views[command]=factor.evaluate([component],[query])
    _,record=build_local_curvature(task,probes,views,factor,{})
    p=np.array([offset,0.,1.]);radius=np.linalg.norm(p);unit=p/radius
    derivative=(unit[2]*unit-np.array([0.,0.,1.]))[:2]/radius
    expected=(2e6*np.outer(derivative,derivative) if offset==.5 else np.zeros((2,2)))
    np.testing.assert_allclose(record['view_curvature'],expected,atol=1e-7,rtol=1e-10)
    assert record['view_components'][0]['active_residual']==(offset==.5)


def test_unavailable_optional_eigenvalue_audit_does_not_reject_finite_SPD_model(monkeypatch):
    task,probes=_toy()
    def unavailable(*args,**kwargs):raise np.linalg.LinAlgError('optional audit unavailable')
    monkeypatch.setattr(np.linalg,'eigvalsh',unavailable)
    H,record=build_local_curvature(task,probes,{},_disabled(),{})
    np.linalg.cholesky(H)
    assert record['eigenvalues'] is None
    assert 'optional audit unavailable' in record['eigenvalue_audit_error']


def test_optional_stationary_solve_and_Cholesky_audits_do_not_reject_finite_model(monkeypatch):
    task,probes=_toy()
    def unavailable(*args,**kwargs):raise np.linalg.LinAlgError('optional audit unavailable')
    monkeypatch.setattr(np.linalg,'solve',unavailable)
    monkeypatch.setattr(np.linalg,'cholesky',unavailable)
    H,record=build_local_curvature(task,probes,{},_disabled(),{})
    assert np.all(np.isfinite(H))
    assert record['unconstrained_equilibrium_command'] is None
    assert record['unconstrained_equilibrium_within_joint_positions'] is None
    assert 'optional audit unavailable' in record['unconstrained_equilibrium_diagnostic_error']
    assert record['positive_definite_cholesky_audit'] is False


@pytest.mark.parametrize('angle', [0.,4.,9.9])
def test_new_default_cone_is_exactly_flat_inside_ten_degrees(angle):
    position=np.array([math.sin(math.radians(angle)),0.,math.cos(math.radians(angle))])
    penalty,gradient=direction_penalty(position,penalty_mode='tolerance_cone')
    assert penalty==0.
    np.testing.assert_array_equal(gradient,np.zeros(3))
    default=VisibilityFactor({'enabled':True,'weight':2e6})
    assert default.penalty_mode=='tolerance_cone'
    assert default.tolerance_angle_deg==10.


@pytest.mark.parametrize('position', [[.4,.2,1.],[-.3,.1,-.9],[1.,0.,0.]])
def test_cone_position_derivative_matches_independent_formula(position):
    position=np.asarray(position,dtype=float)
    value,gradient=direction_penalty(position,penalty_mode='tolerance_cone')
    def independent(p):
        residual=max(0.,math.cos(math.radians(10.))-p[2]/np.linalg.norm(p))
        return .5*residual**2
    assert value==pytest.approx(independent(position),abs=1e-15)
    step=1e-5
    finite=[(independent(position+step*axis)-independent(position-step*axis))/(2*step)
            for axis in np.eye(3)]
    np.testing.assert_allclose(gradient,finite,rtol=2e-9,atol=2e-11)


def test_calibration_rationale_and_all_native_modes_are_recorded():
    from types import SimpleNamespace
    camera=dict(fx=720.,fy=720.,cx=480.,cy=360.,width=960,height=720)
    factor=VisibilityFactor(dict(enabled=True,weight=2e6,intrinsics=camera))
    components=[SimpleNamespace(weight=.25,log_weight=math.log(.25),label='outside'),
                SimpleNamespace(weight=.75,log_weight=math.log(.75),label='inside')]
    queries=[SimpleNamespace(mean=np.r_[p,[0.,0.,0.]],covariance=np.eye(6),
        stamp=10.,command=(0.,),version=1,state_id=str(i))
        for i,p in enumerate(([.4,0.,1.],[0.,0.,1.]))]
    record=factor.evaluate(components,queries)
    assert record['status']=='ok'
    assert len(record['per_mode_record'])==2
    assert record['per_mode_record'][1]['inside_tolerance_cone']
    assert record['per_mode_record'][1]['penalty']==0.
    assert record['penalty']==.25*record['per_mode_record'][0]['penalty']
    assert record['tolerance_basis']['calibration_minimum_half_fov_deg']==pytest.approx(26.565051177)
    assert record['all_marker_corners_or_occlusion_certified'] is False


def _actual(problem,enabled):
    current=controller(problem)
    current.local_quadratic_model_settings=dict(enabled=enabled,
        regularization_absolute=1e-6,regularization_relative=1e-6)
    return current


def test_actual_frozen_law_curvature_reuses_stencil_and_keeps_complete_J_g_K(problems,monkeypatch):
    problem=problems['imu_marker'];source,mapping=problem[3:5]
    factor=VisibilityFactor(dict(enabled=True,weight=2e6))
    legacy=selection_gradient_at_command(_actual(problem,False),source,factor,command_map=mapping).record
    current=_actual(problem,True)
    pushes=[];original=mapping.push
    def push(state,command,body_ids,actions):
        pushes.append((state.state_id,tuple(command)))
        return original(state,command,body_ids,actions)
    monkeypatch.setattr(mapping,'push',push)
    forbid_sampling(monkeypatch)
    saved=[(c.state.state_id,c.state.covariance.copy()) for c in source.stages[-1].components+source.stages[-1].goal_components]
    record=selection_gradient_at_command(current,source,factor,command_map=mapping).record
    assert record['status']=='ok',record['reason']
    for key in ('objective_at_source','objective_at_source_parts','gradient','task_gradient',
                'view_gradient','source_branches','finite_difference_diagnostics',
                'body_state_ids','goal_state_ids','parameter_density_id','selected_K'):
        assert record[key]==legacy[key]
    count=record['finite_difference_diagnostics']['objective_evaluation_count']
    assert len(pushes)==count*len(source.stages[-1].goal_components)
    assert len(set(pushes))==len(pushes)
    diagnostics=record['local_model_diagnostics']
    assert diagnostics['additional_command_pushes']==diagnostics['additional_core_queries']==0
    H=np.asarray(record['curvature']);np.linalg.cholesky(H)
    expected=np.asarray(diagnostics['task_curvature'])+np.asarray(diagnostics['view_curvature'])
    expected+=diagnostics['regularization_added']*np.eye(H.shape[0])
    np.testing.assert_allclose(H,expected,atol=1e-8,rtol=1e-14)
    # Initial observed marker center lies inside the declared cone, so there
    # is no centering force or view curvature to compete with the task here.
    assert all(m['inside_tolerance_cone'] for m in record['visibility_factor']['per_mode_record'])
    np.testing.assert_array_equal(record['view_gradient'],np.zeros(6))
    np.testing.assert_array_equal(diagnostics['view_curvature'],np.zeros((6,6)))
    for key in ('source_stamp','source_command','body_state_ids','goal_state_ids','selected_K','parameter_density_id'):
        assert diagnostics[key]==record[key]
    for item,(identity,covariance) in zip(source.stages[-1].components+source.stages[-1].goal_components,saved):
        assert item.state.state_id==identity
        np.testing.assert_array_equal(item.state.covariance,covariance)
