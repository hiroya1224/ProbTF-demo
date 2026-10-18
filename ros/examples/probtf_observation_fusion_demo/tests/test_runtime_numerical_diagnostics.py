"""Finite numerical quality reports never veto usable runtime calculations."""
from dataclasses import replace

import numpy as np
import pytest

from probtf_observation_fusion_demo.async_stiffness import PhysicalActionEngine
from probtf_observation_fusion_demo.display import goal_pose_records, query_record
from probtf_observation_fusion_demo.node import verify_adapter_identity
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap

from test_async_stiffness import observed
from test_display_identity import combined
from test_goal_axes_display import _fixture_snapshot


def test_finite_anchor_with_large_residual_still_initializes_real_command_action(observed,monkeypatch):
    scene,inference,_,_,settings,packet=observed
    engine=PhysicalActionEngine(scene.urdf,inference.body_ids,settings)
    gravity=engine.identifier.robot.tau_gravity
    monkeypatch.setattr(engine.identifier.robot,'tau_gravity',lambda q:gravity(q)+.001)
    engine.set(packet)
    record=engine.provenance()['numerical_anchor_diagnostics']
    assert record['residual_inf']>1e-9
    assert record['residual_within_legacy_tolerance'] is False
    assert record['numerical_quality_is_acceptance_gate'] is False
    action=engine.anchor(packet.proposal['source_command'])
    assert np.all(np.isfinite(action['command_jacobian']))


def test_nonsingular_negative_curvature_is_not_an_extra_command_map_gate(observed,monkeypatch):
    _,_,result,identifier,_,packet=observed
    monkeypatch.setattr(identifier.sensitivity,'equilibrium_jacobians',
        lambda *_:(-np.eye(6),np.eye(6)))
    mapping=StiffnessCommandMap(identifier,result,identifier.last_record['command_proposal'])
    record=mapping.record()
    assert record['numerical_anchor_diagnostics']['positive_curvature'] is False
    assert np.all(np.isfinite(record['anchor_command_jacobian']))
    assert record['density_id']==packet.proposal['density_id']


def test_diagnostic_only_eigendecomposition_cannot_stop_a_valid_solve(observed,monkeypatch):
    _,_,result,identifier,_,_=observed
    def fail(*_args,**_kwargs):
        raise np.linalg.LinAlgError('diagnostic eigensolver deliberately unavailable')
    monkeypatch.setattr(np.linalg,'eigvalsh',fail)
    mapping=StiffnessCommandMap(identifier,result,identifier.last_record['command_proposal'])
    diagnostics=mapping.record()['numerical_anchor_diagnostics']
    assert diagnostics['positive_curvature'] is None
    assert 'deliberately unavailable' in diagnostics['eigendecomposition_diagnostic_error']
    assert np.all(np.isfinite(mapping.record()['anchor_command_jacobian']))


def test_covariance_decomposition_quality_is_recorded_without_altering_the_display_law(combined):
    _,_,stage,snapshot=combined
    point=snapshot.points[0]
    query=replace(point.query,within=point.query.within+np.eye(3)*1e-5)
    changed=replace(point,query=query)
    view=replace(snapshot,points=(changed,)+snapshot.points[1:])
    verify_adapter_identity(view,stage)
    record=query_record(changed)
    assert record['total_covariance_identity_verified'] is False
    assert record['numerical_quality_is_stop_gate'] is False
    np.testing.assert_array_equal(record['mean'],point.query.mean)
    np.testing.assert_array_equal(record['covariance'],point.query.covariance)


def test_finite_modal_frame_roundoff_does_not_stop_publication():
    _,_,_,snapshot=_fixture_snapshot()
    point=next(p for p in snapshot.points if p.frame_label=='switch_goal' and p.axis=='x')
    query=replace(point.query,mode=point.query.mode+np.array([1e-11,2e-11,0.]))
    view=replace(snapshot,points=tuple(replace(p,query=query) if p is point else p for p in snapshot.points))
    poses=goal_pose_records(view)
    assert poses and all(np.all(np.isfinite(p['rotation_wxyz'])) for p in poses)
    diagnostics=next(p['numerical_diagnostics'] for p in poses if p['component_id']==point.component_id)
    assert diagnostics['orthogonality_residual_inf']>1e-10
    assert diagnostics['numerical_quality_is_stop_gate'] is False
