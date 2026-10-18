"""Common huge log scales cannot erase native mixture masses or FD changes."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from probtf_observation_fusion_demo import moving_control
from probtf_observation_fusion_demo.joint_batch_fusion import (
    add_log_evidence_parts,log_evidence_parts_difference,
)
from test_goal_mixture_criterion import _problem


def test_common_log_pdf_is_removed_before_native_weights_are_added():
    common=-1e20
    branches=[dict(log_density_parts=(common,0.),log_weight=np.log(.25)),
              dict(log_density_parts=(common,1.),log_weight=np.log(.75))]
    old_terms=[float(common+branch['log_density_parts'][1])+branch['log_weight'] for branch in branches]
    assert old_terms==[common,common]  # actual float64 loss, not a tolerance claim
    parts=moving_control._relative_mixture_log_parts(branches)
    expected=float(logsumexp([np.log(.25),1.+np.log(.75)]))
    assert log_evidence_parts_difference(parts,(common,0.))==pytest.approx(expected,abs=1e-15)
    reversed_masses=[dict(branch,log_weight=np.log(weight))
                     for branch,weight in zip(branches,(.75,.25))]
    other=moving_control._relative_mixture_log_parts(reversed_masses)
    assert abs(log_evidence_parts_difference(parts,other))>.4


def test_finite_underflow_mass_can_dominate_at_its_large_relative_density():
    common=-1e20
    branches=[dict(log_density_parts=(common,0.),log_weight=0.),
              dict(log_density_parts=(common,1100.),log_weight=-1000.)]
    parts=moving_control._relative_mixture_log_parts(branches)
    assert log_evidence_parts_difference(parts,(common,0.))==pytest.approx(100.,abs=1e-13)
    assert len(branches)==2 and branches[1]['log_weight']==-1000.


@pytest.mark.parametrize('weights',[(.25,.75),(.75,.25)])
def test_complete_core_task_gradient_is_invariant_to_a_huge_common_constant(monkeypatch,weights):
    _,result,controller=_problem(weights=weights)
    original=moving_control._relative_density
    baseline=controller.gradient_at_command(result).record
    baseline_command,baseline_legacy=controller.step(result,.5)
    covariances=[item.state.covariance.copy() for item in result.stages[-1].components]
    common=-1e20
    def shifted(query,*args):
        value,record=original(query,*args)
        # This is the exact same q-dependent normalized density multiplied
        # by a q-independent common factor. Its complete gradient is equal.
        record['log_density_parts']=list(add_log_evidence_parts((common,0.),record['log_density_parts']))
        return float(common+value),record
    monkeypatch.setattr(moving_control,'_relative_density',shifted)
    actual=controller.gradient_at_command(result).record
    assert actual['status']==baseline['status']=='ok'
    assert abs(actual['gradient'][0])>1e-3
    np.testing.assert_allclose(actual['gradient'],baseline['gradient'],rtol=1e-8,atol=2e-8)
    assert actual['body_state_ids']==baseline['body_state_ids']
    assert actual['goal_state_ids']==baseline['goal_state_ids']
    assert actual['goal_log_weights']==baseline['goal_log_weights']
    assert actual['source_command']==baseline['source_command']
    assert actual['source_stamp']==baseline['source_stamp']
    # The scalar diagnostic cannot express the small difference. The actual
    # derivative uses the retained residual of the same scalar criterion.
    values=[probe['objective'] for probe in actual['finite_difference_diagnostics']['evaluations']]
    assert len(set(values))==1
    assert values[0]==-common
    assert actual['objective_at_source_parts'][1]!=0.
    assert any(probe['objective_difference_from_source']!=0.
        for side in actual['finite_difference_diagnostics']['coordinates'] for probe in (side['forward'],side['backward']))
    # Legacy one-step comparisons/Armijo use the same differences too.
    controller.velocity[:]=0.
    command,legacy=controller.step(result,.5)
    np.testing.assert_allclose(command,baseline_command,rtol=0.,atol=2e-9)
    np.testing.assert_allclose(legacy['objective_gradient'],baseline_legacy['objective_gradient'],rtol=1e-8,atol=2e-8)
    for before,item in zip(covariances,result.stages[-1].components):
        np.testing.assert_array_equal(item.state.covariance,before)


def test_relative_density_retains_gaussian_normalizer_and_orientation_beside_huge_quadratic():
    _,result,controller=_problem()
    query=controller._query(result.stages[-1].components[0].state)
    covariance=np.array(query.covariance,copy=True)
    covariance[:3,:3]=np.eye(3)
    mean=np.array(query.mean,copy=True);mean[:3]=0.
    value=SimpleNamespace(**vars(query));value.covariance=covariance;value.mean=mean
    _,record=moving_control._relative_density(value,np.array([1e10,0.,0.]),
        controller.goal_orientation,controller.integration_steps)
    quadratic=-.5*1e20
    expected=-1.5*np.log(2.*np.pi)+record['log_orientation']
    actual=log_evidence_parts_difference(record['log_density_parts'],(quadratic,0.))
    assert actual==pytest.approx(expected,abs=2e-15)
    assert float(record['log_position']+record['log_orientation'])==quadratic
